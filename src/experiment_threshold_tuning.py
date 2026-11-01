
from __future__ import annotations

import os
import time
import warnings

# Suppress optional GitPython warnings inside Docker.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import mlflow
import numpy as np
import pandas as pd

from pathlib import Path
from sklearn.base import clone
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
)

from src.config import (
    FEATURES,
    TARGET,
    MLFLOW_TRACKING_URI,
)

from src.experiment_time_features import (
    build_pipeline,
    get_candidates,
    RANDOM_STATE,
)


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

DATA_PATH = Path(
    os.getenv(
        "THRESHOLD_DATA_PATH",
        "/app/data/processed/flights_mar_aug_experiment.parquet",
    )
)

EXPERIMENT_NAME = (
    "flight-delay-threshold-tuning-v1"
)

OUTPUT_DIR = DATA_PATH.parent

DETAIL_PATH = (
    OUTPUT_DIR / "threshold_tuning_detail_v1.csv"
)

CALIBRATION_PATH = (
    OUTPUT_DIR / "threshold_calibration_grid_v1.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR / "threshold_tuning_summary_v1.csv"
)

# Only two candidates are carried forward.
SELECTED_CANDIDATES = [
    "logistic_regression_cyclic",
    "extra_trees_cyclic",
]

# Thresholds to evaluate on the calibration period.
THRESHOLD_GRID = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
]

# Desired recall constraint during threshold selection.
TARGET_RECALL = 0.80

TRAIN_ROWS = int(
    os.getenv(
        "THRESHOLD_TRAIN_ROWS",
        "50000",
    )
)

CALIBRATION_ROWS = int(
    os.getenv(
        "THRESHOLD_CALIBRATION_ROWS",
        "30000",
    )
)

VALIDATION_ROWS = int(
    os.getenv(
        "THRESHOLD_VALIDATION_ROWS",
        "50000",
    )
)

# Each fold has:
# 1. Earlier months for model training.
# 2. The immediately preceding month for threshold selection.
# 3. The next month for evaluation.
FOLDS = [
    {
        "name": "fold_may_2026",
        "calibration_start": "2026-04-01",
        "calibration_end": "2026-05-01",
        "validation_start": "2026-05-01",
        "validation_end": "2026-06-01",
    },
    {
        "name": "fold_june_2026",
        "calibration_start": "2026-05-01",
        "calibration_end": "2026-06-01",
        "validation_start": "2026-06-01",
        "validation_end": "2026-07-01",
    },
    {
        "name": "fold_july_2026",
        "calibration_start": "2026-06-01",
        "calibration_end": "2026-07-01",
        "validation_start": "2026-07-01",
        "validation_end": "2026-08-01",
    },
]

TUNED_METRICS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
]

BASELINE_METRICS = [
    "accuracy_at_050",
    "precision_at_050",
    "recall_at_050",
    "f1_at_050",
]


# ============================================================
# 2. LOAD AND VALIDATE DATASET
# ============================================================

def load_dataset():
    """Load the date-aware BTS experiment dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}"
        )

    data = pd.read_parquet(DATA_PATH)

    required_columns = (
        list(FEATURES)
        + [TARGET, "FL_DATE"]
    )

    missing = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom dataset tidak tersedia: {missing}"
        )

    data["FL_DATE"] = pd.to_datetime(
        data["FL_DATE"],
        errors="coerce",
    )

    data[TARGET] = pd.to_numeric(
        data[TARGET],
        errors="coerce",
    )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    data = data.dropna(
        subset=required_columns
    ).copy()

    data[TARGET] = data[TARGET].astype(int)

    if not data[TARGET].isin([0, 1]).all():
        raise ValueError(
            "DELAY_15 hanya boleh bernilai 0 atau 1."
        )

    data = data.sort_values(
        "FL_DATE"
    ).reset_index(drop=True)

    return data


# ============================================================
# 3. STRATIFIED SAMPLING
# ============================================================

def stratified_month_sample(
    frame,
    max_rows,
    random_state,
    description,
):
    """
    Sample rows using MONTH + TARGET as strata.

    This helps preserve the monthly composition and
    positive/negative class proportions.
    """

    if frame.empty:
        raise ValueError(
            f"Data {description} kosong."
        )

    if frame[TARGET].nunique() != 2:
        raise ValueError(
            f"Data {description} harus memiliki dua kelas."
        )

    max_rows = int(max_rows)

    if max_rows <= 0 or max_rows >= len(frame):
        return (
            frame.sample(
                frac=1,
                random_state=random_state,
            )
            .reset_index(drop=True)
        )

    strata = (
        frame["MONTH"].astype(str)
        + "_"
        + frame[TARGET].astype(str)
    )

    group_counts = strata.value_counts()

    if group_counts.min() < 2:
        raise ValueError(
            f"Ada kelompok terlalu kecil pada {description}."
        )

    if max_rows < len(group_counts):
        raise ValueError(
            f"Jumlah sampel terlalu kecil pada {description}."
        )

    selected, _ = train_test_split(
        frame,
        train_size=max_rows,
        random_state=random_state,
        stratify=strata,
    )

    return selected.reset_index(drop=True)


# ============================================================
# 4. BUILD CHRONOLOGICAL PERIODS
# ============================================================

def build_periods(data, fold):
    """
    Create three strictly chronological datasets.

    Training:
        All dates before calibration_start.

    Calibration:
        calibration_start <= FL_DATE < calibration_end.

    Validation:
        validation_start <= FL_DATE < validation_end.
    """

    calibration_start = pd.Timestamp(
        fold["calibration_start"]
    )

    calibration_end = pd.Timestamp(
        fold["calibration_end"]
    )

    validation_start = pd.Timestamp(
        fold["validation_start"]
    )

    validation_end = pd.Timestamp(
        fold["validation_end"]
    )

    if not (
        calibration_start
        < calibration_end
        == validation_start
        < validation_end
    ):
        raise ValueError(
            f"Batas waktu tidak valid pada {fold['name']}."
        )

    train_full = data.loc[
        data["FL_DATE"] < calibration_start
    ].copy()

    calibration_full = data.loc[
        (data["FL_DATE"] >= calibration_start)
        & (data["FL_DATE"] < calibration_end)
    ].copy()

    validation_full = data.loc[
        (data["FL_DATE"] >= validation_start)
        & (data["FL_DATE"] < validation_end)
    ].copy()

    periods = {
        "training": train_full,
        "calibration": calibration_full,
        "validation": validation_full,
    }

    for name, frame in periods.items():

        if frame.empty:
            raise ValueError(
                f"Data {name} kosong pada {fold['name']}."
            )

        if frame[TARGET].nunique() != 2:
            raise ValueError(
                f"Data {name} harus mempunyai dua kelas "
                f"pada {fold['name']}."
            )

    # Ensure each period is temporally separated.
    if not (
        train_full["FL_DATE"].max()
        < calibration_full["FL_DATE"].min()
    ):
        raise RuntimeError(
            "Temporal leakage antara training dan calibration."
        )

    if not (
        calibration_full["FL_DATE"].max()
        < validation_full["FL_DATE"].min()
    ):
        raise RuntimeError(
            "Temporal leakage antara calibration dan validation."
        )

    return (
        train_full,
        calibration_full,
        validation_full,
    )


# ============================================================
# 5. PROBABILITY AND THRESHOLD METRICS
# ============================================================

def positive_probabilities(model, X):
    """Return predicted probability for positive class 1."""

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Model tidak mendukung predict_proba."
        )

    probabilities = model.predict_proba(X)

    classes = list(
        model.classes_
    )

    if 1 not in classes:
        raise ValueError(
            "Model tidak memiliki kelas positif 1."
        )

    positive_index = classes.index(1)

    return probabilities[:, positive_index]


def metrics_from_probabilities(
    y_true,
    probabilities,
    threshold,
):
    """
    Calculate threshold-dependent and threshold-independent
    evaluation metrics.
    """

    y_true = np.asarray(y_true).astype(int)

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    values = {
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
    }

    if np.unique(y_true).size < 2:
        raise ValueError(
            "Evaluasi membutuhkan dua kelas target."
        )

    # ROC-AUC, Average Precision and Brier score
    # do not depend on the classification threshold.
    values["roc_auc"] = float(
        roc_auc_score(
            y_true,
            probabilities,
        )
    )

    values["pr_auc"] = float(
        average_precision_score(
            y_true,
            probabilities,
        )
    )

    values["brier_score"] = float(
        brier_score_loss(
            y_true,
            probabilities,
        )
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    values["true_negative"] = int(tn)
    values["false_positive"] = int(fp)
    values["false_negative"] = int(fn)
    values["true_positive"] = int(tp)

    return values


# ============================================================
# 6. THRESHOLD SELECTION
# ============================================================

def select_threshold(
    y_calibration,
    calibration_probabilities,
    fold_name,
    candidate_name,
):
    """
    Select the threshold using calibration data only.

    Selection policy:
        1. Prefer thresholds with recall >= TARGET_RECALL.
        2. Within eligible thresholds, maximize F1.
        3. Break ties using recall, precision, and proximity to 0.50.
        4. If no threshold reaches target recall, choose
           the highest F1 among all tested thresholds.

    The validation month is never used to select the threshold.
    """

    calibration_results = []

    for threshold in THRESHOLD_GRID:

        metrics = metrics_from_probabilities(
            y_calibration,
            calibration_probabilities,
            threshold,
        )

        row = {
            "fold": fold_name,
            "candidate": candidate_name,
            "threshold": float(threshold),
            **metrics,
            "recall_target_met": (
                metrics["recall"] >= TARGET_RECALL
            ),
        }

        calibration_results.append(row)

    eligible = [
        row
        for row in calibration_results
        if row["recall_target_met"]
    ]

    if eligible:
        selection_pool = eligible
        recall_constraint_met = True
    else:
        selection_pool = calibration_results
        recall_constraint_met = False

    selected = max(
        selection_pool,
        key=lambda row: (
            row["f1"],
            row["recall"],
            row["precision"],
            -abs(row["threshold"] - 0.50),
        ),
    )

    return (
        selected,
        calibration_results,
        recall_constraint_met,
    )


# ============================================================
# 7. MODEL CANDIDATE SELECTION
# ============================================================

def selected_candidates():
    """Get only the two previously shortlisted cyclic models."""

    available = get_candidates()

    selected = [
        (name, estimator, use_cyclic_time)
        for name, estimator, use_cyclic_time in available
        if name in SELECTED_CANDIDATES
    ]

    if len(selected) != len(SELECTED_CANDIDATES):
        raise RuntimeError(
            "Satu atau lebih kandidat tidak ditemukan "
            "di experiment_time_features.py."
        )

    return selected


# ============================================================
# 8. RUN ONE FOLD / CANDIDATE
# ============================================================

def run_candidate(
    fold,
    candidate_name,
    estimator,
    use_cyclic_time,
    train_sample,
    calibration_sample,
    validation_sample,
):
    """
    Train on earlier data, select threshold on the previous
    month, and evaluate on the following month.

    The same fitted model is used for calibration and validation.
    It is NOT refitted after threshold selection.
    """

    fold_name = fold["name"]

    X_train = train_sample[
        list(FEATURES)
    ].copy()

    y_train = train_sample[
        TARGET
    ].astype(int).copy()

    X_calibration = calibration_sample[
        list(FEATURES)
    ].copy()

    y_calibration = calibration_sample[
        TARGET
    ].astype(int).copy()

    X_validation = validation_sample[
        list(FEATURES)
    ].copy()

    y_validation = validation_sample[
        TARGET
    ].astype(int).copy()

    # FL_DATE is metadata only, not a prediction feature.
    for name, X in [
        ("training", X_train),
        ("calibration", X_calibration),
        ("validation", X_validation),
    ]:
        if "FL_DATE" in X.columns:
            raise RuntimeError(
                f"FL_DATE masuk ke fitur {name}."
            )

    model = build_pipeline(
        clone(estimator),
        use_cyclic_time=use_cyclic_time,
    )

    run_name = (
        f"{fold_name}__{candidate_name}"
    )

    print("\n" + "=" * 75)
    print(
        f"FOLD: {fold_name} | CANDIDATE: {candidate_name}"
    )
    print("=" * 75)

    with mlflow.start_run(
        run_name=run_name
    ) as run:

        mlflow.log_params({
            "fold": fold_name,
            "candidate": candidate_name,
            "dataset": DATA_PATH.name,
            "time_encoding": "cyclical_sin_cos",
            "threshold_policy": (
                "maximize_f1_subject_to_recall_target"
            ),
            "target_recall": TARGET_RECALL,
            "threshold_grid": ",".join(
                str(value)
                for value in THRESHOLD_GRID
            ),
            "threshold_selection_period": str(
                calibration_sample["FL_DATE"].min().date()
            ),
            "validation_period": str(
                validation_sample["FL_DATE"].min().date()
            ),
            "training_rows": len(train_sample),
            "calibration_rows": len(calibration_sample),
            "validation_rows": len(validation_sample),
            "model_artifact_saved": False,
            "registered_model": False,
            "champion_promoted": False,
        })

        mlflow.log_metrics({
            "training_delay_rate": float(
                y_train.mean()
            ),
            "calibration_delay_rate": float(
                y_calibration.mean()
            ),
            "validation_delay_rate": float(
                y_validation.mean()
            ),
        })

        # ----------------------------------------------------
        # Train on the earlier period
        # ----------------------------------------------------

        print(
            "Training model...",
            flush=True,
        )

        training_start = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        training_seconds = (
            time.perf_counter() - training_start
        )

        # ----------------------------------------------------
        # Select threshold on the previous month
        # ----------------------------------------------------

        print(
            "Memilih threshold dari calibration period...",
            flush=True,
        )

        calibration_probabilities = positive_probabilities(
            model,
            X_calibration,
        )

        (
            selected_threshold,
            calibration_grid,
            recall_constraint_met,
        ) = select_threshold(
            y_calibration,
            calibration_probabilities,
            fold_name,
            candidate_name,
        )

        selected_value = float(
            selected_threshold["threshold"]
        )

        # ----------------------------------------------------
        # Evaluate on the next month
        # ----------------------------------------------------

        print(
            "Mengevaluasi pada validation period...",
            flush=True,
        )

        validation_probabilities = positive_probabilities(
            model,
            X_validation,
        )

        # Tuned threshold.
        tuned_metrics = metrics_from_probabilities(
            y_validation,
            validation_probabilities,
            selected_value,
        )

        # Reference threshold 0.50, on the same validation data.
        baseline_metrics = metrics_from_probabilities(
            y_validation,
            validation_probabilities,
            0.50,
        )

        # ----------------------------------------------------
        # Log results to MLflow
        # ----------------------------------------------------

        mlflow.log_params({
            "selected_threshold": selected_value,
            "calibration_recall_target_met": str(
                recall_constraint_met
            ),
            "threshold_selected_on_validation_data": False,
        })

        mlflow.log_metrics({
            f"tuned_{key}": float(value)
            for key, value in tuned_metrics.items()
        })

        mlflow.log_metrics({
            f"threshold_050_{key}": float(
                baseline_metrics[key]
            )
            for key in BASELINE_METRICS
            if key in baseline_metrics
        } if False else {
            "threshold_050_accuracy": baseline_metrics["accuracy"],
            "threshold_050_precision": baseline_metrics["precision"],
            "threshold_050_recall": baseline_metrics["recall"],
            "threshold_050_f1": baseline_metrics["f1"],
        })

        mlflow.log_metrics({
            "calibration_selected_f1": float(
                selected_threshold["f1"]
            ),
            "calibration_selected_recall": float(
                selected_threshold["recall"]
            ),
            "calibration_selected_precision": float(
                selected_threshold["precision"]
            ),
            "training_seconds": float(
                training_seconds
            ),
        })

        # ----------------------------------------------------
        # Display fold results
        # ----------------------------------------------------

        print("\nHASIL CALIBRATION")
        print(
            "Selected threshold:",
            f"{selected_value:.2f}",
        )

        print(
            "Calibration F1:",
            f"{selected_threshold['f1']:.4f}",
        )

        print(
            "Calibration Recall:",
            f"{selected_threshold['recall']:.4f}",
        )

        print(
            "Recall target met:",
            recall_constraint_met,
        )

        print("\nHASIL VALIDASI - THRESHOLD TUNED")

        for key, value in tuned_metrics.items():
            print(
                f"{key}: {value:.4f}",
                flush=True,
            )

        print("\nHASIL VALIDASI - THRESHOLD 0.50")

        for key in BASELINE_METRICS:
            original_key = key.replace(
                "_at_050",
                "",
            )

            print(
                f"{key}: "
                f"{baseline_metrics[original_key]:.4f}",
                flush=True,
            )

        # ----------------------------------------------------
        # Build result record
        # ----------------------------------------------------

        result = {
            "fold": fold_name,
            "candidate": candidate_name,
            "selected_threshold": selected_value,
            "calibration_f1": selected_threshold["f1"],
            "calibration_recall": selected_threshold["recall"],
            "calibration_precision": selected_threshold["precision"],
            "calibration_recall_target_met": int(
                recall_constraint_met
            ),
            "training_seconds": training_seconds,
            "run_id": run.info.run_id,
        }

        result.update(tuned_metrics)

        for key in BASELINE_METRICS:
            original_key = key.replace(
                "_at_050",
                "",
            )

            result[key] = baseline_metrics[original_key]

        calibration_rows = calibration_grid

    del model

    return result, calibration_rows


# ============================================================
# 9. AGGREGATE RESULTS
# ============================================================

def build_summary(detail):
    """
    Aggregate results across the three validation periods.

    Mean and standard deviation are reported.
    """

    grouped = detail.groupby(
        "candidate"
    ).agg(
        threshold_mean=(
            "selected_threshold",
            "mean",
        ),
        threshold_std=(
            "selected_threshold",
            "std",
        ),
        recall_constraint_rate=(
            "calibration_recall_target_met",
            "mean",
        ),
        accuracy_mean=(
            "accuracy",
            "mean",
        ),
        accuracy_std=(
            "accuracy",
            "std",
        ),
        precision_mean=(
            "precision",
            "mean",
        ),
        recall_mean=(
            "recall",
            "mean",
        ),
        f1_mean=(
            "f1",
            "mean",
        ),
        f1_std=(
            "f1",
            "std",
        ),
        roc_auc_mean=(
            "roc_auc",
            "mean",
        ),
        roc_auc_std=(
            "roc_auc",
            "std",
        ),
        pr_auc_mean=(
            "pr_auc",
            "mean",
        ),
        brier_score_mean=(
            "brier_score",
            "mean",
        ),
        f1_at_050_mean=(
            "f1_at_050",
            "mean",
        ),
        recall_at_050_mean=(
            "recall_at_050",
            "mean",
        ),
    ).reset_index()

    # Sort primarily by F1 and then ROC-AUC.
    summary = grouped.sort_values(
        by=[
            "f1_mean",
            "roc_auc_mean",
        ],
        ascending=False,
    ).reset_index(drop=True)

    return summary


# ============================================================
# 10. MAIN EXPERIMENT
# ============================================================

def main():

    print("\n" + "=" * 75)
    print("FLIGHT DELAY - NESTED CHRONOLOGICAL THRESHOLD TUNING")
    print("=" * 75)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    data = load_dataset()

    print("\nDATASET")
    print("-" * 75)

    print(
        "File:",
        DATA_PATH,
    )

    print(
        "Rows:",
        f"{len(data):,}",
    )

    print(
        "Date range:",
        data["FL_DATE"].min().date(),
        "to",
        data["FL_DATE"].max().date(),
    )

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print("\nMLFLOW")
    print("-" * 75)

    print(
        "Experiment:",
        EXPERIMENT_NAME,
    )

    print(
        "Tracking URI:",
        MLFLOW_TRACKING_URI,
    )

    print(
        "Candidates:",
        SELECTED_CANDIDATES,
    )

    print(
        "Threshold grid:",
        THRESHOLD_GRID,
    )

    print(
        "Recall target:",
        TARGET_RECALL,
    )

    print(
        "Training sample:",
        TRAIN_ROWS,
    )

    print(
        "Calibration sample:",
        CALIBRATION_ROWS,
    )

    print(
        "Validation sample:",
        VALIDATION_ROWS,
    )

    # --------------------------------------------------------
    # Get candidate estimators
    # --------------------------------------------------------

    candidates = selected_candidates()

    all_results = []
    all_calibration_results = []

    # --------------------------------------------------------
    # Walk-forward folds
    # --------------------------------------------------------

    for fold_index, fold in enumerate(FOLDS):

        print("\n" + "#" * 75)
        print(
            "STARTING",
            fold["name"].upper(),
        )
        print("#" * 75)

        (
            train_full,
            calibration_full,
            validation_full,
        ) = build_periods(
            data,
            fold,
        )

        print("\nCHRONOLOGICAL PERIODS")
        print("-" * 75)

        for name, frame in [
            ("TRAINING", train_full),
            ("CALIBRATION", calibration_full),
            ("VALIDATION", validation_full),
        ]:
            print(
                f"{name}: "
                f"{frame['FL_DATE'].min().date()} "
                f"to {frame['FL_DATE'].max().date()} "
                f"| rows={len(frame):,} "
                f"| delay={frame[TARGET].mean() * 100:.2f}%"
            )

        # ----------------------------------------------------
        # Sample each period once per fold.
        # Both model candidates receive identical data.
        # ----------------------------------------------------

        train_sample = stratified_month_sample(
            train_full,
            TRAIN_ROWS,
            RANDOM_STATE + fold_index,
            f"{fold['name']}_training",
        )

        calibration_sample = stratified_month_sample(
            calibration_full,
            CALIBRATION_ROWS,
            RANDOM_STATE + fold_index + 10,
            f"{fold['name']}_calibration",
        )

        validation_sample = stratified_month_sample(
            validation_full,
            VALIDATION_ROWS,
            RANDOM_STATE + fold_index + 20,
            f"{fold['name']}_validation",
        )

        print("\nSAMPLING")
        print("-" * 75)

        print(
            "Training:",
            f"{len(train_sample):,}",
        )

        print(
            "Calibration:",
            f"{len(calibration_sample):,}",
        )

        print(
            "Validation:",
            f"{len(validation_sample):,}",
        )

        # ----------------------------------------------------
        # Run candidate models
        # ----------------------------------------------------

        for (
            candidate_name,
            estimator,
            use_cyclic_time,
        ) in candidates:

            if not use_cyclic_time:
                raise RuntimeError(
                    f"{candidate_name} bukan kandidat cyclic."
                )

            result, calibration_rows = run_candidate(
                fold=fold,
                candidate_name=candidate_name,
                estimator=estimator,
                use_cyclic_time=use_cyclic_time,
                train_sample=train_sample,
                calibration_sample=calibration_sample,
                validation_sample=validation_sample,
            )

            all_results.append(
                result
            )

            all_calibration_results.extend(
                calibration_rows
            )

        del train_sample
        del calibration_sample
        del validation_sample

    # --------------------------------------------------------
    # Prepare result dataframes
    # --------------------------------------------------------

    detail = pd.DataFrame(
        all_results
    )

    calibration_detail = pd.DataFrame(
        all_calibration_results
    )

    detail = detail.sort_values(
        ["fold", "candidate"]
    ).reset_index(drop=True)

    calibration_detail = calibration_detail.sort_values(
        ["fold", "candidate", "threshold"]
    ).reset_index(drop=True)

    summary = build_summary(
        detail
    )

    # --------------------------------------------------------
    # Save CSV files
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    detail.to_csv(
        DETAIL_PATH,
        index=False,
    )

    calibration_detail.to_csv(
        CALIBRATION_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # --------------------------------------------------------
    # Print detailed validation results
    # --------------------------------------------------------

    detail_columns = [
        "fold",
        "candidate",
        "selected_threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "f1_at_050",
        "recall_at_050",
        "calibration_recall_target_met",
    ]

    print("\n" + "=" * 115)
    print("DETAIL HASIL THRESHOLD TUNING")
    print("=" * 115)

    print(
        detail[detail_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------------
    # Print threshold calibration grid
    # --------------------------------------------------------

    calibration_columns = [
        "fold",
        "candidate",
        "threshold",
        "precision",
        "recall",
        "f1",
        "recall_target_met",
    ]

    print("\n" + "=" * 100)
    print("DETAIL THRESHOLD PADA CALIBRATION DATA")
    print("=" * 100)

    print(
        calibration_detail[calibration_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------------
    # Print aggregate results
    # --------------------------------------------------------

    summary_columns = [
        "candidate",
        "threshold_mean",
        "recall_constraint_rate",
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "f1_std",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "brier_score_mean",
        "f1_at_050_mean",
        "recall_at_050_mean",
    ]

    print("\n" + "=" * 120)
    print("RINGKASAN THRESHOLD TUNING")
    print("Mean dari tiga periode validasi")
    print("=" * 120)

    print(
        summary[summary_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("THRESHOLD TUNING SELESAI")
    print("=" * 75)

    print(
        "Total completed model runs:",
        len(detail),
    )

    print(
        "Detail hasil:",
        DETAIL_PATH,
    )

    print(
        "Threshold calibration grid:",
        CALIBRATION_PATH,
    )

    print(
        "Summary:",
        SUMMARY_PATH,
    )

    print(
        "MLflow experiment:",
        EXPERIMENT_NAME,
    )

    print(
        "Model artifacts saved: NO"
    )

    print(
        "Model registry modified: NO"
    )

    print(
        "Champion alias modified: NO"
    )


# ============================================================
# 11. ENTRY POINT
# ============================================================

if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    main()
