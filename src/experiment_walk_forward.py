
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
from sklearn.model_selection import train_test_split

from src.config import FEATURES, TARGET, MLFLOW_TRACKING_URI

from src.experiment_time_features import (
    build_pipeline,
    get_candidates,
    metrics_for,
    RANDOM_STATE,
)


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

DATA_PATH = Path(
    os.getenv(
        "WALK_FORWARD_DATA_PATH",
        "/app/data/processed/flights_mar_aug_experiment.parquet",
    )
)

EXPERIMENT_NAME = "flight-delay-walk-forward-v1"

OUTPUT_DIR = Path(
    os.getenv(
        "WALK_FORWARD_OUTPUT_DIR",
        "/app/data/processed",
    )
)

DETAIL_OUTPUT = (
    OUTPUT_DIR / "walk_forward_detail_v1.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR / "walk_forward_summary_v1.csv"
)

TRAIN_ROWS = int(
    os.getenv(
        "WALK_FORWARD_TRAIN_ROWS",
        "50000",
    )
)

VALIDATION_ROWS = int(
    os.getenv(
        "WALK_FORWARD_VALIDATION_ROWS",
        "50000",
    )
)

# Each validation period is one month.
# Training always uses all available months before validation.
FOLDS = [
    {
        "name": "fold_may_2026",
        "validation_start": "2026-05-01",
        "validation_end": "2026-06-01",
    },
    {
        "name": "fold_june_2026",
        "validation_start": "2026-06-01",
        "validation_end": "2026-07-01",
    },
    {
        "name": "fold_july_2026",
        "validation_start": "2026-07-01",
        "validation_end": "2026-08-01",
    },
]

METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
]


# ============================================================
# 2. LOAD AND VALIDATE DATASET
# ============================================================

def load_dataset():
    """Load and validate the six-month chronological dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATA_PATH}"
        )

    data = pd.read_parquet(DATA_PATH)

    required = list(FEATURES) + [
        TARGET,
        "FL_DATE",
    ]

    missing = [
        column
        for column in required
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
        subset=required
    ).copy()

    data[TARGET] = data[TARGET].astype(int)

    if not data[TARGET].isin([0, 1]).all():
        raise ValueError(
            "DELAY_15 hanya boleh bernilai 0 atau 1."
        )

    if data[TARGET].nunique() != 2:
        raise ValueError(
            "Dataset harus mempunyai dua kelas target."
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
    Stratified sample using MONTH + TARGET groups.

    This preserves the monthly composition and target
    distribution more closely than target-only sampling.
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
        return frame.sample(
            frac=1,
            random_state=random_state,
        ).reset_index(drop=True)

    strata = (
        frame["MONTH"].astype(str)
        + "_"
        + frame[TARGET].astype(str)
    )

    group_counts = strata.value_counts()

    if group_counts.min() < 2:
        raise ValueError(
            f"Ada kelompok yang terlalu kecil untuk sampling {description}."
        )

    selected, _ = train_test_split(
        frame,
        train_size=max_rows,
        random_state=random_state,
        stratify=strata,
    )

    return selected.reset_index(drop=True)


# ============================================================
# 4. BUILD CHRONOLOGICAL FOLD
# ============================================================

def build_fold(data, fold):
    """
    Build one walk-forward fold.

    Training:
        FL_DATE < validation_start

    Validation:
        validation_start <= FL_DATE < validation_end
    """

    validation_start = pd.Timestamp(
        fold["validation_start"]
    )

    validation_end = pd.Timestamp(
        fold["validation_end"]
    )

    if validation_end <= validation_start:
        raise ValueError(
            f"Batas tanggal tidak valid: {fold['name']}"
        )

    train_full = data.loc[
        data["FL_DATE"] < validation_start
    ].copy()

    validation_full = data.loc[
        (data["FL_DATE"] >= validation_start)
        & (data["FL_DATE"] < validation_end)
    ].copy()

    if train_full.empty or validation_full.empty:
        raise ValueError(
            f"Training atau validation kosong pada {fold['name']}."
        )

    if train_full[TARGET].nunique() != 2:
        raise ValueError(
            f"Training pada {fold['name']} hanya memiliki satu kelas."
        )

    if validation_full[TARGET].nunique() != 2:
        raise ValueError(
            f"Validation pada {fold['name']} hanya memiliki satu kelas."
        )

    if (
        train_full["FL_DATE"].max()
        >= validation_full["FL_DATE"].min()
    ):
        raise RuntimeError(
            f"Temporal leakage terdeteksi pada {fold['name']}."
        )

    return train_full, validation_full


# ============================================================
# 5. TRAIN ONE CANDIDATE
# ============================================================

def run_candidate(
    fold_name,
    candidate_name,
    estimator,
    use_cyclic_time,
    X_train,
    y_train,
    X_validation,
    y_validation,
    train_full,
    validation_full,
):
    """
    Train and evaluate one model candidate.

    Logs metrics and parameters only.

    No model artifact is saved.
    No model is registered.
    No champion alias is changed.
    """

    print(
        "\n"
        + "=" * 70,
        flush=True,
    )

    print(
        f"FOLD: {fold_name} | MODEL: {candidate_name}",
        flush=True,
    )

    print(
        "=" * 70,
        flush=True,
    )

    model = build_pipeline(
        estimator,
        use_cyclic_time=use_cyclic_time,
    )

    run_name = (
        f"{fold_name}__{candidate_name}"
    )

    with mlflow.start_run(
        run_name=run_name
    ) as run:

        mlflow.log_params({
            "fold": fold_name,
            "candidate": candidate_name,
            "dataset": DATA_PATH.name,
            "split_strategy": "walk_forward_chronological",
            "train_start": str(
                train_full["FL_DATE"].min().date()
            ),
            "train_end": str(
                train_full["FL_DATE"].max().date()
            ),
            "validation_start": str(
                validation_full["FL_DATE"].min().date()
            ),
            "validation_end": str(
                validation_full["FL_DATE"].max().date()
            ),
            "train_available_rows": int(
                len(train_full)
            ),
            "validation_available_rows": int(
                len(validation_full)
            ),
            "train_sampled_rows": int(
                len(X_train)
            ),
            "validation_sampled_rows": int(
                len(X_validation)
            ),
            "time_encoding": (
                "cyclical_sin_cos"
                if use_cyclic_time
                else "raw_hhmm"
            ),
            "decision_threshold": 0.5,
            "model_artifact_saved": False,
            "registered_model": False,
            "champion_promoted": False,
            "random_state": RANDOM_STATE,
        })

        mlflow.log_metric(
            "train_delay_rate",
            float(y_train.mean()),
        )

        mlflow.log_metric(
            "validation_delay_rate",
            float(y_validation.mean()),
        )

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        start_time = time.perf_counter()

        model.fit(
            X_train,
            y_train,
        )

        training_seconds = (
            time.perf_counter() - start_time
        )

        # ----------------------------------------------------
        # Evaluation
        # ----------------------------------------------------

        evaluation_start = time.perf_counter()

        metrics = metrics_for(
            model,
            X_validation,
            y_validation,
        )

        evaluation_seconds = (
            time.perf_counter() - evaluation_start
        )

        metrics["training_seconds"] = float(
            training_seconds
        )

        metrics["evaluation_seconds"] = float(
            evaluation_seconds
        )

        # Log all metrics in MLflow.
        mlflow.log_metrics(
            {
                key: float(value)
                for key, value in metrics.items()
            }
        )

        mlflow.log_param(
            "model_artifact_saved",
            False,
        )

        print(
            "\nHASIL VALIDASI",
            flush=True,
        )

        for key, value in metrics.items():
            print(
                f"{key}: {value:.4f}",
                flush=True,
            )

        result = {
            "fold": fold_name,
            "candidate": candidate_name,
            **metrics,
            "run_id": run.info.run_id,
            "train_rows_available": len(train_full),
            "validation_rows_available": len(validation_full),
            "train_rows_sampled": len(X_train),
            "validation_rows_sampled": len(X_validation),
            "train_delay_rate": float(y_train.mean()),
            "validation_delay_rate": float(
                y_validation.mean()
            ),
        }

    del model

    return result


# ============================================================
# 6. AGGREGATE WALK-FORWARD RESULTS
# ============================================================

def aggregate_results(detail):
    """
    Calculate mean and standard deviation of metrics
    across the three chronological validation periods.
    """

    grouped = detail.groupby(
        "candidate"
    )[METRIC_COLUMNS].agg(
        ["mean", "std"]
    )

    grouped.columns = [
        f"{metric}_{statistic}"
        for metric, statistic in grouped.columns
    ]

    summary = grouped.reset_index()

    summary = summary.sort_values(
        by=[
            "roc_auc_mean",
            "f1_mean",
        ],
        ascending=False,
    ).reset_index(drop=True)

    return summary


# ============================================================
# 7. MAIN WALK-FORWARD EXPERIMENT
# ============================================================

def main():

    print("\n" + "=" * 75)
    print("FLIGHT DELAY - WALK-FORWARD VALIDATION")
    print("=" * 75)

    # --------------------------------------------------------
    # Load data
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

    print(
        "Overall delay rate:",
        f"{data[TARGET].mean() * 100:.2f}%",
    )

    # --------------------------------------------------------
    # Monthly overview
    # --------------------------------------------------------

    print("\nMONTHLY DISTRIBUTION")
    print("-" * 75)

    monthly = data.groupby(
        "MONTH"
    ).agg(
        rows=(TARGET, "size"),
        delay_rate=(
            TARGET,
            lambda values: values.mean() * 100,
        ),
    )

    print(
        monthly.to_string(
            float_format=lambda value: f"{value:.2f}"
        )
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
        "Training sample limit:",
        TRAIN_ROWS,
    )

    print(
        "Validation sample limit:",
        VALIDATION_ROWS,
    )

    print(
        "Candidates per fold:",
        5,
    )

    print(
        "Total model runs:",
        len(FOLDS) * 5,
    )

    # --------------------------------------------------------
    # Run all chronological folds
    # --------------------------------------------------------

    all_results = []

    for fold_index, fold in enumerate(FOLDS):

        fold_name = fold["name"]

        print("\n" + "#" * 75)
        print(
            f"STARTING {fold_name.upper()}",
            flush=True,
        )
        print("#" * 75)

        train_full, validation_full = build_fold(
            data,
            fold,
        )

        print("\nCHRONOLOGICAL SPLIT")
        print("-" * 75)

        print(
            "Training:",
            train_full["FL_DATE"].min().date(),
            "to",
            train_full["FL_DATE"].max().date(),
        )

        print(
            "Validation:",
            validation_full["FL_DATE"].min().date(),
            "to",
            validation_full["FL_DATE"].max().date(),
        )

        print(
            "Training rows:",
            f"{len(train_full):,}",
        )

        print(
            "Validation rows:",
            f"{len(validation_full):,}",
        )

        print(
            "Training delay rate:",
            f"{train_full[TARGET].mean() * 100:.2f}%",
        )

        print(
            "Validation delay rate:",
            f"{validation_full[TARGET].mean() * 100:.2f}%",
        )

        # ----------------------------------------------------
        # Sampling is performed once per fold.
        # All models use identical rows for fair comparison.
        # ----------------------------------------------------

        train_sample = stratified_month_sample(
            train_full,
            TRAIN_ROWS,
            RANDOM_STATE + fold_index,
            f"{fold_name}_training",
        )

        validation_sample = stratified_month_sample(
            validation_full,
            VALIDATION_ROWS,
            RANDOM_STATE + fold_index + 10,
            f"{fold_name}_validation",
        )

        print("\nSAMPLING")
        print("-" * 75)

        print(
            "Training sample:",
            f"{len(train_sample):,}",
        )

        print(
            "Validation sample:",
            f"{len(validation_sample):,}",
        )

        print(
            "Sample training delay rate:",
            f"{train_sample[TARGET].mean() * 100:.2f}%",
        )

        print(
            "Sample validation delay rate:",
            f"{validation_sample[TARGET].mean() * 100:.2f}%",
        )

        # FL_DATE is only for chronological splitting.
        # It must never enter model features.
        X_train = train_sample[
            list(FEATURES)
        ].copy()

        y_train = train_sample[
            TARGET
        ].astype(int).copy()

        X_validation = validation_sample[
            list(FEATURES)
        ].copy()

        y_validation = validation_sample[
            TARGET
        ].astype(int).copy()

        if "FL_DATE" in X_train.columns:
            raise RuntimeError(
                "FL_DATE masuk ke fitur training."
            )

        if "FL_DATE" in X_validation.columns:
            raise RuntimeError(
                "FL_DATE masuk ke fitur validation."
            )

        # ----------------------------------------------------
        # Run five candidates
        # ----------------------------------------------------

        candidates = get_candidates()

        for (
            candidate_name,
            estimator,
            use_cyclic_time,
        ) in candidates:

            result = run_candidate(
                fold_name=fold_name,
                candidate_name=candidate_name,
                estimator=estimator,
                use_cyclic_time=use_cyclic_time,
                X_train=X_train,
                y_train=y_train,
                X_validation=X_validation,
                y_validation=y_validation,
                train_full=train_full,
                validation_full=validation_full,
            )

            all_results.append(result)

            print(
                f"\nSelesai: {fold_name} / {candidate_name}",
                flush=True,
            )

        del train_sample
        del validation_sample
        del X_train
        del X_validation
        del y_train
        del y_validation

    # --------------------------------------------------------
    # Detail results
    # --------------------------------------------------------

    detail = pd.DataFrame(
        all_results
    )

    detail = detail.sort_values(
        ["fold", "roc_auc"],
        ascending=[True, False],
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # Aggregate results
    # --------------------------------------------------------

    summary = aggregate_results(
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
        DETAIL_OUTPUT,
        index=False,
    )

    summary.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # Print fold-by-fold table
    # --------------------------------------------------------

    detail_columns = [
        "fold",
        "candidate",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]

    print("\n" + "=" * 100)
    print("DETAIL HASIL PER FOLD")
    print("=" * 100)

    print(
        detail[detail_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------------
    # Print aggregate table
    # --------------------------------------------------------

    print("\n" + "=" * 110)
    print("RINGKASAN WALK-FORWARD VALIDATION")
    print("Mean dan standard deviation dari tiga periode validasi")
    print("=" * 110)

    summary_columns = [
        "candidate",
        "accuracy_mean",
        "precision_mean",
        "recall_mean",
        "f1_mean",
        "roc_auc_mean",
        "roc_auc_std",
        "pr_auc_mean",
        "brier_score_mean",
    ]

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
    print("WALK-FORWARD VALIDATION SELESAI")
    print("=" * 75)

    print(
        "Total completed model runs:",
        len(all_results),
    )

    print(
        "Detail CSV:",
        DETAIL_OUTPUT,
    )

    print(
        "Summary CSV:",
        SUMMARY_OUTPUT,
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
# 8. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    main()
