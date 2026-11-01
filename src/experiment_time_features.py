
from __future__ import annotations

import os

# Suppress optional GitPython warning inside the Docker container.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    StandardScaler,
)

from src.config import (
    FLIGHTS_PATH,
    FEATURES,
    TARGET,
    MLFLOW_TRACKING_URI,
)


# ============================================================
# 1. EXPERIMENT CONFIGURATION
# ============================================================

EXPERIMENT_DATA_PATH = Path(
    os.getenv(
        "EXPERIMENT_DATA_PATH",
        str(
            FLIGHTS_PATH.with_name(
                "flights_with_date.parquet"
            )
        ),
    )
)

EXPERIMENT_NAME = (
    "flight-delay-time-encoding-chronological-v2"
)

SUMMARY_PATH = Path(
    os.getenv(
        "EXPERIMENT_SUMMARY_PATH",
        "/app/data/processed/"
        "time_features_experiment_v2_summary.csv",
    )
)

RANDOM_STATE = 42

CATEGORICAL = [
    "OP_CARRIER",
    "ORIGIN",
    "DEST",
]

RAW_NUMERIC = [
    "CRS_DEP_TIME",
    "MONTH",
    "DAY_OF_WEEK",
    "DISTANCE",
]

CYCLIC_NUMERIC = [
    "MONTH",
    "DAY_OF_WEEK",
    "DISTANCE",
]

# Default experiment period:
# Training: June - July 2026
# Testing : August 2026
DEFAULT_TEST_START = "2026-08-01"
DEFAULT_TEST_END = "2026-09-01"

# Initial sample size.
# The full August dataset will be used in a later validation stage.
DEFAULT_TRAIN_ROWS = 50000
DEFAULT_TEST_ROWS = 50000


# ============================================================
# 2. CYCLICAL TIME ENCODING
# ============================================================

def hhmm_to_cyclic(values):
    """
    Convert BTS HHMM scheduled departure time into
    two cyclical features.

    Example:
        1649 -> 16:49

    Output:
        DEP_TIME_SIN
        DEP_TIME_COS

    Formula:
        angle = 2*pi*minutes_of_day/1440

        SIN = sin(angle)
        COS = cos(angle)

    This representation makes midnight continuous:
        23:59 and 00:01 are close to each other.
    """

    values = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    hours = np.floor(
        values / 100
    )

    minutes = values % 100

    valid = (
        np.isfinite(values)
        & (values >= 0)
        & (values <= 2359)
        & (minutes >= 0)
        & (minutes < 60)
        & (values == np.floor(values))
    )

    minutes_of_day = (
        hours * 60 + minutes
    )

    angle = (
        2 * np.pi * minutes_of_day / 1440
    )

    sin_time = np.sin(angle)
    cos_time = np.cos(angle)

    encoded = np.column_stack([
        sin_time,
        cos_time,
    ])

    encoded[~valid] = np.nan

    return encoded


# ============================================================
# 3. MODEL PIPELINE
# ============================================================

def build_pipeline(
    estimator,
    use_cyclic_time=False,
):
    """
    Build preprocessing and classification pipeline.

    use_cyclic_time=False:
        CRS_DEP_TIME is treated as the original HHMM feature.

    use_cyclic_time=True:
        CRS_DEP_TIME is converted into SIN and COS features.

    FL_DATE is deliberately not included as a model feature.
    """

    transformers = []

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            ),
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=5,
            ),
        ),
    ])

    transformers.append(
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL,
        )
    )

    # --------------------------------------------------------
    # Departure time
    # --------------------------------------------------------

    if use_cyclic_time:

        departure_time_pipeline = Pipeline([
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "cyclic",
                FunctionTransformer(
                    hhmm_to_cyclic,
                    validate=False,
                ),
            ),
        ])

        transformers.append(
            (
                "departure_time_cyclic",
                departure_time_pipeline,
                ["CRS_DEP_TIME"],
            )
        )

        numeric_columns = CYCLIC_NUMERIC

    else:
        numeric_columns = RAW_NUMERIC

    # --------------------------------------------------------
    # Numeric features
    # --------------------------------------------------------

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            ),
        ),
        (
            "scale",
            StandardScaler(
                with_mean=False
            ),
        ),
    ])

    transformers.append(
        (
            "numeric",
            numeric_pipeline,
            numeric_columns,
        )
    )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return Pipeline([
        (
            "preprocess",
            preprocessor,
        ),
        (
            "model",
            estimator,
        ),
    ])


# ============================================================
# 4. STRATIFIED SAMPLING
# ============================================================

def sample_rows(
    frame,
    max_rows,
    random_state,
    dataset_name,
):
    """
    Take reproducible stratified samples.

    max_rows <= 0:
        Use all available rows.

    max_rows >= available rows:
        Use all available rows in randomized order.

    Otherwise:
        Use stratified sampling.
    """

    max_rows = int(max_rows)

    if frame.empty:
        raise ValueError(
            f"Dataset {dataset_name} kosong."
        )

    if frame[TARGET].nunique() < 2:
        raise ValueError(
            f"Dataset {dataset_name} harus memiliki "
            "dua kelas target."
        )

    if max_rows <= 0 or max_rows >= len(frame):

        return (
            frame.sample(
                frac=1,
                random_state=random_state,
            )
            .reset_index(drop=True)
        )

    if max_rows < frame[TARGET].nunique():
        raise ValueError(
            f"Jumlah sampel {dataset_name} terlalu kecil."
        )

    selected, _ = train_test_split(
        frame,
        train_size=max_rows,
        random_state=random_state,
        stratify=frame[TARGET],
    )

    return selected.reset_index(drop=True)


# ============================================================
# 5. EVALUATION METRICS
# ============================================================

def metrics_for(
    model,
    X_test,
    y_test,
):
    """
    Calculate classification metrics.

    Classification threshold:
        0.50

    PR-AUC metric:
        Average Precision score.
    """

    predictions = model.predict(
        X_test
    )

    values = {
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),

        "precision": float(
            precision_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                y_test,
                predictions,
                zero_division=0,
            )
        ),
    }

    # Probability-based evaluation.
    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "Model tidak mendukung predict_proba."
        )

    probabilities = model.predict_proba(
        X_test
    )

    classes = list(
        model.classes_
    )

    if 1 not in classes:
        raise ValueError(
            "Model tidak memiliki kelas positif 1."
        )

    positive_index = classes.index(1)

    positive_probabilities = probabilities[
        :,
        positive_index,
    ]

    if y_test.nunique() < 2:
        raise ValueError(
            "Data testing harus memiliki dua kelas."
        )

    values["roc_auc"] = float(
        roc_auc_score(
            y_test,
            positive_probabilities,
        )
    )

    # Average Precision is used as the PR-AUC-style metric.
    values["pr_auc"] = float(
        average_precision_score(
            y_test,
            positive_probabilities,
        )
    )

    values["brier_score"] = float(
        brier_score_loss(
            y_test,
            positive_probabilities,
        )
    )

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    values["true_negative"] = int(tn)
    values["false_positive"] = int(fp)
    values["false_negative"] = int(fn)
    values["true_positive"] = int(tp)

    return values


# ============================================================
# 6. CHRONOLOGICAL DATA SPLIT
# ============================================================

def load_experiment_data():
    """
    Load experiment dataset and split it chronologically.

    Training:
        FL_DATE < TEST_START

    Testing:
        TEST_START <= FL_DATE < TEST_END

    FL_DATE is only used for splitting and reporting.
    It is never included in X.
    """

    if not EXPERIMENT_DATA_PATH.exists():
        raise FileNotFoundError(
            "Dataset eksperimen tidak ditemukan: "
            f"{EXPERIMENT_DATA_PATH}"
        )

    data = pd.read_parquet(
        EXPERIMENT_DATA_PATH
    )

    required_columns = (
        list(FEATURES)
        + [TARGET, "FL_DATE"]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Kolom dataset tidak lengkap. "
            f"Kolom hilang: {missing_columns}"
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
            "Target DELAY_15 hanya boleh bernilai 0 atau 1."
        )

    if data[TARGET].nunique() < 2:
        raise ValueError(
            "Dataset harus mempunyai dua kelas target."
        )

    # --------------------------------------------------------
    # Split date configuration
    # --------------------------------------------------------

    test_start = pd.Timestamp(
        os.getenv(
            "EXPERIMENT_TEST_START",
            DEFAULT_TEST_START,
        )
    )

    test_end = pd.Timestamp(
        os.getenv(
            "EXPERIMENT_TEST_END",
            DEFAULT_TEST_END,
        )
    )

    if test_end <= test_start:
        raise ValueError(
            "EXPERIMENT_TEST_END harus setelah TEST_START."
        )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    train_data = data.loc[
        data["FL_DATE"] < test_start
    ].copy()

    test_data = data.loc[
        (data["FL_DATE"] >= test_start)
        & (data["FL_DATE"] < test_end)
    ].copy()

    if train_data.empty:
        raise ValueError(
            "Data training kosong. Periksa tanggal pemisahan."
        )

    if test_data.empty:
        raise ValueError(
            "Data testing kosong. Periksa tanggal pemisahan."
        )

    if train_data[TARGET].nunique() < 2:
        raise ValueError(
            "Data training hanya memiliki satu kelas."
        )

    if test_data[TARGET].nunique() < 2:
        raise ValueError(
            "Data testing hanya memiliki satu kelas."
        )

    # Explicit chronological integrity check.
    if (
        train_data["FL_DATE"].max()
        >= test_data["FL_DATE"].min()
    ):
        raise ValueError(
            "Terjadi kebocoran waktu antara training dan testing."
        )

    return (
        data,
        train_data,
        test_data,
        test_start,
        test_end,
    )


# ============================================================
# 7. CANDIDATE MODELS
# ============================================================

def get_candidates():
    """
    Return five candidates for comparison.

    1. Dummy Baseline
    2. Logistic Regression - Raw HHMM
    3. Logistic Regression - Cyclical
    4. Extra Trees - Raw HHMM
    5. Extra Trees - Cyclical
    """

    logistic_raw = LogisticRegression(
        max_iter=250,
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    logistic_cyclic = LogisticRegression(
        max_iter=250,
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    extra_trees_raw = ExtraTreesClassifier(
        n_estimators=60,
        max_depth=16,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    extra_trees_cyclic = ExtraTreesClassifier(
        n_estimators=60,
        max_depth=16,
        min_samples_leaf=10,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    return [
        (
            "dummy_baseline",
            DummyClassifier(
                strategy="prior"
            ),
            False,
        ),

        (
            "logistic_regression_raw",
            logistic_raw,
            False,
        ),

        (
            "logistic_regression_cyclic",
            logistic_cyclic,
            True,
        ),

        (
            "extra_trees_raw",
            extra_trees_raw,
            False,
        ),

        (
            "extra_trees_cyclic",
            extra_trees_cyclic,
            True,
        ),
    ]


# ============================================================
# 8. EXPERIMENT DISPLAY
# ============================================================

def print_experiment_header():
    print("=" * 65)
    print("FLIGHT DELAY - TIME ENCODING EXPERIMENT")
    print("=" * 65)


def print_dataset_information(
    data,
    train_period,
    test_period,
    test_start,
    test_end,
):
    print("\nDATASET")
    print("-" * 65)

    print(
        "Dataset:",
        EXPERIMENT_DATA_PATH,
    )

    print(
        "Total rows:",
        f"{len(data):,}",
    )

    print(
        "Date range:",
        data["FL_DATE"].min().date(),
        "to",
        data["FL_DATE"].max().date(),
    )

    print("\nCHRONOLOGICAL SPLIT")
    print("-" * 65)

    print(
        "Training period:",
        train_period["FL_DATE"].min().date(),
        "to",
        train_period["FL_DATE"].max().date(),
    )

    print(
        "Testing period:",
        test_period["FL_DATE"].min().date(),
        "to",
        test_period["FL_DATE"].max().date(),
    )

    print(
        "Testing date boundary:",
        test_start.date(),
        "to",
        test_end.date(),
        "(exclusive)",
    )

    print(
        "Available training rows:",
        f"{len(train_period):,}",
    )

    print(
        "Available testing rows:",
        f"{len(test_period):,}",
    )

    print(
        "Training delay rate:",
        f"{train_period[TARGET].mean() * 100:.2f}%",
    )

    print(
        "Testing delay rate:",
        f"{test_period[TARGET].mean() * 100:.2f}%",
    )


def print_sampling_information(
    train_data,
    test_data,
):
    print("\nSAMPLING")
    print("-" * 65)

    print(
        "Training rows used:",
        f"{len(train_data):,}",
    )

    print(
        "Testing rows used:",
        f"{len(test_data):,}",
    )

    print(
        "Training delay rate after sampling:",
        f"{train_data[TARGET].mean() * 100:.2f}%",
    )

    print(
        "Testing delay rate after sampling:",
        f"{test_data[TARGET].mean() * 100:.2f}%",
    )


# ============================================================
# 9. RUN ONE EXPERIMENT
# ============================================================

def run_candidate(
    name,
    estimator,
    use_cyclic_time,
    X_train,
    y_train,
    X_test,
    y_test,
    train_data,
    test_data,
    test_start,
    test_end,
):
    """
    Train and evaluate one candidate.

    Important:
        This function logs metrics and parameters only.

        It does NOT:
        - Save a model with MLflow.
        - Register a model.
        - Change a champion alias.
        - Overwrite MODEL_PATH.
    """

    print("\n" + "=" * 65)
    print("TRAINING:", name)
    print("=" * 65)

    model = build_pipeline(
        estimator,
        use_cyclic_time=use_cyclic_time,
    )

    with mlflow.start_run(
        run_name=name
    ) as run:

        # ----------------------------------------------------
        # Log experiment parameters
        # ----------------------------------------------------

        mlflow.log_params({
            "candidate": name,
            "dataset": EXPERIMENT_DATA_PATH.name,
            "target_definition": "DEP_DELAY >= 15",
            "split_strategy": "chronological",
            "train_period_start": str(
                train_data["FL_DATE"].min().date()
            ),
            "train_period_end": str(
                train_data["FL_DATE"].max().date()
            ),
            "test_period_start": str(
                test_data["FL_DATE"].min().date()
            ),
            "test_period_end": str(
                test_data["FL_DATE"].max().date()
            ),
            "test_boundary_start": str(
                test_start.date()
            ),
            "test_boundary_end_exclusive": str(
                test_end.date()
            ),
            "train_rows": int(len(train_data)),
            "test_rows": int(len(test_data)),
            "time_encoding": (
                "cyclical_sin_cos"
                if use_cyclic_time
                else "raw_hhmm"
            ),
            "decision_threshold": 0.5,
            "random_state": RANDOM_STATE,
            "model_artifact_saved": False,
            "registered_model": False,
            "champion_promoted": False,
        })

        mlflow.log_metric(
            "train_positive_rate",
            float(y_train.mean()),
        )

        mlflow.log_metric(
            "test_positive_rate",
            float(y_test.mean()),
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        print("Melatih model...", flush=True)

        model.fit(
            X_train,
            y_train,
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        print("Menghitung metrik...", flush=True)

        metrics = metrics_for(
            model,
            X_test,
            y_test,
        )

        # ----------------------------------------------------
        # Log metrics only
        # ----------------------------------------------------

        mlflow.log_metrics(
            metrics
        )

        # Explicitly record that no candidate model artifact
        # is saved during this comparison experiment.
        mlflow.log_param(
            "model_artifact_saved",
            False,
        )

        print("\nHASIL MODEL")

        for metric_name, value in metrics.items():
            print(
                f"{metric_name}: {value:.4f}",
                flush=True,
            )

        result = {
            "candidate": name,
            **metrics,
            "run_id": run.info.run_id,
        }

    # Return metrics and discard the fitted model from memory.
    del model

    return result


# ============================================================
# 10. MAIN EXPERIMENT
# ============================================================

def main():

    print_experiment_header()

    # --------------------------------------------------------
    # Load dataset and chronological split
    # --------------------------------------------------------

    (
        data,
        train_period,
        test_period,
        test_start,
        test_end,
    ) = load_experiment_data()

    print_dataset_information(
        data,
        train_period,
        test_period,
        test_start,
        test_end,
    )

    # --------------------------------------------------------
    # Sampling configuration
    # --------------------------------------------------------

    max_train_rows = int(
        os.getenv(
            "EXPERIMENT_TRAIN_ROWS",
            str(DEFAULT_TRAIN_ROWS),
        )
    )

    max_test_rows = int(
        os.getenv(
            "EXPERIMENT_TEST_ROWS",
            str(DEFAULT_TEST_ROWS),
        )
    )

    train_data = sample_rows(
        train_period,
        max_train_rows,
        RANDOM_STATE,
        "training",
    )

    test_data = sample_rows(
        test_period,
        max_test_rows,
        RANDOM_STATE + 1,
        "testing",
    )

    print_sampling_information(
        train_data,
        test_data,
    )

    # --------------------------------------------------------
    # Prepare model input
    # --------------------------------------------------------

    # FL_DATE is only used for chronological splitting.
    # It is NOT included in FEATURES.
    X_train = train_data[list(FEATURES)].copy()
    y_train = train_data[TARGET].astype(int).copy()

    X_test = test_data[list(FEATURES)].copy()
    y_test = test_data[TARGET].astype(int).copy()

    # Explicit check to prevent metadata leakage.
    if "FL_DATE" in X_train.columns:
        raise RuntimeError(
            "FL_DATE tidak boleh masuk ke fitur training."
        )

    if "FL_DATE" in X_test.columns:
        raise RuntimeError(
            "FL_DATE tidak boleh masuk ke fitur testing."
        )

    # --------------------------------------------------------
    # MLflow configuration
    # --------------------------------------------------------

    tracking_uri = MLFLOW_TRACKING_URI

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    print("\nMLFLOW")
    print("-" * 65)

    print(
        "Experiment:",
        EXPERIMENT_NAME,
    )

    print(
        "Tracking URI:",
        tracking_uri,
    )

    # --------------------------------------------------------
    # Run candidates
    # --------------------------------------------------------

    results = []

    candidates = get_candidates()

    print("\nJUMLAH KANDIDAT:", len(candidates))

    for (
        name,
        estimator,
        use_cyclic_time,
    ) in candidates:

        result = run_candidate(
            name=name,
            estimator=estimator,
            use_cyclic_time=use_cyclic_time,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            train_data=train_data,
            test_data=test_data,
            test_start=test_start,
            test_end=test_end,
        )

        results.append(result)

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    summary = pd.DataFrame(
        results
    )

    summary = summary.sort_values(
        by=["roc_auc", "f1"],
        ascending=False,
    ).reset_index(drop=True)

    display_columns = [
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
    print("RINGKASAN HASIL EKSPERIMEN")
    print("=" * 100)

    print(
        summary[display_columns].to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # --------------------------------------------------------
    # Save CSV summary
    # --------------------------------------------------------

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("\nHASIL CSV")
    print("-" * 65)

    print(
        "Tabel hasil disimpan di:",
        SUMMARY_PATH,
    )

    print("\nEXPERIMENT SELESAI")
    print("-" * 65)

    print(
        "MLflow experiment:",
        EXPERIMENT_NAME,
    )

    print(
        "Jumlah kandidat selesai:",
        len(results),
    )

    print(
        "Tidak ada model yang disimpan sebagai artefak."
    )

    print(
        "Tidak ada model yang didaftarkan ke Model Registry."
    )

    print(
        "Alias champion tidak diubah."
    )


# ============================================================
# 11. SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    main()
