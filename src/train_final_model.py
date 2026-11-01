from __future__ import annotations

import json
import os
import time

import joblib
import mlflow
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import (
    FEATURES,
    TARGET,
    MODEL_DIR,
    MLFLOW_TRACKING_URI,
)

from src.features import hhmm_to_cyclic

from src.experiment_time_features import (
    get_candidates,
    RANDOM_STATE,
)

from src.experiment_threshold_tuning import (
    DATA_PATH,
    FOLDS,
    load_dataset,
    build_periods,
    stratified_month_sample,
    positive_probabilities,
    select_threshold,
    metrics_from_probabilities,
)


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_CANDIDATE_PATH = (
    MODEL_DIR / "flight_delay_model_cyclic_candidate.joblib"
)

OUTPUT_DIR = DATA_PATH.parent

CALIBRATION_OUTPUT = (
    OUTPUT_DIR / "final_candidate_calibration_grid.csv"
)

METADATA_OUTPUT = (
    OUTPUT_DIR / "final_candidate_metadata.json"
)

EXPERIMENT_NAME = (
    "flight-delay-final-candidate-validation-v1"
)

CANDIDATE_NAME = "logistic_regression_cyclic"

# Reproduce the July fold used in the 100K experiment.
FINAL_FOLD_NAME = "fold_july_2026"

TRAIN_ROWS = int(
    os.getenv(
        "FINAL_TRAIN_ROWS",
        "100000",
    )
)

# Recall target used during threshold selection.
TARGET_RECALL = 0.80

# Final candidate must use cyclic encoding.
USE_CYCLIC_TIME = True


# ============================================================
# PIPELINE
# ============================================================

def build_final_pipeline(estimator):
    """
    Build the same Logistic Regression Cyclic architecture
    used in the previous experiments.

    CRS_DEP_TIME is transformed into sine and cosine.
    FL_DATE is not included in model features.
    """

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

    preprocessor = ColumnTransformer([
        (
            "categorical",
            categorical_pipeline,
            [
                "OP_CARRIER",
                "ORIGIN",
                "DEST",
            ],
        ),
        (
            "departure_time_cyclic",
            departure_time_pipeline,
            [
                "CRS_DEP_TIME",
            ],
        ),
        (
            "numeric",
            numeric_pipeline,
            [
                "MONTH",
                "DAY_OF_WEEK",
                "DISTANCE",
            ],
        ),
    ])

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
# FIND CANDIDATE
# ============================================================

def get_selected_estimator():
    """
    Reuse the exact Logistic Regression parameters
    from the previous experiment.
    """

    candidates = get_candidates()

    for name, estimator, use_cyclic in candidates:

        if name == CANDIDATE_NAME:

            if not use_cyclic:
                raise RuntimeError(
                    "Kandidat yang dipilih bukan model cyclic."
                )

            return estimator

    raise RuntimeError(
        f"Kandidat {CANDIDATE_NAME} tidak ditemukan."
    )


# ============================================================
# SAVE MODEL SAFELY
# ============================================================

def save_candidate_bundle(bundle):
    """
    Save candidate model separately from the production model.

    Uses a temporary file and verifies the serialized model
    before replacing the candidate artifact.
    """

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = MODEL_CANDIDATE_PATH.with_name(
        MODEL_CANDIDATE_PATH.name + ".tmp"
    )

    joblib.dump(
        bundle,
        temporary_path,
        compress=3,
    )

    # Verify that the artifact can be loaded.
    restored = joblib.load(
        temporary_path
    )

    if not isinstance(restored, dict):
        temporary_path.unlink(missing_ok=True)

        raise RuntimeError(
            "Format model kandidat tidak valid."
        )

    if "model" not in restored or "threshold" not in restored:
        temporary_path.unlink(missing_ok=True)

        raise RuntimeError(
            "Bundle model tidak memiliki model atau threshold."
        )

    # Compare predicted probabilities before and after serialization.
    original_model = bundle["model"]
    restored_model = restored["model"]

    return temporary_path, restored_model


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 75)
    print("FINAL LOGISTIC REGRESSION CYCLIC CANDIDATE")
    print("=" * 75)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    data = load_dataset()

    print("\nDATASET")
    print("-" * 75)

    print("File:", DATA_PATH)
    print("Rows:", f"{len(data):,}")
    print(
        "Date range:",
        data["FL_DATE"].min().date(),
        "to",
        data["FL_DATE"].max().date(),
    )

    # --------------------------------------------------------
    # Select July fold
    # --------------------------------------------------------

    fold = next(
        (
            item
            for item in FOLDS
            if item["name"] == FINAL_FOLD_NAME
        ),
        None,
    )

    if fold is None:
        raise ValueError(
            f"Fold {FINAL_FOLD_NAME} tidak ditemukan."
        )

    fold_index = FOLDS.index(fold)

    (
        train_full,
        calibration_full,
        validation_full,
    ) = build_periods(
        data,
        fold,
    )

    print("\nCHRONOLOGICAL DATA")
    print("-" * 75)

    print(
        "Training:",
        train_full["FL_DATE"].min().date(),
        "to",
        train_full["FL_DATE"].max().date(),
        "| rows:",
        f"{len(train_full):,}",
    )

    print(
        "Calibration:",
        calibration_full["FL_DATE"].min().date(),
        "to",
        calibration_full["FL_DATE"].max().date(),
        "| rows:",
        f"{len(calibration_full):,}",
    )

    print(
        "Validation:",
        validation_full["FL_DATE"].min().date(),
        "to",
        validation_full["FL_DATE"].max().date(),
        "| rows:",
        f"{len(validation_full):,}",
    )

    # --------------------------------------------------------
    # Reproduce exact sampling from the 100K July fold
    # --------------------------------------------------------

    train_sample = stratified_month_sample(
        train_full,
        TRAIN_ROWS,
        RANDOM_STATE + fold_index,
        "final_training",
    )

    calibration_sample = stratified_month_sample(
        calibration_full,
        0,
        RANDOM_STATE + fold_index + 10,
        "final_calibration",
    )

    validation_sample = stratified_month_sample(
        validation_full,
        0,
        RANDOM_STATE + fold_index + 20,
        "final_validation",
    )

    print("\nSAMPLING")
    print("-" * 75)

    print("Training:", f"{len(train_sample):,}")
    print("Calibration:", f"{len(calibration_sample):,}")
    print("Validation:", f"{len(validation_sample):,}")

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X_train = train_sample[list(FEATURES)].copy()
    y_train = train_sample[TARGET].astype(int).copy()

    X_calibration = calibration_sample[list(FEATURES)].copy()
    y_calibration = calibration_sample[TARGET].astype(int).copy()

    X_validation = validation_sample[list(FEATURES)].copy()
    y_validation = validation_sample[TARGET].astype(int).copy()

    if "FL_DATE" in X_train.columns:
        raise RuntimeError(
            "FL_DATE tidak boleh menjadi fitur training."
        )

    # --------------------------------------------------------
    # Get estimator and build pipeline
    # --------------------------------------------------------

    estimator = get_selected_estimator()

    model = build_final_pipeline(
        estimator
    )

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

    print("\nTRAINING")
    print("-" * 75)

    start_time = time.perf_counter()

    model.fit(
        X_train,
        y_train,
    )

    training_seconds = (
        time.perf_counter() - start_time
    )

    print(
        "Training selesai dalam:",
        f"{training_seconds:.2f} detik",
    )

    # --------------------------------------------------------
    # Select threshold using June calibration
    # --------------------------------------------------------

    print("\nCALIBRATION")
    print("-" * 75)

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
        FINAL_FOLD_NAME,
        CANDIDATE_NAME,
    )

    threshold = float(
        selected_threshold["threshold"]
    )

    print(
        "Selected threshold:",
        f"{threshold:.2f}",
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

    # --------------------------------------------------------
    # Validate on July
    # --------------------------------------------------------

    print("\nVALIDATION")
    print("-" * 75)

    validation_probabilities = positive_probabilities(
        model,
        X_validation,
    )

    validation_metrics = metrics_from_probabilities(
        y_validation,
        validation_probabilities,
        threshold,
    )

    baseline_metrics = metrics_from_probabilities(
        y_validation,
        validation_probabilities,
        0.50,
    )

    print("\nMETRICS - SELECTED THRESHOLD")

    for key, value in validation_metrics.items():
        print(
            f"{key}: {value:.4f}"
        )

    print("\nMETRICS - DEFAULT THRESHOLD 0.50")

    for key in [
        "accuracy",
        "precision",
        "recall",
        "f1",
    ]:
        print(
            f"{key}: {baseline_metrics[key]:.4f}"
        )

    # --------------------------------------------------------
    # Build model bundle
    # --------------------------------------------------------

    metadata = {
        "candidate": CANDIDATE_NAME,
        "time_encoding": "cyclical_sin_cos",
        "threshold": threshold,
        "decision_rule": "positive_probability >= threshold",
        "target_definition": "DEP_DELAY >= 15",
        "features": list(FEATURES),
        "training_start": str(
            train_full["FL_DATE"].min().date()
        ),
        "training_end": str(
            train_full["FL_DATE"].max().date()
        ),
        "training_available_rows": int(len(train_full)),
        "training_sample_rows": int(len(train_sample)),
        "calibration_start": str(
            calibration_full["FL_DATE"].min().date()
        ),
        "calibration_end": str(
            calibration_full["FL_DATE"].max().date()
        ),
        "validation_start": str(
            validation_full["FL_DATE"].min().date()
        ),
        "validation_end": str(
            validation_full["FL_DATE"].max().date()
        ),
        "calibration_recall_target": TARGET_RECALL,
        "calibration_recall_target_met": bool(
            recall_constraint_met
        ),
        "random_state": RANDOM_STATE,
        "validation_metrics": validation_metrics,
        "default_threshold_metrics": baseline_metrics,
    }

    bundle = {
        "model": model,
        "threshold": threshold,
        "model_name": CANDIDATE_NAME,
        "metadata": metadata,
    }

    # --------------------------------------------------------
    # Serialize and verify
    # --------------------------------------------------------

    print("\nSAVING CANDIDATE")
    print("-" * 75)

    temporary_path, restored_model = save_candidate_bundle(
        bundle
    )

    check_rows = min(
        200,
        len(X_validation),
    )

    original_probabilities = model.predict_proba(
        X_validation.iloc[:check_rows]
    )

    restored_probabilities = restored_model.predict_proba(
        X_validation.iloc[:check_rows]
    )

    np.testing.assert_allclose(
        original_probabilities,
        restored_probabilities,
        rtol=1e-7,
        atol=1e-9,
    )

    temporary_path.replace(
        MODEL_CANDIDATE_PATH
    )

    print(
        "Candidate model tersimpan:",
        MODEL_CANDIDATE_PATH,
    )

    print(
        f"Verifikasi serialisasi berhasil untuk {check_rows} baris."
    )

    # --------------------------------------------------------
    # Save metadata and calibration CSV
    # --------------------------------------------------------

    with METADATA_OUTPUT.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False,
        )

    calibration_frame = pd.DataFrame(
        calibration_grid
    )

    calibration_frame.to_csv(
        CALIBRATION_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------
    # MLflow logging
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name="final_logistic_regression_cyclic_candidate"
    ):

        mlflow.log_params({
            "candidate": CANDIDATE_NAME,
            "dataset": DATA_PATH.name,
            "training_sample_rows": len(train_sample),
            "calibration_rows": len(calibration_sample),
            "validation_rows": len(validation_sample),
            "time_encoding": "cyclical_sin_cos",
            "selected_threshold": threshold,
            "threshold_selection_period": "2026-06",
            "validation_period": "2026-07",
            "model_artifact_saved_to_registry": False,
            "registered_model": False,
            "champion_promoted": False,
        })

        mlflow.log_metrics({
            f"validation_{key}": float(value)
            for key, value in validation_metrics.items()
        })

        mlflow.log_metrics({
            f"default_threshold_{key}": float(
                baseline_metrics[key]
            )
            for key in [
                "accuracy",
                "precision",
                "recall",
                "f1",
            ]
        })

        mlflow.log_artifact(
            str(METADATA_OUTPUT)
        )

        mlflow.log_artifact(
            str(CALIBRATION_OUTPUT)
        )

    # --------------------------------------------------------
    # Completion
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("FINAL CANDIDATE TRAINING SELESAI")
    print("=" * 75)

    print(
        "Candidate:",
        CANDIDATE_NAME,
    )

    print(
        "Threshold:",
        threshold,
    )

    print(
        "Model file:",
        MODEL_CANDIDATE_PATH,
    )

    print(
        "Metadata:",
        METADATA_OUTPUT,
    )

    print(
        "MLflow experiment:",
        EXPERIMENT_NAME,
    )

    print(
        "Model Registry modified: NO"
    )

    print(
        "Champion alias modified: NO"
    )


if __name__ == "__main__":
    main()