from __future__ import annotations

import json
import os
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    MODEL_DIR,
    MLFLOW_TRACKING_URI,
    FEATURES,
    TARGET,
    CATEGORICAL,
    CYCLIC_NUMERIC,
)

from src.historical_features import (
    MODEL_FEATURES,
    CATEGORICAL_FEATURES,
    CYCLICAL_NUMERIC_FEATURES,
    HISTORICAL_RATE_FEATURES,
    add_cyclical_features,
    add_past_only_historical_rates,
    prepare_historical_reference,
)

from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURATION
# ============================================================

HISTORICAL_CANDIDATE_MODEL_PATH = (
    MODEL_DIR / "flight_delay_model_historical_candidate.joblib"
)

# Use the full experimental dataset (March-August 2026)
DATA_PATH = Path("data/processed/flights_mar_aug_experiment.parquet")

EXPERIMENT_NAME = "flight-delay-historical-candidate-training-v1"

RANDOM_STATE = 42

# Training sample sizes (for memory/time efficiency with ExtraTrees)
TRAIN_SAMPLE_ROWS = 200000
CALIBRATION_SAMPLE_ROWS = 100000
VALIDATION_SAMPLE_ROWS = 100000

# Training on March-May, Calibration on June, Validation on July
# This mirrors the final fold from threshold tuning
FOLDS = [
    {
        "name": "fold_july_2026",
        "train_start": "2026-03-01",
        "train_end": "2026-06-01",
        "calibration_start": "2026-06-01",
        "calibration_end": "2026-07-01",
        "validation_start": "2026-07-01",
        "validation_end": "2026-08-01",
    },
]

# Baseline from presentation (LogisticRegression cyclic, threshold 0.50 on July validation)
BASELINE_METRICS = {
    "roc_auc": 0.8337,
    "accuracy": 0.8186,
    "precision": 0.6514,
    "recall": 0.6586,
    "f1": 0.6550,
}

# Thresholds to evaluate
THRESHOLDS_TO_EVALUATE = [0.35, 0.495, 0.50, 0.575]

# Candidate models
CANDIDATES = [
    (
        "logistic_regression_historical",
        LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=4,
        ),
    ),
    (
        "extra_trees_historical",
        ExtraTreesClassifier(
            n_estimators=100,
            max_depth=16,
            min_samples_leaf=10,
            class_weight="balanced",
            n_jobs=4,
            random_state=RANDOM_STATE,
        ),
    ),
]


def load_and_prepare_data():
    """Load the experimental dataset and add historical rate features."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {DATA_PATH}")

    data = pd.read_parquet(DATA_PATH)

    required = list(FEATURES) + [TARGET, "FL_DATE"]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Kolom dataset tidak tersedia: {missing}")

    data["FL_DATE"] = pd.to_datetime(data["FL_DATE"], errors="coerce")
    data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
    data = data.dropna(subset=required).copy()
    data[TARGET] = data[TARGET].astype(int)

    if not data[TARGET].isin([0, 1]).all():
        raise ValueError("DELAY_15 hanya boleh bernilai 0 atau 1.")

    data = data.sort_values("FL_DATE").reset_index(drop=True)

    print("Menambahkan fitur historical rate (past-only)...")
    start = time.perf_counter()
    data = add_past_only_historical_rates(data)
    elapsed = time.perf_counter() - start
    print(f"Historical rates added in {elapsed:.2f}s")

    return data


def build_periods(data, fold):
    """Create strictly chronological train/calibration/validation periods."""
    train_start = pd.Timestamp(fold["train_start"])
    train_end = pd.Timestamp(fold["train_end"])
    cal_start = pd.Timestamp(fold["calibration_start"])
    cal_end = pd.Timestamp(fold["calibration_end"])
    val_start = pd.Timestamp(fold["validation_start"])
    val_end = pd.Timestamp(fold["validation_end"])

    train_full = data.loc[
        (data["FL_DATE"] >= train_start) & (data["FL_DATE"] < train_end)
    ].copy()

    calibration_full = data.loc[
        (data["FL_DATE"] >= cal_start) & (data["FL_DATE"] < cal_end)
    ].copy()

    validation_full = data.loc[
        (data["FL_DATE"] >= val_start) & (data["FL_DATE"] < val_end)
    ].copy()

    for name, frame in [
        ("training", train_full),
        ("calibration", calibration_full),
        ("validation", validation_full),
    ]:
        if frame.empty:
            raise ValueError(f"Data {name} kosong pada {fold['name']}.")
        if frame[TARGET].nunique() != 2:
            raise ValueError(f"Data {name} harus memiliki dua kelas pada {fold['name']}.")

    # Temporal integrity checks
    if not train_full["FL_DATE"].max() < calibration_full["FL_DATE"].min():
        raise RuntimeError("Temporal leakage: training >= calibration")
    if not calibration_full["FL_DATE"].max() < validation_full["FL_DATE"].min():
        raise RuntimeError("Temporal leakage: calibration >= validation")

    # Stratified sampling for memory/time efficiency
    def stratified_sample(frame, max_rows, description):
        if max_rows <= 0 or max_rows >= len(frame):
            return frame
        strata = frame["MONTH"].astype(str) + "_" + frame[TARGET].astype(str)
        sampled, _ = train_test_split(
            frame,
            train_size=max_rows,
            random_state=RANDOM_STATE,
            stratify=strata,
        )
        print(f"  {description}: {len(frame):,} -> {len(sampled):,} (stratified)")
        return sampled.reset_index(drop=True)

    train = stratified_sample(train_full, TRAIN_SAMPLE_ROWS, "Training")
    calibration = stratified_sample(calibration_full, CALIBRATION_SAMPLE_ROWS, "Calibration")
    validation = stratified_sample(validation_full, VALIDATION_SAMPLE_ROWS, "Validation")

    return train, calibration, validation


def prepare_features(frame, is_training=False):
    """Prepare features for model training/prediction."""
    # For training: we already have historical rates added
    # For prediction: we'd use prepare_prediction_features (not used here)

    # Add cyclical features (DEP_TIME_SIN, DEP_TIME_COS, DOW_SIN, DOW_COS, MONTH_SIN, MONTH_COS)
    frame = add_cyclical_features(frame)

    X = frame[MODEL_FEATURES].copy()
    y = frame[TARGET].astype(int).copy()

    if is_training:
        # Verify all historical rate features are present
        missing = [c for c in HISTORICAL_RATE_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"Historical rate features missing: {missing}")

    return X, y


def build_pipeline(estimator):
    """Build preprocessing pipeline for historical features."""

    # Categorical features (OP_CARRIER, ORIGIN, DEST)
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
    ])

    # Cyclical numeric features (DEP_TIME_SIN, DEP_TIME_COS, DOW_SIN, DOW_COS, MONTH_SIN, MONTH_COS, DISTANCE)
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ("cyclical_numeric", numeric_pipeline, CYCLICAL_NUMERIC_FEATURES),
        ("historical_rates", numeric_pipeline, HISTORICAL_RATE_FEATURES),
    ], remainder="drop")

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])


def evaluate_thresholds(model, X, y, thresholds):
    """Evaluate model at multiple thresholds."""
    if not hasattr(model, "predict_proba"):
        raise ValueError("Model tidak mendukung predict_proba.")

    probs = model.predict_proba(X)
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Model tidak memiliki kelas positif 1.")
    pos_idx = classes.index(1)
    probabilities = probs[:, pos_idx]

    results = []
    for thresh in thresholds:
        preds = (probabilities >= thresh).astype(int)

        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        metrics = {
            "threshold": thresh,
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probabilities)),
        }
        results.append(metrics)

    return results


def save_model_bundle(bundle, path):
    """Save model bundle with atomic write and verification."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    tmp_path = path.with_name(path.name + ".tmp")
    joblib.dump(bundle, tmp_path, compress=3)

    # Verify load
    restored = joblib.load(tmp_path)
    if not isinstance(restored, dict):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Format model tidak valid.")
    if "model" not in restored or "threshold" not in restored:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Bundle tidak memiliki model atau threshold.")
    if list(restored.get("feature_schema", [])) != list(MODEL_FEATURES):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Feature schema tidak cocok dengan historical_features.py.")

    # Verify probabilities match
    # (We'll skip detailed check since we don't have test data here)

    tmp_path.replace(path)
    print(f"Model tersimpan: {path}")


def main():
    print("\n" + "=" * 80)
    print("HISTORICAL CANDIDATE MODEL TRAINING")
    print("Leakage-free time-based split with historical delay rates")
    print("=" * 80)

    # Load and prepare data with historical features
    data = load_and_prepare_data()

    print("\nDATASET")
    print("-" * 80)
    print(f"File: {DATA_PATH}")
    print(f"Rows: {len(data):,}")
    print(f"Date range: {data['FL_DATE'].min().date()} to {data['FL_DATE'].max().date()}")
    print(f"Target rate: {data[TARGET].mean():.4f}")
    print(f"Features: {MODEL_FEATURES}")

    all_results = []

    for fold in FOLDS:
        print("\n" + "#" * 80)
        print(f"FOLD: {fold['name']}")
        print("#" * 80)

        train_full, calibration_full, validation_full = build_periods(data, fold)

        print("\nCHRONOLOGICAL PERIODS")
        print("-" * 80)
        for name, frame in [
            ("TRAINING", train_full),
            ("CALIBRATION", calibration_full),
            ("VALIDATION", validation_full),
        ]:
            print(
                f"{name}: {frame['FL_DATE'].min().date()} to {frame['FL_DATE'].max().date()} "
                f"| rows={len(frame):,} | delay={frame[TARGET].mean()*100:.2f}%"
            )

        # Prepare features
        X_train, y_train = prepare_features(train_full, is_training=True)
        X_calibration, y_calibration = prepare_features(calibration_full)
        X_validation, y_validation = prepare_features(validation_full)

        # Verify no FL_DATE leakage
        for name, X in [("training", X_train), ("calibration", X_calibration), ("validation", X_validation)]:
            if "FL_DATE" in X.columns:
                raise RuntimeError(f"FL_DATE masuk ke fitur {name}.")

        print(f"\nFeature matrix shapes:")
        print(f"  Training:      {X_train.shape}")
        print(f"  Calibration:   {X_calibration.shape}")
        print(f"  Validation:    {X_validation.shape}")

        # MLflow setup (skip if server not available)
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)
            mlflow_active = True
        except Exception as e:
            print(f"MLflow not available: {e}")
            mlflow_active = False

        # Train and evaluate each candidate
        for candidate_name, estimator in CANDIDATES:
            print("\n" + "=" * 80)
            print(f"CANDIDATE: {candidate_name}")
            print("=" * 80)

            run_name = f"{fold['name']}__{candidate_name}"

            # MLflow run context
            run_context = mlflow.start_run(run_name=run_name) if mlflow_active else None
            if run_context:
                run_context.__enter__()
            run_id = run_context.info.run_id if run_context else "local"

            try:
                # Log parameters
                if mlflow_active:
                    mlflow.log_params({
                        "candidate": candidate_name,
                        "fold": fold["name"],
                        "dataset": DATA_PATH.name,
                        "target_definition": "DEP_DELAY >= 15",
                        "split_strategy": "chronological_three_way",
                        "train_period": f"{fold['train_start']} to {fold['train_end']}",
                        "calibration_period": f"{fold['calibration_start']} to {fold['calibration_end']}",
                        "validation_period": f"{fold['validation_start']} to {fold['validation_end']}",
                        "training_rows": len(X_train),
                        "calibration_rows": len(X_calibration),
                        "validation_rows": len(X_validation),
                        "time_encoding": "cyclical_sin_cos",
                        "historical_rates": "past_only_leakage_free",
                        "features": ",".join(MODEL_FEATURES),
                        "random_state": RANDOM_STATE,
                        "model_artifact_saved": False,
                    })

                    mlflow.log_metrics({
                        "train_delay_rate": float(y_train.mean()),
                        "calibration_delay_rate": float(y_calibration.mean()),
                        "validation_delay_rate": float(y_validation.mean()),
                    })

                # Train
                print("Melatih model...", flush=True)
                train_start = time.perf_counter()

                model = build_pipeline(estimator)
                model.fit(X_train, y_train)

                train_time = time.perf_counter() - train_start
                print(f"Training selesai dalam {train_time:.2f} detik")

                if mlflow_active:
                    mlflow.log_metric("training_seconds", train_time)

                # Evaluate on calibration at multiple thresholds
                print("\nEvaluasi pada CALIBRATION period:")
                cal_results = evaluate_thresholds(model, X_calibration, y_calibration, THRESHOLDS_TO_EVALUATE)

                for r in cal_results:
                    print(
                        f"  thresh={r['threshold']:.3f} | "
                        f"acc={r['accuracy']:.4f} | "
                        f"prec={r['precision']:.4f} | "
                        f"rec={r['recall']:.4f} | "
                        f"f1={r['f1']:.4f} | "
                        f"auc={r['roc_auc']:.4f}"
                    )

                # Select best threshold on calibration (max F1)
                best_cal = max(cal_results, key=lambda x: x["f1"])
                selected_threshold = best_cal["threshold"]
                print(f"\nThreshold terpilih (calibration max F1): {selected_threshold:.3f}")

                if mlflow_active:
                    mlflow.log_param("selected_threshold", selected_threshold)
                    mlflow.log_metric("calibration_best_f1", best_cal["f1"])
                    mlflow.log_metric("calibration_best_recall", best_cal["recall"])
                    mlflow.log_metric("calibration_best_precision", best_cal["precision"])

                # Evaluate on validation at all thresholds
                print("\nEvaluasi pada VALIDATION period:")
                val_results = evaluate_thresholds(model, X_validation, y_validation, THRESHOLDS_TO_EVALUATE)

                for r in val_results:
                    print(
                        f"  thresh={r['threshold']:.3f} | "
                        f"acc={r['accuracy']:.4f} | "
                        f"prec={r['precision']:.4f} | "
                        f"rec={r['recall']:.4f} | "
                        f"f1={r['f1']:.4f} | "
                        f"auc={r['roc_auc']:.4f}"
                    )

                # Log validation metrics at each threshold
                if mlflow_active:
                    for r in val_results:
                        thresh_str = f"{r['threshold']:.3f}".replace(".", "_")
                        mlflow.log_metrics({
                            f"val_acc_{thresh_str}": r["accuracy"],
                            f"val_prec_{thresh_str}": r["precision"],
                            f"val_rec_{thresh_str}": r["recall"],
                            f"val_f1_{thresh_str}": r["f1"],
                            f"val_auc_{thresh_str}": r["roc_auc"],
                        })

                # Compare with baseline at threshold 0.50
                baseline_val = next(r for r in val_results if abs(r["threshold"] - 0.50) < 0.001)
                print("\nPERBANDINGAN DENGAN BASELINE (threshold 0.50):")
                print("-" * 80)
                print(f"{'Metric':<15} {'Baseline':>12} {'Historical':>12} {'Diff':>10}")
                print("-" * 80)
                for metric in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
                    base_val = BASELINE_METRICS[metric]
                    hist_val = baseline_val[metric]
                    diff = hist_val - base_val
                    print(f"{metric:<15} {base_val:>12.4f} {hist_val:>12.4f} {diff:>+10.4f}")

                # Collect results
                result = {
                    "fold": fold["name"],
                    "candidate": candidate_name,
                    "selected_threshold": selected_threshold,
                    "training_seconds": train_time,
                    "run_id": run_id,
                }
                for r in val_results:
                    thresh = r["threshold"]
                    for k, v in r.items():
                        if k != "threshold":
                            result[f"{k}_at_{thresh:.3f}"] = v
                result.update(best_cal)
                result["calibration_threshold"] = best_cal["threshold"]

                all_results.append(result)

                del model

            finally:
                if run_context:
                    run_context.__exit__(None, None, None)

    # Summary
    print("\n" + "=" * 100)
    print("RINGKASAN HASIL - VALIDATION METRICS AT SELECTED THRESHOLDS")
    print("=" * 100)

    detail_df = pd.DataFrame(all_results)
    detail_df.to_csv("data/processed/historical_candidate_detail.csv", index=False)

    # Print comparison table
    print("\nDetail per kandidat:")
    for candidate in [c[0] for c in CANDIDATES]:
        cand_data = detail_df[detail_df["candidate"] == candidate]
        print(f"\n--- {candidate} ---")
        for _, row in cand_data.iterrows():
            thresh = row["selected_threshold"]
            print(
                f"  Fold: {row['fold']}, Selected thresh: {thresh:.3f}, "
                f"Acc: {row[f'accuracy_at_{thresh:.3f}']:.4f}, "
                f"Prec: {row[f'precision_at_{thresh:.3f}']:.4f}, "
                f"Rec: {row[f'recall_at_{thresh:.3f}']:.4f}, "
                f"F1: {row[f'f1_at_{thresh:.3f}']:.4f}, "
                f"AUC: {row[f'roc_auc_at_{thresh:.3f}']:.4f}"
            )

    # Select best model based on F1 at selected threshold on validation
    best_result = max(all_results, key=lambda r: r[f"f1_at_{r['selected_threshold']:.3f}"])
    best_candidate_name = best_result["candidate"]
    best_threshold = best_result["selected_threshold"]
    best_f1 = best_result[f"f1_at_{best_threshold:.3f}"]

    print("\n" + "=" * 80)
    print("MODEL TERBAIK")
    print("=" * 80)
    print(f"Kandidat: {best_candidate_name}")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"Validation F1: {best_f1:.4f}")

    # Retrain best model on full training + calibration data
    print("\nMelatih ulang model terbaik pada Training + Calibration data...")
    best_estimator = next(est for name, est in CANDIDATES if name == best_candidate_name)

    combined_train = pd.concat([train_full, calibration_full], ignore_index=True)
    X_combined, y_combined = prepare_features(combined_train, is_training=True)

    final_model = build_pipeline(best_estimator)
    final_model.fit(X_combined, y_combined)

    # Final evaluation on validation
    print("\nEvaluasi final pada Validation set:")
    final_probs = final_model.predict_proba(X_validation)
    classes = list(final_model.classes_)
    pos_idx = classes.index(1)
    final_probabilities = final_probs[:, pos_idx]

    from sklearn.metrics import roc_auc_score
    final_auc = roc_auc_score(y_validation, final_probabilities)
    final_preds = (final_probabilities >= best_threshold).astype(int)

    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    final_metrics = {
        "accuracy": float(accuracy_score(y_validation, final_preds)),
        "precision": float(precision_score(y_validation, final_preds, zero_division=0)),
        "recall": float(recall_score(y_validation, final_preds, zero_division=0)),
        "f1": float(f1_score(y_validation, final_preds, zero_division=0)),
        "roc_auc": float(final_auc),
    }

    print(f"  Threshold: {best_threshold:.3f}")
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  F1:        {final_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {final_metrics['roc_auc']:.4f}")

    # Compare with baseline
    print("\nPERBANDINGAN FINAL DENGAN BASELINE:")
    print("-" * 80)
    print(f"{'Metric':<15} {'Baseline':>12} {'Final Model':>12} {'Diff':>10}")
    print("-" * 80)
    for metric in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
        base_val = BASELINE_METRICS[metric]
        hist_val = final_metrics[metric]
        diff = hist_val - base_val
        print(f"{metric:<15} {base_val:>12.4f} {hist_val:>12.4f} {diff:>+10.4f}")

    # Save model bundle
    bundle = {
        "model": final_model,
        "threshold": float(best_threshold),
        "model_name": best_candidate_name,
        "feature_schema": list(MODEL_FEATURES),
        "metadata": {
            "candidate": best_candidate_name,
            "threshold": float(best_threshold),
            "time_encoding": "cyclical_sin_cos",
            "historical_rates": "past_only_leakage_free",
            "features": list(MODEL_FEATURES),
            "training_period": f"{fold['train_start']} to {fold['calibration_end']}",
            "validation_period": f"{fold['validation_start']} to {fold['validation_end']}",
            "validation_metrics": final_metrics,
            "baseline_comparison": {
                metric: final_metrics[metric] - BASELINE_METRICS[metric]
                for metric in BASELINE_METRICS
            },
            "random_state": RANDOM_STATE,
        },
    }

    save_model_bundle(bundle, HISTORICAL_CANDIDATE_MODEL_PATH)

    # Save metadata
    metadata_path = MODEL_DIR / "historical_candidate_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(bundle["metadata"], f, indent=4, ensure_ascii=False)
    print(f"Metadata tersimpan: {metadata_path}")

    print("\n" + "=" * 80)
    print("TRAINING HISTORICAL CANDIDATE SELESAI")
    print("=" * 80)
    print(f"Model: {HISTORICAL_CANDIDATE_MODEL_PATH}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()