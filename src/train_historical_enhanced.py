from __future__ import annotations

import gc
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    MODEL_DIR,
    FEATURES,
    TARGET,
)

from src.historical_features import (
    MODEL_FEATURES,
    CATEGORICAL_FEATURES,
    CYCLICAL_NUMERIC_FEATURES,
    HISTORICAL_RATE_FEATURES,
    add_cyclical_features,
    add_past_only_historical_rates,
)

# ============================================================
# CONFIGURATION
# ============================================================

HISTORICAL_ENHANCED_MODEL_PATH = MODEL_DIR / "flight_delay_model_historical_candidate.joblib"
DATA_PATH = Path("data/processed/flights_mar_aug_experiment.parquet")

RANDOM_STATE = 42

# Sample sizes for efficiency
TRAIN_SAMPLE_ROWS = 500000
CALIBRATION_SAMPLE_ROWS = 200000
VALIDATION_SAMPLE_ROWS = 200000

# Chronological splits
FOLDS = [
    {
        "name": "fold_jul_aug_2026",
        "train_start": "2026-03-01",
        "train_end": "2026-06-01",
        "calibration_start": "2026-06-01",
        "calibration_end": "2026-07-01",
        "validation_start": "2026-07-01",
        "validation_end": "2026-09-01",
    },
]

# Baseline from historical candidate (LogisticRegression)
BASELINE_METRICS = {
    "roc_auc": 0.8337,
    "accuracy": 0.8186,
    "precision": 0.6514,
    "recall": 0.6586,
    "f1": 0.6550,
}

THRESHOLDS_TO_EVALUATE = np.linspace(0.10, 0.90, 81)


# ============================================================
# DATA LOADING
# ============================================================

def load_and_prepare_data():
    """Load dataset and add base features."""
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

    print("Menambahkan fitur cyclical + historical rate (leakage-free)...")
    start = time.perf_counter()
    data = add_cyclical_features(data)
    data = add_past_only_historical_rates(data)
    print(f"  Base features added in {time.perf_counter() - start:.2f}s")

    gc.collect()
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

    if not train_full["FL_DATE"].max() < calibration_full["FL_DATE"].min():
        raise RuntimeError("Temporal leakage: training >= calibration")
    if not calibration_full["FL_DATE"].max() < validation_full["FL_DATE"].min():
        raise RuntimeError("Temporal leakage: calibration >= validation")

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

    return train, calibration, validation, train_full, calibration_full, validation_full


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(frame, is_training=False):
    """Prepare features using MODEL_FEATURES."""
    frame = add_cyclical_features(frame)
    X = frame[MODEL_FEATURES].copy()
    y = frame[TARGET].astype(int).copy()

    if is_training:
        missing = [c for c in HISTORICAL_RATE_FEATURES if c not in X.columns]
        if missing:
            raise ValueError(f"Historical rate features missing: {missing}")

    return X, y


# ============================================================
# PIPELINE BUILDING
# ============================================================

def build_pipeline(estimator):
    """Build pipeline with preprocessing for categorical features."""
    cat_features = list(CATEGORICAL_FEATURES)
    numeric_features = [c for c in MODEL_FEATURES if c not in cat_features]

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler(with_mean=False)),
    ])

    preprocessor = ColumnTransformer([
        ("categorical", categorical_pipeline, cat_features),
        ("numeric", numeric_pipeline, numeric_features),
    ], remainder="drop", sparse_threshold=0.0, verbose_feature_names_out=False)

    return Pipeline([
        ("preprocess", preprocessor),
        ("model", estimator),
    ])


# ============================================================
# EVALUATION
# ============================================================

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
        metrics = {
            "threshold": float(thresh),
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "roc_auc": float(roc_auc_score(y, probabilities)),
        }
        results.append(metrics)

    return results


def calibrate_model(model, X_cal, y_cal):
    """Apply probability calibration."""
    calibrated = CalibratedClassifierCV(model, method="isotonic", cv=None)
    calibrated.fit(X_cal, y_cal)
    return calibrated


def save_model_bundle(bundle, path):
    """Save model bundle with atomic write and verification."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    joblib.dump(bundle, tmp_path, compress=3)

    restored = joblib.load(tmp_path)
    if not isinstance(restored, dict):
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Format model tidak valid.")
    if "model" not in restored or "threshold" not in restored:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError("Bundle tidak memiliki model atau threshold.")

    tmp_path.replace(path)
    print(f"Model tersimpan: {path}")


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def main():
    print("\n" + "=" * 80)
    print("HISTORICAL ENHANCED MODEL TRAINING (v2)")
    print("Leakage-free time-based split with advanced feature engineering")
    print("=" * 80)

    data = load_and_prepare_data()

    print("\nDATASET")
    print("-" * 80)
    print(f"File: {DATA_PATH}")
    print(f"Rows: {len(data):,}")
    print(f"Date range: {data['FL_DATE'].min().date()} to {data['FL_DATE'].max().date()}")
    print(f"Target rate: {data[TARGET].mean():.4f}")

    all_results = []
    best_overall = None
    best_overall_f1 = -1

    for fold in FOLDS:
        print("\n" + "#" * 80)
        print(f"FOLD: {fold['name']}")
        print("#" * 80)

        train, calibration, validation, train_full, cal_full, val_full = build_periods(data, fold)

        print("\nCHRONOLOGICAL PERIODS")
        print("-" * 80)
        for name, frame in [
            ("TRAINING", train_full),
            ("CALIBRATION", cal_full),
            ("VALIDATION", val_full),
        ]:
            print(
                f"{name}: {frame['FL_DATE'].min().date()} to {frame['FL_DATE'].max().date()} "
                f"| rows={len(frame):,} | delay={frame[TARGET].mean()*100:.2f}%"
            )

        X_train, y_train = prepare_features(train, is_training=True)
        X_calibration, y_calibration = prepare_features(calibration)
        X_validation, y_validation = prepare_features(validation)

        print(f"\nFeature matrix shapes:")
        print(f"  Training:      {X_train.shape}")
        print(f"  Calibration:   {X_calibration.shape}")
        print(f"  Validation:    {X_validation.shape}")

        # Train LogisticRegression with enhanced features
        print("\n" + "=" * 80)
        print("CANDIDATE: logistic_regression_enhanced")
        print("=" * 80)

        estimator = LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

        print("Melatih model pada training data...")
        train_start = time.perf_counter()
        model = build_pipeline(estimator)
        model.fit(X_train, y_train)
        train_time = time.perf_counter() - train_start
        print(f"Training selesai dalam {train_time:.2f} detik")

        print("Kalibrasi probabilitas pada calibration set...")
        calibrated_model = calibrate_model(model, X_calibration, y_calibration)

        print("\nEvaluasi pada CALIBRATION period:")
        cal_results = evaluate_thresholds(calibrated_model, X_calibration, y_calibration, THRESHOLDS_TO_EVALUATE)

        for r in cal_results:
            print(
                f"  thresh={r['threshold']:.3f} | "
                f"acc={r['accuracy']:.4f} | prec={r['precision']:.4f} | "
                f"rec={r['recall']:.4f} | f1={r['f1']:.4f} | auc={r['roc_auc']:.4f}"
            )

        best_cal = max(cal_results, key=lambda x: x["f1"])
        selected_threshold = best_cal["threshold"]
        print(f"\nThreshold terpilih (calibration max F1): {selected_threshold:.3f}")

        print("\nEvaluasi pada VALIDATION period:")
        val_results = evaluate_thresholds(calibrated_model, X_validation, y_validation, THRESHOLDS_TO_EVALUATE)

        for r in val_results:
            print(
                f"  thresh={r['threshold']:.3f} | "
                f"acc={r['accuracy']:.4f} | prec={r['precision']:.4f} | "
                f"rec={r['recall']:.4f} | f1={r['f1']:.4f} | auc={r['roc_auc']:.4f}"
            )

        # Compare with baseline at threshold 0.50
        baseline_val = next(r for r in val_results if abs(r["threshold"] - 0.50) < 0.001)
        print("\nPERBANDINGAN DENGAN BASELINE (threshold 0.50):")
        print("-" * 80)
        print(f"{'Metric':<15} {'Baseline':>12} {'Enhanced':>12} {'Diff':>10}")
        print("-" * 80)
        for metric in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
            base_val = BASELINE_METRICS[metric]
            enh_val = baseline_val[metric]
            diff = enh_val - base_val
            print(f"{metric:<15} {base_val:>12.4f} {enh_val:>12.4f} {diff:>+10.4f}")

        result = {
            "fold": fold["name"],
            "candidate": "logistic_regression_enhanced",
            "selected_threshold": selected_threshold,
            "training_seconds": train_time,
            "best_params": {},
            "n_features": len(MODEL_FEATURES),
        }
        for r in val_results:
            thresh = r["threshold"]
            for k, v in r.items():
                if k != "threshold":
                    result[f"{k}_at_{thresh:.3f}"] = v
        result.update(best_cal)
        result["calibration_threshold"] = best_cal["threshold"]

        all_results.append(result)
        val_f1 = result[f"f1_at_{selected_threshold:.3f}"]
        if val_f1 > best_overall_f1:
            best_overall_f1 = val_f1
            best_overall = result.copy()
            best_overall["model"] = calibrated_model

        del model, calibrated_model
        gc.collect()

    # Summary
    print("\n" + "=" * 100)
    print("RINGKASAN HASIL - VALIDATION METRICS AT SELECTED THRESHOLDS")
    print("=" * 100)

    detail_df = pd.DataFrame(all_results)
    detail_df.to_csv("data/processed/historical_enhanced_detail.csv", index=False)

    print("\n--- logistic_regression_enhanced ---")
    for _, row in detail_df.iterrows():
        thresh = row["selected_threshold"]
        print(
            f"  Fold: {row['fold']}, Selected thresh: {thresh:.3f}, "
            f"Acc: {row[f'accuracy_at_{thresh:.3f}']:.4f}, "
            f"Prec: {row[f'precision_at_{thresh:.3f}']:.4f}, "
            f"Rec: {row[f'recall_at_{thresh:.3f}']:.4f}, "
            f"F1: {row[f'f1_at_{thresh:.3f}']:.4f}, "
            f"AUC: {row[f'roc_auc_at_{thresh:.3f}']:.4f}"
        )

    print("\n" + "=" * 80)
    print("MODEL TERBAIK OVERALL")
    print("=" * 80)
    print(f"Kandidat: {best_overall['candidate']}")
    print(f"Threshold: {best_overall['selected_threshold']:.3f}")
    print(f"Validation F1: {best_overall_f1:.4f}")

    print("\nMelatih ulang model terbaik pada Training + Calibration data...")
    estimator = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    combined_train = pd.concat([train_full, cal_full], ignore_index=True)
    X_combined, y_combined = prepare_features(combined_train, is_training=True)

    final_model = build_pipeline(estimator)
    final_model.fit(X_combined, y_combined)

    X_cal_final, y_cal_final = prepare_features(cal_full)
    final_model = calibrate_model(final_model, X_cal_final, y_cal_final)

    print("\nEvaluasi final pada Validation set:")
    final_probs = final_model.predict_proba(X_validation)
    classes = list(final_model.classes_)
    pos_idx = classes.index(1)
    final_probabilities = final_probs[:, pos_idx]
    best_threshold = best_overall["selected_threshold"]
    final_preds = (final_probabilities >= best_threshold).astype(int)

    final_metrics = {
        "accuracy": float(accuracy_score(y_validation, final_preds)),
        "precision": float(precision_score(y_validation, final_preds, zero_division=0)),
        "recall": float(recall_score(y_validation, final_preds, zero_division=0)),
        "f1": float(f1_score(y_validation, final_preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_validation, final_probabilities)),
    }

    print(f"  Threshold: {best_threshold:.3f}")
    print(f"  Accuracy:  {final_metrics['accuracy']:.4f}")
    print(f"  Precision: {final_metrics['precision']:.4f}")
    print(f"  Recall:    {final_metrics['recall']:.4f}")
    print(f"  F1:        {final_metrics['f1']:.4f}")
    print(f"  ROC-AUC:   {final_metrics['roc_auc']:.4f}")

    target_met = all(v >= 0.80 for v in final_metrics.values())
    print(f"\n  TARGET >= 80% ALL METRICS: {'✅ BERHASIL' if target_met else '❌ BELUM TERCAPAI'}")
    for metric, value in final_metrics.items():
        status = "✅" if value >= 0.80 else "❌"
        print(f"    {metric}: {value:.4f} {status}")

    print("\nPERBANDINGAN FINAL DENGAN BASELINE:")
    print("-" * 80)
    print(f"{'Metric':<15} {'Baseline':>12} {'Final Model':>12} {'Diff':>10}")
    print("-" * 80)
    for metric in ["roc_auc", "accuracy", "precision", "recall", "f1"]:
        base_val = BASELINE_METRICS[metric]
        hist_val = final_metrics[metric]
        diff = hist_val - base_val
        print(f"{metric:<15} {base_val:>12.4f} {hist_val:>12.4f} {diff:>+10.4f}")

    bundle = {
        "model": final_model,
        "threshold": float(best_threshold),
        "model_name": "logistic_regression_enhanced",
        "feature_schema": MODEL_FEATURES,
        "metadata": {
            "candidate": "logistic_regression_enhanced",
            "threshold": float(best_threshold),
            "time_encoding": "cyclical_sin_cos",
            "historical_rates": "past_only_leakage_free",
            "features": MODEL_FEATURES,
            "training_period": f"{fold['train_start']} to {fold['calibration_end']}",
            "validation_period": f"{fold['validation_start']} to {fold['validation_end']}",
            "validation_metrics": final_metrics,
            "baseline_comparison": {
                metric: final_metrics[metric] - BASELINE_METRICS[metric]
                for metric in BASELINE_METRICS
            },
            "target_80_met": target_met,
            "best_params": {},
            "random_state": RANDOM_STATE,
        },
    }

    save_model_bundle(bundle, HISTORICAL_ENHANCED_MODEL_PATH)

    metadata_path = MODEL_DIR / "historical_candidate_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(bundle["metadata"], f, indent=4, ensure_ascii=False)
    print(f"Metadata tersimpan: {metadata_path}")

    report_path = Path("data/processed/historical_candidate_v2_report.json")
    report = {
        "best_model": {k: v for k, v in best_overall.items() if k not in ("model",)},
        "all_results": all_results,
        "final_metrics": final_metrics,
        "target_80_met": target_met,
        "baseline_metrics": BASELINE_METRICS,
        "feature_columns": MODEL_FEATURES,
    }
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=4, ensure_ascii=False, default=str)
    print(f"Laporan lengkap tersimpan: {report_path}")

    print("\n" + "=" * 80)
    print("TRAINING HISTORICAL ENHANCED SELESAI")
    print("=" * 80)
    print(f"Model: {HISTORICAL_ENHANCED_MODEL_PATH}")
    print(f"Metadata: {metadata_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()