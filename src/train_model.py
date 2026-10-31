from __future__ import annotations

import os
import warnings

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import FLIGHTS_PATH, FEATURES, TARGET, MODEL_DIR, MODEL_PATH, MLFLOW_TRACKING_URI, MODEL_NAME

CATEGORICAL = ["OP_CARRIER", "ORIGIN", "DEST"]
NUMERIC = ["CRS_DEP_TIME", "MONTH", "DAY_OF_WEEK", "DISTANCE"]


def build_pipeline(estimator):
    preprocessor = ColumnTransformer([
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))]), CATEGORICAL),
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler(with_mean=False))]), NUMERIC),
    ])
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def metrics_for(model, X_test, y_test):
    predictions = model.predict(X_test)
    values = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
    }
    if hasattr(model, "predict_proba") and y_test.nunique() == 2:
        try:
            values["roc_auc"] = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, list(model.classes_).index(1)]))
        except Exception:
            values["roc_auc"] = 0.0
    else:
        values["roc_auc"] = 0.0
    return values


def main():
    if not FLIGHTS_PATH.exists():
        raise FileNotFoundError(f"Dataset bersih tidak ditemukan: {FLIGHTS_PATH}. Jalankan python -m src.data_pipeline terlebih dahulu.")
    os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("flight-delay-model-selection")

    data = pd.read_parquet(FLIGHTS_PATH)
    data = data.dropna(subset=FEATURES + [TARGET])
    if data[TARGET].nunique() < 2:
        raise ValueError("Target harus memiliki dua kelas: delay <15 menit dan delay >=15 menit.")
    cap = int(os.getenv("MAX_TRAIN_ROWS", "50000"))
    if cap > 0 and len(data) > cap:
        # Use sklearn's stratified sampling instead of GroupBy.apply.
        # Newer pandas releases can exclude the grouping column from apply output,
        # which would silently drop TARGET (DELAY_15) and break model training.
        data, _ = train_test_split(
            data,
            train_size=cap,
            random_state=42,
            stratify=data[TARGET],
        )
        data = data.reset_index(drop=True)

    # Fail early with a diagnostic if the cleaned dataset schema is inconsistent.
    missing_features = [column for column in FEATURES if column not in data.columns]
    if TARGET not in data.columns or missing_features:
        raise ValueError(
            "Skema dataset tidak sesuai setelah sampling. "
            f"Target yang dibutuhkan: {TARGET!r}; "
            f"fitur hilang: {missing_features}; "
            f"kolom tersedia: {list(data.columns)}"
        )

    X = data[FEATURES].copy()
    y = data[TARGET].astype(int).copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    candidates = [
        ("dummy_baseline", DummyClassifier(strategy="prior")),
        ("logistic_regression", LogisticRegression(max_iter=250, solver="liblinear", class_weight="balanced", random_state=42)),
        ("extra_trees", ExtraTreesClassifier(n_estimators=60, max_depth=16, min_samples_leaf=10, class_weight="balanced", n_jobs=-1, random_state=42)),
    ]
    results = []
    for name, estimator in candidates:
        model = build_pipeline(estimator)
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_param("candidate", name)
            mlflow.log_param("target_definition", "DEP_DELAY >= 15")
            mlflow.log_param("train_test_split", "80/20 stratified random split")
            mlflow.log_param("rows_sampled", int(len(data)))
            mlflow.log_param("raw_rows_available", int(pd.read_parquet(FLIGHTS_PATH).shape[0]))
            model.fit(X_train, y_train)
            metrics = metrics_for(model, X_test, y_test)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                artifact_path="candidate_model",
                # The model is built in this project; explicitly trust NumPy dtype metadata
                # that skops otherwise flags during MLflow model validation.
                skops_trusted_types=[
                    "numpy.dtype",
                    "sklearn.tree._tree.Tree",
                ],
            )
            print(f"{name}: " + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()))
            results.append({"name": name, "model": model, "metrics": metrics, "run_id": run.info.run_id})

    # Model selection is explicit and metric-based, not based on a guessed score.
    best = max(results, key=lambda item: (item["metrics"].get("roc_auc", 0.0), item["metrics"].get("f1", 0.0)))
    print(f"Model terpilih: {best['name']} | metrics={best['metrics']}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best["model"], MODEL_PATH)

    with mlflow.start_run(run_name="register_selected_model"):
        mlflow.log_param("selected_candidate", best["name"])
        mlflow.log_param("selection_rule", "highest ROC-AUC, tie-break with F1")
        mlflow.log_metrics({f"selected_{key}": value for key, value in best["metrics"].items()})
        model_info = mlflow.sklearn.log_model(
            best["model"],
            artifact_path="selected_model",
            registered_model_name=MODEL_NAME,
            skops_trusted_types=[
                "numpy.dtype",
                "sklearn.tree._tree.Tree",
            ],
        )


    client = MlflowClient()
    # registered_model_version is returned by MLflow model logging when registration succeeds.
    version = getattr(model_info, "registered_model_version", None)
    if version is None:
        latest = client.get_latest_versions(MODEL_NAME)
        if not latest:
            raise RuntimeError("Model tersimpan tetapi tidak ditemukan di MLflow Model Registry.")
        version = latest[-1].version
    client.set_registered_model_alias(MODEL_NAME, "champion", str(version))
    print(f"Model teregistrasi: {MODEL_NAME}, version={version}, alias=champion")
    print(f"Artifact lokal untuk fallback tersimpan di {MODEL_PATH}")

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
