
from __future__ import annotations

import argparse
import os

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from src.config import (
    CHUNKS_PATH,
    FLIGHTS_PATH,
    MLFLOW_TRACKING_URI,
    MODEL_NAME,
)
from src.data_pipeline import run_pipeline
from src.ingest_documents import run_ingestion
from src.train_model import main as train_model


def has_usable_champion() -> bool:
    """Check whether the champion model exists and can be loaded from MLflow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    try:
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

        version = client.get_model_version_by_alias(
            MODEL_NAME,
            "champion",
        )

        model_uri = f"models:/{MODEL_NAME}@champion"

        # Verify that the actual model artifact can be loaded.
        mlflow.sklearn.load_model(model_uri)

        print(
            f"Model Registry siap: {MODEL_NAME}, "
            f"version={version.version}, alias=champion.",
            flush=True,
        )
        return True

    except Exception as exc:
        print(
            "Model champion belum tersedia atau tidak dapat dimuat "
            f"dari MLflow: {exc}",
            flush=True,
        )
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Bootstrap BTS data, document retrieval, and ML model."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Download BTS data again and retrain the model.",
    )

    args = parser.parse_args()
    months = int(os.getenv("BTS_MONTHS", "3"))

    # 1. Prepare the flight dataset.
    dataset_ready = (
        FLIGHTS_PATH.is_file()
        and FLIGHTS_PATH.stat().st_size > 0
    )

    if args.refresh or not dataset_ready:
        print("Dataset belum tersedia atau refresh diminta.", flush=True)
        run_pipeline(months=months, force=args.refresh)
    else:
        print(
            f"Menggunakan dataset bersih yang sudah tersedia: {FLIGHTS_PATH}",
            flush=True,
        )

    # 2. Build the Passenger Rights retrieval index.
    try:
        run_ingestion()
    except Exception as exc:
        if not CHUNKS_PATH.is_file() or CHUNKS_PATH.stat().st_size == 0:
            raise RuntimeError(
                "Ingestion dokumen gagal dan indeks retrieval belum tersedia."
            ) from exc

        print(
            "Peringatan: ingestion dokumen gagal; "
            f"menggunakan indeks sebelumnya: {exc}",
            flush=True,
        )

    if not CHUNKS_PATH.is_file() or CHUNKS_PATH.stat().st_size == 0:
        raise RuntimeError(
            f"Indeks Passenger Rights tidak tersedia: {CHUNKS_PATH}"
        )

    print(
        f"Indeks Passenger Rights siap: {CHUNKS_PATH}",
        flush=True,
    )

    # 3. Ensure MLflow contains a usable champion model.
    if args.refresh:
        print("Refresh diminta. Melatih ulang model...", flush=True)
        train_model()

    elif has_usable_champion():
        print(
            "Training dilewati karena model champion sudah tersedia.",
            flush=True,
        )

    else:
        print(
            "Model champion belum tersedia. "
            "Menjalankan training dan registrasi otomatis...",
            flush=True,
        )
        train_model()

    # 4. Verify registration after training.
    if not has_usable_champion():
        raise RuntimeError(
            "Bootstrap gagal: model champion belum dapat dimuat "
            "setelah proses training."
        )

    print(
        "BOOTSTRAP SELESAI: dataset, retrieval index, "
        "dan model MLflow siap digunakan.",
        flush=True,
    )


if __name__ == "__main__":
    main()
