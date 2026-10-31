from __future__ import annotations

import argparse
import os
from pathlib import Path

from src.config import FLIGHTS_PATH, MODEL_PATH
from src.data_pipeline import run_pipeline
from src.ingest_documents import run_ingestion
from src.train_model import main as train_model


def main():
    parser = argparse.ArgumentParser(description="Bootstrap data, retrieval index, and ML model.")
    parser.add_argument("--refresh", action="store_true", help="Unduh ulang data BTS dan latih ulang model.")
    args = parser.parse_args()
    months = int(os.getenv("BTS_MONTHS", "3"))

    if args.refresh or not FLIGHTS_PATH.exists():
        run_pipeline(months=months, force=args.refresh)
    else:
        print(f"Menggunakan dataset bersih yang sudah ada: {FLIGHTS_PATH}")

    try:
        run_ingestion()
    except Exception as exc:
        # Data/model pipeline can still be tested when the official document site is temporarily unreachable.
        print(f"Peringatan: ingest dokumen tidak selesai: {exc}")

    if args.refresh or not MODEL_PATH.exists():
        train_model()
    else:
        print(f"Artifact model sudah ada: {MODEL_PATH}; training dilewati. Gunakan --refresh untuk melatih ulang.")

if __name__ == "__main__":
    main()
