from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.v3_inference_features import (
    FEATURE_COLUMNS,
    HISTORY_COLUMNS,
    build_v3_prediction_features,
    prepare_v3_history_reference,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "flight_delay_model_v3_candidate.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "flight_delay_model_v3_candidate.metadata.json"
HISTORY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "flights_historical_features_experiment_jan_aug_v3.parquet"
)
DEMO_START = pd.Timestamp("2026-08-01")
DEMO_END = pd.Timestamp("2026-08-31")


@lru_cache(maxsize=1)
def _load_assets(model_mtime: int, metadata_mtime: int, history_mtime: int):
    # Modification times are cache keys. Loading is repeated automatically if
    # any artifact is replaced. The reference is prepared once per app process.
    del model_mtime, metadata_mtime, history_mtime

    for path in (MODEL_PATH, METADATA_PATH, HISTORY_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"File V3 tidak ditemukan: {path}")

    with METADATA_PATH.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    model = joblib.load(MODEL_PATH)

    model_features = getattr(model, "feature_names_in_", None)
    if model_features is None:
        raise RuntimeError("Model V3 tidak memiliki feature_names_in_.")
    if list(model_features) != list(FEATURE_COLUMNS):
        raise RuntimeError("Urutan fitur model V3 tidak cocok dengan inference.")
    if 1 not in list(model.classes_):
        raise RuntimeError("Model V3 tidak memiliki kelas positif 1.")

    history = pd.read_parquet(HISTORY_PATH, columns=HISTORY_COLUMNS)
    history_reference = prepare_v3_history_reference(history)
    del history

    return model, metadata, history_reference


def _get_assets():
    for path in (MODEL_PATH, METADATA_PATH, HISTORY_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"File V3 tidak ditemukan: {path}")

    return _load_assets(
        MODEL_PATH.stat().st_mtime_ns,
        METADATA_PATH.stat().st_mtime_ns,
        HISTORY_PATH.stat().st_mtime_ns,
    )


def predict_v3_historical_demo(flight_inputs: dict | None) -> dict:
    """Run a past-only V3 replay for August 2026; never presented as live data."""
    try:
        if not flight_inputs:
            raise ValueError("Detail penerbangan belum disimpan.")

        prediction_date = pd.to_datetime(
            flight_inputs.get("FL_DATE"),
            errors="coerce",
        )
        if pd.isna(prediction_date):
            raise ValueError("Tanggal penerbangan V3 tidak valid.")
        prediction_date = prediction_date.normalize()

        if not (DEMO_START <= prediction_date <= DEMO_END):
            raise ValueError(
                "Mode V3 hanya tersedia untuk demonstrasi historis "
                "1–31 Agustus 2026."
            )

        required = (
            "OP_CARRIER",
            "ORIGIN",
            "DEST",
            "CRS_DEP_TIME",
            "CRS_ARR_TIME",
            "CRS_ELAPSED_TIME",
            "DISTANCE",
        )
        missing = [name for name in required if flight_inputs.get(name) is None]
        if missing:
            raise ValueError(f"Input V3 belum lengkap: {', '.join(missing)}")

        model, metadata, history_reference = _get_assets()

        features = build_v3_prediction_features(
            flight_inputs,
            history_reference,
        )

        if list(features.columns) != list(FEATURE_COLUMNS):
            raise RuntimeError("Urutan fitur hasil inference V3 tidak cocok.")
        if features.shape != (1, 41):
            raise RuntimeError(f"Jumlah fitur V3 tidak sesuai: {features.shape}.")

        probabilities = model.predict_proba(features)[0]
        positive_index = list(model.classes_).index(1)
        probability = float(probabilities[positive_index])

        if not np.isfinite(probability) or not 0 <= probability <= 1:
            raise RuntimeError("Model V3 menghasilkan probabilitas tidak valid.")

        threshold = float(metadata.get("threshold", 0.5))
        if not 0 <= threshold <= 1:
            raise RuntimeError("Threshold metadata V3 tidak valid.")

        predicted = probability >= threshold
        if abs(probability - threshold) <= 0.05:
            interpretation = (
                "Probabilitas berada dekat ambang klasifikasi. "
                "Hasil perlu ditafsirkan secara hati-hati."
            )
        elif predicted:
            interpretation = (
                "Probabilitas berada pada atau di atas ambang klasifikasi. "
                "Model mengklasifikasikan penerbangan sebagai berisiko delay."
            )
        else:
            interpretation = (
                "Probabilitas berada di bawah ambang klasifikasi. "
                "Model tidak mengklasifikasikan penerbangan sebagai delay, "
                "tetapi tidak menjamin penerbangan tepat waktu."
            )

        model_name = metadata.get(
            "model_name",
            "flight_delay_model_v3_candidate",
        )

        message = (
            "Model V3 mengklasifikasikan skenario ini sebagai berisiko "
            "mengalami keberangkatan terlambat minimal 15 menit."
            if predicted
            else "Model V3 tidak mengklasifikasikan skenario ini sebagai "
            "keberangkatan terlambat minimal 15 menit."
        )

        return {
            "ok": True,
            "model_name": model_name,
            "model_source": "v3_candidate_historical_demo",
            "model_status": metadata.get(
                "candidate_status",
                "development_candidate_not_production",
            ),
            "predicted_delay_15": bool(predicted),
            "probability": probability,
            "threshold": threshold,
            "interpretation": interpretation,
            "message": message,
            "features": len(FEATURE_COLUMNS),
            "caveat": (
                "DEMONSTRASI HISTORIS: model kandidat V3 diuji pada skenario "
                "periode validasi Agustus 2026. Ini bukan prediksi real-time, "
                "bukan status aktual penerbangan, dan bukan model Production. "
                "Fitur histori dihitung dari tanggal sebelum tanggal skenario."
            ),
        }

    except Exception as exc:
        return {
            "ok": False,
            "model_name": "flight_delay_model_v3_candidate",
            "model_source": "v3_candidate_historical_demo",
            "message": f"Demonstrasi V3 belum dapat dijalankan: {exc}",
        }
