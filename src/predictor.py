from __future__ import annotations

import os
from functools import lru_cache

import joblib
import pandas as pd

from src.config import FEATURES, MODEL_NAME, MODEL_PATH, MLFLOW_TRACKING_URI


@lru_cache(maxsize=1)
def load_trained_model():
    """Load an explicit local artifact on Streamlit Cloud, otherwise prefer MLflow Registry."""
    model_source = os.getenv("MODEL_SOURCE", "mlflow").strip().lower()
    if model_source in {"joblib", "local", "file"}:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model lokal tidak ditemukan: {MODEL_PATH}")
        return joblib.load(MODEL_PATH)

    os.environ.setdefault("MLFLOW_TRACKING_URI", MLFLOW_TRACKING_URI)
    try:
        import mlflow.sklearn
        return mlflow.sklearn.load_model(f"models:/{MODEL_NAME}@champion")
    except Exception as registry_error:
        if MODEL_PATH.exists():
            print(f"Peringatan: model MLflow Registry belum bisa dimuat ({registry_error}); memakai artifact lokal.")
            return joblib.load(MODEL_PATH)
        raise RuntimeError("Model belum tersedia. Bootstrap/training harus berhasil terlebih dahulu.") from registry_error


def _interpret_probability(predicted: int, probability: float | None) -> str:
    if probability is None:
        return "Probabilitas tidak tersedia; label model tidak boleh ditafsirkan sebagai kepastian kondisi penerbangan."
    threshold = 0.5
    if abs(probability - threshold) <= 0.05:
        return (
            "Probabilitas berada dekat ambang klasifikasi 50%. Label kelas hanya menunjukkan "
            "hasil keputusan model pada ambang tersebut dan bukan bukti bahwa penerbangan akan tepat waktu."
        )
    if predicted == 1:
        return "Probabilitas berada di atas ambang klasifikasi 50%; ini tetap estimasi historis, bukan kepastian delay."
    return "Probabilitas berada di bawah ambang klasifikasi 50%; ini bukan jaminan bahwa penerbangan akan tepat waktu."

def predict_delay(flight_inputs: dict | None) -> dict:
    required = FEATURES
    if not flight_inputs:
        return {"ok": False, "message": "Detail penerbangan belum disimpan. Isi formulir pada sidebar terlebih dahulu."}
    missing = [name for name in required if name not in flight_inputs or flight_inputs[name] in (None, "")]
    if missing:
        return {"ok": False, "message": "Input belum lengkap: " + ", ".join(missing)}
    row = {name: flight_inputs[name] for name in required}
    for field in ["CRS_DEP_TIME", "MONTH", "DAY_OF_WEEK", "DISTANCE"]:
        try:
            row[field] = float(row[field])
        except (TypeError, ValueError):
            return {"ok": False, "message": f"Nilai {field} tidak valid."}
    for field in ["OP_CARRIER", "ORIGIN", "DEST"]:
        row[field] = str(row[field]).strip().upper()
        if not row[field]:
            return {"ok": False, "message": f"Nilai {field} tidak boleh kosong."}
    try:
        model = load_trained_model()
        frame = pd.DataFrame([row], columns=FEATURES)
        predicted = int(model.predict(frame)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(frame)[0]
            classes = list(model.classes_)
            if 1 in classes:
                probability = float(probabilities[classes.index(1)])
        return {
            "ok": True,
            "predicted_delay_15": predicted,
            "probability": probability,
            "interpretation": _interpret_probability(predicted, probability),
            "message": "Model memperkirakan risiko keberangkatan terlambat minimal 15 menit." if predicted else "Model tidak mengklasifikasikan penerbangan ini sebagai delay ≥15 menit.",
            "features": row,
            "caveat": "Estimasi berbasis pola historis, bukan status real-time dan bukan jaminan hasil aktual.",
        }
    except Exception as exc:
        return {"ok": False, "message": f"Prediksi belum tersedia: {exc}"}
