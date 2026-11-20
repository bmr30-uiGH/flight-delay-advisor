from __future__ import annotations

from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
V3_INFERENCE_FEATURES_PATH = PROJECT_ROOT / "src" / "v3_inference_features.py"
V3_BUILDER_PATH = PROJECT_ROOT / "src" / "build_historical_feature_experiment_jan_aug_v3.py"
V3_HISTORICAL_DEMO_PREDICTOR_PATH = PROJECT_ROOT / "src" / "v3_historical_demo_predictor.py"
V3_MODEL_PATH = PROJECT_ROOT / "models" / "flight_delay_model_v3_candidate.joblib"
V3_HISTORY_PATH = PROJECT_ROOT / "data" / "processed" / "flights_historical_features_experiment_jan_aug_v3.parquet"

if not (V3_INFERENCE_FEATURES_PATH.is_file() and V3_BUILDER_PATH.is_file() and V3_HISTORICAL_DEMO_PREDICTOR_PATH.is_file() and V3_MODEL_PATH.is_file() and V3_HISTORY_PATH.is_file()):
    pytest.skip(
        "V3 experiment dependencies are optional and not included "
        "in the production clean clone.",
        allow_module_level=True,
    )

import pandas as pd

from src.v3_historical_demo_predictor import (
    HISTORY_PATH,
    predict_v3_historical_demo,
)


INPUT_COLUMNS = [
    "FL_DATE",
    "OP_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
]


def main() -> None:
    print("Membaca sampel historis Agustus 2026...")
    frame = pd.read_parquet(
        HISTORY_PATH,
        columns=INPUT_COLUMNS,
        filters=[("FL_DATE", "==", pd.Timestamp("2026-08-31"))],
    )
    frame = frame.dropna(subset=[
        "CRS_DEP_TIME",
        "CRS_ARR_TIME",
        "CRS_ELAPSED_TIME",
        "DISTANCE",
    ])
    if frame.empty:
        raise RuntimeError("Tidak ada sampel jadwal valid pada 31 Agustus 2026.")

    row = frame.iloc[0]
    inputs = row.to_dict()
    inputs["FL_DATE"] = pd.Timestamp(inputs["FL_DATE"]).date().isoformat()
    inputs["PREDICTION_MODE"] = "v3_candidate_historical"

    print("Menjalankan prediksi demonstrasi V3...")
    result = predict_v3_historical_demo(inputs)
    if not result.get("ok"):
        raise RuntimeError(result.get("message", "Smoke test V3 gagal."))

    assert result["model_source"] == "v3_candidate_historical_demo"
    assert result["features"] == 41
    assert 0.0 <= result["probability"] <= 1.0

    print("Status:", result["model_status"])
    print("Tanggal:", inputs["FL_DATE"])
    print("Probabilitas:", f"{result['probability']:.4f}")
    print("Threshold:", result["threshold"])
    print("Jumlah fitur:", result["features"])
    print("V3 HISTORICAL DEMO SMOKE TEST: PASSED")


if __name__ == "__main__":
    main()
