
from __future__ import annotations

import pandas as pd

from src.predictor_router import predict_delay

from src.v3_predictor import HISTORY_PATH

from src.v3_inference_features import (
    PREDICTION_COLUMNS,
)


def main():

    print("=" * 75)
    print("PREDICTOR ROUTER INTEGRATION TEST")
    print("=" * 75)

    # --------------------------------------------------------
    # HISTORICAL DEMO REQUEST
    # --------------------------------------------------------

    print("\nREADING AUGUST VALIDATION SAMPLE...")

    sample_rows = pd.read_parquet(
        HISTORY_PATH,
        columns=PREDICTION_COLUMNS,
        filters=[
            ("FL_DATE", "==", pd.Timestamp("2026-08-31")),
        ],
    )

    if len(sample_rows) <= 100:
        raise RuntimeError(
            "Sampel penerbangan tidak tersedia."
        )

    sample = sample_rows.iloc[100]

    flight_inputs = {
        column: sample[column]
        for column in PREDICTION_COLUMNS
    }

    flight_inputs["PREDICTION_MODE"] = (
        "v3_candidate_historical"
    )

    print("Prediction date:", flight_inputs["FL_DATE"])
    print("Prediction mode:", flight_inputs["PREDICTION_MODE"])

    # --------------------------------------------------------
    # ROUTER TEST
    # --------------------------------------------------------

    print("\nRUNNING PREDICTION ROUTER...")

    result = predict_delay(
        flight_inputs
    )

    for key, value in result.items():
        print(f"{key}: {value}")

    assert result["ok"] is True
    assert result["model_source"] == "v3_candidate_historical"
    assert result["feature_count"] == 41
    assert 0 <= result["probability"] <= 1

    print("\nV3 ROUTING: PASSED")

    # --------------------------------------------------------
    # STALE HISTORY TEST
    # --------------------------------------------------------

    print("\nTESTING STALE HISTORY REJECTION...")

    stale_inputs = dict(flight_inputs)

    stale_inputs["FL_DATE"] = "2026-11-02"

    stale_result = predict_delay(
        stale_inputs
    )

    print(stale_result)

    assert stale_result["ok"] is False

    assert (
        "belum cukup terbaru"
        in stale_result["message"].lower()
    )

    print("\nSTALE HISTORY PROTECTION: PASSED")

    print("\n" + "=" * 75)
    print("PREDICTOR ROUTER TEST PASSED")
    print("=" * 75)


if __name__ == "__main__":
    main()
