from __future__ import annotations

from src.predictor import predict_delay as predict_production_delay
from src.v3_historical_demo_predictor import predict_v3_historical_demo


V3_HISTORICAL_MODE = "v3_candidate_historical"


def predict_delay(flight_inputs: dict | None) -> dict:
    """Preserve the Production predictor and route only explicit V3 demo requests."""
    if not flight_inputs:
        return predict_production_delay(flight_inputs)

    mode = str(flight_inputs.get("PREDICTION_MODE", "production")).strip().lower()
    if mode == V3_HISTORICAL_MODE:
        return predict_v3_historical_demo(flight_inputs)

    # Default and unknown values deliberately fall back to the existing model.
    return predict_production_delay(flight_inputs)
