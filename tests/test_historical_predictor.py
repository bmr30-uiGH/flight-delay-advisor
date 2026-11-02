
import joblib
import numpy as np
import pandas as pd

from src.config import FEATURES, TARGET
from src.historical_features import MODEL_FEATURES
from src import predictor


class FakeHistoricalModel:
    classes_ = np.array([0, 1])
    feature_names_in_ = np.array(MODEL_FEATURES, dtype=object)

    def __init__(self, probability=0.70):
        self.probability = probability
        self.last_features = None

    def predict_proba(self, frame):
        self.last_features = frame.copy()
        return np.array([[1.0 - self.probability, self.probability]])


class FakeLegacyModel:
    classes_ = np.array([0, 1])

    def __init__(self):
        self.last_features = None

    def predict_proba(self, frame):
        self.last_features = frame.copy()
        return np.array([[0.40, 0.60]])


def make_flight_inputs(prediction_date="2026-09-01"):
    date_value = pd.Timestamp(prediction_date)

    return {
        "OP_CARRIER": "AA",
        "ORIGIN": "JFK",
        "DEST": "LAX",
        "CRS_DEP_TIME": 900,
        "MONTH": date_value.month,
        "DAY_OF_WEEK": ((date_value.weekday() + 1) % 7) + 1,
        "DISTANCE": 2475.0,
        "FL_DATE": date_value.date().isoformat(),
    }


def make_history():
    # The first row is from the prior day.
    # The second row is on the same day as prediction and must be excluded.
    return pd.DataFrame([
        {
            "OP_CARRIER": "AA",
            "ORIGIN": "JFK",
            "DEST": "LAX",
            "CRS_DEP_TIME": 900,
            "MONTH": 8,
            "DAY_OF_WEEK": 2,
            "DISTANCE": 2475.0,
            TARGET: 0,
            "FL_DATE": "2026-08-31",
        },
        {
            "OP_CARRIER": "AA",
            "ORIGIN": "JFK",
            "DEST": "LAX",
            "CRS_DEP_TIME": 900,
            "MONTH": 9,
            "DAY_OF_WEEK": 3,
            "DISTANCE": 2475.0,
            TARGET: 1,
            "FL_DATE": "2026-09-01",
        },
    ])


def test_load_historical_candidate_bundle(tmp_path, monkeypatch):
    artifact_path = tmp_path / "historical_candidate.joblib"

    joblib.dump(
        {
            "model": FakeHistoricalModel(),
            "threshold": 0.50,
            "model_name": "test_historical_model",
            "feature_schema": list(MODEL_FEATURES),
            "metadata": {},
        },
        artifact_path,
    )

    monkeypatch.setattr(
        predictor,
        "HISTORICAL_CANDIDATE_MODEL_PATH",
        artifact_path,
    )
    monkeypatch.setenv("MODEL_SOURCE", "historical_candidate")

    predictor.load_model_bundle.cache_clear()

    try:
        bundle = predictor.load_model_bundle()

        assert bundle["model_source"] == "historical_candidate"
        assert bundle["requires_historical_features"] is True
        assert bundle["threshold"] == 0.50
        assert bundle["model_name"] == "test_historical_model"

    finally:
        predictor.load_model_bundle.cache_clear()


def test_historical_prediction_uses_15_features_and_past_only_rates(
    monkeypatch,
):
    model = FakeHistoricalModel(probability=0.70)

    monkeypatch.setattr(
        predictor,
        "load_model_bundle",
        lambda: {
            "model": model,
            "threshold": 0.50,
            "model_name": "test_historical_model",
            "model_source": "historical_candidate",
            "requires_historical_features": True,
        },
    )

    monkeypatch.setattr(
        predictor,
        "load_historical_reference",
        make_history,
    )

    result = predictor.predict_delay(make_flight_inputs())

    assert result["ok"] is True
    assert result["predicted_delay_15"] == 1
    assert result["probability"] == 0.70
    assert list(model.last_features.columns) == list(MODEL_FEATURES)

    # The same-day positive row is excluded.
    # The only prior origin record had target 0.
    assert model.last_features[
        "ORIGIN_HIST_DELAY_RATE"
    ].iloc[0] < 0.01


def test_historical_prediction_requires_flight_date(monkeypatch):
    monkeypatch.setattr(
        predictor,
        "load_model_bundle",
        lambda: {
            "model": FakeHistoricalModel(),
            "threshold": 0.50,
            "model_name": "test_historical_model",
            "model_source": "historical_candidate",
            "requires_historical_features": True,
        },
    )

    inputs = make_flight_inputs()
    inputs.pop("FL_DATE")

    result = predictor.predict_delay(inputs)

    assert result["ok"] is False
    assert "FL_DATE" in result["message"]


def test_historical_prediction_rejects_inconsistent_calendar_inputs(
    monkeypatch,
):
    monkeypatch.setattr(
        predictor,
        "load_model_bundle",
        lambda: {
            "model": FakeHistoricalModel(),
            "threshold": 0.50,
            "model_name": "test_historical_model",
            "model_source": "historical_candidate",
            "requires_historical_features": True,
        },
    )

    inputs = make_flight_inputs()
    inputs["MONTH"] = 8

    result = predictor.predict_delay(inputs)

    assert result["ok"] is False
    assert "MONTH" in result["message"]


def test_legacy_model_still_uses_original_seven_features(monkeypatch):
    model = FakeLegacyModel()

    monkeypatch.setattr(
        predictor,
        "load_model_bundle",
        lambda: {
            "model": model,
            "threshold": 0.50,
            "model_name": "legacy_model",
            "model_source": "mlflow_champion",
        },
    )

    inputs = make_flight_inputs()
    inputs.pop("FL_DATE")

    result = predictor.predict_delay(inputs)

    assert result["ok"] is True
    assert result["probability"] == 0.60
    assert list(model.last_features.columns) == list(FEATURES)
