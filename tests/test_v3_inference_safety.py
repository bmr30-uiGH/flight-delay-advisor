import joblib
import numpy as np
import pandas as pd
import pytest

from src.v3_inference_features import (
    build_v3_prediction_features,
    HISTORY_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    HISTORICAL_RATE_FEATURES,
    ROLLING_FEATURES,
)


MODEL_PATH = "models/flight_delay_model_v3_candidate.joblib"

PREDICTION_DATE = pd.Timestamp("2026-03-15")

def make_flight(**overrides):
    result = {
        "FL_DATE": "2026-03-15",
        "OP_CARRIER": "DL",
        "ORIGIN": "AAA",
        "DEST": "BBB",
        "CRS_DEP_TIME": 900,
        "CRS_ARR_TIME": 1200,
        "CRS_ELAPSED_TIME": 180,
        "DISTANCE": 100.0,
    }
    result.update(overrides)
    return result


def make_history_row(
    date,
    delay,
    carrier="DL",
    origin="AAA",
    destination="BBB",
    departure=900,
):
    return {
        "FL_DATE": date,
        "OP_CARRIER": carrier,
        "ORIGIN": origin,
        "DEST": destination,
        "CRS_DEP_TIME": departure,
        "DELAY_15": delay,
    }


def make_history(*rows):
    return pd.DataFrame(rows, columns=HISTORY_COLUMNS)


def numeric_features(features):
    return features[NUMERIC_FEATURES].to_numpy(dtype=float)


def test_rolling_window_includes_d7_and_excludes_d8():
    d7 = make_history_row("2026-03-08", 1)
    d8 = make_history_row("2026-03-07", 0)

    both = build_v3_prediction_features(
        make_flight(),
        make_history(d7, d8),
    )
    only_d7 = build_v3_prediction_features(
        make_flight(),
        make_history(d7),
    )
    only_d8 = build_v3_prediction_features(
        make_flight(),
        make_history(d8),
    )

    feature = "ORIGIN_ROLLING_7D_DELAY_RATE"

    both_rate = float(both.iloc[0][feature])
    d7_rate = float(only_d7.iloc[0][feature])
    d8_rate = float(only_d8.iloc[0][feature])

    assert np.isclose(both_rate, d7_rate, atol=1e-12)
    assert np.isclose(both_rate, 1.0, atol=1e-12)
    assert np.isclose(d8_rate, 0.50, atol=1e-12)


def test_same_day_and_future_rows_do_not_change_features():
    previous = make_history_row("2026-03-14", 0)

    same_day = make_history_row("2026-03-15", 1)
    future_1 = make_history_row("2026-03-16", 1)
    future_2 = make_history_row("2026-04-01", 1)

    baseline = build_v3_prediction_features(
        make_flight(),
        make_history(previous),
    )

    contaminated = build_v3_prediction_features(
        make_flight(),
        make_history(previous, same_day, future_1, future_2),
    )

    for column in CATEGORICAL_FEATURES:
        assert baseline.iloc[0][column] == contaminated.iloc[0][column]

    np.testing.assert_allclose(
        numeric_features(baseline),
        numeric_features(contaminated),
        atol=1e-12,
        rtol=1e-12,
    )


def test_future_only_history_uses_global_default_050():
    future = make_history_row("2026-03-16", 1)

    features = build_v3_prediction_features(
        make_flight(),
        make_history(future),
    )

    historical_columns = (
        HISTORICAL_RATE_FEATURES + ROLLING_FEATURES
    )

    values = features.iloc[0][historical_columns].to_numpy(
        dtype=float
    )

    assert np.isfinite(values).all()
    np.testing.assert_allclose(
        values,
        np.full(len(historical_columns), 0.50),
        atol=1e-12,
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("CRS_DEP_TIME", 2460),
        ("CRS_DEP_TIME", 1260),
        ("CRS_ARR_TIME", 2460),
        ("CRS_ARR_TIME", 1260),
        ("CRS_ELAPSED_TIME", 0),
        ("CRS_ELAPSED_TIME", -10),
        ("DISTANCE", 0),
        ("DISTANCE", -100),
        ("FL_DATE", "invalid-date"),
    ],
)
def test_invalid_flight_inputs_are_rejected(field, invalid_value):
    future = make_history_row("2026-03-16", 1)
    flight = make_flight(**{field: invalid_value})

    with pytest.raises(ValueError):
        build_v3_prediction_features(
            flight,
            make_history(future),
        )


def test_unseen_categories_work_with_v3_model_encoder():
    future = make_history_row(
        "2026-03-16",
        1,
        carrier="DL",
        origin="AAA",
        destination="BBB",
    )

    flight = make_flight(
        OP_CARRIER="QX",
        ORIGIN="ZZZ",
        DEST="YYY",
    )

    features = build_v3_prediction_features(
        flight,
        make_history(future),
    )

    assert features.shape == (1, 41)
    assert features.iloc[0]["OP_CARRIER"] == "QX"
    assert features.iloc[0]["ORIGIN"] == "ZZZ"
    assert features.iloc[0]["DEST"] == "YYY"

    model = joblib.load(MODEL_PATH)

    assert hasattr(model, "predict_proba")

    probabilities = model.predict_proba(features)

    assert probabilities.shape[0] == 1
    assert np.isfinite(probabilities).all()
    assert np.isclose(probabilities.sum(), 1.0, atol=1e-8)
