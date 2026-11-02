import numpy as np
import pandas as pd

from src.historical_features import (
    MODEL_FEATURES,
    HISTORICAL_RATE_FEATURES,
    add_cyclical_features,
    add_past_only_historical_rates,
    prepare_prediction_features,
)


def make_history():
    return pd.DataFrame({
        "FL_DATE": [
            "2026-06-01",
            "2026-06-01",
            "2026-06-02",
            "2026-06-02",
            "2026-06-03",
        ],
        "OP_CARRIER": ["AA", "AA", "AA", "DL", "AA"],
        "ORIGIN": ["JFK", "JFK", "JFK", "ATL", "JFK"],
        "DEST": ["LAX", "LAX", "LAX", "JFK", "LAX"],
        "CRS_DEP_TIME": [800, 900, 1000, 1100, 1200],
        "MONTH": [6, 6, 6, 6, 6],
        "DAY_OF_WEEK": [2, 2, 3, 3, 4],
        "DISTANCE": [2475, 2475, 2475, 760, 2475],
        "DELAY_15": [1, 0, 1, 0, 1],
    })


def test_cyclical_features_have_expected_columns():
    frame = pd.DataFrame({
        "CRS_DEP_TIME": [2359, 1],
        "MONTH": [12, 1],
        "DAY_OF_WEEK": [7, 1],
    })

    result = add_cyclical_features(frame)

    expected = [
        "DEP_TIME_SIN",
        "DEP_TIME_COS",
        "DOW_SIN",
        "DOW_COS",
        "MONTH_SIN",
        "MONTH_COS",
    ]

    for column in expected:
        assert column in result.columns
        assert result[column].notna().all()


def test_cyclical_features_make_midnight_adjacent():
    frame = pd.DataFrame({
        "CRS_DEP_TIME": [2359, 1],
        "MONTH": [6, 6],
        "DAY_OF_WEEK": [3, 3],
    })

    result = add_cyclical_features(frame)

    first = result.loc[0, ["DEP_TIME_SIN", "DEP_TIME_COS"]].to_numpy(dtype=float)
    second = result.loc[1, ["DEP_TIME_SIN", "DEP_TIME_COS"]].to_numpy(dtype=float)

    assert np.linalg.norm(first - second) < 0.02


def test_training_historical_rates_exclude_same_day_targets():
    original = make_history()

    changed = original.copy()
    changed.loc[
        changed["FL_DATE"] == "2026-06-02",
        "DELAY_15"
    ] = [0, 1]

    first = add_past_only_historical_rates(original)
    second = add_past_only_historical_rates(changed)

    original_day = first.loc[
        first["FL_DATE"] == "2026-06-02",
        HISTORICAL_RATE_FEATURES
    ].reset_index(drop=True)

    changed_day = second.loc[
        second["FL_DATE"] == "2026-06-02",
        HISTORICAL_RATE_FEATURES
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        original_day,
        changed_day,
    )


def test_future_targets_do_not_change_earlier_historical_features():
    original = make_history()

    changed = original.copy()
    changed.loc[
        changed["FL_DATE"] == "2026-06-03",
        "DELAY_15"
    ] = 0

    first = add_past_only_historical_rates(original)
    second = add_past_only_historical_rates(changed)

    original_earlier = first.loc[
        first["FL_DATE"] < "2026-06-03",
        HISTORICAL_RATE_FEATURES
    ].reset_index(drop=True)

    changed_earlier = second.loc[
        second["FL_DATE"] < "2026-06-03",
        HISTORICAL_RATE_FEATURES
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        original_earlier,
        changed_earlier,
    )


def test_historical_rates_are_between_zero_and_one():
    result = add_past_only_historical_rates(
        make_history()
    )

    rates = result[HISTORICAL_RATE_FEATURES]

    assert rates.notna().all().all()
    assert rates.ge(0).all().all()
    assert rates.le(1).all().all()


def test_prediction_features_exclude_same_day_history():
    history = make_history()

    flight = {
        "OP_CARRIER": "AA",
        "ORIGIN": "JFK",
        "DEST": "LAX",
        "CRS_DEP_TIME": 1400,
        "MONTH": 6,
        "DAY_OF_WEEK": 5,
        "DISTANCE": 2475,
        "FL_DATE": "2026-06-03",
    }

    changed = history.copy()
    changed.loc[
        changed["FL_DATE"] == "2026-06-03",
        "DELAY_15"
    ] = 0

    first = prepare_prediction_features(
        flight, history
    )
    second = prepare_prediction_features(
        flight, changed
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_prediction_feature_schema_is_exact():
    history = make_history()

    flight = {
        "OP_CARRIER": "AA",
        "ORIGIN": "JFK",
        "DEST": "LAX",
        "CRS_DEP_TIME": 1649,
        "MONTH": 6,
        "DAY_OF_WEEK": 5,
        "DISTANCE": 2475,
        "FL_DATE": "2026-06-04",
    }

    result = prepare_prediction_features(
        flight, history
    )

    assert list(result.columns) == MODEL_FEATURES
    assert "FL_DATE" not in result.columns
    assert "DELAY_15" not in result.columns
    assert result.shape == (1, len(MODEL_FEATURES))