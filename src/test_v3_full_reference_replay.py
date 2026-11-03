
from __future__ import annotations

import time

import joblib
import numpy as np
import pandas as pd

from src.v3_inference_features import (
    build_v3_prediction_features,
    prepare_v3_history_reference,
    FEATURE_COLUMNS,
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    HISTORY_COLUMNS,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = (
    "data/processed/"
    "flights_historical_features_experiment_jan_aug_v3.parquet"
)

MODEL_PATH = (
    "models/flight_delay_model_v3_candidate.joblib"
)

TEST_DATES = [
    "2026-01-02",
    "2026-01-15",
    "2026-02-28",
    "2026-03-31",
    "2026-06-30",
    "2026-08-31",
]

TARGET_INDEX = 100
THRESHOLD = 0.50

TOLERANCE = 2e-6
PROBABILITY_TOLERANCE = 1e-9

SOURCE_COLUMNS = [
    "FL_DATE",
    "OP_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
    "DELAY_15",
]


# ============================================================
# TEST ONE FLIGHT USING FULL REFERENCE
# ============================================================

def test_one_flight(
    test_date,
    history_reference,
    model,
):

    date = pd.Timestamp(test_date)

    print("\n" + "=" * 75)
    print("TEST DATE:", date.date())
    print("=" * 75)

    print(
        "Reference last date:",
        history_reference.last_date.date(),
    )

    print(
        "Future dates exist in reference:",
        history_reference.last_date > date,
    )

    target_columns = list(dict.fromkeys(
        SOURCE_COLUMNS + FEATURE_COLUMNS
    ))

    target_rows = pd.read_parquet(
        DATA_PATH,
        columns=target_columns,
        filters=[
            ("FL_DATE", "==", date),
        ],
    )

    if target_rows.empty:
        raise RuntimeError(
            f"Tidak ada penerbangan pada {test_date}."
        )

    if len(target_rows) <= TARGET_INDEX:
        raise RuntimeError(
            f"Target index tidak tersedia pada {test_date}."
        )

    target = target_rows.iloc[TARGET_INDEX]

    flight_inputs = {
        column: target[column]
        for column in SOURCE_COLUMNS
        if column != "DELAY_15"
    }

    # --------------------------------------------------------
    # GENERATE FEATURES FROM FULL REFERENCE
    # --------------------------------------------------------

    started = time.perf_counter()

    actual_features = build_v3_prediction_features(
        flight_inputs,
        history_reference,
    )

    inference_seconds = (
        time.perf_counter() - started
    )

    # --------------------------------------------------------
    # LOAD FEATURES ALREADY STORED IN PARQUET
    # --------------------------------------------------------

    expected_features = pd.DataFrame([
        {
            column: target[column]
            for column in FEATURE_COLUMNS
        }
    ])

    for column in CATEGORICAL_FEATURES:
        expected_features[column] = (
            expected_features[column].astype(str)
        )

    for column in NUMERIC_FEATURES:
        expected_features[column] = pd.to_numeric(
            expected_features[column],
            errors="coerce",
        )

    # --------------------------------------------------------
    # COMPARE FEATURES
    # --------------------------------------------------------

    categorical_mismatches = []

    for column in CATEGORICAL_FEATURES:

        actual_value = str(
            actual_features.iloc[0][column]
        ).strip().upper()

        expected_value = str(
            expected_features.iloc[0][column]
        ).strip().upper()

        if actual_value != expected_value:
            categorical_mismatches.append(column)

    actual_numeric = actual_features[
        NUMERIC_FEATURES
    ].to_numpy(dtype=float)

    expected_numeric = expected_features[
        NUMERIC_FEATURES
    ].to_numpy(dtype=float)

    if not np.isfinite(expected_numeric).all():
        raise RuntimeError(
            "Expected features mengandung nilai tidak valid."
        )

    differences = np.abs(
        actual_numeric - expected_numeric
    )

    numeric_matches = np.isclose(
        actual_numeric,
        expected_numeric,
        atol=TOLERANCE,
        rtol=1e-6,
    )

    numeric_mismatches = [
        NUMERIC_FEATURES[i]
        for i in range(len(NUMERIC_FEATURES))
        if not numeric_matches[0, i]
    ]

    max_difference = float(differences.max())

    # --------------------------------------------------------
    # MODEL PREDICTION PARITY
    # --------------------------------------------------------

    positive_index = list(
        model.classes_
    ).index(1)

    actual_probability = float(
        model.predict_proba(
            actual_features
        )[0, positive_index]
    )

    expected_probability = float(
        model.predict_proba(
            expected_features
        )[0, positive_index]
    )

    probability_difference = abs(
        actual_probability - expected_probability
    )

    actual_prediction = int(
        actual_probability >= THRESHOLD
    )

    expected_prediction = int(
        expected_probability >= THRESHOLD
    )

    probability_matches = np.isclose(
        actual_probability,
        expected_probability,
        atol=PROBABILITY_TOLERANCE,
        rtol=PROBABILITY_TOLERANCE,
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print("\nFEATURE COMPARISON")
    print("Feature count:", len(FEATURE_COLUMNS))
    print("Categorical mismatches:", len(categorical_mismatches))
    print("Numeric mismatches:", len(numeric_mismatches))
    print("Maximum numeric difference:", max_difference)

    print("\nPREDICTION COMPARISON")
    print("Inference probability:", actual_probability)
    print("Stored probability:", expected_probability)
    print("Probability difference:", probability_difference)
    print("Inference prediction:", actual_prediction)
    print("Stored prediction:", expected_prediction)
    print("Actual target:", int(target["DELAY_15"]))

    print(
        "Feature generation time:",
        round(inference_seconds * 1000, 3),
        "ms",
    )

    passed = (
        not categorical_mismatches
        and not numeric_mismatches
        and max_difference <= TOLERANCE
        and probability_matches
        and actual_prediction == expected_prediction
    )

    print(
        "\nFULL REFERENCE RESULT:",
        "PASSED" if passed else "FAILED",
    )

    if categorical_mismatches:
        print("Categorical mismatch details:", categorical_mismatches)

    if numeric_mismatches:
        print("Numeric mismatch details:", numeric_mismatches)

    if not passed:
        raise AssertionError(
            f"Full reference replay gagal pada {test_date}."
        )

    return {
        "date": test_date,
        "status": "PASSED",
        "feature_mismatches": (
            len(categorical_mismatches)
            + len(numeric_mismatches)
        ),
        "max_feature_difference": max_difference,
        "probability_difference": probability_difference,
        "inference_ms": inference_seconds * 1000,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 75)
    print("V3 FULL HISTORY REFERENCE REPLAY")
    print("=" * 75)

    # --------------------------------------------------------
    # LOAD FULL HISTORY ONCE
    # --------------------------------------------------------

    print("\nLOADING COMPLETE HISTORICAL DATA...")

    started = time.perf_counter()

    history = pd.read_parquet(
        DATA_PATH,
        columns=HISTORY_COLUMNS,
    )

    loading_seconds = (
        time.perf_counter() - started
    )

    print("Historical rows:", len(history))
    print("History load seconds:", round(loading_seconds, 2))

    # --------------------------------------------------------
    # BUILD ONE REFERENCE
    # --------------------------------------------------------

    print("\nBUILDING ONE FULL HISTORY REFERENCE...")

    started = time.perf_counter()

    history_reference = prepare_v3_history_reference(
        history
    )

    preparation_seconds = (
        time.perf_counter() - started
    )

    print(
        "Reference preparation seconds:",
        round(preparation_seconds, 2),
    )

    print("Reference rows:", history_reference.total_rows)
    print("Reference first date:", history_reference.first_date)
    print("Reference last date:", history_reference.last_date)

    # --------------------------------------------------------
    # LOAD MODEL ONCE
    # --------------------------------------------------------

    print("\nLOADING V3 MODEL...")

    model = joblib.load(MODEL_PATH)

    if len(FEATURE_COLUMNS) != 41:
        raise RuntimeError("Jumlah fitur bukan 41.")

    if not hasattr(model, "predict_proba"):
        raise RuntimeError(
            "Model tidak mendukung predict_proba."
        )

    print("Model loaded:", type(model).__name__)

    # --------------------------------------------------------
    # TEST ALL DATES USING THE SAME REFERENCE
    # --------------------------------------------------------

    results = []

    for test_date in TEST_DATES:

        result = test_one_flight(
            test_date,
            history_reference,
            model,
        )

        results.append(result)

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 75)
    print("FULL REFERENCE TEST SUMMARY")
    print("=" * 75)

    summary = pd.DataFrame(results)

    print(summary.to_string(index=False))

    print("\nTOTAL HISTORY LOAD TIME:")
    print(round(loading_seconds, 2), "seconds")

    print("\nTOTAL REFERENCE PREPARATION TIME:")
    print(round(preparation_seconds, 2), "seconds")

    print("\nAVERAGE FEATURE GENERATION TIME:")
    print(
        round(
            summary["inference_ms"].mean(),
            3,
        ),
        "milliseconds",
    )

    assert all(
        result["status"] == "PASSED"
        for result in results
    )

    print("\nALL FULL REFERENCE TESTS PASSED.")


if __name__ == "__main__":
    main()
