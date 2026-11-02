
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.historical_features import (
    CATEGORICAL_FEATURES,
    HISTORY_GROUPS,
    HISTORICAL_RATE_FEATURES,
    add_cyclical_features,
)

from src.build_historical_feature_experiment_jan_aug_v3 import (
    WINDOWS,
    ROLLING_GROUPS,
    ROLLING_FEATURES,
    add_arrival_cyclical_features,
    add_day_of_month_cyclical_features,
    add_departure_hour,
)


# ============================================================
# FEATURE SCHEMA
# ============================================================

SCHEDULED_NUMERIC_FEATURES = [
    "DEP_TIME_SIN",
    "DEP_TIME_COS",
    "ARR_TIME_SIN",
    "ARR_TIME_COS",
    "CRS_ELAPSED_TIME",
    "DOW_SIN",
    "DOW_COS",
    "MONTH_SIN",
    "MONTH_COS",
    "DAY_OF_MONTH_SIN",
    "DAY_OF_MONTH_COS",
    "DISTANCE",
]

NUMERIC_FEATURES = (
    SCHEDULED_NUMERIC_FEATURES
    + HISTORICAL_RATE_FEATURES
    + ROLLING_FEATURES
)

FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)

HISTORY_COLUMNS = [
    "FL_DATE",
    "OP_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "DELAY_15",
]

PREDICTION_COLUMNS = [
    "FL_DATE",
    "OP_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "CRS_ARR_TIME",
    "CRS_ELAPSED_TIME",
    "DISTANCE",
]

if len(FEATURE_COLUMNS) != 41:
    raise RuntimeError(
        f"Schema V3 harus 41 fitur, ditemukan {len(FEATURE_COLUMNS)}."
    )


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass(frozen=True)
class GroupHistoryIndex:
    days: np.ndarray
    delay_prefix: np.ndarray
    count_prefix: np.ndarray


@dataclass(frozen=True)
class V3HistoryReference:
    group_indexes: dict
    global_days: np.ndarray
    global_delay_prefix: np.ndarray
    global_count_prefix: np.ndarray
    total_rows: int
    first_date: pd.Timestamp
    last_date: pd.Timestamp


# ============================================================
# INPUT VALIDATION
# ============================================================

def _require_columns(frame, required, name):

    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom {name} tidak tersedia: {missing}"
        )


def _normalize_category(value, name):

    if value is None or pd.isna(value):
        raise ValueError(
            f"Input {name} tidak boleh kosong."
        )

    result = str(value).strip().upper()

    if not result:
        raise ValueError(
            f"Input {name} tidak boleh kosong."
        )

    return result


def _numeric_input(value, name):

    result = pd.to_numeric(
        pd.Series([value]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(result) or not np.isfinite(float(result)):
        raise ValueError(
            f"Input numerik {name} tidak valid."
        )

    return float(result)


# ============================================================
# NORMALIZE HISTORICAL DATA
# ============================================================

def _normalize_history(history_frame):

    _require_columns(
        history_frame,
        HISTORY_COLUMNS,
        "historical data",
    )

    history = history_frame[
        HISTORY_COLUMNS
    ].copy()

    history["FL_DATE"] = pd.to_datetime(
        history["FL_DATE"],
        errors="coerce",
    ).dt.normalize()

    history["DELAY_15"] = pd.to_numeric(
        history["DELAY_15"],
        errors="coerce",
    )

    history["CRS_DEP_TIME"] = pd.to_numeric(
        history["CRS_DEP_TIME"],
        errors="coerce",
    )

    for column in CATEGORICAL_FEATURES:

        history[column] = (
            history[column]
            .astype("string")
            .str.strip()
            .str.upper()
        )

    history = history.dropna(
        subset=HISTORY_COLUMNS
    ).copy()

    if history.empty:
        raise ValueError(
            "Historical data tidak memiliki baris valid."
        )

    if not history["DELAY_15"].isin([0, 1]).all():
        raise ValueError(
            "DELAY_15 hanya boleh bernilai 0 atau 1."
        )

    departure = history["CRS_DEP_TIME"].to_numpy(
        dtype=float
    )

    hours = np.floor(departure / 100)
    minutes = departure % 100

    valid_time = (
        np.isfinite(departure)
        & (departure >= 0)
        & (departure <= 2359)
        & (hours <= 23)
        & (minutes >= 0)
        & (minutes < 60)
        & (departure == np.floor(departure))
    )

    if not valid_time.all():
        raise ValueError(
            "CRS_DEP_TIME pada historical data tidak valid."
        )

    history["_DEP_HOUR"] = hours.astype("int8")
    history["_DAY"] = history["FL_DATE"]

    return history


# ============================================================
# BUILD CUMULATIVE ARRAYS
# ============================================================

def _build_prefix_arrays(grouped):

    days = grouped["_DAY"].to_numpy(
        dtype="datetime64[ns]"
    )

    delay_values = grouped["sum"].to_numpy(
        dtype=np.int64
    )

    count_values = grouped["count"].to_numpy(
        dtype=np.int64
    )

    delay_prefix = np.concatenate([
        np.array([0], dtype=np.int64),
        np.cumsum(delay_values, dtype=np.int64),
    ])

    count_prefix = np.concatenate([
        np.array([0], dtype=np.int64),
        np.cumsum(count_values, dtype=np.int64),
    ])

    return GroupHistoryIndex(
        days=days,
        delay_prefix=delay_prefix,
        count_prefix=count_prefix,
    )


# ============================================================
# BUILD GROUP HISTORY INDEX
# ============================================================

def _build_group_history_index(
    history,
    keys,
):

    grouped_daily = (
        history.groupby(
            keys + ["_DAY"],
            observed=True,
            sort=False,
        )["DELAY_15"]
        .agg(["sum", "count"])
        .reset_index()
    )

    grouped_daily = grouped_daily.sort_values(
        keys + ["_DAY"],
        kind="stable",
    ).reset_index(drop=True)

    group_indexes = {}

    if len(keys) == 1:

        iterator = grouped_daily.groupby(
            keys[0],
            observed=True,
            sort=False,
        )

    else:

        iterator = grouped_daily.groupby(
            keys,
            observed=True,
            sort=False,
        )

    for raw_key, group in iterator:

        key = (
            raw_key
            if len(keys) == 1
            else tuple(raw_key)
        )

        group = group.sort_values(
            "_DAY",
            kind="stable",
        )

        group_indexes[key] = _build_prefix_arrays(
            group
        )

    return group_indexes


# ============================================================
# PREPARE HISTORICAL REFERENCE ONCE
# ============================================================

def prepare_v3_history_reference(
    history_frame: pd.DataFrame,
) -> V3HistoryReference:
    """
    Normalize and aggregate the historical dataset once.

    Reusable for multiple prediction requests.

    Group indexes contain daily cumulative:
    - Delay counts
    - Flight counts

    This avoids repeatedly scanning millions of flight rows.
    """

    print("\nPREPARING V3 HISTORY REFERENCE...")

    history = _normalize_history(history_frame)

    first_date = history["_DAY"].min()
    last_date = history["_DAY"].max()

    total_rows = len(history)

    print("Historical rows:", total_rows)
    print("First date:", first_date.date())
    print("Last date:", last_date.date())

    # --------------------------------------------------------
    # GLOBAL DAILY HISTORY
    # --------------------------------------------------------

    print("Building global daily index...")

    global_daily = (
        history.groupby(
            "_DAY",
            sort=True,
        )["DELAY_15"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values("_DAY")
        .reset_index(drop=True)
    )

    global_index = _build_prefix_arrays(
        global_daily
    )

    # --------------------------------------------------------
    # GROUP DAILY HISTORY
    # --------------------------------------------------------

    group_indexes = {}

    for keys, prefix, _ in ROLLING_GROUPS:

        print("Building group index:", prefix)

        group_indexes[prefix] = (
            _build_group_history_index(
                history,
                keys,
            )
        )

    # The raw flight-level DataFrame is deliberately not
    # retained after aggregation.

    reference = V3HistoryReference(
        group_indexes=group_indexes,
        global_days=global_index.days,
        global_delay_prefix=global_index.delay_prefix,
        global_count_prefix=global_index.count_prefix,
        total_rows=total_rows,
        first_date=first_date,
        last_date=last_date,
    )

    del history

    print("V3 history reference ready.")
    print("Indexed groups:", len(group_indexes))

    return reference


# ============================================================
# FAST RANGE QUERY
# ============================================================

def _query_prefix_arrays(
    days,
    delay_prefix,
    count_prefix,
    start_date,
    end_date,
):
    """
    Return aggregates for [start_date, end_date).

    None start_date means all available dates before end_date.
    """

    end_value = np.datetime64(
        pd.Timestamp(end_date).normalize()
    )

    if start_date is None:

        left = 0

    else:

        start_value = np.datetime64(
            pd.Timestamp(start_date).normalize()
        )

        left = int(
            np.searchsorted(
                days,
                start_value,
                side="left",
            )
        )

    right = int(
        np.searchsorted(
            days,
            end_value,
            side="left",
        )
    )

    delay_sum = (
        delay_prefix[right]
        - delay_prefix[left]
    )

    flight_count = (
        count_prefix[right]
        - count_prefix[left]
    )

    return float(delay_sum), int(flight_count)


# ============================================================
# GROUP RANGE QUERY
# ============================================================

def _query_group(
    reference,
    prefix,
    key,
    start_date,
    end_date,
):

    group_index = reference.group_indexes[
        prefix
    ].get(key)

    if group_index is None:
        return 0.0, 0

    return _query_prefix_arrays(
        group_index.days,
        group_index.delay_prefix,
        group_index.count_prefix,
        start_date,
        end_date,
    )


def _query_global(
    reference,
    start_date,
    end_date,
):

    return _query_prefix_arrays(
        reference.global_days,
        reference.global_delay_prefix,
        reference.global_count_prefix,
        start_date,
        end_date,
    )


# ============================================================
# SMOOTHING
# ============================================================

def _smoothed_rate(
    delay_sum,
    flight_count,
    alpha,
    global_rate,
):

    return float(
        (delay_sum + alpha * global_rate)
        / (flight_count + alpha)
    )


# ============================================================
# BUILD ONE FLIGHT INPUT ROW
# ============================================================

def _prepare_flight_row(flight_inputs):

    missing = [
        column
        for column in PREDICTION_COLUMNS
        if column not in flight_inputs
        or flight_inputs[column] is None
        or (
            isinstance(flight_inputs[column], str)
            and not flight_inputs[column].strip()
        )
    ]

    if missing:
        raise ValueError(
            f"Input penerbangan belum lengkap: {missing}"
        )

    prediction_date = pd.to_datetime(
        flight_inputs["FL_DATE"],
        errors="coerce",
    )

    if pd.isna(prediction_date):
        raise ValueError(
            "FL_DATE tidak valid."
        )

    prediction_date = prediction_date.normalize()

    departure_time = _numeric_input(
        flight_inputs["CRS_DEP_TIME"],
        "CRS_DEP_TIME",
    )

    arrival_time = _numeric_input(
        flight_inputs["CRS_ARR_TIME"],
        "CRS_ARR_TIME",
    )

    elapsed_time = _numeric_input(
        flight_inputs["CRS_ELAPSED_TIME"],
        "CRS_ELAPSED_TIME",
    )

    distance = _numeric_input(
        flight_inputs["DISTANCE"],
        "DISTANCE",
    )

    if elapsed_time <= 0:
        raise ValueError(
            "CRS_ELAPSED_TIME harus lebih besar dari 0."
        )

    if distance <= 0:
        raise ValueError(
            "DISTANCE harus lebih besar dari 0."
        )

    # V3 dataset convention:
    # Monday=1 ... Sunday=7.
    day_of_week = prediction_date.weekday() + 1

    row = pd.DataFrame([{
        "FL_DATE": prediction_date,

        "OP_CARRIER": _normalize_category(
            flight_inputs["OP_CARRIER"],
            "OP_CARRIER",
        ),

        "ORIGIN": _normalize_category(
            flight_inputs["ORIGIN"],
            "ORIGIN",
        ),

        "DEST": _normalize_category(
            flight_inputs["DEST"],
            "DEST",
        ),

        "CRS_DEP_TIME": departure_time,
        "CRS_ARR_TIME": arrival_time,
        "CRS_ELAPSED_TIME": elapsed_time,

        "MONTH": prediction_date.month,
        "DAY_OF_MONTH": prediction_date.day,
        "DAY_OF_WEEK": day_of_week,

        "DISTANCE": distance,
    }])

    row = add_departure_hour(row)
    row = add_cyclical_features(row)
    row = add_arrival_cyclical_features(row)
    row = add_day_of_month_cyclical_features(row)

    return row, prediction_date


# ============================================================
# LIFETIME HISTORICAL FEATURES
# ============================================================

def _lifetime_features(
    reference,
    prediction_date,
    lookup,
):

    global_delay, global_count = _query_global(
        reference,
        None,
        prediction_date,
    )

    global_rate = (
        global_delay / global_count
        if global_count > 0
        else 0.50
    )

    features = {}

    for keys, prefix, alpha in HISTORY_GROUPS:

        key = (
            lookup[keys[0]]
            if len(keys) == 1
            else tuple(
                lookup[column]
                for column in keys
            )
        )

        delay_sum, flight_count = _query_group(
            reference,
            prefix,
            key,
            None,
            prediction_date,
        )

        features[
            f"{prefix}_HIST_DELAY_RATE"
        ] = _smoothed_rate(
            delay_sum,
            flight_count,
            alpha,
            global_rate,
        )

    return features


# ============================================================
# ROLLING HISTORICAL FEATURES
# ============================================================

def _rolling_features(
    reference,
    prediction_date,
    lookup,
):

    features = {}

    for window in WINDOWS:

        start_date = (
            prediction_date
            - pd.Timedelta(days=window)
        )

        global_delay, global_count = _query_global(
            reference,
            start_date,
            prediction_date,
        )

        global_rate = (
            global_delay / global_count
            if global_count > 0
            else 0.50
        )

        for keys, prefix, alpha in ROLLING_GROUPS:

            key = (
                lookup[keys[0]]
                if len(keys) == 1
                else tuple(
                    lookup[column]
                    for column in keys
                )
            )

            delay_sum, flight_count = _query_group(
                reference,
                prefix,
                key,
                start_date,
                prediction_date,
            )

            feature_name = (
                f"{prefix}_ROLLING_{window}D_DELAY_RATE"
            )

            features[feature_name] = _smoothed_rate(
                delay_sum,
                flight_count,
                alpha,
                global_rate,
            )

    return features


# ============================================================
# MAIN INFERENCE FEATURE BUILDER
# ============================================================

def build_v3_prediction_features(
    flight_inputs: dict,
    history_frame: pd.DataFrame | V3HistoryReference,
) -> pd.DataFrame:
    """
    Build the exact 41 features used by the V3 Extra Trees model.

    history_frame accepts:
    1. A raw historical DataFrame.
    2. A prepared V3HistoryReference.

    Prepared reference is recommended for repeated predictions.

    Only flights with dates strictly earlier than prediction date
    are included.

    Rolling windows:
        7D  = D-7 through D-1
        14D = D-14 through D-1
        28D = D-28 through D-1
    """

    row, prediction_date = _prepare_flight_row(
        flight_inputs
    )

    if isinstance(history_frame, V3HistoryReference):

        reference = history_frame

    else:

        reference = prepare_v3_history_reference(
            history_frame
        )

    lookup = {
        column: row.iloc[0][column]
        for column in CATEGORICAL_FEATURES
    }

    lookup["_DEP_HOUR"] = int(
        row.iloc[0]["_DEP_HOUR"]
    )

    lifetime = _lifetime_features(
        reference,
        prediction_date,
        lookup,
    )

    rolling = _rolling_features(
        reference,
        prediction_date,
        lookup,
    )

    for name, value in lifetime.items():
        row[name] = value

    for name, value in rolling.items():
        row[name] = value

    features = row[
        FEATURE_COLUMNS
    ].copy()

    for column in CATEGORICAL_FEATURES:
        features[column] = (
            features[column].astype(str)
        )

    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(
            features[column],
            errors="coerce",
        )

    numeric_values = features[
        NUMERIC_FEATURES
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Fitur numerik V3 mengandung NaN atau infinity."
        )

    if features.shape != (1, 41):
        raise RuntimeError(
            f"Schema inference tidak sesuai: {features.shape}"
        )

    return features
