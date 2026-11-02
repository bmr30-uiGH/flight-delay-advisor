from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORICAL_FEATURES = [
    "OP_CARRIER",
    "ORIGIN",
    "DEST",
]

CYCLICAL_NUMERIC_FEATURES = [
    "DEP_TIME_SIN",
    "DEP_TIME_COS",
    "DOW_SIN",
    "DOW_COS",
    "MONTH_SIN",
    "MONTH_COS",
    "DISTANCE",
]

HISTORICAL_RATE_FEATURES = [
    "ORIGIN_HIST_DELAY_RATE",
    "DEST_HIST_DELAY_RATE",
    "CARRIER_HIST_DELAY_RATE",
    "ROUTE_HIST_DELAY_RATE",
    "CARRIER_ROUTE_HIST_DELAY_RATE",
]

MODEL_FEATURES = (
    CATEGORICAL_FEATURES
    + CYCLICAL_NUMERIC_FEATURES
    + HISTORICAL_RATE_FEATURES
)

HISTORY_GROUPS = [
    (["ORIGIN"], "ORIGIN", 100),
    (["DEST"], "DEST", 100),
    (["OP_CARRIER"], "CARRIER", 100),
    (["ORIGIN", "DEST"], "ROUTE", 50),
    (["OP_CARRIER", "ORIGIN", "DEST"], "CARRIER_ROUTE", 50),
]

RAW_FEATURES = [
    "OP_CARRIER",
    "ORIGIN",
    "DEST",
    "CRS_DEP_TIME",
    "MONTH",
    "DAY_OF_WEEK",
    "DISTANCE",
]

TARGET = "DELAY_15"


def add_cyclical_features(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Convert scheduled time, weekday, and month into cyclical features.

    The original raw columns are retained in the returned DataFrame.
    """

    required = [
        "CRS_DEP_TIME",
        "MONTH",
        "DAY_OF_WEEK",
    ]

    missing = [
        column for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom untuk cyclical encoding hilang: {missing}"
        )

    result = frame.copy()

    # HHMM -> minutes from midnight.
    raw_time = pd.to_numeric(
        result["CRS_DEP_TIME"],
        errors="coerce",
    )

    hour = np.floor(raw_time / 100)
    minute = raw_time % 100

    valid_time = (
        raw_time.notna()
        & hour.between(0, 23)
        & minute.between(0, 59)
        & (raw_time == np.floor(raw_time))
    )

    minutes_of_day = (
        hour * 60 + minute
    ).where(valid_time)

    angle = 2 * np.pi * minutes_of_day / 1440

    result["DEP_TIME_SIN"] = np.sin(angle)
    result["DEP_TIME_COS"] = np.cos(angle)

    # BTS weekday: Sunday=1 ... Saturday=7.
    day = pd.to_numeric(
        result["DAY_OF_WEEK"],
        errors="coerce",
    )

    day = day.where(day.between(1, 7))

    day_angle = 2 * np.pi * (day - 1) / 7

    result["DOW_SIN"] = np.sin(day_angle)
    result["DOW_COS"] = np.cos(day_angle)

    # Month: January=1 ... December=12.
    month = pd.to_numeric(
        result["MONTH"],
        errors="coerce",
    )

    month = month.where(month.between(1, 12))

    month_angle = 2 * np.pi * (month - 1) / 12

    result["MONTH_SIN"] = np.sin(month_angle)
    result["MONTH_COS"] = np.cos(month_angle)

    return result


def _normalize_history_data(frame: pd.DataFrame) -> pd.DataFrame:
    required = (
        RAW_FEATURES
        + [TARGET, "FL_DATE"]
    )

    missing = [
        column for column in required
        if column not in frame.columns
    ]

    if missing:
        raise ValueError(
            f"Kolom data historis hilang: {missing}"
        )

    result = frame[
        required
    ].copy()

    result["FL_DATE"] = pd.to_datetime(
        result["FL_DATE"],
        errors="coerce",
    )

    result[TARGET] = pd.to_numeric(
        result[TARGET],
        errors="coerce",
    )

    result = result.dropna(
        subset=required
    ).copy()

    if result.empty:
        raise ValueError(
            "Data historis tidak memiliki baris valid."
        )

    if not result[TARGET].isin([0, 1]).all():
        raise ValueError(
            "DELAY_15 hanya boleh bernilai 0 atau 1."
        )

    for column in CATEGORICAL_FEATURES:
        result[column] = (
            result[column]
            .astype(str)
            .str.strip()
            .str.upper()
        )

    result["_DAY"] = (
        result["FL_DATE"].dt.normalize()
    )

    return result


def add_past_only_historical_rates(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add historical delay rates using only flights from earlier dates.

    All flights on the current date are excluded from their own
    historical aggregates. No future date is used.

    This function is intended for date-aware training/evaluation data.
    """

    data = _normalize_history_data(frame)

    result = data.copy()
    result["_ROW_ID"] = np.arange(len(result))

    # Global historical rate available before each date.
    global_daily = (
        result.groupby("_DAY")[TARGET]
        .agg(["sum", "count"])
        .sort_index()
    )

    global_daily["_PRIOR_SUM"] = (
        global_daily["sum"]
        .cumsum()
        .shift(1)
        .fillna(0)
    )

    global_daily["_PRIOR_COUNT"] = (
        global_daily["count"]
        .cumsum()
        .shift(1)
        .fillna(0)
    )

    global_daily["_PRIOR_RATE"] = np.where(
        global_daily["_PRIOR_COUNT"] > 0,
        (
            global_daily["_PRIOR_SUM"]
            / global_daily["_PRIOR_COUNT"]
        ),
        0.50,
    )

    for keys, prefix, alpha in HISTORY_GROUPS:
        daily = (
            result.groupby(
                keys + ["_DAY"],
                observed=True,
            )[TARGET]
            .agg(["sum", "count"])
            .reset_index()
            .rename(columns={
                "sum": "_DELAY_SUM",
                "count": "_FLIGHT_COUNT",
            })
        )

        daily = daily.sort_values(
            keys + ["_DAY"]
        ).reset_index(drop=True)

        grouped = daily.groupby(
            keys,
            observed=True,
            sort=False,
        )

        # Subtract current-day aggregates so only prior dates remain.
        daily["_PAST_DELAY_SUM"] = (
            grouped["_DELAY_SUM"].cumsum()
            - daily["_DELAY_SUM"]
        )

        daily["_PAST_FLIGHT_COUNT"] = (
            grouped["_FLIGHT_COUNT"].cumsum()
            - daily["_FLIGHT_COUNT"]
        )

        daily = daily.merge(
            global_daily[["_PRIOR_RATE"]],
            left_on="_DAY",
            right_index=True,
            how="left",
            validate="many_to_one",
        )

        feature_name = f"{prefix}_HIST_DELAY_RATE"

        daily[feature_name] = (
            daily["_PAST_DELAY_SUM"]
            + alpha * daily["_PRIOR_RATE"]
        ) / (
            daily["_PAST_FLIGHT_COUNT"] + alpha
        )

        lookup = daily[
            keys + ["_DAY", feature_name]
        ]

        result = result.merge(
            lookup,
            on=keys + ["_DAY"],
            how="left",
            sort=False,
            validate="many_to_one",
        )

        result = result.sort_values(
            "_ROW_ID",
            kind="stable",
        ).reset_index(drop=True)

    result = result.drop(
        columns=["_ROW_ID"]
    )

    if result[HISTORICAL_RATE_FEATURES].isna().any().any():
        raise ValueError(
            "Historical rate tidak boleh mengandung nilai kosong."
        )

    rates = result[HISTORICAL_RATE_FEATURES]

    if not rates.ge(0).all().all():
        raise ValueError(
            "Historical rate tidak boleh kurang dari 0."
        )

    if not rates.le(1).all().all():
        raise ValueError(
            "Historical rate tidak boleh lebih dari 1."
        )

    return result


def prepare_historical_reference(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize historical data once for repeated prediction requests.

    The returned DataFrame is read-only by convention and is suitable
    for caching inside the predictor.
    """

    normalized = _normalize_history_data(frame)
    normalized.attrs["_historical_reference_normalized"] = True
    return normalized


def historical_rates_for_prediction(
    flight_inputs: dict,
    history_frame: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate historical rates for one flight prediction.

    Only source flights with FL_DATE strictly earlier than the
    requested flight date are considered.
    """

    missing_input = [
        column for column in RAW_FEATURES + ["FL_DATE"]
        if column not in flight_inputs
        or flight_inputs[column] in (None, "")
    ]

    if missing_input:
        raise ValueError(
            f"Input prediksi belum lengkap: {missing_input}"
        )

    prediction_date = pd.to_datetime(
        flight_inputs["FL_DATE"],
        errors="coerce",
    )

    if pd.isna(prediction_date):
        raise ValueError(
            "FL_DATE untuk prediksi tidak valid."
        )

    prediction_day = prediction_date.normalize()

    # Reuse the normalized reference when supplied by the predictor.
    # Raw DataFrames remain supported for existing tests/callers.
    if history_frame.attrs.get(
        "_historical_reference_normalized"
    ) is True:
        history = history_frame
    else:
        history = prepare_historical_reference(history_frame)

    # Strict inequality: same-day targets are never included.
    # Avoid copying the entire historical DataFrame.
    prior_mask = history["_DAY"] < prediction_day

    if not bool(prior_mask.any()):
        global_rate = 0.50
    else:
        global_rate = float(
            history.loc[prior_mask, TARGET].mean()
        )

    rates: dict[str, float] = {}

    normalized_input = {}

    for column in CATEGORICAL_FEATURES:
        normalized_input[column] = (
            str(flight_inputs[column])
            .strip()
            .upper()
        )

    for keys, prefix, alpha in HISTORY_GROUPS:
        matched_mask = prior_mask.copy()

        for key in keys:
            matched_mask &= history[key].eq(
                normalized_input[key]
            )

        past_count = int(matched_mask.sum())
        past_delay_sum = float(
            history.loc[matched_mask, TARGET].sum()
        )

        rate = (
            past_delay_sum + alpha * global_rate
        ) / (
            past_count + alpha
        )

        rates[
            f"{prefix}_HIST_DELAY_RATE"
        ] = float(rate)

    return rates


def prepare_prediction_features(
    flight_inputs: dict,
    history_frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce the exact feature schema expected by the new candidate.
    """

    missing_input = [
        column for column in RAW_FEATURES + ["FL_DATE"]
        if column not in flight_inputs
        or flight_inputs[column] in (None, "")
    ]

    if missing_input:
        raise ValueError(
            f"Input prediksi belum lengkap: {missing_input}"
        )

    row = pd.DataFrame([
        {
            column: flight_inputs[column]
            for column in RAW_FEATURES
        }
    ])

    row = add_cyclical_features(row)

    rates = historical_rates_for_prediction(
        flight_inputs,
        history_frame,
    )

    for name, value in rates.items():
        row[name] = value

    features = row[MODEL_FEATURES].copy()

    if features.isna().any().any():
        raise ValueError(
            "Fitur prediksi mengandung nilai kosong."
        )

    return features