from __future__ import annotations

import numpy as np


def hhmm_to_cyclic(values):
    """
    Convert BTS HHMM scheduled departure time
    into cyclical sine and cosine features.

    Example:
        1649 = 16:49

    Output:
        DEP_TIME_SIN
        DEP_TIME_COS
    """

    values = np.asarray(
        values,
        dtype=float,
    ).reshape(-1)

    hours = np.floor(values / 100)
    minutes = values % 100

    valid = (
        np.isfinite(values)
        & (values >= 0)
        & (values <= 2359)
        & (minutes >= 0)
        & (minutes < 60)
        & (values == np.floor(values))
    )

    minutes_of_day = hours * 60 + minutes

    angle = (
        2 * np.pi * minutes_of_day / 1440
    )

    sin_time = np.sin(angle)
    cos_time = np.cos(angle)

    encoded = np.column_stack([
        sin_time,
        cos_time,
    ])

    encoded[~valid] = np.nan

    return encoded