import numpy as np

from src.features import hhmm_to_cyclic


def test_hhmm_to_cyclic_returns_two_features():
    result = hhmm_to_cyclic(
        np.array([[1649], [1550]])
    )

    assert result.shape == (2, 2)


def test_times_around_midnight_are_close():
    result = hhmm_to_cyclic(
        np.array([[2359], [1], [1200]])
    )

    distance_midnight = np.linalg.norm(
        result[0] - result[1]
    )

    distance_noon = np.linalg.norm(
        result[0] - result[2]
    )

    assert distance_midnight < 0.02
    assert distance_midnight < distance_noon


def test_invalid_hhmm_returns_nan():
    result = hhmm_to_cyclic(
        np.array([[1260]])
    )

    assert np.isnan(result).all()