from src.predictor import _interpret_probability


def test_probability_near_threshold_is_not_called_likely_on_time():
    result = _interpret_probability(0, 0.49).lower()
    assert "dekat ambang klasifikasi 50%" in result
    assert "bukan bukti" in result


def test_probability_below_threshold_is_not_a_guarantee():
    result = _interpret_probability(0, 0.30).lower()
    assert "di bawah ambang klasifikasi 50%" in result
    assert "bukan jaminan" in result


def test_missing_probability_is_explicit():
    result = _interpret_probability(0, None).lower()
    assert "probabilitas tidak tersedia" in result
