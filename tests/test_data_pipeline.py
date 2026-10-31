import pandas as pd
from src.data_pipeline import clean_flights


def test_clean_flights_builds_target_without_delay_feature():
    raw = pd.DataFrame({
        "OP_UNIQUE_CARRIER": ["DL", "AA", "UA", "DL"],
        "ORIGIN": ["JFK", "LAX", "ORD", "ATL"],
        "DEST": ["LAX", "JFK", "DEN", "MIA"],
        "CRS_DEP_TIME": [800, 900, 1200, 1800],
        "DEP_DELAY": [0, 15, 20, 5],
        "DISTANCE": [2475, 2475, 888, 600],
        "MONTH": [1, 1, 1, 1],
        "DAY_OF_WEEK": [2, 3, 4, 5],
        "CANCELLED": [0, 0, 0, 0],
        "DIVERTED": [0, 0, 0, 0],
    })
    cleaned = clean_flights(raw)
    assert "DEP_DELAY" not in cleaned.columns
    assert cleaned["DELAY_15"].tolist() == [0, 1, 1, 0]
    assert len(cleaned) == 4
