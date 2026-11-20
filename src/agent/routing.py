"""Routing utilities for the agent: heuristic tool selection."""
from __future__ import annotations

import re

from .constants import RIGHTS_WORDS, PREDICTION_WORDS, _explicitly_requests_prediction


def heuristic_route(question: str) -> list[str]:
    """Local fallback routing for explicit prediction and passenger-rights requests."""
    text = re.sub(r"\s+", " ", (question or "").lower()).strip()

    rights = any(term in text for term in RIGHTS_WORDS)

    # A generic mention of delay is not a prediction request.
    # Require explicit prediction intent instead.
    prediction = _explicitly_requests_prediction(text)

    selected: list[str] = []
    if prediction:
        selected.append("predict_flight_delay")
    if rights:
        selected.append("search_passenger_rights")
    return selected