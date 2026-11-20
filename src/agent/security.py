"""Security utilities for the agent: prompt injection detection and sanitization."""
from __future__ import annotations

import re

from .constants import _INJECTION_CLAUSE_PATTERNS


def _is_injection_clause(clause: str) -> bool:
    text = re.sub(r"\s+", " ", clause.lower()).strip()
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in _INJECTION_CLAUSE_PATTERNS)


def sanitize_mixed_request(question: str) -> tuple[str, bool]:
    """Remove explicit hostile instruction clauses while preserving separable safe questions.

    This is a narrow defense-in-depth pre-filter, not a complete prompt-injection detector.
    It deliberately leaves the original message unchanged if no attack clause is found.
    """
    text = (question or "").strip()
    if not text:
        return "", False

    # Split on paragraphs/newlines first, then sentence boundaries, retaining safe clauses.
    raw_segments: list[str] = []
    for paragraph in re.split(r"\n+", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        raw_segments.extend(
            part.strip()
            for part in re.split(r"(?<=[.!?])\s+", paragraph)
            if part.strip()
        )

    safe_segments: list[str] = []
    detected = False
    for segment in raw_segments:
        if _is_injection_clause(segment):
            detected = True
            continue
        safe_segments.append(segment)

    if not detected:
        return text, False
    cleaned = " ".join(safe_segments).strip()
    return cleaned, True


def looks_like_injection(text: str) -> bool:
    _cleaned, detected = sanitize_mixed_request(text)
    return detected