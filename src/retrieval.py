from __future__ import annotations

import json
import re
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import CHUNKS_PATH

# Small bilingual expansions help Indonesian questions match English DOT summaries.
_QUERY_EXPANSIONS = {
    "refund": "refund ticket price original form of payment",
    "pengembalian dana": "refund ticket price original form of payment",
    "pengembalian uang": "refund ticket price original form of payment",
    "kedatangan": "late arrival scheduled arrival destination airport",
    "tiba": "late arrival scheduled arrival destination airport",
    "keberangkatan lebih awal": "early departure scheduled departure origination airport",
    "berangkat lebih awal": "early departure scheduled departure origination airport",
    "penerbangan alternatif": "alternative flight offers rebook transportation",
    "menolak": "reject offer does not accept alternative transportation",
    "menerima": "accepts takes alternative transportation traveled",
    "kompensasi": "compensation cash credit travel voucher airline commitments",
    "makanan": "meal meal voucher meal cash",
    "hotel": "complimentary hotel ground transportation overnight delay",
    "landasan": "tarmac delay aircraft on the ground opportunity to leave aircraft",
    "di pesawat": "tarmac delay aircraft passengers opportunity to deplane",
    "american airlines": "American airline AA airline customer service dashboard controllable delay",
}


def _expand_query(query: str) -> str:
    lowered = query.lower()
    additions = [value for phrase, value in _QUERY_EXPANSIONS.items() if phrase in lowered]
    return f"{query} {' '.join(additions)}" if additions else query


@lru_cache(maxsize=2)
def _load_chunks(path_string: str, modified_ns: int) -> tuple[dict, ...]:
    del modified_ns
    path = CHUNKS_PATH
    if not path.exists():
        return tuple()
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("text") and item.get("url"):
                records.append(item)
    return tuple(records)


def retrieve_passenger_rights(query: str, top_k: int = 4) -> list[dict]:
    if not CHUNKS_PATH.exists() or not query.strip():
        return []
    chunks = _load_chunks(str(CHUNKS_PATH), CHUNKS_PATH.stat().st_mtime_ns)
    if not chunks:
        return []

    searchable_texts = [
        " ".join((item.get("title", ""), item.get("section", ""), item.get("text", "")))
        for item in chunks
    ]
    expanded_query = _expand_query(query)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=30000)
    try:
        matrix = vectorizer.fit_transform(searchable_texts + [expanded_query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    except ValueError:
        return []

    ranked = scores.argsort()[::-1]
    results: list[dict] = []
    for index in ranked:
        score = float(scores[index])
        if score <= 0.005:
            continue
        item = chunks[int(index)]
        results.append({
            "title": item.get("title", "U.S. DOT document"),
            "section": item.get("section", ""),
            "url": item.get("url", ""),
            "excerpt": item.get("text", ""),
            "score": round(score, 4),
            "chunk_id": item.get("id", ""),
        })
        if len(results) >= max(1, min(top_k, 6)):
            break
    return results
