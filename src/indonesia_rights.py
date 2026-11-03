from __future__ import annotations

from pathlib import Path
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS_DIR = ROOT / "documents"
INDONESIA_FILES = (
    DOCUMENTS_DIR / "id_pm_89_2015.txt",
    DOCUMENTS_DIR / "id_uu_1_2009.txt",
)


def _parse_indonesia_document(path: Path) -> list[dict]:
    if not path.is_file():
        return []

    raw = path.read_text(encoding="utf-8")
    title_match = re.search(r"^TITLE:\s*(.+)$", raw, re.MULTILINE)
    url_match = re.search(r"^URL:\s*(.+)$", raw, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem
    url = url_match.group(1).strip() if url_match else ""

    body = re.sub(
        r"^(?:TITLE|URL|STATUS|JURISDICTION):.*(?:\n|$)",
        "",
        raw,
        flags=re.MULTILINE,
    ).strip()

    parts = re.split(r"(?m)^##\s+", body)
    records = []

    for index, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue

        lines = part.splitlines()
        section = lines[0].strip(" #-") if lines else f"Bagian {index}"
        text = "\n".join(lines[1:]).strip()

        if text:
            records.append({
                "id": f"{path.stem}:{index}",
                "title": title,
                "section": section,
                "url": url,
                "text": text,
            })

    return records


def _load_indonesia_chunks() -> list[dict]:
    records: list[dict] = []

    for path in INDONESIA_FILES:
        records.extend(_parse_indonesia_document(path))

    return records


def retrieve_indonesia_passenger_rights(query: str, top_k: int = 6) -> list[dict]:
    """Retrieve evidence from the curated Indonesian legal corpus."""
    query = re.sub(r"\s+", " ", str(query or "")).strip()
    if not query:
        return []

    records = _load_indonesia_chunks()
    if not records:
        return []

    lowered = query.lower()

    delay_intent = any(term in lowered for term in (
        "keterlambatan", "terlambat", "delay", "delayed",
    ))

    denied_boarding_intent = any(term in lowered for term in (
        "kelebihan kapasitas",
        "overbooking",
        "denied boarding",
        "tidak terangkut",
        "ditolak naik",
        "bumping",
    ))

    status_intent = any(term in lowered for term in (
        "status peraturan",
        "masih berlaku",
        "berlaku atau tidak",
        "status pm",
    ))

    retrieval_query = query

    if delay_intent and not denied_boarding_intent:
        retrieval_query += (
            " kategori keterlambatan penerbangan 30 60 61 120 "
            "121 180 181 240 lebih dari 240 menit "
            "Pasal 3 Pasal 4 Pasal 5 Pasal 6 "
            "Pasal 9 kompensasi minuman makanan snack box "
            "heavy meal Rp300.000 refund pengembalian tiket "
            "Pasal 10 akomodasi penginapan lebih dari 6 jam"
        )

    documents = [
        " ".join((item["title"], item["section"], item["text"]))
        for item in records
    ]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[\w/-]+\b",
    )

    try:
        matrix = vectorizer.fit_transform(documents + [retrieval_query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    except ValueError:
        return []

    ranked = []

    for index, item in enumerate(records):
        section = item["section"].lower()
        text = item["text"].lower()
        score = float(scores[index])

        is_status_note = "catatan status" in section
        is_pasal147 = "pasal 147" in section

        if is_status_note and delay_intent and not status_intent:
            continue

        if is_pasal147 and not denied_boarding_intent:
            continue

        priority = 0

        if delay_intent and not denied_boarding_intent:
            if "pasal 9" in section:
                priority = 100
            elif "pasal 3" in section:
                priority = 90
            elif "pasal 10" in section:
                priority = 80
            elif "pasal 6" in section:
                priority = 60
            elif "pasal 5" in section:
                priority = 50
            elif "pasal 4" in section:
                priority = 40
            elif "pasal 146" in section:
                priority = 30

        elif denied_boarding_intent and "pasal 147" in section:
            priority = 100

        if score <= 0 and priority == 0:
            continue

        ranked.append((priority, score, index))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

    results = []

    for priority, score, index in ranked:
        item = records[index]
        results.append({
            "title": item["title"],
            "section": item["section"],
            "url": item["url"],
            "excerpt": item["text"],
            "score": score,
        })

        if len(results) >= max(1, int(top_k)):
            break

    return results
