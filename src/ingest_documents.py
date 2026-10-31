from __future__ import annotations

import json
import re
from pathlib import Path

from src.config import DOCUMENTS_DIR, CHUNKS_PATH, PROCESSED_DIR

DOCUMENT_SOURCES = {
    "us_dot_refunds.txt": {
        "title": "U.S. DOT Refunds — Local summary (not verbatim)",
        "url": "https://www.transportation.gov/individuals/aviation-consumer-protection/refunds",
    },
    "us_dot_tarmac_delays.txt": {
        "title": "U.S. DOT Tarmac Delays — Local summary (not verbatim)",
        "url": "https://www.transportation.gov/individuals/aviation-consumer-protection/tarmac-delays",
    },
    "us_dot_airline_commitments_american.txt": {
        "title": "U.S. DOT Airline Customer Service Dashboard — American Airlines local summary (not verbatim)",
        "url": "https://www.transportation.gov/airconsumer/airline-customer-service-dashboard",
    },
    "us_cfr_260_2_significant_delay.txt": {
        "title": "eCFR — 14 CFR §260.2 Significant Delay Definition (local summary)",
        "url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-260/section-260.2",
    },
    "us_cfr_260_6_refunds.txt": {
        "title": "eCFR — 14 CFR §260.6 Refund Obligation (local summary)",
        "url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-260/section-260.6",
    },
}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _is_heading(line: str) -> bool:
    value = line.strip()
    if not value or len(value) > 110:
        return False
    if value.startswith(("DOCUMENT:", "JURISDICTION:", "OFFICIAL SOURCE:", "SOURCE CHECKED:", "DOCUMENT TYPE:", "SOURCE PAGE LAST UPDATED:", "LOCAL SUMMARY CHECKED:", "DASHBOARD PAGE LAST-UPDATED LABEL:", "SEARCH TERMS (")):
        return True
    letters = [char for char in value if char.isalpha()]
    return bool(letters) and value == value.upper() and len(value.split()) <= 14


def _split_sentences(text: str) -> list[str]:
    text = _normalise(text)
    if not text:
        return []
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]


def _split_block(text: str, max_chars: int) -> list[str]:
    text = _normalise(text)
    if len(text) <= max_chars:
        return [text] if len(text) >= 40 else []
    chunks: list[str] = []
    current = ""
    for sentence in _split_sentences(text):
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            # Do not silently cut a legal sentence. Mark it as a long complete sentence.
            chunks.append(sentence)
            continue
        candidate = f"{current} {sentence}".strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if len(chunk) >= 40]


def _document_chunks(text: str, max_chars: int = 1000) -> list[dict[str, str]]:
    """Split on headings/paragraphs and retain each section label with its text."""
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Overview"
    current_blocks: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _is_heading(line):
            if current_blocks:
                sections.append((current_heading, current_blocks))
                current_blocks = []
            current_heading = line.rstrip(":")
        else:
            current_blocks.append(line)
    if current_blocks:
        sections.append((current_heading, current_blocks))

    results: list[dict[str, str]] = []
    ignored_prefixes = ("DOCUMENT:", "JURISDICTION:", "OFFICIAL SOURCE:", "SOURCE CHECKED:", "DOCUMENT TYPE:", "SOURCE PAGE LAST UPDATED:", "LOCAL SUMMARY CHECKED:", "DASHBOARD PAGE LAST-UPDATED LABEL:", "SEARCH TERMS (")
    for heading, blocks in sections:
        if heading.startswith(ignored_prefixes):
            continue
        paragraph_chunks: list[str] = []
        for block in blocks:
            paragraph_chunks.extend(_split_block(block, max_chars))
        current = ""
        for piece in paragraph_chunks:
            combined = f"{current}\n\n{piece}".strip() if current else piece
            if len(combined) <= max_chars:
                current = combined
            else:
                if current:
                    results.append({"section": heading, "text": current})
                current = piece
        if current:
            results.append({"section": heading, "text": current})
    return results


def run_ingestion() -> Path:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    for filename, metadata in DOCUMENT_SOURCES.items():
        path = DOCUMENTS_DIR / filename
        if not path.is_file():
            print(f"Dokumen wajib tidak ditemukan, dilewati: {path}")
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        chunks = _document_chunks(content)
        print(f"Dokumen diproses: {filename} | {len(chunks)} chunks")
        for chunk_number, chunk in enumerate(chunks, start=1):
            records.append({
                "id": f"{filename}:{chunk_number}",
                "title": metadata["title"],
                "url": metadata["url"],
                "section": chunk["section"],
                "chunk": chunk_number,
                "text": chunk["text"],
            })

    if not records:
        raise RuntimeError("Tidak ada dokumen hak penumpang yang berhasil diindeks. Periksa folder documents.")

    with CHUNKS_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Indeks dokumen tersimpan: {CHUNKS_PATH} | {len(records)} chunks")
    return CHUNKS_PATH


if __name__ == "__main__":
    run_ingestion()
