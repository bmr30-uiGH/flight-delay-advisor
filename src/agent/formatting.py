"""Formatting utilities for the agent: flight summary, context, citations, source markers."""
from __future__ import annotations

import json
import re
from datetime import date

from .constants import SYSTEM_INSTRUCTIONS


_CFR_REFERENCE_PATTERN = re.compile(
    r"\b14\s+CFR\s+(?:(?:Part)\s+)?(?:§{1,2}\s*)?\d+(?:\.\d+)?",
    flags=re.IGNORECASE,
)


def _format_flight_summary_context(flight_summary: dict | None) -> str:
    """Render calendar and clock details from trusted app form data, not LLM inference."""
    if not flight_summary:
        return "Ringkasan penerbangan dari formulir: tidak tersedia. Jangan menebak tanggal, hari, atau zona waktu."

    summary = dict(flight_summary)
    flight_date = str(summary.get("flight_date", "")).strip()
    weekday_text = ""
    if flight_date:
        try:
            parsed_date = date.fromisoformat(flight_date)
            weekdays_id = ("Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu")
            weekday_text = f"{weekdays_id[parsed_date.weekday()]}, dihitung oleh Python"
        except ValueError:
            weekday_text = "hari tidak dapat diverifikasi dari tanggal yang tersimpan"

    carrier = summary.get("carrier") or summary.get("carrier_code") or "maskapai tidak tersedia"
    origin = summary.get("origin_code") or summary.get("origin") or "bandara asal tidak tersedia"
    destination = summary.get("destination_code") or summary.get("destination") or "bandara tujuan tidak tersedia"
    departure_time = summary.get("departure_time") or "jam tidak tersedia"
    distance = summary.get("distance")

    details = [f"Maskapai: {carrier}", f"Rute: {origin} ke {destination}"]
    if flight_date:
        details.append(f"Tanggal: {flight_date} ({weekday_text})")
    details.append(f"Jam keberangkatan: {departure_time} waktu lokal di bandara asal {origin}; jangan ubah menjadi WIB atau zona waktu perangkat")
    if distance is not None:
        details.append(f"Jarak rute: {distance} miles")
    return "Ringkasan penerbangan tepercaya dari formulir aplikasi: " + "; ".join(details) + "."


def _question_with_flight_context(
    question: str,
    flight_inputs: dict | None,
    flight_summary: dict | None = None,
) -> str:
    if not flight_inputs:
        context = "Detail penerbangan tersimpan: TIDAK ADA. Jika pengguna meminta prediksi, arahkan untuk mengisi dan menyimpan form terlebih dahulu."
    else:
        context = (
            "Detail fitur penerbangan tersimpan dari form (gunakan untuk tool ML; jangan tanya ulang):\n"
            + json.dumps(flight_inputs, ensure_ascii=False, sort_keys=True)
        )
    summary_context = _format_flight_summary_context(flight_summary)
    return (
        f"Pertanyaan substantif pengguna:\n{question}\n\nKonteks aplikasi:\n"
        "PEMISAHAN KONTEKS: Detail penerbangan dan ringkasan formulir di bawah ini disediakan untuk prediksi ML. Jangan gunakan sebagai fakta untuk kasus hak penumpang, termasuk nama maskapai, merchant of record, jenis tarif, status pembatalan, atau pilihan penumpang, kecuali pengguna secara eksplisit menyatakan bahwa kasus hak tersebut membahas penerbangan yang sama. Walaupun penerbangannya sama, jangan anggap OP_CARRIER sebagai merchant of record.\n"
        f"{context}\n{summary_context}"
    )


def _remove_unsupported_cfr_citations(answer: str, sources: list[dict]) -> str:
    """Replace CFR citations that do not appear in this turn's retrieved evidence."""
    evidence = " ".join(
        str(value)
        for source in sources
        for value in (source.get("title", ""), source.get("section", ""), source.get("excerpt", ""))
    )
    evidence_normalised = re.sub(r"[^a-z0-9.]", "", evidence.lower())

    def validate(match: re.Match) -> str:
        citation = re.sub(r"[^a-z0-9.]", "", match.group(0).lower())
        if citation and citation in evidence_normalised:
            return match.group(0)
        return "rujukan CFR yang disebut tidak terverifikasi dalam dokumen yang berhasil diambil"

    return _CFR_REFERENCE_PATTERN.sub(validate, answer or "")


def _strip_internal_source_markers(answer: str) -> str:
    """Remove model-generated internal file/chunk citation markers.

    Verified source links are rendered separately from Python retrieval metadata.
    Internal chunk IDs are not user-facing citations.
    """
    text = answer or ""
    patterns = (
        # Examples: 【...us_dot_refunds.txt:7】, 【us_cfr_260_6_refunds.txt:4】
        r"【[^】]*(?:\.txt|\.jsonl)(?::\d+)?[^】]*】",
        # Examples: 【source: ...】, 【Source 1†...】
        r"【\s*(?:source|sumber|citation|reference)\s*\d*[^】]*】",
        r"\[\s*(?:source|sumber|citation|reference)\s*\d+[^\]]*\]",
    )
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]{2,}", " ", text)
    # Remove whitespace left before punctuation after deleting a citation marker.
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    # Remove unsupported model-generated superscript footnote markers (e.g. ¹, ²).
    text = re.sub(r"(?<=[\w.,;:])[⁰¹²³⁴⁵⁶⁷⁸⁹]+", "", text)
    return text.strip()


def _strip_llm_source_footer(answer: str) -> str:
    """Remove model-generated bibliography; sources are rendered from Python metadata."""
    heading = re.compile(
        r"(?im)^[ \t]*(?:#{1,6}[ \t]*)?(?:\*\*|__)?"
        r"(?:sumber yang digunakan|sumber yang dipakai|sources(?: used)?|references|daftar sumber)"
        r"(?:\*\*|__)?[ \t]*:?[ \t]*$"
    )
    match = heading.search(answer or "")
    if not match:
        return (answer or "").strip()
    prefix = (answer or "")[:match.start()].rstrip()
    # Avoid leaving a horizontal separator immediately before the removed bibliography.
    prefix = re.sub(r"(?:\r?\n)[ \t]*(?:---|\*\*\*)[ \t]*$", "", prefix).rstrip()
    return prefix


def _deduplicate_sources(sources: list[dict]) -> list[dict]:
    """Group retrieved chunks by source URL and preserve distinct supporting excerpts."""
    grouped: dict[str, dict] = {}
    order: list[str] = []

    for source in sources or []:
        if not isinstance(source, dict):
            continue
        raw_url = str(source.get("url", "") or "").strip()
        title = str(source.get("title", "Sumber") or "Sumber").strip()
        # Query strings/fragments do not create separate source pages for display.
        normalized_url = re.sub(r"[?#].*$", "", raw_url).rstrip("/").lower()
        key = normalized_url or title.lower()

        if key not in grouped:
            grouped[key] = {
                "title": title,
                "url": raw_url,
                "sections": [],
                "excerpts": [],
                "normalized_excerpts": set(),
                "chunk_ids": [],
                "score": source.get("score") if isinstance(source.get("score"), (int, float)) else None,
            }
            order.append(key)

        item = grouped[key]
        section = str(source.get("section", "") or "").strip()
        excerpt = str(source.get("excerpt", "") or "").strip()
        chunk_id = str(source.get("chunk_id", "") or "").strip()

        if section and section not in item["sections"]:
            item["sections"].append(section)
        if excerpt:
            normalized_excerpt = re.sub(r"\s+", " ", excerpt).strip().casefold()
            if normalized_excerpt not in item["normalized_excerpts"]:
                label = f"**{section}**\n" if section else ""
                item["excerpts"].append(label + excerpt)
                item["normalized_excerpts"].add(normalized_excerpt)
        if chunk_id and chunk_id not in item["chunk_ids"]:
            item["chunk_ids"].append(chunk_id)
        score = source.get("score")
        if isinstance(score, (int, float)) and (
            not isinstance(item["score"], (int, float)) or score > item["score"]
        ):
            item["score"] = score

    result: list[dict] = []
    for key in order:
        item = grouped[key]
        result.append({
            "title": item["title"],
            "url": item["url"],
            "section": " · ".join(item["sections"]),
            "excerpt": "\n\n".join(item["excerpts"]),
            "chunk_ids": item["chunk_ids"],
            "score": item["score"],
        })
    return result