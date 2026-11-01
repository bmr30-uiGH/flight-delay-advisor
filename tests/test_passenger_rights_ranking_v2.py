import json

from src import retrieval


def _write_index(tmp_path, monkeypatch, records):
    index_path = tmp_path / "chunks.jsonl"
    index_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    monkeypatch.setattr(retrieval, "CHUNKS_PATH", index_path)
    retrieval._load_chunks.cache_clear()
    return index_path


def test_actual_arrival_query_prioritizes_section_260_2_late_arrival_chunk(tmp_path, monkeypatch):
    _write_index(tmp_path, monkeypatch, [
        {
            "id": "generic-refund:1",
            "title": "U.S. DOT — Refunds",
            "section": "REFUND PAYMENT METHOD AND DEADLINES",
            "url": "https://www.transportation.gov/individuals/aviation-consumer-protection/refunds",
            "text": "Refunds are returned to the original form of payment within the prompt refund deadline.",
        },
        {
            "id": "ecfr-260-2-late-arrival:1",
            "title": "eCFR — 14 CFR §260.2 Definitions",
            "section": "LATE ARRIVAL — SCHEDULED ARRIVAL, NOT ACTUAL LANDING TIME",
            "url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-260/section-260.2",
            "text": "For domestic itineraries, the consumer is scheduled to arrive 3 hours or more after the original scheduled arrival due to a carrier-made delay or change. Actual landing time alone is not a substitute for revised scheduled arrival.",
        },
    ])
    results = retrieval.retrieve_passenger_rights(
        "Penerbangan saya tiba aktual 3 jam 30 menit terlambat. Saya tidak tahu apakah maskapai mengubah jadwal kedatangan. Apakah saya berhak refund?"
    )
    assert results
    assert "260.2" in (results[0]["title"] + results[0]["url"])
    assert "original scheduled arrival" in results[0]["excerpt"].lower()


def test_downgrade_query_prioritizes_dot_fare_difference_chunk(tmp_path, monkeypatch):
    _write_index(tmp_path, monkeypatch, [
        {
            "id": "generic-refund:1",
            "title": "eCFR — 14 CFR §260.6 Refund Obligation",
            "section": "REFUND PAYMENT DEADLINE",
            "url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-260/section-260.6",
            "text": "A prompt ticket refund is due within 7 business days for credit card payments or 20 calendar days for other payments.",
        },
        {
            "id": "dot-downgrade:1",
            "title": "U.S. DOT — Refunds (local educational summary)",
            "section": "INVOLUNTARY DOWNGRADE — CONTINUED TRAVEL AND FARE DIFFERENCE",
            "url": "https://www.transportation.gov/individuals/aviation-consumer-protection/refunds",
            "text": "If the downgraded passenger continues to travel on the flight, the passenger is not entitled to a full airfare refund, but the airline must refund the difference between the original fare and the downgraded fare.",
        },
    ])
    results = retrieval.retrieve_passenger_rights(
        "Maskapai menurunkan kelas bisnis menjadi ekonomi dan saya tetap terbang. Apakah refund penuh atau selisih tarif?"
    )
    assert results
    assert "INVOLUNTARY DOWNGRADE" in results[0]["section"]
    assert "difference between the original fare" in results[0]["excerpt"].lower()
