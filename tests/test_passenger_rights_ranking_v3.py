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


def test_actual_arrival_query_prioritizes_real_section_260_2_definition(tmp_path, monkeypatch):
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
            "section": "REGULATORY DEFINITION",
            "url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-260/section-260.2",
            "text": "For domestic itineraries, due to a carrier-made delay or change, the consumer is scheduled to arrive 3 hours or more after the original scheduled arrival. Actual landing time alone is not the scheduled-arrival test.",
        },
    ])
    results = retrieval.retrieve_passenger_rights(
        "Penerbangan saya tiba aktual 3 jam 30 menit terlambat. Saya tidak tahu apakah maskapai mengubah jadwal kedatangan. Apakah saya berhak refund?"
    )
    assert results
    assert "260.2" in (results[0]["title"] + results[0]["url"])
    assert "original scheduled arrival" in results[0]["excerpt"].lower()


def test_fully_refundable_unused_query_prioritizes_specific_dot_section(tmp_path, monkeypatch):
    _write_index(tmp_path, monkeypatch, [
        {
            "id": "generic-260-6-deadline:1",
            "title": "eCFR — 14 CFR §260.6 Refund Obligation",
            "section": "RELATED DEFINITION AND PAYMENT TIMING",
            "url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-260/section-260.6",
            "text": "A prompt refund is due within 7 business days for credit card payments and 20 calendar days for other payments.",
        },
        {
            "id": "dot-fully-refundable:1",
            "title": "U.S. DOT — Refunds (local educational summary)",
            "section": "FULLY REFUNDABLE TICKET",
            "url": "https://www.transportation.gov/individuals/aviation-consumer-protection/refunds",
            "text": "Consumers who purchase fully refundable tickets are entitled to a refund when they do not use the purchased ticket.",
        },
    ])
    results = retrieval.retrieve_passenger_rights(
        "Saya membeli tiket fully refundable dan memutuskan tidak menggunakan tiket tersebut. Apakah berhak refund?"
    )
    assert results
    assert results[0]["section"] == "FULLY REFUNDABLE TICKET"
    assert "do not use the purchased ticket" in results[0]["excerpt"].lower()


def test_no_response_alternative_query_prioritizes_automatic_refund_trigger(tmp_path, monkeypatch):
    _write_index(tmp_path, monkeypatch, [
        {
            "id": "generic-refund:1",
            "title": "U.S. DOT — Refunds",
            "section": "REFUND AFTER CANCELLATION",
            "url": "https://www.transportation.gov/individuals/aviation-consumer-protection/refunds",
            "text": "Consumers may receive a refund when an airline cancels a flight.",
        },
        {
            "id": "dot-auto-refund:1",
            "title": "U.S. DOT — Refunds",
            "section": "IF THE PASSENGER REJECTS AN OFFER OR DOES NOT RESPOND",
            "url": "https://www.transportation.gov/individuals/aviation-consumer-protection/refunds",
            "text": "If the consumer does not respond to an alternative transportation offer and does not fly on the offered alternative, the flight departs without the consumer. Refunds are due within 7 business days for credit card payments or 20 calendar days for other payment methods after the significantly delayed or changed flight or the alternative flight departs from its destination.",
        },
    ])
    results = retrieval.retrieve_passenger_rights(
        "Saya tidak merespons tawaran penerbangan alternatif dan tidak menggunakan penerbangan alternatif tersebut. Apakah refund otomatis?"
    )
    assert results
    assert "DOES NOT RESPOND" in results[0]["section"]
    assert "flight departs without the consumer" in results[0]["excerpt"].lower()
