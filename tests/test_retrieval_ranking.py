import json

from src import retrieval


def _index_test_chunks(tmp_path, monkeypatch, records):
    path = tmp_path / "passenger_rights_chunks.jsonl"
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    monkeypatch.setattr(retrieval, "CHUNKS_PATH", path)
    retrieval._load_chunks.cache_clear()
    return path


def test_retrieval_finds_nonrefundable_rule_for_rejected_alternative(tmp_path, monkeypatch):
    records = [
        {
            "id": "refund-nonrefundable:1",
            "title": "eCFR — 14 CFR §260.6 Refund Obligation (local summary)",
            "section": "SCOPE AND NON-REFUNDABLE TICKETS",
            "url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-260/section-260.6",
            "text": "14 CFR §260.6(a)(1) specifically addresses a consumer holding a nonrefundable ticket. A covered carrier that is merchant of record must provide a full and prompt refund when the consumer rejects the significantly delayed flight, does not accept an alternative flight, and does not accept voucher or credit in lieu of refund.",
        },
        {
            "id": "meal:1",
            "title": "Airline meal commitments",
            "section": "MEAL ASSISTANCE",
            "url": "https://example.org/meals",
            "text": "An airline may commit to provide a meal or meal voucher after a waiting period during a controllable delay.",
        },
        {
            "id": "tarmac:1",
            "title": "Tarmac delay rules",
            "section": "AIRCRAFT ON GROUND",
            "url": "https://example.org/tarmac",
            "text": "A tarmac delay concerns an aircraft on the ground when passengers do not have an opportunity to leave the aircraft.",
        },
    ]
    _index_test_chunks(tmp_path, monkeypatch, records)

    results = retrieval.retrieve_passenger_rights(
        "My non-refundable ticket was significantly changed, and I rejected the alternative flight. Can I receive a refund?"
    )

    assert results
    assert any("14 CFR §260.6(a)(1)" in item["excerpt"] for item in results)


def test_retrieval_finds_actual_vs_scheduled_arrival_rule(tmp_path, monkeypatch):
    records = [
        {
            "id": "arrival-test:1",
            "title": "eCFR — 14 CFR §260.2 Definitions (local summary)",
            "section": "LATE ARRIVAL",
            "url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-260/section-260.2",
            "text": "The late arrival test compares the original scheduled arrival with the revised scheduled arrival. For a domestic itinerary the consumer is scheduled to arrive 3 hours or more after the original scheduled arrival. Actual landing time alone is not a substitute for the revised scheduled arrival time.",
        },
        {
            "id": "hotel-test:1",
            "title": "Airline hotel commitments",
            "section": "OVERNIGHT STAYS",
            "url": "https://example.org/hotel",
            "text": "Some airlines commit to hotel accommodation and ground transport for overnight controllable delays.",
        },
    ]
    _index_test_chunks(tmp_path, monkeypatch, records)

    results = retrieval.retrieve_passenger_rights(
        "My actual arrival was very late. How do I compare actual arrival and revised scheduled arrival for a refund?"
    )

    assert results
    assert any("original scheduled arrival" in item["excerpt"].lower() for item in results)
