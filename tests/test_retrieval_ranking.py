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



def test_tarmac_delay_query_returns_dedicated_tarmac_document_not_refund_definition(
    tmp_path, monkeypatch
):
    tarmac_url = "https://www.transportation.gov/individuals/aviation-consumer-protection/tarmac-delays"
    _index_test_chunks(tmp_path, monkeypatch, [
        {
            "id": "tarmac:food-water",
            "title": "U.S. DOT — Tarmac Delays (local educational summary)",
            "section": "FOOD AND WATER",
            "url": tarmac_url,
            "text": (
                "Under 14 CFR 259.4(c)(4), adequate food and potable water are required by two hours; "
                "the pilot may withhold service only for safety/security. Working toilets, comfortable "
                "cabin temperatures, and medical attention if needed are required. For departing flights, "
                "the aircraft must begin returning by three hours; for arriving flights, passengers must "
                "be offered deplaning before the delay exceeds three hours."
            ),
        },
        {
            "id": "tarmac:deplane",
            "title": "U.S. DOT — Tarmac Delays (local educational summary)",
            "section": "OPPORTUNITY TO DEPLANE",
            "url": tarmac_url,
            "text": (
                "For a domestic departing flight, the airline must begin moving the aircraft to a "
                "location where passengers may safely deplane before three hours, subject to safety, "
                "security, or air-traffic-control exceptions."
            ),
        },
        {
            "id": "ecfr-259-4",
            "title": "eCFR — 14 CFR §259.4 Tarmac Delay Contingency Plan",
            "section": "DOMESTIC DEPARTING AND ARRIVING FLIGHTS",
            "url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-259/section-259.4",
            "text": (
                "14 CFR §259.4(c)(1) requires a domestic arriving flight to offer an opportunity to deplane "
                "before three hours. For a departing flight, §259.4(c)(3)(i) allows return to a suitable "
                "disembarkation point to begin no later than three hours after the main door is closed. "
                "The deplaning exceptions include pilot safety/security determination and ATC advice that "
                "return would significantly disrupt airport operations."
            ),
        },
        {
            "id": "ecfr-260-2",
            "title": "eCFR — 14 CFR §260.2 Significant Delay Definition",
            "section": "LATE ARRIVAL",
            "url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-260/section-260.2",
            "text": (
                "For domestic itineraries, the consumer is scheduled to arrive 3 hours or more after "
                "the original scheduled arrival time. This is the significant late arrival definition."
            ),
        },
        {
            "id": "dot-refund",
            "title": "U.S. DOT — Refunds",
            "section": "AUTOMATIC REFUNDS",
            "url": "https://www.transportation.gov/individuals/aviation-consumer-protection/refunds",
            "text": "A refund is due within 7 business days after a consumer rejects an alternative flight offer.",
        },
    ])

    results = retrieval.retrieve_passenger_rights(
        "Pesawat domestik tertahan di landasan 3 jam 20 menit. Apa kewajiban maskapai "
        "terkait snack, air minum, toilet, dan kesempatan penumpang turun?"
    )

    assert results
    allowed_urls = {
        tarmac_url,
        "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-259/section-259.4",
    }
    assert results
    assert all(item["url"] in allowed_urls for item in results)
    assert any("FOOD AND WATER" in item["section"] for item in results)
    assert any("part-259/section-259.4" in item["url"] for item in results)
    assert not any("260.2" in item["title"] for item in results)



def test_explicit_rejection_query_prioritizes_subsection_two_and_deadline_chunks(
    tmp_path, monkeypatch
):
    _index_test_chunks(tmp_path, monkeypatch, [
        {
            "id": "refund-general",
            "title": "U.S. DOT — Refunds",
            "section": "GENERAL REFUNDS",
            "url": "https://www.transportation.gov/individuals/aviation-consumer-protection/refunds",
            "text": "A consumer with a nonrefundable ticket may qualify for a full refund when a flight is cancelled and the consumer does not travel.",
        },
        {
            "id": "explicit-rejection",
            "title": "eCFR — 14 CFR §260.6 Refund Obligation",
            "section": "AUTOMATIC REFUNDS — EXPLICIT REJECTION",
            "url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-260/section-260.6",
            "text": "14 CFR §260.6(a)(2)(ii) covers explicit rejection of a changed flight or rebooking on an alternative flight; a full refund is due automatically.",
        },
        {
            "id": "rejection-deadline",
            "title": "U.S. DOT — Refunds",
            "section": "TENGGAT SETELAH PENOLAKAN EKSPLISIT",
            "url": "https://www.transportation.gov/individuals/aviation-consumer-protection/refunds",
            "text": "A refund is due within 7 business days after you reject the offer if paid by credit card, and within 20 calendar days after you reject the offer for another payment method.",
        },
        {
            "id": "no-response",
            "title": "U.S. DOT — Refunds",
            "section": "AUTOMATIC REFUND — NO RESPONSE TO AN ALTERNATIVE-FLIGHT OFFER",
            "url": "https://www.transportation.gov/individuals/aviation-consumer-protection/refunds",
            "text": "Under 14 CFR §260.6(a)(2)(iii)(A), a consumer does not respond and the alternative flight departs without the consumer.",
        },
    ])

    results = retrieval.retrieve_passenger_rights(
        "Tiket non-refundable saya dibatalkan dan saya secara eksplisit menolak penerbangan alternatif. "
        "Saya membayar dengan kartu kredit. Apakah refund otomatis dan kapan tenggatnya?"
    )
    sections = [item["section"] for item in results]
    assert sections[0] == "TENGGAT SETELAH PENOLAKAN EKSPLISIT" or sections[0] == "AUTOMATIC REFUNDS — EXPLICIT REJECTION"
    assert "AUTOMATIC REFUND — NO RESPONSE TO AN ALTERNATIVE-FLIGHT OFFER" not in sections[:3]
    assert any("§260.6(a)(2)(ii)" in item["excerpt"] for item in results)
    assert any("7 business days" in item["excerpt"] for item in results)
