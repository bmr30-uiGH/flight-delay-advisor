from src import agent
from src.indonesia_rights import retrieve_indonesia_passenger_rights


def test_indonesia_delay_compensation_uses_indonesian_pm_89_sources():
    results = retrieve_indonesia_passenger_rights(
        "Berdasarkan peraturan resmi Indonesia, berapa kompensasi keterlambatan penerbangan 5 jam?"
    )
    assert results
    assert any("Rp300.000" in item["excerpt"] for item in results)
    assert all(
        item["url"].startswith(("https://peraturan.bpk.go.id/", "https://peraturan.go.id/"))
        for item in results
    )


def test_indonesia_jurisdiction_is_detected_explicitly():
    assert agent._uses_indonesia_law(
        "Berdasarkan peraturan resmi Indonesia, apa hak penumpang?"
    )


def test_agent_bundle_returns_only_indonesian_sources():
    sources = agent._retrieve_passenger_rights_bundle(
        "Penerbangan domestik Jakarta ke Bali terlambat 5 jam, apa kompensasinya?"
    )
    assert sources
    assert all(
        item["url"].startswith(("https://peraturan.bpk.go.id/", "https://peraturan.go.id/"))
        for item in sources
    )


def test_us_jurisdiction_does_not_get_routed_to_indonesia_corpus():
    assert not agent._uses_indonesia_law(
        "My domestic flight in the United States was delayed. What are my rights?"
    )


def test_unrelated_recipe_request_is_blocked_before_llm(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    answer, sources, tools, mode = agent.answer_user(
        "Berikan resep nasi goreng sederhana."
    )
    assert "dirancang untuk membantu" in answer.lower()
    assert sources == []
    assert tools == []
    assert mode == "scope guard (outside Flight Delay Advisor)"


def test_indonesia_delay_retrieval_prioritizes_relevant_pm89_sections():
    results = retrieve_indonesia_passenger_rights(
        "Berdasarkan peraturan Indonesia, apa hak penumpang ketika penerbangan "
        "domestik terlambat 4 jam?"
    )

    sections = [item["section"].lower() for item in results]

    assert sections
    assert "pasal 9" in sections[0]
    assert "pasal 3" in sections[1]
    assert any("pasal 10" in section for section in sections)
    assert not any("pasal 147" in section for section in sections)
    assert not any("catatan status" in section for section in sections)


def test_indonesia_delay_category_boundary_guard():
    cases = (
        (180, "kategori 3", "makanan berat"),
        (181, "kategori 4", "snack box"),
        (240, "kategori 4", "bukan kategori 5"),
        (241, "kategori 5", "rp300.000"),
    )

    for minutes, expected_category, expected_detail in cases:
        question = (
            "Berdasarkan peraturan resmi Indonesia, apa hak penumpang "
            f"ketika penerbangan domestik terlambat {minutes} menit?"
        )

        sources = retrieve_indonesia_passenger_rights(question, top_k=10)
        answer = agent._deterministic_passenger_rights_override(question, sources)

        assert answer is not None, f"Guard tidak aktif untuk {minutes} menit"
        assert expected_category in answer.lower(), answer
        assert expected_detail in answer.lower(), answer

        if minutes == 241:
            assert "241 menit" in answer.lower(), answer
            assert "4 jam" not in answer.lower(), answer
            assert "lebih dari 6 jam" in answer.lower(), answer


def test_indonesia_exact_four_hours_is_not_category_five():
    question = (
        "Berdasarkan peraturan resmi Indonesia, apa hak penumpang "
        "ketika penerbangan domestik terlambat tepat 4 jam?"
    )

    sources = retrieve_indonesia_passenger_rights(question, top_k=10)
    answer = agent._deterministic_passenger_rights_override(question, sources)

    assert answer is not None
    assert "240 menit" in answer
    assert "kategori 4" in answer.lower()
    assert "bukan kategori 5" in answer.lower()
    assert "lebih dari 6 jam" in answer.lower()


def _ota_guard_test_sources():
    return [{
        "title": "U.S. DOT and eCFR refund provisions",
        "section": "Ticket agent, merchant of record, and refund deadlines",
        "url": "https://www.ecfr.gov/current/title-14",
        "excerpt": (
            "14 CFR 260.6(a)(1) non-refundable ticket on a cancelled flight. "
            "14 CFR 260.6(d): after a consumer requests a refund through a "
            "ticket agent, the carrier must inform the ticket agent without "
            "delay whether the consumer is eligible for a refund. "
            "14 CFR 260.6(e): the carrier holding funds must promptly transfer "
            "funds to the ticket agent when required. "
            "14 CFR 399.80(l): a ticket agent that is the merchant of record "
            "must provide a prompt refund upon consumer request. "
            "The merchant of record is identified on the financial charge "
            "statement. For credit card purchases, 7 business days after the "
            "ticket agent receives information from the carrier. For other "
            "methods, 20 calendar days after the refund becomes due. "
            "Fully refundable tickets: consumer does not use the purchased ticket."
        ),
    }]


def test_ota_explicit_merchant_of_record_is_respected():
    question = (
        "My non-refundable ticket for a domestic U.S. flight was cancelled. "
        "I booked through an online travel agency (OTA), and the OTA is "
        "explicitly listed as the merchant of record on my payment statement. "
        "The airline offered an alternative flight, but I explicitly rejected "
        "it and did not accept any voucher or credit. I have not submitted a "
        "refund request to the OTA. Who must process my refund?"
    )

    answer = agent._deterministic_passenger_rights_override(
        question,
        _ota_guard_test_sources(),
    )

    assert answer is not None
    assert "ota adalah merchant of record" in answer.lower()
    assert "ajukan permintaan tertulis" in answer.lower()
    assert "7 business days" in answer.lower()
    assert "20 calendar days" in answer.lower()
    assert "nama mor belum disebutkan" not in answer.lower()
    assert "fully refundable" not in answer.lower()
    assert "jika maskapai adalah merchant of record" not in answer.lower()
    assert "?399.80" not in answer
    assert "?260.6" not in answer


def test_ota_merchant_of_record_is_not_assumed_without_evidence_from_user():
    question = (
        "My non-refundable U.S. flight was cancelled. I booked through an OTA, "
        "rejected the alternative flight, and did not accept a voucher or credit. "
        "Who processes the refund?"
    )

    answer = agent._deterministic_passenger_rights_override(
        question,
        _ota_guard_test_sources(),
    )

    assert answer is not None
    assert "nama mor belum disebutkan" in answer.lower()
