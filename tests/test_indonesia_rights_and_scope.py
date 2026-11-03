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
