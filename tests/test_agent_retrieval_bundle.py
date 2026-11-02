from src import agent


def test_refund_ota_query_merges_focused_sources_without_duplicates(monkeypatch):
    question = (
        "Saya membeli tiket non-refundable untuk penerbangan domestik Amerika Serikat. "
        "Maskapai membatalkan penerbangan dan menawarkan alternatif, tetapi saya "
        "secara eksplisit menolak alternatif tersebut. Saya membayar dengan kartu "
        "kredit. Saya ingin mengetahui tanggung jawab maskapai dan OTA."
    )
    primary = {
        "title": "DOT Refunds", "section": "NON-REFUNDABLE TICKET",
        "url": "https://dot.example/refunds", "excerpt": "Primary refund evidence",
    }
    rejection = {
        "title": "eCFR 260.6", "section": "EXPLICIT REJECTION",
        "url": "https://ecfr.example/260.6-rejection", "excerpt": "Explicit rejection automatic refund evidence",
    }
    agent_source = {
        "title": "eCFR 399.80", "section": "TICKET AGENT REFUND",
        "url": "https://ecfr.example/399.80", "excerpt": "Ticket agent upon request and carrier notice under 260.6(d)",
    }
    transfer_source = {
        "title": "eCFR 260.6", "section": "CARRIER TRANSFER OF FUNDS",
        "url": "https://ecfr.example/260.6-transfer", "excerpt": "Under §260.6(e), carrier must promptly transfer funds to the ticket agent.",
    }
    calls = []

    def fake_retrieval(query):
        calls.append(query)
        if "§260.6(e)" in query:
            return [transfer_source, primary]
        if "§399.80(l)" in query:
            return [agent_source, primary]
        if "§260.6(a)(2)(ii)" in query:
            return [rejection, primary]
        return [primary]

    monkeypatch.setattr(agent, "retrieve_passenger_rights", fake_retrieval)
    result = agent._run_tool("search_passenger_rights", {"query": question}, None)

    assert result["ok"]
    assert len(calls) == 4
    assert len(result["sources"]) == 4
    assert {source["section"] for source in result["sources"]} == {
        "NON-REFUNDABLE TICKET", "EXPLICIT REJECTION",
        "TICKET AGENT REFUND", "CARRIER TRANSFER OF FUNDS",
    }


def test_tarmac_query_uses_primary_retrieval_without_refund_expansion(monkeypatch):
    question = (
        "Pesawat domestik tertahan di landasan selama 3 jam. "
        "Apa kewajiban maskapai mengenai air minum dan kesempatan turun?"
    )

    source = {
        "title": "DOT Tarmac",
        "section": "TARMAC DELAY",
        "url": "https://dot.example/tarmac",
        "excerpt": "Tarmac delay evidence",
    }

    calls = []

    def fake_retrieval(query):
        calls.append(query)
        return [source]

    monkeypatch.setattr(agent, "retrieve_passenger_rights", fake_retrieval)

    result = agent._run_tool(
        "search_passenger_rights",
        {"query": question},
        None,
    )

    assert result["ok"]
    assert calls == [question]
    assert result["sources"] == [source]


def test_merge_passenger_rights_context_preserves_original_ota_intent():
    original = (
        "Tiket non-refundable, pembatalan, menolak alternatif, kartu kredit, "
        "dan pertanyaan tanggung jawab OTA sebagai merchant of record."
    )
    model_query = "What is the explicit rejection refund deadline?"
    merged = agent._merge_passenger_rights_context(original, model_query)
    assert original in merged
    assert model_query in merged
    assert "OTA" in merged
    assert "merchant of record" in merged


def test_merge_passenger_rights_context_avoids_duplicate_original_question():
    original = "Apa hak refund dan tanggung jawab OTA?"
    assert agent._merge_passenger_rights_context(original, original) == original
