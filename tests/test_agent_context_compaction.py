
from src.agent import _compact_passenger_rights_result_for_llm


def test_compact_result_limits_sources_and_excerpt_lengths():
    sources = []

    for index in range(12):
        sources.append({
            "title": f"Generic source {index}",
            "section": f"Generic section {index}",
            "url": f"https://example.com/{index}",
            "excerpt": ("Generic legal content " * 100),
        })

    result = _compact_passenger_rights_result_for_llm({
        "ok": True,
        "query": "refund question",
        "message": "Retrieved",
        "sources": sources,
    })

    assert result["ok"] is True
    assert len(result["sources"]) <= 5

    for source in result["sources"]:
        assert len(source["excerpt"]) <= 450
        assert source["title"]
        assert source["url"]


def test_compact_result_preserves_high_priority_ota_evidence():
    sources = []

    for index in range(8):
        sources.append({
            "title": f"Generic source {index}",
            "section": f"Generic section {index}",
            "url": f"https://example.com/{index}",
            "excerpt": "Generic refund information.",
        })

    sources.extend([
        {
            "title": "eCFR 14 CFR 399.80",
            "section": "Ticket agent refund upon request",
            "url": "https://www.ecfr.gov/399.80",
            "excerpt": (
                "14 CFR 399.80(l) ticket agent merchant of record "
                "must provide prompt refund upon consumer request."
            ),
        },
        {
            "title": "eCFR 14 CFR 260.6",
            "section": "Carrier fund transfer",
            "url": "https://www.ecfr.gov/260.6",
            "excerpt": (
                "14 CFR 260.6(e) carrier must promptly transfer "
                "funds to ticket agent."
            ),
        },
        {
            "title": "eCFR 14 CFR 260.6",
            "section": "Carrier notice to ticket agent",
            "url": "https://www.ecfr.gov/260.6",
            "excerpt": (
                "14 CFR 260.6(d) carrier must inform ticket agent "
                "without delay whether consumer is eligible."
            ),
        },
        {
            "title": "eCFR 14 CFR 260.6",
            "section": "Explicit rejection",
            "url": "https://www.ecfr.gov/260.6",
            "excerpt": (
                "14 CFR 260.6(a)(2)(ii) explicit rejection "
                "of an alternative flight."
            ),
        },
        {
            "title": "U.S. DOT Refunds",
            "section": "Refund deadlines",
            "url": "https://www.transportation.gov/refunds",
            "excerpt": (
                "7 business days for credit cards and "
                "20 calendar days for other payment methods."
            ),
        },
    ])

    result = _compact_passenger_rights_result_for_llm({
        "ok": True,
        "query": "OTA refund",
        "sources": sources,
    })

    evidence = " ".join(
        source["excerpt"].lower()
        for source in result["sources"]
    )

    assert len(result["sources"]) <= 5
    assert "399.80(l)" in evidence
    assert "260.6(e)" in evidence
    assert "260.6(d)" in evidence
    assert "260.6(a)(2)(ii)" in evidence


def test_compact_result_keeps_status_when_no_sources():
    result = _compact_passenger_rights_result_for_llm({
        "ok": False,
        "query": "test",
        "message": "No evidence found",
        "sources": [],
    })

    assert result["ok"] is False
    assert result["sources"] == []
    assert result["message"] == "No evidence found"


def test_compact_result_does_not_modify_original_sources():
    original_excerpt = "Legal evidence " * 200

    original = {
        "ok": True,
        "query": "test",
        "sources": [{
            "title": "Test",
            "section": "Section",
            "url": "https://example.com",
            "excerpt": original_excerpt,
        }],
    }

    _compact_passenger_rights_result_for_llm(original)

    assert original["sources"][0]["excerpt"] == original_excerpt
