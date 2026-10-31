from src.agent import (
    _deduplicate_sources,
    _strip_internal_source_markers,
    _strip_llm_source_footer,
)


def test_internal_chunk_markers_are_removed_from_answer():
    answer = (
        "Ambang refund domestik diatur dalam 14 CFR §260.2 "
        "【...us_cfr_260_2_significant_delay.txt:1】 dan refund diatur "
        "dalam §260.6 【us_cfr_260_6_refunds.txt:2】."
    )
    cleaned = _strip_internal_source_markers(answer)
    assert ".txt:" not in cleaned
    assert "【" not in cleaned
    assert "14 CFR §260.2" in cleaned
    assert "§260.6" in cleaned


def test_bold_source_heading_and_footer_are_removed():
    answer = (
        "Refund depends on the scheduled arrival threshold.\n\n"
        "---\n\n**Sumber yang digunakan**\n\n"
        "- [eCFR](https://example.com)\n- *\n"
    )
    assert _strip_llm_source_footer(answer) == (
        "Refund depends on the scheduled arrival threshold."
    )


def test_source_footer_in_english_is_removed():
    answer = "The refund rule is in Part 260.\n\n### References\n- fake source"
    assert _strip_llm_source_footer(answer) == "The refund rule is in Part 260."


def test_sources_are_grouped_by_url_and_chunks_are_preserved():
    sources = [
        {
            "title": "U.S. DOT Refunds — local summary",
            "url": "https://www.transportation.gov/refunds",
            "section": "REFUND DEADLINE",
            "excerpt": "Credit card refunds: 7 business days.",
            "chunk_id": "refunds:1",
            "score": 0.12,
        },
        {
            "title": "U.S. DOT Refunds — local summary",
            "url": "https://www.transportation.gov/refunds/",
            "section": "REFUND FORM",
            "excerpt": "Refunds go to the original form of payment.",
            "chunk_id": "refunds:2",
            "score": 0.16,
        },
        {
            "title": "eCFR §260.6",
            "url": "https://www.ecfr.gov/current/260.6",
            "section": "CARRIER OBLIGATION",
            "excerpt": "A full and prompt refund is required when the conditions apply.",
            "chunk_id": "ecfr:1",
            "score": 0.15,
        },
    ]

    grouped = _deduplicate_sources(sources)

    assert len(grouped) == 2
    dot_source = grouped[0]
    assert dot_source["section"] == "REFUND DEADLINE · REFUND FORM"
    assert "7 business days" in dot_source["excerpt"]
    assert "original form of payment" in dot_source["excerpt"]
    assert dot_source["chunk_ids"] == ["refunds:1", "refunds:2"]
    assert grouped[1]["url"].startswith("https://www.ecfr.gov/")


def test_duplicate_excerpt_is_not_repeated():
    sources = [
        {
            "title": "DOT",
            "url": "https://example.com/dot",
            "section": "REFUNDS",
            "excerpt": "A refund is due when the passenger declines travel.",
        },
        {
            "title": "DOT",
            "url": "https://example.com/dot",
            "section": "REFUNDS",
            "excerpt": "A refund is due when the passenger declines travel.",
        },
    ]
    grouped = _deduplicate_sources(sources)
    assert len(grouped) == 1
    assert grouped[0]["excerpt"].count("A refund is due") == 1
