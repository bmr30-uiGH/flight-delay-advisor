from src.agent import _format_flight_summary_context, _remove_unsupported_cfr_citations, _strip_llm_source_footer


def test_flight_summary_uses_python_calendar_day_and_origin_local_time():
    summary = {
        "carrier": "American Airlines",
        "origin": "John F. Kennedy International Airport",
        "origin_code": "JFK",
        "destination": "Los Angeles International Airport",
        "destination_code": "LAX",
        "flight_date": "2026-11-15",
        "departure_time": "08:44",
        "distance": 2475,
    }
    context = _format_flight_summary_context(summary)
    assert "Minggu, dihitung oleh Python" in context
    assert "08:44 waktu lokal di bandara asal JFK" in context
    assert "jangan ubah menjadi WIB" in context


def test_invalid_flight_date_does_not_get_a_guessed_weekday():
    context = _format_flight_summary_context({"flight_date": "not-a-date"})
    assert "hari tidak dapat diverifikasi" in context


def test_unsupported_cfr_citation_is_replaced():
    answer = "Refund diatur oleh 14 CFR §250.13 untuk keterlambatan empat jam."
    sources = [{"excerpt": "Rujukan refund: 14 CFR §260.6."}]
    checked = _remove_unsupported_cfr_citations(answer, sources)
    assert "250.13" not in checked
    assert "tidak terverifikasi" in checked


def test_cfr_citation_found_in_retrieved_source_is_preserved():
    answer = "Rujukan refund adalah 14 CFR §260.6."
    sources = [{"excerpt": "14 CFR §260.6 mengatur refund penerbangan."}]
    assert _remove_unsupported_cfr_citations(answer, sources) == answer


def test_model_generated_sources_footer_is_removed():
    answer = "Refund threshold depends on scheduled arrival.\n\nSumber yang digunakan\n-\n*\n-\n*"
    assert _strip_llm_source_footer(answer) == "Refund threshold depends on scheduled arrival."


def test_answer_without_sources_heading_is_preserved():
    answer = "Refund threshold depends on scheduled arrival."
    assert _strip_llm_source_footer(answer) == answer
