from src.retrieval import _expand_query


def test_refund_query_expansion_includes_correct_cfr_refund_section():
    query = _expand_query("What is my refund right for a domestic flight?").lower()
    assert "14 cfr part 260" in query
    assert "section 260.6" in query
    assert "refund obligation" in query
