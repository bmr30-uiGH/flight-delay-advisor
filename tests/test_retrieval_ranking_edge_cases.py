"""Tests for retrieval ranking edge cases and scoring behavior."""
import json
import pytest
from src import retrieval


def _index_test_chunks(tmp_path, monkeypatch, records):
    path = tmp_path / "passenger_rights_chunks.jsonl"
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )
    monkeypatch.setattr(retrieval, "CHUNKS_PATH", path)
    retrieval._load_chunks.cache_clear()
    retrieval._get_vectorizer_and_matrix.cache_clear()
    return path


class TestRetrievalRankingEdgeCases:
    """Edge case tests for retrieval scoring and ranking."""

    def test_actual_arrival_boost_requires_section_260_2_in_title_or_url(self, tmp_path, monkeypatch):
        """Boost should only apply when §260.2 is in title or URL."""
        records = [
            {
                "id": "refund-general:1",
                "title": "General Refund Info",
                "section": "REFUNDS",
                "url": "https://example.org/refund",
                "text": "Original scheduled arrival revised scheduled arrival consumer scheduled to arrive 3 hours domestic 6 hours international.",
            },
            {
                "id": "260-2-definition:1",
                "title": "eCFR — 14 CFR §260.2 Definitions",
                "section": "LATE ARRIVAL",
                "url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-260/section-260.2",
                "text": "Original scheduled arrival revised scheduled arrival consumer scheduled to arrive 3 hours domestic 6 hours international.",
            },
        ]
        _index_test_chunks(tmp_path, monkeypatch, records)

        results = retrieval.retrieve_passenger_rights(
            "Actual arrival was 3.5 hours late. I don't know the revised scheduled arrival."
        )

        # The §260.2 chunk should rank higher due to boost
        assert results
        assert "260.2" in results[0]["title"] or "260.2" in results[0]["url"]

    def test_actual_arrival_boost_requires_scheduled_arrival_definition(self, tmp_path, monkeypatch):
        """Boost requires the chunk to contain the scheduled arrival definition (3 hours/three hours)."""
        records = [
            {
                "id": "260-2-no-definition:1",
                "title": "eCFR — 14 CFR §260.2 Definitions",
                "section": "SCOPE",
                "url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-260/section-260.2",
                "text": "This section defines terms used in this part. Carrier means an air carrier.",
            },
            {
                "id": "260-2-with-definition:1",
                "title": "eCFR — 14 CFR §260.2 Definitions",
                "section": "LATE ARRIVAL",
                "url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-260/section-260.2",
                "text": "The consumer is scheduled to arrive at least 3 hours after the original scheduled arrival for domestic flights.",
            },
        ]
        _index_test_chunks(tmp_path, monkeypatch, records)

        results = retrieval.retrieve_passenger_rights(
            "Tiba aktual 3 jam terlambat, tidak tahu jadwal revisi"
        )

        # The chunk with definition should rank higher
        assert results
        assert "3 hours" in results[0]["excerpt"] or "three hours" in results[0]["excerpt"]

    def test_fully_refundable_boost_requires_transportation_gov_url(self, tmp_path, monkeypatch):
        """Fully refundable boost should only apply to transportation.gov URLs."""
        records = [
            {
                "id": "refund-transportation:1",
                "title": "U.S. DOT — Refunds",
                "section": "FULLY REFUNDABLE TICKET",
                "url": "https://www.transportation.gov/airconsumer/refunds",
                "text": "Fully refundable ticket not used consumer does not use purchased ticket refund guidance.",
            },
            {
                "id": "refund-other:1",
                "title": "Airline Refund Policy",
                "section": "FULLY REFUNDABLE TICKET",
                "url": "https://airline.com/refunds",
                "text": "Fully refundable ticket not used consumer does not use purchased ticket refund guidance.",
            },
        ]
        _index_test_chunks(tmp_path, monkeypatch, records)

        results = retrieval.retrieve_passenger_rights(
            "Saya membeli tiket fully refundable dan memutuskan tidak menggunakan tiket tersebut"
        )

        # transportation.gov chunk should rank higher
        assert results
        assert "transportation.gov" in results[0]["url"]

    def test_fully_refundable_boost_requires_fully_refundable_section(self, tmp_path, monkeypatch):
        """Boost requires the chunk to have the fully refundable section or unused ticket text."""
        records = [
            {
                "id": "transport-general:1",
                "title": "U.S. DOT — Refunds",
                "section": "GENERAL REFUNDS",
                "url": "https://www.transportation.gov/airconsumer/refunds",
                "text": "Refunds are available for cancelled flights under 14 CFR 260.",
            },
            {
                "id": "transport-specific:1",
                "title": "U.S. DOT — Refunds",
                "section": "FULLY REFUNDABLE TICKET",
                "url": "https://www.transportation.gov/airconsumer/refunds",
                "text": "Fully refundable ticket consumer does not use purchased ticket entitled to refund.",
            },
        ]
        _index_test_chunks(tmp_path, monkeypatch, records)

        results = retrieval.retrieve_passenger_rights(
            "Fully refundable ticket not used, can I get refund?"
        )

        # Specific section should rank higher
        assert results
        assert "FULLY REFUNDABLE TICKET" in results[0]["section"]

    def test_no_response_alt_boost_requires_automatic_refund_section(self, tmp_path, monkeypatch):
        """No-response boost requires automatic refund section with departure-without-consumer condition."""
        records = [
            {
                "id": "general-refund:1",
                "title": "General Refunds",
                "section": "REFUNDS",
                "url": "https://example.org/refund",
                "text": "Automatic refunds are available. Flight departs without the consumer. 7 business days 20 calendar days.",
            },
            {
                "id": "specific-auto:1",
                "title": "eCFR — 14 CFR §260.6",
                "section": "AUTOMATIC REFUNDS",
                "url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-260/section-260.6",
                "text": "If the passenger rejects an offer or does not respond, the flight departs without the consumer. 7 business days 20 calendar days.",
            },
        ]
        _index_test_chunks(tmp_path, monkeypatch, records)

        results = retrieval.retrieve_passenger_rights(
            "Tidak merespons tawaran penerbangan alternatif dan tidak menggunakan penerbangan itu"
        )

        # Specific automatic refund section should rank higher
        assert results
        assert "AUTOMATIC REFUNDS" in results[0]["section"]

    def test_downgrade_boost_requires_fare_difference_text(self, tmp_path, monkeypatch):
        """Downgrade boost requires 'difference between the original fare' or 'fare difference' in body."""
        records = [
            {
                "id": "downgrade-general:1",
                "title": "Involuntary Downgrade",
                "section": "DOWNGRADE",
                "url": "https://example.org/downgrade",
                "text": "Airlines may downgrade passengers involuntarily to a lower class of service.",
            },
            {
                "id": "downgrade-specific:1",
                "title": "U.S. DOT — Refunds",
                "section": "INVOLUNTARY DOWNGRADE",
                "url": "https://www.transportation.gov/airconsumer/refunds",
                "text": "Difference between the original fare and the downgraded fare must be refunded if passenger continues travel.",
            },
        ]
        _index_test_chunks(tmp_path, monkeypatch, records)

        results = retrieval.retrieve_passenger_rights(
            "Maskapai menurunkan kelas bisnis ke ekonomi dan saya tetap terbang"
        )

        # Specific fare difference chunk should rank higher
        assert results
        assert "difference between the original fare" in results[0]["excerpt"].lower()

    def test_score_threshold_filters_low_relevance(self, tmp_path, monkeypatch):
        """Results with score <= 0.005 should be filtered out."""
        records = [
            {
                "id": "relevant:1",
                "title": "Refund for Delay",
                "section": "REFUND",
                "url": "https://example.org/refund",
                "text": "Refund for significant delay cancellation nonrefundable ticket 14 CFR 260.6 consumer chooses not to fly.",
            },
            {
                "id": "irrelevant:1",
                "title": "Baggage Policy",
                "section": "BAGGAGE",
                "url": "https://example.org/baggage",
                "text": "Baggage fees apply for checked bags. Carry-on baggage allowed. Meal service not included.",
            },
        ]
        _index_test_chunks(tmp_path, monkeypatch, records)

        results = retrieval.retrieve_passenger_rights(
            "Refund for non-refundable ticket delay cancellation"
        )

        # Both may appear if scores > 0.005, but relevant should rank higher
        assert len(results) >= 1
        assert "Refund for Delay" in results[0]["title"]
        # Verify score threshold works by checking a very low score is filtered
        for r in results:
            assert r["score"] > 0.005

    def test_top_k_limit_respected(self, tmp_path, monkeypatch):
        """Top-k limit should be respected (max 6)."""
        records = [
            {
                "id": f"refund-{i}:1",
                "title": f"Refund Rule {i}",
                "section": "REFUND",
                "url": f"https://example.org/refund{i}",
                "text": f"Refund rule {i} for significant delay nonrefundable ticket cancellation.",
            }
            for i in range(10)
        ]
        _index_test_chunks(tmp_path, monkeypatch, records)

        results = retrieval.retrieve_passenger_rights(
            "Refund for non-refundable ticket delay",
            top_k=6
        )

        assert len(results) <= 6

    def test_top_k_clamped_to_max_6(self, tmp_path, monkeypatch):
        """top_k > 6 should be clamped to 6."""
        records = [
            {
                "id": f"refund-{i}:1",
                "title": f"Refund Rule {i}",
                "section": "REFUND",
                "url": f"https://example.org/refund{i}",
                "text": f"Refund rule {i} for significant delay.",
            }
            for i in range(10)
        ]
        _index_test_chunks(tmp_path, monkeypatch, records)

        results = retrieval.retrieve_passenger_rights(
            "Refund for non-refundable ticket delay",
            top_k=20
        )

        assert len(results) <= 6

    def test_empty_query_returns_empty(self, tmp_path, monkeypatch):
        """Empty or whitespace query should return empty list."""
        records = [
            {
                "id": "refund:1",
                "title": "Refund",
                "section": "REFUND",
                "url": "https://example.org/refund",
                "text": "Refund info.",
            },
        ]
        _index_test_chunks(tmp_path, monkeypatch, records)

        assert retrieval.retrieve_passenger_rights("") == []
        assert retrieval.retrieve_passenger_rights("   ") == []
        assert retrieval.retrieve_passenger_rights(None) == []

    def test_missing_chunks_file_returns_empty(self, tmp_path, monkeypatch):
        """Missing chunks file should return empty list without error."""
        nonexistent = tmp_path / "nonexistent.jsonl"
        monkeypatch.setattr(retrieval, "CHUNKS_PATH", nonexistent)
        retrieval._load_chunks.cache_clear()
        retrieval._get_vectorizer_and_matrix.cache_clear()

        assert retrieval.retrieve_passenger_rights("refund") == []

    def test_vectorizer_cache_invalidation_on_mtime_change(self, tmp_path, monkeypatch):
        """Changing file mtime should invalidate vectorizer cache."""
        records = [
            {
                "id": "refund:1",
                "title": "Refund",
                "section": "REFUND",
                "url": "https://example.org/refund",
                "text": "Refund for delay cancellation.",
            },
        ]
        path = _index_test_chunks(tmp_path, monkeypatch, records)

        # First call builds cache
        results1 = retrieval.retrieve_passenger_rights("refund delay")
        assert results1

        # Modify file mtime by rewriting
        import time
        time.sleep(0.01)
        path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")

        # Second call should rebuild (no error)
        results2 = retrieval.retrieve_passenger_rights("refund delay")
        assert results2

    def test_tfidf_vectorizer_error_handled_gracefully(self, tmp_path, monkeypatch):
        """TfidfVectorizer ValueError should return empty list, not crash."""
        records = [
            {
                "id": "refund:1",
                "title": "Refund",
                "section": "REFUND",
                "url": "https://example.org/refund",
                "text": "Refund.",
            },
        ]
        _index_test_chunks(tmp_path, monkeypatch, records)

        # This query may cause vectorizer issues, but should not crash
        results = retrieval.retrieve_passenger_rights("??" * 1000)
        assert isinstance(results, list)