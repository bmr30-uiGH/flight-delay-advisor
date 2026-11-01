from pathlib import Path

from src.ingest_documents import _document_chunks, _is_heading


DOCUMENTS_DIR = Path(__file__).parents[1] / "documents"


def _combined(filename: str) -> str:
    path = DOCUMENTS_DIR / filename
    chunks = _document_chunks(path.read_text(encoding="utf-8"))
    assert chunks
    return "\n".join(item["text"] for item in chunks).lower()


def test_refund_chunks_keep_important_distinctions():
    combined = _combined("us_dot_refunds.txt")
    assert "late arrival" in combined
    assert "early departure" in combined
    assert "does not by itself establish" in combined
    assert "alternative flight" in combined
    assert "original scheduled arrival" in combined
    assert "revised scheduled arrival" in combined


def test_refund_document_distinguishes_actual_landing_from_revised_schedule():
    combined = _combined("us_dot_refunds.txt")
    assert "actual arrival" in combined
    assert "scheduled to arrive" in combined
    assert "3 hours or more" in combined
    assert "6 hours or more" in combined


def test_refund_document_distinguishes_nonrefundable_and_fully_refundable_tickets():
    combined = _combined("us_dot_refunds.txt")
    assert "nonrefundable ticket" in combined
    assert "fully refundable ticket" in combined
    assert "does not use the purchased ticket" in combined
    assert "14 cfr §260.6(a)" in combined


def test_refund_document_explains_travelled_and_downgrade_outcomes():
    combined = _combined("us_dot_refunds.txt")
    assert "chooses to take" in combined
    assert "not entitled to a refund of the full ticket price" in combined
    assert "difference between the original fare and the downgraded fare" in combined


def test_refund_document_keeps_correct_refund_deadlines():
    combined = _combined("us_dot_refunds.txt")
    assert "7 business days" in combined
    assert "20 calendar days" in combined
    assert "20 business days" not in combined
    assert "original form of payment" in combined


def test_tarmac_doc_distinguishes_gate_and_aircraft():
    combined = _combined("us_dot_tarmac_delays.txt")
    assert "waiting in the terminal or at a gate" in combined
    assert "aircraft" in combined


def test_chunks_are_section_labeled_and_not_cut_mid_word():
    path = DOCUMENTS_DIR / "us_dot_refunds.txt"
    chunks = _document_chunks(path.read_text(encoding="utf-8"))
    assert all(item["section"] for item in chunks)
    assert all(not item["section"].startswith("SEARCH TERMS") for item in chunks)
    assert all(item["text"].strip() for item in chunks)
    assert all(not item["text"].endswith(("penump", "origina", "signifi")) for item in chunks)


def test_american_commitments_include_dot_accountability_and_controllable_scope():
    combined = _combined("us_dot_airline_commitments_american.txt")
    assert "within the airline's control" in combined
    assert "dot will hold airlines accountable" in combined
    assert "3 hours or more" in combined
    assert "does not set a universal numeric 3-hour definition" in combined


def test_cfr_260_2_document_covers_scheduled_arrival_definition():
    combined = _combined("us_cfr_260_2_significant_delay.txt")
    assert "14 cfr §260.2" in combined
    assert "original scheduled arrival" in combined
    assert "revised scheduled arrival" in combined
    assert "actual landing time" in combined
    assert "3 hours" in combined
    assert "6 hours" in combined


def test_cfr_260_6_document_limits_paragraph_a_to_nonrefundable_tickets():
    combined = _combined("us_cfr_260_6_refunds.txt")
    assert "14 cfr §260.6(a)(1)" in combined
    assert "nonrefundable ticket" in combined
    assert "covered carrier" in combined
    assert "merchant of record" in combined
    assert "does not respond" in combined
    assert "section 260.6(a)(1)" not in combined  # Heading itself is metadata; body citation below must remain.
    assert "the text of 14 cfr §260.6(a)(1)" in combined
    assert "document type:" not in combined
    assert "source version:" not in combined
    assert "fully refundable tickets" in combined


def test_ingestion_recognizes_long_metadata_before_length_check():
    assert _is_heading("DOCUMENT TYPE: " + "curated educational summary " * 8)
    assert _is_heading("DOCUMENT: " + "A long legal document title " * 8)


def test_ingestion_recognizes_cfr_subsection_headings_with_lowercase_labels():
    assert _is_heading("SECTION 260.6(a)(1) — SCOPE AND NON-REFUNDABLE TICKETS")
    assert _is_heading("SECTION 260.6(d) AND (e) — TICKET AGENTS / MERCHANT OF RECORD")



def test_refunds_document_distinguishes_fully_refundable_deadline_from_automatic_refund_deadlines():
    path = Path(__file__).parents[1] / "documents" / "us_dot_refunds.txt"
    chunks = _document_chunks(path.read_text(encoding="utf-8"))
    combined = "\n".join(item["section"] + " " + item["text"] for item in chunks).lower()
    assert "fully refundable ticket — processing deadline limits" in combined
    assert "does not state a separate processing deadline" in combined
    assert "automatic refund — no response to an alternative-flight offer" in combined
    assert "flight departs without the consumer" in combined
    assert "20 calendar days" in combined
