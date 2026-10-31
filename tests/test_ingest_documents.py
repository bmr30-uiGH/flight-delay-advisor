from pathlib import Path

from src.ingest_documents import _document_chunks


def test_refund_chunks_keep_important_distinctions():
    path = Path(__file__).parents[1] / "documents" / "us_dot_refunds.txt"
    chunks = _document_chunks(path.read_text(encoding="utf-8"))
    combined = "\n".join(item["text"] for item in chunks).lower()
    assert chunks
    assert "late arrival" in combined
    assert "early departure" in combined
    assert "does not, by itself" in combined
    assert "alternative flight" in combined


def test_tarmac_doc_distinguishes_gate_and_aircraft():
    path = Path(__file__).parents[1] / "documents" / "us_dot_tarmac_delays.txt"
    chunks = _document_chunks(path.read_text(encoding="utf-8"))
    combined = "\n".join(item["text"] for item in chunks).lower()
    assert "waiting in the terminal or at a gate" in combined
    assert "aircraft" in combined


def test_chunks_are_section_labeled_and_not_cut_mid_word():
    path = Path(__file__).parents[1] / "documents" / "us_dot_refunds.txt"
    chunks = _document_chunks(path.read_text(encoding="utf-8"))
    assert all(item["section"] for item in chunks)
    assert all(not item["section"].startswith("SEARCH TERMS") for item in chunks)
    assert all(item["text"].strip() for item in chunks)
    assert all(not item["text"].endswith(("penump", "origina", "signifi")) for item in chunks)


def test_american_commitments_include_dot_accountability_and_controllable_scope():
    path = Path(__file__).parents[1] / "documents" / "us_dot_airline_commitments_american.txt"
    chunks = _document_chunks(path.read_text(encoding="utf-8"))
    combined = "\n".join(item["text"] for item in chunks).lower()
    assert "within the airline's control" in combined
    assert "dot will hold airlines accountable" in combined
    assert "3 hours or more" in combined
    assert "does not set a universal numeric 3-hour definition" in combined


def test_cfr_260_2_document_covers_significant_delay_definition():
    path = Path(__file__).parents[1] / "documents" / "us_cfr_260_2_significant_delay.txt"
    chunks = _document_chunks(path.read_text(encoding="utf-8"))
    combined = "\n".join(item["text"] for item in chunks).lower()
    assert "14 cfr §260.2" in combined
    assert "3 hours" in combined
    assert "6 hours" in combined


def test_cfr_260_6_document_covers_refund_obligation():
    path = Path(__file__).parents[1] / "documents" / "us_cfr_260_6_refunds.txt"
    chunks = _document_chunks(path.read_text(encoding="utf-8"))
    combined = "\n".join(item["text"] for item in chunks).lower()
    assert "14 cfr §260.6" in combined
    assert "full and prompt refund" in combined
    assert "alternative flight" in combined
