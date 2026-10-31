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
