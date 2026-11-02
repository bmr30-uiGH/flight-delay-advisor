from src.agent import _deterministic_passenger_rights_override


def _source(title: str, section: str, excerpt: str) -> dict:
    return {
        "title": title,
        "section": section,
        "url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-260/section-260.6",
        "excerpt": excerpt,
    }


def test_cancelled_flight_with_no_alternative_or_compensation_uses_260_6_a_2_i():
    question = (
        "Maskapai membatalkan penerbangan domestik Amerika Serikat saya dan "
        "tidak menawarkan penerbangan alternatif, voucher, kredit, atau "
        "kompensasi apa pun. Apakah refund otomatis?"
    )
    sources = [
        _source(
            "eCFR â€” 14 CFR Â§260.6",
            "AUTOMATIC REFUNDS â€” NO OFFER",
            "14 CFR Â§260.6(a)(2)(i): A flight is canceled and a consumer is not offered "
            "an alternative flight or any voucher, credit, or other form of compensation "
            "by the air carrier or foreign air carrier pursuant to paragraph (c).",
        ),
        _source(
            "eCFR â€” 14 CFR Â§260.6",
            "NON-REFUNDABLE TICKET SCOPE",
            "A covered carrier that is the merchant of record must provide a full and "
            "prompt refund to a consumer that holds a nonrefundable ticket on a scheduled "
            "flight to, from, or within the United States.",
        ),
    ]

    answer = _deterministic_passenger_rights_override(question, sources)

    assert answer
    assert "Â§260.6(a)(2)(i)" in answer
    assert "otomatis" in answer.lower()
    assert "tidak perlu" in answer.lower()
    assert "Â§260.6(a)(2)(iii)(A)" in answer
    assert "tidak perlu menunggu" in answer.lower()


def test_no_response_to_voucher_uses_260_6_a_2_iii_b_not_alternative_flight_trigger():
    question = (
        "Maskapai membatalkan penerbangan domestik Amerika Serikat saya dan "
        "menawarkan voucher, tetapi saya tidak merespons sampai tanggal "
        "penerbangan semula dijadwalkan berangkat. Bagaimana aturan refund?"
    )
    sources = [
        _source(
            "eCFR â€” 14 CFR Â§260.6",
            "NO RESPONSE TO VOUCHER OR CREDIT",
            "14 CFR Â§260.6(a)(2)(iii)(B): A consumer does not respond to an offer of a "
            "voucher, credit, or other form of compensation by the date on which the "
            "cancelled flight was scheduled to depart or the date that the significantly "
            "delayed or changed flight departs.",
        ),
        _source(
            "eCFR â€” 14 CFR Â§260.2",
            "PROMPT REFUND DEFINITION",
            "Prompt refund means refunds made within 7 business days after the earliest "
            "date the refund was requested as set forth in Â§260.6(a)(2) for credit card "
            "purchases and within 20 calendar days after the earliest date the refund was "
            "requested as set forth in Â§260.6(a)(2) for other forms of purchases.",
        ),
    ]

    answer = _deterministic_passenger_rights_override(question, sources)

    assert answer
    assert "Â§260.6(a)(2)(iii)(B)" in answer
    assert "penerbangan yang dibatalkan semula dijadwalkan berangkat" in answer.lower()
    assert "Â§260.6(a)(2)(iii)(A)" in answer
    assert "hanya berlaku pada no-response" in answer.lower()
    assert "7 business days" in answer
    assert "20 calendar days" in answer


def test_no_response_to_voucher_does_not_invent_refund_rule_without_source():
    question = (
        "Penerbangan saya dibatalkan, maskapai menawarkan voucher, dan saya "
        "tidak merespons sampai tanggal keberangkatan. Apakah refund otomatis?"
    )
    sources = [
        _source(
            "U.S. DOT â€” Refunds",
            "GENERAL REFUND INFORMATION",
            "Consumers may be entitled to a refund after a cancelled flight.",
        )
    ]

    answer = _deterministic_passenger_rights_override(question, sources)

    assert answer
    assert "belum dapat memastikan" in answer.lower()
    assert "Â§260.6(a)(2)(iii)(B)" in answer
    assert "tidak akan menggunakan tenggat keberangkatan alternatif" in answer.lower()
