
from src.agent import _deterministic_passenger_rights_override


def refund_sources():
    return [
        {
            "title": "U.S. DOT Refunds",
            "section": "Explicit rejection and deadlines",
            "url": "https://www.transportation.gov/individuals/aviation-consumer-protection/refunds",
            "excerpt": (
                "For a cancelled flight, 14 CFR 260.6(a)(2)(ii) covers "
                "explicit rejection of an alternative flight. The refund "
                "is due automatically after the consumer rejects the offer. "
                "For credit-card payments, 7 business days after the consumer "
                "rejects the offer; for other payment methods, 20 calendar "
                "days after rejection. Consumers who purchase a fully "
                "refundable ticket are entitled to a refund when they do "
                "not use the purchased ticket."
            ),
        },
        {
            "title": "eCFR 14 CFR 260.6",
            "section": "Non-refundable tickets and merchant of record",
            "url": "https://www.ecfr.gov/current/title-14/part-260/section-260.6",
            "excerpt": (
                "14 CFR 260.6(a)(1) specifically addresses a consumer "
                "holding a nonrefundable ticket on a scheduled flight "
                "to, from, or within the United States. A covered carrier "
                "that is the merchant of record must provide a full and "
                "prompt refund of airfare, taxes and ancillary fees for "
                "a cancelled flight when the consumer chooses not to fly "
                "on the affected flight or accept rebooking on an alternative "
                "flight and does not accept a voucher, credit or other "
                "compensation as an explicit alternative to the refund. "
                "Merchant of record is the carrier or ticket agent shown "
                "on the consumer's financial charge statement."
            ),
        },
        {
            "title": "eCFR 14 CFR 399.80",
            "section": "Ticket agent refund obligation",
            "url": "https://www.ecfr.gov/current/title-14/part-399/section-399.80",
            "excerpt": (
                "14 CFR 399.80(l) requires a ticket agent that is the "
                "merchant of record to provide a prompt refund upon "
                "consumer request. For credit-card purchases, prompt "
                "means 7 business days after the ticket agent receives "
                "information from the carrier under 14 CFR 260.6(d). "
                "For other payment methods, prompt means 20 calendar "
                "days after the refund becomes due."
            ),
        },
    ]


GENERAL_QUESTION = (
    "Jika penerbangan domestik Amerika Serikat saya dibatalkan "
    "dan saya menolak penerbangan alternatif, apa hak refund saya?"
)


def test_unspecified_fare_class_is_explained_as_separate_branches():
    answer = _deterministic_passenger_rights_override(
        GENERAL_QUESTION,
        refund_sources(),
    )

    assert answer is not None
    lowered = answer.lower()

    assert "jenis tiket belum disebutkan" in lowered
    assert "non-refundable" in lowered
    assert "fully refundable" in lowered
    assert "tidak boleh disamakan" in lowered


def test_nonrefundable_fare_uses_section_260_6_a_1():
    question = (
        "Saya memiliki tiket non-refundable. Penerbangan domestik "
        "AS dibatalkan dan saya menolak alternatif. Apa hak saya?"
    )

    answer = _deterministic_passenger_rights_override(
        question,
        refund_sources(),
    )

    assert answer is not None
    assert "§260.6(a)(1)" in answer
    assert "non-refundable" in answer.lower()


def test_fully_refundable_ticket_uses_separate_dot_guidance():
    question = (
        "Saya membeli tiket fully refundable, penerbangan domestik "
        "AS dibatalkan dan tiket tidak saya gunakan setelah menolak "
        "penerbangan alternatif. Bagaimana refund-nya?"
    )

    answer = _deterministic_passenger_rights_override(
        question,
        refund_sources(),
    )

    assert answer is not None
    lowered = answer.lower()

    assert "fully refundable" in lowered
    assert "tidak boleh" in lowered or "terpisah" in lowered
    assert "deadline" in lowered or "tenggat" in lowered


def test_general_fare_guard_requires_retrieved_legal_evidence():
    incomplete_sources = [
        {
            "title": "Generic refund information",
            "section": "General",
            "url": "https://example.com/refunds",
            "excerpt": (
                "A consumer may request a refund after a cancellation."
            ),
        }
    ]

    answer = _deterministic_passenger_rights_override(
        GENERAL_QUESTION,
        incomplete_sources,
    )

    assert answer is None


def test_general_fare_guard_does_not_assume_ota_is_merchant_of_record():
    question = (
        "Saya membeli tiket melalui OTA, penerbangan domestik AS "
        "dibatalkan dan saya menolak alternatif. Apa hak refund?"
    )

    answer = _deterministic_passenger_rights_override(
        question,
        refund_sources(),
    )

    # The dedicated OTA guard has priority. This general fare guard
    # must not intercept OTA-specific questions.
    assert answer is None
