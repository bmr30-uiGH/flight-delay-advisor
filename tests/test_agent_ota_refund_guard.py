
from src.agent import _deterministic_passenger_rights_override


def ota_evidence_sources():
    return [
        {
            "title": "U.S. DOT Refunds",
            "section": "Explicit rejection and refund deadlines",
            "url": "https://www.transportation.gov/individuals/aviation-consumer-protection/refunds",
            "excerpt": (
                "14 CFR 260.6(a)(2)(ii) explicit rejection of an alternative "
                "flight. The automatic refund is due after the consumer "
                "rejects the offer. For credit cards, 7 business days after "
                "the consumer rejects the offer; for other payment methods, "
                "20 calendar days after rejection. "
                "14 CFR 260.6(a)(1) specifically addresses a nonrefundable "
                "ticket on covered scheduled flights."
            ),
        },
        {
            "title": "eCFR 14 CFR 260.6",
            "section": "Merchant of record and carrier duties",
            "url": "https://www.ecfr.gov/current/title-14/part-260/section-260.6",
            "excerpt": (
                "Merchant of record means the carrier or ticket agent shown "
                "on the consumer's financial charge statement as responsible "
                "for processing payment. Under 14 CFR 260.6(d), after the "
                "consumer requests a refund through the ticket agent, the "
                "carrier must inform the ticket agent without delay whether "
                "the consumer is eligible. Under 14 CFR 260.6(e), the carrier "
                "holding funds must promptly transfer them to the ticket "
                "agent when the agent is responsible but does not possess "
                "the consumer's funds."
            ),
        },
        {
            "title": "eCFR 14 CFR 399.80",
            "section": "Ticket agent refund obligation and deadline",
            "url": "https://www.ecfr.gov/current/title-14/part-399/section-399.80",
            "excerpt": (
                "14 CFR 399.80(l): a ticket agent that is the merchant of "
                "record must provide a prompt refund upon consumer request. "
                "For credit-card purchases, prompt means 7 business days "
                "after the ticket agent receives information from the carrier "
                "under 14 CFR 260.6(d). For other payment methods, prompt "
                "means 20 calendar days after the refund becomes due. "
                "The refund must be returned to the original form of payment."
            ),
        },
    ]


OTA_QUESTION = (
    "Jika penerbangan domestik Amerika Serikat saya dibatalkan dan "
    "saya menolak penerbangan alternatif, siapa yang wajib mengembalikan "
    "uang tiket jika saya membeli tiket melalui online travel agent (OTA)? "
    "Jelaskan kewajiban maskapai dan ticket agent."
)


def test_ota_guard_runs_without_fare_class_being_specified():
    answer = _deterministic_passenger_rights_override(
        OTA_QUESTION,
        ota_evidence_sources(),
    )

    assert answer is not None
    assert "merchant of record" in answer.lower()
    assert "399.80(l)" in answer
    assert "260.6(d)" in answer
    assert "non-refundable" in answer


def test_buying_through_ota_does_not_prove_ota_is_merchant_of_record():
    answer = _deterministic_passenger_rights_override(
        OTA_QUESTION,
        ota_evidence_sources(),
    )

    lowered = answer.lower()

    assert "tidak otomatis" in lowered
    assert "belum dapat dipastikan" in lowered
    assert "laporan transaksi" in lowered


def test_ota_refund_is_upon_request_and_deadlines_have_correct_triggers():
    answer = _deterministic_passenger_rights_override(
        OTA_QUESTION,
        ota_evidence_sources(),
    )

    lowered = answer.lower()

    assert "atas permintaan konsumen" in lowered
    assert "7 business days" in lowered
    assert "20 calendar days" in lowered
    assert "sejak ticket agent menerima informasi" in lowered
    assert "bukan otomatis dihitung sejak tanggal" in lowered
    assert "setelah penolakan" in lowered


def test_ota_guard_does_not_make_legal_claims_without_required_evidence():
    incomplete_sources = [
        {
            "title": "Generic refund information",
            "section": "General",
            "url": "https://example.com/refunds",
            "excerpt": "A consumer may request a refund after cancellation.",
        }
    ]

    answer = _deterministic_passenger_rights_override(
        OTA_QUESTION,
        incomplete_sources,
    )

    # Without the required ticket-agent and MOR evidence,
    # the specialized OTA legal override must not fabricate a rule.
    assert answer is None
