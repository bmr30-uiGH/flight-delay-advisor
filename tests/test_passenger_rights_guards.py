from src.agent import (
    SYSTEM_INSTRUCTIONS,
    _actual_arrival_only_with_unknown_schedule,
    _continued_downgrade_question,
    _deterministic_passenger_rights_override,
    _strip_internal_source_markers,
)


def test_actual_arrival_with_unknown_revised_schedule_is_detected():
    question = (
        "Penerbangan domestik AS saya tiba aktual 3 jam 30 menit terlambat. "
        "Saya tidak tahu apakah maskapai mengubah jadwal kedatangan. Apakah saya berhak refund?"
    )
    assert _actual_arrival_only_with_unknown_schedule(question)


def test_actual_arrival_guard_does_not_trigger_when_revised_schedule_is_known():
    question = (
        "Actual arrival was 3.5 hours late, but the revised scheduled arrival was 4 hours later. "
        "Am I eligible for a refund?"
    )
    assert not _actual_arrival_only_with_unknown_schedule(question)


def test_actual_arrival_guard_gives_conditional_answer_from_retrieved_evidence():
    question = (
        "Tiba aktual 3 jam 30 menit terlambat, saya tidak tahu apakah maskapai mengubah jadwal. "
        "Apakah saya berhak refund?"
    )
    sources = [{
        "title": "eCFR — 14 CFR §260.2 (local summary)",
        "section": "LATE ARRIVAL — SCHEDULED ARRIVAL, NOT ACTUAL LANDING TIME",
        "excerpt": (
            "The consumer is scheduled to arrive at least 3 hours after the original scheduled arrival. "
            "Actual landing time is not a substitute for revised scheduled arrival."
        ),
    }]
    answer = _deterministic_passenger_rights_override(question, sources)
    assert answer
    assert "Belum dapat dipastikan" in answer
    assert "actual arrival" in answer.lower()
    assert "revised scheduled arrival" in answer.lower()


def test_downgrade_continued_travel_is_detected():
    question = "Maskapai menurunkan kelas bisnis ke ekonomi, tetapi saya tetap menggunakan penerbangan."
    assert _continued_downgrade_question(question)


def test_downgrade_guard_does_not_invent_processing_deadline():
    question = "Maskapai menurunkan kelas bisnis ke ekonomi, tetapi saya tetap menggunakan penerbangan. Apakah refund penuh atau selisih tarif?"
    sources = [{
        "title": "U.S. DOT — Refunds (local educational summary)",
        "section": "INVOLUNTARY DOWNGRADE — CONTINUED TRAVEL AND FARE DIFFERENCE",
        "excerpt": (
            "If the downgraded passenger continues to travel, the passenger is not entitled to a refund "
            "of the full airfare, but the airline must refund the difference between the original fare "
            "and the downgraded fare. The source does not specify a processing deadline for this fare difference."
        ),
    }]
    answer = _deterministic_passenger_rights_override(question, sources)
    assert answer
    assert "selisih" in answer.lower()
    assert "tidak menetapkan tenggat khusus" in answer.lower()
    assert "tenggat khusus untuk pembayaran selisih tarif downgrade" in answer.lower()
    assert "saya tidak akan mengarang tenggat" in answer.lower()


def test_agent_prompt_prevents_actual_arrival_substitution():
    prompt = SYSTEM_INSTRUCTIONS.lower()
    assert "jangan membandingkan actual landing" in prompt
    assert "revised scheduled arrival" in prompt


def test_agent_prompt_prevents_unsupported_downgrade_deadline():
    prompt = SYSTEM_INSTRUCTIONS.lower()
    assert "tenggat 7 business days/20 calendar days" in prompt


def test_model_footnote_markers_are_removed():
    answer = _strip_internal_source_markers("Minta selisih tarif.¹ Tenggatnya disebut di atas.²")
    assert "¹" not in answer
    assert "²" not in answer



def test_unused_fully_refundable_ticket_guard_does_not_invent_processing_deadline():
    question = "Saya membeli tiket fully refundable dan memutuskan tidak menggunakan tiket tersebut. Apakah saya berhak refund?"
    sources = [{
        "title": "U.S. DOT — Refunds (local educational summary)",
        "section": "FULLY REFUNDABLE TICKET",
        "excerpt": "Consumers who purchase fully refundable tickets are entitled to a refund when they do not use the purchased ticket.",
    }]
    answer = _deterministic_passenger_rights_override(question, sources)
    assert answer
    assert "berhak atas refund" in answer.lower()
    assert "tidak menetapkan deadline pemrosesan khusus" in answer.lower()
    assert "7 business days" in answer.lower()
    assert "20 calendar days" in answer.lower()
    assert "ancillary" in answer.lower()


def test_no_response_alternative_guard_checks_departure_without_consumer_and_deadlines():
    question = (
        "Penerbangan saya mengalami significant delay. Saya tidak merespons tawaran penerbangan alternatif "
        "dan tidak menggunakan penerbangan alternatif tersebut. Bagaimana automatic refund?"
    )
    sources = [{
        "title": "eCFR — 14 CFR §260.6 (local summary)",
        "section": "AUTOMATIC REFUNDS",
        "excerpt": (
            "A consumer does not respond to an offer of a significantly delayed or changed flight or an alternative flight "
            "and the flight departs without the consumer. A prompt refund is due within 7 business days for credit card "
            "payments and 20 calendar days for other payment methods."
        ),
    }]
    answer = _deterministic_passenger_rights_override(question, sources)
    assert answer
    assert "berangkat tanpa konsumen" in answer.lower()
    assert "7 business days" in answer.lower()
    assert "20 calendar days" in answer.lower()
    assert "bukan karena diam saja" in answer.lower()


def test_no_response_alternative_guard_does_not_claim_refund_if_departure_trigger_not_retrieved():
    question = (
        "Saya tidak merespons tawaran penerbangan alternatif dan tidak menggunakan penerbangan tersebut. Apakah refund otomatis?"
    )
    sources = [{
        "title": "U.S. DOT — Refunds",
        "section": "GENERAL REFUNDS",
        "excerpt": "A refund may be available for a significantly delayed flight.",
    }]
    answer = _deterministic_passenger_rights_override(question, sources)
    assert answer is None


def test_agent_prompt_limits_deadlines_for_voluntarily_unused_fully_refundable_ticket():
    prompt = SYSTEM_INSTRUCTIONS.lower()
    assert "jangan otomatis menerapkan tenggat 7 business days/20 calendar days" in prompt
    assert "ancillary service fees" in prompt


def test_agent_prompt_distinguishes_no_response_offer_types_and_trigger_dates():
    prompt = SYSTEM_INSTRUCTIONS.lower()
    assert "flight yang ditawarkan berangkat tanpa konsumen" in prompt
    assert "§260.6(a)(2)(iii)(b)" in prompt
    assert "jangan campur kedua trigger ini" in prompt


_EXPLICIT_REJECTION_QUESTION = (
    "Saya membeli tiket non-refundable untuk penerbangan domestik di Amerika Serikat. "
    "Maskapai membatalkan penerbangan dan menawarkan penerbangan alternatif, tetapi "
    "saya secara eksplisit menolak alternatif tersebut pada 1 November 2026. "
    "Saya membayar tiket menggunakan kartu kredit."
)

_EXPLICIT_REJECTION_SOURCES = [
    {
        "title": "eCFR — 14 CFR §260.6 Refund Obligation (local summary)",
        "section": "SCOPE AND NON-REFUNDABLE TICKETS",
        "excerpt": (
            "14 CFR §260.6(a)(1) specifically addresses a consumer holding a nonrefundable ticket "
            "on a scheduled flight to, from, or within the United States. A covered carrier that is "
            "the merchant of record must provide a full and prompt refund for a cancelled flight "
            "when the consumer chooses not to fly on the affected flight or accept an alternative flight."
        ),
    },
    {
        "title": "eCFR — 14 CFR §260.6 Refund Obligation (local summary)",
        "section": "AUTOMATIC REFUNDS — EXPLICIT REJECTION",
        "excerpt": (
            "14 CFR §260.6(a)(2)(ii) covers explicit rejection: a consumer rejects the significantly "
            "delayed or changed flight or rebooking on an alternative flight. The refund is due automatically."
        ),
    },
    {
        "title": "U.S. DOT — Refunds (local educational summary)",
        "section": "TENGGAT SETELAH PENOLAKAN EKSPLISIT",
        "excerpt": (
            "A refund is due within 7 business days after you reject the offer if paid by credit card, "
            "and within 20 calendar days after you reject the offer for another payment method."
        ),
    },
    {
        "title": "eCFR — 14 CFR §399.80(l)",
        "section": "TICKET AGENT REFUND OBLIGATION",
        "excerpt": (
            "A ticket agent that is the merchant of record must provide a prompt refund upon request. "
            "For credit-card purchases, a prompt refund is within 7 business days of the ticket agent "
            "receiving information from a carrier as specified in §260.6(d); for other methods, 20 calendar "
            "days after the refund becomes due. The original form of payment must be used unless the "
            "consumer agrees to a cash-equivalent method."
        ),
    },
    {
        "title": "eCFR — 14 CFR §260.6",
        "section": "TICKET AGENT NOTICE AND TRANSFER",
        "excerpt": (
            "After receiving a consumer refund request through the ticket agent, the carrier must inform "
            "the ticket agent without delay whether the consumer is eligible. If the agent is responsible "
            "but does not possess the funds, the carrier holding the funds must promptly transfer them."
        ),
    },
]


def test_explicit_rejection_uses_subsection_ii_and_correct_mor_explanation():
    answer = _deterministic_passenger_rights_override(
        _EXPLICIT_REJECTION_QUESTION, _EXPLICIT_REJECTION_SOURCES
    )
    assert answer
    assert "§260.6(a)(2)(ii)" in answer
    assert "no-response §260.6(a)(2)(iii)(A)" in answer
    assert "bukan pada penolakan eksplisit" in answer
    assert "7 hari kerja" in answer
    assert "penerbit kartu/bank bukan merchant" in answer.lower()
    assert "merchant of record anda belum dapat dipastikan" in answer.lower()
    assert "atas permintaan konsumen" in answer.lower()
    assert "7 business days sejak agent menerima informasi" in answer.lower()
    assert "20 calendar days" in answer.lower()


def test_explicit_rejection_guard_does_not_assume_voucher_if_user_did_not_say():
    answer = _deterministic_passenger_rights_override(
        _EXPLICIT_REJECTION_QUESTION, _EXPLICIT_REJECTION_SOURCES
    )
    assert answer
    assert "jika anda tidak menerima kompensasi pengganti refund" in answer.lower()


def test_explicit_rejection_guard_does_not_override_when_voucher_was_accepted():
    question = _EXPLICIT_REJECTION_QUESTION + " Saya kemudian menerima voucher sebagai pengganti refund."
    answer = _deterministic_passenger_rights_override(question, _EXPLICIT_REJECTION_SOURCES)
    assert answer is None


def test_prompt_distinguishes_explicit_rejection_and_no_response_subsections():
    prompt = SYSTEM_INSTRUCTIONS.lower()
    assert "penolakan eksplisit" in prompt
    assert "§260.6(a)(2)(ii)" in prompt
    assert "§260.6(a)(2)(iii)(a) hanya untuk" in prompt
    assert "penerbit kartu kredit" in prompt
    assert "bukan merchant of record" in prompt


_TARMAC_QUESTION = (
    "Pesawat domestik di Amerika Serikat tertahan di landasan selama 3 jam 20 menit. "
    "Penumpang belum diberi kesempatan turun dan air minum mulai habis. "
    "Apa kewajiban snack, air, toilet, suhu kabin, pemberitahuan, dan kesempatan turun?"
)
_TARMAC_SOURCES = [
    {
        "title": "U.S. DOT — Tarmac Delays",
        "section": "FOOD AND POTABLE WATER — TWO HOURS",
        "url": "https://www.transportation.gov/individuals/aviation-consumer-protection/tarmac-delays",
        "excerpt": "14 CFR §259.4(c)(4) requires adequate food and potable water no later than two hours after the start. The pilot-in-command may withhold service only if safety or security considerations preclude it.",
    },
    {
        "title": "U.S. DOT — Tarmac Delays",
        "section": "TOILETS, MEDICAL ATTENTION, AND CABIN TEMPERATURE",
        "url": "https://www.transportation.gov/individuals/aviation-consumer-protection/tarmac-delays",
        "excerpt": "Operable lavatories, comfortable cabin temperatures, and adequate medical attention if needed during tarmac delay.",
    },
    {
        "title": "U.S. DOT — Tarmac Delays",
        "section": "STATUS NOTIFICATION — 30 MINUTES",
        "url": "https://www.transportation.gov/individuals/aviation-consumer-protection/tarmac-delays",
        "excerpt": "Notify passengers when tarmac delay exceeds 30 minutes; further updates may be provided as carrier deems appropriate.",
    },
    {
        "title": "U.S. DOT — Tarmac Delays",
        "section": "DOMESTIC THREE-HOUR LIMIT — DEPARTING AND ARRIVING FLIGHTS",
        "url": "https://www.transportation.gov/individuals/aviation-consumer-protection/tarmac-delays",
        "excerpt": "For a departing domestic flight, the carrier may begin return to a suitable disembarkation point no later than three hours after the main door is closed. For an arriving domestic flight, passengers must be offered the opportunity to deplane before the tarmac delay exceeds three hours. Exceptions: pilot safety/security determination or ATC advice that return would significantly disrupt airport operations.",
    },
    {
        "title": "eCFR — 14 CFR §259.4 Tarmac Delay Contingency Plan",
        "section": "DOMESTIC THREE-HOUR LIMIT AND EXCEPTIONS",
        "url": "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-259/section-259.4",
        "excerpt": "14 CFR §259.4(c)(1), §259.4(c)(3)(i), §259.4(c)(3)(ii), and §259.4(c)(3)(iii) distinguish domestic departing flights from arriving flights. For departing flights, return must begin no later than three hours after the main aircraft door closes. For arriving flights, passengers must be offered deplaning before the delay exceeds three hours, subject to pilot safety/security exceptions and air traffic control advice.",
    },
]

def test_tarmac_guard_distinguishes_departing_and_arriving_flights():
    answer = _deterministic_passenger_rights_override(_TARMAC_QUESTION, _TARMAC_SOURCES)
    assert answer
    assert "Pesawat akan berangkat" in answer
    assert "Pesawat baru mendarat" in answer
    assert "bukan pengganti kewajiban untuk arriving flight" in answer

def test_tarmac_guard_keeps_deplaning_exceptions_separate_from_food_and_facilities():
    answer = _deterministic_passenger_rights_override(_TARMAC_QUESTION, _TARMAC_SOURCES)
    assert answer
    assert "pengecualian umum untuk snack/air" in answer
    assert "pilot-in-command menentukan" in answer
    assert "toilet" in answer.lower()
    assert "suhu kabin" in answer.lower()

def test_tarmac_guard_does_not_invent_fixed_30_minute_update_cycle():
    answer = _deterministic_passenger_rights_override(_TARMAC_QUESTION, _TARMAC_SOURCES)
    assert answer
    assert "tidak menetapkan update wajib setiap 30 menit" in answer

def test_tarmac_guard_requires_relevant_dot_source():
    sources = [{"title": "Generic delay info", "section": "Delay", "url": "https://example.org/delay", "excerpt": "A delay can occur."}]
    assert _deterministic_passenger_rights_override(_TARMAC_QUESTION, sources) is None

def test_prompt_correctly_limits_ticket_agent_automatic_refund_claims():
    prompt = SYSTEM_INSTRUCTIONS.lower()
    assert "atas permintaan" in prompt
    assert "jangan menyebut pemrosesan ota otomatis tanpa permintaan" in prompt
    assert "§399.80(l)" in prompt

def test_prompt_distinguishes_departing_arriving_tarmac_exceptions():
    prompt = SYSTEM_INSTRUCTIONS.lower()
    assert "departing flight" in prompt
    assert "arriving flight" in prompt
    assert "jangan menggeneralisasi pengecualian batas 3 jam" in prompt

def test_explicit_rejection_guard_accepts_short_alternative_wording():
    question = _EXPLICIT_REJECTION_QUESTION.replace(
        "menawarkan penerbangan alternatif",
        "menawarkan alternatif",
    )
    answer = _deterministic_passenger_rights_override(
        question, _EXPLICIT_REJECTION_SOURCES
    )
    assert answer
    assert "§260.6(a)(2)(ii)" in answer
    assert "no-response §260.6(a)(2)(iii)(A)" in answer
    assert "7 hari kerja" in answer
    assert "atas permintaan konsumen" in answer


def test_explicit_rejection_includes_260_6d_notice_without_260_6e_evidence():
    question = _EXPLICIT_REJECTION_QUESTION

    sources = _EXPLICIT_REJECTION_SOURCES[:4] + [{
        "title": "eCFR — 14 CFR §260.6",
        "section": "TICKET AGENT NOTICE",
        "excerpt": (
            "Under 14 CFR §260.6(d), after the consumer requests a refund "
            "through the ticket agent, the carrier must inform the agent "
            "without delay whether the consumer is eligible."
        ),
    }]

    answer = _deterministic_passenger_rights_override(question, sources)

    assert answer
    assert "§260.6(d)" in answer
    assert "carrier memberi tahu agent tanpa penundaan" in answer.lower()


def test_explicit_rejection_guard_explains_260_6e_when_transfer_evidence_is_retrieved():
    transfer_source = {
        "title": "eCFR — 14 CFR §260.6",
        "section": "CARRIER TRANSFER OF FUNDS — §260.6(e)",
        "excerpt": (
            "Under §260.6(e), if the ticket agent is responsible for the refund "
            "but does not possess the consumer's funds, the carrier holding the "
            "funds must promptly transfer them to the ticket agent."
        ),
    }
    answer = _deterministic_passenger_rights_override(
        _EXPLICIT_REJECTION_QUESTION,
        _EXPLICIT_REJECTION_SOURCES + [transfer_source],
    )
    assert answer
    assert "§260.6(e)" in answer
    assert "mentransfer dana tersebut" in answer.lower()


def test_ota_mor_deadline_scope_is_not_generalized_to_standalone_fully_refundable_claims():
    question = (
        "Penerbangan domestik Amerika Serikat saya dibatalkan dan saya "
        "secara eksplisit menolak penerbangan alternatif. Jelaskan hak "
        "untuk tiket non-refundable dan fully refundable, siapa yang "
        "memproses refund jika maskapai atau OTA adalah merchant of record, "
        "serta tenggatnya. Jangan asumsikan jenis tiket atau MOR."
    )
    sources = list(_EXPLICIT_REJECTION_SOURCES) + [
        {
            "title": "U.S. DOT â€” Refunds",
            "section": "FULLY REFUNDABLE TICKET",
            "excerpt": (
                "Consumers who purchase fully refundable tickets are entitled "
                "to a refund when they do not use the purchased ticket."
            ),
        },
        {
            "title": "eCFR â€” 14 CFR §399.80",
            "section": "MERCHANT OF RECORD DEFINITION",
            "excerpt": (
                "Merchant of record means the entity responsible for processing "
                "payments by consumers for airfare, as shown in the consumer's "
                "financial charge statements such as debit or credit card "
                "charge statements."
            ),
        },
    ]

    answer = _deterministic_passenger_rights_override(question, sources)

    assert answer
    lowered = answer.lower()
    assert "§399.80(l)" in answer
    assert ("situasi yang dirujuk oleh " + chr(167) + "260.6(a) dan (b)") in answer
    assert "non-refundable" in lowered
    assert "fully refundable" in lowered
    assert "jangan menyatakan bahwa tenggat statutory carrier" in lowered
    assert "7 business days" in lowered
    assert "20 calendar days" in lowered
