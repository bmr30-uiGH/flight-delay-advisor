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
