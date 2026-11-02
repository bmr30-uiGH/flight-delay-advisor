from src.retrieval import _expand_query


def test_refund_query_expansion_includes_correct_cfr_refund_section():
    query = _expand_query("What is my refund right for a domestic flight?").lower()
    assert "14 cfr part 260" in query
    assert "section 260.6" in query
    assert "refund obligation" in query


def test_fully_refundable_query_expands_to_unused_ticket_guidance():
    query = _expand_query("Saya membeli tiket fully refundable tetapi tidak jadi berangkat.").lower()
    assert "fully refundable ticket" in query
    assert "unused ticket" in query
    assert "nonrefundable ticket" in query


def test_nonrefundable_query_expands_to_section_260_6_and_no_travel_condition():
    query = _expand_query("My non-refundable ticket was changed and I do not want to fly.").lower()
    assert "nonrefundable ticket" in query
    assert "section 260.6(a)(1)" in query
    assert "chooses not to fly" in query


def test_travelled_query_expands_to_no_full_refund_and_downgrade_fare_difference():
    query = _expand_query("Saya sudah terbang setelah jadwal berubah.").lower()
    assert "consumer traveled" in query
    assert "not entitled full ticket price refund" in query
    assert "downgrade fare difference" in query


def test_actual_arrival_query_expands_to_scheduled_arrival_comparison():
    query = _expand_query("My actual arrival was 3.5 hours late.").lower()
    assert "actual landing time" in query
    assert "revised scheduled arrival" in query
    assert "original scheduled arrival" in query


def test_rebooking_query_expands_to_acceptance_and_automatic_refund_conditions():
    query = _expand_query("What if I rejected the alternative flight rebooking?").lower()
    assert "alternative transportation" in query
    assert "rejects rebooking" in query
    assert "automatic refund" in query


def test_already_flew_query_expands_to_travelled_flight_outcome():
    query = _expand_query("I already flew on the alternative flight; can I get a refund?").lower()
    assert "consumer traveled" in query
    assert "generally not entitled full ticket price refund" in query
    assert "downgrade fare difference" in query


def test_indonesian_actual_arrival_phrases_expand_to_scheduled_arrival_rule():
    query = _expand_query("Penerbangan saya tiba aktual 3 jam terlambat, apakah refund?").lower()
    assert "actual landing time" in query
    assert "revised scheduled arrival" in query
    assert "original scheduled arrival" in query


def test_indonesian_downgrade_phrases_expand_to_fare_difference_rule():
    query = _expand_query("Maskapai menurunkan kelas bisnis menjadi ekonomi dan saya tetap terbang.").lower()
    assert "involuntary downgrade" in query
    assert "difference between the original fare" in query
    assert "downgraded fare" in query



def test_fully_refundable_unused_ticket_query_avoids_generic_nonrefundable_deadline_expansion():
    query = _expand_query("Saya membeli tiket fully refundable dan memutuskan tidak menggunakan tiket tersebut.").lower()
    assert "fully refundable ticket" in query
    assert "not used" in query or "not use" in query
    assert "refund obligation covered carrier merchant of record" not in query


def test_no_response_alternative_query_expands_to_departure_without_consumer_condition():
    query = _expand_query("Saya tidak merespons tawaran penerbangan alternatif dan tidak menggunakan penerbangan itu.").lower()
    assert "does not respond" in query
    assert "flight departs without consumer" in query
    assert "automatic refund" in query



def test_indonesian_tarmac_phrases_expand_to_tarmac_delay_rules():
    query = _expand_query(
        "Pesawat tertahan di landasan selama 3 jam 20 menit, bagaimana aturan makanan dan air?"
    ).lower()
    assert "tarmac delay" in query
    assert "drinking water within two hours" in query
    assert "opportunity to deplane" in query


def test_ota_query_expands_to_ticket_agent_upon_request_rule():
    query = _expand_query("Saya beli tiket melalui OTA, siapa yang proses refund?").lower()
    assert "399.80(l)" in query
    assert "upon request" in query
    assert "ticket agent" in query

def test_tarmac_query_expands_departure_arrival_and_exceptions():
    query = _expand_query("Pesawat domestik tertahan di landasan, kapan penumpang bisa turun?").lower()
    assert "departing flight" in query
    assert "arriving flight" in query
    assert "air traffic control" in query
