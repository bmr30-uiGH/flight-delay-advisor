"""Extended tests for guard functions in agent module."""
import json
from src.agent import (
    _actual_arrival_only_with_unknown_schedule,
    _continued_downgrade_question,
    _unused_fully_refundable_ticket_question,
    _no_response_alternative_refund_question,
    _deterministic_passenger_rights_override,
)


class TestActualArrivalOnlyWithUnknownSchedule:
    """Tests for _actual_arrival_only_with_unknown_schedule detector."""

    def test_detects_indonesian_actual_arrival_with_unknown_schedule(self):
        question = "Penerbangan tiba aktual 3 jam terlambat, tidak tahu jadwal revisi, apakah refund?"
        assert _actual_arrival_only_with_unknown_schedule(question)

    def test_detects_english_actual_arrival_with_unknown_schedule(self):
        question = "Actual arrival was 4 hours late, I don't know the revised schedule, am I eligible for refund?"
        assert _actual_arrival_only_with_unknown_schedule(question)

    def test_detects_kedatangan_aktual_phrasing(self):
        question = "Kedatangan aktual terlambat 3 jam 30 menit, tidak mengetahui jadwal revisi, berhak refund?"
        assert _actual_arrival_only_with_unknown_schedule(question)

    def test_detects_mendarat_aktual_phrasing(self):
        question = "Pesawat mendarat aktual jam 15:00, jadwal asli 11:30, tidak tahu apakah maskapai mengubah jadwal, pengembalian dana?"
        assert _actual_arrival_only_with_unknown_schedule(question)

    def test_does_not_trigger_when_revised_schedule_known(self):
        question = "Actual arrival 3.5 hours late, revised scheduled arrival was 4 hours late"
        assert not _actual_arrival_only_with_unknown_schedule(question)

    def test_does_not_trigger_without_refund_context(self):
        question = "Actual arrival was late, but I'm asking about meal vouchers"
        assert not _actual_arrival_only_with_unknown_schedule(question)

    def test_does_not_trigger_without_unknown_schedule_indication(self):
        question = "Actual arrival was 3 hours late, revised schedule shows 3.5 hours late"
        assert not _actual_arrival_only_with_unknown_schedule(question)

    def test_does_not_trigger_empty_question(self):
        assert not _actual_arrival_only_with_unknown_schedule("")
        assert not _actual_arrival_only_with_unknown_schedule(None)


class TestContinuedDowngradeQuestion:
    """Tests for _continued_downgrade_question detector."""

    def test_detects_indonesian_tetap_terbang(self):
        question = "Maskapai menurunkan kelas bisnis ke ekonomi, saya tetap terbang"
        assert _continued_downgrade_question(question)

    def test_detects_tetap_menggunakan_penerbangan(self):
        question = "Downgrade terjadi tapi saya tetap menggunakan penerbangan"
        assert _continued_downgrade_question(question)

    def test_detects_lanjut_perjalanan(self):
        question = "Diturunkan kelas tapi saya lanjut perjalanan tetap terbang"
        assert _continued_downgrade_question(question)

    def test_detects_english_continued_travel(self):
        question = "Involuntary downgrade but I continued to travel on the flight"
        assert _continued_downgrade_question(question)

    def test_detects_still_took_flight(self):
        question = "Downgraded to economy but still took the flight"
        assert _continued_downgrade_question(question)

    def test_does_not_trigger_without_downgrade_term(self):
        question = "Saya tetap terbang setelah delay"
        assert not _continued_downgrade_question(question)

    def test_does_not_trigger_without_continuance_term(self):
        question = "Maskapai menurunkan kelas bisnis ke ekonomi"
        assert not _continued_downgrade_question(question)

    def test_does_not_trigger_empty_question(self):
        assert not _continued_downgrade_question("")
        assert not _continued_downgrade_question(None)


class TestUnusedFullyRefundableTicketQuestion:
    """Tests for _unused_fully_refundable_ticket_question detector."""

    def test_detects_fully_refundable_unused(self):
        question = "Saya membeli tiket fully refundable tetapi tidak jadi terbang, apakah refund?"
        assert _unused_fully_refundable_ticket_question(question)

    def test_detects_tidak_menggunakan_tiket_refundable(self):
        question = "Tiket refundable tidak dipakai, apakah bisa refund?"
        assert _unused_fully_refundable_ticket_question(question)

    def test_detects_not_use_purchased_ticket(self):
        question = "I purchased a fully refundable ticket but did not use it"
        assert _unused_fully_refundable_ticket_question(question)

    def test_detects_unused_ticket(self):
        question = "Unused fully refundable ticket, did not use it, can I get refund?"
        assert _unused_fully_refundable_ticket_question(question)

    def test_does_not_trigger_without_fare_term(self):
        question = "Saya tidak menggunakan tiket, apakah refund?"
        assert not _unused_fully_refundable_ticket_question(question)

    def test_does_not_trigger_without_unused_term(self):
        question = "Saya membeli tiket fully refundable dan berangkat"
        assert not _unused_fully_refundable_ticket_question(question)

    def test_does_not_trigger_empty_question(self):
        assert not _unused_fully_refundable_ticket_question("")
        assert not _unused_fully_refundable_ticket_question(None)


class TestNoResponseAlternativeRefundQuestion:
    """Tests for _no_response_alternative_refund_question detector."""

    def test_detects_tidak_merespons_alternative_unused(self):
        question = "Saya tidak merespons tawaran penerbangan alternatif dan tidak menggunakan penerbangan itu, apakah refund otomatis?"
        assert _no_response_alternative_refund_question(question)

    def test_detects_tidak_menanggapi_alternative(self):
        question = "Tidak menanggapi tawaran alternative flight, tidak memakai penerbangan pengganti, refund?"
        assert _no_response_alternative_refund_question(question)

    def test_detects_english_no_response(self):
        question = "Did not respond to alternative flight offer and did not take the flight, automatic refund?"
        assert _no_response_alternative_refund_question(question)

    def test_detects_no_response_no_reply(self):
        question = "No response to alternative flight offer, never boarded the alternative, pengembalian dana?"
        assert _no_response_alternative_refund_question(question)

    def test_requires_refund_context(self):
        question = "Saya tidak merespons tawaran penerbangan alternatif dan tidak menggunakan penerbangan itu, apakah berhak refund?"
        assert _no_response_alternative_refund_question(question)

    def test_does_not_trigger_without_no_response_term(self):
        question = "Saya menerima tawaran penerbangan alternatif dan tidak menggunakan penerbangan itu"
        assert not _no_response_alternative_refund_question(question)

    def test_does_not_trigger_without_alternative_term(self):
        question = "Saya tidak merespons tawaran voucher dan tidak menggunakan penerbangan"
        assert not _no_response_alternative_refund_question(question)

    def test_does_not_trigger_without_unused_term(self):
        question = "Saya tidak merespons tawaran alternative flight tapi saya naik pesawatnya"
        assert not _no_response_alternative_refund_question(question)

    def test_does_not_trigger_empty_question(self):
        assert not _no_response_alternative_refund_question("")
        assert not _no_response_alternative_refund_question(None)


class TestDeterministicOverrideEdgeCases:
    """Tests for _deterministic_passenger_rights_override edge cases."""

    def test_actual_arrival_guard_without_section_260_2_evidence(self):
        """When evidence lacks §260.2 definition, guard should give conditional answer."""
        question = "Tiba aktual 3 jam terlambat, tidak tahu jadwal revisi. Refund?"
        sources = [{
            "title": "General Refund Info",
            "section": "REFUNDS",
            "excerpt": "Refunds are available for cancelled flights.",
        }]
        answer = _deterministic_passenger_rights_override(question, sources)
        assert answer is not None
        assert "Belum dapat dipastikan" in answer
        assert "Sumber definisi yang diperlukan belum berhasil diambil" in answer

    def test_downgrade_guard_without_fare_difference_evidence(self):
        """When evidence lacks fare difference info, guard should not fire."""
        question = "Maskapai menurunkan kelas, saya tetap terbang"
        sources = [{
            "title": "General Downgrade Info",
            "section": "DOWNGRADE",
            "excerpt": "Airlines sometimes downgrade passengers.",
        }]
        answer = _deterministic_passenger_rights_override(question, sources)
        # Should not fire because evidence lacks "difference between the original fare"
        assert answer is None

    def test_fully_refundable_guard_without_unused_ticket_evidence(self):
        """When evidence lacks unused ticket guidance, guard should not fire."""
        question = "Tiket fully refundable tidak dipakai"
        sources = [{
            "title": "General Refund",
            "section": "REFUND",
            "excerpt": "Refunds available for cancelled flights.",
        }]
        answer = _deterministic_passenger_rights_override(question, sources)
        assert answer is None

    def test_no_response_guard_without_automatic_refund_section(self):
        """When evidence lacks automatic refund trigger, guard should not fire."""
        question = "Tidak merespons alternative flight, tidak naik pesawat"
        sources = [{
            "title": "General Refund",
            "section": "REFUND",
            "excerpt": "Refunds may be available.",
        }]
        answer = _deterministic_passenger_rights_override(question, sources)
        assert answer is None

    def test_override_returns_none_for_irrelevant_question(self):
        """Guard should return None for questions not matching any pattern."""
        question = "Berapa lama delay rata-rata maskapai X?"
        sources = []
        answer = _deterministic_passenger_rights_override(question, sources)
        assert answer is None

    def test_override_handles_empty_sources(self):
        """Guard should handle empty sources gracefully."""
        question = "Tiba aktual 3 jam terlambat, tidak tahu jadwal revisi. Refund?"
        answer = _deterministic_passenger_rights_override(question, [])
        assert answer is not None
        assert "Belum dapat dipastikan" in answer