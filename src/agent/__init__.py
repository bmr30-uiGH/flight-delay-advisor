"""Agent package for Flight Delay + Passenger Rights Advisor.

This package provides the public interface re-exported from submodules:
- constants: SYSTEM_INSTRUCTIONS, RIGHTS_WORDS, PREDICTION_WORDS, _INJECTION_CLAUSE_PATTERNS
- guards: _deterministic_passenger_rights_override, _actual_arrival_only_with_unknown_schedule,
          _continued_downgrade_question, _unused_fully_refundable_ticket_question,
          _no_response_alternative_refund_question
- security: sanitize_mixed_request, looks_like_injection
- routing: heuristic_route
- formatting: _format_flight_summary_context, _question_with_flight_context,
             _remove_unsupported_cfr_citations, _strip_internal_source_markers,
             _strip_llm_source_footer, _deduplicate_sources
- core: answer_user
"""

from .constants import (
    SYSTEM_INSTRUCTIONS,
    RIGHTS_WORDS,
    PREDICTION_WORDS,
    _INJECTION_CLAUSE_PATTERNS,
    _explicitly_requests_prediction,
    _actual_arrival_only_with_unknown_schedule,
    _continued_downgrade_question,
    _unused_fully_refundable_ticket_question,
    _no_response_alternative_refund_question,
    _cancellation_without_any_offer_refund_question,
    _no_response_voucher_compensation_refund_question,
    _uses_indonesia_law,
)
from .guards import _deterministic_passenger_rights_override
from .security import sanitize_mixed_request, looks_like_injection
from .routing import heuristic_route
from .formatting import (
    _format_flight_summary_context,
    _question_with_flight_context,
    _remove_unsupported_cfr_citations,
    _strip_internal_source_markers,
    _strip_llm_source_footer,
    _deduplicate_sources,
)
from .core import (
    answer_user,
    _compact_passenger_rights_result_for_llm,
    _initial_tool_choice,
    _instructions_for_request,
    _prediction_tool_definition,
    _run_tool,
    _fallback_answer,
    _merge_passenger_rights_context,
    _retrieve_passenger_rights_bundle,
    _retrieve_single_query,
    _response_item_to_dict,
    _select_agent_model,
    _agent_output_token_limit,
)
from src.predictor import predict_delay
from src.retrieval import retrieve_passenger_rights

__all__ = [
    # Constants
    "SYSTEM_INSTRUCTIONS",
    "RIGHTS_WORDS",
    "PREDICTION_WORDS",
    "_INJECTION_CLAUSE_PATTERNS",
    "_explicitly_requests_prediction",
    "_actual_arrival_only_with_unknown_schedule",
    "_continued_downgrade_question",
    "_unused_fully_refundable_ticket_question",
    "_no_response_alternative_refund_question",
    "_cancellation_without_any_offer_refund_question",
    "_no_response_voucher_compensation_refund_question",
    "_uses_indonesia_law",
    # Guards
    "_deterministic_passenger_rights_override",
    # Security
    "sanitize_mixed_request",
    "looks_like_injection",
    # Routing
    "heuristic_route",
    # Formatting
    "_format_flight_summary_context",
    "_question_with_flight_context",
    "_remove_unsupported_cfr_citations",
    "_strip_internal_source_markers",
    "_strip_llm_source_footer",
    "_deduplicate_sources",
    # Core
    "answer_user",
    # Core helpers (used by tests)
    "_compact_passenger_rights_result_for_llm",
    "_initial_tool_choice",
    "_instructions_for_request",
    "_prediction_tool_definition",
    "_run_tool",
    "_fallback_answer",
    "_merge_passenger_rights_context",
    "_retrieve_passenger_rights_bundle",
    "_retrieve_single_query",
    "_response_item_to_dict",
    "_select_agent_model",
    "_agent_output_token_limit",
    # Retrieval
    "retrieve_passenger_rights",
]