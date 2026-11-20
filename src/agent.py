"""Compatibility layer: re-exports public interface from the agent package.

All implementations have moved to src/agent/ submodules.
This file exists to preserve imports for app.py and tests.
"""

from src.agent import (
    # Core
    answer_user,
    # Constants
    SYSTEM_INSTRUCTIONS,
    RIGHTS_WORDS,
    PREDICTION_WORDS,
    _INJECTION_CLAUSE_PATTERNS,
    # Guards
    _deterministic_passenger_rights_override,
    _actual_arrival_only_with_unknown_schedule,
    _continued_downgrade_question,
    _unused_fully_refundable_ticket_question,
    _no_response_alternative_refund_question,
    _cancellation_without_any_offer_refund_question,
    _no_response_voucher_compensation_refund_question,
    _uses_indonesia_law,
    _explicitly_requests_prediction,
    # Security
    sanitize_mixed_request,
    looks_like_injection,
    # Routing
    heuristic_route,
    # Formatting
    _format_flight_summary_context,
    _question_with_flight_context,
    _remove_unsupported_cfr_citations,
    _strip_internal_source_markers,
    _strip_llm_source_footer,
    _deduplicate_sources,
    # Core helpers (used by tests)
    _compact_passenger_rights_result_for_llm,
    _initial_tool_choice,
    _instructions_for_request,
    _prediction_tool_definition,
    _run_tool,
    _fallback_answer,
    _merge_passenger_rights_context,
    _retrieve_passenger_rights_bundle,
    _response_item_to_dict,
    _select_agent_model,
    _agent_output_token_limit,
)

__all__ = [
    # Core
    "answer_user",
    # Constants
    "SYSTEM_INSTRUCTIONS",
    "RIGHTS_WORDS",
    "PREDICTION_WORDS",
    "_INJECTION_CLAUSE_PATTERNS",
    # Guards
    "_deterministic_passenger_rights_override",
    "_actual_arrival_only_with_unknown_schedule",
    "_continued_downgrade_question",
    "_unused_fully_refundable_ticket_question",
    "_no_response_alternative_refund_question",
    "_cancellation_without_any_offer_refund_question",
    "_no_response_voucher_compensation_refund_question",
    "_uses_indonesia_law",
    "_explicitly_requests_prediction",
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
    # Core helpers
    "_compact_passenger_rights_result_for_llm",
    "_initial_tool_choice",
    "_instructions_for_request",
    "_prediction_tool_definition",
    "_run_tool",
    "_fallback_answer",
    "_merge_passenger_rights_context",
    "_retrieve_passenger_rights_bundle",
    "_response_item_to_dict",
    "_select_agent_model",
    "_agent_output_token_limit",
]