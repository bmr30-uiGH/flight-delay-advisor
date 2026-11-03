from src.agent import heuristic_route, _initial_tool_choice


def test_prediction_only_forces_prediction_tool_when_flight_is_saved():
    question = (
        "Perkirakan risiko delay minimal 15 menit untuk penerbangan saya."
    )

    routed = heuristic_route(question)

    assert routed == ["predict_flight_delay"]
    assert _initial_tool_choice(routed, True) == {
        "type": "function",
        "name": "predict_flight_delay",
    }


def test_rights_only_forces_passenger_rights_retrieval():
    question = (
        "Apa hak penumpang jika penerbangan domestik Amerika Serikat dibatalkan?"
    )

    routed = heuristic_route(question)

    assert routed == ["search_passenger_rights"]
    assert _initial_tool_choice(routed, True) == {
        "type": "function",
        "name": "search_passenger_rights",
    }


def test_combined_question_allows_both_tools():
    question = (
        "Perkirakan risiko delay minimal 15 menit untuk penerbangan saya, "
        "lalu jelaskan hak penumpang berdasarkan aturan U.S. DOT "
        "jika penerbangan domestik Amerika Serikat saya dibatalkan."
    )

    routed = heuristic_route(question)

    assert routed == [
        "predict_flight_delay",
        "search_passenger_rights",
    ]
    assert _initial_tool_choice(routed, True) == "auto"


def test_prediction_without_saved_flight_details_is_not_forced():
    question = "Perkirakan risiko delay minimal 15 menit."

    routed = heuristic_route(question)

    assert routed == ["predict_flight_delay"]
    assert _initial_tool_choice(routed, False) == "auto"



def test_combined_us_question_uses_compact_instructions_for_both_tools():
    from src.agent import _instructions_for_request

    question = (
        "Perkirakan risiko delay minimal 15 menit untuk penerbangan saya, "
        "lalu jelaskan hak penumpang berdasarkan aturan U.S. DOT "
        "jika penerbangan domestik Amerika Serikat saya dibatalkan."
    )
    routed = heuristic_route(question)
    instructions = _instructions_for_request(question, routed)

    assert routed == ["predict_flight_delay", "search_passenger_rights"]
    assert "predict_flight_delay" in instructions
    assert "search_passenger_rights" in instructions
    assert "?260.6(b)" in instructions
    assert "fitur aksesibilitas" in instructions.lower()
    assert "bukan aturan umum" in instructions.lower()
    assert len(instructions) < 5000


def test_us_cancellation_prompt_does_not_invent_meals_or_hotels():
    from src.agent import _instructions_for_request

    question = (
        "What are my rights under U.S. DOT if my domestic U.S. flight is cancelled?"
    )
    instructions = _instructions_for_request(
        question,
        ["search_passenger_rights"],
    )

    assert "?399.80(l)" in instructions
    assert "atas permintaan konsumen" in instructions.lower()
    assert "?260.6(b)" in instructions
    assert "bukan bantuan makanan, minuman, atau hotel" in instructions.lower()
    assert len(instructions) < 5000


def test_non_us_or_unrelated_requests_keep_full_system_instructions():
    from src.agent import _instructions_for_request, SYSTEM_INSTRUCTIONS

    question = "Perkirakan risiko delay untuk penerbangan saya."
    assert _instructions_for_request(
        question,
        ["predict_flight_delay"],
    ) == SYSTEM_INSTRUCTIONS



def test_prediction_tool_schema_accepts_known_groq_payload():
    from src.agent import _prediction_tool_definition

    definition = _prediction_tool_definition()
    properties = definition["parameters"]["properties"]

    expected = {
        "OP_CARRIER", "ORIGIN", "DEST", "FL_DATE",
        "CRS_DEP_TIME", "DAY_OF_WEEK", "MONTH",
        "DISTANCE", "PREDICTION_MODE",
    }

    assert set(properties) == expected
    assert definition["parameters"]["additionalProperties"] is False
    assert definition["parameters"]["required"] == []


def test_prediction_tool_uses_saved_inputs_not_model_arguments(monkeypatch):
    import src.agent as agent

    saved_inputs = {
        "OP_CARRIER": "DL",
        "ORIGIN": "MKE",
        "DEST": "DTW",
        "FL_DATE": "2026-11-04",
        "CRS_DEP_TIME": 900,
        "DAY_OF_WEEK": 4,
        "MONTH": 11,
        "DISTANCE": 237.0,
        "PREDICTION_MODE": "production",
    }

    model_arguments = {
        "OP_CARRIER": "XX",
        "ORIGIN": "AAA",
        "DEST": "BBB",
        "FL_DATE": "2026-12-31",
        "CRS_DEP_TIME": 1500,
        "DAY_OF_WEEK": 4,
        "MONTH": 12,
        "DISTANCE": 999.0,
        "PREDICTION_MODE": "v3_candidate_historical",
    }

    captured = {}

    def fake_predict_delay(inputs):
        captured["inputs"] = inputs
        return {"ok": True}

    monkeypatch.setattr(agent, "predict_delay", fake_predict_delay)

    result = agent._run_tool(
        "predict_flight_delay",
        model_arguments,
        saved_inputs,
    )

    assert result == {"ok": True}
    assert captured["inputs"] is saved_inputs
    assert captured["inputs"]["ORIGIN"] == "MKE"
    assert captured["inputs"]["PREDICTION_MODE"] == "production"


def test_us_cancellation_prompt_separates_cancellation_from_delay_threshold():
    from src.agent import _instructions_for_request

    question = (
        "What are my rights under U.S. DOT if my domestic U.S. flight is cancelled?"
    )
    instructions = _instructions_for_request(
        question,
        ["search_passenger_rights"],
    )

    assert "pemicu refund tersendiri" in instructions.lower()
    assert "jangan mensyaratkan ambang" in instructions.lower()
    assert "?260.6(a)(2)(ii)" in instructions
    assert "?260.6(a)(2)(iii)" in instructions
    assert "?399.80(l)" in instructions
    assert "20 calendar days" in instructions


def test_groq_uses_qwen_model_by_default(monkeypatch):
    from src.agent import _select_agent_model

    monkeypatch.delenv("GROQ_MODEL", raising=False)
    monkeypatch.setenv("OPENAI_MODEL", "openai/gpt-oss-120b")
    assert _select_agent_model(True) == "qwen/qwen3.8-27b"


def test_groq_model_can_be_overridden(monkeypatch):
    from src.agent import _select_agent_model

    monkeypatch.setenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    assert _select_agent_model(True) == "qwen/qwen3.8-27b"


def test_non_groq_keeps_openai_model_setting(monkeypatch):
    from src.agent import _select_agent_model

    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    assert _select_agent_model(False) == "gpt-4.1-mini"



def test_groq_output_token_limit_is_below_otpm_cap():
    from src.agent import _agent_output_token_limit

    assert _agent_output_token_limit(True) == 900
    assert _agent_output_token_limit(False) is None
