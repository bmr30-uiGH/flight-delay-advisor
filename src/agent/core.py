"""Core agent logic: answer_user function and related helpers."""
from __future__ import annotations

import json
import os
import re

from src.predictor import predict_delay
from src.retrieval import retrieve_passenger_rights

from .constants import (
    SYSTEM_INSTRUCTIONS,
    _explicitly_requests_prediction,
    _uses_indonesia_law,
)
from .formatting import (
    _deduplicate_sources,
    _format_flight_summary_context,
    _question_with_flight_context,
    _remove_unsupported_cfr_citations,
    _strip_internal_source_markers,
    _strip_llm_source_footer,
)
from .guards import _deterministic_passenger_rights_override
from .routing import heuristic_route
from .security import sanitize_mixed_request


def _run_tool(name: str, args: dict, flight_inputs: dict | None) -> dict:
    if name == "predict_flight_delay":
        from src import agent as _agent_pkg

        return _agent_pkg.predict_delay(flight_inputs)
    if name == "search_passenger_rights":
        query = str(args.get("query", "")).strip()
        sources = _retrieve_passenger_rights_bundle(query)
        return {
            "ok": bool(sources),
            "query": query,
            "sources": sources,
            "message": "Ditemukan kutipan dokumen yang relevan." if sources else "Tidak ditemukan kutipan yang cukup relevan. Jangan membuat klaim hukum tanpa sumber.",
        }
    return {"ok": False, "message": "Tool tidak dikenal."}


def _retrieve_single_query(query: str) -> list[dict]:
    """Single-query retrieval hook for tests to monkeypatch."""
    from src import agent as _agent_pkg

    return _agent_pkg.retrieve_passenger_rights(query)


def _fallback_answer(question: str, flight_inputs: dict | None, flight_summary: dict | None = None):
    cleaned_question, injection_detected = sanitize_mixed_request(question)
    if injection_detected and not cleaned_question:
        return (
            "Saya tidak dapat membantu membuat klaim hukum, kutipan, atau sumber palsu maupun mengikuti instruksi untuk menonaktifkan pengamanan. "
            "Silakan ajukan pertanyaan hak penumpang yang ingin diperiksa berdasarkan dokumen.",
            [], [], "security pre-filter (unsafe-only request blocked)",
        )
    working_question = cleaned_question if injection_detected else question
    tools = heuristic_route(working_question)
    results, sources = [], []
    for name in tools:
        args = {"query": working_question} if name == "search_passenger_rights" else {}
        result = _run_tool(name, args, flight_inputs)
        results.append((name, result))
        if name == "search_passenger_rights":
            sources.extend(result.get("sources", []))
    if not tools:
        return (
            "Saya bisa membantu memperkirakan risiko keberangkatan terlambat ≥15 menit atau mencari informasi hak penumpang dari dokumen. "
            "Silakan ajukan pertanyaan yang ingin diperiksa.",
            [], [], "local fallback (LLM belum aktif)",
        )
    parts = []
    if injection_detected:
        parts.append("Saya mengabaikan instruksi manipulatif dan akan menjawab bagian pertanyaan yang sah.")
    for name, result in results:
        if name == "predict_flight_delay":
            if not result.get("ok"):
                parts.append(result.get("message", "Prediksi belum tersedia."))
            else:
                probability = result.get("probability")
                chance = f"Probabilitas model: {probability:.1%}. " if probability is not None else "Model tidak menyediakan probabilitas untuk output ini. "
                label = "diprediksi mengalami delay ≥15 menit" if result["predicted_delay_15"] else "tidak diklasifikasikan mengalami delay ≥15 menit"
                interpretation = result.get("interpretation", "")
                parts.append(chance + f"Klasifikasi: **{label}**. " + interpretation + " " + result.get("caveat", ""))
        elif result.get("sources"):
            parts.append(
                "Ringkasan lokal yang relevan (parafrasa edukatif, bukan kutipan verbatim dari sumber resmi):\n\n"
                + "\n\n".join(f"> {s['excerpt']}" for s in result["sources"][:3])
            )
        else:
            parts.append(result.get("message", "Sumber yang relevan belum tersedia."))
    if "predict_flight_delay" in tools and flight_summary:
        parts.append(_format_flight_summary_context(flight_summary))
    guarded_answer = _deterministic_passenger_rights_override(working_question, sources)
    if guarded_answer:
        return guarded_answer, sources, tools, "local fallback (deterministic passenger-rights guard)"

    return "\n\n".join(parts), sources, tools, "local fallback (LLM belum aktif)"


def _merge_passenger_rights_context(user_question: str, model_query: str) -> str:
    """Keep original legal intent when the LLM generates a narrower retrieval query."""
    original = re.sub(r"\s+", " ", str(user_question or "")).strip()
    focused = re.sub(r"\s+", " ", str(model_query or "")).strip()

    if not original:
        return focused
    if not focused:
        return original

    original_lower = original.casefold()
    focused_lower = focused.casefold()
    if original_lower in focused_lower:
        return focused
    if focused_lower in original_lower:
        return original

    return f"{original}\n\nFokus tambahan dari agent: {focused}"


def _retrieve_passenger_rights_bundle(query: str) -> list[dict]:
    """Retrieve focused legal evidence for complex refund questions."""
    original_query = re.sub(r"\s+", " ", str(query or "")).strip()
    if not original_query:
        return []

    # Keep Indonesian law in a separate corpus. Never rank Indonesian questions
    # against the U.S. DOT/eCFR index.
    if _uses_indonesia_law(original_query):
        from src.indonesia_rights import retrieve_indonesia_passenger_rights
        return retrieve_indonesia_passenger_rights(original_query)

    lowered = original_query.lower()
    queries = [original_query]

    rejection_intent = (
        any(term in lowered for term in (
            "menolak", "menolaknya", "penolakan", "saya tolak",
            "reject", "rejected", "decline", "declined",
        ))
        and any(term in lowered for term in (
            "alternatif", "penerbangan pengganti",
            "alternative flight", "rebooking",
        ))
    )

    agent_intent = any(term in lowered for term in (
        "ota", "online travel agency", "ticket agent",
        "travel agent", "agen perjalanan", "merchant of record",
    ))

    if rejection_intent:
        queries.append(
            "14 CFR §260.6(a)(2)(ii) explicit rejection of alternative flight "
            "automatic refund 7 business days after rejection merchant of record; "
            "distinguish no-response §260.6(a)(2)(iii)(A), where offered flight "
            "departs without consumer"
        )

    if agent_intent:
        queries.append(
            "14 CFR §399.80(l) ticket agent merchant of record refund upon request; "
            "7 business days after ticket agent receives information from carrier; "
            "14 CFR §260.6(d) carrier inform ticket agent without delay"
        )
        queries.append(
            "14 CFR §260.6(e) carrier holding consumer funds must promptly transfer "
            "funds to ticket agent when ticket agent is responsible for refund but "
            "does not possess the consumer's funds"
        )

    merged: list[dict] = []
    seen: set[tuple[str, str, str]] = set()

    for focused_query in queries:
        results = _retrieve_single_query(focused_query)
        for source in results or []:
            if not isinstance(source, dict):
                continue
            key = (
                str(source.get("url", "")).strip(),
                str(source.get("section", "")).strip(),
                str(source.get("excerpt", "")).strip(),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(source)

    return merged


def _compact_passenger_rights_result_for_llm(
    result: dict,
    max_sources: int = 5,
    max_excerpt_chars: int = 450,
) -> dict:
    """
    Produce a compact evidence payload for the LLM.

    Full retrieval results remain untouched in the application for:
    - deterministic legal guards,
    - source display,
    - source deduplication.

    Only the evidence sent back to the LLM is shortened.
    """

    raw_sources = result.get("sources", []) or []

    def source_priority(source: dict) -> int:
        evidence = " ".join(
            str(source.get(field, "") or "")
            for field in ("title", "section", "url", "excerpt")
        ).lower()

        priority = 0

        # Keep the distinct ticket-agent legal path.
        if "399.80" in evidence:
            priority += 100

        # Keep carrier fund-transfer provisions.
        if (
            "260.6(e)" in evidence
            or "promptly transfer" in evidence
            or "transfer the funds to the ticket agent" in evidence
        ):
            priority += 90

        # Keep the carrier notification provision.
        if (
            "260.6(d)" in evidence
            or (
                "ticket agent" in evidence
                and "without delay" in evidence
            )
        ):
            priority += 80

        # Keep the explicit rejection provision.
        if (
            "260.6(a)(2)(ii)" in evidence
            or "explicit rejection" in evidence
            or "penolakan eksplisit" in evidence
        ):
            priority += 70

        # Keep the refund deadline evidence.
        if (
            "7 business days" in evidence
            and "20 calendar days" in evidence
        ):
            priority += 60

        if "merchant of record" in evidence:
            priority += 30

        # Prefer the official DOT refund guidance.
        if (
            "transportation.gov" in evidence
            and "refund" in evidence
        ):
            priority += 20

        return priority

    indexed_sources = [
        (index, source)
        for index, source in enumerate(raw_sources)
        if isinstance(source, dict)
    ]

    # Prioritize legally distinct provisions, preserving retrieval order
    # when two sources have the same priority.
    ranked_sources = sorted(
        indexed_sources,
        key=lambda item: (
            -source_priority(item[1]),
            item[0],
        ),
    )

    selected = []
    seen = set()

    for _, source in ranked_sources:
        title = str(source.get("title", "U.S. DOT document") or "")
        section = str(source.get("section", "") or "")
        url = str(source.get("url", "") or "")
        excerpt = re.sub(
            r"\s+",
            " ",
            str(source.get("excerpt", "") or ""),
        ).strip()

        duplicate_key = (
            url.casefold(),
            section.casefold(),
            excerpt.casefold(),
        )

        if duplicate_key in seen:
            continue

        seen.add(duplicate_key)

        if len(excerpt) > max_excerpt_chars:
            if max_excerpt_chars <= 3:
                excerpt = excerpt[:max_excerpt_chars]
            else:
                shortened = excerpt[:max_excerpt_chars - 3]
                shortened = shortened.rsplit(" ", 1)[0].rstrip()
                excerpt = (shortened + "...")[:max_excerpt_chars]

        selected.append({
            "title": title[:180],
            "section": section[:180],
            "url": url[:500],
            "excerpt": excerpt,
        })

        if len(selected) >= max_sources:
            break

    return {
        "ok": bool(result.get("ok")),
        "query": str(result.get("query", "") or "")[:240],
        "message": str(result.get("message", "") or "")[:250],
        "sources": selected,
        "context_note": (
            "Compact retrieved evidence. Use only the supplied facts. "
            "The application separately retains the complete retrieved sources."
        ),
    }


def _prediction_tool_definition() -> dict:
    """Accept known model-generated fields; execution uses saved form data only."""
    return {
        "type": "function",
        "name": "predict_flight_delay",
        "description": (
            "Predict historical departure-delay probability using the flight "
            "details already saved in the application form. Do not infer or "
            "override flight details. Any arguments generated by the model "
            "are ignored; the application executes prediction strictly from "
            "the saved flight_inputs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "OP_CARRIER": {"type": "string"},
                "ORIGIN": {"type": "string"},
                "DEST": {"type": "string"},
                "FL_DATE": {"type": "string"},
                "CRS_DEP_TIME": {"type": "integer"},
                "DAY_OF_WEEK": {"type": "integer"},
                "MONTH": {"type": "integer"},
                "DISTANCE": {"type": "number"},
                "PREDICTION_MODE": {"type": "string"},
            },
            "required": [],
            "additionalProperties": False,
        },
    }


def _response_item_to_dict(item):
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        return item.model_dump(exclude_none=True)
    if hasattr(item, "dict"):
        return item.dict(exclude_none=True)
    raise TypeError(f"Unsupported Responses API item type: {type(item).__name__}")


def _instructions_for_request(question: str, routed_tools: list[str]) -> str:
    import re
    text = " ".join(str(question or "").casefold().split())

    is_us = any(term in text for term in (
        "united states", "u.s.", "us dot", "u.s. dot",
        "amerika serikat", "14 cfr", "ecfr",
    ))
    is_cancellation = any(term in text for term in (
        "cancelled", "canceled", "cancellation", "cancel",
        "dibatalkan", "membatalkan", "pembatalan",
        "penerbangan batal",
    ))

    is_indonesia = _uses_indonesia_law(question)
    has_prediction = "predict_flight_delay" in routed_tools
    has_rights = "search_passenger_rights" in routed_tools

    if has_prediction and has_rights and is_us and is_cancellation:
        return """Kamu adalah Flight Delay Advisor. Jawab dalam bahasa Indonesia yang alami dan langsung.

BATAS JAWABAN: maksimal 170 kata. Jangan membuat bagian sumber karena aplikasi menampilkan sumber secara terpisah. Hindari terjemahan janggal seperti "oferta" atau "uang kami".

Gunakan kedua tool: predict_flight_delay dan search_passenger_rights. Pisahkan informasi prediksi dari hak penumpang.

PREDIKSI:
- Sebutkan probabilitas, threshold, dan kelas prediksi.
- Jelaskan jika probabilitas dekat dengan threshold 50%.
- Tegaskan prediksi historis bukan status real-time atau jaminan.

HAK PENUMPANG:
- Pembatalan penerbangan domestik AS merupakan pemicu refund tersendiri berdasarkan §260.6(a)(2). Jangan menerapkan ambang 3 jam pada kasus pembatalan.
- Jelaskan hak refund penuh jika syarat sumber terpenuhi dan penumpang memilih tidak bepergian atau tidak menerima voucher/kredit sebagai pengganti refund.
- Refund dikembalikan ke metode pembayaran awal. Travel credit bukan pilihan yang wajib diterima.
- Bedakan carrier sebagai merchant of record dengan ticket agent/OTA. Ticket agent memproses refund atas permintaan konsumen berdasarkan §399.80(l), jika didukung sumber.
§260.6(b) berkaitan dengan perubahan fitur aksesibilitas bagi penumpang dengan disabilitas, bukan aturan umum bantuan makanan, minuman, atau hotel.
- Bedakan penolakan eksplisit dari tidak merespons. Jangan mencampur trigger tenggat 7 business days dan 20 calendar days.
- Jangan mengasumsikan merchant of record, jenis tiket, metode pembayaran, atau tindakan penumpang.

Gunakan hanya bukti kedua tool. Nomor regulasi hanya boleh disebut jika didukung hasil retrieval. Jika syarat belum diketahui, nyatakan secara bersyarat."""


    if has_prediction and has_rights and is_indonesia:
        return '''Kamu adalah Flight Delay Advisor. Jawab dalam bahasa pengguna.
Gunakan hasil predict_flight_delay dan search_passenger_rights; jangan mengarang hasil tool.
Prediksi: sampaikan probabilitas, threshold, dan kelas secara terpisah. Prediksi historis bukan status real-time.
Hak penumpang: gunakan hanya bukti hukum Indonesia dari retrieval. Jangan mencampur aturan U.S. DOT.
Untuk PM 89/2015, kategori 4 adalah 181-240 menit dan kategori 5 lebih dari 240 menit.
Bedakan kompensasi Pasal 9 dari penyebab keterlambatan Pasal 5 dan 6.
Akomodasi Pasal 10 ayat (4) mensyaratkan keterlambatan lebih dari 6 jam dan kebutuhan penginapan.
Jangan mengasumsikan penyebab atau kelayakan kompensasi tanpa bukti.'''

    if has_prediction and has_rights:
        return '''Kamu adalah Flight Delay Advisor. Jawab dalam bahasa pengguna.
Gunakan hasil predict_flight_delay dan search_passenger_rights; jangan mengarang hasil tool.
Prediksi: sampaikan probabilitas, threshold, dan kelas secara terpisah. Prediksi historis bukan status real-time.
Hak penumpang: pastikan yurisdiksi dan gunakan hanya bukti retrieval yang sesuai.
§260.6(b) berkaitan dengan perubahan fitur aksesibilitas bagi penumpang dengan disabilitas, bukan aturan umum bantuan makanan, minuman, atau hotel.
Jika syarat hukum belum lengkap, berikan kesimpulan bersyarat.'''

    if has_prediction:
        return SYSTEM_INSTRUCTIONS

    if has_rights and is_indonesia:
        return '''Kamu adalah Flight Delay Advisor. Jawab dalam bahasa pengguna berdasarkan search_passenger_rights saja.
Gunakan hanya bukti hukum Indonesia dari retrieval. Jangan mencampur aturan U.S. DOT.
Untuk PM 89/2015, kategori 4 adalah 181-240 menit dan kategori 5 lebih dari 240 menit.
Bedakan kompensasi Pasal 9 dari penyebab keterlambatan Pasal 5 dan 6.
Akomodasi Pasal 10 ayat (4) mensyaratkan keterlambatan lebih dari 6 jam dan kebutuhan penginapan.
Jangan mengasumsikan penyebab atau kelayakan kompensasi tanpa bukti.'''

    if has_rights and is_us and is_cancellation:
        return """Kamu adalah Flight Delay Advisor. Jawab dalam bahasa Indonesia yang alami, ringkas, dan langsung menjawab pertanyaan. Gunakan hanya bukti search_passenger_rights.

Batasi jawaban maksimal 140 kata. Jangan mengulang excerpt atau membuat daftar sumber karena aplikasi menampilkan sumber secara terpisah.

Jelaskan:
Pembatalan merupakan pemicu refund tersendiri berdasarkan §260.6(a)(2). Jangan mensyaratkan ambang keterlambatan 3 jam domestik atau 6 jam internasional untuk penerbangan yang dibatalkan.
Penolakan eksplisit diatur dalam §260.6(a)(2)(ii), sedangkan kondisi tidak merespons diatur dalam §260.6(a)(2)(iii). Jangan mencampur kedua kondisi tersebut.
Jika agen/OTA adalah merchant of record, §399.80(l) mengatur refund atas permintaan konsumen. Tenggat yang relevan mencakup 7 business days dan 20 calendar days sesuai metode pembayaran dan kondisi sumber.
§260.6(b) berkaitan dengan perubahan fitur aksesibilitas bagi penumpang dengan disabilitas, bukan bantuan makanan, minuman, atau hotel.

1. Pembatalan penerbangan tercakup ke, dari, atau di AS dapat memberikan hak refund penuh ketika penumpang memilih tidak bepergian dan tidak menerima voucher/kredit sebagai pengganti refund. Jangan menyatakan hak ini tanpa syarat.
2. Bedakan penolakan alternatif secara eksplisit dari tidak merespons. Untuk penolakan eksplisit, jelaskan tenggat 7 hari kerja untuk kartu kredit atau 20 hari kalender untuk metode lain sejak tanggal penolakan, jika didukung sumber.
3. Jika agen/OTA adalah merchant of record, jelaskan bahwa refund dilakukan atas permintaan konsumen. Bedakan kewajiban agen dari kewajiban maskapai.
4. Refund adalah pengembalian dana ke metode pembayaran awal. Travel credit adalah alternatif yang tidak wajib diterima penumpang.

Jangan mengasumsikan merchant of record, metode pembayaran, jenis tiket, atau tindakan penumpang. Jika belum diketahui, nyatakan secara singkat. Sebut nomor aturan hanya jika didukung sumber."""


    if has_rights and is_us:
        return '''Kamu adalah Flight Delay Advisor. Jawab dalam bahasa pengguna berdasarkan search_passenger_rights saja.
Gunakan aturan U.S. DOT/eCFR hanya untuk penerbangan yang tercakup ke, dari, atau di AS.
Bedakan pembatalan dari significantly delayed/changed flight. Untuk perubahan jadwal, gunakan definisi dan ambang dalam sumber.
Bedakan tiket non-refundable dan fully refundable.
Jangan mengasumsikan merchant of record, metode pembayaran, atau tindakan penumpang.
Gunakan hanya fakta retrieval dan berikan kesimpulan bersyarat jika perlu.'''

    if has_rights:
        return '''Kamu adalah Flight Delay Advisor. Jawab dalam bahasa pengguna berdasarkan search_passenger_rights saja.
Pastikan yurisdiksi sebelum menyimpulkan hak penumpang.
Jangan mencampur aturan Indonesia dan U.S. DOT.
Gunakan hanya fakta yang didukung retrieval.
Jangan mengasumsikan penyebab, jenis tiket, metode pembayaran, atau tindakan penumpang.
Jika informasi belum cukup, jelaskan keterbatasannya.'''

    return SYSTEM_INSTRUCTIONS


def _initial_tool_choice(
    routed_tools: list[str],
    has_flight_inputs: bool,
):
    """Force a tool only for single-purpose requests; allow combined routing."""
    if has_flight_inputs and routed_tools == ["predict_flight_delay"]:
        return {"type": "function", "name": "predict_flight_delay"}

    if routed_tools == ["search_passenger_rights"]:
        return {"type": "function", "name": "search_passenger_rights"}

    # Combined requests must allow the model to call both tools.
    return "auto"


def _agent_output_token_limit(is_groq: bool) -> int | None:
    # Stay below Groq Qwen's 1,000 output-tokens-per-minute request limit.
    return 900 if is_groq else None


def _select_agent_model(is_groq: bool) -> str:
    if is_groq:
        return os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")
    return os.getenv("OPENAI_MODEL", "gpt-4.1-mini")


def answer_user(question: str, flight_inputs: dict | None = None, flight_summary: dict | None = None):
    import re

    cleaned_question, injection_detected = sanitize_mixed_request(question)
    if injection_detected and not cleaned_question:
        return (
            "Saya tidak dapat membantu membuat klaim hukum, kutipan, atau sumber palsu maupun mengikuti instruksi untuk menonaktifkan pengamanan. "
            "Silakan ajukan pertanyaan hak penumpang yang ingin diperiksa berdasarkan dokumen.",
            [], [], "security pre-filter (unsafe-only request blocked)",
        )

    working_question = cleaned_question if injection_detected else question

    # INDONESIA_RAG_PATCH_V1: block unrelated requests before calling the LLM.
    if not heuristic_route(working_question):
        return (
            "Saya dirancang untuk membantu memperkirakan risiko keterlambatan penerbangan "
            "atau mencari informasi hak penumpang berdasarkan dokumen. Ajukan pertanyaan "
            "yang berkaitan dengan prediksi delay, refund, pembatalan, kompensasi, atau "
            "kewajiban maskapai.",
            [], [], "scope guard (outside Flight Delay Advisor)",
        )

    api_key = (os.getenv("GROQ_API_KEY", "").strip() or os.getenv("OPENAI_API_KEY", "").strip())
    if not api_key:
        return _fallback_answer(question, flight_inputs, flight_summary)
    try:
        from openai import OpenAI
        base_url = (os.getenv("LLM_BASE_URL", "").strip() or os.getenv("OPENAI_BASE_URL", "").strip())
        is_groq = "api.groq.com" in base_url.lower() or api_key.startswith("gsk_")
        if is_groq and not base_url:
            base_url = "https://api.groq.com/openai/v1"
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)
        model = _select_agent_model(is_groq)

        routed_tools = heuristic_route(working_question)

        tools = []
        if "predict_flight_delay" in routed_tools:
            tools.append(_prediction_tool_definition())

        if "search_passenger_rights" in routed_tools:
            tools.append({
                "type": "function",
                "name": "search_passenger_rights",
                "description": (
                    "Cari dokumen resmi hak penumpang berdasarkan yurisdiksi "
                    "Indonesia atau Amerika Serikat."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            })

        explicit_form_reference = any(term in working_question.casefold() for term in (
            "penerbangan yang tersimpan",
            "penerbangan pada formulir",
            "penerbangan yang sama dengan formulir",
            "detail penerbangan di formulir",
            "saved flight",
            "same flight in the form",
        ))

        rights_only = routed_tools == ["search_passenger_rights"]

        if rights_only and not explicit_form_reference:
            contextual_question = (
                "Pertanyaan substantif pengguna:\n" + working_question
            )
        else:
            contextual_question = _question_with_flight_context(
                working_question,
                flight_inputs,
                flight_summary,
            )

        if injection_detected:
            contextual_question += (
                "\n\nCatatan keamanan: instruksi manipulatif telah dihapus "
                "oleh pre-filter. Jawab pertanyaan sah yang tersisa."
            )

        conversation = [{"role": "user", "content": contextual_question}]
        request_input = conversation if is_groq else contextual_question

        request_instructions = _instructions_for_request(
            working_question,
            routed_tools,
        )

        first_tool_choice = _initial_tool_choice(
            routed_tools,
            has_flight_inputs=bool(flight_inputs),
        )

        response = client.responses.create(
            model=model,
            max_output_tokens=_agent_output_token_limit(is_groq),
            instructions=request_instructions,
            input=request_input,
            tools=tools,
            tool_choice=first_tool_choice,
        )

        used: list[str] = []
        sources: list[dict] = []
        for _ in range(4):
            calls = [item for item in response.output if getattr(item, "type", "") == "function_call"]
            if not calls:
                break
            tool_outputs = []
            if is_groq:
                conversation.extend(_response_item_to_dict(item) for item in response.output)
            for call in calls:
                try:
                    args = json.loads(call.arguments or "{}")
                except (json.JSONDecodeError, TypeError):
                    args = {}

                if call.name == "search_passenger_rights":
                    args["query"] = _merge_passenger_rights_context(
                        working_question,
                        str(args.get("query", "")),
                    )

                result = _run_tool(call.name, args, flight_inputs)
                used.append(call.name)
                if call.name == "search_passenger_rights":
                    sources.extend(result.get("sources", []))
                # Preserve the complete result in local `sources`.
                # Only send a compact evidence payload back to the LLM.
                llm_tool_result = result

                if call.name == "search_passenger_rights":
                    llm_tool_result = (
                        _compact_passenger_rights_result_for_llm(result)
                    )

                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(
                        llm_tool_result,
                        ensure_ascii=False,
                    ),
                })
            all_tools_used = set(routed_tools).issubset(set(used))

            followup_kwargs = {
                "model": model,
                "max_output_tokens": _agent_output_token_limit(is_groq),
                "instructions": request_instructions,
                "tools": [] if all_tools_used else tools,
            }

            if all_tools_used:
                followup_kwargs["instructions"] += (
                    "\n\nSemua tool yang diperlukan sudah dijalankan. "
                    "Gunakan hasil tool yang tersedia untuk menyusun jawaban akhir. "
                    "Jangan mencoba memanggil tool lagi."
                )
            else:
                followup_kwargs["tool_choice"] = "auto"

            if is_groq:
                conversation.extend(tool_outputs)
                followup_kwargs["input"] = conversation
            else:
                followup_kwargs["previous_response_id"] = response.id
                followup_kwargs["input"] = tool_outputs

            response = client.responses.create(**followup_kwargs)

        answer = (getattr(response, "output_text", "") or "").strip()
        if "search_passenger_rights" in used and not sources:
            answer = (
                "Saya belum menemukan kutipan dokumen resmi yang cukup relevan untuk menjawab pertanyaan ini. "
                "Saya tidak akan menyimpulkan hak atau kompensasi tanpa sumber. Periksa kembali yurisdiksi "
                "penerbangan dan ajukan detail keterlambatan atau pembatalannya."
            )
        answer = _strip_llm_source_footer(
            _strip_internal_source_markers(
                _remove_unsupported_cfr_citations(answer, sources)
            )
        )
        guarded_answer = _deterministic_passenger_rights_override(working_question, sources)
        if guarded_answer:
            answer = guarded_answer

        if not answer:
            answer = "Tool telah diproses, tetapi model tidak mengembalikan jawaban akhir. Silakan ulangi pertanyaan."

        deduped = _deduplicate_sources(sources)
        # Display which tool types ran, not repeated calls to the same tool.
        used = list(dict.fromkeys(used))
        provider = "Groq Responses API" if is_groq else "OpenAI Responses API"
        if injection_detected:
            provider += " (unsafe clauses filtered)"
        return answer, deduped, used, f"{provider} ({model})"
    except Exception as exc:
        answer, sources, used, mode = _fallback_answer(question, flight_inputs, flight_summary)
        return (
            answer + f"\n\n> Catatan teknis: mode agent LLM gagal digunakan ({type(exc).__name__}: {exc}). "
            "Jawaban di atas berasal dari fallback lokal; periksa provider, API key, model, atau koneksi sebelum penilaian.",
            sources, used, "fallback setelah error LLM",
        )