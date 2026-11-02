from __future__ import annotations

import json
from functools import lru_cache

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import CHUNKS_PATH

# Indonesian/English phrase expansions help retrieve the exact legal distinction,
# rather than matching only the generic word "delay" or "refund".
_QUERY_EXPANSIONS = {
    "fully refundable": (
        "fully refundable ticket unused ticket refund DOT guidance fare refundable "
        "not used purchased ticket separate from nonrefundable ticket 260.6(a)"
    ),
    "tiket refundable": (
        "fully refundable ticket unused ticket refund DOT guidance "
        "separate from nonrefundable ticket"
    ),
    "non-refundable": (
        "nonrefundable ticket 14 CFR Part 260 section 260.6(a)(1) full prompt refund "
        "consumer chooses not to fly affected flight alternative flight compensation"
    ),
    "nonrefundable": (
        "nonrefundable ticket 14 CFR Part 260 section 260.6(a)(1) full prompt refund "
        "consumer chooses not to fly affected flight alternative flight compensation"
    ),
    "sudah terbang": (
        "consumer traveled took significantly delayed changed flight accepted alternative flight "
        "generally not entitled full ticket price refund downgrade fare difference"
    ),
    "already flew": (
        "consumer traveled took significantly delayed changed flight accepted alternative flight "
        "generally not entitled full ticket price refund downgrade fare difference"
    ),
    "already travelled": (
        "consumer travelled traveled took significantly delayed changed flight accepted alternative flight "
        "generally not entitled full ticket price refund downgrade fare difference"
    ),
    "already traveled": (
        "consumer traveled took significantly delayed changed flight accepted alternative flight "
        "generally not entitled full ticket price refund downgrade fare difference"
    ),
    "tetap terbang": (
        "consumer traveled took significantly delayed changed flight accepted alternative flight "
        "generally not entitled full ticket price refund downgrade fare difference"
    ),
    "sampai tujuan": (
        "actual arrival time scheduled arrival revised scheduled arrival original scheduled arrival "
        "consumer traveled full ticket refund"
    ),
    "actual arrival": (
        "actual landing time revised scheduled arrival original scheduled arrival "
        "scheduled to arrive late arrival threshold actual delay alone"
    ),
    "actual delay": (
        "actual arrival time scheduled arrival revised scheduled arrival original scheduled arrival "
        "carrier made schedule change late arrival trigger"
    ),
    "late arrival": (
        "consumer scheduled to arrive destination airport original scheduled arrival revised scheduled arrival "
        "3 hours domestic 6 hours international 14 CFR section 260.2"
    ),
    "scheduled arrival": (
        "original scheduled arrival revised scheduled arrival consumer is scheduled to arrive "
        "3 hours domestic 6 hours international"
    ),
    "jadwal kedatangan": (
        "original scheduled arrival revised scheduled arrival consumer scheduled to arrive "
        "late arrival 3 hours domestic 6 hours international"
    ),
    "keterlambatan aktual": (
        "actual arrival time actual landing not scheduled arrival revised scheduled arrival "
        "original scheduled arrival significant delay definition"
    ),
    "terlambat mendarat": (
        "actual arrival time actual landing not scheduled arrival revised scheduled arrival "
        "original scheduled arrival significant delay definition"
    ),
    "keberangkatan lebih awal": (
        "early departure scheduled departure earlier than original scheduled departure "
        "3 hours domestic 6 hours international"
    ),
    "berangkat lebih awal": (
        "early departure scheduled departure earlier than original scheduled departure "
        "3 hours domestic 6 hours international"
    ),
    "penerbangan alternatif": (
        "alternative flight alternative transportation rebooking accept reject offered flight "
        "automatic refund does not use alternative transportation"
    ),
    "alternative flight": (
        "alternative transportation rebooking accept reject offered flight "
        "automatic refund does not use alternative transportation"
    ),
    "menolak": (
        "rejects significantly delayed changed flight rejects rebooking alternative flight "
        "rejects voucher credit automatic refund"
    ),
    "menerima": (
        "accepts takes alternative transportation traveled consumer chooses to fly "
        "voucher credit compensation in lieu of refund"
    ),
    "rebooking": (
        "alternative flight alternative transportation accepts rebooking rejects rebooking "
        "consumer chooses not to fly automatic refund"
    ),
    "downgrade": (
        "involuntary downgrade lower class of service difference between original fare and downgraded fare "
        "full airfare refund if passenger continues travel"
    ),
    "diturunkan kelas": (
        "involuntary downgrade lower class of service difference between original fare and downgraded fare "
        "full airfare refund if passenger continues travel"
    ),
    "selisih tarif": (
        "difference between original fare and downgraded fare involuntary downgrade "
        "passenger continues travel not full airfare refund"
    ),
    "pengembalian dana": (
        "refund ticket price original form of payment 14 CFR Part 260 section 260.6 "
        "nonrefundable ticket consumer chooses not to travel"
    ),
    "pengembalian uang": (
        "refund ticket price original form of payment 14 CFR Part 260 section 260.6 "
        "nonrefundable ticket consumer chooses not to travel"
    ),
    "refund": (
        "refund ticket price original form of payment 14 CFR Part 260 section 260.6 "
        "nonrefundable ticket refund obligation covered carrier merchant of record "
        "consumer does not fly on changed or alternative flight"
    ),
    "kompensasi": (
        "compensation cash compensation travel credit voucher meal voucher hotel airline commitments "
        "refund is different from extra monetary compensation"
    ),
    "voucher": (
        "voucher credit compensation in lieu of refund explicit alternative refund right "
        "clear conspicuous notice five years acceptance"
    ),
    "makanan": "meal meal cash voucher meal assistance controllable delay waiting 3 hours airline commitment",
    "hotel": "complimentary hotel ground transportation overnight controllable delay airline commitment",
    "american airlines": (
        "American airline AA airline customer service dashboard controllable delay "
        "same airline partner airline rebooking meal cash voucher hotel ground transportation "
        "cash compensation travel voucher frequent flyer miles DOT accountability"
    ),
    "merchant of record": (
        "merchant of record payment statement ticket agent online travel agency refund processing "
        "carrier transfer funds ticket agent"
    ),
    "travel agent": (
        "ticket agent online travel agency merchant of record refund processing "
        "payment statement carrier ticket agent"
    ),
    "tiket": "airline ticket refundable nonrefundable ticket unused ticket fare type",
    "landasan": "tarmac delay aircraft on the ground opportunity to leave aircraft",
    "di pesawat": "tarmac delay aircraft passengers opportunity to deplane",
}


# Additional natural Indonesian and English phrasings observed in UI regression tests.
_QUERY_EXPANSIONS.update({
    "tiba aktual": (
        "actual arrival actual landing time revised scheduled arrival original scheduled arrival "
        "scheduled to arrive late arrival definition section 260.2"
    ),
    "kedatangan aktual": (
        "actual arrival actual landing time revised scheduled arrival original scheduled arrival "
        "scheduled to arrive late arrival definition section 260.2"
    ),
    "mendarat aktual": (
        "actual landing time actual arrival revised scheduled arrival original scheduled arrival "
        "scheduled to arrive late arrival definition section 260.2"
    ),
    "tidak tahu apakah maskapai mengubah jadwal": (
        "actual arrival is not revised scheduled arrival original scheduled arrival "
        "carrier made delay or change consumer is scheduled to arrive 3 hours domestic 6 hours international"
    ),
    "penurunan kelas": (
        "involuntary downgrade lower class of service continues to travel "
        "difference between the original fare and the downgraded fare not full airfare refund"
    ),
    "menurunkan kelas": (
        "involuntary downgrade lower class of service continues to travel "
        "difference between the original fare and the downgraded fare not full airfare refund"
    ),
    "diturunkan ke ekonomi": (
        "involuntary downgrade business class economy lower class of service "
        "difference between the original fare and the downgraded fare"
    ),
    "kelas bisnis": (
        "business class involuntary downgrade lower class of service economy "
        "difference between the original fare and the downgraded fare"
    ),
    "business class": (
        "business class involuntary downgrade lower class of service economy "
        "difference between the original fare and the downgraded fare"
    ),
    "selisih tarif": (
        "difference between the original fare and the downgraded fare fare difference "
        "involuntary downgrade passenger continues travel"
    ),
    "tidak merespons": (
        "does not respond to offer alternative flight flight departs without consumer "
        "automatic refund section 260.6(a)(2)(iii) original form of payment deadline"
    ),
    "tidak menanggapi": (
        "does not respond to offer alternative flight flight departs without consumer "
        "automatic refund section 260.6(a)(2)(iii) original form of payment deadline"
    ),
    "online travel agency": (
        "ticket agent OTA merchant of record refund upon request 14 CFR 399.80(l) "
        "7 business days after ticket agent receives information from carrier 260.6(d) "
        "carrier transfer funds to ticket agent 260.6(e) original form of payment"
    ),
    "ota": (
        "ticket agent online travel agency merchant of record refund upon request "
        "14 CFR 399.80(l) 7 business days after information from carrier 260.6(d) "
        "20 calendar days refund due original form of payment"
    ),
    "tarmac delay": (
        "14 CFR 259.4 domestic tarmac delay aircraft on ground no opportunity to deplane "
        "adequate food potable water within two hours pilot safety security exception "
        "operable lavatory comfortable cabin temperature medical attention status notice exceeds 30 minutes "
        "departing flight begin return suitable disembarkation point by three hours after main door closed "
        "arriving flight opportunity to deplane before exceeds three hours "
        "pilot safety security ATC significantly disrupt airport operations"
    ),
    "tertahan di landasan": (
        "tarmac delay 14 CFR 259.4 aircraft on the ground no opportunity to leave "
        "food and potable water two hours pilot safety security exception operable lavatory "
        "comfortable cabin temperature status notice 30 minutes domestic three hour deplane limit "
        "drinking water within two hours opportunity to deplane "
        "departing flight begins return by three hours arriving flight opportunity to deplane before three hours "
        "air traffic control significantly disrupt airport operations"
    ),
    "penundaan di landasan": (
        "tarmac delay aircraft on the ground passengers cannot leave aircraft food water two hours "
        "lavatory temperature medical attention opportunity to deplane three hours domestic"
    ),
    "pesawat tertahan": (
        "tarmac delay aircraft on the ground passengers cannot deplane opportunity to leave aircraft "
        "food drinking water toilets medical attention three hours domestic"
    ),
    "deplane": (
        "tarmac delay opportunity to deplane aircraft on the ground domestic three hours international four hours"
    ),
    "penolakan eksplisit": (
        "explicit rejection rejected alternative flight 14 CFR 260.6(a)(2)(ii) automatic refund "
        "7 business days after rejection 20 calendar days after rejection merchant of record "
        "ticket agent online travel agency OTA upon request 14 CFR 399.80(l) "
        "7 business days after ticket agent receives information from carrier under 260.6(d) "
        "20 calendar days from refund becoming due carrier notify agent transfer funds 260.6(e)"
    ),
})


def _expand_query(query: str) -> str:
    lowered = " ".join((query or "").lower().split())
    fully_refundable_intent = any(phrase in lowered for phrase in (
        "fully refundable", "fully-refundable", "tiket refundable",
    ))
    additions = [
        value
        for phrase, value in _QUERY_EXPANSIONS.items()
        if phrase in lowered
        and not (fully_refundable_intent and phrase in {"refund", "pengembalian dana", "pengembalian uang"})
    ]
    return f"{query} {' '.join(additions)}" if additions else query


@lru_cache(maxsize=2)
def _load_chunks(path_string: str, modified_ns: int) -> tuple[dict, ...]:
    # The mtime argument invalidates the small cache when ingestion rewrites the index.
    del path_string, modified_ns
    if not CHUNKS_PATH.exists():
        return tuple()

    records: list[dict] = []
    with CHUNKS_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("text") and item.get("url"):
                records.append(item)
    return tuple(records)


def retrieve_passenger_rights(query: str, top_k: int = 6) -> list[dict]:
    if not CHUNKS_PATH.exists() or not str(query or "").strip():
        return []

    chunks = _load_chunks(str(CHUNKS_PATH), CHUNKS_PATH.stat().st_mtime_ns)
    if not chunks:
        return []

    searchable_texts = [
        " ".join((item.get("title", ""), item.get("section", ""), item.get("text", "")))
        for item in chunks
    ]
    expanded_query = _expand_query(query)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=30000,
    )
    try:
        matrix = vectorizer.fit_transform(searchable_texts + [expanded_query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).ravel()
    except ValueError:
        return []

    # Intent-specific relevance boosts for narrowly framed legal questions.
    lowered_query = str(query or "").lower()
    arrival_intent = any(term in lowered_query for term in (
        "actual arrival", "actual landing", "tiba aktual", "kedatangan aktual",
        "mendarat aktual", "revised scheduled arrival", "jadwal kedatangan",
        "keterlambatan aktual", "terlambat mendarat",
    ))
    downgrade_intent = any(term in lowered_query for term in (
        "downgrade", "penurunan kelas", "menurunkan kelas", "diturunkan ke ekonomi",
        "kelas bisnis", "business class", "selisih tarif",
    ))
    fully_refundable_intent = any(term in lowered_query for term in (
        "fully refundable", "fully-refundable", "tiket refundable",
    ))
    no_response_alt_intent = (
        any(term in lowered_query for term in ("tidak merespons", "tidak menanggapi", "did not respond", "no response"))
        and any(term in lowered_query for term in ("penerbangan alternatif", "alternative flight", "alternative transportation"))
    )
    tarmac_intent = any(term in lowered_query for term in (
        "tarmac", "landasan", "di landasan", "tertahan di landasan",
        "penundaan di landasan", "pesawat tertahan", "deplane", "turun dari pesawat",
    ))
    explicit_rejection_alt_intent = (
        any(term in lowered_query for term in (
            "menolak", "menolaknya", "saya tolak", "penolakan", "rejected", "reject", "declined",
        ))
        and any(term in lowered_query for term in (
            "penerbangan alternatif", "alternatif tersebut", "penerbangan pengganti",
            "alternative flight", "rebooking",
        ))
    )
    tarmac_document_indices: list[int] = []
    tarmac_cfr_document_indices: list[int] = []

    for chunk_index, chunk in enumerate(chunks):
        title = str(chunk.get("title", "")).lower()
        section = str(chunk.get("section", "")).lower()
        body = str(chunk.get("text", "")).lower()
        url = str(chunk.get("url", "")).lower()
        title_section_body = title + " " + section + " " + body

        is_tarmac_document = (
            "transportation.gov/individuals/aviation-consumer-protection/tarmac-delays" in url
            or ("tarmac" in title and "delay" in title)
        )
        is_tarmac_cfr_document = "part-259/section-259.4" in url
        if tarmac_intent and is_tarmac_document:
            tarmac_document_indices.append(chunk_index)
            scores[chunk_index] += 2.0
        if tarmac_intent and is_tarmac_cfr_document:
            tarmac_cfr_document_indices.append(chunk_index)
            # The deterministic tarmac answer needs both the DOT guidance and
            # the underlying §259.4 text. Boost the CFR chunks so at least one
            # is retained in the six-source retrieval window.
            scores[chunk_index] += 4.0

        if explicit_rejection_alt_intent:
            is_explicit_rejection_chunk = (
                "260.6(a)(2)(ii)" in body
                or "penolakan eksplisit" in section
                or "explicit rejection" in section
            )
            is_explicit_rejection_deadline_chunk = (
                "7 business days" in body
                and "20 calendar days" in body
                and ("after you reject" in body or "after the consumer rejects" in body)
            )
            is_no_response_only_chunk = (
                "260.6(a)(2)(iii)(a)" in body
                or ("no response" in section and "explicit rejection" not in section)
            )
            if is_explicit_rejection_chunk:
                scores[chunk_index] += 2.5
            is_nonrefundable_obligation_chunk = (
                "260.6(a)(1)" in body
                and ("nonrefundable" in body or "non-refundable" in body)
                and "merchant of record" in body
            )
            if is_nonrefundable_obligation_chunk:
                scores[chunk_index] += 3.5
            if is_explicit_rejection_deadline_chunk or (
                "tenggat setelah penolakan eksplisit" in section
                and "7 business days" in body
                and "20 calendar days" in body
            ):
                scores[chunk_index] += 2.5
            if is_no_response_only_chunk:
                scores[chunk_index] *= 0.10

        # Give the actual §260.2 late-arrival definition a strong rank boost.
        # A general refund/payment chunk is not an adequate substitute.
        if arrival_intent and "260.2" in (title + " " + url):
            is_scheduled_arrival_definition = (
                "original scheduled arrival" in body
                and ("3 hours" in body or "three hours" in body)
                and ("scheduled to arrive" in body or "scheduled arrival" in body)
            )
            if is_scheduled_arrival_definition or "regulatory definition" in section or "late arrival" in section:
                scores[chunk_index] += 1.50

        # The unused fully-refundable fare question should retrieve the DOT's
        # dedicated section rather than the generic §260.6 payment-deadline text.
        if fully_refundable_intent and "transportation.gov" in url:
            is_fully_refundable_section = (
                "fully refundable ticket" in section
                or ("fully refundable" in body and ("does not use" in body or "not use" in body or "not used" in body))
            )
            if is_fully_refundable_section:
                scores[chunk_index] += 1.50

        # No-response questions need the exact automatic-refund trigger and
        # not merely the generic non-refundable ticket rule.
        if no_response_alt_intent:
            is_automatic_refund_section = (
                "automatic refunds" in section
                or "does not respond" in section
                or "rejects an offer or does not respond" in section
                or "if the passenger rejects an offer or does not respond" in section
            )
            has_departure_without_consumer_condition = (
                "flight departs without the consumer" in body
                or "flight departs without passenger" in body
                or "does not fly on the offered alternative" in body
            )
            if is_automatic_refund_section and has_departure_without_consumer_condition:
                scores[chunk_index] += 1.25

        if downgrade_intent and "downgrade" in title_section_body and (
            "difference between the original fare" in body or "fare difference" in body
        ):
            scores[chunk_index] += 1.00

    # For tarmac-delay questions, prefer the dedicated DOT guidance together
    # with the controlling §259.4 source. Generic refund chunks should not crowd
    # either source out of the retrieval window.
    if tarmac_intent and (tarmac_document_indices or tarmac_cfr_document_indices):
        tarmac_indices = sorted(
            set(tarmac_document_indices + tarmac_cfr_document_indices),
            key=lambda index: float(scores[index]),
            reverse=True,
        )
        # The deterministic answer requires a complete evidence bundle: DOT's
        # food/water, facilities/temperature, and notification guidance, plus
        # the controlling CFR deplaning distinction. Select those source
        # categories first, then fill remaining slots by relevance.
        ranked = []
        preferred_dot_sections = (
            ("FOOD AND POTABLE WATER", "FOOD AND WATER"),
            ("TOILETS", "LAVATORY", "CABIN TEMPERATURE"),
            ("STATUS NOTIFICATION", "UPDATES AND NOTIFICATIONS"),
        )
        for section_terms in preferred_dot_sections:
            candidates = [
                index for index in tarmac_document_indices
                if any(term in str(chunks[index].get("section", "")).upper() for term in section_terms)
            ]
            if candidates:
                ranked.append(max(candidates, key=lambda index: float(scores[index])))

        preferred_cfr_sections = [
            index for index in tarmac_cfr_document_indices
            if "THREE-HOUR LIMIT" in str(chunks[index].get("section", "")).upper()
            or "DEPARTING AND ARRIVING" in str(chunks[index].get("section", "")).upper()
        ]
        if preferred_cfr_sections:
            ranked.append(max(preferred_cfr_sections, key=lambda index: float(scores[index])))
        elif tarmac_cfr_document_indices:
            ranked.append(max(tarmac_cfr_document_indices, key=lambda index: float(scores[index])))

        ranked.extend(index for index in tarmac_indices if index not in ranked)
    else:
        ranked = scores.argsort()[::-1]
    results: list[dict] = []
    limit = max(1, min(int(top_k), 6))
    for index in ranked:
        score = float(scores[index])
        if score <= 0.005:
            continue
        item = chunks[int(index)]
        results.append({
            "title": item.get("title", "U.S. DOT document"),
            "section": item.get("section", ""),
            "url": item.get("url", ""),
            "excerpt": item.get("text", ""),
            "score": round(score, 4),
            "chunk_id": item.get("id", ""),
        })
        if len(results) >= limit:
            break
    return results
