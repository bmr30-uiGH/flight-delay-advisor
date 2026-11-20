from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

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
    del modified_ns
    path = Path(path_string)
    if not path.exists():
        return tuple()

    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if item.get("text") and item.get("url"):
                records.append(item)
    return tuple(records)


@lru_cache(maxsize=2)
def _get_vectorizer_and_matrix(modified_ns: int):
    """
    Build and cache the TF-IDF vectorizer and document matrix.

    The modified_ns (mtime nanoseconds) acts as a cache key — when the
    chunks file is rewritten by ingestion, the mtime changes and we rebuild.
    """
    chunks = _load_chunks(str(CHUNKS_PATH), modified_ns)
    if not chunks:
        return None, None, tuple()

    searchable_texts = [
        " ".join((item.get("title", ""), item.get("section", ""), item.get("text", "")))
        for item in chunks
    ]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=30000,
    )
    try:
        doc_matrix = vectorizer.fit_transform(searchable_texts)
    except ValueError:
        return None, None, chunks

    return vectorizer, doc_matrix, chunks


def retrieve_passenger_rights(query: str, top_k: int = 6) -> list[dict]:
    if not CHUNKS_PATH.exists() or not str(query or "").strip():
        return []

    modified_ns = CHUNKS_PATH.stat().st_mtime_ns
    vectorizer, doc_matrix, chunks = _get_vectorizer_and_matrix(modified_ns)
    if not chunks or vectorizer is None or doc_matrix is None:
        return []

    expanded_query = _expand_query(query)

    try:
        query_vec = vectorizer.transform([expanded_query])
        scores = cosine_similarity(query_vec, doc_matrix).ravel()
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

    # Tarmac delay intent - prioritize tarmac-specific documents
    tarmac_intent = any(term in lowered_query for term in (
        "tarmac", "landasan", "di landasan", "tertahan di landasan",
        "penundaan di landasan", "pesawat tertahan", "deplane", "turun dari pesawat",
    ))

    # Explicit rejection intent - prioritize §260.6(a)(2)(ii) and deadline chunks
    explicit_rejection_intent = (
        any(term in lowered_query for term in (
            "menolak", "menolaknya", "penolakan", "saya tolak", "reject",
            "rejected", "decline", "declined", "explicit rejection", "penolakan eksplisit"
        ))
        and any(term in lowered_query for term in (
            "penerbangan alternatif", "alternative flight", "penerbangan pengganti",
            "alternative", "rebooking", "alternatif"
        ))
    )

    for chunk_index, chunk in enumerate(chunks):
        title = str(chunk.get("title", "")).lower()
        section = str(chunk.get("section", "")).lower()
        body = str(chunk.get("text", "")).lower()
        url = str(chunk.get("url", "")).lower()
        title_section_body = title + " " + section + " " + body

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

        # Tarmac delay queries should prioritize tarmac-specific documents
        # from transportation.gov/tarmac-delays and eCFR part-259/section-259.4
        if tarmac_intent:
            is_tarmac_dot_source = (
                "transportation.gov/individuals/aviation-consumer-protection/tarmac-delays" in url
            )
            is_tarmac_cfr_source = (
                "part-259/section-259.4" in url
            )
            if is_tarmac_dot_source or is_tarmac_cfr_source:
                scores[chunk_index] += 2.00

        # Explicit rejection queries should prioritize §260.6(a)(2)(ii) and
        # deadline chunks, NOT the no-response (§260.6(a)(2)(iii)(A)) section.
        if explicit_rejection_intent:
            is_explicit_rejection_section = (
                "explicit rejection" in section
                or "260.6(a)(2)(ii)" in section
                or "automatic refunds" in section
            )
            is_deadline_section = (
                "7 business days" in body
                and "20 calendar days" in body
                and ("after you reject" in body or "after the consumer rejects" in body
                     or "after rejection" in body or "following rejection" in body
                     or "after explicit rejection" in body or "after an explicit rejection" in body
                     or "after the date you reject" in body
                     or "after the consumer explicitly rejects" in body
                     or "setelah penolakan" in body or "setelah konsumen menolak" in body)
            )
            is_no_response_section = (
                "no response" in section
                or "does not respond" in section
                or "260.6(a)(2)(iii)" in section
            )
            if is_explicit_rejection_section:
                scores[chunk_index] += 1.50
            if is_deadline_section:
                scores[chunk_index] += 1.25
            # Deprioritize no-response section for explicit rejection queries
            if is_no_response_section:
                scores[chunk_index] -= 0.50

        if downgrade_intent and "downgrade" in title_section_body and (
            "difference between the original fare" in body or "fare difference" in body
        ):
            scores[chunk_index] += 1.00

    ranked = scores.argsort()[::-1]
    results: list[dict] = []
    limit = max(1, min(int(top_k), 6))

    # For tarmac queries, only return tarmac-specific documents
    allowed_urls = None
    if tarmac_intent:
        allowed_urls = {
            "https://www.transportation.gov/individuals/aviation-consumer-protection/tarmac-delays",
            "https://www.ecfr.gov/current/title-14/chapter-II/subchapter-A/part-259/section-259.4",
        }

    for index in ranked:
        score = float(scores[index])
        if score <= 0.005:
            continue
        item = chunks[int(index)]
        url = item.get("url", "")

        # Filter non-tarmac documents for tarmac queries
        if allowed_urls is not None and url not in allowed_urls:
            continue

        results.append({
            "title": item.get("title", "U.S. DOT document"),
            "section": item.get("section", ""),
            "url": url,
            "excerpt": item.get("text", ""),
            "score": round(score, 4),
            "chunk_id": item.get("id", ""),
        })
        if len(results) >= limit:
            break
    return results
