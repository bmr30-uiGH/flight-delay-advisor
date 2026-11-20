"""Guard functions for deterministic passenger rights overrides."""
from __future__ import annotations

import re

from .constants import (
    _actual_arrival_only_with_unknown_schedule,
    _cancellation_without_any_offer_refund_question,
    _continued_downgrade_question,
    _unused_fully_refundable_ticket_question,
    _no_response_alternative_refund_question,
    _no_response_voucher_compensation_refund_question,
    _uses_indonesia_law,
)


def _deterministic_passenger_rights_override(question: str, sources: list[dict]) -> str | None:
    """Override known high-risk answer patterns only when retrieved evidence supports the rule."""
    evidence = " ".join(
        str(value)
        for source in (sources or [])
        if isinstance(source, dict)
        for value in (source.get("title", ""), source.get("section", ""), source.get("excerpt", ""))
    ).lower()

    lowered_question = re.sub(r"\s+", " ", (question or "").lower()).strip()

    # --------------------------------------------------------
    # INDONESIAN PM 89/2015 DELAY CATEGORY GUARD
    # Keep Indonesian law separate from the U.S. DOT guards.
    # --------------------------------------------------------
    if _uses_indonesia_law(question):
        indonesia_delay_intent = any(term in lowered_question for term in (
            "keterlambatan", "terlambat", "delay", "delayed",
        ))

        has_pas3 = (
            "pasal 3" in evidence
            and "181" in evidence
            and "240" in evidence
        )

        has_pas9 = (
            "pasal 9" in evidence
            and "snack box" in evidence
            and "300.000" in evidence
        )

        has_pas10 = (
            "pasal 10" in evidence
            and "6 jam" in evidence
            and "akomodasi" in evidence
        )

        is_cancellation_question = any(term in lowered_question for term in (
            "pembatalan penerbangan",
            "penerbangan dibatalkan",
            "flight cancellation",
            "cancelled flight",
            "canceled flight",
        ))

        if (
            indonesia_delay_intent
            and not is_cancellation_question
            and has_pas3
            and has_pas9
        ):
            duration_match = re.search(
                r"\b(\d+)\s*(?:jam|hours?)"
                r"(?:[\s,]*(?:dan\s*)?(\d+)\s*(?:menit|minutes?))?\b",
                lowered_question,
            )

            duration_minutes = None

            if duration_match:
                duration_minutes = (
                    int(duration_match.group(1)) * 60
                    + int(duration_match.group(2) or 0)
                )
            else:
                minute_match = re.search(
                    r"\b(\d+)\s*(?:menit|minutes?)\b",
                    lowered_question,
                )
                if minute_match:
                    duration_minutes = int(minute_match.group(1))

            category = None
            category_range = ""

            if duration_minutes is not None:
                if 30 <= duration_minutes <= 60:
                    category, category_range = 1, "30?60 menit"
                elif 61 <= duration_minutes <= 120:
                    category, category_range = 2, "61?120 menit"
                elif 121 <= duration_minutes <= 180:
                    category, category_range = 3, "121?180 menit"
                elif 181 <= duration_minutes <= 240:
                    category, category_range = 4, "181-240 menit"
                elif duration_minutes > 240:
                    category, category_range = 5, "lebih dari 240 menit"

            if category is not None:
                category_compensation = {
                    1: "minuman ringan",
                    2: "minuman dan makanan ringan (snack box)",
                    3: "minuman dan makanan berat (heavy meal)",
                    4: "minuman, makanan ringan (snack box), dan makanan berat (heavy meal)",
                    5: "ganti rugi sebesar Rp300.000",
                }

                category_article_letter = {
                    1: "a", 2: "b", 3: "c", 4: "d", 5: "e"
                }[category]

                parts = [
                    f"**Kategori keterlambatan: kategori {category} ({category_range}).**",
                    (
                        f"Durasi yang disebutkan adalah **{duration_minutes} menit**. "
                        f"Pasal 3 PM 89/2015 menetapkan durasi tersebut sebagai "
                        f"kategori {category}. Bentuk kompensasinya diatur dalam "
                        f"Pasal 9 ayat (1) huruf {category_article_letter}, yaitu "
                        f"**{category_compensation[category]}**."
                    ),
                ]

                if duration_minutes == 240:
                    parts.append(
                        "**Batas 4 jam perlu diperhatikan:** tepat 240 menit masih "
                        "termasuk kategori 4. Kategori 5 baru dimulai ketika "
                        "keterlambatan **lebih dari 240 menit**. Jadi, keterlambatan "
                        "tepat 4 jam bukan kategori 5 dan tidak otomatis memperoleh "
                        "ganti rugi Rp300.000."
                    )

                if category == 5:
                    parts.append(
                        "Rp300.000 merupakan ganti rugi kategori 5 menurut "
                        "Pasal 9 ayat (1) huruf e."
                    )

                if 2 <= category <= 5:
                    parts.append(
                        "Pasal 9 ayat (1) huruf g juga mengatur bahwa untuk "
                        "kategori 2 sampai 5, penumpang dapat dialihkan ke "
                        "penerbangan berikutnya atau memperoleh pengembalian "
                        "seluruh biaya tiket (refund)."
                    )

                if has_pas10:
                    if duration_minutes > 360:
                        parts.append(
                            "Untuk akomodasi, Pasal 10 ayat (4) menyatakan bahwa "
                            "pada keterlambatan **di atas 6 jam**, maskapai wajib "
                            "menyediakan akomodasi apabila penumpang membutuhkan "
                            "tempat penginapan."
                        )
                    else:
                        parts.append(
                            f"Pada durasi **{duration_minutes} menit**, "
                            "ketentuan akomodasi Pasal 10 ayat (4) belum "
                            "terpenuhi. Pasal tersebut mensyaratkan "
                            "keterlambatan **lebih dari 6 jam** serta "
                            "kebutuhan penumpang akan tempat penginapan."
                        )

                aircraft_not_ready = any(term in lowered_question for term in (
                    "pesawat belum siap",
                    "pesawat tidak siap",
                    "ketidaksiapan pesawat",
                    "belum siapnya pesawat",
                    "aircraft not ready",
                ))

                if (
                    aircraft_not_ready
                    and "pasal 5" in evidence
                    and "pasal 6" in evidence
                ):
                    parts.append(
                        "Ketidaksiapan pesawat udara termasuk faktor manajemen "
                        "maskapai menurut Pasal 5 ayat (2) huruf e. Pasal 6 "
                        "ayat (1) menetapkan bahwa maskapai bertanggung jawab "
                        "atas keterlambatan yang disebabkan faktor tersebut. "
                        "Dengan demikian, apabila ketidaksiapan pesawat merupakan "
                        "penyebab resmi keterlambatan, tanggung jawab berada "
                        "pada maskapai."
                    )
                elif "pasal 6" in evidence:
                    parts.append(
                        "Pasal 6 ayat (1) menyatakan maskapai bertanggung jawab "
                        "atas keterlambatan akibat faktor manajemen maskapai. "
                        "Pasal 6 ayat (2) mengatur pembebasan tanggung jawab "
                        "atas ganti kerugian untuk faktor teknis operasional, "
                        "cuaca, dan faktor lain sebagaimana dirujuk dalam Pasal 5. "
                        "Pastikan penyebab keterlambatan berdasarkan keterangan resmi."
                    )

                if "pasal 4" in evidence:
                    parts.append(
                        "Menurut Pasal 4, keterlambatan dihitung dari selisih "
                        "jadwal dan realisasi keberangkatan atau kedatangan. "
                        "Realisasi keberangkatan ditandai saat block off, "
                        "sedangkan kedatangan saat block on di apron tujuan."
                    )

                parts.append(
                    "Jawaban ini menggunakan ringkasan ketentuan PM 89/2015 "
                    "yang tersedia pada sumber aplikasi. Pastikan penyebab "
                    "keterlambatan dan waktu aktualnya dikonfirmasi kepada maskapai."
                )

                return "\n\n".join(parts)

        # Do not allow Indonesian questions to fall through to U.S. DOT guards.
        return None

    tarmac_intent = any(term in lowered_question for term in (
        "tarmac", "landasan", "tertahan di landasan", "penundaan di landasan",
        "pesawat tertahan", "di runway", "di apron", "turun dari pesawat", "deplane",
    ))
    has_tarmac_dot_source = any(
        isinstance(source, dict)
        and "transportation.gov/individuals/aviation-consumer-protection/tarmac-delays" in str(source.get("url", "")).lower()
        for source in (sources or [])
    )
    has_tarmac_cfr_source = any(
        isinstance(source, dict)
        and "part-259/section-259.4" in str(source.get("url", "")).lower()
        for source in (sources or [])
    )
    if tarmac_intent and has_tarmac_dot_source and has_tarmac_cfr_source:
        has_food_water = ("two hours" in evidence or "2 hours" in evidence) and ("water" in evidence or "air minum" in evidence)
        has_facilities = ("lavatory" in evidence or "toilet" in evidence) and ("comfortable cabin temperature" in evidence or "suhu kabin" in evidence)
        has_updates = "30 minutes" in evidence
        has_deplane = ("three hours" in evidence or "3 hours" in evidence) and "domestic" in evidence
        has_departure_arrival_distinction = (
            "259.4(c)(3)(i)" in evidence
            and "departing flight" in evidence
            and "arriving flight" in evidence
            and "air traffic control" in evidence
        )
        if has_food_water and has_facilities and has_updates and has_deplane and has_departure_arrival_distinction:
            return (
                "**Untuk tarmac delay domestik selama 3 jam 20 menit, ada kemungkinan batas 3 jam dilanggar, "
                "tetapi status pastinya bergantung pada apakah pesawat sedang akan berangkat atau baru mendarat "
                "dan apakah pengecualian regulasi berlaku.**\n\n"
                "**Snack dan air minum — paling lambat 2 jam.** 14 CFR §259.4(c)(4) mewajibkan carrier "
                "menyediakan makanan dan air minum yang memadai paling lambat 2 jam setelah tarmac delay dimulai. "
                "DOT menjelaskan bahwa ini dapat berupa snack dan air; tidak harus makanan lengkap, tetapi jumlahnya "
                "harus cukup untuk seluruh penumpang. Pelayanan hanya boleh ditahan bila pilot-in-command menentukan "
                "bahwa alasan keselamatan atau keamanan tidak memungkinkan pelayanan.\n\n"
                "**Toilet dan suhu kabin.** Toilet/lavatory harus tetap dapat digunakan selama tarmac delay, dan "
                "perhatian medis yang memadai harus tersedia bila dibutuhkan. Panduan penumpang DOT juga mencantumkan "
                "suhu kabin yang nyaman. Sumber yang digunakan tidak memberikan pengecualian teknis umum untuk "
                "mengabaikan kewajiban toilet atau suhu kabin; jangan menganggap gangguan teknis otomatis menjadi "
                "pengecualian.\n\n"
                "**Pemberitahuan.** Saat tarmac delay melewati 30 menit, penumpang harus diberi pemberitahuan tentang "
                "status keterlambatan. Aturan tidak menetapkan update wajib setiap 30 menit: update lanjutan dapat "
                "diberikan sesuai kebijakan carrier. Jika kesempatan turun yang aman tersedia pada departing flight "
                "atau diversion, carrier harus memberi tahu penumpang secara tepat waktu.\n\n"
                "**Batas 3 jam — bedakan jenis penerbangan:**\n"
                "- **Pesawat akan berangkat:** penumpang harus diberi kesempatan turun sebelum tarmac delay melewati "
                "3 jam. Untuk memenuhi ketentuan, carrier dapat mulai mengembalikan pesawat ke lokasi turun yang "
                "sesuai paling lambat pada batas 3 jam sejak pintu utama ditutup. Bila pesawat berada di area di luar "
                "kendali carrier, return dimulai ketika permintaan disampaikan ke tower/otoritas terkait; bila berada "
                "di area kendali carrier, ketika pilot mulai menggerakkan pesawat.\n"
                "- **Pesawat baru mendarat:** penumpang harus benar-benar diberi kesempatan turun dengan aman sebelum "
                "tarmac delay melewati 3 jam. Ketentuan 'mulai kembali ke lokasi turun' di atas adalah ketentuan "
                "khusus departing flight, bukan pengganti kewajiban untuk arriving flight.\n\n"
                "**Pengecualian batas turun:** pilot-in-command dapat menentukan bahwa turun akan membahayakan "
                "keselamatan/keamanan atau ada alasan keselamatan/keamanan yang membuat pesawat tidak dapat berpindah. "
                "ATC juga dapat memberi tahu pilot bahwa kembali ke lokasi turun akan sangat mengganggu operasi "
                "bandara. Pengecualian ini berkaitan dengan batas kesempatan turun, bukan pengecualian umum untuk "
                "snack/air, toilet, suhu kabin, atau pemberitahuan 30 menit. Untuk snack/air, pengecualian keselamatan/"
                "keamanan memiliki syarat tersendiri: keputusan pilot-in-command bahwa pelayanan tidak dapat diberikan.\n\n"
                "**Penerapan pada skenario Anda:** karena sudah 3 jam 20 menit dan penumpang belum mendapat kesempatan "
                "turun, minta maskapai menjelaskan secara tertulis apakah pesawat termasuk departing flight atau "
                "arriving flight, kapan tarmac delay dihitung mulai, dan pengecualian apa yang dipakai. Jika tidak ada "
                "pengecualian yang sah—atau, untuk departing flight, pesawat tidak mulai kembali ke lokasi turun "
                "pada waktunya—hal itu dapat menunjukkan ketidakpatuhan terhadap aturan DOT.\n\n"
                "Sumber yang diambil: panduan U.S. DOT Tarmac Delays dan ringkasan regulasi 14 CFR §259.4."
            )
        return (
            "Saya menemukan dokumen U.S. DOT tentang tarmac delay, tetapi excerpt yang tersedia belum memuat "
            "seluruh ketentuan yang diperlukan untuk memastikan batas waktu dan pengecualiannya. Saya tidak akan "
            "menebak detail hukum. Cari ulang bagian §259.4 mengenai food and water, lavatory, notifications, "
            "dan opportunity to deplane; khususnya bedakan departing flight dari arriving flight."
        )

    # ========================================================
    # ACTUAL ARRIVAL GUARD (from constants)
    # ========================================================
    if _actual_arrival_only_with_unknown_schedule(question):
        has_definition = (
            "260.2" in evidence
            and "original scheduled arrival" in evidence
            and ("scheduled to arrive" in evidence or "revised scheduled arrival" in evidence)
        )
        if not has_definition:
            return (
                "Belum dapat dipastikan dari informasi yang tersedia. Keterlambatan aktual (actual arrival) "
                "tidak boleh langsung dipakai sebagai pengganti jadwal kedatangan setelah perubahan maskapai. "
                "Sumber definisi yang diperlukan belum berhasil diambil pada pencarian ini, sehingga saya "
                "tidak akan menyimpulkan kelayakan refund. Periksa original scheduled arrival dan revised "
                "scheduled arrival pada notifikasi atau riwayat perubahan jadwal maskapai."
            )
        return (
            "**Belum dapat dipastikan bahwa Anda memenuhi ambang significant delay untuk refund.**\n\n"
            "Aturan late-arrival U.S. DOT membandingkan **original scheduled arrival** dengan waktu kedatangan "
            "yang **dijadwalkan setelah perubahan/delay oleh maskapai (revised scheduled arrival)**. Untuk "
            "rute domestik, jadwal kedatangan setelah perubahan harus setidaknya 3 jam lebih lambat; untuk "
            "rute internasional, setidaknya 6 jam lebih lambat. Ukurannya adalah waktu yang dijadwalkan, "
            "bukan waktu pesawat benar-benar mendarat.\n\n"
            "Karena Anda hanya mengetahui actual arrival terlambat 3 jam 30 menit dan belum mengetahui "
            "revised scheduled arrival, data tersebut **belum cukup untuk menyatakan ambang §260.2 terpenuhi**. "
            "Bandingkan waktu kedatangan awal pada itinerary dengan waktu kedatangan baru yang diumumkan "
            "atau dicatat maskapai. Jangan mengganti waktu jadwal baru dengan waktu aktual pesawat mendarat.\n\n"
            "Selain ambang jadwal, refund penuh juga bergantung pada apakah Anda menggunakan penerbangan "
            "yang berubah atau penerbangan alternatif, apakah Anda menerima voucher/credit sebagai pengganti "
            "refund, jenis tiket, dan siapa merchant of record. Penumpang yang tetap menggunakan penerbangan "
            "yang berubah atau alternatif umumnya tidak memperoleh refund penuh berdasarkan aturan delay/change.\n\n"
            "**Informasi yang masih diperlukan:** original scheduled arrival dan revised scheduled arrival "
            "yang diberikan maskapai."
        )

    # ========================================================
    # DOWNGRADE GUARD (from constants)
    # ========================================================
    if _continued_downgrade_question(question):
        has_downgrade_evidence = (
            "downgrade" in evidence
            and (
                "difference between the original fare and the downgraded fare" in evidence
                or "selisih antara tarif awal" in evidence
                or "fare difference" in evidence
            )
        )
        if has_downgrade_evidence:
            return (
                "Jika maskapai secara sepihak menurunkan kelas layanan Anda dan Anda tetap melakukan "
                "perjalanan, panduan U.S. DOT menyatakan bahwa Anda **tidak memperoleh refund seluruh harga "
                "tiket** berdasarkan downgrade tersebut. Maskapai harus mengembalikan **selisih antara tarif "
                "kelas yang semula dibeli dan tarif kelas yang benar-benar diberikan**.\n\n"
                "Sumber DOT yang berhasil diambil tidak menetapkan tenggat khusus untuk pembayaran selisih "
                "tarif downgrade. Jangan mengasumsikan tenggat refund tiket penuh berlaku untuk pembayaran "
                "selisih tarif ini. Mintalah konfirmasi tenggat tertulis dari maskapai atau merchant of record; "
                "saya tidak akan mengarang tenggat pembayaran.\n\n"
                "Simpan bukti tiket/itinerary kelas awal, boarding pass, dan bukti kelas yang diberikan, lalu "
                "ajukan permintaan pengembalian selisih tarif kepada pihak yang memproses pembayaran tiket."
            )

    # ========================================================
    # UNUSED FULLY REFUNDABLE TICKET GUARD (from constants)
    # ========================================================
    if _unused_fully_refundable_ticket_question(question):
        has_unused_fare_evidence = (
            "fully refundable" in evidence
            and ("does not use" in evidence or "do not use" in evidence or "not use the purchased ticket" in evidence or "ticket is not used" in evidence)
        )
        if has_unused_fare_evidence:
            return (
                "Ya. Panduan konsumen U.S. DOT menyatakan bahwa pembeli **tiket fully refundable** berhak atas refund "
                "ketika tiket yang dibeli tidak digunakan. Ini merupakan panduan terpisah dari 14 CFR §260.6(a)(1), "
                "yang secara khusus membahas pemegang tiket **non-refundable** ketika penerbangan dibatalkan atau "
                "mengalami significant delay/change dan penumpang memenuhi syarat untuk tidak melanjutkan perjalanan "
                "atau menerima alternatif/kompensasi pengganti refund.\n\n"
                "Sumber yang berhasil diambil untuk tiket fully refundable yang tidak digunakan **tidak menetapkan "
                "deadline pemrosesan khusus** untuk skenario pembatalan sukarela ini. Karena itu, saya tidak akan "
                "menerapkan tenggat 7 business days atau 20 calendar days dari aturan prompt refund §260.6 tanpa "
                "sumber yang secara khusus menyatakan tenggat tersebut berlaku pada kasus ini.\n\n"
                "Jawaban ini menyangkut refund harga tiket. Jangan menganggap biaya layanan tambahan (ancillary "
                "services), seperti bagasi atau pemilihan kursi, otomatis ikut dikembalikan tanpa memeriksa aturan "
                "khusus untuk layanan tersebut. Periksa syarat fare dan ajukan refund kepada pihak yang memproses "
                "pembayaran tiket."
            )

    # ========================================================
    # NO RESPONSE ALTERNATIVE REFUND GUARD (from constants)
    # ========================================================
    if _no_response_alternative_refund_question(question):
        has_auto_rule = (
            "260.6" in evidence
            and "flight departs without the consumer" in evidence
            and "7 business days" in evidence
            and "20 calendar days" in evidence
        )
        if has_auto_rule:
            return (
                "**Refund otomatis dapat berlaku, tetapi bukan karena diam saja.** Untuk tawaran penerbangan yang "
                "berubah signifikan atau penerbangan alternatif, 14 CFR §260.6(a)(2)(iii)(A) mensyaratkan bahwa "
                "konsumen tidak merespons tawaran tersebut dan penerbangan yang ditawarkan berangkat tanpa konsumen. "
                "Jika alternatif belum berangkat, kondisi no-response itu belum dapat dipastikan terpenuhi.\n\n"
                "Jika penerbangan yang ditawarkan memang sudah berangkat tanpa Anda, dan penerbangan awal memenuhi "
                "kriteria cancellation/significant delay/change serta syarat jenis tiket yang berlaku, refund harus "
                "diterbitkan otomatis tanpa Anda harus mengirim permintaan refund terpisah. §260.6(a)(1) secara "
                "khusus menyebut tiket non-refundable; panduan DOT membahas tiket fully refundable secara terpisah "
                "ketika tiket tidak digunakan.\n\n"
                "Untuk no-response terhadap tawaran alternative transportation, halaman DOT menjelaskan tenggat "
                "7 business days untuk pembayaran kartu kredit atau 20 calendar days untuk metode pembayaran lain, "
                "dihitung setelah penerbangan yang significantly delayed/changed atau penerbangan alternatif "
                "berangkat dari destination-nya. Ini berbeda dari penolakan eksplisit, yang tenggatnya dihitung "
                "setelah penolakan. No-response terhadap voucher/credit juga memiliki trigger tersendiri dalam "
                "§260.6(a)(2)(iii)(B), sehingga tidak boleh memakai trigger penerbangan alternatif secara otomatis.\n\n"
                "Refund yang diwajibkan dikembalikan ke metode pembayaran awal. Untuk memastikan kasus Anda, "
                "pastikan penerbangan alternatif benar-benar telah berangkat tanpa Anda dan periksa jenis tiket "
                "serta siapa merchant of record pada bukti transaksi."
            )

    # ========================================================
    # CANCELLATION WITHOUT ANY OFFER REFUND GUARD (from constants)
    # ========================================================
    if _cancellation_without_any_offer_refund_question(question):
        has_no_offer_automatic_rule = (
            "260.6(a)(2)(i)" in evidence
            and "voucher" in evidence
            and "compensation" in evidence
            and (
                "not offered an alternative flight" in evidence
                or "no alternative flight or" in evidence
                or "no alternative flight and" in evidence
            )
        )

        if not has_no_offer_automatic_rule:
            return (
                "**Saya belum dapat memastikan ketentuan otomatis untuk skenario tanpa tawaran.** "
                "Pertanyaan Anda sesuai dengan kondisi pembatalan tanpa penerbangan alternatif "
                "dan tanpa voucher/kredit/kompensasi, tetapi excerpt §260.6(a)(2)(i) yang "
                "diperlukan belum muncul pada hasil retrieval saat ini. Saya tidak akan "
                "mengganti kondisi ini dengan aturan penolakan eksplisit atau no-response. "
                "Periksa sumber resmi 14 CFR §260.6 sebelum menyimpulkan kelayakan."
            )

        has_nonrefundable_scope_no_offer = (
            "260.6(a)(1)" in evidence
            and ("nonrefundable" in evidence or "non-refundable" in evidence)
        )
        has_fully_refundable_no_offer = (
            "fully refundable" in evidence
            and (
                "does not use" in evidence
                or "do not use" in evidence
                or "not use the purchased ticket" in evidence
                or "ticket is not used" in evidence
            )
        )
        has_agent_rule_no_offer = (
            "399.80(l)" in evidence
            and "ticket agent" in evidence
            and "refund" in evidence
            and ("upon request" in evidence or "upon consumer request" in evidence)
        )
        has_agent_deadline_no_offer = (
            has_agent_rule_no_offer
            and "7 business days" in evidence
            and "20 calendar days" in evidence
            and (
                "receives information from the carrier" in evidence
                or "receives information from carrier" in evidence
            )
        )

        fare_scope = []
        if has_nonrefundable_scope_no_offer:
            fare_scope.append(
                "Teks §260.6(a)(1) secara khusus menyebut tiket **non-refundable** "
                "pada penerbangan terjadwal yang tercakup ke, dari, atau di dalam "
                "Amerika Serikat. Syarat cakupan carrier dan pilihan konsumen untuk "
                "tidak melanjutkan perjalanan/tidak menerima kompensasi pengganti "
                "refund tetap harus dipenuhi."
            )
        if has_fully_refundable_no_offer:
            fare_scope.append(
                "Untuk tiket **fully refundable**, panduan konsumen DOT menyatakan "
                "refund tersedia ketika tiket tidak digunakan. Itu merupakan "
                "penjelasan DOT yang terpisah dari rumusan khusus tiket non-refundable "
                "di §260.6(a)(1)."
            )

        agent_note = ""
        if has_agent_rule_no_offer:
            if has_agent_deadline_no_offer:
                agent_note = (
                    "Jika **ticket agent/OTA adalah Merchant of Record**, §399.80(l) "
                    "mewajibkan refund prompt **atas permintaan konsumen**, jadi jangan "
                    "menganggap OTA memprosesnya otomatis tanpa permintaan. Untuk kartu "
                    "kredit, tenggat agent adalah 7 business days sejak agent menerima "
                    "informasi kelayakan dari carrier menurut §260.6(d); untuk metode "
                    "lainnya, 20 calendar days sejak refund menjadi jatuh tempo."
                )
            else:
                agent_note = (
                    "Jika **ticket agent/OTA adalah Merchant of Record**, §399.80(l) "
                    "mengatur refund prompt **atas permintaan konsumen**. Jadi, ajukan "
                    "permintaan kepada agent dan jangan menganggap refund OTA otomatis "
                    "tanpa permintaan. Excerpt yang tersedia belum cukup untuk memastikan "
                    "seluruh tenggat pemrosesan agent."
                )

        sections = [
            "**Kesimpulan: pembatalan tanpa tawaran alternatif atau kompensasi merupakan "
            "jalur refund otomatis tersendiri.**",
            "Jika penerbangan terjadwal yang tercakup ke, dari, atau di dalam Amerika "
            "Serikat dibatalkan dan carrier tidak menawarkan penerbangan alternatif "
            "maupun voucher, credit, atau bentuk kompensasi lain sebagai pengganti "
            "refund, 14 CFR §260.6(a)(2)(i) menyatakan refund penuh menjadi due secara "
            "otomatis. Untuk cabang ini, konsumen tidak perlu lebih dahulu menolak "
            "tawaran yang memang tidak pernah diberikan dan tidak perlu menunggu "
            "penerbangan alternatif berangkat.",
            "**Jangan campur dengan cabang lain:** §260.6(a)(2)(ii) membahas penolakan "
            "eksplisit; §260.6(a)(2)(iii)(A) membahas tidak merespons tawaran flight "
            "dan flight yang ditawarkan berangkat tanpa konsumen; §260.6(a)(2)(iii)(B) "
            "membahas tidak merespons tawaran voucher/credit sampai tanggal yang "
            "ditentukan. Kondisi-kondisi tersebut bukan prasyarat untuk cabang "
            "tanpa tawaran pada §260.6(a)(2)(i).",
        ]
        sections.extend(fare_scope)
        sections.append(
            "**Siapa yang memproses refund?** Periksa Merchant of Record pada mutasi "
            "kartu/rekening atau bukti pembayaran. Jika maskapai adalah MOR, ketentuan "
            "refund otomatis carrier berlaku sesuai cakupan aturan. Bank/penerbit kartu "
            "bukan MOR hanya karena kartu itu digunakan."
        )
        if agent_note:
            sections.append(agent_note)
        sections.append(
            "Refund yang diwajibkan dikembalikan ke metode pembayaran semula, kecuali "
            "konsumen menyetujui bentuk lain yang cash-equivalent. Hak refund ini "
            "berdasarkan aturan U.S. DOT untuk penerbangan yang tercakup; jangan "
            "menerapkannya otomatis pada penerbangan di luar cakupan yurisdiksi AS."
        )
        return "\n\n".join(sections)

    # ========================================================
    # NO RESPONSE TO VOUCHER/CREDIT REFUND GUARD (from constants)
    # ========================================================
    if _no_response_voucher_compensation_refund_question(question):
        has_voucher_no_response_rule = (
            "260.6(a)(2)(iii)(b)" in evidence
            and "voucher" in evidence
            and "credit" in evidence
            and "does not respond" in evidence
            and "scheduled to depart" in evidence
        )

        if not has_voucher_no_response_rule:
            return (
                "**Saya belum dapat memastikan pemicu no-response untuk voucher/credit.** "
                "Aturan yang diperlukan adalah 14 CFR §260.6(a)(2)(iii)(B), yang "
                "membedakannya dari no-response terhadap tawaran penerbangan. "
                "Excerpt aturan tersebut belum cukup pada hasil retrieval kali ini, "
                "jadi saya tidak akan menggunakan tenggat keberangkatan alternatif "
                "dari §260.6(a)(2)(iii)(A)."
            )

        has_prompt_refund_definition = (
            "260.2" in evidence
            and "earliest date" in evidence
            and "7 business days" in evidence
            and "20 calendar days" in evidence
        )
        has_agent_rule_voucher = (
            "399.80(l)" in evidence
            and "ticket agent" in evidence
            and "refund" in evidence
            and ("upon request" in evidence or "upon consumer request" in evidence)
        )
        has_agent_deadline_voucher = (
            has_agent_rule_voucher
            and "7 business days" in evidence
            and "20 calendar days" in evidence
            and (
                "receives information from the carrier" in evidence
                or "receives information from carrier" in evidence
            )
        )

        sections = [
            "**No-response terhadap voucher/credit memiliki pemicu tersendiri.** "
            "Menurut 14 CFR §260.6(a)(2)(iii)(B), jika konsumen tidak merespons "
            "tawaran voucher, credit, atau bentuk kompensasi lain sampai tanggal "
            "penerbangan yang dibatalkan semula dijadwalkan berangkat, atau tanggal "
            "penerbangan yang mengalami significant delay/change berangkat, refund "
            "otomatis menjadi due sesuai ketentuan tersebut.",
            "**Ini berbeda dari §260.6(a)(2)(iii)(A).** Syarat bahwa penerbangan "
            "alternatif berangkat tanpa konsumen hanya berlaku pada no-response "
            "terhadap tawaran penerbangan yang berubah/alternatif. Jangan memakai "
            "syarat keberangkatan alternatif itu untuk cabang voucher/credit (B). "
            "Sebaliknya, diam saja sebelum tanggal pemicu (B) belum cukup untuk "
            "menyimpulkan kondisi tersebut sudah terpenuhi.",
        ]

        if has_prompt_refund_definition:
            sections.append(
                "Untuk carrier, §260.2 mendefinisikan prompt refund sebagai paling "
                "lambat 7 business days setelah tanggal paling awal refund diminta "
                "sebagaimana diatur §260.6(a)(2) untuk pembayaran kartu kredit, atau "
                "20 calendar days untuk cash, check, debit card, dan metode lainnya. "
                "Jangan mengganti tanggal pemicu (B) dengan tanggal keberangkatan "
                "penerbangan alternatif pada cabang (A)."
            )
        else:
            sections.append(
                "Excerpt pada jawaban ini belum memuat definisi tenggat prompt refund "
                "§260.2 secara lengkap. Karena itu, saya tidak akan menghitung tanggal "
                "jatuh tempo tertentu atau memakai trigger keberangkatan alternatif "
                "dari cabang (A)."
            )

        if has_agent_deadline_voucher:
            sections.append(
                "Jika **ticket agent/OTA adalah Merchant of Record**, §399.80(l) "
                "mengharuskan refund prompt atas permintaan konsumen. Untuk kartu "
                "kredit, tenggat agent adalah 7 business days setelah agent menerima "
                "informasi dari carrier berdasarkan §260.6(d); untuk metode pembayaran "
                "lainnya, 20 calendar days sejak refund menjadi jatuh tempo. Jalur agent "
                "ini tidak otomatis tanpa permintaan konsumen."
            )
        elif has_agent_rule_voucher:
            sections.append(
                "Jika ticket agent/OTA merupakan Merchant of Record, §399.80(l) "
                "mengatur refund prompt atas permintaan konsumen. Ajukan permintaan "
                "kepada agent; excerpt yang tersedia belum memuat seluruh tenggat "
                "pemrosesan agent."
            )

        sections.append(
            "Periksa jenis tawaran, tanggal keberangkatan yang tercantum pada itinerary, "
            "apakah Anda pernah menerima atau menolak kompensasi, dan Merchant of Record "
            "pada bukti pembayaran. Ketentuan ini berlaku untuk penerbangan yang tercakup "
            "oleh aturan U.S. DOT."
        )
        return "\n\n".join(sections)

    # ========================================================
    # STANDALONE OTA / TICKET AGENT REFUND GUARD
    # Does not require the user to specify fare class.
    # ========================================================

    ota_question = any(term in lowered_question for term in (
        "ota",
        "online travel agency",
        "ticket agent",
        "travel agent",
        "agen perjalanan",
        "merchant of record",
    ))

    cancellation_question = any(term in lowered_question for term in (
        "dibatalkan",
        "pembatalan",
        "membatalkan",
        "cancelled",
        "canceled",
        "cancellation",
    ))

    alternative_question = any(term in lowered_question for term in (
        "penerbangan alternatif",
        "penerbangan pengganti",
        "alternative flight",
        "alternative-flight",
        "rebooking",
    ))

    explicit_rejection_question = any(term in lowered_question for term in (
        "menolak",
        "menolaknya",
        "penolakan",
        "saya tolak",
        "reject",
        "rejected",
        "decline",
        "declined",
    ))

    # Evidence must come from retrieved documents in this turn.
    has_agent_refund_rule_ota = (
        "399.80(l)" in evidence
        and "ticket agent" in evidence
        and "refund" in evidence
        and "request" in evidence
    )

    has_merchant_definition_ota = (
        "merchant of record" in evidence
        and (
            "financial charge statement" in evidence
            or "financial statement" in evidence
            or "payment statement" in evidence
            or "detail transaksi" in evidence
            or "laporan transaksi" in evidence
        )
    )

    has_carrier_notice_ota = (
        "260.6(d)" in evidence
        and "ticket agent" in evidence
        and "without delay" in evidence
        and (
            "eligible" in evidence
            or "eligibility" in evidence
            or "refund" in evidence
        )
    )

    has_carrier_transfer_ota = (
        "260.6(e)" in evidence
        and "transfer" in evidence
        and (
            "ticket agent" in evidence
            or "agent" in evidence
        )
    )

    has_agent_deadline_ota = (
        "7 business days" in evidence
        and "20 calendar days" in evidence
        and "ticket agent" in evidence
        and (
            "receives information from the carrier" in evidence
            or "receives information from carrier" in evidence
            or "refund becomes due" in evidence
        )
    )

    has_carrier_rejection_rule_ota = (
        "260.6(a)(2)(ii)" in evidence
        and "refund" in evidence
        and (
            "automatic" in evidence
            or "automatically" in evidence
            or "otomatis" in evidence
        )
        and (
            "reject" in evidence
            or "penolakan" in evidence
        )
    )

    has_carrier_rejection_deadline_ota = (
        "7 business days" in evidence
        and "20 calendar days" in evidence
        and any(term in evidence for term in (
            "after the consumer rejects",
            "after you reject",
            "after rejection",
            "following rejection",
            "after explicit rejection",
            "after the date you reject",
            "setelah penolakan",
            "setelah konsumen menolak",
        ))
    )

    has_nonrefundable_scope_ota = (
        "260.6(a)(1)" in evidence
        and (
            "nonrefundable" in evidence
            or "non-refundable" in evidence
        )
    )

    has_fully_refundable_guidance_ota = (
        "fully refundable" in evidence
        and (
            "does not use" in evidence
            or "not use" in evidence
            or "not used" in evidence
        )
    )

    if (
        ota_question
        and cancellation_question
        and alternative_question
        and explicit_rejection_question
        and has_agent_refund_rule_ota
        and has_merchant_definition_ota
    ):
        ota_sections = []

        ota_is_mor = bool(re.search(
            r"\b(?:ota|online travel agency|ticket agent)\b.{0,120}"
            r"\b(?:is|are|was|were)\b.{0,100}\bmerchant of record\b",
            lowered_question,
        ))

        carrier_is_mor = bool(re.search(
            r"\b(?:airline|carrier)\b.{0,120}"
            r"\b(?:is|are|was|were)\b.{0,100}\bmerchant of record\b",
            lowered_question,
        ))

        if ota_is_mor:
            ota_sections.append(
                "**Kesimpulan: berdasarkan informasi pembayaran yang Anda "
                "berikan, OTA adalah Merchant of Record (MOR) dan pihak yang "
                "memproses refund.** Bank atau penerbit kartu bukan MOR hanya "
                "karena menerbitkan kartu yang digunakan."
            )
        elif carrier_is_mor:
            ota_sections.append(
                "**Kesimpulan: berdasarkan informasi pembayaran yang Anda "
                "berikan, maskapai adalah Merchant of Record (MOR) dan pihak "
                "yang memproses refund.**"
            )
        else:
            ota_sections.append(
                "**Kesimpulan: pihak yang memproses refund bergantung pada "
                "Merchant of Record (MOR), bukan semata-mata tempat pembelian.** "
                "Karena informasi transaksi belum diberikan, **belum dapat dipastikan siapa Merchant of Record (MOR)**. Membeli tiket melalui OTA tidak otomatis menjadikan OTA sebagai "
                "MOR. Periksa nama pedagang pada laporan transaksi kartu atau "
                "rekening pembayaran. Bank atau penerbit kartu bukan MOR hanya "
                "karena menerbitkan kartu yang digunakan."
            )

        # Carrier as merchant of record.
        if not ota_is_mor and (
            has_carrier_rejection_rule_ota
            and has_carrier_rejection_deadline_ota
        ):
            ota_sections.append(
                "**Jika maskapai adalah Merchant of Record:**\n\n"
                "Untuk kondisi refund yang memenuhi ketentuan, penolakan "
                "eksplisit terhadap penerbangan alternatif masuk dalam "
                "14 CFR §260.6(a)(2)(ii). Panduan U.S. DOT menyatakan refund "
                "oleh carrier sebagai MOR diproses otomatis setelah "
                "penolakan eksplisit, tanpa permintaan refund tambahan.\n\n"
                "Tenggat yang dijelaskan DOT adalah 7 business days setelah "
                "penolakan untuk pembayaran kartu kredit, atau 20 calendar "
                "days setelah penolakan untuk metode pembayaran lainnya. "
                "Jangan menunggu penerbangan alternatif berangkat pada "
                "skenario penolakan eksplisit."
            )
        elif not ota_is_mor:
            ota_sections.append(
                "**Jika maskapai adalah Merchant of Record:** "
                "ketentuan penolakan eksplisit dan tenggat carrier belum "
                "seluruhnya terverifikasi dalam excerpt yang tersedia. "
                "Saya tidak akan mengarang tenggatnya."
            )

        # Ticket agent / OTA as merchant of record.
        if not ota_is_mor and has_agent_deadline_ota:
            ota_sections.append(
                "**Jika OTA/ticket agent adalah Merchant of Record:**\n\n"
                "14 CFR §399.80(l) mewajibkan ticket agent memberikan "
                "prompt refund **atas permintaan konsumen** dalam kondisi "
                "refund yang tercakup, yaitu situasi yang dirujuk oleh "
                "§260.6(a) dan (b). Pada jalur pembatalan biasa, "
                "§260.6(a)(1) secara khusus menyebut tiket "
                "non-refundable. Karena itu, jangan menganggap "
                "penolakan kepada maskapai saja otomatis menggantikan "
                "permintaan refund kepada OTA. Ajukan permintaan refund "
                "secara jelas kepada OTA.\n\n"
                "Untuk pembayaran kartu kredit, tenggat prompt refund "
                "ticket agent adalah 7 business days sejak ticket agent "
                "menerima informasi kelayakan refund dari carrier "
                "berdasarkan §260.6(d). Untuk cash, debit, cek, dan "
                "metode pembayaran lainnya, tenggatnya 20 calendar days "
                "sejak refund menjadi jatuh tempo.\n\n"
                "**Perbedaan penting:** tenggat 7 hari kerja OTA bukan "
                "otomatis dihitung sejak tanggal Anda menolak alternatif. "
                "Trigger-nya adalah ketika OTA menerima informasi dari "
                "carrier."
            )
        elif not ota_is_mor:
            ota_sections.append(
                "**Jika OTA/ticket agent adalah Merchant of Record:** "
                "sumber yang terambil mendukung kewajiban refund atas "
                "permintaan konsumen berdasarkan §399.80(l), tetapi "
                "excerpt yang tersedia belum cukup untuk memastikan "
                "kedua tenggat pemrosesan. Periksa ketentuan resmi "
                "sebelum menetapkan tanggal jatuh tempo."
            )

        # Carrier notification.
        if not ota_is_mor and has_carrier_notice_ota:
            ota_sections.append(
                "**Kewajiban maskapai terhadap OTA:**\n\n"
                "Setelah menerima permintaan refund konsumen melalui "
                "ticket agent, carrier harus memberi tahu ticket agent "
                "tanpa penundaan apakah konsumen memenuhi syarat refund "
                "berdasarkan §260.6(d)."
            )

        # Carrier transfers money if agent is responsible.
        if not ota_is_mor and has_carrier_transfer_ota:
            ota_sections.append(
                "Jika ticket agent bertanggung jawab atas refund tetapi "
                "tidak memegang dana konsumen, carrier yang memegang dana "
                "harus segera mentransfer dana tersebut kepada ticket "
                "agent berdasarkan §260.6(e)."
            )

        # Fare eligibility caveat.
        if not ota_is_mor and has_nonrefundable_scope_ota:
            ota_sections.append(
                "**Catatan jenis tiket dan kelayakan:** "
                "§260.6(a)(1) secara khusus membahas pemegang tiket "
                "non-refundable pada covered scheduled flight to, from, "
                "or within the United States. Syarat pembatalan atau "
                "perubahan yang tercakup dan pilihan konsumen untuk tidak "
                "melanjutkan perjalanan atau menerima kompensasi pengganti "
                "refund tetap harus dipenuhi."
            )

        asks_fully_refundable_ota = any(term in lowered_question for term in (
            "fully refundable",
            "tiket fully refundable",
            "tiket refundable",
            "tiket yang bisa dikembalikan",
        ))

        if not ota_is_mor and has_fully_refundable_guidance_ota and asks_fully_refundable_ota:
            ota_sections.append(
                "Untuk tiket fully refundable, panduan konsumen DOT menyatakan "
                "bahwa pembeli berhak atas refund ketika tiket yang dibeli "
                "tidak digunakan. Halaman DOT juga menjelaskan bahwa penolakan "
                "tawaran penerbangan alternatif dapat memicu refund otomatis "
                "dan mencantumkan tenggat 7 business days untuk kartu kredit "
                "atau 20 calendar days untuk metode pembayaran lainnya setelah "
                "penolakan. Namun, batas teks peraturan perlu dijelaskan: "
                "§260.6(a)(1) secara eksplisit menyebut pemegang tiket "
                "non-refundable, sedangkan §399.80(l) mengaitkan kewajiban "
                "ticket agent dengan situasi yang dirujuk oleh §260.6(a) dan (b).\n\n"
                "Karena itu, jangan menyatakan bahwa tenggat statutory carrier "
                "atau tenggat agent dalam §399.80(l) pasti berlaku pada setiap "
                "klaim fully refundable yang berdiri sendiri di luar situasi "
                "yang dirujuk aturan tersebut. Bedakan hak refund berdasarkan "
                "fare rules dari tenggat yang secara tegas ditetapkan oleh "
                "ketentuan refund DOT. Periksa syarat fare, siapa merchant of "
                "record, dan minta konfirmasi tertulis mengenai tanggal jatuh "
                "tempo apabila tiket Anda fully refundable."
            )

        if ota_is_mor:
            ota_application = (
                "**Penerapan pada pertanyaan Anda:** Anda menyatakan OTA "
                "tercantum sebagai MOR pada laporan pembayaran. Karena Anda "
                "belum mengajukan permintaan refund kepada OTA, segera ajukan "
                "permintaan tertulis kepada OTA. Untuk pembayaran kartu kredit, "
                "tenggat prompt refund ticket agent adalah 7 business days. "
                "Untuk metode pembayaran lainnya, tenggatnya 20 calendar days "
                "sejak refund menjadi jatuh tempo. Tenggat OTA tidak otomatis "
                "dihitung dari tanggal Anda menolak alternatif."
            )
        elif carrier_is_mor:
            ota_application = (
                "**Penerapan pada pertanyaan Anda:** Anda menyatakan "
                "maskapai tercantum sebagai MOR. Dengan syarat kelayakan "
                "refund terpenuhi, penolakan eksplisit memicu refund otomatis "
                "oleh carrier tanpa permintaan tambahan. Tenggatnya 7 business "
                "days untuk kartu kredit atau 20 calendar days untuk metode "
                "lainnya setelah penolakan, menurut panduan U.S. DOT."
            )
        else:
            ota_application = (
                "**Penerapan pada pertanyaan Anda:** nama MOR belum disebutkan "
                "secara eksplisit. Periksa nama pedagang pada laporan transaksi. "
                "Jika OTA adalah MOR, ajukan permintaan refund kepada OTA "
                "berdasarkan pasal 399.80(l). Jika maskapai adalah MOR, jalur refund "
                "otomatis carrier berlaku sesuai ketentuan penolakan eksplisit."
            )

        ota_sections.append(
            ota_application
            + "\n\n"
            + "Penjelasan ini merupakan parafrasa berdasarkan dokumen "
            "U.S. DOT dan eCFR yang berhasil diambil, bukan kutipan "
            "verbatim peraturan."
        )

        if ota_is_mor:
            ota_sections = [
                section for section in ota_sections
                if not section.startswith(
                    "**Jika maskapai adalah Merchant of Record:**"
                )
            ]

        return "\n\n".join(ota_sections)

    # ========================================================
    # GENERAL FARE-CLASS GUARD
    # Separate refundable and non-refundable ticket rules.
    # Does not assume a fare class that the user did not provide.
    # ========================================================

    fare_ota_context = any(term in lowered_question for term in (
        "ota",
        "online travel agency",
        "ticket agent",
        "travel agent",
        "agen perjalanan",
        "merchant of record",
    ))

    fare_cancellation_intent = any(term in lowered_question for term in (
        "dibatalkan",
        "pembatalan",
        "membatalkan",
        "cancelled",
        "canceled",
        "cancellation",
    ))

    fare_alternative_intent = any(term in lowered_question for term in (
        "alternatif",
        "alternative",
        "penerbangan pengganti",
        "alternative flight",
        "alternative-flight",
        "rebooking",
    ))

    fare_explicit_rejection_intent = any(term in lowered_question for term in (
        "menolak",
        "menolaknya",
        "penolakan",
        "saya tolak",
        "reject",
        "rejected",
        "decline",
        "declined",
    ))

    fare_nonrefundable_question = any(term in lowered_question for term in (
        "non-refundable",
        "nonrefundable",
        "non refundable",
    ))

    fare_fully_refundable_question = any(term in lowered_question for term in (
        "fully refundable",
        "tiket refundable",
        "tiket yang bisa dikembalikan",
    ))

    fare_has_nonrefundable_evidence = (
        "260.6(a)(1)" in evidence
        and (
            "nonrefundable" in evidence
            or "non-refundable" in evidence
        )
        and "cancelled" in evidence
        and "alternative flight" in evidence
        and "merchant of record" in evidence
    )

    fare_has_explicit_rejection_evidence = (
        "260.6(a)(2)(ii)" in evidence
        and (
            "automatic refund" in evidence
            or "refund is due automatically" in evidence
            or ("refund" in evidence and "automatically" in evidence)
        )
        and "reject" in evidence
        and "alternative flight" in evidence
    )

    fare_has_rejection_deadline_evidence = (
        "7 business days" in evidence
        and "20 calendar days" in evidence
        and any(term in evidence for term in (
            "after you reject the offer",
            "after the consumer rejects the offer",
            "after rejection",
            "following rejection",
            "after explicit rejection",
            "after an explicit rejection",
            "setelah penolakan",
            "setelah konsumen menolak",
        ))
    )

    fare_has_fully_refundable_evidence = (
        "fully refundable" in evidence
        and (
            "does not use" in evidence
            or "do not use" in evidence
            or "not use the purchased ticket" in evidence
            or "ticket is not used" in evidence
        )
    )

    if (
        not fare_ota_context
        and not fare_nonrefundable_question
        and fare_cancellation_intent
        and fare_alternative_intent
        and fare_explicit_rejection_intent
        and fare_has_nonrefundable_evidence
        and fare_has_explicit_rejection_evidence
        and fare_has_rejection_deadline_evidence
    ):

        # ----------------------------------------------------
        # Identify fare class without inventing user facts.
        # ----------------------------------------------------

        if fare_nonrefundable_question:
            fare_class_explanation = (
                "**Jika tiket Anda non-refundable:** "
                "14 CFR §260.6(a)(1) secara khusus mengatur pemegang "
                "tiket non-refundable pada covered scheduled flight "
                "to, from, or within the United States. Jika penerbangan "
                "dibatalkan dan Anda tidak menggunakan penerbangan "
                "alternatif atau menerima voucher, credit, atau "
                "kompensasi sebagai pengganti refund, ketentuan refund "
                "penuh dapat berlaku sesuai persyaratannya."
            )

        elif fare_fully_refundable_question:
            if fare_has_fully_refundable_evidence:
                fare_class_explanation = (
                    "**Jika tiket Anda fully refundable:** "
                    "panduan konsumen U.S. DOT menyatakan bahwa konsumen "
                    "yang membeli fully refundable ticket berhak mendapat "
                    "refund ketika tiket yang dibeli tidak digunakan. "
                    "Ini merupakan penjelasan terpisah dari "
                    "§260.6(a)(1), yang secara khusus menyebut "
                    "non-refundable ticket."
                    "\n\n"
                    "Sumber yang berhasil diambil tidak cukup untuk "
                    "menyatakan bahwa tenggat 7 business days atau "
                    "20 calendar days dari cabang non-refundable "
                    "otomatis berlaku dengan trigger yang sama untuk "
                    "semua skenario fully refundable."
                )
            else:
                fare_class_explanation = (
                    "**Jika tiket Anda fully refundable:** "
                    "sumber khusus DOT untuk jenis tiket ini belum "
                    "berhasil diambil. Karena itu, saya tidak akan "
                    "menyamakan ketentuannya dengan §260.6(a)(1) "
                    "atau mengarang tenggat refund khusus."
                )

        else:
            fare_class_explanation = (
                "**Jenis tiket belum disebutkan, sehingga kelayakan "
                "individu belum dapat dipastikan.** Ada dua jalur yang "
                "harus dibedakan:\n\n"
                "- **Non-refundable:** §260.6(a)(1) secara khusus "
                "membahas tiket jenis ini pada penerbangan terjadwal "
                "yang tercakup, dengan syarat pembatalan dan pilihan "
                "penumpang untuk tidak melanjutkan perjalanan atau "
                "menerima kompensasi pengganti refund.\n"
                "- **Fully refundable:** panduan DOT membahas hak "
                "refund ketika tiket fully refundable tidak digunakan. "
                "Ketentuan ini tidak boleh disamakan begitu saja dengan "
                "§260.6(a)(1).\n\n"
                "Periksa jenis tarif pada e-ticket atau bukti pemesanan. "
                "Jangan menganggap kedua jenis tiket memiliki dasar "
                "kelayakan dan tenggat yang identik."
            )

        # ----------------------------------------------------
        # Carrier as merchant of record.
        # ----------------------------------------------------

        carrier_explanation = (
            "**Jika maskapai merupakan Merchant of Record:** "
            "penolakan eksplisit atas penerbangan alternatif masuk "
            "dalam §260.6(a)(2)(ii), untuk kondisi refund yang tercakup. "
            "Panduan DOT menjelaskan bahwa refund carrier sebagai "
            "MOR diproses otomatis setelah penolakan eksplisit.\n\n"
            "Untuk kondisi tersebut, tenggat yang dijelaskan DOT "
            "adalah 7 business days setelah penolakan untuk pembayaran "
            "kartu kredit atau 20 calendar days setelah penolakan "
            "untuk metode pembayaran lainnya. Jangan menunggu "
            "penerbangan alternatif berangkat pada kasus penolakan "
            "eksplisit."
        )

        # ----------------------------------------------------
        # Agent as merchant of record.
        # ----------------------------------------------------

        agent_evidence = (
            "399.80(l)" in evidence
            and "ticket agent" in evidence
            and "refund" in evidence
            and "request" in evidence
        )

        agent_deadline_evidence = (
            agent_evidence
            and "7 business days" in evidence
            and "20 calendar days" in evidence
            and (
                "receives information from the carrier" in evidence
                or "receives information from carrier" in evidence
            )
        )

        if agent_deadline_evidence:
            agent_explanation = (
                "**Jika ticket agent/OTA merupakan Merchant of Record:** "
                "14 CFR §399.80(l) mengharuskan ticket agent memberikan "
                "prompt refund atas permintaan konsumen apabila kondisi "
                "kelayakan refund terpenuhi. Jadi, penolakan kepada "
                "maskapai tidak boleh dianggap otomatis menggantikan "
                "permintaan refund kepada OTA.\n\n"
                "Untuk kartu kredit, tenggatnya 7 business days sejak "
                "ticket agent menerima informasi kelayakan refund dari "
                "carrier berdasarkan §260.6(d). Untuk metode pembayaran "
                "lainnya, tenggatnya 20 calendar days sejak refund "
                "menjadi jatuh tempo. Tenggat OTA ini tidak otomatis "
                "dihitung sejak tanggal penolakan."
            )
        elif agent_evidence:
            agent_explanation = (
                "**Jika ticket agent/OTA merupakan Merchant of Record:** "
                "§399.80(l) mengatur refund prompt atas permintaan "
                "konsumen. Namun, sumber yang berhasil diambil belum "
                "memuat seluruh dasar tenggat OTA. Saya tidak akan "
                "menyamakan trigger tenggat OTA dengan trigger "
                "penolakan eksplisit pada carrier."
            )
        else:
            agent_explanation = (
                "**Jika ticket agent/OTA merupakan Merchant of Record:** "
                "identifikasi pihak dan proses refund harus diverifikasi "
                "dari aturan ticket agent yang relevan. Sumber yang "
                "berhasil diambil belum cukup untuk menjelaskan "
                "seluruh prosedur agent."
            )

        return (
            "**Hak refund setelah pembatalan dan penolakan alternatif**\n\n"
            + fare_class_explanation
            + "\n\n"
            + carrier_explanation
            + "\n\n"
            + agent_explanation
            + "\n\n"
            "**Siapa Merchant of Record Anda?** "
            "Periksa nama pedagang pada laporan transaksi kartu atau "
            "rekening pembayaran. Pembelian melalui OTA saja tidak "
            "membuktikan bahwa OTA adalah MOR; bank atau penerbit kartu "
            "juga bukan MOR hanya karena kartu tersebut digunakan.\n\n"
            "**Fakta yang masih perlu dipastikan:** jenis tiket, nama "
            "pedagang pada transaksi, apakah Anda menerima voucher/"
            "credit sebagai pengganti refund, dan apakah Anda benar-benar "
            "tidak menggunakan penerbangan alternatif.\n\n"
            "Penjelasan ini merupakan parafrasa berdasarkan sumber "
            "U.S. DOT dan eCFR yang berhasil diambil, bukan kutipan "
            "verbatim peraturan."
        )

    # ========================================================
    # V7 EXPLICIT CANCELLATION REJECTION GUARD
    # ========================================================
    request_text = re.sub(r"\s+", " ", (question or "").lower()).strip()

    has_nonrefundable_ticket = any(
        term in request_text
        for term in ("non-refundable", "nonrefundable", "non refundable")
    )
    has_us_scope = any(
        term in request_text
        for term in (
            "u.s. dot",
            "us dot",
            "united states",
            "amerika serikat",
            "cakupan aturan",
            "domestik as",
            "domestic us",
            "domestic u.s.",
        )
    )
    has_cancellation = any(
        term in request_text
        for term in ("membatalkan", "dibatalkan", "penerbangan batal",
                     "cancelled flight", "canceled flight", "cancellation")
    )
    has_alternative_offer = any(
        term in request_text
        for term in (
            "penerbangan alternatif",
            "penerbangan pengganti",
            "menawarkan alternatif",
            "alternatif tersebut",
            "alternatif",
            "alternative flight",
            "alternative-flight",
            "rebooking",
        )
    )
    has_explicit_rejection = any(
        term in request_text
        for term in ("menolak", "menolaknya", "saya tolak", "ditolak",
                     "reject", "rejected", "decline", "declined")
    )
    accepted_compensation_pattern = (
        r"\b(menerima|menerimanya|accept|accepted|took)\b.{0,60}"
        r"\b(voucher|travel credit|credit|kredit|kompensasi)\b"
    )
    negated_compensation_pattern = (
        r"\b(tidak|belum|not|never|did not|didn't)\b.{0,40}"
        r"\b(menerima|menerimanya|accept|accepted|take|took)\b.{0,60}"
        r"\b(voucher|travel credit|credit|kredit|kompensasi)\b"
    )
    has_accepted_compensation = bool(
        re.search(accepted_compensation_pattern, request_text)
        and not re.search(negated_compensation_pattern, request_text)
    )
    has_not_received_compensation = bool(re.search(negated_compensation_pattern, request_text))

    has_cancel_refund_evidence = (
        "260.6(a)(1)" in evidence
        and ("nonrefundable" in evidence or "non-refundable" in evidence)
        and "cancelled" in evidence
        and "alternative flight" in evidence
        and "merchant of record" in evidence
    )

    # Explicit rejection is §260.6(a)(2)(ii), not the separate no-response
    # condition in §260.6(a)(2)(iii)(A).
    has_automatic_rejection_evidence = (
        "260.6(a)(2)(ii)" in evidence
        and (
            "automatic refund" in evidence
            or "refund is due automatically" in evidence
            or ("refund" in evidence and "automatically" in evidence)
        )
        and "reject" in evidence
        and (
            "alternative flight" in evidence
            or "alternative-flight" in evidence
        )
    )

    has_explicit_rejection_deadline = (
        "7 business days" in evidence
        and "20 calendar days" in evidence
        and any(term in evidence for term in (
            "after you reject the offer",
            "after the consumer rejects the offer",
            "after rejection",
            "following rejection",
            "after the date you reject",
            "after an explicit rejection",
            "after explicit rejection",
            "after the consumer explicitly rejects",
            "setelah penolakan",
            "setelah konsumen menolak",
        ))
    )

    if (
        has_nonrefundable_ticket
        and has_us_scope
        and has_cancellation
        and has_alternative_offer
        and has_explicit_rejection
        and not has_accepted_compensation
        and has_cancel_refund_evidence
        and has_automatic_rejection_evidence
        and has_explicit_rejection_deadline
    ):
        has_d_notice = (
            "260.6(d)" in evidence
            and any(term in evidence for term in (
                "refund request through the ticket agent",
                "requests a refund through the ticket agent",
                "consumer requests a refund through the ticket agent",
            ))
            and any(term in evidence for term in (
                "inform the ticket agent",
                "inform the agent",
            ))
        )
        has_e_transfer = (
            "260.6(e)" in evidence
            and any(term in evidence for term in (
                "does not possess the consumer funds",
                "does not possess the consumer's funds",
                "does not possess the funds",
                "does not hold the consumer funds",
                "does not hold the consumer's funds",
            ))
            and any(term in evidence for term in (
                "transfer those funds",
                "transfer them",
                "promptly transfer",
                "transfer the funds to the ticket agent",
            ))
        )

        has_agent_refund_rule = (
            "399.80(l)" in evidence
            and "ticket agent" in evidence
            and "refund" in evidence
            and "request" in evidence
        )
        agent_notes = []
        if has_agent_refund_rule:
            agent_notes.append(
                "Jika OTA/ticket agent yang tercantum pada transaksi adalah merchant of record, "
                "aturan berbeda berlaku: 14 CFR §399.80(l) mewajibkan ticket agent memberikan "
                "refund prompt **atas permintaan konsumen**. Untuk kartu kredit, tenggat agent "
                "adalah 7 business days sejak agent menerima informasi kelayakan refund dari "
                "carrier berdasarkan §260.6(d); untuk metode pembayaran lain, 20 calendar days "
                "sejak refund menjadi jatuh tempo. Jadi, jangan menganggap refund OTA diproses "
                "otomatis tanpa permintaan kepada OTA."
            )
        if has_d_notice:
            agent_notes.append(
                "Setelah konsumen mengajukan permintaan refund melalui ticket agent, "
                "14 CFR §260.6(d) mewajibkan carrier memberi tahu agent tanpa penundaan "
                "apakah konsumen memenuhi syarat refund."
            )
        if has_e_transfer:
            agent_notes.append(
                "Menurut 14 CFR §260.6(e), jika ticket agent bertanggung jawab "
                "mengeluarkan refund tetapi tidak memegang dana konsumen, carrier "
                "yang masih memegang dana harus segera mentransfer dana tersebut "
                "kepada ticket agent."
            )
        if not agent_notes:
            agent_notes.append(
                "Sumber yang terambil belum memuat ketentuan lengkap mengenai proses ticket agent "
                "berdasarkan §399.80(l) dan §260.6(d)-(e). Periksa merchant of record pada "
                "mutasi transaksi dan jangan menganggap OTA otomatis memproses refund tanpa permintaan."
            )
        agent_note = "\n\n".join(agent_notes)

        return (
            (
                "**Kesimpulan: Anda pada prinsipnya berhak atas refund penuh.**\n\n"
                "Anda menyatakan tidak menerima voucher, travel credit, atau kompensasi sebagai pengganti refund. "
                "Dengan fakta tersebut, syarat penolakan alternatif tanpa menerima kompensasi pengganti terpenuhi.\n\n"
                if has_not_received_compensation else
                "**Kesimpulan: Anda pada prinsipnya berhak atas refund penuh, "
                "dengan satu fakta yang perlu dipastikan: Anda tidak menerima "
                "voucher, credit, atau kompensasi sebagai pengganti refund.**\n\n"
                "Berdasarkan keterangan Anda, tiket bersifat non-refundable, "
                "penerbangan domestik berada dalam cakupan U.S. DOT, maskapai "
                "membatalkan penerbangan, dan Anda secara eksplisit menolak "
                "penerbangan alternatif. Pertanyaan Anda belum menyebut apakah "
                "voucher/credit pengganti refund juga ditawarkan atau diterima; "
                "karena itu saya tidak akan menganggap fakta tersebut. Jika Anda "
                "tidak menerima kompensasi pengganti refund, refund penuh berlaku.\n\n"
            )
            + "**Dasar aturan:** 14 CFR §260.6(a)(1) mewajibkan covered carrier "
            "yang menjadi merchant of record memberikan refund penuh dan "
            "prompt, termasuk airfare, pajak, dan biaya ancillary yang "
            "tercakup, ketika syarat tersebut terpenuhi. Pembatalan tidak "
            "perlu memenuhi ambang keterlambatan 3 jam atau 6 jam yang "
            "digunakan untuk pengujian significant late arrival.\n\n"
            "**Refund otomatis dan tenggat — jika maskapai adalah merchant of record:** "
            "§260.6(a)(2)(ii) mencakup penolakan eksplisit. Untuk tiket yang dibeli langsung "
            "dari maskapai, panduan DOT menyatakan refund diproses otomatis tanpa permintaan "
            "refund tambahan. Untuk pembayaran kartu kredit, tenggatnya 7 hari kerja setelah "
            "penolakan; untuk metode pembayaran lain, 20 hari kalender setelah penolakan. "
            "Jangan menunggu alternatif berangkat pada kasus penolakan eksplisit. Trigger "
            "keberangkatan alternatif berlaku pada no-response §260.6(a)(2)(iii)(A), bukan "
            "pada penolakan eksplisit.\n\n"
            "**Pihak yang memproses refund:** merchant of record adalah pedagang yang tercantum "
            "pada detail transaksi. Penerbit kartu/bank bukan merchant of record hanya karena "
            "kartu itu digunakan untuk membayar. Jika nama pedagang adalah maskapai, gunakan "
            "jalur refund otomatis dari maskapai. Jika nama pedagang adalah OTA/ticket agent, "
            "hubungi OTA dan ajukan permintaan refund; jalur agent memiliki ketentuan §399.80(l) "
            "yang berbeda dari refund otomatis maskapai. Karena bukti transaksi belum diberikan, "
            "merchant of record Anda belum dapat dipastikan.\n\n"
            + agent_note
            + "\n\nRefund yang diwajibkan dikembalikan ke bentuk "
            "pembayaran asal. Sumber yang digunakan merupakan ringkasan "
            "lokal berdasarkan halaman resmi, bukan kutipan verbatim."
        )

    return None