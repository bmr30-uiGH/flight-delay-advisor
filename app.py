from __future__ import annotations

from datetime import date, time
import importlib
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Set page configuration before rendering Streamlit elements.
st.set_page_config(page_title="Flight Delay Advisor", page_icon="✈️", layout="wide")

# Load Streamlit Community Cloud Secrets into the environment BEFORE importing
# the agent. The agent reads GROQ_API_KEY and model settings from os.environ.
# In local Docker, existing environment variables take precedence.
for _secret_name in ("GROQ_API_KEY", "LLM_BASE_URL", "OPENAI_MODEL", "MODEL_SOURCE"):
    try:
        _secret_value = st.secrets.get(_secret_name)
    except Exception:
        # Local development may not have a .streamlit/secrets.toml file.
        _secret_value = None
    if _secret_value and not os.getenv(_secret_name):
        os.environ[_secret_name] = str(_secret_value)

# Explicitly put this repository root first on the import path. Streamlit Cloud
# may launch the script with a working directory different from the repo root.
PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)
importlib.invalidate_caches()

# Fail with a clear message if the repository's local src package is missing.
LOCAL_SRC_INIT = PROJECT_ROOT / "src" / "__init__.py"
if not LOCAL_SRC_INIT.is_file():
    raise FileNotFoundError(
        f"Package 'src' tidak ditemukan. File yang diperlukan: {LOCAL_SRC_INIT}"
    )

from src.agent import answer_user
from src.config import FLIGHTS_PATH
from src.reference_data import (
    AIRLINE_NAMES,
    airport_label,
    build_route_reference,
    carrier_label,
    load_airport_directory,
    load_carrier_directory,
)

st.title("✈️ Flight Delay + Passenger Rights Advisor")
st.caption(
    "Estimasi risiko keberangkatan terlambat ≥15 menit dan pencarian informasi hak penumpang berdasarkan sumber."
)


def _render_sources(sources: list[dict], *, expanded: bool = False) -> None:
    """Render retrieval-backed sources once per source URL, with their supporting sections."""
    if not sources:
        return
    with st.expander("Sumber yang digunakan", expanded=expanded):
        for index, source in enumerate(sources, start=1):
            title = str(source.get("title", "Sumber") or "Sumber")
            url = str(source.get("url", "") or "").strip()
            if url:
                st.markdown(f"{index}. [{title}]({url})")
            else:
                st.markdown(f"{index}. **{title}**")
            section = str(source.get("section", "") or "").strip()
            if section:
                st.caption(f"Bagian dokumen: {section}")
            excerpt = str(source.get("excerpt", "") or "").strip()
            if excerpt:
                st.markdown(excerpt)


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def _load_airports_directory() -> dict:
    return load_airport_directory()


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def _load_carrier_directory() -> dict:
    return load_carrier_directory()


@st.cache_data(show_spinner="Memuat daftar maskapai dan rute BTS...", ttl=60 * 60)
def _load_route_reference(path_string: str, modified_ns: int) -> dict:
    # modified_ns participates in the cache key, so replacing the parquet invalidates lookups.
    del modified_ns
    path = Path(path_string)
    frame = pd.read_parquet(path, columns=["OP_CARRIER", "ORIGIN", "DEST", "DISTANCE"])
    return build_route_reference(frame)


flight_reference = None
reference_error = None
if FLIGHTS_PATH.exists():
    try:
        flight_reference = _load_route_reference(str(FLIGHTS_PATH), FLIGHTS_PATH.stat().st_mtime_ns)
    except Exception as exc:
        reference_error = str(exc)
else:
    reference_error = f"Dataset BTS belum ditemukan di {FLIGHTS_PATH}. Jalankan bootstrap terlebih dahulu."

airport_directory = _load_airports_directory()
carrier_directory = _load_carrier_directory()

with st.sidebar:
    st.header("Detail penerbangan")
    st.caption(
        "Ketik nama maskapai, kota, nama bandara, atau kode IATA pada pilihan di bawah. "
        "Daftar rute berasal dari penerbangan historis BTS yang sudah diproses."
    )
    st.caption("Perubahan pada formulir baru dipakai chatbot setelah tombol Simpan detail penerbangan ditekan.")

    if flight_reference:
        carriers = flight_reference["carriers"]
        origins = flight_reference["origins"]
        destinations_by_origin = flight_reference["destinations_by_origin"]
        distance_by_route = flight_reference["distance_by_route"]

        default_carrier = "AA" if "AA" in carriers else carriers[0]
        carrier = st.selectbox(
            "Maskapai / carrier",
            options=carriers,
            index=carriers.index(default_carrier),
            format_func=lambda code: carrier_label(code, carrier_directory),
            help="Klik pilihan lalu mulai mengetik nama maskapai atau kode, misalnya American Airlines, Delta, AA, atau DL.",
            key="flight_carrier_selector",
        )

        default_origin = "JFK" if "JFK" in origins else origins[0]
        origin = st.selectbox(
            "Bandara asal",
            options=origins,
            index=origins.index(default_origin),
            format_func=lambda code: airport_label(code, airport_directory),
            help="Ketik nama kota/bandara atau kode, misalnya New York atau JFK.",
            key="flight_origin_selector",
        )

        destinations = destinations_by_origin.get(origin, [])
        if not destinations:
            st.error("Tidak ditemukan rute historis dari bandara asal ini.")
            destination = None
            distance = None
        else:
            preferred_dest = "LAX" if "LAX" in destinations else destinations[0]
            destination = st.selectbox(
                "Bandara tujuan",
                options=destinations,
                index=destinations.index(preferred_dest),
                format_func=lambda code: airport_label(code, airport_directory),
                help="Tujuan difilter ke rute yang memiliki catatan di dataset BTS agar jaraknya bisa diisi otomatis.",
                key=f"flight_destination_selector_{origin}",
            )
            distance = distance_by_route.get((origin, destination))

        flight_date = st.date_input(
            "Tanggal keberangkatan",
            value=date.today(),
            help="Tanggal digunakan untuk memperoleh bulan dan hari dalam minggu. Prediksi tetap berdasarkan pola historis, bukan data real-time.",
            key="flight_date_selector",
        )
        departure_time = st.time_input(
            "Jam keberangkatan terjadwal (waktu lokal bandara asal)",
            value=time(8, 44),
            help="Masukkan waktu lokal di bandara asal, bukan otomatis WIB berdasarkan lokasi perangkat. Aplikasi mengubahnya ke format jam BTS, misalnya 08:44 menjadi 0844.",
            key="flight_time_selector",
        )

        st.text_input(
            "Jarak rute (miles) — otomatis dari BTS",
            value=f"{distance:,} miles" if distance is not None else "Rute tidak tersedia",
            disabled=True,
            help="Jarak dihitung dari nilai historis BTS untuk pasangan bandara asal dan tujuan yang dipilih.",
        )

        can_save = destination is not None and distance is not None
        submitted = st.button(
            "Simpan detail penerbangan",
            use_container_width=True,
            type="primary",
            disabled=not can_save,
        )
        if submitted and can_save:
            bts_day_of_week = ((flight_date.weekday() + 1) % 7) + 1  # Sunday=1 ... Saturday=7
            st.session_state.flight_inputs = {
                "OP_CARRIER": carrier,
                "ORIGIN": origin,
                "DEST": destination,
                "CRS_DEP_TIME": departure_time.hour * 100 + departure_time.minute,
                "MONTH": flight_date.month,
                "DAY_OF_WEEK": bts_day_of_week,
                "DISTANCE": float(distance),
            }
            st.session_state.flight_summary = {
                "carrier": carrier_label(carrier),
                "origin": airport_label(origin, airport_directory),
                "origin_code": origin,
                "destination": airport_label(destination, airport_directory),
                "destination_code": destination,
                "flight_date": flight_date.isoformat(),
                "departure_time": departure_time.strftime("%H:%M"),
                "distance": int(distance),
            }
            st.success("Detail penerbangan tersimpan untuk sesi ini.")

        if st.session_state.get("flight_summary"):
            saved = st.session_state.flight_summary
            with st.expander("Detail yang tersimpan", expanded=False):
                st.write(f"**Maskapai:** {saved['carrier']}")
                st.write(f"**Asal:** {saved['origin']}")
                st.write(f"**Tujuan:** {saved['destination']}")
                st.write(f"**Tanggal:** {saved['flight_date']}")
                st.write(f"**Jam terjadwal:** {saved['departure_time']} (waktu lokal di {saved.get('origin_code', 'bandara asal')})")
                st.write(f"**Jarak:** {saved['distance']:,} miles")
    else:
        st.error("Daftar rute belum dapat dimuat. " + (reference_error or "Periksa dataset hasil bootstrap."))
        st.info("Pastikan data/processed/flights.parquet sudah tersedia dan berisi data BTS.")

    st.divider()
    if os.getenv("GROQ_API_KEY", "").strip():
        st.success("LLM agent: API key Groq terdeteksi")
    elif os.getenv("OPENAI_API_KEY", "").strip():
        st.success("LLM agent: API key terdeteksi")
    else:
        st.warning("LLM agent belum aktif: tambahkan GROQ_API_KEY pada Streamlit Secrets atau environment variable aplikasi.")
    st.caption("Model menggunakan pola data historis BTS. Ini bukan pelacakan status penerbangan real-time.")

if "flight_inputs" not in st.session_state:
    st.session_state.flight_inputs = None
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.expander("Contoh pertanyaan", expanded=not st.session_state.messages):
    st.markdown(
        "- `Perkirakan risiko delay untuk penerbangan saya.`\n"
        "- `Apa hak penumpang jika penerbangan terlambat?`\n"
        "- `Perkirakan risiko delay, lalu jelaskan informasi hak penumpang yang relevan.`"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        _render_sources(message.get("sources", []))
        if message.get("tools_used"):
            st.caption("Tool yang dijalankan: " + ", ".join(message["tools_used"]))
        if message.get("mode"):
            st.caption("Mode: " + message["mode"])

prompt = st.chat_input("Tanyakan risiko delay atau hak penumpang...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Menganalisis pertanyaan dan memilih tool yang diperlukan..."):
            answer, sources, tools_used, mode = answer_user(
                prompt,
                st.session_state.flight_inputs,
                st.session_state.get("flight_summary"),
            )
        st.markdown(answer)
        _render_sources(sources, expanded=True)
        if tools_used:
            st.caption("Tool yang dijalankan: " + ", ".join(tools_used))
        st.caption("Mode: " + mode)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources, "tools_used": tools_used, "mode": mode}
    )
