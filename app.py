from __future__ import annotations

from datetime import date, time
import importlib
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# ─── Page Configuration ───
st.set_page_config(
    page_title="Flight Delay Advisor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS Injection ───
CUSTOM_CSS = """
<style>
/* ============================================================================
   DESIGN TOKENS — Flight Delay Advisor (Vibrant & Alive Palette)
   No Pure White (#FFFFFF)
   ============================================================================ */
:root {
  /* Core Palette */
  --color-parchment: #F2EDEB;
  --color-thistle: #D8C0D4;
  --color-dusty-lavender: #925E78;
  --color-lilac: #BD93BD;
  --color-dark-purple: #2D1B24;

  /* Vibrant Accents */
  --color-pink: #F05365;
  --color-tangerine: #F58848;
  --color-amber-flame: #FABC2A;
  --color-parchment-light: #FAF7F5;
  --color-thistle-soft: #EDE4EA;
  --color-amber-soft: #FDF3D7;

  /* Mappings */
  --color-bg-gradient: linear-gradient(135deg, var(--color-parchment) 0%, var(--color-thistle) 100%);
  --color-text-primary: var(--color-dark-purple);
  --color-text-secondary: #4A3A42;
  --color-text-muted: #7A6670;
  --color-text-on-dark: var(--color-parchment);

  /* Spacing Scale */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;

  /* Border Radius */
  --radius-sm: 8px;
  --radius-md: 12px;
  --radius-lg: 20px;
  --radius-full: 9999px;

  /* Shadows */
  --shadow-sm: 0 2px 4px rgba(45, 27, 36, 0.05);
  --shadow-md: 0 4px 12px rgba(45, 27, 36, 0.08);
  --shadow-lg: 0 12px 24px rgba(45, 27, 36, 0.12);
}

/* ============================================================================
   GLOBAL RESETS & BASE
   ============================================================================ */
*, *::before, *::after {
  box-sizing: border-box;
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: 'Inter', sans-serif !important;
  background: var(--color-bg-gradient) fixed !important;
  color: var(--color-text-primary) !important;
}

[data-testid="stHeader"] {
  background: transparent !important;
}

/* Typography */
h1, .stTitle { font-size: 2.5rem !important; font-weight: 800 !important; color: var(--color-dusty-lavender) !important; letter-spacing: -0.03em !important; }
h2 { font-size: 1.75rem !important; font-weight: 700 !important; color: var(--color-dusty-lavender) !important; }
h3 { font-size: 1.25rem !important; font-weight: 600 !important; color: var(--color-dusty-lavender) !important; }

/* ============================================================================
   SIDEBAR STYLING
   ============================================================================ */
[data-testid="stSidebar"] {
  background-color: var(--color-dusty-lavender) !important;
  color: var(--color-parchment) !important;
  border-right: 1px solid var(--color-lilac) !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stCaption {
  color: var(--color-parchment) !important;
}

[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stDateInput input,
[data-testid="stSidebar"] .stTimeInput input,
[data-testid="stSidebar"] .stNumberInput input {
  background-color: var(--color-thistle) !important;
  color: var(--color-dark-purple) !important;
  border: 1px solid var(--color-lilac) !important;
  border-radius: var(--radius-md) !important;
}

[data-testid="stSidebar"] hr {
  border-color: rgba(242, 237, 235, 0.2) !important;
}

/* ============================================================================
   BUTTONS
   ============================================================================ */
.stButton > button {
  border-radius: var(--radius-md) !important;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
  border: none !important;
  font-weight: 700 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
  box-shadow: var(--shadow-sm) !important;
}

/* Primary Button */
.stButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--color-dusty-lavender) 0%, var(--color-lilac) 100%) !important;
  color: var(--color-parchment) !important;
}

.stButton > button[kind="primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: var(--shadow-md) !important;
  filter: brightness(1.1) !important;
}

/* Quick Action Buttons (2x2 Grid) */
[data-testid="column"]:has(button[key^="prompt_chip_"]) .stButton > button {
    border-radius: var(--radius-lg) !important;
    padding: var(--space-4) !important;
    min-height: 80px !important;
    font-size: 0.9rem !important;
}

button[key="prompt_chip_0"] { background-color: var(--color-lilac) !important; color: var(--color-dark-purple) !important; }
button[key="prompt_chip_1"] { background-color: var(--color-tangerine) !important; color: white !important; }
button[key="prompt_chip_2"] { background-color: var(--color-amber-flame) !important; color: var(--color-dark-purple) !important; }
button[key="prompt_chip_3"] { background-color: var(--color-dusty-lavender) !important; color: white !important; }

/* ============================================================================
   CARDS
   ============================================================================ */
.digital-flight-route, .rights-checklist-card, .app-card {
  background-color: var(--color-parchment-light) !important;
  border: 2px solid var(--color-lilac) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: var(--shadow-md) !important;
  overflow: hidden !important;
}

.digital-flight-route-header, .chat-header-banner {
  background: linear-gradient(90deg, var(--color-dusty-lavender) 0%, var(--color-lilac) 100%) !important;
  color: var(--color-parchment) !important;
  padding: var(--space-4) var(--space-5) !important;
}

/* Historical Metrics 3-Color Cards */
.historical-metrics-grid {
    display: grid !important;
    grid-template-columns: repeat(3, 1fr) !important;
    gap: var(--space-3) !important;
}

.historical-metric-item {
    border-radius: var(--radius-md) !important;
    padding: var(--space-4) !important;
    text-align: center !important;
    box-shadow: var(--shadow-sm) !important;
    color: white !important;
}

.historical-metric-item:nth-child(1) { background-color: var(--color-pink) !important; }
.historical-metric-item:nth-child(2) { background-color: var(--color-tangerine) !important; }
.historical-metric-item:nth-child(3) { background-color: var(--color-amber-flame) !important; color: var(--color-dark-purple) !important; }

.historical-metric-value {
    font-size: 1.5rem !important;
    font-weight: 800 !important;
    margin-bottom: 4px !important;
}

.historical-metric-label {
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    opacity: 0.9 !important;
}

/* Risk Badges */
.digital-flight-route-badge {
    padding: 6px 12px !important;
    border-radius: var(--radius-full) !important;
    font-weight: 800 !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
}

.digital-flight-route-badge.low { background-color: #4CAF50 !important; color: white !important; }
.digital-flight-route-badge.medium { background-color: var(--color-tangerine) !important; color: white !important; }
.digital-flight-route-badge.high { background-color: var(--color-pink) !important; color: white !important; }

/* ============================================================================
   CHAT INTERFACE
   ============================================================================ */
.chat-container {
    background-color: var(--color-thistle-soft) !important;
    border: 2px solid var(--color-lilac) !important;
    border-top: none !important;
    border-radius: 0 0 var(--radius-lg) var(--radius-lg) !important;
    padding: var(--space-4) !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
    background-color: var(--color-dusty-lavender) !important;
    color: var(--color-parchment) !important;
    border-radius: 20px 20px 4px 20px !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
    background-color: var(--color-amber-soft) !important;
    border: 2px solid var(--color-tangerine) !important;
    color: var(--color-dark-purple) !important;
    border-radius: 20px 20px 20px 4px !important;
}

[data-testid="stChatInput"] textarea {
    background-color: var(--color-parchment-light) !important;
    border: 2px solid var(--color-lilac) !important;
    border-radius: var(--radius-md) !important;
    color: var(--color-dark-purple) !important;
}

[data-testid="stChatInput"] button {
    background-color: var(--color-dusty-lavender) !important;
    color: white !important;
}

/* Passenger Rights Checklist */
.rights-checklist-icon.teal { background: #4DB6AC !important; color: white !important; }
.rights-checklist-icon.amber { background: var(--color-amber-flame) !important; color: var(--color-dark-purple) !important; }
.rights-checklist-icon.blue { background: var(--color-lilac) !important; color: white !important; }
.rights-checklist-icon.green { background: var(--color-pink) !important; color: white !important; }

.rights-checklist-item {
    background-color: var(--color-thistle-soft) !important;
    border-radius: var(--radius-md) !important;
    margin-bottom: 8px !important;
}

.rights-checklist-badge {
    background-color: var(--color-lilac) !important;
    color: white !important;
}
</style>

/* ============================================================================
   EXPANDERS (Updated for new palette)
   ============================================================================ */
[data-testid="stExpander"] {
  background-color: var(--color-parchment-light) !important;
  border: 2px solid var(--color-lilac) !important;
  border-radius: var(--radius-lg) !important;
  overflow: hidden !important;
}

[data-testid="stExpander"] summary {
  font-size: 1rem !important;
  font-weight: 600 !important;
  color: var(--color-dark-purple) !important;
  background-color: var(--color-thistle-soft) !important;
  padding: var(--space-4) !important;
  cursor: pointer !important;
  list-style: none !important;
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  transition: background-color 0.2s ease !important;
}

[data-testid="stExpander"] summary:hover {
  background-color: var(--color-thistle) !important;
}

[data-testid="stExpander"] summary:focus-visible {
  outline: 2px solid var(--color-dusty-lavender) !important;
  outline-offset: -2px !important;
}

[data-testid="stExpander"] summary::-webkit-details-marker { display: none !important; }
[data-testid="stExpander"] summary::before {
  content: '▶' !important;
  font-size: 0.75rem !important;
  color: var(--color-text-muted) !important;
  transition: transform 0.2s ease !important;
  margin-right: var(--space-2) !important;
}

[data-testid="stExpander"] details[open] summary::before {
  transform: rotate(90deg) !important;
}

[data-testid="stExpander"] > div:last-child {
  padding: 0 var(--space-4) var(--space-4) !important;
  border-top: 1px solid var(--color-lilac) !important;
}

/* ============================================================================
   ALERTS / NOTIFICATIONS (Updated for new palette)
   ============================================================================ */
.stAlert {
  border-radius: var(--radius-md) !important;
  border: none !important;
  padding: var(--space-4) var(--space-5) !important;
  font-size: var(--text-base) !important;
  line-height: 1.5 !important;
}

.stAlert[data-baseweb="notification"][kind="info"] {
  background-color: var(--color-thistle-soft) !important;
  color: var(--color-dusty-lavender) !important;
  border: 2px solid var(--color-lilac) !important;
  border-left: 4px solid var(--color-dusty-lavender) !important;
}

.stAlert[data-baseweb="notification"][kind="success"] {
  background-color: #E8F5E9 !important;
  color: #2E7D32 !important;
  border: 2px solid #C8E6C9 !important;
  border-left: 4px solid #4CAF50 !important;
}

.stAlert[data-baseweb="notification"][kind="warning"] {
  background-color: var(--color-amber-soft) !important;
  color: var(--color-dark-purple) !important;
  border: 2px solid var(--color-amber-flame) !important;
  border-left: 4px solid var(--color-tangerine) !important;
}

.stAlert[data-baseweb="notification"][kind="error"] {
  background-color: #FFEBEE !important;
  color: #C62828 !important;
  border: 2px solid #FFCDD2 !important;
  border-left: 4px solid var(--color-pink) !important;
}

/* ============================================================================
   SPINNER / LOADING (Updated for new palette)
   ============================================================================ */
.stSpinner > div {
  border-color: var(--color-dusty-lavender) transparent var(--color-dusty-lavender) transparent !important;
}

.stSpinner [data-testid="stStatusWidget"] {
  background-color: var(--color-parchment-light) !important;
  border: 2px solid var(--color-lilac) !important;
  border-radius: var(--radius-lg) !important;
  padding: var(--space-4) !important;
  box-shadow: var(--shadow-lg) !important;
}

/* ============================================================================
   TABS (Updated for new palette)
   ============================================================================ */
[data-testid="stTabs"] {
  border-bottom: 2px solid var(--color-lilac) !important;
  margin-bottom: var(--space-4) !important;
}

[data-testid="stTabs"] [role="tablist"] {
  gap: var(--space-1) !important;
}

[data-testid="stTabs"] [role="tab"] {
  background-color: transparent !important;
  border: none !important;
  border-bottom: 3px solid transparent !important;
  color: var(--color-text-muted) !important;
  font-size: 1rem !important;
  font-weight: 600 !important;
  padding: var(--space-3) var(--space-4) !important;
  transition: all 0.2s ease !important;
  min-height: 44px !important;
}

[data-testid="stTabs"] [role="tab"]:hover {
  color: var(--color-dusty-lavender) !important;
  background-color: var(--color-thistle-soft) !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--color-dusty-lavender) !important;
  border-bottom-color: var(--color-dusty-lavender) !important;
  background-color: transparent !important;
}

[data-testid="stTabs"] [role="tab"]:focus-visible {
  outline: 2px solid var(--color-dusty-lavender) !important;
  outline-offset: -2px !important;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
}

[data-testid="stTabs"] [role="tabpanel"] {
  padding: var(--space-4) 0 !important;
}

/* ============================================================================
   DATA TABLE / DATAFRAME (Updated for new palette)
   ============================================================================ */
[data-testid="stDataFrame"] {
  border: 2px solid var(--color-lilac) !important;
  border-radius: var(--radius-lg) !important;
  overflow: hidden !important;
  background-color: var(--color-parchment-light) !important;
}

[data-testid="stDataFrame"] table {
  font-size: 0.875rem !important;
}

[data-testid="stDataFrame"] th {
  background-color: var(--color-thistle-soft) !important;
  color: var(--color-dark-purple) !important;
  font-weight: 700 !important;
  padding: var(--space-3) var(--space-4) !important;
  border-bottom: 2px solid var(--color-lilac) !important;
  white-space: nowrap !important;
}

[data-testid="stDataFrame"] td {
  padding: var(--space-3) var(--space-4) !important;
  border-bottom: 1px solid var(--color-lilac) !important;
  color: var(--color-text-secondary) !important;
}

[data-testid="stDataFrame"] tr:last-child td {
  border-bottom: none !important;
}

[data-testid="stDataFrame"] tr:hover td {
  background-color: var(--color-thistle-soft) !important;
}

/* ============================================================================
   PROGRESS BAR (Updated for new palette)
   ============================================================================ */
.stProgress > div > div {
  background: linear-gradient(90deg, var(--color-dusty-lavender) 0%, var(--color-lilac) 100%) !important;
  border-radius: var(--radius-full) !important;
}

.stProgress > div {
  background-color: var(--color-thistle) !important;
  border-radius: var(--radius-full) !important;
  height: 8px !important;
}

/* ============================================================================
   HORIZONTAL RULE / DIVIDER (Updated for new palette)
   ============================================================================ */
hr, .stDivider {
  border-color: var(--color-lilac) !important;
  margin: var(--space-5) 0 !important;
}

/* ============================================================================
   TOOLTIPS (Updated for new palette)
   ============================================================================ */
[data-testid="stTooltip"] {
  background-color: var(--color-parchment-light) !important;
  border: 2px solid var(--color-lilac) !important;
  border-radius: var(--radius-md) !important;
  box-shadow: var(--shadow-lg) !important;
  padding: var(--space-3) var(--space-4) !important;
  font-size: 0.875rem !important;
  color: var(--color-dark-purple) !important;
}

/* ============================================================================
   MAIN CONTAINER LAYOUT
   ============================================================================ */
[data-testid="stAppViewContainer"] > .main {
  padding: var(--space-5) var(--space-6) !important;
  max-width: 1400px !important;
  margin: 0 auto !important;
}

.block-container {
  padding-top: var(--space-5) !important;
  padding-bottom: var(--space-6) !important;
  max-width: 100% !important;
}

/* ============================================================================
   RESPONSIVE BREAKPOINTS
   ============================================================================ */
@media (max-width: 1024px) {
  [data-testid="stAppViewContainer"] > .main {
    padding: var(--space-4) var(--space-4) !important;
  }
  .historical-metrics-grid {
    grid-template-columns: 1fr !important;
  }
  .digital-flight-route-details {
    grid-template-columns: 1fr !important;
  }
}

@media (max-width: 768px) {
  [data-testid="stSidebar"] {
    padding: var(--space-4) var(--space-3) !important;
  }
}

@media (max-width: 480px) {
  .stButton > button,
  .stFormSubmitButton > button {
    padding: var(--space-3) var(--space-4) !important;
    font-size: 0.875rem !important;
  }
}

/* ============================================================================
   REDUCED MOTION
   ============================================================================ */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }
}

/* ============================================================================
   PRINT STYLES
   ============================================================================ */
@media print {
  [data-testid="stSidebar"],
  [data-testid="stChatInput"],
  .stButton,
  [data-testid="stStatusWidget"] {
    display: none !important;
  }
  [data-testid="stAppViewContainer"] > .main {
    padding: 0 !important;
    max-width: 100% !important;
  }
  .app-card, .metric-card, .digital-flight-route, .historical-metrics-card, .rights-checklist-card {
    break-inside: avoid !important;
    box-shadow: none !important;
    border: 1px solid #ccc !important;
  }
}

/* ============================================================================
   UTILITY CLASSES
   ============================================================================ */
.visually-hidden {
  position: absolute !important;
  width: 1px !important;
  height: 1px !important;
  padding: 0 !important;
  margin: -1px !important;
  overflow: hidden !important;
  clip: rect(0, 0, 0, 0) !important;
  white-space: nowrap !important;
  border: 0 !important;
}

.text-center { text-align: center !important; }
.text-left { text-align: left !important; }
.text-right { text-align: right !important; }

.flex { display: flex !important; }
.flex-col { flex-direction: column !important; }
.items-center { align-items: center !important; }
.justify-center { justify-content: center !important; }
.justify-between { justify-content: space-between !important; }
.gap-2 { gap: var(--space-2) !important; }
.gap-3 { gap: var(--space-3) !important; }
.gap-4 { gap: var(--space-4) !important; }

.w-full { width: 100% !important; }
.mt-4 { margin-top: var(--space-4) !important; }
.mb-4 { margin-bottom: var(--space-4) !important; }
.p-4 { padding: var(--space-4) !important; }
.rounded-lg { border-radius: var(--radius-lg) !important; }
.shadow-md { box-shadow: var(--shadow-md) !important; }
</style>
"""

# ─── Google Fonts Import ───
FONT_IMPORT = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
"""

# ─── Load Secrets ───
for _secret_name in ("GROQ_API_KEY", "LLM_BASE_URL", "GROQ_MODEL", "OPENAI_MODEL", "MODEL_SOURCE"):
    try:
        _secret_value = st.secrets.get(_secret_name)
    except Exception:
        _secret_value = None
    if _secret_value and not os.getenv(_secret_name):
        os.environ[_secret_name] = str(_secret_value)

# ─── Project Root ───
PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR in sys.path:
    sys.path.remove(PROJECT_ROOT_STR)
sys.path.insert(0, PROJECT_ROOT_STR)
importlib.invalidate_caches()

# ─── Validate Local Package ───
LOCAL_SRC_INIT = PROJECT_ROOT / "src" / "__init__.py"
if not LOCAL_SRC_INIT.is_file():
    raise FileNotFoundError(
        f"Package 'src' tidak ditemukan. File yang diperlukan: {LOCAL_SRC_INIT}"
    )

from src.agent import answer_user
from src.config import FLIGHTS_PATH
from src.predictor import predict_delay
from src.reference_data import (
    AIRLINE_NAMES,
    airport_label,
    build_route_reference,
    carrier_label,
    load_airport_directory,
    load_carrier_directory,
)

# ─── Inject Custom Styles ───
st.markdown(FONT_IMPORT, unsafe_allow_html=True)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ─── Cached Data Loaders ───
@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def _load_airports_directory() -> dict:
    return load_airport_directory()


@st.cache_data(show_spinner=False, ttl=24 * 60 * 60)
def _load_carrier_directory() -> dict:
    return load_carrier_directory()


@st.cache_data(show_spinner="Memuat daftar maskapai dan rute BTS...", ttl=60 * 60)
def _load_route_reference(path_string: str, modified_ns: int) -> dict:
    del modified_ns
    path = Path(path_string)
    frame = pd.read_parquet(path, columns=["OP_CARRIER", "ORIGIN", "DEST", "DISTANCE"])
    return build_route_reference(frame)


# ─── Load Reference Data ───
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

# ─── V3 Demo Assets Check ───
V3_ASSET_PATHS = (
    PROJECT_ROOT / "models" / "flight_delay_model_v3_candidate.joblib",
    PROJECT_ROOT / "models" / "flight_delay_model_v3_candidate.metadata.json",
    PROJECT_ROOT / "data" / "processed" / "flights_historical_features_experiment_jan_aug_v3.parquet",
)
V3_DEMO_AVAILABLE = all(path.is_file() for path in V3_ASSET_PATHS)

# ─── Session State Initialization ───
if "flight_inputs" not in st.session_state:
    st.session_state.flight_inputs = None
if "flight_summary" not in st.session_state:
    st.session_state.flight_summary = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_trigger" not in st.session_state:
    st.session_state.chat_trigger = None

# ─── Helper Functions ───
def _render_sources(sources: list[dict], *, expanded: bool = False) -> None:
    """Render retrieval-backed sources once per source URL, with their supporting sections."""
    if not sources:
        return
    with st.expander("📚 Sumber yang digunakan", expanded=expanded):
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
                st.markdown(f"<div style='font-size: 0.875rem; color: var(--color-text-muted); border-left: 2px solid var(--color-royal-blue-border); padding-left: 12px; margin-top: 8px;'>{excerpt}</div>", unsafe_allow_html=True)


def _get_risk_info(probability: float) -> tuple:
    """Get risk level, label, and CSS class based on probability."""
    if probability < 0.35:
        return "low", "Low Risk", "low"
    elif probability < 0.70:
        return "medium", "Elevated Risk", "medium"
    else:
        return "high", "High Risk", "high"


def _handle_prompt_chip_click(prompt_text: str):
    """Handle prompt chip click - add to chat and trigger agent."""
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    st.session_state.chat_trigger = prompt_text


def _get_default_flight_summary():
    """Return default flight summary for initial page load."""
    return {
        "carrier": "American Airlines",
        "origin": "John F Kennedy Intl, New York, NY",
        "origin_code": "JFK",
        "destination": "Los Angeles Intl, Los Angeles, CA",
        "destination_code": "LAX",
        "flight_date": "2026-11-21",
        "departure_time": "08:44",
        "distance": 2475,
        "prediction_mode": "production",
    }


def _get_risk_badge(probability: float):
    """Generate risk badge HTML."""
    risk_class, risk_label, _ = _get_risk_info(probability)
    probability_pct = int(probability * 100)
    return f'<span class="digital-flight-route-badge {risk_class}">{risk_label} (~{probability_pct}%)</span>'


# ═════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

# ─── Sidebar ───
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="margin: 0 0 8px 0; font-size: 1.125rem; color: var(--color-parchment);">Detail Penerbangan</h2>
        <p style="margin: 0; font-size: 0.8125rem; color: rgba(242, 237, 235, 0.8);">
            Ketik nama maskapai, kota, nama bandara, atau kode IATA. Data rute dari BTS historis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Perubahan pada formulir baru dipakai setelah tombol **Simpan detail penerbangan** ditekan.")
    st.divider()

    # ─── Prediction Mode Selector ───
    prediction_mode = st.selectbox(
        "Mesin prediksi",
        options=(["production", "v3_candidate_historical"] if V3_DEMO_AVAILABLE else ["production"]),
        format_func=lambda value: (
            "🏭 Production — MLflow Champion"
            if value == "production"
            else "🧪 V3 Candidate — Demonstrasi Historis (Agustus 2026)"
        ),
        key="prediction_mode_selector",
        help="Pilih model prediksi. Production menggunakan model champion MLflow. V3 Candidate untuk demonstrasi data historis Agustus 2026."
    )

    if prediction_mode == "v3_candidate_historical":
        st.warning(
            "**Mode Demonstrasi Historis**\n\n"
            "Ini untuk menguji kandidat V3 pada periode validasi Agustus 2026. "
            "Bukan prediksi real-time atau penerbangan mendatang. "
            "MLflow Champion tetap menjadi model Production.",
            icon="🧪"
        )

    st.divider()

    # ─── Flight Details Form ───
    if flight_reference:
        carriers = flight_reference["carriers"]
        origins = flight_reference["origins"]
        destinations_by_origin = flight_reference["destinations_by_origin"]
        distance_by_route = flight_reference["distance_by_route"]

        default_carrier = "AA" if "AA" in carriers else carriers[0]
        carrier = st.selectbox(
            "Maskapai / Carrier",
            options=carriers,
            index=carriers.index(default_carrier),
            format_func=lambda code: carrier_label(code, carrier_directory),
            help="Klik lalu ketik nama maskapai atau kode (contoh: American Airlines, Delta, AA, DL).",
            key="flight_carrier_selector",
        )

        default_origin = "JFK" if "JFK" in origins else origins[0]
        origin = st.selectbox(
            "Bandara Asal",
            options=origins,
            index=origins.index(default_origin),
            format_func=lambda code: airport_label(code, airport_directory),
            help="Ketik nama kota/bandara atau kode (contoh: New York, JFK).",
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
                "Bandara Tujuan",
                options=destinations,
                index=destinations.index(preferred_dest),
                format_func=lambda code: airport_label(code, airport_directory),
                help="Tujuan difilter ke rute dengan catatan BTS agar jarak otomatis terisi.",
                key=f"flight_destination_selector_{origin}",
            )
            distance = distance_by_route.get((origin, destination))

        # Date & Time Inputs
        if prediction_mode == "v3_candidate_historical":
            flight_date = st.date_input(
                "Tanggal Demonstrasi Historis",
                value=date(2026, 8, 31),
                min_value=date(2026, 8, 1),
                max_value=date(2026, 8, 31),
                help="Tanggal dibatasi ke periode validasi kandidat V3, 1–31 Agustus 2026.",
                key="flight_date_selector_v3_demo",
            )
        else:
            flight_date = st.date_input(
                "Tanggal Keberangkatan",
                value=date.today(),
                help="Tanggal untuk memperoleh bulan dan hari dalam minggu. Prediksi berdasarkan pola historis, bukan data real-time.",
                key="flight_date_selector",
            )

        departure_time = st.time_input(
            "Jam Keberangkatan Terjadwal (Waktu Lokal Bandara Asal)",
            value=time(8, 44),
            help="Masukkan waktu lokal di bandara asal. Contoh 08:44 → format BTS 0844.",
            key="flight_time_selector",
        )

        arrival_time = None
        elapsed_minutes = None
        if prediction_mode == "v3_candidate_historical":
            st.markdown("**Jadwal Tambahan untuk Fitur V3**")
            arrival_time = st.time_input(
                "Jam Kedatangan Terjadwal (Waktu Lokal Bandara Tujuan)",
                value=time(11, 0),
                key="flight_arrival_time_v3_demo",
            )
            elapsed_minutes = st.number_input(
                "Durasi Penerbangan Terjadwal (Menit)",
                min_value=1,
                max_value=1500,
                value=360,
                step=5,
                key="flight_elapsed_minutes_v3_demo",
            )

        # Distance Display (Read-only)
        st.text_input(
            "Jarak Rute (Mil) — Otomatis dari BTS",
            value=f"{distance:,} miles" if distance is not None else "Rute tidak tersedia",
            disabled=True,
            help="Jarak dihitung dari nilai historis BTS untuk pasangan bandara yang dipilih.",
        )

        # Save Button
        can_save = destination is not None and distance is not None
        submitted = st.button(
            "Simpan Detail Penerbangan",
            use_container_width=True,
            type="primary",
            disabled=not can_save,
            key="save_flight_details",
        )

        if submitted and can_save:
            bts_day_of_week = ((flight_date.weekday() + 1) % 7) + 1

            flight_inputs = {
                "OP_CARRIER": carrier,
                "ORIGIN": origin,
                "DEST": destination,
                "CRS_DEP_TIME": departure_time.hour * 100 + departure_time.minute,
                "MONTH": flight_date.month,
                "DAY_OF_WEEK": bts_day_of_week,
                "DISTANCE": float(distance),
                "FL_DATE": flight_date.isoformat(),
                "PREDICTION_MODE": prediction_mode,
            }

            if prediction_mode == "v3_candidate_historical":
                flight_inputs.update({
                    "CRS_ARR_TIME": arrival_time.hour * 100 + arrival_time.minute,
                    "CRS_ELAPSED_TIME": float(elapsed_minutes),
                })

            st.session_state.flight_inputs = flight_inputs

            flight_summary = {
                "carrier": carrier_label(carrier),
                "origin": airport_label(origin, airport_directory),
                "origin_code": origin,
                "destination": airport_label(destination, airport_directory),
                "destination_code": destination,
                "flight_date": flight_date.isoformat(),
                "departure_time": departure_time.strftime("%H:%M"),
                "distance": int(distance),
                "prediction_mode": prediction_mode,
            }

            if prediction_mode == "v3_candidate_historical":
                flight_summary.update({
                    "arrival_time": arrival_time.strftime("%H:%M"),
                    "elapsed_minutes": int(elapsed_minutes),
                })

            st.session_state.flight_summary = flight_summary
            st.success("✅ Detail penerbangan tersimpan untuk sesi ini.", icon="✅")
    else:
        st.error("Daftar rute belum dapat dimuat. " + (reference_error or "Periksa dataset hasil bootstrap."))
        st.info("Pastikan `data/processed/flights.parquet` sudah tersedia dan berisi data BTS.")

    st.divider()

    # ─── LLM Agent Status ───
    st.markdown("**Status LLM Agent**")
    if os.getenv("GROQ_API_KEY", "").strip():
        st.success("🟢 API Key Groq terdeteksi")
    elif os.getenv("OPENAI_API_KEY", "").strip():
        st.success("🟢 API Key OpenAI terdeteksi")
    else:
        st.warning("🟡 LLM agent belum aktif. Tambahkan `GROQ_API_KEY` pada Streamlit Secrets atau environment variable.")
    st.caption("Model menggunakan pola data historis BTS. Ini bukan pelacakan status penerbangan real-time.")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT - 2 COLUMN LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

# ─── Header ───
st.markdown("""
<div style="margin-bottom: 24px;">
    <h1 style="margin: 0 0 8px 0; display: flex; align-items: center; gap: 12px;">
        <span style="font-size: 2.5rem;">✈️</span>
        <span>Flight Delay Advisor</span>
    </h1>
    <p style="margin: 0; color: var(--color-text-muted); font-size: 1rem;">
        Estimasi risiko keterlambatan ≥15 menit & informasi hak penumpang
    </p>
</div>
""", unsafe_allow_html=True)

# ─── Determine Flight Summary (saved or default) ───
saved = st.session_state.get("flight_summary") or _get_default_flight_summary()

# ─── Get risk probability (from prediction if available) ───
probability = 0.18  # Default low risk
if st.session_state.flight_inputs:
    try:
        result = predict_delay(st.session_state.flight_inputs)
        if result.get("ok") and result.get("probability") is not None:
            probability = result["probability"]
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════════════
# LEFT COLUMN — Digital Flight Route, Historical Metrics, Rights Checklist
# ═════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # ─── Card 1: Digital Flight Route ───
    st.markdown(f"""
    <div class="digital-flight-route">
        <div class="digital-flight-route-header">
            <span class="digital-flight-route-title">Digital Flight Route</span>
            {_get_risk_badge(probability)}
        </div>
        <div class="digital-flight-route-body">
            <div class="digital-flight-route-airline">{saved['carrier']}</div>
            <div class="digital-flight-route-airports">
                <div class="digital-flight-route-origin">
                    <div class="digital-flight-route-code">{saved['origin_code']}</div>
                    <div class="digital-flight-route-city">{saved['origin'].split(',')[0] if ',' in saved['origin'] else saved['origin'][:20]}</div>
                </div>
                <div class="digital-flight-route-arrow">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M5 12h14M12 5l7 7-7 7" /></svg>
                </div>
                <div class="digital-flight-route-dest">
                    <div class="digital-flight-route-code">{saved['destination_code']}</div>
                    <div class="digital-flight-route-city">{saved['destination'].split(',')[0] if ',' in saved['destination'] else saved['destination'][:20]}</div>
                </div>
            </div>
            <div class="digital-flight-route-details">
                <div class="digital-flight-route-detail">
                    <div class="digital-flight-route-detail-label">Departure</div>
                    <div class="digital-flight-route-detail-value">{saved['departure_time']}</div>
                </div>
                <div class="digital-flight-route-detail">
                    <div class="digital-flight-route-detail-label">Distance</div>
                    <div class="digital-flight-route-detail-value">{saved['distance']:,} mi</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Card 2: Historical Route Metrics ───
    st.markdown("""
    <div class="app-card" style="margin-top: 20px;">
        <div class="app-card-header">
            <div class="app-card-title">Historical Route Metrics</div>
        </div>
        <div class="historical-metrics-grid">
            <div class="historical-metric-item">
                <div class="historical-metric-value">12 min</div>
                <div class="historical-metric-label">Average Delay</div>
            </div>
            <div class="historical-metric-item">
                <div class="historical-metric-value">88%</div>
                <div class="historical-metric-label">Route Reliability</div>
            </div>
            <div class="historical-metric-item">
                <div class="historical-metric-value">18:00–19:00</div>
                <div class="historical-metric-label">Peak Delay Hour</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Card 3: Passenger Rights & Legal Checklist ───
    st.markdown("""
    <div class="rights-checklist-card" style="margin-top: 20px;">
        <div class="rights-checklist-header">
            <div class="rights-checklist-title">⚖️ Passenger Rights & Legal Checklist</div>
        </div>
        <div class="rights-checklist-item">
            <span class="rights-checklist-icon teal">✈️</span>
            <div class="rights-checklist-content">
                <div class="rights-checklist-item-title">14 CFR §260.6 — Refund Obligation</div>
                <div class="rights-checklist-item-desc">Airlines must provide full refunds for cancelled or significantly changed flights when passenger chooses not to travel.</div>
            </div>
            <span class="rights-checklist-badge">☑️</span>
        </div>
        <div class="rights-checklist-item">
            <span class="rights-checklist-icon amber">🍽️</span>
            <div class="rights-checklist-content">
                <div class="rights-checklist-item-title">Meal & Care During Delay</div>
                <div class="rights-checklist-item-desc">For delays ≥3 hours (domestic) or ≥6 hours (international), airlines must provide meals and communication access.</div>
            </div>
            <span class="rights-checklist-badge">☑️</span>
        </div>
        <div class="rights-checklist-item">
            <span class="rights-checklist-icon blue">🏨</span>
            <div class="rights-checklist-content">
                <div class="rights-checklist-item-title">Hotel Accommodation</div>
                <div class="rights-checklist-item-desc">Overnight delays require hotel accommodation and ground transport when airline is responsible for the delay.</div>
            </div>
            <span class="rights-checklist-badge">☑️</span>
        </div>
        <div class="rights-checklist-item">
            <span class="rights-checklist-icon green">🔄</span>
            <div class="rights-checklist-content">
                <div class="rights-checklist-item-title">Free Rebooking / Original Payment Refund</div>
                <div class="rights-checklist-item-desc">Passengers may rebook on next available flight at no cost, or receive refund to original payment method (not just vouchers).</div>
            </div>
            <span class="rights-checklist-badge">☑️</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN — Chat Interface with Royal Blue Header & 2x2 Prompt Grid
# ═════════════════════════════════════════════════════════════════════════════

with col_right:
    # Chat Header Banner
    st.markdown("""
    <div class="chat-header-banner">
        <div class="chat-header-title">💬 AI Assistant (Flight & Rights Advisor)</div>
    </div>
    """, unsafe_allow_html=True)

    # Chat Container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            _render_sources(message.get("sources", []))
            if message.get("tools_used"):
                st.caption("🔧 Tool yang dijalankan: " + ", ".join(message["tools_used"]))
            if message.get("mode"):
                mode_label = "🧪 V3 Demo" if message["mode"] == "v3_candidate_historical" else "🏭 Production"
                st.caption(f"Mode: {mode_label}")

    # Chat Input
    prompt = st.chat_input("Tanyakan risiko delay atau hak penumpang...", key="chat_input")

    # Handle prompt chip clicks
    if st.session_state.chat_trigger:
        prompt = st.session_state.chat_trigger
        st.session_state.chat_trigger = None

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
                st.caption("🔧 Tool yang dijalankan: " + ", ".join(tools_used))
            mode_label = "🧪 V3 Demo" if mode == "v3_candidate_historical" else "🏭 Production"
            st.caption(f"Mode: {mode_label}")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources, "tools_used": tools_used, "mode": mode}
        )

    st.markdown('</div>', unsafe_allow_html=True)

    # ─── Clickable Prompt Chips — 2x2 Grid ───
    if not st.session_state.messages:  # Only show when no messages yet
        st.markdown("""
        <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--color-lilac);">
            <p style="margin: 0 0 12px 0; font-size: 0.8125rem; color: var(--color-text-muted); font-weight: 500;">💡 Coba pertanyaan cepat:</p>
        </div>
        """, unsafe_allow_html=True)

        prompt_chips = [
            ("✈️ Delay prediction logic?", "Bagaimana cara model memprediksi risiko delay penerbangan saya?"),
            ("⚖️ Compensation eligibility?", "Apa hak kompensasi saya jika penerbangan terlambat atau dibatalkan?"),
            ("🌧️ Weather impact?", "Bagaimana cuaca mempengaruhi risiko delay pada rute ini?"),
            ("🕒 Best departure time?", "Kapan waktu keberangkatan terbaik untuk menghindari delay pada rute ini?")
        ]

        # Create 2 rows x 2 columns
        for row_idx in range(2):
            cols = st.columns(2, gap="small")
            for col_idx in range(2):
                idx = row_idx * 2 + col_idx
                label, full_prompt = prompt_chips[idx]
                with cols[col_idx]:
                    if st.button(label, key=f"prompt_chip_{idx}", use_container_width=True, help=full_prompt):
                        _handle_prompt_chip_click(full_prompt)
                        st.rerun()


# ─── Footer ───
st.markdown("""
<hr style="margin-top: 48px; margin-bottom: 16px; border-color: var(--color-lilac);">
<div style="text-align: center; color: var(--color-text-muted); font-size: 0.8125rem;">
    <p>Flight Delay Advisor · Data sumber: BTS (Bureau of Transportation Statistics) & Regulasi Hak Penumpang</p>
    <p>⚠️ Prediksi berbasis pola historis, bukan status real-time. Konsultasikan dengan maskapai untuk informasi terkini.</p>
</div>
""", unsafe_allow_html=True)