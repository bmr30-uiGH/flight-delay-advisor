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
   DESIGN TOKENS — Flight Delay Advisor (New Color Palette)
   Royal Blue, Teal, Amber on Light Modern Background
   ============================================================================ */
:root {
  /* New Color Palette */
  --color-bg: #F4F6F9;
  --color-bg-card: #FFFFFF;
  --color-text-primary: #1A202C;
  --color-text-secondary: #4A5568;
  --color-text-muted: #718096;

  --color-royal-blue: #1E3A8A;
  --color-royal-blue-light: #3B82F6;
  --color-royal-blue-bg: #EBF4FF;
  --color-royal-blue-border: #BFDBFE;

  --color-teal: #0D9488;
  --color-teal-light: #14B8A6;
  --color-teal-bg: #CCFBF1;
  --color-teal-border: #99F6E4;

  --color-amber: #F59E0B;
  --color-amber-light: #FBBF24;
  --color-amber-bg: #FEF3C7;
  --color-amber-border: #FDE68A;

  --color-red: #DC2626;
  --color-red-light: #EF4444;
  --color-red-bg: #FEF2F2;
  --color-red-border: #FECACA;

  --color-green: #16A34A;
  --color-green-light: #22C55E;
  --color-green-bg: #DCFCE7;
  --color-green-border: #BBF7D0;

  /* Spacing Scale */
  --space-1: 4px;
  --space-2: 8px;
  --space-3: 12px;
  --space-4: 16px;
  --space-5: 24px;
  --space-6: 32px;
  --space-7: 48px;
  --space-8: 64px;

  /* Typography */
  --font-sans: 'Inter', 'Atkinson Hyperlegible', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
  --text-xs: 0.75rem;    /* 12px */
  --text-sm: 0.875rem;   /* 14px */
  --text-base: 1rem;     /* 16px */
  --text-lg: 1.125rem;   /* 18px */
  --text-xl: 1.25rem;    /* 20px */
  --text-2xl: 1.5rem;    /* 24px */
  --text-3xl: 1.875rem;  /* 30px */
  --text-4xl: 2.25rem;   /* 36px */

  /* Line Heights */
  --leading-tight: 1.25;
  --leading-snug: 1.375;
  --leading-normal: 1.5;
  --leading-relaxed: 1.625;

  /* Border Radius */
  --radius-sm: 6px;
  --radius-md: 10px;
  --radius-lg: 16px;
  --radius-xl: 20px;
  --radius-full: 9999px;

  /* Shadows */
  --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
  --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
  --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
  --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1), 0 8px 10px -6px rgb(0 0 0 / 0.1);
  --shadow-card: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);

  /* Transitions */
  --duration-fast: 150ms;
  --duration-normal: 200ms;
  --duration-slow: 300ms;
  --ease-out: cubic-bezier(0.4, 0, 0.2, 1);
  --ease-in-out: cubic-bezier(0.4, 0, 0.2, 1);

  /* Z-Index */
  --z-dropdown: 100;
  --z-sticky: 200;
  --z-fixed: 300;
  --z-modal-backdrop: 400;
  --z-modal: 500;
  --z-popover: 600;
  --z-tooltip: 700;
}

/* ============================================================================
   GLOBAL RESETS & BASE
   ============================================================================ */
*, *::before, *::after {
  box-sizing: border-box;
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: var(--font-sans) !important;
  background-color: var(--color-bg) !important;
  color: var(--color-text-primary) !important;
  line-height: var(--leading-normal);
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

/* Typography Scale */
h1, .stTitle { font-size: var(--text-3xl) !important; font-weight: 700 !important; line-height: var(--leading-tight) !important; letter-spacing: -0.02em !important; color: var(--color-text-primary) !important; }
h2, .stHeader { font-size: var(--text-2xl) !important; font-weight: 600 !important; line-height: var(--leading-snug) !important; letter-spacing: -0.01em !important; color: var(--color-text-primary) !important; }
h3 { font-size: var(--text-xl) !important; font-weight: 600 !important; line-height: var(--leading-snug) !important; color: var(--color-text-primary) !important; }
h4 { font-size: var(--text-lg) !important; font-weight: 600 !important; line-height: var(--leading-normal) !important; color: var(--color-text-primary) !important; }

p, .stMarkdown, .stCaption, .stText { font-size: var(--text-base) !important; line-height: var(--leading-normal) !important; color: var(--color-text-secondary) !important; }
small, .stCaption { font-size: var(--text-sm) !important; line-height: var(--leading-normal) !important; color: var(--color-text-muted) !important; }

/* Links */
a { color: var(--color-royal-blue) !important; text-decoration: none !important; transition: color var(--duration-fast) var(--ease-out) !important; }
a:hover { color: var(--color-royal-blue-light) !important; text-decoration: underline !important; }
a:focus-visible { outline: 2px solid var(--color-royal-blue) !important; outline-offset: 2px !important; border-radius: var(--radius-sm) !important; }

/* Code */
code, .stCode { font-family: var(--font-mono) !important; font-size: var(--text-sm) !important; background-color: var(--color-royal-blue-bg) !important; padding: 2px 6px !important; border-radius: var(--radius-sm) !important; color: var(--color-royal-blue) !important; }
pre { font-family: var(--font-mono) !important; font-size: var(--text-sm) !important; background-color: #1E293B !important; color: #E2E8F0 !important; padding: var(--space-4) !important; border-radius: var(--radius-md) !important; overflow-x: auto !important; }

/* ============================================================================
   SIDEBAR STYLING
   ============================================================================ */
[data-testid="stSidebar"] {
  background-color: var(--color-bg-card) !important;
  border-right: 1px solid var(--color-royal-blue-border) !important;
  padding: var(--space-5) var(--space-4) !important;
}

[data-testid="stSidebar"] > div:first-child {
  padding-top: var(--space-4) !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
  color: var(--color-text-primary) !important;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] .stMarkdown small {
  color: var(--color-text-muted) !important;
}

/* Sidebar Section Dividers */
[data-testid="stSidebar"] hr {
  border-color: var(--color-royal-blue-border) !important;
  margin: var(--space-5) 0 !important;
}

/* ============================================================================
   FORM ELEMENTS — INPUTS, SELECTS, BUTTONS
   ============================================================================ */
/* Labels */
label, .stSelectbox label, .stTextInput label, .stDateInput label, .stTimeInput label, .stNumberInput label {
  font-size: var(--text-sm) !important;
  font-weight: 600 !important;
  color: var(--color-text-primary) !important;
  margin-bottom: var(--space-2) !important;
  display: block !important;
}

/* Input Base */
.stTextInput input,
.stSelectbox [data-baseweb="select"] > div,
.stDateInput input,
.stTimeInput input,
.stNumberInput input {
  background-color: var(--color-bg-card) !important;
  border: 1px solid var(--color-royal-blue-border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--color-text-primary) !important;
  font-size: var(--text-base) !important;
  font-family: var(--font-sans) !important;
  padding: var(--space-3) var(--space-4) !important;
  transition: all var(--duration-fast) var(--ease-out) !important;
  width: 100% !important;
  min-height: 44px !important;
}

.stTextInput input:focus,
.stSelectbox [data-baseweb="select"] > div:focus-within,
.stDateInput input:focus,
.stTimeInput input:focus,
.stNumberInput input:focus {
  border-color: var(--color-royal-blue) !important;
  box-shadow: 0 0 0 3px var(--color-royal-blue-bg) !important;
  outline: none !important;
}

.stTextInput input:hover,
.stSelectbox [data-baseweb="select"] > div:hover,
.stDateInput input:hover,
.stTimeInput input:hover,
.stNumberInput input:hover {
  border-color: var(--color-royal-blue-light) !important;
}

/* Disabled State */
.stTextInput input:disabled,
.stSelectbox [data-baseweb="select"] > div:has([aria-disabled="true"]),
.stDateInput input:disabled,
.stTimeInput input:disabled,
.stNumberInput input:disabled,
input:disabled,
[disabled] {
  opacity: 0.5 !important;
  cursor: not-allowed !important;
  background-color: var(--color-bg) !important;
  color: var(--color-text-muted) !important;
  -webkit-text-fill-color: var(--color-text-muted) !important;
}

/* Select Dropdown */
[data-baseweb="popover"] [data-baseweb="menu"] {
  background-color: var(--color-bg-card) !important;
  border: 1px solid var(--color-royal-blue-border) !important;
  border-radius: var(--radius-md) !important;
  box-shadow: var(--shadow-lg) !important;
  padding: var(--space-2) !important;
}

[data-baseweb="menu"] [role="option"] {
  padding: var(--space-3) var(--space-4) !important;
  border-radius: var(--radius-sm) !important;
  font-size: var(--text-base) !important;
  color: var(--color-text-primary) !important;
  transition: background-color var(--duration-fast) var(--ease-out) !important;
}

[data-baseweb="menu"] [role="option"]:hover,
[data-baseweb="menu"] [role="option"][aria-selected="true"] {
  background-color: var(--color-royal-blue-bg) !important;
  color: var(--color-royal-blue) !important;
}

/* Date/Time Input Calendar */
[data-baseweb="calendar"] {
  background-color: var(--color-bg-card) !important;
  border: 1px solid var(--color-royal-blue-border) !important;
  border-radius: var(--radius-md) !important;
  box-shadow: var(--shadow-lg) !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button,
.stFormSubmitButton > button {
  font-family: var(--font-sans) !important;
  font-size: var(--text-base) !important;
  font-weight: 600 !important;
  border-radius: var(--radius-md) !important;
  padding: var(--space-3) var(--space-5) !important;
  min-height: 44px !important;
  transition: all var(--duration-fast) var(--ease-out) !important;
  border: none !important;
  cursor: pointer !important;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: var(--space-2) !important;
}

/* Primary Button - Royal Blue */
.stButton > button[kind="primary"],
.stFormSubmitButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
  background-color: var(--color-royal-blue) !important;
  color: var(--color-on-primary) !important;
}

.stButton > button[kind="primary"]:hover,
.stFormSubmitButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
  background-color: var(--color-royal-blue-light) !important;
  transform: translateY(-1px) !important;
  box-shadow: var(--shadow-md) !important;
}

.stButton > button[kind="primary"]:active,
.stFormSubmitButton > button[kind="primary"]:active {
  transform: translateY(0) !important;
  box-shadow: var(--shadow-sm) !important;
}

.stButton > button[kind="primary"]:focus-visible,
.stFormSubmitButton > button[kind="primary"]:focus-visible,
.stDownloadButton > button[kind="primary"]:focus-visible {
  outline: 2px solid var(--color-royal-blue) !important;
  outline-offset: 2px !important;
}

/* Secondary Button */
.stButton > button[kind="secondary"] {
  background-color: var(--color-bg) !important;
  color: var(--color-text-secondary) !important;
  border: 1px solid var(--color-royal-blue-border) !important;
}

.stButton > button[kind="secondary"]:hover {
  background-color: var(--color-royal-blue-bg) !important;
  color: var(--color-royal-blue) !important;
  border-color: var(--color-royal-blue) !important;
}

/* Prompt Chip Buttons */
.stButton > button[data-prompt-chip="true"] {
  background-color: var(--color-bg-card) !important;
  color: var(--color-text-secondary) !important;
  border: 1px solid var(--color-royal-blue-border) !important;
  padding: var(--space-2) var(--space-4) !important;
  font-size: var(--text-sm) !important;
  font-weight: 500 !important;
  border-radius: var(--radius-full) !important;
  min-height: 36px !important;
  transition: all var(--duration-fast) var(--ease-out) !important;
  white-space: nowrap !important;
}

.stButton > button[data-prompt-chip="true"]:hover {
  background-color: var(--color-royal-blue-bg) !important;
  color: var(--color-royal-blue) !important;
  border-color: var(--color-royal-blue) !important;
  transform: translateY(-1px) !important;
  box-shadow: var(--shadow-sm) !important;
}

/* Disabled Button */
.stButton > button:disabled,
.stFormSubmitButton > button:disabled,
.stDownloadButton > button:disabled {
  opacity: 0.5 !important;
  cursor: not-allowed !important;
  transform: none !important;
}

/* ============================================================================
   CARD & CONTAINER COMPONENTS
   ============================================================================ */
.app-card {
  background-color: var(--color-bg-card) !important;
  border: 1px solid var(--color-royal-blue-border) !important;
  border-radius: var(--radius-lg) !important;
  padding: var(--space-5) !important;
  box-shadow: var(--shadow-card) !important;
  transition: box-shadow var(--duration-normal) var(--ease-out), transform var(--duration-normal) var(--ease-out) !important;
}

.app-card:hover {
  box-shadow: var(--shadow-lg) !important;
  transform: translateY(-2px) !important;
}

.app-card-header {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  margin-bottom: var(--space-4) !important;
  padding-bottom: var(--space-3) !important;
  border-bottom: 1px solid var(--color-royal-blue-border) !important;
}

.app-card-title {
  font-size: var(--text-lg) !important;
  font-weight: 600 !important;
  color: var(--color-text-primary) !important;
  margin: 0 !important;
}

.app-card-subtitle {
  font-size: var(--text-sm) !important;
  color: var(--color-text-muted) !important;
  margin: var(--space-1) 0 0 0 !important;
}

/* ============================================================================
   BOARDING PASS CARD
   ============================================================================ */
.boarding-pass {
  background: linear-gradient(135deg, var(--color-royal-blue) 0%, var(--color-royal-blue-light) 100%) !important;
  border-radius: var(--radius-xl) !important;
  padding: var(--space-6) !important;
  color: white !important;
  position: relative !important;
  overflow: hidden !important;
  box-shadow: var(--shadow-lg) !important;
}

.boarding-pass::before {
  content: '' !important;
  position: absolute !important;
  top: -50% !important;
  right: -10% !important;
  width: 200% !important;
  height: 200% !important;
  background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%) !important;
  pointer-events: none !important;
}

.boarding-pass-header {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  margin-bottom: var(--space-5) !important;
  position: relative !important;
  z-index: 1 !important;
}

.boarding-pass-airline {
  font-size: var(--text-sm) !important;
  font-weight: 600 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.1em !important;
  opacity: 0.9 !important;
}

.boarding-pass-flight {
  font-size: var(--text-2xl) !important;
  font-weight: 700 !important;
  letter-spacing: 0.05em !important;
}

.boarding-pass-route {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  margin-bottom: var(--space-5) !important;
  position: relative !important;
  z-index: 1 !important;
}

.boarding-pass-origin,
.boarding-pass-dest {
  text-align: center !important;
  flex: 1 !important;
}

.boarding-pass-code {
  font-size: var(--text-4xl) !important;
  font-weight: 800 !important;
  letter-spacing: 0.1em !important;
  line-height: 1 !important;
  margin-bottom: var(--space-1) !important;
}

.boarding-pass-city {
  font-size: var(--text-sm) !important;
  opacity: 0.8 !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
}

.boarding-pass-arrow {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  padding: 0 var(--space-4) !important;
  color: rgba(255,255,255,0.7) !important;
}

.boarding-pass-arrow svg {
  width: 32px !important;
  height: 32px !important;
}

.boarding-pass-details {
  display: grid !important;
  grid-template-columns: repeat(3, 1fr) !important;
  gap: var(--space-4) !important;
  padding-top: var(--space-4) !important;
  border-top: 1px solid rgba(255,255,255,0.2) !important;
  position: relative !important;
  z-index: 1 !important;
}

.boarding-pass-detail {
  text-align: center !important;
}

.boarding-pass-detail-label {
  font-size: var(--text-xs) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
  opacity: 0.7 !important;
  margin-bottom: var(--space-1) !important;
}

.boarding-pass-detail-value {
  font-size: var(--text-lg) !important;
  font-weight: 700 !important;
  line-height: 1.2 !important;
}

.boarding-pass-risk {
  display: inline-flex !important;
  align-items: center !important;
  gap: var(--space-2) !important;
  padding: var(--space-2) var(--space-4) !important;
  border-radius: var(--radius-full) !important;
  font-size: var(--text-sm) !important;
  font-weight: 600 !important;
  position: relative !important;
  z-index: 1 !important;
}

.boarding-pass-risk.low {
  background: rgba(34, 197, 94, 0.2) !important;
  color: #22C55E !important;
  border: 1px solid rgba(34, 197, 94, 0.3) !important;
}

.boarding-pass-risk.medium {
  background: rgba(245, 158, 11, 0.2) !important;
  color: #F59E0B !important;
  border: 1px solid rgba(245, 158, 11, 0.3) !important;
}

.boarding-pass-risk.high {
  background: rgba(220, 38, 38, 0.2) !important;
  color: #EF4444 !important;
  border: 1px solid rgba(220, 38, 38, 0.3) !important;
}

/* ============================================================================
   METRIC CARDS
   ============================================================================ */
.metric-card {
  background-color: var(--color-bg-card) !important;
  border: 1px solid var(--color-royal-blue-border) !important;
  border-radius: var(--radius-lg) !important;
  padding: var(--space-4) !important;
  text-align: center !important;
  transition: all var(--duration-normal) var(--ease-out) !important;
}

.metric-card:hover {
  transform: translateY(-2px) !important;
  box-shadow: var(--shadow-md) !important;
}

.metric-value {
  font-size: var(--text-3xl) !important;
  font-weight: 700 !important;
  line-height: var(--leading-tight) !important;
  color: var(--color-royal-blue) !important;
  margin: 0 !important;
}

.metric-label {
  font-size: var(--text-xs) !important;
  color: var(--color-text-muted) !important;
  margin-top: var(--space-2) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
}

/* ============================================================================
   CHECKLIST ITEMS
   ============================================================================ */
.checklist-item {
  display: flex !important;
  align-items: flex-start !important;
  gap: var(--space-3) !important;
  padding: var(--space-3) 0 !important;
  border-bottom: 1px solid var(--color-royal-blue-border) !important;
}

.checklist-item:last-child {
  border-bottom: none !important;
}

.checklist-icon {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  width: 28px !important;
  height: 28px !important;
  border-radius: var(--radius-full) !important;
  flex-shrink: 0 !important;
  font-size: var(--text-sm) !important;
}

.checklist-icon.teal {
  background: var(--color-teal-bg) !important;
  color: var(--color-teal) !important;
}

.checklist-icon.amber {
  background: var(--color-amber-bg) !important;
  color: var(--color-amber) !important;
}

.checklist-icon.blue {
  background: var(--color-royal-blue-bg) !important;
  color: var(--color-royal-blue) !important;
}

.checklist-content {
  flex: 1 !important;
}

.checklist-title {
  font-size: var(--text-sm) !important;
  font-weight: 600 !important;
  color: var(--color-text-primary) !important;
  margin: 0 0 var(--space-1) 0 !important;
}

.checklist-desc {
  font-size: var(--text-xs) !important;
  color: var(--color-text-muted) !important;
  margin: 0 !important;
  line-height: var(--leading-normal) !important;
}

/* ============================================================================
   PROMPT CHIPS CONTAINER
   ============================================================================ */
.prompt-chips-container {
  display: flex !important;
  flex-wrap: wrap !important;
  gap: var(--space-2) !important;
  margin-top: var(--space-3) !important;
  padding: var(--space-3) 0 !important;
  border-top: 1px solid var(--color-royal-blue-border) !important;
}

/* ============================================================================
   HISTORICAL METRICS ROW
   ============================================================================ */
.metrics-row {
  display: grid !important;
  grid-template-columns: repeat(3, 1fr) !important;
  gap: var(--space-3) !important;
  margin-top: var(--space-4) !important;
}

/* ============================================================================
   CHAT INTERFACE
   ============================================================================ */
[data-testid="stChatMessage"] {
  background-color: var(--color-bg-card) !important;
  border: 1px solid var(--color-royal-blue-border) !important;
  border-radius: var(--radius-lg) !important;
  padding: var(--space-4) !important;
  margin-bottom: var(--space-3) !important;
  box-shadow: var(--shadow-sm) !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
  border-left: 3px solid var(--color-royal-blue) !important;
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  border-left: 3px solid var(--color-amber) !important;
}

[data-testid="stChatMessageContent"] {
  font-size: var(--text-base) !important;
  line-height: var(--leading-relaxed) !important;
  color: var(--color-text-primary) !important;
}

[data-testid="stChatInput"] {
  background-color: transparent !important;
  border-top: 1px solid var(--color-royal-blue-border) !important;
  padding: var(--space-4) !important;
}

[data-testid="stChatInput"] > div,
[data-testid="stBottom"] div,
[data-testid="stBottomBlockContainer"] div {
  background-color: transparent !important;
}

[data-testid="stChatInput"] textarea {
  background-color: var(--color-bg-card) !important;
  border: 1px solid var(--color-royal-blue-border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--color-text-primary) !important;
  font-family: var(--font-sans) !important;
  font-size: var(--text-base) !important;
  padding: var(--space-3) var(--space-4) !important;
  min-height: 44px !important;
  resize: none !important;
  transition: all var(--duration-fast) var(--ease-out) !important;
}

[data-testid="stChatInput"] textarea::placeholder {
  color: var(--color-text-muted) !important;
}

[data-testid="stChatInput"] textarea:focus {
  border-color: var(--color-royal-blue) !important;
  box-shadow: 0 0 0 3px var(--color-royal-blue-bg) !important;
  outline: none !important;
}

[data-testid="stChatInput"] button {
  background-color: var(--color-royal-blue) !important;
  color: white !important;
  border-radius: var(--radius-md) !important;
  min-width: 44px !important;
  min-height: 44px !important;
  transition: all var(--duration-fast) var(--ease-out) !important;
}

[data-testid="stChatInput"] button:hover {
  background-color: var(--color-royal-blue-light) !important;
}

[data-testid="stChatInput"] button:focus-visible {
  outline: 2px solid var(--color-royal-blue) !important;
  outline-offset: 2px !important;
}

/* ============================================================================
   EXPANDERS
   ============================================================================ */
[data-testid="stExpander"] {
  background-color: var(--color-bg-card) !important;
  border: 1px solid var(--color-royal-blue-border) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden !important;
}

[data-testid="stExpander"] summary {
  font-size: var(--text-base) !important;
  font-weight: 600 !important;
  color: var(--color-text-primary) !important;
  background-color: var(--color-bg-card) !important;
  padding: var(--space-4) !important;
  cursor: pointer !important;
  list-style: none !important;
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  transition: background-color var(--duration-fast) var(--ease-out) !important;
}

[data-testid="stExpander"] summary:hover {
  background-color: var(--color-bg) !important;
}

[data-testid="stExpander"] summary:focus-visible {
  outline: 2px solid var(--color-royal-blue) !important;
  outline-offset: -2px !important;
}

[data-testid="stExpander"] summary::-webkit-details-marker { display: none !important; }
[data-testid="stExpander"] summary::before {
  content: '▶' !important;
  font-size: var(--text-xs) !important;
  color: var(--color-text-muted) !important;
  transition: transform var(--duration-fast) var(--ease-out) !important;
  margin-right: var(--space-2) !important;
}

[data-testid="stExpander"] details[open] summary::before {
  transform: rotate(90deg) !important;
}

[data-testid="stExpander"] > div:last-child {
  padding: 0 var(--space-4) var(--space-4) !important;
  border-top: 1px solid var(--color-royal-blue-border) !important;
}

/* ============================================================================
   ALERTS / NOTIFICATIONS
   ============================================================================ */
.stAlert {
  border-radius: var(--radius-md) !important;
  border: none !important;
  padding: var(--space-4) var(--space-5) !important;
  font-size: var(--text-base) !important;
  line-height: var(--leading-normal) !important;
}

.stAlert[data-baseweb="notification"][kind="info"] {
  background-color: var(--color-royal-blue-bg) !important;
  color: var(--color-royal-blue) !important;
  border: 1px solid var(--color-royal-blue-border) !important;
  border-left: 3px solid var(--color-royal-blue) !important;
}

.stAlert[data-baseweb="notification"][kind="success"] {
  background-color: var(--color-green-bg) !important;
  color: var(--color-green) !important;
  border: 1px solid var(--color-green-border) !important;
  border-left: 3px solid var(--color-green) !important;
}

.stAlert[data-baseweb="notification"][kind="warning"] {
  background-color: var(--color-amber-bg) !important;
  color: var(--color-amber) !important;
  border: 1px solid var(--color-amber-border) !important;
  border-left: 3px solid var(--color-amber) !important;
}

.stAlert[data-baseweb="notification"][kind="error"] {
  background-color: var(--color-red-bg) !important;
  color: var(--color-red) !important;
  border: 1px solid var(--color-red-border) !important;
  border-left: 3px solid var(--color-red) !important;
}

/* ============================================================================
   SPINNER / LOADING
   ============================================================================ */
.stSpinner > div {
  border-color: var(--color-royal-blue) transparent var(--color-royal-blue) transparent !important;
}

.stSpinner [data-testid="stStatusWidget"] {
  background-color: var(--color-bg-card) !important;
  border: 1px solid var(--color-royal-blue-border) !important;
  border-radius: var(--radius-md) !important;
  padding: var(--space-4) !important;
  box-shadow: var(--shadow-lg) !important;
}

/* ============================================================================
   TABS
   ============================================================================ */
[data-testid="stTabs"] {
  border-bottom: 1px solid var(--color-royal-blue-border) !important;
  margin-bottom: var(--space-4) !important;
}

[data-testid="stTabs"] [role="tablist"] {
  gap: var(--space-1) !important;
}

[data-testid="stTabs"] [role="tab"] {
  background-color: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  color: var(--color-text-muted) !important;
  font-size: var(--text-base) !important;
  font-weight: 500 !important;
  padding: var(--space-3) var(--space-4) !important;
  transition: all var(--duration-fast) var(--ease-out) !important;
  min-height: 44px !important;
}

[data-testid="stTabs"] [role="tab"]:hover {
  color: var(--color-text-primary) !important;
  background-color: var(--color-bg) !important;
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
  color: var(--color-royal-blue) !important;
  border-bottom-color: var(--color-royal-blue) !important;
  background-color: transparent !important;
}

[data-testid="stTabs"] [role="tab"]:focus-visible {
  outline: 2px solid var(--color-royal-blue) !important;
  outline-offset: -2px !important;
  border-radius: var(--radius-sm) var(--radius-sm) 0 0 !important;
}

[data-testid="stTabs"] [role="tabpanel"] {
  padding: var(--space-4) 0 !important;
}

/* ============================================================================
   DATA TABLE / DATAFRAME
   ============================================================================ */
[data-testid="stDataFrame"] {
  border: 1px solid var(--color-royal-blue-border) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden !important;
  background-color: var(--color-bg-card) !important;
}

[data-testid="stDataFrame"] table {
  font-size: var(--text-sm) !important;
}

[data-testid="stDataFrame"] th {
  background-color: var(--color-bg) !important;
  color: var(--color-text-primary) !important;
  font-weight: 600 !important;
  padding: var(--space-3) var(--space-4) !important;
  border-bottom: 1px solid var(--color-royal-blue-border) !important;
  white-space: nowrap !important;
}

[data-testid="stDataFrame"] td {
  padding: var(--space-3) var(--space-4) !important;
  border-bottom: 1px solid var(--color-royal-blue-border) !important;
  color: var(--color-text-secondary) !important;
}

[data-testid="stDataFrame"] tr:last-child td {
  border-bottom: none !important;
}

[data-testid="stDataFrame"] tr:hover td {
  background-color: var(--color-bg) !important;
}

/* ============================================================================
   PROGRESS BAR
   ============================================================================ */
.stProgress > div > div {
  background-color: var(--color-royal-blue) !important;
  border-radius: var(--radius-full) !important;
}

.stProgress > div {
  background-color: var(--color-royal-blue-border) !important;
  border-radius: var(--radius-full) !important;
  height: 8px !important;
}

/* ============================================================================
   HORIZONTAL RULE / DIVIDER
   ============================================================================ */
hr, .stDivider {
  border-color: var(--color-royal-blue-border) !important;
  margin: var(--space-5) 0 !important;
}

/* ============================================================================
   TOOLTIPS
   ============================================================================ */
[data-testid="stTooltip"] {
  background-color: var(--color-bg-card) !important;
  border: 1px solid var(--color-royal-blue-border) !important;
  border-radius: var(--radius-md) !important;
  box-shadow: var(--shadow-lg) !important;
  padding: var(--space-3) var(--space-4) !important;
  font-size: var(--text-sm) !important;
  color: var(--color-text-primary) !important;
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
  :root {
    --space-5: 20px;
    --space-6: 28px;
  }
  [data-testid="stAppViewContainer"] > .main {
    padding: var(--space-4) var(--space-4) !important;
  }
  .metrics-row {
    grid-template-columns: 1fr !important;
  }
}

@media (max-width: 768px) {
  :root {
    --text-3xl: 1.375rem;
    --text-2xl: 1.25rem;
    --space-4: 12px;
    --space-5: 16px;
  }
  [data-testid="stSidebar"] {
    padding: var(--space-4) var(--space-3) !important;
  }
  .boarding-pass-route {
    flex-direction: column !important;
  }
  .boarding-pass-arrow {
    transform: rotate(90deg) !important;
    padding: var(--space-4) 0 !important;
  }
  .boarding-pass-details {
    grid-template-columns: 1fr !important;
  }
}

@media (max-width: 480px) {
  :root {
    --text-base: 0.9375rem;
  }
  .stButton > button,
  .stFormSubmitButton > button {
    padding: var(--space-3) var(--space-4) !important;
    font-size: var(--text-sm) !important;
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
  .app-card, .metric-card, .boarding-pass {
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


def _render_risk_gauge(risk_score: float, risk_level: str) -> str:
    """Generate SVG for a circular risk gauge."""
    score = max(0, min(100, risk_score))
    radius = 56
    circumference = 2 * 3.14159 * radius
    offset = circumference - (score / 100) * circumference

    if risk_level.lower() == "low" or score < 35:
        color_class = "low"
    elif risk_level.lower() == "medium" or score < 70:
        color_class = "medium"
    else:
        color_class = "high"

    return f"""
    <div class="risk-gauge" role="img" aria-label="Risk gauge showing {score:.0f}% delay risk, level {risk_level}">
        <svg viewBox="0 0 128 128">
            <circle class="bg" cx="64" cy="64" r="{radius}" />
            <circle class="progress {color_class}" cx="64" cy="64" r="{radius}"
                    stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{offset:.1f}" />
        </svg>
        <div class="risk-gauge-text">
            <div class="risk-score">{score:.0f}%</div>
            <div class="risk-label">Risiko Delay</div>
        </div>
    </div>
    """


def _render_risk_badge(risk_level: str) -> str:
    level = risk_level.lower()
    if level == "low":
        return '<span class="risk-level low">✓ Risiko Rendah</span>'
    elif level == "medium":
        return '<span class="risk-level medium">⚠ Risiko Sedang</span>'
    else:
        return '<span class="risk-level high">⚠ Risiko Tinggi</span>'


def _get_risk_info(probability: float) -> tuple:
    """Get risk level, label, and CSS class based on probability."""
    if probability < 0.35:
        return "low", "Risiko Rendah", "low"
    elif probability < 0.70:
        return "medium", "Risiko Sedang", "medium"
    else:
        return "high", "Risiko Tinggi", "high"


def _handle_prompt_chip_click(prompt_text: str):
    """Handle prompt chip click - add to chat and trigger agent."""
    st.session_state.messages.append({"role": "user", "content": prompt_text})
    st.session_state.chat_trigger = prompt_text


# ═════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT
# ═════════════════════════════════════════════════════════════════════════════

# ─── Inject Google Fonts ───
st.markdown(FONT_IMPORT, unsafe_allow_html=True)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

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
from src.reference_data import (
    AIRLINE_NAMES,
    airport_label,
    build_route_reference,
    carrier_label,
    load_airport_directory,
    load_carrier_directory,
)

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

# ─── Sidebar ───
with st.sidebar:
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h2 style="margin: 0 0 8px 0; font-size: 1.125rem; color: var(--color-text-primary);">Detail Penerbangan</h2>
        <p style="margin: 0; font-size: 0.8125rem; color: var(--color-text-muted);">
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

# ─── Flight Summary / Boarding Pass (if saved) ───
if st.session_state.get("flight_summary"):
    saved = st.session_state.flight_summary

    # Get risk prediction if available
    risk_level = "low"
    risk_label = "Risiko Rendah"
    risk_class = "low"
    probability = 0.18

    if st.session_state.flight_inputs:
        try:
            result = predict_delay(st.session_state.flight_inputs)
            if result.get("ok") and result.get("probability") is not None:
                probability = result["probability"]
                risk_level, risk_label, risk_class = _get_risk_info(probability)
        except:
            pass

    probability_pct = int(probability * 100)
    risk_text = f"{risk_label} (~{probability_pct}%)" if probability > 0 else "Belum diprediksi"

    st.markdown(f"""
    <div class="boarding-pass">
        <div class="boarding-pass-header">
            <span class="boarding-pass-airline">{saved['carrier']}</span>
            <span class="boarding-pass-flight">{saved['origin_code']} → {saved['destination_code']}</span>
        </div>
        <div class="boarding-pass-route">
            <div class="boarding-pass-origin">
                <div class="boarding-pass-code">{saved['origin_code']}</div>
                <div class="boarding-pass-city">{saved['origin'].split(',')[0] if ',' in saved['origin'] else saved['origin'][:20]}</div>
            </div>
            <div class="boarding-pass-arrow">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="2" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" d="M17.25 5.25L5.25 12l12 6.75" /></svg>
            </div>
            <div class="boarding-pass-dest">
                <div class="boarding-pass-code">{saved['destination_code']}</div>
                <div class="boarding-pass-city">{saved['destination'].split(',')[0] if ',' in saved['destination'] else saved['destination'][:20]}</div>
            </div>
        </div>
        <div class="boarding-pass-details">
            <div class="boarding-pass-detail">
                <div class="boarding-pass-detail-label">Tanggal</div>
                <div class="boarding-pass-detail-value">{saved['flight_date']}</div>
            </div>
            <div class="boarding-pass-detail">
                <div class="boarding-pass-detail-label">Waktu</div>
                <div class="boarding-pass-detail-value">{saved['departure_time']}</div>
            </div>
            <div class="boarding-pass-detail">
                <div class="boarding-pass-detail-label">Jarak</div>
                <div class="boarding-pass-detail-value">{saved['distance']:,} mi</div>
            </div>
        </div>
        <div style="margin-top: 16px; display: flex; justify-content: flex-end; position: relative; z-index: 1;">
            <span class="boarding-pass-risk {risk_class}">{risk_text}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Historical Route Metrics ───
    st.markdown("""
    <div class="metrics-row">
        <div class="metric-card">
            <div class="metric-value">12m</div>
            <div class="metric-label">Rata-rata Delay</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">88%</div>
            <div class="metric-label">Keandalan Rute</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">18:00-19:00</div>
            <div class="metric-label">Jam Sibuk</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Passenger Rights Checklist ───
    st.markdown("""
    <div class="app-card" style="margin-top: 24px;">
        <div class="app-card-header">
            <div class="app-card-title">✅ Hak Penumpang & Legal Checklist</div>
        </div>
        <div class="checklist-item">
            <span class="checklist-icon teal">☑️</span>
            <div class="checklist-content">
                <div class="checklist-title">14 CFR §260.6 - Refund Kewajiban Maskapai</div>
                <div class="checklist-desc">Maskapai wajib refund penuh jika penerbangan dibatalkan atau berubah signifikan, dan penumpang memilih tidak bepergian.</div>
            </div>
        </div>
        <div class="checklist-item">
            <span class="checklist-icon amber">☑️</span>
            <div class="checklist-content">
                <div class="checklist-title">14 CFR §260.6(a)(2) - Pembatalan Penerbangan</div>
                <div class="checklist-desc">Pembatalan merupakan pemicu refund tersendiri. Tidak mensyaratkan ambang keterlambatan 3 jam domestik/6 jam internasional.</div>
            </div>
        </div>
        <div class="checklist-item">
            <span class="checklist-icon teal">☑️</span>
            <div class="checklist-content">
                <div class="checklist-title">14 CFR §399.80(l) - Ticket Agent/OTA Refund</div>
                <div class="checklist-desc">Jika OTA adalah merchant of record, refund diproses atas permintaan konsumen (7 hari kerja kartu kredit / 20 hari kalender metode lain).</div>
            </div>
        </div>
        <div class="checklist-item">
            <span class="checklist-icon blue">☑️</span>
            <div class="checklist-content">
                <div class="checklist-title">PM 89/2015 (Indonesia) - Kompensasi Delay</div>
                <div class="checklist-desc">Kategori 4: delay 181-240 menit; Kategori 5: >240 menit. Akomodasi Pasal 10 ayat (4) mensyaratkan delay >6 jam.</div>
            </div>
        </div>
        <div class="checklist-item">
            <span class="checklist-icon teal">☑️</span>
            <div class="checklist-content">
                <div class="checklist-title">Refund ke Metode Pembayaran Asli</div>
                <div class="checklist-desc">Travel credit/voucher bukan pilihan wajib. Penumpang berhak menolak dan meminta refund ke metode pembayaran asal.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# RIGHT COLUMN - CHAT INTERFACE
# ═════════════════════════════════════════════════════════════════════════════

# Create two columns: Left (Boarding Pass) and Right (Chat)
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # This will be rendered above - the boarding pass section
    pass

with col_right:
    # ─── Chat Interface ───
    st.markdown("""
    <div style="margin-bottom: 16px;">
        <h2 style="margin: 0 0 4px 0; font-size: 1.25rem; color: var(--color-text-primary);">💬 Konsultasi Penerbangan</h2>
        <p style="margin: 0; color: var(--color-text-muted); font-size: 0.875rem;">Tanyakan risiko delay, hak penumpang, atau prediksi waktu terbaik</p>
    </div>
    """, unsafe_allow_html=True)

    # Chat messages container
    st.markdown('<div id="chat-container">', unsafe_allow_html=True)

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

    # ─── Clickable Prompt Chips ───
    if not st.session_state.messages:  # Only show when no messages yet
        st.markdown("""
        <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--color-royal-blue-border);">
            <p style="margin: 0 0 12px 0; font-size: 0.8125rem; color: var(--color-text-muted); font-weight: 500;">💡 Coba pertanyaan cepat:</p>
        </div>
        """, unsafe_allow_html=True)

        prompt_chips = [
            ("✈️ Delay prediction logic?", "Bagaimana cara model memprediksi risiko delay penerbangan saya?"),
            ("⚖️ Compensation eligibility?", "Apa hak kompensasi saya jika penerbangan terlambat atau dibatalkan?"),
            ("🌧️ Weather impact?", "Bagaimana cuaca mempengaruhi risiko delay pada rute ini?"),
            ("🕒 Best departure time?", "Kapan waktu keberangkatan terbaik untuk menghindari delay pada rute ini?")
        ]

        cols = st.columns(4, gap="small")
        for idx, (label, full_prompt) in enumerate(prompt_chips):
            with cols[idx]:
                if st.button(label, key=f"prompt_chip_{idx}", use_container_width=True, help=full_prompt):
                    _handle_prompt_chip_click(full_prompt)
                    st.rerun()


# ─── Footer ───
st.markdown("""
<hr style="margin-top: 48px; margin-bottom: 16px;">
<div style="text-align: center; color: var(--color-text-muted); font-size: 0.8125rem;">
    <p>Flight Delay Advisor · Data sumber: BTS (Bureau of Transportation Statistics) & Regulasi Hak Penumpang</p>
    <p>⚠️ Prediksi berbasis pola historis, bukan status real-time. Konsultasikan dengan maskapai untuk informasi terkini.</p>
</div>
""", unsafe_allow_html=True)