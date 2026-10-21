"""Shared page styling so all four sub-topic apps look the same.

Colours and fonts come from .streamlit/config.toml; this adds the card, header
and button details that the config file cannot express. Call apply_theme()
once, right after st.set_page_config().
"""
import streamlit as st

_CSS = """
<style>
.block-container { padding-top: 2rem; max-width: 1200px; }
h1 { font-weight: 700; letter-spacing: -0.02em; }
h1, h2, h3 { color: var(--text-color); }
[data-testid="stMetric"] {
    background: var(--secondary-background-color);
    border: 1px solid rgba(100, 116, 139, 0.25);
    border-radius: 0.5rem;
    padding: 0.75rem 1rem;
}
[data-testid="stMetricLabel"] { opacity: 0.75; }
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {
    border-radius: 0.375rem; font-weight: 500;
}
[data-testid="stAlert"] { border-radius: 0.5rem; }
[data-testid="stDataFrame"] { border: 1px solid rgba(100, 116, 139, 0.25); border-radius: 0.5rem; }
button[data-baseweb="tab"] { font-weight: 500; }
</style>
"""


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
