"""
Standalone app for Sub-topic 1 (flare detection & energy classification).

Run from sub01/:   streamlit run app/Home.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # make flare_cnn importable

import streamlit as st  # noqa: E402

from flare_cnn.ui import render  # noqa: E402

st.set_page_config(page_title="J26-DS-319 - Sub-topic 1", page_icon="🌟", layout="wide")

render()
