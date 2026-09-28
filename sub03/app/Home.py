"""
Standalone Streamlit app for Sub-topic 3 (flare frequency & energy prediction).

Run from sub03/:   streamlit run app/Home.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make sub03/ importable

import streamlit as st  # noqa: E402

from subtopic3.ui import render  # noqa: E402

st.set_page_config(page_title="Sub-topic 3 - Flare prediction", layout="centered")
render()
