"""
Standalone Streamlit app for Sub-topic 3 (flare frequency & energy prediction).

Run from sub03/:   streamlit run app/Home.py
"""
import sys
from pathlib import Path

# Streamlit runs this file as a script, so add sub03/ to the import path to make
# the 'subtopic3' package importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make sub03/ importable

import streamlit as st  # noqa: E402

from subtopic3.ui import render  # noqa: E402

# Browser-tab title/icon and page width; must be the first Streamlit call.
st.set_page_config(page_title="Sub-topic 3 - Flare prediction", page_icon="🌟", layout="centered")
# Draw the page (Streamlit re-executes this script on every user interaction).
render()
