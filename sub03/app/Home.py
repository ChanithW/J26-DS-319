"""
Standalone app for testing Sub-topic 3 inside the 4-stage pipeline.

Run from sub03/:   streamlit run app/Home.py
Pages 1, 3 and 4 are MOCKs (stubs/); page 2 is the real Sub-topic 3 page.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make sub03/ importable

import streamlit as st  # noqa: E402

from subtopic3.contracts import STAGE1_KEY, STAGE3_KEY  # noqa: E402
from subtopic3.ui import get_pipeline  # noqa: E402

st.set_page_config(page_title="J26-DS-319 - Sub-topic 3", layout="wide")
st.title("AI-Driven Stellar Flare Analysis - pipeline test harness")
st.markdown(
    """
| Stage | Page | Owner | In this app |
|---|---|---|---|
| 1 | Flare detection | Sub-topic 1 | **MOCK** |
| 2 | Flare frequency & energy | **Sub-topic 3** | real |
| 3 | CME prediction | Sub-topic 2 | **MOCK** |
| 4 | Habitability risk | Sub-topic 4 | **MOCK** |
"""
)
pipeline = get_pipeline()
st.write(f"Stage 1 result stored: **{STAGE1_KEY in pipeline}** - "
         f"Sub-topic 3 result stored: **{STAGE3_KEY in pipeline}**")
if st.button("Start at Stage 1  ->", type="primary"):
    st.switch_page("pages/1_MOCK_Stage1_Flare_Detection.py")
