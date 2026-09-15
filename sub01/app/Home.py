"""
Standalone app for Sub-topic 1 (flare detection & energy classification).

Run from sub01/:   streamlit run app/Home.py
Page 1 is the real Stage-1 page; page 2 is a MOCK that shows what is handed to Stage 2.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # make flare_cnn importable

import streamlit as st  # noqa: E402

from flare_cnn.contracts import STAGE1_KEY  # noqa: E402
from flare_cnn.ui import get_pipeline  # noqa: E402

st.set_page_config(page_title="J26-DS-319 - Sub-topic 1", page_icon="🌟", layout="wide")
st.title("🌟 Stellar Flare Detection & Energy Classification")
st.markdown(
    """
Stars sometimes release sudden bursts of energy called **flares**, which can affect whether
nearby planets could support life. This stage uses NASA **TESS** light curves and a
deep-learning model to **find flares** and **tell how powerful each one is**
(C, M, X or Superflare) in a single step, with a focus on **weak flares**.

| Stage | Page | Owner | In this app |
|---|---|---|---|
| 1 | Flare detection & energy | **Sub-topic 1** | real |
| 2 | Flare frequency & energy | Sub-topic 3 | **MOCK** (shows the hand-off) |
"""
)
st.write(f"Stage 1 result stored: **{STAGE1_KEY in get_pipeline()}**")
if st.button("Start at Stage 1  ->", type="primary"):
    st.switch_page("pages/1_Flare_Detection_Energy.py")
