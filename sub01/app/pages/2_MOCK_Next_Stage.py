"""MOCK Stage 2: only shows what Stage 1 saved, to prove the hand-off works."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))  # make flare_cnn importable

import streamlit as st  # noqa: E402

from flare_cnn.contracts import STAGE1_KEY, validate_stage1_payload  # noqa: E402
from flare_cnn.ui import get_pipeline  # noqa: E402

st.title("MOCK - Stage 2: Flare frequency & energy (Sub-topic 3)")
st.warning("**MOCK page.** In the group app this is Sub-topic 3's real page. "
           "It shows exactly what Stage 1 passes on.")
stage1 = get_pipeline().get(STAGE1_KEY)
if not stage1:
    st.error("No Stage 1 result yet. Analyse a star on the Stage 1 page and press 'Save result'.")
    if st.button("<- Go to Stage 1"):
        st.switch_page("pages/1_Flare_Detection_Energy.py")
else:
    validate_stage1_payload(stage1)
    st.success("Received a valid Stage 1 payload (contract check passed).")
    d = stage1["flare_details"] or {}
    c1, c2, c3 = st.columns(3)
    c1.metric("is_flare", str(stage1["is_flare"]))
    c2.metric("Flares", d.get("n_flares", 0))
    c3.metric("Strongest tier", d.get("strongest_tier") or "none")
    st.json(stage1)
