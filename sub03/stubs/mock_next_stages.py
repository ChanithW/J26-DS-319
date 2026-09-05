"""
MOCK Stage 3 (Sub-topic 2, CME) and MOCK Stage 4 (Sub-topic 4, habitability).

They only DISPLAY what Sub-topic 3 passed forward, to prove the hand-off works.
No CME or habitability logic lives here.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from subtopic3.contracts import STAGE1_KEY, STAGE3_KEY, validate_stage3_output
from subtopic3.ui import get_pipeline


def _show_stage3() -> bool:
    stage3 = get_pipeline().get(STAGE3_KEY)
    if not stage3:
        st.error("No Sub-topic 3 output in the pipeline yet. Complete Stage 2 first.")
        return False
    validate_stage3_output(stage3)
    st.success("Received a valid Sub-topic 3 payload (contract check passed).")
    st.dataframe(pd.DataFrame({"field": list(stage3), "value": [str(v) for v in stage3.values()]}),
                 hide_index=True, width="stretch")
    return True


def render_cme() -> None:
    st.title("MOCK - Stage 3: CME prediction (Sub-topic 2)")
    st.warning("**MOCK page.** Replace with Sub-topic 2's real CME page. It shows what Sub-topic 3 passes on.")
    if _show_stage3() and st.button("Continue to habitability  ->", type="primary"):
        st.switch_page("pages/4_MOCK_Habitability.py")


def render_habitability() -> None:
    st.title("MOCK - Stage 4: Habitability risk (Sub-topic 4)")
    st.warning("**MOCK page.** Replace with Sub-topic 4's real page. It shows the pipeline state it would read.")
    st.subheader("Stage 1 output")
    st.json(get_pipeline().get(STAGE1_KEY) or {})
    st.subheader("Sub-topic 3 output")
    _show_stage3()
