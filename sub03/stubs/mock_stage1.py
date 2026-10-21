"""
MOCK Stage 1 (stands in for Sub-topic 1's flare-detection page).

Lets the tester toggle is_flare and choose a star, then writes a payload that
follows STAGE1_OUTPUT_FIELDS to st.session_state["pipeline"]["stage1"].
There is NO classification here - the verdict is whatever the tester picks.
"""
from __future__ import annotations

import streamlit as st

from subtopic3.contracts import STAGE1_KEY, STELLAR_PARAM_FIELDS, validate_stage1_output
from subtopic3.pipeline import load_catalog
from subtopic3.ui import PARAM_LABELS, get_pipeline, reset_stage3_state

SUB3_PAGE = "pages/2_Flare_Frequency_Energy.py"


def _example_kics() -> dict[str, int]:
    """One example star per spectral type (the most active one), for quick testing."""
    cat = load_catalog()
    picks = cat.sort_values("Nf", ascending=False).groupby("spectral_type").head(1)
    return {f"KIC {int(r.KIC)} ({r.spectral_type}-type, {int(r.Nf)} flares)": int(r.KIC)
            for r in picks.sort_values("Teff", ascending=False).itertuples()}


def render() -> None:
    st.title("MOCK - Stage 1: Flare detection (Sub-topic 1)")
    st.warning("**MOCK page** for testing Sub-topic 3 only. Replace with Sub-topic 1's real page.")

    verdict = st.radio("Stage 1 verdict (is_flare)", ["Flare", "Not a flare"], horizontal=True)
    prob = st.slider("Mock flare probability", 0.0, 1.0, 0.92 if verdict == "Flare" else 0.12)

    mode = st.radio("Star", ["Example star from dataset", "Any KIC", "Star not in dataset (manual)"])
    kic, stellar_params = None, None
    if mode == "Example star from dataset":
        examples = _example_kics()
        kic = examples[st.selectbox("Example", list(examples))]
    elif mode == "Any KIC":
        kic = int(st.number_input("KIC", min_value=0, value=10000000, step=1))
    else:
        st.caption("Optional parameters may stay blank; Sub-topic 3 will ask for them.")
        cols = st.columns(2)
        stellar_params = {
            name: cols[i % 2].number_input(PARAM_LABELS[name][0], value=None, key=f"mock1_{name}")
            for i, name in enumerate(STELLAR_PARAM_FIELDS)
        }

    payload = {
        "is_flare": verdict == "Flare",
        "KIC": kic,
        "stellar_params": stellar_params,
        "flare_probability": float(prob),
        "flare_details": {"source": "MOCK stage 1"},
    }
    validate_stage1_output(payload)
    with st.expander("Payload that will be saved (STAGE1 contract)"):
        st.json(payload)

    if st.button("Save Stage 1 result and continue  ->", type="primary"):
        get_pipeline()[STAGE1_KEY] = payload
        reset_stage3_state()  # a new Stage 1 result invalidates old Sub-topic 3 output
        st.switch_page(SUB3_PAGE)
