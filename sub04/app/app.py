"""Streamlit page for Sub-topic 4: habitability risk scoring.

Manual input form only: the risk result, the formula label and the SHAP drivers
are added in the next steps.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make sub04/ importable

import streamlit as st  # noqa: E402

from risk_scoring import config, validation  # noqa: E402

INPUT_FIELDS = ["koi_sma", "flare_rate_month", "flare_rate_all", "max_logE", "Teff", "cme_prob"]

FIELD_LABELS = {
    "koi_sma": "Orbital distance (AU)",
    "flare_rate_month": "Flares per month at 1e34 erg or more",
    "flare_rate_all": "All flares per month",
    "max_logE": "Strongest flare, log10 erg",
    "Teff": "Star temperature, Teff (K)",
    "cme_prob": "CME probability (optional)",}

CME_NOTE = "Leave empty to use the default (taken from config). Used in the formula label only."
RESULT_KEY = "s4_result"


def _range_help(name):
    """Return the training range of one field as a help text."""
    low, high = config.TRAIN_RANGES[name]
    return "Training range: {0} to {1}".format("{0:.4g}".format(low), "{0:.4g}".format(high))


def _reset_inputs():
    """Clear the six input fields and the stored result."""
    for name in INPUT_FIELDS:
        st.session_state[name] = ""
    st.session_state.pop(RESULT_KEY, None)


def render_inputs():
    """Draw the six text fields and the two buttons, and validate when the planet is scored."""
    st.subheader("Input")
    for name in INPUT_FIELDS:
        st.text_input(FIELD_LABELS[name], key=name, help=_range_help(name))
        if name == "cme_prob":
            st.caption(CME_NOTE)

    score_col, reset_col = st.columns(2)
    with score_col:
        score = st.button("Score planet", type="primary", key="s4_score")
    with reset_col:
        st.button("Reset", on_click=_reset_inputs, key="s4_reset")

    if score:
        raw = {name: st.session_state.get(name, "") for name in INPUT_FIELDS}
        st.session_state[RESULT_KEY] = validation.validate_inputs(raw)


def render_result():
    """Draw the validation result: the errors, the star type and the flags."""
    st.subheader("Result")
    result = st.session_state.get(RESULT_KEY)
    if result is None:
        st.info("Enter the inputs and press Score planet.")
        return
    if result.errors:
        for message in result.errors:
            st.error(message)
        return
    code = result.values["star_type"]
    st.markdown("Star type: {0} dwarf (type {1})".format(validation.star_type_letter(code), code))
    for flag in result.flags:
        st.caption("- " + flag)
    st.caption("The risk result is added in the next step.")


def main():
    """Draw the whole page."""
    st.set_page_config(page_title="Habitability Risk Scoring", layout="wide")
    st.title("Habitability Risk Scoring")
    st.caption("Sub-topic 4 | AI-Driven Stellar Flare Analysis and Exoplanetary Habitability Impact Assessment")
    st.warning(config.LIMITS_TEXT)

    left, right = st.columns([1, 2])
    with left:
        render_inputs()
    with right:
        render_result()


main()
