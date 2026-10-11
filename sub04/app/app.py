"""Streamlit page for Sub-topic 4: habitability risk scoring.

Manual input form with the risk result, the formula label and the SHAP drivers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make sub04/ importable

import pandas as pd
import streamlit as st  # noqa: E402

from risk_scoring import config, explain, predictor  # noqa: E402

INPUT_FIELDS = ["koi_sma", "flare_rate_month", "flare_rate_all", "max_logE", "Teff", "cme_prob"]

FIELD_LABELS = {
    "koi_sma": "Orbital distance (AU)",
    "flare_rate_month": "Flares per month at 1e34 erg or more",
    "flare_rate_all": "All flares per month",
    "max_logE": "Strongest flare, log10 erg",
    "Teff": "Star temperature, Teff (K)",
    "cme_prob": "CME probability (optional)",
}

CME_NOTE = "Leave empty to use the default (taken from config). Used in the formula label only."
RESULT_KEY = "s4_result"

# One badge per class: (text, background colour, text colour)
BADGE_STYLES = {
    "High": ("HIGH RISK", "#b3261e", "#ffffff"),
    "Medium": ("MEDIUM RISK", "#f0b429", "#1a1a1a"),
    "Low": ("LOW RISK", "#2e7d32", "#ffffff"),
}


def _range_help(name):
    """Return the training range of one field as a help text."""
    low, high = config.TRAIN_RANGES[name]
    return "Training range: {0} to {1}".format("{0:.4g}".format(low), "{0:.4g}".format(high))


def _reset_inputs():
    """Clear the six input fields and the stored result."""
    for name in INPUT_FIELDS:
        st.session_state[name] = ""
    st.session_state.pop(RESULT_KEY, None)


def _star_type_line(result):
    """Return the star type line of a result."""
    return "Star type: {0} dwarf (type {1})".format(result.star_type_letter, result.star_type)


def render_badges(result):
    """Draw the risk badge and the formula label box."""
    badge_text, background, text_color = BADGE_STYLES[result.ml_class]
    badge = (
        "<span style='display:inline-block;padding:8px 16px;border-radius:8px;background:{0};color:{1};"
        "font-weight:700;letter-spacing:1px;border:1px solid rgba(0,0,0,0.35);'>{2}</span>"
    ).format(background, text_color, badge_text)
    box = (
        "<span style='display:inline-block;padding:8px 16px;border-radius:8px;border:1px solid rgba(128,128,128,0.7);"
        "color:inherit;font-weight:600;'>Formula label: {0} (R_eff = {1})</span>"
    ).format(result.formula.label, "{0:.3g}".format(result.formula.r_eff))
    st.markdown(
        "<div style='display:flex;gap:12px;flex-wrap:wrap;align-items:center;'>{0}{1}</div>".format(badge, box),
        unsafe_allow_html=True,
    )
    st.caption("The ML class comes from the dose-label model. The formula label is calculated directly.")


def render_probabilities(result):
    """Draw one progress bar per class, from High to Low."""
    st.subheader("Class probabilities")
    for name in reversed(config.CLASSES):
        probability = result.probabilities.get(name, 0.0)
        st.progress(probability, text="{0} {1}%".format(name, int(round(probability * 100))))


def render_drivers(result):
    """Draw the SHAP bar chart of the top drivers and the explanation sentence."""
    st.subheader("Why this label (SHAP, top drivers)")
    explanation = explain.explain(result.values, result.ml_class)
    contributions = pd.Series({driver.name: driver.contribution for driver in explanation.drivers})
    st.bar_chart(contributions, horizontal=True)
    st.caption(explanation.sentence)
    st.caption(
        "Bars to the right pushed the result toward {0}, bars to the left pushed it away. "
        "This shows what moved the model, not a physical cause.".format(result.ml_class)
    )


def render_context(result):
    """Draw the star type and the orbital distance next to the reference distance."""
    st.subheader("Star context")
    distance = result.values["koi_sma"]
    side = "closer" if distance < config.D_REF else "farther"
    st.markdown(
        "{0} dwarf (type {1}). Orbital distance {2} AU is {3} than the {4} AU reference.".format(
            result.star_type_letter, result.star_type, distance, side, config.D_REF
        )
    )


def render_flags(result):
    """Draw every flag as a small caption."""
    for flag in result.flags:
        st.caption("- " + flag)


def render_limits():
    """Draw the limits note at the bottom of the page."""
    st.markdown(
        "<p style='color:#b3261e;font-weight:700;'>{0} Classes are relative to the training sample.</p>".format(
            config.LIMITS_TEXT
        ),
        unsafe_allow_html=True,
    )


def render_inputs():
    """Draw the six text fields and the two buttons, and score the planet when asked."""
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
        st.session_state[RESULT_KEY] = predictor.predict(raw)


def render_result():
    """Draw the input errors or the full risk result."""
    st.subheader("Result")
    result = st.session_state.get(RESULT_KEY)
    if result is None:
        st.info("Enter the inputs and press Score planet.")
        return
    if not result.ok:
        for message in result.errors:
            st.error(message)
        if result.star_type_letter:
            st.markdown(_star_type_line(result))
        return
    render_badges(result)
    render_probabilities(result)
    render_drivers(result)
    render_context(result)
    render_flags(result)


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

    render_limits()


main()