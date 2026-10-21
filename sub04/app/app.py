"""Streamlit page for Sub-topic 4: habitability risk scoring.

Manual input form, a star catalogue lookup or a CSV upload, with the risk
result, the formula label and the SHAP drivers.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make sub04/ importable

import pandas as pd
import streamlit as st  # noqa: E402

from risk_scoring import catalogue, config, csv_input, explain, predictor  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))  # make theme.py importable
from theme import apply_theme  # noqa: E402

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

# The three input sources. Manual, Catalogue and Previous outputs all work today.
SOURCE_KEY = "s4_source"
SOURCE_OPTIONS = ["Manual", "Catalogue", "Previous outputs"]
MANUAL_SOURCE = "Manual"
CATALOGUE_SOURCE = "Catalogue"
CSV_SOURCE = "Previous outputs"

# The star catalogue: the held-out test stars of the training data.
CATALOGUE_CAPTION = "Demo catalogue: stars the model was not trained on."
CATALOGUE_NOT_FOUND = "Star not in the demo catalogue (test stars only)."
CATALOGUE_EDITED = "Values edited, so the catalogue label no longer applies"
CATALOGUE_HINT = "Type a star ID and press Load star. The catalogue holds the test stars of the training data."
CATALOGUE_ENTRIES_KEY = "s4_catalogue_entries"
CATALOGUE_VALUES_KEY = "s4_catalogue_values"
CATALOGUE_LABEL_KEY = "s4_catalogue_label"
CATALOGUE_MARKER_KEY = "s4_catalogue_marker"
CATALOGUE_PLANET_PREFIX = "s4_catalogue_planet_"

# The CSV upload. The file uploader keeps its state under a key with the token
# in it, because a file uploader cannot be cleared through the session state.
UPLOAD_KEY = "s4_csv_upload"
UPLOAD_TOKEN_KEY = "s4_csv_upload_token"
CSV_ROWS_KEY = "s4_csv_rows"
CSV_NOTES_KEY = "s4_csv_notes"
CSV_SELECT_KEY = "s4_csv_row"
RESULTS_FILE_NAME = "subtopic4_results.csv"
CSV_NOTE = "The file is read to score the rows. Nothing is saved and no row is added to the training data."

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
    """Clear the six input fields, the stored result, the catalogue state and the uploaded CSV."""
    for name in INPUT_FIELDS:
        st.session_state[name] = ""
    st.session_state.pop(RESULT_KEY, None)
    st.session_state.pop(CSV_ROWS_KEY, None)
    st.session_state.pop(CSV_NOTES_KEY, None)
    _clear_catalogue_state()
    st.session_state["future_star_id"] = ""
    _clear_upload()


def _clear_catalogue_state():
    """Drop the catalogue state, so a lookup that fails leaves nothing behind."""
    for name in INPUT_FIELDS:
        st.session_state[name] = ""
    st.session_state.pop(RESULT_KEY, None)
    st.session_state.pop(CATALOGUE_ENTRIES_KEY, None)
    st.session_state.pop(CATALOGUE_VALUES_KEY, None)
    st.session_state.pop(CATALOGUE_LABEL_KEY, None)
    st.session_state.pop(CATALOGUE_MARKER_KEY, None)
    for key in list(st.session_state.keys()):
        if key.startswith(CATALOGUE_PLANET_PREFIX):
            st.session_state.pop(key, None)


def _clear_upload():
    """Clear the uploaded file by giving the file uploader a new key.

    The state of a file uploader is read-only, so the key is the only way to
    empty the widget. The token in the key is raised here and the old key is
    dropped from the session state.
    """
    token = st.session_state.get(UPLOAD_TOKEN_KEY, 0)
    st.session_state.pop("{0}_{1}".format(UPLOAD_KEY, token), None)
    st.session_state[UPLOAD_TOKEN_KEY] = token + 1


def _upload_key():
    """Return the key of the file uploader for the current upload token."""
    return "{0}_{1}".format(UPLOAD_KEY, st.session_state.get(UPLOAD_TOKEN_KEY, 0))


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


def render_limits_section():
    """Draw the plain-English notes about what the score means and where it comes from."""
    st.subheader("About")
    text = (
        "- **What the score is:** a physics-based proxy for how much flare and CME energy reaches a planet. "
        "It is not measured or simulated ozone loss.\n"
        "- **Formula label:** R_eff = flare rate x CME probability x (reference distance / orbital distance) squared, "
        "with a reference distance of {0} AU. High at or above {1}, Medium at or above {2}. "
        "The cut-offs were chosen by the project, not measured.\n"
        "- **ML label:** from a model trained on a dose label. The dose adds up every flare's energy weighted by its "
        "chance of a CME, spread over the planet's distance, per month observed. Classes are thirds of the training "
        "sample, so \"High\" means high among the planets in the sample.\n"
        "- **Model:** XGBoost on five inputs. On 235 held-out planets (no star in both training and test) its macro F1 "
        "is 0.85. The ML class and the formula label can differ; they use different methods.\n"
        "- **CME probability:** a modelled estimate from flare energy (Shi 2025 flares), not an observation. It is only "
        "used in the formula label. If left empty, the default from config (the Lu 2025 value) is used.\n"
        "- **Training data:** 1,185 planets from 862 stars, from published flare catalogues. Values outside the training "
        "range are flagged.\n"
        "- **SHAP bars** show what moved the model's answer, not a physical cause."
    ).format(config.D_REF, config.THRESH_HIGH, config.THRESH_MEDIUM)
    st.markdown(text)


def render_limits():
    """Draw the limits note at the bottom of the page."""
    st.markdown(
        "<p style='color:#b3261e;font-weight:700;'>{0} Classes are relative to the training sample.</p>".format(
            config.LIMITS_TEXT
        ),
        unsafe_allow_html=True,
    )


def render_source_control():
    """Draw the radio that picks where the inputs come from."""
    st.radio(
        "Input source",
        SOURCE_OPTIONS,
        index=0,
        key=SOURCE_KEY,
        horizontal=True,
    )


def render_star_id_lookup(enabled=False):
    """Draw the star ID lookup controls. They work only in the Catalogue mode."""
    st.markdown("**Star ID lookup**")

    id_col, load_col = st.columns([2, 1])
    with id_col:
        st.text_input("Star ID (TIC or KIC)", key="future_star_id", disabled=not enabled)
    with load_col:
        clicked = st.button("Load star", key="future_load", disabled=not enabled)

    st.divider()
    return clicked


def render_inputs():
    """Draw the source control, the star ID lookup and the inputs of that source."""
    st.subheader("Input")
    render_source_control()

    source = st.session_state.get(SOURCE_KEY, MANUAL_SOURCE)
    clicked = render_star_id_lookup(source == CATALOGUE_SOURCE)
    if source == CSV_SOURCE:
        render_csv_upload()
        st.button("Reset", on_click=_reset_inputs, key="s4_reset")
        return
    if source == CATALOGUE_SOURCE:
        render_catalogue(clicked)
        return
    render_manual_inputs()


def render_manual_inputs():
    """Draw the six text fields and the two buttons, and score the planet when asked."""
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


def render_catalogue(clicked=False):
    """Draw the catalogue lookup: the star ID, the planet picker and the filled inputs."""
    st.caption(CATALOGUE_CAPTION)

    entries = st.session_state.get(CATALOGUE_ENTRIES_KEY) or []
    if clicked:
        entries = _catalogue_entries()
        st.session_state[CATALOGUE_ENTRIES_KEY] = entries
        st.session_state.pop(CATALOGUE_MARKER_KEY, None)

    if not entries:
        if clicked:
            _clear_catalogue_state()
            st.warning(CATALOGUE_NOT_FOUND)
        else:
            st.info(CATALOGUE_HINT)
        st.button("Reset", on_click=_reset_inputs, key="s4_reset")
        return

    entry = _pick_planet(entries)
    _fill_catalogue_inputs(entry, force=clicked)
    render_catalogue_inputs()


def _catalogue_entries():
    """Return the catalogue entries of the star in the star ID field. Empty when there is none."""
    star_id = st.session_state.get("future_star_id", "")
    return catalogue.find_star(catalogue.load_catalogue(), star_id)


def _pick_planet(entries):
    """Draw the planet picker when the star has more than one planet, and return the chosen one."""
    if len(entries) == 1:
        return entries[0]
    labels = [catalogue.planet_label(entry) for entry in entries]
    key = CATALOGUE_PLANET_PREFIX + entries[0].star_id
    choice = st.selectbox("Planet", labels, key=key)
    for entry, label in zip(entries, labels):
        if label == choice:
            return entry
    return entries[0]


def _fill_catalogue_inputs(entry, force=False):
    """Put the six values of the entry into the input fields, once per selected planet.

    The values are not filled again while the same planet stays selected, so a
    value that was edited by hand is not thrown away by the next rerun.
    """
    marker = (entry.star_id, entry.planet)
    if not force and st.session_state.get(CATALOGUE_MARKER_KEY) == marker:
        return
    values = catalogue.input_text(entry)
    for name in INPUT_FIELDS:
        st.session_state[name] = values.get(name, "")
    st.session_state[CATALOGUE_VALUES_KEY] = values
    st.session_state[CATALOGUE_LABEL_KEY] = entry.dose_class
    st.session_state[CATALOGUE_MARKER_KEY] = marker


def render_catalogue_inputs():
    """Draw the six filled input fields and the two buttons."""
    for name in INPUT_FIELDS:
        st.text_input(FIELD_LABELS[name], key=name, help=_range_help(name))

    score_col, reset_col = st.columns(2)
    with score_col:
        score = st.button("Score planet", type="primary", key="s4_score")
    with reset_col:
        st.button("Reset", on_click=_reset_inputs, key="s4_reset")

    if score:
        raw = {name: st.session_state.get(name, "") for name in INPUT_FIELDS}
        st.session_state[RESULT_KEY] = predictor.predict(raw)


def render_csv_upload():
    """Draw the CSV uploader, the accepted column names and the error for a file that cannot be used."""
    upload = st.file_uploader("Results CSV", type="csv", key=_upload_key())
    st.caption("Accepted columns: " + csv_input.accepted_columns_text())
    st.caption(CSV_NOTE)

    if upload is None:
        _clear_csv_state()
        return
    try:
        frame = pd.read_csv(upload)
    except Exception as exc:  # noqa: BLE001 - show any parse problem to the user
        st.error("Could not read the CSV: {0}".format(exc))
        _clear_csv_state()
        return

    prepared, notes = csv_input.prepare(frame)
    missing = csv_input.missing_columns(prepared)
    if missing:
        st.error(
            "Missing column(s) in the CSV: {0}. Accepted column names: {1}".format(
                ", ".join(missing), csv_input.accepted_columns_text()
            )
        )
        _clear_csv_state()
        return

    st.session_state[CSV_ROWS_KEY] = csv_input.score_rows(prepared)
    st.session_state[CSV_NOTES_KEY] = notes


def _clear_csv_state():
    """Drop the scored rows, so an unusable file never leaves results behind."""
    st.session_state.pop(CSV_ROWS_KEY, None)
    st.session_state.pop(CSV_NOTES_KEY, None)


def render_result():
    """Draw the input errors or the full risk result."""
    st.subheader("Result")
    source = st.session_state.get(SOURCE_KEY, MANUAL_SOURCE)
    if source == CSV_SOURCE:
        render_csv_results()
        return
    if source == CATALOGUE_SOURCE:
        render_catalogue_result()
        return
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


def render_catalogue_result():
    """Draw the result of the loaded star and the catalogue label for comparison."""
    if not (st.session_state.get(CATALOGUE_ENTRIES_KEY) or []):
        st.info("Load a star from the demo catalogue to see its result.")
        return

    raw = {name: st.session_state.get(name, "") for name in INPUT_FIELDS}
    result = predictor.predict(raw)
    st.session_state[RESULT_KEY] = result

    if not result.ok:
        for message in result.errors:
            st.error(message)
        return

    render_badges(result)
    render_probabilities(result)
    render_drivers(result)
    render_context(result)
    render_flags(result)

    st.divider()
    if _catalogue_label_applies():
        st.markdown("Catalogue label (for comparison): {0}".format(st.session_state.get(CATALOGUE_LABEL_KEY, "")))
        st.caption(
            "The model was not trained on this star. The catalogue label is the dose class stored in "
            "the dataset for this planet; it was not used to score the planet."
        )
    else:
        st.markdown(CATALOGUE_EDITED)
        st.caption(
            "The values in the fields are not the values of the catalogue row any more, so the catalogue "
            "label does not describe this result."
        )


def _catalogue_label_applies():
    """Return True when the values in the fields are still the values of the catalogue row."""
    loaded = st.session_state.get(CATALOGUE_VALUES_KEY) or {}
    if not loaded:
        return False
    current = {name: st.session_state.get(name, "") for name in INPUT_FIELDS}
    return catalogue.values_match(current, loaded)


def render_csv_results():
    """Draw the results table, the row picker and the full result of the picked row."""
    rows = st.session_state.get(CSV_ROWS_KEY) or []
    if not rows:
        st.info("Upload a CSV to score several planets at once.")
        return

    table = csv_input.results_table(rows)
    notes = st.session_state.get(CSV_NOTES_KEY) or []
    if notes:
        st.caption("Columns converted: " + "; ".join(notes))
    st.dataframe(table, hide_index=True, width="stretch")
    st.download_button(
        "Download results CSV",
        table.to_csv(index=False),
        file_name=RESULTS_FILE_NAME,
        mime="text/csv",
    )

    labels = [_row_label(row) for row in rows]
    select_key = "{0}_{1}".format(CSV_SELECT_KEY, st.session_state.get(UPLOAD_TOKEN_KEY, 0))
    choice = st.selectbox("Row", labels, key=select_key)
    row = rows[labels.index(choice)] if choice in labels else rows[0]
    if row.result is None:
        st.error("Not scored: " + row.reason)
        return
    render_badges(row.result)
    render_probabilities(row.result)
    render_drivers(row.result)
    render_context(row.result)
    render_flags(row.result)


def _row_label(row):
    """Return the label of one row in the row picker."""
    if row.star_id:
        return "Row {0} - {1}".format(row.number, row.star_id)
    return "Row {0}".format(row.number)


def main():
    """Draw the whole page."""
    st.set_page_config(page_title="Sub-topic 4 - Habitability risk", page_icon="🪐", layout="wide")
    apply_theme()
    st.title("Habitability Risk Scoring")
    st.caption("Sub-topic 4 | AI-Driven Stellar Flare Analysis and Exoplanetary Habitability Impact Assessment")

    left, right = st.columns([1, 2])
    with left:
        render_inputs()
    with right:
        render_result()

    render_limits_section()
    render_limits()


main()