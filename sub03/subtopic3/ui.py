"""
Streamlit page for Sub-topic 3 (pipeline stage 2): flare frequency & energy.

Call ``render()`` from any Streamlit page. It reads Stage 1's result from
st.session_state["pipeline"]["stage1"] and, on "Continue", writes the
Sub-topic 3 result to st.session_state["pipeline"]["stage3"].
"""
from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from .contracts import PIPELINE_KEY, STAGE1_KEY, STAGE3_KEY, STELLAR_PARAM_FIELDS
from .pipeline import GatingError, check_gate, lookup_star, run_stage3
from .predictor import FlarePredictor, InputError

# Page opened by the "Continue" button (path relative to the app's main script).
# The integrated app should point this at Sub-topic 2's real CME page.
DEFAULT_NEXT_PAGE = "pages/3_MOCK_CME_Prediction.py"

# Labels/units for the parameter form: (label, help text)
PARAM_LABELS = {
    "Teff": ("Effective temperature Teff [K]", "Required. F/G/K/M only: 2600 K <= Teff < 7500 K."),
    "logg": ("Surface gravity log g [cgs]", "Required. Training sample: dwarfs (log g >= 4)."),
    "Prot": ("Rotation period Prot [days]", "Optional - leave blank if unknown."),
    "radius_gspphot": ("Gaia radius [R_sun]", "Optional (Gaia GSP-Phot)."),
    "mh_gspphot": ("Gaia metallicity [M/H]", "Optional (Gaia GSP-Phot)."),
    "bp_rp": ("Gaia BP-RP colour [mag]", "Optional."),
    "parallax": ("Gaia parallax [mas]", "Optional."),
    "phot_g_mean_mag": ("Gaia G magnitude [mag]", "Optional."),
}

BIAS_EXPLANATION = (
    "**Detection-bias flag (F-type star).** Kepler's target selection favoured "
    "Sun-like stars, and flares are harder to detect on bright, hot F stars, so only "
    "the strongest/most frequent flarers enter the catalogue. F-type rates and "
    "energies therefore look inflated compared with M dwarfs. This is an "
    "**observational artefact, not a physical result** - interpret with caution."
)


@st.cache_resource(show_spinner="Loading flare model ...")
def get_predictor() -> FlarePredictor:
    """Load the pickle once per server process."""
    return FlarePredictor()


def get_pipeline() -> dict:
    """The shared pipeline-state dict (created if missing)."""
    return st.session_state.setdefault(PIPELINE_KEY, {})


# ---------------------------------------------------------------- helpers
def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _interval_plot(result: dict):
    """Two small panels: flare rate (log axis) and log E, point + 80 % interval."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 1.9))
    panels = [
        (axes[0], "pred_flare_rate", "Flare rate [flares/day]", True),
        (axes[1], "pred_logE", "Median flare energy log10(E/erg)", False),
    ]
    for ax, key, label, log in panels:
        lo, mid, hi = result[f"{key}_lower"], result[key], result[f"{key}_upper"]
        ax.errorbar([mid], [0], xerr=[[mid - lo], [hi - mid]], fmt="o", color="#c0392b",
                    ecolor="#e67e22", elinewidth=6, capsize=6, ms=8)
        if log:
            ax.set_xscale("log")
        pad = (hi - lo) * 0.25 if not log else 0
        ax.set_xlim((lo / 1.6, hi * 1.6) if log else (lo - pad, hi + pad))
        ax.set_yticks([])
        ax.set_xlabel(label)
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def _show_stage1_context(stage1: dict) -> None:
    with st.expander("Received from Stage 1 (flare detection)", expanded=False):
        st.json(stage1)


def _parameter_form(stage1: dict) -> dict | None:
    """Pre-filled parameter form. Returns submitted params, or None if not submitted."""
    kic = stage1.get("KIC")
    from_dataset = lookup_star(kic)
    if from_dataset is not None:
        defaults = from_dataset
        st.success(f"KIC {kic} found in the dataset - parameters auto-filled (you can edit them).")
    else:
        defaults = dict(stage1.get("stellar_params") or {})
        if kic is not None:
            st.info(f"KIC {kic} is not in the dataset - please enter / check the parameters.")
        else:
            st.info("No KIC provided - please enter the stellar parameters manually.")

    with st.form("s3_params"):
        cols = st.columns(2)
        values = {}
        for i, name in enumerate(STELLAR_PARAM_FIELDS):
            label, help_text = PARAM_LABELS[name]
            default = defaults.get(name)
            values[name] = cols[i % 2].number_input(
                label, value=None if _is_missing(default) else float(default),
                help=help_text, format="%.4f" if name not in ("Teff",) else "%.0f",
                key=f"s3_{name}_{kic}",  # new KIC -> fresh widgets with new defaults
            )
        submitted = st.form_submit_button("Predict flare frequency & energy", type="primary")
    return values if submitted else None


def _show_result(result: dict, warnings: list[str]) -> None:
    level = int(round(result.get("interval_level", 0.8) * 100))
    st.subheader(f"Predictions (point estimate with {level} % interval)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Spectral type", result["spectral_type"])
    c2.metric("Flare rate [flares/day]", f"{result['pred_flare_rate']:.4f}",
              help=f"{level} % interval: {result['pred_flare_rate_lower']:.4f} - "
                   f"{result['pred_flare_rate_upper']:.4f}")
    c3.metric("Median flare energy log10(E/erg)", f"{result['pred_logE']:.2f}",
              help=f"{level} % interval: {result['pred_logE_lower']:.2f} - {result['pred_logE_upper']:.2f}")

    table = pd.DataFrame({
        "Quantity": ["Flare rate [flares/day]", "Flare rate [flares/year]", "Median log10 E [erg]"],
        "Lower": [result["pred_flare_rate_lower"], 365.25 * result["pred_flare_rate_lower"], result["pred_logE_lower"]],
        "Point": [result["pred_flare_rate"], 365.25 * result["pred_flare_rate"], result["pred_logE"]],
        "Upper": [result["pred_flare_rate_upper"], 365.25 * result["pred_flare_rate_upper"], result["pred_logE_upper"]],
    })
    st.dataframe(table.style.format({"Lower": "{:.4g}", "Point": "{:.4g}", "Upper": "{:.4g}"}),
                 hide_index=True, width="stretch")
    st.pyplot(_interval_plot(result), clear_figure=True)

    if result["detection_bias_flag"]:
        st.warning(BIAS_EXPLANATION)
    else:
        st.caption("Detection-bias flag: not set (only F-type stars are flagged).")
    for w in warnings:
        st.caption(f"Note: {w}")

    with st.expander("Output passed to the next stages (contract fields)"):
        st.code(json.dumps(result, indent=2), language="json")


def _single_star_tab(stage1: dict, next_page: str) -> None:
    pipeline = get_pipeline()
    params = _parameter_form(stage1)
    if params is not None:
        try:
            result, warnings = run_stage3(stage1, get_predictor(), params=params)
            st.session_state["s3_pending"] = {"result": result, "warnings": warnings}
        except InputError as exc:
            st.session_state.pop("s3_pending", None)
            st.error(f"Cannot predict: {exc}")

    pending = st.session_state.get("s3_pending")
    if not pending:
        return
    _show_result(pending["result"], pending["warnings"])

    st.divider()
    if st.button("Continue to CME prediction  ->", type="primary"):
        pipeline[STAGE3_KEY] = pending["result"]  # hand-off to Sub-topics 2 and 4
        st.switch_page(next_page)


def _batch_tab() -> None:
    st.markdown(
        "Upload a CSV with columns **Teff, logg** (required) and optionally "
        f"**{', '.join(f for f in STELLAR_PARAM_FIELDS if f not in ('Teff', 'logg'))}, "
        "spectral_type, KIC**. Missing values are allowed (left blank)."
    )
    upload = st.file_uploader("Stellar parameters CSV", type="csv")
    if upload is None:
        return
    try:
        df = pd.read_csv(upload)
    except Exception as exc:  # noqa: BLE001 - show any parse problem to the user
        st.error(f"Could not read CSV: {exc}")
        return
    missing = [c for c in ("Teff", "logg") if c not in df.columns]
    if missing:
        st.error(f"Missing required column(s): {', '.join(missing)}")
        return
    with st.spinner(f"Predicting {len(df)} stars ..."):
        result = get_predictor().predict_frame(df)
    n_err = int(result["error"].notna().sum())
    st.write(f"{len(df) - n_err} predicted, {n_err} rejected (see 'error' column).")
    st.dataframe(result, width="stretch")
    st.download_button("Download predictions CSV", result.to_csv(index=False),
                       file_name="subtopic3_predictions.csv", mime="text/csv")


# ------------------------------------------------------------------ entry
def render(next_page: str = DEFAULT_NEXT_PAGE) -> None:
    st.title("Stage 2 - Flare frequency & energy (Sub-topic 3)")
    st.caption(
        "Predicts how often this star flares and how energetic its typical flare is, "
        "with an 80 % conformal prediction interval. Covers F, G, K and M dwarfs."
    )
    stage1 = get_pipeline().get(STAGE1_KEY)

    # ---- gating: never predict unless Stage 1 confirmed a flare
    try:
        check_gate(stage1)
    except GatingError as exc:
        st.error(str(exc))
        if stage1:
            _show_stage1_context(stage1)
        st.stop()

    _show_stage1_context(stage1)
    single, batch = st.tabs(["This star", "Batch CSV"])
    with single:
        _single_star_tab(stage1, next_page)
    with batch:
        _batch_tab()


def reset_stage3_state() -> None:
    """Clear Sub-topic 3 results; call this whenever Stage 1's result changes."""
    get_pipeline().pop(STAGE3_KEY, None)
    st.session_state.pop("s3_pending", None)
