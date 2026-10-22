"""
Streamlit page for Sub-topic 3: stellar flare frequency & energy prediction.

Standalone: the user picks a star from the dataset by KIC (parameters are
auto-filled) or enters stellar parameters manually, then gets the predicted
flare rate and median flare energy with 80 % prediction intervals.

Integration with the other sub-topics is NOT part of this page; see
subtopic3/contracts.py and subtopic3/pipeline.py for the interface to use later.
"""
from __future__ import annotations

import json
import math

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from . import theme
from .contracts import STELLAR_PARAM_FIELDS
from .pipeline import load_catalog, lookup_star
from .predictor import FlarePredictor, InputError

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

MODE_DATASET = "Find a star by KIC (auto-fill from dataset)"
MODE_MANUAL = "Enter stellar parameters manually"


@st.cache_resource(show_spinner="Loading flare model ...")
def get_predictor() -> FlarePredictor:
    """Load the pickle once per server process."""
    return FlarePredictor()


# ---------------------------------------------------------------- helpers
def _is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def _interval_plot(result: dict):
    """Two small panels: flare rate (log axis) and log E, point + 80 % interval."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 1.9))
    fig.patch.set_alpha(0)  # transparent, so the page's space background shows through
    panels = [
        (axes[0], "pred_flare_rate", "Flare rate [flares/day]", True),
        (axes[1], "pred_logE", "Median flare energy log10(E/erg)", False),
    ]
    for ax, key, label, log in panels:
        lo, mid, hi = result[f"{key}_lower"], result[key], result[f"{key}_upper"]
        ax.set_facecolor((0.07, 0.08, 0.17, 0.6))
        ax.errorbar([mid], [0], xerr=[[mid - lo], [hi - mid]], fmt="o", color=theme.FLARE_YELLOW,
                    ecolor=theme.FLARE_ORANGE, elinewidth=6, capsize=6, ms=9,
                    markeredgecolor="white", markeredgewidth=1)
        if log:
            ax.set_xscale("log")
        pad = (hi - lo) * 0.25 if not log else 0
        ax.set_xlim((lo / 1.6, hi * 1.6) if log else (lo - pad, hi + pad))
        ax.set_yticks([])
        ax.set_xlabel(label, color=theme.STAR_WHITE)
        ax.tick_params(colors=theme.MUTED_TEXT, which="both")
        for spine in ax.spines.values():
            spine.set_color(theme.GRID_LINE)
        ax.grid(axis="x", color=theme.GRID_LINE, alpha=0.8)
    fig.tight_layout()
    return fig


def _choose_star() -> tuple[int | None, dict, str]:
    """Star selection. Returns (KIC or None, default parameters, widget-key suffix)."""
    mode = st.radio("Star", [MODE_DATASET, MODE_MANUAL], horizontal=True)
    if mode == MODE_MANUAL:
        st.info("Teff and log g are required; leave the other fields blank if unknown.")
        return None, {}, "manual"

    catalog = load_catalog()
    kics = sorted(int(k) for k in catalog.index)
    types = catalog["spectral_type"].to_dict()
    kic = st.selectbox(
        "KIC (type to search)", kics, index=None, placeholder="Choose or type a KIC ...",
        format_func=lambda k: f"KIC {k}  ({types[k]}-type)",
    )
    if kic is None:
        return None, {}, "none"
    st.success(f"KIC {kic} found in the dataset - parameters auto-filled (you can edit them).")
    return int(kic), lookup_star(int(kic)) or {}, str(kic)


def _parameter_form(defaults: dict, key_suffix: str) -> dict | None:
    """Pre-filled parameter form. Returns submitted params, or None if not submitted."""
    with st.form(f"s3_params_{key_suffix}"):
        cols = st.columns(2)
        values = {}
        for i, name in enumerate(STELLAR_PARAM_FIELDS):
            label, help_text = PARAM_LABELS[name]
            default = defaults.get(name)
            values[name] = cols[i % 2].number_input(
                label, value=None if _is_missing(default) else float(default),
                help=help_text, format="%.0f" if name == "Teff" else "%.4f",
                key=f"s3_{name}_{key_suffix}",  # new star -> fresh widgets with new defaults
            )
        submitted = st.form_submit_button("Predict flare frequency & energy", type="primary")
    return values if submitted else None


def _show_result(result: dict, warnings: list[str], observed: dict | None) -> None:
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
    if observed:  # dataset star: show the catalogue values next to the prediction
        table["Observed (Kepler)"] = [observed["flares_per_day"], 365.25 * observed["flares_per_day"],
                                      observed["median_logE"]]
    st.dataframe(table.style.format({c: "{:.4g}" for c in table.columns if c != "Quantity"}),
                 hide_index=True, width="stretch")
    st.pyplot(_interval_plot(result), clear_figure=True)

    if result["detection_bias_flag"]:
        st.warning(BIAS_EXPLANATION)
    else:
        st.caption("Detection-bias flag: not set (only F-type stars are flagged).")
    for w in warnings:
        st.caption(f"Note: {w}")

    with st.expander("Output record (JSON)"):
        st.code(json.dumps(result, indent=2), language="json")
    st.download_button("Download result (JSON)", json.dumps(result, indent=2),
                       file_name=f"subtopic3_{result['KIC'] or 'manual'}.json", mime="application/json")


def _single_star_tab() -> None:
    kic, defaults, key_suffix = _choose_star()
    if key_suffix == "none":
        return
    params = _parameter_form(defaults, key_suffix)
    if params is not None:
        try:
            result, warnings = get_predictor().predict_one(params, kic=kic)
            st.session_state["s3_result"] = {"key": key_suffix, "result": result, "warnings": warnings}
        except InputError as exc:
            st.session_state.pop("s3_result", None)
            st.error(f"Cannot predict: {exc}")

    saved = st.session_state.get("s3_result")
    if saved and saved["key"] == key_suffix:  # only show the result for the selected star
        observed = None
        if kic is not None:
            row = load_catalog().loc[kic]
            observed = {"flares_per_day": float(row["flares_per_day"]), "median_logE": float(row["median_logE"])}
        _show_result(saved["result"], saved["warnings"], observed)


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
def render() -> None:
    theme.apply_space_theme()
    theme.hero(
        "Stellar flare frequency &amp; energy prediction",
        "Predicts how often a star flares and how energetic its typical flare is, "
        "with an 80 % conformal prediction interval. Covers F, G, K and M dwarfs.",
    )
    single, batch = st.tabs(["Single star", "Batch CSV"])
    with single:
        _single_star_tab()
    with batch:
        _batch_tab()
