"""Streamlit page for Stage 1: flare detection and energy classification.

Use it in any Streamlit app:

    from flare_cnn.ui import render
    render(next_page="pages/2_Flare_Frequency_Energy.py")   # the page to open next

After an analysis, "Save result and continue" writes the Stage-1 payload to
st.session_state["pipeline"]["stage1"] (sub03's contract) and opens next_page.
"""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from . import __version__
from .contracts import PIPELINE_KEY, STAGE1_KEY, STAGE3_KEY
from .energy import TIER_NAMES
from .pipeline import analyse, load_assets

TIER_COLOURS = {"C": "#2E9E5B", "M": "#E3B21B", "X": "#E8743B", "Superflare": "#C62828"}
RESULT_KEY = "c1_result"          # this page's own state
EXAMPLE = {"tic": "441420236", "sector": 27, "label": "Try AU Mic"}


# ---------------------------------------------------------------- setup
@st.cache_resource(show_spinner="Loading the model ...")
def _assets():
    return load_assets()


def get_pipeline() -> dict:
    """The shared pipeline dict in session_state (created if missing)."""
    return st.session_state.setdefault(PIPELINE_KEY, {})


def _reset_next_stage() -> None:
    """A new Stage-1 result makes any old Stage-2 result invalid."""
    try:                                   # use sub03's own reset when it is importable
        from subtopic3.ui import reset_stage3_state
        reset_stage3_state()
    except ImportError:
        get_pipeline().pop(STAGE3_KEY, None)


# ---------------------------------------------------------------- page
def render(next_page: str | None = None, analyzer=None) -> None:
    """Draw the whole page. analyzer is only for tests (see pipeline.analyse)."""
    st.title("🌟 Stage 1 · Flare Detection & Energy Classification")
    st.caption("Sub-topic 1 · Kariyawasam K.G.L. (IT23377608) · "
               "two-head 1D-CNN on TESS 2-minute light curves")
    st.info("Enter a TESS star. The model finds every flare and grades its energy tier "
            "(C / M / X / Superflare) in one pass. The result can then be passed on to Stage 2.")

    tab_one, tab_batch, tab_info = st.tabs(["Single star", "Batch CSV", "Model info"])
    with tab_one:
        _single_star(next_page, analyzer)
    with tab_batch:
        _batch(analyzer)
    with tab_info:
        _model_info()


def _single_star(next_page, analyzer) -> None:
    c1, c2, c3 = st.columns([2, 1, 1.4])
    tic = c1.text_input("TIC ID", value=st.session_state.get("c1_tic", ""),
                        placeholder="e.g. 441420236")
    sector = c2.number_input("Sector (optional)", min_value=0, step=1, value=0,
                             help="0 = first available 2-minute sector")
    teff = c3.number_input("Teff in K (optional)", min_value=0.0, step=10.0, value=0.0,
                           help="0 = take Teff from the TESS Input Catalog")
    b1, b2, _ = st.columns([1, 1, 3])
    run = b1.button("🔍 Analyse star", type="primary", width="stretch")
    if b2.button(EXAMPLE["label"], width="stretch"):
        tic, sector, run = EXAMPLE["tic"], EXAMPLE["sector"], True

    if run:
        if not str(tic).strip():
            st.error("Please enter a TIC ID.")
            return
        st.session_state["c1_tic"] = str(tic)
        try:
            with st.spinner("Downloading the light curve from NASA MAST and running the model ..."):
                out = analyse(_assets(), tic, sector=int(sector) or None,
                              teff=float(teff) or None, analyzer=analyzer)
            st.session_state[RESULT_KEY] = out
        except FileNotFoundError as e:
            st.error(f"{e}. Copy the three model files into sub01/models/.")
            return
        except Exception as e:                 # bad TIC, no TESS data, no internet ...
            st.error(f"Could not analyse TIC {tic}: {e}")
            return

    out = st.session_state.get(RESULT_KEY)
    if out:
        _show_result(out, next_page)


def _show_result(out: dict, next_page) -> None:
    r, s, flares = out["result"], out["summary"], out["flares"]
    star = r["star"]
    st.success(f"TIC {r['tic']}, sector {r['sector']} · star type **{star['type']}** · "
               f"Teff {star['teff']:.0f} K ({star['teff_source']}) · "
               f"{r['n_windows']} windows checked in {r['seconds']} s")
    for w in out["warnings"]:
        st.warning(w)
    if len(r["all_sectors"]) > 1:
        st.caption(f"Other 2-minute sectors for this star: {r['all_sectors']}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Flares found", s["n_flares"])
    m2.metric("Weak flares (SNR < 5)", s["n_weak"])
    m3.metric("Strongest tier", s["strongest_tier"] or "none")
    m4.metric("Catalogue flares recovered",
              f"{s['catalogue_recovered']} / {s['catalogue_flares']}" if s["catalogue_flares"] else "none listed")
    for col, tier in zip(st.columns(4), TIER_NAMES):
        col.metric("Superflares" if tier == "Superflare" else f"{tier} flares", s["n_" + tier])

    st.subheader("Light curve")
    st.plotly_chart(_light_curve_figure(r, flares), width="stretch")

    st.subheader("Detected flares")
    if len(flares):
        table = flares[["peak_time_btjd", "flare_prob", "energy_tier", "tier_conf",
                        "amplitude", "snr", "weak", "in_catalogue"]].copy()
        table["amplitude"] = (table["amplitude"] * 100).round(2)
        table = table.rename(columns={
            "peak_time_btjd": "Peak time (BTJD)", "flare_prob": "Flare probability",
            "energy_tier": "Energy tier", "tier_conf": "Tier confidence",
            "amplitude": "Peak brightening (%)", "snr": "SNR", "weak": "Weak",
            "in_catalogue": "In catalogue"})
        st.dataframe(table, hide_index=True, width="stretch")
    else:
        st.write("No flares above the detection threshold in this sector.")

    d1, d2, _ = st.columns([1, 1, 2])
    d1.download_button("⬇ Flares CSV", _csv(flares), f"c1_flares_TIC{r['tic']}_S{r['sector']}.csv",
                       "text/csv", width="stretch")
    d2.download_button("⬇ Star summary CSV", _csv(pd.DataFrame([s])),
                       f"c1_summary_TIC{r['tic']}_S{r['sector']}.csv", "text/csv",
                       width="stretch")

    with st.expander("Payload for Stage 2 (stage contract)"):
        st.json(out["payload"])

    if st.button("Save result and continue to Stage 2  →", type="primary"):
        get_pipeline()[STAGE1_KEY] = out["payload"]
        _reset_next_stage()
        if next_page:
            st.switch_page(next_page)
        else:
            st.success("Saved to st.session_state['pipeline']['stage1'].")


def _light_curve_figure(r: dict, flares: pd.DataFrame) -> go.Figure:
    t, f = np.asarray(r["time"]), np.asarray(r["flux"])
    if len(t) > 25000:                      # keep the chart light
        k = int(np.ceil(len(t) / 25000))
        t, f = t[::k], f[::k]
    fig = go.Figure(go.Scattergl(x=t, y=f, mode="lines", line=dict(width=1, color="#5B7FA6"),
                                 name="flux", hoverinfo="skip"))
    for tier in TIER_NAMES:
        sub = flares[flares["energy_tier"] == tier]
        if len(sub):
            y = np.interp(sub["peak_time_btjd"], r["time"], r["flux"])
            fig.add_trace(go.Scatter(
                x=sub["peak_time_btjd"], y=y, mode="markers", name=tier,
                marker=dict(size=10, color=TIER_COLOURS[tier], line=dict(width=1.5, color="white")),
                customdata=np.c_[sub["flare_prob"], sub["snr"]],
                hovertemplate="%{x:.4f} BTJD<br>prob %{customdata[0]:.2f} · SNR %{customdata[1]}"
                              f"<extra>{tier}</extra>"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white",
                      xaxis_title="Time (BTJD)", yaxis_title="Normalised flux",
                      legend=dict(orientation="h", y=1.08))
    return fig


def _batch(analyzer) -> None:
    st.write("Upload a CSV with a **tic** column (and optionally **sector**). "
             "Each star is analysed and the results can be downloaded.")
    up = st.file_uploader("Stars CSV", type="csv")
    if up is None or not st.button("Run batch", type="primary"):
        return
    stars = pd.read_csv(up)
    if "tic" not in stars.columns:
        st.error("The CSV needs a column called 'tic'.")
        return
    flares, summaries, failed = [], [], []
    bar = st.progress(0.0)
    for k, row in enumerate(stars.itertuples(index=False), 1):
        sector = getattr(row, "sector", None)
        sector = None if sector is None or pd.isna(sector) else int(sector)
        try:
            out = analyse(_assets(), row.tic, sector=sector, analyzer=analyzer)
            flares.append(out["flares"])
            summaries.append(out["summary"])
        except Exception as e:                 # one bad star must not stop the batch
            failed.append({"tic": row.tic, "sector": sector, "error": str(e)})
        bar.progress(k / len(stars), text=f"{k} / {len(stars)} stars")
    all_flares = pd.concat(flares, ignore_index=True) if flares else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    st.success(f"{len(summaries)} stars analysed, {len(failed)} failed.")
    st.dataframe(summary, hide_index=True, width="stretch")
    if failed:
        st.dataframe(pd.DataFrame(failed), hide_index=True)
    c1, c2 = st.columns(2)
    c1.download_button("⬇ c1_all_flares.csv", _csv(all_flares), "c1_all_flares.csv", "text/csv")
    c2.download_button("⬇ c1_star_summary.csv", _csv(summary), "c1_star_summary.csv", "text/csv")


def _model_info() -> None:
    try:
        a = _assets()
        threshold = a.threshold
        edges = a.cfg["energy"]["tier_edges_erg"]
    except Exception as e:
        st.error(str(e))
        return
    st.markdown(f"""
| | |
|---|---|
| Model | two-head 1D-CNN, final version v2 (SPLIT_W2, separate top block per head), code version {__version__} |
| Input | 128-point windows (≈ 4.3 h) of TESS 2-minute light curves, 2 channels + star info |
| Detection threshold | {threshold:.2f} (chosen on the validation set) |
| Energy tiers | C < {edges[0]:.0e} erg · M < {edges[1]:.0e} · X < {edges[2]:.0e} · Superflare ≥ {edges[2]:.0e} |
| Weak flare | SNR < 5 |
| Test results (unseen stars) | 2.2× stella's weak-flare recall at the same precision (Lin test) · energy-tier accuracy 78.6 % (Lin) and 79.9 % (Günther) · detection AP 0.887 (Lin), 0.987 (Günther) |
""")


def _csv(df: pd.DataFrame) -> bytes:
    buf = io.StringIO()
    pd.DataFrame(df).to_csv(buf, index=False)
    return ("﻿" + buf.getvalue()).encode("utf-8")      # BOM so Excel shows accents
