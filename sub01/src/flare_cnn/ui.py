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
    st.title("🌟 Flare Detection & Energy Classification")
    st.info("Enter a TESS star. The model finds every flare and grades its energy tier "
            "(C / M / X / Superflare) in one pass.")

    tab_one, tab_batch, tab_info = st.tabs(["Single star", "Batch CSV", "Model info"])
    with tab_one:
        _new_input_form()
    with tab_batch:
        _new_batch_tab()
    with tab_info:
        _model_info()


def _single_star(next_page, analyzer) -> None:
    c1, c2, c3 = st.columns([2, 1, 1.4])
    tic = c1.text_input("TIC ID", value=st.session_state.get("c1_tic", ""),
                        placeholder="e.g. 441420236")
    sector = c2.number_input("Sector", min_value=0, step=1, value=0,
                             help="0 = first available 2-minute sector")
    teff = c3.number_input("Teff in K (optional)", min_value=0.0, step=10.0, value=None,
                           placeholder="e.g. 5700",
                           help="Leave blank to take Teff from the TESS Input Catalog")
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
                              teff=float(teff) if teff else None, analyzer=analyzer)
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
                        "amplitude_pct", "snr", "weak", "in_catalogue"]].copy()
        table = table.rename(columns={
            "peak_time_btjd": "Peak time (BTJD)", "flare_prob": "Flare probability",
            "energy_tier": "Energy tier", "tier_conf": "Tier confidence",
            "amplitude_pct": "Peak brightening (%)", "snr": "SNR", "weak": "Weak",
            "in_catalogue": "In catalogue"})
        st.dataframe(table, hide_index=True, width="stretch")
    else:
        st.write("No flares above the detection threshold in this sector.")

    d1, _ = st.columns([1, 3])
    d1.download_button("⬇ Flares CSV", _csv(flares), f"c1_flares_TIC{r['tic']}_S{r['sector']}.csv",
                       "text/csv", width="stretch")


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
    st.write("Upload a CSV with a **tic** column and **sector**. "
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
    st.success(f"{len(summaries)} stars analysed, {len(failed)} failed.")
    if failed:
        st.dataframe(pd.DataFrame(failed), hide_index=True)
    st.dataframe(all_flares, hide_index=True, width="stretch")
    st.download_button("⬇ c1_all_flares.csv", _csv(all_flares), "c1_all_flares.csv", "text/csv")


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


# ================================================================== new input form (prototype)
# Builds and validates inputs only - it never loads the model or runs an analysis.
# On success it writes st.session_state["c1_inputs"] for the next step to pick up.

NEW_INPUT_KEY = "c1_inputs"
CADENCE_TARGET_DAYS = 2.0 / 1440.0   # 2-minute SPOC cadence, same number as configs/config.yaml
MIN_POINTS = 128                      # same window length as configs/config.yaml


def _new_input_form() -> None:
    st.caption("Prototype of the new input form: builds and checks the inputs only. "
               "Nothing here runs the model yet.")

    st.subheader("Light curve")
    st.caption("One value per line (or comma-separated). The three lists must have the same length.")
    if st.button("Use synthetic demo data instead", key="c1_new_demo"):
        tt, ff, ee = _synthetic_light_curve_columns()
        st.session_state["c1_new_time"] = tt
        st.session_state["c1_new_flux"] = ff
        st.session_state["c1_new_flux_err"] = ee

    lc1, lc2, lc3 = st.columns(3)
    time_text = lc1.text_area("time (BTJD days)", height=70, key="c1_new_time",
                              placeholder="1500.000000\n1500.001389\n...")
    flux_text = lc2.text_area("flux", height=70, key="c1_new_flux",
                              placeholder="1.000200\n0.999800\n...")
    err_text = lc3.text_area("flux_err", height=70, key="c1_new_flux_err",
                             placeholder="0.000500\n0.000500\n...")

    st.subheader("Star parameters")
    s1, s2, s3 = st.columns(3)
    teff = s1.number_input("Teff (K)", min_value=0.0, step=10.0, value=None, key="c1_new_teff_box")
    radius = s2.number_input("Radius (R_sun)", min_value=0.0, step=0.01, value=None, key="c1_new_radius_box")
    distance = s3.number_input("Distance (pc)", min_value=0.0, step=1.0, value=None, key="c1_new_distance_box")

    if not (time_text.strip() or flux_text.strip() or err_text.strip()):
        return

    df, parse_err = _new_parse_columns(time_text, flux_text, err_text)
    if parse_err:
        st.error(parse_err)
        return

    clean_df, n_removed = _new_clean(df)
    errors, warnings_ = _new_validate(df, clean_df, n_removed, teff, radius, distance)
    for w in warnings_:
        st.warning(w)
    if errors:
        for e in errors:
            st.error(e)
        return

    st.success("Inputs look good.")
    sources = {"teff": "user", "radius": "user", "distance": "user"}
    _new_preview(clean_df, n_removed, teff, radius, distance, sources, {"source": "paste"})


def _new_parse_numbers(text: str, name: str):
    """One number per line (or comma-separated). Returns (list_of_floats, error_message)."""
    parts = [p.strip() for p in text.replace(",", "\n").splitlines() if p.strip()]
    values = []
    for i, p in enumerate(parts, 1):
        try:
            values.append(float(p))
        except ValueError:
            return None, f"{name} value {i} is not numeric: '{p}'"
    return values, None


def _new_parse_columns(time_text: str, flux_text: str, err_text: str):
    """Parse the three separate time/flux/flux_err lists. Returns (DataFrame, error_message)."""
    t, err = _new_parse_numbers(time_text, "time")
    if err:
        return None, err
    f, err = _new_parse_numbers(flux_text, "flux")
    if err:
        return None, err
    fe, err = _new_parse_numbers(err_text, "flux_err")
    if err:
        return None, err
    if not (len(t) == len(f) == len(fe)):
        return None, (f"time ({len(t)}), flux ({len(f)}) and flux_err ({len(fe)}) "
                      "must all have the same number of values.")
    if not t:
        return None, "No values entered."
    return pd.DataFrame({"time": t, "flux": f, "flux_err": fe}), None


BATCH_INPUT_KEY = "c1_inputs_batch"
BATCH_COLUMNS = ["star_id", "time", "flux", "flux_err", "teff", "radius", "distance"]


def _new_batch_tab() -> None:
    """Batch version of the new input form: one CSV, several stars. Validates only."""
    st.caption("Prototype of the new batch input form: builds and checks the inputs only. "
               "Nothing here runs the model yet.")
    st.write("Upload a CSV with one row per light-curve point: **star_id, time, flux, flux_err, "
             "teff, radius, distance** (teff/radius/distance repeat the same value for every row "
             "of a star).")
    up = st.file_uploader("Batch light curve CSV", type="csv", key="c1_batch_csv")
    if st.button("Use synthetic demo data instead", key="c1_batch_demo"):
        st.session_state["c1_batch_demo_df"] = _synthetic_batch()

    raw = None
    if up is not None:
        try:
            raw = pd.read_csv(up)
        except Exception as e:
            st.error(f"Could not read the CSV: {e}")
            return
    else:
        raw = st.session_state.get("c1_batch_demo_df")
    if raw is None:
        return

    cols = {c.lower(): c for c in raw.columns}
    missing = [c for c in BATCH_COLUMNS if c not in cols]
    if missing:
        st.error(f"Missing column(s): {', '.join(missing)}.")
        return
    raw = raw.rename(columns={cols[c]: c for c in BATCH_COLUMNS})

    rows, results = [], []
    for star_id, g in raw.groupby("star_id"):
        df = g[["time", "flux", "flux_err"]].apply(pd.to_numeric, errors="coerce")
        teff_vals = pd.to_numeric(g["teff"], errors="coerce").dropna()
        radius_vals = pd.to_numeric(g["radius"], errors="coerce").dropna()
        distance_vals = pd.to_numeric(g["distance"], errors="coerce").dropna()
        teff = float(teff_vals.iloc[0]) if len(teff_vals) else None
        radius = float(radius_vals.iloc[0]) if len(radius_vals) else None
        distance = float(distance_vals.iloc[0]) if len(distance_vals) else None

        clean_df, n_removed = _new_clean(df)
        errors, warnings_ = _new_validate(df, clean_df, n_removed, teff, radius, distance)
        status = "ok" if not errors else "error"
        results.append({
            "star_id": star_id, "status": status, "points": len(clean_df), "rows_removed": n_removed,
            "teff": teff, "radius": radius, "distance": distance,
            "message": "; ".join(errors) if errors else "; ".join(warnings_),
        })
        if status == "ok":
            rows.append({
                "star_id": star_id, "time": clean_df["time"].values, "flux": clean_df["flux"].values,
                "flux_err": clean_df["flux_err"].values, "teff": teff, "radius": radius,
                "distance": distance, "source": "batch_csv",
            })

    summary = pd.DataFrame(results)
    st.subheader("Batch summary")
    st.dataframe(summary, hide_index=True, width="stretch")
    st.caption(f"{sum(r['status'] == 'ok' for r in results)} of {len(results)} star(s) passed validation.")

    if rows and st.button("Continue with the valid stars", type="primary", key="c1_batch_continue"):
        st.session_state[BATCH_INPUT_KEY] = rows
        st.success(f"Saved {len(rows)} star(s) to st.session_state['{BATCH_INPUT_KEY}']. "
                   "Ready for the next step.")


def _synthetic_batch() -> pd.DataFrame:
    """Two small synthetic stars, so the batch form can be tried with no file."""
    frames = []
    for star_id, (teff, radius, distance) in enumerate(
            [(5700.0, 1.0, 50.0), (3500.0, 0.3, 20.0)], start=1):
        df = _synthetic_light_curve().copy()
        df["star_id"] = f"demo_{star_id}"
        df["teff"], df["radius"], df["distance"] = teff, radius, distance
        frames.append(df)
    return pd.concat(frames, ignore_index=True)[BATCH_COLUMNS]


def _new_clean(df: pd.DataFrame):
    """Drop missing/duplicate rows and bad-quality points; sort by time."""
    n0 = len(df)
    d = df.dropna(subset=["time", "flux", "flux_err"]).copy()
    if "quality" in d.columns:
        d = d[d["quality"].fillna(0) == 0]
    d = d.sort_values("time").drop_duplicates(subset="time", keep="first")
    return d.reset_index(drop=True), n0 - len(d)


def _new_validate(raw_df, clean_df, n_removed, teff, radius, distance):
    """Plain-English errors (block continuing) and warnings (allow continuing)."""
    errors, warnings_ = [], []
    if raw_df["flux"].isna().all():
        errors.append("The flux column is entirely missing.")
    if raw_df["flux_err"].isna().all():
        errors.append("The flux_err column is entirely missing.")
    elif (raw_df["flux_err"].dropna() <= 0).all():
        errors.append("flux_err is zero or negative for every point.")
    if len(clean_df) < MIN_POINTS:
        errors.append(f"Only {len(clean_df)} valid points after cleaning (need at least {MIN_POINTS}).")
    if teff is None or teff <= 0:
        errors.append("Teff is required and must be greater than zero.")
    if radius is None or radius <= 0:
        errors.append("Radius is required and must be greater than zero.")
    if distance is None or distance <= 0:
        errors.append("Distance is required and must be greater than zero.")
    if errors:
        return errors, warnings_

    if len(clean_df) > 1:
        cadence = float(np.median(np.diff(clean_df["time"].values)))
        if not (0.8 * CADENCE_TARGET_DAYS <= cadence <= 1.2 * CADENCE_TARGET_DAYS):
            warnings_.append(f"Median time step is {cadence * 1440:.2f} min, not close to the expected 2 min.")
    if len(raw_df) and n_removed / len(raw_df) > 0.20:
        warnings_.append(f"{n_removed} of {len(raw_df)} points ({n_removed / len(raw_df):.0%}) were removed by cleaning.")
    if not (2500 <= teff <= 7500):
        warnings_.append(f"Teff {teff:.0f} K is outside the usual 2500-7500 K range.")
    if radius > 2:
        warnings_.append(f"Radius {radius:.2f} R_sun is large for a dwarf star (possible giant).")
    if distance > 2000:
        warnings_.append(f"Distance {distance:.0f} pc is unusually far.")
    return errors, warnings_


def _new_preview(clean_df, n_removed, teff, radius, distance, sources, meta) -> None:
    t, f = clean_df["time"].values, clean_df["flux"].values
    cadence_min = float(np.median(np.diff(t))) * 1440.0 if len(t) > 1 else float("nan")
    span_days = float(t[-1] - t[0]) if len(t) else 0.0

    st.subheader("Preview")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Points", len(clean_df))
    m2.metric("Time span (days)", f"{span_days:.2f}")
    m3.metric("Median cadence (min)", f"{cadence_min:.2f}" if np.isfinite(cadence_min) else "n/a")
    m4.metric("Rows removed", n_removed)
    st.caption("Median flux_err: {0:.5f} - Teff {1:.0f} K ({2}) - Radius {3:.2f} R_sun ({4}) - "
              "Distance {5:.0f} pc ({6})".format(
                  float(np.median(clean_df["flux_err"].values)), teff, sources.get("teff", "user"),
                  radius, sources.get("radius", "user"), distance, sources.get("distance", "user")))

    st.plotly_chart(_new_gap_figure(t, f), width="stretch")

    if st.button("Continue", type="primary", key="c1_new_continue"):
        st.session_state[NEW_INPUT_KEY] = {
            "time": t, "flux": f, "flux_err": clean_df["flux_err"].values,
            "teff": float(teff), "radius": float(radius), "distance": float(distance),
            "tic_id": meta.get("tic_id"), "sector": meta.get("sector"),
            "source": meta.get("source"), "star_value_sources": sources,
        }
        st.success("Saved to st.session_state['c1_inputs']. Ready for the next step.")


def _new_gap_figure(t: np.ndarray, f: np.ndarray):
    """Light curve line, with gaps longer than ~10 points shown as breaks."""
    if len(t) > 1:
        step = np.median(np.diff(t))
        gap_at = np.where(np.diff(t) > 10 * step)[0]
        tt, ff = list(t), list(f)
        for idx in gap_at[::-1]:
            tt.insert(idx + 1, np.nan)
            ff.insert(idx + 1, np.nan)
    else:
        tt, ff = list(t), list(f)
    fig = go.Figure(go.Scattergl(x=tt, y=ff, mode="lines", line=dict(width=1, color="#5B7FA6")))
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), template="plotly_white",
                      xaxis_title="Time (BTJD)", yaxis_title="Flux")
    return fig


def _synthetic_light_curve() -> pd.DataFrame:
    """A small in-memory demo light curve, so the form can be tried with no file or internet."""
    rng = np.random.RandomState(0)
    t = np.arange(0.0, 3.0, CADENCE_TARGET_DAYS)
    flux = 1.0 + rng.normal(0, 0.0005, len(t))
    peak = len(t) // 2
    minutes_from_peak = (t - t[peak]) * 1440.0
    flux += np.where(minutes_from_peak < 0, 0.0, 0.01 * np.exp(-minutes_from_peak / 15.0))
    flux_err = np.full(len(t), 0.0005)
    return pd.DataFrame({"time": t, "flux": flux, "flux_err": flux_err})


def _synthetic_light_curve_columns():
    """The synthetic demo light curve as three separate newline-joined strings."""
    df = _synthetic_light_curve()
    return (tuple("\n".join(f"{v:.6f}" for v in df[col]) for col in ("time", "flux", "flux_err")))
