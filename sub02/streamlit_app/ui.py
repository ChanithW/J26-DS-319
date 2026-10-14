"""Page furniture used by every view: the limitations banner, error display, charts."""
from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

import helpers

BANNER_TEXT = ("**Trained on solar flares.** Accuracy on stars has not been validated. "
               "The probability is a ranking score, not a chance of a CME.")


def banner() -> None:
    """Always shown, not dismissible. Every view calls this first."""
    st.warning(BANNER_TEXT, icon="⚠️")
    try:
        st.page_link("views/model_info.py", label="Read the limitations", icon="ℹ️")
    except Exception:                                     # e.g. when a view is run on its own in a test
        pass


def error_box(title: str, message: str) -> None:
    st.error(f"**{title}**\n\n{message}")


def probability_bar(p: float, lo: float, hi: float) -> None:
    """Point value and 5-95% whisker on a 0-100% scale, always with the text next to it."""
    st.markdown(f"### {helpers.prob_text(p, lo, hi)}")
    pos = lambda v: min(100.0, max(0.0, v * 100))
    st.markdown(
        f"""<div role="img" aria-label="Ranking score {helpers.pct(p)}, 5 to 95 percent range {round(lo*100)} to {helpers.pct(hi)}"
        style="position:relative;height:16px;background:rgba(128,128,128,.3);border-radius:4px;margin:4px 0 2px">
        <div style="position:absolute;top:7px;height:2px;left:{pos(lo)}%;width:{pos(hi)-pos(lo)}%;background:#64748b"></div>
        <div style="position:absolute;top:0;height:16px;width:2px;left:{pos(lo)}%;background:#64748b"></div>
        <div style="position:absolute;top:0;height:16px;width:2px;left:{pos(hi)}%;background:#64748b"></div>
        <div style="position:absolute;top:-3px;height:22px;width:6px;left:calc({pos(p)}% - 3px);background:#4f46e5;border-radius:3px"></div>
        </div>
        <div style="display:flex;justify-content:space-between;font-size:12px;opacity:.7"><span>0%</span><span>100%</span></div>""",
        unsafe_allow_html=True,
    )
    st.caption("Ranking score, not a chance of a CME. The range reflects variation in the solar training data only, "
               "not the solar-to-stellar gap.")


def light_curve(curve: dict) -> None:
    df = pd.DataFrame({"t": curve["t_min"], "g": curve["g"]})
    band = alt.Chart(pd.DataFrame({"a": [-20], "b": [-8]})).mark_rect(opacity=0.15, color="#64748b").encode(x="a:Q", x2="b:Q")
    line = alt.Chart(df).mark_line(point=True, color="#4f46e5", strokeWidth=1).encode(
        x=alt.X("t:Q", title="minutes from peak", scale=alt.Scale(domain=[-20, 90])),
        y=alt.Y("g:Q", title="brightness above background (peak = 1)"),
        tooltip=[alt.Tooltip("t:Q", title="minutes"), alt.Tooltip("g:Q", format=".3f", title="brightness")],
    )
    peak = alt.Chart(pd.DataFrame({"t": [0]})).mark_rule(strokeDash=[4, 2], color="#4f46e5").encode(x="t:Q")
    st.altair_chart(band + line + peak, width="stretch")
    st.caption(helpers.curve_summary(curve) + " The shaded band (−20 to −8 minutes) is the background window.")
    with st.expander("Show the data as a table"):
        st.dataframe(df.rename(columns={"t": "Minutes from peak", "g": "Brightness above background"}),
                     hide_index=True, width="stretch")
