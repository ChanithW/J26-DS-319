import pandas as pd
import streamlit as st

import api_client
import helpers
import ui

SORT_COLUMNS = {"Ranking score": "_p", "Rank": "Rank", "Flare": "Flare", "Star": "Star", "Label": "Label",
                "Reliability": "Reliability", "Missing features": "Missing features", "Status": "Status"}

ui.banner()
run = st.session_state.get("run")
if not run:
    st.title("Results")
    st.info("There are no results to show yet. Run a prediction first, or open a saved run.")
    try:
        st.page_link("views/predict.py", label="Run a prediction", icon="🔭")
        st.page_link("views/runs.py", label="Open a saved run", icon="🗂️")
    except Exception:
        pass
    st.stop()

results = run["results"]
saved = bool(run.get("run_id"))
summary = run["summary"]

st.title(run.get("name") or ("Saved run" if saved else "Prediction results"))
st.caption(f"{run['created_at']} · model version {run['model_version']}" + ("" if saved else " · this run was not saved"))

d1, d2, d3 = st.columns([1, 1, 2])
d1.download_button("Download hand-off CSV", helpers.handoff_csv(results), "handoff.csv", "text/csv")
d2.download_button("Download full CSV", helpers.full_csv(results), "predictions_full.csv", "text/csv")

if saved:
    if not st.session_state.get("confirm_delete"):
        d3.button("Delete run", on_click=lambda: st.session_state.update(confirm_delete=True))
    else:
        st.warning("Delete this saved run? This cannot be undone.")
        y, n = st.columns(2)
        if y.button("Yes, delete", type="primary"):
            try:
                api_client.delete_run(run["run_id"])
                st.session_state.pop("run", None)
                st.session_state["confirm_delete"] = False
                st.switch_page("views/runs.py")
            except api_client.ApiError as e:
                ui.error_box("The run could not be deleted", e.message)
        n.button("Cancel", on_click=lambda: st.session_state.update(confirm_delete=False))

cols = st.columns(6)
for col, (k, v) in zip(cols, [("Flares", summary["n_flares"]), ("Scored", summary["n_ok"]),
                              ("Predicted eruptive", summary["n_eruptive"]), ("Predicted confined", summary["n_confined"]),
                              ("Confident", summary["n_confident"]), ("Uncertain", summary["n_uncertain"])]):
    col.metric(k, v)
if summary["n_ok"] < summary["n_flares"]:
    st.info(f"{summary['n_flares'] - summary['n_ok']} flare(s) could not be scored. They are greyed out below with the reason.")

# ---- filters and sorting
f1, f2, f3, f4 = st.columns([1, 1, 1, 2])
f_label = f1.selectbox("Label", ["All", "Predicted eruptive", "Predicted confined"])
f_rel = f2.selectbox("Reliability", ["All", "Confident", "Uncertain"])
f_status = f3.selectbox("Status", ["All", "Scored", "Not scored"])
f_search = f4.text_input("Search flare or star id", placeholder="e.g. TESS_294 or 294750180")
s1, s2 = st.columns([2, 1])
sort_by = s1.selectbox("Sort by", list(SORT_COLUMNS), index=0)
descending = s2.radio("Direction", ["Highest first", "Lowest first"], horizontal=True) == "Highest first"

df = helpers.results_table(results)
mask = pd.Series(True, index=df.index)
if f_label != "All":
    mask &= df["Label"].str.contains(f_label.split()[-1], na=False) & df["_ok"]
if f_rel != "All":
    mask &= df["Reliability"].str.contains(f_rel, na=False) & df["_ok"]
if f_status != "All":
    mask &= df["_ok"] if f_status == "Scored" else ~df["_ok"]
if f_search.strip():
    q = f_search.strip().lower()
    mask &= (df["Flare"].astype(str) + " " + df["Star"].astype(str)).str.lower().str.contains(q, regex=False)
view = df[mask].sort_values(SORT_COLUMNS[sort_by], ascending=not descending, na_position="last", kind="stable")

if view.empty:
    st.info("No flares match these filters. Clear a filter or change the search text.")
    st.stop()

shown = view.drop(columns=["_p", "_ok"])
grey = lambda row: (["color: #8a8a8a; font-style: italic"] * len(row)) if not df.loc[row.name, "_ok"] else [""] * len(row)
st.dataframe(shown.style.apply(grey, axis=1), hide_index=True, width="stretch")
st.caption("Click a column heading in the table to sort it. Use the selector below to open a flare's detail.")

# ---- flare detail
order = list(view.index)
if st.session_state.get("detail_idx") not in order:
    st.session_state["detail_idx"] = order[0]


def step(delta: int) -> None:
    cur = order.index(st.session_state["detail_idx"])
    st.session_state["detail_idx"] = order[max(0, min(len(order) - 1, cur + delta))]


st.divider()
st.subheader("Flare detail")
c1, c2, c3 = st.columns([3, 1, 1])
c1.selectbox("Open flare", order, key="detail_idx",
             format_func=lambda i: f"{df.loc[i, 'Rank'] if pd.notna(df.loc[i, 'Rank']) else '–'} · {df.loc[i, 'Flare']}")
pos = order.index(st.session_state["detail_idx"])
c2.button("← Previous flare", on_click=step, args=(-1,), disabled=pos == 0)
c3.button("Next flare →", on_click=step, args=(1,), disabled=pos == len(order) - 1)

r = results[st.session_state["detail_idx"]]
with st.container(border=True):
    st.markdown(f"#### {r.get('flare_id') or 'Unnamed flare'}")
    st.caption(f"Star {r.get('star_id', '—')} · flare {pos + 1} of {len(order)}")
    if r["status"] != "ok":
        st.info(helpers.status_reason(r["status"]) or "This flare was not scored.")
    else:
        st.markdown(f"**{helpers.label_with_reliability(r['predicted_label'], r['reliability'])}**")
        st.write(helpers.explain_label(r))
        for w in r.get("warnings") or []:
            st.warning(helpers.WARNING_TEXT.get(w, w))
        ui.probability_bar(r["calibrated_probability"], r["confidence_lower"], r["confidence_upper"])
        if r.get("curve"):
            st.markdown("**Light curve**")
            ui.light_curve(r["curve"])
        st.markdown("**Shape features**")
        feats = r.get("features") or {}
        st.dataframe(pd.DataFrame({"Feature": list(feats), "Value": [helpers.fmt_num(v) for v in feats.values()]}),
                     hide_index=True, width="stretch")
        st.write(f"Features missing: **{r.get('features_missing', 0)}** of 13 · Outside training range: "
                 f"**{'not checked' if r.get('out_of_range') is None else 'yes' if r.get('out_of_range') else 'no'}**")
