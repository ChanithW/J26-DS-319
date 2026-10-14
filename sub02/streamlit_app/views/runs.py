import pandas as pd
import streamlit as st

import api_client
import ui

ui.banner()
st.title("Saved runs")

try:
    runs = api_client.runs()
except api_client.ApiError as e:
    ui.error_box("Saved runs could not be loaded", e.message)
    st.stop()

if not runs:
    st.info("No saved runs yet. Run a prediction and keep “Save this run” ticked.")
    try:
        st.page_link("views/predict.py", label="Run a prediction", icon="🔭")
    except Exception:
        pass
    st.stop()

table = pd.DataFrame([{"Name": r.get("name") or "Untitled run", "Date": r["created_at"], "Model": f"v{r['model_version']}",
                       "Flares": r["n_flares"], "Predicted eruptive": r["n_eruptive"], "Confident": r["n_confident"]}
                      for r in runs])
st.dataframe(table, hide_index=True, width="stretch")

labels = {r["run_id"]: f"{r.get('name') or 'Untitled run'} · {r['created_at']}" for r in runs}
chosen = st.selectbox("Choose a run", list(labels), format_func=labels.get)
o, d = st.columns(2)
if o.button("Open run", type="primary"):
    try:
        st.session_state["run"] = api_client.run(chosen)
        st.switch_page("views/results.py")
    except api_client.ApiError as e:
        ui.error_box("The run could not be opened", e.message)
if d.button("Delete run"):
    st.session_state["delete_candidate"] = chosen
if st.session_state.get("delete_candidate") == chosen:
    st.warning("Delete this saved run? This cannot be undone.")
    y, n = st.columns(2)
    if y.button("Yes, delete", type="primary"):
        try:
            api_client.delete_run(chosen)
            st.session_state.pop("delete_candidate", None)
            if (st.session_state.get("run") or {}).get("run_id") == chosen:
                st.session_state.pop("run", None)
            st.rerun()
        except api_client.ApiError as e:
            ui.error_box("The run could not be deleted", e.message)
    if n.button("Cancel"):
        st.session_state.pop("delete_candidate", None)
        st.rerun()
