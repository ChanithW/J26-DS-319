import streamlit as st

import api_client
import helpers
import ui

FILE_ERRORS = {"invalid_csv", "file_too_large", "too_many_flares", "too_many_points"}

ui.banner()
st.title("Run a prediction")
st.write("Upload brightness curves of stellar flares, or try the 33 sample TESS flares. For each flare the system gives "
         "a ranking score with a range, a predicted label, and a reliability flag.")

st.subheader("1. Choose the flares")
source = st.radio("Input", ["Upload a CSV file", "Use the 33 sample TESS flares"], horizontal=True)
upload = None
file_error_slot = st.container()
if source == "Upload a CSV file":
    upload = st.file_uploader("CSV file (one row per light-curve point)", type=["csv"])
    file_error_slot = st.container()                      # file errors appear right under the upload box
    with st.expander("What should the CSV look like?"):
        st.write("Columns: `flare_id`, `star_id`, `tpeak`, `time`, `flux`. The first three repeat on every row of a "
                 "flare. `time` and `tpeak` are in TESS days (BTJD = Julian date − 2457000); `flux` can be on any "
                 "positive scale.")
        st.download_button("Download the CSV template", helpers.TEMPLATE_CSV, "component2_template.csv", "text/csv")
else:
    st.info("The 33 sample TESS flares from the research sample will be loaded from the API when you run.")

st.subheader("2. Run")
run_name = st.text_input("Run name (optional)", placeholder="e.g. TESS sample, October")
save_run = st.checkbox("Save this run so I can open it later", value=True)
ready = source != "Upload a CSV file" or upload is not None
go = st.button("Run prediction", type="primary", disabled=not ready)
if not ready:
    st.caption("Choose a file or the sample first.")

if go:
    name = run_name.strip() or None
    try:
        with st.spinner("Running the model…"):
            if source == "Upload a CSV file":
                run = api_client.predict_file(upload.name, upload.getvalue(), save_run, name)
            else:
                run = api_client.predict(api_client.samples(), save_run, name)
    except api_client.ApiError as e:
        if source == "Upload a CSV file" and e.code in FILE_ERRORS:
            with file_error_slot:
                ui.error_box("This file can't be used", e.message)
        else:
            ui.error_box("The server can't be reached" if e.unreachable else "The prediction could not be run", e.message)
    else:
        st.session_state["run"] = run
        st.switch_page("views/results.py")
