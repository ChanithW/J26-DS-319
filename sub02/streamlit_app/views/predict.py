import streamlit as st

import api_client
import helpers
import ui

FILE_ERRORS = {"invalid_csv", "file_too_large", "too_many_flares", "too_many_points"}
UPLOAD, MANUAL, SAMPLE = "Upload a CSV file", "Enter flares manually", "Use the 33 sample TESS flares"
FIELDS = {"m_id": "", "m_star": "", "m_tpeak": "", "m_points": ""}

ss = st.session_state
ss.setdefault("manual_flares", [])
for k, v in FIELDS.items():
    ss.setdefault(k, v)
ss.setdefault("m_unit", "minutes")
for k in ("m_errors", "m_warnings"):
    ss.setdefault(k, [])


def add_flare() -> None:
    flare, errors, warnings = helpers.build_flare(ss["m_id"], ss["m_star"], ss["m_unit"], ss["m_tpeak"], ss["m_points"],
                                                  [f["flare_id"] for f in ss["manual_flares"]])
    ss["m_errors"], ss["m_warnings"], ss["m_added"] = errors, warnings, None
    if flare:
        ss["manual_flares"].append(flare)
        ss["m_added"] = flare["flare_id"]
        ss.update(FIELDS)                                    # clear the form for the next flare


def fill_demo() -> None:
    ss.update(m_id="EXAMPLE_FLARE", m_star="1", m_unit="minutes", m_tpeak="", m_points=helpers.example_points(),
              m_errors=[], m_warnings=[], m_added=None)


def clear_form() -> None:
    ss.update(FIELDS, m_errors=[], m_warnings=[], m_added=None)


def remove_flare(i: int) -> None:
    ss["manual_flares"].pop(i)


ui.banner()
st.title("Run a prediction")
st.write("Upload brightness curves of stellar flares, enter a flare by hand, or try the 33 sample TESS flares. For each "
         "flare the system gives a ranking score with a range, a predicted label, and a reliability flag.")

st.subheader("1. Choose the flares")
source = st.radio("Input", [UPLOAD, MANUAL, SAMPLE], horizontal=True)
upload = None
file_error_slot = st.container()
if source == UPLOAD:
    upload = st.file_uploader("CSV file (one row per light-curve point)", type=["csv"])
    file_error_slot = st.container()                      # file errors appear right under the upload box
    with st.expander("What should the CSV look like?"):
        st.write("Columns: `flare_id`, `star_id`, `tpeak`, `time`, `flux`. The first three repeat on every row of a "
                 "flare. `time` and `tpeak` are in TESS days (BTJD = Julian date − 2457000); `flux` can be on any "
                 "positive scale.")
        st.download_button("Download the CSV template", helpers.TEMPLATE_CSV, "component2_template.csv", "text/csv")
elif source == MANUAL:
    with st.container(border=True):
        c1, c2 = st.columns(2)
        c1.text_input("Flare id", key="m_id", placeholder="e.g. TESS_294750180")
        c2.text_input("Star id (TIC number)", key="m_star", placeholder="e.g. 294750180")
        u1, u2 = st.columns(2)
        u1.selectbox("Time is given as", ["minutes", "btjd"], key="m_unit",
                     format_func=lambda v: "Minutes from the flare peak" if v == "minutes" else "TESS days (BTJD)")
        if ss["m_unit"] == "btjd":
            u2.text_input("Flare peak time (BTJD)", key="m_tpeak", placeholder="e.g. 1500.1234")
        st.text_area("Brightness curve: one “time, flux” pair per line", key="m_points", height=220,
                     placeholder="-20, 1.000\n-18, 1.001\n-16, 1.000\n…")
        st.caption("Paste two columns from a spreadsheet (separated by a comma, a tab or a space), or type them. Use "
                   "2-minute data covering −20 to +90 minutes around the peak (at least 20 points). Flux can be on any "
                   "positive scale.")
        st.caption(("Time 0 must be the flare's brightest point (within about 6 minutes)."
                    if ss["m_unit"] == "minutes" else
                    "The peak time must be the flare's brightest point (within about 6 minutes).")
                   + " Include at least 3 points before -8 minutes (they set the background).")
        if ss["m_errors"]:
            ui.error_box("This flare can't be added yet", "\n".join(f"- {m}" for m in ss["m_errors"]))
        for m in ss["m_warnings"]:
            st.warning(m)
        if ss.get("m_added"):
            st.success(f"Added “{ss['m_added']}” to the list below.")
        b1, b2, b3 = st.columns(3)
        b1.button("Add this flare", type="primary", on_click=add_flare)
        b2.button("Fill with a demo flare", on_click=fill_demo)
        b3.button("Clear the form", on_click=clear_form)
        st.caption("The demo flare is synthetic, only to try the form. It is not real data.")
    if ss["manual_flares"]:
        st.markdown(f"**Manually entered flares ({len(ss['manual_flares'])})**")
        for i, f in enumerate(ss["manual_flares"]):
            a, b = st.columns([4, 1])
            a.write(f"**{f['flare_id']}** · star {f['star_id']} · {len(f['time'])} points")
            b.button("Remove", key=f"rm_{i}_{f['flare_id']}", on_click=remove_flare, args=(i,))
else:
    st.info("The 33 sample TESS flares from the research sample will be loaded from the API when you run.")

st.subheader("2. Run")
run_name = st.text_input("Run name (optional)", placeholder="e.g. TESS sample, October")
save_run = st.checkbox("Save this run so I can open it later", value=True)
ready = (source == UPLOAD and upload is not None) or (source == MANUAL and bool(ss["manual_flares"])) or source == SAMPLE
go = st.button("Run prediction", type="primary", disabled=not ready)
if not ready:
    st.caption("Choose a file, add a flare, or pick the sample first.")

if go:
    name = run_name.strip() or None
    try:
        with st.spinner("Running the model…"):
            if source == UPLOAD:
                run = api_client.predict_file(upload.name, upload.getvalue(), save_run, name)
            elif source == MANUAL:
                run = api_client.predict(ss["manual_flares"], save_run, name)
            else:
                run = api_client.predict(api_client.samples(), save_run, name)
    except api_client.ApiError as e:
        if source == UPLOAD and e.code in FILE_ERRORS:
            with file_error_slot:
                ui.error_box("This file can't be used", e.message)
        else:
            ui.error_box("The server can't be reached" if e.unreachable else "The prediction could not be run", e.message)
    else:
        st.session_state["run"] = run
        st.switch_page("views/results.py")
