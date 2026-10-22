import pandas as pd
import streamlit as st

import api_client
import helpers
import ui

ui.banner()
st.title("Model information")

try:
    info = api_client.model_info()
except api_client.ApiError as e:
    ui.error_box("Model information could not be loaded", e.message)
    st.stop()

m = info["metrics"] or {}
cv, held = m.get("cv5_grouped") or {}, m.get("solar_test_raw_0.5") or {}
conf, split = m.get("solar_test_conformal_weighted") or {}, m.get("split") or {}
st.caption(f"Version {info['bundle_version']} · created {info['created']} · training data {info['data_hash']}")

st.subheader("Limitations")
for line in info["limitations"]:
    st.markdown(f"- {line}")

# Every number below comes from the API; only the wording is fixed here.
st.subheader("How well it works on the Sun")
st.write("These numbers measure solar flares only. They say nothing about accuracy on stars, which has not been measured.")
MEASURES = [
    ("TSS", "tss", helpers.fmt_num, "Hit rate + clear rate − 1. 0 is no skill, 1 is perfect. The main measure here, because most flares are confined."),
    ("AUC", "auc", helpers.fmt_num, "How well the score ranks eruptive flares above confined ones. 0.5 is chance."),
    ("Hit rate", "hit_rate", helpers.fmt_pct, "Share of eruptive flares labelled eruptive."),
    ("Clear rate", "clear_rate", helpers.fmt_pct, "Share of confined flares labelled confined."),
    ("Balanced accuracy", "balanced_accuracy", helpers.fmt_pct, "Average of hit rate and clear rate."),
    ("Plain accuracy", "accuracy", helpers.fmt_pct, "Share labelled correctly. Misleading here: always answering “confined” scores higher."),
]
st.dataframe(pd.DataFrame([{"Measure": n, "Cross-validation (all solar flares)": f(cv.get(k)),
                            "Held-out solar test": f(held.get(k)), "What it means": meaning}
                           for n, k, f, meaning in MEASURES]), hide_index=True, width="stretch")

st.subheader("Other checks")
a, b = st.columns(2)
a.markdown("**Reliability flag coverage (target 90%)**")
a.write(f"Eruptive {helpers.fmt_pct(conf.get('coverage_eruptive'))} · confined {helpers.fmt_pct(conf.get('coverage_confined'))}")
a.markdown("**Held-out flares with a single answer**")
a.write(f"{helpers.fmt_pct(conf.get('single_label_share'))} (those answers were right "
        f"{helpers.fmt_pct(conf.get('single_label_accuracy'))} of the time)")
b.markdown("**Brier score**")
b.write(f"{helpers.fmt_num(m.get('brier_test'))} against {helpers.fmt_num(m.get('brier_test_base_rate_baseline'))} for "
        "always answering the base rate. The probability is a ranking score, not a chance.")
b.markdown("**Label cut-off (eruptive share of the calibration flares)**")
b.write(f"{helpers.fmt_num(m.get('base_rate_cal'))}. A flare is labelled eruptive when its probability is at or above this, not 0.5.")
st.write(f"Solar flares used: {m.get('n_solar_with_features', '—')} ({m.get('n_eruptive', '—')} eruptive) · "
         f"train {split.get('train', '—')} / calibration {split.get('calibration', '—')} / test {split.get('test', '—')}")

st.subheader("How to read an output")
st.markdown(
    "- **Ranking score.** Higher means the flare looks more like the eruptive solar flares. It is for ordering flares, "
    "not a chance of a CME. The range shows how much it moves with the solar training data only.\n"
    "- **Predicted eruptive / confined.** A label, not a detection. It always comes with a reliability flag.\n"
    "- **Confident** (●) means the system could rule out one of the two answers. **Uncertain** (○) means it could not; "
    "treat the label as a weak lean. Most stellar flares are uncertain, and that is shown, not hidden.\n"
    "- **Warnings.** Many missing features make the answer uncertain. A check that forces flares unlike the solar training "
    "flares to uncertain exists, but it is switched off by default (it is not part of the research), so a confident answer "
    "does not mean the flare looks like the training data.\n"
    "- **No rates.** The share labelled eruptive depends strongly on how solar and stellar flares are aligned, so no "
    "stellar CME rate can be read from these outputs."
)
