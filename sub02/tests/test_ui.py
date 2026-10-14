"""Tests for the Streamlit front end (helpers and views, with the API mocked)."""
import sys
from pathlib import Path

import pytest

UI = Path(__file__).resolve().parents[1] / "streamlit_app"
sys.path.insert(0, str(UI))

import api_client  # noqa: E402
import helpers  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

BANNER = "Trained on solar flares."


def flare(**over):
    r = {"flare_id": "TESS_1", "star_id": 1, "status": "ok", "calibrated_probability": 0.51, "confidence_lower": 0.32,
         "confidence_upper": 0.59, "predicted_label": "eruptive", "reliability": "confident", "features_missing": 0,
         "out_of_range": False, "warnings": [], "raw_score": 0.9, "conformal_set": "E",
         "features": {"rise10_min": 6.1, "g5": None}, "curve": {"t_min": [-20, -10, 0, 10, 30], "g": [0, 0.01, 1, 0.5, 0.2]}}
    r.update(over)
    return r


def not_ok(fid, status):
    return flare(flare_id=fid, status=status, calibrated_probability=None, confidence_lower=None, confidence_upper=None,
                 predicted_label=None, reliability=None, features=None, curve=None, features_missing=None)


def make_run(results, run_id="abc"):
    ok = [r for r in results if r["status"] == "ok"]
    return {"run_id": run_id, "name": "t", "model_version": "3", "created_at": "2026-10-21T10:00:00",
            "summary": {"n_flares": len(results), "n_ok": len(ok),
                        "n_eruptive": sum(r["predicted_label"] == "eruptive" for r in ok),
                        "n_confined": sum(r["predicted_label"] == "confined" for r in ok),
                        "n_confident": sum(r["reliability"] == "confident" for r in ok),
                        "n_uncertain": sum(r["reliability"] == "uncertain" for r in ok)}, "results": results}


RESULTS = [flare(flare_id="A"), flare(flare_id="B", calibrated_probability=0.3, confidence_lower=0.2, confidence_upper=0.4,
                                       reliability="uncertain"),
           flare(flare_id="C", calibrated_probability=0.1, confidence_lower=0.05, confidence_upper=0.2,
                 predicted_label="confined"), not_ok("BAD", "too_few_points")]

INFO = {"bundle_version": "3", "created": "2026-10-21", "data_hash": "abc", "training_summary": {},
        "limitations": ["Trained on solar flares only."],
        "metrics": {"cv5_grouped": {"tss": 0.149, "auc": 0.605}, "solar_test_raw_0.5": {"tss": 0.058, "auc": 0.588},
                    "solar_test_conformal_weighted": {"coverage_eruptive": 0.9}, "brier_test": 0.2456,
                    "brier_test_base_rate_baseline": 0.2446, "base_rate_cal": 0.263, "split": {"train": 321}}}


def app(view, run=None, timeout=30):
    at = AppTest.from_file(str(UI / "views" / f"{view}.py"), default_timeout=timeout)
    if run is not None:
        at.session_state["run"] = run
    return at


# ----------------------------------------------------------------- helpers
def test_formats():
    assert helpers.prob_text(0.51, 0.32, 0.59) == "51% (32–59%)"
    assert helpers.prob_text(None, None, None) == "—"
    assert helpers.label_text("eruptive") == "▲ Predicted eruptive"
    assert helpers.label_text("confined") == "■ Predicted confined"
    assert helpers.reliability_text("confident") == "● Confident"
    assert helpers.reliability_text("uncertain") == "○ Uncertain"


@pytest.mark.parametrize("label", ["eruptive", "confined"])
@pytest.mark.parametrize("rel", ["confident", "uncertain"])
def test_label_never_alone_and_no_detection_wording(label, rel):
    s = helpers.label_with_reliability(label, rel)
    assert "Predicted" in s and ("Confident" in s or "Uncertain" in s)
    assert not any(w in s.lower() for w in ("detected", "confirmed", "accurate"))


def test_explanations_are_the_fixed_texts():
    assert helpers.explain_label(flare()) == ("The system gave a single answer: eruptive. It was trained on solar flares "
                                              "and has not been validated on stars.")
    assert helpers.explain_label(flare(predicted_label="confined")).startswith("The system gave a single answer: confined.")
    assert helpers.explain_label(flare(reliability="uncertain")) == (
        "The system could not rule out either answer for this flare. Treat the label as a weak lean, not a result.")
    assert "fewer than 20 data points" in helpers.explain_label(not_ok("x", "too_few_points"))
    assert "Do not rely on its output" in helpers.WARNING_TEXT["out_of_range"]


def test_results_table_ranks_and_marks_unscored():
    t = helpers.results_table(RESULTS)
    assert dict(zip(t["Flare"], t["Rank"]))["A"] == 1
    bad = t[t["Flare"] == "BAD"].iloc[0]
    assert bad["Label"] == "—" and "fewer than 20" in bad["Status"] and bad["Rank"] != bad["Rank"]   # NaN rank
    assert (t[t["_ok"]]["Ranking score (5–95% range)"].str.contains(r"\(")).all()                   # never a bare number


def test_csv_exports():
    assert helpers.handoff_csv(RESULTS).splitlines()[0] == ",".join(helpers.HANDOFF_COLUMNS)
    assert len(helpers.handoff_csv(RESULTS).splitlines()) == 4          # header + 3 scored flares, not-scored left out
    assert len(helpers.full_csv(RESULTS).splitlines()) == 5
    assert "\r" not in helpers.full_csv(RESULTS)


# ----------------------------------------------------------------- views
def banner_ok(at):
    assert any(BANNER in w.value for w in at.warning), "limitations banner missing"


def test_banner_on_every_view(monkeypatch):
    monkeypatch.setattr(api_client, "model_info", lambda: INFO)
    monkeypatch.setattr(api_client, "runs", lambda: [])
    for view, run in [("predict", None), ("results", None), ("results", make_run(RESULTS)), ("runs", None), ("model_info", None)]:
        at = app(view, run).run()
        assert not at.exception, (view, at.exception)
        banner_ok(at)


def test_results_shows_label_with_reliability_and_probability_with_range():
    at = app("results", make_run(RESULTS)).run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "▲ Predicted eruptive · ● Confident" in text              # label next to its reliability
    assert "51% (32–59%)" in text                                     # probability with its range
    assert "The system gave a single answer: eruptive." in " ".join(m.value for m in at.markdown)
    assert at.dataframe                                              # the results table
    assert [m.label for m in at.metric] == ["Flares", "Scored", "Predicted eruptive", "Predicted confined", "Confident", "Uncertain"]


def test_results_unscored_flare_shows_reason():
    at = app("results", make_run(RESULTS)).run()
    bad_index = next(i for i, r in enumerate(RESULTS) if r["flare_id"] == "BAD")
    at.selectbox(key="detail_idx").set_value(bad_index).run()
    assert any("fewer than 20 data points" in i.value for i in at.info)


def test_results_filter_and_empty_state():
    at = app("results", make_run(RESULTS)).run()
    at.text_input[0].set_value("nothing-matches").run()
    assert any("No flares match" in i.value for i in at.info)


def test_results_warnings_are_explained():
    run = make_run([flare(flare_id="W", warnings=["out_of_range", "many_features_missing"], reliability="uncertain")])
    at = app("results", run).run()
    warns = " ".join(w.value for w in at.warning)
    assert "looks unlike the flares the model was trained on" in warns
    assert "Several shape features could not be measured" in warns


def test_predict_shows_unreachable_error(monkeypatch):
    monkeypatch.setattr(api_client, "samples", lambda: (_ for _ in ()).throw(api_client.ApiError("down", 0, "network")))
    at = app("predict").run()
    at.radio[0].set_value("Use the 33 sample TESS flares").run()
    at.button[0].click().run()
    assert any("The server can't be reached" in e.value for e in at.error)


def test_model_info_numbers_come_from_the_api(monkeypatch):
    monkeypatch.setattr(api_client, "model_info", lambda: INFO)
    at = app("model_info").run()
    assert not at.exception
    table = at.dataframe[0].value
    assert "0.149" in table.to_string() and "0.058" in table.to_string()
    assert any("Trained on solar flares only." in m.value for m in at.markdown)


def test_model_info_api_error(monkeypatch):
    monkeypatch.setattr(api_client, "model_info", lambda: (_ for _ in ()).throw(api_client.ApiError("boom", 500)))
    at = app("model_info").run()
    assert any("boom" in e.value for e in at.error)
    banner_ok(at)
