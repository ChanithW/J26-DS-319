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


# ----------------------------------------------------------------- manual entry
def points(n=30, start=-20, step=2):
    return "\n".join(f"{start + i * step}, {1 + i / 100:.3f}" for i in range(n))


def test_parse_points_formats_header_and_errors():
    t, f, e = helpers.parse_points("time,flux\n\n1,2\n3\t4\n5 6\n7;8\n")
    assert (t, f, e) == ([1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0], [])
    _, _, errs = helpers.parse_points("1,2\nabc,4\n5,6,7")
    assert errs[0].startswith("Line 2:") and errs[1].startswith("Line 3:")
    assert helpers.parse_points("-20, 1.5e-3\n0, 2E2")[1] == [0.0015, 200.0]


def test_build_flare_converts_minutes_and_validates():
    flare, errors, warnings = helpers.build_flare("F1", "123", "minutes", "", points())
    assert errors == [] and warnings == []
    assert flare["tpeak"] == 0.0 and flare["star_id"] == 123 and abs(flare["time"][0] - (-20 / 1440)) < 1e-12
    assert helpers.build_flare(" ", "12.5", "minutes", "", points())[1] == [
        "Enter a flare id.", "Star id must be a whole number (the TIC number)."]
    assert "already in the list" in helpers.build_flare("F1", "1", "minutes", "", points(), ["F1"])[1][0]
    assert "peak time in BTJD" in helpers.build_flare("F1", "1", "btjd", "", points())[1][0]
    assert helpers.build_flare("F1", "1", "btjd", "1500.5", points(30, 1499.9, 0.0014))[0]["tpeak"] == 1500.5
    assert "at least two" in helpers.build_flare("F1", "1", "minutes", "", "1,2")[1][0]


def test_build_flare_warns_about_few_points_but_allows_them():
    flare, errors, warnings = helpers.build_flare("F1", "1", "minutes", "", points(10))
    assert flare is not None and errors == [] and "Only 10 point" in warnings[0]


def test_demo_flare_is_valid():
    flare, errors, warnings = helpers.build_flare("D", "1", "minutes", "", helpers.example_points())
    assert errors == [] and warnings == [] and len(flare["time"]) == 55


def fill_and_add(at, flare_id="MANUAL_1"):
    at.radio[0].set_value("Enter flares manually").run()
    at.text_input(key="m_id").set_value(flare_id)
    at.text_input(key="m_star").set_value("777")
    at.text_area(key="m_points").set_value(points())
    next(b for b in at.button if b.label == "Add this flare").click().run()
    return at


def test_manual_form_adds_flare_to_list_and_runs_it(monkeypatch):
    sent = {}

    def fake_predict(flares, save_run, name):
        sent.update(flares=flares, save_run=save_run, name=name)
        return make_run([flare()])

    monkeypatch.setattr(api_client, "predict", fake_predict)
    at = fill_and_add(app("predict").run())
    assert not at.exception
    assert any("MANUAL_1" in m.value and "30 points" in m.value for m in at.markdown)
    assert any("Added" in s.value for s in at.success)
    next(b for b in at.button if b.label == "Run prediction").click().run()
    assert sent["flares"][0]["flare_id"] == "MANUAL_1" and sent["flares"][0]["star_id"] == 777
    assert sent["flares"][0]["tpeak"] == 0.0 and sent["save_run"] is True


def test_manual_form_shows_errors_and_adds_nothing():
    at = app("predict").run()
    at.radio[0].set_value("Enter flares manually").run()
    next(b for b in at.button if b.label == "Add this flare").click().run()
    assert any("Enter a flare id." in e.value for e in at.error)
    assert not at.session_state["manual_flares"]
    assert next(b for b in at.button if b.label == "Run prediction").disabled


def test_manual_form_demo_button_fills_the_form():
    at = app("predict").run()
    at.radio[0].set_value("Enter flares manually").run()
    next(b for b in at.button if b.label == "Fill with a demo flare").click().run()
    assert at.text_input(key="m_id").value == "EXAMPLE_FLARE"
    assert helpers.parse_points(at.text_area(key="m_points").value)[2] == []


def test_streamlit_form_hints_and_spacing_warning():
    at = app("predict").run()
    at.radio[0].set_value("Enter flares manually").run()
    captions = " ".join(c.value for c in at.caption)
    assert "Time 0 must be the flare's brightest point (within about 6 minutes)." in captions
    assert "Include at least 3 points before -8 minutes (they set the background)." in captions
    slow = helpers.build_flare("F1", "1", "minutes", "", points(30, -20, 10))
    assert slow[0] is not None and "about 10.0 minutes" in " ".join(slow[2]) and "does not re-bin" in " ".join(slow[2])
    assert "about 30 seconds" in " ".join(helpers.build_flare("F1", "1", "minutes", "", points(60, -20, 0.5))[2])
    assert helpers.build_flare("F1", "1", "minutes", "", points(30, -20, 2))[2] == []


def test_streamlit_parse_points_tab_separated_spreadsheet():
    t, f, e = helpers.parse_points("-20\t1.0\n-18\t1.1\n-16 1.2\n-14,1.3")
    assert e == [] and t == [-20.0, -18.0, -16.0, -14.0]
