"""Checks for the star catalogue lookup of Sub-topic 4."""

from pathlib import Path

import pandas as pd
import pytest

from risk_scoring import catalogue, predictor

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

SUB04 = Path(__file__).resolve().parents[1]
APP = SUB04 / "app" / "app.py"
DATA_CSV = SUB04 / "data" / "processed" / "training_v12_dose.csv"

# A test star with one planet: label High, dose class High, model class Medium.
ONE_PLANET = "KIC5436161"
# A test star with two planets.
TWO_PLANETS = "KIC10670119"
# A test star where the dose class (High) differs from the model class (Medium).
DOSE_CLASS = "TIC422655579"
# A star of the training split, so it is not in the catalogue.
TRAIN_ONLY = "KIC10000941"


def test_the_catalogue_holds_the_test_rows_only():
    frame = pd.read_csv(DATA_CSV)
    entries = catalogue.load_catalogue()
    test = frame[frame["split"] == "test"]
    train = frame[frame["split"] == "train"]

    assert len(entries) == len(test) == 235
    assert {entry.star_id for entry in entries} == set(test["star_id"])
    assert not set(train["star_id"]) & {entry.star_id for entry in entries}


def test_a_test_star_is_found_with_its_values():
    found = catalogue.find_star(catalogue.load_catalogue(), ONE_PLANET)
    assert len(found) == 1
    entry = found[0]
    assert entry.planet == "K04351.01"
    assert entry.dose_class == "High"
    assert entry.label == "High"
    assert entry.inputs["koi_sma"] == pytest.approx(0.0124)
    assert entry.inputs["Teff"] == pytest.approx(5136.0)
    assert entry.inputs["cme_prob"] == pytest.approx(0.3560476881920121)


def test_the_star_id_is_trimmed_and_matched_exactly():
    entries = catalogue.load_catalogue()
    assert catalogue.find_star(entries, "  " + ONE_PLANET + "  ")
    assert catalogue.find_star(entries, ONE_PLANET.lower()) == []
    assert catalogue.find_star(entries, "") == []


def test_a_star_of_the_training_split_is_not_in_the_catalogue():
    assert catalogue.find_star(catalogue.load_catalogue(), TRAIN_ONLY) == []


def test_a_star_with_several_planets_gives_one_entry_per_planet():
    found = catalogue.find_star(catalogue.load_catalogue(), TWO_PLANETS)
    assert [entry.planet for entry in found] == ["Kepler-369 b", "Kepler-369 c"]
    assert found[0].inputs["koi_sma"] == pytest.approx(0.0316)
    assert found[1].inputs["koi_sma"] == pytest.approx(0.0977)
    assert [catalogue.planet_label(entry) for entry in found] == ["Kepler-369 b", "Kepler-369 c"]


def test_the_label_never_reaches_the_prediction(tmp_path, monkeypatch):
    """The stored label is for display only: changing it changes nothing in the result."""
    frame = pd.read_csv(DATA_CSV)
    changed = tmp_path / "changed_labels.csv"
    frame.loc[frame["star_id"] == ONE_PLANET, ["label", "label_dose"]] = "Low"
    frame.to_csv(changed, index=False)

    original = catalogue.find_star(catalogue.load_catalogue(), ONE_PLANET)[0]
    before = predictor.predict(catalogue.raw_inputs(original))

    monkeypatch.setattr(catalogue, "CATALOGUE_CSV", changed)
    edited = catalogue.find_star(catalogue.load_catalogue(), ONE_PLANET)[0]
    after = predictor.predict(catalogue.raw_inputs(edited))

    assert original.dose_class == "High"
    assert edited.dose_class == "Low"
    assert before.ml_class == after.ml_class
    assert before.formula.label == after.formula.label
    assert before.formula.r_eff == after.formula.r_eff
    assert before.probabilities == after.probabilities


def _catalogue_source(at):
    """Switch the page to the Catalogue source and return the app test."""
    at.radio(key="s4_source").set_value("Catalogue").run()
    return at


def _load_star(at, star_id):
    """Type the star ID, press Load star and rerun the page."""
    at.text_input(key="future_star_id").set_value(star_id)
    at.button(key="future_load").click().run()
    return at


def _markdowns(at):
    """Return the text of every markdown element on the page."""
    return [element.value for element in at.markdown]


def test_the_star_id_lookup_works_only_in_the_catalogue_mode():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert at.text_input(key="future_star_id").disabled
    assert at.button(key="future_load").disabled

    at = _catalogue_source(at)
    assert not at.exception
    assert not at.text_input(key="future_star_id").disabled
    assert not at.button(key="future_load").disabled
    assert any("Demo catalogue: stars the model was not trained on." in element.value for element in at.caption)


def test_loading_a_test_star_fills_the_inputs_and_shows_the_result():
    at = _catalogue_source(AppTest.from_file(str(APP), default_timeout=30).run())
    at = _load_star(at, ONE_PLANET)
    assert not at.exception
    assert at.session_state["koi_sma"] == "0.0124"
    assert float(at.session_state["Teff"]) == pytest.approx(5136.0)
    assert at.text_input(key="koi_sma").value == "0.0124"
    assert any("MEDIUM RISK" in element.value for element in at.markdown)
    assert any("Catalogue label (for comparison): High" in text for text in _markdowns(at))


def test_the_catalogue_label_is_the_dose_class_and_not_the_model_class():
    at = _catalogue_source(AppTest.from_file(str(APP), default_timeout=30).run())
    at = _load_star(at, DOSE_CLASS)
    assert not at.exception
    assert any("MEDIUM RISK" in element.value for element in at.markdown)
    assert any("Catalogue label (for comparison): High" in text for text in _markdowns(at))


def test_a_star_with_several_planets_shows_the_planet_picker():
    at = _catalogue_source(AppTest.from_file(str(APP), default_timeout=30).run())
    at = _load_star(at, TWO_PLANETS)
    assert not at.exception
    assert len(at.selectbox) == 1
    assert list(at.selectbox[0].options) == ["Kepler-369 b", "Kepler-369 c"]
    assert float(at.session_state["koi_sma"]) == pytest.approx(0.0316)

    at.selectbox[0].set_value("Kepler-369 c").run()
    assert not at.exception
    assert float(at.session_state["koi_sma"]) == pytest.approx(0.0977)


def test_an_unknown_star_gives_the_warning_and_fills_nothing():
    at = _catalogue_source(AppTest.from_file(str(APP), default_timeout=30).run())
    at = _load_star(at, "TIC 999999999")
    assert not at.exception
    assert any("Star not in the demo catalogue (test stars only)." in element.value for element in at.warning)
    assert at.session_state["koi_sma"] == ""
    assert not any("RISK" in text for text in _markdowns(at))


def test_an_edited_value_hides_the_catalogue_label():
    at = _catalogue_source(AppTest.from_file(str(APP), default_timeout=30).run())
    at = _load_star(at, ONE_PLANET)
    assert any("Catalogue label (for comparison): High" in text for text in _markdowns(at))

    at.text_input(key="koi_sma").set_value("0.05").run()
    assert not at.exception
    assert any("Values edited, so the catalogue label no longer applies" in text for text in _markdowns(at))
    assert not any("Catalogue label (for comparison)" in text for text in _markdowns(at))

    at.text_input(key="koi_sma").set_value("0.0124").run()
    assert not at.exception
    assert any("Catalogue label (for comparison): High" in text for text in _markdowns(at))


def test_the_reset_button_clears_the_catalogue():
    at = _catalogue_source(AppTest.from_file(str(APP), default_timeout=30).run())
    at = _load_star(at, ONE_PLANET)
    assert at.session_state["koi_sma"] == "0.0124"

    at.button(key="s4_reset").click().run()
    assert not at.exception
    assert at.session_state["koi_sma"] == ""
    assert at.text_input(key="future_star_id").value == ""
    assert not any("Catalogue label (for comparison)" in text for text in _markdowns(at))
