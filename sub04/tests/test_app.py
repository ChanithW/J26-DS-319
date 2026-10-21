"""Smoke test for the Streamlit page of Sub-topic 4."""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "app.py"

EXAMPLE = {
    "koi_sma": "0.045",
    "flare_rate_month": "1.8",
    "flare_rate_all": "9.5",
    "max_logE": "34.6",
    "Teff": "3300",
    "cme_prob": "",
}


def test_the_page_loads_without_an_exception():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception


def test_the_example_planet_gives_the_high_risk_badge():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    for name, value in EXAMPLE.items():
        at.text_input(key=name).set_value(value)
    at.run()
    at.button(key="s4_score").click().run()
    assert not at.exception
    assert any("HIGH RISK" in element.value for element in at.markdown)


def test_the_about_subheader_is_present():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert any(element.value == "About" for element in at.subheader)


# The three verified planets, one for each class, with the cme_prob column empty.
VALID_CSV = (
    "koi_sma,flare_rate_month,flare_rate_all,max_logE,teff\n"
    "0.045,1.8,9.5,34.6,3300\n"
    "0.2,0.6,4.0,34.0,3600\n"
    "2.0,0.05,0.5,33.0,5200\n"
)
MISSING_COLUMN_CSV = "koi_sma,flare_rate_month,flare_rate_all,teff\n0.045,1.8,9.5,3300\n"
PER_DAY_CSV = "koi_sma,flare_rate_per_day,flare_rate_all,max_logE,teff\n0.045,0.059,9.5,34.6,3300\n"


def _csv_source(at):
    """Switch the page to the Previous outputs source and return the app test."""
    at.radio(key="s4_source").set_value("Previous outputs").run()
    return at


def _upload(at, text):
    """Upload the text as a CSV and rerun the page."""
    at.file_uploader[0].set_value(("rows.csv", text.encode("utf-8"), "text/csv"))
    return at.run()


def test_the_previous_outputs_source_shows_one_file_uploader():
    at = _csv_source(AppTest.from_file(str(APP), default_timeout=30).run())
    assert not at.exception
    assert len(at.file_uploader) == 1
    assert [element.label for element in at.text_input] == ["Star ID (TIC or KIC)"]
    assert not any(button.key == "s4_score" for button in at.button)
    assert any(button.key == "s4_reset" for button in at.button)


def test_the_catalogue_source_shows_the_star_id_lookup():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    at.radio(key="s4_source").set_value("Catalogue").run()
    assert not at.exception
    assert not at.file_uploader
    assert any("Demo catalogue" in element.value for element in at.caption)
    assert not at.text_input(key="future_star_id").disabled
    assert not at.button(key="future_load").disabled


def test_an_uploaded_csv_scores_three_planets():
    at = _csv_source(AppTest.from_file(str(APP), default_timeout=30).run())
    at = _upload(at, VALID_CSV)
    assert not at.exception
    assert not at.error
    table = at.dataframe[0].value
    assert list(table["status"]) == ["High", "Medium", "Low"]
    assert list(table["model_class"]) == ["High", "Medium", "Low"]
    assert list(table["formula_class"]) == ["High", "Medium", "Low"]
    assert any("HIGH RISK" in element.value for element in at.markdown)
    assert any(button.label == "Download results CSV" for button in at.download_button)


def test_a_csv_without_a_needed_column_gives_the_error():
    at = _csv_source(AppTest.from_file(str(APP), default_timeout=30).run())
    at = _upload(at, MISSING_COLUMN_CSV)
    assert not at.exception
    assert len(at.error) == 1
    message = at.error[0].value
    assert "max_logE" in message
    assert "flare_rate_month" in message
    assert "flare_rate_per_day" in message
    assert not at.dataframe


def test_the_per_day_rate_is_converted_in_the_table():
    at = _csv_source(AppTest.from_file(str(APP), default_timeout=30).run())
    at = _upload(at, PER_DAY_CSV)
    assert not at.exception
    table = at.dataframe[0].value
    assert table["flare_rate_month"].iloc[0] == pytest.approx(0.059 * 30.44)
    assert any("flare_rate_per_day" in element.value for element in at.caption)


def test_the_reset_button_clears_the_uploaded_file():
    at = _csv_source(AppTest.from_file(str(APP), default_timeout=30).run())
    at = _upload(at, VALID_CSV)
    assert at.dataframe
    at.button(key="s4_reset").click().run()
    assert not at.exception
    assert at.file_uploader[0].value is None
    assert not at.dataframe