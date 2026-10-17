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


def test_the_about_and_limits_subheader_is_present():
    at = AppTest.from_file(str(APP), default_timeout=30).run()
    assert any(element.value == "About and limits" for element in at.subheader)