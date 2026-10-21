"""Checks for the CSV input of Sub-topic 4."""

import pandas as pd
import pytest

from risk_scoring import config, csv_input

# The three verified planets, one for each class. The cme_prob column is empty,
# so the default from config is used, the same as in the manual form.
HIGH = {
    "koi_sma": "0.045",
    "flare_rate_month": "1.8",
    "flare_rate_all": "9.5",
    "max_logE": "34.6",
    "teff": "3300",
}
MEDIUM = {
    "koi_sma": "0.2",
    "flare_rate_month": "0.6",
    "flare_rate_all": "4.0",
    "max_logE": "34.0",
    "teff": "3600",
}
LOW = {
    "koi_sma": "2.0",
    "flare_rate_month": "0.05",
    "flare_rate_all": "0.5",
    "max_logE": "33.0",
    "teff": "5200",
}


def _frame(rows):
    """Return the rows as a CSV frame."""
    return pd.DataFrame(rows)


def _scored(rows):
    """Return the scored rows of the frame built from the rows."""
    prepared, _ = csv_input.prepare(_frame(rows))
    return csv_input.score_rows(prepared)


def test_a_valid_csv_gives_high_medium_and_low():
    rows = _scored([HIGH, MEDIUM, LOW])
    assert [row.status for row in rows] == ["High", "Medium", "Low"]
    assert [row.result.ml_class for row in rows] == ["High", "Medium", "Low"]
    assert [row.result.formula.label for row in rows] == ["High", "Medium", "Low"]


def test_the_results_table_holds_the_five_inputs_and_the_two_classes():
    table = csv_input.results_table(_scored([HIGH, MEDIUM, LOW]))
    assert list(table.columns) == [
        "koi_sma",
        "flare_rate_month",
        "flare_rate_all",
        "max_logE",
        "teff",
        "status",
        "R_eff",
        "formula_class",
        "model_class",
        "reason",
    ]
    assert list(table["model_class"]) == ["High", "Medium", "Low"]
    assert list(table["formula_class"]) == ["High", "Medium", "Low"]
    assert list(table["R_eff"]) == [10.6951, 0.18048, 0.0001504]
    assert list(table["teff"]) == [3300.0, 3600.0, 5200.0]


def test_the_star_id_column_is_shown_when_it_is_there():
    rows = [dict(HIGH, star_id="TIC 12345")]
    table = csv_input.results_table(_scored(rows))
    assert "star_id" in table.columns
    assert list(table["star_id"]) == ["TIC 12345"]


def test_a_missing_column_is_reported_and_is_not_filled_in():
    frame = _frame([HIGH, MEDIUM, LOW]).drop(columns=["max_logE"])
    prepared, _ = csv_input.prepare(frame)
    assert csv_input.missing_columns(prepared) == ["max_logE"]
    assert "max_logE" not in prepared.columns


def test_a_per_day_rate_is_converted_to_a_per_month_rate():
    row = {
        "koi_sma": "0.045",
        "flare_rate_per_day": "0.059",
        "flare_rate_all": "9.5",
        "max_logE": "34.6",
        "teff": "3300",
    }
    prepared, notes = csv_input.prepare(_frame([row]))
    assert prepared["flare_rate_month"].tolist() == [pytest.approx(0.059 * 30.44)]
    assert notes == ["flare_rate_per_day -> flare_rate_month (value x 30.44)"]


def test_the_alternative_log_e_columns_become_max_log_e():
    for alternative in ["median_logE", "max_logE_median"]:
        row = {
            "koi_sma": "0.045",
            "flare_rate_month": "1.8",
            "flare_rate_all": "9.5",
            "teff": "3300",
        }
        row[alternative] = "34.6"
        prepared, notes = csv_input.prepare(_frame([row]))
        assert float(prepared["max_logE"].iloc[0]) == 34.6
        assert notes == [alternative + " -> max_logE"]


def test_the_temperature_column_is_accepted_in_either_case():
    lower = _scored([HIGH])
    upper_row = dict(HIGH)
    upper_row["Teff"] = upper_row.pop("teff")
    upper = _scored([upper_row])
    assert lower[0].status == "High"
    assert upper[0].status == "High"
    assert lower[0].values["teff"] == upper[0].values["teff"] == 3300.0
    assert lower[0].result.ml_class == upper[0].result.ml_class
    assert lower[0].result.formula.r_eff == upper[0].result.formula.r_eff


def test_a_row_with_a_bad_value_is_not_scored_and_the_other_rows_are():
    bad = dict(HIGH)
    bad["koi_sma"] = "not a number"
    rows = _scored([HIGH, bad, LOW])
    assert [row.status for row in rows] == ["High", "Not scored", "Low"]
    assert "koi_sma" in rows[1].reason
    assert rows[0].result is not None
    assert rows[2].result is not None
    table = csv_input.results_table(rows)
    assert list(table["status"]) == ["High", "Not scored", "Low"]
    assert table["reason"][1] != ""


def test_an_empty_cme_prob_uses_the_default_from_config():
    rows = _scored([HIGH])
    assert "CME probability default {0} used".format(config.DEFAULT_CME_PROB) in rows[0].result.flags
