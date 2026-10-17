"""Checks for the R_eff formula of Sub-topic 4."""

import pytest

from risk_scoring import config, formula


@pytest.mark.parametrize(
    "value,expected",
    [(1.0, "High"), (0.9999999, "Medium"), (0.1, "Medium"), (0.0999999, "Low"), (0.0, "Low")],
)
def test_the_boundary_values_get_the_right_label(value, expected):
    assert formula.formula_label(value) == expected


def test_an_orbital_distance_of_zero_raises_a_value_error():
    with pytest.raises(ValueError):
        formula.r_eff(1.8, config.DEFAULT_CME_PROB, 0)


def test_the_formula_label_matches_the_label_column_of_every_row(dose_csv):
    wrong = []
    for row in dose_csv.itertuples():
        value = formula.r_eff(row.flare_rate_month, row.cme_prob, row.koi_sma)
        label = formula.formula_label(value)
        if label != row.label:
            wrong.append((row.star_id, row.planet, row.koi_sma, row.cme_prob, label, row.label))
    assert wrong == []
    assert len(dose_csv) == 1185