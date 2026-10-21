"""Checks for the input validation of Sub-topic 4."""

from risk_scoring import config, validation

NORMAL = {
    "koi_sma": "0.045",
    "flare_rate_month": "1.8",
    "flare_rate_all": "9.5",
    "max_logE": "34.6",
    "Teff": "3300",
    "cme_prob": "",
}


def test_a_normal_row_has_no_errors():
    result = validation.validate_inputs(NORMAL)
    assert result.errors == []
    assert result.values["koi_sma"] == 0.045
    assert result.values["star_type"] == 3


def test_a_normal_row_uses_the_default_cme_probability():
    result = validation.validate_inputs(NORMAL)
    assert result.cme_defaulted is True
    assert result.values["cme_prob"] == config.DEFAULT_CME_PROB
    assert "CME probability default 0.47 used" in result.flags


def test_a_normal_row_is_inside_the_training_range():
    result = validation.validate_inputs(NORMAL)
    assert "All inputs inside training range" in result.flags


def test_a_missing_koi_sma_is_reported():
    result = validation.validate_inputs(dict(NORMAL, koi_sma=""))
    assert any("koi_sma is required" in message for message in result.errors)
    assert "koi_sma" not in result.values


def test_a_negative_koi_sma_is_rejected():
    result = validation.validate_inputs(dict(NORMAL, koi_sma=-1))
    assert any("koi_sma must be greater than 0" in message for message in result.errors)


def test_text_for_max_logE_is_read_as_a_number():
    result = validation.validate_inputs(dict(NORMAL, max_logE="34.6"))
    assert result.errors == []
    assert result.values["max_logE"] == 34.6


def test_text_that_is_not_a_number_is_rejected():
    result = validation.validate_inputs(dict(NORMAL, max_logE="not a number"))
    assert any("max_logE must be a number" in message for message in result.errors)


def test_a_teff_of_zero_is_rejected():
    result = validation.validate_inputs(dict(NORMAL, Teff=0))
    assert any("Teff must be greater than 0" in message for message in result.errors)


def test_a_cme_probability_above_one_is_rejected():
    result = validation.validate_inputs(dict(NORMAL, cme_prob=1.5))
    assert any("cme_prob must be between 0 and 1" in message for message in result.errors)


def test_a_koi_sma_above_the_training_range_is_flagged_but_allowed():
    result = validation.validate_inputs(dict(NORMAL, koi_sma=25))
    assert result.errors == []
    assert any("koi_sma outside training range" in flag for flag in result.flags)


def test_a_big_flare_rate_above_the_all_flare_rate_is_flagged():
    result = validation.validate_inputs(dict(NORMAL, flare_rate_month=5, flare_rate_all=2))
    assert result.errors == []
    assert any("big-flare rate is higher than the all-flare rate" in flag for flag in result.flags)


def test_the_star_type_matches_the_star_type_column_of_every_row(dose_csv):
    wrong = []
    for row in dose_csv.itertuples():
        code = validation.star_type_from_teff(row.Teff)
        if code != int(row.star_type):
            wrong.append((row.star_id, row.planet, row.Teff, int(row.star_type), code))
    assert wrong == []
    assert len(dose_csv) == 1185