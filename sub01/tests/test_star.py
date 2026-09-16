"""Unit tests for star information (no internet needed)."""
import numpy as np

from flare_cnn.star import clean_gaia_id, scale_star_info, star_type, teff_from_colour


def test_star_type_boundaries():
    assert star_type(6500) == "F"
    assert star_type(5800) == "G"
    assert star_type(4500) == "K"
    assert star_type(3100) == "M"
    assert star_type(np.nan) == "unknown"


def test_teff_from_gaia_colour():
    teff, src = teff_from_colour({"gaiabp": 12.0, "gaiarp": 11.18})     # BP-RP = 0.82
    assert abs(teff - 5770) < 10 and "Gaia" in src


def test_teff_falls_back_to_v_minus_k():
    teff, src = teff_from_colour({"Vmag": 11.0, "Kmag": 6.42})           # V-K = 4.58
    assert abs(teff - 3420) < 10 and "V-K" in src


def test_teff_missing_when_no_colour():
    teff, src = teff_from_colour({})
    assert np.isnan(teff) and src == "missing"


def test_gaia_id_kept_as_text():
    assert clean_gaia_id("5853498713190525696") == "5853498713190525696"   # all 19 digits
    assert clean_gaia_id("5853498713190525696.0") == "5853498713190525696"
    assert clean_gaia_id("--") == ""


def test_missing_star_info_uses_training_average():
    mean, std = np.array([4000.0, -0.3, 1.6]), np.array([1000.0, 0.25, 0.35])
    scaled, missing = scale_star_info(np.nan, 0.5, None, mean, std)
    assert missing == ["Teff", "distance"]
    assert scaled[0] == 0 and scaled[2] == 0         # average -> scaled value 0
