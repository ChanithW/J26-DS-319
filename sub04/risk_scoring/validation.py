"""Input validation and star type helpers for Sub-topic 4.
 
Plain Python only: math and the dataclasses module, no Streamlit, no pandas, no numpy.
Every number that also lives in config.py is read from config.py.
"""

import math
from dataclasses import dataclass, field

from . import config

REQUIRED_FIELDS = {
    "koi_sma": ("the orbital distance in AU", "greater than 0"),
    "flare_rate_month": ("the number of big flares per month", "zero or more"),
    "flare_rate_all": ("the number of all flares per month", "zero or more"),
    "max_logE": ("the log10 of the strongest flare energy in erg", "any real number"),
    "Teff": ("the star effective temperature in K", "greater than 0"),
}

RANGE_FIELDS = ("koi_sma", "flare_rate_month", "flare_rate_all", "max_logE", "Teff", "cme_prob")


@dataclass
class ValidationResult:
    """Cleaned values, blocking errors, non-blocking flags and whether cme_prob was defaulted."""

    values: dict = field(default_factory=dict)
    errors: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    cme_defaulted: bool = False


def star_type_from_teff(teff: float) -> int:
    """Map an effective temperature to the star type code 0 (F), 1 (G), 2 (K) or 3 (M)."""
    if teff > config.TEFF_CUTS[0]:
        return 0
    if teff > config.TEFF_CUTS[1]:
        return 1
    if teff > config.TEFF_CUTS[2]:
        return 2
    return 3


def star_type_letter(code: int) -> str:
    """Return the spectral letter F, G, K or M for a star type code."""
    try:
        return config.STAR_TYPE_LETTERS[code]
    except KeyError:
        raise ValueError("Unknown star type code {0}. Use 0 (F), 1 (G), 2 (K) or 3 (M).".format(code)) from None


def _to_float(value):
    """Return the value as a finite float, or None when it is not a usable number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _format_number(value: float) -> str:
    """Format a number for a message, with at most 6 significant digits."""
    return "{0:.6g}".format(value)


def _number_error(name, raw_value, meaning, rule) -> str:
    """Return the plain-English error for a value that is not a usable number."""
    if raw_value is None or (isinstance(raw_value, str) and raw_value.strip() == ""):
        return "{0} is required. Enter {1} ({2}).".format(name, meaning, rule)
    if isinstance(raw_value, float) and (math.isnan(raw_value) or math.isinf(raw_value)):
        return "{0} must be a real number, not NaN or infinity. Enter {1}.".format(name, meaning)
    return "{0} must be a number. Enter {1} ({2}).".format(name, meaning, rule)


def validate_inputs(raw: dict) -> ValidationResult:
    """Check the raw inputs and return the cleaned values, the blocking errors and the flags."""
    raw = raw or {}
    result = ValidationResult()

    for name, (meaning, rule) in REQUIRED_FIELDS.items():
        number = _to_float(raw.get(name))
        if number is None:
            result.errors.append(_number_error(name, raw.get(name), meaning, rule))
            continue
        if name == "koi_sma" and number <= 0:
            result.errors.append("koi_sma must be greater than 0. Enter the orbital distance in AU.")
            continue
        if name == "Teff" and number <= 0:
            result.errors.append("Teff must be greater than 0. Enter the star effective temperature in K.")
            continue
        if name in ("flare_rate_month", "flare_rate_all") and number < 0:
            result.errors.append("{0} must be zero or more. Enter the number of flares per month.".format(name))
            continue
        result.values[name] = number

    cme_raw = raw.get("cme_prob")
    if cme_raw is None or (isinstance(cme_raw, str) and cme_raw.strip() == ""):
        result.values["cme_prob"] = config.DEFAULT_CME_PROB
        result.cme_defaulted = True
        result.flags.append("CME probability default {0} used".format(config.DEFAULT_CME_PROB))
    else:
        cme = _to_float(cme_raw)
        if cme is None:
            result.errors.append("cme_prob must be a number between 0 and 1 (probability of a coronal mass ejection).")
        elif cme < 0 or cme > 1:
            result.errors.append("cme_prob must be between 0 and 1 (probability of a coronal mass ejection). Enter a value such as {0}.".format(config.DEFAULT_CME_PROB))
        else:
            result.values["cme_prob"] = cme

    if "Teff" in result.values:
        result.values["star_type"] = star_type_from_teff(result.values["Teff"])

    any_out_of_range = False
    for name in RANGE_FIELDS:
        if name not in result.values:
            continue
        if name == "cme_prob" and result.cme_defaulted:
            continue
        low, high = config.TRAIN_RANGES[name]
        if result.values[name] < low or result.values[name] > high:
            any_out_of_range = True
            result.flags.append("{0} outside training range ({1} to {2})".format(name, _format_number(low), _format_number(high)))

    if "flare_rate_month" in result.values and "flare_rate_all" in result.values:
        if result.values["flare_rate_month"] > result.values["flare_rate_all"]:
            result.flags.append("big-flare rate is higher than the all-flare rate, check the inputs")

    if not result.errors and not any_out_of_range:
        result.flags.append("All inputs inside training range")

    return result
