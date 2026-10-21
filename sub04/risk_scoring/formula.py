"""Formula label and R_eff for Sub-topic 4.
 
Plain Python only: dataclasses and config.py, no Streamlit, no pandas, no numpy.
"""

from dataclasses import dataclass

from . import config


@dataclass
class FormulaResult:
    """The R_eff value, its label, the CME probability used and a note about a default."""

    r_eff: float
    label: str
    cme_prob: float
    cme_defaulted: bool
    note: str


def r_eff(flare_rate_month: float, cme_prob: float, koi_sma: float) -> float:
    """Return the R_eff value, the relative atmospheric escape at the planet."""
    if koi_sma <= 0:
        raise ValueError("koi_sma must be greater than 0. Enter the orbital distance in AU.")
    return flare_rate_month * cme_prob * (config.D_REF / koi_sma) ** 2


def formula_label(r: float) -> str:
    """Return the label High, Medium or Low for an R_eff value."""
    # config.CLASSES is [Low, Medium, High], so index 2 is High, 1 is Medium and 0 is Low
    if r >= config.THRESH_HIGH:
        return config.CLASSES[2]
    if r >= config.THRESH_MEDIUM:
        return config.CLASSES[1]
    return config.CLASSES[0]


def formula_result(values: dict, cme_defaulted: bool) -> FormulaResult:
    """Return the R_eff result for the cleaned input values."""
    cme_prob = values["cme_prob"]
    r = r_eff(values["flare_rate_month"], cme_prob, values["koi_sma"])
    note = "CME probability default {0} used (number taken from config)".format(config.DEFAULT_CME_PROB) if cme_defaulted else ""
    return FormulaResult(r_eff=r, label=formula_label(r), cme_prob=cme_prob, cme_defaulted=cme_defaulted, note=note)
