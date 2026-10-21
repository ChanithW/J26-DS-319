"""SHAP style top drivers for Sub-topic 4.
 
Uses the XGBoost built-in contributions (pred_contribs), not the shap library.
May import pandas and xgboost. No Streamlit, no printing and no logging.
"""

from dataclasses import dataclass, field

import pandas as pd
import xgboost

from . import config, model_io

FEATURE_NAMES = {
    "koi_sma": "Orbital distance",
    "flare_rate_month": "Flares per month at 1e34 erg or more",
    "flare_rate_all": "All flares per month",
    "max_logE": "Strongest flare energy",
    "star_type": "Star type",
}


@dataclass
class Driver:
    """One input and how much it pushed the result for the chosen class."""

    feature: str
    name: str
    value: float
    contribution: float
    direction: str


@dataclass
class Explanation:
    """The drivers for one class, the plain-English sentence and the bias term."""

    target_class: str
    drivers: list = field(default_factory=list)
    sentence: str = ""
    bias: float = 0.0


def _class_axis(contribs) -> int:
    """Return the axis of the contribution array that holds the classes."""
    n_classes = len(config.CLASSES)
    n_terms = len(config.FEATURES) + 1
    if contribs.ndim == 3 and contribs.shape[1] == n_classes and contribs.shape[2] == n_terms:
        return 1
    if contribs.ndim == 3 and contribs.shape[2] == n_classes and contribs.shape[1] == n_terms:
        return 2
    raise ValueError("The model returned contributions with the shape {0}, which does not fit {1} classes and {2} features. Re-save the model with joblib.dump from the notebook that trained it.".format(contribs.shape, n_classes, len(config.FEATURES)))


def _sentence(contributions: list, target_class: str) -> str:
    """Write the plain-English sentence about the drivers of the result."""
    named = [(FEATURE_NAMES[name], contribution) for name, contribution in zip(config.FEATURES, contributions)]
    positive = sorted([item for item in named if item[1] > 0], key=lambda item: -item[1])
    negative = sorted([item for item in named if item[1] < 0], key=lambda item: item[1])
    if len(positive) >= 2:
        text = "{0} and {1} pushed this result toward {2}.".format(positive[0][0], positive[1][0], target_class)
    elif len(positive) == 1:
        text = "{0} pushed this result toward {1}.".format(positive[0][0], target_class)
    else:
        text = "No single input pushed this result strongly toward {0}.".format(target_class)
    if positive and negative and abs(negative[0][1]) >= 0.25 * positive[0][1]:
        text += " {0} pushed it away.".format(negative[0][0])
    return text


def explain(values: dict, target_class: str) -> Explanation:
    """Return the top drivers, the plain-English sentence and the bias for the chosen class."""
    if target_class not in config.CLASSES:
        raise ValueError("Unknown class {0}. Use one of {1}.".format(target_class, ", ".join(config.CLASSES)))

    missing = [name for name in config.FEATURES if name not in values]
    if missing:
        raise ValueError("The values are missing {0}. Add {0} before asking for an explanation.".format(", ".join(missing)))

    bundle = model_io.load_model(config.MODEL_PATH)
    booster = bundle["model"].get_booster()
    row = {name: values[name] for name in config.FEATURES}
    contribs = booster.predict(xgboost.DMatrix(pd.DataFrame([row]), feature_names=list(config.FEATURES)), pred_contribs=True)

    axis = _class_axis(contribs)
    index = config.CLASSES.index(target_class)
    block = contribs[0, index, :] if axis == 1 else contribs[0, :, index]
    contributions = [float(value) for value in block[:len(config.FEATURES)]]
    bias = float(block[-1])

    drivers = []
    for name, contribution in zip(config.FEATURES, contributions):
        direction = "toward" if contribution > 0 else ("away" if contribution < 0 else "neutral")
        drivers.append(Driver(feature=name, name=FEATURE_NAMES[name], value=row[name], contribution=contribution, direction=direction))
    drivers.sort(key=lambda driver: abs(driver.contribution), reverse=True)

    return Explanation(target_class=target_class, drivers=drivers[:5], sentence=_sentence(contributions, target_class), bias=bias)
