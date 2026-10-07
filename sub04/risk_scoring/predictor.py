"""ML risk class, probabilities and flags for Sub-topic 4.
 
May import pandas. No Streamlit, no printing and no logging.
"""

from dataclasses import dataclass, field

import pandas as pd

from . import config, formula, model_io, validation


@dataclass
class PredictionResult:
    """The outcome of one prediction: class, probabilities, formula label, flags and errors."""

    ok: bool
    errors: list = field(default_factory=list)
    flags: list = field(default_factory=list)
    ml_class: str | None = None
    probabilities: dict = field(default_factory=dict)
    formula: formula.FormulaResult | None = None
    star_type: int | None = None
    star_type_letter: str | None = None
    values: dict = field(default_factory=dict)
    cme_defaulted: bool = False


def predict(raw: dict) -> PredictionResult:
    """Return the ML class, the probabilities, the formula label and the flags for the raw inputs."""
    check = validation.validate_inputs(raw)
    result = PredictionResult(ok=not check.errors, errors=list(check.errors), flags=list(check.flags), values=dict(check.values), cme_defaulted=check.cme_defaulted)

    if "Teff" in check.values:
        result.star_type = check.values["star_type"]
        result.star_type_letter = validation.star_type_letter(result.star_type)

    if check.errors:
        return result

    try:
        bundle = model_io.load_model(config.MODEL_PATH)
    except model_io.ModelError as exc:
        result.ok = False
        result.errors.append(str(exc))
        return result

    model = bundle["model"]
    class_codes = [int(code) for code in model.classes_]
    if class_codes != [0, 1, 2] or len(config.CLASSES) != 3:
        result.ok = False
        result.errors.append("The model returns the class codes {0}, but this app expects [0, 1, 2] for {1}. Re-save the model with the same class order as risk_scoring/config.py.".format(class_codes, ", ".join(config.CLASSES)))
        return result

    row = {name: check.values[name] for name in config.FEATURES}
    probabilities = model.predict_proba(pd.DataFrame([row]))[0]

    result.probabilities = {config.CLASSES[index]: float(probabilities[index]) for index in range(len(config.CLASSES))}
    # config.CLASSES runs Low, Medium, High, so the larger index is the higher risk class and wins a tie
    best = max(range(len(probabilities)), key=lambda index: (probabilities[index], index))
    result.ml_class = config.CLASSES[class_codes[best]]

    result.formula = formula.formula_result(check.values, check.cme_defaulted)
    if result.ml_class != result.formula.label:
        result.flags.append("The two labels use different methods and can differ. See the limits section.")

    return result
