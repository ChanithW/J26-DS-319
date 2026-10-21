"""Load and validate the XGBoost habitability risk model (Sub-topic 4).
 
The file model/xgb_dose_model.joblib holds a dict with the keys
 "model", "features", "classes" and "max_logE_median". Only "model",
 "features" and "classes" are required to run the app.
 
Every problem is reported as a ModelError with a plain-English message.
"""

import functools
import json
import os

import joblib
import pandas as pd
import xgboost

from . import config

REQUIRED_KEYS = ("model", "features", "classes")


class ModelError(RuntimeError):
    """Raised when the model file is missing, unreadable or does not fit the app."""


def _version_tuple(version_text):
    """Turn a version string such as 3.4.1 into a tuple of integers."""
    try:
        return tuple(int(part) for part in version_text.split("."))
    except (AttributeError, ValueError) as exc:
        raise ModelError("Cannot read the XGBoost version text {0}. Install a normal release of xgboost, for example: pip install xgboost==3.4.1".format(version_text)) from exc


def _installed_xgboost_version():
    """The XGBoost version installed in this environment."""
    return xgboost.__version__


def _saved_xgboost_version(model):
    """The XGBoost version that wrote the model file, as a tuple of integers."""
    try:
        saved_config = json.loads(model.get_booster().save_config())
    except Exception as exc:
        raise ModelError("The model file does not contain a readable XGBoost model: the entry called model is a {0}. Re-save it with joblib.dump from the notebook that trained it.".format(type(model).__name__)) from exc

    version = saved_config.get("version")
    if not version:
        raise ModelError("The model file does not record which XGBoost version saved it, so the version cannot be checked. Re-save it with joblib.dump from the notebook that trained it.")

    return tuple(int(part) for part in version)


def validate_bundle(bundle):
    """Check a loaded model bundle and return it unchanged."""
    if not isinstance(bundle, dict):
        raise ModelError("The model file should contain a dict with the keys {0}, but it contains a {1}. Re-save it with joblib.dump from the notebook that trained it.".format(", ".join(REQUIRED_KEYS), type(bundle).__name__))

    for key in REQUIRED_KEYS:
        if key not in bundle:
            raise ModelError("The model file is missing the key {0}. The file has these keys: {1}. Re-save it with joblib.dump using the keys {2}.".format(key, list(bundle), ", ".join(REQUIRED_KEYS)))

    features = list(bundle["features"])
    if features != list(config.FEATURES):
        raise ModelError("The model was trained on the features {0}, but this app is configured for {1}. The feature names and their order must match, so either re-save the model with the same features as risk_scoring/config.py or fix config.FEATURES.".format(features, list(config.FEATURES)))

    classes = list(bundle["classes"])
    if classes != list(config.CLASSES):
        raise ModelError("The model returns the classes {0}, but this app is configured for {1}. The class names and their order must match, so either re-save the model with the same classes as risk_scoring/config.py or fix config.CLASSES.".format(classes, list(config.CLASSES)))

    model = bundle["model"]
    saved_version = _saved_xgboost_version(model)
    installed_version = _installed_xgboost_version()
    if saved_version != _version_tuple(installed_version):
        raise ModelError("The model file was saved with XGBoost {0}, but XGBoost {1} is installed. XGBoost models cannot be read by a different XGBoost version, so install the matching one with: pip install xgboost=={2}".format(".".join(str(p) for p in saved_version), installed_version, ".".join(str(p) for p in saved_version)))

    row = pd.DataFrame([{name: 0.0 for name in config.FEATURES}])
    probabilities = model.predict_proba(row)
    shape = tuple(getattr(probabilities, "shape", ()))
    if shape != (1, len(config.CLASSES)):
        raise ModelError("The model returned probabilities with the shape {0}, but one row with {1} classes was expected. The model does not match the app configuration.".format(shape, len(config.CLASSES)))

    total = float(probabilities.sum())
    if abs(total - 1.0) > 1e-6:
        raise ModelError("The model returned probabilities that add up to {0:.4f} instead of 1.0, so the model file looks damaged. Re-save it with joblib.dump from the notebook that trained it.".format(total))

    return bundle


@functools.lru_cache(maxsize=None)
def _load_cached(path):
    """Load and validate the bundle at path. The cache keeps it to one read per run."""
    if not os.path.isfile(path):
        raise ModelError("The model file was not found: {0}. Copy xgb_dose_model.joblib into the model folder of the app, or pass an existing path to load_model.".format(path))

    try:
        bundle = joblib.load(path)
    except Exception as exc:
        raise ModelError("The model file at {0} could not be read ({1}: {2}). It may be damaged, or it may have been saved with a different joblib version.".format(path, type(exc).__name__, exc)) from exc

    return validate_bundle(bundle)


def load_model(path=config.MODEL_PATH):
    """Return the validated model bundle. The file is read once and kept in memory."""
    return _load_cached(str(path))


def clear_cache():
    """Forget the cached model so the next load_model() reads the file again."""
    _load_cached.cache_clear()
