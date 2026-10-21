"""Checks for the risk prediction of Sub-topic 4."""

import pandas as pd

from risk_scoring import config, model_io, predictor, validation

RAW = {
    "koi_sma": "",
    "flare_rate_month": "1.8",
    "flare_rate_all": "9.5",
    "max_logE": "34.6",
    "Teff": "3300",
    "cme_prob": "",
}


def _features(row):
    return {
        "koi_sma": row.koi_sma,
        "flare_rate_month": row.flare_rate_month,
        "flare_rate_all": row.flare_rate_all,
        "max_logE": row.max_logE,
        "star_type": validation.star_type_from_teff(row.Teff),
    }


def _raw(row):
    return {
        "koi_sma": row.koi_sma,
        "flare_rate_month": row.flare_rate_month,
        "flare_rate_all": row.flare_rate_all,
        "max_logE": row.max_logE,
        "Teff": row.Teff,
        "cme_prob": row.cme_prob,
    }


def test_predict_agrees_with_the_model_on_every_test_row(dose_csv):
    model = model_io.load_model(config.MODEL_PATH)["model"]
    test_rows = dose_csv[dose_csv["split"] == "test"]
    wrong = []
    for row in test_rows.itertuples():
        result = predictor.predict(_raw(row))
        expected = config.CLASSES[int(model.predict(pd.DataFrame([_features(row)]))[0])]
        if result.ml_class != expected:
            wrong.append((row.star_id, row.planet, result.ml_class, expected))
    assert wrong == []
    assert len(test_rows) == 235


def test_the_probabilities_agree_with_the_model_on_every_test_row(dose_csv):
    model = model_io.load_model(config.MODEL_PATH)["model"]
    test_rows = dose_csv[dose_csv["split"] == "test"]
    wrong = []
    for row in test_rows.itertuples():
        result = predictor.predict(_raw(row))
        probabilities = model.predict_proba(pd.DataFrame([_features(row)]))[0]
        for index, name in enumerate(config.CLASSES):
            if abs(result.probabilities[name] - float(probabilities[index])) > 1e-9:
                wrong.append((row.star_id, row.planet, name, result.probabilities[name]))
    assert wrong == []


def test_a_missing_koi_sma_is_not_ok():
    result = predictor.predict(RAW)
    assert result.ok is False
    assert any("koi_sma" in message for message in result.errors)