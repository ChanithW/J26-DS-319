"""Checks for the SHAP top drivers of Sub-topic 4."""

import pandas as pd
import pytest
import xgboost

from risk_scoring import config, explain, model_io

EXAMPLE = {
    "koi_sma": 0.045,
    "flare_rate_month": 1.8,
    "flare_rate_all": 9.5,
    "max_logE": 34.6,
    "star_type": 3,
}


def _margins(model, values):
    row = {name: values[name] for name in config.FEATURES}
    matrix = xgboost.DMatrix(pd.DataFrame([row]), feature_names=list(config.FEATURES))
    return model.get_booster().predict(matrix, output_margin=True)[0]


def test_the_contributions_add_up_to_the_margin(dose_csv):
    model = model_io.load_model(config.MODEL_PATH)["model"]
    test_rows = dose_csv[dose_csv["split"] == "test"].head(20)
    for row in test_rows.itertuples():
        values = {
            "koi_sma": row.koi_sma,
            "flare_rate_month": row.flare_rate_month,
            "flare_rate_all": row.flare_rate_all,
            "max_logE": row.max_logE,
            "star_type": int(row.star_type),
        }
        margins = _margins(model, values)
        for index, name in enumerate(config.CLASSES):
            explanation = explain.explain(values, name)
            total = sum(driver.contribution for driver in explanation.drivers) + explanation.bias
            assert abs(total - margins[index]) < 1e-4


def test_an_unknown_class_raises_a_value_error():
    with pytest.raises(ValueError):
        explain.explain(EXAMPLE, "Extreme")


def test_the_sentence_names_the_two_largest_positive_drivers():
    explanation = explain.explain(EXAMPLE, "High")
    positive = sorted(
        [driver for driver in explanation.drivers if driver.contribution > 0],
        key=lambda driver: -driver.contribution,
    )
    assert len(positive) >= 2
    assert positive[0].name in explanation.sentence
    assert positive[1].name in explanation.sentence