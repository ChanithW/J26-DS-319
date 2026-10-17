"""Checks for loading and checking the model file of Sub-topic 4."""

import pytest

from risk_scoring import config, model_io


def test_the_model_file_loads():
    bundle = model_io.load_model(config.MODEL_PATH)
    assert list(bundle["features"]) == list(config.FEATURES)
    assert list(bundle["classes"]) == list(config.CLASSES)


def test_a_missing_file_raises_a_model_error(tmp_path):
    with pytest.raises(model_io.ModelError):
        model_io.load_model(tmp_path / "not_there.joblib")


def test_a_wrong_feature_order_raises_a_model_error():
    broken = dict(model_io.load_model(config.MODEL_PATH))
    broken["features"] = list(reversed(config.FEATURES))
    with pytest.raises(model_io.ModelError):
        model_io.validate_bundle(broken)


def test_wrong_classes_raise_a_model_error():
    broken = dict(model_io.load_model(config.MODEL_PATH))
    broken["classes"] = ["A", "B", "C"]
    with pytest.raises(model_io.ModelError):
        model_io.validate_bundle(broken)