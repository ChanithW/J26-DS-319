"""The Stage-1 payload must follow sub03's contract exactly."""
import json

import numpy as np
import pandas as pd
import pytest

from fakes import FakeAssets, fake_analyzer
from flare_cnn.contracts import (STELLAR_PARAM_FIELDS, ContractError, stage1_payload,
                                 validate_stage1_payload)
from flare_cnn.pipeline import analyse


def test_payload_has_contract_fields_and_types():
    p = stage1_payload(fake_analyzer(None, "TIC 123", {}, None, None, 0.6))
    assert p["is_flare"] is True and type(p["is_flare"]) is bool
    assert p["KIC"] is None
    assert isinstance(p["flare_probability"], float) and p["flare_probability"] == 0.97
    assert set(STELLAR_PARAM_FIELDS) <= set(p["stellar_params"])
    assert p["stellar_params"]["Teff"] == 3100.0
    assert p["stellar_params"]["parallax"] == 103.0          # numpy float -> Python float
    assert p["stellar_params"]["phot_g_mean_mag"] is None    # NaN -> None
    assert p["stellar_params"]["Prot"] is None
    d = p["flare_details"]
    assert d["tier_counts"] == {"C": 1, "M": 1, "X": 0, "Superflare": 0}
    assert d["strongest_tier"] == "M" and d["n_weak"] == 1 and len(d["flares"]) == 2
    json.dumps(p)                                            # must be JSON-safe


def test_no_flares_gives_is_flare_false():
    r = fake_analyzer(None, 123, {}, None, None, 0.6)
    r["events"] = r["events"].iloc[0:0]
    p = stage1_payload(r)
    assert p["is_flare"] is False and p["flare_probability"] is None
    assert p["flare_details"]["strongest_tier"] is None


@pytest.mark.parametrize("bad", [
    {},                                  # is_flare missing
    {"is_flare": 1},                     # not a bool
    {"is_flare": True, "KIC": True},     # bool is not a KIC
    {"is_flare": True, "stellar_params": []},
    "not a dict",
])
def test_validation_rejects_bad_payloads(bad):
    with pytest.raises(ContractError):
        validate_stage1_payload(bad)


def test_pipeline_analyse_returns_tables_and_payload():
    out = analyse(FakeAssets(), "123", sector=27, analyzer=fake_analyzer)
    assert len(out["flares"]) == 2
    assert out["summary"]["n_flares"] == 2
    assert "flare_rate_per_day" not in out["summary"]        # flare rate belongs to sub03
    assert out["payload"]["flare_details"]["sector"] == 27
