import json

import numpy as np
import pandas as pd
import pytest

from component2.cli import main
from component2.data_io import DataError, load_solar_labels
from tests.conftest import make_table
from tests.test_pipeline import curve


def write_synthetic_data(root, n=80, seed=0):
    rs = np.random.RandomState(seed)
    (root / "solar_raw").mkdir(parents=True)
    (root / "solar_labels").mkdir()
    rows = []
    for i in range(n):
        peak = pd.Timestamp("2012-01-01") + pd.Timedelta(hours=i)
        eru = int(rs.rand() < 0.35)
        tau = rs.uniform(5, 30) * (1.5 if eru else 1.0)
        t = np.arange(-25 * 60, 95 * 60, 2.0) / 60.0       # minutes, 2-second cadence
        f = np.full(len(t), 1e-6)
        r = (t >= -5) & (t < 0)
        f[r] = 1e-6 + 5e-6 * (t[r] + 5) / 5
        d = t >= 0
        f[d] = 1e-6 + 5e-6 * np.exp(-t[d] / tau)
        f *= rs.normal(1, 0.01, len(t))
        pd.DataFrame({"timestamp": peak + pd.to_timedelta(t, unit="m"), "xrsa": f, "xrsb": f,
                      "xrsa_quality": 0, "xrsb_quality": 0}).to_csv(root / "solar_raw" / f"flare_{i}.csv", index=False)
        rows.append(dict(flare_id=i, flare_key=f"k{i}", peak_time=peak, start_time=peak, end_time=peak,
                         goes_class="M1.0", peak_flux=1e-5, eruptive=eru, lat=0, lon=0,
                         ar_number=i // 3, phi_ar=0, ar_area=0))
    pd.DataFrame(rows).to_csv(root / "solar_labels" / "labels_seed_standardized.csv", index=False)
    (root / "solar_raw" / "flare_9999.csv").write_text("timestamp,xrsa,xrsb,xrsa_quality,xrsb_quality\n")
    st = make_table(33, 3, labelled=False).rename(columns={"star_id": "TIC"})
    st.to_csv(root / "tess_sample_features2.csv", index=False)


def test_missing_label_file_message(tmp_path):
    with pytest.raises(DataError, match="not found"):
        load_solar_labels(tmp_path)


def test_cli_train_and_predict(tmp_path, caplog):
    write_synthetic_data(tmp_path / "data")
    out = tmp_path / "b.joblib"
    cfgfile = tmp_path / "c.yaml"
    cfgfile.write_text("n_bootstrap: 4\n")
    with caplog.at_level("WARNING"):
        assert main(["train", "--data", str(tmp_path / "data"), "--out", str(out), "--config", str(cfgfile)]) == 0
    assert "flare_9999.csv" in caplog.text                 # unlabelled file is named in a warning
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps({"flares": [curve()]}))
    pred = tmp_path / "p.csv"
    ho = tmp_path / "h.csv"
    assert main(["predict", "--bundle", str(out), "--input", str(inp), "--out", str(pred), "--handoff", str(ho)]) == 0
    assert pd.read_csv(pred)["status"].tolist() == ["ok"]
    assert list(pd.read_csv(ho).columns)[:5] == ["flare_id", "star_id", "calibrated_probability",
                                                 "confidence_lower", "confidence_upper"]
    assert main(["info", "--bundle", str(out)]) == 0
    assert main(["evaluate", "--bundle", str(out), "--data", str(tmp_path / "data")]) == 0
