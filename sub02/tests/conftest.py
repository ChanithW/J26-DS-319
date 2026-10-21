import numpy as np
import pandas as pd
import pytest

from component2.config import Config
from component2.features import FEATS13
from component2.train import train_from_tables


def make_table(n, seed, labelled=True, shift=0.0):
    rs = np.random.RandomState(seed)
    y = (rs.rand(n) < 0.35).astype(int)
    d = {}
    for c in FEATS13:
        d[c] = rs.lognormal(1.5, 0.5, n) + 0.5 * y * (c in FEATS13[:6]) + shift
    d["rebright"] = rs.randint(0, 2, n).astype(float)
    df = pd.DataFrame(d)
    if labelled:
        df["flare_id"] = np.arange(n)
        df["ar_number"] = rs.randint(0, max(n // 4, 2), n)
        df["eruptive"] = y
    else:
        df["flare_id"] = [f"TESS_{i}" for i in range(n)]
        df["star_id"] = 1000 + np.arange(n)
    return df


@pytest.fixture(scope="session")
def tables():
    return make_table(400, 1), make_table(33, 2, labelled=False)


@pytest.fixture(scope="session")
def small_cfg():
    return Config(n_bootstrap=6)


@pytest.fixture(scope="session")
def bundle(tables, small_cfg):
    solar, stellar = tables
    b, _ = train_from_tables(solar, stellar, small_cfg)
    return b
