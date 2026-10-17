"""Shared setup for the Sub-topic 4 tests.

Puts the sub04 folder on sys.path so the tests can import risk_scoring,
and gives the tests a read-only copy of the training data.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

SUB04 = Path(__file__).resolve().parents[1]
if str(SUB04) not in sys.path:
    sys.path.insert(0, str(SUB04))

DATA_CSV = SUB04 / "data" / "processed" / "training_v12_dose.csv"


@pytest.fixture(scope="session")
def dose_csv():
    """Return the training data with the dose labels. The file is only read."""
    return pd.read_csv(DATA_CSV)