"""Star catalogue lookup for Sub-topic 4: the held-out test stars of the training data.

Plain Python and pandas only: no Streamlit, no printing, no logging.
The processed CSV is only read: nothing is written and no row is added to the
training data. The label columns are carried along for display only; they
never reach the validation, the formula or the model.
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import config

# The processed training data with the dose labels. It holds a split column,
# so the stars the model was not trained on are the test rows of this file.
CATALOGUE_CSV = config.BASE_DIR / "data" / "processed" / "training_v12_dose.csv"

STAR_ID_COLUMN = "star_id"
PLANET_COLUMN = "planet"
SPLIT_COLUMN = "split"
TEST_SPLIT = "test"

# The six values that reach the validation, the formula and the model. The
# star type is not in the list: the app derives it from the temperature.
INPUT_COLUMNS = ("koi_sma", "flare_rate_month", "flare_rate_all", "max_logE", "Teff", "cme_prob")

# The label columns are for display only.
DOSE_LABEL_COLUMN = "label_dose"
MODEL_LABEL_COLUMN = "label"


@dataclass
class CatalogueEntry:
    """One planet of a catalogue star: the six inputs and the two stored labels."""

    star_id: str
    planet: str
    inputs: dict
    dose_class: str = ""
    label: str = ""


def load_catalogue(path=None):
    """Return the catalogue entries of the held-out test stars."""
    csv_path = Path(path) if path is not None else Path(CATALOGUE_CSV)
    frame = pd.read_csv(csv_path)
    if SPLIT_COLUMN not in frame.columns:
        return []
    frame = frame[frame[SPLIT_COLUMN].astype(str).str.strip() == TEST_SPLIT]

    entries = []
    for _, row in frame.iterrows():
        entries.append(
            CatalogueEntry(
                star_id=_text(row[STAR_ID_COLUMN]) if STAR_ID_COLUMN in frame.columns else "",
                planet=_text(row[PLANET_COLUMN]) if PLANET_COLUMN in frame.columns else "",
                inputs={name: row[name] for name in INPUT_COLUMNS if name in frame.columns},
                dose_class=_text(row[DOSE_LABEL_COLUMN]) if DOSE_LABEL_COLUMN in frame.columns else "",
                label=_text(row[MODEL_LABEL_COLUMN]) if MODEL_LABEL_COLUMN in frame.columns else "",
            )
        )
    return entries


def find_star(entries, star_id):
    """Return the entries of one star. The ID is trimmed and matched exactly."""
    wanted = _text(star_id)
    if not wanted:
        return []
    return [entry for entry in entries if entry.star_id == wanted]


def planet_label(entry):
    """Return the label of one planet in the planet picker."""
    if entry.planet:
        return entry.planet
    return format_value(entry.inputs.get("koi_sma", ""))


def raw_inputs(entry):
    """Return the six values of an entry. This is the only data that reaches the model."""
    return {name: entry.inputs.get(name, "") for name in INPUT_COLUMNS}


def input_text(entry):
    """Return the six values of an entry as text, ready for the input fields."""
    return {name: format_value(entry.inputs.get(name, "")) for name in INPUT_COLUMNS}


def values_match(current, loaded):
    """Return True when the current values are the same numbers as the loaded row."""
    for name in INPUT_COLUMNS:
        if not _same_number(current.get(name, ""), loaded.get(name, "")):
            return False
    return True


def format_value(value):
    """Return the value as text for an input field. The text is the same number back."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _text(value):
    """Return the value as plain text, without the spaces around it."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _to_float(value):
    """Return the value as a finite float, or None when it is not a usable number."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _same_number(one, other):
    """Return True when the two values are the same number."""
    left = _to_float(one)
    right = _to_float(other)
    if left is None and right is None:
        return _text(one) == _text(other)
    if left is None or right is None:
        return False
    return left == right
