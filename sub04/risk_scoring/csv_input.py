"""CSV input for Sub-topic 4: column names, conversions and per-row scoring.

Plain Python and pandas only: no Streamlit, no printing, no logging.
The file is read at prediction time only: nothing is stored and no row is
added to the training data.
"""

from dataclasses import dataclass

import pandas as pd

from . import config, predictor, validation

# The four model inputs are the first four names of config.FEATURES. The fifth
# feature (star_type) comes from the temperature column instead.
REQUIRED_COLUMNS = [name for name in config.FEATURES if name != "star_type"] + ["teff"]

# The temperature column may be written in either case.
TEMPERATURE_COLUMN = "teff"

# Alternative column names and the column they are converted into.
ALTERNATIVE_COLUMNS = {
    "flare_rate_per_day": "flare_rate_month",
    "median_logE": "max_logE",
    "max_logE_median": "max_logE",
}

# A per-day flare rate becomes a per-month rate with this factor.
DAYS_PER_MONTH = 30.44

# The five inputs of the results table: the CSV column and the name the app uses.
INPUT_COLUMNS = [
    ("koi_sma", "koi_sma"),
    ("flare_rate_month", "flare_rate_month"),
    ("flare_rate_all", "flare_rate_all"),
    ("max_logE", "max_logE"),
    ("teff", "Teff"),
]


@dataclass
class ScoredRow:
    """One CSV row after scoring: the inputs, the status, the reason and the result."""

    number: int
    star_id: str
    values: dict
    result: predictor.PredictionResult | None = None
    status: str = "Not scored"
    reason: str = ""


def accepted_columns_text():
    """Return one line with every accepted column name and what it is used for."""
    return (
        "Required: {0} (the temperature column may also be written as Teff). "
        "Optional: cme_prob (empty means the default from config), star_id (display only). "
        "Alternative names, converted automatically: flare_rate_per_day (becomes flare_rate_month, value x {1}), "
        "median_logE or max_logE_median (becomes max_logE)."
    ).format(", ".join(REQUIRED_COLUMNS), DAYS_PER_MONTH)


def temperature_column(frame):
    """Return the temperature column of the frame, matched without the case, or None."""
    for column in frame.columns:
        if str(column).strip().lower() == TEMPERATURE_COLUMN:
            return column
    return None


def prepare(frame):
    """Return the frame with the alternative column names converted, and one note per conversion."""
    prepared = frame.copy()
    notes = []

    column = temperature_column(prepared)
    if column is not None and column != TEMPERATURE_COLUMN:
        prepared = prepared.rename(columns={column: TEMPERATURE_COLUMN})
        notes.append(
            "{0} -> {1} (the temperature column is matched without the case)".format(column, TEMPERATURE_COLUMN)
        )

    for alternative, canonical in ALTERNATIVE_COLUMNS.items():
        if alternative not in prepared.columns or canonical in prepared.columns:
            continue
        if canonical == "flare_rate_month":
            # A cell that is not a number is kept as it is, so the validation
            # can tell the user that the value is not a number.
            numbers = pd.to_numeric(prepared[alternative], errors="coerce")
            converted = numbers * DAYS_PER_MONTH
            prepared[canonical] = converted.where(converted.notna(), prepared[alternative])
            notes.append("{0} -> {1} (value x {2})".format(alternative, canonical, DAYS_PER_MONTH))
        else:
            prepared[canonical] = prepared[alternative]
            notes.append("{0} -> {1}".format(alternative, canonical))

    return prepared, notes


def missing_columns(frame):
    """Return the required columns that the frame does not have. Nothing is filled in."""
    columns = list(frame.columns)
    if temperature_column(frame) is not None:
        columns.append(TEMPERATURE_COLUMN)
    missing = [name for name in REQUIRED_COLUMNS if name not in columns]
    return missing


def score_rows(frame):
    """Score every row of the frame and return one ScoredRow per row."""
    rows = []
    temperature = temperature_column(frame)
    for number, (_, row) in enumerate(frame.iterrows(), start=1):
        raw = {
            "koi_sma": _cell(row.get("koi_sma")),
            "flare_rate_month": _cell(row.get("flare_rate_month")),
            "flare_rate_all": _cell(row.get("flare_rate_all")),
            "max_logE": _cell(row.get("max_logE")),
            "Teff": _cell(row.get(temperature)) if temperature is not None else "",
            "cme_prob": _cell(row.get("cme_prob", "")),
        }
        star_id = str(_cell(row.get("star_id", ""))).strip()
        check = validation.validate_inputs(raw)
        if check.errors:
            rows.append(
                ScoredRow(
                    number=number,
                    star_id=star_id,
                    values=_input_values(row, None, temperature),
                    status="Not scored",
                    reason=" ".join(check.errors),
                )
            )
            continue
        result = predictor.predict(raw)
        if not result.ok:
            rows.append(
                ScoredRow(
                    number=number,
                    star_id=star_id,
                    values=_input_values(row, None, temperature),
                    status="Not scored",
                    reason=" ".join(result.errors),
                )
            )
            continue
        rows.append(
            ScoredRow(
                number=number,
                star_id=star_id,
                values=_input_values(row, result, temperature),
                result=result,
                status=result.ml_class,
                reason="",
            )
        )
    return rows


def results_table(rows):
    """Return the results table for the scored rows."""
    has_star_id = any(row.star_id for row in rows)
    records = []
    for row in rows:
        record = {}
        if has_star_id:
            record["star_id"] = row.star_id
        for name, _ in INPUT_COLUMNS:
            record[name] = row.values.get(name, "")
        record["status"] = row.status
        record["R_eff"] = _r_eff(row)
        record["formula_class"] = row.result.formula.label if row.result is not None else ""
        record["model_class"] = row.result.ml_class if row.result is not None else ""
        record["reason"] = row.reason
        records.append(record)
    return pd.DataFrame(records)


def _cell(value):
    """Return the cell as a plain value: an empty string when the cell is empty."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(value, str):
        return value.strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def _input_values(row, result, temperature):
    """Return the five inputs of a row for the results table."""
    values = {}
    for name, app_name in INPUT_COLUMNS:
        if result is not None and app_name in result.values:
            values[name] = result.values[app_name]
        else:
            column = temperature if name == TEMPERATURE_COLUMN else name
            values[name] = _cell(row.get(column))
    return values


def _r_eff(row):
    """Return the R_eff of a scored row, or an empty string when the row was not scored."""
    if row.result is None or row.result.formula is None:
        return ""
    return float("{0:.6g}".format(row.result.formula.r_eff))
