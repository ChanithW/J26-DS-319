# Component 2: CME occurrence prediction (IT4010, J26-DS-319)

Reads a stellar flare's brightness curve (TESS, 2-minute) and outputs a ranking score for "eruptive" vs "confined", a 5-95% range and a reliability flag. Trained on 719 labelled **solar** flares.

> **Honesty notes.** Stellar flares have no CME labels, so accuracy on stars cannot be measured and stellar outputs are not validated. The calibrated probability is a ranking score, not a chance of a CME. The range covers solar training variability only. No stellar CME rate can be read from the outputs. See [MODEL_CARD.md](MODEL_CARD.md).

## Install
```bash
pip install -e ".[dev]"        # add ,fetch for Lightkurve support
```

## Data (git-ignored, copy from Drive `subtopic2_data/`)
```
data/solar_raw/flare_{id}.csv   data/solar_labels/labels_seed_standardized.csv
data/tess_raw/tic_{TIC}.csv     data/tess_sample_features2.csv
```
Raw solar files without a label are ignored with a warning.

## Use
```bash
component2 train --data data/ --out model_bundle_v3.joblib
component2 predict --bundle model_bundle_v3.joblib --input flares.json --out predictions.csv --handoff handoff.csv
component2 evaluate --bundle model_bundle_v3.joblib --data data/
component2 info --bundle model_bundle_v3.joblib
component2 refit-alignment --data data/ --stellar more_stellar_features.csv --out model_bundle_v3.joblib
```
API: `COMPONENT2_BUNDLE=model_bundle_v3.joblib uvicorn component2.api:app` (`POST /predict`, `GET /health`, `GET /model-info`).

`flares.json` is a list (or `{"flares": [...]}`) of `{flare_id, star_id, tpeak, time, flux}`; `time` and `tpeak` in BTJD days.

## Why the label is not "probability above 0.5"
Only about 35% of flares are eruptive, so calibrated probabilities are mostly below 0.5. `predicted_label` is `eruptive` when `calibrated_probability >= base` (about 0.263, the eruptive share of the calibration set). Set `label_rule: raw_0.5` in the config for the older rule.

## Status values
`ok`, `too_few_points` (<20 points in the window), `no_peak`, `invalid_input`. Non-`ok` rows have empty values.

## Tests
```bash
pytest
```
Tests use synthetic data. The golden test (`tests/golden/handoff_v3_expected.csv` + real `data/`) is skipped until both exist.

## Open items
Hand-off columns for Sub-topic 4 and input columns from Sub-topic 1 are unconfirmed (see `handoff.py`, `schemas.py`).
