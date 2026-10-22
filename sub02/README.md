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

## Run it locally (backend + web app)
Needs Python 3.11+, Node 20+, the data in `data/`, and a trained bundle.
```bash
pip install -e ".[dev]"
python -m component2.cli train --data data/ --out model_bundle_v3.joblib   # once
cp .env.example .env                                                        # optional settings
python -m uvicorn component2.api:app --port 8000                            # API, docs at http://localhost:8000/docs
cd frontend && npm install && cp .env.example .env && npm run dev          # web app at http://localhost:5173
```
`make setup`, `make train`, `make dev` (both servers) and `make test` do the same if you have `make`.
After any API change run `make openapi` (rewrites `docs/openapi.json` and the frontend types); a test fails if the committed schema is stale.

Frontend tests: `cd frontend && npm test` (Vitest + MSW). One end-to-end test (`npm run test:e2e`) drives the real backend in a browser; it uses the Microsoft Edge installed on Windows by default (`PW_CHANNEL=chromium` uses Playwright's own browser after `npx playwright install chromium`).

## Entering flares
Both web apps accept flares three ways: a long-format CSV, a manual form, or the 33 sample TESS flares. The manual form takes a flare id, the star (TIC) id, and pasted "time, flux" pairs (one per line, from a spreadsheet or typed). Time can be given as minutes from the flare peak (no peak time needed) or as TESS days (BTJD) with the peak time. Several flares can be added to a list and run together. The form warns if fewer than 20 points fall in the −20 to +90 minute window; such a flare is then returned as `too_few_points` rather than scored. The "demo flare" button fills in a synthetic example, only to try the form.

## Streamlit UI
A Streamlit front end (`streamlit_app/`) offers the same pages as the React app: Predict, Results (table plus flare detail), Saved runs and Model info. It talks only to the API, so start the API first.
```bash
pip install -e ".[ui]"
python -m uvicorn component2.api:app --port 8000      # terminal 1
python -m streamlit run streamlit_app/app.py          # terminal 2, opens http://localhost:8501
```
Set `COMPONENT2_API_URL` if the API is elsewhere. Tests: `python -m pytest tests/test_ui.py` (API mocked). Differences from the React app: no keyboard arrow-key navigation in the flare detail (use the selector and Previous/Next buttons), and the limitations banner is at the top of each page rather than pinned while scrolling.
The two front ends are independent; keep one and delete the other folder (`frontend/` or `streamlit_app/`) if you don't need both.

## Why the label is not "probability above 0.5"
Only about 35% of flares are eruptive, so calibrated probabilities are mostly below 0.5. `predicted_label` is `eruptive` when `calibrated_probability >= base` (about 0.263, the eruptive share of the calibration set). Set `label_rule: raw_0.5` in the config for the older rule.

## Reliability flag and the optional range check
`reliability` is `confident` when the conformal check leaves exactly one answer (and no more than 5 features are missing); otherwise `uncertain`. An extra safety check, `out_of_range_check`, can also force `uncertain` for flares with a feature outside the solar training range. It is not part of the research and is **off by default** (`config/default.yaml`; `OUT_OF_RANGE_CHECK=true` for the API; `component2 predict --out-of-range-check`). While it is off, `out_of_range` is `null` ("not checked") and a confident answer says nothing about whether the flare resembles the training flares.

## Status values
`ok`, `too_few_points` (<20 points in the window), `no_peak`, `invalid_input`. Non-`ok` rows have empty values.

## Tests
```bash
pytest
```
Most tests use synthetic data. Two golden tests compare with the notebook's 33-flare table (`tests/golden/`):
- `test_golden_downstream_exact` runs frozen notebook-model scores (`frozen_scores_v3.json`) through the package's Platt, label and conformal code and must match exactly.
- `test_golden_retrained_agrees` retrains on the real `data/` (skipped if absent) and checks looser agreement, because XGBoost output varies with library version and thread count (see MODEL_CARD).

## Open items
Hand-off columns for Sub-topic 4 and input columns from Sub-topic 1 are unconfirmed (see `handoff.py`, `schemas.py`).
