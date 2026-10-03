# Sub-topic 3 — Stellar flare frequency & energy prediction

Part of **J26-DS-319: AI-Driven Stellar Flare Analysis and Exoplanetary Habitability Impact Assessment**.

Given a star that Stage 1 has confirmed as **flaring**, this stage predicts

* **flare frequency** – flares per day, and
* **flare energy** – median flare energy, log10(E / erg),

each as a point estimate with an **80 % conformal prediction interval**. The outputs are passed on to
Sub-topic 2 (CME prediction) and Sub-topic 4 (habitability risk).

```
Stage 1  Sub-topic 1  flare / not flare ──► Stage 2  Sub-topic 3  (this folder)
                                               │  rate + energy with intervals
                                               ▼
                                         Stage 3  Sub-topic 2  CME ──► Stage 4  Sub-topic 4  habitability
```

---

## 1. Folder layout

| Path | What it is |
|---|---|
| `flare_model.pkl` | Trained model bundle (**do not modify / retrain**) |
| `data/stellar_flare_database_v3.csv` | 2,910-star dataset |
| `subtopic3/config.py` | Paths and constants (spectral thresholds, Teff limits, split settings) |
| `subtopic3/contracts.py` | **Stage interface contract** (field names, validation) |
| `contracts/stage_contract.json` | The same contract as JSON, for anyone not using Python |
| `subtopic3/predictor.py` | Inference: input validation, preprocessing, prediction |
| `subtopic3/pipeline.py` | Gating on `is_flare`, KIC lookup, Stage-1 → Stage-3 glue |
| `subtopic3/ui.py` | Streamlit page for Sub-topic 3 (`render()`) |
| `app/Home.py` | Standalone Streamlit app (Sub-topic 3 only) |
| `scripts/batch_export.py` | Writes `outputs/subtopic3_predictions.csv` for Sub-topic 4 |
| `scripts/evaluate.py` | Test-set metrics and plots, written to `outputs/` |
| `tests/` | pytest suite |

## 2. Setup

The pickle was saved with **xgboost 3.4.1** and **lightgbm 4.6.0**. xgboost 3.4.1 has no wheel for
Python 3.11, so use **Python 3.12+** (developed with 3.13). From the `sub03` folder (Windows cmd):

```bat
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Running

All commands are run from `sub03/` with the venv activated.

| Task | Command |
|---|---|
| Standalone app | `streamlit run app/Home.py` |
| Static predictions for the whole dataset | `python scripts/batch_export.py` |
| Evaluation (metrics + plots) | `python scripts/evaluate.py` |
| Tests | `python -m pytest tests -q` |
| Regenerate the JSON contract | `python -m subtopic3.contracts` |

**Using the app** (Sub-topic 3 only; the other sub-topics are integrated later):

* **Single star** tab: either choose a KIC from the dataset (parameters are auto-filled and the observed
  Kepler values are shown next to the prediction), or enter the stellar parameters manually. Click
  *Predict* to get the flare rate and median flare energy with 80 % intervals, an interval plot, the
  bias-flag explanation for F stars, and a downloadable JSON result.
* **Batch CSV** tab: upload a CSV of stars and download their predictions.

## 4. Stage interface contract (for integration later)

The standalone app does not use this yet. It is defined now so the four sub-topics can be joined later
without renaming anything. In the integrated app, everything is held in `st.session_state["pipeline"]`, a dict with one key per stage. Field definitions
live in `subtopic3/contracts.py` (Python) and `contracts/stage_contract.json`. **Field names are stable.
Add new optional fields; do not rename existing ones.**

### Input — what Sub-topic 1 writes to `st.session_state["pipeline"]["stage1"]`

| Field | Type | Required | Meaning |
|---|---|---|---|
| `is_flare` | bool | **yes** | Stage-1 verdict. Sub-topic 3 runs **only** if `True`. |
| `KIC` | int / None | no | Kepler ID. If it exists in the dataset, its parameters are auto-filled. |
| `stellar_params` | dict / None | no | `Teff, logg, Prot, radius_gspphot, mh_gspphot, bp_rp, parallax, phot_g_mean_mag` (+ optional `spectral_type`). Used when the KIC is unknown. |
| `flare_probability` | float / None | no | Classifier confidence (displayed only). |
| `flare_details` | dict / None | no | Anything else about the event (displayed only). |

### Output — what Sub-topic 3 writes to `st.session_state["pipeline"]["stage3"]`

| Field | Type | Meaning |
|---|---|---|
| `KIC` | int / None | Star ID (None if parameters were entered manually) |
| `spectral_type` | str | `F`, `G`, `K` or `M` |
| `pred_flare_rate` | float | Predicted flare rate [flares/day] |
| `pred_flare_rate_lower` / `_upper` | float | 80 % interval [flares/day] |
| `pred_logE` | float | Predicted median flare energy, log10(E/erg) |
| `pred_logE_lower` / `_upper` | float | 80 % interval, log10(erg) |
| `detection_bias_flag` | bool | `True` for F-type stars (see §6) |
| `interval_level` | float | Nominal interval coverage (0.80), optional |

Guaranteed: `lower ≤ point ≤ upper` for both quantities, all values finite, rate bounds > 0.
`validate_stage3_output(payload)` checks this.

### Gating

`subtopic3.pipeline.run_stage3(stage1, predictor)` refuses to predict (raises `GatingError`) if there is
no Stage-1 result, it is malformed, or `is_flare` is not `True`. This is tested, and it is ready for the
integrated app.

## 5. Integrating with the other sub-topics (later)

* **Sub-topic 1 → 3:** Sub-topic 1 writes its result (input contract above) to
  `st.session_state["pipeline"]["stage1"]`. Sub-topic 3 calls
  `run_stage3(stage1, predictor)`, which applies the gate, auto-fills a known KIC and predicts.
* **Sub-topic 3 → 2 / 4:** store the returned dict in `st.session_state["pipeline"]["stage3"]`.
  Sub-topic 4 can also use the static table `outputs/subtopic3_predictions.csv` (same columns).
* Without Streamlit: `FlarePredictor().predict_one(params, kic)` returns `(output_dict, warnings)`.

## 6. Model details

* **Bundle:** `log_flares_per_day` → XGBoost; `median_logE` → LightGBM. One pooled model per target,
  with `spectral_code` (F=0, G=1, K=2, M=3) as a feature.
* **Features (in order):** `Teff, logg, Prot, spectral_code, radius_gspphot, mh_gspphot, bp_rp,
  parallax, phot_g_mean_mag`. No scaling or encoding is bundled.
* **Ensembles:** 5 point models (squared error), 5 lower (q = 0.10) and 5 upper (q = 0.90) quantile
  models. Predictions are averaged over the 5.
* **Intervals (CQR):** `lower = lo − q̂[type]`, `upper = hi + q̂[type]`, using the per-spectral-type
  conformal corrections stored in the pickle. The result is then clipped so that lower ≤ point ≤ upper.
* **Point estimate:** the point models minimise squared error, so the point value is the conditional
  **mean** on the log scale, not the 0.5 quantile.
* **Back-transform:** the rate is predicted as log10(flares/day). `pred_flare_rate = 10**x`
  (`flares_per_day = Nf / 1459 d` Kepler baseline).
* **Loading note:** with xgboost 3.4.1 on Windows, xgboost's own unpickling rejects the boosters'
  saved buffers ("input stream corrupted"). `predictor.py` falls back to loading the buffer's `Model`
  section (XGBoost's official model format) **in memory**. The `.pkl` file is never changed.
  `scripts/evaluate.py` confirms that the results are identical to training.
* **Missing values:** passed as NaN and handled natively by XGBoost/LightGBM, exactly as in training.
  `Teff` and `logg` are required (they were never missing in training).
* **Spectral type** (Pecaut & Mamajek 2013): F ≥ 6000 K, G ≥ 5300 K, K ≥ 3900 K, M < 3900 K.
  The type is derived from Teff. If a supplied `spectral_type` disagrees with Teff, the input is rejected.
* **Valid domain:** 2600 K ≤ Teff < 7500 K. A-type and hotter stars (≥ 7500 K, dropped in training)
  and stars cooler than the coolest training star are rejected. A warning is shown for logg < 4
  (training used dwarfs only).

### Detection-bias flag

`detection_bias_flag = True` **only for F-type stars**. Kepler's targeting favoured Sun-like stars, and
flares are hard to detect against bright, hot F stars, so only the strongest flarers make the catalogue.
F-type rates and energies therefore look inflated compared with M dwarfs. This is an observational
artefact, not physics. No other flag rules are applied.

## 7. Evaluation

`scripts/evaluate.py` re-creates the original split (20 % test, stratified by `spectral_type`,
`random_state = 42`). It **checks** that the test RMSE matches the training results (rate 0.388,
energy 0.297) and refuses to save anything if it does not. It reports, overall and per spectral type:
RMSE, R², pinball loss at q = 0.10 / 0.90, PICP (coverage) and MPIW (mean interval width). Results go to
`outputs/metrics_overall.csv`, `metrics_by_type.csv`, `metrics.json`, `pred_vs_obs_intervals.png` and
`metrics_by_type.png`. Rate metrics are on the log10 scale, as in training.

## 8. Limitations

* **Kepler selection bias:** the sample is not volume-limited, so F-type results are inflated (see the flag).
* **F, G, K, M dwarfs only:** no O/B/A stars, giants are out of domain, and Teff must be in 2600–7499 K.
* **Noisy targets for low flare counts:** stars with fewer than 5 detected flares have much less
  reliable targets. The training robustness table shows rate R² < 0 and interval coverage of about 0.42
  for these stars.
* **Missing Gaia parameters** (about 33–40 % of stars) widen the error. The rate R² is about 0.41
  without Gaia parameters and about 0.66 with them.
* Intervals are **marginal** 80 % coverage (per spectral type), not guaranteed for each individual star.
* The model predicts the *median* flare energy, not the maximum.
