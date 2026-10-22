# Model card: Component 2, CME occurrence prediction

**Method (wording pending supervisor agreement):** XGBoost on 13 flare-shape features, solar-to-stellar feature alignment, Platt calibration, bootstrap range and class-conditional conformal prediction.

## Intended use
Rank stellar flares by how eruptive-like their brightness curve is, with a reliability flag, as one input to the habitability risk score (Sub-topic 4).

## Training data
719 labelled solar flares (GOES XRS-B, 2-minute averaged, about 712 usable). Stellar flares (33 TESS flares, Gunther et al. 2020) are **not** training examples; they only set the alignment constants and the conformal weights.

## Limitations (read before using any output)
- Trained on the Sun only. Stellar flares have no CME labels, so **accuracy on stars cannot be measured**. Stellar outputs are not validated.
- Skill on the Sun is modest (TSS about 0.1-0.2, AUC about 0.6). Plain accuracy is below the "always confined" baseline (64.9%) by design; report TSS and AUC.
- The calibrated probability is a **ranking score**: its Brier score equals the base-rate baseline. It is not a chance of a CME. The label cut-off is the base rate (about 0.263), not 0.5.
- The 5-95% range reflects variability of the solar training data only, not the solar-to-stellar gap.
- The share called eruptive depends heavily on the alignment choice (6% to 39% in experiments). No stellar CME rate can be read from these outputs.
- Never describe a stellar flare as "a CME" or "a detection". In the research sample 29 of 33 flares were `uncertain` and none had probability above 0.51.
- Retraining does not reproduce the research numbers exactly. XGBoost's output depends on the library version and CPU thread count (a thread change alone shifts raw scores by up to about 0.12), so a retrained model differs from the notebook's by up to about 0.05 in calibrated probability on individual flares. Training here uses one thread so it is repeatable on any machine. Everything after the model (Platt, label rule, conformal reliability) reproduces the notebook's 33-flare table exactly when given the notebook model's raw scores (`tests/test_golden.py`).
- Two input checks were added for the system and were not part of the research runs. `many_features_missing` (more than 5 of the 13 features missing) forces `uncertain` and is on. The **range check** (`out_of_range`: any feature outside the 0.5th-99.5th percentile range of the aligned solar training flares forces `uncertain`) is **switched off by default** (`out_of_range_check: false`; set `OUT_OF_RANGE_CHECK=true` for the API or `--out-of-range-check` for the CLI to turn it on). It was switched off because it flagged 21 of the 33 sample flares, mostly through the yes/no `rebright` feature, which the alignment shifts so that "no re-brightening" falls below the range. With the check off, `out_of_range` is reported as "not checked" (null), and a **confident answer does not mean the flare looks like the training data**: for example `TESS_313934111` is confident but lies above the solar range on `g5` and `g10`. Turning the check on changes only the reliability flag and the warning, never the score, range or label.
- Alignment constants come from only 33 stellar flares.
