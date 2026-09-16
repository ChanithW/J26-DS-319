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
- Input checks (`many_features_missing`, `out_of_range`) were added for the system and were not part of the research runs.
- Alignment constants come from only 33 stellar flares.
