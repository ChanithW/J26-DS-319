# Habitability Risk Scoring (Sub-topic 4)

Project: J26-DS-319
Index: IT23355200
Author: Moremada M M S T

## Scope

This module gives each planet a risk level for atmospheric erosion:
High, Medium or Low.
The risk is calculated from four inputs: flare rate, flare energy,
CME probability and orbital distance.
Every result comes with SHAP explanations, so it is clear which input
pushed the risk up or down.

## Inputs

| Input | Meaning |
| --- | --- |
| `koi_sma` | Semi-major axis in AU (orbital distance). |
| `flare_rate_month` | Flares per month with energy >= 1e34 erg. |
| `flare_rate_all` | All flares per month. |
| `max_logE` | log10 of the strongest flare energy in erg. |
| `star_type` | 0-3, from the star effective temperature (Teff): 0 = above 6000 K, 1 = above 5200 K, 2 = above 3700 K, 3 = below 3700 K. |
| `cme_prob` | Optional, 0-1. Used only in the formula label. |

## Outputs

- ML risk class: High, Medium or Low.
- Class probabilities for each class.
- Formula label, with `R_eff`.
- SHAP top drivers for the result.
- Star type.
- Flags for warnings about the input data.

## Limits

- The score is a physics-based risk score. It is not measured ozone loss.
- The classes are relative to the sample. They compare planets with each
  other and are not absolute risk levels.
- CME probability is a modelled estimate, not an observation.

## Status

- Manual input page and a CSV upload for the results of another sub-topic.
- Star ID lookup in the Catalogue mode: the held-out test stars of the training
  data, with the planet picker for the stars that have more than one planet.
