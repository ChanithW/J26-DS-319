"""flare_cnn: TESS flare detection and energy-tier classification (Component 1).

A two-head 1D-CNN that finds stellar flares in TESS light curves and gives each
flare an energy tier (C / M / X / Superflare) in a single forward pass.

Modules
-------
config    load configs/config.yaml and the saved model settings
features  turn raw light-curve windows into the model's 2 input channels
energy    flare energy (ED x luminosity) and energy tiers
star      star information from the TESS Input Catalog (with a colour fallback)
model     build and load the two-head CNN
detect    scan a whole light curve and return one row per flare
export    save results as CSV files
contracts the hand-off payload for Stage 2 (same format as sub03's contract)
pipeline  load the model once and analyse a star (no Streamlit)
ui        the Streamlit page for this stage
"""

__version__ = "0.2.0"     # model v2 (SPLIT_W2)

TIER_NAMES = ["C", "M", "X", "Superflare"]
