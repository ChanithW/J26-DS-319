"""Star information for the energy head: Teff, radius and distance.

Teff comes from, in order:
    1. a value the user types in
    2. the TESS Input Catalog (TIC)
    3. an estimate from the star's colour (Gaia BP-RP, or V-K if Gaia is missing)
"""

import numpy as np

# Main-sequence colour -> temperature (Pecaut & Mamajek 2013, dwarf stars)
BPRP_TABLE = [(0.38, 7220), (0.59, 6510), (0.76, 5920), (0.82, 5770), (0.98, 5270), (1.14, 4990),
              (1.49, 4440), (1.84, 3850), (2.03, 3720), (2.29, 3560), (2.50, 3420), (2.78, 3210),
              (3.03, 3060), (3.45, 2900), (4.00, 2700)]
VK_TABLE = [(0.82, 7220), (1.10, 6510), (1.46, 5920), (1.54, 5770), (1.89, 5270), (2.22, 4990),
            (2.90, 4440), (3.65, 3850), (3.87, 3720), (4.21, 3560), (4.58, 3420), (5.12, 3210),
            (5.55, 3060), (6.25, 2900), (7.00, 2700)]


def _num(row, key):
    """Read a number from a catalogue row; anything missing or invalid -> NaN."""
    try:
        v = float(row.get(key))
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def teff_from_colour(row):
    """Estimate Teff from colour. Returns (teff, how_it_was_found)."""
    bprp = _num(row, "gaiabp") - _num(row, "gaiarp")
    if np.isfinite(bprp) and BPRP_TABLE[0][0] <= bprp <= BPRP_TABLE[-1][0]:
        x, y = zip(*BPRP_TABLE)
        return float(np.interp(bprp, x, y)), f"estimated from Gaia colour BP-RP={bprp:.2f}"
    vk = _num(row, "Vmag") - _num(row, "Kmag")
    if np.isfinite(vk) and VK_TABLE[0][0] <= vk <= VK_TABLE[-1][0]:
        x, y = zip(*VK_TABLE)
        return float(np.interp(vk, x, y)), f"estimated from V-K colour={vk:.2f}"
    return np.nan, "missing"


def star_type(teff):
    """Spectral type from temperature: F >= 6000 K, G >= 5300 K, K >= 3900 K, else M."""
    if teff is None or not np.isfinite(teff):
        return "unknown"
    return "F" if teff >= 6000 else "G" if teff >= 5300 else "K" if teff >= 3900 else "M"


def scale_star_info(teff, radius, distance, mean, std):
    """Scale [Teff, log10 radius, log10 distance] exactly as in training.

    Missing values are replaced by the training average (scaled value 0).
    Returns (scaled_array, list_of_missing_names).
    """
    raw = np.array([
        teff,
        np.log10(radius) if radius and radius > 0 else np.nan,
        np.log10(distance) if distance and distance > 0 else np.nan,
    ], float)
    missing = [n for n, v in zip(["Teff", "radius", "distance"], raw) if np.isnan(v)]
    raw = np.where(np.isnan(raw), mean, raw)
    return ((raw - mean) / std).astype(np.float32), missing


def clean_gaia_id(value):
    """Gaia DR3 IDs have 19 digits. Keep them as text, never as a float (which rounds them)."""
    s = str(value).strip()
    return "" if s in ("", "--", "nan", "None") else s.split(".")[0]


def star_info(tic, mean, std, teff_user=None):
    """Look up a star in the TESS Input Catalog and prepare its info for the model.

    Needs internet (astroquery). Returns a dictionary.
    """
    from astroquery.mast import Catalogs     # imported here so the tests run without internet

    tab = Catalogs.query_criteria(catalog="Tic", ID=int(tic))
    if len(tab) == 0:
        raise ValueError(f"TIC {tic} not found in the TESS Input Catalog")
    row = tab.to_pandas().iloc[0]
    teff, rad, dist = (_num(row, k) for k in ["Teff", "rad", "d"])

    if teff_user not in (None, "") and str(teff_user).strip():
        teff, source = float(teff_user), "entered by user"
    elif np.isfinite(teff):
        source = "TIC"
    else:
        teff, source = teff_from_colour(row)

    try:
        gaia = clean_gaia_id(tab["GAIA"][0])
    except (KeyError, IndexError):
        gaia = ""

    scaled, missing = scale_star_info(teff, rad, dist, mean, std)
    return dict(tic=int(tic), teff=teff, teff_source=source, radius=rad, distance=dist,
                gaia=gaia, type=star_type(teff), scaled=scaled, missing=missing,
                logg=_num(row, "logg"),
                # extra Gaia values from the TIC, passed on to Stage 2 (not used by this model)
                bp_rp=_num(row, "gaiabp") - _num(row, "gaiarp"),
                parallax=_num(row, "plx"),
                gaia_gmag=_num(row, "GAIAmag"))
