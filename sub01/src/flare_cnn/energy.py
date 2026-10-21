"""Flare energy and energy tiers.

Flare energy = equivalent duration (ED) x star luminosity
    ED  = area under the flare in the normalised light curve, in seconds
          ("how many seconds of the star's normal light the flare added")
    L   = 4 * pi * R^2 * sigma * T^4   (Stefan-Boltzmann law)

The same ED means much more energy on a big, hot star than on a small M dwarf,
which is why the energy head also receives the star's Teff and radius.
"""

import numpy as np

SIGMA_SB = 5.670374e-5        # Stefan-Boltzmann constant, erg s^-1 cm^-2 K^-4
R_SUN_CM = 6.957e10           # solar radius in cm
TIER_NAMES = ["C", "M", "X", "Superflare"]
TIER_EDGES = [1e32, 1e33, 1e34]   # erg


def luminosity(teff, radius_rsun):
    """Star luminosity in erg/s from temperature (K) and radius (solar radii)."""
    r_cm = np.asarray(radius_rsun, float) * R_SUN_CM
    return 4 * np.pi * r_cm ** 2 * SIGMA_SB * np.asarray(teff, float) ** 4


def equivalent_duration(time_days, rel_flux):
    """Area under the flare (flux - 1), converted to seconds.

    rel_flux should be the light curve divided by its quiet level, so quiet = 1.
    """
    t_sec = np.asarray(time_days, float) * 86400
    excess = np.asarray(rel_flux, float) - 1
    return float(np.sum((excess[1:] + excess[:-1]) / 2 * np.diff(t_sec)))   # trapezoid rule


def flare_energy(ed_seconds, teff, radius_rsun):
    """Flare energy in erg = ED x L."""
    return np.asarray(ed_seconds, float) * luminosity(teff, radius_rsun)


def energy_to_tier(energy_erg):
    """Tier index: 0 = C (< 1e32), 1 = M, 2 = X, 3 = Superflare (>= 1e34). NaN -> -1."""
    e = np.asarray(energy_erg, float)
    tier = np.digitize(e, TIER_EDGES)
    return np.where(np.isnan(e), -1, tier)


def tier_name(index):
    """0 -> 'C', 1 -> 'M', 2 -> 'X', 3 -> 'Superflare'."""
    return TIER_NAMES[int(index)]
