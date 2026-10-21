"""Unit tests for flare energy and tiers."""
import numpy as np

from flare_cnn.energy import energy_to_tier, equivalent_duration, flare_energy, luminosity, tier_name


def test_sun_luminosity():
    # The Sun: 5772 K, 1 solar radius -> about 3.83e33 erg/s
    assert abs(luminosity(5772, 1.0) / 3.828e33 - 1) < 0.01


def test_tier_boundaries():
    e = [5e31, 1e32, 5e32, 1e33, 5e33, 1e34, 1e35, np.nan]
    assert list(energy_to_tier(e)) == [0, 1, 1, 2, 2, 3, 3, -1]
    assert [tier_name(i) for i in range(4)] == ["C", "M", "X", "Superflare"]


def test_equivalent_duration_box():
    t = np.arange(0, 1, 1 / 1440)           # 1 day at 1-minute steps
    f = np.ones_like(t)
    f[(t >= 0.5) & (t < 0.5 + 100 / 86400)] = 2.0    # flux doubled for 100 s
    assert abs(equivalent_duration(t, f) - 100) < 70  # about 100 s (edges add one step)


def test_same_ed_more_energy_on_bigger_star():
    ed = 10.0
    m_dwarf = flare_energy(ed, 3200, 0.3)
    sun_like = flare_energy(ed, 5800, 1.0)
    assert sun_like > 50 * m_dwarf          # why the energy head needs star info
