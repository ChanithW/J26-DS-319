"""Optional TESS light-curve fetch via Lightkurve (off by default; needs the 'fetch' extra)."""
from __future__ import annotations

import logging
import time as _time

import numpy as np

log = logging.getLogger(__name__)


def fetch_spoc_2min(star_id: int, tpeak: float, retries: int = 3, wait: float = 2.0):
    """Return (time_btjd, flux) for the 2-minute SPOC curve around ``tpeak``; retries on connection errors."""
    try:
        import lightkurve as lk
    except ImportError as e:
        raise RuntimeError("lightkurve is not installed; install component2[fetch]") from e
    last: Exception | None = None
    for k in range(retries):
        try:
            res = lk.search_lightcurve(f"TIC {star_id}", mission="TESS", author="SPOC", exptime=120)
            for lc in res.download_all():
                t = np.asarray(lc.time.value, float)
                if t.min() <= tpeak <= t.max():
                    return t, np.asarray(lc.flux.value, float)
            raise LookupError(f"no 2-minute SPOC sector covers BTJD {tpeak} for TIC {star_id}")
        except (ConnectionError, TimeoutError, OSError) as e:
            last = e
            log.warning("fetch attempt %d/%d failed: %s", k + 1, retries, e)
            _time.sleep(wait)
    raise ConnectionError(f"could not fetch TIC {star_id}") from last
