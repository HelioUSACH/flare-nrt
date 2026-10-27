import numpy as np

from centella.localize import localize, rate_map

HDR = {"NAXIS1": 1024, "CRPIX1": 512.5, "CRPIX2": 512.5, "CDELT1": 2.4, "CDELT2": 2.4,
       "RSUN_OBS": 960.0, "EXPTIME": 2.9}


def _sun(flare=0.0):
    yy, xx = np.mgrid[:1024, :1024]
    img = np.where((xx - 511.5) ** 2 + (yy - 511.5) ** 2 < 400**2, 100.0, 1.0)
    # flare north-west of centre: column 760 (x > 0), row 700 (y > 0)
    return img + flare * np.exp(-((xx - 760) ** 2 + (yy - 700) ** 2) / (2 * 10**2))


def test_flare_under_short_exposure_is_found_north_west():
    base = rate_map(_sun().astype(np.float32), HDR)
    short = dict(HDR, EXPTIME=2.9 * 0.02)  # AIA exposure control during a flare
    cur_raw = (_sun(flare=3000.0) * 0.02).astype(np.float32)
    assert (cur_raw - _sun()).min() < 0  # raw DN difference is negative everywhere on the disk
    loc = localize(rate_map(cur_raw, short) - base, HDR)
    x, y = loc["centroid_arcsec"]
    assert abs(x - (760.5 + 0.5 - 512.5) * 2.4) < 10 and abs(y - (700.5 + 0.5 - 512.5) * 2.4) < 10
    assert x > 0 and y > 0 and loc["on_limb"] is False


def test_no_brightening_gives_no_location():
    base = rate_map(_sun().astype(np.float32), HDR)
    assert localize(base - base, HDR) is None
