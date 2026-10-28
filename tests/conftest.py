"""Shared fixtures: synthetic synoptic FITS (no real data in the repo) and a
tiny stand-in ensemble checkpoint with the training checkpoint's layout."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pytest
import torch

from centella.config import Settings
from centella.grid import CHANNELS, file_name
from centella.model import build_member, sha256

T0 = dt.datetime(2026, 10, 20, 12, 0, tzinfo=dt.timezone.utc)


def write_fits(path, wave, slot, size=512, quality=0, exptime=2.9, flare=0.0, blank=False):
    from astropy.io import fits

    rng = np.random.default_rng(int(slot.timestamp()) + wave)
    yy, xx = np.mgrid[:size, :size]
    c = size / 2 - 0.5
    disk = ((xx - c) ** 2 + (yy - c) ** 2) < (0.4 * size) ** 2
    data = np.where(disk, 100.0, 1.0) + rng.normal(0, 1, (size, size))
    if flare:
        data += flare * np.exp(-(((xx - 350) ** 2 + (yy - 300) ** 2) / (2 * 8**2)))
    if blank:
        data[:] = 0
    hdr = fits.Header()
    hdr.update({"QUALITY": quality, "EXPTIME": exptime, "WAVELNTH": wave,
                "T_OBS": slot.strftime("%Y-%m-%dT%H:%M:%S.00Z"),
                "CRPIX1": size / 2 + 0.5, "CRPIX2": size / 2 + 0.5,
                "CDELT1": 2.4, "CDELT2": 2.4, "RSUN_OBS": 0.4 * size * 2.4})
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data.astype(np.float32), header=hdr)]).writeto(
        path, overwrite=True)


def write_slot(root, slot, **kw):
    for w in CHANNELS:
        write_fits(root / f"{slot:%Y%m%d}" / file_name(slot, w), w, slot, **kw)


@pytest.fixture
def stand_in_weights(tmp_path):
    torch.manual_seed(0)
    members = []
    for seed in range(2):
        m = build_member()
        sd = {}
        for k, v in m.state_dict().items():
            k2 = "head.1." + k[3:] if k.startswith("fc.") else k
            sd[k2] = v
        members.append({"seed": seed, "model_state": sd})
    p = tmp_path / "stand_in.pt"
    torch.save({"ensemble_members": members, "threshold": 0.58}, p)
    return p


@pytest.fixture
def norm_stats(tmp_path):
    p = tmp_path / "norm.json"
    p.write_text(json.dumps({str(w): {"v_high": 150.0} for w in CHANNELS}))
    return p


@pytest.fixture
def cfg(tmp_path, stand_in_weights, norm_stats):
    c = Settings()
    c.data_dir = tmp_path / "var"
    c.weights = stand_in_weights
    c.weights_sha256 = sha256(stand_in_weights)
    c.norm_stats = norm_stats
    c.torch_threads = 1
    c.min_free_gb = 0  # the disk guard is tested on its own, never by accident
    c.ensure_dirs()
    return c
