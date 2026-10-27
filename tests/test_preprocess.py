"""Pins the preprocessing the threshold was calibrated with."""

import numpy as np
import torch

from centella.preprocess import build_current, build_input, normalize_asinh, resize_512


def test_normalize_asinh_reference_values():
    v_high = 100.0
    t = torch.tensor([-5.0, 0.0, 10.0, 100.0, 1e6])
    out = normalize_asinh(t, v_high).float()
    ref = np.clip(np.arcsinh(np.clip(t.numpy(), 0, None) / 10.0) / np.arcsinh(10.0), 0, 1)
    assert out.dtype == torch.float32 and np.allclose(out.numpy(), ref, atol=1e-3)
    assert float(out[3]) == 1.0 and float(out[0]) == 0.0


def test_resize_bilinear_no_antialias():
    a = np.zeros((1024, 1024), np.float32)
    a[::2, ::2] = 4.0
    r = resize_512(a)
    assert r.shape == (512, 512)
    # bilinear without antialias samples between pixels: mean preserved ~1
    assert abs(float(r.mean()) - 1.0) < 0.05


def test_build_input_layout():
    imgs = {w: np.full((1024, 1024), 50.0, np.float32) for w in (131, 171, 304, 1600)}
    vh = {w: 100.0 for w in imgs}
    cur = build_current(imgs, vh)
    assert cur.shape == (4, 512, 512) and cur.dtype == torch.float16
    x = build_input(cur, [cur, cur, cur])
    assert x.shape == (8, 512, 512) and float(x[4:].abs().max()) == 0.0
