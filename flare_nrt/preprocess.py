"""Image preprocessing. Byte-for-byte the pipeline validated in 2026-10
(`nrt_loop_live.build_current`): bilinear 1024→512 without antialiasing,
clip at 0, asinh with scale 0.1·v_high, divide by asinh(10), clamp [0, 1],
float16. Changing anything here invalidates the threshold calibration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .grid import CHANNELS

TARGET_SIZE = 512


def load_norm_stats(path: Path) -> dict[int, float]:
    raw = json.loads(Path(path).read_text())
    return {w: float(raw[str(w)]["v_high"]) for w in CHANNELS}


def resize_512(data: np.ndarray) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(data, dtype=np.float32))[None, None]
    return F.interpolate(t, size=(TARGET_SIZE, TARGET_SIZE), mode="bilinear", align_corners=False)[0, 0]


def normalize_asinh(t: torch.Tensor, v_high: float) -> torch.Tensor:
    t = torch.clamp(t, min=0)
    scale = 0.1 * v_high
    x = torch.asinh(t / (scale + 1e-6))
    x_vh = torch.asinh(torch.tensor(v_high / (scale + 1e-6)))
    return torch.clamp(x / (x_vh + 1e-6), 0, 1).to(torch.float16)


def build_current(images: dict[int, np.ndarray], v_high: dict[int, float]) -> torch.Tensor:
    """{wave: 2-D array} → (4, 512, 512) float16."""
    return torch.stack([normalize_asinh(resize_512(images[w]), v_high[w]) for w in CHANNELS])


def build_input(current: torch.Tensor, baselines: list[torch.Tensor]) -> torch.Tensor:
    """(8, 512, 512) float16: 4 current channels + current − mean(baselines)."""
    bl = torch.stack([b.float() for b in baselines]).mean(0)
    diff = (current.float() - bl).to(torch.float16)
    return torch.cat([current, diff], dim=0)
