"""What the loop keeps of each processed frame for the portal and for
re-training, besides the result row:

- ``tensors/YYYYMMDD/HHMM.npz``: the exact (8, 512, 512) float16 model
  input (key ``x``), ``np.savez_compressed``; kept ``tensors_days`` (2).
- ``gradcam/YYYYMMDD/HHMM.png``: Grad-CAM of the ensemble over the 131 Å
  channel, 512 px, north up, for the newest frame each time (not for each
  slot of a backlog) and for every alerting frame; kept ``gradcam_days``
  (2), and ``HHMM_alert.png`` for alerting frames, kept
  ``gradcam_alert_days`` (400).

Live mode keeps both. A replay (``python -m flare_nrt.loop --replay``) keeps
only the Grad-CAM of alerting frames, for the retrospective viewer: a month
of tensors would be ~60 GB."""

from __future__ import annotations

import datetime as dt
import io
import logging
import time
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def tensor_path(root: Path, slot: dt.datetime) -> Path:
    return root / f"{slot:%Y%m%d}" / f"{slot:%H%M}.npz"


def gradcam_path(root: Path, slot: dt.datetime, alert: bool) -> Path:
    return root / f"{slot:%Y%m%d}" / f"{slot:%H%M}{'_alert' if alert else ''}.png"


def gradcam_png(cam: np.ndarray, base131: np.ndarray, strength: float = 1.0) -> bytes:
    """Heat map (black→red→yellow→white, alpha ∝ value × ``strength``) over
    the 131 Å channel the model saw (asinh-normalised, grey). Both arrays are
    in FITS row order; the image is flipped so north is up.

    Grad-CAM is scaled to each member's own maximum, so on a quiet frame a
    weak map would look as strong as on a flare. ``strength`` =
    probability / threshold (clipped to [0.15, 1]) keeps that honest: a
    quiet frame is a faint tint, an alert full colour."""
    from PIL import Image

    g = np.clip(base131, 0, 1)
    c = np.clip(cam, 0, 1)
    heat = np.stack([np.clip(3 * c, 0, 1), np.clip(3 * c - 1, 0, 1), np.clip(3 * c - 2, 0, 1)], -1)
    alpha = (np.clip(1.5 * c, 0, 0.85) * float(np.clip(strength, 0.15, 1.0)))[..., None]
    rgb = (1 - alpha) * g[..., None] + alpha * heat
    img = Image.fromarray((np.flipud(rgb) * 255).astype(np.uint8), "RGB")
    buf = io.BytesIO()
    img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def _write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)


class Artifacts:
    def __init__(self, cfg, ensemble, enabled: bool = True, tensors: bool = True):
        self.cfg = cfg
        self.ens = ensemble
        self.enabled = enabled
        self.tensors = tensors
        self.last_gradcam_ms: int | None = None

    def save(self, slot: dt.datetime, x, current, alert: bool, gradcam: bool = True,
             probability: float = 1.0) -> None:
        """Never raises: a failed preview must not cost a result."""
        if not self.enabled:
            return
        try:
            if self.tensors:
                buf = io.BytesIO()
                np.savez_compressed(buf, x=x.cpu().numpy())
                _write(tensor_path(self.cfg.tensors_dir, slot), buf.getvalue())
            if not gradcam:
                return
            t0 = time.perf_counter()
            cam = self.ens.gradcam(x).numpy()
            self.last_gradcam_ms = round(1000 * (time.perf_counter() - t0))
            png = gradcam_png(cam, current[0].float().cpu().numpy(), probability / self.cfg.threshold
                              if self.cfg.threshold > 0 else 1.0)
            _write(gradcam_path(self.cfg.gradcam_dir, slot, False), png)
            if alert:
                _write(gradcam_path(self.cfg.gradcam_dir, slot, True), png)
        except Exception:
            log.exception("saving tensor/Grad-CAM for %s failed", slot)
