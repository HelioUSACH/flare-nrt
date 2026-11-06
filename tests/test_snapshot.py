"""A stored real input → a known probability (checklist C.12).

``tests/fixtures/snapshot_20261028T1552Z.npz`` is the exact model input the
loop built for AIA slot 2026-10-28T15:52Z (quiet Sun; provenance in the
.json next to it). In CI it runs through the tiny stand-in ensemble
(conftest); with the real weights it must give the probability recorded in
the .json, to ``tolerance`` (1e-4):

    CENTELLA_REAL_WEIGHTS=models/deployment_ensemble_thr058.pt pytest tests/test_snapshot.py
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from centella.model import Ensemble, sha256

FIX = Path(__file__).parent / "fixtures"
META = json.loads((FIX / "snapshot_20261028T1552Z.json").read_text())
X = torch.from_numpy(np.load(FIX / "snapshot_20261028T1552Z.npz")["x"])


def test_snapshot_is_the_model_input():
    assert X.shape == (8, 512, 512) and X.dtype == torch.float16
    # channels 4–7 are current − baseline: small next to the currents
    assert float(X[4:].float().abs().mean()) < float(X[:4].float().abs().mean())


def test_snapshot_through_stand_in_ensemble(stand_in_weights):
    ens = Ensemble.load(stand_in_weights, sha256(stand_in_weights), device="cpu")
    p, scores = ens.predict(X)
    assert abs(p - 0.516362) < 1e-4, p      # stand-in (conftest, seed 0), torch per requirements.lock.txt
    cam = ens.gradcam(X)
    assert cam.shape == (512, 512)


@pytest.mark.skipif(not os.environ.get("CENTELLA_REAL_WEIGHTS"), reason="real weights not configured")
def test_snapshot_through_real_weights():
    w = Path(os.environ["CENTELLA_REAL_WEIGHTS"])
    ens = Ensemble.load(w, META["real_weights_sha256"], device="cpu")
    p, scores = ens.predict(X)
    assert abs(p - META["real_probability"]) < META["tolerance"], p
    assert all(abs(a - b) < META["tolerance"] for a, b in zip(scores, META["real_scores"], strict=True))
