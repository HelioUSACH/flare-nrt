"""Second end-to-end test with the real weights (checklist C.12). Too big for
CI; run locally:

    FLARE_REAL_WEIGHTS=models/deployment_ensemble_thr058.pt \
    FLARE_REAL_NORM=metadata/global_norm_stats.json \
    FLARE_REAL_FITS=/path/to/one/slot/with/its/baselines \
    FLARE_REAL_EXPECT=0.0123 pytest tests/test_real_weights.py

``FLARE_REAL_FITS`` holds the 4 channels of a slot and of t−8/−10/−12;
``FLARE_REAL_EXPECT`` is the probability the validated replay gave for it.
"""

import os
from pathlib import Path

import pytest

from flare_nrt.config import Settings
from flare_nrt.loop import run
from flare_nrt.store import ResultStore

pytestmark = pytest.mark.skipif(not os.environ.get("FLARE_REAL_WEIGHTS"), reason="real weights not configured")


def test_real_weights_reproduce_validated_probability(tmp_path):
    cfg = Settings()
    cfg.data_dir = tmp_path
    cfg.weights = Path(os.environ["FLARE_REAL_WEIGHTS"])
    cfg.norm_stats = Path(os.environ["FLARE_REAL_NORM"])
    cfg.weights_sha256 = os.environ.get("FLARE_WEIGHTS_SHA256", "")
    run(cfg, replay_dir=Path(os.environ["FLARE_REAL_FITS"]))
    rows = [r for d in sorted(cfg.results_dir.glob("*.jsonl"))
            for r in ResultStore(cfg.results_dir).read_day(d.stem) if r["status"] == "ok"]
    assert rows, "no slot had a complete baseline"
    assert abs(rows[-1]["probability"] - float(os.environ["FLARE_REAL_EXPECT"])) < 1e-3
