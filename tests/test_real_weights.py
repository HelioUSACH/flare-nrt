"""Second end-to-end test with the real weights (checklist C.12). Too big for
CI; run locally:

    CENTELLA_REAL_WEIGHTS=models/deployment_ensemble_thr058.pt \
    CENTELLA_REAL_NORM=metadata/global_norm_stats.json \
    CENTELLA_REAL_FITS=/path/to/one/slot/with/its/baselines \
    CENTELLA_REAL_EXPECT=0.0123 pytest tests/test_real_weights.py

``CENTELLA_REAL_FITS`` holds the 4 channels of a slot and of t−8/−10/−12;
``CENTELLA_REAL_EXPECT`` is the probability the validated replay gave for it.
"""

import os
from pathlib import Path

import pytest

from centella.config import Settings
from centella.loop import run
from centella.store import ResultStore

pytestmark = pytest.mark.skipif(not os.environ.get("CENTELLA_REAL_WEIGHTS"), reason="real weights not configured")


def test_real_weights_reproduce_validated_probability(tmp_path):
    cfg = Settings()
    cfg.data_dir = tmp_path
    cfg.weights = Path(os.environ["CENTELLA_REAL_WEIGHTS"])
    cfg.norm_stats = Path(os.environ["CENTELLA_REAL_NORM"])
    cfg.weights_sha256 = os.environ.get("CENTELLA_WEIGHTS_SHA256", "")
    run(cfg, replay_dir=Path(os.environ["CENTELLA_REAL_FITS"]))
    rows = [r for d in sorted(cfg.results_dir.glob("*.jsonl"))
            for r in ResultStore(cfg.results_dir).read_day(d.stem) if r["status"] == "ok"]
    assert rows, "no slot had a complete baseline"
    assert abs(rows[-1]["probability"] - float(os.environ["CENTELLA_REAL_EXPECT"])) < 1e-3
