"""Settings. Every value comes from an environment variable (``FLARE_*``) with a
default under ``FLARE_DATA_DIR``; nothing assumes a particular machine."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"FLARE_{name}", default)


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./var")))
    weights: Path = field(default_factory=lambda: Path(_env("WEIGHTS", "./models/deployment_ensemble_thr058.pt")))
    weights_sha256: str = field(default_factory=lambda: _env("WEIGHTS_SHA256", ""))
    norm_stats: Path = field(default_factory=lambda: Path(_env("NORM_STATS", "./metadata/global_norm_stats.json")))

    # detection (validated 2026-06-19 → 07-17 against GOES, docs/model-card.md)
    lag_min: int = field(default_factory=lambda: int(_env("LAG_MIN", "8")))
    n_baseline: int = field(default_factory=lambda: int(_env("N_BASELINE", "3")))
    threshold: float = field(default_factory=lambda: float(_env("THRESHOLD", "0.80")))
    merge_gap_min: int = field(default_factory=lambda: int(_env("MERGE_GAP_MIN", "10")))

    # input
    jsoc_base: str = field(default_factory=lambda: _env("JSOC_BASE", "https://jsoc1.stanford.edu/data/aia/synoptic"))
    poll_sec: int = field(default_factory=lambda: int(_env("POLL_SEC", "60")))
    lookback_min: int = field(default_factory=lambda: int(_env("LOOKBACK_MIN", "180")))
    give_up_min: int = field(default_factory=lambda: int(_env("GIVE_UP_MIN", "120")))
    # a slot still missing this long after a later slot has arrived is skipped
    skip_after_min: int = field(default_factory=lambda: int(_env("SKIP_AFTER_MIN", "20")))
    http_timeout: float = field(default_factory=lambda: float(_env("HTTP_TIMEOUT", "30")))
    torch_threads: int = field(default_factory=lambda: int(_env("TORCH_THREADS", "4")))

    # freshness (the portal's manifest uses stale_after: 10m)
    stale_after_sec: int = field(default_factory=lambda: int(_env("STALE_AFTER_SEC", "600")))
    # a result about an observation older than this is reported stale
    max_input_age_sec: int = field(default_factory=lambda: int(_env("MAX_INPUT_AGE_SEC", "3600")))

    # retention
    staging_days: float = field(default_factory=lambda: float(_env("STAGING_DAYS", "3")))
    archive_local_days: float = field(default_factory=lambda: float(_env("ARCHIVE_LOCAL_DAYS", "14")))
    results_days: float = field(default_factory=lambda: float(_env("RESULTS_DAYS", "400")))
    min_free_gb: float = field(default_factory=lambda: float(_env("MIN_FREE_GB", "10")))

    # curation
    goes_url: str = field(default_factory=lambda: _env(
        "GOES_URL", "https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json"))
    pre_flare_min: int = field(default_factory=lambda: int(_env("PRE_FLARE_MIN", "45")))
    post_flare_min: int = field(default_factory=lambda: int(_env("POST_FLARE_MIN", "15")))
    neg_blocks_per_day: int = field(default_factory=lambda: int(_env("NEG_BLOCKS_PER_DAY", "4")))
    neg_block_min: int = field(default_factory=lambda: int(_env("NEG_BLOCK_MIN", "60")))
    min_flare_class: str = field(default_factory=lambda: _env("MIN_FLARE_CLASS", "C1.0"))

    host: str = field(default_factory=lambda: _env("HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: int(_env("PORT", "5050")))

    @property
    def staging_dir(self) -> Path:
        return self.data_dir / "staging"

    @property
    def results_dir(self) -> Path:
        return self.data_dir / "results"

    @property
    def archive_dir(self) -> Path:
        return self.data_dir / "archive"

    @property
    def state_dir(self) -> Path:
        return self.data_dir / "state"

    @property
    def status_path(self) -> Path:
        return self.state_dir / "status.json"

    @property
    def loop_state_path(self) -> Path:
        return self.state_dir / "loop_state.json"

    @property
    def events_path(self) -> Path:
        return self.data_dir / "events.csv"

    def ensure_dirs(self) -> None:
        for d in (self.staging_dir, self.results_dir, self.archive_dir, self.state_dir):
            d.mkdir(parents=True, exist_ok=True)
