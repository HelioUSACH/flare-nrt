"""Settings. Every value comes from an environment variable (``CENTELLA_*``) with a
default under ``CENTELLA_DATA_DIR``; nothing assumes a particular machine."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    """An unset or empty variable means the default (.env.example lists
    every setting with an empty value)."""
    return os.environ.get(f"CENTELLA_{name}") or default


@dataclass
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "./var")))
    weights: Path = field(default_factory=lambda: Path(_env("WEIGHTS", "./models/deployment_ensemble_thr058.pt")))
    # pinned: the release asset (centella/fetch_weights.py); anything else is refused
    weights_sha256: str = field(default_factory=lambda: _env(
        "WEIGHTS_SHA256", "d3b850e3ec3d2670f4cb9327c397f7391ed1b082d2b6d154bca1b57626676006"))
    norm_stats: Path = field(default_factory=lambda: Path(_env("NORM_STATS", "./metadata/global_norm_stats.json")))

    # detection (validated 2026-06-19 → 07-17 against GOES, docs/model-card.md)
    lag_min: int = field(default_factory=lambda: int(_env("LAG_MIN", "8")))
    n_baseline: int = field(default_factory=lambda: int(_env("N_BASELINE", "3")))
    threshold: float = field(default_factory=lambda: float(_env("THRESHOLD", "0.80")))
    merge_gap_min: int = field(default_factory=lambda: int(_env("MERGE_GAP_MIN", "10")))
    # fewest of the n_baseline frames that must be usable (validated with 3 of 3;
    # 2 of 3 keeps a slot when one baseline frame is missing, n_baseline_used says so)
    min_baseline: int = field(default_factory=lambda: int(_env("MIN_BASELINE", "2")))

    # input
    jsoc_base: str = field(default_factory=lambda: _env("JSOC_BASE", "https://jsoc1.stanford.edu/data/aia/synoptic"))
    # mostrecent/ holds only the newest image: poll faster than its ~2-min update
    poll_sec: int = field(default_factory=lambda: int(_env("POLL_SEC", "30")))
    # the dated archive lags real time: query it at most this often, for at most
    # archive_budget_sec per pass (it only fills gaps)
    archive_every_sec: int = field(default_factory=lambda: int(_env("ARCHIVE_EVERY_SEC", "600")))
    archive_budget_sec: float = field(default_factory=lambda: float(_env("ARCHIVE_BUDGET_SEC", "20")))
    lookback_min: int = field(default_factory=lambda: int(_env("LOOKBACK_MIN", "180")))
    give_up_min: int = field(default_factory=lambda: int(_env("GIVE_UP_MIN", "120")))
    # a slot still missing this long after a later slot has arrived is skipped
    skip_after_min: int = field(default_factory=lambda: int(_env("SKIP_AFTER_MIN", "12")))
    http_timeout: float = field(default_factory=lambda: float(_env("HTTP_TIMEOUT", "15")))
    torch_threads: int = field(default_factory=lambda: int(_env("TORCH_THREADS", "4")))

    # freshness (the portal's manifest uses stale_after: 10m). A frame whose
    # newest T_OBS is older than max_input_age_sec when processed is skipped
    # (status inputs_old), and /api/status says stale once input_age_s passes it
    stale_after_sec: int = field(default_factory=lambda: int(_env("STALE_AFTER_SEC", "600")))
    max_input_age_sec: int = field(default_factory=lambda: int(_env("MAX_INPUT_AGE_SEC", "900")))

    # retention
    staging_days: float = field(default_factory=lambda: float(_env("STAGING_DAYS", "7")))
    tensors_days: float = field(default_factory=lambda: float(_env("TENSORS_DAYS", "2")))
    gradcam_days: float = field(default_factory=lambda: float(_env("GRADCAM_DAYS", "2")))
    gradcam_alert_days: float = field(default_factory=lambda: float(_env("GRADCAM_ALERT_DAYS", "400")))
    image_cache_days: float = field(default_factory=lambda: float(_env("IMAGE_CACHE_DAYS", "1")))
    # > 0: keep only the last N hours of raw FITS (small disks; no curation)
    staging_hours: float = field(default_factory=lambda: float(_env("STAGING_HOURS", "0")))
    archive_local_days: float = field(default_factory=lambda: float(_env("ARCHIVE_LOCAL_DAYS", "7")))
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

    # read-only trees with more AIA FITS (AIAYYYYMMDD_HHMM_WWWW.fits, any depth), searched
    # by the API after staging/: e.g. the FITS a month's replay was run on
    fits_dirs: list = field(default_factory=lambda: [Path(p) for p in _env("FITS_DIRS", "").split(os.pathsep) if p])
    # GOES flare list for the viewer (CSV: tstart,tpeak,tend,fclass); default: the
    # curated archive's flares.csv files
    flares_csv: str = field(default_factory=lambda: _env("FLARES_CSV", ""))

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
    def tensors_dir(self) -> Path:
        return self.data_dir / "tensors"

    @property
    def gradcam_dir(self) -> Path:
        return self.data_dir / "gradcam"

    @property
    def cache_dir(self) -> Path:
        """The only directory the API writes (rendered PNG previews)."""
        return self.data_dir / "cache"

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
        for d in (self.staging_dir, self.results_dir, self.archive_dir, self.state_dir,
                  self.tensors_dir, self.gradcam_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)
