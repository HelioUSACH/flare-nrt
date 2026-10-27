# Changelog

## 2.0.0 — 2026-10-26

Recalibration on live synoptic data; same network weights as 1.x.

- **Renamed to Centella** (package `centella`, settings `CENTELLA_*`, units
  `centella-*.service`). The portal product id stays `usach.flare.nrt`.

- **Rolling baseline**: input difference against the mean of t−8, t−10,
  t−12 min, replacing the hourly "quiet Sun" baseline whose age grew to
  hours and drove the false alarms of 1.x (live TSS 0.10).
- **Threshold 0.80** (was 0.58), chosen on 2026-06-19 → 07-17 against GOES.
- **Frame QC** (`qc.py`): QUALITY/EXPTIME/WAVELNTH/time/blank checks and a
  disk-median check per second of exposure;
  rejected frames are never baselines.
- **Dated JSOC archive on the 2-min grid** instead of `mostrecent`: each slot
  processed exactly once, restart-safe, replayable.
- **Events**: alerts ≤ 10 min apart merged.
- **Archive for retraining** (`curate.py`, docs/archive-format.md) and
  bounded disk use (`retention.py`).
- **Weekly validation** against GOES (`validate.py`).
- `/api/status`: `live_status` computed per request from the loop heartbeat
  and last result; new `observation_time`, `alert`, `inputs`,
  `data_latency_s`, `weights_sha256`; `severity` removed.
- Weights verified by SHA-256 before loading.
- Location: new localiser on the 131 Å difference in DN/s (experimental;
  raw DN differences go negative under AIA exposure control), written from the
  FITS orientation convention (1.x appeared to have y flipped).
