# Changelog

## 2.3.0 — 2026-11-03

Location validated against a catalogue.

- `python -m centella.goes` now writes flare positions (`hpc_x`, `hpc_y`,
  helioprojective arcsec). Each SWPC flare takes the position LMSAL's "SSW
  Latest Events" gives in AIA for the same flare (peak within 6 min, same
  letter), else SWPC's own; (0, 0) means none. `Flare.hpc` is not part of a
  flare's identity.
- `centella.validate` reports `location`: for each detected M/X flare with
  a position, the distance between Centella's location at its
  highest-probability alert and the catalogue (median, p75, fractions
  within 60/120/240″).

## 2.2.1 — 2026-11-02

- `/api/flares` adds SWPC's 7-day list for windows that reach the last 7
  days. The curated archive only covers days up to yesterday, so live pages
  were missing today's flares. The list is fetched at most every 10 min and
  the last good copy is kept on failure; `source` says `archive+swpc`.

## 2.2.0 — 2026-11-02

Retrospective viewer and past-period replays. `/api/status` is unchanged.

- `GET /viewer`: an internal page for browsing any period frame by frame.
  It has a probability timeline with threshold, events, GOES flares and
  gaps (zoom, pan, click to jump), the 4 channels as images or
  differences, the Grad-CAM, the frame's row, and lists of events vs GOES
  flares. Playback at 1–8 frames/s, keyboard steps, and links that reopen
  the same frame (docs/viewer.md).
- New endpoints: `/api/range`, `/api/frame/{slot}`, `/api/flares`.
  `/api/series`, `/api/events` and `/api/frames` take `start`/`end` for any
  past window (series up to 40 days).
- `CENTELLA_FITS_DIRS`: the API also finds FITS by name in these
  read-only trees, so a replayed month is browsed without copying it.
  `CENTELLA_FLARES_CSV`: the GOES list shown by the viewer (default: the
  curated archive's).
- Replay (`--replay`) keeps the Grad-CAM of every alerting frame (no
  tensors; `--no-gradcam` to skip), logs progress every 12 h of data, and
  `--resume` continues after a stop. Replay rows no longer carry
  `input_age_s`, which is only meaningful live.
- `python -m centella.goes --start --end --out`: the SWPC flare list of a
  past period, from HEK.

## 2.1.0 — 2026-10-28

Ready for integration into the portal (`docs/models/CONTRIBUTING-A-MODEL.md`
checklist). Same weights and threshold; `/api/status` only adds fields.

**Contract**
- `/api/status`: `state` (quiet / onset / active / decay), `raw_score`
  (labelled; equal to `probability`, no gating), `input_age_s` (per request;
  `stale` beyond 900 s, was 3600 s on the observation time), `last_frame`
  (what happened to the newest slot and why), `max_input_age_sec`,
  `baseline_missing`, `n_baseline_used`, `event_t_start`. T_OBS in
  `inputs` with `Z`.
- A live frame whose newest T_OBS is older than 15 min is not run through
  the model (`inputs_old`).

**Display endpoints (proposed, INTEGRATION.md):** `/api/series`,
`/api/frames`, `/api/image/{wave}.png?slot=`, `/api/gradcam/{latest|slot}.png`;
events carry the URL of their peak's Grad-CAM. Grad-CAM of the ensemble
(`Ensemble.gradcam`) for the newest frame and every alert.

**Loop and input**
- Polls `mostrecent/` every 30 s (was 60), retries a dropped connection,
  and keeps polling while working through a backlog. The dated archive
  (which lags) is queried at most every 10 min for at most 20 s. Before, one
  pass over 60 min × 4 channels of 404s took more than 2 min and newer
  `mostrecent/` images were missed.
- 2 of the 3 baseline frames suffice (`CENTELLA_MIN_BASELINE=2`); the
  validated rule used 3 of 3 (`CENTELLA_MIN_BASELINE=3` restores it).
- Every result records `aia131_excess` (disk-integrated 131 Å rise over the
  same baseline), the input of the baseline rule.

**Storage:** raw FITS 7 × 24 h (was 3 days), model-input tensors 2 days
(new, `tensors/`), Grad-CAM 2 days and 400 days for alerts (new), local
archive 7 days (was 14), validation reports bounded. Steady state ≈ 25 GB.

**Weights:** `python -m centella.fetch_weights` downloads the release assets
and verifies their SHA-256; the SHA-256 is pinned as the default of
`CENTELLA_WEIGHTS_SHA256`; a file is never loaded unverified, and only with
`weights_only=True` (the fallback to `weights_only=False` is gone).

**Validation:** `centella.validate` scores the named baseline
`aia131_rise` at Centella's false-alarm count and reports the
low-probability alternation index.

**Repo:** `HelioUSACH/flare-nrt`; `requirements.lock.txt`; `deploy/README.md`
(internal service polled by the portal); unit limits `MemoryMax=3G`,
`CPUQuota=200%` for the loop; the API writes only `cache/`. Empty values in
the environment file mean the default. Resources measured with the real
weights (INTEGRATION.md): 10.4 s per inference on 2 vCPU, 1.14 GB peak (2.0's
"0.7 s" was wrong).

**Tests:** a stored real input (`tests/fixtures/snapshot_20261028T1552Z.*`)
→ known probability with the stand-in and the real weights; the state
machine (onset, plateau, decay); input age; channel spread; display
endpoints; retention.

## 2.0.1 (folded into 2.1.0)

- QC: QUALITY bit 30 (0x40000000, quicklook) is no longer a reject. Every near-real-time AIA image has it, so in live mode every slot was `qc_reject` and no probability was ever made.
- Live source: each `mostrecent/` image is filed (as a copy) under every 2-min slot within 90 s of its T_OBS (the QC time tolerance), not only under floor(T_OBS). Before, images taken late in their slot failed `time_offset` and slots rarely had all 4 channels.
- Loop: on a cold start it no longer waits for slots from before the start that only the lagging archive could fill. `CENTELLA_SKIP_AFTER_MIN` default 20 → 12.
- API: `/api/image` and `/api/channels` show the newest staged image of each channel until the first full result exists (the difference once t−8/−10/−12 are staged).

## 2.0.0 — 2026-10-26

Recalibration on live synoptic data; same network weights as 1.x.

- **Live source** reads JSOC `mostrecent/` every poll (filed under the slot of
  its own T_OBS) besides the dated archive, which lags behind real time;
  `CENTELLA_STAGING_HOURS` keeps only the last N hours of raw FITS on a
  small disk; a cold start skips the slots the archive lacks.
- **Python 3.10+** (was 3.11): `dt.timezone.utc`, and one ISO parser
  (`grid.parse_datetime`) for every provider time spelling.
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
