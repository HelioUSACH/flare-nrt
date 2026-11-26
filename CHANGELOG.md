# Changelog

## 1.3.1

After the outage of 2026-11-24/25: no result from 22:58 UT on 2026-11-24
until this release, and nothing said why. **The network is unchanged**:
same weights (same two release files, same SHA-256), same normalisation,
threshold, baselines, state machine and alert rule
(`tests/test_snapshot.py` passes unchanged; `flare_nrt/model.py`,
`preprocess.py`, `detector.py`, `localize.py` and `tests/fixtures/snapshot_*`
are byte-identical to `v1.3.0`). **What changes is which frames are
processed** (item 1): frames flagged `Q_NOASD` were rejected by 1.0.0–1.3.0,
including the v1.0.0 the paper describes, and are processed now.

**The outage.** From the 23:00 UT image of 2026-11-24, JSOC's
near-real-time synoptic AIA images carry `QUALITY = 0x40000004` (was
`0x40000000`): bit 30, near-real-time, which QC always ignored, plus bit 2,
`Q_NOASD`, "ancillary science data not available or error" (JSOC
`proj/lev0/apps/quallev1.h`), with `ASD_REC` and `SAT_ROT` empty. QC
rejected any other bit than 30, so every slot since was written
`qc_reject`, with no log line; the loop kept downloading (newest files on
time) and advancing (`heartbeat.json`), so nothing looked wrong but the
missing results. The images are fine: synoptic frames arrive centred and
de-rotated (`CRPIX1/2 = 512.5`, `CROTA2 = 0`, `CDELT1 = 2.4`) with or
without the flag, preprocessing reads no pointing keyword, and `localize`
uses `CRPIX`/`CDELT`/`RSUN_OBS`, all present and normal. Cause external
(JSOC/SDO); the host, network, disk and memory were fine (the loop sat at
its 3 GB cap, but `memory.stat` shows 0.59 GB anonymous and 2.47 GB file
cache: no leak).

1. **Soft quality bits.** QC ignores bits 30 (`0x40000000`) and 2 (`0x04`,
   `Q_NOASD`); every other bit still rejects (e.g. the daily 21:00 UT frame,
   `0x40010000`, and the spacecraft flags of July 2026, `0x40200000`).
   The ignored mask is a setting, `FLARE_NRT_QC_IGNORE_QUALITY_MASK`
   (default `0x40000004`, hex or decimal; `0x40000000` restores 1.3.0).
   Every result row (and `last_frame`) records the raw QUALITY of each
   channel, `quality: {"131": 1073741828, …}`, and the ignored flags
   present, `quality_soft: ["noasd"]` (bit 30 is on every near-real-time
   frame and isn't listed), so these frames can be told apart later.
   Tests with real header values: `0x40000000` and `0x40000004` accepted;
   `0x40200000`, `0x40010000`, `0x80000000` rejected.
2. **The loop says when it produces no result.** After 15 slots in a row
   without a result (30 min; any status but `ok`; setting
   `FLARE_NRT_NO_RESULT_WARN_SLOTS`) it logs one WARNING with the reason
   codes counted, e.g. `30 min without a result (since
   2026-11-24T23:00:00Z, last result 2026-11-24T22:58:00Z):
   quality:131=1073741828 ×15, …`, then one every hour while it lasts, and
   one INFO when results resume. It also says so at start, the first time
   rebuilding the count from `results/`. `/api/status` gains
   `last_ok_slot`, `rejected_streak`, `rejected_since`,
   `rejected_streak_reasons`, `last_reject_reasons` (from
   `state/no_result.json`, written with every slot), `heartbeat` and
   `last_download` (from `heartbeat.json`; `last_download` now survives a
   restart instead of coming back `None`). The 1.3.0 fields are unchanged
   (`status.json` is still written with each result).
3. **An archive fallback that works.** The fallback used the dated archive
   (`synoptic/YYYY/MM/DD/HHH00/`), which lags weeks: 404 for the current
   days, and those answers were marked missed without a log line. It now
   lists JSOC's near-real-time archive,
   `synoptic/nrt/YYYY/MM/DD/HHH00/` (current to ~20 min; names carry
   seconds, `AIA20261124_230000_0131.fits`; **its cadence is 3 min**, so
   each image is filed under every 2-min slot within 90 s of it, as
   `mostrecent/` images are: 23:03 fills 23:02 and 23:04), once per hour
   directory per pass, and downloads what it lists. The dated archive is
   asked only for an hour `nrt/` doesn't have (old days). Answers other
   than 200 are counted and logged after the first pass that has any, then
   at most once an hour: `archive: 16 answer(s) other than 200 since … (503
   ×16), e.g. HTTP 503 <url>`. Tests with directory listings recorded from
   JSOC (`tests/fixtures/jsoc_nrt_20261124_H2{2,3}00.html`).
4. **Hangs.** A requests `timeout` bounds the connection and each read, not
   a download: every download now also has a total limit
   (`FLARE_NRT_DOWNLOAD_MAX_SEC`, 60 s). And the loop feeds systemd's
   watchdog: `flare-nrt-loop.service` is `Type=notify`, `WatchdogSec=600`,
   `TimeoutStartSec=300`; the loop sends `READY=1` once the model is loaded
   and `WATCHDOG=1` after each download pass, each slot, each replay and
   each Grad-CAM map a replay recomputes (`flare_nrt/sdnotify.py`, standard
   library, no new dependency). **The unit file changes**: copy it
   (deploy/README.md, "Upgrade from 1.3.0 to 1.3.1").
5. **Reprocess the gap.** `python -m flare_nrt.reprocess --since ISO
   [--until ISO] [--dry-run]`, a one-off: slots whose result is
   `qc_reject`, `missing` or `no_baseline` (or none) and whose FITS are
   still in staging (4 days) are run again, in order, through the loop's
   `Processor` (same QC with the new mask, baselines from staging, model,
   alert rule, event tracker). Each new row is appended after the old one
   with `reprocessed: true`, and **the last row of a slot in
   `results/YYYYMMDD.jsonl` is its result**: `ResultStore.read_day`
   returns that one per slot, in slot order, so `/api/series`,
   `/api/frames`, `/api/frame`, curation, validation and replays show the
   new rows; the old rows stay in the file (`read_day(day,
   every_row=True)`). Events found in the gap are created as usual; their
   ids come from a counter the loop and the command share
   (`state/event_ids.json`, under the `events.csv` lock), and it never goes
   past the newest slot the loop has written, so it can run next to the
   loop; `/api/events` lists events in start order. It
   writes Grad-CAM for alerting frames only and leaves the loop's state
   files alone. For this outage: `--since 2026-11-24T23:00:00Z`, before
   2026-11-28 ~23:00 UT, when the first of those FITS leave staging.

GOES during the gap: four B-class flares (B2.0–B6.4); nothing of note was
missed.

Also: version 1.3.1 (`pyproject.toml`, `flare_nrt.__version__`,
`model_version`, INTEGRATION.md, deploy/README.md with the upgrade steps
from 1.3.0); `fetch_weights` reads the `v1.3.1` release, which carries the
same two files as `v1.0.0`–`v1.3.0`; `tests/test_v100_compat.py` allows the
new `/api/status` fields.

## 1.3.0

What an event replay shows, and where an event is. **The model is
unchanged**: same weights (same two release files, same SHA-256), same
normalisation, threshold, baselines, state machine and alert rule
(`tests/test_snapshot.py` passes unchanged; `flare_nrt/model.py`,
`preprocess.py`, `detector.py`, `qc.py` and `tests/fixtures/` are
byte-identical to `v1.2.0`). Every probability, alert and event is the
one 1.2.0 gives; what changes is the experimental location of frames the
1.0.0 rule could not locate (item 2). Every 1.2.0 endpoint answers as
before; `replay.json` and `events.csv` only gain fields or fill an empty
one.

1. **A Grad-CAM map on every replay frame with a result.** Every slot
   whose result is `ok`, alerting or not, gets `gradcam-heat.webp` (same
   rendering). The loop already computes one for its newest frame every
   slot; frames without one (a backlog after a restart) now get it when
   the replay is built: the loop passes its loaded model to the replay
   builder, which recomputes the map from the stored input or the FITS
   (~10 s per such frame on the server, usually none). Each frame in
   `replay.json` gains `alert` (p ≥ threshold, null without a result) and
   `cam_peak` (the map's maximum before normalisation, as
   `X-Gradcam-Peak`); `replay.json` gains `threshold`.
2. **Why event 1 had no location, and the fix.** Not a restart: after a
   restart the loop's buffer starts empty and recomputes the 131 Å rates
   with every frame it reads, so they were always there. `localize` found
   no region: it requires the region to stand above mean + 5σ of the
   difference, and after a flare's 131 Å maximum the baselines (t−8…−12
   min) hold the flare, so the difference has a deep negative lobe whose σ
   exceeds every positive value. On the M4.0 of 2026-07-07 that left 9 of
   64 frames (14:28–14:44 UT) without a location, 4 of them alerts,
   including the peak-probability frame. Event 1 (one alert frame after
   p rose for 20 min) fits this, but its FITS are only on the server, so
   it is confirmed there: after the one-off, `event.location.method` in
   its `replay.json` is `…_robust` if this was the cause, and the first
   rule's method if the rates were missing live after all (deploy/README.md).
   Now, only when that test finds nothing, σ
   is taken from the median absolute deviation (`method`
   `diff131_rate_brightest_region_robust`). Every location the first rule
   finds is unchanged. These new locations are not validated: on the
   M4.0 they fall 27–232″ from the kernel, on the post-flare loops and
   the eruption next to it, so the loop never lets one replace an event's
   location (an event keeps the one 1.2.0 gave it; one with none takes
   it). The loop also reads 131 Å rates it does not hold in memory from
   the FITS, and logs an alert frame it cannot locate and why.
3. **Every event gets a location.** When a replay is built, alert frames
   without a location get one computed from the 131 Å FITS against the
   baselines their result used; an event without a location gets its peak
   frame's, in `replay.json` (`event.location`, `peak.location`) and in
   `events.csv` (`/api/events`): filled once under a lock shared with the
   loop's appends, never changed. Frames: `location` for alert frames,
   null otherwise; `missing["location"]` lists alert frames left without
   one (FITS gone, or no brightening) and `note` says why. INTEGRATION.md
   has a "Location" section: what it is (the 131 Å brightening, not the
   Grad-CAM), its two methods and its limits.
4. **Image geometry.** `frames[].geometry`, `peak.geometry` and the
   replay's `geometry` (the peak frame's): disk centre and radius in px
   and arcsec per px on the 512-px north-up images, from that frame's
   131 Å header (`render.image_geometry`, the convention of `localize` and
   of the heat maps' limb circle), with the formula to place a location.
5. **One-off rebuild, `--fits-root`.** `python -m flare_nrt.replays`
   rebuilds replays built before 1.3.0 (no `geometry`), reusing their
   files, adding maps for non-alert frames, `cam_peak`, locations and
   geometry. `--fits-root PATH` (repeatable; also `FLARE_NRT_FITS_ROOTS`):
   directories laid out like staging (`PATH/YYYYMMDD/…`), searched right
   after it; event 1's FITS are in `/srv/flare-nrt/keep/20261111`. It
   prints per event the frames, the location, the maps recomputed, the time
   and the peak RSS, and what is missing.
6. **Measured** (M4.0 of 2026-07-07, real weights, 2 threads, Ryzen 7
   7700X): a replay as the loop builds it, 3.3 s, nothing recomputed,
   +13 kB per event (`replay.json` 11 → 24 kB; the heat frames were
   already there); in event 1's situation (maps of the alert frames only,
   FITS under `--fits-root`) the one-off recomputed 45 maps in 43.6 s at
   1.0 GB peak RSS (model loaded once per run), +0.96 MB on disk (45 heat
   frames, 21 kB each). The recomputed maps equal the loop's (`cam_peak`
   identical on all 54 frames). Unit limits unchanged.

Also: version 1.3.0 (`pyproject.toml`, `flare_nrt.__version__`,
`model_version`, INTEGRATION.md, deploy/README.md, which has the upgrade
steps from 1.2.0 with the one-off and `--fits-root /srv/flare-nrt/keep`);
`fetch_weights` reads the `v1.3.0` release, which carries the same two
files as `v1.0.0`–`v1.2.0`; model card: the robust fallback and its
limits.

## 1.2.0

Display only, like 1.1.0: what an event replay keeps. **The model is
unchanged**: same weights (same two release files, same SHA-256), same
normalisation, threshold, baselines, state machine and alert rule
(`tests/test_snapshot.py` passes unchanged; `flare_nrt/model.py`,
`preprocess.py`, `detector.py`, `qc.py`, `localize.py`, `render.py`,
`loop.py` and `tests/fixtures/` are byte-identical to `v1.1.0`). Every
1.1.0 endpoint answers as before; the additions are internal (127.0.0.1).

1. **131 Å in every replay frame.** `events/<id>/frames/<slot>/131.webp`,
   512 px, the same bytes as `/api/image/131.png?kind=img&format=webp`,
   rendered from the FITS when the replay is built (they are still on
   disk then) and kept with the replay (365 days). `REPLAY_CHANNELS` gains
   `131`; `GET /api/events/{id}/frames/{ISO}/131.webp` serves it;
   `replay.json` lists it in each frame's `images` and slots without it in
   `missing["131"]`.
2. **The peak frame, once per event.** `events/<id>/peak/`: the four
   channels at the peak slot (`131`, `171`, `304`, `1600`.webp; `kind=img`,
   512 px) and its Grad-CAM in both styles (`gradcam-heat.webp`,
   `gradcam-overlay.webp`), each byte-identical to what `/api/image` and
   `/api/gradcam` give for that slot with `format=webp`. `GET
   /api/events/{id}/peak/{name}.webp` serves them; the Grad-CAM ones carry
   `X-Gradcam-Slot`, `X-Gradcam-Prob` and `X-Gradcam-Peak`. `replay.json`
   gains `peak` (`slot`, `t_obs`, `probability`, `gradcam_peak`, `images`)
   and `missing["peak"]`. The schema stays `flare-nrt-replay/1` (additions
   only).
3. **The one-off command adds the new files to existing replays.** `python
   -m flare_nrt.replays` now also rebuilds replays built by 1.1.0 (no
   `peak` key), reusing the files they have. Where the FITS still exist it
   renders 131 Å and the peak's channels. Where they are gone, `replay.json`
   lists the missing files (`missing`, `complete: false`, `note`), as
   before. The peak's Grad-CAM (kept 400 days for alerting frames) is
   added either way. When a map has to be recomputed, the overlay is
   redrawn from it and the stored model input. The live loop builds new
   replays with the new files and leaves old ones to the command.
4. **Disk and CPU, measured** on the M4.0 of 2026-07-07 with the real
   weights (55-slot window): 131 Å frames 86 kB each (4.7 MB per event),
   peak set 0.30 MB, so **9.3 MB per event** (4.2 MB in 1.1.0) and
   ≈ 3.5–6 GB per year of replays at the calibration month's activity
   (1.5–3 GB in 1.1.0). Rendering a 131 Å frame takes 0.08 s (median), so
   the loop spends ≈ 5 s once per event. INTEGRATION.md "Resources".
5. **`PrivateTmp` for one-off and validation runs.** The `systemd-run`
   one-off documented for 1.1.0 lacked `-p PrivateTmp=yes`.
   `ProtectSystem=strict` makes `/tmp` read-only and torch needs a writable
   temporary directory, so the command failed. Fixed in README.md and
   deploy/README.md (upgrade steps from 1.1.0). `flare-nrt-validate.service`
   had the same problem (it imports torch through `curate`): it gains
   `PrivateTmp=true` like the other units.

Also: version 1.2.0 (`pyproject.toml`, `flare_nrt.__version__`,
`model_version`, INTEGRATION.md, deploy/README.md); `fetch_weights` reads
the `v1.2.0` release, which carries the same two files as `v1.0.0` and
`v1.1.0`.

## 1.1.0

Display only: how results are rendered and kept for the portal. **The
model is unchanged**: same weights (same two release files, same SHA-256),
same normalisation, threshold, baselines, state machine and alert rule, so
every probability is the one 1.0.0 gives (`tests/test_snapshot.py` passes
with the same expected value; `flare_nrt/model.py`, `preprocess.py`,
`detector.py`, `qc.py` and `localize.py` are byte-identical to `v1.0.0`).
The paper uses `v1.0.0`; that tag does not move. All new endpoints are
internal (127.0.0.1) and additive; every 1.0.0 endpoint called without the
new parameters answers as before (`tests/test_v100_compat.py` runs the
`v1.0.0` tag itself on the same data and compares), except as listed.

1. **Grad-CAM heat style.** `/api/gradcam/{latest|ISO}.png?style=heat`:
   the map alone, normalised to its own maximum, inferno colours,
   transparent below 5 % of the maximum, the solar limb as a thin circle,
   no 131 Å underlay. Headers `X-Gradcam-Peak` (the ensemble-mean map's
   maximum before normalisation), `X-Gradcam-Prob`, `X-Gradcam-Slot`. The
   colour bar's stops are in INTEGRATION.md. `style=overlay` (1.0.0's
   composite) stays the default. To serve it, the loop now keeps the map
   itself next to each Grad-CAM PNG (`gradcam/…/HHMM.cam.npz`, 36 kB, same
   retention); slots whose Grad-CAM was made by 1.0.0 have no map and give
   404 for `style=heat`. Comparison on real frames (2026-07-07, quiet
   13:50 UT and the M4.0 peak at 14:30): `docs/img/v1.1.0-gradcam-overlay-vs-heat.jpg`.
2. **1600 Å rendering.** It used the asinh stretch of the EUV channels,
   which put the whole UV disk at 0.66–0.98 of the colour range (a pale,
   uniform disk). Each channel now has its own limits and stretch
   (`render.SCALING`); 1600 Å is linear from 0 to the 99.9th percentile,
   so limb darkening and plage show. 131/171/304 Å are unchanged
   (byte-identical). The API's cache name for 1600 Å images changes, so a
   1.0.0 file is never served. Before/after: `docs/img/v1.1.0-1600-before-after.jpg`.
3. **Formats and sizes.** `format=png|webp` on `/api/image/…` and
   `/api/gradcam/…` (default png; WebP at quality 90: 171 Å 56 kB vs
   244 kB). `size=512|1024` on `/api/image/{wave}` for `kind=img` (default
   512; 1024 from the 1024² FITS); `diff` and Grad-CAM stay 512.
   `Content-Type` follows the format. **Changed default:** an image
   addressed by its slot is now `Cache-Control: public, max-age=31536000,
   immutable` (was max-age 3600, Grad-CAM 86400), except a difference whose
   slot has no result yet (max-age 3600); `latest` stays max-age 60.
4. **Pre-rendered loop.** For each live result the loop writes
   `render/YYYYMMDD/HHMM/171.webp`, `171-diff.webp` and `gradcam-heat.webp`
   (0.21 s CPU per result), kept and pruned with the FITS (4 days).
   `GET /api/loop?channel=171|171-diff|gradcam-heat&hours=1..24` lists them
   in time order; `GET /api/render/{ISO}/{channel}.webp` serves them.
5. **Event replays kept a year.** 60 min after an event closes, the loop
   builds `events/<event_id>/` (`replay.json` + 171 Å and heat frames from
   t_start − 30 min to t_end + 60 min, ≈ 4 MB), kept 365 days
   (`FLARE_NRT_REPLAY_DAYS`) by the existing retention. An event whose
   frames are gone gets a replay that says so (`complete`, `missing`,
   `note`). `GET /api/events/{id}/replay`, `GET
   /api/events/{id}/frames/{ISO}/{171|gradcam-heat}.webp`; `/api/events`
   rows gain `replay: true|false`. One-off for events already in the
   archive: `python -m flare_nrt.replays` (recomputes the Grad-CAM maps
   1.0.0 did not keep; rebuilds incomplete replays, reusing their frames).
6. **Limits and docs.** CPU and disk measured on real frames (INTEGRATION.md
   "Resources"); unit limits unchanged. New settings
   `FLARE_NRT_REPLAY_DAYS`, `_REPLAY_PRE_MIN`, `_REPLAY_POST_MIN`,
   `_WEBP_QUALITY` (`.env.example`, deploy/README.md, which also has the
   upgrade steps from 1.0.0). `FLARE_NRT_IMAGE_CACHE_DAYS`, which existed,
   is now listed in `.env.example`.

Also: version 1.1.0 (`pyproject.toml`, `flare_nrt.__version__`,
`model_version`, INTEGRATION.md, deploy/README.md); `fetch_weights` reads
the `v1.1.0` release, which carries the same two files as `v1.0.0`. CI
checks out the full history so the compatibility test can run the
`v1.0.0` tag.

## 1.0.0

First deployment on the portal's server. No code change from 0.9.0: only
the version moves (`pyproject.toml`, `flare_nrt.__version__`, so
`model_version` and the portal's provenance read `1.0.0`; `INTEGRATION.md`;
`deploy/README.md` clones `v1.0.0`; `fetch_weights` reads the `v1.0.0`
release). The release carries the same two files as 0.9.0, with the same
SHA-256, so the pins in `flare_nrt/fetch_weights.py` do not change. This
tag is the version frozen for the paper.

## 0.9.0 — 2026-11-10

First published release, after the review of the portal maintainers
(2026-11-06). The model and its weights are the same as in internal
version 2.5 (same two files, same SHA-256); what changes is the name,
the numbering and the defaults for the portal's server. Each item below
answers the review item of the same number.

1. **Repository restored to git.** `.gitignore`, `.env.example` and the
   CI workflow (`.github/workflows/ci.yml`: ruff, pytest, and a check that
   the old name is gone) are back; they had been dropped by an upload
   through GitHub's web page. Changes go through git commits from now on.
2. **Renamed: the model is no longer called "Centella". It is `flare-nrt`**
   until the maintainers choose a name:
   - Python package `centella` → `flare_nrt` (`python -m flare_nrt.loop`,
     `uvicorn flare_nrt.server:app`); distribution name `flare-nrt`;
   - settings `CENTELLA_*` → `FLARE_NRT_*`;
   - units `centella-*` → `flare-nrt-*`; system user `centella` →
     `flare-nrt`; `/opt/centella`, `/srv/centella`, `/etc/centella/` →
     `/opt/flare-nrt`, `/srv/flare-nrt`, `/etc/flare-nrt/`
     (`flare-nrt.env`);
   - `/api/status` `model` → `"flare-nrt"` (the portal's provenance
     producer becomes `flare-nrt@0.9.0`); the HTTP `User-Agent` and the
     viewer's titles follow;
   - archive schema `centella-archive/1` → `flare-nrt-archive/1` (same
     layout; nothing had been pulled off-server yet);
   - the validation report's per-flare key `centella` → `flare_nrt`.
   The repository keeps its name, `HelioUSACH/flare-nrt`.
3. **No portal code in this repository.** The proposed portal patches are
   not part of it; the maintainers design the portal's flare page. The
   internal display endpoints (`/api/events`, `/api/series`,
   `/api/frames`, `/api/image/…`, `/api/gradcam/…`, `/api/health`,
   `/viewer`) are unchanged: they are what the portal will read.
4. **Versions restart at 0.9.0** (`pyproject.toml`, `flare_nrt.__version__`,
   `/api/status` `model_version`, `INTEGRATION.md`). The internal versions
   1.x–2.5 are kept below. The weights are fetched from the `v0.9.0`
   release.
5. **`tests/test_real_weights.py`** skips unless all four of
   `FLARE_NRT_REAL_WEIGHTS`, `_REAL_NORM`, `_REAL_FITS` and `_REAL_EXPECT`
   are set (it failed with `KeyError` when only the first one was).
6. **`FLARE_NRT_TORCH_THREADS` defaults to 2**, matching the loop unit's
   `CPUQuota=200%` (was 4).
7. **`FLARE_NRT_STAGING_DAYS` defaults to 4** (was 7): ≈ 11 GB of raw FITS,
   ≈ 17–20 GB for the whole data directory. Tensors stay at 2 days.
   `INTEGRATION.md`, `deploy/README.md` and `docs/archive-format.md` give
   the new numbers.
8. **`INTEGRATION.md`'s open questions are answered:** disk → 4 days of
   FITS; off-server copy → the maintainers pull `/srv/flare-nrt/archive`
   to a backup server, and the archive layout stays stable and versioned;
   display endpoints → the internal contract for now.

Also: `deploy/README.md`'s install block runs as is (it creates the user
and writes the three paths into the settings file).

## Pre-release development (internal versions 1.x–2.5)

These versions were never published. Their numbers are internal.

### 2.5.0 — 2026-11-04

- `location` adds `bbox_arcsec` [x0, y0, x1, y1]: the bounding box of the
  detected region in helioprojective arcsec (x west, y north). Pages use it
  to draw a box around the flare. Additive: the other fields are unchanged.

### 2.4.1 — 2026-11-03

- The experimental 131 Å rule added in 2.4.0 is removed (owner's
  decision: the product is the neural network only). Results, status and
  events are exactly as in 2.3.0; `events_aia131.csv` files left by 2.4.0
  are no longer read.

### 2.4.0 — 2026-11-03 (withdrawn)

Ran the 131 Å rule live as an experimental signal next to the model's.
Withdrawn in 2.4.1.

### 2.3.0 — 2026-11-03

Location validated against a catalogue. Model card updated with the
calibration month replayed in 2.3. Detection is beaten out of sample by
the `aia131_rise` rule (98 % vs 72 % of M/X flares); location error has a
median of 14″; 2 of 3 baselines are identical to 3 of 3; low-probability
alternation is mild (0.60).

- `python -m flare_nrt.goes` now writes flare positions (`hpc_x`, `hpc_y`,
  helioprojective arcsec). Each SWPC flare takes the position LMSAL's "SSW
  Latest Events" gives in AIA for the same flare (peak within 6 min, same
  letter), else SWPC's own; (0, 0) means none. `Flare.hpc` is not part of a
  flare's identity.
- `flare_nrt.validate` reports `location`: for each detected M/X flare with
  a position, the distance between flare-nrt's location at its
  highest-probability alert and the catalogue (median, p75, fractions
  within 60/120/240″).

### 2.2.1 — 2026-11-02

- `/api/flares` adds SWPC's 7-day list for windows that reach the last 7
  days. The curated archive only covers days up to yesterday, so live pages
  were missing today's flares. The list is fetched at most every 10 min and
  the last good copy is kept on failure; `source` says `archive+swpc`.

### 2.2.0 — 2026-11-02

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
- `FLARE_NRT_FITS_DIRS`: the API also finds FITS by name in these
  read-only trees, so a replayed month is browsed without copying it.
  `FLARE_NRT_FLARES_CSV`: the GOES list shown by the viewer (default: the
  curated archive's).
- Replay (`--replay`) keeps the Grad-CAM of every alerting frame (no
  tensors; `--no-gradcam` to skip), logs progress every 12 h of data, and
  `--resume` continues after a stop. Replay rows no longer carry
  `input_age_s`, which is only meaningful live.
- `python -m flare_nrt.goes --start --end --out`: the SWPC flare list of a
  past period, from HEK.

### 2.1.0 — 2026-10-28

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
- 2 of the 3 baseline frames suffice (`FLARE_NRT_MIN_BASELINE=2`); the
  validated rule used 3 of 3 (`FLARE_NRT_MIN_BASELINE=3` restores it).
- Every result records `aia131_excess` (disk-integrated 131 Å rise over the
  same baseline), the input of the baseline rule.

**Storage:** raw FITS 7 × 24 h (was 3 days), model-input tensors 2 days
(new, `tensors/`), Grad-CAM 2 days and 400 days for alerts (new), local
archive 7 days (was 14), validation reports bounded. Steady state ≈ 25 GB.

**Weights:** `python -m flare_nrt.fetch_weights` downloads the release assets
and verifies their SHA-256; the SHA-256 is pinned as the default of
`FLARE_NRT_WEIGHTS_SHA256`; a file is never loaded unverified, and only with
`weights_only=True` (the fallback to `weights_only=False` is gone).

**Validation:** `flare_nrt.validate` scores the named baseline
`aia131_rise` at flare-nrt's false-alarm count and reports the
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

### 2.0.1 (folded into 2.1.0)

- QC: QUALITY bit 30 (0x40000000, quicklook) is no longer a reject. Every near-real-time AIA image has it, so in live mode every slot was `qc_reject` and no probability was ever made.
- Live source: each `mostrecent/` image is filed (as a copy) under every 2-min slot within 90 s of its T_OBS (the QC time tolerance), not only under floor(T_OBS). Before, images taken late in their slot failed `time_offset` and slots rarely had all 4 channels.
- Loop: on a cold start it no longer waits for slots from before the start that only the lagging archive could fill. `FLARE_NRT_SKIP_AFTER_MIN` default 20 → 12.
- API: `/api/image` and `/api/channels` show the newest staged image of each channel until the first full result exists (the difference once t−8/−10/−12 are staged).

### 2.0.0 — 2026-10-26

Recalibration on live synoptic data; same network weights as 1.x.

- **Live source** reads JSOC `mostrecent/` every poll (filed under the slot of
  its own T_OBS) besides the dated archive, which lags behind real time;
  `FLARE_NRT_STAGING_HOURS` keeps only the last N hours of raw FITS on a
  small disk; a cold start skips the slots the archive lacks.
- **Python 3.10+** (was 3.11): `dt.timezone.utc`, and one ISO parser
  (`grid.parse_datetime`) for every provider time spelling.

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
