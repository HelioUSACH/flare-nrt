# Integration sheet: flare-nrt (`usach.flare.nrt`)

| | |
|---|---|
| Model / product id | Model `flare-nrt` (until the maintainers choose a name) → product `usach.flare.nrt`. |
| Kind | event (detection) |
| Target (forecasts) | n/a |
| Owner(s) | Nicolás Campos (HelioUSACH); supervisor and deployment: Víctor A. Pinto |
| Repo, current tag | `HelioUSACH/flare-nrt`, `v1.3.1` (Python package `flare_nrt`). This sheet is updated with every release |
| Run cadence | One result per AIA synoptic slot (2 min), as soon as its 4 channels and at least 2 of its 3 baseline frames are on disk |
| Inputs | **AIA synoptic 131/171/304/1600 Å**, 1024², level 1.5 quicklook. Source: `https://jsoc1.stanford.edu/data/aia/synoptic/mostrecent/AIAsynoptic{0131,0171,0304,1600}.fits`, polled every 30 s. Gaps are filled from JSOC's near-real-time archive (`…/synoptic/nrt/YYYY/MM/DD/HHH00/`, current to ~20 min, 3-min cadence; since 1.3.1), or, for an hour it lacks, from the dated archive (`…/synoptic/YYYY/MM/DD/HHH00/`), which lags real time by weeks; queried at most every 10 min. Frames whose header QUALITY has a bit other than 30 (near-real-time) and 2 (`Q_NOASD`, on every near-real-time frame since 2026-11-24 23:00 UT; ignored since 1.3.1) are rejected (`FLARE_NRT_QC_IGNORE_QUALITY_MASK`); each result row keeps the raw QUALITY (`quality`, `quality_soft`). Typical latency from T_OBS to availability: ~7–10 min (measured on 2026-10-28; `data_latency_s` in every result). Public (NASA/SDO). ≈ 2.75 GB/day. **GOES flare list** (labels for curation and validation only, daily): `https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json`, public domain, < 50 kB/day |
| New data the portal must collect | none (the portal already stores GOES XRS, which it compares itself) |
| Latest-result endpoint | `GET http://127.0.0.1:5050/api/status`; fields below |
| Freshness | Computed per request. The result is `stale` if the loop heartbeat or the last result is older than `FLARE_NRT_STALE_AFTER_SEC` (600 s, the manifest's `stale_after: 10m`), or if `input_age_s` (now − newest T_OBS of the last result) exceeds `FLARE_NRT_MAX_INPUT_AGE_SEC` (900 s). `stale_reason` says which. In the loop, a frame whose newest T_OBS is already older than 900 s is not run through the model (`last_frame.status = "inputs_old"`) |
| Resources | Measured 2026-10-28 with the real weights on 2 vCPU (Intel Xeon 2.8 GHz), 2 torch threads. **Inference 10.4 s per frame**; **Grad-CAM 10.5 s**, newest frame and alerts only (in steady state the newest frame is every frame; slots of a backlog get theirs when their event's replay is built, 1.3.0). Peak RSS **1.14 GB**. **Display renders (1.1.0):** the 3 WebP files per result take **0.21 s CPU** (median; max 0.25 s) against 2.53 s for inference + Grad-CAM on the same machine (AMD Ryzen 7 7700X, 2 torch threads, 65 real frames), i.e. ≈ 8 % there and < 3 % of the server's ~21 s per frame even if rendering is 3× slower on its Xeon; peak RSS of that run 0.91 GB. An event replay is built in 0.02 s from copies (1.1.0); since 1.2.0 the loop also renders its 131 Å frames and the peak's 4 channels from the FITS: 0.08 s per 131 Å frame (median; max 0.19 s), ≈ 5 s once per event of 55 slots (a build with nothing in `render/`, all from FITS and stored maps, took 9.4 s), peak RSS unchanged. **Replays in 1.3.0** (the same M4.0, 55 slots, real weights, 2 torch threads, measured on the Ryzen above): built as the loop builds it (maps and `render/` there), **3.3 s** and no Grad-CAM to recompute; with event 1's situation (maps of the 9 alert frames only, no stored inputs, no `render/`, FITS under `--fits-root`), the one-off recomputed **45 maps from the FITS in 43.6 s** (67 s CPU; ≈ 1 s per map with its FITS reads), **peak RSS 1.0 GB** with the model loaded once per run. On the server's Xeon (≈ 8× slower per frame than that machine) and `CPUQuota=100%`, expect ≈ 6–12 min for such an event; in the loop, ≈ 10 s per backlog slot without a map, usually none The API renders a 1024 px WebP in ≈ 0.1 s (PNG 0.5–0.8 s), peak RSS ≈ 0.1 GB. The unit limits stay as they are (`MemoryMax=3G`, `CPUQuota=200%`; API `512M`, `100%`). At one frame per 2 min that is ≈ 18 % of 2 cores on average; a backlog after a restart is skipped down to the last 15 min. No GPU. **Disk at steady state ≈ 17–20 GB** with the defaults: FITS 4 d × 2.75 GB = 11.0 GB (`FLARE_NRT_STAGING_DAYS=4`), tensors 2 d × 2.3 GB = 4.6 GB, Grad-CAM ≈ 0.4 GB, archive 1.2–3.9 GB extra (it is kept 7 days and its files are hard links into the FITS, so only its 3 oldest days, past the FITS, add space: 0.4–1.3 GB per day), results ≈ 0.4 GB (400 d), preview cache < 0.1 GB at 512 px (see below for 1024 px). **Display files (1.1.0, 1.2.0)**, measured on real frames (M4.0, 2026-07-07): `render/` 137 kB per result (171.webp 56 kB + 171-diff.webp 60 kB + gradcam-heat.webp 18 kB) × 30/h × 96 h ≈ **0.4 GB**; Grad-CAM maps (`gradcam/*.cam.npz`, 36 kB) ≈ 0.05 GB for 2 days plus ≤ 0.15 GB/yr for alerting frames (400 d); `events/` **9.3 MB per event** since 1.2.0 (M4.0 of 2026-07-07, 55-slot window, 55 frames: 171 Å 3.1 MB (56 kB per frame) + heat 1.1 MB (20 kB) as in 1.1.0's 4.2 MB, + **131 Å 4.7 MB (86 kB per frame)** + **peak set 0.30 MB** (131 Å 78 kB, 171 Å 57 kB, 304 Å 71 kB, 1600 Å 48 kB, heat 17 kB, overlay 31 kB); a window is 45 slots + the event's length / 2 min); **1.3.0 adds 13 kB per event** when the loop already had a map for each result (only `replay.json` grows, 11 → 24 kB), and up to ≈ 1 MB for an event whose non-alert maps had to be computed (event 1's situation: 45 heat frames of ≈ 21 kB, 9.1 MB against 8.2 MB with 1.2.0) and ≈ **3.5–6 GB per year** (1.1.0: 1.5–3) at the calibration month's activity (1.4 % of slots alerting ≈ 284 alert frames per 28 days, 5–10 alert frames per event → 30–55 events per month; a quiet Sun gives far fewer; check with `du -sh /srv/flare-nrt/events` after a month live). With a year of replays the data directory is ≈ 21–27 GB. **1024 px previews:** a 1024 px image is 180–370 kB as WebP and 0.9–1.3 MB as PNG; if the portal fetches the latest 4 channels at 1024 px every slot, `cache/` holds ≈ 0.8 GB (WebP) or ≈ 3.4 GB (PNG) for its 1 day (`FLARE_NRT_IMAGE_CACHE_DAYS`, which can be lowered, e.g. 0.25). Floor: `FLARE_NRT_MIN_FREE_GB` (10 GB) deletes tensors, then archive, then FITS older than 2 days, oldest first |
| Weights | Release asset `deployment_ensemble_thr058.pt`, 224,246,080 bytes, SHA-256 `d3b850e3ec3d2670f4cb9327c397f7391ed1b082d2b6d154bca1b57626676006`. Also `global_norm_stats.json`, 808 bytes, SHA-256 `807723e30a6a1c0e066c003bb6ad8c3a926a14e2ad2afd2f64f0aff46b83c075`. Both are fetched and verified by `python -m flare_nrt.fetch_weights` |
| Baseline it must beat | (1) **GOES XRS event start** (SWPC): the reference that defines the labels; not beaten on timing (median alert 9 min after it, 1 min before the peak). (2) **`aia131_rise`**, a non-neural rule on the same frames: **not beaten on detection**. On the calibration month, out of sample, it detects 98 % of M/X flares with 5 false alarms; flare-nrt detects 72 % with 3. flare-nrt's demonstrated contribution is the location (median 14″ from catalogue positions). Details in the model card |
| Known limitations | docs/model-card.md "Known limitations" |
| Model card | `docs/model-card.md` |

## `GET /api/status`

The portal's adapter reads these fields (D-096); their names and meaning
have not changed since the first internal versions: `live_status`, `timestamp`,
`seconds_since_update`, `probability`, `flare_detected`, `location`,
`ensemble_std`. Everything else is an addition. Every time is ISO 8601
UTC with `Z`.

| field | type | units | meaning |
|---|---|---|---|
| `live_status` | string | | `live` / `stale` / `no_data`, computed on every request |
| `stale_reason` | string or null | | `loop_not_running`, `no_new_result` or `inputs_old` |
| `timestamp` | ISO `Z` | | when the latest result was made (the portal's `issued`; unique per result) |
| `last_update` | ISO `Z` | | same as `timestamp` |
| `seconds_since_update` | number | s | now − `last_update` |
| `observation_time` | ISO `Z` | | the AIA slot the result is about |
| `inputs` | object | | `{"131": T_OBS, "171": …, "304": …, "1600": …}`: the provider time of each channel used (ms, `Z`) |
| `input_age_s` | number | s | now − newest of `inputs`, per request |
| `max_input_age_sec`, `stale_after_sec` | integer | s | the freshness thresholds |
| **`state`** | string | | `quiet` (no event open), `onset` (this result opened an event), `active` (an event is open and this result alerts), `decay` (an event is open, this result does not alert; it closes when no alert follows within 10 min) |
| **`probability`** | number | 1 | the signal shown to the public: ensemble mean in [0, 1], the value the alert rule compares with `threshold` |
| **`flare_detected`** | bool | | an event is open: `true` in `onset`, `active` and `decay` |
| `raw_score` | number | 1 | the raw ensemble mean, labelled separately. It equals `probability`: there is no gating or smoothing between them. A high probability always opens or continues an event, so `probability ≥ threshold` with `flare_detected: false` cannot happen |
| `alert` | bool | | `probability ≥ threshold` for this slot |
| `threshold` | number | 1 | 0.80 |
| `event_id`, `event_t_start` | integer / ISO `Z`, or null | | the open event |
| `location` | object or null | | **experimental**: `centroid_arcsec` [x, y] helioprojective, `bbox_arcsec` [x0, y0, x1, y1] (the region's bounding box), `area_px`, `area_arcsec2`, `on_limb`, `method`, `confidence: "experimental"`. Only while an event is open. What it is and its limits: "Location" below |
| `ensemble_scores` | list of number | 1 | the 5 members' probabilities |
| `ensemble_std` | number | 1 | their standard deviation |
| `lag_min`, `baseline_slots`, `baseline_missing`, `n_baseline_used` | | min / ISO lists / int | the baseline frames actually used (t−8, −10, −12 min; at least 2 of 3) |
| `last_frame` | object | | the newest slot the loop looked at, whether or not it produced a result: `slot`, `made_at`, `status` (`ok` / `missing` / `qc_reject` / `no_baseline` / `inputs_old`), `reason`, `qc` (QC reasons), `channels_missing`, `baseline_missing`, `input_age_s`, `inputs` |
| `data_latency_s` | integer | s | processing time − slot time |
| `latency_ms` | integer | ms | inference time |
| `model`, `model_version`, `weights_sha256` | string | | `"flare-nrt"`, `"1.3.1"`, the weights' hash. The portal's provenance producer is `flare-nrt@<model_version>` |
| `frame_count`, `loop_errors`, `loop_last_error`, `loop_last_error_ts` | | | loop health since start |
| `last_ok_slot` (1.3.1) | ISO `Z` or null | | the newest slot with a result (`ok`) |
| `rejected_streak` (1.3.1) | integer | slots | slots in a row without a result (any status but `ok`), up to the newest the loop processed; 0 while results flow. 15 (30 min) logs a WARNING, then one every hour |
| `rejected_since` (1.3.1) | ISO `Z` or null | | the first slot of that run |
| `rejected_streak_reasons`, `last_reject_reasons` (1.3.1) | object / list | | the run's reason codes counted (e.g. `{"quality:131=1073741828": 15, …}`; QC codes, `missing:<wave>`, `baseline_missing`, `inputs_old`, `error`), and the newest slot's |
| `heartbeat`, `last_download` (1.3.1) | ISO `Z` or null | | the loop's last pass, and when it last stored a new FITS file (kept across restarts) |
| `server_time` | ISO `Z` | | |

Not sent: `severity`. It was exploratory in an early internal version and was removed.

**Compatibility:** releases only add fields to `/api/status`; the fields
above keep their names and meaning. One suggested change, not required: use
`observation_time` as the event's `valid_from` instead of `timestamp`, so
the chart shows observation time rather than processing time.

## Display endpoints (internal contract)

These serve what the portal will display. The maintainers keep them as the
internal contract for now (2026-11-06); the portal side may ask for
changes. Everything is internal on 127.0.0.1; the portal proxies or copies
what it shows.

| endpoint | returns |
|---|---|
| `GET /api/events?days=30` (1–400) | `{"events": [{event_id, t_start, t_pmax, t_end, p_max, n_alert_frames, location, closed, gradcam, replay}]}`: alert history; `t_start` = first alert, `t_pmax` = highest probability, `t_end` = last alert; `location` (experimental, "Location" below) = the open event's, or, for an event closed without one, its peak frame's once its replay is built (1.3.0; filled once, never changed); `gradcam` = URL of the peak frame's Grad-CAM, or null; `replay` (1.1.0) = whether the event has a replay to show (`/api/events/{event_id}/replay` exists and has at least one frame); in start order (1.3.1: events of a reprocessed gap are numbered after the loop's, ids are not in time order), the open event last, `closed: false` |
| `GET /api/series?days=3` (≤ 14) | `{"threshold", "lag_min", "fields": ["slot","probability","state","status"], "points": [[…], …]}`: probability per slot for the time series, oldest first; `probability` null when a slot had no result (`status` says why) |
| `GET /api/frames?hours=6` (≤ 168) | `{"frames": [{slot, status, state, probability, alert, inputs, images: [waves on disk], gradcam: bool}]}`: recent frames and what can be shown for each |
| `GET /api/image/{131,171,304,1600}.png?kind=img\|diff&slot=ISO&format=png\|webp&size=512\|1024` | North up: `img` = DN/s with per-channel limits and stretch (131/171/304 asinh to the 99.7th percentile; 1600 linear to the 99.9th, since 1.1.0), SDO-style colours; `diff` = current − mean of the baselines used, grey ±. Without `slot`: the latest result. `format` (1.1.0): `png` (default) or `webp` (quality 90). `size` (1.1.0): 512 (default) or 1024, 1024 for `kind=img` only (rendered from the 1024² FITS); else 422. Available while the FITS are kept (4 days). Header `X-Observation-Time` |
| `GET /api/gradcam/latest.png`, `GET /api/gradcam/{ISO}.png`, `?style=overlay\|heat&format=png\|webp` | 512 px, north up, Grad-CAM of the ensemble (layer4, mean over members). `style=overlay` (default, as in 1.0.0): over the 131 Å input, colour intensity scaled by probability / threshold, so a quiet frame is a faint tint. `style=heat` (1.1.0): see "Grad-CAM heat style" below. `format` as above. Latest frame: kept 2 days; alerting frames: 400 days |
| `GET /api/channels` | the 4 channels of the default image: slot, T_OBS, whether the difference exists |
| `GET /api/results?hours=24` (≤ 168) | `points` = [{t, p, status, state, event_id}] |
| `GET /api/health` | 24 h coverage, status counts, median data latency, last download, free disk |

**Caching (1.1.0):** every image has the `Content-Type` of its format
(`image/png`, `image/webp`). Without `slot` (or with `latest`) it is
`Cache-Control: max-age=60`. Addressed by its slot it is
`Cache-Control: public, max-age=31536000, immutable` (was max-age 3600 for
AIA images and 86400 for Grad-CAM in 1.0.0), except a `kind=diff` whose
slot has no result yet (its baselines may still change): max-age 3600.

`/api/series`, `/api/events` and `/api/frames` also take
`start`/`end` (ISO 8601) to select any past window: ≤ 40 days for series,
≤ 400 for events, ≤ 7 for frames. A window that is too long or reversed
gets a 422.

**Retrospective viewer (internal):** `GET /viewer` is an HTML page for
browsing any period frame by frame (docs/viewer.md). It is built on the
endpoints above plus:

| endpoint | returns |
|---|---|
| `GET /api/range` | `{first, last, days}`: the span with results |
| `GET /api/frame/{ISO}` | `{slot, row, images, differences, gradcam}`: the slot's full result row (or null), the channels whose image and difference can be shown, whether a Grad-CAM exists |
| `GET /api/flares?start=&end=` | GOES flares ≥ C in the window, from `FLARE_NRT_FLARES_CSV` or the curated archive |

The portal's own page can reuse the same pieces (time series with a
cursor, image per slot) for a public "look back" view.

## Location (experimental)

`location` (in `/api/status`, `/api/events`, `replay.json`) is **where 131 Å
brightened**: the brightest connected region of the 131 Å difference in
DN/s (the current frame − the mean of the baseline frames the result used,
t−8…−12 min), smoothed, above half its maximum, in helioprojective arcsec
from the 131 Å header (`flare_nrt/localize.py`). It is computed for alert
frames only. It is **not the Grad-CAM**: the heat map shows where the
network looked, the location where the EUV image changed. They often
agree on a flare and need not.

- Fields: `centroid_arcsec` [x, y] (x west, y north), `bbox_arcsec`
  [x0, y0, x1, y1] (the region's bounding box), `area_px` (on the 512
  grid), `area_arcsec2`, `on_limb` (centroid beyond 0.9 R☉), `method`,
  `confidence: "experimental"`.
- `method`: `diff131_rate_brightest_region` when the region stands above
  mean + 5σ of the difference (the rule validated in the model card:
  median 14″ from catalogue positions, one month, 36 M/X flares);
  `diff131_rate_brightest_region_robust` (1.3.0) when it does not and the
  noise is taken from the median absolute deviation instead. That happens
  after a flare's 131 Å maximum: the baselines then hold the flare, the
  difference has a deep negative lobe and the old rule found nothing. On the
  M4.0 of 2026-07-07 that left 9 of 64 frames without a location
  (14:28–14:44 UT, 4 of them alerts). It is most likely why event 1 of
  2026-11-11 had none; the 1.3.0 one-off can confirm it on the server
  (deploy/README.md). Those
  locations are not validated: on that M4.0 they fell 27–232″ from the
  kernel, on the post-flare loops and the eruption next to it.
- An **event's** location (`/api/events`, `replay.json` `event.location`)
  is that of its highest-probability alert frame located by the first
  rule, as in 1.0.0–1.2.0; a robust location is kept only while the event
  has no other (as for event 1). An event closed without any gets its peak
  frame's when its replay is built (filled once, never changed). A frame's
  own location (`/api/status` while it is current, `replay.json`
  `frames[].location`, `peak.location`) is that frame's, whichever rule
  found it, so `peak.location` and `event.location` can differ (the M4.0:
  peak 14:30 on the post-flare loops, event at the kernel, from 14:20).
- Limits (model card): one month of validation; limb flares, two flares
  at once, and the decay phase above. `null` when the 131 Å FITS of the
  frame or its baselines are gone or 131 Å did not brighten anywhere.

## Display files (1.1.0; replays extended in 1.2.0 and 1.3.0; internal contract)

Additive and internal (127.0.0.1) like the endpoints above. They change
nothing in detection: weights, normalisation, threshold, state machine and
every probability are those of 1.0.0 (in 1.1.0, 1.2.0 and 1.3.0 alike).
1.3.0 also changes how a location is found when the 1.0.0 rule finds none
("Location" above); every location that rule found is unchanged.

### Grad-CAM heat style

`GET /api/gradcam/{latest|ISO}.png?style=heat[&format=webp]` returns the
Grad-CAM map alone, 512 px RGBA, north up, with no 131 Å underlay:

- normalised to its own maximum, so a quiet frame's map looks as strong as
  a flare's: **label it with the headers**;
- colours: matplotlib's *inferno* (perceptually uniform), the 17 stops
  below, linear in between;
- fully transparent where the map is below 5 % of its maximum, opaque
  from 10 %;
- the solar limb as a thin light-grey circle (`rgba(200,200,200,0.82)`,
  ~1.5 px), from the 131 Å header of that frame.

| header | meaning |
|---|---|
| `X-Gradcam-Peak` | the maximum of the ensemble-mean map before that normalisation, in (0, 1]. Each member's map is scaled to its own maximum before averaging, so this measures how much the 5 members agree on where to look (1 = same peak), not how strong the evidence is |
| `X-Gradcam-Prob` | the frame's probability (the same number as in its result) |
| `X-Gradcam-Slot` | the slot, ISO 8601 `Z` (also `X-Observation-Time`) |

`style=heat` needs the stored map (`gradcam/…/HHMM.cam.npz`, written
since 1.1.0). A slot whose Grad-CAM was made by 1.0.0 has only its overlay:
`style=heat` gives 404 there, `style=overlay` still works.

**Colour bar** (map / its maximum, 0 → 1), for the portal's legend:

```
#000004 #0b0724 #210c4a #3d0965 #57106e #71196e #8a226a #a32c61 #bc3754
#d24644 #e45a31 #f1731d #f98e09 #fcac11 #f9cb35 #f2ea69 #fcffa4
```

As CSS: `linear-gradient(to right, #000004 0%, #0b0724 6.25%, #210c4a 12.5%,
#3d0965 18.75%, #57106e 25%, #71196e 31.25%, #8a226a 37.5%, #a32c61 43.75%,
#bc3754 50%, #d24644 56.25%, #e45a31 62.5%, #f1731d 68.75%, #f98e09 75%,
#fcac11 81.25%, #f9cb35 87.5%, #f2ea69 93.75%, #fcffa4 100%)`; values below
5 % are transparent in the image. The stops are `INFERNO_STOPS` in
`flare_nrt/render.py`.

### Pre-rendered loop (last hours)

When a result is made the live loop also writes
`render/YYYYMMDD/HHMM/171.webp` (171 Å, 512 px), `171-diff.webp` (171 Å
current − mean of the baselines that result used) and `gradcam-heat.webp`
(only when that result has a Grad-CAM: the newest frame and every alert;
slots of a backlog have none). WebP quality 90. Kept as long as the raw
FITS (`FLARE_NRT_STAGING_DAYS`, 4 days) and pruned with them.

| endpoint | returns |
|---|---|
| `GET /api/loop?channel=171\|171-diff\|gradcam-heat&hours=6` (1–24) | `{"channel", "hours", "frames": [{"slot", "t_obs", "url", "probability", "state"}]}`, oldest first, slots in [now − hours, now] that have that file. `t_obs`: the 171 Å T_OBS (null without a result row); `url` relative to the service, e.g. `/api/render/2026-11-11T12:02:00Z/171.webp`; `probability`/`state` from the slot's result (null if none). `Cache-Control: max-age=60` |
| `GET /api/render/{ISO}/{171\|171-diff\|gradcam-heat}.webp` | one frame, `image/webp`, immutable; 404 if absent |

### Event replays (kept a year)

Once an event has been closed for 60 min (`FLARE_NRT_REPLAY_POST_MIN`) and
the loop has processed up to its `t_end + 60 min`, the loop builds its
replay once, from `t_start − 30 min` (`FLARE_NRT_REPLAY_PRE_MIN`) to
`t_end + 60 min`: every 2-min slot as `171.webp`, `gradcam-heat.webp` and
(1.2.0) `131.webp`. 171 Å and the heat map are copied from `render/`, else
rendered from the FITS (and the stored Grad-CAM map) still on disk; 131 Å
is rendered from the FITS, exactly as
`/api/image/131.png?kind=img&format=webp` (512 px, the same bytes).

**A heat map on every frame with a result (1.3.0).** Every slot whose
result is `status: "ok"`, alerting or not, gets `gradcam-heat.webp` (the
same heat style as `/api/gradcam?style=heat`, normalised to its own
maximum). The loop computes a Grad-CAM for its newest frame every slot, so
most maps exist already; a frame without one (a backlog after a restart)
gets it computed when the replay is built, from the stored model input or
from the FITS, with the model the loop already holds. Slots without a
result have no map. Because each map is normalised to its own maximum, a
quiet frame's map looks as strong as a flare's: weight it by
`probability` (or show `cam_peak`) as the portal does with
`X-Gradcam-Prob`/`X-Gradcam-Peak`.

Once per event (1.2.0) it also keeps the **peak frame** (the slot of
`t_peak`, the highest probability) as the portal shows a current one, so
an old event's peak can be displayed after its FITS (4 days) and its
2-day Grad-CAM are gone:

| file | the same as |
|---|---|
| `peak/131.webp`, `peak/171.webp`, `peak/304.webp`, `peak/1600.webp` | `/api/image/{wave}.png?kind=img&format=webp&slot=<peak>` (512 px) |
| `peak/gradcam-heat.webp` | `/api/gradcam/<peak>.png?style=heat&format=webp` (also that slot's heat frame) |
| `peak/gradcam-overlay.webp` | `/api/gradcam/<peak>.png?style=overlay&format=webp` |

They live in `/srv/flare-nrt/events/<event_id>/` and are kept 365 days
after the event's end (`FLARE_NRT_REPLAY_DAYS`), pruned whole by the
existing retention (loop hourly, curation daily).

| endpoint | returns |
|---|---|
| `GET /api/events/{event_id}/replay` | `replay.json` (below); 404 if the event has no replay (yet). `Cache-Control: max-age=3600` |
| `GET /api/events/{event_id}/frames/{ISO}/{131\|171\|gradcam-heat}.webp` | one frame of it, `image/webp`, immutable; 404 if the frame is not in the replay (`131` since 1.2.0) |
| `GET /api/events/{event_id}/peak/{131\|171\|304\|1600\|gradcam-heat\|gradcam-overlay}.webp` | (1.2.0) one file of the peak set, `image/webp`, immutable, `X-Observation-Time` = the peak slot; the two Grad-CAM files also carry `X-Gradcam-Slot`, `X-Gradcam-Prob` and `X-Gradcam-Peak`, as `/api/gradcam`. 404 if the file is not in the set (`missing["peak"]` says which) or the replay was built before 1.2.0 |

`replay.json` (schema `flare-nrt-replay/1`; 1.2.0 added `peak`,
`missing["131"]`, `missing["peak"]` and `"131"` in `images`; 1.3.0 adds
`threshold`, `geometry`, the frames' `alert`, `cam_peak`, `location` and
`geometry`, `peak.location`, `peak.geometry` and `missing["location"]`;
nothing is renamed or removed). This one is from the M4.0 of 2026-07-07,
run through the code with the real weights and built as the loop builds
it (the copy of the FITS used here lacks the files of 15:26, so that slot
is listed under `missing`):

```json
{
  "schema": "flare-nrt-replay/1",
  "event": {"event_id": 1, "t_start": "2026-07-07T14:18:00Z", "t_peak": "2026-07-07T14:30:00Z",
            "t_end": "2026-07-07T14:36:00Z", "p_max": 0.997136, "n_alert_frames": 9,
            "location": {"centroid_arcsec": [-732.7, -212.9], "bbox_arcsec": [-744.0, -225.6, -720.0, -201.6],
                         "area_px": 22, "area_arcsec2": 506.9, "on_limb": false,
                         "method": "diff131_rate_brightest_region", "confidence": "experimental"}},
  "window": {"start": "2026-07-07T13:48:00Z", "end": "2026-07-07T15:36:00Z", "pre_min": 30, "post_min": 60},
  "n_slots": 55,
  "threshold": 0.8,
  "geometry": {"size_px": 512, "disk_center_px": [256.0, 256.0], "disk_radius_px": 196.667,
               "arcsec_per_px": [4.8, 4.8], "rsun_arcsec": 944.004},
  "frames": [{"slot": "2026-07-07T13:48:00Z", "t_obs": "2026-07-07T13:48:10.348Z", "probability": 0.076028,
              "state": "quiet", "status": "ok", "images": ["171", "gradcam-heat", "131"],
              "alert": false, "cam_peak": 0.672528, "location": null,
              "geometry": {"size_px": 512, "disk_center_px": [256.0, 256.0], "disk_radius_px": 196.658,
                           "arcsec_per_px": [4.8, 4.8], "rsun_arcsec": 943.957}},
             "…",
             {"slot": "2026-07-07T14:20:00Z", "t_obs": "2026-07-07T14:20:10.349Z", "probability": 0.987029,
              "state": "active", "status": "ok", "images": ["171", "gradcam-heat", "131"],
              "alert": true, "cam_peak": 0.999883,
              "location": {"centroid_arcsec": [-732.7, -212.9], "…": "…", "method": "diff131_rate_brightest_region"},
              "geometry": {"…": "…"}},
             "…"],
  "peak": {"slot": "2026-07-07T14:30:00Z",
           "t_obs": {"131": "2026-07-07T14:30:08.071Z", "171": "2026-07-07T14:30:10.349Z",
                     "304": "2026-07-07T14:30:06.580Z", "1600": "2026-07-07T14:29:51.573Z"},
           "probability": 0.997136, "gradcam_peak": 0.969295,
           "images": ["131", "171", "304", "1600", "gradcam-heat", "gradcam-overlay"],
           "location": {"centroid_arcsec": [-766.4, -318.8], "bbox_arcsec": [-782.4, -340.8, -753.6, -297.6],
                        "area_px": 44, "area_arcsec2": 1013.8, "on_limb": false,
                        "method": "diff131_rate_brightest_region_robust", "confidence": "experimental"},
           "geometry": {"size_px": 512, "disk_center_px": [256.0, 256.0], "disk_radius_px": 196.667,
                        "arcsec_per_px": [4.8, 4.8], "rsun_arcsec": 944.004}},
  "complete": false,
  "missing": {"171": ["2026-07-07T15:26:00Z"], "gradcam-heat": [], "131": ["2026-07-07T15:26:00Z"],
              "peak": [], "location": []},
  "note": "1 of 55 slots have no 131 Å image; 1 of 55 slots have no 171 Å image: their files were gone or never made (…)",
  "built_at": "2026-07-07T18:00:00Z",
  "model_version": "1.3.1"
}
```

- `frames`: the slots that have at least one image, in time order;
  `images` says which. `t_obs` is the 171 Å T_OBS; `probability`, `state`,
  `status` come from the slot's result (null when it has none).
- `threshold` (1.3.0): the alert threshold the replay was built with.
- `frames[].alert` (1.3.0): the result's alert (`probability ≥
  threshold`); null without a result.
- `frames[].cam_peak` (1.3.0): the Grad-CAM map's maximum before
  normalisation, the same number as `X-Gradcam-Peak` (in (0, 1]; how much
  the 5 members agree on where to look, not how strong the evidence is).
  Every frame with a result has a heat map; null without a result, or when
  its map was gone and could not be recomputed (`missing["gradcam-heat"]`).
- `frames[].location` (1.3.0): the experimental location ("Location"
  above) for alert frames: the result's own, else computed when the replay
  is built from the 131 Å FITS against the baselines that result used;
  null for frames that did not alert, and for alert frames whose FITS were
  gone (`missing["location"]`).
- `frames[].geometry`, `geometry` (1.3.0): where helioprojective arcsec
  fall on the 512-px north-up images of that frame (131 Å, 171 Å, heat,
  all on the same grid), from that frame's 131 Å FITS header; null when
  it is gone. Pixel coordinates from the image's top-left corner, x right
  (west), y down (south); pixel (i, j) covers [i, i+1) × [j, j+1). A
  point (x″, y″) is at `col = disk_center_px[0] + x / arcsec_per_px[0]`,
  `row = disk_center_px[1] − y / arcsec_per_px[1]`; the solar disk is the
  circle of radius `disk_radius_px` around `disk_center_px` (the limb
  circle of the heat maps). The disk centre is fixed for synoptic images;
  the radius changes by < 0.1 % over a window (Earth's orbit), so the
  top-level `geometry` (the peak frame's) is enough to draw any frame of
  the replay; use the frame's own to be exact.
- `peak` (1.2.0): `slot` (from `t_peak`), `t_obs` (the 4 channels' T_OBS),
  `probability` (that result's), `gradcam_peak` (the map's maximum, the
  `X-Gradcam-Peak` header) and `images`, the files of the set that exist;
  `location` and `geometry` (1.3.0): the peak frame's own.
- `event.location`: the event's ("Location" above); for an event that had
  none, the peak frame's, also written to the event record
  (`/api/events`) once and never changed.
- `missing["171"]`, `missing["131"]`: slots of the window without that
  image; `missing["gradcam-heat"]`: slots *with a result* but without a
  Grad-CAM; `missing["peak"]`: files of the peak set that could not be
  made; `missing["location"]` (1.3.0): alert frames without a location. An
  event whose frames are gone (service down, retention) still gets a
  replay with what exists: `complete: false` (any image missing) and a
  `note` saying so; a missing location is explained in `note` but does
  not make a replay incomplete.
- A replay built by 1.1.0 or 1.2.0 lacks the newer keys until the one-off
  below rebuilds it.

Events and replays already on disk at an upgrade: `python -m
flare_nrt.replays` (deploy/README.md, "Upgrade from 1.2.0 to 1.3.0"; on
the server through `systemd-run` with `-p PrivateTmp=yes`). It builds
replays that are missing, rebuilds those marked incomplete or built before
1.3.0, and reuses the files they already have. It recomputes Grad-CAM maps
that are gone, from the stored input (2 days) or the FITS, and computes
the locations from the FITS. It finds FITS in staging (4 days), then in
each `--fits-root PATH` (repeatable; laid out like staging,
`PATH/YYYYMMDD/AIA….fits`), then in `FLARE_NRT_FITS_DIRS`. Where they are
gone, `replay.json` lists what is missing and the command prints it. The
peak's Grad-CAM, kept 400 days for alerting frames, is added either way.

## Stored per frame

- **Results:** `results/YYYYMMDD.jsonl` holds one row per slot, with every
  field above plus `aia131_excess`, the input of the baseline rule, and
  (1.3.1) `quality` (raw header QUALITY per channel) and `quality_soft`
  (ignored flags present, e.g. `["noasd"]`). A slot reprocessed by `python
  -m flare_nrt.reprocess` (1.3.1) has a second row, `reprocessed: true`,
  after the first: **the last row of a slot is its result**. Every
  endpoint reads it that way; a reader of the files must too.
- **Model inputs:** `tensors/YYYYMMDD/HHMM.npz` holds the exact
  (8, 512, 512) float16 input, kept 2 days.
- **Raw files:** the raw FITS stay in `staging/` for 4 days.
- **Display (1.1.0):** `gradcam/YYYYMMDD/HHMM.cam.npz` next to each
  Grad-CAM PNG (the map, its peak, the probability, the limb), same
  retention; `render/YYYYMMDD/HHMM/*.webp`, 4 days; `events/<event_id>/`
  (frames, and since 1.2.0 131 Å frames and the peak set; since 1.3.0 a
  heat frame for every result), 365 days. None of these is an input to any result.

With these, any result can be rebuilt (checklist E.23).

## Units

Example systemd units are in `deploy/`, and `deploy/README.md` explains
the layout and the install:

| unit | what | limits |
|---|---|---|
| `flare-nrt-loop.service` | the loop; `Type=notify`, `WatchdogSec=600` (1.3.1) | `MemoryMax=3G`, `CPUQuota=200%` |
| `flare-nrt-api.service` | uvicorn on 127.0.0.1:5050; writes only `cache/` | `MemoryMax=512M`, `CPUQuota=100%` |
| `flare-nrt-curate.{service,timer}` | 04:20 UT daily | `MemoryMax=1G`, `CPUQuota=100%` |
| `flare-nrt-validate.{service,timer}` | Mondays 05:00 UT | `MemoryMax=1G`, `CPUQuota=100%` |

All units run as the unprivileged `flare-nrt` user, with data under
`/srv/flare-nrt`, and log to journald.

## Questions answered by the maintainers (2026-11-06)

1. **Disk:** 4 days of raw FITS (`FLARE_NRT_STAGING_DAYS=4`, the default
   since 0.9.0, ≈ 11 GB). Tensors stay at 2 days. Totals under
   "Resources".
2. **Off-server copy of the archive:** the maintainers pull
   `/srv/flare-nrt/archive` to a backup server. Its layout
   (docs/archive-format.md, `flare-nrt-archive/1`) stays stable and
   versioned: any change gets a new schema version.
3. **Display endpoints:** kept as the internal contract for now; the
   portal side may ask for changes.
