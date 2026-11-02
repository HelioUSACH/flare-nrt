# Integration sheet: Centella (`usach.flare.nrt`)

| | |
|---|---|
| Model / product id | Centella → product `usach.flare.nrt`. The product id is unchanged; Centella is the model's name. |
| Kind | event (detection) |
| Target (forecasts) | n/a |
| Owner(s) | Nicolás Campos (HelioUSACH); supervisor and deployment: Víctor A. Pinto |
| Repo, current tag | `HelioUSACH/flare-nrt`, `v2.2.0` (Python package `centella`) |
| Run cadence | One result per AIA synoptic slot (2 min), as soon as its 4 channels and at least 2 of its 3 baseline frames are on disk |
| Inputs | **AIA synoptic 131/171/304/1600 Å**, 1024², level 1.5 quicklook. Source: `https://jsoc1.stanford.edu/data/aia/synoptic/mostrecent/AIAsynoptic{0131,0171,0304,1600}.fits`, polled every 30 s. Gaps are filled from the dated archive (`…/synoptic/YYYY/MM/DD/HHH00/`), which lags real time; it is queried at most every 10 min. Typical latency from T_OBS to availability: ~7–10 min (measured on 2026-10-28; `data_latency_s` in every result). Public (NASA/SDO). ≈ 2.75 GB/day. **GOES flare list** (labels for curation and validation only, daily): `https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json`, public domain, < 50 kB/day |
| New data the portal must collect | none (the portal already stores GOES XRS, which it compares itself) |
| Latest-result endpoint | `GET http://127.0.0.1:5050/api/status`; fields below |
| Freshness | Computed per request. The result is `stale` if the loop heartbeat or the last result is older than `CENTELLA_STALE_AFTER_SEC` (600 s, the manifest's `stale_after: 10m`), or if `input_age_s` (now − newest T_OBS of the last result) exceeds `CENTELLA_MAX_INPUT_AGE_SEC` (900 s). `stale_reason` says which. In the loop, a frame whose newest T_OBS is already older than 900 s is not run through the model (`last_frame.status = "inputs_old"`) |
| Resources | Measured 2026-10-28 with the real weights on 2 vCPU (Intel Xeon 2.8 GHz), 2 torch threads. **Inference 10.4 s per frame**; **Grad-CAM 10.5 s**, newest frame and alerts only. Peak RSS **1.14 GB**. At one frame per 2 min that is ≈ 18 % of 2 cores on average; a backlog after a restart is skipped down to the last 15 min. No GPU. **Disk at steady state ≈ 25–26 GB**: FITS 7 d × 2.75 GB = 19.3 GB, tensors 2 d × 2.3 GB = 4.6 GB, Grad-CAM ≈ 0.4 GB, archive ≲ 1.3 GB extra (its files are hard links into the FITS kept anyway), results ≈ 0.4 GB (400 d), preview cache < 0.1 GB. Floor: `CENTELLA_MIN_FREE_GB` (10 GB) deletes tensors, then archive, then FITS older than 2 days, oldest first |
| Weights | Release asset `deployment_ensemble_thr058.pt`, 224,246,080 bytes, SHA-256 `d3b850e3ec3d2670f4cb9327c397f7391ed1b082d2b6d154bca1b57626676006`. Also `global_norm_stats.json`, 808 bytes, SHA-256 `807723e30a6a1c0e066c003bb6ad8c3a926a14e2ad2afd2f64f0aff46b83c075`. Both are fetched and verified by `python -m centella.fetch_weights` |
| Baseline it must beat | (1) **GOES XRS event start** (SWPC): the reference that defines the labels; Centella does not beat it on timing (median alert 9 min after it, 1 min before the peak). (2) **`aia131_rise`**: a non-neural rule on the same frames at the same false-alarm count (`centella.validate`). Numbers are in the model card |
| Known limitations | docs/model-card.md "Known limitations" |
| Model card | `docs/model-card.md` |

## `GET /api/status`

The portal's adapter reads these fields (D-096); their names and meaning
are the same as in 1.x: `live_status`, `timestamp`,
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
| `raw_score` | number | 1 | the raw ensemble mean, labelled separately. In 2.x it equals `probability`: there is no gating or smoothing between them. A high probability always opens or continues an event, so `probability ≥ threshold` with `flare_detected: false` cannot happen |
| `alert` | bool | | `probability ≥ threshold` for this slot |
| `threshold` | number | 1 | 0.80 |
| `event_id`, `event_t_start` | integer / ISO `Z`, or null | | the open event |
| `location` | object or null | | **experimental**: `centroid_arcsec` [x, y] helioprojective, `area_px`, `area_arcsec2`, `on_limb`, `method`, `confidence: "experimental"`. Only while an event is open |
| `ensemble_scores` | list of number | 1 | the 5 members' probabilities |
| `ensemble_std` | number | 1 | their standard deviation |
| `lag_min`, `baseline_slots`, `baseline_missing`, `n_baseline_used` | | min / ISO lists / int | the baseline frames actually used (t−8, −10, −12 min; at least 2 of 3) |
| `last_frame` | object | | the newest slot the loop looked at, whether or not it produced a result: `slot`, `made_at`, `status` (`ok` / `missing` / `qc_reject` / `no_baseline` / `inputs_old`), `reason`, `qc` (QC reasons), `channels_missing`, `baseline_missing`, `input_age_s`, `inputs` |
| `data_latency_s` | integer | s | processing time − slot time |
| `latency_ms` | integer | ms | inference time |
| `model`, `model_version`, `weights_sha256` | string | | `"Centella"`, `"2.2.0"`, the weights' hash |
| `frame_count`, `loop_errors`, `loop_last_error`, `loop_last_error_ts` | | | loop health since start |
| `server_time` | ISO `Z` | | |

Not sent: `severity`. It was exploratory in 1.x and was removed in 2.0.

**Compatibility:** 2.1 only adds fields. A portal reading 2.0 or 1.x
fields keeps working. One suggested change, not required: use
`observation_time` as the event's `valid_from` instead of `timestamp`, so
the chart shows observation time rather than processing time.

## Display endpoints (proposed)

These serve what the portal will display. They are built and tested, but
until the maintainers confirm them they are a **proposal**: names and
shapes can change before the portal depends on them. Everything is
internal on 127.0.0.1; the portal proxies or copies what it shows.

| endpoint | returns |
|---|---|
| `GET /api/events?days=30` (1–400) | `{"events": [{event_id, t_start, t_pmax, t_end, p_max, n_alert_frames, location, closed, gradcam}]}`: alert history; `t_start` = first alert, `t_pmax` = highest probability, `t_end` = last alert; `gradcam` = URL of the peak frame's Grad-CAM, or null; the open event last, `closed: false` |
| `GET /api/series?days=3` (≤ 14) | `{"threshold", "lag_min", "fields": ["slot","probability","state","status"], "points": [[…], …]}`: probability per slot for the time series, oldest first; `probability` null when a slot had no result (`status` says why) |
| `GET /api/frames?hours=6` (≤ 168) | `{"frames": [{slot, status, state, probability, alert, inputs, images: [waves on disk], gradcam: bool}]}`: recent frames and what can be shown for each |
| `GET /api/image/{131,171,304,1600}.png?kind=img\|diff&slot=ISO` | 512 px PNG, north up: `img` = DN/s, asinh, SDO-style colours; `diff` = current − mean of the baselines used, grey ±. Without `slot`: the latest result. Available while the FITS are kept (7 days). Header `X-Observation-Time` |
| `GET /api/gradcam/latest.png`, `GET /api/gradcam/{ISO}.png` | 512 px PNG, north up: Grad-CAM of the ensemble (layer4, mean over members) over the 131 Å input. Colour intensity scales with probability / threshold, so a quiet frame is a faint tint. Latest frame: kept 2 days; alerting frames: 400 days |
| `GET /api/channels` | the 4 channels of the default image: slot, T_OBS, whether the difference exists |
| `GET /api/results?hours=24` (≤ 168) | as in 2.0: `points` = [{t, p, status, state, event_id}] |
| `GET /api/health` | 24 h coverage, status counts, median data latency, last download, free disk |

Since 2.2, `/api/series`, `/api/events` and `/api/frames` also take
`start`/`end` (ISO 8601) to select any past window: ≤ 40 days for series,
≤ 400 for events, ≤ 7 for frames. A window that is too long or reversed
gets a 422.

**Retrospective viewer (internal, 2.2):** `GET /viewer` is an HTML page for
browsing any period frame by frame (docs/viewer.md). It is built on the
endpoints above plus:

| endpoint | returns |
|---|---|
| `GET /api/range` | `{first, last, days}`: the span with results |
| `GET /api/frame/{ISO}` | `{slot, row, images, differences, gradcam}`: the slot's full result row (or null), the channels whose image and difference can be shown, whether a Grad-CAM exists |
| `GET /api/flares?start=&end=` | GOES flares ≥ C in the window, from `CENTELLA_FLARES_CSV` or the curated archive |

The portal's own page can reuse the same pieces (time series with a
cursor, image per slot) for a public "look back" view.

## Stored per frame

- **Results:** `results/YYYYMMDD.jsonl` holds one row per slot, with every
  field above plus `aia131_excess`, the input of the baseline rule.
- **Model inputs:** `tensors/YYYYMMDD/HHMM.npz` holds the exact
  (8, 512, 512) float16 input, kept 2 days.
- **Raw files:** the raw FITS stay in `staging/` for 7 days.

With these, any result can be rebuilt (checklist E.23).

## Units

Example systemd units are in `deploy/`, and `deploy/README.md` explains
the layout and the install:

| unit | what | limits |
|---|---|---|
| `centella-loop.service` | the loop | `MemoryMax=3G`, `CPUQuota=200%` |
| `centella-api.service` | uvicorn on 127.0.0.1:5050; writes only `cache/` | `MemoryMax=512M`, `CPUQuota=100%` |
| `centella-curate.{service,timer}` | 04:20 UT daily | `MemoryMax=1G`, `CPUQuota=100%` |
| `centella-validate.{service,timer}` | Mondays 05:00 UT | `MemoryMax=1G`, `CPUQuota=100%` |

All units run as the unprivileged `centella` user, with data under
`/srv/centella`, and log to journald.

## Open questions for the maintainers

1. **Disk:** ≈ 25 GB at steady state, above the ~13 GB expected. Raw FITS
   alone are 19 GB for 7 days, measured at 2.75 GB/day. Options:
   `CENTELLA_STAGING_DAYS=4` (≈ 11 GB), or dropping the 2-day tensors
   (−4.6 GB; they can be rebuilt from the FITS).
2. **Off-server copy of the archive** (docs/archive-format.md): this is
   the retraining dataset. Locally it is kept 7 days. Where should the
   long-term copy go?
3. **The display endpoints above:** confirm them or change them before the
   portal side is written.
