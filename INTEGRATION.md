# Integration sheet: Centella

| | |
|---|---|
| Model / product id | Centella → product `usach.flare.nrt` (the product id is unchanged; Centella is the model's name) |
| Kind | event (detection) |
| Owner(s) | Nicolás Campos (HelioUSACH) |
| Repo, current tag | `HelioUSACH/centella`, `v2.0.0` |
| Run cadence | one result per AIA synoptic slot (2 min), as soon as its 4 channels and baselines are available |
| Inputs | AIA synoptic 131/171/304/1600 Å, `https://jsoc1.stanford.edu/data/aia/synoptic/YYYY/MM/DD/HHH00/AIAYYYYMMDD_HHMM_WWWW.fits`, 2 min, latency: measured live (`data_latency_s`), public (NASA/SDO), ≈ 2.6 GB/day. GOES flare list (labels, daily): `https://services.swpc.noaa.gov/json/goes/primary/xray-flares-7-day.json`, public domain, < 50 kB/day |
| New data the portal must collect | none |
| Latest-result endpoint | `GET http://127.0.0.1:5050/api/status` (below) |
| Freshness | computed per request: `stale` if the loop heartbeat or the last result is older than `CENTELLA_STALE_AFTER_SEC` (600 s, the manifest's `stale_after: 10m`), or the observation behind the last result is older than `CENTELLA_MAX_INPUT_AGE_SEC` (3600 s); `stale_reason` says which |
| Resources | Measured on a 2-vCPU test machine with a same-size stand-in ensemble: **0.7 s per slot** (5 × ResNet-18, 8×512×512, 4 threads), **≈ 1 GB** peak RSS; one slot every 2 min → < 1 % of one core on average. Re-measure on the VM before go-live (`latency_ms` in `/api/status`; `systemctl status centella-loop` for memory). GPU not needed. Disk steady state ≲ 25 GB (docs/archive-format.md) |
| Weights | release asset `deployment_ensemble_thr058.pt`, SHA-256 in the release notes and in `CENTELLA_WEIGHTS_SHA256` (the loop refuses a mismatch) |
| Baseline it must beat | GOES XRS event start (SWPC): **not beaten on timing** (median alert 9 min after it, 1 min before the peak); see the model card |
| Known limitations | docs/model-card.md |
| Model card | `docs/model-card.md` |

## `/api/status`

Fields read by the portal (`clima-espacial/modules/flare/adapters.py`)
keep their names and meaning; the rest are additions.

| field | type | units | meaning |
|---|---|---|---|
| `live_status` | string | | `live` / `stale` / `no_data`, computed on every request |
| `stale_reason` | string or null | | `loop_not_running`, `no_new_result` or `inputs_old` |
| `timestamp` | ISO 8601 `Z` | | when the latest result was made (the portal's `issued`) |
| `last_update` | ISO 8601 `Z` | | same as `timestamp` |
| `seconds_since_update` | number | s | now − `last_update` |
| `stale_after_sec` | integer | s | the freshness threshold |
| `observation_time` | ISO 8601 `Z` | | the AIA slot the result is about (new in 2.0) |
| `probability` | number | 1 | ensemble mean, [0, 1] |
| `alert` | bool | | `probability ≥ threshold` for this slot (new) |
| `flare_detected` | bool | | an event is open: an alert within the last `merge_gap` (10 min) |
| `event_id` | integer or null | | the open event |
| `threshold` | number | 1 | 0.80 |
| `location` | object or null | | `centroid_arcsec` [x, y] helioprojective, `area_px`, `area_arcsec2`, `on_limb`, `method`, `confidence: "experimental"` — only when alerting |
| `ensemble_scores` | list of number | 1 | per-member probabilities |
| `ensemble_std` | number | 1 | their standard deviation |
| `lag_min`, `baseline_slots` | | min / ISO list | the baseline used |
| `inputs` | object | | T_OBS of each channel (provider timestamps, checklist E.23) |
| `data_latency_s` | integer | s | processing time − observation time |
| `latency_ms` | integer | ms | inference time |
| `model`, `model_version`, `weights_sha256` | string | | what produced the result (`model` = `"Centella"`) |
| `frame_count`, `loop_errors`, `loop_last_error`, `loop_last_error_ts` | | | loop health since start |
| `server_time` | ISO 8601 `Z` | | |

Removed in 2.0: `severity` (the portal already dropped it, D-036),
`frames_in_buffer`, `buffer_len`.

**Change the portal side may want (not required):** use
`observation_time` instead of `timestamp` as the event's `valid_from`, so
the chart is in observation time rather than processing time.

Other endpoints (internal): `/api/events?days=30`, `/api/results?hours=24`
(probability series), `/api/health` (24 h coverage, status counts, median
data latency, free disk), `/api/channels` (the 4 channels of the latest
result: provider time, image/difference available) and
`/api/image/{131|171|304|1600}.png?kind=img|diff` (512 px PNG, north up,
rendered on request from the staging FITS, cached a day; `X-Observation-Time`
header). The portal proxies the images; Centella stays on 127.0.0.1.

## Units

`deploy/centella-loop.service` (the loop, `Restart=always`),
`deploy/centella-api.service` (uvicorn on 127.0.0.1:5050),
`deploy/centella-curate.{service,timer}` (04:20 UT daily),
`deploy/centella-validate.{service,timer}` (Mondays 05:00 UT). All run as an
unprivileged `flare` user, data under `/srv/centella`.
