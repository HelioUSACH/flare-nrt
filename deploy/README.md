# Deployment: an internal service polled by the portal

flare-nrt runs on the portal's server, next to it. It is not a public
website: the portal reads it on `127.0.0.1:5050` and publishes the product
itself. Nothing here is exposed through nginx and it has no domain of its
own. The portal maintainers install and run it, from a tagged release.

## Layout on the server (the units assume it; all paths are settings)

| path | what | owner |
|---|---|---|
| `/opt/flare-nrt` | a checkout of the tag, with `.venv/` | root (read-only for the service) |
| `/srv/flare-nrt` | `FLARE_NRT_DATA_DIR`: staging, results, state, tensors, gradcam, archive, cache; since 1.1.0 also `render/` (pre-rendered WebP, 4 days) and `events/` (event replays, 365 days; with 131 Å frames and the peak set since 1.2.0, a Grad-CAM map on every frame with a result and the location since 1.3.0) | `flare-nrt` |
| `/srv/flare-nrt/models`, `/srv/flare-nrt/metadata` | the release files (`python -m flare_nrt.fetch_weights --dest /srv/flare-nrt`) | `flare-nrt` |
| `/etc/flare-nrt/flare-nrt.env` | settings (copy of `.env.example`; no secrets) | root |

`flare-nrt` is a dedicated system user with no login and no sudo:
`useradd --system --home /srv/flare-nrt --shell /usr/sbin/nologin flare-nrt`.

## Units

| unit | what | limits |
|---|---|---|
| `flare-nrt-loop.service` | the detection loop (`python -m flare_nrt.loop`), `Restart=always`; since 1.3.1 `Type=notify` with `WatchdogSec=600` (restarted if it shows no progress for 10 min) | `MemoryMax=3G`, `CPUQuota=200%`, `Nice=5` |
| `flare-nrt-api.service` | uvicorn on `127.0.0.1:5050`, 1 worker; reads the data directory and writes only `cache/` | `MemoryMax=512M`, `CPUQuota=100%` |
| `flare-nrt-curate.{service,timer}` | 04:20 UT daily: archive for retraining, retention | `MemoryMax=1G`, `CPUQuota=100%` |
| `flare-nrt-validate.{service,timer}` | Mondays 05:00 UT: last 7 days against GOES and the baseline rule → `reports/` | `MemoryMax=1G`, `CPUQuota=100%` |

All units have `ProtectSystem=strict`, `NoNewPrivileges=true` and
`PrivateTmp=true`. They write only under `/srv/flare-nrt`, and log to
journald (`journalctl -u flare-nrt-loop`).

`PrivateTmp=true` is required, not only hardening: `ProtectSystem=strict`
makes the whole file system read-only, `/tmp` included, and torch (which
the loop, the curation and the validation import) needs a writable
temporary directory. `PrivateTmp` gives each unit its own writable `/tmp`.
A one-off `systemd-run` with `-p ProtectSystem=strict` needs
`-p PrivateTmp=yes` for the same reason. `flare-nrt-validate.service`
lacked it until 1.2.0.

The settings in `/etc/flare-nrt/flare-nrt.env` for this layout:

```
FLARE_NRT_DATA_DIR=/srv/flare-nrt
FLARE_NRT_WEIGHTS=/srv/flare-nrt/models/deployment_ensemble_thr058.pt
FLARE_NRT_NORM_STATS=/srv/flare-nrt/metadata/global_norm_stats.json
```

## Install (maintainers)

As root, in this order:

```bash
git clone --branch v1.3.1 https://github.com/HelioUSACH/flare-nrt.git /opt/flare-nrt
cd /opt/flare-nrt && python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1
.venv/bin/pip install -r requirements.lock.txt && .venv/bin/pip install --no-deps .
id flare-nrt >/dev/null 2>&1 || useradd --system --home /srv/flare-nrt --shell /usr/sbin/nologin flare-nrt
install -d -o flare-nrt -g flare-nrt /srv/flare-nrt
sudo -u flare-nrt .venv/bin/python -m flare_nrt.fetch_weights --dest /srv/flare-nrt   # SHA-256 checked
install -D -m 644 .env.example /etc/flare-nrt/flare-nrt.env
sed -i -e 's#^FLARE_NRT_DATA_DIR=.*#FLARE_NRT_DATA_DIR=/srv/flare-nrt#' \
       -e 's#^FLARE_NRT_WEIGHTS=.*#FLARE_NRT_WEIGHTS=/srv/flare-nrt/models/deployment_ensemble_thr058.pt#' \
       -e 's#^FLARE_NRT_NORM_STATS=.*#FLARE_NRT_NORM_STATS=/srv/flare-nrt/metadata/global_norm_stats.json#' \
       /etc/flare-nrt/flare-nrt.env
cp deploy/*.service deploy/*.timer /etc/systemd/system/ && systemctl daemon-reload
systemctl enable --now flare-nrt-loop flare-nrt-api flare-nrt-curate.timer flare-nrt-validate.timer
curl -s 127.0.0.1:5050/api/status | python3 -m json.tool | head
```

The first result arrives about 15 minutes after the start, once the
baseline frames (t−8…−12 min) have been downloaded. Until then
`live_status` is `no_data`.

To upgrade: check out the new tag in `/opt/flare-nrt`, reinstall if
`requirements.lock.txt` changed, and run `fetch_weights` if the release
changed the weights. Then `systemctl restart flare-nrt-loop flare-nrt-api`.
State lives in `/srv/flare-nrt/state`, so a restart neither loses nor
duplicates results.

## Upgrade from 1.0.0 to 1.1.0

1.1.0 changes only how results are rendered and kept for display; the
weights, the threshold and every probability are those of 1.0.0, and the
release carries the same two files with the same SHA-256 (so
`fetch_weights` has nothing to do). `requirements.lock.txt` and the units
are unchanged. As root:

```bash
cd /opt/flare-nrt
git fetch --tags && git checkout v1.1.0
.venv/bin/pip install -r requirements.lock.txt && .venv/bin/pip install --no-deps .
systemctl restart flare-nrt-loop flare-nrt-api
curl -s 127.0.0.1:5050/api/status | python3 -c 'import json,sys; print(json.load(sys.stdin)["model_version"])'   # 1.1.0
# once: replays of the events already in the archive whose frames are still on disk (last 4 days)
systemd-run --wait --pty --uid=flare-nrt --gid=flare-nrt -p WorkingDirectory=/opt/flare-nrt \
  -p EnvironmentFile=/etc/flare-nrt/flare-nrt.env -p MemoryMax=2G -p CPUQuota=100% -p Nice=10 \
  -p ProtectSystem=strict -p PrivateTmp=yes -p ReadWritePaths=/srv/flare-nrt \
  /opt/flare-nrt/.venv/bin/python -m flare_nrt.replays
```

(As first published with 1.1.0 this command lacked `-p PrivateTmp=yes`
and failed: `ProtectSystem=strict` leaves no writable `/tmp` for torch.
If you upgrade from 1.0.0 straight to 1.2.0, skip it here: the 1.2.0
one-off below does the same and more.)

The last command loads the model (peak RSS ≈ 0.9 GB) to recompute the
Grad-CAM maps that 1.0.0 did not keep: ≈ 10 s per frame at 2 threads,
twice that under `CPUQuota=100%`, so ≈ 10–20 min per event of ~55 frames,
next to the running loop. `--no-gradcam` skips that (the replays then lack
the heat frames of pre-upgrade slots and say so); `--event ID` limits it to
one event. It can be run again at any time: it builds what is missing and
rebuilds only replays marked incomplete, reusing the frames they already
have. Events older than the FITS (4 days) get a replay that says their
frames are gone. From then on the loop builds every replay itself, 60 min
after the event closes.

Nothing else is needed: the new directories are created at start, and the
existing retention (loop hourly, `flare-nrt-curate.timer` daily) prunes
them.

## Upgrade from 1.1.0 to 1.2.0

1.2.0 changes only what an event replay keeps (detection, weights,
threshold and every probability are those of 1.0.0 and 1.1.0; the release
carries the same two files with the same SHA-256, so `fetch_weights` has
nothing to do). Each replay now also stores `131.webp` for every frame and,
once per event, the peak frame's 4 channels and its Grad-CAM in both
styles (`peak/`). `requirements.lock.txt` is unchanged. One unit changed:
`flare-nrt-validate.service` gains `PrivateTmp=true` (above). As root:

```bash
cd /opt/flare-nrt
git fetch --tags && git checkout v1.2.0
.venv/bin/pip install -r requirements.lock.txt && .venv/bin/pip install --no-deps .
cp deploy/*.service deploy/*.timer /etc/systemd/system/ && systemctl daemon-reload
systemctl restart flare-nrt-loop flare-nrt-api
curl -s 127.0.0.1:5050/api/status | python3 -c 'import json,sys; print(json.load(sys.stdin)["model_version"])'   # 1.2.0
# once: add 131 Å and the peak set to the replays already built (and build any missing or incomplete ones)
systemd-run --wait --pty --uid=flare-nrt --gid=flare-nrt -p WorkingDirectory=/opt/flare-nrt \
  -p EnvironmentFile=/etc/flare-nrt/flare-nrt.env -p MemoryMax=2G -p CPUQuota=100% -p Nice=10 \
  -p ProtectSystem=strict -p PrivateTmp=yes -p ReadWritePaths=/srv/flare-nrt \
  /opt/flare-nrt/.venv/bin/python -m flare_nrt.replays
```

`-p PrivateTmp=yes` is needed: without it the command fails at the start,
because `ProtectSystem=strict` makes `/tmp` read-only and torch needs a
writable temporary directory.

The one-off rebuilds every replay built by 1.1.0 (it has no `peak` key in
its `replay.json`) and every replay marked incomplete. It reuses the files
a replay already has, so 171 Å and heat frames are not rendered or
recomputed again. Where the FITS are still on disk (the last 4 days), it
renders 131 Å for each frame and the peak's four channels. Where they are
gone, the replay keeps what it had, and `replay.json` lists what is missing
(`missing["131"]`, `missing["peak"]`, `complete: false`, `note`). The peak
frame's Grad-CAM survives the FITS (alerting frames' maps are kept 400
days), so even an old event gets both Grad-CAM styles in `peak/`. It loads
the model only to recompute Grad-CAM maps that are gone; a replay built by
1.1.0 already has them, so it takes ≈ 5 s per event whose FITS are still
there (55 renders of 131 Å at 0.08 s, plus the peak set) and less where
they are gone; loading the model takes ≈ 1.3 GB peak RSS.
`--no-gradcam` skips loading the model; `--event ID` limits it to one
event. It can be run again at any time: complete 1.2.0 replays are left
alone. From then on the loop builds every replay with the new files
itself, 60 min after the event closes.

Nothing else is needed: the new files live inside each `events/<id>/`
and are pruned with it (365 days, unchanged).

## Upgrade from 1.2.0 to 1.3.0

1.3.0 changes what an event replay keeps and how the experimental
location is found. Detection is unchanged: weights, normalisation,
threshold, baselines, state machine and every probability are those of
1.0.0, and the release carries the same two files with the same SHA-256,
so `fetch_weights` has nothing to do. `requirements.lock.txt` and the
units are unchanged. As root:

```bash
cd /opt/flare-nrt
git fetch --tags && git checkout v1.3.0
.venv/bin/pip install -r requirements.lock.txt && .venv/bin/pip install --no-deps .
systemctl restart flare-nrt-loop flare-nrt-api
curl -s 127.0.0.1:5050/api/status | python3 -c 'import json,sys; print(json.load(sys.stdin)["model_version"])'   # 1.3.0
# once: Grad-CAM maps on every frame and the location, for the replays already built
# (event 1's FITS were kept in /srv/flare-nrt/keep/20261111 for this)
systemd-run --wait --pty --uid=flare-nrt --gid=flare-nrt -p WorkingDirectory=/opt/flare-nrt \
  -p EnvironmentFile=/etc/flare-nrt/flare-nrt.env -p MemoryMax=2G -p CPUQuota=100% -p Nice=10 \
  -p ProtectSystem=strict -p PrivateTmp=yes -p ReadWritePaths=/srv/flare-nrt \
  /opt/flare-nrt/.venv/bin/python -m flare_nrt.replays --fits-root /srv/flare-nrt/keep
```

`-p PrivateTmp=yes` is needed, as for 1.2.0: `ProtectSystem=strict` makes
`/tmp` read-only and torch needs a writable temporary directory.
`/srv/flare-nrt/keep` is inside `ReadWritePaths`, so the command reads it;
it only reads from it.

The one-off rebuilds every replay built before 1.3.0 (its `replay.json`
has no `geometry`) and every replay marked incomplete, reusing the files it
already has. For each frame with a result that has no heat frame it
computes the Grad-CAM map: from the stored model input (2 days), else from
the FITS (staging, then each `--fits-root`, laid out like staging:
`<root>/YYYYMMDD/AIA….fits`). Each frame gets `alert` and `cam_peak`
(recomputed if its map is gone). Each alert frame gets a location: the
result's own, else computed from the 131 Å FITS. An event without a location
gets its peak frame's, in `replay.json` and in `events.csv`
(`/api/events`); a location already there is never changed. Where the FITS
are gone, `replay.json` says what is missing (`missing["gradcam-heat"]`,
`missing["location"]`, `note`), and the command prints it per event with
the time, the maps recomputed and the peak RSS. Cost, measured on the M4.0 of 2026-07-07 in event 1's situation (45
maps to recompute from the FITS): 44 s and 1.0 GB peak RSS on a desktop
Ryzen with 2 threads, so ≈ 6–12 min on this server under `CPUQuota=100%`,
next to the running loop; replays that already have their maps take
seconds.
`--event ID` limits it to one event; `--no-gradcam` skips loading the
model. It can be run again at any time.

Check event 1 (the command prints `event 1: … location yes, …`):

```bash
curl -s 127.0.0.1:5050/api/events/1/replay | python3 -c 'import json,sys; d=json.load(sys.stdin); \
  print(d["event"]["location"]["method"], d["missing"]["gradcam-heat"], d["missing"]["location"])'
```

`diff131_rate_brightest_region_robust [] []` confirms why it had no
location live (CHANGELOG 1.3.0, item 2); `diff131_rate_brightest_region`
would mean the 131 Å rates were missing in the loop instead. Then
`/srv/flare-nrt/keep` can be deleted.

From then on the loop builds every new replay with the new fields. It
holds the model already, so it computes the maps it lacks (only the slots
of a backlog after a restart: the loop computes a Grad-CAM for its newest
frame every slot) at ≈ 10 s each, once per event.

## Upgrade from 1.3.0 to 1.3.1

1.3.1 ends the outage that began at 23:00 UT on 2026-11-24 (CHANGELOG
1.3.1). Since then every near-real-time AIA frame carries QUALITY bit 2
(`Q_NOASD`), which 1.3.0 rejected, so there has been no result since 22:58 UT.
1.3.1 processes these frames and keeps the raw QUALITY in each row. It
also logs when there has been no result for 30 min, fills gaps from JSOC's
`nrt/` archive, and feeds a systemd watchdog. The network is unchanged:
same weights, normalisation, threshold, baselines and state machine. The
release carries the same two files with the same SHA-256, so
`fetch_weights` has nothing to do, and `requirements.lock.txt` is
unchanged. **The loop's unit file changes** (`Type=notify`,
`WatchdogSec=600`, `TimeoutStartSec=300`). As root:

```bash
cd /opt/flare-nrt
git fetch --tags && git checkout v1.3.1
.venv/bin/pip install -r requirements.lock.txt && .venv/bin/pip install --no-deps .
cp deploy/flare-nrt-loop.service /etc/systemd/system/ && systemctl daemon-reload
systemctl restart flare-nrt-loop flare-nrt-api
systemctl show flare-nrt-loop -p Type -p WatchdogUSec -p ActiveState   # Type=notify, WatchdogUSec=10min, active
curl -s 127.0.0.1:5050/api/status | python3 -c 'import json,sys; print(json.load(sys.stdin)["model_version"])'   # 1.3.1
```

At start the loop rebuilds the count of slots without a result from
`results/` and says so once (`journalctl -u flare-nrt-loop`):
`starting with 1 d … without a result (since 2026-11-24T23:00:00Z, last
result 2026-11-24T22:58:00Z): quality:131=1073741828 ×…`. It carries on
from the slot after the last one it wrote, reading the baselines from
staging, so results come back with the first slot it processes:
`results resumed at … after …`. From then on it logs a WARNING after 30
min without a result and every hour while that lasts.

`Type=notify`: if `systemctl restart` waits and then fails with a timeout,
the loop did not say `READY=1` within `TimeoutStartSec` (300 s; it says so
once the model is loaded, normally in seconds). The journal shows why.

### Once: reprocess the gap (before 2026-11-28 ~23:00 UT)

The FITS of the outage are still in staging, and the first ones (23:00 UT on
2026-11-24) are deleted 4 days later, at about 23:00 UT on 2026-11-28.
The free-space floor (`FLARE_NRT_MIN_FREE_GB`, 10 GB) deletes FITS older
than 2 days sooner if the disk runs low, so check `df -h /srv/flare-nrt`
first. Run the one-off as the service user, under the units' sandbox, next
to the running loop. Run it as a named transient unit, not with `--pty`:
it takes hours, and it must survive a dropped SSH session.

```bash
RUN="systemd-run --uid=flare-nrt --gid=flare-nrt -p WorkingDirectory=/opt/flare-nrt \
  -p EnvironmentFile=/etc/flare-nrt/flare-nrt.env -p MemoryMax=2G -p CPUQuota=100% -p Nice=10 \
  -p ProtectSystem=strict -p PrivateTmp=yes -p ReadWritePaths=/srv/flare-nrt"
# what it would do (no model loaded): slots by current result, how many to reprocess
$RUN --wait --pty /opt/flare-nrt/.venv/bin/python -m flare_nrt.reprocess --since 2026-11-24T23:00:00Z --dry-run
# the run, in the background; follow it in the journal
$RUN --unit=flare-nrt-reprocess /opt/flare-nrt/.venv/bin/python -m flare_nrt.reprocess --since 2026-11-24T23:00:00Z
journalctl -fu flare-nrt-reprocess        # "reprocess at …: n/N slots, {…}, ~M min left" every 30 slots
```

`-p PrivateTmp=yes` is needed, as for the replay one-offs:
`ProtectSystem=strict` makes `/tmp` read-only, and torch needs a writable
temporary directory. It never goes past the newest slot the loop has
written (`state/loop_state.json`), and it leaves alone the slots the 1.3.1
loop has already processed (`ok`). The dry
run should report every slot from 23:00 UT on 2026-11-24 up to the
upgrade as `qc_reject`. Any marked `FITS gone` can no longer be
reprocessed.

What it does (CHANGELOG 1.3.1, item 5): each slot whose result is
`qc_reject`, `missing` or `no_baseline` and whose FITS are in staging
goes through the loop's code again, in order: QC with the 1.3.1 mask,
baselines from staging, model, alert rule and event tracker. It appends
a new row (`reprocessed: true`) after the old one. **The last row of a slot
in `results/YYYYMMDD.jsonl` is its result**, so `/api/series`,
`/api/frames` and `/api/frame` show the new rows, and so do curation and
validation. Events found in the gap get ids the loop has not used (shared
counter `state/event_ids.json`), go into `events.csv`, and get their replay
from the loop within 5 min. Grad-CAM is computed for alerting frames only,
so `render/` (the last hours' loop) is not filled for the gap. The loop's
state files are not touched.

Cost: one inference per slot, ≈ 10 s on this server, more under
`CPUQuota=100%` next to the loop, so ≈ 4–8 h per day of gap (≈ 1.5 days
by 2026-11-26). Give it `-p CPUQuota=150%` if time is short. If it stops
(e.g. a reboot), run the same command again: it only takes the slots that
still have no result. The first 5 slots after 23:00 UT can stay
`no_baseline` if their t−8…−12 min FITS have already left staging.

Then re-curate the days that were curated with the old rows. The curation
timer curates each day at 04:20 UT the next morning, so 2026-11-24 and
probably 2026-11-25 are already done. Their `frames.csv` still says
`qc_reject` and the archive lacks any alert window found in the gap:

```bash
for d in 2026-11-24 2026-11-25; do
  $RUN --wait --pty /opt/flare-nrt/.venv/bin/python -m flare_nrt.curate --day $d --force
done
```

### Checks

```bash
curl -s 127.0.0.1:5050/api/status | python3 -c 'import json,sys; s=json.load(sys.stdin); \
  print(s["live_status"], s["last_ok_slot"], s["rejected_streak"], s["last_frame"]["quality_soft"], s["last_download"])'
# live <a slot within the last ~15 min> 0 ['noasd'] <a recent time>
curl -s "127.0.0.1:5050/api/series?start=2026-11-24T22:00:00Z&end=2026-11-26T00:00:00Z" | python3 -c \
  'import json,sys,collections; print(collections.Counter(p[3] for p in json.load(sys.stdin)["points"]))'
# after the one-off: mostly ok, a few no_baseline / missing, no long qc_reject run
systemctl show flare-nrt-loop -p NRestarts -p WatchdogUSec                # the watchdog is armed
```

`rejected_streak` counts the slots in a row without a result (any status
but `ok`). It is 0 while results flow and falls back to 0 with the first
one. `last_reject_reasons` and `rejected_streak_reasons` say why.

### Settings added in 1.3.1

All optional, in `/etc/flare-nrt/flare-nrt.env`:

| variable | default | what |
|---|---|---|
| `FLARE_NRT_QC_IGNORE_QUALITY_MASK` | `0x40000004` | QUALITY bits that don't reject a frame (hex or decimal): bit 30, near-real-time, and bit 2, `Q_NOASD`. `0x40000000` is the 1.0.0–1.3.0 rule |
| `FLARE_NRT_NO_RESULT_WARN_SLOTS` | `15` | slots in a row without a result (2 min each) before the WARNING |
| `FLARE_NRT_DOWNLOAD_MAX_SEC` | `60` | total time for one download, headers and body (`FLARE_NRT_HTTP_TIMEOUT`, 15 s, still bounds each read) |

## Settings added in 1.1.0

All optional, in `/etc/flare-nrt/flare-nrt.env`:

| variable | default | what |
|---|---|---|
| `FLARE_NRT_REPLAY_DAYS` | `365` | event replays (`events/<id>/`) are kept this long after the event's end |
| `FLARE_NRT_REPLAY_PRE_MIN` | `30` | a replay starts this long before the event's first alert |
| `FLARE_NRT_REPLAY_POST_MIN` | `60` | … ends this long after its last alert, and is built once the event has been closed this long |
| `FLARE_NRT_WEBP_QUALITY` | `90` | WebP quality of `render/`, replays and `format=webp` |

`render/` is kept as long as the raw FITS (`FLARE_NRT_STAGING_DAYS`, 4),
with no setting of its own. `FLARE_NRT_IMAGE_CACHE_DAYS` (1, the API's
preview cache) existed before; lower it if the portal fetches many 1024 px
PNGs (INTEGRATION.md, "Resources").

## Disk

About 17–20 GB at steady state with the defaults (4 days of raw FITS),
plus ≈ 0.5 GB for `render/` and the Grad-CAM maps and ≈ 3.5–6 GB per year
of event replays (9.3 MB per event since 1.2.0, ≤ 1 MB more since 1.3.0; estimate; measure with
`du -sh /srv/flare-nrt/events`).
The breakdown is in INTEGRATION.md, "Resources".
