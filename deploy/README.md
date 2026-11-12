# Deployment: an internal service polled by the portal

flare-nrt runs on the portal's server, next to it. It is not a public
website: the portal reads it on `127.0.0.1:5050` and publishes the product
itself. Nothing here is exposed through nginx and it has no domain of its
own. The portal maintainers install and run it, from a tagged release.

## Layout on the server (the units assume it; all paths are settings)

| path | what | owner |
|---|---|---|
| `/opt/flare-nrt` | a checkout of the tag, with `.venv/` | root (read-only for the service) |
| `/srv/flare-nrt` | `FLARE_NRT_DATA_DIR`: staging, results, state, tensors, gradcam, archive, cache; since 1.1.0 also `render/` (pre-rendered WebP, 4 days) and `events/` (event replays, 365 days; with 131 Å frames and the peak set since 1.2.0) | `flare-nrt` |
| `/srv/flare-nrt/models`, `/srv/flare-nrt/metadata` | the release files (`python -m flare_nrt.fetch_weights --dest /srv/flare-nrt`) | `flare-nrt` |
| `/etc/flare-nrt/flare-nrt.env` | settings (copy of `.env.example`; no secrets) | root |

`flare-nrt` is a dedicated system user with no login and no sudo:
`useradd --system --home /srv/flare-nrt --shell /usr/sbin/nologin flare-nrt`.

## Units

| unit | what | limits |
|---|---|---|
| `flare-nrt-loop.service` | the detection loop (`python -m flare_nrt.loop`), `Restart=always` | `MemoryMax=3G`, `CPUQuota=200%`, `Nice=5` |
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
git clone --branch v1.2.0 https://github.com/HelioUSACH/flare-nrt.git /opt/flare-nrt
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
of event replays (9.3 MB per event since 1.2.0; estimate; measure with
`du -sh /srv/flare-nrt/events`).
The breakdown is in INTEGRATION.md, "Resources".
