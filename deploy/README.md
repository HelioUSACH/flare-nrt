# Deployment: an internal service polled by the portal

Centella runs on the portal's server, next to it. It is not a public
website: the portal reads it on `127.0.0.1:5050` and publishes the product
itself. Nothing here is exposed through nginx and it has no domain of its
own. The portal maintainers install and run it, from a tagged release.

## Layout on the server (the units assume it; all paths are settings)

| path | what | owner |
|---|---|---|
| `/opt/centella` | a checkout of the tag, with `.venv/` | root (read-only for the service) |
| `/srv/centella` | `CENTELLA_DATA_DIR`: staging, results, state, tensors, gradcam, archive, cache | `centella` |
| `/srv/centella/models`, `/srv/centella/metadata` | the release files (`python -m centella.fetch_weights --dest /srv/centella`) | `centella` |
| `/etc/centella/centella.env` | settings (copy of `.env.example`; no secrets) | root |

`centella` is a dedicated system user with no login and no sudo:
`useradd --system --home /srv/centella --shell /usr/sbin/nologin centella`.

## Units

| unit | what | limits |
|---|---|---|
| `centella-loop.service` | the detection loop (`python -m centella.loop`), `Restart=always` | `MemoryMax=3G`, `CPUQuota=200%`, `Nice=5` |
| `centella-api.service` | uvicorn on `127.0.0.1:5050`, 1 worker; reads the data directory and writes only `cache/` | `MemoryMax=512M`, `CPUQuota=100%` |
| `centella-curate.{service,timer}` | 04:20 UT daily: archive for retraining, retention | `MemoryMax=1G`, `CPUQuota=100%` |
| `centella-validate.{service,timer}` | Mondays 05:00 UT: last 7 days against GOES and the baseline rule → `reports/` | `MemoryMax=1G`, `CPUQuota=100%` |

All units have `ProtectSystem=strict`, `NoNewPrivileges=true` and
`PrivateTmp=true`. They write only under `/srv/centella`, and log to
journald (`journalctl -u centella-loop`).

The settings in `/etc/centella/centella.env` for this layout:

```
CENTELLA_DATA_DIR=/srv/centella
CENTELLA_WEIGHTS=/srv/centella/models/deployment_ensemble_thr058.pt
CENTELLA_NORM_STATS=/srv/centella/metadata/global_norm_stats.json
```

## Install (maintainers)

```bash
git clone --branch vX.Y.Z https://github.com/HelioUSACH/flare-nrt.git /opt/centella
cd /opt/centella && python3 -m venv .venv
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1
.venv/bin/pip install -r requirements.lock.txt && .venv/bin/pip install --no-deps .
install -d -o centella -g centella /srv/centella
sudo -u centella .venv/bin/python -m centella.fetch_weights --dest /srv/centella
install -D -m 644 .env.example /etc/centella/centella.env   # then set the three paths above
cp deploy/*.service deploy/*.timer /etc/systemd/system/ && systemctl daemon-reload
systemctl enable --now centella-loop centella-api centella-curate.timer centella-validate.timer
curl -s 127.0.0.1:5050/api/status | python3 -m json.tool | head
```

The first result arrives about 15 minutes after the start, once the
baseline frames (t−8…−12 min) have been downloaded. Until then
`live_status` is `no_data`.

To upgrade: check out the new tag in `/opt/centella`, reinstall if
`requirements.lock.txt` changed, and run `fetch_weights` if the release
changed the weights. Then `systemctl restart centella-loop centella-api`.
State lives in `/srv/centella/state`, so a restart neither loses nor
duplicates results.

## Disk

About 21 GB at steady state with the defaults. The breakdown is in
INTEGRATION.md, "Resources".
