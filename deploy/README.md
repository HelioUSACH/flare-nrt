# Deployment: an internal service polled by the portal

flare-nrt runs on the portal's server, next to it. It is not a public
website: the portal reads it on `127.0.0.1:5050` and publishes the product
itself. Nothing here is exposed through nginx and it has no domain of its
own. The portal maintainers install and run it, from a tagged release.

## Layout on the server (the units assume it; all paths are settings)

| path | what | owner |
|---|---|---|
| `/opt/flare-nrt` | a checkout of the tag, with `.venv/` | root (read-only for the service) |
| `/srv/flare-nrt` | `FLARE_NRT_DATA_DIR`: staging, results, state, tensors, gradcam, archive, cache | `flare-nrt` |
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

The settings in `/etc/flare-nrt/flare-nrt.env` for this layout:

```
FLARE_NRT_DATA_DIR=/srv/flare-nrt
FLARE_NRT_WEIGHTS=/srv/flare-nrt/models/deployment_ensemble_thr058.pt
FLARE_NRT_NORM_STATS=/srv/flare-nrt/metadata/global_norm_stats.json
```

## Install (maintainers)

As root, in this order:

```bash
git clone --branch v1.0.0 https://github.com/HelioUSACH/flare-nrt.git /opt/flare-nrt
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

## Disk

About 17–20 GB at steady state with the defaults (4 days of raw FITS). The breakdown is in
INTEGRATION.md, "Resources".
