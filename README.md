# flare-nrt

**flare-nrt** detects solar flares in near real time on SDO/AIA synoptic
images. It produces the `usach.flare.nrt` product of
[Clima Espacial USACH](https://github.com/HelioUSACH/clima-espacial). It
runs as an internal service on the portal's server, and the portal polls
it on `127.0.0.1:5050`.

- Integration sheet and HTTP contract: [`INTEGRATION.md`](INTEGRATION.md)
- Model card: [`docs/model-card.md`](docs/model-card.md)
- Deployment on the portal's server: [`deploy/README.md`](deploy/README.md)
- Archive for retraining: [`docs/archive-format.md`](docs/archive-format.md)
- Retrospective viewer: [`docs/viewer.md`](docs/viewer.md)

The steps below work on a fresh Ubuntu 24.04 machine with no GPU, in this
order.

## 1. Install

```bash
sudo apt install python3-venv          # once, if missing
git clone https://github.com/HelioUSACH/flare-nrt.git && cd flare-nrt
python3 -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1 torchvision==0.29.1
pip install -r requirements.lock.txt   # exact versions (ranges: requirements.txt)
pip install -e . --no-deps
```

## 2. Fetch the weights

```bash
python -m flare_nrt.fetch_weights       # → models/ and metadata/, SHA-256 verified
```

This downloads two files from the GitHub release:
`deployment_ensemble_thr058.pt` (214 MB) and `global_norm_stats.json`.
Their SHA-256 is pinned in `flare_nrt/fetch_weights.py`. The default of
`FLARE_NRT_WEIGHTS_SHA256` is the same value, so the service refuses any
other file. A file is loaded with `torch.load(weights_only=True)` only
after its hash checks.

## 3. Run

```bash
python -m flare_nrt.loop                                    # live loop (JSOC mostrecent/)
uvicorn flare_nrt.server:app --host 127.0.0.1 --port 5050   # the API the portal polls
curl -s 127.0.0.1:5050/api/status                          # live_status, state, probability …
```

Default paths are relative to the working directory: `./var` for data,
`./models` and `./metadata` for the model files. Every path and parameter
is an environment variable (`.env.example`). Logs go to stdout/stderr.

Other jobs:

```bash
python -m flare_nrt.curate                                  # daily: archive + labels + retention
python -m flare_nrt.validate --start 2026-10-21 --end 2026-10-27   # vs GOES and vs the baseline rule
python -m flare_nrt.loop --replay /path/to/fits             # replay stored FITS through the same code
python -m flare_nrt.goes --start 2026-06-19 --end 2026-07-17 --out flares.csv   # GOES list of a past period
```

`http://127.0.0.1:5050/viewer` is a retrospective viewer. It lets you
scrub through any period frame by frame: the probability over time, the 4
AIA channels and their differences, the Grad-CAM, and the GOES flares.
docs/viewer.md explains how to browse a replayed month without copying its
FITS.

## 4. Test

```bash
ruff check .
pytest                                                      # offline, CPU, ~3 min
FLARE_NRT_REAL_WEIGHTS=models/deployment_ensemble_thr058.pt pytest tests/test_snapshot.py
```

The plain `pytest` run uses a tiny stand-in ensemble. The second command
checks the real weights against a stored real input
(`tests/fixtures/snapshot_20261028T1552Z.*`): the probability must match
to 1e-4.

## Training code

The training code is not in this repository. The network was trained with
the thesis pipeline on AIA level-1 images: solar cycle 24 for training,
2019–2022 for validation and 2023–2025 for testing (docs/model-card.md,
"Training data"). Retraining on the archive this service builds
(docs/archive-format.md) is the next step.

## Layout

```
flare_nrt/  config, grid (2-min slots), source (JSOC), qc, preprocess,
           model (ensemble, SHA-256, Grad-CAM), detector (baseline, alert,
           events), loop, artifacts (tensors, Grad-CAM), localize, render,
           server (HTTP), store, retention, curate, goes, validate,
           fetch_weights
deploy/    systemd units and the deployment note
docs/      model card, archive format
tests/     offline tests; fixtures/ holds one real model input
```

Licence: Apache-2.0. Data products: CC BY 4.0.
