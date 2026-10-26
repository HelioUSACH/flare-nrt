# flare-nrt

Near-real-time solar flare detection on SDO/AIA synoptic images, producer
of the `usach.flare.nrt` product of
[Clima Espacial USACH](https://github.com/HelioUSACH/clima-espacial).
Integration contract: [`INTEGRATION.md`](INTEGRATION.md). Model card:
[`docs/model-card.md`](docs/model-card.md). Archive for retraining:
[`docs/archive-format.md`](docs/archive-format.md).

## Install (Ubuntu 24.04, no GPU)

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-dev.txt      # or: pip install -r requirements.lock.txt
```

## Weights

Download `deployment_ensemble_thr058.pt` and `global_norm_stats.json` from
the GitHub release of the tag you deploy, then:

```bash
sha256sum deployment_ensemble_thr058.pt     # must match the release notes
export FLARE_WEIGHTS=$PWD/deployment_ensemble_thr058.pt
export FLARE_WEIGHTS_SHA256=<that hash>
export FLARE_NORM_STATS=$PWD/global_norm_stats.json
export FLARE_DATA_DIR=$PWD/var
```

## Run

```bash
python -m flare_nrt.loop                       # live loop (downloads from JSOC)
uvicorn flare_nrt.server:app --host 127.0.0.1 --port 5050
python -m flare_nrt.curate                     # daily: archive + labels + retention
python -m flare_nrt.validate --start 2026-10-27 --end 2026-11-02
```

Replay existing FITS through the same code path (validation):

```bash
python -m flare_nrt.loop --replay /path/to/fits --data-dir /tmp/replay
```

Configuration is environment only (`.env.example`); every path is a
setting. On the server the units in `deploy/` run it (maintainers install
them).

## Test

```bash
ruff check .
pytest            # offline, CPU, < 1 min; uses a tiny stand-in ensemble
```

`tests/test_real_weights.py` checks the real weights against a stored slot
when `FLARE_REAL_*` is set (see its docstring).

## Training code

Not in this repository yet: the network was trained with the thesis
pipeline (AIA level-1; docs/model-card.md "Training data").
Retraining on the archive this service builds is the next step
(docs/archive-format.md "Building a training set").

## Layout

```
flare_nrt/  config, grid (2-min slots), source (JSOC/replay), qc, preprocess,
            model (ensemble + SHA-256), detector (baseline, alert, events),
            localize, loop, store, server, curate, goes, retention, validate
deploy/     example systemd units
docs/       model card, archive format
tests/      offline tests with synthetic FITS
```

Licence: Apache-2.0. Data products: CC BY 4.0.
