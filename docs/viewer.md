# Retrospective viewer

`GET /viewer` on the Centella API (internal, 127.0.0.1) is a page for
browsing the results of any period frame by frame. It shows:

- a **timeline** of the probability, with the threshold, Centella's events,
  the GOES flares (C/M/X) and the slots without a result. Wheel to zoom,
  drag to pan, click to jump, double-click to see the whole window;
- the **4 AIA channels** of the selected slot, as an image or as the
  difference the model looks at (current − mean of its baselines);
- the **Grad-CAM** of the frame (alerting frames, and the newest frame live);
- the **frame's row**: status and reason, QC codes, missing channels,
  probability and members, baselines used, the 131 Å disk rise, location,
  the GOES flare it falls in;
- **lists** of Centella's events (each matched to a GOES flare, or "none" =
  false alarm) and of the GOES flares (each matched to Centella's first
  alert, or "no detectado"). A click on a row jumps there.

Controls: ← / → one frame, Shift + ← / → one hour, Space to play (1–8
frames/s), J / K to the previous / next event or M/X flare, and "solo
frames con resultado" to skip slots without one. The URL keeps the slot,
window and view, so a link reopens the same frame.

It reads only the API: `/api/range`, `/api/series`, `/api/events`,
`/api/flares`, `/api/frame/{slot}`, `/api/image/{wave}.png?slot=`,
`/api/gradcam/{slot}.png` (INTEGRATION.md).

## Browsing the live service

On the server, open an SSH tunnel and browse:

```bash
ssh -L 5050:127.0.0.1:5050 <server>
# then http://127.0.0.1:5050/viewer
```

Images are available while the FITS are kept (7 days). Results, events
and the Grad-CAM of alerts are kept longer.

## A past period (e.g. the calibration month on the processing server)

The FITS stay where they are; the replay writes results and Grad-CAMs to
a data directory of its own, and the API finds the FITS through
`CENTELLA_FITS_DIRS`.

```bash
export CENTELLA_DATA_DIR=$HOME/centella-replay/junio-julio     # anything; not the live one
export CENTELLA_WEIGHTS=…/deployment_ensemble_thr058.pt
export CENTELLA_NORM_STATS=…/global_norm_stats.json
export MES=/path/to/the/month/FITS                             # AIAYYYYMMDD_HHMM_WWWW.fits, any depth

# 1. the GOES flare list of the period (SWPC, via HEK)
python -m centella.goes --start 2026-06-19 --end 2026-07-17 --out $CENTELLA_DATA_DIR/flares.csv

# 2. replay: results, events, Grad-CAM of every alerting frame (no tensors)
nohup python -m centella.loop --replay $MES > $CENTELLA_DATA_DIR/replay.log 2>&1 &
tail -f $CENTELLA_DATA_DIR/replay.log      # progress every 12 h of data; --resume continues after a stop

# 3. scores against GOES and against the aia131_rise baseline
python -m centella.validate --start 2026-06-19 --end 2026-07-17 \
    --flares-csv $CENTELLA_DATA_DIR/flares.csv --out $CENTELLA_DATA_DIR/validation.json

# 4. the viewer
CENTELLA_FITS_DIRS=$MES CENTELLA_FLARES_CSV=$CENTELLA_DATA_DIR/flares.csv \
    uvicorn centella.server:app --host 127.0.0.1 --port 5051
# from your machine: ssh -L 5051:127.0.0.1:5051 <server>, then http://127.0.0.1:5051/viewer
```

The first look at a slot renders its 4 PNGs from the FITS, about 0.5 s
each (measured on 2 vCPU; they render in parallel). A first pass therefore
plays at about 1 frame/s, whatever speed is selected. The PNGs are cached
in `$CENTELLA_DATA_DIR/cache` for `CENTELLA_IMAGE_CACHE_DAYS` (1 day), so a
second pass plays at the selected speed.
