# Model card: `usach.flare.nrt`

## Summary

Near-real-time detector of M- and X-class solar flares in SDO/AIA synoptic
images, run every 2 minutes. It reports a probability per AIA slot, an alert
when it is ≥ 0.80, alerts merged into events, and an experimental location.
Maturity: **beta**. Version 2.0.0 changes how the input is built (rolling
baseline) and recalibrates the threshold on live data; the network weights
are the same as 1.x.

## Inputs

AIA synoptic 1024² images, 131/171/304/1600 Å, from JSOC's dated archive
(`https://jsoc1.stanford.edu/data/aia/synoptic/YYYY/MM/DD/HHH00/`), 2-min
cadence. Latency from observation to availability: being measured
(`data_latency_s` in every result; `/api/health`).

## Method

Ensemble of 5 ResNet-18 (8 input channels, one logit), probability = mean
of sigmoids. Input = the 4 current channels + current − mean of the frames
at t−8, t−10, t−12 min ("rolling baseline", L = 8 min). Preprocessing:
bilinear 1024→512, asinh normalisation with global v_high per channel.
Frames failing QC (`flare_nrt/qc.py`) are never used, as current or
baseline. Alerts ≤ 10 min apart are one event.

## Training data

AIA level-1 images from the thesis dataset (flare sequences around GOES M/X peaks,
active-region and quiet-Sun negatives), from the thesis pipeline. The
training baseline was the mean of the first 3 frames of each sequence
(~30 min before the peak for positives). Not trained on synoptic data.

## Performance

Ground truth: GOES flare list (SWPC/HEK). A flare is detected if an alert
falls in [start − 10 min, end + 5 min]; an alert run is a false alarm if it
overlaps no C-or-larger flare in [start − 10, end + 30]; runs on C flares
are counted apart.

**Calibration month, 2026-06-19 → 07-17** (replay of the synoptic stream,
28.2 days, 95 % of slots usable; 54 M/X, 161 C flares). The threshold was
chosen on this month, so these numbers are optimistic:

| L (min) | threshold | M/X detected | ≥ M2 | M1–M2 | false alarms | alert vs GOES peak (median) |
|---|---|---|---|---|---|---|
| 6 | 0.80 | 63 % | 95 % | 41 % | 3 | −1 min |
| **8** | **0.80** | **78 %** | **100 % (22/22)** | **62 %** | **4** (≈1/week) | **−1 min** |
| 10 | 0.80 | 80 % | 100 % | 66 % | 17 | −2 min |

**Independent check, 2026-04-23 → 25** (48 h, 10 M/X incl. X2.4, X2.5;
L = 6, threshold 0.80, the operational loop itself): 10/10 detected,
0 false alarms, 2 alerts on C flares.

**Against the baseline that matters operationally — the GOES X-ray event
start:** the median alert comes **9 min after** the GOES start and 1 min
before the GOES peak. The model does **not** give earlier warning than
GOES XRS. Its value is an independent, imaging-based confirmation with a
location on the disk. Report the live numbers here weekly
(`python -m flare_nrt.validate`).

## Known limitations

- **Baseline-age shortcut.** The network's output grows with the age of the
  baseline even without a flare (training artefact: positives had older
  baselines than negatives). L is therefore fixed at 8 min and the
  threshold is high. Fix: retrain with matched baseline ages (the archive
  in `docs/archive-format.md` is built for that).
- **Domain gap synoptic vs level-1.** With an identical baseline the
  output on synoptic data is ~0.13, vs ~0.001 on level-1.
- **Dense activity.** Missed flares are all M1.0–M1.4 with p_max 0.36–0.68;
  10 of the 12 misses fall on 2026-07-04/05 (a flare every ~40 min), when
  the previous flare sits inside the baseline.
- **Quiet Sun under-tested.** The calibration month was very active; the
  false-alarm rate on long quiet periods is not yet measured.
- **QC gaps.** Frames with QUALITY ≠ 0 are dropped: the daily 21:00 UT
  synoptic frame, and in July 2026 a 30–90 min daily window (QUALITY bit
  2097152). During those windows the product is `stale`.
- **Location is experimental**: brightest 131 Å difference region, not yet
  validated against flare catalogue positions.

## Maintainers / links

Producer repository `HelioUSACH/flare-nrt`; weights as a release asset with
SHA-256; integration sheet `INTEGRATION.md`; archive format
`docs/archive-format.md`.
