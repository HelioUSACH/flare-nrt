# Model card: `usach.flare.nrt` — Centella

## Summary

**Centella** is HelioUSACH's near-real-time detector of M- and X-class
solar flares in SDO/AIA synoptic images. It runs every 2 minutes and
produces the `usach.flare.nrt` product of the Clima Espacial USACH portal.

For each AIA slot it reports:

- a **probability**;
- an **alert** when the probability is ≥ 0.80;
- **events**: alerts less than 10 min apart are merged into one;
- a **state** per slot: quiet / onset / active / decay;
- an **experimental** location;
- a **Grad-CAM** map.

It is meant for people who want an image-based confirmation of a flare and
where it is on the disk, alongside GOES X-rays. Maturity: **beta**, the same
as the portal's manifest. It is not ready for "operational": no
live-period numbers yet, and the threshold was set on the evaluation month
itself.

## Inputs

Four AIA synoptic channels, 131, 171, 304 and 1600 Å, at 1024², level 1.5.

- **Source:** in real time, the quicklook images from JSOC
  `https://jsoc1.stanford.edu/data/aia/synoptic/mostrecent/`; gaps are
  filled from the dated archive (`…/synoptic/YYYY/MM/DD/HHH00/`).
- **Cadence:** 2-min slots.
- **Latency:** T_OBS to availability ≈ 7–10 min (live, 2026-10-28).
- **Rejected frames:** a frame is not used if it fails QC
  (`centella/qc.py`: QUALITY flags other than the quicklook bit, exposure,
  wavelength, blank or non-finite data, T_OBS > 90 s from the slot,
  channels > 120 s apart, sudden median jumps), or if its newest T_OBS is
  older than 15 min.

## Method

An ensemble of 5 ResNet-18 networks with 8 input channels and one logit
each. The probability is the mean of the 5 sigmoids. There is no gating
or smoothing: `raw_score` is the same number.

- **Input:** the 4 current channels, plus current − mean of the frames at
  t−8, t−10 and t−12 min (the "rolling baseline", L = 8 min). From 2.1, at
  least 2 of the 3 baseline frames are required.
- **Preprocessing:** bilinear 1024→512, then asinh normalisation with a
  global v_high per channel.
- **Events:** alerts less than 10 min apart are one event.
- **Location:** the brightest region of the 131 Å difference in DN/s.
- **Grad-CAM:** on each member's last block (`layer4`), averaged over the
  members.

Code: `HelioUSACH/flare-nrt` (`centella/`).

## Training data

The network was trained by the thesis pipeline (training code outside this
repository, see the README), on **AIA level-1** images, not on synoptic
data:

- **Positives:** sequences around GOES M/X peaks.
- **Negatives:** active-region and quiet-Sun sequences.
- **Split:** solar cycle 24 for training, 2019–2022 (cycle 25) for
  validation, 2023–2025 for testing.
- **Baseline during training:** the mean of the first 3 frames of each
  sequence. For positives this is ~30 min before the peak, much older than
  the live baseline (8–12 min).

The weights are unchanged since 1.x. Version 2 changed how the input is
built (the rolling baseline) and recalibrated the threshold on synoptic
data.

## Performance

**Ground truth:** the GOES flare list (SWPC).

- **Detected:** an M/X flare is detected if an alert falls in
  [start − 10 min, end + 5 min].
- **False alarm:** an alert run (alerts merged with a 10-min gap) that
  overlaps no C-or-larger flare in [start − 10, end + 30].
- **Runs on C flares** are counted apart: neither hits nor false alarms.

Code: `centella/validate.py`.

**Evaluation period: 2026-06-19 → 07-17.** This is a replay of the
synoptic stream from the dated archive: 28.2 days, 95 % of slots usable,
54 M/X and 161 C flares. The threshold was chosen on this month, so these
numbers are optimistic.

| rule | M/X detected | ≥ M2 | M1–M2 | false alarms | first alert vs GOES start / peak (median) |
|---|---|---|---|---|---|
| **GOES XRS event rule** (SWPC event start: the flux rise that defines the labels) | 100 % by construction | 100 % | 100 % | 0 by construction | 0 min (reference) |
| **Centella 2.x**, L = 8, threshold 0.80, 3 of 3 baselines | **72 % (39/54)** | **100 % (22/22)** | **53 %** | **3** (≈ 1 per 9 days) | **+9 min / −1 min** |
| **`aia131_rise`** (no network, same frames, disk-integrated 131 Å rise ≥ k, k set for 3 false alarms) | pending | | | 3 | |
| No-skill (climatology) | 0 alerts → 0 % | | | 0 | |

What this says against the baselines:

- **Against GOES XRS:** Centella does **not** detect earlier. Its first
  alert comes a median 9 min after the GOES start and 1 min before the
  GOES peak. Its value is an independent, imaging-based confirmation with
  a position on the disk, not an earlier warning.
- **Against `aia131_rise`:** this is the test of whether the network adds
  anything over a simple brightness rule on the same frames. It needs the
  month replayed with 2.1.0 or later, which records `aia131_excess`:

  ```
  python -m centella.loop --replay <month FITS>
  python -m centella.validate --start 2026-06-19 --end 2026-07-17
  ```

  The report's `baseline` block fills this row.

**Independent check, 2026-04-23 → 25:** 48 h with 10 M/X flares,
including X2.4 and X2.5, run through the operational loop with L = 6 and
threshold 0.80. Result: 10/10 detected, 0 false alarms, 2 alerts on C
flares.

**Live:** no live-period numbers yet. The weekly validation unit
(`centella-validate.timer`) will write them to `reports/`; this section
will be updated from those reports.

## Known limitations

- **Baseline-age shortcut.** The network's output grows with the age of the
  baseline even without a flare. This is a training artefact: positives had
  older baselines than negatives. L is therefore fixed at 8 min and the
  threshold is high. Fix: retrain with matched baseline ages (the archive in
  `docs/archive-format.md` is built for that).
- **Grad-CAM peaks off the disk in quiet Sun.** On a quiet frame
  (2026-10-28 15:52 UT, p = 0.047), the ensemble-mean map is strongest in
  the corners outside the solar disk. Each member's map is scaled to its
  own maximum, so a weak map still shows a peak somewhere. That the peak is
  off-disk is consistent with the shortcut above. The public PNG therefore
  scales the colour with probability / threshold.
- **Domain gap, synoptic vs level-1.** With an identical baseline, the
  output on synoptic data is ~0.05–0.13 in quiet Sun, against ~0.001 on
  level-1.
- **Live inputs differ from the evaluated ones.** The month above used the
  dated synoptic archive with all 3 baseline frames. Since 2.1, the live
  input is the **quicklook** `mostrecent/` stream (QUALITY bit 30 set),
  and 2 of 3 baseline frames are accepted when one is missing
  (`n_baseline_used` in every result). The effect of either on skill is not
  measured yet.
- **Dense activity and small flares.** All 15 missed flares are
  M1.0–M1.4. 12 of them fall on 2026-07-04/05, when there was a flare
  every ~40 min and the previous flare sat inside the baseline.
- **Quiet Sun under-tested.** The evaluation month was very active, so the
  false-alarm rate over long quiet periods is not yet measured.
- **Coverage.** On the first live day (2026-10-28), many slots lacked a
  channel, because `mostrecent/` keeps only the newest image and JSOC drops
  connections. Version 2.1 polls every 30 s and retries; `coverage_24h` in
  `/api/health` tracks it. While inputs are older than 15 min the product
  reads `stale`.
- **Low-probability 2-frame alternation (seen in 1.x).** Not yet measured
  for 2.x. `centella.validate` now reports it (`alternation`: the fraction
  of consecutive low-probability steps that reverse sign; ~0.5 for noise,
  ~1 for strict alternation).
- **Location is experimental.** It is the brightest region of the 131 Å
  difference. Since 2.3, `centella.validate` measures its distance to the
  catalogue position (LMSAL "SSW Latest Events" via HEK) for each detected
  M/X flare; the calibration month's numbers are pending.

## Maintainers / links

- **Producer repository:** `HelioUSACH/flare-nrt`, maintained by Nicolás
  Campos (HelioUSACH).
- **Deployment:** portal maintainers.
- **Weights:** a GitHub release asset, with its SHA-256 pinned in
  `centella/fetch_weights.py`.
- **Documents:** integration sheet `INTEGRATION.md`; archive format
  `docs/archive-format.md`.
