# Archive format `flare-nrt-archive/1`

What the service keeps from its live input, so the model can be retrained
and re-validated on the data it actually runs on.

## Data flow and retention

```
JSOC ──► staging/YYYYMMDD/AIA*.fits      raw downloads, every slot     kept 7 × 24 h
              │  (loop: QC, inference)
              ▼
         results/YYYYMMDD.jsonl            one row per slot, every status   kept 400 days
              │  (curate, daily 04:20 UT, labels from GOES)
              ▼
         archive/YYYY/MM/DD/               curated windows + labels        kept 7 days on the server
              │  (off-server pull, every few days)
              ▼
         long-term archive (another host)  the training set
```

Volumes: raw synoptic ≈ 0.95 MB per file, 3.8 MB per slot, ≈ 2.75 GB per
day for 4 channels (measured on the live stream, 2026-10-28; 2.6 GB/day on
the 2026-06/07 month). Staging steady state ≈ 11 GB (4 days). Archive
≈ 0.4 GB (quiet day) to 1.3 GB (very active day). Its files are hard
links into staging (below), so only the archive days older than staging
(days 5–7) add space: ≈ 1.2–3.9 GB. The
whole data directory: INTEGRATION.md "Resources". `FLARE_NRT_MIN_FREE_GB`
(10 GB) is a hard floor (retention.py).

Archive files are hard links to the staging copy when on the same
filesystem, so curation costs no extra space until staging is pruned.

## Cleaning rules

1. **Per frame (loop, `qc.py`)**: a slot is rejected if any channel is
   unreadable, has a QUALITY flag other than bit 30 (quicklook, set on every
   near-real-time frame), EXPTIME ≤ 0, the wrong WAVELNTH, > 1 %
   non-finite pixels, T_OBS > 90 s from the slot, channels > 120 s apart,
   a blank readout, or a disk median per second of exposure outside
   [0.25, 4]× its recent reference (AIA shortens exposures during flares,
   so raw counts are never compared).
   Rejected slots are never used as baselines either. The reason codes are
   stored (`qc` in results and frames.csv).
2. **What is kept (curate)**: complete 4-channel slots inside
   - flare windows: GOES flares ≥ C1.0, [start − 45 min, end + 15 min];
   - alert windows: the loop's alert runs, same padding;
   - 4 negative blocks of 60 min per day, ≥ 60 min from any of the above,
     chosen with a seed fixed by the date.
   Raw files of QC-rejected slots inside a window are kept too (so the
   rule can be revisited); their status says they were rejected.
3. **What is deleted**: raw staging files 7 × 24 h after their slot
   (curation has run six times by then); archive days on the server after
   7 days; results after 400 days; model-input tensors after 2 days;
   oldest first when the disk floor is hit (flare_nrt/retention.py).

Windows are contiguous so that, for positives and negatives alike, a
training pair can be built with any baseline age up to 45 min — the
condition for retraining without the baseline-age shortcut.

## Layout of one archive day

```
archive/2026/10/25/
├── day.json      summary, settings, windows, SHA-256 of every file (written last)
├── frames.csv    720 rows: every slot of the day
├── flares.csv    GOES flares overlapping the day
└── fits/         AIAYYYYMMDD_HHMM_WWWW.fits  (raw JSOC synoptic files, unmodified)
```

### `frames.csv`

| column | meaning |
|---|---|
| `slot` | slot time, ISO 8601 UTC `Z` |
| `status` | loop status: `ok`, `qc_reject`, `no_baseline`, `missing`, `error`, `not_processed` |
| `qc` | QC reason codes, `;`-separated |
| `probability` | model output (empty unless `ok`) |
| `alert` | 1 if probability ≥ threshold |
| `event_id` | event the slot belongs to |
| `model_version`, `lag_min`, `threshold` | configuration that produced the row |
| `keep`, `keep_reason` | 1 if its FITS are in `fits/`; `flare` / `alert` / `negative` |
| `files_complete` | 1 if all 4 channels were downloaded |
| `flare_id`, `flare_class`, `flare_flux` | biggest GOES flare whose padded window holds the slot |
| `phase` | `none`, `pre` (before start), `rise`, `decay`, `post` (after end) |
| `min_from_start`, `min_from_peak` | minutes relative to that flare |

### `flares.csv`

`flare_id, tstart, tpeak, tend, fclass, flux_wm2` — `flare_id` is
`YYYYMMDDTHHMM_<class>` of the start.

### `day.json`

`schema`, `day`, `generated_at`, `curator_version`, `model_versions`,
`settings`, `labels` (`source`, `complete`, counts by letter), `slots`
(totals and kept by reason), `windows` (the intervals), `files`
(`name → sha256`). `labels.complete` is false when the day was curated
more than 6 days late (SWPC's list covers 7 days): relabel it with
`curate --day … --flares-csv … --force` from a HEK export.

## Building a training set

A sample = (current slot *t*, baseline slots *t − a*, *t − a − 2*, …) with
*t* and its baselines `keep = 1` and `status ≠ qc_reject`, from the same
day folder (or the next/previous one across midnight). Label from
`phase`/`flare_class` of *t*. Draw the baseline age *a* from the same
distribution for positives and negatives.

## Off-server copy

The server keeps 7 days of archive. The maintainers pull it to a backup server; pull at least every few days to the long-term host, e.g.
`rsync -a --ignore-existing <server>:/srv/flare-nrt/archive/ /path/to/archive/`
and check `day.json` hashes after the copy. This is a maintainers' job
(their backup pattern, not this repo's code).
