"""One-off (1.3.1): run again the slots that got no result, through the
same code as the loop, while their FITS are still in staging (4 days).

    python -m flare_nrt.reprocess --since 2026-11-24T23:00:00Z               # up to now
    python -m flare_nrt.reprocess --since ISO --until ISO
    python -m flare_nrt.reprocess --since ISO --dry-run                      # count only, no model

Made for the outage of 2026-11-24/25: from 23:00 UT every near-real-time
frame carried QUALITY bit 2 (``Q_NOASD``), which 1.3.0 rejected, so every
slot was written ``qc_reject``. Slots in [since, until] whose result is
``qc_reject``, ``missing`` or ``no_baseline`` (a baseline that was
rejected), or that have none, and whose 4 FITS are in staging, are
processed in slot order by the loop's ``Processor``: same QC (now with the
1.3.1 mask), baselines (t−8, −10, −12 min, read from staging), model,
alert rule and event tracker. Slots with another result (``ok``,
``inputs_old``, ``error``) are left as they are, and so are slots after
the newest one the loop has written (``loop_state.json``): those are the
running loop's.

Each new row is appended to ``results/YYYYMMDD.jsonl`` after the old one
with ``reprocessed: true``; the last row of a slot is its result
(``ResultStore.read_day``), so ``/api/series``, ``/api/frames``,
``/api/frame``, curation, validation and replays show the new one. The old
row stays in the file (``read_day(day, every_row=True)``).

Events found in the gap are created as usual: appended to ``events.csv``
when they close, with an id the running loop has not used (both take ids
from ``state/event_ids.json`` under the events lock); the loop builds
their replays on its next check (every 5 min), as for any closed event.
The tracker starts with no open event at ``since``: an alert in the first
``merge_gap_min`` after ``since`` is not merged with an event the loop
closed before it (nor, at the other end, with one the loop opens after
the gap). Grad-CAM is computed for alerting frames only (kept 400
days), as in a replay; no model inputs (``tensors/``) or ``render/`` frames
are written. The live loop's state (``status.json``, ``loop_state.json``,
``last_frame.json``, ``no_result.json``) is not touched, so the loop can
keep running.

Cost: one inference per slot (≈ 10 s on the server's 2 vCPU, more under
``CPUQuota=100%`` next to the loop) plus a Grad-CAM per alert: ≈ 4–8 h per
day of gap. It prints its progress every 30 slots and a summary.
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from .config import Settings
from .grid import CADENCE, day_key, floor_slot, iso, parse_iso, utcnow
from .source import StagingSource
from .store import ResultStore, read_json

log = logging.getLogger("flare_nrt.reprocess")

REDO = ("qc_reject", "missing", "no_baseline")


def plan(cfg: Settings, since: dt.datetime, until: dt.datetime) -> tuple[list[dt.datetime], Counter]:
    """(slots to reprocess, in order; what the range holds, by result).
    Never past the newest slot the loop has written (``loop_state.json``):
    the slots after it are the running loop's."""
    store, src = ResultStore(cfg.results_dir), StagingSource(cfg.staging_dir)
    loop_last = (read_json(cfg.loop_state_path, {}) or {}).get("last_slot")
    if loop_last:
        until = min(until, parse_iso(loop_last))
    rows: dict[str, dict] = {}
    d = since.replace(hour=0, minute=0, second=0, microsecond=0)
    while d <= until:
        rows.update({r["slot"]: r for r in store.read_day(day_key(d))})
        d += dt.timedelta(days=1)
    todo, seen = [], Counter()
    s = floor_slot(since)
    s = s if s >= since else s + CADENCE
    while s <= until:
        status = (rows.get(iso(s)) or {}).get("status", "none")
        if status in REDO or status == "none":
            if src.paths(s) is None:
                status += ", FITS gone"
            else:
                todo.append(s)
        seen[status] += 1
        s += CADENCE
    return todo, seen


def reprocess(cfg: Settings, ensemble, since: dt.datetime, until: dt.datetime) -> Counter:
    """Processes ``plan``'s slots; → the new rows' statuses (+ ``events``)."""
    from .artifacts import Artifacts
    from .detector import EventTracker
    from .loop import Processor

    todo, _ = plan(cfg, since, until)
    proc = Processor(cfg, ensemble, StagingSource(cfg.staging_dir), live=False, keep_state=False)
    proc.supersede = True
    proc.artifacts = Artifacts(cfg, ensemble, enabled=True, tensors=False)   # Grad-CAM of alerting frames
    proc.gradcam_wanted = False
    last_id = (read_json(cfg.loop_state_path, {}) or {}).get("last_id", 0)
    proc.tracker = EventTracker(cfg.merge_gap_min, last_id)
    out: Counter = Counter()
    t0 = time.monotonic()
    for i, slot in enumerate(todo, 1):
        row = proc.process(slot, utcnow())
        out[row["status"]] += 1
        out["events"] += row.get("state") == "onset"
        if i % 30 == 0 or i == len(todo):
            left = (time.monotonic() - t0) / i * (len(todo) - i) / 60
            log.info("reprocess at %s: %d/%d slots, %s, ~%.0f min left", iso(slot), i, len(todo), dict(out), left)
    proc.flush()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--since", required=True, help="first slot, ISO 8601 (e.g. 2026-11-24T23:00:00Z)")
    ap.add_argument("--until", help="last slot, ISO 8601 (default: now)")
    ap.add_argument("--dry-run", action="store_true", help="say what would be reprocessed, load nothing")
    ap.add_argument("--data-dir", help="override FLARE_NRT_DATA_DIR")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        stream=sys.stdout)
    cfg = Settings()
    if a.data_dir:
        cfg.data_dir = Path(a.data_dir)
    since = parse_iso(a.since)
    until = min(parse_iso(a.until), utcnow()) if a.until else utcnow()
    if until < since:
        ap.error("--until is before --since")
    todo, seen = plan(cfg, since, until)
    print(f"{iso(since)} → {iso(until)}: {sum(seen.values())} slots by current result {dict(seen)}; "
          f"{len(todo)} to reprocess" + (f", {iso(todo[0])} → {iso(todo[-1])}" if todo else ""))
    if a.dry_run or not todo:
        return 0
    import torch

    from .model import Ensemble

    cfg.ensure_dirs()
    torch.set_num_threads(cfg.torch_threads)
    ens = Ensemble.load(cfg.weights, cfg.weights_sha256)
    t0 = time.monotonic()
    out = reprocess(cfg, ens, since, until)
    n_events = out.pop("events", 0)
    print(f"reprocessed {len(todo)} slots in {(time.monotonic() - t0) / 60:.1f} min: {dict(out)}, "
          f"{n_events} event(s) opened")
    return 0


if __name__ == "__main__":
    sys.exit(main())
