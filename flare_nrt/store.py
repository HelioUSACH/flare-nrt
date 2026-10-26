"""On-disk outputs of the loop. All writes are atomic (tmp + rename) or
append-only, so a crash never leaves a half-written file the API would serve.

- ``results/YYYYMMDD.jsonl``: one line per processed slot, every status
  (``ok``, ``qc_reject``, ``missing``, ``no_baseline``). This is the
  producer's permanent record and the source for curation and validation.
- ``events.csv``: one row per closed event.
- ``state/status.json``: the latest result, read by the API on each request.
- ``state/loop_state.json``: what a restart needs (last slot, open event).
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from .detector import Event
from .grid import day_key, parse_iso

EVENT_FIELDS = ["event_id", "t_start", "t_end", "t_pmax", "p_max", "n_alert_frames", "location"]


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, default=str))
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return default


class ResultStore:
    def __init__(self, results_dir: Path):
        self.dir = Path(results_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._seen_day: str | None = None
        self._seen: set[str] = set()

    def path(self, day: str) -> Path:
        return self.dir / f"{day}.jsonl"

    def _load_seen(self, day: str) -> None:
        if self._seen_day == day:
            return
        self._seen_day, self._seen = day, set()
        p = self.path(day)
        if p.exists():
            for line in p.read_text().splitlines():
                try:
                    self._seen.add(json.loads(line)["slot"])
                except (json.JSONDecodeError, KeyError):
                    continue

    def append(self, row: dict) -> bool:
        """Appends unless that slot is already recorded (restart safety)."""
        day = day_key(parse_iso(row["slot"]))
        self._load_seen(day)
        if row["slot"] in self._seen:
            return False
        with open(self.path(day), "a") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._seen.add(row["slot"])
        return True

    def read_day(self, day: str) -> list[dict]:
        p = self.path(day)
        if not p.exists():
            return []
        rows = []
        for line in p.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows


def append_event(path: Path, ev: Event) -> None:
    new = not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVENT_FIELDS)
        if new:
            w.writeheader()
        row = {k: getattr(ev, k) for k in EVENT_FIELDS}
        row["location"] = json.dumps(ev.location) if ev.location else ""
        w.writerow(row)


def read_events(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["event_id"] = int(r["event_id"])
        r["p_max"] = float(r["p_max"])
        r["n_alert_frames"] = int(r["n_alert_frames"])
        r["location"] = json.loads(r["location"]) if r.get("location") else None
    return rows
