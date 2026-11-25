"""Says when the loop stops producing results (1.3.1).

From 2026-11-24 23:00 UT every slot was written as ``qc_reject`` for more
than a day and nothing said so: the loop was running, downloading and
advancing. ``NoResultWatch`` follows every row the loop writes:

- after ``warn_slots`` slots in a row without a result (default 15 = 30 min;
  any status but ``ok``: ``qc_reject``, ``missing``, ``no_baseline``,
  ``inputs_old``, ``error``) it logs one WARNING with the reason codes
  counted, e.g. ``30 min without a result (since …): quality:131=1073741828
  ×15, …``, then one every hour while it lasts;
- when a result comes again after such a warning, one INFO;
- ``state/no_result.json`` holds ``last_ok_slot``, ``rejected_streak``,
  ``rejected_since``, ``rejected_streak_reasons`` and
  ``last_reject_reasons``, which ``/api/status`` serves.
"""

from __future__ import annotations

import datetime as dt
import logging
from collections import Counter
from pathlib import Path

from .grid import CADENCE, day_key, parse_iso
from .store import read_json, write_json_atomic

log = logging.getLogger("flare_nrt.loop")

SLOTS_PER_HOUR = int(dt.timedelta(hours=1) / CADENCE)
SHOWN_REASONS = 6


def reason_codes(row: dict) -> list[str]:
    """Why a row is not a result, as reason codes (QC's own, e.g.
    ``quality:131=1073741828``; ``missing:171``; else the row's reason)."""
    status = row.get("status")
    if status == "qc_reject":
        return list(row.get("qc") or []) or ["qc"]
    if status == "missing":
        return [f"missing:{w}" for w in row.get("channels_missing") or []] or ["missing"]
    if status == "error":
        return ["error"]
    return [row.get("reason") or status or "unknown"]


def duration(n_slots: int) -> str:
    """``n_slots`` 2-min slots as ``30 min``, ``1 h 30 min``, ``2 d 3 h``."""
    m = int(n_slots * CADENCE.total_seconds() // 60)
    d, h, m = m // 1440, m % 1440 // 60, m % 60
    if d:
        return f"{d} d {h} h" if h else f"{d} d"
    if h:
        return f"{h} h {m} min" if m else f"{h} h"
    return f"{m} min"


def counted(reasons: Counter) -> str:
    top = ", ".join(f"{k} ×{n}" for k, n in reasons.most_common(SHOWN_REASONS))
    more = len(reasons) - SHOWN_REASONS
    return top + (f", … (+{more} more)" if more > 0 else "")


class NoResultWatch:
    def __init__(self, path: Path, warn_slots: int = 15):
        self.path = Path(path)
        self.warn_slots = max(1, int(warn_slots))
        st = read_json(self.path, {}) or {}
        self.known = bool(st)
        self.last_ok_slot: str | None = st.get("last_ok_slot")
        self.streak = int(st.get("rejected_streak") or 0)
        self.since: str | None = st.get("rejected_since")
        self.reasons: Counter = Counter(st.get("rejected_streak_reasons") or {})
        self.last_reasons: list[str] = list(st.get("last_reject_reasons") or [])

    def state(self) -> dict:
        return {"last_ok_slot": self.last_ok_slot, "rejected_streak": self.streak,
                "rejected_since": self.since, "rejected_streak_reasons": dict(self.reasons.most_common()),
                "last_reject_reasons": self.last_reasons}

    def _what(self) -> str:
        last = f", last result {self.last_ok_slot}" if self.last_ok_slot else ""
        return f"since {self.since}{last}: {counted(self.reasons)}"

    def update(self, row: dict, quiet: bool = False) -> None:
        """One row the loop wrote (every status), in slot order."""
        if row.get("status") == "ok":
            if self.streak >= self.warn_slots and not quiet:
                log.info("results resumed at %s after %s without a result (%d slots %s)",
                         row["slot"], duration(self.streak), self.streak, self._what())
            self.last_ok_slot, self.streak, self.since = row["slot"], 0, None
            self.reasons.clear()
        else:
            codes = reason_codes(row)
            if self.streak == 0:
                self.since = row["slot"]
            self.streak += 1
            self.reasons.update(codes)
            self.last_reasons = codes
            due = self.streak >= self.warn_slots and (self.streak - self.warn_slots) % SLOTS_PER_HOUR == 0
            if due and not quiet:
                log.warning("%s without a result (%s)", duration(self.streak), self._what())
        self.known = True
        if not quiet:                     # bootstrap writes once, at its end
            write_json_atomic(self.path, self.state())

    def bootstrap(self, store, last_ok_slot: str | None, now: dt.datetime) -> None:
        """First start with this file (an upgrade): rebuild the streak from
        the results after the last result (``status.json``), quietly."""
        if self.known or not last_ok_slot:
            return
        self.last_ok_slot = last_ok_slot
        t, days = parse_iso(last_ok_slot), []
        while t.date() <= now.date():
            days.append(day_key(t))
            t += dt.timedelta(days=1)
        for d in days:
            for r in store.read_day(d):
                if r.get("slot", "") > last_ok_slot:
                    self.update(r, quiet=True)
        write_json_atomic(self.path, self.state())

    def announce(self) -> None:
        """At start: a streak already past the warning says so once."""
        if self.streak >= self.warn_slots:
            log.warning("starting with %s without a result (%s)", duration(self.streak), self._what())
