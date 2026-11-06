"""Pure decision logic (no I/O): rolling baseline selection, the alert rule,
and alert→event merging. Everything a user sees as "detected" comes from here
and is covered by tests/test_detector.py."""

from __future__ import annotations

import datetime as dt
from collections import OrderedDict
from dataclasses import asdict, dataclass

import torch

from .grid import CADENCE, iso, parse_iso


def baseline_slots(slot: dt.datetime, lag_min: int, n_baseline: int) -> list[dt.datetime]:
    """Slots averaged into the baseline of ``slot``: t−L, t−L−2, …"""
    return [slot - dt.timedelta(minutes=lag_min) - j * CADENCE for j in range(n_baseline)]


class FrameBuffer:
    """Last few preprocessed currents, keyed by slot. ``None`` marks a slot
    rejected by QC (so it is never used as a baseline either)."""

    def __init__(self, keep: dt.timedelta):
        self.keep = keep
        self._d: OrderedDict[dt.datetime, torch.Tensor | None] = OrderedDict()

    def __contains__(self, slot) -> bool:
        return slot in self._d

    def get(self, slot):
        return self._d.get(slot)

    def put(self, slot, tensor) -> None:
        self._d[slot] = tensor
        self._d.move_to_end(slot)
        newest = max(self._d)
        for s in [s for s in self._d if s < newest - self.keep]:
            del self._d[s]


def is_alert(prob: float, threshold: float) -> bool:
    return prob >= threshold


@dataclass
class Event:
    event_id: int
    t_start: str
    t_end: str
    t_pmax: str
    p_max: float
    n_alert_frames: int
    location: dict | None = None
    closed: bool = False


class EventTracker:
    """Alerts closer than ``merge_gap`` belong to one event. An event closes
    when a slot more than ``merge_gap`` after its last alert is processed."""

    def __init__(self, merge_gap_min: int, last_id: int = 0, current: dict | None = None):
        self.gap = dt.timedelta(minutes=merge_gap_min)
        self.last_id = last_id
        self.current: Event | None = Event(**current) if current else None

    def update(self, slot: dt.datetime, alert: bool, prob: float, location: dict | None = None):
        """Returns (event_id or None, opened: bool, closed_event or None)."""
        closed = None
        if self.current is not None and slot - parse_iso(self.current.t_end) > self.gap:
            closed = self.close()
        if not alert:
            return (self.current.event_id if self.current else None), False, closed
        if self.current is None:
            self.last_id += 1
            self.current = Event(self.last_id, iso(slot), iso(slot), iso(slot), prob, 1, location)
            return self.last_id, True, closed
        ev = self.current
        ev.t_end = iso(slot)
        ev.n_alert_frames += 1
        if prob > ev.p_max:
            ev.p_max, ev.t_pmax = prob, iso(slot)
            if location:
                ev.location = location
        return ev.event_id, False, closed

    def close(self) -> Event | None:
        ev, self.current = self.current, None
        if ev:
            ev.closed = True
        return ev

    def state(self) -> dict:
        return {"last_id": self.last_id, "current": asdict(self.current) if self.current else None}
