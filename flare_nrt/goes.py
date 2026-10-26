"""GOES flare list used as labels (curation) and ground truth (validation).

Sources:
- NOAA SWPC ``xray-flares-7-day.json`` (public domain, HTTPS) — the last
  7 days, so curation must run at least weekly to get labels.
- A CSV with ``tstart,tpeak,tend,fclass`` (e.g. an export of HEK / SWPC
  event lists) for backfills and offline tests.
"""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass
from pathlib import Path

from .grid import as_utc

_LETTER = {"A": 1e-8, "B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}


def class_flux(cls: str) -> float:
    """'M2.5' → 2.5e-5 W/m². Unknown → 0."""
    try:
        return _LETTER[cls[0].upper()] * float(cls[1:] or 1)
    except (KeyError, ValueError, IndexError, TypeError):
        return 0.0


@dataclass(frozen=True)
class Flare:
    tstart: dt.datetime
    tpeak: dt.datetime
    tend: dt.datetime
    fclass: str

    @property
    def flux(self) -> float:
        return class_flux(self.fclass)

    @property
    def letter(self) -> str:
        return self.fclass[:1].upper()

    @property
    def flare_id(self) -> str:
        return f"{self.tstart:%Y%m%dT%H%M}_{self.fclass}"


def _t(v) -> dt.datetime | None:
    if not v or str(v).lower() in {"unk", "null", "none", "nan"}:
        return None
    try:
        return as_utc(dt.datetime.fromisoformat(str(v).replace("Z", "+00:00").replace(" ", "T")))
    except ValueError:
        return None


def _valid(tstart, tpeak, tend, cls) -> Flare | None:
    if tstart is None or tpeak is None or not cls:
        return None
    tend = tend or tpeak + (tpeak - tstart)  # end unknown: mirror the rise
    if not tstart <= tpeak <= tend:
        return None
    return Flare(tstart, tpeak, tend, str(cls).strip())


def parse_swpc(payload: list[dict]) -> list[Flare]:
    out = []
    for r in payload:
        f = _valid(_t(r.get("begin_time")), _t(r.get("max_time")), _t(r.get("end_time")),
                   r.get("max_class"))
        if f:
            out.append(f)
    return sorted(set(out), key=lambda f: f.tstart)


def read_csv(path: Path) -> list[Flare]:
    out = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            f = _valid(_t(r.get("tstart")), _t(r.get("tpeak")), _t(r.get("tend")), r.get("fclass"))
            if f:
                out.append(f)
    return sorted(set(out), key=lambda f: f.tstart)


def fetch_swpc(url: str, timeout: float = 30) -> list[Flare]:
    import requests

    r = requests.get(url, timeout=timeout, headers={"User-Agent": "flare-nrt (HelioUSACH)"})
    r.raise_for_status()
    return parse_swpc(r.json())


def write_csv(path: Path, flares: list[Flare]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["flare_id", "tstart", "tpeak", "tend", "fclass", "flux_wm2"])
        for f in flares:
            w.writerow([f.flare_id, f.tstart.isoformat(), f.tpeak.isoformat(), f.tend.isoformat(),
                        f.fclass, f"{f.flux:.3e}"])
