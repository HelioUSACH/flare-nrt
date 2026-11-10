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
from dataclasses import dataclass, field, replace
from pathlib import Path

from .grid import parse_datetime

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
    # helioprojective position (arcsec) when a catalogue gives one; not part of
    # the flare's identity (equality, hashing, flare_id)
    hpc: tuple | None = field(default=None, compare=False)

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
        return parse_datetime(str(v))
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
                try:
                    hpc = (float(r["hpc_x"]), float(r["hpc_y"]))
                except (KeyError, TypeError, ValueError):
                    hpc = None
                out.append(replace(f, hpc=hpc))
    return sorted(set(out), key=lambda f: f.tstart)


def fetch_swpc(url: str, timeout: float = 30) -> list[Flare]:
    import requests

    r = requests.get(url, timeout=timeout, headers={"User-Agent": "flare-nrt (HelioUSACH)"})
    r.raise_for_status()
    return parse_swpc(r.json())


def write_csv(path: Path, flares: list[Flare]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["flare_id", "tstart", "tpeak", "tend", "fclass", "flux_wm2", "hpc_x", "hpc_y"])
        for f in flares:
            x, y = f.hpc if f.hpc else ("", "")
            w.writerow([f.flare_id, f.tstart.isoformat(), f.tpeak.isoformat(), f.tend.isoformat(),
                        f.fclass, f"{f.flux:.3e}", x, y])


HEK_URL = "https://www.lmsal.com/hek/her"


def _hpc(r: dict) -> tuple | None:
    try:
        x, y = float(r["hpc_x"]), float(r["hpc_y"])
    except (KeyError, TypeError, ValueError):
        return None
    return None if x == 0 and y == 0 else (round(x, 1), round(y, 1))   # (0, 0): "no position"


def parse_hek(results: list[dict]) -> list[Flare]:
    """HEK flare events (``event_starttime``, ``event_peaktime``,
    ``event_endtime``, ``fl_goescls``, ``hpc_x``/``hpc_y``) → flares;
    entries without a GOES class are dropped."""
    out = []
    for r in results:
        f = _valid(_t(r.get("event_starttime")), _t(r.get("event_peaktime")), _t(r.get("event_endtime")),
                   r.get("fl_goescls"))
        if f:
            out.append(replace(f, hpc=_hpc(r)))
    return sorted(set(out), key=lambda f: f.tstart)


def _hek_search(frm: str, start: dt.datetime, end: dt.datetime, timeout: float) -> list[dict]:
    import requests

    results, page = [], 1
    while True:
        params = {
            "cosec": 2, "cmd": "search", "type": "column", "event_type": "fl",
            "event_starttime": start.strftime("%Y-%m-%dT%H:%M:%S"),
            "event_endtime": end.strftime("%Y-%m-%dT%H:%M:%S"),
            "event_coordsys": "helioprojective", "x1": -5000, "x2": 5000, "y1": -5000, "y2": 5000,
            "param0": "FRM_NAME", "op0": "=", "value0": frm,
            "return": "event_starttime,event_peaktime,event_endtime,fl_goescls,hpc_x,hpc_y",
            "result_limit": 500, "page": page,
        }
        r = requests.get(HEK_URL, params=params, timeout=timeout, headers={"User-Agent": "flare-nrt (HelioUSACH)"})
        r.raise_for_status()
        js = r.json()
        results += js.get("result", [])
        if not js.get("overmax") or not js.get("result"):
            return results
        page += 1


POSITION_MATCH = dt.timedelta(minutes=6)


def fetch_hek(start: dt.datetime, end: dt.datetime, timeout: float = 60, positions: bool = True) -> list[Flare]:
    """The SWPC flare list for any past period, from the Heliophysics Event
    Knowledgebase (``FRM_NAME = SWPC``), paged 500 at a time. With
    ``positions``, each flare's position is the one LMSAL's "SSW Latest
    Events" (located in AIA images) gives for the same flare (peak within
    6 min, same letter), else SWPC's own when it has one."""
    flares = parse_hek(_hek_search("SWPC", start, end, timeout))
    if not positions:
        return flares
    try:
        lmsal = [f for f in parse_hek(_hek_search("SSW Latest Events", start, end, timeout)) if f.hpc]
    except Exception:  # positions are a bonus: the list itself must not depend on them
        lmsal = []
    out = []
    for f in flares:
        near = [g for g in lmsal if abs(g.tpeak - f.tpeak) <= POSITION_MATCH and g.letter == f.letter]
        best = min(near, key=lambda g: abs(g.tpeak - f.tpeak)) if near else None
        out.append(replace(f, hpc=best.hpc if best else f.hpc))
    return out


def main(argv=None) -> None:
    """python -m flare_nrt.goes --start 2026-06-19 --end 2026-07-17 --out flares.csv"""
    import argparse

    ap = argparse.ArgumentParser(description="GOES flare list (SWPC, via HEK) for a past period → CSV")
    ap.add_argument("--start", required=True, help="first day, YYYY-MM-DD (UTC)")
    ap.add_argument("--end", required=True, help="last day, inclusive")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d0 = dt.datetime.strptime(a.start, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    d1 = dt.datetime.strptime(a.end, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc) + dt.timedelta(days=1)
    try:
        flares = fetch_hek(d0, d1)
    except Exception as e:  # network, HEK down, bad answer: say it in one line
        raise SystemExit(f"HEK did not answer ({type(e).__name__}: {str(e)[:160]})") from None
    write_csv(Path(a.out), flares)
    by = {}
    for f in flares:
        by[f.letter] = by.get(f.letter, 0) + 1
    located = sum(1 for f in flares if f.hpc)
    print(f"{len(flares)} flares → {a.out}  ({', '.join(f'{k}: {v}' for k, v in sorted(by.items()))}; "
          f"{located} with a position)")


if __name__ == "__main__":
    main()
