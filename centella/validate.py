"""Operational validation against GOES, same rules as the calibration
(docs/model-card.md):

- an M/X flare is **detected** if any slot in [tstart − 10 min, tend + 5 min]
  is an alert; detection time = first such alert;
- an alert run (merged with ``merge_gap``) is a **false alarm** if it overlaps
  no C-or-larger flare's [tstart − 10 min, tend + 30 min]; runs on C flares
  are counted apart (neither hit nor false alarm);
- a flare is **evaluable** if at least 2 ``ok`` slots fall in
  [tstart − 10 min, tpeak].

    python -m centella.validate --start 2026-10-27 --end 2026-11-03 [--flares-csv F.csv]

The report also scores the named baseline, ``aia131_rise``: no network, the
same input frames — alert when the disk-integrated 131 Å count rate is up by
≥ k over the same baseline frames (``aia131_excess`` in each result row),
with k the lowest value whose false alarms on the period do not exceed
Centella's. Same rules, same flares: the difference is what the network adds.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

from .config import Settings
from .curate import alert_runs
from .goes import Flare, class_flux, fetch_swpc, read_csv
from .grid import iso, parse_iso
from .store import ResultStore

PRE = dt.timedelta(minutes=10)
POST_HIT = dt.timedelta(minutes=5)
POST_FA = dt.timedelta(minutes=30)


def evaluate(rows: list[dict], flares: list[Flare], merge_gap_min: int = 10) -> dict:
    ok = [parse_iso(r["slot"]) for r in rows if r.get("status") == "ok"]
    alerts = sorted(parse_iso(r["slot"]) for r in rows if r.get("alert"))
    runs = alert_runs([r for r in rows if r.get("alert")], dt.timedelta(minutes=merge_gap_min))
    big = [f for f in flares if f.flux >= class_flux("M1.0")]
    c_or_more = [f for f in flares if f.flux >= class_flux("C1.0")]

    per_flare = []
    for f in big:
        evaluable = sum(1 for t in ok if f.tstart - PRE <= t <= f.tpeak) >= 2
        if not evaluable:
            continue
        hits = [t for t in alerts if f.tstart - PRE <= t <= f.tend + POST_HIT]
        ongoing = any(f.tstart - PRE - dt.timedelta(minutes=4) <= t < f.tstart - PRE for t in alerts)
        t = hits[0] if hits else None
        per_flare.append({
            "flare": f.flare_id, "class": f.fclass, "detected": bool(hits), "first_alert": iso(t),
            "min_from_start": (t - f.tstart).total_seconds() / 60 if t else None,
            "min_from_peak": (t - f.tpeak).total_seconds() / 60 if t else None,
            "already_alerting": ongoing,
        })

    def overlaps(run, fl, post):
        return any(f.tstart - PRE <= run[1] and f.tend + post >= run[0] for f in fl)

    n_mx_runs = sum(overlaps(r, big, POST_FA) for r in runs)
    n_c_runs = sum(overlaps(r, c_or_more, POST_FA) and not overlaps(r, big, POST_FA) for r in runs)
    fa = [r for r in runs if not overlaps(r, c_or_more, POST_FA)]
    det = [p for p in per_flare if p["detected"]]
    fresh = [p for p in det if not p["already_alerting"]]
    span_days = (max(ok) - min(ok)).total_seconds() / 86400 if len(ok) > 1 else 0
    return {
        "slots_ok": len(ok),
        "span_days": round(span_days, 2),
        "mx_evaluable": len(per_flare),
        "mx_detected": len(det),
        "pod_mx": round(len(det) / len(per_flare), 3) if per_flare else None,
        "pod_ge_m2": _pod([p for p in per_flare if class_flux(p["class"]) >= class_flux("M2.0")]),
        "pod_m1_m2": _pod([p for p in per_flare if class_flux(p["class"]) < class_flux("M2.0")]),
        "alert_runs": len(runs),
        "runs_on_mx": n_mx_runs,
        "runs_on_c": n_c_runs,
        "false_alarms": len(fa),
        "false_alarms_per_day": round(len(fa) / span_days, 3) if span_days else None,
        "median_min_from_start": _med([p["min_from_start"] for p in fresh]),
        "median_min_from_peak": _med([p["min_from_peak"] for p in fresh]),
        "false_alarm_runs": [[iso(a), iso(b)] for a, b in fa],
        "flares": per_flare,
    }


def aia131_rule(rows: list[dict], flares: list[Flare], max_false_alarms: int, merge_gap_min: int = 10) -> dict:
    """The baseline at matched false alarms (see the module docstring)."""
    xs = sorted({r["aia131_excess"] for r in rows if r.get("status") == "ok" and r.get("aia131_excess") is not None},
                reverse=True)
    if not xs:
        return {"rule": "aia131_rise", "available": False,
                "why": "no aia131_excess in these rows (results from before 2.1.0: replay again)"}
    best = None
    lo, hi = 0, min(len(xs), 5000) - 1      # FA grows as k falls: binary search on the sorted values
    while lo <= hi:
        mid = (lo + hi) // 2
        k = xs[mid]
        rep = evaluate([dict(r, alert=r.get("status") == "ok" and (r.get("aia131_excess") or -1) >= k) for r in rows],
                       flares, merge_gap_min)
        if rep["false_alarms"] <= max_false_alarms:
            best, lo = (k, rep), mid + 1
        else:
            hi = mid - 1
    if best is None:
        return {"rule": "aia131_rise", "available": True, "k": None, "why": "no k reaches that false-alarm count"}
    k, rep = best
    return {"rule": "aia131_rise", "available": True, "k": k,
            **{key: rep[key] for key in ("mx_evaluable", "mx_detected", "pod_mx", "pod_ge_m2", "pod_m1_m2",
                                         "false_alarms", "median_min_from_start", "median_min_from_peak")}}


def alternation(rows: list[dict], p_max: float = 0.3, min_step: float = 0.02) -> dict:
    """The low-probability 2-frame alternation seen in 1.x: among
    consecutive ``ok`` slots below ``p_max``, the fraction of steps that
    reverse the previous step's sign with both steps ≥ ``min_step``. A
    random walk gives ~0.5 of reversals overall; a strict up-down-up
    alternation gives ~1."""
    ok = sorted((parse_iso(r["slot"]), r["probability"]) for r in rows
                if r.get("status") == "ok" and r.get("probability") is not None)
    steps = []
    for (t0, p0), (t1, p1) in zip(ok, ok[1:], strict=False):
        if t1 - t0 == dt.timedelta(minutes=2) and p0 < p_max and p1 < p_max:
            steps.append((t1, p1 - p0))
    n = rev = 0
    for (ta, a), (tb, b) in zip(steps, steps[1:], strict=False):
        if tb - ta == dt.timedelta(minutes=2) and abs(a) >= min_step and abs(b) >= min_step:
            n += 1
            rev += a * b < 0
    return {"pairs": n, "reversal_fraction": round(rev / n, 3) if n else None,
            "p_max": p_max, "min_step": min_step}


def _pod(ps):
    return round(sum(p["detected"] for p in ps) / len(ps), 3) if ps else None


def _med(xs):
    return statistics.median(xs) if xs else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True, help="inclusive")
    ap.add_argument("--flares-csv")
    ap.add_argument("--out", help="write the JSON report here")
    a = ap.parse_args(argv)
    cfg = Settings()
    d0 = dt.datetime.strptime(a.start, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    d1 = dt.datetime.strptime(a.end, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
    store = ResultStore(cfg.results_dir)
    rows, d = [], d0
    while d <= d1:
        rows += store.read_day(f"{d:%Y%m%d}")
        d += dt.timedelta(days=1)
    flares = read_csv(Path(a.flares_csv)) if a.flares_csv else fetch_swpc(cfg.goes_url)
    rep = evaluate(rows, flares, cfg.merge_gap_min)
    rep["baseline"] = aia131_rule(rows, flares, rep["false_alarms"], cfg.merge_gap_min)
    rep["alternation"] = alternation(rows)
    txt = json.dumps(rep, indent=2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(txt)
    summary = {k: v for k, v in rep.items() if k not in ("flares", "false_alarm_runs")}
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
