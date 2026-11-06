import datetime as dt
import json

from centella.curate import label, merge, negative_blocks
from centella.goes import Flare, class_flux, parse_swpc
from centella.validate import evaluate

D = dt.datetime(2026, 7, 1, tzinfo=dt.timezone.utc)
M = dt.timedelta(minutes=1)


def fl(start_min, peak_min, end_min, cls):
    return Flare(D + start_min * M, D + peak_min * M, D + end_min * M, cls)


def test_class_flux():
    assert class_flux("M2.5") == 2.5e-5 and class_flux("X1") == 1e-4 and class_flux("?") == 0


def test_parse_swpc_fixture():
    # Synthetic payload in SWPC's xray-flares-7-day.json shape (not real data).
    payload = json.loads(open("tests/fixtures/swpc_xray_flares_synthetic.json").read())
    flares = parse_swpc(payload)
    assert [f.fclass for f in flares] == ["C3.4", "M1.2"]
    assert flares[1].tend > flares[1].tpeak  # "end_time": null → mirrored rise


def test_merge_intervals():
    a = [(D, D + 10 * M), (D + 5 * M, D + 20 * M), (D + 30 * M, D + 40 * M)]
    assert merge(a) == [(D, D + 20 * M), (D + 30 * M, D + 40 * M)]


def test_label_phases_pick_biggest_flare():
    f1, f2 = fl(100, 110, 130, "C2.0"), fl(105, 115, 140, "M1.0")
    pre, post = 45 * M, 15 * M
    assert label(D + 50 * M, [f1, f2], pre, post)["phase"] == "none"
    lab = label(D + 108 * M, [f1, f2], pre, post)
    assert lab["flare_class"] == "M1.0" and lab["phase"] == "rise" and lab["min_from_start"] == 3.0
    assert label(D + 150 * M, [f1, f2], pre, post)["phase"] == "post"


def test_negative_blocks_deterministic_and_clear_of_flares():
    forbidden = [(D + 300 * M, D + 400 * M)]
    have = {D + 2 * i * M for i in range(720)}
    b1 = negative_blocks(D, forbidden, have, 4, 60)
    assert b1 == negative_blocks(D, forbidden, have, 4, 60)
    assert len(b1) == 4
    for a, b in b1:
        assert b < D + 240 * M or a > D + 460 * M
    for (_a1, b1_), (a2, _) in zip(b1, b1[1:], strict=False):
        assert b1_ < a2


def _rows(alert_slots, n=720):
    return [{"slot": (D + 2 * i * M).strftime("%Y-%m-%dT%H:%M:%SZ"), "status": "ok",
             "alert": (2 * i) in alert_slots} for i in range(n)]


def test_evaluate_counts_hits_c_runs_and_false_alarms():
    flares = [fl(100, 110, 120, "M1.5"), fl(400, 405, 410, "C3.0")]
    rows = _rows({108, 110, 404, 900})
    rep = evaluate(rows, flares)
    assert rep["mx_evaluable"] == 1 and rep["mx_detected"] == 1
    assert rep["runs_on_mx"] == 1 and rep["runs_on_c"] == 1 and rep["false_alarms"] == 1
    assert rep["flares"][0]["min_from_start"] == 8.0 and rep["flares"][0]["min_from_peak"] == -2.0


def test_baseline_rule_at_matched_false_alarms():
    from centella.validate import aia131_rule

    # 2 h of slots; an M flare at 60–70 min. Centella: alert on it only.
    # The 131 Å excess: high on the flare, but also a bump at 20 min (no flare).
    rows = []
    for i in range(60):
        t = D + 2 * i * M
        m = 2 * i
        ex = 0.5 if 60 <= m <= 66 else (0.3 if m == 20 else 0.01)
        rows.append({"slot": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "status": "ok", "alert": 60 <= m <= 64,
                     "probability": 0.9 if 60 <= m <= 64 else 0.1, "aia131_excess": ex})
    flares = [fl(58, 62, 70, "M2.0")]
    rep = evaluate(rows, flares)
    assert rep["mx_detected"] == 1 and rep["false_alarms"] == 0
    b = aia131_rule(rows, flares, rep["false_alarms"])
    # with 0 false alarms allowed, k must sit above the 0.3 bump: it still catches the flare
    assert b["available"] and b["k"] > 0.3 and b["false_alarms"] == 0 and b["mx_detected"] == 1
    old_rows = [{k: v for k, v in r.items() if k != "aia131_excess"} for r in rows]
    assert aia131_rule(old_rows, flares, 0)["available"] is False


def test_alternation_index():
    from centella.validate import alternation

    def rows(ps):
        return [{"slot": (D + 2 * i * M).strftime("%Y-%m-%dT%H:%M:%SZ"), "status": "ok", "probability": p}
                for i, p in enumerate(ps)]

    assert alternation(rows([0.05, 0.15] * 10))["reversal_fraction"] == 1.0
    assert alternation(rows([0.05 + 0.01 * i for i in range(20)]))["pairs"] == 0      # steps < 0.02
    assert alternation(rows([0.05 + 0.03 * i for i in range(8)]))["reversal_fraction"] == 0.0


def test_hek_flare_list(monkeypatch, tmp_path):
    """HEK's column search, paged (``overmax``); entries without a class dropped."""
    import requests

    from centella import goes

    pages = [
        {"overmax": True, "result": [
            {"event_starttime": "2026-07-04T11:38:00", "event_peaktime": "2026-07-04T11:52:00",
             "event_endtime": "2026-07-04T12:10:00", "fl_goescls": "M2.3"},
            {"event_starttime": "2026-07-04T10:20:00", "event_peaktime": "2026-07-04T10:26:00",
             "event_endtime": "2026-07-04T10:34:00", "fl_goescls": ""}]},
        {"overmax": False, "result": [
            {"event_starttime": "2026-07-05T01:00:00", "event_peaktime": "2026-07-05T01:04:00",
             "event_endtime": "2026-07-05T01:09:00", "fl_goescls": "C1.1"}]},
    ]
    seen = []

    class R:
        def __init__(self, js):
            self.js = js

        def raise_for_status(self):
            pass

        def json(self):
            return self.js

    def get(url, params, **kw):
        seen.append((params["value0"], params["page"]))
        return R(pages[params["page"] - 1] if params["value0"] == "SWPC" else {"result": []})

    monkeypatch.setattr(requests, "get", get)
    out = tmp_path / "f.csv"
    goes.main(["--start", "2026-07-04", "--end", "2026-07-05", "--out", str(out)])
    assert seen == [("SWPC", 1), ("SWPC", 2), ("SSW Latest Events", 1)]
    assert [f.fclass for f in goes.read_csv(out)] == ["M2.3", "C1.1"]



def test_hek_positions_come_from_lmsal_matched_by_peak(monkeypatch, tmp_path):
    import requests

    from centella import goes

    swpc = [  # SWPC: class and times; one with its own position, one with (0, 0) = none
        {"event_starttime": "2026-07-04T11:38:00", "event_peaktime": "2026-07-04T11:52:00",
         "event_endtime": "2026-07-04T12:10:00", "fl_goescls": "M2.3", "hpc_x": 0, "hpc_y": 0},
        {"event_starttime": "2026-07-04T15:00:00", "event_peaktime": "2026-07-04T15:05:00",
         "event_endtime": "2026-07-04T15:12:00", "fl_goescls": "M1.0", "hpc_x": -310.0, "hpc_y": 120.0},
        {"event_starttime": "2026-07-04T18:00:00", "event_peaktime": "2026-07-04T18:05:00",
         "event_endtime": "2026-07-04T18:12:00", "fl_goescls": "C2.0", "hpc_x": 0, "hpc_y": 0}]
    lmsal = [  # LMSAL: positions from AIA; the M2.3 peaks 3 min off, a C nearby must not be taken
        {"event_starttime": "2026-07-04T11:40:00", "event_peaktime": "2026-07-04T11:55:00",
         "event_endtime": "2026-07-04T12:05:00", "fl_goescls": "M2.2", "hpc_x": 854.2, "hpc_y": 303.4},
        {"event_starttime": "2026-07-04T11:45:00", "event_peaktime": "2026-07-04T11:51:00",
         "event_endtime": "2026-07-04T11:58:00", "fl_goescls": "C5.0", "hpc_x": -500.0, "hpc_y": -500.0}]

    class R:
        def __init__(self, js):
            self.js = js

        def raise_for_status(self):
            pass

        def json(self):
            return self.js

    monkeypatch.setattr(requests, "get",
                        lambda url, params, **kw: R({"result": swpc if params["value0"] == "SWPC" else lmsal}))
    fl = {f.fclass: f for f in goes.fetch_hek(goes.dt.datetime(2026, 7, 4, tzinfo=goes.dt.timezone.utc),
                                               goes.dt.datetime(2026, 7, 5, tzinfo=goes.dt.timezone.utc))}
    assert fl["M2.3"].hpc == (854.2, 303.4)           # LMSAL, same letter, nearest peak
    assert fl["M1.0"].hpc == (-310.0, 120.0)           # no LMSAL match: SWPC's own
    assert fl["C2.0"].hpc is None                      # (0, 0) means no position
    out = tmp_path / "f.csv"
    goes.write_csv(out, list(fl.values()))
    back = {f.fclass: f for f in goes.read_csv(out)}
    assert back["M2.3"].hpc == (854.2, 303.4) and back["C2.0"].hpc is None
    assert back["M2.3"] == fl["M2.3"]                   # position is not part of identity


def test_location_errors_against_catalogue():
    from dataclasses import replace

    from centella.validate import location_errors

    f1 = replace(fl(60, 64, 70, "M2.0"), hpc=(500.0, -200.0))
    f2 = replace(fl(200, 204, 210, "M1.0"), hpc=None)
    rows = []
    for m, p, loc in ((60, 0.85, [560, -200]), (62, 0.97, [520, -210]), (202, 0.9, [0, 0])):
        rows.append({"slot": (D + m * M).strftime("%Y-%m-%dT%H:%M:%SZ"), "status": "ok", "alert": True,
                     "probability": p, "location": {"centroid_arcsec": loc, "on_limb": False}})
    r = location_errors(rows, [f1, f2])
    assert r["n_compared"] == 1 and r["mx_detected_without_position"] == 1
    assert r["flares"][0]["centella"] == [520, -210]           # the highest-probability alert
    assert abs(r["median_arcsec"] - 22.4) < 0.1 and r["within_60_arcsec"] == 1.0
