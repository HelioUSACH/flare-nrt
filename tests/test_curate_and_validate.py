import datetime as dt
import json

from flare_nrt.curate import label, merge, negative_blocks
from flare_nrt.goes import Flare, class_flux, parse_swpc
from flare_nrt.validate import evaluate

D = dt.datetime(2026, 7, 1, tzinfo=dt.UTC)
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
