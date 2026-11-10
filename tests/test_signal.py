"""What the public sees: the state machine (quiet → onset → active → decay
→ quiet), how a long plateau behaves, and the input checks that decide
whether a frame is used at all (checklist C.13, E.22)."""

import datetime as dt

from conftest import T0, write_slot

from flare_nrt.detector import EventTracker
from flare_nrt.grid import CADENCE, iso
from flare_nrt.loop import Processor, input_age_s, signal_state
from flare_nrt.model import Ensemble
from flare_nrt.qc import QualityControl
from flare_nrt.source import ReplaySource

THR = 0.8


def _run(probs, gap_min=10):
    """Feeds the tracker one probability per 2-min slot (None = no result);
    returns [(state, flare_detected, event_id)] per slot and the closed events."""
    tr = EventTracker(gap_min)
    out, closed = [], []
    for i, p in enumerate(probs):
        alert = p is not None and p >= THR
        ev, opened, cl = tr.update(T0 + i * CADENCE, alert, p or 0.0)
        if cl:
            closed.append(cl)
        out.append((signal_state(opened, alert, tr.current is not None), tr.current is not None, ev))
    return out, closed


def test_onset_active_decay_quiet():
    probs = [0.1, 0.2, 0.85, 0.95, 0.9, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1]
    out, closed = _run(probs)
    states = [s for s, _, _ in out]
    assert states[:2] == ["quiet", "quiet"]
    assert states[2] == "onset" and states[3:5] == ["active", "active"]
    # no alert after slot 4 (t+8 min): decay while within the 10-min merge gap …
    assert states[5:10] == ["decay"] * 5
    # … and the event closes on the first slot more than 10 min after the last alert
    assert states[10] == "quiet" and len(closed) == 1
    ev = closed[0]
    assert (ev.t_start, ev.t_pmax, ev.t_end) == (iso(T0 + 2 * CADENCE), iso(T0 + 3 * CADENCE), iso(T0 + 4 * CADENCE))
    assert ev.p_max == 0.95 and ev.n_alert_frames == 3
    # flare_detected is true from onset through decay, never in quiet
    assert [f for _, f, _ in out] == [s != "quiet" for s in states]


def test_probability_one_with_no_detection_cannot_happen():
    """A high probability always opens (or continues) an event: the
    contract's 'probability 1.0 and flare_detected false' case can't occur."""
    out, _ = _run([0.1, 1.0, 0.1])
    assert out[1][:2] == ("onset", True)


def test_plateau_is_one_event_then_rolls_off():
    """A long, moderate event: the probability stays above threshold for
    30 min. It is one event (no re-onset), active throughout; the rolling
    baseline (t−8…−12) re-anchors on the plateau by construction, so the
    probability falls back and the state goes decay → quiet without any
    extra rule."""
    probs = [0.1] + [0.85] * 15 + [0.3] * 7
    out, closed = _run(probs)
    states = [s for s, _, _ in out]
    assert states.count("onset") == 1 and states[2:16] == ["active"] * 14
    assert {e for _, _, e in out if e} == {1}
    assert states[16:21] == ["decay"] * 5 and states[21] == "quiet"
    assert closed[0].n_alert_frames == 15


def test_alerts_more_than_gap_apart_are_two_onsets():
    out, closed = _run([0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9])
    assert [s for s, _, _ in out].count("onset") == 2 and len(closed) == 1


def test_input_age_uses_the_newest_channel():
    now = dt.datetime(2026, 10, 28, 16, 0, tzinfo=dt.timezone.utc)
    inputs = {"131": "2026-10-28T15:44:00.000Z", "171": "2026-10-28T15:45:00.000Z"}
    assert input_age_s(inputs, now) == 900.0
    assert input_age_s({}, now) is None


def test_live_frame_with_old_inputs_is_skipped_and_says_why(cfg):
    for i in range(7):
        write_slot(cfg.staging_dir, T0 + i * CADENCE)
    src = ReplaySource(cfg.staging_dir)
    ens = Ensemble.load(cfg.weights, cfg.weights_sha256)
    proc = Processor(cfg, ens, src, live=True)
    proc.artifacts.enabled = False
    for i in range(6):
        proc.current(T0 + i * CADENCE)
    late = T0 + 6 * CADENCE
    row = proc.process(late, late + dt.timedelta(minutes=16))
    assert row["status"] == "inputs_old" and row["reason"] == "inputs_old" and row["input_age_s"] > 900
    assert "probability" not in row and row["state"] == "quiet"
    row = proc.process(late, late + dt.timedelta(minutes=5))
    assert row["status"] == "ok" and row["input_age_s"] < 900
    assert row["raw_score"] == row["probability"]


def test_channels_far_apart_are_not_one_snapshot(tmp_path):
    from conftest import write_fits

    from flare_nrt.grid import file_name

    paths = {}
    for w, dt_s in ((131, 0), (171, 10), (304, 0), (1600, 0)):
        t = T0 + dt.timedelta(seconds=dt_s)
        p = tmp_path / file_name(T0, w)
        write_fits(p, w, t)
        paths[w] = p
    assert QualityControl().check_slot(T0, paths).ok
    # 171 Å taken 85 s after the others, 304 Å 85 s before: 170 s apart > 120 s
    for w, dt_s in ((171, 85), (304, -85)):
        write_fits(paths[w], w, T0 + dt.timedelta(seconds=dt_s))
    q = QualityControl().check_slot(T0, paths)
    assert not q.ok and "channel_spread" in q.reasons
