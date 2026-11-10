import datetime as dt

import torch

from flare_nrt.detector import EventTracker, FrameBuffer, baseline_slots, is_alert

T = dt.datetime(2026, 10, 20, 12, 0, tzinfo=dt.timezone.utc)
M = dt.timedelta(minutes=1)


def test_baseline_slots_lag8():
    assert baseline_slots(T, 8, 3) == [T - 8 * M, T - 10 * M, T - 12 * M]


def test_threshold_is_inclusive():
    assert is_alert(0.80, 0.80)
    assert not is_alert(0.7999, 0.80)


def test_buffer_evicts_old_slots_and_keeps_rejections():
    b = FrameBuffer(keep=dt.timedelta(minutes=6))
    b.put(T, torch.zeros(1))
    b.put(T + 2 * M, None)
    assert T + 2 * M in b and b.get(T + 2 * M) is None
    b.put(T + 8 * M, torch.zeros(1))
    assert T not in b


def test_alerts_within_gap_merge_into_one_event():
    tr = EventTracker(merge_gap_min=10)
    eid, opened, _ = tr.update(T, True, 0.9)
    assert (eid, opened) == (1, True)
    tr.update(T + 2 * M, False, 0.3)
    eid, opened, closed = tr.update(T + 8 * M, True, 0.95)
    assert (eid, opened, closed) == (1, False, None)
    assert tr.current.p_max == 0.95 and tr.current.n_alert_frames == 2


def test_event_closes_after_gap_and_next_alert_is_new():
    tr = EventTracker(merge_gap_min=10)
    tr.update(T, True, 0.9)
    _, _, closed = tr.update(T + 12 * M, False, 0.1)
    assert closed.event_id == 1 and closed.closed
    eid, opened, _ = tr.update(T + 14 * M, True, 0.85)
    assert (eid, opened) == (2, True)


def test_data_gap_longer_than_merge_gap_starts_new_event():
    tr = EventTracker(merge_gap_min=10)
    tr.update(T, True, 0.9)
    eid, opened, closed = tr.update(T + 30 * M, True, 0.9)
    assert closed.event_id == 1 and (eid, opened) == (2, True)


def test_tracker_state_roundtrip():
    tr = EventTracker(merge_gap_min=10)
    tr.update(T, True, 0.9)
    st = tr.state()
    tr2 = EventTracker(10, st["last_id"], st["current"])
    eid, opened, _ = tr2.update(T + 2 * M, True, 0.91)
    assert (eid, opened) == (1, False)


def test_parse_datetime_provider_spellings():
    from flare_nrt.grid import parse_datetime
    ref = dt.datetime(2026, 6, 20, 9, 22, 5, 120000, tzinfo=dt.timezone.utc)
    for s in ("2026-06-20T09:22:05.12Z", "2026-06-20T09:22:05.12", "2026-06-20 09:22:05.120000",
              "2026-06-20T06:22:05.12-03:00", "2026-06-20T09:22:05.12+0000"):
        assert parse_datetime(s) == ref, s
    assert parse_datetime("2026-06-20T09:22Z") == dt.datetime(2026, 6, 20, 9, 22, tzinfo=dt.timezone.utc)
