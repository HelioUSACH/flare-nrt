import datetime as dt

import pytest
from conftest import T0, write_fits

from flare_nrt.grid import CHANNELS, file_name
from flare_nrt.qc import DEFAULT_IGNORE_MASK, QualityControl, soft_flags


def _slot(tmp_path, slot, bad_wave=None, **bad):
    paths = {}
    for w in CHANNELS:
        p = tmp_path / file_name(slot, w)
        write_fits(p, w, slot, **(bad if w == bad_wave else {}))
        paths[w] = p
    return paths


def test_good_slot_passes(tmp_path):
    q = QualityControl().check_slot(T0, _slot(tmp_path, T0))
    assert q.ok and q.reasons == [] and set(q.frames) == set(CHANNELS)


def test_daily_synoptic_artifact_is_rejected(tmp_path):
    q = QualityControl().check_slot(T0, _slot(tmp_path, T0, bad_wave=171, quality=65536, exptime=0))
    assert not q.ok
    assert "quality:171=65536" in q.reasons and "exptime:171" in q.reasons
    assert q.frames == {}


def test_blank_frame_is_rejected(tmp_path):
    q = QualityControl().check_slot(T0, _slot(tmp_path, T0, bad_wave=304, blank=True))
    assert not q.ok and "empty:304" in q.reasons


def test_wrong_time_is_rejected(tmp_path):
    paths = _slot(tmp_path, T0)
    write_fits(paths[131], 131, T0 + dt.timedelta(minutes=5))
    q = QualityControl().check_slot(T0, paths)
    assert "time_offset:131" in q.reasons and "channel_spread" in q.reasons


def test_median_jump_against_recent_reference(tmp_path):
    qc = QualityControl()
    for i in range(6):
        s = T0 + dt.timedelta(minutes=2 * i)
        assert qc.check_slot(s, _slot(tmp_path / str(i), s)).ok
    s = T0 + dt.timedelta(minutes=12)
    paths = _slot(tmp_path / "x", s)
    write_fits(paths[1600], 1600, s, exptime=2.9, flare=0)
    from astropy.io import fits
    with fits.open(paths[1600], mode="update") as h:
        h[1].data = h[1].data * 10
    q = qc.check_slot(s, paths)
    assert "median_jump:1600" in q.reasons


def test_short_exposure_during_flare_is_not_rejected(tmp_path):
    """AIA's exposure control: 131 Å at 2 % of its usual exposure, counts
    scaled the same way (raw disk median rounds to ~0). Must pass."""
    qc = QualityControl()
    for i in range(6):
        s = T0 + dt.timedelta(minutes=2 * i)
        assert qc.check_slot(s, _slot(tmp_path / str(i), s)).ok
    s = T0 + dt.timedelta(minutes=12)
    paths = _slot(tmp_path / "flare", s)
    from astropy.io import fits
    with fits.open(paths[131], mode="update") as h:
        h[1].data = h[1].data * 0.02
        h[1].header["EXPTIME"] = 2.9 * 0.02
    q = qc.check_slot(s, paths)
    assert q.ok, q.reasons


def test_raw_dimming_at_constant_exposure_is_still_caught(tmp_path):
    qc = QualityControl()
    for i in range(6):
        s = T0 + dt.timedelta(minutes=2 * i)
        assert qc.check_slot(s, _slot(tmp_path / str(i), s)).ok
    s = T0 + dt.timedelta(minutes=12)
    paths = _slot(tmp_path / "dim", s)
    from astropy.io import fits
    with fits.open(paths[171], mode="update") as h:
        h[1].data = h[1].data * 0.05
    assert "median_jump:171" in qc.check_slot(s, paths).reasons


def test_quicklook_bit_alone_is_not_a_reject(tmp_path):
    q = QualityControl().check_slot(T0, _slot(tmp_path, T0, bad_wave=171, quality=1 << 30))
    assert q.ok, q.reasons
    q = QualityControl().check_slot(T0, _slot(tmp_path / "b", T0, bad_wave=171, quality=(1 << 30) | 65536))
    assert f"quality:171={(1 << 30) | 65536}" in q.reasons


# 1.3.1: QUALITY as JSOC's near-real-time synoptic headers carry it
# (bit 30 = near-real-time; bit 2 = Q_NOASD, every NRT frame since 2026-11-24 23:00 UT)

def _all(tmp_path, slot, quality):
    paths = {}
    for w in CHANNELS:
        p = tmp_path / file_name(slot, w)
        write_fits(p, w, slot, quality=quality)
        paths[w] = p
    return paths


@pytest.mark.parametrize("quality, ok, soft", [
    (0x40000000, True, []),              # before 2026-11-24 23:00 UT
    (0x40000004, True, ["noasd"]),       # since then
    (0x00000004, True, ["noasd"]),
    (0x40200000, False, []),             # spacecraft flags of July 2026 (bit 21) + NRT
    (0x40010000, False, []),             # the daily 21:00 UT frame (bit 16) + NRT
    (0x40010004, False, ["noasd"]),      # a hard bit still rejects with a soft one
    (0x80000000, False, []),             # bit 31 (as an unsigned value) …
    (-0x80000000, False, []),            # … and as a signed 32-bit header value
])
def test_quality_bits_of_real_headers(tmp_path, quality, ok, soft):
    q = QualityControl().check_slot(T0, _all(tmp_path, T0, quality))
    assert q.ok is ok, q.reasons
    assert q.quality == {w: quality for w in CHANNELS}
    assert q.quality_soft == soft
    if not ok:
        assert f"quality:131={quality}" in q.reasons


def test_ignored_mask_is_a_setting(tmp_path, monkeypatch):
    from flare_nrt.config import Settings

    assert DEFAULT_IGNORE_MASK == Settings().qc_ignore_quality_mask == 0x40000004
    monkeypatch.setenv("FLARE_NRT_QC_IGNORE_QUALITY_MASK", "0x40000000")     # 1.0.0–1.3.0 behaviour
    assert Settings().qc_ignore_quality_mask == 0x40000000
    monkeypatch.setenv("FLARE_NRT_QC_IGNORE_QUALITY_MASK", "1073741828")     # decimal works too
    assert Settings().qc_ignore_quality_mask == 0x40000004
    q = QualityControl(ignore_mask=0x40000000).check_slot(T0, _all(tmp_path, T0, 0x40000004))
    assert not q.ok and "quality:131=1073741828" in q.reasons


def test_soft_flag_names():
    assert soft_flags(0x40000004, DEFAULT_IGNORE_MASK) == ["noasd"]
    assert soft_flags(0x40000000, DEFAULT_IGNORE_MASK) == []              # bit 30 is on every NRT frame
    assert soft_flags(0x40000104, DEFAULT_IGNORE_MASK | 0x100) == ["noasd", "bit8"]
