import datetime as dt

from conftest import T0, write_fits

from flare_nrt.grid import CHANNELS, file_name
from flare_nrt.qc import QualityControl


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
