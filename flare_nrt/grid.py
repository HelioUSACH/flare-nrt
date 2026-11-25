"""The 2-minute AIA synoptic grid: slot times, file names, archive URLs.

All datetimes in this package are timezone-aware UTC."""

from __future__ import annotations

import datetime as dt
import re

CHANNELS = (131, 171, 304, 1600)
CADENCE = dt.timedelta(minutes=2)
FILE_RE = re.compile(r"AIA(\d{8})_(\d{4})_(\d{4})\.fits$")


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def as_utc(t: dt.datetime) -> dt.datetime:
    return t.replace(tzinfo=dt.timezone.utc) if t.tzinfo is None else t.astimezone(dt.timezone.utc)


def floor_slot(t: dt.datetime) -> dt.datetime:
    t = as_utc(t)
    return t.replace(minute=t.minute - t.minute % 2, second=0, microsecond=0)


def iso_ms(t: dt.datetime | None) -> str | None:
    """ISO 8601 UTC with Z, to the millisecond (provider timestamps)."""
    if t is None:
        return None
    t = as_utc(t)
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def iso(t: dt.datetime | None) -> str | None:
    """ISO 8601 UTC with Z, to the second."""
    if t is None:
        return None
    return as_utc(t).strftime("%Y-%m-%dT%H:%M:%SZ")


_ISO_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}(?::\d{2})?)(?:\.(\d+))?\s*(Z|[+-]\d{2}:?\d{2})?$")


def parse_datetime(s: str) -> dt.datetime:
    """ISO 8601 → aware UTC, for any provider spelling: 'Z' or an offset or
    none (= UTC), 'T' or space, any number of fractional digits (AIA's T_OBS
    has 2, Python 3.10's fromisoformat only takes 3 or 6)."""
    m = _ISO_RE.match(str(s).strip())
    if not m:
        raise ValueError(f"not an ISO 8601 time: {s!r}")
    date, clock, frac, tz = m.groups()
    if clock.count(":") == 1:
        clock += ":00"
    text = f"{date}T{clock}"
    if frac:
        text += "." + (frac + "000000")[:6]
    if tz and tz != "Z":
        text += tz if ":" in tz else f"{tz[:3]}:{tz[3:]}"
    return as_utc(dt.datetime.fromisoformat(text))


def parse_iso(s: str) -> dt.datetime:
    return parse_datetime(s)


def file_name(slot: dt.datetime, wave: int) -> str:
    return f"AIA{slot:%Y%m%d}_{slot:%H%M}_{wave:04d}.fits"


def file_url(base: str, slot: dt.datetime, wave: int) -> str:
    return f"{base}/{slot:%Y/%m/%d}/H{slot:%H}00/{file_name(slot, wave)}"


# 1.3.1: JSOC's near-real-time archive, current to ~20 min (the dated one above
# lags weeks). File names carry seconds; its cadence is 3 min, not 2.
NRT_FILE_RE = re.compile(r"AIA(\d{8})_(\d{6})_(\d{4})\.fits")


def nrt_dir_url(base: str, t: dt.datetime) -> str:
    """``{base}/nrt/YYYY/MM/DD/HHH00/``: one hour of near-real-time images."""
    return f"{base}/nrt/{t:%Y/%m/%d}/H{t:%H}00/"


def nrt_file_name(t: dt.datetime, wave: int) -> str:
    return f"AIA{t:%Y%m%d_%H%M%S}_{wave:04d}.fits"


def parse_nrt_listing(html: str) -> list[tuple[dt.datetime, int, str]]:
    """(image time, channel, file name) of each image named in a directory listing."""
    out = {}
    for m in NRT_FILE_RE.finditer(html):
        t = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
        out[m.group(0)] = (t, int(m.group(3)), m.group(0))
    return sorted(out.values())


def parse_file_name(name: str) -> tuple[dt.datetime, int] | None:
    m = FILE_RE.search(name)
    if not m:
        return None
    t = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M").replace(tzinfo=dt.timezone.utc)
    return t, int(m.group(3))


def day_key(t: dt.datetime) -> str:
    return f"{as_utc(t):%Y%m%d}"
