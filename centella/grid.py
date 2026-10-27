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


def parse_file_name(name: str) -> tuple[dt.datetime, int] | None:
    m = FILE_RE.search(name)
    if not m:
        return None
    t = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M").replace(tzinfo=dt.timezone.utc)
    return t, int(m.group(3))


def day_key(t: dt.datetime) -> str:
    return f"{as_utc(t):%Y%m%d}"
