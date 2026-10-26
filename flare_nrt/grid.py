"""The 2-minute AIA synoptic grid: slot times, file names, archive URLs.

All datetimes in this package are timezone-aware UTC."""

from __future__ import annotations

import datetime as dt
import re

CHANNELS = (131, 171, 304, 1600)
CADENCE = dt.timedelta(minutes=2)
FILE_RE = re.compile(r"AIA(\d{8})_(\d{4})_(\d{4})\.fits$")


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def as_utc(t: dt.datetime) -> dt.datetime:
    return t.replace(tzinfo=dt.UTC) if t.tzinfo is None else t.astimezone(dt.UTC)


def floor_slot(t: dt.datetime) -> dt.datetime:
    t = as_utc(t)
    return t.replace(minute=t.minute - t.minute % 2, second=0, microsecond=0)


def iso(t: dt.datetime | None) -> str | None:
    """ISO 8601 UTC with Z, to the second."""
    if t is None:
        return None
    return as_utc(t).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(s: str) -> dt.datetime:
    return as_utc(dt.datetime.fromisoformat(s.replace("Z", "+00:00")))


def file_name(slot: dt.datetime, wave: int) -> str:
    return f"AIA{slot:%Y%m%d}_{slot:%H%M}_{wave:04d}.fits"


def file_url(base: str, slot: dt.datetime, wave: int) -> str:
    return f"{base}/{slot:%Y/%m/%d}/H{slot:%H}00/{file_name(slot, wave)}"


def parse_file_name(name: str) -> tuple[dt.datetime, int] | None:
    m = FILE_RE.search(name)
    if not m:
        return None
    t = dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M").replace(tzinfo=dt.UTC)
    return t, int(m.group(3))


def day_key(t: dt.datetime) -> str:
    return f"{as_utc(t):%Y%m%d}"
