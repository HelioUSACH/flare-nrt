## Summary

This branch moves the `usach.flare.nrt` product to Centella (flare-nrt ≥ 2.0) and gives it its own page. Each decision is recorded in `docs/DECISIONS.md`:

- **D-099** — the producer model is named Centella. The adapter reads flare-nrt 2.x status: `observation_time`, provenance, `input_age_s` and `stale`.
- **D-100** — the flare detector gets its own product page.
- **D-101 / D-102** — the page is driven by a time cursor. It has two sources: live (`FLARE_NRT_URL`) and an optional replayed archive (`FLARE_ARCHIVE_URL`), both reached through whitelisted proxies in `web/views.py`.
- **(no D)** — a box around the flare on the disk, on the Grad-CAM and on the 4 channels, built from Centella 2.5's `location.bbox_arcsec`.
- **D-103** — the page is redesigned from a reviewed mockup:
  - a dark band with the source switch and the time;
  - a state card with Centella's quiet / onset / active / decay states;
  - a two-lane timeline (probability and GOES);
  - a period summary;
  - one list of flares and alerts.
- **(no D)** — the live page opens on the newest frame that has a result.

## Deploying

- **Producer:** `HelioUSACH/flare-nrt` v2.5.0 (code, units in `deploy/`, weights as a release asset). See its `INTEGRATION.md`.
- **Portal environment:**
  - `FLARE_NRT_URL`: the live Centella API. Defaults to `http://127.0.0.1:5050`.
  - `FLARE_ARCHIVE_URL` (optional): a Centella API over a replayed period. When it is unset, the page has no "Archivo" switch.
- **Nothing else changes:** no migrations, no new dependencies, and ECharts stays vendored.

## Testing

- `ruff check .` and `pytest` pass. `web/tests/test_flare_page.py` covers:
  - the page;
  - the proxies' whitelist and source switch;
  - slot images, Grad-CAM and the X-ray window;
  - the box and the heliographic location.
- `test_daily_todo_returns_everything_stored` was failing before this branch (it depends on the date) and is fixed on `main`.
- Run end to end on a test server for several days:
  - live Centella 2.5 at 2-min cadence;
  - the archive source over a replay of 2026-06-19 → 07-17;
  - the portal on PostgreSQL 16.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_019TiLZJK6V4V5hFsgBRSyAK
