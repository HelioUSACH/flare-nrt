"""Download the model files from the GitHub release and verify them
(checklist A.3): nothing is written to its final path unless its SHA-256
matches the one pinned here.

    python -m centella.fetch_weights             # into ./models and ./metadata
    python -m centella.fetch_weights --dest /srv/centella

The same SHA-256 is the default of ``CENTELLA_WEIGHTS_SHA256``; the loop
refuses to load anything else."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

RELEASE = os.environ.get("CENTELLA_RELEASE_URL", "https://github.com/HelioUSACH/flare-nrt/releases/download/v2.5.0")

FILES = {
    # name: (subdirectory, SHA-256, size in bytes)
    "deployment_ensemble_thr058.pt": ("models", "d3b850e3ec3d2670f4cb9327c397f7391ed1b082d2b6d154bca1b57626676006",
                                      224246080),
    "global_norm_stats.json": ("metadata", "807723e30a6a1c0e066c003bb6ad8c3a926a14e2ad2afd2f64f0aff46b83c075", 808),
}
WEIGHTS_SHA256 = FILES["deployment_ensemble_thr058.pt"][1]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dest: Path, release: str = RELEASE) -> list[Path]:
    import requests

    out = []
    for name, (sub, digest, _size) in FILES.items():
        final = dest / sub / name
        if final.exists() and sha256(final) == digest:
            print(f"ok      {final}")
            out.append(final)
            continue
        final.parent.mkdir(parents=True, exist_ok=True)
        part = final.with_suffix(final.suffix + ".part")
        with requests.get(f"{release}/{name}", stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(part, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        got = sha256(part)
        if got != digest:
            part.unlink(missing_ok=True)
            raise SystemExit(f"SHA-256 mismatch for {name}: got {got}, expected {digest}")
        part.replace(final)
        print(f"fetched {final}")
        out.append(final)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=".", help="directory that gets models/ and metadata/ (default: .)")
    ap.add_argument("--release", default=RELEASE, help="release download URL (default: %(default)s)")
    a = ap.parse_args(argv)
    fetch(Path(a.dest), a.release)


if __name__ == "__main__":
    sys.exit(main())
