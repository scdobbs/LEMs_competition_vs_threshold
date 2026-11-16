"""Data manifest: sha256 of every grid under ``$LEMCVT_DATA`` (``data/manifest.csv``).

``lemcvt data verify`` compares the files on disk with the manifest;
``lemcvt data fetch`` copies missing files from the source location given in
``configs/data.yaml`` (group storage) and verifies them.
"""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

from lemcvt.config import REPO_ROOT, data_root, load_data_config
from lemcvt.io.grids import sha256_file

MANIFEST_PATH = REPO_ROOT / "data" / "manifest.csv"


@dataclass
class ManifestEntry:
    path: str          # relative to LEMCVT_DATA
    sha256: str
    bytes: int
    site: str
    tile: str
    kind: str
    role: str          # filled / area / fd / output grid name


def read_manifest(path: Path = MANIFEST_PATH) -> list[ManifestEntry]:
    with path.open() as f:
        return [ManifestEntry(r["path"], r["sha256"], int(r["bytes"]), r["site"], r["tile"], r["kind"], r["role"])
                for r in csv.DictReader(f)]


def write_manifest(entries: list[ManifestEntry], path: Path = MANIFEST_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "sha256", "bytes", "site", "tile", "kind", "role"])
        for e in entries:
            w.writerow([e.path, e.sha256, e.bytes, e.site, e.tile, e.kind, e.role])


def verify(root: Path | None = None, entries: list[ManifestEntry] | None = None,
           only_inputs: bool = False) -> tuple[int, list[str]]:
    """Return ``(n_ok, problems)``; problems are 'missing: ...' or 'mismatch: ...' strings."""
    root = root or data_root()
    entries = entries or read_manifest()
    n_ok = 0
    problems: list[str] = []
    for e in entries:
        if only_inputs and e.kind != "input":
            continue
        p = root / e.path
        if not p.exists():
            problems.append(f"missing: {p}")
            continue
        if sha256_file(p) != e.sha256:
            problems.append(f"mismatch: {p}")
            continue
        n_ok += 1
    return n_ok, problems


def fetch(root: Path | None = None, source: Path | None = None, only_inputs: bool = False) -> tuple[int, list[str]]:
    """Copy missing/mismatching manifest files from ``source`` into ``root`` and verify."""
    root = root or data_root()
    if source is None:
        source = Path(load_data_config()["source"]).expanduser()
        source = Path(str(source).replace("$GROUP_HOME", str(Path.home().parent.parent / "groups" / "hilley")))
    entries = read_manifest()
    copied = 0
    for e in entries:
        if only_inputs and e.kind != "input":
            continue
        dst = root / e.path
        if dst.exists() and sha256_file(dst) == e.sha256:
            continue
        src = source / e.path
        if not src.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    n_ok, problems = verify(root, entries, only_inputs=only_inputs)
    return copied, problems
