"""Tables built from results only. ``lemcvt tables all`` writes LaTeX (booktabs + siunitx S
columns) and CSV into ``tables/``."""

from __future__ import annotations

from pathlib import Path

from lemcvt.config import REPO_ROOT

TABLE_BUILDERS = {"table1": "lemcvt.tables.table1:build"}


def build_tables(which: str = "all", out_dir: Path | None = None) -> list[Path]:
    import importlib

    out_dir = Path(out_dir) if out_dir else REPO_ROOT / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(TABLE_BUILDERS) if which == "all" else [w.strip() for w in which.split(",")]
    paths: list[Path] = []
    for name in names:
        if name not in TABLE_BUILDERS:
            raise ValueError(f"unknown table {name!r}; choose from {sorted(TABLE_BUILDERS)}")
        mod, fn = TABLE_BUILDERS[name].split(":")
        paths += getattr(importlib.import_module(mod), fn)(out_dir)
    return paths
