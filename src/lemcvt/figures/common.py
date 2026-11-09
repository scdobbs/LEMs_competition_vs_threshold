"""Shared figure helpers: Nature-style rcParams (port of v1 ``set_nature_style``),
result lookup, saving."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from lemcvt.results import latest_run  # noqa: E402

COL_GM = "#0072B2"   # Okabe-Ito blue
COL_AP = "#D55E00"   # vermillion
GRAY_ORIG = "0.25"
COLOR_PART = "#3A7D44"
COLOR_AD_NO_OPT = "#1f77b4"
COLOR_AD_OPT = "#D55E00"


def set_nature_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 300, "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "axes.linewidth": 0.5,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": False, "ytick.right": False,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "savefig.bbox": "tight", "savefig.dpi": 300,
    })


def require_run(step: str, label: str | None = None) -> Path:
    run = latest_run(step, label) if label else None
    if run is None and label is None:
        from lemcvt.results import find_runs

        runs = find_runs(step)
        run = runs[-1] if runs else None
    if run is None:
        raise FileNotFoundError(f"no completed '{step}' run{f' for {label}' if label else ''} under results; run it first")
    return run


def load_params(run: Path) -> dict:
    return json.loads((run / "params.json").read_text())


def load_table(run: Path, name: str) -> pd.DataFrame:
    p = run / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    c = run / f"{name}.csv"
    if c.exists():
        return pd.read_csv(c)
    raise FileNotFoundError(f"{name} table missing in {run}")


def save(fig, out_dir: Path, stem: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths


def provenance_note(ax, run: Path) -> None:
    ax.annotate(f"results: {run.name}", xy=(1.0, -0.02), xycoords="axes fraction", ha="right", va="top",
                fontsize=4, color="0.5", annotation_clip=False)
