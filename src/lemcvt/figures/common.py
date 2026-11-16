"""Shared figure helpers: Nature-style rcParams (port of v1 ``set_nature_style``),
result lookup, saving, and the caption-data CSVs written beside every figure (addendum 10, 5.3)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

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

# addendum 10, 1.1: every nonlinear result of v1, Phase 6 and addenda 8-9 solves the centre form D_eff lap(z) = -U
CENTRE_FORM = "centre form, nonstandard"
CENTRE_FORM_TAG = f"({CENTRE_FORM})"
IMAGE_EXTS = (".png", ".pdf")

CaptionData = pd.DataFrame | Sequence[Mapping[str, Any]]


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


def caption_path(out_dir: Path, stem: str, suffix: str = "") -> Path:
    """``<stem>.csv`` or ``<stem>_<suffix>.csv`` beside the figure ``<stem>.png``."""
    return Path(out_dir) / (f"{stem}_{suffix}.csv" if suffix else f"{stem}.csv")


def write_caption(out_dir: Path, stem: str, data: CaptionData | Mapping[str, CaptionData], suffix: str = "") -> list[Path]:
    """Write the data a figure plots (points, lines, medians, intervals, run ids) as CSV beside the figure.

    ``data`` is a table (DataFrame or list of row dicts) written to ``<stem>.csv`` (or ``<stem>_<suffix>.csv``),
    or a mapping ``{suffix: table}`` for figures whose panels plot different data (one CSV per suffix).
    """
    if isinstance(data, Mapping) and not isinstance(data, pd.DataFrame):
        paths: list[Path] = []
        for sfx, tab in data.items():
            paths += write_caption(out_dir, stem, tab, suffix=f"{suffix}_{sfx}" if suffix and sfx else (sfx or suffix))
        return paths
    df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(list(data))
    if df.empty:
        df = pd.DataFrame({"note": ["no data plotted"]})
    p = caption_path(out_dir, stem, suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
    return [p]


def missing_captions(out_dir: Path) -> list[str]:
    """Figure stems in ``out_dir`` (PNG/PDF) without a caption CSV (``<stem>.csv`` or ``<stem>_*.csv``)."""
    out_dir = Path(out_dir)
    stems = sorted({p.stem for p in out_dir.iterdir() if p.suffix in IMAGE_EXTS})
    csvs = {p.stem for p in out_dir.glob("*.csv")}
    return [s for s in stems if s not in csvs and not any(c.startswith(s + "_") for c in csvs)]


def save(fig, out_dir: Path, stem: str, caption: CaptionData | Mapping[str, CaptionData] | None = None) -> list[Path]:
    """Save PDF + PNG and, when given, the caption data CSV(s) (see :func:`write_caption`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("pdf", "png"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    if caption is not None:
        paths += write_caption(out_dir, stem, caption)
    return paths


def savefig_with_caption(fig, out: Path, caption: CaptionData | Mapping[str, CaptionData], dpi: int = 130, pdf: bool = True) -> list[Path]:
    """The report figures' save (PNG at ``dpi`` and optionally PDF) plus the caption CSV(s) under the same stem."""
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    paths = [out]
    if pdf:
        fig.savefig(out.with_suffix(".pdf"))
        paths.append(out.with_suffix(".pdf"))
    plt.close(fig)
    return paths + write_caption(out.parent, out.stem, caption)


def provenance_note(ax, run: Path) -> None:
    ax.annotate(f"results: {run.name}", xy=(1.0, -0.02), xycoords="axes fraction", ha="right", va="top",
                fontsize=4, color="0.5", annotation_clip=False)
