"""supp_fig6 and resolution_study_dinf: Ks_mod/Ks_obs against pixel size, median + IQR by site."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.ticker import LogFormatterMathtext, LogLocator

from lemcvt.figures.common import COL_AP, COL_GM, load_table, plt, provenance_note, save, set_nature_style
from lemcvt.results import find_runs


def _collect() -> tuple[pd.DataFrame, list[Path]]:
    """Concatenate all sweep_resolution runs, keeping the newest row per (tile, dx)."""
    runs = find_runs("sweep_resolution")
    if not runs:
        raise FileNotFoundError("no sweep_resolution results")
    frames = []
    for r in runs:
        df = load_table(r, "table")
        df["run"] = r.name
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values("run").drop_duplicates(subset=["tile", "dx"], keep="last")
    return df, runs


def group_stats(df: pd.DataFrame, site: str, dx_grid: np.ndarray, key: str):
    med, q25, q75 = [], [], []
    sub = df[df.site == site]
    for d in dx_grid:
        vals = sub[np.isclose(sub.dx.to_numpy(), d, rtol=0, atol=1e-10)][key].to_numpy()
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size == 0:
            med.append(np.nan)
            q25.append(np.nan)
            q75.append(np.nan)
        else:
            med.append(np.nanmedian(vals))
            q25.append(np.nanpercentile(vals, 25))
            q75.append(np.nanpercentile(vals, 75))
    return np.array(med), np.array(q25), np.array(q75)


def plot_median_iqr(ax, x, med, q25, q75, color, marker, linestyle, label=None):
    m = np.isfinite(x) & np.isfinite(med) & np.isfinite(q25) & np.isfinite(q75) & (med > 0) & (q25 > 0) & (q75 > 0)
    if np.sum(m) < 2:
        return
    ax.fill_between(x[m], q25[m], q75[m], facecolor=color, alpha=0.18, linewidth=0, edgecolor="none", interpolate=True,
                    zorder=1)
    ax.plot(x[m], med[m], linestyle + marker, color=color, lw=0.9, ms=2.4, mfc=color, mec=color, mew=0.0, zorder=3,
            label=label)


def _axes(ax):
    ax.axhline(1.0, lw=0.7, color="0.35", ls="--", dashes=(4, 3), zorder=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Grid resolution $\Delta x$ (m)")
    ax.set_ylabel(r"$K_{s,\mathrm{mod}}(\Delta x)/K_{s,\mathrm{obs}}(\Delta x)$")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=10))
    ax.yaxis.set_major_locator(LogLocator(base=10.0, subs=(1.0,), numticks=10))
    ax.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
    ax.tick_params(which="both", pad=2)


def _guide(ax, dx_all, series):
    j = np.where(np.isfinite(dx_all) & np.isfinite(series) & (series > 0))[0]
    if j.size:
        j0 = j[0]
        dx_line = np.array([np.nanmin(dx_all), np.nanmax(dx_all)])
        ax.plot(dx_line, series[j0] * (dx_line / dx_all[j0]) ** (-1.0), "--", lw=0.7, color="0.65", dashes=(4, 3), zorder=0)


def build_d8(out_dir: Path) -> list[Path]:
    set_nature_style()
    df, runs = _collect()
    if "ratio_d8" not in df:
        raise FileNotFoundError("sweep results carry no D8 ratios")
    dx_all = np.array(sorted(df.dx.unique()), dtype=float)
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    med_gm, q25_gm, q75_gm = group_stats(df, "GM", dx_all, "ratio_d8")
    med_ap, q25_ap, q75_ap = group_stats(df, "AP", dx_all, "ratio_d8")
    plot_median_iqr(ax, dx_all, med_gm, q25_gm, q75_gm, COL_GM, "o", "-", "GM")
    plot_median_iqr(ax, dx_all, med_ap, q25_ap, q75_ap, COL_AP, "s", "--", "AP")
    _guide(ax, dx_all, med_gm if np.isfinite(med_gm).any() else med_ap)
    _axes(ax)
    ax.legend(frameon=False, fontsize=6)
    provenance_note(ax, runs[-1])
    fig.tight_layout()
    return save(fig, out_dir, "supp_fig6_resolution_ratio")


def build_dinf(out_dir: Path) -> list[Path]:
    set_nature_style()
    df, runs = _collect()
    if "ratio_dinf" not in df:
        raise FileNotFoundError("sweep results carry no D-inf ratios (run with --routing d8,dinf)")
    dx_all = np.array(sorted(df.dx.unique()), dtype=float)
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for site, color, mk in (("GM", COL_GM, ("o", "^")), ("AP", COL_AP, ("s", "D"))):
        m8, l8, h8 = group_stats(df, site, dx_all, "ratio_d8")
        mi, li, hi = group_stats(df, site, dx_all, "ratio_dinf")
        plot_median_iqr(ax, dx_all, m8, l8, h8, color, mk[0], "-", f"{site} (D8)")
        plot_median_iqr(ax, dx_all, mi, li, hi, color, mk[1], "--", f"{site} (D-inf)")
    m8, _, _ = group_stats(df, "GM", dx_all, "ratio_d8")
    _guide(ax, dx_all, m8)
    _axes(ax)
    ax.legend(frameon=False, fontsize=6)
    provenance_note(ax, runs[-1])
    fig.tight_layout()
    return save(fig, out_dir, "resolution_study_dinf")
