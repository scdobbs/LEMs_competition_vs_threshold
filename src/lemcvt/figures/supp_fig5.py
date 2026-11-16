"""supp_fig5: synthetic Peclet analysis, k_s/(U/K) against pixel size for three estimators."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.ticker import LogLocator

from lemcvt.figures.common import load_params, load_table, plt, provenance_note, require_run, save, set_nature_style


def _curve(ax, dx, y, label, color="0.0", marker="o"):
    ax.plot(dx, y, "-" + marker, color=color, lw=0.8, ms=2.0, mfc=color, mec=color, mew=0.0, zorder=3, label=label)


def build(out_dir: Path, run: Path | None = None) -> list[Path]:
    set_nature_style()
    run = run or require_run("synthetic_pe")
    df = load_table(run, "curves")
    params = load_params(run)
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.6))
    panels = [
        ("laplacian", "ks_norm", r"$k_{s,\mathrm{eff}}/(U/K)$ (Laplacian)"),
        ("regression", "ks_norm", r"$k_s/(U/K)$ (slope-area regression)"),
        ("local_ks", "ks_med_norm", r"$\mathrm{median}(S\,A^m)/(U/K)$"),
    ]
    b = params["baseline"]
    cap_rows: list[dict] = []
    for ax, (variant, col, ylabel) in zip(axes, panels, strict=True):
        sub = df[(df.variant == variant) & (df.valid)].sort_values("dx")
        dx = sub.dx.to_numpy()
        y = sub[col].to_numpy()
        for k in range(len(dx)):
            cap_rows.append({"variant": variant, "quantity": col, "dx_m": dx[k], "value": y[k],
                             "q25": sub.ks_q25_norm.iloc[k] if variant == "local_ks" else np.nan,
                             "q75": sub.ks_q75_norm.iloc[k] if variant == "local_ks" else np.nan,
                             "guide_dx_minus1": y[0] * (dx[k] / dx[0]) ** (-1.0),
                             "Pe_primary": b["Pe_primary"], "D_primary_m2_yr": b["D_primary"], "A_inlet_bal_m2": b["A_inlet_bal"],
                             "run": run.name})
        _curve(ax, dx, y, variant)
        if variant == "local_ks":
            ax.fill_between(dx, sub.ks_q25_norm.to_numpy(), sub.ks_q75_norm.to_numpy(), color="0.85", zorder=1, lw=0,
                            label="IQR")
        if dx.size:
            dx_ref = np.array([dx.min(), dx.max()])
            ax.plot(dx_ref, y[0] * (dx_ref / dx[0]) ** (-1.0), "--", color="0.35", lw=0.7, dashes=(4, 3), zorder=2,
                    label=r"$\propto \Delta x^{-1}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e0, 2e2)
        ax.set_xlabel(r"Pixel size $\Delta x$ (m)")
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.tick_params(which="both", pad=2)
    fig.suptitle(f"Pe = {b['Pe_primary']:.3g}, D = {b['D_primary']:.3e} m$^2$/yr, A_inlet = {b['A_inlet_bal']:.3g} m$^2$",
                 fontsize=7)
    provenance_note(axes[-1], run)
    fig.tight_layout()
    return save(fig, out_dir, "supp_fig5_synthetic_pe", caption=cap_rows)
