"""fig_slope_area: 3 x 2 panels (AD no-opt, AD opt, partitioned) x (AP, GM)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib.ticker import LogLocator

from lemcvt.figures.common import (
    COLOR_AD_NO_OPT,
    COLOR_AD_OPT,
    COLOR_PART,
    GRAY_ORIG,
    load_params,
    load_table,
    plt,
    provenance_note,
    require_run,
    save,
    set_nature_style,
)
from lemcvt.measure.slope_area import compute_iqr_errors

ROWS = [("AD_no_opt", "AD no-opt", COLOR_AD_NO_OPT), ("AD_opt", "AD opt", COLOR_AD_OPT), ("part", "Partitioned", COLOR_PART)]


def _dbg(df, tile, series):
    sub = df[(df.tile == tile) & (df.series == series)]
    b = sub[sub.kind == "bin"]
    r = sub[sub.kind == "raw"]
    return {"A": 10 ** r.logA.to_numpy(), "S": 10 ** r.logS.to_numpy(), "bin_logA": b.logA.to_numpy(),
            "bin_logS": b.logS.to_numpy(), "bin_counts": b["count"].to_numpy()}


def build(out_dir: Path, run: Path | None = None) -> list[Path]:
    set_nature_style()
    run = run or require_run("slope_area")
    params = load_params(run)
    df = load_table(run, "slope_area")
    tiles = list(params["tiles"])
    sites = [t[:2] for t in tiles]
    fig, axes = plt.subplots(3, len(tiles), figsize=(3.25 * len(tiles), 7.0), sharey=True, squeeze=False)

    bins_rows: list[dict] = []
    fit_rows: list[dict] = []
    smins, smaxs = [], []
    for tile in tiles:
        for series in ("orig", *[r[0] for r in ROWS]):
            d = _dbg(df, tile, series)
            smins.append(np.nanmin(10 ** d["bin_logS"]))
            smaxs.append(np.nanmax(10 ** d["bin_logS"]))
    ymin, ymax = min(smins) * 0.9, max(smaxs) * 1.1

    for col, tile in enumerate(tiles):
        res = params["tiles"][tile]
        orig = res["orig"]
        dbg_o = _dbg(df, tile, "orig")
        lo_o, hi_o = compute_iqr_errors(dbg_o)
        for k in range(len(dbg_o["bin_logA"])):
            bins_rows.append({"tile": tile, "series": "orig", "label": "observed", "A_bin": 10 ** dbg_o["bin_logA"][k],
                              "S_median": 10 ** dbg_o["bin_logS"][k], "S_err_lo": lo_o[k], "S_err_hi": hi_o[k],
                              "count": dbg_o["bin_counts"][k], "run": run.name})
        fit_rows.append({"tile": tile, "series": "orig", "label": "observed", "Ks": orig["Ks"], "theta": orig["theta"],
                         "R2": orig.get("R2", np.nan), "run": run.name})
        for row, (key, label, color) in enumerate(ROWS):
            ax = axes[row, col]
            mod = res[key]
            dbg_m = _dbg(df, tile, key)
            err_lo_o, err_hi_o = compute_iqr_errors(dbg_o)
            err_lo_m, err_hi_m = compute_iqr_errors(dbg_m)
            for k in range(len(dbg_m["bin_logA"])):
                bins_rows.append({"tile": tile, "series": key, "label": label, "A_bin": 10 ** dbg_m["bin_logA"][k],
                                  "S_median": 10 ** dbg_m["bin_logS"][k], "S_err_lo": err_lo_m[k], "S_err_hi": err_hi_m[k],
                                  "count": dbg_m["bin_counts"][k], "run": run.name})
            fit_rows.append({"tile": tile, "series": key, "label": label, "Ks": mod["Ks"], "theta": mod["theta"],
                             "R2": mod.get("R2", np.nan), "run": run.name})
            logA_min = min(np.nanmin(dbg_o["bin_logA"]), np.nanmin(dbg_m["bin_logA"]))
            logA_max = max(np.nanmax(dbg_o["bin_logA"]), np.nanmax(dbg_m["bin_logA"]))
            A_fit = np.logspace(logA_min, logA_max, 200)
            ax.loglog(dbg_o["A"], dbg_o["S"], "o", markersize=0.2, alpha=0.15, color=GRAY_ORIG, markeredgewidth=0,
                      rasterized=True)
            ax.errorbar(10 ** dbg_o["bin_logA"], 10 ** dbg_o["bin_logS"], yerr=[err_lo_o, err_hi_o], fmt="o",
                        markersize=2.5, mfc="0.7", mec="black", mew=0.15, ecolor="black", elinewidth=0.3, capsize=1.0,
                        zorder=3)
            ax.loglog(A_fit, orig["Ks"] * A_fit ** (-orig["theta"]), color="black", lw=0.6)
            ax.loglog(dbg_m["A"], dbg_m["S"], "o", markersize=0.7, alpha=0.3, color=color, markeredgewidth=0,
                      rasterized=True)
            ax.errorbar(10 ** dbg_m["bin_logA"], 10 ** dbg_m["bin_logS"], yerr=[err_lo_m, err_hi_m], fmt="o",
                        markersize=2.5, mfc=color, mec="black", mew=0.15, ecolor=color, elinewidth=0.3, capsize=1.0,
                        zorder=3)
            ax.loglog(A_fit, mod["Ks"] * A_fit ** (-mod["theta"]), color=color, lw=0.6, linestyle=(0, (3, 3)))
            ax.xaxis.set_major_locator(LogLocator(base=10))
            ax.yaxis.set_major_locator(LogLocator(base=10))
            ax.minorticks_on()
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.set_ylim(ymin, ymax)
            if row == 0:
                ax.set_title(f"{sites[col]} tile{tile[2:]}", pad=3)
            if col == 0:
                ax.text(0.02, 0.95, label, transform=ax.transAxes, ha="left", va="top", fontsize=7)
                ax.set_ylabel(r"Slope $S$")
            ax.text(0.98, 0.05, f"$k_s$={mod['Ks']:.3g}, $\\theta$={mod['theta']:.3f}, $R^2$={mod['R2']:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=5, color=color)
            if row == 2:
                ax.set_xlabel(r"Drainage area $A$ (m$^2$)")
    provenance_note(axes[-1, -1], run)
    fig.tight_layout()
    return save(fig, out_dir, "fig_slope_area", caption={"bins": bins_rows, "fits": fit_rows})
