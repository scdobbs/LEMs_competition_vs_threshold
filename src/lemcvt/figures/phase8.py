"""Figures for the addendum 8 report. Read results only.

* ``er_star``: section 2.2 (3), E*-R* per tile at the fitted S_c on the primary network, with the linear line
  (R* = E*/4, and the addendum's E*/2 dashed for reference) and the steady nonlinear curve; bin medians with
  interquartile ranges in log10 E* bins;
* ``slope_threshold``: section 3, MSE(C) profiles per alpha and tile (normalised by each profile's minimum) and the
  hold-out MSE against alpha.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lemcvt.config import TILE_IDS
from lemcvt.figures.common import save, set_nature_style
from lemcvt.results import find_runs


def _latest(step: str):
    for r in reversed(find_runs(step)):
        try:
            rj = json.loads((r / "run.json").read_text())
            if (rj.get("config") or {}).get("crop"):
                continue
            return r, json.loads((r / "params.json").read_text())
        except Exception:  # noqa: BLE001
            continue
    raise FileNotFoundError(f"no {step} run")


def er_star(out_dir: Path, network: str = "primary_1e3") -> list[Path]:
    import matplotlib.pyplot as plt

    from lemcvt.analyses.er_star import steady_nonlinear_R

    run, p = _latest("er_star")
    df = pd.read_parquet(run / "hilltop_pixels.parquet")
    df = df[(df.network == network) & df.used]
    tiles = [t for t in TILE_IDS if t in p["tiles"]]
    set_nature_style()
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.4), sharex=True, sharey=True)
    E_grid = np.logspace(-2, 2, 200)
    cap_bins: list[dict] = []
    cap_tiles: list[dict] = []
    for ax, tile in zip(axes.ravel(), tiles, strict=False):
        net = p["tiles"][tile]["networks"][network]
        sc = net["S_c_fit"].get("S_c", float("nan"))
        s = df[df.tile == tile]
        E = 2.0 * s.C_ht.to_numpy() * s.L_H.to_numpy() / sc
        R = s.Sbar.to_numpy() / sc
        ax.loglog(E, R, ".", color="#9a9995", ms=2, alpha=0.35, rasterized=True)
        edges = np.logspace(np.log10(max(E.min(), 1e-3)), np.log10(E.max()), 13)
        b = np.clip(np.searchsorted(edges, E, side="right") - 1, 0, len(edges) - 2)
        xc, med, lo, hi = [], [], [], []
        for k in range(len(edges) - 1):
            m = b == k
            if m.sum() >= 5:
                xc.append(np.sqrt(edges[k] * edges[k + 1]))
                med.append(np.median(R[m]))
                lo.append(np.percentile(R[m], 25))
                hi.append(np.percentile(R[m], 75))
        if xc:
            ax.errorbar(xc, med, yerr=[np.array(med) - np.array(lo), np.array(hi) - np.array(med)], fmt="o", color="#2a78d6", ms=4, lw=1, capsize=2, label="bin median, IQR")
        ax.loglog(E_grid, steady_nonlinear_R(E_grid), color="#0b0b0b", lw=1.4, label="steady nonlinear")
        ax.loglog(E_grid, E_grid / 4, color="#eb6834", lw=1.2, label="linear: R* = E*/4")
        ax.loglog(E_grid, E_grid / 2, color="#eb6834", lw=0.8, ls="--", label="E*/2 (addendum 8 text)")
        rn = net["ratio_normalised"]
        ci = net["S_c_fit"].get("S_c_ci68") or [float("nan")] * 2
        cap_bins += [{"tile": tile, "network": network, "E_star_bin_centre": x_, "R_star_median": m_, "R_star_q25": l_, "R_star_q75": h_, "run": run.name}
                     for x_, m_, l_, h_ in zip(xc, med, lo, hi, strict=True)]
        cap_tiles.append({"tile": tile, "network": network, "S_c_fit": sc, "S_c_ci68_lo": ci[0], "S_c_ci68_hi": ci[1],
                          "ratio_normalised_median": rn["median"], "n_pixels": int(len(E)), "run": run.name})
        ax.set_title(f"{tile}: S_c {sc:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]\n2Sbar/(C L) median {rn['median']:.2f}", fontsize=8)
        ax.set_xlim(1e-2, 1e2)
        ax.set_ylim(1e-3, 2)
    for ax in axes[1]:
        ax.set_xlabel("E* = 2 C_ht L_H / S_c")
    for ax in axes[:, 0]:
        ax.set_ylabel("R* = Sbar / S_c")
    axes[0, 0].legend(fontsize=6, loc="upper left")
    fig.suptitle(f"E*-R* on the option (iii) hilltop pixels, network {network} (run {run.name})", fontsize=9)
    fig.tight_layout()
    return save(fig, out_dir, "a8_er_star", caption={"bins": cap_bins, "tiles": cap_tiles})


def slope_threshold(out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    from lemcvt.analyses.slope_threshold import collect

    runs = collect(list(TILE_IDS))
    if not runs:
        raise FileNotFoundError("no slope_threshold runs")
    tiles = [t for t in TILE_IDS if t in runs]
    set_nature_style()
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.4))
    cap_prof: list[pd.DataFrame] = []
    cap_ho: list[dict] = []
    colors = {0.0: "#0b0b0b", 0.5: "#2a78d6", 1.0: "#1baf7a", 1.5: "#eda100", 2.0: "#eb6834", 2.5: "#4a3aa7"}
    for ax, tile in zip(axes.ravel(), tiles, strict=False):
        for a in sorted(runs[tile]):
            r, p = runs[tile][a]
            df = pd.read_parquet(r / "c_profile.parquet")
            cap_prof.append(pd.DataFrame({"tile": tile, "alpha": a, "C": df["C"] if "C" in df else np.nan, "n_channel": df.n_channel,
                                          "mse_joint": df.mse_joint, "mse_over_min": df.mse_joint / df.mse_joint.min(), "run": r.name}))
            cap_ho.append({"tile": tile, "alpha": a, "mse_heldout_mean": p["result"]["holdout"]["mse_heldout_mean"], "mse_min": float(df.mse_joint.min()), "run": r.name})
            ax.plot(df.n_channel, df.mse_joint / df.mse_joint.min(), color=colors.get(a, "#9a9995"), lw=1, label=f"alpha {a:g}: held-out {p['result']['holdout']['mse_heldout_mean']:.2f}")
        ax.set_xscale("log")
        ax.set_ylim(0.95, 3)
        ax.set_title(tile, fontsize=9)
        ax.set_xlabel("channel cells at C")
        ax.legend(fontsize=5.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE(C) / min")
    fig.suptitle("Linear joint MSE over the threshold C of A S^alpha >= C, against the channel-cell count (addendum 8, 3)", fontsize=9)
    fig.tight_layout()
    return save(fig, out_dir, "a8_slope_threshold", caption={"profiles": pd.concat(cap_prof, ignore_index=True), "holdout": cap_ho})
