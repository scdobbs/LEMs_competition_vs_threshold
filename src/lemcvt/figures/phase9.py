"""Figures for the addendum 9 report. Read results only.

* ``pilot``: section 2.2 test 4 and the pilot (5.1), per pilot tile: MSE against D along the A_crit row for Picard and
  Newton on the centre form and Newton on the flux form, marker shape by outcome;
* ``fair``: section 3, per tile, the converged MSE over (A_crit, D) at the central S_c divided by the linear joint MSE,
  with the 1 % and 5 % contours' points, the optimum and the D0 values;
* ``morphology``: section 4.2, E*-R* per tile at the observed fitted S_c: the observed pixels and the bin medians of
  every fitted surface along the observed D8 paths, with the steady nonlinear curve and the linear line.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lemcvt.config import TILE_IDS
from lemcvt.figures.common import save, set_nature_style
from lemcvt.figures.phase8 import _latest
from lemcvt.results import find_runs

# validated categorical order (dataviz reference palette, slots 1-4: all six checks pass on the light surface)
SURFACE_STYLE = {"observed": ("#5f5e5a", "o"), "linear_joint": ("#2a78d6", "s"), "ad": ("#eb6834", "D"),
                 "nonlinear_joint_1.27": ("#1baf7a", "^"), "nonlinear_flux_fitted_Sc": ("#eda100", "v")}
SURFACE_LABEL = {"observed": "observed DEM", "linear_joint": "linear joint", "ad": "AD optimum",
                 "nonlinear_joint_1.27": "nonlinear joint, S_c 1.27 (v1 form)", "nonlinear_flux_fitted_Sc": "nonlinear flux form, fitted S_c"}
OUTCOME_MARKER = {"converged": "o", "no_steady_solution": "x", "not_converged": "s"}
RUN_STYLE = {("picard", "centre"): ("#5f5e5a", "Picard, v1 form"), ("newton", "centre"): ("#2a78d6", "Newton, v1 form"),
             ("newton", "flux"): ("#eb6834", "Newton, flux form")}


def pilot(out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    run, p = _latest("a9_pilot_summary")
    per = p["per_task"]
    tiles = sorted({k.split("/")[0] for k in per})
    set_nature_style()
    fig, axes = plt.subplots(1, len(tiles), figsize=(5.2 * len(tiles), 3.6), squeeze=False)
    for ax, tile in zip(axes[0], tiles, strict=True):
        frames = [pd.read_parquet([r for r in find_runs("a9_pilot") if r.name == per[k]["run"]][0] / "pilot_rows.parquet")
                  for k in per if k.startswith(tile + "/")]
        s = pd.concat(frames, ignore_index=True)
        for (solver, form), (col, lab) in RUN_STYLE.items():
            g = s[(s.solver == solver) & (s.form == form)].sort_values("D")
            if g.empty:
                continue
            ax.plot(g.D, g.mse, color=col, lw=1.2, label=lab)
            for oc, mk in OUTCOME_MARKER.items():
                h = g[g.outcome == oc]
                ax.plot(h.D, h.mse, mk, color=col, ms=5, mfc="none" if oc != "converged" else col, ls="none")
        c = per.get(f"{tile}/centre") or next(v for k, v in per.items() if k.startswith(tile + "/"))
        ax.axvline(float(c["linear_joint"]["D"]), color="#0b0b0b", lw=0.8, ls="--")
        ax.axvline(float(c["D0"]), color="#0b0b0b", lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("D (m$^2$/yr)")
        ax.set_ylabel("MSE (m$^2$)")
        ax.set_title(f"{tile}: A_crit {c['A_crit']:.0f} m$^2$, S_c {c['S_c']:.3f}; dashed linear joint D, dotted D0_A", fontsize=8)
    handles = [plt.Line2D([], [], color=c_, lw=1.2, label=lab) for c_, lab in RUN_STYLE.values()]
    handles += [plt.Line2D([], [], color="#5f5e5a", marker=mk, ls="none", mfc="none" if oc != "converged" else "#5f5e5a", label=oc.replace("_", " "))
                for oc, mk in OUTCOME_MARKER.items()]
    axes[0, 0].legend(handles=handles, fontsize=6, loc="best")
    fig.suptitle(f"Addendum 9 pilot: one A_crit row across the D grid (summary run {run.name})", fontsize=9)
    fig.tight_layout()
    return save(fig, out_dir, "a9_pilot")


def fair(out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    run, p = _latest("a9_fair_summary")
    tiles = [t for t in TILE_IDS if t in p["per_tile"] and "central" in p["per_tile"][t]]
    if not tiles:
        raise FileNotFoundError("no merged a9_fair runs at the central S_c")
    set_nature_style()
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.4), squeeze=False)
    for ax, tile in zip(axes.ravel(), tiles, strict=False):
        m = p["per_tile"][tile]["central"]
        r = [x for x in find_runs("a9_fair_merged") if x.name == m["run"]][0]
        df = pd.read_parquet(r / "fair_rows.parquet")
        lin = float(m["linear_joint"]["mse"])
        conv = df[df.outcome == "converged"]
        sc = ax.scatter(conv.A_crit, conv.D, c=np.log10(conv.mse / lin), cmap="Blues_r", s=14, vmin=0, vmax=1, edgecolors="none")
        for oc, mk in (("no_steady_solution", "x"), ("not_converged", "s")):
            h = df[df.outcome == oc]
            ax.plot(h.A_crit, h.D, mk, color="#5f5e5a", ms=3, mfc="none", ls="none", label=oc.replace("_", " "))
        for lev, col in (("contour_0.05", "#eda100"), ("contour_0.01", "#eb6834")):
            pts = np.array(m.get("contours", {}).get(lev, {}).get("points", []))
            if pts.size:
                ax.plot(pts[:, 0], pts[:, 1], "o", color=col, mfc="none", ms=6, ls="none", label=lev.replace("contour_", "within "))
        if "optimum" in m:
            ax.plot(m["optimum"]["A_crit"], m["optimum"]["D"], "*", color="#0b0b0b", ms=10, label="optimum")
        ax.axhline(m["D0"], color="#0b0b0b", lw=0.8, ls="--")
        if tile.startswith("AP") and m.get("D0_B"):
            ax.axhline(m["D0_B"], color="#0b0b0b", lw=0.8, ls=":")
        ax.axhline(float(m["linear_joint"]["D"]), color="#2a78d6", lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{tile}: S_c {m['S_c']:.3f}, MSE/lin {m.get('mse_ratio_to_linear_joint', float('nan')):.2f}", fontsize=8)
        ax.set_xlabel("A_crit (m$^2$)")
        ax.set_ylabel("D (m$^2$/yr)")
    axes[0, 0].legend(fontsize=5, loc="lower left")
    fig.colorbar(sc, ax=axes, shrink=0.6, label="log10(MSE / linear joint MSE), converged only")
    fig.suptitle(f"Nonlinear (flux form) joint profile at the fitted S_c; dashed D0_A, dotted D0_B (AP), blue the linear joint D (run {run.name})", fontsize=9)
    return save(fig, out_dir, "a9_fair")


def morphology(out_dir: Path) -> list[Path]:
    import matplotlib.pyplot as plt

    from lemcvt.analyses.er_star import steady_nonlinear_R

    run, p = _latest("a9_morphology")
    df = pd.read_parquet(run / "morphology_pixels.parquet")
    tiles = [t for t in TILE_IDS if t in p["per_tile"]]
    set_nature_style()
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.6), sharex=True, sharey=True, squeeze=False)
    E_grid = np.logspace(-2, 2, 200)
    for ax, tile in zip(axes.ravel(), tiles, strict=False):
        s_all = df[(df.tile == tile) & df.used]
        obs = s_all[s_all.surface == "observed"]
        ax.loglog(obs.E_star_at_obs_Sc, obs.R_star_at_obs_Sc, ".", color="#b4b2a9", ms=2, alpha=0.35, rasterized=True)
        for name, (col, mk) in SURFACE_STYLE.items():
            s = s_all[s_all.surface == name]
            ok = (s.C_ht > 0) & (s.R > 0)
            E, R = s.E_star_at_obs_Sc[ok].to_numpy(), s.R_star_at_obs_Sc[ok].to_numpy()
            if E.size < 20:
                continue
            edges = np.logspace(np.log10(max(np.percentile(E, 1), 1e-3)), np.log10(np.percentile(E, 99)), 11)
            b = np.clip(np.searchsorted(edges, E, side="right") - 1, 0, len(edges) - 2)
            xc = [np.sqrt(edges[k] * edges[k + 1]) for k in range(len(edges) - 1) if (b == k).sum() >= 5]
            md = [np.median(R[b == k]) for k in range(len(edges) - 1) if (b == k).sum() >= 5]
            ax.plot(xc, md, marker=mk, color=col, ms=4, lw=1.2, label=SURFACE_LABEL[name])
        ax.loglog(E_grid, steady_nonlinear_R(E_grid), color="#0b0b0b", lw=1.0, label="steady nonlinear (Roering)")
        ax.loglog(E_grid, E_grid / 4, color="#0b0b0b", lw=0.8, ls="--", label="linear: R* = E*/4")
        ax.set_title(f"{tile}: S_c (obs. fit) {p['per_tile'][tile]['S_c_observed_fit']:.2f}", fontsize=8)
        ax.set_xlim(1e-1, 1e2)
        ax.set_ylim(1e-2, 2)
    for ax in axes[1]:
        ax.set_xlabel("E* = 2 C_ht L_H / S_c")
    for ax in axes[:, 0]:
        ax.set_ylabel("R* = Sbar / S_c")
    axes[0, 0].legend(fontsize=5, loc="lower right")
    fig.suptitle(f"E*-R* of every fitted surface along the observed D8 paths, at the observed fitted S_c; grey: observed pixels (run {run.name})", fontsize=9)
    fig.tight_layout()
    return save(fig, out_dir, "a9_morphology")
