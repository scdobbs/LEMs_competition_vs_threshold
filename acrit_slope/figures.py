"""Figures for the A_crit-slope test. All produced from code; no manual post-processing."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from width_analysis.qc import hillshade  # noqa: E402

from .heads import EXCLUSION_REASONS, walk_path  # noqa: E402

# categorical slots of the dataviz reference palette (validated, light surface)
C_SITE = {"AP": "#2a78d6", "GM": "#eb6834"}          # blue, orange
C_AP5 = "#4a3aa7"                                   # violet: AP tile 5
C_REASON = {"tile_edge": "#1baf7a", "path_leaves_tile": "#eda100", "flat_fill": "#e34948"}  # aqua, yellow, red
C_KEPT = "#2a78d6"
C_TEXT, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#efeeea"
M_SITE = {"AP": "s", "GM": "o"}                      # squares AP, circles GM (as in the manuscript)


def _style(ax):
    ax.grid(color=C_GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=C_MUTED, labelsize=8)


def _label(site, tile):
    return f"{site}{tile}"


# ---------------------------------------------------------------------------
# figure 1: A_crit against median head slope
# ---------------------------------------------------------------------------

def fig_main(summary: pd.DataFrame, fits: dict, boot_ab: np.ndarray, ap5_range: dict, cfg: dict, out: Path,
             slope_col: str = "S_up_10") -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.2), facecolor="white")
    x = summary[f"{slope_col}_median"].to_numpy()
    y = summary["A_crit_m2"].to_numpy()
    xlo = x - summary[f"{slope_col}_q25"].to_numpy()
    xhi = summary[f"{slope_col}_q75"].to_numpy() - x

    # fit line and band (variant 1 RMA, all ten tiles)
    v = fits["variants"]["v1_all_RMA"]
    xg = np.logspace(np.log10(x.min()) - 0.15, np.log10(x.max()) + 0.15, 100)
    lx = np.log10(xg)
    ax.plot(xg, 10 ** (v["a"] + v["b"] * lx), color=C_TEXT, lw=2.0,
            label=f"RMA, all ten tiles: b = {v['b']:.2f} [{v['ci_b'][0]:.2f}, {v['ci_b'][1]:.2f}]")
    if boot_ab is not None and len(boot_ab):
        pred = boot_ab[:, [0]] + boot_ab[:, [1]] * lx[None, :]
        lo, hi = np.percentile(pred, cfg["ci_percentiles"], axis=0)
        ax.fill_between(xg, 10 ** lo, 10 ** hi, color=C_TEXT, alpha=0.10, lw=0, label="95 % bootstrap band")
    # -2 reference through the centroid of the nine tiles without AP5
    nine = ~((summary.site == "AP") & (summary.tile == 5)).to_numpy()
    cx, cy = np.mean(np.log10(x[nine])), np.mean(np.log10(y[nine]))
    ax.plot(xg, 10 ** (cy + cfg["reference_exponent"] * (lx - cx)), color=C_MUTED, lw=1.4, ls=(0, (4, 3)),
            label="A ∝ S⁻² through the centroid of the nine other tiles")

    for site in ("AP", "GM"):
        m = (summary.site == site).to_numpy()
        ax.errorbar(x[m], y[m], xerr=[xlo[m], xhi[m]], fmt="none", ecolor=C_SITE[site], elinewidth=1.2, capsize=2.5, alpha=0.9)
        ax.plot(x[m], y[m], M_SITE[site], color=C_SITE[site], ms=8, mec="white", mew=1.0, ls="none",
                label=f"{site} tiles (horizontal bar: IQR of head slope)")
    k = np.nonzero(~nine)[0][0]
    ax.errorbar([x[k]], [y[k]], yerr=[[y[k] - ap5_range["lo"]], [ap5_range["hi"] - y[k]]], fmt="none",
                ecolor=C_AP5, elinewidth=1.6, capsize=4)
    ax.plot([x[k]], [y[k]], M_SITE["AP"], color=C_AP5, ms=11, mec="white", mew=1.2, ls="none",
            label=f"AP tile 5 (vertical bar: misfit within {100 * cfg['misfit_tolerance']:.0f} % of minimum)")
    for xi, yi, s, t in zip(x, y, summary.site, summary.tile):
        ax.annotate(_label(s, t), (xi, yi), xytext=(6, 5), textcoords="offset points", fontsize=8, color=C_TEXT)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"median head slope, {slope_col.replace('_', ' ')} m path (m/m)")
    ax.set_ylabel("optimized A_crit (m²)")
    ax.set_title("Optimized A_crit against slope at the partition channel heads", fontsize=10, loc="left", color=C_TEXT)
    _style(ax)
    ax.legend(fontsize=7, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figure 2: per-tile head slopes
# ---------------------------------------------------------------------------

def fig_boxes(heads: pd.DataFrame, summary: pd.DataFrame, out: Path, slope_col: str = "S_up_10") -> None:
    order = summary.sort_values("A_crit_m2")
    data, labels, colors = [], [], []
    for _, r in order.iterrows():
        sub = heads[(heads.site == r.site) & (heads.tile == r.tile) & heads.kept]
        data.append(sub[slope_col].dropna().to_numpy())
        labels.append(f"{_label(r.site, r.tile)}\n{r.A_crit_m2:.2g}\nn={len(sub)}")
        colors.append(C_AP5 if (r.site == "AP" and r.tile == 5) else C_SITE[r.site])
    fig, ax = plt.subplots(figsize=(8.5, 4.4), facecolor="white")
    bp = ax.boxplot(data, widths=0.55, showfliers=True, patch_artist=True,
                    flierprops=dict(marker=".", ms=3, mec=C_MUTED, alpha=0.4),
                    medianprops=dict(color=C_TEXT, lw=1.6), whiskerprops=dict(color=C_MUTED), capprops=dict(color=C_MUTED))
    for patch, c in zip(bp["boxes"], colors):
        patch.set(facecolor=c, alpha=0.35, edgecolor=c, lw=1.2)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_yscale("log")
    ax.set_ylabel(f"{slope_col.replace('_', ' ')} m at kept heads (m/m)")
    ax.set_title("Head slopes per tile, ordered by A_crit (m²); blue AP, orange GM, violet AP5", fontsize=10, loc="left", color=C_TEXT)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figure 3: sensitivity of the exponent to the slope definition
# ---------------------------------------------------------------------------

def fig_sensitivity(sens: dict, cfg: dict, out: Path) -> None:
    cols = list(sens)
    fig, ax = plt.subplots(figsize=(6.5, 4.0), facecolor="white")
    for k, (method, color, off) in enumerate([("OLS", C_SITE["AP"], -0.12), ("RMA", C_SITE["GM"], 0.12)]):
        b = np.array([sens[c][method]["b"] for c in cols])
        lo = np.array([sens[c][method]["ci_b"][0] for c in cols])
        hi = np.array([sens[c][method]["ci_b"][1] for c in cols])
        xs = np.arange(len(cols)) + off
        ax.errorbar(xs, b, yerr=[b - lo, hi - b], fmt="o", color=color, ms=7, capsize=3, lw=1.4, label=f"{method}, all ten tiles")
    ax.axhline(cfg["reference_exponent"], color=C_MUTED, ls=(0, (4, 3)), lw=1.2, label="b = −2 (A S² = constant)")
    ax.axhline(0, color=C_GRID, lw=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.replace("_", " ") + " m" for c in cols], fontsize=8)
    ax.set_ylabel("fitted exponent b (A_crit ∝ S^b), 95 % bootstrap interval")
    ax.set_title("Sensitivity of the exponent to the slope-measurement choice", fontsize=10, loc="left", color=C_TEXT)
    _style(ax)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figure 4: QC maps
# ---------------------------------------------------------------------------

def fig_qc_map(tile, heads: pd.DataFrame, cfg: dict, out: Path) -> None:
    hs = hillshade(tile.z, tile.dx, tile.dy, cfg["hillshade_azimuth_deg"], cfg["hillshade_altitude_deg"])
    ny, nx = tile.shape
    ext = (tile.x0, tile.x0 + nx * tile.dx, tile.y0 - ny * tile.dy, tile.y0)
    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor="white")
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="nearest")
    kept = heads[heads.kept]
    ax.plot(kept.x, kept.y, ".", color=C_KEPT, ms=4, ls="none", label=f"kept (n = {len(kept)})")
    for r in EXCLUSION_REASONS:
        sub = heads[heads.reason == r]
        if len(sub):
            ax.plot(sub.x, sub.y, "x", color=C_REASON[r], ms=4, mew=1.0, ls="none", label=f"{r} (n = {len(sub)})")
    b = cfg["edge_buffer_m"]
    ax.plot([ext[0] + b, ext[1] - b, ext[1] - b, ext[0] + b, ext[0] + b], [ext[2] + b, ext[2] + b, ext[3] - b, ext[3] - b, ext[2] + b],
            color=C_MUTED, lw=0.8, ls=(0, (3, 3)), label=f"{b:g} m edge buffer")
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{tile.site} tile {tile.tile}: partition heads at A_crit = {heads.a_crit_m2.iloc[0]:.3g} m² "
                 f"({len(heads)} detected, {len(kept)} kept)", fontsize=9.5, loc="left", color=C_TEXT)
    ax.legend(loc="lower left", fontsize=7.5, frameon=True, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figure 5: QC profiles
# ---------------------------------------------------------------------------

def fig_qc_profiles(tile, graph, heads: pd.DataFrame, cfg: dict, rng: np.random.Generator, out: Path) -> None:
    kept = heads[heads.kept]
    n = min(cfg["qc_profiles_per_tile"], len(kept))
    if n == 0:
        return
    pick = kept.iloc[rng.choice(len(kept), n, replace=False)]
    L_values = [float(v) for v in cfg["L_values_m"]]
    L_max = max(L_values)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.9 * nrow), facecolor="white", squeeze=False)
    for ax, (_, h) in zip(axes.ravel(), pick.iterrows()):
        du, zu, _ = walk_path(graph, int(h.row), int(h.col), "up", L_max + 5)
        dd, zd, _ = walk_path(graph, int(h.row), int(h.col), "down", L_max / 2 + 5)
        z0 = zu[0]
        ax.plot(np.concatenate([-dd[::-1], du]), np.concatenate([zd[::-1], zu]) - z0, color=C_TEXT, lw=1.4, marker=".", ms=3)
        for L, c in zip(L_values, [C_SITE["AP"], C_SITE["GM"], C_AP5]):
            s = h[f"S_up_{L:g}"]
            ax.plot([0, L], [0, s * L], color=c, lw=1.2, ls=(0, (3, 2)))
            ax.axvline(L, color=c, lw=0.6, alpha=0.5)
            ax.text(L, s * L, f" S_up_{L:g} = {s:.3f}", fontsize=6.5, color=c, va="bottom")
        s = h["S_centred_10"]
        ax.plot([-5, 5], [-5 * s, 5 * s], color=C_MUTED, lw=1.0, ls=(0, (1, 2)))
        ax.text(-5, -5 * s, f"centred 10 m: {s:.3f}", fontsize=6.5, color=C_MUTED, va="top")
        ax.set_title(f"({int(h.row)}, {int(h.col)})  A = {h.area_m2:.3g} m²  S_pixel = {h.S_pixel:.2f}", fontsize=7.5, loc="left", color=C_TEXT)
        ax.set_xlabel("path distance from head (m), + upslope", fontsize=7)
        ax.set_ylabel("z − z_head (m)", fontsize=7)
        _style(ax)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{tile.site} tile {tile.tile}: upslope path profiles at {n} random kept heads "
                 f"(A_crit = {kept.a_crit_m2.iloc[0]:.3g} m²)", fontsize=10, x=0.01, ha="left", color=C_TEXT)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# figure 6 (supplementary): slope-area structure near A_crit
# ---------------------------------------------------------------------------

def fig_slope_area_structure(pseudo: pd.DataFrame, summary: pd.DataFrame, supp: dict, cfg: dict, out: Path,
                             slope_col: str = "S_up_10") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), facecolor="white", sharey=True)
    for ax, site in zip(axes, ("AP", "GM")):
        for t in sorted(summary[summary.site == site].tile):
            sub = pseudo[(pseudo.site == site) & (pseudo.tile == t)].sort_values("threshold_m2")
            c = C_AP5 if (site == "AP" and t == 5) else C_SITE[site]
            alpha = 1.0 if (site == "AP" and t == 5) else 0.45 + 0.1 * t
            ax.plot(sub.threshold_m2, sub[f"{slope_col}_median"], "-", color=c, lw=1.4, alpha=alpha, marker=".", ms=5)
            r = summary[(summary.site == site) & (summary.tile == t)].iloc[0]
            ax.plot([r.A_crit_m2], [r[f"{slope_col}_median"]], M_SITE[site], color=c, ms=9, mec="white", mew=1.0)
            ax.annotate(_label(site, t), (r.A_crit_m2, r[f"{slope_col}_median"]), xytext=(5, 4), textcoords="offset points", fontsize=8, color=C_TEXT)
        ax.axvline(cfg["reference_area_m2"], color=C_MUTED, lw=0.8, ls=(0, (3, 3)))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("area threshold at the pseudo-heads (m²)")
        ax.set_title(f"{site}: median head slope at each threshold (large marker: at A_crit)", fontsize=9, loc="left", color=C_TEXT)
        _style(ax)
    axes[0].set_ylabel("median head slope (m/m)")
    th = supp["within_tile_slope_area"]
    fig.suptitle(f"Within-tile S(A) of head slopes: pooled log-log slope {th['theta_h']:.2f} "
                 f"→ a common S(A) curve alone would give b = −1/θ_h = {th['null_exponent']:.2f}", fontsize=9.5, x=0.01, ha="left", color=C_TEXT)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
