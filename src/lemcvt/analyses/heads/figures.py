"""QC figures for the head-extraction checkpoint (all from code)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from lemcvt.analyses.width.qc import hillshade  # noqa: E402

from .paths import walk_path  # noqa: E402

C_TEXT, C_MUTED, C_GRID = "#0b0b0b", "#52514e", "#efeeea"


def _style(ax):
    ax.grid(color=C_GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=C_MUTED, labelsize=8)

# dataviz reference palette, categorical slots (validated adjacent pairs)
C_CLASS = {"hollow": "#2a78d6", "valley_floor": "#4a3aa7", "bench_plateau": "#e34948",
           "rill": "#eb6834", "road_ditch": "#eda100", "unclassified": "#1baf7a"}
M_METHOD = {"P": "o", "D": "^", "W": "D"}
NAME_METHOD = {"P": "Method P (Pelletier, Python)", "D": "Method D (DrEICH, LSDTT)", "W": "LSDTT Wiener (comparison)"}


def fig_qc_map(tile, heads: pd.DataFrame, partition: pd.DataFrame | None, qc: dict, ext: dict, out: Path) -> None:
    hs = hillshade(tile.z, tile.dx, tile.dy, qc["hillshade_azimuth_deg"], qc["hillshade_altitude_deg"])
    ny, nx = tile.shape
    extent = (tile.x0, tile.x0 + nx * tile.dx, tile.y0 - ny * tile.dy, tile.y0)
    fig, ax = plt.subplots(figsize=(9.0, 8.4), facecolor="white")
    ax.imshow(hs, cmap="gray", vmin=0, vmax=1, extent=extent, interpolation="nearest")
    a = np.where(tile.area >= qc["area_overlay_min_m2"], tile.area, np.nan)
    im = ax.imshow(a, cmap="YlOrBr", norm=LogNorm(vmin=qc["area_overlay_min_m2"], vmax=max(1e6, np.nanmax(a))),
                   extent=extent, interpolation="nearest", alpha=0.85)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01)
    cb.set_label("drainage area A (m²), pixels with A ≥ 10³ m²", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    if partition is not None and len(partition):
        ax.plot(partition.x, partition.y, ".", color="#9a9995", ms=4, ls="none", alpha=0.9,
                label=f"Test 1 partition heads at final A_crit (n = {len(partition)})")
    ret = heads[heads.retained]
    for m in [m for m in ("P", "D", "W") if (ret.method == m).any()]:
        for cls, c in C_CLASS.items():
            sub = ret[(ret.method == m) & (ret.head_class == cls)]
            if len(sub):
                ax.plot(sub.x, sub.y, M_METHOD[m], color=c, ms=6 if m == "P" else 5, mec="white", mew=0.6, ls="none",
                        label=f"{m}: {cls} (n = {len(sub)})")
    drp = heads[~heads.retained]
    if len(drp):
        ax.plot(drp.x, drp.y, "x", color="#52514e", ms=3.5, mew=0.8, ls="none", label=f"dropped, edge/path (n = {len(drp)})")
    b = ext["edge_buffer_m"]
    ax.plot([extent[0] + b, extent[1] - b, extent[1] - b, extent[0] + b, extent[0] + b],
            [extent[2] + b, extent[2] + b, extent[3] - b, extent[3] - b, extent[2] + b], color=C_MUTED, lw=0.8, ls=(0, (3, 3)))
    ax.plot([extent[0] + 40, extent[0] + 140], [extent[2] + 40, extent[2] + 40], color="white", lw=3)
    ax.text(extent[0] + 90, extent[2] + 48, "100 m", color="white", ha="center", fontsize=8)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    nP = int(((heads.method == "P") & heads.retained).sum())
    ax.set_title(f"{tile.site} tile {tile.tile}: extracted heads (circles = Method P, triangles = Method D, diamonds = LSDTT Wiener); "
                 f"{nP} P heads retained; trunk = A ≥ {qc['trunk_area_m2']:g} m²", fontsize=8.5, loc="left", color=C_TEXT)
    ax.legend(loc="lower right", fontsize=6.5, frameon=True, framealpha=0.9, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_profiles(tile, graph, heads: pd.DataFrame, qc: dict, ext: dict, rng: np.random.Generator, out: Path,
                 cls: str = "hollow", method: str = "P") -> int:
    sub = heads[heads.retained & (heads.head_class == cls) & (heads.method == method)]
    n = min(qc["qc_profiles_per_tile"], len(sub))
    if n == 0:
        return 0
    pick = sub.iloc[rng.choice(len(sub), n, replace=False)]
    L_values = [float(v) for v in ext["L_values_m"]]
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.9 * nrow), facecolor="white", squeeze=False)
    cols = ["#2a78d6", "#eb6834", "#4a3aa7"]
    for ax, (_, h) in zip(axes.ravel(), pick.iterrows(), strict=False):
        du, zu, _ = walk_path(graph, int(h.row), int(h.col), "up", max(L_values) + 5)
        dd, zd, _ = walk_path(graph, int(h.row), int(h.col), "down", 15.0)
        z0 = zu[0]
        ax.plot(np.concatenate([-dd[::-1], du]), np.concatenate([zd[::-1], zu]) - z0, color=C_TEXT, lw=1.4, marker=".", ms=3)
        for L, c in zip(L_values, cols, strict=False):
            s = h[f"S_up_{L:g}"]
            if np.isfinite(s):
                ax.plot([0, L], [0, s * L], color=c, lw=1.2, ls=(0, (3, 2)))
                ax.axvline(L, color=c, lw=0.6, alpha=0.5)
                ax.text(L, s * L, f" {L:g} m: {s:.3f}", fontsize=6.5, color=c, va="bottom")
        ax.set_title(f"({int(h.row)}, {int(h.col)})  A = {h.area_D8_m2:.3g} m²  κ10 = {h.plan_curvature_10m:.3f}", fontsize=7.5, loc="left", color=C_TEXT)
        ax.set_xlabel("path distance from head (m), + upslope", fontsize=7)
        ax.set_ylabel("z − z_head (m)", fontsize=7)
        _style(ax)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(f"{tile.site} tile {tile.tile}: {n} random {cls}-class {NAME_METHOD.get(method, method)} heads", fontsize=10, x=0.01, ha="left", color=C_TEXT)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return n


def fig_threshold_sensitivity(sens: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2), facecolor="white")
    if "smoothing" in sens:
        sens = pd.DataFrame(sens[sens["smoothing"] == "OWF"])
    piv = sens.pivot_table(index=["site", "tile"], columns="kt", values="n_retained")
    x = np.arange(len(piv))
    w = 0.26
    for k, (kt, c) in enumerate(zip(sorted(piv.columns), ["#2a78d6", "#eb6834", "#4a3aa7"], strict=False)):
        ax.bar(x + (k - 1) * w, piv[kt].to_numpy(), width=w, color=c, label=f"κt = {kt:g} m⁻¹")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}{t}" for s, t in piv.index], fontsize=8)
    ax.set_ylabel("Method P heads retained (after edge and path drops)")
    ax.set_title("Head counts against the contour-curvature threshold", fontsize=10, loc="left", color=C_TEXT)
    _style(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
