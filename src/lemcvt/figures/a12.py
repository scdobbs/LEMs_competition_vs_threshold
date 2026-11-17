"""Addendum 12 figures (equal-parameter comparisons and per-tile m), built from results only.

a12_landscape   per tile: hillshade of the observed DEM and of every variant's optimum surface (site m), and of the
                joint and AD optima at m_t; channel cells of the partitioned surfaces in blue
a12_residual    per tile: the same panels as model - observed elevation (diverging, symmetric, shared scale)
a12_kd          K/K0 against D/D0 of every variant, site m (filled) and m_t (hollow, arrow from the site-m point);
                AP against D0_A and D0_B, GM against D0_A
a12_mse         MSE by variant and parameter count (in-sample) and the held-out MSE, per tile

Colours: partitioned = categorical slot 1 (blue), AD = slot 2 (orange) (validated pair, CVD dE 24.7); the diverging
map is blue - neutral grey - red with the neutral at zero. Variants are told apart by marker shape and labels, never by
colour alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lemcvt.figures.common import load_params, plt, require_run, save, set_nature_style

C_PART = "#2a78d6"
C_AD = "#eb6834"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
DIV_NEG, DIV_MID, DIV_POS = "#2a78d6", "#f0efec", "#e34948"

TILES = ("AP1", "AP2", "AP3", "AP4", "AP5", "GM1", "GM2", "GM3", "GM4", "GM5")

LABEL = {
    "obs": "Observed DEM",
    "P1_A": "P1  stage 2: K₀, D₀,A fixed [A_crit]",
    "P1_B": "P1  stage 2: K₀, D₀,B fixed [A_crit]",
    "P2_A": "P2  D = D₀,A [K, A_crit]",
    "P2_B": "P2  D = D₀,B [K, A_crit]",
    "P3": "P3  K = K₀ [D, A_crit]",
    "P4": "P4  joint [K, D, A_crit]",
    "A1": "A1  AD, K = K₀ [D]",
    "A2_A": "A2  AD, D = D₀,A [K]",
    "A2_B": "A2  AD, D = D₀,B [K]",
    "AD": "AD  [K, D]",
    "P4_mt": "P4 joint at m_t",
    "AD_mt": "AD at m_t",
}
LAYOUT: dict[str, list[list[str | None]]] = {
    "AP": [["obs", "P4", "P4_mt", "AD", "AD_mt"], ["P1_A", "P1_B", "P2_A", "P2_B", "P3"], ["A1", "A2_A", "A2_B", None, None]],
    "GM": [["obs", "P4", "P4_mt", "AD", "AD_mt"], ["P1_A", "P2_A", "P3", None, None], ["A1", "A2_A", None, None, None]],
}
ROW_TITLES = ("Best fits: site m and per-tile m_t", "Partitioned, constrained (site m)", "AD, constrained (site m)")


def _diverging():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("a12_div", [DIV_NEG, DIV_MID, DIV_POS])


def _style(ax) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2)
    ax.grid(True, which="major", color=GRID, lw=0.4)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------------- landscapes

def _surface_panels(tile: str) -> tuple[Path, dict[str, Any], dict[str, dict[str, Any]]]:
    from lemcvt.results import find_runs

    runs = find_runs("a12_surfaces", tile)
    if not runs:
        raise FileNotFoundError(f"no a12_surfaces run for {tile}; run `lemcvt a12 surfaces` first")
    run = runs[-1]
    p = load_params(run)
    panels: dict[str, dict[str, Any]] = {}
    for name, s in p["surfaces"].items():
        panels[name] = {"elevation": run / s["elevation"], "channels": run / s["channels"] if s["channels"] else None,
                        "mse": s["mse_rebuilt"], "params": s["params"], "m": p["model"]["m"]}
    for name, s in p.get("m_t_surfaces", {}).items():
        panels[name] = {"elevation": Path(s["elevation"]), "channels": run / s["channels"] if s["channels"] else None, "mse": s["mse"],
                        "params": s["params"], "m": s["m"]}
    return run, p, panels


def _params_text(name: str, pan: dict[str, Any], model: dict[str, Any]) -> str:
    pr = pan["params"]
    bits = [f"MSE {pan['mse']:.1f} m²"]
    if "A_crit" in pr:
        bits.append(f"A_crit {pr['A_crit']:.0f} m²")
    if name.endswith("_mt"):
        bits.append(f"m {pan['m']:.2f}")
    return " · ".join(bits)


def landscapes(out_dir: Path, tiles=TILES) -> list[Path]:
    from scipy.ndimage import binary_dilation

    from lemcvt.config import load_tile
    from lemcvt.hydro.derived import hillshade
    from lemcvt.io.grids import read_geotiff

    set_nature_style()
    paths: list[Path] = []
    for tile in tiles:
        try:
            run, p, panels = _surface_panels(tile)
        except FileNotFoundError:
            continue
        td = load_tile(tile)
        Z0, dx = td.Z, td.dx
        site = tile[:2]
        layout = LAYOUT[site]
        grids: dict[str, np.ndarray] = {"obs": Z0}
        chans: dict[str, np.ndarray | None] = {"obs": None}
        for name, pan in panels.items():
            grids[name] = read_geotiff(pan["elevation"]).data.astype(float)
            chans[name] = read_geotiff(pan["channels"]).data.astype(bool) if pan["channels"] else None
        res = {k: v - Z0 for k, v in grids.items() if k != "obs"}
        # A1 (K = K0) misfits by tens of metres; it would set the scale for every panel, so it is excluded and saturates
        lim = float(np.nanpercentile(np.abs(np.concatenate([r[np.isfinite(r)].ravel() for k, r in res.items() if k != "A1"])), 98))
        rows_cap: list[dict[str, Any]] = []
        for kind in ("landscape", "residual"):
            fig, axs = plt.subplots(3, 5, figsize=(12.5, 8.6))
            fig.patch.set_facecolor("white")
            im = None
            for i, row in enumerate(layout):
                for j, cell in enumerate(row):
                    ax = axs[i, j]
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for s in ax.spines.values():
                        s.set_visible(False)
                    if cell is None or cell not in grids:
                        ax.axis("off")
                        continue
                    name = cell
                    hs = hillshade(grids[name], dx)
                    if kind == "landscape" or name == "obs":
                        ax.imshow(hs, cmap="gray", vmin=0, vmax=255, interpolation="antialiased")
                        ch = chans.get(name)
                        if kind == "landscape" and ch is not None:
                            chd = binary_dilation(ch, iterations=1)
                            ax.imshow(np.ma.masked_where(~chd, np.ones_like(hs)), cmap=_single(C_PART), vmin=0, vmax=1,
                                      interpolation="nearest", alpha=0.85)
                    else:
                        im = ax.imshow(res[name], cmap=_diverging(), vmin=-lim, vmax=lim, interpolation="antialiased")
                    ax.set_title(LABEL[name], fontsize=7.5, color=INK, loc="left", pad=11)
                    sub = "" if name == "obs" else _params_text(name, panels[name], p["model"])
                    ax.text(0.0, 1.012, sub, transform=ax.transAxes, fontsize=6.3, color=INK2, va="bottom", ha="left")
                    if kind == "landscape":
                        rows_cap.append({"tile": tile, "panel": name, "label": LABEL[name], "mse_m2": None if name == "obs" else panels[name]["mse"],
                                         **({} if name == "obs" else {f"param_{k}": v for k, v in panels[name]["params"].items()}),
                                         "m": None if name == "obs" else panels[name]["m"],
                                         "grid": "observed (config)" if name == "obs" else str(panels[name]["elevation"]),
                                         "residual_p98_abs_m": lim})
            for i, t in enumerate(ROW_TITLES):
                axs[i, 0].annotate(t, xy=(-0.06, 0.5), xycoords="axes fraction", rotation=90, va="center", ha="right", fontsize=8, color=INK)
            what = ("hillshade; partitioned channel cells (A ≥ A_crit, dilated by one cell for visibility) in blue" if kind == "landscape"
                    else f"model − observed elevation (m), shared symmetric scale ±{lim:.1f} m (98th percentile of |residual| without A1, which saturates); observed as hillshade")
            fig.suptitle(f"{tile}: {what}", fontsize=8.5, color=INK, x=0.01, ha="left")
            fig.text(0.01, 0.005, f"site m = {p['model']['m']:.2f}; K₀ = {p['model']['K0']:.3e} 1/yr; D₀,A = {p['model']['D0_A']:.3e}"
                     + (f", D₀,B = {p['model']['D0_B']:.3e}" if p['model'].get('D0_B') else "") + f" m²/yr · results: {run.name}",
                     fontsize=6, color=MUTED)
            fig.subplots_adjust(left=0.04, right=0.93 if kind == "residual" else 0.995, top=0.92, bottom=0.03, wspace=0.05, hspace=0.32)
            if kind == "residual" and im is not None:
                cax = fig.add_axes((0.945, 0.25, 0.012, 0.5))
                cb = fig.colorbar(im, cax=cax)
                cb.set_label("model − observed (m)", color=INK2, fontsize=7)
                cb.ax.tick_params(labelsize=6.5, colors=INK2)
            paths += save(fig, out_dir, f"a12_{kind}_{tile}", caption=rows_cap if kind == "landscape" else
                          [{"tile": tile, "panel": r["panel"], "residual_p98_abs_m": lim} for r in rows_cap])
    if not paths:
        raise FileNotFoundError("no a12_surfaces runs")
    return paths


def _single(hexcol: str):
    from matplotlib.colors import ListedColormap

    return ListedColormap([hexcol])


# ----------------------------------------------------------------------------- K/K0 against D/D0

def _summary() -> tuple[Path, dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    run = require_run("a12_summary")
    p = load_params(run)
    return run, p, pd.read_parquet(run / "insample.parquet"), pd.read_parquet(run / "side_by_side.parquet"), pd.read_parquet(run / "heldout.parquet")


def kd(out_dir: Path) -> list[Path]:
    from lemcvt.config import load_tile_config

    set_nature_style()
    run, p, ins, side, _held = _summary()
    rows: list[dict[str, Any]] = []
    for _, r in ins.iterrows():
        cfg = load_tile_config(r["tile"])
        K0, D0A, D0B = float(cfg.model.K0_partitioned), float(cfg.model.D0), cfg.model.D0_B
        for v in ("P1_A", "P1_B", "P2_A", "P2_B", "P3", "P4", "A1", "A2_A", "A2_B", "AD"):
            if f"{v}_K" not in r or pd.isna(r.get(f"{v}_K")):
                continue
            rows.append({"tile": r["tile"], "site": r["tile"][:2], "variant": v, "m": "site", "K_over_K0": r[f"{v}_K"] / K0,
                         "D_over_D0_A": r[f"{v}_D"] / D0A, "D_over_D0_B": (r[f"{v}_D"] / D0B) if D0B else np.nan, "mse": r[f"{v}_mse"],
                         "D_at_bound": bool(r.get(f"{v}_D_at_bound", False))})
    for _, r in side[side.which == "m_t"].iterrows():
        rows.append({"tile": r["tile"], "site": r["site"], "variant": "P4", "m": "m_t", "K_over_K0": r["K_over_K0"], "D_over_D0_A": r["D_over_D0_A"],
                     "D_over_D0_B": r["D_over_D0_B"], "mse": r["mse_partitioned"], "D_at_bound": False, "m_value": r["m"]})
        rows.append({"tile": r["tile"], "site": r["site"], "variant": "AD", "m": "m_t", "K_over_K0": r["AD_K_over_K0"], "D_over_D0_A": r["AD_D_over_D0_A"],
                     "D_over_D0_B": r["AD_D_over_D0_B"], "mse": r["mse_AD"], "D_at_bound": r["AD_where"] != "interior", "m_value": r["m"]})
    df = pd.DataFrame(rows)
    marker = {"P2": "D", "P3": "^", "P4": "o", "A1": "v", "A2": "D", "AD": "o"}
    x_lo, x_hi = 1e-3, 50.0
    row_spec = [("AP", "A"), ("AP", "B"), ("GM", "A")]
    fig, axs = plt.subplots(3, 5, figsize=(13.0, 8.4), sharex=True, sharey=True)
    for i, (site, end) in enumerate(row_spec):
        dcol = f"D_over_D0_{end}"
        tiles = [t for t in TILES if t.startswith(site)]
        for j, tile in enumerate(tiles):
            ax = axs[i, j]
            _style(ax)
            d = df[(df.tile == tile) & df.variant.isin(["P2_" + end, "P3", "P4", "A1", "A2_" + end, "AD"])]
            ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
            ax.axvline(1, color=MUTED, lw=0.6, ls=(0, (3, 2)), zorder=1)
            ax.plot(1, 1, marker="+", ms=11, mew=1.3, color=INK, zorder=7)
            for v, col in (("P4", C_PART), ("AD", C_AD)):
                a = d[(d.variant == v) & (d.m == "site")]
                b = d[(d.variant == v) & (d.m == "m_t")]
                if len(a) and len(b):
                    pa = (float(np.clip(a[dcol].iloc[0], x_lo, x_hi)), a.K_over_K0.iloc[0])
                    pb = (float(np.clip(b[dcol].iloc[0], x_lo, x_hi)), b.K_over_K0.iloc[0])
                    ax.annotate("", xy=pb, xytext=pa, arrowprops={"arrowstyle": "-|>", "color": col, "lw": 0.8, "shrinkA": 4, "shrinkB": 5,
                                                                   "mutation_scale": 8}, zorder=3)
            for _, q in d.iterrows():
                fam = C_PART if q.variant.startswith("P") else C_AD
                base = q.variant.split("_")[0]
                x = q[dcol]
                mk = "<" if x < x_lo else (">" if x > x_hi else marker[base])
                ax.plot(float(np.clip(x, x_lo, x_hi)), q.K_over_K0, marker=mk, ms=7 if base in ("P4", "AD") else 6, ls="none",
                        mfc="white" if q.m == "m_t" else fam, mec=fam, mew=1.1, zorder=5 if base in ("P4", "AD") else 4)
            mt = d[(d.m == "m_t")]
            m_txt = f"m_t {mt.m_value.iloc[0]:.2f}" if len(mt) and "m_value" in mt and pd.notna(mt.m_value.iloc[0]) else ""
            ax.set_title(f"{tile}  ·  vs D₀,{end}" + (f"  ·  {m_txt}" if m_txt else ""), loc="left", fontsize=7.5, color=INK)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(x_lo * 0.6, x_hi * 1.6)
            ax.set_ylim(0.5, 25)
            if i == 2:
                ax.set_xlabel(f"D / D₀,{end}", color=INK)
            if j == 0:
                ax.set_ylabel("K / K₀", color=INK)
        for j in range(len(tiles), 5):
            axs[i, j].axis("off")
    for i, (site, end) in enumerate(row_spec):
        axs[i, 0].annotate(f"{site}, D₀,{end}", xy=(-0.33, 0.5), xycoords="axes fraction", rotation=90, va="center", ha="right", fontsize=8.5, color=INK)
    from matplotlib.lines import Line2D

    h = [Line2D([], [], marker="o", ls="none", mfc=C_PART, mec=C_PART, label="P4 joint [K, D, A_crit]"),
         Line2D([], [], marker="^", ls="none", mfc=C_PART, mec=C_PART, label="P3 K = K₀ [D, A_crit]"),
         Line2D([], [], marker="D", ls="none", mfc=C_PART, mec=C_PART, label="P2 D = D₀ [K, A_crit]"),
         Line2D([], [], marker="o", ls="none", mfc=C_AD, mec=C_AD, label="AD [K, D]"),
         Line2D([], [], marker="v", ls="none", mfc=C_AD, mec=C_AD, label="A1 K = K₀ [D]"),
         Line2D([], [], marker="D", ls="none", mfc=C_AD, mec=C_AD, label="A2 D = D₀ [K]"),
         Line2D([], [], marker="o", ls="none", mfc="white", mec=INK2, label="hollow = at m_t, arrow from the site-m point"),
         Line2D([], [], marker="<", ls="none", mfc=INK2, mec=INK2, label=f"off scale (D/D₀ < {x_lo:g} or > {x_hi:g}), drawn at the edge"),
         Line2D([], [], marker="+", ls="none", mec=INK, ms=9, label="measured (K₀, D₀); P1 sits here by construction")]
    fig.legend(handles=h, loc="lower center", ncol=5, frameon=False, fontsize=6.8, bbox_to_anchor=(0.5, -0.005))
    fig.suptitle("Fitted K/K₀ against D/D₀ per tile: filled = site m, hollow = per-tile m_t (K₀ re-measured at that m)", fontsize=9,
                 color=INK, x=0.01, ha="left")
    fig.text(0.99, 0.003, f"results: {run.name}", fontsize=5.5, color=MUTED, ha="right")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.14, wspace=0.08, hspace=0.25)
    return save(fig, out_dir, "a12_kd", caption=df)


# ----------------------------------------------------------------------------- MSE by variant and count

def mse(out_dir: Path) -> list[Path]:
    set_nature_style()
    run, p, ins, side, held = _summary()
    groups = [("count 1: one free parameter", [("P1_A", C_PART, "s", True), ("P1_B", C_PART, "s", False), ("A1", C_AD, "v", True),
                                               ("A2_A", C_AD, "D", True), ("A2_B", C_AD, "D", False)]),
              ("count 2: two free parameters", [("P2_A", C_PART, "D", True), ("P2_B", C_PART, "D", False), ("P3", C_PART, "^", True),
                                                ("AD", C_AD, "o", True)]),
              ("count 3 (P4) against AD", [("P4", C_PART, "o", True), ("AD", C_AD, "o", True)]),
              ("held-out (fit one half, score the other, mean)", [("P4", C_PART, "o", True), ("P2_A", C_PART, "D", True),
                                                                  ("P2_B", C_PART, "D", False), ("AD", C_AD, "o", True)])]
    tiles = list(ins.tile)
    xpos = {t: i + (0.8 if t.startswith("GM") else 0.0) for i, t in enumerate(tiles)}
    fig, axs = plt.subplots(1, 4, figsize=(13.5, 5.6), sharey=True)
    cap = []
    for k, (ax, (title, vars_)) in enumerate(zip(axs, groups, strict=True)):
        _style(ax)
        for v, col, mk, filled in vars_:
            for t in tiles:
                present = [w for w, *_ in vars_ if not (t.startswith("GM") and w.endswith("_B"))]
                if v not in present:
                    continue
                off = (present.index(v) - (len(present) - 1) / 2) * 0.16
                if k == 3:
                    src = held[held.tile == t]
                    val = src[f"{v}_heldout"].iloc[0] if f"{v}_heldout" in src else np.nan
                else:
                    src = ins[ins.tile == t]
                    val = src[f"{v}_mse"].iloc[0] if f"{v}_mse" in src else np.nan
                if not np.isfinite(val):
                    continue
                ax.plot(xpos[t] + off, val, marker=mk, ms=5.5, ls="none", mfc=col if filled else "white", mec=col, mew=1.0, zorder=4)
                cap.append({"panel": title, "tile": t, "variant": v, "mse_m2": val})
        ax.set_yscale("log")
        ax.set_xticks([xpos[t] for t in tiles])
        ax.set_xticklabels(tiles, rotation=90, fontsize=6.5, color=INK2)
        ax.set_title(title, loc="left", fontsize=7.5, color=INK)
        ax.axvline(4.9, color=GRID, lw=0.8)
        handles = []
        from matplotlib.lines import Line2D

        for v, col, mk, filled in vars_:
            handles.append(Line2D([], [], marker=mk, ls="none", mfc=col if filled else "white", mec=col, label=LABEL[v]))
        ax.legend(handles=handles, fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(0.0, -0.13), ncol=1)
    axs[0].set_ylabel("MSE (m²), log scale", color=INK)
    fig.suptitle("MSE of every variant at the site m, by number of free parameters, and held-out (GM: D₀,B variants omitted, within 5 % of D₀,A)",
                 fontsize=8.5, color=INK, x=0.01, ha="left")
    fig.text(0.99, 0.005, f"results: {run.name}", fontsize=5.5, color=MUTED, ha="right")
    fig.subplots_adjust(left=0.05, right=0.995, top=0.9, bottom=0.3, wspace=0.06)
    return save(fig, out_dir, "a12_mse", caption=cap)


def build_all(out_dir: Path) -> list[Path]:
    return kd(out_dir) + mse(out_dir) + landscapes(out_dir)


__all__ = ["build_all", "kd", "landscapes", "mse"]
