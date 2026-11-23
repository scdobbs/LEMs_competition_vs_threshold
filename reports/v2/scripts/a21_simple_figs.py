#!/usr/bin/env python3
"""Addendum 21, simple figures in the manuscript's style (Steve's request, 2026-11-23), from the stored addendum 21 results:

  simple/landscape_<tile>.png   (Fig. 4 style) per tile: hillshade of the observed DEM and of the AD, P4, PF-C, F-relief and SJ-C surfaces,
                                channel cells of the partitioned models in blue, with K/K0, D/D0 and MSE under each panel
  simple/kd_space.png           (Fig. 5 style) K/K0 against D/D0 for every tile and model (D0 = U/|C_crest|, the measured crest-curvature D
                                at U = 65 m/Myr; bands for D_phys and D_relief); second panel against D_relief
  simple/heads_summary.png      (rule H0/H1 in one picture) median pick-to-channel distance against heads/picks per model, pooled TTV1-TTV3,
                                with the H0 box; and a TTV2 strip of picks against the heads of four surfaces
Colours as the manuscript: partitioned blue, AD orange; models told apart by marker shape and labels.
usage: a21_simple_figs.py
"""
from __future__ import annotations

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a21_common import (  # noqa: E402
    BLOCK_M,
    FIG_DIR,
    H0_DIST_M,
    H0_RATIO,
    MD_TILES,
    TTV_PRIMARY,
    dump,
    hillshade,
    json_ready,
    load_json,
    metre_tile,
    out_path,
)

C_PART, C_AD, INK, INK2, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
OUT = os.path.join(FIG_DIR, "simple")
LABEL = {"obs": "Observed DEM", "AD": "AD  [K, D]", "P4": "P4  tuned A_crit [K, D, A_crit]", "PF_C": "PF-C  field A·S² ≥ C_F [K, D]",
         "PF_A": "PF-A  field A ≥ A_F [K, D]", "F_relief": "F-relief  K₀, D_relief [none free]", "F2_relief": "F2-relief  K₀, D_relief [none free]",
         "SJ_C": "SJ-C  C free [K, D, C]", "F_phys": "F-phys  K₀, D_phys [none free]", "F_form": "F-form  K₀, D_form [none free]"}
MARK = {"AD": ("o", C_AD), "P4": ("o", C_PART), "PF_C": ("s", C_PART), "PF_A": ("^", C_PART), "SJ_C": ("D", C_PART), "F_relief": ("x", C_PART),
        "F2_relief": ("+", C_PART), "obs": ("*", INK), "F_phys": ("x", MUTED), "F_form": ("x", INK2)}


def style(ax):
    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8})
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=7)
    ax.grid(True, which="major", color=GRID, lw=0.4)
    ax.set_axisbelow(True)


def unpack(bits, shape):
    return np.unpackbits(bits)[: shape[0] * shape[1]].reshape(shape).astype(bool)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"path": os.path.relpath(p, os.path.dirname(os.path.dirname(FIG_DIR))), "bytes": os.path.getsize(p)}


def load_tile_results(tile):
    r = load_json(out_path("tiles", f"{tile}.json"))
    s = load_json(out_path("sjc", f"{tile}.json")) if os.path.exists(out_path("sjc", f"{tile}.json")) else None
    return r, s


# ----------------------------------------------------------------------------- landscapes (Fig. 4 style)

def landscape(tile, r, s, out):
    td = metre_tile(tile)
    Z, A, dx = td.Z, td.A, td.dx
    rz = np.load(out_path("tiles", f"{tile}_residuals.npz"))
    hz = np.load(out_path("tiles", f"{tile}_heads.npz"))
    surfaces = {"obs": Z, "AD": Z + rz["AD"], "P4": Z + rz["P4"], "PF_C": Z + rz["PF_C"], "F_relief": Z + rz["F_relief"]}
    chans = {"obs": None, "AD": None, "P4": np.isfinite(A) & (A >= r["rescaled"]["P4"]["A_crit"]), "PF_C": unpack(hz["net_C_1m"], Z.shape), "F_relief": unpack(hz["net_A_1m"], Z.shape)}
    if s is not None:
        surfaces["SJ_C"] = Z + np.load(out_path("sjc", f"{tile}_residual.npz"))["SJ_C"]
        chans["SJ_C"] = unpack(np.load(out_path("sjc", f"{tile}_heads.npz"))["net_SJ_C_1m"], Z.shape)
    names = [n for n in ("obs", "AD", "P4", "PF_C", "F_relief", "SJ_C") if n in surfaces]
    K0, D0 = r["K0"], r["D"]["D_form"]
    fig, axs = plt.subplots(2, 3, figsize=(11, 7.6))
    for ax, name in zip(axs.ravel(), names, strict=False):
        hs = hillshade(surfaces[name], dx)
        ax.imshow(np.nan_to_num(hs, nan=1.0), cmap="gray", vmin=0, vmax=1, interpolation="antialiased")
        if chans[name] is not None:
            from scipy.ndimage import binary_dilation
            ch = binary_dilation(chans[name], iterations=1)
            ax.imshow(np.ma.masked_where(~ch, np.ones(Z.shape)), cmap=matplotlib.colors.ListedColormap([C_PART]), vmin=0, vmax=1, alpha=0.85, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(LABEL[name], loc="left", fontsize=8, color=INK, pad=12)
        if name != "obs":
            mod = s["model"] if name == "SJ_C" else r["models"][name]
            txt = f"MSE {mod['mse']:.0f} m²  ·  K/K₀ {mod['K'] / K0:.2f}  ·  D/D₀ {mod['D'] / D0:.1f}"
            if name == "P4":
                txt += f"  ·  A_crit {mod['A_crit']:,.0f} m²"
            if name == "SJ_C":
                txt += f"  ·  C {mod['C']:,.0f} m² (C_F {r['field_values']['C_F']:,.0f})"
            ax.text(0.0, 1.01, txt, transform=ax.transAxes, fontsize=6.5, color=INK2, va="bottom", ha="left")
    for ax in axs.ravel()[len(names):]:
        ax.axis("off")
    fig.suptitle(f"{tile}: observed and modelled landscapes (hillshade; partitioned channel cells in blue). U = 6.5e-5 m/yr, m = 0.33; "
                 f"K₀ = {K0:.2e}, D₀ = U/|C_crest| = {D0:.4f} m²/yr; D_phys = 0.0025, D_relief = {r['D']['D_relief']:.4f}", fontsize=7.5, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(left=0.02, right=0.99, top=0.9, bottom=0.02, wspace=0.05, hspace=0.2)
    out[f"landscape_{tile}"] = save(fig, f"landscape_{tile}.png")


# ----------------------------------------------------------------------------- K/K0 against D/D0 (Fig. 5 style)

def kd_space(results, out):
    rows = []
    for tile, (r, s) in results.items():
        K0, D0, Dr, Dp = r["K0"], r["D"]["D_form"], r["D"]["D_relief"], r["D"]["D_phys"]
        mods = dict(r["models"])
        if s:
            mods["SJ_C"] = s["model"]
        for name in ("AD", "P4", "PF_A", "PF_C", "SJ_C"):
            if name in mods:
                rows.append({"tile": tile, "model": name, "K_over_K0": mods[name]["K"] / K0, "D_over_D0": mods[name]["D"] / D0, "D_over_D_relief": mods[name]["D"] / Dr,
                             "D_phys_over_D0": Dp / D0, "D_relief_over_D0": Dr / D0, "mse": mods[name]["mse"]})
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, xcol, xlab in zip(axs, ("D_over_D0", "D_over_D_relief"), ("D / D₀   (D₀ = U / |C_crest|, the measured crest-curvature D)", "D / D_relief   (D_relief = U L_H² / 2R, the measured hillslope-relief D)"), strict=True):
        style(ax)
        ax.set_xscale("log")
        ax.set_yscale("log")
        if xcol == "D_over_D0":
            dp = [q["D_phys_over_D0"] for q in rows]
            dr = [q["D_relief_over_D0"] for q in rows]
            ax.axvspan(min(dp), max(dp), color="#f3d9c9", alpha=0.6, lw=0, zorder=0)
            ax.axvspan(min(dr), max(dr), color="#d5e4f7", alpha=0.7, lw=0, zorder=0)
            ax.text(np.sqrt(min(dp) * max(dp)), 0.3, "D_phys / D₀\n(Heimsath)", ha="center", va="bottom", fontsize=6.5, color=INK2)
            ax.text(np.sqrt(min(dr) * max(dr)), 0.3, "D_relief / D₀\n(hillslope relief)", ha="center", va="bottom", fontsize=6.5, color=INK2)
        else:
            ax.axvspan(0.67, 1.5, color="#d5e4f7", alpha=0.7, lw=0, zorder=0)
            ax.text(1.0, 0.3, "rule DX band\n0.67-1.5", ha="center", va="bottom", fontsize=6.5, color=INK2)
        ax.axhline(1, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.axvline(1, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        ax.plot(1, 1, marker="+", ms=12, mew=1.4, color=INK, zorder=7)
        for name in ("AD", "P4", "PF_A", "PF_C", "SJ_C"):
            mk, col = MARK[name]
            xs = [q[xcol] for q in rows if q["model"] == name]
            ys = [q["K_over_K0"] for q in rows if q["model"] == name]
            ax.plot(xs, ys, marker=mk, ls="none", ms=7, mfc=col, mec="white" if name == "AD" else col, mew=0.6, alpha=0.9, label=LABEL[name], zorder=5)
        ax.set_xlabel(xlab, color=INK)
        ax.set_ylabel("K / K₀   (K₀ = U / k_s)", color=INK)
        ax.set_xlim(0.03, 40)
        ax.set_ylim(0.25, 40)
    axs[0].legend(fontsize=7, frameon=False, loc="upper left")
    fig.suptitle("Fitted K/K₀ against D/D₀ on the 11 TTV tiles (one point per tile and model). The cross is the measured pair (K₀, D₀); "
                 "a model that recovers the measurements plots there.", fontsize=8.5, color=INK, x=0.01, ha="left")
    fig.subplots_adjust(left=0.06, right=0.99, top=0.9, bottom=0.12, wspace=0.18)
    out["kd_space"] = save(fig, "kd_space.png")
    return rows


# ----------------------------------------------------------------------------- heads (one picture)

def heads_summary(results, out):
    rules = load_json(out_path("rules", "rules.json"))
    H1 = rules["H1"]
    fig = plt.figure(figsize=(12, 8.2))
    gs = fig.add_gridspec(2, 4, height_ratios=(1.15, 1.0), hspace=0.35, wspace=0.08)
    ax = fig.add_subplot(gs[0, :])
    style(ax)
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0, H0_RATIO[0]), H0_DIST_M, H0_RATIO[1] - H0_RATIO[0], facecolor="#e4efdc", edgecolor="#8fbf7a", lw=0.8, zorder=0))
    ax.text(H0_DIST_M / 2, H0_RATIO[0] * 1.03, "rules H0 / H1 hold inside this box:\nmedian distance ≤ 25 m and 0.67 ≤ heads/picks ≤ 1.5", ha="center", va="bottom", fontsize=7, color="#3a6b2a")
    offsets = {"obs": (-28, 14), "PF_A": (-30, -16), "F2_relief": (22, 12), "PF_C": (10, -16), "F_relief": (8, 10), "SJ_C": (0, -16), "P4": (14, 8), "AD": (0, 12), "F_phys": (18, 0), "F_form": (18, 0)}
    pts = []
    for name in ("obs", "F_phys", "F_form", "F_relief", "F2_relief", "PF_A", "PF_C", "SJ_C", "P4", "AD"):
        h = H1[name]["a2"]
        mk, col = MARK[name]
        x, y = h["median_dist_m"], h["ratio"]
        ax.plot(x, y, marker=mk, ms=13 if name == "obs" else 9, ls="none", mfc=col if name not in ("F_relief", "F2_relief", "F_phys", "F_form") else "none", mec=col, mew=1.8, zorder=5)
        ax.annotate(LABEL[name].split("  ")[0], (x, y), xytext=offsets[name], textcoords="offset points", fontsize=7.5, color=INK, ha="center", va="center",
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.5, "shrinkB": 5})
        pts.append({"model": name, "median_dist_m": x, "ratio": y, "heads": h["n_heads"], "picks": h["n_picks"]})
    ax.set_yscale("log")
    ax.set_xlim(-2, 60)
    ax.set_ylim(0.5, 30)
    ax.set_yticks([0.5, 1, 2, 5, 10, 20])
    ax.set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.set_xlabel("median distance from a mapped head to the model's nearest channel (m), TTV1-TTV3 pooled", color=INK)
    ax.set_ylabel("model heads / mapped heads (inside the M&D area)", color=INK)
    ax.set_title("Where each model puts its channel heads, against the 61 heads M&D mapped (criterion A₁₂·S₁₂² ≥ C_F applied to each 12 m surface)", loc="left", fontsize=8.5, color=INK)
    # TTV2 strip: picks against heads, four surfaces
    tile = "TTV2"
    r, s = results[tile]
    hz = np.load(out_path("tiles", f"{tile}_heads.npz"))
    hs_ = np.load(out_path("sjc", f"{tile}_heads.npz")) if s else None
    pr, pc = hz["pick_rc"]
    poly12 = hz["poly12"]
    for j, name in enumerate(("obs", "PF_C", "SJ_C", "AD")):
        axm = fig.add_subplot(gs[1, j])
        src = hs_ if name == "SJ_C" else hz
        if src is None:
            axm.axis("off")
            continue
        Z12 = src[f"{name}_Z12"]
        hd = src[f"{name}_a2_heads"]
        axm.imshow(np.nan_to_num(hillshade(Z12.astype(float), float(BLOCK_M)), nan=1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axm.contour(poly12.astype(float), levels=[0.5], colors="k", linewidths=0.5, linestyles="--")
        hr, hc = np.nonzero(hd)
        mk, col = MARK[name]
        axm.plot(hc, hr, "o", mfc=col if name != "obs" else C_PART, mec="white", mew=0.4, ms=5, label="model heads")
        axm.plot(pc / BLOCK_M - 0.5, pr / BLOCK_M - 0.5, "^", color="#D55E00", ms=7, mec="k", mew=0.5, label="mapped heads (Steve's picks)")
        e = (s["extraction"] if name == "SJ_C" else r["extraction"][name])["a2"]
        axm.set_title(f"{LABEL[name].split('  ')[0]}: {e['n_heads_in_polygon']} heads / {e['n_picks_in_polygon']} picks, {e['pick_dist_m']['median']:.0f} m", fontsize=7.5, color=INK, loc="left")
        axm.set_xticks([])
        axm.set_yticks([])
        if j == 0:
            axm.legend(fontsize=6.5, loc="lower left", framealpha=0.9)
    fig.text(0.01, 0.005, "Bottom: TTV2 at 12 m (dashed = M&D study area). Each dot is a head the surface's own criterion makes; triangles are the mapped heads.", fontsize=7, color=INK2)
    out["heads_summary"] = save(fig, "heads_summary.png")
    return pts


def main():
    t0 = time.time()
    results = {t: load_tile_results(t) for t in TTV_PRIMARY if os.path.exists(out_path("tiles", f"{t}.json"))}
    out: dict = {}
    for t in TTV_PRIMARY:
        if t in results and "--heads-only" not in sys.argv:
            landscape(t, *results[t], out)
            print(t, "landscape", f"[{time.time() - t0:.0f} s]", flush=True)
    kd = kd_space(results, out)
    pts = heads_summary(results, out)
    if "--heads-only" in sys.argv:
        print(out["heads_summary"])
        return
    dump(json_ready({"figures": out, "kd_points": kd, "head_points": pts}), out_path("figures", "simple_figures.json"))
    print({k: round(v["bytes"] / 1e6, 2) for k, v in out.items()}, f"[{time.time() - t0:.0f} s]")
    _ = MD_TILES


if __name__ == "__main__":
    main()
