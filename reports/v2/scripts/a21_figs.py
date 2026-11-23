#!/usr/bin/env python3
"""Addendum 21 figures (Part B item 6): per tile (TTV1-TTV3) one panel per model with the 12 m hillshade of the model surface, the
extracted channels (alpha = 2 at C_F), the extracted heads, Steve's picks and the M&D traced lines; per tile (all 11) the elevation
residual maps of SJ-C, P4 and AD on one colour scale; the SJ-C profiles MSE(C); the D decomposition. Figures in reports/v2/figures/a21/.
usage: a21_figs.py
"""
from __future__ import annotations

import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a21_common import (  # noqa: E402
    BLOCK_M,
    FIG_DIR,
    MD_TILES,
    MODEL_ORDER,
    TTV_PRIMARY,
    dump,
    hillshade,
    json_ready,
    load_json,
    out_path,
)

LABEL = {"obs": "observed lidar", "F_phys": "F-phys", "F_form": "F-form", "F_relief": "F-relief", "F2_phys": "F2-phys", "F2_form": "F2-form", "F2_relief": "F2-relief",
         "PF_A": "PF-A", "PF_C": "PF-C", "SJ_C": "SJ-C", "P4": "P4", "AD": "AD"}


def unpack(bits: np.ndarray, shape) -> np.ndarray:
    return np.unpackbits(bits)[: shape[0] * shape[1]].reshape(shape).astype(bool)


def save(fig, path):
    for sc in (1.0, 0.8, 0.6, 0.45):
        fig.savefig(path, dpi=int(160 * sc), bbox_inches="tight", facecolor="white")
        if os.path.getsize(path) <= 5_000_000:
            break
    plt.close(fig)
    return {"path": os.path.relpath(path, os.path.dirname(os.path.dirname(FIG_DIR))), "bytes": os.path.getsize(path)}


def model_panels(tile: str, rec: dict, sj: dict | None, out: dict) -> None:
    hz = np.load(out_path("tiles", f"{tile}_heads.npz"))
    hs = np.load(out_path("sjc", f"{tile}_heads.npz")) if sj else None
    shape = tuple(rec["inputs"]["shape"])
    lines = unpack(hz["lines_1m"], shape)
    nb = shape[0] // BLOCK_M
    lines12 = lines[:nb * BLOCK_M, :nb * BLOCK_M].reshape(nb, BLOCK_M, nb, BLOCK_M).any(axis=(1, 3))
    poly12 = hz["poly12"]
    pr, pc = hz["pick_rc"]
    names = [m for m in MODEL_ORDER if (m != "SJ_C" or hs is not None)]
    ncols = 4
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 4.6 * nrows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, name in zip(axes.ravel(), names, strict=False):
        src = hs if name == "SJ_C" else hz
        Z12 = src[f"{name}_Z12"]
        ch = src[f"{name}_a2_channel"]
        hd = src[f"{name}_a2_heads"]
        ax.axis("on")
        ax.imshow(np.nan_to_num(hillshade(Z12.astype(float), float(BLOCK_M)), nan=1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.imshow(np.where(lines12, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#009E73"]), alpha=0.9, interpolation="nearest")
        ax.imshow(np.where(ch, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#0072B2"]), alpha=0.75, interpolation="nearest")
        if poly12.any():
            ax.contour(poly12.astype(float), levels=[0.5], colors="k", linewidths=0.6, linestyles="--")
        hr, hc = np.nonzero(hd)
        ax.plot(hc, hr, ".", color="#E69F00", ms=4)
        if pr.size:
            ax.plot(pc / BLOCK_M - 0.5, pr / BLOCK_M - 0.5, "^", color="#D55E00", ms=5, mec="k", mew=0.4)
        e = (sj["extraction"] if name == "SJ_C" else rec["extraction"][name])["a2"]
        mod = sj["model"] if name == "SJ_C" else rec["models"].get(name, {})
        md = e["pick_dist_m"]["median"]
        ax.set_title(f"{LABEL[name]}" + (f": K/K0 {mod['K'] / rec['K0']:.2f}, D {mod['D']:.3g}, MSE {mod['mse']:.0f}" if mod else "") +
                     f"\nheads {e['n_heads_in_polygon']}/{e['n_picks_in_polygon']} in polygon; median pick dist {md if md is None else round(md, 1)} m", fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{tile}: 12 m hillshade of each surface; criterion channels at alpha = 2, C_F (blue), their heads (orange), Steve's picks (triangles), M&D traced channels (green); "
                 "dashed = M&D study area", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out[f"{tile}_model_heads"] = save(fig, os.path.join(FIG_DIR, "model_heads", f"{tile}_model_heads.png"))


def observed_criterion(tile: str, rec: dict, out: dict) -> None:
    """Section 2.4: the full-site 12 m criterion (alpha = 0 at A_F, alpha = 2 at C_F) on the observed surface, with the picks and the M&D lines."""
    hz = np.load(out_path("tiles", f"{tile}_heads.npz"))
    sd = np.load(out_path("grid12", f"{tile}_seeds.npz"))
    shape = tuple(rec["inputs"]["shape"])
    lines = unpack(hz["lines_1m"], shape)
    poly = unpack(hz["poly_1m"], shape)
    r_off, c_off = (int(v) for v in hz["obsfull_offset_1m"])
    Zb = sd["box_Zf"]
    nb = Zb.shape
    pr, pc = hz["pick_rc"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    for ax, key, lab in zip(axes, ("a0", "a2"), ("alpha = 0 at A_F", "alpha = 2 at C_F"), strict=True):
        ch, hd = hz[f"obsfull_{key}_channel"], hz[f"obsfull_{key}_heads"]
        ax.imshow(np.nan_to_num(hillshade(Zb.astype(float), float(BLOCK_M)), nan=1.0), cmap="gray", vmin=0, vmax=1, interpolation="nearest",
                  extent=(c_off - 0.5, c_off + nb[1] * BLOCK_M - 0.5, r_off + nb[0] * BLOCK_M - 0.5, r_off - 0.5))
        ax.imshow(np.where(ch, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#0072B2"]), alpha=0.75, interpolation="nearest",
                  extent=(c_off - 0.5, c_off + nb[1] * BLOCK_M - 0.5, r_off + nb[0] * BLOCK_M - 0.5, r_off - 0.5))
        ax.imshow(np.where(lines, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#009E73"]), alpha=0.9, interpolation="nearest")
        if poly.any():
            ax.contour(poly.astype(float), levels=[0.5], colors="k", linewidths=0.6, linestyles="--")
        hr, hc = np.nonzero(hd)
        ax.plot(hc * BLOCK_M + BLOCK_M // 2 + c_off, hr * BLOCK_M + BLOCK_M // 2 + r_off, ".", color="#E69F00", ms=5)
        if pr.size:
            ax.plot(pc, pr, "^", color="#D55E00", ms=6, mec="k", mew=0.4)
        e = rec["extraction"]["obs_full_lattice"][key]
        ax.set_xlim(-0.5, shape[1] - 0.5)
        ax.set_ylim(shape[0] - 0.5, -0.5)
        ax.set_title(f"{lab}: C {e['C']:.4g} m²; heads {e['n_heads_in_polygon']}/{e['n_picks_in_polygon']} in polygon; median pick distance {e['pick_dist_m']['median']:.1f} m; "
                     f"\nM&D line share within 15 m {e.get('frac_line_within_15m_of_channel', float('nan')):.2f}", fontsize=7.5)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{tile}: the observed 12 m criterion on the full-site lattice (blue channels, orange heads), Steve's picks (triangles), M&D traced channels (green); dashed = study area", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out[f"{tile}_observed_criterion"] = save(fig, os.path.join(FIG_DIR, "observed", f"{tile}_observed_criterion.png"))


def residual_maps(tile: str, rec: dict, sj: dict | None, out: dict) -> None:
    rz = np.load(out_path("tiles", f"{tile}_residuals.npz"))
    res = {"P4": rz["P4"], "AD": rz["AD"]}
    if sj:
        res = {"SJ_C": np.load(out_path("sjc", f"{tile}_residual.npz"))["SJ_C"]} | res
    v = float(np.nanpercentile(np.abs(np.concatenate([r[np.isfinite(r)].ravel() for r in res.values()])), 95))
    fig, axes = plt.subplots(1, len(res), figsize=(4.4 * len(res), 4.6))
    axes = np.atleast_1d(axes)
    for ax, (name, r) in zip(axes, res.items(), strict=True):
        im = ax.imshow(r, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
        mod = sj["model"] if name == "SJ_C" else rec["models"][name]
        ax.set_title(f"{LABEL[name]}: MSE {mod['mse']:.1f} m²\nK/K0 {mod['K'] / rec['K0']:.2f}, D {mod['D']:.3g} m²/yr", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.tolist(), shrink=0.7, label="model - observed elevation (m)")
    fig.suptitle(f"{tile}: elevation residuals (one colour scale, ± the 95th percentile of |residual| over the panels)", fontsize=9)
    out[f"{tile}_residuals"] = save(fig, os.path.join(FIG_DIR, "residuals", f"{tile}_residuals.png"))


def sjc_profiles(sjc: dict, out: dict) -> None:
    names = [t for t in TTV_PRIMARY if t in sjc]
    if not names:
        return
    ncols = 4
    nrows = int(np.ceil(len(names) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, t in zip(axes.ravel(), names, strict=False):
        df = pd.read_parquet(out_path("sjc", f"{t}_c_profile.parquet"))
        ax.axis("on")
        ax.semilogx(df.C, df.mse_joint, ".", ms=2.5, color="#0072B2", label="whole tile")
        if "mse_h1_fit" in df:
            ax.semilogx(df.C, df.mse_h1_fit, ".", ms=1.5, color="#E69F00", alpha=0.6, label="half 1 fit")
            ax.semilogx(df.C, df.mse_h2_fit, ".", ms=1.5, color="#009E73", alpha=0.6, label="half 2 fit")
        o = sjc[t]["joint_optimum"]
        ax.axvline(o["C"], color="k", lw=0.8)
        for lo, hi in sjc[t]["identifiability"]["level_0.01"]["intervals"]:
            ax.axvspan(lo, hi, color="0.85", zorder=0)
        ax.axvline(sjc[t]["setup"]["C_range"][0], color="0.6", lw=0.5, ls=":")
        ax.set_title(f"{t}: C {o['C']:.3g} m², K/K0 {o['K_over_K0']:.2f}, D {o['D']:.3g}, MSE {o['mse']:.1f}", fontsize=7.5)
        ax.set_xlabel("C = A₁₂ S₁₂² (m²)", fontsize=7)
        ax.set_ylabel("MSE (m²)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_ylim(0, min(ax.get_ylim()[1], 4 * o["mse"] + 1))
    axes.ravel()[0].legend(fontsize=6)
    fig.suptitle("SJ-C: the MSE profile over the map-scale head criterion C (grey = 1 % interval; dotted = the lower end of the C range)", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out["sjc_profiles"] = save(fig, os.path.join(FIG_DIR, "sjc_profiles.png"))


def decomposition(rules: dict, out: dict) -> None:
    dec = rules["DX"]["decomposition"]
    if not dec:
        return
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    for m, mk, col in (("SJ_C", "o", "#0072B2"), ("PF_C", "s", "#E69F00"), ("PF_A", "^", "#009E73"), ("P4", "D", "#CC79A7")):
        xs = [dec[t]["product"] for t in dec if f"{m}_D_over_D_phys" in dec[t]]
        ys = [dec[t][f"{m}_D_over_D_phys"] for t in dec if f"{m}_D_over_D_phys" in dec[t]]
        if xs:
            ax.loglog(xs, ys, mk, color=col, mec="k", mew=0.4, ms=6, label=LABEL[m])
    lim = [0.5, 50]
    ax.loglog(lim, lim, "k-", lw=0.8, label="1:1")
    ax.loglog(lim, [x * 1.5 for x in lim], "k:", lw=0.6)
    ax.loglog(lim, [x / 1.5 for x in lim], "k:", lw=0.6, label="x 1.5")
    ax.set_xlabel("(D_relief / D_form) × (U / E_ridge)")
    ax.set_ylabel("D_fit / D_phys")
    ax.set_title("Rule DX: the fitted D against the relief × non-steady-state product", fontsize=9)
    ax.legend(fontsize=7)
    ax.grid(True, which="both", lw=0.3, alpha=0.5)
    fig.tight_layout()
    out["decomposition"] = save(fig, os.path.join(FIG_DIR, "D_decomposition.png"))


def main() -> None:
    t0 = time.time()
    for sub in ("model_heads", "residuals", "observed"):
        os.makedirs(os.path.join(FIG_DIR, sub), exist_ok=True)
    tiles = {t: load_json(out_path("tiles", f"{t}.json")) for t in TTV_PRIMARY if os.path.exists(out_path("tiles", f"{t}.json"))}
    sjc = {t: load_json(out_path("sjc", f"{t}.json")) for t in TTV_PRIMARY if os.path.exists(out_path("sjc", f"{t}.json")) and os.path.exists(out_path("sjc", f"{t}_heads.npz"))}
    out: dict = {}
    for t in MD_TILES:
        if t in tiles:
            model_panels(t, tiles[t], sjc.get(t), out)
            observed_criterion(t, tiles[t], out)
            print(t, "model panels", f"[{time.time() - t0:.0f} s]", flush=True)
    for t, rec in tiles.items():
        residual_maps(t, rec, sjc.get(t), out)
    sjc_profiles(sjc, out)
    rp = out_path("rules", "rules.json")
    if os.path.exists(rp):
        decomposition(load_json(rp), out)
    dump(json_ready(out), out_path("figures", "figures.json"))
    print({k: v["bytes"] for k, v in out.items()}, f"[{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
