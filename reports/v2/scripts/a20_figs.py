#!/usr/bin/env python3
"""Addendum 20 figures (sections 2.5, 2.6, 3.1, 3.3; decision 4), from <run>/site/partA.json and <run>/heads/heads.json, into
reports/v2/figures/a20/:
  binned_curves_r7.png            Perron Fig. S2a style: binned lap against A|grad z| per site (r = 7 m), plateau bins coloured, Perron's value;
                                  D-infinity and Perron's own survey (NCALM / PAMAP) beside the primary curve; signed mean on a symlog panel
  <site>_D_bracket_intervals.png  per tile: D0_P and D0_N (68 %), D_R (68 %), the two brackets, the addendum 18 AD, P4, P3 D
  <site>_D0_vs_window_gallery.png per tile: D0_P against r (3-15 m) and D0_N against L (3-51 m)
  plateau/<site>_plateau_gallery.png  the per-tile plateau maps, tiled
  heads_area_slope.png            log A against log S: Table 1, the snapped picks (S_up, S_down); M&D eq. 10 and the fitted lines
usage: a20_figs.py [--part bracket]   (--part bracket redraws the interval figures with the Part B fits, after a20_rules.py)
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a20_common import FIG_DIR, MD_EQ10, PERRON_2012, PERRON_RADII_M, PRIMARY, dump, load_json, md_table1, out_path  # noqa: E402

SITES = ("GM", "AP", "TTV")
COL = {"P": "#0072B2", "N": "#E69F00", "R": "#009E73", "AD": "#D55E00", "P4": "#CC79A7", "P3": "#56B4E9"}


def save(fig, path, dpi=150):
    for sc in (1.0, 0.8, 0.6, 0.45):
        fig.savefig(path, dpi=int(dpi * sc), bbox_inches="tight", facecolor="white")
        if os.path.getsize(path) <= 5_000_000:
            break
    plt.close(fig)
    return {"path": os.path.relpath(path, os.path.dirname(os.path.dirname(FIG_DIR))), "bytes": os.path.getsize(path)}


def curve_arrays(est):
    c = est["curve"]
    x = np.array([np.sqrt(b["lo_m2"] * b["hi_m2"]) for b in c])
    L = np.array([abs(b["L_bin"]) if b["L_bin"] is not None else np.nan for b in c])
    ms = np.array([b["mean_signed"] if b["mean_signed"] is not None else np.nan for b in c])
    n = np.array([b["n"] for b in c])
    fc = np.array([b["frac_convex"] if b["frac_convex"] is not None else np.nan for b in c])
    return x, L, ms, n, fc


def binned_curves(A, figs):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), sharex=True, gridspec_kw={"height_ratios": [3, 2, 1.3]})
    for j, site in enumerate(SITES):
        pg = A["pooled"][site]
        est = pg["primary"]["r7"]
        x, L, ms, n, fc = curve_arrays(est)
        ok = n >= 1
        ax = axes[0, j]
        ax.loglog(x[ok], L[ok], "o-", color="0.35", ms=4, lw=1, label="primary tiles, r = 7 m (D8 area)")
        pt = est["point"]
        if pt.get("plateau"):
            b = pt["bins"]
            ax.loglog(x[b], L[b], "o", color=COL["P"], ms=7, label=f"plateau bins (rule): |∇²z_h| = {abs(pt['lap_h']):.4f} m⁻¹")
        for key, sty, lab in (("r7_dinf", "s--", "D∞ area"),):
            if key in pg["primary"]:
                xx, LL, _, nn, _ = curve_arrays(pg["primary"][key])
                ax.loglog(xx[nn >= 1], LL[nn >= 1], sty, color="0.6", ms=3, lw=0.8, label=lab + (f": {abs(pg['primary'][key]['point']['lap_h']):.4f}" if pg["primary"][key]["point"].get("plateau") else ": no plateau"))
        og = {"GM": "ncalm", "AP": "pamap"}.get(site)
        if og and og in pg and "r7" in pg[og]:
            xx, LL, _, nn, _ = curve_arrays(pg[og]["r7"])
            p_ = pg[og]["r7"]["point"]
            ax.loglog(xx[nn >= 1], LL[nn >= 1], "^:", color="#999933", ms=3, lw=0.8, label=f"{og.upper()} tiles" + (f": {abs(p_['lap_h']):.4f}" if p_.get("plateau") else ": no plateau"))
        if site in PERRON_2012:
            ax.axhline(PERRON_2012[site]["curvature_per_m"], color="k", ls="--", lw=1, label=f"Perron et al. 2012: {PERRON_2012[site]['curvature_per_m']:.4f}")
            ax.axhspan(0.67 * PERRON_2012[site]["curvature_per_m"], 1.5 * PERRON_2012[site]["curvature_per_m"], color="k", alpha=0.06, label="R11P range 0.67-1.5 x")
        ax.set_title(f"{site}: geometric-mean |∇²z| of convex cells per bin", fontsize=10)
        ax.set_ylabel("|∇²z| (m⁻¹)")
        ax.legend(fontsize=7)
        ax2 = axes[1, j]
        ax2.semilogx(x[ok], ms[ok], "o-", color="0.35", ms=4, lw=1)
        if pt.get("plateau"):
            ax2.semilogx(x[pt["bins"]], ms[pt["bins"]], "o", color=COL["P"], ms=7)
        ax2.set_yscale("symlog", linthresh=1e-3)
        ax2.axhline(0, color="k", lw=0.5)
        if site in PERRON_2012:
            ax2.axhline(-PERRON_2012[site]["curvature_per_m"], color="k", ls="--", lw=1)
        ax2.set_ylabel("signed mean ∇²z (symlog)")
        ax3 = axes[2, j]
        ax3.semilogx(x[ok], fc[ok], "o-", color="0.35", ms=3)
        ax3.axhline(0.6, color="r", lw=0.6, ls=":")
        ax3.set_ylabel("fraction convex")
        ax3.set_xlabel("A|∇z| (m²)")
        ax3.set_ylim(0, 1)
    fig.suptitle("Addendum 20, decision 2: binned Laplacian against A|∇z| (7 m disk, D8 area, cells of all primary tiles pooled; 4 bins per decade)", fontsize=11)
    fig.tight_layout()
    figs["binned_curves_r7"] = save(fig, os.path.join(FIG_DIR, "binned_curves_r7.png"))


def intervals(A, B, figs):
    for site in SITES:
        tl = [t for t in PRIMARY[site] if t in A["tiles"]]
        fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(tl) + 3), 6))
        for i, t in enumerate(tl):
            x = A["tiles"][t]
            for off, key, ci, col, lab in ((-0.22, "D_C", "D_C_ci68", COL["P"], "D_C (D₀,P; D₀,N if no plateau)"), (-0.08, "D0_N_A", "D0_N_A_ci68", COL["N"], "D₀,N"), (0.08, "D_R", "D_R_ci68", COL["R"], "D_R (10³ network)")):
                v = x.get(key)
                if v:
                    c = x.get(ci) or [v, v]
                    ax.errorbar(i + off, v, yerr=[[v - c[0]], [c[1] - v]], fmt="o", color=col, ms=4, capsize=2, label=lab if i == 0 else None)
            if x.get("bracket_P"):
                ax.plot([i - 0.3, i + 0.3], [x["bracket_P"][0]] * 2, "-", color=COL["P"], lw=0.6)
                ax.plot([i - 0.3, i + 0.3], [x["bracket_P"][1]] * 2, "-", color=COL["R"], lw=0.6)
            f18 = A["a18_fits_for_figures"].get(t, {})
            for off, m, mk in ((0.2, "AD", "x"), (0.25, "P4", "D"), (0.3, "P3", "s")):
                if f18.get(m):
                    ax.plot(i + off, f18[m], mk, color=COL[m], ms=6, mfc="none", mew=1.5, label=f"addendum 18 {m}" if i == 0 else None)
            if B and site == "TTV":
                for off, m, mk, col in ((0.35, "PF", "*", "k"), (0.4, "SJ_a1.65", "P", "#999933")):
                    d = B["R9"].get(m, {}).get("tiles", {}).get(t, {}).get("D")
                    if d:
                        ax.plot(i + off, d, mk, color=col, ms=7, label=m if i == 0 else None)
            if not x["usable_N"]:
                ax.text(i, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 1e-4, "<500", fontsize=6, ha="center", color="r")
        ax.set_yscale("log")
        ax.set_xticks(range(len(tl)))
        ax.set_xticklabels(tl, rotation=60, fontsize=8)
        ax.set_ylabel("D (m²/yr)")
        g = A["rules"].get("R11P", {}).get(site, {})
        ax.set_title(f"{site}: the D bracket per primary tile (68 % intervals; short lines = [D_C,lo, D_R,hi])" + (f"; R11P {'passes' if g.get('pass') else 'fails'}" if g else ""), fontsize=10)
        ax.legend(fontsize=7, ncol=3, loc="upper left")
        ax.grid(alpha=0.3, which="both")
        figs[f"{site}_D_bracket_intervals"] = save(fig, os.path.join(FIG_DIR, f"{site}_D_bracket_intervals.png"))


def window_gallery(A, figs):
    for site in SITES:
        tl = [t for t in PRIMARY[site] if t in A["tiles"]]
        nc = 6
        nr = int(np.ceil(len(tl) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(3.0 * nc, 2.6 * nr), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, t in zip(axes.ravel(), tl, strict=False):
            ax.axis("on")
            x = A["tiles"][t]
            rr = [r for r in PERRON_RADII_M]
            dp = [x.get(f"D0_P_r{r:g}") for r in rr]
            ax.plot([2 * r for r in rr], [v if v else np.nan for v in dp], "o-", color=COL["P"], ms=3, label="D₀,P (x = 2r)")
            Ls = sorted(int(k) for k in x["D0_N_by_L"])
            ax.plot(Ls, [x["D0_N_by_L"][k] if x["D0_N_by_L"][k] else np.nan for k in map(str, Ls)] if isinstance(next(iter(x["D0_N_by_L"])), str) else [x["D0_N_by_L"][k] for k in Ls], "s-", color=COL["N"], ms=2.5, label="D₀,N (x = L)")
            ax.axvline(14, color=COL["P"], lw=0.5, ls=":")
            ax.axvline(11, color=COL["N"], lw=0.5, ls=":")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(t + ("" if x["usable_N"] else " (D₀,N < 500 px)"), fontsize=8)
            ax.tick_params(labelsize=6)
        axes.ravel()[0].legend(fontsize=6)
        fig.suptitle(f"{site}: D₀ against window (D₀,P at r = 3-15 m plotted at the disk diameter 2r; D₀,N at L = 3-51 m); dotted = primary windows", fontsize=10)
        fig.tight_layout()
        figs[f"{site}_D0_vs_window_gallery"] = save(fig, os.path.join(FIG_DIR, f"{site}_D0_vs_window_gallery.png"), dpi=130)


def plateau_gallery(A, figs):
    os.makedirs(os.path.join(FIG_DIR, "plateau"), exist_ok=True)
    for site in SITES:
        tl = [t for t in PRIMARY[site] if os.path.exists(os.path.join(FIG_DIR, "plateau", f"{t}_plateau.png"))]
        nc = 5
        nr = int(np.ceil(len(tl) / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(4.2 * nc, 4.2 * nr), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, t in zip(axes.ravel(), tl, strict=False):
            ax.imshow(mpimg.imread(os.path.join(FIG_DIR, "plateau", f"{t}_plateau.png")))
            ax.set_title(t, fontsize=9)
        fig.suptitle(f"{site} primary tiles: plateau cells (r = 7 m) coloured by ∇²z and the D₀,N pixels outlined (full-size per-tile figures beside this file)", fontsize=10)
        fig.tight_layout()
        figs[f"{site}_plateau_gallery"] = save(fig, os.path.join(FIG_DIR, "plateau", f"{site}_plateau_gallery.png"), dpi=110)


def heads_fig(H, figs):
    t1 = md_table1()
    A1, S1 = np.array([x["A"] for x in t1]), np.array([x["S"] for x in t1])
    P = H["picks"]
    A10 = np.array([q["A_P10_m2"] for q in P])
    Su = np.array([q["S_up_P10"] for q in P])
    Sd = np.array([q["S_down_P10"] for q in P])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.loglog(S1, A1, "o", color="0.4", ms=5, label="M&D (1989) Table 1 (63)")
    ax.loglog(Su, A10, "^", color=COL["P"], ms=6, mfc="none", label="snapped picks: A_P10, S_up (64)")
    ax.loglog(Sd, A10, "v", color=COL["N"], ms=6, mfc="none", label="snapped picks: A_P10, S_down (64)")
    s = np.logspace(np.log10(0.05), np.log10(2), 50)
    ax.loglog(s, MD_EQ10["a_m2"] * s ** -MD_EQ10["b"], "k-", lw=1.5, label="M&D eq. 10: 1978 tan θ^-1.65")
    for key, col, lab in (("i_table1", "0.4", "(i) Table 1"), ("ii_AP10_Sup", COL["P"], "(ii) A_P10, S_up"), ("iii_AP10_Sdown", COL["N"], "(iii) A_P10, S_down")):
        f = H["fits"][key]
        ax.loglog(s, f["a"] * s ** -f["b"], "--", color=col, lw=1, label=f"{lab}: a {f['a']:,.0f}, b {f['b']:.2f} [{f['b_ci95'][0]:.2f}, {f['b_ci95'][1]:.2f}], R² {f['R2']:.2f}")
    ax.set_xlabel("local slope (m/m)")
    ax.set_ylabel("source area (m²)")
    ax.set_title("TTV channel heads: area against slope (fits: OLS on log A, A dependent)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    figs["heads_area_slope"] = save(fig, os.path.join(FIG_DIR, "heads_area_slope.png"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--part", default="A")
    a = p.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    A = load_json(out_path("site", "partA.json"))
    figs_path = out_path("figures", "figures.json")
    figs = load_json(figs_path) if os.path.exists(figs_path) else {}
    if a.part == "bracket":
        B = load_json(out_path("rules", "partB.json"))
        intervals(A, B, figs)
    else:
        H = load_json(out_path("heads", "heads.json"))
        binned_curves(A, figs)
        intervals(A, None, figs)
        window_gallery(A, figs)
        plateau_gallery(A, figs)
        heads_fig(H, figs)
    dump(figs, figs_path)
    print({k: v["bytes"] for k, v in figs.items()})


if __name__ == "__main__":
    main()
