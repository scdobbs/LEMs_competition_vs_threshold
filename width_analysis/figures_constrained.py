"""Figures for the constrained re-run."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import SITES  # noqa: E402
from .fits import predict_band  # noqa: E402
from .regime import width_of  # noqa: E402

SITE_COLOR = {"AP": "#2a78d6", "GM": "#eb6834"}
THRESH_COLOR = {1.0: "#2a78d6", 2.0: "#1baf7a", 5.0: "#eda100", 10.0: "#e34948"}
C_TEXT = "#52514e"
C_MUTED = "#9a9995"


def _style(ax):
    ax.grid(color="#efeeea", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def fig_width_vs_area(points, fits, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, site in zip(axes, SITES):
        d = points[points.site == site]
        col = SITE_COLOR[site]
        ic = d[d.point_class == "inner_channel"]
        wt = d[d.point_class == "wide_or_terrace"]
        un = d[d.point_class == "unresolved_narrow"]
        am = d[d.point_class == "ambiguous"]
        ax.plot(ic.area_m2, ic.W_bank, "o", color=col, ms=6, label=f"inner_channel: W_bank (n={len(ic)})")
        ax.plot(wt.area_m2, wt.censor_right_bound, "^", color=col, ms=8, mfc="none", mew=1.5,
                label=f"wide_or_terrace: W > {wt.censor_right_bound.iloc[0] if len(wt) else 0:g} m (n={len(wt)})")
        ax.plot(un.area_m2, un.censor_left_bound, "v", color=col, ms=8, mfc="none", mew=1.5,
                label=f"unresolved_narrow: W ≤ 2δ (n={len(un)})")
        ax.plot(am.area_m2, np.full(len(am), 0.5), "x", color=C_MUTED, ms=6, label=f"ambiguous, no width (n={len(am)})")
        f = fits.get(site, {}).get("default")
        if f is not None and f.get("qualifies"):
            t = f["tobit"]
            grid = np.logspace(np.log10(d.area_m2.min()), np.log10(d.area_m2.max()), 100)
            ax.plot(grid, width_of(grid, t["a"], t["b"]), color=col, lw=1.8,
                    label=f"censored MLE: W = {t['c']:.3g} A^{t['b']:.2f} [{f['b_ci'][0]:.2f}, {f['b_ci'][1]:.2f}]")
            lo, hi = predict_band(f["_boot_ab"], np.log10(grid))
            ax.fill_between(grid, 10 ** lo, 10 ** hi, color=col, alpha=0.15, lw=0)
        elif f is not None and np.isfinite(f["tobit"]["b"]):
            ax.text(0.02, 0.02, f"fit does not qualify: b = {f['tobit']['b']:.2f} [{f['b_ci'][0]:.2f}, {f['b_ci'][1]:.2f}]; {f['qualify_note']}",
                    transform=ax.transAxes, fontsize=7, color=C_TEXT)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("drainage area A (m²)")
        ax.set_title(SITES[site], loc="left", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        _style(ax)
    axes[0].set_ylabel("bank-to-bank width (m)")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)


def fig_exceedance(exc, edges_by_site, out_png):
    thresholds = sorted(exc.threshold_m.unique())
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    w = 0.8 / len(thresholds)
    for ax, site in zip(axes, SITES):
        e = np.asarray(edges_by_site[site])
        d = exc[(exc.site == site) & (exc.bin_index >= 0)]
        for j, T in enumerate(thresholds):
            t = d[d.threshold_m == T].sort_values("bin_index")
            x = t.bin_index.to_numpy() + (j - (len(thresholds) - 1) / 2) * w
            err = np.vstack([t.fraction - t.ci_lo, t.ci_hi - t.fraction])
            ax.bar(x, t.fraction, width=w * 0.95, color=THRESH_COLOR.get(T, C_MUTED), label=f"W > {T:g} m")
            ax.errorbar(x, t.fraction, yerr=err, fmt="none", ecolor="#0b0b0b", elinewidth=0.8, capsize=2)
        ax.set_xticks(range(len(e) - 1))
        ax.set_xticklabels([f"{np.log10(e[i]):.1f}-{np.log10(e[i + 1]):.1f}\n(n={int(d[(d.bin_index == i) & (d.threshold_m == thresholds[0])].n.iloc[0])})" for i in range(len(e) - 1)], fontsize=7.5)
        ax.set_xlabel("log10 A bin")
        ax.set_title(f"{SITES[site]}: fraction of points shown to exceed each width, 95 % binomial intervals", loc="left", fontsize=8.5)
        ax.legend(fontsize=7.5, frameon=False, loc="upper left", ncol=4)
        ax.set_ylim(0, 1.15)
        _style(ax)
    axes[0].set_ylabel("fraction of points")
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)


def fig_consistency(points, out_png):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ic = points[points.point_class == "inner_channel"]
    for site in SITES:
        d = ic[ic.site == site]
        ax.plot(d.W_stage_bank, d.W_bank, "o", color=SITE_COLOR[site], ms=6, label=f"{SITES[site]} (n={len(d)})")
    lo = max(0.5, min(ic.W_stage_bank.min(), ic.W_bank.min()) * 0.8) if len(ic) else 0.5
    hi = max(ic.W_stage_bank.max(), ic.W_bank.max()) * 1.25 if len(ic) else 20
    ax.plot([lo, hi], [lo, hi], "--", color=C_MUTED, lw=1, label="1:1")
    ax.plot([lo, hi], [1.25 * lo, 1.25 * hi], ":", color=C_MUTED, lw=0.8)
    ax.plot([lo, hi], [0.8 * lo, 0.8 * hi], ":", color=C_MUTED, lw=0.8, label="±25 %")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("W_stage_bank: width at the mean bank height (m)")
    ax.set_ylabel("W_bank: distance between bank picks (m)")
    ax.set_title("inner_channel points", loc="left", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    _style(ax)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)


def fig_sensitivity(sens, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    rels = sorted(sens.bank_max_relief_m.unique())
    shades = ["#86b6ef", "#2a78d6", "#104281"]
    for k, rel in enumerate(rels):
        d = sens[sens.bank_max_relief_m == rel].sort_values("bank_max_offset_m")
        axes[0].plot(d.bank_max_offset_m, d.median_W_bank, marker="o", color=shades[k % 3], label=f"relief ≤ {rel:g} m")
        axes[1].plot(d.bank_max_offset_m, d.frac_wide_or_terrace, marker="o", color=shades[k % 3], label=f"relief ≤ {rel:g} m")
    for off in sorted(sens.bank_max_offset_m.unique()):
        axes[0].axhline(2 * off, color=C_MUTED, lw=0.6, ls=":")
    axes[0].set_xlabel("bank_max_offset_m (m)"); axes[0].set_ylabel("median W_bank of inner_channel points (m)")
    axes[0].set_title("dotted: 2 x offset limit", loc="left", fontsize=8.5)
    axes[1].set_xlabel("bank_max_offset_m (m)"); axes[1].set_ylabel("fraction wide_or_terrace"); axes[1].set_ylim(0, 1)
    for ax in axes:
        ax.legend(fontsize=7.5, frameon=False); _style(ax)
    fig.tight_layout(); fig.savefig(out_png, dpi=200); plt.close(fig)
