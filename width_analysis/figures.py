"""Publication figures for the width analysis. Everything is drawn from the
points table and the fit dicts; no manual post-processing."""

from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from .config import SITES  # noqa: E402
from .fits import predict_band  # noqa: E402
from .regime import width_of  # noqa: E402

SITE_COLOR = {"AP": "#2a78d6", "GM": "#eb6834"}     # categorical slots 1 and 2 (light surface)
C_TEXT = "#52514e"
C_GRID = "#efeeea"
C_MUTED = "#9a9995"


def _style(ax):
    ax.grid(color=C_GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def _bin_centers(edges):
    e = np.asarray(edges)
    return np.sqrt(e[:-1] * e[1:])


def _bin_fraction(df, col, edges):
    """Fraction of True in ``col`` per A bin, with counts."""
    e = np.asarray(edges)
    b = np.clip(np.searchsorted(e, df["area_m2"].to_numpy(), side="right") - 1, 0, len(e) - 2)
    frac, n = [], []
    for k in range(len(e) - 1):
        m = b == k
        n.append(int(m.sum()))
        frac.append(float(df.loc[m, col].mean()) if m.any() else np.nan)
    return np.array(frac), np.array(n)


# ---------------------------------------------------------------------------
# 1. width vs drainage area with censored MLE fit
# ---------------------------------------------------------------------------

def fig_width_vs_area(points, fits, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, site in zip(axes, SITES):
        df = points[points.site == site]
        col = SITE_COLOR[site]
        ch = df[df.channelized]
        nch = df[~df.channelized]
        # not channelized: valley-floor width where flat, else an arrow at the bound
        vf = nch[nch.W_primary_source == "valley_floor"]
        un = nch[nch.W_primary_source == "unresolved"]
        ax.plot(vf.area_m2, vf.W_valley_floor, marker="+", ls="none", color=C_MUTED, ms=7, mew=1.2,
                label=f"not channelized, flat bed: valley-floor width (n={len(vf)})")
        ax.plot(un.area_m2, un.censor_bound, marker="v", ls="none", mfc="none", mec=C_MUTED, ms=6,
                label=f"not channelized, unresolved: bound 2δ (n={len(un)})")
        # channelized
        unc = ch[~ch.W_inner_bankbreak_censored]
        cen = ch[ch.W_inner_bankbreak_censored]
        ax.plot(unc.area_m2, unc.W_inner_bankbreak, marker="o", ls="none", color=col, ms=6,
                label=f"channelized: inner-channel (bank-break) width (n={len(unc)})")
        ax.plot(cen.area_m2, cen.censor_bound, marker="v", ls="none", color=col, ms=7,
                label=f"channelized, censored at 2δ (n={len(cen)})")
        steep = ch[ch.steep_rill_flag]
        if len(steep):
            ax.plot(steep.area_m2, np.where(steep.W_inner_bankbreak_censored, steep.censor_bound, steep.W_inner_bankbreak),
                    marker="^", ls="none", mfc="none", mec="#0b0b0b", ms=11, mew=1.0,
                    label=f"steep (S > 0.5) rills, in 'all' fit only (n={len(steep)})")
        # fits
        grid = np.logspace(np.log10(max(df.area_m2.min(), 1)), np.log10(df.area_m2.max()), 100)
        for variant, ls in (("all", "-"), ("no_steep", "--")):
            f = fits[site][variant]
            t = f["tobit"]
            if not np.isfinite(t["a"]):
                continue
            if variant == "no_steep" and f["n_channelized"] == fits[site]["all"]["n_channelized"]:
                continue        # identical sample: nothing to show
            ax.plot(grid, width_of(grid, t["a"], t["b"]), ls=ls, color=col, lw=1.8,
                    label=f"censored MLE ({variant}): W = {t['c']:.3g} A^{t['b']:.2f}, n={f['n_channelized']}")
            if variant == "all":
                lo, hi = predict_band(f["_boot_ab"], np.log10(grid))
                ax.fill_between(grid, 10 ** lo, 10 ** hi, color=col, alpha=0.15, lw=0, label="95 % bootstrap band (all)")
        ax.axhline(df.censor_bound.iloc[0], color=C_MUTED, lw=0.8, ls=":")
        ax.text(df.area_m2.min(), df.censor_bound.iloc[0] * 1.05, "2δ resolution bound", fontsize=7, color=C_TEXT)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("drainage area A (m²)")
        ax.set_title(SITES[site], loc="left", fontsize=10)
        ax.legend(fontsize=6.5, frameon=False, loc="upper left")
        _style(ax)
    axes[0].set_ylabel("width (m)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. model-free: resolved inner channel or flat bed wider than 2 delta
# ---------------------------------------------------------------------------

def fig_resolved_fraction(points, edges_by_site, out_png):
    fig, ax = plt.subplots(figsize=(7.5, 4))
    w = 0.38
    for k, site in enumerate(SITES):
        df = points[points.site == site]
        edges = edges_by_site[site]
        frac, n = _bin_fraction(df, "resolved_width", edges)
        x = np.arange(len(frac)) + (k - 0.5) * w
        ax.bar(x, frac, width=w * 0.92, color=SITE_COLOR[site], label=SITES[site])
        for xi, fi, ni in zip(x, frac, n):
            if ni:
                ax.text(xi, (fi if np.isfinite(fi) else 0) + 0.02, f"{int(round(fi * ni))}/{ni}", ha="center", fontsize=7, color=C_TEXT)
    e = edges_by_site["AP"]
    ax.set_xticks(np.arange(len(e) - 1))
    ax.set_xticklabels([f"{np.log10(e[i]):.1f}-{np.log10(e[i + 1]):.1f}" for i in range(len(e) - 1)], fontsize=8)
    ax.set_xlabel("log10 A bin (site-specific edges; AP edges shown)")
    ax.set_ylabel("fraction with a resolved width > 2δ")
    ax.set_ylim(0, 1.12)
    ax.set_title("Resolved inner channel (both bank breaks) or flat bed wider than 2 pixels", loc="left", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# channelized fraction vs A with logistic fit and A50
# ---------------------------------------------------------------------------

def fig_channelized_fraction(points, fits, edges_by_site, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, site in zip(axes, SITES):
        df = points[points.site == site]
        col = SITE_COLOR[site]
        edges = edges_by_site[site]
        frac, n = _bin_fraction(df, "channelized", edges)
        xc = _bin_centers(edges)
        ax.plot(xc, frac, marker="o", ls="none", color=col, ms=7, label="fraction channelized per A bin")
        for xi, fi, ni in zip(xc, frac, n):
            if ni:
                ax.text(xi, fi + 0.03, str(ni), ha="center", fontsize=7, color=C_TEXT)
        lf = fits[site]["channelized_logistic"]
        grid = np.logspace(np.log10(edges[0]), np.log10(edges[-1]), 200)
        if np.isfinite(lf["b1"]):
            p = 1 / (1 + np.exp(-(lf["b0"] + lf["b1"] * np.log10(grid))))
            ax.plot(grid, p, color=col, lw=1.6, label="logistic fit")
        if np.isfinite(lf["A50"]):
            ax.axvline(lf["A50"], color=col, ls="--", lw=1.2, label=f"A50 = {lf['A50']:.2g} m² [{lf['A50_ci'][0]:.2g}, {lf['A50_ci'][1]:.2g}]")
            ax.axvspan(max(lf["A50_ci"][0], edges[0] / 10), min(lf["A50_ci"][1], edges[-1] * 10), color=col, alpha=0.12, lw=0)
        ax.set_xlim(edges[0] / 3, edges[-1] * 3)
        ax.axhline(0.5, color=C_MUTED, lw=0.8, ls=":")
        ax.set_xscale("log")
        ax.set_xlabel("drainage area A (m²)")
        ax.set_title(SITES[site], loc="left", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        _style(ax)
    axes[0].set_ylabel("fraction channelized (both bank breaks, inner relief ≥ 0.3 m)")
    axes[0].set_ylim(-0.02, 1.1)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 5. valley-bottom shape class
# ---------------------------------------------------------------------------

def fig_shape_class(points, edges_by_site, out_png):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, site in zip(axes, SITES):
        df = points[points.site == site].copy()
        df["is_flat"] = df.shape_class == "flat_bed"
        edges = edges_by_site[site]
        frac, n = _bin_fraction(df, "is_flat", edges)
        x = np.arange(len(frac))
        f0 = np.nan_to_num(frac)
        ax.bar(x, f0, color=SITE_COLOR[site], label="flat_bed (≥ 2 m within 0.1 m of the thalweg)")
        ax.bar(x, np.where(n > 0, 1 - f0, 0), bottom=f0, color="#d9d8d4", label="v_shape")
        for xi, ni in zip(x, n):
            ax.text(xi, 1.02, f"n={ni}", ha="center", fontsize=7, color=C_TEXT)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{np.log10(edges[i]):.1f}-{np.log10(edges[i + 1]):.1f}" for i in range(len(edges) - 1)], fontsize=7.5)
        ax.set_xlabel("log10 A bin")
        ax.set_title(SITES[site], loc="left", fontsize=10)
        ax.legend(fontsize=7, frameon=False, loc="lower left")
        _style(ax)
    axes[0].set_ylabel("fraction of points")
    axes[0].set_ylim(0, 1.12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# bank-break width vs 0.25 m stage width
# ---------------------------------------------------------------------------

def fig_bankbreak_vs_stage(points, out_png):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ch = points[points.channelized]
    for site in SITES:
        d = ch[ch.site == site]
        ax.plot(d["W_stage_0.25"], d.W_inner_bankbreak, marker="o", ls="none", color=SITE_COLOR[site], ms=6,
                label=f"{SITES[site]} (n={len(d)})")
    lo = max(0.2, min(ch["W_stage_0.25"].min(), ch.W_inner_bankbreak.min()) * 0.8)
    hi = max(ch["W_stage_0.25"].max(), ch.W_inner_bankbreak.max()) * 1.2
    ax.plot([lo, hi], [lo, hi], color=C_MUTED, lw=1, ls="--", label="1:1")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("width at 0.25 m above thalweg (m)")
    ax.set_ylabel("inner-channel (bank-break) width (m)")
    ax.set_title("Channelized points only", loc="left", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# regime-III fraction vs grid spacing
# ---------------------------------------------------------------------------

def fig_regime_fraction(table, out_png):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for site in SITES:
        for variant, ls in (("all", "-"), ("no_steep", "--")):
            t = table[(table.site == site) & (table.variant == variant)]
            if t.empty or t.fraction.isna().all():
                continue
            if variant == "no_steep":
                t_all = table[(table.site == site) & (table.variant == "all")]
                if np.allclose(t.fraction.to_numpy(), t_all.fraction.to_numpy(), equal_nan=True):
                    continue
            ax.plot(t.dx_m, t.fraction, ls=ls, color=SITE_COLOR[site], lw=1.8, label=f"{SITES[site]} ({variant})")
            if variant == "all":
                ax.fill_between(t.dx_m, t.fraction_lo, t.fraction_hi, color=SITE_COLOR[site], alpha=0.15, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("grid spacing δ (m); δ = 1 m plus the 5-100 m sweep")
    ax.set_ylabel("fraction of channel length (A ≥ A_crit) with W(A) > δ")
    ax.set_ylim(-0.02, 1.05)
    ax.set_title("Regime III (channel wider than a pixel) vs grid spacing, 95 % bootstrap band", loc="left", fontsize=9)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# W / delta maps at delta = pixel size
# ---------------------------------------------------------------------------

def fig_wd_maps(map_data, fits, out_png):
    fig, axes = plt.subplots(2, 5, figsize=(16, 7.6))
    cmap = plt.get_cmap("Blues")
    for r, site in enumerate(SITES):
        t = fits[site]["all"]["tobit"]
        for c, tile in enumerate([1, 2, 3, 4, 5]):
            ax = axes[r, c]
            md = map_data[(site, tile)]
            ax.imshow(md["hillshade"], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            ch = md["chan"]
            if np.isfinite(t["a"]):
                ratio = width_of(ch["area"], t["a"], t["b"]) / md["pixel_size"]
                below = ratio <= 1
                ax.scatter(ch["cols"][below], ch["rows"][below], s=0.4, c="#9a9995", marker=".", lw=0)
                sc = ax.scatter(ch["cols"][~below], ch["rows"][~below], s=0.6, c=ratio[~below], cmap=cmap,
                                norm=LogNorm(vmin=1, vmax=max(2, np.nanmax(ratio))), marker=".", lw=0)
                frac = ch["length"][~below].sum() / ch["length"].sum()
                ax.set_title(f"{site} tile {tile}: A_crit {md['chan']['a_crit']:.2g} m²\n{100 * frac:.0f} % of channel length W/δ > 1",
                             fontsize=7.5, loc="left")
            else:
                ax.scatter(ch["cols"], ch["rows"], s=0.4, c="#9a9995", marker=".", lw=0)
                ax.set_title(f"{site} tile {tile}: no fit", fontsize=7.5, loc="left")
            ax.set_xticks([])
            ax.set_yticks([])
    cb = fig.colorbar(sc, ax=axes.ravel().tolist(), shrink=0.6, pad=0.01)
    cb.set_label("W(A) / δ at δ = pixel size (gray: W/δ ≤ 1)")
    fig.suptitle("Fitted inner-channel width per pixel on the channel network (A ≥ tile A_crit)", fontsize=10, x=0.02, ha="left")
    fig.savefig(out_png, dpi=170, bbox_inches="tight")
    plt.close(fig)
