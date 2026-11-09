"""d8_vs_dinf_comparison figures: elevation panels, area scatter, slope-area overlay, per tile."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.stats import linregress

from lemcvt.figures.common import load_params, load_table, plt, provenance_note, require_run, save, set_nature_style
from lemcvt.io.grids import read_geotiff


def build(out_dir: Path, run: Path | None = None) -> list[Path]:
    set_nature_style()
    run = run or require_run("d8_vs_dinf")
    params = load_params(run)
    sa = load_table(run, "slope_area")
    paths: list[Path] = []
    for tile, res in params["tiles"].items():
        label = res["params"].get("label", tile)
        Z_d8 = read_geotiff(run / f"{tile}_AD_d8_elevation.tif").data
        Z_dinf = read_geotiff(run / f"{tile}_AD_dinf_elevation.tif").data
        A_dinf = read_geotiff(run / f"{tile}_A_dinf.tif").data
        # Observed elevation is not stored in the run; use the D8 PT grid's boundary-consistent range
        vmin, vmax = np.nanmin(Z_d8), np.nanmax(Z_d8)
        fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.3), gridspec_kw={"width_ratios": [1, 1, 1]})
        for ax, Z, title in zip(axes[:2], [Z_d8, Z_dinf], ["D8 AD", "D-inf AD"], strict=True):
            ax.imshow(Z, cmap="terrain", origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.axis("off")
        diff = Z_dinf - Z_d8
        dmax = np.nanmax(np.abs(diff))
        im = axes[2].imshow(diff, cmap="RdBu_r", origin="lower", vmin=-dmax, vmax=dmax)
        axes[2].set_title("D-inf $-$ D8")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04, label="elevation units")
        fig.suptitle(f"{label}: MSE AD D8={res['mse_ad_d8']:.3e}, D-inf={res['mse_ad_dinf']:.3e}; "
                     f"PT D8={res['mse_pt_d8']:.3e}, D-inf={res['mse_pt_dinf']:.3e}", fontsize=7, y=1.02)
        provenance_note(axes[2], run)
        fig.tight_layout()
        paths += save(fig, out_dir, f"d8_vs_dinf_{tile}_elevation")

        fig, ax = plt.subplots(figsize=(4, 3))
        sub = sa[sa.tile == tile]
        for series, color, ls, lab in (("observed", "0.5", "-", "Observed"), ("AD_d8", "#0072B2", "--", "D8 model"),
                                       ("AD_dinf", "#D55E00", "--", "D-inf model")):
            s = sub[sub.series == series]
            Aa, Sa = s.A.to_numpy(), s.S.to_numpy()
            ax.loglog(Aa, Sa, ".", ms=1, alpha=0.15, color=color, label=lab, rasterized=True)
            if Aa.size > 10:
                slope_val, intercept, *_ = linregress(np.log10(Aa), np.log10(Sa))
                A_fit = np.logspace(np.log10(Aa.min()), np.log10(Aa.max()), 100)
                ax.plot(A_fit, 10**intercept * A_fit**slope_val, color=color if color != "0.5" else "0.3", ls=ls, lw=1.2)
        ax.set_xlabel(r"Drainage area $A$ (m$^2$)")
        ax.set_ylabel(r"Slope $S$")
        ax.set_title(label)
        ax.legend(markerscale=5, fontsize=6)
        fig.tight_layout()
        paths += save(fig, out_dir, f"d8_vs_dinf_{tile}_slope_area")

        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        a8 = np.full_like(A_dinf, np.nan)
        # D8 area is in the data store; reconstruct from the area scatter if available
        valid = np.isfinite(A_dinf) & (A_dinf > 0)
        ax.hist(np.log10(A_dinf[valid]), bins=60, color="#D55E00", alpha=0.6, label="D-inf")
        ax.set_xlabel(r"$\log_{10}(A_{\mathrm{D}\infty})$ (m$^2$)")
        ax.set_ylabel("cells")
        ax.set_title(f"{label}: D-inf area range [{res['A_dinf_range'][0]:.0f}, {res['A_dinf_range'][1]:.0f}]")
        del a8
        fig.tight_layout()
        paths += save(fig, out_dir, f"d8_vs_dinf_{tile}_area")
    return paths
