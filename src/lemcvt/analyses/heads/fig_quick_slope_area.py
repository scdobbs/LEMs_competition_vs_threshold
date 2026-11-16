"""Quick look (checkpoint, no fits): drainage area against S_up_10 at the extracted hollow-class heads.

    python -m head_extraction.fig_quick_slope_area

Rows: Method P head sets (OWF as specified; OWF + DT2 smoothing). Columns: AP, GM.
Context: Montgomery & Dietrich (1992) envelope A S^2 = 500 and 4000 m^2, Kargere's r^2 on GM,
and each tile's A_crit bracket (stage 2 to final) drawn at that tile's median head slope.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .config import ANALYSIS, acrit_bracket  # noqa: E402
from .figures import C_MUTED, C_TEXT, _style  # noqa: E402

# categorical slots (dataviz reference palette), one per tile within a site
C_TILE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"]
SETS = [("OWF", "Method P, Wiener filter only (as specified)"), ("OWF+DT2", "Method P, Wiener + DT = 2 m² smoothing")]


def main(out_dir):
    from pathlib import Path

    out_dir = Path(out_dir)
    ACRIT_BRACKET = acrit_bracket()
    h = pd.read_csv(out_dir / "heads_extracted.csv")
    h = h[(h.method == "P") & h.retained & (h.head_class == "hollow") & np.isfinite(h.S_up_10) & (h.S_up_10 > 0)]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.6), facecolor="white", sharex=True, sharey=True)
    S = np.logspace(-1.5, 0.3, 50)
    lo, hi = ANALYSIS["md_envelope_AS2_m2"]
    for r, (sm, title) in enumerate(SETS):
        for c, site in enumerate(("AP", "GM")):
            ax = axes[r, c]
            ax.fill_between(S, lo / S**2, hi / S**2, color=C_MUTED, alpha=0.12, lw=0, label="M&D 1992 envelope, A S² = 500–4000 m²")
            if site == "GM":
                ax.axhline(ANALYSIS["kargere_r2_m2"], color=C_MUTED, lw=1.2, ls=(0, (4, 3)), label="Kargère et al. 2025, r² = 3844 m²")
            n_tot = 0
            for k, t in enumerate(range(1, 6)):
                sub = h[(h.site == site) & (h.tile == t) & (h.smoothing == sm)]
                n_tot += len(sub)
                ax.plot(sub.S_up_10, sub.area_D8_m2, "o", color=C_TILE[k], ms=5, mec="white", mew=0.5, alpha=0.9, ls="none",
                        label=f"{site}{t} (n = {len(sub)})")
                if len(sub):
                    xs = float(np.median(sub.S_up_10))
                    a2, af = ACRIT_BRACKET[(site, t)]
                    ax.plot([xs, xs], [a2, af], color=C_TILE[k], lw=3, alpha=0.55, solid_capstyle="butt")
                    ax.plot([xs], [af], "s", color=C_TILE[k], ms=6, mec=C_TEXT, mew=0.6)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1, 3e5)
            ax.set_xlim(0.03, 2.5)
            ax.set_title(f"{site}: {title}; {n_tot} hollow-class heads", fontsize=9, loc="left", color=C_TEXT)
            _style(ax)
            if r == 1:
                ax.set_xlabel("S_up_10 at the head (m/m)")
            if c == 0:
                ax.set_ylabel("drainage area at the head, D8 (m²)")
            ax.legend(fontsize=6.5, frameon=False, loc="lower left", ncol=2)
    fig.suptitle("Extracted heads: area against upslope-path slope (no fits; bars = A_crit bracket stage 2 → final at the tile's median head slope, square = final)",
                 fontsize=9.5, x=0.01, ha="left", color=C_TEXT)
    fig.tight_layout()
    out = out_dir / "figures" / "fig_quick_slope_area_heads.png"
    fig.savefig(out, dpi=180)
    print("wrote", out)

