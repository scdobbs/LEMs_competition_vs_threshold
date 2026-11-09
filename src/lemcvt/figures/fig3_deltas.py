"""Fig. 3 (dK, dD): log10 ratios of fitted to measured K and D per tile, AD and joint stages.

The manuscript figure's code is not in the v1 repository; this implementation uses
the tile_run results (latest run per tile) with dK = log10(K_fit / K0) and
dD = log10(D_fit / D0), where K0 and D0 are the measured initial values recorded in
the run (K0_partitioned for the joint stage, K0_ad for the AD stage; D0 for both).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lemcvt.config import TILE_IDS
from lemcvt.figures.common import COL_AP, COL_GM, load_params, plt, save, set_nature_style
from lemcvt.results import latest_run


def collect_deltas() -> pd.DataFrame:
    rows = []
    for tile in TILE_IDS:
        run = latest_run("tile_run", tile)
        if run is None:
            continue
        p = load_params(run)
        st = p.get("stages_results", {})
        mdl = p["model"]
        for stage in ("ad", "joint"):
            if stage not in st:
                continue
            K0 = mdl["K0_ad"] if stage == "ad" else mdl["K0_partitioned"]
            r = st[stage]["params"]
            rows.append({"tile": tile, "site": tile[:2], "stage": stage, "K0": K0, "D0": mdl["D0"], "K": r["K"], "D": r["D"],
                         "dK": np.log10(r["K"] / K0), "dD": np.log10(r["D"] / mdl["D0"]), "mse": st[stage]["mse"],
                         "run": run.name})
    if not rows:
        raise FileNotFoundError("no tile_run results with AD/joint stages")
    return pd.DataFrame(rows)


def build(out_dir: Path) -> list[Path]:
    set_nature_style()
    df = collect_deltas()
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.8), sharex=True, sharey=True)
    for ax, stage in zip(axes, ("ad", "joint"), strict=True):
        sub = df[df.stage == stage]
        for site, color, mk in (("GM", COL_GM, "o"), ("AP", COL_AP, "s")):
            s = sub[sub.site == site]
            ax.scatter(s.dK, s.dD, c=color, marker=mk, s=18, label=site, zorder=3)
            for _, r in s.iterrows():
                ax.annotate(r.tile[2:], (r.dK, r.dD), fontsize=5, xytext=(2, 2), textcoords="offset points")
        ax.axhline(0, color="0.5", lw=0.5)
        ax.axvline(0, color="0.5", lw=0.5)
        ax.set_title({"ad": "Advection-diffusion", "joint": "Partitioned (joint)"}[stage])
        ax.set_xlabel(r"$\Delta K = \log_{10}(K_{\mathrm{fit}}/K_0)$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"$\Delta D = \log_{10}(D_{\mathrm{fit}}/D_0)$")
    axes[0].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    paths = save(fig, out_dir, "fig3_deltaK_deltaD")
    df.to_csv(out_dir / "fig3_deltaK_deltaD.csv", index=False)
    return paths + [out_dir / "fig3_deltaK_deltaD.csv"]
