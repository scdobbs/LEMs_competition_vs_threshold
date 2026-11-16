"""Fig. 3 (dK, dD): log10 ratios of fitted to measured K and D per tile.

Addendum 7, 1.3: built from the per-stage Phase 6 runs (metres, site m, the configs with the measured block): the AD
optimum (v2 r profile), the linear joint optimum (refined partitioned run) and the nonlinear joint optimum (merged
run), with dK = log10(K_fit / K0), K0 = U/k_s, and dD = log10(D_fit / D0). On AP, D0 is a bracket and dD is drawn at
both ends (D0_A filled, D0_B open, joined); on GM at D0 (= D0_A). The manuscript figure's code is not in the v1
repository.

The nonlinear joint run is the Phase 6 one at S_c = 1.27 (no S_c override, K factor 1), as
``lemcvt.analyses.nonlinear_fair._phase6_nonlinear_joint_optimum``: the addendum 9 runs at the fitted S_c are later
and would otherwise be taken as the latest. Its panel carries the addendum 10, 1.1 label (centre form, nonstandard).
The plotted values go to ``fig3_deltaK_deltaD.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lemcvt.config import TILE_IDS
from lemcvt.figures.common import CENTRE_FORM, CENTRE_FORM_TAG, COL_AP, COL_GM, plt, save, set_nature_style

STAGES = (("ad", "Advection-diffusion"), ("joint", "Partitioned (linear joint)"), ("nonlinear_joint", f"Partitioned (nonlinear joint, S_c 1.27)\n{CENTRE_FORM_TAG}"))


def collect_deltas() -> pd.DataFrame:
    import json

    from lemcvt.analyses.curvature_compare import _runs_by_stage
    from lemcvt.analyses.nonlinear_fair import _phase6_nonlinear_joint_optimum

    rows = []
    for tile in TILE_IDS:
        runs = _runs_by_stage(tile)
        runs.pop("nonlinear_joint", None)   # _runs_by_stage does not exclude S_c overrides (addendum 9 runs)
        p6 = _phase6_nonlinear_joint_optimum(tile)
        if p6 is not None:
            runs["nonlinear_joint"] = (p6["dir"], json.loads((p6["dir"] / "params.json").read_text()))
        for stage, _title in STAGES:
            if stage not in runs:
                continue
            r, p = runs[stage]
            mdl = p["model"]
            K0, D0_A, D0_B = float(mdl["K0_partitioned"]), float(mdl["D0"]), mdl.get("D0_B")
            if stage == "ad":
                o = p["v2"]["ad"]["optimum"]
            elif stage == "joint":
                o = p["v2"]["partitioned"]["joint_optimum"]
            else:
                o = p["v2"]["nonlinear_joint"]["optimum"]
            ends = [("A", D0_A)] + ([("B", float(D0_B))] if tile.startswith("AP") and D0_B else [])
            for end, D0 in ends:
                rows.append({"tile": tile, "site": tile[:2], "stage": stage, "D0_end": end, "K0": K0, "D0": D0, "K": o["K"], "D": o["D"],
                             "dK": np.log10(o["K"] / K0), "dD": np.log10(o["D"] / D0), "mse": o["mse"], "run": r.name,
                             "model_form": CENTRE_FORM if stage == "nonlinear_joint" else ""})
    if not rows:
        raise FileNotFoundError("no Phase 6 per-stage runs (AD, refined partitioned, merged nonlinear joint) found")
    return pd.DataFrame(rows)


def build(out_dir: Path) -> list[Path]:
    set_nature_style()
    df = collect_deltas()
    fig, axes = plt.subplots(1, len(STAGES), figsize=(3.0 * len(STAGES), 2.9), sharex=True, sharey=True)
    for ax, (stage, title) in zip(axes, STAGES, strict=True):
        sub = df[df.stage == stage]
        for site, color, mk in (("GM", COL_GM, "o"), ("AP", COL_AP, "s")):
            s = sub[sub.site == site]
            a = s[s.D0_end == "A"]
            b = s[s.D0_end == "B"]
            ax.scatter(a.dK, a.dD, c=color, marker=mk, s=18, label=f"{site} (D0_A)" if site == "AP" else site, zorder=3)
            if len(b):
                ax.scatter(b.dK, b.dD, facecolors="none", edgecolors=color, marker=mk, s=18, label="AP (D0_B)", zorder=3)
                for t in b.tile:
                    ra, rb = a[a.tile == t].iloc[0], b[b.tile == t].iloc[0]
                    ax.plot([ra.dK, rb.dK], [ra.dD, rb.dD], color=color, lw=0.5, alpha=0.7, zorder=2)
            for _, r in a.iterrows():
                ax.annotate(r.tile[2:], (r.dK, r.dD), fontsize=5, xytext=(2, 2), textcoords="offset points")
        ax.axhline(0, color="0.5", lw=0.5)
        ax.axvline(0, color="0.5", lw=0.5)
        ax.set_title(title, fontsize=7)
        ax.set_xlabel(r"$\Delta K = \log_{10}(K_{\mathrm{fit}}/K_0)$")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"$\Delta D = \log_{10}(D_{\mathrm{fit}}/D_0)$")
    axes[0].legend(frameon=False, fontsize=5)
    fig.tight_layout()
    return save(fig, out_dir, "fig3_deltaK_deltaD", caption=df)
