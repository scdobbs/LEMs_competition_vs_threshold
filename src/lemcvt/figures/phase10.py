"""Addendum 10 figures: the section 2 flux-form profiles on GM (MSE over (A_crit, D) per tile and S_c)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from lemcvt.figures.common import savefig_with_caption

FLUX_FORM = "flux form, div(D_eff grad z) = -U (Newton)"


def flux(out_dir: Path) -> list[Path]:
    """One panel per (tile, S_c): log10 of MSE / linear joint MSE over the converged evaluations, the 1.05 x linear joint
    line (the 2.3 fit condition), the outcome of every evaluation, the optimum, D0 and the 0.67-1.5 D0 band."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lemcvt.analyses.gm_flux import SC_LABELS, TILES, latest_merged

    hits = {(t, s): latest_merged(t, s) for t in TILES for s in SC_LABELS}
    if not any(hits.values()):
        raise FileNotFoundError("no a10_flux_merged runs")
    fig, axes = plt.subplots(len(TILES), len(SC_LABELS), figsize=(16, 17), squeeze=False)
    cap_rows: list[pd.DataFrame] = []
    cap_opt: list[dict] = []
    marks = {"no_steady_solution": ("x", "C3"), "not_converged": ("+", "k")}
    for i, t in enumerate(TILES):
        for j, s in enumerate(SC_LABELS):
            ax = axes[i, j]
            h = hits[(t, s)]
            if h is None:
                ax.set_axis_off()
                ax.set_title(f"{t} S_c {s}: no run", fontsize=8)
                continue
            r, m = h
            df = pd.read_parquet(r / "flux_rows.parquet")
            lin = float(m["linear_joint"]["mse"])
            cap_rows.append(df.assign(mse_over_linear_joint=df.mse / lin, run=r.name, model_form=FLUX_FORM)[
                ["tile", "sc_label", "S_c", "A_crit", "D", "mse", "mse_over_linear_joint", "outcome", "outcome_a9", "frac_clipped",
                 "frac_clipped_interior", "n_iter", "run", "model_form"]])
            conv = df[df.outcome == "converged"]
            if len(conv) >= 4 and conv.A_crit.nunique() > 1 and conv.D.nunique() > 1:
                A, D = np.sort(df.A_crit.unique()), np.sort(df.D.unique())
                M = conv.pivot_table(index="D", columns="A_crit", values="mse").reindex(index=D, columns=A).to_numpy() / lin
                cs = ax.pcolormesh(A, D, np.log10(M), shading="nearest", cmap="viridis", vmin=-0.05, vmax=1.0)
                if np.nanmin(M) <= 1.05:
                    ax.contour(A, D, M, levels=[1.05], colors="w", linewidths=1.0)
                fig.colorbar(cs, ax=ax, label="log10 MSE / linear joint MSE", fraction=0.046)
            for oc, (mk, col) in marks.items():
                g = df[df.outcome == oc]
                if len(g):
                    ax.plot(g.A_crit, g.D, mk, color=col, ms=3, label=f"{oc} ({len(g)})")
            o = m.get("optimum")
            if o:
                ax.plot([o["A_crit"]], [o["D"]], "*", color="w", mec="k", ms=10, label=f"best {o['A_crit']:.0f} m$^2$, D {o['D']:.2e}, x{m['mse_ratio_to_linear_joint']:.2f}")
            d0 = float(m["D0"])
            ax.axhline(d0, color="C1", ls="--", lw=0.8, label=f"D0 {d0:.2e}")
            ax.axhspan(0.67 * d0, 1.5 * d0, color="C1", alpha=0.12, lw=0)
            ax.axvline(float(m["linear_joint"]["A_crit"]), color="0.4", ls=":", lw=0.8)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{t}, S_c {m['S_c']:.3g} ({s}): {m['counts']['converged']}/{m['n_evaluations']} converged", fontsize=8)
            ax.legend(fontsize=5, loc="lower right")
            if i == len(TILES) - 1:
                ax.set_xlabel("A_crit (m$^2$)")
            if j == 0:
                ax.set_ylabel("D (m$^2$/yr)")
            cap_opt.append({"tile": t, "sc_label": s, "S_c": m["S_c"], "n_evaluations": m["n_evaluations"], "n_planned": m["n_planned"],
                            **{f"n_{k}": v for k, v in m["counts"].items()}, **{f"n_{k}_a9rule": v for k, v in m["counts_a9_rule"].items()},
                            "A_crit": (o or {}).get("A_crit"), "D": (o or {}).get("D"), "mse": (o or {}).get("mse"), "linear_joint_mse": lin,
                            "mse_ratio": m.get("mse_ratio_to_linear_joint"), "D0": d0, "D_over_D0": m.get("D_over_D0_A"), "run": r.name, "model_form": FLUX_FORM})
    fig.suptitle("Addendum 10, section 2: GM flux-form nonlinear profile over (A_crit, D), K at the linear joint optimum "
                 "(white line: MSE = 1.05 x linear joint; orange band: D/D0 0.67-1.5; dotted: linear joint A_crit)", fontsize=10)
    fig.tight_layout()
    out = out_dir / "a10_flux.png"
    rows = pd.concat(cap_rows, ignore_index=True) if cap_rows else pd.DataFrame()
    return savefig_with_caption(fig, out, {"rows": rows, "optima": cap_opt})
