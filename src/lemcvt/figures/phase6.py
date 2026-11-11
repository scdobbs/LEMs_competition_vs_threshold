"""Figures for the Phase 6 report (addendum 5). Read results only.

* ``window_test``: section 2.2, median C_ht(L) on the option (iii) pixel set with the segment-bootstrap 68 % bands,
  one panel per tile, the +-10 % band around C_ht(L_f) and the tested window range (L_f to the set's median crest width);
* ``nonlinear_joint``: section 3.4, MSE over (A_crit, D) per tile with the 1 % and 5 % contours (when the runs exist).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lemcvt.config import TILE_IDS
from lemcvt.results import find_runs


def _latest(step: str):
    for r in reversed(find_runs(step)):
        try:
            rj = json.loads((r / "run.json").read_text())
            if (rj.get("config") or {}).get("crop"):
                continue
            return r, json.loads((r / "params.json").read_text())
        except Exception:  # noqa: BLE001
            continue
    raise FileNotFoundError(f"no {step} run")


def window_test(out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run, p = _latest("phase6_inputs")
    tiles = [t for t in TILE_IDS if t in p["tiles"]]
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.5), sharex=True)
    for ax, t in zip(axes.ravel(), tiles, strict=False):
        o = p["tiles"][t]["option_iii"]
        rows = pd.DataFrame(o["rows"])
        L = rows.L_nominal_m.to_numpy(float)
        C = rows.C_ht.to_numpy(float)
        lo = np.array([(r or [np.nan, np.nan])[0] for r in rows.get("C_ht_ci68", [None] * len(rows))], float)
        hi = np.array([(r or [np.nan, np.nan])[1] for r in rows.get("C_ht_ci68", [None] * len(rows))], float)
        Cf = o["C_ht_at_L_f"]
        W50 = o["test"]["median_crest_width_m"]
        ax.axhspan(0.9 * Cf, 1.1 * Cf, color="C2", alpha=0.15, lw=0, label="+-10 % of C_ht(L_f)")
        ax.axvspan(11, max(W50, 11), color="0.85", alpha=0.6, lw=0, label=f"tested range: L_f to W50 = {W50:.1f} m")
        ax.fill_between(L, lo, hi, color="C0", alpha=0.25, lw=0, label="68 % segment bootstrap")
        ax.plot(L, C, "o-", color="C0", ms=4, lw=1, label="median C_ht(L), option (iii) set")
        ax.axvline(11, color="C3", ls="--", lw=0.8, label="L_f = 11 m")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([7, 11, 21, 51])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ok = o["test"]["pass"]
        ax.set_title(f"{t}: {'PASS' if ok else 'FAIL'} (max dev {100 * o['test']['max_abs_deviation']:.0f} %, {o['n_pixels']} px, {o['n_segments']} seg)",
                     fontsize=9, color="C2" if ok else "C3")
        ax.grid(alpha=0.3, which="both")
        if t == tiles[0]:
            ax.legend(fontsize=6, loc="lower left")
    for ax in axes[1]:
        ax.set_xlabel("window length L (m, nominal)")
    for ax in axes[:, 0]:
        ax.set_ylabel("median -laplacian (1/m)")
    fig.suptitle(f"Addendum 5, 2.2: C_ht(L) on the option (iii) pixel set (run {run.name}); pass = within +-10 % of C_ht(L_f) from L_f up to the median crest width",
                 fontsize=10)
    fig.tight_layout()
    out = out_dir / "a5_option_iii_window_test.png"
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return [out, out.with_suffix(".pdf")]


def nonlinear_joint(out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs: dict[str, tuple[Path, dict]] = {}
    for r in find_runs("tile_run"):
        try:
            p = json.loads((r / "params.json").read_text())
            rj = json.loads((r / "run.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        if "nonlinear_joint" not in p.get("v2", {}) or (rj.get("config") or {}).get("crop") or p["v2"]["nonlinear_joint"]["setup"]["k_factor"] != 1.0:
            continue
        runs[p["tile"]] = (r, p)
    tiles = [t for t in TILE_IDS if t in runs]
    if not tiles:
        raise FileNotFoundError("no nonlinear joint runs")
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    for ax, t in zip(axes.ravel(), tiles, strict=False):
        r, p = runs[t]
        df = pd.read_parquet(r / "nonlinear_joint_profile.parquet")
        v = p["v2"]["nonlinear_joint"]
        A = np.sort(df.A_crit.unique())
        D = np.sort(df.D.unique())
        M = df.pivot(index="D", columns="A_crit", values="mse").loc[D, A].to_numpy()
        fmin = v["contours"]["mse_min"]
        cs = ax.contourf(A, D, np.log10(M / fmin), levels=np.linspace(0, np.log10(np.nanmax(M / fmin)), 20), cmap="viridis")
        ax.contour(A, D, M / fmin, levels=[1.01, 1.05], colors=["w", "k"], linewidths=[1.2, 0.8])
        cap = df[~df.converged]
        if len(cap):
            ax.plot(cap.A_crit, cap.D, "x", color="C3", ms=3, label=f"Picard cap hit ({len(cap)})")
        o = v["optimum"]
        ax.plot([o["A_crit"]], [o["D"]], "*", color="w", mec="k", ms=10, label=f"optimum: {o['A_crit']:.0f} m$^2$, D {o['D']:.2e}")
        ax.axhline(p["model"]["D0"], color="C1", ls="--", lw=0.8, label=f"D0 {p['model']['D0']:.2e}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{t} (K = {v['setup']['K']:.2e} 1/yr)", fontsize=9)
        ax.legend(fontsize=5.5, loc="lower right")
        fig.colorbar(cs, ax=ax, label="log10 MSE/MSE_min")
    for ax in axes[1]:
        ax.set_xlabel("A_crit (m$^2$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("D (m$^2$/yr)")
    fig.suptitle("Addendum 5, 3.4: nonlinear partitioned MSE over (A_crit, D) at the linear joint K (white: 1 %, black: 5 % contour)", fontsize=10)
    fig.tight_layout()
    out = out_dir / "a5_nonlinear_joint.png"
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return [out, out.with_suffix(".pdf")]
