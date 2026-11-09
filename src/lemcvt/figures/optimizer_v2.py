"""Figures for `reports/v2/04_optimizer_v2.md` (Phase 4). Read results only.

* ``acrit_profiles``: MSE(A_crit) of the linear partitioned model per tile (stage-2 objective at K0, D0 and the
  joint objective with the exact (U/K, U/D) minimum), with the v1 stage-2 and joint A_crit marked, the 1 % and
  5 % bands, and the nonlinear profile where it exists;
* ``ad_profiles``: MSE*(r) over r = K/D per tile with the v1 and v2 optima marked;
* ``m_contours``: MSE(A_crit, m) with the 1 % contour and the measured theta(A_c) curve (02e equal-weight and
  02c pooled, with 95 % bands) on the same axes (addendum 3, section 5.2, item 3).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lemcvt.config import TILE_IDS
from lemcvt.results import find_runs


def _v2_runs() -> dict[str, dict[str, tuple[Path, dict]]]:
    """Latest v2 tile_run per tile and stage group ('ad', 'partitioned', 'nonlinear', 'm_profile')."""
    out: dict[str, dict[str, tuple[Path, dict]]] = {}
    for r in find_runs("tile_run"):
        try:
            p = json.loads((r / "params.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        if p.get("optimizer") != "v2":
            continue
        t = p["tile"]
        for st in p.get("v2", {}):
            out.setdefault(t, {})[st] = (r, p)
    return out


def _theta_curves() -> dict[str, dict]:
    """Measured theta(A_c) per tile: pooled (02c diagnostics) and equal-weight (02e audit), with 95 % bands."""
    curves: dict[str, dict] = {}
    runs = find_runs("measure_audit")
    if runs:
        p = json.loads((runs[-1] / "params.json").read_text())
        for t, r in p.get("tiles", {}).items():
            for est in ("pooled", "equal"):
                steps = [s for s in r["free_ladder"] if s.get(est) and s[est].get("theta_star") is not None]
                curves.setdefault(t, {})[est] = {
                    "A_c": np.array([s["A_c"] for s in steps]), "theta": np.array([s[est]["theta_star"] for s in steps]),
                    "lo95": np.array([(s[est].get("theta_ci95") or [np.nan, np.nan])[0] for s in steps]),
                    "hi95": np.array([(s[est].get("theta_ci95") or [np.nan, np.nan])[1] for s in steps]),
                    "n": np.array([s["n_basins"] for s in steps]), "run": runs[-1].name}
    return curves


def acrit_profiles(out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = _v2_runs()
    tiles = [t for t in TILE_IDS if t in runs and "partitioned" in runs[t]]
    if not tiles:
        raise FileNotFoundError("no v2 partitioned runs")
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex=True)
    for ax, t in zip(axes.ravel(), tiles, strict=False):
        r, p = runs[t]["partitioned"]
        prof = pd.read_parquet(r / "acrit_profile.parquet")
        v2p = p["v2"]["partitioned"]
        ref = p.get("v1_reference", {})
        ax.plot(prof.A_crit, prof.mse_stage2, "-", color="C0", lw=1, label="stage 2: K0, D0 fixed")
        ax.plot(prof.A_crit, prof.mse_joint, "-", color="C3", lw=1.2, label="joint: exact (K, D) at each A_crit")
        jo = v2p["joint_optimum"]
        s2 = v2p["stage2_optimum"]
        for lev, alpha in ((0.05, 0.10), (0.01, 0.25)):
            ax.axhspan(jo["mse"], jo["mse"] * (1 + lev), color="C3", alpha=alpha, lw=0)
        ax.plot([s2["A_crit"]], [s2["mse"]], "o", color="C0", ms=6, label=f"v2 stage 2: {s2['A_crit']:.0f}")
        ax.plot([jo["A_crit"]], [jo["mse"]], "o", color="C3", ms=6, label=f"v2 joint: {jo['A_crit']:.0f}")
        if "acrit" in ref:
            ax.axvline(ref["acrit"]["A_crit"], color="C0", ls="--", lw=0.8, label=f"v1 stage 2: {ref['acrit']['A_crit']:.0f}")
        if "joint" in ref:
            ax.axvline(ref["joint"]["A_crit"], color="C3", ls="--", lw=0.8, label=f"v1 joint: {ref['joint']['A_crit']:.0f}")
            cert = v2p.get("certificate", {}).get("v1_joint")
            if cert:
                ax.plot([cert["A_crit"]], [cert["mse"]], "x", color="C3", ms=7, label="v1 joint point, same code")
        if "nonlinear" in runs[t]:
            rn, pn = runs[t]["nonlinear"]
            nlp = pd.read_parquet(rn / "nonlinear_profile.parquet")
            nlp = nlp[nlp.kind.isin(["grid", "refine"])].sort_values("A_crit")
            ax.plot(nlp.A_crit, nlp.mse, ".-", color="C2", lw=0.8, ms=3, label="nonlinear (K0, D0, S_c)")
            if "nonlinear" in ref:
                ax.axvline(ref["nonlinear"]["A_crit"], color="C2", ls=":", lw=0.8, label=f"v1 nonlinear: {ref['nonlinear']['A_crit']:.0f}")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"{t} (grid: {v2p['grid']['n_used']} of {v2p['grid']['n_unique']} unique areas)", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=5.5, loc="upper right")
    for ax in axes[1]:
        ax.set_xlabel("A_crit (m$^2$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE (z unit$^2$)")
    fig.suptitle("Phase 4: MSE(A_crit) profiles of the linear partitioned model (bands: 1 % and 5 % above the joint minimum)", fontsize=10)
    fig.tight_layout()
    out = out_dir / "04_acrit_profiles.png"
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return [out, out.with_suffix(".pdf")]


def ad_profiles(out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = _v2_runs()
    tiles = [t for t in TILE_IDS if t in runs and "ad" in runs[t]]
    if not tiles:
        raise FileNotFoundError("no v2 ad runs")
    fig, axes = plt.subplots(2, 5, figsize=(18, 6.5), sharex=False)
    for ax, t in zip(axes.ravel(), tiles, strict=False):
        r, p = runs[t]["ad"]
        v2a = p["v2"]["ad"]
        rows = [q for q in v2a["grid_rows"] if q.get("feasible")]
        rr = np.array([q["r"] for q in rows])
        mm = np.array([q["mse"] for q in rows])
        o = np.argsort(rr)
        ax.plot(rr[o], mm[o], ".-", color="C1", ms=3, lw=0.8, label="MSE*(r), closed-form U/D")
        opt = v2a["optimum"]
        ax.plot([opt["r"]], [opt["mse"]], "o", color="C1", ms=6, label=f"v2: K {opt['K']:.3g}, D {opt['D']:.3g}")
        c1 = v2a.get("certificate", {}).get("v1_optimum")
        if c1:
            ax.plot([c1["K"] / c1["D"]], [c1["mse_same_code"]], "x", color="k", ms=8, label=f"v1: K {c1['K']:.3g}, D {c1['D']:.3g}")
        for lev, alpha in ((0.05, 0.10), (0.01, 0.25)):
            ax.axhspan(opt["mse"], opt["mse"] * (1 + lev), color="C1", alpha=alpha, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(t, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    for ax in axes[1]:
        ax.set_xlabel("r = K / D")
    for ax in axes[:, 0]:
        ax.set_ylabel("MSE*(r)")
    fig.suptitle("Phase 4: AD profile over r = K/D (bands: 1 % and 5 % above the minimum)", fontsize=10)
    fig.tight_layout()
    out = out_dir / "04_ad_profiles.png"
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return [out, out.with_suffix(".pdf")]


def m_contours(out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = _v2_runs()
    tiles = [t for t in TILE_IDS if t in runs and "m_profile" in runs[t]]
    if not tiles:
        raise FileNotFoundError("no v2 m_profile runs")
    curves = _theta_curves()
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharey=True)
    for ax, t in zip(axes.ravel(), tiles, strict=False):
        r, p = runs[t]["m_profile"]
        df = pd.read_parquet(r / "m_profile.parquet")
        piv = df.pivot_table(index="m", columns="A_crit", values="mse_joint")
        X, Y = np.meshgrid(piv.columns.to_numpy(), piv.index.to_numpy())
        Z = piv.to_numpy()
        zmin = np.nanmin(Z)
        levels = zmin * np.array([1.01, 1.05, 1.2, 1.5, 2, 3, 5, 10])
        cs = ax.contourf(X, Y, Z, levels=np.concatenate([[zmin], levels]), cmap="Blues_r", alpha=0.85)
        ax.contour(X, Y, Z, levels=[zmin * 1.01], colors="C3", linewidths=1.5)
        ax.contour(X, Y, Z, levels=[zmin * 1.05], colors="C3", linewidths=0.8, linestyles="--")
        o = p["v2"]["m_profile"]["optimum"]
        ax.plot([o["A_crit"]], [o["m"]], "*", color="C3", ms=12, label=f"fitted (A_crit, m) = ({o['A_crit']:.0f}, {o['m']:.2f})")
        cfg_m = p["model"]["m"]
        ax.axhline(cfg_m, color="k", lw=0.6, ls=":", label=f"config m = {cfg_m:.3f}")
        cv = curves.get(t, {})
        for est, color, lab in (("pooled", "C1", "measured theta(A_c), pooled (02c)"), ("equal", "C2", "measured theta(A_c), equal-weight (02e)")):
            c = cv.get(est)
            if c is None:
                continue
            ax.fill_between(c["A_c"], c["lo95"], c["hi95"], color=color, alpha=0.15, lw=0)
            ax.plot(c["A_c"], c["theta"], "-", color=color, lw=1.3, label=lab)
        ax.set_xscale("log")
        ax.set_xlim(1e2, 1e5)
        ax.set_ylim(0.2, 0.7)
        ax.set_title(t, fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=5.5, loc="lower right")
    for ax in axes[1]:
        ax.set_xlabel("A_crit / A_c (m$^2$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("m / theta")
    fig.colorbar(cs, ax=axes, shrink=0.6, label="MSE (filled: multiples of the minimum)")
    fig.suptitle("Addendum 3 section 5.2: MSE(A_crit, m) with the 1 % (solid) and 5 % (dashed) contours, and the measured theta(A_c) with 95 % bands", fontsize=10)
    out = out_dir / "04_m_contours.png"
    fig.savefig(out, dpi=130)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    return [out, out.with_suffix(".pdf")]


def build_all(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fn in (acrit_profiles, ad_profiles, m_contours):
        try:
            paths += fn(out_dir)
        except FileNotFoundError:
            continue
    return paths
