"""Figures for `reports/v2/02c_measurement_diagnostics.md` (addendum 2, section 6). Read results only.

* ``theta_ladders``: theta*(A_c) with 68 % and 95 % basin-bootstrap bands, free against basin-fixed
  against interior-only, one panel per tile, plus the site-pooled curves;
* ``curvature_scale``: C_ht(L) for every tile (from the latest ``measure`` runs) with L* marked;
* ``ridge_bias``: recovered/true crest curvature against window length for the synthetic ridges,
  per crest width and variant, with each tile's L* marked;
* ``scaling_break``: binned log S - log A with the one-, two- and three-segment fits per tile.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lemcvt.config import TILE_IDS
from lemcvt.results import find_runs, latest_run


def _latest_diag() -> tuple[Path, dict]:
    runs = find_runs("measure_diagnostics")
    if not runs:
        raise FileNotFoundError("no measure_diagnostics run")
    return runs[-1], json.loads((runs[-1] / "params.json").read_text())


def _measure_curvature() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for tile in TILE_IDS:
        r = latest_run("measure", tile)
        if r is None:
            continue
        p = json.loads((r / "params.json").read_text())
        cur = p.get("tiles", {}).get(tile, {}).get("curvature", {})
        if "window_lengths_m" in cur:
            out[tile] = {**cur, "run": r.name}
    return out


def theta_ladders(out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run, p = _latest_diag()
    tiles = list(p["tiles"])
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.5), sharex=True, sharey=True)
    for ax, t in zip(axes.ravel(), tiles, strict=False):
        r = p["tiles"][t]
        for kind, color, label in (("fine_ladder", "C0", "free"), ("fixed_ladder", "C3", "basin-fixed (>= 50 px at 1e4)"),
                                   ("interior_ladder", "C2", "interior basins only")):
            steps = [s for s in r[kind] if s.get("theta_star") is not None]
            if not steps:
                continue
            a = np.array([s["A_c"] for s in steps])
            th = np.array([s["theta_star"] for s in steps])
            lo68 = np.array([(s.get("theta_ci68") or [np.nan, np.nan])[0] for s in steps])
            hi68 = np.array([(s.get("theta_ci68") or [np.nan, np.nan])[1] for s in steps])
            lo95 = np.array([(s.get("theta_ci95") or [np.nan, np.nan])[0] for s in steps])
            hi95 = np.array([(s.get("theta_ci95") or [np.nan, np.nan])[1] for s in steps])
            few = np.array([bool(s.get("flag_few_basins")) for s in steps])
            if kind == "fine_ladder":
                ax.fill_between(a, lo95, hi95, color=color, alpha=0.12, lw=0)
                ax.fill_between(a, lo68, hi68, color=color, alpha=0.25, lw=0)
            ax.plot(a, th, "-", color=color, lw=1.2, label=label)
            ax.plot(a[~few], th[~few], "o", color=color, ms=4)
            ax.plot(a[few], th[few], "o", mfc="none", mec=color, ms=4)
        ax.set_xscale("log")
        ax.set_title(t)
        ax.set_ylim(0.08, 0.92)
        ax.axhline(0.45, color="k", lw=0.5, ls=":")
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("A_c (m$^2$)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\theta^*$")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"theta*(A_c): filled = basin-bootstrap 68 % / 95 % bands (free ladder); open circles = fewer than 10 basins; run {run.name}",
                 fontsize=9)
    fig.tight_layout()
    out = out_dir / "02c_theta_ladders.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (site, pp) in zip(axes, p["pooled"].items(), strict=False):
        steps = [s for s in pp["ladder"] if s.get("theta_star") is not None]
        a = np.array([s["A_c"] for s in steps])
        th = np.array([s["theta_star"] for s in steps])
        lo68 = np.array([(s.get("theta_ci68") or [np.nan, np.nan])[0] for s in steps])
        hi68 = np.array([(s.get("theta_ci68") or [np.nan, np.nan])[1] for s in steps])
        lo95 = np.array([(s.get("theta_ci95") or [np.nan, np.nan])[0] for s in steps])
        hi95 = np.array([(s.get("theta_ci95") or [np.nan, np.nan])[1] for s in steps])
        nb = np.array([s["n_basins"] for s in steps])
        ax.fill_between(a, lo95, hi95, color="C0", alpha=0.12, lw=0)
        ax.fill_between(a, lo68, hi68, color="C0", alpha=0.25, lw=0)
        ax.plot(a, th, "o-", color="C0", ms=4, label="site-pooled (all basins of 5 tiles)")
        for t in pp["tiles"]:
            st = [s for s in p["tiles"][t]["fine_ladder"] if s.get("theta_star") is not None]
            ax.plot([s["A_c"] for s in st], [s["theta_star"] for s in st], "-", lw=0.7, alpha=0.6, label=t)
        for x, y, n in zip(a, th, nb, strict=True):
            if n < 10:
                ax.annotate(str(n), (x, y), fontsize=6, textcoords="offset points", xytext=(0, 5), ha="center")
        for key, color in (("scaling_break_le1e5", "C3"), ("scaling_break", "C1")):
            b = pp.get(key, {}).get("break_A_m2")
            if b:
                ax.axvline(b, color=color, lw=0.8, ls="--", label=f"slope-area break ({'A<=1e5' if 'le1e5' in key else 'all A'}): {b:.3g}")
        ax.set_xscale("log")
        ax.set_title(f"{site} pooled")
        ax.set_xlabel("A_c (m$^2$)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6)
    axes[0].set_ylabel(r"$\theta^*$")
    fig.tight_layout()
    out2 = out_dir / "02c_theta_ladders_pooled.png"
    fig.savefig(out2, dpi=130)
    plt.close(fig)
    return [out, out2]


def curvature_scale(out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cur = _measure_curvature()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, site in zip(axes, ("AP", "GM"), strict=True):
        for t, c in cur.items():
            if not t.startswith(site):
                continue
            L = np.array(c["window_lengths_m"])
            med = np.array(c["C_ht_median"])
            ax.plot(L, med, "o-", ms=3, label=f"{t} (L* = {c['L_star_m']:.0f} m)")
            ax.fill_between(L, c["C_ht_q16"], c["C_ht_q84"], alpha=0.1)
            ax.axvline(c["L_star_m"], color=ax.lines[-1].get_color(), lw=0.6, ls=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("window length L (m)")
        ax.set_title(f"{site}: median hilltop curvature against window (bands 16-84 %)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    axes[0].set_ylabel(r"$C_{ht} = -\nabla^2 z$ (1/m)")
    fig.tight_layout()
    out = out_dir / "02c_curvature_scale.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return [out]


def ridge_bias(out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run, p = _latest_diag()
    syn = p["synthetic_ridges"]
    cur = _measure_curvature()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, site in enumerate(("AP", "GM")):
        s = syn[site]
        L = np.array(s["window_lengths_m"])
        for row, variant in enumerate(("smooth", "tin_noise" if site == "AP" else "noise")):
            ax = axes[row, col]
            cases = sorted(s["cases"], key=lambda c: c["width_m"])
            cmap = plt.get_cmap("viridis")
            for k, c in enumerate(cases):
                rec = c["recovered"].get(variant)
                if rec is None:
                    continue
                ax.plot(L, np.array(rec) / s["c_true"], "-", color=cmap(k / max(1, len(cases) - 1)), lw=1,
                        label=f"W = {c['width_m']:.0f} m")
            for t, cc in cur.items():
                if t.startswith(site):
                    ax.axvline(cc["L_star_m"], color="k", lw=0.5, ls=":")
            ax.axhline(1.0, color="k", lw=0.6)
            ax.axhspan(0.9, 1.1, color="0.9", zorder=0)
            ax.set_xscale("log")
            ax.set_ylim(0, 1.6)
            ax.set_title(f"{site}, {variant} (facet {s['facet_len_m']:.1f} m, noise {s['noise_sigma']:.3f} m); dotted = tile L*", fontsize=9)
            ax.set_xlabel("window length L (m)")
            ax.set_ylabel("recovered / true crest curvature")
            ax.grid(alpha=0.3)
            if row == 0:
                ax.legend(fontsize=6, ncol=2)
    fig.suptitle(f"Parabolic-ridge test: recovered C_ht against window for crest widths W; run {run.name}", fontsize=9)
    fig.tight_layout()
    out = out_dir / "02c_ridge_bias.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return [out]


def scaling_break(out_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run, p = _latest_diag()
    tiles = list(p["tiles"])
    fig, axes = plt.subplots(2, 5, figsize=(17, 6.5), sharex=True, sharey=True)
    for ax, t in zip(axes.ravel(), tiles, strict=False):
        sb = p["tiles"][t]["scaling_break"]
        x, y, w = np.array(sb["bin_log10A"]), np.array(sb["bin_median_log10S"]), np.array(sb["bin_counts"])
        ax.scatter(x, y, s=8 + 20 * np.log10(np.maximum(w, 1)), color="0.3", zorder=3)
        xx = np.linspace(x.min(), x.max(), 200)
        if "one" in sb:
            ax.plot(xx, sb["one"]["intercept"] + sb["one"]["slope"] * xx, "C0-", lw=0.8, label=f"1 seg (BIC {sb['one']['bic']:.0f})")
        for key, color in (("scaling_break", "C3"), ("scaling_break_le1e5", "C2")):
            two = p["tiles"][t].get(key, {}).get("two")
            if two:
                b = two["break_log10A"]
                ax.axvline(b, color=color, lw=0.8, ls="--",
                           label=f"2-seg break {'(A<=1e5) ' if 'le1e5' in key else ''}{10 ** b:.3g} m$^2$ (BIC {two['bic']:.0f})")
        ax.axvline(np.log10(sb["A_rollover"]), color="C1", lw=0.8, ls=":", label=f"rollover {sb['A_rollover']:.3g}")
        ax.set_title(f"{t} (best: {sb.get('bic_best')})", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=5.5)
    for ax in axes[1]:
        ax.set_xlabel("log10 A (m$^2$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("median log10 S (0.1-decade bins)")
    fig.suptitle(f"Slope-area above the rollover with piecewise-linear fits; run {run.name}", fontsize=9)
    fig.tight_layout()
    out = out_dir / "02c_scaling_break.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return [out]


def build_all(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    return theta_ladders(out_dir) + curvature_scale(out_dir) + ridge_bias(out_dir) + scaling_break(out_dir)


def audit_estimators(out_dir: Path) -> list[Path]:
    """02e (addendum 3, section 4): theta*(A_c) by the pooled, equal-weight and median estimators with 95 % bands,
    and n_eff of the pooled estimator, per tile and site-pooled."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = find_runs("measure_audit")
    if not runs:
        raise FileNotFoundError("no measure_audit run")
    run = runs[-1]
    p = json.loads((run / "params.json").read_text())
    tiles = list(p["tiles"])
    panels = tiles + [f"{s}_pooled" for s in p["pooled"]]
    fig, axes = plt.subplots(3, 4, figsize=(17, 10), sharex=True)
    for ax, t in zip(axes.ravel(), panels, strict=False):
        lad = p["tiles"][t]["free_ladder"] if t in p["tiles"] else p["pooled"][t.split("_")[0]]["ladder"]
        ax2 = ax.twinx()
        for est, color, label in (("pooled", "C0", "pooled (02c)"), ("equal", "C2", "equal-weight"), ("median", "C1", "basin median")):
            steps = [s for s in lad if s.get(est) and s[est].get("theta_star") is not None]
            if not steps:
                continue
            a = np.array([s["A_c"] for s in steps])
            th = np.array([s[est]["theta_star"] for s in steps])
            lo = np.array([(s[est].get("theta_ci95") or [np.nan, np.nan])[0] for s in steps])
            hi = np.array([(s[est].get("theta_ci95") or [np.nan, np.nan])[1] for s in steps])
            ax.fill_between(a, lo, hi, color=color, alpha=0.12, lw=0)
            ax.plot(a, th, "o-", color=color, ms=3, lw=1.1, label=label)
        steps = [s for s in lad if s.get("pooled")]
        if steps:
            a = np.array([s["A_c"] for s in steps])
            ne = np.array([s["pooled"]["n_eff_centred"] for s in steps])
            nb = np.array([s["n_basins"] for s in steps])
            ax2.plot(a, nb, "--", color="0.5", lw=0.8, label="basins")
            ax2.plot(a, ne, "-", color="k", lw=0.8, label="n_eff (pooled)")
            ax2.axhline(3, color="k", lw=0.5, ls=":")
            ax2.set_yscale("log")
            ax2.set_ylim(0.8, 1000)
        ax.set_xscale("log")
        ax.set_ylim(0.08, 0.92)
        ax.set_title(t, fontsize=9)
        ax.grid(alpha=0.3)
        if t == panels[0]:
            ax.legend(fontsize=6, loc="upper left")
            ax2.legend(fontsize=6, loc="lower right")
    for ax in axes[-1]:
        ax.set_xlabel("A_c (m$^2$)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"$\theta^*$")
    fig.suptitle(f"theta*(A_c) by estimator (bands: 95 % basin bootstrap); right axis: basins (dashed) and n_eff of the pooled estimator (solid, dotted line at 3); run {run.name}",
                 fontsize=9)
    fig.tight_layout()
    out = out_dir / "02e_theta_estimators.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)

    # hilltop curvature with and without exclusion per network
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, site in zip(axes, ("AP", "GM"), strict=True):
        st = [t for t in tiles if t.startswith(site)]
        x = np.arange(len(st))
        w = 0.13
        k = 0
        for net, color in (("1000", "C0"), ("10000", "C3"), ("rollover", "C2")):
            for s, hatch in (("all", None), ("excluded", "//")):
                vals, los, his = [], [], []
                for t in st:
                    r = p["tiles"][t]["hilltops"]["networks"].get(net, {}).get(s, {})
                    a = r.get("at_fixed_L", {})
                    vals.append(a.get("C_ht", np.nan))
                    ci = a.get("C_ht_ci68") or [np.nan, np.nan]
                    los.append(ci[0])
                    his.append(ci[1])
                vals, los, his = map(np.array, (vals, los, his))
                ax.bar(x + (k - 2.5) * w, vals, w, color=color, alpha=0.9 if s == "all" else 0.5, hatch=hatch, edgecolor="k", lw=0.3,
                       yerr=np.vstack([vals - los, his - vals]), capsize=1.5, error_kw={"lw": 0.6},
                       label=f"A_c {net}{' m^2' if net != 'rollover' else ''}, {s}")
                k += 1
        mr = [p["tiles"][t]["hilltops"].get("measure_run_C_ht", {}).get("C_ht") or np.nan for t in st]
        ax.plot(x, mr, "k_", ms=14, mew=1.5, label="02b measure run (rollover network, no exclusion)")
        ax.set_xticks(x)
        ax.set_xticklabels(st)
        ax.set_ylabel("C_ht at the tile's L* (1/m)")
        ax.set_title(f"{site}: hilltop curvature by network threshold and exclusion (bars: 68 % segment bootstrap)", fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        if site == "AP":
            ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    out2 = out_dir / "02e_hilltop_exclusion.png"
    fig.savefig(out2, dpi=130)
    plt.close(fig)
    return [out, out2]
