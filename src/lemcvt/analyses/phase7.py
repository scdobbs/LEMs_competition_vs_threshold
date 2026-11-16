"""Phase 7 (directive; addendum 8, 1.4): the add-on analyses as lemcvt steps, run on the metre grids.

* ``run_width``: the channel-width analysis of feature/channel-width (Phase 3 sample, fits, regime outputs, figures);
* ``run_width_constrained``: its constrained re-run at A >= 1e5 m^2;
* ``run_heads``: Test 2 head extraction (Method P) up to its review gate;
* ``run_port_check``: the regression check of each port against the feature branch's committed outputs on GM (the
  branch read the GeoTIFFs directly, so its GM grids were the same metre grids; its AP grids were in feet), and the
  AP change between the feet-based branch outputs and the metre rerun.

Each step is a run under ``$LEMCVT_RESULTS`` with the package's own outputs in ``<run>/outputs``; Test 1
(``acrit_slope``) is not ported: addendum 8, 3 replaces it.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from lemcvt.config import REPO_ROOT, config_snapshot, load_tile
from lemcvt.results import RunRecorder, find_runs

logger = logging.getLogger(__name__)

BRANCH_FILES = {
    "width": ("feature/channel-width", "outputs/widths", ["points.csv", "excluded.csv", "fits.json", "bins.json", "run_info.json"]),
    "width_constrained": ("feature/channel-width", "outputs/widths_constrained", ["points.csv", "excluded.csv", "fits.json", "bins.json", "run_info.json"]),
    "heads": ("feature/head-extraction", "outputs/head_extraction", ["heads_extracted.csv", "class_summary.csv", "threshold_sensitivity.csv"]),
}


def _snapshot(tiles) -> dict[str, Any]:
    return {t: config_snapshot(load_tile(t, sanity=False).cfg) for t in tiles}


def run_width(sites, mode: str = "full", make_qc: bool = True, run_id: str | None = None, results_root=None) -> RunRecorder:
    from lemcvt.analyses.width.config import CONFIG, TILES
    from lemcvt.analyses.width.run import run

    rec = RunRecorder("width", "_".join(sites) + f"_{mode}", run_id=run_id, root=results_root)
    t0 = time.perf_counter()
    out = rec.dir / "outputs"
    df = run(list(sites), mode, out, cfg=CONFIG, make_qc=make_qc)
    summary: dict[str, Any] = {"n_points": int(len(df)), "sites": list(sites), "mode": mode, "outputs": str(out)}
    for site in sites:
        d = df[df.site == site]
        summary[site] = {"n_points": int(len(d)), "n_channelized": int(d.channelized.sum()), "n_resolved": int(d.resolved_width.sum()),
                         "median_W_primary": float(d.W_primary.median()), "median_local_slope": float(d.local_slope.median()),
                         "n_steep_rill": int(d.steep_rill_flag.sum())}
    if (out / "fits.json").exists():
        summary["fits"] = json.loads((out / "fits.json").read_text())
    rec.add_params("summary", summary)
    rec.add_params("wall_s", time.perf_counter() - t0)
    rec.finalize({"tiles": _snapshot([f"{s}{t}" for s in sites for t in TILES]), "mode": mode})
    return rec


def run_width_constrained(stage: str = "full", run_id: str | None = None, results_root=None) -> RunRecorder:
    from lemcvt.analyses.width.config import CONFIG, CONSTRAINED, SITES, TILES
    from lemcvt.analyses.width.run_constrained import run

    rec = RunRecorder("width_constrained", stage, run_id=run_id, root=results_root)
    t0 = time.perf_counter()
    out = rec.dir / "outputs"
    df = run(stage, out, cfg=CONFIG, ccfg=CONSTRAINED)
    summary: dict[str, Any] = {"n_points": int(len(df)), "stage": stage, "outputs": str(out), "class_counts": df.point_class.value_counts().to_dict()}
    for site in SITES:
        d = df[df.site == site]
        res = d[d.point_class.isin(("inner_channel", "one_bank"))]
        summary[site] = {"n_points": int(len(d)), "class_counts": d.point_class.value_counts().to_dict(), "n_resolved": int(len(res)),
                         "median_W_trench": float(res.W_trench.median()) if len(res) else None, "median_W_bed": float(res.W_bed.median()) if len(res) else None,
                         "median_trench_depth_m": float(res.trench_depth_m.median()) if len(res) else None}
    if (out / "fits.json").exists():
        summary["fits"] = json.loads((out / "fits.json").read_text())
    rec.add_params("summary", summary)
    rec.add_params("wall_s", time.perf_counter() - t0)
    rec.finalize({"tiles": _snapshot([f"{s}{t}" for s in SITES for t in TILES]), "stage": stage})
    return rec


def run_heads(tiles, use_lsdtt: bool = False, run_id: str | None = None, results_root=None) -> RunRecorder:
    from lemcvt.analyses.heads.config import OUTPUT_DIR
    from lemcvt.analyses.heads.fig_quick_slope_area import main as quick_fig
    from lemcvt.analyses.heads.run import extract

    rec = RunRecorder("heads", f"{len(tiles)}tiles", run_id=run_id, root=results_root)
    t0 = time.perf_counter()
    out = rec.dir / OUTPUT_DIR
    info = extract(out, list(tiles), use_lsdtt=use_lsdtt)
    try:
        quick_fig(out)
    except Exception as exc:  # noqa: BLE001
        logger.warning("quick slope-area figure skipped: %s", exc)
    cs = pd.read_csv(out / "class_summary.csv")
    summary: dict[str, Any] = {"outputs": str(out), "lsdtt_used": use_lsdtt, "wiener": {t: v.get("wiener") for t, v in info["tiles"].items()},
                               "class_summary": cs.to_dict("records"), "A_crit_source": info.get("A_crit_source")}
    rec.add_params("summary", summary)
    rec.add_params("wall_s", time.perf_counter() - t0)
    rec.finalize({"tiles": _snapshot(list(tiles)), "lsdtt": use_lsdtt})
    return rec


# ----------------------------------------------------------------------------- regression check against the branches

def _branch_file(branch: str, path: str) -> str | None:
    try:
        return subprocess.check_output(["git", "show", f"{branch}:{path}"], cwd=REPO_ROOT, text=True)
    except subprocess.CalledProcessError:
        return None


def _latest(step: str) -> tuple[Path, dict[str, Any]] | None:
    for r in reversed(find_runs(step)):
        try:
            return r, json.loads((r / "params.json").read_text())
        except (OSError, ValueError):
            continue
    return None


def _points_compare(new: pd.DataFrame, old: pd.DataFrame, key_cols, value_cols, site: str) -> dict[str, Any]:
    n, o = new[new.site == site].copy(), old[old.site == site].copy()
    m = n.merge(o, on=key_cols, suffixes=("_new", "_old"))
    out: dict[str, Any] = {"n_new": int(len(n)), "n_old": int(len(o)), "n_matched_on_" + "_".join(key_cols): int(len(m))}
    for c in value_cols:
        if f"{c}_new" not in m or f"{c}_old" not in m:
            continue
        a, b = pd.to_numeric(m[f"{c}_new"], errors="coerce").to_numpy(float), pd.to_numeric(m[f"{c}_old"], errors="coerce").to_numpy(float)
        both = np.isfinite(a) & np.isfinite(b)
        same_nan = ~np.isfinite(a) & ~np.isfinite(b)
        d = np.abs(a[both] - b[both])
        out[c] = {"n_compared": int(both.sum()), "max_abs_diff": float(d.max()) if both.any() else None, "n_identical_or_both_nan": int((d <= 1e-9).sum() + same_nan.sum()),
                  "median_new": float(np.nanmedian(a)) if np.isfinite(a).any() else None, "median_old": float(np.nanmedian(b)) if np.isfinite(b).any() else None}
    return out


def run_port_check(run_id: str | None = None, results_root=None) -> RunRecorder:
    rec = RunRecorder("phase7_port_check", "all", run_id=run_id, root=results_root)
    res: dict[str, Any] = {}
    # width (Phase 3 full)
    for step, sub, keyc, cols in (("width", "outputs", ["tile", "row", "col"], ["W_primary", "W_inner_bankbreak", "W_valley_floor", "W_stage_0.5", "local_slope", "area_m2", "inner_relief_m"]),
                                  ("width_constrained", "outputs", ["tile", "row", "col"], ["W_trench", "W_bed", "trench_depth_m", "local_slope", "area_m2"])):
        lat = _latest(step)
        branch, bpath, files = BRANCH_FILES[step]
        entry: dict[str, Any] = {"branch": branch, "branch_path": bpath, "run": lat[0].name if lat else None}
        txt = _branch_file(branch, f"{bpath}/points.csv")
        if lat is None or txt is None or not (lat[0] / sub / "points.csv").exists():
            entry["status"] = "missing run or branch file"
            res[step] = entry
            continue
        (rec.dir / f"{step}_branch_points.csv").write_text(txt)
        old = pd.read_csv(rec.dir / f"{step}_branch_points.csv")
        new = pd.read_csv(lat[0] / sub / "points.csv")
        for site in ("GM", "AP"):
            entry[site] = _points_compare(new, old, keyc, cols, site)
            if "point_class" in new and "point_class" in old:
                entry[site]["class_counts_new"] = new[new.site == site].point_class.value_counts().to_dict()
                entry[site]["class_counts_old"] = old[old.site == site].point_class.value_counts().to_dict()
            if "channelized" in new and "channelized" in old:
                entry[site]["n_channelized_new"] = int(new[new.site == site].channelized.sum())
                entry[site]["n_channelized_old"] = int(old[old.site == site].channelized.sum())
        ftxt = _branch_file(branch, f"{bpath}/fits.json")
        if ftxt and (lat[0] / sub / "fits.json").exists():
            entry["fits_old"] = json.loads(ftxt)
            entry["fits_new"] = json.loads((lat[0] / sub / "fits.json").read_text())
        entry["status"] = "compared"
        res[step] = entry
    # heads
    lat = _latest("heads")
    branch, bpath, files = BRANCH_FILES["heads"]
    entry = {"branch": branch, "branch_path": bpath, "run": lat[0].name if lat else None}
    txt = _branch_file(branch, f"{bpath}/heads_extracted.csv")
    if lat is not None and txt is not None and (lat[0] / "head_extraction" / "heads_extracted.csv").exists():
        (rec.dir / "heads_branch_heads_extracted.csv").write_text(txt)
        old = pd.read_csv(rec.dir / "heads_branch_heads_extracted.csv", low_memory=False)
        new = pd.read_csv(lat[0] / "head_extraction" / "heads_extracted.csv", low_memory=False)
        per: dict[str, Any] = {}
        for key_st, n_ in new[(new.method == "P")].groupby(["site", "tile"]):
            kst = cast(tuple, key_st)
            site, t = str(kst[0]), int(kst[1])
            o_ = old[(old.method == "P") & (old.site == site) & (old.tile == t)]
            rec_t: dict[str, Any] = {}
            for sm in sorted(n_.smoothing.unique()):
                a = n_[(n_.smoothing == sm) & n_.retained]
                b = o_[(o_.smoothing == sm) & o_.retained]
                key = {(int(r), int(c)) for r, c in zip(b.row, b.col, strict=True)}
                same = sum((int(r), int(c)) in key for r, c in zip(a.row, a.col, strict=True))
                rec_t[sm] = {"n_retained_new": int(len(a)), "n_retained_old": int(len(b)), "n_same_pixel": int(same),
                             "n_hollow_new": int((a.head_class == "hollow").sum()), "n_hollow_old": int((b.head_class == "hollow").sum()),
                             "hollow_A_median_new": float(a[a.head_class == "hollow"].area_D8_m2.median()) if (a.head_class == "hollow").any() else None,
                             "hollow_A_median_old": float(b[b.head_class == "hollow"].area_D8_m2.median()) if (b.head_class == "hollow").any() else None,
                             "hollow_S_up_10_median_new": float(a[a.head_class == "hollow"].S_up_10.median()) if (a.head_class == "hollow").any() else None,
                             "hollow_S_up_10_median_old": float(b[b.head_class == "hollow"].S_up_10.median()) if (b.head_class == "hollow").any() else None}
            per[f"{site}{t}"] = rec_t
        entry["per_tile"] = per
        entry["status"] = "compared"
    else:
        entry["status"] = "missing run or branch file"
    res["heads"] = entry
    rec.add_params("checks", res)
    rec.finalize({"branches": {k: v[0] for k, v in BRANCH_FILES.items()}})
    return rec
