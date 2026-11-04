"""Driver.

    python -m head_extraction.run --phase extract [--no-lsdtt] [--tiles AP1,GM3]
    python -m head_extraction.run --phase analyse        # after heads_review.csv exists (not yet implemented)

Extract phase (directive sections 2-4): Method P at three curvature
thresholds, Method D where the LSDTT binary exists, drops, measurements,
classes, method agreement, QC maps and profiles. Then STOP for review.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from acrit_slope.heads import curvature_grid
from width_analysis.network import FlowGraph
from width_analysis.tiles import REPO_ROOT, load_tile

from . import classify as cl
from . import figures, lsdtt, pelletier
from .config import A_CRIT_M2, A_CRIT_STAGE2_M2, ANALYSIS, EXTRACTION, LSDTT, OUTPUT_DIR, QC, THETA, TILE_KEYS

HEAD_COLUMNS = ["head_id", "site", "tile", "method", "smoothing", "kt", "tcp", "row", "col", "x", "y", "crs", "area_D8_m2", "retained", "drop_reason",
                "head_class", "class_detail", "S_up_5", "S_up_10", "S_up_20", "S_centred_5", "S_centred_10", "S_centred_20", "S_pixel",
                "S_down_10", "plan_curvature_10m", "convergent_10m", "dist_to_trunk_m", "flow_azimuth_deg", "edge_distance_m",
                "upslope_path_m", "downslope_path_m", "z_m", "flag_tile_edge", "flag_path_leaves_tile", "flag_short_upslope_path"]


def _git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _json_safe(o):
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, np.ndarray):
        return _json_safe(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    return o


def match_sets(a: pd.DataFrame, b: pd.DataFrame, radius: float) -> dict:
    """Nearest-neighbour matching of two head sets within ``radius`` (both directions)."""
    out = dict(n_a=len(a), n_b=len(b), radius_m=radius)
    if len(a) == 0 or len(b) == 0:
        out.update(frac_a_matched=np.nan, frac_b_matched=np.nan, median_separation_m=np.nan)
        return out
    ta, tb = cKDTree(a[["x", "y"]].to_numpy()), cKDTree(b[["x", "y"]].to_numpy())
    dab, _ = tb.query(a[["x", "y"]].to_numpy())
    dba, _ = ta.query(b[["x", "y"]].to_numpy())
    out.update(frac_a_matched=float(np.mean(dab <= radius)), frac_b_matched=float(np.mean(dba <= radius)),
               median_separation_m=float(np.median(dab[dab <= radius])) if (dab <= radius).any() else np.nan)
    return out


# ---------------------------------------------------------------------------
# extract phase
# ---------------------------------------------------------------------------

def extract_tile(site, t, out_dir: Path, rng, use_lsdtt: bool, partition: pd.DataFrame | None, deadline) -> dict:
    t0 = time.time()
    tile = load_tile(site, t)
    graph = FlowGraph(tile, QC["trunk_area_m2"], 0.0)       # a_min only labels channel pixels; up/down links do not depend on it
    curv = curvature_grid(tile, EXTRACTION["curvature_block_m"])
    kt0 = EXTRACTION["curvature_threshold_default"]
    frames, sens, info = [], [], dict(site=site, tile=t)

    # Method P at every threshold; full measurement at the default only
    for kt in EXTRACTION["curvature_thresholds"]:
        res = pelletier.extract(tile.z, tile.pixel_size, kt, EXTRACTION)
        r, c = res["heads_rc"]
        df = cl.measure(tile, graph, r, c, EXTRACTION, QC, curv)
        rec = dict(site=site, tile=t, kt=kt, n_heads=len(df), n_retained=int(df.retained.sum()),
                   n_component_top=len(res["component_heads_rc"][0]), n_valley_pixels=res["n_valley_pixels"],
                   wiener_rms_change_m=res["wiener"]["rms_change_m"], wiener_beta=res["wiener"]["beta"],
                   wiener_N2=res["wiener"]["N2"], wiener_signal_max_nu=res["wiener"]["signal_max_nu"])
        sens.append(rec)
        rec["smoothing"] = "OWF"
        if kt == kt0:
            df["method"], df["kt"], df["tcp"], df["smoothing"] = "P", kt, np.nan, "OWF"
            frames.append(df)
            info["wiener"] = res["wiener"]
            # directive-wording alternative, retained counts only
            r2, c2 = res["component_heads_rc"]
            df2 = cl.measure(tile, graph, r2, c2, EXTRACTION, QC, curv)
            info["component_top_heads"] = dict(n=len(df2), n_retained=int(df2.retained.sum()))
            m = match_sets(df[df.retained], df2[df2.retained], EXTRACTION["match_radius_m"])
            info["pelletier_vs_component_top"] = m
        print(f"[{site}{t}] P kt={kt:g}: {len(df)} heads, {int(df.retained.sum())} retained ({time.time() - t0:.0f} s)", flush=True)

    # beyond the directive: extra smoothing after the Wiener filter, default threshold
    for DT in EXTRACTION["smoothing_sensitivity_DT_m2"]:
        res = pelletier.extract(tile.z, tile.pixel_size, kt0, EXTRACTION, diffusion_DT=DT)
        r, c = res["heads_rc"]
        df = cl.measure(tile, graph, r, c, EXTRACTION, QC, curv)
        sens.append(dict(site=site, tile=t, kt=kt0, smoothing=f"OWF+DT{DT:g}", n_heads=len(df), n_retained=int(df.retained.sum()),
                         n_component_top=len(res["component_heads_rc"][0]), n_valley_pixels=res["n_valley_pixels"],
                         wiener_rms_change_m=res["wiener"]["rms_change_m"], wiener_beta=res["wiener"]["beta"],
                         wiener_N2=res["wiener"]["N2"], wiener_signal_max_nu=res["wiener"]["signal_max_nu"]))
        df["method"], df["kt"], df["tcp"], df["smoothing"] = "P", kt0, np.nan, f"OWF+DT{DT:g}"
        frames.append(df)
        print(f"[{site}{t}] P kt={kt0:g} OWF+DT{DT:g}: {len(df)} heads, {int(df.retained.sum())} retained", flush=True)

    # Method D (and LSDTT Wiener)
    info["lsdtt"] = dict(skipped=True)
    if use_lsdtt:
        tcps = [LSDTT["threshold_contributing_pixels"]] + list(LSDTT["threshold_contributing_pixels_sensitivity"])
        res = lsdtt.run_tile(tile, THETA[(site, t)], LSDTT, Path(out_dir) / "lsdtt_work", tcps, deadline)
        info["lsdtt"] = dict(skipped=res["exe"] is None, exe=res["exe"], log=res["log"], elapsed_s=res["elapsed_s"],
                             n_by_method_tcp={f"{m}_tcp{tc}": int(n) for (m, tc), n in res["heads"].groupby(["method", "tcp"]).size().items()} if len(res["heads"]) else {})
        for (m, tc), sub in res["heads"].groupby(["method", "tcp"]):
            df = cl.measure(tile, graph, sub.row.to_numpy(), sub.col.to_numpy(), EXTRACTION, QC, curv)
            df["method"], df["kt"], df["tcp"], df["smoothing"] = m, np.nan, tc, "LSDTT"
            frames.append(df)
            print(f"[{site}{t}] {m} tcp={tc}: {len(df)} heads, {int(df.retained.sum())} retained", flush=True)

    heads = pd.concat(frames, ignore_index=True)
    heads["site"], heads["tile"], heads["crs"] = site, t, tile.crs
    # classify within each head set: the rill and road rules count neighbouring heads of the same set only
    heads = pd.concat([cl.classify(sub, QC, f"S_up_{EXTRACTION['L_default_m']:g}")
                       for _, sub in heads.groupby(["method", "smoothing", heads.tcp.fillna(-1)], sort=False)], ignore_index=True)
    heads["head_id"] = [f"{site}{t}_{m}_{int(r)}_{int(c)}" + (f"_tcp{int(tc)}" if np.isfinite(tc) else "") + ("" if sm == "OWF" or sm == "LSDTT" else f"_{sm.replace('+', '')}")
                        for m, r, c, tc, sm in zip(heads.method, heads.row, heads.col, heads.tcp, heads.smoothing)]

    # method agreement at the default settings
    P = heads[(heads.method == "P") & (heads.smoothing == "OWF") & heads.retained]
    agree = {}
    for m in ("D", "W"):
        for tc, sub in heads[(heads.method == m) & heads.retained].groupby("tcp"):
            agree[f"P_vs_{m}_tcp{int(tc)}"] = match_sets(P, sub, EXTRACTION["match_radius_m"])
    for sm, sub in heads[(heads.method == "P") & (heads.smoothing != "OWF") & heads.retained].groupby("smoothing"):
        agree[f"P_OWF_vs_P_{sm}"] = match_sets(P, sub, EXTRACTION["match_radius_m"])
        for m in ("D", "W"):
            for tc, subD in heads[(heads.method == m) & heads.retained].groupby("tcp"):
                agree[f"P_{sm}_vs_{m}_tcp{int(tc)}"] = match_sets(sub, subD, EXTRACTION["match_radius_m"])
    info["method_agreement"] = agree

    # figures
    fig_dir = out_dir / "figures"
    (fig_dir / "qc_maps").mkdir(parents=True, exist_ok=True)
    (fig_dir / "qc_profiles").mkdir(parents=True, exist_ok=True)
    is_default_D = (heads.tcp == LSDTT["threshold_contributing_pixels"])
    default_only = heads[((heads.method == "P") & (heads.smoothing == "OWF")) | is_default_D]
    part = partition[(partition.site == site) & (partition.tile == t) & partition.kept] if partition is not None else None
    figures.fig_qc_map(tile, default_only, part, QC, EXTRACTION, fig_dir / "qc_maps" / f"{site}{t}.png")
    info["n_profiles"] = figures.fig_profiles(tile, graph, heads[heads.smoothing == "OWF"], QC, EXTRACTION, rng, fig_dir / "qc_profiles" / f"{site}{t}.png")
    smk = f"OWF+DT{EXTRACTION['smoothing_map_DT_m2']:g}"
    (fig_dir / f"qc_maps_{smk.replace('+', '_')}").mkdir(parents=True, exist_ok=True)
    (fig_dir / f"qc_profiles_{smk.replace('+', '_')}").mkdir(parents=True, exist_ok=True)
    sm_only = heads[((heads.method == "P") & (heads.smoothing == smk)) | is_default_D]
    figures.fig_qc_map(tile, sm_only, part, QC, EXTRACTION, fig_dir / f"qc_maps_{smk.replace('+', '_')}" / f"{site}{t}.png")
    figures.fig_profiles(tile, graph, heads[heads.smoothing == smk], QC, EXTRACTION, rng, fig_dir / f"qc_profiles_{smk.replace('+', '_')}" / f"{site}{t}.png")
    info["elapsed_s"] = round(time.time() - t0, 1)
    print(f"[{site}{t}] done in {info['elapsed_s']} s", flush=True)
    return dict(heads=heads, sensitivity=pd.DataFrame(sens), info=info)


def class_summary(heads: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (site, t, m, sm, tc), sub in heads.groupby(["site", "tile", "method", "smoothing", heads.tcp.fillna(-1)]):
        rec = dict(site=site, tile=t, method=m, smoothing=sm, tcp=(None if tc < 0 else int(tc)), n_extracted=len(sub),
                   n_dropped_tile_edge=int((sub.drop_reason == "tile_edge").sum()),
                   n_dropped_path_leaves_tile=int((sub.drop_reason == "path_leaves_tile").sum()),
                   n_retained=int(sub.retained.sum()),
                   n_retained_short_upslope_path=int((sub.retained & sub.flag_short_upslope_path).sum()),
                   n_retained_S_up_10_defined=int((sub.retained & np.isfinite(sub.S_up_10)).sum()))
        for c in cl.CLASSES:
            rec[f"n_{c}"] = int(((sub.head_class == c) & sub.retained).sum())
        hol = sub[sub.retained & (sub.head_class == "hollow")]
        rec["hollow_A_median_m2"] = hol.area_D8_m2.median() if len(hol) else np.nan
        rec["hollow_S_up_10_median"] = hol.S_up_10.median() if len(hol) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["extract", "analyse"], required=True)
    p.add_argument("--no-lsdtt", action="store_true")
    p.add_argument("--tiles", default=None, help="comma list like AP1,GM3 (default all ten)")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    out_dir = Path(a.out) if a.out else REPO_ROOT / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.phase == "analyse":
        raise SystemExit("analyse phase is implemented after the checkpoint review (needs heads_review.csv)")

    keys = TILE_KEYS if not a.tiles else [(s[:2], int(s[2:])) for s in a.tiles.split(",")]
    rng = np.random.default_rng(ANALYSIS["random_seed"])
    t_start = time.time()
    deadline = t_start + LSDTT["time_cap_hours"] * 3600 if not a.no_lsdtt else None
    part_path = REPO_ROOT / "outputs" / "acrit_slope" / "heads.csv"
    partition = pd.read_csv(part_path) if part_path.exists() else None
    info = dict(git=_git_hash(), python=sys.version, platform=platform.platform(), started=time.strftime("%Y-%m-%d %H:%M:%S"),
                config=dict(EXTRACTION=EXTRACTION, LSDTT=LSDTT, QC=QC, ANALYSIS=ANALYSIS),
                A_crit_final={f"{k[0]}{k[1]}": v for k, v in A_CRIT_M2.items()}, A_crit_stage2={f"{k[0]}{k[1]}": v for k, v in A_CRIT_STAGE2_M2.items()},
                theta={f"{k[0]}{k[1]}": v for k, v in THETA.items()}, lsdtt_binary=lsdtt.binary_available(LSDTT),
                unfilled_dem_available=False, roads_layer_available=False, tiles={})
    heads, sens = [], []
    for site, t in keys:
        res = extract_tile(site, t, out_dir, rng, not a.no_lsdtt, partition, deadline)
        heads.append(res["heads"]); sens.append(res["sensitivity"]); info["tiles"][f"{site}{t}"] = res["info"]
        pd.concat(heads)[[c for c in HEAD_COLUMNS if c in res["heads"].columns]].to_csv(out_dir / "heads_extracted.csv", index=False)
    H = pd.concat(heads, ignore_index=True)
    cols = [c for c in HEAD_COLUMNS if c in H.columns] + [c for c in H.columns if c not in HEAD_COLUMNS]
    H = H[cols]
    H.to_csv(out_dir / "heads_extracted.csv", index=False)
    H[~H.retained].to_csv(out_dir / "dropped_heads.csv", index=False)
    S = pd.concat(sens, ignore_index=True)
    S.to_csv(out_dir / "threshold_sensitivity.csv", index=False)
    cs = class_summary(H)
    cs.to_csv(out_dir / "class_summary.csv", index=False)
    print(cs.to_string(), flush=True)
    figures.fig_threshold_sensitivity(S, out_dir / "figures" / "fig_threshold_sensitivity.png")
    # review template: every retained Method P head, keep by default
    rev = H[H.retained & (H.method == "P")][["head_id", "site", "tile", "smoothing", "head_class", "area_D8_m2", "S_up_10"]].copy()
    rev["keep"] = True
    rev["reason"] = ""
    rev.to_csv(out_dir / "heads_review_template.csv", index=False)
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info["elapsed_s"] = round(time.time() - t_start, 1)
    with open(out_dir / "run_info.json", "w") as fh:
        json.dump(_json_safe(info), fh, indent=1)
    print(f"wrote {out_dir} in {info['elapsed_s']} s", flush=True)


if __name__ == "__main__":
    main()
