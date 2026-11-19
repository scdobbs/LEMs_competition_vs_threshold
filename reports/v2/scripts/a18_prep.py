#!/usr/bin/env python3
"""Addendum 18, Part A preparation (sections 1, 3.1, 5.6 inputs, 5.7 field data, decision 8 trunk list, decision 9 measured blocks).

Parts (``--part``, in this order; every part writes JSON under <run>/prep/):
  configs    decision 9: the `measured` blocks of the new tile configs from the Part B run (site m, k_s, C_ht at L_f and L_w, D0_A,
             D0_B, hilltop pixel count, provenance); `D0: null` with the reason below 500 hilltop pixels (decision 4); model.m = site m
  gm22       decision 1: the GM site summaries (site-pooled k_s and K0, medians of D0) on the 22-tile primary set and the 19-tile clean
             subset, beside the Part B 19-tile values
  geology    decision 1: the map's full description of QTcl, QTg, QTm, QTpc (every attribute field)
  trunk      decision 8: the largest A of every primary and continuity tile; the trunk list (max A > 5 km2); cells with A >= 1 km2
  ttv_mt     decision 5 / 5.6: k_s and K0 at theta = m_t by chi-z on every primary TTV tile
  samepixel  3.1: the same-pixel curvature diagnostic on the survey pairs
  heads      5.7.1: the field channel-head areas (M&D Table 1; the 77 digitised heads on the Marin 2019 full-grid routing)
  specs      the fit specifications of every (tile, group) for a18_fits.py (reads the parts above)
usage: a18_prep.py --part <name> [--n-boot 200]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a18_common import (  # noqa: E402
    AP1_3DEP,
    AP_PAMAP,
    CHANNEL_HEAD_SEARCH_M,
    CONTINUITY,
    D0_MIN_PIXELS,
    DIRECTIVE,
    E_M_PER_YR,
    FITTED_AT_SITE_M,
    GEOLOGY_FORMATION_LABELS,
    GEOLOGY_SHP,
    GM_CLEAN,
    GM_NCALM,
    GM_OFFMESA,
    GM_PRIMARY,
    L_F_M,
    MD_HEADS_GPKG,
    MD_TABLE1,
    NO_D0,
    OLD_CONFIG_TILES,
    PARTB,
    PARTB_REPORT,
    PRIMARY,
    REPO,
    SAME_PIXEL_RULES,
    SAME_PIXEL_TOL,
    SITE_M,
    SURVEY_PAIRS,
    TRUNK_A_M2,
    TRUNK_LIST_MIN_A_M2,
    TTV_FULL_AREA,
    TTV_M_T,
    TTV_M_T_EDGE,
    TTV_PRIMARY,
    V2_SITE_M,
    W_F,
    dump,
    fnum,
    partb_measurement,
    partb_per_tile,
    run_dir,
    site_of,
    slurm_record,
    spec_key,
)

PREP: str = ""


def out_path(name: str) -> str:
    return os.path.join(PREP, name)


def load_prep(name: str) -> dict:
    return json.load(open(out_path(name)))


# ============================================================================= configs (decision 9)

def measured_block(tile: str, meas: dict, site_m_rec: dict) -> dict:
    """The `measured` block of a new tile config from the Part B record of the tile."""
    site = site_of(tile)
    E = E_M_PER_YR[site]
    t = meas["tiles"][tile]
    k3, k4 = t["ks"]["1000"], t["ks"]["10000"]
    o = t["option_iii"]
    m_site = SITE_M[site]
    assert abs(float(t["m_site"]) - m_site) < 1e-12, (tile, t["m_site"], m_site)
    n_px = int(o["n_pixels"])
    has_d0 = n_px >= D0_MIN_PIXELS and o.get("C_ht_at_L_f") and o["C_ht_at_L_f"] > 0
    assert has_d0 == (tile not in NO_D0), (tile, n_px)
    blk: dict = {"m": m_site, "ks": float(k3["ks"]), "C_ht": float(o["C_ht_at_L_f"]) if has_d0 else None, "z_units": "m",
                 "ks_units": f"m^(2 m) with m = {m_site:.2f}, A0 = 1 m^2", "K0": float(E / k3["ks"]),
                 "D0": float(o["D0"]) if has_d0 else None, "ks_1e4_reported_not_used": float(k4["ks"]) if k4.get("ks") else None,
                 "C_ht_B": float(o["C_ht_at_L_w"]) if has_d0 and o.get("C_ht_at_L_w") else None, "L_w_m": o.get("L_w_m"),
                 "D0_B": float(o["D0_B"]) if has_d0 and o.get("D0_B") else None, "n_hilltop_px": n_px,
                 "E": E, "E_source": "addendum 17 decision 6 (addendum 18 decision 7)"}
    if not has_d0:
        blk["D0_reason"] = (f"addendum 18 decision 4: the option (iii) hilltop set at L_f = 11 m has {n_px} pixels (< {D0_MIN_PIXELS}); "
                            "no D0 at either end; the tile enters every K, MSE and A_crit statement and no D statement; P1, P2 and A2 are not run")
    blk["D0_bracket_note"] = ("addendum 18 decision 2: D0 is carried at both ends everywhere, D0_A = E/C_ht(L_f = 11 m) and D0_B = E/C_ht(L_w) on the "
                              "same option (iii) pixel set, L_w the largest tested window not exceeding the median crest width; every D statement is "
                              "made at both ends; the window-stability test (rule 6.1 of addendum 17) is reported, it no longer decides")
    wt = o["test"]
    prov = {"m": {"definition": "site constant: the minimum of the site exponent profile S(m) over the primary set at the site (addendum 17 Part B, 5.2; "
                                "addendum 18 decision 6)",
                  "site_m": m_site, "S_interval_1pct": site_m_rec.get("interval_0.01", {}).get("intervals"),
                  "S_interval_5pct": site_m_rec.get("interval_0.05", {}).get("intervals"), "S_min": site_m_rec.get("S_min"),
                  "n_tiles": site_m_rec.get("n_tiles"), "m_profile_run": meas["m_profile_runs"].get(tile),
                  "note": ("GM: argmin on the 19-tile Part B primary set; the 22-tile set of addendum 18 decision 1 gives the same 0.29 (1 % 0.26-0.32)" if site == "GM"
                           else "TTV: six of eleven per-tile profiles have their optimum at the grid edge m = 0.05 (addendum 18 decision 5)" if site == "TTV" else ""),
                  "part_b_run": os.path.basename(PARTB), "part_b_report": PARTB_REPORT},
            "ks": {"definition": "pooled through-origin chi-z slope at theta = m over the channel network above A_c = 1e3 m^2 (measurement_methods.md 0)",
                   "A_c": 1000.0, "n_basins": k3.get("n_basins"), "n_eff_centred": k3.get("n_eff_centred"), "n_eff_through_origin": k3.get("n_eff_through_origin"),
                   "n_eff_lt_3": k3.get("n_eff_lt_3"), "ci68": k3.get("ks_ci68"), "ci95": k3.get("ks_ci95"), "bootstrap": k3.get("bootstrap"),
                   "ks_1e4": {"ks": k4.get("ks"), "n_basins": k4.get("n_basins"), "n_eff_centred": k4.get("n_eff_centred"), "ci68": k4.get("ks_ci68")},
                   "measure_run": (t.get("measure_run") or {}).get("run"), "part_b_run": os.path.basename(PARTB), "part_b_job": "52881442"},
            "C_ht": {"definition": "curvature option (iii): median -laplacian at the fixed window L_f on the rollover-network hilltop set with the L_f/2 exclusion",
                     "L_f_m": float(o["L_f_m"]), "n_pixels": n_px, "n_segments": o.get("n_segments"), "A_c_rollover": (t.get("measure_run") or {}).get("A_c_rollover"),
                     "ci68": o.get("C_ht_ci68_at_L_f"), "ci95": o.get("C_ht_ci95_at_L_f"), "D0_ci68": o.get("D0_ci68"), "D0_ci95": o.get("D0_ci95"),
                     "window_test": {"median_crest_width_m": wt.get("median_crest_width_m"), "windows_tested_nominal_m": wt.get("windows_tested_nominal_m"),
                                     "n_windows_tested": wt.get("n_windows_tested"), "max_abs_deviation": wt.get("max_abs_deviation"), "pass": wt.get("pass"),
                                     "only_L_f_in_range": wt.get("only_L_f_in_range"), "non_trivial": bool((wt.get("n_windows_tested") or 0) > 1)},
                     "measure_run": (t.get("measure_run") or {}).get("run"), "part_b_run": os.path.basename(PARTB)},
            "C_ht_B": {"definition": "median -laplacian at L_w on the option (iii) pixel set", "L_w_m": o.get("L_w_m"), "ci68": o.get("C_ht_ci68_at_L_w"),
                       "ratio_to_C_ht_A": (float(o["C_ht_at_L_w"]) / float(o["C_ht_at_L_f"])) if has_d0 and o.get("C_ht_at_L_w") else None,
                       "part_b_run": os.path.basename(PARTB)}}
    blk["provenance"] = prov
    return blk


def part_configs(a) -> None:
    meas = partb_measurement()
    written = {}
    tiles = sorted(set(meas["tiles"]) - set(OLD_CONFIG_TILES))
    for tile in tiles:
        site = site_of(tile)
        path = os.path.join(REPO, "configs", "tiles", f"{tile}.yaml")
        d = yaml.safe_load(open(path))
        assert d["tile"] == tile
        assert not d.get("measured"), (tile, "measured block already present")
        sm = meas["sites"][site]["S_m"]["primary"]
        blk = measured_block(tile, meas, sm)
        d["measured"] = blk
        mdl = d["model"]
        mdl["m"] = SITE_M[site]
        if blk["D0"] is not None:
            mdl["D0"] = "U/C_ht"
            mdl["D0_B"] = "U/C_ht_B"
            mdl.pop("D0_placeholder_note", None)
        else:
            # the loader needs a numeric D0 inside the bounds; the fits of addendum 18 read measured.D0 (null) and run no D-fixed variant on the tile
            ref = d["measured_v1"]
            mdl["D0"] = float(E_M_PER_YR[site] / float(ref["hilltop_curvature"]))
            mdl["D0_placeholder_note"] = ("loader placeholder only: E / the site selection reference curvature of measured_v1; measured.D0 is null "
                                          "(addendum 18 decision 4) and no fit of addendum 18 uses this value")
            mdl.pop("D0_B", None)
        header = (f"# Tile {tile} ({d.get('registration', {}).get('version', '')}, {d.get('registration', {}).get('survey', '')}): grids registered by addendum 15/17; "
                  "`measured` written by reports/v2/scripts/a18_prep.py --part configs (addendum 18 decision 9) from the addendum 17 Part B run\n"
                  f"# {os.path.basename(PARTB)} (measure/measurement.json): the site m (S(m) argmin over the site's primary set; addendum 18 decision 6),\n"
                  "# k_s at the site m (pooled through-origin chi-z above A_c = 1e3 m^2; K0 = E/ks), C_ht by option (iii) at L_f = 11 m and at L_w (D0_A, D0_B;\n"
                  "# decision 2), the hilltop pixel count, and provenance. `measured_v1` holds the SITE SELECTION REFERENCE (not a v1 tile value), kept for the record.\n"
                  "# model.m is the site m; K0 = U/ks and D0 = U/C_ht resolve from `measured` (metres; the v2 metre bounds).\n")
        body = yaml.safe_dump(d, sort_keys=False, default_flow_style=False, width=140, allow_unicode=True)
        with open(path, "w") as f:
            f.write(header + body)
        written[tile] = {"m": blk["m"], "ks": blk["ks"], "K0": blk["K0"], "C_ht": blk["C_ht"], "D0": blk["D0"], "D0_B": blk["D0_B"], "n_hilltop_px": blk["n_hilltop_px"],
                         "D0_null": blk["D0"] is None, "model_m": mdl["m"]}
    # load check through the package loader
    from lemcvt.config import load_tile_config
    check = {}
    for tile in tiles:
        cfg = load_tile_config(tile)
        mp = cfg.model
        d0 = cfg.measured.get("D0")
        check[tile] = {"model_m": mp.m, "K0_partitioned": mp.K0_partitioned, "K0_ad": mp.K0_ad, "D0_model": mp.D0, "D0_B_model": mp.D0_B, "measured_D0": d0,
                       "K0_matches": bool(abs(mp.K0_partitioned / written[tile]["K0"] - 1) < 1e-12),
                       "D0_matches": bool(d0 is None or abs(mp.D0 / d0 - 1) < 1e-12),
                       "D0_B_matches": bool(cfg.measured.get("D0_B") is None or abs((mp.D0_B or np.nan) / cfg.measured["D0_B"] - 1) < 1e-12)}
    bad = [t for t, c in check.items() if not (c["K0_matches"] and c["D0_matches"] and c["D0_B_matches"])]
    rec = {"tiles_written": written, "n_written": len(written), "old_configs_kept": OLD_CONFIG_TILES, "no_D0": [t for t in tiles if written[t]["D0_null"]],
           "load_check": check, "load_check_failures": bad, "source": {"part_b_run": os.path.basename(PARTB), "measurement": "measure/measurement.json"},
           "directive": DIRECTIVE}
    dump(rec, out_path("configs.json"))
    print(f"configs: {len(written)} measured blocks written; no D0 on {rec['no_D0']}; load-check failures {bad}", flush=True)


# ============================================================================= gm22 (decision 1)

def _ks_site(tiles: list[str], m: float, E: float, n_boot: int, labels_cache: dict) -> dict:
    from lemcvt.measure.diagnostics import edge_outlet_labels
    from lemcvt.measure.phase6_inputs import _concat, _ks_from_moments, ks_at_theta
    from lemcvt.measure.pipeline import _metre_tile

    per = {}
    mom = {1e3: [], 1e4: []}
    for t in tiles:
        td = _metre_tile(t)
        Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
        labels = edge_outlet_labels(A, FD)
        per[t] = {}
        for a_c in (1e3, 1e4):
            r = ks_at_theta(Z, A, FD, dx, dy, m, a_c, labels, n_boot=0)
            mom[a_c].append(r.pop("_moments"))
            per[t][f"{a_c:g}"] = {"ks": r["ks"], "n_basins": r["n_basins"], "n_eff_centred": r.get("n_eff_centred"), "K0": (E / r["ks"]) if r.get("ks") else None}
    out = {"tiles": tiles, "n": len(tiles), "m": m, "per_tile": per}
    for a_c in (1e3, 1e4):
        parts = [p for p in mom[a_c] if p["n"].size]
        s = _ks_from_moments(_concat(parts), m, a_c, n_boot) if parts else None
        out[f"ks_site_pooled_{a_c:g}"] = s
        out[f"K0_site_pooled_{a_c:g}"] = (E / s["ks"]) if s and s.get("ks") else None
    return out


def part_gm22(a) -> None:
    meas = partb_measurement()
    per = partb_per_tile()
    E = E_M_PER_YR["GM"]
    m = SITE_M["GM"]
    out = {"m": m, "E": E, "groups": {}}
    for name, tiles in (("primary_22", GM_PRIMARY), ("clean_19", GM_CLEAN)):
        rec = _ks_site(tiles, m, E, a.n_boot, {})
        # check: the per-tile k_s recomputed here equals the Part B value
        rec["per_tile_check_max_rel"] = max(abs(rec["per_tile"][t]["1000"]["ks"] / float(per[t]["ks_1e3"]) - 1) for t in tiles)
        for col in ("K0", "D0_A", "D0_B", "C_ht_Lf", "ks_1e3"):
            v = np.array([fnum(per[t][col]) for t in tiles if fnum(per[t][col]) is not None], float)
            rec[f"{col}_median"] = float(np.median(v)) if v.size else None
            rec[f"{col}_min"] = float(v.min()) if v.size else None
            rec[f"{col}_max"] = float(v.max()) if v.size else None
            rec[f"{col}_n"] = int(v.size)
        rec["window_test"] = {"n_pass": int(sum(per[t]["window_pass"] == "True" for t in tiles)), "n": len(tiles),
                              "non_trivial": int(sum(int(per[t]["windows_tested"]) > 1 for t in tiles)),
                              "trivial_tiles": [t for t in tiles if int(per[t]["windows_tested"]) <= 1]}
        out["groups"][name] = rec
    pb = meas["sites"]["GM"]
    out["part_b"] = {g: {k: pb[g].get(k) for k in ("tiles", "n", "ks_site_pooled_1000", "K0_site_pooled_1000", "ks_site_pooled_10000", "K0_site_pooled_10000",
                                                     "K0_median", "D0_A_median", "D0_B_median", "C_ht_Lf_median", "window_test")} for g in ("primary", "clean", "ncalm")}
    out["S_m_formation_reading"] = meas["sites"]["GM"]["S_m"].get("formation") or None
    # the GM8/GM9/GM21 numbers
    out["returned_tiles"] = {t: {k: fnum(per[t][k]) for k in ("ks_1e3", "K0", "C_ht_Lf", "D0_A", "D0_B", "n_hilltop_px", "theta_pooled")} for t in ("GM8", "GM9", "GM21")}
    # the NCALM-twin D0 for GM1_fema-GM7_fema (decision 2, labelled sensitivity, no rule)
    out["ncalm_twin_D0"] = {f"GM{n}_fema": {"twin": f"GM{n}", "D0_A_twin": fnum(per[f"GM{n}"]["D0_A"]), "D0_B_twin": fnum(per[f"GM{n}"]["D0_B"]),
                                            "D0_A_fema": fnum(per[f"GM{n}_fema"]["D0_A"])} for n in range(1, 8)}
    dump(out, out_path("gm22.json"))
    print("gm22:", {g: (r["ks_site_pooled_1000"]["ks"], r["K0_site_pooled_1000"], r["D0_A_median"]) for g, r in out["groups"].items()}, flush=True)


# ============================================================================= geology (decision 1)

def part_geology(a) -> None:
    from osgeo import ogr

    ogr.UseExceptions()
    ds = ogr.Open(GEOLOGY_SHP)
    lyr = ds.GetLayer(0)
    defn = lyr.GetLayerDefn()
    fields = [defn.GetFieldDefn(k).GetName() for k in range(defn.GetFieldCount())]
    out = {"shapefile": GEOLOGY_SHP, "fields": fields, "labels": {}}
    counts: dict[str, int] = {}
    for f in lyr:
        rec = {n: f.GetField(n) for n in fields}
        lab = str(rec.get("GEN_LAB") or "")
        lab2 = str(rec.get("LABEL") or "")
        for key in {lab, lab2}:
            if key in GEOLOGY_FORMATION_LABELS:
                counts[key] = counts.get(key, 0) + 1
                d = out["labels"].setdefault(key, {"n_polygons": 0, "values": {}})
                d["n_polygons"] += 1
                for n in fields:
                    v = rec.get(n)
                    if isinstance(v, float) and n.lower() in ("shape_area", "shape_leng", "area", "perimeter", "length"):
                        continue
                    vs = d["values"].setdefault(n, {})
                    sv = str(v)
                    vs[sv] = vs.get(sv, 0) + 1
    # keep the distinct values (cap at 12 per field)
    for _key, d in out["labels"].items():
        d["values"] = {n: dict(sorted(vs.items(), key=lambda kv: -kv[1])[:12]) for n, vs in d["values"].items()}
    out["n_polygons_by_label"] = counts
    dump(out, out_path("geology.json"))
    print("geology:", counts, flush=True)


# ============================================================================= trunk (decision 8)

def part_trunk(a) -> None:
    from lemcvt.config import data_root, load_tile_config
    from lemcvt.io.grids import read_geotiff

    tiles = sorted(set(FITTED_AT_SITE_M))
    rows = {}
    for t in tiles:
        cfg = load_tile_config(t)
        A = read_geotiff(data_root() / cfg.grids["area"]).data
        ok = np.isfinite(A)
        amax = float(np.nanmax(A))
        n_trunk = int(np.sum(ok & (A >= TRUNK_A_M2)))
        rows[t] = {"site": cfg.site, "max_A_m2": amax, "max_A_km2": amax / 1e6, "n_cells_trunk": n_trunk, "frac_trunk": n_trunk / int(ok.sum()),
                   "n_valid": int(ok.sum()), "trunk_fit": bool(amax > TRUNK_LIST_MIN_A_M2),
                   "group": "primary" if t in PRIMARY.get(cfg.site, []) else "continuity" if t in CONTINUITY else "offmesa",
                   "area_grid": cfg.grids["area"]}
    lst = [t for t in tiles if rows[t]["trunk_fit"] and rows[t]["group"] in ("primary", "continuity")]
    out = {"rule": f"trunk cells: A >= {TRUNK_A_M2:g} m2 on the tile's area grid; trunk fit on every primary and continuity tile whose largest A exceeds {TRUNK_LIST_MIN_A_M2:g} m2",
           "tiles": rows, "trunk_list": lst, "expected": ["GM2_fema", "GM4_fema", "GM5_fema", "AP3_3dep", "AP4_3dep"],
           "offmesa_above_threshold_not_fitted": [t for t in tiles if rows[t]["trunk_fit"] and rows[t]["group"] == "offmesa"]}
    dump(out, out_path("trunk.json"))
    print("trunk list:", lst, {t: round(rows[t]["max_A_km2"], 2) for t in lst}, flush=True)


# ============================================================================= ttv_mt (decision 5)

def part_ttv_mt(a) -> None:
    from lemcvt.config import load_tile_config
    from lemcvt.measure.diagnostics import edge_outlet_labels
    from lemcvt.measure.phase6_inputs import ks_at_theta
    from lemcvt.measure.pipeline import _metre_tile

    E = E_M_PER_YR["TTV"]
    out = {}
    for t in TTV_PRIMARY:
        cfg = load_tile_config(t)
        td = _metre_tile(t)
        Z, A, FD, dx, dy = td.Z, td.A, td.FD, td.dx, td.filled.dy
        labels = edge_outlet_labels(A, FD)
        m_t = TTV_M_T[t]
        r = ks_at_theta(Z, A, FD, dx, dy, m_t, 1e3, labels, n_boot=a.n_boot)
        r.pop("_moments", None)
        chk = ks_at_theta(Z, A, FD, dx, dy, SITE_M["TTV"], 1e3, labels, n_boot=0)
        chk.pop("_moments", None)
        out[t] = {"m_t": m_t, "m_t_on_grid_edge": t in TTV_M_T_EDGE, "ks_m_t": r["ks"], "ks_ci95": r.get("ks_ci95"), "n_eff_centred": r.get("n_eff_centred"),
                  "n_basins": r["n_basins"], "K0_m_t": (E / r["ks"]) if r.get("ks") else None, "m_site": SITE_M["TTV"], "K0_site_m": cfg.model.K0_partitioned,
                  "ks_site_m_recomputed": chk["ks"], "ks_config": float(cfg.measured["ks"]), "check_rel": chk["ks"] / float(cfg.measured["ks"]) - 1}
        print(t, out[t]["m_t"], out[t]["ks_m_t"], out[t]["K0_m_t"], "check", out[t]["check_rel"], flush=True)
    dump({"E": E, "tiles": out, "source": "addendum 18 decision 5; m_t from Part B Appendix B5; k_s by chi-z at theta = m_t above 1e3 m2 (Part B estimator)"},
         out_path("ttv_mt.json"))


# ============================================================================= samepixel (3.1)

def _curv(Z, dx, dy):
    from lemcvt.measure.hilltop_curvature import total_curvature

    return -total_curvature(Z, W_F, dx, dy)


def _map_set(mask_src, td_src, td_dst, L_f):
    """Map the pixel centres of ``mask_src`` (grid of td_src) onto the lattice of td_dst by map coordinates; nearest cell."""
    from rasterio.warp import transform as rio_transform

    rows, cols = np.nonzero(mask_src)
    tr_s, tr_d = td_src.filled.transform, td_dst.filled.transform
    xs = tr_s.c + (cols + 0.5) * tr_s.a
    ys = tr_s.f + (rows + 0.5) * tr_s.e
    crs_s, crs_d = td_src.filled.crs, td_dst.filled.crs
    if crs_s is not None and crs_d is not None and crs_s != crs_d:
        xd, yd = rio_transform(crs_s, crs_d, xs.tolist(), ys.tolist())
        xd, yd = np.asarray(xd), np.asarray(yd)
        reproj = True
    else:
        xd, yd, reproj = xs, ys, False
    fc = (xd - tr_d.c) / tr_d.a
    fr = (yd - tr_d.f) / tr_d.e
    c_i = np.floor(fc).astype(int)
    r_i = np.floor(fr).astype(int)
    ny, nx = td_dst.Z.shape
    # offset between the source pixel centre and the destination cell centre
    off = np.hypot((fc - (c_i + 0.5)) * abs(tr_d.a), (fr - (r_i + 0.5)) * abs(tr_d.e))
    margin_m = 0.5 * L_f + 1.0
    marg_c = int(np.ceil(margin_m / abs(tr_d.a)))
    marg_r = int(np.ceil(margin_m / abs(tr_d.e)))
    inside = (r_i >= marg_r) & (r_i < ny - marg_r) & (c_i >= marg_c) & (c_i < nx - marg_c)
    keep = inside.copy()
    keep[inside] &= np.isfinite(td_dst.Z[r_i[inside], c_i[inside]])
    info = {"n_source": int(rows.size), "n_dropped_edge": int((~inside).sum()), "n_dropped_nodata": int((inside & ~keep).sum()), "n_mapped": int(keep.sum()),
            "offset_median_m": float(np.median(off[keep])) if keep.any() else None, "offset_max_m": float(np.max(off[keep])) if keep.any() else None,
            "reprojected": reproj, "crs_src": str(crs_s), "crs_dst": str(crs_d), "margin_m": margin_m}
    return r_i[keep], c_i[keep], info


def part_samepixel(a) -> None:
    from lemcvt.measure.phase6_inputs import option_iii_mask
    from lemcvt.measure.pipeline import _metre_tile

    meas = partb_measurement()
    per = partb_per_tile()
    out = {"pairs": {}, "sites": {}, "rule": SAME_PIXEL_RULES, "tolerance_vs_part_b": SAME_PIXEL_TOL, "L_f_m": L_F_M}
    stop = []
    for _site, pairs in SURVEY_PAIRS.items():
        for old, new in pairs:
            t0 = time.time()
            td_o, td_n = _metre_tile(old), _metre_tile(new)
            sets, curvs = {}, {}
            for name, td in (("old", td_o), ("new", td_n)):
                tile = old if name == "old" else new
                A_c = float(meas["tiles"][tile]["measure_run"]["A_c_rollover"])
                mask, _m0, _dist, _diag = option_iii_mask(td.Z, td.A, td.FD, td.dx, td.filled.dy, A_c, W_F)
                sets[name] = mask
                curvs[name] = _curv(td.Z, td.dx, td.filled.dy)
            rec: dict = {"old": old, "new": new, "n_old": int(sets["old"].sum()), "n_new": int(sets["new"].sum())}
            # (a) old surface on old pixels; (d) new surface on new pixels
            va = curvs["old"][sets["old"]]
            vd = curvs["new"][sets["new"]]
            rec["a"] = float(np.nanmedian(va)) if va.size else None
            rec["d"] = float(np.nanmedian(vd)) if vd.size else None
            # (b) new surface on the old pixels mapped onto the new lattice; (c) old surface on the new pixels mapped onto the old lattice
            r_on, c_on, info_on = _map_set(sets["old"], td_o, td_n, L_F_M)
            r_no, c_no, info_no = _map_set(sets["new"], td_n, td_o, L_F_M)
            vb = curvs["new"][r_on, c_on]
            vc = curvs["old"][r_no, c_no]
            rec["b"] = float(np.nanmedian(vb)) if vb.size else None
            rec["c"] = float(np.nanmedian(vc)) if vc.size else None
            rec["mapping_old_to_new"] = info_on
            rec["mapping_new_to_old"] = info_no
            pa, pd_ = fnum(per[old]["C_ht_Lf"]), fnum(per[new]["C_ht_Lf"])
            rec["part_b_a"], rec["part_b_d"] = pa, pd_
            rec["check_a_rel"] = abs(rec["a"] / pa - 1) if rec["a"] and pa else None
            rec["check_d_rel"] = abs(rec["d"] / pd_ - 1) if rec["d"] and pd_ else None
            rec["check_pass"] = bool((rec["check_a_rel"] is not None and rec["check_a_rel"] <= SAME_PIXEL_TOL) and (rec["check_d_rel"] is not None and rec["check_d_rel"] <= SAME_PIXEL_TOL))
            if not rec["check_pass"]:
                stop.append(f"{old}/{new}")
            ok = all(rec[k] for k in ("a", "b", "c", "d"))
            rec["surface_ratios"] = [rec["b"] / rec["a"], rec["d"] / rec["c"]] if ok else None
            rec["selection_ratios"] = [rec["c"] / rec["a"], rec["d"] / rec["b"]] if ok else None
            rec["total_ratio_d_over_a"] = rec["d"] / rec["a"] if ok else None
            rec["wall_s"] = time.time() - t0
            out["pairs"][f"{old}/{new}"] = rec
            print(f"{old}/{new}: a {rec['a']:.5g} b {rec['b']:.5g} c {rec['c']:.5g} d {rec['d']:.5g}; mapped {info_on['n_mapped']}/{info_on['n_source']} "
                  f"(offset med {info_on['offset_median_m']:.2f} max {info_on['offset_max_m']:.2f} m); check {rec['check_pass']} ({rec['wall_s']:.0f} s)", flush=True)

    def reading(S, P):
        r = SAME_PIXEL_RULES
        if S <= r["surface"]["S_max"] and r["surface"]["P_range"][0] <= P <= r["surface"]["P_range"][1]:
            return "surface"
        if r["selection"]["S_range"][0] <= S <= r["selection"]["S_range"][1] and P <= r["selection"]["P_max"]:
            return "selection"
        return "mixed"

    def site_rec(keys, label):
        sr = [x for k in keys for x in (out["pairs"][k]["surface_ratios"] or [])]
        pr = [x for k in keys for x in (out["pairs"][k]["selection_ratios"] or [])]
        tot = [out["pairs"][k]["total_ratio_d_over_a"] for k in keys if out["pairs"][k]["total_ratio_d_over_a"]]
        S, P = float(np.median(sr)), float(np.median(pr))
        return {"label": label, "pairs": keys, "S_surface": S, "P_selection": P, "S_values": sr, "P_values": pr, "total_ratio_median": float(np.median(tot)),
                "part_b_total_ratio": {"GM": 0.78, "AP": 0.65}[label.split()[0]], "reading": reading(S, P)}

    out["sites"]["GM"] = site_rec([f"{o}/{n}" for o, n in SURVEY_PAIRS["GM"]], "GM NCALM -> FEMA")
    ap_keys = [f"{o}/{n}" for o, n in SURVEY_PAIRS["AP"]]
    out["sites"]["AP"] = site_rec(ap_keys, "AP PAMAP -> 3DEP")
    out["sites"]["AP_without_AP2"] = site_rec([k for k in ap_keys if not k.startswith("AP2/")], "AP PAMAP -> 3DEP without AP2")
    out["check_failures"] = stop
    out["section_stopped"] = bool(stop)
    dump(out, out_path("samepixel.json"))
    print("samepixel:", {k: (v["S_surface"], v["P_selection"], v["reading"]) for k, v in out["sites"].items()}, "stop:", stop, flush=True)


# ============================================================================= heads (5.7.1)

def part_heads(a) -> None:
    import csv

    import rasterio
    from osgeo import ogr
    from rasterio.windows import Window

    ogr.UseExceptions()
    t1 = [float(r["source_area_m2"]) for r in csv.DictReader(open(MD_TABLE1))]
    pcts = (5, 25, 50, 75, 95)

    def dist(v):
        v = np.asarray(v, float)
        return {"n": int(v.size), **{f"p{p}": float(np.percentile(v, p)) for p in pcts}, "iqr": [float(np.percentile(v, 25)), float(np.percentile(v, 75))],
                "min": float(v.min()), "max": float(v.max())}

    ds = ogr.Open(MD_HEADS_GPKG)
    lyr = ds.GetLayerByName("channel_heads")
    heads = []
    for f in lyr:
        g = f.GetGeometryRef()
        heads.append({"head_id": f.GetField("head_id"), "label": f.GetField("label"), "continuation": f.GetField("continuation"), "component": f.GetField("component"),
                      "x": g.GetX(), "y": g.GetY()})
    rows = []
    with rasterio.open(TTV_FULL_AREA) as d:
        tr = d.transform
        assert abs(tr.a - 1.0) < 1e-9 and abs(tr.e + 1.0) < 1e-9, tr
        radii = [CHANNEL_HEAD_SEARCH_M, 10.0, 20.0]        # the rule's 3 m; 10 m (about the georeference RMS of 9.5 m) and 20 m as a labelled sensitivity
        R = int(np.ceil(max(radii))) + 1
        for h in heads:
            c0 = int(np.floor((h["x"] - tr.c) / tr.a))
            r0 = int(np.floor((h["y"] - tr.f) / tr.e))
            win = Window(c0 - R, r0 - R, 2 * R + 1, 2 * R + 1)
            blk = d.read(1, window=win).astype(float)
            rr, cc = np.mgrid[r0 - R:r0 + R + 1, c0 - R:c0 + R + 1]
            xc = tr.c + (cc + 0.5) * tr.a
            yc = tr.f + (rr + 0.5) * tr.e
            dd = np.hypot(xc - h["x"], yc - h["y"])
            rec = {**h, "row": r0, "col": c0, "A_at_head_m2": float(blk[R, R]), "chosen": f"max within {CHANNEL_HEAD_SEARCH_M:g} m"}
            for rad in radii:
                inside = dd <= rad
                vals = np.where(inside, blk, -np.inf)
                k = int(np.argmax(vals))
                rec[f"A_max_within_{rad:g}m_m2"] = float(vals.ravel()[k])
                rec[f"offset_of_max_{rad:g}m_m"] = float(dd.ravel()[k])
                rec[f"n_cells_within_{rad:g}m"] = int(inside.sum())
            rows.append(rec)
    a77 = [r["A_max_within_3m_m2"] for r in rows]
    a77_at = [r["A_at_head_m2"] for r in rows]
    out = {"table1": {"source": MD_TABLE1, "areas_m2": t1, "dist": dist(t1)},
           "digitised": {"source": MD_HEADS_GPKG, "routing": TTV_FULL_AREA, "search_radius_m": CHANNEL_HEAD_SEARCH_M, "heads": rows,
                         "dist_max_within_3m": dist(a77), "dist_at_head_cell": dist(a77_at),
                         "sensitivity_radius": {f"{rad:g}m": dist([r[f"A_max_within_{rad:g}m_m2"] for r in rows]) for rad in [3.0, 10.0, 20.0]},
                         "choice": f"the largest A within {CHANNEL_HEAD_SEARCH_M:g} m of the digitised head (georeference RMS 9.5 m at grid scale); the value at the head cell "
                                   "and the largest A within 10 m and 20 m are reported beside it (sensitivity, not used by any rule)"},
           "comparison": {"median_77": float(np.median(a77)), "median_63": float(np.median(t1)), "ratio_77_over_63": float(np.median(a77) / np.median(t1)),
                          "median_77_at_cell": float(np.median(a77_at)),
                          "median_77_sensitivity": {f"{rad:g}m": float(np.median([r[f"A_max_within_{rad:g}m_m2"] for r in rows])) for rad in [3.0, 10.0, 20.0]}}}
    dump(out, out_path("heads.json"))
    print("heads:", out["table1"]["dist"], out["digitised"]["dist_max_within_3m"], out["comparison"], flush=True)


# ============================================================================= specs

def part_specs(a) -> None:
    from lemcvt.config import load_tile_config

    per = partb_per_tile()
    trunk = load_prep("trunk.json")
    ttv = load_prep("ttv_mt.json")["tiles"]
    specs: dict[str, dict] = {}

    def cfg_vals(tile):
        cfg = load_tile_config(tile)
        me = cfg.measured
        return {"m": float(me["m"]), "K0": float(cfg.model.K0_partitioned), "D0A": fnum(me.get("D0")), "D0B": fnum(me.get("D0_B")), "ks": float(me["ks"]),
                "n_hilltop_px": me.get("n_hilltop_px", (me.get("provenance", {}).get("C_ht", {}) or {}).get("n_pixels"))}

    def add(tile, group, m, K0, D0A, D0B, union=False, trunk_fit=False, note=""):
        site = site_of(tile)
        specs[spec_key(tile, group)] = {"tile": tile, "group": group, "site": site, "m": m, "E": E_M_PER_YR[site], "K0": K0, "D0A": D0A, "D0B": D0B,
                                        "has_D0": D0A is not None, "union_phase6": union, "trunk": trunk_fit, "note": note}

    for site, tiles in PRIMARY.items():
        for t in tiles:
            v = cfg_vals(t)
            assert abs(v["m"] - SITE_M[site]) < 1e-12
            add(t, "primary", v["m"], v["K0"], v["D0A"], v["D0B"], note="config measured block (Part B at the site m)")
    for t in GM_OFFMESA:
        v = cfg_vals(t)
        add(t, "offmesa", v["m"], v["K0"], v["D0A"], v["D0B"], note="config measured block (Part B at the site m)")
    for t in GM_NCALM:
        if t in OLD_CONFIG_TILES:
            # the config carries the v2.0.0 block (m = 0.31); the continuity fit at the site m 0.29 takes k_s and D0 from the Part B NCALM record
            p = per[t]
            add(t, "ncalm", SITE_M["GM"], E_M_PER_YR["GM"] / float(p["ks_1e3"]), fnum(p["D0_A"]), fnum(p["D0_B"]),
                note="Part B NCALM record at m = 0.29 (per_tile.csv); the config keeps the v2.0.0 block at m = 0.31")
            v = cfg_vals(t)
            add(t, "v2m", v["m"], v["K0"], v["D0A"], v["D0B"], union=True, note="the v2.0.0 config (m = 0.31): 4.1 gate (GM1) and 5.5 (GM2-GM5)")
        else:
            v = cfg_vals(t)
            add(t, "ncalm", v["m"], v["K0"], v["D0A"], v["D0B"], note="config measured block (Part B at the site m)")
    for t in AP_PAMAP:
        v = cfg_vals(t)
        assert abs(v["m"] - V2_SITE_M["AP"]) < 1e-12
        add(t, "pamap", v["m"], v["K0"], v["D0A"], v["D0B"], union=True, note="the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate)")
    for t in AP1_3DEP:
        v = cfg_vals(t)
        add(t, "ap1_3dep", v["m"], v["K0"], v["D0A"], v["D0B"], note="Tier 1, excluded (road), continuity only")
    for t in TTV_PRIMARY:
        v = cfg_vals(t)
        add(t, "mt", ttv[t]["m_t"], ttv[t]["K0_m_t"], v["D0A"], v["D0B"], note=f"decision 5: m_t = {ttv[t]['m_t']:.2f}" + (" (grid edge, flagged)" if t in TTV_M_T_EDGE else ""))
    for t in trunk["trunk_list"]:
        base = specs.get(spec_key(t, "primary")) or specs.get(spec_key(t, "ncalm")) or specs.get(spec_key(t, "pamap")) or specs.get(spec_key(t, "ap1_3dep"))
        assert base is not None, t
        add(t, "trunk", base["m"], base["K0"], base["D0A"], base["D0B"], trunk_fit=True, note=f"decision 8: max A {trunk['tiles'][t]['max_A_km2']:.2f} km2; base group {base['group']}")
    out = {"specs": specs, "n": len(specs), "by_group": {}, "gate": [spec_key("GM1", "v2m"), spec_key("AP2", "pamap")]}
    for k, s in specs.items():
        out["by_group"].setdefault(s["group"], []).append(k)
    dump(out, out_path("fit_specs.json"))
    print("specs:", {g: len(v) for g, v in out["by_group"].items()}, "total", len(specs), flush=True)


def part_sm22(a) -> None:
    """Decision 1: S(m) over the 22-tile GM primary set and the 19-tile clean subset (the Part B m-profile runs; nothing rerun)."""
    from lemcvt.analyses.exponent_site import latest_m_profile_runs, site_profile

    tiles = tuple(set(GM_PRIMARY) | set(GM_CLEAN))
    runs = latest_m_profile_runs(tiles=tiles, m_min_required=0.05)
    out = {}
    for name, tl in (("primary_22", GM_PRIMARY), ("clean_19", GM_CLEAN)):
        prof = site_profile(runs, tl)
        prof.pop("per_tile_normalised", None)
        prof["runs"] = {t: runs[t][0].name for t in tl}
        prof["leave_one_out"] = {t: site_profile(runs, [u for u in tl if u != t])["m_opt"] for t in tl}
        out[name] = prof
        print(name, prof["m_opt"], prof["interval_0.01"]["intervals"], prof["interval_0.05"]["intervals"], flush=True)
    out["missing_runs"] = [t for t in tiles if t not in runs]
    dump(out, out_path("sm22.json"))


PARTS = {"configs": part_configs, "gm22": part_gm22, "geology": part_geology, "trunk": part_trunk, "ttv_mt": part_ttv_mt, "samepixel": part_samepixel,
         "heads": part_heads, "specs": part_specs, "sm22": part_sm22}


def main():
    global PREP
    p = argparse.ArgumentParser()
    p.add_argument("--part", required=True, choices=list(PARTS))
    p.add_argument("--n-boot", type=int, default=200)
    a = p.parse_args()
    rd = run_dir()
    PREP = os.path.join(rd, "prep")
    os.makedirs(PREP, exist_ok=True)
    slurm_record(PREP)
    t0 = time.time()
    PARTS[a.part](a)
    print(f"{a.part} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
