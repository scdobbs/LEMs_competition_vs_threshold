"""Phase 6.0 measured inputs (addendum 5, sections 1.2, 2.2 and 2.3).

``lemcvt phase6 inputs --tiles all`` computes, from the metre grids and nothing else, everything that
section 2.3 writes into the tile configs, with the provenance the configs must carry:

* **the site exponent m** (section 1.1): read from the latest ``m_profile`` runs through
  :func:`lemcvt.analyses.exponent_site.site_summary` (the S(m) minima with their 1 % and 5 % intervals and
  the run ids); nothing is rerun;
* **k_s at theta = m_site** (section 1.2): the pooled through-origin chi-z slope over the tile's channel
  network above A_c = 10^3 m^2 (and 10^4 m^2, reported, not used), with the pre-registered basin filters
  of ``lemcvt.measure.diagnostics``, n_eff (centred and through-origin), a basin bootstrap of k_s at that
  fixed theta, and the site-pooled k_s. ``K0 = U/k_s`` is the models' K0;
* **the option (iii) test** (section 2.2, pre-registered): on the option (iii) pixel set (the ``measure``
  hilltop set of the rollover network with pixels closer than L_f/2 to that network dropped) the median
  C_ht(L) for L = 7, 9, 11, 15, 21, 25, 31, 41 and 51 m nominal (half-widths 3, 4, 5, 7, 10, 12, 15, 20,
  25 cells) with segment-bootstrap 68 % bands. Pass rule: for every window from L_f = 11 m up to the set's
  own median crest width (2 x the distance to the rollover network), C_ht stays within +-10 % of C_ht(L_f);
  option (iii) stands when at least 4 of 5 tiles pass at each site;
* **C_ht and D0 by option (iii)** at L_f (the audit's numbers, recomputed here so that one run carries the
  provenance) and, for the record, C_ht on the ``02b`` set (the measure set without exclusion at the
  tile's plateau window L*).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, label

from lemcvt.config import config_snapshot, load_tile_config
from lemcvt.measure.audits import _n_eff, basin_moments
from lemcvt.measure.chi import channel_network, channel_pixel_basin_lookup
from lemcvt.measure.diagnostics import N_BOOT, _metre_tile, edge_outlet_labels
from lemcvt.measure.hilltop_curvature import _weighted_median, hilltop_mask, total_curvature
from lemcvt.results import RunRecorder, find_runs

logger = logging.getLogger(__name__)

W_F = 5                                               # L_f = (2 w_f + 1) dx = 11 cells (checkpoint 4b/5, 6.1)
TEST_HALF_WIDTHS = (3, 4, 5, 7, 10, 12, 15, 20, 25)   # L = 7, 9, 11, 15, 21, 25, 31, 41, 51 m nominal
PASS_TOL = 0.10
PASS_MIN_TILES = 4
KS_THRESHOLDS = (1e3, 1e4)
KS_USED_THRESHOLD = 1e3


# ----------------------------------------------------------------------------- the option (iii) pixel set

def option_iii_mask(Z, A, FD, dx, dy, A_c_rollover: float, w_f: int = W_F) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """The ``measure`` hilltop set of the rollover network (divides between outlet basins, two cells from the edge,
    |grad z| <= 0.4) with pixels closer than L_f/2 to a pixel of that network dropped. Returns
    ``(mask, mask_before_exclusion, distance_to_network, diagnostics)``."""
    net = channel_network(Z, A, FD, A_c_rollover, dx, dy)
    mask0, diag = hilltop_mask(Z, A, FD, A_c_rollover, dx, dy, basin_of_channel=channel_pixel_basin_lookup(net))
    channel = np.isfinite(A) & (A >= A_c_rollover)
    dist = distance_transform_edt(~channel, sampling=(dy, dx))
    L_f = (2 * w_f + 1) * dx
    mask = mask0 & (dist >= 0.5 * L_f)
    diag = dict(diag) | {"A_c_rollover": float(A_c_rollover), "L_f_m": float(L_f), "exclusion_m": float(0.5 * L_f),
                         "n_before_exclusion": int(mask0.sum()), "n_after_exclusion": int(mask.sum())}
    return mask, mask0, dist, diag


def _median_with_bootstrap(vals: np.ndarray, seg_ids: np.ndarray, n_boot: int, seed: int = 0) -> dict[str, Any]:
    ok = np.isfinite(vals)
    vals, seg_ids = vals[ok], seg_ids[ok]
    if vals.size == 0:
        return {"C_ht": float("nan"), "n_pixels": 0}
    rec: dict[str, Any] = {"C_ht": float(np.median(vals)), "n_pixels": int(vals.size), "frac_convex": float(np.mean(vals > 0))}
    uniq = np.unique(seg_ids)
    if n_boot > 0 and uniq.size >= 5:
        rng = np.random.default_rng(seed)
        bs = np.empty(n_boot)
        for i in range(n_boot):
            pick = rng.choice(uniq, uniq.size, replace=True)
            counts = np.bincount(pick, minlength=int(seg_ids.max()) + 1)
            bs[i] = _weighted_median(vals, counts[seg_ids])
        rec["C_ht_ci68"] = [float(np.percentile(bs, 16)), float(np.percentile(bs, 84))]
        rec["C_ht_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return rec


def window_test(Z, mask, dist, dx, dy, U: float, half_widths=TEST_HALF_WIDTHS, w_f: int = W_F, n_boot: int = N_BOOT,
                tol: float = PASS_TOL) -> dict[str, Any]:
    """Section 2.2: median C_ht(L) with segment-bootstrap bands on the option (iii) set, and the pre-registered pass rule."""
    seg, n_seg = label(mask, structure=np.ones((3, 3)))
    seg_ids_all = seg[mask]
    d = dist[mask]
    width = {str(q): float(2 * np.percentile(d, q)) for q in (5, 10, 25, 50, 75, 90, 95)} if d.size else None
    rows = []
    for w in half_widths:
        C = total_curvature(Z, w, dx, dy)
        rec = _median_with_bootstrap(-C[mask], seg_ids_all, n_boot)
        rec.update({"w": int(w), "L_m": float((2 * w + 1) * dx), "L_nominal_m": int(2 * w + 1)})
        rows.append(rec)
    ref = next(r for r in rows if r["w"] == w_f)
    C_f = ref["C_ht"]
    W50 = width["50"] if width else float("nan")
    tested = [r for r in rows if r["w"] >= w_f and r["L_nominal_m"] <= W50 + 1e-9]
    for r in rows:
        r["ratio_to_L_f"] = float(r["C_ht"] / C_f) if C_f and np.isfinite(C_f) else None
        r["in_test_range"] = bool(r["w"] >= w_f and r["L_nominal_m"] <= W50 + 1e-9)
        r["within_tol"] = bool(abs(r["ratio_to_L_f"] - 1.0) <= tol) if r["ratio_to_L_f"] is not None else None
    passed = all(r["within_tol"] for r in tested) if tested else True
    worst = max((abs(r["ratio_to_L_f"] - 1.0) for r in tested), default=0.0)
    return {"rows": rows, "n_pixels": int(mask.sum()), "n_segments": int(n_seg), "width_2x_distance_m": width, "L_f_m": float((2 * w_f + 1) * dx),
            "C_ht_at_L_f": C_f, "C_ht_ci68_at_L_f": ref.get("C_ht_ci68"), "C_ht_ci95_at_L_f": ref.get("C_ht_ci95"),
            "D0": float(U / C_f) if C_f and C_f > 0 else None,
            "D0_ci68": [float(U / ref["C_ht_ci68"][1]), float(U / ref["C_ht_ci68"][0])] if ref.get("C_ht_ci68") else None,
            "D0_ci95": [float(U / ref["C_ht_ci95"][1]), float(U / ref["C_ht_ci95"][0])] if ref.get("C_ht_ci95") else None,
            "test": {"rule": f"for every window with L_f <= L (nominal) <= median crest width, |C_ht(L)/C_ht(L_f) - 1| <= {tol:g}",
                     "median_crest_width_m": W50, "windows_tested_nominal_m": [r["L_nominal_m"] for r in tested],
                     "n_windows_tested": len(tested), "max_abs_deviation": float(worst), "pass": bool(passed),
                     "only_L_f_in_range": bool(len(tested) <= 1)}}


# ----------------------------------------------------------------------------- k_s at a fixed theta

def ks_at_theta(Z, A, FD, dx, dy, theta: float, A_c: float, labels, n_boot: int = N_BOOT, seed: int = 0) -> dict[str, Any]:
    """Pooled through-origin k_s at ``theta`` over the network above ``A_c`` (the audit's estimator and filters), with
    n_eff and a basin bootstrap of k_s at that fixed theta. Returns the per-basin moments too (for site pooling)."""
    net = channel_network(Z, A, FD, A_c, dx, dy)
    mom = basin_moments(net, np.array([float(theta)]), labels)
    return _ks_from_moments(mom, theta, A_c, n_boot, seed) | {"_moments": mom}


def _ks_from_moments(mom: dict[str, Any], theta: float, A_c: float, n_boot: int, seed: int = 0) -> dict[str, Any]:
    nb = int(mom["n"].size)
    out: dict[str, Any] = {"theta": float(theta), "A_c": float(A_c), "n_basins": nb, "n_px": int(mom.get("n_px_kept", 0)),
                           "n_basins_total": int(mom.get("n_basins_total", 0))}
    if nb == 0:
        out.update({"ks": None, "n_eff_centred": None, "n_eff_through_origin": None})
        return out
    sxx, sxy, sx, n = mom["sxx"][:, 0], mom["sxy"][:, 0], mom["sx"][:, 0], mom["n"]
    ks = float(sxy.sum() / sxx.sum())
    w_c = sxx - sx**2 / n
    out.update({"ks": ks, "n_eff_centred": _n_eff(w_c), "n_eff_through_origin": _n_eff(sxx),
                "share_largest_basin_centred": float(np.max(w_c) / np.sum(w_c)) if np.sum(w_c) > 0 else None,
                "r2_pooled_through_origin": float(1 - np.sum(mom["syy"] - sxy**2 / sxx) / np.sum(mom["syy"])) if np.sum(mom["syy"]) > 0 else None})
    out["n_eff_lt_3"] = bool(out["n_eff_centred"] < 3.0)
    if n_boot > 0 and nb >= 3:
        rng = np.random.default_rng(seed)
        bs = np.empty(n_boot)
        for i in range(n_boot):
            w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
            bs[i] = (sxy * w).sum() / (sxx * w).sum()
        out["ks_ci68"] = [float(np.percentile(bs, 16)), float(np.percentile(bs, 84))]
        out["ks_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
        out["bootstrap"] = {"unit": "basin", "n_boot": n_boot, "usable": not out["n_eff_lt_3"]}
    return out


def _concat(parts: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("n", "sx", "sxx", "sxy", "sy", "syy")
    out = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    out["n_basins_total"] = sum(p["n_basins_total"] for p in parts)
    out["n_px_kept"] = sum(p["n_px_kept"] for p in parts)
    return out


# ----------------------------------------------------------------------------- driver

def _measure_curvature_params() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for r in find_runs("measure"):
        try:
            p = json.loads((r / "params.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        for t, v in p.get("tiles", {}).items():
            if "curvature" in v and "A_c" in v["curvature"]:
                out[t] = dict(v["curvature"]) | {"_run": r.name}
    return out


def _latest_audit() -> tuple[str | None, dict[str, Any] | None]:
    runs = find_runs("measure_audit")
    for r in reversed(runs):
        try:
            return r.name, json.loads((r / "params.json").read_text())
        except Exception:  # noqa: BLE001
            continue
    return None, None


def run_phase6_inputs(tiles: Sequence[str], n_boot: int = N_BOOT, run_id: str | None = None, results_root=None,
                      site_m: dict[str, float] | None = None, w_f: int = W_F, crop=None) -> RunRecorder:
    rec = RunRecorder("phase6_inputs", "_".join(tiles) if len(tiles) <= 3 else f"{len(tiles)}tiles", run_id=run_id, root=results_root)
    t_start = time.perf_counter()
    # 1.1 site m from the exponent profiles (read, not rerun)
    from lemcvt.analyses.exponent_site import site_summary

    m_prov: dict[str, Any] = {}
    if site_m is not None:
        m_prov = {s_: {"m_site": v, "source": "command-line override"} for s_, v in site_m.items()}
    if site_m is None:
        summ = site_summary()
        site_m = {}
        for site, s in summ["sites"].items():
            site_m[site] = float(s["all"]["m_opt"])
            m_prov[site] = {"m_site": site_m[site], "S_min": s["all"]["S_min"], "interval_1pct": s["all"]["interval_0.01"],
                            "interval_5pct": s["all"]["interval_0.05"], "tiles": s["all"]["tiles"],
                            "m_profile_runs": {t: summ["runs"][t] for t in s["all"]["tiles"]},
                            "leave_one_out_m": {t: v["m_opt"] for t, v in s["leave_one_out"].items()}}
    rec.add_params("site_m", {"values": site_m, "provenance": m_prov, "source": "lemcvt.analyses.exponent_site.site_summary (S(m) minimum per site)"})
    curv = _measure_curvature_params()
    audit_run, audit = _latest_audit()
    rec.add_params("reference_runs", {"measure_audit": audit_run, "measure": {t: curv.get(t, {}).get("_run") for t in tiles}})
    results: dict[str, Any] = {}
    snaps = {}
    moments: dict[str, dict[float, dict]] = {}
    for tile in tiles:
        t0 = time.perf_counter()
        td = _metre_tile(tile, crop)
        Z, A, FD, dx, dy, U = td.Z, td.A, td.FD, td.dx, td.filled.dy, td.site.U
        site = td.site.site
        out: dict[str, Any] = {"tile": tile, "site": site, "U": U, "inputs": td.provenance(), "m_site": site_m[site]}
        # k_s at theta = m_site
        labels = edge_outlet_labels(A, FD)
        moments[tile] = {}
        out["ks"] = {}
        for a_c in KS_THRESHOLDS:
            r = ks_at_theta(Z, A, FD, dx, dy, site_m[site], a_c, labels, n_boot=n_boot)
            moments[tile][a_c] = r.pop("_moments")
            r["K0"] = float(U / r["ks"]) if r.get("ks") else None
            r["used_for_K0"] = bool(a_c == KS_USED_THRESHOLD)
            # cross-check against the audit's ks_at_theta curve at the same theta (same estimator)
            if audit:
                try:
                    th = np.asarray(audit["settings"]["thetas"], float)
                    j = int(np.argmin(np.abs(th - site_m[site])))
                    step = next(s for s in audit["tiles"][tile]["free_ladder"] if s["A_c"] == float(a_c))
                    r["audit_ks_same_theta"] = float(step["pooled"]["ks_at_theta"][j])
                    r["audit_theta_grid_value"] = float(th[j])
                except Exception:  # noqa: BLE001
                    r["audit_ks_same_theta"] = None
            out["ks"][f"{a_c:g}"] = r
            logger.info("%s k_s(theta=%.2f, A_c=%g) = %.4g (n_eff %.1f, %d basins) K0 = %.3e", tile, site_m[site], a_c, r["ks"] or float("nan"),
                        r.get("n_eff_centred") or float("nan"), r["n_basins"], r["K0"] or float("nan"))
        # option (iii) set and the window test
        cp = curv.get(tile)
        if not cp:
            raise FileNotFoundError(f"{tile}: no measure run with a curvature record (rollover A_c) found")
        A_c_roll = float(cp["A_c"])
        mask, mask0, dist, diag = option_iii_mask(Z, A, FD, dx, dy, A_c_roll, w_f)
        wt = window_test(Z, mask, dist, dx, dy, U, w_f=w_f, n_boot=n_boot)
        wt["set"] = diag
        wt["definition"] = ("hilltop pixels of the rollover network as `measure` defines them (divide cells between outlet basins, "
                            f"two cells from the edge, |grad z| <= 0.4), excluding pixels closer than L_f/2 = {0.5 * wt['L_f_m']:.3g} m to a pixel of that network")
        out["option_iii"] = wt
        # the 02b set for the record: the measure set without exclusion at the tile's plateau window L*
        w_star = int(cp.get("w_star") or 0)
        seg0, _n0 = label(mask0, structure=np.ones((3, 3)))
        rec02b: dict[str, Any] = _median_with_bootstrap(-total_curvature(Z, w_star, dx, dy)[mask0], seg0[mask0], n_boot) if w_star else {"C_ht": None}
        rec02b.update({"w": w_star, "L_star_m": cp.get("L_star_m"), "measure_run_C_ht": cp.get("C_ht"), "measure_run": cp.get("_run"),
                       "D0": float(U / rec02b["C_ht"]) if rec02b.get("C_ht") else None, "definition": "measure set, no exclusion, plateau window L*"})
        out["set_02b"] = rec02b
        # audit cross-check of the option (iii) value
        if audit:
            try:
                o3 = audit["tiles"][tile]["hilltops"]["option_iii"]["at_fixed_L"]
                out["option_iii"]["audit_C_ht_at_L_f"] = o3["C_ht"]
                out["option_iii"]["audit_n_pixels"] = o3["n_pixels"]
            except Exception:  # noqa: BLE001
                pass
        out["wall_s"] = time.perf_counter() - t0
        results[tile] = out
        snaps[tile] = config_snapshot(load_tile_config(tile))
        logger.info("%s option (iii): %d pixels, C_ht(L_f) %.4g, D0 %.4g; window test %s (max dev %.1f %% over %d windows, W50 %.1f m)", tile,
                    wt["n_pixels"], wt["C_ht_at_L_f"], wt["D0"] or float("nan"), "PASS" if wt["test"]["pass"] else "FAIL",
                    100 * wt["test"]["max_abs_deviation"], wt["test"]["n_windows_tested"], wt["test"]["median_crest_width_m"])
    # site pooling of k_s and the pass rule
    sites = sorted({t[:2] for t in tiles})
    pooled: dict[str, Any] = {}
    for site in sites:
        st = [t for t in tiles if t.startswith(site)]
        ps: dict[str, Any] = {"tiles": st, "ks_site_pooled": {}}
        for a_c in KS_THRESHOLDS:
            parts = [moments[t][a_c] for t in st if moments[t][a_c]["n"].size]
            ps["ks_site_pooled"][f"{a_c:g}"] = _ks_from_moments(_concat(parts), site_m[site], a_c, n_boot) if parts else None
        n_pass = sum(1 for t in st if results[t]["option_iii"]["test"]["pass"])
        ps["option_iii_test"] = {"n_tiles": len(st), "n_pass": n_pass, "min_pass": PASS_MIN_TILES, "pass": bool(n_pass >= PASS_MIN_TILES),
                                 "per_tile": {t: results[t]["option_iii"]["test"]["pass"] for t in st},
                                 "trivial_tiles_only_L_f_in_range": [t for t in st if results[t]["option_iii"]["test"]["only_L_f_in_range"]]}
        pooled[site] = ps
        logger.info("%s: option (iii) test %d of %d tiles pass -> %s; site-pooled k_s(1e3) %.4g", site, n_pass, len(st),
                    "STANDS" if ps["option_iii_test"]["pass"] else "FAILS (stop)", (ps["ks_site_pooled"]["1000"] or {}).get("ks") or float("nan"))
    rec.add_params("tiles", results)
    rec.add_params("sites", pooled)
    rec.add_params("decision", {"option_iii_stands": bool(all(pooled[s]["option_iii_test"]["pass"] for s in sites)),
                                "rule": f"pass on >= {PASS_MIN_TILES} of 5 tiles at each site; otherwise stop before writing configs"})
    rec.add_params("settings", {"w_f": w_f, "test_half_widths": list(TEST_HALF_WIDTHS), "pass_tol": PASS_TOL, "n_boot": n_boot,
                                "ks_thresholds": list(KS_THRESHOLDS), "ks_used_threshold": KS_USED_THRESHOLD,
                                "bootstrap_units": "basin (k_s), 8-connected hilltop segment (C_ht)"})
    rows = []
    for t, r in results.items():
        for q in r["option_iii"]["rows"]:
            rows.append({"tile": t, "site": r["site"], **{k: v for k, v in q.items() if k not in ("C_ht_ci68", "C_ht_ci95")},
                         "lo68": (q.get("C_ht_ci68") or [None, None])[0], "hi68": (q.get("C_ht_ci68") or [None, None])[1],
                         "lo95": (q.get("C_ht_ci95") or [None, None])[0], "hi95": (q.get("C_ht_ci95") or [None, None])[1]})
    rec.write_table(pd.DataFrame(rows), "window_test")
    rec.add_params("wall_s", time.perf_counter() - t_start)
    rec.finalize({"tiles": snaps, "crop": crop})
    return rec
