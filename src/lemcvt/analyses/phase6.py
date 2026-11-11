"""Phase 6 orchestration helpers (addendum 5): the measured inputs into the configs (2.3), the budget estimate
(3.0), the D-recovery diagnostics (3.7) and the old-against-new record (3.8).

Everything here reads results and configs; the model computations live in :mod:`lemcvt.analyses.optimizer_v2`
and the measurements in :mod:`lemcvt.measure.phase6_inputs`.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from lemcvt.config import REPO_ROOT, TILE_IDS, config_root, load_site, load_tile_config
from lemcvt.results import find_runs, git_hash

logger = logging.getLogger(__name__)

NONLINEAR_MAX_ITER_PHASE6 = 100       # decision 6


# ----------------------------------------------------------------------------- 2.3 configs

def _latest(step: str, label: str | None = None) -> tuple[Path, dict] | None:
    """Latest completed run of ``step`` that is not a cropped smoke run."""
    for r in reversed(find_runs(step, label)):
        try:
            rj = json.loads((r / "run.json").read_text())
            if (rj.get("config") or {}).get("crop"):
                continue
            return r, json.loads((r / "params.json").read_text())
        except Exception:  # noqa: BLE001
            continue
    return None


def _run_meta(run: Path) -> dict[str, Any]:
    rj = json.loads((run / "run.json").read_text())
    return {"run": run.name, "git_hash": rj.get("git_hash"), "slurm_job_id": rj.get("slurm_job_id")}


def build_tile_config(tile: str, inputs: dict[str, Any], inputs_meta: dict[str, Any], seamfix: dict[str, Any] | None,
                      seamfix_meta: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    """The Phase 6 tile config (a dict and a header comment) from the measured-inputs run and, for AP1, the seam fix."""
    raw = yaml.safe_load((config_root() / "tiles" / f"{tile}.yaml").read_text())
    cfg_m = load_tile_config(tile, reproduce_v1_units=False)          # scaled bounds in metres (Phase 5.3)
    site = load_site(raw["site"])
    ti = inputs["tiles"][tile]
    m_site = float(inputs["site_m"]["values"][raw["site"]])
    m_prov = inputs["site_m"]["provenance"].get(raw["site"], {})
    ks_used = ti["ks"]["1000"]
    ks_1e4 = ti["ks"]["10000"]
    o3 = ti["option_iii"]
    site_pool = inputs["sites"][raw["site"]]["ks_site_pooled"]

    v1ref = dict(raw.get("v1_reference", {}))
    if "model" not in v1ref:
        v1ref["model"] = dict(raw["model"])
        if "param_z_units" not in v1ref["model"]:
            v1ref["model"]["param_z_units"] = raw["units"]["z_units"]
    grids = dict(raw["grids"])
    if seamfix and seamfix.get("written", {}).get("fd") and seamfix.get("tile") == tile:
        if "grids" not in v1ref:
            v1ref["grids"] = {"fd": grids["fd"], "area": grids["area"], "sha256": {"fd": grids["sha256"]["fd"], "area": grids["sha256"]["area"]}}
        grids = dict(grids)
        grids["fd"] = seamfix["written"]["fd"]["path"]
        grids["area"] = seamfix["written"]["area"]["path"]
        grids["sha256"] = dict(grids["sha256"]) | {"fd": seamfix["written"]["fd"]["sha256"], "area": seamfix["written"]["area"]["sha256"]}

    measured = {
        "m": m_site,
        "ks": float(ks_used["ks"]),
        "C_ht": float(o3["C_ht_at_L_f"]),
        "z_units": "m",
        "ks_units": f"m^(2 m) with m = {m_site:g}, A0 = 1 m^2",
        "K0": float(site.U / ks_used["ks"]),
        "D0": float(site.U / o3["C_ht_at_L_f"]),
        "ks_1e4_reported_not_used": float(ks_1e4["ks"]) if ks_1e4.get("ks") else None,
        "provenance": {
            "m": {"definition": "site constant: minimum of the site exponent profile S(m) over the five tiles (addendum 5, 1.1)",
                  "S_interval_1pct": m_prov.get("interval_1pct", {}).get("intervals"), "S_interval_5pct": m_prov.get("interval_5pct", {}).get("intervals"),
                  "S_min": m_prov.get("S_min"), "m_profile_runs": m_prov.get("m_profile_runs"), "leave_one_out_m": m_prov.get("leave_one_out_m")},
            "ks": {"definition": "pooled through-origin chi-z slope at theta = m over the channel network above A_c = 1e3 m^2 (addendum 5, 1.2)",
                   "A_c": 1000.0, "n_basins": ks_used["n_basins"], "n_eff_centred": ks_used.get("n_eff_centred"),
                   "n_eff_through_origin": ks_used.get("n_eff_through_origin"), "n_eff_lt_3": ks_used.get("n_eff_lt_3"),
                   "ci68": ks_used.get("ks_ci68"), "ci95": ks_used.get("ks_ci95"), "bootstrap": ks_used.get("bootstrap"),
                   "site_pooled_ks_1e3": (site_pool.get("1000") or {}).get("ks"), "site_pooled_ks_1e4": (site_pool.get("10000") or {}).get("ks"),
                   "ks_1e4": {"ks": ks_1e4.get("ks"), "n_basins": ks_1e4.get("n_basins"), "n_eff_centred": ks_1e4.get("n_eff_centred"),
                              "ci68": ks_1e4.get("ks_ci68")},
                   **inputs_meta},
            "C_ht": {"definition": "curvature option (iii): median -laplacian at the fixed window L_f on the rollover-network hilltop set with the L_f/2 exclusion (addendum 5, 1.3)",
                     "L_f_m": o3["L_f_m"], "n_pixels": o3["n_pixels"], "n_segments": o3["n_segments"], "ci68": o3.get("C_ht_ci68_at_L_f"),
                     "ci95": o3.get("C_ht_ci95_at_L_f"), "D0_ci68": o3.get("D0_ci68"), "D0_ci95": o3.get("D0_ci95"),
                     "window_test": {k: v for k, v in o3["test"].items() if k != "rule"}, "site_test_pass": inputs["sites"][raw["site"]]["option_iii_test"]["pass"],
                     **inputs_meta},
        },
    }
    if seamfix and seamfix.get("tile") == tile:
        measured["provenance"]["grids"] = {"seamfix": seamfix_meta, "n_seam_cells": seamfix["seam"]["n_seam"],
                                           "n_cells_area_changed": seamfix["changes"]["n_cells_area_changed"]}
    old_model = raw["model"]
    model = {
        "param_z_units": "m",
        "m": m_site,
        "S_c": old_model.get("S_c", site.S_c),
        "K0_ad": "U/ks",
        "K0_partitioned": "U/ks",
        "D0": "U/C_ht",
        "K_bounds": [float(cfg_m.model.K_bounds[0]), float(cfg_m.model.K_bounds[1])],
        "D_bounds": [float(cfg_m.model.D_bounds[0]), float(cfg_m.model.D_bounds[1])],
        "A_crit_0": old_model["A_crit_0"],
        "A_crit_bounds": old_model["A_crit_bounds"],
        "nonlinear_max_iter": NONLINEAR_MAX_ITER_PHASE6,
        "nonlinear_tol": old_model.get("nonlinear_tol", 1e-6),
        "methods": old_model.get("methods", {}),
    }
    out = {"tile": raw["tile"], "site": raw["site"], "grids": grids, "units": raw["units"], "measured": measured, "model": model,
           "measured_v1": raw["measured_v1"], "v1_reference": v1ref}
    header = (f"# Tile {tile}: Phase 6 configuration (addendum 5, section 2.3; written by `lemcvt phase6 write-configs` from run\n"
              f"# {inputs_meta.get('run')} at git {inputs_meta.get('git_hash')}).\n"
              f"# `measured` holds the v2 pipeline's inputs with provenance: m is the site constant (S(m) minimum), ks the pooled\n"
              f"# through-origin chi-z slope at theta = m above A_c = 1e3 m^2 (K0 = U/ks), C_ht by curvature option (iii)\n"
              f"# (L_f = 11 cells, rollover-network hilltop set, L_f/2 exclusion; D0 = U/C_ht). The model block refers to z in\n"
              f"# metres (param_z_units: m); the K and D bounds are the Phase 5.3 metre bounds. nonlinear_max_iter is 100\n"
              f"# (decision 6). The v1 values stay under `measured_v1` and `v1_reference` (`v1_reference.model` is the v1 model\n"
              f"# block, used by the loader for feet runs with --units feet; `v1_reference.grids`, where present, the original\n"
              f"# grids replaced by the seam-corrected ones).\n")
    return out, header


def write_tile_configs(tiles=TILE_IDS, inputs_run: str | Path | None = None, seamfix_run: str | Path | None = None,
                       write: bool = True, require_pass: bool = True) -> dict[str, Any]:
    """Section 2.3: write m, K0 (U/k_s), D0 (option iii) and the seam-corrected AP1 grids into the tile configs."""
    from lemcvt.config import results_root

    def _resolve(step, run):
        if run is None:
            got = _latest(step)
            if got is None:
                raise FileNotFoundError(f"no {step} run found")
            return got
        p = Path(run)
        p = p if p.is_absolute() else results_root() / p
        return p, json.loads((p / "params.json").read_text())

    in_run, inputs = _resolve("phase6_inputs", inputs_run)
    in_meta = _run_meta(in_run)
    if require_pass and not inputs["decision"]["option_iii_stands"]:
        raise RuntimeError("the option (iii) window test did not pass at every site; nothing is written (addendum 5, 2.2)")
    sf = sf_meta = None
    if any(t == "AP1" for t in tiles):
        try:
            sf_run, sf = _resolve("seamfix", seamfix_run)
            sf_meta = _run_meta(sf_run)
        except FileNotFoundError:
            logger.warning("no seamfix run: AP1 keeps the stored fd/area grids")
    written = {}
    for tile in tiles:
        cfg, header = build_tile_config(tile, inputs, in_meta, sf if (sf and sf.get("tile") == tile) else None, sf_meta)
        text = header + yaml.safe_dump(cfg, sort_keys=False, width=110, default_flow_style=None)
        path = config_root() / "tiles" / f"{tile}.yaml"
        if write:
            path.write_text(text)
            chk = load_tile_config(tile)         # the loader must accept it and resolve K0, D0
            written[tile] = {"path": str(path), "m": chk.model.m, "K0_partitioned": chk.model.K0_partitioned, "D0": chk.model.D0,
                             "K0_ad": chk.model.K0_ad, "param_scale": chk.model.param_scale, "grids": chk.grids, "nonlinear_max_iter": chk.model.nonlinear_max_iter}
            feet = load_tile_config(tile, reproduce_v1_units=True)
            written[tile]["v1_feet_reload"] = {"m": feet.model.m, "K0_partitioned": feet.model.K0_partitioned, "D0": feet.model.D0, "grids": feet.grids}
        else:
            written[tile] = {"path": str(path), "text": text}
    return {"inputs_run": in_meta, "seamfix_run": sf_meta, "tiles": written}


# ----------------------------------------------------------------------------- 3.0 budget

# per-evaluation wall times (s) measured in Phases 3-5 (03_performance G2; checkpoint 4b/5 sections 2 and 4.6)
COST = {"partitioned_eval": {"AP": 7.0, "GM": 6.0}, "ad_eval": {"AP": 27.0, "GM": 25.0},
        "picard_iteration": {"AP": 5.0, "GM": 5.0}, "dinf_lu": {"AP": 15.0, "GM": 15.0}}
TIME_FACTOR = 1.5          # addendum 2, section 3: limit = 1.5 x the expected time, rounded up to the hour
BUDGET_SUBMISSION = 300.0
BUDGET_TOTAL = 900.0


def _ceil_h(hours: float, min_h: int = 1) -> int:
    return max(min_h, int(math.ceil(hours)))


def estimate(tiles=TILE_IDS, fill_counts: dict[str, int] | None = None) -> dict[str, Any]:
    """Core-hour estimate of every Phase 6 item (expected wall time, the 1.5x limit, tasks, CPUs), and the total."""
    items = []

    def add(name, tasks, exp_h, cpus=1, note="", min_h=1, mem="8G"):
        lim = _ceil_h(TIME_FACTOR * exp_h, min_h)
        items.append({"item": name, "tasks": tasks, "cpus": cpus, "expected_h_per_task": round(exp_h, 2), "limit_h_per_task": lim,
                      "core_hours_limit": tasks * cpus * lim, "core_hours_expected": round(tasks * cpus * exp_h, 1), "mem": mem, "note": note})

    for site in ("AP", "GM"):
        st = [t for t in tiles if t.startswith(site)]
        if not st:
            continue
        c = COST
        # 3.1 AD: 200 grid + Brent (~20)
        add(f"3.1 AD profile ({site})", len(st), 220 * c["ad_eval"][site] / 3600, note="220 reduced solves")
        # 3.2 partitioned: thin 2000 + refine (<= 400) + fill every unique area in the 5 % intervals
        fill = (fill_counts or {}).get(site, {"AP": 4300, "GM": 1800}[site])
        n_eval = 2000 + 400 + fill
        add(f"3.2 linear partitioned: thin + refine + 5 % fill ({site})", len(st), n_eval * c["partitioned_eval"][site] / 3600,
            note=f"~{n_eval} LU+solves ({fill} in the 5 % intervals, from the m-profile intervals at the site m)")
        # 3.3 nonlinear A_crit profile: 260 evaluations; iterations 6-12 near the minimum, up to the cap of 100 at large A_crit
        it = {"AP": 12.0, "GM": 20.0}[site]           # mean Picard iterations per evaluation (checkpoint 4b/5: AP 6-12; GM 6-50 with cap hits)
        add(f"3.3 nonlinear A_crit profile ({site})", len(st), 260 * it * c["picard_iteration"][site] / 3600,
            note=f"260 Picard solves, ~{it:.0f} iterations each (cap 100)")
        # 3.4 nonlinear joint: 30 x 25 Picard solves; small D saturates the slopes and hits the cap
        frac_cap = {"AP": 0.10, "GM": 0.32}[site]      # share of the D grid below ~D0/2
        mean_it = frac_cap * 100 + (1 - frac_cap) * 10
        exp_nlj = 750 * mean_it * c["picard_iteration"][site] / 3600
        add(f"3.4 nonlinear joint (A_crit, D) profile ({site})", len(st), exp_nlj, note=f"750 Picard solves; ~{100 * frac_cap:.0f} % at the cap")
        kchk = [t for t in st if t in ("GM1", "AP5")]
        if kchk:
            add(f"3.4 K check x0.95, x1.05 ({', '.join(kchk)})", 2 * len(kchk), exp_nlj, note="same grid at K x 0.95 and K x 1.05")
        if site in ("AP", "GM") and any(t in ("AP1", "GM1") for t in st):
            add(f"3.5 D-inf joint profile ({site}1)", 1, (2000 + 300) * c["dinf_lu"][site] / 3600, mem="32G",
                note="2000 thinned unique A_dinf values + refinement; one LU of the coupled system each")
    add("3.6 resolution sweep (config params, D8 + D-inf)", len(tiles), 0.5, cpus=2, mem="24G", note="25 dx x 2 routings per tile")
    add("3.6 supp_fig5 synthetic + d8_vs_dinf at the measured parameters (metres)", 1, 1.0, mem="64G", note="d8_vs_dinf still assembles the v1 AD D-inf model with LIL loops")
    add("2.1 + 2.2 + tests: seam fix, measured inputs, tests on serc", 1, 1.0, mem="32G", note="Phase 6.0 utilities job")
    add("3.7 + 3.6 + 3.8: D-recovery, slope-area, figures, record", 1, 1.0, mem="32G", note="after every model run")
    df = pd.DataFrame(items)
    tot_lim = float(df.core_hours_limit.sum())
    tot_exp = float(df.core_hours_expected.sum())
    return {"items": items, "total_core_hours_limit": tot_lim, "total_core_hours_expected": tot_exp, "budget_total": BUDGET_TOTAL,
            "budget_per_submission": BUDGET_SUBMISSION, "within_total_budget": bool(tot_lim <= BUDGET_TOTAL),
            "max_single_item_limit": float(df.core_hours_limit.max()), "cost_assumptions": COST, "time_factor": TIME_FACTOR,
            "table": df.to_string(index=False)}


def fill_counts_from_m_profiles(m_site: dict[str, float], level: float = 0.05) -> dict[str, Any]:
    """Unique drainage areas inside the joint and stage-2 5 % intervals at the site m (from the latest m-profile runs)."""
    from lemcvt.analyses.exponent_site import latest_m_profile_runs
    from lemcvt.config import data_root
    from lemcvt.io.grids import read_geotiff

    runs = latest_m_profile_runs()
    out: dict[str, Any] = {"per_tile": {}}
    for t, (r, p) in runs.items():
        df = pd.read_parquet(r / "m_profile.parquet")
        sub = df[np.isclose(df.m, m_site[t[:2]])]
        if sub.empty:
            continue
        cfg = load_tile_config(t)
        A = read_geotiff(data_root() / cfg.grids["area"]).data
        a = A[np.isfinite(A)]
        rec = {}
        for col in ("mse_joint", "mse_stage2"):
            sel = sub[sub[col] <= (1 + level) * sub[col].min()]
            lo, hi = float(sel.A_crit.min()), float(sel.A_crit.max())
            rec[col] = {"A_min": lo, "A_max": hi, "n_unique_inside": int(np.unique(a[(a >= lo) & (a <= hi)]).size)}
        out["per_tile"][t] = rec
    for site in ("AP", "GM"):
        st = [t for t in out["per_tile"] if t.startswith(site)]
        if st:
            out[site] = int(round(np.mean([out["per_tile"][t]["mse_joint"]["n_unique_inside"] + out["per_tile"][t]["mse_stage2"]["n_unique_inside"] for t in st])))
    return out
