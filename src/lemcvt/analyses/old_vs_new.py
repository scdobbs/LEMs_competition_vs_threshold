"""Addendum 5, 3.8 and addendum 6, 5.8: the old-against-new record, built from results with one command.

Every number the v1 manuscript reports (``reference/v1_reference.json``: the ten tiles' four stages, the measured
inputs, the ``d8_vs_dinf`` MSE table, the resolution-sweep ratios, the slope-area fits) is paired with its Phase 6
value, the ratio, and the causes that apply. Causes are assigned mechanically:

* ``vertical units``: every AP value that carries the vertical unit (K, D, MSE, k_s, C_ht);
* ``optimizer``: every fitted stage (v1 Nelder-Mead / L-BFGS-B against v2 profiles);
* ``measurement``: every measured input and every stage that takes K0 or D0 (stage 2, nonlinear, AD no-opt, sweep);
* ``m``: every model result (the exponent is now the site constant);
* ``AP D0 bracket``: every AP quantity that takes D0 (addendum 6);
* ``AP1 seam``: every AP1 quantity, qualified by the section 3 verdict;
* ``D-inf rebuild``: the D-infinity partitioned rows;
* ``unconverged v1 nonlinear``: the AP nonlinear rows.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lemcvt.config import REPO_ROOT, TILE_IDS, load_tile_config
from lemcvt.io.units import FT_US_IN_M, M_IN_FT_US
from lemcvt.results import find_runs

logger = logging.getLogger(__name__)

REF = REPO_ROOT / "reference" / "v1_reference.json"
CENTRE_FORM_NOTE = "centre form, D_eff lap(z) = -U (nonstandard); v1 and v2 alike (N10)"


def _latest(step: str, pred=None, label: str | None = None):
    for r in reversed(find_runs(step, label)):
        try:
            p = json.loads((r / "params.json").read_text())
            rj = json.loads((r / "run.json").read_text())
        except Exception:  # noqa: BLE001
            continue
        if (rj.get("config") or {}).get("crop"):
            continue
        if pred and not pred(p):
            continue
        return r, p
    return None, None


def phase6_runs(tile: str) -> dict[str, tuple[Path, dict]]:
    from lemcvt.analyses.curvature_compare import _runs_by_stage

    return _runs_by_stage(tile)


def build_record(seam_verdict: str | None = None) -> pd.DataFrame:
    ref = json.loads(REF.read_text())
    rows: list[dict[str, Any]] = []

    def add(tile, quantity, v1, v2, causes, unit_v1="", unit_v2="", note=""):
        ratio = (v2 / v1) if (v1 not in (None, 0) and v2 is not None and np.isfinite(v1) and np.isfinite(v2)) else None
        rows.append({"tile": tile, "quantity": quantity, "v1": v1, "v2": v2, "ratio_v2_over_v1": ratio, "causes": "; ".join(causes),
                     "unit_v1": unit_v1, "unit_v2": unit_v2, "note": note})

    for tile in TILE_IDS:
        site = tile[:2]
        ap = site == "AP"
        t = ref["tiles"][tile]
        cfg = load_tile_config(tile)
        runs = phase6_runs(tile)
        base = ["m"] + (["vertical units"] if ap else []) + (["AP1 seam" + (f" ({seam_verdict})" if seam_verdict else "")] if tile == "AP1" else [])
        c = t["config"]
        # measured inputs
        f = M_IN_FT_US if ap else 1.0
        meas = cfg.measured
        add(tile, "theta / m", c["concavity_theta"], meas.get("m"), ["measurement", "m"], note="v1 per-tile theta; v2 site constant")
        add(tile, "k_s", c["steepness_ks"], meas.get("ks"), ["measurement", "m"] + (["vertical units"] if ap else []), "ft-based" if ap else "m^(2 theta)", "m^(2 m)",
            note="not comparable across theta; v1 AP in feet")
        add(tile, "C_ht", c["hilltop_curvature"], meas.get("C_ht"), ["measurement"] + (["vertical units", "AP D0 bracket"] if ap else []), "1/ft" if ap else "1/m", "1/m")
        add(tile, "K0 = U/k_s", c["K0_from_ks"] * (f if ap else 1.0), cfg.model.K0_partitioned, ["measurement", "m"] + (["vertical units"] if ap else []), "1/yr (converted x c)" if ap else "1/yr", "1/yr")
        add(tile, "D0 = U/C_ht", c["D0_from_curvature"] * (f if ap else 1.0), cfg.model.D0, ["measurement"] + (["vertical units", "AP D0 bracket"] if ap else []), "m^2/yr (converted x c)" if ap else "m^2/yr", "m^2/yr",
            note=f"D0_B = {cfg.model.D0_B:.3e}" if cfg.model.D0_B else "")
        st = t["stages"]
        mse_f = FT_US_IN_M**2 if ap else 1.0
        # AD
        if "ad" in runs:
            o = runs["ad"][1]["v2"]["ad"]["optimum"]
            add(tile, "AD K", st["ad"]["K"] * f, o["K"], base + ["optimizer"], "1/yr (x c)" if ap else "1/yr", "1/yr")
            add(tile, "AD D", st["ad"]["D"] * f, o["D"], base + ["optimizer"], "m^2/yr (x c)" if ap else "m^2/yr", "m^2/yr")
            add(tile, "AD MSE", st["ad"]["MSE"] * mse_f, o["mse"], base + ["optimizer"], "m^2 (x 1/c^2)" if ap else "m^2", "m^2")
        # stage 2 and joint
        if "joint" in runs:
            v = runs["joint"][1]["v2"]["partitioned"]
            s2, jo = v["stage2_optimum"], v["joint_optimum"]
            add(tile, "stage-2 A_crit", st["acrit"]["A_crit"], s2["A_crit"], base + ["optimizer", "measurement"] + (["AP D0 bracket"] if ap else []), "m^2", "m^2")
            add(tile, "stage-2 MSE", st["acrit"]["MSE"] * mse_f, s2["mse"], base + ["optimizer", "measurement"] + (["AP D0 bracket"] if ap else []), "m^2 (x 1/c^2)" if ap else "m^2", "m^2")
            add(tile, "joint A_crit", st["joint"]["A_crit"], jo["A_crit"], base + ["optimizer"], "m^2", "m^2")
            add(tile, "joint K", st["joint"]["K"] * f, jo["K"], base + ["optimizer"], "1/yr (x c)" if ap else "1/yr", "1/yr")
            add(tile, "joint D", st["joint"]["D"] * f, jo["D"], base + ["optimizer"], "m^2/yr (x c)" if ap else "m^2/yr", "m^2/yr")
            add(tile, "joint MSE", st["joint"]["MSE"] * mse_f, jo["mse"], base + ["optimizer"], "m^2 (x 1/c^2)" if ap else "m^2", "m^2")
        if "nonlinear" in runs:
            o = runs["nonlinear"][1]["v2"]["nonlinear"]["optimum"]
            cz = base + ["optimizer", "measurement"] + (["AP D0 bracket", "unconverged v1 nonlinear"] if ap else [])
            # addendum 10, 1.1: both the v1 and the v2 value solve the centre form D_eff lap(z) = -U (N10)
            add(tile, "nonlinear A_crit", st["nonlinear"]["A_crit"], o["A_crit"], cz, "m^2", "m^2", note=CENTRE_FORM_NOTE)
            add(tile, "nonlinear MSE", st["nonlinear"]["MSE"] * mse_f, o["mse"], cz, "m^2 (x 1/c^2)" if ap else "m^2", "m^2", note=CENTRE_FORM_NOTE)
    # d8_vs_dinf (v1 at measured parameters in feet on AP; v2 at the config parameters in metres)
    r, p = _latest("d8_vs_dinf", lambda q: q.get("params_source") == "config")
    if p:
        v1 = ref["figure_notebooks"]["d8_vs_dinf_comparison"]["parsed"]["locations"]
        for tile, lab in (("AP1", "Allegheny Plateau (tile 1)"), ("GM1", "Gabilan Mesa (tile 1)")):
            if tile not in p["tiles"]:
                continue
            ap = tile == "AP"
            q = p["tiles"][tile]
            mse_f = FT_US_IN_M**2 if tile.startswith("AP") else 1.0
            cz = ["measurement", "m"] + (["vertical units", "AP D0 bracket"] if tile.startswith("AP") else []) + (["AP1 seam"] if tile == "AP1" else [])
            add(tile, "d8_vs_dinf MSE AD D8", v1[lab]["MSE_AD_D8"] * mse_f, q["mse_ad_d8"], cz)
            add(tile, "d8_vs_dinf MSE AD D-inf", v1[lab]["MSE_AD_Dinf"] * mse_f, q["mse_ad_dinf"], cz)
            add(tile, "d8_vs_dinf MSE PT D8", v1[lab]["MSE_PT_D8"] * mse_f, q["mse_pt_d8"], cz)
            add(tile, "d8_vs_dinf MSE PT D-inf", v1[lab]["MSE_PT_Dinf"] * mse_f, q.get("mse_pt_dinf"), cz + ["D-inf rebuild"])
    # resolution sweep ratios (unit-free)
    sw = [r for r in find_runs("sweep_resolution") if (json.loads((r / "params.json").read_text()).get("params_source") == "config"
                                                        and json.loads((r / "params.json").read_text()).get("d0_variant", "A") == "A")]
    if sw:
        frames = [pd.read_parquet(r / "table.parquet").assign(run=r.name) for r in sw if (r / "table.parquet").exists()]
        if frames:
            df = pd.concat(frames).sort_values("run").drop_duplicates(subset=["tile", "dx"], keep="last")
            v1rows = ref["figure_notebooks"]["resolution_study_dinf"]["parsed"]["ok_rows"]
            for q in v1rows:
                tile = f"{q['site']}{q['tile']}"
                m = df[(df.tile == tile) & np.isclose(df.dx, q["dx"], rtol=1e-3)]
                if m.empty:
                    continue
                cz = ["measurement", "m"] + (["AP D0 bracket"] if tile.startswith("AP") else []) + (["AP1 seam"] if tile == "AP1" else []) + ["tie-break N1-N3"]
                add(tile, f"sweep ratio D8 at dx {q['dx']}", q["ratio_d8"], float(m.ratio_d8.iloc[0]), cz, "-", "-")
                add(tile, f"sweep ratio D-inf at dx {q['dx']}", q["ratio_dinf"], float(m.ratio_dinf.iloc[0]), cz, "-", "-")
    # slope-area fits
    r, p = _latest("slope_area", lambda q: "v1 saved grids" not in json.dumps(q.get("tiles", {})))
    if p:
        v1 = ref["figure_notebooks"]["fig_slope_area"]["parsed"]["tiles"]
        keymap = {"ORIGINAL": "orig", "AD no-opt": "AD_no_opt", "AD opt": "AD_opt", "PARTITIONED": "part"}
        for tile, series in v1.items():
            if tile not in p["tiles"]:
                continue
            for k1, k2 in keymap.items():
                if k2 not in p["tiles"][tile]:
                    continue
                cz = (["optimizer", "measurement", "m"] if k2 != "orig" else []) + (["vertical units"] if tile.startswith("AP") else []) + (["AP1 seam"] if tile == "AP1" else [])
                for q in ("Ks", "theta", "R2"):
                    add(tile, f"slope-area {k1} {q}", series[k1][q], p["tiles"][tile][k2][q], cz, note="Ks on AP in feet-based units in v1")
    return pd.DataFrame(rows)


def write_record(out_dir: Path, seam_verdict: str | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_record(seam_verdict)
    p_csv = out_dir / "old_vs_new_record.csv"
    df.to_csv(p_csv, index=False)
    lines = ["| tile | quantity | v1 | v2 | ratio v2/v1 | causes |", "|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        f = lambda v: "-" if v is None or (isinstance(v, float) and not np.isfinite(v)) else (f"{v:.4g}" if isinstance(v, float) else str(v))  # noqa: E731
        lines.append(f"| {r.tile} | {r.quantity} | {f(r.v1)} | {f(r.v2)} | {f(r.ratio_v2_over_v1)} | {r.causes} |")
    p_md = out_dir / "old_vs_new_record.md"
    p_md.write_text("\n".join(lines) + "\n")
    return [p_csv, p_md]
