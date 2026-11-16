#!/usr/bin/env python3
"""Extract every number printed in the v1 notebooks into reference/v1_reference.json.

Phase 0, step 3 of reports/directives/refactor_v2_directive.md.

Reads the ten executed model-run notebooks (notebooks/model_runs/*_complete.ipynb)
and the five figure notebooks (notebooks/figures/*.ipynb) and records, per tile and
per stage: configuration (theta, k_s, hilltop curvature, U, n, S_c, K0, D0, bounds),
optimizer results (parameters, MSE), forward-run parameters and saved-grid
prefixes, papermill cell durations, and the printed tables of the figure
notebooks.  Nothing is computed here; only text already present in the notebooks
is parsed.  Run from the repository root with any Python >= 3.8 (stdlib only).
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_RUNS = ROOT / "notebooks" / "model_runs"
FIGURES = ROOT / "notebooks" / "figures"
OUT = ROOT / "reference" / "v1_reference.json"

NUM = r"([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)"


def cell_source(cell) -> str:
    return "".join(cell["source"])


def cell_streams(cell) -> str:
    """Concatenate stdout/stderr streams and text/plain results of a code cell."""
    parts = []
    for o in cell.get("outputs", []):
        if o["output_type"] == "stream":
            parts.append("".join(o["text"]))
        elif o["output_type"] in ("execute_result", "display_data"):
            t = "".join(o.get("data", {}).get("text/plain", ""))
            if t and "Figure size" not in t:
                parts.append(t)
        elif o["output_type"] == "error":
            parts.append(f"ERROR {o.get('ename')}: {o.get('evalue')}")
    return "\n".join(parts)


def parse_assignments(src: str, names: tuple[str, ...]) -> dict:
    """Return {name: rhs_text} for simple `name = rhs` lines in a cell."""
    found = {}
    for line in src.splitlines():
        line = line.split("#", 1)[0].rstrip()
        m = re.match(r"^\s*(\w+)\s*=\s*(.+)$", line)
        if m and m.group(1) in names:
            found[m.group(1)] = m.group(2).strip()
    return found


def safe_eval(expr: str, ctx: dict):
    import math

    env = {"np": _NP(), "math": math}
    env.update(ctx)
    return eval(expr, {"__builtins__": {}}, env)  # noqa: S307 (trusted notebook text)


class _NP:
    """Minimal numpy stand-in so the script stays stdlib-only."""

    @staticmethod
    def power(a, b):
        return a ** b


def extract_model_run(path: Path) -> dict:
    nb = json.loads(path.read_text())
    cells = nb["cells"]
    name = path.stem.replace("_complete", "")
    site = "AP" if name.startswith("allegheny") else "GM"
    tile = int(re.search(r"tile(\d)", name).group(1))

    ctx: dict = {}
    config_names = (
        "concavity", "hilltop_curvature", "steepness", "U", "n", "m",
        "K_bounds", "D_bounds", "Sc", "A_crit_0", "A_crit_bounds", "SAVE", "base", "base_save",
    )
    k0_history = []  # (cell_index, expression, value)
    d0_value = None
    stages = {}
    forward_runs = []
    durations = {}
    misc_outputs = {}
    config_cell_source = None

    stage_patterns = [
        ("ad", r"optimize_advection_diffusion\("),
        ("acrit", r"optimize_threshold_acrit\("),
        ("nonlinear", r"optimize_threshold_acrit_nonlinear\("),
        ("joint", r"optimize_threshold_full\("),
    ]

    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = cell_source(c)
        out = cell_streams(c)
        dur = c.get("metadata", {}).get("papermill", {}).get("duration")
        if dur is not None:
            durations[str(i)] = round(float(dur), 3)

        # configuration assignments (any cell, sequential)
        assigns = parse_assignments(src, config_names)
        if "concavity" in assigns and config_cell_source is None:
            config_cell_source = src
        for k, rhs in assigns.items():
            try:
                ctx[k] = safe_eval(rhs, ctx)
            except Exception:
                ctx[k] = rhs
        # K0 / D0 assignments, in order of appearance
        for line in src.splitlines():
            line_nc = line.split("#", 1)[0]
            m2 = re.match(r"^\s*K0\s*,\s*D0\s*=\s*(.+)$", line_nc)
            if m2:
                try:
                    k0, d0 = safe_eval("(" + m2.group(1) + ")", ctx)
                except Exception:
                    k0, d0 = m2.group(1), None
                ctx["K0"], ctx["D0"] = k0, d0
                d0_value = d0
                k0_history.append({"cell": i, "expr": line_nc.strip(), "K0": k0, "D0": d0})
                continue
            m1 = re.match(r"^\s*K0\s*=\s*(.+)$", line_nc)
            if m1:
                try:
                    k0 = safe_eval(m1.group(1), ctx)
                except Exception:
                    k0 = m1.group(1)
                ctx["K0"] = k0
                k0_history.append({"cell": i, "expr": line_nc.strip(), "K0": k0, "D0": ctx.get("D0")})

        # optimisation stages
        for stage, pat in stage_patterns:
            if re.search(pat, src):
                rec = {"cell": i, "duration_s": durations.get(str(i))}
                for key in ("K", "D", "A_crit", "MSE"):
                    m = re.search(rf"^\s*{key}\s*=\s*{NUM}\s*$", out, re.M)
                    if m:
                        rec[key] = float(m.group(1))
                rec["K0_at_call"] = ctx.get("K0")
                rec["D0_at_call"] = ctx.get("D0")
                rec["raw_output"] = out.strip()
                stages[stage] = rec

        # forward runs (model evaluations that are plotted and/or saved)
        fwd = None
        if re.search(r"advection_diffusion\(U,\s*K0,\s*D0\)", src):
            fwd = {"model": "AD", "params": {"U": ctx.get("U"), "K": ctx.get("K0"), "D": ctx.get("D0")}}
        elif re.search(r"advection_diffusion\(U,\s*K_star,\s*D_star\)", src):
            fwd = {"model": "AD", "params": {"U": ctx.get("U"), "K": "K_star(ad)", "D": "D_star(ad)"}}
        elif re.search(r"partitioned_model_nonlinear\(A_crit,\s*D0,\s*Sc\)", src):
            a = parse_assignments(src, ("A_crit",)).get("A_crit")
            fwd = {"model": "partitioned_nonlinear",
                   "params": {"A_crit": safe_eval(a, ctx) if a else None, "D": ctx.get("D0"),
                              "S_c": ctx.get("Sc"), "K": ctx.get("K0"), "U": ctx.get("U")}}
        elif re.search(r"partitioned_model_nonlinear\(A_crit_star,\s*D0,\s*Sc\)", src):
            fwd = {"model": "partitioned_nonlinear",
                   "params": {"A_crit": "A_crit_star(nonlinear)", "D": ctx.get("D0"),
                              "S_c": ctx.get("Sc"), "K": ctx.get("K0"), "U": ctx.get("U")}}
        elif re.search(r"partitioned_model_all\(A_crit_star,\s*D_star\)", src):
            fwd = {"model": "partitioned",
                   "params": {"A_crit": "A_crit_star(joint)", "D": "D_star(joint)", "K": "K_star(joint)",
                              "U": ctx.get("U")}}
        elif re.search(r"partitioned_model\(A_crit_star,\s*D0\)", src):
            fwd = {"model": "partitioned",
                   "params": {"A_crit": "A_crit_star(acrit)", "D": ctx.get("D0"), "K": ctx.get("K0"),
                              "U": ctx.get("U")}}
        elif re.search(r"partitioned_model\(A_crit,\s*D0\)", src):
            a = parse_assignments(src, ("A_crit",)).get("A_crit")
            fwd = {"model": "partitioned",
                   "params": {"A_crit": safe_eval(a, ctx) if a else None, "D": ctx.get("D0"),
                              "K": ctx.get("K0"), "U": ctx.get("U")}}
        if fwd is not None:
            fwd["cell"] = i
            fwd["duration_s"] = durations.get(str(i))
            msave = re.search(r'save_model_outputs\([^,]+,\s*[^,]+,\s*base_save,\s*"([^"]+)"', src)
            fwd["saved_prefix"] = msave.group(1) if msave else None
            fwd["saved"] = bool(msave and ctx.get("SAVE") is True)
            fwd["printed_output"] = out.strip() or None
            forward_runs.append(fwd)

        # model factory constructions (which K0 each partitioned model was built with)
        if re.search(r"=\s*partitioned_threshold_model\(", src) and "optimize" not in src:
            misc_outputs.setdefault("partitioned_model_builds", []).append(
                {"cell": i, "K_used": ctx.get("K0"), "line": [l.strip() for l in src.splitlines()
                                                              if "partitioned_threshold_model" in l]})
        if re.search(r"=\s*partitioned_threshold_model_nonlinear\(", src):
            misc_outputs.setdefault("partitioned_nonlinear_model_builds", []).append(
                {"cell": i, "K_used": ctx.get("K0"), "S_c": ctx.get("Sc")})

        # histogram outputs (slope distributions), kept raw
        if "plt.hist" in src and out.strip():
            misc_outputs.setdefault("histograms_raw", []).append({"cell": i, "source": src, "output": out})

    for s in stages.values():
        s.pop("raw_output_dup", None)

    total = sum(durations.values())
    rec = {
        "notebook": str(path.relative_to(ROOT)),
        "site": site,
        "tile": tile,
        "papermill_total_duration_s": round(total, 1),
        "papermill_total_duration_h": round(total / 3600.0, 2),
        "config": {
            "concavity_theta": ctx.get("concavity"),
            "hilltop_curvature": ctx.get("hilltop_curvature"),
            "steepness_ks": ctx.get("steepness"),
            "U": ctx.get("U"),
            "n": ctx.get("n"),
            "m": ctx.get("m"),
            "S_c": ctx.get("Sc"),
            "K0_from_ks": (ctx.get("U") / ctx.get("steepness")) ** (1.0 / ctx.get("n"))
            if all(isinstance(ctx.get(k), (int, float)) for k in ("U", "steepness", "n")) else None,
            "D0_from_curvature": d0_value,
            "K0_history": k0_history,
            "K_bounds": list(ctx.get("K_bounds")) if isinstance(ctx.get("K_bounds"), tuple) else ctx.get("K_bounds"),
            "D_bounds": list(ctx.get("D_bounds")) if isinstance(ctx.get("D_bounds"), tuple) else ctx.get("D_bounds"),
            "A_crit_0": ctx.get("A_crit_0"),
            "A_crit_bounds": list(ctx.get("A_crit_bounds")) if isinstance(ctx.get("A_crit_bounds"), tuple) else ctx.get("A_crit_bounds"),
            "SAVE": ctx.get("SAVE"),
            "data_dir": ctx.get("base"),
            "config_cell_source": config_cell_source,
        },
        "stages": stages,
        "forward_runs": forward_runs,
        "cell_durations_s": durations,
        "misc": misc_outputs,
    }
    return rec


# ----------------------------------------------------------------------------
# Figure notebooks
# ----------------------------------------------------------------------------

def parse_fig_slope_area(text: str) -> dict:
    out = {"tiles": {}}
    current = None
    for line in text.splitlines():
        m = re.match(r"^Tile:\s*(\w+)\s+tile(\d)", line)
        if m:
            current = f"{m.group(1)}{m.group(2)}"
            out["tiles"].setdefault(current, {})
            continue
        m = re.match(r"^\s{2}(ORIGINAL|AD no-opt|AD opt|PARTITIONED)\s*:\s*Ks = " + NUM + r" m, θ = " + NUM + r", R² = " + NUM, line)
        if m and current:
            out["tiles"][current][m.group(1)] = {"Ks": float(m.group(2)), "theta": float(m.group(3)), "R2": float(m.group(4))}
    return out


def parse_ok_lines(text: str) -> list:
    rows = []
    for line in text.splitlines():
        if not line.startswith("[OK]"):
            continue
        m = re.match(r"\[OK\]\s+(\w+)\s+T(\d)\s+dx=\s*" + NUM, line)
        if not m:
            continue
        row = {"site": m.group(1), "tile": int(m.group(2)), "dx": float(m.group(3))}
        for key in ("Ks_mod", "Ks_obs", "ratio", "ratio_d8", "ratio_dinf", "n_mod", "n_obs"):
            mm = re.search(rf"{key}=\s*{NUM}", line)
            if mm:
                v = float(mm.group(1))
                row[key] = int(v) if key.startswith("n_") else v
        rows.append(row)
    return rows


def parse_d8_dinf(text: str) -> dict:
    out = {"locations": {}}
    cur = None
    for line in text.splitlines():
        m = re.match(r"^=== (.+?) ===", line)
        if m:
            cur = m.group(1)
            out["locations"][cur] = {}
            continue
        if cur is None:
            continue
        m = re.search(r"DEM shape: \((\d+), (\d+)\)", line)
        if m:
            out["locations"][cur]["shape"] = [int(m.group(1)), int(m.group(2))]
        m = re.search(r"D-inf area range: \[" + NUM + ", " + NUM + r"\]", line)
        if m:
            out["locations"][cur]["A_dinf_range"] = [float(m.group(1)), float(m.group(2))]
        m = re.search(r"D8 area range:\s+\[" + NUM + ", " + NUM + r"\]", line)
        if m:
            out["locations"][cur]["A_d8_range"] = [float(m.group(1)), float(m.group(2))]
        m = re.search(r"MSE \(AD\):\s+D8=" + NUM + r"\s+D-inf=" + NUM, line)
        if m:
            out["locations"][cur]["MSE_AD_D8"] = float(m.group(1))
            out["locations"][cur]["MSE_AD_Dinf"] = float(m.group(2))
        m = re.search(r"MSE \(PT\):\s+D8=" + NUM + r"\s+D-inf=" + NUM, line)
        if m:
            out["locations"][cur]["MSE_PT_D8"] = float(m.group(1))
            out["locations"][cur]["MSE_PT_Dinf"] = float(m.group(2))
    return out


def parse_supp_fig5(text: str) -> dict:
    out = {"primary": [], "trim_bookkeeping": [], "summary": []}
    for line in text.splitlines():
        if line.startswith("[primary]"):
            out["primary"].append(line.strip())
        m = re.match(r"^dx=" + NUM + r"\s+Ny=(\d+)\s+trim=" + NUM + r"\s+j=(\d+):(\d+)\s+n_used=(\d+)", line)
        if m:
            out["trim_bookkeeping"].append({"dx": float(m.group(1)), "Ny": int(m.group(2)), "trim": float(m.group(3)),
                                            "j0": int(m.group(4)), "j1": int(m.group(5)), "n_used": int(m.group(6))})
        m = re.match(r"^dx=" + NUM + r"\s+Ny=(\d+)\s+n_good=(\d+)\s+median ks/\(U/K\)=" + NUM + r"\s+IQR=\[" + NUM + r",\s*" + NUM + r"\]", line)
        if m:
            out["summary"].append({"dx": float(m.group(1)), "Ny": int(m.group(2)), "n_good": int(m.group(3)),
                                   "median_ks_over_UK": float(m.group(4)), "q25": float(m.group(5)), "q75": float(m.group(6))})
    return out


def extract_figure_notebook(path: Path) -> dict:
    nb = json.loads(path.read_text())
    cells = nb["cells"]
    all_text = []
    config_sources = []
    per_cell = []
    for i, c in enumerate(cells):
        if c["cell_type"] != "code":
            continue
        src = cell_source(c)
        out = cell_streams(c)
        if out.strip():
            per_cell.append({"cell": i, "output": out})
            all_text.append(out)
        if re.search(r"^(CONFIGS|LOC_CONFIGS|KD_GM|KD_AP|THETA_GM|THETA_AP|U_BY_LOC|DX_TARGETS|N_DX|MIN_AREA|Lx, Ly|U|K|m|L_PE|PE_BAL|A_INLET_BAL_ANCHOR|D_BAL_ANCHOR|dx_targets|FRAC_TRIM)\s*=", src, re.M):
            config_sources.append({"cell": i, "source": src})
    text = "\n".join(all_text)
    rec = {"notebook": str(path.relative_to(ROOT)), "config_cells": config_sources, "printed_outputs_raw": per_cell}
    stem = path.stem
    if stem == "fig_slope_area":
        rec["parsed"] = parse_fig_slope_area(text)
    elif stem == "supp_fig6":
        rec["parsed"] = {"ok_rows": parse_ok_lines(text)}
        # any other printed summary lines (exponents etc.)
        rec["parsed"]["other_lines"] = [l for l in text.splitlines() if l.strip() and not l.startswith("[OK]")]
    elif stem == "resolution_study_dinf":
        rec["parsed"] = {"ok_rows": parse_ok_lines(text)}
        rec["parsed"]["other_lines"] = [l for l in text.splitlines() if l.strip() and not l.startswith("[OK]")]
    elif stem == "d8_vs_dinf_comparison":
        rec["parsed"] = parse_d8_dinf(text)
    elif stem == "supp_fig5":
        rec["parsed"] = parse_supp_fig5(text)
    return rec


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> int:
    tiles = {}
    for p in sorted(MODEL_RUNS.glob("*_complete.ipynb")):
        rec = extract_model_run(p)
        tiles[f"{rec['site']}{rec['tile']}"] = rec
    figs = {}
    for p in sorted(FIGURES.glob("*.ipynb")):
        figs[p.stem] = extract_figure_notebook(p)

    result = {
        "generated": {
            "script": "reference/extract_v1_reference.py",
            "git_hash": git_hash(),
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": sys.version.split()[0],
            "note": "Values are parsed from notebook text as printed (4 significant figures for optimizer "
                    "results). Forward-run MSEs are not printed by the v1 model-run notebooks; see "
                    "reference/v1_saved_grid_mse.csv for MSEs recomputed from the saved grids.",
        },
        "tiles": tiles,
        "figure_notebooks": figs,
    }
    OUT.write_text(json.dumps(result, indent=1, default=str))

    # Console summary for checking
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"{'tile':<5}{'theta':>7}{'ks':>9}{'C_ht':>10}{'K0_AD':>11}{'D0':>11} | "
          f"{'AD K':>10}{'AD D':>10}{'AD MSE':>10} | {'Ac2':>9}{'MSE2':>9} | {'AcNL':>9}{'MSENL':>9} | "
          f"{'AcJ':>9}{'DJ':>10}{'KJ':>10}{'MSEJ':>9} | hours")
    for k, t in tiles.items():
        c, s = t["config"], t["stages"]
        g = lambda st, key: s.get(st, {}).get(key, float("nan"))  # noqa: E731
        print(f"{k:<5}{c['concavity_theta']:>7}{c['steepness_ks']:>9}{c['hilltop_curvature']:>10.3e}"
              f"{s['ad']['K0_at_call']:>11.3e}{c['D0_from_curvature']:>11.3e} | "
              f"{g('ad','K'):>10.3e}{g('ad','D'):>10.3e}{g('ad','MSE'):>10.3e} | "
              f"{g('acrit','A_crit'):>9.3e}{g('acrit','MSE'):>9.3e} | "
              f"{g('nonlinear','A_crit'):>9.3e}{g('nonlinear','MSE'):>9.3e} | "
              f"{g('joint','A_crit'):>9.3e}{g('joint','D'):>10.3e}{g('joint','K'):>10.3e}{g('joint','MSE'):>9.3e} | "
              f"{t['papermill_total_duration_h']}")
    print("\nstage durations (h): tile  ad  acrit  nonlinear  joint")
    for k, t in tiles.items():
        s = t["stages"]
        print(f"  {k:<5}" + "  ".join(f"{(s.get(st, {}).get('duration_s') or 0)/3600:6.2f}" for st in ("ad", "acrit", "nonlinear", "joint")))
    for k, f in figs.items():
        p = f.get("parsed", {})
        n = len(p.get("ok_rows", [])) if "ok_rows" in p else len(p.get("tiles", p.get("locations", p.get("summary", []))))
        print(f"figure notebook {k}: parsed {n} records, {len(f['printed_outputs_raw'])} cells with printed output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
