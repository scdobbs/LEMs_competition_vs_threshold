"""Table 1: measured inputs and fitted parameters per tile, from the Phase 6 per-stage runs (metres, site m).

Phase 6 wrote one ``tile_run`` per stage, so the table reads (latest of each, uncropped, z in metres, v2 optimizer):

* measured: the tile config's ``measured`` block (site m, k_s at theta = m above 1e3 m^2, C_ht at L_f and L_w,
  K0 = U/k_s, D0_A = U/C_ht, D0_B = U/C_ht_B), sourced from its measured-inputs run and checked against the
  model block every stage run used;
* AD: the v2 r-profile run (``*_v2_ad``, D0 variant A), ``v2.ad.optimum``;
* stage 2 and linear joint: the refined partitioned run (``*_v2_partitioned-refined``), ``v2.partitioned``
  (stage-2 optimum at D0_A and D0_B, joint optimum and its 1 % A_crit interval);
* nonlinear stage: the ``v2.nonlinear`` runs at S_c = 1.27 (D0 variant A; variant B on AP in the long CSV);
* nonlinear joint: the merged (A_crit, D) profile at S_c = 1.27, K fixed to the linear joint K, K factor 1, no S_c
  override (as ``lemcvt.analyses.nonlinear_fair._phase6_nonlinear_joint_optimum``).

Both nonlinear stages solve the v1 centre form D_eff lap(z) = -U; addendum 10, 1.1 requires every table that shows
them to carry the label :data:`NL_LABEL`. Outputs: ``table1.csv`` (wide, one ``run_<group>`` column per quantity
group), ``table1_long.csv`` (one row per value with unit and source run) and ``table1.tex`` (booktabs + siunitx).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from lemcvt.config import TILE_IDS

logger = logging.getLogger(__name__)

NL_LABEL = "centre form, D_eff∇²z = −U (nonstandard)"
NL_LABEL_TEX = r"centre form, $D_\mathrm{eff}\nabla^2 z = -U$ (nonstandard)"
PHASE6_S_C = 1.27
U_DEFAULT = 5.5e-5

# (column, group, quantity, unit, LaTeX header, siunitx format or None for text, in the LaTeX table)
COLUMNS: list[tuple[str, str, str, str, str, str | None, bool]] = [
    ("tile", "", "tile", "", "Tile", None, True),
    ("m", "measured", "m", "-", r"{$m$}", "S[table-format=1.2]", True),
    ("ks", "measured", "k_s", "m^(2m)", r"{$k_s$}", "S[table-format=2.2]", True),
    ("C_ht", "measured", "C_ht (L_f)", "1/m", r"{$C_{ht}$}", "S[table-format=1.2e-1]", True),
    ("C_ht_B", "measured", "C_ht_B (L_w)", "1/m", r"{$C_{ht,B}$}", "S[table-format=1.2e-1]", False),
    ("K0", "measured", "K0 = U/k_s", "1/yr", r"{$K_0$}", "S[table-format=1.2e-1]", True),
    ("D0_A", "measured", "D0_A = U/C_ht", "m^2/yr", r"{$D_{0,A}$}", "S[table-format=1.2e-1]", True),
    ("D0_B", "measured", "D0_B = U/C_ht_B", "m^2/yr", r"{$D_{0,B}$}", "S[table-format=1.2e-1]", True),
    ("ad_K", "ad", "AD K", "1/yr", r"{$K$}", "S[table-format=1.2e-1]", True),
    ("ad_D", "ad", "AD D", "m^2/yr", r"{$D$}", "S[table-format=1.2e-1]", True),
    ("ad_mse", "ad", "AD MSE", "m^2", r"{MSE}", "S[table-format=3.1]", True),
    ("s2_A_crit", "partitioned", "stage 2 A_crit at D0_A", "m^2", r"{$A_\mathrm{crit}$}", "S[table-format=3.0]", True),
    ("s2_mse", "partitioned", "stage 2 MSE at D0_A", "m^2", r"{MSE}", "S[table-format=3.1]", True),
    ("s2B_A_crit", "partitioned", "stage 2 A_crit at D0_B", "m^2", "", None, False),
    ("s2B_mse", "partitioned", "stage 2 MSE at D0_B", "m^2", "", None, False),
    ("joint_A_crit", "partitioned", "linear joint A_crit", "m^2", r"{$A_\mathrm{crit}$}", "S[table-format=4.0]", True),
    ("joint_A_lo1", "partitioned", "linear joint A_crit 1 % interval, min", "m^2", r"{1\,\% min}", "S[table-format=4.0]", True),
    ("joint_A_hi1", "partitioned", "linear joint A_crit 1 % interval, max", "m^2", r"{1\,\% max}", "S[table-format=4.0]", True),
    ("joint_n_int1", "partitioned", "linear joint 1 % interval, number of pieces", "-", "", None, False),
    ("joint_K", "partitioned", "linear joint K", "1/yr", r"{$K$}", "S[table-format=1.2e-1]", True),
    ("joint_D", "partitioned", "linear joint D", "m^2/yr", r"{$D$}", "S[table-format=1.2e-1]", True),
    ("joint_mse", "partitioned", "linear joint MSE", "m^2", r"{MSE}", "S[table-format=2.1]", True),
    ("nl_A_crit", "nonlinear", "nonlinear stage A_crit at D0_A", "m^2", r"{$A_\mathrm{crit}$}", "S[table-format=3.0]", True),
    ("nl_D", "nonlinear", "nonlinear stage D (= D0_A)", "m^2/yr", r"{$D$}", "S[table-format=1.2e-1]", True),
    ("nl_mse", "nonlinear", "nonlinear stage MSE at D0_A", "m^2", r"{MSE}", "S[table-format=3.1]", True),
    ("nl_S_c", "nonlinear", "nonlinear stage S_c", "-", "", None, False),
    ("nlB_A_crit", "nonlinear_B", "nonlinear stage A_crit at D0_B", "m^2", "", None, False),
    ("nlB_D", "nonlinear_B", "nonlinear stage D (= D0_B)", "m^2/yr", "", None, False),
    ("nlB_mse", "nonlinear_B", "nonlinear stage MSE at D0_B", "m^2", "", None, False),
    ("nlj_A_crit", "nonlinear_joint", "nonlinear joint A_crit", "m^2", r"{$A_\mathrm{crit}$}", "S[table-format=4.0]", True),
    ("nlj_K", "nonlinear_joint", "nonlinear joint K (fixed, linear joint K)", "1/yr", "", None, False),
    ("nlj_D", "nonlinear_joint", "nonlinear joint D", "m^2/yr", r"{$D$}", "S[table-format=1.2e-1]", True),
    ("nlj_mse", "nonlinear_joint", "nonlinear joint MSE", "m^2", r"{MSE}", "S[table-format=2.1]", True),
    ("nlj_S_c", "nonlinear_joint", "nonlinear joint S_c", "-", "", None, False),
]
GROUPS = ("measured", "ad", "partitioned", "nonlinear", "nonlinear_B", "nonlinear_joint")
NONLINEAR_GROUPS = ("nonlinear", "nonlinear_B", "nonlinear_joint")
TEX_GROUP_TITLES = {
    "measured": "Measured",
    "ad": "Advection-diffusion",
    "s2": r"Stage 2 at $D_{0,A}$",
    "joint": "Linear joint",
    "nonlinear": "Nonlinear stage at $D_{0,A}$, $S_c = 1.27$",
    "nonlinear_joint": "Nonlinear joint, $S_c = 1.27$",
}


# ------------------------------------------------------------------------------------------------ run selection
def _is_phase6_base(p: Mapping[str, Any], rj: Mapping[str, Any]) -> bool:
    return (p.get("optimizer") == "v2" and not (rj.get("config") or {}).get("crop")
            and (p.get("inputs") or {}).get("z_units_working") == "m")


def _d0_variant(p: Mapping[str, Any]) -> str:
    return (p.get("model") or {}).get("d0_variant") or "A"


def select_phase6_runs(records: Iterable[tuple[str, Mapping[str, Any], Mapping[str, Any]]]) -> dict[str, tuple[str, Mapping[str, Any]]]:
    """Pick the Phase 6 run of each group from ``(run name, params, run.json)`` records of one tile.

    Records are taken in name (time) order and the latest qualifying run of each group wins.
    """
    out: dict[str, tuple[str, Mapping[str, Any]]] = {}
    for name, p, rj in sorted(records, key=lambda x: x[0]):
        if not _is_phase6_base(p, rj):
            continue
        v2 = p.get("v2") or {}
        S_c = float((p.get("model") or {}).get("S_c", np.nan))
        if "ad" in v2 and _d0_variant(p) == "A":
            out["ad"] = (name, p)
        if "partitioned" in v2 and name.endswith("_v2_partitioned-refined"):
            out["partitioned"] = (name, p)
        if "nonlinear" in v2 and np.isclose(S_c, PHASE6_S_C):
            out["nonlinear" if _d0_variant(p) == "A" else "nonlinear_B"] = (name, p)
        nj = v2.get("nonlinear_joint")
        if nj:
            st = nj.get("setup") or {}
            if ("split" not in st and float(st.get("k_factor", 1.0)) == 1.0 and st.get("S_c_override") is None
                    and np.isclose(S_c, PHASE6_S_C)):
                out["nonlinear_joint"] = (name, p)
    return out


def _read_records(tile: str, root: Path | None = None) -> list[tuple[str, dict, dict]]:
    from lemcvt.results import find_runs

    recs = []
    for r in find_runs("tile_run", tile, root=root):
        try:
            recs.append((r.name, json.loads((r / "params.json").read_text()), json.loads((r / "run.json").read_text())))
        except Exception:  # noqa: BLE001
            continue
    return recs


# ------------------------------------------------------------------------------------------------ row assembly
def _f(x: Any) -> float:
    return float(x) if x is not None else float("nan")


def measured_values(measured: Mapping[str, Any], U: float = U_DEFAULT) -> dict[str, float]:
    ks = _f(measured.get("ks"))
    C = _f(measured.get("C_ht"))
    CB = _f(measured.get("C_ht_B"))
    return {"m": _f(measured.get("m")), "ks": ks, "C_ht": C, "C_ht_B": CB,
            "K0": _f(measured.get("K0", U / ks if ks else None)), "D0_A": _f(measured.get("D0", U / C if C else None)),
            "D0_B": _f(measured.get("D0_B", U / CB if CB else None))}


def measured_run(measured: Mapping[str, Any]) -> str:
    prov = measured.get("provenance") or {}
    for k in ("ks", "C_ht", "m"):
        r = (prov.get(k) or {}).get("run")
        if r:
            return str(r)
    return ""


def tile_row(tile: str, measured: Mapping[str, Any], runs: Mapping[str, tuple[str, Mapping[str, Any]]]) -> tuple[dict[str, Any], list[str]]:
    """One wide row of Table 1 and the list of consistency problems found (empty when the runs agree)."""
    row: dict[str, Any] = {"tile": tile}
    problems: list[str] = []
    U = U_DEFAULT
    any_p = next(iter(runs.values()))[1] if runs else None
    if any_p is not None:
        U = float((any_p.get("model") or {}).get("U", U_DEFAULT))
    meas = measured_values(measured, U)
    row.update(meas)
    row["run_measured"] = measured_run(measured)
    for g in GROUPS[1:]:
        row[f"run_{g}"] = runs[g][0] if g in runs else ""
    # the measured inputs every stage run used must equal the config's measured block
    for g, (name, p) in runs.items():
        mdl = p.get("model") or {}
        checks = [("m", "m"), ("K0_partitioned", "K0")]
        if g not in ("nonlinear_B",) and mdl.get("D0") is not None:
            checks.append(("D0", "D0_A"))
        if g == "nonlinear_B":
            checks.append(("D0", "D0_B"))
        for mk, rk in checks:
            if mdl.get(mk) is None or not np.isfinite(meas[rk]):
                continue
            if not np.isclose(float(mdl[mk]), meas[rk], rtol=1e-9):
                problems.append(f"{tile} {g} run {name}: model.{mk} = {mdl[mk]} differs from measured {rk} = {meas[rk]}")
    if "ad" in runs:
        o = runs["ad"][1]["v2"]["ad"]["optimum"]
        row.update({"ad_K": _f(o["K"]), "ad_D": _f(o["D"]), "ad_mse": _f(o["mse"])})
    if "partitioned" in runs:
        v = runs["partitioned"][1]["v2"]["partitioned"]
        s2, jo = v["stage2_optimum"], v["joint_optimum"]
        lev = (v.get("joint_identifiability") or {}).get("level_0.01") or {}
        row.update({"s2_A_crit": _f(s2["A_crit"]), "s2_mse": _f(s2["mse"]),
                    "joint_A_crit": _f(jo["A_crit"]), "joint_K": _f(jo["K"]), "joint_D": _f(jo["D"]), "joint_mse": _f(jo["mse"]),
                    "joint_A_lo1": _f(lev.get("A_crit_min")), "joint_A_hi1": _f(lev.get("A_crit_max")),
                    "joint_n_int1": _f(lev.get("n_intervals", len(lev.get("intervals") or []) or None))})
        sB = v.get("stage2_B_optimum")
        if sB:
            row.update({"s2B_A_crit": _f(sB["A_crit"]), "s2B_mse": _f(sB["mse"])})
    for g, pre in (("nonlinear", "nl"), ("nonlinear_B", "nlB")):
        if g in runs:
            p = runs[g][1]
            o = p["v2"]["nonlinear"]["optimum"]
            row.update({f"{pre}_A_crit": _f(o["A_crit"]), f"{pre}_D": _f(p["model"]["D0"]), f"{pre}_mse": _f(o["mse"])})
            if pre == "nl":
                row["nl_S_c"] = _f(p["model"]["S_c"])
    if "nonlinear_joint" in runs:
        p = runs["nonlinear_joint"][1]
        v = p["v2"]["nonlinear_joint"]
        o = v["optimum"]
        row.update({"nlj_A_crit": _f(o["A_crit"]), "nlj_D": _f(o["D"]), "nlj_mse": _f(o["mse"]),
                    "nlj_K": _f(v["setup"].get("K", o.get("K"))), "nlj_S_c": _f(p["model"]["S_c"])})
        if "partitioned" in runs and np.isfinite(row.get("joint_K", np.nan)) and not np.isclose(row["nlj_K"], row["joint_K"], rtol=1e-9):
            problems.append(f"{tile}: nonlinear joint K {row['nlj_K']} is not the linear joint K {row['joint_K']}")
    for g in NONLINEAR_GROUPS:
        row[f"label_{g}"] = NL_LABEL if g in runs else ""
    return row, problems


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per value: tile, group, column, quantity, unit, value, source run, and the model-form label."""
    rows = []
    for _, r in df.iterrows():
        for col, group, qty, unit, _h, _fmt, _tex in COLUMNS[1:]:
            v = r.get(col, np.nan)
            if v is None or (isinstance(v, float) and not np.isfinite(v)):
                continue
            rows.append({"tile": r["tile"], "group": group, "column": col, "quantity": qty, "unit": unit, "value": float(v),
                         "run": r.get(f"run_{group}", ""), "model_form": NL_LABEL if group in NONLINEAR_GROUPS else ""})
    return pd.DataFrame(rows, columns=["tile", "group", "column", "quantity", "unit", "value", "run", "model_form"])


def collect(root: Path | None = None) -> tuple[pd.DataFrame, list[str]]:
    from lemcvt.config import load_tile_config

    rows, problems = [], []
    for tile in TILE_IDS:
        runs = select_phase6_runs(_read_records(tile, root))
        if not runs:
            logger.warning("table1: %s has no Phase 6 per-stage runs", tile)
            continue
        cfg = load_tile_config(tile)
        row, prob = tile_row(tile, cfg.measured, runs)
        missing = [g for g in GROUPS[1:] if g not in runs and not (g == "nonlinear_B" and tile.startswith("GM"))]
        if missing:
            prob.append(f"{tile}: no run for {', '.join(missing)}")
        rows.append(row)
        problems += prob
    if not rows:
        raise FileNotFoundError("no Phase 6 per-stage tile_run results (AD, refined partitioned, nonlinear, merged nonlinear joint)")
    df = pd.DataFrame(rows)
    order = ["tile"] + [c[0] for c in COLUMNS[1:] if c[0] in df.columns]
    rest = [c for c in df.columns if c not in order]
    return df[order + rest], problems


# ------------------------------------------------------------------------------------------------ LaTeX
def _fmt(v: Any, fmt: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "{--}"
    spec = fmt.split("table-format=")[1].rstrip("]")
    if "e" in spec:
        digits = int(spec.split(".")[1].split("e")[0])
        return f"{float(v):.{digits}e}"
    digits = int(spec.split(".")[1]) if "." in spec else 0
    return f"{float(v):.{digits}f}"


def tex_columns() -> list[tuple[str, str, str, str]]:
    """(column, group, header, siunitx spec) of the LaTeX table, in order."""
    out = []
    for col, group, _q, _u, head, fmt, in_tex in COLUMNS:
        if not in_tex:
            continue
        g = group
        if col.startswith("s2_"):
            g = "s2"
        elif col.startswith("joint_"):
            g = "joint"
        out.append((col, g, head, fmt or "l"))
    return out


def to_latex(df: pd.DataFrame) -> str:
    cols = tex_columns()
    spec = "".join(c[3] for c in cols)
    groups: list[tuple[str, int, int]] = []   # (group, first index 1-based, count)
    for i, (_c, g, _h, _s) in enumerate(cols, start=1):
        if groups and groups[-1][0] == g:
            groups[-1] = (g, groups[-1][1], groups[-1][2] + 1)
        else:
            groups.append((g, i, 1))
    head1, rules = [], []
    for g, first, n in groups:
        if not g:
            head1.append("")
            continue
        title = TEX_GROUP_TITLES.get(g, g)
        if g in NONLINEAR_GROUPS:
            title += r"\tnote{a}"
        head1.append(rf"\multicolumn{{{n}}}{{c}}{{{title}}}")
        rules.append(rf"\cmidrule(lr){{{first}-{first + n - 1}}}")
    lines = [
        r"% Table 1, generated by `lemcvt tables table1` from the Phase 6 per-stage runs (source run ids in table1.csv / table1_long.csv).",
        r"% Needs \usepackage{booktabs,siunitx,threeparttable}.",
        r"\begin{threeparttable}",
        r"\begin{tabular}{" + spec + "}",
        r"\toprule",
        " & ".join(head1) + r" \\",
        " ".join(rules),
        " & ".join(c[2] for c in cols) + r" \\",
        r"\midrule",
    ]
    for _, r in df.iterrows():
        cells = []
        for col, _g, _h, s in cols:
            cells.append(str(r[col]) if col == "tile" else _fmt(r.get(col, np.nan), s))
        lines.append(" & ".join(cells) + r" \\")
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}",
        r"\item Units: $k_s$ in m$^{2m}$, $C_{ht}$ in m$^{-1}$, $K$ in yr$^{-1}$, $D$ in m$^2$\,yr$^{-1}$, $A_\mathrm{crit}$ and MSE in m$^2$. "
        r"$K_0 = U/k_s$ with $k_s$ at $\theta = m$ above $10^3$ m$^2$; $D_{0,A} = U/C_{ht}(L_f)$, $D_{0,B} = U/C_{ht}(L_w)$ (on AP $D_0$ is the bracket $[D_{0,A}, D_{0,B}]$). "
        r"The 1\,\% interval is the range of $A_\mathrm{crit}$ within 1\,\% of the linear joint minimum (it can have gaps).",
        r"\item[a] " + NL_LABEL_TEX + r": the v1 nonlinear model; the nonlinear joint has $K$ fixed to the linear joint $K$.",
        r"\end{tablenotes}",
        r"\end{threeparttable}",
    ]
    return "\n".join(lines) + "\n"


def build(out_dir: Path) -> list[Path]:
    df, problems = collect()
    for msg in problems:
        logger.warning("table1: %s", msg)
    out_dir.mkdir(parents=True, exist_ok=True)
    p_csv = out_dir / "table1.csv"
    df.to_csv(p_csv, index=False)
    p_long = out_dir / "table1_long.csv"
    to_long(df).to_csv(p_long, index=False)
    p_tex = out_dir / "table1.tex"
    p_tex.write_text(to_latex(df))
    p_chk = out_dir / "table1_checks.txt"
    p_chk.write_text("\n".join(problems) + "\n" if problems else "no inconsistencies between the measured block and the stage runs\n")
    return [p_csv, p_long, p_tex, p_chk]
