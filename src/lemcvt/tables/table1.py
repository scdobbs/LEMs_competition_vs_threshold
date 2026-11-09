"""Table 1: measured and optimized parameters per tile and model (from tile_run results).

Columns: tile, theta, k_s, C_ht, K0, D0, AD K, D, MSE, A_crit-only A_crit, MSE,
nonlinear A_crit, MSE, joint A_crit, D, K, MSE. The LaTeX version uses booktabs and
siunitx ``S`` columns; the CSV carries full precision. A second CSV compares each
number with the v1 printed reference recorded in the tile config.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lemcvt.config import TILE_IDS, load_tile_config
from lemcvt.results import latest_run

COLUMNS = [
    ("tile", "Tile", None),
    ("theta", r"$\theta$", "S[table-format=1.3]"),
    ("ks", r"$k_s$", "S[table-format=3.3]"),
    ("C_ht", r"$C_{ht}$ (m$^{-1}$)", "S[table-format=1.2e-2]"),
    ("K0", r"$K_0$", "S[table-format=1.2e-2]"),
    ("D0", r"$D_0$", "S[table-format=1.2e-2]"),
    ("ad_K", r"$K_{AD}$", "S[table-format=1.2e-2]"),
    ("ad_D", r"$D_{AD}$", "S[table-format=1.2e-2]"),
    ("ad_mse", r"MSE$_{AD}$", "S[table-format=1.2e2]"),
    ("acrit_A", r"$A_{crit}$", "S[table-format=1.2e3]"),
    ("acrit_mse", "MSE", "S[table-format=1.2e2]"),
    ("nl_A", r"$A_{crit}^{NL}$", "S[table-format=1.2e3]"),
    ("nl_mse", "MSE", "S[table-format=1.2e2]"),
    ("joint_A", r"$A_{crit}^{J}$", "S[table-format=1.2e3]"),
    ("joint_D", r"$D^{J}$", "S[table-format=1.2e-2]"),
    ("joint_K", r"$K^{J}$", "S[table-format=1.2e-2]"),
    ("joint_mse", r"MSE$^{J}$", "S[table-format=1.2e2]"),
]


def collect() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, cmp_rows = [], []
    for tile in TILE_IDS:
        run = latest_run("tile_run", tile)
        cfg = load_tile_config(tile)
        if run is None:
            continue
        p = json.loads((run / "params.json").read_text())
        st = p.get("stages_results", {})
        mv = cfg.measured_v1
        row = {"tile": tile, "theta": mv["theta"], "ks": mv["ks"], "C_ht": mv["hilltop_curvature"],
               "K0": p["model"]["K0_partitioned"], "D0": p["model"]["D0"], "run": run.name,
               "z_units": p["inputs"]["z_units_working"]}
        g = lambda s, k: st.get(s, {}).get("params", {}).get(k, np.nan)  # noqa: E731
        row.update({"ad_K": g("ad", "K"), "ad_D": g("ad", "D"), "ad_mse": st.get("ad", {}).get("mse", np.nan),
                    "acrit_A": g("acrit", "A_crit"), "acrit_mse": st.get("acrit", {}).get("mse", np.nan),
                    "nl_A": g("nonlinear", "A_crit"), "nl_mse": st.get("nonlinear", {}).get("mse", np.nan),
                    "joint_A": g("joint", "A_crit"), "joint_D": g("joint", "D"), "joint_K": g("joint", "K"),
                    "joint_mse": st.get("joint", {}).get("mse", np.nan)})
        rows.append(row)
        ref = cfg.v1_reference.get("stages", {})
        for stage, keys in (("ad", ("K", "D", "MSE")), ("acrit", ("A_crit", "MSE")), ("nonlinear", ("A_crit", "MSE")),
                            ("joint", ("A_crit", "D", "K", "MSE"))):
            if stage not in st or stage not in ref:
                continue
            for k in keys:
                new = st[stage]["mse"] if k == "MSE" else st[stage]["params"][k]
                old = ref[stage].get(k)
                if old is None:
                    continue
                cmp_rows.append({"tile": tile, "stage": stage, "quantity": k, "v1": old, "v2": new,
                                 "rel_diff": (new - old) / old if old else np.nan})
    if not rows:
        raise FileNotFoundError("no tile_run results found")
    return pd.DataFrame(rows), pd.DataFrame(cmp_rows)


def to_latex(df: pd.DataFrame) -> str:
    cols = [c for c in COLUMNS if c[0] in df.columns]
    spec = "l" + "".join(c[2] or "l" for c in cols[1:])
    lines = [r"\begin{tabular}{" + spec + "}", r"\toprule", " & ".join(f"{{{c[1]}}}" for c in cols) + r" \\", r"\midrule"]
    for _, r in df.iterrows():
        cells = []
        for key, _label, fmt in cols:
            v = r[key]
            if key == "tile":
                cells.append(str(v))
            elif isinstance(v, float) and not np.isfinite(v):
                cells.append("{--}")
            elif fmt and "e" in fmt:
                cells.append(f"{v:.2e}")
            elif key == "theta":
                cells.append(f"{v:.3f}")
            else:
                cells.append(f"{v:.3f}" if isinstance(v, float) else str(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def build(out_dir: Path) -> list[Path]:
    df, cmp_df = collect()
    paths = []
    p_csv = out_dir / "table1.csv"
    df.to_csv(p_csv, index=False)
    p_tex = out_dir / "table1.tex"
    p_tex.write_text(to_latex(df))
    paths += [p_csv, p_tex]
    if len(cmp_df):
        p_cmp = out_dir / "table1_v1_vs_v2.csv"
        cmp_df.to_csv(p_cmp, index=False)
        paths.append(p_cmp)
    return paths
