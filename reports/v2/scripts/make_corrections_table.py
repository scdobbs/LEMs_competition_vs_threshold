"""Addendum 10, 5.4: candidate rows of `reports/v2/corrections_table.md` from the manuscript and supplement .docx files.

Every number in each .docx (`docx_numbers.py`; tracked changes resolved to the current text) is matched to the v1 values
it can be: the old-against-new record (`tables/old_vs_new_record.csv`, built by `lemcvt phase6 record`) and the raw v1
reference (`reference/v1_reference.json`, AP in the units v1 printed). A match means the printed number equals the v1 value
rounded to the printed digits. Each match carries the v2 value, the causes, and the run the v2 value comes from; every
nonlinear-stage number is marked "centre form, nonstandard: relabel or drop" (addendum 10, 1.1). Matches are candidates:
a number can match several v1 values, and a v1 value can be printed in a sentence about something else, so the output is
reviewed by hand before it becomes the corrections table. Unmatched numbers are listed for that review.

    python reports/v2/scripts/make_corrections_table.py reports/inputs/*.docx > reports/v2/_corrections_candidates.md
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from docx_numbers import numbers  # noqa: E402
from make_addendum10_tables import md  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RECORD = REPO / "tables" / "old_vs_new_record.csv"
REF = REPO / "reference" / "v1_reference.json"
NONLINEAR_FLAG = "centre form, nonstandard: relabel or drop"
# record quantity <- (section, key) of the v1 reference
REF_KEYS = {("config", "concavity_theta"): "theta / m", ("config", "steepness_ks"): "k_s", ("config", "hilltop_curvature"): "C_ht",
            ("config", "K0_from_ks"): "K0 = U/k_s", ("config", "D0_from_curvature"): "D0 = U/C_ht",
            ("ad", "K"): "AD K", ("ad", "D"): "AD D", ("ad", "MSE"): "AD MSE", ("acrit", "A_crit"): "stage-2 A_crit", ("acrit", "MSE"): "stage-2 MSE",
            ("nonlinear", "A_crit"): "nonlinear A_crit", ("nonlinear", "MSE"): "nonlinear MSE",
            ("joint", "A_crit"): "joint A_crit", ("joint", "K"): "joint K", ("joint", "D"): "joint D", ("joint", "MSE"): "joint MSE"}
STAGE_OF = {"theta / m": "inputs", "k_s": "inputs", "C_ht": "inputs", "K0 = U/k_s": "inputs", "D0 = U/C_ht": "inputs", "AD K": "ad", "AD D": "ad",
            "AD MSE": "ad", "stage-2 A_crit": "joint", "stage-2 MSE": "joint", "joint A_crit": "joint", "joint K": "joint", "joint D": "joint",
            "joint MSE": "joint", "nonlinear A_crit": "nonlinear", "nonlinear MSE": "nonlinear"}
INPUTS_RUN = "20261111T002913Z_phase6_inputs_10tiles"


def parse_number(s: str) -> tuple[float, float] | None:
    """(value, half a unit in the last printed digit) of a printed number: 3.97 × 10−7, 3.97e-7, 1,234.5, 12 %."""
    t = s.replace("−", "-").replace("–", "-").replace(",", "").replace("%", "").strip()
    m = re.fullmatch(r"(-?\d+(?:\.(\d+))?)(?:\s?[×x]\s?10(-?\d+)|[eE]([-+]?\d+))?", t)
    if not m:
        return None
    mant, frac = m.group(1), m.group(2) or ""
    exp = int(m.group(3) or m.group(4) or 0)
    v = float(mant) * 10.0**exp
    half = 0.5 * 10.0 ** (exp - len(frac))
    return v, half


def run_of(tile: str, quantity: str) -> str:
    from lemcvt.analyses.old_vs_new import phase6_runs

    st = STAGE_OF.get(quantity)
    if st == "inputs":
        return INPUTS_RUN
    if st is None:
        return "see the record row"
    runs = phase6_runs(tile)
    return runs[st][0].name if st in runs else "-"


def candidates() -> pd.DataFrame:
    rec = pd.read_csv(RECORD)
    ref = json.loads(REF.read_text())["tiles"]
    rows = []
    for _, r in rec.iterrows():
        if isinstance(r.v1, float) and math.isfinite(r.v1):
            rows.append({"tile": r.tile, "quantity": r.quantity, "v1_printed_units": r.v1, "v1_form": f"record ({r.unit_v1})", "v2": r.v2,
                         "unit_v2": r.unit_v2, "causes": r.causes, "note": r.note if isinstance(r.note, str) else ""})
    for tile, t in ref.items():
        for (sec, key), q in REF_KEYS.items():
            src = t["config"] if sec == "config" else t["stages"].get(sec, {})
            v = src.get(key)
            hit = rec[(rec.tile == tile) & (rec.quantity == q)]
            if v is None or hit.empty:
                continue
            h = hit.iloc[0]
            rows.append({"tile": tile, "quantity": q, "v1_printed_units": float(v), "v1_form": "v1 raw (as printed by v1)", "v2": h.v2,
                         "unit_v2": h.unit_v2, "causes": h.causes, "note": h.note if isinstance(h.note, str) else ""})
    return pd.DataFrame(rows)


def match(paths: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    cand = candidates()
    hits, miss = [], []
    for p in paths:
        for n in numbers(p):
            pv = parse_number(str(n["number"]))
            if pv is None:
                continue
            v, half = pv
            if v == 0 or abs(v) < 1e-300:
                continue
            m = cand[(cand.v1_printed_units - v).abs() <= half * (1 + 1e-9)]
            if m.empty:
                miss.append({"document": Path(p).name, **n})
                continue
            for _, c in m.iterrows():
                nonlinear = c.quantity.startswith("nonlinear")
                hits.append({"document": Path(p).name, "paragraph": n["paragraph"], "part": n["part"], "printed": n["number"], "context": n["context"],
                             "tile": c.tile, "quantity": c.quantity, "old (v1)": c.v1_printed_units, "v1 form": c.v1_form, "new (v2)": c.v2,
                             "unit": c.unit_v2, "cause": c.causes, "run": run_of(c.tile, c.quantity),
                             "flag": NONLINEAR_FLAG if nonlinear else ""})
    h = pd.DataFrame(hits)
    if len(h):      # one row per (number, tile, quantity); the record form of v1 first, then the raw form
        h = h.drop_duplicates(subset=["document", "part", "paragraph", "printed", "tile", "quantity"], keep="first").reset_index(drop=True)
    return h, pd.DataFrame(miss)


def main(argv: list[str]) -> None:
    hits, miss = match(argv[1:])
    print("# Corrections table: candidate matches (to be reviewed)\n")
    print(f"{len(hits)} matched numbers, {len(miss)} unmatched numbers in {len(argv) - 1} documents.\n")
    if len(hits):
        print(md(hits, nd=5))
    print("\n## Unmatched numbers\n")
    if len(miss):
        print(miss.to_csv(index=False))


if __name__ == "__main__":
    main(sys.argv)
