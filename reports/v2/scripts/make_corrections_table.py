"""Addendum 11, 2.2 (tooling of addendum 10, 5.4): the manuscript corrections table.

    python reports/v2/scripts/make_corrections_table.py papers/<manuscript>.docx papers/<supplement>.docx

writes `reports/v2/corrections_table.md` (the header below, the claim flags of addendum 11, 2.3 quoted from the documents,
the counts and the table), `corrections_table.csv` beside it (one row per printed number that states a quantity) and
`corrections_labels.csv` (numbers that are labels: section, equation, figure and table numbers, citations, the reference
list, addresses, DOIs and software versions). The documents are only read.

Sources of printed numbers:
- every paragraph of both .docx (`docx_numbers.py`; tracked changes resolved to the current text, equations included);
- `reports/v2/corrections_image_numbers.csv`: the numbers held in images (Table 1 from the text of the PDF embedded in its
  EMF; Table S1 and the panel annotations of Figures 1, 4 and 6 read by eye). Axis ticks, colour bars and scale bars are
  not included. Each of these rows names the tile and quantity of its cell, so it is tied by position, then checked by value.

A text number is tied by value: a candidate is any v1 quantity that equals the printed number at the printed precision.
Candidates are the old-against-new record (`tables/old_vs_new_record.csv`, v1 in the record's units, plus the AP values in
the feet-based units v1 printed), statistics derived from it as the text states them (site medians and ranges of
fitted-to-measured ratios, A_crit medians, MSE ratios, the resolution-sweep medians) and the fixed setup values (the
idealized experiment, the sweep, grids, optimizer bounds). The per-dx sweep ratios (500 values) are not candidates; only
their site medians at the end points are. In running text a candidate counts only if its context cue (a regular
expression naming its tile, site or quantity, e.g. "AP tile 1", "Allegheny ... median ... K", "bounds") is in the
sentence: without it every small number would match some value. Candidates with the same quantity and the same v1 and v2
on several tiles are one candidate. If candidates of both sites remain, the site named first after the number in its
sentence selects one; if several still remain the row is `ambiguous`.

Status (addendum 11, 2.2): `unchanged` (v2 equals the printed number at its precision, same units), `update`,
`relabel or drop` (every nonlinear-stage number), `reword` (an unchanged result in a sentence flagged under 2.3; a fixed setup
value such as n = 1 stays `unchanged`), `no match`,
`ambiguous`.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from docx_numbers import NUM, SUP, SUP_RUN, paragraphs  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "reports" / "v2"
RECORD = REPO / "tables" / "old_vs_new_record.csv"
REF = REPO / "reference" / "v1_reference.json"
IMAGES = OUT / "corrections_image_numbers.csv"
FT = 1200 / 3937                       # US survey foot in metres
C = 3937 / 1200
NONLINEAR = "centre form, nonstandard: relabel or drop"
INPUTS_RUN = "20261111T002913Z_phase6_inputs_10tiles"
SITES = {"AP": "Allegheny", "GM": "Gabilan"}

# record quantity <- (section, key) of the v1 reference: the values as v1 printed them (AP in feet-based units)
REF_KEYS = {("config", "concavity_theta"): "theta / m", ("config", "steepness_ks"): "k_s", ("config", "hilltop_curvature"): "C_ht",
            ("config", "K0_from_ks"): "K0 = U/k_s", ("config", "D0_from_curvature"): "D0 = U/C_ht",
            ("ad", "K"): "AD K", ("ad", "D"): "AD D", ("ad", "MSE"): "AD MSE", ("acrit", "A_crit"): "stage-2 A_crit", ("acrit", "MSE"): "stage-2 MSE",
            ("nonlinear", "A_crit"): "nonlinear A_crit", ("nonlinear", "MSE"): "nonlinear MSE",
            ("joint", "A_crit"): "joint A_crit", ("joint", "K"): "joint K", ("joint", "D"): "joint D", ("joint", "MSE"): "joint MSE"}
STAGE_OF = {"inputs": ("theta / m", "k_s", "C_ht", "K0 = U/k_s", "D0 = U/C_ht"), "ad": ("AD K", "AD D", "AD MSE"),
            "joint": ("stage-2 A_crit", "stage-2 MSE", "joint A_crit", "joint K", "joint D", "joint MSE"),
            "nonlinear": ("nonlinear A_crit", "nonlinear MSE")}

# ---------------------------------------------------------------------------------------------------------------- claims
# Addendum 11, 2.3: sentences that make one of the six claims, located by (document, paragraph, a unique substring).
# The sentence quoted in the output is the whole sentence that contains the substring, as the document prints it.
CLAIMS = {
    1: ("The partitioned model recovers or retains measured D", [
        ("Manuscript", 7, "recovers field-measured parameters"),
        ("Manuscript", 9, "retains field-measured parameters"),
        ("Manuscript", 9, "Recovering observed landscapes at field-measured parameters"),
        ("Manuscript", 11, "using independently measured field values"),
        ("Manuscript", 21, "retains field-measured parameter values"),
        ("Manuscript", 92, "while D increases moderately"),
        ("Manuscript", 93, "no longer needs to drive K and D to non-physical values"),
        ("Manuscript", 106, "recovers the independently field-measured K and D"),
        ("Manuscript", 108, "K and D retain field-measured values under optimization"),
        ("Manuscript", 115, "retains field-measured parameters"),
        ("Manuscript", 115, "recovering observed landscapes at field-measured parameters"),
        ("Manuscript", 186, "recovers both parameters close to their measured values"),
        ("Manuscript", 186, "while the partitioned model does so while retaining them"),
    ]),
    2: ("K is matched to an independent or field measurement", [
        ("Manuscript", 26, "we then measured all model parameters independently from that tile"),
        ("Manuscript", 27, "enter only as the tile-selection reference"),
        ("Manuscript", 87, "independently measured erosion rates, diffusivities, and erodibilities"),
        ("Manuscript", 92, "preserves K essentially at the per-tile measured values"),
        ("Manuscript", 106, "recovers the independently field-measured K and D"),
        ("Manuscript", 108, "K and D retain field-measured values under optimization"),
        ("Manuscript", 186, "recovers both parameters close to their measured values"),
        ("Manuscript", 193, "derived independently from each tile"),
    ]),
    3: ("Any nonlinear-stage result (centre form, nonstandard)", [
        ("Manuscript", 106, "improves the fit of the partitioned model to observed topography at every tile"),
        ("Supplement", 35, "improves the fit of the partitioned model to the observed topography at every tile"),
        ("Supplement", 36, "Nonlinear transport thus refines the partitioned model"),
        ("Supplement", 71, "The nonlinear law lowers MSE at every tile"),
        ("Supplement", 71, "for the partitioned model run with linear hillslope diffusion"),
    ]),
    4: ("AP numbers that were in feet, or that came from the v1 AP measurements", [
        ("Manuscript", 26, "range 0.40"),
        ("Manuscript", 88, "Across all Allegheny Plateau tiles"),
        ("Manuscript", 92, "median factor 1.03 in the Allegheny Plateau"),
        ("Manuscript", 92, "~2-fold (Allegheny)"),
        ("Manuscript", 92, "medians 7.6"),
        ("Manuscript", 189, "reducing the median MSE by a factor of 2.0"),
        ("Supplement", 30, "AP tile 1: 6.96"),
    ]),
    5: ("Optimizer-driven values that changed by more than 10 % (GM1 AD optimum, GM2 stage-2 A_crit, joint A_crit GM1, GM3, GM4)", [
        ("Manuscript", 88, "Gabilan Mesa shows the same directional shift"),
        ("Manuscript", 92, "The optimized Acrit values fall within"),
    ]),
    6: ("The D-infinity partitioned MSEs", []),
}
CLAIM_IMAGE_CELLS = {   # image-held numbers under a claim (location prefix -> claim numbers)
    "Table 1, Allegheny Plateau": (4,), "Table S1, Allegheny Plateau": (4,), "Figure 1, Allegheny Plateau": (4,),
    "Figure 4, right (Allegheny Plateau": (4,), "Figure 6": (4,),
    "Table 1, Gabilan Mesa Tile 1, Advection-Diffusion": (5,), "Table 1, Gabilan Mesa Tile 1, Linear Partitioned A_crit": (5,),
    "Table 1, Gabilan Mesa Tile 3, Linear Partitioned A_crit": (5,), "Table 1, Gabilan Mesa Tile 4, Linear Partitioned A_crit": (5,),
    "Table 1, Gabilan Mesa Median": (5,),
}

# ---------------------------------------------------------------------------------------- labels: hand review of the text
# Numbers in running text that are mathematics or labels rather than quantities, found by reading every number the rules
# below leave as a quantity: (document, paragraph, printed, occurrence of that printed string in the paragraph, kind).
LABEL_OVERRIDES: list[tuple[str, int, str, int, str]] = [
    ("Manuscript", 186, "1", 1, "math"), ("Manuscript", 186, "1", 2, "math"),        # plots at (1, 1)
    ("Manuscript", 170, "−1", 1, "reference line"), ("Manuscript", 170, "–1", 1, "reference line"),
    ("Supplement", 67, "1", 1, "reference line"), ("Supplement", 67, "-1", 1, "reference line"),
]

LABEL_BEFORE = re.compile(r"(Eqs?\.\s?|Eq\s|Figures?\s|Fig\.?\s|Tables?\s|Text\s|Texts\s|[Vv]ersion\s|Python\s|tile\s|Tile\s|Tiles\s|"
                          r"Section\s|CA\s)$")
EQ_PAR = re.compile(r"\(Eq\.\s?\d+\)")
HEADING_WORDS = {"Abstract", "Plain Language Summary", "Key Points:", "Supporting Information", "Open Research",
                 "Conflict of Interest Disclosure", "References", "Figures", "Introduction", "Contents of this file"}


def parse_number(s: str) -> tuple[float, float] | None:
    """(value, half a unit in the last printed digit): 3.97 × 10^-7, 3.97e-7, 10^5, 1,234.5, 12 %."""
    t = s.replace("−", "-").replace("–", "-").replace(",", "").replace("%", "").replace(" ", "").strip()
    m = re.fullmatch(r"10\^(-?\d+)", t)
    if m:
        e = int(m.group(1))
        return 10.0**e, 0.5 * 10.0**e
    m = re.fullmatch(r"(-?\d+(?:\.(\d+))?)(?:[×x]10\^?(-?\d+)|[eE]([-+]?\d+))?", t)
    if not m:
        return None
    mant, frac = m.group(1), m.group(2) or ""
    exp = int(m.group(3) or m.group(4) or 0)
    return float(mant) * 10.0**exp, 0.5 * 10.0 ** (exp - len(frac))


def eq(v: float, printed: tuple[float, float]) -> bool:
    return v is not None and math.isfinite(v) and abs(v - printed[0]) <= printed[1] * (1 + 1e-9)


# ---------------------------------------------------------------------------------------------------------- candidates
def runs_of(tile: str) -> dict[str, str]:
    from lemcvt.analyses.old_vs_new import phase6_runs

    return {k: v[0].name for k, v in phase6_runs(tile).items()}


def latest_run(step: str, pred) -> str:
    from lemcvt.results import find_runs

    for r in reversed(find_runs(step)):
        try:
            if pred(json.loads((r / "params.json").read_text())):
                return r.name
        except (OSError, ValueError):
            continue
    return "-"


def sweep_runs() -> str:
    """The resolution-sweep runs the record uses (params_source config, D0 variant A), latest per tile."""
    from lemcvt.results import find_runs

    by_tile: dict[str, str] = {}
    for r in find_runs("sweep_resolution"):
        try:
            q = json.loads((r / "params.json").read_text())
        except (OSError, ValueError):
            continue
        if q.get("params_source") == "config" and q.get("d0_variant", "A") == "A":
            for t in q.get("tiles", [q.get("tile")]):
                if t:
                    by_tile[str(t)] = r.name
    return ", ".join(sorted(set(by_tile.values()))) or "-"


def candidates() -> pd.DataFrame:
    rec = pd.read_csv(RECORD)
    ref = json.loads(REF.read_text())
    runs = {t: runs_of(t) for t in sorted(rec.tile.unique()) if t[:2] in SITES}
    sa_run = latest_run("slope_area", lambda q: "v1 saved grids" not in json.dumps(q.get("tiles", {})))
    dd_run = latest_run("d8_vs_dinf", lambda q: q.get("params_source") == "config")
    sw_run = sweep_runs()

    def run_for(tile: str, q: str) -> str:
        for st, qs in STAGE_OF.items():
            if q in qs:
                return INPUTS_RUN if st == "inputs" else runs.get(tile, {}).get(st, "-")
        if q.startswith("slope-area"):
            return sa_run
        if q.startswith("d8_vs_dinf"):
            return dd_run
        return "-"

    rows = []

    def add(tile, q, v1, v2, unit_v1, unit_v2, causes, note, form, run, cue, ratio=None, nonlinear=False):
        rows.append({"tile": tile, "quantity": q, "v1": v1, "v2": v2, "unit_v1": unit_v1, "unit_v2": unit_v2, "causes": causes,
                     "note": note, "form": form, "run": run, "cue": cue, "nonlinear": nonlinear,
                     "ratio": ratio if ratio is not None else (v2 / v1 if v1 not in (0, None) and v2 is not None and math.isfinite(v2) else np.nan)})

    def tile_cue(tile: str) -> str:
        return rf"({SITES[tile[:2]]}|{tile[:2]})\b.*\b[Tt]ile {tile[2]}\b"

    for _, r in rec.iterrows():
        if r.quantity.startswith("sweep ratio"):
            continue
        note = r.note if isinstance(r.note, str) else ""
        nl = r.quantity.startswith("nonlinear")
        add(r.tile, r.quantity, r.v1, r.v2, r.unit_v1 if isinstance(r.unit_v1, str) else "", r.unit_v2 if isinstance(r.unit_v2, str) else "",
            r.causes if isinstance(r.causes, str) else "", note, "record", run_for(r.tile, r.quantity), tile_cue(r.tile), r.ratio_v2_over_v1, nl)
    # the AP values as v1 printed them (feet-based units)
    for tile, t in ref["tiles"].items():
        if not tile.startswith("AP"):
            continue
        for (sec, key), q in REF_KEYS.items():
            src = t["config"] if sec == "config" else t["stages"].get(sec, {})
            v = src.get(key)
            hit = rec[(rec.tile == tile) & (rec.quantity == q)]
            if v is None or hit.empty:
                continue
            h = hit.iloc[0]
            add(tile, q, float(v), h.v2, "v1 as printed (z in US survey feet)", h.unit_v2, h.causes, h.note if isinstance(h.note, str) else "",
                "v1 raw (feet-based)", run_for(tile, q), tile_cue(tile), h.ratio_v2_over_v1, q.startswith("nonlinear"))
    for _, h in rec[(rec.tile == "AP1") & rec.quantity.str.startswith("d8_vs_dinf")].iterrows():
        add("AP1", h.quantity, h.v1 / FT**2, h.v2, "ft^2 (v1 as printed)", "m^2", h.causes, "", "v1 raw (feet-based)", dd_run,
            r"AP tile 1|Allegheny.*tile 1", h.ratio_v2_over_v1)

    # ---- statistics derived from the record, as the text states them
    def val(tile, q, col="v2"):
        x = rec[(rec.tile == tile) & (rec.quantity == q)]
        return float(x[col].iloc[0]) if len(x) else np.nan

    def d0b(tile):
        x = rec[(rec.tile == tile) & (rec.quantity == "D0 = U/C_ht")]
        m = re.search(r"D0_B = ([0-9.e+-]+)", str(x.note.iloc[0])) if len(x) else None
        return float(m.group(1)) if m else np.nan

    stat_runs = lambda site, st: ", ".join(runs[f"{site}{i}"].get(st, "-") for i in range(1, 6))  # noqa: E731
    for site, name in SITES.items():
        tiles = [f"{site}{i}" for i in range(1, 6)]
        for num, den, label, st in (("AD K", "K0 = U/k_s", "AD K/K0", "ad"), ("AD D", "D0 = U/C_ht", "AD D/D0", "ad"),
                                    ("joint K", "K0 = U/k_s", "partitioned K/K0", "joint"), ("joint D", "D0 = U/C_ht", "partitioned D/D0", "joint")):
            r1 = np.array([val(t, num, "v1") / val(t, den, "v1") for t in tiles])
            r2 = np.array([val(t, num) / val(t, den) for t in tiles])
            note = ""
            if site == "AP" and den.startswith("D0"):
                rb = np.array([val(t, num) / d0b(t) for t in tiles])
                note = (f"v2 with D0,A; with D0,B: median {np.median(rb):.3g}, range {rb.min():.3g}-{rb.max():.3g}")
            word = "K" if "K" in label else "D"
            cue_med = rf"{name}.*\b{word}\b.*median|\b{word}\b.*median.*{name}"
            for stat, f in (("median", np.median), ("min", np.min), ("max", np.max)):
                add(site, f"{stat} {label}", float(f(r1)), float(f(r2)), "-", "-", "optimizer; measurement; m" + ("; vertical units; AP D0 bracket" if site == "AP" else ""),
                    note, "derived", stat_runs(site, st), cue_med if stat == "median" else rf"{name}.*\b{word}\b.*range|\b{word}\b.*range.*{name}")
        a1 = np.array([val(t, "joint A_crit", "v1") for t in tiles])
        a2 = np.array([val(t, "joint A_crit") for t in tiles])
        add(site, "median joint A_crit", float(np.median(a1)), float(np.median(a2)), "m^2", "m^2", "optimizer; m", "", "derived", stat_runs(site, "joint"),
            rf"@[\d.]+ × 10\^\d+ m\^?2 {name}")
        th1 = np.array([val(t, "theta / m", "v1") for t in tiles])
        add(site, "min per-tile theta", float(th1.min()), val(tiles[0], "theta / m"), "-", "-", "measurement; m", "v2: one site constant m (no per-tile theta)", "derived", INPUTS_RUN, rf"per-tile.*range.*{name}")
        add(site, "max per-tile theta", float(th1.max()), val(tiles[0], "theta / m"), "-", "-", "measurement; m", "v2: one site constant m (no per-tile theta)", "derived", INPUTS_RUN, rf"per-tile.*range.*{name}")
        ad1 = [val(t, "AD MSE", "v1") for t in tiles]
        ad2 = [val(t, "AD MSE") for t in tiles]
        jo1 = [val(t, "joint MSE", "v1") for t in tiles]
        jo2 = [val(t, "joint MSE") for t in tiles]
        add(site, "ratio of median MSE, AD / partitioned", float(np.median(ad1) / np.median(jo1)), float(np.median(ad2) / np.median(jo2)), "-", "-",
            "optimizer; m" + ("; vertical units" if site == "AP" else ""), "", "derived", stat_runs(site, "ad") + "; " + stat_runs(site, "joint"), rf"MSE.*fold.*{name}")
        sw = rec[rec.tile.str.startswith(site) & rec.quantity.str.startswith("sweep ratio D8 at dx ")]
        for dx in ("5.0", "100.0"):
            s = sw[sw.quantity == f"sweep ratio D8 at dx {dx}"]
            add(site, f"median sweep ratio k_s,mod/k_s,obs (D8) at dx {dx} m", float(s.v1.median()), float(s.v2.median()), "-", "-",
                "measurement; m; tie-break N1-N3", "", "derived", sw_run, r"R\(Δx\)")
    all_tiles = [f"{s}{i}" for s in SITES for i in range(1, 6)]
    aa1 = [val(t, "joint A_crit", "v1") for t in all_tiles]
    aa2 = [val(t, "joint A_crit") for t in all_tiles]
    add("all", "min joint A_crit (10 tiles)", min(aa1), min(aa2), "m^2", "m^2", "optimizer; m", "", "derived", "phase6 joint runs", r"@within [\d.]+ × 10\^\d")
    add("all", "max joint A_crit (10 tiles)", max(aa1), max(aa2), "m^2", "m^2", "optimizer; m", "", "derived", "phase6 joint runs", r"@–[\d.]+ × 10\^\d+ m\^?2 across")
    # Figure 6: per-tile MSE as v1 plotted it (AP in ft^2) against v2 (m^2)
    ad1 = [val(t, "AD MSE", "v1") / (FT**2 if t.startswith("AP") else 1) for t in all_tiles]
    jo1 = [val(t, "joint MSE", "v1") / (FT**2 if t.startswith("AP") else 1) for t in all_tiles]
    ad2 = [val(t, "AD MSE") for t in all_tiles]
    jo2 = [val(t, "joint MSE") for t in all_tiles]
    add("all", "median MSE reduction, AD / partitioned, 10 tiles", float(np.median(ad1) / np.median(jo1)), float(np.median(ad2) / np.median(jo2)), "-", "-",
        "optimizer; m; vertical units", "v1 mixes AP in ft^2 with GM in m^2; v2 all in m^2", "derived", "phase6 AD and joint runs", r"median MSE|median misfit")

    # ---- fixed setup values (v1 code; v2 code)
    from lemcvt.analyses.synthetic_pe import PARAMS
    setup = [
        ("Lx", 300.0, PARAMS["Lx"], "m", r"@Lx\s?=\s?300"), ("Ly", 200.0, PARAMS["Ly"], "m", r"@Ly\s?=\s?200"),
        ("reference length L_Pe", 50.0, PARAMS["L_PE"], "m", r"@\bL\s?=\s?50"),
        ("idealized K", 1e-4, PARAMS["K"], "m^(1-2m)/yr", r"@\bK\s?=\s?1x10"), ("idealized m", 0.35, PARAMS["m"], "-", r"@\bm = 0.35"),
        ("inlet Peclet number", 100.0, PARAMS["PE_BAL"], "-", r"@Pe\s?=\s?100"), ("baseline Peclet number", 1.0, 1.0, "-", r"@Pe\s?=\s?1\b"),
        ("baseline D", 4.4e-2, 4.4e-2, "m^2/yr", r"@\bD\s?=\s?4.4"), ("A_inlet", 2.59e8, PARAMS["A_INLET_BAL_ANCHOR"], "m^2", r"@inlet\s?=\s?2.59"),
        ("idealized dx min", 1.0, 1.0, "m", r"@Δx = 1 to"), ("idealized dx max", 50.0, 50.0, "m", r"@Δx = 1 to 50"),
        ("sweep dx min", 5.0, 5.0, "m", r"@5–100|Δx = 5 m"), ("sweep dx max", 100.0, 100.0, "m", r"@5–100|Δx = 100 m"),
        ("sweep A_min", 1e5, 1e5, "m^2", r"@Amin=10"), ("stream-power exponent n", 1.0, 1.0, "-", r"@\bn = 1"),
        ("DEM grid spacing", 1.0, 1.0, "m", r"@\b1[- ]m\b|1-meter|1 m–"), ("tile size", 1.0, 1.0, "km", r"@1 km"),
        ("v1 curvature block size, GM", 10.0, 11.0, "m", r"@10 m Gabilan"), ("v1 curvature block size, AP", 30.0, 11.0, "m", r"@30 m Allegheny"),
    ]
    for q, v1, v2, unit, cue in setup:
        note = {"idealized dx max": "v1 and v2 code: 25 log-spaced targets from 1 to 55 m; the coarsest actual dx is 50 m",
                "v1 curvature block size, GM": "v2 measures D0 by option (iii) at an 11 m window (docs/measurement_methods.md section 0)",
                "v1 curvature block size, AP": "v2 measures D0 by option (iii) at an 11 m window; AP D0 is a bracket [D0,A, D0,B]",
                "DEM grid spacing": "AP tiles are 0.9752 m (PAMAP 3.2 ft) in v1 and v2"}.get(q, "")
        cz = "measurement" if q.startswith("v1 curvature") else ""
        add("setup", q, v1, v2, unit, unit, cz, note, "setup", "lemcvt.analyses.synthetic_pe.PARAMS" if q in ("Lx", "Ly", "reference length L_Pe", "idealized K", "idealized m", "inlet Peclet number", "A_inlet") else "configs", cue)
    for tile, t in ref["tiles"].items():
        cfg = t["config"]
        f = C if tile.startswith("AP") else 1.0
        for i, end in enumerate(("lower", "upper")):
            add(tile, f"K bound {end}", cfg["K_bounds"][i], cfg["K_bounds"][i] * f, "1/yr (v1)", "1/yr", "m; vertical units" if f != 1 else "", "", "setup", "configs/tiles", r"bounds")
            add(tile, f"D bound {end}", cfg["D_bounds"][i], cfg["D_bounds"][i] * f, "m^2/yr (v1)", "m^2/yr", "m; vertical units" if f != 1 else "", "", "setup", "configs/tiles", r"bounds")
    return pd.DataFrame(rows)


# -------------------------------------------------------------------------------------------------------- text numbers
def sentences(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z(“\"])", text):
        if re.search(r"(e\.g|i\.e|et al|Eqs?|Fig|vs|approx|ca)\.$", text[start:m.start()]):
            continue
        spans.append((start, m.start()))
        start = m.end()
    spans.append((start, len(text)))
    return spans


def doc_paragraphs(path: str, name: str) -> list[dict]:
    out, section, k, in_refs, front, caption = [], "front matter", 0, False, True, None
    for _part, i, t in paragraphs(path):
        t = SUP_RUN.sub(lambda q: "^" + q.group(0).translate(SUP), t).replace("\u2009", " ").replace("\xa0", " ")
        plain = t.replace("^", "")
        cap = re.match(r"(Figure S?\d+|Table S?\d+)\.", plain)
        is_head = (plain in HEADING_WORDS or re.match(r"\d+(\.\d+)? [A-Z]", plain) and len(plain) < 90
                   or re.match(r"Text S\d\.", plain)
                   or (len(plain) < 70 and not EQ_PAR.search(plain) and not plain.endswith((".", ",")) and plain[:1].isupper()
                       and not cap and plain not in ("where",) and not front))
        if plain in ("Abstract", "Introduction"):
            front = False
        if is_head:
            section, k = plain, 0
            in_refs = plain == "References" or (in_refs and plain != "Figures")
        else:
            k += 1
        if cap:
            caption = cap.group(1)
        elif is_head and section in ("Figures", "References"):
            caption = None
        loc = f"{caption} caption" if caption and not is_head else f"§ {section}, ¶{k}"
        kind = ("front matter" if front else "reference list" if in_refs and not is_head else "heading" if is_head
                else "equation" if EQ_PAR.search(plain) else "")
        out.append({"document": name, "paragraph": i, "location": loc, "text": t, "kind": kind})
    return out


def classify(p: dict, m: re.Match, nth: int, overrides: dict) -> str:
    if p["kind"]:
        return p["kind"]
    key = (p["document"], p["paragraph"], m.group(0), nth)
    if key in overrides:
        return overrides[key]
    t, s = p["text"], m.start()
    before = t[max(0, s - 12):s]
    if LABEL_BEFORE.search(before):
        return "cross-reference or label"
    if re.search(r"(Eqs?\.\s?|Figures?\s|Fig\.?\s|Tables?\s)\d+[a-z]?(\s?[–-]\s?| and )$", t[max(0, s - 16):s]):
        return "cross-reference or label"
    if re.search(r"[A-Za-z]\^[\d\s–−-]*$", before) or re.search(r"m\^?\((1\s?[−–-]\s?)?$", before):
        return "unit exponent"
    if m.group(0).strip() == "0" or re.search(r"[><≈≪≫±]\s?$", before) or re.match(r"\s?\+", t[m.end():m.end() + 3]):
        return "math"
    tok = re.search(r"\S*$", t[:s]).group(0)
    if "doi" in tok or "http" in tok or "/" in tok and "." in tok:
        return "DOI or URL"
    if re.fullmatch(r"(19|20)\d\d[a-z]?", m.group(0).strip()):
        return "citation year"
    if re.match(r"\s?(UL|L/|L\)|/Δx|Δx\)|U/D|UL/D|/\(DΔx\))", t[m.end():m.end() + 6]):
        return "math"
    if parse_number(m.group(0)) is None:
        return "version or identifier"
    return ""


def sentence_of(text: str, pos: int) -> tuple[int, int]:
    for a, b in sentences(text):
        if a <= pos <= b:
            return a, b
    return 0, len(text)


def claim_sentences(pars: list[dict]) -> list[dict]:
    by = {(p["document"], p["paragraph"]): p for p in pars}
    out = []
    for n, (title, keys) in CLAIMS.items():
        for doc, i, sub in keys:
            p = by.get((doc, i))
            if p is None:
                raise SystemExit(f"claim {n}: no paragraph {doc} {i}")
            t = p["text"]
            pos = t.find(sub)
            if pos < 0:
                raise SystemExit(f"claim {n}: '{sub}' not in {doc} paragraph {i}")
            a, b = sentence_of(t, pos)
            out.append({"claim": n, "title": title, "document": doc, "paragraph": i, "location": p["location"], "sentence": pretty(t[a:b]).replace("^", "").strip(),
                        "span": (a, b)})
    return out


def claims_at(claims: list[dict], doc: str, i: int, pos: int) -> list[int]:
    return sorted({c["claim"] for c in claims if c["document"] == doc and c["paragraph"] == i and c["span"][0] <= pos <= c["span"][1]})


# ---------------------------------------------------------------------------------------------------------------- build
def fmt(v) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    if isinstance(v, str):
        return v
    return f"{v:.4g}"


def merge(hit: pd.DataFrame) -> pd.DataFrame:
    """One candidate per (quantity, v1, v2): the same value on several tiles (e.g. a bound shared by AP1, AP3, AP5), or a
    unit-free AP value that is the same in the record and as v1 printed it (the record form is kept)."""
    g = []
    hit = hit.assign(_raw=(hit.form == "v1 raw (feet-based)")).sort_values("_raw", kind="stable").drop(columns="_raw")
    for _, grp in hit.groupby(["quantity", "v1", "v2"], sort=False, dropna=False):
        r = grp.iloc[0].copy()
        r["tile"] = ", ".join(dict.fromkeys(grp.tile))
        r["run"] = " ; ".join(dict.fromkeys(grp.run))
        g.append(r)
    return pd.DataFrame(g)


def site_after(sentence: str, pos: int) -> str | None:
    m = re.search(r"Allegheny|Gabilan|\bAP\b|\bGM\b", sentence[pos:])
    if m:
        return "AP" if m.group(0) in ("Allegheny", "AP") else "GM"
    named = {k for k, v in SITES.items() if v in sentence or re.search(rf"\b{k}\b", sentence)}
    return named.pop() if len(named) == 1 else None


def decide(printed, cands: pd.DataFrame, claims: list[int], context: str, pos: int | None = None) -> dict:
    """pos is given for running text: then a candidate needs its cue in the sentence (context)."""
    pv = parse_number(printed)
    if pv is None or not len(cands):
        return {"status": "no match"}
    hit = cands[[eq(v, pv) for v in cands.v1]]
    how = ""
    if pos is not None and len(hit):
        start = pos - len(printed)
        local = hit[[c.startswith("@") and any(q.start() <= start < q.end() for q in re.finditer(c[1:], context)) for c in hit.cue]]
        if len(local):                      # a cue that spans the number itself wins
            hit, how = local, ""
        else:
            hit = hit[[not c.startswith("@") and bool(re.search(c, context)) for c in hit.cue]]
    if len(hit) > 1:
        hit = merge(hit)
    if len(hit) > 1 and pos is not None:
        site = site_after(context, pos)
        sh = hit[[str(t)[:2] == site for t in hit.tile]] if site else hit.iloc[0:0]
        if len(sh) >= 1 and len(sh) < len(hit):
            hit, how = sh, f"of the candidates of both sites, the {site} one (the site named after the number)"
    if not len(hit):
        return {"status": "no match"}
    if len(hit) > 1:
        return {"status": "ambiguous", "candidates": hit}
    c = hit.iloc[0]
    if c.nonlinear:
        status = "relabel or drop"
    else:
        same_units = c.form != "v1 raw (feet-based)" and c.unit_v1 not in ("1/ft", "ft-based")
        unchanged = same_units and eq(c.v2, pv)
        status = ("reword" if claims and c.form != "setup" else "unchanged") if unchanged else "update"
    return {"status": status, "c": c, "how": how}


def row_from(doc, location, printed, context, source, d: dict, claims: list[int], note: str = "") -> dict:
    r = {"document": doc, "location": location, "printed": printed, "context": context, "source": source, "status": d["status"],
         "v1": "", "v2": "", "ratio": "", "causes": "", "source_run": "", "tile": "", "quantity": "", "claims": ",".join(map(str, claims)), "note": note}
    if "c" in d:
        c = d["c"]
        r.update({"v1": f"{fmt(c.v1)} {c.unit_v1}".strip(), "v2": f"{fmt(c.v2)} {c.unit_v2}".strip(), "ratio": fmt(c.ratio), "causes": c.causes,
                  "source_run": c.run, "tile": c.tile, "quantity": c.quantity + (f" [{c.form}]" if c.form != "record" else "")})
        extra = "; ".join(x for x in (c.note, d.get("how", "")) if x)
        if c.nonlinear:
            extra = "; ".join(x for x in (NONLINEAR, extra) if x)
        r["note"] = "; ".join(x for x in (r["note"], extra) if x)
    if "candidates" in d:
        cc = d["candidates"]
        r["tile"] = " | ".join(cc.tile)
        r["quantity"] = " | ".join(q + (f" [{f}]" if f != "record" else "") for q, f in zip(cc.quantity, cc.form, strict=True))
        r["v1"] = " | ".join(fmt(v) for v in cc.v1)
        r["v2"] = " | ".join(fmt(v) for v in cc.v2)
        r["source_run"] = " | ".join(cc.run)
    return r


def nearest(cands: pd.DataFrame, printed: str, tile: str | None = None, rel: float = 0.05) -> str:
    pv = parse_number(printed)
    if pv is None or pv[0] == 0:
        return ""
    c = cands if tile is None else cands[cands.tile == tile]
    c = c[np.isfinite(c.v1.astype(float))]
    d = (c.v1.astype(float) / pv[0] - 1).abs()
    c = c[d <= rel].assign(d=d[d <= rel]).sort_values("d").head(3)
    return "; ".join(f"{r.tile} {r.quantity} [{r.form}] v1 = {fmt(r.v1)}" for r in c.itertuples())


def build(paths: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    cands = candidates()
    overrides = {(d, p, s, n): k for d, p, s, n, k in LABEL_OVERRIDES}
    names = {}
    for p in paths:
        names[p] = "Supplement" if "Supplement" in Path(p).name else "Manuscript"
    pars = [q for p in paths for q in doc_paragraphs(p, names[p])]
    claims = claim_sentences(pars)
    rows, labels = [], []
    for p in pars:
        seen: dict[str, int] = {}
        for m in NUM.finditer(p["text"]):
            s = m.group(0).strip()
            seen[s] = seen.get(s, 0) + 1
            kind = classify(p, m, seen[s], overrides)
            a, b = sentence_of(p["text"], m.start())
            ctx = pretty(p["text"][max(a, m.start() - 60):min(b, m.end() + 60)]).replace("^", "")
            base = {"document": p["document"], "paragraph": p["paragraph"], "location": p["location"], "printed": s}
            if kind:
                labels.append({**base, "kind": kind})
                continue
            cl = claims_at(claims, p["document"], p["paragraph"], m.start())
            sent = p["text"][a:b]
            rel = m.end() - a
            d = decide(s, cands, cl, sent, rel)
            r = row_from(p["document"], p["location"], s, ctx, "text", d, cl)
            r["paragraph"] = p["paragraph"]
            if d["status"] == "no match":
                r["note"] = "; ".join(x for x in (r["note"], ("within 5 %: " + nearest(cands, s)) if nearest(cands, s) else "") if x)
            rows.append(r)
    with open(IMAGES, newline="") as fh:
        for q in csv.DictReader(fh):
            cl = sorted({n for k, ns in CLAIM_IMAGE_CELLS.items() if q["location"].startswith(k) for n in ns})
            tied = cands[(cands.quantity == q["quantity"]) & ((cands.tile == q["tile"]) | (cands.tile == q["tile"][:2]))]
            if q["tile"] == "all":
                tied = cands[cands.quantity == q["quantity"]]
            d = decide(q["printed"], tied, cl, q["location"])
            note = q["note"]
            if d["status"] == "no match":       # not the value of its own cell: search every candidate
                d2 = decide(q["printed"], cands, cl, q["location"])
                if d2["status"] != "no match" and "c" in d2:
                    d = d2
                    note = "; ".join(x for x in (note, f"the printed value is not this cell's v1 {q['quantity']} ({q['tile']}); it equals the v1 value of the quantity named in this row") if x)
                else:
                    near = nearest(cands, q["printed"], q["tile"])
                    note = "; ".join(x for x in (note, f"v1 {q['quantity']} of {q['tile']}: " + ", ".join(f"{fmt(v)} [{f}]" for v, f in zip(tied.v1, tied.form, strict=True)) if len(tied) else "", ("within 5 %: " + near) if near else "") if x)
            r = row_from(q["document"], q["location"], q["printed"], "", q["source"], d, cl, note)
            r["paragraph"] = ""
            rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(labels), claims


def source_lines(paths: list[str]) -> list[str]:
    import hashlib
    import os
    import time

    out = []
    for p in paths:
        st = os.stat(p)
        h = hashlib.sha256(Path(p).read_bytes()).hexdigest()
        out.append(f"- `{p}`: {st.st_size:,} bytes, modified {time.strftime('%Y-%m-%d %H:%M %Z', time.localtime(st.st_mtime))}, sha256 `{h[:16]}…`")
    return out


HEADER = """# Manuscript corrections table (v1 to v2)

Addendum 11, section 2. Generated by `reports/v2/scripts/make_corrections_table.py` (with `docx_numbers.py`) from:

{sources}

The documents were only read (tracked changes resolved to the current text). The CSV beside this file holds the same
rows with a short context of each number; `corrections_labels.csv` holds the {n_labels} numbers that are labels, not
quantities. {n_images} of the {n_rows} rows are numbers printed inside images: Table 1 (an EMF image; its text was read
from the PDF embedded in it), Table S1 and the panel annotations of Figures 1, 4 and 6 (read by eye from the images;
`reports/v2/corrections_image_numbers.csv`). Axis ticks, colour bars and scale bars are not listed.

**How a number is tied.** A number is tied to a v1 quantity when it equals that quantity at the printed precision. A
number in an image is tied by its cell (tile and quantity), then checked by value. A number in running text is tied by
value among the candidates whose context cue (tile, site or quantity words) is in its sentence. The candidates are the
old-against-new record (`tables/old_vs_new_record.csv`), statistics derived from it as the text states them, and the
fixed setup values; see the script's docstring. The v2 value is the record's v2 value (Phase 6 runs, metres) or the same
statistic computed on v2 values. On AP, v2 D/D₀ uses D₀,A; the note gives D₀,B. The ratio is v2/v1 in consistent units:
AP values that v1 printed in feet-based units are converted to metres first. `no match` rows carry, in the note, the v1
values within 5 % if there are any. Those values are not used as matches.

**Discrepancies found while tying, outside the six claims (reported, not resolved):**

1. **Figure 4 MSE labels are exchanged between the columns.** The left column (Gabilan Mesa surfaces, and the k_s and θ
   of GM2) prints MSE 373.0 and 226.0. Those are AP1's v1 AD and partitioned MSEs in ft². The right column (AP1) prints
   82.0 and 27.8, which are GM2's v1 values in m².
2. **Figure 4 values that match no v1 run.** The AD MSEs at measured parameters, 7.32×10⁵ and 1.82×10⁵, match neither
   v1 forward run (AP1 7.23×10⁵ ft², GM2 1.32×10⁵ m²). AP1's partitioned k_s is printed 119.8; v1 has 119.2.
3. **Figure 1, Gabilan Mesa panel.** k_s 2.37, θ 0.302 and R² 0.952 match no v1 slope-area fit of GM2 (2.488, 0.306,
   0.962). The Allegheny panel matches v1 AP1.
4. **Text S1 prints AP1 MSEs in ft² labelled m².** 6.96×10⁵ and 7.01×10⁵ are v1's feet-based values; in m² they are
   6.46×10⁴ and 6.52×10⁴.
5. **Optimizer bounds (section 2.3).** The printed bounds, K 10⁻¹² to 10⁻³ and D 4×10⁻¹⁰ to 10⁻², are not the bounds of
   any one v1 tile. K's lower bound is the AP tiles'. K's upper bound, 10⁻³, is no tile's (AP1, AP3 and AP5 use
   2×10⁻³; the others 2×10⁻²). D's bounds are those of AP1, AP3 and AP5. GM2-GM5 use K 7×10⁻⁷ to 2×10⁻² and D 4×10⁻⁵
   to 5×10⁻². v2 keeps v1's bounds per tile, multiplied by c = 3937/1200 on AP (metres).
6. **Resolution sweep parameters.** Sections 2.6 and 3.3 say the sweep used the calibrated (optimized) K and D. The v1
   notebook (`resolution_study_dinf`) and v2 (`params_source: config`) both run it at the per-tile measured K and D
   (Table 1's measured columns).
7. **"One to two orders of magnitude below field-constrained D"** (Abstract, sections 3.3 and 4.1, Conclusions, Figure 5
   caption). In v2, AD D/D₀ is 0.004-0.033 on GM, which holds. On AP it is 0.15-0.28 (with D₀,A), less than one order of
   magnitude.
8. **Morphology claims** ("reproduces observed landscape morphology", Abstract, sections 1, 3.4 and 4.2, Conclusions).
   The addendum 10 morphology score found that no fitted surface, the linear partitioned optimum included, reproduces the
   observed hillslope form on any tile.
9. **Measurement method.** θ is now one site constant m (AP 0.47, GM 0.31), not a per-tile value. D₀ is measured by
   option (iii) at an 11 m window, not on 10 m and 30 m block means with a 9-point stencil. On AP, D₀ is a bracket. The
   regional concavities 0.292 and 0.54 (from NED grids) are tied to no v1 or v2 run.
10. **Supplement cross-references.** The supplement's introduction cites "Figure 2" and "Figures 3 and 4" where Figures
    S2-S4 are meant.
11. **Source files.** Addendum 10 named `Dobbs_Hilley_JGR-ES_Manuscript_REVISED-tracked.docx`. `papers/` holds only the
    JGR-formatted manuscript (2026-07-20, no tracked changes). The supplement has two tracked insertions, resolved here.
    The supplement first placed in `papers/` today was truncated (zip central directory missing). Steve replaced it with
    the file listed above, which is intact.
"""


COLS = ["document", "location", "printed", "v1", "v2", "ratio", "causes", "source_run", "status", "tile", "quantity", "claims", "source", "context", "note"]


SUPER = str.maketrans("0123456789-−–()", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻⁻⁻⁽⁾")


def pretty(t) -> str:
    """Superscript runs back to superscript characters for display (10^-12 -> 10⁻¹², m^2 -> m²)."""
    return re.sub(r"\^([-−–]?\d+)", lambda q: q.group(1).translate(SUPER), str(t))


def md_escape(s: str) -> str:
    return pretty(s).replace("|", "\\|").replace("\n", " ")


def write(paths: list[str]) -> None:
    rows, labels, claims = build(paths)
    rows = rows[COLS]
    rows["printed"] = rows.printed.map(pretty)
    rows.to_csv(OUT / "corrections_table.csv", index=False)
    labels.to_csv(OUT / "corrections_labels.csv", index=False)
    L = []
    L.append("## Claims that change in substance (addendum 11, 2.3): flagged, not rewritten\n")
    for n, (title, _) in CLAIMS.items():
        cs = [c for c in claims if c["claim"] == n]
        L.append(f"### {n}. {title}\n")
        if not cs:
            L.append("No sentence in either document states this.\n")
        for c in cs:
            L.append(f"- {c['document']}, {c['location']} (paragraph {c['paragraph']}): “{c['sentence']}”")
        imgs = [k for k, ns in CLAIM_IMAGE_CELLS.items() if n in ns]
        if imgs:
            L.append("- Numbers in images under this claim: " + "; ".join(imgs) + " (rows with this claim number in the table).")
        L.append("")
    L.append("## Counts\n")
    cnt = rows.status.value_counts()
    L.append("| status | rows |\n| --- | --- |")
    for s in ("unchanged", "update", "relabel or drop", "reword", "ambiguous", "no match"):
        L.append(f"| {s} | {int(cnt.get(s, 0))} |")
    L.append(f"| **all quantity rows** | **{len(rows)}** |\n")
    L.append(f"Labels not in the table (`corrections_labels.csv`): {len(labels)} numbers: "
             + ", ".join(f"{k} {v}" for k, v in labels.kind.value_counts().items()) + ".\n")
    L.append("## The table\n")
    L.append("v1 and v2 in the units shown; AP v1 values `[v1 raw (feet-based)]` are as v1 printed them (z in US survey feet), and their "
             "ratio is taken after converting v1 to metres. `ambiguous` rows list every candidate, separated by `|`. Claims: the 2.3 "
             "claim numbers whose sentence contains the number.\n")
    L.append("| # | document | location | printed | v1 | v2 | ratio | cause(s) | source run | status | tile / quantity | claims | note |")
    L.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for i, r in enumerate(rows.itertuples(), 1):
        L.append("| " + " | ".join(md_escape(x) for x in (i, r.document, r.location, r.printed, r.v1, r.v2, r.ratio, r.causes, r.source_run, f"`{r.status}`",
                                                          f"{r.tile} {r.quantity}".strip(), r.claims, r.note)) + " |")
    head = HEADER.format(sources="\n".join(source_lines(paths)), n_rows=len(rows), n_labels=len(labels), n_images=int((rows.source != "text").sum()))
    (OUT / "corrections_table.md").write_text(head + "\n" + "\n".join(L) + "\n")
    print(rows.status.value_counts().to_string())
    print(f"{len(rows)} rows, {len(labels)} labels")


if __name__ == "__main__":
    write(sys.argv[1:])
