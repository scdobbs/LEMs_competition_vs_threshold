"""Addendum 10, 5.3: the caption-data CSV helper of the figures and the Table 1 run selection, columns and labels.

Synthetic inputs only (no $LEMCVT_DATA or results needed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lemcvt.figures.common import CENTRE_FORM_TAG, missing_captions, plt, save, write_caption
from lemcvt.tables import table1 as t1

U = 5.5e-5


# ------------------------------------------------------------------------------------------------ caption helper
def test_write_caption_single_and_suffixes(tmp_path):
    p = write_caption(tmp_path, "figA", [{"x": 1.0, "y": 2.0, "run": "r1"}])
    assert p == [tmp_path / "figA.csv"]
    assert pd.read_csv(p[0]).to_dict("records") == [{"x": 1.0, "y": 2.0, "run": "r1"}]
    ps = write_caption(tmp_path, "figB", {"bins": pd.DataFrame({"a": [1, 2]}), "fits": [{"k": 3}]})
    assert sorted(x.name for x in ps) == ["figB_bins.csv", "figB_fits.csv"]
    # an empty table still leaves a CSV, so every figure has its caption file
    (pe,) = write_caption(tmp_path, "figC", [])
    assert pd.read_csv(pe).columns.tolist() == ["note"]


def test_save_writes_caption_beside_images_and_missing_captions(tmp_path):
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    paths = save(fig, tmp_path, "fig1", caption={"panel_a": [{"x": 1}], "panel_b": [{"x": 2}]})
    names = sorted(p.name for p in paths)
    assert names == ["fig1.pdf", "fig1.png", "fig1_panel_a.csv", "fig1_panel_b.csv"]
    fig2, _ = plt.subplots()
    save(fig2, tmp_path, "fig2")            # no caption
    assert missing_captions(tmp_path) == ["fig2"]
    write_caption(tmp_path, "fig2", [{"x": 0}])
    assert missing_captions(tmp_path) == []


def test_centre_form_tag():
    assert CENTRE_FORM_TAG == "(centre form, nonstandard)"


# ------------------------------------------------------------------------------------------------ Table 1
MEASURED = {"m": 0.47, "ks": 12.0, "C_ht": 0.016, "C_ht_B": 0.008, "K0": U / 12.0, "D0": U / 0.016, "D0_B": U / 0.008,
            "provenance": {"ks": {"run": "20261111T000000Z_phase6_inputs_10tiles"}}}


def _model(d0_variant="A", S_c=1.27, D0=None):
    return {"m": 0.47, "U": U, "S_c": S_c, "K0_partitioned": U / 12.0, "D0": D0 if D0 is not None else U / 0.016, "d0_variant": d0_variant}


def _rec(name, v2, model=None, units="m", crop=False, optimizer="v2"):
    p = {"tile": "AP1", "optimizer": optimizer, "inputs": {"z_units_working": units}, "model": model or _model(), "v2": v2}
    return (name, p, {"config": {"crop": [0, 10, 0, 10]} if crop else {}})


AD = {"ad": {"optimum": {"K": 4e-5, "D": 5e-4, "mse": 59.0, "r": 0.08}}}
PART = {"partitioned": {"stage2_optimum": {"A_crit": 823.0, "mse": 289.0}, "stage2_B_optimum": {"A_crit": 1090.0, "mse": 86.4},
                        "joint_optimum": {"A_crit": 6888.0, "K": 4.0e-6, "D": 1.7e-2, "mse": 22.4},
                        "joint_identifiability": {"level_0.01": {"A_crit_min": 6532.0, "A_crit_max": 6958.0, "n_intervals": 1}}}}
NL = {"nonlinear": {"optimum": {"A_crit": 824.0, "mse": 217.8}}}
NLB = {"nonlinear": {"optimum": {"A_crit": 1490.0, "mse": 70.7}}}


def _nlj(A, setup_extra=None, K=4.0e-6):
    return {"nonlinear_joint": {"optimum": {"A_crit": A, "D": 1.65e-2, "K": K, "mse": 22.07},
                                "setup": {"K": K, "k_factor": 1.0, **(setup_extra or {})}}}


def _records():
    return [
        _rec("20261111T01Z_tile_run_AP1_v2_ad", AD),
        _rec("20261111T00Z_tile_run_AP1_v2_ad_feet", {"ad": {"optimum": {"K": 1, "D": 1, "mse": 1}}}, units="ftUS"),
        _rec("20261111T02Z_tile_run_AP1_v2_partitioned", PART),                       # not refined
        _rec("20261111T05Z_tile_run_AP1_v2_partitioned-refined", PART),
        _rec("20261112T05Z_tile_run_AP1_v2_partitioned-refined", PART, crop=True),   # cropped smoke run
        _rec("20261111T13Z_tile_run_AP1_v2_nonlinear", NL),
        _rec("20261111T14Z_tile_run_AP1_v2_nonlinear_D0B", NLB, model=_model("B", D0=U / 0.008)),
        _rec("20261112T17Z_tile_run_AP1_v2_nonlinear_joint_merged", _nlj(6973.0, {"merged_from": {}})),
        _rec("20261112T18Z_tile_run_AP1_v2_nonlinear_joint_Kx0.95_merged", _nlj(1.0, {"k_factor": 0.95})),
        _rec("20261112T19Z_tile_run_AP1_v2_nonlinear_joint_split0of3", _nlj(2.0, {"split": [0, 3]})),
        _rec("20261113T05Z_tile_run_AP1_v2_nonlinear_joint_Sc0.306_merged", _nlj(3.0, {"S_c_override": 0.306}), model=_model(S_c=0.306)),
    ]


def test_select_phase6_runs_picks_the_per_stage_runs():
    runs = t1.select_phase6_runs(_records())
    assert {k: v[0] for k, v in runs.items()} == {
        "ad": "20261111T01Z_tile_run_AP1_v2_ad",
        "partitioned": "20261111T05Z_tile_run_AP1_v2_partitioned-refined",
        "nonlinear": "20261111T13Z_tile_run_AP1_v2_nonlinear",
        "nonlinear_B": "20261111T14Z_tile_run_AP1_v2_nonlinear_D0B",
        "nonlinear_joint": "20261112T17Z_tile_run_AP1_v2_nonlinear_joint_merged",
    }


def test_tile_row_values_runs_and_labels():
    runs = t1.select_phase6_runs(_records())
    row, problems = t1.tile_row("AP1", MEASURED, runs)
    assert problems == []
    assert row["ks"] == 12.0 and row["K0"] == pytest.approx(U / 12.0) and row["D0_B"] == pytest.approx(U / 0.008)
    assert row["ad_K"] == 4e-5 and row["s2_A_crit"] == 823.0 and row["s2B_mse"] == 86.4
    assert (row["joint_A_crit"], row["joint_A_lo1"], row["joint_A_hi1"]) == (6888.0, 6532.0, 6958.0)
    assert row["nl_A_crit"] == 824.0 and row["nl_D"] == pytest.approx(U / 0.016) and row["nl_S_c"] == 1.27
    assert row["nlB_D"] == pytest.approx(U / 0.008)
    assert row["nlj_A_crit"] == 6973.0 and row["nlj_K"] == row["joint_K"]
    assert row["run_measured"] == "20261111T000000Z_phase6_inputs_10tiles"
    assert row["run_nonlinear_joint"].endswith("nonlinear_joint_merged")
    for g in t1.NONLINEAR_GROUPS:
        assert row[f"label_{g}"] == t1.NL_LABEL
    # the long format carries a source run for every value and the label only on the nonlinear groups
    long = t1.to_long(pd.DataFrame([row]))
    assert (long.run != "").all()
    assert set(long[long.model_form != ""].group) == set(t1.NONLINEAR_GROUPS)
    assert set(long.unit[long.column.isin(["ad_D", "joint_D", "nlj_D"])]) == {"m^2/yr"}
    assert set(long.unit[long.column.str.endswith("_K") | (long.column == "K0")]) == {"1/yr"}


def test_tile_row_flags_inconsistent_inputs():
    recs = _records()
    name, p, rj = recs[0]
    p = {**p, "model": {**p["model"], "K0_partitioned": 1e-3}}
    recs[0] = (name, p, rj)
    _row, problems = t1.tile_row("AP1", MEASURED, t1.select_phase6_runs(recs))
    assert len(problems) == 1 and "K0_partitioned" in problems[0]


def test_nl_label_is_the_directive_text():
    assert t1.NL_LABEL == "centre form, D_eff∇²z = −U (nonstandard)"


def test_latex_booktabs_siunitx_and_label():
    row, _ = t1.tile_row("AP1", MEASURED, t1.select_phase6_runs(_records()))
    gm = {k: (np.nan if isinstance(v, float) else v) for k, v in row.items()}
    gm["tile"] = "GM1"
    tex = t1.to_latex(pd.DataFrame([row, gm]))
    for s in (r"\toprule", r"\midrule", r"\bottomrule", "S[table-format=", r"\cmidrule", "(nonstandard)", r"D_\mathrm{eff}\nabla^2 z = -U"):
        assert s in tex
    assert tex.count(r"\tnote{a}") == 2           # both nonlinear groups
    body = [ln for ln in tex.splitlines() if ln.startswith(("AP1", "GM1"))]
    ncols = len(t1.tex_columns())
    assert all(ln.count("&") == ncols - 1 for ln in body)
    assert "{--}" in body[1]
    assert "6888" in body[0] and "4.00e-06" in body[0]
