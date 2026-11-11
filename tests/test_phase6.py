"""Addendum 5 (Phase 6) machinery on the 100 x 100 synthetic tile: the 5 % interval fill of the partitioned profile,
the nonlinear joint (A_crit, D) profile, the D-infinity joint profile, the AD no-opt forward run, and the config loader
with a `measured` block and the v1 model block for feet runs."""

import json

import numpy as np
import pytest
import yaml

from lemcvt.analyses.optimizer_v2 import refine_partitioned, run_dinf_joint, run_nonlinear_joint, run_tile_v2
from lemcvt.config import load_tile_config
from synthetic_tile import make_synthetic_tile, restore_env, set_env


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    info = make_synthetic_tile(tmp_path_factory.mktemp("p6"), tile="GM1", site="GM")
    prev = set_env(info)
    yield info
    restore_env(prev)


@pytest.fixture(scope="module")
def partitioned_run(env):
    rec = run_tile_v2("GM1", stages=["partitioned"], acrit_grid="thin", thin_n=60, save_grids=False)
    return rec.dir


def test_fill_interval_evaluates_every_unique_area_in_the_bands(env, partitioned_run):
    rec = refine_partitioned("GM1", partitioned_run, max_extra=30, fill_interval=0.05)
    p = json.loads((rec.dir / "params.json").read_text())
    out = p["v2"]["partitioned"]
    fill = out["interval_fill"]
    assert fill["requested"] == 0.05
    assert "mse_joint" in fill and "mse_stage2" in fill and fill["n_unique_evaluated"] >= 0
    import pandas as pd

    prof = pd.read_parquet(rec.dir / "acrit_profile.parquet")
    assert prof.A_crit.is_monotonic_increasing and prof.A_crit.is_unique
    assert out["n_evaluations"] == prof.shape[0]
    # every unique drainage area inside the joint 5 % intervals of the refined profile is now on the grid
    from lemcvt.config import load_tile
    from lemcvt.optimize.v2 import unique_areas

    td = load_tile("GM1")
    ua = unique_areas(td.A, 1e2, 1e5)
    for lo, hi in out["joint_identifiability"]["level_0.05"]["intervals"]:
        inside = ua[(ua >= lo) & (ua <= hi)]
        assert np.all(np.isin(inside, prof.A_crit.to_numpy()))
    assert out["joint_optimum"]["mse"] <= out["refinement"]["joint_before"]["mse"] + 1e-12
    assert "every unique area" in out["grid"]["mode"]


def test_nonlinear_joint_profile(env, partitioned_run):
    rec = refine_partitioned("GM1", partitioned_run, max_extra=10, run_id="p6_refined_for_nlj")
    out = run_nonlinear_joint("GM1", source_run=rec.dir, n_acrit=4, n_d=3, d_range=(5e-3, 5e-2), save_grids=False)
    p = json.loads((out.dir / "params.json").read_text())
    v = p["v2"]["nonlinear_joint"]
    assert v["n_evaluations"] == 12
    assert len(v["setup"]["A_crit_grid"]) == 4 and len(v["setup"]["D_grid"]) == 3
    assert v["setup"]["K"] == pytest.approx(json.loads((rec.dir / "params.json").read_text())["v2"]["partitioned"]["joint_optimum"]["K"])
    c = v["contours"]
    assert c["contour_0.01"]["n_points"] >= 1 and c["contour_0.05"]["n_points"] >= c["contour_0.01"]["n_points"]
    o = v["optimum"]
    assert o["mse"] == c["mse_min"]
    assert "frac_hillslope_gt_0.5Sc" in o and "picard_iterations" in o
    assert v["n_hit_cap"] <= v["n_evaluations"]
    # the K check variant carries its factor
    out2 = run_nonlinear_joint("GM1", source_run=rec.dir, k_factor=1.05, n_acrit=2, n_d=2, d_range=(5e-3, 5e-2), save_grids=False)
    p2 = json.loads((out2.dir / "params.json").read_text())
    assert p2["setup"]["k_factor"] == 1.05 and p2["setup"]["K"] == pytest.approx(1.05 * v["setup"]["K_joint"])


def test_dinf_joint_profile_reduction_is_exact(env):
    out = run_dinf_joint("GM1", thin_n=25, refine_max=10, save_grids=False)
    p = json.loads((out.dir / "params.json").read_text())
    v = p["v2"]["dinf_joint"]
    assert v["direct_check"]["joint"]["rel_diff"] < 1e-8
    assert v["grid"]["n_used"] == v["n_evaluations"] >= 25
    assert v["joint_optimum"]["where"] in ("interior", "x_lo", "x_hi", "y_lo", "y_hi", "corner")
    assert v["channel_monotonicity_at_optimum"]["n_channel"] > 0


def test_ad_stage_writes_the_no_opt_forward(env):
    rec = run_tile_v2("GM1", stages=["ad"], save_grids=False)
    p = json.loads((rec.dir / "params.json").read_text())
    keys = [f["key"] for f in p["forward_runs"]]
    assert keys[:2] == ["ad_no_opt", "ad_opt"]
    no = next(f for f in p["forward_runs"] if f["key"] == "ad_no_opt")
    assert no["params"]["K"] == pytest.approx(p["model"]["K0_ad"]) and no["params"]["D"] == pytest.approx(p["model"]["D0"])


def test_config_measured_block_and_v1_model_block(env):
    path = env["configs"] / "tiles" / "GM1.yaml"
    d = yaml.safe_load(path.read_text())
    v1_model = dict(d["model"])
    d["measured"] = {"m": 0.31, "ks": 4.0, "C_ht": 1.6e-2, "C_ht_B": 1.2e-2, "z_units": "m", "provenance": {"run": "test"}}
    d["model"] = dict(v1_model) | {"m": 0.31, "param_z_units": "m", "nonlinear_max_iter": 100, "D0_B": "U/C_ht_B"}
    d["v1_reference"] = {"stages": {}, "model": dict(v1_model) | {"param_z_units": "m"}}
    path.write_text(yaml.safe_dump(d, sort_keys=False))
    cfg = load_tile_config("GM1")
    assert cfg.model.m == 0.31
    assert cfg.model.K0_partitioned == pytest.approx(1.47e-4 / 4.0)
    assert cfg.model.D0 == pytest.approx(1.47e-4 / 1.6e-2)
    assert cfg.model.nonlinear_max_iter == 100
    assert cfg.measured["ks"] == 4.0
    assert cfg.model.D0_B == pytest.approx(1.47e-4 / 1.2e-2)
    from lemcvt.config import apply_d0_variant

    mb = apply_d0_variant(load_tile_config("GM1").model, "B")
    assert mb.D0 == pytest.approx(1.47e-4 / 1.2e-2) and mb.d0_variant == "B"
    feet = load_tile_config("GM1", reproduce_v1_units=True)
    assert feet.model.m == v1_model["m"]
    assert feet.model.K0_partitioned == pytest.approx(1.47e-4 / 5.0)      # the v1 measured_v1 block
    assert feet.model.nonlinear_max_iter == 5
    # the config writer builds a loadable Phase 6 config from a measured-inputs record
    from lemcvt.analyses.phase6 import build_tile_config

    inputs = {"site_m": {"values": {"GM": 0.31}, "provenance": {"GM": {"interval_1pct": {"intervals": [[0.29, 0.33]]}, "S_min": 5.07}}},
              "tiles": {"GM1": {"ks": {"1000": {"ks": 4.0, "n_basins": 9, "n_eff_centred": 2.8, "ks_ci68": [3.5, 4.7]},
                                       "10000": {"ks": 3.5, "n_basins": 1, "n_eff_centred": 1.0}},
                                "option_iii": {"C_ht_at_L_f": 1.6e-2, "L_f_m": 11.0, "n_pixels": 300, "n_segments": 40, "C_ht_ci68_at_L_f": [1.5e-2, 1.7e-2],
                                               "D0_ci68": [8.6e-3, 9.8e-3], "test": {"pass": True, "rule": "r", "n_windows_tested": 2, "windows_tested_nominal_m": [11, 15]},
                                               "rows": [{"L_nominal_m": 11, "L_m": 11.0, "C_ht": 1.6e-2}, {"L_nominal_m": 15, "L_m": 15.0, "C_ht": 1.4e-2, "ratio_to_L_f": 0.875}]}}},
              "sites": {"GM": {"ks_site_pooled": {"1000": {"ks": 4.0}, "10000": {"ks": 3.5}}, "option_iii_test": {"pass": True}}}}
    cfg_dict, header = build_tile_config("GM1", inputs, {"run": "r", "git_hash": "g"}, None, None)
    assert cfg_dict["model"]["param_z_units"] == "m" and cfg_dict["model"]["nonlinear_max_iter"] == 100
    assert cfg_dict["measured"]["K0"] == pytest.approx(1.47e-4 / 4.0)
    assert cfg_dict["measured"]["L_w_m"] == 15 and cfg_dict["measured"]["D0_B"] == pytest.approx(1.47e-4 / 1.4e-2)
    assert "model" in cfg_dict["v1_reference"]
    path.write_text(header + yaml.safe_dump(cfg_dict, sort_keys=False))
    cfg2 = load_tile_config("GM1")
    assert cfg2.model.K0_partitioned == pytest.approx(1.47e-4 / 4.0) and cfg2.model.D0 == pytest.approx(1.47e-4 / 1.6e-2)
    assert cfg2.model.D0_B == pytest.approx(1.47e-4 / 1.4e-2)


def test_stage2_at_D0_B_uses_the_same_reductions(env):
    """Addendum 6, 5.2: with D0_B in the config the partitioned stage carries the stage-2 profile at D0_B too."""
    rec = run_tile_v2("GM1", stages=["partitioned"], acrit_grid="thin", thin_n=30, save_grids=False)
    p = json.loads((rec.dir / "params.json").read_text())
    v = p["v2"]["partitioned"]
    assert p["model"]["D0_B"] is not None
    assert "stage2_B_optimum" in v and v["stage2_B_optimum"]["D"] == pytest.approx(p["model"]["D0_B"])
    import pandas as pd

    prof = pd.read_parquet(rec.dir / "acrit_profile.parquet")
    assert "mse_stage2_B" in prof.columns and np.isfinite(prof.mse_stage2_B).all()
    assert "acrit_D0B" in p["stages_results"]
    ref = refine_partitioned("GM1", rec.dir, max_extra=10, fill_interval=0.05)
    q = json.loads((ref.dir / "params.json").read_text())
    assert "stage2_B_optimum" in q["v2"]["partitioned"] and "mse_stage2_B" in q["v2"]["partitioned"]["interval_fill"]
    assert any(f["key"] == "acrit_opt_D0B" for f in q["forward_runs"])
    assert any(name.endswith("_filled.tif") and "a_crit-k-d-opt" in name for name in json.loads((ref.dir / "run.json").read_text())["grids"])


def test_nonlinear_joint_split_and_merge(env):
    """The split parts of the (A_crit, D) grid merge into the same rows as one unsplit run."""
    from lemcvt.analyses.optimizer_v2 import merge_nonlinear_joint

    rec = run_tile_v2("GM1", stages=["partitioned"], acrit_grid="thin", thin_n=40, save_grids=False)
    ref = refine_partitioned("GM1", rec.dir, max_extra=5, run_id="p6_refined_for_split")
    full = run_nonlinear_joint("GM1", source_run=ref.dir, n_acrit=4, n_d=2, d_range=(5e-3, 5e-2), save_grids=False, run_id="p6_nlj_full")
    for k in range(2):
        run_nonlinear_joint("GM1", source_run=ref.dir, n_acrit=4, n_d=2, d_range=(5e-3, 5e-2), save_grids=False, split=(k, 2), run_id=f"p6_nlj_split{k}")
    merged = merge_nonlinear_joint("GM1", save_grids=False, run_id="p6_nlj_merged")
    import pandas as pd

    a = pd.read_parquet(full.dir / "nonlinear_joint_profile.parquet").sort_values(["A_crit", "D"]).reset_index(drop=True)
    b = pd.read_parquet(merged.dir / "nonlinear_joint_profile.parquet").sort_values(["A_crit", "D"]).reset_index(drop=True)
    assert len(a) == len(b) == 8
    assert np.allclose(a.mse.to_numpy(), b.mse.to_numpy(), rtol=1e-12)
    pf = json.loads((full.dir / "params.json").read_text())["v2"]["nonlinear_joint"]
    pm = json.loads((merged.dir / "params.json").read_text())["v2"]["nonlinear_joint"]
    assert pm["optimum"]["A_crit"] == pf["optimum"]["A_crit"] and pm["optimum"]["D"] == pf["optimum"]["D"]
    assert "split" not in pm["setup"] and len(pm["setup"]["merged_from"]) == 2
