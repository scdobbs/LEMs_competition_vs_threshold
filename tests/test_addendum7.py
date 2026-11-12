"""Addendum 7: the submit-environment guard (0.4), the slope-area unit regression (2b.5), the D diagnostic by hillslope
class (2.1-2.3) and the per-stage fig3 collector (1.3), on the synthetic tiles."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml

from lemcvt.analyses.optimizer_v2 import refine_partitioned, run_tile_v2
from lemcvt.config import load_tile
from synthetic_tile import make_synthetic_tile, restore_env, set_env


@pytest.fixture(scope="module")
def gm_env(tmp_path_factory):
    info = make_synthetic_tile(tmp_path_factory.mktemp("a7_gm"), tile="GM1", site="GM")
    prev = set_env(info)
    yield info
    restore_env(prev)


@pytest.fixture(scope="module")
def gm_runs(gm_env):
    rec = run_tile_v2("GM1", stages=["ad", "partitioned"], acrit_grid="thin", thin_n=40, save_grids=True, derived_grids=True)
    ref = refine_partitioned("GM1", rec.dir, max_extra=5, run_id="a7_GM1_v2_partitioned-refined", derived_grids=True)
    return rec, ref


def test_render_refuses_without_submit_env(gm_env, monkeypatch):
    from lemcvt.slurm import SubmitEnvironmentError, build_spec, render

    spec = build_spec("custom", ["all"], "lemcvt --help")
    txt = render(spec)
    assert f'LEMCVT_RESULTS:-{gm_env["results"]}' in txt and f'LEMCVT_DATA:-{gm_env["data"]}' in txt
    monkeypatch.setenv("LEMCVT_RESULTS", "")
    with pytest.raises(SubmitEnvironmentError, match="LEMCVT_RESULTS"):
        render(spec)
    monkeypatch.setenv("LEMCVT_RESULTS", str(gm_env["results"]))
    monkeypatch.delenv("LEMCVT_DATA")
    with pytest.raises(SubmitEnvironmentError, match="LEMCVT_DATA"):
        render(spec)


def test_slope_area_model_grids_are_not_converted_twice(tmp_path_factory):
    """An AP synthetic tile stored in ftUS and the same tile stored in metres give the same slope-area fits of the
    model grids (the run writes metres in both cases; before addendum 7 the ftUS run's grids were multiplied by 0.3048)."""
    from lemcvt.analyses.slope_area_fig import MODEL_KEYS, _model_filled_grid
    from lemcvt.measure.slope_area import slope_area_regression_binned

    fits = {}
    for units in ("m", "ftUS"):
        info = make_synthetic_tile(tmp_path_factory.mktemp(f"a7_ap_{units}"), tile="AP1", site="AP", z_units=units)
        cfg_path = info["configs"] / "tiles" / "AP1.yaml"          # both configs declare metre parameters: the same model
        cfg = yaml.safe_load(cfg_path.read_text())
        cfg["model"]["param_z_units"] = "m"
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        prev = set_env(info)
        try:
            rec = run_tile_v2("AP1", stages=["ad", "partitioned"], acrit_grid="thin", thin_n=40, save_grids=True, derived_grids=True)
            td = load_tile("AP1")
            p = json.loads((rec.dir / "params.json").read_text())
            assert p["inputs"]["z_units_working"] == "m"
            z = _model_filled_grid(td, "part", rec.dir)
            ok = np.isfinite(td.Z) & np.isfinite(z)
            jo = p["v2"]["partitioned"]["joint_optimum"]
            mse = float(np.mean((td.Z[ok] - z[ok]) ** 2))
            assert mse < 100 * jo["mse"] + 0.05, f"{units}: filled model grid against the observed: MSE {mse} (joint {jo['mse']})"
            grids = {"orig": td.Z} | {k: _model_filled_grid(td, k, rec.dir) for k in MODEL_KEYS}
            fits[units] = {}
            for key, g in grids.items():          # the figure's regression, with a minimum area the 100 x 100 tile can meet
                Ks, theta, _dbg = slope_area_regression_binned(g, td.A, td.FD, td.dx, min_area=300.0, vertical_interval=10, nbins=12,
                                                               min_per_bin=5, agg="median", require_weighted=True, relax_if_sparse=True)
                fits[units][key] = {"Ks": float(Ks), "theta": float(theta)}
        finally:
            restore_env(prev)
    # AD_no_opt is excluded: its forward run takes the config's K0_ad and D0, which the loader scales with the declared
    # parameter unit, so the two synthetic configs do not describe the same forward model; the fitted grids do.
    for key in ("orig", "AD_opt", "part"):
        for q in ("Ks", "theta"):
            assert fits["ftUS"][key][q] == pytest.approx(fits["m"][key][q], rel=1e-6, abs=1e-9), (key, q, fits["ftUS"][key], fits["m"][key])


def test_d_diagnostic_classes_refits_and_rule(gm_env, gm_runs):
    from lemcvt.analyses.d_diagnostic import run_d_diagnostic

    _rec, ref = gm_runs
    out = run_d_diagnostic("GM1", thin_n=30, refine_max=5, A_roll=30.0, A_c=200.0, run_id="a7_ddiag")
    p = json.loads((out.dir / "params.json").read_text())
    c = p["curvature_by_class"]
    cl = c["classes"]
    assert cl["divergent"]["n"] + cl["hollow"]["n"] == c["all_hillslope"]["n"]
    assert cl["hilltop"]["n"] <= cl["divergent"]["n"]
    assert abs(cl["divergent"]["ssr_share_of_hillslope"] + cl["hollow"]["ssr_share_of_hillslope"] - 1) < 1e-9
    # the model's own 3-cell -laplacian on hillslope cells is U/D_fit (interior cells of the linear model)
    jo = p["setup"]["joint_optimum"]
    assert c["all_hillslope"]["model_lap3"]["median"] == pytest.approx(p["model"]["U"] / jo["D"], rel=1e-6)
    # the 'all' mask reproduces the Phase 6 profile row for row at the common A_crit values
    prof_ref = pd.read_parquet(ref.dir / "acrit_profile.parquet")
    prof = pd.read_parquet(out.dir / "acrit_profile.parquet")
    a = prof[prof["mask"] == "all"].set_index("A_crit")
    common = a.index.intersection(prof_ref.set_index("A_crit").index)
    assert len(common) >= 10
    r = prof_ref.set_index("A_crit").loc[common]
    assert np.allclose(a.loc[common, "mse_joint"].to_numpy(), r["mse_joint"].to_numpy(), rtol=1e-10)
    assert np.allclose(a.loc[common, "D_opt"].to_numpy(), r["D_opt"].to_numpy(), rtol=1e-10)
    # every mask keeps the channel cells of its A_crit: the misfit cell count is class + channel
    d = prof[prof["mask"] == "divergent"].iloc[0]
    assert d["n_misfit"] >= cl["divergent"]["n"] and d["n_misfit"] <= cl["divergent"]["n"] + d["n_channel"]
    v = p["verdict"]
    assert set(v) >= {"a_pass", "b_pass", "c_pass", "all_pass", "D_divergent", "D_hollow", "D_fit"}
    assert v["all_pass"] == (v["a_pass"] and v["b_pass"] and v["c_pass"])
    assert (out.dir / "GM1_hillslope_classes.tif").exists()


def test_fig3_collects_per_stage_runs(gm_env, gm_runs):
    from lemcvt.figures.fig3_deltas import collect_deltas

    df = collect_deltas()
    assert set(df.stage) >= {"ad", "joint"} and (df.tile == "GM1").all() and (df.D0_end == "A").all()
    assert np.isfinite(df.dK).all() and np.isfinite(df.dD).all()
