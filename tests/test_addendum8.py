"""Addendum 8: the E*-R* curve and its linear limit (2.1), hillslope traces (2.1), the slope-dependent threshold with
alpha = 0 reproducing the Phase 6 A_crit model (3.1), the upstream maximum, the basin halves (3.3), the S_c override of
the nonlinear joint profile (2.4) and the ported width tile loader (1.4), on synthetic grids."""

import json
import math

import numpy as np
import pandas as pd
import pytest

from lemcvt.analyses.er_star import fit_S_c, integrate_profile_R, steady_nonlinear_R, trace_to_channel, verify_curve
from lemcvt.analyses.optimizer_v2 import refine_partitioned, run_tile_v2
from lemcvt.analyses.slope_threshold import basin_halves, head_statistic, run_slope_threshold
from lemcvt.config import load_tile
from synthetic_tile import make_synthetic_tile, restore_env, set_env


@pytest.fixture(scope="module")
def gm_env(tmp_path_factory):
    info = make_synthetic_tile(tmp_path_factory.mktemp("a8_gm"), tile="GM1", site="GM")
    prev = set_env(info)
    yield info
    restore_env(prev)


def test_curve_matches_integration_and_linear_limit_is_quarter():
    v = verify_curve()
    assert v["pass"], v["rows"]
    assert v["linear_limit_R_over_E"] == pytest.approx(0.25, rel=1e-5)
    assert float(steady_nonlinear_R(1e4)) == pytest.approx(1.0, rel=1e-3)
    assert integrate_profile_R(0.5) == pytest.approx(float(steady_nonlinear_R(0.5)), rel=1e-7)


def test_fit_S_c_recovers_planted_value():
    rng = np.random.default_rng(0)
    sc = 0.8
    C = rng.uniform(0.005, 0.03, 400)
    L = rng.uniform(10, 120, 400)
    S = sc * steady_nonlinear_R(2 * C * L / sc)
    assert fit_S_c(C, L, S)["S_c"] == pytest.approx(sc, rel=1e-4)


def test_trace_on_parabolic_hillslope_gives_linear_ratio_half():
    # 1-D parabolic ridge z = z0 - (C/2) x^2 (steady linear diffusion), channels at both edges; D8 flows east or west
    nx, ny, dx, C = 201, 5, 1.0, 0.01
    x = (np.arange(nx) - nx // 2) * dx
    Z = np.tile(100 - 0.5 * C * x**2, (ny, 1))
    FD = np.where(x < 0, 16, 1).astype(np.uint8)[None, :].repeat(ny, axis=0)
    FD[:, nx // 2] = 1
    A = np.ones((ny, nx))
    A[:, 0] = A[:, -1] = 1e6
    tr = trace_to_channel(Z, A, FD, dx, dx, np.array([2]), np.array([nx // 2 + 1]), 1e3)
    assert tr["reached"][0]
    ratio = tr["R"][0] / tr["L_H"][0] / (C * tr["L_H"][0])
    x0, x1 = 1.0, float(x[-1])                      # start one cell off the crest, end at the edge channel
    assert tr["L_H"][0] == pytest.approx(x1 - x0)
    assert ratio == pytest.approx(0.5 * (x1**2 - x0**2) / (x1 - x0) ** 2, rel=1e-12)   # -> 0.5 as the start reaches the crest


def test_upstream_max_is_monotone_downstream(gm_env):
    td = load_tile("GM1")
    hs = head_statistic(td.Z, td.A, td.FD, td.dx, td.filled.dy, 2.0)
    M, T = hs["M"], hs["T"]
    ok = np.isfinite(T)
    assert np.all(M[ok] >= T[ok] - 1e-12)
    from lemcvt.hydro.d8 import d8_receiver_arrays

    r, c, g = d8_receiver_arrays(td.FD)
    m = g & np.isfinite(M)
    assert np.all(M[r[m], c[m]] >= M[m] - 1e-12)
    h0 = head_statistic(td.Z, td.A, td.FD, td.dx, td.filled.dy, 0.0)
    assert np.array_equal(h0["M"], np.where(np.isfinite(td.A), td.A, np.nan), equal_nan=True)


def test_basin_halves_partition_cells(gm_env):
    td = load_tile("GM1")
    half, info = basin_halves(td.Z, td.A, td.FD, td.dx, td.filled.dy, A_c=200.0)
    assert set(np.unique(half)) <= {0, 1, 2}
    assert info["n_cells_half1"] > 0 and info["n_cells_half2"] > 0
    assert info["n_cells_half1"] + info["n_cells_half2"] + info["n_cells_no_basin"] == int(np.isfinite(td.A).sum())


def test_alpha0_reproduces_phase6_profile(gm_env):
    rec = run_tile_v2("GM1", stages=["ad", "partitioned"], acrit_grid="thin", thin_n=40, save_grids=False, derived_grids=False)
    ref = refine_partitioned("GM1", rec.dir, max_extra=5, run_id="a8_GM1_v2_partitioned-refined", derived_grids=False)
    prof_ref = pd.read_parquet(ref.dir / "acrit_profile.parquet").set_index("A_crit")
    st = run_slope_threshold("GM1", 0.0, thin_n=40, refine_max=5, run_id="a8_st_a0")
    df = pd.read_parquet(st.dir / "c_profile.parquet").set_index("C")
    common = df.index.intersection(prof_ref.index)
    assert len(common) >= 10
    assert np.allclose(df.loc[common, "mse_joint"].to_numpy(), prof_ref.loc[common, "mse_joint"].to_numpy(), rtol=1e-12)
    assert np.allclose(df.loc[common, "D_opt"].to_numpy(), prof_ref.loc[common, "D_opt"].to_numpy(), rtol=1e-12)
    p = json.loads((st.dir / "params.json").read_text())
    h = p["result"]["holdout"]
    assert math.isfinite(h["mse_heldout_mean"])
    st2 = run_slope_threshold("GM1", 2.0, thin_n=40, refine_max=5, run_id="a8_st_a2")
    p2 = json.loads((st2.dir / "params.json").read_text())
    assert p2["result"]["joint_optimum"]["n_channel"] > 0


def test_width_tile_loader_metres(gm_env):
    from lemcvt.analyses.width.tiles import load_tile as wl

    t = wl("GM", 1)
    td = load_tile("GM1")
    assert np.array_equal(t.z, td.Z) and t.pixel_size == pytest.approx(td.dx)


def test_nonlinear_joint_S_c_override_stop_rule_and_merge(gm_env):
    from lemcvt.analyses.optimizer_v2 import merge_nonlinear_joint, run_nonlinear_joint

    rec = run_tile_v2("GM1", stages=["partitioned"], acrit_grid="thin", thin_n=30, save_grids=False, derived_grids=False)
    ref = refine_partitioned("GM1", rec.dir, max_extra=3, run_id="a8_nlj_src_refined", derived_grids=False)
    parts = [run_nonlinear_joint("GM1", source_run=ref.dir, n_acrit=4, n_d=3, d_range=(1e-3, 5e-2), save_grids=False, split=(k, 2), S_c=0.5,
                                 run_id=f"a8_nlj_part{k}") for k in range(2)]
    for r in parts:
        st = json.loads((r.dir / "params.json").read_text())["v2"]["nonlinear_joint"]["setup"]
        assert st["S_c"] == 0.5 and st["S_c_override"] == 0.5
    m = merge_nonlinear_joint("GM1", S_c=0.5, save_grids=False, run_id="a8_nlj_merged")
    v = json.loads((m.dir / "params.json").read_text())["v2"]["nonlinear_joint"]
    assert v["n_evaluations"] == 12 and 0.0 <= v["frac_hit_cap"] <= 1.0
    assert json.loads((m.dir / "params.json").read_text())["forward_runs"][0]["params"]["S_c"] == 0.5
    with pytest.raises(FileNotFoundError):
        merge_nonlinear_joint("GM1", S_c=0.7, save_grids=False, run_id="a8_nlj_none")
    stop = run_nonlinear_joint("GM1", source_run=ref.dir, n_acrit=4, n_d=3, d_range=(1e-3, 5e-2), save_grids=False, S_c=0.05, cap_stop_frac=-1.0,
                               run_id="a8_nlj_stop")
    vs = json.loads((stop.dir / "params.json").read_text())["v2"]["nonlinear_joint"]
    assert vs["stopped_by_cap_rule"] is not None and vs["n_evaluations"] == 3
