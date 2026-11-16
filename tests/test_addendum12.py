"""Addendum 12: the greedy basin split, the fixed-parameter reductions against direct forward solves, the held-out
scores, the AD variants, and the m override of the v2 optimizer (step a12_mt), on the synthetic tile."""

import numpy as np
import pytest

from lemcvt.analyses.optimizer_v2 import refine_partitioned, run_tile_v2
from lemcvt.analyses.sensitivity import _fit, _q, greedy_basin_halves, run_equal_ad, run_equal_part
from lemcvt.analyses.tile_pipeline import mse
from lemcvt.config import load_tile
from lemcvt.models.ad import advection_diffusion_model
from lemcvt.models.partitioned import partitioned_threshold_model
from lemcvt.optimize.v2 import ADProfiler, ad_profile, box_quadratic_min
from lemcvt.results import find_runs
from synthetic_tile import make_synthetic_tile, restore_env, set_env


@pytest.fixture(scope="module")
def gm_env(tmp_path_factory):
    info = make_synthetic_tile(tmp_path_factory.mktemp("a12_gm"), tile="GM1", site="GM")
    prev = set_env(info)
    yield info
    restore_env(prev)


def _mse_on(Z, Z0, keep):
    ok = keep & np.isfinite(Z) & np.isfinite(Z0)
    return float(np.mean((Z[ok] - Z0[ok]) ** 2))


def test_greedy_split_is_balanced_and_partitions_edge_basins(gm_env):
    td = load_tile("GM1")
    half, info = greedy_basin_halves(td.Z, td.A, td.FD, td.dx, td.filled.dy, A_c=200.0)
    assert set(np.unique(half)) <= {0, 1, 2}
    n1, n2 = info["n_cells_half1"], info["n_cells_half2"]
    assert n1 > 0 and n2 > 0
    sizes = [b["n_cells"] for b in info["basins"]]
    assert sizes == sorted(sizes, reverse=True)
    # greedy to the lighter half: the final imbalance is at most the largest basin
    assert abs(n1 - n2) <= sizes[0]
    assert sum(sizes) == n1 + n2
    assert info["n_cells_half1"] + info["n_cells_half2"] + info["n_cells_interior_outlet_basins"] + info["n_cells_no_basin"] == info["n_cells_valid"]


def test_fixed_parameter_fits_match_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(20):
        M = rng.normal(size=(2, 2))
        G = M @ M.T + 0.1 * np.eye(2)
        t = {"G00": G[0, 0], "G01": G[0, 1], "G11": G[1, 1], "g0": rng.normal(), "g1": rng.normal(), "rr": 10.0}
        lo, hi = np.array([-2.0, -2.0]), np.array([2.0, 2.0])
        xs = np.linspace(lo[0], hi[0], 4001)
        y0 = float(rng.uniform(-1, 1))
        x, y, _ = _fit(t, "fix_y", lo, hi, 0.0, y0)
        assert _q(t, x, y) <= min(_q(t, v, y0) for v in xs) + 1e-9
        x0 = float(rng.uniform(-1, 1))
        x, y, _ = _fit(t, "fix_x", lo, hi, x0, None)
        assert _q(t, x, y) <= min(_q(t, x0, v) for v in xs) + 1e-9
        x, y, _ = _fit(t, "joint", lo, hi, 0.0, None)
        v, _w = box_quadratic_min(G, np.array([t["g0"], t["g1"]]), lo, hi)
        assert (x, y) == (v[0], v[1])


def test_equal_part_matches_direct_solves(gm_env):
    td = load_tile("GM1")
    mp = td.cfg.model
    # a Phase 6-like refined run first, so that the union with the stored rows is exercised
    r0 = run_tile_v2("GM1", stages=["partitioned"], thin_n=30, save_grids=False)
    refine_partitioned("GM1", r0.dir, fill_interval=0.05, derived_grids=False)
    rec = run_equal_part("GM1", thin_n=30, fill_max=40)
    res = rec.params["result"]
    assert res["phase6_check"]["available"]
    assert res["phase6_check"]["max_rel_new_vs_phase6_P4"] < 1e-10
    assert res["phase6_check"]["max_rel_stored_terms_vs_stored_joint"] < 1e-10
    U = mp.U
    for name in ("P1_A", "P2_A", "P3", "P4"):
        o = res["insample"][name]
        Z, _mask = partitioned_threshold_model(td.Z, td.A, td.FD, td.dx, mp.m, U, o["K"])(o["A_crit"], o["D"])
        assert mse(Z, td.Z) == pytest.approx(o["mse"], rel=1e-8), name
    assert res["insample"]["P2_A"]["D"] == pytest.approx(mp.D0, rel=1e-12)
    assert res["insample"]["P3"]["K"] == pytest.approx(mp.K0_partitioned, rel=1e-12)
    assert res["insample"]["P4"]["mse"] <= min(res["insample"][n]["mse"] for n in ("P1_A", "P2_A", "P3")) + 1e-12
    # held-out: the score is the direct MSE on the other half's cells at the parameters fitted on one half
    half, _ = greedy_basin_halves(td.Z, td.A, td.FD, td.dx, td.filled.dy)
    for name in ("P4", "P2_A"):
        f1 = res["heldout"][name]["fit_h1"]
        Z, _mask = partitioned_threshold_model(td.Z, td.A, td.FD, td.dx, mp.m, U, f1["K"])(f1["A_crit"], f1["D"])
        assert _mse_on(Z, td.Z, half == 2) == pytest.approx(f1["score_h2"], rel=1e-8)
        assert _mse_on(Z, td.Z, half == 1) == pytest.approx(f1["mse"], rel=1e-8)


def test_equal_ad_matches_profile_and_direct_solves(gm_env):
    td = load_tile("GM1")
    mp = td.cfg.model
    rec = run_equal_ad("GM1", n_grid=40)
    res = rec.params["result"]
    ad = advection_diffusion_model(td.Z, td.A, td.FD, td.dx, mp.m)
    from lemcvt.models.common import extract_grid_data

    gd = extract_grid_data(td.Z, td.A, td.FD, td.dx)
    ref = ad_profile(ADProfiler(ad, td.Z[gd["rev_j"], gd["rev_i"]], mp.U), mp.K_bounds, mp.D_bounds, n_grid=40)
    assert res["insample"]["AD"]["mse"] == pytest.approx(ref["optimum"]["mse"], rel=1e-10)
    for name in ("AD", "A1", "A2_A"):
        o = res["insample"][name]
        assert mse(ad(mp.U, o["K"], o["D"]), td.Z) == pytest.approx(o["mse"], rel=1e-8), name
    assert res["insample"]["A1"]["K"] == pytest.approx(mp.K0_ad, rel=1e-12)
    assert res["insample"]["A2_A"]["D"] == pytest.approx(mp.D0, rel=1e-12)
    assert res["insample"]["AD"]["mse"] <= min(res["insample"]["A1"]["mse"], res["insample"]["A2_A"]["mse"]) * (1 + 1e-9)
    half, _ = greedy_basin_halves(td.Z, td.A, td.FD, td.dx, td.filled.dy)
    f1 = res["heldout"]["AD"]["fit_h1"]
    Z = ad(mp.U, f1["K"], f1["D"])
    assert _mse_on(Z, td.Z, half == 2) == pytest.approx(f1["score_h2"], rel=1e-8)


def test_model_override_runs_are_not_phase6_runs(gm_env):
    n_before = len(find_runs("tile_run", "GM1"))
    ov = {"m": 0.30, "K0_ad": 2e-4, "K0_partitioned": 2e-4}
    rec = run_tile_v2("GM1", stages=["partitioned"], thin_n=20, save_grids=False, model_override=ov, step="a12_mt")
    assert rec.params["model"]["m"] == 0.30 and rec.params["model"]["K0_partitioned"] == 2e-4
    assert rec.params["model_override"] == ov
    rec2 = refine_partitioned("GM1", rec.dir, fill_interval=0.05, derived_grids=False, step="a12_mt")
    assert rec2.params["model"]["m"] == 0.30 and rec2.params["model_override"] == ov
    assert len(find_runs("tile_run", "GM1")) == n_before
    assert [r.name for r in find_runs("a12_mt", "GM1")] == [rec.dir.name, rec2.dir.name]
    with pytest.raises(ValueError):
        run_tile_v2("GM1", stages=["partitioned"], thin_n=20, save_grids=False, model_override={"D0": 1.0}, step="a12_mt")
