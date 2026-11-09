"""Phase 4 (directive 4.4, addendum 3 section 5.3): the reductions behind optimizer v2, proved on synthetic tiles."""

import numpy as np
import pytest

from lemcvt.models.ad import advection_diffusion_model
from lemcvt.models.common import extract_grid_data
from lemcvt.models.partitioned import partitioned_threshold_model
from lemcvt.optimize.v2 import (
    ADProfiler,
    PartitionedProfiler,
    ad_profile,
    box_quadratic_min,
    exponent_profile,
    partitioned_profile,
    thin_log_uniform,
    unique_areas,
)
from synthetic_tile import hydro_from_padded, synthetic_dem

U = 1.47e-4
K_B, D_B = (7e-7, 2e-2), (4e-5, 5e-2)


@pytest.fixture(scope="module")
def tile():
    z = synthetic_dem(60, 60, 1.0, seed=7)
    zf, fd, area = hydro_from_padded(z, 1.0)
    return {"Z": zf, "A": area, "FD": fd, "dx": 1.0, "relief": float(np.nanmax(zf) - np.nanmin(zf))}


def _flat(t, Z):
    gd = extract_grid_data(t["Z"], t["A"], t["FD"], t["dx"])
    return Z[gd["rev_j"], gd["rev_i"]]


class TestPartitionedReduction:
    """a + (U/K) b + (U/D) c equals the direct forward solve to 1e-10 relative, for random (K, D) and three m."""

    @pytest.mark.parametrize("m", [0.35, 0.45, 0.60])
    def test_reduction_matches_direct(self, tile, m):
        t = tile
        prof = PartitionedProfiler(t["Z"], t["A"], t["FD"], t["dx"], U, m_values=[m])
        rng = np.random.default_rng(int(m * 100))
        for A_crit in (150.0, 400.0):
            red = prof.reduce(A_crit, m)
            for _ in range(3):
                K = 10 ** rng.uniform(np.log10(K_B[0]), np.log10(K_B[1]))
                D = 10 ** rng.uniform(np.log10(D_B[0]), np.log10(D_B[1]))
                Z_direct, mask = partitioned_threshold_model(t["Z"], t["A"], t["FD"], t["dx"], m, U, K)(A_crit, D)
                Z_red = prof.to_grid(red.Z_flat(K, D))
                np.testing.assert_array_equal(mask, prof.to_grid(red.channel_mask.astype(float)) == 1.0)
                ok = np.isfinite(Z_direct)
                assert np.all(np.isfinite(Z_red[ok]))
                scale = np.nanmax(np.abs(Z_direct))
                assert np.nanmax(np.abs(Z_red - Z_direct)) / scale < 1e-10
                assert np.nanmax(np.abs(Z_red - Z_direct)) / t["relief"] < 1e-10

    def test_mse_terms_match_direct_mse(self, tile):
        t = tile
        prof = PartitionedProfiler(t["Z"], t["A"], t["FD"], t["dx"], U, m_values=[0.4])
        red = prof.reduce(300.0, 0.4)
        terms = red.mse_terms(prof.Z_obs_flat)
        for K, D in ((2e-5, 1e-2), (1e-4, 3e-3)):
            Zd, _ = partitioned_threshold_model(t["Z"], t["A"], t["FD"], t["dx"], 0.4, U, K)(300.0, D)
            mse_direct = float(np.nanmean((Zd - t["Z"]) ** 2))
            assert red.mse_from_terms(terms, U / K, U / D) == pytest.approx(mse_direct, rel=1e-9)


class TestBoxQuadratic:

    def test_against_lsq_linear(self):
        from scipy.optimize import lsq_linear

        rng = np.random.default_rng(3)
        for _ in range(40):
            M = rng.normal(size=(50, 2)) * np.array([10.0, 0.1])
            r = rng.normal(size=50) * 5 + M @ rng.uniform(-3, 3, 2)
            lo = rng.uniform(-2, 0, 2)
            hi = lo + rng.uniform(0.5, 4, 2)
            G, g = M.T @ M, M.T @ r
            v, _where = box_quadratic_min(G, g, lo, hi)
            ref = lsq_linear(M, r, bounds=(lo, hi), tol=1e-14).x
            f = lambda q: float(q @ G @ q - 2 * g @ q)  # noqa: E731
            assert np.all(v >= lo - 1e-12) and np.all(v <= hi + 1e-12)
            assert f(v) <= f(ref) + 1e-9 * max(1.0, abs(f(ref)))


class TestBruteForce:
    """On a 60 x 60 crop a brute-force grid over (A_crit, K, D) finds no lower MSE than v2 (directive 4.4 c)."""

    def test_no_lower_mse(self, tile):
        t = tile
        m = 0.4
        ua = unique_areas(t["A"], 50.0, 2000.0)
        grid = thin_log_uniform(ua, 40)
        prof = PartitionedProfiler(t["Z"], t["A"], t["FD"], t["dx"], U, m_values=[m])
        res = partitioned_profile(prof, grid, m, K_B, D_B, 2e-5, 1e-2, log_every=0)
        v2_joint = res["joint_optimum"]["mse"]
        v2_stage2 = res["stage2_optimum"]["mse"]
        Ks = np.logspace(np.log10(K_B[0]), np.log10(K_B[1]), 13)
        Ds = np.logspace(np.log10(D_B[0]), np.log10(D_B[1]), 13)
        best = np.inf
        best_s2 = np.inf
        for a_c in grid:
            for K in np.r_[Ks, 2e-5]:
                f = partitioned_threshold_model(t["Z"], t["A"], t["FD"], t["dx"], m, U, K)
                for D in np.r_[Ds, 1e-2]:
                    Z, _ = f(a_c, D)
                    v = float(np.nanmean((Z - t["Z"]) ** 2))
                    best = min(best, v)
                    if K == 2e-5 and D == 1e-2:
                        best_s2 = min(best_s2, v)
        assert v2_joint <= best * (1 + 1e-9)
        assert v2_stage2 == pytest.approx(best_s2, rel=1e-9)
        # the joint optimum's own (K, D) reproduce its MSE by a direct solve
        jo = res["joint_optimum"]
        Z, _ = partitioned_threshold_model(t["Z"], t["A"], t["FD"], t["dx"], m, U, jo["K"])(jo["A_crit"], jo["D"])
        assert float(np.nanmean((Z - t["Z"]) ** 2)) == pytest.approx(jo["mse"], rel=1e-9)


class TestExponentProfile:
    """Addendum 3 section 5.3: the reduction holds at three values of m (one LU per A_crit, one solve per m)."""

    def test_three_m_values(self, tile):
        t = tile
        ms = np.array([0.3, 0.45, 0.6])
        prof = PartitionedProfiler(t["Z"], t["A"], t["FD"], t["dx"], U, m_values=ms)
        res = exponent_profile(prof, np.array([200.0, 500.0]), ms, K_B, D_B, 2e-5, 1e-2, log_every=0)
        assert res["n_evaluations"] == 6
        for row in res["rows"]:
            Z, _ = partitioned_threshold_model(t["Z"], t["A"], t["FD"], t["dx"], row["m"], U, row["K_opt"])(row["A_crit"], row["D_opt"])
            assert float(np.nanmean((Z - t["Z"]) ** 2)) == pytest.approx(row["mse_joint"], rel=1e-9)
        F = prof.factor(500.0)
        for m in ms:
            red = prof.reduce(500.0, float(m), factored=F)
            Zd, _ = partitioned_threshold_model(t["Z"], t["A"], t["FD"], t["dx"], float(m), U, 3e-5)(500.0, 7e-3)
            assert np.nanmax(np.abs(prof.to_grid(red.Z_flat(3e-5, 7e-3)) - Zd)) / np.nanmax(np.abs(Zd)) < 1e-10
        assert 0.3 <= res["optimum"]["m"] <= 0.6 and res["contour_0.01"]["n_points"] >= 1


class TestADReduction:
    """h(r) + (U/D) g(r) equals the direct AD solve to 1e-10 relative; the r-profile finds the (K, D) optimum."""

    def test_reduction_matches_direct(self, tile):
        t = tile
        ad = advection_diffusion_model(t["Z"], t["A"], t["FD"], t["dx"], 0.4)
        prof = ADProfiler(ad, _flat(t, t["Z"]), U)
        rng = np.random.default_rng(5)
        for _ in range(4):
            K = 10 ** rng.uniform(np.log10(K_B[0]), np.log10(K_B[1]))
            D = 10 ** rng.uniform(np.log10(D_B[0]), np.log10(D_B[1]))
            red = prof.reduce(K / D)
            Z_direct = ad(U, K, D)
            Z_red = red["h"] + (U / D) * red["g"]
            assert np.nanmax(np.abs(_flat(t, Z_direct) - Z_red)) / np.nanmax(np.abs(Z_direct)) < 1e-10
            assert prof.mse_at(red, U / D) == pytest.approx(float(np.nanmean((Z_direct - t["Z"]) ** 2)), rel=1e-9)

    def test_profile_beats_coarse_grid(self, tile):
        t = tile
        ad = advection_diffusion_model(t["Z"], t["A"], t["FD"], t["dx"], 0.4)
        prof = ADProfiler(ad, _flat(t, t["Z"]), U)
        res = ad_profile(prof, K_B, D_B, n_grid=40, log_every=0)
        opt = res["optimum"]
        assert K_B[0] <= opt["K"] <= K_B[1] and D_B[0] <= opt["D"] <= D_B[1]
        best = np.inf
        for K in np.logspace(np.log10(K_B[0]), np.log10(K_B[1]), 12):
            for D in np.logspace(np.log10(D_B[0]), np.log10(D_B[1]), 12):
                best = min(best, float(np.nanmean((ad(U, K, D) - t["Z"]) ** 2)))
        assert opt["mse"] <= best * (1 + 1e-9)
        assert "r_min" in res["identifiability"]["level_0.01"]


def test_unique_and_thin():
    A = np.array([[1.0, 2.0, 2.0, 50.0], [100.0, 150.0, 150.0, 1e6]])
    ua = unique_areas(A, 2.0, 200.0)
    np.testing.assert_array_equal(ua, [2.0, 50.0, 100.0, 150.0])
    th = thin_log_uniform(np.arange(100.0, 100000.0), 2000)
    assert 1900 <= th.size <= 2000 and th[0] == 100.0 and th[-1] == 99999.0
    assert np.all(np.diff(th) > 0)
