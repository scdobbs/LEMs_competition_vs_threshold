"""Tests for lemcvt.models (ported from v1 tests/test_models.py — locks down current behaviour before refactoring."""


import numpy as np

from lemcvt.models import ad as _ad
from lemcvt.models import partitioned as _p
from lemcvt.models import partitioned_nonlinear as _pn
from lemcvt.models.common import find_boundary_value as _find_boundary_value


def _arrs(dem, area, fd):
    return dem._griddata, area._griddata, fd._griddata, dem._georef_info.dx


def advection_diffusion_model(dem, area, fd, m):
    return _ad.advection_diffusion_model(*_arrs(dem, area, fd), m)


def partitioned_threshold_model(dem, area, fd, m, U, K):
    return _p.partitioned_threshold_model(*_arrs(dem, area, fd), m, U, K)


def partitioned_threshold_model_nonlinear(dem, area, fd, m, U, K):
    return _pn.partitioned_threshold_model_nonlinear(*_arrs(dem, area, fd), m, U, K)


# ── advection_diffusion_model ───────────────────────────────────────────────

class TestAdvectionDiffusionModel:

    def test_returns_correct_shape(self, synthetic_dem, synthetic_area,
                                    synthetic_flow_direction, model_params):
        solver = advection_diffusion_model(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"],
        )
        Z = solver(model_params["U"], model_params["K"], model_params["D"])
        assert Z.shape == synthetic_dem._griddata.shape

    def test_boundary_conditions(self, synthetic_dem, synthetic_area,
                                  synthetic_flow_direction, model_params):
        """Boundary cells should match the input DEM."""
        solver = advection_diffusion_model(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"],
        )
        Z = solver(model_params["U"], model_params["K"], model_params["D"])
        Z0 = synthetic_dem._griddata
        Ny, Nx = Z0.shape

        # Check edges — boundary cells with valid data should be preserved
        for j in range(Ny):
            for i in range(Nx):
                if j == 0 or j == Ny - 1 or i == 0 or i == Nx - 1:
                    if np.isfinite(Z0[j, i]) and np.isfinite(Z[j, i]):
                        np.testing.assert_allclose(Z[j, i], Z0[j, i], rtol=1e-10)

    def test_deterministic(self, synthetic_dem, synthetic_area,
                            synthetic_flow_direction, model_params):
        """Same inputs must produce the same output."""
        solver = advection_diffusion_model(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"],
        )
        Z1 = solver(model_params["U"], model_params["K"], model_params["D"])
        Z2 = solver(model_params["U"], model_params["K"], model_params["D"])
        np.testing.assert_array_equal(Z1, Z2)

    def test_positive_elevation(self, synthetic_dem, synthetic_area,
                                 synthetic_flow_direction, model_params):
        """Elevation should remain physically reasonable (no huge negatives)."""
        solver = advection_diffusion_model(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"],
        )
        Z = solver(model_params["U"], model_params["K"], model_params["D"])
        # All finite values should be > -1000 (we start around 900-1000)
        finite = Z[np.isfinite(Z)]
        assert np.all(finite > -1000), f"Min elevation: {finite.min()}"


# ── partitioned_threshold_model ─────────────────────────────────────────────

class TestPartitionedThresholdModel:

    def test_returns_tuple(self, synthetic_dem, synthetic_area,
                            synthetic_flow_direction, model_params):
        f = partitioned_threshold_model(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"], U=model_params["U"], K=model_params["K"],
        )
        result = f(model_params["A_crit"], model_params["D"])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_mask_is_boolean(self, synthetic_dem, synthetic_area,
                              synthetic_flow_direction, model_params):
        f = partitioned_threshold_model(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"], U=model_params["U"], K=model_params["K"],
        )
        Z, mask = f(model_params["A_crit"], model_params["D"])
        assert mask.dtype == bool

    def test_mask_shape(self, synthetic_dem, synthetic_area,
                         synthetic_flow_direction, model_params):
        f = partitioned_threshold_model(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"], U=model_params["U"], K=model_params["K"],
        )
        Z, mask = f(model_params["A_crit"], model_params["D"])
        assert mask.shape == synthetic_dem._griddata.shape

    def test_elevation_shape(self, synthetic_dem, synthetic_area,
                              synthetic_flow_direction, model_params):
        f = partitioned_threshold_model(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"], U=model_params["U"], K=model_params["K"],
        )
        Z, mask = f(model_params["A_crit"], model_params["D"])
        assert Z.shape == synthetic_dem._griddata.shape


# ── partitioned_threshold_model_nonlinear ───────────────────────────────────

class TestPartitionedNonlinearModel:

    def test_returns_tuple(self, synthetic_dem, synthetic_area,
                            synthetic_flow_direction, model_params):
        f = partitioned_threshold_model_nonlinear(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"], U=model_params["U"], K=model_params["K"],
        )
        result = f(model_params["A_crit"], model_params["D"], model_params["S_c"])
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_converges(self, synthetic_dem, synthetic_area,
                        synthetic_flow_direction, model_params):
        """Should complete without raising for reasonable inputs."""
        f = partitioned_threshold_model_nonlinear(
            synthetic_dem, synthetic_area, synthetic_flow_direction,
            m=model_params["m"], U=model_params["U"], K=model_params["K"],
        )
        Z, mask = f(model_params["A_crit"], model_params["D"],
                     model_params["S_c"], max_iter=10, tol=1e-4)
        assert Z.shape == synthetic_dem._griddata.shape
        assert mask.dtype == bool


# ── _find_boundary_value ────────────────────────────────────────────────────

class TestFindBoundaryValue:

    def test_valid_boundary(self):
        """Should return the boundary elevation for a valid edge."""
        Ny, Nx = 10, 10
        Z0 = np.full((Ny, Nx), np.nan)
        Z0[0, :] = 100.0   # top edge valid
        Z0[-1, :] = 50.0   # bottom edge valid
        Z0[:, 0] = 75.0    # left edge valid
        Z0[:, -1] = 80.0   # right edge valid

        # Interior cell — should find left edge value for that row
        val = _find_boundary_value(Z0, 5, 5, Nx, Ny)
        assert np.isfinite(val)

    def test_nan_boundary(self):
        """When all edges are NaN, should return NaN gracefully."""
        Ny, Nx = 10, 10
        Z0 = np.full((Ny, Nx), np.nan)
        val = _find_boundary_value(Z0, 5, 5, Nx, Ny)
        assert np.isnan(val)


class TestPhase3VectorisedAssembly:
    """Phase 3 step 2: the vectorised hillslope assembly and slopes reproduce the v1 loops bit for bit."""

    def _grids(self):
        from tests.synthetic_tile import hydro_from_padded, synthetic_dem

        from lemcvt.models.common import extract_grid_data, precompute_channel_network

        z = synthetic_dem(60, 60, 1.0, seed=7)
        zf, fd, area = hydro_from_padded(z, 1.0)
        gd = extract_grid_data(zf, area, fd, 1.0)
        Zc, _ds, k_out, bc_inds, bc_vals = precompute_channel_network(gd, 0.4, 1e-4, 2e-5)
        ch = gd["A_flat"] >= 200.0
        ch[k_out] = True
        is_fixed = ch.copy()
        is_fixed[bc_inds] = True
        hs_mask = ~is_fixed
        hs_idx = np.where(hs_mask)[0]
        to_red = np.full(gd["n"], -1, dtype=int)
        to_red[hs_idx] = np.arange(hs_idx.size)
        return gd, Zc, ch, bc_inds, bc_vals, hs_mask, hs_idx, to_red

    def test_scalar_D_bit_identical(self):
        from lemcvt.models.common import solve_hillslope_diffusion, solve_hillslope_diffusion_v1

        gd, Zc, ch, bc_inds, bc_vals, hs_mask, hs_idx, to_red = self._grids()
        args = (gd, hs_idx, to_red, hs_mask, 0.013, Zc, ch, bc_inds, bc_vals, 1e-4)
        np.testing.assert_array_equal(solve_hillslope_diffusion(*args), solve_hillslope_diffusion_v1(*args))

    def test_array_D_bit_identical(self):
        from lemcvt.models.common import solve_hillslope_diffusion, solve_hillslope_diffusion_v1

        gd, Zc, ch, bc_inds, bc_vals, hs_mask, hs_idx, to_red = self._grids()
        rs = np.random.RandomState(1)
        D_eff = 0.01 / (1 - rs.uniform(0, 0.8, hs_idx.size) ** 2)
        args = (gd, hs_idx, to_red, hs_mask, D_eff, Zc, ch, bc_inds, bc_vals, 1e-4)
        np.testing.assert_array_equal(solve_hillslope_diffusion(*args), solve_hillslope_diffusion_v1(*args))

    def test_matrix_entries_match_loop(self):
        """The assembled CSR equals the v1 LIL matrix entry by entry (not just the solution)."""
        from scipy.sparse import lil_matrix

        from lemcvt.models.common import assemble_hillslope_system

        gd, Zc, ch, bc_inds, bc_vals, hs_mask, hs_idx, to_red = self._grids()
        D = 0.0123
        M, b = assemble_hillslope_system(gd, hs_idx, to_red, hs_mask, D, Zc, ch, bc_vals, 1e-4)
        # v1 loop, verbatim
        n_h = hs_idx.size
        Ny, Nx, valid, idx_map = gd["Ny"], gd["Nx"], gd["valid"], gd["idx_map"]
        rev_i, rev_j, inv_dx2, inv_dy2 = gd["rev_i"], gd["rev_j"], gd["inv_dx2"], gd["inv_dy2"]
        Mv = lil_matrix((n_h, n_h), dtype=float)
        bv = np.full(n_h, -1e-4, dtype=float)
        for idx, k in enumerate(hs_idx):
            i, j = rev_i[k], rev_j[k]
            for di, dj, inv in [(-1, 0, inv_dx2), (1, 0, inv_dx2), (0, -1, inv_dy2), (0, 1, inv_dy2)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < Nx and 0 <= nj < Ny and valid[nj, ni]:
                    nk = idx_map[nj, ni]
                    if hs_mask[nk]:
                        Mv[idx, to_red[nk]] += D * inv
                        Mv[idx, idx] -= D * inv
                    else:
                        bv[idx] -= D * inv * (Zc[nk] if ch[nk] else bc_vals[nk])
                        Mv[idx, idx] -= D * inv
                else:
                    Mv[idx, idx] -= D * inv
        Mv = Mv.tocsr()
        Mv.sort_indices()
        M.sort_indices()
        np.testing.assert_array_equal(M.indptr, Mv.indptr)
        np.testing.assert_array_equal(M.indices, Mv.indices)
        np.testing.assert_array_equal(M.data, Mv.data)
        np.testing.assert_array_equal(b, bv)

    def test_slopes_bit_identical(self):
        from tests.synthetic_tile import hydro_from_padded, synthetic_dem

        z = synthetic_dem(50, 50, 1.0, seed=8)
        zf, fd, area = hydro_from_padded(z, 1.0)
        f = _pn.partitioned_threshold_model_nonlinear(zf, area, fd, 1.0, 0.4, 1e-4, 2e-5)
        hs = np.where(area.ravel() < 150.0)[0]
        hs = hs[(hs % 7) != 0]              # a non-trivial subset, like a hillslope set
        zr = zf.ravel() + np.random.RandomState(2).normal(0, 0.5, zf.size)
        np.testing.assert_array_equal(f.compute_slopes(zr, hs), f.compute_slopes_v1(zr, hs))


class TestPhase3AdvectionDiffusionAssembly:
    """Phase 3 step 3: vectorised AD assembly with build-time Dirichlet rows equals the v1 construction bit for bit."""

    def test_solution_bit_identical(self):
        from tests.synthetic_tile import hydro_from_padded, synthetic_dem

        z = synthetic_dem(60, 60, 1.0, seed=7)
        zf, fd, area = hydro_from_padded(z, 1.0)
        f_v1 = _ad.advection_diffusion_model_v1(zf, area, fd, 1.0, 0.4)
        f_new = _ad.advection_diffusion_model(zf, area, fd, 1.0, 0.4)
        for U, K, D in ((1e-4, 2e-5, 1e-2), (1.47e-4, 5e-5, 3e-3), (5.5e-5, 1e-6, 2.8e-3)):
            a, b = f_v1(U, K, D), f_new(U, K, D)
            np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
            np.testing.assert_array_equal(a[np.isfinite(a)], b[np.isfinite(b)])

    def test_matrix_identical_to_v1_round_trip(self):
        from tests.synthetic_tile import hydro_from_padded, synthetic_dem

        z = synthetic_dem(40, 40, 1.0, seed=9)
        zf, fd, area = hydro_from_padded(z, 1.0)
        f_new = _ad.advection_diffusion_model(zf, area, fd, 1.0, 0.4)
        M_diff0, M_adv0, I_bc = f_new.matrices
        bc_inds, bc_vals, b_U = f_new.bc
        K, D = 2e-5, 1e-2
        M_new = (D * M_diff0 + K * M_adv0 + I_bc).tocsr()
        M_new.sort_indices()
        # v1 round trip on the v1 stencils
        f_v1 = _ad.advection_diffusion_model_v1(zf, area, fd, 1.0, 0.4)
        cl = f_v1.__closure__
        names = f_v1.__code__.co_freevars
        env = {nm: c.cell_contents for nm, c in zip(names, cl, strict=True)}
        M = D * env["M_diff_csr"] + K * env["M_adv_csr"]
        M = M.tolil()
        M[env["bc_inds_arr"], :] = 0.0
        M[env["bc_inds_arr"], env["bc_inds_arr"]] = 1.0
        M_v1 = M.tocsr()
        M_v1.sort_indices()
        np.testing.assert_array_equal(M_new.indptr, M_v1.indptr)
        np.testing.assert_array_equal(M_new.indices, M_v1.indices)
        np.testing.assert_array_equal(M_new.data, M_v1.data)
        np.testing.assert_array_equal(np.sort(bc_inds), np.sort(env["bc_inds_arr"]))


class TestPhase3ChiNetwork:
    """Phase 3 step 4: the once-integrated chi network reproduces the per-K rebuild to rounding."""

    def test_channel_elevations_and_surfaces(self):
        from tests.synthetic_tile import hydro_from_padded, synthetic_dem

        z = synthetic_dem(80, 80, 1.0, seed=7)
        zf, fd, area = hydro_from_padded(z, 1.0)
        U, m = 1.47e-4, 0.4
        g = _p.partitioned_threshold_model_chi(zf, area, fd, 1.0, m, U)
        relief = float(np.nanmax(zf) - np.nanmin(zf))
        for K in (2e-5, 7e-5):
            f = _p.partitioned_threshold_model(zf, area, fd, 1.0, m, U, K)
            zc_v1 = f.Z_channel_all
            zc_chi = g.z_anchor + (U / K) * g.chi
            np.testing.assert_array_equal(np.isnan(zc_v1), np.isnan(zc_chi))
            ok = np.isfinite(zc_v1)
            assert np.max(np.abs(zc_v1[ok] - zc_chi[ok]) / np.abs(zc_v1[ok])) < 1e-13
            for A_crit, D in ((300.0, 1e-2), (1000.0, 3e-3)):
                Z1, m1 = f(A_crit, D)
                Z2, m2 = g(A_crit, D, K)
                np.testing.assert_array_equal(m1, m2)
                assert np.nanmax(np.abs(Z1 - Z2)) <= 1e-12 * relief
