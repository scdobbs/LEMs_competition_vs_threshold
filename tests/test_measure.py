"""Measurement pipeline: chi/theta/k_s recovery on a landscape built by the partitioned
model itself, curvature kernels on analytic surfaces, hilltop selection and the
threshold rollover."""

import numpy as np
import pytest

from lemcvt.measure.chi import channel_network, chi_for_thetas, relative_elevation
from lemcvt.measure.concavity import fit_chi, ks_at_theta
from lemcvt.measure.hilltop_curvature import curvature_kernel, hilltop_mask, measure_hilltop_curvature, total_curvature
from lemcvt.measure.thresholds import slope_area_rollover
from lemcvt.models.partitioned import partitioned_threshold_model
from synthetic_tile import hydro_from_padded, synthetic_dem


@pytest.fixture(scope="module")
def steady_state_tile():
    """A 120 x 120 landscape produced by the partitioned model: channels follow
    z = z_out + (U/K) chi with m = theta exactly, hillslopes are linear diffusion."""
    theta, U, K, D, A_crit = 0.45, 1e-4, 2e-5, 1e-2, 400.0
    z = synthetic_dem(120, 120, 1.0, seed=7)
    zf, fd, area = hydro_from_padded(z, 1.0)
    Z_ss, mask = partitioned_threshold_model(zf, area, fd, 1.0, theta, U, K)(A_crit, D)
    assert np.isnan(Z_ss[mask]).sum() == 0, "synthetic tile must give finite channel elevations"
    return {"Z": Z_ss, "A": area, "FD": fd, "dx": 1.0, "theta": theta, "ks": U / K, "A_crit": A_crit, "U": U,
            "channel_mask": mask}


class TestChi:

    def test_outlets_have_zero_chi_and_increase_upstream(self, steady_state_tile):
        t = steady_state_tile
        net = channel_network(t["Z"], t["A"], t["FD"], t["A_crit"], t["dx"])
        chi = chi_for_thetas(net, np.array([t["theta"]]))[:, 0]
        assert np.all(chi[net.outlet_index] == 0.0)
        rec = net.receiver
        has = rec >= 0
        assert np.all(chi[has] > chi[rec[has]])
        assert net.n_basins >= 1
        assert net.n_pixels == int(net.mask.sum())

    def test_relative_elevation_zero_at_outlet(self, steady_state_tile):
        t = steady_state_tile
        net = channel_network(t["Z"], t["A"], t["FD"], t["A_crit"], t["dx"])
        zr = relative_elevation(net)
        assert np.all(zr[net.outlet_index] == 0.0)

    def test_recovers_theta_and_ks(self, steady_state_tile):
        """On the model's own steady state the chi fit must find theta and k_s = U/K."""
        t = steady_state_tile
        net = channel_network(t["Z"], t["A"], t["FD"], t["A_crit"], t["dx"])
        thetas = np.round(np.arange(0.2, 0.71, 0.01), 3)
        fit = fit_chi(net, thetas=thetas, min_pixels_per_basin=5, n_boot=30)
        assert fit.theta_star == pytest.approx(t["theta"], abs=0.011)
        assert fit.ks_pooled == pytest.approx(t["ks"], rel=0.02)
        assert fit.r2_pooled > 0.999
        ks_fixed, r2 = ks_at_theta(net, t["theta"], min_pixels_per_basin=5)
        assert ks_fixed == pytest.approx(t["ks"], rel=1e-6)
        assert r2 > 0.9999

    def test_bootstrap_intervals_present(self, steady_state_tile):
        t = steady_state_tile
        net = channel_network(t["Z"], t["A"], t["FD"], t["A_crit"], t["dx"])
        fit = fit_chi(net, thetas=np.array([0.3, 0.4, 0.45, 0.5, 0.6]), min_pixels_per_basin=5, n_boot=20)
        if fit.n_basins_used >= 3:
            assert fit.theta_ci68 is not None and fit.ks_ci68 is not None
            assert fit.ks_ci68[0] <= fit.ks_pooled * 1.05


class TestCurvature:

    @pytest.mark.parametrize("w", [1, 2, 5])
    def test_kernel_exact_on_quadratic(self, w):
        """z = a x^2 + b y^2 + c xy + d x + e y + f -> total curvature 2(a + b), any window."""
        a, b, c, d, e, f = 0.013, -0.004, 0.002, 0.3, -0.1, 50.0
        dx = 0.9752
        yy, xx = np.mgrid[0:41, 0:41]
        x, y = xx * dx, yy * dx
        z = a * x**2 + b * y**2 + c * x * y + d * x + e * y + f
        C = total_curvature(z, w, dx)
        np.testing.assert_allclose(C[w:-w, w:-w], 2 * (a + b), rtol=1e-9, atol=1e-12)

    def test_kernel_sums_to_zero(self):
        k = curvature_kernel(3, 1.0)
        assert abs(k.sum()) < 1e-12
        # linear surfaces have zero curvature
        yy, xx = np.mgrid[0:21, 0:21]
        assert abs(total_curvature(1.0 + 0.5 * xx - 0.2 * yy, 3, 1.0)[5, 5]) < 1e-10

    def test_hilltops_on_ridge_landscape(self):
        """Two parallel valleys: hilltop pixels sit on the ridge between them and have C < 0."""
        ny, nx, dx = 80, 80, 1.0
        y, x = np.meshgrid(np.arange(ny) * dx, np.arange(nx) * dx, indexing="ij")
        z = 300 + 0.2 * (ny - y) + 10 * np.cos(2 * np.pi * x / 40.0)   # ridges at x = 0, 40, 80
        zf, fd, area = hydro_from_padded(z, dx)
        mask, diag = hilltop_mask(zf, area, fd, A_c=200.0, dx=dx, max_slope=None)
        assert diag["n_hilltop"] > 10
        cols = np.nonzero(mask)[1]
        assert np.all(np.abs(cols - 40) <= 2)           # on the central ridge
        res = measure_hilltop_curvature(zf, area, fd, 200.0, U=1e-4, dx=dx, n_boot=10, max_slope=None)
        expected = 10 * (2 * np.pi / 40.0) ** 2           # -d2z/dx2 at the crest
        assert res.C_ht == pytest.approx(expected, rel=0.15)
        assert res.D0 == pytest.approx(1e-4 / res.C_ht)
        assert res.L_star_m >= 3.0


class TestRollover:

    def test_rollover_found(self, steady_state_tile):
        t = steady_state_tile
        r = slope_area_rollover(t["Z"], t["A"], t["FD"], t["dx"], a_min=1.0, a_max=1e4, min_count=10)
        assert np.isfinite(r.A_rollover)
        assert len(r.median_slope) == len(r.log10A_centers) == len(r.counts)
