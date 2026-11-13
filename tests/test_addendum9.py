"""Addendum 9, section 2.2: the Newton solver of the nonlinear partitioned model, on synthetic grids.

1. Agreement with Picard where Picard converges (both forms).
2. The 1-D hillslope strip: the flux form's R* converges to the integrated steady nonlinear curve of Roering et al.
   (2007) as the grid is refined, at E* = 1, 5 and 20; the v1 (centre) form converges to its own curve
   R* = (2/E*) ln cosh(E*/2) instead.
3. The analytic Jacobian against central finite differences (both forms).
Plus: Picard on the centre form is unchanged by the new options, the outcome classes, and continuation from z_init.
Test 4 (AP1 at its fitted S_c across the full D grid) needs the tile and runs in the pilot (`lemcvt a9 pilot`).
"""

import numpy as np
import pytest

from lemcvt.analyses.er_star import integrate_profile_R
from lemcvt.analyses.nonlinear_fair import centre_form_R, strip_R_star
from lemcvt.models import partitioned_nonlinear as pn
from lemcvt.models.common import precompute_channel_network
from lemcvt.models.nonlinear_newton import HillslopeSystem, assemble_flux_system, classify
from synthetic_tile import hydro_from_padded, synthetic_dem

U, K, M = 1e-4, 2e-4, 0.4


@pytest.fixture(scope="module")
def tile():
    z = synthetic_dem(40, 40, 1.0, seed=3)
    zf, fd, area = hydro_from_padded(z, 1.0)
    return zf, fd, area


def _system(f, form, A_crit, D, S_c):
    gd = f.grid_data
    n = gd["n"]
    Zc, _d, k_out, bci, bcv = precompute_channel_network(gd, M, U, K)
    cm = gd["A_flat"] >= A_crit
    cm[k_out] = True
    fixed = cm.copy()
    fixed[bci] = True
    hm = ~fixed
    hs = np.where(hm)[0]
    h2r = np.full(n, -1)
    h2r[hs] = np.arange(hs.size)
    return HillslopeSystem(form, gd, hs, h2r, hm, D, S_c, Zc, cm, bcv, U, f.compute_slopes), hm


@pytest.mark.parametrize("form", ["centre", "flux"])
def test_jacobian_matches_finite_differences(tile, form):
    zf, fd, area = tile
    f = pn.partitioned_threshold_model_nonlinear(zf, area, fd, 1.0, M, U, K)
    A_crit, D, S_c = 200.0, 5e-3, 3.0
    f(A_crit, D, S_c, 200, 1e-12, form=form)
    sysm, hm = _system(f, form, A_crit, D, S_c)
    z = f.last_z_flat.copy() + np.where(hm, np.random.RandomState(0).normal(0, 0.05, hm.size), 0.0)
    st = sysm.state(z)
    assert 0.3 < np.max(st["slopes"]) / S_c < 0.99            # the nonlinearity is engaged and nothing is clipped
    J = sysm.jacobian(z, st).toarray()
    Jfd = np.zeros_like(J)
    h = 1e-4               # the discrepancy scales as 1/h below this (roundoff, eps |F| / h), so 1e-4 is the cleanest step
    for k, c in enumerate(sysm.hs):
        zp, zm = z.copy(), z.copy()
        zp[c] += h
        zm[c] -= h
        Jfd[:, k] = (sysm.state(zp)["F"] - sysm.state(zm)["F"]) / (2 * h)
    assert np.abs(J - Jfd).max() / np.abs(J).max() <= 1e-6
    big = np.abs(J) > 1e-3 * np.abs(J).max()      # entrywise on the entries above the finite-difference noise floor
    assert np.max(np.abs(J - Jfd)[big] / np.abs(J)[big]) <= 1e-6


@pytest.mark.parametrize("form", ["centre", "flux"])
def test_newton_reproduces_converged_picard(tile, form):
    zf, fd, area = tile
    f = pn.partitioned_threshold_model_nonlinear(zf, area, fd, 1.0, M, U, K)
    A_crit, D, S_c = 200.0, 5e-3, 3.0
    Zp, _ = f(A_crit, D, S_c, 100, 1e-6, form=form)
    assert f.last_picard["converged"] and f.last_outcome["outcome"] == "converged"
    Zp_tight, _ = f(A_crit, D, S_c, 2000, 1e-12, form=form)
    Zn, _ = f(A_crit, D, S_c, solver="newton", form=form)
    nw = f.last_newton
    assert nw["converged"] and nw["outcome"] == "converged" and nw["rel_residual"] <= 1e-10
    assert np.nanmax(np.abs(Zn - Zp)) <= 1e-6
    assert np.nanmax(np.abs(Zn - Zp_tight)) <= 1e-9


def test_picard_centre_unchanged_by_new_options(tile):
    zf, fd, area = tile
    f = pn.partitioned_threshold_model_nonlinear(zf, area, fd, 1.0, M, U, K)
    Z1, _ = f(200.0, 5e-3, 0.6, 30, 1e-6)
    it1 = f.last_picard["iterations"]
    Z2, _ = f(200.0, 5e-3, 0.6, 30, 1e-6, solver="picard", form="centre", z_init=np.zeros(f.grid_data["n"]))
    np.testing.assert_array_equal(Z1, Z2)
    assert it1 == f.last_picard["iterations"]


def test_flux_form_equals_centre_form_at_constant_D(tile):
    zf, fd, area = tile
    f = pn.partitioned_threshold_model_nonlinear(zf, area, fd, 1.0, M, U, K)
    sysm, hm = _system(f, "centre", 200.0, 5e-3, 3.0)
    from lemcvt.models.common import assemble_hillslope_system

    args = (f.grid_data, sysm.hs, sysm.h2r, hm, 5e-3, sysm.Zc, sysm.cmask, sysm.bc_vals, U)
    Mc, bc = assemble_hillslope_system(*args)
    Mf, bf = assemble_flux_system(*args)
    assert (Mc != Mf).nnz == 0
    np.testing.assert_array_equal(bc, bf)


def test_outcome_classes():
    assert classify(True, 0.0) == "converged"
    assert classify(True, 0.011) == "no_steady_solution"
    assert classify(False, 0.011) == "no_steady_solution"
    assert classify(False, 0.005) == "not_converged"
    assert classify(True, 0.01) == "converged"


def test_continuation_from_z_init(tile):
    zf, fd, area = tile
    f = pn.partitioned_threshold_model_nonlinear(zf, area, fd, 1.0, M, U, K)
    A_crit, S_c = 200.0, 3.0
    f(A_crit, 6e-3, S_c, solver="newton", form="flux")
    z_prev = f.last_z_flat.copy()
    Zc, _ = f(A_crit, 5e-3, S_c, solver="newton", form="flux", z_init=z_prev)
    assert f.last_newton["initial"] == "z_init" and f.last_outcome["outcome"] == "converged"
    Zl, _ = f(A_crit, 5e-3, S_c, solver="newton", form="flux")
    assert np.nanmax(np.abs(Zc - Zl)) <= 1e-8


# ----------------------------------------------------------------------------- the 1-D strip (2.2, test 2)

@pytest.mark.parametrize("E", [1.0, 5.0, 20.0])
def test_flux_strip_converges_to_roering_curve(E):
    ref = integrate_profile_R(E)
    errs = []
    for dx in (1.0, 0.5):
        R, out = strip_R_star("flux", E, dx)
        assert out["outcome"] == "converged"
        errs.append(abs(R - ref) / ref)
    assert errs[1] < errs[0] / 3.0                      # second order: about / 4 per halving
    assert errs[1] < 3e-3


def test_centre_strip_converges_to_its_own_curve_not_roering():
    E = 5.0
    ref_c = centre_form_R(E)
    errs = []
    for dx in (1.0, 0.5):
        R, out = strip_R_star("centre", E, dx)
        assert out["outcome"] == "converged"
        errs.append(abs(R - ref_c) / ref_c)
    assert errs[1] < errs[0] / 3.0 and errs[1] < 2e-3
    assert R / integrate_profile_R(E) > 1.2                # 22 % above the Roering curve at E* = 5
