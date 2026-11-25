"""Addendum 23, decision 2.1: an arbitrary domain for the linear models.

``extract_grid_data(..., outlets=mask, noflux_outside=True)``: the fixed (anchor, Dirichlet) cells are the outlets instead of the
valid grid-edge cells, and a face towards a NaN or off-grid neighbour carries no flux (a drainage divide). Defaults are the v1/v2
behaviour bit for bit (the v1 bit-identity tests of ``test_models.py`` cover the default path; here the edge ring given as the
outlets must reproduce it exactly).

Checks:
  1. flags and the fixed set default to v1 (edge ring);
  2. the edge ring given as ``outlets`` reproduces the default reduction and AD surface bit for bit, with either no-flux setting;
  3. a tile padded with NaN, its former edge ring as the outlets, reproduces the unpadded default surfaces (partitioned, AD);
  4. mirror symmetry: on a strip whose two halves are mirror images (flow away from the centre line to a fixed row at each end),
     the half strip with no-flux at the centre line reproduces the full strip's solution on that half (an exact discrete identity:
     the symmetric solution has z_{N-1} = z_N, which is the no-flux ghost); the v1 ghost value of zero does not;
  5. the validation errors (outlets off the valid set, the largest-area cell not an outlet).
"""

from __future__ import annotations

import numpy as np
import pytest

from lemcvt.models.ad import advection_diffusion_model
from lemcvt.models.common import boundary_cells, extract_grid_data, make_boundary_mask
from lemcvt.models.partitioned import partitioned_threshold_model
from lemcvt.optimize.v2 import PartitionedProfiler
from synthetic_tile import hydro_from_padded, synthetic_dem

U = 1.0e-4


def _tile(ny: int, nx: int, seed: int):
    z = synthetic_dem(ny, nx, 1.0, seed=seed)
    zf, fd, area = hydro_from_padded(z, 1.0)
    return zf, area, fd


def _ad(Z, A, FD, m, K, D, **kw):
    return advection_diffusion_model(Z, A, FD, 1.0, m, **kw)(U, K, D)


class TestDefaults:

    def test_flags_and_fixed_set(self):
        zf, a, fd = _tile(40, 40, 1)
        gd = extract_grid_data(zf, a, fd, 1.0)
        assert gd["outlets"] is None and gd["noflux_outside"] is False
        np.testing.assert_array_equal(boundary_cells(gd), make_boundary_mask(gd["valid"]))

    @pytest.mark.parametrize("noflux", [False, True])
    def test_edge_ring_as_outlets_is_bit_identical(self, noflux):
        zf, a, fd = _tile(50, 50, 2)
        ring = make_boundary_mask(np.isfinite(a))
        p0 = PartitionedProfiler(zf, a, fd, 1.0, U, m_values=[0.4])
        p1 = PartitionedProfiler(zf, a, fd, 1.0, U, m_values=[0.4], outlets=ring, noflux_outside=noflux)
        for A_crit in (150.0, 600.0):
            r0, r1 = p0.reduce(A_crit, 0.4), p1.reduce(A_crit, 0.4)
            np.testing.assert_array_equal(r0.channel_mask, r1.channel_mask)
            for k in ("a", "b", "c"):
                np.testing.assert_array_equal(getattr(r0, k), getattr(r1, k))
        Z0 = _ad(zf, a, fd, 0.4, 1e-4, 1e-2)
        Z1 = _ad(zf, a, fd, 0.4, 1e-4, 1e-2, outlets=ring, noflux_outside=noflux)
        np.testing.assert_array_equal(Z0, Z1)
        Zp0, m0 = partitioned_threshold_model(zf, a, fd, 1.0, 0.4, U, 2e-5)(300.0, 4e-3)
        Zp1, m1 = partitioned_threshold_model(zf, a, fd, 1.0, 0.4, U, 2e-5, outlets=ring, noflux_outside=noflux)(300.0, 4e-3)
        np.testing.assert_array_equal(Zp0, Zp1)
        np.testing.assert_array_equal(m0, m1)


class TestEmbedding:
    """A tile padded with NaN, with its former edge ring as the outlets, reproduces the unpadded default solution."""

    @pytest.mark.parametrize("noflux", [False, True])
    def test_partitioned_and_ad(self, noflux):
        zf, a, fd = _tile(48, 44, 3)
        pad = 3
        Zp = np.pad(zf, pad, constant_values=np.nan)
        Ap = np.pad(a, pad, constant_values=np.nan)
        FDp = np.pad(fd, pad, constant_values=0)
        inner = (slice(pad, pad + zf.shape[0]), slice(pad, pad + zf.shape[1]))
        ring = np.zeros(Zp.shape, dtype=bool)
        ring[inner] = make_boundary_mask(np.isfinite(a))
        p0 = PartitionedProfiler(zf, a, fd, 1.0, U, m_values=[0.45])
        p1 = PartitionedProfiler(Zp, Ap, FDp, 1.0, U, m_values=[0.45], outlets=ring, noflux_outside=noflux)
        assert p1.n == p0.n
        for A_crit in (120.0, 500.0):
            r0, r1 = p0.reduce(A_crit, 0.45), p1.reduce(A_crit, 0.45)
            Z0 = p0.to_grid(r0.Z_flat(3e-5, 5e-3))
            Z1 = p1.to_grid(r1.Z_flat(3e-5, 5e-3))[inner]
            np.testing.assert_array_equal(np.isfinite(Z0), np.isfinite(Z1))
            assert np.nanmax(np.abs(Z0 - Z1)) <= 1e-9 * np.nanmax(np.abs(Z0))
        Zad0 = _ad(zf, a, fd, 0.45, 2e-4, 8e-3)
        Zad1 = _ad(Zp, Ap, FDp, 0.45, 2e-4, 8e-3, outlets=ring, noflux_outside=noflux)
        assert np.all(np.isnan(Zad1[:pad])) and np.all(np.isnan(Zad1[:, :pad]))
        assert np.nanmax(np.abs(Zad0 - Zad1[inner])) <= 1e-9 * np.nanmax(np.abs(Zad0))


def _mirror_strip(N: int = 30, W: int = 16):
    """A W-column strip of 2N rows, mirror-symmetric about the centre line: flow north in rows 0..N-1 (code 64) and south in rows
    N..2N-1 (code 4); area accumulates along the columns (1 cell at the centre rows, N cells at the end rows); Z symmetric."""
    rows = np.arange(2 * N)
    d = np.minimum(rows, 2 * N - 1 - rows).astype(float)
    cols = np.arange(W).astype(float)
    Z = 100.0 + 0.8 * d[:, None] - 0.01 * d[:, None] ** 2 + 0.3 * np.sin(cols / 3.0)[None, :] + 0.05 * np.cos(0.7 * d[:, None]) * np.cos(0.9 * cols[None, :])
    A = np.repeat((N - d)[:, None], W, axis=1)
    FD = np.where(rows[:, None] < N, 64, 4).astype(np.uint8) * np.ones((1, W), dtype=np.uint8)
    return Z, A, FD


class TestMirrorSymmetry:
    """The half strip with no-flux at the centre line equals the full strip on that half (partitioned and AD)."""

    def _halves(self):
        N, W = 30, 16
        Z, A, FD = _mirror_strip(N, W)
        Zh, Ah, FDh = Z[:N].copy(), A[:N].copy(), FD[:N].copy()
        out = np.zeros((N, W), dtype=bool)
        out[0, :] = True              # the fixed end row (an edge row of the full strip)
        out[:, 0] = out[:, -1] = True  # the full strip's side edges are fixed too
        return N, Z, A, FD, Zh, Ah, FDh, out

    def test_partitioned(self):
        N, Z, A, FD, Zh, Ah, FDh, out = self._halves()
        m, K, D, A_crit = 0.5, 3e-5, 2e-3, 8.0
        full, _ = partitioned_threshold_model(Z, A, FD, 1.0, m, U, K)(A_crit, D)
        half, _ = partitioned_threshold_model(Zh, Ah, FDh, 1.0, m, U, K, outlets=out, noflux_outside=True)(A_crit, D)
        assert np.all(np.isfinite(full)) and np.all(np.isfinite(half))
        np.testing.assert_allclose(full[:N], half, rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(full[N - 1], full[N], rtol=1e-12)          # the symmetry the identity rests on
        ghost0, _ = partitioned_threshold_model(Zh, Ah, FDh, 1.0, m, U, K, outlets=out, noflux_outside=False)(A_crit, D)
        assert np.max(np.abs(ghost0 - full[:N])) > 1.0                          # the v1 ghost value of zero is not a divide
        # the same through the profiler's reduction
        prof = PartitionedProfiler(Zh, Ah, FDh, 1.0, U, m_values=[m], outlets=out, noflux_outside=True)
        red = prof.reduce(A_crit, m)
        np.testing.assert_allclose(prof.to_grid(red.Z_flat(K, D)), half, rtol=1e-10, atol=1e-9)

    def test_ad(self):
        N, Z, A, FD, Zh, Ah, FDh, out = self._halves()
        m, K, D = 0.5, 2e-4, 5e-3
        full = _ad(Z, A, FD, m, K, D)
        half = _ad(Zh, Ah, FDh, m, K, D, outlets=out, noflux_outside=True)
        assert np.all(np.isfinite(full)) and np.all(np.isfinite(half))
        np.testing.assert_allclose(full[:N], half, rtol=1e-10, atol=1e-9)
        ghost0 = _ad(Zh, Ah, FDh, m, K, D, outlets=out, noflux_outside=False)
        assert np.max(np.abs(ghost0 - full[:N])) > 1.0


class TestValidation:

    def test_outlets_must_be_valid_cells(self):
        zf, a, fd = _tile(30, 30, 4)
        Zp, Ap, FDp = np.pad(zf, 2, constant_values=np.nan), np.pad(a, 2, constant_values=np.nan), np.pad(fd, 2, constant_values=0)
        bad = np.zeros(Zp.shape, dtype=bool)
        bad[0, 0] = True
        with pytest.raises(ValueError, match="not valid"):
            extract_grid_data(Zp, Ap, FDp, 1.0, outlets=bad)
        with pytest.raises(ValueError, match="empty"):
            extract_grid_data(zf, a, fd, 1.0, outlets=np.zeros(zf.shape, dtype=bool))

    def test_largest_area_cell_must_be_an_outlet(self):
        N, W = 20, 10
        Z, A, FD = _mirror_strip(N, W)
        out = np.zeros((N, W), dtype=bool)
        out[1, :] = True                                   # row 1, not the end row that holds the largest area
        with pytest.raises(ValueError, match="largest-area"):
            PartitionedProfiler(Z[:N], A[:N], FD[:N], 1.0, U, m_values=[0.5], outlets=out, noflux_outside=True)
        with pytest.raises(ValueError, match="largest-area"):
            advection_diffusion_model(Z[:N], A[:N], FD[:N], 1.0, 0.5, outlets=out, noflux_outside=True)
