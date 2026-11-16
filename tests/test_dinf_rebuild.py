"""Phase 4b (addendum 3 section 6, addendum 4 section 4): the rebuilt D-infinity partitioned model.

1. Aligned V-valley and plane: the D-infinity and D8 partitioned models agree to 1e-10 of relief.
2. Rotated V-valley: the D-infinity model against the D8 model on the same surface at 4x finer
   resolution, sampled back at the coarse nodes. The difference is reported (no pass threshold);
   the D8 model at the coarse resolution is reported beside it for scale.
3. The Phase 4 reduction Z = a + (U/K) b + (U/D) c holds for the new model to 1e-10 relative.
4. Channel monotonicity on AP1 and GM1 needs the full tiles: it is computed by
   ``lemcvt d8-vs-dinf --pt-dinf-model both`` and reported; here the synthetic cases are checked.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from lemcvt.analyses.dinf_diagnostics import _padded_hydro, v_valley
from lemcvt.models.dinf_models import partitioned_threshold_model_dinf_v1
from lemcvt.models.dinf_partitioned import channel_monotonicity, partitioned_threshold_model_dinf
from lemcvt.models.partitioned import partitioned_threshold_model

M, U, K, D = 0.4, 1.47e-4, 2.2e-5, 1.0e-2
A_CRIT = 500.5          # not a multiple of the cell area, so a 1e-10 m^2 area difference cannot move the mask


def _models(z, dx, A_crit=A_CRIT):
    zf, fd, area, angles, r_facet, facet_idx, A_dinf = _padded_hydro(z, dx)
    Z_d8, mask_d8 = partitioned_threshold_model(zf, area, fd, dx, M, U, K)(A_crit, D)
    f = partitioned_threshold_model_dinf(zf, area, fd, dx, M, U, K, angles, r_facet, facet_idx, A_dinf)
    Z_di, mask_di = f(A_crit, D)
    return zf, fd, area, angles, r_facet, facet_idx, A_dinf, Z_d8, mask_d8, f, Z_di, mask_di


class TestAlignedEquivalence:
    @pytest.mark.parametrize("name,side,axial", [("plane_south", 0.0, 0.1), ("v_equal_slopes", 0.1, 0.1)])
    def test_dinf_equals_d8(self, name, side, axial):
        n, dx = 101, 1.0
        zf, fd, area, angles, r_facet, facet_idx, A_dinf, Z_d8, mask_d8, f, Z_di, mask_di = _models(v_valley(n, n, dx, side, axial), dx)
        relief = float(np.nanmax(zf) - np.nanmin(zf))
        assert np.array_equal(mask_d8, mask_di), name
        d = float(np.nanmax(np.abs(Z_di - Z_d8)) / relief)
        assert d <= 1e-10, (name, d)
        assert np.all(np.isfinite(Z_di[np.isfinite(zf)]))
        s = f.last_system
        assert s.info["n_channel_without_receiver_as_hillslope"] == 0
        mono = channel_monotonicity(f.last_z_flat, s)
        assert mono["n_lower_than_every_receiver"] == 0, mono


class TestRotatedV:
    def test_report_difference_against_fine_d8(self):
        """Report-only: D-inf at dx against D8 at dx/4 sampled at the coarse nodes (and D8 at dx for scale)."""
        n, dx = 81, 1.0
        side, axial = 0.1, 0.1 * math.tan(math.radians(20))
        zf, fd, area, angles, r_facet, facet_idx, A_dinf, Z_d8, mask_d8, f, Z_di, mask_di = _models(v_valley(n, n, dx, side, axial), dx)
        relief = float(np.nanmax(zf) - np.nanmin(zf))
        # the same analytic surface at 4x finer resolution: nodes (4 i, 4 j) coincide with the coarse nodes
        nf, dxf = 4 * (n - 1) + 1, dx / 4
        zf_f, fd_f, area_f, *_ = _padded_hydro(v_valley(nf, nf, dxf, side, axial), dxf)
        Z_d8_f, _ = partitioned_threshold_model(zf_f, area_f, fd_f, dxf, M, U, K)(A_CRIT, D)
        Z_fine_at_coarse = Z_d8_f[::4, ::4]
        assert Z_fine_at_coarse.shape == Z_di.shape
        inner = (slice(2, -2), slice(2, -2))
        d_dinf = float(np.nanmax(np.abs(Z_di - Z_fine_at_coarse)[inner]) / relief)
        d_d8 = float(np.nanmax(np.abs(Z_d8 - Z_fine_at_coarse)[inner]) / relief)
        rms_dinf = float(np.sqrt(np.nanmean((Z_di - Z_fine_at_coarse)[inner] ** 2)) / relief)
        rms_d8 = float(np.sqrt(np.nanmean((Z_d8 - Z_fine_at_coarse)[inner] ** 2)) / relief)
        # v1 D-inf model for the record
        Z_v1, _ = partitioned_threshold_model_dinf_v1(zf, area, fd, dx, M, U, K, angles, r_facet, facet_idx, A_dinf)(A_CRIT, D)
        d_v1 = float(np.nanmax(np.abs(Z_v1 - Z_fine_at_coarse)[inner]) / relief)
        print(f"\nrotated V (20 deg, {n}x{n} at dx={dx}, fine {nf}x{nf} at dx={dxf}): max|dZ|/relief against fine D8: "
              f"D-inf rebuilt {d_dinf:.3e} (rms {rms_dinf:.3e}), D8 coarse {d_d8:.3e} (rms {rms_d8:.3e}), D-inf v1 {d_v1:.3e}")
        assert np.isfinite(d_dinf) and np.isfinite(d_d8)
        mono = channel_monotonicity(f.last_z_flat, f.last_system)
        assert mono["n_lower_than_every_receiver"] == 0, mono
        # report only (addendum 3 section 6, test 2): the max is set by single cells where the D8 staircase path and the
        # D-infinity path diverge; the rms over the interior is the quantity to read (reported by `lemcvt d8-vs-dinf`)


class TestReduction:
    def test_a_b_c_matches_direct(self):
        n, dx = 81, 1.0
        z = v_valley(n, n, dx, 0.1, 0.1 * math.tan(math.radians(20)))
        zf, fd, area, angles, r_facet, facet_idx, A_dinf = _padded_hydro(z, dx)
        f = partitioned_threshold_model_dinf(zf, area, fd, dx, M, U, K, angles, r_facet, facet_idx, A_dinf)
        rng = np.random.default_rng(0)
        relief = float(np.nanmax(zf) - np.nanmin(zf))
        for a_c in (300.0, 900.0):
            a, b, c = f.reduce(a_c)
            for _ in range(3):
                Kr = 10 ** rng.uniform(-6, -4)
                Dr = 10 ** rng.uniform(-3, -1)
                Z_direct, _ = f(a_c, Dr, Kr)
                Z_red = f.to_grid(a + (U / Kr) * b + (U / Dr) * c)
                ok = np.isfinite(Z_direct)
                rel = float(np.max(np.abs(Z_red[ok] - Z_direct[ok])) / max(np.max(np.abs(Z_direct[ok])), relief))
                assert rel < 1e-10, (a_c, Kr, Dr, rel)
