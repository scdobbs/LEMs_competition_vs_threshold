"""Addendum 5, section 2.1.6: on a synthetic grid with a planted two-row seam, the in-tile correction of the
flow directions and the drainage area reproduces a full recomputation exactly.

The seam is planted the way the AP1 seam arose: the mosaic is routed in two row blocks with the TopoAnalysis
slab convention (the last two rows of a block get no south comparison), the area is accumulated through those
directions in TopoAnalysis's elevation order, and a tile is clipped out. The correction sees only the clipped
grids and must reproduce the clipped area of the correctly routed mosaic.

Two mosaics: one whose margins drain away from the tile (nothing that leaves the tile comes back; the exact
case), and one with noise up to the mosaic edge, where a path can leave the tile and re-enter it. The second
documents section 2.1.3: the correction then differs from the full recomputation only by the re-entered area,
at the re-entry cell and downstream of it.
"""

import numpy as np
import pytest

from lemcvt.analyses.seamfix import (
    classify_seam,
    correct_area_in_tile,
    external_inflow,
    recompute_with_external_inflow,
)
from lemcvt.hydro.d8 import d8_area, d8_flow_direction
from lemcvt.hydro.fill import priority_flood_fill
from synthetic_tile import synthetic_dem


def mosaic_with_seam(n: int = 200, dx: float = 1.0, seam_row: int = 110, pad: int = 20, seed: int = 1, drain_margins: bool = True):
    z = synthetic_dem(n, n, dx, seed)
    if drain_margins:
        # as tests/synthetic_tile.hydro_from_padded: mirrored margins tilted down, so flow that leaves the tile never returns
        zp = np.pad(z, pad, mode="reflect")
        yy, xx = np.indices(zp.shape)
        dist = np.maximum.reduce([np.maximum(pad - yy, 0), np.maximum(yy - (n + pad - 1), 0),
                                  np.maximum(pad - xx, 0), np.maximum(xx - (n + pad - 1), 0)])
        zm = zp - 2.0 * dist * dx
    else:
        zm = synthetic_dem(n + 2 * pad, n + 2 * pad, dx, seed)
    zf = priority_flood_fill(zm, dx)
    fd_true = d8_flow_direction(zf)
    area_true = d8_area(fd_true, zf, dx)
    R = seam_row + pad
    fd_block = d8_flow_direction(zf[: R + 2])          # block A = rows [0, R + 2): its last two rows see no south comparison
    fd_seam = fd_true.copy()
    c0, c1 = pad + 2, pad + n - 2                       # interior columns of the tile only (as on AP1: columns 2-1010)
    fd_seam[R:R + 2, c0:c1] = fd_block[R:R + 2, c0:c1]
    area_seam = d8_area(fd_seam, zf, dx)
    sl = slice(pad, pad + n)
    return (zf[sl, sl].copy(), fd_true[sl, sl].copy(), area_true[sl, sl].copy(), fd_seam[sl, sl].copy(), area_seam[sl, sl].copy(),
            seam_row)


@pytest.mark.parametrize("dx", [1.0, 0.9752])
def test_in_tile_correction_equals_full_recomputation(dx):
    Z, fd_true, area_true, fd_seam, area_seam, r0 = mosaic_with_seam(dx=dx)
    cls = classify_seam(Z, None, fd_seam)
    assert cls["n_seam"] > 50, "the planted seam must produce many differing interior cells"
    assert set(cls["rows"]) <= {r0, r0 + 1}
    seam_rc, new_codes = cls["seam_rc"], cls["new_codes"]
    # the corrected codes are the mosaic's codes (interior cells: the tile recomputation equals the mosaic one)
    assert np.array_equal(new_codes, fd_true[seam_rc[:, 0], seam_rc[:, 1]])
    fd_new, area_new, info = correct_area_in_tile(fd_seam, area_seam, Z, seam_rc, new_codes)
    assert np.array_equal(fd_new, fd_true)
    tol = 1e-9 * dx * dx
    assert np.allclose(area_new, area_true, rtol=0, atol=tol), float(np.max(np.abs(area_new - area_true)) / dx**2)
    if dx == 1.0:
        assert np.array_equal(area_new, area_true)       # integer cell counts: bit identity
    # the independent recomputation (external inflow + accumulation on the corrected directions) agrees too
    area_chk, ext = recompute_with_external_inflow(fd_new, fd_seam, area_seam, Z, dx)
    assert np.allclose(area_chk, area_true, rtol=0, atol=tol)
    interior = np.zeros_like(fd_new, dtype=bool)
    interior[1:-1, 1:-1] = True
    assert np.all(np.abs(ext[interior]) <= tol)          # external inflow enters at the edge only (the seam is interior)
    assert info["n_seam"] == seam_rc.shape[0]
    assert info["n_uphill_old_receivers"] > 0            # the slab convention does produce uphill codes; they are handled


def test_reentry_is_the_only_residual():
    """Section 2.1.3: with a mosaic in which a seam path leaves the tile and re-enters it, the in-tile correction
    differs from the full recomputation exactly by the re-entered area, and only at and below the re-entry cell."""
    Z, fd_true, area_true, fd_seam, area_seam, r0 = mosaic_with_seam(dx=1.0, drain_margins=False)
    cls = classify_seam(Z, None, fd_seam)
    fd_new, area_new, info = correct_area_in_tile(fd_seam, area_seam, Z, cls["seam_rc"], cls["new_codes"])
    ext_seam = external_inflow(fd_seam, area_seam, Z, 1.0)
    ext_true = external_inflow(fd_true, area_true, Z, 1.0)
    d_ext = ext_seam - ext_true                              # inflow that the seam routing sent out and back in
    edge = np.ones_like(fd_new, dtype=bool)
    edge[1:-1, 1:-1] = False
    assert np.all(d_ext[~edge] == 0)
    # redoing the accumulation with the *true* inflow removes the residual completely
    from lemcvt.analyses.seamfix import accumulate

    fixed = accumulate(fd_new, Z, 1.0 + ext_true)
    assert np.array_equal(fixed, area_true)
    residual = area_new - area_true
    if np.any(d_ext != 0):
        assert np.any(residual != 0)
        assert np.max(np.abs(residual)) <= np.sum(np.abs(d_ext))
    else:
        assert np.array_equal(area_new, area_true)


def test_correction_without_seam_is_identity():
    Z, fd_true, area_true, *_ = mosaic_with_seam(drain_margins=False)
    cls = classify_seam(Z, None, fd_true)
    assert cls["n_seam"] == 0
    ext = external_inflow(fd_true, area_true, Z, 1.0)
    interior = np.zeros_like(fd_true, dtype=bool)
    interior[1:-1, 1:-1] = True
    assert np.all(np.abs(ext[interior]) < 1e-9)
    assert ext[~interior].sum() > 0          # the clipped tile does receive area from the mosaic
