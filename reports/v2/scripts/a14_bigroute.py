"""Addendum 14, decision 7 and 4.1.5 / 4.3.3: depression filling and D8 routing for grids of ~1e9 cells.

The lemcvt ports of TopoAnalysis (lemcvt.hydro.fill.priority_flood_fill, a pure-Python heap; lemcvt.hydro.d8, whole-grid
vectorised with eight float64 temporaries) do not scale to the AP watershed at 1 m (about 9e8 cells). This module keeps
their conventions and changes only the implementation:

  * priority_flood: Barnes et al. (2014) priority flood with the same aggradation increment (1e-12 * dx * distance
    multiplier), the same edge set (valid cells bordering nodata or the grid edge), the same neighbour order and a FIFO
    counter for equal priorities, compiled with numba over an array heap (float64 priority, int32 cell, int32 counter);
  * d8_chunked: lemcvt.hydro.d8.d8_flow_direction applied to overlapping row slabs (halo of 3 rows, the slab's last two
    rows discarded), so the codes equal the whole-grid call cell for cell;
  * d8_area_big: the lemcvt accumulation order (filled elevation descending, ties by flat index) with int32 indices and
    uint32 cell counts (exact; dx = 1 m so count = area in m^2).
"""
from __future__ import annotations

import time

import numba as nb
import numpy as np

from lemcvt.hydro.d8 import d8_flow_direction, d8_receiver_arrays

AGGRADATION_SLOPE = 1e-12
_ROW_KERNEL = np.array([1, 1, 1, 0, 0, -1, -1, -1], dtype=np.int64)
_COL_KERNEL = np.array([-1, 0, 1, -1, 1, -1, 0, 1], dtype=np.int64)
_RT2 = float(np.sqrt(2))
_DX_MULTS = np.array([_RT2, 1.0, _RT2, 1.0, 1.0, _RT2, 1.0, _RT2])


@nb.njit(cache=True)
def _heap_push(pri, idx, cnt, n, p, i, c):
    pri[n] = p
    idx[n] = i
    cnt[n] = c
    k = n
    while k > 0:
        parent = (k - 1) // 2
        if (pri[k] < pri[parent]) or (pri[k] == pri[parent] and cnt[k] < cnt[parent]):
            pri[k], pri[parent] = pri[parent], pri[k]
            idx[k], idx[parent] = idx[parent], idx[k]
            cnt[k], cnt[parent] = cnt[parent], cnt[k]
            k = parent
        else:
            break
    return n + 1


@nb.njit(cache=True)
def _heap_pop(pri, idx, cnt, n):
    p0, i0 = pri[0], idx[0]
    n -= 1
    pri[0], idx[0], cnt[0] = pri[n], idx[n], cnt[n]
    k = 0
    while True:
        left = 2 * k + 1
        right = left + 1
        smallest = k
        if left < n and ((pri[left] < pri[smallest]) or (pri[left] == pri[smallest] and cnt[left] < cnt[smallest])):
            smallest = left
        if right < n and ((pri[right] < pri[smallest]) or (pri[right] == pri[smallest] and cnt[right] < cnt[smallest])):
            smallest = right
        if smallest == k:
            break
        pri[k], pri[smallest] = pri[smallest], pri[k]
        idx[k], idx[smallest] = idx[smallest], idx[k]
        cnt[k], cnt[smallest] = cnt[smallest], cnt[k]
        k = smallest
    return p0, i0, n


@nb.njit(cache=True)
def _flood(Z, valid, dx, rowk, colk, mults, pri, idx, cnt):
    ny, nx = Z.shape
    closed = ~valid
    n = 0
    counter = 0
    # edge set: valid cells with a nodata/outside neighbour (TopoAnalysis findDEMedge: dilation of the nodata mask with border 1)
    for r in range(ny):
        for c in range(nx):
            if not valid[r, c]:
                continue
            edge = False
            for k in range(8):
                rr = r + rowk[k]
                cc = c + colk[k]
                if rr < 0 or rr >= ny or cc < 0 or cc >= nx or not valid[rr, cc]:
                    edge = True
                    break
            if edge:
                closed[r, c] = True
                counter += 1
                n = _heap_push(pri, idx, cnt, n, Z[r, c], r * nx + c, counter)
    while n > 0:
        p, i, n = _heap_pop(pri, idx, cnt, n)
        r = i // nx
        c = i - r * nx
        elev = Z[r, c]
        for k in range(8):
            rr = r + rowk[k]
            cc = c + colk[k]
            if rr < 0 or rr >= ny or cc < 0 or cc >= nx:
                continue
            if not closed[rr, cc]:
                if Z[rr, cc] <= elev:
                    Z[rr, cc] = elev + AGGRADATION_SLOPE * dx * mults[k]
                closed[rr, cc] = True
                counter += 1
                n = _heap_push(pri, idx, cnt, n, Z[rr, cc], rr * nx + cc, counter)
    return Z


def priority_flood(z: np.ndarray, valid: np.ndarray, dx: float) -> np.ndarray:
    """Fill ``z`` (float64, modified in place and returned); ``valid`` marks data cells. Same conventions as lemcvt."""
    n_valid = int(valid.sum())
    pri = np.empty(n_valid + 1, np.float64)
    idx = np.empty(n_valid + 1, np.int32)
    cnt = np.empty(n_valid + 1, np.int32)
    t0 = time.time()
    _flood(z, valid, float(dx), _ROW_KERNEL, _COL_KERNEL, _DX_MULTS, pri, idx, cnt)
    print("priority_flood:", n_valid, "cells in", round(time.time() - t0, 1), "s", flush=True)
    return z


def d8_chunked(zf: np.ndarray, block_rows: int = 1500) -> np.ndarray:
    """lemcvt d8_flow_direction on overlapping row slabs; identical codes to the whole-grid call."""
    ny, nx = zf.shape
    fd = np.zeros((ny, nx), np.uint8)
    halo = 3
    r0 = 0
    t0 = time.time()
    while r0 < ny:
        r1 = min(ny, r0 + block_rows)
        s0 = max(0, r0 - halo)
        s1 = min(ny, r1 + halo)
        sub = d8_flow_direction(zf[s0:s1], border="legacy")   # slab borders: halo rows are dropped, grid-border columns are zeroed by the caller (data-edge rule)
        keep0 = r0 - s0
        keep1 = keep0 + (r1 - r0)
        if s1 < ny:
            # the slab's last two rows never see an east/south comparison (TopoAnalysis quirk): they lie in the halo and are dropped
            assert s1 - r1 >= 2
        fd[r0:r1] = sub[keep0:keep1]
        r0 = r1
    print("d8_chunked in", round(time.time() - t0, 1), "s", flush=True)
    return fd


@nb.njit(cache=True)
def _accumulate(order, target, count):
    for k in order:
        t = target[k]
        if t >= 0:
            count[t] += count[k]
    return count


def d8_area_big(fd: np.ndarray, zf: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Cell counts draining through each cell (uint32), lemcvt order: filled z descending, ties by ascending flat index."""
    ny, nx = fd.shape
    t0 = time.time()
    r_to, c_to, good = d8_receiver_arrays(fd)
    target = np.where(good.ravel() & valid.ravel(), (r_to.ravel().astype(np.int64) * nx + c_to.ravel()), -1).astype(np.int64)
    del r_to, c_to, good
    # receivers outside the valid data are roots
    tv = target >= 0
    target[tv & ~valid.ravel()[np.maximum(target, 0)]] = -1
    zff = np.where(valid, zf, -np.inf).ravel()
    order = np.lexsort((np.arange(zff.size, dtype=np.int64), -zff))
    del zff
    count = np.ones(ny * nx, np.uint32)
    count[~valid.ravel()] = 0
    _accumulate(order, target, count)
    print("d8_area_big in", round(time.time() - t0, 1), "s", flush=True)
    return count.reshape(ny, nx), target
