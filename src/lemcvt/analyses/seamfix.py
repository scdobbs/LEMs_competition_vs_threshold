"""AP1's flow-direction seam, corrected within the tile (addendum 5, section 2.1).

The stored AP1 ``*_fd`` grid carries, on rows 389-390 (0-based), 345 interior cells whose codes are not
the steepest descent of the filled grid (checkpoint 4b/5, section 4.2): the signature of TopoAnalysis's
slab convention applied at a processing-block boundary of the mosaic (the last two rows of a block get
no south comparison). The stored ``*_area`` grid was accumulated through those directions.

The correction is exact inside the tile, without the mosaic. A seam cell's stored area holds all of its
upstream area, including area from outside the tile, so rerouting the cell only moves that quantity
from its old downstream path to its new one:

1. corrected direction: the steepest-descent code of the (metre) filled grid with the TopoAnalysis
   convention (:func:`lemcvt.hydro.d8.d8_flow_direction`; the seam rows are interior, so the slab
   quirk does not touch them);
2. corrected area: the seam cells are processed upstream first (stored area ascending); for each, its
   current area is subtracted along the old D8 path from the old receiver to the tile exit and added
   along the new path, both followed in the current state of the direction grid, so that seam cells
   downstream of other seam cells are handled with their already-corrected areas.

One rule makes the correction reproduce TopoAnalysis's accumulation exactly: ``Area`` passes a cell's
area to its receiver in elevation order (highest cell first, ties by ascending flat index), so a cell
that points *uphill* (which the slab convention can produce: the code goes to the least-uphill of the
remaining neighbours) hands its area to that receiver but the receiver, processed earlier, never passes
it on. The path updates therefore propagate only while the path is "processed later" in that order
(:func:`passes_on`). With that rule, the in-tile correction equals a full recomputation on the mosaic
clipped to the tile (``tests/test_seamfix.py``), and :func:`recompute_with_external_inflow` provides an
independent check on the real tile: the inflow entering each cell from outside the tile is read off the
stored grids and the accumulation is redone on the corrected directions.

What the in-tile correction cannot see is a path that leaves the tile and re-enters it: the area it
carries would re-enter as "external inflow" at an edge cell and not move with the rerouting. Section
2.1.3: the exits of every old and new path are counted and reported together with the edge cells that
receive external inflow.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from lemcvt.config import TileData, config_snapshot, data_root, load_tile, load_tile_config
from lemcvt.hydro.d8 import D8_OFFSETS, d8_flow_direction
from lemcvt.io.grids import Grid, read_geotiff, sha256_file, write_geotiff
from lemcvt.io.manifest import MANIFEST_PATH, ManifestEntry, read_manifest, write_manifest
from lemcvt.results import RunRecorder, find_runs

logger = logging.getLogger(__name__)

_CODES = (1, 2, 4, 8, 16, 32, 64, 128)
_DIV = {1: 1.0, 2: 1.41, 4: 1.0, 8: 1.41, 16: 1.0, 32: 1.41, 64: 1.0, 128: 1.41}


# ----------------------------------------------------------------------------- graph helpers

def receiver_rc(code: int, r: int, c: int, ny: int, nx: int) -> tuple[int, int] | None:
    """Receiver of ``(r, c)`` for ``code``; None when the code is 0 or the receiver is outside the grid."""
    if code not in D8_OFFSETS:
        return None
    dr, dc = D8_OFFSETS[code]
    rr, cc = r + dr, c + dc
    if 0 <= rr < ny and 0 <= cc < nx:
        return rr, cc
    return None


def passes_on(Z: np.ndarray, donor: tuple[int, int], recv: tuple[int, int]) -> bool:
    """Whether TopoAnalysis's elevation-ordered accumulation passes ``donor``'s area on through ``recv``.

    Cells are processed from highest to lowest filled elevation, ties in ascending flat index (N1). A
    donor's area reaches its receiver always (the donor adds to it when processed), but the receiver
    passes it further only if the receiver is processed *after* the donor: lower, or equal and of
    higher flat index. NaN receivers never pass anything on.
    """
    zd, zr = Z[donor], Z[recv]
    if not np.isfinite(zr):
        return False
    if zr < zd:
        return True
    if zr == zd:
        nx = Z.shape[1]
        return recv[0] * nx + recv[1] > donor[0] * nx + donor[1]
    return False


def update_along_path(fd: np.ndarray, area: np.ndarray, Z: np.ndarray, first: tuple[int, int] | None,
                      donor: tuple[int, int], delta: float, max_steps: int | None = None) -> dict[str, Any]:
    """Add ``delta`` to ``first`` and to every cell downstream of it while the accumulation passes on.

    Follows the *current* ``fd``. A cell on the path receives the amount unconditionally (its donor adds to
    it when the donor is processed) and forwards it only if it is processed after that donor
    (:func:`passes_on`). The forwarding chain is therefore strictly increasing in processing order and
    terminates; the last cell of a two-cell loop (a cell pointing uphill at the cell that points at it)
    receives twice, as TopoAnalysis's accumulation does. Stops at the tile exit (receiver off-grid or
    code 0) or where the rule stops the propagation. Returns the cells touched and the exit record.
    """
    ny, nx = fd.shape
    max_steps = max_steps or ny * nx
    touched = 0
    cur = first
    prev = donor
    stop = "no_receiver"
    exit_cell = None
    exit_code = None
    while cur is not None and touched < max_steps:
        area[cur] += delta
        touched += 1
        if not passes_on(Z, prev, cur):
            stop = "uphill_receiver_processed_first"
            exit_cell = cur
            break
        code = int(fd[cur])
        nxt = receiver_rc(code, cur[0], cur[1], ny, nx)
        if nxt is None:
            stop = "tile_exit" if code in D8_OFFSETS else "code_zero"
            exit_cell, exit_code = cur, code
            break
        prev, cur = cur, nxt
    else:
        if cur is not None:
            stop = "max_steps"
    on_edge = exit_cell is not None and (exit_cell[0] in (0, ny - 1) or exit_cell[1] in (0, nx - 1))
    return {"n_cells": touched, "stop": stop, "exit_cell": list(exit_cell) if exit_cell else None, "exit_code": exit_code,
            "exit_on_edge": bool(on_edge)}


def processed_after(Z: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True when cell ``a`` is processed after cell ``b`` in the accumulation order (lower, or equal and higher index)."""
    za, zb = Z[a], Z[b]
    if za < zb:
        return True
    if za == zb:
        nx = Z.shape[1]
        return a[0] * nx + a[1] > b[0] * nx + b[1]
    return False


def passed_local(fd: np.ndarray, area: np.ndarray, Z: np.ndarray, cell: tuple[int, int], _seen: set | None = None) -> float:
    """What ``cell`` handed to its receiver under the current grids: its area minus what its *late* donors (cells
    pointing at it that are processed after it, i.e. uphill-pointing neighbours) added after it had passed its total on.
    Recursive over the late donors; they form short chains at most."""
    ny, nx = fd.shape
    seen = _seen if _seen is not None else set()
    seen.add(cell)
    r, c = cell
    total = float(area[r, c])
    for code, (dr, dc) in D8_OFFSETS.items():
        rr, cc = r - dr, c - dc                 # the neighbour that would point at (r, c) with this code
        if 0 <= rr < ny and 0 <= cc < nx and int(fd[rr, cc]) == code and (rr, cc) not in seen:
            if processed_after(Z, (rr, cc), cell):
                total -= passed_local(fd, area, Z, (rr, cc), seen)
    return total


def correct_area_in_tile(fd_old: np.ndarray, area_old: np.ndarray, Z: np.ndarray, seam_rc: np.ndarray,
                         new_codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """The directive's algorithm (2.1.2): reroute the seam cells one by one, upstream first, moving each cell's
    current area from its old path to its new one. Returns ``(fd_new, area_new, info)``.

    The quantity moved is what the cell passed to its receiver (:func:`passed_local`); it equals the cell's stored
    area unless a neighbour points uphill at the cell (then the receiver, processed earlier, had handed its total
    back and the stored area counts that return).
    """
    fd = fd_old.copy()
    area = area_old.astype(np.float64).copy()
    ny, nx = fd.shape
    order = np.argsort(area_old[seam_rc[:, 0], seam_rc[:, 1]], kind="stable")      # stored area ascending: upstream first
    records = []
    for k in order.tolist():
        r, c = int(seam_rc[k, 0]), int(seam_rc[k, 1])
        a = passed_local(fd, area, Z, (r, c))
        old_code, new_code = int(fd[r, c]), int(new_codes[k])
        r_old = receiver_rc(old_code, r, c, ny, nx)
        r_new = receiver_rc(new_code, r, c, ny, nx)
        rec_old = update_along_path(fd, area, Z, r_old, (r, c), -a)
        fd[r, c] = new_code
        rec_new = update_along_path(fd, area, Z, r_new, (r, c), +a)
        records.append({"row": r, "col": c, "area_moved": a, "area_stored_at_processing": float(area[r, c]) if new_code == old_code else None,
                        "old_code": old_code, "new_code": new_code,
                        "drop_to_old_receiver": float(Z[r, c] - Z[r_old]) if r_old else None,
                        "drop_to_new_receiver": float(Z[r, c] - Z[r_new]) if r_new else None,
                        "old_path": rec_old, "new_path": rec_new})
    info = {"n_seam": int(seam_rc.shape[0]), "records": records,
            "n_old_paths_exit_tile": sum(1 for q in records if q["old_path"]["stop"] == "tile_exit"),
            "n_new_paths_exit_tile": sum(1 for q in records if q["new_path"]["stop"] == "tile_exit"),
            "n_old_paths_stopped_uphill": sum(1 for q in records if q["old_path"]["stop"] == "uphill_receiver_processed_first"),
            "n_new_paths_stopped_uphill": sum(1 for q in records if q["new_path"]["stop"] == "uphill_receiver_processed_first"),
            "n_uphill_old_receivers": sum(1 for q in records if q["drop_to_old_receiver"] is not None and q["drop_to_old_receiver"] <= 0),
            "exit_cells_old": sorted({tuple(q["old_path"]["exit_cell"]) for q in records if q["old_path"]["exit_cell"]}),
            "exit_cells_new": sorted({tuple(q["new_path"]["exit_cell"]) for q in records if q["new_path"]["exit_cell"]})}
    return fd, area, info


def _order_and_targets(fd: np.ndarray, Z: np.ndarray):
    """TopoAnalysis accumulation order (Z descending, ties ascending flat index) and flat receiver per cell."""
    from lemcvt.hydro.d8 import d8_receiver_arrays

    ny, nx = fd.shape
    Zf = np.asarray(Z, dtype=np.float64).ravel()
    order = np.lexsort((np.arange(Zf.size), -Zf))
    r_to, c_to, good = d8_receiver_arrays(fd)
    target = np.where(good.ravel(), r_to.ravel() * nx + c_to.ravel(), -1)
    return order, target


def accumulate(fd: np.ndarray, Z: np.ndarray, init: np.ndarray) -> np.ndarray:
    """Elevation-ordered D8 accumulation (as ``d8_area``) starting from ``init`` instead of ``dx**2``."""
    from lemcvt.hydro.d8 import _accumulate

    order, target = _order_and_targets(fd, Z)
    area = np.asarray(init, dtype=np.float64).copy()
    _accumulate(order, target, area.ravel())
    return area


def _passed_py(order, target, area_f):
    """Amount each cell handed to its receiver in the elevation-ordered accumulation.

    ``passed[k] = A[k] - sum of passed[e] over donors e processed after k`` (a donor lower than its receiver
    adds to it after the receiver has already passed its own total on). Computed in reverse processing order.
    """
    n = order.shape[0]
    pos = np.empty(n, dtype=np.int64)
    for i in range(n):
        pos[order[i]] = i
    late = np.zeros(n)
    passed = np.zeros(n)
    for i in range(n - 1, -1, -1):
        k = order[i]
        passed[k] = area_f[k] - late[k]
        t = target[k]
        if t >= 0 and pos[t] < i:
            late[t] += passed[k]
    return passed


try:
    import numba as _nb

    _passed_nb = _nb.njit(cache=True)(_passed_py)
    _passed = _passed_nb
except Exception:  # noqa: BLE001
    _passed = _passed_py


def passed_amounts(fd: np.ndarray, area: np.ndarray, Z: np.ndarray) -> np.ndarray:
    order, target = _order_and_targets(fd, Z)
    af = np.ascontiguousarray(np.asarray(area, dtype=np.float64).ravel())
    return np.asarray(_passed(np.ascontiguousarray(order), np.ascontiguousarray(target), af)).reshape(fd.shape)


def external_inflow(fd: np.ndarray, area: np.ndarray, Z: np.ndarray, dx: float) -> np.ndarray:
    """Area entering each cell from outside the tile, read off the stored grids.

    ``ext(c) = A(c) - dx^2 - sum over in-tile donors d of passed(d)``, where ``passed(d)`` is what the
    elevation-ordered accumulation handed from d to c (equal to A(d) unless d has donors that were processed
    after it, see :func:`passed_amounts`). Nonzero away from the edge only where the stored grids are inconsistent.
    """
    ny, nx = fd.shape
    order, target = _order_and_targets(fd, Z)
    passed = passed_amounts(fd, area, Z).ravel()
    donors_sum = np.zeros(ny * nx)
    ok = target >= 0
    np.add.at(donors_sum, target[ok], passed[ok])
    af = np.asarray(area, dtype=np.float64).ravel()
    ext = af - dx**2 - donors_sum
    return ext.reshape(ny, nx)


def recompute_with_external_inflow(fd_new: np.ndarray, fd_old: np.ndarray, area_old: np.ndarray, Z: np.ndarray,
                                   dx: float) -> tuple[np.ndarray, np.ndarray]:
    """Independent full recomputation on the corrected directions: accumulate ``dx^2 + ext`` on ``fd_new``."""
    ext = external_inflow(fd_old, area_old, Z, dx)
    return accumulate(fd_new, Z, dx**2 + ext), ext


# ----------------------------------------------------------------------------- consistency of the stored grids

def _inconsistency(fd: np.ndarray, A: np.ndarray, Z: np.ndarray, cell: float, tol: float = 0.5) -> dict[str, Any]:
    """Interior cells where the stored area is not the accumulation of its in-tile donors under ``fd`` (|ext| > tol cell)."""
    ext = external_inflow(fd, A, Z, math.sqrt(cell))
    interior = np.zeros(fd.shape, dtype=bool)
    interior[1:-1, 1:-1] = True
    bad = interior & np.isfinite(ext) & (np.abs(ext) > tol * cell)
    rows = np.unique(np.argwhere(bad)[:, 0]) if bad.any() else np.array([], dtype=int)
    v = ext[bad] / cell
    return {"n_inconsistent_interior": int(bad.sum()), "rows": rows.tolist()[:20], "n_rows": int(rows.size),
            "per_row": {str(int(r)): int(np.sum(np.argwhere(bad)[:, 0] == r)) for r in rows[:20]},
            "abs_quantiles_cells": _quantiles(np.abs(v), (50, 90, 100)) if v.size else {}, "sum_positive_cells": float(v[v > 0].sum()) if v.size else 0.0,
            "sum_negative_cells": float(v[v < 0].sum()) if v.size else 0.0, "tolerance_cells": tol}


def grid_consistency(Z: np.ndarray, A0: np.ndarray, fd0: np.ndarray, fd_new: np.ndarray, A_new: np.ndarray, cell: float,
                     seam_rows: tuple[int, int] = (389, 390)) -> dict[str, Any]:
    """Three checks the correction's premise rests on: the stored area against the stored directions, the stored area
    against the corrected directions, and the elevation step across the seam rows against its neighbours."""
    out: dict[str, Any] = {"cell_area_m2": cell, "stored_area_vs_stored_fd": _inconsistency(fd0, A0, Z, cell),
                           "stored_area_vs_corrected_fd": _inconsistency(fd_new, A0, Z, cell)}
    r0 = seam_rows[0]
    if r0 >= 3 and r0 + 3 < Z.shape[0]:
        ref = ((Z[r0 - 2] - Z[r0 - 1]) + (Z[r0 - 1] - Z[r0]) + (Z[r0 + 1] - Z[r0 + 2]) + (Z[r0 + 2] - Z[r0 + 3])) / 4
        steps = {}
        for r in range(r0 - 2, r0 + 3):
            d = (Z[r] - Z[r + 1] - ref)[2:-2]
            d = d[np.isfinite(d)]
            steps[f"{r}->{r + 1}"] = {"median_m": float(np.median(d)), "q5_m": float(np.percentile(d, 5)), "q95_m": float(np.percentile(d, 95)),
                                     "mean_drop_m": float(np.nanmean((Z[r] - Z[r + 1])[2:-2]))}
        out["elevation_step_minus_local_reference"] = steps
    neg = np.isfinite(A_new) & (A_new < 0.5 * cell)
    out["corrected_area"] = {"n_below_one_cell": int(neg.sum()), "min_cells": float(np.nanmin(A_new) / cell),
                             "rows_below_one_cell": np.unique(np.argwhere(neg)[:, 0]).tolist()[:20] if neg.any() else []}
    out["premise_holds"] = bool(out["stored_area_vs_stored_fd"]["n_inconsistent_interior"] == 0)
    out["note"] = ("the directive's correction moves each seam cell's stored area along its paths, which is exact only if the stored "
                   "area was accumulated through the stored directions (premise_holds). Where it was not, the moved amounts are "
                   "wrong and the result can be negative.")
    return out


# ----------------------------------------------------------------------------- seam identification

def classify_seam(Z_m: np.ndarray, Z_ft: np.ndarray | None, fd0: np.ndarray, rel_tol: float = 1e-9, abs_tol: float = 1e-9,
                  edge_margin: int = 2) -> dict[str, Any]:
    """Interior cells whose stored code is not the steepest descent and not a tie (the checkpoint's 'anything else').

    Recomputes the directions on the metre grid (and, when given, on the stored-unit grid, which must agree on the
    seam cells: scaling z does not change which neighbour is steepest except on exact ties). The new code is the
    metre recomputation.
    """
    from lemcvt.analyses.ap_units import _drops

    ny, nx = Z_m.shape
    fd0 = fd0.astype(np.int64)
    fd_m = d8_flow_direction(Z_m).astype(np.int64)
    fd_ft = d8_flow_direction(Z_ft).astype(np.int64) if Z_ft is not None else fd_m
    rows, cols = np.indices((ny, nx))
    margin = np.minimum(np.minimum(rows, ny - 1 - rows), np.minimum(cols, nx - 1 - cols))
    drops = _drops(Z_m)
    best = np.max(np.stack(list(drops.values())), axis=0)
    drop_orig = np.full(Z_m.shape, np.nan)
    for code, d in drops.items():
        sel = fd0 == code
        drop_orig[sel] = d[sel]
    tie = np.isfinite(drop_orig) & (np.abs(drop_orig - best) <= np.maximum(rel_tol * np.abs(best), abs_tol))
    interior = margin >= edge_margin
    seam = interior & (fd_ft != fd0) & ~tie
    rc = np.argwhere(seam)
    agree = fd_m[seam] == fd_ft[seam]
    return {"seam_rc": rc, "new_codes": fd_m[seam], "old_codes": fd0[seam], "n_seam": int(rc.shape[0]),
            "rows": sorted({int(r) for r in rc[:, 0]}), "metre_feet_codes_agree_on_seam": bool(np.all(agree)),
            "n_metre_feet_disagree_on_seam": int(np.sum(~agree)),
            "drop_best_minus_orig": (best - drop_orig)[seam], "drop_to_orig": drop_orig[seam],
            "n_interior_diff_total": int((interior & (fd_ft != fd0)).sum()), "n_interior_ties": int((interior & (fd_ft != fd0) & tie).sum())}


# ----------------------------------------------------------------------------- driver

def _joint_acrit_from_checkpoint(tile: str) -> dict[str, Any] | None:
    """The tile's v2 joint A_crit of the latest refined partitioned run (the checkpoint value, unit-invariant)."""
    cands = [r for r in find_runs("tile_run", tile) if r.name.endswith("_v2_partitioned-refined")]
    for r in reversed(cands):
        try:
            p = json.loads((r / "params.json").read_text())
            jo = p["v2"]["partitioned"]["joint_optimum"]
            s2 = p["v2"]["partitioned"]["stage2_optimum"]
            return {"run": r.name, "joint_A_crit": float(jo["A_crit"]), "stage2_A_crit": float(s2["A_crit"])}
        except Exception:  # noqa: BLE001
            continue
    return None


def _quantiles(v: np.ndarray, qs=(0, 1, 5, 25, 50, 75, 95, 99, 100)) -> dict[str, float]:
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return {str(q): float(np.percentile(v, q)) for q in qs} if v.size else {}


def run_seamfix(tile: str = "AP1", write: bool = True, update_manifest: bool = True, run_id: str | None = None,
                results_root=None, expected_rows: tuple[int, ...] | None = (389, 390)) -> RunRecorder:
    """Section 2.1 on one tile: identify the seam, correct fd and area in the tile, check against the independent
    recomputation, report the area and channel-mask changes, write ``*_fd_seamfix.tif`` / ``*_area_seamfix.tif``."""
    rec = RunRecorder("seamfix", tile, run_id=run_id, root=results_root)
    t0 = time.perf_counter()
    cfg = load_tile_config(tile, reproduce_v1_units=False)
    td: TileData = load_tile(cfg)
    stored_unit_differs = cfg.z_units != "m"
    td_ft = load_tile(tile, reproduce_v1_units=True) if stored_unit_differs else None
    Z, A0, FD0, dx = td.Z, td.A, td.FD.astype(np.int64), td.dx
    ny, nx = Z.shape
    rec.add_params("tile", tile)
    rec.add_params("inputs", td.provenance())

    cls = classify_seam(Z, td_ft.Z if td_ft is not None else None, FD0)
    seam_rc, new_codes = cls["seam_rc"], cls["new_codes"]
    logger.info("%s: %d seam cells on rows %s (metre/feet recomputations agree on them: %s)", tile, cls["n_seam"], cls["rows"],
                cls["metre_feet_codes_agree_on_seam"])
    seam_info = {k: v for k, v in cls.items() if k not in ("seam_rc", "new_codes", "old_codes", "drop_best_minus_orig", "drop_to_orig")}
    seam_info["old_code_counts"] = {str(int(c)): int(np.sum(cls["old_codes"] == c)) for c in np.unique(cls["old_codes"])}
    seam_info["new_code_counts"] = {str(int(c)): int(np.sum(new_codes == c)) for c in np.unique(new_codes)}
    seam_info["excess_drop_best_minus_orig_m"] = _quantiles(cls["drop_best_minus_orig"])
    seam_info["drop_to_orig_receiver_m"] = _quantiles(cls["drop_to_orig"])
    seam_info["n_orig_receiver_uphill_or_level"] = int(np.sum(cls["drop_to_orig"] <= 0))
    seam_info["cols_min_max"] = [int(seam_rc[:, 1].min()), int(seam_rc[:, 1].max())] if seam_rc.size else None
    if expected_rows is not None:
        seam_info["rows_as_expected"] = bool(set(cls["rows"]) <= set(expected_rows))
    rec.add_params("seam", seam_info)

    # 2.1.1-2.1.2: the correction
    fd_new, A_new, info = correct_area_in_tile(FD0, A0, Z, seam_rc, new_codes)
    # independent check: external inflow + accumulation on the corrected directions
    A_chk, ext = recompute_with_external_inflow(fd_new, FD0, A0, Z, dx)
    valid = np.isfinite(A0)
    diff_chk = np.abs(A_new - A_chk)[valid]
    cell = dx * dx
    interior = np.zeros_like(valid)
    interior[1:-1, 1:-1] = True
    check = {"max_abs_diff_cells": float(diff_chk.max() / cell), "n_cells_differing_gt_1e-6_cell": int(np.sum(diff_chk > 1e-6 * cell)),
             "external_inflow_edge_cells_gt_0": int(np.sum((ext > 1e-6 * cell) & valid & ~interior)),
             "external_inflow_interior_cells_gt_0": int(np.sum((ext > 1e-6 * cell) & valid & interior)),
             "external_inflow_negative_cells": int(np.sum((ext < -1e-6 * cell) & valid)),
             "external_inflow_total_cells": float(np.nansum(np.where(valid, ext, 0.0)) / cell),
             "note": "ext = stored area - cell - sum of in-tile donors' stored areas; the accumulation is redone on the corrected "
                     "directions from cell + ext (TopoAnalysis elevation order). Agreement with the path correction is the test "
                     "of section 2.1 on the real tile; the synthetic test is tests/test_seamfix.py."}
    rec.add_params("independent_recomputation", check)
    logger.info("%s: path correction against external-inflow recomputation: max |dA| = %.3g cells over %d cells", tile,
                check["max_abs_diff_cells"], int(valid.sum()))

    # consistency of the three stored grids on and around the seam (added after the first AP1 run: the correction
    # assumes the stored area was accumulated through the stored directions, which must be checked, not assumed)
    cell = float(np.nanmin(A0))
    cons = grid_consistency(Z, A0, FD0, fd_new, A_new, cell)
    rec.add_params("consistency", cons)
    logger.info("%s: stored area inconsistent with the stored fd at %d interior cells (rows %s); corrected area below one cell at %d cells (min %.0f cells)",
                tile, cons["stored_area_vs_stored_fd"]["n_inconsistent_interior"], cons["stored_area_vs_stored_fd"]["rows"],
                cons["corrected_area"]["n_below_one_cell"], cons["corrected_area"]["min_cells"])

    # 2.1.3 re-entry
    reentry = {k: info[k] for k in ("n_old_paths_exit_tile", "n_new_paths_exit_tile", "n_old_paths_stopped_uphill",
                                   "n_new_paths_stopped_uphill", "n_uphill_old_receivers")}
    reentry.update({"n_distinct_exit_cells_old": len(info["exit_cells_old"]), "n_distinct_exit_cells_new": len(info["exit_cells_new"]),
                    "exit_cells_old": [list(e) for e in info["exit_cells_old"]][:50], "exit_cells_new": [list(e) for e in info["exit_cells_new"]][:50],
                    "edge_cells_with_external_inflow": check["external_inflow_edge_cells_gt_0"],
                    "statement": "every path ends at a tile exit; whether it re-enters cannot be seen from the tile. Re-entering area "
                                 "would appear as external inflow at an edge cell and would not move with the rerouting; the count of "
                                 "edge cells receiving external inflow bounds how many such cells exist."})
    rec.add_params("reentry", reentry)

    # 2.1.4 report quantities
    changed = valid & (A_new != A0)
    rel = np.where(changed, (A_new - A0) / A0, np.nan)
    big = valid & (A0 >= 1e3) & (np.abs(np.nan_to_num(rel)) > 0.01)
    ck = _joint_acrit_from_checkpoint(tile)
    masks = {}
    for name, a_c in (("joint", ck["joint_A_crit"] if ck else None), ("stage2", ck["stage2_A_crit"] if ck else None), ("1e3", 1e3), ("1e4", 1e4)):
        if a_c is None:
            continue
        m0, m1 = valid & (A0 >= a_c), valid & (A_new >= a_c)
        masks[name] = {"A_crit": float(a_c), "n_channel_old": int(m0.sum()), "n_channel_new": int(m1.sum()),
                       "n_gained": int((m1 & ~m0).sum()), "n_lost": int((m0 & ~m1).sum()), "n_differ": int((m0 != m1).sum())}
    changes = {"n_cells_area_changed": int(changed.sum()), "frac_cells_area_changed": float(changed.sum() / valid.sum()),
               "rel_change_quantiles": _quantiles(rel[changed]), "abs_rel_change_quantiles": _quantiles(np.abs(rel[changed])),
               "n_increase": int(np.sum(changed & (A_new > A0))), "n_decrease": int(np.sum(changed & (A_new < A0))),
               "max_abs_change_cells": float(np.nanmax(np.abs(A_new - A0)[valid]) / cell),
               "n_cells_A_ge_1e3_changed_gt_1pct": int(big.sum()), "n_cells_A_ge_1e3": int(np.sum(valid & (A0 >= 1e3))),
               "rows_touched": sorted({int(r) for r in np.argwhere(changed)[:, 0]})[:5] + (["..."] if changed.sum() > 5 else []),
               "n_rows_touched": int(np.unique(np.argwhere(changed)[:, 0]).size), "area_total_old_cells": float(np.nansum(A0[valid]) / cell),
               "area_total_new_cells": float(np.nansum(A_new[valid]) / cell), "channel_mask": masks, "checkpoint_acrit": ck}
    rec.add_params("changes", changes)
    rec.add_params("correction", {k: v for k, v in info.items() if k != "records"} | {"records_first_20": info["records"][:20]})
    rec.write_grid(td.filled.with_data((A_new - A0).astype(np.float64)), f"{tile}_area_seamfix_minus_stored")
    logger.info("%s: %d cells change area (%.2f %% of valid), %d cells with A >= 1e3 change by > 1 %%; joint mask differs on %s cells", tile,
                changes["n_cells_area_changed"], 100 * changes["frac_cells_area_changed"], changes["n_cells_A_ge_1e3_changed_gt_1pct"],
                masks.get("joint", {}).get("n_differ"))

    # 2.1.5 write the corrected grids, keep the originals
    written: dict[str, Any] = {}
    if write:
        root = data_root()
        for key, data, dtype in (("fd", fd_new.astype(np.uint8), "uint8"), ("area", A_new, "float64")):
            src = root / cfg.grids[key]
            g0 = read_geotiff(src, dtype="uint8" if key == "fd" else "float64")
            rel_path = str(Path(cfg.grids[key]).with_name(Path(cfg.grids[key]).stem + "_seamfix.tif"))
            dst = root / rel_path
            write_geotiff(Grid(data=np.ascontiguousarray(data), transform=g0.transform, crs=g0.crs, nodata=None, name=dst.stem), dst, dtype=dtype)
            back = read_geotiff(dst, dtype=dtype).data
            written[key] = {"path": rel_path, "sha256": sha256_file(dst), "bytes": dst.stat().st_size,
                            "round_trip_identical": bool(np.array_equal(back, data, equal_nan=True)), "original": cfg.grids[key],
                            "original_sha256": cfg.sha256[key]}
            logger.info("%s: wrote %s (sha256 %s)", tile, dst.name, written[key]["sha256"][:12])
        if update_manifest:
            entries = {e.path: e for e in read_manifest()}
            for key, w in written.items():
                entries[w["path"]] = ManifestEntry(w["path"], w["sha256"], w["bytes"], cfg.site, f"tile{cfg.index}", "input_seamfix", f"{key}_seamfix")
            ordered = sorted(entries.values(), key=lambda e: (e.site, e.tile, e.kind, e.path))
            write_manifest(ordered, MANIFEST_PATH)
            written["manifest"] = {"path": str(MANIFEST_PATH), "n_entries": len(ordered)}
    rec.add_params("written", written)
    rec.add_params("config_update", {"grids.fd": written.get("fd", {}).get("path"), "grids.area": written.get("area", {}).get("path"),
                                     "grids.sha256.fd": written.get("fd", {}).get("sha256"), "grids.sha256.area": written.get("area", {}).get("sha256"),
                                     "note": "point configs/tiles/AP1.yaml at these; keep the originals under v1_reference.grids"})
    rec.add_params("wall_s", time.perf_counter() - t0)
    rec.finalize(config_snapshot(cfg))
    return rec
