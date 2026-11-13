"""Stratified sampling of channel pixels with logged exclusions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .network import FlowGraph
from .tiles import Tile
from .transect import make_transect
from .widths import confinement, find_thalweg


@dataclass
class Candidate:
    site: str
    tile: int
    row: int
    col: int
    x: float
    y: float
    area_m2: float
    bin_index: int
    local_slope: float = np.nan
    reach_median_slope: float = np.nan
    slope_method: str = ""
    transect: Any = None
    extra: dict = field(default_factory=dict)


def site_area_bins(a_min: float, a_max: float, n_bins: int) -> np.ndarray:
    """Log-spaced bin edges (length n_bins + 1) from a_min to a_max."""
    return np.logspace(np.log10(a_min), np.log10(a_max), n_bins + 1)


def _quotas(n_bins: int, mode: str, cfg: dict) -> list[int]:
    if mode == "pilot":
        total = int(cfg["pilot_points_per_site"])
        q = [total // n_bins] * n_bins
        for k in range(total - sum(q)):          # round-robin the remainder
            q[k % n_bins] += 1
        return q
    return [int(cfg["full_points_per_bin"])] * n_bins


def draw_site_sample(
    site: str,
    tiles: dict[int, Tile],
    graphs: dict[int, FlowGraph],
    cfg: dict,
    mode: str,
    rng: np.random.Generator,
) -> tuple[list[Candidate], list[dict], np.ndarray]:
    """Draw the stratified sample for one site.

    Returns (accepted candidates, exclusion log rows, bin edges). Bins are
    defined once per site, from ``a_min_sample_m2`` to the largest drainage
    area found in any of the site's tiles, so that bins are comparable across
    tiles. Candidates are tested in random order within each bin; every tested
    and rejected candidate is logged with its reason. Exclusion tests run
    cheapest first: tile edge, confluence, separation from accepted points,
    knickpoint (local slope > factor x reach median), filled-depression proxy
    (no fall along the trace), confinement (no rise on both sides of the
    thalweg within the window).
    """
    a_min = float(cfg["a_min_sample_m2"])
    n_bins = int(cfg["n_area_bins"])
    a_max = max(float(np.nanmax(t.area)) for t in tiles.values())
    edges = site_area_bins(a_min, a_max, n_bins)

    # pool candidates over tiles
    pools: list[list[Candidate]] = [[] for _ in range(n_bins)]
    for tnum, t in tiles.items():
        rows, cols = np.nonzero(t.area >= a_min)
        a = t.area[rows, cols]
        b = np.clip(np.searchsorted(edges, a, side="right") - 1, 0, n_bins - 1)
        x, y = t.rc_to_xy(rows, cols)
        for r, c, aa, bb, xx, yy in zip(rows, cols, a, b, x, y, strict=False):
            pools[bb].append(Candidate(site, tnum, int(r), int(c), float(xx), float(yy), float(aa), int(bb)))
    for p in pools:
        rng.shuffle(p)

    quotas = _quotas(n_bins, mode, cfg)
    accepted: list[Candidate] = []
    excluded: list[dict] = []
    cursors = [0] * n_bins

    def log(c: Candidate, reason: str, **kw):
        row = dict(site=c.site, tile=c.tile, row=c.row, col=c.col, x=c.x, y=c.y,
                   area_m2=c.area_m2, bin_index=c.bin_index, reason=reason)
        row.update(kw)
        excluded.append(row)

    for bi in range(n_bins):
        got: int = 0
        while got < quotas[bi] and cursors[bi] < len(pools[bi]):
            c = pools[bi][cursors[bi]]
            cursors[bi] += 1
            g = graphs[c.tile]
            if g.edge_distance(c.row, c.col) < cfg["edge_exclusion_m"]:
                log(c, "tile_edge")
                continue
            dconf = g.distance_to_confluence(c.row, c.col, cfg["confluence_exclusion_m"])
            if dconf <= cfg["confluence_exclusion_m"]:
                log(c, "confluence", distance_m=dconf)
                continue
            same_tile = [a for a in accepted if a.tile == c.tile]
            if same_tile:
                dmin = min(np.hypot(a.x - c.x, a.y - c.y) for a in same_tile)
                if dmin < cfg["min_separation_m"]:
                    log(c, "too_close_to_sample", distance_m=float(dmin))
                    continue
            s0, method = g.slope_with_fallback(c.row, c.col, cfg["vertical_interval_m"],
                                               cfg["horizontal_interval_fallback_m"])
            if not np.isfinite(s0):
                log(c, "slope_undefined")
                continue
            smed, nreach = g.reach_median_slope(c.row, c.col, cfg["reach_half_length_m"],
                                                cfg["vertical_interval_m"], cfg["horizontal_interval_fallback_m"])
            if not np.isfinite(smed):
                log(c, "reach_slope_undefined", local_slope=s0)
                continue
            if s0 > cfg["knickpoint_factor"] * smed:
                log(c, "knickpoint", local_slope=s0, reach_median_slope=smed)
                continue
            tr = make_transect(tiles[c.tile], g, c.row, c.col, cfg)
            # filled-depression proxy: no fall along the +/-10 m trace (epsilon fill)
            zt = tiles[c.tile].z
            tr_rc = tr.trace_rc
            trace_drop = float(zt[tr_rc[0][0], tr_rc[0][1]] - zt[tr_rc[-1][0], tr_rc[-1][1]])
            if tr.trace_length_m > 0 and trace_drop / tr.trace_length_m < cfg["flat_fill_slope_max"]:
                log(c, "flat_fill_proxy", trace_slope=trace_drop / tr.trace_length_m, local_slope=s0)
                continue
            i_t = find_thalweg(tr.s, tr.z, cfg["thalweg_search_m"])
            ok, rl, rr = confinement(tr.s, tr.z, i_t, cfg["confinement_half_window_m"], cfg["confinement_min_relief_m"])
            if not ok:
                log(c, "no_incision", relief_left_m=rl, relief_right_m=rr, local_slope=s0)
                continue
            c.local_slope = s0
            c.slope_method = method
            c.reach_median_slope = smed
            c.transect = tr
            c.extra["n_reach_slopes"] = nreach
            c.extra["trace_slope"] = trace_drop / tr.trace_length_m
            accepted.append(c)
            got += 1
        if got < quotas[bi]:
            excluded.append(dict(site=site, tile=-1, row=-1, col=-1, x=np.nan, y=np.nan,
                                 area_m2=np.nan, bin_index=bi, reason="bin_exhausted",
                                 note=f"wanted {quotas[bi]}, got {got}, tested {cursors[bi]}"))
    return accepted, excluded, edges
