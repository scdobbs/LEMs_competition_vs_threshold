# Logged numerics changes

Phase 1 ported v1 with its numerics unchanged. Every later change that can alter a
floating-point result, however small, is recorded here with the reason, the rule that
replaces the old behaviour, where results can differ, and the run that measured the
difference. Changes are numbered N1, N2, ...; code comments refer to these numbers.

## Why v1 behaviour cannot be preserved for tie order

v1 (TopoAnalysis and `src/dinf.py`) ordered cells with `numpy.argsort` on
floating-point keys (filled elevation, drainage area) and relied on that order where
several cells share the same key. NumPy's default sort (`kind="quicksort"`, an
introsort) is not stable, and the order it gives to equal keys changed between numpy
1.26 (v1) and numpy 2.x (v2). The Phase 2 resolution sweep showed the effect: 2 of the
500 (tile, dx) ratio rows (AP2 at dx = 88.27 m and AP5 at dx = 60.70 m; both the D8
and the D-infinity ratio of each row, so 4 of 500 ratios) differed from the v1 printed
values by 3-4 %, and the same v2 code run in the `lem-v1` environment (numpy 1.26)
reproduced the printed values exactly (`reports/v2/02_regression.md`, 2.3).

The v1 tie order is therefore a property of a numpy version, not of the algorithm, and
no explicit rule can reproduce it in general: with numpy 2.x the v1 order is simply not
available, and even in numpy 1.26 it is an implementation detail that is not
documented. The changes below replace every tie-sensitive sort with an explicit,
stable rule. Where no tie exists the results are bit-identical to Phase 1.

## N1. D8 drainage-area accumulation order (`lemcvt.hydro.d8.d8_area`)

- **Old:** `Z.argsort(axis=None)[::-1]` on the filled elevation (unstable; ties in
  implementation-defined order).
- **New:** `np.lexsort((flat_index, -Z))`: elevation descending, equal elevations in
  ascending flat (row-major) index.
- **Where results can differ:** only when two cells of exactly equal filled elevation
  are in a donor-receiver relation. The priority-flood fill adds an aggradation slope of
  `1e-12 * dx * distance`, so filled flats are not exactly flat; exact ties between a
  donor and its receiver occur only when the aggradation increment is lost to rounding
  (large elevations, large dx). On the 1 m tiles (Phase 2.1 inputs are the stored v1 area
  grids, so the models are unaffected) and on the resolution sweep see the gate below.

## N2. D-infinity drainage-area accumulation order (`lemcvt.hydro.dinf.compute_dinf_area`)

- **Old:** `np.argsort(-elevations)` (unstable).
- **New:** `np.lexsort((index, -elevations))`: elevation descending, ties in ascending
  row-major index.
- **Where results can differ:** as N1, for cells of equal elevation that route into each
  other.

## N3. Main-stem (largest tributary) selection in the smoothed channel slope (`lemcvt.hydro.channel_slope.flow_graph_main_stem`)

- **Old:** cells visited in `area.argsort(axis=None)` order; each cell overwrites its
  receiver's upstream pointer, so among tributaries of equal area the one visited last
  wins, in implementation-defined order. **This is the sort behind the two drift rows.**
- **New:** `np.lexsort((flat_index, area))`: area ascending, equal areas in ascending
  flat index; among equal-area tributaries the one with the highest flat index wins.
- **Where results can differ:** wherever a cell has two or more donors with exactly equal
  drainage area, which is common on coarse resampled grids (areas are integer multiples
  of `dx**2`) and rare but possible at 1 m. Affects Ks_obs and Ks_mod of the resolution
  sweep, the slope-area figure and the binned slope-area cross-check in `measure`.

## N4. Basin labelling order for hilltop pixels (`lemcvt.measure.hilltop_curvature.basin_labels`)

- **Old:** `np.argsort(area, axis=None)[::-1]`.
- **New:** `np.lexsort((flat_index, -area))`.
- **Where results can differ:** a donor and its receiver can only have equal area if the
  D8 graph has a cycle (TopoAnalysis edge quirk), so no change is expected on the tiles.

## N5. Weighted median in the hilltop-curvature bootstrap (`_weighted_median`)

- **Old:** `np.argsort(v)`. **New:** `np.argsort(v, kind="stable")`.
- **Where results can differ:** nowhere; equal values are interchangeable in a weighted
  median. Changed only so that every sort in the package has a stated tie rule.

## Sorts examined and left unchanged

- `lemcvt.hydro.fill.priority_flood_fill`: the heap holds `(z, counter, row, col)`;
  ties are already broken by insertion order (a port of TopoAnalysis).
- `lemcvt.models.common.precompute_channel_network` (D8 BFS): the queue order cannot
  change any value because every cell has exactly one receiver in D8.
- `lemcvt.measure.chi.channel_network`: BFS from outlets over a tree; order-independent.
- `lemcvt.measure.concavity._disorder`: already `kind="stable"`.
- `np.nanargmax` (outlet choice) returns the first maximum: deterministic.
- `np.unique`, `sorted(...)` on values or file names: no tie sensitivity.

## Gate: resolution sweep rerun after N1-N5

Addendum 2 (2.5) requires the sweep to be rerun and every changed row reported. Rows
are expected to change only where ties exist. Results: see
`reports/v2/02d_phase2_closeout.md`, section 2.5 (filled in from the rerun).
