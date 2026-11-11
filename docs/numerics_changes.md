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

## N6. Joint stage: channel network integrated once (Phase 3, step 4)

- **Old:** `optimize_threshold_full` rebuilt the partitioned model for every candidate K, i.e.
  re-ran the channel BFS with `S0 = (U/K)/A^m` and accumulated `z = z_down + S0 dL` term by term.
- **New:** `precompute_chi_network` integrates `chi = sum A^-m dL` and the anchor elevation
  once; `Z_channel(K) = z_anchor + (U/K) chi` (`partitioned_threshold_model_chi`).
- **Where results differ:** everywhere at the last bit only. Scaling the sum once instead of
  each term changes the rounding: on a synthetic tile the channel elevations differ by
  <= 9e-16 relative and the surfaces by <= 1.2e-14 of relief; the Phase 3 gate tolerance is
  1e-8 of relief. The channel mask is unaffected. `rebuild_per_K=True` restores the v1 path.

## N7. AD advection stencil: vectorised power (Phase 3, step 3; found by addendum 3)

- **Old:** v1 built the advection coefficient per cell as ``-(A[j, i] ** m) / dL`` with a numpy
  scalar power. **New:** ``advection_diffusion_model`` computes ``A[rev_j, rev_i] ** m`` for all
  cells at once (the ufunc loop).
- **Where results differ:** numpy dispatches the array power to a SIMD implementation on CPUs
  that have one, and its rounding differs from the scalar power in the last bit for some
  arguments. `serc` mixes AMD Rome nodes (EPYC 7502, `CPU_GEN:RME`, no AVX-512; the login
  nodes are the same CPU) with AMD Genoa nodes (EPYC 8224P, `CPU_GEN:SIE`, AVX-512). On a
  Genoa node 103 of 8126 matrix entries of the 40 x 40 synthetic tile differ by one ulp
  (3.4e-16 relative) and the AD solution by up to 3.5e-16 relative (test job 51704806, node
  sh04-07n24); on a Rome node and on the login node the two agree exactly. The Phase 3 report
  could describe step 3 as bit-identical because its tests were not run in a Slurm job after
  step 3 landed (the test jobs of 2026-11-09 ran at 12:29, 12:42, 12:55 and 13:21 PST; step 3
  was committed at 13:26). The same CPU dependence puts a floor of about 1e-13 of relief
  under any comparison of full-tile grids computed on different node generations (the
  GM2 Picard-audit grid against the step-5 gate grid: 5e-14 of relief; Rome against Genoa). The Phase 3 forward gate
  (1e-8 of relief) and the optimizer pass rule are unaffected; the AD stage trajectories of
  Phase 3 against the section 3 baseline are compared row by row in `03b_phase3_close.md`.
  The tests `TestPhase3AdvectionDiffusionAssembly` now require equality to 1e-14 relative
  (solution) and one ulp (entries) with at least 98 % of the entries bit-identical.

  **Decision (addendum 4, 2026-11-10, section 2.1): N7 is not reverted.** The scalar power would
  restore bit identity with v1 on Genoa nodes but not reproducibility: the L-BFGS-B path of the v1
  AD stage depends on finite-difference gradients at the 1e-13 level of the misfit and therefore
  on the CPU generation the job lands on (`03b_phase3_close.md`, section 2). The AD result is
  decided by the v2 profile over r = K/D (`04_optimizer_v2.md`), which has no finite-difference
  gradient; on AP5 the v1 point is its global optimum and on GM1 it is not.

## N8. Partitioned D-infinity model rebuilt as one coupled system (Phase 4b, addendum 3 section 6)

- **Old** (`lemcvt.models.dinf_models.partitioned_threshold_model_dinf_v1`, kept): channel elevations
  integrated upstream from the boundary for every cell with the explicit two-receiver update
  ``z = z1 + (S0 dx - (z1 - z2) sin r) / cos r`` and the cell's own ``S0 = (U/K) A_dinf^-m``, i.e.
  through hillslope cells and across flow paths (`02d` 2.6: AP1 channel RMS error 2452 ft against
  16 ft with D8).
- **New** (`lemcvt.models.dinf_partitioned.partitioned_threshold_model_dinf`): one sparse system over
  all valid cells; boundary rows Dirichlet, channel rows (``A_dinf >= A_crit`` or the outlet) the
  D-infinity upwind advection stencil of the AD D-infinity model with D = 0, divided by
  ``K A_dinf^m``, hillslope rows the partitioned model's Laplacian with K = 0. ``Z = a + (U/K) b +
  (U/D) c`` exactly (the matrix is K- and D-free), so the Phase 4 reduction and optimizer apply.
- **Where results differ:** everywhere for the partitioned D-infinity model, which is the point of
  the rebuild; nowhere else (the AD D-infinity model and the D8 models are untouched; the
  nonlinear D-infinity model still uses the v1 integration and is not used for any reported
  number). When every facet angle is cardinal or diagonal the new model equals the D8
  partitioned model to 1e-10 of relief (`tests/test_dinf_rebuild.py`, aligned V-valley and
  plane). The `d8_vs_dinf` comparison was rerun with it (checkpoint report of 2026-11-10).

## N9. AP1 seam: in-tile correction implemented, found invalid for AP1, not applied (addendum 5, section 2.1)

- **What was asked:** replace the 345 stored AP1 `*_fd` codes on rows 389-390 that are not the steepest descent
  (checkpoint 4b/5, 4.2) and move each cell's stored area from its old downstream path to its new one inside the tile.
- **What was built:** `lemcvt seamfix` (`lemcvt.analyses.seamfix`). The path update is exact for grids whose area
  was accumulated through their directions by TopoAnalysis's `Area` (elevation order, N1), including cells that
  point uphill, whose area the receiver never passes on: on a synthetic mosaic with a planted two-row seam the
  in-tile correction reproduces the clipped full recomputation bit for bit (`tests/test_seamfix.py`), and an
  independent recomputation from the external inflow agrees with it on AP1 to 2e-8 cells.
- **Why it is not applied:** the premise fails on AP1. The stored area is the accumulation of its in-tile donors
  under the stored directions at every interior cell except 699 on rows 389-391 (133, 565 and 1), where the two
  disagree by up to 5.9e4 cells; it is not consistent with the corrected directions either (959 cells on rows
  388-391); and the stored filled elevation drops by 0.17 m from row 389 to 390 against 0.09-0.10 m between the
  neighbouring rows, by an amount that varies column by column (5-95 %: -0.23 to +0.36 m), so no constant offset
  recovers the stored codes. The three AP1 grids come from mutually inconsistent processing on those rows, and
  the directive's correction, which trusts the stored area, yields 397 cells below one cell (minimum -3.7e4
  cells). The grids `*_seamfix.tif` written by the first run (`20261111T003616Z_seamfix_AP1`) are invalid and are
  **not** in the manifest; the AP1 config keeps the stored grids. No model or measurement result changes.
- **Where results could differ later:** only if AP1's grids are regenerated from the mosaic (not on Sherlock) or
  the tile is re-extracted; the Phase 6 report of 2026-11-10 lists the options.

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
