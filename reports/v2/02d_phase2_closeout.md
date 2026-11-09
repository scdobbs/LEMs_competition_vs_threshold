# Phase 2 close-out: what the Phase 2 report asserted, tested (addendum 2, section 2)

Directive: `reports/directives/refactor_v2_addendum_2.md`, sections 0, 2, 4, 5 (sections 3, 6, 7 follow).
Branch `v2`; code at the commits named in each run's `run.json`. Environment `lemcvt`
(Python 3.12, numpy 2.5.4, scipy 1.18.1). All jobs on `serc`. Date: 2026-11-09.
Machine-generated tables: `02_regression_tables.md` (sections 2.1b, 2.1c, 2.3a), the CSV files
`02d_linearized_recovery.csv`, `02d_sweep_tiebreak_gate.csv`, `02d_sweep_tiebreak_changed_rows.csv`.

Every number below is tied to a run id (`$GROUP_HOME/lemcvt_results/<run id>`) or a Slurm job id.

## Summary

| Item | Result |
| --- | --- |
| 0. State check | `v2` head at start 840203e; tests 149/149 pass on serc (job 51693537); all 43 run directories cited by the Phase 2 reports exist |
| 5. Push check | **Blocked.** `v2` tracks 10 figure PDFs under `reports/v2/figures_v2/` (forbidden by the "any `*.pdf`" rule). No file > 50 MB, no secret on any branch. Nothing pushed; decision needed (section 5) |
| 2.1 Rounding explanation | **Demonstrated.** No full-precision values exist; the linearized recovery passes on all 20 AD-opt and joint runs: every recovered offset lies inside the printed-precision box (largest at 94 % of the box), residual max |.| <= 3.2e-10 x relief against a 1e-8 tolerance, and the predicted MSE differences equal the measured ones to 3 figures |
| 2.2 GM5 joint | **Explained, differently.** The mask cannot change (0 cells in the rounding interval on every tile and run, GM2 acrit aside, where it did not change either). GM5's 6e-6 comes from a misfit gradient dMSE/dlog10 K = -76 at the printed joint point (other tiles: |grad| <= 2.7) times a D rounding offset at 94 % of its box; the v1 joint Nelder-Mead stopped on GM5 far from a stationary point in K. Phase 3 may start |
| 2.3 `n_diff_cells` replaced | by counts at 1e-9 and 1e-6 x relief and the rounding-interval count (code + tables) |
| 2.4 Bookkeeping | 4 of 500 ratios drifted (2 rows x 2 routings); the duplicate dx = 50 row is v1's own (two targets round to the same 4 x 4 grid) |
| 2.5 Tie-breaks | 5 sorts made deterministic (`docs/numerics_changes.md`, N1-N5); gate rerun: 84 rows identical, 163 rows change <= 1e-13 relative, 3 rows change where ties exist |
| 2.6 D-inf partitioned model | **Diagnosed; fix proposed, not applied.** The D-inf channel integration builds channel elevations through hillslope cells and mixes different flow paths: on AP1 5948 of 6459 D-inf channel cells are integrated through cells below A_crit, 3928 are not higher than the receiver they were integrated from, channel elevations reach 5775 ft on a tile whose top is 1441 ft; only 847 channel cells connect to the boundary through channel cells. A synthetic V-valley reproduces it (identical to D8 when every angle is cardinal or diagonal; off by one relief when rotated 20 degrees). The AD D-inf model is unaffected because it solves a linear system instead of integrating |
| Found on the way | `data/manifest.csv` was a 3-entry stub (fixed, 270 entries, job 51695557); the Phase 2 `regression-report` could only show one stage per tile (fixed) |

## 0. Orientation and state check

- Read in order: the directive, addendum 1, addendum 2, `00_reference.md`, `02_regression.md`,
  `02_regression_tables.md`, `02b_measurement_first_results.md`, `docs/measurement_methods.md`,
  `docs/v1_to_v2.md`.
- `v2` head at the start of this session: 840203e ("Phase 2 report: d8_vs_dinf and GM5 joint
  wording"), 11 commits ahead of `origin/main`.
- Test suite on a compute node (serc, job 51693537, 2 CPUs): 149 passed in 50 s at 840203e.
- Run directories: all 43 directories under `$GROUP_HOME/lemcvt_results` named in the Phase 2
  reports exist (10 `tile_forward`, 10 `sweep_resolution`, 20 `measure`, `slope_area`,
  `d8_vs_dinf`, `synthetic_pe`).
- Disagreement between addendum 2 and the repo: none found. One disagreement between the repo
  and its own report: `data/manifest.csv` (Phase 1 requirement 1) listed three GM1 input grids
  of 86 KB, 16 KB and 2.6 KB with hashes that match nothing on disk, i.e. a manifest built from
  a cropped test tile. The 30 input grids on disk match the tile configs' sha256 values. The
  manifest was rebuilt from `$LEMCVT_DATA` in job 51695557 (270 entries: 30 inputs, 240 v1
  outputs) and committed (df4608b).
- Storage (`sh_quota`, 2026-11-09 12:20): `$GROUP_HOME` 260.2 GB of 1 TB (25 %); `$HOME` 10.9 of
  15 GB; `$SCRATCH` 804 GB; `$OAK` 16.5 of 20 TB.

## 5. Push check (done first, as section 8 requires)

Checked every commit not on `origin` for each local-only branch (`fix/ap-vertical-units` does
not exist). Method: `git ls-tree -r` of every commit's tree for the forbidden names;
`git cat-file --batch-check` over every object introduced for sizes; grep of the added lines of
`git diff origin/main..<branch>` for `ghp_`, `github_pat_`, `gho_`, `BEGIN .* PRIVATE KEY`,
`password=`, `token=`.

| Branch | Commits not on origin | Forbidden files | > 50 MB | Secrets |
| --- | --- | --- | --- | --- |
| `feature/channel-width` | 18 | none | none | none |
| `feature/acrit-slope` | 24 | none | none | none |
| `feature/head-extraction` | 35 | none | none | none |
| `v2` | 11 (16 by the end of this session) | **10 `*.pdf`** (below) | none | none |

The ten PDFs are the regenerated figures the Phase 2 report cites, committed in beacffa and
823f4cc: `reports/v2/figures_v2/{d8_vs_dinf_AP1_area, d8_vs_dinf_AP1_elevation,
d8_vs_dinf_AP1_slope_area, d8_vs_dinf_GM1_area, d8_vs_dinf_GM1_elevation,
d8_vs_dinf_GM1_slope_area, fig_slope_area, resolution_study_dinf, supp_fig5_synthetic_pe,
supp_fig6_resolution_ratio}.pdf` (PNG versions sit beside them). They are not papers, but the
rule says any `*.pdf`, so I stopped. **Nothing was pushed.** Options for Steve: (a) allow these
figure PDFs and push as is; (b) remove them from the tip with a new commit and push (the PDFs
remain in the history of the two commits); (c) rewrite the two commits to drop the PDFs, which
the addendum says not to do without asking. The three feature branches are clean and could be
pushed now if Steve wants that done separately.

## 2.1 The rounding explanation, tested

**Full-precision values.** None exist. The v1 notebooks print the optimum with
`f"{x:.4e}"` (5 significant figures) and nothing else: no pickle, no `.npy`, no `res.x`
repr, and the papermill `.out` logs in `legacy/v1_run_logs/` contain no cell output. The
notebook code that saves the grid (GM5 cells 6 and 28, identical in all ten notebooks) runs
the forward model at the **unrounded** `K_star, D_star, A_crit_star` returned by
`scipy.optimize.minimize` and saves that grid. So the saved grid is at full precision and the
printed values are rounded. The "different simplex vertex" hypothesis of addendum 2 (2.2) is
ruled out by this code: `res.x` is the best vertex and `res.fun` its misfit, and the grid is
evaluated at exactly `res.x`.

**Linearized recovery** (`lemcvt closeout linearized`, job array 51694661, one task per tile,
`src/lemcvt/analyses/phase2_closeout.py`). For each AD-opt and joint run: J = dZ/d(log10 K,
log10 D) by central differences with a relative parameter step of 1e-6 (channel mask fixed at
the printed A_crit), delta = argmin ||Z_saved - Z_v2 - J delta||_2 over all valid cells, the
rounding box = half a unit in the 5th significant figure of the printed value in log10 space,
and the residual tolerance 1e-8 x relief.

Results (job array 51694661, runs `20261109T2043..Z_closeout_<tile>`; full table 2.1b in `02_regression_tables.md`, CSV `02d_linearized_recovery.csv`):

| tile | run | delta log10 K | box (+-) | delta log10 D | box (+-) | max dz before / relief | max resid / relief | dMSE measured | dMSE predicted | dMSE/MSE | grad K | grad D | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | AD-opt | 3.5e-07 | 2.6e-06 | -8.0e-07 | 4.0e-06 | 1.9e-06 | 1.9e-12 | -6.21e-08 | -6.20e-08 | -1.7e-10 | -0.32 | 0.05 | yes |
| AP1 | joint | 4.0e-06 | 5.3e-06 | -2.4e-06 | 4.0e-06 | 7.1e-06 | 6.0e-12 | -6.05e-07 | -6.09e-07 | -2.7e-09 | -0.32 | -0.08 | yes |
| AP2 | AD-opt | 2.0e-07 | 3.1e-06 | 1.7e-06 | 3.3e-06 | 4.7e-07 | 8.6e-13 | -3.6e-09 | -4.1e-09 | -9.4e-12 | 0.04 | -0.01 | yes |
| AP2 | joint | 2.2e-06 | 6.2e-06 | -1.7e-06 | 3.1e-06 | 1.8e-06 | 1.7e-12 | 7.69e-07 | 7.71e-07 | 3.3e-09 | 0.31 | -0.03 | yes |
| AP3 | AD-opt | -2.8e-06 | 1.9e-05 | 9.2e-08 | 3.3e-06 | 6.4e-06 | 1.9e-11 | -2.14e-06 | -2.15e-06 | -1.7e-09 | 1.30 | -0.39 | yes |
| AP3 | joint | 2.1e-06 | 3.4e-06 | 1.5e-06 | 3.1e-06 | 2.2e-06 | 9.8e-13 | 2.15e-06 | 2.16e-06 | 4.3e-09 | -1.00 | 2.72 | yes |
| AP4 | AD-opt | 1.3e-06 | 1.3e-05 | 2.4e-06 | 1.0e-05 | 2.0e-06 | 1.2e-12 | -9.54e-08 | -9.47e-08 | -1.7e-10 | -0.31 | 0.08 | yes |
| AP4 | joint | -4.6e-07 | 1.1e-05 | 2.9e-06 | 3.6e-06 | 2.2e-06 | 7.6e-12 | 2.27e-07 | 2.35e-07 | 6.1e-10 | -0.52 | -0.08 | yes |
| AP5 | AD-opt | 8.9e-06 | 1.4e-05 | 7.2e-07 | 5.0e-06 | 3.1e-05 | 2.1e-10 | -1.63e-05 | -1.63e-05 | -1.6e-08 | -3.74 | 0.89 | yes |
| AP5 | joint | -9.8e-07 | 6.3e-06 | -2.1e-06 | 6.0e-06 | 3.2e-06 | 3.6e-12 | -2.25e-06 | -2.24e-06 | -7.8e-09 | 0.71 | 0.81 | yes |
| GM1 | AD-opt | -4.5e-06 | 4.9e-06 | 5.8e-06 | 7.5e-06 | 2.0e-05 | 1.6e-10 | -1.36e-06 | -1.36e-06 | -8.1e-09 | 0.42 | -0.15 | yes |
| GM1 | joint | -4.5e-06 | 1.0e-05 | 6.6e-06 | 7.7e-06 | 7.6e-06 | 4.1e-12 | 4.22e-07 | 4.23e-07 | 1.2e-08 | -0.04 | 0.03 | yes |
| GM2 | AD-opt | 8.9e-06 | 1.7e-05 | -8.8e-07 | 2.6e-06 | 2.6e-05 | 1.4e-10 | -1.62e-06 | -1.61e-06 | -2.0e-08 | -0.36 | 0.07 | yes |
| GM2 | joint | 9.8e-07 | 3.9e-06 | -1.4e-06 | 8.2e-06 | 1.7e-06 | 7.0e-13 | -4.19e-08 | -4.19e-08 | -1.5e-09 | 0.16 | 0.15 | yes |
| GM3 | AD-opt | -5.6e-06 | 1.2e-05 | 3.2e-06 | 9.3e-06 | 2.3e-05 | 8.5e-11 | -8.08e-07 | -8.07e-07 | -1.1e-08 | 0.25 | -0.07 | yes |
| GM3 | joint | -9.0e-06 | 1.1e-05 | -3.5e-06 | 4.8e-06 | 1.7e-05 | 1.7e-12 | -7.66e-08 | -7.65e-08 | -1.8e-09 | 0.04 | 0.03 | yes |
| GM4 | AD-opt | 3.0e-07 | 2.3e-06 | -1.7e-06 | 3.2e-06 | 2.5e-06 | 3.1e-12 | -1.70e-08 | -1.69e-08 | -9.7e-11 | -0.05 | 0.01 | yes |
| GM4 | joint | 2.4e-06 | 6.2e-06 | 7.0e-06 | 8.2e-06 | 9.2e-06 | 2.0e-12 | 7.61e-07 | 7.60e-07 | 1.6e-08 | 0.17 | -0.00 | yes |
| GM5 | AD-opt | 1.0e-05 | 1.4e-05 | -4.1e-06 | 1.2e-05 | 3.4e-05 | 3.2e-10 | -4.59e-06 | -4.58e-06 | -4.9e-08 | -0.81 | 0.21 | yes |
| GM5 | joint | 2.4e-06 | 3.7e-06 | **1.0e-05** | 1.07e-05 | 9.1e-06 | 7.7e-12 | **-1.99e-04** | **-1.99e-04** | **-6.0e-06** | **-76.3** | -1.73 | yes |

Columns: delta = recovered offset of the saved grid's parameter from the printed one (log10);
box = half a unit in the 5th significant figure in log10 (the box is symmetric to the digits
shown); "max dz before" is the Phase 2 difference, "max resid" what remains after removing
J delta; dMSE = MSE(saved grid) - MSE(v2 at printed parameters) in the tile's z unit squared;
grad = dMSE/dlog10 K and dlog10 D at the printed parameters (first-order term g . delta plus
the quadratic term mean((J delta)^2) gives "predicted").

Reading. Every |delta| is inside its box, so the saved grids are consistent with the printed
parameters being 5-significant-figure roundings of the parameters actually used, and with
nothing else: after removing the two-parameter linear correction the residual is 1e-12 to 3e-10
of relief, the same level as the exact-parameter runs of Phase 2 (1e-12). The MSE differences
are reproduced to three figures by the linear model, including their sign. This closes the
Phase 2 assertion for AD-opt and joint on all ten tiles.

A by-product worth recording: the MSE gradients at the printed optima are not zero. For AD-opt
(L-BFGS-B) they are 0.04-3.7 MSE units per decade of K; L-BFGS-B stops on the relative
function decrease (`ftol` = 2.2e-9), not on the gradient, so this is expected. For the joint
stage (Nelder-Mead, `xatol` = `fatol` = 1e-4) they are 0.04-2.7 on nine tiles and **-76 on GM5**.


## 2.2 GM5 joint

The Phase 2 report attributed GM5's joint relative MSE difference of 6e-6 (400 x larger than
the next tile) to the rounded A_crit "moving a few cells across the channel threshold". That
is wrong, as addendum 2 says: the count below is 0, and `n_diff_cells` = 996004 said nothing.
The linearized recovery gives the actual mechanism:

- delta log10 D = 1.007e-5 against a box of 1.0705e-5: the D the optimizer found
  (0.0202845 within 1e-7) sits at the edge of the interval that prints as 0.020284. The K
  offset is 2.4e-6 (box 3.7e-6).
- dMSE/dlog10 K = -76.3 at the printed point (dMSE/dlog10 D = -1.7). On the other nine tiles
  the joint gradients are between 0.04 and 2.7. A gradient of -76 means the misfit falls by
  0.076 MSE units (0.23 %) per 0.001 decade of K: the v1 joint Nelder-Mead stopped on GM5 far
  from a stationary point in K.
- The first-order prediction g . delta = -1.995e-4 reproduces the measured dMSE = -1.992e-4;
  the quadratic term is 3e-7. Relative to MSE = 33.054 that is -6.03e-6, the Phase 2 number.
- The residual after the correction is 7.7e-12 x relief: the saved grid is the v2 model at
  (K, D) inside the rounding box with the same channel mask.

Why a different simplex vertex is not the explanation: the notebook (cell 28) evaluates and
saves the grid at `res.x`, the best vertex; and a different vertex would differ in log10 A_crit,
D or K by order `xatol` = 1e-4, 10 x the whole rounding box, which the recovery would have shown
as an out-of-box delta or a large residual.

Consequence for the v1 numbers: GM5's joint optimum is not a stationary point. Nelder-Mead on
the staircase objective (the misfit is piecewise constant in A_crit) can collapse its simplex
and stop while the K direction still has a strong gradient; GM5 is the only tile where that
happened measurably. This belongs to the Phase 4 comparison (v1 against v2 optimizer) and to
the Phase 2.2 optimizer regression, where GM1 and AP5 are the tiles run, not GM5. It is
reported here and not fixed.


**Rounding-interval counts** (addendum 2, 2.2.1; `rounding_interval_counts` in each closeout
run; table 2.1c of `02_regression_tables.md`). Cells whose D8 area lies inside the rounding
interval [A_printed - h, A_printed + h) of the printed A_crit, h = half a unit in the 5th
significant figure:

| tile | A_crit stage 2 (printed) | cells in interval | A_crit nonlinear | cells | A_crit joint | cells | channel cells (joint) | nearest areas around joint A_crit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 3063.4 | 0 | 4950.3 | 0 | 6414.3 | 0 | 5003 | 6379.01 / 6427.49 |
| AP2 | 2337.8 | 0 | 6165.9 | 0 | 7645.6 | 0 | 6178 | 7645.3 / 7646.25 |
| AP3 | 1722.1 | 0 | 10233 | 0 | 9741.5 | 0 | 4300 | 9739.63 / 9742.48 |
| AP4 | 995.01 | 0 | 5263.5 | 0 | 7797.9 | 0 | 4055 | 7797.41 / 7799.31 |
| AP5 | 1130 | 0 | 3399.1 | 0 | 1887 | 0 | 13046 | 1886.13 / 1887.08 |
| GM1 | 423.49 | 0 | 474.21 | 0 | 8092.4 | 0 | 4953 | 8082 / 8132 |
| GM2 | 100 | **1028** | 244.59 | 0 | 2960 | 0 | 6782 | 2957 / 2965 |
| GM3 | 1297.4 | 0 | 1297.4 | 0 | 6526.3 | 0 | 5260 | 6522 / 6532 |
| GM4 | 422.09 | 0 | 521.38 | 0 | 7396.2 | 0 | 5152 | 7395 / 7400 |
| GM5 | 261.42 | 0 | 299.62 | 0 | 2247.1 | 0 | 9939 | 2247 / 2248 |

(Channel counts and nearest areas: table 2.1c of `02_regression_tables.md`.) The one nonzero
count is GM2's stage-2 A_crit = 100.00, the lower bound of the search: the interval
[99.995, 100.005) contains the integer 100 and 1028 cells have exactly that area. The Phase 2
comparison of that run was exact (dz 2e-13 x relief), so those cells were channel cells in both
v1 and v2 (`A >= 100` with A_crit = 10**log10(100) on both sides); the count says their
membership *could* have changed under rounding, not that it did.


On GM the areas are integer multiples of 1 m^2 and every interval has width 0.1 or 0.01 m^2
without an integer inside, so the count is 0 by construction. On AP the cell area is 0.9507 m^2
and the count is also 0 on every tile: no multiple of 0.9507 m^2 falls inside any of the ten
intervals. The channel mask of every A_crit-bearing run is therefore identical to the one the
saved grid used, and the Phase 2 report's "moving a few cells across the channel threshold"
is wrong for GM5 and for every other tile.

## 2.3 `n_diff_cells` replaced

`compare_with_saved` (`src/lemcvt/analyses/forward.py`) now records, per run, the number of
cells with |dz| > 1e-9 x relief, with |dz| > 1e-6 x relief, and `n_mask_may_differ`, the
rounding-interval count above (v1 saved no mask, so the cells whose membership can differ are
exactly those). `lemcvt regression-report` prints them in table 2.1; for the Phase 2 forward
runs, which predate the change, the closeout run computed them from the same grids
(`difference_counts`; column `counts_from` names the source run).

For the Phase 2 forward runs (table 2.1 of `02_regression_tables.md`, columns
`n_dz_gt_1e-9_relief`, `n_dz_gt_1e-6_relief`, `n_mask_may_differ`; GM 996004 interior cells,
AP 1026169 (AP3 1025155)):

| run key | cells > 1e-9 x relief | cells > 1e-6 x relief | mask may differ |
| --- | --- | --- | --- |
| AD-no-opt (exact parameters) | 0 on all ten tiles | 0 | - |
| A_crit-opt (exact K0, D0; printed A_crit) | 0 on all ten tiles | 0 | 0 (GM2: 1028, see 2.2) |
| AD-opt (printed K, D) | 995593-1026169 (all) | 0 (AP2) to 997973 (AP5) | - |
| joint (printed K, D, A_crit) | 992321-1026163 (all) | 10041 (GM2) to 912363 (GM3) | 0 on all ten tiles |

The exact-parameter runs have **no** cell above 1e-9 x relief (about 1e-7 m), which is a
stronger statement than the Phase 2 max |dz| of 1e-12 x relief. The rounded-parameter runs
differ everywhere at 1e-9 and on 1 %-97 % of the cells at 1e-6 x relief, in proportion to the
rounding offsets of table 2.1b (AP2 AD-opt, whose offsets are the smallest, has 0 cells above
1e-6). The old `n_diff_cells` (996004 on every GM run) carried none of this.


## 2.4 Bookkeeping in the Phase 2 tables

**Drift count.** Table 2.3a's AP2 and AP5 rows each carry two ratios (D8 and D-inf) that share
the same observed steepness Ks_obs; the drift was in Ks_obs, so **4 of the 500 ratios**
(2 rows x 2 routings) differed from the printed values by more than the printed half-unit
(5e-4). `lemcvt regression-report` now prints this count and the drifted rows under 2.3a.

**Duplicate row.** `supp_fig5` builds its synthetic grids from 25 log-spaced target pixel
sizes between 1 and 55 m over a 150 m domain; the grid can only hold an integer number of
cells (`Nx = round(Lx/dx) + 1`), and the two coarsest targets (55.0 and 46.7 m) both give
Nx = Ny = 4, i.e. dx = 50 m. v1 printed that row twice and v2 reproduces the duplicate
faithfully. The generator now writes a `note` on the second row and the docstring of
`_grid_dims` records the reason; the row is kept because it is v1's output.

## 2.5 Deterministic tie-breaking (logged numerics change)

Every sort in the package was examined (`grep` for `argsort`, `sort`, `lexsort`, `unique`,
`searchsorted`, heap use); the ones whose tie order can change a result are listed with their
replacement rule in `docs/numerics_changes.md` (N1-N5), which also states why v1's tie order
(numpy's unstable default sort in numpy 1.26) cannot be preserved:

| Id | Sort | Effect of ties | Rule now |
| --- | --- | --- | --- |
| N1 | D8 area accumulation order (`hydro.d8.d8_area`) | donor/receiver of equal filled elevation | Z descending, ties by ascending flat index |
| N2 | D-inf area accumulation order (`hydro.dinf.compute_dinf_area`) | as N1 | same |
| N3 | main-stem choice in the smoothed channel slope (`hydro.channel_slope`) | equal-area tributaries | area ascending, ties by flat index (highest flat index wins) |
| N4 | basin labelling for hilltop pixels (`measure.hilltop_curvature.basin_labels`) | only through D8 cycles | area descending, ties by flat index |
| N5 | weighted median in the curvature bootstrap | none on the value | stable sort |

Unchanged and why: the priority-flood heap already carries an insertion counter; the D8
channel BFS and the chi BFS are order-independent on a tree; `_disorder` was already stable;
`nanargmax` is deterministic.

**Gate: resolution sweep rerun** (job array 51694660, 10 runs `20261109T2040..Z_sweep_resolution_*`,
code fe729aa/58f0431) against the Phase 2 runs (`20261109T18..Z_sweep_resolution_*`), all 250
(tile, dx) rows (`02d_sweep_tiebreak_gate.csv`):

| Class | Rows | Meaning |
| --- | --- | --- |
| identical | 84 | every column bit-identical |
| last-bit | 163 | max relative change <= 9.6e-14: N1 changed the order in which `dx**2` terms are summed into the area (non-integer dx), nothing else |
| tie-driven | 3 | AP2 dx 88.27 m (5 channel pixels), AP5 dx 60.70 m (15), AP4 dx 100 m (6): N3 picked a different main stem among equal-area tributaries |

The three tie-driven rows (`02d_sweep_tiebreak_changed_rows.csv`):

| tile | dx (m) | n_obs | Ks_obs v1 | Ks_obs Phase 2 | Ks_obs now | ratio_d8 v1 / Phase 2 / now | ratio_dinf v1 / Phase 2 / now |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AP2 | 88.27 | 5 | 147.4 | 142.24 | 164.06 | 1.361 / 1.411 / 1.223 | 1.628 / 1.688 / 1.463 |
| AP5 | 60.70 | 15 | 13.65 | 13.20 | 13.65 | 1.306 / 1.341 / 1.306 | 1.460 / 1.510 / 1.460 |
| AP4 | 100.00 | 6 | 30.37 | 30.37 | 31.80 | 1.024 / 1.024 / 0.969 | 0.995 / 0.995 / 0.947 |

So the explicit rule reproduces v1 on AP5 (where numpy 2.x had drifted) and departs from v1 on
AP4 at 100 m (where numpy 2.x had happened to agree); AP2 at 88 m is a third value. Against the
printed v1 tables the count is unchanged, 4 of 500 ratios on 2 rows. All three rows are the
coarsest, point-poorest cases (5-15 channel pixels in a 10 x 10 to 17 x 17 grid) where the
"main stem" is a toss-up between equal-area tributaries; the median-IQR curves of the figures do
not change visibly. Changes are confined to where ties exist, as predicted, and the GM rows
and every AP row with more than 15 points are unaffected beyond the last bit.

The equivalence test against TopoAnalysis (`tests/test_hydro.py::test_channel_slope`) failed
after N3 on the one synthetic 50 x 50 grid it uses, for the predicted reason; it now checks
bit-equality on every cell whose smoothing path does not pass a receiver with tied donors and
a new test fixes the documented tie rule. Test job 51695841: 150 passed (149 + the new tie-rule test) at 8151c38 on serc.

## 2.6 The D-inf partitioned model: diagnosis

Run `20261109T204305Z_dinf_diagnostics_AP1_GM1` (job 51694667; `lemcvt closeout dinf --tiles AP1,GM1`,
`src/lemcvt/analyses/dinf_diagnostics.py`), at the v1 `d8_vs_dinf` parameters (K0, D0 rounded
to 2 s.f., A_crit = 5000 m^2; AP1 in feet as v1 ran). The partitioned-model MSEs reproduce the
Phase 2 / v1 values (AP1: D8 2.1950e+03, D-inf 3.7650e+06 ft^2; GM1: 7.9204e+02,
4.2395e+03 m^2). Grids written to the run directory: `<tile>_pt_d8_error`, `<tile>_pt_dinf_error`,
`<tile>_Zchannel_dinf`, `<tile>_integration_branch`, `<tile>_masks_dinf1_d8_2`.

### 2.6.1 Where the error sits (Z - Z_obs by cell class; RMS / max |.| / mean, tile z unit)

| tile | model | channel cells | hillslope cells | boundary | SSE share hillslope |
| --- | --- | --- | --- | --- | --- |
| AP1 | partitioned D8 | 15.88 / 62.63 / -3.499 (n 5641) | 47.06 / 122.1 / +36.8 | 0 | 0.999 |
| AP1 | partitioned D-inf | 2452 / 4484 / +2071 (n 5964) | 1941 / 4480 / +1566 | 0 | 0.991 |
| GM1 | partitioned D8 | 4.1 / 19.83 / +0.5639 (n 5292) | 28.27 / 73.6 / +22.16 | 0 | 1.000 |
| GM1 | partitioned D-inf | 65.53 / 239.7 / +31.01 (n 5845) | 65.24 / 234.8 / +47.21 | 0 | 0.994 |

The D-inf error is largest on the **channel cells** themselves: the D-inf channel elevations
are off by 2452 ft RMS on AP1 (D8: 16 ft) and 66 m on GM1 (D8: 4.1 m), with a positive
mean (channels too high by 2071 ft and 31 m). The hillslope cells, 99 % of the tile, carry 99 % of
the SSE in both routings, but with D-inf they inherit the wrong channel elevations as Dirichlet
values (mean hillslope error +1566 ft on AP1 against +37 ft with D8). Boundary cells are exact in
both. No output cell is NaN.

### 2.6.2 The D-inf channel set and its integration

How the model defines them (`_precompute_channel_network_dinf`): every cell has up to two
receivers, the cardinal e1 and the diagonal e2 of its steepest facet, with the facet angle r in
[0, pi/4]. Channel elevations are integrated **for every cell of the grid** upstream from the
boundary by BFS; a cell whose two receivers already have values gets
z = z1 + (S0 dx - (z1 - z2) sin r) / cos r, i.e. z = (1 - tan r) z1 + tan r z2 + S0 dx / cos r,
with S0 = (U/K) A_dinf^-m of the cell itself; a cell reached while only one receiver has a value
gets z1 + S0 dx (cardinal) or z2 + S0 dx sqrt2 (diagonal). "Downstream path", dL and A are
therefore: a weighted average of two receivers' elevations (not a path), dL = dx / cos r along
the facet direction (dx or dx sqrt2 in the fallbacks), and A = the D-inf area of the cell being
integrated. The channel mask is A_dinf >= A_crit. A replay of this loop in the diagnostics
reproduces the model's channel elevations exactly on GM1 (`trace_reproduces_model` true; on AP1
the first replay differed in the last bit through the order of one multiplication, fixed in
job 51697246).

Measured on the channel cells (A_dinf >= 5000 m^2):

| quantity | AP1 | GM1 |
| --- | --- | --- |
| channel cells, D8 / D-inf | 5657 / 6459 | 5318 / 6141 |
| D-inf channel cells connected to the boundary through channel cells (via the receivers used) | 847 | 2896 |
| ... not connected | 5612 | 3245 |
| D8 channel cells whose receiver is a hillslope cell | 4 | 0 |
| D-inf channel cells whose integration path passes a cell with A_dinf < A_crit | 5948 | 5841 |
| D-inf channel cells integrated directly from a receiver of smaller area | 5142 | 5118 |
| branch that set the value: both / cardinal-only / diagonal-only / boundary | 5150 / 225 / 589 / 495 | 5213 / 294 / 338 / 296 |
| channel cells NOT higher than the receiver they were integrated from (monotonicity violated) | 3928 of 5964 | 3918 of 5845 |
| median / min drop to the receiver used (z unit) | -2.05 / -1574 | -0.86 / -45 |
| max channel elevation, D-inf / D8 / observed max | 5775 / 1454 / 1441 | 596 / 400 / 425 |
| channel cells more than one relief above the observed maximum | 4978 | 302 |
| RMS error of the channel elevations against the DEM, D-inf / D8 | 2356 / 15.9 | 64 / 4.1 |
| effective step dL = (z - z_ref)/S0: median / 1 % / min / count negative | 0.975 / -1477 / -9290 / 1690 | 1.000 / -83 / -355 / 1340 |
| S0 at A_crit / S0 at one cell's area (z unit per m) | 1.15 / 141 | 0.265 / 6.68 |
| cells never reached (NaN) | 0 | 0 |

Answers to the four questions of addendum 2 (2.6.2):

- *Is every channel cell connected to the Dirichlet boundary?* Through channel cells, no: only
  847 of 6459 on AP1 and 2896 of 6141 on GM1. The D-inf receivers of a channel cell can have smaller
  D-inf area than the cell itself (flow is split), so A_dinf >= A_crit is not a connected
  network. Through any cells, yes: every cell is reached (0 NaN).
- *Is channel elevation monotone upstream?* No: 3928 (AP1) and 3918 (GM1) of the integrated
  channel cells are not higher than the receiver they were integrated from; the median "drop" is
  negative. The mixing formula subtracts (z1 - z2) sin r / cos r, and when the two receivers lie
  on different flow paths their elevations differ by metres to hundreds of metres, which the
  formula interprets as a slope along the facet.
- *How are the path, dL and A defined?* As above: no path, a two-receiver average; dL = dx/cos r
  nominally, but the effective step (z - z_ref)/S0 ranges from -9290 to 622 m on AP1 (1690 negative);
  A is the integrated cell's own D-inf area, but its receivers' elevations were themselves
  integrated with S0 of **their** (often much smaller) areas: 5948 of 6459 AP1 channel cells
  and 5841 of 6141 GM1 channel cells sit on an integration chain that passed through cells
  below A_crit, where S0 is up to 123 x (AP1) and 25 x (GM1) larger than at A_crit.
  This is what lifts the channel elevations to 5775 ft on a tile whose highest point is 1441 ft.
- *Does any path fail to terminate at the boundary?* No cell is left NaN; every chain ends at
  a boundary cell. Termination is not the problem.

### 2.6.3 A_crit = 5000 against the D-inf area distribution

| | AP1 | GM1 |
| --- | --- | --- |
| channel cells at A_crit = 5000, D8 / D-inf | 5657 / 6459 | 5318 / 6141 |
| D-inf A_crit giving the D8 channel count | 6611 | 7117 |
| area quantiles 50 / 90 / 99 / 99.9 % / max, D8 | 11.4 / 108 / 2.24e+03 / 2.68e+05 / 1.12e+06 | 13 / 100 / 836 / 2.04e+05 / 1.16e+06 |
| same, D-inf | 28.3 / 150 / 1.95e+03 / 1.51e+05 / 7.66e+05 | 27.5 / 94.2 / 1.39e+03 / 1.26e+05 / 2.47e+05 |
| channel cells at 1e3 / 2e3 / 5e3 / 1e4 / 2e4, D8 | 20966 / 11058 / 5657 / 4204 / 3220 | 8751 / 6417 / 5318 / 4750 / 3844 |
| same, D-inf | 15631 / 10164 / 6459 / 4669 / 3293 | 11370 / 8699 / 6141 / 4693 / 3571 |

D-inf spreads area: the median cell area is 2 x the D8 one and the maximum is 0.2-0.7 x. At the
same A_crit the D-inf channel set is 14-15 % larger; matching the D8 count needs A_crit ~ 6600
(AP1) or 7100 m^2 (GM1). This is a modest effect and cannot explain a 10^3 x MSE: at
A_crit = 1e4 (fewer D-inf channel cells than D8 at 5000) the mechanism of 2.6.2 is unchanged.

### 2.6.4 Synthetic cases (same run; `synthetic_*` grids in the run directory)

201 x 201 cells, dx = 1 m, GM-like parameters (m 0.4, U 1.47e-4, K 2.2e-5, D 1e-2,
A_crit 500 m^2), hydrology from a padded copy as in the tests:

| case | D-inf angles not a multiple of 45 deg | max |A_dinf - A_d8| (m^2) | max |Z_pt_dinf - Z_pt_d8| / relief | same for AD | channel cells D8 / D-inf | axis channel vs analytic chi integral |
| --- | --- | --- | --- | --- | --- | --- |
| plane tilted south | 398 (edge rows only) | 0 | 0 | 0 | 1 / 1 | - |
| V-valley, side slope = axial slope (0.1) | 10681 (edge rows and the axis's zero-weight receiver) | 2.75e-10 | 2.84e-15 | 2.64e-13 | 178 / 178 | 0 |
| V-valley rotated 20 deg (axial slope 0.036) | 40198 | 4.4e+03 | **1.08** | 0.183 | 198 / 186 | 0 |

When every facet angle is exactly cardinal or diagonal (plane; V with equal slopes, where the
sides drain at 45 deg and the axis straight south) the two routings give the same areas to
3e-10 m^2 and the same partitioned surfaces to 3e-15 of relief, and the D8 axis profile equals
the analytic chi integral z = z_bc + (U/K) sum A^-m dx exactly. Rotating the V by 20 deg makes
D-inf split flow (40198 non-45-deg angles): the AD surfaces then differ by 0.18 of relief (two
different discretisations of the same PDE at a coarse 201-cell grid), but the partitioned
surfaces differ by **1.08 relief**, with the D-inf channel cells again integrated through
sub-threshold cells and mixed between receivers. The failure is specific to the partitioned
D-inf integration, not to D-inf routing.

### 2.6.5 Ranked hypotheses

1. **The channel integration is not an integration along a path** (confirmed, primary). The
   D-inf partitioned model assigns z = (1 - tan r) z1 + tan r z2 + S0 dx / cos r using the two
   receivers' already-integrated elevations and the cell's own S0, for every cell of the grid.
   Because D-inf area is not monotone along either receiver link, channel cells are built on
   top of hillslope cells whose S0 is 25-120 x larger (2.6.2, "passes a cell with A < A_crit":
   92-95 % of channel cells), and the two receivers often belong to different flow paths, so
   the (z1 - z2) term adds or removes metres per step (negative effective steps on 1340-1690
   channel cells; channel elevations up to 4 relief above the DEM). Evidence: branch trace
   reproducing the model, monotonicity and connectivity counts, channel RMS error 2452 ft /
   64 m against 16 ft / 4 m with D8, the rotated-V synthetic (1.08 relief) against the aligned
   V (3e-15).
2. **The D-inf channel set is not a network** (confirmed, consequence of 1 and of the mask
   definition). With A_dinf >= A_crit, 87 % (AP1) and 53 % (GM1) of channel cells have no
   channel-only route to the boundary, so a threshold model that needs a downstream channel
   path has none to follow. Any fix must either route channels along a tree or solve the
   channel equation as a coupled system.
3. **BFS order dependence** (confirmed, minor). 814 (AP1) and 632 (GM1) channel cells were set
   by a single-receiver fallback because the other receiver had no value yet; their step length
   is dx or dx sqrt2 regardless of r. This adds noise but is not the 10^3 factor.
4. **The threshold itself** (ruled out as the cause). D-inf has 14-15 % more channel cells at
   A_crit = 5000; the equivalent D-inf threshold is 6600-7100 m^2. Changing A_crit cannot repair
   channel elevations that are wrong by a relief.
5. **Unreached cells / NaN propagation** (ruled out). 0 cells unreached, 0 NaN in the output.
6. **The hillslope solver** (ruled out). It is the same code as D8 and the boundary cells are
   exact; the hillslope error is inherited from the channel Dirichlet values.

### 2.6.6 Proposed fix (not implemented; model code unchanged)

Replace the explicit upstream integration in `partitioned_threshold_model_dinf` by a single
sparse linear system over all cells, which is how the AD D-inf model already works and why it
is unaffected: for channel cells (A_dinf >= A_crit) use the D-inf advection row
K A_dinf^m [(z - z1) cos r + (z1 - z2) sin r] / dx = U (the stencil of
`advection_diffusion_model_dinf` with D = 0), for hillslope cells the Laplacian row
D lap z = -U (K = 0), Dirichlet rows on the tile boundary. This (i) needs no downstream path or
connected channel set, because the coupled system propagates information through hillslope
cells correctly as diffusion, not as channel incision; (ii) reduces exactly to the D8
partitioned model when every angle is cardinal or diagonal, because the D8 upwind row
K A^m (z - z_down)/dL = U is the D8 integration step (the aligned-V synthetic is the regression
test, tolerance 1e-10 of relief); (iii) costs one sparse solve per (A_crit, D), as now, and
drops the Python BFS. The linear-in-U/K and U/D structure of the directive's Phase 4 reduction
(Z = a + (U/K) b + (U/D) c) still holds, so the Phase 4 optimizer applies unchanged. A cheaper
but less faithful alternative, routing channel cells along the D8 tree with D-inf areas in S0,
is not recommended: it mixes two routings and keeps the hillslope/channel inconsistency at
split-flow cells. The d8_vs_dinf comparison (Phase 2.3, Phase 6.6) should be rerun with the
fixed model; its v1 printed values (PT D-inf 3.765e6 and 4239.5) are reproductions of a defect.


## 4. Slurm and storage (addendum 2, section 4)

- `configs/slurm/sherlock.yaml`: `partition: serc`, limits from `sh_part` and `sinfo -Nlp serc`
  recorded as comments (max 7 d, 6 GB/core default, 32 GB/core max; 24-256 cores per node).
- Resources from measurement (`sacct`, jobs 51680757 and 51683792): TotalCPU/Elapsed 0.98-1.0,
  so the solves are single-threaded; MaxRSS 32.2-33.9 GB for a full tile. Defaults are now
  2 CPUs and 51 GB (1.5 x MaxRSS); `OMP/MKL/OPENBLAS_NUM_THREADS` are set to the CPU count.
- Every array is throttled (`%10`); the limit without asking is 300 core-hours.
- `run.json` now records `slurm_accounting` (partition, elapsed, CPU time, MaxRSS via `sstat` and
  `scontrol` at finalize) and the job script writes `slurm_sacct.txt` into the run directory.
- Job I/O: results are written directly under `$GROUP_HOME/lemcvt_results` (small: a run
  directory holds a few GeoTIFFs and JSON); logs under `$SCRATCH/lemcvt_jobs`.
- `$GROUP_HOME` usage at this checkpoint: 260 GB of 1 TB.

## What the evidence supports, and where it is weak

**Supported by the evidence.**
- The refactor reproduces v1 at full precision: the only differences from the saved v1 grids
  are 5-significant-figure roundings of the printed parameters, shown by recovering those
  roundings on all 20 runs with residuals at the exact-parameter level (1e-12 to 3e-10 of
  relief). The channel masks are identical on every run (rounding-interval count 0).
- GM5's joint MSE difference is the product of a nonzero misfit gradient (-76 per decade of K)
  and the D rounding offset; the v1 Nelder-Mead stopped there far from a stationary point.
- The partitioned D-inf model's 10^3 x MSE is a construction defect of its channel integration
  (mixing receivers on different paths, integrating through sub-threshold cells), reproduced on
  a synthetic surface, and absent from the AD D-inf model.
- Tie order in v1 was a numpy-version artefact; the explicit rules change 3 of 250 sweep rows,
  all coarse rows with 5-15 channel pixels, and nothing else beyond the last bit.

**Where it is weak.**
- The linearized recovery assumes the saved grid differs only through (K, D) with the mask
  fixed; it cannot detect a compensating pair of errors that happens to look like a rounding
  offset. The residual level (equal to the exact-parameter runs) and the fact that every delta
  is inside its box on all 20 runs make that remote, but it is an inference, not a measurement
  of the unrounded parameters, which do not exist.
- The MSE gradients are first-order quantities at the printed point; the claim that the v1 GM5
  joint optimum is "not stationary" rests on that single gradient and has not been checked by
  re-running the optimizer (that is what the section 3 jobs do, on GM1 and AP5, not GM5).
- The D-inf diagnosis identifies the mechanism but the proposed fix is untested; the aligned-V
  synthetic shows what the fixed model must reproduce, not that the proposal reproduces it.
- The tie-break gate was run on the resolution sweep only; the 1 m tiles use v1's stored area
  grids, so N1 could only show up on recomputed hydrology (Phase 5.2 regenerates D8 grids and
  must compare them with the originals as the directive says).
- Measured versus inferred: everything in the tables is measured from the runs named; the
  mechanism statements in 2.2 (why Nelder-Mead stopped) and 2.6.5-2.6.6 (ranking, fix) are inferences.


## Next

- Section 3 (optimizer regression, GM1 and AP5, one serc job per stage, 140 core-hours):
  submitted after this report (job ids in `reports/v2/03_optimizer_regression_jobs.json`).
- Section 6 (`lemcvt measure-diagnostics --tiles all`): runs while those jobs run; report
  `02c_measurement_diagnostics.md`.
- Section 7 (Phase 3 performance) starts once this close-out is accepted.
- Decisions needed from Steve: the push (section 5 options a-c); whether the D-inf fix of 2.6.6
  is to be implemented in Phase 3 or deferred to Phase 6.6; whether GM5 should be added to the
  optimizer regression given 2.2.

