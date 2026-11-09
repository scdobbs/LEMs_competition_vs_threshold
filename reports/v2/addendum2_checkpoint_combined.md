# Addendum 2 checkpoint: Phase 2 close-out, measurement diagnostics, Phase 3 performance

Combined from `reports/v2/02d_phase2_closeout.md`, `02c_measurement_diagnostics.md` and
`03_performance.md` (branch `v2`, head a5aa74e plus one CSV commit, 2026-11-09), with the
generated tables they cite as appendices. Figures referenced in Part B are
`reports/v2/figures_v2/02c_theta_ladders.png`, `02c_theta_ladders_pooled.png`,
`02c_curvature_scale.png`, `02c_ridge_bias.png`, `02c_scaling_break.png` (attach alongside).
Nothing is pushed; all run directories are under `$GROUP_HOME/lemcvt_results`.

## Decisions needed

1. **Push (addendum 2, section 5).** Blocked by ten tracked figure PDFs on `v2`
   (`reports/v2/figures_v2/*.pdf`); no large files, no secrets, feature branches clean.
   Options: allow the figure PDFs and push; remove them in a new commit; rewrite the two commits.
2. **Threshold rule for theta and k_s** (Part B, section 10): options (a) fixed 1e3 m^2,
   (b) fixed 1e4 m^2, (c) site-specific slope-area break, (d) rollover. Recommendation, kept
   separate from the evidence: (a) with the site-pooled theta.
3. **Curvature method** (Part B, section 7): the pre-registered 10 % rule is not triggered at the
   median crest width but is for narrow crests (28-50 % of AP hilltop pixels); no new C_ht or D0
   values were produced pending the decision.
4. **D-inf partitioned model** (Part A, section 2.6): construction defect diagnosed, fix proposed
   (coupled sparse system), not applied; decide whether it goes into Phase 3 or Phase 6.
5. **GM5 in the optimizer regression** (Part A, section 2.2): v1's joint optimum on GM5 is not a
   stationary point; whether to add GM5 to the section 3 regression.

## Status in one table

| Item | Result |
| --- | --- |
| Phase 2 close-out | rounding explanation demonstrated on all 20 runs; GM5 explained (misfit gradient, not mask); tie-breaks logged (N1-N5); manifest rebuilt |
| Section 3 optimizer regression (GM1, AP5) | acrit stages pass; six baseline stages running (jobs in `03_optimizer_regression_jobs.json`) |
| Section 6 measurement diagnostics | delivered; theta rise with threshold is a pixel-population effect; per-tile theta above ~3e3 m^2 rests on 2-8 basins |
| Phase 3 performance | steps 2-5 committed and gated on ten tiles (<= 1.3e-12 of relief); 8-12x per evaluation; 33 GB -> 3.5 GB; optimizer gate partial (acrit pass), six stages running |
| Stop | Phase 3 review point reached |

---

# Part A. Phase 2 close-out (02d)

## Phase 2 close-out: what the Phase 2 report asserted, tested (addendum 2, section 2)

Directive: `reports/directives/refactor_v2_addendum_2.md`, sections 0, 2, 4, 5 (sections 3, 6, 7 follow).
Branch `v2`; code at the commits named in each run's `run.json`. Environment `lemcvt`
(Python 3.12, numpy 2.5.4, scipy 1.18.1). All jobs on `serc`. Date: 2026-11-09.
Machine-generated tables: `02_regression_tables.md` (sections 2.1b, 2.1c, 2.3a), the CSV files
`02d_linearized_recovery.csv`, `02d_sweep_tiebreak_gate.csv`, `02d_sweep_tiebreak_changed_rows.csv`.

Every number below is tied to a run id (`$GROUP_HOME/lemcvt_results/<run id>`) or a Slurm job id.

### Summary

| Item | Result |
| --- | --- |
| 0. State check | `v2` head at start 840203e; tests 149/149 pass on serc (job 51693537); all 43 run directories cited by the Phase 2 reports exist |
| 5. Push check | **Blocked.** `v2` tracks 10 figure PDFs under `reports/v2/figures_v2/` (forbidden by the "any `*.pdf`" rule). No file > 50 MB, no secret on any branch. Nothing pushed; decision needed (section 5) |
| 2.1 Rounding explanation | **Demonstrated.** No full-precision values exist; the linearized recovery passes on all 20 AD-opt and joint runs: every recovered offset lies inside the printed-precision box (largest at 94 % of the box), residual max |.| <= 3.2e-10 x relief against a 1e-8 tolerance, and the predicted MSE differences equal the measured ones to 3 figures |
| 2.2 GM5 joint | **Explained, differently.** The mask cannot change (0 cells in the rounding interval on every tile and run, GM2 acrit aside, where it did not change either). GM5's 6e-6 comes from a misfit gradient dMSE/dlog10 K = -76 at the printed joint point (other tiles: |grad| <= 2.7) times a D rounding offset at 94 % of its box; the v1 joint Nelder-Mead stopped on GM5 far from a stationary point in K. Phase 3 may start |
| 2.3 `n_diff_cells` replaced | by counts at 1e-9 and 1e-6 x relief and the rounding-interval count (code + tables) |
| 2.4 Bookkeeping | 4 of 500 ratios drifted (2 rows x 2 routings); the duplicate dx = 50 row is v1's own (two targets round to the same 4 x 4 grid) |
| 2.5 Tie-breaks | 5 sorts made deterministic (`docs/numerics_changes.md`, N1-N5); gate rerun: 84 rows identical, 163 rows change <= 1e-13 relative, 3 rows change where ties exist |
| 2.6 D-inf partitioned model | **Diagnosed; fix proposed, not applied.** The D-inf channel integration builds channel elevations through hillslope cells and mixes different flow paths: on AP1 5695 of 6459 D-inf channel cells are integrated through cells below A_crit, 985 are not higher than a receiver that carries weight, channel elevations reach 5775 ft on a tile whose top is 1441 ft; only 847 channel cells connect to the boundary through channel cells. A synthetic V-valley reproduces it (identical to D8 when every angle is cardinal or diagonal; off by one relief when rotated 20 degrees). The AD D-inf model is unaffected because it solves a linear system instead of integrating |
| Found on the way | `data/manifest.csv` was a 3-entry stub (fixed, 270 entries, job 51695557); the Phase 2 `regression-report` could only show one stage per tile (fixed) |

### 0. Orientation and state check

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

### 5. Push check (done first, as section 8 requires)

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

### 2.1 The rounding explanation, tested

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


### 2.2 GM5 joint

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

### 2.3 `n_diff_cells` replaced

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


### 2.4 Bookkeeping in the Phase 2 tables

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

### 2.5 Deterministic tie-breaking (logged numerics change)

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

### 2.6 The D-inf partitioned model: diagnosis

Run `20261109T210700Z_dinf_diagnostics_AP1_GM1` (job 51697246, the replay weighted by receiver coefficients; first run `20261109T204305Z_dinf_diagnostics_AP1_GM1`, job 51694667, gave the same MSEs and branch counts; `lemcvt closeout dinf --tiles AP1,GM1`,
`src/lemcvt/analyses/dinf_diagnostics.py`), at the v1 `d8_vs_dinf` parameters (K0, D0 rounded
to 2 s.f., A_crit = 5000 m^2; AP1 in feet as v1 ran). The partitioned-model MSEs reproduce the
Phase 2 / v1 values (AP1: D8 2.1950e+03, D-inf 3.7650e+06 ft^2; GM1: 7.9204e+02,
4.2395e+03 m^2). Grids written to the run directory: `<tile>_pt_d8_error`, `<tile>_pt_dinf_error`,
`<tile>_Zchannel_dinf`, `<tile>_integration_branch`, `<tile>_masks_dinf1_d8_2`.

#### 2.6.1 Where the error sits (Z - Z_obs by cell class; RMS / max |.| / mean, tile z unit)

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

#### 2.6.2 The D-inf channel set and its integration

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
reproduces the model's channel elevations exactly on both tiles (`trace_reproduces_model` true,
max difference 0.0e+00 and 0.0e+00). Algebraically the two-receiver step is
z = (1 - tan r) z1 + tan r z2 + S0 dx / cos r, so a receiver with zero coefficient (r = 0: z2;
r = pi/4: z1) is not counted as used in the statistics below; on the channel cells set by the
two-receiver branch the fraction with both coefficients nonzero is 0.38 (AP1) and
0.30 (GM1), median tan r 0.24 and 0.06.

Measured on the channel cells (A_dinf >= 5000 m^2):

| quantity | AP1 | GM1 |
| --- | --- | --- |
| channel cells, D8 / D-inf | 5657 / 6459 | 5318 / 6141 |
| D-inf channel cells connected to the boundary through channel cells (via the receivers used) | 847 | 2896 |
| ... not connected | 5612 | 3245 |
| D8 channel cells whose receiver is a hillslope cell | 4 | 0 |
| D-inf channel cells whose integration path passes a cell with A_dinf < A_crit | 5695 | 5671 |
| D-inf channel cells integrated directly from a receiver of smaller area | 2156 | 1641 |
| branch that set the value: both / cardinal-only / diagonal-only / boundary | 5150 / 225 / 589 / 495 | 5213 / 294 / 338 / 296 |
| channel cells NOT higher than the receiver they were integrated from (monotonicity violated) | 985 of 5964 | 545 of 5845 |
| median / min drop to the receiver used (z unit) | 0.31 / -597 | 0.12 / -21 |
| max channel elevation, D-inf / D8 / observed max | 5775 / 1454 / 1441 | 596 / 400 / 425 |
| channel cells more than one relief above the observed maximum | 4978 | 302 |
| RMS error of the channel elevations against the DEM, D-inf / D8 | 2356 / 15.9 | 64 / 4.1 |
| effective step dL = (z - z_ref)/S0: median / 1 % / min / count negative | 0.975 / -25 / -647 / 642 | 1.000 / -4 / -178 / 317 |
| S0 at A_crit / S0 at one cell's area (z unit per m) | 1.15 / 141 | 0.265 / 6.68 |
| cells never reached (NaN) | 0 | 0 |

Answers to the four questions of addendum 2 (2.6.2):

- *Is every channel cell connected to the Dirichlet boundary?* Through channel cells, no: only
  847 of 6459 on AP1 and 2896 of 6141 on GM1. The D-inf receivers of a channel cell can have smaller
  D-inf area than the cell itself (flow is split), so A_dinf >= A_crit is not a connected
  network. Through any cells, yes: every cell is reached (0 NaN).
- *Is channel elevation monotone upstream?* No: 985 (AP1) and 545 (GM1) of the integrated
  channel cells (17 % and 9 %) are not higher than a receiver that carries a nonzero coefficient;
  the median drop is 0.31 and 0.12 (z units) and the minimum -597 and -21. The mixing formula subtracts (z1 - z2) sin r / cos r, and when the two receivers lie
  on different flow paths their elevations differ by metres to hundreds of metres, which the
  formula interprets as a slope along the facet.
- *How are the path, dL and A defined?* As above: no path, a two-receiver average; dL = dx/cos r
  nominally, but the effective step (z - z_ref)/S0 ranges from -647 to 622 m on AP1 (642 negative);
  A is the integrated cell's own D-inf area, but its receivers' elevations were themselves
  integrated with S0 of **their** (often much smaller) areas: 5695 of 6459 AP1 channel cells
  and 5671 of 6141 GM1 channel cells sit on an integration chain that passed through cells
  below A_crit, where S0 is up to 123 x (AP1) and 25 x (GM1) larger than at A_crit.
  This is what lifts the channel elevations to 5775 ft on a tile whose highest point is 1441 ft.
- *Does any path fail to terminate at the boundary?* No cell is left NaN; every chain ends at
  a boundary cell. Termination is not the problem.

#### 2.6.3 A_crit = 5000 against the D-inf area distribution

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

#### 2.6.4 Synthetic cases (same run; `synthetic_*` grids in the run directory)

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

#### 2.6.5 Ranked hypotheses

1. **The channel integration is not an integration along a path** (confirmed, primary). The
   D-inf partitioned model assigns z = (1 - tan r) z1 + tan r z2 + S0 dx / cos r using the two
   receivers' already-integrated elevations and the cell's own S0, for every cell of the grid.
   Because D-inf area is not monotone along either receiver link, channel cells are built on
   top of hillslope cells whose S0 is 25-120 x larger (2.6.2, "passes a cell with A < A_crit":
   88-92 % of channel cells), and the two receivers often belong to different flow paths, so
   the (z1 - z2) term adds or removes metres per step (negative effective steps on 317-642
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

#### 2.6.6 Proposed fix (not implemented; model code unchanged)

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


### 4. Slurm and storage (addendum 2, section 4)

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

### What the evidence supports, and where it is weak

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


### Next

- Section 3 (optimizer regression, GM1 and AP5, one serc job per stage, 140 core-hours):
  submitted after this report (job ids in `reports/v2/03_optimizer_regression_jobs.json`).
- Section 6 (`lemcvt measure-diagnostics --tiles all`): runs while those jobs run; report
  `02c_measurement_diagnostics.md`.
- Section 7 (Phase 3 performance) starts once this close-out is accepted.
- Decisions needed from Steve: the push (section 5 options a-c); whether the D-inf fix of 2.6.6
  is to be implemented in Phase 3 or deferred to Phase 6.6; whether GM5 should be added to the
  optimizer regression given 2.2.



---

# Part B. Measurement diagnostics for the threshold decision (02c)

## Measurement diagnostics for the threshold decision (addendum 2, section 6)

Directive: `reports/directives/refactor_v2_addendum_2.md`, section 6; outline fixed by 6.8.
Run: `20261109T212622Z_measure_diagnostics_10tiles` (serc job 51698894, code 458ed05 at start; the first
run `20261109T211109Z_measure_diagnostics_10tiles`, job 51697704, lacked the A <= 1e5 scaling-break
variant and is otherwise identical). Code: `src/lemcvt/measure/diagnostics.py`; figures:
`src/lemcvt/figures/measure_diagnostics.py` -> `reports/v2/figures_v2/02c_*.png`; tables:
`reports/v2/02c_tables.md` (generated by `reports/v2/scripts/make_02c_tables.py`, T1-T8 below refer
to it). The C_ht(L) curves come from the Phase 2 `measure` runs (`20261109T19..Z_measure_<tile>`).
Nothing here writes to a tile config. Date: 2026-11-09.

**Two framing points, kept throughout.**

- *theta = m in the models.* Whatever A_c rule is chosen fixes the stream-power exponent m of every
  model fitted on that tile, not only the starting K0. Tables T1 and T8 therefore give, per
  option, the exponent the models would run with; a change of A_c from 1e3 to 1e4 m^2 changes m on
  AP from about 0.39 to 0.58 (section 2), which is a different model, not a re-tuned one.
- *k_s has units of m^(2 theta).* k_s values at different theta are not comparable, and the
  k_s columns of T1/T2/T8 are compared only within a column (same A_c, hence nearly the same theta)
  or through the common-theta k_sn of section 5 (T5). The 02b table that placed k_s at theta_roll,
  theta_plateau and the v1 theta side by side should be read with that caveat.

### 1. What was run

- Pre-registered before the run (stated in the module docstring and `params.json["settings"]`):
  basin-fixed set = basins with >= 50 channel pixels at A_c = 1e4 m^2; per-basin filters for every
  chi fit: >= 10 channel pixels, channel relief >= 10 m (ten times the vertical noise of the lidar
  and TIN surfaces, and the vertical interval of the slope-area method), chi range >= 1 m at a
  fixed theta_ref = 0.45 (so that the filter does not move with the theta being fitted; about
  40-100 m of channel at A ~ 1e4 m^2); bootstrap unit = basin, 200 draws, 68 % and 95 % percentile
  intervals; steps with fewer than 10 basins flagged; scaling break = continuous piecewise-linear
  WLS fits (1, 2, 3 segments, weights = pixel counts, breaks on 0.1-decade bin centres, BIC with
  k = 2, 4, 6) to the binned median D8 slope above the rollover, pixel bootstrap (200) of the
  2-segment break, once over all areas and once restricted to A <= 1e5 m^2; edge truncation =
  catchment with > 5 % of its outlet area from outside the tile or touching the edge away from the
  outlet; curvature synthetic = parabolic ridges, crest widths at the 5-90 % of the observed
  hilltop-width distribution plus a fixed set, variants smooth / TIN (facets of the size measured
  on AP) / noise (GM roughness) / both.
- Ladders: free (all basins), basin-fixed (1e2-1e4 m^2 only), interior-only, each 0.1 decade from
  1e2 to 1e5 m^2 (31 steps; 21 for the fixed ladder); site pooling of the per-basin statistics
  across the five tiles of a site at every step.
- Inputs: metre grids (AP converted from US survey feet), the tiles' own D8 area grids, no model
  output; theta grid 0.10-0.90 by 0.01; chi with the model's upstream rule and A0 = 1 m^2.
- Wall time 8 min for ten tiles (2 CPUs); the whole thing is cheap to rerun.

### 2. Basin-fixed against free ladder (6.1)

With N_min = 50 the basin-fixed set holds 2-9 basins per tile (T3: AP1 2 of 10, AP4 6 of 9,
GM1 9 of 23, GM5 2 of 19). Over the 21 steps from 1e2 to 1e4 m^2 the basin-fixed theta* equals
the free theta* within 0.01 on every tile (T3, last column), although the free fit uses 30-150
basins at 1e2 m^2 and the fixed fit the same 2-9 throughout.

Reading: the rise of theta* with A_c (AP 0.32 -> 0.52 pooled, GM 0.27 -> 0.34; T1) is a
**pixel-population effect**: within one and the same set of basins, dropping the low-area
pixels raises the fitted concavity. It is not basin selection. Two caveats that belong with that
statement: (i) the through-origin SSR objective is dominated by the largest basins (largest chi
and z ranges), which is why 2 basins reproduce the fit of 110; the "pooled" theta* is in effect the
theta of the few largest networks on the tile, and the basin bootstrap of the free ladder on AP1
and AP3 reaches down to the grid edge 0.10 for that reason (resamples that drop the dominant
basin leave degenerate fragments); (ii) on GM the free and fixed curves peak at 1e3-4e3 m^2 and
fall again above 1e4, so "rise" describes AP, while GM is flat to slightly concave.

### 3. Fine ladder (6.2)

T1 (theta* with 68 % intervals, basins flagged below 10) and T2 (k_s, basins, pixels, R2, 95 %
intervals); figures `02c_theta_ladders.png` (per tile, free / fixed / interior with 68 and 95 %
bands) and `02c_theta_ladders_pooled.png` (site-pooled with bands, tiles overlaid, breaks marked).

- Bootstrap unit: the basin, for every interval in this report (the breakpoint bootstrap of
  section 4 resamples pixels, stated there).
- Steps with fewer than 10 basins: per tile, every step above 2e3-4e3 m^2 on AP1-AP5 and GM1,
  GM3-GM5, above 8e3 on GM2 (T1, counts in parentheses). Only the site-pooled curves keep >= 10
  basins up to 2e4 m^2 (AP) and 1e5 m^2 (GM).
- Site-pooled theta* (basins): AP 0.32 (492) at 1e2, 0.39 (116) at 1e3, 0.43 (51) at 2e3,
  0.48 (25) at 4e3, 0.52 (16) at 1e4, 0.55 (9) at 4e4, 0.57 (7) at 1e5; GM 0.27 (612), 0.33 (68),
  0.34 (47), 0.33 (32), 0.31 (25), 0.27 (15), 0.26 (10). 68 % half-widths are 0.01-0.02 below 4e3
  m^2 and 0.03-0.05 at 1e4 and above (AP); GM 0.01 throughout except the last step.
- R2 of the chi-z fit is 0.94-0.999 on every step of every tile (T2); it rises with A_c because the
  fit is dominated by fewer, longer profiles, not because the model fits better.
- k_s(theta*) along the ladder rises with theta* on AP by a factor of 20 between 1e2 and 1e4 m^2
  (2.2 -> 24, pooled): that is the unit change of k_s with theta (m^0.64 -> m^1.04), not a
  steepness change; section 5 gives the comparable quantity.

### 4. Scaling break (6.3)

Figure `02c_scaling_break.png`; table T4.

The binned slope-area relation above the rollover has, on every tile, a nearly flat segment
(slope -0.01 to -0.18 in log S per log A) that runs from the rollover (20-450 m^2) up to
10^3-10^4 m^2, followed by a steeper decline. Fitted over the whole area range the two-segment
break lands at 4.5e4-2.2e5 m^2 on AP with slopes of -4 to -14 above it: that is the last one or
two bins (A > 1e5 m^2, trunk-channel pixels whose area is inherited from outside the 1 km^2 tile;
a few hundred pixels), and the pixel bootstrap is degenerate there (the same bin in every draw).
Restricted to A <= 1e5 m^2 the breaks move to AP1 4.5e4, AP2 3.5e3, AP3 2.2e3, AP4 1.1e4, AP5 5.6e4;
GM1 7.1e2, GM2 2.8e3, GM3 35 (a bin at the rollover), GM4 8.9e3, GM5 3.6e2 m^2, with 68 % intervals
spanning 0.1-1.4 decades. BIC prefers three segments on 6 of 10 tiles, i.e. the curve is not a
two-power-law shape. Pooled per site (all pixels of the five tiles): AP 2.2e3 m^2 (A <= 1e5;
1.8e5 over all A), GM 447 m^2 (5.6e4 over all A).

A S^2 at the break (T4, last column): AP 115-1274 (median per tile), GM 4-790 m^2; the pooled
values are 128 [30, 313] (AP) and 49 [8, 117] (GM) for the A <= 1e5 breaks (median and 16-84 %). These are given as the directive asks, as
a report-only alternative to a fixed threshold; their spread across tiles is a decade, so they do
not define a site constant either.

What the evidence supports here: a transition from an area-insensitive (hollow / colluvial)
slope regime to a declining one exists on every tile and sits between a few hundred and a few
thousand m^2 on GM and between 2e3 and 1e4 m^2 on most AP tiles. Where it is weak: the location
depends on the fitted range and on the model order, the per-tile bootstrap intervals are wide,
and the slope of the "fluvial" segment above the break (-0.2 to -0.5 on GM, -0.4 to -2 on AP) is far
from the chi-derived theta of 0.3-0.6, because binned local slopes at 1 m on a TIN or lidar surface
are not the profile slopes the chi fit uses. The break is a guide to the order of magnitude, not a
threshold.

### 5. Common-theta k_sn (6.4)

T5. Reference concavities: theta_ref = 0.45 (the directive's second reference) and the site-pooled
theta* at the ladder step nearest the site's A <= 1e5 break: AP theta(2.0e3 m^2) = 0.43, GM
theta(398 m^2) = 0.31 (the all-A breaks give 0.57 and 0.28, at 1e5 and 6e4 m^2, and are not used
because they sit on the inherited-area bins). k_sn is the pooled through-origin chi-z slope at
that theta over the channel network above the site break (and, second block, above 1e4 m^2).

At theta_ref = 0.45 (units m^0.9): AP 9.1-11.2 (five tiles within 20 %), GM 9.1-16.9 (GM4 the
outlier). At the site thetas the two sites are in different units again (AP m^0.86, GM m^0.62) and
not comparable with each other, only within a site: AP 7.2-8.9, GM 2.2-4.1. These numbers put every
tile in one unit for the record and are **not** inputs to the models.

### 6. Edge effects (6.5)

**Provenance of the `*_area` grids.** They were computed on a larger DEM and then clipped, not on
the 1 km tiles: the Phase 0 manifest shows maximum areas of 1.1-1.7e7 m^2 on AP and 8.9e6 m^2 on GM
inside 1 km^2 tiles (`00_reference.md`, section 4), the GM area grids are integer multiples of 1 m^2
and the AP ones of 0.9507 m^2 (the native cell), and the legacy notebooks
(`legacy/measurement_notebooks/updated_best_fit_tile_finder.ipynb`, cells 5-6) select the tiles on
the 30 m `AP_Lidar_overlap_30m_*` grids and then crop the 1 m `filled`, `area` and `fd` grids from
"full grids" with a `gdal_translate` window. Consequence: the drainage area of every pixel is the
true one, including the part of the catchment outside the tile; what the tile truncates is the
channel network available to the fit (the upstream part of an entering basin is missing, and
entering trunk channels appear as single-pixel "basins" at the edge, which the >= 10-pixel rule
drops), and the hilltop set (divides are defined only from the in-tile network).

**Truncated catchments** (T6; a basin is truncated when its in-tile pixel count times the cell
area is more than 5 % short of its outlet area, or its catchment touches the edge away from the
outlet). Of the ~4000 edge-outlet catchments per tile, 37-71 % are truncated, and they hold most of
the channel pixels: at A_c = 1e3 m^2, 40 % (AP4) to 97 % (AP2, AP5) of the channel pixels on AP and
31 % (GM3) to 99 % (GM5) on GM lie in truncated basins; at 1e4 m^2 the fractions are 0.49-1.00 (AP)
and 0.33-1.00 (GM). The tiles are, by construction (they were chosen for model fit, not for basin
completeness), windows on larger networks.

**Interior-only theta** (T6, last columns; green curves in `02c_theta_ladders.png`). Restricting
the fit to basins whose divides lie inside the tile leaves small catchments only: the fit is
degenerate (theta* at the grid edge 0.10 on 18 of 26 steps on AP1, 14 of 14 on GM5, 17 of 25 on
GM2; 9 tiles have fewer than 10 interior basins at 1e4 m^2). Where enough interior basins exist
the interior theta agrees with the free one: AP4 0.36 (23 basins) against 0.36 at 1e3 m^2, GM1 0.36
(15) against 0.32, GM3 0.38 (7) against 0.35. So the free-ladder values are not visibly biased by
truncation where this can be tested, but the test is possible on three tiles only.

### 7. Curvature checks and synthetic test (6.6)

**Definition.** C_ht is the Laplacian, nabla^2 z = z_xx + z_yy, taken from the least-squares
quadratic over a (2w+1)^2 window as 2(a + b), i.e. the Laplacian of the locally smoothed surface;
for w = 1 it is the 9-point finite-difference Laplacian. This is the quantity in the hilltop balance
U = -D nabla^2 z and matches the models' 5-point Laplacian, so D0 = U/C_ht is consistent with the
diffusion operator. Stated in `docs/measurement_methods.md`, section 7 (commit 8151c38).

**C_ht(L) for every tile.** `02c_curvature_scale.png`: the median hilltop curvature falls with
window length on every tile (roughness and, on AP, facets dominate below 10 m), and the plateau
rule picked L* = 9, 11, 15, 21, 9 m on GM1-GM5 and 50, 20, 40, 40, 20 m on AP1-AP5 (`02b`, table 2).

**Synthetic ridges** (`02c_ridge_bias.png`, T7). Parabolic crests with the site's median C_ht as
the true curvature (AP 0.0057, GM 0.0116 1/m), crest widths from 3 to 284 m covering the 5-90 %
range of the observed hilltop widths (2 x the distance from a hilltop pixel to the nearest channel
pixel at A_c = 1e4: medians 28-50 m on AP, 31-102 m on GM; T7), TIN facets of 2.6-2.8 m (the median
equivalent length of planar patches measured on the AP tiles) and Gaussian noise of 0.043 m (the
GM roughness, median absolute deviation from a 5 x 5 mean):

- Facets and noise change the recovered curvature by less than 1 % at every L >= 9 m; they only
  matter below 7 m, where the window is a few facets wide. The AP L* of 20-50 m is far above that.
- The bias that does exist comes from the crest width: with a window longer than the parabolic
  crest the quadratic fit averages in the straight flanks and under-estimates the curvature.
  At each tile's L*, the bias at the **median** observed width is between -6.5 % (AP3) and +1.1 %
  (GM5): under the pre-registered 10 % at both sites. At the 25 % width it is -14 to -59 % on all
  five AP tiles and -15 % on GM4 (GM1-GM3, GM5: within 3 %). The width below which the bias exceeds
  10 % is 15-32 m on AP (28-50 % of the AP hilltop pixels lie below it) and 6-15 m on GM (8-31 %).

**Pre-registered rule.** "If the bias exceeds 10 % at either site's L*, report it and stop for a
decision on the curvature method." At the site-typical (median) crest width the rule is not
triggered. It is triggered for the narrow-crest part of the hilltop population, which on AP is a
third to a half of the hilltop pixels; because C_ht is the median over all hilltop pixels, a bias
confined to the lower half of the width distribution shifts the median by less than the bias of
the median-width ridge (-6.5 % at most), but it is not zero and it is site-dependent (AP more
than GM, since L* is larger there). I report this as the rule requires and do not produce further
curvature-derived values (no D0 in T8) until the decision; the rest of this report and the Phase 3
work do not depend on C_ht. Two method options for that decision: (i) keep the window rule and
exclude hilltop pixels closer than L*/2 to a channel (removes the biased population; shrinks the
AP hilltop set by about a third); (ii) a width-adaptive window (smaller of L* and the local
hilltop width), which keeps the pixels but reintroduces the small-window roughness bias on AP.

### 8. AP provenance (6.7; read-only, nothing downloaded)

1. **The 990 x 990 grids.** The legacy AP measurements (`updated_best_fit_tile_finder.ipynb`,
   cells 10-12) load `data/best_fit_tiles/after_chi_opt/no_tif_extensions_for_topoanalysis/AP_1m_best_tile_{1..5}_{filled,area,fd}`
   and print `grid 990x990, dx=0.9998259595959833 m`. The path is relative to the notebook's
   project root on Steve's Mac (`/Users/scdobbs/Documents/GitHub/Manuscript_submission`,
   `00_reference.md` section 6); the notebooks carry no file dates. A name search of the
   directories I may read on Sherlock (`$GROUP_HOME/scdobbs_optimize`, `scdobbs_hilley_lem`,
   `scdobbs_energy`, `$SCRATCH`, `$HOME`; patterns `*best_tile*`, `*after_chi_opt*`,
   `*best_fit_tiles*`, `*Lidar_overlap*`, `*taylors_original*`) finds only the 1015-pixel model
   tiles under `landscape-optimizer/new_models_101625/` (sha256-identical to `$LEMCVT_DATA`).
   **The 990 x 990 grids are not on Sherlock.** Georeferencing: 990 x 0.99983 m = 989.83 m per side;
   the printed tile extent for the AP best tile (cell 5: x 556893-557883, y 4422470-4423460,
   EPSG:26917) equals the model tile AP5's extent (556893.65-557883.48, 4422470.78-4423461.0 from
   the GeoTIFF headers) to the metre.
2. **Legacy method on them.** Cannot be run: the grids are absent. What can be said: the same
   legacy chi code run on the model tiles reproduces the hard-coded GM theta and k_s exactly on
   all five GM tiles and does not reproduce the AP ones (AP1 theta 0.575 and k_s 52.3 against the hard-coded 0.562 and
   42.2 after unit conversion; `02b`, section 4). So the hard-coded AP values came from the
   990-pixel rasterization and cannot be regenerated from what is on Sherlock.
3. **The two grid families.** Same footprint (989.83 m), 990 against 1015 pixels, spacings 0.99983
   and 0.97520 m. 0.97520 m is PAMAP's native 3.2 ft (0.97536 m) within 0.02 %, so the model tiles
   are at the native rasterization; 0.99983 m is what a 1 m target over a 989.83 m window gives
   (`gdal_translate -tr 1 1` or `-outsize 990 990`). The 990-pixel family is therefore a resample
   of the 0.9752 m data (or of the same source at 1 m), not an independent product; which kernel
   was used cannot be determined without the files. The 1.3 % spacing difference is below the
   facet scale (2.6-2.8 m) and is unlikely to be the reason the AP theta differs; the different
   hydrology (fill, D8, area on a resampled surface) and the legacy method's edge-outlet basins
   are the likelier reasons, and both are testable only with the files.

### 9. What the evidence supports, and where it is weak

Measured (the tables), then inferred (the sentences).

- **theta depends on A_c because of which pixels enter, not which basins** (T3: free = fixed
  within 0.01 on all tiles). Inferred: the channel-head/hollow transition is inside the 1e2-1e4
  m^2 range on both sites, and the chi fit mixes two regimes below ~1e3 m^2 (AP) and ~5e2 m^2 (GM).
- **AP and GM differ in shape**: AP theta* rises monotonically to 0.52-0.57 by 1e4-1e5 m^2 with
  7-16 pooled basins; GM peaks at 0.34 near 2e3 m^2 and falls to 0.26-0.31 above 1e4 (T1). Inferred:
  on GM the fluvial network above 1e4 m^2 is short inside 1 km tiles and its theta is poorly
  constrained (the 95 % intervals widen to 0.1-0.3 there).
- **Per-tile values above 2e3-4e3 m^2 rest on 2-8 basins** on every tile (T1). Inferred: tile-level
  theta at A_c >= 4e3 m^2 is not a measurement with a usable uncertainty; the site-pooled value is.
- **The slope-area break is of the right order (1e2-1e4 m^2) but not a number** (T4: BIC prefers
  three segments on most tiles; breaks move by a decade with the fitted range).
- **Truncation is severe** (T6): most channel pixels belong to catchments cut by the tile edge;
  the interior-only test is possible on three tiles and agrees with the free ladder there.
- **Curvature**: the window-scale rule is defensible at the site's typical crest width; the
  narrow-crest population biases C_ht low on AP by an amount that depends on how many such pixels
  there are (T7); facets and noise are not the issue at L >= 9 m.
- Weak points not yet mentioned: the basin filters remove 40-80 % of the basins at 1e4 m^2
  (T2 "used / total"), so the >= 10-basin flag is often tripped by the filters rather than by the
  landscape; the pooled SSR objective's weighting by basin size is a choice (an equal-weight or
  per-basin-median theta would be a different estimator and was not run); and no value here has
  been checked against an independent field constraint of channel-head area at either site.

### 10. Options for the threshold rule (T8: theta, k_s, K0 = U/k_s per tile)

| Option | theta (AP / GM, pooled) | consequences |
| --- | --- | --- |
| (a) fixed A_c = 1e3 m^2 for both sites | 0.39 / 0.33 | 13-32 basins per tile (GM5: 4), theta 68 % intervals <= 0.02 on most tiles; m near the slope-area transition on AP; k_s 4-6 (AP), 2.7-6.8 (GM); K0 0.9-1.4e-5 (AP), 2.2-5.5e-5 (GM). The models would run with m about 0.39 on AP and 0.33 on GM; A_c sits below the model's own fitted A_crit (1.9-9.7e3 in v1), so the measured network includes cells the model treats as hillslope |
| (b) fixed A_c = 1e4 m^2 for both sites | 0.52 / 0.31 | 2-8 basins per tile: tile values unusable, site values only (AP 68 % 0.49-0.56); k_s 10-57 (AP), 2.4-8 (GM) with per-tile factors of 5 from basin sampling; K0 1.0e-6-5.4e-6 (AP), 1.8-6.0e-5 (GM). Closest to v1's thresholds (5e4 AP, 1e4 GM) and to the model's A_crit range; m about 0.52 on AP, 0.31 on GM |
| (c) site-specific break (AP 2.2e3, GM 4.5e2 m^2) | 0.43 / 0.30 | 4-17 basins (AP), 23-51 (GM); m 0.40-0.47 on AP, 0.28-0.31 on GM; k_s 5.5-12.6 (AP), 1.6-4.1 (GM); K0 4.4e-6-9.9e-6 (AP), 3.6e-5-9.1e-5 (GM). Site-dependent as the literature expects, but the break itself carries a decade of uncertainty (section 4) |
| (d) slope-area rollover (hollow network, 18-450 m^2; `02b` table 2) | 0.30-0.35 / 0.22-0.26 | theta and k_s of the colluvial network; the sites become almost indistinguishable; not the regime the models apply the channel law to |

Separate from the evidence, my recommendation: (a) with the site-pooled theta as the exponent
and the per-tile k_s at that theta for K0, because it is the only option whose per-tile values
have intervals of their own (>= 10 basins on 9 of 10 tiles), it keeps the measurement independent
of the model's A_crit, and 1e3 m^2 lies inside the slope-area transition on both sites. The cost
is that AP's m (0.39) is lower than v1's hard-coded 0.40-0.59 and than the chi fit at the fluvial
scale (0.52 at 1e4 m^2); whether the models should carry the exponent of the network they
threshold (option a, c) or of the fluvial trunk (option b) is the decision, and it is Steve's.


---

# Part C. Phase 3 performance (03)

## Phase 3: performance (addendum 2, section 7)

Directive: `refactor_v2_directive.md` Phase 3 as ordered by addendum 2, section 7. Branch `v2`,
commits 9d046e3 (step 2), 898467c (step 3), db11fee (step 4), a224750 (step 5), a6f8c64
(evaluator fix). Environment `lemcvt` plus `line_profiler` 5.1.0 (added with `pip install
--no-deps`, recorded in `envs/sherlock.yml`). All jobs on `serc`. Date: 2026-11-09.
Machine-generated tables: `03_tables.md` (`reports/v2/scripts/make_03_tables.py`; P1 profile,
G1 gates, G2 wall times, O1 optimizer stages). Every number is tied to a run id or job id.

### Summary

| Item | Result |
| --- | --- |
| Step 1 profile (GM1, AP5) | AD solve 135 s = 103 s CSR->LIL->CSR + 24 s SuperLU; AD build 170 s of LIL loops; partitioned build 17 s (12 s a Python generator over every cell); hillslope solve 4.9 s = SuperLU; 2 Picard iterations 15 s = SuperLU; fill 23 s |
| Step 2 Picard assembly | vectorised, bit-identical (tests); gate 50/50 pass, max 1.3e-12 of relief |
| Step 3 AD Dirichlet rows | assembled once, no LIL; bit-identical (tests); gate 35/35 on 7 tiles (3 tasks stuck on exit, see 2.3), covered by steps 4 and 5 |
| Step 4 joint chi precompute | last-bit rounding only (N6); gate 50/50, max 1.3e-12 |
| Step 5 D8 BFS / accumulation | numba, bit-identical (tests); gate 50/50, max 1.3e-12 |
| Speedup per forward evaluation | AD 169 -> 25 s (6.8x; 266 -> 25 against Phase 2, 11x); partitioned 11 -> 4-5 s; joint 13 -> 5-7 s; nonlinear 50 Picard: GM 8 min -> 39 s (12x), AP 45 min -> 333 s (8x) |
| Peak memory | 32-34 GB -> 3.0-3.5 GB per full-tile forward run; 2.2-2.6 GB per optimizer stage |
| CPU efficiency | 0.93-0.98 of one core; single-threaded (SuperLU) |
| Optimizer gate | acrit stages: GM1 and AP5 pass against v1 and equal the section 3 baseline to all printed digits; the other six Phase 3 stages and five baseline stages are running (job ids below) |
| Slurm defaults | updated to 1 CPU, 8 GB for model steps (`configs/slurm/sherlock.yaml`) |

### 1. Profile (step 1)

`lemcvt profile --tile {GM1,AP5} --picard-iters 2` (job 51698240, runs
`20261109T211900Z_profile_GM1`, `20261109T211902Z_profile_AP5`, code 0fef8c1 with step 2 already
in the tree; `cProfile` and `line_profiler` outputs are in the run directories:
`<step>.pstats`, `<step>_cprofile.txt`, `<step>_lineprof.txt`). Table P1. cProfile roughly
doubles the cost of Python-level loops, so the LIL numbers are upper bounds; the SuperLU and
numpy times are unaffected.

| operation (GM1) | wall (s) | self time |
| --- | --- | --- |
| AD build (`advection_diffusion_model`, v1 loops) | 170 | LIL `__setitem__`/`__getitem__` and their shape checks; 1e6 cells x 5 entries |
| AD solve (v1: `D*M_diff + K*M_adv`, `tolil`, row zeroing, `tocsr`, `spsolve`) | 135 | 103 s in the LIL round trip, 24 s in SuperLU (`gssv`), 6 s array copies |
| partitioned build (`extract_grid_data` + channel BFS) | 17 | 11.6 s in `deque(k for k in range(n) if not isnan(Z[k]))`, 3.4 s copies, 1.7 s `unravel_index` |
| partitioned solve (vectorised assembly, step 2) | 4.9 | 4.6 s SuperLU |
| nonlinear, 2 Picard iterations (3 solves + 2 slope evaluations) | 14.9 | 13.9 s SuperLU |
| joint rebuild for a new K (v1) | 17 | as the partitioned build |
| priority-flood fill / D8 directions / D8 area / main-stem graph | 23 / 0.1 / 1.7 / 3.1 | heap loop / vectorised / 1e6-cell loop / 1e6-cell loop |

AP5 is within 10 % of GM1 on every row (1.03e6 against 1.0e6 cells). Peak RSS reached 32 GB
at the first AD solve (the LIL copy of a 5e6-entry matrix) and stayed there.

Consequences for the plan: the AD stage was 85 % LIL overhead, the partitioned and nonlinear
stages were already SuperLU-bound after step 2 (which the profile ran with), the joint stage's
per-K rebuild was 17 s of which 12 s was one generator expression, and the hydrology loops
only matter for the resolution sweep.

### 2. The five steps, each with its forward gate

Working rules followed: one logical change per commit; the v1 implementation of every changed
routine is kept under a `_v1` name and a test asserts bit-identity on synthetic tiles
(`tests/test_models.py::TestPhase3*`, 150 -> 159 tests); the forward regression ran after each
step on all ten tiles against the Phase 2 v2 outputs (`lemcvt tile forward --params v1
--compare --reference-run phase2`, tolerance max |dz| <= 1e-8 x relief and relative MSE
<= 1e-10; table G1). Steps 3 and 4 were gated from shared clones checked out at their commits
(`$SCRATCH/lemcvt_snap_898467c`, `..._db11fee`) because the main tree had moved on.

#### 2.1 Step 2: vectorised Picard assembly (9d046e3)

`assemble_hillslope_system` builds the reduced hillslope matrix and right-hand side from index
arrays with v1's floating-point operations in v1's order (diagonal accumulated once per
neighbour in offset order, off-diagonals single products, fixed-neighbour terms subtracted in
the same order), so matrix, right-hand side and solution are bit-identical to the per-cell LIL
loop (test on scalar and per-cell D, and entry-by-entry on the matrix). `_compute_slopes` is
vectorised with the same `(0 - left) + right` then `/ count` order. Used by the linear and
nonlinear partitioned models, D8 and D-inf.

Symbolic-factorisation reuse: SuperLU through `scipy.sparse.linalg` exposes no separate
symbolic phase, so the Picard iterations (fixed pattern, changing values, non-symmetric because
v1 scales row i by D_i) cannot reuse it without a different solver (UMFPACK via
`scikit-umfpack`, not installed). For the linear partitioned model at fixed A_crit the
D-dependence is closed-form (z = L^-1(b/D) terms), which the Phase 4 reduction uses; it was not
added here because the Nelder-Mead stages change A_crit (hence the pattern) at every step. Left
as a documented non-change.

Gate (job 51698701, runs `20261109T2122..Z_tile_forward_<tile>`): 50 of 50 comparisons pass,
max |dz|/relief 1.26e-12, max relative MSE difference 2.5e-12. GM1 is identical to the last bit;
the 1e-13 differences elsewhere are BLAS reductions with 2 instead of 7 threads (the Phase 2
runs had 7 CPUs), not the assembly.

#### 2.2 Step 3: AD stencils from index arrays, Dirichlet rows at build time (898467c)

`advection_diffusion_model` assembles M_diff and M_adv as COO/CSR with v1's entry values and
order, removes the Dirichlet rows once (`R @ M`) and adds an identity on them; each evaluation
is `D*M_diff0 + K*M_adv0 + I_bc` and one `spsolve`. The matrix equals v1's post-round-trip
matrix entry for entry and pattern for pattern (test), and the solutions are bit-identical.

Gate (job 51700403, runs `20261109T2138..Z_tile_forward_<tile>_step3`): 35 of 35 comparisons on
AP1-AP5, GM1, GM2 pass, max 7.3e-13. The GM3, GM4 and GM5 tasks completed their five forward
runs (log lines at 13:41-13:44) but the processes did not exit; they ran on sh02-10n62, -n68 and
-n69, next to sh02-10n61 and -n64, which `sinfo` listed as draining with "NHC: Lustre client"
errors; the snapshot clone is on `$SCRATCH` (Lustre). They were cancelled after 20 min
(section 6). Those three tiles are covered by the step 4 and step 5 gates, which contain the
step 3 code.

#### 2.3 Step 4: joint stage integrates the channel network once (db11fee)

`precompute_chi_network` integrates chi = sum A^-m dL and the anchor elevation per cell;
`partitioned_threshold_model_chi` gives `Z_channel = z_anchor + (U/K) chi` and
`optimize_threshold_full` uses it (`rebuild_per_K=True` restores v1). This is the one step
that is not bit-identical: the sum is scaled once instead of term by term, a last-bit
difference (<= 9e-16 relative on channel elevations, <= 1.2e-14 of relief on the surface of a
synthetic tile; logged as N6 in `docs/numerics_changes.md`; test tolerance 1e-12 of relief).

Gate (job 51700404, `..._step4`): 50 of 50 pass, max 1.26e-12 of relief, i.e. the same level as
steps 2 and 3; N6 does not show above the BLAS-thread noise. The forward runs themselves do not
exercise the chi path (they call the per-K model), so the N6 exemption is only needed, and only
at the 1e-14 level, for the joint-stage optimizer runs of section 4.

#### 2.4 Step 5: D8 receiver tables, donor lists and BFS with numpy and numba (a224750)

`precompute_channel_network` and `precompute_chi_network` now build `downstream_of`, `dL` and a
CSR of donors with numpy and run the breadth-first integration in a numba loop (Python
reference kept); `d8_area`'s accumulation loop is compiled. Arithmetic and visiting order are
v1's: bit-identical (tests on the channel network and on the area grid). The priority-flood fill
(23 s, heap in Python) was left alone: it is used only by the resolution sweep on coarse grids
and by tests.

Gate (job 51700412, runs `20261109T2148..Z_tile_forward_<tile>`): 50 of 50 pass, max 1.26e-12.

#### 2.5 Evaluator fix found by the gate (a6f8c64)

`forward.evaluate` (and the close-out's `_evaluate`) used `cache.setdefault("AD",
advection_diffusion_model(...))`, which evaluates its argument eagerly and so rebuilt the AD
model on every call; the Phase 2 forward runs paid the 170 s build twice per tile and the
close-out's linearized recovery five times per tile. Numerics unaffected; fixed. `run.json` now
also records the git hash at run start (the hash at finish was wrong whenever a commit landed
during a job).

### 3. Speedup, memory and CPU efficiency

Per forward evaluation, median over the five tiles of a site (G2; seconds; `ad_no_opt`
includes the AD build, `nonlinear_opt` runs the full Picard iteration to convergence):

| code | site | AD build + solve | AD solve | partitioned (build + solve) | nonlinear (<= 50 Picard) | joint (build + solve) |
| --- | --- | --- | --- | --- | --- | --- |
| Phase 2 (v1 loops; from the Phase 2 run logs) | GM1 / AP5 | 266 / 250 | 262 / 251 | 50 / 49 | 486 / 2696 | 51 / 50 |
| step 2 | GM / AP | 162 / 169 | 163 / 169 | 10 / 11 | 43 / 353 | 11 / 13 |
| step 3 | GM / AP | 24 / 26 | 24 / 26 | 12 / 12 | 46 / 324 | 14 / 13 |
| step 4 | GM / AP | 25 / 27 | 25 / 27 | 10 / 11 | 36 / 339 | 11 / 13 |
| step 5 | GM / AP | 24 / 28 | 24 / 27 | 4 / 5 | 39 / 333 | 5 / 7 |

(The step-2 AD numbers include a rebuild per call, section 2.5; the step 3-5 numbers do not.
The Phase 2 "AD solve" is the time between the `ad_no_opt` and `ad_opt` log lines of
`20261109T181246Z_tile_forward_GM1` and `20261109T182914Z_tile_forward_AP5`.)

Speedups, Phase 2 -> step 5: AD evaluation 11x (266 -> 24 s; what remains is the SuperLU
factorisation of a 1e6-row non-symmetric matrix, 24 s); partitioned evaluation 10-12x
(50 -> 4-5 s; SuperLU on the reduced Laplacian, 4-5 s); joint evaluation 10x (and no rebuild per
K in the optimizer, section 4); nonlinear forward 12x on GM (486 -> 39 s) and 8x on AP
(2696 -> 333 s; AP needs 50 iterations, GM 10-20). The remaining cost everywhere is one sparse
LU per solve. Hydrology: D8 area 1.7 s -> compiled loop (milliseconds); channel BFS 17 s -> ~2 s.

Peak memory (sacct MaxRSS of the gate tasks): Phase 2 code 32.2-33.9 GB per forward run; step 3
onwards 3.0-3.5 GB (job 51700403: 3.07-3.42 GB; 51700404: 2.96-3.42 GB; 51700412: 3.04-3.51 GB).
Optimizer stages: 2.2-2.6 GB (jobs 51701025, 51701029, 51698287, 51697689).

CPU efficiency: TotalCPU/Elapsed 0.93-0.98 on every gate task with 2 CPUs allocated, i.e. one
core busy; the second was idle. `OMP/MKL/OPENBLAS_NUM_THREADS` are pinned to the allocation.

Slurm defaults updated from these measurements (`configs/slurm/sherlock.yaml`, commented):
1 CPU and 8 GB (1.5 x 3.5 GB rounded up) for `tile_run`, `tile_forward`, `closeout` and
`profile`; `tile_forward` time 45 min; the §3 time rule (1.5 x v1 stage time) is kept for
`submit-stages` although Phase 3 stages are 10x faster, so that the limit never binds.

### 4. Optimizer gate (GM1 and AP5 against the section 3 baseline, section 3 rule)

Section 3 baseline (Phase 2 code; jobs in `03_optimizer_regression_jobs.json`): GM1 acrit and
AP5 acrit are complete and pass against v1 (A_crit within 4e-5, MSE within 3e-5, table O1).
Three of the eight baseline jobs failed at start on a run-directory name collision (two stages
of one tile starting in the same second) and were resubmitted after the fix (stage-specific,
collision-proof run ids); GM1 ad, nonlinear, joint and AP5 ad, nonlinear, joint are running
(jobs 51697692-51697696, 51698285, 51698289; AP5 nonlinear has a 32 h limit).

Phase 3 optimizer runs (code a6f8c64; jobs 51701024-51701031, `03_optimizer_phase3_jobs.json`):
GM1 acrit and AP5 acrit are complete. Against the baseline they give the same A_crit and MSE to
every printed digit (A_crit 423.493 and 1130.04; MSE 100.96 and 359.26) with the same number of
evaluations and solves (35/32 and 34/29), in 122 s against 853 s (GM1, baseline on v1 loops)
and 136 s against 119 s (AP5, whose baseline job started after step 2 was in the tree). The
other six stages are running; the AD stages should take about 60 x 25 s, the joint stages
about 300 x 6 s, the nonlinear stages 1-5 h.

Pass rule to apply when they finish (addendum 2, section 3): parameters within 1 %, MSE within
0.5 %; an A_crit miss counts as a pass with a note if the MSE is within 0.5 % and both A_crit
values select the same channel mask. Steps 2, 3 and 5 are bit-identical, so the Phase 3 AD,
acrit and nonlinear stages must reproduce the baseline trajectories exactly (same evaluations,
same misfits; `trajectory.parquet` allows a row-by-row check); the joint stage may differ at the
N6 level (1e-14 of relief per evaluation), which cannot move a Nelder-Mead step unless two
vertices tie to 14 digits. `lemcvt regression-report` (table 2.2) merges the per-stage runs per
tile; the comparison of Phase 3 against baseline trajectories is the one item left to add to
`make_03_tables.py` once both sets exist.

### 5. What the evidence supports, and where it is weak

- Measured: every Phase 3 step reproduces the Phase 2 outputs on all ten tiles at 1e-12 of
  relief, four orders below the tolerance, and the unit tests show bit-identity for steps 2, 3
  and 5 and last-bit rounding for step 4. The speedups and memory figures are from the gate
  runs and `sacct`, not projections.
- Inferred: the optimizer-stage speedups (section 4) are projected from per-evaluation times for
  the stages still running; the two completed stages agree with the projection.
- Weak: the step 3 gate covered seven tiles directly (three tasks did not exit; the tiles are
  covered by the later gates that include the same code); the Phase 2 reference runs used 7
  BLAS threads and the gates 2, so "bit-identical" is shown by the tests on synthetic tiles and
  by GM1 on the full tile, while the other tiles show thread-count noise at 1e-13; the joint
  stage's N6 exemption has not yet been exercised on a full optimizer run.
- Not done, on purpose: symbolic-factorisation reuse (no API in scipy's SuperLU; would need
  `scikit-umfpack`), the fill (heap in Python, sweep-only), the D-inf models' pure-Python loops
  (the partitioned D-inf model is defective, `02d` 2.6, and should be rebuilt before being
  optimised), and any change of solver, which would move results beyond the last bit.

### 6. Housekeeping

- Three step-3 gate tasks (51700403_7, _8, _9) were cancelled with `scancel` after finishing
  their work but not exiting (section 2.2); see the session summary for the node ids in case
  Steve wants to report them.
- The job script's temporary tee log now goes to the log directory instead of the repository
  root (template fix, 2026-11-09).
- `$GROUP_HOME` at this checkpoint: see the session summary (`sh_quota`).

### Stop

Phase 3 stops here for review, with the optimizer gate's six remaining stages running under
the job ids above. Nothing is pushed (section 5 of addendum 2 is still blocked on the figure
PDFs).


---

# Appendix 1. Tables for Part B (02c_tables.md)

## Tables for 02c (generated by `reports/v2/scripts/make_02c_tables.py` from run `20261109T212622Z_measure_diagnostics_10tiles`, git db11feef, Slurm job 51698894)

Settings: A_ref_fixed = 10000.0, N_min_fixed_basins = 50, min_relief_m = 10.0, min_chi_range_m = 1.0, theta_ref_chi = 0.45, min_pixels_per_basin = 10, min_basins_flag = 10, n_boot = 200, bootstrap_unit_chi = basin, bootstrap_unit_break = pixel, theta_ref_common = 0.45

### T1. Free ladder: theta* [basin-bootstrap 68 %] (n basins when fewer than 10)

| tile | 100 | 200 | 398 | 794 | 1e+03 | 2e+03 | 3.98e+03 | 7.94e+03 | 1e+04 | 2e+04 | 3.98e+04 | 1e+05 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | 0.32 [0.11, 0.32] | 0.34 [0.10, 0.34] | 0.35 [0.10, 0.36] | 0.37 [0.10, 0.37] | 0.39 [0.10, 0.39] | 0.46 [0.10, 0.47] | 0.53 [0.10, 0.54] (7) | 0.58 [0.10, 0.58] (5) | 0.58 [0.10, 0.58] (5) | 0.59 [0.10, 0.60] (3) | 0.60 (1) | 0.52 (1) |
| AP2 | 0.32 [0.28, 0.33] | 0.34 [0.29, 0.36] | 0.36 [0.31, 0.38] | 0.40 [0.35, 0.42] | 0.41 [0.36, 0.43] | 0.47 [0.42, 0.49] (4) | 0.53 (2) | 0.57 (2) | 0.57 (2) | 0.53 (2) | 0.55 (2) | 0.77 (2) |
| AP3 | 0.33 [0.10, 0.33] | 0.35 [0.10, 0.35] | 0.36 [0.11, 0.36] | 0.38 [0.14, 0.38] | 0.38 [0.14, 0.38] | 0.40 [0.16, 0.40] | 0.43 [0.10, 0.43] (4) | 0.45 (2) | 0.45 (2) | 0.48 (1) | 0.54 (1) | 0.59 (1) |
| AP4 | 0.27 [0.24, 0.27] | 0.30 [0.27, 0.31] | 0.32 [0.30, 0.33] | 0.35 [0.34, 0.36] | 0.36 [0.35, 0.37] | 0.39 [0.39, 0.46] | 0.49 [0.48, 0.55] (9) | 0.59 [0.58, 0.78] (5) | 0.59 [0.57, 0.90] (5) | 0.58 [0.57, 0.81] (3) | 0.47 [0.47, 0.74] (3) | 0.21 (1) |
| AP5 | 0.33 [0.28, 0.34] | 0.34 [0.29, 0.36] | 0.36 [0.31, 0.37] | 0.38 [0.34, 0.39] | 0.39 [0.34, 0.40] (9) | 0.42 [0.37, 0.43] (5) | 0.44 [0.43, 0.51] (3) | 0.45 (2) | 0.50 (2) | 0.48 (2) | 0.42 (2) | 0.10 (2) |
| GM1 | 0.25 [0.24, 0.25] | 0.26 [0.26, 0.27] | 0.29 [0.29, 0.30] | 0.32 [0.31, 0.34] | 0.32 [0.31, 0.36] | 0.34 [0.33, 0.40] | 0.37 [0.35, 0.41] (8) | 0.37 [0.36, 0.42] (8) | 0.38 [0.36, 0.43] (8) | 0.37 [0.36, 0.41] (6) | 0.35 [0.33, 0.36] (4) | 0.82 [0.73, 0.90] (4) |
| GM2 | 0.26 [0.25, 0.28] | 0.28 [0.26, 0.30] | 0.28 [0.27, 0.31] | 0.30 [0.29, 0.33] | 0.31 [0.29, 0.33] | 0.32 [0.30, 0.34] | 0.30 [0.28, 0.33] (8) | 0.31 [0.28, 0.37] (6) | 0.31 [0.29, 0.37] (5) | 0.35 [0.32, 0.39] (5) | 0.47 [0.43, 0.48] (3) | 0.42 [0.23, 0.53] (3) |
| GM3 | 0.26 [0.24, 0.28] | 0.27 [0.25, 0.29] | 0.28 [0.26, 0.30] | 0.32 [0.28, 0.36] | 0.35 [0.32, 0.39] | 0.41 [0.39, 0.44] | 0.44 [0.42, 0.49] (9) | 0.42 [0.37, 0.49] (7) | 0.42 [0.35, 0.53] (7) | 0.30 [0.28, 0.35] (5) | 0.14 [0.14, 0.15] (4) | 0.10 (1) |
| GM4 | 0.27 [0.26, 0.27] | 0.29 [0.28, 0.29] | 0.31 [0.30, 0.33] | 0.34 [0.33, 0.37] | 0.35 [0.34, 0.38] | 0.36 [0.35, 0.39] (8) | 0.34 [0.32, 0.35] (5) | 0.32 [0.22, 0.33] (3) | 0.31 [0.21, 0.33] (3) | 0.30 (2) | 0.27 (2) | 0.10 (1) |
| GM5 | 0.28 [0.27, 0.29] | 0.30 [0.29, 0.32] | 0.31 [0.30, 0.33] | 0.31 [0.31, 0.32] (6) | 0.32 [0.31, 0.32] (4) | 0.32 [0.32, 0.33] (3) | 0.32 (2) | 0.30 (2) | 0.30 (2) | 0.28 (2) | 0.27 (2) | 0.26 (1) |
| **AP pooled** | 0.32 [0.31, 0.32] | 0.34 [0.33, 0.34] | 0.35 [0.35, 0.36] | 0.38 [0.37, 0.39] | 0.39 [0.38, 0.40] | 0.43 [0.41, 0.45] | 0.48 [0.45, 0.51] | 0.51 [0.48, 0.55] | 0.52 [0.49, 0.56] | 0.52 [0.49, 0.55] | 0.55 [0.51, 0.56] (9) | 0.57 [0.51, 0.59] (7) |
| **GM pooled** | 0.27 [0.26, 0.27] | 0.28 [0.28, 0.29] | 0.30 [0.30, 0.31] | 0.33 [0.32, 0.33] | 0.33 [0.32, 0.34] | 0.34 [0.33, 0.35] | 0.33 [0.32, 0.35] | 0.32 [0.31, 0.33] | 0.31 [0.30, 0.32] | 0.30 [0.27, 0.31] | 0.27 [0.25, 0.29] | 0.26 [0.17, 0.27] |

### T2. Free ladder: k_s(theta*) (units m^(2 theta), not comparable across columns), basins, pixels, R2

| tile | A_c | theta* | k_s | basins used / total | pixels | R2 | 95 % theta |
|---|---|---|---|---|---|---|---|
| AP1 | 1e+03 | 0.39 | 5.36 | 32 / 92 | 20627 | 0.955 | [0.1, 0.39] |
| AP1 | 2e+03 | 0.46 | 12.5 | 17 / 45 | 10906 | 0.960 | [0.1, 0.47] |
| AP1 | 3.98e+03 | 0.53 | 29.7 | 7 / 25 | 6461 | 0.972 | [0.1, 0.54] |
| AP1 | 1e+04 | 0.58 | 56.8 | 5 / 12 | 4162 | 0.986 | [0.1, 0.58] |
| AP1 | 3.98e+04 | 0.60 | 72.6 | 1 / 6 | 2473 | 0.997 | None |
| AP2 | 1e+03 | 0.41 | 6.26 | 13 / 112 | 28285 | 0.963 | [0.23975, 0.4310000000000002] |
| AP2 | 2e+03 | 0.47 | 12.6 | 4 / 54 | 15689 | 0.964 | [0.1, 0.49] |
| AP2 | 3.98e+03 | 0.53 | 25.8 | 2 / 35 | 9359 | 0.964 | None |
| AP2 | 1e+04 | 0.57 | 43.3 | 2 / 18 | 5059 | 0.970 | None |
| AP2 | 3.98e+04 | 0.55 | 32.9 | 2 / 4 | 2698 | 0.987 | None |
| AP3 | 1e+03 | 0.38 | 4.42 | 32 / 113 | 24307 | 0.953 | [0.11, 0.38] |
| AP3 | 2e+03 | 0.40 | 5.55 | 10 / 49 | 12830 | 0.942 | [0.14, 0.5102500000000001] |
| AP3 | 3.98e+03 | 0.43 | 7.88 | 4 / 22 | 7881 | 0.940 | [0.1, 0.67] |
| AP3 | 1e+04 | 0.45 | 10.1 | 2 / 11 | 4135 | 0.967 | None |
| AP3 | 3.98e+04 | 0.54 | 33 | 1 / 5 | 2548 | 0.996 | None |
| AP4 | 1e+03 | 0.36 | 3.98 | 30 / 93 | 22745 | 0.986 | [0.33, 0.46050000000000013] |
| AP4 | 2e+03 | 0.39 | 5.52 | 15 / 45 | 12182 | 0.983 | [0.39, 0.51] |
| AP4 | 3.98e+03 | 0.49 | 16.3 | 9 / 22 | 6528 | 0.984 | [0.43975000000000003, 0.6202500000000001] |
| AP4 | 1e+04 | 0.59 | 47 | 5 / 9 | 3096 | 0.986 | [0.56, 0.9] |
| AP4 | 3.98e+04 | 0.47 | 13.3 | 3 / 7 | 1384 | 0.999 | [0.47, 0.86] |
| AP5 | 1e+03 | 0.39 | 4.58 | 9 / 70 | 22502 | 0.970 | [0.16975, 0.4] |
| AP5 | 2e+03 | 0.42 | 6.46 | 5 / 36 | 12479 | 0.975 | [0.17875000000000002, 0.6] |
| AP5 | 3.98e+03 | 0.44 | 7.99 | 3 / 17 | 6883 | 0.980 | [0.1, 0.51] |
| AP5 | 1e+04 | 0.50 | 17.6 | 2 / 12 | 3461 | 0.992 | None |
| AP5 | 3.98e+04 | 0.42 | 6.1 | 2 / 12 | 2277 | 0.998 | None |
| GM1 | 1e+03 | 0.32 | 3.39 | 19 / 43 | 8617 | 0.991 | [0.3, 0.39] |
| GM1 | 2e+03 | 0.34 | 4.22 | 11 / 32 | 6285 | 0.993 | [0.32, 0.44] |
| GM1 | 3.98e+03 | 0.37 | 5.94 | 8 / 27 | 5403 | 0.998 | [0.35, 0.48] |
| GM1 | 1e+04 | 0.38 | 6.61 | 8 / 23 | 4656 | 0.999 | [0.35, 0.51] |
| GM1 | 3.98e+04 | 0.35 | 4.59 | 4 / 12 | 2954 | 0.999 | [0.33, 0.37] |
| GM2 | 1e+03 | 0.31 | 2.67 | 16 / 48 | 11134 | 0.971 | [0.26, 0.3802500000000001] |
| GM2 | 2e+03 | 0.32 | 2.91 | 11 / 33 | 7630 | 0.980 | [0.26, 0.4] |
| GM2 | 3.98e+03 | 0.30 | 2.21 | 8 / 27 | 5985 | 0.987 | [0.28, 0.43] |
| GM2 | 1e+04 | 0.31 | 2.43 | 5 / 20 | 4393 | 0.994 | [0.28, 0.42] |
| GM2 | 3.98e+04 | 0.47 | 15.8 | 3 / 12 | 2071 | 1.000 | [0.37, 0.48] |
| GM3 | 1e+03 | 0.35 | 3.53 | 15 / 43 | 12031 | 0.986 | [0.28, 0.41] |
| GM3 | 2e+03 | 0.41 | 6.94 | 14 / 27 | 8373 | 0.989 | [0.39, 0.49] |
| GM3 | 3.98e+03 | 0.44 | 9.94 | 9 / 21 | 6158 | 0.991 | [0.4, 0.57] |
| GM3 | 1e+04 | 0.42 | 7.99 | 7 / 15 | 4181 | 0.993 | [0.34, 0.57025] |
| GM3 | 3.98e+04 | 0.14 | 0.275 | 4 / 5 | 1648 | 0.999 | [0.10975, 0.19] |
| GM4 | 1e+03 | 0.35 | 6.75 | 14 / 49 | 10878 | 0.967 | [0.25925000000000004, 0.38] |
| GM4 | 2e+03 | 0.36 | 7.71 | 8 / 37 | 7675 | 0.970 | [0.28975, 0.5302500000000001] |
| GM4 | 3.98e+03 | 0.34 | 6.02 | 5 / 30 | 5960 | 0.972 | [0.32, 0.61] |
| GM4 | 1e+04 | 0.31 | 4 | 3 / 18 | 4533 | 0.975 | [0.21, 0.33] |
| GM4 | 3.98e+04 | 0.27 | 2.28 | 2 / 7 | 2696 | 0.988 | None |
| GM5 | 1e+03 | 0.32 | 3.3 | 4 / 53 | 14467 | 0.978 | [0.1, 0.32] |
| GM5 | 2e+03 | 0.32 | 3.3 | 3 / 34 | 10327 | 0.980 | [0.1, 0.33] |
| GM5 | 3.98e+03 | 0.32 | 3.28 | 2 / 24 | 8057 | 0.983 | None |
| GM5 | 1e+04 | 0.30 | 2.51 | 2 / 19 | 5645 | 0.987 | None |
| GM5 | 3.98e+04 | 0.27 | 1.65 | 2 / 9 | 3358 | 0.999 | None |
| **AP pooled** | 1e+03 | 0.39 | 5.05 [4.48, 5.39] | 116 (5 tiles) | - | 0.965 | [0.37, 0.41] |
| **AP pooled** | 2e+03 | 0.43 | 8.05 [6.4, 9.9] | 51 (5 tiles) | - | 0.963 | [0.39, 0.46] |
| **AP pooled** | 3.98e+03 | 0.48 | 14.5 [9.96, 20.5] | 25 (5 tiles) | - | 0.965 | [0.43975000000000003, 0.53] |
| **AP pooled** | 1e+04 | 0.52 | 24.1 [15.6, 38.5] | 16 (5 tiles) | - | 0.978 | [0.47, 0.57] |
| **AP pooled** | 3.98e+04 | 0.55 | 36.2 [20.9, 41.8] | 9 (5 tiles) | - | 0.996 | [0.46, 0.58] |
| **AP pooled** | 1e+05 | 0.57 | 47.7 [21.2, 63.8] | 7 (5 tiles) | - | 0.997 | [0.3795, 0.6705000000000001] |
| **GM pooled** | 1e+03 | 0.33 | 3.65 [3.18, 4.31] | 68 (5 tiles) | - | 0.977 | [0.31, 0.35] |
| **GM pooled** | 2e+03 | 0.34 | 4.11 [3.67, 4.99] | 47 (5 tiles) | - | 0.981 | [0.32, 0.37] |
| **GM pooled** | 3.98e+03 | 0.33 | 3.64 [3.16, 4.5] | 32 (5 tiles) | - | 0.984 | [0.31, 0.37] |
| **GM pooled** | 1e+04 | 0.31 | 2.82 [2.45, 3.56] | 25 (5 tiles) | - | 0.988 | [0.28, 0.34] |
| **GM pooled** | 3.98e+04 | 0.27 | 1.69 [1.26, 2.31] | 15 (5 tiles) | - | 0.996 | [0.21975, 0.31] |
| **GM pooled** | 1e+05 | 0.26 | 1.45 [0.462, 1.63] | 10 (5 tiles) | - | 0.997 | [0.1, 0.4410000000000002] |

### T3. Basin-fixed ladder (basins with >= 50 channel pixels at A_c = 1e4) against the free ladder

| tile | basins kept / at 1e4 | 100 | 200 | 398 | 794 | 1e+03 | 2e+03 | 3.98e+03 | 7.94e+03 | 1e+04 | max abs(free - fixed) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | 2 / 10 | 0.32 / 0.32 | 0.34 / 0.34 | 0.36 / 0.35 | 0.38 / 0.37 | 0.39 / 0.39 | 0.47 / 0.46 | 0.53 / 0.53 | 0.58 / 0.58 | 0.58 / 0.58 | 0.01 |
| AP2 | 2 / 18 | 0.32 / 0.32 | 0.34 / 0.34 | 0.36 / 0.36 | 0.40 / 0.40 | 0.41 / 0.41 | 0.47 / 0.47 | 0.53 / 0.53 | 0.57 / 0.57 | 0.57 / 0.57 | 0.01 |
| AP3 | 3 / 11 | 0.33 / 0.33 | 0.35 / 0.35 | 0.36 / 0.36 | 0.38 / 0.38 | 0.38 / 0.38 | 0.40 / 0.40 | 0.43 / 0.43 | 0.45 / 0.45 | 0.45 / 0.45 | 0.00 |
| AP4 | 6 / 9 | 0.27 / 0.27 | 0.30 / 0.30 | 0.32 / 0.32 | 0.35 / 0.35 | 0.36 / 0.36 | 0.39 / 0.39 | 0.49 / 0.49 | 0.59 / 0.59 | 0.59 / 0.59 | 0.00 |
| AP5 | 2 / 12 | 0.33 / 0.33 | 0.34 / 0.34 | 0.36 / 0.36 | 0.38 / 0.38 | 0.39 / 0.39 | 0.42 / 0.42 | 0.44 / 0.44 | 0.45 / 0.45 | 0.50 / 0.50 | 0.00 |
| GM1 | 9 / 23 | 0.25 / 0.25 | 0.26 / 0.26 | 0.29 / 0.29 | 0.32 / 0.32 | 0.32 / 0.32 | 0.34 / 0.34 | 0.37 / 0.37 | 0.37 / 0.37 | 0.38 / 0.38 | 0.01 |
| GM2 | 6 / 20 | 0.26 / 0.26 | 0.28 / 0.28 | 0.28 / 0.28 | 0.30 / 0.30 | 0.31 / 0.31 | 0.32 / 0.32 | 0.30 / 0.30 | 0.31 / 0.31 | 0.31 / 0.31 | 0.01 |
| GM3 | 8 / 15 | 0.26 / 0.26 | 0.27 / 0.27 | 0.28 / 0.28 | 0.32 / 0.32 | 0.35 / 0.35 | 0.41 / 0.41 | 0.44 / 0.44 | 0.42 / 0.42 | 0.42 / 0.42 | 0.00 |
| GM4 | 3 / 18 | 0.27 / 0.27 | 0.29 / 0.29 | 0.31 / 0.31 | 0.34 / 0.34 | 0.35 / 0.35 | 0.36 / 0.36 | 0.34 / 0.34 | 0.32 / 0.32 | 0.31 / 0.31 | 0.01 |
| GM5 | 2 / 19 | 0.28 / 0.28 | 0.30 / 0.30 | 0.31 / 0.31 | 0.31 / 0.31 | 0.32 / 0.32 | 0.32 / 0.32 | 0.32 / 0.32 | 0.30 / 0.30 | 0.30 / 0.30 | 0.00 |

(cells: fixed / free)

### T4. Slope-area scaling break above the rollover (2-segment break; BIC-best model; pixel-bootstrap 68 % of log10 A_break)

| tile | rollover (m^2) | range | best (BIC) | A_break (m^2) | 68 % | slope below / above | A S^2 at break: median [16-84 %] |
|---|---|---|---|---|---|---|---|
| AP1 | 282 | all A | three | 1.78e+05 | [1.78e+05, 1.78e+05] | -0.180 / -11.53 | 414 [21.5, 793] |
| AP1 | 282 | A <= 1e5 | three | 4.47e+04 | [3.55e+04, 4.47e+04] | -0.119 / -1.95 | 1.27e+03 [104, 4.07e+03] |
| AP2 | 56.2 | all A | three | 2.24e+05 | [2.24e+05, 2.24e+05] | -0.064 / -14.43 | 115 [1.77e-20, 1.33e+03] |
| AP2 | 56.2 | A <= 1e5 | three | 3.55e+03 | [3.55e+03, 5.62e+03] | -0.012 / -0.41 | 190 [19.5, 486] |
| AP3 | 17.8 | all A | two | 1.41e+05 | [1.41e+05, 1.78e+05] | -0.052 / -8.17 | 274 [1.21e-20, 1.28e+03] |
| AP3 | 17.8 | A <= 1e5 | two | 2.24e+03 | [1.78e+03, 4.47e+04] | -0.019 / -0.48 | 146 [20.7, 347] |
| AP4 | 447 | all A | two | 4.47e+04 | [1.41e+04, 5.62e+04] | -0.155 / -3.95 | 318 [5.39e-21, 1.49e+03] |
| AP4 | 447 | A <= 1e5 | two | 1.12e+04 | [1.12e+04, 1.41e+04] | -0.092 / -1.22 | 554 [82.5, 1.47e+03] |
| AP5 | 56.2 | all A | two | 5.62e+04 | [5.62e+04, 1.78e+05] | -0.082 / -9.29 | 123 [5.25e-21, 1.22e+03] |
| AP5 | 56.2 | A <= 1e5 | two | 5.62e+04 | [2.24e+03, 5.62e+04] | -0.072 / -37.31 | 123 [5.25e-21, 1.22e+03] |
| GM1 | 35.5 | all A | three | 1.12e+03 | [1.12e+03, 1.12e+03] | -0.047 / -0.36 | 138 [31.7, 285] |
| GM1 | 35.5 | A <= 1e5 | three | 708 | [708, 891] | -0.043 / -0.30 | 94.8 [31.5, 206] |
| GM2 | 35.5 | all A | three | 1.78e+03 | [1.41e+03, 2.24e+03] | -0.082 / -0.38 | 122 [13.4, 324] |
| GM2 | 35.5 | A <= 1e5 | two | 2.82e+03 | [2.82e+03, 2.82e+03] | -0.085 / -0.46 | 265 [7.95, 516] |
| GM3 | 22.4 | all A | three | 1.78e+04 | [1.78e+04, 1.78e+04] | -0.168 / -0.39 | 311 [94.4, 578] |
| GM3 | 22.4 | A <= 1e5 | three | 35.5 | [35.5, 35.5] | -0.043 / -0.18 | 3.96 [0.913, 9.06] |
| GM4 | 56.2 | all A | three | 1.41e+04 | [1.12e+04, 5.62e+05] | -0.097 / -0.67 | 672 [250, 1.47e+03] |
| GM4 | 56.2 | A <= 1e5 | three | 8.91e+03 | [8.91e+03, 1.12e+04] | -0.095 / -0.53 | 790 [126, 1.58e+03] |
| GM5 | 28.2 | all A | two | 1.78e+05 | [1.78e+05, 2.24e+06] | -0.153 / -3.82 | 780 [43.2, 2.61e+03] |
| GM5 | 28.2 | A <= 1e5 | two | 355 | [282, 355] | -0.096 / -0.24 | 42.4 [5.88, 97] |
| **AP pooled** | 56.2 | all A | three | 1.78e+05 | [1.41e+05, 1.78e+05] | -0.070 / -11.93 | 417 [2e-20, 1.38e+03] |
| **AP pooled** | 56.2 | A <= 1e5 | three | 2.24e+03 | [2.24e+03, 2.24e+03] | -0.011 / -0.35 | 128 [30, 313] |
| **GM pooled** | 35.5 | all A | three | 5.62e+04 | [2.82e+04, 2.82e+05] | -0.139 / -1.33 | 372 [53.7, 1.11e+03] |
| **GM pooled** | 35.5 | A <= 1e5 | three | 447 | [447, 562] | -0.089 / -0.24 | 49.1 [7.65, 117] |

### T5. Common-theta steepness k_sn (pooled through-origin chi-z slope at a fixed theta; for comparison only)

| tile | A_c (site break, A <= 1e5) | k_sn(0.45) | k_sn(theta_site) | theta_site | basins | A_c = 1e4: k_sn(0.45) | k_sn(theta_site) | basins |
|---|---|---|---|---|---|---|---|---|
| AP1 | 2.24e+03 | 11.2 | 8.88 | 0.43 | 17 | 11.1 | 8.61 | 5 |
| AP2 | 2.24e+03 | 10.1 | 8.11 | 0.43 | 3 | 9.98 | 7.77 | 2 |
| AP3 | 2.24e+03 | 9.91 | 7.88 | 0.43 | 7 | 10.1 | 7.79 | 2 |
| AP4 | 2.24e+03 | 10.4 | 8.43 | 0.43 | 14 | 9.72 | 7.73 | 5 |
| AP5 | 2.24e+03 | 9.12 | 7.24 | 0.43 | 4 | 9.35 | 7.24 | 2 |
| GM1 | 447 | 12.9 | 3.02 | 0.31 | 45 | 14.8 | 2.93 | 8 |
| GM2 | 447 | 10.2 | 2.71 | 0.31 | 30 | 12.2 | 2.43 | 5 |
| GM3 | 447 | 9.12 | 2.23 | 0.31 | 23 | 11.3 | 2.22 | 7 |
| GM4 | 447 | 16.9 | 4.11 | 0.31 | 31 | 23 | 4 | 3 |
| GM5 | 447 | 13.2 | 2.9 | 0.31 | 20 | 16 | 2.85 | 2 |

### T6. Edge truncation: fraction of channel pixels in basins whose catchment is cut by the tile edge

| tile | basins (edge outlets) | truncated | interior | 1e2 | 1e3 | 1e4 | 1e5 | interior-only theta at 1e3 | at 1e4 |
|---|---|---|---|---|---|---|---|---|---|
| AP1 | 4056 | 2092 | 1964 | 0.69 | 0.75 | 0.84 | 1.00 | 0.10 [0.10, 0.10] | 0.10 (1) |
| AP2 | 4056 | 2866 | 1190 | 0.95 | 0.97 | 1.00 | 1.00 | 0.58 [0.33, 0.71] (8) | - |
| AP3 | 4054 | 2163 | 1891 | 0.75 | 0.86 | 0.96 | 1.00 | 0.14 [0.12, 0.18] | 0.10 (1) |
| AP4 | 4056 | 1969 | 2087 | 0.38 | 0.40 | 0.49 | 0.42 | 0.36 [0.10, 0.36] | 0.56 (2) |
| AP5 | 4056 | 1864 | 2192 | 0.91 | 0.97 | 1.00 | 1.00 | 0.16 [0.13, 0.22] (4) | - |
| GM1 | 3996 | 1474 | 2522 | 0.37 | 0.43 | 0.40 | 0.54 | 0.36 [0.33, 0.37] | 0.43 [0.40, 0.51] (5) |
| GM2 | 3996 | 1924 | 2072 | 0.82 | 0.89 | 0.95 | 1.00 | 0.10 [0.10, 0.23] (9) | 0.10 (1) |
| GM3 | 3996 | 2020 | 1976 | 0.35 | 0.31 | 0.33 | 0.17 | 0.38 [0.37, 0.40] (7) | 0.36 [0.35, 0.44] (4) |
| GM4 | 3996 | 2056 | 1940 | 0.86 | 0.93 | 0.99 | 1.00 | 0.21 [0.15, 0.24] | - |
| GM5 | 3996 | 2430 | 1566 | 0.94 | 0.99 | 1.00 | 1.00 | 0.10 (2) | - |

### T7. Parabolic-ridge test: bias of the recovered crest curvature at each tile's L*

| tile | L* (m) | hilltop width 2d: 10 / 25 / 50 / 75 / 90 % (m) | bias at median width | bias at 25 % width | width below which bias < -10 % | hilltop pixels in that regime |
|---|---|---|---|---|---|---|
| AP1 | 50 | 6 / 11 / 37 / 123 / 207 | -4.0 % | -59 % | 32 m | 46 % |
| AP2 | 20 | 4 / 10 / 32 / 117 / 249 | +0.6 % | -26 % | 15 m | 34 % |
| AP3 | 40 | 4 / 10 / 28 / 92 / 222 | -6.5 % | -56 % | 28 m | 50 % |
| AP4 | 40 | 6 / 15 / 50 / 165 / 284 | -0.1 % | -37 % | 28 m | 39 % |
| AP5 | 20 | 6 / 12 / 45 / 145 / 279 | +0.0 % | -14 % | 15 m | 28 % |

AP: c_true = 0.00574 1/m (the site's median C_ht), facet length 2.76 m, noise sigma 0.043 m; variant `tin_noise`.

| GM1 | 9 | 7 / 28 / 102 / 141 / 182 | +0.2 % | -1 % | 6 m | 8 % |
| GM2 | 11 | 6 / 15 / 91 / 164 / 205 | -0.9 % | -2 % | 7 m | 13 % |
| GM3 | 15 | 4 / 16 / 98 / 152 / 204 | -0.8 % | +0 % | 11 m | 20 % |
| GM4 | 21 | 6 / 12 / 42 / 148 / 210 | +0.1 % | -15 % | 15 m | 31 % |
| GM5 | 9 | 6 / 11 / 31 / 109 / 167 | +1.1 % | +3 % | 6 m | 11 % |

GM: c_true = 0.0116 1/m (the site's median C_ht), facet length 0.00 m, noise sigma 0.043 m; variant `noise`.

### T8. Options for the threshold rule: theta, k_s and K0 = U/k_s per tile (from the free ladder; K0 in the model's units)

| tile | U | (a) A_c = 1e3 m^2: theta / k_s / K0 | (b) A_c = 1e4 m^2: theta / k_s / K0 | (c) site break, A <= 1e5 (AP 2.2e3, GM 447): theta / k_s / K0 |
|---|---|---|---|---|
| AP1 | 5.5e-05 | 0.39 / 5.36 / 1.03e-05 (n=32) | 0.58 / 56.8 / 9.67e-07 (n=5) | 0.46 / 12.5 / 4.41e-06 (n=17) |
| AP2 | 5.5e-05 | 0.41 / 6.26 / 8.79e-06 (n=13) | 0.57 / 43.3 / 1.27e-06 (n=2) | 0.47 / 12.6 / 4.37e-06 (n=4) |
| AP3 | 5.5e-05 | 0.38 / 4.42 / 1.24e-05 (n=32) | 0.45 / 10.1 / 5.44e-06 (n=2) | 0.40 / 5.55 / 9.91e-06 (n=10) |
| AP4 | 5.5e-05 | 0.36 / 3.98 / 1.38e-05 (n=30) | 0.59 / 47 / 1.17e-06 (n=5) | 0.39 / 5.52 / 9.96e-06 (n=15) |
| AP5 | 5.5e-05 | 0.39 / 4.58 / 1.2e-05 (n=9) | 0.50 / 17.6 / 3.12e-06 (n=2) | 0.42 / 6.46 / 8.51e-06 (n=5) |
| GM1 | 0.000147 | 0.32 / 3.39 / 4.33e-05 (n=19) | 0.38 / 6.61 / 2.22e-05 (n=8) | 0.29 / 2.42 / 6.07e-05 (n=51) |
| GM2 | 0.000147 | 0.31 / 2.67 / 5.51e-05 (n=16) | 0.31 / 2.43 / 6.05e-05 (n=5) | 0.28 / 1.99 / 7.4e-05 (n=34) |
| GM3 | 0.000147 | 0.35 / 3.53 / 4.16e-05 (n=15) | 0.42 / 7.99 / 1.84e-05 (n=7) | 0.28 / 1.62 / 9.08e-05 (n=23) |
| GM4 | 0.000147 | 0.35 / 6.75 / 2.18e-05 (n=14) | 0.31 / 4 / 3.67e-05 (n=3) | 0.31 / 4.08 / 3.61e-05 (n=36) |
| GM5 | 0.000147 | 0.32 / 3.3 / 4.45e-05 (n=4) | 0.30 / 2.51 / 5.86e-05 (n=2) | 0.31 / 2.9 / 5.07e-05 (n=23) |

(c) uses the ladder step nearest the pooled site break (AP: 1995 m^2 for 2239; GM: 398 m^2 for 447). (d) is in `02b_measurement_first_results.md`, table 2.



---

# Appendix 2. Tables for Part C (03_tables.md)

### P1. Profile (step 1): wall time per operation on a full tile, cProfile on (adds overhead to Python loops)

| operation | GM1 (s) | AP5 (s) | peak RSS after (GB) | what dominates (self time) |
|---|---|---|---|---|
| hydro_fill | 23.3 | 23.8 | 0.1 | priority_flood_fill 0 s; <lambda> 0 s |
| hydro_d8_fd | 0.1 | 0.1 | 0.2 | <lambda> 0 s; d8_flow_direction 0 s |
| hydro_d8_area | 1.7 | 1.7 | 0.2 | d8_area 0 s; <lambda> 0 s |
| hydro_main_stem | 3.1 | 3.3 | 0.3 | full 0 s; <lambda> 0 s |
| ad_build | 169.6 | 170.6 | 4.3 | extract_grid_data 0 s; <lambda> 0 s |
| ad_solve | 135.1 | 136.5 | 32.1 | __rmul__ 0 s; <lambda> 0 s |
| ad_solve_2 | 136.2 | 140.6 | 32.1 | <lambda> 0 s; output_function 0 s |
| pt_build | 17.1 | 17.5 | 32.1 | extract_grid_data 0 s; <lambda> 0 s |
| pt_solve | 4.9 | 6.5 | 32.1 | where 0 s; output_function 0 s |
| pt_solve_2 | 4.9 | 5.4 | 32.1 | where 0 s; output_function 0 s |
| joint_rebuild_newK | 17.1 | 17.3 | 32.1 | extract_grid_data 0 s; <lambda> 0 s |
| nl_build | 13.4 | 13.7 | 32.1 | extract_grid_data 0 s; <lambda> 0 s |
| nl_solve_2iter | 14.9 | 19.7 | 32.1 | where 0 s; output_function 0 s |

Profile runs: GM1: `20261109T211900Z_profile_GM1`, AP5: `20261109T211902Z_profile_AP5`.

### G1. Forward gates: Phase 3 code against the Phase 2 v2 outputs (max |dz| / relief; rel. MSE diff), tolerances 1e-8 / 1e-10


#### step2 (9d046e3): 10 tiles, 50 of 50 comparisons pass; max |dz|/relief 1.26e-12, max rel. MSE diff 2.51e-12

| tile | run | ad_no_opt | ad_opt | acrit_opt | nonlinear_opt | joint_opt | peak RSS |
|---|---|---|---|---|---|---|---|
| AP1 | 20261109T212205Z_tile_forward_AP1 | 1.8e-13 / 5.4e-14 | 3.5e-14 / 1.5e-15 | 6.1e-13 / 2.5e-12 | 1.7e-13 / 1.0e-13 | 4.4e-13 / 2.2e-13 | - |
| AP2 | 20261109T212207Z_tile_forward_AP2 | 1.1e-13 / 2.2e-14 | 1.9e-14 / 7.0e-15 | 1.9e-13 / 7.5e-13 | 2.5e-13 / 1.7e-12 | 7.3e-13 / 1.4e-14 | - |
| AP3 | 20261109T212205Z_tile_forward_AP3 | 3.8e-13 / 1.6e-13 | 2.4e-14 / 2.5e-14 | 1.0e-13 / 4.6e-14 | 3.5e-13 / 7.2e-14 | 1.3e-12 / 5.5e-13 | - |
| AP4 | 20261109T212205Z_tile_forward_AP4 | 9.4e-13 / 4.3e-13 | 1.8e-14 / 1.3e-14 | 2.7e-13 / 7.9e-13 | 5.7e-13 / 1.0e-12 | 2.5e-13 / 3.4e-14 | - |
| AP5 | 20261109T212205Z_tile_forward_AP5 | 5.1e-13 / 1.5e-13 | 1.6e-14 / 1.7e-15 | 7.4e-13 / 1.2e-12 | 4.3e-13 / 2.3e-13 | 7.4e-13 / 2.4e-13 | - |
| GM1 | 20261109T212205Z_tile_forward_GM1 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |
| GM2 | 20261109T212204Z_tile_forward_GM2 | 2.7e-13 / 4.8e-14 | 1.2e-14 / 2.1e-15 | 8.5e-14 / 3.7e-14 | 5.3e-14 / 1.8e-14 | 1.3e-13 / 1.8e-14 | - |
| GM3 | 20261109T212204Z_tile_forward_GM3 | 6.9e-13 / 2.4e-13 | 1.5e-14 / 8.6e-15 | 3.5e-13 / 1.5e-12 | 8.3e-14 / 1.1e-13 | 2.9e-13 / 3.3e-13 | - |
| GM4 | 20261109T212207Z_tile_forward_GM4 | 4.7e-13 / 7.2e-14 | 4.9e-15 / 1.5e-15 | 2.7e-14 / 4.3e-14 | 4.1e-14 / 2.5e-15 | 1.8e-13 / 2.1e-13 | - |
| GM5 | 20261109T212207Z_tile_forward_GM5 | 7.6e-13 / 1.4e-13 | 1.1e-14 / 6.1e-15 | 6.4e-14 / 2.0e-13 | 5.1e-14 / 2.0e-14 | 8.1e-14 / 1.2e-13 | 32.2 GB |

#### step3 (898467c): 7 tiles, 35 of 35 comparisons pass; max |dz|/relief 7.32e-13, max rel. MSE diff 2.51e-12

| tile | run | ad_no_opt | ad_opt | acrit_opt | nonlinear_opt | joint_opt | peak RSS |
|---|---|---|---|---|---|---|---|
| AP1 | 20261109T213801Z_tile_forward_AP1_step3 | 1.7e-13 / 4.9e-14 | 3.8e-14 / 3.1e-14 | 6.1e-13 / 2.5e-12 | 1.7e-13 / 1.0e-13 | 4.4e-13 / 2.2e-13 | - |
| AP2 | 20261109T213800Z_tile_forward_AP2_step3 | 1.1e-13 / 2.3e-14 | 1.8e-14 / 1.5e-14 | 1.9e-13 / 7.5e-13 | 2.5e-13 / 1.7e-12 | 7.3e-13 / 1.4e-14 | - |
| AP3 | 20261109T213801Z_tile_forward_AP3_step3 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |
| AP4 | 20261109T213800Z_tile_forward_AP4_step3 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |
| AP5 | 20261109T213800Z_tile_forward_AP5_step3 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |
| GM1 | 20261109T213800Z_tile_forward_GM1_step3 | 3.7e-13 / 5.4e-14 | 9.5e-14 / 1.2e-14 | 1.2e-13 / 4.2e-13 | 8.5e-14 / 2.6e-14 | 6.2e-13 / 8.3e-13 | - |
| GM2 | 20261109T213800Z_tile_forward_GM2_step3 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |

#### step4 (db11fee): 10 tiles, 50 of 50 comparisons pass; max |dz|/relief 1.26e-12, max rel. MSE diff 2.51e-12

| tile | run | ad_no_opt | ad_opt | acrit_opt | nonlinear_opt | joint_opt | peak RSS |
|---|---|---|---|---|---|---|---|
| AP1 | 20261109T213800Z_tile_forward_AP1_step4 | 1.7e-13 / 4.9e-14 | 3.8e-14 / 3.1e-14 | 6.1e-13 / 2.5e-12 | 1.7e-13 / 1.0e-13 | 4.4e-13 / 2.2e-13 | - |
| AP2 | 20261109T213800Z_tile_forward_AP2_step4 | 1.1e-13 / 2.3e-14 | 1.8e-14 / 1.5e-14 | 1.9e-13 / 7.5e-13 | 2.5e-13 / 1.7e-12 | 7.3e-13 / 1.4e-14 | - |
| AP3 | 20261109T213800Z_tile_forward_AP3_step4 | 3.8e-13 / 1.6e-13 | 2.5e-14 / 6.0e-14 | 1.0e-13 / 4.6e-14 | 3.5e-13 / 7.2e-14 | 1.3e-12 / 5.5e-13 | - |
| AP4 | 20261109T213800Z_tile_forward_AP4_step4 | 9.5e-13 / 4.3e-13 | 1.5e-14 / 3.7e-15 | 2.7e-13 / 7.9e-13 | 5.7e-13 / 1.0e-12 | 2.5e-13 / 3.4e-14 | - |
| AP5 | 20261109T213800Z_tile_forward_AP5_step4 | 4.9e-13 / 1.4e-13 | 1.9e-14 / 1.2e-14 | 7.4e-13 / 1.2e-12 | 4.3e-13 / 2.3e-13 | 7.4e-13 / 2.4e-13 | - |
| GM1 | 20261109T213800Z_tile_forward_GM1_step4 | 1.2e-14 / 1.9e-16 | 2.2e-14 / 9.9e-15 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |
| GM2 | 20261109T213800Z_tile_forward_GM2_step4 | 2.3e-13 / 3.3e-14 | 1.1e-14 / 3.6e-15 | 8.5e-14 / 3.7e-14 | 5.3e-14 / 1.8e-14 | 1.3e-13 / 1.8e-14 | - |
| GM3 | 20261109T213800Z_tile_forward_GM3_step4 | 6.9e-13 / 2.4e-13 | 1.7e-14 / 1.8e-14 | 3.5e-13 / 1.5e-12 | 8.3e-14 / 1.1e-13 | 2.9e-13 / 3.3e-13 | - |
| GM4 | 20261109T213801Z_tile_forward_GM4_step4 | 4.9e-13 / 7.9e-14 | 5.3e-15 / 0.0e+00 | 2.7e-14 / 4.3e-14 | 4.1e-14 / 2.5e-15 | 1.8e-13 / 2.1e-13 | - |
| GM5 | 20261109T213800Z_tile_forward_GM5_step4 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |

#### step5 (a224750): 10 tiles, 50 of 50 comparisons pass; max |dz|/relief 1.26e-12, max rel. MSE diff 2.51e-12

| tile | run | ad_no_opt | ad_opt | acrit_opt | nonlinear_opt | joint_opt | peak RSS |
|---|---|---|---|---|---|---|---|
| AP1 | 20261109T214307Z_tile_forward_AP1 | 1.7e-13 / 4.9e-14 | 3.8e-14 / 3.1e-14 | 6.1e-13 / 2.5e-12 | 1.7e-13 / 1.0e-13 | 4.4e-13 / 2.2e-13 | - |
| AP2 | 20261109T214308Z_tile_forward_AP2 | 1.1e-13 / 2.3e-14 | 1.8e-14 / 1.5e-14 | 1.9e-13 / 7.5e-13 | 2.5e-13 / 1.7e-12 | 7.3e-13 / 1.4e-14 | - |
| AP3 | 20261109T214303Z_tile_forward_AP3 | 3.8e-13 / 1.6e-13 | 2.5e-14 / 6.0e-14 | 1.0e-13 / 4.6e-14 | 3.5e-13 / 7.2e-14 | 1.3e-12 / 5.5e-13 | - |
| AP4 | 20261109T214303Z_tile_forward_AP4 | 9.5e-13 / 4.3e-13 | 1.5e-14 / 3.7e-15 | 2.7e-13 / 7.9e-13 | 5.7e-13 / 1.0e-12 | 2.5e-13 / 3.4e-14 | - |
| AP5 | 20261109T214305Z_tile_forward_AP5 | 4.9e-13 / 1.4e-13 | 1.9e-14 / 1.2e-14 | 7.4e-13 / 1.2e-12 | 4.3e-13 / 2.3e-13 | 7.4e-13 / 2.4e-13 | - |
| GM1 | 20261109T214305Z_tile_forward_GM1 | 1.2e-14 / 1.9e-16 | 2.2e-14 / 9.9e-15 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |
| GM2 | 20261109T214311Z_tile_forward_GM2 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |
| GM3 | 20261109T214304Z_tile_forward_GM3 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |
| GM4 | 20261109T214303Z_tile_forward_GM4 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |
| GM5 | 20261109T214303Z_tile_forward_GM5 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 3.5 GB |

### G2. Wall time per forward evaluation (s) by step; ad_no_opt includes the AD build, nonlinear_opt runs up to 50 Picard iterations

| step | tile | ad_no_opt | ad_opt | acrit_opt | nonlinear_opt | joint_opt |
|---|---|---|---|---|---|---|
| step2 | AP1 | 169 | 171 | 12 | 353 | 13 |
| step2 | AP2 | 169 | 168 | 11 | 334 | 13 |
| step2 | AP3 | 170 | 167 | 11 | 360 | 13 |
| step2 | AP4 | 170 | 170 | 10 | 377 | 14 |
| step2 | AP5 | 167 | 169 | 10 | 311 | 11 |
| step2 | GM1 | 163 | 164 | 10 | 51 | 12 |
| step2 | GM2 | 162 | 163 | 8 | 34 | 11 |
| step2 | GM3 | 161 | 162 | 10 | 36 | 11 |
| step2 | GM4 | 161 | 162 | 10 | 57 | 12 |
| step2 | GM5 | 166 | 167 | 9 | 43 | 11 |
| step3 | AP1 | 26 | 26 | 12 | 324 | 13 |
| step3 | AP2 | 28 | 28 | 11 | 351 | 13 |
| step3 | AP3 | 23 | 23 | 10 | 297 | 12 |
| step3 | AP4 | 26 | 26 | 13 | 382 | 16 |
| step3 | AP5 | 25 | 24 | 13 | 322 | 14 |
| step3 | GM1 | 24 | 24 | 12 | 54 | 14 |
| step3 | GM2 | 25 | 25 | 11 | 39 | 14 |
| step4 | AP1 | 26 | 26 | 12 | 330 | 12 |
| step4 | AP2 | 27 | 27 | 11 | 339 | 13 |
| step4 | AP3 | 28 | 28 | 11 | 379 | 13 |
| step4 | AP4 | 28 | 28 | 10 | 391 | 14 |
| step4 | AP5 | 26 | 26 | 10 | 306 | 11 |
| step4 | GM1 | 26 | 26 | 10 | 50 | 11 |
| step4 | GM2 | 27 | 27 | 8 | 34 | 12 |
| step4 | GM3 | 25 | 25 | 10 | 36 | 11 |
| step4 | GM4 | 25 | 24 | 10 | 52 | 11 |
| step4 | GM5 | 19 | 20 | 8 | 34 | 9 |
| step5 | AP1 | 26 | 25 | 6 | 316 | 6 |
| step5 | AP2 | 28 | 27 | 6 | 333 | 7 |
| step5 | AP3 | 29 | 28 | 5 | 371 | 8 |
| step5 | AP4 | 28 | 27 | 5 | 377 | 8 |
| step5 | AP5 | 25 | 24 | 5 | 294 | 5 |
| step5 | GM1 | 25 | 24 | 5 | 43 | 5 |
| step5 | GM2 | 21 | 20 | 3 | 23 | 4 |
| step5 | GM3 | 20 | 20 | 4 | 25 | 4 |
| step5 | GM4 | 24 | 24 | 5 | 49 | 6 |
| step5 | GM5 | 24 | 24 | 4 | 39 | 5 |

### O1. Optimizer stages: section 3 baseline (Phase 2 code) and Phase 3 code, against v1 (parameters within 1 %, MSE within 0.5 %)

| tile | stage | code | run | params | rel. diff vs v1 | MSE | rel. diff vs v1 | wall (s) | v1 wall (s) | fev / solves | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AP5 | acrit | Phase 3 | 20261109T214309Z_tile_run_AP5_acrit | A_crit=1130.04 | A_crit +3.9e-05 | 359.26 | +1.8e-06 | 136 | 1323 | 34 / 29 | yes |
| AP5 | acrit | baseline | 20261109T211901Z_tile_run_AP5_acrit | A_crit=1130.04 | A_crit +3.9e-05 | 359.26 | +1.8e-06 | 119 | 1323 | 34 / 29 | yes |
| GM1 | acrit | Phase 3 | 20261109T214306Z_tile_run_GM1_acrit | A_crit=423.493 | A_crit +8.1e-06 | 100.96 | +2.4e-05 | 122 | 1480 | 35 / 32 | yes |
| GM1 | acrit | baseline | 20261109T211103Z_tile_run_GM1 | A_crit=423.493 | A_crit +8.1e-06 | 100.96 | +2.4e-05 | 853 | 1480 | 35 / 32 | yes |


---

# Appendix 3. Phase 2 regression tables, sections 2.1-2.2 (02_regression_tables.md)

## Phase 2 regression tables (generated by `lemcvt regression-report`)

### 2.1 Forward regression against the saved v1 grids

| tile | key | exact_params | max_dz_over_relief | rel_mse_diff | pass_dz_1e-6 | pass_mse_1e-8 | n_dz_gt_1e-9_relief | n_dz_gt_1e-6_relief | n_mask_may_differ | wall_s_run |
|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | ad_no_opt | True | 8.553e-13 | 2.709e-13 | True | True | 0.000e+00 | 0.000e+00 |  | 3.836e+02 |
| AP1 | ad_opt | False | 1.867e-06 | 1.664e-10 | False | True | 1.026e+06 | 2.315e+05 |  | 3.836e+02 |
| AP1 | acrit_opt | False | 6.200e-13 | 1.105e-12 | True | True | 0.000e+00 | 0.000e+00 | 0.000e+00 | 3.836e+02 |
| AP1 | nonlinear_opt | False |  |  |  |  |  |  |  | 3.836e+02 |
| AP1 | joint_opt | False | 7.106e-06 | 2.678e-09 | False | True | 1.025e+06 | 5.365e+05 | 0.000e+00 | 3.836e+02 |
| AP2 | ad_no_opt | True | 5.605e-13 | 1.143e-13 | True | True | 0.000e+00 | 0.000e+00 |  | 4.195e+02 |
| AP2 | ad_opt | False | 4.749e-07 | 9.446e-12 | True | True | 1.018e+06 | 0.000e+00 |  | 4.195e+02 |
| AP2 | acrit_opt | False | 6.734e-13 | 2.124e-12 | True | True | 0.000e+00 | 0.000e+00 | 0.000e+00 | 4.195e+02 |
| AP2 | nonlinear_opt | False |  |  |  |  |  |  |  | 4.195e+02 |
| AP2 | joint_opt | False | 1.753e-06 | 3.294e-09 | False | True | 1.022e+06 | 2.841e+04 | 0.000e+00 | 4.195e+02 |
| AP3 | ad_no_opt | True | 5.656e-13 | 2.080e-13 | True | True | 0.000e+00 | 0.000e+00 |  | 3.745e+02 |
| AP3 | ad_opt | False | 6.366e-06 | 1.732e-09 | False | True | 1.025e+06 | 8.208e+05 |  | 3.745e+02 |
| AP3 | acrit_opt | False | 6.445e-13 | 2.695e-13 | True | True | 0.000e+00 | 0.000e+00 | 0.000e+00 | 3.745e+02 |
| AP3 | nonlinear_opt | False |  |  |  |  |  |  |  | 3.745e+02 |
| AP3 | joint_opt | False | 2.165e-06 | 4.265e-09 | False | True | 1.025e+06 | 4.355e+05 | 0.000e+00 | 3.745e+02 |
| AP4 | ad_no_opt | True | 1.897e-12 | 7.262e-13 | True | True | 0.000e+00 | 0.000e+00 |  | 4.775e+02 |
| AP4 | ad_opt | False | 2.020e-06 | 1.650e-10 | False | True | 1.026e+06 | 2.484e+05 |  | 4.775e+02 |
| AP4 | acrit_opt | False | 1.980e-12 | 5.674e-12 | True | True | 0.000e+00 | 0.000e+00 | 0.000e+00 | 4.775e+02 |
| AP4 | nonlinear_opt | False |  |  |  |  |  |  |  | 4.775e+02 |
| AP4 | joint_opt | False | 2.201e-06 | 6.142e-10 | False | True | 1.024e+06 | 4.673e+05 | 0.000e+00 | 4.775e+02 |
| AP5 | ad_no_opt | True | 1.350e-12 | 4.828e-13 | True | True | 0.000e+00 | 0.000e+00 |  | 3.565e+02 |
| AP5 | ad_opt | False | 3.121e-05 | 1.626e-08 | False | False | 1.026e+06 | 9.980e+05 |  | 3.565e+02 |
| AP5 | acrit_opt | False | 1.129e-12 | 1.674e-12 | True | True | 0.000e+00 | 0.000e+00 | 0.000e+00 | 3.565e+02 |
| AP5 | nonlinear_opt | False |  |  |  |  |  |  |  | 3.565e+02 |
| AP5 | joint_opt | False | 3.213e-06 | 7.753e-09 | False | True | 1.026e+06 | 5.094e+05 | 0.000e+00 | 3.565e+02 |
| GM1 | ad_no_opt | True | 9.271e-13 | 1.580e-13 | True | True | 0.000e+00 | 0.000e+00 |  | 1.065e+02 |
| GM1 | ad_opt | False | 1.986e-05 | 8.074e-09 | False | True | 9.960e+05 | 9.072e+05 |  | 1.065e+02 |
| GM1 | acrit_opt | False | 4.126e-13 | 2.317e-12 | True | True | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.065e+02 |
| GM1 | nonlinear_opt | False |  |  |  |  |  |  |  | 1.065e+02 |
| GM1 | joint_opt | False | 7.593e-06 | 1.246e-08 | False | False | 9.950e+05 | 4.838e+05 | 0.000e+00 | 1.065e+02 |
| GM2 | ad_no_opt | True | 1.715e-12 | 4.392e-13 | True | True | 0.000e+00 | 0.000e+00 |  | 7.718e+01 |
| GM2 | ad_opt | False | 2.593e-05 | 1.971e-08 | False | False | 9.960e+05 | 9.695e+05 |  | 7.718e+01 |
| GM2 | acrit_opt | False | 2.116e-13 | 1.099e-13 | True | True | 0.000e+00 | 0.000e+00 | 1.028e+03 | 7.718e+01 |
| GM2 | nonlinear_opt | False |  |  |  |  |  |  |  | 7.718e+01 |
| GM2 | joint_opt | False | 1.651e-06 | 1.506e-09 | False | True | 9.923e+05 | 1.004e+04 | 0.000e+00 | 7.718e+01 |
| GM3 | ad_no_opt | True | 1.399e-12 | 4.593e-13 | True | True | 0.000e+00 | 0.000e+00 |  | 7.656e+01 |
| GM3 | ad_opt | False | 2.291e-05 | 1.061e-08 | False | False | 9.960e+05 | 9.457e+05 |  | 7.656e+01 |
| GM3 | acrit_opt | False | 5.242e-13 | 1.709e-12 | True | True | 0.000e+00 | 0.000e+00 | 0.000e+00 | 7.656e+01 |
| GM3 | nonlinear_opt | False |  |  |  |  |  |  |  | 7.656e+01 |
| GM3 | joint_opt | False | 1.718e-05 | 1.781e-09 | False | True | 9.960e+05 | 9.124e+05 | 0.000e+00 | 7.656e+01 |
| GM4 | ad_no_opt | True | 1.933e-12 | 3.560e-13 | True | True | 0.000e+00 | 0.000e+00 |  | 1.097e+02 |
| GM4 | ad_opt | False | 2.452e-06 | 9.682e-11 | False | True | 9.956e+05 | 3.158e+05 |  | 1.097e+02 |
| GM4 | acrit_opt | False | 9.007e-14 | 2.665e-13 | True | True | 0.000e+00 | 0.000e+00 | 0.000e+00 | 1.097e+02 |
| GM4 | nonlinear_opt | False |  |  |  |  |  |  |  | 1.097e+02 |
| GM4 | joint_opt | False | 9.153e-06 | 1.594e-08 | False | False | 9.960e+05 | 8.674e+05 | 0.000e+00 | 1.097e+02 |
| GM5 | ad_no_opt | True | 7.939e-13 | 1.377e-13 | True | True | 0.000e+00 | 0.000e+00 |  | 9.917e+01 |
| GM5 | ad_opt | False | 3.434e-05 | 4.926e-08 | False | False | 9.960e+05 | 9.785e+05 |  | 9.917e+01 |
| GM5 | acrit_opt | False | 3.721e-13 | 1.158e-12 | True | True | 0.000e+00 | 0.000e+00 | 0.000e+00 | 9.917e+01 |
| GM5 | nonlinear_opt | False |  |  |  |  |  |  |  | 9.917e+01 |
| GM5 | joint_opt | False | 9.145e-06 | 6.025e-06 | False | False | 9.960e+05 | 8.568e+05 | 0.000e+00 | 9.917e+01 |

Counts: cells with |dz| > 1e-9 x relief and > 1e-6 x relief; `n_mask_may_differ` = cells whose D8 area lies in the rounding interval of the printed A_crit (the only cells whose channel membership can differ from the saved grid; v1 saved no mask). Blank where the run has no saved-grid comparison.

### 2.1b Linearized recovery of the printed-parameter rounding (closeout runs)

| tile | key | delta_log10_K | box_K_lo | box_K_hi | delta_log10_D | box_D_lo | box_D_hi | in_box_K | in_box_D | max_dz_before_over_relief | max_resid_over_relief | pass | dmse_rel_measured | dmse_measured | dmse_predicted | grad_mse_log10K | grad_mse_log10D | run |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | ad_opt | 3.547e-07 | -2.623e-06 | 2.623e-06 | -8.019e-07 | -3.987e-06 | 3.987e-06 | True | True | 1.867e-06 | 1.941e-12 | True | -1.664e-10 | -6.206e-08 | -6.199e-08 | -3.174e-01 | 4.907e-02 | 20261109T204300Z_closeout_AP1 |
| AP1 | joint_opt | 3.970e-06 | -5.338e-06 | 5.338e-06 | -2.440e-06 | -4.036e-06 | 4.036e-06 | True | True | 7.106e-06 | 5.953e-12 | True | -2.678e-09 | -6.052e-07 | -6.091e-07 | -3.191e-01 | -8.424e-02 | 20261109T204300Z_closeout_AP1 |
| AP2 | ad_opt | 2.015e-07 | -3.120e-06 | 3.120e-06 | 1.685e-06 | -3.260e-06 | 3.260e-06 | True | True | 4.749e-07 | 8.584e-13 | True | -9.438e-12 | -3.601e-09 | -4.101e-09 | 4.330e-02 | -1.094e-02 | 20261109T204302Z_closeout_AP2 |
| AP2 | joint_opt | 2.222e-06 | -6.192e-06 | 6.191e-06 | -1.670e-06 | -3.065e-06 | 3.065e-06 | True | True | 1.753e-06 | 1.730e-12 | True | 3.294e-09 | 7.692e-07 | 7.709e-07 | 3.063e-01 | -3.209e-02 | 20261109T204302Z_closeout_AP2 |
| AP3 | ad_opt | -2.810e-06 | -1.948e-05 | 1.947e-05 | 9.191e-08 | -3.268e-06 | 3.268e-06 | True | True | 6.366e-06 | 1.931e-11 | True | -1.732e-09 | -2.144e-06 | -2.151e-06 | 1.303e+00 | -3.859e-01 | 20261109T204302Z_closeout_AP3 |
| AP3 | joint_opt | 2.094e-06 | -3.439e-06 | 3.439e-06 | 1.498e-06 | -3.145e-06 | 3.145e-06 | True | True | 2.165e-06 | 9.769e-13 | True | 4.265e-09 | 2.155e-06 | 2.156e-06 | -1.003e+00 | 2.717e+00 | 20261109T204302Z_closeout_AP3 |
| AP4 | ad_opt | 1.263e-06 | -1.327e-05 | 1.327e-05 | 2.433e-06 | -1.024e-05 | 1.024e-05 | True | True | 2.020e-06 | 1.185e-12 | True | -1.650e-10 | -9.542e-08 | -9.470e-08 | -3.149e-01 | 7.711e-02 | 20261109T204301Z_closeout_AP4 |
| AP4 | joint_opt | -4.557e-07 | -1.066e-05 | 1.066e-05 | 2.877e-06 | -3.559e-06 | 3.559e-06 | True | True | 2.201e-06 | 7.640e-12 | True | 6.143e-10 | 2.273e-07 | 2.346e-07 | -5.199e-01 | -7.618e-02 | 20261109T204301Z_closeout_AP4 |
| AP5 | ad_opt | 8.946e-06 | -1.391e-05 | 1.391e-05 | 7.237e-07 | -5.031e-06 | 5.031e-06 | True | True | 3.121e-05 | 2.059e-10 | True | -1.626e-08 | -1.634e-05 | -1.632e-05 | -3.736e+00 | 8.855e-01 | 20261109T204301Z_closeout_AP5 |
| AP5 | joint_opt | -9.774e-07 | -6.312e-06 | 6.312e-06 | -2.118e-06 | -5.984e-06 | 5.983e-06 | True | True | 3.213e-06 | 3.566e-12 | True | -7.752e-09 | -2.245e-06 | -2.242e-06 | 7.096e-01 | 8.135e-01 | 20261109T204301Z_closeout_AP5 |
| GM1 | ad_opt | -4.511e-06 | -4.868e-06 | 4.868e-06 | 5.762e-06 | -7.528e-06 | 7.528e-06 | True | True | 1.986e-05 | 1.619e-10 | True | -8.074e-09 | -1.362e-06 | -1.362e-06 | 4.232e-01 | -1.545e-01 | 20261109T204301Z_closeout_GM1 |
| GM1 | joint_opt | -4.541e-06 | -1.003e-05 | 1.003e-05 | 6.550e-06 | -7.655e-06 | 7.655e-06 | True | True | 7.593e-06 | 4.127e-12 | True | 1.246e-08 | 4.220e-07 | 4.226e-07 | -4.429e-02 | 2.502e-02 | 20261109T204301Z_closeout_GM1 |
| GM2 | ad_opt | 8.895e-06 | -1.656e-05 | 1.656e-05 | -8.846e-07 | -2.617e-06 | 2.617e-06 | True | True | 2.593e-05 | 1.347e-10 | True | -1.971e-08 | -1.615e-06 | -1.612e-06 | -3.566e-01 | 7.065e-02 | 20261109T204304Z_closeout_GM2 |
| GM2 | joint_opt | 9.766e-07 | -3.879e-06 | 3.879e-06 | -1.375e-06 | -8.221e-06 | 8.221e-06 | True | True | 1.651e-06 | 7.027e-13 | True | -1.506e-09 | -4.192e-08 | -4.192e-08 | 1.634e-01 | 1.480e-01 | 20261109T204304Z_closeout_GM2 |
| GM3 | ad_opt | -5.593e-06 | -1.233e-05 | 1.233e-05 | 3.161e-06 | -9.264e-06 | 9.263e-06 | True | True | 2.291e-05 | 8.451e-11 | True | -1.061e-08 | -8.082e-07 | -8.067e-07 | 2.466e-01 | -7.209e-02 | 20261109T204304Z_closeout_GM3 |
| GM3 | joint_opt | -8.986e-06 | -1.135e-05 | 1.135e-05 | -3.511e-06 | -4.790e-06 | 4.790e-06 | True | True | 1.718e-05 | 1.704e-12 | True | -1.781e-09 | -7.660e-08 | -7.649e-08 | 4.126e-02 | 3.256e-02 | 20261109T204304Z_closeout_GM3 |
| GM4 | ad_opt | 2.975e-07 | -2.295e-06 | 2.295e-06 | -1.687e-06 | -3.201e-06 | 3.201e-06 | True | True | 2.452e-06 | 3.065e-12 | True | -9.682e-11 | -1.698e-08 | -1.688e-08 | -5.201e-02 | 1.094e-02 | 20261109T204300Z_closeout_GM4 |
| GM4 | joint_opt | 2.364e-06 | -6.184e-06 | 6.184e-06 | 6.972e-06 | -8.179e-06 | 8.179e-06 | True | True | 9.153e-06 | 2.040e-12 | True | 1.594e-08 | 7.606e-07 | 7.605e-07 | 1.695e-01 | -5.780e-04 | 20261109T204300Z_closeout_GM4 |
| GM5 | ad_opt | 1.023e-05 | -1.387e-05 | 1.387e-05 | -4.145e-06 | -1.167e-05 | 1.167e-05 | True | True | 3.434e-05 | 3.195e-10 | True | -4.926e-08 | -4.590e-06 | -4.583e-06 | -8.075e-01 | 2.148e-01 | 20261109T204300Z_closeout_GM5 |
| GM5 | joint_opt | 2.387e-06 | -3.729e-06 | 3.729e-06 | 1.007e-05 | -1.071e-05 | 1.071e-05 | True | True | 9.145e-06 | 7.705e-12 | True | -6.025e-06 | -1.992e-04 | -1.992e-04 | -7.626e+01 | -1.733e+00 | 20261109T204300Z_closeout_GM5 |

### 2.1c Cells with D8 area inside the rounding interval of the printed A_crit

| tile | key | A_crit_printed | interval_lo | interval_hi | n_cells_in_interval | n_channel | nearest_below | nearest_above | cell_area | run |
|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | acrit_opt | 3063.4 | 3063.35 | 3063.45 | 0 | 8095 | 3063.06 | 3064.97 | 0.950672 | 20261109T204300Z_closeout_AP1 |
| AP1 | nonlinear_opt | 4950.3 | 4950.25 | 4950.35 | 0 | 5701 | 4949.2 | 4952.05 | 0.950672 | 20261109T204300Z_closeout_AP1 |
| AP1 | joint_opt | 6414.3 | 6414.25 | 6414.35 | 0 | 5003 | 6379.01 | 6427.49 | 0.950672 | 20261109T204300Z_closeout_AP1 |
| AP2 | acrit_opt | 2337.8 | 2337.75 | 2337.85 | 0 | 14043 | 2337.7 | 2338.65 | 0.950672 | 20261109T204302Z_closeout_AP2 |
| AP2 | nonlinear_opt | 6165.9 | 6165.85 | 6165.95 | 0 | 7166 | 6159.4 | 6170.81 | 0.950672 | 20261109T204302Z_closeout_AP2 |
| AP2 | joint_opt | 7645.6 | 7645.55 | 7645.65 | 0 | 6178 | 7645.3 | 7646.25 | 0.950672 | 20261109T204302Z_closeout_AP2 |
| AP3 | acrit_opt | 1722.1 | 1722.05 | 1722.15 | 0 | 15419 | 1721.67 | 1722.62 | 0.950672 | 20261109T204302Z_closeout_AP3 |
| AP3 | nonlinear_opt | 10233 | 10232.5 | 10233.5 | 0 | 4183 | 10225.4 | 10240.6 | 0.950672 | 20261109T204302Z_closeout_AP3 |
| AP3 | joint_opt | 9741.5 | 9741.45 | 9741.55 | 0 | 4300 | 9739.63 | 9742.48 | 0.950672 | 20261109T204302Z_closeout_AP3 |
| AP4 | acrit_opt | 995.01 | 995.005 | 995.015 | 0 | 23274 | 994.403 | 995.353 | 0.950672 | 20261109T204301Z_closeout_AP4 |
| AP4 | nonlinear_opt | 5263.5 | 5263.45 | 5263.55 | 0 | 5460 | 5262.92 | 5263.87 | 0.950672 | 20261109T204301Z_closeout_AP4 |
| AP4 | joint_opt | 7797.9 | 7797.85 | 7797.95 | 0 | 4055 | 7797.41 | 7799.31 | 0.950672 | 20261109T204301Z_closeout_AP4 |
| AP5 | acrit_opt | 1130 | 1129.95 | 1130.05 | 0 | 20433 | 1129.4 | 1130.35 | 0.950672 | 20261109T204301Z_closeout_AP5 |
| AP5 | nonlinear_opt | 3399.1 | 3399.05 | 3399.15 | 0 | 8118 | 3398.65 | 3400.55 | 0.950672 | 20261109T204301Z_closeout_AP5 |
| AP5 | joint_opt | 1887 | 1886.95 | 1887.05 | 0 | 13046 | 1886.13 | 1887.08 | 0.950672 | 20261109T204301Z_closeout_AP5 |
| GM1 | acrit_opt | 423.49 | 423.485 | 423.495 | 0 | 19721 | 423 | 424 | 1 | 20261109T204301Z_closeout_GM1 |
| GM1 | nonlinear_opt | 474.21 | 474.205 | 474.215 | 0 | 17432 | 474 | 475 | 1 | 20261109T204301Z_closeout_GM1 |
| GM1 | joint_opt | 8092.4 | 8092.35 | 8092.45 | 0 | 4953 | 8082 | 8132 | 1 | 20261109T204301Z_closeout_GM1 |
| GM2 | acrit_opt | 100 | 99.995 | 100.005 | 1028 | 109511 | 99 | 101 | 1 | 20261109T204304Z_closeout_GM2 |
| GM2 | nonlinear_opt | 244.59 | 244.585 | 244.595 | 0 | 45533 | 244 | 245 | 1 | 20261109T204304Z_closeout_GM2 |
| GM2 | joint_opt | 2960 | 2959.95 | 2960.05 | 0 | 6782 | 2957 | 2965 | 1 | 20261109T204304Z_closeout_GM2 |
| GM3 | acrit_opt | 1297.4 | 1297.35 | 1297.45 | 0 | 10762 | 1297 | 1298 | 1 | 20261109T204304Z_closeout_GM3 |
| GM3 | nonlinear_opt | 1297.4 | 1297.35 | 1297.45 | 0 | 10762 | 1297 | 1298 | 1 | 20261109T204304Z_closeout_GM3 |
| GM3 | joint_opt | 6526.3 | 6526.25 | 6526.35 | 0 | 5260 | 6522 | 6532 | 1 | 20261109T204304Z_closeout_GM3 |
| GM4 | acrit_opt | 422.09 | 422.085 | 422.095 | 0 | 25462 | 422 | 423 | 1 | 20261109T204300Z_closeout_GM4 |
| GM4 | nonlinear_opt | 521.38 | 521.375 | 521.385 | 0 | 20528 | 521 | 522 | 1 | 20261109T204300Z_closeout_GM4 |
| GM4 | joint_opt | 7396.2 | 7396.15 | 7396.25 | 0 | 5152 | 7395 | 7400 | 1 | 20261109T204300Z_closeout_GM4 |
| GM5 | acrit_opt | 261.42 | 261.415 | 261.425 | 0 | 41774 | 261 | 262 | 1 | 20261109T204300Z_closeout_GM5 |
| GM5 | nonlinear_opt | 299.62 | 299.615 | 299.625 | 0 | 36867 | 299 | 300 | 1 | 20261109T204300Z_closeout_GM5 |
| GM5 | joint_opt | 2247.1 | 2247.05 | 2247.15 | 0 | 9939 | 2247 | 2248 | 1 | 20261109T204300Z_closeout_GM5 |

### 2.2 Optimizer regression (tile_run against printed v1 optima)

| tile | stage | quantity | v1 | v2 | rel_diff | pass_1pct | wall_s_v2 | wall_s_v1 | run | pass_0.5pct |
|---|---|---|---|---|---|---|---|---|---|---|
| AP5 | acrit | A_crit | 1130 | 1130 | 3.85e-05 | True | 135.6 | 1323 | 20261109T214309Z_tile_run_AP5_acrit |  |
| AP5 | acrit | MSE | 359.3 | 359.3 | 1.785e-06 |  | 135.6 | 1323 | 20261109T214309Z_tile_run_AP5_acrit | True |
| GM1 | acrit | A_crit | 423.5 | 423.5 | 8.076e-06 | True | 122.1 | 1480 | 20261109T214306Z_tile_run_GM1_acrit |  |
| GM1 | acrit | MSE | 101 | 101 | 2.353e-05 |  | 122.1 | 1480 | 20261109T214306Z_tile_run_GM1_acrit | True |

