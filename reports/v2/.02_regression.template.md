# Phase 2 regression gate: does v2 reproduce v1?

Directive: `reports/directives/refactor_v2_directive.md` Phase 2 (with addendum 1).
Branch `v2`; code at the commits listed in each run's `run.json`. Environment: `lemcvt`
(Python 3.12, numpy 2.5.4, scipy 1.18.1) built from `envs/sherlock.yml`; drift checks in
`lem-v1` (Python 3.10.12, numpy 1.26.4, scipy 1.15.3). Date: 2026-11-09.

The machine-generated tables behind this report are in `02_regression_tables.md` and the
CSV files `02_forward_regression.csv` and `02_resolution_sweep_regression.csv`, produced by
`lemcvt regression-report` from the run directories under `$GROUP_HOME/lemcvt_results`.

## Summary

| Check | Result |
| --- | --- |
| 2.1 Forward regression, exact parameters (AD-no-opt, ten tiles) | PASS: max dz/relief 1e-12, relative MSE 1e-13 (tolerances 1e-6, 1e-8) |
| 2.1 Forward regression, A_crit-stage grids (rounded A_crit, ten tiles) | PASS at the same level wherever the rounded A_crit selects the same channel mask (see table) |
| 2.1 Forward regression, AD-opt and joint grids (parameters printed to 5 s.f.) | differ by the parameter rounding only: dz/relief 1e-5 to 3e-5, relative MSE 8e-9 to 2e-8; this is the floor set by the printed precision, not by the code |
| 2.2 Optimizer regression (GM1, AP5, four stages, ~21 h each) | NOT RUN: the directive says to ask first; 2 tasks x 4 CPUs x 36 h = 288 core-hours |
| 2.3 fig_slope_area table (AP1, GM2; 24 numbers) | PASS: all within the printed 3 s.f. (max relative difference 1.5 % on an R2 of 0.033 printed as 0.033) |
| 2.3 supp_fig5 synthetic Peclet table (25 rows) | PASS: all within the printed 3 s.f. (max 1.9e-3 relative) |
| 2.3 supp_fig6 / resolution_study_dinf (10 tiles x 25 dx, D8 and D-inf ratios) | 498 of 500 rows within the printed 3 decimals; the 2 others are environment drift (numpy sort-order of ties), reproduced exactly in `lem-v1` |
| 2.3 d8_vs_dinf MSE table (AP1, GM1; 8 MSEs, area ranges) | AP1 PASS to the printed 5 s.f.; GM1 see table |

Decision proposed: the refactor reproduces v1. Phase 3 can start on this code base once the
optimizer regression (2.2) has been run, if Steve wants it run at all given 2.1.

## 2.1 Forward regression against the saved v1 grids

Method. For every tile, `lemcvt tile forward --params v1 --compare` evaluates the models
at the v1 parameters recorded in the tile config (initial K0, D0 exactly; optimized
values as printed by the notebooks, 5 significant figures), with `reproduce_v1_units: true`
on AP (feet, as v1 ran), and compares each grid with the saved v1 output grid of the same
run: max |dz| relative to the tile relief, and the relative difference of the MSE against
the observed grid.

What can be expected. The AD-no-opt run uses exactly reproducible parameters; it is the
bit-level test of the AD solver, the boundary handling and the I/O. The A_crit-only run
uses the exact K0, D0 and a printed A_crit; because the misfit is a staircase in A_crit, the
grid is identical whenever the rounded A_crit selects the same set of channel pixels. The
AD-opt and joint runs use printed K, D (and A_crit); a 5-significant-figure rounding of K
changes the solution at the 1e-5 relative level, so these runs cannot meet 1e-6 x relief
and 1e-8 relative MSE in principle; the attainable check is that the differences are of
the size the rounding implies and nothing larger. The nonlinear optimum was never saved
by v1, so it has no grid to compare with; its MSE is listed against the printed value.

Results (full table: `02_regression_tables.md`, section 2.1):

RESULTS_FORWARD_TABLE

The exact-parameter runs agree to about 1e-10 in elevation (1e-12 of relief) on every
tile, which is the level of floating-point reordering between BLAS/SuperLU builds; the
MSEs agree to 1e-13. This is as close to bit identity as two different numerical
libraries allow.

## 2.2 Optimizer regression

Not run. GM1 and AP5 with all four stages take 12.5 h and 35 h in v1 (nonlinear stage 5 h
and 21 h). Submitting the two-task array costs about 288 core-hours at a 36 h limit on
4 CPUs, above the 200 core-hour line, and the directive asks for approval first. Command
ready to run: `lemcvt slurm submit tile_run --tiles GM1,AP5 --time 1-12:00:00 --approved`.

My recommendation: run it, because it is the only check of the optimizer paths (L-BFGS-B
and Nelder-Mead call sequences, lru-cache behaviour, Picard loop) on real data, and
Phase 3 will need the same baseline to prove its speedups do not move the optima.

## 2.3 Figure-notebook regressions

### fig_slope_area (AP1, GM2)

All 24 printed numbers (Ks, theta, R2 for the observed grid and three model grids per
tile) are reproduced to the printed precision; relative differences are below 1.2e-3
except R2 = 0.033 for AP1 AD-no-opt (0.0325, a rounding of 1.5 %). Table in
`02_regression_tables.md` section 2.3b.

### supp_fig5 (synthetic Peclet)

The local-k_s summary table (25 pixel sizes, median and IQR) matches v1 to the printed
3 significant figures (worst relative difference 1.9e-3). The baseline is identical
(D_bal = 4.402e-2, A_inlet = 2.59e8, Pe = 100). Section 2.3d.

### supp_fig6 and resolution_study_dinf

The sweep was rerun for all ten tiles at the v1 parameter tables (`V1_KD`, `V1_THETA`),
with D8 and D-infinity routing. 498 of the 500 (tile, dx) rows agree with the printed
ratios to the printed 3 decimals (maximum absolute difference 5e-4). Two rows differ:

| tile | dx (m) | quantity | v1 printed | v2 (numpy 2.5) | v2 code in lem-v1 (numpy 1.26) | points |
| --- | --- | --- | --- | --- | --- | --- |
| AP2 | 88.27 | Ks_obs | 147.4 | 142.2 | 147.4 | 5 |
| AP5 | 60.70 | Ks_obs | 13.65 | 13.2 | 13.65 | 15 |

The same lemcvt code run with the v1 numpy reproduces the printed values exactly, so the
difference is environment drift, not a refactor error. Its mechanism: the channel-slope
routine assigns each cell's upstream "main stem" by visiting cells in ascending drainage
area with `argsort`; at coarse pixel sizes many cells have equal area, and numpy 2.x sorts
ties in a different order than numpy 1.26, which changes a few smoothing paths. Only the
observed Ks at the two coarsest, point-poorest rows are affected (Ks_mod is identical);
the median-IQR curves of the figure do not change visibly. Size of the drift: 3.5 % and
3.3 % in those two ratios. I propose to accept it and note it; Steve's agreement is
required by the directive before continuing on that basis.

### d8_vs_dinf_comparison (AP1, GM1)

RESULTS_D8DINF

### Visual comparison of regenerated figures

v1 figures extracted from the executed notebooks are in `reports/v2/figures_v1/`;
the v2 figures built from results with `lemcvt figures` are in `reports/v2/figures_v2/`.

| Figure | v1 image | v2 image | Note |
| --- | --- | --- | --- |
| fig_slope_area | `figures_v1/fig_slope_area_cell01_01.png` | `figures_v2/fig_slope_area.png` | same data, same layout (3 x 2); v2 adds fit annotations and the result id |
| supp_fig5 | `figures_v1/supp_fig5_cell01_01.png`, `..._cell02_02.png`, `..._cell03_28.png` | `figures_v2/supp_fig5_synthetic_pe.png` | v1 drew the three estimators as separate figures; v2 puts them in one 1 x 3 panel |
| supp_fig6 | `figures_v1/supp_fig6_cell02_01.png` | `figures_v2/supp_fig6_resolution_ratio.png` | same curves (median + IQR by site) |
| resolution_study_dinf | `figures_v1/resolution_study_dinf_cell08_01.png` | `figures_v2/resolution_study_dinf.png` | same |
| d8_vs_dinf | `figures_v1/d8_vs_dinf_comparison_cell07_*.png`, `cell09_*`, `cell11_*` | `figures_v2/d8_vs_dinf_*` | elevation panels, area, slope-area per tile |

## Environment drift

Two independent environments were used: `lemcvt` (the v2 environment) and `lem-v1`
(Phase 0 rebuild of v1). Every disagreement with v1 found in Phase 2 was attributable:
rounding of printed parameters (2.1), or numpy tie-ordering (2.3). No disagreement
required `lem-v1` to explain a model or optimizer result.

## Wall times

RESULTS_WALLTIMES

Phase 1 kept the v1 algorithms, so the stage run times of v1 (`00_reference.md` 3.4) are
the expectation for the optimizer regression; Phase 3 changes this.

## Measurement pipeline: first results (addendum 1)

Separate note: `02b_measurement_first_results.md`. These results are not part of the
regression gate (the pipeline is new), but Phase 6 depends on decisions they raise.

## Stop

Phase 2 ends with a review. Open items for Steve: (1) accept the two drift rows; (2) run
or waive the optimizer regression (2.2); (3) the measurement-threshold decision in
`02b_measurement_first_results.md`; (4) still pending from Phase 0: push approval and
Slurm partition/QoS.
