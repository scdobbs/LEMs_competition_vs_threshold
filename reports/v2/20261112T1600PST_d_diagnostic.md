# [2026-11-12 16:00 PST] D diagnostic by hillslope class (addendum 7): Phase 6 decisions closed, the slope-area units bug, and the pre-registered test of the hollow-averaging hypothesis

Directive: `reports/directives/2026-11-12T1100PST_refactor_v2_addendum_7.md` (outline fixed by its section 3, with the
"Slope-area offset" section of 2b after section 2). Branch `v2`, base 12c77f3 (the Phase 6 results report); code
commit 167d3f7 (2026-11-12 12:01 PST) and the commit carrying this report. Environment `lemcvt` (Python 3.12,
numpy 2.5.4, scipy 1.18.1); every job on `serc`, 1 CPU unless stated. Machine-generated tables: section 8 of this file
(appendices C1-C7, `reports/v2/scripts/make_d_diagnostic_tables.py`); figures under `reports/v2/figures_v2/`
(git-ignored, referenced by path). Every number is tied to a run id under `$GROUP_HOME/lemcvt_results` or a Slurm
job id. The work spans two agent sessions: the first wrote and tested the code and submitted the jobs (11:50 PST); the
second (from 15:50 PST) found every job completed and wrote this report. **This is a stop for review** (section 7).

## Summary

| Item | Result |
| --- | --- |
| 0.4 submit guard | `lemcvt slurm submit` refuses to render when `LEMCVT_DATA` or `LEMCVT_RESULTS` is unset or empty (`require_submit_env`, tested) |
| 1 decisions | 1.1 D_NL/D_lin = 0.91-1.05 on every tile (within one grid step, x1.2), no refinement run; 1.2 GM1 K sensitivity reported as is; 1.3 `fig3_deltas` rebuilt from the per-stage Phase 6 runs with the AP D0 bracket; 1.4 CI untouched; 1.5 Phase 7 waits |
| 2b slope-area offset | **A bug, confirmed on all five AP tiles and fixed.** The figure multiplied the Phase 6 metre grids by 0.3048 a second time: old-path relief 0.28-0.30 of observed, MSE 68 400-71 800 m^2 against the joint MSE of 22-42; the raw grids reproduce the joint MSE exactly and their channel slopes match (U/K) A^-m to 1-2 %. Only `fig_slope_area` and the record's AP1 slope-area rows were affected; the curvature comparison, `d8_vs_dinf`, the resolution sweep and the forward-run checks read grids correctly. Figure and record regenerated |
| 2.1 curvature by class | Hollows (A_roll <= A < A_crit) are 4-33 % of hillslope cells (AP), 20-31 % (GM) and concave on average (mean -lap3 -0.023 to -0.068 1/m); divergent cells convex on average (+0.004 to +0.017); the mean over all hillslope cells is **0.36-0.54 (AP) and 0.21-0.34 (GM) of U/D_fit**, i.e. the fitted surface is 2-5x more convex than the observed hillslopes on average, not less |
| 2.2 refits | Restricting the misfit to divergent + channel cells returns the same A_crit (9 of 10 tiles) and **D lower by only 0.3-3.9 %** (D/D0_A 4.3-8.2 on AP, 1.9-2.8 on GM); hollow + channel cells give D higher by 5-51 % |
| 2.3 rule | **Not supported at either site (0 of 5 tiles each).** (a) fails on 10 of 10 (ratio 0.21-0.54, required 0.7-1.3); (b) holds on 10 of 10; (c) fails on 10 of 10 (log gap to D0_A reduced by 0.002-0.017 decades, required 0.301). Per 2.3, nothing further is proposed |
| cost | 103.9 core-hours of limits submitted (printed estimates 0.7 + 1.0 + 60 + 40 + 1.5 + 0.7, all under 300), 33.8 used; `$GROUP_HOME` at 266 GB of 1 TB |

## 1. State check and push

**State at the start (0.2).** `v2` was at 12c77f3 = `origin/v2` (nothing to push); the tests passed on `serc` at
12c77f3 with the addendum 7 code uncommitted (job 52087672, 3 min) and again at 167d3f7 (job 52088254: ruff clean,
mypy clean in 73 files, **193 passed**, with `LEMCVT_DATA` unset); no job from an earlier session was running (the
accounting window from 11:40 PST holds only the addendum 7 jobs, appendix C7). Where I could check the addendum
against the repo they agree: the rollover range it quotes (18-450 m^2) is the `02b` A_roll 17.8-447 m^2 of the ten
tiles, the A_crit range (860-7900) is the Phase 6 joint 860-7232, and the D gap (1.9-8.3x) is C3's joint D/D0_A.

**Submit guard (0.4).** `lemcvt slurm submit` now calls `require_submit_env` before rendering and refuses with a
message naming the unset variable (`tests/test_addendum7.py::test_render_refuses_without_submit_env`). The
addendum 7 submissions were made from a shell with both exported (`$SCRATCH/lemcvt_jobs/phase7_submit.log`).

**Jobs.** Tests 52087672 and 52088254; slope-area units check 52087673 (1 min); D diagnostic arrays 52087674 (AP,
5 tasks, 12 h limit) and 52087675 (GM, 5 tasks, 8 h limit), 3.1-3.6 h each, 2.8-3.0 GB; post-processing 52087676
(slope-area rerun, figures, record; 3 min, 6.1 GB). Accounting in C7.

**Push.** The commits not on `origin/v2` (167d3f7 and the report commit) passed the addendum 2 section 5 check: no
file under `papers/`, no `*.pdf`, no `working-with-steve.md`, `cry_for_help.txt`, `.env` or `.netrc` tracked; nothing
over 50 MB in the tree; no token or key pattern in the diffs. Pushed over SSH at the end of this session (the push
line is in the final message of the session, not here, because it happens after this file is committed).

## 2. Decisions applied (section 1)

1. **Nonlinear joint D grid, option (a).** D_NL = D_lin to within one grid step (x1.2) on every tile; no refinement
   run. From C3 (nonlinear joint D over linear joint D): AP1 0.97, AP2 0.98, AP3 0.92, AP4 1.01, AP5 0.91, GM1 1.02,
   GM2 1.05, GM3 1.03, GM4 0.91, GM5 1.04.
2. **GM1 K sensitivity, reported as is.** The GM1 nonlinear joint A_crit moves by 32 % (1758 to 1200 m^2) under a
   5 % change of K in either direction, with MSE +4.5-5.6 %; AP5's does not move (A_crit unchanged, D one grid step
   at K x 0.95). No K check was run on the other GM tiles.
3. **`fig3_deltas` rewritten.** `lemcvt figures fig3_deltas` now collects, per tile, the AD optimum (v2 r profile),
   the linear joint optimum (refined partitioned run) and the nonlinear joint optimum (merged run) through the same
   run finder as the curvature comparison, and plots dK = log10(K_fit/K0) and dD = log10(D_fit/D0) per tile in three
   panels. On AP, dD is drawn at both ends of the bracket (D0_A filled, D0_B open, joined by a line); on GM at D0.
   Output `reports/v2/figures_v2/fig3_deltaK_deltaD.{png,pdf,csv}` (job 52087676), values in C3; tested on a
   synthetic GM tile (`test_fig3_collects_per_stage_runs`). What the figure shows: AD has dK = +0.93 to +1.01 (AP),
   +0.33 to +0.49 (GM) and dD = -0.55 to -1.11 (AP), -1.5 to -2.4 (GM); both partitioned fits have |dK| <= 0.045 on
   every tile and dD = +0.63 to +0.92 (AP, D0_A), +0.39 to +0.61 (AP, D0_B), +0.28 to +0.47 (GM).
4. **CI.** Nothing done; the trigger stays `workflow_dispatch` only.
5. **Phase 7** waits for this review.

## Slope-area offset (section 2b)

**2b.1-2b.3, the check** (`lemcvt phase7 slope-area-check`, run `20261112T195138Z_slope_area_check_5tiles`, job
52087673; C1). For each AP tile, the Phase 6 refined joint run's filled model grid was read raw and through the
figure's old path (raw x the tile's stored-to-working factor, 0.3048) and compared with the observed metre grid:

| tile | relief raw / obs | relief old path / obs | MSE raw vs obs (m^2) | joint MSE in run | MSE old path vs obs | median S_raw / law | median S_old / law |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 0.997 | 0.304 | 22.384 | 22.384 | 71 814 | 0.998 | 0.348 |
| AP2 | 1.000 | 0.305 | 22.461 | 22.461 | 68 400 | 0.986 | 0.371 |
| AP3 | 0.995 | 0.303 | 42.122 | 42.122 | 70 290 | 0.988 | 0.338 |
| AP4 | 0.930 | 0.283 | 35.156 | 35.156 | 71 609 | 1.010 | 0.389 |
| AP5 | 0.982 | 0.299 | 27.869 | 27.869 | 70 670 | 0.991 | 0.352 |

The raw grids are in metres (mean elevation 375-385 m against the observed 375-385 m; the old path gives 114-117 m),
their MSE against the observed grid equals the joint MSE of the run to every printed digit, and on channel cells
above A_crit their binned median slopes are within 1-2 % of (U/K) A^-m at the bin centres (per-bin table in C1).
The old-path grids have 0.3 of the relief, an MSE of about 7 x 10^4 m^2, and channel slopes at 0.34-0.39 of the law.
So the factor is a bug, not a model property.

**2b.4, where the factor entered.** `slope_area_fig._model_filled_grid` loaded every model grid and multiplied it by
the tile's stored-to-working factor, on the assumption that model output grids are in the tile's *stored* unit. That
is true of the saved v1 outputs (feet on AP), for which the function was first written, and false for every v2
`tile_run` result, which writes its grids in the run's *working* unit (metres since the AP vertical-units fix). The
Phase 6 figure was the first to read v2 run grids on AP through this path, so their metre grids were converted
ftUS -> m a second time: elevations x 0.3048, slopes x 0.3048, model points 3.28x too low at every area, parallel to
the observed. GM (stored in metres, factor 1) was unaffected.

**2b.5, the fix and the other consumers.** The function now takes each grid from its own unit (the run's
`inputs.z_units_working` from its `params.json`; the stored unit for the saved v1 outputs) to the tile's working
unit (`convert_z`), and the regression test `test_slope_area_model_grids_are_not_converted_twice` builds a synthetic
AP tile in ftUS, runs the slope-area builder on a metre run grid and on the same grid stored in feet with the v1
layout, and requires identical k_s, theta and R^2. Every other consumer of model output grids on AP was traced:

| consumer | how it reads model grids | affected |
| --- | --- | --- |
| curvature comparison (addendum 6 section 4; `curvature_compare._grid`) | reads the run's `*_elevation.tif` raw, no conversion; the runs are in metres | no; it found -lap = U/D_fit exactly on the fitted surfaces, which it could only do on correctly read grids (confirmed again in C4: the model's -lap3 median on hillslope cells equals U/D_fit on every tile) |
| `d8_vs_dinf` | computes the forward grids in memory, reads no saved grid | no |
| resolution sweep | same, no saved grid read | no |
| forward-run checks (`forward.compare_with_saved`, `phase2_closeout`) | multiply only the saved **v1** grids (stored unit) by the factor | no (correct) |
| old-against-new record (`old_vs_new.py`) | takes its slope-area rows from the latest `slope_area` run | **yes**: the nine AP1 slope-area rows (AD no-opt, AD-opt, partitioned k_s, theta, R^2); AP1 is the only AP tile with v1 slope-area values in the record |
| `fig_slope_area` | `_model_filled_grid` | **yes** (the bug) |

**Regenerated.** `lemcvt slope-area --tiles all --phase6` (run `20261112T195138Z_slope_area_10tiles`, job 52087676)
and the figure `reports/v2/figures_v2/fig_slope_area.{png,pdf}`; the record rebuilt by `lemcvt phase6 record`. After
the fix the AP partitioned fits have k_s 9.6-14.0, theta 0.460-0.471 and R^2 0.995-0.999 (pre-fix k_s 2.3-8.4, theta
0.41-0.52), the AD-opt fits k_s 4.7-7.3 and theta 0.38-0.41 (pre-fix 1.0-2.0); every GM fit is unchanged to four
digits (C2). The model theta now equals the site m = 0.47 to within 0.01, as it must for S = (U/K) A^-m. The AP
partitioned points in the figure overlay the observed binned slopes. The Phase 6 report's figure (run
`20261112T175641Z_slope_area_10tiles`) and its B10 AP1 slope-area rows are superseded; the record diff is nine
AP1 rows with new values (AD no-opt k_s 0.089 -> 0.248, AD-opt k_s 2.04 -> 7.27, partitioned k_s 8.43 -> 13.99, with
their theta and R^2) plus the seam label "AP1 seam (negligible by the addendum 6 rule)" on the eleven AP1 model
rows, which the post script passes to the record builder as the addendum 6 verdict. No model or measurement
result changes.

## 3. Curvature by area class (section 2.1)

Run `lemcvt phase7 d-diagnostic --tile <tile>` (runs `20261112T1951*_d_diagnostic_<tile>`, jobs 52087674_0-4 and
52087675_0-4; C4). Hillslope cells are the model's A < A_crit cells at the Phase 6 joint optimum with boundary and
channel cells excluded; divergent A < A_roll (the tile's slope-area rollover from `02b`), hollow A_roll <= A < A_crit,
hilltop the option (iii) set (a subset of divergent). -lap3 is the observed -laplacian with the model's own 5-point
stencil; -lap11 the quadratic-surface estimator at L_f = 11 m. The class map is written as
`<tile>_hillslope_classes.tif` in each run.

| tile | A_roll | A_crit | divergent share | hollow share | hilltop n | mean -lap3: divergent / hollow / hilltop / all (1/m) | U/D_fit | all / (U/D_fit) | concave at 3 cells: div / hollow / all | SSR share div / hollow |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 282 | 6889 | 95.1 % | 4.9 % | 3446 | +0.0047 / -0.066 / +0.096 / +0.0012 | 0.0032 | 0.38 | 48 / 68 / 49 % | 95 / 5 % |
| AP2 | 56 | 3121 | 83.7 % | 16.3 % | 944 | +0.0090 / -0.036 / +0.079 / +0.0018 | 0.0033 | 0.54 | 46 / 61 / 49 % | 86 / 14 % |
| AP3 | 18 | 4742 | 67.3 % | 32.7 % | 533 | +0.0171 / -0.032 / +0.076 / +0.0011 | 0.0031 | 0.36 | 44 / 61 / 50 % | 70 / 30 % |
| AP4 | 447 | 7232 | 96.2 % | 3.8 % | 4711 | +0.0040 / -0.068 / +0.087 / +0.0013 | 0.0028 | 0.46 | 48 / 70 / 49 % | 96 / 4 % |
| AP5 | 56 | 1887 | 85.3 % | 14.7 % | 3575 | +0.0090 / -0.038 / +0.074 / +0.0022 | 0.0044 | 0.49 | 46 / 63 / 49 % | 83 / 18 % |
| GM1 | 36 | 1241 | 73.7 % | 26.3 % | 3583 | +0.0124 / -0.025 / +0.063 / +0.0025 | 0.0075 | 0.33 | 44 / 59 / 48 % | 70 / 30 % |
| GM2 | 36 | 3509 | 71.9 % | 28.1 % | 2530 | +0.0116 / -0.023 / +0.062 / +0.0018 | 0.0054 | 0.34 | 45 / 59 / 49 % | 69 / 31 % |
| GM3 | 22 | 860 | 68.5 % | 31.5 % | 1500 | +0.0139 / -0.025 / +0.064 / +0.0016 | 0.0076 | 0.21 | 44 / 61 / 49 % | 74 / 26 % |
| GM4 | 56 | 3853 | 79.7 % | 20.3 % | 3347 | +0.0091 / -0.028 / +0.069 / +0.0016 | 0.0067 | 0.23 | 46 / 59 / 49 % | 77 / 23 % |
| GM5 | 28 | 1885 | 69.0 % | 31.0 % | 1128 | +0.0136 / -0.023 / +0.064 / +0.0021 | 0.0077 | 0.27 | 44 / 59 / 49 % | 64 / 36 % |

What the table says:

- **The classes behave as the hypothesis describes.** Divergent cells are convex on average and hollows concave on
  average, at both windows (hollow mean -lap11 -0.004 to -0.010 1/m; divergent +0.0008 to +0.004); the hilltop set
  is strongly convex (at 11 m, mean 0.016-0.030 1/m, which is C_ht of the configs to within the window; at 3 cells
  2.5-6x larger, the crest curvature being scale dependent as the Phase 6 report's C_obs(L) already showed).
- **But the hillslope average is far below U/D_fit, not near it.** Over all hillslope cells the mean -lap3 is
  0.36-0.54 (AP) and 0.21-0.34 (GM) of U/D_fit; at 11 m it is 0.06-0.16 (AP) and 0.05-0.17 (GM). The fitted
  partitioned surface (whose -lap3 on hillslope cells is U/D_fit at every cell, median = mean, 0 % concave) is on
  average two to five times more convex than the hillslopes it was fitted to.
- **The 3-cell statistic is noise dominated.** 44-48 % of divergent cells and 48-50 % of all hillslope cells are
  concave at the 3-cell window on these 1 m grids; medians sit far from means (AP1 divergent median 0.0008 against
  mean 0.0047; hollow medians -0.008 to -0.037 against means -0.023 to -0.068). The class means are small residuals
  of large positive and negative values. The 11 m window gives the same ordering with smaller magnitudes.
- **The hollows carry no excess misfit.** Each class's share of the hillslope SSR equals its share of cells to
  within 1-6 points, and the class MSEs are alike (AP1 22.4 divergent, 23.8 hollow; GM1 23.6 and 28.1); the hilltop
  set has the largest MSE on 7 tiles (25-103 m^2) but is 0.1-0.5 % of cells.

## 4. Class-restricted refits (section 2.2)

Three linear joint profiles per tile over the full A_crit range (thin 2000 + refinement; 2006-2134 evaluations each),
with the misfit over (1) divergent + channel cells, (2) hollow + channel cells, (3) all cells, the channel cells
being those of each A_crit so that K stays identified (C5). Each A_crit's reduction is computed once and shared by
the three masks. The "all" mask reproduces the Phase 6 optimum (A_crit, K, D, MSE) to 0.00 % on every tile.

| tile | divergent + channel: A_crit, K/K0, D, D/D0_A (D/D0_B) | D_div / D_fit | hollow + channel: A_crit, D, D/D0_A | D_hollow / D_fit |
| --- | --- | --- | --- | --- |
| AP1 | 6889, 0.90, 1.693e-2, 4.72 (2.43) | 0.997 | 7581, 2.092e-2, 5.83 | 1.23 |
| AP2 | 3121, 0.97, 1.671e-2, 5.19 (4.10) | 0.995 | 7004, 2.176e-2, 6.76 | 1.30 |
| AP3 | 4742, 1.01, 1.762e-2, 8.18 (8.18) | 0.984 | 4782, 1.881e-2, 8.73 | 1.05 |
| AP4 | 7232, 0.98, 1.964e-2, 5.41 (2.54) | 0.995 | 7875, 2.653e-2, 7.31 | 1.34 |
| AP5 | 1899, 1.00, 1.249e-2, 4.27 (3.61) | 0.992 | 1900, 1.408e-2, 4.82 | 1.12 |
| GM1 | 1138, 1.00, 1.872e-2, 2.08 | 0.961 | 6047, 2.939e-2, 3.27 | 1.51 |
| GM2 | 3509, 0.99, 2.690e-2, 2.75 | 0.981 | 4151, 3.273e-2, 3.35 | 1.19 |
| GM3 | 860, 1.06, 1.912e-2, 1.89 | 0.994 | 919, 2.022e-2, 2.00 | 1.05 |
| GM4 | 3853, 1.05, 2.142e-2, 2.17 | 0.978 | 3858, 2.468e-2, 2.50 | 1.13 |
| GM5 | 1885, 1.04, 1.845e-2, 2.43 | 0.970 | 3108, 2.408e-2, 3.17 | 1.27 |

- **Removing the hollows barely moves D.** The divergent-only D is 0.3-1.6 % (AP) and 0.6-3.9 % (GM) below D_fit,
  with the same A_crit on 9 of 10 tiles (GM1 1241 -> 1138, inside its 1 % interval 1134-1267) and K within 1 %. The
  divergent cells alone demand D at 4.3-8.2x D0_A on AP and 1.9-2.8x D0 on GM.
- **The hollows pull in the expected direction, weakly.** Fitting the hollows alone (plus channels) raises D by
  5-51 % and moves A_crit up by a factor 1.6-4.9 on three tiles (AP2, GM1, GM5) and by 10 % on AP1: with few cells
  near the hollow-channel transition, A_crit is weakly identified by that mask (several disjoint 1 % intervals).
  The hollow-only MSE is below the all-cell MSE on 7 tiles.

## 5. Rule applied (section 2.3)

Per site, supported if on at least 4 of 5 tiles all three hold (C6). Condition (c) is read as "the log distance
|log10(D/D0_A)| shrinks by at least log10 2 = 0.301 decades"; the strict reading (distance halved) is reported
beside it and gives the same verdict.

| tile | (a) mean -lap3 / (U/D_fit), need 0.7-1.3 | (b) hollow mean / divergent mean, need < 0.5 | (c) log gap to D0_A: D_fit -> D_div (reduction, need >= 0.301) | all three |
| --- | --- | --- | --- | --- |
| AP1 | 0.38, fail | -14.0, pass | 0.675 -> 0.674 (0.002), fail | no |
| AP2 | 0.54, fail | -3.9, pass | 0.718 -> 0.715 (0.002), fail | no |
| AP3 | 0.36, fail | -1.9, pass | 0.920 -> 0.913 (0.007), fail | no |
| AP4 | 0.46, fail | -16.8, pass | 0.736 -> 0.733 (0.002), fail | no |
| AP5 | 0.49, fail | -4.2, pass | 0.634 -> 0.631 (0.003), fail | no |
| GM1 | 0.33, fail | -2.0, pass | 0.335 -> 0.318 (0.017), fail | no |
| GM2 | 0.34, fail | -2.0, pass | 0.448 -> 0.440 (0.008), fail | no |
| GM3 | 0.21, fail | -1.8, pass | 0.279 -> 0.276 (0.003), fail | no |
| GM4 | 0.23, fail | -3.1, pass | 0.347 -> 0.337 (0.010), fail | no |
| GM5 | 0.27, fail | -1.7, pass | 0.399 -> 0.386 (0.013), fail | no |

**The hypothesis is not supported at either site: 0 of 5 tiles at AP, 0 of 5 at GM.** Condition (a) fails on every
tile (the hillslope-mean curvature is 2-5x below U/D_fit, outside +-30 %); condition (b) holds on every tile; condition
(c) fails on every tile (the gap to D0_A shrinks by 0.002-0.017 decades against the 0.301 required, and by less than
half under the strict reading). On AP the verdict is the same against D0_B (gaps 0.39-0.92 decades, reductions
unchanged). Per section 2.3, nothing further is proposed.

## 6. What the evidence supports, and where it is weak

**Supported.**

1. The two-domain partition does put concave ground in its linear-diffusion domain: 4-33 % (AP) and 20-31 % (GM) of
   the model's hillslope cells lie between the rollover and A_crit and are concave on average at both windows.
2. That ground is not what inflates D. A fit that ignores it returns D within 4 % of D_fit on every tile, so the D gap
   of 4.3-8.3x (AP, D0_A) and 1.9-2.8x (GM) is set by the divergent cells themselves. D_fit is not an average over
   convex divides and concave hollows.
3. D_fit is not "U over the mean observed hillslope curvature" either. The observed hillslope mean -lap3 is 2-5x
   smaller than U/D_fit; a diffusivity read from it (U / mean -lap3) would be 1.9-4.8x D_fit, further from D0, not
   closer. The elevation least squares does not match mean curvature: the elevation misfit is dominated by the
   long-wavelength shape of the hillslopes (their relief at the length set by A_crit), while the mean Laplacian of
   a 1 m lidar surface is a small residual between large positive and negative values at nearly every scale tested.
4. The hilltop set behaves as the measurement assumed: its 11 m curvature reproduces C_ht, and it is 0.1-0.5 % of
   hillslope cells with 0.1-0.8 % of the SSR, so the joint fit is insensitive to the crests that D0_A is measured
   on. The D gap is a disagreement between the crest-scale curvature (D0) and the hillslope-scale relief (D_fit) of
   the same surfaces; the rest of the hillslope, divergent as well as hollow, is less convex than either implies.

**Weak.**

1. Condition (a) as pre-registered tests a noise-dominated statistic: 48-50 % of hillslope cells are concave at the
   3-cell window, and the class means depend on the tails. The 11 m window gives the same verdict with smaller
   ratios (0.05-0.17), so the conclusion does not rest on the window, but the +-30 % band was never going to be
   met by a mean of this kind on a 1 m grid.
2. The class split depends on A_roll from `02b`, which spans 18-447 m^2 across tiles; on AP1 and AP4 (282, 447 m^2)
   the hollow class is 4-5 % of cells, on AP3 (18 m^2) 33 %. Condition (c) is insensitive to this because D barely
   moves whichever class is removed, but the hollow shares in section 3 are only as good as the rollover.
3. The hollow-only refits are weakly identified in A_crit (disjoint 1 % intervals on 7 tiles); their D values are
   indicative of direction, not of magnitude.
4. The class-restricted misfits keep every channel cell of each A_crit, so the channel SSR (a few per cent of cells)
   enters all three masks; K is identified, as intended, at the cost of a small common term in the MSEs.

## 7. Decisions needed

1. **Record the verdict.** The hollow-averaging hypothesis is not supported at either site (section 5). Nothing
   further is proposed under section 2.3; whether the D gap gets another pre-registered test, and which, is for the
   next directive.
2. **The Phase 6 report's slope-area figure and its B10 AP1 slope-area rows were wrong** (section 2b). Earlier reports
   are not renamed or edited under addendum 4 section 1; this report supersedes them. Say whether an erratum line
   should be added to the Phase 6 report anyway.
3. **Phase 7.** It waited for section 2; the section 2 result is in. Go or no-go, and whether its run list changes.
4. **CI** is unchanged (addendum 7, 1.4): the trigger stays manual until the logs of run 42940255348 are read.

## 8. Appendices (generated tables)

Generated by `reports/v2/scripts/make_d_diagnostic_tables.py` at 15:53 PST from the run directories named in each
table; per-bin rows and all digits as written by the runs.

## C1. Section 2b.1-2b.3: the Phase 6 joint grid as the figure read it (old path) and as written (raw)

Run `20261112T195138Z_slope_area_check_5tiles` (job 52087673, git 12c77f3d). Old path = raw grid x the tile's stored-to-working factor (0.3048 on AP), what `_model_filled_grid` did before the fix.

| tile | run grids (unit) | factor | relief obs (m) | relief raw / obs | relief old path / obs | mean obs / raw / old (m) | MSE raw vs obs | joint MSE in run | MSE old path vs obs | MSE new path vs obs | channel bins | median S_raw / law | median S_old / law | raw fit k_s, theta | old-path fit k_s, theta |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | `20261111T050726Z...` (m) | 0.3048 | 114.58 | 0.9970 | 0.3039 | 384.7 / 385.2 / 117.4 | 22.384 | 22.384 | 71814 | 22.384 | 22 | 0.998 | 0.348 | 14.2, 0.472 | 6.97, 0.505 |
| AP2 | `20261111T050734Z...` (m) | 0.3048 | 142.43 | 1.0000 | 0.3048 | 375.2 / 375.5 / 114.5 | 22.461 | 22.461 | 68400 | 22.461 | 21 | 0.986 | 0.371 | 11.1, 0.464 | 1.43, 0.366 |
| AP3 | `20261111T045020Z...` (m) | 0.3048 | 124.74 | 0.9945 | 0.3031 | 380.4 / 381.2 / 116.2 | 42.122 | 42.122 | 70290 | 42.122 | 18 | 0.988 | 0.338 | 9.85, 0.456 | 2.38, 0.422 |
| AP4 | `20261111T050505Z...` (m) | 0.3048 | 125.82 | 0.9299 | 0.2834 | 384.2 / 385.0 / 117.4 | 35.156 | 35.156 | 71609 | 35.156 | 23 | 1.010 | 0.389 | 16, 0.493 | 5.86, 0.491 |
| AP5 | `20261111T050104Z...` (m) | 0.3048 | 93.43 | 0.9822 | 0.2994 | 381.8 / 382.1 / 116.5 | 27.869 | 27.869 | 70670 | 27.869 | 21 | 0.991 | 0.352 | 9.74, 0.462 | 1.79, 0.401 |

Channel bins above A_crit (raw model grid), binned median slope against (U/K) A^-m at the bin centre:

| tile | bin A (m^2) | S binned raw | S binned old path | (U/K) A^-m | raw / law | old / law |
|---|---|---|---|---|---|---|
| AP1 | 7.53e+03 | 0.2025 | 0.0690 | 0.2074 | 0.976 | 0.333 |
| AP1 | 9.54e+03 | 0.2111 | 0.0790 | 0.1856 | 1.137 | 0.425 |
| AP1 | 1.18e+04 | 0.1656 | 0.0603 | 0.1679 | 0.987 | 0.359 |
| AP1 | 1.42e+04 | 0.1524 | 0.0613 | 0.1538 | 0.991 | 0.398 |
| AP1 | 1.75e+04 | 0.1374 | 0.0529 | 0.1396 | 0.984 | 0.379 |
| AP1 | 2.08e+04 | 0.1235 | 0.0457 | 0.1287 | 0.959 | 0.355 |
| AP1 | 2.47e+04 | 0.1144 | 0.0363 | 0.1186 | 0.964 | 0.306 |
| AP1 | 3.21e+04 | 0.1366 | 0.0455 | 0.1049 | 1.302 | 0.434 |
| AP1 | 3.92e+04 | 0.0960 | 0.0373 | 0.0955 | 1.006 | 0.391 |
| AP1 | 5e+04 | 0.0937 | 0.0322 | 0.0852 | 1.100 | 0.378 |
| AP1 | 6.01e+04 | 0.0787 | 0.0270 | 0.0781 | 1.007 | 0.346 |
| AP1 | 7.85e+04 | 0.0677 | 0.0212 | 0.0689 | 0.982 | 0.308 |
| AP1 | 9.55e+04 | 0.0626 | 0.0189 | 0.0628 | 0.996 | 0.300 |
| AP1 | 1.25e+05 | 0.0581 | 0.0228 | 0.0555 | 1.047 | 0.411 |
| AP1 | 1.46e+05 | 0.0517 | 0.0180 | 0.0515 | 1.005 | 0.350 |
| AP1 | 1.71e+05 | 0.0449 | 0.0130 | 0.0478 | 0.938 | 0.271 |
| AP1 | 2.61e+05 | 0.0390 | 0.0117 | 0.0392 | 0.996 | 0.299 |
| AP1 | 4.21e+05 | 0.0322 | 0.0103 | 0.0313 | 1.030 | 0.328 |
| AP1 | 5.32e+05 | 0.0281 | 0.0094 | 0.0280 | 1.001 | 0.337 |
| AP1 | 6.71e+05 | 0.0260 | 0.0084 | 0.0251 | 1.034 | 0.334 |
| AP1 | 7.24e+05 | 0.0237 | 0.0084 | 0.0243 | 0.978 | 0.346 |
| AP1 | 9.74e+05 | 0.0215 | 0.0084 | 0.0211 | 1.019 | 0.397 |
| AP2 | 3.5e+03 | 0.2455 | 0.0721 | 0.2584 | 0.950 | 0.279 |
| AP2 | 4.46e+03 | 0.2262 | 0.0619 | 0.2306 | 0.981 | 0.268 |
| AP2 | 5.93e+03 | 0.1987 | 0.0615 | 0.2016 | 0.986 | 0.305 |
| AP2 | 7.68e+03 | 0.1771 | 0.0547 | 0.1785 | 0.992 | 0.306 |
| AP2 | 1e+04 | 0.1568 | 0.0487 | 0.1575 | 0.995 | 0.309 |
| AP2 | 1.28e+04 | 0.1386 | 0.0444 | 0.1407 | 0.985 | 0.316 |
| AP2 | 1.64e+04 | 0.1221 | 0.0480 | 0.1251 | 0.976 | 0.384 |
| AP2 | 2.15e+04 | 0.1069 | 0.0409 | 0.1101 | 0.971 | 0.371 |
| AP2 | 2.85e+04 | 0.0949 | 0.0358 | 0.0964 | 0.984 | 0.371 |
| AP2 | 3.84e+04 | 0.0748 | 0.0324 | 0.0838 | 0.893 | 0.387 |
| AP2 | 4.91e+04 | 0.0791 | 0.0346 | 0.0747 | 1.059 | 0.464 |
| AP2 | 5.95e+04 | 0.0652 | 0.0268 | 0.0682 | 0.957 | 0.393 |
| AP2 | 7.81e+04 | 0.0566 | 0.0193 | 0.0600 | 0.942 | 0.322 |
| AP2 | 1.01e+05 | 0.0525 | 0.0195 | 0.0533 | 0.986 | 0.366 |
| AP2 | 1.36e+05 | 0.0468 | 0.0177 | 0.0462 | 1.012 | 0.382 |
| AP2 | 1.7e+05 | 0.0420 | 0.0156 | 0.0416 | 1.007 | 0.373 |
| AP2 | 2.19e+05 | 0.0367 | 0.0130 | 0.0370 | 0.993 | 0.351 |
| AP2 | 3.18e+05 | 0.0326 | 0.0130 | 0.0310 | 1.051 | 0.419 |
| AP2 | 3.55e+05 | 0.0306 | 0.0130 | 0.0295 | 1.040 | 0.441 |
| AP2 | 1.23e+06 | 0.0163 | 0.0130 | 0.0164 | 0.994 | 0.791 |
| AP2 | 1.44e+06 | 0.0153 | 0.0130 | 0.0153 | 1.002 | 0.852 |
| AP3 | 5.38e+03 | 0.1937 | 0.0662 | 0.2011 | 0.963 | 0.329 |
| AP3 | 6.8e+03 | 0.1780 | 0.0513 | 0.1801 | 0.988 | 0.285 |
| AP3 | 8.73e+03 | 0.1553 | 0.0584 | 0.1602 | 0.970 | 0.365 |
| AP3 | 1.1e+04 | 0.1397 | 0.0495 | 0.1438 | 0.971 | 0.344 |
| AP3 | 1.44e+04 | 0.1224 | 0.0571 | 0.1266 | 0.967 | 0.451 |
| AP3 | 1.86e+04 | 0.1119 | 0.0373 | 0.1123 | 0.996 | 0.332 |
| AP3 | 2.28e+04 | 0.1004 | 0.0298 | 0.1020 | 0.985 | 0.292 |
| AP3 | 3.11e+04 | 0.0928 | 0.0388 | 0.0881 | 1.053 | 0.441 |
| AP3 | 3.86e+04 | 0.0860 | 0.0337 | 0.0796 | 1.080 | 0.423 |
| AP3 | 4.95e+04 | 0.0703 | 0.0231 | 0.0709 | 0.991 | 0.326 |
| AP3 | 6.93e+04 | 0.0632 | 0.0239 | 0.0605 | 1.044 | 0.395 |
| AP3 | 7.72e+04 | 0.0450 | 0.0189 | 0.0575 | 0.783 | 0.328 |
| AP3 | 1.06e+05 | 0.0506 | 0.0221 | 0.0495 | 1.021 | 0.447 |
| AP3 | 1.36e+05 | 0.0430 | 0.0140 | 0.0441 | 0.975 | 0.316 |
| AP3 | 1.7e+05 | 0.0833 | 0.0120 | 0.0397 | 2.096 | 0.301 |
| AP3 | 2.06e+05 | 0.0358 | 0.0118 | 0.0362 | 0.988 | 0.325 |
| AP3 | 2.66e+05 | 0.0276 | 0.0115 | 0.0321 | 0.860 | 0.359 |
| AP3 | 1.65e+06 | 0.0137 | 0.0115 | 0.0136 | 1.005 | 0.846 |
| AP4 | 7.82e+03 | 0.2069 | 0.0711 | 0.1800 | 1.150 | 0.395 |
| AP4 | 9.29e+03 | 0.1888 | 0.0681 | 0.1660 | 1.137 | 0.410 |
| AP4 | 1.06e+04 | 0.1543 | 0.0606 | 0.1558 | 0.991 | 0.389 |
| AP4 | 1.33e+04 | 0.1390 | 0.0579 | 0.1404 | 0.990 | 0.413 |
| AP4 | 1.55e+04 | 0.1307 | 0.0520 | 0.1306 | 1.001 | 0.398 |
| AP4 | 1.76e+04 | 0.1272 | 0.0519 | 0.1228 | 1.036 | 0.423 |
| AP4 | 2.09e+04 | 0.1124 | 0.0438 | 0.1134 | 0.991 | 0.386 |
| AP4 | 2.49e+04 | 0.1051 | 0.0351 | 0.1045 | 1.006 | 0.336 |
| AP4 | 2.92e+04 | 0.0934 | 0.0331 | 0.0968 | 0.964 | 0.342 |
| AP4 | 3.33e+04 | 0.0880 | 0.0390 | 0.0911 | 0.966 | 0.429 |
| AP4 | 3.95e+04 | 0.0848 | 0.0432 | 0.0840 | 1.010 | 0.514 |
| AP4 | 4.78e+04 | 0.0794 | 0.0277 | 0.0768 | 1.033 | 0.360 |
| AP4 | 5.68e+04 | 0.0712 | 0.0301 | 0.0709 | 1.005 | 0.425 |
| AP4 | 6.93e+04 | 0.0698 | 0.0301 | 0.0645 | 1.082 | 0.466 |
| AP4 | 7.63e+04 | 0.0613 | 0.0223 | 0.0617 | 0.994 | 0.362 |
| AP4 | 9.49e+04 | 0.0597 | 0.0260 | 0.0557 | 1.072 | 0.467 |
| AP4 | 1.11e+05 | 0.0528 | 0.0188 | 0.0517 | 1.021 | 0.363 |
| AP4 | 1.27e+05 | 0.0475 | 0.0158 | 0.0486 | 0.977 | 0.326 |
| AP4 | 1.45e+05 | 0.0460 | 0.0153 | 0.0456 | 1.010 | 0.337 |
| AP4 | 2.19e+05 | 0.0417 | 0.0138 | 0.0376 | 1.111 | 0.367 |
| AP4 | 2.43e+05 | 0.0375 | 0.0133 | 0.0358 | 1.048 | 0.371 |
| AP4 | 3e+05 | 0.0328 | 0.0121 | 0.0324 | 1.013 | 0.374 |
| AP4 | 3.49e+05 | 0.0303 | 0.0121 | 0.0302 | 1.005 | 0.401 |
| AP5 | 2.16e+03 | 0.2802 | 0.0850 | 0.2909 | 0.963 | 0.292 |
| AP5 | 2.76e+03 | 0.2491 | 0.0769 | 0.2593 | 0.961 | 0.296 |
| AP5 | 3.64e+03 | 0.2220 | 0.0642 | 0.2277 | 0.975 | 0.282 |
| AP5 | 4.74e+03 | 0.1967 | 0.0576 | 0.2012 | 0.978 | 0.286 |
| AP5 | 6.13e+03 | 0.1732 | 0.0528 | 0.1783 | 0.971 | 0.296 |
| AP5 | 8e+03 | 0.1541 | 0.0469 | 0.1573 | 0.979 | 0.298 |
| AP5 | 1.09e+04 | 0.1274 | 0.0419 | 0.1358 | 0.938 | 0.309 |
| AP5 | 1.39e+04 | 0.1227 | 0.0463 | 0.1213 | 1.011 | 0.382 |
| AP5 | 1.88e+04 | 0.1067 | 0.0379 | 0.1054 | 1.013 | 0.360 |
| AP5 | 2.44e+04 | 0.0967 | 0.0391 | 0.0931 | 1.038 | 0.420 |
| AP5 | 3.09e+04 | 0.0828 | 0.0357 | 0.0834 | 0.993 | 0.428 |
| AP5 | 3.91e+04 | 0.0730 | 0.0278 | 0.0746 | 0.979 | 0.373 |
| AP5 | 5.07e+04 | 0.0659 | 0.0251 | 0.0660 | 0.997 | 0.379 |
| AP5 | 6.52e+04 | 0.0581 | 0.0206 | 0.0587 | 0.991 | 0.351 |
| AP5 | 8.8e+04 | 0.0480 | 0.0170 | 0.0510 | 0.942 | 0.335 |
| AP5 | 1.23e+05 | 0.0430 | 0.0146 | 0.0435 | 0.989 | 0.337 |
| AP5 | 1.66e+05 | 0.0393 | 0.0133 | 0.0379 | 1.038 | 0.352 |
| AP5 | 1.99e+05 | 0.0351 | 0.0130 | 0.0347 | 1.011 | 0.375 |
| AP5 | 2.58e+05 | 0.0312 | 0.0130 | 0.0308 | 1.015 | 0.423 |
| AP5 | 7.64e+05 | 0.0184 | 0.0130 | 0.0185 | 0.996 | 0.705 |
| AP5 | 1.01e+06 | 0.0164 | 0.0130 | 0.0162 | 1.016 | 0.806 |

## C2. Section 2b.5: slope-area fits at the Phase 6 optima after the fix, against the pre-fix run

Run `20261112T195138Z_slope_area_10tiles` (job 52087676); pre-fix run `20261112T175641Z_slope_area_10tiles` (the Phase 6 report's figure).

| tile | series | k_s | theta | R^2 | points | pre-fix k_s | pre-fix theta | k_s ratio new / pre-fix |
|---|---|---|---|---|---|---|---|---|
| AP1 | orig | 35.58 | 0.541 | 0.948 | 3798 | 35.58 | 0.541 | 1.000 |
| AP1 | AD_no_opt | 0.2484 | -0.114 | 0.107 | 2170 | 0.08929 | -0.107 | 2.782 |
| AP1 | AD_opt | 7.271 | 0.411 | 0.971 | 3856 | 2.035 | 0.401 | 3.572 |
| AP1 | part | 13.99 | 0.471 | 0.995 | 3879 | 8.427 | 0.522 | 1.660 |
| AP2 | orig | 36.68 | 0.558 | 0.973 | 4355 | 36.68 | 0.558 | 1.000 |
| AP2 | AD_no_opt | 0.7179 | -0.024 | 0.012 | 3617 | 0.2213 | -0.028 | 3.245 |
| AP2 | AD_opt | 5.852 | 0.393 | 0.981 | 4566 | 0.9785 | 0.333 | 5.981 |
| AP2 | part | 11.53 | 0.467 | 0.999 | 4499 | 2.314 | 0.412 | 4.982 |
| AP3 | orig | 18.27 | 0.494 | 0.923 | 3422 | 18.27 | 0.494 | 1.000 |
| AP3 | AD_no_opt | 2.544 | 0.062 | 0.174 | 2940 | 0.873 | 0.065 | 2.914 |
| AP3 | AD_opt | 6.048 | 0.399 | 0.971 | 3637 | 1.154 | 0.359 | 5.242 |
| AP3 | part | 10.85 | 0.467 | 0.998 | 3544 | 4.664 | 0.482 | 2.326 |
| AP4 | orig | 30.26 | 0.540 | 0.946 | 2838 | 30.26 | 0.540 | 1.000 |
| AP4 | AD_no_opt | 0.3381 | -0.087 | 0.098 | 2789 | 0.0818 | -0.112 | 4.133 |
| AP4 | AD_opt | 5.374 | 0.377 | 0.959 | 3081 | 0.9708 | 0.326 | 5.536 |
| AP4 | part | 12.44 | 0.471 | 0.996 | 2947 | 5.578 | 0.486 | 2.230 |
| AP5 | orig | 15.28 | 0.491 | 0.922 | 2870 | 15.28 | 0.491 | 1.000 |
| AP5 | AD_no_opt | 0.5235 | -0.052 | 0.008 | 2158 | 0.2761 | -0.022 | 1.896 |
| AP5 | AD_opt | 4.656 | 0.383 | 0.979 | 3028 | 1.54 | 0.386 | 3.023 |
| AP5 | part | 9.563 | 0.460 | 0.999 | 2879 | 3.047 | 0.447 | 3.139 |
| GM1 | orig | 3.22 | 0.314 | 0.953 | 4638 | 3.22 | 0.314 | 1.000 |
| GM1 | AD_no_opt | 0.1591 | -0.167 | 0.194 | 3941 | 0.1591 | -0.167 | 1.000 |
| GM1 | AD_opt | 1.74 | 0.264 | 0.958 | 4735 | 1.74 | 0.264 | 1.000 |
| GM1 | part | 2.827 | 0.304 | 0.976 | 4717 | 2.827 | 0.304 | 1.000 |
| GM2 | orig | 2.488 | 0.306 | 0.962 | 4424 | 2.488 | 0.306 | 1.000 |
| GM2 | AD_no_opt | 0.4157 | -0.066 | 0.212 | 3774 | 0.4157 | -0.066 | 1.000 |
| GM2 | AD_opt | 1.666 | 0.273 | 0.908 | 4608 | 1.666 | 0.273 | 1.000 |
| GM2 | part | 2.539 | 0.306 | 1.000 | 4413 | 2.539 | 0.306 | 1.000 |
| GM3 | orig | 7.194 | 0.413 | 0.953 | 4935 | 7.194 | 0.413 | 1.000 |
| GM3 | AD_no_opt | 0.174 | -0.162 | 0.387 | 4436 | 0.174 | -0.162 | 1.000 |
| GM3 | AD_opt | 1.456 | 0.286 | 0.981 | 4772 | 1.456 | 0.286 | 1.000 |
| GM3 | part | 2.166 | 0.311 | 0.999 | 4895 | 2.166 | 0.311 | 1.000 |
| GM4 | orig | 6.704 | 0.352 | 0.984 | 4662 | 6.704 | 0.352 | 1.000 |
| GM4 | AD_no_opt | 0.5733 | -0.051 | 0.078 | 2661 | 0.5733 | -0.051 | 1.000 |
| GM4 | AD_opt | 2.279 | 0.276 | 0.955 | 4721 | 2.279 | 0.276 | 1.000 |
| GM4 | part | 3.974 | 0.310 | 0.999 | 4715 | 3.974 | 0.310 | 1.000 |
| GM5 | orig | 4.027 | 0.337 | 0.942 | 6107 | 4.027 | 0.337 | 1.000 |
| GM5 | AD_no_opt | 0.457 | -0.062 | 0.046 | 4195 | 0.457 | -0.062 | 1.000 |
| GM5 | AD_opt | 2.597 | 0.303 | 0.973 | 6407 | 2.597 | 0.303 | 1.000 |
| GM5 | part | 2.757 | 0.309 | 1.000 | 6329 | 2.757 | 0.309 | 1.000 |

## C3. Section 1.3: fig3_deltas from the per-stage Phase 6 runs

| tile | stage | D0 end | K0 (1/yr) | D0 (m^2/yr) | K_fit | D_fit | dK = log10(K/K0) | dD = log10(D/D0) | MSE | run |
|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | ad | A | 4.428e-06 | 3.587e-03 | 3.995e-05 | 5.447e-04 | +0.955 | -0.819 | 59.02 | `20261111T012439Z_tile_run_AP1_v2_ad` |
| AP1 | ad | B | 4.428e-06 | 6.978e-03 | 3.995e-05 | 5.447e-04 | +0.955 | -1.108 | 59.02 | `20261111T012439Z_tile_run_AP1_v2_ad` |
| AP2 | ad | A | 4.748e-06 | 3.218e-03 | 4.264e-05 | 5.534e-04 | +0.953 | -0.765 | 76.49 | `20261111T012443Z_tile_run_AP2_v2_ad` |
| AP2 | ad | B | 4.748e-06 | 4.078e-03 | 4.264e-05 | 5.534e-04 | +0.953 | -0.867 | 76.49 | `20261111T012443Z_tile_run_AP2_v2_ad` |
| AP3 | ad | A | 4.770e-06 | 2.155e-03 | 4.585e-05 | 5.032e-04 | +0.983 | -0.632 | 152.8 | `20261111T012443Z_tile_run_AP3_v2_ad` |
| AP3 | ad | B | 4.770e-06 | 2.155e-03 | 4.585e-05 | 5.032e-04 | +0.983 | -0.632 | 152.8 | `20261111T012443Z_tile_run_AP3_v2_ad` |
| AP4 | ad | A | 4.610e-06 | 3.629e-03 | 4.710e-05 | 1.019e-03 | +1.009 | -0.551 | 45.72 | `20261111T012445Z_tile_run_AP4_v2_ad` |
| AP4 | ad | B | 4.610e-06 | 7.720e-03 | 4.710e-05 | 1.019e-03 | +1.009 | -0.879 | 45.72 | `20261111T012445Z_tile_run_AP4_v2_ad` |
| AP5 | ad | A | 5.128e-06 | 2.921e-03 | 4.421e-05 | 5.728e-04 | +0.936 | -0.707 | 69.81 | `20261111T012444Z_tile_run_AP5_v2_ad` |
| AP5 | ad | B | 5.128e-06 | 3.456e-03 | 4.421e-05 | 5.728e-04 | +0.936 | -0.780 | 69.81 | `20261111T012444Z_tile_run_AP5_v2_ad` |
| GM1 | ad | A | 4.854e-05 | 9.001e-03 | 1.483e-04 | 2.947e-04 | +0.485 | -1.485 | 48.96 | `20261111T012445Z_tile_run_GM1_v2_ad` |
| GM2 | ad | A | 5.514e-05 | 9.770e-03 | 1.309e-04 | 8.780e-05 | +0.376 | -2.046 | 82.55 | `20261111T012445Z_tile_run_GM2_v2_ad` |
| GM3 | ad | A | 6.413e-05 | 1.012e-02 | 1.356e-04 | 4.091e-05 | +0.325 | -2.393 | 64.72 | `20261111T012445Z_tile_run_GM3_v2_ad` |
| GM4 | ad | A | 3.479e-05 | 9.855e-03 | 9.514e-05 | 6.291e-05 | +0.437 | -2.195 | 174.5 | `20261111T012439Z_tile_run_GM4_v2_ad` |
| GM5 | ad | A | 5.019e-05 | 7.589e-03 | 1.540e-04 | 2.288e-04 | +0.487 | -1.521 | 93.32 | `20261111T012438Z_tile_run_GM5_v2_ad` |
| AP1 | joint | A | 4.428e-06 | 3.587e-03 | 3.995e-06 | 1.699e-02 | -0.045 | +0.675 | 22.38 | `20261111T050726Z_tile_run_AP1_v2_partitioned-refined` |
| AP1 | joint | B | 4.428e-06 | 6.978e-03 | 3.995e-06 | 1.699e-02 | -0.045 | +0.386 | 22.38 | `20261111T050726Z_tile_run_AP1_v2_partitioned-refined` |
| AP2 | joint | A | 4.748e-06 | 3.218e-03 | 4.597e-06 | 1.680e-02 | -0.014 | +0.718 | 22.46 | `20261111T050734Z_tile_run_AP2_v2_partitioned-refined` |
| AP2 | joint | B | 4.748e-06 | 4.078e-03 | 4.597e-06 | 1.680e-02 | -0.014 | +0.615 | 22.46 | `20261111T050734Z_tile_run_AP2_v2_partitioned-refined` |
| AP3 | joint | A | 4.770e-06 | 2.155e-03 | 4.825e-06 | 1.791e-02 | +0.005 | +0.920 | 42.12 | `20261111T045020Z_tile_run_AP3_v2_partitioned-refined` |
| AP3 | joint | B | 4.770e-06 | 2.155e-03 | 4.825e-06 | 1.791e-02 | +0.005 | +0.920 | 42.12 | `20261111T045020Z_tile_run_AP3_v2_partitioned-refined` |
| AP4 | joint | A | 4.610e-06 | 3.629e-03 | 4.522e-06 | 1.974e-02 | -0.008 | +0.736 | 35.16 | `20261111T050505Z_tile_run_AP4_v2_partitioned-refined` |
| AP4 | joint | B | 4.610e-06 | 7.720e-03 | 4.522e-06 | 1.974e-02 | -0.008 | +0.408 | 35.16 | `20261111T050505Z_tile_run_AP4_v2_partitioned-refined` |
| AP5 | joint | A | 5.128e-06 | 2.921e-03 | 5.119e-06 | 1.258e-02 | -0.001 | +0.634 | 27.87 | `20261111T050104Z_tile_run_AP5_v2_partitioned-refined` |
| AP5 | joint | B | 5.128e-06 | 3.456e-03 | 5.119e-06 | 1.258e-02 | -0.001 | +0.561 | 27.87 | `20261111T050104Z_tile_run_AP5_v2_partitioned-refined` |
| GM1 | joint | A | 4.854e-05 | 9.001e-03 | 4.865e-05 | 1.948e-02 | +0.001 | +0.335 | 24.69 | `20261111T050111Z_tile_run_GM1_v2_partitioned-refined` |
| GM2 | joint | A | 5.514e-05 | 9.770e-03 | 5.515e-05 | 2.743e-02 | +0.000 | +0.448 | 26.07 | `20261111T052240Z_tile_run_GM2_v2_partitioned-refined` |
| GM3 | joint | A | 6.413e-05 | 1.012e-02 | 6.810e-05 | 1.923e-02 | +0.026 | +0.279 | 33.03 | `20261111T052022Z_tile_run_GM3_v2_partitioned-refined` |
| GM4 | joint | A | 3.479e-05 | 9.855e-03 | 3.640e-05 | 2.189e-02 | +0.020 | +0.347 | 40.74 | `20261111T051659Z_tile_run_GM4_v2_partitioned-refined` |
| GM5 | joint | A | 5.019e-05 | 7.589e-03 | 5.234e-05 | 1.903e-02 | +0.018 | +0.399 | 32.21 | `20261111T045953Z_tile_run_GM5_v2_partitioned-refined` |
| AP1 | nonlinear_joint | A | 4.428e-06 | 3.587e-03 | 3.995e-06 | 1.652e-02 | -0.045 | +0.663 | 22.07 | `20261112T175112Z_tile_run_AP1_v2_nonlinear_joint_merged` |
| AP2 | nonlinear_joint | A | 4.748e-06 | 3.218e-03 | 4.597e-06 | 1.652e-02 | -0.014 | +0.710 | 22.39 | `20261112T175112Z_tile_run_AP2_v2_nonlinear_joint_merged` |
| AP3 | nonlinear_joint | A | 4.770e-06 | 2.155e-03 | 4.825e-06 | 1.652e-02 | +0.005 | +0.885 | 42.14 | `20261112T175112Z_tile_run_AP3_v2_nonlinear_joint_merged` |
| AP4 | nonlinear_joint | A | 4.610e-06 | 3.629e-03 | 4.522e-06 | 1.987e-02 | -0.008 | +0.738 | 34.38 | `20261112T175112Z_tile_run_AP4_v2_nonlinear_joint_merged` |
| AP5 | nonlinear_joint | A | 5.128e-06 | 2.921e-03 | 5.119e-06 | 1.142e-02 | -0.001 | +0.592 | 28.48 | `20261112T175112Z_tile_run_AP5_v2_nonlinear_joint_merged` |
| GM1 | nonlinear_joint | A | 4.854e-05 | 9.001e-03 | 4.865e-05 | 1.987e-02 | +0.001 | +0.344 | 25.21 | `20261112T175109Z_tile_run_GM1_v2_nonlinear_joint_merged` |
| GM2 | nonlinear_joint | A | 5.514e-05 | 9.770e-03 | 5.515e-05 | 2.874e-02 | +0.000 | +0.469 | 26.51 | `20261112T175116Z_tile_run_GM2_v2_nonlinear_joint_merged` |
| GM3 | nonlinear_joint | A | 6.413e-05 | 1.012e-02 | 6.810e-05 | 1.987e-02 | +0.026 | +0.293 | 33.89 | `20261112T175103Z_tile_run_GM3_v2_nonlinear_joint_merged` |
| GM4 | nonlinear_joint | A | 3.479e-05 | 9.855e-03 | 3.640e-05 | 1.987e-02 | +0.020 | +0.305 | 40.97 | `20261112T175116Z_tile_run_GM4_v2_nonlinear_joint_merged` |
| GM5 | nonlinear_joint | A | 5.019e-05 | 7.589e-03 | 5.234e-05 | 1.987e-02 | +0.018 | +0.418 | 32.14 | `20261112T175112Z_tile_run_GM5_v2_nonlinear_joint_merged` |

## C4. Section 2.1: observed curvature by hillslope class at the Phase 6 joint optimum

Classes on the model hillslope cells (A < A_crit, boundary and channel cells excluded): divergent A < A_roll, hollow A_roll <= A < A_crit, hilltop = the option (iii) set (subset of divergent). -lap3: observed -laplacian at the 3-cell window (the model's 5-point stencil); -lap11: the quadratic-surface estimator at L_f = 11 m. Concave = -lap3 < 0.

| tile | run (job) | A_roll (m^2) | A_crit | hillslope cells | channel cells | class | n | share | mean -lap3 (1/m) | median -lap3 | concave | mean -lap11 | median -lap11 | concave (11 m) | SSR share | class MSE (m^2) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | `20261112T195139Z_d_diagnostic_AP1` (52087750) | 282 | 6889 | 1021294 | 4887 | divergent | 971013 | 95.1 % | 0.004735 | 0.0007826 | 47.8 % | 0.0007772 | 0.002544 | 43.7 % | 94.8 % | 22.37 |
| | | | | | | hollow | 50281 | 4.9 % | -0.06634 | -0.0302 | 68.1 % | -0.009412 | -0.005911 | 61.5 % | 5.2 % | 23.8 |
| | | | | | | hilltop | 3446 | 0.3 % | 0.09609 | 0.05918 | 5.5 % | 0.01699 | 0.01533 | 10.1 % | 0.4 % | 25.48 |
| | | | | | | all hillslope | 1021294 | 100 % | 0.001235 | 0.0003528 | 48.8 % | 0.000276 | 0.00222 | 44.6 % | 100 % | 22.44 |
| AP2 | `20261112T195143Z_d_diagnostic_AP2` (52087751) | 56.2 | 3121 | 1015187 | 11024 | divergent | 849659 | 83.7 % | 0.009036 | 0.001956 | 46.0 % | 0.001101 | 0.002643 | 43.4 % | 85.7 % | 22.94 |
| | | | | | | hollow | 165528 | 16.3 % | -0.03562 | -0.007786 | 61.4 % | -0.004493 | -0.001673 | 54.5 % | 14.3 % | 19.67 |
| | | | | | | hilltop | 944 | 0.1 % | 0.07871 | 0.05607 | 4.8 % | 0.02024 | 0.01709 | 5.1 % | 0.2 % | 39 |
| | | | | | | all hillslope | 1015187 | 100 % | 0.001755 | 0.0005868 | 48.5 % | 0.0001875 | 0.001898 | 45.2 % | 100 % | 22.4 |
| AP3 | `20261112T195139Z_d_diagnostic_AP3` (52087752) | 17.8 | 4742 | 1017927 | 7246 | divergent | 685200 | 67.3 % | 0.01714 | 0.003795 | 44.3 % | 0.002443 | 0.003028 | 43.2 % | 70.0 % | 43.91 |
| | | | | | | hollow | 332727 | 32.7 % | -0.03187 | -0.009899 | 61.2 % | -0.003987 | -0.002725 | 56.2 % | 30.0 % | 38.71 |
| | | | | | | hilltop | 533 | 0.1 % | 0.07619 | 0.0565 | 6.0 % | 0.0301 | 0.02553 | 0.0 % | 0.1 % | 103.4 |
| | | | | | | all hillslope | 1017927 | 100 % | 0.001119 | 0 | 49.8 % | 0.0003502 | 0.001153 | 47.4 % | 100 % | 42.21 |
| AP4 | `20261112T195139Z_d_diagnostic_AP4` (52087753) | 447 | 7232 | 1021835 | 4346 | divergent | 982912 | 96.2 % | 0.004017 | 0.000626 | 48.3 % | 0.000881 | 0.002036 | 44.9 % | 96.2 % | 35.19 |
| | | | | | | hollow | 38923 | 3.8 % | -0.06753 | -0.03689 | 69.7 % | -0.01033 | -0.006907 | 64.7 % | 3.8 % | 35.24 |
| | | | | | | hilltop | 4711 | 0.5 % | 0.08734 | 0.06178 | 8.5 % | 0.01595 | 0.01515 | 10.5 % | 0.8 % | 60.55 |
| | | | | | | all hillslope | 1021835 | 100 % | 0.001291 | 0.0002354 | 49.1 % | 0.0004534 | 0.00176 | 45.6 % | 100 % | 35.19 |
| AP5 | `20261112T195139Z_d_diagnostic_AP5` (52087674) | 56.2 | 1887 | 1013169 | 13040 | divergent | 864736 | 85.3 % | 0.009005 | 0.0018 | 46.4 % | 0.001671 | 0.002967 | 43.1 % | 82.5 % | 26.9 |
| | | | | | | hollow | 148433 | 14.7 % | -0.03778 | -0.01287 | 62.8 % | -0.005163 | -0.00332 | 58.1 % | 17.5 % | 33.24 |
| | | | | | | hilltop | 3575 | 0.4 % | 0.07431 | 0.05501 | 7.5 % | 0.02149 | 0.01883 | 2.3 % | 0.3 % | 20.45 |
| | | | | | | all hillslope | 1013169 | 100 % | 0.002152 | 0.0004315 | 48.8 % | 0.0006703 | 0.002037 | 45.3 % | 100 % | 27.83 |
| GM1 | `20261112T195139Z_d_diagnostic_GM1` (52087746) | 35.5 | 1241 | 988163 | 7880 | divergent | 728407 | 73.7 % | 0.01244 | 0.01175 | 44.3 % | 0.004079 | 0.007293 | 33.7 % | 70.2 % | 23.59 |
| | | | | | | hollow | 259756 | 26.3 % | -0.02536 | -0.01996 | 59.0 % | -0.006717 | -0.002492 | 55.3 % | 29.8 % | 28.05 |
| | | | | | | hilltop | 3583 | 0.4 % | 0.06326 | 0.05365 | 11.2 % | 0.01801 | 0.01633 | 0.3 % | 0.4 % | 25.31 |
| | | | | | | all hillslope | 988163 | 100 % | 0.0025 | 0.003906 | 48.2 % | 0.001242 | 0.004943 | 39.4 % | 100 % | 24.77 |
| GM2 | `20261112T195139Z_d_diagnostic_GM2` (52087747) | 35.5 | 3509 | 989553 | 6481 | divergent | 711423 | 71.9 % | 0.01158 | 0.01071 | 44.7 % | 0.003099 | 0.00533 | 35.4 % | 69.4 % | 25.27 |
| | | | | | | hollow | 278130 | 28.1 % | -0.02318 | -0.01868 | 58.6 % | -0.005208 | -0.001806 | 54.7 % | 30.6 % | 28.52 |
| | | | | | | hilltop | 2530 | 0.3 % | 0.0618 | 0.05373 | 11.9 % | 0.01621 | 0.01505 | 0.2 % | 0.5 % | 49.46 |
| | | | | | | all hillslope | 989553 | 100 % | 0.001806 | 0.002808 | 48.6 % | 0.0007619 | 0.003457 | 40.8 % | 100 % | 26.19 |
| GM3 | `20261112T195141Z_d_diagnostic_GM3` (52087748) | 22.4 | 860 | 982280 | 13769 | divergent | 672432 | 68.5 % | 0.01391 | 0.01132 | 44.0 % | 0.003445 | 0.004214 | 34.2 % | 73.9 % | 35.77 |
| | | | | | | hollow | 309848 | 31.5 % | -0.0251 | -0.02225 | 61.0 % | -0.005361 | -0.002269 | 56.5 % | 26.1 % | 27.38 |
| | | | | | | hilltop | 1500 | 0.2 % | 0.06359 | 0.05087 | 14.6 % | 0.02579 | 0.01453 | 0.0 % | 0.1 % | 30.62 |
| | | | | | | all hillslope | 982280 | 100 % | 0.001606 | 0.0009766 | 49.4 % | 0.0006699 | 0.002531 | 41.2 % | 100 % | 33.12 |
| GM4 | `20261112T195141Z_d_diagnostic_GM4` (52087749) | 56.2 | 3853 | 989880 | 6154 | divergent | 789294 | 79.7 % | 0.00905 | 0.007751 | 45.9 % | 0.002752 | 0.004731 | 37.7 % | 76.6 % | 39.2 |
| | | | | | | hollow | 200586 | 20.3 % | -0.02784 | -0.02154 | 59.2 % | -0.007578 | -0.004163 | 60.5 % | 23.4 % | 47.12 |
| | | | | | | hilltop | 3347 | 0.3 % | 0.06901 | 0.06131 | 12.9 % | 0.01963 | 0.01492 | 1.0 % | 0.5 % | 60.22 |
| | | | | | | all hillslope | 989880 | 100 % | 0.001575 | 0.001602 | 48.6 % | 0.0006595 | 0.003033 | 42.3 % | 100 % | 40.8 |
| GM5 | `20261112T195140Z_d_diagnostic_GM5` (52087675) | 28.2 | 1885 | 985381 | 10658 | divergent | 679742 | 69.0 % | 0.01357 | 0.01218 | 44.1 % | 0.003139 | 0.005008 | 38.1 % | 64.2 % | 30.07 |
| | | | | | | hollow | 305639 | 31.0 % | -0.02336 | -0.01883 | 58.6 % | -0.00583 | -0.003338 | 57.7 % | 35.8 % | 37.22 |
| | | | | | | hilltop | 1128 | 0.1 % | 0.06422 | 0.06035 | 13.7 % | 0.01977 | 0.01937 | 0.1 % | 0.1 % | 37.96 |
| | | | | | | all hillslope | 985381 | 100 % | 0.002114 | 0.002838 | 48.6 % | 0.0003593 | 0.002494 | 44.2 % | 100 % | 32.29 |

All hillslope cells against the fitted convexity U/D_fit:

| tile | U/D_fit (1/m) | mean -lap3 | mean / (U/D_fit) | median / (U/D_fit) | mean -lap11 / (U/D_fit) | model -lap3 on hillslope cells (median) | channel MSE | hillslope MSE | MSE all (check) |
|---|---|---|---|---|---|---|---|---|---|
| AP1 | 0.003237 | 0.001235 | 0.382 | 0.109 | 0.085 | 0.003237 | 28.25 | 22.44 | 22.38 |
| AP2 | 0.003275 | 0.001755 | 0.536 | 0.179 | 0.057 | 0.003275 | 35.99 | 22.4 | 22.46 |
| AP3 | 0.003071 | 0.001119 | 0.364 | 0.000 | 0.114 | 0.003072 | 53.34 | 42.21 | 42.12 |
| AP4 | 0.002786 | 0.001291 | 0.464 | 0.085 | 0.163 | 0.002786 | 59.26 | 35.19 | 35.16 |
| AP5 | 0.004371 | 0.002152 | 0.492 | 0.099 | 0.153 | 0.004371 | 39.43 | 27.83 | 27.87 |
| GM1 | 0.007547 | 0.0025 | 0.331 | 0.518 | 0.165 | 0.007547 | 27.56 | 24.77 | 24.69 |
| GM2 | 0.00536 | 0.001806 | 0.337 | 0.524 | 0.142 | 0.00536 | 24.5 | 26.19 | 26.07 |
| GM3 | 0.007645 | 0.001606 | 0.210 | 0.128 | 0.088 | 0.007645 | 35.58 | 33.12 | 33.03 |
| GM4 | 0.006714 | 0.001575 | 0.235 | 0.239 | 0.098 | 0.006714 | 57.02 | 40.8 | 40.74 |
| GM5 | 0.007725 | 0.002114 | 0.274 | 0.367 | 0.047 | 0.007725 | 37.19 | 32.29 | 32.21 |

## C5. Section 2.2: the linear joint profile with the misfit restricted to each class (plus the channel cells of each A_crit)

| tile | mask | cells in the misfit at the optimum | evaluations | A_crit (m^2) | K (1/yr) | D (m^2/yr) | MSE over the mask (m^2) | box | 1 % interval | K / K0 | D / D0_A | D / D0_B | Phase 6 joint (A_crit, K, D, MSE) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | all | 1030225 | 2048 | 6888.6 | 3.9952e-06 | 1.6990e-02 | 22.384 | interior | 6540-6930 | 0.902 | 4.74 | 2.43 | 6888.6, 3.995e-06, 1.699e-02, 22.38 |
| AP1 | divergent | 975900 | 2048 | 6888.6 | 3.9929e-06 | 1.6931e-02 | 22.399 | interior | 6540-6930 | 0.902 | 4.72 | 2.43 | 6888.6, 3.995e-06, 1.699e-02, 22.38 |
| AP1 | hollow | 54917 | 2048 | 7580.7 | 3.9623e-06 | 2.0920e-02 | 20.924 | interior | 7205-7255; 7534-7905 | 0.895 | 5.83 | 3.00 | 6888.6, 3.995e-06, 1.699e-02, 22.38 |
| AP2 | all | 1030225 | 2088 | 3121.1 | 4.5968e-06 | 1.6796e-02 | 22.461 | interior | 2976-3027; 3070-3177; 3233-3300 | 0.968 | 5.22 | 4.12 | 3121.1, 4.597e-06, 1.680e-02, 22.46 |
| AP2 | divergent | 860683 | 2088 | 3121.1 | 4.5934e-06 | 1.6705e-02 | 23.093 | interior | 2986-3027; 3070-3177; 3233-3300 | 0.967 | 5.19 | 4.10 | 3121.1, 4.597e-06, 1.680e-02, 22.46 |
| AP2 | hollow | 172144 | 2088 | 7003.6 | 4.8306e-06 | 2.1761e-02 | 20.261 | interior | 3119-3133; 6905-7026; 7607-7686; 7737-7767 | 1.017 | 6.76 | 5.34 | 3121.1, 4.597e-06, 1.680e-02, 22.46 |
| AP3 | all | 1029210 | 2095 | 4742.0 | 4.8254e-06 | 1.7907e-02 | 42.122 | interior | 4635-4699; 4731-4829; 5504-5504; 5597-5676 | 1.012 | 8.31 | 8.31 | 4742.0, 4.825e-06, 1.791e-02, 42.12 |
| AP3 | divergent | 692446 | 2095 | 4742.0 | 4.8348e-06 | 1.7615e-02 | 43.915 | interior | 4650-4699; 4731-4796; 5155-5155; 5504-5522; 5597-5676 | 1.014 | 8.18 | 8.18 | 4742.0, 4.825e-06, 1.791e-02, 42.12 |
| AP3 | hollow | 339906 | 2095 | 4781.9 | 4.7866e-06 | 1.8812e-02 | 38.533 | interior | 4650-4699; 4731-4829; 5597-5676 | 1.003 | 8.73 | 8.73 | 4742.0, 4.825e-06, 1.791e-02, 42.12 |
| AP4 | all | 1030225 | 2063 | 7231.8 | 4.5219e-06 | 1.9743e-02 | 35.156 | interior | 6998-7394; 7641-7894 | 0.981 | 5.44 | 2.56 | 7231.8, 4.522e-06, 1.974e-02, 35.16 |
| AP4 | divergent | 987258 | 2063 | 7231.8 | 4.5284e-06 | 1.9642e-02 | 35.287 | interior | 6998-7394; 7641-7894 | 0.982 | 5.41 | 2.54 | 7231.8, 4.522e-06, 1.974e-02, 35.16 |
| AP4 | hollow | 42923 | 2063 | 7875.4 | 4.2493e-06 | 2.6527e-02 | 26.185 | interior | 7841-7894 | 0.922 | 7.31 | 3.44 | 7231.8, 4.522e-06, 1.974e-02, 35.16 |
| AP5 | all | 1030225 | 2134 | 1887.1 | 5.1191e-06 | 1.2584e-02 | 27.869 | interior | 1805-1935; 1968-1981 | 0.998 | 4.31 | 3.64 | 1887.1, 5.119e-06, 1.258e-02, 27.87 |
| AP5 | divergent | 877695 | 2134 | 1899.4 | 5.1084e-06 | 1.2485e-02 | 27.037 | interior | 1805-1935; 1948-1981 | 0.996 | 4.27 | 3.61 | 1887.1, 5.119e-06, 1.258e-02, 27.87 |
| AP5 | hollow | 161387 | 2134 | 1900.4 | 5.0848e-06 | 1.4079e-02 | 32.078 | interior | 1824-1989 | 0.992 | 4.82 | 4.07 | 1887.1, 5.119e-06, 1.258e-02, 27.87 |
| GM1 | all | 1000000 | 2011 | 1241.0 | 4.8653e-05 | 1.9477e-02 | 24.69 | interior | 1134-1267 | 1.002 | 2.16 | 2.08 | 1241.0, 4.865e-05, 1.948e-02, 24.69 |
| GM1 | divergent | 736603 | 2011 | 1138.0 | 4.8741e-05 | 1.8724e-02 | 23.494 | interior | 1134-1267 | 1.004 | 2.08 | 2.00 | 1241.0, 4.865e-05, 1.948e-02, 24.69 |
| GM1 | hollow | 264906 | 2011 | 6047.0 | 4.8450e-05 | 2.9390e-02 | 23.609 | interior | 5754-6238 | 0.998 | 3.27 | 3.14 | 1241.0, 4.865e-05, 1.948e-02, 24.69 |
| GM2 | all | 1000000 | 2041 | 3509.0 | 5.5148e-05 | 2.7426e-02 | 26.071 | interior | 3358-3536 | 1.000 | 2.81 | 2.70 | 3509.0, 5.515e-05, 2.743e-02, 26.07 |
| GM2 | divergent | 717904 | 2041 | 3509.0 | 5.4673e-05 | 2.6898e-02 | 24.909 | interior | 3358-3536 | 0.992 | 2.75 | 2.65 | 3509.0, 5.515e-05, 2.743e-02, 26.07 |
| GM2 | hollow | 284168 | 2041 | 4151.0 | 5.6271e-05 | 3.2726e-02 | 25.466 | interior | 3501-3536; 3687-3687; 3715-4189 | 1.021 | 3.35 | 3.22 | 3509.0, 5.515e-05, 2.743e-02, 26.07 |
| GM3 | all | 1000000 | 2006 | 860.0 | 6.8099e-05 | 1.9229e-02 | 33.027 | interior | 723-728; 733-743; 830-937 | 1.062 | 1.90 | 1.90 | 860.0, 6.810e-05, 1.923e-02, 33.03 |
| GM3 | divergent | 686201 | 2006 | 860.0 | 6.8220e-05 | 1.9118e-02 | 35.764 | interior | 718-746; 830-930 | 1.064 | 1.89 | 1.89 | 860.0, 6.810e-05, 1.923e-02, 33.03 |
| GM3 | hollow | 322978 | 2006 | 919.0 | 6.7442e-05 | 2.0225e-02 | 27.559 | interior | 844-940; 1061-1090; 1540-1550 | 1.052 | 2.00 | 2.00 | 860.0, 6.810e-05, 1.923e-02, 33.03 |
| GM4 | all | 1000000 | 2017 | 3853.0 | 3.6397e-05 | 2.1893e-02 | 40.743 | interior | 3265-3332; 3656-3893 | 1.046 | 2.22 | 2.32 | 3853.0, 3.640e-05, 2.189e-02, 40.74 |
| GM4 | divergent | 795448 | 2017 | 3853.0 | 3.6534e-05 | 2.1418e-02 | 39.103 | interior | 3692-3893 | 1.050 | 2.17 | 2.27 | 3853.0, 3.640e-05, 2.189e-02, 40.74 |
| GM4 | hollow | 206735 | 2017 | 3858.0 | 3.5783e-05 | 2.4685e-02 | 42.694 | interior | 3142-3142; 3187-3187; 3218-3447; 3656-3893 | 1.029 | 2.50 | 2.62 | 3853.0, 3.640e-05, 2.189e-02, 40.74 |
| GM5 | all | 1000000 | 2030 | 1885.0 | 5.2337e-05 | 1.9030e-02 | 32.211 | interior | 1819-1927; 2201-2348; 3104-3124 | 1.043 | 2.51 | 2.54 | 1885.0, 5.234e-05, 1.903e-02, 32.21 |
| GM5 | divergent | 690400 | 2030 | 1885.0 | 5.2182e-05 | 1.8451e-02 | 29.782 | interior | 1428-1428; 1448-1582; 1819-1907; 2201-2340 | 1.040 | 2.43 | 2.46 | 1885.0, 5.234e-05, 1.903e-02, 32.21 |
| GM5 | hollow | 314469 | 2030 | 3108.0 | 5.2394e-05 | 2.4080e-02 | 33.647 | interior | 3051-3136; 3211-3402; 3605-3605 | 1.044 | 3.17 | 3.21 | 1885.0, 5.234e-05, 1.903e-02, 32.21 |

The 'all' mask against the Phase 6 run (same reductions, coarser grid: thin 2000 + refinement, no 5 % fill): relative differences of A_crit, K, D, MSE:

| tile | A_crit | K | D | MSE |
|---|---|---|---|---|
| AP1 | +0.00% | +0.00% | +0.00% | +0.00% |
| AP2 | +0.00% | +0.00% | +0.00% | +0.00% |
| AP3 | +0.00% | +0.00% | +0.00% | +0.00% |
| AP4 | +0.00% | +0.00% | +0.00% | +0.00% |
| AP5 | +0.00% | +0.00% | +0.00% | +0.00% |
| GM1 | +0.00% | +0.00% | +0.00% | +0.00% |
| GM2 | +0.00% | +0.00% | +0.00% | +0.00% |
| GM3 | +0.00% | +0.00% | +0.00% | +0.00% |
| GM4 | +0.00% | +0.00% | +0.00% | +0.00% |
| GM5 | +0.00% | +0.00% | +0.00% | +0.00% |

## C6. Section 2.3: the rule, per tile and per site

(a) mean observed -lap3 over all hillslope cells within +-30 % of U/D_fit; (b) hollow mean -lap3 below half the divergent mean; (c) D from the divergent-only refit at least a factor 2 closer to D0_A than D_fit, in log ratio (|log10(D_div/D0_A)| <= |log10(D_fit/D0_A)| - log10 2); strict reading (log distance halved) beside it. Site: supported if all three hold on >= 4 of 5 tiles.

| tile | (a) mean -lap3 / (U/D_fit) | (a) | (b) hollow / divergent | (b) | D_fit | D_div | D_hollow | D0_A | D0_B | gap_fit (dec) | gap_div (dec) | reduction (dec) | (c) | (c) strict | all three |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | 0.382 | False | -14.013 | True | 1.699e-02 | 1.693e-02 | 2.092e-02 | 3.587e-03 | 6.978e-03 | 0.675 | 0.674 | +0.002 | False | False | **False** |
| AP2 | 0.536 | False | -3.942 | True | 1.680e-02 | 1.670e-02 | 2.176e-02 | 3.218e-03 | 4.078e-03 | 0.718 | 0.715 | +0.002 | False | False | **False** |
| AP3 | 0.364 | False | -1.860 | True | 1.791e-02 | 1.761e-02 | 1.881e-02 | 2.155e-03 | 2.155e-03 | 0.920 | 0.913 | +0.007 | False | False | **False** |
| AP4 | 0.464 | False | -16.813 | True | 1.974e-02 | 1.964e-02 | 2.653e-02 | 3.629e-03 | 7.720e-03 | 0.736 | 0.733 | +0.002 | False | False | **False** |
| AP5 | 0.492 | False | -4.195 | True | 1.258e-02 | 1.248e-02 | 1.408e-02 | 2.921e-03 | 3.456e-03 | 0.634 | 0.631 | +0.003 | False | False | **False** |
| GM1 | 0.331 | False | -2.040 | True | 1.948e-02 | 1.872e-02 | 2.939e-02 | 9.001e-03 | 9.354e-03 | 0.335 | 0.318 | +0.017 | False | False | **False** |
| GM2 | 0.337 | False | -2.003 | True | 2.743e-02 | 2.690e-02 | 3.273e-02 | 9.770e-03 | 1.015e-02 | 0.448 | 0.440 | +0.008 | False | False | **False** |
| GM3 | 0.210 | False | -1.804 | True | 1.923e-02 | 1.912e-02 | 2.022e-02 | 1.012e-02 | 1.012e-02 | 0.279 | 0.276 | +0.003 | False | False | **False** |
| GM4 | 0.235 | False | -3.076 | True | 2.189e-02 | 2.142e-02 | 2.468e-02 | 9.855e-03 | 9.435e-03 | 0.347 | 0.337 | +0.010 | False | False | **False** |
| GM5 | 0.274 | False | -1.722 | True | 1.903e-02 | 1.845e-02 | 2.408e-02 | 7.589e-03 | 7.496e-03 | 0.399 | 0.386 | +0.013 | False | False | **False** |

| site | tiles run | (a) passes | (b) passes | (c) passes | all three | supported (>= 4 of 5) |
|---|---|---|---|---|---|---|
| AP | 5 | 0 | 5 | 0 | 0 | **False** |
| GM | 5 | 0 | 5 | 0 | 0 | **False** |

## C7. Slurm accounting of the addendum 7 jobs

| job | name | state | elapsed | limit | CPUs | mem | max RSS (GB) | core-h used |
|---|---|---|---|---|---|---|---|---|
| 52087672_0 | lemcvt_a7_tests | COMPLETED | 00:03:14 | 00:20:00 | 2 | 8G | 1.1 | 0.1 |
| 52087673_0 | p7_slope_area_check | COMPLETED | 00:01:03 | 01:00:00 | 1 | 16G | 1.0 | 0.0 |
| 52087674_0 | p7_ddiag_AP | COMPLETED | 03:08:18 | 12:00:00 | 1 | 16G | 2.9 | 3.1 |
| 52087674_1 | p7_ddiag_AP | COMPLETED | 03:22:26 | 12:00:00 | 1 | 16G | 3.0 | 3.4 |
| 52087674_2 | p7_ddiag_AP | COMPLETED | 03:06:56 | 12:00:00 | 1 | 16G | 2.9 | 3.1 |
| 52087674_3 | p7_ddiag_AP | COMPLETED | 03:20:33 | 12:00:00 | 1 | 16G | 2.9 | 3.3 |
| 52087674_4 | p7_ddiag_AP | COMPLETED | 03:22:43 | 12:00:00 | 1 | 16G | 2.9 | 3.4 |
| 52087675_0 | p7_ddiag_GM | COMPLETED | 03:09:24 | 08:00:00 | 1 | 16G | 2.8 | 3.2 |
| 52087675_1 | p7_ddiag_GM | COMPLETED | 03:26:30 | 08:00:00 | 1 | 16G | 2.9 | 3.4 |
| 52087675_2 | p7_ddiag_GM | COMPLETED | 03:24:23 | 08:00:00 | 1 | 16G | 2.9 | 3.4 |
| 52087675_3 | p7_ddiag_GM | COMPLETED | 03:37:50 | 08:00:00 | 1 | 16G | 3.0 | 3.6 |
| 52087675_4 | p7_ddiag_GM | COMPLETED | 03:31:24 | 08:00:00 | 1 | 16G | 2.8 | 3.5 |
| 52087676_0 | p7_post | COMPLETED | 00:02:35 | 01:30:00 | 2 | 32G | 6.1 | 0.1 |
| 52088254_0 | lemcvt_a7_tests2 | COMPLETED | 00:03:15 | 00:20:00 | 2 | 8G | 1.1 | 0.1 |
| 52088675 | p7_wait | COMPLETED | 00:00:11 | 00:02:00 | 1 | 6000M | 0.0 | 0.0 |
| total | | | | | | | | 33.8 |

