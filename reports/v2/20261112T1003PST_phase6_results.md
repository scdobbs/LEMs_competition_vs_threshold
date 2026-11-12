# [2026-11-12 10:03 PST] Phase 6 results (addendum 6): CI, configs with the D0 bracket, the AP1 seam test, and the rerun

Directive: `reports/directives/2026-11-10T1715PST_refactor_v2_addendum_6.md` (outline fixed by its section 6).
Branch `v2`; code commits 7b8c9bf, 6ca929f, 92c4c30, 944e3c6, c7477f1, 4a2632b (first session, 2026-11-10/11) and the
commit carrying this report (second session, 2026-11-12: the merge task script, three mypy fixes, the appendix
generator's interval formatting). Environment `lemcvt` (Python 3.12, numpy 2.5.4, scipy 1.18.1); every job on `serc`,
1 CPU unless stated. Machine-generated tables: section 11 of this file (appendices B1-B10,
`reports/v2/scripts/make_phase6_results_tables.py`); figures under `reports/v2/figures_v2/` (`lemcvt figures all`,
job 52074991). Every number is tied to a run id under `$GROUP_HOME/lemcvt_results` or a Slurm job id.
Date: 2026-11-10 to 2026-11-12. **This is a stop for review** (section 10). Phase 7 waits.

## Summary

| Item | Result |
| --- | --- |
| 2 CI | Reproduced on Sherlock: ruff, mypy and the fast suite are green in the project environment (189 tests, job 52075532); the GitHub run on 92c4c30 still failed (mypy on 3.11, tests on 3.12; logs unreadable from Sherlock), so the trigger is `workflow_dispatch` only (944e3c6) and **the 2.5 gate is not passed**; open item |
| 1.3 configs | Written from run `20261111T002913Z_phase6_inputs_10tiles`: site m, K0 = U/k_s, D0_A (option (iii), L_f = 11 m) as primary and D0_B (C_ht at L_w) beside it; the bracket D0_B/D0_A is 1.95, 1.27, 1.00, 2.13, 1.18 on AP1-AP5 and 0.96-1.04 on GM |
| 3 AP1 seam | S = 18 417 cells (1.79 %). Masking S moves the joint A_crit by 0 (same step, inside the 1 % interval), K by 0.044 %, D by 0.235 %, the AD r by 0.0023 decades: **negligible** by the pre-registered rule; AP1 is used normally |
| 4 budget | 856 core-hours used, 2215 of limits submitted (every submission <= 300; at most 70 CPUs at once); the first-session estimate of 777 was exceeded in limits by resubmissions (7 h AP nonlinear limits too short by 2-4x; the joint arrays cancelled in a login-node outage and re-run as split parts) and in use by the AP3 nonlinear stage (26.7 h against 4.3 expected: 96 of 261 evaluations at the Picard cap) |
| 5.1 AD | Interior optima on every tile; K 4.0-4.7e-5 (AP), 0.95-1.5e-4 (GM) 1/yr, D 5.0e-4-1.0e-3 (AP), 4.1e-5-2.9e-4 (GM) m^2/yr; MSE 46-153 (AP), 49-174 (GM) m^2, i.e. 1.3-4.3x the linear joint MSE of the same tile |
| 5.2 linear partitioned | Joint: A_crit 1887-7232 (AP), 860-3853 (GM) m^2; **K_fit/K0 = 0.90-1.01 (AP), 1.00-1.06 (GM)**; D_fit 1.26-1.97e-2 (AP), 1.90-2.74e-2 (GM) m^2/yr, i.e. **4.3-8.3x D0_A and 2.4-8.3x D0_B on AP, 1.9-2.8x D0 on GM**. Stage 2 at (K0, D0_A): A_crit 355-823 (AP), 186-423 (GM) with MSE 7-13x (AP) and 1.9-2.9x (GM) the joint's |
| 5.3 nonlinear | At D0_A: A_crit 444-824 (AP), 250-522 (GM); MSE 193-288 (AP), 45-91 (GM), 15-25 % (AP) and 6-20 % (GM) below the linear stage 2 at the same D0 and still 6-10x (AP), 1.8-2.5x (GM) the linear joint. Cap hits 0 except AP3 (96 of 261, D0_A = 2.2e-3), AP2 (12), GM1 (2). At D0_B on AP: A_crit 554-1490, MSE 71-166 |
| 5.4 nonlinear joint | Optimum D within one grid step (x1.2) of the linear joint D on every tile; MSE within +-3 % of the linear joint; hillslope cells above 0.5 S_c at the optimum 1-11 %: the nonlinear hillslopes are planar at the optimum. The 1 % contour is a single grid cell on 7 tiles (grid too coarse for it). K check: AP5 A_crit unchanged, D +20 % (one step) at K x 0.95; GM1 A_crit 1758 -> 1200 (-32 %) at both K x 0.95 and x 1.05, MSE +4.5-5.6 % |
| 5.5 D-inf | AP1: A_crit 7178, K 5.99e-6, D 1.66e-2, MSE 40.9 (D8: 6889, 4.00e-6, 1.70e-2, 22.4); GM1: 210.5, 6.25e-5, 1.42e-2, 52.5 (D8: 1241, 4.87e-5, 1.95e-2, 24.7); MSE ratios 1.83 and 2.13 |
| 4 / 5.7 curvature | On the option (iii) pixels, C_obs/C_model at 11 m is 4.3-8.3 (AP) and 1.9-2.8 (GM) for the linear joint surface, 3.9-7.7 and 2.0-2.9 for the nonlinear joint, 0.22-0.39 and 0.04-0.10 for AD; the fitted partitioned surfaces have -laplacian = U/D_fit at every window (planar hillslopes), so the ratio follows C_obs(L): on AP it falls from 6-12 at 7 m to 2.0-4.0 at 51 m, on GM it is flat at 1.3-3.1. No option (iii) pixel is a channel cell in any fitted model |
| 5.8 record | 692 rows (`tables/old_vs_new_record.csv`), built by one command; largest systematic ratios: AP K0 and joint K x2.3-5.0 on AP1-AP3 (units, m, measurement), AP nonlinear A_crit x0.05-0.17 (unconverged v1, D0 bracket), GM1/GM3/GM4 joint A_crit x0.13-0.52 (optimizer), AD D on GM1/GM3 x0.01-0.02 (optimizer) |
| Decisions | section 10: the nonlinear joint D grid (refine or accept one-step resolution), the GM1 K sensitivity, `fig3_deltas`, CI, Phase 7 |

## 1. State check and the CI fix (section 2)

**State.** `v2` was at 7ee3c52 = `origin/v2` at the start of addendum 6 (the Phase 6.0 report commit); the test suite
had passed on `serc` at 6f956a6 (job 51836825, 182 tests) and at 92c4c30 (job 51843914: 188 tests, with the nine
tests added under this addendum, run with `LEMCVT_DATA` unset); no job from an earlier session was running. The
addendum and the repo did not disagree anywhere I could check. The work spans two agent sessions: the first ended
after submitting the merge arrays (section 4); the second (2026-11-12, from 09:40 PST) found `v2` at 4a2632b with
`origin/v2` at 92c4c30 (three commits unpushed), every compute array finished, the merges failed and nothing else
running. At 4a2632b ruff and the fast suite were clean (job 52074889: 189 passed) and **mypy reported three errors**
introduced by that commit (the split/merge code: a list typed from `DataFrame.to_dict` and a `tuple[int, ...]`
split argument); fixed in code in this session, no ignore added; at the reported tree ruff, mypy and the fast suite
are clean (job 52075532: 189 passed, `LEMCVT_DATA` unset).

**What CI was doing.** The Actions API (no `gh` on Sherlock; the public endpoints suffice for run and step
conclusions) shows every push to `v2` failing at the **ruff** step of `test (3.11)` after about 36 s, which cancels
`test (3.12)` under `fail-fast`: runs 42755697228 (0fde82f), 42923822289 (4f9d584) and 42934047292 (7ee3c52) all end
`Install: success, Lint (ruff): failure, mypy: skipped, tests: skipped`. The install step succeeds on both versions,
so Python 3.11 is supportable and `requires-python` stays `>=3.11`.

**Reproduction on Sherlock (2.1).** The workflow's install step cannot be reproduced on Sherlock: in fresh 3.11 and
3.12 micromamba environments, `pip install -e ".[dev]"` from PyPI fails for both (job 51840114, 17 min) because the
CentOS 7 login and compute nodes (glibc 2.17) cannot use the current `manylinux_2_28` wheels and pip falls back to
building scipy from source. The three later steps were reproduced with the same tools in the project environment
(ruff 0.17.4, mypy 2.4.1, pytest; jobs 51841306, 51842172, 51843294, 51843914), with `LEMCVT_DATA` unset:

| step | before the fix | cause | after the fix (job 51843914) |
| --- | --- | --- | --- |
| ruff | 6 errors | all in code added this session (unsorted imports, unused imports and loop variables) | clean |
| mypy | 133 errors in 19 files, then 93 after the first pass | 23 attribute accesses on the model closures (`last_picard`, `reduce`, ...), 13 trajectory records typed `dict[str, float]`, the rest inferred dict and variable types, reused variable names, `Series.idxmin` indexing | clean; no ignore added |
| fast suite | 11 failures | `from tests.synthetic_tile import ...` in `test_models.py` resolves under `python -m pytest` (the earlier jobs) but not under `pytest` as the workflow runs it; no test needs `$LEMCVT_DATA` | 188 passed |

**Fixes (2.2).** Ruff: the findings fixed in code. Mypy: `Protocol` types for the two model closures that carry
attributes (`NonlinearModel`, `DinfPartitionedModel`, returned through `cast`), `Trajectory.record` takes
`dict[str, Any]`, explicit `dict[str, Any]` annotations where result dicts were inferred too narrowly, renamed
variables that were reused with a different type, `np.argmin` instead of `Series.idxmin` indexing, an `assert` where
an optional is known to be set; no entry was added to the ignore lists. Tests: the import path. `fail-fast: false`
set (2.3).

**The GitHub run on the pushed fix (2.5), and the trigger change (2.4).** Run 42940255348 on 92c4c30: `test (3.12)`
passes ruff and mypy and **fails in the fast test suite**; `test (3.11)` passes ruff and **fails at mypy**. The job logs
need an authenticated request (HTTP 403 from Sherlock), so the reasons cannot be read here; on Sherlock the same three
steps pass. Two differences between the environments are candidates and are not verified: the CI install has no
`numba` (an optional extra; the package falls back to the Python loops and every test passes either way on Sherlock)
and different numpy and pandas-stubs versions on 3.11. Per 2.4 the workflow trigger is now `workflow_dispatch` only
(commit 944e3c6), so no push after this section triggers a failing run, and **the CI fix remains an open item**: read
the logs of run 42940255348 (the Actions page), fix, restore the push trigger. The gate of 2.5 is therefore not
passed; what is established is that ruff, mypy and the fast suite are green on Sherlock with the workflow's commands.

## 2. Configs written (section 1.3), with the D0 bracket table

`lemcvt phase6 write-configs --force` (commit 7b8c9bf; `--force` because the recorded decision of the option (iii)
test is "failed" and decision 1 of the addendum overrides the stop) wrote the ten tile configs from the recorded
measured-inputs run `20261111T002913Z_phase6_inputs_10tiles` (addendum 5) with: the site m (AP 0.47, GM 0.31);
K0 = U/k_s at theta = m over the network above 10^3 m^2 for every tile (AP3 included: decision 4 of addendum 5);
D0_A = U/C_ht(L_f), option (iii) at L_f = 11 m, as the primary `D0`; D0_B = U/C_ht(L_w) on the same option (iii) pixel
set with L_w the largest tested window not exceeding the tile's median crest width, from A2 of the Phase 6.0 report
(`D0_B`, with `--d0-variant B` selecting it); the Picard cap 100; `param_z_units: m` with the Phase 5.3 metre bounds;
the v1 model block under `v1_reference.model` (feet runs still reproduce v1 with `--units feet`; checked by reloading
every config both ways). AP1 keeps its stored grids (decision 2). Every value carries its provenance (run id, git
hash, n_eff, bootstrap intervals, S(m) intervals, the window test) in `measured.provenance` (B1).

| tile | m | k_s at 10^3 (n_eff) | K0 (1/yr) | C_ht_A (1/m) | D0_A (m^2/yr) | L_w (m) | C_ht_B (1/m) | D0_B (m^2/yr) | D0_B / D0_A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 0.47 | 12.42 (1.5) | 4.43e-6 | 1.53e-2 | 3.59e-3 | 31 | 7.88e-3 | 6.98e-3 | 1.95 |
| AP2 | 0.47 | 11.58 (1.9) | 4.75e-6 | 1.71e-2 | 3.22e-3 | 15 | 1.35e-2 | 4.08e-3 | 1.27 |
| AP3 | 0.47 | 11.53 (1.1) | 4.77e-6 | 2.55e-2 | 2.16e-3 | 11 | 2.55e-2 | 2.16e-3 | 1.00 |
| AP4 | 0.47 | 11.93 (2.3) | 4.61e-6 | 1.52e-2 | 3.63e-3 | 41 | 7.12e-3 | 7.72e-3 | 2.13 |
| AP5 | 0.47 | 10.73 (1.6) | 5.13e-6 | 1.88e-2 | 2.92e-3 | 15 | 1.59e-2 | 3.46e-3 | 1.18 |
| GM1 | 0.31 | 3.03 (3.5) | 4.85e-5 | 1.63e-2 | 9.00e-3 | 15 | 1.57e-2 | 9.35e-3 | 1.04 |
| GM2 | 0.31 | 2.67 (3.7) | 5.51e-5 | 1.51e-2 | 9.77e-3 | 15 | 1.45e-2 | 1.02e-2 | 1.04 |
| GM3 | 0.31 | 2.29 (2.7) | 6.41e-5 | 1.45e-2 | 1.01e-2 | 11 | 1.45e-2 | 1.01e-2 | 1.00 |
| GM4 | 0.31 | 4.23 (1.4) | 3.48e-5 | 1.49e-2 | 9.86e-3 | 15 | 1.56e-2 | 9.44e-3 | 0.96 |
| GM5 | 0.31 | 2.93 (1.2) | 5.02e-5 | 1.94e-2 | 7.59e-3 | 15 | 1.96e-2 | 7.50e-3 | 0.99 |

The bracket is a factor 1.2-2.1 wide on AP1, AP2, AP4 and AP5 and collapses on AP3 (L_w = L_f: its 12.5 m median
crest width admits no other window) and on every GM tile (within 4 %), where option (iii) stands. The stage-2
profile at D0_B costs nothing extra: it is the same reductions evaluated at (K0, D0_B) (`mse_stage2_B`, 5.2), and the
nonlinear stage and the sweep were run at D0_B on AP as separate jobs.

## 3. AP1 seam test (section 3)

Run `20261111T013013Z_seam_test_AP1` (job 51842989, 5 h 20 min; B2). **The affected set S** is 18 417 cells, 1.79 % of the
1 030 225 valid cells: the band of rows 385-395 (11 165 cells), every cell downstream of the 345 seam cells along the
stored directions (9 057, reaching the tile's south edge along the paths), and the 699 cells whose stored area is
inconsistent with their in-tile donors (the three parts overlap). The linear joint profile (2000 thinned areas plus
the refinement around both minima), the stage-2 profile at D0_A and the AD profile (200-point r grid plus Brent) were
each evaluated twice at the same points, with the misfit over all valid cells and over all valid cells except S, from
the same reductions (one LU per A_crit, one per r; only the Gram matrix of the misfit changes with the mask).

| profile | misfit over | A_crit (m^2) | K (1/yr) | D (m^2/yr) | MSE (m^2) | 1 % interval |
| --- | --- | --- | --- | --- | --- | --- |
| linear joint | all cells | 6888.6 | 3.9952e-6 | 1.6990e-2 | 22.384 | 6540-6930 |
| linear joint | all except S | 6888.6 | 3.9934e-6 | 1.6950e-2 | 22.423 | 6540-6930 |
| stage 2 at D0_A | all / except S | 823.3 / 823.3 | - | - | 289.23 / 288.65 | 823-830 (both) |
| AD | all | - | 3.9950e-5 | 5.4467e-4 | 59.018 | r = 7.335e-2 |
| AD | all except S | - | 4.0016e-5 | 5.4848e-4 | 59.179 | r = 7.296e-2 |

**Rule (3.4): the seam is negligible.** The masked joint A_crit is the same staircase step as the unmasked one and lies
inside the unmasked 1 % interval; K moves by 0.044 % (threshold 2 %), D by 0.235 % (5 %), the AD r by 0.0023 decades
(0.1). The profile shapes differ by at most 4.2 % (joint) and 2.5 % (stage 2) of MSE/MSE_min, mostly far from the
minima. AP1 is therefore used normally in every section below, with this note; no AP site statistic needs a
with-and-without-AP1 variant, and the N9 record in `docs/numerics_changes.md` stands as the description of the
defect. (The masked MSE is larger than the unmasked one on the joint profile because S holds cells near the channel
network's south exit that the model fits better than average; it is a 0.2 % effect.)


## 4. Budget and runs

**Budget (1.5; A4 of the Phase 6.0 report, updated in the first session).** With the D grid of 1.4 (20 values from
1.5e-3 to 5e-2 m^2/yr) and the additions of addendum 6 (the seam test, the D0_B nonlinear and sweep runs), the
first-session estimate was **777 core-hours of limits (495 expected)**. Steve's approval of more than 900 core-hours
stands recorded. What was actually submitted and used, from `sacct` over every job of this addendum (CI, Phase 6,
merges, post-processing, test jobs; limit core-hours = time limit x CPUs x tasks, used = elapsed x CPUs):

| job | name | tasks | states | CPUs x mem | limit core-h | used core-h | max RSS (GB) |
|---|---|---|---|---|---|---|---|
| 51840114 | lemcvt_a6_ci | 1 | COMPLETED 1 | 2 x 16G | 3 | 0.6 | 2.5 |
| 51841306 | lemcvt_a6_cisteps | 1 | COMPLETED 1 | 2 x 16G | 1 | 0.1 | 1.0 |
| 51842172 | lemcvt_a6_mypy | 1 | COMPLETED 1 | 2 x 16G | 1 | 0.1 | 1.0 |
| 51842893 | p6_ad | 10 | COMPLETED 10 | 1 x 8G | 30 | 17.4 | 3.7 |
| 51842895 | p6_part_AP | 5 | FAILED 5 (fault 1) | 1 x 8G | 100 | 0.0 | 0.2 |
| 51842898 | p6_part_GM | 5 | FAILED 5 (fault 1) | 1 x 8G | 55 | 0.0 | 0.3 |
| 51842902 | p6_nl_AP | 5 | TIMEOUT 1, CANCELLED 4 | 1 x 8G | 35 | 19.7 | 2.8 |
| 51842904 | p6_nl_GM | 5 | COMPLETED 5 | 1 x 8G | 55 | 43.1 | 2.8 |
| 51842906 | p6_nl_AP_D0B | 5 | COMPLETED 2, CANCELLED 3 | 1 x 8G | 35 | 18.1 | 2.8 |
| 51842942 | p6_dinf | 2 | COMPLETED 2 | 2 x 32G | 60 | 52.9 | 3.5 |
| 51842944 | p6_sweep | 10 | COMPLETED 10 | 2 x 24G | 20 | 0.2 | 1.0 |
| 51842953 | p6_sweep_AP_D0B | 5 | COMPLETED 5 | 2 x 24G | 10 | 0.1 | 1.0 |
| 51842965 | p6_util | 1 | CANCELLED (fault 1) | - | 0 | 0.0 | - |
| 51842989 | p6_seamtest | 1 | COMPLETED 1 | 1 x 16G | 18 | 5.3 | 3.2 |
| 51843294 | lemcvt_a6_mypy2 | 1 | COMPLETED 1 | 2 x 16G | 1 | 0.1 | 0.9 |
| 51843914 | lemcvt_a6_cisteps2 | 1 | COMPLETED 1 | 2 x 16G | 1 | 0.1 | 0.5 |
| 51845546 | p6_part_AP | 5 | COMPLETED 5 | 1 x 8G | 100 | 66.6 | 2.9 |
| 51845553 | p6_part_GM | 5 | COMPLETED 5 | 1 x 8G | 55 | 22.3 | 2.9 |
| 51845561 | p6_nlj_GM | 5 | CANCELLED 5 (outage) | 1 x 8G | 175 | 29.9 | 2.1 |
| 51845600 | p6_nlj_GM1_k095 | 1 | CANCELLED 1 (outage) | 1 x 8G | 35 | 6.0 | 2.1 |
| 51845602 | p6_nlj_GM1_k105 | 1 | CANCELLED 1 (outage) | 1 x 8G | 35 | 6.0 | 2.3 |
| 51845603 | p6_util | 1 | COMPLETED 1 | 3 x 64G | 9 | 0.7 | 33.6 |
| 51860388 | p6_nl_AP_16h | 4 | CANCELLED 2, COMPLETED 2 | 1 x 8G | 64 | 45.6 | 2.5 |
| 51860389 | p6_nl_AP_D0B_16h | 3 | COMPLETED 2, CANCELLED 1 | 1 x 8G | 48 | 28.5 | 2.9 |
| 51906822 | p6_nlj_GM123 | 15 | COMPLETED 15 | 1 x 8G | 300 | 103.5 | 2.6 |
| 51906823 | p6_nlj_GM45 | 10 | COMPLETED 10 | 1 x 8G | 200 | 79.7 | 2.6 |
| 51906824 | p6_nlj_GM1_K | 10 | COMPLETED 10 | 1 x 8G | 200 | 64.6 | 2.5 |
| 51906827 | p6_nlj_AP | 15 | COMPLETED 15 | 1 x 8G | 300 | 128.0 | 2.8 |
| 51906828 | p6_nlj_AP5_K | 6 | COMPLETED 6 | 1 x 8G | 120 | 35.6 | 2.7 |
| 51906829/30/33 | p6_nlj_merge_GM/AP/K | 14 | FAILED 14 (fault 2) | 1 x 16G | 14 | 0.0 | 0.1 |
| 51906834 | p6_nl_AP1_16h | 1 | COMPLETED 1 | 1 x 8G | 16 | 8.5 | 2.8 |
| 51906835 | p6_nl_AP3_30h | 1 | COMPLETED 1 | 1 x 8G | 30 | 26.7 | 2.8 |
| 51906836 | p6_nl_AP3_D0B_30h | 1 | COMPLETED 1 | 1 x 8G | 30 | 26.7 | 2.8 |
| 51926513 | p6_nl_AP2_30h | 1 | COMPLETED 1 | 1 x 8G | 30 | 18.8 | 2.3 |
| 52074315/23/27 | p6_nlj_merge_GM/AP/K | 14 | FAILED 14 (fault 3) | 1 x 16G | 14 | 0.1 | 0.6 |
| 52074889 | lemcvt_a6_tests3 | 1 | COMPLETED 1 | 2 x 8G | 1 | 0.1 | 1.0 |
| 52074963/77/88 | p6_nlj_merge_GM/AP/K | 14 | COMPLETED 14 | 1 x 16G | 14 | 0.2 | 2.4 |
| 52074991 | p6_post | 1 | COMPLETED 1 | 1 x 32G | 3 | 0.1 | - |
| 52075532 | lemcvt_a6_tests4 | 1 | COMPLETED 1 | 2 x 8G | 1 | 0.1 | - |
| **total** | | | | | **2215** | **856** | |

The largest single submissions are the two 15-task nonlinear joint arrays at 300 core-hours of limits each (the
per-submission line); every array was throttled (`%10`); at most about 70 single-CPU tasks plus the sweep's 2-CPU
tasks ran at once; all on `serc` with explicit limits; memory from measurements (max RSS 2.1-3.7 GB against 8 GB,
33.6 GB for `d8_vs_dinf` against 64 GB). Storage (`sh_quota`, 2026-11-12 09:58 PST): `$GROUP_HOME` 266.1 GB of 1 TB
(`lemcvt_results` holds every run); `$HOME` 10.9 of 15 GB; `$SCRATCH` 804 GB (job logs and rendered scripts only);
`$OAK` 16.5 of 20 TB. Logs under `$SCRATCH/lemcvt_jobs`, every submission appended to
`$SCRATCH/lemcvt_jobs/phase6_submit.log`.

**Why the limits exceeded the estimate.** (a) The 7 h limit of the AP nonlinear stage (estimate 4.3 h) was short by
2-4x: the Picard iteration at D0_A on AP runs to 10-15 iterations per converged evaluation and to the cap of 100 on
AP3 (96 of 261 evaluations at D0_A = 2.2e-3 m^2/yr) and AP2 (12); one task timed out, the others were cancelled at
3 h and resubmitted at 16 h, and AP2 and AP3 again at 30 h (AP3 took 26.7 h; see section 5.3). (b) The first
nonlinear joint arrays (51845561 and the GM1 K checks) were cancelled after 6 h in what the first session's log
calls a login-node outage (2026-11-11 05:07 PST) and resubmitted as parallel parts of the A_crit grid (3 per AP tile,
5 per GM tile, commit 4a2632b, exact merge tested), which is why their limits appear twice. (c) 28 core-hours of
limits (0.1 used) went to the two failed rounds of merge arrays. Used core-hours (856) are below the 900 line;
limits (2215) are not, and the approval covered that.

**Four submission faults, all the agents'.** (1) The first partitioned arrays (51842895/51842898) and the utilities
job chained two commands inside a quoted `bash -c` string, which `micromamba run` re-splits; the tasks died after
10 s and Slurm cancelled their dependents. Resubmitted as task scripts (6ca929f) 25 min later; nothing lost. (2) The
first session wrote the per-part task script for the split nonlinear joint profile but never created the merge task
script `reports/v2/scripts/phase6_nlj_merge_task.sh` that the merge arrays call; all 14 merge tasks died with exit 127
(2026-11-11 15:20 to 2026-11-12 04:29) and Slurm cancelled the post-processing job (51906837, 51926515) that depended
on them. The session ended before this was seen. (3) The second session created the script and resubmitted, from a
shell without `LEMCVT_RESULTS` and `LEMCVT_DATA` exported; `lemcvt slurm submit` renders the submitting shell's values
into the job script, so the tasks ran with an empty results root and found no runs (52074315/23/27, exit 1, 18 s
each; nothing written). (4) Resubmitted with the variables exported (README): 14 of 14 merges completed in 31-58 s
each, and the post-processing job (52074991: curvature comparison, slope-area at the new optima, figures, tables,
record) completed in 5 min. The merge reads the parts, concatenates the rows, re-solves the optimum once and writes
the unsplit layout (`*_nonlinear_joint_merged`); the test of 4a2632b shows the merged rows and optimum equal an
unsplit run's.

## 5. Results by model

All runs: ten tiles, metres, v2 optimizer, site m (AP 0.47, GM 0.31), the configs of section 2. Tables: B3-B7.
Comparisons with the checkpoint 4b/5 values (metres, site m, v1 K0/D0) are stated where the quantity does not depend
on K0 or D0.

### 5.1 AD profile (B3; array 51842893, 1.5-1.7 h per tile)

The profile over r = K/D (200-point grid, Brent) has an interior optimum on every tile. AP: r = 0.046-0.091,
K = 4.0-4.7e-5 1/yr, D = 5.0e-4-1.0e-3 m^2/yr, MSE 45.7 (AP4) to 152.8 (AP3) m^2. GM: r = 0.50-3.3, K = 0.95-1.54e-4,
D = 4.1e-5-2.9e-4, MSE 49.0 (GM1) to 174.5 (GM4). The 1 % intervals in r are 0.0-0.4 decades wide (a single grid
point on AP4 and AP5). These equal the checkpoint's AD results on GM (the AD stage does not use K0 or D0) and are
the AP results converted to metres. Against the linear joint of the same tile (5.2), the AD MSE is 1.3x (AP4) to
4.3x (GM4) larger; on the other nine tiles the ratio is 1.96-4.3. The AD forward run at the measured (K0, D0_A), which
`fig_slope_area` needs, has MSE 1.5e5-8.7e5 m^2 on every tile: the measured K0 is the channel chi-slope parameter and
the AD model with D = D0_A at that K puts the whole surface far from the DEM; the fitted AD K is 8.6-10.2x K0 on AP and
2.1-3.1x on GM, and the fitted AD D is 0.15-0.28 x D0_A on AP and 0.003-0.03 x D0 on GM.

### 5.2 Linear partitioned model (B4; arrays 51845546/51845553, 6.6-11.8 h per AP tile, 0.2-2.1 h per GM tile)

**Joint profile** (thin 2000, staircase refinement, every unique area in the 5 % interval: 2132-11 821 evaluations
per tile, of which 107-9452 are the fill). AP: A_crit 1887 (AP5) to 7232 (AP4) m^2, K 4.00-5.12e-6 1/yr,
D 1.26-1.97e-2 m^2/yr, MSE 22.4-42.1 m^2. GM: A_crit 860 (GM3) to 3853 (GM4), K 3.64-6.81e-5, D 1.90-2.74e-2, MSE
24.7-40.7. The direct check (a full forward solve at the optimum against the reduction) agrees to 1e-13-5e-12
relative. The 1 % intervals are staircase steps, often in islands, of 1-70 % total width in A_crit (GM5 1819-3130); the K
range over the 1 % interval is within 4 % of the optimum (AP4), the D range within 14 % (GM5). These are the checkpoint's site-m joint
values (A9) re-found with the finer grid: A_crit differs by 0-16 % (AP1 6889 against 6686; AP2 3121 against 3117;
GM1 1241 against 1135; GM5 1885 against 2247), K and D by under 4 %.

**K recovery.** K_fit/K0 (K0 = U/k_s at theta = m above 10^3 m^2): AP1 0.902, AP2 0.968, AP3 1.012, AP4 0.981,
AP5 0.998; GM1 1.002, GM2 1.000, GM3 1.062, GM4 1.046, GM5 1.043. Both numbers come from the same chi integration
(N6), so this is the statement that the joint optimum puts the channel elevations at the chi-z slope of the
network above 10^3 m^2, to within 10 % on AP1 and 6 % elsewhere.

**D recovery** (also 6): D_fit/D0_A = 4.74, 5.22, 8.31, 5.44, 4.31 on AP1-AP5 and 2.16, 2.81, 1.90, 2.22, 2.51 on
GM1-GM5; D_fit/D0_B = 2.43, 4.12, 8.31, 2.56, 3.64 on AP. The linear joint D is 1.3-2.0e-2 m^2/yr on AP and
1.9-2.7e-2 on GM, a narrow range across all ten tiles, where D0_A spans 2.2-3.6e-3 (AP) and 7.6-10.1e-3 (GM).

**Stage 2** (A_crit profile at the measured K0 and D0). At D0_A: A_crit 355-823 m^2 on AP with MSE 226-341 (7-13x
the joint MSE of the same tile), 186-423 m^2 on GM with MSE 51-114 (1.9-2.9x). At D0_B on AP: A_crit 442-1090 with
MSE 81-246 (AP1 86.4 against 289.2 at D0_A; AP4 81.1 against 255.2; AP3 identical, L_w = L_f). The stage-2 1 %
intervals are 1-25 % wide. The stage-2 A_crit is 5-14x (AP) and 4-17x (GM) below the joint A_crit: with D fixed
at a value 2-8x below D_fit, the profile buys back hillslope misfit by extending the channel network.

### 5.3 Nonlinear stage (B5; S_c = 1.27, Picard cap 100; 200 grid + refinement)

At D0_A (all tiles): A_crit 824 (AP1), 682 (AP2), 480 (AP3), 762 (AP4), 444 (AP5); 272, 250, 254, 522, 261 (GM1-GM5).
MSE 193-288 on AP, 44.7-91.4 on GM: 15-25 % (AP) and 6-20 % (GM) below the linear stage 2 at the same (K0, D0_A), and
still 6-10x (AP) and 1.8-2.5x (GM) the linear joint. Cap hits: 0 of 218-261 evaluations on seven tiles, 2 on GM1,
12 on AP2 and **96 of 261 on AP3** (D0_A = 2.16e-3, the smallest D of the set; its optimum converged in 15
iterations; its 1 % interval is 401-511). Hillslope cells steeper than 0.5 S_c at the
optimum: 21-39 % on AP, 7-29 % on GM; steeper than 0.9 S_c: 1.7-5.9 % on AP, 0.8-2.3 % on GM. At D0_B on AP: A_crit
1490, 731, 480, 1052, 554 with MSE 70.7, 149, 288, 74.9, 166; the > 0.5 S_c fraction falls to 5-23 %. The nonlinear
A_crit is 0-42 % above the linear stage-2 A_crit at the same D0 on every tile (AP1 824 against 823; AP2 682
against 495; GM5 261 against 186). Against the checkpoint's AP nonlinear profiles in metres at the v1 D0 (A7:
A_crit 4196, 2502, 1800, 995, 1354; 0 cap hits), the present values are 3-6x smaller, which is the D0 change (v1 D0
8.2-9.3e-3 against D0_A 2.2-3.6e-3); the GM values equal the checkpoint's to within the refinement (GM1 272 against
the cap-50 value).

Wall time: 7.0-10.3 h on GM, 8.5-26.7 h on AP (AP3 26.7 h, AP2 18.8 h at D0_A), against the 4.3-7.2 h estimated;
this is the cost overrun of section 4.

### 5.4 Nonlinear joint profile over (A_crit, D) (B6; figure `a5_nonlinear_joint.png`)

K fixed at the linear joint optimum of each tile; 30 log-spaced unique areas spanning the linear joint 5 % interval
widened x3 on each side; 20 log-spaced D values from 1.5e-3 to 5e-2 m^2/yr (a factor 1.203 between neighbours);
600 Picard solves per tile, run as parallel parts and merged (section 4). Results per tile:

| tile | K (1/yr) | optimum A_crit (m^2) | optimum D (m^2/yr) | MSE (m^2) | linear joint A_crit, D, MSE | MSE ratio NL/lin | 1 % contour | 5 % contour (grid points) | cap hits (of 600) | hillslope > 0.5 S_c |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 4.00e-6 | 6973 | 1.652e-2 | 22.07 | 6889, 1.699e-2, 22.38 | 0.986 | 1 cell | 5854-6973, D 1.65e-2 (3) | 36 | 0.011 |
| AP2 | 4.60e-6 | 3160 | 1.652e-2 | 22.39 | 3121, 1.680e-2, 22.46 | 0.997 | 1 cell | 2901-3441, D 1.65e-2 (3) | 47 | 0.013 |
| AP3 | 4.83e-6 | 4621 | 1.652e-2 | 42.14 | 4742, 1.791e-2, 42.12 | 1.000 | 1 cell | 4621-7246, D 1.65-1.99e-2 (6) | 85 | 0.016 |
| AP4 | 4.52e-6 | 7877 | 1.987e-2 | 34.38 | 7232, 1.974e-2, 35.16 | 0.978 | 1 cell | 7213-8601, D 1.99e-2 (3) | 87 | 0.010 |
| AP5 | 5.12e-6 | 1906 | 1.142e-2 | 28.48 | 1887, 1.258e-2, 27.87 | 1.022 | 1 cell | 1481-2669, D 1.14-1.37e-2 (6) | 34 | 0.021 |
| GM1 | 4.87e-5 | 1758 | 1.987e-2 | 25.21 | 1241, 1.948e-2, 24.69 | 1.021 | 1597-1758 | 1452-1758, D 1.99e-2 (3) | 151 | 0.034 |
| GM2 | 5.51e-5 | 3880 | 2.874e-2 | 26.51 | 3509, 2.743e-2, 26.07 | 1.017 | 1 cell | 3066-4196, D 2.39-2.87e-2 (5) | 196 | 0.018 |
| GM3 | 6.81e-5 | 907 | 1.987e-2 | 33.89 | 860, 1.923e-2, 33.03 | 1.026 | 717-907, D 1.65-1.99e-2 | 505-1628, D 1.65-2.39e-2 (12) | 129 | 0.013 |
| GM4 | 3.64e-5 | 3901 | 1.987e-2 | 40.97 | 3853, 2.189e-2, 40.74 | 1.005 | 1 cell | 3231-3901, D 1.99e-2 (3) | 213 | 0.107 |
| GM5 | 5.23e-5 | 2852 | 1.987e-2 | 32.14 | 1885, 1.903e-2, 32.21 | 0.998 | 1449-3193, D 1.65-1.99e-2 | 1295-3573, D 1.65-1.99e-2 (10) | 158 | 0.050 |

- **The optimum D equals the linear joint D to within one grid step on every tile** (ratios 0.91-1.05; the grid
  values 1.142e-2, 1.652e-2, 1.987e-2 and 2.874e-2 are the four neighbours of the linear D values) and the MSE is
  within -2.2 % to +2.6 % of the linear joint's. No optimum is on a grid edge.
- **The nonlinearity is barely engaged at the optimum**: 1.0-3.4 % of hillslope cells are steeper than 0.5 S_c
  (GM4 10.7 %, GM5 5.0 %) and 0.05-0.5 % steeper than 0.9 S_c; the Picard iteration converged in 6-9 iterations
  there. The cap hits (34-87 on AP, 129-213 on GM, of 600) all lie at D <= 4-5e-3 m^2/yr, the lower rows of the grid
  (figure), and none inside any 5 % contour.
- **The grid is too coarse for the 1 % contour**: it is a single grid cell on 7 of 10 tiles and one step in D on
  the other three; the 5 % contour spans 3-12 grid points (1-3 D values, 2-9 A_crit values). The 5 % contour spans one
  to three D values (one step is 20 %) and a factor 1.2-3.2 in A_crit.
- **A_crit** moves against the linear joint by +1 to +9 % on AP1-AP5 (AP3 -3 %) and by +1 to +51 % on GM (GM1 1758
  against 1241, GM5 2852 against 1885; the 5 % contours of those two tiles reach down to 1452 and 1295).

**K check** (same grid at K x 0.95 and x 1.05): AP5 keeps A_crit = 1906 at both; at K x 0.95 the optimum D moves one
step up (1.142e-2 -> 1.374e-2, +20 %) and the MSE is +2.0 %, at x 1.05 D and A_crit are unchanged and the MSE +0.5 %.
GM1 moves from A_crit 1758 to 1200 (-32 %) at both K factors, with D unchanged at x 0.95 and one step down
(1.652e-2, -17 %) at x 1.05; MSE +5.6 % and +4.5 %. The GM1 optimum in A_crit is therefore not stable against a 5 %
change of K (its 5 % contour at K x 1 reaches 1452, not 1200); the AP5 optimum is.

### 5.5 D-infinity joint profile (B7; array 51842942, 2 CPUs, 12.8-13.6 h)

Rebuilt partitioned D-infinity model (N8), v2 joint profile over the unique D-infinity areas (2286 and 2400 of
162 458 and 89 530 used), one LU per A_crit, direct check 4e-14. AP1: A_crit 7178 m^2 (1 % interval 7034-7615 and
five islands up to 8415), K 5.99e-6, D 1.66e-2, MSE 40.9 against the D8 joint's 6889, 4.00e-6, 1.70e-2, 22.4
(MSE ratio 1.83). GM1: A_crit 210.5 (1 % 166-269), K 6.25e-5, D 1.42e-2, MSE 52.5 against 1241, 4.87e-5, 1.95e-2,
24.7 (ratio 2.13). No channel cell lies below all of its receivers at either optimum (0 of 5131 and 0 of 29 409).
The D-infinity K is 1.5x (AP1) and 1.3x (GM1) the D8 K, which with K0 = U/k_s measured on the D8 network makes
K_fit/K0 = 1.35 and 1.29; the D-infinity D is within 3 % (AP1) and 27 % (GM1) of the D8 D. On GM1 the D-infinity
optimum is at an A_crit 6x below the D8 one, with a flat profile (1 % interval 166-269). At the config parameters
(B9, `d8_vs_dinf`, A_crit 5000), the rebuilt D-infinity partitioned MSE is 5191 (AP1) and 2132 (GM1) against 2879
and 1089 for D8 and 54 942 and 2790 for the v1 D-infinity model.

## 6. Curvature comparison (section 4) and D-recovery diagnostics (5.7)

Run `20261112T175547Z_curvature_compare_10tiles` (job 52074991); B8; figure `a6_curvature_compare.png` (one panel
per tile, the four curves). Same estimator on every surface: the median -laplacian of the quadratic fit at
L = 7, 9, 11, 15, 21, 25, 31, 41 and 51 m on the option (iii) pixel set of the tile (533-4711 pixels), for the
observed DEM (A2 of the Phase 6.0 report), the linear joint surface, the nonlinear joint surface (merged runs, 5.4)
and the AD optimum surface. No pass rule. The ratio C_obs(L)/C_model(L):

| tile | surface | 7 | 9 | 11 | 15 | 21 | 25 | 31 | 41 | 51 m | channel fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | linear joint | 7.07 | 5.58 | 4.74 | 3.66 | 2.84 | 2.68 | 2.43 | 2.18 | 2.07 | 0 |
| AP1 | nonlinear joint | 6.91 | 5.45 | 4.63 | 3.58 | 2.78 | 2.62 | 2.38 | 2.13 | 2.03 | 0 |
| AP1 | AD | 0.32 | 0.27 | 0.24 | 0.21 | 0.20 | 0.21 | 0.22 | 0.25 | 0.29 | - |
| AP2 | linear joint | 7.56 | 6.11 | 5.22 | 4.12 | 3.69 | 3.44 | 3.39 | 3.21 | 3.06 | 0 |
| AP2 | nonlinear joint | 7.47 | 6.04 | 5.16 | 4.07 | 3.65 | 3.41 | 3.36 | 3.17 | 3.03 | 0 |
| AP2 | AD | 0.30 | 0.25 | 0.22 | 0.20 | 0.21 | 0.21 | 0.23 | 0.27 | 0.33 | - |
| AP3 | linear joint | 12.12 | 9.93 | 8.31 | 6.54 | 5.16 | 4.78 | 4.44 | 4.16 | 3.96 | 0 |
| AP3 | nonlinear joint | 11.23 | 9.20 | 7.69 | 6.05 | 4.77 | 4.42 | 4.10 | 3.85 | 3.66 | 0 |
| AP3 | AD | 0.43 | 0.37 | 0.32 | 0.28 | 0.27 | 0.28 | 0.32 | 0.39 | 0.44 | - |
| AP4 | linear joint | 9.17 | 6.90 | 5.44 | 3.98 | 3.10 | 2.88 | 2.71 | 2.56 | 2.51 | 0 |
| AP4 | nonlinear joint | 9.26 | 6.96 | 5.49 | 4.02 | 3.13 | 2.91 | 2.74 | 2.59 | 2.54 | 0 |
| AP4 | AD | 0.61 | 0.47 | 0.39 | 0.32 | 0.28 | 0.28 | 0.29 | 0.33 | 0.38 | - |
| AP5 | linear joint | 6.01 | 4.90 | 4.31 | 3.64 | 3.26 | 3.12 | 2.98 | 2.79 | 2.64 | 0 |
| AP5 | nonlinear joint | 5.47 | 4.46 | 3.92 | 3.32 | 2.97 | 2.84 | 2.72 | 2.54 | 2.41 | 0 |
| AP5 | AD | 0.34 | 0.28 | 0.26 | 0.24 | 0.25 | 0.27 | 0.30 | 0.35 | 0.41 | - |
| GM1 | linear joint | 2.50 | 2.26 | 2.16 | 2.08 | 2.02 | 2.02 | 2.02 | 2.02 | 1.98 | 0 |
| GM1 | nonlinear joint | 2.57 | 2.32 | 2.22 | 2.14 | 2.07 | 2.08 | 2.08 | 2.08 | 2.04 | 0 |
| GM1 | AD | 0.07 | 0.07 | 0.08 | 0.10 | 0.13 | 0.16 | 0.21 | 0.30 | 0.39 | - |
| GM2 | linear joint | 3.21 | 2.90 | 2.81 | 2.70 | 2.80 | 2.89 | 2.91 | 2.91 | 2.93 | 0 |
| GM2 | nonlinear joint | 3.38 | 3.05 | 2.95 | 2.84 | 2.94 | 3.03 | 3.06 | 3.06 | 3.09 | 0 |
| GM2 | AD | 0.04 | 0.05 | 0.05 | 0.07 | 0.11 | 0.15 | 0.19 | 0.28 | 0.38 | - |
| GM3 | linear joint | 2.38 | 2.05 | 1.90 | 1.74 | 1.67 | 1.64 | 1.57 | 1.46 | 1.33 | 0 |
| GM3 | nonlinear joint | 2.47 | 2.14 | 1.98 | 1.82 | 1.74 | 1.71 | 1.64 | 1.52 | 1.39 | 0 |
| GM3 | AD | 0.04 | 0.05 | 0.06 | 0.08 | 0.12 | 0.15 | 0.19 | 0.26 | 0.32 | - |
| GM4 | linear joint | 2.51 | 2.23 | 2.22 | 2.32 | 2.32 | 2.34 | 2.39 | 2.46 | 2.31 | 0 |
| GM4 | nonlinear joint | 2.30 | 2.04 | 2.03 | 2.13 | 2.12 | 2.15 | 2.20 | 2.26 | 2.13 | 0 |
| GM4 | AD | 0.03 | 0.04 | 0.04 | 0.06 | 0.09 | 0.12 | 0.16 | 0.24 | 0.29 | - |
| GM5 | linear joint | 2.67 | 2.56 | 2.51 | 2.54 | 2.54 | 2.53 | 2.51 | 2.34 | 2.16 | 0 |
| GM5 | nonlinear joint | 2.79 | 2.68 | 2.62 | 2.66 | 2.66 | 2.66 | 2.64 | 2.46 | 2.27 | 0 |
| GM5 | AD | 0.08 | 0.09 | 0.10 | 0.14 | 0.21 | 0.25 | 0.33 | 0.42 | 0.51 | - |

The numbers, without interpretation beyond them:

- The two partitioned surfaces have the same median -laplacian at every window, equal to U/D_fit to three digits
  (B8: linear 0.00324 1/m on AP1 = 5.5e-5/1.699e-2; nonlinear 0.00331 = 5.5e-5/1.652e-2 within 0.6 %), because
  every option (iii) pixel is a hillslope cell of every fitted model (channel fraction 0 on all ten tiles) and the
  hillslope rows of both models have -laplacian = U/D there (the nonlinear correction is below 1 % where the slope
  is below 0.1 S_c). The ratio curves of the partitioned surfaces are therefore C_obs(L)/(U/D_fit), and their
  shape in L is the shape of the observed curve of A2.
- On AP the ratio for the partitioned surfaces is 4.3-8.3 at 11 m and 2.0-4.0 at 51 m; at no window on any AP tile
  does the observed median curvature of the option (iii) set come within a factor 2 of the fitted surfaces'. On GM
  the ratio is 1.9-2.8 at 11 m and 1.3-3.1 at 51 m, flat on GM1, GM2, GM4 and GM5 (the observed curve is a
  plateau there) and falling on GM3.
- The AD surfaces are 2.3-25x more curved than the observed set at 7-11 m (ratio 0.04-0.43), and the ratio rises
  with L to 0.29-0.51 at 51 m on every tile: the AD optimum (D = 4e-5 to 1e-3 m^2/yr) puts sharp ridges on the
  hilltops that the longer windows smooth.

**D recovery (5.7).** Per tile, D_fit against both ends of the bracket (B8):

| tile | D0_A | D0_B | D_fit linear joint | /D0_A | /D0_B | D_fit nonlinear joint | /D0_A | /D0_B | D_fit AD | /D0_A | D0 by the 02b set | D_fit lin / D0_02b |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 3.59e-3 | 6.98e-3 | 1.70e-2 | 4.74 | 2.43 | 1.65e-2 | 4.61 | 2.37 | 5.45e-4 | 0.15 | 1.81e-2 | 0.94 |
| AP2 | 3.22e-3 | 4.08e-3 | 1.68e-2 | 5.22 | 4.12 | 1.65e-2 | 5.13 | 4.05 | 5.53e-4 | 0.17 | 8.97e-3 | 1.87 |
| AP3 | 2.16e-3 | 2.16e-3 | 1.79e-2 | 8.31 | 8.31 | 1.65e-2 | 7.67 | 7.67 | 5.03e-4 | 0.23 | 1.16e-2 | 1.55 |
| AP4 | 3.63e-3 | 7.72e-3 | 1.97e-2 | 5.44 | 2.56 | 1.99e-2 | 5.48 | 2.57 | 1.02e-3 | 0.28 | 9.59e-3 | 2.06 |
| AP5 | 2.92e-3 | 3.46e-3 | 1.26e-2 | 4.31 | 3.64 | 1.14e-2 | 3.91 | 3.31 | 5.73e-4 | 0.20 | 6.19e-3 | 2.03 |
| GM1 | 9.00e-3 | 9.35e-3 | 1.95e-2 | 2.16 | 2.08 | 1.99e-2 | 2.21 | 2.12 | 2.95e-4 | 0.03 | 1.01e-2 | 1.92 |
| GM2 | 9.77e-3 | 1.02e-2 | 2.74e-2 | 2.81 | 2.70 | 2.87e-2 | 2.94 | 2.83 | 8.78e-5 | 0.01 | 1.48e-2 | 1.85 |
| GM3 | 1.01e-2 | 1.01e-2 | 1.92e-2 | 1.90 | 1.90 | 1.99e-2 | 1.96 | 1.96 | 4.09e-5 | 0.00 | 1.81e-2 | 1.06 |
| GM4 | 9.86e-3 | 9.44e-3 | 2.19e-2 | 2.22 | 2.32 | 1.99e-2 | 2.02 | 2.11 | 6.29e-5 | 0.01 | 1.27e-2 | 1.72 |
| GM5 | 7.59e-3 | 7.50e-3 | 1.90e-2 | 2.51 | 2.54 | 1.99e-2 | 2.62 | 2.65 | 2.29e-4 | 0.03 | 1.00e-2 | 1.90 |

(D0 by the `02b` set, fixed plateau window and no exclusion, from A3 of the Phase 6.0 report, for the record.) The
fitted surface's median Laplacian on the option (iii) pixels is -U/D_fit on every tile for both partitioned models
(all of those pixels are hillslope cells), next to the observed C_ht(L_f) of 1.5-2.6e-2 (AP) and 1.5-1.9e-2 (GM) 1/m
(B8). The AP statements above cite both ends of the bracket, as decision 1 requires.

## 7. Figures

All under `reports/v2/figures_v2/` (gitignored), built by `lemcvt figures all` in job 52074991 from results only.

- `supp_fig6_resolution_ratio.png` and `resolution_study_dinf.png`: the resolution sweep at the config parameters
  (site m, K0, D0_A; D8 and the rebuilt D-infinity), runs `20261111T0127*-0130*_sweep_resolution_<tile>` (B9).
  Median Ks_mod/Ks_obs by site at dx = 5, 10, 20, 50, 100 m: AP D8 16.4, 6.5, 3.0, 1.6, 0.85; AP D-inf 14.3, 5.9,
  2.9, 1.6, 1.15; GM D8 11.9, 6.6, 3.5, 1.8, 1.3; GM D-inf 10.9, 6.1, 3.3, 1.8, 1.4. At D0_B on AP (runs
  `*_sweep_resolution_AP<n>_D0B`): 16.8, 7.7, 3.4, 1.7, 0.89 (D8) and 15.1, 7.0, 3.2, 1.6, 1.15 (D-inf); the figure
  is drawn at D0_A, the D0_B rows are in B9 and the record's CSV.
- `fig_slope_area.png`: observed against AD-no-opt, AD-opt and partitioned, now ten tiles at the new optima
  (run `20261112T175641Z_slope_area_10tiles`; B9). Log-binned fits with the v1 thresholds: observed k_s 15-37
  (AP, metres), 2.5-7.2 (GM) with theta 0.49-0.56 (AP), 0.31-0.41 (GM); the partitioned surfaces give theta
  0.41-0.52 (AP) and 0.30-0.31 (GM) with R^2 0.83-0.98 (AP) and 0.976-1.000 (GM); the AD-opt surfaces theta
  0.33-0.40 (AP), 0.26-0.30 (GM); the AD-no-opt surfaces have no slope-area relation (theta -0.17 to 0.07,
  R^2 0.00-0.39), see 5.1.
- `supp_fig5_synthetic_pe.png`: regenerated from run `20261111T015253Z_synthetic_pe_supp_fig5` (job 51845603);
  synthetic, unchanged in content (D_bal 0.04402, Pe 100, 25 local-k_s rows).
- `a5_nonlinear_joint.png`: the (A_crit, D) MSE surfaces of 5.4 with the 1 % (white) and 5 % (black) contours,
  the cap hits (red crosses along the low-D rows) and D0_A (dashed).
- `a6_curvature_compare.png`: section 6, one panel per tile, four curves, with U/D0_A and U/D0_B as horizontal
  references.
- `d8_vs_dinf_*`: AP1 and GM1 at the config parameters in metres, both D-infinity partitioned models (B9).
- **Not built:** `fig3_deltas` (log10 K_fit/K0 and D_fit/D0 per tile). Its builder looks for one tile run holding
  both the AD and the joint stage, the v1 layout; the Phase 6 runs are one stage per run, so it reported "no
  tile_run results with AD/joint stages" and was skipped. The numbers it would draw are in 5.1, 5.2 and 6
  (K_fit/K0 0.90-1.06 for the joint, 2-10 for AD; D_fit/D0_A 1.9-8.3 for the joint, 0.003-0.28 for AD). Open item
  (section 10).

## 8. Old-against-new record

Built by `lemcvt phase6 record` in job 52074991: `tables/old_vs_new_record.csv` and `.md` (gitignored data
products; the tile and stage rows are reproduced in B10). 692 rows: 160 tile x quantity rows (theta/m, k_s, C_ht,
K0, D0, AD K/D/MSE, stage-2 A_crit/MSE, joint A_crit/K/D/MSE, nonlinear A_crit/MSE), 8 `d8_vs_dinf` MSE rows, 24
slope-area rows and 500 resolution-sweep ratio rows. Causes are assigned mechanically from the addendum lists; the
two added by addendum 6 appear as "AP D0 bracket" on every AP quantity that takes D0 and "AP1 seam" on every AP1
quantity (verdict of section 3: negligible by the pre-registered rule; the N9 inconsistency stands as the
description of the grids). No manual entry. The largest systematic changes:

| quantity | v2 / v1 | tiles | causes |
| --- | --- | --- | --- |
| K0 = U/k_s | 2.9-5.0 | AP1-AP3 | vertical units, m (0.47 against 0.54-0.59), measurement (k_s at theta = m above 10^3) |
| K0 = U/k_s | 0.44-0.71 | AP4, AP5 | the same; v1 theta 0.40-0.44 on these tiles |
| K0 | 2.2, 3.4 | GM1, GM3 | m (0.31 against 0.38-0.42), measurement |
| joint K | 2.3-4.0 | AP1-AP3 | units, m, optimizer; joint K follows K0 (5.2) |
| joint A_crit | 0.13-0.52 | GM1, GM3, GM4 | optimizer (v1 local optima at 6526-8092 against the v2 global 860-3853) |
| AD D | 0.010, 0.017 | GM1, GM3 | optimizer (the r profile finds D = 2.9e-4, 4.1e-5 against the v1 2.9e-2, 2.3e-3) |
| D0 = U/C_ht | 0.25-0.44 | AP | measurement (option (iii) set, exclusion), units; the AP D0 bracket |
| stage-2 A_crit | 0.21-0.54 | AP | measurement (K0, D0_A), units, optimizer; the AP D0 bracket |
| stage-2 MSE | 0.63-6.8 | AP | the same |
| nonlinear A_crit | 0.05-0.17 | AP | unconverged v1 nonlinear (v1 at the 50-iteration cap in feet), D0 bracket |
| nonlinear MSE | 3.4-11.6 | AP | the same; v1 reported MSE of an unconverged solve |
| joint A_crit, K, D, MSE | 0.84-1.19, 0.90-0.99, 0.94-1.04, 0.94-0.97 | GM2, GM5 | optimizer; unchanged in substance |
| `d8_vs_dinf` PT D-inf MSE | 0.015 (AP1), 0.50 (GM1) | | D-inf rebuild |
| `d8_vs_dinf` PT D8 MSE | 14.1 (AP1), 1.4 (GM1) | | measurement (the config now carries K0 and D0_A, not the v1 values) |
| slope-area ORIGINAL k_s | 0.32 (AP1), 1.00 (GM2) | | vertical units (k_s in ft^(2 theta)); GM2 identical to 4 digits |
| resolution-sweep ratios | 500 rows in the CSV | | measurement, m, units; tie-break N1-N3 |

## 9. What the evidence supports, and where it is weak

- **CI (section 1).** Measured: ruff, mypy and the fast suite pass in the project environment on Sherlock at the
  reported tree (189 tests); the GitHub run on 92c4c30 failed at mypy (3.11) and in the tests (3.12). Inferred:
  the GitHub failures come from the install differing from the project environment (no `numba`, other numpy and
  stubs versions). Weak: the logs were not read, so the inference is unverified; the gate of 2.5 is not passed and
  the push trigger stays off.
- **The D0 bracket and configs (section 2).** Measured: every value with provenance; the bracket is 1.2-2.1 wide on
  AP1, AP2, AP4 and AP5 and collapses on AP3 and GM. Weak: as the Phase 6.0 report said, the 11 m value on AP is not
  a plateau and L_w is a rule, not a measurement; the AP D0 numbers remain window choices.
- **The seam (section 3).** Measured: with S excluded from the misfit the AP1 optima move by less than 0.3 % and the
  AD r by 0.002 decades; the pre-registered thresholds are met by factors of 9-45. Inferred: that the three-row
  inconsistency does not affect AP1's model results. Weak: the test masks the misfit, not the routing; cells
  downstream of the seam keep the stored areas in the forward model, so an area error that propagates far
  downstream is tested only through its effect on the misfit outside S.
- **K recovery (5.2).** Measured: K_fit/K0 = 0.90-1.06 on all ten tiles with the chi-based K0 at theta = m above
  10^3 m^2. Inferred: that the joint optimum's K is the channel chi-slope. Weak: K0 and the joint K share the chi
  integration and the D8 network, so the agreement is between two readings of the same channel elevations; n_eff of
  the per-tile k_s is 1.1-2.3 on AP (Phase 6.0 report, 8); the D-infinity joint K is 1.3-1.5x the D8 one (5.5).
- **D (5.2, 5.4, 6).** Measured: the fitted D of the linear joint (1.3-2.7e-2) is 4.3-8.3x D0_A and 2.4-8.3x D0_B on AP
  and 1.9-2.8x D0 on GM; the nonlinear joint returns the same D within a grid step and the same MSE within 3 % with
  planar hillslopes at the optimum (1-11 % of hillslope cells above 0.5 S_c); on the option (iii) pixels the fitted
  surfaces are 2-8x less curved than the DEM at 11 m and still 1.3-4x at 51 m. Inferred: nothing beyond the
  comparison (addendum 6, 4). Weak: the nonlinear joint grid resolves D only to a factor 1.2 (the 1 % contour is one
  cell on 7 tiles); the fitted-surface curvature on the hilltop set is not a measurement but the identity U/D_fit;
  the observed C_ht depends on the window on AP (A2).
- **Nonlinear transport (5.3, 5.4).** Measured: at the measured (K0, D0_A) the nonlinear stage lowers the stage-2
  MSE by 15-25 % on AP and 5-13 % on GM and leaves it 6-10x and 1.8-2.5x the joint's; with D free the nonlinearity
  disengages. Weak: the AP3 profile at D0_A hit the cap on 96 of 261 evaluations (the optimum did not); the
  estimate of its cost was off by 6x.
- **AD against partitioned (5.1).** Measured: the AD MSE is 1.3-4.3x the linear joint MSE on every tile at the v2
  optima; the AD-opt slope-area theta (0.26-0.40) is below the observed (0.31-0.56) on every tile. Weak: the AD
  model has two free parameters against the partitioned model's three.
- **D-infinity (5.5).** Measured: the rebuilt model's joint MSE is 1.8-2.1x the D8 model's on AP1 and GM1, with K
  1.3-1.5x larger. Weak: two tiles; the GM1 D-infinity A_crit (210) sits at the low end of its flat profile.
- **The record (8).** Measured: 692 ratios, causes assigned by rule. Weak: the causes are lists, not attributions;
  no row isolates one cause, and the first-session plan of a per-cause decomposition was not run.
- **Budget (4).** Measured: 856 core-hours used, 2215 of limits. Weak: the Picard cost model under-predicted the
  AP nonlinear stage by 2-6x and should be re-fitted before Phase 7.

## 10. Decisions needed

1. **The nonlinear joint D grid.** The 1 % contour is a single grid cell on 7 of 10 tiles (step x1.2 in D). Options:
   (a) accept one-step resolution and report D_NL = D_lin within 20 %, which the numbers already support; (b) a
   local refinement, 5 D values at x1.05 spacing around each optimum over the 5 % A_crit range (about 30-60 Picard
   solves per tile, roughly 1/10 of the grid's cost), to give a 1 % interval in D. (b) is cheap; it does not change
   any conclusion of 5.4.
2. **GM1's K sensitivity.** A 5 % change of K moves the GM1 nonlinear joint A_crit from 1758 to 1200 (-32 %) with MSE
   +5 %; the AP5 optimum is stable. Report as is, or run the K check on the other GM tiles (4 tiles x 2 factors,
   about 8 x 35 core-hours of limits) to see whether this is GM1 or GM.
3. **`fig3_deltas`.** Rewrite the builder to read the per-stage Phase 6 runs (small), or drop the figure; its
   numbers are in this report.
4. **CI.** Read the logs of run 42940255348 on the Actions page, fix, restore the push trigger. Nothing on Sherlock
   remains to be fixed for it.
5. **Phase 7** waits for this review. The run list of the directive's side analyses is unchanged by anything here
   except that the cost model for the nonlinear stages needs the 5.3 wall times (8.5-26.7 h on AP).

## 11. Appendices (generated tables)

Generated by `reports/v2/scripts/make_phase6_results_tables.py` on 2026-11-12 10:02 PST from the runs named in each table (`$LEMCVT_RESULTS` = `$GROUP_HOME/lemcvt_results`; the latest Phase 6 run of each step and tile, metres, uncropped, Picard cap 100, configs with the measured block).

## B1. Section 1.3: the measured inputs written into the configs (D0 bracket)

| tile | m | k_s (m^(2m)) at 1e3 [68 %] (n_eff) | K0 = U/k_s (1/yr) | C_ht_A at L_f (1/m) | D0_A (m^2/yr) [68 %] | L_w (m) | C_ht_B (1/m) | D0_B (m^2/yr) | D0_B / D0_A | Picard cap |
|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | 0.47 | 12.421 [12.3, 13] (1.5) | 4.428e-06 | 0.01533 | 3.587e-03 [0.00339, 0.00385] | 31 | 0.007882 | 6.978e-03 | 1.95 | 100 |
| AP2 | 0.47 | 11.584 [11.2, 12.2] (1.9) | 4.748e-06 | 0.01709 | 3.218e-03 [0.00303, 0.00347] | 15 | 0.01349 | 4.078e-03 | 1.27 | 100 |
| AP3 | 0.47 | 11.530 [10, 11.6] (1.1) | 4.770e-06 | 0.02553 | 2.155e-03 [0.00191, 0.00249] | 11 | 0.02553 | 2.155e-03 | 1.00 | 100 |
| AP4 | 0.47 | 11.932 [9.9, 12.4] (2.3) | 4.610e-06 | 0.01515 | 3.629e-03 [0.00334, 0.00384] | 41 | 0.007124 | 7.720e-03 | 2.13 | 100 |
| AP5 | 0.47 | 10.726 [10, 12.6] (1.6) | 5.128e-06 | 0.01883 | 2.921e-03 [0.00268, 0.0033] | 15 | 0.01592 | 3.456e-03 | 1.18 | 100 |
| GM1 | 0.31 | 3.029 [2.94, 3.09] (3.5) | 4.854e-05 | 0.01633 | 9.001e-03 [0.00853, 0.00935] | 15 | 0.01572 | 9.354e-03 | 1.04 | 100 |
| GM2 | 0.31 | 2.666 [2.52, 2.85] (3.7) | 5.514e-05 | 0.01505 | 9.770e-03 [0.00917, 0.0104] | 15 | 0.01448 | 1.015e-02 | 1.04 | 100 |
| GM3 | 0.31 | 2.292 [2.21, 2.41] (2.7) | 6.413e-05 | 0.01453 | 1.012e-02 [0.00955, 0.0109] | 11 | 0.01453 | 1.012e-02 | 1.00 | 100 |
| GM4 | 0.31 | 4.225 [4.18, 4.25] (1.4) | 3.479e-05 | 0.01492 | 9.855e-03 [0.00883, 0.0109] | 15 | 0.01558 | 9.435e-03 | 0.96 | 100 |
| GM5 | 0.31 | 2.929 [2.85, 2.94] (1.2) | 5.019e-05 | 0.01937 | 7.589e-03 [0.00739, 0.00794] | 15 | 0.01961 | 7.496e-03 | 0.99 | 100 |

Provenance: measured-inputs run `20261111T002913Z_phase6_inputs_10tiles` (git 6f956a6d); site m from the S(m) minima (AP 1 % interval [[0.44, 0.49]]).

## B2. Section 3: the AP1 seam test (pre-registered rule)

Run `20261111T013013Z_seam_test_AP1` (job 51842989, git 6ca929f6); misfit over all valid cells (1030225) and over all valid cells except S (1011808).

|S| = **18417** cells (1.79 % of 1030225 valid): rows 385-395 11165, downstream of the 345 seam cells 9057, inconsistent 699 (union).

| profile | mask | A_crit (m^2) | K (1/yr) | D (m^2/yr) | MSE (m^2) | 1 % interval | 5 % interval |
|---|---|---|---|---|---|---|---|
| linear joint | all | 6888.6 | 3.9952e-06 | 1.6990e-02 | 22.384 | 6540-6930 | 5637-7905 |
| stage 2 at D0_A | all | 823.3 | - | - | 289.23 | 823-830 | - |
| AD | all | - | 3.9950e-05 | 5.4467e-04 | 59.018 | r = 7.3347e-02 | - |
| linear joint | masked | 6888.6 | 3.9934e-06 | 1.6950e-02 | 22.423 | 6540-6930 | 5504-7905 |
| stage 2 at D0_A | masked | 823.3 | - | - | 288.65 | 823-830 | - |
| AD | masked | - | 4.0016e-05 | 5.4848e-04 | 59.179 | r = 7.2958e-02 | - |

Profile shapes (max relative difference of MSE/MSE_min between the masks): joint 0.0423, stage 2 0.0245.

| criterion | value | threshold | pass |
|---|---|---|---|
| masked joint A_crit inside the unmasked 1 % interval | 6888.6 in 6540-6930 | - | True |
| K relative change | 0.044 % | < 2 % | True |
| D relative change | 0.235 % | < 5 % | True |
| AD r change (decades) | 0.0023 | < 0.1 | True |
| **seam negligible** | | | **True** |

## B3. Section 5.1: AD profile (r = K/D, closed-form D; 200-point grid + Brent)

| tile | run (job) | r = K/D | K (1/yr) | D (m^2/yr) | MSE (m^2) | box | 1 % interval in r | 5 % interval in r | AD no-opt MSE at (K0, D0_A) | v1 K, D, MSE (converted) | wall (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | `20261111T012439Z_tile_run_AP1_v2_ad` (51842899) | 7.3347e-02 | 3.9950e-05 | 5.4467e-04 | 59.018 | interior | 6.472e-02-7.851e-02 | 6.472e-02-7.851e-02 | 4.7875e+05 | 2.716e-05, 1.787e-03, 34.649 | 1.7 |
| AP2 | `20261111T012443Z_tile_run_AP2_v2_ad` (51842900) | 7.7037e-02 | 4.2636e-05 | 5.5344e-04 | 76.493 | interior | 6.582e-02-7.865e-02 | 5.508e-02-9.399e-02 | 4.1694e+05 | 2.283e-05, 2.185e-03, 35.442 | 1.6 |
| AP3 | `20261111T012443Z_tile_run_AP3_v2_ad` (51843107) | 9.1117e-02 | 4.5846e-05 | 5.0316e-04 | 152.81 | interior | 7.851e-02-1.155e-01 | 5.335e-02-1.701e-01 | 8.6832e+05 | 3.658e-05, 2.180e-03, 115.01 | 1.7 |
| AP4 | `20261111T012445Z_tile_run_AP4_v2_ad` (51843108) | 4.6204e-02 | 4.7099e-05 | 1.0194e-03 | 45.717 | interior | 4.609e-02-4.609e-02 | 3.857e-02-5.508e-02 | 3.4462e+05 | 5.368e-05, 6.959e-04, 53.721 | 1.7 |
| AP5 | `20261111T012444Z_tile_run_AP5_v2_ad` (51843109) | 7.7181e-02 | 4.4212e-05 | 5.7283e-04 | 69.808 | interior | 7.851e-02-7.851e-02 | 6.472e-02-9.525e-02 | 5.6735e+05 | 5.123e-05, 1.416e-04, 93.396 | 1.7 |
| GM1 | `20261111T012445Z_tile_run_GM1_v2_ad` (51843110) | 5.0306e-01 | 1.4826e-04 | 2.9471e-04 | 48.963 | interior | 4.702e-01-5.761e-01 | 3.838e-01-5.761e-01 | 3.7096e+05 | 4.461e-04, 2.884e-02, 168.65 | 1.6 |
| GM2 | `20261111T012445Z_tile_run_GM2_v2_ad` (51843111) | 1.4912e+00 | 1.3092e-04 | 8.7798e-05 | 82.546 | interior | 1.203e+00-1.862e+00 | 9.253e-01-2.641e+00 | 1.489e+05 | 1.311e-04, 8.298e-05, 81.956 | 1.6 |
| GM3 | `20261111T012445Z_tile_run_GM3_v2_ad` (51843112) | 3.3159e+00 | 1.3564e-04 | 4.0907e-05 | 64.723 | interior | 2.420e+00-3.145e+00 | 1.706e+00-3.432e+00 | 2.4195e+05 | 1.761e-04, 2.344e-03, 76.187 | 1.5 |
| GM4 | `20261111T012439Z_tile_run_GM4_v2_ad` (51843113) | 1.5123e+00 | 9.5138e-05 | 6.2910e-05 | 174.48 | interior | 1.313e+00-1.862e+00 | 9.253e-01-2.217e+00 | 3.5447e+05 | 9.461e-05, 6.784e-05, 175.39 | 1.6 |
| GM5 | `20261111T012438Z_tile_run_GM5_v2_ad` (51842893) | 6.7297e-01 | 1.5397e-04 | 2.2879e-04 | 93.319 | interior | 5.977e-01-7.769e-01 | 5.019e-01-9.253e-01 | 5.8559e+05 | 1.566e-04, 1.860e-04, 93.177 | 1.5 |

## B4. Section 5.2: linear partitioned model (stage 2 at D0_A and D0_B, joint; thin 2000 + refinement + every unique area in the 5 % intervals)

| tile | run (job) | evaluations (fill) | stage 2 A_crit at D0_A (MSE) | 1 % | stage 2 A_crit at D0_B (MSE) | 1 % | joint A_crit | K | D | MSE | joint 1 % | joint 5 % (points) | K, D range over the 1 % interval | direct check | wall (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | `20261111T050726Z_tile_run_AP1_v2_partitioned-refined` (51845567) | 9180 (6933) | 823.3 (289.23) | 823-832 | 1090.4 (86.402) | 1078-1112; 1114-1115; 1115-1116; 1116-1117; 1488-1500; 1501-1501 | 6888.6 | 3.9952e-06 | 1.6990e-02 | 22.384 | 6532-6958 | 5637-7905 (800) | K 3.98e-06-4.00e-06, D 1.69e-02-1.71e-02 | 5.4e-13 | 9.9 + 9.9 |
| AP2 | `20261111T050734Z_tile_run_AP2_v2_partitioned-refined` (51845568) | 11821 (9452) | 495.3 (245.75) | 461-517 | 682.6 (185.66) | 667-690; 690-691; 696-707; 708-709; 711-727 | 3121.1 | 4.5968e-06 | 1.6796e-02 | 22.461 | 2975-2976; 2976-3036; 3036-3036; 3064-3185; 3223-3223; 3225-3305 | 2637-3476 (2381) | K 4.58e-06-4.60e-06, D 1.67e-02-1.69e-02 | 8.6e-13 | 11.8 + 11.8 |
| AP3 | `20261111T045020Z_tile_run_AP3_v2_partitioned-refined` (51845644) | 7096 (4819) | 354.6 (340.57) | 337-373; 374-374; 375-376; 376-376; 376-377; 378-378 | 354.6 (340.57) | 337-373; 374-374; 375-376; 376-376; 376-377; 378-378 | 4742.0 | 4.8254e-06 | 1.7907e-02 | 42.122 | 4622-4704; 4717-4829; 4835-4837; 5159-5162; 5163-5164; 5487-5495; 5504-5505; 5585-5585; 5589-5693 | 4297-6552 (1895) | K 4.80e-06-4.87e-06, D 1.78e-02-1.95e-02 | 4.8e-12 | 6.6 + 6.6 |
| AP4 | `20261111T050505Z_tile_run_AP4_v2_partitioned-refined` (51845645) | 11410 (9079) | 536.2 (255.23) | 515-619; 619-620 | 983.9 (81.075) | 949-1057 | 7231.8 | 4.5219e-06 | 1.9743e-02 | 35.156 | 6983-6983; 6987-7402; 7641-7895 | 6292-9029 (1287) | K 4.52e-06-4.71e-06, D 1.96e-02-2.02e-02 | 3.0e-12 | 11.8 + 11.8 |
| AP5 | `20261111T050104Z_tile_run_AP5_v2_partitioned-refined` (51845546) | 10481 (8245) | 378.4 (226.29) | 332-333; 333-414 | 442.1 (190.48) | 405-406; 406-406; 406-454; 455-455; 455-455 | 1887.1 | 5.1191e-06 | 1.2584e-02 | 27.869 | 1801-1939; 1948-1949; 1965-1981; 1988-1988 | 1763-2244 (2254) | K 5.11e-06-5.12e-06, D 1.25e-02-1.28e-02 | 4.3e-12 | 10.1 + 10.1 |
| GM1 | `20261111T050111Z_tile_run_GM1_v2_partitioned-refined` (51845646) | 2553 (547) | 256.0 (51.29) | 249-271 | 259.0 (48.447) | 251-276 | 1241.0 | 4.8653e-05 | 1.9477e-02 | 24.69 | 1133-1270 | 1041-1847 (716) | K 4.86e-05-4.87e-05, D 1.92e-02-1.96e-02 | 7.7e-13 | 0.9 + 0.9 |
| GM2 | `20261111T052240Z_tile_run_GM2_v2_partitioned-refined` (51845647) | 2132 (107) | 204.0 (57.32) | 183-217 | 210.0 (55.357) | 192-219 | 3509.0 | 5.5148e-05 | 2.7426e-02 | 26.071 | 3352-3539 | 3321-3585 (154) | K 5.51e-05-5.51e-05, D 2.71e-02-2.79e-02 | 1.4e-13 | 0.2 + 0.2 |
| GM3 | `20261111T052022Z_tile_run_GM3_v2_partitioned-refined` (51845648) | 2758 (739) | 231.0 (61.265) | 212-212; 215-257 | 231.0 (61.265) | 212-212; 215-257 | 860.0 | 6.8099e-05 | 1.9229e-02 | 33.027 | 723-729; 731-745; 830-937 | 526-1750 (1089) | K 6.76e-05-6.83e-05, D 1.81e-02-2.02e-02 | 5.2e-13 | 1.0 + 1.0 |
| GM4 | `20261111T051659Z_tile_run_GM4_v2_partitioned-refined` (51845649) | 2549 (511) | 423.0 (113.8) | 395-407; 412-445 | 422.0 (122.92) | 342-345; 365-438 | 3853.0 | 3.6397e-05 | 2.1893e-02 | 40.743 | 3257-3258; 3261-3342; 3643-3895 | 2586-4442 (665) | K 3.64e-05-3.64e-05, D 2.12e-02-2.20e-02 | 3.7e-13 | 0.9 + 0.9 |
| GM5 | `20261111T045953Z_tile_run_GM5_v2_partitioned-refined` (51845553) | 3448 (1434) | 186.0 (92.792) | 168-202 | 186.0 (93.966) | 158-200 | 1885.0 | 5.2337e-05 | 1.9030e-02 | 32.211 | 1819-1930; 2197-2349; 3098-3130 | 1255-3681 (1761) | K 5.22e-05-5.27e-05, D 1.90e-02-2.17e-02 | 1.2e-13 | 2.1 + 2.1 |

## B5. Section 5.3: nonlinear A_crit profile at D0_A (all tiles) and D0_B (AP), S_c = 1.27, Picard cap 100

D0_A:

| tile | run (job) | D0 used | grid + refine | cap hits | optimum A_crit (Picard it.) | MSE (m^2) | 1 % interval | hillslope > 0.5 / > 0.9 S_c | linear stage-2 A_crit, same D0 | wall (h) |
|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | `20261111T131504Z_tile_run_AP1_v2_nonlinear` (51906834) | 3.587e-03 | 200 + 60 | 0 of 261 | 824.232 (12) | 217.774 | 824.2-839.4 | 0.284 / 0.0246 | 823.3 | 8.5 |
| AP2 | `20261111T162039Z_tile_run_AP2_v2_nonlinear` (51926513) | 3.218e-03 | 200 + 60 | 12 of 261 | 681.632 (11) | 197.684 | 664.5-727.3 | 0.249 / 0.0303 | 495.3 | 18.8 |
| AP3 | `20261111T131504Z_tile_run_AP3_v2_nonlinear` (51906835) | 2.155e-03 | 200 + 60 | 96 of 261 | 480.089 (15) | 287.653 | 401.2-510.5 | 0.394 / 0.0586 | 354.6 | 26.7 |
| AP4 | `20261111T043738Z_tile_run_AP4_v2_nonlinear` (51860490) | 3.629e-03 | 200 + 60 | 0 of 261 | 762.439 (10) | 209.724 | 607.5-796.7 | 0.207 / 0.0174 | 536.2 | 10.1 |
| AP5 | `20261111T043726Z_tile_run_AP5_v2_nonlinear` (51860388) | 2.921e-03 | 200 + 60 | 0 of 261 | 443.964 (10) | 193.089 | 417.3-459.2 | 0.297 / 0.0401 | 378.4 | 15.3 |
| GM1 | `20261111T012729Z_tile_run_GM1_v2_nonlinear` (51843442) | 9.001e-03 | 200 + 17 | 2 of 218 | 272 (9) | 44.6951 | 264-283 | 0.130 / 0.0076 | 256.0 | 7.0 |
| GM2 | `20261111T012730Z_tile_run_GM2_v2_nonlinear` (51843443) | 9.770e-03 | 200 + 15 | 0 of 216 | 250 (9) | 52.4181 | 215-283 | 0.146 / 0.0229 | 204.0 | 8.8 |
| GM3 | `20261111T012717Z_tile_run_GM3_v2_nonlinear` (51843444) | 1.012e-02 | 200 + 16 | 0 of 217 | 254 (7) | 57.8282 | 230-273 | 0.071 / 0.0076 | 231.0 | 7.5 |
| GM4 | `20261111T013017Z_tile_run_GM4_v2_nonlinear` (51843870) | 9.855e-03 | 200 + 33 | 0 of 234 | 522 (11) | 91.3937 | 504-558 | 0.291 / 0.0170 | 423.0 | 10.3 |
| GM5 | `20261111T013017Z_tile_run_GM5_v2_nonlinear` (51842904) | 7.589e-03 | 200 + 16 | 0 of 217 | 261 (11) | 78.9911 | 235-270 | 0.227 / 0.0232 | 186.0 | 9.4 |

D0_B:

| tile | run (job) | D0 used | grid + refine | cap hits | optimum A_crit (Picard it.) | MSE (m^2) | 1 % interval | hillslope > 0.5 / > 0.9 S_c | linear stage-2 A_crit, same D0 | wall (h) |
|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | `20261111T012447Z_tile_run_AP1_v2_nonlinear_D0B` (51843114) | 6.978e-03 | 200 + 60 | 0 of 261 | 1489.7 (8) | 70.7013 | 1488-1539 | 0.068 / 0.0059 | 1090.4 | 4.1 |
| AP2 | `20261111T043742Z_tile_run_AP2_v2_nonlinear_D0B` (51860488) | 4.078e-03 | 200 + 60 | 0 of 261 | 731.067 (9) | 148.648 | 711.1-752.9 | 0.193 / 0.0235 | 682.6 | 9.9 |
| AP3 | `20261111T131502Z_tile_run_AP3_v2_nonlinear_D0B` (51906836) | 2.155e-03 | 200 + 60 | 96 of 261 | 480.089 (15) | 287.653 | 401.2-510.5 | 0.394 / 0.0586 | 354.6 | 26.7 |
| AP4 | `20261111T012439Z_tile_run_AP4_v2_nonlinear_D0B` (51843117) | 7.720e-03 | 200 + 60 | 0 of 261 | 1052.39 (7) | 74.9403 | 985.8-1068 | 0.054 / 0.0066 | 983.9 | 4.4 |
| AP5 | `20261111T043738Z_tile_run_AP5_v2_nonlinear_D0B` (51860389) | 3.456e-03 | 200 + 60 | 0 of 261 | 554.242 (10) | 165.716 | 532.4-570.4 | 0.228 / 0.0255 | 442.1 | 10.0 |

## B6. Section 5.4: nonlinear joint profile over (A_crit, D) at K fixed to the linear joint optimum (30 x 20 grid), and the K check

| tile | run (job) | K (1/yr) | A_crit range (m^2) | optimum A_crit | optimum D (m^2/yr) | MSE (m^2) | on grid edge (A_crit / D) | 1 % contour: A_crit, D | 5 % contour: A_crit, D (points) | cap hits (in 5 %) | hillslope > 0.5 / > 0.9 S_c | Picard it. | linear joint D | wall (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | `20261112T175112Z_tile_run_AP1_v2_nonlinear_joint_merged` (52075163) | 3.9952e-06 | 1879-23715 | 6973.2 | 1.6522e-02 | 22.072 | False / False | 6973-6973, 1.65e-02-1.65e-02 | 5854-6973, 1.65e-02-1.65e-02 (3) | 36 (0) | 0.011 / 0.0005 | 6 | 1.6990e-02 | 23.7 |
| AP2 | `20261112T175112Z_tile_run_AP2_v2_nonlinear_joint_merged` (52075164) | 4.5968e-06 | 879-10427 | 3160.0 | 1.6522e-02 | 22.389 | False / False | 3160-3160, 1.65e-02-1.65e-02 | 2901-3441, 1.65e-02-1.65e-02 (3) | 47 (0) | 0.013 / 0.0019 | 6 | 1.6796e-02 | 22.0 |
| AP3 | `20261112T175112Z_tile_run_AP3_v2_nonlinear_joint_merged` (52075165) | 4.8254e-06 | 1432-19656 | 4621.2 | 1.6522e-02 | 42.139 | False / False | 4621-4621, 1.65e-02-1.65e-02 | 4621-7246, 1.65e-02-1.99e-02 (6) | 85 (0) | 0.016 / 0.0008 | 6 | 1.7907e-02 | 30.3 |
| AP4 | `20261112T175112Z_tile_run_AP4_v2_nonlinear_joint_merged` (52075166) | 4.5219e-06 | 2097-27088 | 7877.3 | 1.9871e-02 | 34.383 | False / False | 7877-7877, 1.99e-02-1.99e-02 | 7213-8601, 1.99e-02-1.99e-02 (3) | 87 (0) | 0.010 / 0.0017 | 6 | 1.9743e-02 | 34.5 |
| AP5 | `20261112T175112Z_tile_run_AP5_v2_nonlinear_joint_merged` (52074977) | 5.1191e-06 | 588-6731 | 1906.1 | 1.1422e-02 | 28.478 | False / False | 1906-1906, 1.14e-02-1.14e-02 | 1481-2669, 1.14e-02-1.37e-02 (6) | 34 (0) | 0.021 / 0.0032 | 6 | 1.2584e-02 | 17.5 |
| GM1 | `20261112T175109Z_tile_run_GM1_v2_nonlinear_joint_merged` (52074971) | 4.8653e-05 | 347-5541 | 1758.0 | 1.9871e-02 | 25.21 | False / False | 1597-1758, 1.99e-02-1.99e-02 | 1452-1758, 1.99e-02-1.99e-02 (3) | 151 (0) | 0.034 / 0.0016 | 7 | 1.9477e-02 | 32.7 |
| GM2 | `20261112T175116Z_tile_run_GM2_v2_nonlinear_joint_merged` (52074972) | 5.5148e-05 | 1107-10755 | 3880.0 | 2.8742e-02 | 26.51 | False / False | 3880-3880, 2.87e-02-2.87e-02 | 3066-4196, 2.39e-02-2.87e-02 (5) | 196 (0) | 0.018 / 0.0010 | 7 | 2.7426e-02 | 44.5 |
| GM3 | `20261112T175103Z_tile_run_GM3_v2_nonlinear_joint_merged` (52075161) | 6.8099e-05 | 175-5250 | 907.0 | 1.9871e-02 | 33.891 | False / False | 717-907, 1.65e-02-1.99e-02 | 505-1628, 1.65e-02-2.39e-02 (12) | 129 (0) | 0.013 / 0.0010 | 6 | 1.9229e-02 | 26.2 |
| GM4 | `20261112T175116Z_tile_run_GM4_v2_nonlinear_joint_merged` (52075162) | 3.6397e-05 | 862-13326 | 3901.0 | 1.9871e-02 | 40.965 | False / False | 3901-3901, 1.99e-02-1.99e-02 | 3231-3901, 1.99e-02-1.99e-02 (3) | 213 (0) | 0.107 / 0.0052 | 9 | 2.1893e-02 | 46.5 |
| GM5 | `20261112T175112Z_tile_run_GM5_v2_nonlinear_joint_merged` (52074963) | 5.2337e-05 | 418-11043 | 2852.0 | 1.9871e-02 | 32.139 | False / False | 1449-3193, 1.65e-02-1.99e-02 | 1295-3573, 1.65e-02-1.99e-02 (10) | 158 (0) | 0.050 / 0.0036 | 8 | 1.9030e-02 | 33.2 |

K check (same grid at K x 0.95 and K x 1.05):

| tile | K factor | run (job) | optimum A_crit | optimum D | MSE | MSE change vs K x 1 | A_crit change | D change |
|---|---|---|---|---|---|---|---|---|
| AP5 | 0.95 | `20261112T175112Z_tile_run_AP5_v2_nonlinear_joint_Kx0.95_merged` (52075158) | 1906.1 | 1.3738e-02 | 29.041 | +1.978% | +0.00% | +20.27% |
| AP5 | 1.05 | `20261112T175113Z_tile_run_AP5_v2_nonlinear_joint_Kx1.05_merged` (52075159) | 1906.1 | 1.1422e-02 | 28.629 | +0.530% | +0.00% | +0.00% |
| GM1 | 0.95 | `20261112T175112Z_tile_run_GM1_v2_nonlinear_joint_Kx0.95_merged` (52075160) | 1200.0 | 1.9871e-02 | 26.634 | +5.648% | -31.74% | +0.00% |
| GM1 | 1.05 | `20261112T175103Z_tile_run_GM1_v2_nonlinear_joint_Kx1.05_merged` (52074988) | 1200.0 | 1.6522e-02 | 26.354 | +4.538% | -31.74% | -16.85% |

## B7. Section 5.5: D-infinity joint profile (rebuilt partitioned model) beside the D8 joint optimum

| tile | run (job) | D-inf unique areas (used) | joint A_crit | K | D | MSE | box | 1 % interval | direct check | channel cells lower than every receiver | D8 joint A_crit, K, D, MSE | MSE ratio D-inf / D8 | wall (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | `20261111T013415Z_tile_run_AP1_v2_dinf_joint` (51843456) | 162458 (2286) | 7178.1 | 5.9867e-06 | 1.6572e-02 | 40.92 | interior | 7034-7615; 7883-7883; 7940-7993; 8047-8076; 8164-8275; 8341-8415 | 4.1e-14 | 0 of 5131 | 6888.6, 3.995e-06, 1.699e-02, 22.384 | 1.83 | 13.6 |
| GM1 | `20261111T013415Z_tile_run_GM1_v2_dinf_joint` (51842942) | 89530 (2400) | 210.5 | 6.2528e-05 | 1.4204e-02 | 52.499 | interior | 166-269 | 4.4e-14 | 0 of 29409 | 1241.0, 4.865e-05, 1.948e-02, 24.69 | 2.13 | 12.8 |

## B8. Sections 4 and 5.7: C_ht(L) of the fitted surfaces on the option (iii) set, and D recovery

Run `20261112T175547Z_curvature_compare_10tiles` (job 52074991). Ratio C_obs(L)/C_model(L) at every window; channel fraction = fraction of the option (iii) pixels that are channel cells of the fitted model.

| tile | surface | pixels | 7 | 9 | 11 | 15 | 21 | 25 | 31 | 41 | 51 m | channel fraction | median -laplacian5 (all / hillslope px) | U/D_fit |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | observed (C_ht, 1/m) | 3446 | 0.0229 | 0.0181 | 0.0153 | 0.0119 | 0.00921 | 0.00868 | 0.00788 | 0.00707 | 0.00672 | - | 0.0592 / - | - |
| AP1 | linear_joint (obs/model) | 3446 | 7.07 | 5.58 | 4.74 | 3.66 | 2.84 | 2.68 | 2.43 | 2.18 | 2.07 | 0.000 | 0.00324 / 0.00324 | 0.00324 |
| AP1 | nonlinear_joint (obs/model) | 3446 | 6.91 | 5.45 | 4.63 | 3.58 | 2.78 | 2.62 | 2.38 | 2.13 | 2.03 | 0.000 | 0.00331 / 0.00331 | 0.00333 |
| AP1 | ad (obs/model) | 3446 | 0.32 | 0.27 | 0.24 | 0.21 | 0.20 | 0.21 | 0.22 | 0.25 | 0.29 | - | 0.0837 / - | - |
| AP2 | observed (C_ht, 1/m) | 944 | 0.0248 | 0.02 | 0.0171 | 0.0135 | 0.0121 | 0.0113 | 0.0111 | 0.0105 | 0.01 | - | 0.0561 / - | - |
| AP2 | linear_joint (obs/model) | 944 | 7.56 | 6.11 | 5.22 | 4.12 | 3.69 | 3.44 | 3.39 | 3.21 | 3.06 | 0.000 | 0.00327 / 0.00327 | 0.00327 |
| AP2 | nonlinear_joint (obs/model) | 944 | 7.47 | 6.04 | 5.16 | 4.07 | 3.65 | 3.41 | 3.36 | 3.17 | 3.03 | 0.000 | 0.00332 / 0.00332 | 0.00333 |
| AP2 | ad (obs/model) | 944 | 0.30 | 0.25 | 0.22 | 0.20 | 0.21 | 0.21 | 0.23 | 0.27 | 0.33 | - | 0.0935 / - | - |
| AP3 | observed (C_ht, 1/m) | 533 | 0.0372 | 0.0305 | 0.0255 | 0.0201 | 0.0158 | 0.0147 | 0.0136 | 0.0128 | 0.0121 | - | 0.0565 / - | - |
| AP3 | linear_joint (obs/model) | 533 | 12.12 | 9.93 | 8.31 | 6.54 | 5.16 | 4.78 | 4.44 | 4.16 | 3.96 | 0.000 | 0.00307 / 0.00307 | 0.00307 |
| AP3 | nonlinear_joint (obs/model) | 533 | 11.23 | 9.20 | 7.69 | 6.05 | 4.77 | 4.42 | 4.10 | 3.85 | 3.66 | 0.000 | 0.00331 / 0.00331 | 0.00333 |
| AP3 | ad (obs/model) | 533 | 0.43 | 0.37 | 0.32 | 0.28 | 0.27 | 0.28 | 0.32 | 0.39 | 0.44 | - | 0.086 / - | - |
| AP4 | observed (C_ht, 1/m) | 4711 | 0.0255 | 0.0192 | 0.0152 | 0.0111 | 0.00863 | 0.00801 | 0.00755 | 0.00712 | 0.007 | - | 0.0618 / - | - |
| AP4 | linear_joint (obs/model) | 4711 | 9.17 | 6.90 | 5.44 | 3.98 | 3.10 | 2.88 | 2.71 | 2.56 | 2.51 | 0.000 | 0.00279 / 0.00279 | 0.00279 |
| AP4 | nonlinear_joint (obs/model) | 4711 | 9.26 | 6.96 | 5.49 | 4.02 | 3.13 | 2.91 | 2.74 | 2.59 | 2.54 | 0.000 | 0.00276 / 0.00276 | 0.00277 |
| AP4 | ad (obs/model) | 4711 | 0.61 | 0.47 | 0.39 | 0.32 | 0.28 | 0.28 | 0.29 | 0.33 | 0.38 | - | 0.0466 / - | - |
| AP5 | observed (C_ht, 1/m) | 3575 | 0.0263 | 0.0214 | 0.0188 | 0.0159 | 0.0143 | 0.0136 | 0.013 | 0.0122 | 0.0115 | - | 0.055 / - | - |
| AP5 | linear_joint (obs/model) | 3575 | 6.01 | 4.90 | 4.31 | 3.64 | 3.26 | 3.12 | 2.98 | 2.79 | 2.64 | 0.000 | 0.00437 / 0.00437 | 0.00437 |
| AP5 | nonlinear_joint (obs/model) | 3575 | 5.47 | 4.46 | 3.92 | 3.32 | 2.97 | 2.84 | 2.72 | 2.54 | 2.41 | 0.000 | 0.0048 / 0.0048 | 0.00482 |
| AP5 | ad (obs/model) | 3575 | 0.34 | 0.28 | 0.26 | 0.24 | 0.25 | 0.27 | 0.30 | 0.35 | 0.41 | - | 0.0863 / - | - |
| GM1 | observed (C_ht, 1/m) | 3583 | 0.0189 | 0.017 | 0.0163 | 0.0157 | 0.0152 | 0.0153 | 0.0153 | 0.0152 | 0.0149 | - | 0.0536 / - | - |
| GM1 | linear_joint (obs/model) | 3583 | 2.50 | 2.26 | 2.16 | 2.08 | 2.02 | 2.02 | 2.02 | 2.02 | 1.98 | 0.000 | 0.00755 / 0.00755 | 0.00755 |
| GM1 | nonlinear_joint (obs/model) | 3583 | 2.57 | 2.32 | 2.22 | 2.14 | 2.07 | 2.08 | 2.08 | 2.08 | 2.04 | 0.000 | 0.00735 / 0.00735 | 0.0074 |
| GM1 | ad (obs/model) | 3583 | 0.07 | 0.07 | 0.08 | 0.10 | 0.13 | 0.16 | 0.21 | 0.30 | 0.39 | - | 0.36 / - | - |
| GM2 | observed (C_ht, 1/m) | 2530 | 0.0172 | 0.0156 | 0.015 | 0.0145 | 0.015 | 0.0155 | 0.0156 | 0.0156 | 0.0157 | - | 0.0537 / - | - |
| GM2 | linear_joint (obs/model) | 2530 | 3.21 | 2.90 | 2.81 | 2.70 | 2.80 | 2.89 | 2.91 | 2.91 | 2.93 | 0.000 | 0.00536 / 0.00536 | 0.00536 |
| GM2 | nonlinear_joint (obs/model) | 2530 | 3.38 | 3.05 | 2.95 | 2.84 | 2.94 | 3.03 | 3.06 | 3.06 | 3.09 | 0.000 | 0.0051 / 0.0051 | 0.00511 |
| GM2 | ad (obs/model) | 2530 | 0.04 | 0.05 | 0.05 | 0.07 | 0.11 | 0.15 | 0.19 | 0.28 | 0.38 | - | 0.749 / - | - |
| GM3 | observed (C_ht, 1/m) | 1500 | 0.0182 | 0.0157 | 0.0145 | 0.0133 | 0.0128 | 0.0126 | 0.012 | 0.0112 | 0.0102 | - | 0.0509 / - | - |
| GM3 | linear_joint (obs/model) | 1500 | 2.38 | 2.05 | 1.90 | 1.74 | 1.67 | 1.64 | 1.57 | 1.46 | 1.33 | 0.000 | 0.00764 / 0.00764 | 0.00764 |
| GM3 | nonlinear_joint (obs/model) | 1500 | 2.47 | 2.14 | 1.98 | 1.82 | 1.74 | 1.71 | 1.64 | 1.52 | 1.39 | 0.000 | 0.00734 / 0.00734 | 0.0074 |
| GM3 | ad (obs/model) | 1500 | 0.04 | 0.05 | 0.06 | 0.08 | 0.12 | 0.15 | 0.19 | 0.26 | 0.32 | - | 0.722 / - | - |
| GM4 | observed (C_ht, 1/m) | 3347 | 0.0168 | 0.015 | 0.0149 | 0.0156 | 0.0156 | 0.0157 | 0.0161 | 0.0165 | 0.0156 | - | 0.0613 / - | - |
| GM4 | linear_joint (obs/model) | 3347 | 2.51 | 2.23 | 2.22 | 2.32 | 2.32 | 2.34 | 2.39 | 2.46 | 2.31 | 0.000 | 0.00671 / 0.00671 | 0.00671 |
| GM4 | nonlinear_joint (obs/model) | 3347 | 2.30 | 2.04 | 2.03 | 2.13 | 2.12 | 2.15 | 2.20 | 2.26 | 2.13 | 0.000 | 0.00733 / 0.00733 | 0.0074 |
| GM4 | ad (obs/model) | 3347 | 0.03 | 0.04 | 0.04 | 0.06 | 0.09 | 0.12 | 0.16 | 0.24 | 0.29 | - | 1.06 / - | - |
| GM5 | observed (C_ht, 1/m) | 1128 | 0.0206 | 0.0198 | 0.0194 | 0.0196 | 0.0196 | 0.0196 | 0.0194 | 0.0181 | 0.0167 | - | 0.0603 / - | - |
| GM5 | linear_joint (obs/model) | 1128 | 2.67 | 2.56 | 2.51 | 2.54 | 2.54 | 2.53 | 2.51 | 2.34 | 2.16 | 0.000 | 0.00772 / 0.00772 | 0.00772 |
| GM5 | nonlinear_joint (obs/model) | 1128 | 2.79 | 2.68 | 2.62 | 2.66 | 2.66 | 2.66 | 2.64 | 2.46 | 2.27 | 0.000 | 0.00738 / 0.00738 | 0.0074 |
| GM5 | ad (obs/model) | 1128 | 0.08 | 0.09 | 0.10 | 0.14 | 0.21 | 0.25 | 0.33 | 0.42 | 0.51 | - | 0.35 / - | - |

D recovery (addendum 5, 3.7; addendum 6, 5.7):

| tile | D0_A | D0_B | D_fit linear joint | / D0_A | / D0_B | D_fit nonlinear joint | / D0_A | / D0_B | D_fit AD | / D0_A | C_ht obs at L_f | U/D0_A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | 3.587e-03 | 6.978e-03 | 1.699e-02 | 4.74 | 2.43 | 1.652e-02 | 4.61 | 2.37 | 5.447e-04 | 0.15 | 0.0153 | 0.0153 |
| AP2 | 3.218e-03 | 4.078e-03 | 1.680e-02 | 5.22 | 4.12 | 1.652e-02 | 5.13 | 4.05 | 5.534e-04 | 0.17 | 0.0171 | 0.0171 |
| AP3 | 2.155e-03 | 2.155e-03 | 1.791e-02 | 8.31 | 8.31 | 1.652e-02 | 7.67 | 7.67 | 5.032e-04 | 0.23 | 0.0255 | 0.0255 |
| AP4 | 3.629e-03 | 7.720e-03 | 1.974e-02 | 5.44 | 2.56 | 1.987e-02 | 5.48 | 2.57 | 1.019e-03 | 0.28 | 0.0152 | 0.0152 |
| AP5 | 2.921e-03 | 3.456e-03 | 1.258e-02 | 4.31 | 3.64 | 1.142e-02 | 3.91 | 3.31 | 5.728e-04 | 0.20 | 0.0188 | 0.0188 |
| GM1 | 9.001e-03 | 9.354e-03 | 1.948e-02 | 2.16 | 2.08 | 1.987e-02 | 2.21 | 2.12 | 2.947e-04 | 0.03 | 0.0163 | 0.0163 |
| GM2 | 9.770e-03 | 1.015e-02 | 2.743e-02 | 2.81 | 2.70 | 2.874e-02 | 2.94 | 2.83 | 8.780e-05 | 0.01 | 0.015 | 0.015 |
| GM3 | 1.012e-02 | 1.012e-02 | 1.923e-02 | 1.90 | 1.90 | 1.987e-02 | 1.96 | 1.96 | 4.091e-05 | 0.00 | 0.0145 | 0.0145 |
| GM4 | 9.855e-03 | 9.435e-03 | 2.189e-02 | 2.22 | 2.32 | 1.987e-02 | 2.02 | 2.11 | 6.291e-05 | 0.01 | 0.0149 | 0.0149 |
| GM5 | 7.589e-03 | 7.496e-03 | 1.903e-02 | 2.51 | 2.54 | 1.987e-02 | 2.62 | 2.65 | 2.288e-04 | 0.03 | 0.0194 | 0.0194 |

## B9. Section 5.6: figure inputs (resolution sweep at D0_A and D0_B, d8_vs_dinf at the config parameters, slope-area at the new optima, supp_fig5)

Resolution sweep at D0_A (runs 20261111T012717Z_sweep_resolution_AP1, 20261111T012728Z_sweep_resolution_AP2, 20261111T012728Z_sweep_resolution_AP3, 20261111T013013Z_sweep_resolution_GM5, 20261111T013017Z_sweep_resolution_GM3, 20261111T013018Z_sweep_resolution_GM4, 20261111T013023Z_sweep_resolution_AP5, 20261111T013029Z_sweep_resolution_GM1, 20261111T013029Z_sweep_resolution_GM2, 20261111T013031Z_sweep_resolution_AP4): median Ks_mod/Ks_obs by site at selected dx

| site | routing | dx 5 | 10 | 20 | 50 | 100 m | rows |
|---|---|---|---|---|---|---|---|
| AP | d8 | 16.42 | 6.53 | 3.04 | 1.62 | 0.85 | 125 |
| AP | dinf | 14.30 | 5.86 | 2.90 | 1.63 | 1.15 | 125 |
| GM | d8 | 11.91 | 6.57 | 3.52 | 1.75 | 1.33 | 125 |
| GM | dinf | 10.88 | 6.12 | 3.25 | 1.76 | 1.35 | 125 |

Resolution sweep at D0_B (runs 20261111T012439Z_sweep_resolution_AP2_D0B, 20261111T012445Z_sweep_resolution_AP1_D0B, 20261111T012718Z_sweep_resolution_AP4_D0B, 20261111T012718Z_sweep_resolution_AP5_D0B, 20261111T012727Z_sweep_resolution_AP3_D0B): median Ks_mod/Ks_obs by site at selected dx

| site | routing | dx 5 | 10 | 20 | 50 | 100 m | rows |
|---|---|---|---|---|---|---|---|
| AP | d8 | 16.77 | 7.70 | 3.35 | 1.65 | 0.89 | 125 |
| AP | dinf | 15.06 | 6.97 | 3.19 | 1.63 | 1.15 | 125 |
| GM | d8 | - | - | - | - | - | 0 |
| GM | dinf | - | - | - | - | - | 0 |

d8_vs_dinf at the config parameters, metres (run `20261111T015657Z_d8_vs_dinf_AP1_GM1`, job 51845603):

| tile | K | D | m | A_crit | MSE AD D8 | AD D-inf | PT D8 | PT D-inf rebuilt | PT D-inf v1 | channel cells D8 / D-inf |
|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | 4.428e-06 | 3.587e-03 | 0.47 | 5000 | 4.7875e+05 | 4.8682e+05 | 2878.6 | 5191.2 | 54942 | 5657 / 6450 |
| GM1 | 4.854e-05 | 9.001e-03 | 0.31 | 5000 | 3.7096e+05 | 3.4635e+05 | 1089.4 | 2131.8 | 2789.7 | 5318 / 6141 |

Slope-area fits at the new optima (run `20261112T175641Z_slope_area_10tiles`, job 52074991; log-binned, v1 thresholds):

| tile | series | k_s | theta | R^2 | points |
|---|---|---|---|---|---|
| AP1 | orig | 35.58 | 0.541 | 0.948 | 3798 |
| AP1 | AD_no_opt | 0.08929 | -0.107 | 0.088 | 2058 |
| AP1 | AD_opt | 2.035 | 0.401 | 0.976 | 2746 |
| AP1 | part | 8.427 | 0.522 | 0.977 | 2672 |
| AP2 | orig | 36.68 | 0.558 | 0.973 | 4355 |
| AP2 | AD_no_opt | 0.2213 | -0.028 | 0.016 | 3417 |
| AP2 | AD_opt | 0.9785 | 0.333 | 0.915 | 2995 |
| AP2 | part | 2.314 | 0.412 | 0.932 | 2896 |
| AP3 | orig | 18.27 | 0.494 | 0.923 | 3422 |
| AP3 | AD_no_opt | 0.873 | 0.065 | 0.217 | 2764 |
| AP3 | AD_opt | 1.154 | 0.359 | 0.884 | 2288 |
| AP3 | part | 4.664 | 0.482 | 0.869 | 2297 |
| AP4 | orig | 30.26 | 0.540 | 0.946 | 2838 |
| AP4 | AD_no_opt | 0.0818 | -0.112 | 0.143 | 2663 |
| AP4 | AD_opt | 0.9708 | 0.326 | 0.952 | 2382 |
| AP4 | part | 5.578 | 0.486 | 0.937 | 2329 |
| AP5 | orig | 15.28 | 0.491 | 0.922 | 2870 |
| AP5 | AD_no_opt | 0.2761 | -0.022 | 0.002 | 2051 |
| AP5 | AD_opt | 1.54 | 0.386 | 0.956 | 1813 |
| AP5 | part | 3.047 | 0.447 | 0.834 | 1659 |
| GM1 | orig | 3.22 | 0.314 | 0.953 | 4638 |
| GM1 | AD_no_opt | 0.1591 | -0.167 | 0.194 | 3941 |
| GM1 | AD_opt | 1.74 | 0.264 | 0.958 | 4735 |
| GM1 | part | 2.827 | 0.304 | 0.976 | 4717 |
| GM2 | orig | 2.488 | 0.306 | 0.962 | 4424 |
| GM2 | AD_no_opt | 0.4157 | -0.066 | 0.212 | 3774 |
| GM2 | AD_opt | 1.666 | 0.273 | 0.908 | 4608 |
| GM2 | part | 2.539 | 0.306 | 1.000 | 4413 |
| GM3 | orig | 7.194 | 0.413 | 0.953 | 4935 |
| GM3 | AD_no_opt | 0.174 | -0.162 | 0.387 | 4436 |
| GM3 | AD_opt | 1.456 | 0.286 | 0.981 | 4772 |
| GM3 | part | 2.166 | 0.311 | 0.999 | 4895 |
| GM4 | orig | 6.704 | 0.352 | 0.984 | 4662 |
| GM4 | AD_no_opt | 0.5733 | -0.051 | 0.078 | 2661 |
| GM4 | AD_opt | 2.279 | 0.276 | 0.955 | 4721 |
| GM4 | part | 3.974 | 0.310 | 0.999 | 4715 |
| GM5 | orig | 4.027 | 0.337 | 0.942 | 6107 |
| GM5 | AD_no_opt | 0.457 | -0.062 | 0.046 | 4195 |
| GM5 | AD_opt | 2.597 | 0.303 | 0.973 | 6407 |
| GM5 | part | 2.757 | 0.309 | 1.000 | 6329 |

supp_fig5 regenerated: run `20261111T015253Z_synthetic_pe_supp_fig5` (job 51845603); local k_s rows 25; D_bal 0.04402, Pe 100.


## B10. Section 5.8: the old-against-new record

Built by `lemcvt phase6 record` (`tables/old_vs_new_record.csv`, 692 rows). Tile and stage rows:

| tile | quantity | v1 | v2 | ratio v2/v1 | causes |
|---|---|---|---|---|---|
| AP1 | theta / m | 0.562 | 0.47 | 0.8363 | measurement; m |
| AP1 | k_s | 138.6 | 12.42 | 0.08965 | measurement; m; vertical units |
| AP1 | C_ht | 0.0199 | 0.01533 | 0.7706 | measurement; vertical units; AP D0 bracket |
| AP1 | K0 = U/k_s | 1.302e-06 | 4.428e-06 | 3.4 | measurement; m; vertical units |
| AP1 | D0 = U/C_ht | 0.009068 | 0.003587 | 0.3956 | measurement; vertical units; AP D0 bracket |
| AP1 | AD K | 2.716e-05 | 3.995e-05 | 1.471 | m; vertical units; AP1 seam; optimizer |
| AP1 | AD D | 0.001787 | 0.0005447 | 0.3048 | m; vertical units; AP1 seam; optimizer |
| AP1 | AD MSE | 34.65 | 59.02 | 1.703 | m; vertical units; AP1 seam; optimizer |
| AP1 | stage-2 A_crit | 3063 | 823.3 | 0.2687 | m; vertical units; AP1 seam; optimizer; measurement; AP D0 bracket |
| AP1 | stage-2 MSE | 178 | 289.2 | 1.625 | m; vertical units; AP1 seam; optimizer; measurement; AP D0 bracket |
| AP1 | joint A_crit | 6414 | 6889 | 1.074 | m; vertical units; AP1 seam; optimizer |
| AP1 | joint K | 1.335e-06 | 3.995e-06 | 2.993 | m; vertical units; AP1 seam; optimizer |
| AP1 | joint D | 0.01765 | 0.01699 | 0.9625 | m; vertical units; AP1 seam; optimizer |
| AP1 | joint MSE | 21 | 22.38 | 1.066 | m; vertical units; AP1 seam; optimizer |
| AP1 | nonlinear A_crit | 4950 | 824.2 | 0.1665 | m; vertical units; AP1 seam; optimizer; measurement; AP D0 bracket; unconverged v1 nonlinear |
| AP1 | nonlinear MSE | 18.85 | 217.8 | 11.56 | m; vertical units; AP1 seam; optimizer; measurement; AP D0 bracket; unconverged v1 nonlinear |
| AP2 | theta / m | 0.592 | 0.47 | 0.7939 | measurement; m |
| AP2 | k_s | 190.9 | 11.58 | 0.06067 | measurement; m; vertical units |
| AP2 | C_ht | 0.01944 | 0.01709 | 0.8791 | measurement; vertical units; AP D0 bracket |
| AP2 | K0 = U/k_s | 9.451e-07 | 4.748e-06 | 5.024 | measurement; m; vertical units |
| AP2 | D0 = U/C_ht | 0.009282 | 0.003218 | 0.3467 | measurement; vertical units; AP D0 bracket |
| AP2 | AD K | 2.283e-05 | 4.264e-05 | 1.867 | m; vertical units; optimizer |
| AP2 | AD D | 0.002185 | 0.0005534 | 0.2533 | m; vertical units; optimizer |
| AP2 | AD MSE | 35.44 | 76.49 | 2.158 | m; vertical units; optimizer |
| AP2 | stage-2 A_crit | 2338 | 495.3 | 0.2119 | m; vertical units; optimizer; measurement; AP D0 bracket |
| AP2 | stage-2 MSE | 392.7 | 245.8 | 0.6257 | m; vertical units; optimizer; measurement; AP D0 bracket |
| AP2 | joint A_crit | 7646 | 3121 | 0.4082 | m; vertical units; optimizer |
| AP2 | joint K | 1.151e-06 | 4.597e-06 | 3.995 | m; vertical units; optimizer |
| AP2 | joint D | 0.02325 | 0.0168 | 0.7225 | m; vertical units; optimizer |
| AP2 | joint MSE | 21.69 | 22.46 | 1.035 | m; vertical units; optimizer |
| AP2 | nonlinear A_crit | 6166 | 681.6 | 0.1105 | m; vertical units; optimizer; measurement; AP D0 bracket; unconverged v1 nonlinear |
| AP2 | nonlinear MSE | 58.56 | 197.7 | 3.376 | m; vertical units; optimizer; measurement; AP D0 bracket; unconverged v1 nonlinear |
| AP3 | theta / m | 0.542 | 0.47 | 0.8672 | measurement; m |
| AP3 | k_s | 111.5 | 11.53 | 0.1035 | measurement; m; vertical units |
| AP3 | C_ht | 0.0206 | 0.02553 | 1.239 | measurement; vertical units; AP D0 bracket |
| AP3 | K0 = U/k_s | 1.619e-06 | 4.77e-06 | 2.946 | measurement; m; vertical units |
| AP3 | D0 = U/C_ht | 0.00876 | 0.002155 | 0.246 | measurement; vertical units; AP D0 bracket |
| AP3 | AD K | 3.658e-05 | 4.585e-05 | 1.253 | m; vertical units; optimizer |
| AP3 | AD D | 0.00218 | 0.0005032 | 0.2308 | m; vertical units; optimizer |
| AP3 | AD MSE | 115 | 152.8 | 1.329 | m; vertical units; optimizer |
| AP3 | stage-2 A_crit | 1722 | 354.6 | 0.2059 | m; vertical units; optimizer; measurement; AP D0 bracket |
| AP3 | stage-2 MSE | 126 | 340.6 | 2.703 | m; vertical units; optimizer; measurement; AP D0 bracket |
| AP3 | joint A_crit | 9742 | 4742 | 0.4868 | m; vertical units; optimizer |
| AP3 | joint K | 2.072e-06 | 4.825e-06 | 2.329 | m; vertical units; optimizer |
| AP3 | joint D | 0.02265 | 0.01791 | 0.7905 | m; vertical units; optimizer |
| AP3 | joint MSE | 46.94 | 42.12 | 0.8974 | m; vertical units; optimizer |
| AP3 | nonlinear A_crit | 1.023e+04 | 480.1 | 0.04692 | m; vertical units; optimizer; measurement; AP D0 bracket; unconverged v1 nonlinear |
| AP3 | nonlinear MSE | 60.47 | 287.7 | 4.757 | m; vertical units; optimizer; measurement; AP D0 bracket; unconverged v1 nonlinear |
| AP4 | theta / m | 0.438 | 0.47 | 1.073 | measurement; m |
| AP4 | k_s | 27.91 | 11.93 | 0.4276 | measurement; m; vertical units |
| AP4 | C_ht | 0.02194 | 0.01515 | 0.6907 | measurement; vertical units; AP D0 bracket |
| AP4 | K0 = U/k_s | 6.466e-06 | 4.61e-06 | 0.7129 | measurement; m; vertical units |
| AP4 | D0 = U/C_ht | 0.008225 | 0.003629 | 0.4413 | measurement; vertical units; AP D0 bracket |
| AP4 | AD K | 5.368e-05 | 4.71e-05 | 0.8774 | m; vertical units; optimizer |
| AP4 | AD D | 0.0006959 | 0.001019 | 1.465 | m; vertical units; optimizer |
| AP4 | AD MSE | 53.72 | 45.72 | 0.851 | m; vertical units; optimizer |
| AP4 | stage-2 A_crit | 995 | 536.2 | 0.5389 | m; vertical units; optimizer; measurement; AP D0 bracket |
| AP4 | stage-2 MSE | 55.55 | 255.2 | 4.594 | m; vertical units; optimizer; measurement; AP D0 bracket |
| AP4 | joint A_crit | 7798 | 7232 | 0.9274 | m; vertical units; optimizer |
| AP4 | joint K | 6.682e-06 | 4.522e-06 | 0.6767 | m; vertical units; optimizer |
| AP4 | joint D | 0.02002 | 0.01974 | 0.9864 | m; vertical units; optimizer |
| AP4 | joint MSE | 34.38 | 35.16 | 1.023 | m; vertical units; optimizer |
| AP4 | nonlinear A_crit | 5264 | 762.4 | 0.1449 | m; vertical units; optimizer; measurement; AP D0 bracket; unconverged v1 nonlinear |
| AP4 | nonlinear MSE | 28.86 | 209.7 | 7.268 | m; vertical units; optimizer; measurement; AP D0 bracket; unconverged v1 nonlinear |
| AP5 | theta / m | 0.4 | 0.47 | 1.175 | measurement; m |
| AP5 | k_s | 15.47 | 10.73 | 0.6935 | measurement; m; vertical units |
| AP5 | C_ht | 0.02199 | 0.01883 | 0.8563 | measurement; vertical units; AP D0 bracket |
| AP5 | K0 = U/k_s | 1.167e-05 | 5.128e-06 | 0.4395 | measurement; m; vertical units |
| AP5 | D0 = U/C_ht | 0.008206 | 0.002921 | 0.3559 | measurement; vertical units; AP D0 bracket |
| AP5 | AD K | 5.123e-05 | 4.421e-05 | 0.8631 | m; vertical units; optimizer |
| AP5 | AD D | 0.0001416 | 0.0005728 | 4.046 | m; vertical units; optimizer |
| AP5 | AD MSE | 93.4 | 69.81 | 0.7474 | m; vertical units; optimizer |
| AP5 | stage-2 A_crit | 1130 | 378.4 | 0.3348 | m; vertical units; optimizer; measurement; AP D0 bracket |
| AP5 | stage-2 MSE | 33.38 | 226.3 | 6.78 | m; vertical units; optimizer; measurement; AP D0 bracket |
| AP5 | joint A_crit | 1887 | 1887 | 1 | m; vertical units; optimizer |
| AP5 | joint K | 1.129e-05 | 5.119e-06 | 0.4536 | m; vertical units; optimizer |
| AP5 | joint D | 0.01191 | 0.01258 | 1.057 | m; vertical units; optimizer |
| AP5 | joint MSE | 26.91 | 27.87 | 1.036 | m; vertical units; optimizer |
| AP5 | nonlinear A_crit | 3399 | 444 | 0.1306 | m; vertical units; optimizer; measurement; AP D0 bracket; unconverged v1 nonlinear |
| AP5 | nonlinear MSE | 24.28 | 193.1 | 7.952 | m; vertical units; optimizer; measurement; AP D0 bracket; unconverged v1 nonlinear |
| GM1 | theta / m | 0.379 | 0.31 | 0.8179 | measurement; m |
| GM1 | k_s | 6.539 | 3.029 | 0.4631 | measurement; m |
| GM1 | C_ht | 0.01442 | 0.01633 | 1.133 | measurement |
| GM1 | K0 = U/k_s | 2.248e-05 | 4.854e-05 | 2.159 | measurement; m |
| GM1 | D0 = U/C_ht | 0.01019 | 0.009001 | 0.883 | measurement |
| GM1 | AD K | 0.0004461 | 0.0001483 | 0.3323 | m; optimizer |
| GM1 | AD D | 0.02884 | 0.0002947 | 0.01022 | m; optimizer |
| GM1 | AD MSE | 168.7 | 48.96 | 0.2903 | m; optimizer |
| GM1 | stage-2 A_crit | 423.5 | 256 | 0.6045 | m; optimizer; measurement |
| GM1 | stage-2 MSE | 101 | 51.29 | 0.508 | m; optimizer; measurement |
| GM1 | joint A_crit | 8092 | 1241 | 0.1534 | m; optimizer |
| GM1 | joint K | 2.165e-05 | 4.865e-05 | 2.247 | m; optimizer |
| GM1 | joint D | 0.02837 | 0.01948 | 0.6866 | m; optimizer |
| GM1 | joint MSE | 33.87 | 24.69 | 0.7291 | m; optimizer |
| GM1 | nonlinear A_crit | 474.2 | 272 | 0.5736 | m; optimizer; measurement |
| GM1 | nonlinear MSE | 77.07 | 44.7 | 0.5799 | m; optimizer; measurement |
| GM2 | theta / m | 0.308 | 0.31 | 1.006 | measurement; m |
| GM2 | k_s | 2.414 | 2.666 | 1.104 | measurement; m |
| GM2 | C_ht | 0.0145 | 0.01505 | 1.038 | measurement |
| GM2 | K0 = U/k_s | 6.089e-05 | 5.514e-05 | 0.9054 | measurement; m |
| GM2 | D0 = U/C_ht | 0.01014 | 0.00977 | 0.9637 | measurement |
| GM2 | AD K | 0.0001311 | 0.0001309 | 0.9983 | m; optimizer |
| GM2 | AD D | 8.298e-05 | 8.78e-05 | 1.058 | m; optimizer |
| GM2 | AD MSE | 81.96 | 82.55 | 1.007 | m; optimizer |
| GM2 | stage-2 A_crit | 100 | 204 | 2.04 | m; optimizer; measurement |
| GM2 | stage-2 MSE | 57.87 | 57.32 | 0.9904 | m; optimizer; measurement |
| GM2 | joint A_crit | 2960 | 3509 | 1.185 | m; optimizer |
| GM2 | joint K | 5.598e-05 | 5.515e-05 | 0.9851 | m; optimizer |
| GM2 | joint D | 0.02641 | 0.02743 | 1.038 | m; optimizer |
| GM2 | joint MSE | 27.83 | 26.07 | 0.9369 | m; optimizer |
| GM2 | nonlinear A_crit | 244.6 | 250 | 1.022 | m; optimizer; measurement |
| GM2 | nonlinear MSE | 47.53 | 52.42 | 1.103 | m; optimizer; measurement |
| GM3 | theta / m | 0.417 | 0.31 | 0.7434 | measurement; m |
| GM3 | k_s | 7.72 | 2.292 | 0.2969 | measurement; m |
| GM3 | C_ht | 0.00711 | 0.01453 | 2.043 | measurement |
| GM3 | K0 = U/k_s | 1.904e-05 | 6.413e-05 | 3.368 | measurement; m |
| GM3 | D0 = U/C_ht | 0.02068 | 0.01012 | 0.4894 | measurement |
| GM3 | AD K | 0.0001761 | 0.0001356 | 0.7701 | m; optimizer |
| GM3 | AD D | 0.002344 | 4.091e-05 | 0.01745 | m; optimizer |
| GM3 | AD MSE | 76.19 | 64.72 | 0.8495 | m; optimizer |
| GM3 | stage-2 A_crit | 1297 | 231 | 0.178 | m; optimizer; measurement |
| GM3 | stage-2 MSE | 63.35 | 61.27 | 0.9671 | m; optimizer; measurement |
| GM3 | joint A_crit | 6526 | 860 | 0.1318 | m; optimizer |
| GM3 | joint K | 1.913e-05 | 6.81e-05 | 3.56 | m; optimizer |
| GM3 | joint D | 0.04533 | 0.01923 | 0.4242 | m; optimizer |
| GM3 | joint MSE | 43.02 | 33.03 | 0.7678 | m; optimizer |
| GM3 | nonlinear A_crit | 1297 | 254 | 0.1958 | m; optimizer; measurement |
| GM3 | nonlinear MSE | 58.98 | 57.83 | 0.9804 | m; optimizer; measurement |
| GM4 | theta / m | 0.313 | 0.31 | 0.9904 | measurement; m |
| GM4 | k_s | 4.153 | 4.225 | 1.017 | measurement; m |
| GM4 | C_ht | 0.01496 | 0.01492 | 0.9971 | measurement |
| GM4 | K0 = U/k_s | 3.54e-05 | 3.479e-05 | 0.9829 | measurement; m |
| GM4 | D0 = U/C_ht | 0.009826 | 0.009855 | 1.003 | measurement |
| GM4 | AD K | 9.461e-05 | 9.514e-05 | 1.006 | m; optimizer |
| GM4 | AD D | 6.784e-05 | 6.291e-05 | 0.9273 | m; optimizer |
| GM4 | AD MSE | 175.4 | 174.5 | 0.9948 | m; optimizer |
| GM4 | stage-2 A_crit | 422.1 | 423 | 1.002 | m; optimizer; measurement |
| GM4 | stage-2 MSE | 85.53 | 113.8 | 1.331 | m; optimizer; measurement |
| GM4 | joint A_crit | 7396 | 3853 | 0.5209 | m; optimizer |
| GM4 | joint K | 3.511e-05 | 3.64e-05 | 1.037 | m; optimizer |
| GM4 | joint D | 0.02655 | 0.02189 | 0.8246 | m; optimizer |
| GM4 | joint MSE | 47.73 | 40.74 | 0.8536 | m; optimizer |
| GM4 | nonlinear A_crit | 521.4 | 522 | 1.001 | m; optimizer; measurement |
| GM4 | nonlinear MSE | 72.77 | 91.39 | 1.256 | m; optimizer; measurement |
| GM5 | theta / m | 0.3 | 0.31 | 1.033 | measurement; m |
| GM5 | k_s | 2.507 | 2.929 | 1.168 | measurement; m |
| GM5 | C_ht | 0.01635 | 0.01937 | 1.185 | measurement |
| GM5 | K0 = U/k_s | 5.864e-05 | 5.019e-05 | 0.856 | measurement; m |
| GM5 | D0 = U/C_ht | 0.008991 | 0.007589 | 0.8441 | measurement |
| GM5 | AD K | 0.0001566 | 0.000154 | 0.9833 | m; optimizer |
| GM5 | AD D | 0.000186 | 0.0002288 | 1.23 | m; optimizer |
| GM5 | AD MSE | 93.18 | 93.32 | 1.002 | m; optimizer |
| GM5 | stage-2 A_crit | 261.4 | 186 | 0.7115 | m; optimizer; measurement |
| GM5 | stage-2 MSE | 61.02 | 92.79 | 1.521 | m; optimizer; measurement |
| GM5 | joint A_crit | 2247 | 1885 | 0.8389 | m; optimizer |
| GM5 | joint K | 5.823e-05 | 5.234e-05 | 0.8988 | m; optimizer |
| GM5 | joint D | 0.02028 | 0.01903 | 0.9382 | m; optimizer |
| GM5 | joint MSE | 33.05 | 32.21 | 0.9745 | m; optimizer |
| GM5 | nonlinear A_crit | 299.6 | 261 | 0.8711 | m; optimizer; measurement |
| GM5 | nonlinear MSE | 54.81 | 78.99 | 1.441 | m; optimizer; measurement |
| AP1 | d8_vs_dinf MSE AD D8 | 6.464e+04 | 4.787e+05 | 7.406 | measurement; m; vertical units; AP D0 bracket; AP1 seam |
| AP1 | d8_vs_dinf MSE AD D-inf | 6.516e+04 | 4.868e+05 | 7.472 | measurement; m; vertical units; AP D0 bracket; AP1 seam |
| AP1 | d8_vs_dinf MSE PT D8 | 203.9 | 2879 | 14.12 | measurement; m; vertical units; AP D0 bracket; AP1 seam |
| AP1 | d8_vs_dinf MSE PT D-inf | 3.498e+05 | 5191 | 0.01484 | measurement; m; vertical units; AP D0 bracket; AP1 seam; D-inf rebuild |
| GM1 | d8_vs_dinf MSE AD D8 | 3.229e+05 | 3.71e+05 | 1.149 | measurement; m |
| GM1 | d8_vs_dinf MSE AD D-inf | 3.184e+05 | 3.463e+05 | 1.088 | measurement; m |
| GM1 | d8_vs_dinf MSE PT D8 | 792 | 1089 | 1.375 | measurement; m |
| GM1 | d8_vs_dinf MSE PT D-inf | 4240 | 2132 | 0.5028 | measurement; m; D-inf rebuild |
| AP1 | slope-area ORIGINAL Ks | 112.8 | 35.58 | 0.3154 | vertical units; AP1 seam |
| AP1 | slope-area ORIGINAL theta | 0.542 | 0.5407 | 0.9977 | vertical units; AP1 seam |
| AP1 | slope-area ORIGINAL R2 | 0.962 | 0.9477 | 0.9851 | vertical units; AP1 seam |
| AP1 | slope-area AD no-opt Ks | 0.5409 | 0.08929 | 0.1651 | optimizer; measurement; m; vertical units; AP1 seam |
| AP1 | slope-area AD no-opt theta | -0.07 | -0.1067 | 1.525 | optimizer; measurement; m; vertical units; AP1 seam |
| AP1 | slope-area AD no-opt R2 | 0.033 | 0.08751 | 2.652 | optimizer; measurement; m; vertical units; AP1 seam |
| AP1 | slope-area AD opt Ks | 43.55 | 2.035 | 0.04674 | optimizer; measurement; m; vertical units; AP1 seam |
| AP1 | slope-area AD opt theta | 0.46 | 0.4013 | 0.8724 | optimizer; measurement; m; vertical units; AP1 seam |
| AP1 | slope-area AD opt R2 | 0.977 | 0.976 | 0.999 | optimizer; measurement; m; vertical units; AP1 seam |
| AP1 | slope-area PARTITIONED Ks | 119.2 | 8.427 | 0.0707 | optimizer; measurement; m; vertical units; AP1 seam |
| AP1 | slope-area PARTITIONED theta | 0.553 | 0.5218 | 0.9435 | optimizer; measurement; m; vertical units; AP1 seam |
| AP1 | slope-area PARTITIONED R2 | 1 | 0.9768 | 0.9768 | optimizer; measurement; m; vertical units; AP1 seam |
| GM2 | slope-area ORIGINAL Ks | 2.488 | 2.488 | 0.9999 |  |
| GM2 | slope-area ORIGINAL theta | 0.306 | 0.3062 | 1.001 |  |
| GM2 | slope-area ORIGINAL R2 | 0.962 | 0.9621 | 1 |  |
| GM2 | slope-area AD no-opt Ks | 0.397 | 0.4157 | 1.047 | optimizer; measurement; m |
| GM2 | slope-area AD no-opt theta | -0.065 | -0.0657 | 1.011 | optimizer; measurement; m |
| GM2 | slope-area AD no-opt R2 | 0.205 | 0.2124 | 1.036 | optimizer; measurement; m |
| GM2 | slope-area AD opt Ks | 1.666 | 1.666 | 1 | optimizer; measurement; m |
| GM2 | slope-area AD opt theta | 0.273 | 0.2729 | 0.9998 | optimizer; measurement; m |
| GM2 | slope-area AD opt R2 | 0.91 | 0.9079 | 0.9977 | optimizer; measurement; m |
| GM2 | slope-area PARTITIONED Ks | 2.499 | 2.539 | 1.016 | optimizer; measurement; m |
| GM2 | slope-area PARTITIONED theta | 0.304 | 0.3064 | 1.008 | optimizer; measurement; m |
| GM2 | slope-area PARTITIONED R2 | 1 | 0.9996 | 0.9996 | optimizer; measurement; m |

(500 resolution-sweep ratio rows are in the CSV only.)

