# Phase 3: performance (addendum 2, section 7)

Directive: `refactor_v2_directive.md` Phase 3 as ordered by addendum 2, section 7. Branch `v2`,
commits 9d046e3 (step 2), 898467c (step 3), db11fee (step 4), a224750 (step 5), a6f8c64
(evaluator fix). Environment `lemcvt` plus `line_profiler` 5.1.0 (added with `pip install
--no-deps`, recorded in `envs/sherlock.yml`). All jobs on `serc`. Date: 2026-11-09.
Machine-generated tables: `03_tables.md` (`reports/v2/scripts/make_03_tables.py`; P1 profile,
G1 gates, G2 wall times, O1 optimizer stages). Every number is tied to a run id or job id.

## Summary

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

## 1. Profile (step 1)

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

## 2. The five steps, each with its forward gate

Working rules followed: one logical change per commit; the v1 implementation of every changed
routine is kept under a `_v1` name and a test asserts bit-identity on synthetic tiles
(`tests/test_models.py::TestPhase3*`, 150 -> 159 tests); the forward regression ran after each
step on all ten tiles against the Phase 2 v2 outputs (`lemcvt tile forward --params v1
--compare --reference-run phase2`, tolerance max |dz| <= 1e-8 x relief and relative MSE
<= 1e-10; table G1). Steps 3 and 4 were gated from shared clones checked out at their commits
(`$SCRATCH/lemcvt_snap_898467c`, `..._db11fee`) because the main tree had moved on.

### 2.1 Step 2: vectorised Picard assembly (9d046e3)

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

### 2.2 Step 3: AD stencils from index arrays, Dirichlet rows at build time (898467c)

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

### 2.3 Step 4: joint stage integrates the channel network once (db11fee)

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

### 2.4 Step 5: D8 receiver tables, donor lists and BFS with numpy and numba (a224750)

`precompute_channel_network` and `precompute_chi_network` now build `downstream_of`, `dL` and a
CSR of donors with numpy and run the breadth-first integration in a numba loop (Python
reference kept); `d8_area`'s accumulation loop is compiled. Arithmetic and visiting order are
v1's: bit-identical (tests on the channel network and on the area grid). The priority-flood fill
(23 s, heap in Python) was left alone: it is used only by the resolution sweep on coarse grids
and by tests.

Gate (job 51700412, runs `20261109T2148..Z_tile_forward_<tile>`): 50 of 50 pass, max 1.26e-12.

### 2.5 Evaluator fix found by the gate (a6f8c64)

`forward.evaluate` (and the close-out's `_evaluate`) used `cache.setdefault("AD",
advection_diffusion_model(...))`, which evaluates its argument eagerly and so rebuilt the AD
model on every call; the Phase 2 forward runs paid the 170 s build twice per tile and the
close-out's linearized recovery five times per tile. Numerics unaffected; fixed. `run.json` now
also records the git hash at run start (the hash at finish was wrong whenever a commit landed
during a job).

## 3. Speedup, memory and CPU efficiency

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

## 4. Optimizer gate (GM1 and AP5 against the section 3 baseline, section 3 rule)

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

## 5. What the evidence supports, and where it is weak

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

## 6. Housekeeping

- Three step-3 gate tasks (51700403_7, _8, _9) were cancelled with `scancel` after finishing
  their work but not exiting (section 2.2); see the session summary for the node ids in case
  Steve wants to report them.
- The job script's temporary tee log now goes to the log directory instead of the repository
  root (template fix, 2026-11-09).
- `$GROUP_HOME` at this checkpoint: see the session summary (`sh_quota`).

## Stop

Phase 3 stops here for review, with the optimizer gate's six remaining stages running under
the job ids above. Nothing is pushed (section 5 of addendum 2 is still blocked on the figure
PDFs).
