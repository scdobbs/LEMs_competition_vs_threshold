# Addendum 2 to the v2 refactor directive: Phase 2 close-out, measurement diagnostics, Phase 3 (2026-11-09)

This addendum overrides the directive and Addendum 1 wherever they conflict.

## 0. Orientation (you are a new agent instance)

A previous instance did Phases 0–2 and the first measurement run. Before anything else, read these in order:

1. `reports/directives/refactor_v2_directive.md` and `refactor_v2_addendum_1.md`. Steve will put this file beside them.
2. `reports/v2/00_reference.md`, `02_regression.md`, `02_regression_tables.md`, `02b_measurement_first_results.md`.
3. `docs/measurement_methods.md` and `docs/v1_to_v2.md`.

Then confirm the state against the repo:
- the `v2` head commit;
- that the test suite passes (run on a compute node, not the login node);
- that the run directories cited in the reports exist under `$GROUP_HOME/lemcvt_results`.

Where this addendum and the repo disagree, the repo wins: report the disagreement and do not work around it.

## 1. Decisions already made

- **Slurm.** Use `serc` for every job. Don't use `owners`, and pass no `--account`. Rules are in §4.
- **Push.** Approved for all local-only branches, including `v2`, but only after the check in §5 comes back clean.
- **AP3.** Run it as-run (K = 1×10⁻⁶) for regression, and with K = U/k_s for the Phase 6 rerun.
- **The two environment-drift rows in the resolution sweep** (AP2 dx 88.27 and AP5 dx 60.70) are accepted. The determinism fix in §2.5 still applies.
- **The optimizer regression (Phase 2.2)** is approved, modified as described in §3.
- **The measurement threshold rule is not decided.**
  - Write no measured values into any tile config.
  - §6 asks for the diagnostics the decision needs.
  - The model's fitted A_crit stays forbidden as an input to measurement.

## 2. Phase 2 close-out: verify what the report asserted

Report: `reports/v2/02d_phase2_closeout.md`.

### 2.1 The rounding explanation is plausible but not demonstrated

The report says the AP-opt and joint differences come from parameters printed to 5 significant figures. Test this directly.

1. **Look for full-precision values first.** Candidates include pickled optimizer results, `.out` logs, extra digits in notebook outputs, and any saved arrays. If you find them, rerun forward with them. The original tolerances apply: |Δz| ≤ 10⁻⁶ × relief, relative MSE ≤ 10⁻⁸.
2. **Otherwise, do a linearized recovery.** For each AD-opt and joint run:
   - Compute J = ∂Z/∂(log K, log D) by central differences with a relative step of 10⁻⁶, holding the channel mask fixed.
   - Solve min ‖Z_saved − Z_v2 − Jδ‖₂ for δ.
   - **Pass** when two conditions both hold:
     - every |δᵢ| lies within the rounding box, i.e. half a unit in the 5th significant figure of the printed value, expressed in log space;
     - the residual's max |·| is ≤ 10⁻⁸ × relief.
   - Report δ, the box and the residual for every run.

### 2.2 The GM5 joint explanation is wrong as stated

The report attributes GM5's relative MSE difference of 6×10⁻⁶ to the rounded A_crit of 2247.1 m² "moving a few cells across the channel threshold". That cannot happen:

- **GM D8 areas are integer multiples of 1 m².** No integer lies in the rounding interval [2247.05, 2247.15), so the channel mask cannot change.
- **`n_diff_cells` can't support the claim.** It equals 996004 on every GM run, i.e. every interior cell. It counts any nonzero difference at all.

Required:

1. **Rounding-interval count.** For every A_crit-bearing run (acrit_opt and joint, all ten tiles), report how many cells have D8 area inside the rounding interval of the printed A_crit. On AP the cell area is 0.951 m², so the count may be nonzero there.
2. **Explain GM5 joint with the §2.1 test.** One hypothesis to check: v1 saved a grid from an evaluation other than the reported optimum, for example a different Nelder–Mead simplex vertex. Look at the v1 notebook code that saves the grid.
3. **If GM5 remains unexplained after 2.1 and 2.2, stop and report.** Do not start Phase 3 code changes.

### 2.3 Replace `n_diff_cells` in `regression-report`

Use counts of cells with |Δz| > 10⁻⁹ × relief and with |Δz| > 10⁻⁶ × relief, plus the count of cells where the channel mask differs.

### 2.4 Bookkeeping in the Phase 2 tables

- **The drift count.** Table 2.3a shows both the D8 and the D∞ ratios moved on AP2 (0.050 and 0.060) and on AP5. State how many of the 500 ratios are affected.
- **The duplicate row.** Table 2.3d has the dx = 50 row twice. Fix the generator, or explain why the row is there.

### 2.5 Make tie-breaking deterministic (a logged numerics change)

The drift came from `argsort` ordering of equal drainage areas. Default numpy sorts are not stable, so tie order is effectively an implementation detail.

1. **Find every affected sort.** Search the whole package for sorts on floating-point keys where tie order affects results: D8 accumulation order, BFS, main-stem selection, χ integration, and slope-area binning.
2. **Replace each one** with an explicit, stable tie-break, for example `np.lexsort((flat_index, key))`.
3. **Record each change** in a new `docs/numerics_changes.md`.
   - v1 behaviour cannot be preserved, because it depended on the numpy version. Say so in the file.
4. **Gate.** Rerun the resolution sweep and report which rows change and by how much. Changes are expected only where ties exist.

### 2.6 The D∞ partitioned model: diagnose it, don't fix it yet

**Context.** Phase 2 reproduced the v1 D∞ numbers exactly, including the suspect ones:

| tile | partitioned MSE, D∞ | partitioned MSE, D8 |
| --- | --- | --- |
| AP1 (ft²) | 3.765×10⁶ | 2195 |
| GM1 (m²) | 4.24×10³ | 792 |

The AD model is unaffected (AP1: 7.01×10⁵ with D∞ against 6.96×10⁵ with D8), so the fault lies in the partitioned D∞ path. The original directive left this investigation out of Phase 2, which was a planner error.

On AP1 and GM1:

1. **Map Z − Z_obs** for both routings. Say where the error sits: channels, hillslopes or boundaries.
2. **Check the D∞ channel set:**
   - Is every channel cell connected to the Dirichlet boundary?
   - Is channel elevation monotone upstream?
   - How are "downstream path", ΔL and A defined for χ integration when D∞ splits flow?
   - Does any channel cell's integration path fail to terminate at the boundary?
3. **Check A_crit = 5000 against the D∞ area distribution.** Report the channel-cell count under each routing at the same A_crit.
4. **Run a synthetic case** where D8 and D∞ must agree, such as a planar or V-shaped surface aligned with the grid, with a known steady-state answer.
5. **Report ranked hypotheses with their evidence, and propose a fix.** Do not change the code.

## 3. Optimizer regression (Phase 2.2): approved, run this way

The proposed submission (2 tasks × 4 CPUs × 36 h) is not safe as it stands. AP5 took 35 h in v1, so a 36 h limit on fewer CPUs risks a timeout that loses the whole run.

- **Partition:** `serc`.
- **Split by stage.**
  - Work out from the code which stages initialize from another stage's output.
  - Submit each independent stage as its own array task, and chain dependent stages with `--dependency=afterok`.
  - Each stage writes `params.json` and its trajectory when it finishes, and is resumable.
- **Time limit per stage:** 1.5 × that tile's v1 stage time (from `00_reference.md` §3.4), rounded up, and below the serc maximum. Check the maximum with `sh_part`.
- **CPUs and memory from measurements, not v1 defaults.**
  - Use `sacct` (TotalCPU/Elapsed, MaxRSS) or `seff` on the ten forward-run jobs.
  - If the solves are effectively single-threaded, request 1–2 CPUs.
  - Set `OMP_NUM_THREADS`, `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to the CPU count.
  - Request memory at 1.5 × MaxRSS.
- **Budget.** Print the core-hour estimate. Submit without asking again if it is ≤ 300 core-hours; otherwise ask.
- **Pass criteria are unchanged:** parameters within 1%, MSE within 0.5%.
- **Pre-registered rule for the A_crit stages** (staircase objective). A miss on the A_crit tolerance counts as a pass, with a note, when two conditions both hold:
  - the MSE is within 0.5%;
  - both A_crit values select the same channel mask, or evaluating both with the same code gives MSEs within 0.5%.

  Any other miss is a fail: rerun it in `lem-v1` as the directive requires.
- **These trajectories are the Phase 3 baseline.** Keep them.

## 4. Slurm and storage rules (all phases)

- **Record the partition in `configs/slurm/sherlock.yaml`** as `partition: serc`. Confirm the current limits with `sh_part` and `sinfo -Nlp serc`, and record them as comments.
- **Concurrency.**
  - Keep total concurrent CPUs ≤ 300, per the SDSS guidance for big, long-running jobs.
  - Throttle every array with `%N`.
- **No compute on login nodes.** Full-tile tests, profiling and anything beyond trivial go through `sbatch` or `sdev -p serc`.
- **Storage.**
  - Job I/O goes on `$SCRATCH` or `$GROUP_SCRATCH`.
  - Copy final results to `$GROUP_HOME/lemcvt_results`. Scratch purges files left unmodified for 90 days.
  - `$GROUP_HOME` has a 1 TB quota. Report current usage in each checkpoint.
- **Record Slurm accounting** (Elapsed, MaxRSS, TotalCPU and partition) in each run's `run.json`.
- **The Phase 2 forward runs went to the default partition.** From now on everything goes to `serc`.

## 5. Push: pre-push check, then push

For each local-only branch (`feature/channel-width`, `feature/acrit-slope`, `feature/head-extraction`, `v2`, and `fix/ap-vertical-units` if it exists), check every commit not on `origin`. Report the output before you push.

1. **Forbidden files.** None of these may be tracked:
   - anything under `papers/`, or any `*.pdf`;
   - `working-with-steve.md` or `cry_for_help.txt`;
   - `.env` or `.netrc`.
2. **Large files.** Nothing over 50 MB. GitHub rejects files over 100 MB.
3. **Secrets.** Grep the diffs for `ghp_`, `github_pat_`, `gho_`, `BEGIN .* PRIVATE KEY`, `password=` and `token=`.

The repository is public, so anything pushed is public.

- **If the check is clean,** push.
- **If anything turns up,** stop, list it and ask. Do not rewrite history without asking.

## 6. Measurement: diagnostics for the threshold decision

Report: `reports/v2/02c_measurement_diagnostics.md`.

**Two framing points.** Address both explicitly in the report.

- **θ = m in the models.** The threshold rule therefore changes the model fitted at each tile, not just K₀.
- **k_s has units of m^{2θ}.** k_s values at different θ are not comparable. Table 2 of `02b` places k_s at θ_roll, at θ_plateau and at the v1 θ side by side. Do not compare k_s across different θ except through the common-θ column in §6.4.

### 6.1 A basin-fixed ladder

The θ rise with A_c could come from either of two effects:
- **pixel population:** hollow and colluvial pixels at low A;
- **basin selection:** small basins dropping out as A_c rises.

To separate them:
1. Fix the set of basins that hold ≥ N_min channel pixels at A_c = 10⁴ m². State N_min before you run.
2. Vary A_c from 10² to 10⁴ m² within those basins only.
3. Compare the result with the free ladder.

### 6.2 A finer ladder with honest intervals

Use 0.1-decade steps from 10² to 10⁵ m².

Fix the per-basin minimum χ range and minimum relief before running, and state them with their rationale.

At each step, report:
- θ*;
- k_s(θ*);
- the number of basins and the number of channel pixels;
- the χ–z fit R²;
- bootstrap 68% and 95% intervals.

Then:
- **State the bootstrap unit** (basin or segment).
- **Flag every step with fewer than 10 basins.**
- **Plot θ(A_c)** with uncertainty bands, per tile and pooled by site.

### 6.3 A slope–area scaling break, per tile

1. Fit a piecewise-linear log S – log A model above the rollover, choosing the breakpoint by maximum likelihood.
2. Compare it with one- and three-segment fits by BIC.
3. Bootstrap the breakpoint.
4. Report the break per tile and pooled per site.

**Do not impose a site-independent threshold.** Channel-initiation area is expected to differ between landscapes and with slope (Montgomery & Dietrich 1989, 1992). As a report-only alternative, give the distribution of A·S² at the scaling break.

### 6.4 A common-θ k_sn, for comparison only

Report k_sn at two reference concavities:
- θ_ref = the site-pooled θ from §6.2, evaluated at the §6.3 break;
- θ_ref = 0.45.

This puts every tile in the same units. It is not an input to the models.

### 6.5 Tile-edge truncation

1. **Provenance.** State whether the `*_area` grids were computed on the 1 km tile alone or on a larger DEM and then clipped. Base this on Phase 0 findings and `old-notebooks/`.
2. **Truncated catchments.** At each A_c, report the fraction of channel pixels in basins whose divide reaches the tile edge, since their true drainage area is underestimated.
3. **Interior-only θ.** Report θ(A_c) restricted to basins whose divides lie fully inside the tile, where there are enough of them.

### 6.6 Hilltop curvature

1. **Confirm the curvature definition.** C_ht must be the Laplacian ∇²z, consistent with D₀ = U/|∇²z| and the model's diffusion operator. State this in `measurement_methods.md`.
2. **Synthetic test:**
   - Build parabolic ridges with a known ∇²z and crest widths spanning the observed hilltop widths at each site.
   - Run them with and without triangular facets sized like the AP TIN facets, and with noise.
   - Report recovered C_ht against window size, and the bias at each site's chosen L*.
3. **Pre-registered rule.** If the bias exceeds 10% at either site's L*, report it and stop for a decision on the curvature method.
4. **Show the C_ht(L) curve for every tile.**

### 6.7 AP measurement provenance (read-only; no downloads)

1. **Locate the 990 × 990 grids at 0.9998 m** that the legacy AP notebooks used. Report their paths, dates and georeferencing.
2. **Run the legacy method on them** and say whether it reproduces the hard-coded AP θ and k_s.
3. **Compare the two AP grid families.** The model tiles' 0.9752 m spacing is close to PAMAP's native 3.2 ft (0.9754 m). Report whether one grid family is a resampling of the other.

### 6.8 Report outline (fixed)

1. What was run.
2. Basin-fixed against free ladder.
3. Fine ladder.
4. Scaling break.
5. Common-θ k_sn.
6. Edge effects.
7. Curvature checks and synthetic test.
8. AP provenance.
9. What the evidence supports, and where it is weak.
10. The options for the threshold rule, each with its consequences for θ, k_s and K₀ per tile.

A recommendation is allowed, but keep it separate from the evidence.

## 7. Phase 3: performance (start once §2 is closed; can run alongside the §3 jobs)

Follow the directive's Phase 3, in this order:

1. **Profile GM1 and AP5** with `cProfile` and `line_profiler` in one `serc` job.
2. **The nonlinear stage.**
   - Vectorize the Picard matrix assembly; it dominates forward and optimizer wall time.
   - Reuse the symbolic factorization where the sparsity pattern is fixed.
3. **AD.** Impose Dirichlet rows without the CSR → LIL → CSR round trip.
4. **The joint stage.** Precompute χ_D8 and the anchor elevations once per tile.
5. **D8 accumulation and BFS.** Speed these up with numba or vectorized topological ordering.

**Working rules.** One logical change per commit. Run the forward regression after each numbered step.

**Gate:**
- **Forward.** Compare Phase 3 code with the Phase 2 v2 outputs at identical parameters, on all ten tiles.
  - The tolerance is max |Δz| ≤ 10⁻⁸ × relief and relative MSE ≤ 10⁻¹⁰.
  - The §2.5 tie-break change is exempt only where `numerics_changes.md` predicts a difference.
- **Optimizer.** Compare GM1 and AP5 with the §3 baseline, applying the §3 rule.
- **Report** per-stage speedup, peak memory and CPU efficiency. Update the Slurm defaults from these measurements.

Report: `reports/v2/03_performance.md`. **Stop for review.**

## 8. Order of work and stops

1. Do the §5 push check; push if it is clean.
2. Do §2. Deliver `02d_phase2_closeout.md` as a checkpoint.
   - **Stop everything** if GM5 (§2.2) is unexplained.
3. Submit the §3 jobs.
4. Do §6 while the jobs run. Deliver `02c_measurement_diagnostics.md` as a checkpoint.
5. Do §7, and stop at its gate.

**Every report must:**
- state plainly what the evidence supports and where it is weak;
- separate what was measured from what was inferred;
- tie every number to a run ID.

Do not fix numerical issues outside this addendum silently. Report them at the next checkpoint.
