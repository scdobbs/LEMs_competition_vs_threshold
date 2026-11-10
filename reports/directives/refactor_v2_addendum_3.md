# Addendum 3 to the v2 refactor directive: Phase 3 close, three audits, Phase 4 (2026-11-09)

This addendum overrides the directive and Addenda 1–2 wherever they conflict.

## 0. Orientation (you are a new agent instance)

Before doing anything, read these in order:

1. `reports/directives/refactor_v2_directive.md`, then `refactor_v2_addendum_1.md`, `refactor_v2_addendum_2.md`, and this file.
2. The reports: `reports/v2/00_reference.md`, `02_regression.md`, `02b_measurement_first_results.md`, `02d_phase2_closeout.md`, `02c_measurement_diagnostics.md` (with `02c_tables.md`), and `03_performance.md` (with `03_tables.md`).
3. The docs: `docs/numerics_changes.md` and `docs/measurement_methods.md`.

Then confirm the state of the repo:

- the `v2` head commit;
- that the tests pass, run on `serc`;
- the status of the optimizer jobs listed in `03_optimizer_regression_jobs.json` and `03_optimizer_phase3_jobs.json`.

Where this addendum and the repo disagree, the repo wins. Report the disagreement and stop.

All the Slurm, storage, push and reporting rules of Addendum 2 §4, §5 and §8 still apply.

## 1. Decisions

Steve will strike any he does not accept before this reaches you.

- **Push: option (a).** The ten figure PDFs under `reports/v2/figures_v2/` are generated figures, not papers, so they may be pushed. Push all four branches.
  - Going forward, generated figures (PDF and PNG) are not tracked. Add `reports/v2/figures_v2/` to `.gitignore` in a new commit. `lemcvt figures` regenerates them, and the reports keep citing their paths.
- **GM5 is not added to the optimizer regression.** The regression tests that v2 reproduces v1's optimizer; it does not test whether v1's optimizer found the optimum. Phase 4 answers that on all ten tiles, GM5 included.
- **The D∞ partitioned model gets fixed after Phase 4, as §6 ("Phase 4b").** It does not go into Phase 3, because Phase 3 must leave the numerics unchanged. The defective model is kept, unchanged, as `partitioned_threshold_model_dinf_v1`, for the record.
- **Curvature method: implement option (i)** (exclude near-channel pixels) as specified in §4. Report C_ht and D₀ both with and without the exclusion.
- **The threshold rule for θ and k_s is deferred until Phase 4 results exist.**
  - One option is added, and Phase 4 provides it (§5.2): fit the exponent m within the partitioned model, then test the fitted m against the measured θ(A_c) curve at the fitted A_crit.
  - No measured θ, k_s or D₀ goes into any tile config yet.

## 2. Close Phase 3

When the six baseline stages and the six Phase 3 optimizer stages finish:

1. **Apply the Addendum 2 §3 pass rule** to every stage, comparing:
   - the baseline against v1;
   - Phase 3 against the baseline.
2. **Compare trajectories row by row.** Steps 2, 3 and 5 are bit-identical, so the Phase 3 AD, A_crit and nonlinear stages must reproduce the baseline `trajectory.parquet` exactly: the same evaluations, in the same order, with the same misfits.
   - Report the first row that differs, if any.
   - For the joint stage, report the largest per-evaluation misfit difference, and whether any Nelder–Mead step differed.
3. **Add the comparison to `make_03_tables.py`** and write `reports/v2/03b_phase3_close.md`.
4. **If AP5's nonlinear stage hits its 32 h limit,** report it. Do not resubmit; §3 bears on whether that stage is meaningful in feet at all.

## 3. Audit: Picard convergence of the nonlinear model

`03_performance.md` says AP needs 50 Picard iterations while GM needs 10–20, and 50 is the iteration cap.

**Hypothesis to test** (do not assume it). Because AP z is in feet and x, y in metres, every AP gradient is 3.2808× too steep relative to the dimensionless S_c = 1.27. That pushes hillslope cells toward |∇z| → S_c, where D_eff = D/(1 − (|∇z|/S_c)²) diverges and the Picard iteration stalls.

1. **Instrument the iteration** without changing it: log every Picard iteration's update norm and its relative residual, plus whether the stopping test or the cap ended the loop. Pass the information through to `params.json`.
2. **Run the nonlinear forward model** at the v1 printed nonlinear optimum on all ten tiles, AP in feet as v1 ran. For each tile, report:
   - the iterations used;
   - the final update norm and residual;
   - whether the loop converged or hit the cap;
   - the fraction of hillslope cells with |∇z| > 0.5 S_c and with > 0.9 S_c, and the maximum |∇z|/S_c.
3. **Repeat on AP1 and AP5 in metres.** Use a metre copy that exists only in memory, by passing `reproduce_v1_units: false` on the command line. Do not change the configs. D must be scaled by 1/3.2808 so the linear part is equivalent. Report the same quantities.
4. **Decision rule, stated now.**
   - If any AP tile hits the cap in feet, the v1 AP nonlinear results are unconverged and are reported as such in the Phase 6 old-versus-new record.
   - If the metre runs converge, the hypothesis is supported.
   - Either way, nothing is fixed here; Phase 5 makes the unit correction.

## 4. Audit: the χ estimator and the hilltop set

### 4.1 The basin-fixed test cannot discriminate as run

`02c` concludes that the θ rise is "a pixel-population effect, not basin selection", because the basin-fixed θ* equals the free θ* within 0.01. But the same report notes that the pooled through-origin SSR is dominated by the largest basins, so that 2 basins reproduce the fit of 110. If the largest basins dominate both fits, the free and fixed fits must agree whatever the cause. The test therefore cannot separate the two effects as constructed.

1. **Report n_eff at every ladder step**, per tile and pooled: n_eff = 1/Σwᵢ², where wᵢ is basin i's share of the total SSR weight (Σ(χ − χ̄)² over its pixels). Add it to T1 and T2.
2. **Add an equal-weight estimator.**
   - Fit θ by minimizing Σᵢ SSRᵢ(θ)/SSRᵢ,null, where SSRᵢ,null is basin i's SSR about its own mean z. Each basin then counts equally.
   - Also report the median over basins of the per-basin best θ, using only basins that pass the filters.
3. **Rerun the free ladder, the basin-fixed ladder and the site pooling** with both estimators. State whether the pixel-population conclusion survives the equal-weight estimator.
4. **Report n_eff beside every value in T8.** Where n_eff < 3, the bootstrap interval is not a usable uncertainty; say so.

### 4.2 Truncation is a sample-size issue, not a bias

The area grids hold true areas (clipped from a larger DEM), and χ is integrated upstream from each in-tile outlet with those areas. A truncated catchment therefore yields a valid, shorter χ–z profile.

- Correct `docs/measurement_methods.md` and the `02c` wording to say this.
- Drop the interior-only ladder from future reports, unless the equal-weight estimator shows a difference between truncated and complete basins. Report that comparison once, at A_c = 1e3 m².

### 4.3 Hilltop exclusion (curvature option i)

1. **Exclude near-channel pixels.** Drop hilltop pixels closer than L*/2 to a channel pixel. Channel pixels are those of the network at A_c = 1e4 m², which the current hilltop definition uses.
2. **Report per tile, with and without the exclusion:**
   - pixel count and segment count;
   - C_ht with its bootstrap interval;
   - D₀ = U/C_ht.

   Repeat for channel networks at A_c = 1e3 m² and 1e4 m², to show how strongly the hilltop set depends on the network threshold.
3. **Rerun the synthetic-ridge bias calculation** on the post-exclusion width distribution, and report the bias at the median and 25th-percentile widths.
4. **State one known limitation in the report.** The synthetic assumes a crest that is parabolic all the way to the channel, which is the steady linear-diffusion profile. Real crests with planar flanks are narrower than 2 × the divide-to-channel distance, so the bias is a lower bound.

Write §3 and §4 up together in `reports/v2/02e_audits.md`, with a fixed outline:

1. Picard audit.
2. n_eff and the equal-weight estimator.
3. Truncation wording.
4. Hilltop exclusion.
5. What the evidence supports, and where it is weak.

This is a checkpoint, not a stop.

## 5. Phase 4: optimizer v2

Implement Phase 4 as the directive specifies (§4.1–4.5), with AP still on `reproduce_v1_units: true`. Add the following.

### 5.1 Additions to the directive's Phase 4

- **The A_crit grid.** Use every unique drainage-area value between 10² and 10⁵ m², not a thinned set. The step locations are known exactly, and each evaluation costs a few seconds, so the full staircase is affordable.
  - Print the core-hour estimate first, under the Addendum 2 §3 budget rule.
  - If the estimate exceeds 300 core-hours, thin the grid to the 2000 log-spaced unique values nearest a uniform log grid, and say so.
- **Optimality certificate.** For each tile and stage, report the v2 global optimum together with the v1 optimum evaluated by the same code. For the v1 joint optima, also report the MSE gradient at v1's point, as in the `02d` §2.1b table. The GM5 joint case is the known example.
- **Identifiability.**
  - For the partitioned profile, report the A_crit interval within which MSE is no more than 1% above the minimum. Report it also at 5%.
  - Report how K and D vary across those intervals.
  - Report the same for r in the AD profile.

### 5.2 Extra, flagged: fit the exponent m in the linear partitioned model

The χ values change with m, so b changes, but the reduction Z = a + (U/K)·b + (U/D)·c still holds at fixed (A_crit, m).

1. **Profile over m** from 0.20 to 0.70 in steps of 0.01, on a log-spaced A_crit grid of 200 unique values. This produces MSE(A_crit, m), with the non-negative-least-squares (U/K, U/D) solution at each point.
2. **Report per tile:**
   - the joint optimum (A_crit, m, K, D);
   - the 1% contour;
   - the profile MSE(m), minimized over A_crit and K, D.
3. **Plot the measured θ(A_c) curve on the same axes** (from `02c`, or from the equal-weight estimator if §4.1 changes it), with the fitted (A_crit, m) and its 1% contour.

Do not interpret the plot in the report beyond saying whether the fitted point lies inside the measured curve's 95% band. Interpretation is Steve's.

### 5.3 Required tests

Run the directive's three tests, plus a fourth: the reduction holds at three values of m on a cropped tile, to 10⁻¹⁰ relative.

### 5.4 Report and stop

Write `reports/v2/04_optimizer_v2.md`, following the directive's outline, with §5.1 and §5.2 as additional sections.

**Stop for review.**

## 6. Phase 4b: rebuild the D∞ partitioned model (only after the Phase 4 review)

Implement the coupled system proposed in `02d` §2.6.6 as `partitioned_threshold_model_dinf`. It is one sparse linear system:

- channel cells (A_dinf ≥ A_crit) get D∞ upwind advection rows with D = 0;
- hillslope cells get Laplacian rows with K = 0;
- boundary cells get Dirichlet rows.

The old model stays as `partitioned_threshold_model_dinf_v1`.

**Tests:**

1. **The aligned V-valley and the plane:** the D∞ and D8 partitioned models agree to 10⁻¹⁰ of relief.
2. **The rotated V-valley:** compare the D∞ partitioned model with the D8 partitioned model on the same surface at 4× finer resolution, downsampled. Report the difference; no pass threshold.
3. **The Phase 4 reduction** holds for the new model, to 10⁻¹⁰ relative.
4. **Channel monotonicity:** zero channel cells lower than every receiver with a nonzero weight, on AP1 and GM1.

Then rerun `d8_vs_dinf` on AP1 and GM1 (AP still in feet), and report the new MSE table beside the v1 one.

Report: `reports/v2/04b_dinf_rebuild.md`. **Stop for review.**

## 7. Order of work and stops

1. Push, per §1.
2. §2 as the jobs finish; deliver `03b`.
3. §3 and §4; deliver `02e` as a checkpoint.
4. §5 (Phase 4). **Stop at `04_optimizer_v2.md`.**
5. §6 only after Steve's review of Phase 4.

Do not fix numerical issues outside this addendum silently. Report them at the next checkpoint.
