# [2026-11-13 05:40 PST] Addendum 9 to the v2 refactor directive: a fair test of the nonlinear hillslope model (robust solver, S_c uncertainty, and a morphology score beside the MSE)

This addendum overrides the directive and Addenda 1–8 wherever they conflict. Addendum 4 §1 (timestamped titles, one report per checkpoint), Addendum 2 §4–§5 (Slurm, storage, push) and Addendum 7 §0.4 (submit guard) still apply.

## 0. Orientation (you are a new agent instance)

1. **Read, in order:**
   - the directive and Addenda 1–8 in `reports/directives/`;
   - this file;
   - `reports/v2/20261113T0450PST_ER_threshold_phase7.md` (latest state), then `20261112T1600PST_d_diagnostic.md` and `20261112T1003PST_phase6_results.md`;
   - `docs/measurement_methods.md` (§10 defines E*–R*) and `docs/numerics_changes.md`.
2. **Confirm** that `v2` equals `origin/v2`, that the tests pass on `serc`, and that no earlier job is running.
3. **Where this addendum and the repo disagree,** the repo wins. Report it and stop.

## 1. Decisions

Steve will strike any he does not accept before this reaches you.

1. **Linear limit.** The Addendum 8 reading stands: the linear limit is R* = E*/4, and the rule is applied to the normalised ratio 2S̄/(C_ht L_H), which is 1 on a linear hillslope.
2. **Addendum 8 §2.4 is recorded as "not testable with the Picard solver".** It is not rerun as specified. This addendum replaces it.
3. **Network.** The 10³ m² network stays decisive for every E*–R* quantity here. The rollover and 10⁴ m² networks are reported as sensitivity.
4. **Not in this addendum:** the balanced hold-out rerun of Addendum 8 §3.3, Test 2 beyond its review gate, and CI (which stays manual).

**Why this addendum.** The E*–R* figures show hillslopes with sharp crests over flanks whose mean gradient saturates. That is the signature of slope-dependent transport. Yet no nonlinear model run so far has fitted the topography better than linear diffusion. Three reasons are possible, and this addendum separates them:

- **the nonlinear model was never run fairly.** At S_c = 1.27 (Phase 6) the nonlinearity barely engaged. At the fitted S_c (Addendum 8) the Picard iteration failed below D ≈ 10⁻² m²/yr;
- **the elevation MSE does not reward the right hillslope form,** because it is dominated by channels and large-scale relief;
- **the simple nonlinear law is too crude** to fit these surfaces better than an effective linear D.

## 2. A robust solver for the nonlinear partitioned model

**Problem.** `partitioned_threshold_model_nonlinear` solves the steady hillslope equation by Picard iteration, with D_eff = D / (1 − (|∇z|/S_c)²) evaluated at cell centres (the `_compute_slopes` stencil) and the slope ratio clipped at 0.99. At low S_c and low D this stalls at the cap.

### 2.1 Implement a Newton solver for the same discrete equations

- **Same equations.** Solve exactly the discrete system the Picard iteration converges to: the same stencil, the same D_eff, the same clipping, the same channel and boundary values. A converged Picard solution must also be a solution of the Newton system.
- **Jacobian.** Use an analytic sparse Jacobian of the residual with respect to the hillslope elevations, including the derivative of D_eff through the slope stencil. Check it against finite differences in a test.
- **Globalisation.** Use a backtracking line search on the residual norm.
- **Continuation in D.** Along each A_crit row, start at the largest D and step down the grid, initialising each solve from the converged solution at the previous D. Start the largest D from the linear solution.
- **Stopping.** Converged when the relative residual norm is ≤ 10⁻¹⁰ and the maximum update is ≤ 10⁻⁶ m. Cap at 50 Newton iterations.
- **Classify every evaluation as one of three outcomes:**
  - `converged`;
  - `no_steady_solution`: the clip is active on more than 1 % of hillslope cells at the last iterate, so the flux cannot be carried below S_c;
  - `not_converged`: anything else.

  Report the three counts separately. Do not treat clipped solutions as converged.
- **Keep Picard as the reference.** Select the solver with a `solver` option (`picard` or `newton`). Picard stays the default for every existing command.

### 2.2 Tests (the gate for §3)

1. **Agreement with Picard.** Where Picard converges (Phase 6 settings, synthetic and cropped real tiles), Newton reproduces Picard to max |Δz| ≤ 10⁻⁶ m.
2. **The 1-D analytic profile.** On a 1-D hillslope strip, R* from the Newton solution matches the integrated steady nonlinear curve (`lemcvt.analyses.er_star.integrate_profile_R`) within the discretisation error. Show that the error falls as the grid is refined, at E* = 1, 5 and 20.
3. **The Jacobian** matches finite differences to 10⁻⁶ relative.
4. **Continuation.** On AP1 at its Addendum 8 fitted S_c (0.306), one A_crit row across the full D grid: report the outcome class and iteration count at every D. Newton must converge at every D where Picard converged.

If test 4 shows `no_steady_solution` for most low-D evaluations, stop after §2 and report. A higher cap or a different solver would not help in that case, and §3 would test nothing.

## 3. The nonlinear joint profile, run fairly

Run only after the §2 gate.

### 3.1 Runs

**Per tile, at three values of S_c:** the Addendum 8 primary-network fit and the 16th and 84th percentiles of its segment bootstrap (from run `20261113T015030Z_er_star_10tiles`).

**Grid and settings:**
- K fixed at the linear joint optimum (the refined Phase 6 run);
- the full Addendum 6 A_crit grid, 30 values;
- D: 22 log-spaced values from 10⁻³ to 5 × 10⁻² m²/yr, as in Addendum 8;
- the Newton solver with continuation.

Split into parallel parts as before, by A_crit, so that continuation in D stays inside each part.

**Report per tile and S_c:**
- the optimum (A_crit, D) and its MSE against the linear joint MSE;
- D/D₀,A, and D/D₀,B on AP;
- the three outcome counts;
- the fraction of hillslope cells above 0.5 S_c at the optimum;
- the 1 % and 5 % contours.

### 3.2 Rule, written now

Apply it per site, at the central S_c, with the two bootstrap S_c reported beside it. The nonlinear model is **preferred** at a site if, on at least 4 of 5 tiles, both of these hold:

- **fit:** the optimum MSE is at most 1.05 × the linear joint MSE;
- **D recovery:** D/D₀,A lies within 0.67–1.5. On AP, accept anything within the D₀ bracket widened by × 1.5 on each side.

If only the D condition holds, report "closer D at a cost in fit" with the MSE ratio. If neither holds, report which failed on which tiles.

A tile with more than 30 % of evaluations in `not_converged` is reported as untested and counts as failing. `no_steady_solution` evaluations are physical outcomes, not failures. They are excluded from the optimum and counted.

## 4. A second score: does the model reproduce the hillslope form?

Score every fitted surface with the same measurements as the data, beside the MSE.

### 4.1 Surfaces

For every tile:
- the linear joint optimum (Phase 6);
- the AD optimum (Phase 6);
- the nonlinear joint optimum at S_c = 1.27 (Phase 6 merged runs);
- the §3 nonlinear optimum at the central S_c, written as a grid.

### 4.2 Measurements

Apply each to the observed DEM and to each model surface, on the same option (iii) pixels and the same D8 paths, with the Addendum 8 code:

1. **Crest curvature:** median C_ht at L_f = 11 m. Report the ratio model/observed.
2. **E*–R* ratio:** the median normalised ratio 2S̄/(C_ht L_H) on the 10³ m² network, with S̄ and L_H measured on the model surface along the observed D8 paths. Report model against observed.
3. **Hillslope gradients:** the median and the 90th percentile of |∇z| on the model's hillslope cells. Use the models' own `_compute_slopes` stencil for both surfaces.

Plot the E*–R* figure of each model surface over the observed points: one panel per tile, at the observed fitted S_c.

### 4.3 Rule, written now

A surface **reproduces the hillslope form** on a tile if all three measurements lie within a factor 1.25 of the observed value: |log₁₀(model/observed)| ≤ log₁₀ 1.25. Report per tile and per site, and count the tiles that pass for each surface.

The question this answers: does any model match both the topography (MSE) and the hillslope form? No combined score is pre-registered. Report the two side by side.

## 5. Cost

1. **Pilot first.** Run the §2.2 test 4 row on AP1 and the same row on GM1, as a single short job. Measure the wall time per Newton solve and per continuation step.
2. **Estimate.** From the pilot, print the core-hour estimate for §3 (10 tiles × 3 S_c × 660 evaluations) and for §4 (cheap: measurements only, plus one forward solve per §3 optimum).
3. **Submit** without asking only if:
   - the total is at most 900 core-hours;
   - every submission is at most 300 core-hours;
   - every array is throttled, with concurrent CPUs well under 300.

   Otherwise stop and report the estimate.
4. Write partial results after every A_crit row, so that a timeout loses at most one row.

## 6. Report and stop

One report, `reports/v2/<YYYYMMDDTHHMM>PST_nonlinear_fair_test.md`, titled with its timestamp. Fixed outline:

1. State check.
2. The Newton solver and its tests (§2), with the AP1 and GM1 pilot rows.
3. The nonlinear joint profile at fitted S_c (§3), with the rule applied.
4. Morphology scores of every surface (§4), with the rule applied, beside the MSE.
5. What the evidence supports, and where it is weak. In particular, say which of the three reasons in §1 the results support.
6. Decisions needed.
7. Appendices: all generated tables.

**Stop for review.**

Do not fix numerical issues outside this addendum silently. Report them.
