# [2026-11-13 13:30 PST] A fair test of the nonlinear hillslope model (addendum 9): the Newton solver, the model-form finding, and the stop after section 2

Directive: `reports/directives/2026-11-13T0540PST_refactor_v2_addendum_9.md` (outline fixed by its section 6). Branch `v2`; code
commits 4c64633, 6e96acc, ccf119e and the commit carrying this report. Environment `lemcvt` (Python 3.12, numpy 2.5.4, scipy
1.18.1); every job on `serc`. Machine-generated tables: section 7 (appendices E1-E6, `reports/v2/scripts/make_addendum9_tables.py`);
figure `reports/v2/figures_v2/a9_pilot.png` (git-ignored; `lemcvt figures a9_pilot`). Every number is tied to a run id under
`$GROUP_HOME/lemcvt_results` or a Slurm job id. **This is a stop for review** (section 6): the pre-registered stop rule of 2.2 applies,
and the section 3 estimate exceeds the section 5 limit, so sections 3 and 4 were not run.

## Summary

| Item | Result |
| --- | --- |
| Model form (found at the start; asked) | The repo's nonlinear model solves **D_eff(abs grad z) lap(z) = -U**, with D_eff at the cell centre multiplying the 5-point Laplacian. That is not the Roering law div(D_eff grad z) = -U. In 1-D its steady slope is S_c tanh(U x/(D S_c)) and R* = (2/E*) ln cosh(E*/2): 6 % above the Roering curve at E* = 1, 22 % at E* = 5, and its slope reaches the 0.99 S_c clip about 20 times closer to the crest. Every nonlinear result so far (v1, Phase 6, addendum 8 2.4) is for this equation. Steve decided in the session: add the divergence ("flux") form as an option, Newton for both forms, and continue to sections 3-4 with the flux form if it passed 2.2 under the section 5 budget rules. Logged as N10 and N11 in `docs/numerics_changes.md` |
| 2.1 solver | Newton with an analytic sparse Jacobian through the slope stencil, Armijo backtracking, continuation in D, the three outcome classes; `solver=picard/newton`, `form=centre/flux`; Picard on the v1 (centre) form stays the default and is bit-identical to before |
| 2.2 test 1 | **Pass.** At the Phase 6 setting (S_c = 1.27) Newton reproduces Picard to 2.2e-9 m (AP1) and 4.8e-8 m (GM1) on the full tile, 2.0e-9 and 4.1e-8 m on a 300 x 300 crop; against a tightened Picard (1e-10 m) to 1.4e-11 to 5.2e-11 m. Also on the synthetic tile, both forms |
| 2.2 test 2 | **Pass for the flux form; the v1 form converges to a different curve.** Flux form: error against `integrate_profile_R` 2.3e-3, 5.8e-4, 1.5e-4 at E* = 1 and 4.5e-3, 1.1e-3, 2.7e-4 at E* = 5 and 8.4e-3, 2.2e-3, 5.6e-4 at E* = 20, for dx = 1, 0.5, 0.25 m (second order). v1 form: converges to (2/E*) ln cosh(E*/2) (error 2.5e-5 and 2.9e-4 at dx = 0.25, E* = 1 and 5), +6 % and +22 % off the Roering curve; at E* = 20 it is `no_steady_solution` (72-74 % of cells clipped) |
| 2.2 test 3 | **Pass.** Jacobian against central differences: max error / max entry 2.9e-9 (v1 form), 7.1e-10 (flux form); entrywise 1.0e-8 and 2.9e-7 on entries above 1e-3 of the maximum (h = 1e-4; the error grows as 1/h below that, i.e. roundoff) |
| 2.2 test 4 | AP1 at S_c = 0.306 and GM1 at 0.544, one A_crit row, 22 values of D: **every evaluation, Picard and Newton alike, is `no_steady_solution` (more than 1 % of hillslope cells clipped), 44 of 44 per tile.** Where Picard met its stopping test (AP1: the 9 values D >= 1.13e-2; GM1: the 10 values D >= 9.35e-3) Newton met its stopping test too, at the same clip fraction and MSE to the printed digits, and on AP1 also at 3 lower D (6.4e-3 to 9.4e-3) where Picard hit the cap; below that Picard hits the cap and Newton stops on a failed line search or its cap. **Stop rule (2.2) applies on both tiles** (16 of 16 evaluations below the linear joint D are `no_steady_solution`) |
| Flux form, pilot | AP1: `no_steady_solution` at all 3 D tested (12-71 % clipped). GM1: `converged` at all 3 (0.8 % clipped at each); at D = 9.35e-3 (1.04 x D0_A) MSE 38.3 against the linear joint 24.7, on one A_crit row |
| Why (diagnostic) | At the addendum 8 fitted S_c, **31-52 % (AP) and 11-48 % (GM) of the observed hillslope cells already have a 1 m gradient above 0.99 S_c**; 1-2 % of those are next to a channel or boundary cell. The fitted S_c is a path-mean quantity (S_c ~ Sbar/0.8); cell-scale gradients exceed it widely |
| 5.2 estimate | Section 3 with the flux form: 990 s (AP) and 321 s (GM) per evaluation, so about **3600 core-hours** (AP 2720, GM 880), against the limit of 900. Not submitted |
| Sections 3 and 4 | **Not run** (2.2 stop rule; the estimate also exceeds 5.3). Code for both is written and tested on synthetic grids only |
| Cost | 11.2 core-hours used (E6; about 125 core-hours of limits); `$GROUP_HOME` 266 GB of 1 TB |

## 1. State check

**At the start.** `v2` was at 0189702 = `origin/v2`; no job was running. The tests passed on `serc` on the code of this
addendum (job 52199605: ruff clean, 246 passed with `LEMCVT_DATA` unset); mypy reported 6 errors in the new module
`analyses/nonlinear_fair.py` (pandas indexing types and an untyped rule dict). They were fixed in 6e96acc, no ignore was added;
the final check of the reported tree (job 52230236) is clean: ruff, mypy (106 files), 246 passed.

**Where the addendum and the repo disagree.** Section 2.2 test 2 asks the Newton solution to match "the integrated steady
nonlinear curve (`integrate_profile_R`) within the discretisation error". That curve belongs to the divergence form; the repo's
model is not that form (summary, and section 2.1 below). By section 0.3 I stopped and asked rather than proceeding. Steve's
answer, given in the session (not in the addendum): add the flux form, Newton for both forms, and run sections 3-4 with the flux
form if it passes 2.2, under the section 5 budget rules. Everything below follows that decision.

**Other deviations, stated.** (1) A diagnostic not in the addendum, the slope-exceedance run (E5), because it explains the test 4
result. (2) The flux form's pilot uses 3 values of D per tile, not 22: a full flux row costs about 6 h (AP1) per tile, and three
values are enough for the outcome classes and the cost per solve. (3) Three short interactive `srun` diagnostics on `serc`
(E6) while designing the solver. (4) A GMRES option for the flux form's Newton step was tried and failed (section 2.2); the
reported runs use the direct LU.

## 2. The Newton solver and its tests (section 2), with the AP1 and GM1 pilot rows

### 2.1 What the code solves, and what was added

On each hillslope cell the v1 model (`partitioned_threshold_model_nonlinear`, unchanged since v1) assembles
`D_eff,i * sum_n (z_n - z_i)/dx^2 = -U`, with `D_eff,i = D / (1 - (s_i/S_c)^2)` from the cell-centre slope s_i (centred
differences, ratio clipped at 0.99). That is D_eff lap(z) = -U. Roering et al. (1999, 2007) write the flux q = -D grad z /
(1 - (abs grad z / S_c)^2) and the steady equation div q = U; the two agree only for constant D. In 1-D:

| | slope at distance x from the crest | R*(E*) | x where slope = 0.99 S_c |
| --- | --- | --- | --- |
| divergence form (Roering) | S_c (sqrt(1+u^2) - 1)/u, u = 2 U x/(D S_c) | (1/E*)[sqrt(1+E*^2) - ln((1+sqrt(1+E*^2))/2) - 1] | about 50 D S_c/U |
| v1 form (repo) | S_c tanh(U x/(D S_c)) | (2/E*) ln cosh(E*/2) | 2.65 D S_c/U |

The v1 row is derived here (the 1-D equation integrates in closed form); the Roering row is the published curve, which addendum 8
checked by integration. On AP1 at the
fitted S_c (0.306) and D = 1e-3 the v1 form reaches the clip 15 m from the crest, the flux form 280 m; median L_H there is 85 m.

Added (N11): `solver="picard"|"newton"`, `form="centre"|"flux"` and `z_init` on the model; the flux form uses the face
diffusivity (D_eff,i + D_eff,n)/2 between hillslope cells and D_eff,i on faces to channel and boundary cells, with the same
slope stencil and clip. With a constant D it gives the v1 matrix bit for bit (tested). Newton (`models/nonlinear_newton.py`):
Jacobian J = M(D_eff) + G diag(dD_eff/ds) dS/dz, with dD_eff/ds = 0 on clipped cells; Armijo backtracking on the residual 2-norm;
converged when ||F||/||b|| <= 1e-10 (the relative residual of the addendum 3 Picard audit) and the max update <= 1e-6 m; cap 50.
Outcome: `no_steady_solution` if more than 1 % of hillslope cells are clipped at the last iterate (whether or not the clipped
system converged), else `converged` if the stopping test held, else `not_converged`. Continuation: along an A_crit row D steps
down from the largest value, the first solve from the linear solution, each later one from the most recent converged solution.

### 2.2 Tests (E1-E4)

- **Test 1 (E3): pass**, numbers in the summary. Newton needed 4-5 iterations (33-34 s) where Picard needed 6-7 (41-44 s) on the
  full tiles.
- **Test 2 (E1): the flux form passes; the v1 form converges to its own curve.** The strip is the 2-D model on a ridge 20 m
  wide (channels at both long edges), 6 ridge widths long, with the short edges set to the analytic profile; R* is read on the
  middle row. The flux error falls by 4.0, 4.1 (E* = 5) and 3.8, 3.9 (E* = 20) per halving of dx.
- **Test 3 (E2): pass.**
- **Test 4 (E4; figure `a9_pilot.png`).** Row: A_crit 6973 m^2 (AP1) and 1200 m^2 (GM1), K at the linear joint optimum, S_c the
  addendum 8 primary-network fit. All 88 evaluations on the v1 form are `no_steady_solution`. The test's condition "Newton must
  converge at every D where Picard converged" holds when read as the stopping test: at all 19 values of D where Picard met its
  tolerance, Newton met its own, at the same clip fraction (to 4 digits) and MSE (to 2 decimals); on AP1 Newton also met it
  at D = 6.4e-3, 7.8e-3 and 9.4e-3, where Picard hit the cap. The maximum elevation difference between the two was not recorded. Read as the outcome class it
  cannot be applied, because no Picard evaluation is `converged`; the `pass` flag stored in the run (`False`) uses the class and is
  therefore not informative. At lower D, Picard hits the cap of 100 and Newton stops on a failed line search (step below 2^-12)
  after 3-38 iterations or at its cap of 50; both are clipped there on 12-99 % of cells. Newton's clipped iterates there have far lower MSE than
  Picard's (for example AP1 at D = 1e-3: 40 against 27 190 m^2), but neither is a solution.
- **Clip near the threshold on GM1.** On the v1 form at D >= 3.4e-2, GM1 is clipped on 1.07-1.16 % of cells (just over 1 %); the
  flux form on 0.80 % at each of its 3 D. The fraction is the same at every D on the flux form, which suggests a fixed set of
  cells set by the fixed channel and boundary values; I did not map them.

**Flux-form pilot (E4).**

| tile | D | outcome | Newton it | clipped | MSE | linear joint MSE | wall s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 5.00e-2 | no_steady_solution (clipped system converged) | 26 | 0.121 | 196.7 | 22.38 | 802 |
| AP1 | 1.64e-2 | no_steady_solution (cap) | 50 | 0.172 | 33.0 | 22.38 | 1454 |
| AP1 | 3.68e-3 | no_steady_solution (cap) | 50 | 0.712 | 518.3 | 22.38 | 713 |
| GM1 | 5.00e-2 | converged | 22 | 0.008 | 149.2 | 24.69 | 456 |
| GM1 | 1.97e-2 | converged | 11 | 0.008 | 71.5 | 24.69 | 234 |
| GM1 | 9.35e-3 | converged | 13 | 0.008 | 38.3 | 24.69 | 274 |

These are three points of one A_crit row; they are not an optimum and do not test the section 3 rule. On GM1 the MSE falls as D
falls toward D0_A (9.0e-3); the lower D values were not evaluated.

**The GMRES option.** The flux form's Jacobian has a 13-point stencil, and its LU on a full tile costs about 30-60 s per Newton
step. GMRES on the exact Jacobian, preconditioned with the LU of its 5-point part, reached its 600-iteration limit at every Newton
step on a 300 x 300 AP1 crop (`srun` job 52196932), so it is kept as an option and not used.

### 2.3 Why nearly everything is clipped at the fitted S_c (E5)

Run `20261113T165639Z_a9_slope_exceedance_10tiles`. On the hillslope cells of the linear joint optimum, with the models' own slope
stencil on the 1 m grid:

| site | observed cells above 0.99 S_c | linear joint surface above 0.99 S_c | of those, next to a channel or boundary cell |
| --- | --- | --- | --- |
| AP | 0.31-0.52 | 0.32-0.46 | 0.01-0.04 |
| GM | 0.11-0.48 (GM3 0.48, the others 0.11-0.18) | 0.07-0.58 (GM3 0.58) | 0.01-0.06 |

At the 16th and 84th percentile S_c the AP range is 0.35-0.65 and 0.28-0.43. A steady model with this S_c must keep every
hillslope cell below 0.99 S_c to count as converged; on AP the observed surface and the best linear fit violate that on a third
to a half of the cells. Clipping on far more than 1 % of cells is then what any model has to do to approach the observed relief,
whichever solver or form is used. On GM, S_c is higher (0.54-0.63 except GM3), and the flux form converged on GM1.

### 2.4 The stop rule and the estimate

- **2.2 rule:** "If test 4 shows `no_steady_solution` for most low-D evaluations, stop after section 2." On both tiles 16 of 16
  evaluations below the linear joint D are `no_steady_solution` (44 of 44 at all D). It applies. The flux form on AP1 is the same
  (3 of 3); on GM1 it is not.
- **5.2-5.3 estimate:** 3 S_c x 30 A_crit x 22 D = 1980 flux evaluations per tile, at the pilot's mean 990 s (AP1) and 321 s
  (GM1): 2720 + 880 = **about 3600 core-hours**, against 900. The AP cost is set by evaluations that end clipped after 26-50
  iterations.
- So **section 3 was not run, and section 4 was not run** (the rule says to stop after section 2).

## 3. The nonlinear joint profile at fitted S_c (section 3), with the rule applied

Not run (section 2.4). No result. The code (`lemcvt a9 fair-part`, `fair-merge`, `fair-summary`; the rule of 3.2 in
`rule3_tile`) is written and covered only by the synthetic tests of the solver; it has not run on a tile.

## 4. Morphology scores of every surface (section 4), with the rule applied, beside the MSE

Not run (the 2.2 rule stops after section 2). The code (`lemcvt a9 morphology`, `docs/measurement_methods.md` section 11) is
written and has not run on a tile. For the three Phase 6 surfaces (linear joint, AD, nonlinear joint at S_c = 1.27) it needs no
model solve; its cost is a few core-hours.

## 5. What the evidence supports, and where it is weak

The addendum names three reasons why no nonlinear run has beaten linear diffusion.

1. **"The nonlinear model was never run fairly": supported, though not for the reason the addendum gives.**
   - Confirmed: the nonlinear model run so far is not the Roering law. It is D_eff lap(z) = -U, whose hillslopes steepen toward
     S_c about 20 times faster than the divergence form's (2.1, E1). Every earlier nonlinear number belongs to that equation.
   - Contradicted: that the addendum 8 stalls were a weakness of the Picard iteration. Newton solves the same equations; where
     Picard stalled, Newton either reached the same clipped solution or stalled on a clipped iterate. The evaluations have no
     unclipped steady solution at the 1 % threshold (test 4). A higher cap or a better solver would not change that, as the
     addendum anticipated.
   - Supported, with weak evidence: at the fitted S_c the model cannot represent the AP topography, since a third to a half of the
     observed 1 m hillslope cells exceed the clip (E5). On GM the flux form does converge (one tile, one row, 3 D values).
2. **"The MSE does not reward the right hillslope form": not tested.** Section 4 was not run.
3. **"The simple law is too crude": weak and indirect evidence.** E5 shows that one S_c cannot describe both the path-mean
   gradients that set the E*-R* fit and the 1 m cell gradients. That fits reason 3, and it equally fits an S_c measured at the
   wrong scale. The data here do not separate the two. The 1 m gradients include roughness, and on AP the TIN facets (02c),
   which a smooth steady model is not expected to match.

**Where the evidence is weak.**
- Test 4 and the flux pilot are one A_crit row per tile on two tiles. The flux form was tested at 3 D values, not 22.
- The 1 % clip threshold is pre-registered but sharp. GM1 sits at 0.8 % (flux) and 1.07-1.16 % (v1 form at D >= 3.4e-2).
- The E5 exceedance uses the 1 m stencil. A gradient on a smoothed surface would exceed S_c on fewer cells; I did not compute it.
- The flux discretisation (arithmetic-mean face D_eff from cell-centre slopes) is one of several reasonable choices. It is
  second-order on the 1-D strip; its 2-D behaviour was not tested against an independent solution.
- The Roering et al. (1999, 2007) and Hurst et al. (2012) references are from memory, as in addendum 8, and should be checked.

## 6. Decisions needed

1. **Model form.** Accept that the nonlinear results of v1, Phase 6 and addendum 8 are for D_eff lap(z) = -U, and decide how to
   label them. Separately, decide whether any later nonlinear statement should use the flux form. I recommend the flux form,
   because it is the law the E*-R* curve and the literature use.
2. **S_c for any further nonlinear test.** The path-scale fit leaves AP untestable at the 1 % rule. Options:
   - (a) GM only, flux form, central S_c: 5 tiles x 660 evaluations x about 320 s, about 300 core-hours (two submissions);
   - (b) a pre-registered S_c set from the cell-scale gradient distribution (for example a high percentile of observed hillslope
     gradients), stated before any run;
   - (c) stop the nonlinear line here.
   I recommend (a) only if the GM-only answer would carry weight. Otherwise (c).
3. **Section 4 on the existing surfaces.** Run the morphology score for the linear joint, AD and S_c = 1.27 nonlinear surfaces
   (a few core-hours, no model solve)? It answers reason 2 independently of the nonlinear model.
4. **The 1 % clip rule.** Keep it, or restate it before any rerun (for example, excluding cells next to fixed cells). GM1 sits on
   the threshold.
5. **Push.** The code commits and this report are pushed with this session, after the addendum 2 section 5 check. CI stays manual.

## 7. Appendices (generated tables)

Generated by `reports/v2/scripts/make_addendum9_tables.py` from the run directories named in each table.

## E1. Section 2.2 test 2: the 1-D ridge strip, both forms

Run `20261113T170607Z_a9_section2_tests_strip_jacobian`. Ridge strip with channels (z = 0) at distance L = 10 m from the crest, solved by the 2-D model with Newton; U = 1e-4 m/yr, S_c = 0.5, D = 2 U L/(E* S_c). Roering = `integrate_profile_R` (the integrated steady curve of the divergence form); centre-form curve = (2/E*) ln cosh(E*/2), the exact 1-D curve of the v1 equation.

| form | E* | dx (m) | cells | R* | R* Roering | rel. error vs Roering | R* centre-form curve | rel. error vs centre-form curve | outcome | clipped | Newton it |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flux | 1 | 1 | 2541 | 0.226510 | 0.225987 | +2.31e-03 | 0.240229 | -5.71e-02 | converged | 0.000 | 4 |
| flux | 1 | 0.5 | 9881 | 0.226118 | 0.225987 | +5.81e-04 | 0.240229 | -5.87e-02 | converged | 0.000 | 4 |
| flux | 1 | 0.25 | 38961 | 0.226020 | 0.225987 | +1.45e-04 | 0.240229 | -5.91e-02 | converged | 0.000 | 4 |
| flux | 5 | 1 | 2541 | 0.599468 | 0.596808 | +4.46e-03 | 0.725427 | -1.74e-01 | converged | 0.000 | 13 |
| flux | 5 | 0.5 | 9881 | 0.597461 | 0.596808 | +1.10e-03 | 0.725427 | -1.76e-01 | converged | 0.000 | 11 |
| flux | 5 | 0.25 | 38961 | 0.596970 | 0.596808 | +2.71e-04 | 0.725427 | -1.77e-01 | converged | 0.000 | 14 |
| flux | 20 | 1 | 2541 | 0.840605 | 0.833621 | +8.38e-03 | 0.930685 | -9.68e-02 | converged | 0.000 | 13 |
| flux | 20 | 0.5 | 9881 | 0.835454 | 0.833621 | +2.20e-03 | 0.930685 | -1.02e-01 | converged | 0.000 | 13 |
| flux | 20 | 0.25 | 38961 | 0.834089 | 0.833621 | +5.62e-04 | 0.930685 | -1.04e-01 | converged | 0.000 | 14 |
| centre | 1 | 1 | 2541 | 0.240325 | 0.225987 | +6.34e-02 | 0.240229 | +3.99e-04 | converged | 0.000 | 3 |
| centre | 1 | 0.5 | 9881 | 0.240253 | 0.225987 | +6.31e-02 | 0.240229 | +9.98e-05 | converged | 0.000 | 3 |
| centre | 1 | 0.25 | 38961 | 0.240235 | 0.225987 | +6.30e-02 | 0.240229 | +2.50e-05 | converged | 0.000 | 3 |
| centre | 5 | 1 | 2541 | 0.728863 | 0.596808 | +2.21e-01 | 0.725427 | +4.74e-03 | converged | 0.000 | 9 |
| centre | 5 | 0.5 | 9881 | 0.726282 | 0.596808 | +2.17e-01 | 0.725427 | +1.18e-03 | converged | 0.000 | 8 |
| centre | 5 | 0.25 | 38961 | 0.725641 | 0.596808 | +2.16e-01 | 0.725427 | +2.94e-04 | converged | 0.000 | 8 |
| centre | 20 | 1 | 2541 | 1.001999 | 0.833621 | +2.02e-01 | 0.930685 | +7.66e-02 | no_steady_solution | 0.737 | 14 |
| centre | 20 | 0.5 | 9881 | 0.982937 | 0.833621 | +1.79e-01 | 0.930685 | +5.61e-02 | no_steady_solution | 0.718 | 14 |
| centre | 20 | 0.25 | 38961 | 0.979094 | 0.833621 | +1.75e-01 | 0.930685 | +5.20e-02 | no_steady_solution | 0.731 | 13 |

## E2. Section 2.2 test 3: analytic Jacobian against central finite differences

Run `20261113T170607Z_a9_section2_tests_strip_jacobian`. Synthetic 40 x 40 tile (`tests/synthetic_tile.synthetic_dem`, seed 3), A_crit 200 m^2, D 5e-3, S_c 3 (nonlinearity engaged, nothing clipped), z perturbed by N(0, 0.05 m). The error grows as 1/h below h = 1e-4, i.e. it is finite-difference roundoff.

| form | h (m) | unknowns | max slope / S_c | max abs error / max abs J | max entrywise rel. error (entries > 1e-3 max) |
|---|---|---|---|---|---|
| centre | 0.0001 | 1404 | 0.840 | 2.9e-09 | 1.0e-08 |
| centre | 1e-05 | 1404 | 0.840 | 2.6e-09 | 1.9e-08 |
| centre | 1e-06 | 1404 | 0.840 | 1.9e-08 | 1.8e-07 |
| flux | 0.0001 | 1404 | 0.695 | 7.1e-10 | 2.9e-07 |
| flux | 1e-05 | 1404 | 0.695 | 5.7e-09 | 2.9e-06 |
| flux | 1e-06 | 1404 | 0.695 | 4.5e-08 | 3.4e-05 |

## E3. Section 2.2 test 1: Newton against Picard at the Phase 6 setting (S_c = 1.27, v1 form)

Pilot summary `20261113T202656Z_a9_pilot_summary_AP1_GM1`. Picard at the configured cap (100) and tolerance (max update 1e-6 m); 'tight' = Picard to 1e-10 m with cap 2000; Newton to relative residual 1e-10 and max update 1e-6 m.

| case | A_crit | D | Picard: outcome, it, wall s | tight Picard: outcome, it | Newton: outcome, it, wall s, rel. residual | max abs dz Newton-Picard (m) | Newton-tight (m) | pass (1e-6 m) |
|---|---|---|---|---|---|---|---|---|
| AP1 full tile (20261112T175112Z_tile_run_AP1_v2_nonlinear_joint_merged) | 6973.2 | 1.652e-02 | converged, 6, 44 | converged, 9 | converged, 4, 34, 6.6e-16 | 2.2e-09 | 5.2e-11 | True |
| AP1 crop (300, 600, 300, 600) | 6973.2 | 1.652e-02 | converged, 5, 2 | converged, 7 | converged, 4, 2, 5.2e-16 | 2.0e-09 | 1.8e-11 | True |
| GM1 full tile (20261112T175109Z_tile_run_GM1_v2_nonlinear_joint_merged) | 1758.0 | 1.987e-02 | converged, 7, 41 | converged, 11 | converged, 5, 33, 6.7e-16 | 4.8e-08 | 3.2e-11 | True |
| GM1 crop (300, 600, 300, 600) | 1758.0 | 1.987e-02 | converged, 6, 2 | converged, 10 | converged, 4, 2, 5.0e-16 | 4.1e-08 | 1.4e-11 | True |

## E4. Section 2.2 test 4 and the pilot rows (5.1)

Pilot summary `20261113T202656Z_a9_pilot_summary_AP1_GM1`. One A_crit row per tile (the addendum 6 grid value nearest the linear joint A_crit), K at the linear joint optimum, S_c = the addendum 8 primary-network fit; Newton with continuation from the largest D. clipped = fraction of hillslope cells with |grad z|/S_c > 0.99 at the last iterate.

| tile | solver/form | D | outcome | iterations | stop | clipped | > 0.5 S_c | MSE | wall s | run |
|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | newton/centre | 1.000e-03 | no_steady_solution | 6 | line_search_failed | 0.3047 | 0.890 | 39.73 | 59 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 1.205e-03 | no_steady_solution | 14 | line_search_failed | 0.3705 | 0.925 | 29.44 | 128 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 1.451e-03 | no_steady_solution | 7 | line_search_failed | 0.3149 | 0.892 | 44.29 | 68 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 1.749e-03 | no_steady_solution | 4 | line_search_failed | 0.3392 | 0.868 | 56.90 | 42 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 2.107e-03 | no_steady_solution | 9 | line_search_failed | 0.4084 | 0.933 | 31.09 | 84 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 2.538e-03 | no_steady_solution | 23 | line_search_failed | 0.9110 | 0.987 | 1054.71 | 198 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 3.058e-03 | no_steady_solution | 6 | line_search_failed | 0.3695 | 0.910 | 39.73 | 58 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 3.684e-03 | no_steady_solution | 21 | line_search_failed | 0.4283 | 0.948 | 24.09 | 182 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 4.438e-03 | no_steady_solution | 19 | line_search_failed | 0.8000 | 0.971 | 277.09 | 169 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 5.347e-03 | no_steady_solution | 11 | line_search_failed | 0.7252 | 0.962 | 152.95 | 99 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 6.442e-03 | no_steady_solution | 28 | tolerance | 0.3014 | 0.917 | 20.30 | 214 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 7.761e-03 | no_steady_solution | 17 | tolerance | 0.2652 | 0.900 | 22.32 | 127 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 9.351e-03 | no_steady_solution | 13 | tolerance | 0.2323 | 0.879 | 26.43 | 98 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 1.127e-02 | no_steady_solution | 11 | tolerance | 0.2030 | 0.855 | 33.14 | 83 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 1.357e-02 | no_steady_solution | 9 | tolerance | 0.1799 | 0.826 | 42.90 | 68 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 1.635e-02 | no_steady_solution | 9 | tolerance | 0.1619 | 0.793 | 55.99 | 68 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 1.970e-02 | no_steady_solution | 9 | tolerance | 0.1492 | 0.752 | 72.33 | 69 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 2.373e-02 | no_steady_solution | 10 | tolerance | 0.1409 | 0.705 | 91.39 | 76 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 2.859e-02 | no_steady_solution | 9 | tolerance | 0.1354 | 0.653 | 112.31 | 69 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 3.445e-02 | no_steady_solution | 9 | tolerance | 0.1320 | 0.610 | 134.03 | 68 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 4.150e-02 | no_steady_solution | 8 | tolerance | 0.1300 | 0.576 | 155.55 | 61 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/centre | 5.000e-02 | no_steady_solution | 7 | tolerance | 0.1290 | 0.547 | 176.06 | 54 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 1.000e-03 | no_steady_solution | 100 | max_iter | 0.9214 | 0.987 | 27190.07 | 664 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 1.205e-03 | no_steady_solution | 100 | max_iter | 0.6885 | 0.942 | 9643.63 | 668 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 1.451e-03 | no_steady_solution | 100 | max_iter | 0.8200 | 0.947 | 4927.88 | 673 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 1.749e-03 | no_steady_solution | 100 | max_iter | 0.4246 | 0.841 | 378.93 | 670 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 2.107e-03 | no_steady_solution | 100 | max_iter | 0.5894 | 0.921 | 549.78 | 672 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 2.538e-03 | no_steady_solution | 100 | max_iter | 0.6639 | 0.949 | 297.02 | 666 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 3.058e-03 | no_steady_solution | 100 | max_iter | 0.3917 | 0.882 | 49.14 | 670 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 3.684e-03 | no_steady_solution | 100 | max_iter | 0.5504 | 0.921 | 67.69 | 672 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 4.438e-03 | no_steady_solution | 100 | max_iter | 0.4850 | 0.913 | 48.78 | 663 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 5.347e-03 | no_steady_solution | 100 | max_iter | 0.4269 | 0.911 | 29.58 | 668 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 6.442e-03 | no_steady_solution | 100 | max_iter | 0.3684 | 0.910 | 19.01 | 666 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 7.761e-03 | no_steady_solution | 100 | max_iter | 0.2859 | 0.895 | 23.47 | 665 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 9.351e-03 | no_steady_solution | 100 | max_iter | 0.2329 | 0.880 | 27.11 | 668 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 1.127e-02 | no_steady_solution | 86 | tolerance | 0.2030 | 0.855 | 33.14 | 576 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 1.357e-02 | no_steady_solution | 42 | tolerance | 0.1799 | 0.826 | 42.90 | 283 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 1.635e-02 | no_steady_solution | 28 | tolerance | 0.1619 | 0.793 | 55.99 | 191 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 1.970e-02 | no_steady_solution | 21 | tolerance | 0.1492 | 0.752 | 72.33 | 144 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 2.373e-02 | no_steady_solution | 17 | tolerance | 0.1409 | 0.705 | 91.39 | 119 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 2.859e-02 | no_steady_solution | 14 | tolerance | 0.1354 | 0.653 | 112.31 | 99 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 3.445e-02 | no_steady_solution | 12 | tolerance | 0.1320 | 0.610 | 134.03 | 85 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 4.150e-02 | no_steady_solution | 11 | tolerance | 0.1300 | 0.576 | 155.55 | 79 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | picard/centre | 5.000e-02 | no_steady_solution | 9 | tolerance | 0.1290 | 0.547 | 176.06 | 66 | `20261113T165217Z_a9_pilot_AP1_centre` |
| AP1 | newton/flux | 3.684e-03 | no_steady_solution | 50 | max_iter | 0.7117 | 0.981 | 518.25 | 713 | `20261113T165223Z_a9_pilot_AP1_flux` |
| AP1 | newton/flux | 1.635e-02 | no_steady_solution | 50 | max_iter | 0.1717 | 0.812 | 33.02 | 1454 | `20261113T165223Z_a9_pilot_AP1_flux` |
| AP1 | newton/flux | 5.000e-02 | no_steady_solution | 26 | tolerance | 0.1206 | 0.560 | 196.68 | 802 | `20261113T165223Z_a9_pilot_AP1_flux` |
| GM1 | newton/centre | 1.000e-03 | no_steady_solution | 12 | line_search_failed | 0.9851 | 0.998 | 34676.64 | 91 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 1.205e-03 | no_steady_solution | 3 | line_search_failed | 0.9686 | 0.996 | 13780.30 | 27 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 1.451e-03 | no_steady_solution | 7 | line_search_failed | 0.2285 | 0.802 | 55.43 | 55 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 1.749e-03 | no_steady_solution | 38 | line_search_failed | 0.9377 | 0.992 | 2647.83 | 272 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 2.107e-03 | no_steady_solution | 3 | line_search_failed | 0.2839 | 0.759 | 73.80 | 25 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 2.538e-03 | no_steady_solution | 9 | line_search_failed | 0.3340 | 0.811 | 77.00 | 73 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 3.058e-03 | no_steady_solution | 18 | line_search_failed | 0.4290 | 0.939 | 76.28 | 132 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 3.684e-03 | no_steady_solution | 19 | line_search_failed | 0.4056 | 0.938 | 69.51 | 140 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 4.438e-03 | no_steady_solution | 50 | line_search_failed | 0.3745 | 0.934 | 64.62 | 358 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 5.347e-03 | no_steady_solution | 20 | line_search_failed | 0.2878 | 0.909 | 52.22 | 138 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 6.442e-03 | no_steady_solution | 50 | max_iter | 0.1698 | 0.856 | 38.33 | 315 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 7.761e-03 | no_steady_solution | 3 | line_search_failed | 0.2771 | 0.868 | 49.84 | 28 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 9.351e-03 | no_steady_solution | 15 | tolerance | 0.0858 | 0.787 | 27.60 | 89 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 1.127e-02 | no_steady_solution | 9 | tolerance | 0.0585 | 0.742 | 25.56 | 53 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 1.357e-02 | no_steady_solution | 8 | tolerance | 0.0390 | 0.689 | 27.03 | 48 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 1.635e-02 | no_steady_solution | 9 | tolerance | 0.0263 | 0.627 | 32.80 | 53 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 1.970e-02 | no_steady_solution | 8 | tolerance | 0.0192 | 0.557 | 43.12 | 48 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 2.373e-02 | no_steady_solution | 7 | tolerance | 0.0152 | 0.482 | 57.58 | 43 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 2.859e-02 | no_steady_solution | 7 | tolerance | 0.0129 | 0.403 | 75.22 | 42 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 3.445e-02 | no_steady_solution | 7 | tolerance | 0.0116 | 0.323 | 94.78 | 43 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 4.150e-02 | no_steady_solution | 6 | tolerance | 0.0110 | 0.263 | 115.00 | 37 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/centre | 5.000e-02 | no_steady_solution | 6 | tolerance | 0.0107 | 0.222 | 134.83 | 37 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 1.000e-03 | no_steady_solution | 100 | max_iter | 0.9698 | 0.994 | 40752.30 | 509 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 1.205e-03 | no_steady_solution | 100 | max_iter | 0.9508 | 0.989 | 23125.96 | 513 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 1.451e-03 | no_steady_solution | 100 | max_iter | 0.7471 | 0.934 | 8137.62 | 513 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 1.749e-03 | no_steady_solution | 100 | max_iter | 0.8707 | 0.973 | 8067.44 | 512 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 2.107e-03 | no_steady_solution | 100 | max_iter | 0.4967 | 0.835 | 1215.65 | 513 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 2.538e-03 | no_steady_solution | 100 | max_iter | 0.5780 | 0.913 | 1058.02 | 512 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 3.058e-03 | no_steady_solution | 100 | max_iter | 0.4971 | 0.868 | 296.87 | 513 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 3.684e-03 | no_steady_solution | 100 | max_iter | 0.3657 | 0.887 | 114.36 | 514 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 4.438e-03 | no_steady_solution | 100 | max_iter | 0.4111 | 0.887 | 124.62 | 513 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 5.347e-03 | no_steady_solution | 100 | max_iter | 0.3064 | 0.874 | 67.60 | 513 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 6.442e-03 | no_steady_solution | 100 | max_iter | 0.2089 | 0.854 | 42.47 | 513 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 7.761e-03 | no_steady_solution | 100 | max_iter | 0.1201 | 0.825 | 32.19 | 514 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 9.351e-03 | no_steady_solution | 91 | tolerance | 0.0858 | 0.787 | 27.60 | 466 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 1.127e-02 | no_steady_solution | 42 | tolerance | 0.0585 | 0.742 | 25.56 | 219 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 1.357e-02 | no_steady_solution | 28 | tolerance | 0.0390 | 0.689 | 27.03 | 147 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 1.635e-02 | no_steady_solution | 21 | tolerance | 0.0263 | 0.627 | 32.80 | 112 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 1.970e-02 | no_steady_solution | 17 | tolerance | 0.0192 | 0.557 | 43.12 | 92 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 2.373e-02 | no_steady_solution | 14 | tolerance | 0.0152 | 0.482 | 57.58 | 76 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 2.859e-02 | no_steady_solution | 12 | tolerance | 0.0129 | 0.403 | 75.22 | 66 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 3.445e-02 | no_steady_solution | 10 | tolerance | 0.0116 | 0.323 | 94.78 | 56 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 4.150e-02 | no_steady_solution | 10 | tolerance | 0.0110 | 0.263 | 115.00 | 56 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | picard/centre | 5.000e-02 | no_steady_solution | 9 | tolerance | 0.0107 | 0.222 | 134.83 | 51 | `20261113T165223Z_a9_pilot_GM1_centre` |
| GM1 | newton/flux | 9.351e-03 | converged | 13 | tolerance | 0.0080 | 0.746 | 38.28 | 274 | `20261113T165223Z_a9_pilot_GM1_flux` |
| GM1 | newton/flux | 1.970e-02 | converged | 11 | tolerance | 0.0080 | 0.496 | 71.49 | 234 | `20261113T165223Z_a9_pilot_GM1_flux` |
| GM1 | newton/flux | 5.000e-02 | converged | 22 | tolerance | 0.0080 | 0.204 | 149.24 | 456 | `20261113T165223Z_a9_pilot_GM1_flux` |

Stop rule (2.2) and estimate (5.2):

```json
{
 "stop_rule_2_2": {
  "AP1": {
   "n_low_D": 16,
   "n_no_steady": 16,
   "frac_no_steady": 1.0,
   "most_no_steady": true
  },
  "GM1": {
   "n_low_D": 16,
   "n_no_steady": 16,
   "frac_no_steady": 1.0,
   "most_no_steady": true
  },
  "applies": true
 },
 "estimate_section3": {
  "evaluations_per_tile": 1980,
  "per_site": {
   "AP": {
    "wall_per_evaluation_s": 989.5489138023307,
    "core_hours_per_tile": 544.2519025912819,
    "core_hours_site": 2721.2595129564093
   },
   "GM": {
    "wall_per_evaluation_s": 321.374056478031,
    "core_hours_per_tile": 176.75573106291705,
    "core_hours_site": 883.7786553145852
   }
  },
  "total_core_hours": 3605.0381682709944,
  "limit_total": 900,
  "limit_per_submission": 300,
  "within_limits": false
 }
}
```

## E5. Hillslope cells above the clip at the fitted S_c (diagnostic)

Run `20261113T165639Z_a9_slope_exceedance_10tiles`. Hillslope cells = A below the linear joint A_crit, tile edge excluded; gradient from the models' cell-centre slope stencil on the 1 m grid. S_c: the addendum 8 primary-network fit and its 68 % interval.

| tile | surface | cells | median slope | p90 slope | S_c | > 0.99 S_c | > 0.5 S_c | > 0.99 S_c at S_c p16 / p84 | share of clipped cells next to a channel or boundary cell |
|---|---|---|---|---|---|---|---|---|---|
| AP1 | observed | 1021294 | 0.269 | 0.461 | 0.306 | 0.394 | 0.844 | 0.418 / 0.368 | 0.012 |
| AP1 | linear_joint | 1021294 | 0.252 | 0.413 | 0.306 | 0.317 | 0.811 | 0.340 / 0.292 | 0.020 |
| AP2 | observed | 1015187 | 0.267 | 0.443 | 0.293 | 0.410 | 0.862 | 0.466 / 0.362 | 0.022 |
| AP2 | linear_joint | 1015187 | 0.270 | 0.407 | 0.293 | 0.397 | 0.894 | 0.470 / 0.339 | 0.027 |
| AP3 | observed | 1017927 | 0.271 | 0.450 | 0.267 | 0.520 | 0.846 | 0.648 / 0.408 | 0.011 |
| AP3 | linear_joint | 1017927 | 0.255 | 0.429 | 0.267 | 0.463 | 0.898 | 0.633 / 0.346 | 0.018 |
| AP4 | observed | 1021835 | 0.272 | 0.455 | 0.271 | 0.512 | 0.894 | 0.583 / 0.431 | 0.010 |
| AP4 | linear_joint | 1021835 | 0.248 | 0.411 | 0.271 | 0.448 | 0.790 | 0.498 / 0.381 | 0.014 |
| AP5 | observed | 1013169 | 0.271 | 0.464 | 0.341 | 0.310 | 0.796 | 0.346 / 0.279 | 0.021 |
| AP5 | linear_joint | 1013169 | 0.280 | 0.450 | 0.341 | 0.315 | 0.813 | 0.357 / 0.279 | 0.036 |
| GM1 | observed | 988163 | 0.377 | 0.600 | 0.544 | 0.181 | 0.705 | 0.207 / 0.159 | 0.010 |
| GM1 | linear_joint | 988163 | 0.322 | 0.558 | 0.544 | 0.118 | 0.614 | 0.135 / 0.104 | 0.054 |
| GM2 | observed | 989553 | 0.366 | 0.561 | 0.540 | 0.142 | 0.685 | 0.187 / 0.103 | 0.012 |
| GM2 | linear_joint | 989553 | 0.297 | 0.501 | 0.540 | 0.071 | 0.578 | 0.091 / 0.055 | 0.047 |
| GM3 | observed | 982280 | 0.250 | 0.512 | 0.264 | 0.479 | 0.740 | 0.515 / 0.426 | 0.019 |
| GM3 | linear_joint | 982280 | 0.286 | 0.468 | 0.264 | 0.584 | 0.880 | 0.644 / 0.489 | 0.039 |
| GM4 | observed | 989880 | 0.449 | 0.634 | 0.634 | 0.110 | 0.733 | 0.135 / 0.086 | 0.009 |
| GM4 | linear_joint | 989880 | 0.384 | 0.650 | 0.634 | 0.119 | 0.647 | 0.135 / 0.104 | 0.040 |
| GM5 | observed | 985381 | 0.376 | 0.595 | 0.571 | 0.137 | 0.683 | 0.198 / 0.112 | 0.015 |
| GM5 | linear_joint | 985381 | 0.338 | 0.579 | 0.571 | 0.112 | 0.638 | 0.153 / 0.096 | 0.058 |

## E6. Jobs (sacct)

| job | what | state | elapsed | CPUs | core-h used | time limit |
|---|---|---|---|---|---|---|
| 52186353_0 | design probe, AP1 (scancel issued during the flux solve) | COMPLETED | 00:37:02 | 2 | 1.23 | 01:00:00 |
| 52196422_0 | design probe, AP1, flux first (cancelled) | CANCELLED by 313293 | 00:24:42 | 2 | 0.82 | 01:00:00 |
| 52196932 | srun diagnostic: 300 x 300 crop, GMRES vs direct | COMPLETED | 00:21:27 | 1 | 0.36 | 00:25:00 |
| 52198633 | srun diagnostic: crop continuation row (stopped by me) | FAILED | 00:06:55 | 1 | 0.12 | 00:50:00 |
| 52198921 | srun diagnostic: slope exceedance draft | COMPLETED | 00:00:43 | 1 | 0.01 | 00:25:00 |
| 52199605_0 | ruff, mypy, fast suite, slope exceedance | COMPLETED | 00:05:00 | 1 | 0.08 | 01:30:00 |
| 52199606_0 | pilot, centre part (AP1, GM1) | COMPLETED | 03:30:48 | 1 | 3.51 | 08:00:00 |
| 52199606_1 | pilot, centre part (AP1, GM1) | COMPLETED | 02:43:34 | 1 | 2.73 | 08:00:00 |
| 52199672_0 | pilot, flux part (AP1, GM1) | COMPLETED | 00:49:56 | 2 | 1.66 | 1-00:00:00 |
| 52199672_1 | pilot, flux part (AP1, GM1) | COMPLETED | 00:16:33 | 2 | 0.55 | 1-00:00:00 |
| 52199689_0 | pilot summary and figure | COMPLETED | 00:00:42 | 1 | 0.01 | 00:30:00 |
| 52202379_0 | recorded section 2.2 tests 2-3 | COMPLETED | 00:00:56 | 1 | 0.02 | 01:00:00 |
| 52230236_0 | final ruff, mypy, fast suite | COMPLETED | 00:04:34 | 1 | 0.08 | 01:00:00 |

Total used: 11.2 core-hours.

