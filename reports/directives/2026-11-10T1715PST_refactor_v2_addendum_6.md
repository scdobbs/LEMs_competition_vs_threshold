# [2026-11-10 17:15 PST] Addendum 6 to the v2 refactor directive: after the option (iii) failure, fix CI, then run Phase 6

This addendum overrides the directive and Addenda 1–5 wherever they conflict. The reporting conventions of Addendum 4 §1 are permanent. Addendum 2 §4 (Slurm and storage) and §5 (push) still apply.

## 0. Orientation (you are a new agent instance)

1. **Read, in order:**
   - the directive and Addenda 1–5 in `reports/directives/`;
   - this file;
   - `reports/v2/20261110T1648PST_phase6.md` (latest state), then `20261110T1540PST_checkpoint_4b_5.md`;
   - `docs/numerics_changes.md` and `docs/measurement_methods.md`.
2. **Confirm the state:** the `v2` head equals `origin/v2`; tests pass on `serc`; no earlier job is running.
3. **Where this addendum and the repo disagree,** the repo wins. Report it and stop.

## 1. Decisions

Steve will strike any he does not accept before this reaches you.

1. **The option (iii) test failed on AP, and that result stands.** The pre-registered rule is not revised after the fact. On AP, D₀ is therefore treated as a bracket, not a measurement:
   - **D₀,A = U/C_ht(L_f)**, option (iii) at L_f = 11 m (as computed in run `20261111T002913Z_phase6_inputs_10tiles`);
   - **D₀,B = U/C_ht(L_w)**, on the same option (iii) pixel set, where L_w is the largest tested window not exceeding that tile's median crest width (AP1 31 m, AP2 15 m, AP3 11 m, AP4 41 m, AP5 15 m; compute it from A2 of the Phase 6 report and state it).
   - On GM option (iii) stands: D₀,A is the measured value. Report D₀,B for GM too, for symmetry.
   - Configs carry D₀,A as the primary input (one rule at both sites) and D₀,B beside it. Any AP statement about D recovery in a report must cite both ends.
2. **AP1 keeps its stored grids,** with the three-row inconsistency recorded (N9). §3 tests whether it matters.
3. **Write the measured inputs into the configs now:** m (AP 0.47, GM 0.31), K₀ = U/k_s at the site m above 10³ m², D₀,A and D₀,B, and AP3 at U/k_s. Use `lemcvt phase6 write-configs` from the recorded run, extended for D₀,B. Keep v1 values under `v1_reference`.
4. **Narrow the nonlinear joint D grid** to 20 log-spaced values from 1.5×10⁻³ to 5×10⁻² m²/yr. That brackets every D₀ (AP 2.2×10⁻³ at the low end) and every linear D_fit (≤ 2.7×10⁻²) with margin, and drops the low-D region where the iteration saturates. Keep the 30-value A_crit grid.
5. **Budget:** Steve approved more than 900 core-hours, provided every Sherlock and serc rule is followed. No single submission over 300 core-hours; concurrent CPUs well under 300; every array throttled. Print the estimate first, then submit.

## 2. Stop the failing CI runs (do this first)

Steve is getting a failure email for every push to `v2`: `ci / test (3.11)` fails in about 36 s and cancels `test (3.12)`. Failure that early points to the install, ruff or mypy step, not the tests.

1. **Reproduce it on Sherlock.** In fresh environments for Python 3.11 and 3.12, run exactly the workflow's steps: `pip install -e ".[dev]"`, `ruff check src tests`, `mypy src/lemcvt`, `pytest -q -m "not fulltile"`. Report which step fails and why.
2. **Fix it properly.**
   - If the package cannot support 3.11 (for example, a dependency in `envs/sherlock.yml` or `pyproject.toml` needs 3.12), set `requires-python = ">=3.12"` and drop 3.11 from the matrix. Do not pin dependencies down to rescue 3.11.
   - Fix ruff and mypy findings in the code, not by widening ignore lists, unless a finding is a false positive; list any ignore you add.
   - Make sure the fast suite needs no data from `$LEMCVT_DATA` (mark any such test `fulltile`).
3. **Set `fail-fast: false`** in the matrix so one version's failure no longer cancels the other.
4. **If CI cannot be made green in this session,** change the trigger to `workflow_dispatch` only, say so in the report, and leave the fix as an open item. Either way, no push may trigger a failing run after this section.
5. **Gate:** the workflow passes on the pushed commit. Report the run's status (a `gh` call or the Actions page is fine).

## 3. AP1: does the seam matter? (pre-registered)

1. **Define the affected set S** on AP1: rows 385–395; every cell downstream (D8, stored directions) of the 345 seam cells; and the 699 cells whose stored area is inconsistent with their donors.
2. **Rerun on AP1, in metres at the site m,** the linear joint profile and the AD profile with the misfit computed over all valid cells *except* S. The reduction takes a cell mask; use it. Run the stage-2 profile at D₀,A the same way.
3. **Report** |S|, the optimum A_crit, K, D and MSE with and without S excluded, and the profile shapes.
4. **Rule, written now.** The seam is negligible if, for the joint profile, the masked A_crit lies inside the unmasked 1% interval, K moves by less than 2%, D by less than 5%, and the AD r by less than 0.1 decade. Then AP1 is used normally, with a note. Otherwise AP1 results are reported separately, and every AP site statistic (S(m), site-pooled values) is given with and without AP1.

## 4. A curvature comparison that does not depend on the window choice

The D question should not hinge on picking one window. So apply the *same* estimator to the model surfaces.

For each tile, on the option (iii) pixel set, compute median C_ht(L) at L = 7, 9, 11, 15, 21, 25, 31, 41 and 51 m from:

- the observed DEM (already in A2 of the Phase 6 report);
- the fitted surface of the linear joint optimum at the site m;
- the fitted surface of the nonlinear joint optimum (§5.4);
- the fitted surface of the AD optimum.

Report the ratio C_obs(L)/C_model(L) at every window, plus the fraction of those pixels that are channel cells in each fitted model. No pass rule; no interpretation beyond the numbers. Plot one panel per tile with the four curves.

## 5. Phase 6: the rerun

Run Addendum 5 §3 with these changes. All tiles, metres, v2 optimizer, site m.

1. **AD profile** (5 §3.1): unchanged.
2. **Linear partitioned** (5 §3.2): stage 2 at D₀,A on all tiles, and also at D₀,B on AP. Joint profile unchanged, including the 5% interval fill.
3. **Nonlinear stage** (5 §3.3): at D₀,A on all tiles, and also at D₀,B on AP. Picard cap 100.
4. **Nonlinear joint** (5 §3.4): with the D grid of §1.4. K fixed at the linear joint optimum; the K check at ×0.95 and ×1.05 on GM1 and AP5.
5. **D∞ joint** (5 §3.5): unchanged.
6. **Figures** (5 §3.6): the resolution sweep at D₀,A on all tiles, and at D₀,B on AP as well (cheap); `fig_slope_area` at the new optima; `supp_fig5` regenerated.
7. **D-recovery diagnostics** (5 §3.7): report D_fit (linear joint and nonlinear joint) against D₀,A and D₀,B, per tile, beside §4.
8. **Old-against-new record** (5 §3.8): add "AP D₀ bracket" and "AP1 seam (per §3 outcome)" as causes where they apply.

Order: §2, then §1.3 (configs), then §3, then the §5 arrays (§3 can run alongside them; AP1's main results simply wait for its verdict), then §4 once the §5 optima exist.

## 6. Report and stop

One report, `reports/v2/<YYYYMMDDTHHMM>PST_phase6_results.md`, titled with its timestamp. Fixed outline:

1. State check, and the CI fix (§2).
2. Configs written (§1.3), with the D₀ bracket table.
3. AP1 seam test (§3).
4. Budget and runs.
5. Results by model: AD, linear partitioned, nonlinear, nonlinear joint, D∞.
6. Curvature comparison (§4) and D-recovery diagnostics.
7. Figures.
8. Old-against-new record.
9. What the evidence supports, and where it is weak.
10. Decisions needed.
11. Appendices: all generated tables.

**Stop for review.** Phase 7 waits.

Do not fix numerical issues outside this addendum silently. Report them.
