# [2026-11-13 13:30 PST] Addendum 10 to the v2 refactor directive: a bounded GM flux-form test, then wrap-up

This addendum overrides the directive and Addenda 1–9 wherever they conflict. Addendum 4 §1 (timestamped titles, one report per checkpoint), Addendum 2 §4–§5 (Slurm, storage, push) and Addendum 7 §0.4 (submit guard) still apply.

## 0. Orientation (you are a new agent instance)

1. **Read, in order:**
   - the directive and Addenda 1–9 in `reports/directives/`;
   - this file;
   - `reports/v2/20261113T1330PST_nonlinear_fair_test.md` (latest state), then `20261113T0450PST_ER_threshold_phase7.md`, `20261112T1600PST_d_diagnostic.md` and `20261112T1003PST_phase6_results.md`;
   - `docs/measurement_methods.md` and `docs/numerics_changes.md` (N10, N11).
2. **Confirm** that `v2` equals `origin/v2`, that the tests pass on `serc`, and that no earlier job is running.
3. **Where this addendum and the repo disagree,** the repo wins. Report it and stop.

## 1. Decisions

1. **Model form.**
   - Every nonlinear result to date (v1, Phase 6, Addenda 8–9) is labelled **"centre form, D_eff∇²z = −U (nonstandard)"**, in `docs/numerics_changes.md`, in the old-against-new record, and in any table that shows it.
   - All new nonlinear work uses the flux form, ∇·(D_eff∇z) = U, with the Newton solver.
   - Picard on the centre form stays available, but is no longer the default for any new command.
2. **AP nonlinear work stops.** At any S_c the E*–R* data support, a third to a half of the AP 1 m hillslope cells exceed the clip. A fair AP test needs better source data, which stays deferred.
3. **Test 2 passes its review gate.** Steve judged the AP heads at DT = 2 m² (Addendum 8 run `20261113T022845Z_heads_10tiles`) plausible channel heads. Use that head set on AP and the existing GM sets.
4. **The 1 % clip rule of Addendum 9 is replaced for §2:** a solution counts as `converged` if fewer than 1 % of hillslope cells are clipped, *excluding cells adjacent to a channel or boundary cell*. Report the count both ways.

## 2. Bounded nonlinear test on GM (flux form, S_c free); runs first

**Question.** On Gabilan Mesa, does a partitioned model with flux-form nonlinear hillslopes fit as well as the linear joint, *and* return a D near the measured hilltop D₀?

### 2.1 Runs

On GM1–GM5 only, at the site m = 0.31:

- **Form and solver:** flux form, Newton with continuation in D, as built in Addendum 9.
- **K:** fixed at the linear joint optimum.
- **S_c:** four values per tile: 1.25, 0.9, the Addendum 8 primary-network fit, and that fit's 84th percentile.
- **D:** 14 log-spaced values from 3×10⁻³ to 5×10⁻² m²/yr. This brackets D₀ (7.6–10.1×10⁻³) and the linear D_fit (1.9–2.7×10⁻²).
- **A_crit:** 12 log-spaced unique areas spanning the linear joint 5 % interval, widened ×2 on each side.

That is 4 × 14 × 12 = 672 evaluations per tile.

### 2.2 Budget and safeguards

1. **Pilot first:** one A_crit row at S_c = 1.25 on GM1 and GM3.
2. **Estimate.** Print the core-hour estimate. Submit without asking only if:
   - the total is at most 600 core-hours;
   - every submission is at most 300 core-hours;
   - every array is throttled.

   Otherwise cut the A_crit values from 12 to 8, re-estimate, and if it is still over 600, stop and report.
3. **Partial results.** Write them after every A_crit row.
4. **Stop rule per tile and S_c:** if more than 30 % of a tile's evaluations at that S_c end `not_converged` (not `no_steady_solution`), stop that S_c and report it.

### 2.3 Rule, written now

The flux-form model is **preferred on GM** if, on at least 4 of 5 tiles, at some S_c ≥ 0.9, both of these hold at the best (A_crit, D):

- **fit:** MSE ≤ 1.05 × the linear joint MSE;
- **D recovery:** D/D₀ within 0.67–1.5.

**Reporting:**
- Report the same for every S_c, the fitted ones included, with the outcome counts.
- If only the fit or only the D condition holds, say which, on which tiles.
- **S_c < 0.9 can support "closer D" but cannot carry "preferred".** Those values sit well below any literature S_c, and §1.2 records why.

## 3. Morphology score (Addendum 9 §4, on existing surfaces)

Run `lemcvt a9 morphology` on every tile for:
- the linear joint optimum;
- the AD optimum;
- the Phase 6 nonlinear joint (centre form, labelled as such);
- the §2 best flux-form surface on GM, if one exists.

**Measures:**
1. Median crest curvature at 11 m.
2. Median E*–R* ratio 2S̄/(C_ht L_H) on the 10³ m² network.
3. Median and 90th-percentile hillslope gradient.

Each measure is computed identically on the observed DEM and on each model surface, on the same pixels and paths.

**Rule,** as in Addendum 9 §4.3: all three measures within ×1.25 of observed. Report per tile, beside the MSE. No combined score.

## 4. Test 2: fits on the accepted head sets

Run the remaining Test 2 steps from the feature branch's design:
- on AP, use the DT = 2 m² head sets;
- on GM, use the existing head sets.

Compare head area and head slope with the fitted A_crit and with Montgomery & Dietrich's AS² envelope (500–4000 m²). Report; no rule beyond the branch's own.

## 5. Close-out

1. **CI.**
   - Read the logs of GitHub run 42940255348. They are public run logs: try `gh run view --log-failed` from a machine with `gh`. If unreadable from Sherlock, say so; Steve will paste the failing step.
   - Fix the cause and restore the `push` trigger.
   - Gate: one green push run.
2. **Labels and docs.**
   - Update `docs/measurement_methods.md` with the final methods: θ/m, k_s, D₀ (the GM value and the AP bracket), E*–R*, and the threshold.
   - Bring `docs/numerics_changes.md` up to date.
   - The README: install, data fetch, and one-command reproduction of every figure and table, with a run-time table.
3. **Final build.**
   - `lemcvt figures all` and `lemcvt tables all` from results only, Table 1 included (LaTeX booktabs + siunitx, and CSV).
   - Every figure's caption data in a CSV beside it.
   - Copy the final set to `results/v2.0.0/` on group storage.
4. **Manuscript corrections list.**
   - `reports/v2/corrections_table.md`: every number in the current manuscript (`Dobbs_Hilley_JGR-ES_Manuscript_REVISED-tracked.docx`) and its supplement, with old value, new value, cause, and the run ID.
   - Steve will place a copy of both `.docx` files in the repo's `reports/inputs/` (gitignored); read them there. Do not edit them.
   - Mark every nonlinear-stage number "centre form, nonstandard: relabel or drop".
5. **Release.**
   - Merge `v2` into `main` and tag `v2.0.0`, after Steve approves the final report.
   - **Ask before the merge and the tag.** Do not push the tag without that approval.

## 6. Order and report

1. §2 pilot and estimate.
2. §2 runs; §3, §4 and §5.1–5.4 run alongside them.
3. One report: `reports/v2/<YYYYMMDDTHHMM>PST_wrapup.md`. Fixed outline:
   1. State check.
   2. GM flux-form test, with the rule applied.
   3. Morphology scores.
   4. Test 2.
   5. CI.
   6. Final build and corrections list.
   7. What the evidence supports, and where it is weak.
   8. Decisions needed: the release approval.
   9. Appendices: all generated tables.
4. **Stop for review.** §5.5 runs only after Steve approves.

Do not fix numerical issues outside this addendum silently. Report them.
