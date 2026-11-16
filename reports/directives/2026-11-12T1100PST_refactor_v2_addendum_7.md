# [2026-11-12 11:00 PST] Addendum 7 to the v2 refactor directive: close the Phase 6 decisions, then one diagnostic for the D gap

This addendum overrides the directive and Addenda 1–6 wherever they conflict. Addendum 4 §1 (timestamped titles, one report per checkpoint) and Addendum 2 §4–§5 (Slurm, storage, push) still apply.

## 0. Orientation (you are a new agent instance)

1. **Read, in order:**
   - the directive and Addenda 1–6 in `reports/directives/`;
   - this file;
   - `reports/v2/20261112T1003PST_phase6_results.md` (latest state);
   - `docs/numerics_changes.md` and `docs/measurement_methods.md`.
2. **Confirm the state:**
   - push any unpushed commits on `v2` under the Addendum 2 §5 check;
   - the tests pass on `serc`;
   - no earlier job is running.
3. **Where this addendum and the repo disagree,** the repo wins. Report it and stop.
4. **Every command that renders a Slurm script** runs from a shell with `LEMCVT_DATA` and `LEMCVT_RESULTS` exported. Add a check to `lemcvt slurm submit` that refuses to render if either is unset or empty. This was submission fault 3 of the last session.

## 1. Decisions on the Phase 6 report

Steve will strike any he does not accept before this reaches you.

1. **Nonlinear joint D grid: option (a).** Report D_NL = D_lin to within one grid step (×1.2). No refinement run.
2. **GM1 K sensitivity: report as is.** No K check on the other GM tiles. The text states that the GM1 nonlinear joint A_crit moves by 32% under a 5% change of K, and that AP5's does not.
3. **`fig3_deltas`: rewrite the builder** to read the per-stage Phase 6 runs. Plot log₁₀(K_fit/K₀) and log₁₀(D_fit/D₀) per tile, for AD, the linear joint and the nonlinear joint. AP D is drawn at both ends of the bracket.
4. **CI.** The push trigger stays off until Steve or a planner reads the logs of run 42940255348. Do nothing more on CI in this session.
5. **Phase 7 waits** for the §2 result.

## 2. Where does the fitted D come from? (pre-registered diagnostic)

**Context.** The D gap survived every test so far:
- the linear joint D is 1.9–8.3× the measured D₀;
- the nonlinear joint returns the same D, with the nonlinearity barely engaged (1–11% of hillslope cells above 0.5 S_c);
- the gap holds at every curvature window.

So nonlinear hillslope transport with S_c = 1.27 does not explain it.

**Hypothesis to test (not assumed).** In the partitioned model every cell with A < A_crit is a linear-diffusion hillslope cell, so its −∇²z equals U/D: convex everywhere. Real landscapes between the slope–area rollover (18–450 m²) and A_crit (860–7900 m²) contain colluvial hollows, which are concave or nearly planar. A least-squares fit over all hillslope cells must average convex divides with non-convex hollows. That average lowers the fitted convexity U/D and so raises D. On this reading D_fit is an area-averaged effective diffusivity, not the hilltop diffusivity, and the gap is a structural limit of a two-domain partition, not a measurement error.

All runs are in metres, at the site m, at each tile's Phase 6 linear joint optimum. Use the same reduction and misfit-mask machinery as the AP1 seam test.

### 2.1 Observed curvature by area class

Split the model hillslope cells (A < A_crit at the joint optimum; boundary and channel cells excluded) into three classes:

- **divergent:** A < A_roll, the tile's slope–area rollover from `02b`;
- **hollow:** A_roll ≤ A < A_crit;
- **hilltop:** the option (iii) pixel set. This is a subset of divergent; report it separately.

For each class, report:
- the cell count and its share of hillslope cells;
- the mean and median observed −∇²z at a 3-cell window (the model's own stencil scale) and at L_f = 11 m;
- the fraction of cells with −∇²z < 0 (concave);
- the class's share of the joint-optimum hillslope SSR.

Report the same mean over all hillslope cells, against U/D_fit.

### 2.2 Refit with the misfit restricted to each class

Rerun the linear joint profile three times per tile, with the misfit computed over only:

1. divergent cells plus channel cells;
2. hollow cells plus channel cells;
3. all cells (the Phase 6 run; do not rerun it).

Channel cells stay in each mask so that K stays identified. Report A_crit, K, D and the MSE over the mask for each case, and D/D₀ (both ends of the bracket on AP).

### 2.3 Rule, written now

Apply it per site. The hypothesis is supported at a site if, on at least 4 of 5 tiles, all three hold:

- the mean observed −∇²z over all hillslope cells at the 3-cell window is within ±30% of U/D_fit;
- the hollow class's mean −∇²z is below half of the divergent class's;
- D from the divergent-only refit is at least a factor 2 closer to D₀,A than D_fit is, in log ratio.

If the hypothesis is not supported, report which condition failed and on which tiles. Propose nothing further.

### 2.4 Cost

Two extra profiles per tile, with no nonlinear solves. Print the estimate. It should be well under 300 core-hours in total; submit without asking if it is.

## 2b. Suspected units bug in `fig_slope_area` on AP (do this first; stop if confirmed elsewhere)

**The observation.** In `fig_slope_area` (run `20261112T175641Z_slope_area_10tiles`), the AP partitioned and AD-opt points sit below the observed points by a roughly constant factor of about 3 at every drainage area. The lines are parallel, not rotated. GM matches.

**Why this looks like a bug, not a model property.** The fitted linear joint model's channel slope is exactly S = (U/K)·A^(−m). On AP1 (U = 5.5×10⁻⁵, K = 4.00×10⁻⁶, m = 0.47) that gives S ≈ 0.18 at 10⁴ m² and ≈ 0.021 at 10⁶ m², on top of the observed points (≈ 0.22 and 0.02). The figure's partitioned fit (k_s = 8.43, θ = 0.522) gives ≈ 0.069 and ≈ 0.007, about 2.6–3× lower. A factor near 3.28 that appears on AP only, and on both model rows, points to the model surfaces being converted from feet to metres twice, or the figure applying the AP `z_units` conversion to a grid that is already in metres.

1. For each AP tile, compare the relief (max − min) and the mean of the model grid used by the figure with the observed metre grid and with the Phase 6 joint-run output grid. Report the ratios.
2. Recompute the MSE of the figure's model grids against the observed metre grid. It should equal the joint MSE (22–42 m²). Report it.
3. On channel cells above A_crit, compare the figure's binned model slopes with (U/K)·A^(−m) evaluated at the bin centres.
4. Trace the unit handling in the slope–area builder and its forward-run path. Find where the factor enters.
5. **If it is a bug:** fix it, add a regression test (an AP synthetic grid in ftUS whose slope–area output must match the metre version), and regenerate the figure. Check every other consumer of model output grids on AP for the same conversion: the curvature comparison of addendum 6 §4, `d8_vs_dinf`, the resolution sweep and the old-against-new record. Report which ones were affected. Note: the curvature comparison found −∇²z = U/D_fit exactly on the fitted surfaces, which suggests that path read the grids correctly; confirm it.
6. **If it is not a bug,** explain the factor quantitatively before anything else in this addendum is interpreted.

Add a section "Slope–area offset" to the report, after §2 of the outline.

## 3. Report and stop

One report, `reports/v2/<YYYYMMDDTHHMM>PST_d_diagnostic.md`, titled with its timestamp. Fixed outline:

1. State check and push.
2. Decisions applied (§1), including `fig3_deltas`.
3. Curvature by area class (§2.1).
4. Class-restricted refits (§2.2).
5. Rule applied (§2.3).
6. What the evidence supports, and where it is weak.
7. Decisions needed.
8. Appendices: all generated tables.

**Stop for review.**

Do not fix numerical issues outside this addendum silently. Report them.
