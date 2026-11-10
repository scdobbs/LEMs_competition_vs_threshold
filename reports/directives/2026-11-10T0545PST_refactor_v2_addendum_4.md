# [2026-11-10 05:45 PST] Addendum 4 to the v2 refactor directive: Phase 4 decisions, Phase 4b, Phase 5, and four short analyses

This addendum overrides the directive and Addenda 1–3 wherever they conflict.

## 0. Orientation (you are a new agent instance)

1. **Read, in order:**
   - `reports/directives/refactor_v2_directive.md` and Addenda 1–3 in `reports/directives/`;
   - this file;
   - the reports `00_reference.md`, `02_regression.md`, `02d_phase2_closeout.md`, `02c_measurement_diagnostics.md`, `02e_audits.md`, `03_performance.md`, `03b_phase3_close.md` and `04_optimizer_v2.md`, with their `*_tables.md`;
   - `docs/numerics_changes.md` and `docs/measurement_methods.md`.
2. **Confirm the state of the repo:**
   - the `v2` head;
   - that the tests pass, run on `serc`;
   - whether the Addendum 3 §1 push happened (if not, do it now under its rules);
   - the status of jobs 51697696 (AP5 nonlinear baseline), 51707582 (GM nonlinear, v2) and 51707588 (AP nonlinear, v2).
3. **Where this addendum and the repo disagree,** the repo wins. Report the disagreement and stop.

Addendum 2 §4 (Slurm and storage) and §5 (push) still apply.

## 1. Reporting conventions (new, permanent)

- **Every directive and every report starts its title with a timestamp** in the form `[YYYY-MM-DD HH:MM PST]`.
  - Report files are named `reports/v2/<YYYYMMDDTHHMM>PST_<short-name>.md`.
- **One report per checkpoint.** All parts of the checkpoint go in a single markdown file, with the generated tables as appendices of that same file.
  - Data products (CSV, parquet) may stay separate.
  - Figures are referenced by path.
  - Earlier reports are not renamed.

## 2. Decisions

Steve will strike any he does not accept before this reaches you.

1. **v2 is the optimizer for every reported number.** v1 stays available under `--optimizer v1` for the record.
   - Do not revert N7 (vectorized `A**m`). The L-BFGS-B path is not reproducible across CPU generations; record this in `docs/numerics_changes.md`.
   - The AD result is decided by the v2 profile: the AP5 v1 point is the global optimum.
2. **Cancel job 51707588,** the AP nonlinear v2 profile in feet. Every evaluation in it is an unconverged 50-iteration Picard result in the wrong units (`02e` §1), so the profile carries no information worth 24 core-hours per tile.
   - The AP nonlinear profile is rerun in metres in §5.6.
   - Let the GM nonlinear job (51707582) and the AP5 nonlinear baseline (51697696) finish. Report their rows in this checkpoint's report.
3. **Proceed with Phase 4b** (the D∞ rebuild) exactly as Addendum 3 §6 specifies. Its tests are its gate; it is no longer a separate stop.
4. **Proceed with Phase 5,** with the corrections in §5.
5. **The threshold rule for θ and k_s, and the curvature method, stay deferred.** §6 provides more evidence. No measured value goes into a tile config.

## 3. Corrections to earlier directives (planner errors)

- **The metre conversion was inverted in Addendum 3 §3.3.** In the linear models, z_m = z_ft/c corresponds to K_m = c·K_ft and D_m = c·D_ft, with MSE_m = MSE_ft/c². Use the exact US survey foot, c = 3937/1200 = 3.2808333…, so 1/c² = 0.09290341. The directive's Phase 5.1 factor of 0.3048 is the international foot; use 1200/3937 instead.
- **The premise of Addendum 3 §4.3.1 was wrong.** `lemcvt measure` defines hilltops from the rollover network, not from the 1e4 m² network.
- **Some AP quantities are unit-invariant and may be interpreted now.** For both linear models, scaling z by a constant leaves the following unchanged:
  - A_crit, m and r = K/D;
  - the shapes of the A_crit and m profiles (MSE scales by a constant);
  - every fitted-to-measured ratio formed in consistent units.

  Only absolute K, D and MSE need conversion. The `04` report's "no AP value is interpreted" was over-cautious for these quantities. The nonlinear model is not invariant (`02e` §1).

## 4. Phase 4b: the D∞ rebuild

As specified in Addendum 3 §6, with AP in feet. The `d8_vs_dinf` rerun goes into this checkpoint's report.

## 5. Phase 5: correct the AP vertical units (directive Phase 5, amended)

### 5.1 Corrected grids

Set `reproduce_v1_units: false` in the AP configs, converting with c = 3937/1200. Write corrected filled-elevation GeoTIFFs with new manifest entries, and keep the originals.

### 5.2 Do not regenerate the area grids on the tiles

The area grids were computed on a larger DEM and then clipped (`02c` §6). Recomputing them from the 1 km tiles would truncate every catchment that crosses the tile edge, and destroy the true-area property the measurement and the models rely on.

1. Keep the original `*_fd` and `*_area` grids.
2. **As a check only:** recompute D8 directions from the corrected filled tile and compare cell by cell with the original `*_fd`.
   - Away from the edges they should be identical, because scaling z by a constant does not change which neighbour is steepest.
   - Report the count of differing cells, separated into: edge-adjacent cells, flats (N1 tie order) and anything else.
   - Investigate the "anything else" class. Do not replace any grid.

### 5.3 Bounds

Scale the AP K and D bounds by c, and record the change in the config.

### 5.4 Measurement

Already done on metre grids (Addendum 1); nothing to redo here.

### 5.5 Invariance check (the directive's Phase 5.4)

Run the v2 partitioned profile (stage 2 and joint), the AD profile and the exponent profile on AP1 and AP5 in metres. Use the same A_crit grids as the feet runs, and the converted v1 K₀ and D₀ for stage 2. Then check that:

- the A_crit, m and r optima and the profile shapes equal the feet runs to 10⁻⁸ relative;
- K and D equal c × the feet values;
- MSE equals the feet MSE × 0.09290341.

If any of these fails, stop and report.

### 5.6 AP nonlinear in metres

Run the v2 nonlinear profile on all five AP tiles, at K₀ and D₀ converted from v1 (multiplied by c), with S_c = 1.27.

Report per tile:
- the optimum, the MSE and the 1% interval;
- the number of evaluations that hit the Picard cap (expected 0);
- the fraction of hillslope cells above 0.5 S_c and above 0.9 S_c.

This is the first converged AP nonlinear result. It still uses the v1 measured parameters, and Phase 6 replaces those.

Write `reports/v2/05_ap_units` as a section of this checkpoint's single report, not as a separate file.

## 6. Four short analyses for the deferred decisions

### 6.1 The exponent profile on a wider grid, then a site-level m

1. **Rerun the m-profile on all ten tiles** (AP in metres, after §5) with m from 0.05 to 0.90 in steps of 0.01. AP4's optimum sits on the old grid edge at 0.20. Use the same 200-value A_crit grid. The budget is about 1 h per tile.
2. **Site profile.** For each site, compute S(m) = Σ_t MSE*_t(m)/MSE*_t,min over the five tiles, where MSE*_t(m) is minimized over A_crit and (K, D) at each m.
   - Report the site optimum and its 1% and 5% intervals in m.
   - Also report S(m) with AP4 left out, and with each tile left out in turn (leave-one-out).
3. **Plot S(m) against the measured site-pooled θ(A_c) curves,** pooled and equal-weight with their 95% bands, evaluated at the median fitted A_crit of the site.

Do not interpret the plot beyond overlap statements.

### 6.2 A K-recovery comparison that is consistent in units

**Context.** k_s has units of m^{2θ}, so K₀ = U/k_s is comparable to a fitted K only when both use the same m.

1. For each tile, at the site-level m from §6.1, compute two quantities:
   - the fitted K from the joint profile at that fixed m (the A_crit profile at fixed m, which is one existing m-profile row set);
   - k_s from χ–z at the same m, using the site-pooled estimator over the channel network above a fixed A_c of 1e3 m², and again above 1e4 m². This keeps the measurement independent of the model's A_crit.
2. **Report U/k_s ÷ K_fit per tile, at both thresholds.** No interpretation.

### 6.3 Curvature: a third method option, anchored externally

1. **Choose a fixed window L_f.** Take the smallest window at which the `02c` synthetic gives facet-plus-noise bias below 1%, at both sites (the `02c` text suggests about 9 m). State L_f, and the synthetic results that set it, before computing any C_ht.
2. **Build the hilltop set** from the rollover network, as `measure` does. Exclude pixels closer than L_f/2 to a pixel of that network.
3. **Report per tile:**
   - pixel count and segment count;
   - C_ht at L_f with its segment bootstrap;
   - D₀ = U/C_ht;
   - the synthetic bias at the post-exclusion median width and at the 25th-percentile width.
4. **Put this option beside the two in `02e` A6** (fixed L* with and without exclusion, against each network) in one table. Every row must carry its hilltop definition, window and pixel count.

### 6.4 Two fixes in the measurement code (log both in `docs/measurement_methods.md`)

1. **`measure/diagnostics.hilltop_widths`** builds a hilltop set different from the one `measure` uses (`02e` §4.4). Make it use the same set. Leave `02c` as its record.
2. **Extend the θ grid down to 0.00.** At each ladder step, report the fraction of kept basins whose best θ lies at the lower edge, so that degenerate fits are counted rather than censored.

   Then rerun the `02e` free ladder: pooled, equal-weight and median estimators, with site pooling, n_eff and bootstrap. The run is cheap.

## 7. Report and stop

Write one report, `reports/v2/<YYYYMMDDTHHMM>PST_checkpoint_4b_5.md`, titled with its timestamp. Its outline is fixed:

1. State check and push status.
2. Nonlinear rows: GM v2 and the AP5 baseline.
3. Phase 4b.
4. Phase 5, with §5.2 and §5.5–5.6.
5. §6.1 and §6.2.
6. §6.3 and §6.4.
7. What the evidence supports, and where it is weak.
8. Decisions needed.
9. Appendices: all generated tables.

**Stop for review.** Phase 6 needs Steve's decisions on the threshold rule (or m), the curvature method, and the Phase 6 run list.

Do not fix numerical issues outside this addendum silently. Report them.
