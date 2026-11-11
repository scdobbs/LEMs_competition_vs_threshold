# [2026-11-10 16:00 PST] Addendum 5 to the v2 refactor directive: Phase 6.0 checks, then the Phase 6 rerun

This addendum overrides the directive and Addenda 1–4 wherever they conflict. The reporting conventions of Addendum 4 §1 (timestamped titles, one report per checkpoint) are permanent.

## 0. Orientation (you are a new agent instance)

1. **Read, in order:**
   - `reports/directives/refactor_v2_directive.md` and Addenda 1–4;
   - this file;
   - `reports/v2/20261110T1540PST_checkpoint_4b_5.md` (the latest state; it cites everything earlier);
   - `docs/numerics_changes.md` and `docs/measurement_methods.md`.
2. **Confirm the state:**
   - the `v2` head equals `origin/v2`;
   - the tests pass on `serc`;
   - no jobs from earlier sessions are still running.
3. **Where this addendum and the repo disagree,** the repo wins. Report the disagreement and stop.

Addendum 2 §4 (Slurm and storage) and §5 (push) still apply.

## 1. Decisions

Steve will strike any he does not accept before this reaches you.

1. **The exponent m is a site constant: AP m = 0.47, GM m = 0.31.** These are the minima of the site profile S(m) (checkpoint 4b/5, §5.2). Every partitioned, nonlinear and AD stage uses it.
   - Per-tile m stays an SI analysis. The m-profiles already run in metres are its data, so nothing is rerun for it.
   - The site m is written into the tile configs, with provenance (run IDs, S(m) intervals).
2. **K₀ = U/k_s,** with k_s the per-tile χ–z slope at θ = site m over the channel network above A_c = 10³ m², using the pooled through-origin estimator.
   - Record n_eff and the site-pooled k_s beside each value.
   - The 10⁴ m² value is reported, not used.
3. **D₀ comes from curvature option (iii),** conditional on the §2.2 test: fixed window L_f = 11 m, applied to the rollover-network hilltop set with the L_f/2 exclusion.
4. **AP3's partitioned stages use K₀ = U/k_s,** not the as-run value of 10⁻⁶.
5. **AP1's flow-direction seam is corrected within the tile,** by §2.1. New grids are written; the originals are kept.
6. **Raise the Picard cap to 100** for every nonlinear evaluation, and record cap hits as before.
7. **Add a nonlinear joint profile over (A_crit, D),** per §3.4. It gives the D comparison a model whose hillslopes can be planar away from the crest.

## 2. Phase 6.0: checks and measured inputs (before the full rerun)

### 2.1 Correct AP1's seam within the tile

The stored `*_area` values at the 345 seam cells include all their upstream area, including area from outside the tile. Rerouting a seam cell therefore only moves that quantity from its old downstream path to its new one. This can be done exactly inside the tile, without the mosaic.

1. **Corrected flow directions:** for each of the 345 cells (rows 389–390), set fd to the steepest-descent code computed from the filled grid with the TopoAnalysis convention.
2. **Corrected areas:** process seam cells in topological order (upstream first).
   - Subtract the cell's stored area along its old D8 path, from the old receiver downstream to the tile exit.
   - Add it along the new path.
   - Handle seam cells that lie downstream of other seam cells by using their already-corrected areas.
3. **Re-entry check.** If any corrected or old path leaves the tile and could re-enter it, count those paths and report them; the in-tile correction cannot see re-entry.
4. **Report:**
   - the number of cells whose area changes, and the distribution of the relative changes;
   - the number of cells with A ≥ 10³ m² whose area changes by more than 1%;
   - the change in the channel mask at the AP1 joint A_crit from the checkpoint.
5. **Write the corrected grids** as `AP_1m_best_tile_{fd,area}_seamfix.tif`, add them to the manifest, and point the AP1 config at them. Keep the originals.
6. **Required test:** on a synthetic grid with a planted two-row seam, the in-tile correction reproduces a full recomputation exactly.

### 2.2 Does option (iii) hold on its own pixel set? (pre-registered test)

The decline of C_ht with window length out to 20–50 m on AP (`02b`, checkpoint §6.1) was measured on the hilltop set without exclusion. That set's median crest width (2d) is 4–14 m, so windows longer than about 10 m are expected to average in the flanks. The decline is therefore what the width bias predicts, and not yet evidence against option (iii).

1. **Compute median C_ht(L) on the option-(iii) pixel set** (post-exclusion) for L = 7, 9, 11, 15, 21, 25, 31, 41 and 51 m, per tile, with segment-bootstrap 68% bands. Plot one panel per tile.
2. **Pass rule, written now.** For L from L_f (11 m) up to the set's own median crest width, C_ht must stay within ±10% of C_ht(L_f).
   - If it passes on at least 4 of 5 tiles at each site, option (iii) stands.
   - Otherwise, stop and report before anything is written into the configs.

### 2.3 Measured inputs

Once §2.1 and §2.2 pass, write into each tile config, with provenance (run ID, git hash, n_eff, bootstrap interval):

- m (§1.1);
- K₀ (§1.2);
- D₀ = U/C_ht from option (iii);
- the AP3 change (§1.4).

This is the first time measured values from the new pipeline enter the configs. Keep the v1 values in the config under `v1_reference`, for the old-against-new record.

## 3. Phase 6: the full rerun (all ten tiles, metres, v2 optimizer)

### 3.0 Budget first

Print the core-hour estimate for every item below, and the total. Every single submission must stay at or below 300 core-hours; split arrays if needed. Submit without asking only if the total is at or below 900 core-hours; otherwise ask.

### 3.1 AD profile

The profile over r = K/D at the site m, as in Phase 4.

### 3.2 Linear partitioned model

- **Stage 2:** the A_crit profile at the new K₀ and D₀.
- **Joint profile** at the site m.

For both: thin to 2000 values and refine, as in Phase 4. Then evaluate every unique area inside the 5% interval of the refined profile, not only around the minima. Report how many values that took.

### 3.3 Nonlinear stage

The A_crit profile at the new K₀ and D₀, with S_c = 1.27 and the Picard cap at 100.

### 3.4 Nonlinear joint profile (new)

**Why it decouples.** In the partitioned model, channel elevations depend only on K, so K is identified mainly by the channels.

**Setup.**
- Fix K at the linear joint optimum of §3.2.
- Profile MSE over a grid of (A_crit, D):
  - A_crit: 30 log-spaced unique areas spanning the linear joint 5% interval, widened by a factor of 3 on each side;
  - D: 25 log-spaced values from 10⁻³ to 10⁻¹ m²/yr.
- S_c = 1.27; Picard cap 100.

**Report per tile:**
- the optimum, the MSE and the 1% and 5% contours;
- cap hits;
- the hillslope slope fractions at the optimum.

**K check.** On GM1 and AP5, repeat the profile at K × 0.95 and K × 1.05, and report how much the optimum and its MSE move.

### 3.5 D∞ comparison (SI)

On AP1 and GM1, run the v2 joint profile for the rebuilt D∞ partitioned model at the site m (the reduction applies). Report it beside the D8 joint optimum.

### 3.6 Figures and their inputs

- **The resolution sweep** (`supp_fig6`, `resolution_study_dinf`) runs on metre grids at the new measured K₀, D₀ and site m, as in v1's design (measured, not fitted, parameters). Use the rebuilt D∞ model.
- **`fig_slope_area`** uses the new optima.
- **`supp_fig5`** is synthetic and unchanged, but regenerate it so every figure comes from one pipeline version.

### 3.7 D-recovery diagnostics

Report these with no interpretation beyond the stated comparisons. Per tile:

- D_fit from the linear joint and from the nonlinear joint;
- D₀ by option (iii), and by the `02b` set (fixed L*, no exclusion) for the record;
- their ratios;
- the fitted surface's median Laplacian on the option-(iii) hilltop pixels (for the linear model this is −U/D_fit wherever those pixels are hillslope cells), next to the observed C_ht at L_f;
- the fraction of option-(iii) pixels that are channel cells in each fitted model.

### 3.8 Old-against-new record

Write one table covering every number the v1 manuscript reports: v1 value, v2 value, ratio, and cause or causes. The causes are:

- vertical units;
- optimizer (v1 against v2);
- measurement (new pipeline);
- m (site constant);
- AP1 seam;
- D∞ rebuild;
- unconverged v1 nonlinear.

Build it from results with one command. No manual entries.

## 4. Report and stop

Write one report, `reports/v2/<YYYYMMDDTHHMM>PST_phase6.md`, titled with its timestamp. Its outline is fixed:

1. State check.
2. Phase 6.0: seam, the option-(iii) test and the measured inputs.
3. Budget and runs.
4. Results by model: AD, linear partitioned, nonlinear, nonlinear joint, D∞.
5. D-recovery diagnostics.
6. Figures.
7. Old-against-new record.
8. What the evidence supports, and where it is weak.
9. Decisions needed.
10. Appendices: all generated tables.

**Stop for review.** Phase 7 (the side analyses) waits.

Do not fix numerical issues outside this addendum silently. Report them.
