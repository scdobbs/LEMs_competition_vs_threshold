# [2026-11-12 16:30 PST] Addendum 8 to the v2 refactor directive: after the hollow test, a pre-registered test of nonuniform hillslope curvature (E*–R*), a slope-dependent threshold, and Phase 7

This addendum overrides the directive and Addenda 1–7 wherever they conflict. Addendum 4 §1 (timestamped titles, one report per checkpoint), Addendum 2 §4–§5 (Slurm, storage, push) and Addendum 7 §0.4 (submit guard) still apply.

## 0. Orientation (you are a new agent instance)

1. **Read, in order:**
   - the directive and Addenda 1–7 in `reports/directives/`;
   - this file;
   - `reports/v2/20261112T1600PST_d_diagnostic.md` (latest state), then `20261112T1003PST_phase6_results.md`;
   - `docs/measurement_methods.md` and `docs/numerics_changes.md`.
2. **Confirm** that `v2` equals `origin/v2`, that the tests pass on `serc`, and that no earlier job is running.
3. **Where this addendum and the repo disagree,** the repo wins. Report it and stop.

## 1. Decisions

Steve will strike any he does not accept before this reaches you.

1. **Record the verdict.** The hollow-averaging hypothesis is not supported at either site. The D gap is set by the divergent hillslope cells themselves.
2. **Erratum line.** Add one line under the title of `20261112T1003PST_phase6_results.md`:
   > Erratum: the slope–area figure and the B10 AP1 slope–area rows in this report were wrong (feet conversion applied twice); see 20261112T1600PST_d_diagnostic.md §2b.

   This is the only edit to an earlier report that is allowed.
3. **Planner note, for the record.** Condition (a) of Addendum 7 §2.3 tested a noise-dominated statistic: the mean Laplacian of a 1 m grid, with about half the cells concave at the 3-cell window. Do not reuse it.
4. **Phase 7 goes ahead in parallel with §2 and §3,** per the directive's Phase 7: width analysis, Test 2 (head extraction) and the AP reruns. **Test 1 (A_crit against head slope) is replaced by §3.** Test 2 still stops at its own review gate. Phase 7 output goes in its own section of this checkpoint's report.
5. **CI stays manual.**

## 2. Is the D gap the signature of nonuniform hillslope curvature? (pre-registered)

**Context.** Measured crest curvature implies D₀. Hillslope relief implies D_fit, which is 2–8× larger. Steady linear diffusion forces one uniform curvature, U/D, on every hillslope cell, so it cannot honour both. Sharp crests over gentle, nearly planar flanks are the signature of slope-dependent transport (Roering et al. 1999, 2007). The nonlinear joint fit of Phase 6 did not engage, because with S_c = 1.27 only a few per cent of hillslope cells exceed 0.5 S_c. The open question is whether the hillslopes fall on a nonlinear steady-state curve with a lower effective S_c, or on none.

The tool is the E*–R* framework of Roering et al. (2007), as applied per hilltop pixel by Hurst et al. (2012). All quantities come from the observed DEM in metres; none comes from a model.

### 2.1 Definitions (state them in `docs/measurement_methods.md`)

- **C_ht:** −∇²z at each option (iii) hilltop pixel, at L_f = 11 m (the configs' estimator).
- **L_H:** the hillslope length. Trace steepest descent from the hilltop pixel along the D8 directions to the first cell of the channel network. The primary network is A ≥ 10³ m² (the K₀ network); report sensitivity to 10⁴ m² and to the rollover network.
- **R:** the elevation drop along that trace. **S̄ = R/L_H.**
- **E\* = 2 C_ht L_H / S_c** and **R\* = S̄ / S_c.**
- **Steady nonlinear curve** (Roering et al. 2007, eq. as in Hurst et al. 2012): R\* = (1/E\*)·[√(1+E\*²) − ln(½(1+√(1+E\*²))) − 1]. The linear-diffusion limit is R\* = E\*/2.
  - Verify this curve numerically before use: integrate the 1-D steady nonlinear hillslope profile and check it to 10⁻⁶. If the published form disagrees with the integration, use the integrated curve and say so.

### 2.2 Analyses, per tile and pooled per site

1. **Linear test, independent of S_c.** In the linear limit, R* = E*/2 reduces to S̄ = C_ht·L_H. Report the median of S̄/(C_ht·L_H) over hilltop pixels, with a segment bootstrap. A value of 1 is linear; below 1 is the nonlinear signature.
2. **Fit S_c** by least squares of log R* against the steady nonlinear curve, over hilltop pixels, with a segment bootstrap. Report S_c and its 68% interval.
3. **Plot** E*–R* per tile at the fitted S_c, with the linear line and the nonlinear curve. Bin medians with interquartile ranges.

### 2.3 Rule, written now

The nonuniform-curvature reading is supported at a site if, on at least 4 of 5 tiles:

- the median S̄/(C_ht·L_H) is below 0.67, with the 68% interval below 1; **and**
- the fitted S_c lies inside its own bootstrap range and is less than 1.27.

Report the result against all three networks. The primary network decides.

### 2.4 Follow-up, run only if §2.3 is supported at a site

Rerun the nonlinear joint profile over (A_crit, D) on that site's tiles, with S_c set to each tile's fitted value:

- K fixed at the linear joint optimum;
- the Addendum 6 grid extended down to D = 10⁻³ m²/yr;
- Picard cap 100.

Report the optimum D, D/D₀ (both ends of the bracket on AP), the MSE against the linear joint MSE, cap hits, and the fraction of hillslope cells above 0.5 S_c.

Pre-registered reading:
- the gap is **closed** if D/D₀,A falls within 0.67–1.5 on at least 4 of 5 tiles (on AP, within the bracket widened by ×1.5);
- the gap is **narrowed** if it shrinks by at least a factor 2 in log ratio;
- **otherwise not.**

Print the core-hour estimate before submitting. Low S_c may stall the Picard iteration; if more than 30% of evaluations hit the cap on any tile, stop that tile and report rather than raising the cap.

## 3. A slope-dependent channel threshold (replaces Phase 7 Test 1)

Montgomery & Dietrich (1992) found channel heads bounded by AS² = 500–4000 m², per unit contour length (A/b)S² = 25–200 m. Montgomery & Dietrich (1989) fit A = 1978·(tan θ)^−1.65 m². This section generalizes the partition threshold to A·S^α ≥ C.

### 3.1 Definitions

- **Slope:** the observed local valley slope over a 10 m reach along D8, from the metre DEM. Fixed, so the reduction stays exact.
- **Channel heads:** cells where A·S^α ≥ C first holds along a flow path. Every cell downstream of a head is a channel. Head cells must drain to the tile boundary; report any that cannot.
- With α = 0 this must reproduce the Phase 6 A_crit model exactly. Test that.

### 3.2 Runs

The linear joint profile over C, at α = 0, 0.5, 1, 1.5, 2 and 2.5 on all ten tiles, at the site m:
- thin to 2000 unique values of the head statistic, then refine;
- report per α the optimum C, K, D, MSE and channel-cell count.

### 3.3 Comparison, pre-registered

- **Hold-out.** Split each tile's basins (outlets on the tile edge, at the 10³ m² network) into two halves by alternating rank of basin area.
- **Procedure.** Fit on one half and score the MSE on the other, both ways. Average the two.
- **Rule.** A slope-dependent threshold is preferred at a site if the best α > 0 lowers the held-out MSE by more than 2% on at least 4 of 5 tiles.
- **Report** the best α per tile and per site.
- **Compare the fitted C at α = 2** with M&D's 500–4000 m², noting that theirs came from field-mapped heads with a different slope measure.

Report the D at each α as well. If a slope-dependent threshold changes the D gap, the next directive needs to know.

## 4. Report and stop

One report, `reports/v2/<YYYYMMDDTHHMM>PST_ER_threshold_phase7.md`, titled with its timestamp. Fixed outline:

1. State check and the erratum line.
2. E*–R* (§2.1–2.3), with the rule applied.
3. The nonlinear joint at fitted S_c (§2.4), if run.
4. Slope-dependent threshold (§3).
5. Phase 7 (width, Test 2 up to its gate, AP reruns).
6. What the evidence supports, and where it is weak.
7. Decisions needed.
8. Appendices: all generated tables.

**Stop for review.**

Do not fix numerical issues outside this addendum silently. Report them.
