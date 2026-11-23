# [2026-11-23 10:30 PST] Addendum 21: the Tennessee Valley channel-head test

This addendum follows Addendum 20 and its report `reports/v2/20261121T0630PST_perron_D0_heads_partB.md`, especially its section 12 (the TTV calibration). Continue on branch `v2.3-tiles`.
- **Scope:** Tennessee Valley only; GM and AP are paused.
- **Stop:** one, at the end. No gate stops the work.
- **Rules:** written in §6, before any result.

## 0. Orientation (you are a new agent instance)

### 0.1 The question
Can the threshold-partitioned model, with physically calibrated parameters, reproduce the Tennessee Valley landscape and put its channel heads where they were mapped in the field? And can the co-located advection–diffusion (AD) model do the same?

TTV is the one site with independent channel heads:
- Montgomery & Dietrich (1989) mapped them in the field (Fig. 3, Table 1);
- Steve picked 64 of them on that map, registered to the 2019 lidar.

### 0.2 What Addendum 20 established (section 12 of its report; read it in full)
1. **The configured TTV erosion rate was wrong.** 77 m/Myr is Heimsath et al.'s (1999) soil production rate under zero soil (P₀), not a lowering rate. Measured rates (Heimsath et al. 1999, Table 1):
   - soil-covered noses: 21–60 m/Myr (median 27, mean 35);
   - basin-averaged from stream sediment: 64 ± 12 and 66 ± 13 m/Myr (sub-basin 1), 102 ± 25 (sub-basin 2).

   TTV is not in morphological steady state (Heimsath et al. 1997): ridges lower 2–3 × more slowly than basins.
2. **The crest (divide-line) curvature estimator is validated.** It reproduces Perron's GM hilltop curvature and, with Heimsath's E and D, Heimsath's TTV rates. The published rock-equivalent D is **0.0025 m²/yr** (0.0032 with the measured densities). Perron's binned estimator does not work on 1 m D8 data.
3. **K is reasonably determined.** S* gives D/K ≈ 84, m ≈ 0.30, K_cal ≈ 2.4 × 10⁻⁵ m^0.34/yr at m = 0.33. K₀ = U/k_s and the fitted K lie within 0.4–1.3 × K_cal.
4. **Steve's pick positions are good; 1 m D8 areas at heads are not.**
   - On 1 m D8, hollow flow splits into parallel threads, so the area at a head is 10–60 × too small.
   - On a 12 m grid (map scale) the picks give a median of 7,200 m², against M&D's 7,400 m².
   - Every head quantity built on 1 m areas (A_F = 529 m², C_F, R5, R6, R14 of Addendum 20) is void.
5. **Only the area threshold fixed at M&D's 7,400 m² put model heads where they are mapped** (median 15 m, ratio 1.39). It beats AD on 8/11 tiles at 1.6 × the tuned model's MSE. The tuned threshold (P4) gets the count right but the places wrong (62 m).
6. **Fitted D at TTV is 5–17 × the published D.** AD's D is 0.003–0.5 ×.

   **Planner reading, tested here (§6, rule DX):** with correct heads, the fitted D is the relief-equivalent D of the real hillslopes. Its excess over the published D is the product of two factors:
   - the nonlinear-hillslope factor D_R/D_C;
   - the steady-state mismatch U/E_ridge.

### 0.3 Read, in this order
1. `reports/v2/20261121T0630PST_perron_D0_heads_partB.md`: section 12 in full, then sections 4–9.
2. `reports/v2/scripts/a20_ttv_cal.py`, `a20_fixed.py`, `a20_modelheads.py`, `a20_dstat.py`, and the Addendum 19 head and D_R scripts (`a19_heads.py`, `a19_common.py`, `a19_site.py`).
3. `src/lemcvt/analyses/slope_threshold.py`, `src/lemcvt/optimize/v2.py` (`PartitionedProfiler`), `src/lemcvt/analyses/er_star.py`.
4. `configs/sites/TTV.yaml` and the TTV tile configs.
5. Addendum 2 §5 (pre-push check).
6. This file.

### 0.4 Where things are
- **Addendum 20 run directory:** `$GROUP_HOME/lemcvt_results/20261120T185052Z_a20_perron_D0_heads_partB/` (including `ttv_cal/`).
- **Addendum 18 fits (AD, P4 and variants):** `$GROUP_HOME/lemcvt_results/20261119T143945Z_a18_expanded_fits/`.
- **TTV full-site 1 m grids:** `$GROUP_HOME/lemcvt_results/20261118T160326Z_a15_stage_b/grid_TTV/`.
- **M&D georeference:** `.../md_georef/md1989_fig3.gpkg`.
- **M&D Table 1:** `.../20261118T185206Z_a16_tile_finish/md_georef/md1989_table1.csv`.
- **Steve's heads:** `data/ttv/ttv_heads_manual_steve_20261119.csv` (64 rows; checksum as in Addendum 20).

### 0.5 Standing rules
As Addendum 19 §0.4, unchanged:
- reports, Slurm limits and the 300 core-hour submission cap;
- the never-commit list (including `papers/`) and units;
- no package code changes unless unavoidable. If unavoidable: tests, the full suite on `serc`, and the Addendum 18 regression gate.

New code goes in `reports/v2/scripts/a21_*.py`. Figures go in `reports/v2/figures/a21/`.

### 0.6 Before starting
1. **Confirm** `v2.3-tiles` against `origin`, that the tests pass on `serc`, and that no job is running.
2. **Confirm** the paths above.
3. **Where this addendum and the repo disagree,** the repo wins. Report it and stop.

## 1. Decisions (planner, 2026-11-23, approved by Steve)

### 1.1 Parameter values for the goal state
The models are steady-state models with one uniform rate U. TTV is not in steady state, so each parameter is assigned to the process it represents. All rates are rock-equivalent; D follows our convention D = E/|∇²z|.

| symbol | value | source | role |
| --- | --- | --- | --- |
| **U** | **6.5 × 10⁻⁵ m/yr** (65 m/Myr) | Heimsath et al. 1999 Table 1, sub-basin 1 (64 ± 12, 66 ± 13) | the model's uplift / boundary-lowering rate; sets K₀ |
| U sensitivities | 1.02 × 10⁻⁴ (sub-basin 2); 7.7 × 10⁻⁵ (the old value, for continuity) | ibid. | reported, not decisive |
| **E_ridge** | **2.7 × 10⁻⁵ m/yr** (median); 3.5 × 10⁻⁵ (mean) as sensitivity | ibid., soil-covered greywacke noses | lowering rate of hilltops; used only for the physical D check |
| **m** | **0.33** (site value, Addendum 17); 0.30 (S*) as sensitivity | Addenda 17, 20 | stream-power exponent, n = 1 |
| **K₀** | U / k_s at m = 0.33 (the configured K₀ × 65/77) | our χ–z k_s | measured K; K_cal ≈ 2.4 × 10⁻⁵ m^0.34/yr (A20 §12.4) as the independent check |
| **D_phys** | **0.0025 m²/yr** (0.0032 with measured densities) | Heimsath et al. 1999 (K = 50 cm²/yr × ρ_s/ρ_r); reproduced by our crest curvature at E_ridge (A20 §12.3a) | the physical transport coefficient |
| **D_form** | U / \|C_crest\| per tile | crest (D₀,N) curvature of A20, ×65/77 | the D a steady-state linear model needs to reproduce the measured crest **curvature** at rate U |
| **D_relief** | U · L_H² / (2R) per tile, at U = 65 m/Myr | relief traced to the **mapped-head network** (§1.3) | the D a steady-state linear model needs to reproduce the measured hillslope **relief** at rate U |
| **A_F** | median 12 m D8 area at the picks (decision 1.2; expected ≈ 7,200 m²); 7,400 m² (Table 1) as sensitivity | map-scale measurement | field area threshold (α = 0) |
| **C_F** | median A₁₂ · S₁₂² at the picks (decision 1.2) | map-scale measurement | field A·S² threshold (α = 2) |

Notes:
- **D_form / D_phys = U / E_ridge ≈ 2.4.** This is the steady-state mismatch. No single steady-state D can match both the physical D and the measured crest form at TTV. The report states this wherever D is compared.
- **Exact rescaling.** The steady-state solution depends on K/U and D/U only. Fits stored at U = 7.7 × 10⁻⁵ convert to U = 6.5 × 10⁻⁵ by multiplying K and D by 65/77; elevations and MSE are unchanged.
  - Rescale the Addendum 18 and 20 fits.
  - Verify by one direct evaluation per model type on TTV2 at the new U (tolerance 10⁻⁸ relative).
  - Bounds scale the same way: the D upper bound becomes 0.05 × 65/77 = 0.0422 m²/yr. Report hits.
- **Config.** In `configs/sites/TTV.yaml`:
  - set U = 6.5 × 10⁻⁵ with the source line above;
  - keep the old value as `U_legacy_P0: 7.7e-5` with the note "Heimsath 1999 P₀, not a lowering rate";
  - add `E_ridge: 2.7e-5`.

  Confirm that nothing outside TTV reads these keys, and that the regression gate (GM1, AP2) is unaffected.

### 1.2 Map-scale head measurement
The criterion is measured at the scale M&D measured on.
- **Grid:** block-mean the TTV full-site 1 m surface to **12 m**, pit-fill it, and route D8 (sensitivities: 10 m grid; D∞ at 12 m).
- **Area at a pick:** the largest A in the 3 × 3 block of 12 m cells around the pick (as A20 §12.3b).
- **S₁₂ at a pick:** the slope along the hollow axis above the head, as M&D measured it.
  - From the pick's 12 m cell (the cell giving its area), step one cell upslope to the donor with the largest area.
  - Divide the elevation difference by the centre-to-centre distance.
  - Sensitivity: two cells.
- **Fits:** log₁₀ A = log₁₀ a − b log₁₀ S over the picks (OLS on log A, 10,000-draw bootstrap, seed 20261123). Compare with M&D eq. 10 (a = 1,978 m², b = 1.65).
- **Field values:**
  - A_F = median A₁₂ at the picks;
  - C_F = median A₁₂ · S₁₂² over the picks, and C_F,1.65 with α = 1.65 as a sensitivity.
- **Pick 49** (55 m from the traced channels) is kept. Every statistic is also reported without it.

### 1.3 Channel networks from the map-scale criterion, used in the 1 m model
- **On the 12 m grid**, a cell is a **criterion head** where A₁₂ · S₁₂^α ≥ C first holds along its flow path, with S₁₂ the one-cell upslope max-area-donor slope of §1.2. Every cell downstream is channel.
  - For α = 0 this is A₁₂ ≥ A_F.
  - Cells with no upslope donor (divides) never start a channel.
- **Seeding the 1 m network:** inside each 12 m head cell, take the 1 m cell with the largest 1 m D8 area. That cell and every 1 m D8 cell downstream of it is channel in the 1 m model.
- **The mapped-head network** (for D_relief at TTV1–TTV3): seed in the same way from the 12 m cells of Steve's 64 picks.
  - For tiles outside the M&D polygon, use the observed criterion network at α = 0 and A_F. Flag those tiles.
  - D_relief is traced with `er_star.trace_to_channel` from the A20 crest (D₀,N) pixels to that network.
  - Report median L_H, R, D_relief and its segment-bootstrap 68 % interval.
- **If the partitioned model cannot take an arbitrary channel mask:**
  1. Find whether `PartitionedProfiler` (or the Phase 6 code) accepts one. If it does, use it.
  2. If not, add a mask entry point with tests: the full suite on `serc` and the regression gate.
  3. Check that the mask route reproduces PF at A_crit = 7,400 m² on TTV2 exactly when given the 1 m A ≥ 7,400 mask.

### 1.4 One head-extraction method for every surface
To compare models fairly, channel heads are extracted the same way from every surface:
- the observed lidar;
- each fitted model's steady-state surface;
- AD's surface.

Method:
1. Block-mean the surface to 12 m.
2. Route D8 on that 12 m surface (its own routing, not the observed).
3. Apply the α = 2 criterion with C_F (and the α = 0 criterion with A_F).

This is the model's own A·S² at map scale: it asks whether the model's landscape would make channels where the field does.

**Sensitivity:** valley heads from Perron et al.'s (2012) contour-curvature procedure, applied to each surface at Gaussian σ = 10 m:
- valley clusters with contour curvature above two standard deviations of its distribution and at least 10 m²;
- heads = the upslope ends of the cluster network.

**Measures, on TTV1–TTV3 inside the M&D polygon:**
- the distance from each pick to the nearest extracted channel cell;
- the ratio of extracted heads to picks;
- the same for M&D's traced channel lines (`md1989_fig3.gpkg`): the share of traced channel length within 15 m of an extracted channel, and the reverse.

### 1.5 Runs
All on the 11 TTV primary tiles at U = 6.5 × 10⁻⁵, m = 0.33.

| id | channel criterion | K | D | parameters | purpose |
| --- | --- | --- | --- | --- | --- |
| **F-phys / F-form / F-relief** | observed criterion network, α = 0 at A_F | K₀ | D_phys / D_form / D_relief | 0 | fixed step: does measured physics reproduce the landscape? |
| **F2-phys / F2-form / F2-relief** | observed criterion network, α = 2 at C_F | K₀ | as above | 0 | the same with M&D's A·S² criterion |
| **PF-A** | α = 0 at A_F (12 m seeded) | free | free | 2 | free K, D at the field area threshold |
| **PF-C** | α = 2 at C_F (12 m seeded) | free | free | 2 | free K, D at the field A·S² threshold |
| **SJ-C (headline)** | α = 2, C free (12 m seeded), joint profile over C | free | free | 3 | George's step: K, D and the head criterion all free, fitted to elevations only |
| **P4** | α = 0, 1 m A_crit free | free | free | 3 | the tuned model (Addendum 18; rescaled, not rerun) |
| **AD** | none | free | free | 2 | the co-located model (Addendum 18; rescaled, not rerun) |

Notes on the runs:
- **The fixed runs (F, F2)** are single evaluations. In them the heads come from the observed surface and do not depend on D. The head comparison of a fixed run therefore tests the criterion, and its misfit tests the model.
- **The self-consistent version** (heads iterated on the model's own slopes) is not run in this addendum. §1.4 applied to the fitted surfaces is its one-pass analogue.
- **SJ-C:**
  - profile over the unique values of the 12 m head statistic between the C values that give 1 m-equivalent channel extents of A = 10³ and 10⁵ m²;
  - thin to at most 400 values and refine around the minimum;
  - at each C, closed-form K, D (non-negative least squares), as in `slope_threshold`.
  - Report the MSE profile against C, and the 1 % interval of C.
- **Hold-out** (if the existing basin-half machinery takes the mask route without changes): report held-out MSE for SJ-C against P4. Otherwise report "not run".

## 2. Part A: values and map-scale heads
1. **Config and rescaling** (decision 1.1): the new TTV config; the rescaled Addendum 18 and 20 fits; the TTV2 direct checks.
2. **D values per tile:** D_phys, D_form (with E_ridge median and mean), D_relief (mapped-head network at TTV1–TTV3; criterion network elsewhere, flagged), and the ratios D_form/D_phys and D_relief/D_form.
3. **Map-scale heads** (decision 1.2): A₁₂, S₁₂ (one and two cells) and the 10 m and D∞ sensitivities at every pick; the fits against M&D eq. 10; A_F, C_F, C_F,1.65.
   - Figure: log A against log S with Table 1, the picks, eq. 10 and the fits.
4. **The criterion on the observed surface** (decision 1.4, observed lidar):
   - heads at α = 0 (A_F) and α = 2 (C_F) on TTV1–TTV3;
   - pick distances, head ratio, traced-line overlap;
   - figures: 12 m hillshade, criterion channels, picks, M&D lines; one per tile.

## 3. Part B: fits and the head test
1. **The mask route check** (decision 1.3) on TTV2.
2. **The fixed runs** F and F2 (6 single evaluations per tile).
3. **PF-A, PF-C** (single profiles in K, D at fixed networks).
4. **SJ-C** on 11 tiles. Time one tile first; print the estimate before submitting.
5. **Head extraction** (decision 1.4) on every surface: observed, F (three D), F2 (three D), PF-A, PF-C, SJ-C, P4, AD. Both methods.
6. **Figures:**
   - per tile (TTV1–TTV3), one panel per model: 12 m hillshade of the model surface, extracted channels, picks, M&D lines;
   - per tile (all 11): elevation residual maps for SJ-C, P4 and AD on one colour scale.

## 4. Cost
- **SJ-C** dominates: about 4 h per tile on Addendum 20 timings, so about 45 core-hours for 11 tiles. Time it first.
- **Everything else** is single evaluations, 12 m grids and figures, about 10 core-hours.
- **Expected total ≤ 80 core-hours** (about 880 of Steve's approved 1,000–2,000 used to date). No submission over 300.

## 5. Gates
None stops the work.
- **G0** (rule H0) and the mask check are reported first.
- If the mask route cannot reproduce PF at 7,400 m² on TTV2, run SJ-C and PF-C anyway and flag every mask-route result.

## 6. Rules (written now; "holds" = at least 80 % of N, rounded up; every rule reports its count and denominator)

**H0, the criterion on the real landscape.** On TTV1–TTV3, the observed 12 m criterion at α = 2 and C_F **locates the mapped heads** if both hold:
- the median pick-to-criterion-channel distance is ≤ 25 m;
- the ratio of criterion heads to picks inside the M&D polygon is within 0.67–1.5.

The same is reported for α = 0 and A_F.

**H0b, the picks against M&D's relation at map scale.** Consistent if both hold:
- 1.65 lies inside the 95 % interval of b;
- the median of A₁₂ / (1978 · S₁₂^−1.65) lies within 0.5–2.

**H1, model heads.** For each model, the heads extracted from its surface (decision 1.4, α = 2 at C_F) agree with the mapped heads by the two conditions of H0. Report the α = 0 extraction and the Perron-method sensitivity beside it, without verdicts.

**H2, the head comparison between models.** The partitioned headline (SJ-C) **places heads better than AD** if, on TTV1–TTV3 pooled, both hold:
- its median pick distance is smaller than AD's;
- its head ratio is closer to 1 on the log scale.

**K1, K.** A model's K is **physical** on a tile if K/K₀ lies within 0.5–2. It holds at the site if it holds on at least 9 of 11 tiles. Report K/K_cal beside it.

**DX, the D explanation.** For each free-D partitioned model (PF-A, PF-C, SJ-C, P4), D is **explained by relief** on a tile if D_fit / D_relief lies within 0.67–1.5. It holds at the site on at least 9 of 11 tiles (TTV1–TTV3 use the mapped-head network; the rest are flagged). Readings, written now:
- **If DX holds for SJ-C and PF-C:** the fitted D is what linear diffusion needs for the real hillslope relief at rate U. Its excess over D_phys equals (D_relief/D_form) × (U/E_ridge): nonlinear transport times non-steady state. It is not a defect of the partitioned model.
- **If D_fit > 1.5 × D_relief on most tiles:** the excess is not explained by relief; read it with H1 (wrong hillslope lengths) and K1.
- **AD:** report D_fit/D_phys, D_fit/D_form and D_fit/D_relief. No reading is set.

**M1, misfit.** SJ-C beats AD if MSE_SJ-C < MSE_AD on at least 9 of 11 tiles. The same is reported for PF-A, PF-C and the fixed runs.

**F0, the fixed step (report, with readings written now).** For F and F2 at each D, report MSE/MSE_AD and MSE/MSE_SJ-C. The planner predicts:
- **F-relief** fits best of the three;
- **F-phys** fits worst, with crests too sharp (model crest curvature U/D_phys = 0.026 m⁻¹ against the measured ≈ 0.016).

Report whether each prediction holds.

## 7. Report and STOP
One report, `reports/v2/<YYYYMMDDTHHMM>PST_ttv_channel_head_test.md`; figures in `reports/v2/figures/a21/`. Fixed outline:

1. State check, deviations, decisions as applied (config change, rescaling checks, mask route).
2. Parameter values per tile (the §1.1 table, filled).
3. Map-scale heads: measurements, fits, rules H0b and H0.
4. Fixed step: rule F0.
5. Free fits: PF-A, PF-C, SJ-C (with its C profile), P4, AD; rules K1, M1.
6. Model heads: rules H1 and H2, with figures.
7. The D explanation: rule DX, and the decomposition table (D_fit / D_phys against (D_relief/D_form) × (U/E_ridge), per tile).
8. Sensitivities: U (102, 77), E_ridge (mean), m (0.30), the 10 m grid, D∞, two-cell slope, pick 49.
9. What the evidence supports, and where it is weak (state the non-steady-state limitation plainly).
10. Decisions needed.
11. Appendices: tables, run ids, checksums, Slurm accounting.

**Stop.** Do not merge or tag. Push after the Addendum 2 §5 check.
