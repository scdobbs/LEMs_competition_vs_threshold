# [2026-11-17 09:00 PST] Sensitivity (addendum 12): equal-parameter comparisons and per-tile concavity

Directive: `reports/directives/2026-11-16T1415PST_v2_addendum_12_sensitivity.md` (outline fixed by its section 4).
Branch `v2.1-sensitivity`, cut from `main` at the `v2.0.0` tag (b79947e); `main` and the tag are unchanged. Code
2090f03 and later on the branch (`lemcvt.analyses.sensitivity`, CLI group `lemcvt a12`; submission script
`reports/v2/scripts/a12_submit.sh`). Environment `lemcvt` (Python 3.12, numpy 2.5.4, scipy 1.18.1); every job on `serc`,
1 CPU. Every number is tied to a run id under `$GROUP_HOME/lemcvt_results` (appendix B11) or a Slurm job id.
Machine-generated tables: section 7 (`reports/v2/scripts/make_sensitivity_tables.py`, from summary run
`20261117T163709Z_a12_summary_10tiles`). **This is a stop for review.**

## Summary

| Item | Result |
| --- | --- |
| 1 state check | `main` = `origin/main` = tag `v2.0.0` = b79947e; no job running; tests passed on `serc` (job 52562243, 264 passed). Six points where the addendum and the repo or data differ; none needed a stop (section 1). The main one: **equal halves are impossible on 6 of 10 tiles** because one edge-outlet basin holds more than half the cells |
| 2 equal parameters, in-sample | **GM: the partitioned model wins at every count** (count 1 P1, count 2 P2 and P3: 5/5 each; both D₀ ends). **AP: it wins at count 2 only with D free (P3, 5/5).** With D fixed at the measured D₀ it loses to AD: P2 0/5 at both ends; P1 0/5 at D₀,A and 2/5 at D₀,B. The count-3 model (P4) is lower than AD on 10/10 |
| 2 held-out | P4 beats AD out of sample on **AP 4/5** (fails on AP4: 80.6 against 72.5 m²) and **GM 5/5**. P2 against AD (equal counts): AP 3/5 at both ends (not a win), GM 5/5. The split is lopsided where one basin dominates (AP1 0.80, AP3 0.81, GM5 0.78 of the split's cells) |
| 3 per-tile m | m_t: AP 0.54, 0.59, 0.39, 0.21, 0.42; GM 0.29, 0.29, 0.29, 0.36, 0.34. No m_t is on a grid edge; AP4's 5 % interval (0.05-0.38) reaches the lower edge. GM: the site m (0.31) lies inside all four measured θ bands. AP: the site m (0.47) lies outside both 10³ m² bands (pooled 0.39 [0.37, 0.41], equal-weight 0.29 [0.21, 0.34]) and inside both 10⁴ m² bands |
| 4 at m_t | **GM: the site-m choice is immaterial** by the 1.4 rule (partitioned < AD 5/5; median D/D₀,A −2 %, median AD K/K₀ −9 %). **AP: not immaterial.** The partitioned conclusions hold (MSE below AD 5/5, median D/D₀,A +5 %, K/K₀ 0.77-1.41). **The AD conclusion changes**: median AD K/K₀ 9.0 → 5.8 (−36 %), per tile 1.3-15 at m_t against 8.6-10.2 at the site m (AP4: 10.2 → 1.3) |
| cost | 165 core-hours used (all addendum 12 jobs, tests and smokes included); estimate 234 expected; largest submission 150 core-hours of limits; every array throttled (%10) |

## 1. State check

**At the start (2026-11-16 14:20 PST).**
- `main` = `origin/main` = the `v2.0.0` tag = b79947e, the merge commit (parents 28a9c90 and c1dd6bf).
- `squeue --me` was empty.
- Tests passed on `serc`: job 52562243 at b79947e, ruff clean, mypy clean, 264 passed.
- The branch `v2.1-sensitivity` was cut from the tag. Nothing on `main` or the tag was touched.
- At the report's code (80c0cd4 plus the report), the suite passed on `serc`: job 52679200, ruff and mypy clean (110 files), 269 passed.

**Where the addendum and the repo or data differ.** None needed a stop. Six points:

1. **Equal halves (2.2).** The greedy rule cannot give equal cell counts when one basin holds more than half the cells,
   and that happens on 6 of 10 tiles. Shares of the split's cells, half 1 / half 2:

   | tile | AP1 | AP2 | AP3 | AP4 | AP5 | GM1 | GM2 | GM3 | GM4 | GM5 |
   | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
   | half 1 / half 2 | 0.80 / 0.20 | 0.60 / 0.40 | 0.81 / 0.19 | 0.50 / 0.50 | 0.66 / 0.34 | 0.50 / 0.50 | 0.50 / 0.50 | 0.50 / 0.50 | 0.61 / 0.39 | 0.78 / 0.22 |

   On those six tiles the larger half is a single basin. I applied the rule as written and report the shares (B2). The
   split covers 76-97 % of the valid cells. The rest drain to network outlets inside the tile, or reach no 10³ m²
   channel; they are in neither half.
2. **What "fit on a half" means.** The model surface is always the whole-tile solution at the given parameters. Fitting
   on a half minimises the MSE over that half's cells, and the score is the MSE over the other half's cells. For the
   partitioned variants the fitting half also chooses A_crit; for AD it chooses r.
3. **The A_crit grid for P2 and P3.** The in-sample P2 and P3 minima are taken over the union of two row sets:
   - the Phase 6 refined rows, through their stored Gram terms;
   - this run's rows: the thinned 2000, a staircase around every variant's minimum, and that variant's 1 % interval.

   On AP the 1 % fill was capped at 1500 values; 3807-7054 were in the intervals. The minima from the thinned grid alone
   differ from the final ones by at most 0.3 % (B3).
4. **Count 2 has two partitioned variants.** P2 and P3 are reported separately against AD. "The partitioned model wins at
   count 2" holds for P3 on both sites and for P2 only on GM.
5. **The θ comparison (1.1)** uses the site-pooled estimators at A_c = 10³ and 10⁴ m², as the addendum says. The 4b/5
   checkpoint had interpolated them at the median fitted A_crit instead.
6. **Cost rule (3.1).** I read "the total" as the expected core-hours and "every submission" as its time limits. That is
   the reading used in addendum 10.
   - Expected total: 234 core-hours (run `20261116T223818Z_a12_estimate_10tiles`).
   - Time limits: 380 core-hours in all; the largest submission had 150.
   - Used: 165 core-hours.

   If the 300 was meant to bound the limits too, these submissions exceeded it, though no single submission did.

**Checks against Phase 6 (B10).**
- The partitioned rows recomputed here equal the Phase 6 rows at the same A_crit: exactly on AP, within 5.3 × 10⁻¹² on
  GM.
- The stored Gram terms reproduce the stored joint and stage-2 MSEs to 10⁻¹³.
- The AD optimum at the site m equals the Phase 6 AD optimum exactly on 10 of 10 tiles.
- So P1, P4 and AD here are the Phase 6 results.

## 2. Equal-parameter comparisons, in-sample and held-out, with the rule applied

**Variants**, all at the site m (AP 0.47, GM 0.31), from the exact reductions:
- **Partitioned:** P1 = stage 2 [A_crit]; P2 = D fixed at D₀ [K, A_crit]; P3 = K fixed at K₀ [D, A_crit]; P4 = joint.
- **AD:** A1 = K fixed at K₀ [D]; A2 = D fixed at D₀ [K]; AD = [K, D].
- Suffixes _A and _B give the D₀ end. GM's D₀,B lies within 5 % of D₀,A and is reported for symmetry only.

### 2.1 In-sample MSE (m²)

Full table in B3, parameters in B3b.

| tile | P1_A | P1_B | A1 | A2_A | A2_B | P2_A | P2_B | P3 | AD | P4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 289.2 | 86.4 | 6511 | 128.0 | 160.7 | 137.3 | 67.4 | 25.7 | 59.0 | 22.4 |
| AP2 | 245.8 | 185.7 | 867 | 130.0 | 144.2 | 113.1 | 92.3 | 22.8 | 76.5 | 22.5 |
| AP3 | 340.6 | 340.6 | 4533 | 183.5 | 183.5 | 156.1 | 156.1 | 42.2 | 152.8 | 42.1 |
| AP4 | 255.2 | 81.1 | 415 | 86.3 | 137.3 | 119.1 | 63.0 | 35.2 | 45.7 | 35.2 |
| AP5 | 226.3 | 190.5 | 3954 | 110.4 | 118.4 | 94.9 | 85.6 | 27.9 | 69.8 | 27.9 |
| GM1 | 51.3 | 48.4 | 483 | 171.4 | 172.9 | 38.6 | 37.3 | 24.7 | 49.0 | 24.7 |
| GM2 | 57.3 | 55.4 | 5119 | 284.0 | 285.9 | 48.6 | 47.6 | 26.1 | 82.5 | 26.1 |
| GM3 | 61.3 | 61.3 | 4825 | 120.5 | 120.5 | 39.0 | 39.0 | 34.3 | 64.7 | 33.0 |
| GM4 | 113.8 | 122.9 | 13661 | 579.1 | 575.1 | 72.2 | 74.3 | 42.3 | 174.5 | 40.7 |
| GM5 | 92.8 | 94.0 | 8588 | 273.4 | 272.6 | 62.5 | 62.8 | 33.6 | 93.3 | 32.2 |

### 2.2 The rule (2.3), in-sample

A variant wins on a tile if its MSE is lower than every AD variant of the same count; it wins at a site on at least 4 of
5 tiles.

| count | comparison | AP | GM |
| --- | --- | --- | --- |
| 1 | P1_A < min(A1, A2_A) | 0/5, no | 5/5, **wins** |
| 1 | P1_B < min(A1, A2_B) | 2/5 (AP1, AP4), no | 5/5, **wins** |
| 2 | P2_A < AD | 0/5, no | 5/5, **wins** |
| 2 | P2_B < AD | 0/5, no | 5/5, **wins** |
| 2 | P3 < AD | 5/5, **wins** | 5/5, **wins** |
| (3 against 2) | P4 < AD | 5/5 | 5/5 |

**What decides the AP result:**
- On AP every partitioned variant that holds D at the measured D₀ loses to AD at its own count. Every variant that frees
  D wins.
- With K fixed at K₀ and D free (P3), D goes to the same place as in the joint fit: 1.26-1.95 × 10⁻² m²/yr, 4.3-8.4 × D₀,A
  (B3b). The MSE is within 0.0-15 % of P4's (AP1: 25.7 against 22.4).
- So on AP, at equal counts, the partitioned model's advantage over AD comes with a hillslope D several times the
  measured one. Its K, by contrast, can be the measured K₀ at almost no cost.
- On GM the advantage survives D = D₀: P2_A 38.6-72.2 against AD 49.0-174.5.

**The AD variants with one parameter fixed fit badly:**
- **A1 (K = K₀)** fits worst everywhere: MSE 415-13,661 m². On AP1, AP3, AP5 and GM1 its D goes to the lower bound
  (≈ 10⁻⁹). AD needs K ≈ 9 × K₀ on AP and 2-3 × K₀ on GM (B8).
- **A2 (D = D₀)** gives 86-579 m². On AP it beats P1 on every tile at D₀,A. That is what loses count 1 on AP.

### 2.3 Held-out (2.2): fit on one half, score on the other, both ways, averaged (MSE, m²; B5, B5b)

| tile | split shares | P4 | AD | P2_A | P2_B | P4 < AD | P2_A < AD | P2_B < AD |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 0.80 / 0.20 | 36.2 | 1477.9 | 114.8 | 57.4 | yes | yes | yes |
| AP2 | 0.60 / 0.40 | 29.0 | 84.8 | 118.4 | 97.0 | yes | no | no |
| AP3 | 0.81 / 0.19 | 78.7 | 187.0 | 184.3 | 184.3 | yes | yes | yes |
| AP4 | 0.50 / 0.50 | 80.6 | 72.5 | 201.3 | 118.0 | **no** | no | no |
| AP5 | 0.66 / 0.34 | 62.4 | 124.7 | 109.0 | 100.6 | yes | yes | yes |
| GM1 | 0.50 / 0.50 | 35.7 | 67.7 | 46.5 | 45.6 | yes | yes | yes |
| GM2 | 0.50 / 0.50 | 40.3 | 131.6 | 52.2 | 52.7 | yes | yes | yes |
| GM3 | 0.50 / 0.50 | 41.6 | 69.4 | 42.0 | 42.0 | yes | yes | yes |
| GM4 | 0.61 / 0.39 | 61.6 | 193.0 | 80.6 | 82.9 | yes | yes | yes |
| GM5 | 0.78 / 0.22 | 61.2 | 167.7 | 78.0 | 78.7 | yes | yes | yes |

**Rule:**
- P4 beats AD out of sample on **AP 4/5** and **GM 5/5**: it passes at both sites.
- P2 against AD (equal counts): **AP 3/5 at both D₀ ends, not a win**; GM 5/5 at both ends.

**Features of the held-out numbers:**
- **AP1's AD held-out value is one direction.** AD fitted on the 20 % half scores 2916 m² on the other half (39.7 the
  other way). The mean, 1478, is a property of the lopsided split as much as of AD.
- **P2 against AD flips between in-sample and held-out.** On AP1, AP3 and AP5, P2 loses to AD in-sample (0/5) but wins
  held-out. AD's held-out MSE rises more over its in-sample MSE than P2's does. So whether P2 "wins" there depends on
  which test is used.
- **A_crit moves with the fitting half.** The A_crit chosen by the two halves differs by up to a factor 5 (GM1: 5798 and
  1145; GM3: 328 and 1098; B5b). The held-out P4 values are 1.3-2.3 × its in-sample MSE.
- **AP4 is the one tile where P4 loses out of sample.** It also has the only exactly equal AP split, and the smallest
  in-sample margin of P4 over AD (35.2 against 45.7).

## 3. Per-tile m: m_t against the measured θ and the site m

**m_t** is the minimum of each tile's own profile MSE*(m) from the 4b/5 m-profile runs (grid 0.05-0.90; B6):

| tile | AP1 | AP2 | AP3 | AP4 | AP5 | GM1 | GM2 | GM3 | GM4 | GM5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| m_t | 0.54 | 0.59 | 0.39 | 0.21 | 0.42 | 0.29 | 0.29 | 0.29 | 0.36 | 0.34 |
| 1 % interval | 0.53-0.55 | 0.56-0.62 | 0.36-0.41 | 0.09-0.28 | 0.41-0.44 | 0.27-0.30 | 0.26-0.33 | 0.27-0.31 | 0.34-0.38 | 0.32-0.35 |
| 5 % interval | 0.49-0.57 | 0.47-0.67 | 0.31-0.45 | **0.05**-0.38 | 0.38-0.47 | 0.23-0.33 | 0.22-0.37 | 0.24-0.34 | 0.30-0.40 | 0.30-0.38 |

**Grid edges.** No m_t is on a grid edge. AP4 (0.21) is interior, but its 5 % interval reaches the lower edge (0.05), so
AP4 bounds m from above only. Every interval is a single run of grid points.

**Where the site m falls in the tile intervals:**
- **AP:** the site m (0.47) is inside the 5 % interval of AP2 and AP5 only, and inside no AP tile's 1 % interval.
- **GM:** the site m (0.31) is inside the 1 % interval of GM2 and GM3 and the 5 % interval of every GM tile.

**Against the measured site-pooled θ** (addendum 4 §6.4 ladder rerun, run `20261110T160633Z_measure_audit_10tiles`;
95 % basin-bootstrap bands; B6):

| site | A_c | pooled | equal-weight | site m inside (pooled / equal) | tiles with m_t inside (pooled / equal) |
| --- | --- | --- | --- | --- | --- |
| AP | 10³ m² | 0.39 [0.37, 0.41] | 0.29 [0.21, 0.34] | no / no | AP3 / AP4 (at the band's lower edge, 0.21) |
| AP | 10⁴ m² | 0.52 [0.47, 0.57] | 0.53 [0.45, 0.59] | yes / yes | AP1 / AP1, AP2 |
| GM | 10³ m² | 0.33 [0.31, 0.35] | 0.30 [0.24, 0.34] | yes / yes | GM5 / GM1, GM2, GM3, GM5 |
| GM | 10⁴ m² | 0.31 [0.28, 0.34] | 0.29 [0.26, 0.37] | yes / yes | GM1, GM2, GM3, GM5 / all five |

**Reading the AP table:**
- **The measured θ cannot locate the AP site m.** The AP θ estimate depends on the network threshold, rising from 0.39
  (pooled) at 10³ m² to 0.52 at 10⁴ m². The site m sits between the two estimates. At 10⁴ m² the AP bands rest on 16
  basins.
- **AP5 lies in no band.** Its m_t (0.42) is inside none of the four.
- **The AP tiles span 0.21-0.59.** No single measured band covers that range. Only the 10³ m² equal-weight band
  (0.21-0.34) and the 10⁴ m² equal-weight band (0.45-0.59) together reach both ends, and AP5 (0.42) falls between them.

**k_s and K₀ at θ = m_t (1.2.1; B7).**
- The Phase 6 estimator is used: pooled through-origin χ-z above 10³ m², the same basins.
- The check passes: recomputing at the site m gives the config k_s exactly on 10 of 10 tiles.
- K₀(m_t)/K₀(site m): AP1 0.51, AP2 0.32, AP3 2.3, AP4 15, AP5 1.7; GM 0.56-1.26.
- K's units depend on m, so only K/K₀ at the same m is compared below.
- n_eff is 1.1-1.8 on AP and GM4-GM5, and 2.6-3.6 on GM1-GM3. The AP k_s values rest on one to two effective basins, as
  at the site m.

## 4. Results at m_t against the site m, with the rule applied

Linear partitioned joint (thin 2000 + staircase + every unique area in the 5 % intervals, as in Phase 6) and AD (profile
over r + Brent), both at (m_t, K₀(m_t)). Runs: step `a12_mt` (B8, B11).

| tile | m | A_crit | K/K₀ | D/D₀,A | D/D₀,B | MSE part. | MSE AD | AD/part. | AD K/K₀ | AD D/D₀,A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 0.47 | 6889 | 0.90 | 4.74 | 2.44 | 22.38 | 59.02 | 2.64 | 9.0 | 0.152 |
| AP1 | 0.54 | 4932 | 0.77 | 4.69 | 2.41 | 20.52 | 40.12 | 1.96 | 13.4 | 0.384 |
| AP2 | 0.47 | 3121 | 0.97 | 5.22 | 4.12 | 22.46 | 76.49 | 3.41 | 9.0 | 0.172 |
| AP2 | 0.59 | 7004 | 0.78 | 6.95 | 5.49 | 21.35 | 35.88 | 1.68 | 15.2 | 0.667 |
| AP3 | 0.47 | 4742 | 1.01 | 8.31 | 8.31 | 42.12 | 152.81 | 3.63 | 9.6 | 0.234 |
| AP3 | 0.39 | 4782 | 1.16 | 8.18 | 8.18 | 39.62 | 138.83 | 3.50 | 4.3 | 0.033 |
| AP4 | 0.47 | 7232 | 0.98 | 5.44 | 2.56 | 35.16 | 45.72 | 1.30 | 10.2 | 0.281 |
| AP4 | 0.21 | 7875 | 1.41 | 5.49 | 2.58 | 31.53 | 66.73 | 2.12 | 1.3 | 0.0025 |
| AP5 | 0.47 | 1887 | 1.00 | 4.31 | 3.64 | 27.87 | 69.81 | 2.50 | 8.6 | 0.196 |
| AP5 | 0.42 | 1887 | 1.04 | 4.15 | 3.51 | 26.61 | 88.68 | 3.33 | 5.8 | 0.076 |
| GM1 | 0.31 | 1241 | 1.00 | 2.16 | 2.08 | 24.69 | 48.96 | 1.98 | 3.1 | 0.033 |
| GM1 | 0.29 | 1241 | 1.00 | 2.15 | 2.07 | 24.38 | 48.92 | 2.01 | 2.5 | 0.021 |
| GM2 | 0.31 | 3509 | 1.00 | 2.81 | 2.70 | 26.07 | 82.55 | 3.17 | 2.4 | 0.009 |
| GM2 | 0.29 | 3517 | 1.01 | 2.80 | 2.70 | 25.99 | 76.19 | 2.93 | 1.9 | 0.005 |
| GM3 | 0.31 | 860 | 1.06 | 1.90 | 1.90 | 33.03 | 64.72 | 1.96 | 2.1 | 0.004 |
| GM3 | 0.29 | 860 | 1.06 | 1.88 | 1.88 | 32.80 | 59.63 | 1.82 | 1.9 | 0.004 (D at its lower bound) |
| GM4 | 0.31 | 3853 | 1.05 | 2.22 | 2.32 | 40.74 | 174.48 | 4.28 | 2.7 | 0.006 |
| GM4 | 0.36 | 2753 | 1.01 | 2.18 | 2.28 | 39.28 | 182.33 | 4.64 | 4.5 | 0.022 |
| GM5 | 0.31 | 1885 | 1.04 | 2.51 | 2.54 | 32.21 | 93.32 | 2.90 | 3.1 | 0.030 |
| GM5 | 0.34 | 2259 | 1.04 | 2.69 | 2.72 | 31.62 | 91.65 | 2.90 | 4.1 | 0.051 |

**The rule (1.4).**

| site | partitioned < AD at m_t | median D/D₀,A, site m → m_t | median AD K/K₀, site m → m_t | immaterial? |
| --- | --- | --- | --- | --- |
| AP | 5/5 | 5.22 → 5.49 (+5 %) | 9.02 → 5.77 (**−36 %**) | **no** |
| GM | 5/5 | 2.22 → 2.18 (−2 %) | 2.74 → 2.50 (−9 %) | yes |

**What changes on AP, and where.** The AD K/K₀ condition fails.
- At m_t, AD K/K₀ moves:
  - upward on AP1 and AP2 (9.0 → 13.4, 15.2), whose m_t is above the site m;
  - downward on AP3, AP4 and AP5 (9.6 → 4.3, 10.2 → 1.3, 8.6 → 5.8), whose m_t is below it.
- So the statement "AD needs K about 9 × the measured K₀ on AP", with its AP spread of 8.6-10.2, holds only at the site
  m. At each tile's own m it ranges 1.3-15. On AP4 at m = 0.21, AD's K is 1.3 × K₀.
- AD's D/D₀,A moves the same way: 0.15-0.28 at the site m, 0.0025-0.67 at m_t.

**What does not change on AP:**
- The partitioned model fits better than AD on 5/5 tiles at m_t. The AD/partitioned ratio is 1.7-3.5, against 1.3-3.6 at
  the site m.
- Its D/D₀ is unchanged: 4.1-8.2 × D₀,A at m_t against 4.3-8.3 at the site m; at most +33 % on one tile (AP2).
- Its K/K₀ is 0.77-1.41 at m_t, against 0.90-1.01 at the site m. K recovery is weaker at m_t on AP1, AP2 (0.77, 0.78) and
  AP4 (1.41).
- A_crit moves by more than 25 % on AP1 (6889 → 4932) and AP2 (3121 → 7004).

**GM:** the site medians move by less than 10 %. Per tile, three quantities move by more than 25 %:
- AD K/K₀ on GM4 (2.7 → 4.5) and GM5 (3.1 → 4.1);
- AD D, which was already 0.004-0.03 × D₀;
- GM4's A_crit, which falls 29 % (3853 → 2753).

GM3's AD D at m_t sits on its lower bound.

## 5. What the evidence supports, and where it is weak

**Supported.**
1. **On GM the partitioned model's advantage is not a parameter-count effect.**
   - It wins at every equal count: P1 against A1 and A2, P2 and P3 against AD, 5/5 each, at either D₀.
   - It wins out of sample: P4 5/5, P2 5/5.
   - It does not depend on the site-m choice (1.4 passes).
2. **On AP the advantage at equal counts requires a free D.**
   - With D free (P3, K = K₀) the partitioned model beats AD on 5/5 tiles.
   - With D held at the measured D₀ (either end of the bracket) it does not: P2 0/5; P1 0/5 at D₀,A and 2/5 at D₀,B.
   - What the AP comparison shows is that a partitioned model with D ≈ 4-8 × D₀,A fits better than AD. Neither model fits
     well with the measured D₀: on AP, A2 gives 86-184 m² and P2 63-156 m², against 22-42 m² for P4.
3. **The partitioned model's K is the measured K₀.** P3 is within 0-15 % of the P4 MSE, and the partitioned K/K₀ stays at
   0.77-1.41 at either m. Fixing K at K₀ costs AD a factor 9-110 in MSE (A1 against AD).
4. **The concavity choice does not move the partitioned conclusions** at either site: MSE below AD 5/5, D/D₀ unchanged
   within 33 %. It does move the AP AD parameters (1.4 fails on AD K/K₀).

**Weak.**
1. **The held-out split is lopsided on 6 of 10 tiles**, unavoidably at basin granularity.
   - On AP1, AP3 and GM5 one direction fits on about 20 % of the cells.
   - AP1's AD held-out value (1478 m²) comes from one such direction (2916 against 39.7).
   - On AP the held-out rule passes by one tile (4/5). AP4, the one exactly equal AP split, is the failure.
   - The out-of-sample result on AP is therefore not robust, and P2's held-out 3/5 on AP is the opposite of its 0/5
     in-sample.
2. **The fit-half A_crit is poorly constrained.** It moves by up to a factor 5 between halves (GM1, GM3, AP3), consistent
   with the flat A_crit profiles reported since Phase 4.
3. **The AP measured θ depends on the network threshold**: 0.39 at 10³ m², 0.52 at 10⁴ m², with few basins at 10⁴ (16).
   - It cannot arbitrate between the site m and m_t on AP.
   - AP4 constrains m from above only.
   - The AP k_s and K₀ at any m rest on one to two effective basins (n_eff 1.1-1.8).
4. **The AP 1 % fills of §2 were capped at 1500 values** (3807-7054 in the intervals). Thinned-grid and final minima
   differ by ≤ 0.3 %, smaller than every margin the rules turn on except AP4's held-out (80.6 against 72.5, 11 %).
5. **A1 is degenerate**: with K = K₀, AD sends D to its bound on 4 tiles. Count 1 on AP is decided by A2, not A1.

## 6. Decisions needed

1. **The split (2.2).** Accept the greedy basin split with its shares as the held-out test, or replace it for the 6
   tiles where one basin exceeds half. Options: sub-basins at a higher network threshold inside the dominant basin, or
   spatial blocks. A replacement changes what "held-out" tests: spatial blocks are not hydrologically independent. I
   recommend reporting the current result with its shares and not rerunning, since GM passes 5/5 regardless and the AP
   answer is marginal under any split.
2. **How to state the parameter-count result for AP.** My recommendation: "at equal parameter counts the partitioned
   model beats AD on AP only when D is free (and then D ≈ 4-8 × D₀,A)". Do not write "the partitioned model beats AD at
   equal counts" for AP.
3. **The AD K/K₀ statement on AP.** It fails the 1.4 rule. Either drop the AP AD K/K₀ numbers from the claims, or state
   them as conditional on the site m (range 1.3-15 across per-tile m).
4. **The cost-rule reading** (section 1, point 6): confirm that "total ≤ 300" means expected core-hours (234; used 165;
   limits 380).
5. **Count 2 wording** (section 1, point 4): whether "wins at count 2" should require both P2 and P3, either one, or name
   them separately (as here).
6. **Merge.** Nothing is merged or tagged. Push only (section 4 of the addendum).

## 7. Appendices (generated tables)

Generated by `reports/v2/scripts/make_sensitivity_tables.py` from the runs named in each table.

## B1. Section 3: the core-hour estimate

Run `20261116T223818Z_a12_estimate_10tiles`; expected core-hours from the Phase 6 seconds per evaluation of the same computations.

| tile | n_unique | s_per_part_eval | s_per_ad_eval | n_joint_5pct_at_m_t | equal_part_h | equal_ad_h | mt_part_h | mt_ad_h |
|---|---|---|---|---|---|---|---|---|
| AP1 | 36295 | 6.01 | 31.2 | 1123 | 6.39 | 3.03 | 16.9 | 1.8 |
| AP2 | 43784 | 6.01 | 29.4 | 1294 | 6.5 | 2.86 | 21.4 | 1.7 |
| AP3 | 38560 | 5.49 | 31.7 | 1160 | 5.86 | 3.08 | 15.3 | 1.83 |
| AP4 | 38816 | 5.93 | 31.3 | 1015 | 6.33 | 3.04 | 19.6 | 1.81 |
| AP5 | 37158 | 5.81 | 31.3 | 1881 | 6.2 | 3.04 | 22.5 | 1.81 |
| GM1 | 6138 | 5.82 | 29.5 | 708 | 5.74 | 2.87 | 5.28 | 1.7 |
| GM2 | 6701 | 6.46 | 30 | 144 | 6.38 | 2.92 | 4.96 | 1.73 |
| GM3 | 8131 | 6.39 | 28.3 | 723 | 6.34 | 2.75 | 5.95 | 1.63 |
| GM4 | 6969 | 6.29 | 29.5 | 763 | 6.21 | 2.87 | 6.06 | 1.71 |
| GM5 | 7826 | 5.78 | 27.6 | 1509 | 5.73 | 2.69 | 6.56 | 1.6 |

Totals (expected core-hours): equal_part_h 61.7, equal_ad_h 29.1, mt_part_h 124.5, mt_ad_h 17.3, inputs_h 1.0, summary_h 0.5, total_h 234.2

## B2. Section 2.2: the greedy basin split (edge-outlet basins of the 10³ m² network)

| tile | n_basins_edge | n_interior_outlets | n_basins_half1 | n_basins_half2 | n_cells_half1 | n_cells_half2 | share_half1 | share_half2 | frac_valid_in_split | largest_basin_share |
|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | 82 | 10 | 1 | 81 | 627284 | 157303 | 0.7995 | 0.2005 | 0.7616 | 0.7995 |
| AP2 | 112 | 0 | 1 | 111 | 600005 | 399606 | 0.6002 | 0.3998 | 0.9703 | 0.6002 |
| AP3 | 113 | 0 | 1 | 112 | 762198 | 177284 | 0.8113 | 0.1887 | 0.9128 | 0.8113 |
| AP4 | 93 | 0 | 63 | 30 | 475197 | 475206 | 0.5 | 0.5 | 0.9225 | 0.4497 |
| AP5 | 70 | 0 | 1 | 69 | 618870 | 321523 | 0.6581 | 0.3419 | 0.9128 | 0.6581 |
| GM1 | 43 | 0 | 22 | 21 | 445194 | 445193 | 0.5 | 0.5 | 0.8904 | 0.2707 |
| GM2 | 48 | 0 | 10 | 38 | 458861 | 458656 | 0.5001 | 0.4999 | 0.9175 | 0.2669 |
| GM3 | 43 | 0 | 24 | 19 | 468321 | 468342 | 0.5 | 0.5 | 0.9367 | 0.3594 |
| GM4 | 49 | 0 | 1 | 48 | 556552 | 360450 | 0.6069 | 0.3931 | 0.917 | 0.6069 |
| GM5 | 53 | 0 | 1 | 52 | 727994 | 202558 | 0.7823 | 0.2177 | 0.9306 | 0.7823 |

## B3. Section 2.1: in-sample MSE (m²) of every variant at the site m

Partitioned variants: minimum over the union of the Phase 6 refined A_crit rows (stored Gram terms) and this run's rows; `P*_mse_new_rows` = the minimum over this run's rows only (thinned grid + staircase + 1 % fill).

| tile | P1_A_mse | P1_B_mse | P2_A_mse | P2_B_mse | P3_mse | P4_mse | A1_mse | A2_A_mse | A2_B_mse | AD_mse |
|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | 289.23 | 86.402 | 137.3 | 67.396 | 25.721 | 22.384 | 6510.7 | 127.99 | 160.74 | 59.018 |
| AP2 | 245.75 | 185.66 | 113.11 | 92.27 | 22.804 | 22.461 | 867.48 | 130.02 | 144.21 | 76.493 |
| AP3 | 340.57 | 340.57 | 156.11 | 156.11 | 42.155 | 42.122 | 4532.7 | 183.54 | 183.54 | 152.81 |
| AP4 | 255.23 | 81.075 | 119.15 | 62.963 | 35.209 | 35.156 | 415.1 | 86.268 | 137.35 | 45.717 |
| AP5 | 226.29 | 190.48 | 94.903 | 85.574 | 27.87 | 27.869 | 3954.3 | 110.44 | 118.39 | 69.808 |
| GM1 | 51.29 | 48.447 | 38.635 | 37.308 | 24.694 | 24.69 | 483.33 | 171.4 | 172.88 | 48.963 |
| GM2 | 57.32 | 55.357 | 48.553 | 47.611 | 26.071 | 26.071 | 5118.8 | 283.95 | 285.92 | 82.546 |
| GM3 | 61.265 | 61.265 | 38.98 | 38.98 | 34.334 | 33.027 | 4824.8 | 120.49 | 120.49 | 64.723 |
| GM4 | 113.8 | 122.92 | 72.169 | 74.257 | 42.307 | 40.743 | 1.3661e+04 | 579.14 | 575.05 | 174.48 |
| GM5 | 92.792 | 93.966 | 62.465 | 62.76 | 33.571 | 32.211 | 8587.9 | 273.39 | 272.63 | 93.319 |

### B3b. Parameters of each variant's optimum

| tile | variant | A_crit | K | D | mse | mse_new_rows | K_at_bound | D_at_bound |
|---|---|---|---|---|---|---|---|---|
| AP1 | P1_A | 823.3 | 4.428e-06 | 0.003587 | 289.2 | 289.2 | - | - |
| AP1 | P1_B | 1090 | 4.428e-06 | 0.006978 | 86.4 | 86.4 | - | - |
| AP1 | P2_A | 461.1 | 5.745e-06 | 0.003587 | 137.3 | 137.3 | - | - |
| AP1 | P2_B | 1085 | 4.911e-06 | 0.006978 | 67.4 | 67.4 | - | - |
| AP1 | P3 | 7011 | 4.428e-06 | 0.01536 | 25.72 | 25.72 | - | - |
| AP1 | P4 | 6889 | 3.995e-06 | 0.01699 | 22.38 | 22.38 | - | - |
| AP1 | A1 | - | 4.428e-06 | 1.368e-09 | 6511 | - | no | yes |
| AP1 | A2_A | - | 7.076e-05 | 0.003587 | 128 | - | no | no |
| AP1 | A2_B | - | 8.909e-05 | 0.006978 | 160.7 | - | no | no |
| AP1 | AD | - | 3.995e-05 | 5.447e-04 | 59.02 | - | no | no |
| AP2 | P1_A | 495.3 | 4.748e-06 | 0.003218 | 245.8 | 245.8 | - | - |
| AP2 | P1_B | 682.6 | 4.748e-06 | 0.004078 | 185.7 | 185.7 | - | - |
| AP2 | P2_A | 464.9 | 6.071e-06 | 0.003218 | 113.1 | 113.1 | - | - |
| AP2 | P2_B | 495.3 | 5.881e-06 | 0.004078 | 92.27 | 92.27 | - | - |
| AP2 | P3 | 3121 | 4.748e-06 | 0.01615 | 22.8 | 22.8 | - | - |
| AP2 | P4 | 3121 | 4.597e-06 | 0.0168 | 22.46 | 22.46 | - | - |
| AP2 | A1 | - | 4.748e-06 | 0.1532 | 867.5 | - | no | no |
| AP2 | A2_A | - | 6.539e-05 | 0.003218 | 130 | - | no | no |
| AP2 | A2_B | - | 6.994e-05 | 0.004078 | 144.2 | - | no | no |
| AP2 | AD | - | 4.264e-05 | 5.534e-04 | 76.49 | - | no | no |
| AP3 | P1_A | 354.6 | 4.770e-06 | 0.002155 | 340.6 | 340.6 | - | - |
| AP3 | P1_B | 354.6 | 4.770e-06 | 0.002155 | 340.6 | 340.6 | - | - |
| AP3 | P2_A | 294.7 | 6.359e-06 | 0.002155 | 156.1 | 156.1 | - | - |
| AP3 | P2_B | 294.7 | 6.359e-06 | 0.002155 | 156.1 | 156.1 | - | - |
| AP3 | P3 | 4742 | 4.770e-06 | 0.01804 | 42.15 | 42.15 | - | - |
| AP3 | P4 | 4742 | 4.825e-06 | 0.01791 | 42.12 | 42.12 | - | - |
| AP3 | A1 | - | 4.770e-06 | 1.473e-09 | 4533 | - | no | yes |
| AP3 | A2_A | - | 6.626e-05 | 0.002155 | 183.5 | - | no | no |
| AP3 | A2_B | - | 6.626e-05 | 0.002155 | 183.5 | - | no | no |
| AP3 | AD | - | 4.585e-05 | 5.032e-04 | 152.8 | - | no | no |
| AP4 | P1_A | 536.2 | 4.610e-06 | 0.003629 | 255.2 | 255.2 | - | - |
| AP4 | P1_B | 983.9 | 4.610e-06 | 0.00772 | 81.07 | 81.07 | - | - |
| AP4 | P2_A | 414.5 | 5.966e-06 | 0.003629 | 119.1 | 119.1 | - | - |
| AP4 | P2_B | 995.4 | 5.127e-06 | 0.00772 | 62.96 | 62.96 | - | - |
| AP4 | P3 | 7232 | 4.610e-06 | 0.01951 | 35.21 | 35.21 | - | - |
| AP4 | P4 | 7232 | 4.522e-06 | 0.01974 | 35.16 | 35.16 | - | - |
| AP4 | A1 | - | 4.610e-06 | 0.1488 | 415.1 | - | no | no |
| AP4 | A2_A | - | 6.634e-05 | 0.003629 | 86.27 | - | no | no |
| AP4 | A2_B | - | 8.375e-05 | 0.00772 | 137.3 | - | no | no |
| AP4 | AD | - | 4.710e-05 | 0.001019 | 45.72 | - | no | no |
| AP5 | P1_A | 378.4 | 5.128e-06 | 0.002921 | 226.3 | 226.3 | - | - |
| AP5 | P1_B | 442.1 | 5.128e-06 | 0.003456 | 190.5 | 190.5 | - | - |
| AP5 | P2_A | 329.9 | 6.619e-06 | 0.002921 | 94.9 | 94.9 | - | - |
| AP5 | P2_B | 334.6 | 6.496e-06 | 0.003456 | 85.57 | 85.57 | - | - |
| AP5 | P3 | 1887 | 5.128e-06 | 0.01256 | 27.87 | 27.87 | - | - |
| AP5 | P4 | 1887 | 5.119e-06 | 0.01258 | 27.87 | 27.87 | - | - |
| AP5 | A1 | - | 5.128e-06 | 1.584e-09 | 3954 | - | no | yes |
| AP5 | A2_A | - | 6.795e-05 | 0.002921 | 110.4 | - | no | no |
| AP5 | A2_B | - | 7.140e-05 | 0.003456 | 118.4 | - | no | no |
| AP5 | AD | - | 4.421e-05 | 5.728e-04 | 69.81 | - | no | no |
| GM1 | P1_A | 256 | 4.854e-05 | 0.009001 | 51.29 | 51.29 | - | - |
| GM1 | P1_B | 259 | 4.854e-05 | 0.009354 | 48.45 | 48.45 | - | - |
| GM1 | P2_A | 257 | 5.198e-05 | 0.009001 | 38.63 | 38.63 | - | - |
| GM1 | P2_B | 260 | 5.176e-05 | 0.009354 | 37.31 | 37.31 | - | - |
| GM1 | P3 | 1241 | 4.854e-05 | 0.01957 | 24.69 | 24.69 | - | - |
| GM1 | P4 | 1241 | 4.865e-05 | 0.01948 | 24.69 | 24.69 | - | - |
| GM1 | A1 | - | 4.854e-05 | 4.298e-10 | 483.3 | - | no | yes |
| GM1 | A2_A | - | 4.725e-04 | 0.009001 | 171.4 | - | no | no |
| GM1 | A2_B | - | 4.795e-04 | 0.009354 | 172.9 | - | no | no |
| GM1 | AD | - | 1.483e-04 | 2.947e-04 | 48.96 | - | no | no |
| GM2 | P1_A | 204 | 5.514e-05 | 0.00977 | 57.32 | 57.32 | - | - |
| GM2 | P1_B | 210 | 5.514e-05 | 0.01015 | 55.36 | 55.36 | - | - |
| GM2 | P2_A | 210 | 5.896e-05 | 0.00977 | 48.55 | 48.55 | - | - |
| GM2 | P2_B | 211 | 5.872e-05 | 0.01015 | 47.61 | 47.61 | - | - |
| GM2 | P3 | 3509 | 5.514e-05 | 0.02743 | 26.07 | 26.07 | - | - |
| GM2 | P4 | 3509 | 5.515e-05 | 0.02743 | 26.07 | 26.07 | - | - |
| GM2 | A1 | - | 5.514e-05 | 4.201e-05 | 5119 | - | no | no |
| GM2 | A2_A | - | 4.775e-04 | 0.00977 | 284 | - | no | no |
| GM2 | A2_B | - | 4.838e-04 | 0.01015 | 285.9 | - | no | no |
| GM2 | AD | - | 1.309e-04 | 8.780e-05 | 82.55 | - | no | no |
| GM3 | P1_A | 231 | 6.413e-05 | 0.01012 | 61.27 | 61.27 | - | - |
| GM3 | P1_B | 231 | 6.413e-05 | 0.01012 | 61.27 | 61.27 | - | - |
| GM3 | P2_A | 278 | 7.294e-05 | 0.01012 | 38.98 | 38.98 | - | - |
| GM3 | P2_B | 278 | 7.294e-05 | 0.01012 | 38.98 | 38.98 | - | - |
| GM3 | P3 | 1079 | 6.413e-05 | 0.02395 | 34.33 | 34.33 | - | - |
| GM3 | P4 | 860 | 6.810e-05 | 0.01923 | 33.03 | 33.03 | - | - |
| GM3 | A1 | - | 6.413e-05 | 4.102e-05 | 4825 | - | no | no |
| GM3 | A2_A | - | 5.860e-04 | 0.01012 | 120.5 | - | no | no |
| GM3 | A2_B | - | 5.860e-04 | 0.01012 | 120.5 | - | no | no |
| GM3 | AD | - | 1.356e-04 | 4.091e-05 | 64.72 | - | no | no |
| GM4 | P1_A | 423 | 3.479e-05 | 0.009855 | 113.8 | 113.8 | - | - |
| GM4 | P1_B | 422 | 3.479e-05 | 0.009435 | 122.9 | 122.9 | - | - |
| GM4 | P2_A | 450 | 3.911e-05 | 0.009855 | 72.17 | 72.17 | - | - |
| GM4 | P2_B | 450 | 3.955e-05 | 0.009435 | 74.26 | 74.26 | - | - |
| GM4 | P3 | 3858 | 3.479e-05 | 0.02313 | 42.31 | 42.31 | - | - |
| GM4 | P4 | 3853 | 3.640e-05 | 0.02189 | 40.74 | 40.74 | - | - |
| GM4 | A1 | - | 3.479e-05 | 0.04869 | 1.366e+04 | - | no | no |
| GM4 | A2_A | - | 3.985e-04 | 0.009855 | 579.1 | - | no | no |
| GM4 | A2_B | - | 3.923e-04 | 0.009435 | 575.1 | - | no | no |
| GM4 | AD | - | 9.514e-05 | 6.291e-05 | 174.5 | - | no | no |
| GM5 | P1_A | 186 | 5.019e-05 | 0.007589 | 92.79 | 92.79 | - | - |
| GM5 | P1_B | 186 | 5.019e-05 | 0.007496 | 93.97 | 93.97 | - | - |
| GM5 | P2_A | 158 | 5.532e-05 | 0.007589 | 62.46 | 62.46 | - | - |
| GM5 | P2_B | 158 | 5.537e-05 | 0.007496 | 62.76 | 62.76 | - | - |
| GM5 | P3 | 1885 | 5.019e-05 | 0.02059 | 33.57 | 33.57 | - | - |
| GM5 | P4 | 1885 | 5.234e-05 | 0.01903 | 32.21 | 32.21 | - | - |
| GM5 | A1 | - | 5.019e-05 | 4.173e-05 | 8588 | - | no | no |
| GM5 | A2_A | - | 4.948e-04 | 0.007589 | 273.4 | - | no | no |
| GM5 | A2_B | - | 4.926e-04 | 0.007496 | 272.6 | - | no | no |
| GM5 | AD | - | 1.540e-04 | 2.288e-04 | 93.32 | - | no | no |

## B4. Section 2.3: equal-count comparisons per tile (True = the partitioned variant is lower than every AD variant of its count)

| tile | count1_A_P1_wins | count1_B_P1_wins | count2_A_P2_wins | count2_B_P2_wins | count2_P3_wins | P4_beats_AD |
|---|---|---|---|---|---|---|
| AP1 | no | yes | no | no | yes | yes |
| AP2 | no | no | no | no | yes | yes |
| AP3 | no | no | no | no | yes | yes |
| AP4 | no | yes | no | no | yes | yes |
| AP5 | no | no | no | no | yes | yes |
| GM1 | yes | yes | yes | yes | yes | yes |
| GM2 | yes | yes | yes | yes | yes | yes |
| GM3 | yes | yes | yes | yes | yes | yes |
| GM4 | yes | yes | yes | yes | yes | yes |
| GM5 | yes | yes | yes | yes | yes | yes |

Per site (rule: >= 4 of 5 tiles):

| site | comparison | tiles | wins |
|---|---|---|---|
| AP | count1_A_P1_wins | 0 of 5 | no |
| AP | count2_A_P2_wins | 0 of 5 | no |
| AP | count1_B_P1_wins | 2 of 5 | no |
| AP | count2_B_P2_wins | 0 of 5 | no |
| AP | count2_P3_wins | 5 of 5 | yes |
| AP | P4_beats_AD | 5 of 5 | yes |
| AP | heldout_P4_beats_AD | 4 of 5 | yes |
| AP | heldout_P2_A_beats_AD | 3 of 5 | no |
| AP | heldout_P2_B_beats_AD | 3 of 5 | no |
| GM | count1_A_P1_wins | 5 of 5 | yes |
| GM | count2_A_P2_wins | 5 of 5 | yes |
| GM | count1_B_P1_wins | 5 of 5 | yes |
| GM | count2_B_P2_wins | 5 of 5 | yes |
| GM | count2_P3_wins | 5 of 5 | yes |
| GM | P4_beats_AD | 5 of 5 | yes |
| GM | heldout_P4_beats_AD | 5 of 5 | yes |
| GM | heldout_P2_A_beats_AD | 5 of 5 | yes |
| GM | heldout_P2_B_beats_AD | 5 of 5 | yes |

## B5. Section 2.2: held-out MSE (m²): fit on one half, score on the other, both ways, averaged

| tile | P4_heldout | P2_A_heldout | P2_B_heldout | AD_heldout | P4_beats_AD | P2_A_beats_AD | P2_B_beats_AD |
|---|---|---|---|---|---|---|---|
| AP1 | 36.234 | 114.77 | 57.404 | 1477.9 | yes | yes | yes |
| AP2 | 29.015 | 118.39 | 96.976 | 84.809 | yes | no | no |
| AP3 | 78.736 | 184.28 | 184.28 | 187.02 | yes | yes | yes |
| AP4 | 80.562 | 201.32 | 118.01 | 72.547 | no | no | no |
| AP5 | 62.377 | 109.01 | 100.58 | 124.66 | yes | yes | yes |
| GM1 | 35.667 | 46.45 | 45.571 | 67.75 | yes | yes | yes |
| GM2 | 40.314 | 52.23 | 52.729 | 131.64 | yes | yes | yes |
| GM3 | 41.567 | 41.962 | 41.962 | 69.388 | yes | yes | yes |
| GM4 | 61.612 | 80.614 | 82.864 | 193.03 | yes | yes | yes |
| GM5 | 61.156 | 78.046 | 78.695 | 167.7 | yes | yes | yes |

### B5b. The two directions and the A_crit chosen by each fitting half

| tile | P4_h1_fit_score_h2 | P4_h2_fit_score_h1 | P4_A_crit_h1 | P4_A_crit_h2 | AD_h1_fit_score_h2 | AD_h2_fit_score_h1 | P2_A_h1_fit_score_h2 | P2_A_h2_fit_score_h1 | P2_A_A_crit_h1 | P2_A_A_crit_h2 | P2_B_h1_fit_score_h2 | P2_B_h2_fit_score_h1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | 28.335 | 44.133 | 5930.3 | 3014.6 | 39.735 | 2916 | 98.496 | 131.04 | 436.36 | 484.84 | 47.834 | 66.974 |
| AP2 | 30.094 | 27.936 | 7641.5 | 3283.6 | 62.632 | 106.99 | 130.76 | 106.02 | 464.88 | 465.83 | 104.73 | 89.222 |
| AP3 | 69.533 | 87.939 | 1.0300e+04 | 5487.3 | 164.09 | 209.96 | 121.95 | 246.61 | 334.64 | 208.2 | 121.95 | 246.61 |
| AP4 | 61.871 | 99.254 | 4100.2 | 7727.1 | 78.702 | 66.393 | 143.24 | 259.39 | 429.7 | 329.88 | 89.5 | 146.51 |
| AP5 | 38.103 | 86.652 | 1901.3 | 3856.9 | 87.661 | 161.66 | 118.31 | 99.701 | 310.87 | 343.19 | 106.45 | 94.715 |
| GM1 | 31.5 | 39.834 | 5798 | 1145 | 65.312 | 70.188 | 33.908 | 58.991 | 273 | 200 | 32.836 | 58.307 |
| GM2 | 33.115 | 47.513 | 3523 | 3358 | 168.79 | 94.485 | 35.735 | 68.726 | 211 | 192 | 38.269 | 67.189 |
| GM3 | 46.314 | 36.821 | 328 | 1098 | 85.546 | 53.23 | 50.546 | 33.378 | 326 | 231 | 50.546 | 33.378 |
| GM4 | 64.349 | 58.875 | 4368 | 3832 | 213.5 | 172.55 | 99.279 | 61.948 | 456 | 453 | 101.58 | 64.151 |
| GM5 | 51.694 | 70.618 | 1561 | 3104 | 195.9 | 139.5 | 88.912 | 67.18 | 158 | 262 | 89.02 | 68.37 |

## B6. Section 1.1: per-tile m_t, its intervals, and the measured site theta

m-profile runs of the 4b/5 checkpoint (grid 0.05-0.90, step 0.01; 200 A_crit values); inputs run `20261116T224609Z_a12_inputs_10tiles`.

| tile | m_site | m_t | m_t on grid edge | 1 % interval | 5 % interval | 5 % reaches edge | MSE* at m_t | A_crit at m_t | run |
|---|---|---|---|---|---|---|---|---|---|
| AP1 | 0.47 | 0.54 | no | 0.53-0.55 (1) | 0.49-0.57 (1) | no | 20.54 | 4894 | 20261110T160413Z_tile_run_AP1_v2_m_profile |
| AP2 | 0.47 | 0.59 | no | 0.56-0.62 (1) | 0.47-0.67 (1) | no | 21.47 | 6919 | 20261110T160413Z_tile_run_AP2_v2_m_profile |
| AP3 | 0.47 | 0.39 | no | 0.36-0.41 (1) | 0.31-0.45 (1) | no | 39.83 | 5158 | 20261110T160405Z_tile_run_AP3_v2_m_profile |
| AP4 | 0.47 | 0.21 | no | 0.09-0.28 (1) | 0.05-0.38 (1) | yes | 31.69 | 7831 | 20261110T160405Z_tile_run_AP4_v2_m_profile |
| AP5 | 0.47 | 0.42 | no | 0.41-0.44 (1) | 0.38-0.47 (1) | no | 26.69 | 1855 | 20261110T160405Z_tile_run_AP5_v2_m_profile |
| GM1 | 0.31 | 0.29 | no | 0.27-0.30 (1) | 0.23-0.33 (1) | no | 24.47 | 1135 | 20261110T160404Z_tile_run_GM1_v2_m_profile |
| GM2 | 0.31 | 0.29 | no | 0.26-0.33 (1) | 0.22-0.37 (1) | no | 26.2 | 3444 | 20261110T160420Z_tile_run_GM2_v2_m_profile |
| GM3 | 0.31 | 0.29 | no | 0.27-0.31 (1) | 0.24-0.34 (1) | no | 32.81 | 859 | 20261110T160420Z_tile_run_GM3_v2_m_profile |
| GM4 | 0.31 | 0.36 | no | 0.34-0.38 (1) | 0.30-0.40 (1) | no | 39.51 | 2718 | 20261110T160413Z_tile_run_GM4_v2_m_profile |
| GM5 | 0.31 | 0.34 | no | 0.32-0.35 (1) | 0.30-0.38 (1) | no | 31.65 | 2247 | 20261110T160416Z_tile_run_GM5_v2_m_profile |

Site-pooled theta (run `20261110T160633Z_measure_audit_10tiles`, 95 % basin-bootstrap band):

| site | A_c (m²) | estimator | theta | 95 % | basins |
|---|---|---|---|---|---|
| AP | 1000 | pooled | 0.39 | [0.37, 0.41] | 116 |
| AP | 1000 | equal | 0.29 | [0.21, 0.34] | 116 |
| AP | 10000 | pooled | 0.52 | [0.47, 0.57] | 16 |
| AP | 10000 | equal | 0.53 | [0.45, 0.59] | 16 |
| GM | 1000 | pooled | 0.33 | [0.31, 0.35] | 68 |
| GM | 1000 | equal | 0.3 | [0.24, 0.34] | 68 |
| GM | 10000 | pooled | 0.31 | [0.28, 0.34] | 25 |
| GM | 10000 | equal | 0.29 | [0.26, 0.37] | 25 |

Inside the band? (m_t / site m)

| tile | pooled_1000 | equal_1000 | pooled_10000 | equal_10000 |
|---|---|---|---|---|
| AP1 | no / no | no / no | yes / yes | yes / yes |
| AP2 | no / no | no / no | no / yes | yes / yes |
| AP3 | yes / no | no / no | no / yes | no / yes |
| AP4 | no / no | yes / no | no / yes | no / yes |
| AP5 | no / no | no / no | no / yes | no / yes |
| GM1 | no / yes | yes / yes | yes / yes | yes / yes |
| GM2 | no / yes | yes / yes | yes / yes | yes / yes |
| GM3 | no / yes | yes / yes | yes / yes | yes / yes |
| GM4 | no / yes | no / yes | no / yes | yes / yes |
| GM5 | yes / yes | yes / yes | yes / yes | yes / yes |

## B7. Section 1.2.1: k_s and K0 at theta = m_t (pooled through-origin chi-z above 10³ m²)

| tile | m_t | k_s 1e3 | 95 % | basins | n_eff | K0(m_t) | K0(site m) | k_s 1e4 (reported) | check: k_s at site m / config - 1 |
|---|---|---|---|---|---|---|---|---|---|
| AP1 | 0.54 | 24.5 | [21.29, 25.05] | 32 | 1.511 | 2.245e-06 | 4.428e-06 | 34.62 | 0 |
| AP2 | 0.59 | 36.14 | [20.31, 38.38] | 13 | 1.833 | 1.522e-06 | 4.748e-06 | 54.9 | 0 |
| AP3 | 0.39 | 4.943 | [4.557, 6.055] | 32 | 1.054 | 1.113e-05 | 4.770e-06 | 4.6 | 0 |
| AP4 | 0.21 | 0.7882 | [0.638, 1.116] | 30 | 1.686 | 6.978e-05 | 4.610e-06 | 0.5847 | 0 |
| AP5 | 0.42 | 6.346 | [5.906, 7.566] | 9 | 1.57 | 8.667e-06 | 5.128e-06 | 6.367 | 0 |
| GM1 | 0.29 | 2.411 | [2.289, 2.505] | 19 | 3.472 | 6.096e-05 | 4.854e-05 | 2.32 | 0 |
| GM2 | 0.29 | 2.134 | [1.958, 2.436] | 16 | 3.618 | 6.888e-05 | 5.514e-05 | 1.923 | 0 |
| GM3 | 0.29 | 1.842 | [1.742, 2.215] | 15 | 2.646 | 7.981e-05 | 6.413e-05 | 1.757 | 0 |
| GM4 | 0.36 | 7.564 | [6.933, 7.653] | 14 | 1.51 | 1.943e-05 | 3.479e-05 | 7.586 | 0 |
| GM5 | 0.34 | 4.183 | [3.906, 4.208] | 4 | 1.238 | 3.514e-05 | 5.019e-05 | 4.153 | 0 |

## B8. Sections 1.2-1.3: site m against m_t

| tile | which | m | K0 | A_crit | K_over_K0 | D_over_D0_A | D_over_D0_B | mse_partitioned | mse_AD | ratio_AD_over_part | AD_K_over_K0 | AD_D_over_D0_A | AD_D_over_D0_B | joint_where | AD_where |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | site | 0.47 | 4.428e-06 | 6889 | 0.9022 | 4.737 | 2.435 | 22.38 | 59.02 | 2.637 | 9.022 | 0.1519 | 0.07806 | interior | interior |
| AP1 | m_t | 0.54 | 2.245e-06 | 4932 | 0.7746 | 4.687 | 2.409 | 20.52 | 40.12 | 1.956 | 13.39 | 0.3844 | 0.1976 | interior | interior |
| AP2 | site | 0.47 | 4.748e-06 | 3121 | 0.9681 | 5.219 | 4.119 | 22.46 | 76.49 | 3.406 | 8.979 | 0.172 | 0.1357 | interior | interior |
| AP2 | m_t | 0.59 | 1.522e-06 | 7004 | 0.778 | 6.952 | 5.487 | 21.35 | 35.88 | 1.681 | 15.18 | 0.667 | 0.5264 | interior | interior |
| AP3 | site | 0.47 | 4.770e-06 | 4742 | 1.012 | 8.311 | 8.311 | 42.12 | 152.8 | 3.628 | 9.611 | 0.2335 | 0.2335 | interior | interior |
| AP3 | m_t | 0.39 | 1.113e-05 | 4782 | 1.155 | 8.18 | 8.18 | 39.62 | 138.8 | 3.504 | 4.254 | 0.03263 | 0.03263 | interior | interior |
| AP4 | site | 0.47 | 4.610e-06 | 7232 | 0.981 | 5.44 | 2.557 | 35.16 | 45.72 | 1.3 | 10.22 | 0.2809 | 0.132 | interior | interior |
| AP4 | m_t | 0.21 | 6.978e-05 | 7875 | 1.411 | 5.486 | 2.579 | 31.53 | 66.73 | 2.117 | 1.293 | 0.002489 | 0.00117 | interior | interior |
| AP5 | site | 0.47 | 5.128e-06 | 1887 | 0.9984 | 4.308 | 3.642 | 27.87 | 69.81 | 2.505 | 8.622 | 0.1961 | 0.1658 | interior | interior |
| AP5 | m_t | 0.42 | 8.667e-06 | 1887 | 1.035 | 4.147 | 3.505 | 26.61 | 88.68 | 3.333 | 5.771 | 0.07578 | 0.06405 | interior | interior |
| GM1 | site | 0.31 | 4.854e-05 | 1241 | 1.002 | 2.164 | 2.082 | 24.69 | 48.96 | 1.983 | 3.054 | 0.03274 | 0.03151 | interior | interior |
| GM1 | m_t | 0.29 | 6.096e-05 | 1241 | 1.002 | 2.149 | 2.068 | 24.38 | 48.92 | 2.006 | 2.502 | 0.02094 | 0.02015 | interior | interior |
| GM2 | site | 0.31 | 5.514e-05 | 3509 | 1 | 2.807 | 2.702 | 26.07 | 82.55 | 3.166 | 2.375 | 0.008987 | 0.008649 | interior | interior |
| GM2 | m_t | 0.29 | 6.888e-05 | 3517 | 1.008 | 2.8 | 2.695 | 25.99 | 76.19 | 2.931 | 1.949 | 0.005194 | 0.004999 | interior | interior |
| GM3 | site | 0.31 | 6.413e-05 | 860 | 1.062 | 1.901 | 1.901 | 33.03 | 64.72 | 1.96 | 2.115 | 0.004043 | 0.004043 | interior | interior |
| GM3 | m_t | 0.29 | 7.981e-05 | 860 | 1.059 | 1.876 | 1.876 | 32.8 | 59.63 | 1.818 | 1.941 | 0.003953 | 0.003953 | interior | y_hi |
| GM4 | site | 0.31 | 3.479e-05 | 3853 | 1.046 | 2.222 | 2.321 | 40.74 | 174.5 | 4.282 | 2.735 | 0.006384 | 0.006668 | interior | interior |
| GM4 | m_t | 0.36 | 1.943e-05 | 2753 | 1.006 | 2.179 | 2.276 | 39.28 | 182.3 | 4.642 | 4.51 | 0.02226 | 0.02325 | interior | interior |
| GM5 | site | 0.31 | 5.019e-05 | 1885 | 1.043 | 2.508 | 2.539 | 32.21 | 93.32 | 2.897 | 3.068 | 0.03015 | 0.03052 | interior | interior |
| GM5 | m_t | 0.34 | 3.514e-05 | 2259 | 1.037 | 2.687 | 2.72 | 31.62 | 91.65 | 2.898 | 4.094 | 0.05083 | 0.05146 | interior | interior |

Runs:

| tile | which | run_partitioned | run_ad |
|---|---|---|---|
| AP1 | site | 20261111T050726Z_tile_run_AP1_v2_partitioned-refined | 20261111T012439Z_tile_run_AP1_v2_ad |
| AP1 | m_t | 20261117T020518Z_a12_mt_AP1_v2_partitioned-refined | 20261116T225951Z_a12_mt_AP1_v2_ad |
| AP2 | site | 20261111T050734Z_tile_run_AP2_v2_partitioned-refined | 20261111T012443Z_tile_run_AP2_v2_ad |
| AP2 | m_t | 20261117T021246Z_a12_mt_AP2_v2_partitioned-refined | 20261116T225950Z_a12_mt_AP2_v2_ad |
| AP3 | site | 20261111T045020Z_tile_run_AP3_v2_partitioned-refined | 20261111T012443Z_tile_run_AP3_v2_ad |
| AP3 | m_t | 20261117T020238Z_a12_mt_AP3_v2_partitioned-refined | 20261116T225950Z_a12_mt_AP3_v2_ad |
| AP4 | site | 20261111T050505Z_tile_run_AP4_v2_partitioned-refined | 20261111T012445Z_tile_run_AP4_v2_ad |
| AP4 | m_t | 20261117T020841Z_a12_mt_AP4_v2_partitioned-refined | 20261116T225954Z_a12_mt_AP4_v2_ad |
| AP5 | site | 20261111T050104Z_tile_run_AP5_v2_partitioned-refined | 20261111T012444Z_tile_run_AP5_v2_ad |
| AP5 | m_t | 20261117T020132Z_a12_mt_AP5_v2_partitioned-refined | 20261116T225950Z_a12_mt_AP5_v2_ad |
| GM1 | site | 20261111T050111Z_tile_run_GM1_v2_partitioned-refined | 20261111T012445Z_tile_run_GM1_v2_ad |
| GM1 | m_t | 20261117T020430Z_a12_mt_GM1_v2_partitioned-refined | 20261116T225950Z_a12_mt_GM1_v2_ad |
| GM2 | site | 20261111T052240Z_tile_run_GM2_v2_partitioned-refined | 20261111T012445Z_tile_run_GM2_v2_ad |
| GM2 | m_t | 20261117T022340Z_a12_mt_GM2_v2_partitioned-refined | 20261116T230001Z_a12_mt_GM2_v2_ad |
| GM3 | site | 20261111T052022Z_tile_run_GM3_v2_partitioned-refined | 20261111T012445Z_tile_run_GM3_v2_ad |
| GM3 | m_t | 20261117T022224Z_a12_mt_GM3_v2_partitioned-refined | 20261116T230031Z_a12_mt_GM3_v2_ad |
| GM4 | site | 20261111T051659Z_tile_run_GM4_v2_partitioned-refined | 20261111T012439Z_tile_run_GM4_v2_ad |
| GM4 | m_t | 20261117T022343Z_a12_mt_GM4_v2_partitioned-refined | 20261116T230036Z_a12_mt_GM4_v2_ad |
| GM5 | site | 20261111T045953Z_tile_run_GM5_v2_partitioned-refined | 20261111T012438Z_tile_run_GM5_v2_ad |
| GM5 | m_t | 20261117T020655Z_a12_mt_GM5_v2_partitioned-refined | 20261116T230031Z_a12_mt_GM5_v2_ad |

## B9. Section 1.4: the rule per site

| site | median of | site m | m_t | relative change |
|---|---|---|---|---|
| AP | D_over_D0_A | 5.219 | 5.486 | 0.0511 |
| AP | AD_K_over_K0 | 9.022 | 5.771 | -0.3603 |
| AP | K_over_K0 | 0.981 | 1.035 | 0.05471 |
| AP | AD_D_over_D0_A | 0.1961 | 0.07578 | -0.6136 |
| AP | D_over_D0_B | 3.642 | 3.505 | -0.0375 |
| AP | partitioned MSE < AD MSE at m_t: 5 of 5 | - | - | - |
| AP | immaterial: False (MSE condition True, medians condition False) | - | - | - |
| GM | D_over_D0_A | 2.222 | 2.179 | -0.01923 |
| GM | AD_K_over_K0 | 2.735 | 2.502 | -0.08519 |
| GM | K_over_K0 | 1.043 | 1.008 | -0.03327 |
| GM | AD_D_over_D0_A | 0.008987 | 0.02094 | 1.33 |
| GM | D_over_D0_B | 2.321 | 2.276 | -0.01923 |
| GM | partitioned MSE < AD MSE at m_t: 5 of 5 | - | - | - |
| GM | immaterial: True (MSE condition True, medians condition True) | - | - | - |

## B10. Checks against Phase 6

| tile | phase6_check_part | phase6_check_ad |
|---|---|---|
| AP1 | 0 | 0 |
| AP2 | 0 | 0 |
| AP3 | 0 | 0 |
| AP4 | 0 | 0 |
| AP5 | 0 | 0 |
| GM1 | 3.52e-12 | 0 |
| GM2 | 3.27e-12 | 0 |
| GM3 | 2.10e-12 | 0 |
| GM4 | 1.30e-12 | 0 |
| GM5 | 5.31e-12 | 0 |

`phase6_check_part` = largest relative difference of the P4 MSE between this run's rows and the Phase 6 rows at the same A_crit; `phase6_check_ad` = AD optimum MSE / Phase 6 AD optimum MSE - 1.

## B11. Runs used

| tile | equal_part | equal_ad | mt_partitioned | mt_ad | inputs |
|---|---|---|---|---|---|
| AP1 | 20261116T230005Z_a12_equal_part_AP1 | 20261116T230003Z_a12_equal_ad_AP1 | 20261117T020518Z_a12_mt_AP1_v2_partitioned-refined | 20261116T225951Z_a12_mt_AP1_v2_ad | 20261116T224609Z_a12_inputs_10tiles |
| AP2 | 20261116T230005Z_a12_equal_part_AP2 | 20261116T230003Z_a12_equal_ad_AP2 | 20261117T021246Z_a12_mt_AP2_v2_partitioned-refined | 20261116T225950Z_a12_mt_AP2_v2_ad | 20261116T224609Z_a12_inputs_10tiles |
| AP3 | 20261116T230005Z_a12_equal_part_AP3 | 20261116T230003Z_a12_equal_ad_AP3 | 20261117T020238Z_a12_mt_AP3_v2_partitioned-refined | 20261116T225950Z_a12_mt_AP3_v2_ad | 20261116T224609Z_a12_inputs_10tiles |
| AP4 | 20261116T230005Z_a12_equal_part_AP4 | 20261116T230003Z_a12_equal_ad_AP4 | 20261117T020841Z_a12_mt_AP4_v2_partitioned-refined | 20261116T225954Z_a12_mt_AP4_v2_ad | 20261116T224609Z_a12_inputs_10tiles |
| AP5 | 20261116T230005Z_a12_equal_part_AP5 | 20261116T230003Z_a12_equal_ad_AP5 | 20261117T020132Z_a12_mt_AP5_v2_partitioned-refined | 20261116T225950Z_a12_mt_AP5_v2_ad | 20261116T224609Z_a12_inputs_10tiles |
| GM1 | 20261116T230030Z_a12_equal_part_GM1 | 20261116T230003Z_a12_equal_ad_GM1 | 20261117T020430Z_a12_mt_GM1_v2_partitioned-refined | 20261116T225950Z_a12_mt_GM1_v2_ad | 20261116T224609Z_a12_inputs_10tiles |
| GM2 | 20261116T230029Z_a12_equal_part_GM2 | 20261116T230003Z_a12_equal_ad_GM2 | 20261117T022340Z_a12_mt_GM2_v2_partitioned-refined | 20261116T230001Z_a12_mt_GM2_v2_ad | 20261116T224609Z_a12_inputs_10tiles |
| GM3 | 20261116T230029Z_a12_equal_part_GM3 | 20261116T230005Z_a12_equal_ad_GM3 | 20261117T022224Z_a12_mt_GM3_v2_partitioned-refined | 20261116T230031Z_a12_mt_GM3_v2_ad | 20261116T224609Z_a12_inputs_10tiles |
| GM4 | 20261116T230029Z_a12_equal_part_GM4 | 20261116T230005Z_a12_equal_ad_GM4 | 20261117T022343Z_a12_mt_GM4_v2_partitioned-refined | 20261116T230036Z_a12_mt_GM4_v2_ad | 20261116T224609Z_a12_inputs_10tiles |
| GM5 | 20261116T230029Z_a12_equal_part_GM5 | 20261116T230005Z_a12_equal_ad_GM5 | 20261117T020655Z_a12_mt_GM5_v2_partitioned-refined | 20261116T230031Z_a12_mt_GM5_v2_ad | 20261116T224609Z_a12_inputs_10tiles |

Summary run `20261117T163709Z_a12_summary_10tiles` under `/home/groups/hilley/lemcvt_results`.
