# [2026-11-19 06:15 PST] Addendum 18: model fits of the expanded tile set (Figs. 4–6, equal-parameter and held-out tests, trunk, continuity, TTV channel heads)

This addendum follows Addendum 17 and its Part B report `reports/v2/20261118T2045PST_measurement.md`. Continue on branch `v2.3-tiles`. It has **one stop, at the end**, plus two gates that stop the run early (§4.1, §4.2).

- **Part A:** settle the tile set and the D₀ inputs, run a same-pixel curvature diagnostic, run the regression gate, and print the cost estimate. Part A needs no review. If both gates pass, go on to Part B.
- **Part B:** the fits and every test of §5–§6.

## 0. Orientation (you are a new agent instance)

### 0.1 The project in brief
- **Repo:** `scdobbs/LEMs_competition_vs_threshold` (public), package `lemcvt`. Clone at `~/LEMs_competition_vs_threshold-JGR-branch` on Sherlock.
- **The paper (Dobbs & Hilley, JGR-ES resubmission)** compares two steady-state landscape models fitted to 1 m lidar tiles by elevation MSE:
  - the co-located advection–diffusion (AD) model: stream power plus linear diffusion at every cell;
  - the threshold-partitioned model: stream power where A ≥ A_crit, linear diffusion elsewhere.

  For each tile it compares the fitted K and D with per-tile measured values, K₀ = E/k_s and D₀ = E/|C_ht|, and it compares the MSEs (paper Figs. 4–6). K/K₀, D/D₀ and the MSEs do not depend on E.
- **Addenda 13–17 built and measured an expanded tile set** at three sites:
  - Gabilan Mesa (GM, CA), on the FEMA 2018 survey, with NCALM 2003 as the continuity check;
  - Allegheny Plateau (AP, PA), on USGS 3DEP 2019–20, with PAMAP as the continuity check;
  - Tennessee Valley (TTV, Marin County, CA), on Marin 2019 QL1.

  This addendum fits both models to that set.
- **Steve Dobbs** reviews every report through a separate planning conversation. Directives reach you as addenda in `reports/directives/`.

### 0.2 Read, in this order
1. `reports/v2/20261118T2045PST_measurement.md`, the Addendum 17 Part B report. This is the essential one: the primary set, the site m, k_s, C_ht and D₀ per tile.
2. `reports/directives/2026-11-18T1630PST_v2_addendum_17_partB_goahead.md`: the exclusions, the tiers, the FEMA decision and the screens.
3. `reports/v2/20261117T0900PST_sensitivity.md` and `reports/directives/2026-11-16T1415PST_v2_addendum_12_sensitivity.md`. This addendum repeats their equal-parameter and held-out tests on the larger set (variants P1–P4, A1, A2, AD).
4. `reports/v2/20261116T0921PST_wrapup.md`: the v2 results that §4.1 must reproduce, and the Figs. 4–6 equivalents.
5. `docs/measurement_methods.md` §0 and `docs/numerics_changes.md`.
6. Addendum 4 §1, Addendum 2 §4–§5 and Addendum 7 §0.4 (conventions, §0.4 below).
7. This file.

### 0.3 Where things are
- **Addendum 17 Part B run directory:** `$GROUP_HOME/lemcvt_results/20261119T002830Z_a17b_measurement/`:
  - `configs/`, `screens/` (`bedrock_screen.json`, `gapfill_screen.json`, `primary_set.json`);
  - `measure/` (`measurement.json`, `per_tile.csv`, `appendix_tables.md`).
- **The per-tile `measure` runs** (`*_measure_a17b_t<k>`) and the m-profile runs (`*_tile_run_<tile>_v2_m_profile`) are under `$GROUP_HOME/lemcvt_results/`. Appendix B9 of the Part B report maps tiles to run ids.
- **The v2.0.0 build:** `$GROUP_HOME/lemcvt_results/v2.0.0/`.
- **Earlier run directories:**
  - Addendum 17 Part A: `20261118T225314Z_a17_register_figures`;
  - Addendum 16: `20261118T185206Z_a16_tile_finish` (`md_georef/`, `ttv_reference/`, the M&D Table 1 transcription);
  - Addendum 15: `20261118T160326Z_a15_stage_b` (`md_georef/md1989_fig3.gpkg`: the 77 digitized channel heads).
- **Copies of record:** `$GROUP_HOME/lemcvt_data/source/`. Registered tiles: `$LEMCVT_DATA/<version>/<tile>/`, listed in `configs/data.yaml` and `data/manifest.csv`.
- **Tile configs:** `configs/tiles/<tile>.yaml` (76 new in Addendum 17, with `measured` blocks still empty); site configs in `configs/sites/`.
- **Scripts:** `reports/v2/scripts/a14_*` to `a17b_*`.
- **Papers:** `papers/` (never committed).
- **Environments:** `lemcvt`, `ci312` (ruff).

### 0.4 Standing rules
- **Reports:**
  - one report per checkpoint, titled `[YYYY-MM-DD HH:MM PST]`, file `reports/v2/<YYYYMMDDTHHMM>PST_<name>.md`;
  - follow the fixed outline;
  - stop for review.
- **Rules are set before results and never revised afterwards.** If a rule cannot be applied as written, report it and stop that part only. Carry on with the parts it does not touch.
- **Do not fix numerical issues silently.** List every deviation in the state check.
- **Slurm:**
  - every job on `serc`, no `--account`, arrays throttled, ≤ 300 concurrent CPUs;
  - no submission over 300 core-hours of CPU-weighted limits;
  - each job's limit at most twice its estimate (30 min floor). Time the first tasks of a new kind before you size an array (Addendum 17 deviation 9);
  - print the estimate first;
  - follow Sherlock and `serc` usage guidelines at all times; no heavy work on login nodes;
  - make run ids unique per task (Addendum 17 deviation 3).
- **Push** only after the Addendum 2 §5 pre-push check.
- **Never commit** `papers/`, PDFs, manuscripts, `working-with-steve.md`, `cry_for_help.txt`, point clouds, data rasters or the geology map. Figure PNGs under 5 MB may be committed under `reports/v2/figures/a18/`.
- **Units:** read them from headers; convert US survey feet with 1200/3937; never convert on an assumption. PAMAP AP1–AP5 are in feet and are converted at load.
- **Do not change package code** unless a step cannot be done otherwise. If you must change it, add tests, rerun the full suite on `serc`, and confirm that §4.1 still passes.

### 0.5 Before starting
1. **Confirm** the state of `v2.3-tiles` against `origin`, that the tests pass on `serc`, and that no job is running.
2. **Confirm** the paths above exist.
3. **Where this addendum and the repo disagree,** the repo wins. Report it and stop.

## 1. Decisions (planner, 2026-11-19, under Steve's delegation; Steve will strike any he does not accept)

1. **Bedrock screen: read the reference unit at formation level.** The reference unit is the Paso Robles Formation: labels QTp, QTcl, QTg, QTm and QTpc, all with `GEN_NAME` "Paso Robles Formation".
   - This settles an ambiguity in a rule written before the results. The expected answer was stated as the formation. The decision is made after the m-profiles exist, but the GM site m is the same either way (0.29; Part B §2.1).
   - **GM8, GM9 and GM21 return to the primary set**, which becomes 22 GM tiles: GM1_fema–GM7_fema, GM8, GM9, GM11, GM12, GM13, GM15, GM16, GM17, GM20, GM21, GM22, GM24, GM26, GM27, GM29.
   - **The clean subset** is 19: minus GM12, GM26 and GM29.
   - **Off-mesa, fitted and reported separately:** GM14, GM18, GM19, GM25, GM28, GM30.
   - **The GM site m stays 0.29.** Recompute the GM site summaries (site-pooled k_s and K₀, the medians of D₀) on the 22 tiles, and report them beside the 19-tile values of Part B.
   - **Report the map's full description for QTcl, QTg, QTm and QTpc,** from every attribute field of the map, so that Steve can check the unit call.
2. **D₀ is carried at both ends, D₀,A (L_f = 11 m) and D₀,B (L_w), at every site and in every group.** Every D statement in this addendum is made at both ends.
   - The window-stability test (rule 6.1) no longer decides between "measurement" and "bracket". Report its outcome as it stands in Part B, with the non-trivial count beside it: the tiles where more than one window was tested.
   - **GM sensitivity:** for GM1_fema–GM7_fema, also report D/D₀ with D₀ taken from the NCALM twin (GM1–GM7, D₀,A). It is labelled "NCALM-twin D₀" and does not enter any rule.
3. **The same-pixel curvature diagnostic (§3) runs in Part A.** It is report only, and it changes no input to the fits.
4. **Tiles with no usable hilltop set.** A tile whose option (iii) hilltop set at L_f = 11 m has fewer than **500 pixels** has no D₀, and it is left out of every D statement:
   - AP2_3dep (13), AP9 (30), AP10 (145), AP11 (0), AP14 (118), AP18 (25), AP27 (83), AP28 (119);
   - TTV3 (122), TTV4 (86).

   These tiles stay in every K, MSE and A_crit statement. They are fitted by P4, P3, AD and A1. The variants that fix D at D₀ (P1, P2, A2) are not run on them. Every rule that uses those variants states its denominator. Do not substitute a site-median D₀. The threshold was chosen at the clear gap in the Part B counts, between 145 and 579 pixels.
5. **TTV exponent:** fit at the rule's site m = 0.33. Add a per-tile m sensitivity at TTV only (§5.6). The six tiles whose profile optimum is at the grid edge (m = 0.05: TTV1, TTV6, TTV7, TTV8, TTV11, TTV15) are fitted at m_t = 0.05 as the profile gives, and flagged.
6. **Site m for the fits:** GM 0.29, AP 0.47, TTV 0.33. Every group at a site uses its site's m: primary, clean subset, off-mesa, continuity. The single exception is the regression gate (§4.1).
7. **E** (Addendum 17 decision 6), unchanged: GM 1.47 × 10⁻⁴, AP 5.5 × 10⁻⁵, TTV 7.7 × 10⁻⁵ m/yr.
8. **Trunk cells (Addendum 17 decision 7).** Trunk cells are the cells with A ≥ **1 km²** on the tile's area grid. A 1 km tile holds 1 km², so every such cell receives drainage from outside the tile.
   - In the trunk fit, trunk cells are held at their observed elevations, as fixed-elevation cells, and left out of the MSE.
   - The trunk fit runs on every primary and continuity tile whose largest A exceeds 5 km². The expected list is GM2_fema, GM4_fema, GM5_fema, AP3_3dep and AP4_3dep. Check every tile and report the full list with its largest A.
   - If the solver or the optimizer fails on a tile's trunk fit (no convergence, or no steady solution), report it and drop that tile's trunk fit. Do not modify the solver to make it work. This is Steve's condition: cut far-sourced trunks "as long as it doesn't break the model/optimization".
9. **The `measured` blocks** of the new tile configs are written now, from the Part B run (`measure/measurement.json`, `per_tile.csv`). Each block holds the site m, k_s (at the site m), C_ht at L_f and L_w, D₀,A, D₀,B, the hilltop pixel count, and provenance (run ids, the Part B report). Write the blocks for GM8, GM9 and GM21 too. Tiles below 500 hilltop pixels carry `D0: null` with the reason.
10. **Not in scope; these stay on the v2 tiles unless Steve asks:** the resolution-divergence runs, the channel-head Test 2 at GM and AP, the morphology metrics, and per-tile m sensitivity at GM and AP.
11. **Budget:** expected **≤ 900 core-hours** for this addendum, from Steve's approved 1,000–2,000 for the expanded set. Sherlock and `serc` guidelines are binding (§4.2).

## 2. Groups

| site | group | tiles | n | enters the rules |
| --- | --- | --- | --- | --- |
| GM | primary | GM1_fema–GM7_fema, GM8, GM9, GM11, GM12, GM13, GM15, GM16, GM17, GM20, GM21, GM22, GM24, GM26, GM27, GM29 | 22 | yes |
| GM | clean | primary minus GM12, GM26, GM29 | 19 | rules recomputed (§6.6) |
| GM | off-mesa | GM14, GM18, GM19, GM25, GM28, GM30 | 6 | report only |
| GM | NCALM continuity | GM1–GM7 (NCALM 2003; GM1–GM5 on the `area_model` old grids) | 7 | report only, plus the §4.1 gate |
| AP | primary | AP2_3dep–AP5_3dep, AP6–AP11, AP13, AP14, AP18–AP20, AP23–AP25, AP27–AP29 | 21 | yes |
| AP | clean | primary minus AP25 | 20 | rules recomputed (§6.6) |
| AP | PAMAP continuity | AP1–AP5 (PAMAP) | 5 | report only, plus the §4.1 gate |
| AP | AP1_3dep | Tier 1, "excluded (road), continuity only" | 1 | report only |
| TTV | primary | TTV1–TTV9, TTV11, TTV15 | 11 | yes |
| TTV | clean | primary minus TTV11 | 10 | rules recomputed (§6.6) |

Tier 1 exclusions (GM10, GM23, AP12, AP15, AP16, AP17, AP21, AP22, AP26, TTV10, TTV12, TTV13, TTV14) are not fitted.

# Part A

## 3. Same-pixel curvature diagnostic and hilltop QC figures (report only)

### 3.1 Same-pixel diagnostic

**Purpose.** Part B found that on the same ground FEMA gives a C_ht(L_f) 22 % lower than NCALM, and 3DEP 35 % lower than PAMAP. But each survey also selected different hilltop pixels:
- the rollover A_c is 22–56 m² on NCALM against 56–562 m² on FEMA;
- the median crest widths are 13–20 m against 20–86 m.

This diagnostic separates the effect of the surface from the effect of pixel selection.

1. **Pairs:**
   - GM1–GM7 (NCALM) with GM1_fema–GM7_fema;
   - AP1–AP5 (PAMAP) with AP1_3dep–AP5_3dep.
2. **Pixel sets:** for each tile in a pair, take the option (iii) hilltop pixel set at L_f = 11 m from its Part B `measure` run.
   - Map each set to the other survey's lattice by map coordinates: reproject if the CRSs differ, then take the nearest cell.
   - Drop pixels that fall within L_f/2 + 1 m of the other grid's edge or on its nodata.
   - Report the median and maximum mapping offset, and the pixels dropped.
3. **Four numbers per pair:** C_ht(L_f) with the same estimator as Part B (the `measure` code's hilltop-curvature function, called directly), for:
   - (a) the old survey's surface on the old survey's pixels;
   - (b) the new survey's surface on the old survey's pixels;
   - (c) the old survey's surface on the new survey's pixels;
   - (d) the new survey's surface on the new survey's pixels.

   (a) and (d) must equal the Part B values to within 0.1 %. If they don't, report it and stop this section only.
4. **Surface ratio** S = the median over tiles of the two same-pixel ratios, (b)/(a) and (d)/(c), pooled. **Selection ratio** P = the median of (c)/(a) and (d)/(b), pooled. Report S, P and the Part B total ratio (GM 0.78, AP 0.65) per site.
5. **Reading, written now:**
   - **"surface"** if S ≤ 0.85 and P is within 0.90–1.10;
   - **"selection"** if S is within 0.90–1.10 and P ≤ 0.85;
   - **"mixed"** otherwise.

   AP2's 3DEP set has 13 pixels, so report AP both with and without AP2.
6. The result goes to Steve for the writing of the D result. It changes no fit input.

### 3.2 Hilltop QC figures (for Steve's visual check)

**Purpose.** Steve will check by eye which cells the hilltop-curvature measurement counts and which it doesn't, and why the surveys differ. Make these figures from the Part B `measure` runs. They need no new measurement. Put them in `reports/v2/figures/a18/hilltop_qc/` as PNG at 200 dpi, each under 5 MB. Use the Addendum 17 hillshade convention: azimuth 315°, altitude 45°, no vertical exaggeration, from the tile's own 1 m filled grid.

1. **Pixel classes.** Every cell of a tile falls in exactly one class. Draw each class in a fixed, colour-blind-safe colour over the hillshade, at 60 % opacity, and use the same legend on every figure:
   - **(i) channel:** the rollover network, A ≥ the tile's rollover A_c;
   - **(ii) excluded near channel:** divide cells dropped because they lie within L_f/2 of a channel cell;
   - **(iii) counted hilltop:** the option (iii) hilltop set at L_f = 11 m, i.e. the pixels that give C_ht and D₀;
   - **(iv) divide, dropped for any other reason:** state the reason in the legend; omit the class if it is empty;
   - **(v) not part of the measurement:** everything else, left uncoloured.

   Mark cells in `nnfill_mask` or `gapfill_mask` with a thin hatch on top of any class.
2. **Per-tile QC figure** (`<tile>_hilltop_qc.png`), one per measured tile: primary, clean, off-mesa, continuity and AP1_3dep. Four panels:
   - (a) the full 1 km tile with the pixel classes;
   - (b) a 200 × 200 m zoom with the pixel classes, centred on the 200 m box that holds the most class (iii) pixels. Draw the box on panel (a);
   - (c) the same zoom, coloured by the local curvature at L_f = 11 m on the class (iii) pixels only, on a fixed colour scale per site (the 2nd–98th percentiles of the site's class (iii) values);
   - (d) the histogram of class (iii) curvature, with C_ht(L_f) marked.

   Panel title: tile, survey, rollover A_c, the class (iii) pixel count, the median crest width, C_ht(L_f), D₀,A, D₀,B, the window-test outcome, and "no D₀ (< 500 px)" where decision 4 applies.
3. **Survey-pair QC figure** (`<pair>_hilltop_pair.png`): GM1–GM7 (NCALM against FEMA) and AP1–AP5 (PAMAP against 3DEP).
   - Panels (a) and (b) of item 2 for both surveys, side by side, at the **same** zoom box. Take the box from the old survey (NCALM or PAMAP).
   - A third row overlays the two class (iii) sets on the new survey's hillshade: old only, new only, both, each in its own colour.
   - One histogram per pair with both surveys' class (iii) curvature.
   - The four §3.1 numbers in the caption.
4. **Gallery** (`<site>_hilltop_gallery_p<k>.png`): panel (a) of item 2 for every primary tile of a site, 6 columns, same scale, titled as in item 2. Order the panels by class (iii) pixel count, smallest first, so the collapsed sets come first.
5. **Model partition (added in Part B, after the fits):** for every primary tile, one more figure (`<tile>_partition.png`) with two panels at the item 2 zoom:
   - (a) the P4 partition at the fitted A_crit: channel (A ≥ A_crit) against hillslope (A < A_crit);
   - (b) the §3.2 item 1 pixel classes.

   This shows how the model's hillslope domain compares with the cells that set D₀.
6. **Index:** `reports/v2/figures/a18/hilltop_qc/README.md` lists every figure and the colour legend.
7. **When Part A is done,** commit and push these figures (all but item 5) after the Addendum 2 §5 check. Do not stop, so that Steve can look while Part B runs. Item 5 is pushed with the Part B report.

## 4. Gates

### 4.1 Regression gate (before any array)
With the v2 settings (metres, v2 optimizer, the v2.0.0 configs), rerun the P4 joint fit and the AD fit on **GM1 (m = 0.31) and AP2 (PAMAP, m = 0.47)**.
- The minimum MSE and the fitted K, D and A_crit (and r for AD) must equal the v2.0.0 build to a relative 1 × 10⁻⁹. Addendum 12 found 5 × 10⁻¹² or exact.
- If they do not match, **stop** before Part B and report.

### 4.2 Cost gate
Time one P4 fit, one AD fit and the P1–P3, A1, A2 reductions on one tile of each site. Then print the expected core-hours per group and per section, in the priority order of §7.
- If the expected total exceeds **900 core-hours**, **stop**. Report the estimate with a plan staged by that priority order.
- Otherwise go on to Part B.

# Part B

## 5. Fits

All fits use the v2 optimizer (exact profile over A_crit with closed-form K and D; for AD, exact profile over r = K/D with Brent refinement), metres, the site m of decision 6, and D8 routing as in v2. Every tile uses its registered area grid (`area_model` where registered). For P4, thin the A_crit grid to 2000 values and refine, as in Phase 6.

### 5.1 Primary fits (the Figs. 4–6 equivalents)
On every tile of every group in §2, except where §5 says otherwise:
- **P4**, the partitioned joint fit [K, D, A_crit];
- **AD** [K, D].

Report per tile:
- MSE_P4, MSE_AD and R_MSE = MSE_AD / MSE_P4;
- K/K₀ for both models;
- D/D₀,A and D/D₀,B for both models;
- A_crit with its 1 % and 5 % intervals.

### 5.2 Equal-parameter variants (Addendum 12 §2.1, unchanged)
On the primary and clean groups at every site:
- P1 [A_crit]: K₀ and D₀ fixed;
- P2 [K, A_crit]: D fixed at D₀;
- P3 [D, A_crit]: K fixed at K₀;
- A1 [D]: K fixed at K₀;
- A2 [K]: D fixed at D₀.

Run each variant that fixes D at both D₀,A and D₀,B. Use the exact reductions of Addendum 12 §2; no new solver.

### 5.3 Held-out (Addendum 12 §2.2, unchanged)
On the primary groups:
- the greedy basin split: edge-outlet basins of the 10³ m² network, largest first, each to the lighter half;
- fit on one half, score on the other, both ways, and average;
- P4 against AD, and P2 (at both D₀ ends) against AD.

Report each tile's cell shares. Mark as **lopsided** a split whose smaller half holds less than 30 % of the cells.

### 5.4 Trunk sensitivity (decision 8)
P4 and AD with the trunk cells held fixed and left out of the MSE. Report the same quantities as §5.1, plus the number and fraction of trunk cells.

### 5.5 Continuity
- NCALM GM1–GM7 and PAMAP AP1–AP5 at the site m (GM 0.29, AP 0.47): P4 and AD.
- Report the ratios against their survey twins (FEMA, 3DEP) of K, D, A_crit, MSE_P4, MSE_AD and R_MSE: per tile, median, and the largest single-tile difference.
- **GM1–GM5 at m = 0.31 against v2.0.0** is already covered by §4.1 for GM1. Run GM2–GM5 at 0.31 as well, and report whether they match v2.0.0 to the same tolerance. AP1–AP5 at 0.47 are the v2 setting; report whether they match.

### 5.6 TTV per-tile m (decision 5)
On every primary TTV tile, at its own m_t (Part B Appendix B5: TTV1 0.05, TTV2 0.45, TTV3 0.40, TTV4 0.42, TTV5 0.41, TTV6 0.05, TTV7 0.05, TTV8 0.06, TTV9 0.35, TTV11 0.05, TTV15 0.05):
- compute k_s and K₀ at m_t by χ–z, as in Part B;
- run P4 and AD.

Report beside the site-m fits.

### 5.7 TTV channel-head comparison
1. **Field head areas:**
   - the 63 source areas of M&D (1989) Table 1, from the Addendum 16 transcription;
   - the contributing area of each of the 77 digitized heads (Addendum 15), read from the Marin 2019 full-grid routing at the head location. Take the largest A within 3 m, to absorb the georeference RMS of 9.5 m at grid scale, and report the choice.

   Report both distributions: n and the 5th, 25th, 50th, 75th and 95th percentiles. Compare the 77 against the 63 by their medians and the ratio of the medians.
2. **Model:** the P4 A_crit of every primary TTV tile, at the site m and at m_t, with its 1 % interval.
3. **Classification (written now),** against the interquartile range (IQR) of Table 1:
   - **consistent** if the tile's A_crit 1 % interval overlaps the Table 1 IQR;
   - **above** if the interval's lower end exceeds the Table 1 75th percentile;
   - **below** if the interval's upper end is under the Table 1 25th percentile.

   Report the counts with their denominator, at the site m and at m_t. Report TTV1–TTV3 (the M&D study area) separately, also against the IQR of the 77 digitized heads.
4. **How the result will be read (written now):**
   - If at least 8 of the 11 tiles are "consistent", the paper states that the fitted A_crit at TTV agrees with the field channel heads.
   - If at least 6 of the 11 are "above", the paper states that, at the 1 km tile scale, the partitioned model's A_crit lies above the field channel heads at TTV.
   - Otherwise the paper reports the split.

   No retuning follows any outcome.

## 6. Rules (written now; per site, on the primary group unless stated)

Let N be the number of tiles in the group, or, for D statements and for any rule using P1, P2 or A2, the number with a usable D₀ (decision 4). A rule **holds** on at least **80 %** of N, rounded up (the v2 rules required "4 of 5"). Every rule reports its count and denominator, including where it fails.

1. **F1, fit:** MSE_P4 < MSE_AD. Report the median R_MSE and its range.
2. **F2, K:** |log₁₀(K/K₀)| is lower for P4 than for AD. Report the site medians of K/K₀ for both. The partitioned K agrees with the χ estimate partly by construction; state that beside the result.
3. **F3, D:** |log₁₀(D/D₀)| is lower for P4 than for AD, at D₀,A and at D₀,B separately. Report the site medians of D/D₀ for both models at both ends.
4. **F4, "AD's D is off by more than an order of magnitude":** holds at a site if AD's site median |log₁₀(D/D₀)| ≥ 1.0 at **both** ends. Report the value at each end.
5. **Equal parameters (Addendum 12 rule 2.3):**
   - **E1, count 1:** P1 beats both A1 and A2, at each D₀ end.
   - **E2, count 2:** P2 beats AD (at each D₀ end) and P3 beats AD, each reported separately.

   A variant "beats" another on a tile if its in-sample MSE is lower.
6. **Held-out:**
   - **H1:** the P4 held-out MSE is lower than AD's.
   - **H2:** P2 (at each D₀ end) beats AD held-out.

   Apply each rule on all primary tiles, and again on the tiles without a lopsided split. The all-tile result is the headline; the balanced result goes beside it.
7. **Clean subset (§6.6):** recompute F1–F4, E1, E2, H1 and H2 on the clean group, and report every outcome that differs from the primary group.
8. **Trunk (report only):** a trunk fit is **material** on a tile if K/K₀, D/D₀,A or A_crit (P4), or R_MSE, changes by more than 25 %, or if the tile's F1 outcome flips.
9. **TTV per-tile m (Addendum 12 rule 1.4, adapted):** the site-m choice is **immaterial** at TTV if both of these hold:
   - F1 holds at m_t;
   - the site medians of P4 D/D₀,A and AD K/K₀ change by less than 25 %.
10. **Off-mesa and continuity groups:** report F1–F4 counts, without "holds" verdicts.

## 7. Order and cost

Priority order, for the staged plan if §4.2 stops:
1. Part A (§3, including the QC figures, and §4).
2. §5.1 on the primary groups.
3. §5.2 and §5.3 on the primary groups.
4. §5.7 and §5.6.
5. §5.4.
6. §5.5.
7. §5.1 and §5.2 on the clean and off-mesa groups. The clean group needs no new fits: its tiles are a subset of the primary group, so its rules are recomputed from the primary fits. Only the off-mesa fits are new.

Expected ≤ 900 core-hours (decision 11). Print the estimate first. Throttle arrays to stay well under 300 concurrent CPUs.

## 8. Report and STOP

One report, `reports/v2/<YYYYMMDDTHHMM>PST_expanded_fits.md`. Figures in `reports/v2/figures/a18/`, as PNG at 200 dpi. Make the Figs. 4–6 equivalents with the v2 figure code if it exists; otherwise:
- per site, MSE_AD against MSE_P4 on log axes, with the 1:1 line;
- K/K₀ per tile for both models;
- D/D₀ per tile for both models, at both D₀ ends;
- primary, clean and off-mesa groups marked differently.

Also the model-partition figures of §3.2 item 5, under `reports/v2/figures/a18/hilltop_qc/`.

Fixed outline:
1. State check, deviations, and the decisions of §1 as applied: the GM primary set of 22, the `measured` blocks, the trunk list, the tiles without D₀.
2. Part A: the same-pixel diagnostic (§3.1) with its reading; the hilltop QC figures (§3.2), listed with their paths, with any tile where the classes look wrong noted; the regression gate; the cost estimate.
3. Primary fits per site (§5.1): the tables, the figures, and rules F1–F4.
4. Equal-parameter and held-out tests (§5.2, §5.3), with rules E1, E2, H1 and H2.
5. Clean subset and off-mesa groups (§6.7, §6.10).
6. Trunk sensitivity (§5.4, §6.8).
7. Continuity: NCALM and PAMAP against their twins; GM2–GM5 and AP1–AP5 against v2.0.0 (§5.5).
8. TTV: per-tile m (§5.6, §6.9) and the channel-head comparison (§5.7), with the classification and its reading as written.
9. What the fits support, and where they are weak. Address the claims of the original submission one by one: the fit, K, D, and "orders of magnitude".
10. Decisions needed.
11. Appendices: all tables, the run ids, checksums, Slurm accounting.

**Stop.** Do not merge or tag. Push after the Addendum 2 §5 check.
