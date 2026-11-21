# [2026-11-20 10:50 PST] Addendum 20: Perron's hilltop estimator, the D bracket without the segment filter, snapped TTV heads, and Part B

This addendum follows Addendum 19 and its report `reports/v2/20261120T1030PST_hilltops_heads_slope.md`. Continue on branch `v2.3-tiles`. It has **one stop, at the end**. No gate in this addendum stops the work; gates only decide which D₀ a statement uses (§4). Rules are written in §6, before any result.

## 0. Orientation (you are a new agent instance)

### 0.1 Where things stand
- **The project, the sites and the models** are as in Addendum 19 §0.1. Read that section first.
- **Addendum 19 Part A found:**
  1. **The fixed 10³ m² network gives the right hilltops.** The counted pixels are the ridge crests between 10³ basins. At GM the two surveys agree to 1 % on the same pixels, and the crest curvature reproduces Perron et al.'s value (0.95 ×).
  2. **The 50-pixel segment filter of decision 1 was wrong in its premise** (planner error). D8 divide cells along a crest come in fragments of median 2–4 pixels, not as continuous lines, so the filter removed 60–80 % of every crest and R1 failed at every site. Without it every primary tile has ≥ 500 pixels.
  3. **At AP the 11 m crest curvature is 2.8 × Perron's value** at every window up to 51 m. Only the coarse block-mean route lands on it (1.07 ×).
  4. **The relief bracket.**
     - D_R/D_C = 2.0 (GM), 3.3 (AP), 3.4 (TTV).
     - Read from the figures, P4's D is inside the bracket on 14 of 18 GM tiles, and above it on 11 of 12 AP and 8 of 8 TTV tiles.
     - Each P4 fit's D is about 0.85 × the D_R of its own channel network.
  5. **Steve's picks sit on hillslope cells 2–5 m off the D8 valley axis.** Their D8 area depends mostly on the search radius: A_3 611 m², A_10 2,848 m², against 7,400 m² for M&D's Table 1. The 10 m search can jump into a neighbouring valley (pick 17: 2,589 → 33,875 m²).
- **New since then: Perron's own method.** The Supplementary Information of Perron, Kirchner & Dietrich (2009, Nature 460, §2.1) gives the method behind the GM and AP values in Perron et al. (2012) Table S2:
  - gridded elevations "with a horizontal point spacing of 1 m";
  - gradient and Laplacian "from the coefficients of a least-squares quadratic fit to the points within a 7 m radius";
  - the hilltop Laplacian "from the mean values within logarithmically spaced bins over the range of A|∇z| for which the binned Laplacian was roughly constant". Bins hold means of log-transformed data. In their Fig. S2a the plateau spans roughly A|∇z| = 10⁻³ to 3 m² at GM and 10⁻² to 3 m² at Eaton Hollow (their AP site, PAMAP data);
  - the values: GM −11.8 ± 0.4, Eaton Hollow −5.5 ± 0.1 (× 10⁻³ m⁻¹).

  The plateau range was chosen by eye. This addendum replicates the method with the range set by a rule written now (decision 2).
- **What the planner expects to learn (not a rule).**
  - Perron averages every low-A|∇z| cell (the whole rounded ridge top). Option (iii) averages divide-line cells (the sharpest strip).
  - On a parabolic hilltop these agree; on a crest-concentrated (nonlinear) hilltop the divide line is sharper. GM agreeing and AP not would fit that.
  - This addendum tests it directly.

### 0.2 Read, in this order
1. `reports/v2/20261120T1030PST_hilltops_heads_slope.md` (Addendum 19): the essential one.
2. `reports/directives/2026-11-20T0755PST_v2_addendum_19_hilltops_heads_slope.md`: its decisions and rules carry over except where this file changes them.
3. `reports/v2/scripts/a19_common.py`, `a19_hilltops.py`, `a19_site.py`, `a19_heads.py`, and the Part B scripts written but not run (`a19_fixed.py`, `a19_submit_sj.py`, `a19_dstat.py`, `a19_modelheads.py`, `a19_rules.py`).
4. `src/lemcvt/measure/hilltop_curvature.py` (`total_curvature`) and `src/lemcvt/analyses/er_star.py`.
5. Addendum 18 §6 (rules F3, F4, E1, E2, H2) and Addendum 2 §5 (pre-push check).
6. This file.

### 0.3 Where things are
- **Addendum 19 run directory:** `$GROUP_HOME/lemcvt_results/20261120T160830Z_a19_hilltops_heads_slope/` (`roads/`, `heads/`, `hilltops/`, `site/`, `fixed/`).
- **Addendum 18 run directory:** `$GROUP_HOME/lemcvt_results/20261119T143945Z_a18_expanded_fits/`.
- **TTV full-site 1 m grids:** `$GROUP_HOME/lemcvt_results/20261118T160326Z_a15_stage_b/grid_TTV/`.
- **M&D georeference:** `.../20261118T160326Z_a15_stage_b/md_georef/md1989_fig3.gpkg`.
- **M&D Table 1:** `$GROUP_HOME/lemcvt_results/20261118T185206Z_a16_tile_finish/md_georef/md1989_table1.csv`.
- **Steve's heads (copy of record):** `data/ttv/ttv_heads_manual_steve_20261119.csv` (64 rows; sha256 `ac87de3d…e702b6`).
- **TIGER roads:** `$GROUP_HOME/lemcvt_data/source/TIGER2024_roads/`.

### 0.4 Standing rules
As Addendum 19 §0.4, unchanged:
- reports, Slurm limits and the 300 core-hour submission cap;
- the never-commit list and units;
- no package code changes unless unavoidable.

New analysis code goes in `reports/v2/scripts/a20_*.py`. Figures go in `reports/v2/figures/a20/` (PNGs under 5 MB).

### 0.5 Before starting
1. **Confirm** `v2.3-tiles` against `origin`, that the tests pass on `serc`, and that no job is running.
2. **Confirm** the Addendum 19 run directory and its per-tile outputs exist.
3. **Where this addendum and the repo disagree,** the repo wins. Report it and stop.

## 1. Decisions (planner, 2026-11-20, approved by Steve)

1. **The segment filter is removed.**
   - **D₀,N** is from now on the Addendum 19 set without the segment filter:
     - the geometric set of the 10³ network (divides between outlet basins, edge margin, |∇z| ≤ 0.4, farther than L_f/2 from the network);
     - then the 10 m road and trail buffer;
     - then the convex drop.
   - The 20-pixel fragment filter is reported beside it as a sensitivity.
   - **This is a change after results.** It removes a filter whose premise was false; R1's threshold is unchanged. The report states this in its first section.
   - Addendum 19 deviations 2 (L_w) and 3 (convex drop) are accepted as reported.
2. **Perron's estimator, D₀,P, is the primary crest D (D_C) from this addendum on.** D₀,N is reported beside it in every statement.
   - **Grid:** the same 1 m elevation grid of each tile that `total_curvature` used in Addendum 19. State which grid (filled or not).
   - **Curvature:** at every cell, fit z = a + bx + cy + dx² + exy + fy² by least squares to all cells whose centres lie within radius r of the cell's centre.
     - Gradient |∇z| = √(b² + c²); Laplacian ∇²z = 2d + 2f.
     - **Primary r = 7 m**; sensitivities r = 3, 5, 10, 15 m.
     - Exclude cells within r + 2 cells of the tile edge or of nodata.
     - Check the implementation against an analytic paraboloid (∇²z known) and a plane (∇²z = 0) before use; report the errors.
   - **Area:** the tile's registered D8 area (cells × cell area).
     - D∞ is a sensitivity on the GM and AP primary tiles, only if it fits within 20 core-hours. Otherwise report "not run".
   - **Bins:** logarithmically spaced in A|∇z| (m²), 4 per decade, edges from 10⁻³ to 10⁵ m². Cells below 10⁻³ m² go to the first bin.
   - **Per bin, report:**
     - n and the fraction convex;
     - **L_bin = −10^⟨log₁₀(−∇²z)⟩** over the convex cells (∇²z < 0);
     - the signed arithmetic mean of ∇²z, as a sensitivity.
   - **Plateau rule (written now, replaces Perron's by-eye choice):**
     1. Start at the lowest bin with at least 200 cells.
     2. Add bins in order of increasing A|∇z| while every condition holds:
        - (a) the bin has at least 200 cells;
        - (b) its fraction convex is at least 0.6;
        - (c) |L_bin / L_plat − 1| ≤ 0.20, where L_plat is the cell-weighted geometric mean of the bins already included.
     3. Stop at the first bin that fails.
     4. **∇²z_h** = the cell-weighted geometric mean over the included bins.
     5. If fewer than 3 bins are included, the tile has **no plateau**; report it, and use D₀,N for that tile.
   - **Fixed-range sensitivity:** ∇²z_h as the geometric mean over all convex cells with A|∇z| ≤ 1 m², and ≤ 3 m² (the range read from Perron's Fig. S2a).
   - **D₀,P = E/|∇²z_h|.**
   - **Uncertainty:** block bootstrap with 100 m × 100 m blocks of the tile (2,000 draws, seed 20261120). The plateau rule is re-applied in every draw; report the 68 % interval and the share of draws with a plateau.
   - **Pooled per site:** the cells of all primary tiles of a site go into the same bins, and the rule is applied once. The bootstrap resamples blocks across all tiles.
   - **Selection against window shape:** on the D₀,N pixels, also compute the Laplacian with the r = 7 m disk. Then on the same tiles there are four estimates:
     - divide-line pixels with the 11 m square;
     - divide-line pixels with the 7 m disk;
     - plateau cells with the 7 m disk;
     - plateau cells with the 11 m square.

     Report all four, so that pixel selection and window shape are separated.
3. **The D bracket uses D₀,P at the lower end.**
   - Bracket = [D_C,lo, D_R,hi], where:
     - D_C,lo is the lower 68 % bound of D₀,P;
     - D_R,hi is the upper 68 % bound of D_R.
   - **D_R** is recomputed as in Addendum 19 decision 12, on the D₀,N crest pixels of decision 1 (the start points must be divide cells; D_R uses no curvature): on the 10³ network, at TTV1–TTV3 also on the picked-head network, and with the Euclidean-length sensitivity.
   - **The bracket with D₀,N at the lower end is reported beside it.**
   - A fit classed **above** depends only on D_R and is the same under both lower ends; the report says so wherever it applies.
4. **The second window end (D₀,N,B at L_w) is dropped.** For both estimators, report the full curve of D₀ against window (r = 3–15 m for D₀,P; L = 3–51 m for D₀,N) in place of a two-point window test.
5. **The TTV heads are snapped along their own flow paths.**
   - **Path:** for each pick, trace the D8 path downstream from the pick cell on the stage B full-site grid.
   - **Areas along the path:** read A at path distances 0, 5, 10, 15, 20 and 30 m. **A_P10** (area at 10 m of path) is primary.
   - **Axis entry:** the first path cell whose area is at least 10 × the area at the pick cell. Report its distance from the pick and its area.
   - **Slopes from the A_P10 cell:**
     - S_up: as Addendum 19 decision 3 (largest-area donor chain, 10 m), from the A_P10 cell;
     - S_down: 10 m downstream from the A_P10 cell.
   - **QC figure for every pick, 40 m × 40 m:**
     - the 1 m hillshade;
     - D8 cells with A ≥ 100 m², coloured by log A;
     - the pick, its traced path, and the cells at 5, 10 and 20 m;
     - the Addendum 19 A_3 and A_10 cells.

     Arrange them 16 to a page, labelled by pick id, in `reports/v2/figures/a20/heads/`. Steve will review them and may flag picks; nothing in this addendum waits on that review.
   - **Fits (i)–(iv) of Addendum 19 decision 4, repeated with A_P10:**
     - (ii) picks with A_P10 and S_up (primary);
     - (iii) with S_down;
     - (iv) without pick 49.

     Also report fits with A_P5 and A_P20, and the A·S² distributions.
   - **Field values:** A_F = the median A_P10. C_F(α) = the median of A_P10 · S_down^α.
6. **Part B runs as prepared in Addendum 19**, with these changes:
   - **(a) Field-fixed area threshold** (PF: K, D free; PF3: K = K₀). One evaluation per tile at each of these A values:
     - A_F (A_P10);
     - 7,400 m² (Table 1 median);
     - 2,848 m² (A_10);
     - 611 m² (A_3).

     Each is cheap (≈ 4 s); run all four on all 11 TTV primary tiles so that the result does not wait on Steve's head review.
   - **(b) Slope-dependent threshold SF** (C fixed at C_F) at α = 0, 1.65, 2.
   - **(c) SJ** (C free) at α = 0, 1.65 and 2. Add α = b̂ (fit (ii) with A_P10, rounded to 0.05) only if it differs from all three by more than 0.1. α = 1.65 is M&D's own exponent and replaces b̂ as the main non-zero case, because b̂ from the picks has a wide interval (Addendum 19: 1.35–3.43).
   - **(d) Model heads against snapped heads (R8):**
     - for P4 (Addendum 18), for PF at each of the four values, for SF and for SJ;
     - distances from each snapped head (its A_P10 cell) to the nearest model channel cell.
   - **(e) The D statements with the new D₀ (Addendum 19 decision 7)**, using the Gram-term route the Addendum 19 report describes in its §1.1:
     - P1, P2 and A2 refitted at D₀,P, and at D₀,N as the sensitivity;
     - one direct evaluation per tile as the check (tolerance 10⁻⁸ relative).
7. **`docs/measurement_methods.md`:** add a section defining D₀,P (this file's decision 2) and the revised D₀,N (decision 1), citing Perron et al. (2009, SI §2.1). This is a documentation change, not package code.
8. **Budget:** expected **≤ 250 core-hours**, from Steve's approved 1,000–2,000 (about 736 used to date). No submission over 300.
   - SJ at three α: 33 runs ≈ 134 core-hours expected (268 of limits). If b̂ is added, it goes in a second submission.
   - Everything else is expected ≤ 40 core-hours.

# Part A: measurement (no stop)

## 2. Crest D
1. **Verification.** The decision 2 estimator against a paraboloid and a plane; errors reported.
2. **D₀,P per tile** on every measured tile of Addendum 19 (73 tiles):
   - the plateau bins (range, number, cells), ∇²z_h with 68 % interval, the share of bootstrap draws with a plateau;
   - the fixed-range values at 1 and 3 m²;
   - D₀,P at r = 3, 5, 7, 10, 15 m;
   - D₀,P/D₀,N.
3. **D₀,N per tile** (decision 1), with the 20-pixel sensitivity and rule R1.
4. **The four-way comparison** (decision 2, last item) per site, pooled.
5. **Pooled per site:** the binned curve of ∇²z against A|∇z| (Perron's Fig. S2a style, signed log axis, plateau bins coloured), one panel per site with Perron's value marked.
6. **Map figures, per tile,** in `reports/v2/figures/a20/plateau/`:
   - the hillshade;
   - the plateau cells coloured by ∇²z;
   - the D₀,N pixels as an outline.

   This shows which cells each estimator counts. Also a gallery per site.
7. **The survey pairs:** D₀,P on both surveys of the 12 pairs, with the ratio.
8. **Rules R1, R11P, R13.**

## 3. Bracket and heads
1. **D_R recomputed** on the decision 1 crest pixels (Addendum 19 §2.7, every item), and the bracket of decision 3 per tile.
2. **E\* per site,** as Addendum 19 decision 13, on the new pixel set.
3. **The snapped heads** (decision 5): attributes, QC pages, fits, A·S², field values, rules R4 (unchanged; report the Addendum 19 value) and R5.
4. **The TTV anchor** (Addendum 19 decision 14) with D₀,P added.

## 4. Gates (no stop)
1. **R1** (D₀,N counts) and **R13** (D₀,P plateau) decide which tiles carry a D₀.
2. **R11P decides which D₀ the lower-end statements use at GM and AP:**
   - **pass:** D₀,P;
   - **fail:** both D₀,P and D₀,N are reported, and the lower-end rules (F3's "D₀ within bracket", R10 "below") are reported without a verdict at that site.

   The upper-end reading (R10 "above") stands in every case.
3. Time the Part B submissions; print the estimate first.

# Part B: fits

## 5. Fits
1. **SJ** (decision 6c).
2. **PF, PF3, SF** (decision 6a, 6b).
3. **The D statements** (decision 6e) and the bracket (rule R10) for every fit:
   - Addendum 18: AD, P4, P3 and A1 on every primary tile;
   - this addendum: PF, PF3, SF and SJ at TTV;
   - the model-network D_R for each.
4. **Model heads against snapped heads** (decision 6d), with figures as Addendum 19 decision 6.

## 6. Rules (written now; "holds" = at least 80 % of N, rounded up, as in Addendum 18; every rule reports its count and denominator)

**R1, crest count (D₀,N).** At each site, at least 500 counted pixels on at least 90 % of the primary tiles. If it fails at a site, D₀,N is reported there only on the tiles that pass. Nothing stops.

**R13, plateau (D₀,P).** At each site, the plateau rule finds a plateau on at least 90 % of the primary tiles, and the pooled site curve has one. Where a tile has none, it uses D₀,N (decision 2).

**R11P, calibration by Perron's method.** At GM and at AP, the pooled site |∇²z_h| at r = 7 m lies within 0.67–1.5 × Perron et al.'s value (11.8 and 5.5 × 10⁻³ m⁻¹). Reading, written now, from the four-way comparison:
- If the plateau estimate passes at AP while the divide-line estimate at the same window does not, the AP difference of Addendum 19 is **pixel selection**.
- If neither passes at AP, the difference is **location, survey or method detail**; say which the evidence favours.
- If the plateau estimate passes at GM and AP, Perron's values are reproduced by his method on our data.

**R2, D statements.** F3, F4, E1, E2 (P2) and H2 as Addendum 18 §6, with D₀,P as the primary D₀ (subject to §4.2), D₀,N beside it, and the Addendum 18 D₀,R outcomes as the third column. List every verdict that changes against Addendum 18.

**R10, the D bracket.** As Addendum 19 R10, with decision 3's bracket. Report the classification under both lower ends; the verdict uses D₀,P. The readings written in Addendum 19 R10 apply unchanged.

**R4, R5, R6, R7, R8, R9: as Addendum 19, with these substitutions.**
- **R5:** A_P10 in place of A_3, with S_up from the A_P10 cell.
- **R6:** PF and SF at A_F (A_P10) decide. The other three A values are reported without verdicts.
- **R7:** the best α > 0 among {1.65, 2, and b̂ if run}, against α = 0.
- **R8:** distances from the snapped heads.
- **R9:** add D/D₀,P.

**R14, field threshold and the bracket (report, with a reading written now).**
- For PF at each of the four A values on each TTV tile, report:
  - MSE/MSE_P4 and MSE/MSE_AD;
  - D's bracket class;
  - the model-network D_R.
- **Reading:** if PF at A_F puts D inside the bracket on at least 80 % of determined tiles while MSE_PF < MSE_AD on at least 9 of 11, then a field-measured threshold gives physically consistent D at a fit cost that still beats AD. Report the median MSE/MSE_P4 as that cost.

## 7. Cost
Print the estimate first.
- **D₀,P:** a 7 m disk quadratic on 73 tiles at five radii plus 2,000 block-bootstrap draws. Time it on one tile first; it is expected to be a few core-hours.
- **SJ:** dominates (≈ 134 core-hours).
- **Expected total:** ≤ 250 core-hours, no submission over 300.

## 8. Report and STOP
One report, `reports/v2/<YYYYMMDDTHHMM>PST_perron_D0_heads_partB.md`; figures in `reports/v2/figures/a20/`. Fixed outline:

1. **State check, deviations, decisions as applied.** State first that decision 1 changes a filter after results, and why.
2. **Crest D:**
   - the verification;
   - D₀,P and D₀,N per site;
   - the four-way comparison;
   - the binned curves against Perron;
   - the plateau maps;
   - the survey pairs;
   - rules R1, R13, R11P.
3. **The bracket:** D_R, E*, the TTV anchor, the interval figures under both lower ends.
4. **The heads:** snapping, QC pages, fits, rules R4, R5, field values.
5. **Field-fixed thresholds:** PF, PF3, SF; rules R6, R14.
6. **The slope-dependent threshold:** SJ; rule R7; TTV D (R9).
7. **Model heads against snapped heads:** rule R8.
8. **The D statements and the bracket:** rules R2, R10; every changed verdict.
9. **What the evidence supports, and where it is weak.**
10. **Decisions needed.**
11. **Appendices:** all tables, run ids, checksums, Slurm accounting.

**Stop.** Do not merge or tag. Push after the Addendum 2 §5 check.
