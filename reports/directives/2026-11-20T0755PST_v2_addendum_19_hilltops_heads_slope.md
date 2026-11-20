# [2026-11-20 07:55 PST] Addendum 19 (revised): hilltop curvature on a fixed network, a measured D bracket, the hand-picked TTV channel heads, and the area–slope threshold

**This file replaces the 07:40 PST version of Addendum 19 in full.** If you have started on the 07:40 version, stop, report what was done, and continue from this file. Nothing in the 07:40 version is withdrawn; this version adds the D₀ calibration, the window scaling-break scan, hilltop filters, the relief-based D_R and the bracket rule, the E* report, and the TTV outside anchor (decisions 1, 10–14; §2.5–§2.8; rules R10–R12).

This addendum follows Addendum 18 and its report `reports/v2/20261119T2100PST_expanded_fits.md`. Continue on branch `v2.3-tiles`. It has **one stop, at the end**, plus gates that stop early (§4.1). Rules are written in §6, before any result.

## 0. Orientation (you are a new agent instance)

### 0.1 The project in brief
- **Repo:** `scdobbs/LEMs_competition_vs_threshold` (public), package `lemcvt`. Clone at `~/LEMs_competition_vs_threshold-JGR-branch` on Sherlock.
- **The paper (Dobbs & Hilley, JGR-ES resubmission)** compares two steady-state landscape models fitted to 1 m lidar tiles by elevation MSE:
  - the co-located advection–diffusion (AD) model;
  - a threshold-partitioned model (stream power where A ≥ A_crit, linear diffusion elsewhere).

  For each tile it compares the fitted K and D with the measured K₀ = E/k_s and D₀ = E/|C_ht|, and it compares the MSEs.
- **Sites:**
  - Gabilan Mesa (GM), FEMA 2018, with NCALM 2003 as continuity;
  - Allegheny Plateau (AP), 3DEP 2019–20, with PAMAP as continuity;
  - Tennessee Valley (TTV), Marin 2019 QL1. TTV1–TTV3 cover the Montgomery & Dietrich (1989) study area ("M&D").
- **Addendum 18 found:**
  - The partitioned model fits better than AD on 53 of 54 primary tiles, and its K matches K₀ at every site.
  - D agrees with D₀ only at GM.
  - At TTV the fitted A_crit is 27,000–89,000 m² on 9 of 11 tiles (M&D's field heads have a median of 7,400 m²), and the partitioned D is 15–20 × D₀, at the upper bound on four tiles.
- **Three problems this addendum addresses:**
  1. **D₀ depends on a hilltop network that is mostly a D8 artifact.** The option (iii) hilltop set is built from the slope–area *rollover* network. Its A_c is 11–562 m², and at 1 m D8 that network is largely parallel flow lines on planar slopes. The pixel selection changes with the survey (Addendum 18 §2.1: GM "selection"), collapses on 8 AP tiles and 2 TTV tiles, and picks up road edges (TTV5).
  2. **Crest curvature alone is not the right yardstick for a linear model's D.** Hillslopes at these sites are nonlinear (slopes flatten toward a critical gradient). A linear-diffusion model fitted to a nonlinear hillslope must raise D above the crest value to match the hillslope relief. So D from crest curvature (D_C) is the lower reference, and the D that matches the measured relief (D_R) is the upper reference. Addendum 8's E*–R* analysis already traces the quantities needed (`lemcvt.analyses.er_star`).
  3. **The TTV head comparison used automatically traced heads that are wrong.** The contour/channel confusion is in the Addendum 15 trace. Steve has hand-picked the M&D (1989) heads on the registered map and lidar.
- **Steve Dobbs** reviews every report through a separate planning conversation. Directives reach you as addenda in `reports/directives/`.

### 0.2 Read, in this order
1. `reports/v2/20261119T2100PST_expanded_fits.md` (Addendum 18): the essential one.
2. `reports/v2/20261118T2045PST_measurement.md` (Addendum 17 Part B): measurement methods as applied, and the window test.
3. `docs/measurement_methods.md` §0, and `reports/directives/2026-11-10T1715PST_refactor_v2_addendum_6.md` §1 (option (iii), L_f = 11 m, L_w, the window test).
4. `src/lemcvt/analyses/er_star.py` (module docstring, `trace_to_channel`, `fit_S_c`, `S_C_REFERENCE`), and the Addendum 8 report section on E*–R*.
5. `src/lemcvt/analyses/slope_threshold.py` (its module docstring), `reports/directives/2026-11-12T1630PST_refactor_v2_addendum_8.md` §2–§3, and that addendum's report section on the slope-dependent threshold.
6. `src/lemcvt/measure/audits.py` (`hilltop_audit`: C_ht on fixed networks, including 10³ m²).
7. The manuscript's D₀ method as the package holds it, if it does: search `src/lemcvt/` and `configs/` for the 10 m (GM) / 30 m (AP) block-mean, 9-point-stencil, first-order-divide route (`apply_d0_variant` is a starting point). Do not read `papers/`.
8. `reports/v2/20261118T1445PST_tiles_finish.md` §2.5 (M&D Table 1 transcription) and the Addendum 15 report section on the M&D georeference.
9. Addendum 4 §1, Addendum 2 §4–§5, Addendum 7 §0.4 (conventions).
10. This file.

### 0.3 Where things are
- **Addendum 18 run directory:** `$GROUP_HOME/lemcvt_results/20261119T143945Z_a18_expanded_fits/` (`prep/fit_specs.json`, `summary/`, `hilltop_qc/`). The fit runs are `*_a18_part_<tile>_<group>` and `*_a18_ad_<tile>_<group>` (its Appendix C16).
- **Addendum 17 Part B run directory:** `$GROUP_HOME/lemcvt_results/20261119T002830Z_a17b_measurement/`.
- **TTV full-site 1 m grids (stage B):** `$GROUP_HOME/lemcvt_results/20261118T160326Z_a15_stage_b/grid_TTV/` (`TTV_1m_filled.tif`, `TTV_1m_fd.tif`, `TTV_1m_area_cells.tif`; EPSG:6339).
- **M&D georeference:** `.../20261118T160326Z_a15_stage_b/md_georef/md1989_fig3.gpkg`.
- **M&D Table 1 transcription:** `$GROUP_HOME/lemcvt_results/20261118T185206Z_a16_tile_finish/md_georef/md1989_table1.csv` (63 rows: local valley slope above the head, source area, source-basin length, colluvium width).
- **Steve's heads:** `reports/directives/2026-11-19T1111PST_ttv_heads_manual_steve.csv`, placed by Steve beside this file.
  - 64 points, columns `pick_id, x_epsg6339, y_epsg6339, origin, uncertain, note`.
  - Picked by hand on the M&D (1989) Fig. 3 map registered on the Marin 2019 hillshade.
- **Copies of record:** `$GROUP_HOME/lemcvt_data/source/`. Tile configs in `configs/tiles/`.
- **Environments:** `lemcvt`, `ci312` (ruff).

### 0.4 Standing rules
- **Reports:**
  - one report per checkpoint, titled `[YYYY-MM-DD HH:MM PST]`, file `reports/v2/<YYYYMMDDTHHMM>PST_<name>.md`;
  - follow the fixed outline;
  - stop for review.
- **Rules are set before results and never revised afterwards.** If a rule cannot be applied as written, report it and stop that part only.
- **Do not fix numerical issues silently.** List every deviation in the state check.
- **Slurm:**
  - every job on `serc`, no `--account`, arrays throttled, ≤ 300 concurrent CPUs;
  - no submission over 300 core-hours of CPU-weighted limits;
  - each job's limit at most twice its estimate (30 min floor); time first tasks before sizing arrays;
  - print the estimate first;
  - follow Sherlock and `serc` guidelines at all times; no heavy work on login nodes;
  - unique run ids per task.
- **Push** only after the Addendum 2 §5 pre-push check.
- **Never commit** `papers/`, PDFs, manuscripts, `working-with-steve.md`, `cry_for_help.txt`, point clouds, data rasters, road vector data or the geology map. Figure PNGs under 5 MB may be committed under `reports/v2/figures/a19/`. Steve's heads CSV (64 coordinate rows, his own picks) may be committed (decision 2).
- **Units:** read them from headers; convert US survey feet with 1200/3937; never convert on an assumption. Areas from `*_area_cells.tif` are cell counts: multiply by the cell area read from the transform.
- **Do not change package code** unless a step cannot be done otherwise. New analysis code for this addendum goes in `reports/v2/scripts/a19_*.py` and may import from `lemcvt`. If you must change package code, add tests, rerun the full suite on `serc`, and confirm that the Addendum 18 regression gate (GM1 at m = 0.31, AP2 PAMAP) still reproduces v2.0.0 exactly.

### 0.5 Before starting
1. **Confirm** the state of `v2.3-tiles` against `origin`, that the tests pass on `serc`, and that no job is running.
2. **Confirm** the paths above exist, and that Steve's CSV has 64 rows, unique ids, and all points inside the M&D study-area polygon.
3. **Where this addendum and the repo disagree,** the repo wins. Report it and stop.

## 1. Decisions (planner, 2026-11-20, under Steve's delegation of technical choices; Steve will strike any he does not accept)

1. **The fixed-network crest D (D₀,N, also written D_C) is the primary D₀ from this addendum on.**
   - **Network:** option (iii) as in Addendum 6 §1 and the Part B code, with one change: the network is the fixed **A ≥ 10³ m²** D8 network on the tile's registered area grid, instead of the rollover network. This is the same network that k_s uses.
   - **Hilltop pixels:** the divide cells between outlet basins of that network.
     - Drop pixels within L_f/2 of a network cell, any with |∇z| > 0.4, and any with C_ht ≥ 0. Apply the same edge margin.
     - **Segment filter (new):** drop 8-connected hilltop segments with fewer than 50 pixels (about 50 m of crest at 1 m; Hurst et al. 2012 dropped segments under 50 m).
     - **Road and trail filter (new):** drop pixels within 10 m of a mapped road or trail (Hurst et al. 2012). Use a road/trail vector layer from `$GROUP_HOME/lemcvt_data/source/` if one exists. Otherwise fetch US Census TIGER/Line roads (all road classes) for the counties covering each site, store them under `source/` with a sha256, and use those. If no layer can be obtained for a site, apply no road filter there and say so. Report, per tile, the pixels removed by each filter.
   - **Curvature:** C_ht at L_f = 11 m and at L_w (the largest tested window not above the tile's median crest width, measured to the 10³ network), with the same estimator (`total_curvature`).
     - D₀,N,A = E/|C_ht(L_f)|; D₀,N,B = E/|C_ht(L_w)|.
     - Tile value: the median over counted pixels, with the segment bootstrap of Addendum 8 §2.2 (segments = 8-connected hilltop groups; 2,000 resamples, seed 20261120); report the 68 % interval.
     - Report the C_ht(L) curve and the window test.
   - **Comparisons:**
     - The Addendum 17 rollover values are kept as D₀,R,A and D₀,R,B, reported beside D₀,N in every D statement, as the sensitivity.
     - D₀,N without the two new filters is reported as D₀,N,unf, so the effect of the filters is visible.
   - **Why chosen now, before results:** the rollover network is a D8 artifact at 1 m (Addendum 18 QC figures). It makes D₀ depend on the survey through pixel selection, and it collapses on 10 tiles. The 10³ network is survey-independent at the scale that matters, and it is the network the K comparison already uses.
   - **Decision 4 of Addendum 18 applies to D₀,N:** no D₀ comparison on a tile with fewer than 500 counted pixels.
   - **Convention:** D = E/|C_ht|, with E the bedrock lowering rate and no density ratio. This matches the models' D and Perron et al. (2012). Hurst et al. (2012) and Grieve et al. (2016) include ρ_r/ρ_s; any value taken from such work must be converted before comparison (decision 14).
2. **Steve's 64 hand-picked heads replace the Addendum 15 automatic heads** in every channel-head comparison. The 77 automatic heads are not used again.
   - Copy the CSV to `data/ttv/ttv_heads_manual_steve_20261119.csv`, record its sha256, and commit it.
   - Pick 49 lies 55 m from the nearest traced M&D channel line (planner check). Keep it, and report every head statistic with and without it.
3. **Head attributes, read on the stage B full-site TTV grids** (not on cut tiles, so heads near tile edges are unaffected):
   - **Area:**
     - A_cell, at the pick cell;
     - **A_3, the largest D8 area within 3 m (primary)**;
     - A_10, within 10 m (sensitivity).
   - **Slope, two definitions:**
     - **S_up,** the local valley slope above the head, which is what M&D measured with a compass. Start at the cell that gives A_3 and follow the D8 donor with the largest area upslope for 10 m of path. Divide the elevation gain by the path length. Sensitivities: 5 m and 20 m. If the donor chain ends before the reach length, use the length available and flag it.
     - **S_down,** the slope over 10 m of D8 path downstream of the A_3 cell. This is exactly the S of `slope_threshold.valley_slope_reach`, the S the slope-threshold model uses.
   - **Location:** the TTV tile each pick falls in, or "none".
   - **D∞ area (sensitivity):** A_3 from D∞ routing on the full TTV grid, only if `lemcvt` can route it in ≤ 10 core-hours. Otherwise report "not run".
4. **The area–slope relation (M&D 1989, Table 1 and eq. 10: A = 1978 m² × tan θ^−1.65, R² = 0.75, least squares on log A).**
   - Fit log₁₀ A = log₁₀ a − b log₁₀ S by ordinary least squares, with A the dependent variable as M&D did, to:
     - (i) Table 1 (63 rows);
     - (ii) the picks with A_3 and S_up (primary);
     - (iii) the picks with A_3 and S_down;
     - (iv) (ii) without pick 49.
   - Report a, b, R², and the 95 % interval of b and of a by bootstrap over points (10,000 resamples, seed 20261120).
   - Report the distribution of A·S² over the picks and over Table 1. The M&D (1992) range is coded as 500–4,000 m² in `slope_threshold.MD_AS2_RANGE`.
5. **Field-fixed thresholds at TTV.** The aim: replace the free, tuned A_crit with a threshold measured in the field, and see whether the partitioned model still beats AD at an equal parameter count. On all 11 TTV primary tiles at the site m = 0.33:
   - **(a) Area threshold fixed at the field value,** A_F = the median A_3 of the picks.
     - Fit K and D free (**PF**, 2 parameters).
     - Also D free with K = K₀ (**PF3**, 1 parameter).
     - Sensitivity: A_F = 7,400 m², the Table 1 median.
   - **(b) Slope-dependent threshold A·S^α ≥ C** (`slope_threshold`, unchanged), at α = 0, α = b̂ (decision 4 (ii), rounded to 0.05), and α = 2.
     - **SF:** C fixed at the field value C_F = the median of A_3·S_down^α over the picks; K and D free (2 parameters).
     - **SJ:** C free (the module's joint profile over C, with K and D at each C; 3 parameters).
     - Use the module's hold-out as it is coded. α = 0 is run through the same module so that the held-out comparison is like for like. Check that α = 0 reproduces the Addendum 18 P4 optimum on every tile (Addendum 8's own `_phase6_check`).
   - The AD fit, and P4 and its variants, are the Addendum 18 runs. Do not rerun them.
6. **Model heads against the picked heads (TTV1–TTV3, the tiles inside the M&D map).** For P4 (Addendum 18), PF, SF and SJ at each α:
   - the model's channel heads (the most upstream channel cells, as `slope_threshold` defines heads; for the area threshold, the channel cells with no channel donor);
   - the distance from each pick inside these tiles to the nearest model channel cell;
   - the count of model heads against the count of picks inside the M&D polygon on each tile.

   Also make a figure per tile: hillshade, model channel cells, model heads, and the picks.
7. **Recompute the Addendum 18 D statements with D₀,N.**
   - Rules F3, F4, E1, E2 (P2) and H2, on every group.
   - The variants that fix D at D₀ (P1, P2, A2) must be refitted at D₀,N. If the stored Addendum 18 rows hold the Gram terms that make these reductions exact, recompute them from the stored rows. Otherwise rerun those variants only, never P4, P3, A1 or AD.
   - Report which route you took and why.
8. **Fix the Addendum 18 appendices** and commit the corrected report as a new file, `reports/v2/<timestamp>PST_expanded_fits_appendix_fix.md`. The original stays untouched.
   - **C16:** the "spec" column reads "AP5:pamap" on every row, and the "note" column repeats one string. Regenerate both from the run records.
   - **The checksum table is empty:** fill it with the sha256 (first 16 hex) of the filled, area and flow-direction grids of every fitted tile.
9. **Bounds are unchanged.** The TTV D upper bound (0.05 m²/yr) stays. Report bound hits in every new fit.
10. **Calibration of D₀ against Perron et al. (2012) at GM and AP.** Perron et al. (2012, Supplementary Table S2) measured hilltop curvature at both sites with the same E as ours:
    - GM: E = 147 m/Myr, ∇²z_h = −11.8 × 10⁻³ m⁻¹, D = 124 ± 19 cm²/yr;
    - AP: E = 55 m/Myr, ∇²z_h = −5.5 × 10⁻³ m⁻¹, D = 100 ± 8 cm²/yr.

    Their D equals E/|∇²z_h| with no density ratio (147 × 10⁻⁶ / 11.8 × 10⁻³ = 0.0125 m²/yr), our convention. Because E is the same, compare curvature directly.
    - On the primary tiles of GM and AP, compute the pooled site median of |C_ht| for D₀,N at L_f (counted pixels of all primary tiles pooled, segment bootstrap over tiles' segments) and its ratio to Perron's |∇²z_h|.
    - Also report the ratio at every tested window L, and at L_w.
    - **The manuscript's original route:** if the package still holds the manuscript's D₀ method (10 m GM / 30 m AP block-mean grids, 9-point stencil, first-order divides, median of convex values), run it on the same tiles and report its ratio to Perron's value too. If it does not exist in the package, report "not available" and do not rebuild it.
    - Rule R11 (§6) decides what this gate does.
11. **Window from the curvature scaling break.** Hurst et al. (2012) chose their window (12 m) at the break in the curve of curvature standard deviation against window size; Struble & Roering (2021) found the break at 15–20 m and used 15 m. Check ours, per site:
    - On every primary tile, compute `total_curvature` at L = 3, 5, 7, …, 31 m (odd metres) on all cells inside the edge margin, and the standard deviation of C at each L.
    - Fit log σ_C against log L with a two-segment piecewise linear model per site (tiles pooled, each tile's curve normalised by its value at 11 m). The break is L_break. Give its 68 % interval by bootstrap over tiles (2,000 resamples, seed 20261120).
    - Compute D₀,N at the odd window nearest L_break (**D₀,N,S**). This is a sensitivity. **L_f = 11 m stays primary**, because it was set before results (Addendum 6) and moving it now would be a revision after seeing data.
    - Report D₀,N against L for every window up to L_w on each tile (the full curve, not two points).
12. **Relief-based D (D_R) and the D bracket.** On a steady hillslope eroding at E with linear diffusion, z = −(E/2D)x², so a hillslope of length L_H and relief R implies
    - **D_R = E·L_H² / (2R).**
    - Per counted D₀,N hilltop pixel, trace L_H and R with `er_star.trace_to_channel` on the tile's grids:
      - **network for the bracket:** A ≥ 10³ m² (the same network as D₀,N);
      - **at TTV1–TTV3 also:** the picked-head network, i.e. the union of D8 flow paths downstream of the 64 picks (A_3 cells) on the stage B grid, cut to each tile. Only pixels inside the M&D polygon use this network.
      - Drop pixels whose path leaves the tile or never reaches the network (the module's flags). Report the counts.
    - **D_R per tile:** the median of per-pixel D_R, with the segment bootstrap and its 68 % interval.
    - **Consistency check:** per pixel, D_R/D_C = C_ht·L_H²/(2R) = 1 / (2·S̄/(C_ht·L_H)), the inverse of Addendum 8's normalised ratio. Report the tile medians of D_R/D_C, and the Addendum 8 normalised-ratio medians beside them (different pixel sets; no rule).
    - **Path-length sensitivity:** D8 path length is longer than the straight crest-to-channel distance, and L_H enters squared. Also report D_R with L_H = the Euclidean distance from the hilltop pixel to the path's end cell.
    - **Model-network D_R (report only, no rule):** for each TTV fit (P4, PF, SF, SJ) and each GM and AP P4 fit, D_R traced to that fit's own channel network (the cells the fit treats as channel). This shows whether a fit's D is consistent with the hillslope length the fit itself creates.
    - **The bracket:** [D_C,lo, D_R,hi], where D_C,lo is the lower 68 % bound of D₀,N,A and D_R,hi the upper 68 % bound of D_R (10³ network) on the tile. Rule R10 uses it. **These definitions are fixed now, before D_R is computed.** The Addendum 18 F3 test is kept and reported unchanged (R2); the bracket is an additional test, not a replacement.
13. **E* per site, to bound the known curvature bias.** Struble & Roering (2021) showed that measured |C_ht| is systematically too small at high E* (about 10 % at E* = 10, over 20 % at E* = 30, larger with wider windows). A too-small |C_ht| makes D_C too high. Report, per tile and per site, the median E* = 2·|C_ht|·L_H / S_c over the bracket pixels:
    - at S_c = `er_star.S_C_REFERENCE`;
    - at the site's fitted S_c from `er_star.fit_S_c`, if the fit lies inside its own 68 % interval.

    Give the share of pixels with E* < 10, 10–30 and > 30. Apply no correction.
14. **TTV outside anchor (report only).** Dietrich and coworkers report a soil transport coefficient of about 5 × 10⁻³ m²/yr for the humid northern California and Oregon coast ranges. That value is in the soil-flux convention; in our bedrock convention it is about 5 × 10⁻³ × ρ_s/ρ_r ≈ 2.5 × 10⁻³ m²/yr (ρ_s/ρ_r = 0.5 as in Hurst et al. 2012). Report the TTV site median D₀,N,A, D₀,N,S and D_R as ratios to 2.5 × 10⁻³ and to 5 × 10⁻³ m²/yr. No rule; the planner will confirm the original convention.
15. **Budget:** expected **≤ 320 core-hours**, from Steve's approved 1,000–2,000 (about 730 used to date). The additions in decisions 10–14 are measurement and D8 tracing, expected ≤ 20 core-hours. No single submission over 300.

# Part A: measurement (cheap; no stop until §4)

## 2. D₀ on the fixed network, its calibration, and the D bracket

### 2.1 D₀,N per tile (decision 1)
On every measured tile of Addendum 18 (primary, off-mesa, NCALM, PAMAP, AP1_3dep; 73 tiles), compute D₀,N,A and D₀,N,B. Report per tile:
- the counted pixel count, and the pixels removed by the segment filter and by the road filter;
- C_ht at L_f and L_w, with 68 % intervals;
- L_w and the median crest width;
- the window test (non-trivial count beside it);
- D₀,N/D₀,R and D₀,N/D₀,N,unf at each end.

### 2.2 Same-pixel diagnostic
Addendum 18 §3.1, unchanged, on the 12 survey pairs with the D₀,N pixel sets. Give the S and P ratios and the reading.

### 2.3 QC figures
In Addendum 18 §3.2 style, under `reports/v2/figures/a19/hilltop_qc/`:
- per tile, the pair figures and the site galleries, with the classes now relative to the 10³ network, and two added classes: removed by the segment filter, removed by the road filter;
- a one-page comparison per site: the counted pixels of the rollover set against the 10³ set on 6 tiles chosen by rank of D₀,N/D₀,R (lowest two, middle two, highest two);
- TTV5 specifically, with the road layer drawn.

### 2.4 Site summaries
The median D₀,N,A and D₀,N,B per site and group, beside the D₀,R and D₀,N,unf medians.

### 2.5 Calibration against Perron et al. (2012) (decision 10)
1. The pooled |C_ht| ratio to Perron's value at GM and AP, with its 68 % interval, at L_f, at L_w, and at every tested L.
2. The manuscript route's ratio, or "not available".
3. One figure per site: |C_ht| against L, with Perron's value as a horizontal line.
4. Rule R11.

### 2.6 Window scaling break (decision 11)
1. Per site: the σ_C(L) curves, the two-segment fit, L_break and its 68 % interval.
2. D₀,N,S per tile and site; D₀,N,S/D₀,N,A.
3. The D₀,N(L) curve per tile (a gallery per site).
4. Rule R12.

### 2.7 D_R and E* (decisions 12, 13)
1. Per tile: traced and dropped pixel counts; median L_H and R; D_R with 68 % interval; D_R/D_C; the Euclidean-length D_R; the Addendum 8 normalised ratio beside it.
2. At TTV1–TTV3: the same on the picked-head network.
3. Per tile and site: E* medians and the E* band shares.
4. Figures: per site, D_C and D_R per tile as an interval plot (tiles on the x axis), with the Addendum 18 fitted D of AD, P4 and P3 drawn on it. This figure is the visual of rule R10, and is made now with the Addendum 18 fits only.

### 2.8 TTV anchor (decision 14)
The ratios of decision 14, in one table.

## 3. The heads and the area–slope relation (decisions 2–4)
1. Head attributes for all 64 picks: a table and a CSV in the run directory.
2. Fits (i)–(iv), with figures:
   - log A against log S with Table 1, the picks (S_up) and the picks (S_down) in three symbols;
   - M&D's eq. 10 line and the fitted lines drawn on it.
3. The A·S² distributions, and the distributions of A_3, A_10 and Table 1 area: n and the 5th, 25th, 50th, 75th and 95th percentiles.
4. The field values for §5: A_F, and C_F for each α.

## 4. Gates

### 4.1 Hilltop gate, calibration gate, and cost gate
1. Apply rule R1 (§6). If it fails at any site, **stop** after Part A and report.
2. Apply rule R11 (§6). If it fails at GM or AP, do **not** run §5.3 or rule R10 at the failing site; run the rest of Part B. State this at the top of the report.
3. Time one SJ run, one SF evaluation and one PF evaluation on TTV2. Then print the expected core-hours for Part B.
   - If they exceed **300**, stop with a plan staged in this order: §5.2 SF and PF, §5.3 the D statements and the bracket, §5.1 SJ.

# Part B: fits

## 5. Fits
1. **SJ** at α = 0, b̂, 2 on the 11 TTV primary tiles (decision 5b; 33 runs).
2. **PF, PF3, SF** on the 11 TTV primary tiles (decision 5), at A_F and at 7,400 m² for PF and PF3. Each is a single evaluation per spec at the fixed threshold.
3. **The D statements with D₀,N** (decision 7), on every group of Addendum 18, and **the bracket** (rule R10) for every fit: Addendum 18 AD, P4, P3 and A1 on every primary tile; PF, PF3, SF and SJ at TTV. Also the model-network D_R (decision 12).
4. **The head comparison** (decision 6) and its figures.

## 6. Rules (written now; "holds" = at least 80 % of N, rounded up, as in Addendum 18; every rule reports its count and denominator)

**R1, hilltop gate.** At each site, D₀,N has at least 500 counted pixels on at least 90 % of the primary tiles, and a QC look at the six comparison tiles (§2.3) shows the counted set on crests. If either fails, stop after Part A.

**R2, D statements.** F3, F4, E1, E2 (P2) and H2 as written in Addendum 18 §6, with D₀,N as the primary D₀. Report the D₀,R outcomes beside them and list every verdict that changes.

**R3, same-pixel reading.** The Addendum 18 §3.1.5 rule, applied to the D₀,N sets.

**R4, Table 1 reproduction.** Fit (i) must give a within 10 % of 1,978 m² and b within 0.05 of 1.65. If it does not, report the difference, flag every Table 1 comparison, and continue.

**R5, the picks against M&D's relation.** The picks are **consistent** with M&D's area–slope relation if both hold:
- M&D's b (1.65) lies inside the 95 % interval of b from fit (ii);
- the median of A_3 / (1978 × S_up^−1.65) over the picks lies within 0.5–2.

Report the same for fits (iii) and (iv), without verdicts.

**R6, field-fixed area threshold against AD at equal count.** PF beats AD if MSE_PF < MSE_AD on at least 9 of 11 tiles. The same rule applies to SF at each α. Report PF3 against A1 and A2 (count 1), with the same threshold.

**R7, slope threshold against area only.**
- The slope-dependent threshold is **preferred at TTV** if the best α > 0 in SJ lowers the held-out MSE by more than 2 % against SJ at α = 0 on at least 9 of 11 tiles (the module's rule, scaled to 11 tiles).
- Report the in-sample MSE ratios beside it.

**R8, model heads against picked heads** (report, with a reading written now). On TTV1–TTV3 together, a fit's heads **agree with the mapped heads** if both hold:
- the median pick-to-model-channel distance is ≤ 25 m;
- the ratio of model heads to picks inside the M&D polygon is within 0.67–1.5.

Otherwise say whether the model has too few channels (ratio < 0.67, distances large) or too many.

**R9, TTV D.** For every TTV fit, report D/D₀,N, D/D₀,R, D/D_R and bound hits. No rule.

**R10, the D bracket.** For each fit on each tile, classify the fitted D:
- **inside** if D_C,lo ≤ D ≤ D_R,hi;
- **below** if D < D_C,lo;
- **above** if D > D_R,hi;
- **undetermined** if D is at a bound and the bound lies inside the bracket, or the tile has fewer than 500 counted pixels.

A model's D is **consistent with linear diffusion standing in for the measured hillslopes** at a site if it is inside on at least 80 % of the determined primary tiles. Report the counts of all four classes per model and site. Readings, written now:
- If a partitioned fit at TTV is **above** on at least 80 % of determined tiles, its excess D cannot be explained as a linear model compensating for nonlinear hillslopes; read it with R5–R8 as a sign that the area-only threshold is mis-specified there.
- If it is **inside**, the excess over D_C is what a linear model needs to match the measured relief, and the F3 failure at TTV is not by itself evidence against the model.
- If it is **below**, say so; no reading is set.

**R11, calibration gate.** At GM and at AP, the pooled median |C_ht| of D₀,N at L_f lies within 0.67–1.5 × Perron et al.'s (2012) |∇²z_h|. If it does not at a site, the D statements (§5.3, R2, R10) are not run there in this addendum, and the report says at which window, if any, the ratio does lie in range (no D statement may be made at that window). TTV has no such value and is not gated.

**R12, window (report, with a reading written now).** L_f = 11 m is **consistent with the scaling break** at a site if it lies inside the 68 % interval of L_break, or L_break lies within 8–20 m (the range of Hurst et al. 2012 and Struble & Roering 2021). Otherwise report D₀,N,S/D₀,N,A at the site; no verdict changes.

## 7. Cost
Print the estimate first. The SJ runs dominate (about 4 h each on the Addendum 18 timings; 33 runs). Everything else is single evaluations, D8 traces or measurement. Expected ≤ 320 core-hours in total, no submission over 300.

## 8. Report and STOP
One report, `reports/v2/<YYYYMMDDTHHMM>PST_hilltops_heads_slope.md`; figures in `reports/v2/figures/a19/`. Fixed outline:

1. State check, deviations, decisions as applied (the heads CSV and its checksum; the road layers and their checksums; the appendix fix).
2. D₀ on the fixed network: per-site summaries, D₀,N/D₀,R, D₀,N/D₀,N,unf, the window test, the same-pixel diagnostic, the QC figures, rule R1.
3. Calibration and window: the Perron comparison and rule R11; the scaling break, D₀,N,S and rule R12.
4. The D bracket: D_R, D_R/D_C, E*, the TTV anchor, the interval figures.
5. The heads: attributes, the area–slope fits, rules R4 and R5, A·S².
6. Field-fixed thresholds: PF, PF3, SF; rule R6.
7. The slope-dependent threshold: SJ at α = 0, b̂, 2; rule R7; TTV D (R9).
8. Model heads against picked heads: rule R8, with figures.
9. The Addendum 18 D statements with D₀,N (rule R2, every changed verdict) and the bracket (rule R10), with the model-network D_R.
10. What the evidence supports, and where it is weak.
11. Decisions needed.
12. Appendices: all tables, run ids, checksums, Slurm accounting.

**Stop.** Do not merge or tag. Push after the Addendum 2 §5 check.
