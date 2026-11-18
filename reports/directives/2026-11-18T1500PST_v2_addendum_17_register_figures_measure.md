# [2026-11-18 15:00 PST] Addendum 17: register the tile set, tile figures for visual review (STOP), then per-tile measurement (STOP)

This addendum follows Addendum 16 and its report `reports/v2/20261118T1445PST_tiles_finish.md`. Continue on branch `v2.3-tiles`. It has **two parts with a stop after each**:
- **Part A:** registration, the GM survey-control tiles, and the tile figures. **Stop.** Steve reviews the figures and may exclude tiles.
- **Part B:** per-tile measurement, run only after Steve's written go-ahead, which lists any excluded tiles. **Stop.**

Model fitting of the expanded set is Addendum 18.

## 0. Orientation (you are a new agent instance)

### 0.1 The project in brief
- **Repo:** `scdobbs/LEMs_competition_vs_threshold` (public), package `lemcvt`. Clone at `~/LEMs_competition_vs_threshold-JGR-branch` on Sherlock.
- **The paper (Dobbs & Hilley, JGR-ES resubmission)** compares two steady-state landscape models fitted to 1 m lidar tiles by elevation MSE:
  - the co-located advection–diffusion (AD) model;
  - a threshold-partitioned model (stream power where A ≥ A_crit, linear diffusion elsewhere).

  For each tile it compares the fitted K and D with per-tile measured values, K₀ = E/k_s and D₀ = E/|C_ht| (paper Figs. 4–6).
- **Addenda 13–16 built an expanded tile set:**
  - Gabilan Mesa (GM, CA): GM1–GM7 on NCALM 2003, and GM8–GM30 on the 2018 FEMA Region 9 survey.
  - Allegheny Plateau (AP, PA): AP1–AP29 on USGS 3DEP 2019–20.
  - Tennessee Valley (TTV, Marin County, CA): TTV1–TTV15 on Marin 2019 QL1. TTV1–TTV3 cover the Montgomery & Dietrich (1989) study area.
- **Steve Dobbs** reviews every report through a separate planning conversation. Directives reach you as addenda in `reports/directives/`.

### 0.2 Read, in this order
1. `reports/v2/20261118T1445PST_tiles_finish.md` (Addendum 16: TTV, the GM extension, the GM area check, the D8 guard). This is the essential one.
2. `reports/v2/20261118T1030PST_tiles_stage_b.md` and `20261117T2000PST_tiles_stage_a.md`.
3. `docs/measurement_methods.md` §0 (the v2 measurement methods; Part B uses them unchanged).
4. `reports/directives/2026-11-10T1715PST_refactor_v2_addendum_6.md` §1 (the option (iii) hilltop rule and the AP D₀ bracket).
5. Addendum 4 §1, Addendum 2 §4–§5 and Addendum 7 §0.4 (conventions, §0.4 below).
6. This file.

### 0.3 Where things are
- **Addendum 16 run directory:** `$GROUP_HOME/lemcvt_results/20261118T185206Z_a16_tile_finish/`:
  - `tiles/TTV-x_unregistered/`, `tiles_gmf/GM-FEMA2018-x_unregistered/`: parked tiles;
  - `grid_GMF/`, `ept_GMF/`, `draw/`, `draw_gmf/`, `gm_area_check/`, `gm_area_diag/`, `md_georef/`, `ttv_reference/`, `gm_footprints/`, `gm_gate/`.
- **Stage B run directory:** `$GROUP_HOME/lemcvt_results/20261118T160326Z_a15_stage_b/` (`grid_<site>/`, `md_georef/md1989_fig3.gpkg`, `draw/`, `tiles/`).
- **Stage A run directory:** `$GROUP_HOME/lemcvt_results/20261118T002225Z_a14_stage_a/` (`pools/`).
- **Copies of record:** `$GROUP_HOME/lemcvt_data/source/`. Registered tiles: `$LEMCVT_DATA/<version>/<tile>/`, listed in `configs/data.yaml` and `data/manifest.csv`.
- **Scripts:** `reports/v2/scripts/a14_*.py`, `a15_*.py`, `a16_*.py`. The unit-check windows are in `a15_common.Z_RANGE`.
- **Papers:** `papers/` (never committed).
- **Environments:** `lidar`, `lemcvt`, `ci312` (ruff).

### 0.4 Standing rules
- **Reports:**
  - one report per checkpoint, titled `[YYYY-MM-DD HH:MM PST]`, file `reports/v2/<YYYYMMDDTHHMM>PST_<name>.md`;
  - follow the fixed outline;
  - stop for review.
- **Rules are set before results and never revised afterwards.** If a rule cannot be applied as written, report it and stop.
- **Do not fix numerical issues silently.** List every deviation in the state check.
- **Slurm:**
  - every job on `serc`, no `--account`, arrays throttled, ≤ 300 concurrent CPUs;
  - no submission over 300 core-hours of limits;
  - each job's limit at most twice its estimate (30 min floor);
  - print the estimate first;
  - follow Sherlock and `serc` usage guidelines at all times (no heavy work on login nodes).
- **Push** only after the Addendum 2 §5 pre-push check.
- **Never commit** `papers/`, PDFs, manuscripts, `working-with-steve.md`, `cry_for_help.txt`, point clouds or data rasters. Figure PNGs under 5 MB may be committed under `reports/v2/figures/a17/`.
- **Units:** read them from headers; convert US survey feet with 1200/3937; never convert on an assumption.

### 0.5 Before starting
1. **Confirm** the state of `v2.3-tiles` against `origin`, that the tests pass on `serc`, and that no job is running.
2. **Confirm** the paths above exist.
3. **Where this addendum and the repo disagree,** the repo wins. Report it and stop.

## 1. Decisions (settled by Steve on 2026-11-18)

1. **The unit-check windows were a units test, not a selection rule.** Units are established independently:
   - Theil–Sen slope 0.99999 (Marin 2019 against NCALM 2006);
   - 0.99995 (FEMA 2018 against NED);
   - 1.0012 (FEMA 2018 against NCALM 2003).

   Set `Z_RANGE` to **TTV 0–800 m** (Mount Tamalpais is 784 m) and **GM-FEMA 0–750 m** (the mesa top near the range front is about 700 m). Re-run the unit check on the parked tiles and register them as cut: TTV1–TTV15 as `TTV-x`, GM8–GM30 as `GM-FEMA2018-x`. Do not re-draw.
2. **GM1–GM5 modelling area = the old tile area grids.** GM2's and GM5's large trunks agree with the coarse routing to 1–9 %. GM5's second trunk agrees with the NED routed over the whole box to 3 %. The rule's failure came from the narrowness of Perron's 60.5 km² coarse strip, not from the old grids.
   - Re-register role `area_model` for GM1–GM5 as the old grids. GM3 already is.
   - Keep the Addendum 16 rule grids beside them as `area_model_rule`, unused.
3. **The GM landscape polygon** of Addendum 16 (`a16_common.GM_LANDSCAPE_POLY_UTM`) is accepted.
4. **GM survey control.** Cut FEMA 2018 versions of GM1–GM7 (`GM<n>_fema`) at the same footprints on the FEMA lattice (§2.2). Part B measures GM1–GM7 on both surveys; the rule in Part B §6.4 decides whether GM can be pooled across surveys.
5. **Visual review before measurement.** Part A makes the figures of §3. After reviewing them, Steve may exclude tiles **before any measurement**, for stated visual reasons:
   - grid artifacts;
   - human disturbance not caught by the screens;
   - fresh landsliding or other obvious non-steady features;
   - anything else he names.

   Each exclusion and its reason is recorded in the Part B report; excluded tiles stay registered and are listed in an appendix.
6. **Erosion rates E** (K₀ = E/k_s, D₀ = E/|C_ht|):
   - GM: 1.47 × 10⁻⁴ m/yr (Perron et al. 2012), applied to every GM tile;
   - AP: 5.5 × 10⁻⁵ m/yr;
   - TTV: 7.7 × 10⁻⁵ m/yr (77 ± 15 m/Myr, Heimsath et al. 1997).

   K/K₀, D/D₀ and the MSEs do not depend on E; absolute K and D do. Record that E is carried over to the extension tiles.
7. **Recorded for Addendum 18 (modelling):** a trunk sensitivity fit for every tile whose largest drainage area inside exceeds 5 km² (GM2, GM4, AP3, AP4 and any others). Trunk cells are held at their observed elevations and excluded from the MSE. This is reported beside the primary fit.
8. **Budget, this addendum:** Part A expected ≤ 20 core-hours; Part B expected ≤ 150 core-hours, from the approved 1,000–2,000. Sherlock and `serc` guidelines are binding.

# Part A

## 2. Registration and the control tiles

### 2.1 Register
- Apply decision 1. Re-run the unit check on the 15 TTV and 23 GM-FEMA tiles with the new windows, and register them, recording the window used.
- Apply decision 2 to `area_model`.
- Report the final registered list per site: name, version, survey, footprint, area source, kind (original, drawn, M&D by design, control).

### 2.2 GM survey-control tiles (decision 4)
1. **Footprint:** GM1–GM7 and their upstream catchments on the 10 m NED box routing of Addendum 16 (`gm_area_diag`), plus 500 m. Fetch the missing FEMA 2018 class-2 boxes from the Entwine copy, as in Addendum 16 §3.3.
2. **Grid and route** with the Addendum 16 conventions (1 m, EPSG:6339, gap rule, full-grid routing). Run rule 5.3 against the NED.
3. **Cut** GM1_fema–GM7_fema at the NCALM footprints (reproject the EPSG:32610 corners; snap to the lattice; report the offsets). Register them as `GM-FEMA2018-ctrl`.
4. **Area check, report only:** at the GM2 and GM5 trunk entries of Addendum 16 §4, compare the FEMA full-grid area with the old grids. This is a third, independent check of decision 2; it does not change the decision.

## 3. Figures for visual review

All figures as PNG at 200 dpi, and PDF, in `reports/v2/figures/a17/`. Use the same hillshade everywhere: azimuth 315°, altitude 45°, no vertical exaggeration, computed from the tile's own 1 m filled grid.

1. **Overview map, one per site** (`<site>_overview`):
   - the hillshade of the full 1 m grid, block-averaged to 5 m (10 m for AP) for display;
   - the lidar footprint outline;
   - every eligible window of the draw as a thin grey outline;
   - every tile filled and labelled, coloured by kind: original, drawn, M&D by design, control;
   - at GM, the NCALM 2003 footprint outline;
   - at TTV, the clipped M&D polygon and the 77 digitized heads;
   - scale bar, north arrow, coordinates, CRS in the caption.
2. **Tile gallery, one per site** (`<site>_gallery`):
   - one panel per tile, all at the same 1 km extent and scale, in a grid of 6 columns (more than one page if needed);
   - each panel shows the hillshade with the channel network (A ≥ 10³ m² from the tile's area grid; line width scaled by log A);
   - panel title: tile name, kind, survey, z min–max, largest A inside, nearest-neighbour-filled fraction.
3. **Survey-control pairs** (`GM_control_pairs`): GM1–GM7, NCALM 2003 and FEMA 2018 side by side, same rendering.
4. **AP survey pairs** (`AP_pamap_3dep_pairs`): AP1–AP5, PAMAP against 3DEP, side by side.
5. Keep each PNG under 5 MB. Write a one-page index (`reports/v2/figures/a17/README.md`) listing the figures.

## 4. Part A report and STOP

One report, `reports/v2/<YYYYMMDDTHHMM>PST_register_figures.md`. Fixed outline:
1. State check.
2. Registration: unit checks and the final tile list.
3. GM control tiles: fetch, grid, units, cut, the area check of §2.2.4.
4. The figures: a list with paths, and any rendering choices.
5. Decisions needed: the tile exclusions (Steve's).

**Stop.** Push after the Addendum 2 §5 check. Do not start Part B until Steve's go-ahead, which lists any excluded tiles, is placed in `reports/directives/` beside this file.

# Part B (only after Steve's go-ahead)

## 5. Measurement plan

On every registered tile not excluded, plus the seven GM control tiles, measure with the v2 methods of `docs/measurement_methods.md` §0, unchanged:

1. **Concavity.**
   - Per-tile θ by χ–z (reported).
   - The site-pooled θ at A_c = 10³ and 10⁴ m², pooled and equal-weight, with 95 % basin-bootstrap bands.
2. **Site m from the model's misfit profile, S(m).** For every tile, the exponent profile MSE*(m) on the grid 0.05–0.90, step 0.01, with the 200-value A_crit grid, as in Addendum 4 §6.1. Then S(m) = Σ_t MSE*_t(m) / MSE*_t,min per site.
   - **The site m is the argmin of S(m)** (rule fixed now).
   - Report its 1 % and 5 % intervals, the leave-one-out minima, and the v2 site m beside it (GM 0.31, AP 0.47; AP was on PAMAP).
   - The GM control tiles do not enter S(m).
3. **k_s and K₀:** k_s by χ–z at θ = site m, over the network above 10³ m², pooled through-origin. Report n_eff. K₀ = E/k_s.
4. **Hilltop curvature and D₀:** C_ht on option (iii) hilltop pixels at L_f = 11 m, and at L_w (the largest tested window not exceeding the tile's median crest width). Also report the C_ht(L) curve. D₀,A = E/|C_ht(L_f)| and D₀,B = E/|C_ht(L_w)|.

## 6. Rules written now

1. **AP D₀ (the option (iii) window-stability test of Addendum 6 §1, unchanged).** If AP passes on the 3DEP tiles, AP D₀ is a measurement (D₀,A), as at GM. If it fails, the bracket remains. Apply the same test to GM (both surveys) and to TTV, and report each.
2. **Site m stability:** report whether the site m moves by more than 0.05 from its v2 value at GM (NCALM tiles only, and all GM tiles) and at AP. No action follows; it is for the record.
3. **AP survey comparison** (AP1–AP5, PAMAP against 3DEP, same methods and same m): report the ratios of k_s, C_ht and θ. Report only.
4. **GM pooling rule.** On GM1–GM7, compare FEMA against NCALM:
   - the GM tiles are pooled across the two surveys if the median FEMA/NCALM ratio of k_s and of |C_ht(L_f)| are both within ±15 %, and no single tile differs by more than ±30 % in either;
   - otherwise GM is reported by survey in Addendum 18, with the pooled result beside it.

## 7. Cost

Print the estimate first. The m-profiles dominate: about 1 core-hour per tile at the Addendum 4 grid, about 90 tiles in all. Throttle arrays to stay well under 300 concurrent CPUs. Expected ≤ 150 core-hours.

## 8. Part B report and STOP

One report, `reports/v2/<YYYYMMDDTHHMM>PST_measurement.md`. Fixed outline:
1. State check, and the exclusions with their reasons.
2. Concavity: per-tile θ, the site-pooled θ, S(m) and the site m per site, with the rule applied.
3. k_s and K₀ per tile.
4. Hilltop curvature, D₀, and the window-stability test per site (rule 6.1).
5. Survey comparisons: AP PAMAP against 3DEP; GM NCALM against FEMA, with rule 6.4 applied.
6. What the measurements support, and where they are weak.
7. Decisions needed for Addendum 18.
8. Appendices: all tables, the run ids, checksums, Slurm accounting.

**Stop.** Do not run the AD or partitioned joint fits of the expanded set (Addendum 18). Do not merge or tag. Push after the Addendum 2 §5 check.
