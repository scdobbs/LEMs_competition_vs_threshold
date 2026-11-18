# [2026-11-18 10:45 PST] Addendum 16: finish the tile set (TTV re-draw, GM lidar extension, the GM drainage-area rule, the D8 border guard). No measurement, no modelling

This addendum follows Addendum 15 and its report `reports/v2/20261118T1030PST_tiles_stage_b.md`. Continue on branch `v2.3-tiles`.

## 0. Orientation (you are a new agent instance)

### 0.1 The project in brief
- **Repo:** `scdobbs/LEMs_competition_vs_threshold` (public), package `lemcvt`. Clone at `~/LEMs_competition_vs_threshold-JGR-branch` on Sherlock.
- **The paper (Dobbs & Hilley, JGR-ES resubmission)** compares two steady-state landscape models fitted to 1 m lidar tiles by elevation MSE:
  - the co-located advection–diffusion (AD) model;
  - a threshold-partitioned model (stream power where A ≥ A_crit, linear diffusion elsewhere).

  For each tile it compares the fitted K and D with per-tile measured values, K₀ = E/k_s and D₀ = E/|C_ht| (paper Figs. 4–6).
- **Addenda 13–16 expand the tile set:**
  - Allegheny Plateau (AP, PA): 29 tiles on USGS 3DEP.
  - Gabilan Mesa (GM, CA): 7 tiles so far, limited by the NCALM 2003 lidar footprint.
  - Tennessee Valley (TTV, Marin County, CA): the Montgomery & Dietrich (1989, 1992) channel-head field area. Its tiles are provisional.
- **Steve Dobbs** reviews every report through a separate planning conversation. Directives reach you as addenda in `reports/directives/`.

### 0.2 Read, in this order
1. `reports/v2/20261118T1030PST_tiles_stage_b.md` (stage B: georeference, re-routing, draw, tiles). This is the essential one.
2. `reports/v2/20261117T2000PST_tiles_stage_a.md` (stage A: pools, screens, full grids, units).
3. `reports/directives/2026-11-18T0800PST_v2_addendum_15_tile_draw.md` (the settled selection rules, §1).
4. Addendum 4 §1, Addendum 2 §4–§5 and Addendum 7 §0.4 in `reports/directives/` (conventions, §0.4 below).
5. This file.

### 0.3 Where things are
- **Stage B run directory:** `$GROUP_HOME/lemcvt_results/20261118T160326Z_a15_stage_b/`:
  - `md_georef/`: `md1989_fig3.gpkg` and the masks;
  - `grid_<site>/`: the re-routed 1 m full grids;
  - `ttv_reference/`, `draw/`, `tiles/`, `tiles_ttv_provisional/`.
- **Stage A run directory:** `$GROUP_HOME/lemcvt_results/20261118T002225Z_a14_stage_a/` (`pools/` and the rest).
- **Copies of record:** `$GROUP_HOME/lemcvt_data/source/`.
- **Registered tiles:** `$LEMCVT_DATA/<version>/<tile>/`, listed in `configs/data.yaml` (`versions`) and `data/manifest.csv`.
- **Scripts:** `reports/v2/scripts/a14_*.py`, `a15_*.py`. Thresholds: `reports/v2/scripts/a14_thresholds_B.json`.
- **Papers:** `papers/` (never committed), including Montgomery & Dietrich (1989).
- **Environments:**
  - `lidar` (PDAL, laspy, rasterio, numba);
  - `lemcvt` (package, pools code, draw, cutting);
  - `ci312` (ruff).

### 0.4 Standing rules
- **Reports:**
  - one report per checkpoint, titled `[YYYY-MM-DD HH:MM PST]`, file `reports/v2/<YYYYMMDDTHHMM>PST_<name>.md`;
  - follow the fixed outline;
  - stop for review at the end.
- **Rules are set before results and never revised afterwards.** If a rule cannot be applied as written, report it and stop.
- **Do not fix numerical issues silently.** List every deviation in the state check.
- **Slurm:**
  - every job on `serc`, no `--account`, arrays throttled, ≤ 300 concurrent CPUs;
  - no submission over 300 core-hours of limits;
  - each job's time limit at most twice its own estimate (30 min floor);
  - print the estimate before submitting;
  - follow Sherlock and `serc` usage guidelines at all times (no heavy work on login nodes; data transfer on compute or transfer nodes).
- **Push** only after the Addendum 2 §5 pre-push check.
- **Never commit** `papers/`, PDFs, manuscripts, `working-with-steve.md`, `cry_for_help.txt`, point clouds or rasters. The repo is public.
- **Units:** read units from each file's header; convert US survey feet with 1200/3937 and international feet with 0.3048; never convert on an assumption.

### 0.5 Before starting
1. **Confirm** the state of `v2.3-tiles` against `origin`, that the tests pass on `serc`, and that no job is running.
2. **Confirm** the stage A and B paths above exist.
3. **Where this addendum and the repo disagree,** the repo wins. Report it and stop.

## 1. Decisions (settled by Steve on 2026-11-18)

1. **TTV study-area polygon.** Clip the digitized M&D polygon at the 61 m contour (the paper's lower bound, the 200 ft contour of the quadrangle). Take polygon ∩ {z ≥ 61 m on the TTV 1 m grid}, keep its largest connected component, and fill its holes. Then re-run rule 2.2 of Addendum 15. The area test (2.1 km² ± 15 %) is the binding check: the lower bound now passes by construction, so state that. The M&D windows (Addendum 15 §5.2) are re-chosen against the clipped polygon by the same rule.
2. **TTV steepness reference.** Use the **pooled through-origin χ–z slope** over the channel points of the clipped polygon (A ≥ 10⁴ m² on the 10 m NED, at the polygon's θ by the legacy χ method), not the median of per-basin slopes. **The reason, recorded now:** the per-basin median is taken over basins truncated by the polygon boundary, and fails to describe the area it is computed from (the M&D windows themselves lay at 4.6–7.1 × that value). The pooled slope is also the statistic the v2 measurement uses. |C_ht| keeps the stage A method. GM and AP keep their legacy references.
3. **TTV re-draw** under the Addendum 15 rules, with this reference. Seed 20261118, from scratch: the TTV tiles were not registered. The widening rule of Addendum 15 decision 3 applies once, to the new eligible count. The target is 30 tiles in all, including the M&D windows.
4. **GM drainage-area rule.** For modelling, a tile's drainage area must include the area draining into it from outside the lidar footprint.
   - **GM1–GM5:** use the existing tile area grids. They equal the new cuts on 99.8–100 % of cells and are complete on the trunk cells. Elevation and flow direction may come from either, since they are identical.
   - **New GM tiles, TTV tiles, AP tiles:** the full-grid areas, because they pass the "upstream area fully in valid data" rule (GM, TTV) or come from the whole watershed (AP).
   - **Check, written now** (§4): at every cell where a trunk of GM1–GM5 enters the lidar footprint, (old area − new area) must equal the area the 10 m coarse grid routes in from outside, within 10 %. If a tile fails, use the new area plus the coarse-grid inflow added at its entry cells, and report it.
5. **GM lidar extension.** Check whether newer public lidar covers Gabilan Mesa beyond the NCALM 2003 footprint (§3). If it does and adds at least 10 eligible windows, extend GM under the Addendum 15 rules. Otherwise report and stop that branch; GM stays at 7 tiles.
6. **D8 border guard** (§5): the lemcvt D8 must refuse, with a clear error, any grid whose valid cells touch the grid border without a designated outlet treatment. No existing v2 result may change.
7. **The M&D Table 1 source areas** are transcribed (§2.4) as the field channel-head data for the later A_crit comparison. The 77 digitized heads stay as they are.
8. **Budget, this addendum:** expected ≤ 30 core-hours; storage ≤ 60 GB new. Sherlock and `serc` guidelines are binding.
9. **Recorded for Addendum 17 (modelling):** Steve approved about 1,000–2,000 core-hours for the measurement and modelling of the expanded tile set, on condition that no Sherlock or `serc` guideline is broken. This addendum spends none of it.

## 2. TTV: polygon, reference, re-draw, Table 1

1. **Clip the polygon** (decision 1). Report:
   - area before and after;
   - the cells removed at each exit;
   - rule 2.2 of Addendum 15 (area, z_min, z_max), with the lower bound marked as "by construction".

   If the area fails, stop.
2. **The reference** (decision 2), on the 10 m NED and the clipped polygon:
   - θ by the legacy χ method;
   - k_s pooled through-origin at that θ, with the per-basin median beside it for the record;
   - |C_ht| by the stage A method.

   Recompute the TTV window statistics that depend on θ, as in Addendum 15 (deviation 4).
3. **Eligibility and re-draw** (decision 3).
   - Re-choose the M&D windows: the fewest non-overlapping windows covering at least 80 % of the clipped polygon.
   - Then apply the screens, the tolerances, the widening rule once, the gap rule, and the greedy draw with seed 20261118.
   - Report the ordered list with every skip reason.
   - Cut the tiles as in Addendum 15 §6, run the unit check, and **register** them as `TTV-x`. The provisional Addendum 15 TTV tiles stay in their run directory, unregistered.
4. **Transcribe M&D (1989) Table 1:** local valley slope, source area, source-basin length, and colluvial-fill width for every channel head.
   - Write `md_georef/md1989_table1.csv` with the source page.
   - Check the transcription: the row count, and the minimum and maximum source area against the paper's text and table. Report any value you cannot read.
   - Do not link rows to digitized heads; the paper gives no coordinates.

## 3. GM: is there more lidar?

1. **Footprints.** Check the public footprints of:
   - the 2018 FEMA Region 9 CA lidar (NOAA InPort 64466 and 64465, QL1);
   - the 2019 FEMA Estrella CA lidar (InPort 64364);
   - any USGS 3DEP work unit over Gabilan Mesa (the USGS 3DEP index or the Entwine `usgs-lidar-public` list).

   Report, for each, the area it covers within the Gabilan Mesa landscape, meaning the dissected mesa east of the Salinas River around the GM tiles. Report beyond the NCALM 2003 footprint and inside it separately, with density and collection dates.
2. **Gate, written now.** Continue only if one survey adds candidate windows that, under the Addendum 15 rules (set B screens, k_s and |C_ht| tolerance, gap rule, upstream area fully in that survey's valid data), give **at least 10 eligible windows** that do not overlap GM1–GM7. Compute this on the 10 m NED (download the 1/3″ cell or cells, state the product date) before downloading any point cloud. Otherwise report and stop this branch.
3. **If the gate passes:**
   1. Fetch the class-2 points (prefer the USGS Entwine copy if one exists, as in stage A).
   2. Grid them by Addendum 14 decision 9 (1 m, mean of class-2, NN fill ≤ 10 m, NAD83(2011) UTM 10N, EPSG:6339), and route on the full grid with the Addendum 15 gap rule.
   3. **Units and datum:** run the stage A rule 5.3 against the NED. Compare with NCALM 2003 on any overlap: expect an offset of about 30–34 m if NCALM is ellipsoidal and the new survey is NAVD88, and a Theil–Sen slope of 0.98–1.02. Stop on a slope near 3.28 or 0.305.
   4. **Eligibility:** apply the Addendum 15 rules to the combined GM pool (NCALM and new). Apply the widening rule once, to the combined count.
   5. **Draw:** GM1–GM7 are pre-placed. Draw from the eligible windows of the new survey, seed 20261119, written before the draw, with the same greedy acceptance. Target: up to 30 GM tiles in all.
   6. **Cut** as in Addendum 15 §6, register as a new version `GM-<survey>-x`, and report.

## 4. The GM drainage-area check (decision 4)

For GM1–GM5:
1. Find every cell where a trunk channel enters the lidar footprint upstream of the tile: the D8 path from the tile's trunk cells leaves valid lidar data.
2. At each such entry, compare (old tile area − new full-grid area) on the trunk cells just downstream of the entry with the area the stage A 10 m routing assigns to the inflow there.
3. **Pass** within 10 %. Report the table for each tile.
4. Then write the modelling area grid for each original (`<tile>_1m_area_model.tif`): the old grid where the check passes, or new area plus coarse inflow where it fails. Register it as role `area_model`. Do not overwrite anything.

## 5. The D8 border guard (decision 6)

1. **Audit** every code path in `lemcvt` that computes D8 directions, including flow routing on tiles, the resolution sweep's recomputed hydrology, and any measurement that routes. For each, state whether it can receive a grid with valid cells on its border, and how the border is treated today (outlets designated, padded, or neither).
2. **Add a guard:** where neither outlets nor padding are designated, raise a clear error naming the grid and the fix (pad, or pass designated border outlets). Do not change the D8 algorithm itself (the TopoAnalysis-compatible behaviour is kept).
3. **Add tests:**
   - the guard raises on an unpadded border-valid grid;
   - a padded grid routes with a closed budget;
   - every existing caller passes.
4. **Rule, written now:** the full test suite passes. A regression run of the AD and partitioned profiles on GM1 and AP1 (v2 data versions) reproduces the `v2.0.0` results exactly. If any v2 result changes, stop and report.

## 6. Report and stop

One report, `reports/v2/<YYYYMMDDTHHMM>PST_tiles_finish.md`, titled with its timestamp. Fixed outline:

1. State check.
2. TTV: the clipped polygon and rule 2.2; the reference; eligibility and the draw (ordered list); the tiles and their registration; Table 1 transcription.
3. GM lidar extension: footprints, the gate, and (if it passed) the data, units, eligibility, draw and tiles.
4. The GM drainage-area check and the modelling area grids.
5. The D8 border guard: audit, guard, tests, regression.
6. The final tile list per site: name, version, footprint, area source, and kind (original, drawn, or M&D by design).
7. What is weak, and decisions needed for Addendum 17.
8. Appendices: tables, configs, checksums, Slurm accounting.

**Stop for review.** Do not measure tile parameters, do not run any model beyond the §5 regression, do not merge or tag. Push `v2.3-tiles` after the Addendum 2 §5 check.
