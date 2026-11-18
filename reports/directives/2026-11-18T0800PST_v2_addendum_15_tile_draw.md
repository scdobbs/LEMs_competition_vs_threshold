# [2026-11-18 08:00 PST] Addendum 15: tile expansion, stage B (settled rules, M&D georeference, random draw, tile cutting). No measurement, no modelling

This addendum follows Addendum 14 and its report `reports/v2/20261117T2000PST_tiles_stage_a.md`. Addendum 4 §1 (timestamped titles, one report per checkpoint), Addendum 2 §4–§5 (Slurm, storage, push) and Addendum 7 §0.4 (submit guard) still apply. Continue on branch `v2.3-tiles`.

## 0. Orientation (you are a new agent instance)

### 0.1 The project in brief
- **Repo:** `scdobbs/LEMs_competition_vs_threshold` (public), package `lemcvt`. Clone at `~/LEMs_competition_vs_threshold-JGR-branch` on Sherlock.
- **The paper (Dobbs & Hilley, JGR-ES resubmission)** compares two steady-state landscape models fitted to 1 m lidar tiles by elevation MSE:
  - the co-located advection–diffusion (AD) model;
  - a threshold-partitioned model (stream power where A ≥ A_crit, linear diffusion elsewhere).

  For each tile it compares the fitted K and D with per-tile measured values, K₀ = E/k_s and D₀ = E/|C_ht|. These are paper Figs. 4–6.
- **Sites so far:** five 1 km tiles each at Gabilan Mesa (GM, CA) and the Allegheny Plateau (AP, PA).
- **Addenda 13–15 expand this** to about 30 tiles at GM and AP, and add Tennessee Valley (TTV, Marin County, CA), the Montgomery & Dietrich (1989, 1992) channel-head field area.
- **Steve Dobbs** reviews every report through a separate planning conversation; directives reach you as addenda in `reports/directives/`.

### 0.2 Read, in this order
1. `reports/v2/20261117T2000PST_tiles_stage_a.md` (stage A: pools, screens, full grids, units). This is the essential one.
2. `reports/directives/2026-11-17T1600PST_v2_addendum_14_tile_expansion.md` (the stage A directive).
3. `reports/v2/20261117T1445PST_data.md` (Addendum 13: data inventory, 3DEP checks, the AP5 offset).
4. `docs/measurement_methods.md` §0 (the v2 measurement methods, for context only; nothing is measured in this addendum).
5. Addendum 4 §1, Addendum 2 §4–§5 and Addendum 7 §0.4 in `reports/directives/` (conventions below).
6. This file.

### 0.3 Where things are
- **Stage A run directory:** `$GROUP_HOME/lemcvt_results/20261118T002225Z_a14_stage_a/`, with these subdirectories:
  - `pools/`: per-site windows CSV, pools JSON and conflict graphs;
  - `grid_AP/`, `grid_TTV/`: the 1 m full grids and routing products;
  - `ttv_inventory/`: includes the 10 m NED warped to EPSG:6339;
  - `footprints/`, `marin_select/`, `usgs_vs_pasda/`.
- **Thresholds:** `reports/v2/scripts/a14_thresholds_B.json` (set B, adopted). Stage A scripts: `reports/v2/scripts/a14_*.py`.
- **Copies of record:** `$GROUP_HOME/lemcvt_data/source/`:
  - `AP-3DEP/ept_class2_laz_utm17n/`, `TTV/Marin2019_B5b_ept_class2_laz_utm10n/`;
  - `TTV/NCALM_2006_be/`, `TTV/NED_13arcsec/`;
  - `Perron_coarse/`, `GM_original/Gabilan_mesa.tif`, `PA_original/`.
- **Papers:** `papers/` in the repo, never committed. They include Montgomery & Dietrich (1989), Heimsath et al. (1997), Perron et al. (2008, 2009, 2012) and the submitted manuscript.
- **Environments:**
  - `lemcvt`: pools, routing conventions, the package;
  - `lidar` at `$GROUP_HOME/scdobbs-envs/lidar`: PDAL, laspy, rasterio, numba, for point clouds and large grids.

### 0.4 Standing rules (from the earlier addenda; they still apply)
- **Reports:**
  - one report per checkpoint, titled `[YYYY-MM-DD HH:MM PST]`, file `reports/v2/<YYYYMMDDTHHMM>PST_<name>.md`;
  - follow the fixed outline;
  - stop for review at the end.
- **Rules are set before results and are never revised afterwards.** If a rule cannot be applied as written, report it and stop.
- **Do not fix numerical issues silently.** Report every deviation from this addendum in the report's state check.
- **Slurm and push:**
  - every job on `serc`, no `--account`, arrays throttled, no submission over 300 core-hours of limits;
  - print the cost estimate before submitting;
  - push only after the Addendum 2 §5 pre-push check.
- **Never commit** `papers/`, PDFs, manuscripts, `working-with-steve.md`, `cry_for_help.txt`, point clouds or rasters. The repo is public.
- **Units:** read units from each file's header; convert US survey feet with 1200/3937; never convert on an assumption.

### 0.5 Before starting
1. **Confirm** the state of `v2.3-tiles` against `origin`, that the tests pass on `serc`, and that no job is running.
2. **Confirm** that the stage A paths above exist.
3. **Where this addendum and the repo disagree,** the repo wins. Report it and stop.

**Why this addendum.** Stage A showed that the selection rules as first written leave almost no eligible windows, mainly because window-level θ on the coarse grids is noise. Steve has settled the rules (§1). This stage:
- georeferences the Montgomery & Dietrich (1989) map;
- applies the settled rules and draws the tiles at random;
- cuts the tiles, with drainage area from the full grids;
- then stops. Measurement and modelling come later.

## 1. Decisions (settled by Steve on 2026-11-18)

1. **Screens:** threshold set B (`scripts/a14_thresholds_B.json`):
   - largest drainage area inside the window ≤ 5 km²;
   - largest single inflow ≤ 2 km²;
   - χ–z R² ≥ 0.90;
   - slope–area ΔBIC ≤ 20;
   - NLCD classes 11, 22–24, 31 ≤ 10 %;
   - AP longwall panels = 0;
   - GM and TTV: upstream area fully in valid data.
2. **θ is dropped from the selection metrics.** Window-level θ on the coarse grids scatters by ±0.1–0.2, and the models use one site m (Addendum 12 showed that choice does not move the partitioned conclusions). Report each window's θ, but do not select on it.
3. **Tolerances:** k_s and |C_ht| within ±25 % of the site reference. **Widening rule, fixed now:** if a site has fewer than 25 eligible windows at ±25 % (TTV: fewer than 30), use ±50 % at that site. Count eligibility after the decision-8 gap exclusion and before the draw. Apply the rule once; no other widening.
4. **References:**
   - GM and AP: the legacy `TAYLOR_REF` values (GM k_s 1.9 at θ 0.292, |C_ht| 12.8 × 10⁻³ m⁻¹; AP k_s 47.2 at θ 0.54, |C_ht| 5.93 × 10⁻³ m⁻¹);
   - TTV: k_s and |C_ht| from the georeferenced M&D study-area polygon (§2), by the stage A code on the 10 m NED. This replaces the 0.67 km² proxy.
5. **TTV erosion rate:** E = 77 ± 15 m Myr⁻¹, the mean of the three basin-averaged detrital rates of Heimsath et al. (1997) (Creek1a 64, Creek1b 66, Creek2 102). Record the range 64–102. Absolute K and D scale with E; the K/K₀ and D/D₀ ratios and the MSEs do not.
6. **AP full-grid source:** the USGS Entwine copy stands. The stage A content test (same points within 1 cm; heights equal to PASDA's ftUS × 1200/3937; rule 5.1 to 2–8 mm) is the right test; the checksum test could not pass under reprojection.
7. **Targets:**
   - GM and AP: the original five plus up to 25 new;
   - TTV: up to 30, with the M&D windows included by design (§5.2).
   
   Fewer is accepted. Tiles never overlap.
8. **Water gaps in the 1 m grids:**
   - Gaps **not** connected to the footprint edge or the sea (ponds, river reaches inside the footprint) are filled by nearest neighbour for routing, so flow passes through them.
   - Gaps connected to the edge or the sea stay nodata and act as outlets.
   - Record the filled-gap mask. A drawn window is **excluded** if any cell of its channel network (A ≥ 10³ m² on the 1 m grid) is a filled-gap cell; the draw then moves to the next window in its random order (§5).
9. **The original five stay** at GM and AP, whatever the rules say. Report their status under every rule. AP5 needs no further action.
10. **Budget:** expected ≤ 40 core-hours. **Each job's time limit is at most twice its own estimate (minimum 30 min), and the sum of all limits is at most twice the expected total.** Every job on `serc`; arrays throttled.

## 2. Georeference the Montgomery & Dietrich (1989) map

Source: `papers/Montgomery_and_Dietrich_1989*` (WRR 25(8), 1907–1918), Fig. 3: channels, colluvial deposits and channel heads on a 40 ft contour base, with a 500 m scale bar. The study area is 2.1 km² on the eastern flank of Coyote Ridge, 61–314 m, on the USGS Point Bonita 7.5′ quadrangle.

1. **Extract** Fig. 3 at the highest resolution the PDF holds; record the page and pixel size.
2. **Ground control:** at least 8 points, spread over the map, at features identifiable on both the figure and a hillshade or contours of the TTV 1 m grid (contours at 40 ft ≈ 12.19 m). Use summits, saddles, contour re-entrants at channel junctions, the man-made pond, the road culvert, and the ridge crest. Fit an affine transform; use a second-order polynomial only if the affine RMS fails.
3. **Rule 2.1, written now:** the fit passes if the RMS residual is ≤ 30 m and every control point is within 60 m. If it fails, stop and report.
4. **Digitize** into a GeoPackage in EPSG:6339:
   - the study-area boundary;
   - every mapped channel head, with the paper's labels where the figure gives them (A–D, the culvert head);
   - the channels;
   - the colluvial-deposit polygons.
5. **Rule 2.2, written now:**
   - the polygon's area must be 2.1 km² ± 15 %;
   - its elevation range on the 1 m grid must be 61–314 m, ± 15 m at each end.

   If either fails, report and stop.
6. **Report:** the number of heads digitized. The paper reports 68 channelled hollows of 90; 5 were excluded for trails and 1 is the culvert head. Note which heads cannot be placed.

## 3. Re-route the AP and TTV full grids with the gap rule (decision 8)

- Classify every nodata gap in `AP_1m_*` and `TTV_1m_*` as enclosed or edge-connected. Fill the enclosed gaps by nearest neighbour, then fill depressions, route D8 and accumulate area on the whole grid, as in stage A.
- Keep the stage A products; write the new ones beside them.
- Report: the number and area of enclosed gaps filled; the largest basins; and the closed-budget check.
- GM: also route `Gabilan_mesa.tif` at 1 m as one full grid, with the stage A conventions. The tiles' drainage areas must come from it (stage A routed only the coarse grid at GM).

## 4. Eligibility under the settled rules

For each site, using the stage A candidates and screen statistics (no recomputation unless a rule needs it):
1. Apply decision 1 and the decision 3 tolerances at ±25 %, with the decision 4 references (TTV from §2).
2. Count the eligible windows. Apply the widening rule once where it triggers, and state which tolerance each site uses.
3. Report the counts per offset, and the original ten tiles' status under every rule.

## 5. The random draw

### 5.1 Procedure (fixed now)
- **Seed:** 20261118, written into the run config before the draw runs.
- **Ordering:** pool all eligible windows of a site across the four offsets, then random-permute them with the seed.
- **Pre-placed windows:** at GM and AP, the five original tiles. At TTV, the M&D windows of 5.2.
- **Greedy acceptance:** accept a window if it
  - overlaps no accepted or pre-placed window by any area;
  - shares no direct drains-through edge with one (stage A conflict graph, screen 4);
  - passes the decision-8 gap exclusion.

  Otherwise skip it. Stop at the target or when the list is exhausted.
- **Report:** the full ordered list with the reason for every skip, so the draw can be reproduced exactly.

### 5.2 TTV M&D windows
Place the fewest non-overlapping 1 km windows (from the stage A lattices) that together cover at least 80 % of the M&D polygon. These are included by design: report their screen and tolerance status, but do not apply the screens to them. If 80 % cannot be reached with non-overlapping windows, report the best coverage.

## 6. Cut the tiles

For every drawn tile, original and new:
1. **Cut** a 1,000 × 1,000-cell square on the site's 1 m lattice, with its south-west corner at the window's south-west corner snapped to the lattice. AP coarse windows are 990 m; take 1,000 m from the same corner.
2. **Products per tile:**
   - filled elevation (m);
   - flow direction and drainage area, **clipped from the full-grid routing**, never recomputed on the tile;
   - the class-2 count per cell, the nearest-neighbour fill mask and the filled-gap mask (AP, TTV).
3. **The original tiles:**
   - **AP:** cut AP-3DEP versions at the original footprints (reproject the EPSG:26917 footprints to the EPSG:6346 lattice), named `AP<n>_3dep`. The PAMAP tiles are not touched.
   - **GM:** cut new versions from the full-grid routing. Compare their drainage area with the existing tile area grids cell by cell, and report the differences. Do not replace the existing tiles.
4. **Unit check per tile:** z in metres; median and range plausible for the site (GM 100–500 m, ellipsoidal; AP 230–490 m; TTV 0–320 m). Any tile outside its range: stop.
5. **Register** the tiles in `configs/data.yaml` and `data/manifest.csv` as new data versions (`GM-x`, `AP-3DEP-x`, `TTV-x`). Data stay outside git.

## 7. Report and stop

One report, `reports/v2/<YYYYMMDDTHHMM>PST_tiles_stage_b.md`, titled with its timestamp. Fixed outline:

1. State check.
2. M&D georeference: control points, residuals, rules 2.1–2.2, the digitized layers, the head count (§2).
3. Re-routing with the gap rule, and the GM full grid (§3).
4. Eligibility per site, the tolerance used, the original ten tiles' status (§4).
5. The draw: the ordered list, the acceptances and skips, the final tile list per site, with a map per site (§5).
6. Tile cutting: products, the per-tile unit check, the GM drainage-area comparison (§6).
7. What is weak, and decisions needed.
8. Appendices: all tables, the seeds and configs, checksums.

**Stop for review.** Do not measure k_s, θ or C_ht on the 1 m tiles, do not run any model, do not merge or tag. Push `v2.3-tiles` after the Addendum 2 §5 check.
