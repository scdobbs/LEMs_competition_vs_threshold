# [2026-11-18 14:45 PST] Tile set finish (addendum 16): TTV re-draw on the clipped M&D polygon, the GM lidar extension gate, the GM drainage-area rule, the D8 border guard

Directive: `reports/directives/2026-11-18T1045PST_v2_addendum_16_tile_finish.md` (outline fixed by its section 6). Branch `v2.3-tiles`
(from 2e7a035, the stage B report). **Nothing was measured on the 1 m tiles; no model was run beyond the section 5 regression; no merge,
no tag.** Scripts: `reports/v2/scripts/a16_common.py`, `a16_ttv_clip.py`, `a16_ttv_reference.py`, `a16_md_table1.py`, `a16_gm_footprints.py`,
`a16_gm_gate.py`, `a16_gm_area_check.py`, `a16_gm_area_diag.py`, `a16_regression_task.sh`, `a16_regression_check.py`, `make_a16_tables.py`
(ruff clean); `a15_draw.py` and `a15_cut_tiles.py` gained backward-compatible options (section 1). Code: `src/lemcvt/hydro/d8.py` (the
guard), seven callers, `tests/test_hydro.py`, `docs/numerics_changes.md` N13. Environments: `lemcvt` (everything), `ci312` (ruff).
Every number is tied to the run directory `$GROUP_HOME/lemcvt_results/20261118T185206Z_a16_tile_finish/` (subdirectory named in each
section) or to a Slurm job id. Appendices (section 8) are machine-generated. **This is a stop for review.**

## Summary

| item | result |
| --- | --- |
| 1 state | `v2.3-tiles` = origin = 2e7a035; no job running; tests on `serc` 269 passed at 2e7a035 and **274 passed with the guard** (job 52834227). Deviations and the two stops are listed in section 1 |
| 2 TTV | Polygon clipped at 61 m: **1.943 km², rule 2.2 passes** (area binding; z_min 60.9 m through 12 hole-filled cells; z_max 310.1). Reference on the clipped polygon: **θ 0.442, k_s pooled 26.9** (median 18.6 beside it), \|C_ht\| 17.9 × 10⁻³. **34 eligible windows at ±25 % (no widening)**; the 3 M&D windows cover 92.7 %; draw (seed 20261118, target 27 drawn) **accepted 12 → 15 tiles**. **Cut stopped by the unit check** (addendum 15 §6.4 as written): TTV4, TTV5, TTV7, TTV9 reach 321-656 m, above the 0-320 m site window (the Mount Tamalpais flank, metres, not a units error); **nothing registered**. Table 1: 63 rows transcribed, every value found in the scan's OCR layer |
| 3 GM lidar | The 2018 FEMA Region 9 survey (InPort 64466 = USGS `CA_FEMA_Z4_B2_2018`, QL2, 5.6 pts m⁻² all returns, 2018-01-22 to 04-22) covers the whole mesa: 39.5 km² inside the NCALM footprint, 21.1 beyond it within the stage A grid, 667 km² beyond it within 15 km of the tiles. The QL1 Salinas block (64465) and Estrella 2019 (64364) are 0 km² within 15 km; the 2023 California Gaps 3DEP units likewise; ARRA 2010 (1.8 pts m⁻², 2010) covers 148 km² to the south-west. **Gate: the FEMA survey adds 48 eligible windows at ±25 % (62 combined, no widening) that do not overlap GM1-GM7: PASS**; ARRA 2010 would add 17 only at ±50 % and is not used |
| 4 GM area | Old − new area at the tiles' trunk entries against the stage A 10 m routing: GM2's trunk **2.7 %**, GM5's two trunks **1.3 % and 8.6 %** (pass); GM5's second northern trunk carries **2.29 km² that the 10 m strip routes in nowhere** (fail), GM1's 9,633 m² inflow against 5,500 (fail), and four entries below 500 m² fail trivially; GM3 has no difference. As the rule is written **only GM3 keeps its old grid**; GM1, GM2, GM4, GM5 get new + coarse inflow (registered as role `area_model`). Diagnostic on the NED routed over the whole 35 × 47 km box: GM5's missing trunk is **2.22 km² there (3 % from the old grid)**, so the failure is the strip's truncation, not the old grid; the old grids match one of the two coarse routings at every entry above 1 ha. **The GM5 `area_model` grid written by the rule drops 2.2 km² and should not be used until Steve decides (section 7)** |
| 5 D8 guard | Nine code paths audited; every one runs grids with valid border cells under the TopoAnalysis treatment and now declares it (`border="legacy"`); the guard raises on an undeclared border-valid grid; five new tests; **AD and partitioned forward runs on GM1 and AP1 reproduce the v2.0.0 MSEs exactly (max \|dz\| = 0) and every derived grid (filled, flow direction, area, log-area, hillshade) identical cell for cell** |
| 6 tiles | GM 7 + AP 29 registered (unchanged), TTV **0 registered** (15 cut, parked), GM extension: **23 drawn (30 GM tiles in all) but the cut stopped on the unit check** (10 tiles on the higher eastern mesa reach 497-695 m, above the 68-468 m window written for the NAVD88 grid): **0 registered**, 23 cut and parked |
| cost | 21 submissions (20 job ids, one array of 2), **4.7 core-hours used** (expected ≤ 30), sum of CPU-weighted limits 26.2 core-hours (every limit at most twice its estimate, 30 min floor), every job on `serc`, largest job 8 CPUs × 30 min (the fetch) and 4 CPUs × 100 GB (the grid); new storage on group storage: the two NED cells (0.9 GB) and the 292 class-2 LAZ (5.8 GB) as copies of record, 1.7 GB in the run directory (the 1 m extension grid 1.3 GB), 187 MB of parked tiles, 6 MB of area_model grids; scratch cleaned (26 MB left: job logs and metadata); `$GROUP_HOME` at 319.6 GB of 1 TB |

## 1. State check

**At the start (2026-11-18 10:45 PST, login node sh04-ln01; resumed at 11:10 on sh03-ln05 after a disconnection).**
- `v2.3-tiles` = `origin/v2.3-tiles` = 2e7a035 (the stage B report); no job was running (`squeue --me` empty).
- Tests on `serc`: 274 passed in 3 min 52 s with the guard and its five tests in place (job 52834227, 2 CPUs; 269 at 2e7a035 in stage B);
  ruff (`ci312`) clean on `src`, `tests` and the a16 scripts.
- The stage A and B paths of section 0.3 exist (`grid_<site>/`, `md_georef/`, `ttv_reference/`, `draw/`, `tiles/`, `tiles_ttv_provisional/`,
  `pools/`, `ttv_inventory/`, the copies of record, the papers, the three environments).
- Read in the order of section 0.2.

**Where the addendum and the repo or the data differ, and every deviation** (each handled as stated; three stops):
1. **STOP 1 (section 2.3): the TTV cut fails the unit check of addendum 15 §6.4** ("any tile outside its range: stop"). Four drawn tiles
   lie on the Mount Tamalpais flank north-west of the M&D area and reach 321, 656, 389 and 369 m against the 0-320 m window the rule
   gives TTV. The grid is in metres (stage A rule 5.2: Marin 2019 against NCALM 2006, Theil-Sen slope 0.99999); the window was set from the
   M&D area's 61-314 m, not from the Marin Headlands. The rule says stop, so the 15 tiles were cut but **not registered** (parked in
   `tiles/TTV-x_unregistered/`); the decision is Steve's (section 7).
2. **STOP 2 (section 4): the GM drainage-area check fails on four of the five originals** as written, so their modelling grids are the
   "new + coarse inflow" variant. For GM5 that drops 2.29 km² the old grid carries on its second northern trunk, because the stage A
   10 m routing is of Perron's 58 km² strip and the strip truncates that basin (section 4.3); the rule was applied as written and the
   NED routed over the whole box is reported beside it as a diagnostic, not as a substitute.
3. **Rule 2.2's "by construction" z_min** (decision 1): clipping at 61 m and keeping the largest component leaves 12 enclosed cells
   below 61 m that the hole filling re-admits (minimum 60.88 m). z_min therefore passes by construction up to the hole filling.
4. **The window statistic is unchanged** (decision 2 replaces the reference statistic only): each window's k_s is still the median of
   its per-basin slopes (addendum 15 rules, `ks_ref_median`), compared with the pooled polygon reference. The count with the windows'
   pooled slope is reported for the record (section 2.2).
5. **The provisional addendum 15 TTV tiles were under `$LEMCVT_DATA/v1/TTV-x/`**, not in the stage B run directory (the cut script writes
   the files before it registers). They were moved to `…/20261118T160326Z_a15_stage_b/tiles_ttv_provisional/TTV-x_unregistered/`
   (a rename on the same filesystem), as the directive says, before the new cut.
6. **`configs/data.yaml` carried the `versions` comment twice** (the stage B cut appended it on each of its two runs); de-duplicated,
   and `a15_cut_tiles.py` now writes it once.
7. **`a15_draw.py --ttv-target-total 30`** (decision 3: 30 tiles in all including the M&D windows; the addendum 15 default counted drawn
   tiles only) and **`a15_cut_tiles.py --run-label/--draw-note`** (provenance in `data.yaml`); defaults keep the stage B behaviour.
8. **The GM area check compares at the tile's trunk entry** (section 4.1): old − new is only defined inside the tile, and it is constant
   along a trunk between confluences, so its value at the first tile cell of a trunk is the outside inflow upstream of that cell; the
   cell where the trunk enters the lidar footprint is found by walking the 1 m main stem upstream and is reported with it.
9. **The "Gabilan Mesa landscape" is my polygon** (section 3.2, written into `a16_common.py` before the gate was computed): the mesa
   between the Salinas valley floor and the range front, within 15 km of the tiles' centroid; the gate count is also given for the
   15 km disc alone and the polygon alone.
10. **The gap rule cannot be applied at the gate** (it needs the 1 m grid); stated, not applied.
11. **"AD and partitioned profiles" of section 5.4** were run as the forward evaluations at the v2.0.0 optima (the release runs'
    `params.json`, 2 + 3 run keys per tile) with the derived grids, compared cell by cell with the release runs; the figure profiles
    themselves would be 2,770 solves.
12. **No `pyproj`, `requests` or PDF library in the environments**; the InPort pages, WESM and tile indexes were fetched with `curl`, and
    Table 1 was read from the page rendered with the poppler 0.47 build and checked against its OCR layer.
13. **STOP 3 (section 3.3.6): the GM extension cut fails the same unit check.** The window for the extension grid was written before the
    cut as the GM window less the 32 m ellipsoid-geoid offset of stage A rule 5.3 (68-468 m, `a15_common.Z_RANGE["GMF"]`, section 1
    of this list is the only place it is set); ten of the 23 drawn tiles, on the higher eastern mesa toward the range front, reach
    497-695 m. Real elevations (rule 5.3 and the NCALM overlap pass); as the rule is written nothing was registered; the 23 tiles are
    parked in `tiles_gmf/GM-FEMA2018-x_unregistered/`.
14. **The NCALM-overlap flag of `a16_gmf_check.py` tested the sign the wrong way round** ("NCALM above"): the offset is −33.85 m (NCALM,
    ellipsoidal, below the NAVD88 grid, as stage A's +32.27 m NED − NCALM says), within 30-34 m in magnitude; the json flag reads False;
    the script was corrected after the run and the number is reported as measured (section 3.3.3).
15. **Three script errors found and corrected by rerunning** (the wrong runs are kept beside the right ones): the diagnostic and the gate
    were first run with a crop box 1 km above the NED box (gate 52837804 cancelled after 10 min, diagnostic 52837838 superseded); the
    diagnostic then read the whole raster against a windowed transform (52840484 superseded by 52842281; the same slip was removed from
    two extension scripts before they ran); the first extension footprint took terminal basins and got the whole crop (52853733
    superseded by 52855272). Per-offset checkpointing was added to the gate script after its second run started (not used).
16. **Fetch footprint and lattice**: the windows' upstream catchments rather than terminal basins (section 3.3.1); the 1 m lattice is the
    footprint's 22 × 33 km envelope (727 M cells, 167 M valid), as `lattice_from_footprint` does.
17. `cry_for_help.txt`, `working-with-steve.md`, `papers/` stay untracked (pre-push check, section 8).

## 2. TTV: polygon, reference, re-draw, Table 1

### 2.1 The clipped polygon and rule 2.2 (decision 1)
Script `a16_ttv_clip.py` (job 52834763); products `md_georef/md1989_fig3_a16.gpkg` (the stage B GeoPackage copied with the layer
`study_area_clipped` added), `md_study_area_clipped_1m.tif`, `removed_cells_1m.tif`, `clip.json`, `rule_2_2_a16.json`, `clip_map.png`.
Appendix C1.

The stage B polygon (2.058 km², 2,058,028 cells at 1 m) intersected with z ≥ 61 m on the measured 1 m surface (stage A `TTV_1m_mean.tif`,
the grid rule 2.2 was evaluated on) loses 115,080 cells; the largest 8-connected component keeps 1,942,930 of the rest (five crumbs of
1-9 cells are dropped), and 12 enclosed cells are filled back: **1,942,942 cells, 1.943 km²**. The removed cells form two clusters,
the two valley exits of the stage B report: **88,532 cells (0.089 km²) at the southern exit** toward the trailhead (centroid 540,498 E,
4,190,141 N; 40.4-61.2 m) and **26,554 cells (0.027 km²) at the eastern exit** (540,947 E, 4,190,754 N; 36.9-61.1 m). Nothing was added.

| test | value | window | result |
| --- | --- | --- | --- |
| area | **1.943 km²** | 1.785-2.415 (binding) | **pass** |
| z_min on the 1 m grid | 60.9 m | 46-76 | pass, by construction (12 hole-filled cells at 60.88-61.0 m) |
| z_max on the 1 m grid | 310.1 m | 299-329 | pass |

The polygon chain is no longer provisional. The M&D windows re-chosen against the clipped polygon are the same three as in stage B
(section 2.3).

### 2.2 The reference (decision 2)
Script `a16_ttv_reference.py` (job 52834765), same code path as stage B (`a14_pools`: legacy χ θ, per-basin through-origin slopes
at that θ, |C_ht| at convex first-order divides of the A ≥ 10⁴ m² network on the 10 m NED), the polygon mask replaced by the clipped one
and the k_s reference taken as the **pooled** slope. Products `ttv_reference/TTV_reference.json`, `TTV_windows_a16.csv`,
`TTV_md_polygon_clipped_10m.tif`. Appendix C2.

| quantity | clipped polygon (this report) | stage B polygon | stage A proxy |
| --- | --- | --- | --- |
| θ (legacy χ) | **0.442** (coarse 0.450) | 0.437 | 0.387 |
| k_s pooled at θ (**the reference**) | **26.86** | 22.74 | - |
| k_s median of basins at θ (record) | 18.64 (16-84 %: 10.7-26.8) | 4.02 (the stage B reference) | 15.4 |
| basins / channel points | 24 / 1,716 | 33 | 15 |
| \|C_ht\| (m⁻¹) | **17.9 × 10⁻³** (1,953 convex divide pixels) | 17.3 × 10⁻³ | 15.1 × 10⁻³ |
| cells (10 m) / km² | 19,459 / 1.946 | 20,578 / 2.058 | 6,729 / 0.673 |

The pooled slope moves from 22.7 to 26.9 because the clip removes the two lowest reaches (the valley floors below 61 m, where χ-z
flattens); the per-basin median moves from 4.0 to 18.6 because the clip removes the shortest boundary-truncated basins at the exits.
The whole-grid pooled slope at θ = 0.442 is 22.4 (median 14.5, 932 basins). θ moved by 0.005, so the θ-dependent window statistics
(χ-z R², k_s at θ) were recomputed at 0.442 for all 420 candidates, as in stage B: R² ≥ 0.90 keeps 344 windows (343 at 0.437).

### 2.3 Eligibility and the re-draw (decision 3)
Script `a15_draw.py --sites TTV --ttv-target-total 30` (job 52834766), config written at 11:02:47 PST before the draw (Appendix C10).
Appendix C3 has the counts, the M&D windows and the full ordered list.

| candidates | set B screens | within ±25 % | **eligible at ±25 %** | eligible at ±50 % | widening (< 30) | tolerance used | per offset |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 420 | 100 | 111 | **34** | 68 | not triggered | **±25 %** | 5, 12, 9, 8 |

- With the pooled reference, 34 windows are eligible at ±25 % (stage B: 1 with the median reference, 37 forecast with the pooled one;
  the clip and the θ move account for the difference). The widening rule does not trigger (34 ≥ 30). The gap rule excludes none.
- **M&D windows**: 27 candidates intersect the clipped polygon; the fewest non-overlapping ones covering ≥ 80 % are the same three as
  in stage B (offset 0 window 135 at 539,000 E, 4,190,000 N; offset 2 windows 118 and 136 at 540,000 E, 4,190,500 N and 4,189,500 N),
  covering **92.7 %**. Their status, reported and not applied: only the third passes the screens (the first two fail ΔBIC); the first is
  within ±25 % of the new reference (k_s +13 %, |C_ht| −5 %), the second (k_s −27 %) and third (|C_ht| −25.4 %) just outside.
  Pre-placed as TTV1-TTV3.
- **Draw** (seed 20261118, the 34 eligible windows sorted by offset and window id, permuted once; target 27 drawn = 30 − 3):
  **12 accepted, 22 skipped** (12 overlap an accepted window, 8 share a drains-through edge with one, 2 overlap TTV1); the list was
  exhausted. **15 tiles in all.** The accepted windows lie east and south-east of the M&D area (6), north of it toward Mount
  Tamalpais (4: ranks 3, 10, 17, 19) and west of it (2); map `draw/TTV_draw_map.png`.

### 2.4 The tiles: cut, unit check, registration — STOP 1
Script `a15_cut_tiles.py --sites TTV` (job 52834767); records `tiles/tiles.json`, `tiles/tiles.csv`; the 15 tiles' files in
`tiles/TTV-x_unregistered/` (53 MB). Appendix C4 and C11.

All 15 tiles were cut from the stage B re-routed grid (1,000 × 1,000 m, every cell valid, no outlet cell inside, no filled-gap cell
on a channel). **The unit check fails on four**: TTV4 (543,000 E, 4,188,500 N) reaches 321.1 m, TTV5 (533,500 E, 4,195,500 N) spans
445-656 m, TTV7 (535,000 E, 4,193,000 N) 164-389 m, TTV9 (533,000 E, 4,193,500 N) 129-369 m; the other eleven lie within 26-318 m.
The four are on the south-west flank of Mount Tamalpais (summit 784 m) and in the hills above Muir Beach: real elevations in metres
(rule 5.2 of stage A fixed the vertical unit and datum of the Marin grid), outside a window that was written for the M&D area. As
addendum 15 §6.4 is written ("any tile outside its range: stop") **nothing was registered**: `configs/data.yaml` and `data/manifest.csv`
have no TTV-x version. The cut is reproducible in 40 s once the window is settled (section 7, decision 1).

### 2.5 Montgomery & Dietrich (1989) Table 1 (decision 7)
Script `a16_md_table1.py` (login node, 2 s); products `md_georef/md1989_table1.csv`, `md1989_table1_check.json`. Appendix C5 has the
63 rows.

Table 1 is on journal page 1910 (PDF page 4), right-hand column: local valley slope above the channel head (m/m, Brunton compass),
source area (m², average of several measurements rounded to the nearest 100 m²), source-basin length (m, nearest 10 m) and width of the
colluvial fill at the channel head (m). The page was rendered at 200 dpi with the poppler 0.47 build on Sherlock and read in three
crops; the 63 rows were then checked against the scan's OCR text layer: **every (area, length, width) triple and every slope value occurs
in the layer; no value was unreadable**. Checks: **63 rows = 68 channelled hollows − 5 trail-affected heads** (the paper's text: "Twenty-two
of the 90 hollows ... had no channel. In the remaining 68 ... Five hollows where hiking trails concentrate runoff ... were omitted");
source area **1,200-40,100 m²** (median 7,400), slope 0.24-0.91, length 50-400 m, width 5-60 m. The rows carry no coordinates and are
not linked to the 77 digitised heads, which stay as they are.

## 3. GM: is there more lidar?

### 3.1 Footprints (section 3.1)
Script `a16_gm_footprints.py` (job 52835250: the two NED cells, 0.9 GB, downloaded in 60 s and copied to
`$GROUP_HOME/lemcvt_data/source/GM_NED_13arcsec/` with manifest entries; warp; binning of the 1 m lidar; polygons); products in
`gm_footprints/`: `GM_NED_10m_utm10n.tif` (50 × 50 km box, EPSG:32610, 10 m, bilinear; product Last-Modified 2025-08-27),
`GM_ncalm_valid_frac_10m.tif`, `GM_perron_valid_10m.tif`, `GM_survey_<name>_10m.tif`, `GM_noaa_<id>_10m.tif`, `footprints.json`,
`gm_footprints_map.png`. Sources: the NOAA InPort pages (64466, 64465, 64364) and their Digital Coast tile indexes (IDs 9269, 9270,
9263), the USGS Entwine `usgs-lidar-public` boundaries and `ept.json` files, the USGS WESM work-unit table (2026-11-18). Appendix C6.

| survey | collection | QL / spacing | density (pts m⁻², all returns, whole survey) | inside NCALM (km²) | beyond NCALM, in the stage A grid | beyond NCALM < 5 / 10 / 15 km of the tiles |
| --- | --- | --- | --- | --- | --- | --- |
| **2018 FEMA Region 9, CA** (InPort 64466 = NOAA 9269 = USGS `CA_FEMA_Z4_B2_2018`) | 2018-01-22 to 04-22 | QL2, NPS 0.7 m | 5.6 | **39.5 (all)** | **21.1 (all)** | **45 / 275 / 667** |
| 2018 FEMA Region 9 QL1 Salinas block (64465 = NOAA 9270 = `CA_FEMA_Z4_QL1_2018`) | 2018-02-16 to 02-18 | QL1, NPS 0.35 m | - | 0 | 0 | 0 (the block is at 36.17-36.31° N, 25 km north) |
| 2019 FEMA Estrella (64364 = NOAA 9263 = `CA_FEMAR9Estrella_2019`) | 2019-12-15 to 2020-01-04 | QL2, NPS 0.71 m | 4.7 | 0 | 0 | 0 (its western edge at −120.71° is 12 km east of the box's 15 km ring) |
| USGS 3DEP `CA_CaliforniaGaps_1_B23` / `_3_B23` | 2023-10/11, 2023-09/12 | QL1 | 11.3 / 17.8 | 0 | 0 | 0 (north-east corner of the box only / none) |
| ARRA Central Coast Z4 2010 (`ARRA-CA_CentralCoast-Z4_2010`, legacy, "does not meet" 3DEP) | 2010-08-17 to 09-14 | - | 1.8 | 9.5 | 1.5 | 1.5 / 74 / 148 (south-west, across the Salinas) |

The NOAA index of 64466 and the USGS Entwine polygon of `CA_FEMA_Z4_B2_2018` give the same areas on the lattice to 0.1 km²: they are
the same survey (NAD83(2011) UTM 10N in feet / NAVD88 GEOID12B on the USGS side; the NOAA copy is in metres, GEOID18). It covers the
whole Gabilan Mesa. The NCALM 2003 footprint is 39.5 km² (39.3 fully valid at 10 m); the stage A coarse grid is a 60.5 km² strip
around it (map: `gm_footprints_map.png`).

### 3.2 The gate (section 3.2)
Script `a16_gm_gate.py` (job 52840477, 1 h 46 min; a first run, 52837804, was cancelled after 10 min for a crop-box error in the
diagnostic's sibling code, section 1); products `gm_gate/gm_gate.json`, `<pool>_windows.csv` (NCALM2003, CA_FEMA_Z4_B2_2018,
ARRA-CA_CentralCoast-Z4_2010), `GM_landscape_10m.tif`, `GM16_nlcd2021.tif`, `gm_gate_map.png`. Appendix C7.

**The landscape, written before the count** (`a16_common.GM_LANDSCAPE_POLY_UTM`, section 1 deviation 9): the dissected mesa between the
Salinas valley floor and the range front as seen on the NED hillshade, a 14-vertex polygon (772 km²), intersected with the 15 km disc
about the tiles' centroid (707 km²): **473 km², of which 434 beyond the NCALM footprint**. The NED crop (35 × 47 km, 683-718 km E,
3,958-4,005 km N) was masked to each survey's footprint and routed with the stage A conventions; the four stage A lattice offsets
gave the candidate windows (every cell valid in the masked grid and inside the landscape); set B screens, the legacy GM reference
(θ 0.292, k_s 1.9, |C_ht| 12.8 × 10⁻³) at ±25 % and ±50 %, "upstream fully valid" in the survey's own data, NLCD 2021 fetched for the
crop; the gap rule needs the 1 m grid and is applied in section 3.3. The baseline pool (the NED masked to the NCALM footprint, same
lattice) gives 83 candidates, 9 eligible at ±25 % and 22 at ±50 % (stage B on Perron's grid: 14 and 33; the lattices differ by 1 m
and the NED by its vintage).

| survey | candidates in the landscape | screens pass | eligible ±25 % | not overlapping GM1-7 | **added** (not in the NCALM baseline) | beyond NCALM | combined with the baseline at ±25 % | tolerance by the widening rule | **count for the gate** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **2018 FEMA Region 9** (`CA_FEMA_Z4_B2_2018`) | 1,698 | 460 | **62** | 49 | **48** | 48 | 62 ≥ 25 | **±25 %** | **48 ≥ 10: PASS** |
| ARRA 2010 | 172 | 35 | 7 | 7 | 7 | 7 | 16 < 25 | ±50 % | 17 ≥ 10: pass |

Under the disc alone the FEMA survey has 77 added windows at ±25 % (91 eligible of 2,591 candidates), under the polygon alone 58
(72 of 2,833); at ±50 % 189 under the landscape. The eligible windows lie all over the mesa (map): south-east of the NCALM footprint
along the mesa's axis (the largest cluster), north-west of it toward Gonzales, and across the Salinas on the ARRA side (excluded by the
polygon). **The gate passes for the 2018 FEMA survey at ±25 %; the extension (section 3.3) uses it.** ARRA 2010 passes only at the wide
tolerance, is a 2010 legacy survey at 1.8 points m⁻² that "does not meet" the 3DEP specification, and is not used.

### 3.3 Extension (section 3.3)
Scripts `a16_gm_ext_footprint.py` (job 52855272; a first version, 52853733, took the terminal basins as stage A did at TTV and got the
whole crop, 1,362 km², because every window drains to the Salinas), `a14_ept_fetch.py` (job 52856554), `a14_grid_route.py --site GMF`
+ `a15_reroute.py --site GMF` + `a16_gmf_check.py` (job 52856556), `a16_gm_draw.py` + `a15_cut_tiles.py --sites GMF` (job 52856563),
the copy of record (job 52858552). Products `gm_ext/`, `ept_GMF/`, `grid_GMF/`, `draw_gmf/`, `tiles_gmf/`; data under
`$LEMCVT_DATA/GM-FEMA2018-x/`; copies of record `$GROUP_HOME/lemcvt_data/source/GM_FEMA2018/ept_class2_laz_utm10n/` (292 LAZ,
sha256 manifest). Appendices C14-C16.

1. **Footprint and fetch (3.3.1).** The 49 eligible windows (±25 %, not overlapping GM1-GM7, in the landscape: 37.75 km²) and their
   upstream catchments on the survey-masked NED routing (34.7 km²), buffered by 500 m: **166.8 km²** in a 22 × 33 km box (693-715 km E,
   3,964-3,997 km N). 292 1-km boxes were read from the Entwine copy of `CA_FEMA_Z4_B2_2018` at full resolution, class 2 only,
   reprojected to EPSG:6339 and written as LAZ with the compound CRS EPSG:6339+5703 (the WESM record gives the source's vertical CRS as
   NAVD88 GEOID12B, in US survey feet; the Entwine copy carries EPSG:3857 and no vertical CRS; that its Z is in metres is established by
   the checks of 3.3.3, the evidence string is in the run metadata): **5.34 GB in 414 s (12.9 MB s⁻¹), 292 of 292 boxes**.
2. **Grid and routing (3.3.2).** Lattice 32,681 × 22,261 cells at 1 m (origin 692,935 E, 3,997,155 N, EPSG:6339; 727 M cells, 166.9 M inside the footprint); each
   cell the mean of its class-2 points (983.7 M points; **3.41 points m⁻², 5-50-95 % of points per cell 1-3-8**; 96.9 % of the cells inside
   the footprint hold a ground point); 5.18 M empty cells (3.1 %) filled by nearest neighbour within 10 m (median 1 m, maximum 10 m), 8,321
   cells (0.005 %) left nodata. Flight dates (adjusted GPS time, inferred from magnitude, the header flag unset): **2018-02-07 to
   2018-02-16**. Gap rule: 24 nodata components, **23 enclosed (0.126 km², one above 1 ha) filled for routing**, the rest edge-connected;
   priority flood 76 s, D8 57 s, area 72 s; **167.0 M valid cells, 240,844 data-edge outlets whose areas sum to the valid count (closed
   budget)**; largest basins 7.44 km² (700,408 E, 3,975,310 N) and 6.16 km² (707,600 E, 3,970,000 N); no pad needed (the lattice carries the
   5 m pad of `lattice_from_footprint`); peak 37.3 GB. Grid stage 309 s, 29.9 GB.
3. **Units and datum (3.3.3).** **Rule 5.3 against the 10 m NED** (both NAVD88; the EPSG:32610 → EPSG:6339 shift is the identity offline): on 711,470 fully
   covered 10 m cells, median NED − grid **+0.058 m** (5-95 %: −0.81 to +0.89 m, RMS 0.51), Theil-Sen slope **0.99995** (0.9999-1.0000):
   **pass**. **NCALM 2003 on the overlap** (465,568 sampled cells, every fifth NCALM cell, bilinear in the ground grid): median
   NCALM − grid **−33.85 m** (5-95 %: −34.13 to −33.59; the stage A rule 5.3 gave NED − NCALM = +32.27 m on 10 m block means), Theil-Sen
   slope **1.0012** (1.0011-1.0012), OLS 1.0011: the NCALM grid is ellipsoidal and lies 33.8 m below the NAVD88 survey, as expected;
   **no slope near 3.28 or 0.305, no stop**. The check script's own flag `offset_in_30_34m` came out False because it tested the sign
   the wrong way round (NCALM above, not below); the magnitude is within the window; the script was corrected after the run (section 1).
4. **Eligibility on the combined pool (3.3.4).** The gate already applied the addendum 15 rules to the combined pool on the NED (the
   baseline's 9 and the survey's 62 eligible windows at ±25 %: 62 combined, so the widening rule does not trigger and ±25 % stands). On
   the 1 m grid the decision-8 gap exclusion removed **none** of the 49 gate-eligible windows (no channel cell of A ≥ 10³ m² in a filled
   enclosed gap; the 23 enclosed gaps are stock ponds and road cuts of 0.13 km² in all) and none fell outside the 1 m grid:
   **49 eligible windows** (9, 9, 16, 15 per offset).
5. **Draw (3.3.5).** Seed **20261119**, written to `draw_gmf/draw_config.json` at 14:17:09 PST before the draw; GM1-GM7 pre-placed;
   the 49 eligible windows sorted by (offset, window) and permuted once; greedy acceptance with the stage B independence rules
   (overlap; direct drains-through edges on the survey-masked NED routing, within an offset the stage A rule, across offsets and
   with the pre-placed tiles the first-entered rule; gap exclusion); target 23 = 30 − 7. **23 accepted at ranks 1-49, 26 skipped**
   (15 overlap an accepted window, 11 share a drains-through edge: 3 with GM1, GM4 or GM5, 8 with accepted windows); the list was
   exhausted exactly at the target. The 23 lie across the mesa (map `draw_gmf/GMF_draw_map.png`): south-east of the NCALM footprint along
   its axis (GM8, GM9, GM11, GM13, GM15, GM16, GM24, GM27, GM29), north and north-east toward the range front (GM12, GM14, GM18, GM21, GM23,
   GM28, GM30), north-west toward Gonzales (GM19, GM22, GM25), and south-west at the Salinas edge (GM10, GM17, GM20, GM26). Appendix C15 has
   the ordered list with every skip reason.
6. **Cut and registration (3.3.6) — STOP 3.** `a15_cut_tiles.py --sites GMF` cut the 23 tiles (1,000 × 1,000 m on the EPSG:6339
   lattice; filled elevation, flow direction and area clipped from the full-grid routing, class-2 count, nearest-neighbour and
   filled-gap masks; every cell valid, no outlet cell inside, no filled-gap channel cell; class-2 density 2.85-3.75 points m⁻² per tile,
   nearest-neighbour-filled fraction 0.4-8.5 %). **The unit check fails on ten**: GM12, GM14, GM15, GM18, GM21, GM23, GM27, GM28, GM29 and
   GM30 reach 497-695 m (medians 379-641 m) against the 68-468 m window written before the cut (section 1, deviation 13); the other
   thirteen lie within 182-444 m. The failing ten are the tiles on the higher mesa east and north-east of the NCALM footprint; the
   window was the NCALM range shifted to NAVD88 and the extension reaches a higher part of the same landscape. As the rule is written
   **nothing was registered** (`configs/data.yaml` and `data/manifest.csv` have no `GM-FEMA2018-x` version); the files are parked in
   `tiles_gmf/GM-FEMA2018-x_unregistered/` (134 MB) and the cut is reproducible in a minute once the window is settled (section 7).
   Appendix C16 has the per-tile table and checksums.

## 4. The GM drainage-area check and the modelling area grids (decision 4)
Script `a16_gm_area_check.py` (job 52835252); products `gm_area_check/gm_area_check.json`, the five `GM-x/GM<n>/GM<n>_1m_area_model.tif`
(registered, role `area_model`); diagnostic `a16_gm_area_diag.py` (job 52837838), `gm_area_diag/gm_area_diag.json`,
`GM1/GM2/GM5_area_channels.png`. Appendix C8.

### 4.1 Entries
For each original, D = old − new (the existing tile area grid minus the stage B re-cut from the lidar-only routing) is the area that
enters the lidar footprint from outside and reaches the cell; it is constant along a trunk between confluences with other outside-fed
tributaries. The **tile entries** are the ring cells with D > 0 whose receiver lies inside the tile (the most upstream such cell of each
trunk); from each, the 1 m main stem was walked upstream on the stage B full-grid routing (largest-area donor at every step) to the cell
where the trunk enters the lidar (the walk ends one cell inside the data-edge ring, whose cells carry code 0). Eight entries in all:
GM1 one (its north-east corner, D = 9,633 m²), GM2 two (the trunk, D = 4.090 km², and a 451 m² side entry), GM3 none, GM4 one (16 m²,
one edge cell), GM5 four (two northern trunks, D = 1.554 and 2.287 km², the eastern trunk, 0.245 km², and a 20 m² cell).

### 4.2 The coarse inflow and the check
On the stage A 10 m routing of Perron's grid, the area each cell receives from outside the NCALM footprint was taken as A − A_inside,
A_inside accumulated on the same D8 with the outside cells carrying no area; at each tile entry the check compares D with that
quantity at the matching 10 m cell (maximum over the 3 × 3 neighbourhood).

| tile | entry (E, N) | old (km²) | new (km²) | **D = old − new** | **coarse inflow (strip)** | rel. diff | NED-box inflow (diagnostic) | result |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GM1 | 701,991 E, 3,981,150 N (north-east corner) | 0.908 | 0.899 | 0.0096 | 0.0055 | −43 % | 0.0087 (−10 %, pass) | **fail** |
| GM2 | 699,265 E, 3,981,451 N (side entry) | 0.149 | 0.148 | 0.0005 | 0 | −100 % | 0.134 (a 10 m basin the 1 m routing sends elsewhere) | fail (negligible) |
| GM2 | 699,437 E, 3,981,451 N (**the trunk**) | 8.452 | 4.362 | **4.090** | **3.978** | **−2.7 %** | 5.272 (+29 %) | **pass** |
| GM3 | - | - | - | 0 | - | - | - | pass (no difference) |
| GM4 | 695,066 E, 3,977,451 N | 0.579 | 0.579 | 0.00002 | 0 | −100 % | 0 | fail (one edge cell) |
| GM5 | 700,123 E, 3,982,451 N (**northern trunk 1**) | 1.763 | 0.209 | **1.554** | **1.534** | **−1.3 %** | 2.372 (+53 %) | **pass** |
| GM5 | 700,334 E, 3,982,451 N (**northern trunk 2**) | 2.667 | 0.380 | **2.287** | **0** | **−100 %** | **2.217 (−3.1 %, pass)** | **fail** |
| GM5 | 699,992 E, 3,982,354 N (west edge) | 0.044 | 0.044 | 0.00002 | 0 | −100 % | 0.0004 | fail (negligible) |
| GM5 | 700,991 E, 3,981,481 N (**eastern trunk**) | 1.927 | 1.682 | **0.245** | **0.224** | **−8.6 %** | 0.232 (−5.5 %, pass) | **pass** |

### 4.3 What the failures are
The 1 m main-stem walk from every trunk entry ends at the lidar's data-edge ring north and east of the tiles (GM2's and GM5's trunks
enter at 702,643 E, 3,982,943 N and 700,553 E, 3,982,844 N on the grid's northern edge); the 10 m channel of the strip routing is more
than 15 m from those 1 m entry cells (its 3 × 3 neighbourhood holds 200-800 m²), so the check is made at the tile entries, where the
two routings coincide.

**What passes and what fails.** The three large trunks that the strip resolves agree with the old grids to 1.3-8.6 %: GM2's trunk
(4.09 against 3.98 km²), GM5's first northern trunk (1.55 against 1.53) and GM5's eastern trunk (0.245 against 0.224). The old grids
therefore carry exactly the outside area the stage A routing routes in, where the strip holds the basin. The failures are of three
kinds:
1. **GM5's second northern trunk** (old 2.67 km², new 0.38, D = 2.29): the strip routes **nothing** in from outside at that cell. Its
   valid area is a 60.5 km² band around the lidar (gm_footprints map, yellow), and the basin that feeds this trunk lies north of the
   band: truncated at the band's edge, it never reaches the entry. The diagnostic routing of the NED over the whole 35 × 47 km box
   (`a16_gm_area_diag.py`: every valid cell, data-edge outlets, the same inside-only accumulation against the NCALM footprint) gives
   **2.22 km² at that entry, 3.1 % from the old grid's 2.29**. The old grid is right; the rule's coarse grid is not wide enough to
   show it.
2. **GM1's north-east corner** (D = 9,633 m²): the strip gives 5,500 m² (−43 %), the box routing 8,700 (−10 %, pass). A 1 ha inflow
   measured on 100 m² cells; the 10 % rule is tight at that scale.
3. **Four entries below 500 m²** (GM2 451 m², GM4 16 m², GM5 20 m², and nothing at GM3): one to five 1 m cells of edge drainage that
   no 10 m routing resolves; they fail the relative test by construction and matter to nothing.

The box routing is a diagnostic, not the rule's grid: where the strip matches the old grid (GM2's trunk, GM5's first trunk) the box
disagrees by +29 % and +53 %, because the current 1/3″ product (2025, resampled from the 2018 lidar) drains the mesa north of the
footprint differently from Perron's 2008 grid, from which the legacy pipeline's areas evidently descend. So the two coarse grids
disagree with each other, and the old tile grids agree with one or the other at every entry that matters.

### 4.4 The modelling grids
Written as the rule says: GM3 (no difference) keeps the old grid; GM1, GM2, GM4 and GM5 get the new grid plus the strip's inflow at each
entry, propagated along the tile's D8 downstream of it. Registered as role `area_model` (five manifest rows, a note in the `GM-x` block of
`configs/data.yaml`); nothing was overwritten. For GM2 this is 4.80 + 3.98 = 8.78 km² at the outlet against 8.89 old (−1.3 %); for GM5
it is 2.73 + 1.53 + 0.22 = 4.49 km² against 4.73 old at the outlet, with the second northern trunk at 0.38 km² against 2.67 old.
**Decision needed (section 7)** before the GM5 grid is used.

## 5. The D8 border guard (decision 6)

### 5.1 Audit
Every code path that computes D8 directions (`d8_flow_direction`), whether it can receive a grid with valid cells on its border, and how
the border was treated before this addendum:

| code path | grid it routes | valid cells on the border? | treatment before | now |
| --- | --- | --- | --- | --- |
| `analyses/outputs.derive_model_outputs` (every model surface written by `tile run`, `tile forward`, the figures) | the 1 km tile surface | yes, every border cell | neither: TopoAnalysis codes (least-uphill at the border) | declared `legacy` |
| `analyses/resolution_sweep.run_at_target_dx` | the tile resampled to dx | yes | neither | declared `legacy` |
| `analyses/profiling.run_profile` | the tile | yes | neither | declared `legacy` |
| `measure/legacy_v1.legacy_curvature` | the block-mean tile | yes | neither | declared `legacy` |
| `analyses/seamfix.classify_seam`, `analyses/ap_units.d8_check` | the tile (metres and feet) | yes; only interior cells with a margin are classified | neither | declared `legacy` |
| `analyses/dinf_diagnostics._padded_hydro`, `tests/synthetic_tile.hydro_from_padded` | reflect-padded synthetic grids whose margins are tilted outward | yes | padding by construction drains outward; codes at the border still TopoAnalysis | declared `legacy` |
| `reports/v2/scripts/a14_pools.route_full_grid` (stage A and B coarse routing, the TTV reference, the draw, the gate) | coarse grids with nodata | yes where the NED reaches the grid edge | **outlets**: every data-edge cell set to code 0 after the call | declared `outlets` (identical codes) |
| `reports/v2/scripts/a14_bigroute.d8_chunked` (the 1 m full grids of stages A and B) | row slabs of the full grid | yes (slab rows, and the grid's first/last columns) | halo rows dropped; the caller zeroes every data-edge cell; GM needed the 2-cell nodata pad of stage B | declared `legacy` on the slabs (the caller's outlet rule is unchanged) |
| the stored tile grids (`load_tile`: `*_fd.tif`) | not computed; clipped from larger grids | - | - | - |
| the D-infinity routines, the heads and width analyses | take the D8 codes as input or route synthetic grids by topological accumulation | - | - | - |

So every v1/v2 model-surface grid was routed with its border cells under the TopoAnalysis treatment (an inward least-uphill code; the
last two rows and columns never compared eastward or southward). The stored tile flow directions, which the models and measurements
use, are not affected (they were clipped from larger grids and their edges are outlets); the derived grids of model surfaces (the
`_flow_direction`, `_area`, `_logarea` files of every run; the slope-area figure and the resolution sweep use them) are.

### 5.2 The guard
`d8_flow_direction(z, border=None, outlets=None, name="grid")`: if any finite cell lies on the grid border and neither `border` nor an
`outlets` mask covering every such cell is given, it raises `D8BorderError` naming the grid, the count of border cells and the three
fixes (pad with NaN so that no valid cell touches the border; `border="outlets"`, every data-edge cell gets code 0, the convention of
the addendum 14-15 full-grid routing, with the warning that the second-to-last row and column keep the quirk unless the grid is padded;
`border="legacy"`, the TopoAnalysis behaviour kept knowingly). Helpers `valid_border_cells`, `data_edge_cells`, `check_border`,
`d8_area_budget` (root-area sum against the valid-cell count, two-cell cycle count). The algorithm is untouched: with `border="legacy"`
the codes are those of before, bit for bit (`tests/test_hydro.py::TestTopoAnalysisEquivalence` still passes against TopoAnalysis).
`docs/numerics_changes.md` N13 records it.

### 5.3 Tests
`tests/test_hydro.py::TestBorderGuard` (5 tests): the guard raises on an unpadded border-valid synthetic grid with a message naming the
grid and the fixes; an outlets mask must cover the whole border; a NaN-padded grid passes the guard undeclared and, with
`border="outlets"`, routes with a closed budget (1,600 of 1,600 cells at the roots, no two-cell cycle); `legacy` codes equal the
`outlets` codes away from the data edge and the legacy border cells carry non-zero codes; a NaN cell still gets code 0. Every existing
caller passes: **274 tests pass** on `serc` (job 52834227).

### 5.4 Regression against v2.0.0 (rule of section 5.4)
Scripts `a16_regression_task.sh` (array 52837608, GM1 and AP1, 2 min 22 s each) and `a16_regression_check.py` (job 52838424); runs
`20261118T185206Z_a16_tile_finish_reg_<tile>_<ad|partitioned-refined>`; record `regression/regression.json`. Appendix C9.

With the guard in place (every caller declaring `legacy`), `lemcvt tile forward --derived` was run on GM1 and AP1 at the optima of the
v2.0.0 release runs (`tables/table1.csv`: `20261111T012445Z_tile_run_GM1_v2_ad`, `20261111T050111Z_tile_run_GM1_v2_partitioned-refined`,
`20261111T012439Z_tile_run_AP1_v2_ad`, `20261111T050726Z_tile_run_AP1_v2_partitioned-refined`; run keys `ad_no_opt`, `ad_opt`,
`acrit_opt`, `joint_opt`), against those runs' grids:

| tile | run key | MSE (this run) | MSE (v2.0.0) | max \|dz\| | derived grids (filled, flow direction, area, log-area, hillshade) |
| --- | --- | --- | --- | --- | --- |
| GM1 | ad_no_opt / ad_opt | 370,959.8235631284 / 48.96275516402614 | identical | 0 / 0 | all identical, value for value |
| GM1 | acrit_opt / joint_opt | 51.29013030703221 / 24.690389367787056 | identical | 0 / 0 | joint: all identical; acrit: the release run holds the elevation only (identical) |
| AP1 | ad_no_opt / ad_opt | 478,747.3727397729 / 59.01825736555197 | identical | 0 / 0 | all identical |
| AP1 | acrit_opt / joint_opt | 289.2277726940696 / 22.38435538937004 | identical | 0 / 0 | joint: all identical; acrit: elevation only (identical) |

**Every MSE equals the release value to the last digit and every grid is identical cell for cell: no v2 result changed.** (The release
runs were made at git e9ad39c with the same derived-grid path; the `acrit_opt_D0B` key is not a forward key and was not re-run.)

## 6. The final tile list per site
Appendix C13 lists every registered tile with its version, footprint, area source and kind. In short:

| site | registered | version | kind | area source for modelling |
| --- | --- | --- | --- | --- |
| GM | **7**: GM1-GM5 (original), GM6, GM7 (drawn, stage B) | `GM-x` (EPSG:32610, NCALM 2003) | original / drawn | GM1-GM5: role `area_model` (GM3 = old grid; GM1, GM2, GM4, GM5 = new + strip inflow, section 4; GM5's to be decided); GM6, GM7: full-grid routing |
| GM extension | **0** (23 drawn and cut, parked) | `GM-FEMA2018-x` (EPSG:6339, 2018 FEMA Region 9) would be | drawn | full-grid routing of the survey (would be) |
| AP | **29**: AP1-AP5 as `AP<n>_3dep` (original), AP6-AP29 (drawn, stage B) | `AP-3DEP-x` (EPSG:6346) | original / drawn | full-grid routing |
| TTV | **0** (15 drawn and cut, parked: TTV1-TTV3 M&D by design, TTV4-TTV15 drawn) | `TTV-x` (EPSG:6339) would be | M&D by design / drawn | full-grid routing (would be) |

Registered in all: 36 tiles (GM 7, AP 29), unchanged from stage B except the five `area_model` grids. Parked, unregistered: 15 TTV and
23 GM-extension tiles, both waiting on the unit-window decisions of section 7.

## 7. What is weak, and decisions needed for Addendum 17
**Weak.**
1. **The TTV unit window** (2.4): the 0-320 m window of addendum 15 §6.4 was written for the M&D area; the Marin Headlands rise to
   784 m (Mount Tamalpais) and four of the twelve drawn tiles lie on its flank. The check caught real elevations, not a units error,
   and stopped the registration of all fifteen tiles.
2. **GM5's modelling area** (4.3): the rule's coarse grid is the 60.5 km² strip of stage A, which truncates the basin feeding GM5's
   second northern trunk; the `area_model` grid written by the rule carries 0.38 km² there where the old grid (and the NED routed
   over the whole box, to 3 %) carry 2.2-2.3 km². The GM2 grid is fine (−1.3 % at the outlet); GM1's and GM4's differ from the old
   by ≤ 9,633 m² and 16 m².
3. **The two coarse grids disagree** (4.3): Perron's 2008 grid and the current 1/3″ product route the mesa north of the lidar
   differently (GM2's trunk 3.98 against 5.27 km²; GM5's first trunk 1.53 against 2.37). The old tile grids follow Perron's. Which
   is nearer the truth cannot be told from either; the 2018 FEMA lidar would tell (section 3).
4. **The Gabilan Mesa landscape is a hand-drawn polygon** (3.2, `a16_common.GM_LANDSCAPE_POLY_UTM`), set from the NED hillshade before
   the gate was computed, and intersected with a 15 km disc. The counts for the disc alone and the polygon alone are in Appendix C7.
5. **The GM extension survey is QL2** (0.7 m nominal pulse spacing, 5.6 points m⁻² all returns, about 2-3 ground points m⁻² on the
   mesa's grassland), half the density of the NCALM 2003 grid's source and a third of the TTV survey's; hollows will have more
   nearest-neighbour-filled cells than at TTV (stage A: 14.8 % empty cells per window there at 6.9 points m⁻²).
6. **The hole-filled cells of the clipped polygon** (2.1): 12 cells at 60.88-61.0 m; immaterial, but "by construction" is not exact.
7. **The window statistic at TTV** (2.2): windows are compared by the median of their per-basin slopes against a pooled reference; the
   two statistics differ by a factor 1.4 on the polygon itself (26.9 against 18.6). With the windows' pooled slope the count at
   ±25 % would be different (Appendix C3 gives it); the rules of addendum 15 were kept as written.
8. **The 1 m entries of the GM check** (4.1) are found by a main-stem walk that stops one cell inside the data-edge ring (the ring
   carries code 0); the 10 m channels sit more than 15 m from those cells, so the comparison is made at the tile entries instead.
9. **The regression** (5.4) covers the forward evaluations at the release optima with their derived grids; the optimizer trajectories
   and the figure profiles were not re-run (2,770 solves).

**Decisions needed for Addendum 17.**
1. **TTV registration.** (a) Widen the TTV unit-check window to the Marin Headlands (0-800 m, the Mount Tamalpais summit) and register
   the 15 tiles as cut (one 40 s job); (b) keep 0-320 m as a selection rule, drop TTV4, TTV5, TTV7, TTV9 and register 11; or (c)
   redraw with an elevation screen written before the draw. (b) and (c) change the draw's outcome; (a) does not.
2. **GM5's area.** (a) Accept the old grids for GM2 and GM5 as the modelling areas, the check having been met at every trunk above 1 ha
   by the stage A strip or by the NED box (section 4.3), and re-register `area_model` accordingly; or (b) keep the rule's grids and
   model GM5 with 2.2 km² less on its second trunk. For GM1 and GM4 the choice is immaterial (≤ 9,633 m² and 16 m²).
3. **GM extension registration.** The same question as decision 1: (a) widen the extension window to the mesa's full range (the
   survey's grid spans 182-695 m on the drawn tiles; the mesa top near the range front is 700 m) and register the 23 as cut; (b) keep
   68-468 m as a selection rule, drop the ten high tiles and register 13 (20 GM tiles in all); or (c) redraw with an elevation screen
   written before the draw. The tolerances on k_s and |C_ht| were met by all 23; whether the high mesa is the same landscape as the
   NCALM footprint is a judgment the elevation window stands in for.
4. **The landscape polygon** for the GM extension: confirm or redraw (section 3.2); the eligible counts under the disc alone and the
   polygon alone are in Appendix C7.
5. **The coarse-grid question** (weak 3): which 10 m routing the modelling areas of future GM tiles should follow where the 1 m grid's
   footprint truncates a basin; the FEMA survey's 1 m full grid removes the question for the extension tiles.
6. **Measurement and modelling of the expanded set** (decision 9 of the directive): GM 7 (+ extension), AP 29, TTV 0 registered until
   decision 1; the per-tile area source is in section 6.

## 8. Appendices: cost, pre-push check, generated tables

### 8.1 Cost (decision 8)
Estimates were printed before every submission (job scripts in `$SCRATCH/lemcvt_jobs/a16/`). Appendix C12 has the accounting of all
21 submissions (20 job ids; the regression was an array of 2): **4.7 core-hours used** (expected ≤ 30); **sum of CPU-weighted limits
26.2 core-hours** (the gate's 2 h twice, once cancelled, is 4 of them); every limit at most twice its estimate with the 30 min floor
(the longest, the gate, 1 h 46 min against a 2 h limit on 1 CPU); largest submissions the fetch (8 CPUs × 30 min) and the grid
(4 CPUs × 2 h, 100 GB; it ran 12 min 51 s at 44.1 GB); every job on `serc`, one throttled array. Six of the 21 were reruns after
the script errors of section 1 (deviation 15). The directive's "about 1,000-2,000 core-hours" for Addendum 17 are untouched.
New data on group storage: the NED cells (0.9 GB) and the extension LAZ (5.8 GB) as copies of record under
`$GROUP_HOME/lemcvt_data/source/` with manifest entries, 1.7 GB in the run directory (the 1 m extension grid and its routing 1.3 GB),
187 MB of parked tiles, 6 MB of `area_model` grids; 8.6 GB in all against the 60 GB cap; `$GROUP_HOME` at 319.6 GB of 1 TB. Scratch:
the LAZ (5.0 GB) and NED (0.9 GB) staging copies were removed after their copies of record were verified; 26 MB of job logs and
pipeline metadata remain under `$SCRATCH/a16/` and `$SCRATCH/lemcvt_jobs/a16/`.

### 8.2 Pre-push check (addendum 2 section 5)
Run on the staged tree before the commit (39 files) and again on the commit against `origin/v2.3-tiles` before pushing: forbidden
tracked files (`papers/`, `*.pdf`, `working-with-steve.md`, `cry_for_help.txt`, `.env`, `.netrc`): **none**; files over 50 MB:
**none** (the largest is this report, 115 kB); secret patterns in the diff (`ghp_`, `github_pat_`, `gho_`, private keys, `password=`,
`token=`): **none**. The `.docx` manuscripts, the PDFs and the untracked drafts stay out; no raster or point cloud is tracked
(`data/manifest.csv` gains the five `area_model` rows, `configs/data.yaml` the `area_model` note; the TTV and GM-extension versions are
not registered).

### 8.3 Generated tables (`make_a16_tables.py`, from the run directory; also `appendix_tables.md` there)

### C1. The clipped M&D polygon and rule 2.2 (section 2.1), from md_georef/clip.json
| quantity | stage B polygon | clipped (a16) |
|---|---|---|
| cells (1 m) / km2 | 2,058,028 / 2.058 | 1,942,942 / 1.943 |
| z min / max on the 1 m grid (m) | 36.9 / 310.1 | 60.9 / 310.1 |
| z p01 / p99 (m) | 48.9 / 285.2 | 62.8 / 285.6 |
| cells below 61 m | 115,080 | 0 |
| components after the clip (largest kept) | - | 6 (sizes [1942930, 9, 6, 1, 1, 1]) |
| cells dropped as minor components / filled as holes | - | 18 / 12 (hole cells z min 60.883 m) |
| removed cells in all / added | - | 115,086 / 0 |
| polygon layer | study_area (md1989_fig3.gpkg) | study_area_clipped (md1989_fig3_a16.gpkg), 1.943 km2, 1 part |

Removed clusters (8-connected):
| cluster | cells | km2 | centroid E, N | z min-max (m) | cells below 61 m |
|---|---|---|---|---|---|
| 2 | 88,532 | 0.089 | 540,498, 4,190,141 | 40.4-61.2 | 88,515 |
| 1 | 26,554 | 0.027 | 540,947, 4,190,754 | 36.9-61.1 | 26,553 |

Rule 2.2: area 1.943 km2 in [1.785, 2.415] (binding): pass; z_min 60.9 m in [46, 76]: pass (by construction up to the hole filling); z_max 310.1 m in [299, 329]: pass. **PASS**

### C2. The TTV reference from the clipped polygon on the 10 m NED (section 2.2), from ttv_reference/TTV_reference.json
| quantity | clipped polygon (a16) | stage B polygon (a15) | stage A proxy |
|---|---|---|---|
| theta (legacy chi) / coarse | 0.442 / 0.450 | 0.437 | 0.387 |
| k_s pooled at theta (**the reference**) | **26.86** | 22.74 | - |
| k_s median of basins at theta (record) | 18.64 (q16-q84 10.7-26.8) | **4.02** (was the reference) | 15.42 |
| basins in the k_s fit | 24 | 33 | 15 |
| |C_ht| (1/m) | 0.0179 | 0.0173 | 0.0151 |
| convex divide pixels | 1,953 | 2,258 | - |
| cells / km2 (10 m) | 19,459 / 1.946 | 20,578 / 2.058 | 6,729 / 0.673 |
| z range (m) | 59.9-309.3 | 38.3-309.3 | 61-317 |
| largest A inside (km2) | 0.884 | 1.218 | 0.67 |
| chi R2 at theta (pooled, through origin) | 0.951 | - | - |

Whole-grid k_s at the polygon theta: median 14.52, pooled 22.43 (932 basins).

### C3. TTV eligibility and the draw (section 2.3), from draw/TTV_draw.json
| candidates | set B screens | within +-25 % | screens and +-25 % | gap-excluded | eligible at +-25 % | within +-50 % | screens and +-50 % | eligible at +-50 % | widening threshold | tolerance used | eligible | per offset |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 420 | 100 | 111 | 34 | 0 | 34 | 238 | 68 | 68 | 30 | +-25 % | 34 | 0: 5, 1: 12, 2: 9, 3: 8 |

Windows removed by each set B screen (alone): trunk: A_max_inside <= 5e+06 m2: 62; trunk: largest single edge inflow <= 2e+06 m2: 90; transient: chi-z R2 (area-weighted) >= 0.90: 76; transient: slope-area dBIC (area-weighted) <= 20: 149; disturbance: NLCD classes [11, 22, 23, 24, 31] fraction <= 0.10: 167; upstream area fully in valid data (decision 7): 48

M&D windows (5.2 of addendum 15, re-chosen against the clipped polygon): 19,459 cells (1.946 km2), 27 candidate windows intersect it; chosen 3 non-overlapping windows covering 92.7% (>= 80 %: pass).
| window (offset, id) | footprint SW (E, N) | set B screens pass | screens failed | within +-25 % | k_s dev | |C_ht| dev | gap-excluded |
|---|---|---|---|---|---|---|---|
| 0, 135 | 539000, 4190000 | no | transient: slope-area dBIC (area-weighted) <= 20 | yes | +13% | -5% | False (0) |
| 2, 118 | 540000, 4190500 | no | transient: slope-area dBIC (area-weighted) <= 20 | no | -27% | +3% | False (0) |
| 2, 136 | 540000, 4189500 | yes | none | no | -3% | -25% | False (0) |

Pre-placed: TTV1 (M&D) SW 539000, 4190000; TTV2 (M&D) SW 540000, 4190500; TTV3 (M&D) SW 540000, 4189500

Within-offset direct-edge check against stage A: offset 0: stage A 390, here 390, equal True; offset 1: stage A 382, here 382, equal True; offset 2: stage A 357, here 357, equal True; offset 3: stage A 339, here 339, equal True

Ordered list (seed 20261118, target 27 drawn tiles = 30 in all minus the 3 M&D windows; eligible windows sorted by offset and window id, permuted once):
| rank | offset | window | footprint SW (E, N) | k_s dev | |C_ht| dev | theta | result |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 157 | 543000, 4188500 | -2% | -10% | 0.383 | accepted |
| 2 | 2 | 139 | 543000, 4189500 | -10% | +3% | 0.275 | skipped: drains-through edge with accepted rank 1 (offset 2 window 157) |
| 3 | 3 | 20 | 533500, 4195500 | -19% | +14% | 0.100 | accepted |
| 4 | 0 | 155 | 541000, 4189000 | -16% | -25% | 0.100 | accepted |
| 5 | 0 | 173 | 541000, 4188000 | +5% | +10% | 0.362 | skipped: drains-through edge with accepted rank 4 (offset 0 window 155) |
| 6 | 2 | 173 | 541000, 4187500 | +9% | -1% | 0.283 | skipped: drains-through edge with accepted rank 4 (offset 0 window 155) |
| 7 | 1 | 149 | 543500, 4189000 | +10% | -2% | 0.163 | skipped: overlaps accepted rank 1 (offset 2 window 157) |
| 8 | 1 | 131 | 542500, 4190000 | +21% | +10% | 0.296 | skipped: drains-through edge with accepted rank 1 (offset 2 window 157) |
| 9 | 1 | 183 | 543500, 4187000 | -23% | +6% | 0.375 | skipped: drains-through edge with accepted rank 1 (offset 2 window 157) |
| 10 | 0 | 77 | 535000, 4193000 | -23% | +16% | 0.271 | accepted |
| 11 | 1 | 127 | 538500, 4190000 | +8% | -4% | 0.221 | skipped: overlaps pre-placed TTV1 |
| 12 | 1 | 147 | 541500, 4189000 | -17% | -15% | 0.175 | skipped: overlaps accepted rank 4 (offset 0 window 155) |
| 13 | 1 | 73 | 535500, 4193000 | +7% | +19% | 0.175 | skipped: overlaps accepted rank 10 (offset 0 window 77) |
| 14 | 3 | 92 | 537500, 4191500 | -6% | +11% | 0.288 | accepted |
| 15 | 3 | 149 | 543500, 4188500 | +4% | -12% | 0.317 | skipped: overlaps accepted rank 1 (offset 2 window 157) |
| 16 | 0 | 153 | 539000, 4189000 | -3% | -23% | 0.433 | skipped: drains-through edge with pre-placed TTV1 |
| 17 | 2 | 57 | 533000, 4193500 | -15% | +25% | 0.275 | accepted |
| 18 | 2 | 138 | 542000, 4189500 | +3% | +2% | 0.100 | accepted |
| 19 | 3 | 59 | 538500, 4193500 | -0% | +15% | 0.102 | accepted |
| 20 | 2 | 117 | 539000, 4190500 | +17% | +15% | 0.367 | skipped: overlaps pre-placed TTV1 |
| 21 | 1 | 167 | 544500, 4188000 | -23% | +22% | 0.354 | accepted |
| 22 | 3 | 76 | 538500, 4192500 | -8% | -10% | 0.221 | skipped: drains-through edge with accepted rank 14 (offset 3 window 92) |
| 23 | 3 | 31 | 544500, 4195500 | -14% | -24% | 0.100 | accepted |
| 24 | 3 | 130 | 541500, 4189500 | -9% | -22% | 0.100 | skipped: overlaps accepted rank 4 (offset 0 window 155) |
| 25 | 3 | 131 | 542500, 4189500 | +21% | +7% | 0.100 | skipped: overlaps accepted rank 18 (offset 2 window 138) |
| 26 | 1 | 71 | 533500, 4193000 | +5% | +16% | 0.100 | skipped: overlaps accepted rank 17 (offset 2 window 57) |
| 27 | 1 | 109 | 537500, 4191000 | +7% | +4% | 0.262 | skipped: overlaps accepted rank 14 (offset 3 window 92) |
| 28 | 1 | 76 | 538500, 4193000 | -5% | +5% | 0.140 | skipped: overlaps accepted rank 19 (offset 3 window 59) |
| 29 | 2 | 176 | 544000, 4187500 | -19% | +2% | 0.387 | skipped: overlaps accepted rank 21 (offset 1 window 167) |
| 30 | 2 | 155 | 541000, 4188500 | -15% | +3% | 0.304 | skipped: overlaps accepted rank 4 (offset 0 window 155) |
| 31 | 2 | 2 | 532000, 4196500 | +3% | +23% | 0.250 | skipped: drains-through edge with accepted rank 3 (offset 3 window 20) |
| 32 | 1 | 130 | 541500, 4190000 | +11% | -1% | 0.212 | skipped: overlaps accepted rank 18 (offset 2 window 138) |
| 33 | 0 | 194 | 544000, 4187000 | +6% | +17% | 0.408 | accepted |
| 34 | 1 | 93 | 538500, 4192000 | -3% | +0% | 0.263 | accepted |

Skip reasons: drains-through edge with: 8; overlaps accepted rank: 12; overlaps pre-placed TTV1: 2

Links of the pre-placed windows: TTV1: 1 eligible windows directly linked; TTV2: 4 eligible windows directly linked; TTV3: 6 eligible windows directly linked

### C4. TTV tiles cut (section 2.3) and the unit check, from tiles/tiles.json
| tile | kind | offset, window | SW corner (E, N) | z median | z min | z max | unit check (0-320 m) | A max (km2) | outlet cells | filled-gap channel cells | class-2 pts/m2 | NN-filled frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | M&D | 0, 135 | 539000, 4190000 | 203.9 | 76.103 | 305.1 | yes | 0.451 | 0 | 0 | 10.748 | 0.057 |
| TTV2 | M&D | 2, 118 | 540000, 4190500 | 139.9 | 41.684 | 241.5 | yes | 0.635 | 0 | 0 | 12.336 | 0.044 |
| TTV3 | M&D | 2, 136 | 540000, 4189500 | 83.207 | 31.262 | 166.8 | yes | 2.707 | 0 | 0 | 15.989 | 0.026 |
| TTV4 | drawn | 2, 157 | 543000, 4188500 | 150.4 | 51.670 | 321.1 | no | 0.474 | 0 | 0 | 8.122 | 0.069 |
| TTV5 | drawn | 3, 20 | 533500, 4195500 | 605.4 | 444.8 | 656.3 | no | 0.830 | 0 | 0 | 9.528 | 0.049 |
| TTV6 | drawn | 0, 155 | 541000, 4189000 | 200.6 | 60.668 | 317.3 | yes | 0.508 | 0 | 0 | 10.481 | 0.035 |
| TTV7 | drawn | 0, 77 | 535000, 4193000 | 289.4 | 164.2 | 389.2 | no | 0.230 | 0 | 0 | 9.467 | 0.104 |
| TTV8 | drawn | 3, 92 | 537500, 4191500 | 191.0 | 53.359 | 264.3 | yes | 1.036 | 0 | 0 | 7.266 | 0.141 |
| TTV9 | drawn | 2, 57 | 533000, 4193500 | 266.4 | 128.9 | 369.1 | no | 2.745 | 0 | 0 | 6.351 | 0.157 |
| TTV10 | drawn | 2, 138 | 542000, 4189500 | 228.8 | 100.1 | 317.6 | yes | 0.192 | 0 | 0 | 6.631 | 0.069 |
| TTV11 | drawn | 3, 59 | 538500, 4193500 | 194.1 | 53.714 | 263.1 | yes | 0.188 | 0 | 0 | 7.059 | 0.092 |
| TTV12 | drawn | 1, 167 | 544500, 4188000 | 166.2 | 58.186 | 288.5 | yes | 0.264 | 0 | 0 | 6.564 | 0.092 |
| TTV13 | drawn | 3, 31 | 544500, 4195500 | 131.2 | 36.320 | 185.9 | yes | 0.216 | 0 | 0 | 8.962 | 0.041 |
| TTV14 | drawn | 0, 194 | 544000, 4187000 | 173.6 | 26.150 | 287.6 | yes | 0.450 | 0 | 0 | 10.685 | 0.056 |
| TTV15 | drawn | 1, 93 | 538500, 4192000 | 190.6 | 71.376 | 278.1 | yes | 0.475 | 0 | 0 | 7.197 | 0.080 |

Unit-check failures: ['TTV4', 'TTV5', 'TTV7', 'TTV9'] -> stop, nothing registered (addendum 15 section 6.4 as written).

### C5. Montgomery & Dietrich (1989) Table 1 transcription check (section 2.4), from md_georef/md1989_table1_check.json
| rows | expected | source area min / median / max (m2) | slope min-max | length min-max (m) | width min-max (m) | rows whose (area, length, width) triple is absent from the OCR text | rows whose slope is absent | values unreadable |
|---|---|---|---|---|---|---|---|---|
| 63 | 63 | 1,200 / 7,400 / 40,100 | 0.24-0.91 | 50-400 | 5-60 | none | none | none |

Paper text mentions (regex on the OCR layer): 1,?200: True; 40,?100: True; 68 hollows: True; Twenty-two: True; Five hollows: True

The transcription (63 rows, md_georef/md1989_table1.csv):
| row | slope (m/m) | source area (m2) | source-basin length (m) | colluvial fill width (m) |
|---|---|---|---|---|
| 1 | 0.45 | 7,400 | 120 | 60 |
| 2 | 0.43 | 12,000 | 180 | 60 |
| 3 | 0.49 | 7,900 | 140 | 50 |
| 4 | 0.42 | 9,100 | 180 | 60 |
| 5 | 0.62 | 2,700 | 50 | 36 |
| 6 | 0.78 | 3,800 | 100 | 40 |
| 7 | 0.44 | 9,300 | 130 | 5 |
| 8 | 0.53 | 3,700 | 80 | 45 |
| 9 | 0.55 | 6,700 | 110 | 30 |
| 10 | 0.42 | 10,000 | 190 | 5 |
| 11 | 0.42 | 7,900 | 160 | 5 |
| 12 | 0.47 | 9,900 | 140 | 16 |
| 13 | 0.58 | 5,400 | 130 | 8 |
| 14 | 0.53 | 2,100 | 100 | 30 |
| 15 | 0.51 | 6,100 | 110 | 50 |
| 16 | 0.53 | 4,500 | 120 | 32 |
| 17 | 0.60 | 4,900 | 80 | 20 |
| 18 | 0.44 | 8,500 | 120 | 60 |
| 19 | 0.47 | 9,800 | 160 | 28 |
| 20 | 0.44 | 5,500 | 100 | 28 |
| 21 | 0.62 | 4,300 | 100 | 20 |
| 22 | 0.34 | 8,200 | 160 | 32 |
| 23 | 0.57 | 8,000 | 110 | 16 |
| 24 | 0.65 | 2,600 | 50 | 50 |
| 25 | 0.24 | 40,100 | 400 | 28 |
| 26 | 0.58 | 8,900 | 140 | 12 |
| 27 | 0.49 | 14,500 | 150 | 48 |
| 28 | 0.58 | 8,400 | 90 | 48 |
| 29 | 0.38 | 15,100 | 150 | 43 |
| 30 | 0.36 | 9,300 | 120 | 28 |
| 31 | 0.36 | 8,300 | 120 | 32 |
| 32 | 0.58 | 6,300 | 160 | 40 |
| 33 | 0.44 | 4,200 | 120 | 24 |
| 34 | 0.55 | 5,400 | 140 | 24 |
| 35 | 0.47 | 11,400 | 120 | 48 |
| 36 | 0.67 | 5,200 | 120 | 30 |
| 37 | 0.42 | 7,200 | 100 | 44 |
| 38 | 0.40 | 11,900 | 150 | 32 |
| 39 | 0.81 | 4,900 | 90 | 28 |
| 40 | 0.51 | 3,800 | 80 | 5 |
| 41 | 0.34 | 11,400 | 170 | 48 |
| 42 | 0.25 | 14,900 | 170 | 56 |
| 43 | 0.27 | 14,000 | 190 | 40 |
| 44 | 0.36 | 7,500 | 170 | 40 |
| 45 | 0.29 | 13,300 | 170 | 52 |
| 46 | 0.25 | 24,000 | 190 | 28 |
| 47 | 0.45 | 7,400 | 120 | 52 |
| 48 | 0.45 | 12,700 | 260 | 8 |
| 49 | 0.47 | 4,700 | 110 | 12 |
| 50 | 0.36 | 10,700 | 180 | 32 |
| 51 | 0.32 | 13,100 | 170 | 28 |
| 52 | 0.45 | 4,000 | 90 | 40 |
| 53 | 0.30 | 11,000 | 170 | 32 |
| 54 | 0.33 | 6,700 | 120 | 24 |
| 55 | 0.46 | 3,800 | 80 | 24 |
| 56 | 0.84 | 1,800 | 60 | 30 |
| 57 | 0.76 | 1,200 | 80 | 20 |
| 58 | 0.70 | 6,800 | 140 | 48 |
| 59 | 0.91 | 1,200 | 60 | 8 |
| 60 | 0.62 | 6,600 | 160 | 8 |
| 61 | 0.67 | 5,600 | 110 | 8 |
| 62 | 0.45 | 10,800 | 170 | 5 |
| 63 | 0.78 | 2,000 | 80 | 5 |

### C6. Public lidar over Gabilan Mesa (section 3.1), from gm_footprints/footprints.json
NED cells (1/3 arc-second, current product): n36w121: HTTP Last-Modified Wed, 27 Aug 2025 16:56:08 GMT, 445,063,554 bytes, sha256 8098be86d67530d3..., copy of record GM_NED_13arcsec/USGS_13_n36w121.tif verified True; n37w121: HTTP Last-Modified Wed, 27 Aug 2025 16:53:03 GMT, 453,581,984 bytes, sha256 41a0f5ad41d40c07..., copy of record GM_NED_13arcsec/USGS_13_n37w121.tif verified True

NED warped: 5000 x 5000 cells at 10 m, EPSG:32610, box [675000.0, 3955000.0, 725000.0, 4005000.0], 22,332,510 valid cells, z 106-1204 m, bilinear. NCALM 2003 footprint on this lattice: 39.3 km2 fully valid (39.5 km2 any data). Stage A coarse grid (Perron): 60.5 km2 valid, 39.3 inside NCALM, 21.1 beyond. Tiles' centroid 698,560 E, 3,979,668 N.

USGS Entwine copies (usgs-lidar-public), with WESM work-unit metadata; areas in km2 on the 10 m lattice:
| survey | collection | QL | spec | horizontal / vertical CRS, geoid | points (Entwine) | polygon km2 | nominal density (pts/m2, all returns) | in box | inside NCALM | beyond NCALM in Perron grid | beyond NCALM < 5 km | < 10 km | < 15 km |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CA_FEMA_Z4_B2_2018 | 2018/01/22 to 2018/04/22 | QL 2 | USGS Lidar Base Specification 1.2 | 6422 / 6360 GEOID12B | 56.7 G | 10203.1 | 5.561 | 2192.9 | 39.541 | 21.068 | 45.199 | 274.6 | 667.3 |
| CA_FEMAR9Estrella_2019 | 2019/12/15 to 2020/01/04 | QL 2 | USGS Lidar Base Specification 2.1 | 6339 / 5703 GEOID12B | 13.6 G | 2897.7 | 4.683 | 316.5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| CA_CaliforniaGaps_1_B23 | 2023/10/24 to 2023/11/29 | QL 1 | USGS Lidar Base Specification 2023 Revision A | 6339 / 5703 GEOID18 | 53.8 G | 4738.8 | 11.343 | 166.6 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| CA_CaliforniaGaps_3_B23 | 2023/09/05 to 2023/12/05 | QL 1 | USGS Lidar Base Specification 2023 Revision A | 6339 / 5703 GEOID18 | 55.8 G | 3129.8 | 17.841 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ARRA-CA_CentralCoast-Z4_2010 | 2010/08/17 to 2010/09/14 | Other | Other | 2228 / 6360 GEOID03 | 5.2 G | 2852.6 | 1.809 | 314.6 | 9.460 | 1.509 | 1.536 | 73.853 | 147.6 |
| USGS_LPC_CA_WestCoastElNinoUTM10_2016_LAS_2017 | None to None | - | - | None / None None | 34.4 G | 5973.6 | 5.757 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

NOAA Digital Coast tile indexes (the InPort items of the directive):
| NOAA ID | InPort | tiles | in box km2 | in Perron grid | inside NCALM | beyond NCALM in Perron grid | beyond NCALM < 15 km |
|---|---|---|---|---|---|---|---|
| 9269 | 64466 (2018 FEMA Lidar: Region 9, CA; QL2, NPS 0.7 m; 2018-01-22 to 2018-04-21) | 10,516 | 2153.8 | 60.481 | 39.541 | 21.068 | 667.3 |
| 9270 | 64465 (2018 FEMA Lidar: Region 9, CA (QL1), Salinas AOI 12.6 sq mi; NPS 0.35 m; 2018-02-16 to 02-18) | 41 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| 9263 | 64364 (2019 FEMA Lidar: Estrella, CA; QL2, NPS 0.71 m; 2019-12-15 to 2020-01-04) | 2,790 | 274.3 | 0.000 | 0.000 | 0.000 | 0.000 |

### C7. The GM extension gate on the 10 m NED (section 3.2), from gm_gate/gm_gate.json
Landscape polygon (EPSG:32610): [(690000, 4005000), (692500, 3996000), (693500, 3988000), (693000, 3980000), (693500, 3975000), (696500, 3970000), (701000, 3965000), (705000, 3960000), (715000, 3968000), (716000, 3980000), (712000, 3992000), (706000, 4002000), (700000, 4005000), (690000, 4005000)]; radius 15 km about 698,560 E, 3,979,668 N. Landscape areas: polygon_and_disc 472.9 km2 (433.7 beyond NCALM); disc 706.9 km2 (667.3 beyond NCALM); polygon 771.8 km2 (732.6 beyond NCALM). NED crop [683000.0, 3958000.0, 718000.0, 4005000.0], 16,450,000 valid cells. Reference: GM legacy (theta 0.292, k_s 1.9, |C_ht| 0.0128). Gap rule: not applicable on the 10 m NED (needs the 1 m grid); stated. NLCD: {'11': 82076, '21': 273564, '22': 76472, '23': 35734, '24': 5137, '31': 65939, '42': 17390, '43': 1005187, '52': 4917970, '71': 8901008, '81': 36453, '82': 790614, '90': 119581, '95': 122875}

Baseline pool (NED masked to the NCALM footprint, windows in the landscape union): 83 candidates, 9 eligible at +-25 %, 22 at +-50 %.

**CA_FEMA_Z4_B2_2018**: tolerance used 25 % (combined eligible at +-25 %: 62); windows counted for the gate 48 (>= 10: **PASS**).
| landscape | tolerance | candidates | screens pass | eligible | eligible, not overlapping GM1-7 | added (not in the NCALM baseline), not overlapping | beyond the NCALM footprint, not overlapping | baseline eligible | combined eligible |
|---|---|---|---|---|---|---|---|---|---|
| polygon_and_disc | +-25 % | 1,698 | 460 | 62 | 49 | 48 | 48 | 9 | 62 |
| polygon_and_disc | +-50 % | 1,698 | 460 | 215 | 193 | 189 | 189 | 22 | 217 |
| disc | +-25 % | 2,591 | 722 | 91 | 78 | 77 | 77 | 9 | 91 |
| disc | +-50 % | 2,591 | 722 | 310 | 288 | 284 | 284 | 22 | 312 |
| polygon | +-25 % | 2,833 | 688 | 72 | 59 | 58 | 58 | 9 | 72 |
| polygon | +-50 % | 2,833 | 688 | 273 | 251 | 247 | 247 | 22 | 275 |

**ARRA-CA_CentralCoast-Z4_2010**: tolerance used 50 % (combined eligible at +-25 %: 16); windows counted for the gate 17 (>= 10: **PASS**).
| landscape | tolerance | candidates | screens pass | eligible | eligible, not overlapping GM1-7 | added (not in the NCALM baseline), not overlapping | beyond the NCALM footprint, not overlapping | baseline eligible | combined eligible |
|---|---|---|---|---|---|---|---|---|---|
| polygon_and_disc | +-25 % | 172 | 35 | 7 | 7 | 7 | 7 | 9 | 16 |
| polygon_and_disc | +-50 % | 172 | 35 | 17 | 17 | 17 | 17 | 22 | 39 |
| disc | +-25 % | 415 | 108 | 14 | 14 | 14 | 14 | 9 | 23 |
| disc | +-50 % | 415 | 108 | 33 | 33 | 33 | 33 | 22 | 55 |
| polygon | +-25 % | 256 | 50 | 8 | 8 | 8 | 8 | 9 | 17 |
| polygon | +-50 % | 256 | 50 | 22 | 22 | 22 | 22 | 22 | 44 |

### C8. The GM drainage-area check (section 4), from gm_area_check/gm_area_check.json and gm_area_diag/gm_area_diag.json
| tile | tile entry cell (E, N) | old area (km2) | new area (km2) | old - new (km2) | coarse inflow, Perron strip (km2) | rel. diff | lidar entry by the 1 m main-stem walk | NED box inflow (km2), rel. diff (diagnostic) | result |
|---|---|---|---|---|---|---|---|---|---|
| GM1 | 701,990, 3,981,150 | 0.908 | 0.899 | 9.633e-03 | 5.500e-03 | +43% | 702,746, 3,982,946 (1868 steps) | (0.0087, '+10%') | FAIL |
| **GM1** | cells with old > new: 764; D max 0.010 km2 |  |  |  |  |  |  |  | **FAIL** -> new area + coarse inflow added at the entry cells and propagated downstream (check failed) |
| GM2 | 699,264, 3,981,450 | 0.149 | 0.148 | 4.510e-04 | 0.000 | +100% | 699,066, 3,982,060 (630 steps) | (0.1339, '+29590%') | FAIL |
| GM2 | 699,436, 3,981,450 | 8.452 | 4.362 | 4.090 | 3.978 | +3% | 702,644, 3,982,942 (3906 steps) | (5.2718, '+29%') | pass |
| **GM2** | cells with old > new: 692; D max 4.091 km2 |  |  |  |  |  |  |  | **FAIL** -> new area + coarse inflow added at the entry cells and propagated downstream (check failed) |
| GM3 | - | - | - | - | - | - | - | - | pass (no difference) |
| **GM3** | cells with old > new: 0; D max 0.000 km2 |  |  |  |  |  |  |  | **pass** -> old tile area grid (check passed) |
| GM4 | 695,066, 3,977,450 | 0.579 | 0.579 | 1.600e-05 | 0.000 | +100% | 695,190, 3,978,672 (1445 steps) | (0.0, '+100%') | FAIL |
| **GM4** | cells with old > new: 508; D max 0.000 km2 |  |  |  |  |  |  |  | **FAIL** -> new area + coarse inflow added at the entry cells and propagated downstream (check failed) |
| GM5 | 700,122, 3,982,450 | 1.763 | 0.209 | 1.554 | 1.534 | +1% | 700,552, 3,982,844 (657 steps) | (2.372, '+53%') | pass |
| GM5 | 700,334, 3,982,450 | 2.667 | 0.380 | 2.287 | 0.000 | +100% | 701,234, 3,982,700 (1135 steps) | (2.2171, '+3%') | FAIL |
| GM5 | 699,992, 3,982,354 | 0.044 | 0.044 | 2.000e-05 | 0.000 | +100% | 699,748, 3,982,616 (303 steps) | (0.0004, '+1900%') | FAIL |
| GM5 | 700,990, 3,981,480 | 1.927 | 1.682 | 0.245 | 0.224 | +9% | 702,644, 3,982,942 (2239 steps) | (0.2318, '+6%') | pass |
| **GM5** | cells with old > new: 1,794; D max 3.841 km2 |  |  |  |  |  |  |  | **FAIL** -> new area + coarse inflow added at the entry cells and propagated downstream (check failed) |

Modelling area grids registered (role area_model):
| tile | file | sha256 (first 16) | bytes | max (km2) | equals old | equals new | note |
|---|---|---|---|---|---|---|---|
| GM1 | GM-x/GM1/GM1_1m_area_model.tif | 5e4b0918a0e5b54e... | 1,030,164 | 1.154 | no | no | new area + coarse inflow added at the entry cells and propagated downstream (check failed) |
| GM2 | GM-x/GM2/GM2_1m_area_model.tif | 820883c70c60d0d8... | 1,184,724 | 8.782 | no | no | new area + coarse inflow added at the entry cells and propagated downstream (check failed) |
| GM3 | GM-x/GM3/GM3_1m_area_model.tif | 8196b4e43eab2bca... | 1,169,423 | 0.337 | yes | yes | old tile area grid (check passed) |
| GM4 | GM-x/GM4/GM4_1m_area_model.tif | c85ddc2679a7dbd4... | 1,192,917 | 6.186 | no | yes | new area + coarse inflow added at the entry cells and propagated downstream (check failed) |
| GM5 | GM-x/GM5/GM5_1m_area_model.tif | 15f75c0cf9719069... | 1,243,444 | 2.956 | no | no | new area + coarse inflow added at the entry cells and propagated downstream (check failed) |

Coarse grid: 380,347 10 m cells inside the NCALM footprint, 304 cells receive flow directly from outside; largest direct inflows: 704,186 E, 3,982,946 N: 6.30 km2; 704,196 E, 3,982,296 N: 4.07 km2; 701,086 E, 3,982,946 N: 2.24 km2; 700,326 E, 3,982,946 N: 1.54 km2; 703,176 E, 3,982,946 N: 0.92 km2; 703,856 E, 3,982,946 N: 0.36 km2; 702,216 E, 3,982,946 N: 0.14 km2

### C9. The v2.0.0 regression (section 5.4), from regression/regression.json
| run | key | MSE (this run) | MSE (v2.0.0) | equal | max |dz| vs release elevation | derived grids identical | wall s | result |
|---|---|---|---|---|---|---|---|---|
| GM1_ad | ad_no_opt | 370959.8236 | 370959.8236 | yes | 0.000 | elevation: =, filled: =, flow_direction: =, area: =, logarea: =, hillshade: = | 26 | PASS |
| GM1_ad | ad_opt | 48.96275516 | 48.96275516 | yes | 0.000 | elevation: =, filled: =, flow_direction: =, area: =, logarea: =, hillshade: = | 26 | PASS |
| GM1_partitioned-refined | acrit_opt | 51.29013031 | 51.29013031 | yes | 0.000 | elevation: =, filled: {'reference': 'absent'}, flow_direction: {'reference': 'absent'}, area: {'reference': 'absent'}, logarea: {'reference': 'absent'}, hillshade: {'reference': 'absent'} | 4 | PASS |
| GM1_partitioned-refined | joint_opt | 24.69038937 | 24.69038937 | yes | 0.000 | elevation: =, filled: =, flow_direction: =, area: =, logarea: =, hillshade: = | 5 | PASS |
| AP1_ad | ad_no_opt | 478747.3727 | 478747.3727 | yes | 0.000 | elevation: =, filled: =, flow_direction: =, area: =, logarea: =, hillshade: = | 26 | PASS |
| AP1_ad | ad_opt | 59.01825737 | 59.01825737 | yes | 0.000 | elevation: =, filled: =, flow_direction: =, area: =, logarea: =, hillshade: = | 26 | PASS |
| AP1_partitioned-refined | acrit_opt | 289.2277727 | 289.2277727 | yes | 0.000 | elevation: =, filled: {'reference': 'absent'}, flow_direction: {'reference': 'absent'}, area: {'reference': 'absent'}, logarea: {'reference': 'absent'}, hillshade: {'reference': 'absent'} | 5 | PASS |
| AP1_partitioned-refined | joint_opt | 22.38435539 | 22.38435539 | yes | 0.000 | elevation: =, filled: =, flow_direction: =, area: =, logarea: =, hillshade: = | 7 | PASS |

Code: this run at git 2e7a0350a9831785b05eb2bf3994db0ad465b7f1-dirty, release runs at 6ca929f6763a6eb8a6233549e0039f0545d15a65-dirty. **ALL PASS**

### C10. Draw configuration (written before the draw)

```json
{
 "seed": 20261118,
 "thresholds": "/home/users/scdobbs/LEMs_competition_vs_threshold-JGR-branch/reports/v2/scripts/a14_thresholds_B.json",
 "tolerance": 0.25,
 "tolerance_wide": 0.5,
 "min_eligible": {
  "GM": 25,
  "AP": 25,
  "TTV": 30
 },
 "target_new": {
  "GM": 25,
  "AP": 25,
  "TTV": 30
 },
 "ttv_target_total": 30,
 "gap_rule_A_m2": 1000.0,
 "references": {
  "GM": {
   "theta": 0.292,
   "ks": 1.9,
   "C_ht": 0.0128,
   "source": "legacy TAYLOR_REF"
  },
  "AP": {
   "theta": 0.54,
   "ks": 47.2,
   "C_ht": 0.00593,
   "source": "legacy TAYLOR_REF"
  },
  "TTV": {
   "theta": 0.44200000000000006,
   "ks": 26.856895757009767,
   "C_ht": 0.01793975830078125,
   "source": "clipped M&D (1989) study-area polygon (z >= 61 m) on the 10 m NED; k_s pooled (addendum 16 decision 2)"
  }
 },
 "written": "2026-11-18T11:03:00-0800",
 "note": "written before the draw"
}
```

Threshold set B: {"_comment": "Addendum 14 stage A: ALTERNATIVE (relaxed) screening set B, shown beside set A so the consequence of each rule is visible. Not applied as final.", "trunk_A_max_inside_m2_max": 5000000.0, "trunk_A_inflow_max_single_m2_max": 2000000.0, "transient_chi_r2_area_weighted_min": 0.9, "transient_sa_dbic_area_weighted_max": 20.0, "disturbance_nlcd_disturbed_frac_max": 0.1, "disturbance_nlcd_classes": [11, 22, 23, 24, 31], "disturbance_mining_frac_max": 0.0, "disturbance_mining_layers": ["longwall_panels"], "upstream_fully_valid_required": {"GM": true, "TTV": true, "AP": false}, "tolerance_ks_rel": 0.25, "tolerance_C_ht_rel": 0.25, "tolerance_theta_abs": 0.05}

### C11. Checksums (sha256) of the TTV tiles cut (unregistered, parked in tiles/TTV-x_unregistered)
| tile | role | path | sha256 (first 16) | bytes |
|---|---|---|---|---|
| TTV1 | filled | TTV-x/TTV1/TTV1_1m_filled.tif | 6f7665fba53f9f86... | 3,030,768 |
| TTV1 | fd | TTV-x/TTV1/TTV1_1m_fd.tif | c134203cbe6abbaf... | 237,251 |
| TTV1 | area | TTV-x/TTV1/TTV1_1m_area.tif | c8ebf9be6e486106... | 1,184,004 |
| TTV1 | gapfill_mask | TTV-x/TTV1/TTV1_1m_gapfill_mask.tif | c1821514a9cab1e5... | 1,866 |
| TTV1 | class2_count | TTV-x/TTV1/TTV1_1m_class2_count.tif | c5d635b5f8102a5e... | 808,633 |
| TTV1 | nnfill_mask | TTV-x/TTV1/TTV1_1m_nnfill_mask.tif | 8b2fd1fdbfab7fae... | 57,967 |
| TTV2 | filled | TTV-x/TTV2/TTV2_1m_filled.tif | 95737de40f09260e... | 3,128,362 |
| TTV2 | fd | TTV-x/TTV2/TTV2_1m_fd.tif | c908af16706e1b51... | 223,223 |
| TTV2 | area | TTV-x/TTV2/TTV2_1m_area.tif | caa75970e66c0995... | 1,209,432 |
| TTV2 | gapfill_mask | TTV-x/TTV2/TTV2_1m_gapfill_mask.tif | 8e943a26c5b1db33... | 1,866 |
| TTV2 | class2_count | TTV-x/TTV2/TTV2_1m_class2_count.tif | 16a1ef1220e398a2... | 810,587 |
| TTV2 | nnfill_mask | TTV-x/TTV2/TTV2_1m_nnfill_mask.tif | 56c9133c29b4616e... | 44,065 |
| TTV3 | filled | TTV-x/TTV3/TTV3_1m_filled.tif | 53fdd1b7108a289f... | 3,184,431 |
| TTV3 | fd | TTV-x/TTV3/TTV3_1m_fd.tif | 03cd522064c8381c... | 225,018 |
| TTV3 | area | TTV-x/TTV3/TTV3_1m_area.tif | 3fcaf91c61ca040b... | 1,202,444 |
| TTV3 | gapfill_mask | TTV-x/TTV3/TTV3_1m_gapfill_mask.tif | 5549ca2b6611e8e4... | 1,866 |
| TTV3 | class2_count | TTV-x/TTV3/TTV3_1m_class2_count.tif | fbbd55f56fa3338c... | 836,491 |
| TTV3 | nnfill_mask | TTV-x/TTV3/TTV3_1m_nnfill_mask.tif | 9123e4a26e000d18... | 32,363 |
| TTV4 | filled | TTV-x/TTV4/TTV4_1m_filled.tif | 1c39b85216225656... | 2,983,381 |
| TTV4 | fd | TTV-x/TTV4/TTV4_1m_fd.tif | 7a1649ea3ba3ccaa... | 231,513 |
| TTV4 | area | TTV-x/TTV4/TTV4_1m_area.tif | 574a94e13dd8102e... | 1,205,748 |
| TTV4 | gapfill_mask | TTV-x/TTV4/TTV4_1m_gapfill_mask.tif | f1ac76dd9be745f8... | 1,866 |
| TTV4 | class2_count | TTV-x/TTV4/TTV4_1m_class2_count.tif | 046434db88a99593... | 734,272 |
| TTV4 | nnfill_mask | TTV-x/TTV4/TTV4_1m_nnfill_mask.tif | 5e811d559aa91c5c... | 61,970 |
| TTV5 | filled | TTV-x/TTV5/TTV5_1m_filled.tif | d4600f9ed4b192a0... | 2,777,383 |
| TTV5 | fd | TTV-x/TTV5/TTV5_1m_fd.tif | 119ba7f75f0b098a... | 226,701 |
| TTV5 | area | TTV-x/TTV5/TTV5_1m_area.tif | 1023d8462ccb2b23... | 1,189,496 |
| TTV5 | gapfill_mask | TTV-x/TTV5/TTV5_1m_gapfill_mask.tif | 713837d5afd59a60... | 1,866 |
| TTV5 | class2_count | TTV-x/TTV5/TTV5_1m_class2_count.tif | 37df6cc9e71eaa2d... | 718,397 |
| TTV5 | nnfill_mask | TTV-x/TTV5/TTV5_1m_nnfill_mask.tif | d7cd3ff6cb7f7690... | 48,328 |
| TTV6 | filled | TTV-x/TTV6/TTV6_1m_filled.tif | ad15a4d19905e3a3... | 3,054,408 |
| TTV6 | fd | TTV-x/TTV6/TTV6_1m_fd.tif | 8fc59cdb1d49fbe2... | 217,483 |
| TTV6 | area | TTV-x/TTV6/TTV6_1m_area.tif | fff9a8e76627cfe7... | 1,227,656 |
| TTV6 | gapfill_mask | TTV-x/TTV6/TTV6_1m_gapfill_mask.tif | 1bddbb983703376e... | 1,866 |
| TTV6 | class2_count | TTV-x/TTV6/TTV6_1m_class2_count.tif | f1e47c8dbd77abee... | 755,248 |
| TTV6 | nnfill_mask | TTV-x/TTV6/TTV6_1m_nnfill_mask.tif | 11547a30b1142cdf... | 40,468 |
| TTV7 | filled | TTV-x/TTV7/TTV7_1m_filled.tif | 9b42aee8e542df9a... | 2,794,177 |
| TTV7 | fd | TTV-x/TTV7/TTV7_1m_fd.tif | 0aec56039561497e... | 239,211 |
| TTV7 | area | TTV-x/TTV7/TTV7_1m_area.tif | 33dba98b212e150a... | 1,185,781 |
| TTV7 | gapfill_mask | TTV-x/TTV7/TTV7_1m_gapfill_mask.tif | 94124c5a1b78f39d... | 1,866 |
| TTV7 | class2_count | TTV-x/TTV7/TTV7_1m_class2_count.tif | 56e2bc81ba52b368... | 742,851 |
| TTV7 | nnfill_mask | TTV-x/TTV7/TTV7_1m_nnfill_mask.tif | 33a30b733419805d... | 66,919 |
| TTV8 | filled | TTV-x/TTV8/TTV8_1m_filled.tif | fecb3b7c344a6ebb... | 2,764,172 |
| TTV8 | fd | TTV-x/TTV8/TTV8_1m_fd.tif | 279b626e7040e7c3... | 262,500 |
| TTV8 | area | TTV-x/TTV8/TTV8_1m_area.tif | 6a7590468a4dfb7d... | 1,166,182 |
| TTV8 | gapfill_mask | TTV-x/TTV8/TTV8_1m_gapfill_mask.tif | 597a9d8798b098f4... | 1,866 |
| TTV8 | class2_count | TTV-x/TTV8/TTV8_1m_class2_count.tif | 8f0fc5b70a3d5ecb... | 727,681 |
| TTV8 | nnfill_mask | TTV-x/TTV8/TTV8_1m_nnfill_mask.tif | 2870c640cc54deaa... | 89,910 |
| TTV9 | filled | TTV-x/TTV9/TTV9_1m_filled.tif | 2743718c0b789b23... | 2,675,895 |
| TTV9 | fd | TTV-x/TTV9/TTV9_1m_fd.tif | 1c1963c28e1d30e0... | 257,430 |
| TTV9 | area | TTV-x/TTV9/TTV9_1m_area.tif | 002f077d0e8c0f25... | 1,191,446 |
| TTV9 | gapfill_mask | TTV-x/TTV9/TTV9_1m_gapfill_mask.tif | 1ee248517c07f660... | 1,866 |
| TTV9 | class2_count | TTV-x/TTV9/TTV9_1m_class2_count.tif | 226fab22ff6c7ef6... | 704,172 |
| TTV9 | nnfill_mask | TTV-x/TTV9/TTV9_1m_nnfill_mask.tif | 6b587914fc008fb1... | 96,872 |
| TTV10 | filled | TTV-x/TTV10/TTV10_1m_filled.tif | 789700536f5ea90f... | 2,869,088 |
| TTV10 | fd | TTV-x/TTV10/TTV10_1m_fd.tif | b11054e51b1664f8... | 230,315 |
| TTV10 | area | TTV-x/TTV10/TTV10_1m_area.tif | 7f4aab4fb6a686e0... | 1,237,399 |
| TTV10 | gapfill_mask | TTV-x/TTV10/TTV10_1m_gapfill_mask.tif | bda45948ed8d444a... | 1,866 |
| TTV10 | class2_count | TTV-x/TTV10/TTV10_1m_class2_count.tif | b35d0cf162576720... | 681,566 |
| TTV10 | nnfill_mask | TTV-x/TTV10/TTV10_1m_nnfill_mask.tif | 8e1b872b2f44674b... | 60,346 |
| TTV11 | filled | TTV-x/TTV11/TTV11_1m_filled.tif | d9811cf45716ca06... | 2,892,254 |
| TTV11 | fd | TTV-x/TTV11/TTV11_1m_fd.tif | 6de2c764e3fbf190... | 240,484 |
| TTV11 | area | TTV-x/TTV11/TTV11_1m_area.tif | e0977f9216393a3a... | 1,176,689 |
| TTV11 | gapfill_mask | TTV-x/TTV11/TTV11_1m_gapfill_mask.tif | 001f51f5a3af344a... | 1,866 |
| TTV11 | class2_count | TTV-x/TTV11/TTV11_1m_class2_count.tif | cc3828a10d7cf025... | 698,166 |
| TTV11 | nnfill_mask | TTV-x/TTV11/TTV11_1m_nnfill_mask.tif | 6e1cee3f9ac5e6cc... | 67,860 |
| TTV12 | filled | TTV-x/TTV12/TTV12_1m_filled.tif | 247864fc53754a46... | 2,852,977 |
| TTV12 | fd | TTV-x/TTV12/TTV12_1m_fd.tif | 6a2264fdd9c41233... | 232,843 |
| TTV12 | area | TTV-x/TTV12/TTV12_1m_area.tif | 819f6c5ba168a29a... | 1,199,794 |
| TTV12 | gapfill_mask | TTV-x/TTV12/TTV12_1m_gapfill_mask.tif | f39dcc0d910a8fa5... | 1,866 |
| TTV12 | class2_count | TTV-x/TTV12/TTV12_1m_class2_count.tif | d9e24fb6108ecef1... | 694,710 |
| TTV12 | nnfill_mask | TTV-x/TTV12/TTV12_1m_nnfill_mask.tif | fa78bd2cdcaf7a05... | 72,089 |
| TTV13 | filled | TTV-x/TTV13/TTV13_1m_filled.tif | c1abe1fdc0dc5db4... | 2,994,286 |
| TTV13 | fd | TTV-x/TTV13/TTV13_1m_fd.tif | 3a3a65f8ba365d27... | 220,063 |
| TTV13 | area | TTV-x/TTV13/TTV13_1m_area.tif | 2f2d8626f91bb2f1... | 1,236,394 |
| TTV13 | gapfill_mask | TTV-x/TTV13/TTV13_1m_gapfill_mask.tif | a3365a56f7a38b64... | 1,876 |
| TTV13 | class2_count | TTV-x/TTV13/TTV13_1m_class2_count.tif | fd1effa3ddb4d717... | 657,825 |
| TTV13 | nnfill_mask | TTV-x/TTV13/TTV13_1m_nnfill_mask.tif | f76215108fcc9a4d... | 29,756 |
| TTV14 | filled | TTV-x/TTV14/TTV14_1m_filled.tif | ee5f7a34685a664c... | 3,074,788 |
| TTV14 | fd | TTV-x/TTV14/TTV14_1m_fd.tif | 9723e581f4934fe6... | 224,653 |
| TTV14 | area | TTV-x/TTV14/TTV14_1m_area.tif | 52ce8dc79f1b4f40... | 1,203,606 |
| TTV14 | gapfill_mask | TTV-x/TTV14/TTV14_1m_gapfill_mask.tif | 0ad5219e4150b2f1... | 1,866 |
| TTV14 | class2_count | TTV-x/TTV14/TTV14_1m_class2_count.tif | c12c73ad5ddd9f4f... | 774,048 |
| TTV14 | nnfill_mask | TTV-x/TTV14/TTV14_1m_nnfill_mask.tif | e40c3ba33d2484fe... | 52,776 |
| TTV15 | filled | TTV-x/TTV15/TTV15_1m_filled.tif | f76f94a8de7794f9... | 2,916,130 |
| TTV15 | fd | TTV-x/TTV15/TTV15_1m_fd.tif | ffea456c0a8d953e... | 242,996 |
| TTV15 | area | TTV-x/TTV15/TTV15_1m_area.tif | 36355c80e1983baa... | 1,172,679 |
| TTV15 | gapfill_mask | TTV-x/TTV15/TTV15_1m_gapfill_mask.tif | b8f1576be3972ec8... | 1,866 |
| TTV15 | class2_count | TTV-x/TTV15/TTV15_1m_class2_count.tif | a469ce1fc95f9bd7... | 712,491 |
| TTV15 | nnfill_mask | TTV-x/TTV15/TTV15_1m_nnfill_mask.tif | 66953b0ccffdaa41... | 66,685 |

### C14. The GM extension: fetch footprint, fetch, grid, routing and checks (section 3.3.1-3.3.3)
| quantity | value |
|---|---|
| eligible windows (gate tolerance, in the landscape, not overlapping GM1-7) | 49 |
| windows km2 / upstream km2 / footprint km2 (500 m buffer) | 37.75 / 34.7 / 166.8 |
| bounding box (EPSG:32610) | [692940.0, 3964480.0, 715190.0, 3997150.0] |
| 1 m lattice cells (G) | 0.727 |
| EPT boxes fetched / ok | 292 / 292 |
| LAZ bytes / seconds / MB s-1 | 5.34 GB / 414 / 12.9 |
| CRS written | EPSG:6339+5703 |
| vertical evidence | USGS WESM work unit CA_FEMA_Z4_B2_2018 (project CA_AZ_FEMA_R9_Lidar_2017_D18): vertical CRS EPSG:6360 (NAVD88 height, ftUS) GEOID12B on the source; the Entwine copy carries EPSG:3857 and no vertical CRS; its Z is in metres by the rule 5.3 / NCALM checks of a16_gmf_check.py |

Grid (a14_grid_route.py --site GMF, decision 9 of addendum 14):
| lattice | cells inside footprint | cells with class 2 (inside) | class-2 points | density (pts m-2) | points per cell 5-50-95 % | NN-filled within 10 m | left nodata beyond 10 m | NN fill distance 50-95-99-max (m) | seconds | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 32681 x 22261 | 166,895,820 | 161,709,230 | 983,738,893 | 3.410 | [1, 3, 8] | 5,178,269 (3.1%) | 8,321 | [1.0, 1.0, 2.0, 10.0] | 309.3 | 29.893 |

Class-2 flight dates over the 292 boxes: 2018-02-07 20:35 to 2018-02-16 00:53 UTC (adjusted standard GPS time (inferred from magnitude; header flag unset)).

Re-routing with the gap rule (a15_reroute.py --site GMF):
| nodata components | enclosed (filled) | enclosed cells / km2 | enclosed >= 1e4 cells | valid after | data-edge outlets | largest basin km2 | budget closed | pad | seconds | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 24 | 23 | 126,118 / 0.126 | 1 | 167,013,617 | 240,844 | 7.443 | yes | 0 | 304.3 | 37.295 |

Largest outlets: (700,408, 3,975,310) 7.44 km2 (4.5%); (707,600, 3,970,000) 6.16 km2 (3.7%); (710,768, 3,981,000) 6.10 km2 (3.7%); (700,082, 3,982,230) 5.31 km2 (3.2%); (698,800, 3,992,990) 4.36 km2 (2.6%)

Units and datum (a16_gmf_check.py):
| comparison | cells | median dz (m) | 5-95 % | RMS | Theil-Sen slope (range) | criterion | result |
|---|---|---|---|---|---|---|---|
| rule 5.3: NED 10 m - grid block mean (ground cells) | 711,470 | 0.058 | -0.81, 0.89 | 0.506 | 0.99995 (0.99988-1.00003) | |median| < 5 m, slope 0.98-1.02 | pass |
| NCALM 2003 - grid (every 5th NCALM cell, bilinear) | 465,568 | -33.846 | -34.13, -33.59 | 33.856 | 1.0012 (1.0011-1.0012) | |offset| 30-34 m (NCALM ellipsoidal), slope 0.98-1.02; stop near 3.28 / 0.305 | offset magnitude 33.8 m in window; slope pass; no stop (the script's own sign-flipped flag read False) |

### C15. The GM extension draw (section 3.3.4-3.3.5), from draw_gmf/GMF_draw.json
Eligibility on the 1 m grid: 49 gate-eligible windows, 0 gap-excluded, 0 outside the 1 m grid, **49 eligible** (per offset {'0': 9, '1': 9, '2': 16, '3': 15}). Seed 20261119 written at 2026-11-18T14:17:38-0800 before the draw; tolerance +-25 %; target 23 (30 in all minus GM1-GM7); **accepted 23**, skipped 26 (overlaps accepted rank: 15; drains-through edge with: 11).
| rank | offset | window | footprint SW (E, N) | k_s dev | |C_ht| dev | theta | inside NCALM | result |
|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 1,180 | 708000, 3971000 | +5% | -21% | 0.354 | no | accepted |
| 2 | 1 | 1,076 | 705500, 3973000 | +13% | -23% | 0.429 | no | accepted |
| 3 | 3 | 861 | 694500, 3978500 | +7% | -15% | 0.442 | no | accepted |
| 4 | 0 | 1,135 | 698000, 3972000 | +16% | -11% | 0.379 | no | accepted |
| 5 | 2 | 894 | 702000, 3978500 | +7% | +3% | 0.363 | no | accepted |
| 6 | 1 | 935 | 700500, 3977000 | +23% | -24% | 0.338 | no | accepted |
| 7 | 2 | 869 | 712000, 3979500 | -16% | +14% | 0.246 | no | accepted |
| 8 | 0 | 921 | 694000, 3978000 | +9% | +6% | 0.396 | no | skipped: overlaps accepted rank 3 (offset 3 window 861) |
| 9 | 1 | 975 | 706500, 3976000 | +8% | +17% | 0.367 | no | accepted |
| 10 | 2 | 1,138 | 701000, 3971500 | +19% | +15% | 0.379 | no | accepted |
| 11 | 1 | 1,309 | 700500, 3966000 | +9% | -23% | 0.525 | no | accepted |
| 12 | 3 | 429 | 704500, 3991500 | -16% | -3% | 0.200 | no | accepted |
| 13 | 3 | 355 | 698500, 3993500 | -15% | +10% | 0.358 | no | accepted |
| 14 | 3 | 894 | 693500, 3977500 | +12% | +10% | 0.287 | no | skipped: drains-through edge with accepted rank 3 (offset 3 window 861) |
| 15 | 3 | 941 | 706500, 3976500 | +17% | +8% | 0.333 | no | skipped: overlaps accepted rank 9 (offset 1 window 975) |
| 16 | 3 | 731 | 700500, 3982500 | +16% | +18% | 0.342 | no | skipped: drains-through edge with pre-placed GM5 |
| 17 | 2 | 1,276 | 699000, 3967500 | +4% | -23% | 0.329 | no | accepted |
| 18 | 0 | 1,003 | 706000, 3976000 | +21% | +5% | 0.329 | no | skipped: overlaps accepted rank 9 (offset 1 window 975) |
| 19 | 3 | 901 | 700500, 3977500 | +13% | -4% | 0.329 | no | skipped: overlaps accepted rank 6 (offset 1 window 935) |
| 20 | 0 | 1,008 | 711000, 3976000 | +18% | -9% | 0.304 | no | accepted |
| 21 | 2 | 747 | 695000, 3982500 | +3% | -23% | 0.213 | no | accepted |
| 22 | 2 | 366 | 699000, 3993500 | -16% | -6% | 0.292 | no | skipped: overlaps accepted rank 13 (offset 3 window 355) |
| 23 | 2 | 728 | 711000, 3983500 | +10% | +15% | 0.100 | no | accepted |
| 24 | 2 | 788 | 701000, 3981500 | +25% | +11% | 0.342 | yes | skipped: drains-through edge with pre-placed GM1 |
| 25 | 3 | 860 | 693500, 3978500 | -12% | -1% | 0.296 | no | skipped: drains-through edge with accepted rank 3 (offset 3 window 861) |
| 26 | 0 | 962 | 700000, 3977000 | +21% | -17% | 0.337 | no | skipped: overlaps accepted rank 6 (offset 1 window 935) |
| 27 | 0 | 1,033 | 701000, 3975000 | +20% | -18% | 0.300 | no | accepted |
| 28 | 3 | 970 | 701500, 3975500 | +11% | -25% | 0.308 | no | skipped: overlaps accepted rank 27 (offset 0 window 1033) |
| 29 | 1 | 661 | 698500, 3985000 | +6% | +11% | 0.500 | no | accepted |
| 30 | 3 | 741 | 710500, 3982500 | +7% | +16% | 0.233 | no | skipped: drains-through edge with accepted rank 23 (offset 2 window 728) |
| 31 | 2 | 921 | 694000, 3977500 | +11% | -6% | 0.417 | no | skipped: drains-through edge with pre-placed GM4 |
| 32 | 2 | 441 | 704000, 3991500 | +10% | +10% | 0.317 | no | skipped: overlaps accepted rank 12 (offset 3 window 429) |
| 33 | 1 | 894 | 693500, 3978000 | +4% | +14% | 0.383 | no | skipped: drains-through edge with accepted rank 3 (offset 3 window 861) |
| 34 | 1 | 1,146 | 707500, 3971000 | -20% | -17% | 0.371 | no | skipped: overlaps accepted rank 1 (offset 0 window 1180) |
| 35 | 1 | 895 | 694500, 3978000 | +7% | -24% | 0.508 | no | skipped: overlaps accepted rank 3 (offset 3 window 861) |
| 36 | 2 | 969 | 707000, 3976500 | -2% | +11% | 0.131 | no | skipped: overlaps accepted rank 9 (offset 1 window 975) |
| 37 | 1 | 974 | 705500, 3976000 | +23% | -13% | 0.383 | no | skipped: drains-through edge with accepted rank 9 (offset 1 window 975) |
| 38 | 3 | 428 | 703500, 3991500 | +7% | +23% | 0.263 | no | skipped: drains-through edge with accepted rank 12 (offset 3 window 429) |
| 39 | 0 | 763 | 711000, 3983000 | +21% | +8% | 0.317 | no | skipped: overlaps accepted rank 23 (offset 2 window 728) |
| 40 | 3 | 828 | 695500, 3979500 | +13% | -3% | 0.283 | no | accepted |
| 41 | 3 | 940 | 705500, 3976500 | +20% | -25% | 0.417 | no | skipped: drains-through edge with accepted rank 9 (offset 1 window 975) |
| 42 | 3 | 1,146 | 707500, 3970500 | -21% | -6% | 0.437 | no | skipped: overlaps accepted rank 1 (offset 0 window 1180) |
| 43 | 0 | 899 | 707000, 3979000 | +24% | +24% | 0.388 | no | accepted |
| 44 | 2 | 973 | 711000, 3976500 | +19% | +0% | 0.392 | no | skipped: overlaps accepted rank 20 (offset 0 window 1008) |
| 45 | 2 | 1,241 | 699000, 3968500 | -5% | -25% | 0.371 | no | skipped: drains-through edge with accepted rank 17 (offset 2 window 1276) |
| 46 | 2 | 939 | 712000, 3977500 | +20% | -24% | 0.296 | no | accepted |
| 47 | 2 | 896 | 704000, 3978500 | +17% | -24% | 0.250 | no | accepted |
| 48 | 3 | 627 | 698500, 3985500 | +19% | +7% | 0.446 | no | skipped: overlaps accepted rank 29 (offset 1 window 661) |
| 49 | 2 | 797 | 710000, 3981500 | -6% | +2% | 0.346 | no | accepted |

Draw configuration:

```json
{
 "seed": 20261119,
 "survey": "CA_FEMA_Z4_B2_2018",
 "thresholds": "/home/users/scdobbs/LEMs_competition_vs_threshold-JGR-branch/reports/v2/scripts/a14_thresholds_B.json",
 "tolerance_used": 0.25,
 "tolerance_from": "the widening rule on the combined count (gate)",
 "combined_eligible_at_25": 62,
 "target_total_GM": 30,
 "pre_placed": [
  "GM1",
  "GM2",
  "GM3",
  "GM4",
  "GM5",
  "GM6",
  "GM7"
 ],
 "target_new": 23,
 "gap_rule_A_m2": 1000.0,
 "reference": {
  "theta": 0.292,
  "ks": 1.9,
  "C_ht": 0.0128,
  "A_min": 10000.0,
  "U": 0.000147,
  "E_m_per_Myr": 147.0,
  "K_per_Myr": 100.2
 },
 "written": "2026-11-18T14:17:38-0800",
 "note": "written before the draw"
}
```

### C16. The GM extension tiles cut (section 3.3.6) and the unit check, from tiles_gmf/tiles.json
| tile | offset, window | SW corner (E, N) | z median | z min | z max | unit check (68-468 m) | A max (km2) | outlet cells | filled-gap channel cells | class-2 pts/m2 | NN-filled frac |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GM8 | 0, 1,180 | 708000, 3971000 | 345.9 | 284.9 | 399.8 | yes | 1.544 | 0 | 0 | 3.534 | 0.017 |
| GM9 | 1, 1,076 | 705500, 3973000 | 374.8 | 325.8 | 427.0 | yes | 1.685 | 0 | 0 | 3.395 | 0.014 |
| GM10 | 3, 861 | 694500, 3978500 | 263.3 | 217.7 | 294.9 | yes | 0.298 | 0 | 0 | 3.582 | 0.015 |
| GM11 | 0, 1,135 | 698000, 3972000 | 246.1 | 192.4 | 294.2 | yes | 1.491 | 0 | 0 | 3.606 | 9.282e-03 |
| GM12 | 2, 894 | 702000, 3978500 | 447.4 | 355.1 | 497.1 | no | 0.425 | 0 | 0 | 3.473 | 0.015 |
| GM13 | 1, 935 | 700500, 3977000 | 373.4 | 322.1 | 430.1 | yes | 2.367 | 0 | 0 | 3.517 | 0.011 |
| GM14 | 2, 869 | 712000, 3979500 | 572.5 | 503.1 | 693.1 | no | 0.452 | 0 | 0 | 2.854 | 0.085 |
| GM15 | 1, 975 | 706500, 3976000 | 378.9 | 306.6 | 497.0 | no | 0.557 | 0 | 0 | 3.210 | 0.029 |
| GM16 | 2, 1,138 | 701000, 3971500 | 259.5 | 207.6 | 325.8 | yes | 1.707 | 0 | 0 | 3.634 | 8.413e-03 |
| GM17 | 1, 1,309 | 700500, 3966000 | 287.8 | 229.2 | 318.6 | yes | 1.086 | 0 | 0 | 3.694 | 8.788e-03 |
| GM18 | 3, 429 | 704500, 3991500 | 583.7 | 540.0 | 670.4 | no | 1.342 | 0 | 0 | 3.390 | 0.032 |
| GM19 | 3, 355 | 698500, 3993500 | 332.3 | 275.1 | 426.5 | yes | 1.547 | 0 | 0 | 3.748 | 0.012 |
| GM20 | 2, 1,276 | 699000, 3967500 | 229.8 | 181.8 | 282.1 | yes | 1.275 | 0 | 0 | 3.576 | 9.201e-03 |
| GM21 | 0, 1,008 | 711000, 3976000 | 589.2 | 502.9 | 653.8 | no | 0.517 | 0 | 0 | 3.245 | 0.043 |
| GM22 | 2, 747 | 695000, 3982500 | 262.8 | 204.9 | 319.9 | yes | 1.636 | 0 | 0 | 3.729 | 4.321e-03 |
| GM23 | 2, 728 | 711000, 3983500 | 463.6 | 421.0 | 547.0 | no | 1.037 | 0 | 0 | 3.318 | 0.021 |
| GM24 | 0, 1,033 | 701000, 3975000 | 328.6 | 273.5 | 367.7 | yes | 2.437 | 0 | 0 | 3.596 | 8.828e-03 |
| GM25 | 1, 661 | 698500, 3985000 | 332.3 | 271.5 | 444.0 | yes | 0.890 | 0 | 0 | 3.175 | 0.034 |
| GM26 | 3, 828 | 695500, 3979500 | 287.2 | 235.0 | 334.8 | yes | 1.998 | 0 | 0 | 3.666 | 8.801e-03 |
| GM27 | 0, 899 | 707000, 3979000 | 456.0 | 396.2 | 533.5 | no | 0.582 | 0 | 0 | 2.896 | 0.072 |
| GM28 | 2, 939 | 712000, 3977500 | 641.3 | 542.5 | 695.2 | no | 3.259 | 0 | 0 | 2.957 | 0.079 |
| GM29 | 2, 896 | 704000, 3978500 | 473.9 | 387.0 | 530.0 | no | 0.385 | 0 | 0 | 3.300 | 0.028 |
| GM30 | 2, 797 | 710000, 3981500 | 408.9 | 369.1 | 509.3 | no | 1.375 | 0 | 0 | 3.496 | 0.021 |

Unit-check failures: ['GM12', 'GM14', 'GM15', 'GM18', 'GM21', 'GM23', 'GM27', 'GM28', 'GM29', 'GM30'] -> stop, nothing registered; files parked in tiles_gmf/GM-FEMA2018-x_unregistered.

Checksums (sha256) of the products cut:
| tile | role | path | sha256 (first 16) | bytes |
|---|---|---|---|---|
| GM8 | filled | GM-FEMA2018-x/GM8/GM8_1m_filled.tif | 7d794d455650fd50... | 2,584,262 |
| GM8 | fd | GM-FEMA2018-x/GM8/GM8_1m_fd.tif | f676337ee72beed0... | 220,113 |
| GM8 | area | GM-FEMA2018-x/GM8/GM8_1m_area.tif | 050b7845ccea409f... | 1,165,778 |
| GM8 | gapfill_mask | GM-FEMA2018-x/GM8/GM8_1m_gapfill_mask.tif | 420bef5b2a3cd89f... | 1,866 |
| GM8 | class2_count | GM-FEMA2018-x/GM8/GM8_1m_class2_count.tif | 58a07af9ce890893... | 528,074 |
| GM8 | nnfill_mask | GM-FEMA2018-x/GM8/GM8_1m_nnfill_mask.tif | e5a9f633ab0a9435... | 28,127 |
| GM9 | filled | GM-FEMA2018-x/GM9/GM9_1m_filled.tif | cfbe4e9b9d0c947a... | 2,607,577 |
| GM9 | fd | GM-FEMA2018-x/GM9/GM9_1m_fd.tif | 976b6e0cf54c96e3... | 209,157 |
| GM9 | area | GM-FEMA2018-x/GM9/GM9_1m_area.tif | ad915fac8a95e775... | 1,183,397 |
| GM9 | gapfill_mask | GM-FEMA2018-x/GM9/GM9_1m_gapfill_mask.tif | c0eac6378f70771b... | 1,866 |
| GM9 | class2_count | GM-FEMA2018-x/GM9/GM9_1m_class2_count.tif | 181eaa2afed1cecf... | 504,076 |
| GM9 | nnfill_mask | GM-FEMA2018-x/GM9/GM9_1m_nnfill_mask.tif | 1d193cc766bbeee6... | 25,405 |
| GM10 | filled | GM-FEMA2018-x/GM10/GM10_1m_filled.tif | e0a48120fb3aae23... | 2,444,400 |
| GM10 | fd | GM-FEMA2018-x/GM10/GM10_1m_fd.tif | 54103f6a5b304fd4... | 255,772 |
| GM10 | area | GM-FEMA2018-x/GM10/GM10_1m_area.tif | 5816897ab061e64d... | 1,117,372 |
| GM10 | gapfill_mask | GM-FEMA2018-x/GM10/GM10_1m_gapfill_mask.tif | cf6b35b4486c37b0... | 1,866 |
| GM10 | class2_count | GM-FEMA2018-x/GM10/GM10_1m_class2_count.tif | 10980d9a6e105bcc... | 517,864 |
| GM10 | nnfill_mask | GM-FEMA2018-x/GM10/GM10_1m_nnfill_mask.tif | 2ff8f7e031e803ac... | 24,525 |
| GM11 | filled | GM-FEMA2018-x/GM11/GM11_1m_filled.tif | 6b4968e46106d996... | 2,668,435 |
| GM11 | fd | GM-FEMA2018-x/GM11/GM11_1m_fd.tif | e73c25f3411dc3c8... | 205,061 |
| GM11 | area | GM-FEMA2018-x/GM11/GM11_1m_area.tif | 310bc06e362b4a6d... | 1,168,462 |
| GM11 | gapfill_mask | GM-FEMA2018-x/GM11/GM11_1m_gapfill_mask.tif | 9a627a949adea7a3... | 1,866 |
| GM11 | class2_count | GM-FEMA2018-x/GM11/GM11_1m_class2_count.tif | 92588aa2c68f5288... | 513,306 |
| GM11 | nnfill_mask | GM-FEMA2018-x/GM11/GM11_1m_nnfill_mask.tif | 5f2e545ddca596ee... | 18,639 |
| GM12 | filled | GM-FEMA2018-x/GM12/GM12_1m_filled.tif | 8dda4ea297200fcf... | 2,607,109 |
| GM12 | fd | GM-FEMA2018-x/GM12/GM12_1m_fd.tif | 06e9033a07425ebf... | 206,575 |
| GM12 | area | GM-FEMA2018-x/GM12/GM12_1m_area.tif | d088f40a8b29ecd5... | 1,232,695 |
| GM12 | gapfill_mask | GM-FEMA2018-x/GM12/GM12_1m_gapfill_mask.tif | b575bbd57b7b85a1... | 1,866 |
| GM12 | class2_count | GM-FEMA2018-x/GM12/GM12_1m_class2_count.tif | 87ad57cc2e647fa5... | 516,397 |
| GM12 | nnfill_mask | GM-FEMA2018-x/GM12/GM12_1m_nnfill_mask.tif | 4bff4cc75bfc6ba8... | 25,831 |
| GM13 | filled | GM-FEMA2018-x/GM13/GM13_1m_filled.tif | 4ec923366ae9d3d8... | 2,612,934 |
| GM13 | fd | GM-FEMA2018-x/GM13/GM13_1m_fd.tif | 7386f412426ba45e... | 202,010 |
| GM13 | area | GM-FEMA2018-x/GM13/GM13_1m_area.tif | 4ba190e17720062a... | 1,194,509 |
| GM13 | gapfill_mask | GM-FEMA2018-x/GM13/GM13_1m_gapfill_mask.tif | deeb70b961792712... | 1,866 |
| GM13 | class2_count | GM-FEMA2018-x/GM13/GM13_1m_class2_count.tif | bb496a8d820c38d4... | 511,059 |
| GM13 | nnfill_mask | GM-FEMA2018-x/GM13/GM13_1m_nnfill_mask.tif | 93ebc7de2570e3db... | 21,680 |
| GM14 | filled | GM-FEMA2018-x/GM14/GM14_1m_filled.tif | f955bee89ddf0e25... | 2,307,187 |
| GM14 | fd | GM-FEMA2018-x/GM14/GM14_1m_fd.tif | e1befa80a47a89cb... | 246,404 |
| GM14 | area | GM-FEMA2018-x/GM14/GM14_1m_area.tif | 4e90c974aa4732cf... | 1,194,076 |
| GM14 | gapfill_mask | GM-FEMA2018-x/GM14/GM14_1m_gapfill_mask.tif | 9abcea0734495026... | 1,866 |
| GM14 | class2_count | GM-FEMA2018-x/GM14/GM14_1m_class2_count.tif | b1671d54c299e2b6... | 529,539 |
| GM14 | nnfill_mask | GM-FEMA2018-x/GM14/GM14_1m_nnfill_mask.tif | ba86905f0ed14d17... | 74,451 |
| GM15 | filled | GM-FEMA2018-x/GM15/GM15_1m_filled.tif | 464298a281ca916a... | 2,560,056 |
| GM15 | fd | GM-FEMA2018-x/GM15/GM15_1m_fd.tif | f8eae4a97d9259c5... | 222,569 |
| GM15 | area | GM-FEMA2018-x/GM15/GM15_1m_area.tif | b40d6f3259862856... | 1,167,115 |
| GM15 | gapfill_mask | GM-FEMA2018-x/GM15/GM15_1m_gapfill_mask.tif | 9abb5576d4a25b25... | 1,866 |
| GM15 | class2_count | GM-FEMA2018-x/GM15/GM15_1m_class2_count.tif | 1be21814e0896629... | 516,422 |
| GM15 | nnfill_mask | GM-FEMA2018-x/GM15/GM15_1m_nnfill_mask.tif | 617d7a4fb8464250... | 38,731 |
| GM16 | filled | GM-FEMA2018-x/GM16/GM16_1m_filled.tif | 2430fd9e3b21b96e... | 2,649,248 |
| GM16 | fd | GM-FEMA2018-x/GM16/GM16_1m_fd.tif | ec82c87221ba27ba... | 201,360 |
| GM16 | area | GM-FEMA2018-x/GM16/GM16_1m_area.tif | 439c97db410163da... | 1,189,046 |
| GM16 | gapfill_mask | GM-FEMA2018-x/GM16/GM16_1m_gapfill_mask.tif | f39fbc346bf4408f... | 1,866 |
| GM16 | class2_count | GM-FEMA2018-x/GM16/GM16_1m_class2_count.tif | f8b245544db62bbc... | 508,370 |
| GM16 | nnfill_mask | GM-FEMA2018-x/GM16/GM16_1m_nnfill_mask.tif | 4d0ca1cb7d76f2b1... | 17,649 |
| GM17 | filled | GM-FEMA2018-x/GM17/GM17_1m_filled.tif | 3adef0174cb61866... | 2,628,860 |
| GM17 | fd | GM-FEMA2018-x/GM17/GM17_1m_fd.tif | fb68ad3241c925ea... | 204,887 |
| GM17 | area | GM-FEMA2018-x/GM17/GM17_1m_area.tif | 103b8c68ae0dee9f... | 1,221,061 |
| GM17 | gapfill_mask | GM-FEMA2018-x/GM17/GM17_1m_gapfill_mask.tif | 5abfddc9cb6821a1... | 1,866 |
| GM17 | class2_count | GM-FEMA2018-x/GM17/GM17_1m_class2_count.tif | fe2af68dde965e15... | 516,044 |
| GM17 | nnfill_mask | GM-FEMA2018-x/GM17/GM17_1m_nnfill_mask.tif | 4ad9c1b084f7865c... | 18,485 |
| GM18 | filled | GM-FEMA2018-x/GM18/GM18_1m_filled.tif | 027fe19a6edddbe9... | 2,364,012 |
| GM18 | fd | GM-FEMA2018-x/GM18/GM18_1m_fd.tif | bc9e60ff5527ff32... | 233,198 |
| GM18 | area | GM-FEMA2018-x/GM18/GM18_1m_area.tif | 337449476d7c8370... | 1,175,000 |
| GM18 | gapfill_mask | GM-FEMA2018-x/GM18/GM18_1m_gapfill_mask.tif | c108d21756dfd88b... | 1,866 |
| GM18 | class2_count | GM-FEMA2018-x/GM18/GM18_1m_class2_count.tif | 187865b0f53e84d1... | 522,484 |
| GM18 | nnfill_mask | GM-FEMA2018-x/GM18/GM18_1m_nnfill_mask.tif | 9dfb9c77bc5d4256... | 37,859 |
| GM19 | filled | GM-FEMA2018-x/GM19/GM19_1m_filled.tif | 3e9ea5bd8500d152... | 2,644,082 |
| GM19 | fd | GM-FEMA2018-x/GM19/GM19_1m_fd.tif | 6e24840ae16d97ac... | 211,271 |
| GM19 | area | GM-FEMA2018-x/GM19/GM19_1m_area.tif | 685d1f77c155245c... | 1,168,505 |
| GM19 | gapfill_mask | GM-FEMA2018-x/GM19/GM19_1m_gapfill_mask.tif | dc2843c84a54fe00... | 1,866 |
| GM19 | class2_count | GM-FEMA2018-x/GM19/GM19_1m_class2_count.tif | 66085bde36b67ca5... | 540,145 |
| GM19 | nnfill_mask | GM-FEMA2018-x/GM19/GM19_1m_nnfill_mask.tif | a0da66f9a6e28782... | 22,445 |
| GM20 | filled | GM-FEMA2018-x/GM20/GM20_1m_filled.tif | f6b6c1932b4a961b... | 2,539,448 |
| GM20 | fd | GM-FEMA2018-x/GM20/GM20_1m_fd.tif | 1ad4f6d4c89d79fb... | 207,546 |
| GM20 | area | GM-FEMA2018-x/GM20/GM20_1m_area.tif | a30e0ed1d4bc2b81... | 1,212,818 |
| GM20 | gapfill_mask | GM-FEMA2018-x/GM20/GM20_1m_gapfill_mask.tif | 63e55363bee5f807... | 1,866 |
| GM20 | class2_count | GM-FEMA2018-x/GM20/GM20_1m_class2_count.tif | 2275cc546eebe82b... | 509,158 |
| GM20 | nnfill_mask | GM-FEMA2018-x/GM20/GM20_1m_nnfill_mask.tif | 3b58e35958672303... | 19,079 |
| GM21 | filled | GM-FEMA2018-x/GM21/GM21_1m_filled.tif | f8abbd7cb10fac82... | 2,499,120 |
| GM21 | fd | GM-FEMA2018-x/GM21/GM21_1m_fd.tif | 692cd82e91f18cbd... | 222,321 |
| GM21 | area | GM-FEMA2018-x/GM21/GM21_1m_area.tif | 3c02debb4b21d698... | 1,176,336 |
| GM21 | gapfill_mask | GM-FEMA2018-x/GM21/GM21_1m_gapfill_mask.tif | 1f6c1261d6d27dc8... | 1,866 |
| GM21 | class2_count | GM-FEMA2018-x/GM21/GM21_1m_class2_count.tif | dc7cd01eea842123... | 521,900 |
| GM21 | nnfill_mask | GM-FEMA2018-x/GM21/GM21_1m_nnfill_mask.tif | 8dc55d6766217e4c... | 44,748 |
| GM22 | filled | GM-FEMA2018-x/GM22/GM22_1m_filled.tif | 4c779bf3b0f76c4b... | 2,691,852 |
| GM22 | fd | GM-FEMA2018-x/GM22/GM22_1m_fd.tif | 8cfbc4094c79b584... | 194,023 |
| GM22 | area | GM-FEMA2018-x/GM22/GM22_1m_area.tif | 37c05a8cdfbf54bd... | 1,178,138 |
| GM22 | gapfill_mask | GM-FEMA2018-x/GM22/GM22_1m_gapfill_mask.tif | e0f4985ce7d21c29... | 1,866 |
| GM22 | class2_count | GM-FEMA2018-x/GM22/GM22_1m_class2_count.tif | a91057b9b275b84f... | 492,729 |
| GM22 | nnfill_mask | GM-FEMA2018-x/GM22/GM22_1m_nnfill_mask.tif | 5c4d38f92e5ce22f... | 11,148 |
| GM23 | filled | GM-FEMA2018-x/GM23/GM23_1m_filled.tif | fa230d3eb953a940... | 2,588,601 |
| GM23 | fd | GM-FEMA2018-x/GM23/GM23_1m_fd.tif | daa780664e6053ed... | 224,274 |
| GM23 | area | GM-FEMA2018-x/GM23/GM23_1m_area.tif | a7a3da6eb9e8315e... | 1,168,794 |
| GM23 | gapfill_mask | GM-FEMA2018-x/GM23/GM23_1m_gapfill_mask.tif | 95f67f11eb6d91cf... | 1,866 |
| GM23 | class2_count | GM-FEMA2018-x/GM23/GM23_1m_class2_count.tif | c96763a54b5b92c3... | 519,618 |
| GM23 | nnfill_mask | GM-FEMA2018-x/GM23/GM23_1m_nnfill_mask.tif | 149976b91674e9e5... | 32,569 |
| GM24 | filled | GM-FEMA2018-x/GM24/GM24_1m_filled.tif | d41acb705ca5ae59... | 2,561,027 |
| GM24 | fd | GM-FEMA2018-x/GM24/GM24_1m_fd.tif | 354e0ca983caeea5... | 204,900 |
| GM24 | area | GM-FEMA2018-x/GM24/GM24_1m_area.tif | 3ab8d08a14103ac0... | 1,209,743 |
| GM24 | gapfill_mask | GM-FEMA2018-x/GM24/GM24_1m_gapfill_mask.tif | 06df04740ed6b05e... | 1,866 |
| GM24 | class2_count | GM-FEMA2018-x/GM24/GM24_1m_class2_count.tif | b7dbf852e65e5554... | 504,741 |
| GM24 | nnfill_mask | GM-FEMA2018-x/GM24/GM24_1m_nnfill_mask.tif | 5d17267520d958fe... | 18,045 |
| GM25 | filled | GM-FEMA2018-x/GM25/GM25_1m_filled.tif | d9e873cd823bd387... | 2,606,851 |
| GM25 | fd | GM-FEMA2018-x/GM25/GM25_1m_fd.tif | 648dd05be7bfa22a... | 224,487 |
| GM25 | area | GM-FEMA2018-x/GM25/GM25_1m_area.tif | 34657921d80ca4a3... | 1,194,859 |
| GM25 | gapfill_mask | GM-FEMA2018-x/GM25/GM25_1m_gapfill_mask.tif | 502bf2ee578697af... | 1,866 |
| GM25 | class2_count | GM-FEMA2018-x/GM25/GM25_1m_class2_count.tif | 1c9842965f51977f... | 519,838 |
| GM25 | nnfill_mask | GM-FEMA2018-x/GM25/GM25_1m_nnfill_mask.tif | 8ac9371127c66d14... | 44,320 |
| GM26 | filled | GM-FEMA2018-x/GM26/GM26_1m_filled.tif | 273647897cb523a2... | 2,614,300 |
| GM26 | fd | GM-FEMA2018-x/GM26/GM26_1m_fd.tif | 571c5911c54cd94e... | 210,437 |
| GM26 | area | GM-FEMA2018-x/GM26/GM26_1m_area.tif | da4fb32eb8a5fa14... | 1,205,988 |
| GM26 | gapfill_mask | GM-FEMA2018-x/GM26/GM26_1m_gapfill_mask.tif | 39299d18b6d7c149... | 1,866 |
| GM26 | class2_count | GM-FEMA2018-x/GM26/GM26_1m_class2_count.tif | 618086a8a070dfcf... | 514,508 |
| GM26 | nnfill_mask | GM-FEMA2018-x/GM26/GM26_1m_nnfill_mask.tif | f0e628128ba9de25... | 18,631 |
| GM27 | filled | GM-FEMA2018-x/GM27/GM27_1m_filled.tif | fcefa2d86178b732... | 2,490,607 |
| GM27 | fd | GM-FEMA2018-x/GM27/GM27_1m_fd.tif | f118c5d643d1c5aa... | 237,182 |
| GM27 | area | GM-FEMA2018-x/GM27/GM27_1m_area.tif | 64063c296e7fbb12... | 1,175,129 |
| GM27 | gapfill_mask | GM-FEMA2018-x/GM27/GM27_1m_gapfill_mask.tif | 52e984782a72cb09... | 1,866 |
| GM27 | class2_count | GM-FEMA2018-x/GM27/GM27_1m_class2_count.tif | e84c71b404864584... | 528,200 |
| GM27 | nnfill_mask | GM-FEMA2018-x/GM27/GM27_1m_nnfill_mask.tif | 8e2c900dd4f5ccff... | 66,962 |
| GM28 | filled | GM-FEMA2018-x/GM28/GM28_1m_filled.tif | ab3a00807d4bb49c... | 2,393,823 |
| GM28 | fd | GM-FEMA2018-x/GM28/GM28_1m_fd.tif | 4632a97fd0af9672... | 236,605 |
| GM28 | area | GM-FEMA2018-x/GM28/GM28_1m_area.tif | 11d3ad33add80c49... | 1,158,800 |
| GM28 | gapfill_mask | GM-FEMA2018-x/GM28/GM28_1m_gapfill_mask.tif | cf0ce372ede105ce... | 1,866 |
| GM28 | class2_count | GM-FEMA2018-x/GM28/GM28_1m_class2_count.tif | 6ef8dbc88676e7a3... | 510,842 |
| GM28 | nnfill_mask | GM-FEMA2018-x/GM28/GM28_1m_nnfill_mask.tif | 774b034e87ef72ca... | 63,287 |
| GM29 | filled | GM-FEMA2018-x/GM29/GM29_1m_filled.tif | 425b0e13e87497a7... | 2,569,861 |
| GM29 | fd | GM-FEMA2018-x/GM29/GM29_1m_fd.tif | feff2c8d8f385a8f... | 206,585 |
| GM29 | area | GM-FEMA2018-x/GM29/GM29_1m_area.tif | e3c4725bfbb45612... | 1,202,911 |
| GM29 | gapfill_mask | GM-FEMA2018-x/GM29/GM29_1m_gapfill_mask.tif | 5b604ca3d8f8c972... | 1,866 |
| GM29 | class2_count | GM-FEMA2018-x/GM29/GM29_1m_class2_count.tif | 206024d9e4ac2f31... | 518,941 |
| GM29 | nnfill_mask | GM-FEMA2018-x/GM29/GM29_1m_nnfill_mask.tif | 46778a10de28ed91... | 36,367 |
| GM30 | filled | GM-FEMA2018-x/GM30/GM30_1m_filled.tif | c53d59a3069aa029... | 2,553,689 |
| GM30 | fd | GM-FEMA2018-x/GM30/GM30_1m_fd.tif | 2d28b22c3b71c8ef... | 224,468 |
| GM30 | area | GM-FEMA2018-x/GM30/GM30_1m_area.tif | 2fb5e462056a13a4... | 1,187,342 |
| GM30 | gapfill_mask | GM-FEMA2018-x/GM30/GM30_1m_gapfill_mask.tif | 5e8ef80bc56a2121... | 1,866 |
| GM30 | class2_count | GM-FEMA2018-x/GM30/GM30_1m_class2_count.tif | 7eada1c627e2b64e... | 538,370 |
| GM30 | nnfill_mask | GM-FEMA2018-x/GM30/GM30_1m_nnfill_mask.tif | 0b88a076b93204e4... | 33,038 |

### C13. The registered tile list per site (section 6), from configs/data.yaml, the stage B and a16 tile records
| site | tile | version | footprint SW corner (E, N) | area source for modelling | kind |
|---|---|---|---|---|---|
| GM | GM1 | GM-x | 700991, 3980451 (EPSG:32610) | area_model (new area + coarse inflow added at the entry cells and propagated downstream (check failed)) | original |
| GM | GM2 | GM-x | 698991, 3980451 (EPSG:32610) | area_model (new area + coarse inflow added at the entry cells and propagated downstream (check failed)) | original |
| GM | GM3 | GM-x | 695991, 3978451 (EPSG:32610) | area_model (old tile area grid (check passed)) | original |
| GM | GM4 | GM-x | 694991, 3976451 (EPSG:32610) | area_model (new area + coarse inflow added at the entry cells and propagated downstream (check failed)) | original |
| GM | GM5 | GM-x | 699991, 3981451 (EPSG:32610) | area_model (new area + coarse inflow added at the entry cells and propagated downstream (check failed)) | original |
| GM | GM6 | GM-x | 697481, 3978961 (EPSG:32610) | full-grid routing (role area) | drawn |
| GM | GM7 | GM-x | 697981, 3977961 (EPSG:32610) | full-grid routing (role area) | drawn |
| AP | AP1_3dep | AP-3DEP-x | 568503, 4408340 (EPSG:6346) | full-grid routing (role area) | original |
| AP | AP2_3dep | AP-3DEP-x | 561573, 4426160 (EPSG:6346) | full-grid routing (role area) | original |
| AP | AP3_3dep | AP-3DEP-x | 551854, 4419230 (EPSG:6346) | full-grid routing (role area) | original |
| AP | AP4_3dep | AP-3DEP-x | 555093, 4419230 (EPSG:6346) | full-grid routing (role area) | original |
| AP | AP5_3dep | AP-3DEP-x | 556893, 4422470 (EPSG:6346) | full-grid routing (role area) | original |
| AP | AP6 | AP-3DEP-x | 566493, 4407378 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP7 | AP-3DEP-x | 556083, 4420246 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP8 | AP-3DEP-x | 552633, 4420755 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP9 | AP-3DEP-x | 570452, 4418776 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP10 | AP-3DEP-x | 558573, 4407378 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP11 | AP-3DEP-x | 563523, 4412837 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP12 | AP-3DEP-x | 559053, 4413317 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP13 | AP-3DEP-x | 555603, 4410347 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP14 | AP-3DEP-x | 565983, 4420755 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP15 | AP-3DEP-x | 568472, 4420755 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP16 | AP-3DEP-x | 562023, 4418776 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP17 | AP-3DEP-x | 570932, 4421745 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP18 | AP-3DEP-x | 574892, 4425705 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP19 | AP-3DEP-x | 564003, 4424205 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP20 | AP-3DEP-x | 565503, 4424715 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP21 | AP-3DEP-x | 558063, 4416286 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP22 | AP-3DEP-x | 555093, 4414307 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP23 | AP-3DEP-x | 555093, 4408368 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP24 | AP-3DEP-x | 566493, 4423215 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP25 | AP-3DEP-x | 560553, 4416286 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP26 | AP-3DEP-x | 560043, 4425195 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP27 | AP-3DEP-x | 555603, 4412837 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP28 | AP-3DEP-x | 576872, 4416796 (EPSG:6346) | full-grid routing (role area) | drawn |
| AP | AP29 | AP-3DEP-x | 562533, 4417276 (EPSG:6346) | full-grid routing (role area) | drawn |

TTV: no registered version (the 15 cut tiles are parked unregistered, section 2.4). Registered tiles in all: 36.

### C12. Slurm accounting of the addendum 16 jobs
| JobID | JobName | Partition | State | Elapsed | Timelimit | AllocCPUS | MaxRSS | TotalCPU |
|---|---|---|---|---|---|---|---|---|
| 52834227 | a16_tests | serc | COMPLETED | 00:04:00 | 00:30:00 | 2 |  | 00:00:00 |
| 52834763 | a16_ttvclip | serc | COMPLETED | 00:00:29 | 00:30:00 | 1 |  | 00:00:00 |
| 52834765 | a16_ttvref | serc | COMPLETED | 00:01:33 | 00:30:00 | 1 |  | 00:00:00 |
| 52834766 | a16_draw | serc | COMPLETED | 00:00:54 | 00:30:00 | 2 |  | 00:00:00 |
| 52834767 | a16_cut | serc | COMPLETED | 00:00:36 | 00:30:00 | 1 |  | 00:00:00 |
| 52835250 | a16_gmfoot | serc | COMPLETED | 00:01:25 | 00:40:00 | 1 |  | 00:00:00 |
| 52835252 | a16_gmarea | serc | COMPLETED | 00:00:26 | 00:30:00 | 1 |  | 00:00:00 |
| 52837608_0 | a16_reg | serc | COMPLETED | 00:02:20 | 00:40:00 | 1 |  | 00:00:00 |
| 52837608_1 | a16_reg | serc | COMPLETED | 00:02:22 | 00:40:00 | 1 |  | 00:00:00 |
| 52837804 | a16_gmgate | serc | CANCELLED by 313293 | 00:09:55 | 02:00:00 | 1 |  | 00:00:00 |
| 52837838 | a16_gmdiag | serc | COMPLETED | 00:06:18 | 00:30:00 | 1 |  | 00:00:00 |
| 52838424 | a16_regcheck | serc | COMPLETED | 00:00:16 | 00:30:00 | 1 |  | 00:00:00 |
| 52840477 | a16_gmgate | serc | COMPLETED | 01:46:51 | 02:00:00 | 1 |  | 00:00:00 |
| 52840484 | a16_gmdiag | serc | COMPLETED | 00:06:25 | 00:30:00 | 1 |  | 00:00:00 |
| 52842281 | a16_gmdiag | serc | COMPLETED | 00:04:45 | 00:30:00 | 1 |  | 00:00:00 |
| 52853733 | a16_extfp | serc | COMPLETED | 00:04:51 | 00:30:00 | 1 |  | 00:00:00 |
| 52855272 | a16_extfp | serc | COMPLETED | 00:05:31 | 00:30:00 | 1 |  | 00:00:00 |
| 52856554 | a16_extfetch | serc | COMPLETED | 00:07:25 | 00:30:00 | 8 |  | 00:00:00 |
| 52856556 | a16_extgrid | serc | COMPLETED | 00:12:51 | 02:00:00 | 4 |  | 00:00:00 |
| 52856563 | a16_extdraw | serc | COMPLETED | 00:05:45 | 00:40:00 | 1 |  | 00:00:00 |
| 52858552 | a16_copyrec | serc | COMPLETED | 00:02:46 | 00:30:00 | 1 |  | 00:00:00 |

Peak memory per job (GB, from the step records): 52834227: 1.2; 52834763: 2.5; 52834765: 1.2; 52834766: 2.1; 52834767: 0.6; 52835250: 10.0; 52835252: 1.9; 52837608_0: 3.2; 52837608_1: 3.4; 52837804: 3.7; 52837838: 5.7; 52838424: 0.4; 52840477: 4.5; 52840484: 4.9; 52842281: 4.2; 52853733: 2.7; 52855272: 2.7; 52856554: 3.1; 52856556: 44.1; 52856563: 7.7; 52858552: 4.2
