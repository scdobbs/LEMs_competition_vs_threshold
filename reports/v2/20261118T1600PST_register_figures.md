# [2026-11-18 16:00 PST] Addendum 17 Part A: the tile set registered, the GM survey-control tiles, the figures for visual review

Directive: `reports/directives/2026-11-18T1500PST_v2_addendum_17_register_figures_measure.md` (Part A, outline fixed by its section 4). Branch
`v2.3-tiles` (from ff5cb9c, the addendum 16 report). **Nothing was measured on the 1 m tiles; no model was run; no merge, no tag. Part B (the
per-tile measurement) has not started and waits for Steve's written go-ahead with the exclusion list.** Scripts: `reports/v2/scripts/a17_common.py`,
`a17_area_model.py`, `a17_register_check.py`, `a17_gm_ctrl_footprint.py`, `a17_gm_ctrl_tiles.py`, `a17_gm_ctrl_area_check.py`, `a17_figures.py`,
`make_a17_tables.py` (ruff clean); `a15_common.py` (the decision-1 windows), `a15_cut_tiles.py`, `a14_grid_route.py`, `a15_reroute.py` and
`a16_gmf_check.py` gained the control site `GMC` (section 1). Environments: `lemcvt` (everything but the point clouds), `lidar` (PDAL fetch,
gridding), `ci312` (ruff). Every number is tied to the run directory `$GROUP_HOME/lemcvt_results/20261118T225314Z_a17_register_figures/`
(subdirectory named in each section) or to a Slurm job id. Appendices (section 6) are machine-generated. **This is a stop for review.**

## Summary

| item | result |
| --- | --- |
| 1 state | `v2.3-tiles` = origin = ff5cb9c; no job running; tests on `serc` **274 passed** (job 52862632, 4 min 42 s); ruff clean. Deviations in section 1, including one correction re-run (the control footprints were first floored 1 m south; unregistered and re-cut, section 3.3) |
| 2 registration | Decision 1 applied: windows TTV 0-800 m, GM-FEMA 0-750 m re-run on the parked tiles: **15 of 15 TTV and 23 of 23 GM-extension tiles pass** (4 and 10 had failed the old windows) and the re-cut files are **byte-identical** to the addendum 16 parked cut; registered as `TTV-x` and `GM-FEMA2018-x`. Decision 2: GM1-GM5 `area_model` = the old grids, the rule grids kept as `area_model_rule` (GM5's rule grid was 2.31 km² short on its second trunk). Decision 3 recorded. **Registered in all: 81 tiles** (GM 7 NCALM + 23 FEMA + 7 control, AP 29, TTV 15); the 74 non-control tiles are the measurement set |
| 3 GM control | Footprint = GM1-GM7 + their catchments on the NED box routing + 500 m = **43.7 km²** inside the FEMA survey; 66 one-km boxes, 49 already on record, **17 fetched** (0.33 GB, 50 s) and added to the copy of record. Grid 10,211 × 10,851 at 1 m (3.79 pts m⁻², 2.2 % nearest-neighbour filled), gap rule (17 enclosed gaps, 0.004 km²), closed budget. **Rule 5.3 passes** (NED − grid +0.09 m, Theil-Sen 0.9997); NCALM lies 33.84 m below (slope 1.0012): no units stop. **GM1_fema-GM7_fema cut at the NCALM corners exactly** (the offline EPSG:32610 → 6339 transformation moves the corners by 1 µm E and −0.1 mm N; snapped to the nearest lattice line), all pass 0-750 m, registered as `GM-FEMA2018-ctrl`. **Area check (report only): at GM5's second trunk, the one the addendum 16 rule lost, FEMA gives 2.653 km² against the old grid's 2.667 (−0.5 %)**: a third, independent confirmation of decision 2. But at GM2's trunk (+15.5 %) and GM5's first trunk (+48 %) the FEMA routing carries more area than the old grids, as the NED box did (+29 %, +53 %), and GM3 receives 2.48 km² from outside that its lidar-only old grid does not have (section 3.4, decision needed) |
| 4 figures | **11 figures** in `reports/v2/figures/a17/` (PNG 200 dpi, all under 5 MB, 33 MB in all; PDFs beside them, not committed; `README.md` index): three overviews, six gallery pages (81 panels), the GM survey-control pairs, the AP PAMAP/3DEP pairs; rendering choices in section 4; candidate exclusions I noticed (GM10; the high eastern mesa GM14/18/21/27/28; AP17's highway; well pads on AP16/21/22/25/26; TTV12/13's graded ground) listed in section 5 for Steve's decision |
| 5 decisions | Tile exclusions (Steve's, from the figures); plus three area questions raised by the control grid (section 5) |
| cost | 10 submissions, **1.12 core-hours used** (expected ≤ 20), 12.0 core-hours of CPU-weighted limits, every job on `serc`, largest 4 CPUs × 30 min, peak 9.6 GB; +1.1 GB on group storage (17 LAZ on record 0.33 GB, run directory 433 MB, 274 MB of registered tiles, 33 MB of PNGs); scratch staging (1.3 GB) removed; `$GROUP_HOME` at 320.5 GB of 1 TB |

## 1. State check

**At the start (2026-11-18 14:48 PST, login node sh03-ln05).**
- `v2.3-tiles` = `origin/v2.3-tiles` = ff5cb9c (the addendum 16 report); no job was running (`squeue --me` empty).
- Tests on `serc`: **274 passed in 4 min 42 s** (job 52862632, 2 CPUs, at ff5cb9c with a clean tree); ruff (`ci312`) clean on `src` and `tests`, and
  on every a17 script and every edited a14-a16 script (login node).
- The paths of section 0.3 exist (the addendum 16 run directory with `tiles/TTV-x_unregistered/`, `tiles_gmf/GM-FEMA2018-x_unregistered/`,
  `grid_GMF/`, `ept_GMF/`, `draw/`, `draw_gmf/`, `gm_area_check/`, `gm_area_diag/`, `md_georef/`, `ttv_reference/`, `gm_footprints/`, `gm_gate/`; the
  stage B and stage A run directories; the copies of record; `configs/data.yaml`, `data/manifest.csv`; the papers; the three environments).
- Read in the order of section 0.2.

**Where the addendum and the repo or the data differ, and every deviation** (each handled as stated; no stop):
1. **The decision-1 window is also applied to the control grid** (`Z_RANGE["GMC"] = (0, 750)`, the GM-FEMA window): the directive names TTV and
   GM-FEMA only; the control tiles are on the same survey and the same lattice, so the same units test applies. Written into `a15_common.py`
   before any cut ran, with the old values kept in the comment and in `a17_common.Z_RANGE_A15`.
2. **Figure PDFs are written to `reports/v2/figures/a17/` as the directive says but are not committed** (`.gitignore`:
   `reports/v2/figures/a17/*.pdf`): the standing rule (never commit PDFs; the addendum 2 §5 check treats any `*.pdf` as forbidden) wins. The PNGs
   are committed; the PDFs are also kept in the run directory (`figures/`).
3. **The first control cut put every control tile 1 m south of its NCALM footprint and was unregistered** (section 3.3). `a17_gm_ctrl_tiles.py`
   floored the reprojected south-west corner to the lattice, as `a15_cut_tiles.py` does for the AP originals (whose corners are not integers);
   here the offline EPSG:32610 → EPSG:6339 transformation moves the integer NCALM corners by −0.107 mm in northing, so the floor fell to the
   next lattice line south. Found on reading the cut log (`fp_y0` = 3980450 against 3980451); the 42 manifest rows and the `GM-FEMA2018-ctrl`
   block were removed from `data/manifest.csv` and `configs/data.yaml` on the login node, the records of the first cut were renamed
   (`tiles_gmc_floor_shift/`, `draw_gmc_floor_shift/`, `gm_ctrl_area_first/`), the script now snaps to the **nearest** lattice line, and the cut,
   the area check and the registration were re-run (job 52868585; first cut 52865321). The figures job that had started on the shifted tiles
   (52865322) was cancelled after one minute and resubmitted behind the re-cut (52868593). The area check does not depend on the cut (it reads
   the control grid at world coordinates) and gave the same numbers both times.
4. **The addendum 16 scratch copies of the FEMA LAZ were gone** (removed after the copy of record, as that report says), so `ept_files_6339.txt`
   pointed at files that no longer exist. The 49 record boxes inside the control footprint were staged back to `$SCRATCH/a17/ept_GMC/` by the
   fetch job (rsync, 0.9 GB) and verified against the record `MANIFEST.sha256` before gridding; the 17 new boxes went to the same directory and
   were then copied to the record (`MANIFEST_a17_control.sha256`, appended to `MANIFEST.sha256` and `MANIFEST_all.sha256`; the record now
   holds 309 LAZ, 6.1 GB). The scratch copies (1.3 GB) are removed in the cleanup of section 6.1.
5. **The fetch job's last step, writing the LAZ list for the grid job, failed** (`KeyError: 'RUN_DIR'`: the variable was not exported to the
   embedded Python; the job itself exited 0). The list (66 files) was written on the login node from the same box record before the grid job,
   pending on `afterok`, started; the generator `a17_chain.sh` now exports the variables. No job was resubmitted for this.
6. **`a15_cut_tiles.py` records the unit window** it applied in `configs/data.yaml` (`unit_check_window_m`, `unit_check_note`) and takes
   `--unit-window-note`; `a14_grid_route.py`, `a15_reroute.py`, `a16_gmf_check.py` (`--site GMC`, output `gmc_check.json`) and
   `a15_cut_tiles.py` accept the control site; `a15_common.TILE_EPSG["GMC"] = 6339`. Defaults and the GMF code paths are unchanged.
7. **Registration re-cut the parked tiles from the same grids rather than moving the parked files**: the cut script computes the records
   (unit check, checksums) as it writes, so the 15 + 23 tiles were cut again into `$LEMCVT_DATA/TTV-x/` and `GM-FEMA2018-x/` (80 s in all) and
   every product's sha256 was compared with the addendum 16 parked copy: **all 228 files identical** (`register/register_check.json`). The parked
   copies (187 MB) are left where the addendum 16 report put them; they are now redundant.
8. **`area_model` has no consumer in `src/lemcvt`** (`grep area_model src/` is empty): the tile loader reads the `area` role. Part B must load
   the `area_model` grid explicitly for GM1-GM5 (and the figures of section 3 do so). Recorded for Part B; nothing changed in the package.
9. **Decision 7's list is confirmed and extended by the data**: on the registered area grids GM2 (8.89 km²), GM4 (6.19), AP3 (11.07) and AP4
   (17.14) exceed 5 km², and so do the control copies GM2_fema (10.34), GM4_fema (6.19) and **GM5_fema (5.54; GM5 on the old grid is 4.73)**.
   The lidar-only cut of GM2 (4.80) would not have qualified; the list is made on the `area_model` grids (Appendix A6).
10. **The GM overview shows the FEMA 2018 hillshade** (the addendum 16 extension grid and the control grid mosaicked at 5 m, both on the same
    lattice), with the NCALM 2003 footprint as a dashed outline: the directive's "full 1 m grid" is two grids at GM. The eligible windows drawn
    are the stage B NCALM pool (eligible at ±50 %) and the addendum 16 gate's FEMA pool (49 at ±25 %).
11. **Gallery statistics**: "largest A inside" is taken from the tile's registered area source (`area_model` for GM1-GM5, `area` otherwise), so
    for GM2 it is 8.89 km² and not the lidar-only 4.80; the nearest-neighbour-filled fraction is "n/a" for the NCALM tiles (`GM-x` carries no
    fill mask: the NCALM grid was delivered as a raster) and for the PAMAP tiles of the AP pairs.
12. **The NCALM-overlap flag of `a16_gmf_check.py`** now tests the offset's magnitude (corrected in addendum 16 after its run): `True` here.
13. `cry_for_help.txt`, `working-with-steve.md`, `papers/` and the other untracked drafts stay untracked (pre-push check, section 6.2).

## 2. Registration

### 2.1 The unit checks under the decision-1 windows and the registration
Job 52863568 (`a15_cut_tiles.py --sites TTV` from the stage B grid and the addendum 16 draw; `--sites GMF` from the addendum 16 grid and
draw; `a17_area_model.py`; `a17_register_check.py`); records `tiles_ttv/tiles.json`, `tiles_gmf/tiles.json`, `register/register_check.json`.
Appendix A1.

| version | tiles | window (m) | old window (m) | pass | failed the old window | identical to the a16 parked cut | registered |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `TTV-x` | 15 (TTV1-TTV3 M&D by design, TTV4-TTV15 drawn) | **0-800** | 0-320 | **15 / 15** | 4 (TTV4 321 m, TTV5 445-656, TTV7 389, TTV9 369) | 90 / 90 files | yes (90 manifest rows) |
| `GM-FEMA2018-x` | 23 (GM8-GM30, drawn) | **0-750** | 68-468 | **23 / 23** | 10 (GM12, 14, 15, 18, 21, 23, 27, 28, 29, 30: 497-695 m) | 138 / 138 files | yes (138 manifest rows) |

The highest tile is GM28 (543-695 m, the mesa top near the range front) and TTV5 (445-656 m, the Mount Tamalpais flank); both are well inside
the windows. The windows and the note "a units test" are recorded in `configs/data.yaml` beside each version. No draw was changed (decision 1).

### 2.2 GM1-GM5 modelling area (decision 2)
`a17_area_model.py` (same job); record `area_model/area_model.json`. Appendix A2. For each of GM1-GM5 the addendum 16 rule grid was renamed
`<tile>_1m_area_model_rule.tif` (kept, unused, five new manifest rows) and `<tile>_1m_area_model.tif` rewritten as a copy of the old tile area
grid `GM/tile<n>/GM_1m_best_tile_area.tif` (float64, the GM-x transform, which the old grid shares); the five `area_model` manifest rows carry
the new checksums; `configs/data.yaml` names both roles.

| tile | `area_model` (= old grid) outlet area (km²) | rule grid outlet (km²) | lidar-only outlet (km²) | cells where the rule grid is smaller | largest shortfall of the rule grid |
| --- | --- | --- | --- | --- | --- |
| GM1 | 1.158 | 1.154 | 1.148 | 764 | 4,133 m² |
| GM2 | 8.895 | 8.782 | 4.804 | 692 | 0.113 km² |
| GM3 | 0.337 | 0.337 (identical) | 0.337 | 0 | 0 |
| GM4 | 6.186 | 6.186 | 6.186 | 508 | 16 m² |
| GM5 | **4.725** | **2.956** | 2.732 | 1,794 | **2.306 km²** (the second northern trunk) |

### 2.3 The landscape polygon (decision 3)
`a16_common.GM_LANDSCAPE_POLY_UTM` stands as written in addendum 16; nothing was recomputed.

### 2.4 The final registered tile list
Appendix A6 lists every tile with its version, survey, footprint, area source and kind. In short:

| site | registered | version (CRS) | survey | kinds | area source for modelling |
| --- | --- | --- | --- | --- | --- |
| GM | **7** GM1-GM7 | `GM-x` (EPSG:32610) | NCALM 2003 | 5 original, 2 drawn | GM1-GM5 `area_model` (old grids); GM6-GM7 full-grid routing |
| GM | **23** GM8-GM30 | `GM-FEMA2018-x` (EPSG:6339) | FEMA 2018 | drawn | full-grid routing of the 167 km² extension grid |
| GM | **7** GM1_fema-GM7_fema | `GM-FEMA2018-ctrl` (EPSG:6339) | FEMA 2018 | control | full-grid routing of the 44 km² control grid (the tiles' whole catchments) |
| AP | **29** AP1_3dep-AP5_3dep, AP6-AP29 | `AP-3DEP-x` (EPSG:6346) | 3DEP 2019-20 | 5 original, 24 drawn | full-grid routing |
| TTV | **15** TTV1-TTV15 | `TTV-x` (EPSG:6339) | Marin 2019 | 3 M&D by design, 12 drawn | full-grid routing |

**81 tiles in all; 74 without the controls** (GM 30, AP 29, TTV 15). `data/manifest.csv` has 758 rows (483 before). **Decision 6** (E, m/yr): GM
1.47 × 10⁻⁴ (Perron et al. 2012) for every GM tile, NCALM and FEMA alike, including the control copies; AP 5.5 × 10⁻⁵; TTV 7.7 × 10⁻⁵ (Heimsath
et al. 1997, 77 ± 15 m/Myr); carried over to the extension tiles, recorded in `a17_common.E_M_PER_YR`. **Decision 7** (for addendum 18): the
tiles whose largest drainage area inside exceeds 5 km² on the registered area source are **GM2 (8.89 km²), GM4 (6.19), AP3 (11.07), AP4
(17.14)**, and the control copies GM2_fema (10.34), GM4_fema (6.19) and GM5_fema (5.54); GM5 on its old grid (4.73) is below. Next largest:
AP13 3.97, AP7 3.65, GM28 3.26, AP25 3.27, TTV9 2.75, TTV3 2.71.

## 3. The GM survey-control tiles (decision 4)

### 3.1 Footprint and fetch (section 2.2.1)
Script `a17_gm_ctrl_footprint.py` (job 52863574, 5 min 38 s; NED box routing 245 s); products `gm_ctrl/GMC_fetch_footprint.geojson` (EPSG:32610),
`GMC_fetch_footprint_10m.tif`, `GMC_catchments_10m.tif`, `GMC_boxes.json`, `GMC_fetch_missing.geojson`, `GMC_existing_files.txt`. Fetch
`a14_ept_fetch.py` (job 52865313, `lidar`, 4 processes); record `ept_GMC/ept_fetch_6339.json`, `ept_files_GMC.txt`. Appendix A3.

The 10 m NED was routed over the addendum 16 box (683-718 km E, 3,958-4,005 km N, every valid cell, data-edge outlets: the `gm_area_diag`
conventions) and the upstream catchment of each registered GM1-GM7 footprint taken by propagating the tile flag from receivers to donors:

| tile | catchment on the NED box, tile included (km²) | upstream outside the tile (km²) | largest NED area inside (km²) |
| --- | --- | --- | --- |
| GM1 | 1.97 | 0.97 | 1.16 |
| GM2 | **11.25** | 10.25 | 10.33 |
| GM3 | 3.51 | 2.51 | 2.52 |
| GM4 | **9.07** | 8.07 | 6.18 |
| GM5 | **8.58** | 7.58 | 5.56 |
| GM6 | 2.04 | 1.04 | 1.41 |
| GM7 | 2.01 | 1.01 | 1.25 |

Union of tiles and catchments **23.9 km²**, every cell inside the FEMA survey; with the 500 m buffer **43.7 km²** (bounding box 694,040-704,880 E,
3,975,850-3,986,050 N; 110.8 M cells at 1 m). It is cut by **66 one-km boxes** of the EPSG:6339 lattice (the `a14_ept_fetch` boxes), of which
**49 are in the addendum 16 copy of record** and **17 were missing** (the column 698-699 km E west of GM2 and GM5, 694-697 km E south of GM4,
and 703 km E east of GM5: the parts of the tiles' catchments that no eligible extension window drained through). The 17 were read from the
Entwine copy of `CA_FEMA_Z4_B2_2018` as in addendum 16 §3.3 (class 2, EPSG:6339, compound CRS EPSG:6339+5703 with the same vertical-evidence
string): **17 of 17 boxes, 0.33 GB in 50 s**, copied to the record with their sha256 manifest.

### 3.2 Grid, routing, units (section 2.2.2)
Job 52865319 (`a14_grid_route.py --site GMC --stage grid` in `lidar`; `a15_reroute.py --site GMC`; `a16_gmf_check.py --site GMC`), 4 CPUs,
5 min 57 s in all, peak 6.0 GB; products `grid_GMC/GMC_1m_{mean,filled,fd,area_cells,class2_count,filled_mask,nn_fill_dist,gapfill_mask}.tif`,
`GMC_grid.json`, `GMC_gaps.json`, `GMC_route.json`, `gmc_check.json`. Appendix A3.

- **Grid**: lattice 10,211 × 10,851 cells (origin 694,035 E, 3,986,055 N, EPSG:6339; the same integer-metre lattice as the extension grid);
  43.73 M cells inside the footprint, 42.78 M (97.8 %) with a class-2 point; 240.9 M points, **3.79 points m⁻²** (5-50-95 % per cell 1-3-8);
  944,674 cells (2.2 %) filled by nearest neighbour within 10 m (median 1 m), 4,218 left nodata. Flight dates 2018-02-07 to 02-15 (adjusted
  GPS time, inferred). 68 s, 5.1 GB.
- **Gap rule and routing**: 18 nodata components, **17 enclosed (4,355 cells, 0.004 km², none above 1 ha) filled for routing**; priority flood
  and D8 over the whole grid; **43,730,787 valid cells, 47,916 data-edge outlets whose areas sum to the valid count (closed budget)**; no pad
  needed. Largest basins **14.98 km²** at 697,526 E, 3,980,480 N (GM2's trunk where it leaves the footprint west of the tile) and **9.56 km²** at
  694,120 E, 3,977,082 N (GM4's trunk at the western edge). 59 s, 5.9 GB.
- **Rule 5.3 against the 10 m NED** (226,978 fully covered 10 m cells): median NED − grid **+0.088 m** (5-95 % −0.89 to +0.99, RMS 0.55),
  Theil-Sen slope **0.99970** (0.99961-0.99990): **pass** (addendum 16 on the extension grid: +0.058 m, 0.99995). **NCALM 2003 on the overlap**
  (1,160,536 sampled cells): median NCALM − grid **−33.84 m** (−34.11 to −33.61), slope **1.0012**: the expected ellipsoid-geoid offset, in the
  30-34 m window; no slope near 3.28 or 0.305, **no stop** (addendum 16: −33.85 m, 1.0012).

### 3.3 The cut (section 2.2.3)
Scripts `a17_gm_ctrl_tiles.py` and `a15_cut_tiles.py --sites GMC` (job 52868585; the first cut, job 52865321, is deviation 3 of section 1);
products `draw_gmc/GMC_tiles.csv`, `GMC_footprints.json`, `tiles_gmc/tiles.json`; data `$LEMCVT_DATA/GM-FEMA2018-ctrl/GM<n>_fema/` (42 files,
42 MB). Appendix A4.

**Offsets.** The registered GM-x corners (integer metres, EPSG:32610, NAD83 / UTM 10N) reprojected to EPSG:6339 (NAD83(2011) / UTM 10N) with
GDAL/PROJ offline move by **+1.1 µm in easting and −0.107 mm in northing at every corner of every tile** (no NAD83 → NAD83(2011) grid is
installed, as in addenda 14-16: the transformation is the identity to 0.1 mm; the true datum shift in central California is of order a metre and is
not applied anywhere in this project). Snapped to the nearest lattice line the control footprints are **the NCALM footprints exactly** (snap
residual −0.0001 m). The tiles were clipped from the control grid's full-grid routing (filled elevation, flow direction, area, class-2 count,
nearest-neighbour and filled-gap masks):

| tile | SW corner (E, N) | z range (m) | largest A inside (km²) | outlet cells | filled-gap channel cells | class-2 pts m⁻² | NN-filled | unit check 0-750 m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GM1_fema | 700991, 3980451 | 329-458 | 1.149 | 0 | 0 | 3.39 | 1.5 % | pass |
| GM2_fema | 698991, 3980451 | 280-404 | **10.339** | 0 | 0 | 4.94 | 0.4 % | pass |
| GM3_fema | 695991, 3978451 | 250-338 | 2.480 | 0 | 0 | 4.91 | 0.3 % | pass |
| GM4_fema | 694991, 3976451 | 169-307 | 6.188 | 0 | 0 | 4.06 | 0.9 % | pass |
| GM5_fema | 699991, 3981451 | 317-454 | **5.536** | 0 | 0 | 4.77 | 0.5 % | pass |
| GM6_fema | 697481, 3978961 | 244-376 | 1.486 | 0 | 0 | 4.99 | 0.3 % | pass |
| GM7_fema | 697981, 3977961 | 297-382 | 1.249 | 0 | 0 | 3.67 | 0.5 % | pass |

The FEMA tiles are NAVD88 heights (33.8 m above the ellipsoidal NCALM grid); the z ranges differ from the NCALM tiles' by that offset.
Registered as `GM-FEMA2018-ctrl` (42 manifest rows; `site_label` GM). The seven are measured in Part B on both surveys and do not enter S(m).

### 3.4 The area check (section 2.2.4, report only)
Script `a17_gm_ctrl_area_check.py` (job 52868585); record `gm_ctrl_area/gm_ctrl_area_check.json`. Appendix A5. At every entry of the addendum 16
check (the tile-ring cells where outside area enters), the control grid's area was read at the cell and as the maximum over the 3 × 3 and 11 × 11
neighbourhoods (the two surveys' channels sit a cell or two apart), against the old grid's area there:

| tile, entry | old grid (km²) | lidar-only (km²) | D = old − new | strip inflow (a16) | NED box (a16 diag.) | **FEMA full grid (3 × 3)** | FEMA vs old |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GM2 trunk (699,437 E, 3,981,451 N) | 8.452 | 4.362 | 4.090 | 3.978 (−2.7 %) | 5.272 (+29 %) | **9.760** | **+15.5 %** |
| GM5 northern trunk 1 (700,123 E, 3,982,451 N) | 1.763 | 0.209 | 1.554 | 1.534 (−1.3 %) | 2.372 (+53 %) | **2.605** | **+47.8 %** |
| GM5 northern trunk 2 (700,334 E, 3,982,451 N) | 2.667 | 0.380 | 2.287 | **0** (−100 %) | 2.217 (−3.1 %) | **2.653** | **−0.5 %** |
| GM5 eastern trunk (700,991 E, 3,981,481 N) | 1.927 | 1.682 | 0.245 | 0.224 (−8.6 %) | 0.232 (−5.5 %) | 1.858 (11 × 11: 1.940) | −3.6 % (+0.6 %) |
| GM1 north-east corner | 0.908 | 0.899 | 0.0096 | 0.0055 | 0.0087 | 0.904 | −0.5 % |
| GM2 side entry, GM4 edge cell, GM5 west edge (< 500 m²) | 0.149 / 0.579 / 0.044 | - | - | 0 | - | 0.287 / 0.099 / 0.000 | different minor channels pass these cells |

And the largest area inside each footprint (FEMA full grid against the registered grid): GM1 −0.7 %, **GM2 +16.2 % (10.34 against 8.89 km²)**,
**GM3 +637 % (2.48 against 0.34)**, GM4 +0.0 %, **GM5 +17.2 % (5.54 against 4.73)**, GM6 +0.2 %, GM7 +5.4 %.

**What this says.**
1. **GM5's second northern trunk, the entry the addendum 16 rule lost** because Perron's 60.5 km² strip truncates its basin: the FEMA lidar,
   routed over the whole catchment at 1 m, gives **2.653 km² against the old grid's 2.667 (−0.5 %)**. The NED box gave 2.22 (−3 %). Decision 2
   is confirmed a third time, by the finest and most recent survey.
2. **GM2's trunk and GM5's first northern trunk carry more area on the FEMA routing than on the old grids** (+15.5 % and +48 %), and the NED
   box, which descends from the same 2018 survey, said the same (+29 %, +53 %). The old grids follow Perron's 2008 10 m grid at these two
   entries (−2.7 %, −1.3 %). So two routings of 2018 data agree with each other against the 2008 strip on where the divide north of the NCALM
   footprint lies. This does not change decision 2 (report only), but the GM2 and GM5 modelling areas are open to the same question at a
   different entry (section 5, decision 2), and the control copies GM2_fema and GM5_fema carry the FEMA areas, so Part B measures both.
3. **GM3 receives 2.48 km² from outside on the FEMA grid where its old grid has nothing**: the old GM3 grid equals the lidar-only cut (the
   addendum 16 "no difference, pass"), because the NCALM footprint boundary runs just south-east of the tile and the legacy pipeline never
   had the basin that enters at the south-east corner (cells 999, 980-995). The NED box catchment of GM3 is 3.5 km² (2.5 outside the tile).
   GM3's registered `area_model` understates A along that trunk by about 2.1 km² (section 5, decision 2; the pair figure shows the channel).
4. GM1, GM4, GM6 and GM7 agree to within 1 % at their outlets (GM7 +5 %), and the minor entries (< 500 m²) are, as in addendum 16, different
   small channels on different surveys.

## 4. The figures (section 3 of the directive)
Script `a17_figures.py` (job 52870152, 6 min 34 s, 2 CPUs, 9.6 GB; a first full run, 52868593, drew the galleries at 30 panels a page, which
the 5 MB limit shrank to 72 %, and the NCALM outline too thin: re-run with 15 panels a page, a heavier outline and the overview limits
trimmed to the data; the earlier run's job 52865322 is deviation 3). Products in `reports/v2/figures/a17/` (PNG at 200 dpi, each under
5 MB, committed; PDF beside each, not committed) and in the run directory (`figures/`, `figures.json` with every panel's statistics).
`reports/v2/figures/a17/README.md` is the index. Appendix A7.

| figure | file | content |
| --- | --- | --- |
| GM overview | `GM_overview.png` (1.35 MB) | FEMA 2018 hillshade (extension + control grids, 5 m); footprints; NCALM 2003 footprint (purple dashed); 33 NCALM-pool and 49 FEMA-pool eligible windows; GM1-GM7, GM8-GM30, GM1_fema-GM7_fema (orange dashed, on the GM1-GM7 squares); EPSG:6339 |
| AP overview | `AP_overview.png` (1.41 MB) | 3DEP hillshade (10 m); footprint; 54 eligible windows; AP1_3dep-AP5_3dep, AP6-AP29; EPSG:6346 |
| TTV overview | `TTV_overview.png` (1.30 MB) | Marin 2019 hillshade (5 m); footprint; 34 eligible windows; the clipped M&D polygon and 77 heads; TTV1-TTV15; EPSG:6339 |
| GM gallery | `GM_gallery_p1.png` (3.75 MB: GM1-GM15), `GM_gallery_p2.png` (3.91 MB: GM16-GM30), `GM_gallery_control.png` (1.71 MB: GM1_fema-GM7_fema) | 37 panels |
| AP gallery | `AP_gallery_p1.png` (4.71 MB: AP1_3dep-AP15), `AP_gallery_p2.png` (4.34 MB: AP16-AP29) | 29 panels |
| TTV gallery | `TTV_gallery.png` (4.74 MB) | 15 panels |
| GM survey-control pairs | `GM_control_pairs.png` (3.22 MB) | GM1-GM7, NCALM 2003 beside FEMA 2018, 4 columns |
| AP survey pairs | `AP_pamap_3dep_pairs.png` (2.95 MB) | AP1-AP5, PAMAP (the v1/v2 metre tile) beside 3DEP |

**Rendering choices** (all written into `a17_common.py` / `a17_figures.py` before the first run; the three re-run changes are named above):
1. **Hillshade**: Horn's 3 × 3 gradient on the 1 m filled grid, azimuth 315°, altitude 45°, z-factor 1, everywhere (the lemcvt `hillshade`
   routine uses an inclination of 20° and was not used); quantised to 64 grey levels so the PNGs compress; nodata white.
2. **Overviews**: the hillshade is computed at 1 m in row chunks and block-averaged onto a world-aligned 5 m (GM, TTV) or 10 m (AP) lattice;
   the "lidar footprint" is the 0.5 contour of the valid-cell fraction on that lattice; at GM the two FEMA grids (extension, control) are
   mosaicked on their common lattice and the NCALM outline is the same contour of the stage B NCALM grid (its raster edge leaves the band
   open at the north-east). Eligible windows: the stage B `GM_eligibility.csv` (33, ±50 %) and the addendum 16 `GMF_eligible_windows.csv`
   (49, ±25 %) at GM, `AP_eligibility.csv` (54) at AP, the addendum 16 `TTV_eligibility.csv` (34) at TTV. Control tiles are dashed
   outlines on the GM1-GM7 squares. Scale bar, north arrow, km ticks, CRS in the axis labels and the caption. Limits trimmed to the data.
3. **Panels**: 1 × 1 km each at the same scale (the PAMAP tiles are 1,015 × 0.975 m = 990 m, drawn in their own EPSG:26917 frame); channel
   network = every cell with A ≥ 10³ m² on the tile's registered area grid, drawn as the segment to its D8 receiver, width
   0.15 + 0.45 (log₁₀ A − 3) pt; **GM1-GM5 use `area_model`** (so GM2's network carries the 8.9 km² trunk); the title's nearest-neighbour
   fraction is the `nnfill_mask` share (n/a for NCALM and PAMAP, delivered as rasters); 15 panels a page in 6 columns.
4. **What the pairs show**: the seven FEMA control tiles reproduce the NCALM tiles' networks line for line (the only visible differences are
   the trunk areas of section 3.4: GM2_fema's 10.3 against 8.9 km², GM5_fema's 5.5 against 4.7, and the 2.5 km² trunk that enters GM3_fema at
   its south-east corner, absent on GM3); the FEMA hillshades are slightly grainier (3.4-5.0 against the NCALM grid's smoother surface) and
   show the same roads and stock ponds. The PAMAP tiles show the known grid striping (AP4 most) that the 3DEP tiles do not.

## 5. Decisions needed

**1. Tile exclusions (Steve's, from the figures; decision 5).** Nothing is excluded here. What I noticed on the galleries, for the record and as
candidates only (each exclusion and its reason goes into the Part B report; excluded tiles stay registered):
- **GM10** (FEMA, 694,500 E, 3,978,500 N; A 0.30 km²): the whole tile is a dense irregular texture of benches and hummocks, unlike every
  other GM tile; it reads as cultivated or terraced ground or an old landslide complex. The strongest candidate at GM.
- **GM14** (503-693 m, NN 8.5 %), **GM28** (543-695 m, NN 7.8 %), **GM27** (NN 7.2 %), **GM21**, **GM18**: the high eastern mesa toward the range
  front; rugged, with bench-like lineaments (GM14, GM28) and the highest nearest-neighbour fill fractions of the set; GM18's network is much
  denser than the mesa tiles'. Whether this is the same landscape is the judgment the old elevation window stood in for (addendum 16 §7).
- **GM12, GM13, GM29, GM30**: a ridge road crosses the tile (the bright line along the divide); GM25 and GM19 show faint parallel striping on
  the smooth slopes (tillage or grazing terracettes); GM3 and GM3_fema have the same farm tracks as before.
- **AP17** (570,932 E, 4,421,745 N): a highway corridor with cut and embankment crosses the north-west third of the tile. Candidate.
- **AP16, AP21, AP22, AP25, AP26** (and AP18): cleared flat pads with access roads on the ridges (well pads, 50-150 m across), the typical
  Allegheny Plateau disturbance that the NLCD screen does not see at this size; AP22 and AP26 have several. **AP6, AP7, AP8, AP13, AP23, AP5_3dep**:
  a road along a valley floor or ridge.
- **TTV12** (544,500 E, 4,188,000 N): the south-east quarter is graded ground with switchback roads (the Marin Headlands batteries); **TTV13**
  (544,500 E, 4,195,500 N): the western half shows roads and lot-scale terracing (the edge of Tamalpais Valley / Marin City); **TTV14**: coastal
  switchback roads across the north; **TTV10, TTV11, TTV4, TTV3**: fire roads and the Tennessee Valley trail (TTV3 is M&D by design).
  **TTV8, TTV9** have the highest fill fractions (14 and 16 %: dense canopy); TTV5 is the Mount Tamalpais flank (445-656 m).
- No grid artefact was seen on the 3DEP, Marin or FEMA tiles; the PAMAP striping is only in the AP pairs figure (not a registered tile).

**2. GM modelling areas, three questions raised by the control grid (section 3.4; decision 2 stands as written for Part B unless changed):**
- (a) **GM2 and GM5**: keep the old grids (Perron's 2008 routing at the trunk entries, 8.89 and 4.73 km² at the outlets) or take the FEMA
  full-grid areas (10.34 and 5.54), which two 2018-based routings support against the 2008 strip. The ratio is 1.16-1.17 on the trunks and
  shifts log₁₀ A by 0.07 there; the control copies carry the FEMA areas either way.
- (b) **GM3**: its old grid has no outside inflow because the NCALM footprint cut it off; the FEMA grid shows a 2.5 km² trunk at the south-east
  corner. Keep the old grid (as v2.0.0 used) or write an `area_model` from the FEMA routing (2.48 km²).
- (c) Whether the **FEMA control grid**, which holds the whole catchment of every GM1-GM7 tile, should become the area source of the
  survey-control tiles' NCALM twins in addendum 18 (a one-line change in the tile configs) rather than a decision per tile.

**3. Part B inputs to confirm**: E per site (decision 6) as recorded; the decision-7 list GM2, GM4, AP3, AP4 and the control copies GM2_fema,
GM4_fema, GM5_fema (section 2.4); `area_model` is loaded explicitly for GM1-GM5 (section 1, deviation 8).

**Stop.** Part B starts only when the go-ahead listing the exclusions is placed in `reports/directives/`.

## 6. Appendices: cost, pre-push check, generated tables

### 6.1 Cost (decision 8) and storage
Estimates were printed before every submission (job scripts and logs in `$SCRATCH/lemcvt_jobs/a17/`, chain generator `a17_chain.sh`). Appendix
A8 has the accounting of all 10 submissions: **1.12 core-hours used** (expected ≤ 20); **sum of CPU-weighted limits 12.0 core-hours**; every
limit at most twice its estimate with the 30 min floor (the longest job, the figures, 7 min 30 s against 50 min on 2 CPUs); largest
submission 4 CPUs × 30 min (fetch, grid); peak memory 9.6 GB (figures); every job on `serc`; no array. Two of the ten were reruns (the
control cut after the 1 m shift; the figures after the paging change) and one was cancelled after a minute (the figures job that had
started on the shifted tiles). New data on group storage: the 17 LAZ boxes (0.33 GB) in the copy of record, 433 MB in the run directory
(the control grid and its routing 380 MB), 274 MB of registered tiles (`TTV-x` 98, `GM-FEMA2018-x` 134, `GM-FEMA2018-ctrl` 42), 5 MB of
`area_model_rule` grids, 33 MB of figure PNGs in the repository; about 1.1 GB in all. `$GROUP_HOME` at 320.5 GB of 1 TB. Scratch: the staged
and fetched LAZ (1.3 GB, verified against the record manifest) were removed after the grid was built and checked; the job logs stay.

### 6.2 Pre-push check (addendum 2 section 5)
Run on the staged tree before the commit and on the commit against `origin/v2.3-tiles` before pushing: forbidden tracked files (`papers/`,
`*.pdf`, `working-with-steve.md`, `cry_for_help.txt`, `.env`, `.netrc`): **none** (the figure PDFs are ignored by `.gitignore`); files over 50 MB:
**none** (the largest is `AP_gallery_p1.png`, 4.7 MB); secret patterns in the diff (`ghp_`, `github_pat_`, `gho_`, private keys, `password=`,
`token=`): **none**. No raster, point cloud or manuscript is tracked (`data/manifest.csv` and `configs/data.yaml` list the data; the data stay
on group storage).

### 6.3 Generated tables (`make_a17_tables.py`, from the run directory; also `appendix_tables.md` there)

### A1. Unit checks of the parked tiles under the decision-1 windows (section 2.1), from register/register_check.json

**TTV-x**: window 0-800 m (addendum 15/16 window 0-320 m); 15 tiles, all pass: yes; failed the old window: 4; files identical to the addendum 16 parked cut: yes.

| tile | kind | SW corner (E, N) | z min | z median | z max | pass (new window) | pass (old window) | identical to a16 | A max (km2) |
|---|---|---|---|---|---|---|---|---|---|
| TTV1 | M&D | 539000, 4190000 | 76.103 | 203.9 | 305.1 | yes | yes | yes | 0.451 |
| TTV2 | M&D | 540000, 4190500 | 41.684 | 139.9 | 241.5 | yes | yes | yes | 0.635 |
| TTV3 | M&D | 540000, 4189500 | 31.262 | 83.207 | 166.8 | yes | yes | yes | 2.707 |
| TTV4 | drawn | 543000, 4188500 | 51.670 | 150.4 | 321.1 | yes | no | yes | 0.474 |
| TTV5 | drawn | 533500, 4195500 | 444.8 | 605.4 | 656.3 | yes | no | yes | 0.830 |
| TTV6 | drawn | 541000, 4189000 | 60.668 | 200.6 | 317.3 | yes | yes | yes | 0.508 |
| TTV7 | drawn | 535000, 4193000 | 164.2 | 289.4 | 389.2 | yes | no | yes | 0.230 |
| TTV8 | drawn | 537500, 4191500 | 53.359 | 191.0 | 264.3 | yes | yes | yes | 1.036 |
| TTV9 | drawn | 533000, 4193500 | 128.9 | 266.4 | 369.1 | yes | no | yes | 2.745 |
| TTV10 | drawn | 542000, 4189500 | 100.1 | 228.8 | 317.6 | yes | yes | yes | 0.192 |
| TTV11 | drawn | 538500, 4193500 | 53.714 | 194.1 | 263.1 | yes | yes | yes | 0.188 |
| TTV12 | drawn | 544500, 4188000 | 58.186 | 166.2 | 288.5 | yes | yes | yes | 0.264 |
| TTV13 | drawn | 544500, 4195500 | 36.320 | 131.2 | 185.9 | yes | yes | yes | 0.216 |
| TTV14 | drawn | 544000, 4187000 | 26.150 | 173.6 | 287.6 | yes | yes | yes | 0.450 |
| TTV15 | drawn | 538500, 4192000 | 71.376 | 190.6 | 278.1 | yes | yes | yes | 0.475 |

**GM-FEMA2018-x**: window 0-750 m (addendum 15/16 window 68-468 m); 23 tiles, all pass: yes; failed the old window: 10; files identical to the addendum 16 parked cut: yes.

| tile | kind | SW corner (E, N) | z min | z median | z max | pass (new window) | pass (old window) | identical to a16 | A max (km2) |
|---|---|---|---|---|---|---|---|---|---|
| GM8 | drawn | 708000, 3971000 | 284.9 | 345.9 | 399.8 | yes | yes | yes | 1.544 |
| GM9 | drawn | 705500, 3973000 | 325.8 | 374.8 | 427.0 | yes | yes | yes | 1.685 |
| GM10 | drawn | 694500, 3978500 | 217.7 | 263.3 | 294.9 | yes | yes | yes | 0.298 |
| GM11 | drawn | 698000, 3972000 | 192.4 | 246.1 | 294.2 | yes | yes | yes | 1.491 |
| GM12 | drawn | 702000, 3978500 | 355.1 | 447.4 | 497.1 | yes | no | yes | 0.425 |
| GM13 | drawn | 700500, 3977000 | 322.1 | 373.4 | 430.1 | yes | yes | yes | 2.367 |
| GM14 | drawn | 712000, 3979500 | 503.1 | 572.5 | 693.1 | yes | no | yes | 0.452 |
| GM15 | drawn | 706500, 3976000 | 306.6 | 378.9 | 497.0 | yes | no | yes | 0.557 |
| GM16 | drawn | 701000, 3971500 | 207.6 | 259.5 | 325.8 | yes | yes | yes | 1.707 |
| GM17 | drawn | 700500, 3966000 | 229.2 | 287.8 | 318.6 | yes | yes | yes | 1.086 |
| GM18 | drawn | 704500, 3991500 | 540.0 | 583.7 | 670.4 | yes | no | yes | 1.342 |
| GM19 | drawn | 698500, 3993500 | 275.1 | 332.3 | 426.5 | yes | yes | yes | 1.547 |
| GM20 | drawn | 699000, 3967500 | 181.8 | 229.8 | 282.1 | yes | yes | yes | 1.275 |
| GM21 | drawn | 711000, 3976000 | 502.9 | 589.2 | 653.8 | yes | no | yes | 0.517 |
| GM22 | drawn | 695000, 3982500 | 204.9 | 262.8 | 319.9 | yes | yes | yes | 1.636 |
| GM23 | drawn | 711000, 3983500 | 421.0 | 463.6 | 547.0 | yes | no | yes | 1.037 |
| GM24 | drawn | 701000, 3975000 | 273.5 | 328.6 | 367.7 | yes | yes | yes | 2.437 |
| GM25 | drawn | 698500, 3985000 | 271.5 | 332.3 | 444.0 | yes | yes | yes | 0.890 |
| GM26 | drawn | 695500, 3979500 | 235.0 | 287.2 | 334.8 | yes | yes | yes | 1.998 |
| GM27 | drawn | 707000, 3979000 | 396.2 | 456.0 | 533.5 | yes | no | yes | 0.582 |
| GM28 | drawn | 712000, 3977500 | 542.5 | 641.3 | 695.2 | yes | no | yes | 3.259 |
| GM29 | drawn | 704000, 3978500 | 387.0 | 473.9 | 530.0 | yes | no | yes | 0.385 |
| GM30 | drawn | 710000, 3981500 | 369.1 | 408.9 | 509.3 | yes | no | yes | 1.375 |

Manifest rows against the files on disk: TTV-x: 90 rows, all ok yes; GM-FEMA2018-x: 138 rows, all ok yes; GM-x: 10 rows, all ok yes

### A2. GM1-GM5 modelling area grids (decision 2), from area_model/area_model.json

| tile | area_model = old grid: outlet A (km2) | rule grid outlet A (km2) | lidar-only outlet A (km2) | rule == old | cells where rule < old | max old - rule (km2) | area_model sha256 (16) | area_model_rule sha256 (16) |
|---|---|---|---|---|---|---|---|---|
| GM1 | 1.158 | 1.154 | 1.148 | no | 764 | 4.133e-03 | 343eb7b38872431d... | 5e4b0918a0e5b54e... |
| GM2 | 8.895 | 8.782 | 4.804 | no | 692 | 0.113 | 20c0ebd7d2c9dc05... | 820883c70c60d0d8... |
| GM3 | 0.337 | 0.337 | 0.337 | yes | 0 | 0.000 | 8196b4e43eab2bca... | 8196b4e43eab2bca... |
| GM4 | 6.186 | 6.186 | 6.186 | no | 508 | 1.600e-05 | f371cc54d9d7bd2e... | c85ddc2679a7dbd4... |
| GM5 | 4.725 | 2.956 | 2.732 | no | 1,794 | 2.306 | fd291ab366317276... | 15f75c0cf9719069... |

Manifest: 5 area_model rows updated, 5 area_model_rule rows added.

### A3. GM survey-control grid: footprint, fetch, grid, routing and checks (section 2.2.1-2.2.2)

| tile | NCALM SW corner (E, N, EPSG:32610) | catchment on the NED box (km2, tile included) | upstream outside the tile (km2) | upstream cells outside the FEMA survey | largest NED area in the tile (km2) |
|---|---|---|---|---|---|
| GM1 | 700991, 3980451 | 1.967 | 0.967 | 0 | 1.157 |
| GM2 | 698991, 3980451 | 11.252 | 10.252 | 0 | 10.327 |
| GM3 | 695991, 3978451 | 3.512 | 2.512 | 0 | 2.518 |
| GM4 | 694991, 3976451 | 9.071 | 8.071 | 0 | 6.178 |
| GM5 | 699991, 3981451 | 8.578 | 7.578 | 0 | 5.565 |
| GM6 | 697481, 3978961 | 2.039 | 1.039 | 0 | 1.406 |
| GM7 | 697981, 3977961 | 2.008 | 1.008 | 0 | 1.248 |

| union tiles + catchments (km2) | footprint with 500 m buffer, in the survey (km2) | bounding box (EPSG:32610) | 1 m lattice cells (M) | 1 km boxes | in the copy of record | fetched now | NED routing (s) |
|---|---|---|---|---|---|---|---|
| 23.919 | 43.718 | 694040, 3975850, 704880, 3986050 | 110.8 | 66 | 49 | 17 | 245.0 |

Fetch of the missing boxes (a14_ept_fetch.py): 17 / 17 boxes, 0.33 GB in 50 s (6.6 MB s-1); CRS written EPSG:6339+5703.

Grid (a14_grid_route.py --site GMC):

| lattice | cells inside footprint | cells with class 2 (inside) | class-2 points | density (pts m-2) | points per cell 5-50-95 % | NN-filled within 10 m | left nodata beyond 10 m | NN fill distance 50-95-99-max (m) | seconds | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 10211 x 10851 | 43,730,650 | 42,781,758 | 240,899,840 | 3.792 | [1, 3, 8] | 944,674 (2.2%) | 4,218 | [1.0, 1.0, 5.0, 10.0] | 68.394 | 5.056 |

Class-2 flight dates over the 66 boxes: 2018-02-07 20:37 to 2018-02-15 22:21 UTC (adjusted standard GPS time (inferred from magnitude; header flag unset)).

Re-routing with the gap rule (a15_reroute.py --site GMC):

| nodata components | enclosed (filled) | enclosed cells / km2 | enclosed >= 1e4 cells | valid after | data-edge outlets | largest basin km2 | budget closed | pad | seconds | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|
| 18 | 17 | 4,355 / 0.004 | 0 | 43,730,787 | 47,916 | 14.979 | yes | 0 | 58.903 | 5.950 |

Largest outlets: (697,526, 3,980,480) 14.98 km2 (34.3%); (694,120, 3,977,082) 9.56 km2 (21.9%); (698,388, 3,977,410) 1.59 km2 (3.6%); (698,938, 3,977,460) 1.44 km2 (3.3%); (701,826, 3,979,950) 1.25 km2 (2.9%)

Units and datum (a16_gmf_check.py --site GMC):

| comparison | cells | median dz (m) | 5-95 % | RMS | Theil-Sen slope (range) | criterion | result |
|---|---|---|---|---|---|---|---|
| rule 5.3: NED 10 m - grid block mean (ground cells) | 226,978 | 0.088 | -0.89, 0.99 | 0.554 | 0.99970 (0.99961-0.99990) | |median| < 5 m, slope 0.98-1.02 | pass |
| NCALM 2003 - grid (every 5th NCALM cell, bilinear) | 1,160,536 | -33.841 | -34.11, -33.61 | 33.852 | 1.0012 (1.0011-1.0012) | |offset| 30-34 m (NCALM ellipsoidal), slope 0.98-1.02; stop near 3.28 / 0.305 | offset in window, slope pass; no stop |

### A4. The control tiles GM1_fema-GM7_fema (section 2.2.3), from draw_gmc/GMC_footprints.json and tiles_gmc/tiles.json

Transformation: EPSG:32610 -> EPSG:6339 by GDAL/PROJ offline; control grid bounds (EPSG:6339): 694,035, 3,975,844, 704,886, 3,986,055.

| tile | NCALM SW corner (EPSG:32610) | reprojected SW corner (EPSG:6339) | shift at SW corner (dx, dy m) | shift at NE corner (dx, dy m) | snapped SW corner | snap residual (m) |
|---|---|---|---|---|---|---|
| GM1 | 700991.000, 3980451.000 | 700991.000, 3980451.000 | 0.0000, -0.0001 | 0.0000, -0.0001 | 700991, 3980451 | 0.0000, -0.0001 |
| GM2 | 698991.000, 3980451.000 | 698991.000, 3980451.000 | 0.0000, -0.0001 | 0.0000, -0.0001 | 698991, 3980451 | 0.0000, -0.0001 |
| GM3 | 695991.000, 3978451.000 | 695991.000, 3978451.000 | 0.0000, -0.0001 | 0.0000, -0.0001 | 695991, 3978451 | 0.0000, -0.0001 |
| GM4 | 694991.000, 3976451.000 | 694991.000, 3976451.000 | 0.0000, -0.0001 | 0.0000, -0.0001 | 694991, 3976451 | 0.0000, -0.0001 |
| GM5 | 699991.000, 3981451.000 | 699991.000, 3981451.000 | 0.0000, -0.0001 | 0.0000, -0.0001 | 699991, 3981451 | 0.0000, -0.0001 |
| GM6 | 697481.000, 3978961.000 | 697481.000, 3978961.000 | 0.0000, -0.0001 | 0.0000, -0.0001 | 697481, 3978961 | 0.0000, -0.0001 |
| GM7 | 697981.000, 3977961.000 | 697981.000, 3977961.000 | 0.0000, -0.0001 | 0.0000, -0.0001 | 697981, 3977961 | 0.0000, -0.0001 |

| tile | SW corner (E, N) | z median | z min | z max | unit check (0-750 m) | A max (km2) | outlet cells | filled-gap channel cells | class-2 pts/m2 | NN-filled frac |
|---|---|---|---|---|---|---|---|---|---|---|
| GM1_fema | 700991, 3980451 | 391.4 | 329.1 | 458.5 | yes | 1.149 | 0 | 0 | 3.392 | 0.015 |
| GM2_fema | 698991, 3980451 | 342.5 | 280.2 | 404.4 | yes | 10.339 | 0 | 0 | 4.938 | 3.694e-03 |
| GM3_fema | 695991, 3978451 | 307.2 | 250.4 | 338.3 | yes | 2.480 | 0 | 0 | 4.911 | 3.062e-03 |
| GM4_fema | 694991, 3976451 | 245.8 | 168.9 | 307.3 | yes | 6.188 | 0 | 0 | 4.055 | 9.210e-03 |
| GM5_fema | 699991, 3981451 | 373.9 | 316.5 | 453.7 | yes | 5.536 | 0 | 0 | 4.772 | 4.813e-03 |
| GM6_fema | 697481, 3978961 | 334.4 | 243.9 | 375.9 | yes | 1.486 | 0 | 0 | 4.993 | 2.682e-03 |
| GM7_fema | 697981, 3977961 | 347.1 | 297.0 | 382.0 | yes | 1.249 | 0 | 0 | 3.674 | 5.223e-03 |

Checksums (sha256) of the control tiles' products:

| tile | role | path | sha256 (first 16) | bytes |
|---|---|---|---|---|
| GM1_fema | filled | GM-FEMA2018-ctrl/GM1_fema/GM1_fema_1m_filled.tif | 0f6863c8bb7eb7f1... | 2,616,976 |
| GM1_fema | fd | GM-FEMA2018-ctrl/GM1_fema/GM1_fema_1m_fd.tif | 1dbe84d3f1de5088... | 211,248 |
| GM1_fema | area | GM-FEMA2018-ctrl/GM1_fema/GM1_fema_1m_area.tif | baaa86ff7a7bb822... | 1,132,195 |
| GM1_fema | gapfill_mask | GM-FEMA2018-ctrl/GM1_fema/GM1_fema_1m_gapfill_mask.tif | 0f4053d56415a92b... | 1,866 |
| GM1_fema | class2_count | GM-FEMA2018-ctrl/GM1_fema/GM1_fema_1m_class2_count.tif | ae1390edf27f15e6... | 508,996 |
| GM1_fema | nnfill_mask | GM-FEMA2018-ctrl/GM1_fema/GM1_fema_1m_nnfill_mask.tif | 61f2cb9e2368bef4... | 25,700 |
| GM2_fema | filled | GM-FEMA2018-ctrl/GM2_fema/GM2_fema_1m_filled.tif | c1a153d829f42775... | 2,836,035 |
| GM2_fema | fd | GM-FEMA2018-ctrl/GM2_fema/GM2_fema_1m_fd.tif | 7c5e51cbd6d8ef54... | 192,063 |
| GM2_fema | area | GM-FEMA2018-ctrl/GM2_fema/GM2_fema_1m_area.tif | eb4c4453bf90cb6c... | 1,223,767 |
| GM2_fema | gapfill_mask | GM-FEMA2018-ctrl/GM2_fema/GM2_fema_1m_gapfill_mask.tif | 2cade4a23bf22649... | 1,866 |
| GM2_fema | class2_count | GM-FEMA2018-ctrl/GM2_fema/GM2_fema_1m_class2_count.tif | c6a7fa2f7475db47... | 573,115 |
| GM2_fema | nnfill_mask | GM-FEMA2018-ctrl/GM2_fema/GM2_fema_1m_nnfill_mask.tif | 58feaf088a94982d... | 9,428 |
| GM3_fema | filled | GM-FEMA2018-ctrl/GM3_fema/GM3_fema_1m_filled.tif | 8242ccae21b8717a... | 2,696,816 |
| GM3_fema | fd | GM-FEMA2018-ctrl/GM3_fema/GM3_fema_1m_fd.tif | d05c17a3f29a3877... | 204,247 |
| GM3_fema | area | GM-FEMA2018-ctrl/GM3_fema/GM3_fema_1m_area.tif | b1f8cc664b0862c9... | 1,208,960 |
| GM3_fema | gapfill_mask | GM-FEMA2018-ctrl/GM3_fema/GM3_fema_1m_gapfill_mask.tif | ae7d9bb60a2960b2... | 1,866 |
| GM3_fema | class2_count | GM-FEMA2018-ctrl/GM3_fema/GM3_fema_1m_class2_count.tif | 0efb1d16caf798e3... | 553,214 |
| GM3_fema | nnfill_mask | GM-FEMA2018-ctrl/GM3_fema/GM3_fema_1m_nnfill_mask.tif | d7c68f9c3a022151... | 8,233 |
| GM4_fema | filled | GM-FEMA2018-ctrl/GM4_fema/GM4_fema_1m_filled.tif | da15f0a51b301800... | 2,758,560 |
| GM4_fema | fd | GM-FEMA2018-ctrl/GM4_fema/GM4_fema_1m_fd.tif | e423f8b2b4ef3e71... | 204,083 |
| GM4_fema | area | GM-FEMA2018-ctrl/GM4_fema/GM4_fema_1m_area.tif | 53d1a6006e7ee292... | 1,226,593 |
| GM4_fema | gapfill_mask | GM-FEMA2018-ctrl/GM4_fema/GM4_fema_1m_gapfill_mask.tif | 7f8c520ad229d3a8... | 1,866 |
| GM4_fema | class2_count | GM-FEMA2018-ctrl/GM4_fema/GM4_fema_1m_class2_count.tif | cc780ac624c7814b... | 540,176 |
| GM4_fema | nnfill_mask | GM-FEMA2018-ctrl/GM4_fema/GM4_fema_1m_nnfill_mask.tif | 592910f3c011af55... | 18,238 |
| GM5_fema | filled | GM-FEMA2018-ctrl/GM5_fema/GM5_fema_1m_filled.tif | 2ede1f73fd73e058... | 2,797,511 |
| GM5_fema | fd | GM-FEMA2018-ctrl/GM5_fema/GM5_fema_1m_fd.tif | 829d7cbbe1455a77... | 197,493 |
| GM5_fema | area | GM-FEMA2018-ctrl/GM5_fema/GM5_fema_1m_area.tif | a623da22933fe6ce... | 1,250,995 |
| GM5_fema | gapfill_mask | GM-FEMA2018-ctrl/GM5_fema/GM5_fema_1m_gapfill_mask.tif | bc4a0111e04186bb... | 1,866 |
| GM5_fema | class2_count | GM-FEMA2018-ctrl/GM5_fema/GM5_fema_1m_class2_count.tif | f3ff87f9a589d364... | 565,749 |
| GM5_fema | nnfill_mask | GM-FEMA2018-ctrl/GM5_fema/GM5_fema_1m_nnfill_mask.tif | 2ba1112f3e903075... | 11,398 |
| GM6_fema | filled | GM-FEMA2018-ctrl/GM6_fema/GM6_fema_1m_filled.tif | 58a14ecdff487985... | 2,790,040 |
| GM6_fema | fd | GM-FEMA2018-ctrl/GM6_fema/GM6_fema_1m_fd.tif | 55ac00f94bb41b6f... | 188,899 |
| GM6_fema | area | GM-FEMA2018-ctrl/GM6_fema/GM6_fema_1m_area.tif | 1d259ef33ac56105... | 1,242,982 |
| GM6_fema | gapfill_mask | GM-FEMA2018-ctrl/GM6_fema/GM6_fema_1m_gapfill_mask.tif | 53b5c149dbdf4f61... | 1,866 |
| GM6_fema | class2_count | GM-FEMA2018-ctrl/GM6_fema/GM6_fema_1m_class2_count.tif | ff66f35d93b8a0eb... | 564,705 |
| GM6_fema | nnfill_mask | GM-FEMA2018-ctrl/GM6_fema/GM6_fema_1m_nnfill_mask.tif | bab0c8cde795c64a... | 7,955 |
| GM7_fema | filled | GM-FEMA2018-ctrl/GM7_fema/GM7_fema_1m_filled.tif | 8840ff7646ff246e... | 2,549,647 |
| GM7_fema | fd | GM-FEMA2018-ctrl/GM7_fema/GM7_fema_1m_fd.tif | 2d396821abe48528... | 196,042 |
| GM7_fema | area | GM-FEMA2018-ctrl/GM7_fema/GM7_fema_1m_area.tif | 83a7c2599a8af7eb... | 1,244,988 |
| GM7_fema | gapfill_mask | GM-FEMA2018-ctrl/GM7_fema/GM7_fema_1m_gapfill_mask.tif | 77f515f2c9ecc764... | 1,866 |
| GM7_fema | class2_count | GM-FEMA2018-ctrl/GM7_fema/GM7_fema_1m_class2_count.tif | e4221296fe79288d... | 499,198 |
| GM7_fema | nnfill_mask | GM-FEMA2018-ctrl/GM7_fema/GM7_fema_1m_nnfill_mask.tif | b2aec336f658bdc0... | 13,052 |

### A5. The FEMA full-grid area against the old grids at the addendum 16 entries (section 2.2.4, report only), from gm_ctrl_area/gm_ctrl_area_check.json

| tile | entry (E, N) | old (km2) | lidar-only (km2) | D = old - new | strip inflow (a16) | FEMA at cell | FEMA max 3x3 | FEMA max 11x11 | 11x11 argmax offset (dr, dc) | FEMA 3x3 vs old | FEMA 11x11 vs old | kind |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM1 | 701,990.5, 3,981,149.5 | 0.908 | 0.899 | 9.633e-03 | 5.500e-03 | 0.904 | 0.904 | 0.905 | [5, -4] | -0.5% | -0.4% | minor |
| GM2 | 699,264.5, 3,981,450.5 | 0.149 | 0.148 | 4.510e-04 | 0.000 | 0.287 | 0.287 | 0.288 | [5, 1] | +92.9% | +93.6% | minor |
| GM2 | 699,436.5, 3,981,450.5 | 8.452 | 4.362 | 4.090 | 3.978 | 8.000e-06 | 9.760 | 9.760 | [1, 5] | +15.5% | +15.5% | trunk |
| GM4 | 695,065.5, 3,977,450.5 | 0.579 | 0.579 | 1.600e-05 | 0.000 | 1.000e-06 | 0.099 | 0.100 | [5, 2] | -82.9% | -82.8% | minor |
| GM5 | 700,122.5, 3,982,450.5 | 1.763 | 0.209 | 1.554 | 1.534 | 1.600e-05 | 2.605 | 2.605 | [5, -1] | +47.8% | +47.8% | trunk |
| GM5 | 700,333.5, 3,982,450.5 | 2.667 | 0.380 | 2.287 | 0.000 | 1.000e-06 | 2.653 | 2.654 | [1, -5] | -0.5% | -0.5% | trunk |
| GM5 | 699,991.5, 3,982,353.5 | 0.044 | 0.044 | 2.000e-05 | 0.000 | 1.000e-06 | 2.000e-05 | 0.047 | [5, 1] | -100.0% | +6.5% | minor |
| GM5 | 700,990.5, 3,981,480.5 | 1.927 | 1.682 | 0.245 | 0.224 | 1.858 | 1.858 | 1.940 | [-1, -5] | -3.6% | +0.6% | trunk |

Largest area inside each footprint, FEMA full grid against the registered grid:

| tile | registered grid | registered A max (km2) | argmax (r, c) | FEMA A max (km2) | argmax (r, c) | rel. diff | cells >= 0.1 km2 registered / FEMA |
|---|---|---|---|---|---|---|---|
| GM1 | area_model | 1.158 | [999, 696] | 1.149 | [999, 697] | -0.7% | 1,792 / 1,824 |
| GM2 | area_model | 8.895 | [349, 0] | 10.339 | [348, 0] | +16.2% | 2,061 / 2,058 |
| GM3 | area_model | 0.337 | [999, 980] | 2.480 | [999, 995] | +636.6% | 903 / 822 |
| GM4 | area_model | 6.186 | [452, 0] | 6.188 | [451, 0] | +0.0% | 1,784 / 2,076 |
| GM5 | area_model | 4.725 | [331, 0] | 5.536 | [330, 0] | +17.2% | 2,937 / 2,982 |
| GM6 | area | 1.483 | [999, 409] | 1.486 | [999, 410] | +0.2% | 1,119 / 1,046 |
| GM7 | area | 1.185 | [999, 758] | 1.249 | [999, 758] | +5.4% | 1,180 / 1,189 |

### A6. The registered tile list per site (section 2.1), from configs/data.yaml and the cut records

| site | tile | version | survey | footprint SW corner (E, N) | area source for modelling | kind | largest A inside (km2, on the area source) |
|---|---|---|---|---|---|---|---|
| GM | GM1 | GM-x | NCALM 2003 | 700991, 3980451 (EPSG:32610) | area_model (old tile grid, decision 2) | original | 1.158 |
| GM | GM2 | GM-x | NCALM 2003 | 698991, 3980451 (EPSG:32610) | area_model (old tile grid, decision 2) | original | 8.895 |
| GM | GM3 | GM-x | NCALM 2003 | 695991, 3978451 (EPSG:32610) | area_model (old tile grid, decision 2) | original | 0.337 |
| GM | GM4 | GM-x | NCALM 2003 | 694991, 3976451 (EPSG:32610) | area_model (old tile grid, decision 2) | original | 6.186 |
| GM | GM5 | GM-x | NCALM 2003 | 699991, 3981451 (EPSG:32610) | area_model (old tile grid, decision 2) | original | 4.725 |
| GM | GM6 | GM-x | NCALM 2003 | 697481, 3978961 (EPSG:32610) | full-grid routing (role area) | drawn | 1.483 |
| GM | GM7 | GM-x | NCALM 2003 | 697981, 3977961 (EPSG:32610) | full-grid routing (role area) | drawn | 1.185 |
| AP | AP1_3dep | AP-3DEP-x | 3DEP 2019-20 | 568503, 4408340 (EPSG:6346) | full-grid routing (role area) | original | 1.175 |
| AP | AP2_3dep | AP-3DEP-x | 3DEP 2019-20 | 561573, 4426160 (EPSG:6346) | full-grid routing (role area) | original | 1.685 |
| AP | AP3_3dep | AP-3DEP-x | 3DEP 2019-20 | 551854, 4419230 (EPSG:6346) | full-grid routing (role area) | original | 11.072 |
| AP | AP4_3dep | AP-3DEP-x | 3DEP 2019-20 | 555093, 4419230 (EPSG:6346) | full-grid routing (role area) | original | 17.143 |
| AP | AP5_3dep | AP-3DEP-x | 3DEP 2019-20 | 556893, 4422470 (EPSG:6346) | full-grid routing (role area) | original | 1.167 |
| AP | AP6 | AP-3DEP-x | 3DEP 2019-20 | 566493, 4407378 (EPSG:6346) | full-grid routing (role area) | drawn | 1.493 |
| AP | AP7 | AP-3DEP-x | 3DEP 2019-20 | 556083, 4420246 (EPSG:6346) | full-grid routing (role area) | drawn | 3.647 |
| AP | AP8 | AP-3DEP-x | 3DEP 2019-20 | 552633, 4420755 (EPSG:6346) | full-grid routing (role area) | drawn | 1.625 |
| AP | AP9 | AP-3DEP-x | 3DEP 2019-20 | 570452, 4418776 (EPSG:6346) | full-grid routing (role area) | drawn | 1.518 |
| AP | AP10 | AP-3DEP-x | 3DEP 2019-20 | 558573, 4407378 (EPSG:6346) | full-grid routing (role area) | drawn | 0.425 |
| AP | AP11 | AP-3DEP-x | 3DEP 2019-20 | 563523, 4412837 (EPSG:6346) | full-grid routing (role area) | drawn | 1.958 |
| AP | AP12 | AP-3DEP-x | 3DEP 2019-20 | 559053, 4413317 (EPSG:6346) | full-grid routing (role area) | drawn | 1.248 |
| AP | AP13 | AP-3DEP-x | 3DEP 2019-20 | 555603, 4410347 (EPSG:6346) | full-grid routing (role area) | drawn | 3.967 |
| AP | AP14 | AP-3DEP-x | 3DEP 2019-20 | 565983, 4420755 (EPSG:6346) | full-grid routing (role area) | drawn | 0.555 |
| AP | AP15 | AP-3DEP-x | 3DEP 2019-20 | 568472, 4420755 (EPSG:6346) | full-grid routing (role area) | drawn | 2.075 |
| AP | AP16 | AP-3DEP-x | 3DEP 2019-20 | 562023, 4418776 (EPSG:6346) | full-grid routing (role area) | drawn | 1.692 |
| AP | AP17 | AP-3DEP-x | 3DEP 2019-20 | 570932, 4421745 (EPSG:6346) | full-grid routing (role area) | drawn | 2.321 |
| AP | AP18 | AP-3DEP-x | 3DEP 2019-20 | 574892, 4425705 (EPSG:6346) | full-grid routing (role area) | drawn | 1.982 |
| AP | AP19 | AP-3DEP-x | 3DEP 2019-20 | 564003, 4424205 (EPSG:6346) | full-grid routing (role area) | drawn | 2.109 |
| AP | AP20 | AP-3DEP-x | 3DEP 2019-20 | 565503, 4424715 (EPSG:6346) | full-grid routing (role area) | drawn | 0.338 |
| AP | AP21 | AP-3DEP-x | 3DEP 2019-20 | 558063, 4416286 (EPSG:6346) | full-grid routing (role area) | drawn | 0.421 |
| AP | AP22 | AP-3DEP-x | 3DEP 2019-20 | 555093, 4414307 (EPSG:6346) | full-grid routing (role area) | drawn | 0.395 |
| AP | AP23 | AP-3DEP-x | 3DEP 2019-20 | 555093, 4408368 (EPSG:6346) | full-grid routing (role area) | drawn | 1.724 |
| AP | AP24 | AP-3DEP-x | 3DEP 2019-20 | 566493, 4423215 (EPSG:6346) | full-grid routing (role area) | drawn | 1.600 |
| AP | AP25 | AP-3DEP-x | 3DEP 2019-20 | 560553, 4416286 (EPSG:6346) | full-grid routing (role area) | drawn | 3.266 |
| AP | AP26 | AP-3DEP-x | 3DEP 2019-20 | 560043, 4425195 (EPSG:6346) | full-grid routing (role area) | drawn | 0.536 |
| AP | AP27 | AP-3DEP-x | 3DEP 2019-20 | 555603, 4412837 (EPSG:6346) | full-grid routing (role area) | drawn | 1.765 |
| AP | AP28 | AP-3DEP-x | 3DEP 2019-20 | 576872, 4416796 (EPSG:6346) | full-grid routing (role area) | drawn | 0.750 |
| AP | AP29 | AP-3DEP-x | 3DEP 2019-20 | 562533, 4417276 (EPSG:6346) | full-grid routing (role area) | drawn | 1.781 |
| TTV | TTV1 | TTV-x | Marin 2019 | 539000, 4190000 (EPSG:6339) | full-grid routing (role area) | M&D by design | 0.451 |
| TTV | TTV2 | TTV-x | Marin 2019 | 540000, 4190500 (EPSG:6339) | full-grid routing (role area) | M&D by design | 0.635 |
| TTV | TTV3 | TTV-x | Marin 2019 | 540000, 4189500 (EPSG:6339) | full-grid routing (role area) | M&D by design | 2.707 |
| TTV | TTV4 | TTV-x | Marin 2019 | 543000, 4188500 (EPSG:6339) | full-grid routing (role area) | drawn | 0.474 |
| TTV | TTV5 | TTV-x | Marin 2019 | 533500, 4195500 (EPSG:6339) | full-grid routing (role area) | drawn | 0.830 |
| TTV | TTV6 | TTV-x | Marin 2019 | 541000, 4189000 (EPSG:6339) | full-grid routing (role area) | drawn | 0.508 |
| TTV | TTV7 | TTV-x | Marin 2019 | 535000, 4193000 (EPSG:6339) | full-grid routing (role area) | drawn | 0.230 |
| TTV | TTV8 | TTV-x | Marin 2019 | 537500, 4191500 (EPSG:6339) | full-grid routing (role area) | drawn | 1.036 |
| TTV | TTV9 | TTV-x | Marin 2019 | 533000, 4193500 (EPSG:6339) | full-grid routing (role area) | drawn | 2.745 |
| TTV | TTV10 | TTV-x | Marin 2019 | 542000, 4189500 (EPSG:6339) | full-grid routing (role area) | drawn | 0.192 |
| TTV | TTV11 | TTV-x | Marin 2019 | 538500, 4193500 (EPSG:6339) | full-grid routing (role area) | drawn | 0.188 |
| TTV | TTV12 | TTV-x | Marin 2019 | 544500, 4188000 (EPSG:6339) | full-grid routing (role area) | drawn | 0.264 |
| TTV | TTV13 | TTV-x | Marin 2019 | 544500, 4195500 (EPSG:6339) | full-grid routing (role area) | drawn | 0.216 |
| TTV | TTV14 | TTV-x | Marin 2019 | 544000, 4187000 (EPSG:6339) | full-grid routing (role area) | drawn | 0.450 |
| TTV | TTV15 | TTV-x | Marin 2019 | 538500, 4192000 (EPSG:6339) | full-grid routing (role area) | drawn | 0.475 |
| GM | GM8 | GM-FEMA2018-x | FEMA 2018 | 708000, 3971000 (EPSG:6339) | full-grid routing (role area) | drawn | 1.544 |
| GM | GM9 | GM-FEMA2018-x | FEMA 2018 | 705500, 3973000 (EPSG:6339) | full-grid routing (role area) | drawn | 1.685 |
| GM | GM10 | GM-FEMA2018-x | FEMA 2018 | 694500, 3978500 (EPSG:6339) | full-grid routing (role area) | drawn | 0.298 |
| GM | GM11 | GM-FEMA2018-x | FEMA 2018 | 698000, 3972000 (EPSG:6339) | full-grid routing (role area) | drawn | 1.491 |
| GM | GM12 | GM-FEMA2018-x | FEMA 2018 | 702000, 3978500 (EPSG:6339) | full-grid routing (role area) | drawn | 0.425 |
| GM | GM13 | GM-FEMA2018-x | FEMA 2018 | 700500, 3977000 (EPSG:6339) | full-grid routing (role area) | drawn | 2.367 |
| GM | GM14 | GM-FEMA2018-x | FEMA 2018 | 712000, 3979500 (EPSG:6339) | full-grid routing (role area) | drawn | 0.452 |
| GM | GM15 | GM-FEMA2018-x | FEMA 2018 | 706500, 3976000 (EPSG:6339) | full-grid routing (role area) | drawn | 0.557 |
| GM | GM16 | GM-FEMA2018-x | FEMA 2018 | 701000, 3971500 (EPSG:6339) | full-grid routing (role area) | drawn | 1.707 |
| GM | GM17 | GM-FEMA2018-x | FEMA 2018 | 700500, 3966000 (EPSG:6339) | full-grid routing (role area) | drawn | 1.086 |
| GM | GM18 | GM-FEMA2018-x | FEMA 2018 | 704500, 3991500 (EPSG:6339) | full-grid routing (role area) | drawn | 1.342 |
| GM | GM19 | GM-FEMA2018-x | FEMA 2018 | 698500, 3993500 (EPSG:6339) | full-grid routing (role area) | drawn | 1.547 |
| GM | GM20 | GM-FEMA2018-x | FEMA 2018 | 699000, 3967500 (EPSG:6339) | full-grid routing (role area) | drawn | 1.275 |
| GM | GM21 | GM-FEMA2018-x | FEMA 2018 | 711000, 3976000 (EPSG:6339) | full-grid routing (role area) | drawn | 0.517 |
| GM | GM22 | GM-FEMA2018-x | FEMA 2018 | 695000, 3982500 (EPSG:6339) | full-grid routing (role area) | drawn | 1.636 |
| GM | GM23 | GM-FEMA2018-x | FEMA 2018 | 711000, 3983500 (EPSG:6339) | full-grid routing (role area) | drawn | 1.037 |
| GM | GM24 | GM-FEMA2018-x | FEMA 2018 | 701000, 3975000 (EPSG:6339) | full-grid routing (role area) | drawn | 2.437 |
| GM | GM25 | GM-FEMA2018-x | FEMA 2018 | 698500, 3985000 (EPSG:6339) | full-grid routing (role area) | drawn | 0.890 |
| GM | GM26 | GM-FEMA2018-x | FEMA 2018 | 695500, 3979500 (EPSG:6339) | full-grid routing (role area) | drawn | 1.998 |
| GM | GM27 | GM-FEMA2018-x | FEMA 2018 | 707000, 3979000 (EPSG:6339) | full-grid routing (role area) | drawn | 0.582 |
| GM | GM28 | GM-FEMA2018-x | FEMA 2018 | 712000, 3977500 (EPSG:6339) | full-grid routing (role area) | drawn | 3.259 |
| GM | GM29 | GM-FEMA2018-x | FEMA 2018 | 704000, 3978500 (EPSG:6339) | full-grid routing (role area) | drawn | 0.385 |
| GM | GM30 | GM-FEMA2018-x | FEMA 2018 | 710000, 3981500 (EPSG:6339) | full-grid routing (role area) | drawn | 1.375 |
| GM | GM1_fema | GM-FEMA2018-ctrl | FEMA 2018 | 700991, 3980451 (EPSG:6339) | full-grid routing (role area) | control | 1.149 |
| GM | GM2_fema | GM-FEMA2018-ctrl | FEMA 2018 | 698991, 3980451 (EPSG:6339) | full-grid routing (role area) | control | 10.339 |
| GM | GM3_fema | GM-FEMA2018-ctrl | FEMA 2018 | 695991, 3978451 (EPSG:6339) | full-grid routing (role area) | control | 2.480 |
| GM | GM4_fema | GM-FEMA2018-ctrl | FEMA 2018 | 694991, 3976451 (EPSG:6339) | full-grid routing (role area) | control | 6.188 |
| GM | GM5_fema | GM-FEMA2018-ctrl | FEMA 2018 | 699991, 3981451 (EPSG:6339) | full-grid routing (role area) | control | 5.536 |
| GM | GM6_fema | GM-FEMA2018-ctrl | FEMA 2018 | 697481, 3978961 (EPSG:6339) | full-grid routing (role area) | control | 1.486 |
| GM | GM7_fema | GM-FEMA2018-ctrl | FEMA 2018 | 697981, 3977961 (EPSG:6339) | full-grid routing (role area) | control | 1.249 |

Registered per site: GM 37, AP 29, TTV 15; in all 81 (the 7 control tiles included in GM).

Decision 7 (for addendum 18): tiles whose largest drainage area inside exceeds 5 km2 on the registered grid: GM2 (GM-x, 8.89 km2), GM4 (GM-x, 6.19 km2), AP3_3dep (AP-3DEP-x, 11.07 km2), AP4_3dep (AP-3DEP-x, 17.14 km2), GM2_fema (GM-FEMA2018-ctrl, 10.34 km2), GM4_fema (GM-FEMA2018-ctrl, 6.19 km2), GM5_fema (GM-FEMA2018-ctrl, 5.54 km2).

Decision 6: E carried over to every tile of the site: GM 1.47e-04 m/yr (Perron et al. (2012), applied to every GM tile (NCALM 2003 and FEMA 2018)); AP 5.50e-05 m/yr (5.5e-5 m/yr (the v1/v2 value)); TTV 7.70e-05 m/yr (77 +- 15 m/Myr, Heimsath et al. (1997)).

### A7. The figures (section 3), from figures.json

| figure | PNG bytes | PDF bytes | size (in) | scale applied | dpi | notes |
|---|---|---|---|---|---|---|
| GM_overview | 1,350,889 | 937,966 | 11.0 x 16.24 | 1.000 | 200 | block 5 m; eligible windows {'NCALM 2003 pool (stage B, ±50 %)': 33, 'FEMA 2018 pool (addendum 16 gate, ±25 %)': 49}; tiles 37 |
| AP_overview | 1,410,992 | 892,807 | 11.0 x 9.06 | 1.000 | 200 | block 10 m; eligible windows {'3DEP pool (stage B, ±25 %)': 54}; tiles 29 |
| TTV_overview | 1,303,825 | 866,533 | 11.0 x 8.66 | 1.000 | 200 | block 5 m; eligible windows {'Marin pool (addendum 16, ±25 %)': 34}; tiles 15 |
| GM_gallery_p1 | 3,750,167 | 4,596,090 | 15.3 x 8.61 | 1.000 | 200 |  |
| GM_gallery_p2 | 3,907,572 | 4,651,730 | 15.3 x 8.61 | 1.000 | 200 |  |
| GM_gallery_control | 1,713,848 | 2,031,523 | 15.3 x 5.74 | 1.000 | 200 |  |
| AP_gallery_p1 | 4,710,072 | 6,260,357 | 15.3 x 8.61 | 1.000 | 200 |  |
| AP_gallery_p2 | 4,338,557 | 5,716,713 | 15.3 x 8.61 | 1.000 | 200 |  |
| TTV_gallery | 4,736,353 | 5,684,819 | 15.3 x 8.61 | 1.000 | 200 |  |
| GM_control_pairs | 3,215,741 | 4,121,159 | 10.2 x 11.48 | 1.000 | 200 |  |
| AP_pamap_3dep_pairs | 2,953,404 | 4,419,650 | 10.2 x 8.61 | 1.000 | 200 |  |

Panel statistics (title values):

| panel | kind | survey | area grid | z min | z max | largest A inside (km2) | NN-filled frac | network cells (A >= 1e3) |
|---|---|---|---|---|---|---|---|---|
| GM1 | original | NCALM 2003 | area_model | 295.2 | 424.7 | 1.158 | - | 8,723 |
| GM2 | original | NCALM 2003 | area_model | 246.3 | 370.7 | 8.895 | - | 11,314 |
| GM3 | original | NCALM 2003 | area_model | 216.3 | 304.5 | 0.337 | - | 12,401 |
| GM4 | original | NCALM 2003 | area_model | 134.9 | 273.4 | 6.186 | - | 11,027 |
| GM5 | original | NCALM 2003 | area_model | 282.4 | 419.9 | 4.725 | - | 14,562 |
| GM6 | drawn | NCALM 2003 | area | 210.1 | 342.1 | 1.483 | - | 13,040 |
| GM7 | drawn | NCALM 2003 | area | 263.1 | 348.1 | 1.185 | - | 10,910 |
| GM8 | drawn | FEMA 2018 | area | 284.9 | 399.8 | 1.544 | 0.017 | 11,793 |
| GM9 | drawn | FEMA 2018 | area | 325.8 | 427.0 | 1.685 | 0.014 | 14,499 |
| GM10 | drawn | FEMA 2018 | area | 217.7 | 294.9 | 0.298 | 0.015 | 20,362 |
| GM11 | drawn | FEMA 2018 | area | 192.4 | 294.2 | 1.491 | 9.282e-03 | 13,587 |
| GM12 | drawn | FEMA 2018 | area | 355.1 | 497.1 | 0.425 | 0.015 | 12,094 |
| GM13 | drawn | FEMA 2018 | area | 322.1 | 430.1 | 2.367 | 0.011 | 12,828 |
| GM14 | drawn | FEMA 2018 | area | 503.1 | 693.1 | 0.452 | 0.085 | 20,069 |
| GM15 | drawn | FEMA 2018 | area | 306.6 | 497.0 | 0.557 | 0.029 | 18,804 |
| GM16 | drawn | FEMA 2018 | area | 207.6 | 325.8 | 1.707 | 8.413e-03 | 10,650 |
| GM17 | drawn | FEMA 2018 | area | 229.2 | 318.6 | 1.086 | 8.788e-03 | 10,740 |
| GM18 | drawn | FEMA 2018 | area | 540.0 | 670.4 | 1.342 | 0.032 | 22,236 |
| GM19 | drawn | FEMA 2018 | area | 275.1 | 426.5 | 1.547 | 0.012 | 12,453 |
| GM20 | drawn | FEMA 2018 | area | 181.8 | 282.1 | 1.275 | 9.201e-03 | 10,874 |
| GM21 | drawn | FEMA 2018 | area | 502.9 | 653.8 | 0.517 | 0.043 | 13,630 |
| GM22 | drawn | FEMA 2018 | area | 204.9 | 319.9 | 1.636 | 4.321e-03 | 12,126 |
| GM23 | drawn | FEMA 2018 | area | 421.0 | 547.0 | 1.037 | 0.021 | 17,984 |
| GM24 | drawn | FEMA 2018 | area | 273.5 | 367.7 | 2.437 | 8.828e-03 | 11,899 |
| GM25 | drawn | FEMA 2018 | area | 271.5 | 444.0 | 0.890 | 0.034 | 12,374 |
| GM26 | drawn | FEMA 2018 | area | 235.0 | 334.8 | 1.998 | 8.801e-03 | 15,231 |
| GM27 | drawn | FEMA 2018 | area | 396.2 | 533.5 | 0.582 | 0.072 | 14,347 |
| GM28 | drawn | FEMA 2018 | area | 542.5 | 695.2 | 3.259 | 0.079 | 18,671 |
| GM29 | drawn | FEMA 2018 | area | 387.0 | 530.0 | 0.385 | 0.028 | 12,062 |
| GM30 | drawn | FEMA 2018 | area | 369.1 | 509.3 | 1.375 | 0.021 | 15,967 |
| GM1_fema | control | FEMA 2018 | area | 329.1 | 458.5 | 1.149 | 0.015 | 9,442 |
| GM2_fema | control | FEMA 2018 | area | 280.2 | 404.4 | 10.339 | 3.694e-03 | 12,361 |
| GM3_fema | control | FEMA 2018 | area | 250.4 | 338.3 | 2.480 | 3.062e-03 | 12,573 |
| GM4_fema | control | FEMA 2018 | area | 168.9 | 307.3 | 6.188 | 9.210e-03 | 12,676 |
| GM5_fema | control | FEMA 2018 | area | 316.5 | 453.7 | 5.536 | 4.813e-03 | 14,380 |
| GM6_fema | control | FEMA 2018 | area | 243.9 | 375.9 | 1.486 | 2.682e-03 | 12,672 |
| GM7_fema | control | FEMA 2018 | area | 297.0 | 382.0 | 1.249 | 5.223e-03 | 11,006 |
| AP1_3dep | original | 3DEP 2019-20 | area | 324.5 | 440.5 | 1.175 | 0.016 | 19,807 |
| AP2_3dep | original | 3DEP 2019-20 | area | 320.6 | 463.7 | 1.685 | 0.028 | 29,662 |
| AP3_3dep | original | 3DEP 2019-20 | area | 328.0 | 453.0 | 11.072 | 0.019 | 24,627 |
| AP4_3dep | original | 3DEP 2019-20 | area | 317.4 | 443.7 | 17.143 | 0.016 | 22,274 |
| AP5_3dep | original | 3DEP 2019-20 | area | 330.5 | 424.9 | 1.167 | 0.030 | 22,251 |
| AP6 | drawn | 3DEP 2019-20 | area | 327.0 | 450.6 | 1.493 | 0.028 | 24,327 |
| AP7 | drawn | 3DEP 2019-20 | area | 322.9 | 438.3 | 3.647 | 0.015 | 25,154 |
| AP8 | drawn | 3DEP 2019-20 | area | 339.0 | 457.9 | 1.625 | 0.021 | 25,949 |
| AP9 | drawn | 3DEP 2019-20 | area | 311.5 | 430.2 | 1.518 | 0.037 | 22,269 |
| AP10 | drawn | 3DEP 2019-20 | area | 348.1 | 480.5 | 0.425 | 0.020 | 21,668 |
| AP11 | drawn | 3DEP 2019-20 | area | 309.4 | 441.8 | 1.958 | 0.023 | 25,097 |
| AP12 | drawn | 3DEP 2019-20 | area | 315.6 | 441.5 | 1.248 | 0.023 | 25,025 |
| AP13 | drawn | 3DEP 2019-20 | area | 322.4 | 428.7 | 3.967 | 0.027 | 24,958 |
| AP14 | drawn | 3DEP 2019-20 | area | 324.3 | 438.1 | 0.555 | 0.021 | 24,166 |
| AP15 | drawn | 3DEP 2019-20 | area | 329.8 | 432.3 | 2.075 | 0.040 | 26,147 |
| AP16 | drawn | 3DEP 2019-20 | area | 301.3 | 430.5 | 1.692 | 0.017 | 20,021 |
| AP17 | drawn | 3DEP 2019-20 | area | 295.1 | 406.2 | 2.321 | 0.030 | 26,110 |
| AP18 | drawn | 3DEP 2019-20 | area | 320.9 | 416.5 | 1.982 | 0.025 | 26,157 |
| AP19 | drawn | 3DEP 2019-20 | area | 318.7 | 442.9 | 2.109 | 0.020 | 25,308 |
| AP20 | drawn | 3DEP 2019-20 | area | 357.1 | 457.3 | 0.338 | 0.028 | 24,249 |
| AP21 | drawn | 3DEP 2019-20 | area | 326.6 | 432.5 | 0.421 | 0.018 | 18,685 |
| AP22 | drawn | 3DEP 2019-20 | area | 327.8 | 455.4 | 0.395 | 0.025 | 17,849 |
| AP23 | drawn | 3DEP 2019-20 | area | 334.8 | 443.2 | 1.724 | 0.029 | 28,357 |
| AP24 | drawn | 3DEP 2019-20 | area | 326.9 | 458.2 | 1.600 | 0.028 | 24,714 |
| AP25 | drawn | 3DEP 2019-20 | area | 313.4 | 420.3 | 3.266 | 0.023 | 22,479 |
| AP26 | drawn | 3DEP 2019-20 | area | 324.0 | 442.9 | 0.536 | 0.021 | 23,684 |
| AP27 | drawn | 3DEP 2019-20 | area | 330.4 | 436.9 | 1.765 | 0.024 | 21,378 |
| AP28 | drawn | 3DEP 2019-20 | area | 300.6 | 414.4 | 0.750 | 0.022 | 23,223 |
| AP29 | drawn | 3DEP 2019-20 | area | 327.5 | 422.4 | 1.781 | 0.019 | 21,693 |
| TTV1 | M&D | Marin 2019 | area | 76.103 | 305.1 | 0.451 | 0.057 | 18,823 |
| TTV2 | M&D | Marin 2019 | area | 41.684 | 241.5 | 0.635 | 0.044 | 16,853 |
| TTV3 | M&D | Marin 2019 | area | 31.262 | 166.8 | 2.707 | 0.026 | 22,629 |
| TTV4 | drawn | Marin 2019 | area | 51.670 | 321.1 | 0.474 | 0.069 | 21,015 |
| TTV5 | drawn | Marin 2019 | area | 444.8 | 656.3 | 0.830 | 0.049 | 15,241 |
| TTV6 | drawn | Marin 2019 | area | 60.668 | 317.3 | 0.508 | 0.035 | 21,915 |
| TTV7 | drawn | Marin 2019 | area | 164.2 | 389.2 | 0.230 | 0.104 | 14,788 |
| TTV8 | drawn | Marin 2019 | area | 53.359 | 264.3 | 1.036 | 0.141 | 15,723 |
| TTV9 | drawn | Marin 2019 | area | 128.9 | 369.1 | 2.745 | 0.157 | 15,772 |
| TTV10 | drawn | Marin 2019 | area | 100.1 | 317.6 | 0.192 | 0.069 | 20,536 |
| TTV11 | drawn | Marin 2019 | area | 53.714 | 263.1 | 0.188 | 0.092 | 16,576 |
| TTV12 | drawn | Marin 2019 | area | 58.186 | 288.5 | 0.264 | 0.092 | 20,740 |
| TTV13 | drawn | Marin 2019 | area | 36.320 | 185.9 | 0.216 | 0.041 | 21,560 |
| TTV14 | drawn | Marin 2019 | area | 26.150 | 287.6 | 0.450 | 0.056 | 19,277 |
| TTV15 | drawn | Marin 2019 | area | 71.376 | 278.1 | 0.475 | 0.080 | 16,108 |
| AP1 (PAMAP) | original | PAMAP 2006-08 (v1/v2 tile, EPSG:26917) | area | 324.5 | 439.1 | 1.116 | - | 20,928 |
| AP2 (PAMAP) | original | PAMAP 2006-08 (v1/v2 tile, EPSG:26917) | area | 321.1 | 463.6 | 1.738 | - | 28,456 |
| AP3 (PAMAP) | original | PAMAP 2006-08 (v1/v2 tile, EPSG:26917) | area | 328.3 | 453.0 | 11.010 | - | 24,796 |
| AP4 (PAMAP) | original | PAMAP 2006-08 (v1/v2 tile, EPSG:26917) | area | 318.0 | 443.8 | 17.002 | - | 23,150 |
| AP5 (PAMAP) | original | PAMAP 2006-08 (v1/v2 tile, EPSG:26917) | area | 331.9 | 425.3 | 1.220 | - | 22,879 |

### A8. Slurm accounting of the addendum 17 Part A jobs

| JobID | JobName | Partition | State | Elapsed | Timelimit | AllocCPUS | MaxRSS | TotalCPU |
|---|---|---|---|---|---|---|---|---|
| 52862632 | a17_tests | serc | COMPLETED | 00:04:57 | 00:30:00 | 2 |  | 00:00:00 |
| 52863568 | a17_register | serc | COMPLETED | 00:00:58 | 00:30:00 | 1 |  | 00:00:00 |
| 52863574 | a17_ctrlfp | serc | COMPLETED | 00:06:00 | 00:30:00 | 1 |  | 00:00:00 |
| 52865313 | a17_ctrlfetch | serc | COMPLETED | 00:01:24 | 00:30:00 | 4 |  | 00:00:00 |
| 52865319 | a17_ctrlgrid | serc | COMPLETED | 00:03:24 | 00:30:00 | 4 |  | 00:00:00 |
| 52865321 | a17_ctrlcut | serc | COMPLETED | 00:00:33 | 00:30:00 | 1 |  | 00:00:00 |
| 52865322 | a17_figures | serc | CANCELLED by 313293 | 00:01:02 | 00:50:00 | 2 |  | 00:00:00 |
| 52868585 | a17_ctrlcut2 | serc | COMPLETED | 00:00:18 | 00:30:00 | 1 |  | 00:00:00 |
| 52868593 | a17_figures | serc | COMPLETED | 00:07:30 | 00:50:00 | 2 |  | 00:00:00 |
| 52870152 | a17_figures | serc | COMPLETED | 00:06:34 | 00:50:00 | 2 |  | 00:00:00 |

Peak memory per job (GB, from the step records): 52862632: 1.2; 52863568: 1.7; 52863574: 2.5; 52865313: 3.2; 52865319: 7.9; 52865321: 1.2; 52865322: 5.0; 52868585: 0.2; 52868593: 9.5; 52870152: 9.6

Core-hours used (elapsed x CPUs): 1.12; sum of CPU-weighted limits: 12.0.
