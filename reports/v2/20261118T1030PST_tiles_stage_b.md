# [2026-11-18 10:30 PST] Tile expansion, stage B (addendum 15): M&D georeference, re-routing with the gap rule, eligibility, the random draw, tile cutting

Directive: `reports/directives/2026-11-18T0800PST_v2_addendum_15_tile_draw.md` (outline fixed by its section 7). Branch `v2.3-tiles`
(from 9d08244, the stage A report). **Nothing was measured on the 1 m tiles, no model was run, no merge, no tag.** Scripts:
`reports/v2/scripts/a15_common.py`, `a15_reroute.py`, `a15_budget_diag.py`, `a15_md_georef.py`, `a15_ttv_reference.py`, `a15_draw.py`,
`a15_cut_tiles.py`, `make_a15_tables.py` (ruff clean). Environments: `lidar` (grids, the scan), `lemcvt` (pools code, draw, cutting).
Every number is tied to the run directory `$GROUP_HOME/lemcvt_results/20261118T160326Z_a15_stage_b/` (subdirectories named in each
section) or to a Slurm job id. Appendices (section 8) are machine-generated. **This is a stop for review.**

## Summary

| item | result |
| --- | --- |
| 1 state | `v2.3-tiles` = origin = 9d08244; no job running; tests on `serc` 269 passed (job 52815535). Nine deviations listed in section 1; **one stop** (rule 2.2) |
| 2 M&D map | Fig. 3 is page 3 of the WRR scan, 300 dpi, 1-bit, cropped to 1,790 × 1,710 px at 1.256 m/px. Placed by cross-correlating its channel network with the 1 m D8 network (one peak), then **13 ground-control confluences and the pond**: affine **RMS 9.5 m, max 19.3 m (rule 2.1 pass)**. The study area is the block between the Tennessee Valley trailhead and Coyote Ridge's 1,000-ft crest at 539.3 km E (not the 317 m summit stage A used). Digitised: boundary (hand, 116 vertices snapped to the dashes, **2.058 km²**), channels, **77 channel heads** (86 semi-automatic candidates, 9 removed by eye; A-D and the culvert placed; paper: 68 + culvert), 110 colluvial polygons (0.20 km²). **Rule 2.2 fails on its lower bound**: the 1 m grid inside the polygon reaches 36.9 m at the two valley exits (window 46-76; area and the 310 m top pass). TTV polygon work is therefore provisional |
| 3 re-routing | Enclosed gaps filled (AP 1,050, 0.70 km²; TTV 525, 0.49 km²), edge-connected gaps left as outlets; AP's watershed outlet now drains 89 % (514 km²); GM routed as one 1 m grid (**after a 2-cell nodata pad: the lemcvt D8 quirk on grid-border data made 134 two-cell cycles and an unclosed budget**); budgets closed at all three sites |
| 4 eligibility | Set B screens, θ dropped, k_s and \|C_ht\| ±25 %, gap rule: **GM 14 → widened to ±50 %: 33; AP 54 at ±25 % (no widening); TTV 1 (widened)**. TTV's polygon k_s by the rule's statistic is **4.0**, a median over boundary-truncated basins (pooled 22.7); it leaves one eligible window. Of the original ten, only GM3 would be eligible as the rules stand; AP1, AP2, AP5 lie 74-86 % over longwall panels |
| 5 draw | Seed 20261118, config written first. **GM 2 new tiles** (20 of 33 eligible windows overlap an original, 7 drain into one); **AP 24 new** (list exhausted one short of 25); **TTV 1 new + the 3 M&D windows** (92.9 % cover of the polygon). Ordered lists with every skip reason in Appendix B7; maps `draw/<site>_draw_map.png` |
| 6 tiles | **36 tiles cut and registered**: `GM-x` (GM1-GM5 re-cut, GM6, GM7), `AP-3DEP-x` (AP1-AP5 as `_3dep`, AP6-AP29), 202 manifest rows, `versions` block in `configs/data.yaml`; 4 TTV tiles cut provisionally, not registered. Unit check passes on every tile. GM re-cuts equal the existing grids in z and flow direction exactly; drainage area differs only on the trunk cells of GM2 and GM5 (0.06-0.18 % of cells), where the old grids carried upstream area from beyond the lidar footprint |
| cost | 35 submissions, **2.4 core-hours used** (expected ≤ 40), **sum of limits 34 core-hours** (cap 80), largest job 4 CPUs × 80 min; every job on `serc`; new data 233 MB of tiles under `$LEMCVT_DATA`, 3.3 GB in the run directory; `$GROUP_HOME` 312 GB of 1 TB |

## 1. State check

**At the start (2026-11-18 08:00 PST, login node sh04-ln01).**
- `v2.3-tiles` = `origin/v2.3-tiles` = 9d08244 (the stage A report); no job was running (`squeue --me` empty).
- Tests on `serc` at 9d08244: job 52815535, **269 passed** (4 min 26 s, 2 CPUs). `ruff` is not installed in the `lemcvt`
  environment (the job printed "No module named ruff"); the `ci312` environment's ruff was run on the new scripts instead
  (`reports/v2/scripts/a15_*.py`, `make_a15_tables.py`): clean with the repository's rules.
- The stage A paths of section 0.3 all exist: `$GROUP_HOME/lemcvt_results/20261118T002225Z_a14_stage_a/{pools,grid_AP,grid_TTV,
  ttv_inventory,footprints,marin_select,usgs_vs_pasda}`, `scripts/a14_thresholds_B.json`, the copies of record under
  `$GROUP_HOME/lemcvt_data/source/`, the papers, and the `lemcvt` and `lidar` environments.
- Read in the order of section 0.2 (stage A report, addendum 14, the data report, `docs/measurement_methods.md` section 0,
  addendum 4 section 1, addendum 2 sections 4-5, addendum 7 section 0.4).

**Where this addendum and the repo or the data differ, and every deviation from the addendum** (none stopped the work
except the one marked STOP; each is handled as stated):
1. **Rule 2.2 fails on its lower bound (STOP, section 2).** The digitised study-area polygon is 2.06 km² (pass) and reaches
   310 m (pass), but the 1 m grid inside it goes down to 36.9 m where the rule allows 46-76 m. The low cells are the two
   valley-floor exits on the eastern and southern sides (section 2.5). As the rule is written the TTV polygon-dependent
   chain must stop; GM and AP do not depend on it. The TTV reference, eligibility, draw and tiles below were still computed
   so that the review has them, and they are marked **provisional** throughout; the TTV tiles were not registered.
2. **The GM full grid needed a nodata pad (section 3).** `Gabilan_mesa.tif` has valid cells on its grid border; the lemcvt
   D8 port (a TopoAnalysis quirk kept deliberately) never compares the last two rows and columns eastward and southward,
   which gave 134 mutual two-cell cycles there and an unclosed area budget (38,883,247 of 39,375,330 cells). The grid was
   padded by two nodata cells on every side and routed again; the products are cropped back to the original lattice; the
   budget closes. The stage A AP and TTV lattices carry a 5 m pad, so they were never affected. The first pass is kept in
   `grid_GM/first_pass_unpadded/`.
3. **Products "beside" the stage A ones** are written in the stage B run directory (`grid_<site>/`), not inside the stage
   A run directory, so that stage A stays untouched.
4. **TTV window statistics were recomputed** (section 4): the stage A χ-z R² and k_s of the TTV windows were computed at
   the proxy θ = 0.387; decision 4 replaces the reference by the polygon's θ = 0.450, so both statistics were recomputed
   at that θ for every candidate window (`ttv_reference/TTV_windows_a15.csv`). Nothing else was recomputed.
5. **The conflict graph across lattice offsets** is not in stage A (its direct edges are per offset). The draw pools the
   four offsets, so the same "next window downstream" rule was extended: window u drains directly into window v when a
   cell of u reaches v along its D8 path before entering any other candidate window of v's lattice offset; for u and v of
   the same offset this is exactly the stage A edge (checked: identical edge sets on every offset, section 5). The
   pre-placed tiles form a fifth group of the same rule.
6. **AP original tiles** are 990 m by 1,015 cells of 0.9752 m in EPSG:26917. Their `_3dep` versions are 1,000 × 1,000 m on
   the EPSG:6346 lattice from the floored south-west corner (section 6.1 and 6.3); the NAD83 → NAD83(2011) step is the
   identity offline (PROJ has no grid), as in stage A.
7. **No `pyproj` in the `lemcvt` environment**: the draw script uses GDAL/OSR for the one reprojection it needs.
8. **Channel heads were digitised semi-automatically** (section 2.4): the mapped channel lines are thinned and their
   upstream end points are the heads; 86 candidates were reviewed by eye on gridded views and 9 removed. The paper's
   count (68 + the culvert) is not reproduced exactly.
9. **`cry_for_help.txt` and `working-with-steve.md`** are untracked in the working tree as before; nothing under `papers/`
   is tracked (pre-push check, section 8).


## 2. The Montgomery & Dietrich (1989) map: georeference and digitising (section 2)

Script `reports/v2/scripts/a15_md_georef.py` (jobs 52817132, 52817891, 52819086, 52820070, 52820695, 52822212, 52822793
and their `check` stages); products in `md_georef/`: `md1989_fig3.gpkg` (EPSG:6339; layers `study_area`, `channels`,
`channel_heads`, `colluvial_deposits`, `ground_control`), `md_study_area_1m.tif`, the masks, `digitized_overlay.png` (the layers
on the 1 m hillshade) and `digitized_on_figure.png` (the same in figure space). Appendices B1-B2.

### 2.1 Extraction (2.1)
Fig. 3 is on **page 3** of the WRR scan, a 1-bit CCITT image of **2,455 × 3,379 px at 300 dpi**; the map was cropped at page
pixels (330, 250)-(2120, 1960), **1,790 × 1,710 px**. The 0-500 m scale bar spans **398 px**, so the map is at **1.2563 m/px**
(1:14,838, a 1.6× enlargement of the 7.5′ quadrangle). On the scan the contour lines are 4-5 px wide, the channels 7-9 px,
the boundary dashes 8-10 px and the stipple dots 3-5 px, which is what separates them: a 6-px disk opening keeps the thick
lines (1,095 components; 201 of them above 120 px form the channel network and the boundary), the rest are contours,
outlines, dots and text.

### 2.2 Placement and ground control (2.2)
- **Coarse placement.** The figure's channel network, at the scale-bar scale, was cross-correlated with the D8 network
  (A ≥ 2 × 10⁴ m²) of the stage A TTV 1 m grid over 538-546 km E, 4,185-4,194 km N at 5 m, over scales 0.9-1.1 and
  rotations ±3°. One peak stands out: scale 1.0, rotation 0, crop top-left at 538,950 E, 4,191,715 N (ncc 0.214 against
  a second peak of 0.161 at 541,055 E, 4,188,100 N). The study area is therefore the 2 km block between the Tennessee
  Valley trailhead and Coyote Ridge's 1,000-ft crest at 539.3 km E, not the 317 m summit at 542.0 km E that stage A had
  taken for Coyote Ridge (its proxy catchment, 0.67 km² above the trailhead, lies inside the southern third of the polygon).
- **Local refinement.** 248 patches (150 m) of the figure's network matched within ±110 m on the 1 m D8 network: 38 good
  matches, 33 used, patch affine RMS 12.0 m, max 26.1 m.
- **Ground control (rule 2.1).** 13 features: 12 channel confluences identified on both the figure (junctions of the thinned
  channel skeleton) and the 1 m grid (D8 confluences of two tributaries ≥ 2 × 10³ m² at ≥ 4 × 10³ m²), each pair checked by
  eye on side-by-side chips (figure junction, hillshade with the D8 network), and the man-made pond (the solid black symbol
  at the map's centre; a flat-floored basin with a dam on the hillshade at 539,998 E, 4,190,672 N). They span rows 270-1,305
  and columns 384-1,590 of the 1,710 × 1,790 crop. Affine least squares: **RMS 9.5 m, maximum 19.3 m (the pond)**, pixel
  size 1.2278 × 1.2748 m (the scan is 3.8 % taller than wide; the scale bar gives 1.2563 horizontally), rotation −0.3° /
  +0.7°. **Rule 2.1 passes**; no second-order polynomial was needed. Appendix B1 lists the points and residuals.
- The contour labels confirm the placement: the figure's 1,000-ft label sits on the 304.8 m contour of the 1 m grid at the
  west edge, 800 on the 243.8 m contour to the north-west, 200 at the eastern exits (Appendix figure `overlay_prelim.png`).

### 2.3 Digitised layers (2.4)
- **Study-area boundary**: the dash-dot line digitised by hand from gridded views of the scan (116 vertices, each then
  snapped to the nearest pixel of a dash or dot within 25 px; 106 of 116 moved). On the south-east the boundary follows the
  solid tributary channel between the last dash of the north-east arc and the dashes at row 1,425. **2.058 km²**.
- **Channels**: the thinned network inside the boundary corridor, traced node to node (9,500 line pieces in 67 connected
  trees). Their summed length, 31 km, is not comparable with the paper's 9.1 km: it counts every spur of the thinned mask
  (merged contours attached to channel lines, loops at junctions). The main trees are right by eye (`digitized_overlay.png`:
  the red lines follow the hollows and the D8 network).
- **Channel heads**: the upstream end points of the thinned network. Per connected tree the lowest end point on the 1 m
  grid is its outlet, end points within 10 px of the boundary line are exits, terminal branches thinner than 5.8 px over
  their junction-side 70 % (contour material) or shorter than 15 px were removed first (65 branches). This gave **86
  candidates**, reviewed one by one on gridded quadrant views (`head_candidates_<quadrant>.png`); **9 were removed** (a spur on
  the main east-west channel, two downstream ends of discontinuous reaches on the north-facing slope that the 1 m grid put
  above their upper ends, five second tips of forks a few pixels from a kept head, one downstream end below head 76):
  **77 heads**, of which 16 are flagged `continuation = 1` (the tree's outlet lies within 50 m of, and not above, another
  tree's channel, i.e. a lower reach of a discontinuous channel; the paper counted one head per hollow, so 61 heads plus 16
  possible continuations). The paper mapped 68 channelled hollows of 90 (5 excluded for trails) and the culvert head, so the
  digitised set is about 10 % over, which is the price of the semi-automatic extraction: some first-order stubs of the scan
  are too short or too thin to survive the filters (missed heads) and some fork tips survive (extra heads). Heads **A, B, C,
  D** were placed as the nearest head within 90 px of each letter (A at 540,299 E, 4,190,731 N; B 540,548, 4,190,418;
  C 540,225, 4,189,994; D 540,302, 4,191,297) and the **culvert head** as the nearest head within 40 px of the triangle
  (539,622 E, 4,190,782 N; the triangle itself maps to 539,627 E, 4,190,738 N, and the road bench crossing the hollow is
  visible on the hillshade between the two). Heads that cannot be placed: none of the labelled
  ones; the 5 trail-affected heads are not on the map.
- **Colluvial deposits**: stipple-dot density (dots ≤ 30 px, density > 0.03 in a 21 px window, inside the boundary):
  **110 polygons, 0.202 km²**, 9.8 % of the study area; their blocky outlines are the 1 m raster polygons of the warp.
- **The 1 m grid inside the polygon**: 36.9-310.1 m (1st-99th percentiles 48.9-285.2 m).

### 2.4 Rule 2.2 — **FAIL on the lower bound; STOP for the TTV polygon chain**
| test | value | window | result |
| --- | --- | --- | --- |
| area | 2.058 km² | 1.785-2.415 | pass |
| z_min on the 1 m grid | **36.9 m** | 46-76 | **fail** |
| z_max on the 1 m grid | 310.1 m | 299-329 | pass |

The cells below 46 m are 8,686 of 2,058,028 (0.4 %), in two clusters: 6,115 cells at the eastern exit (541,015 E, 4,190,831 N,
minimum 36.9 m) and 2,434 at the southern exit toward the trailhead (540,541 E, 4,189,930 N, minimum 39.9 m); 6.1 % of the
polygon lies below 61 m. The paper's "61 m in the valleys on the eastern end" is the 200-ft contour of the quadrangle; the
digitised boundary reaches 40-80 m beyond that contour at both exits, where the valley floors of the lidar are 37-46 m
(NAVD88; NGVD29 differs by under 1 m here). Whether this is a 40-80 m digitising excess at two vertices or the map's
rounding cannot be told from the figure; the georeference itself (rule 2.1) is not in question. **As written, rule 2.2
fails, so the TTV reference, eligibility, draw and tiles that depend on the polygon are reported below as provisional**
(section 1, deviation 1) and the TTV tiles were not registered; GM and AP do not depend on it.


## 3. Re-routing with the gap rule, and the GM full grid (section 3, decision 8)

Script `reports/v2/scripts/a15_reroute.py`; jobs 52815802 (AP, 4 CPUs, 14 min 45 s, 49.9 GB), 52815791 (TTV, 3 CPUs, 3 min 53 s,
19.7 GB), 52817205 (GM, padded pass, 2 CPUs, 1 min 9 s, 6.1 GB); products in `grid_<site>/`: `<site>_1m_gapfill_mask.tif`
(1 = enclosed gap filled for routing), `<site>_1m_filled.tif`, `<site>_1m_fd.tif`, `<site>_1m_area_cells.tif`, `<site>_gaps.json`,
`<site>_route.json`. Appendix B3.

**Gap classification.** Every nodata component of the stage A 1 m grid (8-connected) is edge-connected if it touches the grid
border (the area outside the footprint, the sea and the Bay, and every gap joined to them) and enclosed otherwise. Enclosed
gaps are filled by nearest neighbour from the valid cells (Euclidean, in a sub-block padded by half the gap's smaller
dimension), so that flow passes through them; edge-connected gaps stay nodata and their rims are outlets, as in stage A.
Then priority flood, D8 and area on the whole grid (`a14_bigroute`, the lemcvt conventions).

| site | nodata components | enclosed (filled) | enclosed cells | enclosed km² | size p50 / p90 / p99 / max (cells) | fill distance p50 / p90 / p99 / max (m) | enclosed ≥ 10⁴ cells | valid cells after | outlets | largest basin | budget closed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP | 1,051 | **1,050** | 700,025 | 0.700 | 28 / 1,354 / 6,656 / 158,941 | 2.0 / 10.8 / 28.7 / 179.7 | 5 | 577,668,660 | 149,881 | **514.3 km² (89.0 %)** at 581,957 E, 4,425,297 N | yes |
| TTV | 526 | **525** | 492,163 | 0.492 | 17 / 409 / 6,180 / 207,788 | 1.4 / 7.1 / 28.3 / 106.2 | 3 | 141,299,405 | 118,919 | 22.6 km² (16.0 %) at 537,129 E, 4,190,296 N (Tennessee Cove); 18.5 km² (13.1 %) at 542,121 E, 4,193,755 N (Richardson Bay side) | yes |
| GM | 1 (padded) | 0 | 0 | 0 | - | - | - | 39,375,330 | 39,238 | 15.0 km² (38.1 %) at 697,277 E, 3,980,645 N; 14.7 km² (37.4 %) at 693,138 E, 3,977,282 N | yes (padded); **no** unpadded |

- **AP.** With the ponds, pits and river reaches inside the footprint filled, the watershed outlet drains 89.0 % of the grid
  (514.3 km²), as in the stage A first pass (89.1 % with every gap filled); the next outlets drain 1.5 and 0.9 km². The
  stage A second pass (the products kept then) had split the watershed into basins of 158, 101 and 63 km² at the gap rims.
  The five enclosed gaps above 10⁴ cells (1.0-15.9 ha) are river reaches without ground returns; 381 of the 1,050 are
  above 100 cells. The fill distance reaches 180 m in the widest reach.
- **TTV.** 525 enclosed gaps, the largest 20.8 and 13.6 ha (the lagoons behind Rodeo Beach and on the Bay side; fill
  distance up to 106 m); the Pacific, the Bay and the strip outside the footprint are one edge-connected component.
  The Tennessee Cove outlet (22.6 km²) is unchanged from stage A; the Richardson Bay side now drains 18.5 km² to one
  outlet instead of 9.5 and 9.3 km² to two.
- **GM.** `Gabilan_mesa.tif` (12,639 × 7,802 cells, WGS 84 / UTM 10N, metres, ellipsoidal) has no enclosed gap: its two
  nodata components both touch the border. Routed as one grid with the stage A conventions after the pad of deviation 2:
  priority flood 12.8 s, D8 8.3 s, area 13.2 s, 5.2 GB; 39,238 data-edge outlets whose areas sum exactly to the valid
  cells. The two largest basins (15.0 and 14.7 km²) are the trunk valleys that cross the lidar footprint; the tiles'
  drainage areas of section 6 come from this grid.
- **Budget diagnostic** (`a15_budget_diag.py`, jobs 52816789 and 52817235): unpadded, 69 cells had a receiver higher than
  themselves and 134 formed two-cell cycles, all in the last two columns of the grid; padded, 0 and 0. The float32 product
  `GM_1m_filled.tif` stores 586,822 receiver pairs at equal elevation because the 10⁻¹² aggradation increments of the flood
  do not survive the float32 write; the flow directions and areas were computed from the float64 surface in memory, as in
  stage A.

## 4. Eligibility under the settled rules (section 4)

Script `a15_draw.py` (job 52824241; the first pass, job 52822218, used a stricter cross-group independence rule and is kept in
`draw_v1_groupwise/`, section 5). Inputs: the stage A candidate windows and screen statistics (`pools/<site>_windows.csv`),
threshold set B, the references of decision 4, the re-routed 1 m grids for the gap rule. Appendices B4-B6.

### 4.1 The TTV reference (decision 4) — provisional, and degenerate as the code stands
On the 10 m NED, the polygon (20,578 cells, 2.058 km², 38-309 m) by the stage A code gives **θ = 0.437** (legacy χ, 33
basins; coarse grid 0.425), **|C_ht| = 17.3 × 10⁻³ m⁻¹** (2,258 convex divide pixels) and **k_s = 4.02**, the median of the 33
per-basin through-origin slopes at θ = 0.437. That median is not a usable reference: the 33 "basins" are the coastline
outlets of the polygon mask, most of them truncated by the boundary on the hillslope side, with slopes from 2.7 (16th
percentile) to 23.7 (84th); the pooled slope of the same points is **22.7**, the whole-grid pooled slope at the same θ is
21.0, the median of the 420 candidate windows' k_s at this θ is 21.5, and the three windows that cover the M&D area
themselves (section 5.2) have k_s 4.6-7.1 times the "reference". The stage A proxy catchment (0.67 km², 15 basins) had
given 0.387 / 15.4 / 15.1 × 10⁻³; the whole-grid median at θ = 0.437 is 13.7. The rule says the stage A code and the
median, so 4.02 was used; the consequence and the alternatives are in section 7.

Because the reference θ moved from 0.387 to 0.437, the two TTV window statistics that depend on it were recomputed at
0.437 for all 420 candidates (section 1, deviation 4): χ-z R² ≥ 0.90 now keeps 343 windows instead of 349; k_s moved from
a median of 12.7 to 21.5; ΔBIC and every other column are the stage A values.

### 4.2 Counts
| site | candidates | set B screens | within ±25 % (k_s and \|C_ht\|) | screens and ±25 % | gap-excluded | **eligible at ±25 %** | within ±50 % | screens and ±50 % | gap-excluded | eligible at ±50 % | threshold | **tolerance used** | **eligible** (per offset) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GM | 71 | 38 | 23 | 14 | 0 | **14** | 58 | 33 | 0 | 33 | 25 | **±50 %** (widened) | **33** (7, 9, 7, 10) |
| AP | 1,653 | 320 | 651 | 68 | 14 | **54** | 1,460 | 276 | 77 | 199 | 25 | **±25 %** | **54** (14, 17, 11, 12) |
| TTV (provisional) | 420 | 100 | 3 | 1 | 0 | **1** | 16 | 1 | 0 | 1 | 30 | **±50 %** (widened) | **1** (0, 0, 0, 1) |

- **θ is reported, not used** (decision 2): the eligible windows' θ at GM spans 0.25-0.55 (median 0.40 against the reference 0.292), at AP 0.10-0.80 (median 0.57 against 0.54) (Appendix B7).
- **The gap rule** (decision 8) removes 14 of the 68 AP windows that pass the screens at ±25 % (77 of 276 at ±50 %): a
  channel cell (A ≥ 10³ m²) crossing a filled river reach or pond. It removes nothing at GM (no enclosed gap) or TTV.
- **Widening** (decision 3) triggers at GM (14 < 25) and TTV (1 < 30), not at AP (54 ≥ 25).
- The screens alone keep GM 38, AP 320, TTV 100 (stage A: 38, 320, 102; the TTV change is the R² recomputation).

### 4.3 The original ten tiles under every rule (decision 9)
| tile | trunk inside (km²) | largest inflow (km²) | χ R² | ΔBIC | NLCD 11, 22-24, 31 | longwall | upstream valid | set B screens failed | θ | k_s dev | \|C_ht\| dev | within ±25 % | within ±50 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GM1 | 1.15 | 0.90 | 0.995 | 2.5 | 0.000 | - | yes | none | 0.362 | +23 % | +34 % | no | yes |
| GM2 | 8.88 | 8.47 | 0.982 | −4.8 | 0.000 | - | yes | trunk, inflow | 0.367 | +11 % | +9 % | yes | yes |
| GM3 | 0.34 | 0.002 | 0.983 | −5.3 | 0.000 | - | yes | none | 0.262 | +8 % | −6 % | yes | yes |
| GM4 | 6.20 | 5.16 | 0.990 | 6.7 | 0.000 | - | no | trunk, inflow, upstream | 0.321 | +88 % | +18 % | no | no |
| GM5 | 4.73 | 2.66 | 0.980 | −9.3 | 0.000 | - | yes | inflow | 0.312 | +16 % | +33 % | no | yes |
| AP1 | 1.06 | 0.12 | 0.994 | 49.0 | 0.008 | **0.81** | - | ΔBIC, longwall | 0.621 | −31 % | −14 % | no | yes |
| AP2 | 1.67 | 0.99 | 0.983 | 46.6 | 0.002 | **0.74** | - | ΔBIC, longwall | 0.780 | −15 % | −19 % | yes | yes |
| AP3 | 10.7 | 8.70 | 0.990 | −10.5 | 0.002 | 0.09 | - | trunk, inflow, longwall | 0.512 | +14 % | −21 % | yes | yes |
| AP4 | 16.6 | 16.6 | 0.980 | 14.1 | 0.007 | 0.00 | - | trunk, inflow | 0.692 | −22 % | 0 % | yes | yes |
| AP5 | 1.23 | 0.60 | 0.993 | 10.6 | 0.003 | **0.86** | - | longwall | 0.433 | −30 % | +6 % | no | yes |

Under the settled rules **GM3 alone** would be eligible at ±25 % (GM2 and the AP tiles meet the tolerance but fail a
screen); at ±50 % GM1 and GM5 would also meet the tolerance but GM5 fails the inflow screen. Four AP tiles sit over longwall
panels (74-86 % of AP1, AP2, AP5; 9 % of AP3). They all stay (decision 9); AP5 needs no further action.

## 5. The draw (section 5)

### 5.1 Procedure as run
- Seed **20261118**, written to `draw/draw_config.json` (with the thresholds, tolerances, targets and references) at
  09:35:55 PST before any draw.
- The eligible windows of the four offsets pooled, sorted by (offset, window id), permuted once with
  `numpy.random.default_rng(20261118).permutation`; greedy acceptance in that order.
- Overlap: any shared area between 1,000 m tile footprints (the window's south-west corner floored to the 1 m lattice;
  edge-to-edge contact allowed).
- Independence: within one lattice offset the stage A direct edge (recomputed here and identical to
  `pools/<site>_conflicts.json` on every offset: GM 50/58/53/52 edges, AP 1,564/1,505/1,548/1,546, TTV 390/382/357/339);
  across offsets and with the pre-placed tiles, u drains directly into v when v contains the first cell of a D8 path from u
  that lies in any candidate window or pre-placed tile not overlapping u, i.e. no other window lies between them (section 1,
  deviation 5). The first pass had used a stricter per-group rule (the first pre-placed tile reached, however many
  candidates lie between), under which every GM window upstream of GM4's trunk was blocked and GM accepted nothing; it is
  kept in `draw_v1_groupwise/` and the difference is only at GM (AP 23 instead of 24 accepted, TTV the same).
- The gap exclusion was applied before the count (section 4) and re-checked at acceptance.
- Pre-placed: GM1-GM5 and AP1-AP5 at their footprints (AP: the EPSG:26917 bounds reprojected, floored); TTV: the three M&D
  windows of 5.2.

### 5.2 TTV: the M&D windows
28 candidate windows intersect the polygon; the fewest non-overlapping ones covering ≥ 80 % are **three** (offset 0 window 135
at 539,000 E, 4,190,000 N; offset 2 window 118 at 540,000, 4,190,500; offset 2 window 136 at 540,000, 4,189,500), covering
**92.9 %** of the polygon (one window reaches 45.8 % at most, two non-overlapping ones 72.2 %). Their status, reported and not applied: only the third passes
the set B screens (the first two fail the trunk or inflow screen; the northern window holds the 2.3 km² channel that
crosses the polygon); none is within the k_s tolerance at either width (k_s 4.6-7.1 × the degenerate reference, |C_ht|
within −22 % to +7 %); none is gap-excluded. Pre-placed as TTV1-TTV3.

### 5.3 Result
| site | eligible | tolerance | target | accepted | skipped: overlaps original | overlaps accepted | drains-through edge | **tiles in all** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GM | 33 | ±50 % | 25 | **2** | 20 | 4 | 7 (3 with GM1/GM4, 4 with accepted) | **7** (GM1-5, GM6, GM7) |
| AP | 54 | ±25 % | 25 | **24** | 2 (AP4) | 24 | 4 | **29** (AP1-5 as `_3dep`, AP6-AP29) |
| TTV (provisional) | 1 | ±50 % | 30 | **1** | 0 | 0 | 0 | **4** (TTV1-3 M&D, TTV4) |

- **GM**: the lidar footprint (39 km² of fully valid 10 m cells) holds the five originals in its middle, and 20 of the 33
  eligible windows overlap one of them (GM1, GM2, GM3: 6 each; GM4, GM5: 1 each); of the 13 others, 7 drain directly into
  an original or an accepted window and 4 overlap the two accepted ones. GM6 (offset 3, window 447; 697,481 E, 3,978,961 N)
  and GM7 (offset 2, window 470; 697,981 E, 3,977,961 N) are the new tiles, both between GM3 and GM5. Far fewer than the
  25 asked for, and no rule change short of allowing overlap with the originals would add many (section 7).
- **AP**: 24 of 25 at ±25 %, spread over the watershed (`AP_draw_map.png`); the list was exhausted at rank 54 with 24
  accepted (the 25th would have needed one more eligible window). 
- **TTV**: one eligible window (offset 3, window 148; 542,500 E, 4,188,500 N, south-east of the M&D area), accepted as TTV4.
The full ordered lists with every skip reason are in Appendix B7; the maps are `draw/<site>_draw_map.png`.

## 6. Tile cutting (section 6)

Script `a15_cut_tiles.py` (jobs 52827880 GM + AP, registered; 52827037 TTV, provisional, `--no-register`); records `tiles/tiles.json`,
`tiles/tiles.csv`, `tiles_ttv_provisional/`. Appendix B8 has the per-tile table and the sha256 of every product.

**Products per tile** (1,000 × 1,000 cells on the site's 1 m lattice, south-west corner at the window's floored corner;
AP coarse windows are 990 m and the footprint takes 1,000 m from the same corner): `<tile>_1m_filled.tif` (m, float32,
clipped from the re-routed full grid), `<tile>_1m_fd.tif` (lemcvt D8 codes) and `<tile>_1m_area.tif` (m², float64, the
cell count at 1 m, the format of the existing tiles) **clipped from the full-grid routing, never recomputed**;
`<tile>_1m_gapfill_mask.tif`; and for AP and TTV `<tile>_1m_class2_count.tif` and `<tile>_1m_nnfill_mask.tif` (stage A:
1 = nearest-neighbour filled within 10 m, 2 = gap left nodata). No tile has an outlet cell (flow direction 0) inside it,
as expected of a clip.

| version | tiles | kind | z median range (m) | z range over tiles (m) | site window | largest A inside (km²) | class-2 pts m⁻² | NN-filled fraction | filled-gap cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `GM-x` | GM1-GM5 (re-cut), GM6, GM7 | 7 | 212-358 | 134.9-424.7 | 100-500 | 0.34-6.19 | - | - | 0 |
| `AP-3DEP-x` | AP1_3dep-AP5_3dep, AP6-AP29 | 29 | 348-409 | 295.1-480.5 | 230-490 | 0.34-17.1 | 2.73-3.26 | 0.015-0.040 | AP12 3,736, AP16 409, AP18 13 (none on a channel cell) |
| `TTV-x` (provisional, not registered) | TTV1-TTV3 (M&D), TTV4 | 4 | 83-204 | 29.7-305.1 | 0-320 | 0.45-2.71 | 8.9-16.0 | 0.026-0.076 | 0 |

- **Unit check (6.4): every tile passes** (median, minimum and maximum inside the site window).
- **AP originals as `AP<n>_3dep`** at the reprojected, floored footprints (AP1 568,503 E, 4,408,340 N; AP2 561,573, 4,426,160;
  AP3 551,854, 4,419,230; AP4 555,093, 4,419,230; AP5 556,893, 4,422,470; the old tiles' corners are at .648/.778 m). Their
  medians (374-388 m) are the PAMAP tiles' metre values within the rule 5.1 offsets of stage A. The PAMAP tiles are untouched.
- **GM originals re-cut and compared cell by cell** (Appendix B8): same footprints; **filled elevation identical to the
  millimetre on 100 % of cells and flow direction identical on 100 %** (the old tiles were cut from the same Gabilan grid with
  the same conventions); drainage area identical on 99.8-100 % of cells. The differing cells are trunk cells where the
  new area is smaller: GM2 by up to 4.09 km² (old 8.89, new 4.80 km²; 0.06 % of cells), GM5 by up to 3.84 km² (4.73 →
  2.73 km²; 0.18 %), GM1 by up to 9,633 m² (0.06 %), GM4 by 16 m², GM3 none. The old GM2 and GM5 area grids therefore
  carried upstream area that enters the lidar footprint from outside (the stage A coarse routing on the 58 km² NED gives
  the same 8.9 and 4.7 km² trunks); routed on `Gabilan_mesa.tif` alone, with data-edge outlets, that area is not there. It
  is decision 7's "upstream area fully in valid data" rule in action: GM2 and GM5 fail it, GM1, GM3, GM4's interiors do not
  (GM4's 16 m² is one edge cell). The existing tiles are not replaced.
- **Registration (6.5)**: 202 rows appended to `data/manifest.csv` (site = the version name, kind `input`, roles `filled`,
  `fd`, `area`, `gapfill_mask`, `class2_count`, `nnfill_mask`), and a `versions` block in `configs/data.yaml` (directory,
  CRS, source, run, products, tile list). Files live under `$LEMCVT_DATA/<version>/<tile>/` (GM-x 38 MB, AP-3DEP-x 168 MB,
  TTV-x 27 MB); nothing of them is in git. `lemcvt data verify` covers them with the rest.


## 7. What is weak, and decisions needed

**Weak.**
1. **The TTV k_s reference** (4.1): the rule's statistic (median of per-basin slopes, stage A code) on a polygon mask is a
   median over boundary-truncated basins and comes out at 4.0, a fifth of the pooled slope of the same points (22.7), of the
   whole-grid pooled slope (21.0) and of the candidate windows' median (21.5). With it TTV has one eligible window; with
   the polygon's pooled slope it would have 37 at ±25 % (67 at ±50 %) after the screens; with the whole-grid median at the
   polygon θ (13.7), 6 and 20. The M&D windows themselves miss the rule's reference by a factor 4.6-7.1.
2. **Rule 2.2's lower bound** (2.4): 0.4 % of the polygon lies below 46 m at its two valley exits. The digitised
   boundary may overshoot the map's by 40-80 m there, or the paper's 61 m is the 200-ft contour rounded; the figure cannot
   tell. Everything downstream of the polygon at TTV is provisional until this is settled.
3. **GM has almost no room**: 20 of 33 eligible windows overlap an original and the rest drain into one; two new tiles.
   The lidar footprint is 39 km², the five originals cover 5 km² of its middle, and the trunk valleys cross it, so most
   windows either touch an original or drain through it. Only overlap with the originals (not allowed), or a different
   lattice, could add more.
4. **Channel heads from a scan** (2.3): 77 digitised against 69 mapped; the semi-automatic extraction has both missed
   heads (short or thin first-order stubs) and spurious ones (fork tips); the labelled heads A-D and the culvert were
   matched by proximity to their letters. The heads layer is good enough to locate the M&D hollows on the 1 m grid, not to
   reproduce the paper's census.
5. **The channel-length total** (31 km) is a count of thinned-mask pixels including spurs; the paper's 9.1 km is not
   recovered by this method.
6. **Colluvial polygons** are stipple-density footprints with 1 m raster edges (110 polygons, 0.20 km²); fine for "is this
   hollow mapped as filled", not for areas.
7. **The GM D8 edge quirk** (3): valid data on the grid border give two-cell cycles in the last two columns; the pad fixes
   it here, but every lemcvt routing of a grid whose data reach its border has the same exposure (the ten original tiles
   are cut from larger grids and their edges are outlets, so they are not affected; `tests/test_hydro.py` passes because
   its grids are padded).
8. **The cross-group independence rule** (5.1) is my generalisation of stage A's "next window downstream"; the first pass
   with a stricter reading accepted nothing at GM. Both passes are in the run directory.
9. **AP's gap rule** removed 14 of 68 windows at ±25 %: channels crossing filled river reaches; those reaches are where the
   lidar has no ground returns, so the exclusion is working as intended, but it biases the AP draw away from the valley
   bottoms.

**Decisions needed.**
1. **TTV reference statistic**: keep the median of basins (TTV = 3 M&D windows + 1), or re-define the polygon k_s as the
   pooled through-origin slope (the statistic the v2 measurement uses; 37 eligible at ±25 %), or require a minimum basin
   size. A re-draw at TTV is a few minutes once decided.
2. **Rule 2.2**: accept the polygon with the lower bound explained (the exits), or move the two exit vertices inward to
   the 200-ft contour and re-run (changes the reference slightly and the M&D windows not at all).
3. **GM**: accept 7 tiles, or allow new windows to overlap the originals' footprints (the overlap rule is decision 7's
   "tiles never overlap"), or draw from the 10 m lattice at a 500 m step without the originals pre-placed.
4. **AP**: 24 new tiles at ±25 % (29 in all). Accept, or widen to ±50 % for the last one (not allowed by decision 3 as
   written).
5. **Heads**: use the 77 as they are, or hand-check the 16 continuations and the missed stubs on the figure before the
   TTV measurements (section 2.3).
6. **Registration of TTV tiles** once 1 and 2 are settled.

## 8. Appendices: cost, pre-push check, generated tables

### 8.1 Cost (decision 10)
Estimates were printed before every submission (job scripts in `$SCRATCH/lemcvt_jobs/a15/`). Appendix B10 has the
accounting of all 35 submissions: **2.4 core-hours used** (expected ≤ 40); **sum of CPU-weighted time limits 34 core-hours**
(cap 80); every limit at most twice its estimate with the 30 min floor (the longest, AP re-routing, 4 CPUs × 80 min against
a 35 min estimate, ran 14 min 45 s at 49.9 GB); largest submission 5.3 core-hours of limits; every job on `serc`, no
array. About a third of them were resubmissions after script fixes or cancelled chains (the georeference stages ran seven
times while the digitising was tuned; the draw three times; the cut four times, twice because Slurm's `--export` splits
on commas and the first "GM,AP" run cut GM only). New data on group storage: 233 MB of tiles, 3.3 GB in the run directory
(the three re-routed 1 m grids with their masks); `$GROUP_HOME` at 312 GB of 1 TB. No scratch is left behind except the
job logs and the figure extractions under `$SCRATCH/a15/` (16 MB).

### 8.2 Pre-push check (addendum 2 section 5)
Run on commit f68d3f1 against `origin/v2.3-tiles` before pushing: forbidden tracked files (`papers/`, `*.pdf`,
`working-with-steve.md`, `cry_for_help.txt`, `.env`, `.netrc`): **none**; files over 50 MB in the new commit: **none** (the
largest are this report, 90 kB, and `data/manifest.csv`, 77 kB); secret patterns in the diff: **none**. The `.docx`
manuscripts and the M&D PDFs stay untracked.

### 8.3 Generated tables (`make_a15_tables.py`, from the run directory; also `appendix_tables.md` there)

### B1. M&D (1989) Fig. 3: extraction, placement and ground control (section 2)

| item | value |
|---|---|
| PDF page | 3 |
| page pixels | 2455 x 3379 at 300 dpi (1-bit CCITT scan) |
| crop (page px: left, top, right, bottom) | [330, 250, 2120, 1960] |
| crop pixels | 1790 x 1710 |
| scale bar | 500 m = 398 px -> 1.2563 m/px (1:14,838) |
| thick-line components / network components | 1095 / 201 |


Coarse placement (channel-network cross-correlation at 5 m):

| rank | ncc | scale | rotation (deg) | top-left of crop (E, N) | second peak ncc | second peak (E, N) |
|---|---|---|---|---|---|---|
| 1 | 0.214 | 1.000 | 0 | 538950, 4191715 | 0.161 | 541055, 4188100 |
| 2 | 0.196 | 1.000 | -1.500 | 538950, 4191715 | 0.160 | 541045, 4188125 |
| 3 | 0.192 | 0.950 | -1.500 | 541090, 4188075 | 0.175 | 539010, 4191655 |
| 4 | 0.185 | 0.950 | -3 | 541075, 4188075 | 0.157 | 539010, 4191645 |
| 5 | 0.184 | 0.950 | 0 | 539020, 4191665 | 0.183 | 541100, 4188060 |


Local refinement: 248 patches, 38 good matches, 33 used; patch affine RMS 12.0 m, max 26.1 m; pixel size 1.2339 x 1.2668 m; rotation -0.13 / 0.37 deg.


Ground control points (figure pixel -> EPSG:6339) and residuals of the affine fit:

| id | feature | figure (row, col) | E | N | residual (m) |
|---|---|---|---|---|---|
| J1 | channel confluence (figure junction; D8 confluence on the 1 m grid) | 1305, 1155 | 540,390 | 4,190,064 | 2.208 |
| J2 | channel confluence (figure junction; D8 confluence on the 1 m grid) | 824, 1590 | 540,928 | 4,190,672 | 2.876 |
| J3 | channel confluence (figure junction; D8 confluence on the 1 m grid) | 1057, 911 | 540,094 | 4,190,384 | 5.369 |
| J4 | channel confluence (figure junction; D8 confluence on the 1 m grid) | 818, 791 | 539,954 | 4,190,688 | 4.214 |
| J6 | channel confluence (figure junction; D8 confluence on the 1 m grid) | 1018, 513 | 539,592 | 4,190,434 | 12.939 |
| J7 | channel confluence (figure junction; D8 confluence on the 1 m grid) | 708, 614 | 539,740 | 4,190,820 | 8.824 |
| J8 | channel confluence (figure junction; D8 confluence on the 1 m grid) | 656, 533 | 539,632 | 4,190,908 | 15.731 |
| J9 | channel confluence (figure junction; D8 confluence on the 1 m grid) | 978, 384 | 539,444 | 4,190,476 | 9.026 |
| J10 | channel confluence (figure junction; D8 confluence on the 1 m grid) | 270, 1072 | 540,304 | 4,191,376 | 4.821 |
| J11 | channel confluence (figure junction; D8 confluence on the 1 m grid) | 408, 1227 | 540,482 | 4,191,202 | 9.513 |
| J12 | channel confluence (figure junction; D8 confluence on the 1 m grid) | 1223, 870 | 540,040 | 4,190,162 | 6.664 |
| J13 | channel confluence (figure junction; D8 confluence on the 1 m grid) | 716, 1224 | 540,484 | 4,190,816 | 4.181 |
| P | man-made pond (solid black symbol; flat-floored basin with a dam on the 1 m hillshade) | 825, 815 | 539,998 | 4,190,672 | 19.339 |


Affine (col, row, 1) -> (E, N): E = 1.227757 col + -0.015375 row + 538991.126; N = -0.006908 col + -1.274705 row + 4191733.188. Pixel size 1.2278 x 1.2748 m (scale bar 1.2563); rotation -0.32 / 0.69 deg. Rule 2.1: RMS 9.5 m, max 19.3 m, n = 13: **pass**. Figure junctions 1225, D8 confluences 941, unambiguous pairs 121.


### B2. Digitised layers (section 2.4) and rule 2.2

| layer | content |
|---|---|
| study_area | area 2.058 km2 (raster 2.058), 116 vertices (106 snapped to dashes); z on the 1 m grid min 36.9, p01 48.9, p99 285.2, max 310.1 m |
| channels | 9500 node-to-node lines, 31.1 km in 67 connected trees (paper: 9.1 km) |
| channel_heads | 114 skeleton end points; 1 exits at the boundary; 77 upstream end points; 61 excluding discontinuous-reach continuations (paper: 68 + 1 culvert; 5 trail heads excluded) |
| labels placed | A (540299, 4190731), B (540548, 4190418), C (540225, 4189994), D (540302, 4191297), culvert (539622, 4190782); culvert triangle at 539627, 4190738 |
| colluvial_deposits | 110 polygons, 0.202 km2 (stipple-dot density > 0.03 in a 21 px window) |
| ground_control | the GCP table above |
| file | /home/groups/hilley/lemcvt_results/20261118T160326Z_a15_stage_b/md_georef/md1989_fig3.gpkg |


Rule 2.2: area 2.058 km2 in [1.785, 2.415]: pass; z_min 36.9 m in [46, 76]: fail; z_max 310.1 m in [299, 329]: pass. **FAIL**


### B3. Re-routing with the gap rule (section 3), from grid_<site>/<site>_gaps.json and <site>_route.json

| site | grid | pad | valid before | nodata comps | enclosed comps | enclosed cells | enclosed km2 | enclosed size p50/p90/p99/max (cells) | fill distance p50/p90/p99/max (m) | enclosed >= 1e4 cells | valid after | outlets | largest basin km2 | budget closed | s | peak GB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM | 7802 x 12639 | 2 | 39,375,330 | 1 | 0 | 0 | 0.000 | - / - / - / - | - / - / - / - | - | 39,375,330 | 39,238 | 15.012 | yes | 56.016 | 5.239 |
| AP | 25157 x 35600 | 0 | 576,968,635 | 1,051 | 1,050 | 700,025 | 0.700 | 28.000 / 1354.1 / 6656.3 / 158,941 | 2.000 / 10.770 / 28.684 / 179.7 | 5 | 577,668,660 | 149,881 | 514.3 | yes | 865.3 | 46.773 |
| TTV | 19010 x 19010 | 0 | 140,807,242 | 526 | 525 | 492,163 | 0.492 | 17.000 / 408.6 / 6179.9 / 207,788 | 1.414 / 7.071 / 28.311 / 106.2 | 3 | 141,299,405 | 118,919 | 22.647 | yes | 212.7 | 18.544 |


GM largest outlets: (697276, 3980644) 15.01 km2 (38.1 %); (693138, 3977282) 14.73 km2 (37.4 %); (697712, 3976464) 2.75 km2 (7.0 %); (703146, 3980890) 2.24 km2 (5.7 %); (696174, 3979718) 0.63 km2 (1.6 %)


AP largest outlets: (581956, 4425296) 514.29 km2 (89.0 %); (553766, 4424300) 1.48 km2 (0.3 %); (581978, 4425332) 0.95 km2 (0.2 %); (576922, 4427754) 0.90 km2 (0.2 %); (580004, 4417604) 0.85 km2 (0.1 %)


TTV largest outlets: (537128, 4190296) 22.65 km2 (16.0 %); (542120, 4193754) 18.52 km2 (13.1 %); (540482, 4187230) 11.43 km2 (8.1 %); (542490, 4193080) 9.64 km2 (6.8 %); (539370, 4188314) 6.45 km2 (4.6 %)


GM budget diagnostic (padded run): receivers higher 0, two-cycles 0, receivers equal in the float32 product 586822, root sum 39,375,330 of 39,375,330.

GM first pass (unpadded, superseded): root sum 38,883,247 of 39,375,330; receivers higher 69 (all in the last two columns), two-cycles 134.


### B4. TTV reference from the M&D polygon on the 10 m NED (decision 4), against the stage A proxy

| quantity | M&D polygon (a15) | proxy catchment (stage A) |
|---|---|---|
| theta (legacy chi) | 0.437 | 0.387 |
| k_s at theta, median of basins | 4.024 | 15.416 |
| k_s pooled | 22.737 | - |
| basins in the k_s fit | 33 | - |
| |C_ht| (1/m) | 0.017 | 0.015 |
| convex divide pixels | 2,258 | - |
| cells / km2 | 20578 / 2.058 | 6,729 / 0.673 |
| z range (m) | 38.3-309.3 | 61-317 |
| largest A inside (km2) | 1.218 | 0.67 |
| chi R2 (area-weighted) at theta | - | - |
| dBIC (area-weighted) | - | - |


Whole-grid k_s at the polygon theta: median 13.66, pooled 20.97 (932 basins).


### B5. Eligibility under the settled rules (section 4), from draw/<site>_draw.json

| site | candidates | pass set B screens | within +-25 % | screens and +-25 % | gap-excluded (25 %) | eligible at +-25 % | within +-50 % | screens and +-50 % | gap-excluded (50 %) | eligible at +-50 % | widening threshold | tolerance used | eligible | per offset |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM | 71 | 38 | 23 | 14 | 0 | 14 | 58 | 33 | 0 | 33 | 25 | +-50 % | 33 | 0: 7, 1: 9, 2: 7, 3: 10 |
| AP | 1,653 | 320 | 651 | 68 | 14 | 54 | 1,460 | 276 | 77 | 199 | 25 | +-25 % | 54 | 0: 14, 1: 17, 2: 11, 3: 12 |
| TTV | 420 | 100 | 3 | 1 | 0 | 1 | 16 | 1 | 0 | 1 | 30 | +-50 % | 1 | 0: 0, 1: 0, 2: 0, 3: 1 |


GM windows removed by each set B screen (alone): trunk: A_max_inside <= 5e+06 m2: 23; trunk: largest single edge inflow <= 2e+06 m2: 31; transient: chi-z R2 (area-weighted) >= 0.90: 2; transient: slope-area dBIC (area-weighted) <= 20: 4; disturbance: NLCD classes [11, 22, 23, 24, 31] fraction <= 0.10: 0; upstream area fully in valid data (decision 7): 2.


AP windows removed by each set B screen (alone): trunk: A_max_inside <= 5e+06 m2: 587; trunk: largest single edge inflow <= 2e+06 m2: 771; transient: chi-z R2 (area-weighted) >= 0.90: 279; transient: slope-area dBIC (area-weighted) <= 20: 343; disturbance: NLCD classes [11, 22, 23, 24, 31] fraction <= 0.10: 69; disturbance (AP): longwall_panels fraction <= 0.00: 854.


TTV windows removed by each set B screen (alone): trunk: A_max_inside <= 5e+06 m2: 62; trunk: largest single edge inflow <= 2e+06 m2: 90; transient: chi-z R2 (area-weighted) >= 0.90: 77; transient: slope-area dBIC (area-weighted) <= 20: 149; disturbance: NLCD classes [11, 22, 23, 24, 31] fraction <= 0.10: 167; upstream area fully in valid data (decision 7): 48.


### B6. The original ten tiles under the settled rules (sections 1.9 and 4.3), from stage A pools/<site>_pools.json

| tile | trunk inside (km2) | largest inflow (km2) | chi R2 (w) | dBIC (w) | NLCD (11, 22-24, 31) | longwall | upstream valid | set B screens failed | theta (reported) | k_s dev | |C_ht| dev | within +-25 % | within +-50 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM1 | 1.146 | 0.895 | 0.995 | 2.541 | 0.000 | - | yes | none | 0.362 | +23 % | +34 % | no | yes |
| GM2 | 8.880 | 8.471 | 0.982 | -4.844 | 0.000 | - | yes | trunk <= 5 km2, inflow <= 2 km2 | 0.367 | +11 % | +9 % | yes | yes |
| GM3 | 0.344 | 2.100e-03 | 0.983 | -5.346 | 0.000 | - | yes | none | 0.262 | +8 % | -6 % | yes | yes |
| GM4 | 6.198 | 5.159 | 0.990 | 6.678 | 0.000 | - | no | trunk <= 5 km2, inflow <= 2 km2, upstream valid | 0.321 | +88 % | +18 % | no | no |
| GM5 | 4.729 | 2.656 | 0.980 | -9.251 | 0.000 | - | yes | inflow <= 2 km2 | 0.312 | +16 % | +33 % | no | yes |
| AP1 | 1.065 | 0.115 | 0.994 | 49.016 | 8.264e-03 | 0.813 | - | dBIC <= 20, longwall = 0 | 0.621 | -31 % | -14 % | no | yes |
| AP2 | 1.667 | 0.994 | 0.983 | 46.582 | 1.837e-03 | 0.743 | - | dBIC <= 20, longwall = 0 | 0.780 | -15 % | -19 % | yes | yes |
| AP3 | 10.655 | 8.702 | 0.990 | -10.453 | 1.837e-03 | 0.088 | - | trunk <= 5 km2, inflow <= 2 km2, longwall = 0 | 0.512 | +14 % | -21 % | yes | yes |
| AP4 | 16.637 | 16.572 | 0.980 | 14.065 | 7.346e-03 | 0.000 | - | trunk <= 5 km2, inflow <= 2 km2 | 0.692 | -22 % | +0 % | yes | yes |
| AP5 | 1.228 | 0.596 | 0.993 | 10.571 | 2.755e-03 | 0.855 | - | longwall = 0 | 0.433 | -30 % | +6 % | no | yes |


NLCD and longwall fractions recomputed over the tiles' coarse cells from the stage A rasters (pools/<site>_nlcd2021.tif, AP_padep_longwall_panels.tif).


### B7-GM. The draw at GM (section 5): seed 20261118, 33 eligible windows, target 25, accepted 2, skipped 31


Pre-placed: GM1 (original) SW 700991, 3980451; GM2 (original) SW 698991, 3980451; GM3 (original) SW 695991, 3978451; GM4 (original) SW 694991, 3976451; GM5 (original) SW 699991, 3981451


Within-offset direct-edge check against stage A: offset 0: stage A 50, here 50, equal True; offset 1: stage A 58, here 58, equal True; offset 2: stage A 53, here 53, equal True; offset 3: stage A 52, here 52, equal True


Ordered list (random permutation of the eligible windows sorted by offset and window id):

| rank | offset | window | footprint SW (E, N) | k_s dev | |C_ht| dev | theta | result |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 447 | 697481, 3978961 | +32 % | +14 % | 0.400 | accepted |
| 2 | 2 | 447 | 696981, 3978961 | +32 % | +25 % | 0.550 | skipped: overlaps pre-placed GM3 |
| 3 | 3 | 446 | 696481, 3978961 | +13 % | +13 % | 0.429 | skipped: overlaps pre-placed GM3 |
| 4 | 0 | 468 | 695981, 3978461 | +11 % | -7 % | 0.258 | skipped: overlaps pre-placed GM3 |
| 5 | 0 | 469 | 696981, 3978461 | +17 % | +32 % | 0.371 | skipped: overlaps pre-placed GM3 |
| 6 | 3 | 406 | 700481, 3980961 | +20 % | +2 % | 0.358 | skipped: overlaps pre-placed GM1 |
| 7 | 1 | 491 | 697481, 3977461 | +12 % | -29 % | 0.454 | skipped: drains-through edge with accepted rank 1 (offset 3 window 447) |
| 8 | 1 | 468 | 696481, 3978461 | +13 % | +19 % | 0.367 | skipped: overlaps pre-placed GM3 |
| 9 | 2 | 406 | 699981, 3980961 | +29 % | +8 % | 0.325 | skipped: overlaps pre-placed GM2 |
| 10 | 0 | 428 | 699981, 3980461 | +31 % | +14 % | 0.367 | skipped: overlaps pre-placed GM2 |
| 11 | 1 | 427 | 699481, 3980461 | +22 % | +30 % | 0.346 | skipped: overlaps pre-placed GM2 |
| 12 | 1 | 469 | 697481, 3978461 | +19 % | +2 % | 0.417 | skipped: overlaps accepted rank 1 (offset 3 window 447) |
| 13 | 0 | 512 | 695981, 3976461 | +47 % | +4 % | 0.462 | skipped: overlaps pre-placed GM4 |
| 14 | 3 | 468 | 696481, 3977961 | +24 % | +21 % | 0.400 | skipped: overlaps pre-placed GM3 |
| 15 | 3 | 490 | 696481, 3976961 | +47 % | -12 % | 0.433 | skipped: drains-through edge with pre-placed GM4 |
| 16 | 0 | 429 | 700981, 3980461 | +28 % | +32 % | 0.350 | skipped: overlaps pre-placed GM1 |
| 17 | 2 | 427 | 698981, 3979961 | +3 % | -10 % | 0.246 | skipped: overlaps pre-placed GM2 |
| 18 | 2 | 428 | 699981, 3979961 | +7 % | +22 % | 0.317 | skipped: overlaps pre-placed GM2 |
| 19 | 3 | 407 | 701481, 3980961 | +27 % | +30 % | 0.383 | skipped: overlaps pre-placed GM1 |
| 20 | 1 | 512 | 696481, 3976461 | +12 % | -17 % | 0.492 | skipped: drains-through edge with pre-placed GM4 |
| 21 | 2 | 470 | 697981, 3977961 | -7 % | -31 % | 0.467 | accepted |
| 22 | 3 | 428 | 700481, 3979961 | +25 % | +40 % | 0.246 | skipped: overlaps pre-placed GM1 |
| 23 | 3 | 469 | 697481, 3977961 | +5 % | -31 % | 0.446 | skipped: overlaps accepted rank 21 (offset 2 window 470) |
| 24 | 3 | 470 | 698481, 3977961 | -1 % | -23 % | 0.388 | skipped: overlaps accepted rank 21 (offset 2 window 470) |
| 25 | 0 | 471 | 698981, 3978461 | +19 % | -23 % | 0.529 | skipped: drains-through edge with accepted rank 1 (offset 3 window 447) |
| 26 | 1 | 408 | 702481, 3981461 | +34 % | +5 % | 0.500 | skipped: drains-through edge with pre-placed GM1 |
| 27 | 1 | 406 | 700481, 3981461 | +20 % | +26 % | 0.425 | skipped: overlaps pre-placed GM5 |
| 28 | 3 | 427 | 699481, 3979961 | +16 % | +15 % | 0.292 | skipped: overlaps pre-placed GM2 |
| 29 | 2 | 491 | 696981, 3976961 | +11 % | -10 % | 0.438 | skipped: drains-through edge with accepted rank 21 (offset 2 window 470) |
| 30 | 2 | 407 | 700981, 3980961 | +32 % | +15 % | 0.388 | skipped: overlaps pre-placed GM1 |
| 31 | 1 | 428 | 700481, 3980461 | +29 % | +11 % | 0.350 | skipped: overlaps pre-placed GM1 |
| 32 | 0 | 470 | 697981, 3978461 | +3 % | -16 % | 0.467 | skipped: overlaps accepted rank 1 (offset 3 window 447) |
| 33 | 1 | 407 | 701481, 3981461 | +44 % | +4 % | 0.437 | skipped: drains-through edge with pre-placed GM1 |


Skip reasons: overlaps pre-placed GM3: 6; overlaps pre-placed GM1: 6; drains-through edge with: 7; overlaps pre-placed GM2: 6; overlaps accepted rank: 4; overlaps pre-placed GM4: 1; overlaps pre-placed GM5: 1


### B7-AP. The draw at AP (section 5): seed 20261118, 54 eligible windows, target 25, accepted 24, skipped 30


Pre-placed: AP1 (original) SW 568503, 4408340; AP2 (original) SW 561573, 4426160; AP3 (original) SW 551854, 4419230; AP4 (original) SW 555093, 4419230; AP5 (original) SW 556893, 4422470


Within-offset direct-edge check against stage A: offset 0: stage A 1564, here 1564, equal True; offset 1: stage A 1505, here 1505, equal True; offset 2: stage A 1548, here 1548, equal True; offset 3: stage A 1546, here 1546, equal True


Ordered list (random permutation of the eligible windows sorted by offset and window id):

| rank | offset | window | footprint SW (E, N) | k_s dev | |C_ht| dev | theta | result |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 1,433 | 566493, 4407378 | -23 % | +24 % | 0.760 | accepted |
| 2 | 1 | 720 | 556083, 4420246 | -20 % | +2 % | 0.537 | accepted |
| 3 | 2 | 663 | 552633, 4420755 | -21 % | -6 % | 0.596 | accepted |
| 4 | 2 | 789 | 570452, 4418776 | +11 % | -22 % | 0.800 | accepted |
| 5 | 0 | 1,425 | 558573, 4407378 | -15 % | -2 % | 0.780 | accepted |
| 6 | 0 | 718 | 553623, 4420246 | -8 % | -14 % | 0.800 | skipped: overlaps accepted rank 3 (offset 2 window 663) |
| 7 | 2 | 1,106 | 563523, 4412837 | -24 % | -9 % | 0.633 | accepted |
| 8 | 1 | 1,101 | 559053, 4413317 | +1 % | -3 % | 0.208 | accepted |
| 9 | 0 | 1,260 | 555603, 4410347 | -17 % | -6 % | 0.492 | accepted |
| 10 | 3 | 676 | 565983, 4420755 | -6 % | -24 % | 0.108 | accepted |
| 11 | 1 | 1,378 | 565983, 4408368 | -20 % | +3 % | 0.717 | skipped: overlaps accepted rank 1 (offset 0 window 1433) |
| 12 | 2 | 679 | 568472, 4420755 | -6 % | -24 % | 0.208 | accepted |
| 13 | 1 | 1,105 | 563013, 4413317 | -6 % | -8 % | 0.800 | skipped: overlaps accepted rank 7 (offset 2 window 1106) |
| 14 | 1 | 1,106 | 564003, 4413317 | +11 % | -15 % | 0.208 | skipped: overlaps accepted rank 7 (offset 2 window 1106) |
| 15 | 3 | 780 | 562023, 4418776 | +12 % | +6 % | 0.717 | accepted |
| 16 | 3 | 627 | 570932, 4421745 | -2 % | -16 % | 0.458 | accepted |
| 17 | 3 | 663 | 553113, 4420755 | -20 % | -4 % | 0.587 | skipped: overlaps accepted rank 3 (offset 2 window 663) |
| 18 | 1 | 1,099 | 557073, 4413317 | -11 % | +3 % | 0.254 | skipped: drains-through edge with accepted rank 8 (offset 1 window 1101) |
| 19 | 0 | 665 | 554613, 4421235 | -24 % | -4 % | 0.800 | skipped: drains-through edge with accepted rank 2 (offset 1 window 720) |
| 20 | 1 | 834 | 562023, 4418266 | -13 % | -1 % | 0.450 | skipped: overlaps accepted rank 15 (offset 3 window 780) |
| 21 | 3 | 415 | 574892, 4425705 | +6 % | -22 % | 0.383 | accepted |
| 22 | 1 | 512 | 564003, 4424205 | -16 % | -22 % | 0.788 | accepted |
| 23 | 2 | 460 | 565503, 4424715 | +8 % | -18 % | 0.100 | accepted |
| 24 | 1 | 938 | 558063, 4416286 | -24 % | +14 % | 0.658 | accepted |
| 25 | 3 | 720 | 556083, 4419766 | -21 % | +24 % | 0.400 | skipped: overlaps pre-placed AP4 |
| 26 | 1 | 1,424 | 558063, 4407378 | -20 % | -1 % | 0.775 | skipped: overlaps accepted rank 5 (offset 0 window 1425) |
| 27 | 1 | 1,043 | 555093, 4414307 | -11 % | +19 % | 0.479 | accepted |
| 28 | 2 | 780 | 561543, 4418776 | +6 % | -3 % | 0.467 | skipped: overlaps accepted rank 15 (offset 3 window 780) |
| 29 | 0 | 469 | 574412, 4425195 | +9 % | -24 % | 0.596 | skipped: overlaps accepted rank 21 (offset 3 window 415) |
| 30 | 1 | 1,367 | 555093, 4408368 | -8 % | +13 % | 0.772 | accepted |
| 31 | 2 | 719 | 554613, 4419766 | -20 % | -1 % | 0.558 | skipped: overlaps pre-placed AP4 |
| 32 | 0 | 1,379 | 566493, 4408368 | -21 % | -2 % | 0.621 | skipped: overlaps accepted rank 1 (offset 0 window 1433) |
| 33 | 0 | 569 | 566493, 4423215 | +9 % | -24 % | 0.646 | accepted |
| 34 | 0 | 941 | 560553, 4416286 | -15 % | +8 % | 0.538 | accepted |
| 35 | 1 | 454 | 560043, 4425195 | -15 % | -13 % | 0.798 | accepted |
| 36 | 1 | 1,425 | 559053, 4407378 | -10 % | +11 % | 0.683 | skipped: overlaps accepted rank 5 (offset 0 window 1425) |
| 37 | 2 | 1,098 | 555603, 4412837 | -18 % | +14 % | 0.392 | accepted |
| 38 | 3 | 1,425 | 559053, 4406898 | -13 % | -5 % | 0.571 | skipped: overlaps accepted rank 5 (offset 0 window 1425) |
| 39 | 3 | 903 | 576872, 4416796 | +13 % | -10 % | 0.733 | accepted |
| 40 | 0 | 889 | 562533, 4417276 | -22 % | -15 % | 0.417 | accepted |
| 41 | 1 | 469 | 574892, 4425195 | -7 % | -22 % | 0.563 | skipped: overlaps accepted rank 21 (offset 3 window 415) |
| 42 | 3 | 666 | 556083, 4420755 | +6 % | +7 % | 0.483 | skipped: overlaps accepted rank 2 (offset 1 window 720) |
| 43 | 3 | 460 | 565983, 4424715 | -1 % | -18 % | 0.151 | skipped: overlaps accepted rank 23 (offset 2 window 460) |
| 44 | 0 | 1,107 | 564513, 4413317 | +24 % | -21 % | 0.638 | skipped: overlaps accepted rank 7 (offset 2 window 1106) |
| 45 | 3 | 1,099 | 557073, 4412837 | +2 % | -3 % | 0.638 | skipped: drains-through edge with accepted rank 27 (offset 1 window 1043) |
| 46 | 3 | 1,106 | 564003, 4412837 | +16 % | -6 % | 0.567 | skipped: overlaps accepted rank 7 (offset 2 window 1106) |
| 47 | 2 | 889 | 562533, 4416796 | -4 % | -13 % | 0.563 | skipped: overlaps accepted rank 40 (offset 0 window 889) |
| 48 | 2 | 1,379 | 566493, 4407888 | -22 % | +4 % | 0.646 | skipped: overlaps accepted rank 1 (offset 0 window 1433) |
| 49 | 2 | 1,100 | 557583, 4412837 | +2 % | -8 % | 0.571 | skipped: drains-through edge with accepted rank 37 (offset 2 window 1098) |
| 50 | 1 | 1,097 | 555093, 4413317 | -22 % | +13 % | 0.462 | skipped: overlaps accepted rank 27 (offset 1 window 1043) |
| 51 | 1 | 569 | 566973, 4423215 | +1 % | -23 % | 0.612 | skipped: overlaps accepted rank 33 (offset 0 window 569) |
| 52 | 0 | 1,422 | 555603, 4407378 | -21 % | +24 % | 0.550 | skipped: overlaps accepted rank 30 (offset 1 window 1367) |
| 53 | 0 | 720 | 555603, 4420246 | -1 % | +8 % | 0.567 | skipped: overlaps accepted rank 2 (offset 1 window 720) |
| 54 | 0 | 1,152 | 555603, 4412327 | -14 % | +20 % | 0.633 | skipped: overlaps accepted rank 37 (offset 2 window 1098) |


Skip reasons: overlaps accepted rank: 24; drains-through edge with: 4; overlaps pre-placed AP4: 2


### B7-TTV. The draw at TTV (section 5): seed 20261118, 1 eligible windows, target 30, accepted 1, skipped 0

M&D windows (5.2): polygon 2.058 km2 on the 10 m lattice, 28 candidate windows intersect it; chosen 3 non-overlapping windows covering 92.9 % (>= 80 %: pass).

| window (offset, id) | footprint SW (E, N) | set B screens pass | screens failed | within +-25 % | within +-50 % | k_s dev | |C_ht| dev | theta dev | gap-excluded (channel cells in filled gaps) |
|---|---|---|---|---|---|---|---|---|---|
| 0, 135 | 539000, 4190000 | no | transient: slope-area dBIC (area-weighted) <= 20 | no | no | +612 % | -1 % | +0.00 | False (0) |
| 2, 118 | 540000, 4190500 | no | transient: slope-area dBIC (area-weighted) <= 20 | no | no | +360 % | +7 % | +0.02 | False (0) |
| 2, 136 | 540000, 4189500 | yes | none | no | no | +515 % | -22 % | +0.02 | False (0) |


Pre-placed: TTV1 (M&D) SW 539000, 4190000; TTV2 (M&D) SW 540000, 4190500; TTV3 (M&D) SW 540000, 4189500


Within-offset direct-edge check against stage A: offset 0: stage A 390, here 390, equal True; offset 1: stage A 382, here 382, equal True; offset 2: stage A 357, here 357, equal True; offset 3: stage A 339, here 339, equal True


Ordered list (random permutation of the eligible windows sorted by offset and window id):

| rank | offset | window | footprint SW (E, N) | k_s dev | |C_ht| dev | theta | result |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 148 | 542500, 4188500 | +6 % | -10 % | 0.196 | accepted |


Skip reasons: 


### B8. Tiles cut (section 6): per-tile unit check and products, from tiles/tiles.json

| tile | kind | version | SW corner (E, N) | EPSG | valid frac | z median | z min | z max | unit check | A max (km2) | outlet cells | filled-gap cells | filled-gap channel cells | class-2 pts/m2 | NN-filled frac | stage A gap frac |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM1 | original | GM-x | 700991, 3980451 | 32,610 | 1.000 | 357.7 | 295.2 | 424.7 | yes | 1.148 | 0 | 0 | 0 | - | - | - |
| GM2 | original | GM-x | 698991, 3980451 | 32,610 | 1.000 | 308.6 | 246.3 | 370.7 | yes | 4.804 | 0 | 0 | 0 | - | - | - |
| GM3 | original | GM-x | 695991, 3978451 | 32,610 | 1.000 | 273.3 | 216.3 | 304.5 | yes | 0.337 | 0 | 0 | 0 | - | - | - |
| GM4 | original | GM-x | 694991, 3976451 | 32,610 | 1.000 | 211.8 | 134.9 | 273.4 | yes | 6.186 | 0 | 0 | 0 | - | - | - |
| GM5 | original | GM-x | 699991, 3981451 | 32,610 | 1.000 | 340.1 | 282.4 | 419.9 | yes | 2.732 | 0 | 0 | 0 | - | - | - |
| GM6 | drawn | GM-x | 697481, 3978961 | 32,610 | 1.000 | 300.5 | 210.1 | 342.1 | yes | 1.483 | 0 | 0 | 0 | - | - | - |
| GM7 | drawn | GM-x | 697981, 3977961 | 32,610 | 1.000 | 313.3 | 263.1 | 348.1 | yes | 1.185 | 0 | 0 | 0 | - | - | - |
| AP1_3dep | original | AP-3DEP-x | 568503, 4408340 | 6,346 | 1.000 | 386.3 | 324.5 | 440.5 | yes | 1.175 | 0 | 0 | 0 | 3.238 | 0.016 | 0.000 |
| AP2_3dep | original | AP-3DEP-x | 561573, 4426160 | 6,346 | 1.000 | 374.2 | 320.6 | 463.7 | yes | 1.685 | 0 | 0 | 0 | 3.048 | 0.028 | 0.000 |
| AP3_3dep | original | AP-3DEP-x | 551854, 4419230 | 6,346 | 1.000 | 378.9 | 328.0 | 453.0 | yes | 11.072 | 0 | 0 | 0 | 3.261 | 0.019 | 0.000 |
| AP4_3dep | original | AP-3DEP-x | 555093, 4419230 | 6,346 | 1.000 | 387.7 | 317.4 | 443.7 | yes | 17.143 | 0 | 0 | 0 | 3.070 | 0.016 | 0.000 |
| AP5_3dep | original | AP-3DEP-x | 556893, 4422470 | 6,346 | 1.000 | 381.7 | 330.5 | 424.9 | yes | 1.167 | 0 | 0 | 0 | 2.880 | 0.030 | 0.000 |
| AP6 | drawn | AP-3DEP-x | 566493, 4407378 | 6,346 | 1.000 | 381.9 | 327.0 | 450.6 | yes | 1.493 | 0 | 0 | 0 | 2.806 | 0.028 | 0.000 |
| AP7 | drawn | AP-3DEP-x | 556083, 4420246 | 6,346 | 1.000 | 364.3 | 322.9 | 438.3 | yes | 3.647 | 0 | 0 | 0 | 2.903 | 0.015 | 0.000 |
| AP8 | drawn | AP-3DEP-x | 552633, 4420755 | 6,346 | 1.000 | 385.5 | 339.0 | 457.9 | yes | 1.625 | 0 | 0 | 0 | 2.888 | 0.021 | 0.000 |
| AP9 | drawn | AP-3DEP-x | 570452, 4418776 | 6,346 | 1.000 | 364.4 | 311.5 | 430.2 | yes | 1.518 | 0 | 0 | 0 | 2.846 | 0.037 | 0.000 |
| AP10 | drawn | AP-3DEP-x | 558573, 4407378 | 6,346 | 1.000 | 404.4 | 348.1 | 480.5 | yes | 0.425 | 0 | 0 | 0 | 2.726 | 0.020 | 0.000 |
| AP11 | drawn | AP-3DEP-x | 563523, 4412837 | 6,346 | 1.000 | 364.2 | 309.4 | 441.8 | yes | 1.958 | 0 | 0 | 0 | 3.148 | 0.023 | 0.000 |
| AP12 | drawn | AP-3DEP-x | 559053, 4413317 | 6,346 | 1.000 | 374.9 | 315.6 | 441.5 | yes | 1.248 | 0 | 3,736 | 0 | 3.063 | 0.023 | 3.736e-03 |
| AP13 | drawn | AP-3DEP-x | 555603, 4410347 | 6,346 | 1.000 | 364.8 | 322.4 | 428.7 | yes | 3.967 | 0 | 0 | 0 | 3.068 | 0.027 | 0.000 |
| AP14 | drawn | AP-3DEP-x | 565983, 4420755 | 6,346 | 1.000 | 391.7 | 324.3 | 438.1 | yes | 0.555 | 0 | 0 | 0 | 2.813 | 0.021 | 0.000 |
| AP15 | drawn | AP-3DEP-x | 568472, 4420755 | 6,346 | 1.000 | 378.9 | 329.8 | 432.3 | yes | 2.075 | 0 | 0 | 0 | 3.029 | 0.040 | 0.000 |
| AP16 | drawn | AP-3DEP-x | 562023, 4418776 | 6,346 | 1.000 | 356.2 | 301.3 | 430.5 | yes | 1.692 | 0 | 409 | 0 | 2.825 | 0.017 | 4.090e-04 |
| AP17 | drawn | AP-3DEP-x | 570932, 4421745 | 6,346 | 1.000 | 348.1 | 295.1 | 406.2 | yes | 2.321 | 0 | 0 | 0 | 2.922 | 0.030 | 0.000 |
| AP18 | drawn | AP-3DEP-x | 574892, 4425705 | 6,346 | 1.000 | 362.8 | 320.9 | 416.5 | yes | 1.982 | 0 | 13 | 0 | 3.142 | 0.025 | 1.300e-05 |
| AP19 | drawn | AP-3DEP-x | 564003, 4424205 | 6,346 | 1.000 | 379.8 | 318.7 | 442.9 | yes | 2.109 | 0 | 0 | 0 | 2.880 | 0.020 | 0.000 |
| AP20 | drawn | AP-3DEP-x | 565503, 4424715 | 6,346 | 1.000 | 409.3 | 357.1 | 457.3 | yes | 0.338 | 0 | 0 | 0 | 3.075 | 0.028 | 0.000 |
| AP21 | drawn | AP-3DEP-x | 558063, 4416286 | 6,346 | 1.000 | 382.8 | 326.6 | 432.5 | yes | 0.421 | 0 | 0 | 0 | 2.907 | 0.018 | 0.000 |
| AP22 | drawn | AP-3DEP-x | 555093, 4414307 | 6,346 | 1.000 | 394.2 | 327.8 | 455.4 | yes | 0.395 | 0 | 0 | 0 | 3.022 | 0.025 | 0.000 |
| AP23 | drawn | AP-3DEP-x | 555093, 4408368 | 6,346 | 1.000 | 380.4 | 334.8 | 443.2 | yes | 1.724 | 0 | 0 | 0 | 2.854 | 0.029 | 0.000 |
| AP24 | drawn | AP-3DEP-x | 566493, 4423215 | 6,346 | 1.000 | 390.3 | 326.9 | 458.2 | yes | 1.600 | 0 | 0 | 0 | 2.930 | 0.028 | 0.000 |
| AP25 | drawn | AP-3DEP-x | 560553, 4416286 | 6,346 | 1.000 | 362.6 | 313.4 | 420.3 | yes | 3.266 | 0 | 0 | 0 | 2.905 | 0.023 | 0.000 |
| AP26 | drawn | AP-3DEP-x | 560043, 4425195 | 6,346 | 1.000 | 397.6 | 324.0 | 442.9 | yes | 0.536 | 0 | 0 | 0 | 2.937 | 0.021 | 0.000 |
| AP27 | drawn | AP-3DEP-x | 555603, 4412837 | 6,346 | 1.000 | 386.7 | 330.4 | 436.9 | yes | 1.765 | 0 | 0 | 0 | 2.999 | 0.024 | 0.000 |
| AP28 | drawn | AP-3DEP-x | 576872, 4416796 | 6,346 | 1.000 | 356.4 | 300.6 | 414.4 | yes | 0.750 | 0 | 0 | 0 | 3.257 | 0.022 | 0.000 |
| AP29 | drawn | AP-3DEP-x | 562533, 4417276 | 6,346 | 1.000 | 376.1 | 327.5 | 422.4 | yes | 1.781 | 0 | 0 | 0 | 2.873 | 0.019 | 0.000 |


GM originals: the re-cut grids against the existing tiles, cell by cell:

| tile | same footprint | area equal frac | |dA|/A > 1 % frac | same, interior | max |dA| (m2) | old A max | new A max | new > old frac | new < old frac | fd equal frac | fd equal interior | fd 0 old | fd 0 new | z equal (1 mm) frac | max |dz| | median dz |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM1 | yes | 0.999 | 6.300e-05 | 6.225e-05 | 9633.0 | 1,157,840 | 1,148,207 | 0.000 | 7.640e-04 | 1.000 | 1.000 | 0 | 0 | 1.000 | 0.000 | 0.000 |
| GM2 | yes | 0.999 | 5.920e-04 | 5.894e-04 | 4,090,726 | 8,894,970 | 4,804,244 | 0.000 | 6.920e-04 | 1.000 | 1.000 | 0 | 0 | 1.000 | 0.000 | 0.000 |
| GM3 | yes | 1.000 | 0.000 | 0.000 | 0.000 | 336,733 | 336,733 | 0.000 | 0.000 | 1.000 | 1.000 | 0 | 0 | 1.000 | 0.000 | 0.000 |
| GM4 | yes | 0.999 | 0.000 | 0.000 | 16.000 | 6,186,105 | 6,186,089 | 0.000 | 5.080e-04 | 1.000 | 1.000 | 0 | 0 | 1.000 | 0.000 | 0.000 |
| GM5 | yes | 0.998 | 1.647e-03 | 1.647e-03 | 3,840,639 | 4,725,035 | 2,732,082 | 0.000 | 1.794e-03 | 1.000 | 1.000 | 0 | 0 | 1.000 | 0.000 | 0.000 |


Checksums (sha256) of the products:

| tile | role | path (under LEMCVT_DATA) | sha256 (first 16) | bytes |
|---|---|---|---|---|
| GM1 | filled | GM-x/GM1/GM1_1m_filled.tif | e05250b7bb8018d7... | 3,140,991 |
| GM1 | fd | GM-x/GM1/GM1_1m_fd.tif | a7b87a85d70d917d... | 144,614 |
| GM1 | area | GM-x/GM1/GM1_1m_area.tif | 1d914a402ce07f23... | 1,030,139 |
| GM1 | gapfill_mask | GM-x/GM1/GM1_1m_gapfill_mask.tif | 64ee1c2345f7534b... | 1,856 |
| GM2 | filled | GM-x/GM2/GM2_1m_filled.tif | 713291c56c0fd3a0... | 3,160,426 |
| GM2 | fd | GM-x/GM2/GM2_1m_fd.tif | 015ba555700079bf... | 141,724 |
| GM2 | area | GM-x/GM2/GM2_1m_area.tif | 7f32d8fed43ee94c... | 1,184,695 |
| GM2 | gapfill_mask | GM-x/GM2/GM2_1m_gapfill_mask.tif | 4e0f080942cc8ea7... | 1,856 |
| GM3 | filled | GM-x/GM3/GM3_1m_filled.tif | a4877da75b103299... | 3,088,533 |
| GM3 | fd | GM-x/GM3/GM3_1m_fd.tif | bc6bebfd75eb6f1b... | 175,288 |
| GM3 | area | GM-x/GM3/GM3_1m_area.tif | 8196b4e43eab2bca... | 1,169,423 |
| GM3 | gapfill_mask | GM-x/GM3/GM3_1m_gapfill_mask.tif | affc7ac875cea735... | 1,856 |
| GM4 | filled | GM-x/GM4/GM4_1m_filled.tif | cd63509522a9d50e... | 3,247,814 |
| GM4 | fd | GM-x/GM4/GM4_1m_fd.tif | febd3058e43b3b67... | 136,261 |
| GM4 | area | GM-x/GM4/GM4_1m_area.tif | c85ddc2679a7dbd4... | 1,192,917 |
| GM4 | gapfill_mask | GM-x/GM4/GM4_1m_gapfill_mask.tif | b025e8af738c2e9f... | 1,856 |
| GM5 | filled | GM-x/GM5/GM5_1m_filled.tif | 6aeff7e436f21998... | 3,138,895 |
| GM5 | fd | GM-x/GM5/GM5_1m_fd.tif | 57ac153fc735394c... | 145,406 |
| GM5 | area | GM-x/GM5/GM5_1m_area.tif | 4f531edb3b710565... | 1,243,462 |
| GM5 | gapfill_mask | GM-x/GM5/GM5_1m_gapfill_mask.tif | 061679bf5374d3bd... | 1,856 |
| GM6 | filled | GM-x/GM6/GM6_1m_filled.tif | 0eb04554d52a91c6... | 3,123,192 |
| GM6 | fd | GM-x/GM6/GM6_1m_fd.tif | 96809f424ccab28e... | 148,510 |
| GM6 | area | GM-x/GM6/GM6_1m_area.tif | 3bfdb1491052ea72... | 1,234,114 |
| GM6 | gapfill_mask | GM-x/GM6/GM6_1m_gapfill_mask.tif | 175f998b5fff53c2... | 1,856 |
| GM7 | filled | GM-x/GM7/GM7_1m_filled.tif | 2d0a4819bda89bdf... | 3,061,043 |
| GM7 | fd | GM-x/GM7/GM7_1m_fd.tif | 0cfa7d3f49c18253... | 171,116 |
| GM7 | area | GM-x/GM7/GM7_1m_area.tif | 41a5869f527dac01... | 1,264,526 |
| GM7 | gapfill_mask | GM-x/GM7/GM7_1m_gapfill_mask.tif | 5839e115605b7997... | 1,856 |
| AP1_3dep | filled | AP-3DEP-x/AP1_3dep/AP1_3dep_1m_filled.tif | c7046f4116917126... | 2,569,558 |
| AP1_3dep | fd | AP-3DEP-x/AP1_3dep/AP1_3dep_1m_fd.tif | f438c5b33f6db2c4... | 212,170 |
| AP1_3dep | area | AP-3DEP-x/AP1_3dep/AP1_3dep_1m_area.tif | 96cee58f6d30e0f6... | 1,270,027 |
| AP1_3dep | gapfill_mask | AP-3DEP-x/AP1_3dep/AP1_3dep_1m_gapfill_mask.tif | 519a9b6033df711e... | 1,866 |
| AP1_3dep | class2_count | AP-3DEP-x/AP1_3dep/AP1_3dep_1m_class2_count.tif | 91ef3b01aee8badf... | 476,099 |
| AP1_3dep | nnfill_mask | AP-3DEP-x/AP1_3dep/AP1_3dep_1m_nnfill_mask.tif | 4ff7e8d796062f3d... | 23,679 |
| AP2_3dep | filled | AP-3DEP-x/AP2_3dep/AP2_3dep_1m_filled.tif | bcf3953d1f301f48... | 2,619,611 |
| AP2_3dep | fd | AP-3DEP-x/AP2_3dep/AP2_3dep_1m_fd.tif | b83980142850928e... | 218,707 |
| AP2_3dep | area | AP-3DEP-x/AP2_3dep/AP2_3dep_1m_area.tif | 604036331e605d82... | 1,145,560 |
| AP2_3dep | gapfill_mask | AP-3DEP-x/AP2_3dep/AP2_3dep_1m_gapfill_mask.tif | f96de67b83c011de... | 1,866 |
| AP2_3dep | class2_count | AP-3DEP-x/AP2_3dep/AP2_3dep_1m_class2_count.tif | ef0dbe8f6b60c09a... | 476,334 |
| AP2_3dep | nnfill_mask | AP-3DEP-x/AP2_3dep/AP2_3dep_1m_nnfill_mask.tif | 664d95cf855e400b... | 34,110 |
| AP3_3dep | filled | AP-3DEP-x/AP3_3dep/AP3_3dep_1m_filled.tif | dabb919ac1a5d563... | 2,591,973 |
| AP3_3dep | fd | AP-3DEP-x/AP3_3dep/AP3_3dep_1m_fd.tif | 33f881b4c15dad87... | 217,293 |
| AP3_3dep | area | AP-3DEP-x/AP3_3dep/AP3_3dep_1m_area.tif | d7384a58cd99d48f... | 1,206,323 |
| AP3_3dep | gapfill_mask | AP-3DEP-x/AP3_3dep/AP3_3dep_1m_gapfill_mask.tif | d276286f80de53e0... | 1,866 |
| AP3_3dep | class2_count | AP-3DEP-x/AP3_3dep/AP3_3dep_1m_class2_count.tif | e23f09f191d1ea86... | 486,291 |
| AP3_3dep | nnfill_mask | AP-3DEP-x/AP3_3dep/AP3_3dep_1m_nnfill_mask.tif | 626ed69243bf8402... | 26,953 |
| AP4_3dep | filled | AP-3DEP-x/AP4_3dep/AP4_3dep_1m_filled.tif | 7d73f03bf054fc35... | 2,614,442 |
| AP4_3dep | fd | AP-3DEP-x/AP4_3dep/AP4_3dep_1m_fd.tif | 46d591984b4fa68d... | 215,088 |
| AP4_3dep | area | AP-3DEP-x/AP4_3dep/AP4_3dep_1m_area.tif | fd22efdbc0a5bb14... | 1,201,099 |
| AP4_3dep | gapfill_mask | AP-3DEP-x/AP4_3dep/AP4_3dep_1m_gapfill_mask.tif | a57cc16b687c7e71... | 1,866 |
| AP4_3dep | class2_count | AP-3DEP-x/AP4_3dep/AP4_3dep_1m_class2_count.tif | 658337d1722cd3ba... | 468,111 |
| AP4_3dep | nnfill_mask | AP-3DEP-x/AP4_3dep/AP4_3dep_1m_nnfill_mask.tif | cee81ab34bde3dfc... | 24,776 |
| AP5_3dep | filled | AP-3DEP-x/AP5_3dep/AP5_3dep_1m_filled.tif | dfe858efe75825ed... | 2,463,133 |
| AP5_3dep | fd | AP-3DEP-x/AP5_3dep/AP5_3dep_1m_fd.tif | 2fc84eae8eaefd48... | 218,614 |
| AP5_3dep | area | AP-3DEP-x/AP5_3dep/AP5_3dep_1m_area.tif | ad795f4c05de586c... | 1,242,531 |
| AP5_3dep | gapfill_mask | AP-3DEP-x/AP5_3dep/AP5_3dep_1m_gapfill_mask.tif | 3d30c530c55b264e... | 1,866 |
| AP5_3dep | class2_count | AP-3DEP-x/AP5_3dep/AP5_3dep_1m_class2_count.tif | a2c7d2657d63f8e3... | 456,958 |
| AP5_3dep | nnfill_mask | AP-3DEP-x/AP5_3dep/AP5_3dep_1m_nnfill_mask.tif | 7a2695e96e51a86f... | 31,298 |
| AP6 | filled | AP-3DEP-x/AP6/AP6_1m_filled.tif | f6654d364e8ef1f6... | 2,537,548 |
| AP6 | fd | AP-3DEP-x/AP6/AP6_1m_fd.tif | db1879d57298539c... | 223,747 |
| AP6 | area | AP-3DEP-x/AP6/AP6_1m_area.tif | 7d498640366bed4c... | 1,189,323 |
| AP6 | gapfill_mask | AP-3DEP-x/AP6/AP6_1m_gapfill_mask.tif | 53fbc50a0dc5330b... | 1,866 |
| AP6 | class2_count | AP-3DEP-x/AP6/AP6_1m_class2_count.tif | 98d669cee06b5b0b... | 457,670 |
| AP6 | nnfill_mask | AP-3DEP-x/AP6/AP6_1m_nnfill_mask.tif | 4c08a4205b9f31c3... | 36,145 |
| AP7 | filled | AP-3DEP-x/AP7/AP7_1m_filled.tif | e4a6801d3b5a6f22... | 2,474,215 |
| AP7 | fd | AP-3DEP-x/AP7/AP7_1m_fd.tif | 907a0da1b497e0eb... | 212,659 |
| AP7 | area | AP-3DEP-x/AP7/AP7_1m_area.tif | 654970d4da7e5d95... | 1,232,961 |
| AP7 | gapfill_mask | AP-3DEP-x/AP7/AP7_1m_gapfill_mask.tif | 185f057f9b8b44b4... | 1,866 |
| AP7 | class2_count | AP-3DEP-x/AP7/AP7_1m_class2_count.tif | 362b3eab550dd16f... | 441,902 |
| AP7 | nnfill_mask | AP-3DEP-x/AP7/AP7_1m_nnfill_mask.tif | 891dfb573772eb3c... | 21,768 |
| AP8 | filled | AP-3DEP-x/AP8/AP8_1m_filled.tif | fcfc7b3589d86988... | 2,514,489 |
| AP8 | fd | AP-3DEP-x/AP8/AP8_1m_fd.tif | fe96cd884c8a9ce7... | 219,325 |
| AP8 | area | AP-3DEP-x/AP8/AP8_1m_area.tif | 2113996b06ad5088... | 1,197,836 |
| AP8 | gapfill_mask | AP-3DEP-x/AP8/AP8_1m_gapfill_mask.tif | 4e3ed72b7b0ee2e2... | 1,866 |
| AP8 | class2_count | AP-3DEP-x/AP8/AP8_1m_class2_count.tif | b371d1820c758fa6... | 437,409 |
| AP8 | nnfill_mask | AP-3DEP-x/AP8/AP8_1m_nnfill_mask.tif | a67fc1a18fdc57ff... | 25,395 |
| AP9 | filled | AP-3DEP-x/AP9/AP9_1m_filled.tif | 6eff4e2e80fa7066... | 2,470,457 |
| AP9 | fd | AP-3DEP-x/AP9/AP9_1m_fd.tif | 284b7a19427d8f4b... | 222,671 |
| AP9 | area | AP-3DEP-x/AP9/AP9_1m_area.tif | b66f8490498a9ad2... | 1,200,160 |
| AP9 | gapfill_mask | AP-3DEP-x/AP9/AP9_1m_gapfill_mask.tif | 5a731e946a7ab456... | 1,866 |
| AP9 | class2_count | AP-3DEP-x/AP9/AP9_1m_class2_count.tif | 424b29d97c7cfc13... | 472,378 |
| AP9 | nnfill_mask | AP-3DEP-x/AP9/AP9_1m_nnfill_mask.tif | 8011f8f1bfef577d... | 40,819 |
| AP10 | filled | AP-3DEP-x/AP10/AP10_1m_filled.tif | 558da54fda8404d8... | 2,558,377 |
| AP10 | fd | AP-3DEP-x/AP10/AP10_1m_fd.tif | ab5ebe8ab7ac9353... | 214,721 |
| AP10 | area | AP-3DEP-x/AP10/AP10_1m_area.tif | 4f641af2fa6977db... | 1,232,350 |
| AP10 | gapfill_mask | AP-3DEP-x/AP10/AP10_1m_gapfill_mask.tif | 2345e35c9c39d260... | 1,866 |
| AP10 | class2_count | AP-3DEP-x/AP10/AP10_1m_class2_count.tif | 9cdebaa63a0aba86... | 433,815 |
| AP10 | nnfill_mask | AP-3DEP-x/AP10/AP10_1m_nnfill_mask.tif | e41a0707e7925fed... | 28,655 |
| AP11 | filled | AP-3DEP-x/AP11/AP11_1m_filled.tif | c5afa265dba886bb... | 2,586,670 |
| AP11 | fd | AP-3DEP-x/AP11/AP11_1m_fd.tif | 9238ea447d2ab1ce... | 205,224 |
| AP11 | area | AP-3DEP-x/AP11/AP11_1m_area.tif | 22ab93cda170307b... | 1,300,871 |
| AP11 | gapfill_mask | AP-3DEP-x/AP11/AP11_1m_gapfill_mask.tif | e39deff5f31d871e... | 1,866 |
| AP11 | class2_count | AP-3DEP-x/AP11/AP11_1m_class2_count.tif | 2d9a121af51cb1ee... | 473,715 |
| AP11 | nnfill_mask | AP-3DEP-x/AP11/AP11_1m_nnfill_mask.tif | 9151164db9f8148b... | 26,255 |
| AP12 | filled | AP-3DEP-x/AP12/AP12_1m_filled.tif | 12579c3346ee7ea4... | 2,616,703 |
| AP12 | fd | AP-3DEP-x/AP12/AP12_1m_fd.tif | 84638d6d6e38854a... | 214,378 |
| AP12 | area | AP-3DEP-x/AP12/AP12_1m_area.tif | 917bffcdd171c3cb... | 1,175,743 |
| AP12 | gapfill_mask | AP-3DEP-x/AP12/AP12_1m_gapfill_mask.tif | 82c5bc529a02332b... | 2,115 |
| AP12 | class2_count | AP-3DEP-x/AP12/AP12_1m_class2_count.tif | 7a8de36c08006cd7... | 471,033 |
| AP12 | nnfill_mask | AP-3DEP-x/AP12/AP12_1m_nnfill_mask.tif | bd4aaf82c7556527... | 28,636 |
| AP13 | filled | AP-3DEP-x/AP13/AP13_1m_filled.tif | d22dc7b768ded4d8... | 2,539,216 |
| AP13 | fd | AP-3DEP-x/AP13/AP13_1m_fd.tif | aa0edc66997fd3ec... | 221,623 |
| AP13 | area | AP-3DEP-x/AP13/AP13_1m_area.tif | 2c422156ee4c47af... | 1,242,723 |
| AP13 | gapfill_mask | AP-3DEP-x/AP13/AP13_1m_gapfill_mask.tif | 29d24d506db87d11... | 1,866 |
| AP13 | class2_count | AP-3DEP-x/AP13/AP13_1m_class2_count.tif | 92adcbe3782d8ccb... | 484,947 |
| AP13 | nnfill_mask | AP-3DEP-x/AP13/AP13_1m_nnfill_mask.tif | 8f892f5e994db854... | 33,973 |
| AP14 | filled | AP-3DEP-x/AP14/AP14_1m_filled.tif | f85ef07c45c79063... | 2,580,998 |
| AP14 | fd | AP-3DEP-x/AP14/AP14_1m_fd.tif | 754ef5bd86be9162... | 217,096 |
| AP14 | area | AP-3DEP-x/AP14/AP14_1m_area.tif | 7d9df7907da183d6... | 1,189,546 |
| AP14 | gapfill_mask | AP-3DEP-x/AP14/AP14_1m_gapfill_mask.tif | 18283046ff51fea3... | 1,866 |
| AP14 | class2_count | AP-3DEP-x/AP14/AP14_1m_class2_count.tif | 556cbb4ed5b3c885... | 442,711 |
| AP14 | nnfill_mask | AP-3DEP-x/AP14/AP14_1m_nnfill_mask.tif | cc677acfc64f20ff... | 29,487 |
| AP15 | filled | AP-3DEP-x/AP15/AP15_1m_filled.tif | 1a12fa4c0ec0ecca... | 2,587,774 |
| AP15 | fd | AP-3DEP-x/AP15/AP15_1m_fd.tif | 549106c87dd271f5... | 230,728 |
| AP15 | area | AP-3DEP-x/AP15/AP15_1m_area.tif | 508c6a81c87750f8... | 1,192,912 |
| AP15 | gapfill_mask | AP-3DEP-x/AP15/AP15_1m_gapfill_mask.tif | ec9ab67c090c5424... | 1,866 |
| AP15 | class2_count | AP-3DEP-x/AP15/AP15_1m_class2_count.tif | 0efdd77303ec97c5... | 484,796 |
| AP15 | nnfill_mask | AP-3DEP-x/AP15/AP15_1m_nnfill_mask.tif | 8157e3abf9d8549e... | 41,257 |
| AP16 | filled | AP-3DEP-x/AP16/AP16_1m_filled.tif | aba3ea13b07fa2ea... | 2,524,822 |
| AP16 | fd | AP-3DEP-x/AP16/AP16_1m_fd.tif | 51b82adaa3b6c193... | 208,789 |
| AP16 | area | AP-3DEP-x/AP16/AP16_1m_area.tif | 60653ee5bdc491e3... | 1,229,783 |
| AP16 | gapfill_mask | AP-3DEP-x/AP16/AP16_1m_gapfill_mask.tif | 9576b542cbbc3b5f... | 1,926 |
| AP16 | class2_count | AP-3DEP-x/AP16/AP16_1m_class2_count.tif | 5d5968dfdea74ddb... | 438,967 |
| AP16 | nnfill_mask | AP-3DEP-x/AP16/AP16_1m_nnfill_mask.tif | 4fc119164c1965f5... | 25,347 |
| AP17 | filled | AP-3DEP-x/AP17/AP17_1m_filled.tif | 888a51ea02b777d7... | 2,440,589 |
| AP17 | fd | AP-3DEP-x/AP17/AP17_1m_fd.tif | 4281cf26d521cb88... | 228,090 |
| AP17 | area | AP-3DEP-x/AP17/AP17_1m_area.tif | 0655ce03d9356a16... | 1,217,728 |
| AP17 | gapfill_mask | AP-3DEP-x/AP17/AP17_1m_gapfill_mask.tif | 87a8f2f3421a803c... | 1,866 |
| AP17 | class2_count | AP-3DEP-x/AP17/AP17_1m_class2_count.tif | e4be36656d330d92... | 476,944 |
| AP17 | nnfill_mask | AP-3DEP-x/AP17/AP17_1m_nnfill_mask.tif | a9f0f06c915da7a3... | 37,329 |
| AP18 | filled | AP-3DEP-x/AP18/AP18_1m_filled.tif | 34a5995142121c48... | 2,547,657 |
| AP18 | fd | AP-3DEP-x/AP18/AP18_1m_fd.tif | 5f3513df02810797... | 220,660 |
| AP18 | area | AP-3DEP-x/AP18/AP18_1m_area.tif | c8961e2f6d8ea65b... | 1,199,629 |
| AP18 | gapfill_mask | AP-3DEP-x/AP18/AP18_1m_gapfill_mask.tif | 77a84544cc61b2c3... | 1,890 |
| AP18 | class2_count | AP-3DEP-x/AP18/AP18_1m_class2_count.tif | b7ec68dc51385027... | 470,355 |
| AP18 | nnfill_mask | AP-3DEP-x/AP18/AP18_1m_nnfill_mask.tif | 57174014c95d878c... | 28,603 |
| AP19 | filled | AP-3DEP-x/AP19/AP19_1m_filled.tif | 567efb8e8f9e589a... | 2,584,643 |
| AP19 | fd | AP-3DEP-x/AP19/AP19_1m_fd.tif | b25f95a8ce08d8fa... | 211,445 |
| AP19 | area | AP-3DEP-x/AP19/AP19_1m_area.tif | 53f428d2b5039f3f... | 1,213,929 |
| AP19 | gapfill_mask | AP-3DEP-x/AP19/AP19_1m_gapfill_mask.tif | 35e50cd73abbf00e... | 1,866 |
| AP19 | class2_count | AP-3DEP-x/AP19/AP19_1m_class2_count.tif | 778d922fffa4cac2... | 449,147 |
| AP19 | nnfill_mask | AP-3DEP-x/AP19/AP19_1m_nnfill_mask.tif | d9ac2423e0fcf551... | 28,302 |
| AP20 | filled | AP-3DEP-x/AP20/AP20_1m_filled.tif | 35aa681857e69056... | 2,575,539 |
| AP20 | fd | AP-3DEP-x/AP20/AP20_1m_fd.tif | 5dc5967e4a253f2c... | 225,930 |
| AP20 | area | AP-3DEP-x/AP20/AP20_1m_area.tif | 7f58dd69305d389c... | 1,149,769 |
| AP20 | gapfill_mask | AP-3DEP-x/AP20/AP20_1m_gapfill_mask.tif | 430fd01ffd6fb348... | 1,866 |
| AP20 | class2_count | AP-3DEP-x/AP20/AP20_1m_class2_count.tif | a111de52161ed483... | 478,221 |
| AP20 | nnfill_mask | AP-3DEP-x/AP20/AP20_1m_nnfill_mask.tif | c78d4c4221a84880... | 34,946 |
| AP21 | filled | AP-3DEP-x/AP21/AP21_1m_filled.tif | 6f7586c425206566... | 2,541,492 |
| AP21 | fd | AP-3DEP-x/AP21/AP21_1m_fd.tif | 174e8c7837d59162... | 215,315 |
| AP21 | area | AP-3DEP-x/AP21/AP21_1m_area.tif | 7f2adbbaf76f46bf... | 1,197,264 |
| AP21 | gapfill_mask | AP-3DEP-x/AP21/AP21_1m_gapfill_mask.tif | 6d08f2333bf9de37... | 1,866 |
| AP21 | class2_count | AP-3DEP-x/AP21/AP21_1m_class2_count.tif | 35310fc26408e32d... | 453,826 |
| AP21 | nnfill_mask | AP-3DEP-x/AP21/AP21_1m_nnfill_mask.tif | 00801e60c5ccc982... | 28,122 |
| AP22 | filled | AP-3DEP-x/AP22/AP22_1m_filled.tif | 775465af47e15eec... | 2,570,041 |
| AP22 | fd | AP-3DEP-x/AP22/AP22_1m_fd.tif | bf00d352ce7f0242... | 214,889 |
| AP22 | area | AP-3DEP-x/AP22/AP22_1m_area.tif | 9761f0b964b472a1... | 1,205,387 |
| AP22 | gapfill_mask | AP-3DEP-x/AP22/AP22_1m_gapfill_mask.tif | 78cb82c5467ac594... | 1,866 |
| AP22 | class2_count | AP-3DEP-x/AP22/AP22_1m_class2_count.tif | 2905d24aa9d24456... | 471,789 |
| AP22 | nnfill_mask | AP-3DEP-x/AP22/AP22_1m_nnfill_mask.tif | 7a00a94c8ac7366a... | 29,610 |
| AP23 | filled | AP-3DEP-x/AP23/AP23_1m_filled.tif | 5c904a40bb73aafc... | 2,503,883 |
| AP23 | fd | AP-3DEP-x/AP23/AP23_1m_fd.tif | f2098b2d30781e1c... | 221,856 |
| AP23 | area | AP-3DEP-x/AP23/AP23_1m_area.tif | 4657ae049b9d8c74... | 1,251,792 |
| AP23 | gapfill_mask | AP-3DEP-x/AP23/AP23_1m_gapfill_mask.tif | 8b3dabcc9043930f... | 1,866 |
| AP23 | class2_count | AP-3DEP-x/AP23/AP23_1m_class2_count.tif | 15e6f3949822bf25... | 468,357 |
| AP23 | nnfill_mask | AP-3DEP-x/AP23/AP23_1m_nnfill_mask.tif | edc8ad279d7c8449... | 37,506 |
| AP24 | filled | AP-3DEP-x/AP24/AP24_1m_filled.tif | 83d2b2329f17566a... | 2,483,811 |
| AP24 | fd | AP-3DEP-x/AP24/AP24_1m_fd.tif | e5e6ae1ecddb825e... | 220,699 |
| AP24 | area | AP-3DEP-x/AP24/AP24_1m_area.tif | 777b95d898893f54... | 1,236,383 |
| AP24 | gapfill_mask | AP-3DEP-x/AP24/AP24_1m_gapfill_mask.tif | eba581e5653f461b... | 1,866 |
| AP24 | class2_count | AP-3DEP-x/AP24/AP24_1m_class2_count.tif | 10864df5c1a63eb6... | 466,338 |
| AP24 | nnfill_mask | AP-3DEP-x/AP24/AP24_1m_nnfill_mask.tif | 508a9d9e8e0a6f16... | 35,189 |
| AP25 | filled | AP-3DEP-x/AP25/AP25_1m_filled.tif | 4b4d13298ad6735d... | 2,518,196 |
| AP25 | fd | AP-3DEP-x/AP25/AP25_1m_fd.tif | ca7b591ebb8793d2... | 222,604 |
| AP25 | area | AP-3DEP-x/AP25/AP25_1m_area.tif | 22d52a6e4261f17b... | 1,207,958 |
| AP25 | gapfill_mask | AP-3DEP-x/AP25/AP25_1m_gapfill_mask.tif | af59657ad7cc1d16... | 1,866 |
| AP25 | class2_count | AP-3DEP-x/AP25/AP25_1m_class2_count.tif | b996b3c3a693fe8a... | 455,533 |
| AP25 | nnfill_mask | AP-3DEP-x/AP25/AP25_1m_nnfill_mask.tif | a1bb5c3bb7bdf099... | 30,481 |
| AP26 | filled | AP-3DEP-x/AP26/AP26_1m_filled.tif | 9efd57077c236206... | 2,495,715 |
| AP26 | fd | AP-3DEP-x/AP26/AP26_1m_fd.tif | eda60fe469bcd6a4... | 215,337 |
| AP26 | area | AP-3DEP-x/AP26/AP26_1m_area.tif | 1e616fe991a27da2... | 1,199,939 |
| AP26 | gapfill_mask | AP-3DEP-x/AP26/AP26_1m_gapfill_mask.tif | 59fe08e520b0ff98... | 1,866 |
| AP26 | class2_count | AP-3DEP-x/AP26/AP26_1m_class2_count.tif | a8d1520eb6d7e68d... | 450,792 |
| AP26 | nnfill_mask | AP-3DEP-x/AP26/AP26_1m_nnfill_mask.tif | 2ebdb2d2afa10c22... | 28,087 |
| AP27 | filled | AP-3DEP-x/AP27/AP27_1m_filled.tif | 3bc0159b5ea644e4... | 2,604,527 |
| AP27 | fd | AP-3DEP-x/AP27/AP27_1m_fd.tif | f79712a95add161e... | 225,932 |
| AP27 | area | AP-3DEP-x/AP27/AP27_1m_area.tif | 34c492ce8d639fe8... | 1,178,273 |
| AP27 | gapfill_mask | AP-3DEP-x/AP27/AP27_1m_gapfill_mask.tif | 6c1217da733446b4... | 1,866 |
| AP27 | class2_count | AP-3DEP-x/AP27/AP27_1m_class2_count.tif | 38b90fc295e9f9ae... | 483,267 |
| AP27 | nnfill_mask | AP-3DEP-x/AP27/AP27_1m_nnfill_mask.tif | 4dc4f70cbebe2493... | 33,923 |
| AP28 | filled | AP-3DEP-x/AP28/AP28_1m_filled.tif | dcaac0ed4f55a638... | 2,580,548 |
| AP28 | fd | AP-3DEP-x/AP28/AP28_1m_fd.tif | b80d088e8a9fa3d7... | 220,558 |
| AP28 | area | AP-3DEP-x/AP28/AP28_1m_area.tif | 52cd231b2b1e07a3... | 1,186,532 |
| AP28 | gapfill_mask | AP-3DEP-x/AP28/AP28_1m_gapfill_mask.tif | 8ef540714a25980e... | 1,866 |
| AP28 | class2_count | AP-3DEP-x/AP28/AP28_1m_class2_count.tif | 7891fa604e26fd32... | 476,948 |
| AP28 | nnfill_mask | AP-3DEP-x/AP28/AP28_1m_nnfill_mask.tif | 76c62f9c9d698a30... | 27,645 |
| AP29 | filled | AP-3DEP-x/AP29/AP29_1m_filled.tif | fc55f9e15b9c8eaf... | 2,532,441 |
| AP29 | fd | AP-3DEP-x/AP29/AP29_1m_fd.tif | cbcb87696b5c9524... | 216,584 |
| AP29 | area | AP-3DEP-x/AP29/AP29_1m_area.tif | 77a052d647e42deb... | 1,191,048 |
| AP29 | gapfill_mask | AP-3DEP-x/AP29/AP29_1m_gapfill_mask.tif | 5a69f65899cfef8b... | 1,866 |
| AP29 | class2_count | AP-3DEP-x/AP29/AP29_1m_class2_count.tif | 473dcadda0b7e8ee... | 444,799 |
| AP29 | nnfill_mask | AP-3DEP-x/AP29/AP29_1m_nnfill_mask.tif | a5214c2d3c3caa71... | 25,829 |


### B9. Draw configuration (written before the draw)

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
   "theta": 0.43700000000000006,
   "ks": 4.024132093268996,
   "C_ht": 0.01727684020996094,
   "source": "georeferenced M&D (1989) study-area polygon on the 10 m NED, same code (a15)"
  }
 },
 "written": "2026-11-18T09:35:55-0800",
 "note": "written before the draw"
}
```

Threshold set B: {"_comment": "Addendum 14 stage A: ALTERNATIVE (relaxed) screening set B, shown beside set A so the consequence of each rule is visible. Not applied as final.", "trunk_A_max_inside_m2_max": 5000000.0, "trunk_A_inflow_max_single_m2_max": 2000000.0, "transient_chi_r2_area_weighted_min": 0.9, "transient_sa_dbic_area_weighted_max": 20.0, "disturbance_nlcd_disturbed_frac_max": 0.1, "disturbance_nlcd_classes": [11, 22, 23, 24, 31], "disturbance_mining_frac_max": 0.0, "disturbance_mining_layers": ["longwall_panels"], "upstream_fully_valid_required": {"GM": true, "TTV": true, "AP": false}, "tolerance_ks_rel": 0.25, "tolerance_C_ht_rel": 0.25, "tolerance_theta_abs": 0.05}


### B10. Slurm accounting of the stage B jobs

| JobID | JobName | Partition | State | Elapsed | Timelimit | AllocCPUS | MaxRSS | TotalCPU |
|---|---|---|---|---|---|---|---|---|
| 52815535 | a15_tests | serc | COMPLETED | 00:04:50 | 00:30:00 | 2 |  | 00:00:00 |
| 52815788 | a15_reroute_GM | serc | COMPLETED | 00:01:31 | 00:30:00 | 2 |  | 00:00:00 |
| 52815791 | a15_reroute_TTV | serc | COMPLETED | 00:03:53 | 00:40:00 | 3 |  | 00:00:00 |
| 52815802 | a15_reroute_AP | serc | COMPLETED | 00:14:45 | 01:20:00 | 4 |  | 00:00:00 |
| 52816789 | a15_diag_GM | serc | COMPLETED | 00:00:21 | 00:30:00 | 2 |  | 00:00:00 |
| 52817132 | a15_georef | serc | COMPLETED | 00:02:30 | 00:30:00 | 2 |  | 00:00:00 |
| 52817205 | a15_reroute_GM | serc | COMPLETED | 00:01:09 | 00:30:00 | 2 |  | 00:00:00 |
| 52817235 | a15_diag_GM | serc | COMPLETED | 00:00:21 | 00:30:00 | 2 |  | 00:00:00 |
| 52817635 | a15_ttvref | serc | COMPLETED | 00:01:28 | 00:40:00 | 1 |  | 00:00:00 |
| 52817891 | a15_georef | serc | COMPLETED | 00:02:03 | 00:30:00 | 2 |  | 00:00:00 |
| 52818249 | a15_ttvref | serc | COMPLETED | 00:01:32 | 00:40:00 | 1 |  | 00:00:00 |
| 52819086 | a15_georef | serc | COMPLETED | 00:02:05 | 00:30:00 | 2 |  | 00:00:00 |
| 52819087 | a15_ttvref | serc | COMPLETED | 00:01:45 | 00:40:00 | 1 |  | 00:00:00 |
| 52819195 | a15_georef | serc | COMPLETED | 00:00:22 | 00:30:00 | 2 |  | 00:00:00 |
| 52819289 | a15_draw | serc | CANCELLED by 313293 | 00:00:00 | 00:40:00 | 0 |  | 00:00:00 |
| 52820070 | a15_georef | serc | COMPLETED | 00:02:37 | 00:30:00 | 2 |  | 00:00:00 |
| 52820071 | a15_ttvref | serc | COMPLETED | 00:01:00 | 00:40:00 | 1 |  | 00:00:00 |
| 52820073 | a15_draw | serc | CANCELLED by 313293 | 00:00:33 | 00:40:00 | 2 |  | 00:00:00 |
| 52820695 | a15_georef | serc | COMPLETED | 00:02:41 | 00:30:00 | 2 |  | 00:00:00 |
| 52820696 | a15_georef | serc | COMPLETED | 00:00:19 | 00:30:00 | 2 |  | 00:00:00 |
| 52820987 | a15_ttvref | serc | CANCELLED by 313293 | 00:00:00 | 00:40:00 | 0 |  | 00:00:00 |
| 52822212 | a15_georef | serc | COMPLETED | 00:01:52 | 00:30:00 | 2 |  | 00:00:00 |
| 52822213 | a15_georef | serc | COMPLETED | 00:00:18 | 00:30:00 | 2 |  | 00:00:00 |
| 52822216 | a15_ttvref | serc | COMPLETED | 00:01:23 | 00:40:00 | 1 |  | 00:00:00 |
| 52822218 | a15_draw | serc | COMPLETED | 00:02:24 | 00:40:00 | 2 |  | 00:00:00 |
| 52822793 | a15_georef | serc | COMPLETED | 00:02:01 | 00:30:00 | 2 |  | 00:00:00 |
| 52822796 | a15_georef | serc | COMPLETED | 00:00:19 | 00:30:00 | 2 |  | 00:00:00 |
| 52824241 | a15_draw | serc | COMPLETED | 00:01:33 | 00:40:00 | 2 |  | 00:00:00 |
| 52824977 | a15_cut | serc | COMPLETED | 00:00:18 | 00:40:00 | 1 |  | 00:00:00 |
| 52824981 | a15_cut | serc | COMPLETED | 00:00:22 | 00:40:00 | 1 |  | 00:00:00 |
| 52826454 | a15_cut | serc | COMPLETED | 00:00:15 | 00:40:00 | 1 |  | 00:00:00 |
| 52826460 | a15_cut | serc | CANCELLED by 313293 | 00:00:00 | 00:40:00 | 0 |  | 00:00:00 |
| 52827033 | a15_cut | serc | COMPLETED | 00:00:19 | 00:40:00 | 1 |  | 00:00:00 |
| 52827037 | a15_cut | serc | COMPLETED | 00:00:18 | 00:40:00 | 1 |  | 00:00:00 |
| 52827880 | a15_cut | serc | COMPLETED | 00:00:59 | 00:40:00 | 1 |  | 00:00:00 |
