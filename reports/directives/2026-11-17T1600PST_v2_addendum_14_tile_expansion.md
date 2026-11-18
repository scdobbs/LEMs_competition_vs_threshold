# [2026-11-17 16:00 PST] Addendum 14: tile expansion, stage A (selection pools, full-grid data, units). No draw, no modelling

This addendum follows Addendum 13 and its report `reports/v2/20261117T1445PST_data.md`. Addendum 4 §1 (timestamped titles, one report per checkpoint), Addendum 2 §4–§5 (Slurm, storage, push) and Addendum 7 §0.4 (submit guard) still apply. Work on a new branch `v2.3-tiles`, cut from `v2.2-data`.

## 0. Orientation (you are a new agent instance)

1. **Read:**
   - `reports/v2/20261117T1445PST_data.md` (the data checkpoint);
   - `reports/v2/20261117T0900PST_sensitivity.md`;
   - `docs/measurement_methods.md` §0;
   - the submitted manuscript's Methods §2.1–2.2 (tile selection; `papers/`, never committed);
   - this file.
2. **Confirm** the state of `main`, `v2.2-data` and `origin`, that the tests pass on `serc`, and that no job is running.
3. **Where this addendum and the repo disagree,** the repo wins. Report it and stop.

**Why this addendum.**
- The paper's tests rest on five 1 km tiles per site. Steve wants about 30 per site at GM and AP, and a third site, Tennessee Valley (TTV, Marin County, CA), the Montgomery & Dietrich (1989, 1992) channel-head field area.
- **The tests stay the same as in the submitted paper.** For every tile: per-tile measured K₀ = E/k_s and D₀ = E/|C_ht| from the tile's own lidar and the site's cosmogenic E; the co-located AD and threshold-partitioned models fitted by elevation MSE; then the MSE comparison (paper Fig. 6), the parameter ratios to the measured values (Fig. 5) and the slope–area comparison (Fig. 4). All of this uses the v2 methods (`docs/measurement_methods.md` §0), not the v1 methods printed in the manuscript.
- This addendum is **stage A**: build the candidate pools, propose the screening thresholds, get the full-grid data and check units. **No tiles are drawn and nothing is modelled.** Steve approves the thresholds; the draw, gridding of tiles, measurement and modelling come in later addenda.

## 1. Decisions

Steve will strike any he does not accept before this reaches you.

1. **Counts.** GM and AP: 30 tiles each, made of the five original tiles plus 25 drawn at random. Fewer if the eligible pool is smaller; never overlap tiles to make up the number. TTV: as many as are eligible, up to 30.
2. **The original five stay** at GM and AP, whatever the new screens say. Report whether each would pass the new rules.
3. **Selection is eligibility plus a random draw, not "best match".** A window is eligible if it passes every screen (§3) and each selection metric is within tolerance of the site reference (§3.3). The draw (stage B) is random among eligible windows, with a seed fixed and recorded before the draw. This replaces the weighted three-term RMS of the submitted Methods §2.2.
4. **Selection metrics are computed on the coarse grids only:** GM 10 m NED, AP 30 m NED (the Perron grids in `source/Perron_coarse/`), TTV 10 m NED (to be downloaded). The lidar is not used for selection, only for footprint coverage, so the tiles' measured parameters stay independent of how the tiles were chosen.
5. **Tolerances:** k_s and |C_ht| within ±25 % of the site reference; θ within ±0.05.
6. **References:**
   - **GM and AP:** the reference the original selection used. Find it in the repo, `old-notebooks/` or the v1 configs, and report where it came from. If it cannot be found, rebuild it from the submitted Methods §2.2 and Perron et al. (2012, SI Table 2): regional θ by χ on the NED (GM 0.292, AP 0.54); |∇²z_h| from Perron; and k_s = E/K with K = D ÷ (D/K) and D = E/|∇²z_h|. State the rebuild and every number.
   - **TTV:** there are no Perron values. The reference is the M&D study area: its θ, k_s and |C_ht| on the 10 m NED, by the same code. Tiles covering the M&D area are included by design.
7. **Drainage area is routed on the full grid, then clipped to the tiles, never computed on a cut tile.** Full grids:
   - **AP:** 3DEP over the whole watershed that `PAMAP_DEM_FIXED_REPROJECT_12-2025.tif` is clipped to. That file is PAMAP, not 3DEP; Addendum 13 downloaded 3DEP only under AP1–AP5 plus 500 m. §4.1 downloads the rest.
   - **GM:** `Gabilan_mesa.tif`, with the rule that a tile's whole upstream area must lie in valid data.
   - **TTV:** the main TTV survey (decision 8) over the watersheds containing the TTV pool, with the same rule.
8. **TTV data:**
   - **Main:** Marin County 2019 QL1 lidar (countywide, ground-classified point cloud; MarinMap, "2019 LIDAR & Orthos").
   - **Cross-check:** the 2006 NCALM survey. Steve put an OpenTopography bare-earth raster of it at `$SCRATCH/lem_topography/output_be.tif`; copy it to storage of record and inventory it (§2). OpenTopography lists that survey as NAD83(CORS96) / UTM 10N (EPSG:26910), NAVD88 (GEOID03), metres. Verify this from the file.
   - If Marin 2019 cannot be obtained, report why and stop. Do not substitute silently.
9. **Gridding rule for all point clouds** (AP 3DEP, Marin 2019): 1 m cells; the mean of class-2 (ground) points per cell; empty cells filled by nearest neighbour, with the filled fraction recorded per cell and per tile. **No TIN.** CRS: NAD83(2011) UTM of the site's zone (AP EPSG:6346; TTV EPSG:6339). Heights in metres. The lattice starts at integer metres.
10. **Units, everywhere:** read the units from each file's header or metadata. Convert US survey feet with 1 ftUS = 1200/3937 m, and international feet with 0.3048. Record which. Every converted grid must pass an independent comparison (§5) before use. Do not convert on an assumption.
11. **TTV erosion rate:** take E from Heimsath et al. (1997, *Nature*) and, if present, Heimsath et al. (1999, *Geomorphology*) in `papers/`. Report the value, its uncertainty, the method (basin-averaged detrital ¹⁰Be/²⁶Al, or point soil production), and the sample locations relative to the M&D area. Also record Reneau's (1988) linear diffusivity of about 50 cm²/yr (0.005 m²/yr) as cited in Heimsath et al. (2005). It is a check only, not an input.
12. **Budget:** expected ≤ 60 core-hours; sum of time limits ≤ 120; every submission ≤ 300 core-hours of limits; arrays throttled; every job on `serc`. New data on group storage ≤ 100 GB; if more is needed, stop and report first.

## 2. Inventory the new TTV raster

`$SCRATCH/lem_topography/output_be.tif`: copy to `$GROUP_HOME/lemcvt_data/source/TTV/` with a sha256 manifest, verify, then report as in Addendum 13 §2.1:
- format, CRS (horizontal and vertical), units and how established;
- resolution, extent, nodata, value range;
- the gridding method if the file or its OpenTopography job metadata states it;
- whether it covers the M&D area (Addendum 13 §3.3) and the eastern strip of the Point Bonita quadrangle east of 122.515 W.

## 3. Candidate pools and screens (on the coarse grids)

### 3.1 Windows

- Lay a regular lattice of non-overlapping 1 km windows over each site's lidar footprint. Use 4 lattice offsets (0 and 500 m in each axis) and report the count for each offset.
- A window is a **candidate** only if its 1 km square is 100 % valid in the lidar footprint (AP: the 3DEP tile index; GM: `Gabilan_mesa.tif`; TTV: the main survey's footprint).

### 3.2 Screens (propose thresholds; do not apply them as final)

For every candidate, compute the quantities below, then **propose** a threshold for each with its reason. Report the distribution and how many windows each threshold removes.
1. **Trunk channels:** the largest drainage area inside the window, and the area entering across its edge, from routing on the full grid (the coarse grid for this stage). This is the paper's "complete tributary networks, exclude trunk channels".
2. **Transient signals:** per basin, the R² of the χ–z fit at the reference θ; a slope–area break test (two-segment against one-segment fit, by BIC, above 10³ m²). The window statistic is the worst basin weighted by area.
3. **Disturbance:** the fraction of the window that is roads, ponds, development or quarries from NLCD (or a better layer you find and justify). At AP also include surface mining and longwall-mining subsidence from Pennsylvania DEP mining layers, if obtainable. If not, say so.
4. **Independence:** no two drawn windows may share an edge-outlet basin. Report the conflict graph's size; the draw will respect it.

### 3.3 Selection metrics against the reference (decision 6)

For every candidate, on the coarse grid, by the same code at all sites:
- θ by χ (the original method, Perron & Royden 2013);
- k_s at the site reference θ;
- |C_ht| by the original coarse-grid method (block-mean, 9-point stencil, median of convex first-order divide pixels, saddles and flats excluded; submitted Methods §2.2).

Report each metric's distribution and the number of windows within tolerance (decision 5), alone and jointly with every proposed screen.

### 3.4 The original five against the new rules

For AP1–AP5 and GM1–GM5, report every screen statistic and metric, and whether each tile would pass. They stay in the set either way (decision 2).

## 4. Full-grid data

### 4.1 AP: 3DEP over the PAMAP watershed

1. **Footprint:** the valid area of `PAMAP_DEM_FIXED_REPROJECT_12-2025.tif` plus a 500 m buffer. List the 5,000 ft PA_South quadrants with `pasda_tiles_for_ap.py` logic (adapt it to a polygon; keep the naming rule and the Addendum 13 checks).
2. **Source:** first try the USGS 3DEP cloud copy of `PA_WesternPA_1_2019` (the AWS-hosted point clouds; check whether the project is there and whether the points equal PASDA's for one quadrant, by count and checksum of class-2 coordinates). If it is not available or not equal, use PASDA as in Addendum 13. Report rates.
3. **Size gate:** the expected size is about 45–50 GB. Stop if the listed total exceeds 60 GB.
4. Re-run Addendum 13 rules 5.1 and 5.2 on every new file.
5. **Grid** the whole watershed by decision 9; fill depressions and route D8 on the full grid. If memory forbids one grid, route per sub-watershed and prove the sub-watersheds' outlets are the watershed's (no flow crosses a cut). Report the memory and time used.

### 4.2 GM

No new data. Report the valid footprint, and for every candidate whether its upstream area is fully valid.

### 4.3 TTV

1. **Marin 2019 QL1:** find the tile index and metadata (MarinMap, or a USGS 3DEP or NOAA copy of the same collection; prefer the copy with a stated CRS and units). Download the tiles covering the Marin Headlands watersheds that contain the TTV pool, plus 500 m. Size gate 30 GB.
2. **Report:** CRS, horizontal and vertical units (California State Plane in US survey feet is likely; verify), point format, classes, class-2 density over the pool and in hollows, collection dates.
3. **Grid** by decision 9; route on the full grid.
4. **10 m NED** over the same area, for §3: download from the USGS National Map and state the product and its date.

## 5. Unit and datum checks (rules written now)

**Rule 5.1, AP.** On AP1–AP4, the new watershed grid must agree with the Addendum 13 3DEP check: median Δz within ±0.1 m of the Addendum 13 values, and Theil–Sen slope 0.99–1.01 against the PAMAP tiles. AP5 is excluded (Addendum 13 §6.2).

**Rule 5.2, TTV.**
- Marin 2019 against the NCALM raster (`output_be.tif`) on their common area, ground cells only: median Δz within ±1.0 m, Theil–Sen slope 0.98–1.02.
- A slope near 3.28 or 0.305 means a units error. An offset near −32 m means ellipsoidal heights. Either one: stop.

**Rule 5.3, NED.** Every coarse grid against its site's lidar, block-averaged to the coarse cell:
- slope 0.98–1.02;
- GM expects an offset of 30–34 m (ellipsoidal lidar, Addendum 13 §3.4); the other sites expect |offset| < 5 m.

Report each case. Do not edit or filter any grid to pass.

## 6. Cost

Print the estimate first. Expect most time in the downloads (I/O, 1 CPU) and the AP full-grid gridding and routing. Submit without asking only within decision 12.

## 7. Report and stop

One report, `reports/v2/<YYYYMMDDTHHMM>PST_tiles_stage_a.md`, titled with its timestamp. Fixed outline:

1. State check.
2. The TTV raster `output_be.tif` (§2) and the TTV erosion rate (decision 11).
3. References per site and where they came from (decision 6).
4. Candidate pools: counts per lattice offset (§3.1).
5. Screens: distributions, proposed thresholds with reasons, windows removed by each (§3.2).
6. Selection metrics and the eligible count, alone and jointly (§3.3); the expected eligible count under the proposed thresholds, against the target of 30.
7. The original ten tiles against the new rules (§3.4).
8. Full-grid data: downloads, gridding, routing (§4).
9. Unit and datum checks with the rules applied (§5).
10. Decisions needed: in particular the thresholds to approve, and what to do if a site has fewer than 25 eligible windows.
11. Appendices: all generated tables, file lists with checksums.

**Stop for review.** Do not draw tiles, do not measure tile parameters on the lidar, do not run any model, do not merge or tag. Push `v2.3-tiles` after the Addendum 2 §5 check.
