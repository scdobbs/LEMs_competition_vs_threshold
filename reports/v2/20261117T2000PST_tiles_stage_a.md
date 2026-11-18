# [2026-11-17 20:00 PST] Tile expansion, stage A (addendum 14): candidate pools, screens, full-grid data, units

Directive: `reports/directives/2026-11-17T1600PST_v2_addendum_14_tile_expansion.md` (outline fixed by its section 7). Branch
`v2.3-tiles`, cut from `v2.2-data` (d938d20). **No tile was drawn, nothing was measured on the lidar tiles, no model was run,
no merge, no tag.** Scripts: `reports/v2/scripts/a14_*.py`, `a14_download.sh`, `a14_thresholds.json` (set A, proposed),
`a14_thresholds_B.json` (set B, relaxed), `make_a14_tables.py`. Environments: `lemcvt` (pools, unchanged) and `lidar`
(point clouds; numba 0.68.1 added in job 52737645). Every number is tied to the run directory
`$GROUP_HOME/lemcvt_results/20261118T002225Z_a14_stage_a/` (subdirectories named in each section) or to a Slurm job id.
Appendices (section 11) are machine-generated. **This is a stop for review.**

## Summary

| Item | Result |
| --- | --- |
| 1 state | `main` = `origin/main` = ffa9367; `v2.2-data` = origin = d938d20; no job running; tests on `serc` passed (job 52730900, 269 passed, ruff clean). Nine points where the addendum and the repo or data differ (section 1); none required a stop |
| 2 TTV raster | `output_be.tif` is a 1 m NAD83 / UTM 10N (EPSG:26910) COG, vertical CRS not stated (metres, orthometric inferred from the sea surface at +1.2 to +1.9 m); an **interpolated surface** fitted through the NCALM ground points (57 % of its cells hold no class-2 point; it equals the per-cell mean in 1 % of cells), not a mean-of-points grid; inside the Point Bonita quadrangle and the 61-314 m band; **does not cover the strip east of 122.515 W** (its east edge is at 122.5150 W). Copies of record with manifests. TTV erosion rate from Heimsath et al. (1997): basin-averaged detrital ¹⁰Be/²⁶Al 64 ± 12, 66 ± 13, 102 ± 25 m Myr⁻¹; point soil-production rates 15-107 m Myr⁻¹; Reneau's 50 cm² yr⁻¹ recorded |
| 3 references | **Found in the repo**: `TAYLOR_REF` of `legacy/measurement_notebooks/updated_best_fit_tile_finder.ipynb` (GM θ 0.292, k_s 1.9, |C_ht| 12.8 × 10⁻³; AP 0.54, 47.2, 5.93 × 10⁻³; weights curvature 5, k_s 1, θ 1); θ and k_s (median of per-basin slopes) traced to the χ notebook's outputs on the coarse NED grids, curvature to the notebook's own whole-grid divide medians (8 % above Perron's SI Table S2). TTV reference from the M&D proxy catchment (0.67 km² above 61 m): θ 0.387, k_s 15.4, |C_ht| 15.1 × 10⁻³ |
| 4 pools | Candidates per offset: **GM 17-19 (71 in all), AP 411-414 (1,653), TTV 99-110 (420)** |
| 5 screens | Set A (proposed) removes, jointly, 50 of 71 at GM, 1,636 of 1,653 at AP (NLCD open space and the PA DEP mined-out layers do most of it) and 387 of 420 at TTV (Sausalito, Marin City, Mill Valley); set B (built-up classes only, longwall only, R² ≥ 0.90, ΔBIC ≤ 20) leaves **GM 38, AP 320, TTV 102** |
| 6 metrics | Within all three tolerances (θ ± 0.05, k_s and |C_ht| ± 25 %): GM 9 of 71, AP 181 of 1,653, TTV 12 of 420; **eligible (tolerances and screens): GM 4, AP 0 (set A) / 13 (set B), TTV 1 / 3**. θ is the binding constraint: the legacy window-level θ scatters by ±0.09 (GM) to ±0.2 (AP, TTV). With θ free and ±25 %: GM 14, AP 68, TTV 20 (set B) |
| 7 original tiles | GM1 and GM3 pass the set A screens; AP3 and GM3 meet all three tolerances; only GM3 does both. AP1, AP2 and AP5 lie 87-100 % over PA DEP mined-out or longwall ground; GM2, GM4, AP3, AP4 hold 6-17 km² trunk channels |
| 8 full grids | AP: PASDA and rockyweb throttle to 0.05-0.4 MB/s (days for 52 GB); the **AWS Entwine copy holds the same points** (6,877,263 against 6,877,258 class-2 in the test quadrant; every sampled point matched within 0.05 ft, median 3.6 mm; Z in metres, ratio 0.3048019) but its checksum differs (reprojection, 1 cm quantisation). Fetched from it: **AP 642 1 km boxes, 1.92 G ground points, 12.1 GB in 17 min; TTV (Marin 2019 QL1 = USGS B5b) 235 boxes, 1.10 G points, 7.2 GB in 14 min**; copies of record 19.5 GB. Gridded at 1 m (mean of class-2, NN fill ≤ 10 m, NAD83(2011) UTM) and routed on the whole grids: AP 577 M valid cells, fill 6.1 min, D8 1.4 min, area 3.2 min, 47 GB peak; with every gap filled the watershed outlet drains 89 % of the grid, with gaps > 10 m left nodata (the products kept) the gap rims become outlets (decision 8) |
| 9 units | Rule 5.1 (AP1-AP4): median Δz −0.071, −0.073, −0.066, −0.021 m against addendum 13's −0.079, −0.077, −0.066, −0.026; slopes 0.9993-1.0006: **pass**; AP5 −1.213 m (excluded, as in addendum 13). Rule 5.2 (Marin 2019 against NCALM 2006, 12.8 M ground cells): median +0.084 m, slope 0.99999: **pass, no stop**. Rule 5.3: GM +32.3 m (expected 30-34), slope 0.999; AP +1.2 m, 0.986; TTV +0.23 m, 1.0003: **all pass** |
| cost | **15.9 core-hours used** in 37 submissions (cap 60); the **sum of CPU-weighted time limits is 973 core-hours (248 wall-clock hours), over the 120 cap** (section 8.5: 8-CPU fetch jobs with 12 h walls, resubmitted after three bugs; two cancelled 30-36 h download jobs); largest single submission 96 core-hours of limits (cap 300); every job on `serc`; new data on group storage 19.5 GB of LAZ plus about 15 GB of 1 m grids (cap 100) |

## 1. State check

**At the start (2026-11-17 16:00 PST, login node sh04-ln01).**
- `main` = `origin/main` = ffa9367 (the addendum 12 merge); `v2.2-data` = `origin/v2.2-data` = d938d20 (the addendum 13 report); no
  job was running (`squeue --me` empty). Branch `v2.3-tiles` cut from d938d20.
- Tests on `serc` at d938d20: job 52730900, ruff clean, **269 passed** (4 min 24 s, 2 CPUs).
- Read: the data checkpoint report, the sensitivity report, `docs/measurement_methods.md` §0, the submitted Methods §2.1-2.2
  (extracted from the .docx with pandoc into `$SCRATCH/a14/papers/`, not committed), the addendum.

**Where the addendum and the repo or the data differ** (none stopped the work; each is handled as stated):
1. **Steve pre-staged the TTV files.** `$SCRATCH/lem_topography/` held `output_be.tif` (decision 8) and the 10 m NED
   `USGS_13_n38w123_20120801.tif` (4.3.4 says "download"); both were copied to storage of record instead of downloaded (§2).
   Steve's note with the directive said no download should be needed; the addendum's 4.1 (3DEP over the whole PAMAP watershed)
   and 4.3.1 (Marin 2019 tiles) still require downloads, and neither dataset is on Sherlock. They were run as the addendum
   says, within decision 12 (§8).
2. **The "Marin County 2019 QL1 lidar" is a USGS 3DEP work unit.** MarinMap's 2019 LiDAR page points to the county's
   derivative services; the classified point cloud is `CA_NoCal_Wildfires_B5b_QL1_2018` (Lot 12 of the USGS Northern
   California QL1 project, Quantum Spatial, contract G16PC00016 / 140G0218F0251), on rockyweb and as an Entwine copy on AWS.
   Its best-use metadata states the CRS and units (Albers NAD83(2011), metres; NAVD88 Geoid12B, metres; 1,000 m tiles; flights
   2018-07-07 to 2019-09-05 for the whole project). This is the "USGS 3DEP copy of the same collection" that 4.3.1 allows and
   prefers (stated CRS and units). The "TL_QL1" unit of the same project lies in Butte County and does not cover Marin.
3. **The AWS copy of `PA_WesternPA_1_2019` is an Entwine octree in EPSG:3857** (Web Mercator, 0.01 m scale, 117.9 G points),
   so its coordinates cannot equal PASDA's State Plane feet by checksum; its source list names the delivered LAZ tiles
   (`s3://usgs-lidar/Projects/PA_WesternPA_2019_D20/PA_WesternPA_1_2019/LAZ/`, requester pays, no credentials here), which USGS
   also stages on rockyweb (7,254 LAZ, 902 GB, USGS 1 km tiles named by UTM 17N kilometre, e.g. `17SNE565407`). Both were
   tested against PASDA for one quadrant (§8).
4. **Throttling.** Every HTTP source tested (PASDA, rockyweb, from the login node and from compute nodes) delivers about
   75-120 KB/s per connection, and rockyweb answers HTTP 429 to bursts of new connections (a 12-stream job got only 429s and
   was cancelled). The AWS S3 copy is the only unthrottled source. This sets the download plan (§8).
5. **On the AP coarse grid no 1 km window is 100 % valid in PAMAP.** The watershed-clipped PAMAP grid has scattered nodata
   cells inside the watershed, so the literal "every lidar cell valid" test leaves no 33 × 33-cell window. The addendum's AP
   footprint is the 3DEP tile index (3.1), which covers the whole area; the PAMAP watershed (decision 7) is applied as
   "every coarse cell holds PAMAP data", and the PAMAP hole fraction is reported per window (§4).
6. **AP tile index coverage.** The clipped 5,000 ft index stops at the PA border; 9.2 % of the AP 30 m grid's valid cells (south
   of the border, in West Virginia) are outside it and are excluded from the pool.
7. **The legacy tile finder's hand-picked AP window is AP5.** Its UTM extent (556,893-557,883 E, 4,422,470-4,423,460 N) equals
   AP5's; its hand-picked GM window (699,991-700,991 E, 3,981,451-3,982,451 N) is the 10 m-lattice window west of GM1. The
   notebook therefore does not record which of its ranked candidates became the five tiles.
8. **PASDA's listing bug in the first footprint pass**: the 28 addendum 13 files were counted as "not downloaded" (name
   pattern); corrected in the URL list used for the download (266 remaining files).
9. **Data-edge outlets.** TopoAnalysis's D8 (and so lemcvt's port) gives a cell with no lower neighbour the code of its
   least-uphill neighbour, so cells on the footprint edge pointed back inward and no cell was an outlet (the AP1 pilot showed
   zero roots). In both routing paths here every valid cell bordering nodata or the grid edge is an outlet (code 0), which is
   the priority flood's own boundary; the pools were rerun with it (GM's largest coarse basin went from 67 to 33 km²).

## 2. The TTV raster `output_be.tif` and the TTV erosion rate

### 2.1 `output_be.tif` (decision 8, §2) — copy of record `source/TTV/NCALM_2006_be/output_be.tif`, sha256 691c5ae9…e5aa, 65,603,380 bytes, verified

| quantity | value | how established |
| --- | --- | --- |
| format | GeoTIFF, Float32, one band, cloud-optimised layout (COG), LZW, 256 × 256 blocks; tags `AREA_OR_POINT=Area` only; no provenance or method tag | header |
| CRS | **NAD83 / UTM zone 10N, EPSG:26910**, metres (WKT in the file); **vertical CRS not stated** | header |
| units | horizontal metre (header); vertical *inferred* metres and orthometric: the sea surface in Tennessee Cove sits at z = 1.24-1.89 m (0.1-50 percentiles of the 534,322 cells below 3 m), where ellipsoidal heights would be about −32 m | header; inferred from the low surface |
| resolution | 1.0 × 1.0 m, lattice at half-metre offsets (origin 538,470.5, 4,191,961.5) | header |
| extent | E 538,470.5-542,654.5, N 4,187,853.5-4,191,961.5 (4,184 × 4,108 cells, 17.2 km²); 122.5628-122.5150 W, 37.8384-37.8744 N | header |
| nodata | −3.4028235 × 10³⁸; 1,455,259 of 17,187,872 cells (8.5 %), the sea and the survey edge | header, read |
| value range | 0.15-317.37 m; mean 139.5 m | read |
| gridding method | **not stated in the file or in any sidecar** (OpenTopography's job metadata was not staged). Test against the NCALM LAS class-2 points (8.89 M, 8,887,541 inside the raster): the raster at the nearest cell differs from the point by −0.0003 m in the median (5-95 %: −0.21 to +0.21 m); sampled bilinearly at the point, −0.001 m (−0.07 to +0.08 m); the raster equals the mean of its cell's points within 1 mm in only 1.3 % of cells and the minimum or maximum in 1.0 %; it lies between the cell's minimum and maximum in 61 % of cells; **57.4 % of the valid cells hold no class-2 point** (the raster is continuous there). So it is an interpolated surface fitted through the ground points (a TIN or IDW of the kind OpenTopography builds), not a per-cell statistic, and not decision 9's mean-of-points grid | computed |
| M&D area | inside the Point Bonita quadrangle (122.625-122.5 W, 37.75-37.875 N); 82.8 % of its valid cells lie in the 61-314 m band; maximum 317.4 m matches Coyote Ridge (314 m on the 1989 map) | computed |
| eastern strip east of 122.515 W | **not covered**: the raster's east edge is at 122.51500 W; 0 columns lie east of it | computed |
| datum shift to the working lattice (EPSG:6339) | 0.00 m (PROJ applies no NAD83 → NAD83(2011) shift offline; the true shift here is below 1 m, immaterial at 1 m and 10 m) | PROJ |

### 2.2 The 10 m NED — copy of record `source/TTV/NED_13arcsec/USGS_13_n38w123_20120801.tif`, sha256 4e4fa132…494d43, 218,380,355 bytes, verified

USGS 1/3 arc-second NED, cell n38w123, product date 2012-08-01 (from the file name; no metadata tag in the file). GeoTIFF Float32
COG (LZW, predictor 3, overviews), CRS NAD83 geographic (EPSG:4269), 1/3″ = 9.26 × 10⁻⁵ °, 10,812 × 10,812 cells, extent
123.0006-121.9994 W, 36.9994-38.0006 N; nodata −999,999 (36 cells); values −66.8 to 986.3 m; vertical CRS not stated (the NED
specification is NAVD88 metres). Warped to the TTV lattice (EPSG:6339, 10 m, integer-metre origin, bilinear; 1,800 × 1,800 cells
over E 530,000-548,000, N 4,180,000-4,198,000; 1,524,547 land cells with z > 0) as `ttv_inventory/TTV_NED_10m_utm10n.tif`.

### 2.3 TTV erosion rate (decision 11)

Only Heimsath et al. (1997, *Nature* 388, 358-361) is in `papers/`; Heimsath et al. (1999) and (2005) are not, so Reneau's
diffusivity is taken from the 1997 paper's own citation. From the 1997 paper (text extracted with pdftotext):
- **Method.** In situ cosmogenic ¹⁰Be and ²⁶Al in bedrock from the base of soil pits and from outcrops on four small ridges
  (point soil-production / lowering rates), and basin-averaged rates from stream sediments of two steep tributaries ("1" and
  "2" in their Fig. 3 inset). The Table 1 footnote places the site at 37.98 N, 122.68 W (as printed; Tennessee Valley itself is
  at 37.86 N, 122.54 W), soil density 1.4 g cm⁻³.
- **Point rates** (Table 1, −∂e/∂t, m Myr⁻¹, ±1σ): 15 ± 4 to 107 ± 23 over 14 samples (TV-2 39 ± 8, TV-3 47 ± 15, TV-4 20 ± 5,
  TV-5 15 ± 4, TV-6 26 ± 3, TV-7 21 ± 3, TV-10 25 ± 4, TV-11 107 ± 23, TV-12 60 ± 16, TV-13 26 ± 5, TV-15 48 ± 8, TV-16 33 ± 5,
  TV-17 27 ± 5, TV-23 91 ± 24); soil production declines exponentially with soil depth.
- **Basin-averaged detrital rates**: Creek1a 64 ± 12, Creek1b 66 ± 13 (the same tributary, two samples), Creek2 102 ± 25 m Myr⁻¹,
  "close to the maximum soil production rates".
- **Reneau's diffusivity**: the paper computes soil production from curvature "using a regional diffusivity of 50 cm² yr⁻¹
  (refs 7, 16, 23)", i.e. 0.005 m² yr⁻¹, Reneau (1988) among the sources. Recorded as a check only.
- **Relation to the M&D area.** The sampled ridges and the two tributaries are in the Tennessee Valley catchment on the
  eastern flank of Coyote Ridge, the same hillslopes M&D mapped; the paper gives no coordinates beyond the footnote.
- **Value to carry.** The basin-averaged rates bracket E at 64-102 m Myr⁻¹ (6.4-10.2 × 10⁻⁵ m yr⁻¹); the point rates on the
  divergent ridges cluster at 20-50 m Myr⁻¹. Which one the TTV tiles should use (a catchment-averaged E like GM and AP, so
  the mean of the three detrital samples, 77 ± 15 m Myr⁻¹, or the two tributaries separately) is a decision for Steve (§10).

## 3. References per site (decision 6)

**GM and AP: found in the repo.** `legacy/measurement_notebooks/updated_best_fit_tile_finder.ipynb`, cell 1, dictionary
`TAYLOR_REF` (the notebook labels it "Taylor et al. (2006)"; the site configs call the same numbers "Perron et al. (2012)"):

| site | θ | k_s | curvature (stored negative) | A_crit (m²) | U (m Myr⁻¹) | K (Myr⁻¹) | weights of the three-term RMS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GM | 0.292 | 1.9 | −1.28 × 10⁻² | 1.0 × 10⁴ | 147 | 100.2 | curvature 5, k_s 1, θ 1 |
| AP | 0.54 | 47.2 | −5.93 × 10⁻³ | 5.04 × 10⁴ | 55 | 12.5 | curvature 5, k_s 1, θ 1 |

Where each number came from:
- **θ and k_s**: `chi_optmize.ipynb` (cells 0-2 and their outputs), the χ concavity optimisation with the trapezoid χ traversal
  from every coastline outlet on "taylors_original_dataset" grids: **AP** `AP_filled` (30 m, 2,202 × 3,010 cells, larger than
  the `AP_Lidar_overlap_30m` grid used for the search) gave θ* = 0.537 (refined from 0.525) over 118,379 outlets and "ks
  median = 47.1781" (the median of per-basin through-origin slopes, n = 1,820 basins), rounded to 0.54 and 47.2; **GM**
  `GM_10mN_filled` (the northern 10 m grid, 2,997 × 2,297 cells = the `GM_Lidar_overlap_10m` grid) gave θ* = 0.292 (from 0.300)
  over 441,765 outlets and ks median = 1.9142 (n = 16,010), rounded to 0.292 and 1.9. (A southern GM grid, `GM_10mS`,
  3,881 × 3,783 cells, gave θ* = 0.237 and ks 0.53 and was not used.) The per-basin χ fit used A ≥ A_MIN with the Perron A_min
  (5.04 × 10⁴ and 10⁴ m²), A₀ = 1 m², z relative to the outlet. So the reference k_s is a **median of per-basin slopes**, in
  m^(2θ) with θ = 0.54 (AP) and 0.292 (GM); the pooled slope was not the statistic.
- **Curvature**: the notebook's cell 0 (`measure_curvature_targets.py`) computes "whole-dataset hilltop curvature targets
  (median |κ|)" on the same two coarse grids with `topo_analysis.extract_hilltop_curvature_first_order` (first-order divides
  of the A_crit network, 9-point stencil) and writes `curvature_targets.json`; its output is not stored in the notebook. The
  stored values (12.8 × 10⁻³ and 5.93 × 10⁻³ m⁻¹) differ from Perron et al. (2012, SI Table S2) (11.8 ± 0.4 × 10⁻³ and
  5.5 ± 0.1 × 10⁻³, "Ref. 21") by +8 % and +8 %, which is consistent with their being the notebook's own coarse-grid medians
  rather than Perron's numbers. The site configs' `hilltop_curvature` entries (1.18 × 10⁻² and 5.5 × 10⁻³) are Perron's.
- **The rebuild of the addendum** (regional θ by χ on the NED, GM 0.292 and AP 0.54; |∇²z_h| from Perron; k_s = E/K with
  K = D ÷ (D/K) and D = E/|∇²z_h|) was therefore not needed for θ; for completeness, E/K from Table S2 gives k_s = 55/12.5 = 4.4
  (AP) and 147/100.2 = 1.47 (GM) in Perron's units (m^(2m) with m = 0.37 and 0.35), which are not the notebook's k_s and are not
  comparable to it (different θ, different definition).
- **The same code on the same grids today** (§3.3 code, whole valid coarse grid, data-edge outlets; `pools/<site>_pools.json`
  → `whole_grid`): **AP** θ (legacy χ, 1,271 outlets with channels) **0.571**, k_s(0.54) median **39.5** (pooled 37.0),
  |C_ht| **5.82 × 10⁻³**; **GM** θ (legacy χ, 761 outlets) **0.354**, k_s(0.292) median **1.66** (pooled 2.11), |C_ht|
  **15.2 × 10⁻³**. AP reproduces the legacy reference to within θ +0.03, k_s −16 %, |C_ht| −2 %; GM does not: θ +0.06,
  k_s −13 %, curvature +18 %. The legacy GM θ came from TopoAnalysis's fill, D8 and area on the same grid with every ring
  cell as an outlet and 441,765 "basins", and the legacy curvature from TopoAnalysis's divide extraction, which the lemcvt
  port approximates (`legacy_v1.py` docstring); which reference the tolerance bands should use is decision 3 of section 10.

**TTV: no published reference.** The M&D study area was approximated (no polygon on Sherlock) as the catchment of the main
Tennessee Valley channel above the cell nearest 61 m elevation near the trailhead (outlet at 540,875 E, 4,190,285 N, z = 61.4 m,
A = 0.67 km²): 6,729 cells of the 10 m NED, 0.67 km², 61-317 m. It covers the eastern flank of Coyote Ridge but only a third of
the 2.1 km² M&D area, which also held headwaters draining to the Bay. On this proxy the same code gives **θ = 0.387 (coarse grid
0.375), k_s(0.387) median 15.4 (pooled 15.4, 15 basins), |C_ht| = 15.1 × 10⁻³ m⁻¹**. These are the TTV reference used for the
tolerance counts, flagged provisional until Steve's polygon replaces the proxy (§10).

## 4. Candidate pools (section 3.1) — counts per lattice offset

Windows are W × W coarse cells, W = round(1000 m / cell): 100 cells (1,000 m) at GM and TTV, 33 cells (990 m) at AP; the four
offsets are 0 and W/2 cells in each axis (500 m at GM and TTV; 16 cells = 480 m at AP). A window is a candidate when every
coarse cell is valid and inside the lidar footprint (3.1): GM `Gabilan_mesa.tif` (every 1 m cell valid in every 10 m cell);
AP the 3DEP tile index and, for decision 7, the PAMAP watershed (every 30 m cell holds PAMAP data; see §1 point 5); TTV the
Marin 2019 (B5b) tile index, which covers 99.999 % of the NED land cells in the 18 × 18 km domain. Appendix A1.

| site | lattice windows per offset | coarse grid fully valid | **candidates** (offsets 0, 1, 2, 3) | notes |
| --- | --- | --- | --- | --- |
| GM | 638 | 29-31 | **17, 19, 18, 17 (71)** | footprint 39.2 km² of fully valid 10 m cells; the coarse grid is valid over 58 km² |
| AP | 2,106 | 841-856 | **414, 411, 414, 414 (1,653)** | 761-775 windows lie in the 3DEP index with a valid coarse grid; 414 of them inside the PAMAP watershed |
| TTV | 289-324 | 99-110 | **110, 108, 103, 99 (420)** | all land windows of the domain; only 5-8 per offset also lie inside the NCALM 2006 raster |

Routing on the full coarse grids (fill, D8, area; data-edge cells are the outlets, §1 point 9): GM 583,283 valid cells,
largest basin 33.4 km²; AP 1,152,659 cells, 179 km²; TTV 1,524,547 land cells, 45.3 km². The M&D proxy and the whole-grid
references in §3 come from these grids.

## 5. Screens (section 3.2): distributions, proposed thresholds, windows removed

Statistics per candidate window (Appendix A2) and the two threshold sets (A: `scripts/a14_thresholds.json`, proposed; B:
`scripts/a14_thresholds_B.json`, relaxed, shown so that the cost of each rule is visible; Appendix A3 for both):

| screen | statistic (window) | GM median (p05-p95) | AP | TTV | set A | set B | reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 trunk | largest drainage area inside the window (m², full-grid routing) | 2.5 M (0.40-22 M) | 2.8 M (0.49-61 M) | 1.2 M (0.25-14 M) | ≤ 5 × 10⁶ | ≤ 5 × 10⁶ | a 5 km² channel at the paper's sites is a trunk valley (GM2, GM4, AP3, AP4 hold 6-17 km² trunks; GM1, GM3, GM5, AP1, AP2, AP5 hold ≤ 4.7 km²); the paper's rule is "complete tributary networks, exclude trunk channels" |
| 1 trunk | largest single inflow across the window edge (m²) | 1.7 M (0.13-22 M) | 1.7 M (0.03-59 M) | 0.46 M (0.02-11 M) | ≤ 1 × 10⁶ | ≤ 2 × 10⁶ | a channel entering with more than 1-2 km² upstream is a through-flowing trunk whose network is not complete inside the window |
| 2 transient | χ-z R² (through origin, at θ_ref) per edge-outlet basin, basin-area-weighted | 0.980 (0.92-0.996) | 0.975 (0.66-0.994) | 0.971 (0.76-0.993) | ≥ 0.95 | ≥ 0.90 | the submitted paper kept tiles "only where χ-elevation profiles were approximately linear"; 0.95 keeps the upper 90 % at GM, 0.90 the upper 85-90 % at AP and TTV |
| 2 transient | slope-area break, ΔBIC = BIC(one segment) − BIC(two) above 10³ m², area-weighted | −1.4 (−7.4 to +20.5) | +3.6 (−9.3 to +45) | +8.3 (−7.8 to +116) | ≤ 10 | ≤ 20 | ΔBIC > 10 is "very strong" evidence for a break on the Kass-Raftery scale; AP and TTV windows show breaks far more often than GM (a 990 m window at 30 m holds few channel pixels, so the AP statistic is noisier) |
| 3 disturbance | NLCD 2021 fraction of classes 11, 21-24, 31 (water, developed incl. open space, barren) | 0 (0-0.07) | 0.073 (0.02-0.17) | 0.10 (0-0.98) | ≤ 0.05 | classes 11, 22-24, 31 ≤ 0.10 (open space 21 dropped) | class 21 "developed, open space" is lawns, farm roads and road verges at 30 m and covers 5-10 % of nearly every AP window; set B keeps built-up classes only |
| 3 disturbance (AP) | PA DEP fraction: longwall panels, mined-out underground, historic surface permits | - | 0.70 (0-1.0); longwall alone 0.006 (0-0.86) | - | all three = 0 | longwall panels = 0 | longwall subsidence (up to 1-2 m) is the deformation that matters; room-and-pillar "mined-out" areas and surface permits cover most of Greene County and set A would remove 73 % of windows for them |
| 4 independence | window A drains into window B (direct edge, any cell of A reaches B's edge outlets) | degree 4 (2-5) | 5 (3-6) | 4 (2-6) | graph reported | graph reported | the draw respects it; the transitive closure (all upstream-downstream chains) joins almost every window to every other along the trunk valleys and is reported, not used |
| decision 7 | upstream area fully in valid data (no cell of the window has an upstream cell on the footprint edge) | 46 of 71 pass | n/a (watershed) | 305 of 420 | required GM, TTV | required GM, TTV | decision 7 |

Windows removed (alone, and jointly) — Appendix A3:

| site | candidates | set A: each screen removes | set A joint survivors | set B joint survivors |
| --- | --- | --- | --- | --- |
| GM | 71 | trunk 23, inflow 45, R² 8, ΔBIC 12, NLCD 5, upstream 25 | **21** (5-6 per offset) | **38** (8-10 per offset) |
| AP | 1,653 | trunk 587, inflow 976, R² 475, ΔBIC 619, NLCD 1,269, mining 1,213 | **17** (3-5 per offset) | **320** (70-85 per offset) |
| TTV | 420 | trunk 65, inflow 147, R² 149, ΔBIC 194, NLCD 274, upstream 115 | **33** (6-11 per offset) | **102** (22-30 per offset) |

## 6. Selection metrics (section 3.3) and the eligible count

Per candidate window, by the same code at all sites (Appendix A4): θ by the legacy χ method (tile-edge outlets, A ≥ A_c
with A_c = 10⁴ m² at GM and TTV and 5.04 × 10⁴ m² at AP, θ grid 0.10-0.80 then refinement); k_s at the site reference θ
(median of per-basin through-origin slopes, the legacy statistic; the pooled slope beside it); |C_ht| by the coarse-grid
method (3 × 3 Laplacian at first-order divides of the A_c network, median of convex pixels, saddles and flats excluded).

| metric | GM: median (p05-p95) of 71 | reference | AP: of 1,653 | reference | TTV: of 420 | reference (proxy) |
| --- | --- | --- | --- | --- | --- | --- |
| θ | 0.387 (0.25-0.50) | 0.292 | 0.592 (0.13-0.80) | 0.54 | 0.340 (0.10-0.59) | 0.387 |
| k_s(θ_ref), median of basins | 2.38 (1.8-3.4) | 1.9 | 38.2 (19.8-58.9) | 47.2 | 12.7 (1.0-23.6) | 15.4 |
| |C_ht| (10⁻³ m⁻¹) | 13.8 (9.3-17.5) | 12.8 | 5.0 (3.2-6.9) | 5.93 | 20 (6.5-30) | 15.1 |
| basins in the χ fit | 23 (11-44) | | 10 (3-26) | | 39 (17-69) | |

**Within tolerance (decision 5), alone and jointly with the screens** (Appendix A4, A4b):

| site | θ ± 0.05 | k_s ± 25 % | |C_ht| ± 25 % | k_s and C_ht | **all three** | all three and set A screens | all three and set B screens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GM (71) | 20 | 34 | 56 | 23 | **9** | **4** (1, 0, 2, 1 per offset) | **4** |
| AP (1,653) | 358 | 795 | 1,012 | 651 | **181** | **0** | **13** (4, 2, 4, 3) |
| TTV (420) | 92 | 189 | 122 | 46 | **12** | **1** | **3** |

**Against the target of 30 (25 new at GM and AP).** With the thresholds as proposed and the ±0.05 θ tolerance, the expected
eligible counts are GM 4, AP 0-13, TTV 1-3: far short at every site. The binding constraint is θ: the legacy χ method on a
1 km window uses 10-40 small basins and its θ scatters by ±0.09 (GM) to ±0.2 (AP, TTV) around the site value, so a ±0.05
band keeps only 28 % (GM), 22 % (AP) and 22 % (TTV) of candidates before any screen. The sensitivity table (A4b) gives the
alternatives: with θ unconstrained and k_s, |C_ht| within ±25 %, the eligible counts after set B screens are GM 14, AP 68,
TTV 20; with ±50 %, GM 33, AP 276, TTV 52; θ ± 0.10 with ±25 % gives GM 7, AP 29, TTV 6. The conflict graph (direct degree
2-8) will remove more at the draw.

## 7. The original ten tiles against the new rules (section 3.4; Appendix A5)

| tile | trunk inside (km²) | largest edge inflow (km²) | χ R² (w) | ΔBIC (w) | NLCD disturbed | mining (all / longwall) | passes set A screens | θ | k_s(θ_ref) | |C_ht| ×10⁻³ | within tolerance |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 1.06 | 0.12 | 0.994 | 49.0 | 0.043 | 1.00 / see A5 | no (ΔBIC, mining) | 0.621 | 32.6 | 5.09 | C_ht only |
| AP2 | 1.67 | 0.99 | 0.983 | 46.6 | 0.048 | 0.87 | no (ΔBIC, mining) | 0.780 | 40.0 | 4.81 | k_s, C_ht |
| AP3 | 10.7 | 8.7 | 0.990 | −10.5 | 0.064 | 0.088 | no (trunk, inflow, NLCD, mining) | 0.512 | 54.0 | 4.71 | **all three** |
| AP4 | 16.6 | 16.6 | 0.980 | 14.1 | 0.043 | 0.00 | no (trunk, inflow, ΔBIC) | 0.692 | 36.6 | 5.94 | k_s, C_ht |
| AP5 | 1.23 | 0.60 | 0.993 | 10.6 | 0.073 | 1.00 | no (ΔBIC, NLCD, mining) | 0.433 | 32.9 | 6.26 | C_ht only |
| GM1 | 1.15 | 0.90 | 0.995 | 2.5 | 0.000 | - | **yes** | 0.362 | 2.33 | 17.1 | k_s only |
| GM2 | 8.88 | 8.47 | 0.982 | −4.8 | 0.000 | - | no (trunk, inflow) | 0.367 | 2.11 | 13.9 | k_s, C_ht |
| GM3 | 0.34 | 0.002 | 0.983 | −5.3 | 0.018 | - | **yes** | 0.262 | 2.05 | 12.0 | **all three** |
| GM4 | 6.20 | 5.16 | 0.990 | 6.7 | 0.000 | - | no (trunk, inflow, upstream; coarse grid has 2 nodata cells) | 0.321 | 3.56 | 15.1 | θ, C_ht |
| GM5 | 4.73 | 2.66 | 0.980 | −9.3 | 0.000 | - | no (inflow) | 0.312 | 2.20 | 17.0 | θ, k_s |

So of the ten, two (GM1, GM3) pass the set A screens and two (AP3, GM3) meet all three tolerances; only GM3 does both. The
AP tiles sit on mined ground: AP1 and AP5 lie entirely, AP2 87 %, over PA DEP mined-out or longwall areas (AP3 9 %, AP4 0 %).
They stay in the set (decision 2); the table says what the new rules would have done.

## 8. Full-grid data (section 4)

### 8.1 AP footprint, file lists and sources (4.1.1-4.1.3)

**AP watershed footprint (4.1.1).** PAMAP's valid area on the 30 m lattice is one polygon of 516.6 km² (the single watershed
clip of addendum 13); buffered by 500 m it is 578.6 km², envelope UTM 17N E 549,464-585,052, N 4,405,649-4,430,794. In
PA South ftUS it meets **294 quadrants** (`footprints/footprints.json`): all 294 are in the PASDA listing and in the 5,000 ft
index with polygons equal to the naming rule; 28 are already on Sherlock (addendum 13); **57.45 GB zipped (gate 60 GB: pass),
52.08 GB remaining**, about 87 GB as LAS. The 3DEP tile index covers 100 % of the PAMAP footprint.

**Sources (4.1.2).**
- AWS `usgs-lidar-public/PA_WesternPA_1_2019`: present (Entwine 1.1.0, 117.9 G points, EPSG:3857, Z unchanged, no vertical
  CRS). Its source list points to 7,254 delivered LAZ on `s3://usgs-lidar` (requester pays) which rockyweb mirrors
  (`PA_WesternPA_2019_D20/PA_WesternPA_1_2019/LAZ/`, 902 GB); **300 LAZ tiles, 39.37 GB, 3.80 G points** intersect the buffered
  footprint. Equality test for quadrant 19001270PAS_NE: §8.2.
- PASDA: 266 remaining zips started at 17:03 PST with 6 streams (job 52736647): the first 1.8 GB arrived at 8-37 MB/s, then the
  rate fell to about 60 KB/s per stream (0.35 MB/s aggregate), as in addendum 13.
- rockyweb: about 75-170 KB/s per connection from compute nodes; HTTP 429 to bursts of new connections (12-stream job
  52735027 cancelled after 43 refusals; the system curl 7.29 does not retry 429).

**Gridding and routing pilot (4.1.5, decision 9) on AP1 + 500 m** (`grid_AP1_smoke/`, job 52738203/52739892): the six AP1
quadrants (42.27 M class-2 points; headers: compound NAD83(2011)/PA South ftUS + NAVD88 ftUS Geoid12B, converted with
1200/3937) binned to a 2,001 × 2,001-cell 1 m lattice in EPSG:6346 (integer-metre origin 567,998 E, 4,409,836 N): class-2
density 3.17 pts m⁻² (5-50-95 % of points per cell 2-3-5), **1.50 % of cells inside the footprint empty and filled by nearest
neighbour** (fill distance median 1 m, 95 % 2 m, max 9 m); 47 s to grid, 5 s to fill and route (0.65 GB). **Rule 5.1 on AP1**:
median Δz (new grid − PAMAP tile, metres) −0.073 m against the addendum 13 point check −0.079 m; Theil-Sen slope 0.9998
(OLS 0.9999); 95.4 % of the tile's cells compared: **pass**. The pilot is the full pipeline that the whole-watershed run uses.

### 8.2 AP (4.1): the equality test, the fetch, the 1 m grid and its routing

- **Footprint and file lists** (4.1.1-4.1.3): §8.1; Appendix A10 lists the 294 PASDA quadrants and the 300 USGS
  LAZ tiles with sizes.
- **Equality test, quadrant 19001270PAS_NE** (4.1.2; `usgs_vs_pasda/`, jobs 52742762, 52746580; Appendix A11). PASDA class-2
  points inside the quadrant square: 6,877,258.
  - *AWS Entwine copy* (`usgs-lidar-public/PA_WesternPA_1_2019`, EPSG:3857, 117.9 G points): read in 18-22 s with PDAL
    `readers.ept`, reprojected to EPSG:6565: **6,877,263 points** (+5, boundary duplicates after reprojection); of 200,000 PASDA
    sample points **100 % have an Entwine point within 0.05 ft (median 0.012 ft = 3.6 mm, 95 % 0.021 ft)**, 40 % within 0.01 ft
    (the 0.01 m quantisation of the octree is 0.025 ft on the ground); the heights are the same points in **metres**
    (median ratio 0.3048019 against 1200/3937 = 0.3048006). The sha256 of the sorted coordinates rounded to 0.01 ft differs.
  - *rockyweb LAZ* (the "delivered" tiles): the three tiles that arrived intact (one of four came back as an HTTP 429 body twice,
    2,436 s of wall time for 410 MB) are LAS 1.4 pf6, QSI software, **NAD83(2011) / UTM 17N metres + NAVD88 Geoid12B metres**
    (so USGS re-projected and re-tiled PASDA's State Plane feet delivery as well); 6,412,097 class-2 points in the quadrant from
    3 of 4 tiles; 93.3 % of the PASDA sample matched within 0.1 ft (the rest in the missing tile), median 0.008 ft (2.5 mm).
  - **Reading of the rule.** By the addendum's literal test (count and checksum) neither USGS copy equals PASDA, so PASDA would
    be the source. PASDA (and rockyweb) delivered the first 1.8 GB in 90 s and then 0.05-0.4 MB/s for the rest of the evening
    (14 of 266 zips in 72 min; job 52736647, cancelled), which puts the 52 GB at days. The Entwine copy is the same point set
    to within 1 cm horizontally and 1 cm in height (its quantisation), from S3 at 12 MB/s. **The watershed grid below is built
    from the Entwine copy; whether that stands is decision 6 of section 10.** The fetched LAZ carry the compound CRS
    EPSG:6346 + 5703 written from this evidence (the Entwine schema has no vertical CRS); nothing was converted on an assumption.
- **Fetch** (job 52746468, `ept_AP/`): 642 1 km boxes over the 578.6 km² buffered footprint, class 2 only, reprojected to
  EPSG:6346, LAZ 0.001 m scale: **1,924.4 M ground points, 12.11 GB, 1,005 s (12.1 MB/s), 642 of 642 boxes**. Copy of record
  `source/AP-3DEP/ept_class2_laz_utm17n/` (642 LAZ, 14 GB on disk, sha256 manifest verified; job 52748077).
- **Rules 5.1 and 5.2 of addendum 13 on the new files** (4.1.4): the naming rule does not apply (USGS 1 km boxes named by
  their south-west UTM corner, written here); units: every header states metres horizontally and vertically (as written), the
  heights' metre unit being the independent result above; the agreement test is rule 5.1 below.
- **Grid** (4.1.5, decision 9; job 52746471 first pass, 52750908 with the 10 m fill limit; `grid_AP/`): lattice 25,157 × 35,600
  cells at 1 m, origin 549,458 E, 4,430,800 N (EPSG:6346); 578.6 M cells inside the buffered footprint; **class-2 density 2.99
  points m⁻² (points per cell 5-50-95 %: 1-3-5)**; collection dates **2019-11-18 (217 boxes) and 2019-11-21 (425 boxes)** from the GPS times (adjusted standard GPS time inferred from magnitude, the
  writer's header flag being unset). Of the cells inside the footprint, 97.0 % hold at least one ground point; **3.02 % were empty and filled by nearest neighbour within 10 m** (fill distance median 1 m, 95 % 5 m), and **0.29 % (1.66 M cells) lie farther than 10 m from any ground point and were left nodata** (ponds, pits, the river margin); 576.97 M valid cells remain. The first pass had filled those far cells too (fill distances to 255 m), which decision 9 did not intend; the 10 m limit is the rule used for both sites. Peak memory 36.7 GB for the
  gridding (two 7.2 GB accumulators and the nearest-neighbour transform), 579 s.
- **Routing** (decision 7; `a14_bigroute`, the lemcvt conventions, verified identical to the lemcvt functions cell for cell
  on a test grid): priority flood 369 s, D8 83 s, area 190 s, 46.9 GB peak; one grid, no sub-watershed split; the data-edge
  outlets' areas sum exactly to the valid cells (closed budget) in both passes.
  - *First pass* (job 52746471, every empty cell filled by nearest neighbour, up to 255 m): 144,204 outlets, all on the
    footprint edge; **the outlet at 582,500 E, 4,425,636 N drained 515.5 M cells, 89.1 % of the grid** (the watershed; the
    rest is the 500 m buffer), the next outlets 1.5 and 0.9 M cells.
  - *Second pass* (job 52750908, the 10 m fill limit, the products kept): the 1.66 M gap cells (ponds, pits, the wider river
    reaches without ground returns) are nodata inside the grid, so their rims become outlets too: 270,490 outlets; the largest
    basin is 157.9 M cells (27 %, outlet at 566,172 E, 4,415,230 N), the next 101.1 M (18 %) and 63.0 M (11 %). Hillslope and
    tributary areas upstream of any gap are the same in both passes; only the trunk accumulation downstream of a gap differs.
    Which treatment of water gaps stands (fill across them, or mask them) is decision 8 of section 10.
  Products (`grid_AP/`, second pass): `AP_1m_mean.tif`, `AP_1m_filled.tif`, `AP_1m_fd.tif`, `AP_1m_area_cells.tif` (uint32
  cell counts), `AP_1m_class2_count.tif`, `AP_1m_filled_mask.tif` (1 = NN-filled, 2 = gap left nodata), `AP_1m_nn_fill_dist.tif`.
- **Pilot** on AP1 + 500 m (`grid_AP1_smoke/`): the same pipeline on the six addendum 13 quadrants (US survey feet converted
  with 1200/3937) gave density 3.17 m⁻², 1.5 % filled cells (median 1 m, max 9 m) and rule 5.1 on AP1 −0.073 m / 0.9998, the
  same as the watershed grid to 2 mm: the feet-to-metres path and the Entwine path agree.

### 8.3 TTV (4.3): Marin 2019 QL1 from the USGS Entwine copy
- **Tile index and metadata** (4.3.1): the Lot 12 tile index (6,759 1,000 m tiles, Albers NAD83(2011)) and buffered project
  boundary from rockyweb (`metadata/.../CA_NoCal_Wildfires_B5b_QL1_2018/shapefiles/`), the best-use metadata XML
  (`Northern_California_QL1_ClassifiedPointCloud.xml`) and the Entwine source list (1,799 LAZ) are copies of record under
  `source/TTV/Marin2019_B5b_ancillary/`. The index covers 99.999 % of the NED land cells in the TTV domain.
- **Pool watersheds**: the 420 candidate windows drain to 2,994 terminal basins of the 10 m grid whose union is 138.1 km²;
  buffered by 500 m, 184.3 km² (`marin_select/TTV_pool_watersheds_buffer500.geojson`). 237 index tiles intersect it, 198 of
  which are named in the Entwine source list (4.15 G points; the rockyweb listing was rate-limited, so the listed sizes
  could not be summed; the point count puts the LAZ at about 14.5 GB, under the 30 GB gate).
- **Fetch** (job 52746473): 235 1 km boxes read from `USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018/ept.json` at full resolution,
  class 2 only, reprojected to EPSG:6339 and written as LAZ with the compound CRS EPSG:6339+5703 (NAVD88 metres, the unit the
  best-use metadata states for the source; verified by rule 5.2): **1,098.5 M ground points, 7.21 GB, 816 s (8.8 MB/s)**, 235 of
  235 boxes. Copy of record `source/TTV/Marin2019_B5b_ept_class2_laz_utm10n/` (235 LAZ, 7.8 GB, sha256 manifest).
- **Headers** (4.3.2): LAS 1.4, point format 6, horizontal NAD83(2011) / UTM 10N metres, vertical NAVD88 metres (as written
  from the evidence above; the Entwine copy itself carries EPSG:3857 and no vertical CRS); the source LAZ are Albers metres per
  the metadata (not fetched). GPS times are adjusted standard GPS time (values 2.3 × 10⁸): class-2 flight dates per 1 km box
  in the rerun (`grid_TTV/TTV_grid.json`).
- **Grid** (4.3.3, decision 9): lattice 19,010 × 19,010 at 1 m (origin 529,495 E, 4,198,505 N, EPSG:6339); 184.3 M cells inside
  the buffered footprint; **class-2 density 5.6 points m⁻² (5-50-95 % of points per cell 1-8-20)**; 120.7 M cells hold a
  ground point. In the first pass every empty cell inside the footprint was filled by nearest neighbour (35 %, median
  distance 248 m: the 500 m buffer reaches into the Bay and the Pacific, where there are no ground points); the rerun fills
  only within 10 m and leaves farther cells nodata (numbers below). Fill 40 s, D8 32 s, area 55 s, 18.6 GB peak.
- **Routing**: 184.3 M valid cells, 103,485 data-edge outlets whose areas sum to the valid count (closed budget); the largest
  outlet (31.3 km², 17 % of the grid) is at 543,281 E, 4,192,810 N (Richardson Bay side), the second (23.1 km²) at
  537,080 E, 4,189,530 N (Tennessee Cove / Rodeo side).
- **Class-2 density over the pool** (4.3.2): on the 420 candidate windows, 6.9 points m⁻² (5-95 %: 0-11.6), in hollows
  (Laplacian > +0.02 m⁻¹ at 11 m, the addendum 13 stand-in) 5.9 (0-10.1); empty 1 m cells 14.8 % per window (19 % in
  hollows) — the QL1 survey's 0.35 m nominal pulse spacing gives about 8 pulses m⁻², of which the ground share is 5-7.

### 8.4 GM (4.2)
No new data. The lidar (`Gabilan_mesa.tif`) covers 39.6 km² of the 10 m grid's 58.3 km² valid area; 39.2 km² (391,950 10 m
cells) are fully valid (every 1 m cell). Of the 71 candidate windows **46 have their whole upstream area in valid data** and
25 do not (`upstream_fully_valid` in `pools/GM_windows.csv`; Appendix A2); GM4's window fails it (two nodata cells in the
coarse grid and an upstream edge contact), GM1, GM2, GM3 and GM5 pass.

### 8.5 Cost (section 6, decision 12)
The estimate printed before the first submission was "a few core-hours of compute; downloads dominated by I/O at 1 CPU"
(the stage 1 and pools jobs took 2-23 minutes each). Actual (Appendix A13, `sacct`): **15.9 core-hours used** across 37
submissions (cap 60); every job on `serc`, no array needed; largest single submission 96 core-hours of limits (8 CPUs × 12 h,
cap 300). **The sum of CPU-weighted time limits is 973 core-hours, eight times the 120 cap.** Three causes, all mine: the
fetch and grid jobs were given 6-12 h walls at 3-8 CPUs when they needed 12-28 min (the 36 h and 30 h single-CPU download
jobs, cancelled, add 66); each of three bugs in the fetch chain (OGR feature lifetime, PROJ data path after the numba install,
metadata key) cost a resubmission of a 96 + 40 core-hour pair; and the grid jobs were re-run once for the 10 m fill limit.
Counted in wall-clock hours the submitted limits sum to 248 h. New data on group storage: 19.5 GB of LAZ, 0.3 GB TTV rasters,
about 14 GB of 1 m grids in the run directory (cap 100 GB); `$GROUP_HOME` 305.6 GB of 1 TB after the copies.
Scratch left behind (purged in 90 days): `$SCRATCH/a14/` holds the EPT LAZ (19 GB, duplicates of the copies of record,
not deleted: the managed settings refused the delete while Steve was away), the 14 PASDA zips (2.2 GB), the comparison files and the NLCD/PA DEP downloads.

## 9. Unit and datum checks (section 5; Appendix A8)

| rule | comparison | cells | median Δz (m) | 5-95 % | Theil-Sen slope | criterion | result |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 5.1 AP1 | new watershed grid (Entwine, metres) − PAMAP tile (ftUS × 1200/3937) | 982,830 | −0.071 | −0.41, +0.28 | 0.9997 | within 0.1 m of addendum 13's −0.079; slope 0.99-1.01 | **pass** |
| 5.1 AP2 | | 952,978 | −0.073 | −0.32, +0.14 | 1.0001 | a13 −0.077 | **pass** |
| 5.1 AP3 | | 971,593 | −0.066 | −0.40, +0.24 | 0.9993 | a13 −0.066 | **pass** |
| 5.1 AP4 | | 979,177 | −0.021 | −0.36, +0.26 | 1.0006 | a13 −0.026 | **pass** |
| 5.1 AP5 | | 956,091 | −1.213 | −1.80, −0.20 | 0.9891 | excluded (addendum 13, 6.2); a13 −1.206 | reproduced, excluded |
| 5.2 TTV | Marin 2019 grid (ground cells) − NCALM 2006 raster | 12,813,875 | +0.084 | −0.33, +0.41 | 0.99999 | median within ±1.0 m, slope 0.98-1.02; stop if slope ≈ 3.28/0.305 or offset ≈ −32 m | **pass, no stop** |
| 5.3 GM | 10 m NED − 1 m lidar block-mean | 378,600 | +32.27 | +26.2, +38.0 | 0.9993 | offset 30-34 m (ellipsoidal lidar), slope 0.98-1.02 | **pass** |
| 5.3 AP | 30 m NED − PAMAP block-mean (ftUS × 1200/3937) | 523,526 | +1.17 | −5.6, +12.4 | 0.9860 | |offset| < 5 m, slope 0.98-1.02 | **pass** (block medians −4 to +7 m: NED quality, no units signal) |
| 5.3 TTV | 10 m NED (EPSG:6339 warp) − NCALM 2006 block-mean | 151,369 | +0.23 | −0.45, +1.11 | 1.0003 | |offset| < 5 m, slope 0.98-1.02 | **pass** |

The 5.1 medians equal addendum 13's point-based values to 2-8 mm, so the Entwine heights, converted by the octree to
metres, are the PASDA heights (US survey feet × 1200/3937) within the comparison's resolution; the AP5 band offset of
addendum 13 is reproduced exactly (−1.213 against −1.206 m) and remains unexplained. No grid was edited or filtered.

### 9.1 Rule 5.2 in detail: the Marin 2019 grid against the NCALM 2006 raster (ground cells only, 12.81 M cells)
median Δz (Marin − NCALM) **+0.084 m** (5-95 %: −0.33 to +0.41 m; RMS 0.44; |Δz| > 1 m on 0.9 %), Theil-Sen slope **0.99999**
(OLS 0.99990); 10 × 10 block medians −0.13 to +0.22 m with no banding; slope nowhere near 3.28 or 0.305, offset nowhere near
−32 m: **pass, no stop**. The two surveys (2006 NCALM, 2019 QL1) agree to within a decimetre on the same orthometric datum,
which also settles the NCALM raster's inferred vertical datum (NAVD88, metres).

## 10. Decisions needed

1. **Thresholds (3.2).** Approve set A, set B, or a mix. The consequences are in §5-§6 and A3-A4b. The two rules that
   decide AP are the NLCD class list (open space in or out) and the mining layers (all three or longwall only); the rule that
   decides TTV is the NLCD fraction (the Headlands pool borders Sausalito, Marin City, Mill Valley and Fort Baker).
2. **The θ tolerance (decision 5)** cannot deliver 25 windows per site with the legacy window-level θ at any site. Options:
   (a) keep ±0.05 and accept GM 4, AP 0-13, TTV 1-3; (b) widen to ±0.10 (GM 7, AP 29, TTV 6 after set B screens);
   (c) drop θ from the selection metrics, since the models use the site constant m anyway (addendum 5), and select on k_s
   and |C_ht| alone (±25 %: GM 14, AP 68, TTV 20; ±50 %: GM 33, AP 276, TTV 52); (d) replace the per-window legacy θ by a
   pooled estimator over the window's basins (not computed here).
3. **The reference for GM and AP.** The legacy TAYLOR_REF values were used as the addendum says; the same code on the same
   coarse grids today gives θ 0.354 / k_s 1.66 / |C_ht| 15.1 × 10⁻³ at GM and 0.571 / 39.5 / 5.8 × 10⁻³ at AP (§3). Whether
   the tolerance bands should be centred on the legacy numbers or on the same-code numbers (TTV can only use the latter)
   changes GM's eligible count (A4b).
4. **The TTV reference polygon.** The M&D proxy (0.67 km² catchment above 61 m) is a third of the 2.1 km² study area; Steve's
   polygon would replace it and change the TTV reference (θ 0.387, k_s 15.4, |C_ht| 15.1 × 10⁻³).
5. **TTV erosion rate**: which Heimsath et al. (1997) value (§2.3).
6. **AP full-grid source**: the AWS Entwine copy passes the content test (same points within 1 cm, Z in metres) but not the
   literal checksum test; PASDA and rockyweb deliver at 0.05-0.4 MB/s (days for 52 GB). The watershed grid here is built from
   the Entwine copy (§8); Steve decides whether that stands or whether the PASDA download (14 of 266 zips received) resumes.
7. **What to do if a site has fewer than 25 eligible windows** (the addendum's question): fewer tiles (decision 1 allows it),
   the relaxations in 2 and 1, or a different window size/offset scheme (overlapping offsets cannot be drawn together).
8. **Water gaps in the 1 m grids** (decision 9): fill them by nearest neighbour whatever the distance (first pass: one outlet
   for the AP watershed, artificial flat surfaces over ponds and river reaches), or leave cells farther than 10 m from a ground
   point as nodata (second pass, the products kept: 0.29 % of AP cells, 24 % of the TTV footprint which is the Bay and the ocean;
   gap rims become outlets). For the tiles' own drainage areas the two agree except downstream of a gap.

## 11. Appendices

Generated by `reports/v2/scripts/make_a14_tables.py` from the run directory (`appendix_tables_A.md`, `appendix_tables_B.md` there). Set A tables in full; for set B only A3, A4 and A4b.

### A1. Windows per lattice offset (section 3.1), from `pools/<site>_pools.json`

| site | offset | offset (cells, m) | lattice windows | coarse grid all valid | candidates (footprint 100 %) | AP: in 3DEP index / TTV: also in NCALM raster | AP: inside PAMAP watershed |
|---|---|---|---|---|---|---|---|
| GM | offset_0 | [0, 0] cells (0, 0 m) | 638 | 29 | 17 | - | - |
| GM | offset_1 | [0, 50] cells (0, 500 m) | 638 | 30 | 19 | - | - |
| GM | offset_2 | [50, 0] cells (500, 0 m) | 638 | 30 | 18 | - | - |
| GM | offset_3 | [50, 50] cells (500, 500 m) | 638 | 31 | 17 | - | - |
| AP | offset_0 | [0, 0] cells (0, 0 m) | 2,106 | 848 | 414 | 774 | 414 |
| AP | offset_1 | [0, 16] cells (0, 480 m) | 2,106 | 841 | 411 | 775 | 411 |
| AP | offset_2 | [16, 0] cells (480, 0 m) | 2,106 | 856 | 414 | 764 | 414 |
| AP | offset_3 | [16, 16] cells (480, 480 m) | 2,106 | 844 | 414 | 761 | 414 |
| TTV | offset_0 | [0, 0] cells (0, 0 m) | 324 | 110 | 110 | 5 | - |
| TTV | offset_1 | [0, 50] cells (0, 500 m) | 306 | 108 | 108 | 5 | - |
| TTV | offset_2 | [50, 0] cells (500, 0 m) | 306 | 103 | 103 | 7 | - |
| TTV | offset_3 | [50, 50] cells (500, 500 m) | 289 | 99 | 99 | 8 | - |

| site | coarse grid | shape | cell (m) | valid cells | W (cells) | W (m) | footprint mask | A_c (m2) | routing s |
|---|---|---|---|---|---|---|---|---|---|
| GM | GM_Lidar_overlap_10m.tif | 2997 x 2297 | 10.0000 | 583,283 | 100 | 1000 | /scratch/users/scdobbs/lem_topography/Gabilan_mesa.tif | 10000.0 | 9.238 |
| AP | AP_Lidar_overlap_30m.tif | 1314 x 1801 | 29.9990 | 1,152,659 | 33 | 990 | 3DEP tile index (QL2_5K_SPSouth_TileIndex_Clipped); PAMAP watershed = any-data mask | 50400.0 | 16.465 |
| TTV | TTV_NED_10m_utm10n.tif | 1800 x 1800 | 10.0000 | 1,524,547 | 100 | 1000 | Marin 2019 QL1 (USGS CA_NoCal_Wildfires_B5b_QL1_2018, Lot 12) tile index, cell centres | 10000.0 | 19.718 |


### A2. Screen statistics of the candidate windows (section 3.2), percentiles over all four offsets, from `pools/<site>_windows.csv`

**GM** (71 candidate windows)

| statistic | n | min | p05 | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|
| largest drainage area inside (m2) | 71 | 285,700 | 395,450 | 1,214,300 | 2,488,500 | 8,458,300 | 22,281,600 | 33,075,500 |
| area entering across the edge, total (m2) | 71 | 34800.0 | 212,650 | 1,036,400 | 2,268,700 | 7,955,450 | 26,760,550 | 32,497,300 |
| largest single edge inflow (m2) | 71 | 9100.0 | 132,100 | 516,000 | 1,681,000 | 4,270,900 | 21,565,400 | 22,281,600 |
| fraction of cells with upstream area touching the footprint edge | 71 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 2.000e-04 |
| edge-outlet basins | 71 | 396.0 | 396.0 | 396.0 | 396.0 | 396.0 | 396.0 | 396.0 |
| chi-z R2 at theta_ref, area-weighted | 71 | 0.872 | 0.923 | 0.971 | 0.980 | 0.988 | 0.996 | 0.996 |
| chi-z R2, worst basin >= 10 % | 71 | 0.725 | 0.890 | 0.960 | 0.976 | 0.987 | 0.995 | 0.997 |
| slope-area dBIC (one - two segments), area-weighted | 71 | -13.188 | -7.407 | -4.410 | -1.398 | 5.699 | 20.509 | 34.900 |
| dBIC, worst basin >= 10 % | 71 | -14.972 | -8.260 | -1.247 | 6.613 | 14.416 | 31.304 | 37.488 |
| fraction of draining cells in basins where a break is preferred | 71 | 0.000 | 0.000 | 0.060 | 0.295 | 0.621 | 0.882 | 0.909 |
| NLCD 2021 disturbed fraction (11, 21-24, 31) | 71 | 0.000 | 0.000 | 0.000 | 0.000 | 0.012 | 0.069 | 0.188 |
| NLCD developed (21-24) | 71 | 0.000 | 0.000 | 0.000 | 0.000 | 0.012 | 0.069 | 0.188 |
| NLCD open water (11) | 71 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 6.100e-03 |
| AP: PA DEP mining fraction (longwall, mined-out, surface permits) | 0 | - | - | - | - | - | - | - |
| conflict degree (direct drains-through edges) | 71 | 2.000 | 2.000 | 3.000 | 4.000 | 4.000 | 5.000 | 6.000 |
| conflict degree (transitive closure) | 71 | 18.000 | 18.000 | 18.000 | 19.000 | 20.000 | 20.000 | 20.000 |
| relief (m) | 71 | 81.365 | 93.716 | 113.2 | 124.2 | 132.9 | 155.3 | 179.1 |
| mean gradient | 71 | 0.200 | 0.234 | 0.268 | 0.305 | 0.325 | 0.348 | 0.359 |

**AP** (1653 candidate windows)

| statistic | n | min | p05 | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|
| largest drainage area inside (m2) | 1,653 | 220,486 | 486,149 | 1,115,028 | 2,774,520 | 10,552,716 | 61,248,232 | 178,905,710 |
| area entering across the edge, total (m2) | 1,653 | 0.000 | 130,492 | 710,054 | 2,600,832 | 10,799,300 | 75,583,944 | 344,602,076 |
| largest single edge inflow (m2) | 1,653 | 0.000 | 26998.3 | 386,975 | 1,676,591 | 8,042,779 | 59,331,176 | 177,614,294 |
| fraction of cells with upstream area touching the footprint edge | 1,653 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 6.428e-03 |
| edge-outlet basins | 1,653 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 | 128.0 |
| chi-z R2 at theta_ref, area-weighted | 1,653 | 1.084e-03 | 0.663 | 0.938 | 0.975 | 0.987 | 0.994 | 0.998 |
| chi-z R2, worst basin >= 10 % | 1,653 | 1.727e-04 | 0.470 | 0.925 | 0.970 | 0.986 | 0.993 | 0.998 |
| slope-area dBIC (one - two segments), area-weighted | 1,653 | -12.700 | -9.311 | -4.589 | 3.590 | 17.010 | 45.386 | 134.8 |
| dBIC, worst basin >= 10 % | 1,653 | -12.700 | -9.921 | -3.401 | 8.594 | 27.345 | 62.325 | 138.1 |
| fraction of draining cells in basins where a break is preferred | 1,653 | 0.000 | 0.000 | 0.043 | 0.472 | 0.702 | 0.817 | 0.884 |
| NLCD 2021 disturbed fraction (11, 21-24, 31) | 1,653 | 0.000 | 0.020 | 0.051 | 0.073 | 0.096 | 0.174 | 0.805 |
| NLCD developed (21-24) | 1,653 | 0.000 | 0.019 | 0.051 | 0.072 | 0.096 | 0.164 | 0.805 |
| NLCD open water (11) | 1,653 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3.673e-03 | 0.185 |
| AP: PA DEP mining fraction (longwall, mined-out, surface permits) | 1,653 | 0.000 | 0.000 | 0.000 | 0.702 | 0.972 | 1.000 | 1.000 |
| AP: longwall panels | 1,653 | 0.000 | 0.000 | 0.000 | 5.510e-03 | 0.510 | 0.864 | 0.906 |
| AP: mined-out underground | 1,653 | 0.000 | 0.000 | 0.000 | 0.122 | 0.830 | 1.000 | 1.000 |
| AP: historic surface-mining permits | 1,653 | 0.000 | 0.000 | 0.000 | 0.000 | 0.093 | 1.000 | 1.000 |
| conflict degree (direct drains-through edges) | 1,653 | 0.000 | 3.000 | 4.000 | 5.000 | 5.000 | 6.000 | 8.000 |
| conflict degree (transitive closure) | 1,653 | 0.000 | 409.0 | 409.0 | 410.0 | 411.0 | 412.0 | 412.0 |
| relief (m) | 1,653 | 44.041 | 77.888 | 97.504 | 108.1 | 118.7 | 134.5 | 165.8 |
| mean gradient | 1,653 | 0.067 | 0.147 | 0.183 | 0.231 | 0.254 | 0.274 | 0.305 |

**TTV** (420 candidate windows)

| statistic | n | min | p05 | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|
| largest drainage area inside (m2) | 420 | 138,200 | 246,245 | 551,275 | 1,249,100 | 2,919,500 | 13,679,100 | 22,287,200 |
| area entering across the edge, total (m2) | 420 | 13900.0 | 67260.0 | 335,100 | 972,600 | 2,734,550 | 15,116,160 | 40,728,500 |
| largest single edge inflow (m2) | 420 | 2600.0 | 18645.0 | 132,125 | 462,200 | 1,443,300 | 11,138,500 | 19,205,800 |
| fraction of cells with upstream area touching the footprint edge | 420 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.010 | 0.020 |
| edge-outlet basins | 420 | 396.0 | 396.0 | 396.0 | 396.0 | 396.0 | 396.0 | 396.0 |
| chi-z R2 at theta_ref, area-weighted | 420 | 0.341 | 0.759 | 0.929 | 0.971 | 0.986 | 0.993 | 0.997 |
| chi-z R2, worst basin >= 10 % | 420 | 0.102 | 0.675 | 0.904 | 0.959 | 0.979 | 0.990 | 0.996 |
| slope-area dBIC (one - two segments), area-weighted | 420 | -13.736 | -7.808 | -1.353 | 8.316 | 34.894 | 116.2 | 307.5 |
| dBIC, worst basin >= 10 % | 420 | -14.844 | -8.359 | 3.337 | 21.850 | 66.364 | 151.7 | 324.7 |
| fraction of draining cells in basins where a break is preferred | 420 | 0.000 | 0.013 | 0.281 | 0.577 | 0.778 | 0.904 | 0.954 |
| NLCD 2021 disturbed fraction (11, 21-24, 31) | 420 | 0.000 | 0.000 | 0.018 | 0.102 | 0.673 | 0.978 | 1.000 |
| NLCD developed (21-24) | 420 | 0.000 | 0.000 | 0.018 | 0.101 | 0.670 | 0.970 | 1.000 |
| NLCD open water (11) | 420 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.011 | 0.230 |
| AP: PA DEP mining fraction (longwall, mined-out, surface permits) | 0 | - | - | - | - | - | - | - |
| conflict degree (direct drains-through edges) | 420 | 0.000 | 2.000 | 3.000 | 4.000 | 5.000 | 6.000 | 8.000 |
| conflict degree (transitive closure) | 420 | 0.000 | 11.000 | 92.000 | 92.000 | 96.000 | 97.000 | 97.000 |
| relief (m) | 420 | 26.824 | 53.303 | 147.1 | 218.6 | 266.0 | 397.2 | 541.6 |
| mean gradient | 420 | 0.035 | 0.078 | 0.269 | 0.361 | 0.444 | 0.561 | 0.698 |


### A3. Threshold set A and the windows each screen removes (section 3.2), from `scripts/a14_thresholds.json`

Thresholds: trunk_A_max_inside_m2_max = 5000000.0; trunk_A_inflow_max_single_m2_max = 1000000.0; transient_chi_r2_area_weighted_min = 0.95; transient_sa_dbic_area_weighted_max = 10.0; disturbance_nlcd_disturbed_frac_max = 0.05; disturbance_mining_frac_max = 0.0; upstream_fully_valid_required = {'GM': True, 'TTV': True, 'AP': False}; tolerance_ks_rel = 0.25; tolerance_C_ht_rel = 0.25; tolerance_theta_abs = 0.05

**GM** (71 candidates; passing all screens per offset: {0: 6, 1: 5, 2: 6, 3: 7})

| screen | windows removed (alone) | share | windows kept (alone) |
|---|---|---|---|
| trunk: A_max_inside <= 5e+06 m2 | 23 | 32.4 % | 48 |
| trunk: largest single edge inflow <= 1e+06 m2 | 45 | 63.4 % | 26 |
| transient: chi-z R2 (area-weighted) >= 0.95 | 8 | 11.3 % | 63 |
| transient: slope-area dBIC (area-weighted) <= 10 | 12 | 16.9 % | 59 |
| disturbance: NLCD disturbed fraction (11, 21-24, 31) <= 0.05 | 5 | 7.0 % | 66 |
| upstream area fully in valid data (decision 7) | 2 | 2.8 % | 69 |
| **all screens jointly** | 47 | 66.2 % | 24 |

**AP** (1653 candidates; passing all screens per offset: {0: 3, 1: 5, 2: 5, 3: 4})

| screen | windows removed (alone) | share | windows kept (alone) |
|---|---|---|---|
| trunk: A_max_inside <= 5e+06 m2 | 587 | 35.5 % | 1,066 |
| trunk: largest single edge inflow <= 1e+06 m2 | 976 | 59.0 % | 677 |
| transient: chi-z R2 (area-weighted) >= 0.95 | 475 | 28.7 % | 1,178 |
| transient: slope-area dBIC (area-weighted) <= 10 | 619 | 37.4 % | 1,034 |
| disturbance: NLCD disturbed fraction (11, 21-24, 31) <= 0.05 | 1,269 | 76.8 % | 384 |
| disturbance (AP): mining fraction (all three layers) <= 0.00 | 1,213 | 73.4 % | 440 |
| **all screens jointly** | 1,636 | 99.0 % | 17 |

**TTV** (420 candidates; passing all screens per offset: {0: 8, 1: 11, 2: 7, 3: 9})

| screen | windows removed (alone) | share | windows kept (alone) |
|---|---|---|---|
| trunk: A_max_inside <= 5e+06 m2 | 62 | 14.8 % | 358 |
| trunk: largest single edge inflow <= 1e+06 m2 | 147 | 35.0 % | 273 |
| transient: chi-z R2 (area-weighted) >= 0.95 | 149 | 35.5 % | 271 |
| transient: slope-area dBIC (area-weighted) <= 10 | 193 | 46.0 % | 227 |
| disturbance: NLCD disturbed fraction (11, 21-24, 31) <= 0.05 | 274 | 65.2 % | 146 |
| upstream area fully in valid data (decision 7) | 48 | 11.4 % | 372 |
| **all screens jointly** | 385 | 91.7 % | 35 |


### A4. Selection metrics against the reference (section 3.3): distributions, tolerance counts alone and jointly with the set A screens

**GM** reference: theta 0.292, k_s 1.900, |C_ht| 0.013 (legacy TAYLOR_REF (updated_best_fit_tile_finder.ipynb cell 1))

| metric | n | min | p05 | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|
| theta by chi (legacy method) | 71 | 0.196 | 0.246 | 0.310 | 0.388 | 0.438 | 0.496 | 0.562 |
| k_s at theta_ref, median of basins | 71 | 1.213 | 1.823 | 2.141 | 2.382 | 2.675 | 3.384 | 4.555 |
| k_s at theta_ref, pooled | 71 | 1.494 | 1.746 | 2.076 | 2.230 | 2.520 | 2.889 | 3.463 |
| |C_ht| (3x3 Laplacian, convex first-order divides) | 71 | 8.628e-03 | 9.252e-03 | 0.012 | 0.014 | 0.015 | 0.017 | 0.018 |
| basins in the chi fit | 71 | 9.000 | 11.000 | 19.000 | 23.000 | 29.500 | 44.000 | 49.000 |
| convex divide pixels | 71 | 608.0 | 646.0 | 708.0 | 750.0 | 828.5 | 957.0 | 984.0 |

| tolerance | windows within (alone) | within and passing all screens |
|---|---|---|
| theta within +-0.05 | 20 | 6 |
| k_s within +-25 % | 34 | 13 |
| |C_ht| within +-25 % | 56 | 17 |
| k_s and |C_ht| (theta free) | 23 | 10 |
| **all three** | 9 | 4 |

Eligible (all screens and all tolerances) per offset: {0: 1, 1: 0, 2: 2, 3: 1}; total candidates 71.

**AP** reference: theta 0.540, k_s 47.200, |C_ht| 5.930e-03 (legacy TAYLOR_REF (updated_best_fit_tile_finder.ipynb cell 1))

| metric | n | min | p05 | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|
| theta by chi (legacy method) | 1,653 | 0.100 | 0.128 | 0.479 | 0.592 | 0.704 | 0.800 | 0.800 |
| k_s at theta_ref, median of basins | 1,653 | -40.026 | 19.845 | 31.984 | 38.166 | 45.554 | 58.874 | 146.3 |
| k_s at theta_ref, pooled | 1,653 | -7.000 | 21.957 | 32.004 | 38.072 | 44.658 | 57.504 | 83.625 |
| |C_ht| (3x3 Laplacian, convex first-order divides) | 1,653 | 1.697e-03 | 3.174e-03 | 3.985e-03 | 5.014e-03 | 5.927e-03 | 6.906e-03 | 8.650e-03 |
| basins in the chi fit | 1,653 | 3.000 | 3.000 | 7.000 | 10.000 | 15.000 | 26.000 | 48.000 |
| convex divide pixels | 1,653 | 31.000 | 62.600 | 85.000 | 100.0 | 115.0 | 136.0 | 165.0 |

| tolerance | windows within (alone) | within and passing all screens |
|---|---|---|
| theta within +-0.05 | 384 | 1 |
| k_s within +-25 % | 955 | 10 |
| |C_ht| within +-25 % | 990 | 5 |
| k_s and |C_ht| (theta free) | 651 | 1 |
| **all three** | 181 | 0 |

Eligible (all screens and all tolerances) per offset: {0: 0, 1: 0, 2: 0, 3: 0}; total candidates 1653.

**TTV** reference: theta 0.387, k_s 15.416, |C_ht| 0.015 (M&D proxy catchment on the 10 m NED, same code)

| metric | n | min | p05 | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|
| theta by chi (legacy method) | 420 | 0.100 | 0.100 | 0.221 | 0.340 | 0.448 | 0.592 | 0.800 |
| k_s at theta_ref, median of basins | 420 | 1.753e-03 | 1.000 | 6.933 | 12.711 | 16.639 | 23.735 | 40.382 |
| k_s at theta_ref, pooled | 420 | 0.504 | 1.628 | 6.813 | 12.917 | 18.366 | 28.634 | 41.010 |
| |C_ht| (3x3 Laplacian, convex first-order divides) | 420 | 3.250e-03 | 6.455e-03 | 0.015 | 0.020 | 0.024 | 0.030 | 0.035 |
| basins in the chi fit | 420 | 6.000 | 16.000 | 29.000 | 38.000 | 49.250 | 68.000 | 105.0 |
| convex divide pixels | 420 | 632.0 | 764.8 | 990.0 | 1150.0 | 1253.5 | 1449.3 | 1935.0 |

| tolerance | windows within (alone) | within and passing all screens |
|---|---|---|
| theta within +-0.05 | 93 | 5 |
| k_s within +-25 % | 190 | 24 |
| |C_ht| within +-25 % | 122 | 12 |
| k_s and |C_ht| (theta free) | 46 | 8 |
| **all three** | 13 | 1 |

Eligible (all screens and all tolerances) per offset: {0: 0, 1: 0, 2: 1, 3: 0}; total candidates 420.


### A4b. Sensitivity of the eligible count to the reference and the tolerances (after all set A screens)

| site | reference | theta / k_s / |C_ht| | theta tolerance | k_s and C_ht tolerance | within tolerance (alone) | eligible (with screens) | per offset |
|---|---|---|---|---|---|---|---|
| GM | legacy / M&D proxy | 0.292 / 1.90 / 0.0128 | +-0.05 | +-25 % | 9 | 4 | {0: 1, 1: 0, 2: 2, 3: 1} |
| GM | legacy / M&D proxy | 0.292 / 1.90 / 0.0128 | +-0.05 | +-50 % | 16 | 5 | {0: 1, 1: 0, 2: 2, 3: 2} |
| GM | legacy / M&D proxy | 0.292 / 1.90 / 0.0128 | +-0.10 | +-25 % | 15 | 6 | {0: 1, 1: 0, 2: 2, 3: 3} |
| GM | legacy / M&D proxy | 0.292 / 1.90 / 0.0128 | +-0.10 | +-50 % | 33 | 13 | {0: 3, 1: 2, 2: 3, 3: 5} |
| GM | legacy / M&D proxy | 0.292 / 1.90 / 0.0128 | none | +-25 % | 23 | 10 | {0: 3, 1: 1, 2: 2, 3: 4} |
| GM | legacy / M&D proxy | 0.292 / 1.90 / 0.0128 | none | +-50 % | 58 | 21 | {0: 5, 1: 5, 2: 4, 3: 7} |
| GM | same code, whole coarse grid | 0.354 / 1.66 / 0.0151 | +-0.05 | +-25 % | 2 | 1 | {0: 0, 1: 0, 2: 1, 3: 0} |
| GM | same code, whole coarse grid | 0.354 / 1.66 / 0.0151 | +-0.05 | +-50 % | 15 | 8 | {0: 2, 1: 2, 2: 1, 3: 3} |
| GM | same code, whole coarse grid | 0.354 / 1.66 / 0.0151 | +-0.10 | +-25 % | 3 | 1 | {0: 0, 1: 0, 2: 1, 3: 0} |
| GM | same code, whole coarse grid | 0.354 / 1.66 / 0.0151 | +-0.10 | +-50 % | 30 | 13 | {0: 3, 1: 4, 2: 1, 3: 5} |
| GM | same code, whole coarse grid | 0.354 / 1.66 / 0.0151 | none | +-25 % | 5 | 2 | {0: 0, 1: 0, 2: 2, 3: 0} |
| GM | same code, whole coarse grid | 0.354 / 1.66 / 0.0151 | none | +-50 % | 42 | 18 | {0: 5, 1: 4, 2: 3, 3: 6} |
| AP | legacy / M&D proxy | 0.540 / 47.20 / 0.0059 | +-0.05 | +-25 % | 181 | 0 | {0: 0, 1: 0, 2: 0, 3: 0} |
| AP | legacy / M&D proxy | 0.540 / 47.20 / 0.0059 | +-0.05 | +-50 % | 348 | 1 | {0: 0, 1: 1, 2: 0, 3: 0} |
| AP | legacy / M&D proxy | 0.540 / 47.20 / 0.0059 | +-0.10 | +-25 % | 328 | 0 | {0: 0, 1: 0, 2: 0, 3: 0} |
| AP | legacy / M&D proxy | 0.540 / 47.20 / 0.0059 | +-0.10 | +-50 % | 655 | 6 | {0: 2, 1: 2, 2: 1, 3: 1} |
| AP | legacy / M&D proxy | 0.540 / 47.20 / 0.0059 | none | +-25 % | 651 | 1 | {0: 0, 1: 0, 2: 0, 3: 1} |
| AP | legacy / M&D proxy | 0.540 / 47.20 / 0.0059 | none | +-50 % | 1,460 | 16 | {0: 3, 1: 5, 2: 5, 3: 3} |
| AP | same code, whole coarse grid | 0.571 / 39.50 / 0.0058 | +-0.05 | +-25 % | 228 | 1 | {0: 0, 1: 0, 2: 1, 3: 0} |
| AP | same code, whole coarse grid | 0.571 / 39.50 / 0.0058 | +-0.05 | +-50 % | 378 | 3 | {0: 1, 1: 1, 2: 1, 3: 0} |
| AP | same code, whole coarse grid | 0.571 / 39.50 / 0.0058 | +-0.10 | +-25 % | 391 | 1 | {0: 0, 1: 0, 2: 1, 3: 0} |
| AP | same code, whole coarse grid | 0.571 / 39.50 / 0.0058 | +-0.10 | +-50 % | 688 | 6 | {0: 2, 1: 2, 2: 1, 3: 1} |
| AP | same code, whole coarse grid | 0.571 / 39.50 / 0.0058 | none | +-25 % | 745 | 2 | {0: 0, 1: 0, 2: 1, 3: 1} |
| AP | same code, whole coarse grid | 0.571 / 39.50 / 0.0058 | none | +-50 % | 1,463 | 15 | {0: 3, 1: 5, 2: 4, 3: 3} |
| TTV | legacy / M&D proxy | 0.387 / 15.42 / 0.0151 | +-0.05 | +-25 % | 13 | 1 | {0: 0, 1: 0, 2: 1, 3: 0} |
| TTV | legacy / M&D proxy | 0.387 / 15.42 / 0.0151 | +-0.05 | +-50 % | 48 | 3 | {0: 1, 1: 1, 2: 1, 3: 0} |
| TTV | legacy / M&D proxy | 0.387 / 15.42 / 0.0151 | +-0.10 | +-25 % | 24 | 2 | {0: 0, 1: 0, 2: 2, 3: 0} |
| TTV | legacy / M&D proxy | 0.387 / 15.42 / 0.0151 | +-0.10 | +-50 % | 96 | 7 | {0: 2, 1: 2, 2: 2, 3: 1} |
| TTV | legacy / M&D proxy | 0.387 / 15.42 / 0.0151 | none | +-25 % | 46 | 8 | {0: 1, 1: 3, 2: 3, 3: 1} |
| TTV | legacy / M&D proxy | 0.387 / 15.42 / 0.0151 | none | +-50 % | 179 | 19 | {0: 4, 1: 7, 2: 3, 3: 5} |


### A5. The original ten tiles against the new rules (section 3.4), from `pools/<site>_pools.json` (`original_tiles`)

| tile | row0, col0 | coarse valid | footprint full | A_max inside (m2) | largest edge inflow (m2) | upstream fully valid | chi R2 (w) | dBIC (w) | NLCD disturbed | mining | passes all proposed screens | theta | k_s(theta_ref) median | |C_ht| | tol theta | tol k_s | tol C_ht | tol all |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | 826, 1024 | yes | no | 1,064,631 | 115,193 | yes | 0.994 | 49.016 | 0.043 | 1.000 | no | 0.621 | 32.629 | 5.089e-03 | no | no | yes | no |
| AP2 | 232, 793 | yes | no | 1,666,692 | 994,436 | yes | 0.983 | 46.582 | 0.048 | 0.872 | no | 0.780 | 40.002 | 4.807e-03 | no | yes | yes | no |
| AP3 | 463, 469 | yes | no | 10,655,310 | 8,701,536 | yes | 0.990 | -10.453 | 0.064 | 0.088 | no | 0.512 | 53.952 | 4.711e-03 | yes | yes | yes | yes |
| AP4 | 463, 577 | yes | no | 16,637,222 | 16,572,426 | yes | 0.980 | 14.065 | 0.043 | 0.000 | no | 0.692 | 36.628 | 5.936e-03 | no | yes | yes | no |
| AP5 | 355, 637 | yes | no | 1,227,520 | 595,761 | yes | 0.993 | 10.571 | 0.073 | 1.000 | no | 0.433 | 32.853 | 6.264e-03 | no | no | yes | no |
| GM1 | 1901, 1101 | yes | yes | 1,146,000 | 895,400 | yes | 0.995 | 2.541 | 0.000 | - | yes | 0.362 | 2.333 | 0.017 | no | yes | no | no |
| GM2 | 1901, 901 | yes | yes | 8,879,600 | 8,470,900 | yes | 0.982 | -4.844 | 0.000 | - | no | 0.367 | 2.105 | 0.014 | no | yes | yes | no |
| GM3 | 2101, 601 | yes | yes | 344,100 | 2100.0 | yes | 0.983 | -5.346 | 0.018 | - | yes | 0.262 | 2.053 | 0.012 | yes | yes | yes | yes |
| GM4 | 2301, 501 | no | yes | 6,198,200 | 5,159,000 | no | 0.990 | 6.678 | 0.000 | - | no | 0.321 | 3.563 | 0.015 | yes | no | yes | no |
| GM5 | 1801, 1001 | yes | yes | 4,729,200 | 2,656,100 | yes | 0.980 | -9.251 | 0.000 | - | no | 0.312 | 2.202 | 0.017 | yes | yes | no | no |


### A6. References per site (decision 6)

| site | legacy theta | legacy k_s (median of basins) | legacy |C_ht| | Perron S2 |lap z_h| | Perron S2 m | E/K from Perron S2 (k_s, m^(2m)... units of m=Perron) | this code, whole coarse grid: theta (legacy chi) | k_s(theta_ref) median | k_s pooled | |C_ht| | theta (vectorised chi fit, >= 10 px basins) | basins |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM | 0.292 | 1.900 | 0.013 | 0.012 | 0.350 | 1.47 | 0.354 | 1.664 | 2.105 | 0.015 | 0.360 | 328 |
| AP | 0.540 | 47.200 | 5.930e-03 | 5.500e-03 | 0.370 | 4.40 | 0.571 | 39.500 | 37.023 | 5.819e-03 | 0.570 | 728 |


**TTV M&D proxy** (`pools/TTV_pools.json` `md_proxy`): {"theta_legacy": 0.387, "theta_legacy_coarse": 0.375, "ks_at_theta_legacy_pooled": 15.4211, "chi_r2_at_theta_legacy": 0.9935, "n_basins_chi": 15, "n_points_chi": 4724, "C_ht": 0.0151, "C_ht_n_convex": 455, "C_ht_n_divide": 618, "C_ht_n_saddle": 161, "C_ht_n_flat": 1, "ks_ref_median": 15.4162, "ks_ref_pooled": 15.4211, "ks_ref_n_basins": 15, "outlet_row": 771, "outlet_col": 1087, "outlet_x": 540875.0, "outlet_y": 4190285.0, "outlet_z": 61.4146, "outlet_area_km2": 0.6729, "catchment_cells": 6729, "catchment_km2": 0.6729, "z_min": 61.4146, "z_max": 317.3718}

TTV whole 10 m grid (18 x 18 km domain), this code: {"ks_ref_median": 7.8071, "ks_ref_pooled": 10.6329, "n_basins": 932, "C_ht": 0.0178, "C_ht_n_convex": 189718, "C_ht_n_divide": 268127, "seconds": 6.2481, "theta_fit_chi_vectorised": 0.39, "theta_fit_chi_n_basins": 240, "ks_pooled_fit_chi_at_theta_star": 11.0717}


### A7. Conflict graphs (section 3.2, screen 4), from `pools/<site>_conflicts.json`

| site | offset | candidates | direct edges (A drains into B) | direct pairs | max direct degree | closure pairs | windows in closure | max closure degree |
|---|---|---|---|---|---|---|---|---|
| GM | offset_0 | 17 | 50 | 31 | 5 | 153 | 17 | 18 |
| GM | offset_1 | 19 | 58 | 36 | 6 | 190 | 19 | 20 |
| GM | offset_2 | 18 | 53 | 32 | 5 | 171 | 18 | 19 |
| GM | offset_3 | 17 | 52 | 31 | 6 | 153 | 17 | 18 |
| AP | offset_0 | 414 | 1,564 | 957 | 7 | 83,856 | 414 | 410 |
| AP | offset_1 | 411 | 1,505 | 913 | 8 | 83,437 | 410 | 409 |
| AP | offset_2 | 414 | 1,548 | 946 | 7 | 84,667 | 414 | 412 |
| AP | offset_3 | 414 | 1,546 | 937 | 8 | 84,224 | 412 | 411 |
| TTV | offset_0 | 110 | 390 | 225 | 7 | 4,652 | 110 | 96 |
| TTV | offset_1 | 108 | 382 | 221 | 7 | 4,721 | 108 | 97 |
| TTV | offset_2 | 103 | 357 | 203 | 8 | 4,242 | 103 | 92 |
| TTV | offset_3 | 99 | 339 | 201 | 7 | 4,213 | 98 | 92 |


### A8. Unit and datum checks (section 5)

| site | rule | cells | median dz (m) | p05 | p95 | Theil-Sen slope | expected offset (m) | pass |
|---|---|---|---|---|---|---|---|---|
| GM | 5.3 coarse grid vs lidar block-mean | 378,600 | 32.273 | 26.240 | 37.955 | 0.999 | [30.0, 34.0] | yes |
| AP | 5.3 coarse grid vs lidar block-mean | 523,526 | 1.174 | -5.650 | 12.437 | 0.986 | [-5.0, 5.0] | yes |
| TTV | 5.3 NED (10 m, EPSG:6339) vs NCALM 2006 block-mean | 151,369 | 0.235 | -0.450 | 1.107 | 1.000 | [-5, 5] | yes |


**Rule 5.1, AP** (`grid_AP/AP_checks.json`): new 1 m watershed grid (class-2 mean, unfilled cells) against the PAMAP tiles in metres

| tile | cells | median dz new - PAMAP (m) | addendum 13 median | p05 | p95 | Theil-Sen slope | median within 0.1 of a13 | slope 0.99-1.01 | pass |
|---|---|---|---|---|---|---|---|---|---|
| AP1 | 982,830 | -0.071 | -0.079 | -0.413 | 0.284 | 1.000 | yes | yes | yes |
| AP2 | 952,978 | -0.073 | -0.077 | -0.323 | 0.141 | 1.000 | yes | yes | yes |
| AP3 | 971,593 | -0.066 | -0.066 | -0.402 | 0.238 | 0.999 | yes | yes | yes |
| AP4 | 979,177 | -0.021 | -0.026 | -0.355 | 0.260 | 1.001 | yes | yes | yes |
| AP5 | 956,091 | -1.213 | -1.206 | -1.798 | -0.200 | 0.989 | yes | no | excluded (addendum 13, 6.2) |


**Rule 5.2, TTV** (`grid_TTV/TTV_checks.json`): Marin 2019 1 m grid against the NCALM 2006 raster on ground cells

| cells | median dz (m) | p05 | p95 | Theil-Sen slope | median within 1 m | slope 0.98-1.02 | units flag | ellipsoid flag | pass | stop |
|---|---|---|---|---|---|---|---|---|---|---|
| 12,813,875 | 0.084 | -0.326 | 0.411 | 1.000 | yes | yes | no | no | yes | no |


### A9. The TTV NCALM raster and the NED (section 2), from `ttv_inventory/ttv_inventory.json`

| file | bytes | sha256 (16) | format | CRS (header) | horizontal units | vertical CRS | resolution | extent | nodata | value range | valid |
|---|---|---|---|---|---|---|---|---|---|---|---|
| output_be.tif | 65,603,380 | 691c5ae984cc08cd | GTiff float32 (COG, LZW) | NAD83 / UTM zone 10N | metre | not stated (no vertical CRS in file) | 1.0 x 1.0 | 538470.5, 4187853.5, 542654.5, 4191961.5 | -340,282,346,638,528,859,811,704,183,484,516,925,440 | 0.15-317.37 | 91.5 % |
| USGS_13_n38w123_20120801.tif | 218,380,355 | 4e4fa132638e89b2 | GTiff float32 (COG, LZW) | NAD83 | degree | not stated (no vertical CRS in file) | 9.25926e-05 deg | -123.0006, 36.9994, -121.9994, 38.0006 | -999,999 | -66.8-986.3 | 100.000 % |


**output_be.tif against the NCALM LAS class-2 points** (8,888,408 points): nearest-cell dz median -0.0003 m, p05-p95 -0.212 to 0.213; bilinear p05-p95 -0.073 to 0.079 m; cells equal to the mean of their points within 1 mm: 1.3 %, within 1 cm 12.1 %; equal to the min 1.0 %, max 1.0 %; raster between the cell's min and max 18.5 %; valid cells without a class-2 point 57.4 %.

Coverage: {"inside_point_bonita_quad": true, "east_edge_lon": -122.51500003899038, "covers_east_of_122.515W": false, "frac_valid_cells_east_of_122.515W": 0.0, "frac_valid_in_MD_band_61_314": 0.827769678183783, "max_valid_z": 317.3689270019531, "min_valid_z": 0.14598320424556732, "n_valid": 15732613, "n_cells": 17187872, "n_columns_east_of_122.515W": 0}
Low surface: {"n_cells_below_3m": 534322, "percentiles_0.1_1_5_50": [1.2368310158252716, 1.461076738834381, 1.5908035457134246, 1.8912776112556458]}


### A10. Full-grid data: file lists (section 4)

AP watershed footprint: valid 516.6 km2, buffered (500 m) 578.6 km2, envelope UTM 17N [549464, 585052, 4405649, 4430794].


**PASDA quadrants** (`footprints/footprints.json`): 294 files, 294 listed, 0 absent, 0 not in index, 0 index/rule mismatches, 0 already on Sherlock; total 57.45 GB zipped (gate 60 GB: True), LAS estimate 86.7 GB.

| quadrant zip | bytes | listed | in index | index = rule | already downloaded |
|---|---|---|---|---|---|
| 18001240PAS_NE_LAS.zip | 214,797,853 | yes | yes | yes | no |
| 18001260PAS_NE_LAS.zip | 187,606,532 | yes | yes | yes | no |
| 18001260PAS_NW_LAS.zip | 191,049,569 | yes | yes | yes | no |
| 19001230PAS_NE_LAS.zip | 211,777,506 | yes | yes | yes | no |
| 19001230PAS_NW_LAS.zip | 230,302,207 | yes | yes | yes | no |
| 19001230PAS_SE_LAS.zip | 219,494,491 | yes | yes | yes | no |
| 19001230PAS_SW_LAS.zip | 223,254,989 | yes | yes | yes | no |
| 19001240PAS_NE_LAS.zip | 223,999,629 | yes | yes | yes | no |
| 19001240PAS_NW_LAS.zip | 191,413,408 | yes | yes | yes | no |
| 19001240PAS_SE_LAS.zip | 239,638,302 | yes | yes | yes | no |
| 19001240PAS_SW_LAS.zip | 192,573,407 | yes | yes | yes | no |
| 19001250PAS_NE_LAS.zip | 205,802,488 | yes | yes | yes | no |
| 19001250PAS_NW_LAS.zip | 199,317,711 | yes | yes | yes | no |
| 19001250PAS_SE_LAS.zip | 237,273,659 | yes | yes | yes | no |
| 19001250PAS_SW_LAS.zip | 211,649,514 | yes | yes | yes | no |
| 19001260PAS_NE_LAS.zip | 216,915,582 | yes | yes | yes | no |
| 19001260PAS_NW_LAS.zip | 184,682,463 | yes | yes | yes | no |
| 19001260PAS_SE_LAS.zip | 190,835,804 | yes | yes | yes | no |
| 19001260PAS_SW_LAS.zip | 179,765,122 | yes | yes | yes | no |
| 19001270PAS_NE_LAS.zip | 209,200,481 | yes | yes | yes | no |
| 19001270PAS_NW_LAS.zip | 242,175,410 | yes | yes | yes | no |
| 19001270PAS_SE_LAS.zip | 211,069,900 | yes | yes | yes | no |
| 19001270PAS_SW_LAS.zip | 225,506,731 | yes | yes | yes | no |
| 19001280PAS_NE_LAS.zip | 170,052,126 | yes | yes | yes | no |
| 19001280PAS_NW_LAS.zip | 202,428,027 | yes | yes | yes | no |
| 19001280PAS_SE_LAS.zip | 208,971,127 | yes | yes | yes | no |
| 19001280PAS_SW_LAS.zip | 219,046,975 | yes | yes | yes | no |
| 20001230PAS_NE_LAS.zip | 200,293,022 | yes | yes | yes | no |
| 20001230PAS_NW_LAS.zip | 207,376,664 | yes | yes | yes | no |
| 20001230PAS_SE_LAS.zip | 221,641,206 | yes | yes | yes | no |
| 20001230PAS_SW_LAS.zip | 233,072,914 | yes | yes | yes | no |
| 20001240PAS_NE_LAS.zip | 210,820,087 | yes | yes | yes | no |
| 20001240PAS_NW_LAS.zip | 204,716,524 | yes | yes | yes | no |
| 20001240PAS_SE_LAS.zip | 225,166,522 | yes | yes | yes | no |
| 20001240PAS_SW_LAS.zip | 208,666,017 | yes | yes | yes | no |
| 20001250PAS_NE_LAS.zip | 194,754,916 | yes | yes | yes | no |
| 20001250PAS_NW_LAS.zip | 221,186,157 | yes | yes | yes | no |
| 20001250PAS_SE_LAS.zip | 217,927,711 | yes | yes | yes | no |
| 20001250PAS_SW_LAS.zip | 205,414,998 | yes | yes | yes | no |
| 20001260PAS_NE_LAS.zip | 205,830,417 | yes | yes | yes | no |
| 20001260PAS_NW_LAS.zip | 195,737,334 | yes | yes | yes | no |
| 20001260PAS_SE_LAS.zip | 189,016,597 | yes | yes | yes | no |
| 20001260PAS_SW_LAS.zip | 186,308,927 | yes | yes | yes | no |
| 20001270PAS_NE_LAS.zip | 199,174,903 | yes | yes | yes | no |
| 20001270PAS_NW_LAS.zip | 196,008,865 | yes | yes | yes | no |
| 20001270PAS_SE_LAS.zip | 202,415,480 | yes | yes | yes | no |
| 20001270PAS_SW_LAS.zip | 226,343,718 | yes | yes | yes | no |
| 20001280PAS_NE_LAS.zip | 167,327,760 | yes | yes | yes | no |
| 20001280PAS_NW_LAS.zip | 182,643,585 | yes | yes | yes | no |
| 20001280PAS_SE_LAS.zip | 136,716,899 | yes | yes | yes | no |
| 20001280PAS_SW_LAS.zip | 203,918,796 | yes | yes | yes | no |
| 20001290PAS_NE_LAS.zip | 209,718,317 | yes | yes | yes | no |
| 20001290PAS_NW_LAS.zip | 189,777,054 | yes | yes | yes | no |
| 20001300PAS_NW_LAS.zip | 200,478,737 | yes | yes | yes | no |
| 21001220PAS_NE_LAS.zip | 212,489,959 | yes | yes | yes | no |
| 21001220PAS_SE_LAS.zip | 225,646,095 | yes | yes | yes | no |
| 21001230PAS_NE_LAS.zip | 198,521,150 | yes | yes | yes | no |
| 21001230PAS_NW_LAS.zip | 191,973,401 | yes | yes | yes | no |
| 21001230PAS_SE_LAS.zip | 186,065,378 | yes | yes | yes | no |
| 21001230PAS_SW_LAS.zip | 200,837,657 | yes | yes | yes | no |
| 21001240PAS_NE_LAS.zip | 200,902,002 | yes | yes | yes | no |
| 21001240PAS_NW_LAS.zip | 188,444,859 | yes | yes | yes | no |
| 21001240PAS_SE_LAS.zip | 208,150,323 | yes | yes | yes | no |
| 21001240PAS_SW_LAS.zip | 194,875,146 | yes | yes | yes | no |
| 21001250PAS_NE_LAS.zip | 192,938,501 | yes | yes | yes | no |
| 21001250PAS_NW_LAS.zip | 210,720,123 | yes | yes | yes | no |
| 21001250PAS_SE_LAS.zip | 206,790,979 | yes | yes | yes | no |
| 21001250PAS_SW_LAS.zip | 196,517,878 | yes | yes | yes | no |
| 21001260PAS_NE_LAS.zip | 182,085,808 | yes | yes | yes | no |
| 21001260PAS_NW_LAS.zip | 168,554,612 | yes | yes | yes | no |
| 21001260PAS_SE_LAS.zip | 162,835,777 | yes | yes | yes | no |
| 21001260PAS_SW_LAS.zip | 179,158,333 | yes | yes | yes | no |
| 21001270PAS_NE_LAS.zip | 197,777,584 | yes | yes | yes | no |
| 21001270PAS_NW_LAS.zip | 192,619,461 | yes | yes | yes | no |
| 21001270PAS_SE_LAS.zip | 205,694,246 | yes | yes | yes | no |
| 21001270PAS_SW_LAS.zip | 194,372,928 | yes | yes | yes | no |
| 21001280PAS_NE_LAS.zip | 161,597,688 | yes | yes | yes | no |
| 21001280PAS_NW_LAS.zip | 217,231,816 | yes | yes | yes | no |
| 21001280PAS_SE_LAS.zip | 168,315,219 | yes | yes | yes | no |
| 21001280PAS_SW_LAS.zip | 196,594,701 | yes | yes | yes | no |
| 21001290PAS_NE_LAS.zip | 196,057,911 | yes | yes | yes | no |
| 21001290PAS_NW_LAS.zip | 177,992,838 | yes | yes | yes | no |
| 21001290PAS_SE_LAS.zip | 199,363,372 | yes | yes | yes | no |
| 21001290PAS_SW_LAS.zip | 184,189,021 | yes | yes | yes | no |
| 21001300PAS_NE_LAS.zip | 216,662,054 | yes | yes | yes | no |
| 21001300PAS_NW_LAS.zip | 161,193,740 | yes | yes | yes | no |
| 21001300PAS_SE_LAS.zip | 219,884,117 | yes | yes | yes | no |
| 21001300PAS_SW_LAS.zip | 190,022,704 | yes | yes | yes | no |
| 21001310PAS_NW_LAS.zip | 183,716,873 | yes | yes | yes | no |
| 22001220PAS_NE_LAS.zip | 175,904,616 | yes | yes | yes | no |
| 22001220PAS_SE_LAS.zip | 216,672,368 | yes | yes | yes | no |
| 22001230PAS_NE_LAS.zip | 134,690,360 | yes | yes | yes | no |
| 22001230PAS_NW_LAS.zip | 178,723,042 | yes | yes | yes | no |
| 22001230PAS_SE_LAS.zip | 158,290,845 | yes | yes | yes | no |
| 22001230PAS_SW_LAS.zip | 194,097,577 | yes | yes | yes | no |
| 22001240PAS_NE_LAS.zip | 214,088,782 | yes | yes | yes | no |
| 22001240PAS_NW_LAS.zip | 202,752,043 | yes | yes | yes | no |
| 22001240PAS_SE_LAS.zip | 215,207,454 | yes | yes | yes | no |
| 22001240PAS_SW_LAS.zip | 186,036,734 | yes | yes | yes | no |
| 22001250PAS_NE_LAS.zip | 193,909,909 | yes | yes | yes | no |
| 22001250PAS_NW_LAS.zip | 203,344,037 | yes | yes | yes | no |
| 22001250PAS_SE_LAS.zip | 178,007,168 | yes | yes | yes | no |
| 22001250PAS_SW_LAS.zip | 208,074,076 | yes | yes | yes | no |
| 22001260PAS_NE_LAS.zip | 184,324,798 | yes | yes | yes | no |
| 22001260PAS_NW_LAS.zip | 167,409,540 | yes | yes | yes | no |
| 22001260PAS_SE_LAS.zip | 173,016,960 | yes | yes | yes | no |
| 22001260PAS_SW_LAS.zip | 172,720,721 | yes | yes | yes | no |
| 22001270PAS_NE_LAS.zip | 183,805,544 | yes | yes | yes | no |
| 22001270PAS_NW_LAS.zip | 224,336,340 | yes | yes | yes | no |
| 22001270PAS_SE_LAS.zip | 170,413,728 | yes | yes | yes | no |
| 22001270PAS_SW_LAS.zip | 194,559,520 | yes | yes | yes | no |
| 22001280PAS_NE_LAS.zip | 178,478,642 | yes | yes | yes | no |
| 22001280PAS_NW_LAS.zip | 191,659,020 | yes | yes | yes | no |
| 22001280PAS_SE_LAS.zip | 158,870,787 | yes | yes | yes | no |
| 22001280PAS_SW_LAS.zip | 169,411,070 | yes | yes | yes | no |
| 22001290PAS_NE_LAS.zip | 167,069,718 | yes | yes | yes | no |
| 22001290PAS_NW_LAS.zip | 192,789,532 | yes | yes | yes | no |
| 22001290PAS_SE_LAS.zip | 166,689,484 | yes | yes | yes | no |
| 22001290PAS_SW_LAS.zip | 162,283,060 | yes | yes | yes | no |
| 22001300PAS_NE_LAS.zip | 205,551,991 | yes | yes | yes | no |
| 22001300PAS_NW_LAS.zip | 155,095,063 | yes | yes | yes | no |
| 22001300PAS_SE_LAS.zip | 203,285,804 | yes | yes | yes | no |
| 22001300PAS_SW_LAS.zip | 148,162,602 | yes | yes | yes | no |
| 22001310PAS_NE_LAS.zip | 199,232,358 | yes | yes | yes | no |
| 22001310PAS_NW_LAS.zip | 167,425,825 | yes | yes | yes | no |
| 22001310PAS_SE_LAS.zip | 181,426,343 | yes | yes | yes | no |
| 22001310PAS_SW_LAS.zip | 182,056,345 | yes | yes | yes | no |
| 22001320PAS_NE_LAS.zip | 154,211,752 | yes | yes | yes | no |
| 22001320PAS_NW_LAS.zip | 202,031,628 | yes | yes | yes | no |
| 23001210PAS_NE_LAS.zip | 425,151,151 | yes | yes | yes | no |
| 23001220PAS_NE_LAS.zip | 206,598,732 | yes | yes | yes | no |
| 23001220PAS_NW_LAS.zip | 290,173,068 | yes | yes | yes | no |
| 23001220PAS_SE_LAS.zip | 199,658,713 | yes | yes | yes | no |
| 23001220PAS_SW_LAS.zip | 285,328,117 | yes | yes | yes | no |
| 23001230PAS_NE_LAS.zip | 156,704,615 | yes | yes | yes | no |
| 23001230PAS_NW_LAS.zip | 199,686,877 | yes | yes | yes | no |
| 23001230PAS_SE_LAS.zip | 161,617,797 | yes | yes | yes | no |
| 23001230PAS_SW_LAS.zip | 192,484,274 | yes | yes | yes | no |
| 23001240PAS_NE_LAS.zip | 201,498,376 | yes | yes | yes | no |
| 23001240PAS_NW_LAS.zip | 208,627,001 | yes | yes | yes | no |
| 23001240PAS_SE_LAS.zip | 178,165,571 | yes | yes | yes | no |
| 23001240PAS_SW_LAS.zip | 174,047,107 | yes | yes | yes | no |
| 23001250PAS_NE_LAS.zip | 208,522,098 | yes | yes | yes | no |
| 23001250PAS_NW_LAS.zip | 181,993,775 | yes | yes | yes | no |
| 23001250PAS_SE_LAS.zip | 209,000,694 | yes | yes | yes | no |
| 23001250PAS_SW_LAS.zip | 186,408,979 | yes | yes | yes | no |
| 23001260PAS_NE_LAS.zip | 207,128,594 | yes | yes | yes | no |
| 23001260PAS_NW_LAS.zip | 191,965,108 | yes | yes | yes | no |
| 23001260PAS_SE_LAS.zip | 220,220,759 | yes | yes | yes | no |
| 23001260PAS_SW_LAS.zip | 184,756,830 | yes | yes | yes | no |
| 23001270PAS_NE_LAS.zip | 215,331,662 | yes | yes | yes | no |
| 23001270PAS_NW_LAS.zip | 245,483,480 | yes | yes | yes | no |
| 23001270PAS_SE_LAS.zip | 203,784,250 | yes | yes | yes | no |
| 23001270PAS_SW_LAS.zip | 223,053,126 | yes | yes | yes | no |
| 23001280PAS_NE_LAS.zip | 186,079,994 | yes | yes | yes | no |
| 23001280PAS_NW_LAS.zip | 201,437,997 | yes | yes | yes | no |
| 23001280PAS_SE_LAS.zip | 198,258,831 | yes | yes | yes | no |
| 23001280PAS_SW_LAS.zip | 223,046,825 | yes | yes | yes | no |
| 23001290PAS_NE_LAS.zip | 235,549,350 | yes | yes | yes | no |
| 23001290PAS_NW_LAS.zip | 209,393,981 | yes | yes | yes | no |
| 23001290PAS_SE_LAS.zip | 239,024,085 | yes | yes | yes | no |
| 23001290PAS_SW_LAS.zip | 213,636,465 | yes | yes | yes | no |
| 23001300PAS_NE_LAS.zip | 167,539,616 | yes | yes | yes | no |
| 23001300PAS_NW_LAS.zip | 176,645,999 | yes | yes | yes | no |
| 23001300PAS_SE_LAS.zip | 167,201,376 | yes | yes | yes | no |
| 23001300PAS_SW_LAS.zip | 164,064,414 | yes | yes | yes | no |
| 23001310PAS_NE_LAS.zip | 162,054,156 | yes | yes | yes | no |
| 23001310PAS_NW_LAS.zip | 171,043,510 | yes | yes | yes | no |
| 23001310PAS_SE_LAS.zip | 193,264,653 | yes | yes | yes | no |
| 23001310PAS_SW_LAS.zip | 152,424,318 | yes | yes | yes | no |
| 23001320PAS_NE_LAS.zip | 185,948,425 | yes | yes | yes | no |
| 23001320PAS_NW_LAS.zip | 176,130,876 | yes | yes | yes | no |
| 23001320PAS_SE_LAS.zip | 204,243,793 | yes | yes | yes | no |
| 23001320PAS_SW_LAS.zip | 221,550,984 | yes | yes | yes | no |
| 23001330PAS_NE_LAS.zip | 149,390,708 | yes | yes | yes | no |
| 23001330PAS_NW_LAS.zip | 171,980,152 | yes | yes | yes | no |
| 23001330PAS_SW_LAS.zip | 198,496,859 | yes | yes | yes | no |
| 24001210PAS_SE_LAS.zip | 413,363,536 | yes | yes | yes | no |
| 24001220PAS_NE_LAS.zip | 129,653,718 | yes | yes | yes | no |
| 24001220PAS_NW_LAS.zip | 135,235,566 | yes | yes | yes | no |
| 24001220PAS_SE_LAS.zip | 201,370,253 | yes | yes | yes | no |
| 24001220PAS_SW_LAS.zip | 260,899,575 | yes | yes | yes | no |
| 24001230PAS_NE_LAS.zip | 169,488,744 | yes | yes | yes | no |
| 24001230PAS_NW_LAS.zip | 141,226,027 | yes | yes | yes | no |
| 24001230PAS_SE_LAS.zip | 166,921,104 | yes | yes | yes | no |
| 24001230PAS_SW_LAS.zip | 180,797,127 | yes | yes | yes | no |
| 24001240PAS_NE_LAS.zip | 200,029,230 | yes | yes | yes | no |
| 24001240PAS_NW_LAS.zip | 188,267,162 | yes | yes | yes | no |
| 24001240PAS_SE_LAS.zip | 183,503,733 | yes | yes | yes | no |
| 24001240PAS_SW_LAS.zip | 188,575,698 | yes | yes | yes | no |
| 24001250PAS_NE_LAS.zip | 189,242,314 | yes | yes | yes | no |
| 24001250PAS_NW_LAS.zip | 184,112,585 | yes | yes | yes | no |
| 24001250PAS_SE_LAS.zip | 186,125,733 | yes | yes | yes | no |
| 24001250PAS_SW_LAS.zip | 196,188,206 | yes | yes | yes | no |
| 24001260PAS_NE_LAS.zip | 193,680,218 | yes | yes | yes | no |
| 24001260PAS_NW_LAS.zip | 199,485,365 | yes | yes | yes | no |
| 24001260PAS_SE_LAS.zip | 198,616,464 | yes | yes | yes | no |
| 24001260PAS_SW_LAS.zip | 179,385,633 | yes | yes | yes | no |
| 24001270PAS_NE_LAS.zip | 175,208,944 | yes | yes | yes | no |
| 24001270PAS_NW_LAS.zip | 209,626,253 | yes | yes | yes | no |
| 24001270PAS_SE_LAS.zip | 200,582,054 | yes | yes | yes | no |
| 24001270PAS_SW_LAS.zip | 210,929,188 | yes | yes | yes | no |
| 24001280PAS_NE_LAS.zip | 170,329,755 | yes | yes | yes | no |
| 24001280PAS_NW_LAS.zip | 177,590,770 | yes | yes | yes | no |
| 24001280PAS_SE_LAS.zip | 194,131,468 | yes | yes | yes | no |
| 24001280PAS_SW_LAS.zip | 207,678,228 | yes | yes | yes | no |
| 24001290PAS_NE_LAS.zip | 195,057,484 | yes | yes | yes | no |
| 24001290PAS_NW_LAS.zip | 177,841,871 | yes | yes | yes | no |
| 24001290PAS_SE_LAS.zip | 197,847,960 | yes | yes | yes | no |
| 24001290PAS_SW_LAS.zip | 208,237,020 | yes | yes | yes | no |
| 24001300PAS_NE_LAS.zip | 207,178,645 | yes | yes | yes | no |
| 24001300PAS_NW_LAS.zip | 165,704,949 | yes | yes | yes | no |
| 24001300PAS_SE_LAS.zip | 181,507,903 | yes | yes | yes | no |
| 24001300PAS_SW_LAS.zip | 156,157,534 | yes | yes | yes | no |
| 24001310PAS_NE_LAS.zip | 157,960,960 | yes | yes | yes | no |
| 24001310PAS_NW_LAS.zip | 181,760,087 | yes | yes | yes | no |
| 24001310PAS_SE_LAS.zip | 192,386,002 | yes | yes | yes | no |
| 24001310PAS_SW_LAS.zip | 159,667,039 | yes | yes | yes | no |
| 24001320PAS_NE_LAS.zip | 171,110,227 | yes | yes | yes | no |
| 24001320PAS_NW_LAS.zip | 193,054,296 | yes | yes | yes | no |
| 24001320PAS_SE_LAS.zip | 165,510,179 | yes | yes | yes | no |
| 24001320PAS_SW_LAS.zip | 175,090,802 | yes | yes | yes | no |
| 24001330PAS_NW_LAS.zip | 204,824,878 | yes | yes | yes | no |
| 24001330PAS_SE_LAS.zip | 159,354,146 | yes | yes | yes | no |
| 24001330PAS_SW_LAS.zip | 156,117,766 | yes | yes | yes | no |
| 25001220PAS_SE_LAS.zip | 175,408,272 | yes | yes | yes | no |
| 25001230PAS_NE_LAS.zip | 151,236,837 | yes | yes | yes | no |
| 25001230PAS_SE_LAS.zip | 164,155,545 | yes | yes | yes | no |
| 25001230PAS_SW_LAS.zip | 165,441,692 | yes | yes | yes | no |
| 25001240PAS_NE_LAS.zip | 181,802,423 | yes | yes | yes | no |
| 25001240PAS_NW_LAS.zip | 182,851,093 | yes | yes | yes | no |
| 25001240PAS_SE_LAS.zip | 179,360,087 | yes | yes | yes | no |
| 25001240PAS_SW_LAS.zip | 200,225,106 | yes | yes | yes | no |
| 25001250PAS_NE_LAS.zip | 182,781,572 | yes | yes | yes | no |
| 25001250PAS_NW_LAS.zip | 202,291,058 | yes | yes | yes | no |
| 25001250PAS_SE_LAS.zip | 191,238,908 | yes | yes | yes | no |
| 25001250PAS_SW_LAS.zip | 184,253,276 | yes | yes | yes | no |
| 25001260PAS_NE_LAS.zip | 218,183,540 | yes | yes | yes | no |
| 25001260PAS_NW_LAS.zip | 203,518,602 | yes | yes | yes | no |
| 25001260PAS_SE_LAS.zip | 222,702,228 | yes | yes | yes | no |
| 25001260PAS_SW_LAS.zip | 179,074,793 | yes | yes | yes | no |
| 25001270PAS_NE_LAS.zip | 181,673,901 | yes | yes | yes | no |
| 25001270PAS_NW_LAS.zip | 247,890,669 | yes | yes | yes | no |
| 25001270PAS_SE_LAS.zip | 228,718,247 | yes | yes | yes | no |
| 25001270PAS_SW_LAS.zip | 212,941,972 | yes | yes | yes | no |
| 25001280PAS_NE_LAS.zip | 204,920,750 | yes | yes | yes | no |
| 25001280PAS_NW_LAS.zip | 197,021,726 | yes | yes | yes | no |
| 25001280PAS_SE_LAS.zip | 204,996,957 | yes | yes | yes | no |
| 25001280PAS_SW_LAS.zip | 177,320,859 | yes | yes | yes | no |
| 25001290PAS_NE_LAS.zip | 213,609,996 | yes | yes | yes | no |
| 25001290PAS_NW_LAS.zip | 206,737,091 | yes | yes | yes | no |
| 25001290PAS_SE_LAS.zip | 218,725,258 | yes | yes | yes | no |
| 25001290PAS_SW_LAS.zip | 212,575,265 | yes | yes | yes | no |
| 25001300PAS_NE_LAS.zip | 198,710,361 | yes | yes | yes | no |
| 25001300PAS_NW_LAS.zip | 193,015,415 | yes | yes | yes | no |
| 25001300PAS_SE_LAS.zip | 183,775,173 | yes | yes | yes | no |
| 25001300PAS_SW_LAS.zip | 181,060,895 | yes | yes | yes | no |
| 25001310PAS_NE_LAS.zip | 200,225,982 | yes | yes | yes | no |
| 25001310PAS_NW_LAS.zip | 199,946,520 | yes | yes | yes | no |
| 25001310PAS_SE_LAS.zip | 185,392,726 | yes | yes | yes | no |
| 25001310PAS_SW_LAS.zip | 182,616,712 | yes | yes | yes | no |
| 25001320PAS_NW_LAS.zip | 180,696,414 | yes | yes | yes | no |
| 25001320PAS_SE_LAS.zip | 183,547,953 | yes | yes | yes | no |
| 25001320PAS_SW_LAS.zip | 198,943,772 | yes | yes | yes | no |
| 25001330PAS_SW_LAS.zip | 175,340,370 | yes | yes | yes | no |
| 26001240PAS_NE_LAS.zip | 176,761,817 | yes | yes | yes | no |
| 26001240PAS_NW_LAS.zip | 197,984,918 | yes | yes | yes | no |
| 26001240PAS_SE_LAS.zip | 161,377,433 | yes | yes | yes | no |
| 26001240PAS_SW_LAS.zip | 225,239,896 | yes | yes | yes | no |
| 26001250PAS_NE_LAS.zip | 188,840,318 | yes | yes | yes | no |
| 26001250PAS_NW_LAS.zip | 217,466,010 | yes | yes | yes | no |
| 26001250PAS_SE_LAS.zip | 171,671,780 | yes | yes | yes | no |
| 26001250PAS_SW_LAS.zip | 187,555,353 | yes | yes | yes | no |
| 26001260PAS_NE_LAS.zip | 201,254,154 | yes | yes | yes | no |
| 26001260PAS_NW_LAS.zip | 210,131,582 | yes | yes | yes | no |
| 26001260PAS_SE_LAS.zip | 226,576,139 | yes | yes | yes | no |
| 26001260PAS_SW_LAS.zip | 216,515,005 | yes | yes | yes | no |
| 26001270PAS_NE_LAS.zip | 217,324,406 | yes | yes | yes | no |
| 26001270PAS_NW_LAS.zip | 247,171,095 | yes | yes | yes | no |
| 26001270PAS_SE_LAS.zip | 155,209,410 | yes | yes | yes | no |
| 26001270PAS_SW_LAS.zip | 210,897,406 | yes | yes | yes | no |
| 26001280PAS_NE_LAS.zip | 195,382,942 | yes | yes | yes | no |
| 26001280PAS_NW_LAS.zip | 170,345,504 | yes | yes | yes | no |
| 26001280PAS_SE_LAS.zip | 186,127,484 | yes | yes | yes | no |
| 26001280PAS_SW_LAS.zip | 186,513,315 | yes | yes | yes | no |
| 26001290PAS_NW_LAS.zip | 215,868,518 | yes | yes | yes | no |
| 26001290PAS_SE_LAS.zip | 233,696,056 | yes | yes | yes | no |
| 26001290PAS_SW_LAS.zip | 219,884,086 | yes | yes | yes | no |
| 26001300PAS_SE_LAS.zip | 181,887,043 | yes | yes | yes | no |
| 26001300PAS_SW_LAS.zip | 221,029,624 | yes | yes | yes | no |
| 26001310PAS_SW_LAS.zip | 190,698,061 | yes | yes | yes | no |
| 27001240PAS_SE_LAS.zip | 179,416,972 | yes | yes | yes | no |
| 27001250PAS_SE_LAS.zip | 179,880,901 | yes | yes | yes | no |
| 27001250PAS_SW_LAS.zip | 187,613,374 | yes | yes | yes | no |


**USGS LAZ tiles** (rockyweb `PA_WesternPA_2019_D20/PA_WesternPA_1_2019/LAZ/`): 300 files, 39.37 GB, 3,795,340,548 points (Entwine source list); the project lists 7254 LAZ, 902 GB.

| LAZ | bytes (listing) | points | sha256 (16) |
|---|---|---|---|
| USGS_LPC_PA_WesternPA_2019_D20_17SNE549417.laz | 259,201,199 | 23,765,546 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE549419.laz | 272,789,434 | 24,999,162 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE549420.laz | 262,400,459 | 23,996,421 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE549422.laz | 150,359,772 | 14,070,480 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE550413.laz | 123,150,600 | 11,851,760 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE550414.laz | 131,525,772 | 12,571,228 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE550416.laz | 98,401,873 | 9,699,985 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE550417.laz | 118,400,763 | 11,509,818 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE550419.laz | 123,007,371 | 11,887,034 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE550420.laz | 101,003,885 | 9,973,807 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE550422.laz | 83,212,015 | 8,208,821 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE550423.laz | 82,093,846 | 8,215,419 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE552411.laz | 149,278,654 | 14,234,934 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE552413.laz | 138,100,194 | 13,276,172 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE552414.laz | 137,167,089 | 13,248,066 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE552416.laz | 118,837,180 | 11,582,532 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE552417.laz | 134,555,088 | 13,007,219 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE552419.laz | 134,002,716 | 12,926,297 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE552420.laz | 135,826,968 | 13,136,286 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE552422.laz | 89,198,090 | 8,835,924 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE552423.laz | 110,699,731 | 10,763,700 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE553405.laz | 150,556,137 | 14,326,570 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE553407.laz | 151,070,072 | 14,357,419 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE553408.laz | 155,928,793 | 14,868,520 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE553410.laz | 133,560,580 | 12,879,645 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE553411.laz | 127,159,266 | 12,295,949 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE553413.laz | 133,188,620 | 12,855,150 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE553414.laz | 119,836,884 | 11,612,873 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE553416.laz | 117,670,622 | 11,466,180 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE553417.laz | 127,235,617 | 12,324,028 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE553419.laz | 122,180,684 | 11,796,900 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE553420.laz | 122,862,716 | 11,904,042 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE553422.laz | 112,951,245 | 10,949,747 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE553423.laz | 126,461,852 | 12,194,134 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE553425.laz | 110,240,261 | 10,792,580 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE555405.laz | 152,292,419 | 14,488,243 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE555407.laz | 144,237,423 | 13,748,215 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE555408.laz | 162,567,983 | 15,400,134 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE555410.laz | 151,903,509 | 14,430,970 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE555411.laz | 140,729,553 | 13,446,060 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE555413.laz | 149,048,965 | 14,202,787 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE555414.laz | 133,315,027 | 12,804,348 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE555416.laz | 116,567,178 | 11,300,725 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE555417.laz | 122,252,798 | 11,777,409 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE555419.laz | 133,930,673 | 12,841,311 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE555420.laz | 133,696,963 | 12,831,666 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE555422.laz | 133,095,045 | 12,731,805 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE555423.laz | 124,479,476 | 12,028,818 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE555425.laz | 127,711,623 | 12,292,609 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE555426.laz | 141,214,399 | 13,516,995 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE556405.laz | 142,320,699 | 13,646,468 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE556407.laz | 139,018,936 | 13,387,328 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE556408.laz | 136,614,390 | 13,169,680 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE556410.laz | 132,550,555 | 12,761,362 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE556411.laz | 130,097,242 | 12,563,814 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE556413.laz | 125,654,859 | 12,164,785 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE556414.laz | 125,662,650 | 12,201,706 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE556416.laz | 132,721,051 | 12,788,704 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE556417.laz | 113,640,132 | 11,139,861 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE556419.laz | 126,705,654 | 12,296,528 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE556420.laz | 118,685,569 | 11,547,323 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE556422.laz | 123,282,389 | 11,948,800 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE556423.laz | 126,216,383 | 12,241,530 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE556425.laz | 122,548,622 | 11,840,520 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE556426.laz | 109,117,989 | 10,710,001 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE558405.laz | 146,687,611 | 14,049,375 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE558407.laz | 137,351,165 | 13,225,994 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE558408.laz | 137,224,528 | 13,211,101 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE558410.laz | 134,679,783 | 12,971,739 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE558411.laz | 122,814,523 | 11,921,155 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE558413.laz | 118,247,694 | 11,480,612 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE558414.laz | 128,170,403 | 12,402,193 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE558416.laz | 125,573,582 | 12,166,000 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE558417.laz | 110,604,096 | 10,810,114 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE558419.laz | 98,839,102 | 9,810,490 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE558420.laz | 107,398,307 | 10,468,635 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE558422.laz | 119,137,495 | 11,512,984 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE558423.laz | 101,104,455 | 9,931,727 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE558425.laz | 114,080,403 | 11,135,128 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE558426.laz | 110,306,866 | 10,796,654 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE559405.laz | 144,565,928 | 13,770,515 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE559407.laz | 136,489,605 | 13,055,752 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE559408.laz | 135,997,410 | 13,006,067 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE559410.laz | 141,463,419 | 13,487,916 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE559411.laz | 148,650,621 | 14,120,329 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE559413.laz | 134,180,102 | 12,865,488 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE559414.laz | 137,023,867 | 13,100,354 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE559416.laz | 133,520,575 | 12,781,402 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE559417.laz | 141,447,226 | 13,527,882 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE559419.laz | 140,365,421 | 13,477,037 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE559420.laz | 124,786,024 | 12,040,243 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE559422.laz | 137,031,084 | 13,131,744 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE559423.laz | 127,128,129 | 12,250,966 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE559425.laz | 124,984,605 | 12,083,311 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE559426.laz | 127,453,020 | 12,311,741 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE561405.laz | 161,312,673 | 15,265,434 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE561407.laz | 144,743,371 | 13,840,121 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE561408.laz | 139,545,079 | 13,346,550 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE561410.laz | 147,204,351 | 13,993,426 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE561411.laz | 125,984,937 | 12,173,099 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE561413.laz | 150,423,383 | 14,283,600 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE561414.laz | 122,247,669 | 11,826,902 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE561416.laz | 136,798,333 | 13,127,506 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE561417.laz | 141,775,636 | 13,590,092 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE561419.laz | 161,321,591 | 15,245,798 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE561420.laz | 144,668,969 | 13,749,809 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE561422.laz | 146,649,888 | 14,005,904 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE561423.laz | 142,248,542 | 13,603,683 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE561425.laz | 141,368,416 | 13,497,709 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE561426.laz | 144,654,216 | 13,840,511 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE561428.laz | 142,884,582 | 13,682,383 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE562405.laz | 122,787,086 | 11,948,678 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE562407.laz | 133,615,015 | 12,873,413 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE562408.laz | 133,159,075 | 12,796,399 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE562410.laz | 146,548,062 | 14,031,025 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE562411.laz | 121,087,489 | 11,725,888 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE562413.laz | 140,100,602 | 13,474,073 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE562414.laz | 118,738,050 | 11,515,028 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE562416.laz | 116,818,573 | 11,349,301 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE562417.laz | 127,642,536 | 12,399,689 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE562419.laz | 136,906,090 | 13,176,382 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE562420.laz | 133,200,557 | 12,875,870 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE562422.laz | 133,730,410 | 12,873,749 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE562423.laz | 129,680,389 | 12,516,943 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE562425.laz | 137,247,875 | 13,211,938 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE562426.laz | 152,915,417 | 14,638,370 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE562428.laz | 142,799,328 | 13,661,095 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE564405.laz | 125,829,684 | 12,219,919 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE564407.laz | 146,403,224 | 14,110,341 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE564408.laz | 130,485,320 | 12,617,461 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE564410.laz | 129,970,596 | 12,565,512 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE564411.laz | 109,958,767 | 10,757,303 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE564413.laz | 120,570,225 | 11,742,213 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE564414.laz | 110,776,752 | 10,847,168 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE564416.laz | 128,772,550 | 12,506,318 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE564417.laz | 134,574,999 | 13,027,062 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE564419.laz | 142,633,839 | 13,725,441 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE564420.laz | 132,890,826 | 12,869,534 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE564422.laz | 125,832,632 | 12,237,179 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE564423.laz | 136,840,334 | 13,255,701 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE564425.laz | 147,055,567 | 14,227,198 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE564426.laz | 152,549,960 | 14,608,255 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE564428.laz | 140,492,415 | 13,562,318 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE565405.laz | 147,407,294 | 14,179,731 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE565407.laz | 156,948,519 | 14,955,098 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE565408.laz | 147,170,478 | 14,114,360 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE565410.laz | 124,652,164 | 12,089,864 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE565411.laz | 131,175,944 | 12,689,538 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE565413.laz | 119,297,748 | 11,595,346 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE565414.laz | 130,252,494 | 12,563,819 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE565416.laz | 121,954,452 | 11,855,649 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE565417.laz | 136,806,037 | 13,158,116 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE565419.laz | 146,022,275 | 13,992,606 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE565420.laz | 139,260,462 | 13,389,641 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE565422.laz | 128,367,400 | 12,346,128 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE565423.laz | 138,001,898 | 13,353,003 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE565425.laz | 149,191,297 | 14,356,864 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE565426.laz | 117,163,650 | 11,476,535 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE565428.laz | 113,131,080 | 11,153,212 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE567405.laz | 148,467,696 | 14,184,147 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE567407.laz | 141,460,968 | 13,578,891 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE567408.laz | 133,815,550 | 12,876,801 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE567410.laz | 130,868,114 | 12,612,917 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE567411.laz | 135,539,339 | 13,007,501 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE567413.laz | 141,349,800 | 13,553,827 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE567414.laz | 111,126,767 | 10,761,776 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE567416.laz | 114,566,280 | 11,047,942 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE567417.laz | 138,953,758 | 13,239,180 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE567419.laz | 145,183,828 | 13,865,137 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE567420.laz | 136,499,757 | 13,110,627 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE567422.laz | 127,155,733 | 12,335,279 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE567423.laz | 128,797,047 | 12,476,019 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE567425.laz | 121,875,306 | 11,823,592 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE567426.laz | 122,449,209 | 11,794,661 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE567428.laz | 134,361,870 | 12,964,515 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE568405.laz | 151,249,733 | 14,449,233 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE568407.laz | 137,489,557 | 13,227,136 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE568408.laz | 135,098,085 | 13,027,414 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE568410.laz | 130,519,535 | 12,595,596 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE568411.laz | 132,071,171 | 12,690,470 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE568413.laz | 143,232,953 | 13,686,698 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE568414.laz | 142,318,139 | 13,506,080 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE568416.laz | 117,046,117 | 11,104,541 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE568417.laz | 161,987,767 | 15,277,834 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE568419.laz | 153,791,128 | 14,625,391 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE568420.laz | 151,370,391 | 14,423,300 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE568422.laz | 134,702,577 | 12,970,662 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE568423.laz | 130,798,481 | 12,582,138 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE568425.laz | 165,003,656 | 15,611,144 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE568426.laz | 156,114,900 | 14,822,222 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE568428.laz | 150,209,785 | 14,295,765 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE570407.laz | 122,538,953 | 11,866,915 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE570408.laz | 107,542,572 | 10,599,613 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE570410.laz | 107,194,071 | 10,588,906 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE570411.laz | 126,747,337 | 12,279,775 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE570413.laz | 119,960,951 | 11,680,908 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE570414.laz | 129,067,088 | 12,446,022 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE570416.laz | 106,944,175 | 10,356,657 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE570417.laz | 135,601,081 | 13,052,720 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE570419.laz | 138,131,987 | 13,279,868 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE570420.laz | 137,465,378 | 13,224,140 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE570422.laz | 131,928,872 | 12,670,477 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE570423.laz | 136,074,728 | 13,052,592 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE570425.laz | 129,776,404 | 12,520,482 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE570426.laz | 125,735,693 | 12,220,785 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE570428.laz | 141,305,299 | 13,620,334 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE571410.laz | 102,872,179 | 10,202,934 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE571411.laz | 111,352,220 | 10,942,613 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE571413.laz | 103,644,532 | 10,255,281 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE571414.laz | 110,028,918 | 10,765,044 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE571416.laz | 92,279,178 | 9,029,070 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE571417.laz | 131,587,583 | 12,684,816 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE571419.laz | 134,646,135 | 13,024,845 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE571420.laz | 140,715,902 | 13,544,639 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE571422.laz | 112,405,544 | 10,968,648 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE571423.laz | 117,756,585 | 11,512,537 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE571425.laz | 138,088,725 | 13,370,702 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE571426.laz | 145,997,400 | 14,077,715 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE571428.laz | 139,523,289 | 13,434,960 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE573411.laz | 136,980,562 | 13,172,613 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE573413.laz | 135,982,620 | 13,112,981 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE573414.laz | 113,237,348 | 11,047,982 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE573416.laz | 100,786,961 | 9,891,994 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE573417.laz | 124,897,484 | 12,024,787 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE573419.laz | 146,985,765 | 14,097,313 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE573420.laz | 137,887,809 | 13,280,084 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE573422.laz | 125,953,760 | 12,213,181 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE573423.laz | 127,394,218 | 12,350,401 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE573425.laz | 127,156,136 | 12,330,038 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE573426.laz | 136,577,805 | 13,195,063 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE573428.laz | 140,684,467 | 13,534,867 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE574411.laz | 133,101,042 | 12,850,991 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE574413.laz | 115,099,909 | 11,208,253 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE574414.laz | 110,482,385 | 10,846,717 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE574416.laz | 117,800,282 | 11,448,920 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE574417.laz | 112,341,897 | 10,904,302 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE574419.laz | 107,597,471 | 10,543,737 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE574420.laz | 112,339,068 | 11,045,490 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE574422.laz | 100,657,576 | 9,967,363 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE574423.laz | 116,661,356 | 11,347,060 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE574425.laz | 124,254,916 | 12,108,668 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE574426.laz | 140,398,422 | 13,521,969 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE576411.laz | 152,564,582 | 14,599,189 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE576413.laz | 141,187,586 | 13,575,372 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE576414.laz | 157,108,045 | 14,984,059 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE576416.laz | 141,421,489 | 13,558,994 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE576417.laz | 140,537,261 | 13,521,985 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE576419.laz | 112,414,395 | 11,003,564 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE576420.laz | 132,453,529 | 12,835,626 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE576422.laz | 137,734,843 | 13,246,423 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE576423.laz | 154,464,498 | 14,753,985 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE576425.laz | 144,772,037 | 13,888,952 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE576426.laz | 145,395,800 | 13,979,367 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE577416.laz | 119,829,581 | 11,743,993 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE577417.laz | 121,176,977 | 11,802,364 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE577419.laz | 115,763,598 | 11,221,391 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE577420.laz | 134,102,756 | 12,929,860 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE577422.laz | 115,176,807 | 11,370,497 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE577423.laz | 123,145,559 | 11,988,192 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE577425.laz | 131,049,622 | 12,700,907 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE577426.laz | 127,608,023 | 12,377,864 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE579416.laz | 117,623,690 | 11,465,368 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE579417.laz | 124,513,017 | 12,044,056 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE579419.laz | 104,842,180 | 10,191,223 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE579420.laz | 95,001,986 | 9,277,616 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE579422.laz | 98,484,902 | 9,772,895 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE579423.laz | 104,545,149 | 10,332,489 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE579425.laz | 115,481,341 | 11,267,296 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE579426.laz | 98,673,151 | 9,819,951 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE580417.laz | 135,816,185 | 13,133,763 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE580419.laz | 145,586,953 | 13,985,220 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE580420.laz | 122,029,253 | 11,861,378 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE580422.laz | 122,176,818 | 11,913,261 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE580423.laz | 129,284,882 | 12,432,804 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE580425.laz | 126,137,351 | 12,156,591 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE580426.laz | 123,483,935 | 12,005,537 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE582417.laz | 108,072,437 | 10,724,501 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE582419.laz | 143,665,202 | 13,818,955 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE582420.laz | 111,478,425 | 11,040,880 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE582422.laz | 108,610,106 | 10,795,264 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE582423.laz | 124,228,028 | 12,156,389 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE582425.laz | 122,117,966 | 11,785,717 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE583419.laz | 135,106,336 | 13,066,473 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE583420.laz | 118,118,525 | 11,549,320 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE583422.laz | 121,476,170 | 11,901,832 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE583423.laz | 132,554,896 | 12,609,298 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE585420.laz | 104,649,186 | 10,344,852 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17SNE585422.laz | 113,466,977 | 10,967,089 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17TNE555428.laz | 140,425,144 | 13,440,949 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17TNE556428.laz | 125,416,139 | 12,182,498 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17TNE558428.laz | 112,915,371 | 10,990,022 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17TNE558429.laz | 122,884,568 | 11,892,123 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17TNE559428.laz | 132,542,682 | 12,767,209 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17TNE559429.laz | 156,900,801 | 14,974,091 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17TNE561429.laz | 132,317,309 | 12,658,622 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17TNE562429.laz | 137,805,713 | 13,251,386 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17TNE564429.laz | 149,668,586 | 14,352,796 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17TNE565429.laz | 144,424,398 | 13,888,963 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17TNE568429.laz | 137,712,169 | 13,198,482 | not downloaded (Entwine copy used) |
| USGS_LPC_PA_WesternPA_2019_D20_17TNE570429.laz | 122,963,801 | 11,965,295 | not downloaded (Entwine copy used) |


**Marin 2019 tiles** (`marin_select/marin_select.json`): pool watersheds 138.1 km2 (2994 terminal basins from 420 candidate windows), buffered 184.3 km2; 237 tiles, 198 in the Entwine source list, 4,145,002,910 points, listed bytes 0.00 GB, estimate 14.51 GB.

| tile | file | in source list | points | bytes (listing) |
|---|---|---|---|---|
| w2275n1966 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2275n1966.laz | yes | 5,351,840 | - |
| w2275n1967 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2275n1967.laz | yes | 8,441,093 | - |
| w2275n1968 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2275n1968.laz | yes | 15,749,709 | - |
| w2275n1969 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2275n1969.laz | yes | 9,689,419 | - |
| w2275n1970 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2275n1970.laz | yes | 514,274 | - |
| w2276n1966 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2276n1966.laz | yes | 3,760,426 | - |
| w2276n1967 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2276n1967.laz | yes | 8,597,521 | - |
| w2276n1968 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2276n1968.laz | yes | 13,151,778 | - |
| w2276n1969 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2276n1969.laz | yes | 15,304,968 | - |
| w2276n1970 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2276n1970.laz | yes | 8,743,777 | - |
| w2276n1971 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2276n1971.laz | yes | 4,573,701 | - |
| w2276n1972 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2276n1972.laz | yes | 4,528,377 | - |
| w2276n1973 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2276n1973.laz | yes | 712,655 | - |
| w2277n1959 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2277n1959.laz | no | - | - |
| w2277n1960 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2277n1960.laz | no | - | - |
| w2277n1966 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2277n1966.laz | yes | 5,800,467 | - |
| w2277n1967 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2277n1967.laz | yes | 10,712,785 | - |
| w2277n1968 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2277n1968.laz | yes | 8,976,634 | - |
| w2277n1969 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2277n1969.laz | yes | 11,720,545 | - |
| w2277n1970 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2277n1970.laz | yes | 14,979,910 | - |
| w2277n1971 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2277n1971.laz | yes | 14,435,850 | - |
| w2277n1972 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2277n1972.laz | yes | 13,386,498 | - |
| w2277n1973 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2277n1973.laz | yes | 8,358,572 | - |
| w2277n1974 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2277n1974.laz | yes | 1,051,051 | - |
| w2278n1956 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2278n1956.laz | no | - | - |
| w2278n1957 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2278n1957.laz | no | - | - |
| w2278n1958 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2278n1958.laz | no | - | - |
| w2278n1959 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2278n1959.laz | no | - | - |
| w2278n1960 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2278n1960.laz | no | - | - |
| w2278n1963 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2278n1963.laz | yes | 3,927,790 | - |
| w2278n1964 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2278n1964.laz | yes | 3,664,078 | - |
| w2278n1965 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2278n1965.laz | yes | 3,773,531 | - |
| w2278n1966 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2278n1966.laz | yes | 4,022,402 | - |
| w2278n1967 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2278n1967.laz | yes | 3,215,624 | - |
| w2278n1968 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2278n1968.laz | yes | 3,701,269 | - |
| w2278n1969 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2278n1969.laz | yes | 3,837,278 | - |
| w2278n1970 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2278n1970.laz | yes | 11,723,966 | - |
| w2278n1971 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2278n1971.laz | yes | 13,917,132 | - |
| w2278n1972 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2278n1972.laz | yes | 12,467,851 | - |
| w2278n1973 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2278n1973.laz | yes | 13,265,983 | - |
| w2278n1974 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2278n1974.laz | yes | 10,672,763 | - |
| w2278n1975 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2278n1975.laz | yes | 10,060,306 | - |
| w2279n1955 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1955.laz | no | - | - |
| w2279n1956 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1956.laz | no | - | - |
| w2279n1957 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1957.laz | no | - | - |
| w2279n1958 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1958.laz | no | - | - |
| w2279n1959 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1959.laz | no | - | - |
| w2279n1960 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1960.laz | no | - | - |
| w2279n1961 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1961.laz | no | - | - |
| w2279n1962 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1962.laz | yes | 6,661,394 | - |
| w2279n1963 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1963.laz | yes | 13,086,579 | - |
| w2279n1964 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1964.laz | yes | 14,551,183 | - |
| w2279n1965 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1965.laz | yes | 14,568,351 | - |
| w2279n1966 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1966.laz | yes | 12,845,451 | - |
| w2279n1967 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1967.laz | yes | 4,474,599 | - |
| w2279n1968 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1968.laz | yes | 3,716,636 | - |
| w2279n1969 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1969.laz | yes | 6,235,569 | - |
| w2279n1970 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1970.laz | yes | 10,150,417 | - |
| w2279n1971 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1971.laz | yes | 13,817,569 | - |
| w2279n1972 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1972.laz | yes | 16,268,049 | - |
| w2279n1973 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1973.laz | yes | 16,123,021 | - |
| w2279n1974 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1974.laz | yes | 14,216,376 | - |
| w2279n1975 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2279n1975.laz | yes | 17,531,198 | - |
| w2280n1955 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1955.laz | no | - | - |
| w2280n1956 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1956.laz | no | - | - |
| w2280n1957 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1957.laz | no | - | - |
| w2280n1958 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1958.laz | no | - | - |
| w2280n1959 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1959.laz | no | - | - |
| w2280n1960 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1960.laz | no | - | - |
| w2280n1961 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1961.laz | no | - | - |
| w2280n1962 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1962.laz | yes | 4,964,032 | - |
| w2280n1963 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1963.laz | yes | 24,976,765 | - |
| w2280n1964 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1964.laz | yes | 18,056,307 | - |
| w2280n1965 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1965.laz | yes | 15,182,443 | - |
| w2280n1966 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1966.laz | yes | 17,141,514 | - |
| w2280n1967 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1967.laz | yes | 14,437,659 | - |
| w2280n1968 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1968.laz | yes | 9,746,493 | - |
| w2280n1969 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1969.laz | yes | 8,320,449 | - |
| w2280n1970 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1970.laz | yes | 14,874,055 | - |
| w2280n1971 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1971.laz | yes | 18,242,717 | - |
| w2280n1972 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1972.laz | yes | 19,130,905 | - |
| w2280n1973 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1973.laz | yes | 30,311,731 | - |
| w2280n1974 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1974.laz | yes | 26,939,810 | - |
| w2280n1975 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2280n1975.laz | yes | 28,271,857 | - |
| w2281n1955 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1955.laz | no | - | - |
| w2281n1956 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1956.laz | no | - | - |
| w2281n1957 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1957.laz | no | - | - |
| w2281n1958 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1958.laz | no | - | - |
| w2281n1959 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1959.laz | no | - | - |
| w2281n1960 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1960.laz | no | - | - |
| w2281n1962 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1962.laz | yes | 4,348,126 | - |
| w2281n1963 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1963.laz | yes | 19,023,743 | - |
| w2281n1964 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1964.laz | yes | 24,436,455 | - |
| w2281n1965 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1965.laz | yes | 19,513,373 | - |
| w2281n1966 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1966.laz | yes | 14,281,362 | - |
| w2281n1967 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1967.laz | yes | 18,662,327 | - |
| w2281n1968 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1968.laz | yes | 20,676,754 | - |
| w2281n1969 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1969.laz | yes | 23,628,508 | - |
| w2281n1970 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1970.laz | yes | 17,504,940 | - |
| w2281n1971 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1971.laz | yes | 24,989,426 | - |
| w2281n1972 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1972.laz | yes | 29,822,195 | - |
| w2281n1973 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1973.laz | yes | 41,693,049 | - |
| w2281n1974 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1974.laz | yes | 40,866,566 | - |
| w2281n1975 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2281n1975.laz | yes | 34,952,698 | - |
| w2282n1955 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1955.laz | no | - | - |
| w2282n1956 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1956.laz | no | - | - |
| w2282n1957 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1957.laz | no | - | - |
| w2282n1958 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1958.laz | no | - | - |
| w2282n1959 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1959.laz | no | - | - |
| w2282n1962 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1962.laz | yes | 3,751,077 | - |
| w2282n1963 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1963.laz | yes | 9,610,863 | - |
| w2282n1964 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1964.laz | yes | 20,288,310 | - |
| w2282n1965 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1965.laz | yes | 23,894,722 | - |
| w2282n1966 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1966.laz | yes | 16,656,186 | - |
| w2282n1967 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1967.laz | yes | 21,233,143 | - |
| w2282n1968 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1968.laz | yes | 34,214,134 | - |
| w2282n1969 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1969.laz | yes | 36,633,577 | - |
| w2282n1970 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1970.laz | yes | 34,257,451 | - |
| w2282n1971 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1971.laz | yes | 32,614,954 | - |
| w2282n1972 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1972.laz | yes | 32,043,076 | - |
| w2282n1973 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1973.laz | yes | 33,644,371 | - |
| w2282n1974 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1974.laz | yes | 33,972,638 | - |
| w2282n1975 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1975.laz | yes | 31,372,940 | - |
| w2282n1976 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2282n1976.laz | yes | 28,035,083 | - |
| w2283n1956 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2283n1956.laz | no | - | - |
| w2283n1957 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2283n1957.laz | no | - | - |
| w2283n1958 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2283n1958.laz | no | - | - |
| w2283n1959 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2283n1959.laz | no | - | - |
| w2283n1963 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2283n1963.laz | yes | 6,485,457 | - |
| w2283n1964 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2283n1964.laz | yes | 13,826,061 | - |
| w2283n1965 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2283n1965.laz | yes | 18,275,619 | - |
| w2283n1966 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2283n1966.laz | yes | 21,643,639 | - |
| w2283n1967 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2283n1967.laz | yes | 28,026,801 | - |
| w2283n1968 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2283n1968.laz | yes | 31,998,817 | - |
| w2283n1969 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2283n1969.laz | yes | 37,404,147 | - |
| w2283n1970 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2283n1970.laz | yes | 31,692,917 | - |
| w2283n1971 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2283n1971.laz | yes | 30,618,160 | - |
| w2283n1972 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2283n1972.laz | yes | 27,781,221 | - |
| w2283n1973 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2283n1973.laz | yes | 25,225,891 | - |
| w2283n1974 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2283n1974.laz | yes | 24,750,381 | - |
| w2283n1975 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2283n1975.laz | yes | 23,769,647 | - |
| w2283n1976 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2283n1976.laz | yes | 24,824,026 | - |
| w2284n1957 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2284n1957.laz | no | - | - |
| w2284n1958 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2284n1958.laz | no | - | - |
| w2284n1959 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2284n1959.laz | no | - | - |
| w2284n1963 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2284n1963.laz | yes | 6,838,320 | - |
| w2284n1964 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2284n1964.laz | yes | 8,004,416 | - |
| w2284n1965 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2284n1965.laz | yes | 12,674,717 | - |
| w2284n1966 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2284n1966.laz | yes | 19,171,674 | - |
| w2284n1967 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2284n1967.laz | yes | 29,758,589 | - |
| w2284n1968 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2284n1968.laz | yes | 28,661,664 | - |
| w2284n1969 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2284n1969.laz | yes | 23,730,267 | - |
| w2284n1970 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2284n1970.laz | yes | 25,699,812 | - |
| w2284n1971 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2284n1971.laz | yes | 20,707,624 | - |
| w2284n1972 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2284n1972.laz | yes | 22,229,820 | - |
| w2284n1973 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2284n1973.laz | yes | 27,664,230 | - |
| w2284n1974 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2284n1974.laz | yes | 27,658,335 | - |
| w2284n1975 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2284n1975.laz | yes | 24,483,200 | - |
| w2284n1976 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2284n1976.laz | yes | 27,514,241 | - |
| w2285n1964 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2285n1964.laz | yes | 694,447 | - |
| w2285n1965 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2285n1965.laz | yes | 6,386,871 | - |
| w2285n1966 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2285n1966.laz | yes | 9,911,480 | - |
| w2285n1967 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2285n1967.laz | yes | 24,002,447 | - |
| w2285n1968 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2285n1968.laz | yes | 27,357,166 | - |
| w2285n1969 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2285n1969.laz | yes | 26,917,891 | - |
| w2285n1970 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2285n1970.laz | yes | 21,672,759 | - |
| w2285n1971 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2285n1971.laz | yes | 24,046,723 | - |
| w2285n1972 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2285n1972.laz | yes | 31,496,799 | - |
| w2285n1973 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2285n1973.laz | yes | 32,767,694 | - |
| w2285n1974 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2285n1974.laz | yes | 31,893,636 | - |
| w2285n1975 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2285n1975.laz | yes | 25,523,816 | - |
| w2285n1976 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2285n1976.laz | yes | 28,812,884 | - |
| w2285n1977 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2285n1977.laz | yes | 32,048,462 | - |
| w2286n1965 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2286n1965.laz | yes | 658,876 | - |
| w2286n1966 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2286n1966.laz | yes | 3,065,432 | - |
| w2286n1967 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2286n1967.laz | yes | 6,538,681 | - |
| w2286n1968 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2286n1968.laz | yes | 23,455,206 | - |
| w2286n1969 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2286n1969.laz | yes | 28,372,261 | - |
| w2286n1970 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2286n1970.laz | yes | 32,111,509 | - |
| w2286n1971 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2286n1971.laz | yes | 30,478,855 | - |
| w2286n1972 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2286n1972.laz | yes | 33,902,616 | - |
| w2286n1973 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2286n1973.laz | yes | 41,281,012 | - |
| w2286n1974 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2286n1974.laz | yes | 32,889,578 | - |
| w2286n1975 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2286n1975.laz | yes | 26,603,587 | - |
| w2286n1976 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2286n1976.laz | yes | 31,665,687 | - |
| w2286n1977 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2286n1977.laz | yes | 35,836,118 | - |
| w2287n1967 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2287n1967.laz | yes | 707,978 | - |
| w2287n1968 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2287n1968.laz | yes | 5,716,601 | - |
| w2287n1969 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2287n1969.laz | yes | 21,083,096 | - |
| w2287n1970 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2287n1970.laz | yes | 33,582,556 | - |
| w2287n1971 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2287n1971.laz | yes | 32,845,365 | - |
| w2287n1972 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2287n1972.laz | yes | 39,445,012 | - |
| w2287n1973 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2287n1973.laz | yes | 46,110,390 | - |
| w2287n1974 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2287n1974.laz | yes | 46,656,044 | - |
| w2287n1975 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2287n1975.laz | yes | 38,975,837 | - |
| w2287n1976 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2287n1976.laz | yes | 41,320,710 | - |
| w2287n1977 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2287n1977.laz | yes | 41,738,176 | - |
| w2288n1968 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2288n1968.laz | yes | 719,385 | - |
| w2288n1969 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2288n1969.laz | yes | 3,914,344 | - |
| w2288n1970 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2288n1970.laz | yes | 15,605,833 | - |
| w2288n1971 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2288n1971.laz | yes | 26,553,397 | - |
| w2288n1972 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2288n1972.laz | yes | 33,315,128 | - |
| w2288n1973 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2288n1973.laz | yes | 42,431,411 | - |
| w2288n1974 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2288n1974.laz | yes | 36,025,555 | - |
| w2288n1975 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2288n1975.laz | yes | 35,509,187 | - |
| w2288n1976 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2288n1976.laz | yes | 36,083,162 | - |
| w2288n1977 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2288n1977.laz | yes | 39,727,043 | - |
| w2288n1978 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2288n1978.laz | yes | 39,534,929 | - |
| w2289n1970 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2289n1970.laz | yes | 1,229,443 | - |
| w2289n1971 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2289n1971.laz | yes | 13,837,182 | - |
| w2289n1972 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2289n1972.laz | yes | 28,207,300 | - |
| w2289n1973 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2289n1973.laz | yes | 46,158,319 | - |
| w2289n1974 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2289n1974.laz | yes | 36,929,744 | - |
| w2289n1975 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2289n1975.laz | yes | 32,476,022 | - |
| w2289n1976 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2289n1976.laz | yes | 27,716,783 | - |
| w2289n1977 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2289n1977.laz | yes | 38,194,151 | - |
| w2289n1978 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2289n1978.laz | yes | 34,924,747 | - |
| w2290n1971 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2290n1971.laz | yes | 3,990,407 | - |
| w2290n1972 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2290n1972.laz | yes | 13,993,199 | - |
| w2290n1973 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2290n1973.laz | yes | 25,121,398 | - |
| w2290n1974 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2290n1974.laz | yes | 38,274,046 | - |
| w2290n1975 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2290n1975.laz | yes | 39,209,798 | - |
| w2290n1976 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2290n1976.laz | yes | 36,788,575 | - |
| w2290n1977 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2290n1977.laz | yes | 23,121,792 | - |
| w2290n1978 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2290n1978.laz | yes | 30,557,488 | - |
| w2291n1972 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2291n1972.laz | yes | 771,520 | - |
| w2291n1973 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2291n1973.laz | yes | 4,615,170 | - |
| w2291n1974 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2291n1974.laz | yes | 18,795,865 | - |
| w2291n1975 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2291n1975.laz | yes | 31,548,166 | - |
| w2291n1976 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2291n1976.laz | yes | 39,583,787 | - |
| w2291n1977 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2291n1977.laz | yes | 41,992,527 | - |
| w2291n1978 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2291n1978.laz | yes | 31,321,359 | - |
| w2292n1974 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2292n1974.laz | yes | 963,828 | - |
| w2292n1975 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2292n1975.laz | yes | 12,478,148 | - |
| w2292n1976 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2292n1976.laz | yes | 26,207,748 | - |
| w2292n1977 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2292n1977.laz | yes | 39,983,911 | - |
| w2292n1978 | USGS_LPC_CA_NoCal_Wildfires_B5b_QL1_2018_w2292n1978.laz | yes | 38,640,163 | - |


### A11. USGS copy against PASDA (section 4.1.2)

{
 "quadrant": "19001270PAS_NE",
 "quadrant_bounds_ftus": [
  1275000.0,
  185000.0,
  1280000.0,
  190000.0
 ],
 "download_wall_seconds": 3.5762786865234375e-06,
 "ept": {
  "url": "https://s3-us-west-2.amazonaws.com/usgs-lidar-public/PA_WesternPA_1_2019/ept.json",
  "bounds_3857": "([-8930038.079359168,-8927902.14544514],[4838981.16559329,4841125.193443755])",
  "exit": 0,
  "seconds": 18.10471200942993,
  "stderr_tail": "",
  "laz_bytes": 47976376,
  "reader_metadata": {
   "srs": {
    "compoundwkt": "PROJCS[\"WGS 84 / Pseudo-Mercator\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Mercator_1SP\"],PARAMETER[\"central_meridian\",0],PARAMETER[\"scale_factor\",1],PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"X\",EAST],AXIS[\"Y\",NORTH],EXTENSION[\"PROJ4\",\"+proj=merc +a=6378137 +b=6378137 +lat_ts=0.0 +lon_0=0.0 +x_0=0.0 +y_0=0 +k=1.0 +units=m +nadgrids=@null +wktext +no_defs\"],AUTHORITY[\"EPSG\",\"3857\"]]",
    "horizontal": "PROJCS[\"WGS 84 / Pseudo-Mercator\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Mercator_1SP\"],PARAMETER[\"central_meridian\",0],PARAMETER[\"scale_factor\",1],PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"X\",EAST],AXIS[\"Y\",NORTH],EXTENSION[\"PROJ4\",\"+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs\"],AUTHORITY[\"EPSG\",\"3857\"]]",
    "isgeocentric": false,
    "isgeographic": false,
    "json": {
     "type": "ProjectedCRS",
     "name": "WGS 84 / Pseudo-Mercator",
     "base_crs": {
      "type": "GeographicCRS",
      "name": "WGS 84",
      "datum": {
       "type": "GeodeticReferenceFrame",
       "name": "World Geodetic System 1984",
       "ellipsoid": {
        "name": "WGS 84",
        "semi_major_axis": 6378137,
        "inverse_flattening": 298.257223563
       }
      },
      "coordinate_system": {
       "subtype": "ellipsoidal",
       "axis": [
        {
         "name": "Latitude",
         "abbreviation": "lat",
         "direction": "north",
         "unit": "degree"
        },
        {
         "name": "Longitude",
         "abbreviation": "lon",
         "direction": "east",
         "unit": "degree"
        }
       ]
      },
      "id": {
       "authority": "EPSG",
       "code": 4326
      }
     },
     "conversion": {
      "name": "unknown",
      "method": {
       "name": "Popular Visualisation Pseudo Mercator",
       "id": {
        "authority": "EPSG",
        "code": 1024
       }
      },
      "parameters": [
       {
        "name": "Latitude of natural origin",
        "value": 0,
        "unit": "degree",
        "id": {
         "authority": "EPSG",
         "code": 8801
        }
       },
       {
        "name": "Longitude of natural origin",
        "value": 0,
        "unit": "degree",
        "id": {
         "authority": "EPSG",
         "code": 8802
        }
       },
       {
        "name": "False easting",
        "value": 0,
        "unit": "metre",
        "id": {
         "authority": "EPSG",
         "code": 8806
        }
       },
       {
        "name": "False northing",
        "value": 0,
        "unit": "metre",
        "id": {
         "authority": "EPSG",
         "code": 8807
        }
       }
      ]
     },
     "coordinate_system": {
      "subtype": "Cartesian",
      "axis": [
       {
        "name": "X",
        "abbreviation": "",
        "direction": "east",
        "unit": "metre"
       },
       {
        "name": "Y",
        "abbreviation": "",
        "direction": "north",
        "unit": "metre"
       }
      ]
     },
     "id": {
      "authority": "EPSG",
      "code": 3857
     }
    },
    "prettycompoundwkt": "PROJCS[\"WGS 84 / Pseudo-Mercator\",\n    GEOGCS[\"WGS 84\",\n        DATUM[\"WGS_1984\",\n            SPHEROID[\"WGS 84\",6378137,298.257223563,\n                AUTHORITY[\"EPSG\",\"7030\"]],\n            AUTHORITY[\"EPSG\",\"6326\"]],\n        PRIMEM[\"Greenwich\",0,\n            AUTHORITY[\"EPSG\",\"8901\"]],\n        UNIT[\"degree\",0.0174532925199433,\n            AUTHORITY[\"EPSG\",\"9122\"]],\n        AUTHORITY[\"EPSG\",\"4326\"]],\n    PROJECTION[\"Mercator_1SP\"],\n    PARAMETER[\"central_meridian\",0],\n    PARAMETER[\"scale_factor\",1],\n    PARAMETER[\"false_easting\",0],\n    PARAMETER[\"false_northing\",0],\n    UNIT[\"metre\",1,\n        AUTHORITY[\"EPSG\",\"9001\"]],\n    AXIS[\"X\",EAST],\n    AXIS[\"Y\",NORTH],\n    EXTENSION[\"PROJ4\",\"+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs\"],\n    AUTHORITY[\"EPSG\",\"3857\"]]",
    "prettywkt": "PROJCS[\"WGS 84 / Pseudo-Mercator\",\n    GEOGCS[\"WGS 84\",\n        DATUM[\"WGS_1984\",\n            SPHEROID[\"WGS 84\",6378137,298.257223563,\n                AUTHORITY[\"EPSG\",\"7030\"]],\n            AUTHORITY[\"EPSG\",\"6326\"]],\n        PRIMEM[\"Greenwich\",0,\n            AUTHORITY[\"EPSG\",\"8901\"]],\n        UNIT[\"degree\",0.0174532925199433,\n            AUTHORITY[\"EPSG\",\"9122\"]],\n        AUTHORITY[\"EPSG\",\"4326\"]],\n    PROJECTION[\"Mercator_1SP\"],\n    PARAMETER[\"central_meridian\",0],\n    PARAMETER[\"scale_factor\",1],\n    PARAMETER[\"false_easting\",0],\n    PARAMETER[\"false_northing\",0],\n    UNIT[\"metre\",1,\n        AUTHORITY[\"EPSG\",\"9001\"]],\n    AXIS[\"X\",EAST],\n    AXIS[\"Y\",NORTH],\n    EXTENSION[\"PROJ4\",\"+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs\"],\n    AUTHORITY[\"EPSG\",\"3857\"]]",
    "proj4": "+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs",
    "units": {
     "horizontal": "metre",
     "vertical": ""
    },
    "vertical": "",
    "wkt": "PROJCS[\"WGS 84 / Pseudo-Mercator\",GEOGCS[\"WGS 84\",DATUM[\"WGS_1984\",SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],AUTHORITY[\"EPSG\",\"6326\"]],PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],UNIT[\"degree\",0.0174532925199433,AUTHORITY[\"EPSG\",\"9122\"]],AUTHORITY[\"EPSG\",\"4326\"]],PROJECTION[\"Mercator_1SP\"],PARAMETER[\"central_meridian\",0],PARAMETER[\"scale_factor\",1],PARAMETER[\"false_easting\",0],PARAMETER[\"false_northing\",0],UNIT[\"metre\",1,AUTHORITY[\"EPSG\",\"9001\"]],AXIS[\"X\",EAST],AXIS[\"Y\",NORTH],EXTENSION[\"PROJ4\",\"+proj=merc +a=6378137 +b=6378137 +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 +k=1 +units=m +nadgrids=@null +wktext +no_defs\"],AUTHORITY[\"EPSG\",\"3857\"]]"
   }
  },
  "points_read_class2_in_quadrant": null,
  "metadata_keys": [
   "filters.crop",
   "filters.range",
   "filters.reprojection",
   "readers.ept",
   "writers.las"
  ]
 },
 "pasda_range_test": {
  "curl": "206 8388608 2478907.000",
  "seconds": 3.409797430038452
 },
 "pasda_class2_in_quadrant": 6877258,
 "usgs_class2_in_quadrant": 0,
 "count_equal": false,
 "ept_class2_in_quadrant": 6877263,
 "ept_count_equal_pasda": false,
 "sha256_sorted_xyz_0.01ft": {
  "ept": "ea0c6f705923d2be85394b05270ce94bfff4b572932ccf465311b9387625220d",
  "pasda": "34c406224e4f865c9c0c99e0dc4a3f4c90a9f28cd1bee32b605f2764148acf3c"
 },
 "sha256_sorted_xyz_0.1ft": null,
 "ept_nn_match": {
  "n_sample": 200000,
  "frac_xy_within_0.01ft": 0.39751,
  "frac_xy_within_0.05ft": 1.0,
  "frac_xy_within_0.1ft": 1.0,
  "frac_xy_within_1ft": 1.0,
  "median_xy_dist_ft": 0.011734564337614838,
  "p95_xy_dist_ft": 0.02122498511761603,
  "median_abs_dz_ft_xy_matched_0.1": 848.8999999999999,
  "frac_z_equal_0.01ft_given_xy_0.1": 0.0,
  "z_ratio_median_ept_over_pasda": 0.3048018572068586
 },
 "z_range_ftus": {
  "pasda": [
   1034.47,
   1463.54
  ],
  "usgs": null
 },
 "seconds_compare": 33.72471809387207
}

| USGS tile | points | version/pf | CRS horizontal | CRS vertical | units | x range | z range |
|---|---|---|---|---|---|---|---|

| file | curl (code bytes B/s) | wall s |
|---|---|---|


### A12. Gridding and routing of the full grids (sections 4.1.5, 4.3.3)

**AP grid** (`grid_AP/AP_grid.json`): {"n_files": 642, "n_files_ok": 642, "lattice_shape": [25157, 35600], "transform": [1.0, 0.0, 549458.0, 0.0, -1.0, 4430800.0], "epsg": 6346, "n_inside_footprint": 578631848, "n_cells_with_class2": 612629774, "n_cells_with_class2_inside": 559542282, "max_fill_distance_m": 10.0, "n_filled_within_max": 17426353, "n_gap_beyond_max_left_nodata": 1663213, "frac_inside_filled": 0.03020329345979093, "frac_inside_gap_nodata": 0.0028743889672661086, "n_filled": 17426353, "n_valid_cells_in_grid": 576968635, "nn_fill_distance_percentiles_50_95_99_max_m": [1.0, 5.0, 8.0, 10.0], "class2_density_inside_pts_per_m2": 2.9966496757661707, "points_per_cell_percentiles_5_50_95": [1, 3, 5], "frac_data_cells_with_ge1_point": 0.9697967065402091, "total_class2": 1924388587, "seconds_total": 576.860009431839, "seconds_fill": 102.48060894012451, "peak_rss_GB": 36.684212}

files: 642, skipped: 0; vertical units seen: ['metre']; horizontal: ['NAD83(2011) / UTM zone 17N + NAVD88 height']

**AP routing** (`grid_AP/AP_route.json`): {"n_valid": 576968635, "n_edge_cells": 270490, "max_area_cells": 157945294, "max_area_km2": 157.945294, "n_roots": 270490, "root_area_sum_cells": 576968635, "root_area_sum_equals_valid": true, "seconds_total": 856.6889793872833, "peak_rss_GB_after_fill": 36.684212, "peak_rss_GB": 46.895672}
largest outlets: [{"row": 15570, "col": 16714, "x": 566172.5, "y": 4415229.5, "area_cells": 157945294, "share_of_valid": 0.27375022560801765}, {"row": 13964, "col": 18386, "x": 567844.5, "y": 4416835.5, "area_cells": 101099149, "share_of_valid": 0.17522468790699516}, {"row": 11969, "col": 27023, "x": 576481.5, "y": 4418830.5, "area_cells": 62977351, "share_of_valid": 0.1091521222813091}, {"row": 14829, "col": 20239, "x": 569697.5, "y": 4415970.5, "area_cells": 23578184, "share_of_valid": 0.04086562521721827}, {"row": 6651, "col": 31395, "x": 580853.5, "y": 4424148.5, "area_cells": 14230536, "share_of_valid": 0.024664314724837686}]

**TTV grid** (`grid_TTV/TTV_grid.json`): {"n_files": 235, "n_files_ok": 235, "lattice_shape": [19010, 19010], "transform": [1.0, 0.0, 529495.0, 0.0, -1.0, 4198505.0], "epsg": 6339, "n_inside_footprint": 184338013, "n_cells_with_class2": 120723190, "n_cells_with_class2_inside": 119291345, "max_fill_distance_m": 10.0, "n_filled_within_max": 21515897, "n_gap_beyond_max_left_nodata": 43530771, "frac_inside_filled": 0.1528039090489394, "frac_inside_gap_nodata": 0.2361464696920651, "n_filled": 21515897, "n_valid_cells_in_grid": 140807242, "nn_fill_distance_percentiles_50_95_99_max_m": [1.0, 5.0, 8.0, 10.0], "class2_density_inside_pts_per_m2": 7.332439016169353, "points_per_cell_percentiles_5_50_95": [1, 8, 20], "frac_data_cells_with_ge1_point": 0.8471960909510606, "total_class2": 1098499075, "seconds_total": 370.2302589416504, "seconds_fill": 39.58337354660034, "peak_rss_GB": 14.641164}

files: 235, skipped: 0; vertical units seen: ['metre']; horizontal: ['NAD83(2011) / UTM zone 10N + NAVD88 height']

**TTV routing** (`grid_TTV/TTV_route.json`): {"n_valid": 140807242, "n_edge_cells": 163708, "max_area_cells": 22644569, "max_area_km2": 22.644569, "n_roots": 163708, "root_area_sum_cells": 140807242, "root_area_sum_equals_valid": true, "seconds_total": 207.25357747077942, "peak_rss_GB_after_fill": 14.641164, "peak_rss_GB": 18.654244}
largest outlets: [{"row": 8209, "col": 7633, "x": 537128.5, "y": 4190295.5, "area_cells": 22644569, "share_of_valid": 0.16081963312654046}, {"row": 11241, "col": 12195, "x": 541690.5, "y": 4187263.5, "area_cells": 9466042, "share_of_valid": 0.06722695413635046}, {"row": 5424, "col": 12995, "x": 542490.5, "y": 4193080.5, "area_cells": 9260152, "share_of_valid": 0.06576474241289379}, {"row": 9905, "col": 10051, "x": 539546.5, "y": 4188599.5, "area_cells": 6263749, "share_of_valid": 0.04448456564471308}, {"row": 4747, "col": 12622, "x": 542117.5, "y": 4193757.5, "area_cells": 5334938, "share_of_valid": 0.03788823589059432}]


### A13. Slurm accounting of the stage A jobs

| JobID | JobName | Partition | State | Elapsed | Timelimit | AllocCPUS | MaxRSS | TotalCPU |
|---|---|---|---|---|---|---|---|---|
| 52730900 | a14_tests | serc | COMPLETED | 00:04:38 | 00:40:00 | 2 |  | 00:00:00 |
| 52732789 | a14_stage1 | serc | COMPLETED | 00:01:56 | 02:00:00 | 1 |  | 00:00:00 |
| 52733671 | a14_pools | serc | COMPLETED | 00:15:57 | 04:00:00 | 1 |  | 00:00:00 |
| 52733794 | a14_usgs_pasda | serc | CANCELLED by 313293 | 00:09:09 | 01:30:00 | 2 |  | 00:00:00 |
| 52735021 | a14_usgs_pasda | serc | CANCELLED by 313293 | 00:00:00 | 03:00:00 | 0 |  | 00:00:00 |
| 52735027 | a14_ap_dl | serc | CANCELLED by 313293 | 00:06:39 | 1-12:00:00 | 1 |  | 00:00:00 |
| 52735399 | a14_usgs_pasda | serc | CANCELLED by 313293 | 00:04:12 | 03:00:00 | 2 |  | 00:00:00 |
| 52736361 | a14_pools2 | serc | COMPLETED | 00:22:49 | 03:00:00 | 1 |  | 00:00:00 |
| 52736600 | a14_usgs_pasda | serc | COMPLETED | 00:50:16 | 03:00:00 | 2 |  | 00:00:00 |
| 52736647 | a14_ap_pasda | serc | CANCELLED by 313293 | 01:12:32 | 1-06:00:00 | 1 |  | 00:00:00 |
| 52737645 | a14_env_numba | serc | COMPLETED | 00:01:16 | 00:30:00 | 2 |  | 00:00:00 |
| 52738203 | a14_grid_AP1 | serc | COMPLETED | 00:01:29 | 01:00:00 | 2 |  | 00:00:00 |
| 52739888 | a14_pools3 | serc | COMPLETED | 00:21:01 | 03:00:00 | 1 |  | 00:00:00 |
| 52739892 | a14_grid_AP1 | serc | COMPLETED | 00:01:24 | 01:00:00 | 2 |  | 00:00:00 |
| 52741260 | a14_ept_test | serc | COMPLETED | 00:00:14 | 01:00:00 | 2 |  | 00:00:00 |
| 52742133 | a14_ept_test | serc | COMPLETED | 00:01:03 | 01:00:00 | 2 |  | 00:00:00 |
| 52742762 | a14_ept_test | serc | COMPLETED | 00:01:07 | 01:00:00 | 2 |  | 00:00:00 |
| 52742941 | a14_ept_AP | serc | COMPLETED | 00:00:12 | 12:00:00 | 8 |  | 00:00:00 |
| 52742942 | a14_grid_AP | serc | COMPLETED | 00:00:19 | 10:00:00 | 4 |  | 00:00:00 |
| 52744111 | a14_ept_AP | serc | COMPLETED | 00:00:11 | 12:00:00 | 8 |  | 00:00:00 |
| 52744112 | a14_grid_AP | serc | COMPLETED | 00:00:15 | 10:00:00 | 4 |  | 00:00:00 |
| 52744114 | a14_ept_TTV | serc | COMPLETED | 00:00:11 | 08:00:00 | 8 |  | 00:00:00 |
| 52744132 | a14_grid_TTV | serc | COMPLETED | 00:00:15 | 06:00:00 | 3 |  | 00:00:00 |
| 52744211 | a14_ept_test | serc | COMPLETED | 00:01:12 | 01:00:00 | 2 |  | 00:00:00 |
| 52745102 | a14_ept_AP | serc | COMPLETED | 00:00:12 | 12:00:00 | 8 |  | 00:00:00 |
| 52745105 | a14_grid_AP | serc | COMPLETED | 00:00:19 | 10:00:00 | 4 |  | 00:00:00 |
| 52745114 | a14_ept_TTV | serc | COMPLETED | 00:00:12 | 08:00:00 | 8 |  | 00:00:00 |
| 52745116 | a14_grid_TTV | serc | COMPLETED | 00:00:16 | 06:00:00 | 3 |  | 00:00:00 |
| 52745222 | a14_usgs_pasda | serc | COMPLETED | 00:01:31 | 02:00:00 | 2 |  | 00:00:00 |
| 52746468 | a14_ept_AP | serc | COMPLETED | 00:16:57 | 12:00:00 | 8 |  | 00:00:00 |
| 52746471 | a14_grid_AP | serc | COMPLETED | 00:28:03 | 10:00:00 | 4 |  | 00:00:00 |
| 52746473 | a14_ept_TTV | serc | COMPLETED | 00:13:49 | 08:00:00 | 8 |  | 00:00:00 |
| 52746475 | a14_grid_TTV | serc | COMPLETED | 00:11:44 | 06:00:00 | 3 |  | 00:00:00 |
| 52746580 | a14_usgs_pasda | serc | COMPLETED | 00:42:30 | 02:00:00 | 2 |  | 00:00:00 |
| 52748077 | a14_copy | serc | COMPLETED | 00:09:58 | 02:00:00 | 2 |  | 00:00:00 |
| 52749307 | a14_grid_TTV | serc | COMPLETED | 00:11:26 | 06:00:00 | 3 |  | 00:00:00 |
| 52750908 | a14_grid_AP | serc | COMPLETED | 00:27:55 | 10:00:00 | 4 |  | 00:00:00 |


---

### A3. Threshold set B and the windows each screen removes (section 3.2), from `scripts/a14_thresholds_B.json`

Thresholds: trunk_A_max_inside_m2_max = 5000000.0; trunk_A_inflow_max_single_m2_max = 2000000.0; transient_chi_r2_area_weighted_min = 0.9; transient_sa_dbic_area_weighted_max = 20.0; disturbance_nlcd_disturbed_frac_max = 0.1; disturbance_nlcd_classes = [11, 22, 23, 24, 31]; disturbance_mining_frac_max = 0.0; disturbance_mining_layers = ['longwall_panels']; upstream_fully_valid_required = {'GM': True, 'TTV': True, 'AP': False}; tolerance_ks_rel = 0.25; tolerance_C_ht_rel = 0.25; tolerance_theta_abs = 0.05

**GM** (71 candidates; passing all screens per offset: {0: 8, 1: 10, 2: 10, 3: 10})

| screen | windows removed (alone) | share | windows kept (alone) |
|---|---|---|---|
| trunk: A_max_inside <= 5e+06 m2 | 23 | 32.4 % | 48 |
| trunk: largest single edge inflow <= 2e+06 m2 | 31 | 43.7 % | 40 |
| transient: chi-z R2 (area-weighted) >= 0.90 | 2 | 2.8 % | 69 |
| transient: slope-area dBIC (area-weighted) <= 20 | 4 | 5.6 % | 67 |
| disturbance: NLCD classes [11, 22, 23, 24, 31] fraction <= 0.10 | 0 | 0.0 % | 71 |
| upstream area fully in valid data (decision 7) | 2 | 2.8 % | 69 |
| **all screens jointly** | 33 | 46.5 % | 38 |

**AP** (1653 candidates; passing all screens per offset: {0: 80, 1: 85, 2: 85, 3: 70})

| screen | windows removed (alone) | share | windows kept (alone) |
|---|---|---|---|
| trunk: A_max_inside <= 5e+06 m2 | 587 | 35.5 % | 1,066 |
| trunk: largest single edge inflow <= 2e+06 m2 | 771 | 46.6 % | 882 |
| transient: chi-z R2 (area-weighted) >= 0.90 | 279 | 16.9 % | 1,374 |
| transient: slope-area dBIC (area-weighted) <= 20 | 343 | 20.8 % | 1,310 |
| disturbance: NLCD classes [11, 22, 23, 24, 31] fraction <= 0.10 | 69 | 4.2 % | 1,584 |
| disturbance (AP): longwall_panels fraction <= 0.00 | 854 | 51.7 % | 799 |
| **all screens jointly** | 1,333 | 80.6 % | 320 |

**TTV** (420 candidates; passing all screens per offset: {0: 23, 1: 27, 2: 22, 3: 30})

| screen | windows removed (alone) | share | windows kept (alone) |
|---|---|---|---|
| trunk: A_max_inside <= 5e+06 m2 | 62 | 14.8 % | 358 |
| trunk: largest single edge inflow <= 2e+06 m2 | 90 | 21.4 % | 330 |
| transient: chi-z R2 (area-weighted) >= 0.90 | 71 | 16.9 % | 349 |
| transient: slope-area dBIC (area-weighted) <= 20 | 149 | 35.5 % | 271 |
| disturbance: NLCD classes [11, 22, 23, 24, 31] fraction <= 0.10 | 167 | 39.8 % | 253 |
| upstream area fully in valid data (decision 7) | 48 | 11.4 % | 372 |
| **all screens jointly** | 318 | 75.7 % | 102 |


### A4. Selection metrics against the reference (section 3.3): distributions, tolerance counts alone and jointly with the set B screens

**GM** reference: theta 0.292, k_s 1.900, |C_ht| 0.013 (legacy TAYLOR_REF (updated_best_fit_tile_finder.ipynb cell 1))

| metric | n | min | p05 | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|
| theta by chi (legacy method) | 71 | 0.196 | 0.246 | 0.310 | 0.388 | 0.438 | 0.496 | 0.562 |
| k_s at theta_ref, median of basins | 71 | 1.213 | 1.823 | 2.141 | 2.382 | 2.675 | 3.384 | 4.555 |
| k_s at theta_ref, pooled | 71 | 1.494 | 1.746 | 2.076 | 2.230 | 2.520 | 2.889 | 3.463 |
| |C_ht| (3x3 Laplacian, convex first-order divides) | 71 | 8.628e-03 | 9.252e-03 | 0.012 | 0.014 | 0.015 | 0.017 | 0.018 |
| basins in the chi fit | 71 | 9.000 | 11.000 | 19.000 | 23.000 | 29.500 | 44.000 | 49.000 |
| convex divide pixels | 71 | 608.0 | 646.0 | 708.0 | 750.0 | 828.5 | 957.0 | 984.0 |

| tolerance | windows within (alone) | within and passing all screens |
|---|---|---|
| theta within +-0.05 | 20 | 7 |
| k_s within +-25 % | 34 | 20 |
| |C_ht| within +-25 % | 56 | 28 |
| k_s and |C_ht| (theta free) | 23 | 14 |
| **all three** | 9 | 4 |

Eligible (all screens and all tolerances) per offset: {0: 1, 1: 0, 2: 2, 3: 1}; total candidates 71.

**AP** reference: theta 0.540, k_s 47.200, |C_ht| 5.930e-03 (legacy TAYLOR_REF (updated_best_fit_tile_finder.ipynb cell 1))

| metric | n | min | p05 | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|
| theta by chi (legacy method) | 1,653 | 0.100 | 0.128 | 0.479 | 0.592 | 0.704 | 0.800 | 0.800 |
| k_s at theta_ref, median of basins | 1,653 | -40.026 | 19.845 | 31.984 | 38.166 | 45.554 | 58.874 | 146.3 |
| k_s at theta_ref, pooled | 1,653 | -7.000 | 21.957 | 32.004 | 38.072 | 44.658 | 57.504 | 83.625 |
| |C_ht| (3x3 Laplacian, convex first-order divides) | 1,653 | 1.697e-03 | 3.174e-03 | 3.985e-03 | 5.014e-03 | 5.927e-03 | 6.906e-03 | 8.650e-03 |
| basins in the chi fit | 1,653 | 3.000 | 3.000 | 7.000 | 10.000 | 15.000 | 26.000 | 48.000 |
| convex divide pixels | 1,653 | 31.000 | 62.600 | 85.000 | 100.0 | 115.0 | 136.0 | 165.0 |

| tolerance | windows within (alone) | within and passing all screens |
|---|---|---|
| theta within +-0.05 | 384 | 69 |
| k_s within +-25 % | 955 | 162 |
| |C_ht| within +-25 % | 990 | 130 |
| k_s and |C_ht| (theta free) | 651 | 68 |
| **all three** | 181 | 13 |

Eligible (all screens and all tolerances) per offset: {0: 4, 1: 2, 2: 4, 3: 3}; total candidates 1653.

**TTV** reference: theta 0.387, k_s 15.416, |C_ht| 0.015 (M&D proxy catchment on the 10 m NED, same code)

| metric | n | min | p05 | p25 | median | p75 | p95 | max |
|---|---|---|---|---|---|---|---|---|
| theta by chi (legacy method) | 420 | 0.100 | 0.100 | 0.221 | 0.340 | 0.448 | 0.592 | 0.800 |
| k_s at theta_ref, median of basins | 420 | 1.753e-03 | 1.000 | 6.933 | 12.711 | 16.639 | 23.735 | 40.382 |
| k_s at theta_ref, pooled | 420 | 0.504 | 1.628 | 6.813 | 12.917 | 18.366 | 28.634 | 41.010 |
| |C_ht| (3x3 Laplacian, convex first-order divides) | 420 | 3.250e-03 | 6.455e-03 | 0.015 | 0.020 | 0.024 | 0.030 | 0.035 |
| basins in the chi fit | 420 | 6.000 | 16.000 | 29.000 | 38.000 | 49.250 | 68.000 | 105.0 |
| convex divide pixels | 420 | 632.0 | 764.8 | 990.0 | 1150.0 | 1253.5 | 1449.3 | 1935.0 |

| tolerance | windows within (alone) | within and passing all screens |
|---|---|---|
| theta within +-0.05 | 93 | 15 |
| k_s within +-25 % | 190 | 63 |
| |C_ht| within +-25 % | 122 | 34 |
| k_s and |C_ht| (theta free) | 46 | 20 |
| **all three** | 13 | 3 |

Eligible (all screens and all tolerances) per offset: {0: 1, 1: 0, 2: 2, 3: 0}; total candidates 420.


### A4b. Sensitivity of the eligible count to the reference and the tolerances (after all set B screens)

| site | reference | theta / k_s / |C_ht| | theta tolerance | k_s and C_ht tolerance | within tolerance (alone) | eligible (with screens) | per offset |
|---|---|---|---|---|---|---|---|
| GM | legacy / M&D proxy | 0.292 / 1.90 / 0.0128 | +-0.05 | +-25 % | 9 | 4 | {0: 1, 1: 0, 2: 2, 3: 1} |
| GM | legacy / M&D proxy | 0.292 / 1.90 / 0.0128 | +-0.05 | +-50 % | 16 | 6 | {0: 1, 1: 0, 2: 3, 3: 2} |
| GM | legacy / M&D proxy | 0.292 / 1.90 / 0.0128 | +-0.10 | +-25 % | 15 | 7 | {0: 1, 1: 1, 2: 2, 3: 3} |
| GM | legacy / M&D proxy | 0.292 / 1.90 / 0.0128 | +-0.10 | +-50 % | 33 | 16 | {0: 4, 1: 3, 2: 4, 3: 5} |
| GM | legacy / M&D proxy | 0.292 / 1.90 / 0.0128 | none | +-25 % | 23 | 14 | {0: 3, 1: 3, 2: 3, 3: 5} |
| GM | legacy / M&D proxy | 0.292 / 1.90 / 0.0128 | none | +-50 % | 58 | 33 | {0: 7, 1: 9, 2: 7, 3: 10} |
| GM | same code, whole coarse grid | 0.354 / 1.66 / 0.0151 | +-0.05 | +-25 % | 2 | 1 | {0: 0, 1: 0, 2: 1, 3: 0} |
| GM | same code, whole coarse grid | 0.354 / 1.66 / 0.0151 | +-0.05 | +-50 % | 15 | 12 | {0: 3, 1: 3, 2: 2, 3: 4} |
| GM | same code, whole coarse grid | 0.354 / 1.66 / 0.0151 | +-0.10 | +-25 % | 3 | 1 | {0: 0, 1: 0, 2: 1, 3: 0} |
| GM | same code, whole coarse grid | 0.354 / 1.66 / 0.0151 | +-0.10 | +-50 % | 30 | 19 | {0: 4, 1: 5, 2: 3, 3: 7} |
| GM | same code, whole coarse grid | 0.354 / 1.66 / 0.0151 | none | +-25 % | 5 | 2 | {0: 0, 1: 0, 2: 2, 3: 0} |
| GM | same code, whole coarse grid | 0.354 / 1.66 / 0.0151 | none | +-50 % | 42 | 26 | {0: 6, 1: 7, 2: 5, 3: 8} |
| AP | legacy / M&D proxy | 0.540 / 47.20 / 0.0059 | +-0.05 | +-25 % | 181 | 13 | {0: 4, 1: 2, 2: 4, 3: 3} |
| AP | legacy / M&D proxy | 0.540 / 47.20 / 0.0059 | +-0.05 | +-50 % | 348 | 58 | {0: 16, 1: 11, 2: 16, 3: 15} |
| AP | legacy / M&D proxy | 0.540 / 47.20 / 0.0059 | +-0.10 | +-25 % | 328 | 29 | {0: 10, 1: 6, 2: 7, 3: 6} |
| AP | legacy / M&D proxy | 0.540 / 47.20 / 0.0059 | +-0.10 | +-50 % | 655 | 110 | {0: 31, 1: 27, 2: 29, 3: 23} |
| AP | legacy / M&D proxy | 0.540 / 47.20 / 0.0059 | none | +-25 % | 651 | 68 | {0: 17, 1: 20, 2: 16, 3: 15} |
| AP | legacy / M&D proxy | 0.540 / 47.20 / 0.0059 | none | +-50 % | 1,460 | 276 | {0: 72, 1: 73, 2: 72, 3: 59} |
| AP | same code, whole coarse grid | 0.571 / 39.50 / 0.0058 | +-0.05 | +-25 % | 228 | 24 | {0: 6, 1: 6, 2: 9, 3: 3} |
| AP | same code, whole coarse grid | 0.571 / 39.50 / 0.0058 | +-0.05 | +-50 % | 378 | 61 | {0: 15, 1: 12, 2: 18, 3: 16} |
| AP | same code, whole coarse grid | 0.571 / 39.50 / 0.0058 | +-0.10 | +-25 % | 391 | 38 | {0: 11, 1: 10, 2: 12, 3: 5} |
| AP | same code, whole coarse grid | 0.571 / 39.50 / 0.0058 | +-0.10 | +-50 % | 688 | 116 | {0: 30, 1: 28, 2: 33, 3: 25} |
| AP | same code, whole coarse grid | 0.571 / 39.50 / 0.0058 | none | +-25 % | 745 | 91 | {0: 20, 1: 29, 2: 23, 3: 19} |
| AP | same code, whole coarse grid | 0.571 / 39.50 / 0.0058 | none | +-50 % | 1,463 | 294 | {0: 75, 1: 78, 2: 76, 3: 65} |
| TTV | legacy / M&D proxy | 0.387 / 15.42 / 0.0151 | +-0.05 | +-25 % | 13 | 3 | {0: 1, 1: 0, 2: 2, 3: 0} |
| TTV | legacy / M&D proxy | 0.387 / 15.42 / 0.0151 | +-0.05 | +-50 % | 48 | 12 | {0: 6, 1: 3, 2: 3, 3: 0} |
| TTV | legacy / M&D proxy | 0.387 / 15.42 / 0.0151 | +-0.10 | +-25 % | 24 | 6 | {0: 1, 1: 0, 2: 4, 3: 1} |
| TTV | legacy / M&D proxy | 0.387 / 15.42 / 0.0151 | +-0.10 | +-50 % | 96 | 22 | {0: 7, 1: 6, 2: 7, 3: 2} |
| TTV | legacy / M&D proxy | 0.387 / 15.42 / 0.0151 | none | +-25 % | 46 | 20 | {0: 2, 1: 7, 2: 7, 3: 4} |
| TTV | legacy / M&D proxy | 0.387 / 15.42 / 0.0151 | none | +-50 % | 179 | 52 | {0: 11, 1: 15, 2: 13, 3: 13} |


