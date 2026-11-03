# Channel-width extraction: Phase 2 pilot report

Date: 2026-11-02
Branch: `feature/channel-width` (commit 5ec898f and earlier)
Directive: `channel_width_directive.md` (Phases 0-2)
Outputs: `outputs/widths/pilot/` (`points.csv`, `excluded.csv`, `transects.csv`,
`points.gpkg`, `bins.json`, `run_info.json`, `qc/*.png`)
Code: `width_analysis/` (method and thresholds documented in `width_analysis/README.md`)
Tests: `tests/test_width_analysis.py`, 11 synthetic tests, all passing

## 1. Status

Phases 0 and 1 are complete. Phase 2 (pilot, 10 points per site) has been run
twice: the first run exposed four method problems (Section 3), the second run
is the one reported here. Phase 3 (full run, fits, figures) has not started
and waits on the decisions in Section 6.

Decisions already taken by Steve on 2026-11-02:

- the filled grids are sufficient; no unfilled DEM or point cloud for now, so
  the fill-depth exclusion is replaced by a proxy (Section 3);
- the sampling floor is lowered from 10^3.5 m² to 10^3 m² because the
  smallest optimized A_crit is 1.9 x 10^3 m² (AP tile 5);
- the conda environment goes under `$GROUP_HOME/scdobbs-envs/`.

Environment: the repo's `environment.yml` is a macOS conda export and does not
solve on Linux. Both pilot runs used the group's existing
`/home/groups/hilley/envs/jasy-lidar/bin/python` (3.10; numpy 2.2.6, scipy
1.15.2, rasterio 1.4.3, geopandas 1.1.4). A dedicated environment
(`$GROUP_HOME/scdobbs-envs/lem-widths`, Python 3.11 with GDAL so that
TopoAnalysis imports) was being built by micromamba in Slurm job 50834283 at
the time of writing; the classic conda solver was killed for memory at 16 GB.
`width_analysis/requirements-linux.txt` lists the dependencies.

## 2. Data facts that affect the method

| Site | Grid | Pixel (m) | CRS | Censor bound 2δ (m) |
|---|---|---|---|---|
| AP tiles 1-5 | 1015 x 1015 (tile 3: 1014 x 1015) | 0.9752 | NAD83 / UTM 17N | 1.95 |
| GM tiles 1-5 | 1000 x 1000 | 1.000 | WGS84 / UTM 10N | 2.00 |

- Only filled DEMs exist (`_filled`); no flat cells and no uphill D8 steps in
  any tile, so the fill used an epsilon gradient. Filled reaches appear as
  near-zero along-path gradients.
- Drainage areas were computed on a larger DEM before clipping: maximum A
  exceeds the 1 km² tile area in AP tiles 3 and 4 (1.1e7, 1.7e7 m²) and GM
  tiles 2, 4, 5 (8.9e6, 6.2e6, 4.7e6 m²).
- Per-tile A_crit (Table 1, identical to the final stage of the
  `*_complete.ipynb` notebooks): AP 6.4e3, 7.6e3, 9.7e3, 7.8e3, 1.9e3 m²;
  GM 8.1e3, 3.0e3, 6.5e3, 7.4e3, 2.2e3 m².
- Resolution sweep: 25 log-spaced Δx from 5 to 100 m, A_min = 1e5 m²
  (`supp_fig6.ipynb`, `resolution_study_dinf.ipynb`).

## 3. Method changes made after the first pilot

The first pilot (set aside, untracked, in `outputs/widths/pilot_v1_superseded/`)
returned 19 points and 582 logged exclusions, and its QC gallery showed four
problems. Each has a fix in the committed code and a synthetic test.

| Problem seen in pilot 1 | Fix | Config key |
|---|---|---|
| Every 10^3 m² swale joining a trunk counted as a confluence; 319 of 546 candidates in the largest AP bin were rejected and the bin came up empty | A confluence requires the second-largest donor to carry A >= max(floor, 0.05 x A of the receiver), i.e. a jump in A along the path | `confluence_min_fraction` |
| The 10 m vertical-interval slope is undefined on low-gradient trunks inside a 1 km tile (100 rejections) | Flagged fallback: slope over the main-stem path +/- 50 m; `slope_method` records which was used | `horizontal_interval_fallback_m` |
| Straight diagonal D8 flow lines across planar hillslopes (the parallel-flow artifact) reach 10^3 m² and were sampled as channels (AP tile 3: "slope" 1.59, profile monotonic) | Confinement test: the profile must rise >= 0.1 m on both sides of the thalweg within 5 m | `confinement_half_window_m`, `confinement_min_relief_m` |
| A point on an epsilon-filled flat (GM tile 3: 0.05 m of relief over 5 x 5 pixels) was sampled | Filled-depression proxy: reject if the fall along the +/-10 m trace is below 0.002 | `flat_fill_slope_max` |
| Height-above-thalweg spans ran over a low crest into the neighbouring parallel channel (AP tile 5) | A span stops at a crest when the profile then drops by more than 0.3 m; flagged `..._cut_at_crest` | `crest_drop_m` |

Other method notes:

- Local slope re-implements the vertical-interval construction of the repo's
  `ChannelSlopeWithSmoothing` (TopoAnalysis needs GDAL, which the pilot
  environment lacks). The TopoAnalysis version sums one extra pixel length into
  the denominator; this version uses the exact path length.
- Bins are defined per site (8 log-spaced bins from 10^3 m² to the site's
  largest A), not per tile, so that bins are comparable across tiles.
- The thalweg is the middle sample of the lowest run (ties on flat beds).
- The valley-floor slope is taken on the unsmoothed bilinear profile; the
  0.5 m Gaussian bled bank slope into a 4 m flat bed in the synthetic test.

## 4. Pilot sample

20 points, 10 per site. Exclusions logged while drawing:

| Site | confluence | no_incision | flat_fill_proxy | tile_edge | too_close_to_sample | knickpoint |
|---|---|---|---|---|---|---|
| AP | 3 | 3 | 2 | 1 | 1 | 0 |
| GM | 4 | 3 | 2 | 2 | 0 | 1 |

No bin was exhausted. Slope method: 17 vertical-interval, 3 horizontal
fallback (AP3_009, GM3_007, GM5_009, the three largest trunks).

Widths in metres. "none" = no convex bank break satisfying the thresholds on
one or both sides. HAND = height-above-thalweg width at the fixed stage.

| id | A (m²) | S | HAND 0.25 | HAND 0.5 | HAND 1.0 | bankfull (h) | bank-break | valley floor | shape | flags |
|---|---|---|---|---|---|---|---|---|---|---|
| AP5_001 | 1.3e3 | 0.66 | 2.31 | 4.91 | 6.75 | 8.64 (2.00) | 12.00 | 0.25 | v | crest cut from h=0.55 |
| AP5_000 | 2.9e3 | 0.87 | 2.49 | 5.22 | 6.00 | 6.00 (2.00) | 5.75 | 0.25 | v | crest cut from h=0.55 |
| AP2_003 | 3.4e3 | 0.91 | 1.75 | 2.98 | 7.14 | 8.16 (1.45) | none | 0.25 | v | crest cut from h=0.35; 0.25 m stage censored |
| AP2_002 | 8.5e3 | 0.24 | 15.09 | 25.53 | 36.75 | 34.43 (0.75) | none | 15.00 | flat | hits transect end at h=0.80 |
| AP1_004 | 1.4e4 | 0.60 | 3.31 | 4.12 | 4.73 | 6.00 (1.60) | none | 0.25 | v | crest cut from h=0.30 |
| AP5_005 | 4.5e4 | 0.27 | 0.58 | 1.19 | 2.46 | 4.33 (2.00) | 6.50 | 0.25 | v | 0.25 m stage censored |
| AP4_006 | 1.6e5 | 0.08 | 1.17 | 1.89 | 3.32 | 6.22 (2.00) | 7.25 | 0.25 | v | 0.25 m stage censored |
| AP2_007 | 1.2e6 | 0.06 | 2.03 | 2.78 | 4.46 | 3.97 (0.90) | 5.00 | 0.75 | v | |
| AP3_008 | 1.8e6 | 0.04 | 2.84 | 4.71 | 8.21 | 9.50 (2.00) | 9.50 | 0.25 | v | crest cut from h=1.00 |
| AP3_009 | 9.1e6 | 0.02 | 5.41 | 5.69 | 6.16 | 6.89 (2.00) | 13.25 | 2.25 | flat | crest cut from h=0.25; horizontal slope |
| GM2_000 | 1.7e3 | 0.10 | 9.53 | 12.41 | 16.00 | 21.02 (2.00) | 32.25 | 3.75 | flat | |
| GM3_001 | 2.1e3 | 0.46 | 8.54 | 14.26 | 23.85 | 29.66 (2.00) | none | 1.50 | flat | crest cut from h=0.95 |
| GM4_002 | 5.6e3 | 0.05 | 4.51 | 5.21 | 6.64 | 10.30 (2.00) | none | 0.75 | flat | crest cut from h=0.20 |
| GM5_003 | 8.5e3 | 0.18 | 7.74 | 10.09 | 14.19 | 19.11 (2.00) | 30.50 | 3.50 | flat | |
| GM5_004 | 2.1e4 | 0.32 | 9.16 | 13.25 | 21.35 | 32.85 (1.60) | none | 0.25 | flat | hits transect end at h=1.75 |
| GM1_005 | 7.9e4 | 0.07 | 5.12 | 7.23 | 10.45 | 15.02 (2.00) | none | 1.25 | flat | |
| GM4_006 | 1.1e5 | 0.07 | 2.84 | 4.77 | 10.07 | 6.42 (0.70) | 7.00 | 0.25 | v | crest cut from h=0.90 |
| GM3_007 | 3.3e5 | 0.04 | 14.06 | 17.14 | 23.31 | 32.43 (2.00) | none | 3.75 | flat | horizontal slope |
| GM1_008 | 1.1e6 | 0.04 | 2.05 | 3.10 | 4.65 | 6.63 (1.55) | 7.00 | 0.25 | v | |
| GM5_009 | 2.9e6 | 0.03 | 2.20 | 3.34 | 5.70 | 5.39 (0.95) | 6.00 | 1.00 | v | horizontal slope |

Tallies (denominator 20 unless stated):

- bankfull HAND width censored (below 2δ): 0; bank-break width censored: 0 of 12;
  0.25 m stage censored: 3 (AP2_003, AP5_005, AP4_006).
- fraction with W > 1 m: 19/20 at the 0.25 m stage, 20/20 at the 0.5 m stage,
  12/12 bank-break.
- bankfull stage pinned at the top of the range (h = 2.0 m): 12.
- no bank break on at least one side: 8.
- shape class: AP 8 v_shape, 2 flat_bed; GM 3 v_shape, 7 flat_bed.

## 5. Reading of the QC gallery

**Measurements that look right.** Every point with A above about 5e4 m² on AP
and 1e5 m² on GM shows an inner channel 1 to 2 m deep with convex bank breaks
bracketing it, 5 to 13 m wide (AP5_005, AP4_006, AP2_007, AP3_008, AP3_009,
GM4_006, GM1_008, GM5_009). The bank-break and HAND picks agree with the
profiles by eye. These channels are several pixels wide at 1 m, which is the
regime-III condition.

**Method problems.**

1. *Bankfull pick.* The minimum of W/h pins at h = 2.0 m in 12 of 20 points.
   On a parabolic or U-shaped hollow W grows roughly as h^0.5, so W/h
   decreases monotonically and the minimum is always at the top of the range.
   The criterion only identifies a stage where a distinct bank exists. The
   d2W/dh2 alternative is recorded (`h_bankfull_alt_d2W`) but is noisy on
   0.25 m samples.
2. *Gabilan points below about 1e5 m² are hollows, not channels.* GM2_000,
   GM3_001, GM5_003, GM5_004, GM1_005 and GM3_007 are parabolic or
   flat-floored swales 10 to 30 m wide with no inner channel that the lidar
   resolves; GM4_002 has a 0.5 m notch on a bench beside a deeper gully. Their
   widths are valley-bottom widths. A single W = cA^b fit through these and the
   inner-channel widths above 1e5 m² would be fitting two different objects.
3. *AP tile 5 small-A points are parallel rills on 40-degree slopes* (AP5_000,
   AP5_001; S = 0.66 to 0.87, spacing about 10 m). They pass every test and
   are confined, but they are not the channels the model partition describes.
   AP tile 5 has the smallest A_crit (1.9e3 m²), which places the partition's
   channel head on these rills.
4. *D8 lines off the real thalweg.* GM4_002 shows D8 routing flow along a
   bench parallel to a deeper gully 5 m away. The crest cut kept the span
   from running into the gully, but the sampled "channel" is the bench.
5. *Broad swales on AP at 8.5e3 m²* (AP2_002): an asymmetric 15 m hollow with
   no channel. Same issue as item 2, less frequent on AP.

**Not seen in the pilot.** No riparian-vegetation artifacts (the DEMs are
bare-earth), no flow-direction errors beyond item 4, no censored bank-break
widths.

## 6. Decisions needed before Phase 3

1. Primary width for the fit: (a) bank-break width where present, fixed-stage
   HAND width (0.5 m) otherwise, HAND bankfull secondary; or (b) keep HAND
   bankfull primary and raise `h_max_m` above 2 m.
2. Fit W(A) over the whole range, or only above a minimum A where an inner
   channel exists (about 5e4 m² on AP, 1e5 m² on GM from this pilot), with the
   small-A hollows reported separately through the shape-class figure.
3. Whether the steep AP tile 5 rills (S > 0.5) stay in the sample, or a slope
   cap excludes them.
4. Whether the 0.5 m stage (not 0.25 m) should be the model-free "W > 1 m"
   measure, since 0.25 m is within the vertical noise of the gridded lidar.

## 7. Reproducing the pilot

```bash
cd ~/LEMs_competition_vs_threshold-JGR-branch
PY=/home/groups/hilley/envs/jasy-lidar/bin/python     # or scdobbs-envs/lem-widths/bin/python once built
$PY -m pytest tests/test_width_analysis.py -q
$PY -m width_analysis.run --site all --mode pilot --out outputs/widths/pilot
```

The job script used is `$SCRATCH/width_jobs/pilot.sbatch` (normal partition,
40 min, 2 CPUs, 8 GB; the run itself takes about 12 s plus tile loading).
