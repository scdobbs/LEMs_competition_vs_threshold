# width_analysis: channel width on the 1 m lidar tiles

Measurement-only package. It reads the repository's existing per-tile rasters
(`{AP,GM}_1m_best_tile_{filled,area,fd}`) and writes everything under
`outputs/widths/`. No model code, result, or figure is touched.

Run everything with one command:

```bash
python -m width_analysis.run --site all --mode pilot   # 10 points per site, then stop
python -m width_analysis.run --site all --mode full    # stratified sample, fits, figures
python -m pytest tests/test_width_analysis.py           # synthetic cross-section tests
```

All thresholds live in `width_analysis/config.py` (`CONFIG`); the per-tile
`A_crit` of Table 1 and the 5-100 m sweep definition are in the same file.

## Data

Both sites are read from the repo's filled DEMs. AP tiles are 1015 x 1015
pixels of 0.975 m (NAD83 / UTM 17N); GM tiles are 1000 x 1000 pixels of
1.000 m (WGS84 / UTM 10N). The DEMs are depression-filled with an epsilon
gradient (no flat cells, no uphill D8 steps), and no unfilled DEM or point
cloud is available on Sherlock, so the directive's fill-depth exclusion cannot
be applied. Reaches that were filled therefore appear in the sample with an
artificially flat or uniformly sloping bed; the QC gallery is the only check on
that. Drainage areas are inherited from the larger DEM the tiles were clipped
from, so several tiles contain channels with A larger than the tile area.

## Method

**Channel network and sampling.** Channel pixels are those with D8 drainage
area A >= `a_min_sample_m2` (10^3 m², lowered from the directive's 10^3.5 m²
because the smallest optimized A_crit is 1.9 x 10^3 m²). Sample points are
stratified in `n_area_bins` (8) log-spaced bins between that floor and the
largest A found in any tile of the site, so bins are shared across the site's
tiles. Within each bin candidates are visited in random order
(`numpy.random.default_rng([random_seed, site_index])`, seed 20261102) until the
bin's quota is met. Each tested candidate is kept or logged with one reason, in
this order:

| reason | rule |
|---|---|
| `tile_edge` | closer than `edge_exclusion_m` (30 m) to the grid edge |
| `confluence` | a confluence lies within `confluence_exclusion_m` (15 m) along the main stem upstream or downstream. A confluence is a pixel whose second-largest donor is a channel with A >= max(floor, `confluence_min_fraction` (0.05) x A of the receiver), i.e. a jump in A along the path; with an absolute floor alone every 10^3 m² swale joining a trunk counted and emptied the trunk bin |
| `too_close_to_sample` | closer than `min_separation_m` (50 m) to an accepted point on the same tile |
| `slope_undefined` / `reach_slope_undefined` | neither the vertical-interval path nor the horizontal fallback can be built (at the outlet or a divide) |
| `knickpoint` | local slope > `knickpoint_factor` (3) x the median local slope of channel pixels within `reach_half_length_m` (100 m) along the path |
| `flat_fill_proxy` | the fall along the +/-10 m channel trace is below `flat_fill_slope_max` (0.002): the proxy for a filled depression, since no unfilled DEM is available |
| `no_incision` | the cross-section does not rise by `confinement_min_relief_m` (0.1 m) on both sides of the thalweg within `confinement_half_window_m` (5 m): straight D8 flow lines across planar hillslopes (the parallel-flow artifact) and thalwegs on monotonic slopes are not channels |
| `bin_exhausted` | the bin ran out of candidates before its quota (a bookkeeping row, not a point) |

The fill-depth exclusion of the directive is not applied (see Data).

**Local slope.** Re-implements the vertical-interval construction of the repo's
k_s routine (`TopoAnalysis.ChannelSlopeWithSmoothing`, `vertical_interval` =
10 m): from the pixel, extend one step up the largest-area donor and one step
down the receiver alternately until the end-to-end drop reaches 10 m; slope is
that drop over the geometric path length. The TopoAnalysis implementation sums
one extra pixel length into the denominator; this version uses the exact path
length, a difference of one pixel in several tens. Where the 10 m drop cannot
be reached inside the tile (low-gradient trunks), the slope over the main-stem
path +/- `horizontal_interval_fallback_m` (50 m) is used instead and
`slope_method` records `horizontal_fallback`.

**Cross-sections.** The local flow direction is the first principal axis of the
main-stem trace within `trace_half_length_m` (10 m) up- and downstream of the
pixel, which removes the 45-degree D8 quantization. The transect runs
perpendicular to it, `transect_half_length_m` (25 m) each way, sampled every
`transect_step_m` (0.25 m) by bilinear interpolation of the filled DEM.
Positive distance is the left bank looking downstream. The thalweg is the lowest
sample within `thalweg_search_m` (3 m) of the centreline; where several samples
tie, the middle one is used. `confined`, `relief_left_m` and `relief_right_m`
record the confinement test on every accepted point as well.

**Width measures (all recorded).**

- *Height-above-thalweg width W(h)*: for h from 0.05 to 2.0 m in 0.05 m steps,
  the width of the contiguous span around the thalweg with z - z_thalweg <= h,
  with linearly interpolated crossings. A span also stops at a crest when the
  profile beyond it drops by more than `crest_drop_m` (0.3 m), so that a width
  never runs across a divide into the neighbouring channel (flag
  `..._cut_at_crest`). Widths are also stored at fixed stages
  h = 0.25, 0.5, 1.0 m. The bankfull stage is the h that minimises W/h
  (`bankfull_method = "min_W_over_h"`), which is the stage just before the banks
  are overtopped; the stage maximising d2W/dh2 is stored alongside
  (`h_bankfull_alt_d2W`). Stages whose span reaches the transect end are
  ineligible. If W/h varies by less than 10 % over the stage range (a pure
  V-shape) the pick is arbitrary and the point carries the flag `W_over_h_flat`.
- *Bank-break width*: z(s) is smoothed with a Gaussian of
  `bankbreak_smooth_sigma_m` (0.5 m); convexity is -d2z/ds2 by central
  differences. On each side of the thalweg the first local maximum of
  convexity that exceeds `bankbreak_min_convexity_1_per_m` (0.05 m^-1) and whose
  mean slope from the thalweg exceeds `bankbreak_min_bank_slope` (0.10) is the
  bank top; the width is the distance between the two picks, NaN if either side
  has none (flag `no_bank_break`).
- *Valley-floor width*: the contiguous span around the thalweg where the
  cross-section slope |dz/ds| is below `valley_slope_threshold` (0.05), measured
  on the unsmoothed bilinear profile.

**Resolution limit and censoring.** A grid of spacing delta cannot resolve a
channel narrower than about 2 delta. Any width below `censor_multiple` (2) x
pixel size is an upper bound: `censored = True` and `censor_bound` = 2 delta
(1.95 m on AP, 2.00 m on GM). The measured value is kept in every case.
`censored` refers to the bankfull HAND width; each other measure carries its
own `<name>_censored` column.

**Valley-bottom shape class.** `flat_bed` if the samples within
`flat_bed_tol_m` (0.1 m) of the thalweg elevation form a contiguous span of at
least `flat_bed_min_width_m` (2 m); otherwise `v_shape`. The span length is
stored as `flat_span_m`.

**QC gallery.** One PNG per point in `outputs/widths/<mode>/qc/`: the profile
with the thalweg, the fixed-stage spans, the bankfull span, the bank-break
picks, the valley-floor span and the censoring statement, plus a 120 m hillshade
inset with the channel pixels, the fitted trace and the transect.

## Outputs

`points.csv` (one row per point; the directive's columns first, then the
auxiliary ones: span end-points, alternative picks, per-measure censoring,
`flat_span_m`, reach-median slope, pixel size, flow azimuth), `excluded.csv`,
`transects.csv` (every sampled s, z, x, y for re-plotting or manual checking),
`points.gpkg` (one layer per site, in that site's UTM CRS), `bins.json`,
`run_info.json` (git hash, interpreter, elapsed time, the full config).

## Tests

`tests/test_width_analysis.py` builds synthetic cross-sections on the 0.25 m
transect grid: a V-shape (widths scale with stage, flagged as flat W/h, no bank
break), a 4 m flat bed with 1 m banks (bankfull at h = 1.0, W = 6 m, bank-break
6 m, valley floor 4 m, class flat_bed, not censored), a trapezoid, a 1 m bed
that must come back censored with the value retained, the censor bound at the
AP pixel size, the continuous flow-line direction on a staircase trace, and the
flow graph on a synthetic tile.
