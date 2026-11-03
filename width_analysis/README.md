# width_analysis: channel width on the 1 m lidar tiles

Measurement-only package. It reads the repository's existing per-tile rasters
(`{AP,GM}_1m_best_tile_{filled,area,fd}`) and writes everything under
`outputs/widths/`. No model code, result, or figure is touched.

Run everything with one command:

```bash
python -m width_analysis.run --site all --mode pilot   # 10 points per site, then stop
python -m width_analysis.run --site all --mode full    # stratified sample, fits, figures, regime outputs
python -m width_analysis.acrit_profile --site AP --tile 5  # MSE vs A_crit at optimized K, D
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

**Width measures (all recorded, never conflated).**

- *Inner-channel (bank-break) width* `W_inner_bankbreak`: z(s) is smoothed
  with a Gaussian of `bankbreak_smooth_sigma_m` (0.5 m); convexity is -d2z/ds2
  by central differences. On each side of the thalweg the first local maximum
  of convexity that exceeds `bankbreak_min_convexity_1_per_m` (0.05 m^-1) and
  whose mean slope from the thalweg exceeds `bankbreak_min_bank_slope` (0.10)
  is the bank top; the width is the distance between the two picks, NaN if
  either side has none (flag `no_bank_break`). `inner_relief_m` is the height
  of the lower bank top above the thalweg.
- *Valley-floor width* `W_valley_floor`: the contiguous span around the
  thalweg where the cross-section slope |dz/ds| is below
  `valley_slope_threshold` (0.05), on the unsmoothed bilinear profile.
- *Fixed-stage widths* `W_stage_0.25`, `W_stage_0.5`, `W_stage_1.0`: the
  contiguous span with z - z_thalweg <= h, linearly interpolated crossings; a
  span stops at a crest when the profile beyond it drops by more than
  `crest_drop_m` (0.3 m) so it never runs into the neighbouring channel. These
  are diagnostics only. On a V-shaped section the stage width is 2h divided
  by the side slope: it measures the valley side slope, not a channel width,
  and it grows without limit as the section flattens. It is therefore never
  used as the channel width. `W_stage_0.5` is kept as the secondary
  diagnostic and `W_stage_0.25` is compared against the bank-break width in
  figure 7.

**Primary width and censoring.** `W_primary` is the bank-break width where
both banks exist (`W_primary_source = bankbreak`), the valley-floor width
where `shape_class = flat_bed` (`valley_floor`), and otherwise unresolved
(`unresolved`, `W_primary = NaN`). A grid of spacing delta cannot resolve a
channel narrower than about 2 delta, so `censored = True` when the primary
width is unresolved or below `censor_multiple` (2) x pixel size, and
`censor_bound` = 2 delta (1.95 m on AP, 2.00 m on GM). Every measured value is
kept; each measure also carries its own `<name>_censored` column.

**Channelized flag.** `channelized = True` when both bank breaks exist and
`inner_relief_m` >= `channelized_min_relief_m` (0.3 m). The W = cA^b fits use
channelized points only; hollows and swales enter the shape-class and
channelized-fraction figures instead.

**Model-free statistic.** `resolved_width = True` when the point has a
resolved inner channel (both bank breaks, `W_inner_bankbreak` > 2 delta) or a
flat bed wider than 2 delta (`shape_class = flat_bed` and `W_valley_floor` >
2 delta). Figure 2 reports its fraction per A bin and site.

**Valley-bottom shape class.** `flat_bed` if the samples within
`flat_bed_tol_m` (0.1 m) of the thalweg elevation form a contiguous span of at
least `flat_bed_min_width_m` (2 m); otherwise `v_shape`. The span length is
stored as `flat_span_m`.

**Steep rills.** Points with local slope above `steep_slope_flag` (0.5) carry
`steep_rill_flag` (the AP tile 5 parallel rills). They stay in the sample;
every fit is reported with (`all`) and without (`no_steep`) them.

**Fits (`fits.json`).** Per site and variant, log10 W = a + b log10 A is fit
to the channelized points by censored maximum likelihood (Tobit: normal
residuals, points with `W_inner_bankbreak` below 2 delta contribute the
probability mass below the bound; `scipy.optimize.minimize`, Nelder-Mead).
A 1000-sample pairs bootstrap gives percentile intervals on c = 10^a and b.
An OLS fit on uncensored points only is reported alongside and labelled
biased. The channelized fraction is fit by logistic regression on log10 A;
`A50` is the area where the fitted probability crosses 0.5, with a bootstrap
interval.

**Regime outputs.** `regime_fraction_vs_dx.csv`: for delta = 1 m and each of
the 25 sweep spacings, the fraction of channel length (pixels with A >= the
tile's A_crit, pooled per site, each weighted by its D8 step length) whose
fitted W(A) exceeds delta, with a bootstrap band. `regime_boundary.csv`:
c, b and A* = (delta / c)^(1/b) per site, variant and delta, with bootstrap
intervals. Figure 3 maps W(A)/delta at delta = pixel size on every tile.

**A_crit misfit profile.** `python -m width_analysis.acrit_profile --site AP
--tile 5` evaluates the repo's partitioned model (src/models.py) at the
jointly optimized K and D of that tile for 31 log-spaced A_crit between 1e2
and 1e5 m² plus the notebook optimum, and writes MSE(A_crit) to
`outputs/widths/acrit_profile/`.

**QC gallery.** One PNG per point: the profile with the thalweg, the stage
spans, the bank-break picks with the inner-channel span, the valley-floor span
and the censoring statement, plus a 120 m hillshade inset with the channel
pixels, the fitted trace and the transect. The full run writes every point to
`qc_all/` and a random `qc_gallery_per_site` (20) per site to `qc/`.

## Outputs

`points.csv` (one row per point; primary width and source, censoring,
inner-channel, valley-floor and stage widths, channelized and resolved flags,
shape class, then span end-points, slopes, pixel size, flow azimuth),
`excluded.csv`, `transects.csv` (every sampled s, z, x, y), `points.gpkg`
(one layer per site in its UTM CRS), `validation_points.gpkg` and `.csv`
(channelized points with A >= 1e5 m², priority 1, plus 10 random others per
site, priority 2, with stage and bank-break widths as attributes),
`fits.json`, `regime_fraction_vs_dx.csv`, `regime_boundary.csv`,
`figures/fig1..fig7`, `qc/`, `qc_all/`, `bins.json`, `run_info.json`.

## Tests

`tests/test_width_analysis.py` builds synthetic cross-sections on the 0.25 m
transect grid: a V-shape (widths scale with stage, flagged as flat W/h, no bank
break), a 4 m flat bed with 1 m banks (bankfull at h = 1.0, W = 6 m, bank-break
6 m, valley floor 4 m, class flat_bed, not censored), a trapezoid, a 1 m bed
that must come back censored with the value retained, the censor bound at the
AP pixel size, the continuous flow-line direction on a staircase trace, and the
flow graph on a synthetic tile.
