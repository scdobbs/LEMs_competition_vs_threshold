# head_extraction: channel heads without an area threshold (Test 2)

Directive: `reports/directives/head_extraction_test2_directive.md`.
Measurement-only package; reads the repo's per-tile rasters
(`{AP,GM}_1m_best_tile_{filled,area,fd}`), reuses `acrit_slope` and
`width_analysis` helpers, writes to `outputs/head_extraction/`. No model
code, result or figure is modified; no optimization is re-run.

```bash
python -m head_extraction.run --phase extract            # sections 2-4; Method P, Method D if LSDTT is installed
python -m head_extraction.run --phase extract --no-lsdtt # Method P only
python -m head_extraction.run --phase analyse            # sections 5-8, after heads_review.csv (implemented after review)
python -m pytest tests/test_head_extraction.py
```

Every threshold is in `head_extraction/config.py`: `EXTRACTION` (Method P and
the drops; contains no area value), `LSDTT` (Method D), `QC` (classes),
`ANALYSIS` (fits, references). The A_crit bracket per tile is
`ACRIT_BRACKET = (stage 2, final)` from `acrit_slope.config`.

## Inputs

- The grids the notebooks load, from `data/{site}/tile{n}/`.
- **No unfilled DEM exists** for any tile on Sherlock (searched the repo
  `data/` tree and the user's group directories `scdobbs_hilley_lem`,
  `scdobbs_optimize`, `scdobbs_minimize_unit_stream_power`; only `*_filled`
  grids and model outputs). Method P therefore runs on the filled DEM: the
  Wiener filter is applied to the filled surface and heads at the downstream
  lips of filled flats are possible; the `bench_plateau` class and the QC
  profiles are the check.
- **No roads layer exists** in the repo or the user's group storage. The
  `road_ditch` class therefore uses the geometric rule only (collinear
  heads across the fall line). Nothing was downloaded.

## Method P (Pelletier 2013, steps 1-3, Python; `pelletier.py`)

1. **Optimal Wiener filter.** The tile is reflect-padded by `wiener_pad_px`
   (128) pixels, a plane is removed, and the radially averaged power
   spectrum S(ν) is computed in `wiener_n_radial_bins` (200) bins to the
   Nyquist frequency. |N|² is the mean of S over `wiener_noise_band`
   (0.75–1.0 of Nyquist). |H|² = a ν^β is a log-log least-squares fit to the
   bins where S > `wiener_signal_min_ratio` (3) × |N|², skipping the
   `wiener_skip_lowest_bins` (2) lowest bins. The transfer function
   |H|²/(|H|² + |N|²) (Pelletier eq. 4) is applied radially in 2-D frequency
   space; plane and padding are restored. The fitted β, |N|² and the RMS
   change are recorded per tile. No user parameter sets the amount of
   smoothing.
2. **Contour curvature** (Pelletier eq. 1, Mitasova & Hofierka 1993) by
   central differences on the filtered DEM; positive = convergent.
3. **Heads.** The filtered DEM is conditioned by priority-flood filling with
   epsilon gradients (richdem; Pelletier used his own fill-pits-and-flats).
   Pixels are visited from highest to lowest; a pixel is *downstream* if any
   strictly higher 8-neighbour is a head or downstream (every descending
   pathway, not only steepest descent); a pixel is a head if κ > κt and it
   is not downstream. This is Pelletier's paragraph-17 definition. The
   directive's wording ("most upslope pixel of each connected valley
   segment") is computed too (`component_top_heads`) and its counts and
   agreement with the Pelletier heads are in `run_info.json`.

κt = `curvature_threshold_default` (0.1 m⁻¹); runs at 0.05 and 0.2 m⁻¹ give
the threshold sensitivity (`threshold_sensitivity.csv`).

**No drainage area enters steps 1-3.** `tests/test_head_extraction.py::
test_heads_bit_identical_when_area_config_changes` changes every area value
in `QC`, `LSDTT` and the A_crit table and asserts identical heads.

## Method D (DrEICH, LSDTopoTools 0.9 from conda-forge; `lsdtt.py`)

`lsdtt-channel-extraction` with `print_dreich_channels: true`, `m_over_n` =
the tile's θ, `A_0` = 1, `connected_components_threshold` = 10 (Clubb et al.
2014: 10 linked pixels), `curvature_threshold` = 0.1 m⁻¹ for valley
identification, `surface_fitting_radius` = 6 m, `raster_is_filled: true`.
The tile is exported as ENVI `.bil` in the tile's UTM CRS; LSDTT's source
CSV (WGS84) is converted back with pyproj.

**Status (2026-11-04): Method D did not run.** `lsdtt-channel-extraction`
0.9 (conda-forge build h33989ec_0) aborts inside
`LSDJunctionNetwork::create` with `malloc(): memory corruption` (exit 134,
or signal 11 under the Python wrapper) immediately after it reports the
number of source nodes, on AP1 and GM1 alike. Reproduced with
`raster_is_filled` true and false, `threshold_contributing_pixels` 1000 and
4000, `number_of_junctions_dreich` 1 and 2, `connected_components_threshold`
10 and 100, `A_0` 1 and 1000, and with a two-pixel NoData border added to
the raster. The Wiener option of the same binary runs in 3 s and writes
sources, so the installation itself works. About two hours of the four-hour
cap were used; the remaining options (build from source, the LSDTT
container) are not parameter fixes and were not attempted. Logs are in
`$SCRATCH/head_jobs/lsdtt_debug*/`.

**Caveat the directive did not anticipate.** LSDTT's DrEICH builds an
initial test network with `threshold_contributing_pixels` (default 1000,
i.e. 1000 m² at 1 m) and prunes with `pruning_drainage_area` (1000 m²);
LSDTT's Pelletier/Wiener options use `minimum_drainage_area` (400 m²). So
Method D as implemented in LSDTT is not area-free upstream of its chi
segment fit. It is run as the directive asks, at
`threshold_contributing_pixels` = 1000 and at 250 and 4000 as a
sensitivity, so the dependence of the head positions on that parameter is
measured rather than assumed. LSDTT's own Wiener heads (`print_wiener_channels`)
are kept as a third set, `method = W`, for comparison only.

## Drops and measurements (`classify.py`)

Every head pixel, by method: `tile_edge` (< 30 m from the edge),
`path_leaves_tile` (largest-donor upslope path shorter than 20 m). For each
head: A on the model's D8 area grid, `S_up_5/10/20`, `S_centred_*`,
`S_pixel`, `S_down_10` (slope over the 10 m receiver path), flow azimuth
(direction of the 10 m receiver path), plan curvature on the 10 m block-mean
grid, distance to the nearest trunk pixel (A ≥ 10⁵ m²).

## QC classes (directive section 4, tested in this order)

| class | rule (values in `QC`) |
|---|---|
| `valley_floor` | within 30 m of a trunk pixel and S_up_10 < 0.10 |
| `bench_plateau` | S_up_20 < 0.05, or S_up_10 < 0.05 with S_down_10 > 0.15 (flat above, convex drop below) |
| `rill` | S_up_10 > 0.58 and ≥ 3 other heads within 30 m whose flow azimuths lie within 20° of its own |
| `road_ditch` | inside a 10 m road buffer (no layer exists), else ≥ 4 other heads within 60 m, PCA line RMS residual < 3 m, line more than 30° from the head's fall line |
| `hollow` | none of the above and plan curvature convergent on the 10 m grid |
| `unclassified` | the rest |

Method agreement: nearest-neighbour matching within 20 m, both directions,
median separation of matched pairs.

## Outputs (extract phase)

`heads_extracted.csv` (every head of every method and setting),
`dropped_heads.csv`, `class_summary.csv`, `threshold_sensitivity.csv`,
`heads_review_template.csv` (retained Method P heads, `keep = True`,
blank `reason`: copy to `heads_review.csv` and edit), `run_info.json`
(Wiener fits, method agreement, LSDTT logs), `figures/qc_maps/<tile>.png`
(log A overlay, classes by colour, methods by marker, Test 1 partition heads
as grey dots), `figures/qc_profiles/<tile>.png` (15 random hollow-class
Method P heads), `figures/fig_threshold_sensitivity.png`.

## Tests

A parabolic hollow on a plane whose curvature decays to the top (the heads
must lie within 2 m of the analytic κ = κt crossing, both definitions); a
known surface plus white noise (the Wiener filter must lower the RMS error);
bit-identical heads when every area value in the config changes; the rill
rule on eight parallel gullies on a 35° plane and not on a single head; the
road rule on a row of heads across the fall line and not on a line down it.
