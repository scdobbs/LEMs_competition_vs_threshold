# acrit_slope: does the optimized A_crit follow A ∝ S⁻² across tiles?

Measurement-only package for Test 1 of the resubmission
(`reports/directives/acrit_slope_test1_directive.md`). It reads the repo's
per-tile rasters (`{AP,GM}_1m_best_tile_{filled,area,fd}`), reuses the D8 flow
graph of `width_analysis`, and writes everything to `outputs/acrit_slope/`.
No model code, optimization result or figure is modified; no optimization is
re-run.

```bash
python -m acrit_slope.run                 # everything: heads, summary, fits, figures, QC
python -m pytest tests/test_acrit_slope.py  # synthetic-surface tests
```

All thresholds and choices are in `acrit_slope/config.py` (`CONFIG`); the
per-tile A_crit comes from `width_analysis.config.A_CRIT_M2` (Table 1, final
notebook stage), the per-tile θ (`THETA`) from the `concavity = ...` line of
`notebooks/model_runs/*_complete.ipynb`.

## Method

**Inputs.** The 1 m filled DEM, D8 flow directions and D8 drainage area of
each tile, exactly the grids the partitioned model was optimized on. Areas were
accumulated on the larger DEM before clipping, so A can exceed the tile area.

**Partition channel head.** A pixel with A ≥ A_crit whose D8 donors all have
A < A_crit (equivalently: largest donor area < A_crit). This is the upstream
end of each channel segment of the partition.

**Exclusions** (tested in this order; the first that fires is the logged
`reason`, every rule is also recorded as a boolean column):

| reason | rule |
|---|---|
| `tile_edge` | closer than `edge_buffer_m` (30 m) to the grid edge |
| `path_leaves_tile` | the upslope valley-axis path (below) is shorter than max(`L_values_m`) = 20 m, i.e. it reaches the edge or a cell with no donor before the longest measurement window |
| `flat_fill` | the fall over the ±10 m main-stem trace through the head, divided by its length, is below `flat_fill_slope_max` (0.002): the width config's proxy for an epsilon-filled depression (no unfilled DEM is available) |

A fourth boolean, `flag_upslope_flat` (`S_up_10` < `flat_fill_slope_max`), is
recorded but is not an exclusion: it marks heads at the downstream lip of a
filled flat, which the ±10 m trace proxy does not catch because the trace
includes the steep drop below the head. `n_kept_upslope_flat` in the summary
counts them (0 to 2 per tile).

**Upslope valley-axis path.** From the head, step repeatedly to the donor with
the largest drainage area (`FlowGraph.up`); downslope, step to the D8 receiver.
Cumulative distance uses the exact step length, √2 δ on diagonals.

**Slopes at a head** (δ is the pixel size, 0.975 m on AP, 1.000 m on GM):

- `S_up_L` = (z at exactly L m up the path − z at the head) / L, for
  L = 5, 10, 20 m; z at distance L is interpolated linearly between the two
  path pixels bracketing it. **`S_up_10` is used in every fit.**
- `S_centred_L` = (z at L/2 upslope − z at L/2 downslope) / L.
- `S_pixel` = D8 drop from the head to its receiver over the step length
  (diagnostic only; dominated by gridding noise).
- `AS2` = A × `S_up_10`².

**Plan curvature (QC only).** The DEM is block-mean subsampled to about
`curvature_block_m` (10 m: 10 pixels, so 9.75 m on AP, 10 m on GM) and the
Zevenbergen & Thorne (1987) 9-point plan curvature is evaluated on that grid.
Sign convention: positive = convergent (trough), negative = divergent (nose),
checked on z = a x² + b y in the tests. `convergent_fraction` is the share of
kept heads whose 10 m block has positive plan curvature.

**Per-tile summary** (`tile_summary.csv`): A_crit, θ, counts detected / kept /
excluded by reason, median and quartiles of `S_up_10`, medians of `S_up_5`,
`S_up_20`, `S_centred_10`, `S_pixel`, median A (and its ratio to A_crit), median
AS2, convergent fraction.

**Across-tile fits** (`fits.json`): y = log₁₀ A_crit on x = log₁₀(median
`S_up_10`); the exponent b means A_crit ∝ S^b and Montgomery & Dietrich's
A S² = const is b = −2.

| variant | fit |
|---|---|
| `v1_all_OLS`, `v1_all_RMA` | ten tiles, one line, ordinary least squares and reduced major axis (b = sign(r)·sd(y)/sd(x)) |
| `v2_common_slope` | ten tiles, one slope, separate AP and GM intercepts (least squares with site dummies) |
| `v3_{AP,GM}_{OLS,RMA}` | each site alone, n = 5: reported, labelled low-power |
| `v4_noAP5_*` | variants 1 and 2 on the nine tiles without AP tile 5 |
| `sensitivity` | variant 1 with `S_up_5`, `S_up_20`, `S_centred_10` |

**Uncertainty.** Two-level bootstrap, `bootstrap_n` (2000) iterations, seed
`random_seed`: resample the tiles of the variant with replacement; inside each
resampled tile resample its kept heads with replacement and take the median
slope; refit. 95 % percentile intervals; `includes_minus2` records whether
−2 lies inside. Iterations with zero spread in x, or (common slope) with a
single site, are skipped and counted.

**AP tile 5.** Against each nine-tile fit: residual of log₁₀ A_crit
(and as a factor and in units of the fit's residual SD), and whether AP5's
A_crit lies inside the 95 % prediction interval. Two prediction intervals are
given: the classical OLS interval (t, n − 2 d.f.), and a bootstrap interval
built from the bootstrap (a, b) pairs plus a residual drawn from the nine-tile
fit plus a resampled AP5 median slope (available for OLS, RMA and the
common-slope model). The A_crit error bar on AP5 is the range of A_crit whose
misfit lies within `misfit_tolerance` (10 %) of the minimum of
`outputs/widths/acrit_profile/AP5.csv`, interpolated log-linearly between the
sampled values.

## Supplementary analysis (beyond the directive)

`fits.json["supplementary"]`, `pseudo_head_summary.csv`, `pseudo_heads.csv`,
figure 6. These exist because of a validity problem with the across-tile test
that the directive does not address:

*If the tiles shared one slope–area curve S(A) in the hollows, the head slope
at A = A_crit would be S(A_crit) on every tile, and any variation of A_crit,
whatever its cause, would produce log A_crit = const − (1/θ_h) log S
exactly.* The across-tile test can then return an inverse relation with an
exponent near −1/θ_h even when A_crit has nothing to do with slope. Three
additions address this:

1. **Pseudo-heads.** Heads are also detected at fixed thresholds
   `pseudo_head_thresholds_m2` (1e3 … 2e4 m²) on every tile, with the same
   exclusions and slopes. The per-tile median `S_up_10` against threshold is
   the within-tile S(A) of head slopes; its pooled log-log slope (tile
   intercepts) is θ_h, and −1/θ_h is the exponent a common curve alone would
   produce (`within_tile_slope_area`).
2. **Slope at a fixed reference area.** log A_crit against log median
   `S_up_10` of the heads at `reference_area_m2` (5e3 m²) on every tile. This
   slope cannot depend on A_crit mechanically, so a relation here is evidence
   that steeper hollows go with smaller A_crit (`reference_area_fit`).
3. **Stage-2 A_crit.** The same variant-1 fit with the A_crit obtained when K
   and D were held at the per-tile measured values (notebook stage 2,
   `A_CRIT_STAGE2_M2`), which differ from the jointly optimized Table 1 values
   by factors up to 30 (`stage2_fit`). This shows how much the conclusion
   depends on which A_crit one takes as the tile's value.

## Outputs

`heads.csv` (every detected head at A_crit with slopes, QC fields, flags and
reason), `excluded_heads.csv` (its excluded rows), `pseudo_heads.csv`,
`pseudo_head_summary.csv`, `tile_summary.csv`, `fits.json`, `run_info.json`,
`figures/fig1_acrit_vs_slope.png`, `fig2_head_slopes_by_tile.png`,
`fig3_sensitivity.png`, `fig6_supp_slope_area_structure.png`,
`figures/qc_maps/<site><tile>.png` (heads on the 1 m hillshade, kept and
excluded by reason), `figures/qc_profiles/<site><tile>.png` (10 random kept
heads per tile: upslope path profile with the 5, 10, 20 m chords and the
centred 10 m chord).

## Tests

`tests/test_acrit_slope.py`: a plane dipping south (every `S_up_L`,
`S_centred_L` and `S_pixel` equals the plane slope), a plane dipping along the
diagonal (path steps are √2 δ and the slopes still equal the plane slope), a
short upslope path (NaN beyond its length), a 7 × 7 synthetic network with a
known threshold (detected heads are exactly the expected cells for two
thresholds), the exclusion rules and their precedence, the filled-flat proxy
on a near-flat band, the plan-curvature sign convention, the block mean, and
the fits and two-level bootstrap on data with a known exponent of −2.
