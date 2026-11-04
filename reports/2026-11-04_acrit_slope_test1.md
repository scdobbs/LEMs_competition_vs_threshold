# Test 1: does the optimized A_crit follow the Montgomery & Dietrich inverse area–slope relation?

Date: 2026-11-04. Branch `feature/acrit-slope` (commit 3e61011), package
`acrit_slope/`, outputs `outputs/acrit_slope/`, Slurm job 51084181 (74 s).
Directive: `reports/directives/acrit_slope_test1_directive.md`. Nothing pushed.

## Summary

Across the ten tiles there is no inverse relation between the optimized
A_crit and the slope at the partition channel heads. The ten-tile OLS
exponent is 0.04 with a 95 % interval of [−0.65, 0.73] and r² = 0.00; the
reduced-major-axis fit is meaningless at r² = 0 (its sign flips across the
bootstrap). The only fit whose interval contains −2 is the common-slope model
with separate site intercepts, and that interval, [−2.6, 0.7], also contains
zero: the two sites separate cleanly in head slope (AP 0.58–0.79, GM
0.18–0.35) but not in A_crit (both 2e3–1e4 m²), so the "common slope" is a
line through two clusters, not a relation within either. AP tile 5 lies off
the nine-tile relation (A_crit a factor of 4 below prediction, 2.7–2.9
residual SDs, outside every 95 % prediction interval), but its head slopes
(median 0.73) are indistinguishable from those of AP tiles 1–4
(0.58–0.79), so slope does not explain why its A_crit is low. The evidence
supports the second outcome of the directive: the A_crit values do not follow
A ∝ S⁻², and AP tile 5 is off the trend for a reason other than slope.

Two findings beyond the directive matter for how the manuscript can use this
test. First, head slope within a tile barely changes with threshold area over
1e3–2e4 m² (pooled log-log slope 0.12), so the across-tile comparison is not
an artefact of a shared slope–area curve; the null result stands. Second,
A_crit is not a stable tile property: the stage-2 values (A_crit optimized
with K and D at their measured values) differ from the Table 1 values by
factors of 1.7 to 30 and give a *positive* exponent (OLS 0.91 [0.01, 3.11]).
Any statement about what A_crit tracks has to say which A_crit.

## 1. Input verification

**A_crit.** The ten values in the directive table are the final (stage 4,
joint K–D–A_crit) optima printed in `notebooks/model_runs/*_complete.ipynb`,
to four significant figures (`width_analysis.config.A_CRIT_M2`). No
discrepancy. The same notebooks print two earlier A_crit per tile; the
stage-2 value (A_crit alone, K and D fixed at the per-tile measured values)
is used in the supplementary analysis:

| tile | θ | A_crit stage 2 (m²) | A_crit final, Table 1 (m²) | ratio |
|---|---|---|---|---|
| AP1 | 0.562 | 3063 | 6414 | 2.1 |
| AP2 | 0.592 | 2338 | 7646 | 3.3 |
| AP3 | 0.542 | 1722 | 9742 | 5.7 |
| AP4 | 0.438 | 995 | 7798 | 7.8 |
| AP5 | 0.400 | 1130 | 1887 | 1.7 |
| GM1 | 0.379 | 423 | 8092 | 19 |
| GM2 | 0.308 | 100 (lower bound of the search) | 2960 | 30 |
| GM3 | 0.417 | 1297 | 6526 | 5.0 |
| GM4 | 0.313 | 422 | 7396 | 18 |
| GM5 | 0.300 | 261 | 2247 | 8.6 |

**θ.** The `concavity = ...` line of each notebook, used as m in every
optimization stage (column above). The manuscript's stated ranges
(0.40–0.59 AP, 0.30–0.42 GM) agree.

**Grids.** The 1 m filled DEM, D8 flow direction and D8 area of each tile
(`data/{site}/tile{n}/`), the grids the notebooks load. Area is inherited from
the larger DEM: the median A at kept heads is 1.02–1.10 × A_crit on every
tile, as expected for heads one pixel past the threshold.

**One claim in the directive did not check out.** Pixel slope at the heads
is not "about 0.88 on every AP tile": the median D8 pixel slope at kept AP
heads is 0.37–0.62 (GM 0.16–0.28). The 0.88 figure may refer to another
sample; it is not needed, because the fits use the path slopes.

## 2. Heads

A head is a pixel with A ≥ A_crit whose D8 donors all have A < A_crit.
Exclusions in order: `tile_edge` (< 30 m from the edge), `path_leaves_tile`
(upslope largest-donor path shorter than 20 m), `flat_fill` (fall over the
±10 m trace below 0.002, the width config's filled-flat proxy).

| tile | A_crit (m²) | detected | tile_edge | path_leaves_tile | flat_fill | kept |
|---|---|---|---|---|---|---|
| AP1 | 6414 | 41 | 10 | 1 | 0 | 30 |
| AP2 | 7646 | 56 | 16 | 0 | 0 | 40 |
| AP3 | 9742 | 34 | 8 | 0 | 0 | 26 |
| AP4 | 7798 | 41 | 6 | 0 | 0 | 35 |
| AP5 | 1887 | 190 | 30 | 0 | 1 | 159 |
| GM1 | 8092 | 18 | 5 | 0 | 0 | 13 |
| GM2 | 2960 | 62 | 16 | 0 | 0 | 46 |
| GM3 | 6526 | 42 | 7 | 0 | 1 | 34 |
| GM4 | 7396 | 38 | 14 | 0 | 1 | 23 |
| GM5 | 2247 | 106 | 34 | 0 | 0 | 72 |

Total 628 detected, 478 kept. GM1 has only 13 kept heads; its median slope
is the least certain. `excluded_heads.csv` lists every excluded head with its
reason.

The QC profiles (`figures/qc_profiles/`) show one failure mode the directive's
proxy does not catch: heads at the downstream lip of a filled flat (flat for
20 m upslope, a step of 1–2 m immediately downslope). The ±10 m trace
includes the step, so its slope is not small. Counted as
`n_kept_upslope_flat` (`S_up_10` < 0.002): AP1 1, AP2 2, AP3 1, AP4 1, all
others 0. They are kept, as the directive specifies; the medians are not
affected.

## 3. Per-tile summary (`tile_summary.csv`)

Slopes are m/m. `S_up_10` is the fit variable: elevation gain over exactly
10 m up the largest-area-donor path, divided by 10 m.

| tile | A_crit | θ | kept | S_up_10 median | q25 | q75 | S_up_5 | S_up_20 | S_centred_10 | S_pixel | A median (m²) | AS² median (m²) | convergent |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | 6414 | 0.562 | 30 | 0.623 | 0.161 | 0.943 | 0.592 | 0.681 | 0.586 | 0.366 | 6977 | 3303 | 0.83 |
| AP2 | 7646 | 0.592 | 40 | 0.646 | 0.266 | 0.910 | 0.598 | 0.725 | 0.615 | 0.429 | 8338 | 3555 | 0.85 |
| AP3 | 9742 | 0.542 | 26 | 0.583 | 0.101 | 0.880 | 0.643 | 0.653 | 0.557 | 0.402 | 10361 | 3638 | 0.85 |
| AP4 | 7798 | 0.438 | 35 | 0.785 | 0.508 | 0.957 | 0.787 | 0.818 | 0.721 | 0.617 | 7988 | 4888 | 0.91 |
| AP5 | 1887 | 0.400 | 159 | 0.729 | 0.579 | 0.932 | 0.736 | 0.806 | 0.715 | 0.615 | 2077 | 1260 | 0.76 |
| GM1 | 8092 | 0.379 | 13 | 0.201 | 0.165 | 0.246 | 0.193 | 0.202 | 0.178 | 0.212 | 8326 | 335 | 1.00 |
| GM2 | 2960 | 0.308 | 46 | 0.348 | 0.116 | 0.393 | 0.338 | 0.360 | 0.313 | 0.276 | 3253 | 393 | 0.96 |
| GM3 | 6526 | 0.417 | 34 | 0.176 | 0.139 | 0.235 | 0.185 | 0.186 | 0.177 | 0.156 | 7056 | 227 | 0.97 |
| GM4 | 7396 | 0.313 | 23 | 0.306 | 0.270 | 0.386 | 0.314 | 0.310 | 0.299 | 0.263 | 7642 | 792 | 0.96 |
| GM5 | 2247 | 0.300 | 72 | 0.232 | 0.072 | 0.408 | 0.225 | 0.248 | 0.200 | 0.158 | 2456 | 145 | 0.85 |

Observations:

- The sites separate in head slope by a factor of about 3 (AP medians
  0.58–0.79, GM 0.18–0.35) but overlap completely in A_crit.
- AP slope distributions are wide and bimodal (figure 2): AP1–AP3 have a
  quarter to a third of their heads below 0.2, on benches, road cuts and
  terrace treads (visible in `qc_maps/AP1.png`, `AP5.png`), and the rest on
  valley walls at 0.6–1.0. AP5's heads are mostly on the steep walls and
  along the roads that cross the tile. GM heads sit at hollow heads on broad
  noses and the distributions are narrow (IQR 0.08–0.12 on GM1, GM3, GM4).
- Convergent-curvature fractions are 0.76–1.00: the heads are in hollows or
  valley axes on the 10 m grid, not on noses. AP5 (0.76) and GM5 (0.85) are
  the lowest; both have the smallest A_crit and therefore heads farthest up
  the hollows.
- AS² at the AP heads (median 3300–4900 m²) is at the top of the Tennessee
  Valley envelope of Montgomery & Dietrich (1992), A S² = 500 to 4000 m², and
  at the GM heads (145–790 m²) at or below its bottom. The humid site needs
  the larger AS², the semi-arid site the smaller, which is the reverse of the
  precipitation ordering Montgomery & Dietrich (1988) report (drier regions,
  larger source areas at the same slope).

## 4. Across-tile fits (`fits.json`)

y = log₁₀ A_crit on x = log₁₀(median `S_up_10`); exponent b means
A_crit ∝ S^b; 2000 two-level bootstrap iterations (tiles, then heads within
tiles); 95 % percentile intervals.

| variant | tiles | method | b | 95 % interval | −2 inside | r² |
|---|---|---|---|---|---|---|
| 1 | all ten | OLS | 0.04 | [−0.65, 0.73] | no | 0.00 |
| 1 | all ten | RMA | 1.04 | [−1.42, 1.50] | no | 0.00 |
| 2 | all ten | common slope, site intercepts | −0.97 | [−2.58, 0.71] | yes (and 0 inside) | 0.15 |
| 3 | AP only (n = 5) | OLS | −2.35 | [−7.68, 2.46] | yes | 0.19 |
| 3 | AP only (n = 5) | RMA | −5.41 | [−10.4, 11.8] | yes | 0.19 |
| 3 | GM only (n = 5) | OLS | −0.73 | [−2.70, 1.82] | yes | 0.13 |
| 3 | GM only (n = 5) | RMA | −2.04 | [−4.62, 3.61] | yes | 0.13 |
| 4 | nine without AP5 | OLS | 0.32 | [−0.11, 1.05] | no | 0.13 |
| 4 | nine without AP5 | RMA | 0.88 | [−0.98, 1.49] | no | 0.13 |
| 4 | nine without AP5 | common slope, site intercepts | −0.68 | [−1.74, 0.95] | no | 0.35 |

Variant 2 intercepts: AP 3.60 [3.19, 3.98], GM 3.09 [1.98, 4.14]; the AP − GM
difference is 0.51 log₁₀ units (factor 3.3) with interval [−0.29, 1.28].
Read literally, AP would need a three times larger A_crit than GM at the same
head slope, but the interval includes zero and the slopes of the two sites do
not overlap, so the intercept difference and the common slope trade off
freely. The per-site fits (variant 3) are reported because the directive
asks for them; with five points and r² below 0.2 their intervals span
−7.7 to +2.5 (AP) and −2.7 to +1.8 (GM) and resolve nothing. RMA with r² near
zero returns ±sd(y)/sd(x) with a sign that flips between bootstrap draws;
the RMA intervals should be read as "undetermined", not as evidence of a
positive slope.

**AP tile 5 against the nine-tile fits.** Median slope 0.729, A_crit 1887 m².

| fit | predicted A_crit (m²) | residual (log₁₀) | factor | residual / σ | 95 % PI (m²) | inside |
|---|---|---|---|---|---|---|
| variant 4 OLS | 7352 | −0.59 | 0.26 | −2.75 | classical 1956–27635; bootstrap 2790–13890 | no (both) |
| variant 4 RMA | 10632 | −0.75 | 0.18 | −2.88 | bootstrap 2090–30784 | no |
| variant 4 common slope (AP intercept) | 7271 | −0.59 | 0.26 | −2.92 | bootstrap 3149–14493 | no |

AP tile 5 is off the trend of the other nine tiles by every measure. The
classical OLS interval is the widest (t with 7 d.f.) and AP5 still falls
just below it. Note what "the trend" is here: a near-flat line with
r² = 0.13 through nine tiles. AP5's A_crit is low relative to tiles whose
head slopes are the same as its own.

**A_crit range for AP tile 5.** Misfit within 10 % of the profile minimum
(MSE 289.6 at 1887 m², threshold 318.6): the sampled values inside are 1887
and 1995 m²; log-linear interpolation gives 1658–2051 m². That is the
vertical bar on AP5 in figure 1, and it is invisible at the scale of the
scatter: at fixed K and D the profile is sharp. The range does not cover the
joint K–D–A_crit trade-off, which the stage-2 values above show is much
larger (a factor of 1.7 on AP5, up to 30 elsewhere).

**Sensitivity to the slope definition** (variant 1, all ten tiles):

| slope | OLS b | 95 % | RMA b | 95 % |
|---|---|---|---|---|
| S_up_5 | 0.06 | [−0.65, 0.72] | 1.04 | [−1.47, 1.42] |
| S_up_10 (default) | 0.04 | [−0.66, 0.76] | 1.04 | [−1.49, 1.52] |
| S_up_20 | 0.03 | [−0.65, 0.76] | 1.00 | [−1.39, 1.44] |
| S_centred_10 | 0.06 | [−0.63, 0.75] | 1.02 | [−1.40, 1.42] |

The exponent moves by at most 0.02 (OLS) or 0.04 (RMA) across the four
definitions. The conclusion does not depend on the window.

## 5. Supplementary: is the test able to see a relation at all?

These analyses are mine, not the directive's; they are kept in
`fits.json["supplementary"]` and figure 6.

**Common slope–area curve null.** If all tiles shared one S(A) relation for
their hollows, the slope at A = A_crit would be S(A_crit) and any variation of
A_crit would yield log A_crit = const − (1/θ_h) log S regardless of cause.
Heads were therefore also detected at fixed thresholds of 1e3, 2e3, 5e3, 1e4
and 2e4 m² on every tile. Within tiles the median head slope changes little
with threshold: the pooled log-log slope θ_h is 0.12 (r² = 0.94 with tile
intercepts; AP medians 0.57–0.87 across the whole range, GM 0.13–0.37). A
common curve would thus imply b = −1/θ_h = −8, which is nowhere near the
fitted values. The slope at a head is a tile property, as Montgomery &
Dietrich (1989) found for hollows ("valley slope does not vary with drainage
area upslope of channel heads"), and the across-tile test is a genuine test.
It simply comes out negative.

**Slope at a fixed area.** log A_crit against the median slope of the heads
detected at A = 5000 m² on every tile: OLS b = 0.14 [−0.54, 0.81], r² = 0.02.
Steeper tiles do not have smaller A_crit.

**Stage-2 A_crit.** The same variant-1 fit with the stage-2 A_crit (K and D
held at the measured values; heads re-detected at those thresholds, 91–3239
kept per tile): OLS b = 0.91 [0.01, 3.11], RMA b = 1.95 [0.28, 3.46],
r² = 0.22. With this A_crit the relation is positive: steeper tiles have the
larger threshold. The sign of the result depends on which optimization stage
supplies A_crit.

## 6. Figures (`outputs/acrit_slope/figures/`)

- `fig1_acrit_vs_slope.png`: A_crit against median `S_up_10`, IQR bars, AP5
  misfit bar, variant-1 RMA line with its bootstrap band, A ∝ S⁻² reference
  through the nine-tile centroid.
- `fig2_head_slopes_by_tile.png`: box plots per tile ordered by A_crit.
- `fig3_sensitivity.png`: exponent against slope definition.
- `fig6_supp_slope_area_structure.png`: median head slope against threshold
  area per tile, with the A_crit point marked.
- `qc_maps/<tile>.png`: heads on the hillshade, kept and excluded by reason.
- `qc_profiles/<tile>.png`: ten random kept heads per tile with the 5, 10,
  20 m chords and the centred 10 m chord.

## 7. What the evidence supports

Observation: across the ten tiles the optimized A_crit is unrelated to the
slope at the partition heads (b ≈ 0, r² ≈ 0), under every slope definition;
AP tile 5 has an A_crit about four times lower than the other AP tiles at
the same head slope. The sites differ threefold in head slope and not at all
in A_crit.

Interpretation: the Table 1 A_crit values do not behave like a Montgomery &
Dietrich channel-head threshold. Whatever the optimizer is fitting with
A_crit, it is not the position on an A S² = const envelope. The AP5 anomaly
needs another explanation; the obvious candidate from the width work is that
AP5's small A_crit lets the partition reproduce the parallel rills and
road-side channels on its steep walls, which are not hollows at all.

Where the evidence is weak:

- Ten tiles, two sites, five each. The intervals of every variant are wide
  enough to contain exponents between about −0.7 and +0.7 even where they
  exclude −2; the test can reject A ∝ S⁻² but cannot establish a precise
  alternative.
- Head slopes at AP are bimodal because of benches and roads; the medians
  are robust to that, the IQR bars are not informative.
- A_crit uncertainty is quantified only at fixed K, D and only for AP5; the
  joint trade-off is large (stage-2 versus final values) and changes the sign
  of the fitted exponent. This is the largest unquantified source of
  uncertainty in the test.
- The sites differ in climate, lithology and land use; with n = 5 per site,
  the site-intercept model cannot separate a slope effect from a site effect.

## 8. Evaluation of the plan against Montgomery & Dietrich

Read before deciding how to use this test in the manuscript.

1. **What M&D measured and what this test measures are different things.**
   M&D (1988, 1989) mapped channel heads in the field ("the upslope limit of
   erosion and concentration of flow within steepened banks"), measured the
   hollow-axis slope immediately above the head with a Brunton compass, and
   planimetered the source area from a basemap. Their inverse relation is a
   relation between where a channel actually starts and the slope of the
   hollow it starts in. The partition heads here are where a single fitted
   area threshold is crossed; by construction they all sit at A ≈ A_crit, and
   the test asks whether the one number per tile moves with the average
   slope at those points. A negative result does not say the tiles violate
   M&D; it says the model parameter is not that threshold. The manuscript
   already says this in section 4.3 ("A_crit should be understood as a
   steady-state stand-in for this richer threshold physics"); Test 1 as run
   supports keeping that sentence and not claiming more.

2. **The −2 exponent is the envelope form, not M&D's regression.** The
   A S² = 4000 and 500 m² bounds of M&D (1992, their equations 3 and 4) are
   drawn by eye around the Tennessee Valley data; the regression in M&D
   (1989, equation 10) is A ∝ tan θ^(−1.65) (R² = 0.75, OLS with A as
   dependent). The 1988 Nature paper gives no exponent. Their landslide
   theory (1988 equation 1; 1989 equation 7) is not a power law in S at all
   and only applies above tan θ ≈ 0.5. "Consistent with −2" is therefore a
   soft target; the intervals here would also need to be compared with −1.65
   and with the theory's non-power-law shape. Given b ≈ 0, that does not
   change the conclusion, but the report and any manuscript text should not
   present −2 as a sharp prediction.

3. **M&D found the slope within a hollow to be nearly constant above the
   head** (1988 figure 1c; 1989 figure 8), which is what makes their test
   work: the head's position sets A, the hollow sets S, and the two are
   independent measurements. The pseudo-head analysis shows the same on
   these tiles (θ_h = 0.12), so the design of Test 1 is sound in that
   respect. What the directive did not anticipate is the opposite problem:
   the sites separate so completely in slope that the ten points form two
   clusters, and any "common slope" is a line between clusters.

4. **The climate ordering goes the wrong way.** M&D (1988) report larger
   source areas at the same slope in drier regions (southern Sierra, 260 mm,
   above Marin, 760 mm, above Coos Bay, 1500 mm). Here the humid Allegheny
   Plateau needs AS² of 3300–4900 m² and the semi-arid Gabilan Mesa
   145–790 m². If A_crit were a hydrologic channel-initiation threshold, the
   sites would plot the other way round. This is a second, independent reason
   to read A_crit as a morphologic fitting parameter rather than a
   channel-head threshold. (I have not checked the site precipitation values
   for AP and GM against a source; the qualitative contrast humid versus
   semi-arid is from the manuscript.)

5. **The A_crit uncertainty is the weak point of any A_crit-based test.** The
   stage-2 to final-stage ratios (1.7–30) show that A_crit trades off with K
   and D over more than an order of magnitude at nearly the same misfit. A
   misfit profile at fixed K, D (the AP5 bar) understates this badly. If this
   test, or any A_crit-versus-topography comparison, is to carry weight in
   the resubmission, it needs an A_crit interval from the joint problem
   (profile likelihood over A_crit with K and D re-optimized at each value,
   or the MSE surface already computed in the notebooks' grid searches). That
   is a model computation and therefore outside this directive.

6. **What would be the stronger test.** Compare the partition heads not with
   each other but with mapped or objectively extracted channel heads on the
   same tiles (Clubb et al. 2014 curvature-based extraction, or the
   bank-break channelized points of the width analysis). M&D's relation is
   between real heads and their hollow slopes; whether those heads obey
   A S² = const on these tiles, and whether A_crit falls inside the resulting
   envelope, would test the manuscript's section 4.3 claim directly. The
   width work already found that bank-break channels below ~1e5 m² are
   hollows and rills, which bears on whether such heads can be located at
   all on these DEMs.

## Files

`outputs/acrit_slope/`: `heads.csv` (628 rows), `excluded_heads.csv` (150),
`pseudo_heads.csv`, `pseudo_head_summary.csv`, `tile_summary.csv`,
`fits.json`, `run_info.json`, `figures/` as listed. Code: `acrit_slope/`
(`config.py` holds every threshold), tests `tests/test_acrit_slope.py`
(10 passed in the job).

## 9. Head-quality check (added 2026-11-04 after review)

Steve's review of the QC maps questioned whether the blue dots are channel
heads at all (GM5 heads inside the trunk valley; convex and plateau
profiles). `acrit_slope/qc_diagnostics.py` redraws every QC map with log₁₀ A
overlaid on the hillshade (`figures/qc_maps/`, pixels with A ≥ 10³ m² in a
single-hue ramp, trunk = A ≥ 10⁵ m²), classifies every kept head
(`heads_qc.csv`, `qc_class_summary.csv`) and repeats the fits on the
remaining heads (`fits_hollow_only.json`). Job 51085854.

Classes: `trunk_floor` = within 30 m of a trunk pixel and S_up_10 < 0.10;
`plateau` = S_up_20 < 0.05; `hollow` = everything else (the name is the
residual class, not a morphologic verdict). Profile shape over the upslope
path: slope over 10–20 m divided by slope over 0–10 m, > 1.25 `steepening`,
< 0.8 `flattening`, else `uniform`.

| tile | kept | trunk_floor | plateau | residual "hollow" | steepening | uniform | flattening | median S_up_10 all | median S_up_10 residual |
|---|---|---|---|---|---|---|---|---|---|
| AP1 | 30 | 6 | 0 | 24 | 7 | 17 | 4 | 0.623 | 0.761 |
| AP2 | 40 | 4 | 1 | 35 | 16 | 18 | 3 | 0.646 | 0.690 |
| AP3 | 26 | 6 | 1 | 19 | 7 | 10 | 7 | 0.583 | 0.706 |
| AP4 | 35 | 1 | 0 | 34 | 13 | 15 | 6 | 0.785 | 0.790 |
| AP5 | 159 | 1 | 0 | 158 | 54 | 81 | 24 | 0.729 | 0.729 |
| GM1 | 13 | 1 | 0 | 12 | 0 | 11 | 1 | 0.201 | 0.201 |
| GM2 | 46 | 10 | 0 | 36 | 10 | 33 | 1 | 0.348 | 0.363 |
| GM3 | 34 | 1 | 0 | 33 | 5 | 25 | 4 | 0.176 | 0.176 |
| GM4 | 23 | 3 | 0 | 20 | 3 | 19 | 1 | 0.306 | 0.340 |
| GM5 | 72 | 15 | 3 | 54 | 18 | 39 | 7 | 0.232 | 0.349 |

What the overlays show, tile by tile (my reading; the maps are the evidence):

- **GM5, GM2 (trunk floors).** 15 of 72 and 10 of 46 kept heads sit on the
  floor of the trunk valley: parallel D8 paths run along the 30–60 m wide
  floor beside the trunk and cross A_crit before they join it. These are
  not heads of anything. The automatic class catches most of them; a few
  more sit on the floor with slopes above 0.10.
- **AP5 (rills and roads).** The great majority of its 159 heads are at the
  tops of straight, parallel, closely spaced flow lines running directly
  down the 35–40° valley walls in the south and north-east of the tile, and
  along the two roads that cross it (ditch-line heads every few tens of
  metres). The maps for the width work already identified AP5's small-A
  "channels" as parallel rills on steep slopes. Almost none of these are
  hollows in the Montgomery & Dietrich sense; the class filter does not
  remove them because they are steep and far from the trunk.
- **AP1–AP4 (walls, benches, roads).** Heads at A ≈ 6–10 × 10³ m² sit at
  the lower ends of hollows where they drop over the steep valley walls,
  on the walls themselves (parallel lines in the south-east of AP1), along
  the trunk floor (triangles), and on benches and road cuts (the plateau
  profiles in `qc_profiles/AP1.png`). The steepening profiles (7–16 per
  tile) are heads at the base of a convex wall below a bench: slope
  increases upslope because the path climbs the wall.
- **GM1, GM3, GM4, and the hollow tips of GM2** are the only tiles where the
  heads mostly look like M&D heads: single lines at the convergent tips of
  broad hollows on the noses, uniform profiles (11 of 13, 25 of 34, 19 of
  23), narrow slope distributions.

Effect on the fits: removing the trunk_floor and plateau classes raises the
GM5 median slope from 0.23 to 0.35 and the AP1/AP3 medians by 0.1–0.14,
and leaves the result unchanged: ten-tile OLS b = −0.01 [−0.63, 0.88],
r² = 0.00; common slope with site intercepts b = −1.08 [−2.57, 1.13],
r² = 0.21.

Assessment. The question "do these heads obey A S² = const" cannot be
answered with this head set, independently of the fit result, because on
AP the set is dominated by rills, wall-base points and road ditches, and on
GM it is contaminated by valley-floor paths. The negative result in
sections 4–7 therefore says only that the partition's threshold crossings
are not channel heads; it does not test whether real channel heads on these
tiles follow M&D. To test that, the heads have to be found
morphologically (bank-defined, as in the width work's channelized points,
or by curvature-based extraction), and the hollow slope above them measured;
A_crit then enters as the model's prediction to be compared with the
envelope of real heads. That is a different directive.
