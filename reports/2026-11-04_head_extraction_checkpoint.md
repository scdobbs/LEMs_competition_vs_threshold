# Test 2 checkpoint: independently extracted channel heads (directive section 4)

Date: 2026-11-04. Branch `feature/head-extraction`, package `head_extraction/`,
outputs `outputs/head_extraction/` (final extraction job 51099893, 290 s).
Directive: `reports/directives/head_extraction_test2_directive.md`.
Literature check: `outputs/head_extraction/literature_check.md`. Nothing
pushed. **No fits have been run.** This report stops at the review gate.

## Summary

1. **Method P (Pelletier 2013 in Python) runs on all ten tiles, but the
   optimal Wiener filter leaves these DEMs essentially unsmoothed** (RMS
   change 0.04 m on AP, 0.015 m on GM). The AP surfaces are TIN-faceted and
   their radial power spectrum follows ν⁻³ almost to the Nyquist frequency,
   so the filter reads the facets as signal; 8–9 % of all AP pixels exceed
   κ = 0.1 m⁻¹ after filtering (GM: 0.4–0.6 %). On AP the resulting heads
   are crest pixels (median A = 3–4 m², 98 % with an upslope path shorter
   than 20 m), and Pelletier's downstream flagging from those crest pixels
   removes the real hollows below. **Extraction at AP fails under the
   directive's Method P as specified.**
2. On GM, Method P as specified gives 43–53 retained heads per tile, 26–38
   of them hollow-class, at the very tops of the hollows (median A =
   59–105 m², 25th–75th percentile roughly 20–400 m²), far above the
   partition heads and an order of magnitude below Kargère's r² = 3844 m².
3. **Head area is set by the smoothing scale.** With Gaussian smoothing
   added after the Wiener step (beyond the directive; DT = 0.5, 2, 5 m²),
   the GM hollow-class median area rises from ~80 m² to ~400 m² (DT = 2)
   and ~1800 m² (DT = 5), while the count falls from ~33 to ~15 to ~8 per
   tile. Any statement about "the head area" of these tiles is a statement
   about the filter.
4. **Method D (DrEICH) did not run.** The conda-forge LSDTopoTools 0.9
   binary aborts with `malloc(): memory corruption` inside
   `LSDJunctionNetwork::create` on every tile and under every parameter
   variant tried (details below). LSDTopoTools' own Wiener heads (Method W)
   do run and are kept as a comparison set; they agree with Method P within
   20 m for only 9–35 % of heads.
5. **Decisions needed at this gate** (section 7 of the directive says these
   are revisable now, not after the fits): which head set to review for GM
   (OWF, OWF + DT2, or W); whether AP is declared "extraction failed"
   without further work or whether a stronger-smoothing variant is
   admitted; and how to treat the method-agreement rule with Method D
   absent.

## 1. Inputs

- Grids: the notebooks' 1 m filled DEM, D8 flow direction and D8 area for
  each tile. A_crit bracket per tile = (stage 2, final) from the notebooks,
  as in the directive table; the values match `acrit_slope.config` exactly.
- **No unfilled DEM exists** on Sherlock for any tile (searched `data/` and
  the user's group directories `scdobbs_hilley_lem`, `scdobbs_optimize`,
  `scdobbs_minimize_unit_stream_power`: only `*_filled` grids and model
  outputs). Method P therefore runs on the filled surface; the
  `bench_plateau` class and the QC profiles are the only check on filled
  flats.
- **No roads layer exists** in the repo or the group storage; nothing was
  downloaded. The `road_ditch` class is the geometric rule only.

## 2. Methods as run

**Method P** (`head_extraction/pelletier.py`; no area anywhere, verified by
`test_heads_bit_identical_when_area_config_changes`): reflect-padded FFT,
radial power spectrum, |N|² = mean power over 0.75–1.0 Nyquist, |H|² =
power-law fit to the bins with S > 3|N|², transfer |H|²/(|H|² + |N|²);
Mitasova & Hofierka contour curvature by central differences; priority-flood
conditioning with epsilon gradients; Pelletier's paragraph-17 head
definition (highest-to-lowest visit, a head is an unflagged pixel with
κ > κt, everything on any descending pathway below a head is flagged).
κt = 0.1 m⁻¹ default, 0.05 and 0.2 m⁻¹ sensitivity.

The directive's wording ("most upslope pixel of each connected valley
segment") was also computed: it gives 8000–10800 heads per AP tile and
470–770 per GM tile (the κ > κt mask is fragmented into thousands of
components), and 98–100 % of the Pelletier heads coincide with a component
top. The Pelletier definition is used.

**Wiener-fit diagnostics per tile** (β = fitted power-law exponent of
|H|², |N|² in m² per bin, RMS = change of the DEM):

| tile | β | |N|² | RMS change (m) | signal fit reaches ν (cycles/m) |
|---|---|---|---|---|
| AP1 | −3.00 | 4.26 | 0.045 | 0.288 |
| AP2 | −2.98 | 4.30 | 0.042 | 0.288 |
| AP3 | −2.96 | 10.6 | 0.043 | 0.276 |
| AP4 | −2.99 | 4.96 | 0.042 | 0.288 |
| AP5 | −3.05 | 2.23 | 0.043 | 0.288 |
| GM1 | −3.18 | 0.25 | 0.015 | 0.269 |
| GM2 | −3.06 | 0.55 | 0.014 | 0.269 |
| GM3 | −3.19 | 0.15 | 0.013 | 0.269 |
| GM4 | −3.05 | 0.67 | 0.015 | 0.269 |
| GM5 | −3.10 | 0.38 | 0.014 | 0.269 |

The signal model dominates to 0.27–0.29 cycles/m (wavelengths down to
3.5 m); the transfer function is 0.5 at about 2.7 m wavelength on AP.
Pelletier's own real-DEM cases were "appropriately" less smoothed than his
synthetic ones, but his DEMs were not TIN interpolations of sparse ground
returns under forest. The filter is working as written; the premise that
the high-frequency content is white noise fails on AP.

**Smoothing sensitivity (beyond the directive).** Gaussian smoothing with
σ² = 2DT, DT = 0.5, 2, 5 m² (Pelletier's diffusion-filter equivalent; his
analytic case matched the OWF at DT ≈ 0.5 m²), applied after the Wiener
step, κt = 0.1. QC maps and profiles for DT = 2 are in
`figures/qc_maps_OWF_DT2/` and `figures/qc_profiles_OWF_DT2/`.

**Method D (DrEICH).** `lsdtt-channel-extraction` 0.9 (conda-forge
h33989ec_0, installed in `$GROUP_HOME/scdobbs-envs/lsdtt`) with
`print_dreich_channels: true`, m/n = tile θ, `connected_components_threshold`
10, `curvature_threshold` 0.1, `surface_fitting_radius` 6 m. It reports
"No of source nodes" and then aborts (exit 134 / signal 11) with
`*** Error in lsdtt-channel-extraction: malloc(): memory corruption ***`,
backtrace in `LSDJunctionNetwork::create`. Reproduced on AP1 (0.975 m
pixels) and GM1 (1.0 m pixels), with `raster_is_filled` true and false,
`threshold_contributing_pixels` 250 / 1000 / 4000, `number_of_junctions_dreich`
1 / 2, `connected_components_threshold` 10 / 100, `A_0` 1 / 1000, and with
a two-pixel NoData border. The Wiener option of the same binary runs in
3 s and writes sources, so the install works. About two hours of the
four-hour cap were spent; the remaining routes (build from source, the
LSDTT container) are not parameter fixes and I stopped. Logs:
`$SCRATCH/head_jobs/lsdtt_debug*/`. **Also note** (literature check and
README): LSDTT's DrEICH builds its initial network with
`threshold_contributing_pixels`, an area parameter, so even a working
Method D would not have been area-free upstream of its chi fit.

**Method W (comparison only).** LSDTT `print_wiener_channels` (Grieve et al.
2016 implementation: 6 m polynomial surface fit, q-q-plot curvature
threshold chosen by the code, e.g. 0.025 m⁻¹ on AP1, connected components,
`minimum_drainage_area` 400 m², `pruning_drainage_area` 1000 m²). Not
area-free. 63–127 heads per tile.

## 3. Head counts

Drops: `tile_edge` (< 30 m from the edge) and `path_leaves_tile` (the
largest-donor upslope path reaches the grid edge within 20 m). A path that
ends at an interior divide within 20 m is **kept** and flagged
`short_upslope_path`; such heads have no `S_up_10` (or `S_up_20`) and
cannot enter the slope-based classes. This is a correction from the first
run, which dropped every short path; the directive's rule is about paths
leaving the tile.

**Method P at κt = 0.1, OWF only (the directive's specification)**

| tile | extracted | dropped (edge) | retained | short path (< 20 m) | S_up_10 defined | hollow | valley_floor | bench_plateau | rill | road_ditch | unclassified | hollow A median (m²) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | 218 | 76 | 142 | 140 | 7 | 56 | 0 | 0 | 1 | 12 | 73 | 3 |
| AP2 | 158 | 78 | 80 | 69 | 13 | 46 | 0 | 0 | 0 | 4 | 30 | 4 |
| AP3 | 330 | 54 | 276 | 274 | 8 | 105 | 3 | 1 | 0 | 6 | 161 | 4 |
| AP4 | 184 | 52 | 132 | 123 | 16 | 39 | 2 | 0 | 0 | 8 | 83 | 3 |
| AP5 | 163 | 59 | 104 | 102 | 7 | 33 | 0 | 0 | 0 | 19 | 52 | 3 |
| GM1 | 61 | 8 | 53 | 10 | 49 | 34 | 0 | 0 | 0 | 1 | 18 | 105 |
| GM2 | 60 | 13 | 47 | 12 | 41 | 34 | 0 | 0 | 0 | 2 | 11 | 71 |
| GM3 | 55 | 9 | 46 | 20 | 29 | 31 | 0 | 3 | 0 | 0 | 12 | 71 |
| GM4 | 53 | 10 | 43 | 15 | 39 | 26 | 1 | 0 | 0 | 3 | 13 | 59 |
| GM5 | 64 | 16 | 48 | 10 | 43 | 38 | 2 | 0 | 0 | 0 | 8 | 77 |

No head was dropped for `path_leaves_tile` on any tile. On AP, the
"hollow" class here is meaningless: 55 of 56 AP1 hollow heads (and
similarly elsewhere) are crest pixels with A ≈ 3 m² that happen to sit in a
10 m block of convergent curvature; the rill and road rules cannot fire
because S_up_10 is undefined on a 2 m path.

**Method P with added smoothing, κt = 0.1** (retained / hollow / hollow A
median in m²):

| tile | OWF + DT0.5 | OWF + DT2 | OWF + DT5 |
|---|---|---|---|
| AP1 | 45 / 13 / 6 | 22 / 16 / 350 | 10 / 10 / 757 |
| AP2 | 18 / 7 / 8 | 13 / 11 / 39 | 13 / 13 / 329 |
| AP3 | 87 / 27 / 7 | 16 / 15 / 29 | 6 / 5 / 7 |
| AP4 | 62 / 16 / 41 | 24 / 18 / 169 | 23 / 22 / 74 |
| AP5 | 52 / 12 / 5 | 33 / 27 / 59 | 27 / 26 / 26 |
| GM1 | 24 / 24 / 109 | 16 / 16 / 1243 | 11 / 11 / 5273 |
| GM2 | 22 / 19 / 84 | 11 / 11 / 417 | 11 / 10 / 1805 |
| GM3 | 29 / 23 / 266 | 15 / 14 / 372 | 3 / 3 / 1402 |
| GM4 | 20 / 18 / 83 | 13 / 13 / 176 | 6 / 5 / 7173 |
| GM5 | 37 / 32 / 107 | 20 / 20 / 444 | 12 / 12 / 828 |

Median S_up_10 of the hollow class: GM 0.27–0.63 at every smoothing; AP
0.87–1.10 at DT = 2 and DT = 5 (41–48°: these are wall points, see
section 5).

**Curvature-threshold sensitivity, OWF, retained heads:** AP1 189 / 142 /
82, AP2 118 / 80 / 27, AP3 414 / 276 / 120, AP4 163 / 132 / 72, AP5 131 /
104 / 67, GM1 71 / 53 / 17, GM2 57 / 47 / 12, GM3 135 / 46 / 25, GM4 97 /
43 / 18, GM5 53 / 48 / 28 at κt = 0.05 / 0.10 / 0.20 m⁻¹
(`figures/fig_threshold_sensitivity.png`). The threshold changes the count
by a factor of 2–5 but, on AP, not the nature of the heads.

**Method W (LSDTT Wiener), retained / short path / hollow / valley_floor /
unclassified / hollow A median:** AP1 98 / 52 / 81 / 2 / 15 / 24; AP2 114 /
68 / 103 / 5 / 6 / 12; AP3 97 / 56 / 87 / 2 / 8 / 16; AP4 87 / 55 / 80 / 0 /
7 / 16; AP5 116 / 83 / 104 / 2 / 10 / 7; GM1 56 / 11 / 35 / 18 / 2 / 167;
GM2 71 / 20 / 57 / 13 / 1 / 140; GM3 90 / 25 / 78 / 10 / 1 / 336; GM4 67 /
15 / 46 / 18 / 3 / 130; GM5 72 / 28 / 56 / 14 / 1 / 27. On AP, W heads are
also mostly crest points (over half with paths < 20 m, median A 7–24 m²);
on GM a fifth to a quarter of them sit on trunk floors (`valley_floor`),
which Method P almost never does.

## 4. Method agreement (nearest neighbour within 20 m; fraction of the first
set matched, fraction of the second set matched, median separation)

| tile | P(OWF) vs W | P(OWF) vs P(DT2) | P(DT2) vs W |
|---|---|---|---|
| AP1 | 0.24, 0.15, 12.0 m | 0.25, 0.45, 13.3 m | 0.27, 0.06, 5.4 m |
| AP2 | 0.35, 0.12, 9.8 m | 0.00, 0.00, – | 0.46, 0.05, 10.1 m |
| AP3 | 0.28, 0.32, 13.6 m | 0.01, 0.06, 11.7 m | 0.31, 0.05, 16.7 m |
| AP4 | 0.02, 0.05, 13.3 m | 0.22, 0.25, 9.8 m | 0.46, 0.11, 5.9 m |
| AP5 | 0.30, 0.16, 10.9 m | 0.00, 0.00, – | 0.45, 0.14, 7.4 m |
| GM1 | 0.09, 0.11, 10.0 m | 0.25, 0.56, 10.8 m | 0.50, 0.16, 11.5 m |
| GM2 | 0.34, 0.14, 12.5 m | 0.30, 0.73, 6.8 m | 0.55, 0.08, 13.7 m |
| GM3 | 0.13, 0.06, 17.8 m | 0.24, 0.47, 10.3 m | 0.40, 0.07, 5.0 m |
| GM4 | 0.23, 0.18, 10.9 m | 0.21, 0.54, 9.5 m | 0.46, 0.12, 8.2 m |
| GM5 | 0.25, 0.15, 14.5 m | 0.38, 0.70, 9.8 m | 0.65, 0.18, 8.5 m |

The directive's 60 % agreement rule referred to Methods P and D; D did not
run. No pair of the sets that did run reaches 60 %. Full table, including
DT = 0.5 and 5, in `run_info.json` under `method_agreement`.

## 5. Reading of the maps

Maps viewed in full: AP1, AP3, AP4, AP5, GM1, GM3, GM5 (OWF and DT = 2);
GM2, GM4 and AP2 were judged from the tables only.

**AP tiles.** With the OWF alone, the heads are a scatter of crest pixels
across the plateau surfaces and along ridge lines (AP3: 276 retained,
274 with paths under 20 m, concentrated on the north-east plateau). With
DT = 2 the heads reduce to 13–33 per tile and sit (AP1, AP5) in a row along
the road or railroad cut at the top of the tile, at the plateau rim where
the walls begin, and (AP4, AP5) at the tops of the parallel gullies on the
40° walls; the AP1 DT = 2 profiles show 42–48° straight walls above the
head, benches and steps. **No AP tile produces a set that looks like
hollow heads**, at any smoothing or threshold tried. The classes do not
catch this because the rim and gully points are convergent at 10 m, steeper
than 30° but not grouped as parallel neighbours within 30 m, and not
collinear by the road rule except where they line the road itself (AP1:
12 road_ditch at OWF, 0 at DT = 2 because the count falls below 4 within
60 m). On the directive's own validity rule, **AP fails** unless the review
admits the DT = 2 or DT = 5 rim-and-gully heads as the AP "heads", which
I would not recommend.

**GM tiles.** With the OWF, the blue (hollow) heads sit at the upslope tips
of the hollows on the broad noses, consistently upstream of the Test 1
partition heads (grey), with a scatter of unclassified heads on planar
side slopes (GM1: a row of five on the south-east slope; these are
terracette or noise heads, not hollows). With DT = 2 the planar-slope heads
disappear and 11–20 heads per tile remain, one per hollow, at the hollow
tips (GM1, GM3, GM5 viewed): these are the cleanest Montgomery & Dietrich
valley heads this test has produced, at A ≈ 200–1200 m² (median per tile),
i.e. still above Kargère's 3844 m² characteristic area by a factor of 3–10.
GM5 has 2 heads on the trunk floor at OWF (`valley_floor`), none at DT = 2.
**GM passes the "≥ 10 hollow heads on ≥ 4 tiles" rule** at OWF (26–38) and
at DT = 2 (11–20 on all five).

## 6. What needs deciding before the analysis phase

1. **Which head set for GM.** Options: (a) OWF as specified (33 hollow heads
   per tile, A ≈ 60–100 m², includes some planar-slope false heads);
   (b) OWF + DT2 (11–20 per tile, A ≈ 200–1200 m², visually clean);
   (c) W (LSDTT, area-dependent, 35–78 hollow per tile at A ≈ 130–340 m²
   plus trunk-floor points). My recommendation is (b) as the primary set
   with (a) as the sensitivity, stated as a deviation from the directive,
   because the OWF step is not doing what Pelletier intended on these
   surfaces and the DT = 2 heads are the ones that look like hollow heads
   on the maps. Either way, head area is a function of filter scale and the
   report must say so.
2. **AP.** Declare "extraction failed at this site" now (the directive
   allows this), or admit a rim-and-gully set for 6a–6f with that label.
   I recommend the former for 6b–6d and running only 6a and 6e on the
   DT = 2 set as a described, not tested, comparison.
3. **Method agreement rule.** With D absent, the rule cannot be applied as
   written. P vs W agreement is 9–35 % and the two implementations differ
   in smoothing scale and in area dependence, so I would not substitute W
   for D.
4. **Review file.** `heads_review_template.csv` lists every retained Method
   P head (all smoothings, `smoothing` column) with `keep = True`; copy to
   `heads_review.csv`, set `keep` and `reason` for edits, and the analysis
   phase will read it. If the choice is (b), filter on
   `smoothing == OWF+DT2`.

## Files

`outputs/head_extraction/`: `heads_extracted.csv` (every head, every
method and setting; 2 896 P rows across 4 smoothings, 893 W rows),
`dropped_heads.csv`, `class_summary.csv`, `threshold_sensitivity.csv`,
`heads_review_template.csv`, `run_info.json` (Wiener fits, component-top
counts, agreement, LSDTT logs and exit codes), `literature_check.md`,
`figures/qc_maps/` and `qc_maps_OWF_DT2/` (log A on hillshade, classes by
colour, P circles, W diamonds, Test 1 partition heads grey),
`figures/qc_profiles/` and `qc_profiles_OWF_DT2/` (15 random hollow-class
P heads per tile), `figures/fig_threshold_sensitivity.png`. Code:
`head_extraction/` (README documents every rule), tests
`tests/test_head_extraction.py` (6 passed in the job). Working files not
committed: `outputs/head_extraction/lsdtt_work/`, `P_only/`, `smoke/`.
