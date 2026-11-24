# [2026-11-23 16:00 PST] Addendum 22: Tennessee Valley clean-up: M&D area only, better channel-head metrics, and plain-language figures

This addendum follows Addendum 21 and its report `reports/v2/20261123T1245PST_ttv_channel_head_test.md`. Continue on branch `v2.3-tiles`. One stop, at the end. Readings are written in §6, before any result.

## 0. Why this addendum exists

Steve reviewed Addendum 21 and raised three problems.

1. **Scope.** The test is meant to cover only the ground Montgomery & Dietrich (1989) mapped. Addendum 21 fitted all 11 TTV tiles and computed misfit over whole tiles, including ground outside the M&D map.
2. **Head metrics.** The two Addendum 21 head measures were (a) the median distance from each mapped head to the nearest *model channel cell* and (b) model heads ÷ mapped heads. They do not measure what we want:
   - **The distance runs one way only.** Model heads far from every mapped head cost nothing.
   - **It is measured to channel cells, not to heads.** A model channel that extends upslope past a mapped head scores 0 m. That is why the "nothing fitted, physical D" model scored about 0 m with 21 × too many heads.
   - **The count ratio says nothing about where the heads are.**
   - In the Addendum 21 TTV2 panels, the AD surface makes a long string of heads along one line, probably a valley wall made far too steep by AD's low D. The metrics barely register it.
3. **Figures.** They relied on internal shorthand (PF-A, PF-C, SJ-C, F2-relief, P4, H0/H1, α, C_F…) that was never defined on the figure, used a log axis that made a 2.3 × over-count look small, and could not be read without the report open beside them. Section 5 sets the figure rules for this addendum and every later one.

**Pre-registration note.** These metrics were chosen after Steve saw the Addendum 21 head figure. They are stated in §6 before they are computed. The Addendum 21 head verdicts (H0, H1, H2) are superseded, not revised: report them unchanged in an appendix beside the new ones.

## 1. Orientation

- **Read, in order:**
  1. the Addendum 21 report (all of it);
  2. its scripts `reports/v2/scripts/a21_*.py`;
  3. Addendum 21 §1 (parameter values, criterion, head extraction);
  4. this file.
- **Runs and data:**
  - Addendum 21 run directory: `$GROUP_HOME/lemcvt_results/20261123T184712Z_a21_ttv_channel_head_test/` (stored model surfaces, the 12 m grids, the criterion networks);
  - M&D georeference: `.../md_georef/md1989_fig3.gpkg`, with the study polygon and the traced channel lines;
  - Steve's heads: `data/ttv/ttv_heads_manual_steve_20261119.csv`.
- **Standing rules** as Addendum 19 §0.4:
  - Slurm limits, no submission over 300 core-hours;
  - never commit `papers/`, data rasters or manuscripts;
  - no package code changes unless unavoidable (then tests, the full suite on `serc`, and the regression gate).

  New code goes in `reports/v2/scripts/a22_*.py`, figures in `reports/v2/figures/a22/`.
- **Before starting:** confirm branch state, tests on `serc`, and that no job is running. Where this addendum and the repo disagree, the repo wins; report it and stop.

## 2. Decisions (planner, 2026-11-23, approved by Steve)

### 2.1 Scope: the M&D study area only
- **Tiles:** TTV1, TTV2, TTV3 only. The other eight TTV tiles are not used in this addendum.
- **Misfit:** computed only over 1 m cells inside the M&D study polygon (`study_area` in `md1989_fig3.gpkg`). The model is still solved on the whole tile (it needs the tile's boundaries), but only cells inside the polygon enter the misfit.
  - Report the number of polygon cells per tile.
  - Report the whole-tile misfit beside it, for continuity with Addendum 21.
- **Refit under the polygon misfit.** Every fitted model of §3 is refitted with polygon-only misfit; fixed models are re-scored.
  - If the reductions accept a cell mask (the Gram terms are sums over cells, so they should), use it.
  - If not, add a mask argument with tests.
  - **Check:** with the mask set to every tile cell, each model must reproduce its Addendum 21 fit exactly (relative 10⁻⁸).
- **Parameter values:** unchanged from Addendum 21 §1.1 (U = 65 m/Myr, m = 0.33, K₀ = U/k_s, D_phys = 0.0025 m²/yr, D_relief per tile on the mapped-head network). Field thresholds unchanged:
  - area threshold 6,120 m²;
  - area × slope² threshold 876 m².

  Sensitivity: the 16-placement medians (6,228 m² and 817 m²). One line in the report; no verdict depends on it.

### 2.2 The models, with plain names
Use exactly these names in every figure, table heading and sentence of the report. Internal codes may appear only in the appendix tables, and only beside the plain name.

| Plain name (use this) | What it is | Fitted parameters | Internal code (appendix only) |
| --- | --- | --- | --- |
| **Real landscape (lidar)** | The observed surface, run through the same head-finding method as the models | none | obs |
| **AD model** | Co-located advection–diffusion; incision and diffusion act in every cell | K, D | AD |
| **Split model, M&D area rule** | Stream power where drainage area ≥ 6,120 m² (measured at M&D's heads), diffusion elsewhere | K, D | PF-A |
| **Split model, M&D area × slope² rule** | Stream power where area × slope² ≥ 876 m² (measured at M&D's heads), diffusion elsewhere | K, D | PF-C |
| **Split model, area × slope² rule fitted** | As above, but the threshold is chosen by the fit | K, D, threshold | SJ-C |
| **Split model, area rule fitted** | Area threshold chosen by the fit (the original tuned model) | K, D, threshold | P4 |
| **Split model, nothing fitted** | M&D area × slope² rule, K = measured K₀, D = relief D | none | F2-relief |
| **Split model, nothing fitted, physical D** | As above, but D = Heimsath's 0.0025 m²/yr | none | F2-phys |

"Split model" means the threshold-partitioned model. Define it once in the report's first section in one sentence. Every figure that shows models carries the colour and marker for each name from §5.3.

### 2.3 Finding heads on a surface (unchanged from Addendum 21)
1. Block-average the surface to 12 m.
2. Route flow (D8) on that 12 m surface.
3. A cell is a head where area × slope² first reaches 876 m² along a flow path.

The area-only version (6,120 m²) is a sensitivity. Only heads and mapped heads inside the M&D polygon are counted. The Perron valley method is dropped: it finds valley floors, not channel heads.

### 2.4 The new head metrics
For each model on TTV1–TTV3, pooled and per tile:

1. **One-to-one matching.** Pair mapped heads with model heads by minimum total distance (Hungarian assignment), allowing a pair only if the two heads are within a tolerance *t*. Each mapped head and each model head can be used once. Count:
   - **matched heads:** mapped heads with a model head paired to them;
   - **missed heads:** mapped heads with no pair;
   - **extra heads:** model heads with no pair.

   Report:
   - **share of mapped heads found** = matched ÷ mapped (higher is better);
   - **share of model heads that are real** = matched ÷ model heads (higher is better).

   Compute both at t = 10, 15, 20, 25, 30, 40 and 50 m. **t = 25 m is primary** (about two 12 m cells, and the georeference error of the M&D map is about 10 m).
2. **Along-valley offset of each matched pair.** On the *observed* 12 m flow routing (the same routing for every model, so offsets are comparable):
   - trace the flow path downstream from the mapped head;
   - if the model head lies on or within one cell of that path, the model channel starts **downslope** of the mapped head (channel too short): offset = + path distance;
   - trace downstream from the model head; if the mapped head lies on that path, the model channel starts **upslope** (channel too long): offset = − path distance;
   - otherwise the pair is **in a neighbouring hollow**: no offset, counted separately.

   Report the median offset, the share too long, too short and in a neighbouring hollow, per model.
3. **Channel length inside the M&D area.**
   - Total length of the model's 12 m channel network inside the polygon (sum of cell-to-cell flow distances).
   - Compare with M&D's traced channel length inside the polygon, from `md1989_fig3.gpkg`.
   - Report model ÷ mapped (1 is a match).
4. **Where the extra heads are.** For each model, classify every extra head by the local slope of the model surface at 12 m:
   - on slopes steeper than the steepest 5 % of the real landscape (a "too-steep wall");
   - on a tile edge (within 24 m of the tile boundary);
   - other.

   This tests the planner's reading of the AD string of heads.

### 2.5 The AD check
On TTV2:
- map the AD surface's slope and the real landscape's slope side by side, on one colour scale, with AD's heads overlaid;
- draw a profile across the line where AD makes the string of heads, AD and real landscape on one plot.

Report in one paragraph what the feature is.

## 3. Runs
On TTV1–TTV3, misfit inside the M&D polygon:

1. **Refits:**
   - AD model;
   - split model, M&D area rule;
   - split model, M&D area × slope² rule;
   - split model, area × slope² rule fitted (profile over the threshold as in Addendum 21; time one tile first);
   - split model, area rule fitted (profile over the area threshold as in Addendum 18).
2. **Re-scores:** the two "nothing fitted" models.
3. **Head finding (§2.3)** on every surface, including the real landscape, then the metrics of §2.4.
4. **The AD check (§2.5).**

**Cost:** expected ≤ 30 core-hours (the two threshold-fitted models dominate: about 1.5–4 h per tile each). Print the estimate first. No submission over 300.

## 4. What to report for each model (one row per model, TTV1–TTV3)
- misfit inside the M&D area (m²), and ÷ the AD model's;
- K ÷ measured K₀;
- D, with D ÷ physical D and D ÷ relief D;
- the fitted threshold where fitted, against the M&D value;
- at t = 25 m: matched, missed and extra heads, the two shares, the median along-valley offset, channel length ÷ mapped.

## 5. Figure and writing rules (apply to every figure and table in this report and every later one)

### 5.1 Every figure
1. **Title states the question in plain words**, e.g. "How many of M&D's mapped channel heads does each model find?", not "H1 at α = 2".
2. **No undefined abbreviation anywhere on the figure.**
   - Model names are the plain names of §2.2.
   - Quantities are spelled out ("drainage area × slope²", not "A·S²"; "measured erosion-rate-based K", not "K₀").
   - A symbol may appear only if a legend line on the same figure defines it.
3. **Axes:**
   - plain labels with units;
   - linear unless there is a stated reason;
   - a log axis carries its actual values at the tick marks, plus a note in the caption: "log scale: 2 means twice as many".
4. **Show which way is good** on every axis where it matters: an arrow and the words "better" or "closer to the map", or a shaded "matches the map" band labelled in words.
5. **Reference values are drawn and labelled in words**, e.g. a dashed line labelled "Heimsath's measured D (0.0025 m²/yr)".
6. **Values are written on the points or bars** when there are 10 or fewer of them.
7. **Every panel has its own title** naming the model and tile in words.
8. **A two-line caption under every figure:**
   - "How to read: …"
   - "What it shows: …"

   The caption names the tiles and the M&D area restriction.
9. **One colour and marker per model**, the same in every figure (§5.3).
10. **Maps:**
    - a scale bar and a north arrow;
    - the M&D study area outlined and labelled "M&D study area";
    - mapped heads drawn as **orange triangles labelled "M&D heads (mapped)"**;
    - model heads as circles: **filled = matched to a mapped head, open = extra head**;
    - missed mapped heads drawn as an **orange triangle with a black cross**;
    - matched pairs joined by a thin line.
11. **No more than six models in one panel or chart.** Where all eight are needed, split the figure.

### 5.2 The report
- Open with a **plain-language summary of at most 10 lines**: one line per finding, no codes.
- Then a **model key** (the §2.2 table, without the code column) before any result.
- Then the **hypothesis table**:

  | Hypothesis | Verdict | Evidence |
  | --- | --- | --- |
  | 1. The AD model fits only with non-physical K and D | | |
  | 2. The split model fits better than the AD model | | |
  | 3. The split model fits with physical K and D | | |
  | 4. With everything fixed at measured values, the split model reproduces the landscape and the mapped heads | | |
  | 5. With K, D and the threshold all fitted, the split model recovers the mapped heads | | |

  Evidence is written in words with numbers, TTV1 / TTV2 / TTV3, M&D area only.
- Rule names (§6) appear only in a rules appendix; the body says what was tested in words.

### 5.3 Colours and markers (fixed)

| Model | Colour | Marker |
| --- | --- | --- |
| Real landscape (lidar) | black | star |
| AD model | orange-red #d95f02 | circle |
| Split model, M&D area rule | blue #1f78b4 | triangle |
| Split model, M&D area × slope² rule | dark blue #08306b | square |
| Split model, area × slope² rule fitted | teal #1b9e77 | diamond |
| Split model, area rule fitted | purple #7570b3 | pentagon |
| Split model, nothing fitted | grey #666666 | plus |
| Split model, nothing fitted, physical D | light grey #bbbbbb | cross |

Mapped heads are always orange triangles (#e6550d); no model uses a triangle in that colour.

### 5.4 Required figures
1. **"How well does each model fit the M&D area?"** Bars: misfit inside the M&D area per model, grouped by tile, with the AD model's value marked.
2. **"How many of M&D's mapped heads does each model find, and how many extra heads does it make?"** Grouped bars at t = 25 m, per model: matched, missed, extra. A dashed line at 61 labelled "number of M&D heads".
3. **"Does the answer depend on how close counts as a match?"** Two panels:
   - share of mapped heads found against tolerance (10–50 m);
   - share of model heads that are real against tolerance.

   One line per model. The real landscape's line is the best achievable with this head-finding method, labelled as such.
4. **"Does each model start its channels too high or too low?"** Per model: along-valley offset of matched pairs, as a strip of points with the median marked. The zero line is labelled "starts where M&D mapped"; negative is labelled "channel starts upslope (too long)", positive "channel starts downslope (too short)".
5. **"Total channel length inside the M&D area"** Bars per model, with the mapped length as a labelled line.
6. **Maps, one page per tile (TTV1, TTV2, TTV3):** six panels:
   - real landscape;
   - AD model;
   - split model, M&D area × slope² rule;
   - split model, M&D area rule;
   - split model, area × slope² rule fitted;
   - split model, nothing fitted.

   Each follows §5.1 item 10.
7. **"What is the line of extra heads on the AD surface?"** The §2.5 slope maps and profile.
8. **"Are the fitted K and D physical?"** Two panels, one point per model per tile:
   - K ÷ measured K, with a shaded band 0.5–2 labelled "within a factor of 2 of measured";
   - D on a labelled log axis, with lines for Heimsath's physical D and for the relief D.

## 6. Readings (written now)

**Head matching (§2.4).**
- A model **reproduces the mapped heads** if, pooled over TTV1–TTV3 at t = 25 m, all three hold:
  - the share of mapped heads found is at least 0.5;
  - the share of model heads that are real is at least 0.5;
  - the median along-valley offset is within ± 25 m.
- The **real landscape** is the reference: report each model's two shares as a fraction of the real landscape's, and state plainly if the real landscape itself fails the reading. A model cannot be expected to beat the method's own ceiling.
- Report the result at every tolerance; say whether any verdict changes between 15 and 40 m.

**Misfit.** The split model, M&D area × slope² rule, fits better than the AD model inside the M&D area if its misfit is lower on all three tiles.

**K and D.**
- K is **physical** if K ÷ measured K is within 0.5–2 on all three tiles.
- D is **explained by relief** if D ÷ relief D is within 0.67–1.5 on all three tiles.

**The AD check.** No rule. One paragraph saying what the feature is.

## 7. Report and STOP
One report, `reports/v2/<YYYYMMDDTHHMM>PST_ttv_cleanup_head_metrics.md`, figures in `reports/v2/figures/a22/`. Outline:

1. Plain-language summary (≤ 10 lines).
2. Model key.
3. Hypothesis table.
4. Fit, K and D inside the M&D area (figures 1 and 8).
5. Channel heads: matching, tolerance, offsets, channel length, maps (figures 2–6).
6. The AD check (figure 7).
7. What the evidence supports, and where it is weak.
8. Decisions needed.
9. Appendices:
   - rules and verdicts;
   - the Addendum 21 head verdicts unchanged, for comparison;
   - the whole-tile misfits;
   - the mask checks;
   - run ids;
   - Slurm accounting.

**Stop.** Do not merge or tag. Push after the Addendum 2 §5 check.
