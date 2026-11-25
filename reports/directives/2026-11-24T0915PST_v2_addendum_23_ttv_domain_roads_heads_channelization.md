# [2026-11-24 09:15 PST] Addendum 23: Tennessee Valley on one clipped domain, the road check, a 1 m head finder, and a fair measure of channelization

This addendum follows Addendum 22 and its report `reports/v2/20261124T0830PST_ttv_cleanup_head_metrics.md`. Continue on branch `v2.3-tiles`. One stop, at the end. Readings are written in §7, before any result.

## 0. Why this addendum exists

Steve's review of Addendum 22 raised four points.

1. **The road.** The straight line of extra heads on the AD model's TTV2 surface may follow Tennessee Valley Road. A road bed can force the observed flow routing into a straight line along the valley floor, which every model then incises (the AD model most deeply).
2. **The head finder fails on the real landscape.** Run on the lidar itself, the 12 m method pairs only 19 of 61 mapped heads within 25 m. A method that cannot find the heads on the real landscape is not a fair test of the models. This addendum makes one bounded attempt at a better head finder on the 1 m grid. If it also fails on the lidar, channel heads are reported as supporting evidence only and not pursued further.
3. **Drainage density.** The AD model visibly makes more channels, as its K/D ratio implies, but channel counts from a head finder are not a fair measure of that. This addendum adds two measures that need no head finder and no channel definition.
4. **The 12 m grid.** The Addendum 22 maps show 12 m surfaces, which makes it look as if the models were fitted at 12 m. They were not: every model is fitted on the 1 m grid. Only the head finding and the field thresholds used 12 m block means. The report must say exactly which grid and window each quantity uses, and the figures must show the 1 m surfaces.

**Also:** the analysis moves from three separate 1 km tiles to one domain cut from the full-site 1 m grid around the M&D area, so that there is one grid and one lattice, and no tile seams.

## 1. Orientation
- **Read, in order:**
  1. the Addendum 22 report (all of it);
  2. its scripts `reports/v2/scripts/a22_*.py`;
  3. Addendum 21 §1 (parameter values);
  4. Addendum 19 §1 decision 1 and its report §1 item 5 (the TIGER road layer);
  5. this file.
- **Paths:**
  - Addendum 22 run: `$GROUP_HOME/lemcvt_results/20261124T000138Z_a22_ttv_cleanup_head_metrics/`;
  - TTV full-site 1 m grids (filled, area, flow direction; EPSG:6339): `$GROUP_HOME/lemcvt_results/20261118T160326Z_a15_stage_b/grid_TTV/`;
  - M&D georeference: `.../md_georef/md1989_fig3.gpkg` (`study_area`, `channels`);
  - roads: `$GROUP_HOME/lemcvt_data/source/TIGER2024_roads/` (Marin County, all road and trail classes);
  - Steve's heads: `data/ttv/ttv_heads_manual_steve_20261119.csv`.
- **Standing rules** as Addendum 19 §0.4:
  - Slurm limits, no submission over 300 core-hours;
  - never commit `papers/`, data rasters or manuscripts.
- **Code:** new code goes in `reports/v2/scripts/a23_*.py`, figures in `reports/v2/figures/a23/`. A package change is allowed only where §2.1 needs it (an arbitrary domain mask), with tests, the full suite on `serc`, and the Addendum 18 regression gate.
- **Before starting:** branch state, tests on `serc`, no job running. Where this addendum and the repo disagree, the repo wins; report and stop.

## 2. Decisions (planner, 2026-11-24, approved by Steve)

### 2.1 One domain around the M&D area
- **Source grid:** the TTV full-site 1 m grid. No tile is used in this addendum.
- **Domain:** the model needs real drainage boundaries, and the M&D polygon edge is not one. So:
  1. on the full-site 1 m D8 routing, find every cell on the polygon boundary where flow leaves the polygon with drainage area ≥ 10⁴ m² (the outlets);
  2. the domain is the union of the drainage basins upslope of those outlets;
  3. the outlet cells are held at their observed elevation; every other domain edge is a drainage divide.
  - Report the domain area, the number of outlets, and the share of the polygon the domain covers.
  - Polygon cells that drain out of the polygon at less than 10⁴ m² (the hillslope fringe) are outside the domain. Report how much of the polygon they are.
- **Scoring:** misfit is computed over domain cells inside the M&D polygon. The domain-wide misfit is reported beside it.
- **Re-measured inputs on the domain**, by the same methods as before, each reported beside the Addendum 21/22 per-tile values:
  - K₀ = U/k_s, with k_s by χ–z on the domain's 1 m network at m = 0.33;
  - the crest curvature and the physical-D check;
  - the relief D, traced to the mapped-head network.

  U, m and the physical D (0.0025 m²/yr) are unchanged.
- **Domain check:**
  - If the package solver accepts an arbitrary domain mask with fixed outlet cells, use it.
  - If not, add that with tests.
  - Check: on TTV2 cut as before, the domain route with the tile as its mask reproduces the Addendum 22 TTV2 fits (relative 10⁻⁸).

### 2.2 The road check
- Rasterise the TIGER roads and trails (all classes) on the 1 m domain grid. Draw them on every map in this report.
- **Is the AD gorge line a road?**
  - On the Addendum 22 TTV2 surfaces, report the share of the AD model's 54 too-steep-wall extra heads within 15 m of a road or trail.
  - Report the share of the observed 1 m D8 trunk flow path (the cells along that line with drainage area ≥ 10⁵ m²) within 5 m of a road.
  - Figure: the Addendum 22 figure 7 redrawn on the 1 m lidar hillshade, with the roads drawn.
- **Every surface in this addendum:** report the number of extra heads within 15 m of a road or trail.

### 2.3 Which grid each quantity uses (state it, change nothing)
The report's second section is a table with one row per quantity:
- the elevation fit;
- K₀ (χ–z);
- the crest curvature;
- the relief D;
- the field thresholds;
- head finding;
- the measures of §2.5.

For each, give the grid it is computed on (1 m, or a 12 m block mean), any window or smoothing (for example, the 11 m quadratic window for curvature), and the routing. Fitting is on the 1 m grid throughout. Say so in the report's first lines.

### 2.4 A head finder on the 1 m grid (one bounded attempt)
**Candidates**, each run on the 1 m lidar of the domain:
1. **Multiple-flow-direction drainage area** (Freeman 1991, exponent 1.1) on the 1 m filled grid. Slope at a cell: the steepest-descent slope over 10 m downstream (as `slope_threshold.valley_slope_reach`).
2. **D8 on a smoothed 1 m surface:** Gaussian smoothing with σ = 3 m and σ = 6 m, then D8 area and the same 10 m slope, all on the 1 m grid.
3. **The Addendum 22 12 m method**, as the reference.

**Calibrate and test without circularity:**
- set each candidate's area × slope² threshold as the median at the picks on TTV2 ground only (34 picks);
- test it on the picks on TTV1 and TTV3 ground (27 picks);
- then repeat the other way (calibrate on TTV1 + TTV3, test on TTV2);
- report both, and the pooled result.

Heads, matching and the two shares are as in Addendum 22 §2.4 (one-to-one, 25 m primary, tolerances 10–50 m). The offset metric is dropped.

**Acceptance (§7, H-A):** a candidate is accepted if, on the real lidar, it meets the reading on both test halves. The best accepted candidate (highest share found) becomes the head finder.
- **If one is accepted:** use it for the observed criterion network that the split models' channels follow, and for the head comparison on every model surface, all on the 1 m grid.
- **If none is accepted:** keep the 12 m method for the split models' channel networks (as in Addenda 21–22), report the head comparison only in an appendix, and state that channel heads are not a usable test with these methods.

### 2.5 Two measures of channelization that need no head finder
Both are applied identically to the lidar and to every model surface on the 1 m domain. Each model is compared with the lidar, not with M&D's map.

1. **Power spectrum (Perron et al. 2008):**
   - the 2-D power spectrum of each surface over the largest square inside the domain;
   - detrend with a fitted plane and apply a Hann window, as in Perron et al. (2008);
   - bin the power by wavelength in logarithmic bins from 4 m to 500 m;
   - report, for each model, model power ÷ lidar power in each bin, and the wavelength where the lidar spectrum rolls off.

   More power than the lidar at wavelengths of tens of metres means finer dissection (more, closer-spaced valleys).
2. **Share of the area that is valley:**
   - contour (planform) curvature of each surface after Gaussian smoothing with σ = 10 m (as Perron et al. 2012);
   - one threshold, fixed from the lidar: its mean contour curvature + 2 standard deviations inside the domain;
   - report the share of domain area above that threshold on each surface, ÷ the lidar's share.

   Report the same at σ = 5 m as a sensitivity.

**Planner's expectation, written now (§7, C-1):** the AD model shows more power than the lidar at short wavelengths and a larger valley share, because its fitted K/D is high. The split models with M&D's rules are closer to the lidar.

### 2.6 Models (five)
All on the domain, U = 65 m/Myr, m = 0.33, misfit inside the polygon, fitted on the 1 m grid:

| Plain name | Fitted parameters |
| --- | --- |
| AD model | K, D |
| Split model, M&D area × slope² rule | K, D |
| Split model, M&D area rule | K, D |
| Split model, nothing fitted (K = measured, D = relief D, M&D area × slope² rule) | none |
| Split model, area × slope² rule fitted | K, D, threshold |

- **Thresholds:** M&D's area × slope² and area values are re-measured at the picks with the accepted head finder (§2.4). If none is accepted, they stay at 876 m² and 6,120 m² on the full-site 12 m lattice.
- **Dropped:** "split model, area rule fitted" and "nothing fitted, physical D". Their Addendum 22 results stand and are not repeated.

## 3. Part A: domain, roads, grids, head finder (no stop)
1. The domain (§2.1): figure and numbers; the re-measured K₀, crest curvature and relief D; the domain check.
2. The road check (§2.2).
3. The grid table (§2.3).
4. The head-finder candidates on the lidar (§2.4), with the reading H-A.

## 4. Part B: fits and measures
1. Fit the five models (§2.6) on the domain. Time the fitted-threshold model first; print the estimate before submitting.
2. Misfit, K and D: Addendum 22 §4 tables and readings, on the domain.
3. Channelization (§2.5) on the lidar and every model surface.
4. Heads (§2.4) on every model surface, with the accepted head finder, or in an appendix if none was accepted.

**Cost:** expected ≤ 50 core-hours. The fitted-threshold model dominates; the domain is about 2–3 km², near three tiles. No submission over 300.

## 5. Figures (rules of Addendum 22 §5, with these changes)
1. **Less text on the plot.**
   - Labels on the plot only where they do not overlap anything, and at most one short label per element.
   - Numbers go in the caption and the tables, not on the plot, unless there are six or fewer values and they fit cleanly.
   - Model names appear in a legend, in full plain words.
2. **Look at every figure before it goes in the report.** Render it, check that no text overlaps another text, a symbol or an axis, and fix it if any does. State in the report that this was done.
3. **Maps:**
   - the 1 m hillshade of the surface shown (lidar or model), never a 12 m block mean;
   - roads and trails drawn as thin grey lines, labelled "roads and trails (TIGER)" in the legend;
   - the M&D study area outlined and labelled;
   - the domain outline;
   - scale bar and north arrow.

   Where heads came from a 12 m method, the caption says so: "heads found on a 12 m block mean of this 1 m surface".
4. **No undefined shorthand.** Spell out quantities ("drainage area × slope²", "power spectrum"). A symbol appears only if a legend line on the same figure defines it.
5. **Required figures:**
   - the domain (lidar hillshade, polygon, domain, outlets, roads);
   - the road check (§2.2);
   - the head-finder test (share found and share real against tolerance, one line per candidate, both test halves);
   - misfit per model;
   - K and D per model against measured K, physical D and relief D;
   - power spectrum ratio, model ÷ lidar, against wavelength (one line per model; the band 0.5–2 shaded and labelled "within a factor of 2 of the lidar");
   - valley share per model ÷ lidar;
   - maps of the lidar and each model on 1 m hillshades with roads.

## 6. Plain names
As Addendum 22 §2.2 for the five models of §2.6, plus "Real landscape (lidar)". Internal codes only in appendix tables beside the plain name.

## 7. Readings (written now)

**H-A, head finder accepted.** A candidate is accepted if, on the real lidar, both test halves give at 25 m:
- share of mapped heads found ≥ 0.6;
- share of found heads that are real ≥ 0.5.

**Road.** The AD gorge line is **road-influenced** if at least half of its too-steep-wall extra heads lie within 15 m of a road or trail, or the observed trunk flow path lies within 5 m of a road for at least half its length. Report both numbers whatever the reading.

**C-1, channelization.** A model **matches the lidar's channelization** if:
- its spectral power ratio lies within 0.5–2 in every wavelength bin from 20 to 200 m;
- its valley share ÷ the lidar's lies within 0.67–1.5.

Report the planner's expectation (§2.5) as held or not.

**Misfit, K, D:** as Addendum 22 §6, on the domain:
- the split models beat the AD model;
- K within 0.5–2 × measured;
- D within 0.67–1.5 × relief D.

## 8. Report and STOP
One report, `reports/v2/<YYYYMMDDTHHMM>PST_ttv_domain_roads_heads_channelization.md`, figures in `reports/v2/figures/a23/`. Outline:

1. Plain-language summary (≤ 10 lines), starting with: "Every model is fitted on the 1 m lidar grid."
2. The grid table (§2.3) and the model key.
3. The domain and the re-measured inputs.
4. The road check.
5. The head finder: candidates and reading H-A.
6. Misfit, K and D.
7. Channelization: power spectrum and valley share; reading C-1.
8. Heads on the model surfaces (main text if H-A accepts a finder; otherwise one line here, details in an appendix).
9. What the evidence supports, and where it is weak.
10. Decisions needed.
11. Appendices: rules and verdicts, the domain check, run ids, Slurm accounting.

**Stop.** Do not merge or tag. Push after the Addendum 2 §5 check.
