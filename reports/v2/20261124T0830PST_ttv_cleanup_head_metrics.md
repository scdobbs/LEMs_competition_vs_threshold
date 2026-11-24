# [2026-11-24 08:30 PST] Addendum 22: Tennessee Valley clean-up: M&D area only, one-to-one channel-head metrics, plain-language figures

Directive: `reports/directives/2026-11-23T1600PST_v2_addendum_22_ttv_head_metrics_cleanup.md` (outline fixed by its section 7). Branch
`v2.3-tiles` (from dc5ed7b, the addendum 21 report and its hillshade fix). **No package code was changed.** Every number is tied to the run
directory `$GROUP_HOME/lemcvt_results/20261124T000138Z_a22_ttv_cleanup_head_metrics/` or to a Slurm job (Appendix F). Scripts
`reports/v2/scripts/a22_{common,tile,profile,post,figs}.py`; figures `reports/v2/figures/a22/`; the machine-generated tables are in
Appendix H and the state check with every choice made in applying the directive is Appendix G. **This is the addendum's stop.** The
addendum 21 head verdicts are superseded by section 5 and are repeated unchanged in Appendix B.

"Split model" means the threshold-partitioned model: stream-power incision where a channel rule holds, linear diffusion elsewhere.

## 1. Summary

1. Inside the M&D study area the split models with M&D's own channel rules fit better than the AD model on all three tiles (0.33–0.68 × its misfit); fitting the threshold as well brings that to 0.14–0.45 ×.
2. Those split models fit with the measured K (0.85–1.24 × on every tile) and, for the two M&D-rule models, a D within 0.69–1.30 × the relief D; the AD model needs K 2.2–10 × the measured value and a D 1.4–200 × below Heimsath's.
3. With the threshold fitted, D moves above the relief range on two of three tiles (1.6–2.2 ×) and the fitted thresholds leave M&D's values: 559–4,607 m² for area × slope² against 876, and 8,536–59,849 m² for area against 6,120.
4. With nothing fitted (measured K, relief D) the split model's misfit is 0.44–1.25 × the two-parameter AD model's; with Heimsath's D instead it is 31–41 × worse.
5. The head-finding method has a low ceiling: run on the real lidar it pairs only 19 of the 61 mapped heads one-to-one within 25 m and makes 67 extra heads, so the real landscape itself fails the pre-registered reading and no model can pass it.
6. Measured against that ceiling, the nothing-fitted split model and the two M&D-rule split models find 0.8–1.1 × as many mapped heads as the real landscape with 0.8–1.2 × its share of real heads; the AD model finds 0.4 × with a quarter of the real share; the two threshold-fitted models find 0.7–0.8 × with half the real share.
7. At 25 m nearly every matched pair lies in the same or an adjacent 12 m cell, so the along-valley offset is 0 m for 71–93 % of pairs and separates no model from another.
8. Every model's 12 m channel network inside the M&D area is 0.36–0.51 × M&D's traced length, and so is the real landscape's (0.41 ×): that ratio measures the method, not the models (model ÷ real landscape 0.9–1.2).
9. The string of AD heads on TTV2 is the wall of a gorge the AD surface cuts along the trunk valley, 19 m below the lidar: 54 of its 66 extra heads sit on 12 m slopes of 1.4 where the real landscape has 0.2.
10. No head verdict changes between tolerances of 15 and 40 m; restricting the misfit to the M&D area moves the split models' K and D by under 10 % on TTV1–TTV2 and by up to 20 % on TTV3 (where the polygon holds 43 % of the tile), and the fitted area threshold on TTV3 by 61 %.

## 2. Model key

| Plain name | What it is | Fitted parameters |
| --- | --- | --- |
| **Real landscape (lidar)** | The observed surface, run through the same head-finding method as the models | none |
| **AD model** | Co-located advection–diffusion; incision and diffusion act in every cell | K, D |
| **Split model, M&D area rule** | Stream power where drainage area ≥ 6,120 m² (measured at M&D's heads), diffusion elsewhere | K, D |
| **Split model, M&D area × slope² rule** | Stream power where area × slope² ≥ 876 m² (measured at M&D's heads), diffusion elsewhere | K, D |
| **Split model, area × slope² rule fitted** | As above, but the threshold is chosen by the fit | K, D, threshold |
| **Split model, area rule fitted** | Area threshold chosen by the fit (the original tuned model) | K, D, threshold |
| **Split model, nothing fitted** | M&D area × slope² rule, K = measured K₀, D = relief D | none |
| **Split model, nothing fitted, physical D** | As above, but D = Heimsath's 0.0025 m²/yr | none |

Fixed values (addendum 21 section 1): U = 65 m/Myr, m = 0.33, K₀ = U/k_s per tile (TTV1 9.5 × 10⁻⁶, TTV2 1.3 × 10⁻⁵, TTV3 2.2 × 10⁻⁵),
physical D = 0.0025 m²/yr, relief D per tile on the mapped-head network (TTV1 0.0175, TTV2 0.0159, TTV3 0.0230 m²/yr). Head finding on
every surface: block-average to 12 m, D8 routing, a head where area × slope² first reaches 876 m² along a flow path. The 16-placement
medians of the thresholds (6,228 and 817 m²) are a sensitivity in section 5.5 only; no verdict depends on them.

## 3. Hypothesis table

Evidence is TTV1 / TTV2 / TTV3, misfit and heads inside the M&D area only; heads pooled over the three tiles at a 25 m tolerance.

| Hypothesis | Verdict | Evidence |
| --- | --- | --- |
| 1. The AD model fits only with non-physical K and D | **Supported** | Fitted K is 10.05 / 4.84 / 2.21 × the measured K (outside 0.5–2 on all three); fitted D is 0.10 / 0.013 / 0.0005 × the relief D and 0.70 / 0.08 / 0.005 × Heimsath's. |
| 2. The split model fits better than the AD model | **Supported** | Misfit ÷ AD's: M&D area × slope² rule 0.65 / 0.51 / 0.43; M&D area rule 0.68 / 0.33 / 0.36; area × slope² fitted 0.45 / 0.43 / 0.28; area fitted 0.38 / 0.30 / 0.14. Lower on 3 of 3 tiles for every fitted split model. The nothing-fitted model is 1.25 / 0.61 / 0.44 (2 of 3). |
| 3. The split model fits with physical K and D | **Supported for K; for D only in the relief sense, and only with M&D's rules** | K ÷ measured 0.85–1.24 on every tile for all four fitted split models. D ÷ relief D: M&D area rule 1.02 / 0.74 / 0.69 and M&D area × slope² rule 1.13 / 0.80 / 1.30 (within 0.67–1.5 on 3 of 3); area × slope² fitted 1.80 / 0.66 / 1.62 and area fitted 2.15 / 0.86 / 1.83 (fail, D at the box bound on TTV3 for the area fit). Every split D is 4.2–17 × Heimsath's. |
| 4. With everything fixed at measured values, the split model reproduces the landscape and the mapped heads | **Partly: the landscape to within 0.44–1.25 × AD's misfit; the heads as well as the real landscape does, which is below the reading** | Misfit 1.25 / 0.61 / 0.44 × AD with no free parameter. Heads: 21 of 61 found (share 0.34), 62 extra (share real 0.25), median offset 0 m; the real landscape through the same method gives 19 found and 67 extra (0.31 / 0.22), so the model is 1.11 / 1.15 × its ceiling, but both are below the 0.5 reading. With Heimsath's D the misfit is 41 / 33 / 31 × AD's and the surface makes 1,215 extra heads. |
| 5. With K, D and the threshold all fitted, the split model recovers the mapped heads | **Not supported** | Area × slope² fitted: 14 found (0.23), 102 extra (0.12); thresholds 4,607 / 559 / 1,553 m² against M&D's 876. Area fitted: 15 found (0.25), 132 extra (0.10); thresholds 59,849 / 8,536 / 35,026 m² against 6,120. Both find fewer mapped heads and make more extra heads than the models with M&D's rules fixed (15–18 found, 65–81 extra), at 0.46–0.55 × the real landscape's share of real heads. |

## 4. Fit, K and D inside the M&D area (figures 1 and 8)

**The M&D area on the tiles.** The study polygon holds 630,864 of TTV1's 1,000,000 1 m cells (63 %), 855,216 of TTV2's (86 %) and
426,566 of TTV3's (43 %); 22, 34 and 5 of Steve's picks fall inside it (61 in all; every pick on these tiles is inside), and M&D's traced
channels inside it run 11.2, 13.5 and 4.7 km (29.4 km in all). The model is solved on the whole tile and only polygon cells enter the
misfit. With the mask set to every cell, every model reproduces its addendum 21 fit to a relative 0 to 1.4 × 10⁻¹⁴ (Appendix D), so the
polygon numbers below differ from addendum 21 only through the mask.

**Misfit (figure 1).** One row per model and tile, misfit inside the M&D area in m² (mean squared elevation error) and divided by the AD
model's on the same tile:

| model | TTV1 | TTV2 | TTV3 | ÷ AD, TTV1 / TTV2 / TTV3 | lower than AD on all three |
| --- | --- | --- | --- | --- | --- |
| AD model | 224.0 | 206.7 | 252.5 | 1 / 1 / 1 | – |
| Split model, M&D area rule | 151.5 | 68.2 | 91.6 | 0.68 / 0.33 / 0.36 | yes |
| Split model, M&D area × slope² rule | 146.7 | 104.7 | 107.4 | 0.65 / 0.51 / 0.43 | yes |
| Split model, area × slope² rule fitted | 101.9 | 89.0 | 70.8 | 0.45 / 0.43 / 0.28 | yes |
| Split model, area rule fitted | 85.2 | 61.1 | 34.6 | 0.38 / 0.30 / 0.14 | yes |
| Split model, nothing fitted | 280.9 | 125.7 | 110.9 | 1.25 / 0.61 / 0.44 | no (2 of 3) |
| Split model, nothing fitted, physical D | 9,149 | 6,838 | 7,709 | 40.8 / 33.1 / 30.5 | no |

The ordering of addendum 21 holds inside the polygon: every fitted split model beats AD on every tile, the two-parameter M&D-rule
models by a third to two thirds, the three-parameter ones by more. The nothing-fitted model, with no free parameter, is within 0.44–1.25 ×
of the two-parameter AD; its physical-D twin is thirty to forty times worse (its crests are far too sharp, as in addendum 21).

**K and D (figure 8).** Fitted K divided by the measured erosion-rate K₀, and fitted D against the two reference values:

| model | K ÷ K₀, TTV1 / TTV2 / TTV3 | K physical (0.5–2 on all three) | D (m²/yr), TTV1 / TTV2 / TTV3 | D ÷ relief D | D explained by relief (0.67–1.5 on all three) | D ÷ Heimsath's |
| --- | --- | --- | --- | --- | --- | --- |
| AD model | 10.05 / 4.84 / 2.21 | no | 0.0018 / 0.0002 / 0.00001 | 0.10 / 0.013 / 0.0005 | no | 0.70 / 0.08 / 0.005 |
| Split model, M&D area rule | 0.85 / 1.02 / 1.04 | yes | 0.0178 / 0.0117 / 0.0160 | 1.02 / 0.74 / 0.69 | yes | 7.1 / 4.7 / 6.4 |
| Split model, M&D area × slope² rule | 0.85 / 0.99 / 0.96 | yes | 0.0198 / 0.0128 / 0.0298 | 1.13 / 0.80 / 1.30 | yes | 7.9 / 5.1 / 11.9 |
| Split model, area × slope² rule fitted | 0.91 / 0.98 / 0.99 | yes | 0.0315 / 0.0105 / 0.0372 | 1.80 / 0.66 / 1.62 | no | 12.6 / 4.2 / 14.9 |
| Split model, area rule fitted | 1.09 / 1.03 / 1.24 | yes | 0.0376 / 0.0137 / 0.0422 (bound) | 2.15 / 0.86 / 1.83 | no | 15.0 / 5.5 / 16.9 |

Fitted thresholds against M&D's: area × slope² 4,607 / 559 / 1,553 m² (M&D 876; 5.3 ×, 0.64 ×, 1.8 ×); area 59,849 / 8,536 / 35,026 m²
(M&D 6,120; 9.8 ×, 1.4 ×, 5.7 ×). Addendum 21's whole-tile values were 3,706 / 559 / 1,482 and 59,499 / 9,131 / 89,124 m².

Three things to note.
- **The AD model's D is not merely low, it is unconstrained from below on TTV3** (the 5 % identifiability interval spans 0.9 decades in D
  there, 0.26–0.31 elsewhere); the fit is advection-dominated and the surface is cut into gorges (section 6).
- **The area-rule fit on TTV3 sits at the D upper bound** of the addendum 18 box (0.0422 m²/yr) for every area threshold above 24,900 m²
  on the polygon objective. Its polygon optimum (35,026 m²) is not the whole-tile optimum (89,124 m²): the whole-tile misfit at the
  polygon fit is 171.9 m² against 87.4 m² at the whole-tile fit, so this fit is specific to the 43 % of TTV3 inside the polygon. The
  other models' polygon fits transfer to the whole tile within 2–8 % of the whole-tile optimum on TTV3, within 3 % on TTV1 and TTV2
  except the area × slope² fit on TTV1 (11 %; Appendix C).
- **Restricting the misfit to the polygon moves the split models' K and D by less than 10 % on TTV1 and TTV2** and by 7–20 % on TTV3
  (M&D area × slope² D 0.0250 → 0.0298). The AD model's D moves by 30 % on TTV1 and 48 % on TTV3 (it is poorly determined); the fitted
  area × slope² threshold moves by 24 % on TTV1 and the fitted area threshold by 61 % on TTV3 (Appendix C).

## 5. Channel heads: matching, tolerance, offsets, channel length, maps (figures 2–6)

All counts are heads inside the M&D polygon, pooled over TTV1–TTV3 (61 mapped heads), one-to-one Hungarian matching, 25 m tolerance.

### 5.1 How many mapped heads each model finds, and how many extra it makes (figure 2)

| model | matched | missed | extra | share of mapped heads found | ÷ real landscape | share of model heads that are real | ÷ real landscape | reading (both ≥ 0.5, offset within 25 m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Real landscape (lidar) | 19 | 42 | 67 | 0.31 | 1.00 | 0.22 | 1.00 | fails |
| AD model | 8 | 53 | 122 | 0.13 | 0.42 | 0.06 | 0.28 | fails |
| Split model, M&D area rule | 18 | 43 | 81 | 0.30 | 0.95 | 0.18 | 0.82 | fails |
| Split model, M&D area × slope² rule | 15 | 46 | 65 | 0.25 | 0.79 | 0.19 | 0.85 | fails |
| Split model, area × slope² rule fitted | 14 | 47 | 102 | 0.23 | 0.74 | 0.12 | 0.55 | fails |
| Split model, area rule fitted | 15 | 46 | 132 | 0.25 | 0.79 | 0.10 | 0.46 | fails |
| Split model, nothing fitted | 21 | 40 | 62 | 0.34 | 1.11 | 0.25 | 1.15 | fails |
| Split model, nothing fitted, physical D | 39 | 22 | 1,215 | 0.64 | 2.05 | 0.03 | 0.14 | fails |

**The real landscape fails the reading.** The lidar surface through the same method (12 m block mean, D8, area × slope² ≥ 876 m²) pairs
19 of 61 mapped heads within 25 m and makes 67 heads with no mapped head near them. The reading's bar (both shares at least 0.5) is
therefore above the method's ceiling, as the directive anticipated, and no model passes it. Against the ceiling the models separate into
three groups: the nothing-fitted model and the two M&D-rule models at 0.8–1.1 × the real landscape on both shares; the two
threshold-fitted models at 0.7–0.8 × on heads found and 0.5 × on heads real; the AD model at 0.4 × and 0.3 ×. The physical-D model shows
why the addendum 21 one-way distance failed: it "finds" 39 of 61 mapped heads by making 1,254 heads in all.

Per tile (matched of mapped, extra): TTV1 real landscape 8 of 22 (35 extra), AD 3 (11), M&D area rule 3 (20), M&D area × slope² rule 3
(21), area × slope² fitted 3 (52), area fitted 1 (65), nothing fitted 7 (27). TTV2: 9 of 34 (28), 5 (66), 14 (52), 10 (40), 11 (42),
14 (61), 11 (29). TTV3: 2 of 5 (4), 0 (45), 1 (9), 2 (4), 0 (8), 0 (6), 3 (6). TTV1 is where every model does worst: the mapped heads
there sit in a dense set of short, steep hollows on the north-east of the tile where the 12 m method places its heads one or two cells
off, so that even the real landscape finds only 8.

### 5.2 Does the answer depend on the tolerance (figure 3)

Share found / share real at 10, 15, 20, 25, 30, 40, 50 m:

| model | 10 m | 15 m | 20 m | 25 m | 30 m | 40 m | 50 m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Real landscape (lidar) | 0.00 / 0.00 | 0.16 / 0.12 | 0.28 / 0.20 | 0.31 / 0.22 | 0.34 / 0.24 | 0.49 / 0.35 | 0.56 / 0.40 |
| AD model | 0.02 / 0.01 | 0.05 / 0.02 | 0.10 / 0.05 | 0.13 / 0.06 | 0.15 / 0.07 | 0.23 / 0.11 | 0.28 / 0.13 |
| Split model, M&D area rule | 0.08 / 0.05 | 0.21 / 0.13 | 0.25 / 0.15 | 0.30 / 0.18 | 0.34 / 0.21 | 0.49 / 0.30 | 0.59 / 0.36 |
| Split model, M&D area × slope² rule | 0.03 / 0.03 | 0.16 / 0.12 | 0.23 / 0.17 | 0.25 / 0.19 | 0.31 / 0.24 | 0.49 / 0.38 | 0.57 / 0.44 |
| Split model, area × slope² rule fitted | 0.05 / 0.03 | 0.13 / 0.07 | 0.16 / 0.09 | 0.23 / 0.12 | 0.30 / 0.16 | 0.33 / 0.17 | 0.43 / 0.22 |
| Split model, area rule fitted | 0.08 / 0.03 | 0.16 / 0.07 | 0.20 / 0.08 | 0.25 / 0.10 | 0.31 / 0.13 | 0.36 / 0.15 | 0.43 / 0.18 |
| Split model, nothing fitted | 0.07 / 0.05 | 0.26 / 0.19 | 0.30 / 0.22 | 0.34 / 0.25 | 0.44 / 0.33 | 0.51 / 0.37 | 0.57 / 0.42 |
| Split model, nothing fitted, physical D | 0.26 / 0.01 | 0.44 / 0.02 | 0.56 / 0.03 | 0.64 / 0.03 | 0.72 / 0.04 | 0.82 / 0.04 | 0.90 / 0.04 |

No verdict changes between 15 and 40 m (or at any tolerance): no model, and not the real landscape, reaches 0.5 on both shares even at
50 m. The grouping of section 5.1 holds at every tolerance from 15 m up: the nothing-fitted and M&D-rule models track the real landscape's
line (the M&D area rule is above it on heads found at 40–50 m), the threshold-fitted models fall below it from 20 m, the AD model is lowest
throughout. At 10 m the real landscape finds nothing: a mapped head is a 1 m pick and a model head is the centre of a 12 m cell, so even a
pick inside the right cell is up to 8.5 m from it.

### 5.3 Does each model start its channels too high or too low (figure 4)

On the real landscape's 12 m routing, a matched pair whose two heads are in the same or in adjacent 12 m cells has offset 0 ("same cell",
the adjacency rule of decision 2.4.2 applied to the pair's own cells). At 25 m, about two cells, that is almost every pair:

| model | pairs | same or adjacent cell | too short (model head downslope) | too long (model head upslope) | neighbouring hollow | median offset (m) |
| --- | --- | --- | --- | --- | --- | --- |
| Real landscape (lidar) | 19 | 17 | 1 | 0 | 1 | 0 |
| AD model | 8 | 6 | 0 | 2 | 0 | 0 |
| Split model, M&D area rule | 18 | 15 | 2 | 1 | 0 | 0 |
| Split model, M&D area × slope² rule | 15 | 14 | 0 | 1 | 0 | 0 |
| Split model, area × slope² rule fitted | 14 | 10 | 4 | 0 | 0 | 0 |
| Split model, area rule fitted | 15 | 12 | 3 | 0 | 0 | 0 |
| Split model, nothing fitted | 21 | 18 | 2 | 1 | 0 | 0 |
| Split model, nothing fitted, physical D | 39 | 32 | 1 | 3 | 3 | 0 |

The median offset is 0 m for every model and the metric separates nothing. What little signal there is points the expected way: the
AD model's two offset pairs are both too long (its channels reach upslope past the mapped head), the threshold-fitted models' offset pairs
are all too short (4 of 4 and 3 of 3; their channels start below the mapped head, consistent with thresholds above M&D's on TTV1 and
TTV3). At 50 m the pairs spread (real landscape 9 too short, 4 too long, 4 in another hollow of 34) but the medians stay at 0 m.

### 5.4 Total channel length inside the M&D area (figure 5)

| model | 12 m channel length inside the area (km) | ÷ M&D's traced 29.4 km | ÷ real landscape |
| --- | --- | --- | --- |
| Real landscape (lidar) | 12.0 | 0.41 | 1.00 |
| AD model | 14.9 | 0.51 | 1.24 |
| Split model, M&D area rule | 12.1 | 0.41 | 1.01 |
| Split model, M&D area × slope² rule | 10.6 | 0.36 | 0.89 |
| Split model, area × slope² rule fitted | 11.7 | 0.40 | 0.98 |
| Split model, area rule fitted | 12.5 | 0.42 | 1.04 |
| Split model, nothing fitted | 10.8 | 0.37 | 0.90 |
| Split model, nothing fitted, physical D | 68.7 | 2.34 | 5.73 |

The 876 m² rule on a 12 m lattice gives 0.41 × M&D's traced length on the lidar itself, so the models' 0.36–0.51 × are the method's ratio,
not the models' (0.9–1.2 × the real landscape; the AD model's 1.24 × is its gorge walls, section 6). Addendum 21 measured the same gap the
other way round: 69–74 % of M&D's traced length lies within 15 m of a criterion channel, which means the criterion network on the 12 m
lattice is the trunk of M&D's network without its upper reaches. The physical-D surface, 5.7 × the real landscape, channels everything.

### 5.5 Sensitivities of the head finding

Share found / share real / extra heads, pooled at 25 m, under the four head rules (the 876 m² area × slope² rule is primary):

| model | area × slope² ≥ 876 m² | area ≥ 6,120 m² | area × slope² ≥ 817 m² (16-placement median) | area ≥ 6,228 m² (16-placement median) |
| --- | --- | --- | --- | --- |
| Real landscape (lidar) | 0.31 / 0.22 / 67 | 0.23 / 0.18 / 64 | 0.31 / 0.20 / 76 | 0.18 / 0.14 / 66 |
| AD model | 0.13 / 0.06 / 122 | 0.28 / 0.21 / 63 | 0.15 / 0.06 / 134 | 0.30 / 0.23 / 60 |
| Split model, M&D area rule | 0.30 / 0.18 / 81 | 0.23 / 0.25 / 43 | 0.33 / 0.19 / 87 | 0.21 / 0.23 / 44 |
| Split model, M&D area × slope² rule | 0.25 / 0.19 / 65 | 0.20 / 0.22 / 43 | 0.30 / 0.20 / 71 | 0.20 / 0.22 / 43 |
| Split model, area × slope² rule fitted | 0.23 / 0.12 / 102 | 0.21 / 0.19 / 54 | 0.25 / 0.12 / 111 | 0.21 / 0.20 / 52 |
| Split model, area rule fitted | 0.25 / 0.10 / 132 | 0.05 / 0.06 / 51 | 0.25 / 0.09 / 145 | 0.05 / 0.06 / 49 |
| Split model, nothing fitted | 0.34 / 0.25 / 62 | 0.21 / 0.29 / 32 | 0.34 / 0.24 / 68 | 0.23 / 0.31 / 31 |
| Split model, nothing fitted, physical D | 0.64 / 0.03 / 1,215 | 0.23 / 0.27 / 38 | 0.62 / 0.03 / 1,248 | 0.23 / 0.27 / 38 |

The 16-placement medians change no share by more than 0.05 and no verdict. The area-only rule is a different test: it removes the slope²
term that places heads on steep walls, so the AD model's heads become as good as any (0.28 / 0.21) while the real landscape's fall to
0.23 / 0.18 and the area-rule-fitted model's collapse (0.05 / 0.06: its own fitted thresholds are 1.4–9.8 × the 6,120 m² rule). The head
verdicts therefore depend on the head rule as much as on the model; the primary rule is the one M&D's heads were measured with.

### 5.6 Where the extra heads are (figure 6)

Extra heads at 25 m classed by the local 12 m slope of the model's own surface (steeper than the real landscape's 95th percentile inside
the polygon: 0.69 on TTV1, 0.62 on TTV2, 0.45 on TTV3), by tile edge (within 24 m), or other:

| model | extra | on too-steep walls | on a tile edge | other |
| --- | --- | --- | --- | --- |
| Real landscape (lidar) | 67 | 9 | 4 | 54 |
| AD model | 122 | 93 | 0 | 29 |
| Split model, M&D area rule | 81 | 50 | 3 | 28 |
| Split model, M&D area × slope² rule | 65 | 24 | 2 | 39 |
| Split model, area × slope² rule fitted | 102 | 43 | 2 | 57 |
| Split model, area rule fitted | 132 | 64 | 1 | 67 |
| Split model, nothing fitted | 62 | 26 | 3 | 33 |
| Split model, nothing fitted, physical D | 1,215 | 1,213 | 0 | 2 |

The real landscape's extra heads are mostly "other": heads the method puts in hollows M&D did not map as channels, or one cell off a
mapped head (54 of 67). The AD model's are mostly on walls (93 of 122; 54 of 66 on TTV2 and 39 of 45 on TTV3). The wall class is not
specific to AD: on TTV2 the M&D area rule puts 41 of its 52 extra heads on too-steep walls and the area-rule fit 46 of 61, both along the
same trunk valley, where every incising model steepens the valley sides above the real landscape's 95th percentile. Tile edges contribute
0–4 heads per model.

**The maps (figure 6, one page per tile).** On TTV2 the AD panel shows the trunk valley as a dark gorge with its heads strung along both
walls; the M&D area-rule panel shows a cluster of open circles on the same valley's north wall; the nothing-fitted panel has the fewest
open circles and its matched heads sit on the hollows M&D mapped. On TTV1 every model misses the north-east cluster of mapped heads; the
real landscape's own heads there are one or two cells from the picks. TTV3 has only 5 mapped heads inside the polygon (the south-west
third of the tile); the AD panel's 45 extra heads cover the incised ground outside and inside the polygon alike.

## 6. The AD check (figure 7)

The 66 extra AD heads on TTV2 are the two walls of a gorge that the AD surface cuts along the trunk valley running north to south through
the east-centre of the tile. 54 of them sit on cells whose 12 m slope on the AD surface is 1.39 at the median, where the real landscape's
slope at the same cells is 0.21 (its 95th percentile inside the polygon is 0.62); the AD surface there is 18.6 m below the lidar at the
median, and the median drainage area at those cells is 432 m² (288 m² on the real routing), so the area × slope² rule is met by slope
alone: 432 m² × 1.39² = 835 m², at the threshold. The principal axis of the 54 wall heads is 896 m long with a 149 m root-mean-square
scatter across it, the two walls. The transect drawn across that axis through their centroid shows the real valley as a 25 m deep,
200 m wide trough; the AD surface crosses it as a sharp crest about 15 m above the lidar and then a trough about 30 m below it. This is what the AD fit's K and D mean physically: with D 200 × below Heimsath's and K 4.8 × the measured value
the surface is advection-dominated, incises the trunk as a slot and leaves the hillslopes unsmoothed, and the head-finding rule then
reads every cell on the slot's walls as a channel head. The addendum 21 metrics could not see this because a head on the gorge wall is
within a few cells of a channel cell.

## 7. What the evidence supports, and where it is weak

**Supported.**
1. **Inside the ground M&D mapped, the split models with M&D's own rules fit better than the AD model on every tile** (0.33–0.68 ×), with
   the measured K (0.85–1.04 ×) and a D within 0.69–1.30 × the relief D; the AD model needs K 2.2–10 × and D 1.4–200 × below Heimsath's.
   This is the addendum 21 result restated on the right ground, and the polygon restriction changed the split models' parameters by under
   10 % on the two tiles where the polygon is most of the tile.
2. **Fitting the threshold improves the misfit but does not recover M&D's rules or heads.** The fitted thresholds are 0.6–10 × M&D's values,
   D leaves the relief range on two of three tiles, and the heads are worse than with M&D's rules fixed (0.5 × the real landscape's share
   of real heads against 0.8–0.9 ×).
3. **The one-to-one metrics separate the models in the order the planner expected**, with the nothing-fitted and M&D-rule models at the
   method's ceiling, the threshold-fitted models below it, and the AD model at the bottom; the AD surface's string of heads is a gorge
   wall, as read in advance.
4. **The addendum 21 verdict that the real landscape and the nothing-fitted model "locate the mapped heads" while the AD and threshold-
   fitted models do not** survives the change of metric (Appendix B). What does not survive is any claim that a model finds the heads:
   under the pre-registered reading none does, and nor does the lidar.

**Weak.**
1. **The head-finding method, not the models, sets the numbers.** The real landscape pairs 19 of 61 mapped heads within 25 m and 34 within
   50 m. A 12 m block mean, D8 routing and a single threshold put a head in the right hollow but one or two cells from a 1 m pick, and
   the lattice phase alone moves the field thresholds by ± 15 % (addendum 21). Every share in section 5 should be read as a fraction of
   the real landscape's, and the pre-registered reading (0.5 on both shares) cannot be met by anything.
2. **The offset metric is degenerate at 25 m.** Pairs within two cells are "same or adjacent cell" by the adjacency rule and have offset
   0; 71–93 % of pairs are such. The metric would need a larger tolerance or a 1 m routing to say anything.
3. **TTV3 carries 5 mapped heads inside the polygon** (43 % of the tile), so its per-tile head shares are 0, 0.2, 0.4 or 0.6; the pooled
   numbers are TTV2 (34) and TTV1 (22). On TTV3 the area-rule fit is at the D bound and polygon-specific.
4. **The "too-steep wall" class catches every incising model on the TTV2 trunk**, not only AD; the M&D area rule puts 41 of 52 extra
   heads there. The partitioned model with stream power in the trunk also oversteepens the trunk's walls relative to the lidar, by less.
5. **The head rule matters as much as the model** (section 5.5): under an area-only rule the AD model's heads look as good as the split
   models', because the rule no longer sees its walls. The primary rule is justified as the rule M&D's heads were measured with, not as
   a neutral test.
6. **Non-steady state and the single U** remain as in addendum 21: no steady-state D is "the" physical D here, and the relief D is a
   calibrated, not a measured, quantity.

## 8. Decisions needed

1. **The head reading.** The real landscape fails the reading as written. Options: (a) re-base it on the real landscape's own shares (a
   model reproduces the mapped heads if both its shares are at least a stated fraction, say 0.8, of the real landscape's at 25 m); (b)
   replace the head-finding method by one that resolves the picks (the 1 m D8 routing with the 12 m rule, or M&D's own source-area
   measurement); (c) keep the reading and report that nothing passes. I recommend (a) for this addendum's purpose, because it keeps the
   pre-registered metric and the method's ceiling is now measured; under it the nothing-fitted model and the two M&D-rule models pass
   (0.79–1.11 and 0.82–1.15) and the others fail.
2. **Which M&D rule the paper carries for TTV.** On the polygon the area rule fits better on TTV2 and TTV3 (0.33 / 0.36 against 0.51 /
   0.43 × AD), its D sits within the relief range on all three tiles (0.69–1.02), and it finds as many mapped heads as the real landscape
   (0.95 ×) with more extra heads (81 against 65). The area × slope² rule is the rule the field thresholds and the head finding use. I
   recommend the area rule for the fits and figures, with the area × slope² rule as the stated sensitivity, since the misfit and D verdicts
   favour it and the head verdicts do not separate the two.
3. **The area-rule fit on TTV3** (D at the box bound, polygon optimum not the whole-tile optimum): report it as bound-limited, or widen
   the D box for this tile. I recommend reporting it as bound-limited; widening the box would move D further from both reference values.
4. **The offset metric:** drop it, or report it at 50 m with the 1 m routing. I recommend dropping it from the paper and keeping the
   matched counts and the two shares.
5. Nothing is merged or tagged. The other eight TTV tiles are untouched by this addendum.

## 9. Appendices

Run directory: `$GROUP_HOME/lemcvt_results/20261124T000138Z_a22_ttv_cleanup_head_metrics/` (tiles/, profiles/, post/). Jobs and logs:
`$SCRATCH/lemcvt_jobs/a22/` (JOBS.txt lists every submission).

### A. Rules and verdicts (directive section 6)

| rule | statement | result |
| --- | --- | --- |
| Head matching | pooled TTV1–TTV3 at 25 m: share found ≥ 0.5, share real ≥ 0.5, median offset within ± 25 m | fails for every model and for the real landscape (shares 0.31 / 0.22); the offset condition holds everywhere (0 m) |
| Head matching, relative to the real landscape | reported, not a rule | nothing fitted 1.11 / 1.15; M&D area rule 0.95 / 0.82; M&D area × slope² rule 0.79 / 0.85; area × slope² fitted 0.74 / 0.55; area fitted 0.79 / 0.46; AD 0.42 / 0.28; physical D 2.05 / 0.14 |
| Tolerance | any verdict change between 15 and 40 m | none |
| Misfit | the M&D area × slope² rule beats AD on all three tiles | holds (0.65 / 0.51 / 0.43); also holds for the M&D area rule, both fitted-threshold models; fails for nothing fitted (1.25 on TTV1) |
| K physical | K ÷ K₀ within 0.5–2 on all three | holds for all four fitted split models; fails for AD |
| D explained by relief | D ÷ relief D within 0.67–1.5 on all three | holds for the M&D area rule and the M&D area × slope² rule; fails for the two fitted-threshold models and AD |
| The AD check | no rule | gorge wall (section 6) |

### B. The addendum 21 head verdicts, unchanged, beside the new reading

Addendum 21 rule H1 (pooled TTV1–TTV3, α = 2): median distance from each pick to the nearest model channel cell ≤ 25 m and heads ÷ picks in
0.67–1.5.

| model | addendum 21 median pick-to-channel distance (m) | heads ÷ picks | H1 (addendum 21) | this addendum: matched / extra at 25 m | reading (this addendum) |
| --- | --- | --- | --- | --- | --- |
| Real landscape (lidar) | 21.8 | 1.41 (86/61) | holds | 19 / 67 | fails |
| AD model | 41.2 | 2.28 (139/61) | fails | 8 / 122 | fails |
| Split model, M&D area rule | 19.0 | 1.64 (100/61) | fails (count) | 18 / 81 | fails |
| Split model, M&D area × slope² rule | 27.5 | 1.36 (83/61) | fails (distance) | 15 / 65 | fails |
| Split model, area × slope² rule fitted | 44.1 | 1.80 (110/61) | fails | 14 / 102 | fails |
| Split model, area rule fitted | 46.9 | 2.41 (147/61) | fails | 15 / 132 | fails |
| Split model, nothing fitted | 22.6 | 1.36 (83/61) | holds | 21 / 62 | fails |
| Split model, nothing fitted, physical D | 0.0 | 20.56 (1254/61) | fails | 39 / 1,215 | fails |

Addendum 21's H1 and H2 verdicts (H1: the observed surface and F2-relief locate the mapped heads, PF-C and PF-A each miss one condition,
SJ-C, P4 and AD miss both; H2 fails) are superseded, not revised. The physical-D row shows the defect the directive named: a 0 m
one-way distance with 20 × too many heads.

### C. Whole-tile misfits beside the polygon misfits

Polygon misfit of the polygon fit; whole-tile misfit at the polygon fit's parameters; polygon misfit of the addendum 21 whole-tile fit;
whole-tile misfit of the addendum 21 fit (m²); then the two fits' K ÷ K₀, D and threshold.

| tile | model | polygon fit: polygon | polygon fit: whole tile | a21 fit: polygon | a21 fit: whole tile | a21 K ÷ K₀ / polygon-fit K ÷ K₀ | a21 D / polygon-fit D | a21 / polygon-fit threshold |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TTV1 | AD model | 224.0 | 219.9 | 230.9 | 214.2 | 9.23 / 10.05 | 0.00135 / 0.00175 | – |
| TTV1 | Split model, M&D area rule | 151.5 | 118.7 | 152.4 | 117.6 | 0.86 / 0.85 | 0.0165 / 0.0178 | – |
| TTV1 | Split model, M&D area × slope² rule | 146.7 | 117.0 | 148.2 | 115.5 | 0.85 / 0.85 | 0.0187 / 0.0198 | – |
| TTV1 | Split model, area × slope² rule fitted | 101.9 | 93.5 | 103.8 | 84.4 | 0.90 / 0.91 | 0.0270 / 0.0315 | 3,706 / 4,607 |
| TTV1 | Split model, area rule fitted | 85.2 | 83.4 | 85.5 | 82.7 | 1.10 / 1.09 | 0.0370 / 0.0376 | 59,499 / 59,849 |
| TTV1 | Split model, nothing fitted | 280.9 | 230.8 | 280.9 | 230.8 | 1 / 1 | 0.0175 | – |
| TTV2 | AD model | 206.7 | 209.7 | 206.8 | 209.7 | 4.82 / 4.84 | 0.000200 / 0.000201 | – |
| TTV2 | Split model, M&D area rule | 68.2 | 69.6 | 68.3 | 69.5 | 1.02 / 1.02 | 0.0115 / 0.0117 | – |
| TTV2 | Split model, M&D area × slope² rule | 104.7 | 101.5 | 104.9 | 101.3 | 0.99 / 0.99 | 0.0125 / 0.0128 | – |
| TTV2 | Split model, area × slope² rule fitted | 89.0 | 87.4 | 89.1 | 87.3 | 0.98 / 0.98 | 0.0104 / 0.0105 | 559 / 559 |
| TTV2 | Split model, area rule fitted | 61.1 | 59.9 | 61.3 | 59.8 | 1.04 / 1.03 | 0.0140 / 0.0137 | 9,131 / 8,536 |
| TTV2 | Split model, nothing fitted | 125.7 | 124.0 | 125.7 | 124.0 | 1 / 1 | 0.0159 | – |
| TTV3 | AD model | 252.5 | 334.1 | 260.0 | 326.8 | 1.96 / 2.21 | 0.0000078 / 0.0000115 | – |
| TTV3 | Split model, M&D area rule | 91.6 | 157.5 | 98.3 | 152.5 | 0.97 / 1.04 | 0.0157 / 0.0160 | – |
| TTV3 | Split model, M&D area × slope² rule | 107.4 | 173.8 | 122.3 | 161.1 | 0.89 / 0.96 | 0.0250 / 0.0298 | – |
| TTV3 | Split model, area × slope² rule fitted | 70.8 | 147.7 | 82.7 | 138.4 | 0.92 / 0.99 | 0.0310 / 0.0372 | 1,482 / 1,553 |
| TTV3 | Split model, area rule fitted | 34.6 | 171.9 | 66.6 | 87.4 | 1.42 / 1.24 | 0.0422 / 0.0422 (bound) | 89,124 / 35,026 |
| TTV3 | Split model, nothing fitted | 110.9 | 169.2 | 110.9 | 169.2 | 1 / 1 | 0.0230 | – |

On TTV3 the polygon (the south-west 43 % of the tile) is easier ground for every model than the rest of the tile: the whole-tile misfit
of each polygon fit exceeds the polygon misfit by 1.3–2.1 × (5 × for the area-rule fit), while on TTV1 the polygon is the harder ground
for the split models (polygon misfit 1.2–1.3 × the whole-tile one at the same parameters; 1.0–1.1 × for AD and the two fitted-threshold
models).

### D. Mask checks (mask = every cell must reproduce the addendum 21 fit; tolerance relative 10⁻⁸)

| tile | check | addendum 21 misfit | all-cells misfit here | relative difference | pass |
| --- | --- | --- | --- | --- | --- |
| TTV1 | M&D area rule, free K, D | 117.557728 | 117.557728 | 0 | yes |
| TTV1 | M&D area × slope² rule, free K, D | 115.465599 | 115.465599 | 0 | yes |
| TTV1 | nothing fitted / physical D | 230.821411 / 10402.633302 | same | 0 / 0 | yes |
| TTV1 | area rule at the a21 threshold, a21 K, D | 82.734993 | 82.734993 | 1.4 × 10⁻¹⁴ | yes |
| TTV1 | AD at the a21 K, D | 214.189904 | 214.189904 | 0 | yes |
| TTV1 | AD profile, all cells (optimum) | 214.189904 | 214.189904 | 0; minimum not below a21 | yes |
| TTV1 | area × slope² profile, all cells (at the a21 threshold) | 84.441695 | 84.441695 | 0; minimum not below a21 | yes |
| TTV1 | area profile, all cells (at the a21 threshold) | 82.734993 | 82.734993 | 0; minimum not below a21 | yes |
| TTV2 | M&D area rule / M&D area × slope² rule | 69.537859 / 101.315331 | same | 0 / 0 | yes |
| TTV2 | nothing fitted / physical D | 123.963182 / 6197.127771 | same | 0 / 0 | yes |
| TTV2 | area rule at the a21 threshold / AD at the a21 K, D | 59.790823 / 209.660359 | same | 0 / 2.9 × 10⁻¹⁵ | yes |
| TTV2 | AD, area × slope², area profiles (all cells) | 209.660359 / 87.310752 / 59.790823 | same | 0 / 0 / 0; no minimum below a21 | yes |
| TTV3 | M&D area rule / M&D area × slope² rule | 152.494878 / 161.065230 | same | 0 / 0 | yes |
| TTV3 | nothing fitted / physical D | 169.167007 / 8446.264508 | same | 0 / 0 | yes |
| TTV3 | area rule at the a21 threshold / AD at the a21 K, D | 87.396516 / 326.849525 | same | 1.0 × 10⁻¹⁵ / 1.4 × 10⁻¹⁵ | yes |
| TTV3 | AD, area × slope², area profiles (all cells) | 326.849525 / 138.426619 / 87.396516 | same | 0 / 0 / 0; no minimum below a21 | yes |

The profile checks are exact because the addendum 21 optimum threshold was added to every grid; the grids themselves are the addendum 18
and 21 grids, and the all-cells minimum over the whole profile is never below the addendum 21 value (the search found no better point).

### E. Run ids, inputs and checksums

- Tile inputs: `$LEMCVT_DATA/TTV-x/<tile>/<tile>_1m_{filled,area,fd}.tif` (sha256 of the filled grids: TTV1 6f7665fb…, TTV2 95737de4…,
  TTV3 53fdd1b7…; full digests in `tiles/<tile>.json` under `inputs`).
- Steve's heads: `data/ttv/ttv_heads_manual_steve_20261119.csv`, sha256 ac87de3d4ca1f479e6eb2ceda2fedb033740b3d8f37186fd267329d5e4e702b6
  (checked in code before use; 64 picks, 61 on TTV1–TTV3, all inside the polygon).
- M&D georeference: `$GROUP_HOME/lemcvt_results/20261118T160326Z_a15_stage_b/md_georef/md1989_fig3.gpkg`, layers `study_area` and `channels`.
- Addendum 21 run (networks, seeds, fits, D_relief): `$GROUP_HOME/lemcvt_results/20261123T184712Z_a21_ttv_channel_head_test/`.
- Addendum 22 run: `$GROUP_HOME/lemcvt_results/20261124T000138Z_a22_ttv_cleanup_head_metrics/`: `tiles/<tile>.json` and `_products.npz`
  (polygon, mask checks, fixed-rule fits, 12 m products of the real landscape and the four fixed-rule surfaces); `profiles/<model>_<tile>.json`,
  `_profile.parquet`, `_products.npz` (AD, area × slope² fitted, area fitted; both objectives on every profile point); `post/metrics.json`,
  `rules.json`, `ad_check.json`, `tables.md`, `captions.json`.
- Git: scripts and figures at this commit; the run was made at dc5ed7b with no package code change (`git=dc5ed7b` in every job log).

### F. Slurm accounting

| job name | tasks | core-h used | core-h of limits | states | job ids |
| --- | --- | --- | --- | --- | --- |
| a22_tests | 1 | 0.15 | 2.0 | COMPLETED | 53477999 |
| a22_tile | 3 | 0.06 | 2.0 | COMPLETED 3 | 53478797 (TTV2 test), 53479674 (TTV1, TTV3) |
| a22_pilot | 3 | 0.12 | 1.5 | COMPLETED 3 | 53478799 |
| a22_profile | 9 | 17.06 | 42.0 | COMPLETED 9 | 53479678 (AD), 53479682 (area × slope²), 53479686 (area); array tasks carry their own raw job ids in `profiles/job.json` |
| a22_post | 5 | 0.09 | 3.3 | FAILED 2, COMPLETED 3 | 53486215, 53487424 (debug runs before the profiles were in: missing metrics file, then a key error on the absent AD profile), 53488885, 53494867 (before the area profiles finished), 53559772 (final, all profiles in) |

Core-hours used: **17.5** (directive estimate ≤ 30; the pilot estimate printed before submission was ~14: AD 1.6 h, area × slope² 1.0 h
and area 2.0 h per tile, the area profiles then took 3.0–3.4 h each because the time per evaluation rises from 2.4 s at small thresholds to
6 s at large ones). CPU-weighted limits: 50.8; largest single submission 24 core-hours of limits (the area profile array); peak MaxRSS 3.6
GB; 12 submissions. Wall time per profile: AD 1.7–1.8 h (213–218 factorisations at 28 s), area × slope² 0.7–0.8 h (401–403 evaluations),
area 3.0–3.4 h (2,011–2,057 evaluations). The two figure regenerations after the final post run (layout fixes only, 10 s each) ran on the
login node.

### G. State check, decisions as applied, deviations

**At the start (2026-11-23 16:05 PST, sh02-ln01).** `v2.3-tiles` = `origin` = dc5ed7b (the addendum 21 report and its hillshade fix); no job
running; `/etc/agents/AGENTS.md` and its Slurm topic file read. Tests on `serc` at dc5ed7b (53477999): ruff and mypy clean, pytest exit 0,
suite exit 0. The addendum 21 run directory, the M&D georeference (`md1989_fig3.gpkg`: layers `study_area`, `channels`, `channel_heads`,
`colluvial_deposits`, `ground_control`) and Steve's heads CSV (checksum checked in code before use) are where the directive says. **No
package code was changed.** The directive and the repo did not disagree anywhere that required a stop.

**Hand-over.** The compute and three post-processing runs were made in the session of 2026-11-23 16:00–20:00 PST; the final
post-processing run, the figure layout fixes, this report and the push were made in a second session on 2026-11-24 from 08:00 PST
(sh03-ln06), which re-read `/etc/agents/AGENTS.md` and the Slurm topic file first. The earlier session's draft numbers were checked
against the final tables; two draft statements were wrong and are corrected here (the polygon shares, 63 / 86 / 43 %, not 86–96 %; the
AD model's D moves 30–48 % between the whole-tile and polygon fits, not under 20 %).

**Decisions as applied.** Every choice the directive left open is listed here (the same list heads `a22_common.py`).

1. **The polygon misfit (2.1).** The partitioned reduction is Z = a + (U/K) b + (U/D) c cell by cell, so the misfit over any cell set is the
   Gram sum over that set: `lemcvt.analyses.sensitivity._terms_masked(red, Z_obs, keep)` with `keep` the polygon mask on the profiler's
   cells (the function already exists; it is how the addendum 18 basin halves were scored). For the AD model the observed elevation is set
   to NaN outside the polygon before `ADProfiler.best_y`, whose cell mask is `isfinite(obs)`; the LU factorisation at each r does not
   depend on the observation, so the whole-tile and the polygon objectives come from the same factorisation. No mask argument needed
   adding; no tests were added. **The check:** with the mask set to every cell, every fixed-rule model, the area-rule model at its a21
   threshold and AD at its a21 K, D reproduce the addendum 21 misfits to a relative 0 to 1.4 × 10⁻¹⁴ on every tile (Appendix D), and the
   AD, area × slope² and area profiles over all cells reproduce the addendum 21 optima exactly (the addendum 21 optimum threshold was
   added to every grid; the grids themselves are the addendum 18 / 21 grids).
2. **The polygon on the lattices.** The M&D study polygon is rasterised on each tile's 1 m lattice as in addenda 19–21
   (`md_polygon_mask`, cell centres). A 12 m head or channel cell is inside the polygon when its centre cell (6, 6) is; a mapped head is
   inside when its 1 m cell is. The polygon holds 63 / 86 / 43 % of TTV1 / TTV2 / TTV3 (section 4).
3. **Head finding (2.3)** is the addendum 21 extraction unchanged, on each surface's own 12 m tile lattice (83 × 83 full blocks, the 4 m
   remainder dropped; pit-filled; D8 with outlets at the border; S₁₂ to the largest-area donor; a head where the upstream maximum of
   A₁₂ S₁₂² first reaches 876 m²). The 12 m statistics are stored per surface, so the area-only rule (6,120 m²) and the 16-placement
   medians (817 and 6,228 m²) are re-applications of a threshold, not reruns. The Perron valley method is dropped as the directive says.
4. **One-to-one matching (2.4.1)** is the Hungarian assignment on the rectangular mapped × model distance matrix with every pair farther
   than t given a cost of 10⁹, so the assignment first maximises the number of pairs within t and then minimises their total distance;
   pairs farther than t are dropped. Distances are between the pick's 1 m cell and the centre of the model head's 12 m cell.
5. **Along-valley offsets (2.4.2)** are traced on the real landscape's 12 m D8 routing (the same routing for every model). The mapped head's
   cell is the 12 m block holding the pick; "on or within one cell of the path" is 8-neighbour adjacency; the offset is the D8 path
   distance (12 or 12√2 m per step) from the start cell to the first path cell at or adjacent to the other head. The downstream trace from
   the mapped head is tried first (model channel too short, offset > 0), then the trace from the model head (too long, offset < 0); a pair
   whose two heads fall in the same or in adjacent 12 m cells has offset 0 ("same cell"); otherwise the pair is in a neighbouring hollow
   and has no offset. **Consequence (found after the fact):** at t = 25 m almost every pair is within two cells, so the metric is
   degenerate (section 5.3).
6. **Channel length (2.4.3):** every 12 m channel cell whose centre is inside the polygon contributes the distance to its D8 receiver (12 or
   12√2 m; nothing where it has none). M&D's traced length is the `channels` layer of the georeference clipped to the tile rectangle
   intersected with the study polygon (OGR geometry intersection), so model and map are measured over the same ground.
7. **Extra-head classes (2.4.4):** the local slope is the gradient magnitude of the pit-filled 12 m surface; "steeper than the steepest 5 %
   of the real landscape" is the 95th percentile of the real landscape's 12 m slope inside the polygon on the same tile; "tile edge" is a
   12 m cell whose centre lies within 24 m of the tile boundary (the two outer rings). A steep head on the edge counts as a wall.
8. **The profiles (3.1).** The area × slope² profile reuses the addendum 21 C range and seeds (thin 400, refine ≤ 400); the area profile
   uses the addendum 18 grid (the unique drainage areas in 10²–10⁵ m² thinned to 2,000, staircase refinement around the minima ≤ 1,000;
   the 1 % fill step of addendum 18, which only serves the identifiability intervals, is not run); AD uses the addendum 18 grid (200
   log-uniform r between the bounds' ratio limits, then bounded Brent in log₁₀ r), for both objectives. The refinement is done around
   both the polygon and the whole-tile minima.
9. **Parameter values (2.1)** unchanged from addendum 21: U = 6.5 × 10⁻⁵ m/yr, m = 0.33, K₀ = U/k_s per tile, D_phys = 0.0025 m²/yr, D_relief
   per tile on the mapped-head network (TTV1 0.0175, TTV2 0.0159, TTV3 0.0230 m²/yr), the addendum 18 box in (U/K, U/D) for the bounds.
   Thresholds 6,120 m² and 876.27 m² (the directive's 876).
10. **The AD check (2.5).** The AD surface used is the polygon refit (not the addendum 21 whole-tile fit); the "line" is the principal axis
    (singular-value decomposition) of the extra AD heads that fall on too-steep walls inside the polygon, and the profile is drawn through
    their centroid perpendicular to that axis, ± 240 m.
11. **Figures (section 5).** All eight figures follow the directive's rules: plain names, spelled-out quantities, linear axes except the two
    stated log axes in figure 8 (values at the ticks, "2 means twice" in the label), the good direction marked, reference values labelled
    in words, values on bars, one colour and marker per model, at most six models per panel (the two nothing-fitted models are in a second
    panel), the map conventions of 5.1.10, two-line captions under every figure. The hillshades are lit from the north-west with the
    corrected helper of dc5ed7b.
12. **Cost.** The directive's estimate (≤ 30 core-hours) was printed before submission from the TTV2 pilots; actual use is in Appendix F.

### H. Machine-generated tables (`post/tables.md`, verbatim)

#### Model key

| Plain name | What it is | Fitted parameters |
|---|---|---|
| **Real landscape (lidar)** | The observed surface, run through the same head-finding method as the models | none |
| **AD model** | Co-located advection–diffusion; incision and diffusion act in every cell | K, D |
| **Split model, M&D area rule** | Stream power where drainage area ≥ 6,120 m² (measured at M&D's heads), diffusion elsewhere | K, D |
| **Split model, M&D area × slope² rule** | Stream power where area × slope² ≥ 876 m² (measured at M&D's heads), diffusion elsewhere | K, D |
| **Split model, area × slope² rule fitted** | As above, but the threshold is chosen by the fit | K, D, threshold |
| **Split model, area rule fitted** | Area threshold chosen by the fit (the original tuned model) | K, D, threshold |
| **Split model, nothing fitted** | M&D area × slope² rule, K = measured K₀, D = relief D | none |
| **Split model, nothing fitted, physical D** | As above, but D = Heimsath's 0.0025 m²/yr | none |

#### Polygon cells per tile

| tile | 1 m cells in the M&D polygon | valid 1 m cells | share | profiler cells in polygon | 12 m cells (centre inside) | mapped heads inside | M&D traced length inside (m) |
|---|---|---|---|---|---|---|---|
| TTV1 | 630,864 | 1,000,000 | 0.63 | 630,864 | 4,352 | 22 | 11,153 |
| TTV2 | 855,216 | 1,000,000 | 0.86 | 855,216 | 5,910 | 34 | 13,542 |
| TTV3 | 426,566 | 1,000,000 | 0.43 | 426,566 | 2,956 | 5 | 4,678 |

#### Section 4 table: one row per model and tile (misfit inside the M&D area; t = 25 m)

| model | tile | misfit in M&D area (m²) | ÷ AD | whole-tile misfit at these parameters (m²) | K ÷ K₀ | D (m²/yr) | D ÷ physical D | D ÷ relief D | fitted threshold (M&D value) | matched | missed | extra | share found | share real | median offset (m) | channel length ÷ mapped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real landscape (lidar) | TTV1 | 0.00 | – | 0.00 | – | – | – | – | – | 8 | 14 | 35 | 0.36 | 0.19 | 0 | 0.41 |
| Real landscape (lidar) | TTV2 | 0.00 | – | 0.00 | – | – | – | – | – | 9 | 25 | 28 | 0.26 | 0.24 | 0 | 0.43 |
| Real landscape (lidar) | TTV3 | 0.00 | – | 0.00 | – | – | – | – | – | 2 | 3 | 4 | 0.40 | 0.33 | 0 | 0.35 |
| AD model | TTV1 | 224.00 | 1.00 | 219.94 | 10.05 | 1.75e-03 | 0.7 | 0.10 | – | 3 | 19 | 11 | 0.14 | 0.21 | 0 | 0.26 |
| AD model | TTV2 | 206.73 | 1.00 | 209.73 | 4.84 | 2.01e-04 | 0.1 | 0.01 | – | 5 | 29 | 66 | 0.15 | 0.07 | 0 | 0.55 |
| AD model | TTV3 | 252.46 | 1.00 | 334.07 | 2.21 | 1.15e-05 | 4.62e-03 | 5.01e-04 | – | 0 | 5 | 45 | 0.00 | 0.00 | – | 0.97 |
| Split model, M&D area rule | TTV1 | 151.50 | 0.68 | 118.75 | 0.85 | 0.0178 | 7.1 | 1.02 | 6,120 m² area (fixed) | 3 | 19 | 20 | 0.14 | 0.13 | 0 | 0.32 |
| Split model, M&D area rule | TTV2 | 68.18 | 0.33 | 69.63 | 1.02 | 0.0117 | 4.7 | 0.74 | 6,120 m² area (fixed) | 14 | 20 | 52 | 0.41 | 0.21 | 0 | 0.47 |
| Split model, M&D area rule | TTV3 | 91.63 | 0.36 | 157.52 | 1.04 | 0.0160 | 6.4 | 0.69 | 6,120 m² area (fixed) | 1 | 4 | 9 | 0.20 | 0.10 | 0 | 0.45 |
| Split model, M&D area × slope² rule | TTV1 | 146.70 | 0.65 | 117.00 | 0.85 | 0.0198 | 7.9 | 1.13 | 876 m² area × slope² (fixed) | 3 | 19 | 21 | 0.14 | 0.12 | 0 | 0.32 |
| Split model, M&D area × slope² rule | TTV2 | 104.74 | 0.51 | 101.48 | 0.99 | 0.0128 | 5.1 | 0.80 | 876 m² area × slope² (fixed) | 10 | 24 | 40 | 0.29 | 0.20 | 0 | 0.42 |
| Split model, M&D area × slope² rule | TTV3 | 107.38 | 0.43 | 173.83 | 0.96 | 0.0298 | 11.9 | 1.30 | 876 m² area × slope² (fixed) | 2 | 3 | 4 | 0.40 | 0.33 | -6 | 0.29 |
| Split model, area × slope² rule fitted | TTV1 | 101.90 | 0.45 | 93.53 | 0.91 | 0.0315 | 12.6 | 1.80 | 4,607 m² area × slope² (876) | 3 | 19 | 52 | 0.14 | 0.05 | 12 | 0.37 |
| Split model, area × slope² rule fitted | TTV2 | 88.95 | 0.43 | 87.40 | 0.98 | 0.0105 | 4.2 | 0.66 | 559 m² area × slope² (876) | 11 | 23 | 42 | 0.32 | 0.21 | 0 | 0.45 |
| Split model, area × slope² rule fitted | TTV3 | 70.84 | 0.28 | 147.66 | 0.99 | 0.0372 | 14.9 | 1.62 | 1,553 m² area × slope² (876) | 0 | 5 | 8 | 0.00 | 0.00 | – | 0.31 |
| Split model, area rule fitted | TTV1 | 85.17 | 0.38 | 83.42 | 1.09 | 0.0376 | 15.0 | 2.15 | 59,849 m² area (6,120) | 1 | 21 | 65 | 0.05 | 0.02 | 0 | 0.45 |
| Split model, area rule fitted | TTV2 | 61.08 | 0.30 | 59.94 | 1.03 | 0.0137 | 5.5 | 0.86 | 8,536 m² area (6,120) | 14 | 20 | 61 | 0.41 | 0.19 | 0 | 0.47 |
| Split model, area rule fitted | TTV3 | 34.60 | 0.14 | 171.91 | 1.24 | 0.0422 | 16.9 | 1.83 | 35,026 m² area (6,120) | 0 | 5 | 6 | 0.00 | 0.00 | – | 0.23 |
| Split model, nothing fitted | TTV1 | 280.91 | 1.25 | 230.82 | 1.00 | 0.0175 | 7.0 | 1.00 | 876 m² area × slope² (fixed) | 7 | 15 | 27 | 0.32 | 0.21 | 0 | 0.34 |
| Split model, nothing fitted | TTV2 | 125.66 | 0.61 | 123.96 | 1.00 | 0.0159 | 6.4 | 1.00 | 876 m² area × slope² (fixed) | 11 | 23 | 29 | 0.32 | 0.28 | 0 | 0.39 |
| Split model, nothing fitted | TTV3 | 110.88 | 0.44 | 169.17 | 1.00 | 0.0230 | 9.2 | 1.00 | 876 m² area × slope² (fixed) | 3 | 2 | 6 | 0.60 | 0.33 | 0 | 0.37 |
| Split model, nothing fitted, physical D | TTV1 | 9,149 | 40.84 | 10,403 | 1.00 | 2.50e-03 | 1.0 | 0.14 | 876 m² area × slope² (fixed) | 11 | 11 | 411 | 0.50 | 0.03 | 0 | 2.17 |
| Split model, nothing fitted, physical D | TTV2 | 6,838 | 33.08 | 6,197 | 1.00 | 2.50e-03 | 1.0 | 0.16 | 876 m² area × slope² (fixed) | 25 | 9 | 553 | 0.74 | 0.04 | 0 | 2.27 |
| Split model, nothing fitted, physical D | TTV3 | 7,709 | 30.54 | 8,446 | 1.00 | 2.50e-03 | 1.0 | 0.11 | 876 m² area × slope² (fixed) | 3 | 2 | 251 | 0.60 | 0.01 | 0 | 2.92 |

#### Pooled over TTV1–TTV3 at t = 25 m (section 6 head reading)

| model | mapped | matched | missed | extra | share found | ÷ real landscape | share real | ÷ real landscape | median offset (m) | q25–q75 | too short | too long | neighbouring hollow | same cell | channel length (m) | ÷ mapped | ÷ real landscape | extra: too-steep wall | tile edge | other | reproduces the mapped heads |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Real landscape (lidar) | 61 | 19 | 42 | 67 | 0.31 | 1.00 | 0.22 | 1.00 | 0 | 0–0 | 0.05 | 0.00 | 0.05 | 0.89 | 11,991 | 0.41 | 1.00 | 9 | 4 | 54 | no |
| AD model | 61 | 8 | 53 | 122 | 0.13 | 0.42 | 0.06 | 0.28 | 0 | -3–0 | 0.00 | 0.25 | 0.00 | 0.75 | 14,864 | 0.51 | 1.24 | 93 | 0 | 29 | no |
| Split model, M&D area rule | 61 | 18 | 43 | 81 | 0.30 | 0.95 | 0.18 | 0.82 | 0 | 0–0 | 0.11 | 0.06 | 0.00 | 0.83 | 12,116 | 0.41 | 1.01 | 50 | 3 | 28 | no |
| Split model, M&D area × slope² rule | 61 | 15 | 46 | 65 | 0.25 | 0.79 | 0.19 | 0.85 | 0 | 0–0 | 0.00 | 0.07 | 0.00 | 0.93 | 10,638 | 0.36 | 0.89 | 24 | 2 | 39 | no |
| Split model, area × slope² rule fitted | 61 | 14 | 47 | 102 | 0.23 | 0.74 | 0.12 | 0.55 | 0 | 0–9 | 0.29 | 0.00 | 0.00 | 0.71 | 11,707 | 0.40 | 0.98 | 43 | 2 | 57 | no |
| Split model, area rule fitted | 61 | 15 | 46 | 132 | 0.25 | 0.79 | 0.10 | 0.46 | 0 | 0–0 | 0.20 | 0.00 | 0.00 | 0.80 | 12,473 | 0.42 | 1.04 | 64 | 1 | 67 | no |
| Split model, nothing fitted | 61 | 21 | 40 | 62 | 0.34 | 1.11 | 0.25 | 1.15 | 0 | 0–0 | 0.10 | 0.05 | 0.00 | 0.86 | 10,772 | 0.37 | 0.90 | 26 | 3 | 33 | no |
| Split model, nothing fitted, physical D | 61 | 39 | 22 | 1215 | 0.64 | 2.05 | 0.03 | 0.14 | 0 | 0–0 | 0.03 | 0.08 | 0.08 | 0.82 | 68,680 | 2.34 | 5.73 | 1213 | 0 | 2 | no |

M&D traced channel length inside the polygon on the three tiles: 29,374 m.


#### Every tolerance (pooled; share found / share real / median offset m / verdict)

| model | t = 10 m | t = 15 m | t = 20 m | t = 25 m | t = 30 m | t = 40 m | t = 50 m | verdict changes between 15 and 40 m |
|---|---|---|---|---|---|---|---|---|
| Real landscape (lidar) | 0.00 / 0.00 / – / no | 0.16 / 0.12 / 0 / no | 0.28 / 0.20 / 0 / no | 0.31 / 0.22 / 0 / no | 0.34 / 0.24 / 0 / no | 0.49 / 0.35 / 0 / no | 0.56 / 0.40 / 0 / no | no |
| AD model | 0.02 / 7.69e-03 / 0 / no | 0.05 / 0.02 / 0 / no | 0.10 / 0.05 / 0 / no | 0.13 / 0.06 / 0 / no | 0.15 / 0.07 / 0 / no | 0.23 / 0.11 / 0 / no | 0.28 / 0.13 / 0 / no | no |
| Split model, M&D area rule | 0.08 / 0.05 / 0 / no | 0.21 / 0.13 / 0 / no | 0.25 / 0.15 / 0 / no | 0.30 / 0.18 / 0 / no | 0.34 / 0.21 / 0 / no | 0.49 / 0.30 / 0 / no | 0.59 / 0.36 / 0 / no | no |
| Split model, M&D area × slope² rule | 0.03 / 0.03 / 0 / no | 0.16 / 0.12 / 0 / no | 0.23 / 0.17 / 0 / no | 0.25 / 0.19 / 0 / no | 0.31 / 0.24 / 0 / no | 0.49 / 0.38 / 0 / no | 0.57 / 0.44 / 0 / no | no |
| Split model, area × slope² rule fitted | 0.05 / 0.03 / 0 / no | 0.13 / 0.07 / 0 / no | 0.16 / 0.09 / 0 / no | 0.23 / 0.12 / 0 / no | 0.30 / 0.16 / 0 / no | 0.33 / 0.17 / 0 / no | 0.43 / 0.22 / 0 / no | no |
| Split model, area rule fitted | 0.08 / 0.03 / 0 / no | 0.16 / 0.07 / 0 / no | 0.20 / 0.08 / 0 / no | 0.25 / 0.10 / 0 / no | 0.31 / 0.13 / 0 / no | 0.36 / 0.15 / 0 / no | 0.43 / 0.18 / 0 / no | no |
| Split model, nothing fitted | 0.07 / 0.05 / 0 / no | 0.26 / 0.19 / 0 / no | 0.30 / 0.22 / 0 / no | 0.34 / 0.25 / 0 / no | 0.44 / 0.33 / 0 / no | 0.51 / 0.37 / 0 / no | 0.57 / 0.42 / 0 / no | no |
| Split model, nothing fitted, physical D | 0.26 / 0.01 / 0 / no | 0.44 / 0.02 / 0 / no | 0.56 / 0.03 / 0 / no | 0.64 / 0.03 / 0 / no | 0.72 / 0.04 / 0 / no | 0.82 / 0.04 / 0 / no | 0.90 / 0.04 / 0 / no | no |

#### Sensitivities of the head finding (pooled, t = 25 m): share found / share real / median offset / extra

| model | area × slope² ≥ 876 m² (primary) | area ≥ 6,120 m² | area × slope² ≥ 817 m² (16-placement median) | area ≥ 6,228 m² (16-placement median) |
|---|---|---|---|---|
| Real landscape (lidar) | 0.31 / 0.22 / 0 / 67 | 0.23 / 0.18 / 0 / 64 | 0.31 / 0.20 / 0 / 76 | 0.18 / 0.14 / 0 / 66 |
| AD model | 0.13 / 0.06 / 0 / 122 | 0.28 / 0.21 / 0 / 63 | 0.15 / 0.06 / 0 / 134 | 0.30 / 0.23 / 0 / 60 |
| Split model, M&D area rule | 0.30 / 0.18 / 0 / 81 | 0.23 / 0.25 / 0 / 43 | 0.33 / 0.19 / 0 / 87 | 0.21 / 0.23 / 0 / 44 |
| Split model, M&D area × slope² rule | 0.25 / 0.19 / 0 / 65 | 0.20 / 0.22 / 0 / 43 | 0.30 / 0.20 / 0 / 71 | 0.20 / 0.22 / 0 / 43 |
| Split model, area × slope² rule fitted | 0.23 / 0.12 / 0 / 102 | 0.21 / 0.19 / 0 / 54 | 0.25 / 0.12 / 0 / 111 | 0.21 / 0.20 / 0 / 52 |
| Split model, area rule fitted | 0.25 / 0.10 / 0 / 132 | 0.05 / 0.06 / 0 / 51 | 0.25 / 0.09 / 0 / 145 | 0.05 / 0.06 / 0 / 49 |
| Split model, nothing fitted | 0.34 / 0.25 / 0 / 62 | 0.21 / 0.29 / 0 / 32 | 0.34 / 0.24 / 0 / 68 | 0.23 / 0.31 / 0 / 31 |
| Split model, nothing fitted, physical D | 0.64 / 0.03 / 0 / 1215 | 0.23 / 0.27 / 0 / 38 | 0.62 / 0.03 / 0 / 1248 | 0.23 / 0.27 / 0 / 38 |

#### Misfit, K and D readings (section 6)

| model | misfit ÷ AD: TTV1 / TTV2 / TTV3 | beats AD on all three | K ÷ K₀: TTV1 / TTV2 / TTV3 | K physical (0.5–2 on all three) | D ÷ relief D: TTV1 / TTV2 / TTV3 | explained by relief (0.67–1.5 on all three) | D ÷ physical D: TTV1 / TTV2 / TTV3 |
|---|---|---|---|---|---|---|---|
| AD model | 1.00 / 1.00 / 1.00 | no | 10.05 / 4.84 / 2.21 | no | 0.10 / 0.01 / 5.01e-04 | no | 0.7 / 0.1 / 4.62e-03 |
| Split model, M&D area rule | 0.68 / 0.33 / 0.36 | yes | 0.85 / 1.02 / 1.04 | yes | 1.02 / 0.74 / 0.69 | yes | 7.1 / 4.7 / 6.4 |
| Split model, M&D area × slope² rule | 0.65 / 0.51 / 0.43 | yes | 0.85 / 0.99 / 0.96 | yes | 1.13 / 0.80 / 1.30 | yes | 7.9 / 5.1 / 11.9 |
| Split model, area × slope² rule fitted | 0.45 / 0.43 / 0.28 | yes | 0.91 / 0.98 / 0.99 | yes | 1.80 / 0.66 / 1.62 | no | 12.6 / 4.2 / 14.9 |
| Split model, area rule fitted | 0.38 / 0.30 / 0.14 | yes | 1.09 / 1.03 / 1.24 | yes | 2.15 / 0.86 / 1.83 | no | 15.0 / 5.5 / 16.9 |
| Split model, nothing fitted | 1.25 / 0.61 / 0.44 | no | 1.00 / 1.00 / 1.00 | – | 1.00 / 1.00 / 1.00 | – | 7.0 / 6.4 / 9.2 |
| Split model, nothing fitted, physical D | 40.84 / 33.08 / 30.54 | no | 1.00 / 1.00 / 1.00 | – | 0.14 / 0.16 / 0.11 | – | 1.0 / 1.0 / 1.0 |

#### Appendix: mask checks (mask = every cell reproduces the addendum 21 fit)

| tile | model | addendum 21 misfit (m²) | all-cells misfit here | relative difference | pass |
|---|---|---|---|---|---|
| TTV1 | PF_A | 117.557728 | 117.557728 | 0.0e+00 | yes |
| TTV1 | PF_C | 115.465599 | 115.465599 | 0.0e+00 | yes |
| TTV1 | F2_relief | 230.821411 | 230.821411 | 0.0e+00 | yes |
| TTV1 | F2_phys | 10402.633302 | 10402.633302 | 0.0e+00 | yes |
| TTV1 | P4_at_a21_A_crit | 82.734993 | 82.734993 | 1.4e-14 | yes |
| TTV1 | AD_at_a21_K_D | 214.189904 | 214.189904 | 0.0e+00 | yes |
| TTV1 | AD (profile) | 214.189904 | 214.189904 | 0.0e+00; profile minimum below a21: no | yes |
| TTV1 | SJ_C (profile) | 84.441695 | 84.441695 | 0.0e+00; at the a21 threshold 0.0e+00; profile minimum below a21: no | yes |
| TTV1 | P4 (profile) | 82.734993 | 82.734993 | 0.0e+00; at the a21 threshold 0.0e+00; profile minimum below a21: no | yes |
| TTV2 | PF_A | 69.537859 | 69.537859 | 0.0e+00 | yes |
| TTV2 | PF_C | 101.315331 | 101.315331 | 0.0e+00 | yes |
| TTV2 | F2_relief | 123.963182 | 123.963182 | 0.0e+00 | yes |
| TTV2 | F2_phys | 6197.127771 | 6197.127771 | 0.0e+00 | yes |
| TTV2 | P4_at_a21_A_crit | 59.790823 | 59.790823 | 0.0e+00 | yes |
| TTV2 | AD_at_a21_K_D | 209.660359 | 209.660359 | 2.9e-15 | yes |
| TTV2 | AD (profile) | 209.660359 | 209.660359 | 0.0e+00; profile minimum below a21: no | yes |
| TTV2 | SJ_C (profile) | 87.310752 | 87.310752 | 0.0e+00; at the a21 threshold 0.0e+00; profile minimum below a21: no | yes |
| TTV2 | P4 (profile) | 59.790823 | 59.790823 | 0.0e+00; at the a21 threshold 0.0e+00; profile minimum below a21: no | yes |
| TTV3 | PF_A | 152.494878 | 152.494878 | 0.0e+00 | yes |
| TTV3 | PF_C | 161.065230 | 161.065230 | 0.0e+00 | yes |
| TTV3 | F2_relief | 169.167007 | 169.167007 | 0.0e+00 | yes |
| TTV3 | F2_phys | 8446.264508 | 8446.264508 | 0.0e+00 | yes |
| TTV3 | P4_at_a21_A_crit | 87.396516 | 87.396516 | 1.0e-15 | yes |
| TTV3 | AD_at_a21_K_D | 326.849525 | 326.849525 | 1.4e-15 | yes |
| TTV3 | AD (profile) | 326.849525 | 326.849525 | 0.0e+00; profile minimum below a21: no | yes |
| TTV3 | SJ_C (profile) | 138.426619 | 138.426619 | 0.0e+00; at the a21 threshold 0.0e+00; profile minimum below a21: no | yes |
| TTV3 | P4 (profile) | 87.396516 | 87.396516 | 0.0e+00; at the a21 threshold 0.0e+00; profile minimum below a21: no | yes |

#### Appendix: whole-tile misfits beside the polygon misfits

| tile | model | polygon misfit of the polygon fit | whole-tile misfit of the polygon fit | polygon misfit of the addendum 21 (whole-tile) fit | whole-tile misfit of the addendum 21 fit | a21 K ÷ K₀ | a21 D | polygon-fit K ÷ K₀ | polygon-fit D | a21 threshold | polygon-fit threshold |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | AD model | 224.00 | 219.94 | 230.90 | 214.19 | 9.23 | 1.35e-03 | 10.05 | 1.75e-03 | – | – |
| TTV1 | Split model, M&D area rule | 151.50 | 118.75 | 152.45 | 117.56 | 0.86 | 0.0165 | 0.85 | 0.0178 | – | – |
| TTV1 | Split model, M&D area × slope² rule | 146.70 | 117.00 | 148.19 | 115.47 | 0.85 | 0.0187 | 0.85 | 0.0198 | – | – |
| TTV1 | Split model, area × slope² rule fitted | 101.90 | 93.53 | 103.84 | 84.44 | 0.90 | 0.0270 | 0.91 | 0.0315 | 3,706 | 4,607 |
| TTV1 | Split model, area rule fitted | 85.17 | 83.42 | 85.47 | 82.73 | 1.10 | 0.0370 | 1.09 | 0.0376 | 59,499 | 59,849 |
| TTV1 | Split model, nothing fitted | 280.91 | 230.82 | 280.91 | 230.82 | 1.00 | 0.0175 | 1.00 | 0.0175 | – | – |
| TTV1 | Split model, nothing fitted, physical D | 9,149 | 10,403 | 9,149 | 10,403 | 1.00 | 2.50e-03 | 1.00 | 2.50e-03 | – | – |
| TTV2 | AD model | 206.73 | 209.73 | 206.81 | 209.66 | 4.82 | 2.00e-04 | 4.84 | 2.01e-04 | – | – |
| TTV2 | Split model, M&D area rule | 68.18 | 69.63 | 68.29 | 69.54 | 1.02 | 0.0115 | 1.02 | 0.0117 | – | – |
| TTV2 | Split model, M&D area × slope² rule | 104.74 | 101.48 | 104.92 | 101.32 | 0.99 | 0.0125 | 0.99 | 0.0128 | – | – |
| TTV2 | Split model, area × slope² rule fitted | 88.95 | 87.40 | 89.05 | 87.31 | 0.98 | 0.0104 | 0.98 | 0.0105 | 559 | 559 |
| TTV2 | Split model, area rule fitted | 61.08 | 59.94 | 61.28 | 59.79 | 1.04 | 0.0140 | 1.03 | 0.0137 | 9,131 | 8,536 |
| TTV2 | Split model, nothing fitted | 125.66 | 123.96 | 125.66 | 123.96 | 1.00 | 0.0159 | 1.00 | 0.0159 | – | – |
| TTV2 | Split model, nothing fitted, physical D | 6,838 | 6,197 | 6,838 | 6,197 | 1.00 | 2.50e-03 | 1.00 | 2.50e-03 | – | – |
| TTV3 | AD model | 252.46 | 334.07 | 260.04 | 326.85 | 1.96 | 7.79e-06 | 2.21 | 1.15e-05 | – | – |
| TTV3 | Split model, M&D area rule | 91.63 | 157.52 | 98.29 | 152.49 | 0.97 | 0.0157 | 1.04 | 0.0160 | – | – |
| TTV3 | Split model, M&D area × slope² rule | 107.38 | 173.83 | 122.32 | 161.07 | 0.89 | 0.0250 | 0.96 | 0.0298 | – | – |
| TTV3 | Split model, area × slope² rule fitted | 70.84 | 147.66 | 82.71 | 138.43 | 0.92 | 0.0310 | 0.99 | 0.0372 | 1,482 | 1,553 |
| TTV3 | Split model, area rule fitted | 34.60 | 171.91 | 66.56 | 87.40 | 1.42 | 0.0422 | 1.24 | 0.0422 | 89,124 | 35,026 |
| TTV3 | Split model, nothing fitted | 110.88 | 169.17 | 110.88 | 169.17 | 1.00 | 0.0230 | 1.00 | 0.0230 | – | – |
| TTV3 | Split model, nothing fitted, physical D | 7,709 | 8,446 | 7,709 | 8,446 | 1.00 | 2.50e-03 | 1.00 | 2.50e-03 | – | – |

#### Appendix: the addendum 21 head verdicts, unchanged (pooled TTV1–TTV3, α = 2; median pick-to-channel distance ≤ 25 m and heads ÷ picks in 0.67–1.5)

| model (plain name) | internal code | median pick-to-channel distance (m) | heads in polygon ÷ picks | addendum 21 H1 holds | addendum 22 reading (t = 25 m) |
|---|---|---|---|---|---|
| Real landscape (lidar) | obs | 21.8 | 1.41 (86/61) | yes | no |
| AD model | AD | 41.2 | 2.28 (139/61) | no | no |
| Split model, M&D area rule | PF_A | 19.0 | 1.64 (100/61) | no | no |
| Split model, M&D area × slope² rule | PF_C | 27.5 | 1.36 (83/61) | no | no |
| Split model, area × slope² rule fitted | SJ_C | 44.1 | 1.80 (110/61) | no | no |
| Split model, area rule fitted | P4 | 46.9 | 2.41 (147/61) | no | no |
| Split model, nothing fitted | F2_relief | 22.6 | 1.36 (83/61) | yes | no |
| Split model, nothing fitted, physical D | F2_phys | 0.0 | 20.56 (1254/61) | no | no |

#### Appendix: the AD check geometry (TTV2)

- extra AD heads at t = 25 m: 66, of which on too-steep walls 54; classes {'too-steep wall': 54, 'tile edge': 0, 'other': 12}
- principal axis of the wall heads: spread 896 m along, rms 149 m across; centroid cell [34.888888888888886, 53.407407407407405]
- slope at the wall heads: AD median 1.39, real landscape at the same cells 0.21 (95th percentile of the real landscape inside the polygon 0.62)
- drainage area at the wall heads (12 m): AD median 432 m², real landscape 288 m²; AD − real elevation median -18.6 m

### I. Figures (`reports/v2/figures/a22/`; captions as printed on each figure, from `post/captions.json`)

- **Figure 1. How well does each model fit the M&D area?** `figures/a22/fig1_misfit_md_area.png` (0.21 MB)
  - How to read: Each bar is a model's misfit (mean squared elevation error, m²) computed only over the cells inside the M&D study area, one panel per tile (TTV1, TTV2, TTV3); the dashed line marks the AD model's value; lower is better. The fourth panel puts the physical-D model on its own scale because its misfit is tens of times larger. misfit computed only inside the M&D study area on tiles TTV1, TTV2 and TTV3.
  - What it shows: TTV1: AD 224, split M&D area×slope² rule 147, split area×slope² fitted 102, split area rule fitted 85, nothing fitted 281 m²; TTV2: AD 207, split M&D area×slope² rule 105, split area×slope² fitted 89, split area rule fitted 61, nothing fitted 126 m²; TTV3: AD 252, split M&D area×slope² rule 107, split area×slope² fitted 71, split area rule fitted 35, nothing fitted 111 m².
- **Figure 2. How many of M&D's mapped heads does each model find, and how many extra heads does it make?** `figures/a22/fig2_matched_missed_extra.png` (0.21 MB)
  - How to read: For each model, three bars pooled over tiles TTV1–TTV3 inside the M&D study area: mapped heads matched one-to-one to a model head within 25 m (solid), mapped heads missed (hatched orange), and extra model heads with no mapped head within 25 m (pale). The dashed line is the number of M&D heads (61). Good is matched near 61, missed and extra near 0. The right panel holds the two nothing-fitted models; the real landscape (lidar) is repeated as the reference.
  - What it shows: Real landscape 19 matched / 67 extra; AD 8 / 122; split M&D area×slope² rule 15 / 65; split area×slope² fitted 14 / 102; split area rule fitted 15 / 132; nothing fitted 21 / 62; physical D 39 / 1215.
- **Figure 3. Does the answer depend on how close counts as a match?** `figures/a22/fig3_tolerance.png` (0.40 MB)
  - How to read: Left: the share of M&D's mapped heads that have a model head paired to them (one-to-one), against the match tolerance from 10 to 50 m. Right: the share of the model's heads that are paired to a mapped head. Higher is better on both. The black dashed line is the real landscape (lidar) through the same head-finding method, the best achievable; the vertical dotted line is the primary tolerance (25 m). Pooled over tiles TTV1–TTV3 inside the M&D study area.
  - What it shows: At 25 m the real landscape reaches 0.31 found / 0.22 real; AD 0.13 / 0.06; split M&D area×slope² rule 0.25 / 0.19; split area×slope² fitted 0.23 / 0.12; nothing fitted 0.34 / 0.25.
- **Figure 4. Does each model start its channels too high or too low?** `figures/a22/fig4_offsets.png` (0.20 MB)
  - How to read: Each point is one matched pair (a mapped head and its model head within 25 m): the distance along the real landscape's 12 m flow path between the two. Positive means the model channel starts downslope of the mapped head (too short), negative upslope (too long); the zero line is 'starts where M&D mapped'; the black bar is the median; the grey band is ± 25 m. Pairs whose two heads are in different hollows have no offset and are counted in the text under each strip. Pooled over tiles TTV1–TTV3 inside the M&D study area.
  - What it shows: Medians: Real landscape (lidar) +0 m; AD model +0 m; Split model, M&D area rule +0 m; Split model, M&D area × slope² rule +0 m; Split model, area × slope² rule fitted +0 m; Split model, area rule fitted +0 m; Split model, nothing fitted +0 m; Split model, nothing fitted, physical D +0 m.
- **Figure 5. Total channel length inside the M&D area** `figures/a22/fig5_channel_length.png` (0.18 MB)
  - How to read: Each bar is the total length of a model's 12 m channel network inside the M&D study area on tiles TTV1–TTV3 (sum of cell-to-cell flow distances; 876 m² area × slope² rule applied to the model's own surface); the dashed line is the length of the channels M&D traced inside the same area. A bar at the line matches the map.
  - What it shows: Model ÷ mapped: Real landscape (lidar) 0.41; AD model 0.51; Split model, M&D area rule 0.41; Split model, M&D area × slope² rule 0.36; Split model, area × slope² rule fitted 0.40; Split model, area rule fitted 0.42; Split model, nothing fitted 0.37; Split model, nothing fitted, physical D 2.34.
- **Figure 6a. Where are the channel heads on tile TTV1?** `figures/a22/fig6_map_TTV1.png` (0.55 MB)
  - How to read: Six panels for tile TTV1: the real landscape and five models, each shown as the hillshade of its own 12 m surface with its channel cells shaded. Orange triangles are M&D's mapped heads (a black cross marks one no model head came within 25 m of); filled circles are model heads matched one-to-one to a mapped head (a thin line joins the pair); open circles are extra model heads. Only heads inside the dashed M&D study area are counted. Scale bar 200 m; north up.
  - What it shows: Real landscape (lidar): 8 matched of 22, 35 extra; AD model: 3 matched of 22, 11 extra; Split model, M&D area × slope² rule: 3 matched of 22, 21 extra; Split model, M&D area rule: 3 matched of 22, 20 extra; Split model, area × slope² rule fitted: 3 matched of 22, 52 extra; Split model, nothing fitted: 7 matched of 22, 27 extra.
- **Figure 6b. Where are the channel heads on tile TTV2?** `figures/a22/fig6_map_TTV2.png` (0.66 MB)
  - How to read: Six panels for tile TTV2: the real landscape and five models, each shown as the hillshade of its own 12 m surface with its channel cells shaded. Orange triangles are M&D's mapped heads (a black cross marks one no model head came within 25 m of); filled circles are model heads matched one-to-one to a mapped head (a thin line joins the pair); open circles are extra model heads. Only heads inside the dashed M&D study area are counted. Scale bar 200 m; north up.
  - What it shows: Real landscape (lidar): 9 matched of 34, 28 extra; AD model: 5 matched of 34, 66 extra; Split model, M&D area × slope² rule: 10 matched of 34, 40 extra; Split model, M&D area rule: 14 matched of 34, 52 extra; Split model, area × slope² rule fitted: 11 matched of 34, 42 extra; Split model, nothing fitted: 11 matched of 34, 29 extra.
- **Figure 6c. Where are the channel heads on tile TTV3?** `figures/a22/fig6_map_TTV3.png` (0.44 MB)
  - How to read: Six panels for tile TTV3: the real landscape and five models, each shown as the hillshade of its own 12 m surface with its channel cells shaded. Orange triangles are M&D's mapped heads (a black cross marks one no model head came within 25 m of); filled circles are model heads matched one-to-one to a mapped head (a thin line joins the pair); open circles are extra model heads. Only heads inside the dashed M&D study area are counted. Scale bar 200 m; north up.
  - What it shows: Real landscape (lidar): 2 matched of 5, 4 extra; AD model: 0 matched of 5, 45 extra; Split model, M&D area × slope² rule: 2 matched of 5, 4 extra; Split model, M&D area rule: 1 matched of 5, 9 extra; Split model, area × slope² rule fitted: 0 matched of 5, 8 extra; Split model, nothing fitted: 3 matched of 5, 6 extra.
- **Figure 7. What is the line of extra heads on the AD surface?** `figures/a22/fig7_ad_check.png` (0.43 MB)
  - How to read: Left and middle: the local slope of the AD model's 12 m surface and of the real landscape's, on one colour scale (brighter = steeper), with the AD model's heads inside the M&D study area overlaid (filled = matched to a mapped head, open = extra; cyan open = on a too-steep slope). The green line is the transect of the right panel, drawn through the extra heads' centroid across their principal axis (white dotted). Right: the elevation of both surfaces along the transect.
  - What it shows: AD makes 66 extra heads on TTV2 at 25 m, 54 of them on slopes above the real landscape's 95th percentile (0.62); at those cells the AD slope is 1.39 against 0.21 in the real landscape, and the AD surface sits -18.6 m from it.
- **Figure 8. Are the fitted K and D physical?** `figures/a22/fig8_K_D_physical.png` (0.27 MB)
  - How to read: Left: each model's fitted K divided by the measured erosion-rate-based K, one point per tile; the shaded band is within a factor of two of the measured value (log scale: 2 means twice as large). Right: the fitted D on a log axis (tick marks carry the values), with Heimsath's measured D as the dashed line and each tile's relief D as a short dash-dot line. Fits use the misfit inside the M&D study area on TTV1–TTV3.
  - What it shows: K ÷ measured: AD model 10.05 / 4.84 / 2.21; Split model, M&D area rule 0.85 / 1.02 / 1.04; Split model, M&D area × slope² rule 0.85 / 0.99 / 0.96; Split model, area × slope² rule fitted 0.91 / 0.98 / 0.99; Split model, area rule fitted 1.09 / 1.03 / 1.24. D ÷ relief D: AD model 0.10 / 0.01 / 0.00; Split model, M&D area rule 1.02 / 0.74 / 0.69; Split model, M&D area × slope² rule 1.13 / 0.80 / 1.30; Split model, area × slope² rule fitted 1.80 / 0.66 / 1.62; Split model, area rule fitted 2.15 / 0.86 / 1.83.
