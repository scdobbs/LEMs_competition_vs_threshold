# Constrained channel-width re-run (A >= 10^5 m²): final report

Date: 2026-11-02
Branch: `feature/channel-width` (code 78a6190; outputs committed after this report)
Directive: `reports/directives/channel_width_constrained_directive.md` with Steve's
amendments after the checkpoint (relief ceiling 4 m, `one_bank` rule, `unresolved_wide`,
W_trench and W_bed kept separate, trench depth).
Outputs: `outputs/widths_constrained/` (`points.csv`, `excluded.csv`, `transects.csv`,
`points.gpkg`, `validation_points.gpkg` and `.csv`, `exceedance.csv`,
`window_sensitivity.csv`, `fits.json`, `regime_outputs_note.txt`, `figures/fig1..fig5`,
`qc/` random 20 per site, `qc_all/` every point (untracked), `checkpoint/` (first pass,
relief ceiling 1.5 m), `checkpoint_v2/` (same 18 points under the amended rules)).
Job 50851178 (217 s). Tests: 27 of 27 pass. `outputs/widths/` untouched.

## 1. Answer to the two questions

**"Channels at A >= 10^5 m² are wider than 1 m."** Supported, with a stated
coverage. Every point where a channel could be resolved (51 of 85) has a bed
width at 0.5 m above the thalweg greater than 1 m (51/51), and 49/51 exceed
2 m. As a fraction of all sampled points, including the 34 where no channel
could be resolved, the conservative bed-width statistic is 23/35 on AP
(0.66, 95 % interval 0.48-0.81) and 32/50 on GM (0.64, 0.49-0.77). The
unresolved points are not evidence of narrow channels: none is
`unresolved_narrow`; they are flat valley floors with no bank (9, all GM),
or sections where the two sides disagree (21 `ambiguous`). So the honest
statement is: wherever these DEMs resolve a channel at A >= 10^5 m², it is
wider than 1 m, and that is two-thirds of the sampled reaches.

**"Some are wider than 5 m."** Supported for the trench, not for the bed.
Break-to-break widths exceed 5 m at 42 of 51 resolved points (AP 18/35 of all
points, 0.51; GM 28/50, 0.56), and the four `wide_or_terrace` points are at
least 12 m. The bed at 0.5 m exceeds 5 m at only 3/35 AP and 5/50 GM points.
Nothing resolved exceeds 10 m break-to-break (0/51); the only W > 10 m
evidence is the four right-censored `wide_or_terrace` points.

## 2. Sample and classes

85 points: GM 50 (10 per bin, 5 bins from 10^5 to 8.9 x 10^6 m²); AP 35. AP bin 3
(2.2 x 10^6 to 6.1 x 10^6 m²) has no channel pixels on any AP tile, and bin 4
(trunk, tiles 3-4) gave 5 of 10 after all 546 candidates were tested. Rules were
not relaxed. Exclusions: AP 559 (321 spacing, 144 edge, 51 confluence, 39 fill
proxy, 2 confinement, 2 bin short), GM 43.

| class | AP | GM | all | width reported |
|---|---|---|---|---|
| inner_channel | 10 | 29 | 39 | W_trench, W_bed, depth |
| one_bank (bank + wall) | 12 | 0 | 12 | W_trench at the bank's relief, W_bed, depth |
| wide_or_terrace | 1 | 3 | 4 | W_trench > 12 m (right-censored); W_bed |
| unresolved_narrow | 0 | 0 | 0 | - |
| unresolved_wide | 0 | 9 | 9 | none (flat floors 4-32 m at 0.5 m) |
| ambiguous | 12 | 9 | 21 | none |

Flags: `wall_bounded` 12 (AP), `asymmetric_banks` 8 (AP 7, GM 1), `class_tie` 8,
`inconsistent_along_stream` 0. Validation export: 55 points (39 + 12 + 4).

What the ambiguous points are: 8 are transect ties (2 inner_channel vs 2
wide_or_terrace or one_bank, with the fifth different); 7 have a bank on one
side and nothing qualifying on the other (a gentle side with no convex break
and no wall within 6 m); 6 have no break on either side but are not flat and
not narrow. The checkpoint comparison shows the amended rules did what was
intended: of the 5 points the first pass called `unresolved_narrow`, 4 became
`one_bank` (W_trench 4.2-6.4 m, depth 2.0-2.9 m) and 1 `inner_channel`; of the
12 `ambiguous`, 4 became `inner_channel`, 2 `wide_or_terrace`, 2
`unresolved_wide`, 4 stayed ambiguous.

## 3. Widths and depth (resolved points)

| | AP (n=22) | GM (n=29) |
|---|---|---|
| median W_trench (break-to-break) | 5.6 m | 6.5 m |
| median W_bed (0.5 m stage) | 2.6 m | 3.4 m |
| median trench depth | 1.65 m | 1.37 m |
| W_bed > 1 m / > 2 m | 22/22, 21/22 | 29/29, 28/29 |
| W_trench > 5 m / > 10 m | 15/22, 0/22 | 27/29, 0/29 |

Per bin, median W_trench runs 5.4-6.3 m on AP and 5.0-8.1 m on GM with no
clear trend from 10^5 to 10^7 m²; W_bed 2.3-2.9 m (AP) and 3.0-4.2 m (GM).
Trench depth is 0.7-3.1 m on AP and 0.55-2.95 m on GM, without an A trend
(figure 5). The W_trench / W_bed ratio is about 2: the channels are
trapezoidal to V-shaped trenches 1-3 m deep whose bed is a third to a half
of the bank-to-bank width.

**Consistency.** W_trench agrees with W_stage_bank (the stage width at the
mean pick height) within 25 % for 27/29 GM inner_channel points (median ratio
0.97) and 6/10 AP points (median 0.87). The four AP outliers have ratios
0.54-0.7 and carry `asymmetric_banks`: a bench on one side makes the stage
width run out over the bench while the pick stops at the bench edge
(GM2_008 is the same pattern on GM).

## 4. Model-free exceedance (figures 2 and 2b; `exceedance.csv`)

Fraction of all sampled points shown to exceed each width, with 95 %
Clopper-Pearson intervals. Bed = W_stage_0.5 (conservative); trench =
break-to-break (wide_or_terrace counted up to 12 m).

| site | measure | > 1 m | > 2 m | > 5 m | > 10 m |
|---|---|---|---|---|---|
| AP (n=35) | bed | 0.66 [0.48, 0.81] | 0.60 [0.42, 0.76] | 0.09 [0.02, 0.23] | 0.00 [0, 0.10] |
| AP | trench | 0.66 [0.48, 0.81] | 0.66 [0.48, 0.81] | 0.51 [0.34, 0.69] | 0.03 [0, 0.15] |
| GM (n=50) | bed | 0.64 [0.49, 0.77] | 0.64 [0.49, 0.77] | 0.10 [0.03, 0.22] | 0.00 [0, 0.07] |
| GM | trench | 0.64 [0.49, 0.77] | 0.64 [0.49, 0.77] | 0.56 [0.41, 0.70] | 0.06 [0.01, 0.17] |

By bin (bed measure, > 1 m): AP 0.9, 0.6, 0.6, -, 0.4; GM 0.3, 0.3, 0.7, 1.0,
0.9. The GM fraction rises with A because the two lowest GM bins hold most of
the flat-floor `unresolved_wide` and `ambiguous` points; above 1.6 x 10^6 m²
19 of 20 GM points resolve. The AP trunk bin (n=5) is 2/5 because three trunk
sections are ambiguous (terraces: `asymmetric_banks` on two of them).

## 5. Is the window binding?

Partly, and in a specific way. Pile-up: no W_trench lies within 1 m of
2 x offset at any setting (0-4 %). The `wide_or_terrace` fraction is 0.22 at
offset 4 m, 0.05 at 6 m and 0.01 at 8 m (relief 4 m); the median W_trench of
resolved points moves 5.75 -> 6.25 -> 6.5 m. The automated test flags
"binding" because the fraction changes by more than 0.15 between 4 and 8 m,
but the change is almost entirely between 4 and 6 m. At the default 6 m the
window is only weakly binding: 4 of 85 points are right-censored and
widening to 8 m adds 0.25 m to the median. Widths are therefore mild lower
bounds at the 6 m window, not strongly truncated. The relief ceiling matters
more: at 1.5 m only 20-31 points resolve, at 2.5 m 45-50, at 4 m 51-58
(`window_sensitivity.csv`, figure 4).

## 6. Fit (secondary; does not qualify)

Tobit on W_trench (resolved observed, wide_or_terrace right-censored, no
left-censored points), 1000 bootstrap samples:

| site | window | n | c (m) | b [95 % CI] |
|---|---|---|---|---|
| AP | default (6 m, 4 m) | 23 | 1.28 | 0.116 [0.010, 0.183] |
| AP | 4 m, 1.5 m | 7 | 1.76 | 0.070 [-0.188, 0.199] |
| AP | 8 m, 4 m | 27 | 1.39 | 0.112 [0.035, 0.164] |
| GM | default | 32 | 1.87 | 0.093 [-0.017, 0.177] |
| GM | 4 m, 1.5 m | 14 | 3.01 | 0.047 [-0.026, 0.147] |
| GM | 8 m, 4 m | 32 | 1.33 | 0.117 [0.061, 0.183] |

b is 0.05-0.12 everywhere: a weak positive dependence of trench width on
area over two decades, with the GM default interval crossing zero. The
qualification rule (CI within [0, 1] and window not binding) fails on the
binding flag at both sites and on the CI at GM, so no `regime_boundary.csv`
or regime-III fraction was written (`regime_outputs_note.txt`). Had it
qualified, b of about 0.1 would put A* for delta = 5 m at 10^6 to 10^7 m² and
for delta = 10 m far outside the tiles; the exceedance table says the same
thing without a model.

## 7. Main uncertainties

- Coverage, not width, is the weak point: a third of the sampled reaches do
  not resolve. On GM they are flat valley floors where no bank exists within
  25 m at this resolution; on AP they are mostly transect disagreements and
  benches. The > 1 m claim holds for the resolved two-thirds and says nothing
  about the rest.
- The bed width at 0.5 m is a stage, not bankfull. It is the conservative
  number because any real channel is at least this wide at half a metre of
  flow depth; it under-reads trapezoidal beds.
- `one_bank` widths (12, all AP) are the stage width at the accepted bank's
  relief; they assume the wall side is vertical-ish. Their median (5.7 m)
  matches the inner_channel median.
- Benches and terraces (8 `asymmetric_banks`) bias W_trench low on one side
  and the stage width high; the consistency figure isolates them.
- 10 points per bin: bin fractions have intervals of +/- 0.3.
- Filled DEMs only; 40 candidates removed by the fill proxy on AP.
- AP bin 3 is empty by geometry and the AP trunk bin has 5 points.

## 8. Reproducing

```bash
cd ~/LEMs_competition_vs_threshold-JGR-branch
PY=/home/groups/hilley/scdobbs-envs/lem-widths/bin/python
$PY -m pytest tests/test_width_analysis.py -q
$PY -m width_analysis.run_constrained --stage checkpoint --out outputs/widths_constrained/checkpoint_v2
$PY -m width_analysis.run_constrained --stage full          # ~220 s
```
Job script: `$SCRATCH/width_jobs/constrained_full.sbatch`.
