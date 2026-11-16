# [2026-11-13 04:50 PST] E*-R*, a slope-dependent threshold, and Phase 7 (addendum 8)

Directive: `reports/directives/2026-11-12T1630PST_refactor_v2_addendum_8.md` (outline fixed by its section 4). Branch `v2`;
code commits 755e5b3, 97d9e83, ae49928 and the commit carrying this report. Environment `lemcvt` (Python 3.12, numpy
2.5.4, scipy 1.18.1); every job on `serc`, 1 CPU. Machine-generated tables: section 8 (appendices D1-D8,
`reports/v2/scripts/make_addendum8_tables.py`); figures under `reports/v2/figures_v2/` (git-ignored) and in the run
directories named below. Every number is tied to a run id under `$GROUP_HOME/lemcvt_results` or a Slurm job id. The
work spans three agent sessions (a Fable agent wrote the first code; Opus agents finished it, ran it and wrote this).
**This is a stop for review** (section 7). Test 2 stops at its own review gate (section 5).

## Summary

| Item | Result |
| --- | --- |
| 1.2 erratum | One line added under the title of `20261112T1003PST_phase6_results.md`, as worded in the addendum; no other edit to an earlier report |
| 2.1 the curve | The published steady nonlinear E*-R* curve agrees with a direct integration of the 1-D profile to 8e-12. **The addendum's linear limit is wrong: the curve and the parabolic linear hillslope give R* = E*/4, not E*/2**, so S̄/(C_ht L_H) = 0.5 on a linear hillslope. The rule was applied to the normalised ratio 2S̄/(C_ht L_H), which is 1 on a linear hillslope; the literal reading gives the same verdicts on the primary network |
| 2.3 rule | **Supported at both sites on the primary network (10³ m²): 5 of 5 tiles each.** Normalised ratio 0.19-0.32 on AP and 0.38-0.52 on GM, with 68 % upper bounds all below 0.55; fitted S_c 0.27-0.34 on AP and 0.26-0.63 on GM, each inside its own 68 % interval. Also supported on 10⁴ m² (5 and 5). On the rollover network AP passes 4 of 5 and GM 0 of 5 (ratio 0.84-0.95) |
| 2.4 nonlinear joint at fitted S_c | **Stopped on all ten tiles by the pre-registered cap rule.** In every part, 42-67 % of the first 22 evaluations hit the Picard cap of 100; all cap hits are at D below 0.6-1.6 × 10⁻² m²/yr. The gap verdict (closed / narrowed / not) cannot be applied. Converged evaluations exist only at D ≥ 0.64-1.6 × 10⁻², with MSE 1.0-2.1 × the linear joint's on the 6 lowest of the 30 A_crit values |
| 3 slope-dependent threshold | Implemented as A S^α ≥ C first holding along a path (exact upstream maximum); α = 0 reproduces the Phase 6 profile to 3e-11 on every tile. **Not preferred at either site** (held-out MSE lower by > 2 % on 2 of 5 AP tiles, 1 of 5 GM tiles). In-sample MSE changes by -16 % to +0 % (best α). **D is unchanged by α**: 4.0-9.1 × D₀,A on AP and 1.7-3.9 × on GM at every α. Fitted C at α = 2: 370-1554 m² on AP and 83-745 m² on GM, against Montgomery & Dietrich's 500-4000 m² |
| 5 Phase 7 | Width, constrained width and Test 2 head extraction ported to `lemcvt` and run on the metre grids. **On GM the ports reproduce the feature branches exactly** (every width, class, fit and head pixel). On AP everything changes, because the branches read the AP DEMs in US survey feet; the AP width fits and AP head sets are new results. Test 2 stops at its review gate |
| cost | 1010 core-hours of limits submitted (largest single submission 300), 304 used; at most about 120 CPUs at once; `$GROUP_HOME` 266 GB of 1 TB |

## 1. State check and the erratum line

**State at the start.** `v2` was at 211469b = `origin/v2` (the addendum 7 report). The tests passed on `serc` at that
head (job 52117501: ruff and mypy clean, 193 passed). No earlier job was running. Where I could check, the addendum
and the repo agree, with one exception that is the addendum's own algebra, not the repo (section 2, the linear limit).
That exception does not change which code is right, so I did not stop; I state it and apply the rule both ways.

**Erratum (1.2).** Inserted as a quoted line under the title of `reports/v2/20261112T1003PST_phase6_results.md`, with
the exact wording of the addendum. Nothing else in an earlier report was edited.

**Decisions recorded (1.1, 1.3).** The hollow-averaging verdict of addendum 7 stands. Condition (a) of addendum 7 §2.3
(the mean Laplacian of the 1 m grid) is not reused anywhere in this addendum.

**Tests and code.** New modules `analyses/er_star.py` (§2), `analyses/slope_threshold.py` (§3), the S_c override and
cap-hit stop of the nonlinear joint profile (§2.4), the Phase 7 ports `analyses/width/` and `analyses/heads/`, the
driver `analyses/phase7.py` and the figures `a8_er_star`, `a8_slope_threshold`. Commands `lemcvt phase7 {width,
width-constrained, heads, port-check}` and `lemcvt phase8 {er-star, slope-threshold, slope-threshold-summary}`. The
feature-branch test suites were ported (27 width, 6 heads tests) and 8 tests added (`tests/test_addendum8.py`): the
curve against the integration, the S_c fit recovering a planted value, the parabolic trace, the upstream maximum, the
basin halves, α = 0 against the Phase 6 profile, the S_c override with the stop rule and merge, and the metre tile
loader of the width port. Final suite: ruff and mypy clean, 233 passed (job 52122905, plus 52125281 for the last
changes).

## 2. E*-R* (§2.1-2.3), with the rule applied

Run `20261113T015030Z_er_star_10tiles` (job 52123352, 2.5 min); figure `reports/v2/figures_v2/a8_er_star.png`
(primary network, one panel per tile). Definitions are in `docs/measurement_methods.md` §10. On each option (iii)
hilltop pixel: C_ht = −∇²z at L_f = 11 m; L_H and R along the D8 steepest-descent path to the first network cell;
S̄ = R/L_H; E* = 2 C_ht L_H / S_c; R* = S̄ / S_c. Segment bootstrap, 200 draws.

**The curve (D1).** The published form and a Simpson integration of the 1-D steady nonlinear profile agree to 8e-12 at
E* = 0.01 to 50, so the published curve is used. Its small-E* limit is R* = E*/4. The parabola z = z₀ − (C_ht/2)x²,
the steady linear hillslope, gives the same: S̄ = C_ht L_H / 2. The addendum writes the linear limit as E*/2, so its
linear test "S̄/(C_ht L_H) = 1 is linear" is off by a factor 2. Read literally, a perfectly linear hillslope (ratio 0.5)
would pass "median below 0.67". I therefore applied the thresholds to the normalised ratio 2S̄/(C_ht L_H), which is 1
on a linear hillslope, and report the literal ratio beside it. On the primary and 10⁴ m² networks the two readings
give identical verdicts; on the rollover network the literal reading passes GM where the normalised one does not.

**Results, primary network (D2).**

| tile | pixels used / total | L_H median (m) | S̄ median | ratio 2S̄/(C L) (68 %) | S_c (68 %) | E* median at S_c | S̄ > S_c | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 2568 / 3446 | 85 | 0.230 | 0.32 [0.29, 0.34] | 0.31 [0.30, 0.32] | 8.5 | 9 % | yes |
| AP2 | 705 / 944 | 107 | 0.236 | 0.23 [0.21, 0.25] | 0.29 [0.28, 0.31] | 13.9 | 12 % | yes |
| AP3 | 444 / 533 | 102 | 0.256 | 0.19 [0.17, 0.21] | 0.27 [0.22, 0.30] | 19.2 | 43 % | yes |
| AP4 | 3779 / 4711 | 90 | 0.228 | 0.29 [0.27, 0.30] | 0.27 [0.25, 0.30] | 9.5 | 28 % | yes |
| AP5 | 2248 / 3575 | 108 | 0.260 | 0.26 [0.24, 0.27] | 0.34 [0.33, 0.35] | 11.3 | 10 % | yes |
| GM1 | 2733 / 3583 | 78 | 0.324 | 0.52 [0.49, 0.54] | 0.54 [0.53, 0.56] | 4.6 | 1 % | yes |
| GM2 | 2072 / 2530 | 96 | 0.348 | 0.51 [0.47, 0.55] | 0.54 [0.52, 0.57] | 4.7 | 0 % | yes |
| GM3 | 1209 / 1500 | 72 | 0.229 | 0.38 [0.33, 0.42] | 0.26 [0.24, 0.29] | 8.8 | 27 % | yes |
| GM4 | 2534 / 3347 | 96 | 0.362 | 0.47 [0.44, 0.52] | 0.63 [0.62, 0.65] | 4.7 | 3 % | yes |
| GM5 | 663 / 1128 | 96 | 0.379 | 0.40 [0.38, 0.42] | 0.57 [0.53, 0.59] | 6.5 | 0 % | yes |

The pixels not used are those whose path leaves the tile before reaching the network (89-1263 per tile).

| network | AP tiles passing | GM tiles passing | supported (normalised) | supported (literal) |
| --- | --- | --- | --- | --- |
| primary, 10³ m² | 5 | 5 | AP yes, GM yes | AP yes, GM yes |
| 10⁴ m² | 5 | 5 | AP yes, GM yes | AP yes, GM yes |
| rollover (18-447 m²) | 4 | 0 | AP yes, GM no | AP yes, GM yes |

**By the pre-registered rule, the nonuniform-curvature reading is supported at both sites; the primary network
decides.** What the numbers say beyond the verdict is in section 6; three facts belong here:

- **The fitted S_c is close to the mean gradient itself.** Median E* at the fitted S_c is 4.6-19, where the curve is
  flat (R* = 0.7-0.95), so the fit returns S_c ≈ S̄/0.8. The data occupy the saturated limb; they do not trace the
  bend of the curve (figure).
- **On AP, 9-43 % of hilltop pixels have S̄ > S_c**, which the steady nonlinear curve cannot produce (R* < 1). On GM
  it is 0-3 % except GM3 (27 %).
- **The ratio depends on the network more than on the site.** From the rollover network to 10³ to 10⁴ m², L_H grows
  by 2-12 ×, while S̄ barely changes (AP 0.21 → 0.23 → 0.24 on AP1), so the ratio falls in proportion. GM fails
  on the rollover network, where L_H is only 9-19 m.

## 3. The nonlinear joint at fitted S_c (§2.4)

Run because §2.3 was supported at both sites. Per tile: K fixed at the linear joint optimum (the latest refined Phase 6
run); the addendum 6 A_crit grid (30 values); D extended down to 10⁻³ m²/yr as 22 log-spaced values from 10⁻³ to
5 × 10⁻² (step × 1.205, which is the addendum 6 step of × 1.203 continued); Picard cap 100; S_c = the tile's primary-
network fit. Each tile was split into 6 interleaved parts (one array task each), and each part applied the
pre-registered rule after every A_crit row of 22 evaluations: more than 30 % at the cap stops it. **Estimate printed
before submitting:** 2 submissions × 30 tasks × 10 h = 300 core-hours of limits each (jobs 52125659 AP, 52125670 GM;
merges 52125661, 52125686). Used: 124 core-hours.

**Every part stopped after its first A_crit row**, so each tile has 132 of 660 evaluations, all at the 6 lowest A_crit
values of the grid (D3):

| tile | S_c | at the cap | all evaluations converge for D ≥ | best converged: A_crit, D, MSE, D/D₀,A | linear joint MSE |
| --- | --- | --- | --- | --- | --- |
| AP1 | 0.306 | 55 % | 1.13e-2 | 2667, 9.35e-3, 36.3, 2.61 | 22.4 |
| AP2 | 0.293 | 50 % | 9.35e-3 | 1237, 7.76e-3, 40.9, 2.41 | 22.5 |
| AP3 | 0.267 | 64 % | 1.64e-2 | 1433, 1.13e-2, 89.7, 5.23 | 42.1 |
| AP4 | 0.271 | 59 % | 1.13e-2 | 2098, 1.13e-2, 85.2, 3.10 | 35.2 |
| AP5 | 0.341 | 42 % | 6.44e-3 | 895, 6.44e-3, 46.9, 2.21 | 27.9 |
| GM1 | 0.544 | 49 % | 7.76e-3 | 559, 9.35e-3, 30.9, 1.04 | 24.7 |
| GM2 | 0.540 | 59 % | 1.13e-2 | 1638, 1.36e-2, 31.1, 1.39 | 26.1 |
| GM3 | 0.264 | 67 % | 1.64e-2 | 198, 1.36e-2, 48.4, 1.34 | 33.0 |
| GM4 | 0.634 | 59 % | 1.13e-2 | 1382, 1.13e-2, 65.7, 1.14 | 40.7 |
| GM5 | 0.571 | 51 % | 9.35e-3 | 658, 7.76e-3, 38.0, 1.02 | 32.2 |

- **The pre-registered reading (closed, narrowed, otherwise not) is not applied**: the addendum says to stop and report
  rather than raise the cap, and that is what happened on all ten tiles.
- The cap hits are not scattered: every evaluation below a D threshold hits the cap and every evaluation above it
  converges (D3, and `nonlinear_joint_profile.parquet` in each merged run). With S_c near the mean hillslope gradient,
  low D asks the hillslopes to carry the uplift flux at gradients the model cannot reach below S_c, and the Picard
  iteration stalls with D_eff clipped.
- **For the record only:** where evaluations converge, 73-94 % of hillslope cells exceed 0.5 S_c, so the nonlinearity is
  engaged, unlike Phase 6. On GM the best converged evaluations sit at D/D₀ = 1.0-1.4, inside the addendum's 0.67-1.5
  window, but at MSE 1.2-1.6 × the linear joint's and on only the 6 lowest A_crit values (198-1638 m²), below the
  linear joint A_crit on four of five GM tiles. On AP they sit at D/D₀,A = 2.2-5.2 with MSE 1.6-2.4 ×. None of this is
  an optimum.
- Bookkeeping: the AP5 parts computed the D grid on different CPU generations, and two of the 22 values differ in the
  last bit between parts (the merged table therefore shows 24 distinct D); harmless, noted.

## 4. Slope-dependent threshold (§3)

**Definitions (3.1).** S is the drop over the D8 path from each cell to the first cell at or beyond 10 m, divided by
that path length (median 0.25-0.26 on AP, 0.24-0.43 on GM; 7 000-17 000 cells per tile whose path leaves the tile
before 10 m use the length available). A cell is a channel when A S^α ≥ C holds at it or anywhere upstream of it,
i.e. when M = the upstream maximum of A S^α is ≥ C. This is "every cell downstream of a head is a channel", it keeps
the channel set monotone in C, and it changes exactly at the unique values of M, so the same exact profile machinery
applies (the Phase 6 reduction with M in place of A). M is computed in one pass in ascending area, repeated to a fixed
point to stay exact where stored areas are inconsistent (the AP1 seam rows). With α = 0, M = A and **the profile
equals the Phase 6 profile to 3e-11 in MSE and 2e-11 in D at 1029-1427 common A_crit values on every tile** (D5). Every
channel cell drains to the tile boundary (0 channel cells without χ at every optimum).

**Runs (3.2).** 10 tiles × 6 values of α, each thinned to 2000 unique values of M between the C that give the
channel-cell counts of A_crit = 10² and 10⁵ m², then refined (2000-2335 evaluations). Arrays 52123384 (AP), 52123396
(GM); 30 × 6 h limits each, 176 core-hours used; summary run `20261113T054020Z_slope_threshold_summary_10tiles`;
figure `a8_slope_threshold.png` (MSE(C)/min against the channel-cell count, every α). Full table in D4.

**Hold-out (3.3) and the rule (D5).** Basins are split by alternating rank of outlet area at the 10³ m² network; at
every C the joint (K, D) is fitted on each half and scored on the other, from the same reductions; each half's own
best C is scored on the other half and the two scores averaged.

| tile | held-out MSE at α = 0 | best α > 0 | held-out improvement | preferred | best in-sample α (improvement) |
| --- | --- | --- | --- | --- | --- |
| AP1 | 23.4 | 0.5 | −6.4 % | no | 1.0 (+2.9 %) |
| AP2 | 80.5 | 0.5 | +1.6 % | no | 0.5 (+3.8 %) |
| AP3 | 40.5 | 1.0 | −19.1 % | no | 0.5 (+4.8 %) |
| AP4 | 42.9 | 1.0 | +19.2 % | yes | 1.0 (+6.7 %) |
| AP5 | 62.6 | 2.5 | +10.7 % | yes | 2.0 (+7.3 %) |
| GM1 | 28.1 | 0.5 | −16.8 % | no | 0 |
| GM2 | 30.0 | 0.5 | −2.9 % | no | 0 |
| GM3 | 52.7 | 2.0 | −3.0 % | no | 2.5 (+16.1 %) |
| GM4 | 61.2 | 0.5 | +4.8 % | yes | 0.5 (+2.7 %) |
| GM5 | 61.0 | 0.5 | +1.6 % | no | 0 |

**A slope-dependent threshold is not preferred at either site** (AP 2 of 5 tiles, GM 1 of 5; the rule needs 4).
Best α per tile: AP 0.5, 0.5, 1, 1, 2.5; GM 0.5, 0.5, 2, 0.5, 0.5.

**D at every α (the question for the next directive).** The fitted D moves by less than the gap at every α: AP
1.2-2.7 × 10⁻² m²/yr, D/D₀,A = 4.0-9.1; GM 1.7-4.0 × 10⁻², D/D₀,A = 1.7-3.9 (D4). **A slope-dependent threshold does not
change the D gap.** K stays within 25 % of the α = 0 value (the largest change is AP2 at α = 1.5-2, +22 to +25 %).

**C at α = 2 against Montgomery & Dietrich (1992).** The fitted C is A S² at the model's channel heads: AP1 674, AP2
1554, AP3 315, AP4 370, AP5 196 m²; GM1 83, GM2 268, GM3 27, GM4 745, GM5 184 m². Two AP tiles and one GM tile lie
inside M&D's 500-4000 m²; the rest lie below it. The comparison is loose: their heads are field-mapped and their S is a
field slope at the head, while here S is a 10 m D8 slope from a 1 m DEM and C is whatever minimises the elevation
misfit, with flat 1 % intervals (D4).

## 5. Phase 7 (width, Test 2 up to its gate, AP reruns)

**What was ported.** `width_analysis` (feature/channel-width) and `head_extraction` (feature/head-extraction) now live
under `lemcvt.analyses.width` and `lemcvt.analyses.heads`, read their grids through the `lemcvt` loader (manifest-
checked, metres), take A_crit and m from the Phase 6 runs and configs instead of the v1 Table 1 values, and write runs
under `$LEMCVT_RESULTS`. Test 1 (`acrit_slope`) is not ported: §3 replaces it; its path helpers came along as
`heads/paths.py`. Two deliberate changes: the head extraction's hydrologic conditioning uses the package's priority-
flood fill instead of richdem's epsilon fill (richdem is not in the environment), and Method D (LSDTopoTools) stays
off by default because its binary aborted on every tile in Test 2. Jobs 52123374 (width, constrained, heads, port
check; 1.8 core-hours) and 52126453 (heads; the two heads runs are byte-identical).

**Regression against the branches on GM (D6).** Run `20261113T024349Z_phase7_port_check_all`. GM reads the same metre
grids in both, so the ports must reproduce the branch outputs there, and they do exactly:

- width (Phase 3 sample): 48 of 48 points on the same pixels; W_primary, bank-break width, valley-floor width, stage
  widths, local slope and inner relief identical; Tobit c = 29.0, b = −0.099 (31 channelized) and A50 = 4362 m² identical;
- constrained width: 50 of 50 points; classes (29 inner_channel, 9 ambiguous, 9 unresolved_wide, 3 wide_or_terrace),
  W_trench, W_bed, depth and the Tobit fit identical;
- heads: on all five GM tiles and all four smoothings, the same number of retained heads on the same pixels with the
  same classes and areas. The fill change makes no difference on GM.

**AP reruns on metres (D6, D7).** The branches read the AP grids in US survey feet with x, y in metres, so every
AP slope, relief and curvature threshold acted on a surface 3.28 × too steep. Now:

- **Width.** Local slopes fall by exactly the factor 0.3048 (median 0.181 → 0.055). Channelized points drop from 36 to
  19 of 46-47, because the 0.3 m relief and 0.10 bank-slope thresholds now apply to metres. The Tobit fit changes from
  c = 12.2 m, b = −0.021 to c = 8.2 m, b = +0.016 (19 points). The b ≈ 0 conclusion stands; A50 moves from 120 to
  3.1 × 10⁵ m².
- **Constrained width.** AP classes change from 10 inner_channel, 12 one_bank, 12 ambiguous, 1 wide to 8 inner_channel,
  0 one_bank, 18 ambiguous, 6 wide_or_terrace, 2 unresolved_wide: the 0.3-4 m relief window in metres no longer finds
  the "walls" it found in feet. Resolved AP points: 22 → 8; median W_trench 5.6 → 7.9 m; depth 1.6 → 0.9 m; the
  > 1 m bed-width fraction 0.66 → 0.41 [0.25, 0.59]. The AP fit (b = 0.12 [−0.01, 0.35]) does not qualify, and the
  window is binding (wide_or_terrace 0.31 at 4 m against 0.06 at 8 m). The constrained run's bootstrap took most of
  its 20 minutes on degenerate resamples (8 resolved points).
- **Heads, Test 2, extract phase only.** With the Wiener filter alone (as specified), AP heads are still crest pixels
  (hollow-class median A 5-45 m², 7-27 per tile). With the DT = 2 m² smoothing the AP sets change in kind: 8-30 heads per
  tile, nearly all hollow class, median A 1460-3360 m² and S_up_10 0.30-0.44 (17-24°). On the branch in feet these were
  13-33 rim and wall points at 41-48°. The AP4 DT2 map (`20261113T022845Z_heads_10tiles/head_extraction/figures/
  qc_maps_OWF_DT2/AP4.png`) shows them at the tips of incised gullies and hollows. GM is unchanged (section above).
  **Test 2 stops here, at its review gate**: no fits were run, and the decisions of the Test 2 checkpoint (which head
  set; whether AP is "extraction failed") are still Steve's, now with metre AP maps to review.

## 6. What the evidence supports, and where it is weak

**Supported.**

1. On the observed DEMs, mean hillslope gradients from the option (iii) crests to the 10³ m² network are far below what
   a parabolic (linear-diffusion) hillslope with the measured crest curvature would have: 0.19-0.32 of it on AP and
   0.38-0.52 on GM. Sharp crests over flanks whose mean gradient saturates near 0.23-0.38 is the E*-R* signature of
   slope-dependent transport, and it passes the pre-registered rule at both sites on the decisive network.
2. The linear partitioned model cannot honour both the crest curvature and the hillslope relief, which is the D gap of
   Phase 6 and addendum 7 seen from the data side: D₀ reads the crest, D_fit reads the relief.
3. A slope-dependent channel threshold neither improves held-out prediction at either site nor moves D. The D gap is
   not a by-product of the area-only threshold.
4. The Phase 7 ports are faithful: exact on GM.

**Weak.**

1. **S_c is weakly constrained.** The pixels sit at E* = 4.6-19 on the flat limb of the curve, so the fit returns
   S_c ≈ S̄/0.8 and does not test the curve's shape. On AP, 9-43 % of pixels have S̄ > S_c, which the steady curve
   excludes. The fitted values (0.26-0.63, i.e. 15-32°) are well below the S_c ≈ 1.25 usually quoted for soil-mantled
   hillslopes (Roering et al. 1999); the citation is from memory and should be checked.
2. **The ratio depends on where the hillslope is said to end.** It halves from the 10³ to the 10⁴ m² network and
   roughly doubles on the rollover network (GM fails there), with S̄ nearly constant. A D8 path that runs down a hollow
   or a valley floor before reaching the network lengthens L_H at low gradient, which lowers the ratio whether or not
   transport is nonlinear. The pre-registration made 10³ m² decisive; the dependence is still the largest caveat.
3. **§2.4 could not test the reading.** At the fitted S_c the nonlinear model has no converged steady solution below
   D ≈ 10⁻² on most tiles; the rule stopped every tile, and the converged evaluations cover only the 6 lowest A_crit
   values. Whether a nonlinear model with a lower S_c closes the gap is open.
4. **The hold-out split is lopsided.** Alternating by outlet-area rank puts the largest basin in half 1, so the larger half holds
   63-97 % of the cells on every tile (D5). The held-out MSEs (and the AP2 value of 80 against an in-sample 22) are
   driven by that asymmetry; a balanced split might change the per-tile verdicts, though it cannot create the 4-of-5
   agreement the rule needs at either site from the present in-sample gains (≤ 7 % on 9 tiles).
5. On AP the option (iii) curvature is still the bracket of addendum 6 (D₀,A); all §2 numbers use C_ht at 11 m.

## 7. Decisions needed

1. **The E*-R* linear limit.** Confirm the reading used here (normalised ratio, linear = 1) or restate the §2.3 rule.
   On the decisive network both readings give the same verdict.
2. **§2.4 after the stop.** The rule stopped every tile. Options: (a) accept "not testable at the fitted S_c" and stop;
   (b) a pre-registered follow-up with a different nonlinear solver (Newton, or continuation in D from the converged
   region) rather than a higher Picard cap; (c) restrict the D grid to the converged region, which would no longer test
   the gap at D₀ on AP. I recommend (a) for this round, with (b) only if the nonuniform-curvature reading is to carry
   weight in the paper.
3. **The network dependence of §2.** Whether a hillslope-length definition that stops at the hollow (for example the
   rollover network, or a curvature-based valley mask) should be the reference for any later E*-R* statement.
4. **Hold-out split.** Whether to rerun §3.3 with balanced halves (a full rerun of the 60 profiles, about 180
   core-hours used). The site verdict is unlikely to change (section 6, item 4).
5. **Test 2.** The review gate decisions, now with AP heads on metre grids that look like hollow and gully tips at
   DT = 2 m².
6. **Push.** The code commits and this report are pushed with this session (addendum 2 §5 check clean); CI stays manual.

## 8. Appendices (generated tables)

Generated by `reports/v2/scripts/make_addendum8_tables.py` from the run directories named in each table.

## D1. Section 2.1: the steady nonlinear E*-R* curve against the integrated profile

Run `20261113T015030Z_er_star_10tiles`. Max relative difference 7.9e-12 (tolerance 1e-06); R*/E* at E* = 1e-3: 0.2500000 (the linear limit is E*/4).

| E* | R* published form | R* integrated | relative difference |
|---|---|---|---|
| 0.01 | 0.0024999688 | 0.0024999688 | 7.9e-12 |
| 0.1 | 0.0249688537 | 0.0249688537 | 2.9e-15 |
| 0.5 | 0.1213857494 | 0.1213857494 | 3.4e-16 |
| 1 | 0.2259871559 | 0.2259871559 | 1.6e-15 |
| 2 | 0.3774280762 | 0.3774280762 | 1.5e-16 |
| 5 | 0.5968077343 | 0.5968077343 | 1.9e-16 |
| 10 | 0.7340603630 | 0.7340603630 | 0.0e+00 |
| 50 | 0.9154224902 | 0.9154224902 | 3.8e-15 |

## D2. Sections 2.2-2.3: per tile and network

ratio = 2 Sbar / (C_ht L_H) (1 on a linear hillslope), median with segment-bootstrap 68 % interval; literal = Sbar / (C_ht L_H). S_c by least squares of log10 R* against the curve, with the segment-bootstrap 68 % interval and full range. Pass = ratio median < 0.67 with 68 % upper bound < 1, and S_c inside its 68 % interval and < 1.27.

| tile | network | A_c (m^2) | hilltop px | used | left tile | L_H median (m) | Sbar median | C_ht median (1/m) | ratio (68 %) | literal | S_c (68 %) | S_c range | log10 rms | E* median at S_c | frac R* > 1 | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | primary_1e3 | 1000 | 3446 | 2568 | 634 | 84.9 | 0.230 | 0.0169 | 0.315 [0.294, 0.342] | 0.157 | 0.306 [0.298, 0.316] | [0.276, 0.346] | 0.236 | 8.53 | 0.089 | yes |
| AP2 | primary_1e3 | 1000 | 944 | 705 | 214 | 106.5 | 0.236 | 0.0179 | 0.231 [0.207, 0.251] | 0.115 | 0.293 [0.279, 0.307] | [0.250, 0.336] | 0.199 | 13.93 | 0.119 | yes |
| AP3 | primary_1e3 | 1000 | 533 | 444 | 89 | 102.3 | 0.256 | 0.0254 | 0.188 [0.170, 0.207] | 0.094 | 0.267 [0.224, 0.304] | [0.149, 0.348] | 0.290 | 19.17 | 0.428 | yes |
| AP4 | primary_1e3 | 1000 | 4711 | 3779 | 578 | 89.9 | 0.228 | 0.0171 | 0.289 [0.273, 0.302] | 0.145 | 0.271 [0.252, 0.295] | [0.204, 0.322] | 0.333 | 9.46 | 0.282 | yes |
| AP5 | primary_1e3 | 1000 | 3575 | 2248 | 1263 | 108.4 | 0.260 | 0.0187 | 0.257 [0.238, 0.274] | 0.129 | 0.341 [0.327, 0.354] | [0.303, 0.367] | 0.152 | 11.27 | 0.099 | yes |
| GM1 | primary_1e3 | 1000 | 3583 | 2733 | 840 | 77.7 | 0.324 | 0.0163 | 0.518 [0.491, 0.544] | 0.259 | 0.544 [0.529, 0.559] | [0.506, 0.590] | 0.131 | 4.60 | 0.005 | yes |
| GM2 | primary_1e3 | 1000 | 2530 | 2072 | 446 | 95.8 | 0.348 | 0.0139 | 0.509 [0.465, 0.548] | 0.254 | 0.540 [0.515, 0.565] | [0.480, 0.628] | 0.144 | 4.70 | 0.000 | yes |
| GM3 | primary_1e3 | 1000 | 1500 | 1209 | 291 | 72.0 | 0.229 | 0.0148 | 0.380 [0.330, 0.422] | 0.190 | 0.264 [0.244, 0.292] | [0.213, 0.348] | 0.288 | 8.83 | 0.270 | yes |
| GM4 | primary_1e3 | 1000 | 3347 | 2534 | 785 | 95.9 | 0.362 | 0.0159 | 0.473 [0.441, 0.516] | 0.237 | 0.634 [0.618, 0.651] | [0.590, 0.676] | 0.172 | 4.67 | 0.034 | yes |
| GM5 | primary_1e3 | 1000 | 1128 | 663 | 457 | 96.0 | 0.379 | 0.0198 | 0.396 [0.383, 0.416] | 0.198 | 0.571 [0.529, 0.590] | [0.478, 0.614] | 0.102 | 6.53 | 0.000 | yes |
| AP1 | sensitivity_1e4 | 1e+04 | 3446 | 1658 | 1671 | 172.3 | 0.238 | 0.0170 | 0.165 [0.144, 0.185] | 0.082 | 0.269 [0.252, 0.287] | [0.225, 0.308] | 0.200 | 20.01 | 0.212 | yes |
| AP2 | sensitivity_1e4 | 1e+04 | 944 | 529 | 401 | 225.3 | 0.238 | 0.0179 | 0.118 [0.109, 0.134] | 0.059 | 0.266 [0.255, 0.277] | [0.237, 0.292] | 0.133 | 28.39 | 0.059 | yes |
| AP3 | sensitivity_1e4 | 1e+04 | 533 | 237 | 296 | 76.7 | 0.183 | 0.0207 | 0.119 [0.105, 0.160] | 0.060 | 0.162 [0.119, 0.231] | [0.056, 0.270] | 0.353 | 28.19 | 0.527 | yes |
| AP4 | sensitivity_1e4 | 1e+04 | 4711 | 2637 | 1886 | 191.3 | 0.225 | 0.0173 | 0.151 [0.144, 0.161] | 0.075 | 0.236 [0.211, 0.263] | [0.165, 0.287] | 0.322 | 23.71 | 0.415 | yes |
| AP5 | sensitivity_1e4 | 1e+04 | 3575 | 1849 | 1676 | 216.0 | 0.265 | 0.0189 | 0.131 [0.116, 0.151] | 0.065 | 0.309 [0.304, 0.315] | [0.293, 0.328] | 0.099 | 25.50 | 0.107 | yes |
| GM1 | sensitivity_1e4 | 1e+04 | 3583 | 2596 | 977 | 89.7 | 0.324 | 0.0163 | 0.458 [0.434, 0.485] | 0.229 | 0.505 [0.487, 0.520] | [0.458, 0.541] | 0.139 | 5.62 | 0.002 | yes |
| GM2 | sensitivity_1e4 | 1e+04 | 2530 | 1899 | 619 | 115.3 | 0.322 | 0.0137 | 0.415 [0.358, 0.460] | 0.208 | 0.467 [0.441, 0.491] | [0.405, 0.551] | 0.145 | 6.20 | 0.000 | yes |
| GM3 | sensitivity_1e4 | 1e+04 | 1500 | 830 | 670 | 110.4 | 0.226 | 0.0134 | 0.278 [0.252, 0.301] | 0.139 | 0.302 [0.282, 0.320] | [0.266, 0.367] | 0.154 | 10.50 | 0.171 | yes |
| GM4 | sensitivity_1e4 | 1e+04 | 3347 | 1917 | 1407 | 138.7 | 0.369 | 0.0169 | 0.308 [0.289, 0.327] | 0.154 | 0.528 [0.517, 0.537] | [0.496, 0.560] | 0.149 | 8.88 | 0.000 | yes |
| GM5 | sensitivity_1e4 | 1e+04 | 1128 | 621 | 499 | 139.9 | 0.353 | 0.0198 | 0.248 [0.242, 0.266] | 0.124 | 0.459 [0.436, 0.473] | [0.396, 0.506] | 0.080 | 12.07 | 0.002 | yes |
| AP1 | rollover | 281.8 | 3446 | 3096 | 0 | 50.2 | 0.209 | 0.0165 | 0.472 [0.447, 0.512] | 0.236 | 0.332 [0.317, 0.350] | [0.301, 0.379] | 0.301 | 4.88 | 0.062 | yes |
| AP2 | rollover | 56.23 | 944 | 892 | 0 | 20.0 | 0.144 | 0.0178 | 0.661 [0.599, 0.767] | 0.331 | 0.274 [0.262, 0.296] | [0.248, 0.343] | 0.301 | 2.96 | 0.052 | yes |
| AP3 | rollover | 17.78 | 533 | 531 | 0 | 10.7 | 0.112 | 0.0255 | 0.742 [0.715, 0.776] | 0.371 | 0.354 [0.299, 0.412] | [0.227, 0.573] | 0.249 | 1.65 | 0.009 | no |
| AP4 | rollover | 446.7 | 4711 | 4211 | 0 | 59.1 | 0.220 | 0.0168 | 0.398 [0.376, 0.418] | 0.199 | 0.290 [0.267, 0.322] | [0.220, 0.354] | 0.365 | 6.09 | 0.197 | yes |
| AP5 | rollover | 56.23 | 3575 | 3458 | 0 | 21.1 | 0.157 | 0.0193 | 0.670 [0.652, 0.693] | 0.335 | 0.395 [0.375, 0.417] | [0.333, 0.463] | 0.285 | 2.02 | 0.011 | yes |
| GM1 | rollover | 35.48 | 3583 | 3568 | 0 | 13.5 | 0.110 | 0.0163 | 0.932 [0.909, 0.951] | 0.466 | 0.570 [0.519, 0.612] | [0.459, 0.699] | 0.218 | 0.77 | 0.003 | no |
| GM2 | rollover | 35.48 | 2530 | 2506 | 0 | 14.1 | 0.100 | 0.0151 | 0.917 [0.898, 0.936] | 0.458 | 0.703 [0.607, 0.765] | [0.520, 0.925] | 0.223 | 0.58 | 0.000 | no |
| GM3 | rollover | 22.39 | 1500 | 1500 | 0 | 9.2 | 0.064 | 0.0145 | 0.844 [0.816, 0.889] | 0.422 | 0.348 [0.318, 0.379] | [0.280, 0.455] | 0.217 | 0.82 | 0.047 | no |
| GM4 | rollover | 56.23 | 3347 | 3310 | 0 | 19.2 | 0.150 | 0.0150 | 0.845 [0.821, 0.877] | 0.423 | 0.658 [0.610, 0.705] | [0.563, 0.789] | 0.275 | 0.93 | 0.014 | no |
| GM5 | rollover | 28.18 | 1128 | 1114 | 0 | 13.7 | 0.136 | 0.0194 | 0.954 [0.933, 0.993] | 0.477 | 0.826 [0.752, 0.974] | [0.605, 1.383] | 0.207 | 0.64 | 0.000 | no |

Site rule (>= 4 of 5 tiles) and the site-pooled values:

| network | site | tiles passing | (a) ratio | (b) S_c | supported | pooled ratio (68 %) | pooled S_c (68 %) |
|---|---|---|---|---|---|---|---|
| primary_1e3 | AP | 5 | 5 | 5 | **True** | 0.274 [0.266, 0.283] | 0.299 [0.289, 0.308] |
| primary_1e3 | GM | 5 | 5 | 5 | **True** | 0.470 [0.454, 0.487] | 0.514 [0.495, 0.530] |
| sensitivity_1e4 | AP | 5 | 5 | 5 | **True** | 0.146 [0.140, 0.153] | 0.265 [0.253, 0.275] |
| sensitivity_1e4 | GM | 5 | 5 | 5 | **True** | 0.363 [0.349, 0.384] | 0.470 [0.460, 0.481] |
| rollover | AP | 4 | 4 | 5 | **True** | 0.529 [0.506, 0.549] | 0.321 [0.305, 0.335] |
| rollover | GM | 0 | 0 | 5 | **False** | 0.901 [0.885, 0.916] | 0.617 [0.590, 0.645] |

## D3. Section 2.4: nonlinear joint profile at the fitted S_c (stopped by the pre-registered cap rule)

K fixed at the linear joint optimum; A_crit grid as addendum 6 (30 values, split into 6 interleaved parts); D: 22 log-spaced values 1e-3 to 5e-2 m^2/yr; Picard cap 100. Each part stops after an A_crit row when more than 30 % of its evaluations hit the cap. 'Best converged' is the lowest MSE among evaluations that converged; it is reported for the record and is not an optimum (6 of 30 A_crit values evaluated).

| tile | S_c | evaluated / grid | at cap | parts stopped | lowest D converging everywhere | best converged: A_crit, D, MSE, D/D0_A, >0.5 S_c | linear joint MSE |
|---|---|---|---|---|---|---|---|
| AP1 | 0.306 | 132 / 660 | 73 (55%) | 6 of 6 | 1.127e-02 | 2667, 9.351e-03, 36.29, 2.61, 0.89 | 22.38 |
| AP2 | 0.293 | 132 / 660 | 66 (50%) | 6 of 6 | 9.351e-03 | 1237, 7.761e-03, 40.87, 2.41, 0.93 | 22.46 |
| AP3 | 0.267 | 132 / 660 | 84 (64%) | 6 of 6 | 1.635e-02 | 1433, 1.127e-02, 89.65, 5.23, 0.92 | 42.12 |
| AP4 | 0.271 | 132 / 660 | 78 (59%) | 6 of 6 | 1.127e-02 | 2098, 1.127e-02, 85.23, 3.10, 0.87 | 35.16 |
| AP5 | 0.341 | 132 / 660 | 55 (42%) | 6 of 6 | 6.442e-03 | 895, 6.442e-03, 46.92, 2.21, 0.88 | 27.87 |
| GM1 | 0.544 | 132 / 660 | 65 (49%) | 6 of 6 | 7.761e-03 | 559, 9.351e-03, 30.85, 1.04, 0.78 | 24.69 |
| GM2 | 0.540 | 132 / 660 | 78 (59%) | 6 of 6 | 1.127e-02 | 1638, 1.357e-02, 31.11, 1.39, 0.73 | 26.07 |
| GM3 | 0.264 | 132 / 660 | 88 (67%) | 6 of 6 | 1.635e-02 | 198, 1.357e-02, 48.36, 1.34, 0.94 | 33.03 |
| GM4 | 0.634 | 132 / 660 | 78 (59%) | 6 of 6 | 1.127e-02 | 1382, 1.127e-02, 65.69, 1.14, 0.78 | 40.74 |
| GM5 | 0.571 | 132 / 660 | 67 (51%) | 6 of 6 | 9.351e-03 | 658, 7.761e-03, 38.01, 1.02, 0.82 | 32.21 |

## D4. Section 3.2: the linear joint profile over C at each alpha

Summary run `20261113T054020Z_slope_threshold_summary_10tiles`. C in m^2 (at alpha = 2, A S^2 in m^2). Hold-out = mean of the two half-on-half scores at each half's own best C.

| tile | alpha | C | K (1/yr) | D (m^2/yr) | D/D0_A | MSE | channel cells | 1 % interval of C | held-out MSE | C fit on half 1 / half 2 | run |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | 0 | 6889 | 3.995e-06 | 1.699e-02 | 4.74 | 22.384 | 4887 | 6540-6930 | 23.405 | 6889 / 6587 | `20261113T014925Z_slope_threshold_AP1_a0` |
| AP1 | 0.5 | 3941 | 4.017e-06 | 1.776e-02 | 4.95 | 22.135 | 4364 | 3812-4217 | 24.903 | 4196 / 3812 | `20261113T014925Z_slope_threshold_AP1_a0.5` |
| AP1 | 1 | 2134 | 3.986e-06 | 1.844e-02 | 5.14 | 21.739 | 4376 | 2101-2169; 2288-2332 | 27.969 | 2134 / 3644 | `20261113T014926Z_slope_threshold_AP1_a1` |
| AP1 | 1.5 | 963.6 | 3.958e-06 | 1.746e-02 | 4.87 | 22.927 | 5283 | 939-968.3; 978.9-985; 994.6-1008 | 27.753 | 955 / 1716 | `20261113T014925Z_slope_threshold_AP1_a1.5` |
| AP1 | 2 | 674 | 3.942e-06 | 1.857e-02 | 5.18 | 23.811 | 4806 | 621.6-714.6 | 28.874 | 674 / 792.3 | `20261113T014925Z_slope_threshold_AP1_a2` |
| AP1 | 2.5 | 413.8 | 3.965e-06 | 1.844e-02 | 5.14 | 25.441 | 5070 | 369.8-371.5; 373-418.7 | 37.526 | 413.8 / 698.7 | `20261113T014925Z_slope_threshold_AP1_a2.5` |
| AP2 | 0 | 3121 | 4.597e-06 | 1.680e-02 | 5.22 | 22.461 | 11024 | 2980-3032; 3074-3182; 3226-3294 | 80.522 | 3121 / 1139 | `20261113T014925Z_slope_threshold_AP2_a0` |
| AP2 | 0.5 | 1483 | 4.569e-06 | 1.649e-02 | 5.13 | 21.609 | 12209 | 1469-1483 | 79.210 | 1483 / 726.9 | `20261113T014925Z_slope_threshold_AP2_a0.5` |
| AP2 | 1 | 802.4 | 4.636e-06 | 1.573e-02 | 4.89 | 22.392 | 13083 | 789.3-803.9; 3534-3714; 3770-3770 | 150.135 | 3553 / 136.8 | `20261113T014920Z_slope_threshold_AP2_a1` |
| AP2 | 1.5 | 2282 | 5.588e-06 | 2.695e-02 | 8.38 | 22.359 | 4162 | 2264-2282; 2293-2302; 2310-2353 | 146.260 | 2282 / 99.81 | `20261113T014920Z_slope_threshold_AP2_a1.5` |
| AP2 | 2 | 1554 | 5.762e-06 | 2.690e-02 | 8.36 | 22.927 | 3904 | 1459-1471; 1517-1571 | 149.618 | 1554 / 64.02 | `20261113T014920Z_slope_threshold_AP2_a2` |
| AP2 | 2.5 | 333.7 | 4.792e-06 | 1.962e-02 | 6.10 | 25.463 | 8384 | 328.8-335.6 | 158.591 | 333.7 / 49.89 | `20261113T014920Z_slope_threshold_AP2_a2.5` |
| AP3 | 0 | 4742 | 4.825e-06 | 1.791e-02 | 8.31 | 42.122 | 7246 | 4623-4689; 4722-4824; 5597-5676 | 40.464 | 5658 / 4030 | `20261113T014920Z_slope_threshold_AP3_a0` |
| AP3 | 0.5 | 2539 | 4.675e-06 | 1.840e-02 | 8.54 | 40.115 | 6390 | 2502-2561 | 54.522 | 2539 / 653.5 | `20261113T014920Z_slope_threshold_AP3_a0.5` |
| AP3 | 1 | 1262 | 4.523e-06 | 1.801e-02 | 8.36 | 42.027 | 6924 | 1240-1262 | 48.207 | 1262 / 568.5 | `20261113T014920Z_slope_threshold_AP3_a1` |
| AP3 | 1.5 | 892.2 | 4.488e-06 | 1.955e-02 | 9.07 | 43.604 | 6240 | 582.3-583; 589.9-596.8; 602.2-602.2 ... | 55.441 | 906.7 / 310.1 | `20261113T014920Z_slope_threshold_AP3_a1.5` |
| AP3 | 2 | 314.5 | 4.742e-06 | 1.417e-02 | 6.57 | 44.159 | 11139 | 313.9-315 | 60.917 | 697.3 / 211.4 | `20261113T014920Z_slope_threshold_AP3_a2` |
| AP3 | 2.5 | 273.4 | 4.594e-06 | 1.793e-02 | 8.32 | 44.450 | 7810 | 193.5-193.5; 273.4-273.8; 388.1-425.7 | 63.459 | 418.6 / 131.5 | `20261113T014920Z_slope_threshold_AP3_a2.5` |
| AP4 | 0 | 7232 | 4.522e-06 | 1.974e-02 | 5.44 | 35.156 | 4346 | 7011-7384; 7643-7886 | 42.919 | 7857 / 2241 | `20261113T014920Z_slope_threshold_AP4_a0` |
| AP4 | 0.5 | 2416 | 4.310e-06 | 1.796e-02 | 4.95 | 34.280 | 5871 | 1840-1849; 1855-1859; 1911-1914 ... | 38.305 | 2416 / 1327 | `20261113T014920Z_slope_threshold_AP4_a0.5` |
| AP4 | 1 | 1503 | 4.356e-06 | 1.838e-02 | 5.07 | 32.790 | 5433 | 1499-1574 | 34.667 | 1518 / 788.9 | `20261113T014920Z_slope_threshold_AP4_a1` |
| AP4 | 1.5 | 862.4 | 4.352e-06 | 1.830e-02 | 5.04 | 33.037 | 5586 | 849.7-864.9 | 35.032 | 863.2 / 472.1 | `20261113T014920Z_slope_threshold_AP4_a1.5` |
| AP4 | 2 | 370.4 | 4.269e-06 | 1.760e-02 | 4.85 | 34.420 | 7961 | 329.2-330.6; 338.5-374; 377-377 ... | 40.803 | 370.4 / 274.7 | `20261113T014920Z_slope_threshold_AP4_a2` |
| AP4 | 2.5 | 310.4 | 4.415e-06 | 1.873e-02 | 5.16 | 33.751 | 5804 | 308.3-316.2 | 42.236 | 310.4 / 181.2 | `20261113T014922Z_slope_threshold_AP4_a2.5` |
| AP5 | 0 | 1887 | 5.119e-06 | 1.258e-02 | 4.31 | 27.869 | 13040 | 1802-1938; 1971-1978 | 62.609 | 1901 / 3857 | `20261113T014922Z_slope_threshold_AP5_a0` |
| AP5 | 0.5 | 987.8 | 5.109e-06 | 1.205e-02 | 4.12 | 26.994 | 13166 | 886.3-886.3; 897.7-946.5; 950.9-950.9 ... | 59.988 | 925.2 / 1887 | `20261113T014923Z_slope_threshold_AP5_a0.5` |
| AP5 | 1 | 601.3 | 5.131e-06 | 1.220e-02 | 4.18 | 26.179 | 12665 | 593.4-613.3 | 61.813 | 601.3 / 1101 | `20261113T014922Z_slope_threshold_AP5_a1` |
| AP5 | 1.5 | 335 | 5.193e-06 | 1.176e-02 | 4.03 | 26.048 | 13412 | 306.8-323.5; 326.7-346.1 | 61.499 | 316.9 / 707.6 | `20261113T014922Z_slope_threshold_AP5_a1.5` |
| AP5 | 2 | 195.7 | 5.224e-06 | 1.172e-02 | 4.01 | 25.830 | 13930 | 192.6-195.8; 250.7-251.6 | 60.120 | 195.7 / 550.6 | `20261113T014922Z_slope_threshold_AP5_a2` |
| AP5 | 2.5 | 155 | 5.193e-06 | 1.334e-02 | 4.57 | 26.485 | 11236 | 149.8-158.1; 177.7-179.1 | 55.892 | 150.6 / 246.7 | `20261113T014922Z_slope_threshold_AP5_a2.5` |
| GM1 | 0 | 1241 | 4.865e-05 | 1.948e-02 | 2.16 | 24.690 | 7880 | 1133-1268 | 28.062 | 1774 / 1138 | `20261113T014925Z_slope_threshold_GM1_a0` |
| GM1 | 0.5 | 1140 | 4.758e-05 | 2.225e-02 | 2.47 | 26.489 | 6332 | 811-811; 1113-1123; 1128-1143 | 32.769 | 1140 / 680.8 | `20261113T014918Z_slope_threshold_GM1_a0.5` |
| GM1 | 1 | 748 | 4.739e-05 | 2.277e-02 | 2.53 | 27.776 | 6120 | 696.1-720.4; 723.7-778.2 | 37.176 | 776 / 115.9 | `20261113T014918Z_slope_threshold_GM1_a1` |
| GM1 | 1.5 | 142.3 | 4.679e-05 | 1.927e-02 | 2.14 | 28.676 | 17246 | 142.1-146.2; 146.9-147.6; 152.7-153.2 | 39.920 | 173.4 / 84.64 | `20261113T015006Z_slope_threshold_GM1_a1.5` |
| GM1 | 2 | 82.75 | 4.728e-05 | 1.921e-02 | 2.13 | 29.886 | 22289 | 72.27-73; 73.97-77.05; 80.61-87.08 ... | 39.793 | 115.8 / 62.57 | `20261113T015006Z_slope_threshold_GM1_a2` |
| GM1 | 2.5 | 63.15 | 4.695e-05 | 2.093e-02 | 2.33 | 30.817 | 21565 | 40.04-44.33; 52.85-55.63; 55.8-55.97 ... | 41.310 | 94.56 / 41.32 | `20261113T015006Z_slope_threshold_GM1_a2.5` |
| GM2 | 0 | 3509 | 5.515e-05 | 2.743e-02 | 2.81 | 26.071 | 6481 | 3358-3539 | 29.998 | 3509 / 1637 | `20261113T015006Z_slope_threshold_GM2_a0` |
| GM2 | 0.5 | 2227 | 5.531e-05 | 2.799e-02 | 2.87 | 26.682 | 5969 | 2161-2256; 2271-2271 | 30.865 | 2227 / 1153 | `20261113T015006Z_slope_threshold_GM2_a0.5` |
| GM2 | 1 | 1433 | 5.587e-05 | 2.904e-02 | 2.97 | 28.939 | 5606 | 1414-1436 | 33.733 | 1433 / 812.4 | `20261113T015006Z_slope_threshold_GM2_a1` |
| GM2 | 1.5 | 1056 | 5.631e-05 | 3.382e-02 | 3.46 | 34.742 | 4629 | 1026-1026; 1040-1074 | 41.849 | 1026 / 97.95 | `20261113T014918Z_slope_threshold_GM2_a1.5` |
| GM2 | 2 | 267.5 | 5.108e-05 | 2.685e-02 | 2.75 | 37.553 | 8057 | 262.2-262.7; 264.4-276.4 | 41.182 | 267.5 / 62.36 | `20261113T014918Z_slope_threshold_GM2_a2` |
| GM2 | 2.5 | 177.4 | 5.078e-05 | 2.744e-02 | 2.81 | 38.521 | 8020 | 165.4-166.3; 173.5-175.7; 176.3-179.5 | 42.605 | 165.8 / 45.33 | `20261113T014918Z_slope_threshold_GM2_a2.5` |
| GM3 | 0 | 919 | 6.791e-05 | 1.977e-02 | 1.95 | 33.030 | 13130 | 723-728; 733-745; 830-937 | 52.654 | 1546 / 756 | `20261113T015003Z_slope_threshold_GM3_a0` |
| GM3 | 0.5 | 270.4 | 6.731e-05 | 1.853e-02 | 1.83 | 35.679 | 17169 | 175-176.1; 188.3-188.3; 255.5-255.5 ... | 61.360 | 94.12 / 124.3 | `20261113T015003Z_slope_threshold_GM3_a0.5` |
| GM3 | 1 | 62.5 | 6.659e-05 | 2.447e-02 | 2.42 | 32.357 | 44377 | 62.47-65.78 | 60.434 | 62.5 / 47.87 | `20261113T015025Z_slope_threshold_GM3_a1` |
| GM3 | 1.5 | 36.75 | 6.548e-05 | 3.318e-02 | 3.28 | 29.593 | 46272 | 35.42-35.83; 36.08-37.93; 38.58-38.8 ... | 57.217 | 43.77 / 36.5 | `20261113T015024Z_slope_threshold_GM3_a1.5` |
| GM3 | 2 | 26.64 | 6.412e-05 | 3.789e-02 | 3.75 | 27.820 | 40103 | 23.83-24.01; 24.14-24.5; 24.63-27.49 ... | 54.242 | 26.84 / 25.21 | `20261113T015024Z_slope_threshold_GM3_a2` |
| GM3 | 2.5 | 17.28 | 6.406e-05 | 3.956e-02 | 3.91 | 27.721 | 41436 | 16.73-16.73; 16.97-18.34 | 55.789 | 17.28 / 24.02 | `20261113T015024Z_slope_threshold_GM3_a2.5` |
| GM4 | 0 | 3853 | 3.640e-05 | 2.189e-02 | 2.22 | 40.743 | 6154 | 3258-3339; 3643-3894 | 61.209 | 3853 / 4002 | `20261113T014915Z_slope_threshold_GM4_a0` |
| GM4 | 0.5 | 2400 | 3.627e-05 | 2.191e-02 | 2.22 | 39.652 | 5913 | 2350-2401 | 58.251 | 2400 / 2383 | `20261113T014910Z_slope_threshold_GM4_a0.5` |
| GM4 | 1 | 1680 | 3.603e-05 | 2.326e-02 | 2.36 | 42.372 | 5606 | 1543-1547; 1604-1604; 1632-1638 ... | 59.702 | 2033 / 1604 | `20261113T014910Z_slope_threshold_GM4_a1` |
| GM4 | 1.5 | 1201 | 3.626e-05 | 2.457e-02 | 2.49 | 44.674 | 5468 | 1178-1206; 1212-1219 | 59.099 | 1201 / 1166 | `20261113T015026Z_slope_threshold_GM4_a1.5` |
| GM4 | 2 | 745.2 | 3.562e-05 | 2.538e-02 | 2.58 | 51.958 | 5617 | 717.3-745.2 | 63.398 | 745.2 / 782 | `20261113T015043Z_slope_threshold_GM4_a2` |
| GM4 | 2.5 | 176 | 3.542e-05 | 1.699e-02 | 1.72 | 58.282 | 9236 | 172.6-176; 177.3-177.3 | 101.527 | 176 / 35.96 | `20261113T015026Z_slope_threshold_GM4_a2.5` |
| GM5 | 0 | 2265 | 5.267e-05 | 1.976e-02 | 2.60 | 32.211 | 9915 | 1820-1930; 2202-2343; 3098-3123 | 60.957 | 3104 / 1561 | `20261113T015026Z_slope_threshold_GM5_a0` |
| GM5 | 0.5 | 1418 | 5.224e-05 | 2.068e-02 | 2.73 | 32.981 | 8919 | 1409-1429; 1434-1459; 1931-1931 ... | 59.983 | 1973 / 879.1 | `20261113T015026Z_slope_threshold_GM5_a0.5` |
| GM5 | 1 | 710.5 | 5.189e-05 | 2.018e-02 | 2.66 | 33.699 | 9160 | 683.3-725.9 | 64.617 | 1502 / 710.5 | `20261113T015026Z_slope_threshold_GM5_a1` |
| GM5 | 1.5 | 471.6 | 5.170e-05 | 2.104e-02 | 2.77 | 37.357 | 8987 | 232.6-235.9; 455.2-468; 471.6-488.8 ... | 76.895 | 998.8 / 184.3 | `20261113T014925Z_slope_threshold_GM5_a1.5` |
| GM5 | 2 | 184.3 | 5.057e-05 | 1.951e-02 | 2.57 | 38.752 | 12306 | 179-184.3 | 82.684 | 636 / 114 | `20261113T014926Z_slope_threshold_GM5_a2` |
| GM5 | 2.5 | 102.2 | 4.978e-05 | 2.079e-02 | 2.74 | 40.704 | 14641 | 101-107.4; 107.9-109.2; 110-110 | 78.877 | 420 / 76.35 | `20261113T014925Z_slope_threshold_GM5_a2.5` |

## D5. Section 3.3: the hold-out rule, the split, and the alpha = 0 check against Phase 6

| tile | held-out alpha 0 | best alpha > 0 | held-out there | improvement | preferred (> 2 %) | best in-sample alpha | in-sample improvement | basins | cells half 1 / half 2 / none | alpha 0 vs Phase 6: common C, max rel diff MSE |
|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | 23.405 | 0.5 | 24.903 | -6.4% | False | 1 | +2.9% | 92 | 792537 / 147095 / 90593 | 1061, 2.7e-11 |
| AP2 | 80.522 | 0.5 | 79.210 | +1.6% | False | 0.5 | +3.8% | 112 | 968028 / 31583 / 30614 | 1029, 3.0e-12 |
| AP3 | 40.464 | 1 | 48.207 | -19.1% | False | 0.5 | +4.8% | 113 | 855735 / 83747 / 89728 | 1124, 1.2e-11 |
| AP4 | 42.919 | 1 | 34.667 | +19.2% | True | 1 | +6.7% | 93 | 737002 / 213401 / 79822 | 1175, 4.3e-12 |
| AP5 | 62.609 | 2.5 | 55.892 | +10.7% | True | 2 | +7.3% | 70 | 641658 / 298735 / 89832 | 1205, 3.5e-12 |
| GM1 | 28.062 | 0.5 | 32.769 | -16.8% | False | 0 | +0.0% | 43 | 618634 / 271753 / 109613 | 1189, 0.0e+00 |
| GM2 | 29.998 | 0.5 | 30.865 | -2.9% | False | 0 | +0.0% | 48 | 648122 / 269395 / 82483 | 1244, 0.0e+00 |
| GM3 | 52.654 | 2 | 54.242 | -3.0% | False | 2.5 | +16.1% | 43 | 593811 / 342852 / 63337 | 1352, 0.0e+00 |
| GM4 | 61.209 | 0.5 | 58.251 | +4.8% | True | 0.5 | +2.7% | 49 | 828756 / 88246 / 82998 | 1270, 0.0e+00 |
| GM5 | 60.957 | 0.5 | 59.983 | +1.6% | False | 0 | +0.0% | 53 | 199024 / 731528 / 69448 | 1427, 5.3e-12 |

| site | tiles preferring alpha > 0 | preferred at the site |
|---|---|---|
| AP | 2 of 5 | **False** |
| GM | 1 of 5 | **False** |

## D6. Phase 7: ported analyses against the feature-branch outputs

Run `20261113T024349Z_phase7_port_check_all`. GM grids are identical in both (metres); AP were in US survey feet on the branches and are in metres here.

| analysis | site | points new / branch | matched | quantity | identical (of matched) | median new | median branch |
|---|---|---|---|---|---|---|---|
| width | GM | 48 / 48 | 48 | W_primary | 48 | 6.75 | 6.75 |
| width | GM | 48 / 48 | 48 | W_inner_bankbreak | 48 | 7.5 | 7.5 |
| width | GM | 48 / 48 | 48 | W_stage_0.5 | 48 | 4.814 | 4.814 |
| width | GM | 48 / 48 | 48 | local_slope | 48 | 0.07371 | 0.07371 |
| width | GM | 48 / 48 | 48 | inner_relief_m | 48 | 1.353 | 1.353 |
| width | AP | 46 / 47 | 35 | W_primary | 13 | 6 | 8.125 |
| width | AP | 46 / 47 | 35 | W_inner_bankbreak | 16 | 8.25 | 9 |
| width | AP | 46 / 47 | 35 | W_stage_0.5 | 0 | 11.53 | 4.426 |
| width | AP | 46 / 47 | 35 | local_slope | 0 | 0.05495 | 0.1806 |
| width | AP | 46 / 47 | 35 | inner_relief_m | 3 | 0.6303 | 1.223 |
| width_constrained | GM | 50 / 50 | 50 | W_trench | 50 | 6.5 | 6.5 |
| width_constrained | GM | 50 / 50 | 50 | W_bed | 50 | 4.744 | 4.744 |
| width_constrained | GM | 50 / 50 | 50 | trench_depth_m | 50 | 1.374 | 1.374 |
| width_constrained | GM | 50 / 50 | 50 | local_slope | 50 | 0.04657 | 0.04657 |
| width_constrained | AP | 34 / 35 | 32 | W_trench | 10 | 7.938 | 5.617 |
| width_constrained | AP | 34 / 35 | 32 | W_bed | 0 | 9.154 | 2.764 |
| width_constrained | AP | 34 / 35 | 32 | trench_depth_m | 10 | 0.8616 | 1.647 |
| width_constrained | AP | 34 / 35 | 32 | local_slope | 0 | 0.0244 | 0.07772 |

| width fit | site | variant | branch c, b, n | new c, b, n |
|---|---|---|---|---|
| Phase 3 Tobit | GM | all | 29, -0.099, 31 | 29, -0.099, 31 |
| Phase 3 Tobit | GM | no_steep | 32.6, -0.108, 29 | 32.6, -0.108, 29 |
| Phase 3 Tobit | AP | all | 12.2, -0.021, 36 | 8.21, 0.016, 19 |
| Phase 3 Tobit | AP | no_steep | 8.9, 0.002, 27 | 8.21, 0.016, 19 |
| constrained Tobit | GM | default window | 1.87, 0.093 [-0.017, 0.177], 29 resolved | 1.87, 0.093 [-0.015, 0.179], 29 resolved |
| constrained Tobit | AP | default window | 1.28, 0.116 [0.010, 0.183], 22 resolved | 2.02, 0.121 [-0.015, 0.354], 8 resolved |

Constrained classes GM: branch {'inner_channel': 29, 'ambiguous': 9, 'unresolved_wide': 9, 'wide_or_terrace': 3}; new {'inner_channel': 29, 'ambiguous': 9, 'unresolved_wide': 9, 'wide_or_terrace': 3}.

Constrained classes AP: branch {'one_bank': 12, 'ambiguous': 12, 'inner_channel': 10, 'wide_or_terrace': 1}; new {'ambiguous': 18, 'inner_channel': 8, 'wide_or_terrace': 6, 'unresolved_wide': 2}.

Head extraction (Method P, kt = 0.1), retained heads branch / new, heads on the same pixel, hollow-class heads branch / new, hollow median A (m^2) branch / new:

| tile | OWF | OWF + DT0.5 | OWF + DT2 | OWF + DT5 |
|---|---|---|---|---|
| AP1 | 142/87, 5, 56/24, 3/16 | 45/41, 1, 13/27, 6/11 | 22/12, 0, 16/12, 350/1457 | 10/8, 0, 10/8, 757/19 |
| AP2 | 80/23, 0, 46/10, 4/10 | 18/17, 0, 7/16, 8/384 | 13/16, 0, 11/16, 39/3286 | 13/6, 0, 13/6, 329/142 |
| AP3 | 276/38, 3, 105/7, 4/5 | 87/19, 0, 27/15, 7/47 | 16/8, 0, 15/8, 29/2852 | 6/3, 0, 5/3, 7/3317 |
| AP4 | 132/68, 1, 39/19, 3/45 | 62/32, 0, 16/26, 41/157 | 24/30, 0, 18/26, 169/3359 | 23/13, 0, 22/13, 74/8684 |
| AP5 | 104/63, 1, 33/27, 3/41 | 52/31, 0, 12/22, 5/60 | 33/27, 0, 27/26, 59/2460 | 27/9, 0, 26/9, 26/5322 |
| GM1 | 53/53, 53, 34/34, 104/104 | 24/24, 24, 24/24, 108/108 | 16/16, 16, 16/16, 1242/1242 | 11/11, 11, 11/11, 5273/5273 |
| GM2 | 47/47, 47, 34/34, 71/71 | 22/22, 22, 19/19, 84/84 | 11/11, 11, 11/11, 417/417 | 11/11, 11, 10/10, 1804/1804 |
| GM3 | 46/46, 46, 31/31, 71/71 | 29/29, 29, 23/23, 266/266 | 15/15, 15, 14/14, 372/372 | 3/3, 3, 3/3, 1402/1402 |
| GM4 | 43/43, 43, 26/26, 59/59 | 20/20, 20, 18/18, 83/83 | 13/13, 13, 13/13, 176/176 | 6/6, 6, 5/5, 7173/7173 |
| GM5 | 48/48, 48, 38/38, 76/76 | 37/37, 37, 32/32, 107/107 | 20/20, 20, 20/20, 444/444 | 12/12, 12, 12/12, 828/828 |

## D7. Phase 7 results on the metre grids

Width `20261113T015005Z_width_AP_GM_full`, constrained `20261113T015155Z_width_constrained_full`, heads `20261113T022845Z_heads_10tiles` (outputs and figures under each run's `outputs/` or `head_extraction/`).

| site | measure | > 1 m | > 2 m | > 5 m | > 10 m |
|---|---|---|---|---|---|
| AP (n=34) | bed | 0.41 [0.25, 0.59] | 0.41 [0.25, 0.59] | 0.26 [0.13, 0.44] | 0.09 [0.02, 0.24] |
| AP (n=34) | trench | 0.41 [0.25, 0.59] | 0.41 [0.25, 0.59] | 0.41 [0.25, 0.59] | 0.21 [0.09, 0.38] |
| GM (n=50) | bed | 0.64 [0.49, 0.77] | 0.64 [0.49, 0.77] | 0.10 [0.03, 0.22] | 0.00 [0.00, 0.07] |
| GM (n=50) | trench | 0.64 [0.49, 0.77] | 0.64 [0.49, 0.77] | 0.56 [0.41, 0.70] | 0.06 [0.01, 0.17] |

| tile | smoothing | retained | hollow | valley_floor | bench_plateau | rill | road_ditch | unclassified | hollow A median (m^2) | hollow S_up_10 median |
|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | OWF | 87 | 24 | 0 | 0 | 0 | 11 | 52 | 15.69 | 0.242 |
| AP1 | OWF+DT0.5 | 41 | 27 | 0 | 1 | 0 | 0 | 13 | 11.41 | 0.311 |
| AP1 | OWF+DT2 | 12 | 12 | 0 | 0 | 0 | 0 | 0 | 1457 | 0.361 |
| AP1 | OWF+DT5 | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 19.49 | 0.438 |
| AP2 | OWF | 23 | 10 | 0 | 0 | 0 | 0 | 13 | 9.507 | 0.359 |
| AP2 | OWF+DT0.5 | 17 | 16 | 0 | 0 | 0 | 0 | 1 | 384.1 | 0.348 |
| AP2 | OWF+DT2 | 16 | 16 | 0 | 0 | 0 | 0 | 0 | 3286 | 0.315 |
| AP2 | OWF+DT5 | 6 | 6 | 0 | 0 | 0 | 0 | 0 | 141.7 | 0.396 |
| AP3 | OWF | 38 | 7 | 3 | 0 | 0 | 0 | 28 | 4.753 | nan |
| AP3 | OWF+DT0.5 | 19 | 15 | 1 | 0 | 0 | 0 | 3 | 46.58 | 0.303 |
| AP3 | OWF+DT2 | 8 | 8 | 0 | 0 | 0 | 0 | 0 | 2852 | 0.444 |
| AP3 | OWF+DT5 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 3317 | 0.395 |
| AP4 | OWF | 68 | 19 | 2 | 1 | 0 | 9 | 37 | 44.68 | 0.278 |
| AP4 | OWF+DT0.5 | 32 | 26 | 0 | 0 | 0 | 0 | 6 | 156.9 | 0.327 |
| AP4 | OWF+DT2 | 30 | 26 | 0 | 0 | 0 | 0 | 4 | 3359 | 0.298 |
| AP4 | OWF+DT5 | 13 | 13 | 0 | 0 | 0 | 0 | 0 | 8684 | 0.255 |
| AP5 | OWF | 63 | 27 | 0 | 0 | 0 | 1 | 35 | 40.88 | 0.181 |
| AP5 | OWF+DT0.5 | 31 | 22 | 0 | 0 | 0 | 0 | 9 | 59.89 | 0.245 |
| AP5 | OWF+DT2 | 27 | 26 | 0 | 0 | 0 | 0 | 1 | 2460 | 0.295 |
| AP5 | OWF+DT5 | 9 | 9 | 0 | 0 | 0 | 0 | 0 | 5322 | 0.303 |
| GM1 | OWF | 53 | 34 | 0 | 0 | 0 | 1 | 18 | 104.5 | 0.437 |
| GM1 | OWF+DT0.5 | 24 | 24 | 0 | 0 | 0 | 0 | 0 | 108.5 | 0.496 |
| GM1 | OWF+DT2 | 16 | 16 | 0 | 0 | 0 | 0 | 0 | 1242 | 0.447 |
| GM1 | OWF+DT5 | 11 | 11 | 0 | 0 | 0 | 0 | 0 | 5273 | 0.296 |
| GM2 | OWF | 47 | 34 | 0 | 0 | 0 | 2 | 11 | 71 | 0.470 |
| GM2 | OWF+DT0.5 | 22 | 19 | 0 | 0 | 0 | 0 | 3 | 84 | 0.508 |
| GM2 | OWF+DT2 | 11 | 11 | 0 | 0 | 0 | 0 | 0 | 417 | 0.591 |
| GM2 | OWF+DT5 | 11 | 10 | 1 | 0 | 0 | 0 | 0 | 1804 | 0.341 |
| GM3 | OWF | 46 | 31 | 0 | 3 | 0 | 0 | 12 | 71 | 0.268 |
| GM3 | OWF+DT0.5 | 29 | 23 | 0 | 1 | 0 | 0 | 5 | 266 | 0.259 |
| GM3 | OWF+DT2 | 15 | 14 | 0 | 0 | 0 | 0 | 1 | 371.5 | 0.281 |
| GM3 | OWF+DT5 | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 1402 | 0.487 |
| GM4 | OWF | 43 | 26 | 1 | 0 | 0 | 3 | 13 | 59 | 0.509 |
| GM4 | OWF+DT0.5 | 20 | 18 | 0 | 0 | 0 | 0 | 2 | 83 | 0.485 |
| GM4 | OWF+DT2 | 13 | 13 | 0 | 0 | 0 | 0 | 0 | 176 | 0.543 |
| GM4 | OWF+DT5 | 6 | 5 | 1 | 0 | 0 | 0 | 0 | 7173 | 0.364 |
| GM5 | OWF | 48 | 38 | 2 | 0 | 0 | 0 | 8 | 76.5 | 0.489 |
| GM5 | OWF+DT0.5 | 37 | 32 | 1 | 0 | 2 | 0 | 2 | 107 | 0.546 |
| GM5 | OWF+DT2 | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 444 | 0.632 |
| GM5 | OWF+DT5 | 12 | 12 | 0 | 0 | 0 | 0 | 0 | 827.5 | 0.598 |

## D8. Slurm accounting of the addendum 8 jobs

| job name | tasks | states | limit core-h | used core-h |
|---|---|---|---|---|
| a8_er_star | 1 | {'COMPLETED': 1} | 3.0 | 0.0 |
| a8_heads | 1 | {'COMPLETED': 1} | 12.0 | 0.5 |
| a8_nlj_AP | 30 | {'COMPLETED': 30} | 300.0 | 66.4 |
| a8_nlj_GM | 30 | {'COMPLETED': 30} | 300.0 | 57.9 |
| a8_nlj_merge_AP | 5 | {'COMPLETED': 5} | 5.0 | 0.7 |
| a8_nlj_merge_GM | 5 | {'COMPLETED': 5} | 5.0 | 0.5 |
| a8_phase7 | 1 | {'COMPLETED': 1} | 20.0 | 1.8 |
| a8_st_AP | 30 | {'COMPLETED': 30} | 180.0 | 84.6 |
| a8_st_GM | 30 | {'COMPLETED': 30} | 180.0 | 91.6 |
| a8_st_post | 1 | {'COMPLETED': 1} | 1.0 | 0.0 |
| a8_wait_all | 1 | {'COMPLETED': 1} | 0.0 | 0.0 |
| lemcvt_a8_tests0 | 1 | {'COMPLETED': 1} | 0.8 | 0.1 |
| lemcvt_a8_tests1 | 1 | {'COMPLETED': 1} | 1.0 | 0.1 |
| lemcvt_a8_tests2 | 1 | {'COMPLETED': 1} | 1.0 | 0.1 |
| lemcvt_a8_tests3 | 1 | {'COMPLETED': 1} | 0.7 | 0.0 |
| lemcvt_a8_tests4 | 1 | {'COMPLETED': 1} | 0.7 | 0.1 |
| total | | | 1010.2 | 304.5 |
