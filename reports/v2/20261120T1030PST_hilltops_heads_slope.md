# [2026-11-20 10:30 PST] Addendum 19: hilltop curvature on the fixed 10³ m² network, the measured D bracket, Steve's 64 TTV channel heads, and the area–slope threshold. Part A, stopped at the hilltop gate

Directive: `reports/directives/2026-11-20T0755PST_v2_addendum_19_hilltops_heads_slope.md` (the 07:55 PST revision; outline fixed by its
section 8). Branch `v2.3-tiles` (from c5901e1, the addendum 18 report). **No package code was changed.** Scripts (all `reports/v2/scripts/`,
ruff-clean): `a19_common.py` (decisions, rules, the hilltop set on a fixed network, the D8 tracer, the road buffer), `a19_roads.py` (TIGER
roads), `a19_heads.py` (the 64 picks on the full-site grids, the fits, D∞), `a19_hilltops.py` (Part A per tile), `a19_site.py` (site level,
rules R1, R11, R12, the brackets), `a19_samepixel.py`, `a19_qc.py` (figures), `a19_appendix_fix.py` (decision 8), `make_a19_tables.py`
(appendices); written for Part B but **not run** (section 1): `a19_fixed.py` (PF, PF3, SF; its gate-3 timing on TTV2 did run), `a19_submit_sj.py`,
`a19_dstat.py` (decision 7), `a19_modelheads.py` (decision 6), `a19_rules.py` (R2, R6–R10). Every number is tied to the run directory
`$GROUP_HOME/lemcvt_results/20261120T160830Z_a19_hilltops_heads_slope/` (`roads/`, `heads/`, `hilltops/`, `samepixel/`, `site/`, `fixed/`,
`figures/`, `appendix_fix/`) or to a Slurm job id (`$SCRATCH/lemcvt_jobs/a19/`, Appendix D12). Figures: `reports/v2/figures/a19/` and
`reports/v2/figures/a19/hilltop_qc/` (`README.md` there is the index). **This is a stop for review: rule R1 fails at every site, so the
directive's section 4.1 stops the work after Part A.** The appendices (section 12) are machine-generated.

## Summary

| item | result |
| --- | --- |
| 1 state, decisions | `v2.3-tiles` = origin = c5901e1; tests on `serc` 274 passed; every path present; **the companion CSV is `reports/directives/ttv_heads_manual_20261119T1911.csv`** (the directive names a different file): 64 rows, unique ids, all 64 inside the M&D polygon, pick 49 at 55 m from the traced channels (the planner's check holds). Ten deviations or choices, stated in section 1; none a stop. The appendix fix of decision 8 is `reports/v2/20261120T0841PST_expanded_fits_appendix_fix.md` |
| 2 D₀ on the 10³ network | The counted set is the main ridge crests; the 50-px **segment filter removes 60–80 % of the geometric set** (the D8 divide cells form fragments of a median 6–10 px along a crest, not ridges), so **R1 fails at every site: GM 18 of 22 tiles with ≥ 500 pixels (20 needed), AP 12 of 21 (19), TTV 8 of 11 (10)**. Without the segment filter 22/22, 21/21, 11/11 pass. Site medians D₀,N,A: GM 0.0129 (D₀,R 0.0149; ratio 0.91), AP 0.0035 (0.0041; 0.87), TTV 0.0046 (0.0027; **1.56**). The filters change D₀ by 7–14 % at GM and AP; the convex drop by 1–5 %. **L_w as written is 51 m on 37 of 38 usable primary tiles** (the crest width to the 10³ network is the hillslope length), so D₀,N,B is a 51 m curvature (AP, TTV 1.6 × D₀,N,A); with the rollover crest width (D₀,N,B′) it stays within 10 % of D₀,N,A. Same-pixel: GM S 0.99, P 1.13 (the 22 % FEMA deficit of addendum 18 is gone on this set), AP S 0.88, P 0.91: both "mixed" |
| 3 calibration, window | **R11: GM passes** (pooled \|C_ht\|(L_f) 0.0112 m⁻¹ = 0.95 [0.91, 0.99] × Perron's 0.0118); **AP fails** (0.0154 = 2.81 [2.65, 2.96] ×; in range at no window up to 51 m; the manuscript's approximate block-mean route gives 1.07 ×, GM 0.96 ×): no D statement at AP in this addendum. **R12: L_break = 5.3–6.1 m at every site** (not in 8–20 m; L_f = 11 m outside the 68 % band), D₀,N,S/D₀,N,A = 0.57–0.68 |
| 4 D bracket | D_R = E L_H²/(2R) on the 10³ network: GM 0.030 (D_R/D_C 2.0), AP 0.013 (3.3), TTV 0.015 (3.4); the normalised ratio 2S̄/(C L_H) 0.49 / 0.30 / 0.30 (addendum 8: 0.38–0.52 / 0.19–0.31). E* at S_c = 1.27: medians 1.7–2.5, 97–100 % below 10 (the curvature bias of Struble & Roering is small here); fitted S_c 0.38 / 0.29 / 0.28. Read from the interval figure (the addendum 18 fits; rule R10 itself not run): **P4's D is inside the bracket on 14 of 18 determined GM tiles, above on 11 of 12 at AP and 8 of 8 at TTV**; traced to P4's own channel network the TTV hillslopes are 150–380 m long and their D_R (0.018–0.066) matches the fitted D. TTV anchor: D₀,N,A = 1.9 × 2.5 × 10⁻³ |
| 5 heads | Table 1 reproduces M&D: a = 1,978 m², b = 1.647 (**R4 passes**; R² 0.60 against the paper's 0.75). The picks sit on hillslope cells (A at the pick cell: median 4 m²); **A_3 median 611 m²** (IQR 169–2,152), A_10 2,848; fit (ii): a = 64 m², b = 2.40 [1.35, 3.43], R² 0.22; **R5 fails** (b inside the 95 % band but the picks lie at 0.047 × M&D's relation). A·S² median 62–78 m² against Table 1's 1,500. A_F = 611 m², α = b̂ → 2.4, C_F(2.4) = 41 m² |
| 6–9 Part B | **Not run** (R1). Gate 3 timed: PF 4 s, SF 3–4 s, SJ 6.1 s per evaluation → 33 SJ runs ≈ 134 core-hours expected, 268 of limits; the whole of Part B ≈ 150 core-hours. On TTV2 (the gate tile) PF at A_F = 611 m² fits 2.2 × worse than P4 (MSE 130 against 60; AD 210) and at 7,400 m² 1.1 × worse (67) |
| 10–11 | What the evidence supports and the decisions needed: the segment filter's definition, the R11 failure at AP, L_w, the head snapping radius, and whether Part B (which does not depend on the hilltop set except for section 5.3) is run as prepared |
| cost | 8 submissions, 80 tasks, **5.7 core-hours used**, 66.5 of limits (expected ≤ 320); largest submission 36.5 core-hours of limits (the 73-tile array); peak memory 25.7 GB (the D∞ run); +0.17 GB on group storage (run directory 74 MB, TIGER roads 100 MB) |

## 1. State check, deviations, decisions as applied

**At the start (2026-11-20 07:58 PST, login node sh04-ln07).**
- `v2.3-tiles` = `origin/v2.3-tiles` = c5901e1; no job was running; `/etc/agents/AGENTS.md` read; the documents of section 0.2 read in order.
- Tests rerun on `serc` at c5901e1 (job 53094599): ruff clean, mypy clean, **274 passed**; for the state check, no package code changed.
- Every path of section 0.3 exists, with one difference: **Steve's heads file is `reports/directives/ttv_heads_manual_20261119T1911.csv`**, not
  `2026-11-19T1111PST_ttv_heads_manual_steve.csv`. It has the stated columns, 64 rows, ids 1–64 unique, `origin` "manual" and `uncertain` 0
  on every row, no notes; every point lies inside the `study_area` polygon of `md1989_fig3.gpkg`; the distance to the nearest traced M&D
  channel line is 6.2 m at the median, 14 m at the 95th percentile, and **55.0 m for pick 49** (the planner's check). Copied to
  `data/ttv/ttv_heads_manual_steve_20261119.csv` (sha256 in Appendix D1; `data/ttv/` is not ignored by `.gitignore`, `data/TTV/` is).

**Where the addendum and the repo or the data differ, and the choices made.** Nothing required a stop before the gate. Each is handled as
written and the alternative reported beside it, labelled:

1. **The hilltop gate R1 fails at every site (section 4.1.1), so the work stops after Part A.** Part B was not run. The sbatch files and
   scripts for Part B are written and tested on one tile (gate 3), and the Part B estimate is in section 6.
2. **L_w "measured to the 10³ network" (decision 1).** The crest width, 2 × the distance to the network, is then twice the hillslope length
   (median 100–200 m), so L_w is the largest tested window, 51 m, on 37 of the 38 usable primary tiles (TTV9: 15 m), and the window test
   runs over 11–51 m on every tile. D₀,N,B is computed as written; **D₀,N,B′**, with the crest width measured to the rollover network (the
   Part B meaning of the window test), is reported beside it, labelled, with its own window test.
3. **The convex drop (C_ht(L_f) ≥ 0 removed) is new against Part B**, which kept non-convex pixels in the median (frac_convex 0.87–0.96 at
   L_f). It is applied as written (it removes 1–5 % of the pixels and raises C_ht by 1–5 %); D₀,N without it is reported beside it.
4. **Filter order and the segment definition.** The geometric filters of Part B (divide cells between the outlet basins of the 10³ network,
   two cells from the edge, |∇z| ≤ 0.4, farther than L_f/2 from the network) come first, then the road buffer, then the segment filter on the
   8-connected components of what survives, then the convex drop (a value filter, last). The bootstrap segments are the 8-connected groups of
   the counted pixels. Because the outcome of R1 turns on this filter, the counted pixel count is also reported (report only, no rule) at
   thresholds of 0, 10, 20 and 50 px, and with the segment measured on the divide set before the geometric drops ("parent" segments, the
   crest's own length), Appendix D2–D3.
5. **The road and trail layer.** `$GROUP_HOME/lemcvt_data/source/` holds no road vector layer, so TIGER/Line 2024 ROADS (all MTFCC classes,
   including vehicular trails S1500, walkways S1710, bike and bridle paths) were fetched for the three counties the tiles touch: Monterey
   (GM, 18,200 features), Greene PA (AP, 5,704), Marin (TTV, 8,072); stored under `source/TIGER2024_roads/` with a sha256 manifest (D1).
   TIGER carries public roads and some trails; the fire roads of TTV11 and the farm tracks of GM are in it only where the Census has them.
   The buffer removes a median of 14 (GM), 131 (AP) and 672 (TTV) pixels per tile from the geometric set.
6. **D∞ on the full TTV grid (decision 3).** A pilot on TTV2 timed the package's D∞ routines at 35 µs per cell, so the full 19,010 × 19,010
   grid is 3.5 core-hours, under the 10 core-hour cap; it was run as written (job 53095652, 3 CPUs for 64 GB). Result in section 5.
7. **S_down on the full grid.** `slope_threshold.valley_slope_reach` is vectorised over every cell, which on the 361 M-cell grid would need
   tens of GB; the same definition (step along D8 receivers until the path length reaches 10 m; the drop over that length) is coded per pick
   in `a19_heads.py`. S_up follows the largest-area donor chain, as decision 3 says; one pick's chain ended short of 10 m (flagged).
8. **The segment bootstrap** uses 2,000 draws with seed 20261120 everywhere (decision 1), against the 200 of the package; the pooled
   Perron medians use 2,000 at L_f and at L_w and 400 at the other windows; the S_c fits 400 draws.
9. **The manuscript's D₀ route (decision 10)** exists in the package only as the labelled approximation `legacy_v1.legacy_curvature`
   (block-mean grid at 10 m GM / 30 m AP, coarse D8, divides between coarse basins, 3 × 3 Laplacian; the `topo_analysis` first-order-divide
   module is not available). It was run and is reported as that approximation, not rebuilt.
10. **The picks' D8 areas.** The picks were placed on the hillshade and sit on hillslope cells (the D8 area at the pick cell is 1–30 m² on
    three quarters of them); the 3 m search of decision 3 reaches the hollow axis on few of them (A_3 median 611 m² against A_10's 2,848 and
    Table 1's 7,400). Decision 3 is applied as written (A_3 primary); the median A_10 is carried as a labelled sensitivity through the field
    values (A_F,10 = 2,848 m²) and the gate-3 evaluations, so that the choice of radius can be made from numbers.

Also stated: the hilltop array's first version lacked the segment-filter sensitivities and was rerun before any site-level number was
produced (job 53095376 was the three-tile test; 53095730 the array of record); the per-tile QC figures failed once on a title format for the
tiles without a Part B D₀ and were regenerated (jobs 53096852, 53097281); the report draft lived in the repository, untracked.

### 1.1 The decisions of section 1 as applied
- **Decision 1 (D₀,N):** network A ≥ 10³ m² on each tile's registered area grid (GM1–GM5: the `area_model` old grids, as registered); hilltop
  pixels as above; curvature `total_curvature` at L_f = 11 m and at 17 windows (the Part B test set 7–51 m and the decision-11 scan 3–31 m);
  the tile value is the median over counted pixels with the segment bootstrap; D₀,R from the Part B `per_tile.csv`; D₀,N,unf = the geometric
  set with the convex drop but without the road and segment filters. Decision 4 of addendum 18 (no D₀ below 500 pixels) is applied to D₀,N.
  The Part B rollover set was rebuilt on every tile and its C_ht(L_f) equals the Part B value to 0.0 (73 of 73).
- **Decision 2:** the CSV copied and committed; every head statistic is given with and without pick 49.
- **Decision 3:** attributes on the stage B full-site grids (19,010 × 19,010, EPSG:6339, 1 m; areas = cells × 1 m²); A_3 (primary) and A_10;
  S_up at 10, 5 and 20 m; S_down at 10 m; the tile of each pick (TTV1 22, TTV2 34, TTV3 5, none 3); D∞ A_3 on the full grid.
- **Decisions 4–5:** fits (i)–(iv) by OLS on log₁₀ A with the 10,000-draw bootstrap (seed 20261120); α = b̂ rounded to 0.05 = 2.4; A_F and
  C_F computed (section 5); the fits of decision 5 not run (R1).
- **Decisions 6–7, 9:** not run (R1). The code for decision 7 recomputes P1 and P2 at D₀,N exactly from the Gram terms stored in every
  addendum 18 row (the rows hold G, g, rr and n for the whole tile and both halves); for A2 (AD with D fixed) the stored rows hold no Gram
  terms, but the MSE at a given r is a quadratic in U/D whose three coefficients are fixed by the stored values at the AD optimum, at A1 and at
  the old A2, so A2 at D₀,N can be reconstructed exactly on the stored grid and refined by Brent with fresh evaluations, as addendum 18 did;
  the route would be stated and checked (one direct evaluation per tile, tolerance 10⁻⁸) when it runs.
- **Decision 8:** the appendix fix is committed as `reports/v2/20261120T0841PST_expanded_fits_appendix_fix.md` (the whole addendum 18 report
  with C16 regenerated from the 98 + 98 run records: 98 distinct specs, the notes per spec, the 98-row checksum table; a header note states
  the change; the original is untouched). The bug was a stale loop variable in `make_a18_tables.py` (`specs.get(k)` after `for _k, r in …`)
  and a wrong root for the run records.
- **Decisions 10–14:** sections 3 and 4. **Decision 15:** 5.7 core-hours used, 66.5 of limits (Appendix D12).

## 2. D₀ on the fixed network (section 2; rule R1)

### 2.1 The counted set and the filters
`a19_hilltops.py` on the 73 measured tiles (array 53095730, 7–15 s per tile). On the 10³ network the divides between outlet basins are the
main ridge crests only; the geometric set (the Part B definition on this network) has a median of 4,280 (GM), 3,198 (AP) and 5,837 (TTV)
pixels per tile, against 920–8,600 on the rollover network. The road buffer removes 14 / 131 / 672 (medians), the convex drop 18 / 29 / 43,
and **the segment filter 2,664 / 2,104 / 3,771**: a crest line on a 1 m D8 grid is not a continuous line of divide cells (a crest cell that
receives flow from its neighbour along the crest is not a divide), so the 8-connected pieces have a median length of 6–10 px and most of
them fall under 50 px. The counted sets are therefore the few crest stretches long enough to survive: 280–2,833 pixels per GM tile,
84–1,686 per AP tile, 300–1,332 per TTV tile (Appendix D2). The QC figures (section 2.3) show the counted pixels on the crests, as sparse
segments. Measuring the segment on the divide set before the geometric drops ("parent" segments) changes the counts by a few per cent at AP and
TTV (12 and 8 usable, the same tiles) and lifts GM to 20 of 22 (Appendix D2–D3): the fragmentation is in the divide definition, not in
the filters.

**Rule R1** (at least 500 counted pixels on at least 90 % of the primary tiles, and the counted set on crests):

| site | tiles ≥ 500 px | needed | count part | below 500 | no segment filter | segments ≥ 20 px | segments ≥ 10 px |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GM | **18 of 22** | 20 | **fail** | GM5_fema (280), GM22 (469), GM24 (416), GM26 (445) | 22 pass | 21 pass | 22 pass |
| AP | **12 of 21** | 19 | **fail** | AP2_3dep (151), AP3_3dep (399), AP7, AP8, AP11 (472), AP18, AP19 (84), AP25, AP29 (463) | 21 pass | 17 fail | 20 pass |
| TTV | **8 of 11** | 10 | **fail** | TTV3 (323), TTV4 (458), TTV7 (300) | 11 pass | 11 pass | 11 pass |

The QC look (the second condition) passes: on the six comparison tiles per site (section 2.3) the counted pixels lie on the crest lines
between the 10³ basins, where the rollover set of Part B is a dense web of minor divides on the flanks (GM24, GM20) or a handful of pixels
(TTV4, TTV15). Two things to be aware of on the AP page: the counted stretches of AP18 and AP27 follow straight ridge-top lines that may be
tracks the TIGER layer does not carry, and AP19's set is 84 pixels in one spot. **R1 fails on the count at every site, and the directive
stops the work after Part A.** AP2_3dep, AP9, AP10, AP14, AP27 and
AP28 and TTV3 and TTV4, which had no D₀ in addendum 18, have 1,118–1,686 (AP9, AP10, AP14, AP27, AP28) or 151–458 pixels here; AP11, with no
rollover pixel at all, has 472.

### 2.2 D₀,N against D₀,R and the filters' effect
Site medians over the usable primary tiles (Appendix D3; per tile D2):

| site | N (usable) | C_ht(L_f) | **D₀,N,A** | D₀,R,A | N/R (A) | D₀,N,unf,A | N/unf | N/no-convex-drop | D₀,N,B (L_w) | D₀,N,B′ (L_w′) | D₀,N,S (L_S) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GM | 22 (18) | 0.0115 | **0.0129** | 0.0149 | 0.91 | 0.0151 | 0.93 | 0.99 | 0.0123 (51 m) | 0.0123 | 0.0084 (5 m) |
| AP | 21 (12) | 0.0157 | **0.0035** | 0.0041 | 0.87 | 0.0044 | 0.86 | 0.98 | 0.0057 (51 m) | 0.0043 | 0.0020 (5 m) |
| TTV | 11 (8) | 0.0167 | **0.0046** | 0.0027 | 1.56 | 0.0044 | 1.01 | 0.95 | 0.0075 (51 m; TTV9 15 m) | 0.0051 | 0.0029 (7 m) |

- **GM.** The crest set is 9 % more curved than the rollover set at the median (D₀,N,A/D₀,R,A 0.73–1.21 per tile) and the filters add 7 %
  (the segment filter keeps the longer, sharper crest stretches). The C_ht(L) curve of the counted set is flat from 7 to 51 m (the pooled
  ratio to Perron's value is 0.89–0.95 at every window from 7 m, Appendix D5), so D₀,N,B at 51 m is within 5 % of D₀,N,A and the window test
  as written passes on 5 of 18 tiles only because the ±10 % band is tight over 11–51 m; with the rollover width (L_w′ 11–51 m) 8 of 18 pass.
  The NCALM twins give D₀,N,A 0.0161 (D₀,R 0.0100): the 10³ crest sets of NCALM and FEMA now agree to 20 % (section 2.4).
- **AP.** D₀,N,A 0.0035 against D₀,R 0.0041 at the median, but the per-tile ratio runs from 0.21 (AP19, 84 px) to 2.2: on the 3DEP tiles
  whose rollover set had collapsed (AP9, AP10, AP14, AP27, AP28) the crest set is 1.5–2.2 × less curved than the few rollover pixels, and on
  AP6, AP8, AP13, AP24, AP29 it is 2–3 × more curved. The C_ht(L) curve keeps falling to 51 m (0.0154 at 11 m, 0.0098 at 51 m, Appendix D5),
  so D₀,N,B at 51 m is 1.6 × D₀,N,A and the window test fails on every tile; at L_w′ (11–25 m on 8 of 12) it passes on 5.
- **TTV.** The crest set is **less** curved than the rollover set (D₀,N,A 1.56 × D₀,R,A), most on the tiles where the rollover set had been
  tiny and steep (TTV4: 86 px at 0.088 m⁻¹ against 458 px at 0.0135; TTV15 3.96 ×; TTV6 2.84 ×); on TTV2, TTV7, TTV8 the two agree within
  10 %. The road buffer matters here (672 px per tile; TTV5 387 of 4,911, section 2.3).

### 2.3 QC figures
`reports/v2/figures/a19/hilltop_qc/` (`README.md`): 73 per-tile figures `<tile>_hilltop_qc.png` (classes relative to the 10³ network, with
two added classes, (vi) dropped by the segment filter and (vii) dropped by the road buffer; the histogram now shows the rollover set beside the
counted set), 12 pair figures, 5 gallery pages, the three comparison pages `<site>_rollover_vs_1e3_compare.png` (six tiles by rank of
D₀,N/D₀,R: GM24, GM20, GM17, GM11, GM9, GM26; AP19, AP8, AP7, AP3_3dep, AP18, AP27; TTV9, TTV7, TTV5, TTV11, TTV15, TTV4) and
`TTV5_roads.png`. What they show: the counted pixels are the crest lines of the ridges between the 10³ basins; the pink class (vi) is the
rest of the same crest lines in short pieces; the rollover set is the web of minor flank divides (its selection effect of addendum 18 §2.1
was this web); on TTV5 the Coastal Trail and the fire roads on the Mt Tamalpais flank run along the crests and the buffer removes the
rollover pixels on them (423 of TTV5's 3,203 rollover pixels lie inside the buffer).

### 2.4 Same-pixel diagnostic (rule R3)
`a19_samepixel.py` on the D₀,N counted sets of the 12 pairs (Appendix D4). Offsets 0.00 m (GM) and 0.4–0.7 m (AP) as in addendum 18.

| site | pairs | S (surface) | P (selection) | (d)/(a) | reading (addendum 18 §3.1.5) | addendum 18 on the rollover sets |
| --- | --- | --- | --- | --- | --- | --- |
| GM NCALM → FEMA | GM1–GM7 | **0.99** (0.88–1.10) | **1.13** (0.93–1.59) | 1.14 | **mixed** (P > 1.10) | selection (S 0.97, P 0.80) |
| AP PAMAP → 3DEP | AP1–AP5 | **0.88** (0.70–1.47) | **0.91** (0.79–1.40) | 0.82 | **mixed** | mixed (0.85, 0.69) |

On the 10³ crest sets the FEMA surface carries 99 % of the NCALM curvature on the same pixels and each survey's crest set has slightly
higher curvature than the other's mapped onto it (P 1.13): the 22 % FEMA deficit of Part B, which addendum 18 traced to the rollover
network's pixel selection, does not appear on this network. At AP the 3DEP surface carries 88 % of the PAMAP curvature (the PAMAP striping,
as before) and the selection ratio is 0.91 (pairs with fewer than 500 pixels on one side: AP2, AP3). The reading is "mixed" at both sites by
the rule's letter (P outside 0.90–1.10 at GM; S below 0.90 at AP), and the number that matters for the D result is that the survey no
longer selects a different crest set at GM.

## 3. Calibration and the window (sections 2.5–2.6; rules R11, R12)

### 3.1 Against Perron et al. (2012) (decision 10, rule R11)
Pooled counted pixels of the primary tiles (GM 28,664 px, AP 16,438), segment bootstrap over the pooled segments (Appendix D5; figures
`<site>_Cht_vs_L_perron.png`). E is the same as Perron's at both sites (147 and 55 m/Myr).

| site | pooled \|C_ht\|(L_f) | ratio to Perron | 68 % | at L_w (51 m) | windows in 0.67–1.5 | legacy (approximate manuscript route) | **R11** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GM | 0.0112 m⁻¹ | **0.95** | [0.91, 0.99] | 0.89 | every window 7–51 m (0.89–0.95); 3–5 m 1.0–2.5 | 0.0113 m⁻¹, 0.96 × | **pass** |
| AP | 0.0154 m⁻¹ | **2.81** | [2.65, 2.96] | 1.79 | none (1.79 at 51 m) | 0.0059 m⁻¹, 1.07 × | **fail** |

At GM the crest curvature of the 10³ set reproduces Perron's value at every window from 7 m, and the manuscript's coarse route gives the
same number. At AP the 11 m crest curvature is 2.8 × Perron's, it is still 1.8 × at 51 m, and only the 30 m block-mean route (the
manuscript's) lands on it (1.07 ×): Perron's AP value is a coarse-scale curvature of these ridges, which are sharper at the metre scale on
either survey (addendum 18 §2.1: 3DEP carries 68–94 % of PAMAP's 11 m curvature). **By R11 no D statement is made at AP in this addendum**;
the report says at which window the ratio would lie in range: none of the tested windows. TTV is not gated.

### 3.2 The window scaling break (decision 11, rule R12)
σ_C(L) on all cells inside the margin at L = 3–31 m, normalised at 11 m, pooled per site, two-segment continuous fit in log–log with the break
on a 0.1 m grid, 2,000 bootstrap draws over tiles (Appendix D6; figures `<site>_sigmaC_scaling_break.png`).

| site | L_break (m) | 68 % | log slopes below / above | L_S (m) | D₀,N,S | D₀,N,S/D₀,N,A | R12 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GM | 5.8 | [5.8, 5.9] | −2.29 / −0.56 | 5 | 0.0081 | 0.63 | not consistent |
| AP | 5.3 | [5.3, 5.3] | −1.96 / −1.00 | 5 | 0.0020 | 0.57 | not consistent |
| TTV | 6.1 | [6.1, 6.2] | −2.36 / −0.81 | 7 | 0.0029 | 0.68 | not consistent |

σ_C(L) falls as L⁻² from 3 to 5 m (metre-scale roughness and, at AP, the TIN facets) and more gently above; the only break a two-segment fit
can find is this roll-off at 5–6 m, well below the 8–20 m of Hurst et al. (2012) and Struble & Roering (2021), and the bootstrap over tiles
hardly moves it because every tile has the same shape. L_f = 11 m is therefore **not consistent with the scaling break** by R12's letter at
any site; what the curves say is that there is no break in 8–20 m to be consistent with, and that at 5–7 m the curvature is 1.5–1.8 × the
11 m value (D₀,N,S = 0.57–0.68 × D₀,N,A). L_f = 11 m stays primary, as decision 11 says. The D₀,N(L) galleries (`<site>_D0N_vs_L_gallery.png`)
give every tile's curve from 3 to 51 m with L_f, L_w and L_w′ marked.

## 4. The D bracket (section 2.7–2.8; decisions 12–14)

### 4.1 D_R, D_R/D_C and E*
Per counted pixel, L_H and R along the D8 path to the 10³ network (`a19_common.trace_to_mask`, the stepping of `er_star.trace_to_channel`);
D_R = E L_H²/(2R); segment bootstrap (Appendix D7; interval figures `<site>_D_bracket_intervals.png`). Every counted pixel reaches the network
(no path leaves a tile: the counted crests lie between 10³ basins that drain inside the tile).

| site (usable primary) | L_H median (m) | D_R median | D_R (Euclidean) | D_R/D_C median | 2S̄/(C L_H) | addendum 8 ratio (option (iii) set) | E* at 1.27: median; < 10 / 10–30 / > 30 | S_c fit [68 %] | E* at the fit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GM (18) | 99 | **0.030** | 0.023 | **2.0** (1.4–2.9) | 0.49 | 0.38–0.52 | 1.7; 1.00 / 0.00 / 0.00 | 0.38 [0.37, 0.39] | 5.6 |
| AP (12) | 109 | **0.013** | 0.011 | **3.3** (2.8–4.6) | 0.30 | 0.19–0.31 | 2.5; 0.97 / 0.03 / 0.00 | 0.29 [0.28, 0.30] | 10.8 |
| TTV (8) | 71 | **0.015** | 0.011 | **3.4** (2.4–3.6) | 0.30 | - | 1.9; 0.97 / 0.03 / 0.00 | 0.28 [0.27, 0.29] | 8.6 |

- D_R exceeds D_C by a factor 2 at GM and 3.3–3.4 at AP and TTV: the relief of these hillslopes is what a linear model needs 2–3 × the crest
  D to produce, the nonlinear signature of addendum 8 on a cleaner pixel set (the normalised ratio is the inverse of D_R/D_C).
- E* at S_c = 1.27 is below 10 on 97–100 % of the pixels, so the Struble & Roering (2021) curvature bias (10 % at E* = 10) is at most a few
  per cent here; at the fitted S_c (0.28–0.38, inside its own 68 % band at every site, and on the flat limb of the curve as in addendum 8)
  E* is 6–11 and 20–45 % of the pixels fall in 10–30. No correction is applied.
- The path-length sensitivity: with the Euclidean crest-to-channel distance in place of the D8 path length, D_R is 22–24 % lower at the median (0.023 / 0.011 / 0.011), which does not change the readings below.
- **The picked-head network (TTV1–TTV3, inside the M&D polygon):** 194 / 429 / 33 pixels reach the network of Steve's heads (101 / 103 / 1
  leave the tile first); L_H 202 / 153 / 156 m; D_R 0.032 / 0.017 / 0.024 against 0.021 / 0.014 / 0.019 on the 10³ network.
- **The model-network D_R (report only):** traced to the addendum 18 P4 channel set (A ≥ A_crit, 9,100–89,000 m²) the TTV hillslopes are
  157–353 m long (median 285) and D_R is 0.018–0.066 (median 0.045; TTV2 0.018, TTV1 0.051, TTV5 0.066); at GM (A_crit 650–15,200 m²) the P4
  hillslopes are 71–164 m (median 122) with D_R 0.020–0.061 (median 0.033); at AP 149–219 m (median 170), D_R 0.018–0.025 (median 0.021). **At every
  site the fitted P4 D is 0.84–0.89 × the D_R of its own network at the median**: a fit's D is the relief term of the hillslope length the fit
  itself creates. What the 10³-network bracket asks is whether that length is the landscape's; at TTV the fit's hillslopes are 4 × the 10³
  network's.

### 4.2 The bracket, read from the interval figure (rule R10 is Part B and was not run)
The bracket [D_C,lo, D_R,hi] = [the lower 68 % bound of D₀,N,A, the upper 68 % bound of D_R] is fixed per tile (Appendix D7) on the 38
usable primary tiles. The interval figures draw the addendum 18 fits on it; counted by eye and by `partA.json` (no rule applied):

| site | determined tiles | AD | P4 | P3 | A1 |
| --- | --- | --- | --- | --- | --- |
| GM | 18 | below 18 | **inside 14**, above 4 (GM6_fema, GM8, GM16, GM20) | inside 14, above 4 | below 18 |
| AP | 12 | below 11, inside 1 | **above 11**, inside 1 (AP25) | above 10, inside 2 | below 9, above 3 |
| TTV | 8 | below 4, above 3, inside 1 | **above 8** | above 7, inside 1 | below 4, above 4 |

P4's D at GM lies inside the measured bracket on 14 of 18 tiles (its excess over D_C is the relief term); at AP and TTV it lies above D_R,hi
on nearly every tile (AP P4/D_R 1.1–2.1, TTV 1.2–5.3), i.e. above what the measured hillslope length and relief admit for a linear model;
AD's D lies below D_C at GM and AP. These are the readings decision 12 wrote in advance; the counts become a verdict only when R10 runs.

### 4.3 TTV anchor (decision 14)
TTV site medians against Dietrich's 5 × 10⁻³ m²/yr in the soil-flux convention (2.5 × 10⁻³ in ours): D₀,N,A 0.0046 = **1.86 ×** (0.93 × the
soil-flux value); D₀,N,S 0.0029 = 1.17 ×; D₀,R,A 0.0027 = 1.08 ×; D_R 0.015 = 5.9 ×; D₀,N,B 0.0075 = 3.0 × (Appendix D7).

## 5. The heads and the area–slope relation (section 3; rules R4, R5)

`a19_heads.py` (jobs 53095206, 53095652; Appendix D8; figure `ttv_heads_area_slope.png`).

**Attributes.** The 64 picks sit on hillslope cells: the D8 area at the pick cell is 4 m² at the median (IQR 2–17; 95th percentile 669); the
largest area within 3 m, **A_3**, is **611 m²** at the median (5th–95th: 64 / 169 / 611 / 2,152 / 7,255), within 10 m **2,848** (275 / 1,042 /
2,848 / 7,587 / 18,797); M&D's Table 1: 2,010 / 4,800 / **7,400** / 10,350 / 14,860. The 3 m search reaches the hollow axis of the 1 m D8
routing on few picks (A_3 is the area at a cell 2–3 m from the pick on every one of the 64); the 10 m search reaches it on most (the A_10 median sits at
Table 1's 25th percentile). The picks are 6 m from the traced M&D channel lines at the median (georeference RMS 9.5 m); their placement
precision is not 3 m. S_up (10 m): median 0.40 (0.25–0.64); S_down: 0.38 (0.24–0.64); Table 1: 0.47 (0.27–0.78). One S_up chain is shorter
than 10 m; no S_down path leaves the grid. Without pick 49 nothing moves (A_3 median 606).

**D∞ (decision 3).** On the full grid (141 M valid cells; job 53095652, 1.6 h wall on 3 CPUs = 4.8 core-hours allocated, peak 25.7 GB): A_3 median **387 m²** (IQR 120–1,638; 5th–95th 49–5,342), A_10 median 2,212 (IQR 631–5,704); the median per-pick ratio D∞/D8 is 0.72 for A_3 and 0.73 for A_10; the D∞ area at the pick cell itself has a median of 23 m². D∞ routing spreads the hollow flow and lowers the areas by about a quarter; it does not move the picks toward Table 1 (A_F from D∞ would be 387 m²). Per pick in Appendix D8.

**Fits (decision 4; log₁₀ A = log₁₀ a − b log₁₀ S, A dependent, 10,000-draw bootstrap):**

| fit | n | a (m²) | 95 % | b | 95 % | R² |
| --- | --- | --- | --- | --- | --- | --- |
| (i) Table 1 | 63 | **1,978** | [1,459, 2,774] | **1.647** | [1.26, 2.01] | 0.60 |
| (ii) picks A_3, S_up | 64 | 64 | [22, 190] | 2.40 | [1.35, 3.43] | 0.22 |
| (iii) picks A_3, S_down | 64 | 65 | [20, 218] | 2.21 | [1.04, 3.28] | 0.19 |
| (iv) (ii) without 49 | 63 | 62 | [20, 189] | 2.44 | [1.37, 3.52] | 0.22 |

**R4 passes**: a within 0.0 % of 1,978 and b within 0.003 of 1.65 (the transcription of Table 1 reproduces M&D's eq. 10; the R² of 0.60 against
the paper's 0.75 is the one difference: M&D may have fitted differently or on a slightly different row set). **R5 fails**: b = 1.65 lies inside
fit (ii)'s 95 % band, but the median of A_3/(1978 S_up^−1.65) is **0.047** (IQR 0.02–0.19), far below 0.5–2; (iii) 0.045, (iv) 0.049, the same.
With A_10 in place of A_3 the median ratio is 0.32 (IQR 0.11–0.71): still below 0.5, though inside the band on a third of the picks. The picks' A·S²: median 78 m² with S_up, 62 with S_down (Table 1: 1,500; M&D 1992's range 500–4,000
holds 13–16 % of the picks).

**Field values for section 5 (not used):** A_F = 611 m² (7,400 from Table 1; 2,848 from A_10, labelled); b̂ = 2.40 → α = 2.4; C_F = 611 (α = 0),
62 (α = 2), 41 m² (α = 2.4) (without pick 49: 606, 60, 42).

## 6. Field-fixed thresholds (section 5.2, rule R6): not run
R1 fails. The gate-3 timing (job 53096475, `fixed/TTV2.json`, Appendix D9) ran PF, PF3 and SF on TTV2: one PF evaluation 3.9 s, one SF 3.4 s
(the head statistic 1 s), so the 11 tiles take under a core-hour. On TTV2 (addendum 18: P4 MSE 59.8 at A_crit 9,131, AD 209.7): PF at
A_F = 611 m² MSE 129.6 (K/K₀ 1.02, D 0.0080), at 7,400 m² 66.7 (D 0.0149), at A_10's 2,848 90.0; PF3 (K = K₀) 131.3 / 68.1 / 90.1; SF at
α = 2 (C_F 62) 129.2, at α = 2.4 (C_F 41) 132.7; SF at α = 0 equals PF to 10⁻¹² (the check of decision 5b). On this one tile the field-fixed
area threshold at the picks' A_3 fits 2.2 × worse than the tuned one and still 1.6 × better than AD.

## 7. The slope-dependent threshold (section 5.1, rule R7): not run
The SJ pilot (TTV2, α = 2, 20 evaluations) gives 6.09 s per evaluation; a full run (2,000 + 400 evaluations) ≈ 4.1 h; **33 runs ≈ 134
core-hours expected, limits 8.1 h each = 268 core-hours of limits in one submission** (under 300). `a19_submit_sj.py` prints this estimate and
submits the array; the module's `summarize` handles AP and GM only, so `a19_rules.py` computes R7 for TTV from the run records and checks
α = 0 against the addendum 18 rows (common A_crit values) in place of `_phase6_check`, which finds no Phase 6 run for TTV.

## 8. Model heads against the picked heads (rule R8): not run
`a19_modelheads.py` is written (channel sets, heads, pick-to-channel distances, the head/pick ratio inside the polygon, the 3 × 3 figures).

## 9. The D statements with D₀,N and the bracket (section 5.3, rules R2, R10): not run
`a19_dstat.py` (per addendum 18 spec, the routes of section 1.1) and `a19_rules.py` are written; R11 would withhold AP in any case. The
Part B estimate in all: SJ 134, the D statements ≈ 8 (98 specs × 5 min: one AD model build and ~10 fresh evaluations each), PF/SF/heads/
figures ≈ 2: **≈ 145 core-hours expected**, 330 of limits over three submissions, inside the 320 budget.

## 10. What the evidence supports, and where it is weak

**Supported.**
1. **The fixed-network crest set behaves as decision 1 intended.** The counted pixels are the ridge crests between 10³ basins; on them the
   two GM surveys agree to 1 % on the same pixels and the selection ratio is 1.13 (the rollover network's 0.80 is gone); the GM crest
   curvature reproduces Perron et al.'s value (0.95 ×, and 0.89–0.95 at every window 7–51 m) and the manuscript's coarse route (0.96 ×); the
   NCALM and FEMA crest D₀ agree to 20 % where the rollover sets differed by 50 %.
2. **D_R/D_C = 2.0 (GM), 3.3 (AP), 3.4 (TTV)**: the relief term is measured, with narrow intervals, on every usable tile; E* is low enough
   that the curvature bias is not the cause.
3. **Read from the interval figure:** P4's D is inside the measured bracket on 14 of 18 GM tiles and above it on 11 of 12 AP and 8 of 8 TTV
   tiles; AD's D is below D_C at GM and AP. At TTV the fitted D matches the relief of the 150–380 m hillslopes the fit itself creates, not of
   the 70 m hillslopes of the 10³ network.
4. **Table 1 reproduces M&D's eq. 10** (a = 1,978, b = 1.647).

**Weak.**
1. **The segment filter is the wrong tool for this hilltop definition**: D8 divide cells along a crest are fragments of 6–10 px, so a
   50-px threshold keeps a few crest stretches per tile and R1 fails everywhere. The filter changes D₀ by 7–14 % at GM and AP (it keeps the
   sharper stretches) and the counts matter more than the values: without it every primary tile has ≥ 500 pixels. The cheap definitions that
   would keep the intent (50 m of crest) are a threshold on the "parent" crest (which does not help: the divide set itself is fragmented) or a
   dilation-based crest line; neither was coded.
2. **R11 fails at AP** on the hilltop definition that passes at GM, at every window: the metre-scale crest curvature of the Allegheny ridges
   is 2–3 × Perron's coarse value. Whether the calibration gate should bind the 11 m D₀ at AP to a 30 m number is a decision.
3. **L_w as written is the largest window on every tile**; D₀,N,B is a 51 m curvature and its window test is a test over 11–51 m. The B′
   values with the rollover crest width are the like-for-like ones.
4. **The scaling-break fit finds the roughness roll-off at 5–6 m**, not a break in 8–20 m; R12's letter says "not consistent" at every site,
   but there is nothing in 8–20 m to be consistent with.
5. **The picks' D8 area depends on the search radius more than on anything else** (611 m² at 3 m, 2,848 at 10 m, 7,400 in Table 1); R5 fails
   on the A_3 primary by a factor 20 and would pass or fail on A_10 by the choice of radius. The hand picks mark hollow tips on the hillshade
   to ± 5–10 m, not the D8 axis to ± 3 m.
6. **The same-pixel readings are "mixed" by the letter** (P 1.13 at GM, S 0.88 at AP) on a rule written for a different pixel set.
7. **Part B was not run**, so nothing here tests the field-fixed thresholds, the slope-dependent threshold or the D statements with D₀,N;
   the gate-3 numbers on one tile are not a result.

## 11. Decisions needed
1. **The segment filter (decision 1).** Options: (a) keep 50 px as written and accept the stop (18 / 12 / 8 usable tiles); (b) drop the
   segment filter (22 / 21 / 11 usable; D₀,N,A medians 0.0151 / 0.0044 / 0.0044); (c) lower the threshold to 20 px (21 / 17 / 11) or 10 px
   (22 / 20 / 11), or measure it on the parent crest at 20 px (21 / 19 / 11, R1 passes everywhere) or 50 px (20 / 12 / 8); (d) redefine the
   segment as a continuous crest line (not coded). R1 was written before results; changing it is Steve's call.
   Every per-tile value at (b) and (c) is in Appendix D2.
2. **R11 at AP.** Keep the gate as written (no AP D statement from the 11 m crest curvature), or calibrate against the manuscript's coarse
   route (1.07 ×) and state that Perron's value is a 30 m number.
3. **L_w (decision 1):** adopt B′ (the rollover crest width) as the second end, or keep B at 51 m.
4. **The head radius (decision 3):** A_3 as written, or A_10 (median 2,848 m²; closer to the placement precision), for A_F, C_F and R5.
5. **Part B as prepared** (≈ 145 core-hours, 330 of limits): sections 5.1–5.2 (SJ, PF, SF) and 5.4 do not depend on the hilltop set and could
   run under any decision on 1–4; section 5.3 needs decision 1 (and 2 for AP). One line from Steve starts them.
6. The reading of the bracket against the addendum 18 fits (section 4.2) is written as a reading from a figure; whether the paper may use it
   before R10 runs.
7. Nothing is merged or tagged. Push only.

## 12. Appendices: all tables, run ids, checksums, Slurm accounting

Generated by `reports/v2/scripts/make_a19_tables.py` from `/home/groups/hilley/lemcvt_results/20261120T160830Z_a19_hilltops_heads_slope` (also `appendix_tables.md` there).

### D1. The heads CSV, the road layers, the appendix fix (section 1)

- Steve's heads: source `reports/directives/ttv_heads_manual_20261119T1911.csv` (sha256 `ac87de3d4ca1f479e6eb2ceda2fedb033740b3d8f37186fd267329d5e4e702b6`), copy of record `data/ttv/ttv_heads_manual_steve_20261119.csv` (sha256 `ac87de3d4ca1f479e6eb2ceda2fedb033740b3d8f37186fd267329d5e4e702b6`), 64 rows; 64 of 64 inside the M&D polygon; by tile {'TTV1': 22, 'TTV2': 34, 'TTV3': 5, 'none': 3}.
- Road/trail layer: US Census Bureau TIGER/Line Shapefiles 2024, ROADS (all MTFCC road classes incl. S1500 vehicular trail, S1710 walkway/pedestrian trail, S1820 bike path, S1830 bridle path). County file `tl_2024_us_county.zip` sha256 `04e668d3502757c837c13444730547cd967f28a2c49aeffb873d1792ab2cb97b`. Manifest `/home/groups/hilley/lemcvt_data/source/TIGER2024_roads/MANIFEST.sha256`.
| site | GEOID | county | features | size | sha256 (16) | MTFCC counts |
|---|---|---|---|---|---|---|
| GM | 06053 | Monterey County | 18,200 | 3.3 MB | 5f0a377302a5a80b... | S1100 29, S1200 85, S1400 15809, S1500 144, S1630 259, S1640 7, S1710 21, S1730 52, S1740 1752, S1750 19, S1780 22, S1820 1 |
| AP | 42059 | Greene County | 5,704 | 1.4 MB | cbff34d40456b8dc... | S1100 2, S1200 78, S1400 4880, S1500 12, S1630 20, S1640 1, S1710 2, S1730 46, S1740 648, S1750 14, S1780 1 |
| TTV | 06041 | Marin County | 8,072 | 2.2 MB | 3c5dc18cc5c5074b... | S1100 9, S1200 38, S1400 7218, S1500 144, S1630 207, S1640 28, S1710 57, S1720 5, S1730 1, S1740 315, S1750 36, S1780 12, S1820 2 |

- Appendix fix (decision 8): `reports/v2/20261120T0841PST_expanded_fits_appendix_fix.md`; 98 C16 rows regenerated (98 unique specs; the original's spec column held ['AP5:pamap']), 98 checksum rows; 98 partitioned and 98 AD run records read; spec mismatches [].

### D2. D₀ on the fixed network per tile (section 2.1): the counted set, the filters, C_ht, L_w, the window test, D₀,N against D₀,R and D₀,N,unf

| tile | site | counted | geom | −road | −segment | −non-convex | segments | usable (≥ 500) | C_ht(L_f) | 68 % | D₀,N,A | 68 % | L_w (m) | width→10³ (m) | D₀,N,B | window test (L_w) | L_w′ (m) | width→rollover (m) | D₀,N,B′ | window test (L_w′) | D₀,R,A | D₀,R,B | n rollover | N/R (A) | N/R (B) | D₀,N,unf,A | N/unf (A) | N/no-convex-drop (A) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | AP | 763 | 2,688 | 418 | 1,507 | 0 | 6 | yes | 0.01574 | [0.0145, 0.0171] | 0.003494 | [0.00321, 0.0038] | 51 | 110 | 0.0051 | fail (7 windows, max dev 0.34) | 51 | 65 | 0.0051 | fail | 0.003587 | 0.006978 | 3,446 | 0.97 | 0.73 | 0.003104 | 1.13 | 1.000 |
| AP10 | AP | 1,349 | 5,294 | 74 | 3,842 | 29 | 13 | yes | 0.01356 | [0.0119, 0.0163] | 0.004057 | [0.00338, 0.00462] | 51 | 162 | 0.005645 | fail (7 windows, max dev 0.28) | 11 | 7 | 0.004057 | pass (trivial) | 0.002614 | 0.002614 | 145 | 1.55 | 2.16 | 0.004715 | 0.86 | 0.986 |
| AP11 | AP | 472 | 1,933 | 95 | 1,363 | 3 | 6 | **no** | 0.01357 | [0.0108, 0.016] | 0.004053 | [0.00344, 0.00512] | 51 | 177 | 0.006855 | fail (7 windows, max dev 0.42) | 11 | 6 | 0.004053 | pass (trivial) | - | - | 0 | - | - | 0.005328 | 0.76 | 0.997 |
| AP13 | AP | 855 | 2,584 | 57 | 1,672 | 0 | 6 | yes | 0.02258 | [0.0182, 0.0249] | 0.002436 | [0.00221, 0.00303] | 51 | 156 | 0.003784 | fail (7 windows, max dev 0.36) | 51 | 85 | 0.003784 | fail | 0.004471 | 0.007562 | 3,087 | 0.54 | 0.50 | 0.00353 | 0.69 | 1.000 |
| AP14 | AP | 1,686 | 5,950 | 578 | 3,641 | 45 | 22 | yes | 0.01409 | [0.0124, 0.0159] | 0.003905 | [0.00347, 0.00444] | 51 | 134 | 0.005799 | fail (7 windows, max dev 0.33) | 11 | 6 | 0.003905 | pass (trivial) | 0.002095 | 0.002095 | 118 | 1.86 | 2.77 | 0.00446 | 0.88 | 0.978 |
| AP18 | AP | 320 | 1,702 | 33 | 1,317 | 32 | 6 | **no** | 0.01428 | [0.00744, 0.0241] | 0.003852 | [0.00228, 0.00739] | 51 | 113 | 0.007567 | fail (7 windows, max dev 0.49) | 11 | 7 | 0.003852 | pass (trivial) | 0.001885 | 0.001885 | 25 | 2.04 | 4.01 | 0.00448 | 0.86 | 0.812 |
| AP19 | AP | 84 | 3,823 | 115 | 3,624 | 0 | 1 | **no** | 0.04223 | - | 0.001302 | - | 51 | 136 | 0.003046 | fail (7 windows, max dev 0.57) | 51 | 80 | 0.003046 | fail | 0.006118 | 0.007877 | 6,160 | 0.21 | 0.39 | 0.004593 | 0.28 | 1.000 |
| AP1_3dep | AP | 620 | 3,198 | 203 | 2,323 | 52 | 11 | yes | 0.02096 | [0.0187, 0.0259] | 0.002625 | [0.00212, 0.00295] | 51 | 110 | 0.005119 | fail (7 windows, max dev 0.49) | 31 | 32 | 0.003926 | fail | 0.005507 | 0.006664 | 4,575 | 0.48 | 0.77 | 0.003693 | 0.71 | 0.962 |
| AP2 | AP | 494 | 1,692 | 19 | 1,150 | 29 | 10 | **no** | 0.01707 | [0.0149, 0.0187] | 0.003222 | [0.00294, 0.00369] | 51 | 135 | 0.005207 | fail (7 windows, max dev 0.38) | 15 | 16 | 0.003908 | fail | 0.003218 | 0.004078 | 944 | 1.00 | 1.28 | 0.003688 | 0.87 | 0.970 |
| AP20 | AP | 1,435 | 6,838 | 324 | 4,961 | 118 | 28 | yes | 0.01367 | [0.0113, 0.0159] | 0.004023 | [0.00347, 0.00485] | 51 | 109 | 0.006535 | fail (7 windows, max dev 0.38) | 25 | 26 | 0.006097 | fail | 0.0054 | 0.007204 | 6,710 | 0.75 | 0.91 | 0.004438 | 0.91 | 0.904 |
| AP23 | AP | 616 | 1,533 | 3 | 908 | 6 | 7 | yes | 0.01822 | [0.0164, 0.0195] | 0.003018 | [0.00282, 0.00336] | 51 | 157 | 0.00469 | fail (7 windows, max dev 0.36) | 51 | 73 | 0.00469 | fail | 0.003661 | 0.004505 | 2,197 | 0.82 | 1.04 | 0.003355 | 0.90 | 0.999 |
| AP24 | AP | 719 | 3,479 | 828 | 1,927 | 5 | 7 | yes | 0.01589 | [0.0109, 0.0493] | 0.003462 | [0.00111, 0.00503] | 51 | 76 | 0.006195 | fail (7 windows, max dev 0.44) | 21 | 24 | 0.004472 | fail | 0.005939 | 0.008242 | 5,040 | 0.58 | 0.75 | 0.004618 | 0.75 | 0.993 |
| AP25 | AP | 290 | 1,421 | 568 | 558 | 5 | 3 | **no** | 0.009429 | - | 0.005833 | - | 41 | 47 | 0.02114 | fail (6 windows, max dev 0.83) | 15 | 16 | 0.009527 | fail | 0.003859 | 0.004573 | 1,124 | 1.51 | 4.62 | 0.003733 | 1.56 | 0.998 |
| AP27 | AP | 1,164 | 3,981 | 51 | 2,709 | 57 | 18 | yes | 0.01933 | [0.0151, 0.0253] | 0.002846 | [0.00218, 0.00364] | 51 | 142 | 0.00448 | fail (7 windows, max dev 0.36) | 11 | 6 | 0.002846 | pass (trivial) | 0.001319 | 0.001319 | 83 | 2.16 | 3.40 | 0.003682 | 0.77 | 0.938 |
| AP28 | AP | 1,336 | 4,953 | 927 | 2,652 | 38 | 16 | yes | 0.01333 | [0.0119, 0.0152] | 0.004127 | [0.00363, 0.0046] | 51 | 152 | 0.006046 | fail (7 windows, max dev 0.32) | 11 | 6 | 0.004127 | pass (trivial) | 0.002704 | 0.002704 | 119 | 1.53 | 2.24 | 0.004346 | 0.95 | 0.969 |
| AP29 | AP | 463 | 2,886 | 46 | 2,343 | 34 | 8 | **no** | 0.01983 | [0.0144, 0.0249] | 0.002773 | [0.00221, 0.00383] | 51 | 127 | 0.005728 | fail (7 windows, max dev 0.52) | 31 | 40 | 0.003939 | fail | 0.006297 | 0.006704 | 3,606 | 0.44 | 0.85 | 0.005192 | 0.53 | 0.912 |
| AP2_3dep | AP | 151 | 1,754 | 131 | 1,472 | 0 | 2 | **no** | 0.0137 | - | 0.004016 | - | 51 | 122 | 0.005201 | fail (7 windows, max dev 0.23) | 11 | 6 | 0.004016 | pass (trivial) | 0.002742 | 0.002742 | 13 | 1.46 | 1.90 | 0.005682 | 0.71 | 1.000 |
| AP3 | AP | 758 | 2,969 | 63 | 2,113 | 35 | 8 | yes | 0.02081 | [0.0173, 0.0249] | 0.002643 | [0.00221, 0.00317] | 51 | 128 | 0.004452 | fail (7 windows, max dev 0.41) | 11 | 10 | 0.002643 | pass (trivial) | 0.002155 | 0.002155 | 533 | 1.23 | 2.07 | 0.003019 | 0.88 | 0.941 |
| AP3_3dep | AP | 399 | 3,963 | 177 | 3,358 | 29 | 7 | **no** | 0.01404 | [0.0117, 0.0218] | 0.003918 | [0.00253, 0.00471] | 51 | 146 | 0.004693 | fail (7 windows, max dev 0.17) | 51 | 63 | 0.004693 | fail | 0.004372 | 0.005272 | 4,094 | 0.90 | 0.89 | 0.004302 | 0.91 | 0.944 |
| AP4 | AP | 2,063 | 4,141 | 208 | 1,781 | 89 | 29 | yes | 0.01837 | [0.017, 0.0197] | 0.002994 | [0.0028, 0.00323] | 51 | 104 | 0.00703 | fail (7 windows, max dev 0.57) | 51 | 59 | 0.00703 | fail | 0.003629 | 0.00772 | 4,711 | 0.83 | 0.91 | 0.003203 | 0.93 | 0.978 |
| AP4_3dep | AP | 1,403 | 5,245 | 351 | 3,454 | 37 | 20 | yes | 0.01543 | [0.0131, 0.0185] | 0.003564 | [0.00298, 0.00419] | 51 | 111 | 0.00806 | fail (7 windows, max dev 0.56) | 15 | 21 | 0.00449 | fail | 0.00536 | 0.007554 | 6,417 | 0.66 | 1.07 | 0.004173 | 0.85 | 0.976 |
| AP5 | AP | 1,225 | 2,549 | 3 | 1,277 | 44 | 16 | yes | 0.0158 | [0.0134, 0.0189] | 0.003481 | [0.00292, 0.00409] | 51 | 151 | 0.006079 | fail (7 windows, max dev 0.43) | 15 | 19 | 0.004381 | fail | 0.002921 | 0.003456 | 3,575 | 1.19 | 1.76 | 0.003595 | 0.97 | 0.979 |
| AP5_3dep | AP | 1,017 | 3,322 | 107 | 2,169 | 29 | 12 | yes | 0.01167 | [0.00995, 0.0141] | 0.004714 | [0.00391, 0.00553] | 51 | 156 | 0.006584 | fail (7 windows, max dev 0.28) | 51 | 65 | 0.006584 | fail | 0.005144 | 0.006494 | 5,502 | 0.92 | 1.01 | 0.005258 | 0.90 | 0.974 |
| AP6 | AP | 930 | 3,446 | 370 | 2,104 | 42 | 12 | yes | 0.01661 | [0.0125, 0.0219] | 0.003312 | [0.00251, 0.00439] | 51 | 132 | 0.004528 | fail (7 windows, max dev 0.27) | 51 | 69 | 0.004528 | fail | 0.005952 | 0.007218 | 4,391 | 0.56 | 0.63 | 0.004375 | 0.76 | 0.943 |
| AP7 | AP | 260 | 2,359 | 311 | 1,760 | 28 | 5 | **no** | 0.01174 | [0.00901, 0.0307] | 0.004683 | [0.00179, 0.00611] | 51 | 134 | 0.00672 | fail (7 windows, max dev 0.33) | 51 | 66 | 0.00672 | fail | 0.005239 | 0.005771 | 2,756 | 0.89 | 1.16 | 0.003818 | 1.23 | 0.938 |
| AP8 | AP | 371 | 2,330 | 40 | 1,919 | 0 | 3 | **no** | 0.02815 | - | 0.001954 | - | 51 | 170 | 0.003508 | fail (7 windows, max dev 0.44) | 51 | 74 | 0.003508 | fail | 0.006024 | 0.007696 | 3,340 | 0.32 | 0.46 | 0.003669 | 0.53 | 1.000 |
| AP9 | AP | 1,118 | 5,734 | 696 | 3,881 | 39 | 16 | yes | 0.01652 | [0.0149, 0.0182] | 0.00333 | [0.00302, 0.00369] | 51 | 128 | 0.005664 | fail (7 windows, max dev 0.41) | 11 | 6 | 0.00333 | pass (trivial) | 0.001847 | 0.001847 | 30 | 1.80 | 3.07 | 0.00397 | 0.84 | 0.983 |
| GM1 | GM | 1,657 | 5,040 | 0 | 3,367 | 16 | 20 | yes | 0.01353 | [0.0124, 0.0153] | 0.01087 | [0.00962, 0.0118] | 51 | 108 | 0.0103 | pass (7 windows, max dev 0.07) | 11 | 13 | 0.01087 | pass (trivial) | 0.009001 | 0.009354 | 3,583 | 1.21 | 1.10 | 0.0105 | 1.04 | 0.996 |
| GM11 | GM | 1,008 | 3,502 | 92 | 2,364 | 38 | 13 | yes | 0.008829 | [0.00746, 0.01] | 0.01665 | [0.0147, 0.0197] | 51 | 143 | 0.02118 | fail (7 windows, max dev 0.21) | 51 | 79 | 0.02118 | fail | 0.01831 | 0.01974 | 4,273 | 0.91 | 1.07 | 0.01702 | 0.98 | 0.968 |
| GM12 | GM | 2,226 | 5,505 | 176 | 3,056 | 47 | 18 | yes | 0.01341 | [0.0112, 0.0157] | 0.01096 | [0.00936, 0.0131] | 51 | 158 | 0.01178 | fail (7 windows, max dev 0.15) | 31 | 35 | 0.0109 | fail | 0.01144 | 0.01273 | 7,208 | 0.96 | 0.92 | 0.01033 | 1.06 | 0.984 |
| GM13 | GM | 933 | 3,873 | 19 | 2,915 | 6 | 14 | yes | 0.005294 | [0.00467, 0.00631] | 0.02777 | [0.0233, 0.0315] | 51 | 128 | 0.02573 | pass (7 windows, max dev 0.08) | 51 | 75 | 0.02573 | pass | 0.02885 | 0.02786 | 5,046 | 0.96 | 0.92 | 0.02565 | 1.08 | 0.997 |
| GM14 | GM | 1,295 | 5,228 | 0 | 3,894 | 39 | 16 | yes | 0.01993 | [0.0178, 0.0236] | 0.007375 | [0.00622, 0.00828] | 51 | 74 | 0.01615 | fail (7 windows, max dev 0.54) | 41 | 44 | 0.01344 | fail | 0.009355 | 0.0117 | 6,475 | 0.79 | 1.38 | 0.008985 | 0.82 | 0.977 |
| GM15 | GM | 1,004 | 4,298 | 626 | 2,664 | 4 | 12 | yes | 0.02014 | [0.0172, 0.0227] | 0.0073 | [0.00647, 0.00853] | 51 | 85 | 0.008129 | fail (7 windows, max dev 0.10) | 41 | 43 | 0.007801 | pass | 0.00898 | 0.01073 | 4,822 | 0.81 | 0.76 | 0.007714 | 0.95 | 1.000 |
| GM16 | GM | 1,073 | 4,280 | 0 | 3,194 | 13 | 18 | yes | 0.01245 | [0.00797, 0.0159] | 0.01181 | [0.00924, 0.0184] | 51 | 117 | 0.01022 | fail (7 windows, max dev 0.29) | 41 | 45 | 0.009161 | fail | 0.01218 | 0.01037 | 6,454 | 0.97 | 0.99 | 0.01282 | 0.92 | 0.972 |
| GM17 | GM | 1,699 | 2,896 | 153 | 987 | 57 | 19 | yes | 0.01069 | [0.00938, 0.0116] | 0.01374 | [0.0127, 0.0157] | 51 | 118 | 0.01867 | fail (7 windows, max dev 0.27) | 41 | 41 | 0.01881 | fail | 0.01552 | 0.01987 | 4,224 | 0.89 | 0.94 | 0.01549 | 0.89 | 0.986 |
| GM18 | GM | 99 | 2,152 | 171 | 1,882 | 0 | 1 | **no** | 0.0347 | - | 0.004236 | - | 51 | 80 | 0.0148 | fail (7 windows, max dev 0.71) | 31 | 35 | 0.007871 | fail | 0.005326 | 0.006456 | 2,655 | 0.80 | 2.29 | 0.005966 | 0.71 | 1.000 |
| GM19 | GM | 813 | 3,872 | 0 | 3,041 | 18 | 13 | yes | 0.01573 | [0.0128, 0.018] | 0.009344 | [0.00816, 0.0115] | 51 | 118 | 0.01053 | fail (7 windows, max dev 0.11) | 51 | 56 | 0.01053 | fail | 0.01241 | 0.012 | 7,075 | 0.75 | 0.88 | 0.01218 | 0.77 | 0.972 |
| GM1_fema | GM | 1,827 | 6,089 | 0 | 4,246 | 16 | 27 | yes | 0.01478 | [0.0138, 0.016] | 0.009944 | [0.00917, 0.0107] | 51 | 110 | 0.01027 | pass (7 windows, max dev 0.04) | 51 | 54 | 0.01027 | pass | 0.01025 | 0.01093 | 8,554 | 0.97 | 0.94 | 0.0102 | 0.97 | 0.994 |
| GM2 | GM | 1,338 | 4,661 | 1,070 | 2,248 | 5 | 12 | yes | 0.01006 | [0.0092, 0.0115] | 0.01461 | [0.0128, 0.016] | 51 | 139 | 0.01317 | fail (7 windows, max dev 0.14) | 11 | 13 | 0.01461 | pass (trivial) | 0.00977 | 0.01015 | 2,530 | 1.50 | 1.30 | 0.01257 | 1.16 | 0.999 |
| GM20 | GM | 1,048 | 4,887 | 631 | 3,175 | 33 | 14 | yes | 0.01067 | [0.00943, 0.0123] | 0.01377 | [0.0119, 0.0156] | 51 | 91 | 0.01285 | fail (7 windows, max dev 0.20) | 11 | 14 | 0.01377 | pass (trivial) | 0.01825 | 0.01929 | 5,234 | 0.75 | 0.67 | 0.0202 | 0.68 | 0.974 |
| GM21 | GM | 1,641 | 5,539 | 184 | 3,658 | 56 | 30 | yes | 0.01574 | [0.0133, 0.0182] | 0.009341 | [0.00808, 0.0111] | 51 | 126 | 0.01023 | pass (7 windows, max dev 0.09) | 51 | 55 | 0.01023 | pass | 0.01193 | 0.01274 | 7,692 | 0.78 | 0.80 | 0.01093 | 0.85 | 0.968 |
| GM22 | GM | 469 | 1,630 | 0 | 1,155 | 6 | 5 | **no** | 0.01041 | [0.00964, 0.0113] | 0.01413 | [0.013, 0.0152] | 51 | 148 | 0.0144 | pass (7 windows, max dev 0.07) | 15 | 18 | 0.01431 | pass | 0.01609 | 0.01888 | 2,509 | 0.88 | 0.76 | 0.01535 | 0.92 | 0.996 |
| GM24 | GM | 416 | 3,392 | 19 | 2,947 | 10 | 9 | **no** | 0.006478 | [0.00509, 0.0075] | 0.02269 | [0.0196, 0.0289] | 51 | 93 | 0.02135 | fail (7 windows, max dev 0.11) | 31 | 34 | 0.02046 | fail | 0.03127 | 0.03167 | 6,482 | 0.73 | 0.67 | 0.03052 | 0.74 | 0.983 |
| GM25 | GM | 1,102 | 6,066 | 0 | 4,905 | 59 | 15 | yes | 0.01878 | [0.0141, 0.0214] | 0.007829 | [0.00687, 0.0104] | 51 | 81 | 0.01028 | fail (7 windows, max dev 0.24) | 51 | 75 | 0.01028 | fail | 0.01061 | 0.01185 | 5,909 | 0.74 | 0.87 | 0.01011 | 0.77 | 0.944 |
| GM26 | GM | 445 | 1,607 | 171 | 926 | 65 | 9 | **no** | 0.006962 | [0.006, 0.00903] | 0.02111 | [0.0163, 0.0245] | 51 | 143 | 0.02474 | fail (7 windows, max dev 0.16) | 41 | 46 | 0.02525 | fail | 0.0174 | 0.01901 | 3,448 | 1.21 | 1.30 | 0.01783 | 1.18 | 0.887 |
| GM27 | GM | 1,442 | 4,533 | 0 | 3,091 | 0 | 17 | yes | 0.02281 | [0.0218, 0.024] | 0.006446 | [0.00612, 0.00676] | 51 | 100 | 0.008287 | fail (7 windows, max dev 0.22) | 41 | 44 | 0.007806 | fail | 0.00728 | 0.008165 | 5,349 | 0.89 | 1.01 | 0.007144 | 0.90 | 1.000 |
| GM28 | GM | 1,795 | 4,331 | 8 | 2,487 | 41 | 26 | yes | 0.009361 | [0.00758, 0.0117] | 0.0157 | [0.0126, 0.0194] | 51 | 126 | 0.02266 | fail (7 windows, max dev 0.31) | 31 | 39 | 0.02052 | fail | 0.01462 | 0.01718 | 6,318 | 1.07 | 1.32 | 0.01424 | 1.10 | 0.968 |
| GM29 | GM | 2,038 | 5,459 | 80 | 3,339 | 2 | 22 | yes | 0.01099 | [0.0095, 0.0142] | 0.01337 | [0.0103, 0.0155] | 51 | 135 | 0.01282 | fail (7 windows, max dev 0.15) | 51 | 56 | 0.01282 | fail | 0.01455 | 0.0139 | 6,738 | 0.92 | 0.92 | 0.01568 | 0.85 | 1.000 |
| GM2_fema | GM | 1,519 | 4,541 | 1,048 | 1,967 | 7 | 18 | yes | 0.01209 | [0.0106, 0.0139] | 0.01216 | [0.0106, 0.0138] | 51 | 131 | 0.01118 | pass (7 windows, max dev 0.09) | 31 | 39 | 0.01183 | pass | 0.01251 | 0.01197 | 5,425 | 0.97 | 0.93 | 0.01219 | 1.00 | 0.996 |
| GM3 | GM | 3,312 | 7,041 | 932 | 2,665 | 132 | 49 | yes | 0.008078 | [0.00729, 0.00897] | 0.0182 | [0.0164, 0.0202] | 51 | 104 | 0.01959 | fail (7 windows, max dev 0.11) | 11 | 8 | 0.0182 | pass (trivial) | 0.01012 | 0.01012 | 1,500 | 1.80 | 1.94 | 0.01718 | 1.06 | 0.963 |
| GM30 | GM | 1,493 | 3,242 | 0 | 1,729 | 20 | 18 | yes | 0.01634 | [0.0142, 0.018] | 0.008996 | [0.00817, 0.0103] | 51 | 133 | 0.01026 | fail (7 windows, max dev 0.12) | 31 | 35 | 0.008611 | pass | 0.01114 | 0.01171 | 6,877 | 0.81 | 0.88 | 0.01034 | 0.87 | 0.993 |
| GM3_fema | GM | 2,507 | 6,544 | 893 | 3,094 | 50 | 35 | yes | 0.00922 | [0.00802, 0.0107] | 0.01594 | [0.0137, 0.0183] | 51 | 106 | 0.0204 | fail (7 windows, max dev 0.23) | 15 | 18 | 0.01619 | pass | 0.01545 | 0.01583 | 6,283 | 1.03 | 1.29 | 0.01759 | 0.91 | 0.989 |
| GM4 | GM | 1,207 | 3,138 | 26 | 1,875 | 30 | 18 | yes | 0.01166 | [0.00926, 0.015] | 0.01261 | [0.00982, 0.0159] | 51 | 110 | 0.01019 | fail (7 windows, max dev 0.24) | 15 | 19 | 0.01227 | pass | 0.009855 | 0.009435 | 3,347 | 1.28 | 1.08 | 0.0113 | 1.12 | 0.964 |
| GM4_fema | GM | 1,329 | 3,299 | 5 | 1,958 | 7 | 20 | yes | 0.01434 | [0.0107, 0.0175] | 0.01025 | [0.00839, 0.0137] | 51 | 134 | 0.009123 | fail (7 windows, max dev 0.22) | 51 | 54 | 0.009123 | fail | 0.01178 | 0.01097 | 5,676 | 0.87 | 0.83 | 0.01097 | 0.93 | 0.996 |
| GM5 | GM | 344 | 962 | 0 | 618 | 0 | 2 | **no** | 0.02047 | - | 0.007182 | - | 51 | 129 | 0.008811 | fail (7 windows, max dev 0.18) | 15 | 18 | 0.007108 | pass | 0.007589 | 0.007496 | 1,128 | 0.95 | 1.18 | 0.007921 | 0.91 | 1.000 |
| GM5_fema | GM | 280 | 1,254 | 13 | 961 | 0 | 2 | **no** | 0.02055 | - | 0.007153 | - | 51 | 135 | 0.008725 | fail (7 windows, max dev 0.18) | 41 | 41 | 0.008045 | fail | 0.008725 | 0.008471 | 3,310 | 0.82 | 1.03 | 0.009448 | 0.76 | 1.000 |
| GM6 | GM | 1,158 | 4,324 | 14 | 3,118 | 34 | 19 | yes | 0.008401 | [0.00662, 0.0114] | 0.0175 | [0.0129, 0.0222] | 51 | 140 | 0.01745 | fail (7 windows, max dev 0.10) | 11 | 13 | 0.0175 | pass (trivial) | 0.01094 | 0.01031 | 1,987 | 1.60 | 1.69 | 0.01696 | 1.03 | 0.959 |
| GM6_fema | GM | 973 | 3,697 | 1 | 2,716 | 7 | 10 | yes | 0.01191 | [0.00809, 0.0141] | 0.01234 | [0.0104, 0.0182] | 51 | 114 | 0.01068 | fail (7 windows, max dev 0.29) | 51 | 78 | 0.01068 | fail | 0.01625 | 0.01432 | 4,549 | 0.76 | 0.75 | 0.01692 | 0.73 | 0.987 |
| GM7 | GM | 1,249 | 3,678 | 0 | 2,360 | 69 | 26 | yes | 0.005734 | [0.00532, 0.00607] | 0.02564 | [0.0242, 0.0276] | 51 | 125 | 0.0294 | fail (7 windows, max dev 0.16) | 11 | 8 | 0.02564 | pass (trivial) | 0.02203 | 0.02203 | 920 | 1.16 | 1.33 | 0.02863 | 0.90 | 0.957 |
| GM7_fema | GM | 1,135 | 3,552 | 3 | 2,404 | 10 | 16 | yes | 0.005298 | [0.00463, 0.00575] | 0.02775 | [0.0255, 0.0318] | 51 | 140 | 0.02654 | fail (7 windows, max dev 0.15) | 51 | 103 | 0.02654 | fail | 0.03544 | 0.03574 | 3,585 | 0.78 | 0.74 | 0.03291 | 0.84 | 0.992 |
| GM8 | GM | 2,833 | 4,859 | 0 | 1,925 | 101 | 38 | yes | 0.01061 | [0.00983, 0.0114] | 0.01385 | [0.0129, 0.015] | 51 | 111 | 0.01412 | pass (7 windows, max dev 0.09) | 51 | 52 | 0.01412 | pass | 0.01523 | 0.01564 | 4,963 | 0.91 | 0.90 | 0.01468 | 0.94 | 0.972 |
| GM9 | GM | 819 | 2,988 | 490 | 1,660 | 19 | 12 | yes | 0.006112 | [0.00544, 0.00692] | 0.02405 | [0.0212, 0.027] | 51 | 139 | 0.02147 | fail (7 windows, max dev 0.13) | 51 | 66 | 0.02147 | fail | 0.02036 | 0.0211 | 3,779 | 1.18 | 1.02 | 0.02083 | 1.15 | 0.988 |
| TTV1 | TTV | 853 | 6,186 | 1,553 | 3,617 | 163 | 34 | yes | 0.01266 | [0.0106, 0.0154] | 0.00608 | [0.00501, 0.00724] | 51 | 61 | 0.01693 | fail (7 windows, max dev 0.64) | 11 | 8 | 0.00608 | pass (trivial) | 0.002905 | 0.002905 | 966 | 2.09 | 5.83 | 0.004502 | 1.35 | 0.792 |
| TTV11 | TTV | 883 | 7,120 | 1,457 | 4,737 | 43 | 12 | yes | 0.0177 | [0.0146, 0.02] | 0.00435 | [0.00386, 0.00528] | 51 | 87 | 0.007143 | fail (7 windows, max dev 0.39) | 11 | 10 | 0.00435 | pass (trivial) | 0.002312 | 0.002558 | 3,192 | 1.88 | 2.79 | 0.00407 | 1.07 | 0.954 |
| TTV15 | TTV | 779 | 5,837 | 1,231 | 3,771 | 56 | 18 | yes | 0.01292 | [0.0113, 0.0149] | 0.005959 | [0.00516, 0.00682] | 51 | 90 | 0.008004 | fail (7 windows, max dev 0.26) | 11 | 7 | 0.005959 | pass (trivial) | 0.001506 | 0.001506 | 579 | 3.96 | 5.31 | 0.00514 | 1.16 | 0.946 |
| TTV2 | TTV | 532 | 3,561 | 348 | 2,675 | 6 | 8 | yes | 0.01562 | [0.0106, 0.0188] | 0.004931 | [0.00409, 0.00727] | 51 | 100 | 0.006127 | fail (7 windows, max dev 0.20) | 11 | 13 | 0.004931 | pass (trivial) | 0.004461 | 0.004708 | 1,995 | 1.11 | 1.30 | 0.004863 | 1.01 | 0.971 |
| TTV3 | TTV | 323 | 1,579 | 298 | 923 | 35 | 10 | **no** | 0.01202 | [0.0099, 0.013] | 0.006404 | [0.00594, 0.00778] | 51 | 101 | 0.009238 | fail (7 windows, max dev 0.31) | 11 | 6 | 0.006404 | pass (trivial) | 0.003262 | 0.003262 | 122 | 1.96 | 2.83 | 0.00475 | 1.35 | 0.893 |
| TTV4 | TTV | 458 | 4,642 | 987 | 3,149 | 48 | 9 | **no** | 0.01352 | [0.0117, 0.0185] | 0.005694 | [0.00415, 0.0066] | 51 | 53 | 0.006294 | pass (7 windows, max dev 0.10) | 11 | 4 | 0.005694 | pass (trivial) | 0.0008796 | 0.0008796 | 86 | 6.47 | 7.16 | 0.004233 | 1.35 | 0.908 |
| TTV5 | TTV | 676 | 4,911 | 387 | 3,827 | 21 | 11 | yes | 0.025 | [0.0214, 0.0289] | 0.00308 | [0.00266, 0.00359] | 51 | 98 | 0.004441 | fail (7 windows, max dev 0.31) | 15 | 17 | 0.003158 | pass | 0.002498 | 0.002699 | 3,203 | 1.23 | 1.65 | 0.003264 | 0.94 | 0.980 |
| TTV6 | TTV | 820 | 6,959 | 672 | 5,375 | 92 | 21 | yes | 0.01358 | [0.0122, 0.0153] | 0.005668 | [0.00504, 0.00633] | 51 | 96 | 0.009101 | fail (7 windows, max dev 0.38) | 11 | 7 | 0.005668 | pass (trivial) | 0.001993 | 0.001993 | 612 | 2.84 | 4.57 | 0.005691 | 1.00 | 0.891 |
| TTV7 | TTV | 300 | 5,897 | 485 | 5,104 | 8 | 5 | **no** | 0.0272 | [0.0223, 0.029] | 0.002831 | [0.00265, 0.00346] | 51 | 80 | 0.003744 | fail (7 windows, max dev 0.24) | 21 | 22 | 0.002944 | pass | 0.003098 | 0.003314 | 2,983 | 0.91 | 1.13 | 0.004777 | 0.59 | 0.979 |
| TTV8 | TTV | 1,332 | 7,109 | 992 | 4,633 | 152 | 37 | yes | 0.01883 | [0.0172, 0.0207] | 0.004088 | [0.00372, 0.00448] | 51 | 79 | 0.007902 | fail (7 windows, max dev 0.48) | 15 | 20 | 0.005328 | fail | 0.003984 | 0.004479 | 5,287 | 1.03 | 1.76 | 0.004349 | 0.94 | 0.907 |
| TTV9 | TTV | 547 | 2,892 | 615 | 1,730 | 0 | 8 | yes | 0.02983 | [0.02, 0.0711] | 0.002582 | [0.00108, 0.00385] | 15 | 17 | 0.00304 | fail (2 windows, max dev 0.15) | 15 | 16 | 0.00304 | fail | 0.004202 | 0.006252 | 3,291 | 0.61 | 0.49 | 0.0038 | 0.68 | 1.000 |

Segment-filter sensitivity (report only; counted pixels at thresholds 0/10/20/50 px, segments measured on the surviving fragments as written, and on the divide set before the geometric drops ('parent'); C_ht(L_f) at each):

| tile | no filter | frag ≥ 10 | frag ≥ 20 | frag ≥ 50 (as written) | parent ≥ 10 | parent ≥ 20 | parent ≥ 50 | median fragment (px) | median parent segment (px) |
|---|---|---|---|---|---|---|---|---|---|
| AP1 | 2,053 / 0.01659 | 1,501 / 0.01734 | 1,210 / 0.01682 | 763 / 0.01574 | 1,538 / 0.01737 | 1,285 / 0.01698 | 775 / 0.0158 | 2 | 2 |
| AP10 | 4,401 / 0.0116 | 2,346 / 0.01296 | 1,809 / 0.01307 | 1,349 / 0.01356 | 2,370 / 0.01305 | 1,818 / 0.01307 | 1,384 / 0.01375 | 2 | 2 |
| AP11 | 1,620 / 0.01019 | 761 / 0.01298 | 596 / 0.01325 | 472 / 0.01357 | 762 / 0.01298 | 597 / 0.01323 | 473 / 0.01355 | 2 | 2 |
| AP13 | 2,108 / 0.01552 | 1,341 / 0.02036 | 1,170 / 0.02114 | 855 / 0.02258 | 1,346 / 0.02033 | 1,213 / 0.02106 | 955 / 0.02335 | 2 | 2 |
| AP14 | 4,604 / 0.01209 | 2,658 / 0.01349 | 2,180 / 0.01414 | 1,686 / 0.01409 | 2,705 / 0.01357 | 2,249 / 0.01431 | 1,793 / 0.01425 | 2 | 2 |
| AP18 | 1,351 / 0.01204 | 842 / 0.01487 | 608 / 0.01519 | 320 / 0.01428 | 853 / 0.01505 | 608 / 0.01519 | 320 / 0.01428 | 2 | 2 |
| AP19 | 3,119 / 0.01159 | 1,403 / 0.01324 | 739 / 0.01277 | 84 / 0.04223 | 1,435 / 0.01324 | 774 / 0.01335 | 101 / 0.03917 | 2 | 2 |
| AP1_3dep | 2,373 / 0.01456 | 1,386 / 0.01749 | 1,094 / 0.01894 | 620 / 0.02096 | 1,409 / 0.01767 | 1,118 / 0.01911 | 696 / 0.02042 | 2 | 2 |
| AP2 | 1,429 / 0.0149 | 1,033 / 0.01529 | 733 / 0.01669 | 494 / 0.01707 | 1,044 / 0.0152 | 733 / 0.01669 | 532 / 0.01781 | 2 | 2 |
| AP20 | 5,405 / 0.01236 | 3,473 / 0.0132 | 2,612 / 0.0131 | 1,435 / 0.01367 | 3,520 / 0.01326 | 2,704 / 0.01292 | 1,584 / 0.01262 | 2 | 2 |
| AP23 | 1,309 / 0.0164 | 940 / 0.01745 | 783 / 0.01754 | 616 / 0.01822 | 940 / 0.01745 | 783 / 0.01754 | 653 / 0.018 | 2 | 2 |
| AP24 | 2,212 / 0.01047 | 1,250 / 0.0143 | 956 / 0.01634 | 719 / 0.01589 | 1,286 / 0.01402 | 968 / 0.01614 | 759 / 0.01589 | 2 | 2 |
| AP25 | 718 / 0.01023 | 455 / 0.01041 | 359 / 0.009817 | 290 / 0.009429 | 468 / 0.01058 | 395 / 0.01015 | 326 / 0.009725 | 2 | 2 |
| AP27 | 3,351 / 0.01489 | 2,123 / 0.017 | 1,656 / 0.01815 | 1,164 / 0.01933 | 2,151 / 0.01682 | 1,678 / 0.01794 | 1,219 / 0.01813 | 2 | 2 |
| AP28 | 3,349 / 0.01216 | 2,245 / 0.01247 | 1,850 / 0.01282 | 1,336 / 0.01333 | 2,299 / 0.01254 | 1,918 / 0.01295 | 1,386 / 0.01337 | 2 | 2 |
| AP29 | 2,416 / 0.01045 | 1,344 / 0.01328 | 1,022 / 0.01218 | 463 / 0.01983 | 1,348 / 0.01328 | 1,037 / 0.01235 | 463 / 0.01983 | 2 | 2 |
| AP2_3dep | 1,387 / 0.009227 | 511 / 0.013 | 386 / 0.01326 | 151 / 0.0137 | 514 / 0.01308 | 386 / 0.01326 | 151 / 0.0137 | 2 | 2 |
| AP3 | 2,448 / 0.01833 | 1,326 / 0.02142 | 971 / 0.02165 | 758 / 0.02081 | 1,346 / 0.02149 | 988 / 0.02201 | 786 / 0.02149 | 2 | 2 |
| AP3_3dep | 3,123 / 0.0127 | 1,052 / 0.0149 | 683 / 0.01558 | 399 / 0.01404 | 1,084 / 0.015 | 702 / 0.01594 | 399 / 0.01404 | 2 | 2 |
| AP4 | 3,636 / 0.01726 | 2,829 / 0.01786 | 2,531 / 0.01816 | 2,063 / 0.01837 | 2,896 / 0.01806 | 2,621 / 0.01837 | 2,159 / 0.01869 | 3 | 3 |
| AP4_3dep | 4,286 / 0.0128 | 2,712 / 0.01437 | 2,183 / 0.01479 | 1,403 / 0.01543 | 2,762 / 0.01457 | 2,260 / 0.01522 | 1,420 / 0.0156 | 2 | 2 |
| AP5 | 2,326 / 0.01529 | 1,968 / 0.01556 | 1,582 / 0.01531 | 1,225 / 0.0158 | 1,968 / 0.01556 | 1,582 / 0.01531 | 1,225 / 0.0158 | 3 | 3 |
| AP5_3dep | 2,895 / 0.01012 | 1,945 / 0.01047 | 1,478 / 0.0108 | 1,017 / 0.01167 | 1,947 / 0.01047 | 1,511 / 0.01056 | 1,070 / 0.01114 | 2 | 2 |
| AP6 | 2,468 / 0.01304 | 1,471 / 0.01504 | 1,159 / 0.01548 | 930 / 0.01661 | 1,476 / 0.01504 | 1,170 / 0.01548 | 933 / 0.01658 | 2 | 2 |
| AP7 | 1,737 / 0.01355 | 855 / 0.01441 | 629 / 0.01441 | 260 / 0.01174 | 865 / 0.01441 | 633 / 0.01446 | 300 / 0.01274 | 2 | 2 |
| AP8 | 2,045 / 0.01503 | 1,281 / 0.01985 | 910 / 0.02408 | 371 / 0.02815 | 1,305 / 0.02002 | 948 / 0.02402 | 425 / 0.02845 | 2 | 2 |
| AP9 | 4,078 / 0.01261 | 2,021 / 0.01565 | 1,543 / 0.01605 | 1,118 / 0.01652 | 2,054 / 0.01564 | 1,566 / 0.01608 | 1,127 / 0.01659 | 2 | 2 |
| GM1 | 4,858 / 0.01401 | 3,280 / 0.01401 | 2,669 / 0.01384 | 1,657 / 0.01353 | 3,280 / 0.01401 | 2,669 / 0.01384 | 1,657 / 0.01353 | 3 | 3 |
| GM11 | 3,138 / 0.008588 | 1,958 / 0.008461 | 1,602 / 0.008922 | 1,008 / 0.008829 | 1,971 / 0.008378 | 1,638 / 0.008709 | 1,009 / 0.008832 | 3 | 3 |
| GM12 | 4,932 / 0.01397 | 3,426 / 0.01371 | 2,956 / 0.01372 | 2,226 / 0.01341 | 3,452 / 0.01357 | 2,971 / 0.01366 | 2,291 / 0.01363 | 2 | 2 |
| GM13 | 3,601 / 0.005736 | 2,309 / 0.00602 | 1,685 / 0.00631 | 933 / 0.005294 | 2,309 / 0.00602 | 1,685 / 0.00631 | 933 / 0.005294 | 3 | 2 |
| GM14 | 4,474 / 0.01636 | 2,858 / 0.01872 | 2,260 / 0.01962 | 1,295 / 0.01993 | 2,875 / 0.01868 | 2,296 / 0.01962 | 1,349 / 0.01985 | 2 | 2 |
| GM15 | 3,388 / 0.01717 | 2,374 / 0.01866 | 1,827 / 0.01967 | 1,004 / 0.02014 | 2,434 / 0.01886 | 1,924 / 0.01982 | 1,111 / 0.02099 | 3 | 3 |
| GM16 | 4,062 / 0.01146 | 2,512 / 0.01294 | 1,825 / 0.01244 | 1,073 / 0.01245 | 2,512 / 0.01294 | 1,825 / 0.01244 | 1,073 / 0.01245 | 3 | 3 |
| GM17 | 2,511 / 0.009402 | 2,204 / 0.009996 | 1,994 / 0.01046 | 1,699 / 0.01069 | 2,205 / 0.009997 | 1,995 / 0.01046 | 1,760 / 0.0107 | 3 | 3 |
| GM18 | 1,733 / 0.02427 | 855 / 0.03175 | 615 / 0.03272 | 99 / 0.0347 | 865 / 0.03141 | 638 / 0.03227 | 106 / 0.03438 | 2 | 2 |
| GM19 | 3,309 / 0.01207 | 1,690 / 0.0138 | 1,194 / 0.01508 | 813 / 0.01573 | 1,691 / 0.01378 | 1,194 / 0.01508 | 813 / 0.01573 | 2 | 2 |
| GM1_fema | 5,781 / 0.01441 | 3,647 / 0.01459 | 2,685 / 0.01454 | 1,827 / 0.01478 | 3,647 / 0.01459 | 2,685 / 0.01454 | 1,827 / 0.01478 | 3 | 3 |
| GM2 | 3,453 / 0.01118 | 2,603 / 0.01073 | 2,169 / 0.01097 | 1,338 / 0.01006 | 2,654 / 0.01074 | 2,259 / 0.01118 | 1,433 / 0.01038 | 3 | 3 |
| GM20 | 3,631 / 0.007197 | 2,418 / 0.008119 | 1,546 / 0.009633 | 1,048 / 0.01067 | 2,442 / 0.008115 | 1,643 / 0.009587 | 1,118 / 0.01071 | 3 | 4 |
| GM21 | 4,756 / 0.01358 | 3,354 / 0.01478 | 2,523 / 0.01534 | 1,641 / 0.01574 | 3,375 / 0.01477 | 2,558 / 0.01531 | 1,776 / 0.01597 | 2 | 2 |
| GM22 | 1,411 / 0.009575 | 911 / 0.009884 | 710 / 0.009505 | 469 / 0.01041 | 916 / 0.00992 | 715 / 0.009531 | 516 / 0.01001 | 3 | 3 |
| GM24 | 2,962 / 0.004801 | 1,593 / 0.00483 | 1,094 / 0.00518 | 416 / 0.006478 | 1,603 / 0.00485 | 1,094 / 0.00518 | 416 / 0.006478 | 3 | 3 |
| GM25 | 5,525 / 0.01455 | 3,861 / 0.01546 | 2,732 / 0.01605 | 1,102 / 0.01878 | 3,885 / 0.01538 | 2,761 / 0.01591 | 1,116 / 0.01853 | 4 | 4 |
| GM26 | 1,251 / 0.007849 | 906 / 0.006944 | 803 / 0.006674 | 445 / 0.006962 | 962 / 0.007218 | 870 / 0.007146 | 593 / 0.007592 | 2 | 2 |
| GM27 | 4,317 / 0.02058 | 2,998 / 0.0214 | 2,348 / 0.02209 | 1,442 / 0.02281 | 3,000 / 0.02139 | 2,348 / 0.02209 | 1,442 / 0.02281 | 3 | 3 |
| GM28 | 4,068 / 0.01033 | 3,163 / 0.01025 | 2,725 / 0.01013 | 1,795 / 0.009361 | 3,163 / 0.01025 | 2,725 / 0.01013 | 1,828 / 0.009539 | 2 | 2 |
| GM29 | 5,047 / 0.009325 | 3,526 / 0.01037 | 2,805 / 0.01052 | 2,038 / 0.01099 | 3,545 / 0.01039 | 2,869 / 0.01044 | 2,042 / 0.01101 | 2 | 2 |
| GM2_fema | 3,278 / 0.01148 | 2,341 / 0.01172 | 1,925 / 0.01188 | 1,519 / 0.01209 | 2,365 / 0.01168 | 1,996 / 0.01197 | 1,574 / 0.01222 | 2 | 2 |
| GM3 | 5,713 / 0.007885 | 4,866 / 0.007915 | 3,980 / 0.007866 | 3,312 / 0.008078 | 4,889 / 0.00792 | 3,995 / 0.007874 | 3,340 / 0.008119 | 4 | 4 |
| GM30 | 2,960 / 0.01421 | 2,225 / 0.01523 | 1,888 / 0.01525 | 1,493 / 0.01634 | 2,235 / 0.01519 | 1,898 / 0.01522 | 1,493 / 0.01634 | 2 | 2 |
| GM3_fema | 5,265 / 0.007683 | 4,194 / 0.007787 | 3,504 / 0.00806 | 2,507 / 0.00922 | 4,207 / 0.007791 | 3,556 / 0.008048 | 2,550 / 0.009229 | 3 | 3 |
| GM4 | 3,019 / 0.01294 | 2,216 / 0.01334 | 1,782 / 0.01368 | 1,207 / 0.01166 | 2,216 / 0.01334 | 1,789 / 0.01364 | 1,214 / 0.01163 | 3 | 3 |
| GM4_fema | 3,204 / 0.0134 | 2,222 / 0.01395 | 1,910 / 0.01377 | 1,329 / 0.01434 | 2,222 / 0.01395 | 1,924 / 0.01388 | 1,343 / 0.01441 | 2 | 2 |
| GM5 | 915 / 0.01856 | 585 / 0.01944 | 438 / 0.01979 | 344 / 0.02047 | 592 / 0.01936 | 438 / 0.01979 | 344 / 0.02047 | 2 | 2 |
| GM5_fema | 1,145 / 0.01559 | 646 / 0.01723 | 485 / 0.01848 | 280 / 0.02055 | 656 / 0.01714 | 487 / 0.01844 | 280 / 0.02055 | 2 | 2 |
| GM6 | 4,074 / 0.008669 | 2,550 / 0.007773 | 1,845 / 0.007769 | 1,158 / 0.008401 | 2,554 / 0.007773 | 1,864 / 0.007783 | 1,175 / 0.008438 | 3 | 3 |
| GM6_fema | 3,474 / 0.008688 | 1,857 / 0.009059 | 1,504 / 0.009258 | 973 / 0.01191 | 1,873 / 0.00912 | 1,520 / 0.009445 | 1,033 / 0.01225 | 2 | 2 |
| GM7 | 3,394 / 0.005135 | 2,265 / 0.00515 | 1,725 / 0.005308 | 1,249 / 0.005734 | 2,298 / 0.005178 | 1,725 / 0.005308 | 1,249 / 0.005734 | 3 | 3 |
| GM7_fema | 3,197 / 0.004463 | 2,175 / 0.00463 | 1,647 / 0.004779 | 1,135 / 0.005298 | 2,195 / 0.004675 | 1,670 / 0.004783 | 1,172 / 0.00526 | 3 | 3 |
| GM8 | 4,537 / 0.01001 | 3,952 / 0.01038 | 3,576 / 0.01046 | 2,833 / 0.01061 | 3,959 / 0.01037 | 3,591 / 0.01042 | 2,872 / 0.01047 | 4 | 4 |
| GM9 | 2,337 / 0.006848 | 1,493 / 0.006938 | 1,208 / 0.006873 | 819 / 0.006112 | 1,530 / 0.006881 | 1,226 / 0.006794 | 871 / 0.006278 | 2 | 2 |
| TTV1 | 3,820 / 0.01594 | 2,282 / 0.01474 | 1,673 / 0.01391 | 853 / 0.01266 | 2,429 / 0.01549 | 1,969 / 0.01555 | 1,198 / 0.01488 | 2 | 2 |
| TTV11 | 4,889 / 0.0167 | 2,612 / 0.01806 | 1,938 / 0.01747 | 883 / 0.0177 | 2,661 / 0.01822 | 1,973 / 0.01746 | 921 / 0.01752 | 2 | 2 |
| TTV15 | 4,014 / 0.01405 | 2,108 / 0.01414 | 1,446 / 0.01342 | 779 / 0.01292 | 2,151 / 0.01393 | 1,494 / 0.01312 | 816 / 0.01284 | 2 | 2 |
| TTV2 | 2,919 / 0.01492 | 1,609 / 0.01644 | 1,312 / 0.01617 | 532 / 0.01562 | 1,643 / 0.01636 | 1,326 / 0.01633 | 560 / 0.01635 | 2 | 2 |
| TTV3 | 1,052 / 0.01468 | 708 / 0.01427 | 586 / 0.01444 | 323 / 0.01202 | 734 / 0.01429 | 623 / 0.01433 | 390 / 0.0133 | 2 | 2 |
| TTV4 | 2,648 / 0.01236 | 1,269 / 0.01352 | 899 / 0.01298 | 458 / 0.01352 | 1,321 / 0.01366 | 974 / 0.01339 | 468 / 0.01366 | 2 | 2 |
| TTV5 | 3,967 / 0.02228 | 2,520 / 0.02322 | 1,785 / 0.02365 | 676 / 0.025 | 2,531 / 0.0233 | 1,821 / 0.02403 | 706 / 0.02591 | 2 | 2 |
| TTV6 | 4,927 / 0.01388 | 2,703 / 0.01539 | 1,780 / 0.01413 | 820 / 0.01358 | 2,752 / 0.01531 | 1,831 / 0.01423 | 916 / 0.01263 | 2 | 2 |
| TTV7 | 4,476 / 0.01627 | 2,477 / 0.01992 | 1,572 / 0.02037 | 300 / 0.0272 | 2,490 / 0.01985 | 1,609 / 0.02022 | 391 / 0.02319 | 2 | 2 |
| TTV8 | 5,165 / 0.01757 | 3,383 / 0.01838 | 2,313 / 0.01901 | 1,332 / 0.01883 | 3,400 / 0.01836 | 2,445 / 0.01849 | 1,527 / 0.0183 | 3 | 3 |
| TTV9 | 1,903 / 0.0212 | 1,085 / 0.025 | 794 / 0.02773 | 547 / 0.02983 | 1,125 / 0.0251 | 850 / 0.02737 | 619 / 0.03123 | 2 | 2 |

### D3. Site summaries per group (section 2.4) and rule R1

| site | group | N | usable | counted (median) | D₀,N,A | D₀,N,B | D₀,N,B′ | D₀,N,S | D₀,R,A | D₀,R,B | D₀,N,unf,A | N/R A | N/R B | N/unf A | N/no-convex A | C_ht(L_f) | window test passes (L_w) | passes (L_w′) | D_R | D_R/D_C | below 500 px |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM | primary | 22 | 18 | 1104 | 0.01286 | 0.0123 | 0.01232 | 0.008401 | 0.01489 | 0.01411 | 0.01509 | 0.91 | 0.92 | 0.93 | 0.988 | 0.01145 | 5 (0 trivial) | 8 (1 trivial) | 0.02981 | 2.04 | GM5_fema, GM22, GM24, GM26 |
| GM | clean | 19 | 16 | 1073 | 0.01304 | 0.01202 | 0.0128 | - | 0.01534 | 0.01498 | 0.01509 | 0.90 | 0.93 | 0.93 | 0.988 | 0.0113 | 5 (0 trivial) | 8 (1 trivial) | 0.02987 | 1.98 | GM5_fema, GM22, GM24 |
| GM | offmesa | 6 | 5 | 1198 | 0.008996 | 0.01053 | 0.01053 | - | 0.01114 | 0.01185 | 0.01034 | 0.79 | 0.88 | 0.82 | 0.972 | 0.01634 | 0 (0 trivial) | 1 (0 trivial) | 0.02822 | 2.64 | GM18 |
| GM | ncalm | 7 | 6 | 1249 | 0.01605 | 0.01531 | 0.01605 | - | 0.009986 | 0.01013 | 0.01476 | 1.39 | 1.32 | 1.05 | 0.963 | 0.009233 | 1 (0 trivial) | 6 (5 trivial) | 0.02644 | 1.77 | GM5 |
| AP | primary | 21 | 12 | 719 | 0.003513 | 0.005732 | 0.0043 | 0.002015 | 0.004066 | 0.005499 | 0.004361 | 0.87 | 1.05 | 0.86 | 0.977 | 0.01566 | 0 (0 trivial) | 5 (5 trivial) | 0.01316 | 3.34 | AP2_3dep, AP3_3dep, AP7, AP8, AP11, AP18, AP19, AP25, AP29 |
| AP | clean | 20 | 12 | 787 | 0.003513 | 0.005732 | 0.0043 | - | 0.004066 | 0.005499 | 0.004361 | 0.87 | 1.05 | 0.86 | 0.977 | 0.01566 | 0 (0 trivial) | 5 (5 trivial) | 0.01316 | 3.34 | AP2_3dep, AP3_3dep, AP7, AP8, AP11, AP18, AP19, AP29 |
| AP | pamap | 5 | 4 | 763 | 0.003237 | 0.00559 | 0.004741 | - | 0.003254 | 0.005217 | 0.003154 | 1.08 | 1.33 | 0.95 | 0.979 | 0.01709 | 0 (0 trivial) | 1 (1 trivial) | 0.01245 | 3.85 | AP2 |
| AP | ap1_3dep | 1 | 1 | 620 | 0.002625 | 0.005119 | 0.003926 | - | 0.005507 | 0.006664 | 0.003693 | 0.48 | 0.77 | 0.71 | 0.962 | 0.02096 | 0 (0 trivial) | 0 (0 trivial) | 0.01272 | 4.90 |  |
| TTV | primary | 11 | 8 | 676 | 0.00464 | 0.007523 | 0.005129 | 0.002917 | 0.002701 | 0.002802 | 0.004425 | 1.56 | 2.28 | 1.01 | 0.950 | 0.01666 | 0 (0 trivial) | 6 (5 trivial) | 0.01479 | 3.36 | TTV3, TTV4, TTV7 |
| TTV | clean | 10 | 7 | 612 | 0.004931 | 0.007902 | 0.005328 | - | 0.002905 | 0.002905 | 0.004502 | 1.23 | 1.76 | 1.00 | 0.946 | 0.01562 | 0 (0 trivial) | 5 (4 trivial) | 0.01451 | 3.37 | TTV3, TTV4, TTV7 |

Rule R1 (hilltop gate): at least 500 counted pixels on at least 90 % of the primary tiles.

| site | tiles ≥ 500 px | N | needed | count part | tiles below 500 |
|---|---|---|---|---|---|
| GM | 18 | 22 | 20 | **FAIL** | GM5_fema, GM22, GM24, GM26 |
| AP | 12 | 21 | 19 | **FAIL** | AP2_3dep, AP3_3dep, AP7, AP8, AP11, AP18, AP19, AP25, AP29 |
| TTV | 8 | 11 | 10 | **FAIL** | TTV3, TTV4, TTV7 |

| site | N | needed | no segment filter | frag ≥ 10 | frag ≥ 20 | frag ≥ 50 (as written) | parent ≥ 20 | parent ≥ 50 |
|---|---|---|---|---|---|---|---|---|
| GM | 22 | 20 | 22 pass | 22 pass | 21 pass | 18 fail | 21 pass | 20 pass |
| AP | 21 | 19 | 21 pass | 20 pass | 19 pass | 12 fail | 19 pass | 12 fail |
| TTV | 11 | 10 | 11 pass | 11 pass | 11 pass | 8 fail | 11 pass | 8 fail |

### D4. The same-pixel diagnostic on the D₀,N sets (section 2.2, rule R3)

| pair | n old | n new | (a) | (b) | (c) | (d) | surface (b)/(a), (d)/(c) | selection (c)/(a), (d)/(b) | (d)/(a) | both ≥ 500 | offset (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GM1/GM1_fema | 1,657 | 1,827 | 0.01353 | 0.01336 | 0.01459 | 0.01478 | 0.99, 1.01 | 1.08, 1.11 | 1.09 | yes | 0.00 |
| GM2/GM2_fema | 1,338 | 1,519 | 0.01006 | 0.01002 | 0.01161 | 0.01209 | 1.00, 1.04 | 1.15, 1.21 | 1.20 | yes | 0.00 |
| GM3/GM3_fema | 3,312 | 2,507 | 0.008078 | 0.007327 | 0.008725 | 0.00922 | 0.91, 1.06 | 1.08, 1.26 | 1.14 | yes | 0.00 |
| GM4/GM4_fema | 1,207 | 1,329 | 0.01166 | 0.0107 | 0.01585 | 0.01434 | 0.92, 0.90 | 1.36, 1.34 | 1.23 | yes | 0.00 |
| GM5/GM5_fema | 344 | 280 | 0.02047 | 0.02028 | 0.02083 | 0.02055 | 0.99, 0.99 | 1.02, 1.01 | 1.00 | no | 0.00 |
| GM6/GM6_fema | 1,158 | 973 | 0.008401 | 0.007508 | 0.01081 | 0.01191 | 0.89, 1.10 | 1.29, 1.59 | 1.42 | yes | 0.00 |
| GM7/GM7_fema | 1,249 | 1,135 | 0.005734 | 0.005056 | 0.005316 | 0.005298 | 0.88, 1.00 | 0.93, 1.05 | 0.92 | yes | 0.00 |
| AP1/AP1_3dep | 763 | 620 | 0.01574 | 0.015 | 0.01428 | 0.02096 | 0.95, 1.47 | 0.91, 1.40 | 1.33 | yes | 0.37 |
| AP2/AP2_3dep | 494 | 151 | 0.01707 | 0.0142 | 0.0149 | 0.0137 | 0.83, 0.92 | 0.87, 0.96 | 0.80 | no | 0.39 |
| AP3/AP3_3dep | 758 | 399 | 0.02081 | 0.01537 | 0.01715 | 0.01404 | 0.74, 0.82 | 0.82, 0.91 | 0.67 | no | 0.42 |
| AP4/AP4_3dep | 2,063 | 1,403 | 0.01837 | 0.01283 | 0.01567 | 0.01543 | 0.70, 0.99 | 0.85, 1.20 | 0.84 | yes | 0.42 |
| AP5/AP5_3dep | 1,225 | 1,017 | 0.0158 | 0.01202 | 0.01253 | 0.01167 | 0.76, 0.93 | 0.79, 0.97 | 0.74 | yes | 0.39 |

| site | S | P | (d)/(a) median | reading | pairs |
|---|---|---|---|---|---|
| GM | 0.989 | 1.130 | 1.14 | mixed | GM1/GM1_fema, GM2/GM2_fema, GM3/GM3_fema, GM4/GM4_fema, GM5/GM5_fema, GM6/GM6_fema, GM7/GM7_fema |
| AP | 0.876 | 0.910 | 0.80 | mixed | AP1/AP1_3dep, AP2/AP2_3dep, AP3/AP3_3dep, AP4/AP4_3dep, AP5/AP5_3dep |
| AP_without_AP2 | 0.875 | 0.910 | 0.79 | mixed | AP1/AP1_3dep, AP3/AP3_3dep, AP4/AP4_3dep, AP5/AP5_3dep |
| GM_usable_pairs_only | 0.991 | 1.180 | 1.17 | mixed | GM1/GM1_fema, GM2/GM2_fema, GM3/GM3_fema, GM4/GM4_fema, GM6/GM6_fema, GM7/GM7_fema |
| AP_usable_pairs_only | 0.942 | 0.939 | 0.84 | mixed | AP1/AP1_3dep, AP4/AP4_3dep, AP5/AP5_3dep |

### D5. Calibration against Perron et al. (2012) (section 2.5, rule R11)

**GM** (E 0.000147 m/yr; Perron: E 147.0 m/Myr, |∇²z_h| 0.0118 m⁻¹, D 124.0 ± 19.0 cm²/yr; E matches: True); pooled pixels 28,664 of 22 primary tiles.

| L (m) | pooled |C_ht| | 68 % | ratio to Perron | 68 % | in 0.67-1.5 |
|---|---|---|---|---|---|
| 3 | 0.02921 | [0.0283, 0.03] | 2.48 | [2.40, 2.54] | no |
| 5 | 0.01906 | [0.0183, 0.0198] | 1.62 | [1.55, 1.67] | no |
| 7 | 0.01474 | [0.0141, 0.0153] | 1.25 | [1.20, 1.29] | yes |
| 9 | 0.01241 | [0.0119, 0.0129] | 1.05 | [1.01, 1.09] | yes |
| 11 | 0.01117 | [0.0107, 0.0117] | 0.95 | [0.91, 0.99] | yes |
| 13 | 0.01074 | [0.0102, 0.0112] | 0.91 | [0.86, 0.95] | yes |
| 15 | 0.01055 | [0.01, 0.0109] | 0.89 | [0.85, 0.93] | yes |
| 17 | 0.01045 | [0.00994, 0.0108] | 0.89 | [0.84, 0.92] | yes |
| 19 | 0.01046 | [0.00992, 0.0108] | 0.89 | [0.84, 0.92] | yes |
| 21 | 0.01051 | [0.00999, 0.0109] | 0.89 | [0.85, 0.92] | yes |
| 23 | 0.01055 | [0.00999, 0.0109] | 0.89 | [0.85, 0.92] | yes |
| 25 | 0.01058 | [0.01, 0.011] | 0.90 | [0.85, 0.93] | yes |
| 27 | 0.01062 | [0.01, 0.0111] | 0.90 | [0.85, 0.94] | yes |
| 29 | 0.01066 | [0.0101, 0.0111] | 0.90 | [0.85, 0.94] | yes |
| 31 | 0.01069 | [0.0101, 0.0111] | 0.91 | [0.86, 0.94] | yes |
| 41 | 0.01081 | [0.0103, 0.0112] | 0.92 | [0.87, 0.95] | yes |
| 51 | 0.01055 | [0.01, 0.0111] | 0.89 | [0.85, 0.94] | yes |

At each tile's L_w: pooled |C_ht| 0.01055 [0.01, 0.0111], ratio 0.89 [0.85, 0.94]. Tile-median C_ht(L_f) 0.01145. Legacy (manuscript, approximate) route: C_ht site median 0.01131 (range [0.00573, 0.0158]), D₀ 0.01304, ratio to Perron 0.96; approximation (lemcvt.measure.legacy_v1.legacy_curvature: block-mean grid, coarse D8, divides between coarse basins, 3x3 Laplacian; the topo_analysis first-order-divide module is not available, as the package labels it).

**AP** (E 5.5e-05 m/yr; Perron: E 55.0 m/Myr, |∇²z_h| 0.0055 m⁻¹, D 100.0 ± 8.0 cm²/yr; E matches: True); pooled pixels 16,438 of 21 primary tiles.

| L (m) | pooled |C_ht| | 68 % | ratio to Perron | 68 % | in 0.67-1.5 |
|---|---|---|---|---|---|
| 3 | 0.03843 | [0.0369, 0.0408] | 6.99 | [6.71, 7.42] | no |
| 5 | 0.02609 | [0.0249, 0.0277] | 4.74 | [4.53, 5.03] | no |
| 7 | 0.0205 | [0.0195, 0.0217] | 3.73 | [3.55, 3.94] | no |
| 9 | 0.01737 | [0.0165, 0.0184] | 3.16 | [3.00, 3.35] | no |
| 11 | 0.01544 | [0.0146, 0.0163] | 2.81 | [2.65, 2.96] | no |
| 13 | 0.01426 | [0.0135, 0.0152] | 2.59 | [2.46, 2.76] | no |
| 15 | 0.01352 | [0.0128, 0.0144] | 2.46 | [2.33, 2.61] | no |
| 17 | 0.01291 | [0.0123, 0.0136] | 2.35 | [2.24, 2.47] | no |
| 19 | 0.01234 | [0.0118, 0.013] | 2.24 | [2.14, 2.37] | no |
| 21 | 0.01192 | [0.0113, 0.0126] | 2.17 | [2.06, 2.29] | no |
| 23 | 0.01163 | [0.011, 0.0123] | 2.11 | [2.01, 2.24] | no |
| 25 | 0.01141 | [0.0109, 0.0121] | 2.07 | [1.98, 2.19] | no |
| 27 | 0.01124 | [0.0108, 0.0118] | 2.04 | [1.96, 2.15] | no |
| 29 | 0.01103 | [0.0106, 0.0116] | 2.01 | [1.92, 2.10] | no |
| 31 | 0.01086 | [0.0104, 0.0113] | 1.98 | [1.89, 2.06] | no |
| 41 | 0.01022 | [0.00985, 0.0106] | 1.86 | [1.79, 1.93] | no |
| 51 | 0.009837 | [0.00953, 0.0101] | 1.79 | [1.73, 1.84] | no |

At each tile's L_w: pooled |C_ht| 0.00983 [0.0095, 0.0101], ratio 1.79 [1.73, 1.84]. Tile-median C_ht(L_f) 0.01566. Legacy (manuscript, approximate) route: C_ht site median 0.0059 (range [0.00371, 0.00698]), D₀ 0.009322, ratio to Perron 1.07; approximation (lemcvt.measure.legacy_v1.legacy_curvature: block-mean grid, coarse D8, divides between coarse basins, 3x3 Laplacian; the topo_analysis first-order-divide module is not available, as the package labels it).

| site | ratio at L_f | 68 % | R11 | windows in range (m) | D statements |
|---|---|---|---|---|---|
| GM | 0.95 | [0.91, 0.99] | pass | 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 41, 51 | run |
| AP | 2.81 | [2.65, 2.96] | **FAIL** | - | **not run** |
| TTV | - | - | not gated | - | run |

### D6. The window scaling break (section 2.6, rule R12) and D₀,N,S

| site | L_break (m) | 68 % | 95 % | log slope below | log slope above | L_S (m) | D₀,N,S median | D₀,N,S/D₀,N,A median | range | L_f in 68 % | L_break in 8-20 m | R12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM | 5.8 | [5.8, 5.9] | [5.7, 5.9] | -2.29 | -0.56 | 5 | 0.008401 | 0.63 | [0.42, 0.91] | no | no | **not consistent** |
| AP | 5.3 | [5.3, 5.3] | [5.2, 5.4] | -1.96 | -1.00 | 5 | 0.002015 | 0.57 | [0.42, 0.75] | no | no | **not consistent** |
| TTV | 6.1 | [6.1, 6.2] | [6.0, 6.2] | -2.36 | -0.80 | 7 | 0.002917 | 0.68 | [0.50, 0.86] | no | no | **not consistent** |

GM pooled median σ_C(L)/σ_C(11 m) at L = 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 m: 6.864, 2.109, 1.417, 1.156, 1.000, 0.900, 0.829, 0.778, 0.737, 0.708, 0.683, 0.660, 0.641, 0.621, 0.603

AP pooled median σ_C(L)/σ_C(11 m) at L = 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 m: 6.372, 2.344, 1.579, 1.218, 1.000, 0.850, 0.738, 0.649, 0.582, 0.527, 0.481, 0.443, 0.413, 0.387, 0.363

TTV pooled median σ_C(L)/σ_C(11 m) at L = 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31 m: 8.878, 2.656, 1.631, 1.219, 1.000, 0.864, 0.771, 0.696, 0.638, 0.596, 0.561, 0.532, 0.507, 0.485, 0.465

### D7. D_R, D_R/D_C, E* and the bracket per tile (section 2.7, decisions 12-13)

| tile | traced | used | left tile | L_H (m) | R (m) | D_R | 68 % | D_R (Euclid) | D_R/D_C | 2S̄/(C L_H) | E* (S_c 1.27) | E* (fit S_c) | E* bands < 10 / 10-30 / > 30 | D_R picked net | 68 % | n used | L_H picked (m) | D_R model P4 net | L_H P4 (m) | A_crit P4 | D P4 | bracket [D_C,lo, D_R,hi] |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | 763 | 763 | 0 | 84 | 17.4 | 0.01109 | [0.00958, 0.0124] | 0.00933 | 3.11 | 0.32 | 2.0 | - | 0.99 / 0.01 / 0.00 | - | - | - | - | 0.02033 | 161 | 6889 | 0.01699 | [0.00321, 0.0124] |
| AP10 | 1,349 | 1,349 | 0 | 121 | 30.7 | 0.01273 | [0.0119, 0.0137] | 0.01065 | 3.25 | 0.31 | 2.5 | 11.0 | 0.99 / 0.01 / 0.00 | - | - | - | - | 0.02357 | 203 | 9381 | 0.02045 | [0.00338, 0.0137] |
| AP11 | 472 | 472 | 0 | 121 | 29.6 | 0.01528 | [0.0149, 0.0159] | 0.0128 | 3.84 | 0.26 | 2.6 | 11.3 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.02086 | 178 | 3490 | 0.01763 | undetermined |
| AP13 | 855 | 855 | 0 | 110 | 29.6 | 0.01182 | [0.0104, 0.0123] | 0.009973 | 4.50 | 0.22 | 3.5 | 15.1 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.01761 | 170 | 6085 | 0.01763 | [0.00221, 0.0123] |
| AP14 | 1,686 | 1,686 | 0 | 109 | 23.6 | 0.01379 | [0.0133, 0.0143] | 0.01117 | 3.65 | 0.27 | 2.3 | 10.1 | 0.99 / 0.01 / 0.00 | - | - | - | - | 0.02523 | 211 | 7631 | 0.02326 | [0.00347, 0.0143] |
| AP18 | 320 | 320 | 0 | 97 | 14.5 | 0.01634 | [0.0161, 0.0179] | 0.01337 | 5.79 | 0.17 | 2.2 | 9.6 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.03124 | 236 | 14274 | 0.02964 | undetermined |
| AP19 | 84 | 84 | 0 | 114 | 32.2 | 0.0103 | - | 0.008709 | 6.61 | 0.15 | 6.8 | 29.5 | 0.88 / 0.12 / 0.00 | - | - | - | - | 0.01604 | 181 | 3402 | 0.01734 | undetermined |
| AP1_3dep | 620 | 620 | 0 | 112 | 26.5 | 0.01272 | [0.012, 0.0144] | 0.01001 | 4.90 | 0.20 | 2.9 | - | 0.87 / 0.13 / 0.00 | - | - | - | - | 0.02102 | 170 | 8405 | 0.01786 | [0.00212, 0.0144] |
| AP2 | 494 | 494 | 0 | 107 | 29.0 | 0.01397 | [0.0117, 0.0158] | 0.01129 | 4.26 | 0.23 | 3.4 | - | 0.99 / 0.01 / 0.00 | - | - | - | - | 0.0193 | 168 | 3121 | 0.0168 | undetermined |
| AP20 | 1,435 | 1,435 | 0 | 112 | 22.5 | 0.01478 | [0.0142, 0.0154] | 0.01121 | 3.29 | 0.30 | 1.9 | 8.4 | 0.99 / 0.01 / 0.00 | - | - | - | - | 0.01961 | 151 | 2690 | 0.0169 | [0.00347, 0.0154] |
| AP23 | 616 | 616 | 0 | 118 | 31.8 | 0.0109 | [0.0103, 0.0114] | 0.008608 | 3.33 | 0.30 | 2.9 | 12.8 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.01987 | 219 | 6014 | 0.0167 | [0.00282, 0.0114] |
| AP24 | 719 | 719 | 0 | 93 | 12.8 | 0.01339 | [0.0127, 0.0148] | 0.01098 | 4.36 | 0.23 | 2.5 | 11.0 | 0.83 / 0.17 / 0.00 | - | - | - | - | 0.02033 | 149 | 4516 | 0.0203 | [0.00111, 0.0148] |
| AP25 | 290 | 290 | 0 | 53 | 0.4 | 0.03575 | - | 0.02263 | 9.89 | 0.10 | 0.7 | 3.0 | 0.94 / 0.06 / 0.00 | - | - | - | - | 0.02977 | 204 | 9624 | 0.02043 | undetermined |
| AP27 | 1,164 | 1,164 | 0 | 122 | 31.4 | 0.0131 | [0.0127, 0.0136] | 0.01059 | 4.63 | 0.22 | 3.6 | 15.7 | 0.94 / 0.06 / 0.00 | - | - | - | - | 0.01765 | 169 | 5110 | 0.016 | [0.00218, 0.0136] |
| AP28 | 1,336 | 1,336 | 0 | 103 | 23.2 | 0.01304 | [0.0125, 0.0138] | 0.0108 | 2.78 | 0.36 | 2.1 | 8.9 | 0.98 / 0.02 / 0.00 | - | - | - | - | 0.02096 | 168 | 4522 | 0.01891 | [0.00363, 0.0138] |
| AP29 | 463 | 463 | 0 | 111 | 23.4 | 0.01465 | [0.0134, 0.0158] | 0.01217 | 4.24 | 0.24 | 2.6 | 11.3 | 0.96 / 0.04 / 0.00 | - | - | - | - | 0.02966 | 238 | 16545 | 0.02325 | undetermined |
| AP2_3dep | 151 | 151 | 0 | 126 | 26.0 | 0.01642 | - | 0.0143 | 4.58 | 0.22 | 3.3 | 14.5 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.01973 | 172 | 7518 | 0.02103 | undetermined |
| AP3 | 758 | 758 | 0 | 114 | 28.7 | 0.01299 | [0.0123, 0.0139] | 0.0112 | 5.05 | 0.20 | 3.3 | - | 0.95 / 0.05 / 0.00 | - | - | - | - | 0.01853 | 159 | 4742 | 0.01791 | [0.00221, 0.0139] |
| AP3_3dep | 399 | 399 | 0 | 133 | 36.9 | 0.01406 | [0.0127, 0.0146] | 0.01139 | 3.71 | 0.27 | 3.2 | 13.7 | 0.87 / 0.13 / 0.00 | - | - | - | - | 0.02147 | 205 | 14529 | 0.0234 | undetermined |
| AP4 | 2,063 | 2,063 | 0 | 85 | 20.6 | 0.01354 | [0.0127, 0.0142] | 0.01086 | 4.29 | 0.23 | 2.3 | - | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.02183 | 182 | 7232 | 0.01974 | [0.0028, 0.0142] |
| AP4_3dep | 1,403 | 1,403 | 0 | 89 | 13.7 | 0.01459 | [0.0129, 0.0155] | 0.01097 | 3.11 | 0.32 | 1.5 | 6.5 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.02228 | 155 | 5144 | 0.01864 | [0.00298, 0.0155] |
| AP5 | 1,225 | 1,225 | 0 | 100 | 25.0 | 0.01191 | [0.0109, 0.0124] | 0.009997 | 3.41 | 0.29 | 2.6 | - | 0.99 / 0.01 / 0.00 | - | - | - | - | 0.01461 | 139 | 1887 | 0.01258 | [0.00292, 0.0124] |
| AP5_3dep | 1,017 | 1,017 | 0 | 112 | 27.7 | 0.01379 | [0.0129, 0.015] | 0.01104 | 3.34 | 0.30 | 2.2 | 9.7 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.01899 | 168 | 2629 | 0.01464 | [0.00391, 0.015] |
| AP6 | 930 | 930 | 0 | 106 | 28.0 | 0.01105 | [0.0101, 0.0123] | 0.008525 | 3.26 | 0.31 | 2.5 | 11.0 | 0.98 / 0.02 / 0.00 | - | - | - | - | 0.02081 | 194 | 11491 | 0.017 | [0.00251, 0.0123] |
| AP7 | 260 | 260 | 0 | 115 | 31.3 | 0.01253 | [0.0115, 0.0129] | 0.01015 | 2.89 | 0.35 | 1.9 | 8.3 | 0.80 / 0.20 / 0.00 | - | - | - | - | 0.01979 | 183 | 5594 | 0.01817 | undetermined |
| AP8 | 371 | 371 | 0 | 118 | 34.4 | 0.01236 | - | 0.01037 | 5.93 | 0.17 | 5.3 | 22.9 | 0.95 / 0.05 / 0.00 | - | - | - | - | 0.02293 | 208 | 9172 | 0.02536 | undetermined |
| AP9 | 1,118 | 1,118 | 0 | 99 | 20.7 | 0.01321 | [0.012, 0.0141] | 0.01047 | 3.70 | 0.27 | 2.5 | 10.9 | 0.99 / 0.01 / 0.00 | - | - | - | - | 0.02229 | 176 | 5703 | 0.01918 | [0.00302, 0.0141] |
| GM1 | 1,657 | 1,657 | 0 | 75 | 24.9 | 0.01741 | [0.0165, 0.0185] | 0.01528 | 1.59 | 0.63 | 1.6 | - | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.01757 | 78 | 1241 | 0.01934 | [0.00962, 0.0185] |
| GM11 | 1,008 | 1,008 | 0 | 111 | 28.0 | 0.03736 | [0.0333, 0.0404] | 0.02935 | 2.09 | 0.48 | 1.5 | 5.0 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.04177 | 129 | 3361 | 0.02848 | [0.0147, 0.0404] |
| GM12 | 2,226 | 2,226 | 0 | 111 | 31.2 | 0.02736 | [0.0254, 0.0306] | 0.02168 | 2.76 | 0.36 | 2.2 | 7.4 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.03244 | 132 | 4923 | 0.02736 | [0.00936, 0.0306] |
| GM13 | 933 | 933 | 0 | 111 | 21.9 | 0.03858 | [0.0327, 0.0421] | 0.02961 | 1.42 | 0.70 | 0.9 | 3.1 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.0415 | 122 | 3363 | 0.03626 | [0.0233, 0.0421] |
| GM14 | 1,295 | 1,295 | 0 | 60 | 7.0 | 0.03819 | [0.0327, 0.0421] | 0.02736 | 5.10 | 0.20 | 2.0 | - | 0.95 / 0.05 / 0.00 | - | - | - | - | 0.05506 | 94 | 3482 | 0.0295 | [0.00622, 0.0421] |
| GM15 | 1,004 | 1,004 | 0 | 72 | 17.1 | 0.0195 | [0.0173, 0.0219] | 0.01589 | 2.78 | 0.36 | 2.5 | 8.3 | 0.99 / 0.01 / 0.00 | - | - | - | - | 0.02213 | 78 | 1805 | 0.02131 | [0.00647, 0.0219] |
| GM16 | 1,073 | 1,073 | 0 | 85 | 24.0 | 0.02177 | [0.0206, 0.0245] | 0.01775 | 1.97 | 0.51 | 1.6 | 5.3 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.03537 | 130 | 5988 | 0.03172 | [0.00924, 0.0245] |
| GM17 | 1,699 | 1,699 | 0 | 104 | 22.3 | 0.03235 | [0.0294, 0.036] | 0.02516 | 2.23 | 0.45 | 1.5 | 5.1 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.03029 | 87 | 651 | 0.01558 | [0.0127, 0.036] |
| GM18 | 99 | 99 | 0 | 53 | 11.5 | 0.02633 | - | 0.0213 | 6.36 | 0.16 | 2.9 | - | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.03592 | 76 | 1859 | 0.02438 | undetermined |
| GM19 | 813 | 813 | 0 | 91 | 28.5 | 0.02822 | [0.0238, 0.0305] | 0.02299 | 2.60 | 0.38 | 2.1 | - | 0.94 / 0.06 / 0.00 | - | - | - | - | 0.04122 | 161 | 14154 | 0.05 | [0.00816, 0.0305] |
| GM1_fema | 1,827 | 1,827 | 0 | 79 | 22.8 | 0.01962 | [0.0186, 0.0204] | 0.01536 | 1.91 | 0.52 | 1.8 | 5.9 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.01993 | 79 | 1213 | 0.01986 | [0.00917, 0.0204] |
| GM2 | 1,338 | 1,338 | 0 | 101 | 34.5 | 0.0225 | [0.0203, 0.0284] | 0.01995 | 1.68 | 0.60 | 1.6 | - | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.02647 | 112 | 3517 | 0.02735 | [0.0128, 0.0284] |
| GM20 | 1,048 | 1,048 | 0 | 88 | 20.0 | 0.03011 | [0.0281, 0.0331] | 0.02417 | 2.59 | 0.39 | 1.4 | 4.8 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.0308 | 114 | 7037 | 0.05 | [0.0119, 0.0331] |
| GM21 | 1,641 | 1,641 | 0 | 99 | 24.5 | 0.03063 | [0.0267, 0.0331] | 0.02282 | 2.84 | 0.35 | 2.2 | 7.5 | 0.98 / 0.02 / 0.00 | - | - | - | - | 0.03509 | 123 | 2038 | 0.01985 | [0.00808, 0.0331] |
| GM22 | 469 | 469 | 0 | 121 | 30.8 | 0.0334 | [0.0318, 0.0348] | 0.02792 | 2.63 | 0.38 | 1.9 | 6.2 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.03913 | 142 | 5818 | 0.02879 | undetermined |
| GM24 | 416 | 416 | 0 | 93 | 14.6 | 0.0591 | [0.0462, 0.07] | 0.04671 | 1.71 | 0.58 | 0.9 | 2.9 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.06497 | 122 | 4368 | 0.04309 | undetermined |
| GM25 | 1,102 | 1,102 | 0 | 72 | 15.3 | 0.02408 | [0.0211, 0.0273] | 0.01659 | 2.64 | 0.38 | 2.0 | - | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.02826 | 104 | 13881 | 0.03298 | [0.00687, 0.0273] |
| GM26 | 445 | 445 | 0 | 96 | 14.7 | 0.04154 | [0.0371, 0.0488] | 0.03207 | 2.12 | 0.47 | 1.0 | 3.4 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.04161 | 115 | 4364 | 0.02668 | undetermined |
| GM27 | 1,442 | 1,442 | 0 | 87 | 26.7 | 0.02009 | [0.0184, 0.0211] | 0.01462 | 2.84 | 0.35 | 2.9 | 9.8 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.02112 | 94 | 1923 | 0.01756 | [0.00612, 0.0211] |
| GM28 | 1,795 | 1,795 | 0 | 99 | 23.3 | 0.03838 | [0.0315, 0.0466] | 0.03109 | 2.60 | 0.39 | 1.7 | - | 0.89 / 0.11 / 0.00 | - | - | - | - | 0.05252 | 165 | 11271 | 0.03443 | [0.0126, 0.0466] |
| GM29 | 2,038 | 2,038 | 0 | 110 | 28.8 | 0.02997 | [0.0253, 0.0373] | 0.02276 | 2.90 | 0.35 | 2.2 | 7.4 | 0.99 / 0.01 / 0.00 | - | - | - | - | 0.0335 | 129 | 1323 | 0.02362 | [0.0103, 0.0373] |
| GM2_fema | 1,519 | 1,519 | 0 | 94 | 32.5 | 0.02236 | [0.0207, 0.0255] | 0.01889 | 1.90 | 0.53 | 1.8 | 6.1 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.0253 | 101 | 1519 | 0.02328 | [0.0106, 0.0255] |
| GM3 | 3,312 | 3,312 | 0 | 77 | 13.3 | 0.03037 | [0.0288, 0.0335] | 0.02433 | 1.77 | 0.56 | 1.0 | - | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.02976 | 75 | 860 | 0.01898 | [0.0164, 0.0335] |
| GM30 | 1,493 | 1,493 | 0 | 86 | 25.1 | 0.02563 | [0.0238, 0.0274] | 0.01944 | 2.86 | 0.35 | 2.3 | - | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.04557 | 189 | 10691 | 0.04608 | [0.00817, 0.0274] |
| GM3_fema | 2,507 | 2,507 | 0 | 75 | 11.9 | 0.03254 | [0.0289, 0.0388] | 0.02543 | 1.98 | 0.50 | 1.1 | 3.6 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.03001 | 71 | 698 | 0.018 | [0.0137, 0.0388] |
| GM4 | 1,207 | 1,207 | 0 | 87 | 28.2 | 0.02071 | [0.0187, 0.0237] | 0.01848 | 1.78 | 0.56 | 1.7 | - | 0.97 / 0.03 / 0.00 | - | - | - | - | 0.02296 | 111 | 3855 | 0.02147 | [0.00982, 0.0237] |
| GM4_fema | 1,329 | 1,329 | 0 | 100 | 32.7 | 0.02161 | [0.0196, 0.0229] | 0.01788 | 1.94 | 0.52 | 2.1 | 7.0 | 0.98 / 0.02 / 0.00 | - | - | - | - | 0.02399 | 113 | 2319 | 0.01986 | [0.00839, 0.0229] |
| GM5 | 344 | 344 | 0 | 87 | 35.1 | 0.01674 | - | 0.01439 | 2.44 | 0.41 | 3.0 | - | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.02139 | 111 | 3108 | 0.02118 | undetermined |
| GM5_fema | 280 | 280 | 0 | 112 | 40.5 | 0.02046 | - | 0.01812 | 2.82 | 0.35 | 3.3 | 11.2 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.02188 | 119 | 2017 | 0.02091 | undetermined |
| GM6 | 1,158 | 1,158 | 0 | 98 | 25.3 | 0.03041 | [0.025, 0.0356] | 0.02457 | 1.78 | 0.56 | 1.3 | - | 0.99 / 0.01 / 0.00 | - | - | - | - | 0.03507 | 132 | 3588 | 0.02752 | [0.0129, 0.0356] |
| GM6_fema | 973 | 973 | 0 | 95 | 27.2 | 0.02411 | [0.0224, 0.0323] | 0.01962 | 1.90 | 0.53 | 1.7 | 5.7 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.04578 | 164 | 10533 | 0.04431 | [0.0104, 0.0323] |
| GM7 | 1,249 | 1,249 | 0 | 97 | 14.9 | 0.04984 | [0.0445, 0.0574] | 0.03931 | 1.84 | 0.54 | 0.9 | - | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.05601 | 123 | 4302 | 0.047 | [0.0242, 0.0574] |
| GM7_fema | 1,135 | 1,135 | 0 | 113 | 16.5 | 0.0601 | [0.0522, 0.0634] | 0.05001 | 1.95 | 0.51 | 0.9 | 2.9 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.06129 | 136 | 6042 | 0.04669 | [0.0255, 0.0634] |
| GM8 | 2,833 | 2,833 | 0 | 100 | 24.1 | 0.02964 | [0.0273, 0.0317] | 0.0224 | 2.30 | 0.43 | 1.6 | 5.5 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.03641 | 130 | 15244 | 0.04201 | [0.0129, 0.0317] |
| GM9 | 819 | 819 | 0 | 99 | 21.0 | 0.03715 | [0.0346, 0.0408] | 0.02886 | 1.69 | 0.59 | 1.0 | 3.2 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.04473 | 128 | 5912 | 0.03308 | [0.0212, 0.0408] |
| TTV1 | 853 | 853 | 0 | 73 | 9.8 | 0.01977 | [0.0179, 0.0234] | 0.01488 | 3.47 | 0.29 | 1.3 | 6.1 | 1.00 / 0.00 / 0.00 | 0.03212 | [0.0285, 0.0353] | 194 | 202 | 0.05053 | 353 | 59499 | 0.04388 | [0.00501, 0.0234] |
| TTV11 | 883 | 883 | 0 | 67 | 14.3 | 0.01506 | [0.0131, 0.017] | 0.01161 | 3.22 | 0.31 | 2.1 | 9.5 | 0.99 / 0.01 / 0.00 | - | - | - | - | 0.04753 | 309 | 30362 | 0.04108 | [0.00386, 0.017] |
| TTV15 | 779 | 779 | 0 | 71 | 12.9 | 0.01798 | [0.0167, 0.0194] | 0.01275 | 2.85 | 0.35 | 1.5 | 7.0 | 1.00 / 0.00 / 0.00 | - | - | - | - | 0.03801 | 229 | 26819 | 0.02858 | [0.00516, 0.0194] |
| TTV2 | 532 | 532 | 0 | 93 | 24.8 | 0.01368 | [0.0123, 0.0148] | 0.01076 | 2.40 | 0.42 | 2.1 | 9.3 | 0.98 / 0.02 / 0.00 | 0.01733 | [0.0164, 0.019] | 429 | 153 | 0.01792 | 169 | 9131 | 0.0166 | [0.00409, 0.0148] |
| TTV3 | 323 | 323 | 0 | 95 | 20.1 | 0.01888 | [0.0176, 0.0206] | 0.01441 | 2.68 | 0.37 | 1.7 | 7.5 | 1.00 / 0.00 / 0.00 | 0.02373 | - | 33 | 156 | 0.06378 | 380 | 89124 | 0.05 | undetermined |
| TTV4 | 458 | 458 | 0 | 62 | 7.9 | 0.01112 | [0.00973, 0.0121] | 0.008532 | 1.82 | 0.55 | 1.1 | 5.0 | 0.98 / 0.02 / 0.00 | - | - | - | - | 0.03438 | 229 | 40309 | 0.05 | undetermined |
| TTV5 | 676 | 676 | 0 | 69 | 16.9 | 0.01143 | [0.00996, 0.0128] | 0.009319 | 3.46 | 0.29 | 2.6 | 11.9 | 0.98 / 0.02 / 0.00 | - | - | - | - | 0.06578 | 295 | 58612 | 0.05 | [0.00266, 0.0128] |
| TTV6 | 820 | 820 | 0 | 101 | 23.3 | 0.01978 | [0.0164, 0.0229] | 0.015 | 3.62 | 0.28 | 2.1 | 9.5 | 0.90 / 0.10 / 0.00 | - | - | - | - | 0.02967 | 157 | 10973 | 0.03327 | [0.00504, 0.0229] |
| TTV7 | 300 | 300 | 0 | 66 | 18.7 | 0.009384 | [0.00733, 0.0114] | 0.007247 | 3.08 | 0.32 | 2.7 | 12.2 | 0.98 / 0.02 / 0.00 | - | - | - | - | 0.02416 | 143 | 35966 | 0.05 | undetermined |
| TTV8 | 1,332 | 1,332 | 0 | 71 | 15.3 | 0.01451 | [0.0131, 0.0161] | 0.01042 | 3.34 | 0.30 | 2.2 | 9.8 | 0.99 / 0.01 / 0.00 | - | - | - | - | 0.04223 | 276 | 38544 | 0.04557 | [0.00372, 0.0161] |
| TTV9 | 547 | 547 | 0 | 57 | 16.8 | 0.008578 | [0.00649, 0.011] | 0.006769 | 3.37 | 0.30 | 2.1 | 9.5 | 0.80 / 0.20 / 0.00 | - | - | - | - | 0.04815 | 305 | 36561 | 0.04377 | [0.00108, 0.011] |

E* per site (pooled bracket pixels of the primary tiles): the fitted S_c and the band shares.

| site | pixels | S_c fit | 68 % | inside own 68 % | E* median at 1.27 | bands at 1.27 | E* median at fit | bands at fit | addendum 8 S_c (pooled, option (iii) set) | addendum 8 normalised-ratio medians |
|---|---|---|---|---|---|---|---|---|---|---|
| GM | 28,664 | 0.379 | [0.365, 0.392] | yes | 1.7 | 1.00 / 0.00 / 0.00 | 5.6 | 0.79 / 0.21 / 0.01 | 0.514 | GM1 0.52, GM2 0.51, GM3 0.38, GM4 0.47, GM5 0.40 |
| AP | 16,438 | 0.292 | [0.283, 0.303] | yes | 2.5 | 0.97 / 0.03 / 0.00 | 10.8 | 0.46 / 0.45 / 0.09 | 0.299 | AP1 0.31, AP2 0.23, AP3 0.19, AP4 0.29, AP5 0.26 |
| TTV | 7,503 | 0.280 | [0.268, 0.292] | yes | 1.9 | 0.97 / 0.03 / 0.00 | 8.6 | 0.57 / 0.36 / 0.07 | - | - |

TTV anchor (decision 14, section 2.8):

| quantity (TTV site median) | value (m²/yr) | / 2.5 × 10⁻³ (bedrock convention) | / 5 × 10⁻³ (soil-flux convention) |
|---|---|---|---|
| D0_N_A | 0.00464 | 1.86 | 0.93 |
| D0_N_S | 0.002917 | 1.17 | 0.58 |
| D_R | 0.01479 | 5.91 | 2.96 |
| D0_N_B | 0.007523 | 3.01 | 1.50 |
| D0_R_A | 0.002701 | 1.08 | 0.54 |

### D8. The 64 picked heads (section 3): attributes, fits, A·S², field values, D∞

| pick | x (EPSG:6339) | y | tile | in polygon | dist to traced channel (m) | A_cell | A_3 | A_3 offset (m) | A_10 | S_up (10 m) | L (m) | S_up 5 m | S_up 20 m | S_down (10 m) | L (m) | z (m) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 540170.40 | 4189949.08 | TTV3 | yes | 10.7 | 3 | 2864 | 2.8 | 3703 | 0.292 | 10.8 | 0.301 | 0.274 | 0.340 | 11.2 | 105.2 |
| 2 | 540091.27 | 4190025.55 | TTV3 | yes | 5.3 | 38 | 6324 | 3.0 | 13943 | 0.185 | 10.2 | 0.125 | 0.227 | 0.258 | 10.1 | 112.0 |
| 3 | 539962.35 | 4190164.24 | TTV1 | yes | 14.2 | 1 | 171 | 3.0 | 13354 | 0.339 | 10.9 | 0.350 | 0.331 | 0.272 | 10.7 | 138.7 |
| 4 | 539902.78 | 4190238.04 | TTV1 | yes | 8.0 | 3 | 227 | 2.0 | 408 | 0.322 | 10.2 | 0.263 | 0.275 | 0.332 | 10.5 | 151.9 |
| 5 | 539949.01 | 4190361.62 | TTV1 | yes | 3.8 | 1 | 424 | 2.8 | 1789 | 0.584 | 10.1 | 0.724 | 0.544 | 0.375 | 10.1 | 111.6 |
| 6 | 539764.09 | 4190261.15 | TTV1 | yes | 9.6 | 19739 | 19748 | 2.8 | 20039 | 0.248 | 10.5 | 0.256 | 0.252 | 0.253 | 10.5 | 157.8 |
| 7 | 539678.73 | 4190343.84 | TTV1 | yes | 1.7 | 4 | 2687 | 3.0 | 4612 | 0.298 | 10.9 | 0.310 | 0.309 | 0.254 | 10.2 | 163.1 |
| 8 | 539544.48 | 4190298.49 | TTV1 | yes | 4.0 | 9 | 619 | 3.0 | 7356 | 0.262 | 11.3 | 0.260 | 0.246 | 0.282 | 10.2 | 192.0 |
| 9 | 539494.54 | 4190381.67 | TTV1 | yes | 7.8 | 4 | 136 | 2.8 | 1049 | 0.645 | 10.2 | 0.789 | 0.486 | 0.383 | 10.2 | 175.2 |
| 10 | 539396.11 | 4190353.65 | TTV1 | yes | 10.3 | 2 | 141 | 2.0 | 14960 | 0.488 | 10.1 | 0.594 | 0.515 | 0.390 | 10.1 | 204.3 |
| 11 | 539317.07 | 4190426.22 | TTV1 | yes | 10.2 | 1 | 509 | 2.8 | 628 | 0.335 | 10.5 | 0.285 | 0.354 | 0.237 | 10.7 | 234.0 |
| 12 | 539311.32 | 4190452.80 | TTV1 | yes | 0.6 | 2 | 134 | 3.0 | 1286 | 0.578 | 10.7 | 0.550 | 0.573 | 0.332 | 11.1 | 233.6 |
| 13 | 539285.45 | 4190569.20 | TTV1 | yes | 6.2 | 6 | 4221 | 3.0 | 7455 | 0.495 | 10.7 | 0.549 | 0.386 | 0.209 | 10.1 | 252.5 |
| 14 | 539352.28 | 4190629.56 | TTV1 | yes | 6.1 | 7 | 168 | 3.0 | 2037 | 0.279 | 10.5 | 0.332 | 0.273 | 0.321 | 10.9 | 253.4 |
| 15 | 539523.28 | 4190602.97 | TTV1 | yes | 1.5 | 4 | 20272 | 2.0 | 20309 | 0.271 | 10.2 | 0.260 | 0.272 | 0.125 | 10.1 | 188.9 |
| 16 | 539523.28 | 4190665.49 | TTV1 | yes | 5.0 | 2 | 1259 | 2.2 | 1445 | 0.606 | 10.2 | 0.483 | 0.549 | 0.513 | 10.5 | 205.2 |
| 17 | 539442.81 | 4190825.71 | TTV1 | yes | 6.4 | 2 | 2589 | 3.0 | 33875 | 0.174 | 10.7 | 0.216 | 0.186 | 0.253 | 10.4 | 250.0 |
| 18 | 539469.40 | 4190935.65 | TTV1 | yes | 0.1 | 5 | 7 | 2.2 | 13494 | 0.624 | 4.8* | 0.624 | 0.624* | 0.488 | 10.2 | 250.9 |
| 19 | 539546.28 | 4191027.62 | none | yes | 6.3 | 1 | 5152 | 2.2 | 19474 | 0.531 | 10.1 | 0.503 | 0.522 | 0.359 | 11.1 | 239.8 |
| 20 | 539610.22 | 4191068.57 | none | yes | 1.8 | 18 | 1952 | 3.0 | 2230 | 0.395 | 10.7 | 0.357 | 0.411 | 0.266 | 10.2 | 247.3 |
| 21 | 539709.38 | 4191007.50 | none | yes | 11.7 | 17 | 63 | 3.0 | 11323 | 0.424 | 10.1 | 0.329 | 0.400 | 0.433 | 10.9 | 233.8 |
| 22 | 539805.66 | 4190972.29 | TTV1 | yes | 9.7 | 4 | 67 | 2.8 | 415 | 0.605 | 10.0 | 0.583 | 0.608 | 0.504 | 10.8 | 218.7 |
| 23 | 539777.64 | 4190896.85 | TTV1 | yes | 0.2 | 4 | 2330 | 2.8 | 2704 | 0.654 | 10.0 | 0.662 | 0.641 | 0.713 | 10.7 | 175.2 |
| 24 | 539886.14 | 4190819.25 | TTV1 | yes | 3.6 | 2 | 606 | 2.8 | 659 | 0.618 | 10.9 | 0.534 | 0.563 | 0.657 | 10.8 | 141.5 |
| 25 | 539824.34 | 4190684.17 | TTV1 | yes | 8.0 | 4 | 169 | 2.8 | 1019 | 0.393 | 11.1 | 0.356 | 0.415 | 0.442 | 10.9 | 139.7 |
| 26 | 539930.68 | 4190814.22 | TTV1 | yes | 9.5 | 4 | 109 | 2.2 | 227 | 0.581 | 10.4 | 0.552 | 0.628 | 0.496 | 11.2 | 148.7 |
| 27 | 539940.74 | 4190966.54 | TTV1 | yes | 2.7 | 75 | 82 | 2.2 | 1476 | 0.355 | 10.2 | 0.422 | 0.325 | 0.551 | 11.1 | 200.1 |
| 28 | 540018.34 | 4190964.39 | TTV2 | yes | 6.6 | 1 | 641 | 2.2 | 668 | 0.268 | 10.2 | 0.271 | 0.297 | 0.422 | 10.0 | 192.5 |
| 29 | 540125.40 | 4190937.09 | TTV2 | yes | 12.2 | 5 | 1180 | 2.8 | 1654 | 0.557 | 10.8 | 0.513 | 0.559 | 0.538 | 10.1 | 190.2 |
| 30 | 540155.58 | 4190860.92 | TTV2 | yes | 1.6 | 20 | 97 | 2.2 | 102 | 0.478 | 11.3 | 0.474 | 0.469 | 0.577 | 11.3 | 172.4 |
| 31 | 540197.25 | 4190804.88 | TTV2 | yes | 5.0 | 66 | 91 | 2.8 | 315 | 0.703 | 10.0 | 0.712 | 0.657 | 0.727 | 10.0 | 154.4 |
| 32 | 540235.33 | 4190707.16 | TTV2 | yes | 11.9 | 1 | 2044 | 2.2 | 8133 | 0.300 | 10.9 | 0.346 | 0.336 | 0.387 | 10.9 | 134.4 |
| 33 | 540211.62 | 4190665.49 | TTV2 | yes | 14.8 | 28 | 172 | 2.0 | 426 | 0.481 | 10.9 | 0.463 | 0.500 | 0.382 | 10.5 | 128.0 |
| 34 | 540223.84 | 4190526.09 | TTV2 | yes | 4.8 | 168 | 1786 | 2.8 | 2935 | 0.374 | 10.9 | 0.398 | 0.365 | 0.508 | 10.1 | 118.4 |
| 35 | 540327.30 | 4190473.64 | TTV3 | yes | 6.2 | 9 | 571 | 2.8 | 7820 | 0.299 | 10.5 | 0.253 | 0.405 | 0.322 | 10.7 | 117.3 |
| 36 | 540524.90 | 4190334.97 | TTV3 | yes | 1.2 | 9 | 7257 | 2.8 | 8129 | 0.402 | 10.9 | 0.451 | 0.329 | 0.357 | 10.9 | 89.7 |
| 37 | 540625.49 | 4190378.08 | TTV3 | yes | 13.2 | 8 | 38 | 2.2 | 1298 | 0.497 | 10.4 | 0.498 | 0.492 | 0.300 | 10.0 | 84.6 |
| 38 | 540539.99 | 4190501.66 | TTV2 | yes | 6.9 | 17 | 701 | 2.8 | 1074 | 0.334 | 10.2 | 0.335 | 0.329 | 0.333 | 11.2 | 114.9 |
| 39 | 540504.78 | 4190569.20 | TTV2 | yes | 1.6 | 4 | 639 | 2.2 | 2188 | 0.436 | 10.4 | 0.442 | 0.451 | 0.650 | 10.4 | 121.0 |
| 40 | 540475.32 | 4190644.65 | TTV2 | yes | 1.8 | 40 | 187 | 2.2 | 7330 | 0.642 | 10.8 | 0.547 | 0.599 | 0.419 | 10.1 | 122.9 |
| 41 | 540440.11 | 4190700.69 | TTV2 | yes | 0.6 | 1 | 207 | 2.2 | 268 | 0.631 | 11.1 | 0.605 | 0.671 | 0.598 | 10.7 | 132.5 |
| 42 | 540444.42 | 4190812.06 | TTV2 | yes | 1.6 | 4 | 79 | 3.0 | 8786 | 0.604 | 10.1 | 0.579 | 0.641 | 0.389 | 10.2 | 125.6 |
| 43 | 540329.46 | 4190932.77 | TTV2 | yes | 10.9 | 12 | 257 | 2.8 | 376 | 0.395 | 11.3 | 0.393 | 0.358 | 0.421 | 11.3 | 168.9 |
| 44 | 540231.74 | 4191016.12 | TTV2 | yes | 5.1 | 2 | 3723 | 3.0 | 4134 | 0.534 | 11.1 | 0.539 | 0.495 | 0.405 | 10.7 | 165.8 |
| 45 | 540177.85 | 4191101.63 | TTV2 | yes | 7.2 | 1 | 4465 | 3.0 | 4479 | 0.332 | 10.1 | 0.292 | 0.284 | 0.321 | 10.8 | 179.4 |
| 46 | 540826.67 | 4190705.72 | TTV2 | yes | 18.7 | 1190 | 1193 | 3.0 | 1314 | 0.347 | 10.1 | 0.342 | 0.334 | 0.363 | 10.5 | 83.8 |
| 47 | 540920.08 | 4190744.52 | TTV2 | yes | 4.1 | 4 | 23 | 2.0 | 112 | 0.511 | 10.9 | 0.476 | 0.505 | 0.552 | 10.1 | 66.6 |
| 48 | 540850.39 | 4190862.36 | TTV2 | yes | 7.6 | 3 | 6181 | 2.8 | 6416 | 0.419 | 10.5 | 0.394 | 0.489 | 0.433 | 10.2 | 88.3 |
| 49 | 540624.77 | 4190909.06 | TTV2 | yes | 55.0 | 1 | 726 | 2.8 | 2761 | 0.268 | 10.1 | 0.405 | 0.308 | 0.297 | 10.9 | 105.0 |
| 50 | 540565.13 | 4191039.83 | TTV2 | yes | 8.2 | 1 | 7242 | 3.0 | 7509 | 0.312 | 10.8 | 0.310 | 0.320 | 0.344 | 10.4 | 123.7 |
| 51 | 540488.25 | 4191038.40 | TTV2 | yes | 7.3 | 3 | 1387 | 2.8 | 1396 | 0.282 | 10.1 | 0.237 | 0.349 | 0.259 | 10.5 | 111.3 |
| 52 | 540205.88 | 4191215.15 | TTV2 | yes | 1.4 | 7265 | 7272 | 2.2 | 7280 | 0.323 | 10.1 | 0.326 | 0.374 | 0.236 | 10.1 | 158.7 |
| 53 | 540157.02 | 4191251.08 | TTV2 | yes | 3.2 | 10 | 346 | 2.2 | 7899 | 0.284 | 10.2 | 0.282 | 0.267 | 0.256 | 10.8 | 172.6 |
| 54 | 540098.10 | 4191291.31 | TTV2 | yes | 4.7 | 2 | 4527 | 3.0 | 4545 | 0.220 | 10.8 | 0.247 | 0.214 | 0.302 | 10.2 | 191.0 |
| 55 | 540116.78 | 4191356.70 | TTV2 | yes | 2.2 | 1 | 482 | 2.2 | 5375 | 0.331 | 10.2 | 0.318 | 0.374 | 0.279 | 10.1 | 189.9 |
| 56 | 540149.83 | 4191445.80 | TTV2 | yes | 9.1 | 2 | 345 | 2.0 | 976 | 0.261 | 10.1 | 0.197 | 0.523 | 0.186 | 10.1 | 196.4 |
| 57 | 540185.76 | 4191490.34 | TTV2 | yes | 9.2 | 47 | 87 | 2.2 | 3481 | 0.532 | 10.9 | 0.497 | 0.543 | 0.581 | 10.7 | 208.4 |
| 58 | 540278.45 | 4191418.49 | TTV2 | yes | 2.9 | 9 | 295 | 3.0 | 8846 | 0.354 | 10.9 | 0.355 | 0.358 | 0.378 | 11.3 | 176.6 |
| 59 | 540385.50 | 4191465.20 | TTV2 | yes | 12.7 | 33 | 795 | 3.0 | 823 | 0.395 | 11.2 | 0.378 | 0.402 | 0.372 | 10.7 | 187.9 |
| 60 | 540524.90 | 4191312.15 | TTV2 | yes | 13.7 | 757 | 781 | 2.2 | 8225 | 0.400 | 10.2 | 0.410 | 0.421 | 0.475 | 10.8 | 184.2 |
| 61 | 540545.73 | 4191228.80 | TTV2 | yes | 4.2 | 2 | 2093 | 2.8 | 2332 | 0.420 | 10.7 | 0.426 | 0.375 | 0.496 | 10.7 | 167.9 |
| 62 | 540650.64 | 4191234.55 | TTV2 | yes | 1.5 | 1 | 616 | 3.0 | 4030 | 0.524 | 11.1 | 0.480 | 0.440 | 0.307 | 10.7 | 177.7 |
| 63 | 540762.01 | 4191082.94 | TTV2 | yes | 2.2 | 9 | 70 | 2.8 | 428 | 0.425 | 10.7 | 0.399 | 0.405 | 0.508 | 11.3 | 126.2 |
| 64 | 540793.62 | 4190987.38 | TTV2 | yes | 12.8 | 34 | 226 | 3.0 | 4107 | 0.373 | 10.7 | 0.376 | 0.432 | 0.408 | 10.4 | 111.1 |

(* the donor chain ended before the reach length; the length available was used.)

| fit | n | a (m²) | a 95 % | b | b 95 % | R² |
|---|---|---|---|---|---|---|
| i_table1 | 63 | 1978 | [1459, 2774] | 1.647 | [1.26, 2.01] | 0.598 |
| ii_picks_A3_Sup | 64 | 64 | [22, 190] | 2.396 | [1.35, 3.43] | 0.219 |
| iii_picks_A3_Sdown | 64 | 65 | [20, 218] | 2.211 | [1.04, 3.28] | 0.195 |
| iv_picks_A3_Sup_without_49 | 63 | 62 | [20, 189] | 2.439 | [1.37, 3.52] | 0.222 |

R4: a = 1978 m² (-0.0 % of 1,978), b = 1.647 (-0.003 of 1.65): **pass**.

| fit | b = 1.65 inside 95 % | median A_3 / (1978 S^−1.65) | IQR | median in 0.5-2 | R5 |
|---|---|---|---|---|---|
| ii | yes | 0.047 | [0.022, 0.192] | no | **not consistent** |
| iii_report_only | yes | 0.045 | [0.018, 0.215] | no | **not consistent** |
| iv_report_only | yes | 0.049 | [0.022, 0.194] | no | **not consistent** |

Distributions (n; 5th, 25th, 50th, 75th, 95th percentiles):

| quantity | n | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|
| A_cell | 64 | 1 | 2 | 4 | 17 | 668.6 |
| A_3 | 64 | 63.6 | 168.8 | 611 | 2152 | 7255 |
| A_10 | 64 | 275.1 | 1042 | 2848 | 7587 | 1.88e+04 |
| table1_area | 63 | 2010 | 4800 | 7400 | 1.035e+04 | 1.486e+04 |
| A_3_without_49 | 63 | 63.4 | 168.5 | 606 | 2212 | 7256 |
| S_up | 64 | 0.2503 | 0.3088 | 0.3952 | 0.5311 | 0.6405 |
| S_down | 64 | 0.2364 | 0.2995 | 0.3764 | 0.4904 | 0.6419 |
| table1_slope | 63 | 0.272 | 0.41 | 0.47 | 0.58 | 0.78 |
| A_3_offset_m | 64 | 2 | 2.236 | 2.828 | 3 | 3 |
| dist_to_traced_channel_m | 64 | 0.682 | 2.608 | 6.195 | 9.622 | 14.15 |

| A·S² (m²) | n | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|
| picks_Sup | 64 | 10.51 | 30.8 | 77.53 | 246 | 1159 |
| picks_Sdown | 64 | 11.83 | 25.07 | 61.81 | 319.2 | 914 |
| table1 | 63 | 804.7 | 1052 | 1500 | 2184 | 2977 |

Share of picks inside M&D (1992)'s 500-4,000 m²: with S_down 0.12, with S_up 0.16.

Field values for section 5: A_F = 611 m² (without pick 49 606; Table 1 median 7400; median A_10 2848); b̂ = 2.396 → α = 2.4; C_F = 0: 611, 2: 61.81, 2.4: 41.38 (without 49: 0: 606, 2: 59.73, 2.4: 42.4).

D∞ pilot on TTV2: 35.4 µs per cell → full grid 3.55 core-hours (cap 10.0): within the cap.
D∞ on the full grid (141,299,405 cells, 1.58 core-hours): A_3 median 387 m² (IQR 120-1638); A_10 median 2212; median D∞/D8 ratio of A_3 0.72; A_F from D∞ 387 m².

| pick | A_3 D8 | A_3 D∞ | A_10 D8 | A_10 D∞ | D∞ area at the pick cell |
|---|---|---|---|---|---|
| 1 | 2864 | 1579 | 3703 | 1648 | 51.8 |
| 2 | 6324 | 7131 | 13943 | 8510 | 9.0 |
| 3 | 171 | 132 | 13354 | 6622 | 16.3 |
| 4 | 227 | 242 | 408 | 377 | 2.0 |
| 5 | 424 | 368 | 1789 | 795 | 48.8 |
| 6 | 19748 | 9265 | 20039 | 9265 | 8157.8 |
| 7 | 2687 | 769 | 4612 | 1008 | 4.0 |
| 8 | 619 | 643 | 7356 | 4380 | 191.9 |
| 9 | 136 | 141 | 1049 | 335 | 24.7 |
| 10 | 141 | 51 | 14960 | 7858 | 2.8 |
| 11 | 509 | 2289 | 628 | 2550 | 1.0 |
| 12 | 134 | 246 | 1286 | 1297 | 50.5 |
| 13 | 4221 | 1937 | 7455 | 5351 | 16.6 |
| 14 | 168 | 55 | 2037 | 1143 | 21.4 |
| 15 | 20272 | 14713 | 20309 | 19878 | 3.7 |
| 16 | 1259 | 1318 | 1445 | 1506 | 13.8 |
| 17 | 2589 | 3247 | 33875 | 26615 | 4.0 |
| 18 | 7 | 6 | 13494 | 12003 | 4.1 |
| 19 | 5152 | 5361 | 19474 | 19200 | 12.5 |
| 20 | 1952 | 1024 | 2230 | 2113 | 777.1 |
| 21 | 63 | 372 | 11323 | 9254 | 15.5 |
| 22 | 67 | 204 | 415 | 204 | 4.0 |
| 23 | 2330 | 1396 | 2704 | 2312 | 29.7 |
| 24 | 606 | 274 | 659 | 373 | 89.9 |
| 25 | 169 | 156 | 1019 | 197 | 5.8 |
| 26 | 109 | 48 | 227 | 74 | 24.1 |
| 27 | 82 | 44 | 1476 | 239 | 13.5 |
| 28 | 641 | 237 | 668 | 607 | 42.6 |
| 29 | 1180 | 946 | 1654 | 1438 | 52.3 |
| 30 | 97 | 49 | 102 | 51 | 43.7 |
| 31 | 91 | 86 | 315 | 86 | 38.9 |
| 32 | 2044 | 1370 | 8133 | 3848 | 42.3 |
| 33 | 172 | 99 | 426 | 316 | 41.2 |
| 34 | 1786 | 1816 | 2935 | 2400 | 270.8 |
| 35 | 571 | 414 | 7820 | 5751 | 155.3 |
| 36 | 7257 | 5234 | 8129 | 7260 | 253.1 |
| 37 | 38 | 151 | 1298 | 878 | 150.7 |
| 38 | 701 | 402 | 1074 | 639 | 35.6 |
| 39 | 639 | 61 | 2188 | 2336 | 19.5 |
| 40 | 187 | 90 | 7330 | 6919 | 34.4 |
| 41 | 207 | 121 | 268 | 170 | 12.6 |
| 42 | 79 | 50 | 8786 | 5770 | 16.8 |
| 43 | 257 | 83 | 376 | 325 | 7.7 |
| 44 | 3723 | 3299 | 4134 | 4061 | 10.5 |
| 45 | 4465 | 3058 | 4479 | 4456 | 1.0 |
| 46 | 1193 | 431 | 1314 | 888 | 151.0 |
| 47 | 23 | 109 | 112 | 109 | 14.1 |
| 48 | 6181 | 2600 | 6416 | 5991 | 2.8 |
| 49 | 726 | 555 | 2761 | 1441 | 18.5 |
| 50 | 7242 | 3747 | 7509 | 7416 | 324.0 |
| 51 | 1387 | 1577 | 1396 | 1577 | 40.1 |
| 52 | 7272 | 3223 | 7280 | 3223 | 1303.3 |
| 53 | 346 | 166 | 7899 | 2356 | 70.8 |
| 54 | 4527 | 3258 | 4545 | 5689 | 156.1 |
| 55 | 482 | 229 | 5375 | 858 | 11.2 |
| 56 | 345 | 264 | 976 | 447 | 11.9 |
| 57 | 87 | 76 | 3481 | 3405 | 17.8 |
| 58 | 295 | 209 | 8846 | 3958 | 29.5 |
| 59 | 795 | 117 | 823 | 984 | 48.7 |
| 60 | 781 | 4733 | 8225 | 6979 | 193.0 |
| 61 | 2093 | 770 | 2332 | 967 | 11.2 |
| 62 | 616 | 859 | 4030 | 3061 | 7.2 |
| 63 | 70 | 106 | 428 | 310 | 10.9 |
| 64 | 226 | 595 | 4107 | 3355 | 82.0 |


### D9. Gate 4.1.3 timing on TTV2 and the Part B estimate

One PF evaluation 4.0 s; one SF evaluation 3.3 s (head statistic 0.5 s); SJ pilot 20 evaluations in 122 s (6.09 s each) → a full SJ run of 2400 evaluations ≈ 4.06 h; 33 runs ≈ 134 core-hours expected, limits 8.1 h each = 268 core-hours of limits (one submission, under 300). Pilot run `20261120T164102Z_a19_sj_pilot_TTV2_a2`.

PF, PF3 and SF on TTV2 (gate-3 evaluations; the section 5.2 values of this tile):

| threshold | A_crit (m²) | channel cells | heads | PF mse | PF K/K₀ | PF D | where | PF3 mse | PF3 D | where |
|---|---|---|---|---|---|---|---|---|---|---|
| A_F | 611 | 24,750 | 589 | 129.63 | 1.021 | 0.008006 | interior | 131.28 | 0.009318 | interior |
| A_F_table1 | 7400 | 6,440 | 48 | 66.67 | 1.030 | 0.0149 | interior | 68.10 | 0.01571 | interior |
| A_F_without_49 | 606 | 24,900 | 597 | 130.31 | 1.021 | 0.008011 | interior | 131.97 | 0.009344 | interior |
| A_F_A10_sensitivity | 2848 | 9,576 | 96 | 90.04 | 0.997 | 0.0105 | interior | 90.06 | 0.01042 | interior |

| SF | α | C_F | channel cells | heads | mse | K/K₀ | D | where |
|---|---|---|---|---|---|---|---|---|
| alpha_0:C_F | 0 | 611 | 24,750 | 589 | 129.63 | 1.021 | 0.008006 | interior |
| alpha_0:C_F_without_49 | 0 | 606 | 24,900 | 597 | 130.31 | 1.021 | 0.008011 | interior |
| alpha_2:C_F | 2 | 61.81 | 42,301 | 1,187 | 129.23 | 1.038 | 0.009585 | interior |
| alpha_2:C_F_without_49 | 2 | 59.73 | 43,979 | 1,249 | 130.52 | 1.041 | 0.009489 | interior |
| alpha_2.4:C_F | 2.4 | 41.38 | 48,498 | 1,410 | 132.74 | 1.048 | 0.009724 | interior |
| alpha_2.4:C_F_without_49 | 2.4 | 42.4 | 47,332 | 1,372 | 131.19 | 1.044 | 0.009813 | interior |

α = 0 equals PF: {'mse_PF': 129.6337165356141, 'mse_SF0': 129.6337165356141, 'rel_diff': 0.0, 'n_channel_equal': True, 'pass': True}. TTV2 addendum 18: P4 mse 59.79, AD mse 209.66.

### D11. Figures

| file | size | dpi | note |
|---|---|---|---|
| AP10_hilltop_qc.png | 1.76 MB | 200 |  |
| AP11_hilltop_qc.png | 1.69 MB | 200 |  |
| AP13_hilltop_qc.png | 1.78 MB | 200 |  |
| AP14_hilltop_qc.png | 1.77 MB | 200 |  |
| AP18_hilltop_qc.png | 1.69 MB | 200 |  |
| AP19_hilltop_qc.png | 1.78 MB | 200 |  |
| AP1_3dep_hilltop_qc.png | 1.73 MB | 200 |  |
| AP1_AP1_3dep_hilltop_pair.png | 2.72 MB | 200 |  |
| AP1_hilltop_qc.png | 1.35 MB | 200 |  |
| AP20_hilltop_qc.png | 1.79 MB | 200 |  |
| AP23_hilltop_qc.png | 1.81 MB | 200 |  |
| AP24_hilltop_qc.png | 1.76 MB | 200 |  |
| AP25_hilltop_qc.png | 1.77 MB | 200 |  |
| AP27_hilltop_qc.png | 1.77 MB | 200 |  |
| AP28_hilltop_qc.png | 1.75 MB | 200 |  |
| AP29_hilltop_qc.png | 1.70 MB | 200 |  |
| AP2_3dep_hilltop_qc.png | 1.75 MB | 200 |  |
| AP2_AP2_3dep_hilltop_pair.png | 2.82 MB | 200 |  |
| AP2_hilltop_qc.png | 1.43 MB | 200 |  |
| AP3_3dep_hilltop_qc.png | 1.78 MB | 200 |  |
| AP3_AP3_3dep_hilltop_pair.png | 2.84 MB | 200 |  |
| AP3_hilltop_qc.png | 1.46 MB | 200 |  |
| AP4_3dep_hilltop_qc.png | 1.75 MB | 200 |  |
| AP4_AP4_3dep_hilltop_pair.png | 2.76 MB | 200 |  |
| AP4_hilltop_qc.png | 1.39 MB | 200 |  |
| AP5_3dep_hilltop_qc.png | 1.78 MB | 200 |  |
| AP5_AP5_3dep_hilltop_pair.png | 2.76 MB | 200 |  |
| AP5_hilltop_qc.png | 1.36 MB | 200 |  |
| AP6_hilltop_qc.png | 1.83 MB | 200 |  |
| AP7_hilltop_qc.png | 1.76 MB | 200 |  |
| AP8_hilltop_qc.png | 1.78 MB | 200 |  |
| AP9_hilltop_qc.png | 1.76 MB | 200 |  |
| AP_Cht_vs_L_perron.png | 0.15 MB | 200 |  |
| AP_D0N_vs_L_gallery.png | 0.62 MB | 200 |  |
| AP_D_bracket_intervals.png | 0.07 MB | 200 |  |
| AP_hilltop_gallery_p1.png | 4.54 MB | 144 |  |
| AP_hilltop_gallery_p2.png | 1.59 MB | 200 |  |
| AP_rollover_vs_1e3_compare.png | 2.95 MB | 200 |  |
| AP_sigmaC_scaling_break.png | 0.12 MB | 200 |  |
| GM11_hilltop_qc.png | 1.66 MB | 200 |  |
| GM12_hilltop_qc.png | 1.78 MB | 200 |  |
| GM13_hilltop_qc.png | 1.69 MB | 200 |  |
| GM14_hilltop_qc.png | 1.82 MB | 200 |  |
| GM15_hilltop_qc.png | 1.78 MB | 200 |  |
| GM16_hilltop_qc.png | 1.69 MB | 200 |  |
| GM17_hilltop_qc.png | 1.71 MB | 200 |  |
| GM18_hilltop_qc.png | 1.68 MB | 200 |  |
| GM19_hilltop_qc.png | 1.68 MB | 200 |  |
| GM1_GM1_fema_hilltop_pair.png | 2.81 MB | 200 |  |
| GM1_fema_hilltop_qc.png | 1.77 MB | 200 |  |
| GM1_hilltop_qc.png | 1.42 MB | 200 |  |
| GM20_hilltop_qc.png | 1.54 MB | 200 |  |
| GM21_hilltop_qc.png | 1.79 MB | 200 |  |
| GM22_hilltop_qc.png | 1.66 MB | 200 |  |
| GM24_hilltop_qc.png | 1.66 MB | 200 |  |
| GM25_hilltop_qc.png | 1.76 MB | 200 |  |
| GM26_hilltop_qc.png | 1.70 MB | 200 |  |
| GM27_hilltop_qc.png | 1.86 MB | 200 |  |
| GM28_hilltop_qc.png | 1.78 MB | 200 |  |
| GM29_hilltop_qc.png | 1.85 MB | 200 |  |
| GM2_GM2_fema_hilltop_pair.png | 2.72 MB | 200 |  |
| GM2_fema_hilltop_qc.png | 1.69 MB | 200 |  |
| GM2_hilltop_qc.png | 1.43 MB | 200 |  |
| GM30_hilltop_qc.png | 1.81 MB | 200 |  |
| GM3_GM3_fema_hilltop_pair.png | 2.63 MB | 200 |  |
| GM3_fema_hilltop_qc.png | 1.64 MB | 200 |  |
| GM3_hilltop_qc.png | 1.45 MB | 200 |  |
| GM4_GM4_fema_hilltop_pair.png | 2.79 MB | 200 |  |
| GM4_fema_hilltop_qc.png | 1.76 MB | 200 |  |
| GM4_hilltop_qc.png | 1.43 MB | 200 |  |
| GM5_GM5_fema_hilltop_pair.png | 2.75 MB | 200 |  |
| GM5_fema_hilltop_qc.png | 1.73 MB | 200 |  |
| GM5_hilltop_qc.png | 1.45 MB | 200 |  |
| GM6_GM6_fema_hilltop_pair.png | 2.69 MB | 200 |  |
| GM6_fema_hilltop_qc.png | 1.66 MB | 200 |  |
| GM6_hilltop_qc.png | 1.45 MB | 200 |  |
| GM7_GM7_fema_hilltop_pair.png | 2.62 MB | 200 |  |
| GM7_fema_hilltop_qc.png | 1.61 MB | 200 |  |
| GM7_hilltop_qc.png | 1.42 MB | 200 |  |
| GM8_hilltop_qc.png | 1.77 MB | 200 |  |
| GM9_hilltop_qc.png | 1.74 MB | 200 |  |
| GM_Cht_vs_L_perron.png | 0.16 MB | 200 |  |
| GM_D0N_vs_L_gallery.png | 0.66 MB | 200 |  |
| GM_D_bracket_intervals.png | 0.09 MB | 200 |  |
| GM_hilltop_gallery_p1.png | 4.37 MB | 144 |  |
| GM_hilltop_gallery_p2.png | 2.00 MB | 200 |  |
| GM_rollover_vs_1e3_compare.png | 2.85 MB | 200 |  |
| GM_sigmaC_scaling_break.png | 0.14 MB | 200 |  |
| TTV11_hilltop_qc.png | 1.92 MB | 200 |  |
| TTV15_hilltop_qc.png | 1.88 MB | 200 |  |
| TTV1_hilltop_qc.png | 1.89 MB | 200 |  |
| TTV2_hilltop_qc.png | 1.79 MB | 200 |  |
| TTV3_hilltop_qc.png | 1.75 MB | 200 |  |
| TTV4_hilltop_qc.png | 1.99 MB | 200 |  |
| TTV5_hilltop_qc.png | 1.73 MB | 200 |  |
| TTV5_roads.png | 2.56 MB | 200 |  |
| TTV6_hilltop_qc.png | 1.84 MB | 200 |  |
| TTV7_hilltop_qc.png | 1.90 MB | 200 |  |
| TTV8_hilltop_qc.png | 1.96 MB | 200 |  |
| TTV9_hilltop_qc.png | 1.97 MB | 200 |  |
| TTV_D0N_vs_L_gallery.png | 0.34 MB | 200 |  |
| TTV_D_bracket_intervals.png | 0.07 MB | 200 |  |
| TTV_hilltop_gallery_p1.png | 4.04 MB | 170 |  |
| TTV_rollover_vs_1e3_compare.png | 3.25 MB | 200 |  |
| TTV_sigmaC_scaling_break.png | 0.12 MB | 200 |  |
| ttv_heads_area_slope.png | 0.17 MB | 200 |  |

### D12. Slurm accounting of every addendum 19 job

| job name | tasks | core-hours used | core-hours of limits | states | job ids |
|---|---|---|---|---|---|
| a19_tests | 1 | 0.2 | 2.0 | COMPLETED 1 | 53094599 |
| a19_heads | 1 | 0.0 | 1.5 | COMPLETED 1 | 53095206 |
| a19_hill_test | 1 | 0.0 | 1.0 | COMPLETED 1 | 53095376 |
| a19_dinf | 1 | 4.8 | 21.0 | COMPLETED 1 | 53095652 |
| a19_hilltops | 73 | 0.6 | 36.5 | COMPLETED 73 | 53095730 |
| a19_gate3 | 1 | 0.0 | 1.0 | COMPLETED 1 | 53096475 |
| a19_partA | 1 | 0.1 | 2.5 | FAILED 1 | 53096852 |
| a19_qc2 | 1 | 0.0 | 1.0 | COMPLETED 1 | 53097281 |

Core-hours used (elapsed × CPUs over all tasks): **5.7**; sum of CPU-weighted limits: 66.5; tasks: 80; peak MaxRSS 25.7 GB.

