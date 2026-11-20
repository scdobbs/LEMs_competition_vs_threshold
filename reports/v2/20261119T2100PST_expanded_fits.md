# [2026-11-19 21:00 PST] Addendum 18: model fits of the expanded tile set (Figs. 4-6, equal-parameter and held-out tests, trunk, continuity, TTV channel heads)

Directive: `reports/directives/2026-11-19T0615PST_v2_addendum_18_expanded_fits.md` (outline fixed by its section 8). Branch `v2.3-tiles`
(from e85221b, the addendum 17 Part B report; Part A committed as 862ee83 and pushed so that the hilltop QC figures could be looked at
while Part B ran). **No package code was changed.** Scripts: `reports/v2/scripts/a18_common.py` (the groups, decisions and rules as
constants), `a18_prep.py` (Part A: the `measured` blocks, the GM 22-tile summaries, the map labels, the trunk list, K₀ at m_t, the
same-pixel diagnostic, the field head areas, the fit specifications), `a18_hilltop_qc.py` (the QC and partition figures), `a18_fits.py`
(every fit: the exact reductions of addendum 12, `lemcvt.analyses.sensitivity`, with three additions outside the package: tiles without
D₀, model overrides, and the trunk cells as fixed anchors), `a18_estimate.py`, `a18_gate_check.py`, `a18_submit_partB.py`,
`a18_summary.py` (the rules and the Figs. 4-6 equivalents), `make_a18_tables.py` (the appendices); all ruff-clean. Environments `lemcvt`,
`ci312`. Every number is tied to the run directory `$GROUP_HOME/lemcvt_results/20261119T143945Z_a18_expanded_fits/` (`prep/`,
`hilltop_qc/`, `gate/`, `estimate/`, `summary/`), to a lemcvt run id (`*_a18_part_<tile>_<group>`, `*_a18_ad_<tile>_<group>` under
`$GROUP_HOME/lemcvt_results/`, Appendix C16) or to a Slurm job id (`$SCRATCH/lemcvt_jobs/a18/`, Appendix C17). Appendices (section 11)
are machine-generated. Figures: `reports/v2/figures/a18/` (the Figs. 4-6 equivalents) and `reports/v2/figures/a18/hilltop_qc/` (QC,
pairs, galleries, partitions; `README.md` there is the index). **This is a stop for review.**

## Summary

| item | result |
| --- | --- |
| 1 state, decisions | `v2.3-tiles` = origin = e85221b at the start; tests on `serc` 274 passed; every path present. Decisions applied as written: GM primary **22** (QTcl, QTg, QTm, QTpc are the Paso Robles Formation on the map), clean 19; 63 `measured` blocks written; 10 tiles without D₀; the trunk rule as written gives **9** tiles (the five expected plus their NCALM GM2, GM4 and PAMAP AP3, AP4 twins). Eleven deviations stated, none a stop (section 1) |
| 2 Part A | **Same-pixel diagnostic:** GM reads **"selection"** (surface ratio S 0.97, selection ratio P 0.80: the FEMA surface carries the NCALM curvature on the same pixels; the 22 % deficit is which pixels each survey's rollover network selects), AP reads **"mixed"** (S 0.85, P 0.69). 91 hilltop QC figures (73 tiles, 12 pairs, 5 gallery pages) plus 54 partition figures. **Regression gate passed**: GM1 (m 0.31) and AP2 (PAMAP) reproduce v2.0.0 exactly (A_crit, K, D, r identical; MSE within 4 × 10⁻¹⁵). Cost gate: 837 of 900 core-hours expected, **648 used** |
| 3 fits, F1-F4 | **F1 (fit) holds at every site**: GM 22/22 (median R_MSE 2.25), AP 21/21 (2.34), TTV 10/11 (2.59; TTV11 fails). **F2 (K) holds everywhere**: partitioned K/K₀ medians 1.05, 0.99, 0.96 against AD 2.4, 10.1, 10.1. **F3 (D) holds at GM (21/22) and AP (12/13 at D₀,A, 13/13 at D₀,B), fails at TTV (4/9)**. **F4 ("AD's D off by an order of magnitude") holds at GM (2.2 decades) and TTV (1.05-1.10), fails at AP (0.82, 0.89)** |
| 4 equal counts, held-out | **GM: every rule holds** (E1 20-21/22, E2 21/22, H1 20/22, H2 19/22). **AP: with D free the partitioned model wins (P3 21/21, H1 21/21); with D held at D₀ it does not** (E1 7/13 at D₀,A but 12/13 at D₀,B; E2-P2 7-8/13; H2 9-10/13). **TTV: P3 9/11 and H1 10/11 hold; every D₀-fixed rule fails (2/9)**. Lopsided splits: GM 3, AP 7, TTV 2 tiles; the balanced results agree with the headline |
| 5 clean, off-mesa | Clean subsets change one verdict at AP (H2 at D₀,B: 10/12 holds) and one at TTV (F4 fails, 0.99 / 0.95); nothing at GM. Off-mesa GM (6): F1 6/6, F2 6/6, F3 4-5/6, partitioned D/D₀,A 3.6 against 1.7 on the mesa |
| 6 trunk | Material on **AP3_3dep** (K/K₀ 1.16 → 1.62, R_MSE 4.4 → 2.4) and AP3 (R_MSE 3.6 → 2.6); not material on the other seven (changes ≤ 4 %); no F1 flip, no solver failure |
| 7 continuity | FEMA/NCALM at m 0.29: K 1.02, D 0.99, A_crit 0.81 (GM2 0.43, GM6 2.9), MSE_P4 0.96, R_MSE 0.90. 3DEP/PAMAP at 0.47: K 1.09, D 1.16, A_crit 1.4 (AP3 3.1), MSE_P4 0.97, R_MSE 1.13. **GM1-GM5 at 0.31 and AP1-AP5 at 0.47 match v2.0.0 exactly** (10 of 10) |
| 8 TTV | Site-m choice **not immaterial** (F1 at m_t 10/11; partitioned D/D₀,A −2 %, but AD K/K₀ 10.1 → 2.9). Channel heads: **the fitted A_crit lies above the field channel heads at the 1 km tile scale** (9 of 11 "above", 2 "consistent", at the site m; 10 of 11 at m_t); the 77 digitised heads sit on the hillslope at grid scale (median 132 m² against Table 1's 7,400 m²) |
| 9-10 | What holds and what is weak; decisions on the AP D wording, the TTV D at its bound, the trunk fit of AP3_3dep, the channel-head wording (sections 9-10) |
| cost | 215 tasks in 36 submissions, **648 core-hours used** (expected 837; budget 900), 1,737 of CPU-weighted limits, the largest submission 280; at most 196 CPUs at once; peak memory 3.8 GB; +0.2 GB on group storage (`$GROUP_HOME` 321 GB of 1 TB) |

## 1. State check, deviations, and the decisions of section 1 as applied

**At the start (2026-11-19 06:25 PST, login node sh03-ln05).**
- `v2.3-tiles` = `origin/v2.3-tiles` = e85221b (the Part B report, pushed); no job was running; `/etc/agents/AGENTS.md` and its Slurm and
  storage topic files read (and re-read at every session change); the documents of section 0.2 read in order.
- The test suite was rerun on `serc` at e85221b (job 52954918): ruff clean, mypy clean (111 files), **274 passed**; run for the state
  check, not because package code changed (none did).
- Every path of section 0.3 exists (the Part B run directory with `configs/`, `screens/`, `measure/`; the eight `measure_a17b_t<k>` runs;
  the m-profile runs; the v2.0.0 build; the addendum 15 and 16 directories; the copies of record; the 76 new tile configs with empty
  `measured` blocks).

**Where the addendum and the repo or the data differ, and the choices made.** Nothing required a stop. Stated here, each handled as
written:

1. **The cost gate (4.2) was timed on pilot runs, not on complete fits.** "Time one P4 fit, one AD fit and the reductions on one tile of
   each site" would have taken 7-9 wall-clock hours per site before any array could be sized. Instead one pilot per site and stage (GM11,
   AP6, TTV2: the thinned grid of 150 A_crit values with the Gram terms of the whole tile and of both halves, so every variant's cost is
   in it; the AD grid of 25 values) timed one evaluation of each kind (section 2.4), and the expected hours are the upper evaluation counts
   of a production run (3,900 partitioned, 290 AD) times those timings. Every task's limit was twice its estimate; the production runs
   took 3.3-7.1 h against limits of 11.75-13.25 h (the 7.1 h is TTV11 on a slow node, 8.9 s per evaluation against the pilot's 5.4 s).
2. **The P4 grid design.** "Thin the A_crit grid to 2000 values and refine, as in Phase 6" is followed for the thinning and the staircase
   refinement; the fill is the addendum 12 design (every unique area in the **1 %** interval of every variant, capped at 1,500 values),
   not the Phase 6 fill of the whole 5 % interval, which cost 3,800-7,000 extra evaluations per AP tile in Phase 6 and would have put the
   addendum over budget. Addendum 12 found the two designs' minima within 0.3 %; here the gate reproduced the Phase 6 optima exactly with
   this design (section 2.3). The 1 % intervals are complete (the cap bound on no primary tile; on the five PAMAP tiles, whose 0.975 m
   pixels give 36,000-44,000 unique areas, it did, as in addendum 12); the 5 % interval is read from the thinned grid (resolution 0.35 %
   in A_crit).
3. **The trunk list (decision 8) by the rule as written has nine tiles, not five.** "Every primary and continuity tile whose largest A
   exceeds 5 km²" includes the NCALM twins GM2 (8.9 km²) and GM4 (6.2) and the PAMAP twins AP3 (11.0) and AP4 (17.0) beside the expected
   GM2_fema (10.3), GM4_fema (6.2), GM5_fema (5.5), AP3_3dep (11.1) and AP4_3dep (17.1). All nine were run (the four twins cost 41
   core-hours); GM5 on NCALM (4.7 km²) and the off-mesa tiles (at most 3.3) stay out. Appendix C1 has every tile's largest A.
4. **The `measured` blocks of the ten tiles without D₀ (decision 4).** The loader (`lemcvt.config.load_tile_config`) refuses a config whose
   `model.D0` is not a number inside the bounds, so on those tiles `model.D0` keeps a numeric placeholder (E / the site selection reference
   curvature of `measured_v1`, labelled `D0_placeholder_note`) while `measured.D0`, `C_ht`, `D0_B` are `null` with the reason. Every fit
   of this addendum reads `measured.D0` through `prep/fit_specs.json`; the placeholder enters nothing.
5. **GM1-GM5 at the site m 0.29 (continuity).** Their configs carry the v2.0.0 blocks (m = 0.31) and are not rewritten; the continuity fits
   take k_s at 0.29 and D₀ from the Part B NCALM record (`per_tile.csv`), and the fits at 0.31 (section 7) take the configs.
6. **Channel heads (5.7.1).** The rule's "largest A within 3 m" is applied; because the digitised heads sit far from the 1 m channel
   network (section 8.2), the largest A within 10 m and 20 m is reported beside it as a labelled sensitivity that enters no rule.
7. **The regression gate (4.1)** ran the P4 fit on the design of point 2 with the stored Phase 6 rows of the v2.0.0 run joined to the row set
   (whole-tile Gram terms at the same m, K₀, D₀), as addendum 12 did, and compared the optimum over the union, and over the new rows alone,
   with the release; the AD fit is the same profile and Brent refinement as Phase 6.
8. **"Under 5 MB"** is read as 5,000,000 bytes: the first pair figures of three survey pairs were 5.17-5.18 MB (under 5 MiB) and were
   regenerated at a lower dpi (both jobs in Appendix C17).
9. **The same-pixel mapping (3.1)** reprojects the NCALM pixel centres (EPSG:32610) onto the FEMA lattice (EPSG:6339) with PROJ through
   rasterio; PROJ's default WGS84-NAD83(2011) step leaves the centres on the FEMA cell centres (offsets 0.00 m), consistent with the
   control tiles having been cut at the NCALM corners (addendum 17 Part A). The PAMAP-3DEP mapping (EPSG:26917 to 6346, 0.975 m against
   1 m pixels) has median offsets of 0.4 m and maxima of 0.7 m.
10. **The pilot run ids** carry the production step name (`a18_part`, `a18_ad`) with `_pilot_` in the label; the estimate and the summary
    find them by name, and the summary excludes them. A first version of the summary stored the A_crit intervals under the wrong keys
    (`i01`); it was fixed before any production run was read, and the dry runs on the gate runs are the record (jobs 52957430, 52972736).
11. **Four of the eleven TTV partitioned fits sit on the upper D bound** (D = 0.05 m²/yr: TTV3, TTV4, TTV5, TTV7; also GM20 and the
    off-mesa GM19). The bounds are the v2 metre bounds of every config (section 8.1); nothing was widened, as the standing rule requires,
    and the affected numbers are marked. The report draft lived in the repository (`reports/v2/_draft_a18.md`, untracked) while the
    arrays ran; the final job's `sacct` sidecar was written to a wrong path (a shell-expansion slip, cosmetic: the accounting of
    Appendix C17 comes from `sacct` directly).

### 1.1 The decisions of section 1 as applied
- **Decision 1, the bedrock screen at formation level.** The map (Rosenberg, `GEOLOGY.shp`) describes QTp as "Paso Robles Formation,
  undifferentiated" (252 polygons; 20 of them queried, "QTp?"), **QTcl as "Paso Robles Formation, clay"** (20 polygons, `GEN_NAME` "Paso
  Robles Formation"), QTg as "Paso Robles Formation, gravel" (6), QTm as "Paso Robles Formation, marl" (7) and QTpc as "Paso Robles
  Formation, clay and gravel" (1); all five share class Sedimentary, lithology Clastic sedimentary, era Cenozoic, period
  Quaternary-Tertiary, epoch Pliocene-Pleistocene?, strength C (Appendix C2, every field). **GM8, GM9 and GM21 return**: the GM primary
  set is 22 (GM1_fema-GM7_fema, GM8, GM9, GM11, GM12, GM13, GM15, GM16, GM17, GM20, GM21, GM22, GM24, GM26, GM27, GM29), the clean subset
  19 (minus GM12, GM26, GM29), off-mesa GM14, GM18, GM19, GM25, GM28, GM30. On the 22 tiles S(m) has its argmin at **0.29** (1 % 0.26-0.32,
  5 % 0.22-0.36; clean 19: 0.29, 0.26-0.31), so the site m stays 0.29. The GM site summaries on the 22 tiles, beside Part B's 19: site-pooled
  k_s 2.036 [1.925, 2.142] (n_eff 40.8) against 2.051, **K₀ 7.22 × 10⁻⁵** against 7.17 × 10⁻⁵ (tile K₀ 4.5-9.1 × 10⁻⁵, median 7.22 × 10⁻⁵);
  D₀,A median **0.0153** (0.0073-0.035) against 0.0155, D₀,B median 0.0150 against 0.0143; C_ht(L_f) median 0.0096. The three returned
  tiles: GM8 k_s 1.74, D₀,A 0.0152; GM9 1.78, 0.0204; GM21 2.37, 0.0119 (Appendix C3).
- **Decision 2, D₀ at both ends everywhere.** Every D statement below is made at D₀,A and D₀,B. The window test as it stands in Part B:
  GM primary **14 of 22** pass (all 22 non-trivial), clean 12 of 19; AP 9 of 21 (8 trivial); TTV 9 of 11 (5 trivial). The NCALM-twin D₀
  for GM1_fema-GM7_fema (GM1-GM7 D₀,A 0.0090, 0.0098, 0.0101, 0.0099, 0.0076, 0.0109, 0.0220; FEMA/twin 1.14, 1.28, 1.53, 1.20, 1.15, 1.49,
  1.61) is in Appendix C3; with it the FEMA controls' partitioned D/D₀ would be 1.14-1.61 times the values of section 3 (D/D₀,A 1.17-2.73
  becomes 1.7-3.1), still inside the NCALM range of section 7; it enters no rule.
- **Decision 4, no D₀ below 500 hilltop pixels.** AP2_3dep (13), AP9 (30), AP10 (145), AP11 (0), AP14 (118), AP18 (25), AP27 (83), AP28 (119),
  TTV3 (122), TTV4 (86): `measured.D0: null`, in every K, MSE and A_crit statement, in no D statement; P1, P2 and A2 not run on them. The
  D denominators are GM 22, AP 13, TTV 9 (clean: GM 19, AP 12, TTV 8).
- **Decision 5, TTV at m = 0.33 plus m_t** (TTV1 0.05, TTV2 0.45, TTV3 0.40, TTV4 0.42, TTV5 0.41, TTV6 0.05, TTV7 0.05, TTV8 0.06, TTV9 0.35,
  TTV11 0.05, TTV15 0.05; the six at 0.05 flagged). K₀ at m_t by χ-z above 10³ m² (the Part B estimator; the recomputation at the site m
  equals the config k_s on 11 of 11 tiles): at m_t = 0.05 k_s is 0.26-0.39 and K₀ 2.0-2.9 × 10⁻⁴ (other units of A); at m_t = 0.35-0.45 k_s is
  6.3-19.0 and K₀ 4.1-12 × 10⁻⁶ (Appendix C14).
- **Decisions 6-7:** every group at a site fitted at its site m (GM 0.29, AP 0.47, TTV 0.33) with E = 1.47 × 10⁻⁴, 5.5 × 10⁻⁵, 7.7 × 10⁻⁵ m/yr;
  the only exceptions are the gate and GM2-GM5 at 0.31 (section 7).
- **Decision 8, trunk cells** = A ≥ 1 km² on the tile's area grid, 213-1,644 cells per tile (0.02-0.16 % of the valid cells); held at the
  observed elevation as anchors of the channel integration (the partitioned model: extra boundary cells of the chi BFS, so every tributary
  starts from the trunk's observed elevation) and as Dirichlet rows (AD), left out of every MSE; point 3 above for the list.
- **Decision 9, the `measured` blocks:** 63 written (every measured tile without a v2.0.0 block: the 22 + 6 GM, 21 AP, 11 TTV, GM6, GM7,
  AP1_3dep), each with the site m, k_s (A_c = 10³ m², n_basins, n_eff, the basin bootstrap), C_ht at L_f and at L_w, D₀,A, D₀,B, the hilltop
  pixel count, the window-test record and the run ids (`prep/configs.json`; every config loads and resolves K₀ and D₀ to the written values).
  The GM8, GM9, GM21 blocks are written like the others. Committed in 862ee83 (Appendix C1).

## 2. Part A: the same-pixel diagnostic, the hilltop QC figures, the regression gate, the cost estimate

### 2.1 The same-pixel curvature diagnostic (3.1)
`a18_prep.py --part samepixel` (job 52954916; Appendix C4). For every pair the option (iii) set at L_f = 11 m was rebuilt from the tile's
rollover A_c with the Part B function (`option_iii_mask`) and the curvature grid at L_f with the Part B estimator (`total_curvature`,
w = 5). **(a) and (d) reproduce the Part B C_ht(L_f) exactly on 12 of 12 pairs** (relative difference 0.0; the section ran). Pixels mapped
by map coordinates and the nearest cell; dropped within L_f/2 + 1 m of the edge or on nodata: GM 2-25 of 920-3,583 per tile, AP 0-10 of
533-4,711, none on nodata.

| site | pairs | S = median of (b)/(a), (d)/(c) | P = median of (c)/(a), (d)/(b) | total (d)/(a), median | Part B total | reading (rule of 3.1.5) |
| --- | --- | --- | --- | --- | --- | --- |
| GM, NCALM → FEMA | GM1-GM7 | **0.974** (0.93-1.01) | **0.797** (0.66-0.90) | 0.78 | 0.78 | **selection** |
| AP, PAMAP → 3DEP | AP1-AP5 | **0.849** (0.68-1.07) | **0.687** (0.49-1.56) | 0.65 | 0.65 | **mixed** |
| AP without AP2 | AP1, AP3-AP5 | 0.849 | 0.631 | 0.61 | | mixed |

- **At GM the surface is not the cause.** On the same pixels the FEMA surface carries 93-101 % of the NCALM curvature (S 0.97); the 22 %
  deficit of Part B comes from which pixels each survey's rollover network selects (P 0.80): the FEMA sets are 2-4 times larger (3,310-8,554
  against 920-3,583 pixels) and reach onto broader, flatter crests, where the curvature at 11 m is lower on either surface. On GM3, GM6 and
  GM7 the pixel selection alone gives 0.66-0.70.
- **At AP both act.** The 3DEP surface carries 68-94 % of the PAMAP curvature on the PAMAP pixels (S 0.85, the striping of the PAMAP DEM
  adding curvature at 11 m, as Part B read it) and the 3DEP sets select lower-curvature pixels on top (P 0.69; AP2 is the exception with 13
  pixels, P 1.3-1.6). Neither criterion of the "surface" or "selection" reading is met alone, so the reading is "mixed", with or without AP2.
- For the writing of the D result: the GM FEMA D₀ (0.0153) exceeds the NCALM D₀ (0.0099) because the FEMA hilltop set is a different,
  broader set of crest pixels, not because the FEMA surface is smoother at 11 m. Section 7 shows that the fitted D itself does not depend
  on the survey (FEMA/NCALM 0.99), so the FEMA D/D₀ is lower than the NCALM D/D₀ by the D₀ ratio (1.7 against 2.2).

### 2.2 The hilltop QC figures (3.2)
`a18_hilltop_qc.py` (jobs 52954919, 52956806, 53016507); 145 PNG under `reports/v2/figures/a18/hilltop_qc/` with `README.md` (the legend and
the list): 73 per-tile figures `<tile>_hilltop_qc.png` (22 + 6 GM, 7 NCALM, 21 AP, 5 PAMAP, AP1_3dep, 11 TTV), 12 pair figures
`<old>_<new>_hilltop_pair.png`, 5 gallery pages (`GM_hilltop_gallery_p1-2.png`, `AP_…_p1-2`, `TTV_…_p1`), and 54 partition figures
`<tile>_partition.png` (item 5). Classes as defined in item 1 (Appendix C5 has every tile's counts): (i) the rollover network, 8 % of a GM or
AP tile at the median and 17 % at TTV (up to 35-37 % where A_c is 11 m²); (ii) the near-channel exclusion; (iii) the counted hilltop set;
(iv) the other divide cells, a quarter of every tile, almost all of them "not between two basins of the rollover network" (the cells without
a D8 donor on the flanks; 114-1,700 per tile are the steep ones, a few hundred the edge margin); (v) the rest. The hatch marks the filled cells.

What the figures show, for the visual check:
- The counted set (iii) sits on the crest lines between rollover-network basins on every GM, NCALM, PAMAP and TTV tile and on the AP tiles
  with a rollover A_c of 178-355 m²; on the eight AP tiles with A_c = 11 m² the "channel" class fills the flanks and the counted set is a
  handful of pixels (AP9: 30, in two spots), which is the collapse Part B reported; the figures make its cause plain (the rollover of the
  3DEP slope-area curve at 11 m² turns nearly every cell into a channel cell).
- The pair figures show the FEMA sets wider and denser than the NCALM sets on the same crests (third row; on GM3 630 pixels in both sets,
  868 NCALM-only, 5,653 FEMA-only) and the FEMA histogram shifted to lower curvature with the NCALM tail above 0.04 m⁻¹ missing.
- **The partition figures** (item 5): at the fitted A_crit (section 3; 650-16,500 m² at GM and AP, 9,000-89,000 at TTV) the 200 m hilltop
  zoom holds no channel cell on nearly every tile; the model's hillslope domain covers the crests, the counted hilltop set and the whole
  rollover network of the measurement (A_c 11-562 m²). The cells that set D₀ are inside the model's hillslope on every tile, 1-2 orders of
  magnitude in area below the model's channel head.
- No tile's classes look wrong against the definition. Two things to be aware of: class (iv) is dense on every flank because the D8
  divide test marks every cell without a donor; and the 200 m zoom box of a collapsed AP set is centred on a few pixels.

### 2.3 The regression gate (4.1)
Jobs 52954921 (the four fits, 3.5-5.3 h for the partitioned ones) and 52956673 (`a18_gate_check.py`); `gate/gate.json`, Appendix C6.
**Pass.** With the v2.0.0 configs (metres, the v2 bounds), GM1 at m = 0.31 and AP2 (PAMAP) at m = 0.47:

| tile | P4: A_crit, K, D | MSE_P4 (v2.0.0 / here) | AD: K, D, r | MSE_AD (v2.0.0 / here) |
| --- | --- | --- | --- | --- |
| GM1 | 1241 m², 4.8653 × 10⁻⁵, 0.019477: identical | 24.690389367768148 / 24.690389367768063 (3.4 × 10⁻¹⁵) | identical | 48.96275516405692 / the same (0) |
| AP2 | 3121.06 m², 4.5968 × 10⁻⁶, 0.016796: identical | 22.46082883708459 / 22.46082883708452 (3.8 × 10⁻¹⁵) | identical | 76.49290337325623 / the same (0) |

The P4 optimum is the same over the union of this run's rows with the stored Phase 6 rows and over this run's rows alone (the row at the
v2.0.0 A_crit is in both; 2,101 common rows on GM1 agree to 0.0); the AD Brent refinement lands on the same r. Part B could start.

### 2.4 The cost estimate (4.2) and the cost
Pilot timings (`*_a18_part_pilot_*`, `*_a18_ad_pilot_*`, job 52954920): one partitioned reduction with the Gram terms on the whole tile and
on both halves takes **5.78 s (GM11), 6.05 s (AP6), 5.41 s (TTV2)**; one AD reduction 26.5, 29.1, 28.9 s. With 3,900 partitioned and 290 AD
evaluations per run (upper counts: 2,000 + ≤ 1,000 + ≤ 1,500; 200 + 6 Brent refinements) the expected core-hours in the priority order of
section 7 (`estimate/estimate.json`, printed before any array was submitted):

| section (priority order) | specs | expected core-hours | limits (2 × estimate) | used |
| --- | --- | --- | --- | --- |
| 5.1-5.3 primary groups (P4, AD, the variants, held-out) | 54 | 461.7 | 939 | 352.5 |
| 5.6-5.7 TTV per-tile m | 11 | 90.2 | 182 | 74.5 |
| 5.4 trunk sensitivity | 9 | 77.6 | 158 | 60.3 |
| 5.5 continuity (NCALM, PAMAP, AP1_3dep; GM2-GM5 at 0.31) | 18 | 136.9 | 280 | 106.0 |
| 5.1 off-mesa | 6 | 50.4 | 104 | 39.1 |
| Part B | 98 | **816.7** | 1,662 | **632.4** |

Part A had used 2.8 core-hours at that point and the gate was expected to take 17.3 more (it took 13.8): **expected total 836.9 of 900
core-hours**, so Part B was submitted without a staged plan: 22 arrays (one per stage, group and site), every submission at most 280
core-hours of CPU-weighted limits, every array at its full width (192 concurrent single-CPU tasks plus the 4 gate tasks, under the
300-CPU line), all dependent on the gate check (`afterok`), so that a failed gate would have left them unstarted. **Used in all: 648
core-hours** (Appendix C17): the production runs took 2,100-2,900 partitioned evaluations (3.3-7.1 h; the PAMAP tiles 3,800-4,300, 5 h)
and 260-270 AD evaluations (2.0-2.2 h), under the upper counts of the estimate. Part B ran from 13:15 to 20:28 PST; no task failed or
timed out.

## 3. Primary fits per site (5.1): the tables, the figures, and rules F1-F4

Every tile of every group: P4 [K, D, A_crit] (the thinned grid of 2,000 unique areas, the staircase refinement, the 1 % fill) and AD [K, D]
(200-point profile over r = K/D, Brent), at the site m, D8 routing, the registered area grid. Per tile in Appendix C8 (every value), the
variants in C9, the figures `reports/v2/figures/a18/<site>_fig4_mse.png` (MSE_AD against MSE_P4), `<site>_fig5_K.png` (K/K₀, both models)
and `<site>_fig6_D.png` (D/D₀ at both ends, both models), with the primary, Tier 2, off-mesa and continuity tiles marked.

| site, primary group | N | A_crit, median (range) | MSE_P4 (m²) | MSE_AD | R_MSE median (range) | K/K₀ P4 \| AD, median (range) | D/D₀,A P4 \| AD, median | D/D₀,B P4 \| AD, median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GM | 22 | 3,360 (650-15,200) | 20-167 | 33-230 | **2.25** (1.33-4.17) | **1.05** (0.84-1.20) \| 2.37 (1.15-7.0) | **1.74** (1.00-2.76) \| 0.0067 | **1.76** (0.78-3.10) \| 0.0069 |
| AP | 21 (D₀ on 13) | 6,010 (2,630-16,500) | 19-48 | 35-169 | **2.34** (1.39-5.20) | **0.99** (0.88-1.16) \| 10.1 (6.2-12.5) | **3.48** (2.84-5.35) \| 0.152 | **2.47** (2.20-4.47) \| 0.129 |
| TTV | 11 (D₀ on 9) | 36,600 (9,100-89,100) | 60-242 | 134-843 | **2.59** (0.68-6.11) | **0.96** (0.69-1.42) \| 10.1 (2.0-20.9) | **16.1** (3.7-20.0) \| 0.56 | **15.1** (3.5-19.0) \| 0.56 |

**The rules (section 6; holds = at least 80 % of N, rounded up; the count, the denominator and the failing tiles for every rule in
Appendix C11):**

| rule | GM (N 22, D₀ 22) | AP (N 21, D₀ 13) | TTV (N 11, D₀ 9) |
| --- | --- | --- | --- |
| F1, MSE_P4 < MSE_AD | **22/22 holds** | **21/21 holds** | **10/11 holds** (TTV11: 242 against 164) |
| F2, \|log K/K₀\| lower for P4 | **21/22 holds** (GM11: 1.20 against 1.15) | **21/21 holds** | **11/11 holds** |
| F3, \|log D/D₀\| lower for P4, at D₀,A | **21/22 holds** (GM27: 2.41 against 0.65) | **12/13 holds** (AP25: 5.3 against 0.22) | **4/9 fails** (needs 8; TTV1, TTV6, TTV7, TTV9, TTV11) |
| F3 at D₀,B | **21/22 holds** (GM27) | **13/13 holds** | **4/9 fails** |
| F4, AD median \|log₁₀ D/D₀\| ≥ 1 at both ends | **holds**: 2.18 (A), 2.17 (B) | **fails**: 0.82 (A), 0.89 (B) | **holds**: 1.10 (A), 1.05 (B) |

What the numbers say, per site:
- **GM.** The partitioned model fits better on every tile (R_MSE 1.3-4.2), recovers K within ±20 % of K₀ on every tile (median 1.05), and
  puts D at 1.0-2.8 × D₀,A (median 1.74, lower than the 2.2 of NCALM in v2 because the FEMA D₀ is higher, section 2.1); AD needs 1.2-7 × K₀
  and a D 2-3 decades below D₀ (median 0.0067; at the lower bound of the box, 4 × 10⁻¹⁰, on GM11 and GM21, where the AD fit is pure
  advection). The two single-tile exceptions: GM11 (AD K/K₀ 1.15 against the partitioned 1.20, both close to K₀) and GM27 (AD D/D₀ 0.65,
  its one near-D₀ AD fit, with K/K₀ 7.0 and MSE twice P4's). GM20's partitioned D sits on the upper bound (0.05, D/D₀ 2.74) with K/K₀ 0.84.
- **AP.** The partitioned model fits better on every tile (R_MSE 1.4-5.2) and recovers K₀ within ±16 % on every tile (median 0.99); AD needs
  6-12.5 × K₀ (median 10.1), as in v2. The partitioned D is 2.8-5.4 × D₀,A (median 3.5) and 2.2-4.5 × D₀,B (median 2.5), AD's 0.02-0.25 × D₀,A
  (median 0.15) and 0.02-0.19 × D₀,B (0.13): closer to D₀ on 12 of 13 and 13 of 13 tiles, but the AD deficit is 0.6-1.6 decades, median 0.82
  (A) and 0.89 (B), **so F4 fails at AP**: "off by more than an order of magnitude" holds on 5 (A) and 6 (B) of the 13 tiles, not at the site
  median. The partitioned excess is 0.4-0.5 decades. AP25 (Tier 2, pads on the ridges) is the F3 exception at D₀,A (5.3 against 0.22).
- **TTV.** The partitioned model fits better on 10 of 11 tiles (R_MSE 1.7-6.1) and recovers K₀ within ±42 % (median 0.96; 0.69-0.76 on TTV6,
  TTV7, TTV8, TTV9, TTV11); AD needs 2-21 × K₀. **The partitioned D is 3.7-20 × D₀,A (median 16)**, and on four tiles (TTV3, TTV4, TTV5,
  TTV7) it sits on the upper bound of the box, 0.05 m²/yr, so the TTV D is bounded, not fitted, there (deviation 11). AD's D is 0.02-33 ×
  D₀ (median 0.56; at the upper bound on TTV8 and TTV15, where AD's D exceeds D₀ 12-33 times). F3 fails: on five of nine tiles AD's D,
  low as it is, is closer to D₀ in log terms than the partitioned 15-20 ×. F4 holds on the primary set (1.10 / 1.05) and fails on the clean
  subset (0.99 / 0.95, section 5): the AD "order of magnitude" is marginal at TTV. TTV11 (Tier 2, fire roads) is the one F1 failure, with
  the worst partitioned MSE (242) and K/K₀ 0.69.
- **The A_crit intervals** (Appendix C8): the 1 % interval is a few per cent wide on most tiles (GM1_fema 1,208-1,254; AP29 16,412-16,636;
  TTV3 89,070-89,144) and wide where the profile is flat (GM5_fema 1,179-2,197; GM26 4,285-9,521; AP11 3,379-8,940; TTV11 16,982-31,139);
  the 5 % interval spans a factor 2-20 on GM15, GM17 and GM29. P4 K and D sit inside the box on every tile but the six named above.

## 4. Equal-parameter and held-out tests (5.2, 5.3), with rules E1, E2, H1 and H2

**In-sample (5.2; Appendix C9).** P1 [A_crit] at (K₀, D₀), P2 [K, A_crit] at D₀, P3 [D, A_crit] at K₀, P4; A1 [D] at K₀, A2 [K] at D₀, AD;
the D₀-fixed variants at both ends. Site medians of the MSE ratios on the tiles with D₀:

| site | P3/P4 (max) | P2_A/P4 \| P2_B/P4 | P1_A/AD \| P1_B/AD | P2_A/AD \| P2_B/AD | A1/AD | A2_A/AD \| A2_B/AD |
| --- | --- | --- | --- | --- | --- | --- |
| GM | 1.03 (1.88, GM20) | 1.42 \| 1.34 | 0.70 \| 0.69 | 0.64 \| 0.62 | 2.2 | 2.65 \| 2.66 |
| AP | 1.02 (1.11) | 2.74 \| 2.11 | 1.65 \| 1.04 | 0.82 \| 0.68 | 31 | 1.84 \| 1.95 |
| TTV | 1.25 (2.85) | 4.94 \| 4.17 | 2.27 \| 2.27 | 2.0 \| 1.69 | 7.0 | 1.59 \| 1.61 |

| rule | GM (D₀ on 22) | AP (D₀ on 13) | TTV (D₀ on 9) |
| --- | --- | --- | --- |
| E1 (count 1): P1 beats A1 and A2, at D₀,A | **20/22 holds** (GM20, GM27) | **7/13 fails** (needs 11; AP4_3dep, AP13, AP19, AP20, AP24, AP25) | **2/9 fails** (TTV2, TTV5 only) |
| E1 at D₀,B | **21/22 holds** (GM20) | **12/13 holds** (AP25) | **2/9 fails** |
| E2 (count 2): P2 beats AD, at D₀,A | **21/22 holds** (GM20) | **7/13 fails** (AP4_3dep, AP19, AP20, AP24, AP25, AP29) | **2/9 fails** |
| E2: P2 beats AD, at D₀,B | **21/22 holds** (GM20) | **8/13 fails** (AP4_3dep, AP19, AP24, AP25, AP29) | **2/9 fails** |
| E2: P3 beats AD (N all) | **21/22 holds** (GM20) | **21/21 holds** | **9/11 holds** (TTV9, TTV11) |

- **GM:** the partitioned model wins at every count, with D fixed at either D₀ or free, on 20-21 of 22 tiles; the single exception is
  GM20 (its P4 D at the bound; with D at D₀ its MSE doubles to 78 and AD, 35, wins). This is the v2 result on 22 tiles.
- **AP:** the partitioned model wins at count 2 with D free (P3, 21/21, within 11 % of P4) and loses the count with D fixed at D₀,A on 6 of
  13 tiles (P2_A 1.1-2.9 × P4; AD 1.4-1.7 × P4 on the failing tiles, which are the ones where AD fits best: AP4_3dep, AP19, AP24, R_MSE 1.4).
  At D₀,B (the wider window, D₀ 1.4-2 × higher) P1 wins count 1 on 12 of 13 (E1_B holds) but P2 still loses count 2 on 5. The AP conclusion
  of addendum 12 stands on 13 tiles: **at equal counts the partitioned model beats AD at AP only when D is free, and then with D 2.5-3.5 ×
  D₀.** A1 (AD with K = K₀) fits 31 × worse than AD: AD cannot take the measured K₀ at all.
- **TTV:** only TTV2 and TTV5 win with D fixed; elsewhere the partitioned model with D = D₀ fits 4-5 × worse than P4 and 1.7-2 × worse than
  AD. The TTV partitioned advantage rests on D free, and on D sitting 15-20 × above D₀ (at the box bound on four tiles). P3 (K = K₀) is
  within 25 % of P4 at the median but 1.5-2.9 × on TTV3, TTV6, TTV7, TTV8, TTV9, TTV11, where the fitted K/K₀ is 0.7-0.75 or 1.4.

**Held-out (5.3; Appendix C10).** The greedy basin split (edge-outlet basins of the 10³ m² network, largest first to the lighter half). Cell
shares of the smaller half: GM 0.08-0.50 (lopsided, under 30 %: GM5_fema 0.22, GM22 0.14, GM26 0.08); AP 0.08-0.50 (lopsided 7: AP3_3dep,
AP6, AP8, AP13, AP18, AP23, AP25, 0.08-0.28); TTV 0.20-0.50 (lopsided TTV3 0.20, TTV9 0.23). Fit on one half, score on the other, both ways,
averaged; site medians: held-out P4 70 / AD 161 m² (GM), 48 / 123 (AP), 169 / 635 (TTV); held-out over in-sample 1.6-1.7 for P4 and
1.8-2.2 for AD.

| rule | GM all \| balanced | AP all \| balanced | TTV all \| balanced |
| --- | --- | --- | --- |
| H1: P4 held-out < AD | **20/22 holds** (GM15, GM17) \| 17/19 holds | **21/21 holds** \| 14/14 | **10/11 holds** (TTV11) \| 8/9 |
| H2: P2_A held-out < AD | **19/22 holds** (GM9, GM15, GM20) \| 16/19 holds | **9/13 fails** (needs 11; AP4_3dep, AP19, AP24, AP25) \| 4/7 fails | **5/9 fails** (TTV1, TTV6, TTV7, TTV11) \| 4/8 fails |
| H2: P2_B held-out < AD | **19/22 holds** \| 16/19 | **10/13 fails** (AP4_3dep, AP24, AP25) \| 5/7 fails | **5/9 fails** \| 4/8 fails |

Out of sample the partitioned joint fit beats AD on 51 of 54 primary tiles (the three failures: GM15, where the P4 held-out MSE is 515
against AD's 402 after a 167 in-sample; GM17, 121 against 112; TTV11). With D held at D₀ the held-out result follows the in-sample one: GM
yes, AP and TTV no. The balanced subsets (the splits whose smaller half holds at least 30 % of the cells) give the same verdicts
everywhere. The held-out A_crit of the two halves differs by up to a factor 5 (GM1_fema 1,213 against the halves' 814 and 1,665 in the
5 % interval), as in addendum 12.

## 5. Clean subset and off-mesa groups (6.7, 6.10)

**Clean subsets** (GM 19: minus GM12, GM26, GM29; AP 20: minus AP25; TTV 10: minus TTV11), every rule recomputed (Appendix C11). Outcomes that
differ from the primary group: **GM none** (every rule holds on 16-19 of 19); **AP one**: H2 at D₀,B becomes 10/12, which holds (10
needed) where 10/13 failed, because AP25 was one of its three failures; **TTV one**: F4 fails on the clean subset (AD median |log₁₀ D/D₀|
0.99 at A, 0.95 at B, against 1.10 / 1.05 with TTV11). Everything else keeps its verdict with the same or better fractions (TTV F1 10/10,
E2-P3 9/10; the D₀-fixed rules still 2/8).

**Off-mesa GM (6 tiles, report only; Appendix C8, C11):** F1 6/6 (R_MSE median 2.42, 1.31-3.69), F2 6/6 (K/K₀ P4 1.00, AD 4.3), F3 4/6 at
D₀,A and 5/6 at D₀,B, F4 1.17 / 1.17. The partitioned D on the Pancho Rico and Tertiary-clastic tiles is **3.6 × D₀,A** at the median
(2.4-4.6; 4.0-4.6 on GM18, GM19, GM30), twice the mesa's 1.74, with K/K₀ 0.91-1.21; A_crit 1,900-14,200 m². GM14 (landslide deposits) has
the worst partitioned fit of the group (111 m², R 1.31) and GM19's D sits on the bound. The off-mesa tiles behave like the mesa in K and in
the fit, and differ in D.

## 6. Trunk sensitivity (5.4, 6.8)
Nine tiles (deviation 3), each with P4 and AD rerun with the trunk cells (A ≥ 1 km²; 213-1,644 cells, 0.02-0.16 % of the tile) held at
their observed elevation and left out of the MSE (Appendix C12). **No solver or optimizer failure** (every reduction converged; nothing
dropped). A trunk fit is material if K/K₀, D/D₀,A or A_crit (P4) or R_MSE changes by more than 25 %, or F1 flips:

| tile (group) | trunk cells | A_crit | K/K₀ | D/D₀,A | R_MSE | MSE_P4, MSE_AD | 6.8 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **AP3_3dep** (primary) | 1,514 | 14,529 → 14,389 | 1.16 → **1.62** (+40 %) | 5.35 → 5.23 | 4.44 → **2.44** (−45 %) | 38.1 → 36.5; 169 → 89 | **material** |
| **AP3** (PAMAP) | 1,521 | 4,742 → 4,782 | 1.01 → 1.06 | 8.31 → 8.36 | 3.63 → **2.57** (−29 %) | 42.1 → 41.0; 153 → 106 | **material** |
| AP4_3dep, AP4 | 213-218 | unchanged | unchanged | unchanged | −2 % | | not material |
| GM2_fema, GM2, GM4_fema, GM4, GM5_fema | 581-1,644 | ≤ 0.2 % | ≤ 4 % | ≤ 4 % | ≤ 4 % | | not material |

The 11 km² trunk of AP3 (the Mid Bailey Run main stem, 1,500 cells crossing the tile) is where the far-sourced inflow matters: with the
trunk held at its observed elevation **AD's MSE halves** (its advection no longer has to carve the trunk from the tile's own area) and the
partitioned K rises to 1.6 × K₀ on 3DEP (the trunk cells were carrying the channel integration's anchor). F1 does not flip (P4 36-41 against
AD 89-106). On the 17 km² trunk of AP4 (only 213 cells inside the tile) and on the GM trunks (0.6-1.6 km of channel at 6-10 km²) nothing
moves: the GM trunks are short and the partitioned integration from the tile boundary already fixes them within a few metres. The
headline results of sections 3-4 are from the standard fits (the trunk fit is a sensitivity, as decision 8 says); on AP3_3dep the K/K₀ of
record is 1.16 and the trunk value 1.62 is reported beside it.

## 7. Continuity (5.5)
**NCALM GM1-GM7 against FEMA GM1_fema-GM7_fema, both at m = 0.29** (ratios FEMA / NCALM; Appendix C13):

| | K (P4) | D (P4) | A_crit | MSE_P4 | MSE_AD | R_MSE | AD K | AD D |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| median | **1.017** | **0.987** | 0.81 | 0.96 | 0.88 | 0.90 | 1.18 | 1.35 |
| largest single-tile difference | 3.4 % (GM6) | GM6 1.61, GM2 0.85 | GM6 2.94, GM2 0.43, GM4 0.60 | GM6 1.13, GM3 1.12 | GM7 0.78 | GM7 0.81 | | |

The fitted K and D do not depend on the survey (within 3 % and within 1 % at the median; GM6's D 1.6 goes with its A_crit × 2.9, where
the FEMA fit moves to the 10,533 m² end of a flat profile); both models fit the FEMA surface slightly better (MSE 0.96 and 0.88 of the
NCALM values). A_crit is the one quantity that moves, downward on GM2, GM4, GM5 (0.43-0.65) and upward on GM6, GM7: the FEMA areas come
from the full-grid routing of the 2018 survey (GM2's trunk +15 %, GM3's added 2.5 km² inflow, addendum 17 Part A), and the A_crit
profiles are flat enough (section 3) for the optimum to shift. NCALM R_MSE 1.8-4.3 (median 2.8), FEMA 1.5-3.7 (2.5 on the same seven).

**PAMAP AP1-AP5 against 3DEP AP1_3dep and AP2_3dep-AP5_3dep, both at m = 0.47** (ratios 3DEP / PAMAP): K **1.085** (AP3 1.20), D 1.16 (AP3
1.31, AP4 0.94), A_crit 1.39 (AP3 3.06, AP2 2.41, AP4 0.71), MSE_P4 0.97 (0.81-1.24), MSE_AD 1.11 (0.88-1.16), R_MSE 1.13 (0.94-1.22); AD
K 1.12, AD D 1.30. The 3DEP fits carry 8 % more K (the 3DEP k_s is 4 % lower, Part B) and 16 % more D (the 3DEP D₀ is 35 % lower, so D/D₀
is higher on 3DEP, 2.3-5.4 against 4.3-8.3 on PAMAP at D₀,A); A_crit rises on four of five tiles. The v2 conclusions on PAMAP (R_MSE 1.3-3.6,
K/K₀ 0.90-1.01, AD K/K₀ 8.6-10.2) are the 3DEP conclusions (R 1.4-4.4, K/K₀ 0.93-1.16, AD K/K₀ 9.4-10.9).

**Against v2.0.0.** GM2-GM5 at m = 0.31 (and GM1 of the gate) and AP1-AP5 at m = 0.47, with the stored Phase 6 rows joined: **the P4 optimum
(A_crit, K, D, MSE) and the AD optimum (K, D, r, MSE) match the v2.0.0 build on 10 of 10 tiles**, MSE within 2.6 × 10⁻¹⁴ relative, the
parameters identical (Appendix C13). Nothing in the registered grids, the D8 border guard of addendum 16 or the a12 grid design changes a
v2.0.0 number.

## 8. TTV: per-tile m (5.6, 6.9) and the channel-head comparison (5.7)

### 8.1 Per-tile m (5.6, 6.9)
Every primary TTV tile refitted at its own m_t with K₀(m_t) by χ-z (Appendix C14), beside the site-m fits:

| | site m = 0.33 | m_t | change |
| --- | --- | --- | --- |
| F1, MSE_P4 < MSE_AD | 10/11 | **10/11** (TTV11 fails: 182 against 177) | holds at m_t (9 needed) |
| median P4 D/D₀,A | 16.1 | 15.9 | **−1.6 %** |
| median AD K/K₀ | 10.1 | 2.86 | **−72 %** |
| median P4 K/K₀ | 0.96 | 0.99 | +3 % |
| median P4 D/D₀,B | 15.1 | 15.1 | 0 % |

**Rule 6.9: the site-m choice is not immaterial at TTV**: F1 holds at m_t and the partitioned D does not move, but AD's K/K₀ falls from
10.1 to 2.9 at the median (on the six tiles at m_t = 0.05 from 7.6-20.9 to 0.98-2.9; on TTV2, TTV4 and TTV5 at m_t = 0.41-0.45 it rises to
13-33), exactly the pattern addendum 12 found at AP. The partitioned conclusions do not depend on m: K/K₀ 0.69-1.58 at m_t against
0.69-1.42, D/D₀,A 3.7-20 at both, A_crit unchanged on 10 of 11 tiles (TTV6 10,973 → 14,492), R_MSE 1.7-6.1 against 0.97-5.0. At m_t = 0.05
the channel steepness is 0.26-0.39 in units of m^0.1 and K₀ 2-3 × 10⁻⁴ (not comparable with the 0.33 values; only K/K₀ at the same m is).

### 8.2 The channel-head comparison (5.7)
**Field head areas (5.7.1)**: `a18_prep.py --part heads` (job 52955788; Appendix C15). M&D (1989) Table 1, 63 source areas: 5th-95th
percentiles 2,010 / 4,800 / 7,400 / 10,350 / 14,860 m² (**IQR 4,800-10,350**). The 77 digitised heads on the Marin 2019 full-grid routing, the
largest A within 3 m of the head (the rule): **31.6 / 76 / 132 / 304 / 2,298 m²** (IQR 76-304); at the head cell itself 1 / 2 / 6 / 30 / 256 m².
Median 132 against 7,400 m², **ratio 0.018**. The digitised heads do not sit on the 1 m channel network: with the georeference RMS of 9.5 m a
3 m search reaches the hollow axis on 8 of 77 heads (A ≥ 1,000 m²); within 10 m the median is 355 m² and within 20 m 903 m² (34 of 77 above
1,000 m²), still an order of magnitude under Table 1. The 77-head distribution is a georeference-limited quantity at grid scale; the
Table 1 IQR is the field reference the classification uses, and the 77-head IQR is applied to TTV1-TTV3 as the rule says, with this caveat.

**The model (5.7.2) and the classification (5.7.3), against the Table 1 IQR:**

| tile | site m: P4 A_crit (1 % interval) | class | m_t: A_crit (1 %) | class |
| --- | --- | --- | --- | --- |
| TTV1 (M&D area) | 59,499 (59,066-60,312) | above | 0.05: 59,499 (59,152-60,312) | above |
| TTV2 (M&D area) | 9,131 (8,399-9,159) | **consistent** | 0.45: 8,607 (8,467-9,153) | **consistent** |
| TTV3 (M&D area) | 89,124 (89,070-89,144) | above | 0.40: the same | above |
| TTV4 | 40,309 (38,925-40,734) | above | 0.42: the same | above |
| TTV5 | 58,612 (58,350-60,066) | above | 0.41: the same | above |
| TTV6 | 10,973 (10,344-11,683) | **consistent** | 0.05: 14,492 (14,370-14,498) | above |
| TTV7 | 35,966 (35,470-36,940) | above | 0.05: the same | above |
| TTV8 | 38,544 (28,341-40,094) | above | 0.06: 39,096 (38,165-39,653) | above |
| TTV9 | 36,561 (33,976-39,662) | above | 0.35: the same | above |
| TTV11 | 30,362 (16,982-31,139) | above | 0.05: 29,923 (16,982-30,550) | above |
| TTV15 | 26,819 (26,314-27,056) | above | 0.05: 26,938 (26,699-27,525) | above |

Counts: at the site m **consistent 2, above 9, below 0 of 11**; at m_t **1, 10, 0**. TTV1-TTV3 against Table 1: 1 consistent (TTV2), 2 above; against
the 77-head IQR (76-304 m²): 3 above at both m. **Reading, as written (5.7.4): at least 6 of 11 are "above", so the paper states that, at the
1 km tile scale, the partitioned model's A_crit lies above the field channel heads at TTV.** The fitted A_crit is 3-12 times the Table 1
median on nine tiles (27,000-89,000 m²; only TTV2 and TTV6 fall at 9,000-11,000 m², inside or just above the IQR), the 1 % intervals are
narrow, and the four tiles with D at its bound are among the "above" ones (their A_crit is the largest-area end of the profile, where the
tile is nearly all hillslope). No retuning follows.

## 9. What the fits support, and where they are weak

**Supported.**
1. **The fit.** The partitioned model fits the 1 m surface better than AD on 53 of 54 primary tiles (F1 holds at every site: GM 22/22, AP
   21/21, TTV 10/11), by a factor 1.3-6.1 (site medians 2.25, 2.34, 2.59), in sample and out of sample (H1 51/54), on the surveys of 2003,
   2006-08, 2018 and 2019-20 alike, on the mesa and off it, with and without the trunks. The original submission's claim of a better fit
   stands on the expanded set.
2. **K.** The partitioned K is the measured K₀ = E/k_s at every site: medians 1.05, 0.99, 0.96, within ±20 % on 48 of 54 tiles (±42 % on
   the TTV tiles with K/K₀ 0.69-0.76), and holding K at K₀ costs the partitioned fit 2-3 % at GM and AP (P3). AD's K is 2.4 × K₀ at GM, 10 ×
   at AP and TTV, and A1 (AD at K₀) fits 2-31 × worse than AD. F2 holds everywhere (53/54). Stated beside it, as the rule asks: the
   partitioned K agrees with the χ estimate partly by construction (both integrate z along the D8 network with the same m; the fit chooses
   K to match the channel profile, and k_s is that profile's slope).
3. **D at GM and AP.** The partitioned D is closer to the measured D₀ than AD's on 21/22 (GM) and 12-13/13 (AP) tiles at both ends (F3
   holds), with D 1.0-2.8 × D₀ at GM and 2.2-5.4 × at AP; AD's D is 2-3 decades below D₀ at GM (its hillslopes are cut by advection) and 0.6-1.6
   decades below at AP. **"AD's D is off by more than an order of magnitude" is supported at GM** (2.2 decades at the median, on 22 of 22
   tiles) **and not at AP** (0.82 and 0.89 decades at the median: F4 fails; it holds on 5-6 of the 13 AP tiles with a D₀) and **marginally at
   TTV** (1.10 / 1.05 on the primary set, 0.99 / 0.95 on the clean subset).
4. **Equal counts at GM.** The partitioned model wins with the same number of free parameters as AD, with D fixed at D₀ or free, in sample
   and out of sample (E1, E2, H1, H2 hold on 19-21 of 22). GM's advantage is not a parameter-count effect, on 22 tiles.
5. **Continuity.** The FEMA and NCALM fits agree in K and D within 3 % and 1 % at the median; 3DEP and PAMAP within 9 % and 16 %; the
   v2.0.0 numbers are reproduced exactly; the trunk fit changes nothing material on 7 of 9 tiles.
6. **A_crit against the field heads at TTV**: above them on 9 of 11 tiles, by a factor 3-12, with narrow intervals.

**Weak.**
1. **The AP D statement.** The partitioned D at AP is 2.5-3.5 × D₀ (the excess is the same 0.4-0.5 decades at either end of the bracket and
   on either survey), AD's 0.15 × D₀, and neither is within the bracket; the "order of magnitude" wording fails at the site median. With D
   held at D₀ the partitioned model loses the equal-count comparisons at AP (E1_A 7/13, E2 7-8/13, H2 9-10/13) as in addendum 12.
2. **TTV's D is not resolved.** The partitioned D is 15-20 × D₀ and on four tiles on the box bound (0.05 m²/yr); F3 fails; the D₀-fixed
   variants fail 2/9; TTV's hillslopes (C_ht 2-3 × GM's, D₀ 0.003) are fitted by a D of 0.03-0.05, the GM value. Whether a wider box moves
   it, or whether this is the nonlinear-hillslope signature the v2 work found at GM, is a decision (section 10). The m choice does not
   change it (−2 %).
3. **Bound hits.** P4's D at the upper bound on GM20, GM19 and four TTV tiles; AD's D at the lower bound on GM11, GM21 (and NCALM GM3,
   GM4) and at the upper bound on TTV8, TTV15. Those AD fits are degenerate (pure advection or pure diffusion), and the rules count them
   as written.
4. **Lopsided splits** on 12 of 54 primary tiles (7 at AP); the balanced subsets agree with every headline verdict, but the AP held-out
   H2 rests on 7 balanced tiles.
5. **The trunk fit is material on AP3** (both surveys): AD's MSE halves and the 3DEP partitioned K/K₀ goes 1.16 → 1.62 with the 11 km²
   trunk held. The R_MSE of record on that tile (4.4) overstates AD's deficit by about half.
6. **The 77-head comparison** is georeference-limited (median 132 m² at grid scale against Table 1's 7,400); only the Table 1 IQR carries
   the TTV classification, and the TTV1-TTV3 subset against the 77 heads says "above" for that reason as much as any other.
7. **The site m at TTV** rests on a profile with six tiles at the grid edge (Part B); the partitioned results do not depend on it, AD's
   K/K₀ does (10 → 2.9).
8. **The P4 fill** is the 1 % design (deviation 2); the 5 % intervals are read from the thinned grid; the gate shows the optima are the
   Phase 6 ones where they can be compared.

**The claims of the original submission, one by one, on the expanded set:**
- *the partitioned model fits better than AD*: **supported** at every site (53/54 tiles, in and out of sample);
- *the partitioned K is the measured K₀, AD's is not*: **supported** at every site (medians 1.05 / 0.99 / 0.96 against 2.4 / 10 / 10);
- *the partitioned D is the measured D₀, AD's is not*: **supported in direction at GM and AP** (closer to D₀ on 21/22 and 12-13/13; GM within
  × 1.0-2.8, AP × 2.2-5.4), **not at TTV** (× 15-20, bounded); AD's D is below D₀ everywhere;
- *AD's parameters are off by orders of magnitude*: **K**: AD needs 10 × K₀ at AP and TTV (one order), 2.4 × at GM (not an order); **D**: 2.2
  decades at GM (yes), 0.8-0.9 at AP (no, F4 fails), 1.0-1.1 at TTV (marginal).

## 10. Decisions needed
1. **The AP D wording.** F3 holds and F4 fails at AP. Proposed wording: "at AP the partitioned D lies 2-5 × above the measured D₀ at either
   end of the bracket and AD's 4-8 × below it; the partitioned D is the closer on 12-13 of 13 tiles; AD's deficit is 0.8-0.9 decades at the
   median, not an order of magnitude". Accept, or restrict the "order of magnitude" statement to GM.
2. **TTV's D at the box bound (4 of 11 tiles) and 15-20 × D₀ elsewhere.** Options: (a) report as is, with F3 failing and the bound marked;
   (b) widen the D bound at TTV (a rerun of 11 + 11 partitioned fits, about 110 core-hours) to see where D goes; (c) treat TTV as the site
   where the linear hillslope fails (its hillslopes are 2-3 × more curved than GM's at 11 m) and say so. I recommend (a) in this report and
   (b) only if the paper needs a TTV D value.
3. **The TTV channel-head reading** ("A_crit lies above the field channel heads at the 1 km tile scale") is the rule's; whether to keep the
   77-head comparison in the paper given the georeference limit (section 8.2), or to cite Table 1 only.
4. **The trunk fit of AP3_3dep** (material): report both values (as here) or adopt the trunk fit for the trunk tiles in the paper's tables.
5. **The equal-count wording at AP and TTV**: as addendum 12 recommended for AP, "at equal parameter counts the partitioned model beats AD
   only when D is free"; the same now holds at TTV.
6. **TTV11** (Tier 2, fire roads): the one F1 failure. Keep in the primary set (the rule) with the clean subset beside it, or move it to Tier
   1. The clean subset changes no TTV verdict except F4 (which then fails).
7. **The bound-degenerate AD fits** (GM11, GM21, NCALM GM3, GM4; TTV8, TTV15): keep them in the counts (the rule) or mark them.
8. **The site m at TTV.** AD's K/K₀ depends on it (10 → 2.9); the partitioned results do not. Whether the paper reports AD's TTV K/K₀ at all.
9. Nothing is merged or tagged. Push only.

## 11. Appendices: all tables, the run ids, checksums, Slurm accounting

### C1. The decisions of section 1 as applied: the measured blocks (decision 9), the tiles without D0 (decision 4), the trunk list (decision 8)

63 `measured` blocks written into `configs/tiles/<tile>.yaml` from `20261119T002830Z_a17b_measurement/measure/measurement.json`; the v2.0.0 blocks of GM1, GM2, GM3, GM4, GM5, AP1, AP2, AP3, AP4, AP5 are untouched. Loader check: 63 configs load; failures: none.

| tile | m (site) | k_s | K0 = E/k_s | C_ht(L_f) | D0_A | D0_B | hilltop px | D0 |
|---|---|---|---|---|---|---|---|---|
| AP10 | 0.47 | 13.18 | 4.173e-06 | - | - | - | 145 | null |
| AP11 | 0.47 | 13.47 | 4.085e-06 | - | - | - | 0 | null |
| AP13 | 0.47 | 10.22 | 5.382e-06 | 0.0123 | 0.004471 | 0.007562 | 3,087 |  |
| AP14 | 0.47 | 13.1 | 4.199e-06 | - | - | - | 118 | null |
| AP18 | 0.47 | 10.43 | 5.275e-06 | - | - | - | 25 | null |
| AP19 | 0.47 | 12.78 | 4.303e-06 | 0.00899 | 0.006118 | 0.007877 | 6,160 |  |
| AP1_3dep | 0.47 | 12.27 | 4.482e-06 | 0.009987 | 0.005507 | 0.006664 | 4,575 |  |
| AP20 | 0.47 | 11.54 | 4.765e-06 | 0.01019 | 0.0054 | 0.007204 | 6,710 |  |
| AP23 | 0.47 | 10.59 | 5.194e-06 | 0.01502 | 0.003661 | 0.004505 | 2,197 |  |
| AP24 | 0.47 | 12.11 | 4.540e-06 | 0.009261 | 0.005939 | 0.008242 | 5,040 |  |
| AP25 | 0.47 | 11.51 | 4.779e-06 | 0.01425 | 0.003859 | 0.004573 | 1,124 |  |
| AP27 | 0.47 | 11.52 | 4.774e-06 | - | - | - | 83 | null |
| AP28 | 0.47 | 12.55 | 4.384e-06 | - | - | - | 119 | null |
| AP29 | 0.47 | 10.75 | 5.117e-06 | 0.008735 | 0.006297 | 0.006704 | 3,606 |  |
| AP2_3dep | 0.47 | 10.9 | 5.047e-06 | - | - | - | 13 | null |
| AP3_3dep | 0.47 | 11.04 | 4.984e-06 | 0.01258 | 0.004372 | 0.005272 | 4,094 |  |
| AP4_3dep | 0.47 | 11.43 | 4.811e-06 | 0.01026 | 0.00536 | 0.007554 | 6,417 |  |
| AP5_3dep | 0.47 | 10.38 | 5.299e-06 | 0.01069 | 0.005144 | 0.006494 | 5,502 |  |
| AP6 | 0.47 | 11.65 | 4.720e-06 | 0.00924 | 0.005952 | 0.007218 | 4,391 |  |
| AP7 | 0.47 | 10.25 | 5.364e-06 | 0.0105 | 0.005239 | 0.005771 | 2,756 |  |
| AP8 | 0.47 | 10.07 | 5.463e-06 | 0.00913 | 0.006024 | 0.007696 | 3,340 |  |
| AP9 | 0.47 | 12 | 4.583e-06 | - | - | - | 30 | null |
| GM11 | 0.29 | 1.834 | 8.016e-05 | 0.008028 | 0.01831 | 0.01974 | 4,273 |  |
| GM12 | 0.29 | 1.96 | 7.500e-05 | 0.01285 | 0.01144 | 0.01273 | 7,208 |  |
| GM13 | 0.29 | 2.204 | 6.670e-05 | 0.005095 | 0.02885 | 0.02786 | 5,046 |  |
| GM14 | 0.29 | 2.125 | 6.918e-05 | 0.01571 | 0.009355 | 0.0117 | 6,475 |  |
| GM15 | 0.29 | 2.27 | 6.475e-05 | 0.01637 | 0.00898 | 0.01073 | 4,822 |  |
| GM16 | 0.29 | 2.207 | 6.660e-05 | 0.01207 | 0.01218 | 0.01037 | 6,454 |  |
| GM17 | 0.29 | 2.054 | 7.158e-05 | 0.009473 | 0.01552 | 0.01987 | 4,224 |  |
| GM18 | 0.29 | 1.465 | 1.003e-04 | 0.0276 | 0.005326 | 0.006456 | 2,655 |  |
| GM19 | 0.29 | 2.388 | 6.155e-05 | 0.01185 | 0.01241 | 0.012 | 7,075 |  |
| GM1_fema | 0.29 | 2.384 | 6.166e-05 | 0.01434 | 0.01025 | 0.01093 | 8,554 |  |
| GM20 | 0.29 | 1.733 | 8.482e-05 | 0.008055 | 0.01825 | 0.01929 | 5,234 |  |
| GM21 | 0.29 | 2.369 | 6.205e-05 | 0.01232 | 0.01193 | 0.01274 | 7,692 |  |
| GM22 | 0.29 | 2.083 | 7.056e-05 | 0.009136 | 0.01609 | 0.01888 | 2,509 |  |
| GM24 | 0.29 | 1.887 | 7.790e-05 | 0.004701 | 0.03127 | 0.03167 | 6,482 |  |
| GM25 | 0.29 | 2.254 | 6.522e-05 | 0.01386 | 0.01061 | 0.01185 | 5,909 |  |
| GM26 | 0.29 | 1.609 | 9.138e-05 | 0.008447 | 0.0174 | 0.01901 | 3,448 |  |
| GM27 | 0.29 | 1.937 | 7.589e-05 | 0.02019 | 0.00728 | 0.008165 | 5,349 |  |
| GM28 | 0.29 | 2.263 | 6.496e-05 | 0.01005 | 0.01462 | 0.01718 | 6,318 |  |
| GM29 | 0.29 | 2.106 | 6.981e-05 | 0.0101 | 0.01455 | 0.0139 | 6,738 |  |
| GM2_fema | 0.29 | 2.02 | 7.276e-05 | 0.01175 | 0.01251 | 0.01197 | 5,425 |  |
| GM30 | 0.29 | 1.525 | 9.642e-05 | 0.01319 | 0.01114 | 0.01171 | 6,877 |  |
| GM3_fema | 0.29 | 1.827 | 8.045e-05 | 0.009512 | 0.01545 | 0.01583 | 6,283 |  |
| GM4_fema | 0.29 | 3.238 | 4.540e-05 | 0.01247 | 0.01178 | 0.01097 | 5,676 |  |
| GM5_fema | 0.29 | 2.265 | 6.490e-05 | 0.01685 | 0.008725 | 0.008471 | 3,310 |  |
| GM6 | 0.29 | 2.124 | 6.920e-05 | 0.01344 | 0.01094 | 0.01031 | 1,987 |  |
| GM6_fema | 0.29 | 2.147 | 6.847e-05 | 0.009044 | 0.01625 | 0.01432 | 4,549 |  |
| GM7 | 0.29 | 1.832 | 8.025e-05 | 0.006672 | 0.02203 | 0.02203 | 920 |  |
| GM7_fema | 0.29 | 1.781 | 8.255e-05 | 0.004148 | 0.03544 | 0.03574 | 3,585 |  |
| GM8 | 0.29 | 1.738 | 8.457e-05 | 0.009652 | 0.01523 | 0.01564 | 4,963 |  |
| GM9 | 0.29 | 1.775 | 8.281e-05 | 0.007219 | 0.02036 | 0.0211 | 3,779 |  |
| TTV1 | 0.33 | 6.85 | 1.124e-05 | 0.02651 | 0.002905 | 0.002905 | 966 |  |
| TTV11 | 0.33 | 6.642 | 1.159e-05 | 0.0333 | 0.002312 | 0.002558 | 3,192 |  |
| TTV15 | 0.33 | 5.613 | 1.372e-05 | 0.05112 | 0.001506 | 0.001506 | 579 |  |
| TTV2 | 0.33 | 4.871 | 1.581e-05 | 0.01726 | 0.004461 | 0.004708 | 1,995 |  |
| TTV3 | 0.33 | 2.921 | 2.636e-05 | - | - | - | 122 | null |
| TTV4 | 0.33 | 4.772 | 1.614e-05 | - | - | - | 86 | null |
| TTV5 | 0.33 | 3.538 | 2.176e-05 | 0.03083 | 0.002498 | 0.002699 | 3,203 |  |
| TTV6 | 0.33 | 6.19 | 1.244e-05 | 0.03864 | 0.001993 | 0.001993 | 612 |  |
| TTV7 | 0.33 | 7.136 | 1.079e-05 | 0.02485 | 0.003098 | 0.003314 | 2,983 |  |
| TTV8 | 0.33 | 6.695 | 1.150e-05 | 0.01933 | 0.003984 | 0.004479 | 5,287 |  |
| TTV9 | 0.33 | 7.311 | 1.053e-05 | 0.01832 | 0.004202 | 0.006252 | 3,291 |  |

Trunk rule: trunk cells: A >= 1e+06 m2 on the tile's area grid; trunk fit on every primary and continuity tile whose largest A exceeds 5e+06 m2. Trunk list (max A > 5 km2 on primary and continuity tiles): **AP3, AP3_3dep, AP4, AP4_3dep, GM2, GM2_fema, GM4, GM4_fema, GM5_fema** (expected GM2_fema, GM4_fema, GM5_fema, AP3_3dep, AP4_3dep).

| tile | site | group | largest A (km2) | cells A >= 1 km2 | fraction of valid cells | trunk fit |
|---|---|---|---|---|---|---|
| AP4_3dep | AP | primary | 17.14 | 218 | 0.02 % | yes |
| AP4 | AP | continuity | 17.00 | 213 | 0.02 % | yes |
| AP3_3dep | AP | primary | 11.07 | 1,514 | 0.15 % | yes |
| AP3 | AP | continuity | 11.01 | 1,521 | 0.15 % | yes |
| GM2_fema | GM | primary | 10.34 | 581 | 0.06 % | yes |
| GM2 | GM | continuity | 8.89 | 592 | 0.06 % | yes |
| GM4_fema | GM | primary | 6.19 | 1,424 | 0.14 % | yes |
| GM4 | GM | continuity | 6.19 | 1,471 | 0.15 % | yes |
| GM5_fema | GM | primary | 5.54 | 1,644 | 0.16 % | yes |
| GM5 | GM | continuity | 4.73 | 1,647 | 0.16 % | no |
| AP13 | AP | primary | 3.97 | 1,100 | 0.11 % | no |
| AP7 | AP | primary | 3.65 | 1,332 | 0.13 % | no |
| AP25 | AP | primary | 3.27 | 1,100 | 0.11 % | no |
| GM28 | GM | offmesa | 3.26 | 966 | 0.10 % | no |
| TTV9 | TTV | primary | 2.75 | 1,493 | 0.15 % | no |
| TTV3 | TTV | primary | 2.71 | 1,030 | 0.10 % | no |
| GM3_fema | GM | primary | 2.48 | 8 | 0.00 % | no |
| GM24 | GM | primary | 2.44 | 924 | 0.09 % | no |
| GM13 | GM | primary | 2.37 | 956 | 0.10 % | no |
| AP19 | AP | primary | 2.11 | 651 | 0.07 % | no |
| GM26 | GM | primary | 2.00 | 591 | 0.06 % | no |
| AP18 | AP | primary | 1.98 | 677 | 0.07 % | no |
| AP11 | AP | primary | 1.96 | 364 | 0.04 % | no |
| AP29 | AP | primary | 1.78 | 838 | 0.08 % | no |
| AP27 | AP | primary | 1.77 | 898 | 0.09 % | no |
| AP2 | AP | continuity | 1.74 | 1,190 | 0.12 % | no |
| AP23 | AP | primary | 1.72 | 478 | 0.05 % | no |
| GM16 | GM | primary | 1.71 | 646 | 0.06 % | no |
| GM9 | GM | primary | 1.69 | 400 | 0.04 % | no |
| AP2_3dep | AP | primary | 1.69 | 1,185 | 0.12 % | no |
| GM22 | GM | primary | 1.64 | 317 | 0.03 % | no |
| AP8 | AP | primary | 1.63 | 835 | 0.08 % | no |
| AP24 | AP | primary | 1.60 | 59 | 0.01 % | no |
| GM19 | GM | offmesa | 1.55 | 1,102 | 0.11 % | no |
| GM8 | GM | primary | 1.54 | 847 | 0.08 % | no |
| AP9 | AP | primary | 1.52 | 398 | 0.04 % | no |
| AP6 | AP | primary | 1.49 | 479 | 0.05 % | no |
| GM11 | GM | primary | 1.49 | 713 | 0.07 % | no |
| GM6_fema | GM | primary | 1.49 | 505 | 0.05 % | no |
| GM6 | GM | continuity | 1.48 | 513 | 0.05 % | no |
| GM30 | GM | offmesa | 1.37 | 829 | 0.08 % | no |
| GM18 | GM | offmesa | 1.34 | 73 | 0.01 % | no |
| GM20 | GM | primary | 1.28 | 102 | 0.01 % | no |
| GM7_fema | GM | primary | 1.25 | 330 | 0.03 % | no |
| AP5 | AP | continuity | 1.22 | 601 | 0.06 % | no |
| GM7 | GM | continuity | 1.18 | 239 | 0.02 % | no |
| AP1_3dep | AP | continuity | 1.18 | 387 | 0.04 % | no |
| AP5_3dep | AP | primary | 1.17 | 652 | 0.07 % | no |
| GM1 | GM | continuity | 1.16 | 567 | 0.06 % | no |
| GM1_fema | GM | primary | 1.15 | 588 | 0.06 % | no |
| AP1 | AP | continuity | 1.12 | 244 | 0.02 % | no |
| GM17 | GM | primary | 1.09 | 151 | 0.02 % | no |
| TTV8 | TTV | primary | 1.04 | 3 | 0.00 % | no |

### C2. The map's description of the Paso Robles Formation labels (decision 1), from every attribute field of GEOLOGY.shp

| label | polygons | attribute values (count) |
|---|---|---|
| QTp | 252 | SOURCE: Dibblee (1971l) (60), Dibblee (1971b) (50), Dibblee (1971g) (46), Dibblee (1971a) (26), Dibblee (1971m) (25), Dibblee (1971d) (19), Dibblee (1971f) (7), Tinsley (1975) (6), Dibblee (1971j) (4), Dibblee (1971e) (4), Rosenberg (2001, unpub. mapping) (2), Geomatrix (1997) (1); DET_LAB: QTp (232), QTp? (20); CLASS: Sedimentary (252); LITHOLOGY: Clastic sedimentary (252); DET_NAME: Paso Robles Formation, undifferentiated (232), Paso Robles Formation, undifferentiated? (20); GEN_NAME: Paso Robles Formation, undifferentiated (252); GEN_LAB: QTp (252); ERA: Cenozoic (252); PERIOD: Quaternary-Tertiary (252); EPOCH: Pliocene-Pleistocene? (252); LIQ_SUSC: Low (252); STRENGTH: C (252); UBC: C (252) |
| QTcl | 20 | SOURCE: Dibblee (1971m) (16), Dibblee (1971a) (2), Dibblee (1971f) (2); DET_LAB: QTcl (20); CLASS: Sedimentary (20); LITHOLOGY: Clastic sedimentary (20); DET_NAME: Paso Robles Formation, clay (20); GEN_NAME: Paso Robles Formation (20); GEN_LAB: QTcl (20); ERA: Cenozoic (20); PERIOD: Quaternary-Tertiary (20); EPOCH: Pliocene-Pleistocene? (20); LIQ_SUSC: Low (20); STRENGTH: C (20); UBC: C (20) |
| QTg | 6 | SOURCE: Dibblee (1971l) (5), Dibblee (1971m) (1); DET_LAB: QTg (6); CLASS: Sedimentary (6); LITHOLOGY: Clastic sedimentary (6); DET_NAME: Paso Robles Formation, gravel (6); GEN_NAME: Paso Robles Formation (6); GEN_LAB: QTg (6); ERA: Cenozoic (6); PERIOD: Quaternary-Tertiary (6); EPOCH: Pliocene-Pleistocene? (6); LIQ_SUSC: Low (6); STRENGTH: C (6); UBC: C (6) |
| QTm | 7 | SOURCE: Dibblee (1971m) (7); DET_LAB: QTm (7); CLASS: Sedimentary (7); LITHOLOGY: Clastic sedimentary (7); DET_NAME: Paso Robles Formation, marl (7); GEN_NAME: Paso Robles Formation (7); GEN_LAB: QTm (7); ERA: Cenozoic (7); PERIOD: Quaternary-Tertiary (7); EPOCH: Pliocene-Pleistocene? (7); LIQ_SUSC: Low (7); STRENGTH: C (7); UBC: C (7) |
| QTpc | 1 | SOURCE: Dibblee (1971i) (1); DET_LAB: QTpc (1); CLASS: Sedimentary (1); LITHOLOGY: Clastic sedimentary (1); DET_NAME: Paso Robles Formation, clay and gravel (1); GEN_NAME: Paso Robles Formation, clay and gravel (1); GEN_LAB: QTpc (1); ERA: Cenozoic (1); PERIOD: Quaternary-Tertiary (1); EPOCH: Pliocene-Pleistocene? (1); LIQ_SUSC: Low (1); STRENGTH: C (1); UBC: C (1) |

### C3. GM site summaries on the 22-tile primary set and the 19-tile clean subset (decision 1), beside the Part B 19-tile values

| group | n | site-pooled k_s (1e3) | 95 % | n_eff | K0 site-pooled | K0 median | D0_A median | D0_B median | C_ht median | window test | k_s recheck max rel |
|---|---|---|---|---|---|---|---|---|---|---|---|
| primary_22 | 22 | 2.036 | [1.925, 2.142] | 40.8 | 7.221e-05 | 7.217e-05 | 0.01534 | 0.01498 | 0.009582 | 14 / 22 (non-trivial 22) | 0.0e+00 |
| clean_19 | 19 | 2.071 | [1.945, 2.181] | 36.5 | 7.097e-05 | 7.158e-05 | 0.01545 | 0.01564 | 0.009512 | 12 / 19 (non-trivial 19) | 0.0e+00 |
| Part B primary | 19 | 2.051 | [1.908, 2.183] | 33.9 | 7.167e-05 | 7.158e-05 | 0.01545 | 0.01432 | 0.009512 | 11 / 19 | - |
| Part B clean | 16 | 2.096 | [1.944, 2.238] | 29.5 | 7.014e-05 | 7.107e-05 | 0.01549 | 0.01507 | 0.009492 | 9 / 16 | - |
| Part B ncalm | 7 | 2.215 | [2, 2.43] | 11.2 | 6.637e-05 | 6.888e-05 | 0.009855 | 0.01012 | 0.01492 | 7 / 7 | - |

S(m) over the m-profile runs (decision 1): primary_22: argmin 0.29, 1 % [[0.26, 0.32]], 5 % [[0.22, 0.36]], leave-one-out 0.29-0.30; clean_19: argmin 0.29, 1 % [[0.26, 0.31]], 5 % [[0.22, 0.35]], leave-one-out 0.28-0.30

| returned tile | k_s | K0 | C_ht(L_f) | D0_A | D0_B | hilltop px | theta (1e3) |
|---|---|---|---|---|---|---|---|
| GM8 | 1.738 | 8.457e-05 | 0.009652 | 0.01523 | 0.01564 | 4,963 | 0.35 |
| GM9 | 1.775 | 8.281e-05 | 0.007219 | 0.02036 | 0.0211 | 3,779 | 0.40 |
| GM21 | 2.369 | 6.205e-05 | 0.01232 | 0.01193 | 0.01274 | 7,692 | 0.25 |

| FEMA tile | NCALM twin | D0_A FEMA | D0_A twin | D0_B twin | FEMA / twin |
|---|---|---|---|---|---|
| GM1_fema | GM1 | 0.01025 | 0.009001 | 0.009354 | 1.14 |
| GM2_fema | GM2 | 0.01251 | 0.00977 | 0.01015 | 1.28 |
| GM3_fema | GM3 | 0.01545 | 0.01012 | 0.01012 | 1.53 |
| GM4_fema | GM4 | 0.01178 | 0.009855 | 0.009435 | 1.20 |
| GM5_fema | GM5 | 0.008725 | 0.007589 | 0.007496 | 1.15 |
| GM6_fema | GM6 | 0.01625 | 0.01094 | 0.01031 | 1.49 |
| GM7_fema | GM7 | 0.03544 | 0.02203 | 0.02203 | 1.61 |

### C4. The same-pixel curvature diagnostic (3.1): C_ht(L_f) medians on the old and new surveys' surfaces and pixel sets

| old / new | n old | n new | (a) old on old | (b) new on old px | (c) old on new px | (d) new on new | (b)/(a) | (d)/(c) | (c)/(a) | (d)/(b) | (d)/(a) | old px mapped | offset median (m) | offset max (m) | dropped edge / nodata | (a) vs Part B | (d) vs Part B | check |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM1/GM1_fema | 3,583 | 8,554 | 0.01633 | 0.01586 | 0.01429 | 0.01434 | 0.971 | 1.004 | 0.875 | 0.904 | 0.878 | 3570 / 3583 | 0.00 | 0.00 | 13 / 0 | 0.0e+00 | 0.0e+00 | yes |
| GM2/GM2_fema | 2,530 | 5,425 | 0.01505 | 0.01458 | 0.01184 | 0.01175 | 0.969 | 0.993 | 0.787 | 0.806 | 0.781 | 2505 / 2530 | 0.00 | 0.00 | 25 / 0 | 0.0e+00 | 0.0e+00 | yes |
| GM3/GM3_fema | 1,500 | 6,283 | 0.01453 | 0.01354 | 0.009612 | 0.009512 | 0.932 | 0.990 | 0.662 | 0.702 | 0.655 | 1498 / 1500 | 0.00 | 0.00 | 2 / 0 | 0.0e+00 | 0.0e+00 | yes |
| GM4/GM4_fema | 3,347 | 5,676 | 0.01492 | 0.01427 | 0.01284 | 0.01247 | 0.957 | 0.971 | 0.861 | 0.874 | 0.836 | 3341 / 3347 | 0.00 | 0.00 | 6 / 0 | 0.0e+00 | 0.0e+00 | yes |
| GM5/GM5_fema | 1,128 | 3,310 | 0.01937 | 0.01893 | 0.01699 | 0.01685 | 0.977 | 0.992 | 0.877 | 0.890 | 0.870 | 1112 / 1128 | 0.00 | 0.00 | 16 / 0 | 0.0e+00 | 0.0e+00 | yes |
| GM6/GM6_fema | 1,987 | 4,549 | 0.01344 | 0.01342 | 0.008918 | 0.009044 | 0.999 | 1.014 | 0.663 | 0.674 | 0.673 | 1981 / 1987 | 0.00 | 0.00 | 6 / 0 | 0.0e+00 | 0.0e+00 | yes |
| GM7/GM7_fema | 920 | 3,585 | 0.006672 | 0.006185 | 0.004406 | 0.004148 | 0.927 | 0.942 | 0.660 | 0.671 | 0.622 | 905 / 920 | 0.00 | 0.00 | 15 / 0 | 0.0e+00 | 0.0e+00 | yes |
| AP1/AP1_3dep | 3,446 | 4,575 | 0.01533 | 0.01039 | 0.009356 | 0.009987 | 0.678 | 1.067 | 0.610 | 0.961 | 0.651 | 3446 / 3446 | 0.39 | 0.70 | 0 / 0 | 0.0e+00 | 0.0e+00 | yes |
| AP2/AP2_3dep | 944 | 13 | 0.01709 | 0.01282 | 0.02145 | 0.02006 | 0.750 | 0.935 | 1.255 | 1.564 | 1.174 | 941 / 944 | 0.43 | 0.68 | 3 / 0 | 0.0e+00 | 0.0e+00 | yes |
| AP3/AP3_3dep | 533 | 4,094 | 0.02553 | 0.02022 | 0.01252 | 0.01258 | 0.792 | 1.005 | 0.490 | 0.622 | 0.493 | 530 / 533 | 0.41 | 0.66 | 3 / 0 | 0.0e+00 | 0.0e+00 | yes |
| AP4/AP4_3dep | 4,711 | 6,417 | 0.01515 | 0.01097 | 0.009623 | 0.01026 | 0.724 | 1.066 | 0.635 | 0.936 | 0.677 | 4708 / 4711 | 0.41 | 0.70 | 3 / 0 | 0.0e+00 | 0.0e+00 | yes |
| AP5/AP5_3dep | 3,575 | 5,502 | 0.01883 | 0.01446 | 0.0118 | 0.01069 | 0.768 | 0.906 | 0.627 | 0.740 | 0.568 | 3565 / 3575 | 0.41 | 0.70 | 10 / 0 | 0.0e+00 | 0.0e+00 | yes |

| site | pairs | S (surface) | P (selection) | total (d)/(a) median | Part B total ratio | reading |
|---|---|---|---|---|---|---|
| GM | GM NCALM -> FEMA | 0.974 | 0.797 | 0.781 | 0.78 | selection |
| AP | AP PAMAP -> 3DEP | 0.849 | 0.687 | 0.651 | 0.65 | mixed |
| AP_without_AP2 | AP PAMAP -> 3DEP without AP2 | 0.849 | 0.631 | 0.610 | 0.65 | mixed |

Reading rule: surface if S <= 0.85 and 0.90 <= P <= 1.10; selection if 0.90 <= S <= 1.10 and P <= 0.85; mixed otherwise. Check against Part B: tolerance 0.001; failures: none.

### C5. Hilltop QC figures (3.2): the pixel classes per tile, from hilltop_qc/tiles.json

| tile | survey | rollover A_c (m2) | (i) channel | (ii) excluded near channel | (iii) counted hilltop | (iv) divide dropped | of which steep | of which edge | hatched (filled) cells | C_ht(L_f) recomputed | figure |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | PAMAP 2006-08 | 282 | 55,369 | 2,658 | 3,446 | 87,886 | 382 | 206 | 0 | 0.01533 | AP1_hilltop_qc.png |
| AP10 | 3DEP 2019-20 | 11 | 350,662 | 23,233 | 145 | 209,111 | 2,629 | 470 | 19,698 | 0.02104 | AP10_hilltop_qc.png |
| AP11 | 3DEP 2019-20 | 11 | 372,852 | 5,553 | 0 | 216,132 | 550 | 425 | 22,601 | - | AP11_hilltop_qc.png |
| AP13 | 3DEP 2019-20 | 355 | 60,439 | 1,515 | 3,087 | 241,290 | 244 | 465 | 26,704 | 0.0123 | AP13_hilltop_qc.png |
| AP14 | 3DEP 2019-20 | 11 | 351,825 | 17,449 | 118 | 214,825 | 1,379 | 450 | 21,138 | 0.02625 | AP14_hilltop_qc.png |
| AP18 | 3DEP 2019-20 | 11 | 353,624 | 6,618 | 25 | 230,551 | 334 | 516 | 24,671 | 0.02918 | AP18_hilltop_qc.png |
| AP19 | 3DEP 2019-20 | 355 | 59,299 | 2,824 | 6,160 | 219,323 | 679 | 452 | 19,854 | 0.00899 | AP19_hilltop_qc.png |
| AP1_3dep | 3DEP 2019-20 | 224 | 77,226 | 4,455 | 4,575 | 214,701 | 748 | 492 | 16,100 | 0.009987 | AP1_3dep_hilltop_qc.png |
| AP2 | PAMAP 2006-08 | 56 | 177,197 | 2,553 | 944 | 99,089 | 166 | 199 | 0 | 0.01709 | AP2_hilltop_qc.png |
| AP20 | 3DEP 2019-20 | 224 | 72,432 | 7,512 | 6,710 | 230,709 | 799 | 426 | 28,498 | 0.01019 | AP20_hilltop_qc.png |
| AP23 | 3DEP 2019-20 | 282 | 74,884 | 1,179 | 2,197 | 249,514 | 786 | 508 | 28,795 | 0.01502 | AP23_hilltop_qc.png |
| AP24 | 3DEP 2019-20 | 224 | 77,301 | 4,733 | 5,040 | 236,253 | 192 | 477 | 28,176 | 0.009261 | AP24_hilltop_qc.png |
| AP25 | 3DEP 2019-20 | 112 | 113,790 | 1,628 | 1,124 | 236,661 | 179 | 422 | 22,864 | 0.01425 | AP25_hilltop_qc.png |
| AP27 | 3DEP 2019-20 | 11 | 342,428 | 16,475 | 83 | 229,975 | 2,054 | 484 | 24,407 | 0.04169 | AP27_hilltop_qc.png |
| AP28 | 3DEP 2019-20 | 11 | 346,690 | 14,575 | 119 | 217,571 | 803 | 451 | 22,061 | 0.02034 | AP28_hilltop_qc.png |
| AP29 | 3DEP 2019-20 | 224 | 77,848 | 2,916 | 3,606 | 223,378 | 380 | 451 | 19,339 | 0.008735 | AP29_hilltop_qc.png |
| AP2_3dep | 3DEP 2019-20 | 11 | 350,750 | 7,715 | 13 | 229,009 | 627 | 494 | 27,619 | 0.02006 | AP2_3dep_hilltop_qc.png |
| AP3 | PAMAP 2006-08 | 18 | 341,724 | 8,876 | 533 | 107,924 | 673 | 194 | 0 | 0.02553 | AP3_hilltop_qc.png |
| AP3_3dep | 3DEP 2019-20 | 178 | 94,160 | 7,007 | 4,094 | 230,697 | 1,231 | 425 | 19,236 | 0.01258 | AP3_3dep_hilltop_qc.png |
| AP4 | PAMAP 2006-08 | 447 | 43,416 | 1,425 | 4,711 | 93,558 | 184 | 189 | 0 | 0.01515 | AP4_hilltop_qc.png |
| AP4_3dep | 3DEP 2019-20 | 282 | 65,813 | 4,823 | 6,417 | 218,587 | 930 | 435 | 16,422 | 0.01026 | AP4_3dep_hilltop_qc.png |
| AP5 | PAMAP 2006-08 | 56 | 162,113 | 3,532 | 3,575 | 94,953 | 145 | 214 | 0 | 0.01883 | AP5_hilltop_qc.png |
| AP5_3dep | 3DEP 2019-20 | 282 | 62,949 | 2,784 | 5,502 | 230,513 | 306 | 446 | 29,557 | 0.01069 | AP5_3dep_hilltop_qc.png |
| AP6 | 3DEP 2019-20 | 355 | 56,595 | 2,404 | 4,391 | 241,412 | 773 | 457 | 28,064 | 0.00924 | AP6_hilltop_qc.png |
| AP7 | 3DEP 2019-20 | 282 | 70,226 | 1,838 | 2,756 | 226,759 | 623 | 465 | 14,762 | 0.0105 | AP7_hilltop_qc.png |
| AP8 | 3DEP 2019-20 | 224 | 80,497 | 2,539 | 3,340 | 222,446 | 234 | 463 | 20,619 | 0.00913 | AP8_hilltop_qc.png |
| AP9 | 3DEP 2019-20 | 11 | 343,427 | 19,945 | 30 | 231,202 | 1,795 | 480 | 37,013 | 0.02979 | AP9_hilltop_qc.png |
| GM1 | NCALM 2003 | 35 | 268,775 | 7,529 | 3,583 | 97,275 | 1,987 | 194 | 0 | 0.01633 | GM1_hilltop_qc.png |
| GM11 | FEMA 2018 | 355 | 45,809 | 742 | 4,273 | 256,839 | 114 | 482 | 9,282 | 0.008028 | GM11_hilltop_qc.png |
| GM12 | FEMA 2018 | 178 | 80,667 | 4,652 | 7,208 | 269,608 | 3,595 | 519 | 14,644 | 0.01285 | GM12_hilltop_qc.png |
| GM13 | FEMA 2018 | 355 | 43,646 | 1,261 | 5,046 | 274,511 | 849 | 542 | 11,246 | 0.005095 | GM13_hilltop_qc.png |
| GM14 | FEMA 2018 | 178 | 76,521 | 4,135 | 6,475 | 320,855 | 1,613 | 634 | 85,335 | 0.01571 | GM14_hilltop_qc.png |
| GM15 | FEMA 2018 | 141 | 84,505 | 3,752 | 4,822 | 274,589 | 943 | 586 | 29,030 | 0.01637 | GM15_hilltop_qc.png |
| GM16 | FEMA 2018 | 141 | 90,876 | 3,116 | 6,454 | 251,494 | 1,604 | 557 | 8,413 | 0.01207 | GM16_hilltop_qc.png |
| GM17 | FEMA 2018 | 178 | 80,741 | 1,969 | 4,224 | 285,741 | 2,390 | 637 | 8,788 | 0.009473 | GM17_hilltop_qc.png |
| GM18 | FEMA 2018 | 89 | 116,860 | 3,436 | 2,655 | 300,228 | 3,816 | 595 | 32,085 | 0.0276 | GM18_hilltop_qc.png |
| GM19 | FEMA 2018 | 178 | 77,644 | 3,810 | 7,075 | 255,378 | 1,946 | 563 | 12,187 | 0.01185 | GM19_hilltop_qc.png |
| GM1_fema | FEMA 2018 | 178 | 67,262 | 2,714 | 8,554 | 273,650 | 4,085 | 600 | 14,695 | 0.01434 | GM1_fema_hilltop_qc.png |
| GM2 | NCALM 2003 | 35 | 285,672 | 6,657 | 2,530 | 99,134 | 1,093 | 218 | 0 | 0.01505 | GM2_hilltop_qc.png |
| GM20 | FEMA 2018 | 112 | 99,748 | 8,452 | 5,234 | 251,034 | 88 | 630 | 9,201 | 0.008055 | GM20_hilltop_qc.png |
| GM21 | FEMA 2018 | 282 | 50,185 | 3,147 | 7,692 | 289,041 | 1,874 | 639 | 42,977 | 0.01232 | GM21_hilltop_qc.png |
| GM22 | FEMA 2018 | 224 | 64,350 | 1,582 | 2,509 | 245,468 | 217 | 480 | 4,321 | 0.009136 | GM22_hilltop_qc.png |
| GM24 | FEMA 2018 | 224 | 62,263 | 2,963 | 6,482 | 264,879 | 407 | 554 | 8,828 | 0.004701 | GM24_hilltop_qc.png |
| GM25 | FEMA 2018 | 562 | 22,030 | 890 | 5,909 | 276,119 | 256 | 550 | 33,760 | 0.01386 | GM25_hilltop_qc.png |
| GM26 | FEMA 2018 | 141 | 87,248 | 3,060 | 3,448 | 274,409 | 390 | 582 | 8,801 | 0.008447 | GM26_hilltop_qc.png |
| GM27 | FEMA 2018 | 141 | 86,462 | 2,392 | 5,349 | 310,711 | 918 | 670 | 72,285 | 0.02019 | GM27_hilltop_qc.png |
| GM28 | FEMA 2018 | 178 | 79,724 | 2,327 | 6,318 | 302,734 | 2,090 | 574 | 78,500 | 0.01005 | GM28_hilltop_qc.png |
| GM29 | FEMA 2018 | 224 | 72,521 | 3,764 | 6,738 | 294,571 | 2,367 | 648 | 28,117 | 0.0101 | GM29_hilltop_qc.png |
| GM2_fema | FEMA 2018 | 112 | 113,462 | 5,861 | 5,425 | 231,015 | 3,244 | 544 | 3,694 | 0.01175 | GM2_fema_hilltop_qc.png |
| GM3 | NCALM 2003 | 22 | 324,955 | 12,530 | 1,500 | 137,094 | 444 | 262 | 0 | 0.01453 | GM3_hilltop_qc.png |
| GM30 | FEMA 2018 | 112 | 109,723 | 7,110 | 6,877 | 283,400 | 3,022 | 542 | 21,345 | 0.01319 | GM30_hilltop_qc.png |
| GM3_fema | FEMA 2018 | 56 | 168,282 | 8,795 | 6,283 | 234,680 | 1,056 | 503 | 3,062 | 0.009512 | GM3_fema_hilltop_qc.png |
| GM4 | NCALM 2003 | 56 | 207,521 | 2,878 | 3,347 | 94,852 | 2,198 | 203 | 0 | 0.01492 | GM4_hilltop_qc.png |
| GM4_fema | FEMA 2018 | 178 | 83,304 | 1,650 | 5,676 | 255,495 | 5,343 | 502 | 9,210 | 0.01247 | GM4_fema_hilltop_qc.png |
| GM5 | NCALM 2003 | 28 | 317,687 | 3,836 | 1,128 | 106,897 | 1,290 | 226 | 0 | 0.01937 | GM5_hilltop_qc.png |
| GM5_fema | FEMA 2018 | 71 | 153,950 | 6,464 | 3,310 | 242,713 | 3,912 | 581 | 4,813 | 0.01685 | GM5_fema_hilltop_qc.png |
| GM6 | NCALM 2003 | 45 | 229,550 | 7,434 | 1,987 | 107,734 | 2,180 | 214 | 0 | 0.01344 | GM6_hilltop_qc.png |
| GM6_fema | FEMA 2018 | 355 | 36,810 | 1,515 | 4,549 | 238,372 | 2,816 | 489 | 2,682 | 0.009044 | GM6_fema_hilltop_qc.png |
| GM7 | NCALM 2003 | 35 | 262,796 | 9,974 | 920 | 139,670 | 185 | 353 | 0 | 0.006672 | GM7_hilltop_qc.png |
| GM7_fema | FEMA 2018 | 562 | 21,449 | 1,275 | 3,585 | 267,677 | 55 | 611 | 5,223 | 0.004148 | GM7_fema_hilltop_qc.png |
| GM8 | FEMA 2018 | 355 | 36,161 | 1,109 | 4,963 | 291,600 | 66 | 599 | 16,608 | 0.009652 | GM8_hilltop_qc.png |
| GM9 | FEMA 2018 | 355 | 42,819 | 1,109 | 3,779 | 282,640 | 974 | 613 | 14,174 | 0.007219 | GM9_hilltop_qc.png |
| TTV1 | Marin 2019 | 35 | 195,972 | 12,523 | 966 | 247,489 | 3,932 | 449 | 57,180 | 0.02651 | TTV1_hilltop_qc.png |
| TTV11 | Marin 2019 | 45 | 168,680 | 13,046 | 3,192 | 243,983 | 4,226 | 593 | 91,681 | 0.0333 | TTV11_hilltop_qc.png |
| TTV15 | Marin 2019 | 18 | 274,983 | 13,382 | 579 | 251,955 | 5,043 | 566 | 79,551 | 0.05112 | TTV15_hilltop_qc.png |
| TTV2 | Marin 2019 | 56 | 168,414 | 7,251 | 1,995 | 216,849 | 1,812 | 477 | 43,674 | 0.01726 | TTV2_hilltop_qc.png |
| TTV3 | Marin 2019 | 18 | 294,758 | 6,949 | 122 | 226,792 | 2,024 | 460 | 26,136 | 0.0236 | TTV3_hilltop_qc.png |
| TTV4 | Marin 2019 | 11 | 346,458 | 16,971 | 86 | 242,892 | 2,320 | 533 | 68,786 | 0.08754 | TTV4_hilltop_qc.png |
| TTV5 | Marin 2019 | 45 | 166,418 | 5,458 | 3,203 | 216,281 | 2,747 | 472 | 49,360 | 0.03083 | TTV5_hilltop_qc.png |
| TTV6 | Marin 2019 | 22 | 270,845 | 17,503 | 612 | 210,300 | 7,564 | 469 | 34,800 | 0.03864 | TTV6_hilltop_qc.png |
| TTV7 | Marin 2019 | 56 | 148,663 | 8,724 | 2,983 | 251,798 | 2,678 | 490 | 103,622 | 0.02485 | TTV7_hilltop_qc.png |
| TTV8 | Marin 2019 | 89 | 112,671 | 8,466 | 5,287 | 296,430 | 4,319 | 658 | 140,599 | 0.01933 | TTV8_hilltop_qc.png |
| TTV9 | Marin 2019 | 178 | 71,711 | 2,768 | 3,291 | 296,931 | 1,379 | 606 | 157,172 | 0.01832 | TTV9_hilltop_qc.png |

Site colour scales of panel (c) (2nd-98th percentiles of the class (iii) curvature): GM -0.00466-0.05952; AP -0.01449-0.06149; TTV -0.006388-0.1326

144 PNG files under `reports/v2/figures/a18/hilltop_qc/`; over 5 MB: none; downscaled below 200 dpi: ['GM1_GM1_fema_hilltop_pair.png', 'GM2_GM2_fema_hilltop_pair.png', 'GM3_GM3_fema_hilltop_pair.png', 'GM4_GM4_fema_hilltop_pair.png', 'GM5_GM5_fema_hilltop_pair.png', 'GM6_GM6_fema_hilltop_pair.png', 'AP2_AP2_3dep_hilltop_pair.png', 'AP3_AP3_3dep_hilltop_pair.png', 'AP5_AP5_3dep_hilltop_pair.png', 'GM_hilltop_gallery_p1.png', 'AP_hilltop_gallery_p1.png', 'TTV_hilltop_gallery_p1.png'].

### C6. The regression gate (4.1): GM1 (m = 0.31) and AP2 (PAMAP, m = 0.47) against the v2.0.0 build

| spec | m | v2.0.0 A_crit | A_crit (union) | v2.0.0 MSE_P4 | MSE_P4 (union) | rel diff (A_crit K D mse) | max rel at common A_crit | P4 pass | v2.0.0 MSE_AD | MSE_AD | rel diff (K D mse r) | AD pass | runs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM1:v2m | 0.31 | 1241 | 1241 | 24.69038937 | 24.69038937 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 3.4e-15 | 0.0e+00 | yes | 48.96275516 | 48.96275516 | K 0.0e+00 D 0.0e+00 mse 0.0e+00 r 0.0e+00 | yes | 20261119T144900Z_a18_part_GM1_v2m / 20261119T144905Z_a18_ad_GM1_v2m |
| AP2:pamap | 0.47 | 3121 | 3121 | 22.46082884 | 22.46082884 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 3.8e-15 | 0.0e+00 | yes | 76.49290337 | 76.49290337 | K 0.0e+00 D 0.0e+00 mse 0.0e+00 r 0.0e+00 | yes | 20261119T144900Z_a18_part_AP2_pamap / 20261119T144905Z_a18_ad_AP2_pamap |

Tolerance 1e-09; gate PASS.

### C7. The cost gate (4.2): pilot timings and the expected core-hours per section, in the priority order of section 7

| stage | site (pilot tile) | s per evaluation (mean) | median | n timed | unique areas | pilot run |
|---|---|---|---|---|---|---|
| part | AP | 6.05 | 6.02 | 150 | 8,831 | 20261119T144912Z_a18_part_pilot_AP6_primary |
| part | GM | 5.78 | 6.08 | 150 | 6,253 | 20261119T144912Z_a18_part_pilot_GM11_primary |
| part | TTV | 5.41 | 4.97 | 150 | 9,170 | 20261119T144912Z_a18_part_pilot_TTV2_primary |
| ad | AP | 29.06 | 27.96 | 25 | - | 20261119T144900Z_a18_ad_pilot_AP6_primary |
| ad | TTV | 28.95 | 27.69 | 25 | - | 20261119T144900Z_a18_ad_pilot_TTV2_primary |
| ad | GM | 26.49 | 25.36 | 25 | - | 20261119T144905Z_a18_ad_pilot_GM11_primary |

| section | specs | expected core-hours | limits (2 x estimate, 30 min floor) |
|---|---|---|---|
| 5.1-5.3 primary groups (P4, AD, variants, held-out) | 54 | 461.7 | 939.0 |
| 5.7 and 5.6 TTV per-tile m | 11 | 90.2 | 181.5 |
| 5.4 trunk sensitivity | 9 | 77.6 | 158.2 |
| 5.5 continuity (NCALM, PAMAP, AP1_3dep; GM1-GM5 at 0.31) | 18 | 136.9 | 279.8 |
| 5.1 off-mesa group | 6 | 50.4 | 103.5 |

Assumptions: 3900 partitioned and 290 AD evaluations per run. Part B expected 816.7 core-hours; Part A used at the estimate 2.8; gate remaining 17.3; expected total 836.9 of 900 -> within budget.

### C8. Primary fits per site (5.1): P4 and AD at the site m

**GM primary (22 tiles; [T2] = Tier 2, outside the clean subset)**

| tile | m | A_crit (m2) | 1 % interval | 5 % interval | MSE_P4 | MSE_AD | R_MSE | K/K0 P4 | K/K0 AD | D/D0_A P4 | D/D0_A AD | D/D0_B P4 | D/D0_B AD | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM1_fema | 0.29 | 1213 | 1208-1254 | 814-1665 | 22.49 | 41.99 | 1.87 | 1.01 | 2.91 | 1.94 | 0.0248 | 1.82 | 0.0232 |  |
| GM2_fema | 0.29 | 1519 | 1514-1525 | 1172-1556 | 23.80 | 67.23 | 2.82 | 0.98 | 2.27 | 1.86 | 0.00818 | 1.95 | 0.00855 |  |
| GM3_fema | 0.29 | 698 | 631-757 | 359-1450 | 36.76 | 67.14 | 1.83 | 1.07 | 1.83 | 1.17 | 0.00168 | 1.14 | 0.00164 |  |
| GM4_fema | 0.29 | 2319 | 2286-2699 | 1758-2942 | 40.35 | 140.76 | 3.49 | 1.09 | 2.63 | 1.68 | 0.00547 | 1.81 | 0.00588 |  |
| GM5_fema | 0.29 | 2017 | 1179-2197 | 907-2344 | 32.85 | 83.08 | 2.53 | 1.03 | 2.79 | 2.40 | 0.0223 | 2.47 | 0.0229 |  |
| GM6_fema | 0.29 | 10533 | 9138-10546 | 8590-10587 | 51.18 | 188.56 | 3.68 | 0.95 | 1.86 | 2.73 | 0.00169 | 3.10 | 0.00192 |  |
| GM7_fema | 0.29 | 6042 | 5276-6136 | 3965-8364 | 21.49 | 32.75 | 1.52 | 1.05 | 3.10 | 1.32 | 0.0115 | 1.31 | 0.0114 |  |
| GM8 | 0.29 | 15244 | 14882-15898 | 3852-17282 | 30.83 | 96.66 | 3.14 | 1.09 | 2.27 | 2.76 | 0.00924 | 2.69 | 0.009 |  |
| GM9 | 0.29 | 5912 | 5874-6343 | 5256-10751 | 38.69 | 75.61 | 1.95 | 1.13 | 2.27 | 1.62 | 0.00504 | 1.57 | 0.00486 |  |
| GM11 | 0.29 | 3361 | 3335-3363 | 2806-3394 | 38.53 | 96.60 | 2.51 | 1.20 | 1.15 | 1.56 | 2.18e-08 | 1.44 | 2.03e-08 |  |
| GM12 [T2] | 0.29 | 4923 | 4838-4972 | 4060-7260 | 40.74 | 155.81 | 3.82 | 1.10 | 1.48 | 2.39 | 0.000888 | 2.15 | 0.000798 |  |
| GM13 | 0.29 | 3363 | 3241-3538 | 3081-4499 | 26.66 | 111.14 | 4.17 | 1.03 | 2.48 | 1.26 | 0.00354 | 1.30 | 0.00367 |  |
| GM15 | 0.29 | 1805 | 1614-1993 | 1340-26884 | 167.10 | 221.68 | 1.33 | 1.01 | 3.46 | 2.37 | 0.0302 | 1.98 | 0.0253 |  |
| GM16 | 0.29 | 5988 | 5978-7126 | 5256-7319 | 21.42 | 63.40 | 2.96 | 1.11 | 2.99 | 2.60 | 0.0203 | 3.06 | 0.0238 |  |
| GM17 | 0.29 | 651 | 619-682 | 425-13706 | 72.45 | 105.01 | 1.45 | 1.11 | 2.02 | 1.00 | 0.00336 | 0.78 | 0.00263 |  |
| GM20 | 0.29 | 7037 | 6478-7082 | 4914-9806 | 20.20 | 35.40 | 1.75 | 0.84 | 5.24 | 2.74 | 0.132 | 2.59 | 0.125 | P4 y_lo;  |
| GM21 | 0.29 | 2038 | 2034-2060 | 1712-2435 | 78.55 | 230.34 | 2.93 | 1.12 | 1.24 | 1.66 | 3.35e-08 | 1.56 | 3.14e-08 |  |
| GM22 | 0.29 | 5818 | 3292-6202 | 1469-7730 | 25.37 | 96.47 | 3.80 | 1.04 | 2.93 | 1.79 | 0.0208 | 1.52 | 0.0177 |  |
| GM24 | 0.29 | 4368 | 3680-5872 | 1683-5959 | 30.01 | 50.73 | 1.69 | 1.01 | 3.07 | 1.38 | 0.00793 | 1.36 | 0.00783 |  |
| GM26 [T2] | 0.29 | 4364 | 4285-9521 | 2349-10550 | 27.81 | 50.57 | 1.82 | 1.00 | 2.17 | 1.53 | 0.00448 | 1.40 | 0.00411 |  |
| GM27 | 0.29 | 1923 | 1764-4388 | 1429-4800 | 90.78 | 180.63 | 1.99 | 0.98 | 7.00 | 2.41 | 0.654 | 2.15 | 0.583 |  |
| GM29 [T2] | 0.29 | 1323 | 1316-1339 | 777-4088 | 66.79 | 122.74 | 1.84 | 1.08 | 1.38 | 1.62 | 0.000283 | 1.70 | 0.000297 |  |

**AP primary (21 tiles; [T2] = Tier 2, outside the clean subset)**

| tile | m | A_crit (m2) | 1 % interval | 5 % interval | MSE_P4 | MSE_AD | R_MSE | K/K0 P4 | K/K0 AD | D/D0_A P4 | D/D0_A AD | D/D0_B P4 | D/D0_B AD | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP2_3dep | 0.47 | 7518 | 7369-7524 | 5275-7788 | 21.84 | 86.34 | 3.95 | 0.99 | 9.70 | - | - | - | - | no D0;  |
| AP3_3dep | 0.47 | 14529 | 14290-14576 | 13763-15622 | 38.05 | 168.95 | 4.44 | 1.16 | 10.90 | 5.35 | 0.176 | 4.44 | 0.146 |  |
| AP4_3dep | 0.47 | 5144 | 5008-5161 | 4734-5269 | 28.34 | 40.12 | 1.42 | 1.02 | 10.76 | 3.48 | 0.235 | 2.47 | 0.167 |  |
| AP5_3dep | 0.47 | 2629 | 2588-2670 | 1406-2841 | 34.61 | 81.13 | 2.34 | 0.97 | 9.36 | 2.85 | 0.125 | 2.25 | 0.099 |  |
| AP6 | 0.47 | 11491 | 11262-11601 | 8057-11799 | 34.00 | 118.30 | 3.48 | 1.04 | 9.19 | 2.86 | 0.0945 | 2.36 | 0.0779 |  |
| AP7 | 0.47 | 5594 | 5499-8811 | 4305-10898 | 41.59 | 123.32 | 2.97 | 1.01 | 9.03 | 3.47 | 0.152 | 3.15 | 0.138 |  |
| AP8 | 0.47 | 9172 | 8794-9442 | 7999-9578 | 40.44 | 149.17 | 3.69 | 1.03 | 7.37 | 4.21 | 0.0397 | 3.30 | 0.0311 |  |
| AP9 | 0.47 | 5703 | 5324-5762 | 3714-8188 | 27.15 | 45.31 | 1.67 | 0.88 | 11.25 | - | - | - | - | no D0;  |
| AP10 | 0.47 | 9381 | 9244-9503 | 8866-11064 | 28.99 | 61.08 | 2.11 | 1.09 | 10.73 | - | - | - | - | no D0;  |
| AP11 | 0.47 | 3490 | 3379-8940 | 2503-10519 | 32.05 | 59.00 | 1.84 | 0.91 | 10.11 | - | - | - | - | no D0;  |
| AP13 | 0.47 | 6085 | 6038-6213 | 4358-7053 | 26.08 | 92.87 | 3.56 | 1.07 | 9.32 | 3.94 | 0.174 | 2.33 | 0.103 |  |
| AP14 | 0.47 | 7631 | 7310-7837 | 2427-8347 | 23.00 | 61.29 | 2.67 | 0.99 | 9.98 | - | - | - | - | no D0;  |
| AP18 | 0.47 | 14274 | 14123-18419 | 11593-18975 | 19.34 | 51.40 | 2.66 | 0.90 | 10.70 | - | - | - | - | no D0;  |
| AP19 | 0.47 | 3402 | 3313-3550 | 2628-7176 | 27.07 | 37.75 | 1.39 | 0.96 | 11.02 | 2.84 | 0.24 | 2.20 | 0.187 |  |
| AP20 | 0.47 | 2690 | 2665-2752 | 2588-3519 | 36.84 | 64.06 | 1.74 | 0.90 | 10.07 | 3.13 | 0.142 | 2.35 | 0.106 |  |
| AP23 | 0.47 | 6014 | 5964-6130 | 5644-6548 | 31.20 | 162.26 | 5.20 | 1.13 | 6.21 | 4.56 | 0.0243 | 3.71 | 0.0198 |  |
| AP24 | 0.47 | 4516 | 4502-4577 | 4349-4725 | 24.75 | 35.15 | 1.42 | 0.89 | 11.76 | 3.42 | 0.249 | 2.46 | 0.179 |  |
| AP25 [T2] | 0.47 | 9624 | 9505-9673 | 9013-9879 | 48.17 | 111.62 | 2.32 | 0.97 | 10.64 | 5.30 | 0.215 | 4.47 | 0.181 |  |
| AP27 | 0.47 | 5110 | 5075-5174 | 4105-6587 | 25.51 | 58.26 | 2.28 | 1.00 | 9.07 | - | - | - | - | no D0;  |
| AP28 | 0.47 | 4522 | 4476-4535 | 4089-5225 | 34.41 | 58.06 | 1.69 | 0.94 | 12.51 | - | - | - | - | no D0;  |
| AP29 | 0.47 | 16545 | 16412-16636 | 16221-16677 | 20.84 | 49.32 | 2.37 | 1.14 | 10.10 | 3.69 | 0.137 | 3.47 | 0.129 |  |

**TTV primary (11 tiles; [T2] = Tier 2, outside the clean subset)**

| tile | m | A_crit (m2) | 1 % interval | 5 % interval | MSE_P4 | MSE_AD | R_MSE | K/K0 P4 | K/K0 AD | D/D0_A P4 | D/D0_A AD | D/D0_B P4 | D/D0_B AD | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 0.33 | 59499 | 59066-60312 | 48902-62092 | 82.73 | 214.19 | 2.59 | 1.10 | 9.23 | 15.11 | 0.551 | 15.11 | 0.551 |  |
| TTV2 | 0.33 | 9131 | 8399-9159 | 8279-9191 | 59.79 | 209.66 | 3.51 | 1.04 | 4.82 | 3.72 | 0.053 | 3.53 | 0.0502 |  |
| TTV3 | 0.33 | 89124 | 89070-89144 | 88198-95118 | 87.40 | 326.85 | 3.74 | 1.42 | 1.96 | - | - | - | - | no D0; P4 y_lo;  |
| TTV4 | 0.33 | 40309 | 38925-40734 | 38153-44997 | 96.82 | 293.97 | 3.04 | 0.96 | 10.09 | - | - | - | - | no D0; P4 y_lo;  |
| TTV5 | 0.33 | 58612 | 58350-60066 | 58248-61531 | 137.88 | 842.95 | 6.11 | 1.22 | 2.78 | 20.02 | 0.0208 | 18.52 | 0.0193 | P4 y_lo;  |
| TTV6 | 0.33 | 10973 | 10344-11683 | 9403-15186 | 94.27 | 204.34 | 2.17 | 0.76 | 7.60 | 16.69 | 0.56 | 16.69 | 0.56 |  |
| TTV7 | 0.33 | 35966 | 35470-36940 | 32608-39469 | 79.24 | 134.26 | 1.69 | 0.75 | 20.90 | 16.14 | 7.65 | 15.09 | 7.15 | P4 y_lo;  |
| TTV8 | 0.33 | 38544 | 28341-40094 | 21256-45698 | 114.67 | 272.96 | 2.38 | 0.73 | 13.20 | 11.44 | 12.5 | 10.17 | 11.2 | AD y_lo |
| TTV9 | 0.33 | 36561 | 33976-39662 | 31421-64483 | 115.18 | 284.91 | 2.47 | 0.73 | 11.35 | 10.42 | 0.467 | 7.00 | 0.314 |  |
| TTV11 [T2] | 0.33 | 30362 | 16982-31139 | 15044-35960 | 241.77 | 164.20 | 0.68 | 0.69 | 12.53 | 17.77 | 13.9 | 16.06 | 12.6 |  |
| TTV15 | 0.33 | 26819 | 26314-27056 | 18316-28690 | 103.42 | 327.53 | 3.17 | 1.11 | 15.39 | 18.98 | 33.2 | 18.98 | 33.2 | AD y_lo |

**GM off-mesa (report only)**

| tile | m | A_crit (m2) | 1 % interval | 5 % interval | MSE_P4 | MSE_AD | R_MSE | K/K0 P4 | K/K0 AD | D/D0_A P4 | D/D0_A AD | D/D0_B P4 | D/D0_B AD | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM14 | 0.29 | 3482 | 3297-3684 | 2876-3956 | 111.35 | 146.28 | 1.31 | 1.00 | 6.59 | 3.15 | 0.371 | 2.52 | 0.297 |  |
| GM18 | 0.29 | 1859 | 1838-1948 | 1198-4329 | 32.89 | 115.82 | 3.52 | 1.05 | 1.31 | 4.58 | 0.000973 | 3.78 | 0.000803 |  |
| GM19 | 0.29 | 14154 | 10583-18606 | 8969-22833 | 91.59 | 145.97 | 1.59 | 0.91 | 5.52 | 4.03 | 0.163 | 4.17 | 0.169 | P4 y_lo;  |
| GM25 | 0.29 | 13881 | 8221-16427 | 1387-27491 | 79.98 | 207.55 | 2.59 | 0.94 | 13.12 | 3.11 | 1.89 | 2.78 | 1.69 |  |
| GM28 | 0.29 | 11271 | 11008-11362 | 9332-12134 | 84.25 | 311.05 | 3.69 | 1.21 | 3.03 | 2.35 | 0.0115 | 2.00 | 0.00976 |  |
| GM30 | 0.29 | 10691 | 10573-10727 | 9441-11755 | 61.09 | 137.23 | 2.25 | 1.05 | 2.64 | 4.14 | 0.0286 | 3.94 | 0.0273 |  |

**Continuity groups (report only): NCALM GM1-GM7 at m = 0.29; PAMAP AP1-AP5 at m = 0.47; AP1_3dep**

| tile | m | A_crit (m2) | 1 % interval | 5 % interval | MSE_P4 | MSE_AD | R_MSE | K/K0 P4 | K/K0 AD | D/D0_A P4 | D/D0_A AD | D/D0_B P4 | D/D0_B AD | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM1 | 0.29 | 1241 | 1133-1268 | 1041-1853 | 24.38 | 48.92 | 2.01 | 1.00 | 2.50 | 2.15 | 0.0209 | 2.07 | 0.0201 |  |
| GM2 | 0.29 | 3517 | 3353-3536 | 3334-3576 | 25.99 | 76.19 | 2.93 | 1.01 | 1.95 | 2.80 | 0.00519 | 2.69 | 0.005 |  |
| GM3 | 0.29 | 860 | 736-929 | 519-1690 | 32.80 | 59.63 | 1.82 | 1.06 | 1.94 | 1.88 | 0.00395 | 1.88 | 0.00395 | AD y_hi |
| GM4 | 0.29 | 3855 | 3692-3893 | 2604-4437 | 42.09 | 168.25 | 4.00 | 1.06 | 2.26 | 2.18 | 0.00406 | 2.28 | 0.00424 | AD y_hi |
| GM5 | 0.29 | 3108 | 1551-3124 | 1259-3692 | 33.13 | 92.68 | 2.80 | 1.06 | 2.48 | 2.79 | 0.0193 | 2.83 | 0.0196 |  |
| GM6 | 0.29 | 3588 | 3529-3592 | 3434-4463 | 45.41 | 195.57 | 4.31 | 1.02 | 1.46 | 2.52 | 0.00077 | 2.67 | 0.000817 |  |
| GM7 | 0.29 | 4302 | 4047-8507 | 2965-12695 | 22.31 | 41.76 | 1.87 | 1.05 | 2.97 | 2.13 | 0.0163 | 2.13 | 0.0163 |  |
| AP1 | 0.47 | 6889 | 6532-6958 | 5623-7905 | 22.38 | 59.02 | 2.64 | 0.90 | 9.02 | 4.74 | 0.152 | 2.43 | 0.0781 |  |
| AP2 | 0.47 | 3121 | 2976-3305 | 2637-3485 | 22.46 | 76.49 | 3.41 | 0.97 | 8.98 | 5.22 | 0.172 | 4.12 | 0.136 |  |
| AP3 | 0.47 | 4742 | 4622-5676 | 4297-10318 | 42.12 | 152.81 | 3.63 | 1.01 | 9.61 | 8.31 | 0.234 | 8.31 | 0.234 |  |
| AP4 | 0.47 | 7232 | 6998-7894 | 6292-9029 | 35.16 | 45.72 | 1.30 | 0.98 | 10.22 | 5.44 | 0.281 | 2.56 | 0.132 |  |
| AP5 | 0.47 | 1887 | 1805-1981 | 1763-2244 | 27.87 | 69.81 | 2.50 | 1.00 | 8.62 | 4.31 | 0.196 | 3.64 | 0.166 |  |
| AP1_3dep | 0.47 | 8405 | 8053-9046 | 7543-10896 | 21.94 | 65.09 | 2.97 | 0.93 | 9.87 | 3.24 | 0.129 | 2.68 | 0.107 |  |

### C9. In-sample MSE (m2) of every variant at the site m (5.2): P1 [A_crit], P2 [K, A_crit], P3 [D, A_crit], P4 [K, D, A_crit]; A1 [D], A2 [K], AD [K, D]; _A / _B = the D0 end

**GM**

| tile | P1_A | P1_B | P2_A | P2_B | P3 | P4 | A1 | A2_A | A2_B | AD | E1_A | E1_B | E2 P2_A | E2 P2_B | E2 P3 | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM1_fema | 32.21 | 30.61 | 30.06 | 28.81 | 22.51 | 22.49 | 220.97 | 167.59 | 169.72 | 41.99 | yes | yes | yes | yes | yes | yes |
| GM2_fema | 34.85 | 35.55 | 34.78 | 35.55 | 23.91 | 23.80 | 143.44 | 310.54 | 308.08 | 67.23 | yes | yes | yes | yes | yes | yes |
| GM3_fema | 43.64 | 43.06 | 37.29 | 37.21 | 39.04 | 36.76 | 172.41 | 117.27 | 117.40 | 67.14 | yes | yes | yes | yes | yes | yes |
| GM4_fema | 83.78 | 91.34 | 60.15 | 63.59 | 45.76 | 40.35 | 674.65 | 570.42 | 563.94 | 140.76 | yes | yes | yes | yes | yes | yes |
| GM5_fema | 66.75 | 68.12 | 57.40 | 58.28 | 33.55 | 32.85 | 253.31 | 269.59 | 267.77 | 83.08 | yes | yes | yes | yes | yes | yes |
| GM6_fema | 100.60 | 115.04 | 79.11 | 94.51 | 52.10 | 51.18 | 308.80 | 590.04 | 583.69 | 188.56 | yes | yes | yes | yes | yes | yes |
| GM7_fema | 26.17 | 25.85 | 24.06 | 23.90 | 21.99 | 21.49 | 185.49 | 176.07 | 176.23 | 32.75 | yes | yes | yes | yes | yes | yes |
| GM8 | 73.25 | 71.73 | 64.39 | 62.72 | 33.05 | 30.83 | 182.65 | 258.71 | 259.43 | 96.66 | yes | yes | yes | yes | yes | yes |
| GM9 | 65.05 | 63.85 | 60.15 | 57.90 | 42.77 | 38.69 | 172.16 | 353.58 | 355.51 | 75.61 | yes | yes | yes | yes | yes | yes |
| GM11 | 49.47 | 47.88 | 46.34 | 43.89 | 45.17 | 38.53 | 156.16 | 215.26 | 215.88 | 96.60 | yes | yes | yes | yes | yes | yes |
| GM12 [T2] | 119.57 | 113.91 | 117.97 | 112.88 | 43.67 | 40.74 | 213.96 | 455.84 | 458.64 | 155.81 | yes | yes | yes | yes | yes | yes |
| GM13 | 33.04 | 34.52 | 32.29 | 32.90 | 26.87 | 26.66 | 291.27 | 197.15 | 196.73 | 111.14 | yes | yes | yes | yes | yes | yes |
| GM15 | 215.25 | 200.78 | 207.64 | 194.73 | 167.15 | 167.10 | 485.16 | 300.37 | 307.07 | 221.68 | yes | yes | yes | yes | yes | yes |
| GM16 | 49.96 | 56.25 | 40.92 | 42.72 | 24.95 | 21.42 | 268.43 | 167.25 | 161.68 | 63.40 | yes | yes | yes | yes | yes | yes |
| GM17 | 81.85 | 79.84 | 72.45 | 77.52 | 79.09 | 72.45 | 237.43 | 271.20 | 276.34 | 105.01 | yes | yes | yes | yes | yes | yes |
| GM20 | 78.34 | 78.69 | 78.34 | 78.67 | 38.04 | 20.20 | 159.84 | 57.89 | 58.87 | 35.40 | no | no | no | no | no | yes |
| GM21 | 121.59 | 116.99 | 108.88 | 105.63 | 83.95 | 78.55 | 385.46 | 357.52 | 361.61 | 230.34 | yes | yes | yes | yes | yes | yes |
| GM22 | 41.76 | 34.48 | 39.98 | 33.76 | 25.91 | 25.37 | 202.03 | 168.75 | 171.72 | 96.47 | yes | yes | yes | yes | yes | yes |
| GM24 | 32.36 | 32.26 | 31.91 | 31.73 | 30.02 | 30.01 | 248.50 | 164.83 | 165.04 | 50.73 | yes | yes | yes | yes | yes | yes |
| GM26 [T2] | 30.87 | 30.22 | 30.83 | 30.14 | 27.81 | 27.81 | 92.37 | 134.98 | 137.01 | 50.57 | yes | yes | yes | yes | yes | yes |
| GM27 | 185.00 | 173.32 | 178.40 | 153.60 | 91.11 | 90.78 | 291.22 | 183.66 | 185.28 | 180.63 | no | yes | yes | yes | yes | yes |
| GM29 [T2] | 76.41 | 77.19 | 72.57 | 74.62 | 69.06 | 66.79 | 207.14 | 322.58 | 320.44 | 122.74 | yes | yes | yes | yes | yes | yes |

**AP**

| tile | P1_A | P1_B | P2_A | P2_B | P3 | P4 | A1 | A2_A | A2_B | AD | E1_A | E1_B | E2 P2_A | E2 P2_B | E2 P3 | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP2_3dep | - | - | - | - | 21.87 | 21.84 | 3081.22 | - | - | 86.34 | - | - | - | - | yes | yes |
| AP3_3dep | 213.74 | 175.55 | 129.11 | 115.17 | 40.27 | 38.05 | 3414.77 | 226.76 | 239.45 | 168.95 | yes | yes | yes | yes | yes | yes |
| AP4_3dep | 150.99 | 84.99 | 78.46 | 59.89 | 28.39 | 28.34 | 3723.07 | 94.67 | 118.09 | 40.12 | no | yes | no | no | yes | yes |
| AP5_3dep | 107.24 | 72.75 | 65.01 | 50.32 | 34.83 | 34.61 | 2652.61 | 134.50 | 144.57 | 81.13 | yes | yes | yes | yes | yes | yes |
| AP6 | 83.69 | 67.39 | 68.28 | 61.28 | 34.19 | 34.00 | 3188.32 | 218.74 | 231.26 | 118.30 | yes | yes | yes | yes | yes | yes |
| AP7 | 124.70 | 111.89 | 88.78 | 81.28 | 41.59 | 41.59 | 2698.67 | 196.08 | 202.93 | 123.32 | yes | yes | yes | yes | yes | yes |
| AP8 | 138.20 | 107.21 | 83.44 | 74.66 | 40.62 | 40.44 | 2754.77 | 271.62 | 283.43 | 149.17 | yes | yes | yes | yes | yes | yes |
| AP9 | - | - | - | - | 30.07 | 27.15 | 4174.04 | - | - | 45.31 | - | - | - | - | yes | yes |
| AP10 | - | - | - | - | 29.78 | 28.99 | 3345.47 | - | - | 61.08 | - | - | - | - | yes | yes |
| AP11 | - | - | - | - | 34.20 | 32.05 | 5703.66 | - | - | 59.00 | - | - | - | - | yes | yes |
| AP13 | 153.49 | 71.94 | 75.80 | 50.89 | 26.40 | 26.08 | 2642.05 | 128.84 | 153.98 | 92.87 | no | yes | yes | yes | yes | yes |
| AP14 | - | - | - | - | 23.01 | 23.00 | 2579.72 | - | - | 61.29 | - | - | - | - | yes | yes |
| AP18 | - | - | - | - | 20.50 | 19.34 | 3223.26 | - | - | 51.40 | - | - | - | - | yes | yes |
| AP19 | 174.78 | 102.86 | 88.76 | 66.19 | 27.58 | 27.07 | 4942.86 | 92.58 | 110.20 | 37.75 | no | yes | no | no | yes | yes |
| AP20 | 144.20 | 82.38 | 77.55 | 57.56 | 39.94 | 36.84 | 3316.64 | 118.00 | 133.81 | 64.06 | no | yes | no | yes | yes | yes |
| AP23 | 147.53 | 122.33 | 85.46 | 78.17 | 33.50 | 31.20 | 2622.18 | 314.74 | 328.21 | 162.26 | yes | yes | yes | yes | yes | yes |
| AP24 | 137.79 | 77.43 | 91.25 | 63.66 | 27.20 | 24.75 | 4208.30 | 70.22 | 85.97 | 35.15 | no | yes | no | no | yes | yes |
| AP25 [T2] | 201.80 | 168.32 | 124.96 | 115.72 | 48.36 | 48.17 | 3457.19 | 156.64 | 165.12 | 111.62 | no | no | no | no | yes | yes |
| AP27 | - | - | - | - | 25.51 | 25.51 | 3479.34 | - | - | 58.26 | - | - | - | - | yes | yes |
| AP28 | - | - | - | - | 35.92 | 34.41 | 4560.03 | - | - | 58.06 | - | - | - | - | yes | yes |
| AP29 | 109.79 | 100.59 | 73.79 | 71.78 | 21.74 | 20.84 | 3139.52 | 119.91 | 123.96 | 49.32 | yes | yes | no | no | yes | yes |

**TTV**

| tile | P1_A | P1_B | P2_A | P2_B | P3 | P4 | A1 | A2_A | A2_B | AD | E1_A | E1_B | E2 P2_A | E2 P2_B | E2 P3 | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 633.45 | 633.45 | 472.70 | 472.70 | 85.30 | 82.73 | 3141.27 | 228.83 | 228.83 | 214.19 | no | no | no | no | yes | yes |
| TTV2 | 186.62 | 172.71 | 149.35 | 141.63 | 61.83 | 59.79 | 1466.33 | 518.16 | 524.45 | 209.66 | yes | yes | yes | yes | yes | yes |
| TTV3 | - | - | - | - | 109.35 | 87.40 | 590.34 | - | - | 326.85 | - | - | - | - | yes | yes |
| TTV4 | - | - | - | - | 100.15 | 96.82 | 1700.18 | - | - | 293.97 | - | - | - | - | yes | yes |
| TTV5 | 764.07 | 753.80 | 710.21 | 709.07 | 148.99 | 137.88 | 1240.79 | 1341.32 | 1354.79 | 842.95 | yes | yes | yes | yes | yes | yes |
| TTV6 | 981.16 | 981.16 | 485.65 | 485.65 | 166.04 | 94.27 | 1453.79 | 223.85 | 223.85 | 204.34 | no | no | no | no | yes | yes |
| TTV7 | 752.77 | 707.13 | 526.59 | 508.50 | 132.65 | 79.24 | 2634.03 | 269.90 | 265.38 | 134.26 | no | no | no | no | yes | yes |
| TTV8 | 443.74 | 397.76 | 342.41 | 327.27 | 165.18 | 114.67 | 1194.68 | 337.38 | 346.46 | 272.96 | no | no | no | no | yes | yes |
| TTV9 | 595.67 | 480.33 | 569.51 | 480.28 | 328.78 | 115.18 | 3738.96 | 343.68 | 412.19 | 284.91 | no | no | no | no | no | yes |
| TTV11 [T2] | 855.42 | 806.34 | 593.25 | 575.35 | 335.16 | 241.77 | 1045.96 | 304.35 | 297.84 | 164.20 | no | no | no | no | no | no |
| TTV15 | 744.07 | 744.07 | 365.69 | 365.69 | 109.08 | 103.42 | 1963.28 | 589.34 | 589.34 | 327.53 | no | no | no | no | yes | yes |

### C9b. Parameters of every variant's optimum (primary groups)

| tile | variant | A_crit | K | D | MSE | where / bounds |
|---|---|---|---|---|---|---|
| GM1_fema | P1_A | 298 | 6.166e-05 | 1.025e-02 | 32.21 | fixed |
| GM1_fema | P1_B | 346 | 6.166e-05 | 1.093e-02 | 30.61 | fixed |
| GM1_fema | P2_A | 305 | 6.349e-05 | 1.025e-02 | 30.06 | interior |
| GM1_fema | P2_B | 428 | 6.401e-05 | 1.093e-02 | 28.81 | interior |
| GM1_fema | P3 | 1213 | 6.166e-05 | 2.009e-02 | 22.51 | interior |
| GM1_fema | P4 | 1213 | 6.201e-05 | 1.986e-02 | 22.49 | interior |
| GM1_fema | A1 | - | 6.166e-05 | 4.457e-10 | 220.97 | D at bound |
| GM1_fema | A2_A | - | 6.276e-04 | 1.025e-02 | 167.59 |  |
| GM1_fema | A2_B | - | 6.430e-04 | 1.093e-02 | 169.72 |  |
| GM1_fema | AD | - | 1.795e-04 | 2.539e-04 | 41.99 |  |
| GM2_fema | P1_A | 360 | 7.276e-05 | 1.251e-02 | 34.85 | fixed |
| GM2_fema | P1_B | 354 | 7.276e-05 | 1.197e-02 | 35.55 | fixed |
| GM2_fema | P2_A | 360 | 7.227e-05 | 1.251e-02 | 34.78 | interior |
| GM2_fema | P2_B | 354 | 7.270e-05 | 1.197e-02 | 35.55 | interior |
| GM2_fema | P3 | 1519 | 7.276e-05 | 2.277e-02 | 23.91 | interior |
| GM2_fema | P4 | 1519 | 7.155e-05 | 2.328e-02 | 23.80 | interior |
| GM2_fema | A1 | - | 7.276e-05 | 4.292e-10 | 143.44 | D at bound |
| GM2_fema | A2_A | - | 6.431e-04 | 1.251e-02 | 310.54 |  |
| GM2_fema | A2_B | - | 6.332e-04 | 1.197e-02 | 308.08 |  |
| GM2_fema | AD | - | 1.652e-04 | 1.023e-04 | 67.23 |  |
| GM3_fema | P1_A | 459 | 8.045e-05 | 1.545e-02 | 43.64 | fixed |
| GM3_fema | P1_B | 459 | 8.045e-05 | 1.583e-02 | 43.06 | fixed |
| GM3_fema | P2_A | 565 | 8.732e-05 | 1.545e-02 | 37.29 | interior |
| GM3_fema | P2_B | 565 | 8.687e-05 | 1.583e-02 | 37.21 | interior |
| GM3_fema | P3 | 840 | 8.045e-05 | 2.221e-02 | 39.04 | interior |
| GM3_fema | P4 | 698 | 8.597e-05 | 1.800e-02 | 36.76 | interior |
| GM3_fema | A1 | - | 8.045e-05 | 4.746e-10 | 172.41 | D at bound |
| GM3_fema | A2_A | - | 7.931e-04 | 1.545e-02 | 117.27 |  |
| GM3_fema | A2_B | - | 7.998e-04 | 1.583e-02 | 117.40 |  |
| GM3_fema | AD | - | 1.475e-04 | 2.593e-05 | 67.14 |  |
| GM4_fema | P1_A | 525 | 4.540e-05 | 1.178e-02 | 83.78 | fixed |
| GM4_fema | P1_B | 476 | 4.540e-05 | 1.097e-02 | 91.34 | fixed |
| GM4_fema | P2_A | 682 | 5.058e-05 | 1.178e-02 | 60.15 | interior |
| GM4_fema | P2_B | 682 | 5.185e-05 | 1.097e-02 | 63.59 | interior |
| GM4_fema | P3 | 2693 | 4.540e-05 | 2.271e-02 | 45.76 | interior |
| GM4_fema | P4 | 2319 | 4.945e-05 | 1.986e-02 | 40.35 | interior |
| GM4_fema | A1 | - | 4.540e-05 | 4.020e-10 | 674.65 | D at bound |
| GM4_fema | A2_A | - | 5.329e-04 | 1.178e-02 | 570.42 |  |
| GM4_fema | A2_B | - | 5.195e-04 | 1.097e-02 | 563.94 |  |
| GM4_fema | AD | - | 1.192e-04 | 6.446e-05 | 140.76 |  |
| GM5_fema | P1_A | 267 | 6.490e-05 | 8.725e-03 | 66.75 | fixed |
| GM5_fema | P1_B | 214 | 6.490e-05 | 8.471e-03 | 68.12 | fixed |
| GM5_fema | P2_A | 270 | 6.879e-05 | 8.725e-03 | 57.40 | interior |
| GM5_fema | P2_B | 267 | 6.902e-05 | 8.471e-03 | 58.28 | interior |
| GM5_fema | P3 | 1906 | 6.490e-05 | 2.195e-02 | 33.55 | interior |
| GM5_fema | P4 | 2017 | 6.698e-05 | 2.091e-02 | 32.85 | interior |
| GM5_fema | A1 | - | 6.490e-05 | 4.690e-10 | 253.31 | D at bound |
| GM5_fema | A2_A | - | 6.238e-04 | 8.725e-03 | 269.59 |  |
| GM5_fema | A2_B | - | 6.173e-04 | 8.471e-03 | 267.77 |  |
| GM5_fema | AD | - | 1.809e-04 | 1.942e-04 | 83.08 |  |
| GM6_fema | P1_A | 927 | 6.847e-05 | 1.625e-02 | 100.60 | fixed |
| GM6_fema | P1_B | 775 | 6.847e-05 | 1.432e-02 | 115.04 | fixed |
| GM6_fema | P2_A | 1914 | 8.378e-05 | 1.625e-02 | 79.11 | interior |
| GM6_fema | P2_B | 1762 | 8.911e-05 | 1.432e-02 | 94.51 | interior |
| GM6_fema | P3 | 10533 | 6.847e-05 | 4.228e-02 | 52.10 | interior |
| GM6_fema | P4 | 10533 | 6.513e-05 | 4.431e-02 | 51.18 | interior |
| GM6_fema | A1 | - | 6.847e-05 | 4.039e-10 | 308.80 | D at bound |
| GM6_fema | A2_A | - | 7.189e-04 | 1.625e-02 | 590.04 |  |
| GM6_fema | A2_B | - | 6.896e-04 | 1.432e-02 | 583.69 |  |
| GM6_fema | AD | - | 1.275e-04 | 2.744e-05 | 188.56 |  |
| GM7_fema | P1_A | 1452 | 8.255e-05 | 3.544e-02 | 26.17 | fixed |
| GM7_fema | P1_B | 1565 | 8.255e-05 | 3.574e-02 | 25.85 | fixed |
| GM7_fema | P2_A | 1866 | 8.837e-05 | 3.544e-02 | 24.06 | interior |
| GM7_fema | P2_B | 1866 | 8.805e-05 | 3.574e-02 | 23.90 | interior |
| GM7_fema | P3 | 6032 | 8.255e-05 | 4.927e-02 | 21.99 | interior |
| GM7_fema | P4 | 6042 | 8.677e-05 | 4.669e-02 | 21.49 | interior |
| GM7_fema | A1 | - | 8.255e-05 | 4.870e-10 | 185.49 | D at bound |
| GM7_fema | A2_A | - | 1.197e-03 | 3.544e-02 | 176.07 |  |
| GM7_fema | A2_B | - | 1.202e-03 | 3.574e-02 | 176.23 |  |
| GM7_fema | AD | - | 2.558e-04 | 4.058e-04 | 32.75 |  |
| GM8 | P1_A | 676 | 8.457e-05 | 1.523e-02 | 73.25 | fixed |
| GM8 | P1_B | 676 | 8.457e-05 | 1.564e-02 | 71.73 | fixed |
| GM8 | P2_A | 919 | 9.434e-05 | 1.523e-02 | 64.39 | interior |
| GM8 | P2_B | 949 | 9.404e-05 | 1.564e-02 | 62.72 | interior |
| GM8 | P3 | 14911 | 8.457e-05 | 4.537e-02 | 33.05 | interior |
| GM8 | P4 | 15244 | 9.260e-05 | 4.201e-02 | 30.83 | interior |
| GM8 | A1 | - | 8.457e-05 | 4.072e-10 | 182.65 | D at bound |
| GM8 | A2_A | - | 9.501e-04 | 1.523e-02 | 258.71 |  |
| GM8 | A2_B | - | 9.608e-04 | 1.564e-02 | 259.43 |  |
| GM8 | AD | - | 1.921e-04 | 1.408e-04 | 96.66 |  |
| GM9 | P1_A | 983 | 8.281e-05 | 2.036e-02 | 65.05 | fixed |
| GM9 | P1_B | 1044 | 8.281e-05 | 2.110e-02 | 63.85 | fixed |
| GM9 | P2_A | 1490 | 9.403e-05 | 2.036e-02 | 60.15 | interior |
| GM9 | P2_B | 1616 | 9.490e-05 | 2.110e-02 | 57.90 | interior |
| GM9 | P3 | 5912 | 8.281e-05 | 3.678e-02 | 42.77 | interior |
| GM9 | P4 | 5912 | 9.368e-05 | 3.308e-02 | 38.69 | interior |
| GM9 | A1 | - | 8.281e-05 | 4.885e-10 | 172.16 | D at bound |
| GM9 | A2_A | - | 9.689e-04 | 2.036e-02 | 353.58 |  |
| GM9 | A2_B | - | 9.825e-04 | 2.110e-02 | 355.51 |  |
| GM9 | AD | - | 1.880e-04 | 1.026e-04 | 75.61 |  |
| GM11 | P1_A | 665 | 8.016e-05 | 1.831e-02 | 49.47 | fixed |
| GM11 | P1_B | 781 | 8.016e-05 | 1.974e-02 | 47.88 | fixed |
| GM11 | P2_A | 781 | 8.483e-05 | 1.831e-02 | 46.34 | interior |
| GM11 | P2_B | 961 | 8.881e-05 | 1.974e-02 | 43.89 | interior |
| GM11 | P3 | 1075 | 8.016e-05 | 2.546e-02 | 45.17 | interior |
| GM11 | P4 | 3361 | 9.595e-05 | 2.848e-02 | 38.53 | interior |
| GM11 | A1 | - | 8.016e-05 | 4.049e-10 | 156.16 | D at bound |
| GM11 | A2_A | - | 8.504e-04 | 1.831e-02 | 215.26 |  |
| GM11 | A2_B | - | 8.728e-04 | 1.974e-02 | 215.88 |  |
| GM11 | AD | - | 9.222e-05 | 4.000e-10 | 96.60 | D at bound |
| GM12 | P1_A | 345 | 7.500e-05 | 1.144e-02 | 119.57 | fixed |
| GM12 | P1_B | 518 | 7.500e-05 | 1.273e-02 | 113.91 | fixed |
| GM12 | P2_A | 518 | 7.836e-05 | 1.144e-02 | 117.97 | interior |
| GM12 | P2_B | 562 | 7.766e-05 | 1.273e-02 | 112.88 | interior |
| GM12 | P3 | 4966 | 7.500e-05 | 3.009e-02 | 43.67 | interior |
| GM12 | P4 | 4923 | 8.235e-05 | 2.736e-02 | 40.74 | interior |
| GM12 | A1 | - | 7.500e-05 | 4.424e-10 | 213.96 | D at bound |
| GM12 | A2_A | - | 6.555e-04 | 1.144e-02 | 455.84 |  |
| GM12 | A2_B | - | 6.785e-04 | 1.273e-02 | 458.64 |  |
| GM12 | AD | - | 1.111e-04 | 1.016e-05 | 155.81 |  |
| GM13 | P1_A | 1474 | 6.670e-05 | 2.885e-02 | 33.04 | fixed |
| GM13 | P1_B | 1474 | 6.670e-05 | 2.786e-02 | 34.52 | fixed |
| GM13 | P2_A | 1800 | 7.034e-05 | 2.885e-02 | 32.29 | interior |
| GM13 | P2_B | 1474 | 6.943e-05 | 2.786e-02 | 32.90 | interior |
| GM13 | P3 | 3363 | 6.670e-05 | 3.739e-02 | 26.87 | interior |
| GM13 | P4 | 3363 | 6.845e-05 | 3.626e-02 | 26.66 | interior |
| GM13 | A1 | - | 6.670e-05 | 4.821e-10 | 291.27 | D at bound |
| GM13 | A2_A | - | 9.286e-04 | 2.885e-02 | 197.15 |  |
| GM13 | A2_B | - | 9.159e-04 | 2.786e-02 | 196.73 |  |
| GM13 | AD | - | 1.651e-04 | 1.022e-04 | 111.14 |  |
| GM15 | P1_A | 540 | 6.475e-05 | 8.980e-03 | 215.25 | fixed |
| GM15 | P1_B | 625 | 6.475e-05 | 1.073e-02 | 200.78 | fixed |
| GM15 | P2_A | 552 | 6.817e-05 | 8.980e-03 | 207.64 | interior |
| GM15 | P2_B | 699 | 6.802e-05 | 1.073e-02 | 194.73 | interior |
| GM15 | P3 | 1806 | 6.475e-05 | 2.163e-02 | 167.15 | interior |
| GM15 | P4 | 1805 | 6.517e-05 | 2.131e-02 | 167.10 | interior |
| GM15 | A1 | - | 6.475e-05 | 4.680e-10 | 485.16 | D at bound |
| GM15 | A2_A | - | 6.188e-04 | 8.980e-03 | 300.37 |  |
| GM15 | A2_B | - | 6.522e-04 | 1.073e-02 | 307.07 |  |
| GM15 | AD | - | 2.243e-04 | 2.715e-04 | 221.68 |  |
| GM16 | P1_A | 307 | 6.660e-05 | 1.218e-02 | 49.96 | fixed |
| GM16 | P1_B | 245 | 6.660e-05 | 1.037e-02 | 56.25 | fixed |
| GM16 | P2_A | 334 | 7.135e-05 | 1.218e-02 | 40.92 | interior |
| GM16 | P2_B | 271 | 7.213e-05 | 1.037e-02 | 42.72 | interior |
| GM16 | P3 | 5988 | 6.660e-05 | 3.622e-02 | 24.95 | interior |
| GM16 | P4 | 5988 | 7.405e-05 | 3.172e-02 | 21.42 | interior |
| GM16 | A1 | - | 6.660e-05 | 4.813e-10 | 268.43 | D at bound |
| GM16 | A2_A | - | 6.508e-04 | 1.218e-02 | 167.25 |  |
| GM16 | A2_B | - | 6.159e-04 | 1.037e-02 | 161.68 |  |
| GM16 | AD | - | 1.990e-04 | 2.471e-04 | 63.40 |  |
| GM17 | P1_A | 476 | 7.158e-05 | 1.552e-02 | 81.85 | fixed |
| GM17 | P1_B | 651 | 7.158e-05 | 1.987e-02 | 79.84 | fixed |
| GM17 | P2_A | 651 | 7.927e-05 | 1.552e-02 | 72.45 | interior |
| GM17 | P2_B | 713 | 7.465e-05 | 1.987e-02 | 77.52 | interior |
| GM17 | P3 | 11054 | 7.158e-05 | 3.335e-02 | 79.09 | interior |
| GM17 | P4 | 651 | 7.918e-05 | 1.558e-02 | 72.45 | interior |
| GM17 | A1 | - | 7.158e-05 | 4.223e-10 | 237.43 | D at bound |
| GM17 | A2_A | - | 8.436e-04 | 1.552e-02 | 271.20 |  |
| GM17 | A2_B | - | 9.346e-04 | 1.987e-02 | 276.34 |  |
| GM17 | AD | - | 1.447e-04 | 5.216e-05 | 105.01 |  |
| GM20 | P1_A | 248 | 8.482e-05 | 1.825e-02 | 78.34 | fixed |
| GM20 | P1_B | 248 | 8.482e-05 | 1.929e-02 | 78.69 | fixed |
| GM20 | P2_A | 248 | 8.472e-05 | 1.825e-02 | 78.34 | interior |
| GM20 | P2_B | 248 | 8.450e-05 | 1.929e-02 | 78.67 | interior |
| GM20 | P3 | 7037 | 8.482e-05 | 3.398e-02 | 38.04 | interior |
| GM20 | P4 | 7037 | 7.099e-05 | 5.000e-02 | 20.20 | y_lo |
| GM20 | A1 | - | 8.482e-05 | 4.084e-10 | 159.84 | D at bound |
| GM20 | A2_A | - | 1.003e-03 | 1.825e-02 | 57.89 |  |
| GM20 | A2_B | - | 1.026e-03 | 1.929e-02 | 58.87 |  |
| GM20 | AD | - | 4.440e-04 | 2.417e-03 | 35.40 |  |
| GM21 | P1_A | 581 | 6.205e-05 | 1.193e-02 | 121.59 | fixed |
| GM21 | P1_B | 749 | 6.205e-05 | 1.274e-02 | 116.99 | fixed |
| GM21 | P2_A | 791 | 6.864e-05 | 1.193e-02 | 108.88 | interior |
| GM21 | P2_B | 1231 | 7.490e-05 | 1.274e-02 | 105.63 | interior |
| GM21 | P3 | 2038 | 6.205e-05 | 2.281e-02 | 83.95 | interior |
| GM21 | P4 | 2038 | 6.964e-05 | 1.985e-02 | 78.55 | interior |
| GM21 | A1 | - | 6.205e-05 | 4.484e-10 | 385.46 | D at bound |
| GM21 | A2_A | - | 6.249e-04 | 1.193e-02 | 357.52 |  |
| GM21 | A2_B | - | 6.401e-04 | 1.274e-02 | 361.61 |  |
| GM21 | AD | - | 7.671e-05 | 4.000e-10 | 230.34 | D at bound |
| GM22 | P1_A | 663 | 7.056e-05 | 1.609e-02 | 41.76 | fixed |
| GM22 | P1_B | 939 | 7.056e-05 | 1.888e-02 | 34.48 | fixed |
| GM22 | P2_A | 861 | 7.441e-05 | 1.609e-02 | 39.98 | interior |
| GM22 | P2_B | 939 | 7.205e-05 | 1.888e-02 | 33.76 | interior |
| GM22 | P3 | 3317 | 7.056e-05 | 2.845e-02 | 25.91 | interior |
| GM22 | P4 | 5818 | 7.369e-05 | 2.879e-02 | 25.37 | interior |
| GM22 | A1 | - | 7.056e-05 | 4.162e-10 | 202.03 | D at bound |
| GM22 | A2_A | - | 8.124e-04 | 1.609e-02 | 168.75 |  |
| GM22 | A2_B | - | 8.695e-04 | 1.888e-02 | 171.72 |  |
| GM22 | AD | - | 2.068e-04 | 3.344e-04 | 96.47 |  |
| GM24 | P1_A | 1110 | 7.790e-05 | 3.127e-02 | 32.36 | fixed |
| GM24 | P1_B | 1313 | 7.790e-05 | 3.167e-02 | 32.26 | fixed |
| GM24 | P2_A | 1787 | 8.084e-05 | 3.127e-02 | 31.91 | interior |
| GM24 | P2_B | 1787 | 8.047e-05 | 3.167e-02 | 31.73 | interior |
| GM24 | P3 | 4368 | 7.790e-05 | 4.341e-02 | 30.02 | interior |
| GM24 | P4 | 4368 | 7.834e-05 | 4.309e-02 | 30.01 | interior |
| GM24 | A1 | - | 7.790e-05 | 4.595e-10 | 248.50 | D at bound |
| GM24 | A2_A | - | 1.081e-03 | 3.127e-02 | 164.83 |  |
| GM24 | A2_B | - | 1.086e-03 | 3.167e-02 | 165.04 |  |
| GM24 | AD | - | 2.395e-04 | 2.478e-04 | 50.73 |  |
| GM26 | P1_A | 791 | 9.138e-05 | 1.740e-02 | 30.87 | fixed |
| GM26 | P1_B | 1373 | 9.138e-05 | 1.901e-02 | 30.22 | fixed |
| GM26 | P2_A | 791 | 9.084e-05 | 1.740e-02 | 30.83 | interior |
| GM26 | P2_B | 1373 | 9.215e-05 | 1.901e-02 | 30.14 | interior |
| GM26 | P3 | 4364 | 9.138e-05 | 2.671e-02 | 27.81 | interior |
| GM26 | P4 | 4364 | 9.144e-05 | 2.668e-02 | 27.81 | interior |
| GM26 | A1 | - | 9.138e-05 | 4.400e-10 | 92.37 | D at bound |
| GM26 | A2_A | - | 9.647e-04 | 1.740e-02 | 134.98 |  |
| GM26 | A2_B | - | 9.939e-04 | 1.901e-02 | 137.01 |  |
| GM26 | AD | - | 1.985e-04 | 7.805e-05 | 50.57 |  |
| GM27 | P1_A | 358 | 7.589e-05 | 7.280e-03 | 185.00 | fixed |
| GM27 | P1_B | 426 | 7.589e-05 | 8.165e-03 | 173.32 | fixed |
| GM27 | P2_A | 712 | 9.245e-05 | 7.280e-03 | 178.40 | interior |
| GM27 | P2_B | 745 | 8.893e-05 | 8.165e-03 | 153.60 | interior |
| GM27 | P3 | 1923 | 7.589e-05 | 1.690e-02 | 91.11 | interior |
| GM27 | P4 | 1923 | 7.407e-05 | 1.756e-02 | 90.78 | interior |
| GM27 | A1 | - | 7.589e-05 | 4.477e-10 | 291.22 | D at bound |
| GM27 | A2_A | - | 6.295e-04 | 7.280e-03 | 183.66 |  |
| GM27 | A2_B | - | 6.597e-04 | 8.165e-03 | 185.28 |  |
| GM27 | AD | - | 5.309e-04 | 4.762e-03 | 180.63 |  |
| GM29 | P1_A | 510 | 6.981e-05 | 1.455e-02 | 76.41 | fixed |
| GM29 | P1_B | 510 | 6.981e-05 | 1.390e-02 | 77.19 | fixed |
| GM29 | P2_A | 722 | 7.593e-05 | 1.455e-02 | 72.57 | interior |
| GM29 | P2_B | 702 | 7.677e-05 | 1.390e-02 | 74.62 | interior |
| GM29 | P3 | 1323 | 6.981e-05 | 2.552e-02 | 69.06 | interior |
| GM29 | P4 | 1323 | 7.512e-05 | 2.362e-02 | 66.79 | interior |
| GM29 | A1 | - | 6.981e-05 | 4.118e-10 | 207.14 | D at bound |
| GM29 | A2_A | - | 6.659e-04 | 1.455e-02 | 322.58 |  |
| GM29 | A2_B | - | 6.544e-04 | 1.390e-02 | 320.44 |  |
| GM29 | AD | - | 9.614e-05 | 4.121e-06 | 122.74 |  |
| AP2_3dep | P3 | 7518 | 5.047e-06 | 2.084e-02 | 21.87 | interior |
| AP2_3dep | P4 | 7518 | 4.985e-06 | 2.103e-02 | 21.84 | interior |
| AP2_3dep | A1 | - | 5.047e-06 | 1.559e-09 | 3081.22 | D at bound |
| AP2_3dep | AD | - | 4.896e-05 | 7.260e-04 | 86.34 |  |
| AP3_3dep | P1_A | 625 | 4.984e-06 | 4.372e-03 | 213.74 | fixed |
| AP3_3dep | P1_B | 708 | 4.984e-06 | 5.272e-03 | 175.55 | fixed |
| AP3_3dep | P2_A | 593 | 6.227e-06 | 4.372e-03 | 129.11 | interior |
| AP3_3dep | P2_B | 701 | 6.051e-06 | 5.272e-03 | 115.17 | interior |
| AP3_3dep | P3 | 14529 | 4.984e-06 | 2.472e-02 | 40.27 | interior |
| AP3_3dep | P4 | 14529 | 5.767e-06 | 2.340e-02 | 38.05 | interior |
| AP3_3dep | A1 | - | 4.984e-06 | 1.539e-09 | 3414.77 | D at bound |
| AP3_3dep | A2_A | - | 8.420e-05 | 4.372e-03 | 226.76 |  |
| AP3_3dep | A2_B | - | 8.856e-05 | 5.272e-03 | 239.45 |  |
| AP3_3dep | AD | - | 5.434e-05 | 7.691e-04 | 168.95 |  |
| AP4_3dep | P1_A | 662 | 4.811e-06 | 5.360e-03 | 150.99 | fixed |
| AP4_3dep | P1_B | 902 | 4.811e-06 | 7.554e-03 | 84.99 | fixed |
| AP4_3dep | P2_A | 595 | 5.891e-06 | 5.360e-03 | 78.46 | interior |
| AP4_3dep | P2_B | 797 | 5.503e-06 | 7.554e-03 | 59.89 | interior |
| AP4_3dep | P3 | 5072 | 4.811e-06 | 1.866e-02 | 28.39 | interior |
| AP4_3dep | P4 | 5144 | 4.906e-06 | 1.864e-02 | 28.34 | interior |
| AP4_3dep | A1 | - | 4.811e-06 | 1.486e-09 | 3723.07 | D at bound |
| AP4_3dep | A2_A | - | 7.695e-05 | 5.360e-03 | 94.67 |  |
| AP4_3dep | A2_B | - | 8.562e-05 | 7.554e-03 | 118.09 |  |
| AP4_3dep | AD | - | 5.176e-05 | 1.262e-03 | 40.12 |  |
| AP5_3dep | P1_A | 696 | 5.299e-06 | 5.144e-03 | 107.24 | fixed |
| AP5_3dep | P1_B | 885 | 5.299e-06 | 6.494e-03 | 72.75 | fixed |
| AP5_3dep | P2_A | 746 | 6.326e-06 | 5.144e-03 | 65.01 | interior |
| AP5_3dep | P2_B | 891 | 6.056e-06 | 6.494e-03 | 50.32 | interior |
| AP5_3dep | P3 | 2630 | 5.299e-06 | 1.418e-02 | 34.83 | interior |
| AP5_3dep | P4 | 2629 | 5.163e-06 | 1.464e-02 | 34.61 | interior |
| AP5_3dep | A1 | - | 5.299e-06 | 1.349e-09 | 2652.61 | D at bound |
| AP5_3dep | A2_A | - | 8.652e-05 | 5.144e-03 | 134.50 |  |
| AP5_3dep | A2_B | - | 9.290e-05 | 6.494e-03 | 144.57 |  |
| AP5_3dep | AD | - | 4.958e-05 | 6.431e-04 | 81.13 |  |
| AP6 | P1_A | 944 | 4.720e-06 | 5.952e-03 | 83.69 | fixed |
| AP6 | P1_B | 1083 | 4.720e-06 | 7.218e-03 | 67.39 | fixed |
| AP6 | P2_A | 954 | 5.204e-06 | 5.952e-03 | 68.28 | interior |
| AP6 | P2_B | 1105 | 5.035e-06 | 7.218e-03 | 61.28 | interior |
| AP6 | P3 | 11491 | 4.720e-06 | 1.735e-02 | 34.19 | interior |
| AP6 | P4 | 11491 | 4.893e-06 | 1.700e-02 | 34.00 | interior |
| AP6 | A1 | - | 4.720e-06 | 1.458e-09 | 3188.32 | D at bound |
| AP6 | A2_A | - | 8.345e-05 | 5.952e-03 | 218.74 |  |
| AP6 | A2_B | - | 8.879e-05 | 7.218e-03 | 231.26 |  |
| AP6 | AD | - | 4.335e-05 | 5.624e-04 | 118.30 |  |
| AP7 | P1_A | 652 | 5.364e-06 | 5.239e-03 | 124.70 | fixed |
| AP7 | P1_B | 791 | 5.364e-06 | 5.771e-03 | 111.89 | fixed |
| AP7 | P2_A | 837 | 6.514e-06 | 5.239e-03 | 88.78 | interior |
| AP7 | P2_B | 862 | 6.335e-06 | 5.771e-03 | 81.28 | interior |
| AP7 | P3 | 5594 | 5.364e-06 | 1.822e-02 | 41.59 | interior |
| AP7 | P4 | 5594 | 5.395e-06 | 1.817e-02 | 41.59 | interior |
| AP7 | A1 | - | 5.364e-06 | 1.366e-09 | 2698.67 | D at bound |
| AP7 | A2_A | - | 7.709e-05 | 5.239e-03 | 196.08 |  |
| AP7 | A2_B | - | 7.919e-05 | 5.771e-03 | 202.93 |  |
| AP7 | AD | - | 4.844e-05 | 7.969e-04 | 123.32 |  |
| AP8 | P1_A | 681 | 5.463e-06 | 6.024e-03 | 138.20 | fixed |
| AP8 | P1_B | 765 | 5.463e-06 | 7.696e-03 | 107.21 | fixed |
| AP8 | P2_A | 681 | 6.630e-06 | 6.024e-03 | 83.44 | interior |
| AP8 | P2_B | 771 | 6.367e-06 | 7.696e-03 | 74.66 | interior |
| AP8 | P3 | 8930 | 5.463e-06 | 2.477e-02 | 40.62 | interior |
| AP8 | P4 | 9172 | 5.653e-06 | 2.536e-02 | 40.44 | interior |
| AP8 | A1 | - | 5.463e-06 | 1.391e-09 | 2754.77 | D at bound |
| AP8 | A2_A | - | 9.414e-05 | 6.024e-03 | 271.62 |  |
| AP8 | A2_B | - | 1.020e-04 | 7.696e-03 | 283.43 |  |
| AP8 | AD | - | 4.027e-05 | 2.393e-04 | 149.17 |  |
| AP9 | P3 | 8178 | 4.583e-06 | 1.957e-02 | 30.07 | interior |
| AP9 | P4 | 5703 | 4.026e-06 | 1.918e-02 | 27.15 | interior |
| AP9 | A1 | - | 4.583e-06 | 1.416e-09 | 4174.04 | D at bound |
| AP9 | AD | - | 5.157e-05 | 1.085e-03 | 45.31 |  |
| AP10 | P3 | 9342 | 4.173e-06 | 2.129e-02 | 29.78 | interior |
| AP10 | P4 | 9381 | 4.543e-06 | 2.045e-02 | 28.99 | interior |
| AP10 | A1 | - | 4.173e-06 | 3.124e-02 | 3345.47 |  |
| AP10 | AD | - | 4.480e-05 | 9.753e-04 | 61.08 |  |
| AP11 | P3 | 8521 | 4.085e-06 | 1.947e-02 | 34.20 | interior |
| AP11 | P4 | 3490 | 3.734e-06 | 1.763e-02 | 32.05 | interior |
| AP11 | A1 | - | 4.085e-06 | 3.058e-02 | 5703.66 |  |
| AP11 | AD | - | 4.132e-05 | 7.345e-04 | 59.00 |  |
| AP13 | P1_A | 597 | 5.382e-06 | 4.471e-03 | 153.49 | fixed |
| AP13 | P1_B | 957 | 5.382e-06 | 7.562e-03 | 71.94 | fixed |
| AP13 | P2_A | 582 | 6.879e-06 | 4.471e-03 | 75.80 | interior |
| AP13 | P2_B | 1025 | 6.382e-06 | 7.562e-03 | 50.89 | interior |
| AP13 | P3 | 6085 | 5.382e-06 | 1.820e-02 | 26.40 | interior |
| AP13 | P4 | 6085 | 5.761e-06 | 1.763e-02 | 26.08 | interior |
| AP13 | A1 | - | 5.382e-06 | 1.370e-09 | 2642.05 | D at bound |
| AP13 | A2_A | - | 7.494e-05 | 4.471e-03 | 128.84 |  |
| AP13 | A2_B | - | 8.652e-05 | 7.562e-03 | 153.98 |  |
| AP13 | AD | - | 5.016e-05 | 7.782e-04 | 92.87 |  |
| AP14 | P3 | 7631 | 4.199e-06 | 2.312e-02 | 23.01 | interior |
| AP14 | P4 | 7631 | 4.156e-06 | 2.326e-02 | 23.00 | interior |
| AP14 | A1 | - | 4.199e-06 | 3.143e-02 | 2579.72 |  |
| AP14 | AD | - | 4.192e-05 | 7.613e-04 | 61.29 |  |
| AP18 | P3 | 18045 | 5.275e-06 | 2.836e-02 | 20.50 | interior |
| AP18 | P4 | 14274 | 4.762e-06 | 2.964e-02 | 19.34 | interior |
| AP18 | A1 | - | 5.275e-06 | 1.343e-09 | 3223.26 | D at bound |
| AP18 | AD | - | 5.646e-05 | 8.398e-04 | 51.40 |  |
| AP19 | P1_A | 940 | 4.303e-06 | 6.118e-03 | 174.78 | fixed |
| AP19 | P1_B | 1149 | 4.303e-06 | 7.877e-03 | 102.86 | fixed |
| AP19 | P2_A | 752 | 5.319e-06 | 6.118e-03 | 88.76 | interior |
| AP19 | P2_B | 1155 | 5.059e-06 | 7.877e-03 | 66.19 | interior |
| AP19 | P3 | 3394 | 4.303e-06 | 1.653e-02 | 27.58 | interior |
| AP19 | P4 | 3402 | 4.122e-06 | 1.734e-02 | 27.07 | interior |
| AP19 | A1 | - | 4.303e-06 | 3.221e-02 | 4942.86 |  |
| AP19 | A2_A | - | 7.142e-05 | 6.118e-03 | 92.58 |  |
| AP19 | A2_B | - | 7.768e-05 | 7.877e-03 | 110.20 |  |
| AP19 | AD | - | 4.743e-05 | 1.469e-03 | 37.75 |  |
| AP20 | P1_A | 902 | 4.765e-06 | 5.400e-03 | 144.20 | fixed |
| AP20 | P1_B | 1097 | 4.765e-06 | 7.204e-03 | 82.38 | fixed |
| AP20 | P2_A | 626 | 5.827e-06 | 5.400e-03 | 77.55 | interior |
| AP20 | P2_B | 1097 | 5.452e-06 | 7.204e-03 | 57.56 | interior |
| AP20 | P3 | 1570 | 4.765e-06 | 1.299e-02 | 39.94 | interior |
| AP20 | P4 | 2690 | 4.309e-06 | 1.690e-02 | 36.84 | interior |
| AP20 | A1 | - | 4.765e-06 | 2.940e-02 | 3316.64 |  |
| AP20 | A2_A | - | 7.161e-05 | 5.400e-03 | 118.00 |  |
| AP20 | A2_B | - | 7.701e-05 | 7.204e-03 | 133.81 |  |
| AP20 | AD | - | 4.797e-05 | 7.655e-04 | 64.06 |  |
| AP23 | P1_A | 533 | 5.194e-06 | 3.661e-03 | 147.53 | fixed |
| AP23 | P1_B | 743 | 5.194e-06 | 4.505e-03 | 122.33 | fixed |
| AP23 | P2_A | 544 | 6.216e-06 | 3.661e-03 | 85.46 | interior |
| AP23 | P2_B | 579 | 6.053e-06 | 4.505e-03 | 78.17 | interior |
| AP23 | P3 | 6014 | 5.194e-06 | 1.772e-02 | 33.50 | interior |
| AP23 | P4 | 6014 | 5.847e-06 | 1.670e-02 | 31.20 | interior |
| AP23 | A1 | - | 5.194e-06 | 1.322e-09 | 2622.18 | D at bound |
| AP23 | A2_A | - | 8.011e-05 | 3.661e-03 | 314.74 |  |
| AP23 | A2_B | - | 8.488e-05 | 4.505e-03 | 328.21 |  |
| AP23 | AD | - | 3.226e-05 | 8.908e-05 | 162.26 |  |
| AP24 | P1_A | 1020 | 4.540e-06 | 5.939e-03 | 137.79 | fixed |
| AP24 | P1_B | 1565 | 4.540e-06 | 8.242e-03 | 77.43 | fixed |
| AP24 | P2_A | 925 | 5.318e-06 | 5.939e-03 | 91.25 | interior |
| AP24 | P2_B | 1565 | 5.001e-06 | 8.242e-03 | 63.66 | interior |
| AP24 | P3 | 8650 | 4.540e-06 | 2.312e-02 | 27.20 | interior |
| AP24 | P4 | 4516 | 4.032e-06 | 2.030e-02 | 24.75 | interior |
| AP24 | A1 | - | 4.540e-06 | 1.402e-09 | 4208.30 | D at bound |
| AP24 | A2_A | - | 7.649e-05 | 5.939e-03 | 70.22 |  |
| AP24 | A2_B | - | 8.418e-05 | 8.242e-03 | 85.97 |  |
| AP24 | AD | - | 5.339e-05 | 1.476e-03 | 35.15 |  |
| AP25 | P1_A | 628 | 4.779e-06 | 3.859e-03 | 201.80 | fixed |
| AP25 | P1_B | 751 | 4.779e-06 | 4.573e-03 | 168.32 | fixed |
| AP25 | P2_A | 492 | 5.897e-06 | 3.859e-03 | 124.96 | interior |
| AP25 | P2_B | 631 | 5.705e-06 | 4.573e-03 | 115.72 | interior |
| AP25 | P3 | 9624 | 4.779e-06 | 1.993e-02 | 48.36 | interior |
| AP25 | P4 | 9624 | 4.633e-06 | 2.043e-02 | 48.17 | interior |
| AP25 | A1 | - | 4.779e-06 | 1.476e-09 | 3457.19 | D at bound |
| AP25 | A2_A | - | 7.945e-05 | 3.859e-03 | 156.64 |  |
| AP25 | A2_B | - | 8.380e-05 | 4.573e-03 | 165.12 |  |
| AP25 | AD | - | 5.085e-05 | 8.297e-04 | 111.62 |  |
| AP27 | P3 | 5110 | 4.774e-06 | 1.601e-02 | 25.51 | interior |
| AP27 | P4 | 5110 | 4.778e-06 | 1.600e-02 | 25.51 | interior |
| AP27 | A1 | - | 4.774e-06 | 1.474e-09 | 3479.34 | D at bound |
| AP27 | AD | - | 4.329e-05 | 5.348e-04 | 58.26 |  |
| AP28 | P3 | 4489 | 4.384e-06 | 1.719e-02 | 35.92 | interior |
| AP28 | P4 | 4522 | 4.136e-06 | 1.891e-02 | 34.41 | interior |
| AP28 | A1 | - | 4.384e-06 | 1.354e-09 | 4560.03 | D at bound |
| AP28 | AD | - | 5.485e-05 | 1.174e-03 | 58.06 |  |
| AP29 | P1_A | 815 | 5.117e-06 | 6.297e-03 | 109.79 | fixed |
| AP29 | P1_B | 916 | 5.117e-06 | 6.704e-03 | 100.59 | fixed |
| AP29 | P2_A | 698 | 6.032e-06 | 6.297e-03 | 73.79 | interior |
| AP29 | P2_B | 719 | 5.965e-06 | 6.704e-03 | 71.78 | interior |
| AP29 | P3 | 16544 | 5.117e-06 | 2.461e-02 | 21.74 | interior |
| AP29 | P4 | 16545 | 5.820e-06 | 2.325e-02 | 20.84 | interior |
| AP29 | A1 | - | 5.117e-06 | 1.580e-09 | 3139.52 | D at bound |
| AP29 | A2_A | - | 8.969e-05 | 6.297e-03 | 119.91 |  |
| AP29 | A2_B | - | 9.152e-05 | 6.704e-03 | 123.96 |  |
| AP29 | AD | - | 5.166e-05 | 8.621e-04 | 49.32 |  |
| TTV1 | P1_A | 630 | 1.124e-05 | 2.905e-03 | 633.45 | fixed |
| TTV1 | P1_B | 630 | 1.124e-05 | 2.905e-03 | 633.45 | fixed |
| TTV1 | P2_A | 630 | 1.286e-05 | 2.905e-03 | 472.70 | interior |
| TTV1 | P2_B | 630 | 1.286e-05 | 2.905e-03 | 472.70 | interior |
| TTV1 | P3 | 49535 | 1.124e-05 | 4.321e-02 | 85.30 | interior |
| TTV1 | P4 | 59499 | 1.235e-05 | 4.388e-02 | 82.73 | interior |
| TTV1 | A1 | - | 1.124e-05 | 4.125e-10 | 3141.27 | D at bound |
| TTV1 | A2_A | - | 1.226e-04 | 2.905e-03 | 228.83 |  |
| TTV1 | A2_B | - | 1.226e-04 | 2.905e-03 | 228.83 |  |
| TTV1 | AD | - | 1.038e-04 | 1.600e-03 | 214.19 |  |
| TTV2 | P1_A | 644 | 1.581e-05 | 4.461e-03 | 186.62 | fixed |
| TTV2 | P1_B | 771 | 1.581e-05 | 4.708e-03 | 172.71 | fixed |
| TTV2 | P2_A | 776 | 1.709e-05 | 4.461e-03 | 149.35 | interior |
| TTV2 | P2_B | 808 | 1.699e-05 | 4.708e-03 | 141.63 | interior |
| TTV2 | P3 | 8536 | 1.581e-05 | 1.704e-02 | 61.83 | interior |
| TTV2 | P4 | 9131 | 1.645e-05 | 1.660e-02 | 59.79 | interior |
| TTV2 | A1 | - | 1.581e-05 | 4.734e-10 | 1466.33 | D at bound |
| TTV2 | A2_A | - | 2.197e-04 | 4.461e-03 | 518.16 |  |
| TTV2 | A2_B | - | 2.239e-04 | 4.708e-03 | 524.45 |  |
| TTV2 | AD | - | 7.622e-05 | 2.365e-04 | 209.66 |  |
| TTV3 | P3 | 89124 | 2.636e-05 | 5.000e-02 | 109.35 | y_lo |
| TTV3 | P4 | 89124 | 3.733e-05 | 5.000e-02 | 87.40 | y_lo |
| TTV3 | A1 | - | 2.636e-05 | 4.293e-10 | 590.34 | D at bound |
| TTV3 | AD | - | 5.170e-05 | 9.232e-06 | 326.85 |  |
| TTV4 | P3 | 40119 | 1.614e-05 | 4.722e-02 | 100.15 | interior |
| TTV4 | P4 | 40309 | 1.543e-05 | 5.000e-02 | 96.82 | y_lo |
| TTV4 | A1 | - | 1.614e-05 | 4.833e-10 | 1700.18 | D at bound |
| TTV4 | AD | - | 1.628e-04 | 3.601e-03 | 293.97 |  |
| TTV5 | P1_A | 403 | 2.176e-05 | 2.498e-03 | 764.07 | fixed |
| TTV5 | P1_B | 404 | 2.176e-05 | 2.699e-03 | 753.80 | fixed |
| TTV5 | P2_A | 458 | 2.579e-05 | 2.498e-03 | 710.21 | interior |
| TTV5 | P2_B | 458 | 2.523e-05 | 2.699e-03 | 709.07 | interior |
| TTV5 | P3 | 58386 | 2.176e-05 | 5.000e-02 | 148.99 | y_lo |
| TTV5 | P4 | 58612 | 2.651e-05 | 5.000e-02 | 137.88 | y_lo |
| TTV5 | A1 | - | 2.176e-05 | 4.342e-10 | 1240.79 | D at bound |
| TTV5 | A2_A | - | 1.974e-04 | 2.498e-03 | 1341.32 |  |
| TTV5 | A2_B | - | 2.028e-04 | 2.699e-03 | 1354.79 |  |
| TTV5 | AD | - | 6.051e-05 | 5.207e-05 | 842.95 |  |
| TTV6 | P1_A | 351 | 1.244e-05 | 1.993e-03 | 981.16 | fixed |
| TTV6 | P1_B | 351 | 1.244e-05 | 1.993e-03 | 981.16 | fixed |
| TTV6 | P2_A | 301 | 1.580e-05 | 1.993e-03 | 485.65 | interior |
| TTV6 | P2_B | 301 | 1.580e-05 | 1.993e-03 | 485.65 | interior |
| TTV6 | P3 | 29646 | 1.244e-05 | 3.695e-02 | 166.04 | interior |
| TTV6 | P4 | 10973 | 9.512e-06 | 3.327e-02 | 94.27 | interior |
| TTV6 | A1 | - | 1.244e-05 | 4.854e-02 | 1453.79 |  |
| TTV6 | A2_A | - | 1.085e-04 | 1.993e-03 | 223.85 |  |
| TTV6 | A2_B | - | 1.085e-04 | 1.993e-03 | 223.85 |  |
| TTV6 | AD | - | 9.452e-05 | 1.116e-03 | 204.34 |  |
| TTV7 | P1_A | 677 | 1.079e-05 | 3.098e-03 | 752.77 | fixed |
| TTV7 | P1_B | 677 | 1.079e-05 | 3.314e-03 | 707.13 | fixed |
| TTV7 | P2_A | 642 | 1.296e-05 | 3.098e-03 | 526.59 | interior |
| TTV7 | P2_B | 677 | 1.281e-05 | 3.314e-03 | 508.50 | interior |
| TTV7 | P3 | 38616 | 1.079e-05 | 4.113e-02 | 132.65 | interior |
| TTV7 | P4 | 35966 | 8.130e-06 | 5.000e-02 | 79.24 | y_lo |
| TTV7 | A1 | - | 1.079e-05 | 4.210e-02 | 2634.03 |  |
| TTV7 | A2_A | - | 1.361e-04 | 3.098e-03 | 269.90 |  |
| TTV7 | A2_B | - | 1.391e-04 | 3.314e-03 | 265.38 |  |
| TTV7 | AD | - | 2.255e-04 | 2.370e-02 | 134.26 |  |
| TTV8 | P1_A | 1033 | 1.150e-05 | 3.984e-03 | 443.74 | fixed |
| TTV8 | P1_B | 1034 | 1.150e-05 | 4.479e-03 | 397.76 | fixed |
| TTV8 | P2_A | 1090 | 1.314e-05 | 3.984e-03 | 342.41 | interior |
| TTV8 | P2_B | 1092 | 1.282e-05 | 4.479e-03 | 327.27 | interior |
| TTV8 | P3 | 64151 | 1.150e-05 | 4.991e-02 | 165.18 | interior |
| TTV8 | P4 | 38544 | 8.340e-06 | 4.557e-02 | 114.67 | interior |
| TTV8 | A1 | - | 1.150e-05 | 4.488e-02 | 1194.68 |  |
| TTV8 | A2_A | - | 1.509e-04 | 3.984e-03 | 337.38 |  |
| TTV8 | A2_B | - | 1.562e-04 | 4.479e-03 | 346.46 |  |
| TTV8 | AD | - | 1.518e-04 | 5.000e-02 | 272.96 | D at bound |
| TTV9 | P1_A | 1419 | 1.053e-05 | 4.202e-03 | 595.67 | fixed |
| TTV9 | P1_B | 2829 | 1.053e-05 | 6.252e-03 | 480.33 | fixed |
| TTV9 | P2_A | 1441 | 1.116e-05 | 4.202e-03 | 569.51 | interior |
| TTV9 | P2_B | 2829 | 1.050e-05 | 6.252e-03 | 480.28 | interior |
| TTV9 | P3 | 62865 | 1.053e-05 | 3.223e-02 | 328.78 | interior |
| TTV9 | P4 | 36561 | 7.648e-06 | 4.377e-02 | 115.18 | interior |
| TTV9 | A1 | - | 1.053e-05 | 4.735e-10 | 3738.96 | D at bound |
| TTV9 | A2_A | - | 1.596e-04 | 4.202e-03 | 343.68 |  |
| TTV9 | A2_B | - | 1.870e-04 | 6.252e-03 | 412.19 |  |
| TTV9 | AD | - | 1.195e-04 | 1.963e-03 | 284.91 |  |
| TTV11 | P1_A | 447 | 1.159e-05 | 2.312e-03 | 855.42 | fixed |
| TTV11 | P1_B | 617 | 1.159e-05 | 2.558e-03 | 806.34 | fixed |
| TTV11 | P2_A | 598 | 1.417e-05 | 2.312e-03 | 593.25 | interior |
| TTV11 | P2_B | 614 | 1.393e-05 | 2.558e-03 | 575.35 | interior |
| TTV11 | P3 | 13974 | 1.159e-05 | 1.634e-02 | 335.16 | interior |
| TTV11 | P4 | 30362 | 7.948e-06 | 4.108e-02 | 241.77 | interior |
| TTV11 | A1 | - | 1.159e-05 | 4.523e-02 | 1045.96 |  |
| TTV11 | A2_A | - | 1.180e-04 | 2.312e-03 | 304.35 |  |
| TTV11 | A2_B | - | 1.210e-04 | 2.558e-03 | 297.84 |  |
| TTV11 | AD | - | 1.453e-04 | 3.213e-02 | 164.20 |  |
| TTV15 | P1_A | 199 | 1.372e-05 | 1.506e-03 | 744.07 | fixed |
| TTV15 | P1_B | 199 | 1.372e-05 | 1.506e-03 | 744.07 | fixed |
| TTV15 | P2_A | 172 | 1.736e-05 | 1.506e-03 | 365.69 | interior |
| TTV15 | P2_B | 172 | 1.736e-05 | 1.506e-03 | 365.69 | interior |
| TTV15 | P3 | 26819 | 1.372e-05 | 3.182e-02 | 109.08 | interior |
| TTV15 | P4 | 26819 | 1.525e-05 | 2.858e-02 | 103.42 | interior |
| TTV15 | A1 | - | 1.372e-05 | 4.369e-02 | 1963.28 |  |
| TTV15 | A2_A | - | 1.261e-04 | 1.506e-03 | 589.34 |  |
| TTV15 | A2_B | - | 1.261e-04 | 1.506e-03 | 589.34 |  |
| TTV15 | AD | - | 2.112e-04 | 5.000e-02 | 327.53 | D at bound |

### C10. Held-out (5.3): the greedy basin split and the held-out MSE (m2; fit on one half, score on the other, both ways, averaged)

**GM**

| tile | cell shares h1 / h2 | largest basin share | valid cells in split | basins h1 / h2 | lopsided (< 30 %) | P4 held-out | AD held-out | P2_A held-out | P2_B held-out | H1 | H2_A | H2_B | P4 A_crit h1 / h2 | P4 scores h1->h2 / h2->h1 | AD scores h1->h2 / h2->h1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM1_fema | 0.50 / 0.50 | 0.27 | 0.90 | 37 / 31 | no | 31.97 | 61.34 | 32.52 | 31.30 | yes | yes | yes | 819 / 4089 | 39.3 / 24.6 | 76.3 / 46.4 |
| GM2_fema | 0.50 / 0.50 | 0.26 | 0.91 | 28 / 15 | no | 33.55 | 125.10 | 45.89 | 40.64 | yes | yes | yes | 1519 / 1525 | 29.0 / 38.1 | 153.6 / 96.6 |
| GM3_fema | 0.50 / 0.50 | 0.36 | 0.93 | 25 / 24 | no | 45.03 | 83.88 | 41.14 | 40.39 | yes | yes | yes | 387 / 1669 | 52.0 / 38.1 | 100.9 / 66.9 |
| GM4_fema | 0.57 / 0.43 | 0.57 | 0.92 | 1 / 44 | no | 76.32 | 155.80 | 98.02 | 72.69 | yes | yes | yes | 2563 / 3282 | 55.1 / 97.5 | 155.2 / 156.4 |
| GM5_fema | 0.78 / 0.22 | 0.78 | 0.93 | 1 / 52 | yes | 51.58 | 120.04 | 78.39 | 78.83 | yes | yes | yes | 1906 / 1015 | 55.6 / 47.6 | 156.7 / 83.4 |
| GM6_fema | 0.54 / 0.46 | 0.54 | 0.91 | 1 / 48 | no | 85.16 | 382.90 | 106.87 | 132.53 | yes | yes | yes | 1915 / 10533 | 109.6 / 60.7 | 412.8 / 353.0 |
| GM7_fema | 0.52 / 0.48 | 0.52 | 0.94 | 1 / 56 | no | 33.18 | 38.39 | 26.60 | 26.54 | yes | yes | yes | 6134 / 2865 | 28.5 / 37.9 | 33.0 / 43.8 |
| GM8 | 0.49 / 0.51 | 0.36 | 0.95 | 29 / 2 | no | 91.19 | 178.98 | 113.07 | 112.76 | yes | yes | yes | 8449 / 2171 | 105.5 / 76.9 | 155.9 / 202.1 |
| GM9 | 0.51 / 0.49 | 0.51 | 0.95 | 1 / 65 | no | 100.65 | 109.38 | 138.71 | 133.24 | yes | no | no | 5235 / 4032 | 88.2 / 113.0 | 119.1 / 99.6 |
| GM11 | 0.50 / 0.50 | 0.48 | 0.95 | 12 / 18 | no | 101.59 | 434.98 | 96.54 | 95.84 | yes | yes | yes | 5850 / 584 | 84.7 / 118.4 | 231.8 / 638.1 |
| GM12 | 0.50 / 0.50 | 0.37 | 0.89 | 23 / 17 | no | 49.88 | 198.53 | 142.68 | 146.07 | yes | yes | yes | 4250 / 8638 | 48.8 / 50.9 | 221.4 / 175.6 |
| GM13 | 0.50 / 0.50 | 0.44 | 0.94 | 17 / 18 | no | 41.66 | 307.26 | 48.44 | 50.66 | yes | yes | yes | 3253 / 1474 | 34.5 / 48.8 | 156.9 / 457.6 |
| GM15 | 0.50 / 0.50 | 0.29 | 0.97 | 70 / 4 | no | 515.16 | 402.33 | 452.00 | 425.98 | no | no | no | 8235 / 2292 | 589.9 / 440.4 | 409.5 / 395.1 |
| GM16 | 0.50 / 0.50 | 0.26 | 0.92 | 21 / 16 | no | 23.29 | 68.88 | 56.84 | 57.76 | yes | yes | yes | 5616 / 7125 | 23.2 / 23.3 | 61.3 / 76.5 |
| GM17 | 0.60 / 0.40 | 0.60 | 0.94 | 1 / 28 | no | 121.08 | 112.32 | 79.60 | 82.78 | no | yes | yes | 651 / 26840 | 74.4 / 167.7 | 110.3 / 114.3 |
| GM20 | 0.55 / 0.45 | 0.55 | 0.93 | 1 / 45 | no | 31.80 | 45.36 | 151.14 | 141.10 | yes | no | no | 4383 / 7362 | 25.1 / 38.5 | 47.8 / 42.9 |
| GM21 | 0.50 / 0.50 | 0.43 | 0.91 | 16 / 42 | no | 245.14 | 1042.50 | 255.24 | 229.62 | yes | yes | yes | 1538 / 2042 | 189.0 / 301.3 | 625.1 / 1459.9 |
| GM22 | 0.86 / 0.14 | 0.86 | 0.94 | 1 / 46 | yes | 62.95 | 165.41 | 92.46 | 81.82 | yes | yes | yes | 3314 / 2885 | 22.4 / 103.5 | 183.5 / 147.3 |
| GM24 | 0.63 / 0.37 | 0.63 | 0.91 | 1 / 40 | no | 49.10 | 118.67 | 63.11 | 61.48 | yes | yes | yes | 5512 / 3789 | 41.6 / 56.6 | 61.0 / 176.3 |
| GM26 | 0.92 / 0.08 | 0.92 | 0.96 | 1 / 56 | yes | 90.38 | 726.66 | 53.50 | 50.60 | yes | yes | yes | 4312 / 5051 | 67.7 / 113.0 | 86.7 / 1366.6 |
| GM27 | 0.50 / 0.50 | 0.43 | 0.96 | 31 / 33 | no | 270.56 | 701.23 | 450.20 | 419.89 | yes | yes | yes | 1168 / 1913 | 231.1 / 310.0 | 648.2 / 754.3 |
| GM29 | 0.50 / 0.50 | 0.39 | 0.92 | 30 / 17 | no | 118.26 | 298.57 | 109.92 | 109.22 | yes | yes | yes | 507 / 1439 | 107.4 / 129.1 | 167.1 / 430.0 |

**AP**

| tile | cell shares h1 / h2 | largest basin share | valid cells in split | basins h1 / h2 | lopsided (< 30 %) | P4 held-out | AD held-out | P2_A held-out | P2_B held-out | H1 | H2_A | H2_B | P4 A_crit h1 / h2 | P4 scores h1->h2 / h2->h1 | AD scores h1->h2 / h2->h1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP2_3dep | 0.55 / 0.45 | 0.55 | 0.97 | 1 / 111 | no | 31.10 | 149.75 | - | - | yes | - | - | 7512 / 4430 | 29.2 / 33.0 | 116.9 / 182.6 |
| AP3_3dep | 0.72 / 0.28 | 0.72 | 0.92 | 1 / 135 | yes | 111.88 | 292.89 | 157.54 | 142.96 | yes | yes | yes | 14529 / 862 | 50.2 / 173.5 | 215.0 / 370.8 |
| AP4_3dep | 0.50 / 0.50 | 0.43 | 0.92 | 40 / 44 | no | 50.92 | 69.78 | 128.50 | 98.86 | yes | no | no | 2019 / 7317 | 50.2 / 51.6 | 81.4 / 58.2 |
| AP5_3dep | 0.59 / 0.41 | 0.59 | 0.91 | 1 / 70 | no | 74.97 | 231.24 | 75.34 | 57.34 | yes | yes | yes | 1566 / 18497 | 50.7 / 99.2 | 138.8 / 323.7 |
| AP6 | 0.79 / 0.21 | 0.79 | 0.94 | 1 / 93 | yes | 72.17 | 1833.59 | 71.88 | 59.52 | yes | yes | yes | 10494 / 4680 | 42.1 / 102.3 | 93.4 / 3573.8 |
| AP7 | 0.60 / 0.40 | 0.60 | 0.96 | 1 / 76 | no | 61.68 | 215.92 | 112.54 | 105.58 | yes | yes | yes | 4485 / 9061 | 53.3 / 70.0 | 170.8 / 261.0 |
| AP8 | 0.72 / 0.28 | 0.72 | 0.96 | 1 / 76 | yes | 100.35 | 738.20 | 145.09 | 145.80 | yes | yes | yes | 8930 / 2913 | 52.5 / 148.2 | 332.6 / 1143.8 |
| AP9 | 0.50 / 0.50 | 0.48 | 0.90 | 48 / 71 | no | 38.76 | 57.60 | - | - | yes | - | - | 5703 / 6437 | 38.1 / 39.4 | 51.6 / 63.6 |
| AP10 | 0.50 / 0.50 | 0.36 | 0.89 | 47 / 46 | no | 34.04 | 79.31 | - | - | yes | - | - | 10348 / 9332 | 31.5 / 36.6 | 110.9 / 47.7 |
| AP11 | 0.59 / 0.41 | 0.59 | 0.98 | 1 / 88 | no | 33.39 | 106.26 | - | - | yes | - | - | 3480 / 9028 | 25.2 / 41.6 | 46.1 / 166.4 |
| AP13 | 0.81 / 0.19 | 0.81 | 0.96 | 1 / 93 | yes | 24.11 | 160.05 | 78.03 | 52.73 | yes | yes | yes | 6085 / 4671 | 14.3 / 33.9 | 96.6 / 223.5 |
| AP14 | 0.50 / 0.50 | 0.30 | 0.93 | 64 / 57 | no | 29.98 | 76.75 | - | - | yes | - | - | 2541 / 7842 | 34.7 / 25.3 | 100.1 / 53.4 |
| AP18 | 0.92 / 0.08 | 0.92 | 0.97 | 1 / 115 | yes | 44.58 | 134.09 | - | - | yes | - | - | 14274 / 4383 | 16.1 / 73.1 | 46.6 / 221.6 |
| AP19 | 0.50 / 0.50 | 0.43 | 0.90 | 52 / 53 | no | 42.20 | 86.07 | 98.05 | 74.18 | yes | no | yes | 3464 / 6593 | 35.2 / 49.2 | 55.3 / 116.9 |
| AP20 | 0.50 / 0.50 | 0.36 | 0.91 | 45 / 28 | no | 70.76 | 231.99 | 100.51 | 72.89 | yes | yes | yes | 2865 / 3170 | 68.6 / 72.9 | 134.8 / 329.2 |
| AP23 | 0.81 / 0.19 | 0.81 | 0.96 | 1 / 79 | yes | 121.26 | 1016.75 | 145.98 | 146.39 | yes | yes | yes | 5671 / 2232 | 41.3 / 201.2 | 358.1 / 1675.4 |
| AP24 | 0.62 / 0.38 | 0.62 | 0.93 | 1 / 103 | no | 37.94 | 52.21 | 127.56 | 86.09 | yes | no | no | 4516 / 4523 | 32.1 / 43.8 | 45.4 / 59.0 |
| AP25 | 0.82 / 0.18 | 0.82 | 0.99 | 1 / 70 | yes | 58.68 | 96.73 | 101.42 | 96.79 | yes | no | no | 9624 / 9040 | 37.5 / 79.8 | 29.0 / 164.5 |
| AP27 | 0.51 / 0.49 | 0.51 | 0.92 | 1 / 96 | no | 39.69 | 63.16 | - | - | yes | - | - | 2164 / 5110 | 38.8 / 40.6 | 90.8 / 35.5 |
| AP28 | 0.63 / 0.37 | 0.63 | 0.95 | 1 / 96 | no | 49.72 | 119.69 | - | - | yes | - | - | 4902 / 4255 | 42.2 / 57.2 | 73.9 / 165.5 |
| AP29 | 0.57 / 0.43 | 0.57 | 0.95 | 1 / 61 | no | 47.54 | 123.30 | 114.42 | 119.41 | yes | yes | yes | 9621 / 9058 | 45.5 / 49.6 | 69.7 / 176.9 |

**TTV**

| tile | cell shares h1 / h2 | largest basin share | valid cells in split | basins h1 / h2 | lopsided (< 30 %) | P4 held-out | AD held-out | P2_A held-out | P2_B held-out | H1 | H2_A | H2_B | P4 A_crit h1 / h2 | P4 scores h1->h2 / h2->h1 | AD scores h1->h2 / h2->h1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 0.50 / 0.50 | 0.27 | 0.94 | 41 / 47 | no | 159.27 | 417.74 | 583.24 | 583.24 | yes | no | no | 20640 / 53562 | 209.7 / 108.8 | 377.0 / 458.5 |
| TTV2 | 0.62 / 0.38 | 0.62 | 0.95 | 1 / 72 | no | 281.56 | 2307.38 | 225.33 | 222.97 | yes | yes | yes | 9138 / 6206 | 100.4 / 462.7 | 431.1 / 4183.6 |
| TTV3 | 0.80 / 0.20 | 0.80 | 0.95 | 1 / 88 | yes | 164.95 | 727.62 | - | - | yes | - | - | 89124 / 15293 | 172.4 / 157.5 | 361.5 / 1093.7 |
| TTV4 | 0.50 / 0.50 | 0.39 | 0.93 | 46 / 38 | no | 113.64 | 522.28 | - | - | yes | - | - | 44244 / 39401 | 136.4 / 90.9 | 657.1 / 387.5 |
| TTV5 | 0.53 / 0.47 | 0.53 | 0.96 | 1 / 67 | no | 429.43 | 2975.63 | 2025.61 | 2041.29 | yes | yes | yes | 31497 / 73783 | 580.8 / 278.1 | 2540.4 / 3410.9 |
| TTV6 | 0.50 / 0.50 | 0.30 | 0.90 | 46 / 41 | no | 275.21 | 635.06 | 833.05 | 833.05 | yes | no | no | 11173 / 5899 | 148.9 / 401.5 | 545.2 / 724.9 |
| TTV7 | 0.50 / 0.50 | 0.24 | 0.95 | 31 / 28 | no | 90.36 | 152.01 | 552.07 | 532.24 | yes | no | no | 36074 / 29339 | 97.7 / 83.0 | 110.1 / 193.9 |
| TTV8 | 0.50 / 0.50 | 0.34 | 0.92 | 42 / 22 | no | 171.47 | 552.57 | 386.53 | 377.30 | yes | yes | yes | 38712 / 29646 | 111.5 / 231.4 | 553.7 / 551.4 |
| TTV9 | 0.77 / 0.23 | 0.77 | 0.96 | 1 / 77 | yes | 112.07 | 2405.72 | 1214.33 | 1107.62 | yes | yes | yes | 36561 / 44157 | 63.2 / 161.0 | 314.0 / 4497.4 |
| TTV11 | 0.50 / 0.50 | 0.16 | 0.93 | 24 / 62 | no | 347.51 | 174.82 | 715.72 | 702.19 | no | no | no | 39505 / 15238 | 330.0 / 365.1 | 219.2 / 130.4 |
| TTV15 | 0.50 / 0.50 | 0.36 | 0.93 | 35 / 38 | no | 168.74 | 722.13 | 486.56 | 486.56 | yes | yes | yes | 21412 / 18353 | 131.8 / 205.7 | 562.3 / 881.9 |

### C11. The rules of section 6 per site: counts, denominators and verdicts (holds = count >= 80 % of N, rounded up)

**GM primary** (N = 22, with D0 = 22)

| rule | count / N | needed | verdict | tiles failing | values |
|---|---|---|---|---|---|
| F1 | 22 / 22 | 18 | holds | - | median R_MSE 2.25, range 1.33-4.17 |
| F2 | 21 / 22 | 18 | holds | GM11 | median K/K0 P4 1.05, AD 2.37 |
| F3_A | 21 / 22 | 18 | holds | GM27 | median D/D0 P4 1.74, AD 0.0067 |
| F3_B | 21 / 22 | 18 | holds | GM27 | median D/D0 P4 1.76, AD 0.00685 |
| E1_A | 20 / 22 | 18 | holds | GM20, GM27 |  |
| E1_B | 21 / 22 | 18 | holds | GM20 |  |
| E2_P2_A | 21 / 22 | 18 | holds | GM20 |  |
| E2_P2_B | 21 / 22 | 18 | holds | GM20 |  |
| E2_P3 | 21 / 22 | 18 | holds | GM20 |  |
| F4 | A: median |log10(D/D0)| AD 2.18 (n 22); B: 2.17 (n 22) | - | holds | - | holds if >= 1.0 at both ends |
| H1 (all) | 20 / 22 | 18 | holds | GM15, GM17 | lopsided tiles: GM5_fema, GM22, GM26 |
| H1 (balanced) | 17 / 19 | 16 | holds | GM15, GM17 |  |
| H2_A (all) | 19 / 22 | 18 | holds | GM9, GM15, GM20 | lopsided tiles: GM5_fema, GM22, GM26 |
| H2_A (balanced) | 16 / 19 | 16 | holds | GM9, GM15, GM20 |  |
| H2_B (all) | 19 / 22 | 18 | holds | GM9, GM15, GM20 | lopsided tiles: GM5_fema, GM22, GM26 |
| H2_B (balanced) | 16 / 19 | 16 | holds | GM9, GM15, GM20 |  |

**GM clean** (N = 19, with D0 = 19)

| rule | count / N | needed | verdict | tiles failing | values |
|---|---|---|---|---|---|
| F1 | 19 / 19 | 16 | holds | - | median R_MSE 2.51, range 1.33-4.17 |
| F2 | 18 / 19 | 16 | holds | GM11 | median K/K0 P4 1.04, AD 2.63 |
| F3_A | 18 / 19 | 16 | holds | GM27 | median D/D0 P4 1.79, AD 0.00818 |
| F3_B | 18 / 19 | 16 | holds | GM27 | median D/D0 P4 1.81, AD 0.00855 |
| E1_A | 17 / 19 | 16 | holds | GM20, GM27 |  |
| E1_B | 18 / 19 | 16 | holds | GM20 |  |
| E2_P2_A | 18 / 19 | 16 | holds | GM20 |  |
| E2_P2_B | 18 / 19 | 16 | holds | GM20 |  |
| E2_P3 | 18 / 19 | 16 | holds | GM20 |  |
| F4 | A: median |log10(D/D0)| AD 2.09 (n 19); B: 2.07 (n 19) | - | holds | - | holds if >= 1.0 at both ends |
| H1 (all) | 17 / 19 | 16 | holds | GM15, GM17 | lopsided tiles: GM5_fema, GM22 |
| H1 (balanced) | 15 / 17 | 14 | holds | GM15, GM17 |  |
| H2_A (all) | 16 / 19 | 16 | holds | GM9, GM15, GM20 | lopsided tiles: GM5_fema, GM22 |
| H2_A (balanced) | 14 / 17 | 14 | holds | GM9, GM15, GM20 |  |
| H2_B (all) | 16 / 19 | 16 | holds | GM9, GM15, GM20 | lopsided tiles: GM5_fema, GM22 |
| H2_B (balanced) | 14 / 17 | 14 | holds | GM9, GM15, GM20 |  |

**GM offmesa** (N = 6, with D0 = 6)

| rule | count / N | needed | verdict | tiles failing | values |
|---|---|---|---|---|---|
| F1 | 6 / 6 | 5 | report only | - | median R_MSE 2.42, range 1.31-3.69 |
| F2 | 6 / 6 | 5 | report only | - | median K/K0 P4 1.03, AD 4.27 |
| F3_A | 4 / 6 | 5 | report only | GM14, GM25 | median D/D0 P4 3.59, AD 0.0959 |
| F3_B | 5 / 6 | 5 | report only | GM25 | median D/D0 P4 3.28, AD 0.0979 |
| E1_A | 5 / 6 | 5 | report only | GM14 |  |
| E1_B | 6 / 6 | 5 | report only | - |  |
| E2_P2_A | 4 / 6 | 5 | report only | GM14, GM19 |  |
| E2_P2_B | 5 / 6 | 5 | report only | GM19 |  |
| E2_P3 | 6 / 6 | 5 | report only | - |  |
| F4 | A: median |log10(D/D0)| AD 1.17 (n 6); B: 1.17 (n 6) | - | report only | - | holds if >= 1.0 at both ends |
| H1 (all) | 5 / 6 | 5 | holds | GM14 | lopsided tiles: GM18 |
| H1 (balanced) | 4 / 5 | 4 | holds | GM14 |  |
| H2_A (all) | 3 / 6 | 5 | fails | GM14, GM18, GM19 | lopsided tiles: GM18 |
| H2_A (balanced) | 3 / 5 | 4 | fails | GM14, GM19 |  |
| H2_B (all) | 3 / 6 | 5 | fails | GM14, GM18, GM19 | lopsided tiles: GM18 |
| H2_B (balanced) | 3 / 5 | 4 | fails | GM14, GM19 |  |

**GM ncalm** (N = 7, with D0 = 7)

| rule | count / N | needed | verdict | tiles failing | values |
|---|---|---|---|---|---|
| F1 | 7 / 7 | 6 | report only | - | median R_MSE 2.80, range 1.82-4.31 |
| F2 | 7 / 7 | 6 | report only | - | median K/K0 P4 1.05, AD 2.26 |
| F3_A | 7 / 7 | 6 | report only | - | median D/D0 P4 2.18, AD 0.00519 |
| F3_B | 7 / 7 | 6 | report only | - | median D/D0 P4 2.28, AD 0.005 |
| E1_A | 7 / 7 | 6 | report only | - |  |
| E1_B | 7 / 7 | 6 | report only | - |  |
| E2_P2_A | 7 / 7 | 6 | report only | - |  |
| E2_P2_B | 7 / 7 | 6 | report only | - |  |
| E2_P3 | 7 / 7 | 6 | report only | - |  |
| F4 | A: median |log10(D/D0)| AD 2.28 (n 7); B: 2.30 (n 7) | - | report only | - | holds if >= 1.0 at both ends |
| H1 (all) | 7 / 7 | 6 | holds | - | lopsided tiles: GM5 |
| H1 (balanced) | 6 / 6 | 5 | holds | - |  |
| H2_A (all) | 7 / 7 | 6 | holds | - | lopsided tiles: GM5 |
| H2_A (balanced) | 6 / 6 | 5 | holds | - |  |
| H2_B (all) | 7 / 7 | 6 | holds | - | lopsided tiles: GM5 |
| H2_B (balanced) | 6 / 6 | 5 | holds | - |  |

Clean subset (6.7), outcomes that differ from the primary group: none.

**AP primary** (N = 21, with D0 = 13)

| rule | count / N | needed | verdict | tiles failing | values |
|---|---|---|---|---|---|
| F1 | 21 / 21 | 17 | holds | - | median R_MSE 2.34, range 1.39-5.20 |
| F2 | 21 / 21 | 17 | holds | - | median K/K0 P4 0.99, AD 10.10 |
| F3_A | 12 / 13 | 11 | holds | AP25 | median D/D0 P4 3.48, AD 0.152 |
| F3_B | 13 / 13 | 11 | holds | - | median D/D0 P4 2.47, AD 0.129 |
| E1_A | 7 / 13 | 11 | fails | AP4_3dep, AP13, AP19, AP20, AP24, AP25 |  |
| E1_B | 12 / 13 | 11 | holds | AP25 |  |
| E2_P2_A | 7 / 13 | 11 | fails | AP4_3dep, AP19, AP20, AP24, AP25, AP29 |  |
| E2_P2_B | 8 / 13 | 11 | fails | AP4_3dep, AP19, AP24, AP25, AP29 |  |
| E2_P3 | 21 / 21 | 17 | holds | - |  |
| F4 | A: median |log10(D/D0)| AD 0.82 (n 13); B: 0.89 (n 13) | - | fails | - | holds if >= 1.0 at both ends |
| H1 (all) | 21 / 21 | 17 | holds | - | lopsided tiles: AP3_3dep, AP6, AP8, AP13, AP18, AP23, AP25 |
| H1 (balanced) | 14 / 14 | 12 | holds | - |  |
| H2_A (all) | 9 / 13 | 11 | fails | AP4_3dep, AP19, AP24, AP25 | lopsided tiles: AP3_3dep, AP6, AP8, AP13, AP18, AP23, AP25 |
| H2_A (balanced) | 4 / 7 | 6 | fails | AP4_3dep, AP19, AP24 |  |
| H2_B (all) | 10 / 13 | 11 | fails | AP4_3dep, AP24, AP25 | lopsided tiles: AP3_3dep, AP6, AP8, AP13, AP18, AP23, AP25 |
| H2_B (balanced) | 5 / 7 | 6 | fails | AP4_3dep, AP24 |  |

**AP clean** (N = 20, with D0 = 12)

| rule | count / N | needed | verdict | tiles failing | values |
|---|---|---|---|---|---|
| F1 | 20 / 20 | 16 | holds | - | median R_MSE 2.35, range 1.39-5.20 |
| F2 | 20 / 20 | 16 | holds | - | median K/K0 P4 1.00, AD 10.08 |
| F3_A | 12 / 12 | 10 | holds | - | median D/D0 P4 3.47, AD 0.147 |
| F3_B | 12 / 12 | 10 | holds | - | median D/D0 P4 2.47, AD 0.117 |
| E1_A | 7 / 12 | 10 | fails | AP4_3dep, AP13, AP19, AP20, AP24 |  |
| E1_B | 12 / 12 | 10 | holds | - |  |
| E2_P2_A | 7 / 12 | 10 | fails | AP4_3dep, AP19, AP20, AP24, AP29 |  |
| E2_P2_B | 8 / 12 | 10 | fails | AP4_3dep, AP19, AP24, AP29 |  |
| E2_P3 | 20 / 20 | 16 | holds | - |  |
| F4 | A: median |log10(D/D0)| AD 0.83 (n 12); B: 0.93 (n 12) | - | fails | - | holds if >= 1.0 at both ends |
| H1 (all) | 20 / 20 | 16 | holds | - | lopsided tiles: AP3_3dep, AP6, AP8, AP13, AP18, AP23 |
| H1 (balanced) | 14 / 14 | 12 | holds | - |  |
| H2_A (all) | 9 / 12 | 10 | fails | AP4_3dep, AP19, AP24 | lopsided tiles: AP3_3dep, AP6, AP8, AP13, AP18, AP23 |
| H2_A (balanced) | 4 / 7 | 6 | fails | AP4_3dep, AP19, AP24 |  |
| H2_B (all) | 10 / 12 | 10 | holds | AP4_3dep, AP24 | lopsided tiles: AP3_3dep, AP6, AP8, AP13, AP18, AP23 |
| H2_B (balanced) | 5 / 7 | 6 | fails | AP4_3dep, AP24 |  |

**AP pamap** (N = 5, with D0 = 5)

| rule | count / N | needed | verdict | tiles failing | values |
|---|---|---|---|---|---|
| F1 | 5 / 5 | 4 | report only | - | median R_MSE 2.64, range 1.30-3.63 |
| F2 | 5 / 5 | 4 | report only | - | median K/K0 P4 0.98, AD 9.02 |
| F3_A | 3 / 5 | 4 | report only | AP3, AP4 | median D/D0 P4 5.22, AD 0.196 |
| F3_B | 4 / 5 | 4 | report only | AP3 | median D/D0 P4 3.64, AD 0.136 |
| E1_A | 0 / 5 | 4 | report only | AP1, AP2, AP3, AP4, AP5 |  |
| E1_B | 2 / 5 | 4 | report only | AP2, AP3, AP5 |  |
| E2_P2_A | 0 / 5 | 4 | report only | AP1, AP2, AP3, AP4, AP5 |  |
| E2_P2_B | 0 / 5 | 4 | report only | AP1, AP2, AP3, AP4, AP5 |  |
| E2_P3 | 5 / 5 | 4 | report only | - |  |
| F4 | A: median |log10(D/D0)| AD 0.71 (n 5); B: 0.87 (n 5) | - | report only | - | holds if >= 1.0 at both ends |
| H1 (all) | 4 / 5 | 4 | holds | AP4 | lopsided tiles: AP1, AP3 |
| H1 (balanced) | 2 / 3 | 3 | fails | AP4 |  |
| H2_A (all) | 3 / 5 | 4 | fails | AP2, AP4 | lopsided tiles: AP1, AP3 |
| H2_A (balanced) | 1 / 3 | 3 | fails | AP2, AP4 |  |
| H2_B (all) | 3 / 5 | 4 | fails | AP2, AP4 | lopsided tiles: AP1, AP3 |
| H2_B (balanced) | 1 / 3 | 3 | fails | AP2, AP4 |  |

**AP ap1_3dep** (N = 1, with D0 = 1)

| rule | count / N | needed | verdict | tiles failing | values |
|---|---|---|---|---|---|
| F1 | 1 / 1 | 1 | report only | - | median R_MSE 2.97, range 2.97-2.97 |
| F2 | 1 / 1 | 1 | report only | - | median K/K0 P4 0.93, AD 9.87 |
| F3_A | 1 / 1 | 1 | report only | - | median D/D0 P4 3.24, AD 0.129 |
| F3_B | 1 / 1 | 1 | report only | - | median D/D0 P4 2.68, AD 0.107 |
| E1_A | 0 / 1 | 1 | report only | AP1_3dep |  |
| E1_B | 1 / 1 | 1 | report only | - |  |
| E2_P2_A | 0 / 1 | 1 | report only | AP1_3dep |  |
| E2_P2_B | 0 / 1 | 1 | report only | AP1_3dep |  |
| E2_P3 | 1 / 1 | 1 | report only | - |  |
| F4 | A: median |log10(D/D0)| AD 0.89 (n 1); B: 0.97 (n 1) | - | report only | - | holds if >= 1.0 at both ends |
| H1 (all) | 1 / 1 | 1 | holds | - | lopsided tiles: AP1_3dep |
| H1 (balanced) | 0 / 0 | 0 | fails | - |  |
| H2_A (all) | 1 / 1 | 1 | holds | - | lopsided tiles: AP1_3dep |
| H2_A (balanced) | 0 / 0 | 0 | fails | - |  |
| H2_B (all) | 1 / 1 | 1 | holds | - | lopsided tiles: AP1_3dep |
| H2_B (balanced) | 0 / 0 | 0 | fails | - |  |

Clean subset (6.7), outcomes that differ from the primary group: H2_B (all): 10/13 -> 10/12.

**TTV primary** (N = 11, with D0 = 9)

| rule | count / N | needed | verdict | tiles failing | values |
|---|---|---|---|---|---|
| F1 | 10 / 11 | 9 | holds | TTV11 | median R_MSE 2.59, range 0.68-6.11 |
| F2 | 11 / 11 | 9 | holds | - | median K/K0 P4 0.96, AD 10.09 |
| F3_A | 4 / 9 | 8 | fails | TTV1, TTV6, TTV7, TTV9, TTV11 | median D/D0 P4 16.14, AD 0.56 |
| F3_B | 4 / 9 | 8 | fails | TTV1, TTV6, TTV7, TTV9, TTV11 | median D/D0 P4 15.11, AD 0.56 |
| E1_A | 2 / 9 | 8 | fails | TTV1, TTV6, TTV7, TTV8, TTV9, TTV11, TTV15 |  |
| E1_B | 2 / 9 | 8 | fails | TTV1, TTV6, TTV7, TTV8, TTV9, TTV11, TTV15 |  |
| E2_P2_A | 2 / 9 | 8 | fails | TTV1, TTV6, TTV7, TTV8, TTV9, TTV11, TTV15 |  |
| E2_P2_B | 2 / 9 | 8 | fails | TTV1, TTV6, TTV7, TTV8, TTV9, TTV11, TTV15 |  |
| E2_P3 | 9 / 11 | 9 | holds | TTV9, TTV11 |  |
| F4 | A: median |log10(D/D0)| AD 1.10 (n 9); B: 1.05 (n 9) | - | holds | - | holds if >= 1.0 at both ends |
| H1 (all) | 10 / 11 | 9 | holds | TTV11 | lopsided tiles: TTV3, TTV9 |
| H1 (balanced) | 8 / 9 | 8 | holds | TTV11 |  |
| H2_A (all) | 5 / 9 | 8 | fails | TTV1, TTV6, TTV7, TTV11 | lopsided tiles: TTV3, TTV9 |
| H2_A (balanced) | 4 / 8 | 7 | fails | TTV1, TTV6, TTV7, TTV11 |  |
| H2_B (all) | 5 / 9 | 8 | fails | TTV1, TTV6, TTV7, TTV11 | lopsided tiles: TTV3, TTV9 |
| H2_B (balanced) | 4 / 8 | 7 | fails | TTV1, TTV6, TTV7, TTV11 |  |

**TTV clean** (N = 10, with D0 = 8)

| rule | count / N | needed | verdict | tiles failing | values |
|---|---|---|---|---|---|
| F1 | 10 / 10 | 8 | holds | - | median R_MSE 2.81, range 1.69-6.11 |
| F2 | 10 / 10 | 8 | holds | - | median K/K0 P4 1.00, AD 9.66 |
| F3_A | 4 / 8 | 7 | fails | TTV1, TTV6, TTV7, TTV9 | median D/D0 P4 15.62, AD 0.555 |
| F3_B | 4 / 8 | 7 | fails | TTV1, TTV6, TTV7, TTV9 | median D/D0 P4 15.10, AD 0.555 |
| E1_A | 2 / 8 | 7 | fails | TTV1, TTV6, TTV7, TTV8, TTV9, TTV15 |  |
| E1_B | 2 / 8 | 7 | fails | TTV1, TTV6, TTV7, TTV8, TTV9, TTV15 |  |
| E2_P2_A | 2 / 8 | 7 | fails | TTV1, TTV6, TTV7, TTV8, TTV9, TTV15 |  |
| E2_P2_B | 2 / 8 | 7 | fails | TTV1, TTV6, TTV7, TTV8, TTV9, TTV15 |  |
| E2_P3 | 9 / 10 | 8 | holds | TTV9 |  |
| F4 | A: median |log10(D/D0)| AD 0.99 (n 8); B: 0.95 (n 8) | - | fails | - | holds if >= 1.0 at both ends |
| H1 (all) | 10 / 10 | 8 | holds | - | lopsided tiles: TTV3, TTV9 |
| H1 (balanced) | 8 / 8 | 7 | holds | - |  |
| H2_A (all) | 5 / 8 | 7 | fails | TTV1, TTV6, TTV7 | lopsided tiles: TTV3, TTV9 |
| H2_A (balanced) | 4 / 7 | 6 | fails | TTV1, TTV6, TTV7 |  |
| H2_B (all) | 5 / 8 | 7 | fails | TTV1, TTV6, TTV7 | lopsided tiles: TTV3, TTV9 |
| H2_B (balanced) | 4 / 7 | 6 | fails | TTV1, TTV6, TTV7 |  |

Clean subset (6.7), outcomes that differ from the primary group: F4: holds -> fails.

### C12. Trunk sensitivity (5.4, 6.8): trunk cells (A >= 1 km2) held at the observed elevation and left out of the MSE

| tile | base group | trunk cells | fraction | A_crit base | A_crit trunk | K/K0 base | K/K0 trunk | D/D0_A base | D/D0_A trunk | R_MSE base | R_MSE trunk | changes | 6.8 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP3 | pamap | 1,521 | 0.15 % | 4742 | 4782 | 1.01 | 1.06 | 8.31 | 8.36 | 3.63 | 2.57 | K_over_K0 +5 % A_crit +1 % R_MSE -29 % D_over_D0A +1 % | material |
| AP3_3dep | primary | 1,514 | 0.15 % | 14529 | 14389 | 1.16 | 1.62 | 5.35 | 5.23 | 4.44 | 2.44 | K_over_K0 +40 % A_crit -1 % R_MSE -45 % D_over_D0A -2 % | material |
| AP4 | pamap | 213 | 0.02 % | 7232 | 7232 | 0.98 | 0.98 | 5.44 | 5.44 | 1.30 | 1.28 | K_over_K0 -0 % A_crit +0 % R_MSE -2 % D_over_D0A +0 % | not material |
| AP4_3dep | primary | 218 | 0.02 % | 5144 | 5144 | 1.02 | 1.02 | 3.48 | 3.48 | 1.42 | 1.39 | K_over_K0 -0 % A_crit +0 % R_MSE -2 % D_over_D0A +0 % | not material |
| GM2 | ncalm | 592 | 0.06 % | 3517 | 3509 | 1.01 | 1.01 | 2.80 | 2.80 | 2.93 | 2.90 | K_over_K0 +0 % A_crit -0 % R_MSE -1 % D_over_D0A -0 % | not material |
| GM2_fema | primary | 581 | 0.06 % | 1519 | 1519 | 0.98 | 0.98 | 1.86 | 1.88 | 2.82 | 2.81 | K_over_K0 -0 % A_crit +0 % R_MSE -1 % D_over_D0A +1 % | not material |
| GM4 | ncalm | 1,471 | 0.15 % | 3855 | 3850 | 1.06 | 1.02 | 2.18 | 2.26 | 4.00 | 4.08 | K_over_K0 -4 % A_crit -0 % R_MSE +2 % D_over_D0A +4 % | not material |
| GM4_fema | primary | 1,424 | 0.14 % | 2319 | 2319 | 1.09 | 1.11 | 1.68 | 1.72 | 3.49 | 3.35 | K_over_K0 +2 % A_crit +0 % R_MSE -4 % D_over_D0A +2 % | not material |
| GM5_fema | primary | 1,644 | 0.16 % | 2017 | 2017 | 1.03 | 1.03 | 2.40 | 2.43 | 2.53 | 2.57 | K_over_K0 -0 % A_crit +0 % R_MSE +2 % D_over_D0A +1 % | not material |

### C13. Continuity (5.5): the surveys against their twins at the site m, and GM1-GM5 (m = 0.31), AP1-AP5 (m = 0.47) against v2.0.0

**GM**: NCALM GM1-GM7 (group ncalm, m 0.29) against FEMA GM1_fema-GM7_fema (primary); ratio = FEMA / NCALM

| old | new | K (P4) | D (P4) | A_crit | MSE_P4 | MSE_AD | R_MSE | K (AD) | D (AD) |
|---|---|---|---|---|---|---|---|---|---|
| GM1:ncalm | GM1_fema:primary | 1.015 | 1.027 | 0.977 | 0.922 | 0.858 | 0.931 | 1.177 | 1.347 |
| GM2:ncalm | GM2_fema:primary | 1.030 | 0.851 | 0.432 | 0.916 | 0.882 | 0.964 | 1.231 | 2.016 |
| GM3:ncalm | GM3_fema:primary | 1.017 | 0.949 | 0.812 | 1.121 | 1.126 | 1.004 | 0.952 | 0.648 |
| GM4:ncalm | GM4_fema:primary | 1.053 | 0.925 | 0.602 | 0.959 | 0.837 | 0.873 | 1.189 | 1.612 |
| GM5:ncalm | GM5_fema:primary | 0.993 | 0.987 | 0.649 | 0.992 | 0.896 | 0.904 | 1.142 | 1.324 |
| GM6:ncalm | GM6_fema:primary | 0.924 | 1.610 | 2.936 | 1.127 | 0.964 | 0.855 | 1.258 | 3.259 |
| GM7:ncalm | GM7_fema:primary | 1.027 | 0.993 | 1.404 | 0.963 | 0.784 | 0.814 | 1.073 | 1.128 |
| median |  | 1.017 | 0.987 | 0.812 | 0.963 | 0.882 | 0.904 | 1.177 | 1.347 |
| largest |log10| (tile) |  | 0.03 (GM6) | 0.21 (GM6) | 0.47 (GM6) | 0.05 (GM6) | 0.11 (GM7) | 0.09 (GM7) | - | - |

**AP**: PAMAP AP1-AP5 (group pamap, m 0.47) against 3DEP AP1_3dep (ap1_3dep) and AP2_3dep-AP5_3dep (primary); ratio = 3DEP / PAMAP

| old | new | K (P4) | D (P4) | A_crit | MSE_P4 | MSE_AD | R_MSE | K (AD) | D (AD) |
|---|---|---|---|---|---|---|---|---|---|
| AP1:pamap | AP1_3dep:ap1_3dep | 1.041 | 1.051 | 1.220 | 0.980 | 1.103 | 1.125 | 1.107 | 1.304 |
| AP2:pamap | AP2_3dep:primary | 1.085 | 1.252 | 2.409 | 0.973 | 1.129 | 1.161 | 1.148 | 1.312 |
| AP3:pamap | AP3_3dep:primary | 1.195 | 1.307 | 3.064 | 0.903 | 1.106 | 1.224 | 1.185 | 1.528 |
| AP4:pamap | AP4_3dep:primary | 1.085 | 0.944 | 0.711 | 0.806 | 0.878 | 1.089 | 1.099 | 1.238 |
| AP5:pamap | AP5_3dep:primary | 1.009 | 1.163 | 1.393 | 1.242 | 1.162 | 0.936 | 1.121 | 1.123 |
| median |  | 1.085 | 1.163 | 1.393 | 0.973 | 1.106 | 1.125 | 1.121 | 1.304 |
| largest |log10| (tile) |  | 0.08 (AP3) | 0.12 (AP3) | 0.49 (AP3) | 0.09 (AP5) | 0.07 (AP5) | 0.09 (AP3) | - | - |

| spec | m | P4 rel diff vs v2.0.0 (union rows) | P4 rel diff (new rows only) | P4 match 1e-9 | AD rel diff | AD match 1e-9 | runs |
|---|---|---|---|---|---|---|---|
| GM1:v2m | 0.31 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 3.4e-15 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 3.4e-15 | yes | K 0.0e+00 D 0.0e+00 mse 0.0e+00 r 0.0e+00 | yes | 20261119T144900Z_a18_part_GM1_v2m / 20261119T144905Z_a18_ad_GM1_v2m |
| GM2:v2m | 0.31 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 2.2e-15 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 2.2e-15 | yes | K 0.0e+00 D 0.0e+00 mse 0.0e+00 r 0.0e+00 | yes | 20261119T212133Z_a18_part_GM2_v2m / 20261119T211606Z_a18_ad_GM2_v2m |
| GM3:v2m | 0.31 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 1.3e-15 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 1.3e-15 | yes | K 0.0e+00 D 0.0e+00 mse 0.0e+00 r 0.0e+00 | yes | 20261119T211937Z_a18_part_GM3_v2m / 20261119T211606Z_a18_ad_GM3_v2m |
| GM4:v2m | 0.31 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 1.4e-14 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 1.4e-14 | yes | K 0.0e+00 D 0.0e+00 mse 0.0e+00 r 0.0e+00 | yes | 20261119T211937Z_a18_part_GM4_v2m / 20261119T211606Z_a18_ad_GM4_v2m |
| GM5:v2m | 0.31 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 2.6e-14 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 2.6e-14 | yes | K 2.2e-16 D 0.0e+00 mse 0.0e+00 r 0.0e+00 | yes | 20261119T212102Z_a18_part_GM5_v2m / 20261119T211608Z_a18_ad_GM5_v2m |
| AP1:pamap | 0.47 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 7.7e-15 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 7.7e-15 | yes | K 0.0e+00 D 0.0e+00 mse 0.0e+00 r 0.0e+00 | yes | 20261119T211951Z_a18_part_AP1_pamap / 20261119T211944Z_a18_ad_AP1_pamap |
| AP2:pamap | 0.47 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 3.8e-15 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 3.8e-15 | yes | K 0.0e+00 D 0.0e+00 mse 0.0e+00 r 0.0e+00 | yes | 20261119T144900Z_a18_part_AP2_pamap / 20261119T144905Z_a18_ad_AP2_pamap |
| AP3:pamap | 0.47 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 2.7e-15 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 2.7e-15 | yes | K 1.1e-16 D 0.0e+00 mse 0.0e+00 r 0.0e+00 | yes | 20261119T211946Z_a18_part_AP3_pamap / 20261119T211944Z_a18_ad_AP3_pamap |
| AP4:pamap | 0.47 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 0.0e+00 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 0.0e+00 | yes | K 0.0e+00 D 0.0e+00 mse 0.0e+00 r 0.0e+00 | yes | 20261119T211946Z_a18_part_AP4_pamap / 20261119T211944Z_a18_ad_AP4_pamap |
| AP5:pamap | 0.47 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 1.4e-14 | A_crit 0.0e+00 K 0.0e+00 D 0.0e+00 mse 1.4e-14 | yes | K 0.0e+00 D 0.0e+00 mse 0.0e+00 r 0.0e+00 | yes | 20261119T211946Z_a18_part_AP5_pamap / 20261119T211946Z_a18_ad_AP5_pamap |

### C14. TTV per-tile m (5.6, 6.9): the fits at m_t beside the site m

| tile | which | m | K0 | A_crit | 1 % interval | K/K0 | D/D0_A | D/D0_B | MSE_P4 | MSE_AD | R_MSE | AD K/K0 | AD D/D0_A |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | site | 0.33 | 1.124e-05 | 59499 | 59066-60312 | 1.10 | 15.11 | 15.11 | 82.73 | 214.19 | 2.59 | 9.23 | 0.551 |
| TTV1 | m_t | 0.05 (edge) | 2.226e-04 | 59499 | 59152-60312 | 1.58 | 15.08 | 15.08 | 80.44 | 343.26 | 4.27 | 0.98 | 0.000622 |
| TTV2 | site | 0.33 | 1.581e-05 | 9131 | 8399-9159 | 1.04 | 3.72 | 3.53 | 59.79 | 209.66 | 3.51 | 4.82 | 0.053 |
| TTV2 | m_t | 0.45 | 4.059e-06 | 8607 | 8467-9153 | 0.92 | 4.06 | 3.85 | 50.02 | 127.00 | 2.54 | 12.96 | 0.25 |
| TTV3 | site | 0.33 | 2.636e-05 | 89124 | 89070-89144 | 1.42 | - | - | 87.40 | 326.85 | 3.74 | 1.96 | - |
| TTV3 | m_t | 0.40 | 1.215e-05 | 89124 | 89070-89144 | 1.17 | - | - | 87.40 | 362.67 | 4.15 | 8.16 | - |
| TTV4 | site | 0.33 | 1.614e-05 | 40309 | 38925-40734 | 0.96 | - | - | 96.82 | 293.97 | 3.04 | 10.09 | - |
| TTV4 | m_t | 0.42 | 6.215e-06 | 40309 | 38925-42304 | 0.83 | - | - | 96.21 | 201.65 | 2.10 | 15.88 | - |
| TTV5 | site | 0.33 | 2.176e-05 | 58612 | 58350-60066 | 1.22 | 20.02 | 18.52 | 137.88 | 842.95 | 6.11 | 2.78 | 0.0208 |
| TTV5 | m_t | 0.41 | 8.991e-06 | 58612 | 58350-60066 | 1.09 | 20.02 | 18.52 | 138.00 | 693.43 | 5.02 | 32.96 | 20 |
| TTV6 | site | 0.33 | 1.244e-05 | 10973 | 10344-11683 | 0.76 | 16.69 | 16.69 | 94.27 | 204.34 | 2.17 | 7.60 | 0.56 |
| TTV6 | m_t | 0.05 (edge) | 2.198e-04 | 14492 | 14370-14498 | 1.04 | 15.88 | 15.88 | 79.23 | 201.42 | 2.54 | 1.04 | 0.00313 |
| TTV7 | site | 0.33 | 1.079e-05 | 35966 | 35470-36940 | 0.75 | 16.14 | 15.09 | 79.24 | 134.26 | 1.69 | 20.90 | 7.65 |
| TTV7 | m_t | 0.05 (edge) | 2.041e-04 | 35966 | 35721-36727 | 0.99 | 16.14 | 15.09 | 63.76 | 167.13 | 2.62 | 2.86 | 12.7 |
| TTV8 | site | 0.33 | 1.150e-05 | 38544 | 28341-40094 | 0.73 | 11.44 | 10.17 | 114.67 | 272.96 | 2.38 | 13.20 | 12.5 |
| TTV8 | m_t | 0.06 (edge) | 1.966e-04 | 39096 | 38165-39653 | 0.98 | 11.10 | 9.88 | 112.09 | 273.07 | 2.44 | 1.03 | 0.00157 |
| TTV9 | site | 0.33 | 1.053e-05 | 36561 | 33976-39662 | 0.73 | 10.42 | 7.00 | 115.18 | 284.91 | 2.47 | 11.35 | 0.467 |
| TTV9 | m_t | 0.35 | 8.397e-06 | 36561 | 35510-39867 | 0.69 | 10.68 | 7.18 | 114.49 | 264.97 | 2.31 | 13.22 | 0.59 |
| TTV11 | site | 0.33 | 1.159e-05 | 30362 | 16982-31139 | 0.69 | 17.77 | 16.06 | 241.77 | 164.20 | 0.68 | 12.53 | 13.9 |
| TTV11 | m_t | 0.05 (edge) | 2.037e-04 | 29923 | 16982-30550 | 0.89 | 18.27 | 16.51 | 182.28 | 176.97 | 0.97 | 1.65 | 15.7 |
| TTV15 | site | 0.33 | 1.372e-05 | 26819 | 26314-27056 | 1.11 | 18.98 | 18.98 | 103.42 | 327.53 | 3.17 | 15.39 | 33.2 |
| TTV15 | m_t | 0.05 (edge) | 2.915e-04 | 26938 | 26699-27525 | 1.41 | 18.55 | 18.55 | 97.36 | 362.41 | 3.72 | 1.72 | 33.2 |

Rule 6.9: F1 at m_t 10 / 11 (holds); medians site m -> m_t: D_over_D0A 16.1 -> 15.9 (-2 %); AD_K_over_K0 10.1 -> 2.86 (-72 %); K_over_K0 0.956 -> 0.987 (+3 %); AD_D_over_D0A 0.56 -> 0.59 (+5 %); D_over_D0B 15.1 -> 15.1 (-0 %); the site-m choice is not immaterial at TTV.

| tile | m_t | grid edge | k_s at m_t | 95 % | n_eff | basins | K0(m_t) | K0(site m) | k_s check at site m (rel) |
|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 0.05 | yes | 0.3459 | [0.3213, 0.4628] | 3.4 | 23 | 2.226e-04 | 1.124e-05 | 0.0e+00 |
| TTV2 | 0.45 | no | 18.97 | [16.11, 21.77] | 1.4 | 16 | 4.059e-06 | 1.581e-05 | 0.0e+00 |
| TTV3 | 0.40 | no | 6.337 | [6.221, 9.372] | 1.2 | 7 | 1.215e-05 | 2.636e-05 | 0.0e+00 |
| TTV4 | 0.42 | no | 12.39 | [6.988, 13.09] | 1.9 | 24 | 6.215e-06 | 1.614e-05 | 0.0e+00 |
| TTV5 | 0.41 | no | 8.564 | [6.279, 16.77] | 1.9 | 22 | 8.991e-06 | 2.176e-05 | 0.0e+00 |
| TTV6 | 0.05 | yes | 0.3504 | [0.2945, 0.3881] | 2.6 | 43 | 2.198e-04 | 1.244e-05 | 0.0e+00 |
| TTV7 | 0.05 | yes | 0.3772 | [0.3151, 0.4122] | 2.9 | 20 | 2.041e-04 | 1.079e-05 | 0.0e+00 |
| TTV8 | 0.06 | yes | 0.3917 | [0.3619, 0.4888] | 2.3 | 23 | 1.966e-04 | 1.150e-05 | 0.0e+00 |
| TTV9 | 0.35 | no | 9.17 | [5.663, 9.776] | 1.6 | 13 | 8.397e-06 | 1.053e-05 | 0.0e+00 |
| TTV11 | 0.05 | yes | 0.3781 | [0.3529, 0.406] | 4.7 | 28 | 2.037e-04 | 1.159e-05 | 0.0e+00 |
| TTV15 | 0.05 | yes | 0.2641 | [0.2206, 0.4065] | 2.2 | 26 | 2.915e-04 | 1.372e-05 | 0.0e+00 |

### C15. TTV channel-head comparison (5.7)

| distribution (m2) | n | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|
| M&D (1989) Table 1 source areas | 63 | 2010 | 4800 | 7400 | 10350 | 14860 |
| 77 digitised heads, largest A within 3 m | 77 | 32 | 76 | 132 | 304 | 2298 |
| 77 digitised heads, A at the head cell | 77 | 1 | 2 | 6 | 30 | 256 |
| 77 digitised heads, largest A within 3m (sensitivity) | 77 | 32 | 76 | 132 | 304 | 2298 |
| 77 digitised heads, largest A within 10m (sensitivity) | 77 | 116 | 189 | 355 | 843 | 7495 |
| 77 digitised heads, largest A within 20m (sensitivity) | 77 | 177 | 354 | 903 | 2996 | 17109 |

Medians: 77 heads 132 m2 against Table 1 7400 m2, ratio 0.018; sensitivity medians {'3m': 132.0, '10m': 355.0, '20m': 903.0}.

**At site m**: Table 1 IQR [4800.0, 10350.0] m2; 77-head IQR [76.0, 304.0] m2

| tile | m | P4 A_crit | 1 % interval | 5 % interval | class vs Table 1 IQR | class vs 77-head IQR |
|---|---|---|---|---|---|---|
| TTV1 (M&D area) | 0.33 | 59499 | 59066-60312 | 48902-62092 | above | above |
| TTV2 (M&D area) | 0.33 | 9131 | 8399-9159 | 8279-9191 | consistent | above |
| TTV3 (M&D area) | 0.33 | 89124 | 89070-89144 | 88198-95118 | above | above |
| TTV4 | 0.33 | 40309 | 38925-40734 | 38153-44997 | above | above |
| TTV5 | 0.33 | 58612 | 58350-60066 | 58248-61531 | above | above |
| TTV6 | 0.33 | 10973 | 10344-11683 | 9403-15186 | consistent | above |
| TTV7 | 0.33 | 35966 | 35470-36940 | 32608-39469 | above | above |
| TTV8 | 0.33 | 38544 | 28341-40094 | 21256-45698 | above | above |
| TTV9 | 0.33 | 36561 | 33976-39662 | 31421-64483 | above | above |
| TTV11 | 0.33 | 30362 | 16982-31139 | 15044-35960 | above | above |
| TTV15 | 0.33 | 26819 | 26314-27056 | 18316-28690 | above | above |

Counts (of 11 fitted, 11 tiles): {'consistent': 2, 'above': 9, 'below': 0}; TTV1-TTV3: vs Table 1 {'consistent': 1, 'above': 2, 'below': 0}, vs the 77 heads {'consistent': 0, 'above': 3, 'below': 0} (N 3); reading: **lies above the field channel heads at the 1 km tile scale**.

**At m t**: Table 1 IQR [4800.0, 10350.0] m2; 77-head IQR [76.0, 304.0] m2

| tile | m | P4 A_crit | 1 % interval | 5 % interval | class vs Table 1 IQR | class vs 77-head IQR |
|---|---|---|---|---|---|---|
| TTV1 (M&D area) | 0.05 | 59499 | 59152-60312 | 55874-62092 | above | above |
| TTV2 (M&D area) | 0.45 | 8607 | 8467-9153 | 6744-9191 | consistent | above |
| TTV3 (M&D area) | 0.40 | 89124 | 89070-89144 | 87840-99994 | above | above |
| TTV4 | 0.42 | 40309 | 38925-42304 | 38153-44997 | above | above |
| TTV5 | 0.41 | 58612 | 58350-60066 | 58248-61861 | above | above |
| TTV6 | 0.05 | 14492 | 14370-14498 | 14123-16664 | above | above |
| TTV7 | 0.05 | 35966 | 35721-36727 | 27381-38014 | above | above |
| TTV8 | 0.06 | 39096 | 38165-39653 | 21550-43825 | above | above |
| TTV9 | 0.35 | 36561 | 35510-39867 | 32876-64617 | above | above |
| TTV11 | 0.05 | 29923 | 16982-30550 | 13727-33893 | above | above |
| TTV15 | 0.05 | 26938 | 26699-27525 | 18341-28690 | above | above |

Counts (of 11 fitted, 11 tiles): {'consistent': 1, 'above': 10, 'below': 0}; TTV1-TTV3: vs Table 1 {'consistent': 1, 'above': 2, 'below': 0}, vs the 77 heads {'consistent': 0, 'above': 3, 'below': 0} (N 3); reading: **lies above the field channel heads at the 1 km tile scale**.

### C16. Run ids, git hashes, Slurm jobs and input checksums of every fit

| spec | m | K0 | D0_A | D0_B | trunk | partitioned run | AD run | git | jobs part / ad | evals part | evals AD | wall part (h) | wall AD (h) | fill capped | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP5:pamap | 0.29 | 6.166e-05 | 0.01025 | 0.01093 | no | 20261119T211515Z_a18_part_GM1_fema_primary | 20261119T211515Z_a18_ad_GM1_fema_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52989273 / 52989720 | 2,106 | 264 | 3.27 | 2.05 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 7.276e-05 | 0.01251 | 0.01197 | no | 20261119T211515Z_a18_part_GM2_fema_primary | 20261119T211515Z_a18_ad_GM2_fema_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52989274 / 52989721 | 2,112 | 268 | 3.63 | 2.17 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 8.045e-05 | 0.01545 | 0.01583 | no | 20261119T212102Z_a18_part_GM3_fema_primary | 20261119T211531Z_a18_ad_GM3_fema_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990488 / 52989722 | 2,236 | 286 | 4.23 | 2.06 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 4.540e-05 | 0.01178 | 0.01097 | no | 20261119T212103Z_a18_part_GM4_fema_primary | 20261119T211532Z_a18_ad_GM4_fema_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990489 / 52989723 | 2,231 | 261 | 4.31 | 1.87 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 6.490e-05 | 0.008725 | 0.008471 | no | 20261119T212102Z_a18_part_GM5_fema_primary | 20261119T211608Z_a18_ad_GM5_fema_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990490 / 52989724 | 2,270 | 266 | 4.02 | 1.87 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 6.847e-05 | 0.01625 | 0.01432 | no | 20261119T212102Z_a18_part_GM6_fema_primary | 20261119T211531Z_a18_ad_GM6_fema_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990491 / 52989725 | 2,493 | 269 | 4.62 | 2.11 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 8.255e-05 | 0.03544 | 0.03574 | no | 20261119T212102Z_a18_part_GM7_fema_primary | 20261119T211608Z_a18_ad_GM7_fema_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990492 / 52989726 | 2,847 | 266 | 5.67 | 2.07 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 8.457e-05 | 0.01523 | 0.01564 | no | 20261119T212102Z_a18_part_GM8_primary | 20261119T211532Z_a18_ad_GM8_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990493 / 52989727 | 2,319 | 268 | 4.19 | 2.13 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 8.281e-05 | 0.02036 | 0.0211 | no | 20261119T212102Z_a18_part_GM9_primary | 20261119T211604Z_a18_ad_GM9_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990494 / 52989728 | 2,635 | 267 | 4.80 | 2.10 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 8.016e-05 | 0.01831 | 0.01974 | no | 20261119T212102Z_a18_part_GM11_primary | 20261119T211531Z_a18_ad_GM11_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990495 / 52989729 | 2,214 | 283 | 3.99 | 2.21 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 7.500e-05 | 0.01144 | 0.01273 | no | 20261119T212102Z_a18_part_GM12_primary | 20261119T211604Z_a18_ad_GM12_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990496 / 52989730 | 2,296 | 274 | 4.37 | 2.23 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 6.670e-05 | 0.02885 | 0.02786 | no | 20261119T212102Z_a18_part_GM13_primary | 20261119T211531Z_a18_ad_GM13_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990497 / 52989731 | 2,425 | 269 | 4.80 | 2.10 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 6.475e-05 | 0.00898 | 0.01073 | no | 20261119T212102Z_a18_part_GM15_primary | 20261119T211611Z_a18_ad_GM15_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990498 / 52989732 | 2,677 | 268 | 4.57 | 2.04 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 6.660e-05 | 0.01218 | 0.01037 | no | 20261119T212003Z_a18_part_GM16_primary | 20261119T211515Z_a18_ad_GM16_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990499 / 52989733 | 2,114 | 268 | 3.80 | 2.04 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 7.158e-05 | 0.01552 | 0.01987 | no | 20261119T212003Z_a18_part_GM17_primary | 20261119T211515Z_a18_ad_GM17_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990500 / 52989734 | 2,250 | 267 | 3.79 | 2.03 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 8.482e-05 | 0.01825 | 0.01929 | no | 20261119T212003Z_a18_part_GM20_primary | 20261119T211515Z_a18_ad_GM20_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990501 / 52989735 | 2,549 | 263 | 4.05 | 2.00 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 6.205e-05 | 0.01193 | 0.01274 | no | 20261119T212003Z_a18_part_GM21_primary | 20261119T211515Z_a18_ad_GM21_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990502 / 52989736 | 2,361 | 285 | 4.47 | 2.18 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 7.056e-05 | 0.01609 | 0.01888 | no | 20261119T212008Z_a18_part_GM22_primary | 20261119T211606Z_a18_ad_GM22_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990503 / 52989737 | 2,419 | 274 | 4.24 | 2.04 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 7.790e-05 | 0.03127 | 0.03167 | no | 20261119T212003Z_a18_part_GM24_primary | 20261119T211606Z_a18_ad_GM24_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990504 / 52989738 | 2,849 | 267 | 5.44 | 2.13 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 9.138e-05 | 0.0174 | 0.01901 | no | 20261119T211937Z_a18_part_GM26_primary | 20261119T211606Z_a18_ad_GM26_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990505 / 52989739 | 2,683 | 265 | 4.13 | 2.08 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 7.589e-05 | 0.00728 | 0.008165 | no | 20261119T211949Z_a18_part_GM27_primary | 20261119T211606Z_a18_ad_GM27_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990506 / 52989740 | 2,281 | 286 | 3.41 | 2.12 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 6.981e-05 | 0.01455 | 0.0139 | no | 20261119T211949Z_a18_part_GM29_primary | 20261119T211606Z_a18_ad_GM29_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52956702 / 52956742 | 2,176 | 273 | 3.82 | 2.14 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 5.047e-06 | - | - | no | 20261119T212010Z_a18_part_AP2_3dep_primary | 20261119T211608Z_a18_ad_AP2_3dep_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52989791 / 52989744 | 2,266 | 247 | 4.07 | 1.90 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.984e-06 | 0.004372 | 0.005272 | no | 20261119T212009Z_a18_part_AP3_3dep_primary | 20261119T211608Z_a18_ad_AP3_3dep_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52989792 / 52989745 | 2,171 | 267 | 3.72 | 2.14 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.811e-06 | 0.00536 | 0.007554 | no | 20261119T212151Z_a18_part_AP4_3dep_primary | 20261119T211608Z_a18_ad_AP4_3dep_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990440 / 52989746 | 2,371 | 267 | 4.24 | 2.13 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 5.299e-06 | 0.005144 | 0.006494 | no | 20261119T212151Z_a18_part_AP5_3dep_primary | 20261119T211608Z_a18_ad_AP5_3dep_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990441 / 52989747 | 2,390 | 267 | 3.99 | 2.11 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.720e-06 | 0.005952 | 0.007218 | no | 20261119T212151Z_a18_part_AP6_primary | 20261119T211608Z_a18_ad_AP6_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990442 / 52989748 | 2,500 | 269 | 4.43 | 2.16 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 5.364e-06 | 0.005239 | 0.005771 | no | 20261119T212151Z_a18_part_AP7_primary | 20261119T211608Z_a18_ad_AP7_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990443 / 52989749 | 2,455 | 260 | 4.68 | 2.06 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 5.463e-06 | 0.006024 | 0.007696 | no | 20261119T212151Z_a18_part_AP8_primary | 20261119T211608Z_a18_ad_AP8_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990444 / 52989750 | 2,527 | 270 | 4.87 | 2.33 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.583e-06 | - | - | no | 20261119T212151Z_a18_part_AP9_primary | 20261119T211608Z_a18_ad_AP9_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990445 / 52989751 | 2,242 | 243 | 4.21 | 2.09 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.173e-06 | - | - | no | 20261119T212151Z_a18_part_AP10_primary | 20261119T211608Z_a18_ad_AP10_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990446 / 52989752 | 2,107 | 249 | 3.90 | 2.07 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.085e-06 | - | - | no | 20261119T212151Z_a18_part_AP11_primary | 20261119T211608Z_a18_ad_AP11_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990447 / 52989753 | 2,676 | 246 | 5.00 | 1.93 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 5.382e-06 | 0.004471 | 0.007562 | no | 20261119T212151Z_a18_part_AP13_primary | 20261119T211608Z_a18_ad_AP13_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990448 / 52989754 | 2,354 | 267 | 4.22 | 2.11 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.199e-06 | - | - | no | 20261119T212151Z_a18_part_AP14_primary | 20261119T211608Z_a18_ad_AP14_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990449 / 52989755 | 2,321 | 249 | 4.91 | 1.93 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 5.275e-06 | - | - | no | 20261119T212151Z_a18_part_AP18_primary | 20261119T211515Z_a18_ad_AP18_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990450 / 52989756 | 2,350 | 249 | 4.16 | 1.88 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.303e-06 | 0.006118 | 0.007877 | no | 20261119T211937Z_a18_part_AP19_primary | 20261119T211515Z_a18_ad_AP19_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990451 / 52989757 | 2,687 | 264 | 4.52 | 2.02 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.765e-06 | 0.0054 | 0.007204 | no | 20261119T211937Z_a18_part_AP20_primary | 20261119T211515Z_a18_ad_AP20_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990452 / 52989758 | 2,659 | 265 | 4.82 | 2.04 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 5.194e-06 | 0.003661 | 0.004505 | no | 20261119T211937Z_a18_part_AP23_primary | 20261119T211515Z_a18_ad_AP23_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990453 / 52989759 | 2,518 | 265 | 4.12 | 2.12 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.540e-06 | 0.005939 | 0.008242 | no | 20261119T211937Z_a18_part_AP24_primary | 20261119T211515Z_a18_ad_AP24_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990454 / 52989760 | 2,555 | 267 | 4.28 | 2.10 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.779e-06 | 0.003859 | 0.004573 | no | 20261119T212010Z_a18_part_AP25_primary | 20261119T211515Z_a18_ad_AP25_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990455 / 52989761 | 2,184 | 264 | 3.49 | 2.26 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.774e-06 | - | - | no | 20261119T212010Z_a18_part_AP27_primary | 20261119T211515Z_a18_ad_AP27_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990456 / 52989762 | 2,436 | 242 | 4.19 | 1.97 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.384e-06 | - | - | no | 20261119T212010Z_a18_part_AP28_primary | 20261119T211516Z_a18_ad_AP28_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990457 / 52989763 | 2,173 | 247 | 3.74 | 2.02 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 5.117e-06 | 0.006297 | 0.006704 | no | 20261119T212010Z_a18_part_AP29_primary | 20261119T211516Z_a18_ad_AP29_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52956704 / 52956747 | 2,313 | 267 | 4.13 | 2.22 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.33 | 1.124e-05 | 0.002905 | 0.002905 | no | 20261119T212010Z_a18_part_TTV1_primary | 20261119T212010Z_a18_ad_TTV1_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990458 / 52990413 | 2,419 | 263 | 4.15 | 2.25 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.33 | 1.581e-05 | 0.004461 | 0.004708 | no | 20261119T212010Z_a18_part_TTV2_primary | 20261119T212010Z_a18_ad_TTV2_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990459 / 52990414 | 2,423 | 265 | 4.08 | 2.20 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.33 | 2.636e-05 | - | - | no | 20261119T212010Z_a18_part_TTV3_primary | 20261119T212010Z_a18_ad_TTV3_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990460 / 52990415 | 2,464 | 249 | 4.06 | 2.16 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.33 | 1.614e-05 | - | - | no | 20261119T212010Z_a18_part_TTV4_primary | 20261119T212010Z_a18_ad_TTV4_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990461 / 52990416 | 2,107 | 247 | 3.72 | 2.05 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.33 | 2.176e-05 | 0.002498 | 0.002699 | no | 20261119T212134Z_a18_part_TTV5_primary | 20261119T212010Z_a18_ad_TTV5_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990462 / 52990417 | 2,182 | 290 | 4.12 | 2.55 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.33 | 1.244e-05 | 0.001993 | 0.001993 | no | 20261119T212134Z_a18_part_TTV6_primary | 20261119T212010Z_a18_ad_TTV6_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990463 / 52990418 | 2,696 | 259 | 5.18 | 2.14 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.33 | 1.079e-05 | 0.003098 | 0.003314 | no | 20261119T212134Z_a18_part_TTV7_primary | 20261119T211950Z_a18_ad_TTV7_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990464 / 52990419 | 2,400 | 269 | 4.54 | 2.09 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.33 | 1.150e-05 | 0.003984 | 0.004479 | no | 20261119T212134Z_a18_part_TTV8_primary | 20261119T212110Z_a18_ad_TTV8_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990465 / 52990420 | 2,918 | 272 | 5.99 | 2.34 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.33 | 1.053e-05 | 0.004202 | 0.006252 | no | 20261119T212134Z_a18_part_TTV9_primary | 20261119T212110Z_a18_ad_TTV9_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990466 / 52990421 | 3,445 | 268 | 6.21 | 2.10 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.33 | 1.159e-05 | 0.002312 | 0.002558 | no | 20261119T212134Z_a18_part_TTV11_primary | 20261119T212110Z_a18_ad_TTV11_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990467 / 52990422 | 3,385 | 264 | 7.10 | 2.37 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.33 | 1.372e-05 | 0.001506 | 0.001506 | no | 20261119T212134Z_a18_part_TTV15_primary | 20261119T211944Z_a18_ad_TTV15_primary | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52956711 / 52956751 | 2,199 | 258 | 4.10 | 1.98 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 6.918e-05 | 0.009355 | 0.0117 | no | 20261119T212151Z_a18_part_GM14_offmesa | 20261119T211516Z_a18_ad_GM14_offmesa | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990435 / 52989764 | 2,801 | 283 | 4.69 | 2.37 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 1.003e-04 | 0.005326 | 0.006456 | no | 20261119T212151Z_a18_part_GM18_offmesa | 20261119T211516Z_a18_ad_GM18_offmesa | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990436 / 52989765 | 2,145 | 269 | 3.44 | 2.30 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 6.155e-05 | 0.01241 | 0.012 | no | 20261119T212151Z_a18_part_GM19_offmesa | 20261119T211516Z_a18_ad_GM19_offmesa | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990437 / 52989766 | 2,630 | 269 | 4.92 | 2.24 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 6.522e-05 | 0.01061 | 0.01185 | no | 20261119T212151Z_a18_part_GM25_offmesa | 20261119T211516Z_a18_ad_GM25_offmesa | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990438 / 52989767 | 2,408 | 274 | 3.87 | 2.20 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 6.496e-05 | 0.01462 | 0.01718 | no | 20261119T212151Z_a18_part_GM28_offmesa | 20261119T211516Z_a18_ad_GM28_offmesa | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990439 / 52989768 | 2,505 | 269 | 4.62 | 2.21 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 9.642e-05 | 0.01114 | 0.01171 | no | 20261119T212151Z_a18_part_GM30_offmesa | 20261119T211516Z_a18_ad_GM30_offmesa | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52956738 / 52956779 | 2,185 | 264 | 3.76 | 2.10 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 6.096e-05 | 0.009001 | 0.009354 | no | 20261119T211946Z_a18_part_GM1_ncalm | 20261119T212010Z_a18_ad_GM1_ncalm | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990429 / 52990407 | 2,147 | 265 | 3.29 | 2.25 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.31 | 4.854e-05 | 0.009001 | 0.009354 | no | 20261119T144900Z_a18_part_GM1_v2m | 20261119T144905Z_a18_ad_GM1_v2m | e85221b3930e3fb649cd20cad7082139523799f0-dirty | 52955499 / 52955501 | 2,143 | 266 | 3.53 | 1.99 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 6.888e-05 | 0.00977 | 0.01015 | no | 20261119T211947Z_a18_part_GM2_ncalm | 20261119T212010Z_a18_ad_GM2_ncalm | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990430 / 52990408 | 2,090 | 269 | 3.45 | 2.31 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.31 | 5.514e-05 | 0.00977 | 0.01015 | no | 20261119T212133Z_a18_part_GM2_v2m | 20261119T211606Z_a18_ad_GM2_v2m | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990475 / 52989741 | 2,090 | 262 | 3.84 | 2.06 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 7.981e-05 | 0.01012 | 0.01012 | no | 20261119T211947Z_a18_part_GM3_ncalm | 20261119T212009Z_a18_ad_GM3_ncalm | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990431 / 52990409 | 2,228 | 267 | 3.63 | 2.26 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.31 | 6.413e-05 | 0.01012 | 0.01012 | no | 20261119T211937Z_a18_part_GM3_v2m | 20261119T211606Z_a18_ad_GM3_v2m | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990476 / 52989742 | 2,275 | 262 | 3.77 | 2.07 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 4.425e-05 | 0.009855 | 0.009435 | no | 20261119T211949Z_a18_part_GM4_ncalm | 20261119T212009Z_a18_ad_GM4_ncalm | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990432 / 52990410 | 2,130 | 268 | 3.90 | 2.31 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.31 | 3.479e-05 | 0.009855 | 0.009435 | no | 20261119T211937Z_a18_part_GM4_v2m | 20261119T211606Z_a18_ad_GM4_v2m | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990477 / 52989743 | 2,146 | 264 | 3.74 | 2.12 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 6.389e-05 | 0.007589 | 0.007496 | no | 20261119T211951Z_a18_part_GM5_ncalm | 20261119T212009Z_a18_ad_GM5_ncalm | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990433 / 52990411 | 2,325 | 266 | 3.66 | 2.26 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.31 | 5.019e-05 | 0.007589 | 0.007496 | no | 20261119T212102Z_a18_part_GM5_v2m | 20261119T211608Z_a18_ad_GM5_v2m | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52956734 / 52956773 | 2,319 | 271 | 4.12 | 2.11 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 6.920e-05 | 0.01094 | 0.01031 | no | 20261119T211951Z_a18_part_GM6_ncalm | 20261119T212010Z_a18_ad_GM6_ncalm | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990434 / 52990412 | 2,259 | 267 | 3.76 | 2.29 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 8.025e-05 | 0.02203 | 0.02203 | no | 20261119T212003Z_a18_part_GM7_ncalm | 20261119T212010Z_a18_ad_GM7_ncalm | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52956726 / 52956760 | 2,233 | 259 | 4.21 | 2.22 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.428e-06 | 0.003587 | 0.006978 | no | 20261119T211951Z_a18_part_AP1_pamap | 20261119T211944Z_a18_ad_AP1_pamap | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990426 / 52990423 | 4,124 | 277 | 6.58 | 2.28 | yes | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.748e-06 | 0.003218 | 0.004078 | no | 20261119T144900Z_a18_part_AP2_pamap | 20261119T144905Z_a18_ad_AP2_pamap | e85221b3930e3fb649cd20cad7082139523799f0-dirty | 52955500 / 52954921 | 4,047 | 266 | 6.27 | 2.00 | yes | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.770e-06 | 0.002155 | 0.002155 | no | 20261119T211946Z_a18_part_AP3_pamap | 20261119T211944Z_a18_ad_AP3_pamap | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990427 / 52990424 | 3,815 | 259 | 5.45 | 2.19 | yes | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.610e-06 | 0.003629 | 0.00772 | no | 20261119T211946Z_a18_part_AP4_pamap | 20261119T211944Z_a18_ad_AP4_pamap | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990428 / 52990425 | 4,269 | 266 | 6.69 | 2.22 | yes | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 5.128e-06 | 0.002921 | 0.003456 | no | 20261119T211946Z_a18_part_AP5_pamap | 20261119T211946Z_a18_ad_AP5_pamap | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52956729 / 52956765 | 4,170 | 267 | 6.19 | 2.23 | yes | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.482e-06 | 0.005507 | 0.006664 | no | 20261119T212003Z_a18_part_AP1_3dep_ap1_3dep | 20261119T212010Z_a18_ad_AP1_3dep_ap1_3dep | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52956731 / 52956770 | 2,540 | 269 | 3.94 | 2.20 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.05 | 2.226e-04 | 0.002905 | 0.002905 | no | 20261119T212102Z_a18_part_TTV1_mt | 20261119T211516Z_a18_ad_TTV1_mt | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990478 / 52989769 | 2,226 | 268 | 4.22 | 2.24 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.45 | 4.059e-06 | 0.004461 | 0.004708 | no | 20261119T212102Z_a18_part_TTV2_mt | 20261119T211516Z_a18_ad_TTV2_mt | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990479 / 52989770 | 2,677 | 263 | 4.73 | 2.06 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.40 | 1.215e-05 | - | - | no | 20261119T212102Z_a18_part_TTV3_mt | 20261119T211516Z_a18_ad_TTV3_mt | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990480 / 52989771 | 2,484 | 249 | 4.38 | 2.01 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.42 | 6.215e-06 | - | - | no | 20261119T212102Z_a18_part_TTV4_mt | 20261119T211516Z_a18_ad_TTV4_mt | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990481 / 52989772 | 2,104 | 249 | 3.91 | 1.92 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.41 | 8.991e-06 | 0.002498 | 0.002699 | no | 20261119T212102Z_a18_part_TTV5_mt | 20261119T211533Z_a18_ad_TTV5_mt | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990482 / 52989773 | 2,210 | 269 | 4.25 | 2.20 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.05 | 2.198e-04 | 0.001993 | 0.001993 | no | 20261119T212102Z_a18_part_TTV6_mt | 20261119T211515Z_a18_ad_TTV6_mt | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990483 / 52989774 | 2,136 | 253 | 4.07 | 1.88 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.05 | 2.041e-04 | 0.003098 | 0.003314 | no | 20261119T212102Z_a18_part_TTV7_mt | 20261119T211515Z_a18_ad_TTV7_mt | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990484 / 52989775 | 2,182 | 258 | 4.25 | 2.17 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.06 | 1.966e-04 | 0.003984 | 0.004479 | no | 20261119T212102Z_a18_part_TTV8_mt | 20261119T211515Z_a18_ad_TTV8_mt | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990485 / 52989776 | 2,328 | 262 | 4.70 | 2.31 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.35 | 8.397e-06 | 0.004202 | 0.006252 | no | 20261119T212102Z_a18_part_TTV9_mt | 20261119T211533Z_a18_ad_TTV9_mt | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990486 / 52989777 | 3,454 | 266 | 6.29 | 1.91 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.05 | 2.037e-04 | 0.002312 | 0.002558 | no | 20261119T212102Z_a18_part_TTV11_mt | 20261119T211545Z_a18_ad_TTV11_mt | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990487 / 52989778 | 2,791 | 280 | 5.96 | 2.33 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.05 | 2.915e-04 | 0.001506 | 0.001506 | no | 20261119T212102Z_a18_part_TTV15_mt | 20261119T212010Z_a18_ad_TTV15_mt | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52956715 / 52956756 | 2,090 | 261 | 4.04 | 2.21 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.770e-06 | 0.002155 | 0.002155 | yes | 20261119T212134Z_a18_part_AP3_trunk | 20261119T212009Z_a18_ad_AP3_trunk | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990472 / 52990404 | 3,755 | 264 | 5.63 | 2.32 | yes | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.984e-06 | 0.004372 | 0.005272 | yes | 20261119T212112Z_a18_part_AP3_3dep_trunk | 20261119T212009Z_a18_ad_AP3_3dep_trunk | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990473 / 52990405 | 2,233 | 269 | 3.54 | 2.25 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.610e-06 | 0.003629 | 0.00772 | yes | 20261119T212112Z_a18_part_AP4_trunk | 20261119T212009Z_a18_ad_AP4_trunk | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990474 / 52990406 | 4,269 | 266 | 6.58 | 2.50 | yes | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.47 | 4.811e-06 | 0.00536 | 0.007554 | yes | 20261119T212112Z_a18_part_AP4_3dep_trunk | 20261119T212010Z_a18_ad_AP4_3dep_trunk | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52956725 / 52956759 | 2,369 | 268 | 4.17 | 2.38 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 6.888e-05 | 0.00977 | 0.01015 | yes | 20261119T212134Z_a18_part_GM2_trunk | 20261119T211515Z_a18_ad_GM2_trunk | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990468 / 52989716 | 2,090 | 270 | 3.92 | 1.98 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 7.276e-05 | 0.01251 | 0.01197 | yes | 20261119T212134Z_a18_part_GM2_fema_trunk | 20261119T211515Z_a18_ad_GM2_fema_trunk | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990469 / 52989717 | 2,127 | 272 | 4.16 | 2.07 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 4.425e-05 | 0.009855 | 0.009435 | yes | 20261119T212134Z_a18_part_GM4_trunk | 20261119T211610Z_a18_ad_GM4_trunk | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990470 / 52989718 | 2,112 | 264 | 4.13 | 1.93 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 4.540e-05 | 0.01178 | 0.01097 | yes | 20261119T212134Z_a18_part_GM4_fema_trunk | 20261119T211531Z_a18_ad_GM4_fema_trunk | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52990471 / 52989719 | 2,268 | 266 | 4.26 | 1.96 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |
| AP5:pamap | 0.29 | 6.490e-05 | 0.008725 | 0.008471 | yes | 20261119T212134Z_a18_part_GM5_fema_trunk | 20261119T211621Z_a18_ad_GM5_fema_trunk | 862ee83b9930fa3bcfb24f9157e643625acb2088-dirty | 52956720 / 52956758 | 2,384 | 265 | 4.04 | 2.09 | no | the v2.0.0 config (m = 0.47 = the site m): 5.5 continuity and the v2.0.0 match (AP2: the 4.1 gate) |

| tile | group | sha256 filled (16) | sha256 area (16) | sha256 fd (16) | area grid |
|---|---|---|---|---|---|

### C17. Slurm accounting of every addendum 18 job

| job name | tasks | core-hours used | core-hours of limits | states |
|---|---|---|---|---|
| a18_prep | 1 | 0.0 | 2.0 | COMPLETED 1 |
| a18_tests | 1 | 0.1 | 2.0 | COMPLETED 1 |
| a18_qc | 3 | 0.1 | 9.0 | COMPLETED 3 |
| a18_pilot | 6 | 1.4 | 9.0 | COMPLETED 6 |
| a18_gate | 4 | 13.8 | 48.0 | COMPLETED 4 |
| a18_prep2 | 1 | 0.0 | 0.5 | COMPLETED 1 |
| a18_prep3 | 1 | 0.0 | 0.5 | COMPLETED 1 |
| a18_gatecheck | 1 | 0.0 | 0.5 | COMPLETED 1 |
| a18_part_primary_GM | 22 | 94.1 | 280.5 | COMPLETED 22 |
| a18_part_primary_AP | 21 | 90.2 | 278.2 | COMPLETED 21 |
| a18_part_primary_TTV | 11 | 53.6 | 129.2 | COMPLETED 11 |
| a18_part_mt_TTV | 11 | 51.1 | 129.2 | COMPLETED 11 |
| a18_part_trunk_GM | 5 | 20.7 | 63.8 | COMPLETED 5 |
| a18_part_trunk_AP | 4 | 20.0 | 53.0 | COMPLETED 4 |
| a18_part_ncalm_GM | 7 | 26.0 | 89.2 | COMPLETED 7 |
| a18_part_pamap_AP | 4 | 24.9 | 53.0 | COMPLETED 4 |
| a18_part_ap1_3dep_AP | 1 | 4.0 | 13.2 | COMPLETED 1 |
| a18_part_v2m_GM | 4 | 15.5 | 51.0 | COMPLETED 4 |
| a18_part_offmesa_GM | 6 | 25.6 | 76.5 | COMPLETED 6 |
| a18_ad_primary_GM | 22 | 46.2 | 99.0 | COMPLETED 22 |
| a18_ad_primary_AP | 21 | 44.0 | 99.8 | COMPLETED 21 |
| a18_ad_primary_TTV | 11 | 24.4 | 52.2 | COMPLETED 11 |
| a18_ad_mt_TTV | 11 | 23.4 | 52.2 | COMPLETED 11 |
| a18_ad_trunk_GM | 5 | 10.1 | 22.5 | COMPLETED 5 |
| a18_ad_trunk_AP | 4 | 9.5 | 19.0 | COMPLETED 4 |
| a18_ad_ncalm_GM | 7 | 16.0 | 31.5 | COMPLETED 7 |
| a18_ad_pamap_AP | 4 | 8.9 | 19.0 | COMPLETED 4 |
| a18_ad_ap1_3dep_AP | 1 | 2.2 | 4.8 | COMPLETED 1 |
| a18_ad_v2m_GM | 4 | 8.5 | 18.0 | COMPLETED 4 |
| a18_ad_offmesa_GM | 6 | 13.5 | 27.0 | COMPLETED 6 |
| a18_qc_pairs2 | 1 | 0.0 | 0.8 | COMPLETED 1 |
| a18_sumdry | 3 | 0.2 | 1.5 | COMPLETED 3 |
| a18_final | 1 | 0.1 | 1.5 | COMPLETED 1 |
| a18_figs | 1 | 0.1 | 0.5 | RUNNING 1 |

Core-hours used (elapsed x CPUs over all tasks): 648.3; sum of CPU-weighted limits: 1737.8; tasks: 216.

| JobID | JobName | Partition | State | Elapsed | Timelimit | AllocCPUS |
|---|---|---|---|---|---|---|
| 52954916 | a18_prep | serc | COMPLETED | 00:02:31 | 02:00:00 | 1 |
| 52954918 | a18_tests | serc | COMPLETED | 00:04:23 | 01:00:00 | 2 |
| 52954919_0 | a18_qc | serc | COMPLETED | 00:03:50 | 03:00:00 | 1 |
| 52954919_1 | a18_qc | serc | COMPLETED | 00:02:09 | 03:00:00 | 1 |
| 52954919_2 | a18_qc | serc | COMPLETED | 00:02:32 | 03:00:00 | 1 |
| 52954920_0 | a18_pilot | serc | COMPLETED | 00:15:24 | 01:30:00 | 1 |
| 52954920_1 | a18_pilot | serc | COMPLETED | 00:15:40 | 01:30:00 | 1 |
| 52954920_2 | a18_pilot | serc | COMPLETED | 00:14:08 | 01:30:00 | 1 |
| 52954920_3 | a18_pilot | serc | COMPLETED | 00:11:31 | 01:30:00 | 1 |
| 52954920_4 | a18_pilot | serc | COMPLETED | 00:12:35 | 01:30:00 | 1 |
| 52954920_5 | a18_pilot | serc | COMPLETED | 00:12:33 | 01:30:00 | 1 |
| 52954921_0 | a18_gate | serc | COMPLETED | 03:32:08 | 12:00:00 | 1 |
| 52954921_1 | a18_gate | serc | COMPLETED | 06:16:27 | 12:00:00 | 1 |
| 52954921_2 | a18_gate | serc | COMPLETED | 01:59:46 | 12:00:00 | 1 |
| 52954921_3 | a18_gate | serc | COMPLETED | 02:00:18 | 12:00:00 | 1 |
| 52955530 | a18_prep2 | serc | COMPLETED | 00:00:22 | 00:30:00 | 1 |
| 52955788 | a18_prep3 | serc | COMPLETED | 00:00:11 | 00:30:00 | 1 |
| 52956673 | a18_gatecheck | serc | COMPLETED | 00:00:28 | 00:30:00 | 1 |
| 52956702_0 | a18_part_primary_GM | serc | COMPLETED | 03:16:58 | 12:45:00 | 1 |
| 52956702_1 | a18_part_primary_GM | serc | COMPLETED | 03:38:28 | 12:45:00 | 1 |
| 52956702_2 | a18_part_primary_GM | serc | COMPLETED | 04:15:45 | 12:45:00 | 1 |
| 52956702_3 | a18_part_primary_GM | serc | COMPLETED | 04:20:17 | 12:45:00 | 1 |
| 52956702_4 | a18_part_primary_GM | serc | COMPLETED | 04:02:42 | 12:45:00 | 1 |
| 52956702_5 | a18_part_primary_GM | serc | COMPLETED | 04:38:58 | 12:45:00 | 1 |
| 52956702_6 | a18_part_primary_GM | serc | COMPLETED | 05:41:40 | 12:45:00 | 1 |
| 52956702_7 | a18_part_primary_GM | serc | COMPLETED | 04:13:06 | 12:45:00 | 1 |
| 52956702_8 | a18_part_primary_GM | serc | COMPLETED | 04:49:28 | 12:45:00 | 1 |
| 52956702_9 | a18_part_primary_GM | serc | COMPLETED | 04:01:13 | 12:45:00 | 1 |
| 52956702_10 | a18_part_primary_GM | serc | COMPLETED | 04:24:09 | 12:45:00 | 1 |
| 52956702_11 | a18_part_primary_GM | serc | COMPLETED | 04:49:57 | 12:45:00 | 1 |
| 52956702_12 | a18_part_primary_GM | serc | COMPLETED | 04:36:08 | 12:45:00 | 1 |
| 52956702_13 | a18_part_primary_GM | serc | COMPLETED | 03:48:56 | 12:45:00 | 1 |
| 52956702_14 | a18_part_primary_GM | serc | COMPLETED | 03:47:55 | 12:45:00 | 1 |
| 52956702_15 | a18_part_primary_GM | serc | COMPLETED | 04:03:55 | 12:45:00 | 1 |
| 52956702_16 | a18_part_primary_GM | serc | COMPLETED | 04:28:48 | 12:45:00 | 1 |
| 52956702_17 | a18_part_primary_GM | serc | COMPLETED | 04:15:12 | 12:45:00 | 1 |
| 52956702_18 | a18_part_primary_GM | serc | COMPLETED | 05:27:21 | 12:45:00 | 1 |
| 52956702_19 | a18_part_primary_GM | serc | COMPLETED | 04:07:45 | 12:45:00 | 1 |
| 52956702_20 | a18_part_primary_GM | serc | COMPLETED | 03:24:53 | 12:45:00 | 1 |
| 52956702_21 | a18_part_primary_GM | serc | COMPLETED | 03:49:39 | 12:45:00 | 1 |
| 52956704_0 | a18_part_primary_AP | serc | COMPLETED | 04:04:42 | 13:15:00 | 1 |
| 52956704_1 | a18_part_primary_AP | serc | COMPLETED | 03:43:47 | 13:15:00 | 1 |
| 52956704_2 | a18_part_primary_AP | serc | COMPLETED | 04:16:36 | 13:15:00 | 1 |
| 52956704_3 | a18_part_primary_AP | serc | COMPLETED | 04:01:57 | 13:15:00 | 1 |
| 52956704_4 | a18_part_primary_AP | serc | COMPLETED | 04:28:21 | 13:15:00 | 1 |
| 52956704_5 | a18_part_primary_AP | serc | COMPLETED | 04:43:13 | 13:15:00 | 1 |
| 52956704_6 | a18_part_primary_AP | serc | COMPLETED | 04:54:34 | 13:15:00 | 1 |
| 52956704_7 | a18_part_primary_AP | serc | COMPLETED | 04:14:49 | 13:15:00 | 1 |
| 52956704_8 | a18_part_primary_AP | serc | COMPLETED | 03:56:31 | 13:15:00 | 1 |
| 52956704_9 | a18_part_primary_AP | serc | COMPLETED | 05:02:43 | 13:15:00 | 1 |
| 52956704_10 | a18_part_primary_AP | serc | COMPLETED | 04:15:58 | 13:15:00 | 1 |
| 52956704_11 | a18_part_primary_AP | serc | COMPLETED | 04:57:06 | 13:15:00 | 1 |
| 52956704_12 | a18_part_primary_AP | serc | COMPLETED | 04:12:03 | 13:15:00 | 1 |
| 52956704_13 | a18_part_primary_AP | serc | COMPLETED | 04:31:39 | 13:15:00 | 1 |
| 52956704_14 | a18_part_primary_AP | serc | COMPLETED | 04:49:20 | 13:15:00 | 1 |
| 52956704_15 | a18_part_primary_AP | serc | COMPLETED | 04:07:14 | 13:15:00 | 1 |
| 52956704_16 | a18_part_primary_AP | serc | COMPLETED | 04:17:13 | 13:15:00 | 1 |
| 52956704_17 | a18_part_primary_AP | serc | COMPLETED | 03:30:11 | 13:15:00 | 1 |
| 52956704_18 | a18_part_primary_AP | serc | COMPLETED | 04:12:23 | 13:15:00 | 1 |
| 52956704_19 | a18_part_primary_AP | serc | COMPLETED | 03:45:00 | 13:15:00 | 1 |
| 52956704_20 | a18_part_primary_AP | serc | COMPLETED | 04:08:37 | 13:15:00 | 1 |
| 52956711_0 | a18_part_primary_TTV | serc | COMPLETED | 04:10:05 | 11:45:00 | 1 |
| 52956711_1 | a18_part_primary_TTV | serc | COMPLETED | 04:05:31 | 11:45:00 | 1 |
| 52956711_2 | a18_part_primary_TTV | serc | COMPLETED | 04:04:40 | 11:45:00 | 1 |
| 52956711_3 | a18_part_primary_TTV | serc | COMPLETED | 03:44:17 | 11:45:00 | 1 |
| 52956711_4 | a18_part_primary_TTV | serc | COMPLETED | 04:09:41 | 11:45:00 | 1 |
| 52956711_5 | a18_part_primary_TTV | serc | COMPLETED | 05:13:20 | 11:45:00 | 1 |
| 52956711_6 | a18_part_primary_TTV | serc | COMPLETED | 04:34:43 | 11:45:00 | 1 |
| 52956711_7 | a18_part_primary_TTV | serc | COMPLETED | 06:01:51 | 11:45:00 | 1 |
| 52956711_8 | a18_part_primary_TTV | serc | COMPLETED | 06:14:36 | 11:45:00 | 1 |
| 52956711_9 | a18_part_primary_TTV | serc | COMPLETED | 07:08:10 | 11:45:00 | 1 |
| 52956711_10 | a18_part_primary_TTV | serc | COMPLETED | 04:08:37 | 11:45:00 | 1 |
| 52956715_0 | a18_part_mt_TTV | serc | COMPLETED | 04:14:41 | 11:45:00 | 1 |
| 52956715_1 | a18_part_mt_TTV | serc | COMPLETED | 04:45:37 | 11:45:00 | 1 |
| 52956715_2 | a18_part_mt_TTV | serc | COMPLETED | 04:24:32 | 11:45:00 | 1 |
| 52956715_3 | a18_part_mt_TTV | serc | COMPLETED | 03:56:34 | 11:45:00 | 1 |
| 52956715_4 | a18_part_mt_TTV | serc | COMPLETED | 04:16:36 | 11:45:00 | 1 |
| 52956715_5 | a18_part_mt_TTV | serc | COMPLETED | 04:05:40 | 11:45:00 | 1 |
| 52956715_6 | a18_part_mt_TTV | serc | COMPLETED | 04:16:22 | 11:45:00 | 1 |
| 52956715_7 | a18_part_mt_TTV | serc | COMPLETED | 04:43:22 | 11:45:00 | 1 |
| 52956715_8 | a18_part_mt_TTV | serc | COMPLETED | 06:19:16 | 11:45:00 | 1 |
| 52956715_9 | a18_part_mt_TTV | serc | COMPLETED | 05:59:05 | 11:45:00 | 1 |
| 52956715_10 | a18_part_mt_TTV | serc | COMPLETED | 04:03:58 | 11:45:00 | 1 |
| 52956720_0 | a18_part_trunk_GM | serc | COMPLETED | 03:57:39 | 12:45:00 | 1 |
| 52956720_1 | a18_part_trunk_GM | serc | COMPLETED | 04:11:44 | 12:45:00 | 1 |
| 52956720_2 | a18_part_trunk_GM | serc | COMPLETED | 04:09:58 | 12:45:00 | 1 |
| 52956720_3 | a18_part_trunk_GM | serc | COMPLETED | 04:17:33 | 12:45:00 | 1 |
| 52956720_4 | a18_part_trunk_GM | serc | COMPLETED | 04:04:42 | 12:45:00 | 1 |
| 52956725_0 | a18_part_trunk_AP | serc | COMPLETED | 05:39:53 | 13:15:00 | 1 |
| 52956725_1 | a18_part_trunk_AP | serc | COMPLETED | 03:34:32 | 13:15:00 | 1 |
| 52956725_2 | a18_part_trunk_AP | serc | COMPLETED | 06:36:51 | 13:15:00 | 1 |
| 52956725_3 | a18_part_trunk_AP | serc | COMPLETED | 04:11:44 | 13:15:00 | 1 |
| 52956726_0 | a18_part_ncalm_GM | serc | COMPLETED | 03:17:43 | 12:45:00 | 1 |
| 52956726_1 | a18_part_ncalm_GM | serc | COMPLETED | 03:27:15 | 12:45:00 | 1 |
| 52956726_2 | a18_part_ncalm_GM | serc | COMPLETED | 03:38:24 | 12:45:00 | 1 |
| 52956726_3 | a18_part_ncalm_GM | serc | COMPLETED | 03:54:50 | 12:45:00 | 1 |
| 52956726_4 | a18_part_ncalm_GM | serc | COMPLETED | 03:40:17 | 12:45:00 | 1 |
| 52956726_5 | a18_part_ncalm_GM | serc | COMPLETED | 03:46:09 | 12:45:00 | 1 |
| 52956726_6 | a18_part_ncalm_GM | serc | COMPLETED | 04:13:27 | 12:45:00 | 1 |
| 52956729_0 | a18_part_pamap_AP | serc | COMPLETED | 06:35:04 | 13:15:00 | 1 |
| 52956729_1 | a18_part_pamap_AP | serc | COMPLETED | 05:27:21 | 13:15:00 | 1 |
| 52956729_2 | a18_part_pamap_AP | serc | COMPLETED | 06:42:02 | 13:15:00 | 1 |
| 52956729_3 | a18_part_pamap_AP | serc | COMPLETED | 06:12:01 | 13:15:00 | 1 |
| 52956731_0 | a18_part_ap1_3dep_AP | serc | COMPLETED | 03:57:11 | 13:15:00 | 1 |
| 52956734_0 | a18_part_v2m_GM | serc | COMPLETED | 03:52:35 | 12:45:00 | 1 |
| 52956734_1 | a18_part_v2m_GM | serc | COMPLETED | 03:46:11 | 12:45:00 | 1 |
| 52956734_2 | a18_part_v2m_GM | serc | COMPLETED | 03:44:26 | 12:45:00 | 1 |
| 52956734_3 | a18_part_v2m_GM | serc | COMPLETED | 04:08:48 | 12:45:00 | 1 |
| 52956738_0 | a18_part_offmesa_GM | serc | COMPLETED | 04:43:54 | 12:45:00 | 1 |
| 52956738_1 | a18_part_offmesa_GM | serc | COMPLETED | 03:29:11 | 12:45:00 | 1 |
| 52956738_2 | a18_part_offmesa_GM | serc | COMPLETED | 04:57:36 | 12:45:00 | 1 |
| 52956738_3 | a18_part_offmesa_GM | serc | COMPLETED | 03:54:50 | 12:45:00 | 1 |
| 52956738_4 | a18_part_offmesa_GM | serc | COMPLETED | 04:39:56 | 12:45:00 | 1 |
| 52956738_5 | a18_part_offmesa_GM | serc | COMPLETED | 03:48:23 | 12:45:00 | 1 |
| 52956742_0 | a18_ad_primary_GM | serc | COMPLETED | 02:03:55 | 04:30:00 | 1 |
| 52956742_1 | a18_ad_primary_GM | serc | COMPLETED | 02:10:42 | 04:30:00 | 1 |
| 52956742_2 | a18_ad_primary_GM | serc | COMPLETED | 02:04:21 | 04:30:00 | 1 |
| 52956742_3 | a18_ad_primary_GM | serc | COMPLETED | 01:53:07 | 04:30:00 | 1 |
| 52956742_4 | a18_ad_primary_GM | serc | COMPLETED | 01:53:53 | 04:30:00 | 1 |
| 52956742_5 | a18_ad_primary_GM | serc | COMPLETED | 02:07:42 | 04:30:00 | 1 |
| 52956742_6 | a18_ad_primary_GM | serc | COMPLETED | 02:05:53 | 04:30:00 | 1 |
| 52956742_7 | a18_ad_primary_GM | serc | COMPLETED | 02:08:44 | 04:30:00 | 1 |
| 52956742_8 | a18_ad_primary_GM | serc | COMPLETED | 02:07:16 | 04:30:00 | 1 |
| 52956742_9 | a18_ad_primary_GM | serc | COMPLETED | 02:13:26 | 04:30:00 | 1 |
| 52956742_10 | a18_ad_primary_GM | serc | COMPLETED | 02:15:21 | 04:30:00 | 1 |
| 52956742_11 | a18_ad_primary_GM | serc | COMPLETED | 02:07:10 | 04:30:00 | 1 |
| 52956742_12 | a18_ad_primary_GM | serc | COMPLETED | 02:04:14 | 04:30:00 | 1 |
| 52956742_13 | a18_ad_primary_GM | serc | COMPLETED | 02:02:51 | 04:30:00 | 1 |
| 52956742_14 | a18_ad_primary_GM | serc | COMPLETED | 02:02:35 | 04:30:00 | 1 |
| 52956742_15 | a18_ad_primary_GM | serc | COMPLETED | 02:00:49 | 04:30:00 | 1 |
| 52956742_16 | a18_ad_primary_GM | serc | COMPLETED | 02:11:14 | 04:30:00 | 1 |
| 52956742_17 | a18_ad_primary_GM | serc | COMPLETED | 02:04:01 | 04:30:00 | 1 |
| 52956742_18 | a18_ad_primary_GM | serc | COMPLETED | 02:09:26 | 04:30:00 | 1 |
| 52956742_19 | a18_ad_primary_GM | serc | COMPLETED | 02:06:22 | 04:30:00 | 1 |
| 52956742_20 | a18_ad_primary_GM | serc | COMPLETED | 02:08:49 | 04:30:00 | 1 |
| 52956742_21 | a18_ad_primary_GM | serc | COMPLETED | 02:10:02 | 04:30:00 | 1 |
| 52956747_0 | a18_ad_primary_AP | serc | COMPLETED | 01:55:19 | 04:45:00 | 1 |
| 52956747_1 | a18_ad_primary_AP | serc | COMPLETED | 02:09:38 | 04:45:00 | 1 |
| 52956747_2 | a18_ad_primary_AP | serc | COMPLETED | 02:09:24 | 04:45:00 | 1 |
| 52956747_3 | a18_ad_primary_AP | serc | COMPLETED | 02:08:08 | 04:45:00 | 1 |
| 52956747_4 | a18_ad_primary_AP | serc | COMPLETED | 02:11:14 | 04:45:00 | 1 |
| 52956747_5 | a18_ad_primary_AP | serc | COMPLETED | 02:05:07 | 04:45:00 | 1 |
| 52956747_6 | a18_ad_primary_AP | serc | COMPLETED | 02:21:30 | 04:45:00 | 1 |
| 52956747_7 | a18_ad_primary_AP | serc | COMPLETED | 02:06:44 | 04:45:00 | 1 |
| 52956747_8 | a18_ad_primary_AP | serc | COMPLETED | 02:05:29 | 04:45:00 | 1 |
| 52956747_9 | a18_ad_primary_AP | serc | COMPLETED | 01:57:19 | 04:45:00 | 1 |
| 52956747_10 | a18_ad_primary_AP | serc | COMPLETED | 02:08:04 | 04:45:00 | 1 |
| 52956747_11 | a18_ad_primary_AP | serc | COMPLETED | 01:57:19 | 04:45:00 | 1 |
| 52956747_12 | a18_ad_primary_AP | serc | COMPLETED | 01:53:32 | 04:45:00 | 1 |
| 52956747_13 | a18_ad_primary_AP | serc | COMPLETED | 02:01:46 | 04:45:00 | 1 |
| 52956747_14 | a18_ad_primary_AP | serc | COMPLETED | 02:03:21 | 04:45:00 | 1 |
| 52956747_15 | a18_ad_primary_AP | serc | COMPLETED | 02:07:39 | 04:45:00 | 1 |
| 52956747_16 | a18_ad_primary_AP | serc | COMPLETED | 02:06:56 | 04:45:00 | 1 |
| 52956747_17 | a18_ad_primary_AP | serc | COMPLETED | 02:16:15 | 04:45:00 | 1 |
| 52956747_18 | a18_ad_primary_AP | serc | COMPLETED | 01:58:42 | 04:45:00 | 1 |
| 52956747_19 | a18_ad_primary_AP | serc | COMPLETED | 02:01:49 | 04:45:00 | 1 |
| 52956747_20 | a18_ad_primary_AP | serc | COMPLETED | 02:13:43 | 04:45:00 | 1 |
| 52956751_0 | a18_ad_primary_TTV | serc | COMPLETED | 02:15:41 | 04:45:00 | 1 |
| 52956751_1 | a18_ad_primary_TTV | serc | COMPLETED | 02:12:39 | 04:45:00 | 1 |
| 52956751_2 | a18_ad_primary_TTV | serc | COMPLETED | 02:10:09 | 04:45:00 | 1 |
| 52956751_3 | a18_ad_primary_TTV | serc | COMPLETED | 02:04:04 | 04:45:00 | 1 |
| 52956751_4 | a18_ad_primary_TTV | serc | COMPLETED | 02:33:49 | 04:45:00 | 1 |
| 52956751_5 | a18_ad_primary_TTV | serc | COMPLETED | 02:09:21 | 04:45:00 | 1 |
| 52956751_6 | a18_ad_primary_TTV | serc | COMPLETED | 02:06:03 | 04:45:00 | 1 |
| 52956751_7 | a18_ad_primary_TTV | serc | COMPLETED | 02:22:12 | 04:45:00 | 1 |
| 52956751_8 | a18_ad_primary_TTV | serc | COMPLETED | 02:07:58 | 04:45:00 | 1 |
| 52956751_9 | a18_ad_primary_TTV | serc | COMPLETED | 02:24:09 | 04:45:00 | 1 |
| 52956751_10 | a18_ad_primary_TTV | serc | COMPLETED | 01:59:03 | 04:45:00 | 1 |
| 52956756_0 | a18_ad_mt_TTV | serc | COMPLETED | 02:15:22 | 04:45:00 | 1 |
| 52956756_1 | a18_ad_mt_TTV | serc | COMPLETED | 02:03:58 | 04:45:00 | 1 |
| 52956756_2 | a18_ad_mt_TTV | serc | COMPLETED | 02:01:27 | 04:45:00 | 1 |
| 52956756_3 | a18_ad_mt_TTV | serc | COMPLETED | 01:55:56 | 04:45:00 | 1 |
| 52956756_4 | a18_ad_mt_TTV | serc | COMPLETED | 02:12:39 | 04:45:00 | 1 |
| 52956756_5 | a18_ad_mt_TTV | serc | COMPLETED | 01:53:33 | 04:45:00 | 1 |
| 52956756_6 | a18_ad_mt_TTV | serc | COMPLETED | 02:10:41 | 04:45:00 | 1 |
| 52956756_7 | a18_ad_mt_TTV | serc | COMPLETED | 02:19:16 | 04:45:00 | 1 |
| 52956756_8 | a18_ad_mt_TTV | serc | COMPLETED | 01:55:39 | 04:45:00 | 1 |
| 52956756_9 | a18_ad_mt_TTV | serc | COMPLETED | 02:20:43 | 04:45:00 | 1 |
| 52956756_10 | a18_ad_mt_TTV | serc | COMPLETED | 02:13:12 | 04:45:00 | 1 |
| 52956758_0 | a18_ad_trunk_GM | serc | COMPLETED | 01:59:29 | 04:30:00 | 1 |
| 52956758_1 | a18_ad_trunk_GM | serc | COMPLETED | 02:05:02 | 04:30:00 | 1 |
| 52956758_2 | a18_ad_trunk_GM | serc | COMPLETED | 01:57:20 | 04:30:00 | 1 |
| 52956758_3 | a18_ad_trunk_GM | serc | COMPLETED | 01:58:26 | 04:30:00 | 1 |
| 52956758_4 | a18_ad_trunk_GM | serc | COMPLETED | 02:07:14 | 04:30:00 | 1 |
| 52956759_0 | a18_ad_trunk_AP | serc | COMPLETED | 02:19:48 | 04:45:00 | 1 |
| 52956759_1 | a18_ad_trunk_AP | serc | COMPLETED | 02:15:41 | 04:45:00 | 1 |
| 52956759_2 | a18_ad_trunk_AP | serc | COMPLETED | 02:30:54 | 04:45:00 | 1 |
| 52956759_3 | a18_ad_trunk_AP | serc | COMPLETED | 02:23:46 | 04:45:00 | 1 |
| 52956760_0 | a18_ad_ncalm_GM | serc | COMPLETED | 02:15:57 | 04:30:00 | 1 |
| 52956760_1 | a18_ad_ncalm_GM | serc | COMPLETED | 02:19:18 | 04:30:00 | 1 |
| 52956760_2 | a18_ad_ncalm_GM | serc | COMPLETED | 02:16:11 | 04:30:00 | 1 |
| 52956760_3 | a18_ad_ncalm_GM | serc | COMPLETED | 02:19:30 | 04:30:00 | 1 |
| 52956760_4 | a18_ad_ncalm_GM | serc | COMPLETED | 02:16:33 | 04:30:00 | 1 |
| 52956760_5 | a18_ad_ncalm_GM | serc | COMPLETED | 02:18:05 | 04:30:00 | 1 |
| 52956760_6 | a18_ad_ncalm_GM | serc | COMPLETED | 02:13:44 | 04:30:00 | 1 |
| 52956765_0 | a18_ad_pamap_AP | serc | COMPLETED | 02:17:04 | 04:45:00 | 1 |
| 52956765_1 | a18_ad_pamap_AP | serc | COMPLETED | 02:11:46 | 04:45:00 | 1 |
| 52956765_2 | a18_ad_pamap_AP | serc | COMPLETED | 02:13:37 | 04:45:00 | 1 |
| 52956765_3 | a18_ad_pamap_AP | serc | COMPLETED | 02:14:04 | 04:45:00 | 1 |
| 52956770_0 | a18_ad_ap1_3dep_AP | serc | COMPLETED | 02:13:03 | 04:45:00 | 1 |
| 52956773_0 | a18_ad_v2m_GM | serc | COMPLETED | 02:05:01 | 04:30:00 | 1 |
| 52956773_1 | a18_ad_v2m_GM | serc | COMPLETED | 02:05:34 | 04:30:00 | 1 |
| 52956773_2 | a18_ad_v2m_GM | serc | COMPLETED | 02:08:30 | 04:30:00 | 1 |
| 52956773_3 | a18_ad_v2m_GM | serc | COMPLETED | 02:08:06 | 04:30:00 | 1 |
| 52956779_0 | a18_ad_offmesa_GM | serc | COMPLETED | 02:22:45 | 04:30:00 | 1 |
| 52956779_1 | a18_ad_offmesa_GM | serc | COMPLETED | 02:18:22 | 04:30:00 | 1 |
| 52956779_2 | a18_ad_offmesa_GM | serc | COMPLETED | 02:15:21 | 04:30:00 | 1 |
| 52956779_3 | a18_ad_offmesa_GM | serc | COMPLETED | 02:12:45 | 04:30:00 | 1 |
| 52956779_4 | a18_ad_offmesa_GM | serc | COMPLETED | 02:13:23 | 04:30:00 | 1 |
| 52956779_5 | a18_ad_offmesa_GM | serc | COMPLETED | 02:06:44 | 04:30:00 | 1 |
| 52956806 | a18_qc_pairs2 | serc | COMPLETED | 00:02:16 | 00:45:00 | 1 |
| 52957430 | a18_sumdry | serc | COMPLETED | 00:03:58 | 00:30:00 | 1 |
| 52972736 | a18_sumdry | serc | COMPLETED | 00:03:31 | 00:30:00 | 1 |
| 53014207 | a18_sumdry | serc | COMPLETED | 00:03:57 | 00:30:00 | 1 |
| 53016507 | a18_final | serc | COMPLETED | 00:06:27 | 01:30:00 | 1 |
| 53031571 | a18_figs | serc | RUNNING | 00:04:03 | 00:30:00 | 1 |

Peak memory per job or array (GB): 52954916: 1.3; 52954918: 1.2; 52954919: 3.0; 52954920: 3.7; 52954921: 3.8; 52955530: 0.3; 52955788: 0.1; 52956673: 0.3; 52956702: 2.6; 52956704: 2.5; 52956711: 2.6; 52956715: 2.5; 52956720: 2.5; 52956725: 2.6; 52956726: 3.0; 52956729: 2.5; 52956731: 2.6; 52956734: 2.4; 52956738: 2.4; 52956742: 3.8; 52956747: 3.8; 52956751: 3.8; 52956756: 3.9; 52956758: 3.7; 52956759: 3.3; 52956760: 3.2; 52956765: 3.4; 52956770: 3.2; 52956773: 3.3; 52956779: 3.3; 52956806: 1.5; 52957430: 0.1; 52972736: 0.4; 53014207: 0.4; 53016507: 1.3
