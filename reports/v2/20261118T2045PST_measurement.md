# [2026-11-18 20:45 PST] Addendum 17 Part B: the per-tile measurement of the expanded tile set

Directive: `reports/directives/2026-11-18T1500PST_v2_addendum_17_register_figures_measure.md` (Part B, sections 5-8) and the go-ahead
`reports/directives/2026-11-18T1630PST_v2_addendum_17_partB_goahead.md` (Steve, 2026-11-18 16:30 PST: exclusions, the FEMA survey as the
primary GM survey, two screens). Branch `v2.3-tiles` (from 1d052b6, the Part A report). **No AD or partitioned joint fit of the expanded set
was run (addendum 18); no merge, no tag.** Scripts: `reports/v2/scripts/a17b_common.py`, `a17b_tile_configs.py`, `a17b_measure_runs.py`,
`a17b_screens.py`, `a17b_measure.py`, `make_a17b_tables.py` (ruff clean); the measurement itself is the v2 code of
`docs/measurement_methods.md` section 0 (`lemcvt measure` through its `run_measure` function, `lemcvt tile run --stages m_profile
--optimizer v2`, and the functions of `lemcvt.measure.phase6_inputs`, `lemcvt.measure.audits`, `lemcvt.analyses.exponent_site`),
unchanged; no package code was changed. Environments: `lemcvt`, `ci312` (ruff). Every number is tied to the run directory
`$GROUP_HOME/lemcvt_results/20261119T002830Z_a17b_measurement/` (`configs/`, `screens/`, `measure/`), to a lemcvt run id (the eight
`measure_a17b_t<k>` runs and the 63 `tile_run_<tile>_v2_m_profile` runs under `$GROUP_HOME/lemcvt_results/`) or to a Slurm job id.
Appendices (section 8) are machine-generated. **This is a stop for review.**

## Summary

| item | result |
| --- | --- |
| 1 state | `v2.3-tiles` = origin = 1d052b6; no job running; the Part A tests stand (no package code changed). Exclusions applied (Tier 1 14, Tier 2 5). **Bedrock screen: reference unit QTp (Paso Robles Formation), all seven control tiles pass; nine primary GM tiles fail as off-mesa** (GM14, GM18, GM19, GM25, GM28, GM30 on Pancho Rico and unnamed Tertiary units; **GM8, GM9, GM21 only because the Paso Robles subunit QTcl is a different label**: decision needed). Gap-fill screen: no tile above 20 % (max TTV8, 19 % of its channel pixels). **Primary set fixed before the profiles: GM 19, AP 21, TTV 11 (51 tiles); 73 tiles measured, 63 new m-profiles.** Deviations in section 1 (a `measure` wrapper; a run-id collision rerun) |
| 2 concavity | **Site m (argmin S): GM 0.29 (1 % 0.26-0.31), AP 0.47 (0.43-0.51), TTV 0.33 (0.28-0.39)**; clean subsets 0.28 / 0.48 / 0.35; NCALM GM1-GM7 0.33 (0.31-0.36); PAMAP AP1-AP5 recomputed 0.47. Rule 6.2: no site moved by more than 0.05 from v2 (GM −0.02, AP 0.00). χ-z θ at 10³ m², site-pooled: GM 0.32 [0.30-0.33], AP 0.39 [0.37-0.40], **TTV 0.19 [0.15-0.22]** (equal-weight 0.27 / 0.30 / 0.08); at 10⁴: 0.31 / 0.50 / 0.26. Six of the eleven TTV profiles are degenerate (optimum at the grid edge m = 0.05) |
| 3 k_s, K₀ | Site-pooled k_s at the site m above 10³ m²: **GM 2.05 (K₀ 7.2 × 10⁻⁵), AP 11.5 (4.8 × 10⁻⁶), TTV 5.39 (1.43 × 10⁻⁵)**; tile K₀ ranges GM 4.5-9.1 × 10⁻⁵, AP 4.1-5.5 × 10⁻⁶, TTV 1.1-2.6 × 10⁻⁵; n_eff per tile 1.0-6.2 (median 2.2) |
| 4 C_ht, D₀ | Option (iii) at L_f = 11 m, medians: GM 0.0101 (D₀,A 0.015, tiles 0.007-0.035), AP 0.0105 (D₀,A 0.0044, 0.0013-0.0063), TTV 0.0265 (D₀,A 0.0029, 0.0009-0.0045). **Rule 6.1: the window test stands on NCALM GM1-GM7 (7/7, as v2) and on TTV (9/11, five trivially), fails on the FEMA GM set (11/19 = 58 %) and on 3DEP AP (9/21, eight of the nine trivially: collapsed hilltop sets; AP11 has no hilltop pixel)**: D₀ is a bracket [D₀,A, D₀,B] at GM-FEMA and AP, a measurement at TTV and on NCALM |
| 5 surveys | Rule 6.3 (AP, 3DEP/PAMAP): k_s 0.96 (0.94-0.99), θ 1.00, **C_ht 0.65 (0.49-1.17)**. Rule 6.4 (GM, FEMA/NCALM, report only): **k_s 0.985 (every tile within 5.3 %); C_ht 0.78 (0.62-0.88): median outside ±15 %, three tiles outside ±30 %, criteria not met**. The FEMA surface (3.4 pts m⁻², smoother) carries 22 % less hilltop curvature than NCALM on the same footprints, and its rollover-network crests are 2-5 × wider |
| 6-7 | What holds and what is weak, and the decisions for addendum 18 (QTcl; D₀ on FEMA and 3DEP; the TTV concavity; AP11): sections 6-7 |
| cost | 93 tasks in 10 submissions, **76.0 core-hours used** (expected ≤ 150; the m-profiles 73.5), 233 core-hours of CPU-weighted limits (largest submission 189: the 63-task array at 3 h, 40 concurrent CPUs), peak 5.2 GB; +0.5 GB on group storage |

## 1. State check, the exclusions, the tiers, the two screens and the final primary set

**At the start (2026-11-18 16:20 PST, login node sh03-ln05).**
- `v2.3-tiles` = `origin/v2.3-tiles` = 1d052b6 (the Part A report, pushed); no job was running; the Part A test run (274 passed, job 52862632)
  stands (no package code changed in Part A or here).
- `/etc/agents/AGENTS.md` re-read at the start of this session; the go-ahead and the Part B sections of the directive read.
- The Part A run directory, the copies of record and the registered tiles of Part A exist (`configs/data.yaml`, 758 manifest rows).

**Deviations and choices, each handled as stated:**
1. **Tile configs for the 76 registered tiles without one** (`configs/tiles/<tile>.yaml`, written by `a17b_tile_configs.py`, job 52874596) and a
   site config for TTV (`configs/sites/TTV.yaml`: E = 7.7 × 10⁻⁵ m/yr, n = 1, S_c = 1.27, sanity range 0-800 m, median-slope limit 1.0). Each
   config carries the grids with their manifest checksums, metres, a `measured_v1` block holding the **site selection reference** (the loader
   needs a measured block; these are labelled as references, not v1 tile values) and the site's model block (the v2 metre bounds; `model.m`
   a placeholder). The exponent profile takes the exact box-constrained (K, D) optimum at every (m, A_crit), so the starting K0 = E/k_s and
   D0 = E/C_ht from the reference do not affect it. GM1-GM5 keep their existing configs (old grids = the `area_model` grids: go-ahead 2.2);
   AP1-AP5 (PAMAP, feet converted at load) keep theirs. Every measured tile loads with its checksums and passes the site sanity ranges
   (`configs/load_check.json`, `tile_configs.json`; median pixel slopes GM 0.17-0.44, AP 0.24-0.32, TTV 0.25-0.47). The `measured` block
   (site m, k_s, C_ht with provenance) is left for addendum 18.
2. **`lemcvt measure` cannot name the new tiles** (`cli._parse_tiles` validates against the fixed ten v2 tiles; `tile run` does not), so the
   `measure` step was run through `a17b_measure_runs.py`, a wrapper that calls the same `run_measure` with the v1 legacy rerun off (its site
   tables know AP and GM only). No package code was changed. The first array (52874660) failed on this before writing anything.
3. **The second `measure` array collided on its run ids** (a timestamp to the second plus "9tiles": five tasks starting in the same second
   wrote one directory; array 52877061). The three surviving directories were moved to `$GROUP_HOME/lemcvt_results/_a17b_collided/` (kept,
   outside the run lookup) and the array was rerun with the task id in the label (`<ts>_measure_a17b_t<k>`, array 52880639): 8 runs, 73
   tiles, each once. The gap-fill jobs chained to the failed arrays were cancelled and resubmitted (52874664, 52877071 → 52880640).
4. **The bedrock screen's reference unit is taken by the map's general label** (`GEN_LAB`, written into `a17b_common.py` and the screen script
   before the screen ran): the largest non-Quaternary label under the seven control footprints. It is **QTp, "Paso Robles Formation,
   undifferentiated"**, as expected. The map also carries labels QTcl, QTg, QTm, QTpc with `GEN_NAME` "Paso Robles Formation"; under the
   rule as written these are "other bedrock". GM8, GM9 and GM21 fail only for QTcl (12.6, 57.0 and 28.7 % of their footprints). The rule was
   applied as written; the formation-level reading is reported beside it (section 1.3, section 6) and the choice is Steve's (section 7).
   Quaternary units counted with the reference unit: Qal, Qfp, Qsc, Qhf, Qfpl, Qfpm, Qfu, Qt, Qtmp, Qtlp, Qc, Qb (alluvium, flood-plain,
   stream channel, fans, fluvial terraces, colluvium, basin deposits); not counted: Qls (landslides), Qfl (fill), eolian, dune, beach,
   coastal terrace, Aromas Sand. Queried labels ("Qal?") count with their general label.
5. **Rule 6.1's pass rule "at least 4 of 5 tiles"** (addendum 6 §1, written for five tiles) is applied as **≥ 80 % of the group's tiles**
   (`a17b_common.WINDOW_TEST_PASS_FRAC`, set before the measurement); the counts are reported with it, and the "trivial" passes (only L_f in
   the tested range) are listed.
6. **Per-tile θ** is the pooled through-origin χ-z estimator of the audit at A_c = 10³ m² (the equal-weight estimator and A_c = 10⁴ beside it,
   basin-bootstrap 95 % bands); the `measure` run's θ* at the rollover A_c is reported for the record (Appendix B6).
7. **The gap-fill screen's masks**: `nnfill_mask == 1` (nearest-neighbour filled within 10 m, stage A) or `gapfill_mask == 1` (enclosed gap
   filled for routing, stage B); the NCALM and PAMAP tiles have no masks and are exempt (continuity only).
8. **Option (iii) sets that collapse** on the dense 3DEP networks: AP2_3dep keeps 13 hilltop pixels after the L_f/2 exclusion, AP18 25, AP9 30,
   AP27 83, AP14 118, AP28 119, AP10 145 and **AP11 none** (AP3 had 2 in v2); their C_ht and window tests are reported and carry no weight.
9. **The m-profile limit** (3 h a task against 1.1-1.3 h measured) is 2.3-2.7 × the estimate, above the "twice" rule: the array was sized
   before the first new-tile timings were known; every task finished in 1.0-1.3 h. Recorded, not repeated.
10. The report draft lived in the repository (not `/tmp`) while the jobs ran; a build error deleted it once and it was rebuilt from the
    run directory (nothing in the run directory was touched).

### 1.1 Exclusions (go-ahead section 1)
Tier 1, excluded from every result (not measured, except AP1_3dep, which is measured for the survey comparison and continuity only); all
stay registered; Appendix B1 lists each with its reason: **GM10, GM23, AP1_3dep, AP12, AP15, AP16, AP17, AP21, AP22, AP26, TTV10, TTV12,
TTV13, TTV14** (14 tiles). Tier 2, kept in the primary set and left out of the clean subset: **GM12, GM29, GM26, AP25, TTV11** (roads or pads
on ridges, which flatten hilltops).

### 1.2 The GM survey (go-ahead section 2)
The FEMA 2018 survey is the primary survey for every GM tile: GM1_fema-GM7_fema (the control grid's full-grid routing as the area source)
and GM8-GM30 (the extension grid's). The NCALM tiles GM1-GM7 are measured as the continuity check (GM1-GM5 on their old grids, which are the
`area_model` grids of Part A, through the existing tile configs; GM6-GM7 on their registered areas) and do not enter S(m).

### 1.3 The screens (go-ahead section 3), written before any result
**Bedrock (go-ahead 3.1)**, `a17b_screens.py --part bedrock` (job 52874604); map: the Digital Geologic Map of Monterey County (Rosenberg,
compiled 1934-2001, 1:24,000; `https://purl.stanford.edu/cm427jp1187`, public domain), fetched from the Stanford Digital Repository with
`curl` on the login node (9 files, SHA-1 digests equal to the repository record's), copy of record
`$GROUP_HOME/lemcvt_data/source/Geology_Monterey_Rosenberg_cm427jp1187/` with a sha256 manifest (GEOLOGY.shp
`edd559992b98d894...`, GEOLOGY.dbf `f8e4e20847f41ccf...`; not committed). The map is NAD83 / California zone 4 (ftUS), 7,261 polygons;
the EPSG:6339 footprints were reprojected to it with GDAL/PROJ and intersected. **Reference unit: QTp (Paso Robles Formation,
undifferentiated)**, 98.6 % of the seven control footprints combined (QTp 100 % on GM1, GM3, GM4, GM7; 95-96 % on GM2 and GM5 with 4-5 %
Qal; 99.98 % on GM6). Appendix B2 has every tile's unit fractions and `reports/v2/figures/a17b/GM_bedrock_screen.png` the map.

| result | tiles |
| --- | --- |
| pass (≥ 90 %) | GM1_fema-GM7_fema (99.5-100 %), GM10 (Tier 1), GM11, GM12, GM13, GM15, GM16, GM17, GM20, GM22, GM24 (97.7 %), GM26, GM27 (97.3 %), GM29: 20 of 30 |
| **fail, off-mesa** | **GM8 (87.4 %: QTcl 12.6 %)**, **GM9 (43.0 %: QTcl 57.0 %)**, GM14 (7.6 %: Tucs 31 %, Tucc 28 %, Qls 24 %, Tpo), GM18 (46.7 %: Tucs 53 %), GM19 (23.5 %: Tpo 76 %), **GM21 (71.3 %: QTcl 28.7 %)**, GM23 (18.7 %, Tier 1: Tpo 79 %), GM25 (19.1 %: Tucs 81 %), GM28 (80.8 %: Qal 25 %, Tucc 19 %), GM30 (26.3 %: Tpo 73 %) |

No control tile fails (no stop). **Nine primary tiles leave the set as off-mesa.** The high eastern tiles (GM14, GM18, GM21, GM28, GM30), GM23
and the north-western GM19 and GM25 sit on the Pancho Rico Formation (Tpo) and the unnamed Tertiary clastic units (Tucs, Tucc), with
landslide deposits on GM14: the different landscape the old elevation window stood in for. **GM8, GM9 and GM21 are the Paso Robles
Formation under its QTcl label** (deviation 4); read at the formation level (QTp + QTcl + QTg + QTm + QTpc) they pass at 100 % and the GM
primary set would be 22 (GM28 stays out at 80.8 %).

**Gap fill on the measurement pixels (go-ahead 3.2)**, `a17b_screens.py --part gapfill` (job 52880640), for the 60 expected primary tiles:
the largest fractions are TTV8 (19.1 % of the k_s channel pixels, 6.1 % of the hilltop pixels; 14.1 % of the tile), TTV11 (14.5 % / 4.3 %),
TTV7 (11.1 % / 9.0 %), GM14 (8.5 % / 8.2 %) and GM28 (5.2 % / 6.3 %); AP ≤ 6.1 % / 7.7 % (AP2_3dep's 7.7 % is 1 of 13 pixels). **No tile
exceeds 20 % on either set; none leaves the primary set.** The tile-wide fractions of Part A (TTV8 14 %, TTV9 16 %) translate into 19 % and
3.4 % on the channel pixels and 6 % and 1.3 % on the hilltops. Appendix B3.

### 1.4 The primary set
Fixed at 2026-11-19 01:38:55 UTC (17:39 PST), before the m-profile array was submitted (job 52881438 at 17:39:41 PST):

| site | primary set | n | removed by the screens | clean subset (minus Tier 2) |
| --- | --- | --- | --- | --- |
| GM | GM1_fema-GM7_fema, GM11, GM12, GM13, GM15, GM16, GM17, GM20, GM22, GM24, GM26, GM27, GM29 | **19** (expected 28) | GM8, GM9, GM14, GM18, GM19, GM21, GM25, GM28, GM30 (off-mesa) | 16 (minus GM12, GM26, GM29) |
| AP | AP2_3dep-AP5_3dep, AP6-AP11, AP13, AP14, AP18-AP20, AP23-AP25, AP27-AP29 | **21** (expected 21) | none | 20 (minus AP25) |
| TTV | TTV1-TTV9, TTV11, TTV15 | **11** (expected 11) | none | 10 (minus TTV11) |

**51 primary tiles** (47 in the clean subset). Also measured, outside the primary set: GM1-GM7 on NCALM (continuity), AP1_3dep (continuity,
survey comparison), AP1-AP5 on PAMAP (rule 6.3), and the nine off-mesa GM tiles (measured and profiled, reported separately): **73 tiles
measured, 63 new m-profiles** (GM1-GM5 and AP1-AP5 have the addendum 4 profiles at the same settings).

## 2. Concavity: per-tile θ, the site-pooled θ, S(m) and the site m

### 2.1 S(m) and the site m (section 5.2; rule: the site m is the argmin of S(m) over the primary set)
`a17b_measure.py` (job 52881442) over the 63 new m-profile runs (array 52881438, `tile run --stages m_profile --optimizer v2 --m-range 0.05,0.90`,
200-value A_crit grid, 86 m values; 1.0-1.3 h per tile) and the addendum 4 runs of GM1-GM5 and AP1-AP5 (same settings). Appendix B5;
figures `reports/v2/figures/a17b/<site>_S_m.png`.

| site | group | n | **m = argmin S** | 1 % interval | 5 % interval | leave-one-out range | v2 site m |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GM | **primary (FEMA)** | 19 | **0.29** | 0.26-0.31 | 0.22-0.35 | 0.28-0.30 | 0.31 (NCALM GM1-GM5) |
| GM | clean subset | 16 | 0.28 | 0.25-0.31 | 0.22-0.35 | - | |
| GM | NCALM GM1-GM7 (continuity, go-ahead 2.4) | 7 | 0.33 | 0.31-0.36 | 0.27-0.40 | - | |
| GM | the 12 drawn FEMA tiles alone (record) | 12 | 0.26 | 0.23-0.30 | - | - |  |
| GM | formation reading, + GM8, GM9, GM21 (record) | 22 | 0.29 | 0.26-0.32 | 0.22-0.36 | - |  |
| AP | **primary (3DEP)** | 21 | **0.47** | 0.43-0.51 | 0.38-0.60 | 0.46-0.49 | 0.47 (PAMAP) |
| AP | clean subset | 20 | 0.48 | 0.45-0.53 | 0.39-0.62 | - | |
| AP | PAMAP AP1-AP5 (recomputed) | 5 | 0.47 | 0.44-0.49 | 0.41-0.53 | - | |
| TTV | **primary** | 11 | **0.33** | 0.28-0.39 | 0.15-0.46 | 0.22-0.35 | - |
| TTV | clean subset | 10 | 0.35 | 0.30-0.40 | 0.21-0.47 | - | |

**Rule 6.2 (for the record): no site m moves by more than 0.05.** GM: 0.29 against 0.31 (−0.02); the seven NCALM tiles alone give 0.33
(+0.02; the drawn GM6 and GM7 pull the NCALM set up, their own optima being 0.45 and 0.46). AP: 0.47 on 3DEP, 0.47 on PAMAP. The 1 %
intervals widen with the larger sets (GM ±0.025, AP ±0.04) because the per-tile profiles disagree more than the five originals did: the
per-tile optima span 0.05-0.46 at GM (GM15 and GM22 at 0.05 and 0.11; the seven FEMA controls 0.24-0.46), 0.28-0.70 at AP (AP10 0.70,
AP18 0.63, AP25 0.28), and at TTV **six of eleven tiles have their optimum at the grid edge m = 0.05** (TTV1, TTV6, TTV7, TTV8, TTV11, TTV15;
A_crit 15,000-59,000 m²) while the other five sit at 0.35-0.45: the TTV S(m) is a compromise between two populations and its
leave-one-out minima run 0.22-0.35. The twelve drawn FEMA tiles alone give 0.26 (1 % 0.23-0.30); the seven FEMA controls pull the primary set up to 0.29.

### 2.2 The per-tile θ by χ-z (section 5.1) and the site-pooled θ
Pooled through-origin estimator at A_c = 10³ m² (the audit's basin filters; basin bootstrap, 200 draws), equal-weight beside it, 10⁴ m²
beside both; per tile in Appendix B6, site-pooled in B7.

| site | group | per-tile θ (10³, pooled): range, median | **site-pooled θ (10³)** [95 %] | equal-weight (10³) [95 %] | basins | site-pooled (10⁴) [95 %] | equal (10⁴) | basins (10⁴) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GM | primary | 0.25-0.40, 0.32 | **0.32** [0.30, 0.33] | 0.27 [0.24, 0.30] | 224 | 0.31 [0.28, 0.33] | 0.30 | 103 |
| GM | NCALM | 0.31-0.35, 0.32 | 0.33 [0.31, 0.35] | 0.29 [0.24, 0.32] | 101 | 0.32 [0.30, 0.34] | 0.30 | 40 |
| AP | primary | 0.28-0.44, 0.38 | **0.39** [0.37, 0.40] | 0.30 [0.27, 0.32] | 403 | 0.50 [0.46, 0.54] | 0.51 | 81 |
| AP | PAMAP | 0.36-0.41, 0.39 | 0.39 [0.37, 0.41] | 0.29 [0.21, 0.34] | 116 | 0.52 [0.47, 0.57] | 0.53 | 16 |
| TTV | primary | 0.00-0.31, 0.17 | **0.19** [0.15, 0.22] | 0.08 [0.04, 0.12] | 245 | 0.26 [0.16, 0.32] | 0.17 | 100 |

The site m and the χ-z θ agree at GM (0.29 against 0.32 at 10³ and 0.31 at 10⁴) and bracket each other at AP (0.47 against 0.39 at 10³
and 0.50 at 10⁴, as in v2). At TTV they do not: the model prefers 0.33 (with six degenerate profiles) while the χ-z concavity of the 1 m
networks above 10³ m² is 0.19 (0.26 at 10⁴, equal-weight 0.08-0.17), far below the 0.44 of the M&D polygon on the 10 m NED (addendum 16)
and the 0.45-0.55 the M&D interpretation presumes. Section 6.

## 3. k_s and K₀ per tile
k_s by χ-z at θ = site m over the network above 10³ m², pooled through origin per tile and per site (basin bootstrap, n_eff); K₀ = E/k_s with
E = GM 1.47 × 10⁻⁴, AP 5.5 × 10⁻⁵, TTV 7.7 × 10⁻⁵ m/yr (decision 6). Appendix B6 (per tile), B7 (site).

| site | group | n | site-pooled k_s [95 %] | n_eff | **K₀ (site-pooled)** | tile K₀ median, range | tile n_eff range | site k_s at 10⁴ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GM | primary (m 0.29) | 19 | 2.05 [1.91, 2.18] | 33.9 | **7.17 × 10⁻⁵** | 7.16 × 10⁻⁵, 4.5-9.1 × 10⁻⁵ | 1.1-3.6 | 1.99 |
| GM | clean | 16 | 2.10 [1.94, 2.24] | 29.5 | 7.01 × 10⁻⁵ | 7.11 × 10⁻⁵ | | |
| GM | NCALM | 7 | 2.22 [2.00, 2.43] | 11.2 | 6.64 × 10⁻⁵ | 6.89 × 10⁻⁵, 4.4-8.0 × 10⁻⁵ | 1.2-3.6 | 2.12 |
| AP | primary (m 0.47) | 21 | 11.48 [11.09, 11.90] | 34.5 | **4.79 × 10⁻⁶** | 4.78 × 10⁻⁶, 4.1-5.5 × 10⁻⁶ | 1.0-4.3 | 12.58 |
| AP | PAMAP | 5 | 11.60 [10.96, 12.21] | 7.1 | 4.74 × 10⁻⁶ | 4.75 × 10⁻⁶ | 1.1-2.3 | 12.99 |
| TTV | primary (m 0.33) | 11 | 5.39 [4.67, 6.11] | 17.8 | **1.43 × 10⁻⁵** | 1.24 × 10⁻⁵, 1.05-2.64 × 10⁻⁵ | 1.2-6.2 | 5.93 |

At GM the FEMA primary set gives k_s 2.05 against 2.22 on the NCALM originals (both at the site m 0.29) and K₀ 7.2 against 6.6 × 10⁻⁵
(+8 %); the v2 configs carry 2.3-4.2 per tile at m = 0.31. At AP, 3DEP and PAMAP agree within 1 % on the site-pooled k_s (11.5 against
11.6; v2 11.6). TTV's k_s = 5.4 at θ = 0.33 is in different units from the 26.9 of the addendum 16 reference (θ = 0.44 on the 10 m NED) and
is not comparable with it. n_eff is 1-4 on most tiles (the largest basin dominates), so the per-tile intervals understate the uncertainty,
as in v2; the site-pooled values rest on 11-34 effective basins.

## 4. Hilltop curvature, D₀, and the window-stability test (rule 6.1)
Option (iii) hilltop set (rollover-network divides, L_f/2 exclusion) from each tile's `measure` run; C_ht(L) at L = 7-51 m with segment
bootstraps; L_w = the largest tested window not exceeding the tile's median crest width; D₀,A = E/C_ht(L_f), D₀,B = E/C_ht(L_w). Rule 6.1:
the pre-registered test of addendum 6 §1 (C_ht within ±10 % of C_ht(L_f) for every window from L_f to the median crest width) stands at
≥ 80 % of the group's tiles. Appendix B6 and B7; figures `<site>_C_ht_L.png`.

| site | group | n | C_ht(L_f) median (range) | **D₀,A median (range)** | D₀,B median | median crest widths (m) | window test | trivial passes (only L_f in range) | **stands** | D₀ status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GM | primary (FEMA) | 19 | 0.0101 (0.0041-0.020) | **0.015** (0.007-0.035) | 0.014 | 20-86 | 11 / 19 = 58 % | 0 | **no** | **bracket** |
| GM | clean | 16 | | 0.015 | 0.015 | | 9 / 16 = 56 % | 0 | no | bracket |
| GM | NCALM GM1-GM7 | 7 | 0.0149 (0.0067-0.019) | **0.0099** (0.0076-0.022) | 0.010 | 13-20 | 7 / 7 | 2 (GM3, GM7) | **yes** | measurement (as v2) |
| AP | primary (3DEP) | 21 | 0.0105 (0.0087-0.042) | **0.0044** (0.0013-0.0063) | 0.0055 | 12-50 | 9 / 21 = 43 % | **8** (AP2_3dep, AP9, AP10, AP11, AP14, AP18, AP27, AP28) | **no** | **bracket** (as v2) |
| AP | PAMAP AP1-AP5 | 5 | 0.0171 | 0.0032 (0.0022-0.0036) | 0.0041 | 12-44 | 1 / 5 | 1 (AP3) | no | bracket (as v2) |
| TTV | primary | 11 | 0.0265 (0.0173-0.088) | **0.0029** (0.0009-0.0045) | 0.0029 | 12-26 | 9 / 11 = 82 % | **5** (TTV1, TTV3, TTV4, TTV6, TTV15) | **yes** | measurement |
| TTV | clean | 10 | | 0.0030 | 0.0031 | | 8 / 10 = 80 % | 5 | yes | measurement |

**What the test says.**
1. **GM on the FEMA survey fails the test that GM passed on NCALM.** On the same seven footprints the NCALM crests are 13-20 m wide and the
   test runs over one or two windows; on FEMA the rollover networks are sparser, the hilltop sets larger (3,300-8,600 pixels against
   900-3,600) and the crests 20-86 m wide, so the test runs to 31-51 m, where C_ht has fallen 10-20 % (GM4_fema and GM6_fema fail; the drawn
   FEMA tiles GM12, GM15, GM16, GM17, GM22, GM27 fail at 11-22 %). The C_ht(L) curves (`GM_C_ht_L.png`) are flat to 15-25 m and decline beyond.
2. **AP on 3DEP fails as on PAMAP, and its hilltop sets collapse**: on eight tiles the rollover A_c is so small (the 3DEP networks are dense)
   that fewer than 150 pixels survive the L_f/2 exclusion (AP2_3dep 13, AP18 25, AP9 30; **AP11 none: no C_ht, no D₀**); their "passes" are
   trivial. On the thirteen tiles with a real set the C_ht(L) curve falls 17-41 % by the crest width (AP13 41 %, AP4_3dep 29 %), so AP D₀
   stays a bracket, D₀,A 0.0044 to D₀,B 0.0055 (medians).
3. **TTV passes** (9 of 11) with crests of 12-26 m, five of them trivially; TTV8 and TTV9 fail (11 % and 33 %). Its C_ht(L_f) is 2-3 × GM's
   and D₀,A = 0.0029 (0.0009-0.0045) m²/yr.
4. **AP D₀ (rule 6.1 as written)**: the test failed on the 3DEP tiles, so the bracket remains; the same holds for GM on FEMA, while GM on NCALM
   and TTV carry measurements. Addendum 18 has to run the FEMA GM tiles with the bracket, or a decision is needed (section 7).

## 5. Survey comparisons

### 5.1 AP: PAMAP against 3DEP (rule 6.3, report only; same methods, same m = 0.47)
| tile | k_s PAMAP → 3DEP (ratio) | C_ht(L_f) PAMAP → 3DEP (ratio) | θ (10³) PAMAP → 3DEP |
| --- | --- | --- | --- |
| AP1 (3DEP tile Tier 1) | 12.42 → 12.27 (0.988) | 0.0153 → 0.0100 (0.65) | 0.39 → 0.40 |
| AP2 | 11.58 → 10.90 (0.941) | 0.0171 → 0.0201 (1.17) | 0.41 → 0.39 |
| AP3 | 11.53 → 11.04 (0.957) | 0.0255 → 0.0126 (0.49) | 0.38 → 0.37 |
| AP4 | 11.93 → 11.43 (0.958) | 0.0152 → 0.0103 (0.68) | 0.36 → 0.36 |
| AP5 | 10.73 → 10.38 (0.968) | 0.0188 → 0.0107 (0.57) | 0.39 → 0.39 |

Medians: k_s **0.958** (3DEP 1-6 % lower on every tile), θ **1.00**, C_ht **0.65** (0.49-1.17; AP2 is the exception at 1.17, with 944
PAMAP against 13 3DEP pixels). The 3DEP grid, gridded from class-2 points at 1 m with nearest-neighbour fill, is smoother at the hilltop
scale than the PAMAP DEM, whose grid striping (Part A pairs figure) adds curvature at 11 m.

### 5.2 GM: NCALM 2003 against FEMA 2018 (rule 6.4, report only under go-ahead 2.3; same methods, same m = 0.29)
| tile | k_s NCALM → FEMA (ratio) | C_ht(L_f) NCALM → FEMA (ratio) | θ (10³) NCALM → FEMA | median crest width NCALM → FEMA (m) |
| --- | --- | --- | --- | --- |
| GM1 | 2.41 → 2.38 (0.989) | 0.0163 → 0.0143 (0.88) | 0.32 → 0.32 | 15 → 44 |
| GM2 | 2.13 → 2.02 (0.947) | 0.0150 → 0.0118 (0.78) | 0.31 → 0.31 | 16 → 32 |
| GM3 | 1.84 → 1.83 (0.992) | 0.0145 → 0.0095 (0.65) | 0.35 → 0.32 | 13 → 20 |
| GM4 | 3.32 → 3.24 (0.975) | 0.0149 → 0.0125 (0.84) | 0.35 → 0.33 | 20 → 48 |
| GM5 | 2.30 → 2.27 (0.985) | 0.0194 → 0.0168 (0.87) | 0.32 → 0.30 | 15 → 26 |
| GM6 | 2.12 → 2.15 (1.011) | 0.0134 → 0.0090 (0.67) | 0.35 → 0.34 | 15 → 68 |
| GM7 | 1.83 → 1.78 (0.972) | 0.0067 → 0.0041 (0.62) | 0.32 → 0.31 | 13 → 86 |

**k_s: median ratio 0.985, every tile within 5.3 %** (within ±15 % and ±30 %: yes). **|C_ht(L_f)|: median ratio 0.78** (0.62-0.88),
outside ±15 %, and GM3 (0.66), GM6 (0.67) and GM7 (0.62) outside ±30 %: **the criteria of rule 6.4 are not met.** No pooling decision
follows (the primary set is one survey), but the number matters for addendum 18: on the same ground the FEMA survey's hilltop curvature is
22 % lower and its D₀ 28 % higher than NCALM's, and its rollover-network crests are 2-5 times wider. The channel steepness is
survey-independent; the hilltop curvature at 11 m is not. The site-level D₀,A on NCALM (0.0099; the v2 configs 0.0076-0.0101) against FEMA
(0.015 on the 19 primary tiles; 0.0125 on the seven controls) is the clearest survey effect in this report.

## 6. What the measurements support, and where they are weak
**Supported.**
1. **The site m is stable under the expansion** (rule 6.2): GM 0.29 against 0.31, AP 0.47 against 0.47, with 1 % intervals of ±0.03-0.04 on
   19-21 tiles; the leave-one-out minima move by ≤ 0.01 at GM and AP; the clean subsets move by 0.01.
2. **k_s and K₀ are survey-independent**: PAMAP/3DEP within 6 %, NCALM/FEMA within 5 %; site-pooled K₀ 7.2 × 10⁻⁵ (GM), 4.8 × 10⁻⁶ (AP) with
   bootstrap bands of ±7 % and ±4 %; the tile-to-tile spread (GM 4.5-9.1 × 10⁻⁵, AP 4.1-5.5 × 10⁻⁶) is the natural variability addendum 18
   will fit against.
3. **TTV is a third, distinct site**: K₀ 1.4 × 10⁻⁵ (between AP and GM), D₀,A 0.0029 (an order of magnitude below GM, like AP), C_ht 2-3 × GM's,
   the window test passing.

**Weak.**
1. **D₀ on the primary GM set is a bracket, not a measurement** (section 4): the FEMA surface fails the window test that NCALM passes, and
   its C_ht(L_f) is 22 % lower than NCALM's on the same footprints (rule 6.4). Point density (3.4 pts m⁻² against the NCALM grid's denser
   source) and the nearest-neighbour fill (0.3-1.5 % of cells) smooth the 11 m curvature, and the rollover networks differ (the crest widths).
   Which survey's C_ht is the better estimate of the hilltop curvature the model needs cannot be told from this report; addendum 18 will show
   which D₀ the fits recover.
2. **The option (iii) set collapses on 8 of 21 AP tiles** (rollover A_c of order 10 m² on the dense 3DEP networks: nearly every divide
   pixel is within 5.5 m of a channel pixel) and is small on two TTV tiles; AP11 has no hilltop pixel. Those tiles have no usable C_ht, and the
   AP site D₀ rests on 13 tiles.
3. **TTV's concavity is not one number**: the model m (0.33, with six profiles at the grid edge) and the χ-z θ (0.19 at 10³, 0.26 at 10⁴,
   0.08 equal-weight) disagree, and both are far below the M&D-era 0.45-0.55. The degenerate profiles (TTV1, 6, 7, 8, 11, 15) have their
   best A_crit at 15,000-59,000 m², i.e. almost no channel inside the tile: the partitioned model finds these tiles nearly all hillslope.
   TTV5 (the Mount Tamalpais flank) has θ = 0.00 at 10³.
4. **n_eff**: 1-4 effective basins per tile on most tiles (the per-tile k_s intervals understate the uncertainty); the site-pooled values
   are sound (n_eff 11-34).
5. **The bedrock screen's label question** (section 1.3): GM8, GM9, GM21 are Paso Robles Formation (QTcl) and are out only by the label
   reading; with them the GM primary set is 22 and S(m) gives argmin 0.29, 1 % 0.26-0.32.
6. **The GM primary m rests on 19 tiles of which 7 share footprints with the NCALM originals**; the 12 drawn FEMA tiles alone give
   0.26 (1 % 0.23-0.30), the 7 FEMA controls pull the set up (their optima 0.24-0.46), and the NCALM originals give 0.33: the GM exponent is
   0.26-0.33 depending on which tiles are asked, inside the v2 5 % interval (0.26-0.37).

## 7. Decisions needed for Addendum 18
1. **The bedrock screen's reference unit: label or formation?** As written (label QTp) the GM primary set is 19; read as the Paso Robles
   Formation (QTp + QTcl + QTg + QTm + QTpc) GM8, GM9 and GM21 return and it is 22, with the same site m (0.29). The rule was applied as
   written; the reading is Steve's to fix for addendum 18 (every number for the three tiles is in Appendix B6).
2. **D₀ for the FEMA GM tiles in addendum 18**: the bracket [D₀,A, D₀,B] as the rule gives (medians 0.015 / 0.014), or D₀ from the NCALM
   twins where they exist (0.0099 on GM1-GM7), or the FEMA value with the survey ratio (0.78) recorded. The same question for the 3DEP AP tiles
   (bracket 0.0044 / 0.0055; PAMAP 0.0032).
3. **AP11** (no option (iii) pixel) and the seven AP tiles with fewer than 150: run them with the site-median D₀, or leave them out of the
   D-recovery statements.
4. **TTV's exponent**: the site m = 0.33 (rule) against the χ-z θ 0.19-0.26 and six degenerate profiles; whether addendum 18 fits TTV at
   0.33, or reports it with the per-tile m as a sensitivity.
5. **The `measured` blocks** of the 76 new configs (site m, k_s, C_ht with provenance) are written by addendum 18 from this run
   (`measure/measurement.json`, `per_tile.csv`); the decision-7 trunk list is GM2_fema, GM4_fema, GM5_fema, AP3_3dep, AP4_3dep (go-ahead 2.5).

## 8. Appendices: all tables, the run ids, checksums, Slurm accounting

### 8.1 Cost
Estimates were printed before every submission (job scripts and logs in `$SCRATCH/lemcvt_jobs/a17b/`). Appendix B10 has the accounting of all
93 tasks in 10 submissions: **76.0 core-hours used** (expected ≤ 150: the 63 m-profiles 73.5 h at 1.0-1.3 h each, the rest 2.5 h); **sum of
CPU-weighted limits 233 core-hours** (the m-profile array 189 at 3 h per task, below the 300 per submission; 40 concurrent single-CPU tasks);
every other limit at most twice its estimate with the 30 min floor (the array's 3 h is deviation 9); peak memory 5.2 GB (a profile task);
every job on `serc`. Two submissions were wasted on the deviations of section 1 (the array that could not name the tiles, 8 × 13 s; the
run-id collision, 8 × 2.5 min). New data on group storage: the 63 profile runs (0.4 GB), 8 measure runs (40 MB), the geology map (18 MB), the
driver's outputs (6 MB); the three collided measure runs (30 MB) are kept under `_a17b_collided/`. `$GROUP_HOME` at 321 GB of 1 TB. No scratch
staging was needed; the job logs stay.

### 8.2 Pre-push check (addendum 2 section 5)
Run on the staged tree before the commit and on the commit against `origin/v2.3-tiles` before pushing: forbidden tracked files (`papers/`,
`*.pdf`, `working-with-steve.md`, `cry_for_help.txt`, `.env`, `.netrc`): **none** (the figure PDFs are ignored); files over 50 MB: **none**;
secret patterns in the diff: **none**. The 76 new tile configs and the TTV site config are committed (they hold paths and checksums, no data);
the geology map stays on group storage.

### 8.3 Generated tables (`make_a17b_tables.py`, from the run directory; also `appendix_tables.md` there)

### B1. Exclusions and tiers (go-ahead section 1)

Tier 1 (not measured, except AP1_3dep for the survey comparison; registered):

| tile | reason |
|---|---|
| GM10 | benched or hummocky ground over the whole tile (terraced ground or an old landslide complex) |
| GM23 | a large flat area in the middle of the tile (alluvial fill or a mesa remnant), not hillslope-and-channel topography |
| AP1_3dep | the main channel follows a road |
| AP12 | a flat, apparently man-made surface (pad or fill) |
| AP15 | the channel network follows a road |
| AP16 | a large graded pad and access roads on the ridge |
| AP17 | a divided highway with cut and embankment across about a third of the tile |
| AP21 | man-made flat tops (pads) on the ridges |
| AP22 | a pad and a rectangular road grid; the network follows man-made features |
| AP26 | a man-made flat top (pad) and its access road |
| TTV10 | the network follows roads |
| TTV12 | graded ground and switchback roads (Marin Headlands batteries); the network follows roads |
| TTV13 | roads and lot-scale terracing (the urban edge) |
| TTV14 | the network follows switchback roads |

Tier 2 (measured, in the primary set; the clean subset leaves them out):

| tile | reason |
|---|---|
| GM12 | a road along the divide |
| GM29 | a road along the divide |
| GM26 | switchback roads and terracing in the lower left |
| AP25 | small pads on the ridges |
| TTV11 | fire roads |

### B2. The GM bedrock screen (go-ahead 3.1), from screens/bedrock_screen.json

Map: /home/groups/hilley/lemcvt_data/source/Geology_Monterey_Rosenberg_cm427jp1187/GEOLOGY.shp (https://purl.stanford.edu/cm427jp1187; sha256 of GEOLOGY.shp edd559992b98d894..., GEOLOGY.dbf f8e4e20847f41ccf...; CRS NAD83 / California zone 4 (ftUS); 7,261 polygons). Reference unit: **QTp (Paso Robles Formation, undifferentiated)**, 98.6 % of the seven control footprints combined (candidates: QTp 98.6 %, Tucs 0.1 %). Counted Quaternary units: Qal, Qb, Qc, Qfp, Qfpl, Qfpm, Qfu, Qhf, Qsc, Qt, Qtlp, Qtmp; not counted: landslide deposits (Qls), fill, eolian, dune, beach, coastal terrace, Aromas Sand. Pass at >= 90 %.

| tile | kind | reference unit | counted Quaternary | **rule fraction** | landslide (not counted) | other bedrock | other Quaternary | map coverage | pass | Tier 1 | other units in the footprint |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GM1_fema | control | 100.0 % | 0.0 % | 100.0 % | 0.0 % | 0.0 % | 0.0 % | 100.0 % | yes | no |  |
| GM2_fema | control | 95.6 % | 4.4 % | 100.0 % | 0.0 % | 0.0 % | 0.0 % | 100.0 % | yes | no | Qal (Alluvial deposits, undifferentiated) 4.4 % |
| GM3_fema | control | 100.0 % | 0.0 % | 100.0 % | 0.0 % | 0.0 % | 0.0 % | 100.0 % | yes | no |  |
| GM4_fema | control | 100.0 % | 0.0 % | 100.0 % | 0.0 % | 0.0 % | 0.0 % | 100.0 % | yes | no |  |
| GM5_fema | control | 94.9 % | 4.5 % | 99.5 % | 0.0 % | 0.5 % | 0.0 % | 100.0 % | yes | no | Qal (Alluvial deposits, undifferentiated) 4.5 %; Tucs (Unnamed clastic sedimentary unit) 0.5 % |
| GM6_fema | control | 100.0 % | 0.0 % | 100.0 % | 0.0 % | 0.0 % | 0.0 % | 100.0 % | yes | no | Qfp (Flood-plain deposits, undifferentiated) 0.0 % |
| GM7_fema | control | 100.0 % | 0.0 % | 100.0 % | 0.0 % | 0.0 % | 0.0 % | 100.0 % | yes | no |  |
| GM8 | drawn | 87.4 % | 0.0 % | 87.4 % | 0.0 % | 12.6 % | 0.0 % | 100.0 % | no | no | QTcl (Paso Robles Formation) 12.6 % |
| GM9 | drawn | 43.0 % | 0.0 % | 43.0 % | 0.0 % | 57.0 % | 0.0 % | 100.0 % | no | no | QTcl (Paso Robles Formation) 57.0 % |
| GM10 | drawn | 100.0 % | 0.0 % | 100.0 % | 0.0 % | 0.0 % | 0.0 % | 100.0 % | yes | yes |  |
| GM11 | drawn | 100.0 % | 0.0 % | 100.0 % | 0.0 % | 0.0 % | 0.0 % | 100.0 % | yes | no |  |
| GM12 | drawn | 100.0 % | 0.0 % | 100.0 % | 0.0 % | 0.0 % | 0.0 % | 100.0 % | yes | no |  |
| GM13 | drawn | 100.0 % | 0.0 % | 100.0 % | 0.0 % | 0.0 % | 0.0 % | 100.0 % | yes | no |  |
| GM14 | drawn | 5.8 % | 1.8 % | 7.6 % | 23.9 % | 68.5 % | 0.0 % | 100.0 % | no | no | Tucs (Unnamed clastic sedimentary unit) 31.2 %; Tucc (Unnamed clastic sedimentary unit) 27.7 %; Qls (Landslide deposits) 23.9 %; Tpo (Pancho Rico Formation, mudstone) 9.5 %; Qal (Alluvial deposits, undifferentiated) 1.8 % |
| GM15 | drawn | 97.9 % | 2.1 % | 100.0 % | 0.0 % | 0.0 % | 0.0 % | 100.0 % | yes | no | Qt (Fluvial terrace deposits, undifferentiated) 2.1 % |
| GM16 | drawn | 98.2 % | 1.8 % | 100.0 % | 0.0 % | 0.0 % | 0.0 % | 100.0 % | yes | no | Qt (Fluvial terrace deposits, undifferentiated) 1.8 % |
| GM17 | drawn | 100.0 % | 0.0 % | 100.0 % | 0.0 % | 0.0 % | 0.0 % | 100.0 % | yes | no |  |
| GM18 | drawn | 0.0 % | 46.7 % | 46.7 % | 0.0 % | 53.3 % | 0.0 % | 100.0 % | no | no | Tucs (Unnamed clastic sedimentary unit) 53.3 %; Qt (Fluvial terrace deposits, undifferentiated) 46.7 % |
| GM19 | drawn | 0.0 % | 23.5 % | 23.5 % | 0.0 % | 76.5 % | 0.0 % | 100.0 % | no | no | Tpo (Pancho Rico Formation, mudstone) 76.5 %; Qt (Fluvial terrace deposits, undifferentiated) 12.9 %; Qal (Alluvial deposits, undifferentiated) 10.6 % |
| GM20 | drawn | 100.0 % | 0.0 % | 100.0 % | 0.0 % | 0.0 % | 0.0 % | 100.0 % | yes | no |  |
| GM21 | drawn | 71.3 % | 0.0 % | 71.3 % | 0.0 % | 28.7 % | 0.0 % | 100.0 % | no | no | QTcl (Paso Robles Formation) 28.7 % |
| GM22 | drawn | 100.0 % | 0.0 % | 100.0 % | 0.0 % | 0.0 % | 0.0 % | 100.0 % | yes | no |  |
| GM23 | drawn | 0.0 % | 18.7 % | 18.7 % | 0.0 % | 81.3 % | 0.0 % | 100.0 % | no | yes | Tpo (Pancho Rico Formation, mudstone) 79.1 %; Qal (Alluvial deposits, undifferentiated) 18.7 %; Tm (Monterey Formation, siliceous) 2.2 % |
| GM24 | drawn | 97.7 % | 0.0 % | 97.7 % | 0.0 % | 2.3 % | 0.0 % | 100.0 % | yes | no | QTcl (Paso Robles Formation) 2.3 % |
| GM25 | drawn | 18.1 % | 1.0 % | 19.1 % | 0.0 % | 80.9 % | 0.0 % | 100.0 % | no | no | Tucs (Unnamed clastic sedimentary unit) 80.9 %; Qfp (Flood-plain deposits, undifferentiated) 1.0 % |
| GM26 | drawn | 100.0 % | 0.0 % | 100.0 % | 0.0 % | 0.0 % | 0.0 % | 100.0 % | yes | no |  |
| GM27 | drawn | 97.3 % | 0.0 % | 97.3 % | 0.0 % | 2.7 % | 0.0 % | 100.0 % | yes | no | Tucs (Unnamed clastic sedimentary unit) 2.7 % |
| GM28 | drawn | 55.7 % | 25.1 % | 80.8 % | 0.0 % | 19.2 % | 0.0 % | 100.0 % | no | no | Qal (Alluvial deposits, undifferentiated) 25.1 %; Tucc (Unnamed clastic sedimentary unit) 19.2 % |
| GM29 | drawn | 100.0 % | 0.0 % | 100.0 % | 0.0 % | 0.0 % | 0.0 % | 100.0 % | yes | no |  |
| GM30 | drawn | 0.0 % | 26.3 % | 26.3 % | 0.0 % | 73.7 % | 0.0 % | 100.0 % | no | no | Tpo (Pancho Rico Formation, mudstone) 59.4 %; Qal (Alluvial deposits, undifferentiated) 26.3 %; Tpo (Pancho Rico Formation, mudstone) 13.7 %; Tm (Monterey Formation, siliceous) 0.6 % |

Fail (off-mesa): ['GM8', 'GM9', 'GM14', 'GM18', 'GM19', 'GM21', 'GM23', 'GM25', 'GM28', 'GM30']; control tiles failing: none.

### B3. The gap-fill screen on the measurement pixels (go-ahead 3.2), from screens/gapfill_screen.json

Filled cells = nnfill_mask == 1 (nearest-neighbour fill within 10 m) or gapfill_mask == 1 (enclosed gap filled for routing). (a) = channel pixels of the k_s fit (A >= 1000 m2, kept basins); (b) = option (iii) hilltop pixels at L_f = 11 m; a tile leaves the primary set above 20 % on either.

| tile | site | Tier 2 | channel px (kept basins) | basins kept / all | **(a) filled on channel px** | hilltop px | **(b) filled on hilltop px** | filled, whole tile | rollover A_c (m2) | measure run | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GM1_fema | GM | no | 9,249 | 20 / 68 | 1.5 % | 8,554 | 1.4 % | 1.5 % | 177.8 | 20261119T012857Z_measure_a17b_t0 | yes |
| GM2_fema | GM | no | 12,135 | 12 / 43 | 1.1 % | 5,425 | 0.3 % | 0.4 % | 112.2 | 20261119T012858Z_measure_a17b_t1 | yes |
| GM3_fema | GM | no | 12,026 | 16 / 49 | 0.7 % | 6,283 | 0.3 % | 0.3 % | 56.234 | 20261119T012858Z_measure_a17b_t2 | yes |
| GM4_fema | GM | no | 12,540 | 18 / 45 | 1.7 % | 5,676 | 0.7 % | 0.9 % | 177.8 | 20261119T012858Z_measure_a17b_t3 | yes |
| GM5_fema | GM | no | 14,209 | 4 / 53 | 0.7 % | 3,310 | 0.3 % | 0.5 % | 70.795 | 20261119T012858Z_measure_a17b_t4 | yes |
| GM6_fema | GM | no | 12,449 | 26 / 49 | 0.4 % | 4,549 | 0.2 % | 0.3 % | 354.8 | 20261119T012858Z_measure_a17b_t5 | yes |
| GM7_fema | GM | no | 10,868 | 8 / 57 | 1.2 % | 3,585 | 1.3 % | 0.5 % | 562.3 | 20261119T012859Z_measure_a17b_t6 | yes |
| GM8 | GM | no | 11,759 | 5 / 31 | 2.8 % | 4,963 | 1.2 % | 1.7 % | 354.8 | 20261119T012857Z_measure_a17b_t7 | yes |
| GM9 | GM | no | 14,392 | 14 / 66 | 2.6 % | 3,779 | 1.2 % | 1.4 % | 354.8 | 20261119T012857Z_measure_a17b_t0 | yes |
| GM11 | GM | no | 13,360 | 7 / 30 | 0.8 % | 4,273 | 2.0 % | 0.9 % | 354.8 | 20261119T012858Z_measure_a17b_t1 | yes |
| GM12 | GM | yes | 11,987 | 21 / 40 | 1.9 % | 7,208 | 1.7 % | 1.5 % | 177.8 | 20261119T012858Z_measure_a17b_t2 | yes |
| GM13 | GM | no | 12,765 | 14 / 35 | 1.9 % | 5,046 | 0.8 % | 1.1 % | 354.8 | 20261119T012858Z_measure_a17b_t3 | yes |
| GM14 | GM | no | 19,675 | 18 / 80 | 8.5 % | 6,475 | 8.2 % | 8.5 % | 177.8 | 20261119T012858Z_measure_a17b_t4 | yes |
| GM15 | GM | no | 18,481 | 8 / 74 | 1.5 % | 4,822 | 1.9 % | 2.9 % | 141.3 | 20261119T012858Z_measure_a17b_t5 | yes |
| GM16 | GM | no | 10,169 | 9 / 37 | 1.4 % | 6,454 | 0.7 % | 0.8 % | 141.3 | 20261119T012859Z_measure_a17b_t6 | yes |
| GM17 | GM | no | 10,640 | 6 / 29 | 1.0 % | 4,224 | 0.9 % | 0.9 % | 177.8 | 20261119T012857Z_measure_a17b_t7 | yes |
| GM18 | GM | no | 21,369 | 7 / 112 | 3.9 % | 2,655 | 5.0 % | 3.2 % | 89.125 | 20261119T012857Z_measure_a17b_t0 | yes |
| GM19 | GM | no | 11,643 | 13 / 42 | 0.9 % | 7,075 | 1.4 % | 1.2 % | 177.8 | 20261119T012858Z_measure_a17b_t1 | yes |
| GM20 | GM | no | 10,159 | 7 / 46 | 1.5 % | 5,234 | 0.6 % | 0.9 % | 112.2 | 20261119T012858Z_measure_a17b_t2 | yes |
| GM21 | GM | no | 13,042 | 17 / 58 | 3.8 % | 7,692 | 3.6 % | 4.3 % | 281.8 | 20261119T012858Z_measure_a17b_t3 | yes |
| GM22 | GM | no | 11,816 | 5 / 47 | 0.5 % | 2,509 | 0.2 % | 0.4 % | 223.9 | 20261119T012858Z_measure_a17b_t4 | yes |
| GM24 | GM | no | 11,747 | 10 / 41 | 1.7 % | 6,482 | 0.7 % | 0.9 % | 223.9 | 20261119T012858Z_measure_a17b_t5 | yes |
| GM25 | GM | no | 12,031 | 8 / 62 | 1.8 % | 5,909 | 5.5 % | 3.4 % | 562.3 | 20261119T012859Z_measure_a17b_t6 | yes |
| GM26 | GM | yes | 14,889 | 6 / 57 | 1.6 % | 3,448 | 0.7 % | 0.9 % | 141.3 | 20261119T012857Z_measure_a17b_t7 | yes |
| GM27 | GM | no | 13,931 | 11 / 64 | 3.5 % | 5,349 | 4.4 % | 7.2 % | 141.3 | 20261119T012857Z_measure_a17b_t0 | yes |
| GM28 | GM | no | 18,455 | 13 / 55 | 5.2 % | 6,318 | 6.3 % | 7.8 % | 177.8 | 20261119T012858Z_measure_a17b_t1 | yes |
| GM29 | GM | yes | 11,694 | 16 / 47 | 3.9 % | 6,738 | 1.9 % | 2.8 % | 223.9 | 20261119T012858Z_measure_a17b_t2 | yes |
| GM30 | GM | no | 15,793 | 11 / 39 | 2.4 % | 6,877 | 2.6 % | 2.1 % | 112.2 | 20261119T012858Z_measure_a17b_t3 | yes |
| AP2_3dep | AP | no | 29,501 | 9 / 112 | 6.0 % | 13 | 7.7 % | 2.8 % | 11.220 | 20261119T012858Z_measure_a17b_t4 | yes |
| AP3_3dep | AP | no | 24,045 | 38 / 136 | 3.5 % | 4,094 | 1.3 % | 1.9 % | 177.8 | 20261119T012858Z_measure_a17b_t5 | yes |
| AP4_3dep | AP | no | 21,775 | 23 / 84 | 3.6 % | 6,417 | 2.5 % | 1.6 % | 281.8 | 20261119T012859Z_measure_a17b_t6 | yes |
| AP5_3dep | AP | no | 21,956 | 12 / 71 | 4.8 % | 5,502 | 1.6 % | 3.0 % | 281.8 | 20261119T012857Z_measure_a17b_t7 | yes |
| AP6 | AP | no | 23,952 | 20 / 94 | 4.2 % | 4,391 | 2.2 % | 2.8 % | 354.8 | 20261119T012857Z_measure_a17b_t0 | yes |
| AP7 | AP | no | 24,891 | 9 / 77 | 3.4 % | 2,756 | 3.9 % | 1.5 % | 281.8 | 20261119T012858Z_measure_a17b_t1 | yes |
| AP8 | AP | no | 25,206 | 11 / 77 | 3.8 % | 3,340 | 0.5 % | 2.1 % | 223.9 | 20261119T012858Z_measure_a17b_t2 | yes |
| AP9 | AP | no | 21,722 | 32 / 119 | 4.4 % | 30 | 0.0 % | 3.7 % | 11.220 | 20261119T012858Z_measure_a17b_t3 | yes |
| AP10 | AP | no | 21,386 | 37 / 93 | 4.5 % | 145 | 0.0 % | 2.0 % | 11.220 | 20261119T012858Z_measure_a17b_t4 | yes |
| AP11 | AP | no | 24,973 | 7 / 89 | 4.1 % | 0 | - | 2.3 % | 11.220 | 20261119T012858Z_measure_a17b_t5 | yes |
| AP13 | AP | no | 24,825 | 15 / 94 | 5.9 % | 3,087 | 1.9 % | 2.7 % | 354.8 | 20261119T012859Z_measure_a17b_t6 | yes |
| AP14 | AP | no | 23,891 | 31 / 121 | 3.9 % | 118 | 0.0 % | 2.1 % | 11.220 | 20261119T012857Z_measure_a17b_t7 | yes |
| AP18 | AP | no | 25,812 | 12 / 116 | 5.7 % | 25 | 4.0 % | 2.5 % | 11.220 | 20261119T012857Z_measure_a17b_t0 | yes |
| AP19 | AP | no | 24,918 | 20 / 105 | 3.6 % | 6,160 | 1.2 % | 2.0 % | 354.8 | 20261119T012858Z_measure_a17b_t1 | yes |
| AP20 | AP | no | 23,974 | 38 / 73 | 5.4 % | 6,710 | 5.2 % | 2.8 % | 223.9 | 20261119T012858Z_measure_a17b_t2 | yes |
| AP23 | AP | no | 28,177 | 4 / 80 | 4.2 % | 2,197 | 4.6 % | 2.9 % | 281.8 | 20261119T012858Z_measure_a17b_t3 | yes |
| AP24 | AP | no | 24,100 | 18 / 104 | 3.7 % | 5,040 | 3.4 % | 2.8 % | 223.9 | 20261119T012858Z_measure_a17b_t4 | yes |
| AP25 | AP | yes | 22,381 | 6 / 71 | 4.8 % | 1,124 | 2.1 % | 2.3 % | 112.2 | 20261119T012858Z_measure_a17b_t5 | yes |
| AP27 | AP | no | 20,943 | 20 / 97 | 6.1 % | 83 | 2.4 % | 2.4 % | 11.220 | 20261119T012859Z_measure_a17b_t6 | yes |
| AP28 | AP | no | 22,821 | 28 / 97 | 4.3 % | 119 | 0.0 % | 2.2 % | 11.220 | 20261119T012857Z_measure_a17b_t7 | yes |
| AP29 | AP | no | 21,504 | 13 / 62 | 4.5 % | 3,606 | 1.0 % | 1.9 % | 223.9 | 20261119T012857Z_measure_a17b_t0 | yes |
| TTV1 | TTV | no | 18,475 | 23 / 88 | 6.0 % | 966 | 3.5 % | 5.7 % | 35.481 | 20261119T012858Z_measure_a17b_t1 | yes |
| TTV2 | TTV | no | 16,464 | 16 / 73 | 4.6 % | 1,995 | 2.4 % | 4.4 % | 56.234 | 20261119T012858Z_measure_a17b_t2 | yes |
| TTV3 | TTV | no | 22,376 | 7 / 89 | 3.3 % | 122 | 0.0 % | 2.6 % | 17.783 | 20261119T012858Z_measure_a17b_t3 | yes |
| TTV4 | TTV | no | 20,419 | 24 / 84 | 13.4 % | 86 | 0.0 % | 6.9 % | 11.220 | 20261119T012858Z_measure_a17b_t4 | yes |
| TTV5 | TTV | no | 15,024 | 22 / 68 | 6.9 % | 3,203 | 3.4 % | 4.9 % | 44.668 | 20261119T012858Z_measure_a17b_t5 | yes |
| TTV6 | TTV | no | 21,469 | 43 / 87 | 7.0 % | 612 | 4.2 % | 3.5 % | 22.387 | 20261119T012859Z_measure_a17b_t6 | yes |
| TTV7 | TTV | no | 14,447 | 20 / 59 | 11.1 % | 2,983 | 9.0 % | 10.4 % | 56.234 | 20261119T012857Z_measure_a17b_t7 | yes |
| TTV8 | TTV | no | 15,273 | 23 / 64 | 19.1 % | 5,287 | 6.1 % | 14.1 % | 89.125 | 20261119T012857Z_measure_a17b_t0 | yes |
| TTV9 | TTV | no | 15,563 | 13 / 78 | 17.9 % | 3,291 | 3.1 % | 15.7 % | 177.8 | 20261119T012858Z_measure_a17b_t1 | yes |
| TTV11 | TTV | yes | 16,097 | 28 / 86 | 14.5 % | 3,192 | 4.3 % | 9.2 % | 44.668 | 20261119T012858Z_measure_a17b_t2 | yes |
| TTV15 | TTV | no | 15,861 | 26 / 73 | 11.5 % | 579 | 0.5 % | 8.0 % | 17.783 | 20261119T012858Z_measure_a17b_t3 | yes |

Fail (gap-filled): none.

### B4. The primary set, fixed before the m-profile runs, from screens/primary_set.json

Fixed at 2026-11-19T01:38:55Z UTC. **GM: 19** (expected 28); **AP: 21** (expected 21); **TTV: 11** (expected 11).

| site | primary tiles | removed by the screens | clean subset (minus Tier 2) |
|---|---|---|---|
| GM | GM1_fema, GM2_fema, GM3_fema, GM4_fema, GM5_fema, GM6_fema, GM7_fema, GM11, GM12, GM13, GM15, GM16, GM17, GM20, GM22, GM24, GM26, GM27, GM29 | GM8: off-mesa (bedrock screen); GM9: off-mesa (bedrock screen); GM14: off-mesa (bedrock screen); GM18: off-mesa (bedrock screen); GM19: off-mesa (bedrock screen); GM21: off-mesa (bedrock screen); GM25: off-mesa (bedrock screen); GM28: off-mesa (bedrock screen); GM30: off-mesa (bedrock screen) | GM1_fema, GM2_fema, GM3_fema, GM4_fema, GM5_fema, GM6_fema, GM7_fema, GM11, GM13, GM15, GM16, GM17, GM20, GM22, GM24, GM27 |
| AP | AP2_3dep, AP3_3dep, AP4_3dep, AP5_3dep, AP6, AP7, AP8, AP9, AP10, AP11, AP13, AP14, AP18, AP19, AP20, AP23, AP24, AP25, AP27, AP28, AP29 | none | AP2_3dep, AP3_3dep, AP4_3dep, AP5_3dep, AP6, AP7, AP8, AP9, AP10, AP11, AP13, AP14, AP18, AP19, AP20, AP23, AP24, AP27, AP28, AP29 |
| TTV | TTV1, TTV2, TTV3, TTV4, TTV5, TTV6, TTV7, TTV8, TTV9, TTV11, TTV15 | none | TTV1, TTV2, TTV3, TTV4, TTV5, TTV6, TTV7, TTV8, TTV9, TTV15 |

### B5. The site exponent profiles S(m) (section 5.2), from measure/measurement.json

| site | group | tiles | **m = argmin S** | S_min / n | 1 % interval | 5 % interval | note |
|---|---|---|---|---|---|---|---|
| GM | primary | 19 | 0.290 | 1.075 | [0.26, 0.31] | [0.22, 0.35] |  |
| GM | clean | 16 | 0.280 | 1.083 | [0.25, 0.31] | [0.22, 0.35] |  |
| GM | ncalm | 7 | 0.330 | 1.044 | [0.31, 0.36] | [0.27, 0.40] |  |
| AP | primary | 21 | 0.470 | 1.054 | [0.43, 0.51] | [0.38, 0.60] |  |
| AP | clean | 20 | 0.480 | 1.043 | [0.45, 0.53] | [0.39, 0.62] |  |
| AP | pamap | 5 | 0.470 | 1.073 | [0.44, 0.49] | [0.41, 0.53] |  |
| TTV | primary | 11 | 0.330 | 1.095 | [0.28, 0.39] | [0.15, 0.46] |  |
| TTV | clean | 10 | 0.350 | 1.070 | [0.30, 0.40] | [0.21, 0.47] |  |

GM primary set, leave-one-out minima: GM1_fema 0.29, GM2_fema 0.29, GM3_fema 0.29, GM4_fema 0.28, GM5_fema 0.28, GM6_fema 0.28, GM7_fema 0.28, GM11 0.28, GM12 0.28, GM13 0.28, GM15 0.29, GM16 0.28, GM17 0.28, GM20 0.29, GM22 0.29, GM24 0.29, GM26 0.29, GM27 0.30, GM29 0.28.

GM per-tile optima (m, A_crit): GM1_fema (0.26, 1,217), GM2_fema (0.29, 1,496), GM3_fema (0.24, 698), GM4_fema (0.34, 2,701), GM5_fema (0.32, 2,024), GM6_fema (0.45, 9,355), GM7_fema (0.46, 3,106), GM11 (0.35, 3,330), GM12 (0.35, 4,877), GM13 (0.31, 3,329), GM15 (0.05, 6,221), GM16 (0.32, 6,003), GM17 (0.37, 669), GM20 (0.29, 6,903), GM22 (0.11, 10,107), GM24 (0.21, 5,607), GM26 (0.25, 9,092), GM27 (0.05, 1,976), GM29 (0.37, 1,350).

AP primary set, leave-one-out minima: AP2_3dep 0.46, AP3_3dep 0.48, AP4_3dep 0.47, AP5_3dep 0.49, AP6 0.47, AP7 0.47, AP8 0.47, AP9 0.46, AP10 0.47, AP11 0.47, AP13 0.47, AP14 0.47, AP18 0.47, AP19 0.47, AP20 0.47, AP23 0.47, AP24 0.47, AP25 0.48, AP27 0.47, AP28 0.48, AP29 0.47.

AP per-tile optima (m, A_crit): AP2_3dep (0.60, 7,335), AP3_3dep (0.33, 14,122), AP4_3dep (0.44, 5,039), AP5_3dep (0.39, 1,252), AP6 (0.58, 11,541), AP7 (0.47, 5,543), AP8 (0.52, 9,119), AP9 (0.59, 5,631), AP10 (0.70, 7,932), AP11 (0.55, 8,761), AP13 (0.42, 6,215), AP14 (0.51, 7,399), AP18 (0.63, 17,803), AP19 (0.58, 6,677), AP20 (0.49, 2,691), AP23 (0.47, 5,992), AP24 (0.45, 4,532), AP25 (0.28, 9,769), AP27 (0.51, 5,034), AP28 (0.41, 4,550), AP29 (0.53, 16,432).

TTV primary set, leave-one-out minima: TTV1 0.33, TTV2 0.22, TTV3 0.33, TTV4 0.33, TTV5 0.33, TTV6 0.35, TTV7 0.35, TTV8 0.33, TTV9 0.31, TTV11 0.35, TTV15 0.34.

TTV per-tile optima (m, A_crit): TTV1 (0.05, 59,405), TTV2 (0.45, 8,608), TTV3 (0.40, 92,563), TTV4 (0.42, 40,368), TTV5 (0.41, 58,329), TTV6 (0.05, 15,154), TTV7 (0.05, 36,147), TTV8 (0.06, 38,520), TTV9 (0.35, 37,399), TTV11 (0.05, 30,346), TTV15 (0.05, 27,570).

Rule 6.2 (site m stability): GM: v2 0.31, primary 0.29 (moved > 0.05: no), NCALM GM1-GM7 0.330 (moved: no); AP: v2 0.47, primary 0.47 (moved > 0.05: no), PAMAP AP1-AP5 recomputed 0.470.

### B6. Per-tile measurements at the site m (sections 5.1, 5.3, 5.4), from measure/per_tile.csv

| tile | site | group | m_site | theta (1e3, pooled) | 95 % | theta (1e3, equal) | theta (1e4, pooled) | basins (1e3) | theta at rollover | A_c rollover | k_s (1e3) | 95 % | n_eff | K0 = E/k_s | k_s (1e4) | C_ht(L_f) | 95 % | D0_A | L_w (m) | C_ht(L_w) | D0_B | hilltop px | segments | median crest width (m) | windows tested | max |dev| | window test |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM1_fema | GM | primary | 0.290 | 0.320 | [0.30, 0.38] | 0.290 | 0.380 | 20 | 0.250 | 177.8 | 2.384 | [2.254, 2.466] | 3.464 | 6.166e-05 | 2.294 | 0.014 | [0.0134, 0.0153] | 0.010 | 41 | 0.013 | 0.011 | 8,554 | 1,340 | 44.045 | 6 | 0.064 | yes |
| GM2_fema | GM | primary | 0.290 | 0.310 | [0.19, 0.36] | 0.220 | 0.260 | 12 | 0.280 | 112.2 | 2.020 | [1.955, 2.189] | 3.235 | 7.276e-05 | 1.898 | 0.012 | [0.0105, 0.0135] | 0.013 | 31 | 0.012 | 0.012 | 5,425 | 1,005 | 32.311 | 5 | 0.045 | yes |
| GM3_fema | GM | primary | 0.290 | 0.320 | [0.27, 0.37] | 0.320 | 0.400 | 16 | 0.260 | 56.234 | 1.827 | [1.687, 2.171] | 2.916 | 8.045e-05 | 1.782 | 9.512e-03 | [8.724e-03, 0.0108] | 0.015 | 15 | 9.287e-03 | 0.016 | 6,283 | 708 | 19.799 | 2 | 0.024 | yes |
| GM4_fema | GM | primary | 0.290 | 0.330 | [0.25, 0.35] | 0.320 | 0.310 | 18 | 0.290 | 177.8 | 3.238 | [3.001, 3.782] | 1.506 | 4.540e-05 | 3.041 | 0.012 | [0.0109, 0.0148] | 0.012 | 41 | 0.013 | 0.011 | 5,676 | 1,029 | 48.415 | 6 | 0.103 | no |
| GM5_fema | GM | primary | 0.290 | 0.300 | [0.25, 0.66] | 0.290 | 0.300 | 4 | 0.270 | 70.795 | 2.265 | [1.395, 2.278] | 1.199 | 6.490e-05 | 2.177 | 0.017 | [0.0154, 0.0183] | 8.725e-03 | 25 | 0.017 | 8.471e-03 | 3,310 | 536 | 26.077 | 4 | 0.030 | yes |
| GM6_fema | GM | primary | 0.290 | 0.340 | [0.08, 0.37] | 0.250 | 0.390 | 26 | 0.320 | 354.8 | 2.147 | [2.010, 3.749] | 1.807 | 6.847e-05 | 1.939 | 9.044e-03 | [7.700e-03, 0.0109] | 0.016 | 51 | 0.010 | 0.014 | 4,549 | 1,014 | 68.029 | 7 | 0.135 | no |
| GM7_fema | GM | primary | 0.290 | 0.310 | [0.18, 0.40] | 0.160 | 0.440 | 8 | 0.320 | 562.3 | 1.781 | [1.683, 1.887] | 2.245 | 8.255e-05 | 1.708 | 4.148e-03 | [3.697e-03, 4.578e-03] | 0.035 | 51 | 4.113e-03 | 0.036 | 3,585 | 557 | 86.023 | 7 | 0.065 | yes |
| GM8 | GM | screened | 0.290 | 0.350 | [0.28, 0.36] | 0.330 | 0.320 | 5 | 0.320 | 354.8 | 1.738 | [1.414, 2.207] | 3.490 | 8.457e-05 | 1.604 | 9.652e-03 | [8.511e-03, 0.0108] | 0.015 | 51 | 9.399e-03 | 0.016 | 4,963 | 399 | 59.666 | 7 | 0.073 | yes |
| GM9 | GM | screened | 0.290 | 0.400 | [0.29, 0.43] | 0.360 | 0.460 | 14 | 0.370 | 354.8 | 1.775 | [1.609, 2.276] | 2.500 | 8.281e-05 | 1.599 | 7.219e-03 | [6.683e-03, 7.935e-03] | 0.020 | 51 | 6.968e-03 | 0.021 | 3,779 | 777 | 65.513 | 7 | 0.041 | yes |
| GM11 | GM | primary | 0.290 | 0.340 | [0.28, 0.40] | 0.360 | 0.250 | 7 | 0.330 | 354.8 | 1.834 | [1.665, 2.436] | 2.181 | 8.016e-05 | 1.710 | 8.028e-03 | [6.817e-03, 9.258e-03] | 0.018 | 51 | 7.448e-03 | 0.020 | 4,273 | 657 | 72.471 | 7 | 0.072 | yes |
| GM12 | GM | primary (Tier 2) | 0.290 | 0.290 | [0.15, 0.33] | 0.190 | 0.220 | 21 | 0.310 | 177.8 | 1.960 | [1.766, 3.586] | 2.542 | 7.500e-05 | 1.764 | 0.013 | [0.0111, 0.0145] | 0.011 | 31 | 0.012 | 0.013 | 7,208 | 1,421 | 36.770 | 5 | 0.110 | no |
| GM13 | GM | primary | 0.290 | 0.320 | [0.29, 0.37] | 0.290 | 0.260 | 14 | 0.300 | 354.8 | 2.204 | [1.974, 2.443] | 3.084 | 6.670e-05 | 2.021 | 5.095e-03 | [4.669e-03, 5.618e-03] | 0.029 | 51 | 5.277e-03 | 0.028 | 5,046 | 834 | 68.118 | 7 | 0.036 | yes |
| GM14 | GM | screened | 0.290 | 0.230 | [0.17, 0.33] | 0.270 | 0.110 | 18 | 0.220 | 177.8 | 2.125 | [1.225, 2.405] | 2.030 | 6.918e-05 | 2.396 | 0.016 | [0.0130, 0.0181] | 9.355e-03 | 31 | 0.013 | 0.012 | 6,475 | 1,119 | 36.770 | 5 | 0.201 | no |
| GM15 | GM | primary | 0.290 | 0.300 | [0.26, 0.38] | 0.300 | 0.340 | 8 | 0.250 | 141.3 | 2.270 | [1.704, 2.709] | 3.251 | 6.475e-05 | 2.326 | 0.016 | [0.0144, 0.0181] | 8.980e-03 | 31 | 0.014 | 0.011 | 4,822 | 640 | 32.802 | 5 | 0.163 | no |
| GM16 | GM | primary | 0.290 | 0.340 | [0.33, 0.35] | 0.320 | 0.290 | 9 | 0.260 | 141.3 | 2.207 | [2.116, 2.386] | 3.596 | 6.660e-05 | 2.072 | 0.012 | [0.0100, 0.0137] | 0.012 | 31 | 0.014 | 0.010 | 6,454 | 1,252 | 38.419 | 5 | 0.175 | no |
| GM17 | GM | primary | 0.290 | 0.400 | [0.00, 0.43] | 0.310 | 0.460 | 6 | 0.340 | 177.8 | 2.054 | [2.024, 3.222] | 1.631 | 7.158e-05 | 1.937 | 9.473e-03 | [7.635e-03, 0.0113] | 0.016 | 31 | 7.400e-03 | 0.020 | 4,224 | 474 | 37.363 | 5 | 0.219 | no |
| GM18 | GM | screened | 0.290 | 0.260 | [0.26, 0.63] | 0.270 | 0.220 | 7 | 0.260 | 89.125 | 1.465 | [1.199, 2.151] | 1.015 | 1.003e-04 | 1.557 | 0.028 | [0.0204, 0.0327] | 5.326e-03 | 25 | 0.023 | 6.456e-03 | 2,655 | 553 | 27.857 | 4 | 0.175 | no |
| GM19 | GM | screened | 0.290 | 0.240 | [0.00, 0.29] | 0.080 | 0.350 | 13 | 0.230 | 177.8 | 2.388 | [1.020, 2.505] | 2.179 | 6.155e-05 | 2.441 | 0.012 | [0.0106, 0.0132] | 0.012 | 41 | 0.012 | 0.012 | 7,075 | 1,745 | 42.426 | 6 | 0.048 | yes |
| GM20 | GM | primary | 0.290 | 0.290 | [0.28, 0.33] | 0.280 | 0.330 | 7 | 0.210 | 112.2 | 1.733 | [0.787, 2.004] | 2.640 | 8.482e-05 | 1.936 | 8.055e-03 | [7.014e-03, 9.018e-03] | 0.018 | 15 | 7.621e-03 | 0.019 | 5,234 | 958 | 19.698 | 2 | 0.054 | yes |
| GM21 | GM | screened | 0.290 | 0.250 | [0.22, 0.33] | 0.210 | 0.290 | 17 | 0.220 | 281.8 | 2.369 | [2.120, 3.338] | 2.290 | 6.205e-05 | 2.375 | 0.012 | [0.0113, 0.0135] | 0.012 | 41 | 0.012 | 0.013 | 7,692 | 1,441 | 43.681 | 6 | 0.063 | yes |
| GM22 | GM | primary | 0.290 | 0.280 | [0.13, 0.28] | 0.240 | 0.240 | 5 | 0.280 | 223.9 | 2.083 | [1.662, 2.090] | 1.061 | 7.056e-05 | 2.067 | 9.136e-03 | [8.002e-03, 9.951e-03] | 0.016 | 31 | 7.785e-03 | 0.019 | 2,509 | 489 | 36.056 | 5 | 0.148 | no |
| GM24 | GM | primary | 0.290 | 0.330 | [0.25, 0.33] | 0.230 | 0.240 | 10 | 0.300 | 223.9 | 1.887 | [1.721, 2.321] | 1.476 | 7.790e-05 | 1.773 | 4.701e-03 | [4.314e-03, 5.139e-03] | 0.031 | 41 | 4.642e-03 | 0.032 | 6,482 | 1,266 | 45.343 | 6 | 0.021 | yes |
| GM25 | GM | screened | 0.290 | 0.280 | [0.21, 0.34] | 0.350 | 0.450 | 8 | 0.270 | 562.3 | 2.254 | [2.037, 2.472] | 3.354 | 6.522e-05 | 2.173 | 0.014 | [0.0125, 0.0150] | 0.011 | 51 | 0.012 | 0.012 | 5,909 | 785 | 79.649 | 7 | 0.105 | no |
| GM26 | GM | primary (Tier 2) | 0.290 | 0.250 | [0.00, 0.25] | 0.150 | 0.230 | 6 | 0.290 | 141.3 | 1.609 | [1.539, 2.112] | 1.096 | 9.138e-05 | 1.583 | 8.447e-03 | [7.481e-03, 0.0101] | 0.017 | 31 | 7.732e-03 | 0.019 | 3,448 | 531 | 31.113 | 5 | 0.085 | yes |
| GM27 | GM | primary | 0.290 | 0.340 | [0.14, 0.43] | 0.260 | 0.380 | 11 | 0.280 | 141.3 | 1.937 | [1.402, 2.395] | 2.932 | 7.589e-05 | 2.053 | 0.020 | [0.0189, 0.0211] | 7.280e-03 | 31 | 0.018 | 8.165e-03 | 5,349 | 747 | 39.446 | 5 | 0.108 | no |
| GM28 | GM | screened | 0.290 | 0.320 | [0.09, 0.37] | 0.210 | 0.300 | 13 | 0.300 | 177.8 | 2.263 | [1.824, 2.621] | 1.760 | 6.496e-05 | 2.060 | 0.010 | [8.686e-03, 0.0117] | 0.015 | 31 | 8.558e-03 | 0.017 | 6,318 | 1,026 | 38.053 | 5 | 0.149 | no |
| GM29 | GM | primary (Tier 2) | 0.290 | 0.350 | [0.15, 0.41] | 0.290 | 0.280 | 16 | 0.320 | 223.9 | 2.106 | [1.968, 2.622] | 3.053 | 6.981e-05 | 2.008 | 0.010 | [8.839e-03, 0.0118] | 0.015 | 51 | 0.011 | 0.014 | 6,738 | 1,144 | 52.345 | 7 | 0.047 | yes |
| GM30 | GM | screened | 0.290 | 0.300 | [0.04, 0.36] | 0.170 | 0.370 | 11 | 0.290 | 112.2 | 1.525 | [1.420, 1.800] | 2.497 | 9.642e-05 | 1.444 | 0.013 | [0.0114, 0.0153] | 0.011 | 31 | 0.013 | 0.012 | 6,877 | 1,180 | 33.287 | 5 | 0.048 | yes |
| AP2_3dep | AP | primary | 0.470 | 0.390 | [0.25, 0.42] | 0.350 | 0.510 | 9 | 0.290 | 11.220 | 10.898 | [10.460, 12.604] | 1.953 | 5.047e-06 | 12.019 | 0.020 | [-, -] | 2.742e-03 | 11 | 0.020 | 2.742e-03 | 13 | 3 | 11.662 | 1 | 0.000 | yes |
| AP3_3dep | AP | primary | 0.470 | 0.370 | [0.12, 0.51] | 0.270 | 0.380 | 38 | 0.320 | 177.8 | 11.035 | [8.192, 11.313] | 1.253 | 4.984e-06 | 11.893 | 0.013 | [0.0114, 0.0136] | 4.372e-03 | 25 | 0.010 | 5.272e-03 | 4,094 | 1,144 | 30.463 | 4 | 0.171 | no |
| AP4_3dep | AP | primary | 0.470 | 0.360 | [0.31, 0.38] | 0.310 | 0.610 | 23 | 0.310 | 281.8 | 11.432 | [8.490, 12.136] | 2.477 | 4.811e-06 | 12.066 | 0.010 | [9.098e-03, 0.0115] | 5.360e-03 | 25 | 7.281e-03 | 7.554e-03 | 6,417 | 1,527 | 28.284 | 4 | 0.290 | no |
| AP5_3dep | AP | primary | 0.470 | 0.390 | [0.32, 0.74] | 0.380 | 0.520 | 12 | 0.350 | 281.8 | 10.380 | [5.631, 12.139] | 2.141 | 5.299e-06 | 11.878 | 0.011 | [9.400e-03, 0.0125] | 5.144e-03 | 41 | 8.470e-03 | 6.494e-03 | 5,502 | 1,039 | 47.539 | 6 | 0.208 | no |
| AP6 | AP | primary | 0.470 | 0.430 | [0.00, 0.43] | 0.360 | 0.550 | 20 | 0.380 | 354.8 | 11.653 | [9.865, 11.699] | 1.229 | 4.720e-06 | 12.145 | 9.240e-03 | [7.054e-03, 0.0118] | 5.952e-03 | 41 | 7.620e-03 | 7.218e-03 | 4,391 | 1,176 | 46.000 | 6 | 0.175 | no |
| AP7 | AP | primary | 0.470 | 0.380 | [0.00, 0.39] | 0.310 | 0.480 | 9 | 0.350 | 281.8 | 10.254 | [9.101, 12.178] | 2.085 | 5.364e-06 | 11.349 | 0.010 | [8.795e-03, 0.0124] | 5.239e-03 | 41 | 9.530e-03 | 5.771e-03 | 2,756 | 755 | 42.426 | 6 | 0.102 | no |
| AP8 | AP | primary | 0.470 | 0.380 | [0.15, 0.39] | 0.330 | 0.400 | 11 | 0.350 | 223.9 | 10.067 | [9.270, 13.349] | 1.483 | 5.463e-06 | 10.751 | 9.130e-03 | [7.480e-03, 0.0109] | 6.024e-03 | 41 | 7.147e-03 | 7.696e-03 | 3,340 | 737 | 50.000 | 6 | 0.217 | no |
| AP9 | AP | primary | 0.470 | 0.350 | [0.09, 0.37] | 0.280 | 0.680 | 32 | 0.220 | 11.220 | 12.000 | [10.316, 12.722] | 2.223 | 4.583e-06 | 13.816 | 0.030 | [0.0273, 0.0788] | 1.847e-03 | 11 | 0.030 | 1.847e-03 | 30 | 9 | 11.662 | 1 | 0.000 | yes |
| AP10 | AP | primary | 0.470 | 0.360 | [0.00, 0.37] | 0.270 | 0.680 | 37 | 0.230 | 11.220 | 13.179 | [12.353, 13.926] | 2.644 | 4.173e-06 | 14.389 | 0.021 | [0.0172, 0.0555] | 2.614e-03 | 11 | 0.021 | 2.614e-03 | 145 | 18 | 12.166 | 1 | 0.000 | yes |
| AP11 | AP | primary | 0.470 | 0.400 | [0.33, 0.61] | 0.400 | 0.580 | 7 | 0.260 | 11.220 | 13.465 | [8.997, 13.921] | 1.933 | 4.085e-06 | 14.624 | - | [-, -] | - | 11 | - | - | 0 | 0 | - | 0 | 0.000 | yes |
| AP13 | AP | primary | 0.470 | 0.410 | [0.09, 0.41] | 0.330 | 0.430 | 15 | 0.360 | 354.8 | 10.219 | [10.016, 11.868] | 1.279 | 5.382e-06 | 10.257 | 0.012 | [8.430e-03, 0.0163] | 4.471e-03 | 41 | 7.273e-03 | 7.562e-03 | 3,087 | 722 | 44.407 | 6 | 0.409 | no |
| AP14 | AP | primary | 0.470 | 0.350 | [0.14, 0.38] | 0.250 | 0.530 | 31 | 0.220 | 11.220 | 13.098 | [11.695, 14.496] | 4.295 | 4.199e-06 | 15.409 | 0.026 | [0.0201, 0.0344] | 2.095e-03 | 11 | 0.026 | 2.095e-03 | 118 | 21 | 12.000 | 1 | 0.000 | yes |
| AP18 | AP | primary | 0.470 | 0.390 | [0.00, 0.39] | 0.230 | 0.560 | 12 | 0.280 | 11.220 | 10.427 | [10.417, 12.378] | 1.040 | 5.275e-06 | 12.939 | 0.029 | [0.0197, 0.0313] | 1.885e-03 | 11 | 0.029 | 1.885e-03 | 25 | 5 | 12.000 | 1 | 0.000 | yes |
| AP19 | AP | primary | 0.470 | 0.350 | [0.21, 0.40] | 0.300 | 0.630 | 20 | 0.280 | 354.8 | 12.781 | [10.111, 14.268] | 2.763 | 4.303e-06 | 14.932 | 8.990e-03 | [8.034e-03, 9.947e-03] | 6.118e-03 | 41 | 6.983e-03 | 7.877e-03 | 6,160 | 1,744 | 50.478 | 6 | 0.223 | no |
| AP20 | AP | primary | 0.470 | 0.280 | [0.09, 0.35] | 0.130 | 0.510 | 38 | 0.230 | 223.9 | 11.542 | [10.393, 12.968] | 3.553 | 4.765e-06 | 14.200 | 0.010 | [9.176e-03, 0.0114] | 5.400e-03 | 25 | 7.635e-03 | 7.204e-03 | 6,710 | 1,629 | 28.284 | 4 | 0.250 | no |
| AP23 | AP | primary | 0.470 | 0.440 | [0.29, 0.49] | 0.400 | 0.530 | 4 | 0.400 | 281.8 | 10.589 | [9.807, 13.783] | 1.365 | 5.194e-06 | 10.082 | 0.015 | [0.0121, 0.0169] | 3.661e-03 | 41 | 0.012 | 4.505e-03 | 2,197 | 503 | 44.045 | 6 | 0.187 | no |
| AP24 | AP | primary | 0.470 | 0.360 | [0.00, 0.37] | 0.240 | 0.520 | 18 | 0.280 | 223.9 | 12.113 | [8.662, 12.593] | 1.718 | 4.540e-06 | 14.674 | 9.261e-03 | [7.581e-03, 0.0108] | 5.939e-03 | 25 | 6.673e-03 | 8.242e-03 | 5,040 | 1,129 | 25.456 | 4 | 0.279 | no |
| AP25 | AP | primary (Tier 2) | 0.470 | 0.430 | [0.28, 0.43] | 0.380 | 0.400 | 6 | 0.350 | 112.2 | 11.510 | [9.582, 11.738] | 1.384 | 4.779e-06 | 12.278 | 0.014 | [0.0104, 0.0198] | 3.859e-03 | 15 | 0.012 | 4.573e-03 | 1,124 | 168 | 19.698 | 2 | 0.156 | no |
| AP27 | AP | primary | 0.470 | 0.440 | [0.14, 0.44] | 0.370 | 0.590 | 20 | 0.300 | 11.220 | 11.521 | [9.513, 12.511] | 2.049 | 4.774e-06 | 11.771 | 0.042 | [0.0304, 0.0567] | 1.319e-03 | 11 | 0.042 | 1.319e-03 | 83 | 15 | 12.166 | 1 | 0.000 | yes |
| AP28 | AP | primary | 0.470 | 0.350 | [0.09, 0.35] | 0.190 | 0.570 | 28 | 0.210 | 11.220 | 12.546 | [8.977, 12.961] | 1.463 | 4.384e-06 | 14.085 | 0.020 | [0.0159, 0.0397] | 2.704e-03 | 11 | 0.020 | 2.704e-03 | 119 | 20 | 12.000 | 1 | 0.000 | yes |
| AP29 | AP | primary | 0.470 | 0.340 | [0.22, 0.39] | 0.320 | 0.480 | 13 | 0.300 | 223.9 | 10.749 | [10.264, 12.473] | 2.011 | 5.117e-06 | 12.062 | 8.735e-03 | [7.611e-03, 0.0102] | 6.297e-03 | 31 | 8.204e-03 | 6.704e-03 | 3,606 | 879 | 33.106 | 5 | 0.061 | yes |
| TTV1 | TTV | primary | 0.330 | 0.190 | [0.00, 0.22] | 0.010 | 0.500 | 23 | 0.120 | 35.481 | 6.850 | [5.918, 7.745] | 4.587 | 1.124e-05 | 7.982 | 0.027 | [0.0209, 0.0317] | 2.905e-03 | 11 | 0.027 | 2.905e-03 | 966 | 242 | 12.649 | 1 | 0.000 | yes |
| TTV2 | TTV | primary | 0.330 | 0.310 | [0.22, 0.32] | 0.180 | 0.400 | 16 | 0.210 | 56.234 | 4.871 | [4.750, 6.633] | 1.264 | 1.581e-05 | 4.757 | 0.017 | [0.0146, 0.0195] | 4.461e-03 | 15 | 0.016 | 4.708e-03 | 1,995 | 333 | 16.125 | 2 | 0.053 | yes |
| TTV3 | TTV | primary | 0.330 | 0.300 | [0.05, 0.48] | 0.280 | 0.400 | 7 | 0.260 | 17.783 | 2.921 | [2.857, 5.372] | 1.185 | 2.636e-05 | 3.086 | 0.024 | [0.0190, 0.0330] | 3.262e-03 | 11 | 0.024 | 3.262e-03 | 122 | 32 | 12.166 | 1 | 0.000 | yes |
| TTV4 | TTV | primary | 0.330 | 0.220 | [0.07, 0.30] | 0.150 | 0.240 | 24 | 0.170 | 11.220 | 4.772 | [3.149, 5.326] | 1.605 | 1.614e-05 | 5.108 | 0.088 | [0.0757, 0.0931] | 8.796e-04 | 11 | 0.088 | 8.796e-04 | 86 | 30 | 11.662 | 1 | 0.000 | yes |
| TTV5 | TTV | primary | 0.330 | 0.000 | [0.00, 0.26] | 0.080 | 0.000 | 22 | 0.160 | 44.668 | 3.538 | [2.582, 7.510] | 1.727 | 2.176e-05 | 3.839 | 0.031 | [0.0276, 0.0353] | 2.498e-03 | 15 | 0.029 | 2.699e-03 | 3,203 | 491 | 16.971 | 2 | 0.075 | yes |
| TTV6 | TTV | primary | 0.330 | 0.070 | [0.00, 0.08] | 0.050 | 0.140 | 43 | 0.080 | 22.387 | 6.190 | [4.974, 6.793] | 2.984 | 1.244e-05 | 8.303 | 0.039 | [0.0280, 0.0557] | 1.993e-03 | 11 | 0.039 | 1.993e-03 | 612 | 126 | 12.649 | 1 | 0.000 | yes |
| TTV7 | TTV | primary | 0.330 | 0.150 | [0.14, 0.17] | 0.070 | 0.290 | 20 | 0.120 | 56.234 | 7.136 | [5.451, 7.819] | 3.890 | 1.079e-05 | 8.426 | 0.025 | [0.0220, 0.0278] | 3.098e-03 | 15 | 0.023 | 3.314e-03 | 2,983 | 620 | 18.111 | 2 | 0.065 | yes |
| TTV8 | TTV | primary | 0.330 | 0.130 | [9.750e-03, 0.18] | 0.000 | 0.300 | 23 | 0.120 | 89.125 | 6.695 | [5.102, 7.397] | 3.522 | 1.150e-05 | 7.756 | 0.019 | [0.0177, 0.0219] | 3.984e-03 | 15 | 0.017 | 4.479e-03 | 5,287 | 950 | 18.974 | 2 | 0.110 | no |
| TTV9 | TTV | primary | 0.330 | 0.200 | [0.00, 0.37] | 0.190 | 0.250 | 13 | 0.190 | 177.8 | 7.311 | [4.702, 7.707] | 1.526 | 1.053e-05 | 8.425 | 0.018 | [0.0157, 0.0235] | 4.202e-03 | 25 | 0.012 | 6.252e-03 | 3,291 | 660 | 26.077 | 4 | 0.328 | no |
| TTV11 | TTV | primary (Tier 2) | 0.330 | 0.050 | [0.00, 0.12] | 0.000 | 0.000 | 28 | 0.110 | 44.668 | 6.642 | [5.646, 7.138] | 6.223 | 1.159e-05 | 8.038 | 0.033 | [0.0282, 0.0386] | 2.312e-03 | 15 | 0.030 | 2.558e-03 | 3,192 | 479 | 15.232 | 2 | 0.096 | yes |
| TTV15 | TTV | primary | 0.330 | 0.170 | [0.00, 0.32] | 0.060 | 0.030 | 26 | 0.200 | 17.783 | 5.613 | [5.024, 6.865] | 3.645 | 1.372e-05 | 5.574 | 0.051 | [0.0378, 0.0711] | 1.506e-03 | 11 | 0.051 | 1.506e-03 | 579 | 65 | 12.806 | 1 | 0.000 | yes |
| GM1 | GM | ncalm | 0.290 | 0.320 | [0.30, 0.39] | 0.280 | 0.380 | 19 | 0.220 | 35.481 | 2.411 | [2.289, 2.505] | 3.472 | 6.096e-05 | 2.320 | 0.016 | [0.0151, 0.0178] | 9.001e-03 | 15 | 0.016 | 9.354e-03 | 3,583 | 507 | 15.232 | 2 | 0.038 | yes |
| GM2 | GM | ncalm | 0.290 | 0.310 | [0.26, 0.38] | 0.260 | 0.310 | 16 | 0.240 | 35.481 | 2.134 | [1.958, 2.436] | 3.618 | 6.888e-05 | 1.923 | 0.015 | [0.0126, 0.0171] | 9.770e-03 | 15 | 0.014 | 0.010 | 2,530 | 295 | 16.000 | 2 | 0.038 | yes |
| GM3 | GM | ncalm | 0.290 | 0.350 | [0.28, 0.41] | 0.290 | 0.420 | 15 | 0.250 | 22.387 | 1.842 | [1.742, 2.215] | 2.646 | 7.981e-05 | 1.757 | 0.015 | [0.0130, 0.0174] | 0.010 | 11 | 0.015 | 0.010 | 1,500 | 190 | 12.649 | 1 | 0.000 | yes |
| GM4 | GM | ncalm | 0.290 | 0.350 | [0.26, 0.38] | 0.340 | 0.310 | 14 | 0.260 | 56.234 | 3.322 | [3.258, 4.316] | 1.418 | 4.425e-05 | 3.084 | 0.015 | [0.0116, 0.0178] | 9.855e-03 | 15 | 0.016 | 9.435e-03 | 3,347 | 373 | 19.698 | 2 | 0.045 | yes |
| GM5 | GM | ncalm | 0.290 | 0.320 | [0.00, 0.32] | 0.310 | 0.300 | 4 | 0.250 | 28.184 | 2.301 | [2.275, 2.802] | 1.205 | 6.389e-05 | 2.209 | 0.019 | [0.0181, 0.0203] | 7.589e-03 | 15 | 0.020 | 7.496e-03 | 1,128 | 144 | 15.232 | 2 | 0.012 | yes |
| GM6 | GM | ncalm | 0.290 | 0.350 | [0.19, 0.39] | 0.250 | 0.350 | 26 | 0.240 | 44.668 | 2.124 | [1.994, 3.404] | 1.951 | 6.920e-05 | 1.906 | 0.013 | [0.0110, 0.0158] | 0.011 | 15 | 0.014 | 0.010 | 1,987 | 391 | 15.232 | 2 | 0.061 | yes |
| GM7 | GM | ncalm | 0.290 | 0.320 | [0.20, 0.39] | 0.260 | 0.420 | 7 | 0.250 | 35.481 | 1.832 | [1.766, 2.080] | 2.270 | 8.025e-05 | 1.716 | 6.672e-03 | [6.070e-03, 7.560e-03] | 0.022 | 11 | 6.672e-03 | 0.022 | 920 | 146 | 12.806 | 1 | 0.000 | yes |
| AP1_3dep | AP | tier1_continuity | 0.470 | 0.400 | [0.00, 0.40] | 0.130 | 0.560 | 16 | 0.340 | 223.9 | 12.270 | [11.285, 14.092] | 1.142 | 4.482e-06 | 13.857 | 9.987e-03 | [8.197e-03, 0.0121] | 5.507e-03 | 31 | 8.253e-03 | 6.664e-03 | 4,575 | 1,199 | 33.941 | 5 | 0.174 | no |
| AP1 | AP | pamap | 0.470 | 0.390 | [0.00, 0.39] | 0.020 | 0.580 | 32 | 0.350 | 281.8 | 12.421 | [12.048, 13.739] | 1.457 | 4.428e-06 | 14.336 | 0.015 | [0.0131, 0.0171] | 3.587e-03 | 31 | 7.882e-03 | 6.978e-03 | 3,446 | 505 | 35.855 | 5 | 0.486 | no |
| AP2 | AP | pamap | 0.470 | 0.410 | [0.24, 0.43] | 0.390 | 0.570 | 13 | 0.310 | 56.234 | 11.584 | [8.233, 12.213] | 1.860 | 4.748e-06 | 12.807 | 0.017 | [0.0148, 0.0192] | 3.218e-03 | 15 | 0.013 | 4.078e-03 | 944 | 122 | 16.549 | 2 | 0.211 | no |
| AP3 | AP | pamap | 0.470 | 0.380 | [0.11, 0.38] | 0.310 | 0.450 | 32 | 0.300 | 17.783 | 11.530 | [8.722, 11.583] | 1.077 | 4.770e-06 | 13.068 | 0.026 | [0.0207, 0.0317] | 2.155e-03 | 11 | 0.026 | 2.155e-03 | 533 | 61 | 12.488 | 1 | 0.000 | yes |
| AP4 | AP | pamap | 0.470 | 0.360 | [0.33, 0.46] | 0.310 | 0.590 | 30 | 0.330 | 446.7 | 11.932 | [7.644, 12.709] | 2.299 | 4.610e-06 | 12.212 | 0.015 | [0.0133, 0.0173] | 3.629e-03 | 41 | 7.124e-03 | 7.720e-03 | 4,711 | 579 | 43.612 | 6 | 0.530 | no |
| AP5 | AP | pamap | 0.470 | 0.390 | [0.17, 0.40] | 0.360 | 0.500 | 9 | 0.320 | 56.234 | 10.726 | [8.767, 12.589] | 1.584 | 5.128e-06 | 12.070 | 0.019 | [0.0151, 0.0220] | 2.921e-03 | 15 | 0.016 | 3.456e-03 | 3,575 | 346 | 19.209 | 2 | 0.155 | no |

### B7. Site summaries per group: pooled theta, site-pooled k_s and K0, D0, the window-stability test (rule 6.1)

| site | group | n | m_site | theta site-pooled (1e3) | 95 % | theta equal-weight (1e3) | 95 % | basins | theta pooled (1e4) | 95 % | theta equal (1e4) | basins (1e4) | k_s site-pooled (1e3) | 95 % | n_eff | K0 site-pooled | K0 median of tiles | K0 range | D0_A median | D0_A range | D0_B median | window test pass | stands (>= 80 %) | D0 status | tiles with only L_f in range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM | primary | 19 | 0.290 | 0.320 | [0.30, 0.33] | 0.270 | [0.24, 0.30] | 224 | 0.310 | [0.28, 0.33] | 0.300 | 103 | 2.051 | [1.908, 2.183] | 33.895 | 7.167e-05 | 7.158e-05 | 4.540e-05-9.138e-05 | 0.015 | 7.280e-03-0.035 | 0.014 | 11 / 19 | no | bracket [D0_A, D0_B] | - |
| GM | clean | 16 | 0.290 | 0.320 | [0.30, 0.34] | 0.280 | [0.25, 0.30] | 181 | 0.310 | [0.29, 0.34] | 0.300 | 85 | 2.096 | [1.944, 2.238] | 29.542 | 7.014e-05 | 7.107e-05 | 4.540e-05-8.482e-05 | 0.015 | 7.280e-03-0.035 | 0.015 | 9 / 16 | no | bracket [D0_A, D0_B] | - |
| GM | ncalm | 7 | 0.290 | 0.330 | [0.31, 0.35] | 0.290 | [0.24, 0.32] | 101 | 0.320 | [0.30, 0.34] | 0.300 | 40 | 2.215 | [2.000, 2.430] | 11.204 | 6.637e-05 | 6.888e-05 | 4.425e-05-8.025e-05 | 9.855e-03 | 7.589e-03-0.022 | 0.010 | 7 / 7 | yes | measurement (D0_A) | GM3, GM7 |
| AP | primary | 21 | 0.470 | 0.390 | [0.37, 0.40] | 0.300 | [0.27, 0.32] | 403 | 0.500 | [0.46, 0.54] | 0.510 | 81 | 11.481 | [11.091, 11.904] | 34.482 | 4.791e-06 | 4.779e-06 | 4.085e-06-5.463e-06 | 4.421e-03 | 1.319e-03-6.297e-03 | 5.522e-03 | 9 / 21 | no | bracket [D0_A, D0_B] | AP2_3dep, AP9, AP10, AP11, AP14, AP18, AP27, AP28 |
| AP | clean | 20 | 0.470 | 0.380 | [0.37, 0.40] | 0.300 | [0.27, 0.32] | 397 | 0.510 | [0.47, 0.55] | 0.510 | 78 | 11.479 | [11.030, 11.880] | 33.114 | 4.791e-06 | 4.792e-06 | 4.085e-06-5.463e-06 | 4.471e-03 | 1.319e-03-6.297e-03 | 5.771e-03 | 9 / 20 | no | bracket [D0_A, D0_B] | AP2_3dep, AP9, AP10, AP11, AP14, AP18, AP27, AP28 |
| AP | pamap | 5 | 0.470 | 0.390 | [0.37, 0.41] | 0.290 | [0.21, 0.34] | 116 | 0.520 | [0.47, 0.57] | 0.530 | 16 | 11.602 | [10.957, 12.214] | 7.113 | 4.741e-06 | 4.748e-06 | 4.428e-06-5.128e-06 | 3.218e-03 | 2.155e-03-3.629e-03 | 4.078e-03 | 1 / 5 | no | bracket [D0_A, D0_B] | AP3 |
| TTV | primary | 11 | 0.330 | 0.190 | [0.15, 0.22] | 0.080 | [0.04, 0.12] | 245 | 0.260 | [0.16, 0.32] | 0.170 | 100 | 5.388 | [4.674, 6.107] | 17.848 | 1.429e-05 | 1.244e-05 | 1.053e-05-2.636e-05 | 2.905e-03 | 8.796e-04-4.461e-03 | 2.905e-03 | 9 / 11 | yes | measurement (D0_A) | TTV1, TTV3, TTV4, TTV6, TTV15 |
| TTV | clean | 10 | 0.330 | 0.190 | [0.16, 0.23] | 0.090 | [0.04, 0.13] | 217 | 0.270 | [0.23, 0.36] | 0.190 | 86 | 5.302 | [4.725, 6.337] | 15.519 | 1.452e-05 | 1.308e-05 | 1.053e-05-2.636e-05 | 3.002e-03 | 8.796e-04-4.461e-03 | 3.084e-03 | 8 / 10 | yes | measurement (D0_A) | TTV1, TTV3, TTV4, TTV6, TTV15 |

### B8. Survey comparisons: AP PAMAP against 3DEP (rule 6.3) and GM NCALM against FEMA (rule 6.4, report only)

| tile | k_s PAMAP | k_s 3DEP | ratio 3DEP/PAMAP | C_ht PAMAP | C_ht 3DEP | ratio | theta PAMAP | theta 3DEP | ratio | 3DEP tile excluded (Tier 1) |
|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | 12.421 | 12.270 | 0.988 | 0.015 | 9.987e-03 | 0.651 | 0.390 | 0.400 | 1.026 | yes |
| AP2 | 11.584 | 10.898 | 0.941 | 0.017 | 0.020 | 1.174 | 0.410 | 0.390 | 0.951 | no |
| AP3 | 11.530 | 11.035 | 0.957 | 0.026 | 0.013 | 0.493 | 0.380 | 0.370 | 0.974 | no |
| AP4 | 11.932 | 11.432 | 0.958 | 0.015 | 0.010 | 0.677 | 0.360 | 0.360 | 1.000 | no |
| AP5 | 10.726 | 10.380 | 0.968 | 0.019 | 0.011 | 0.568 | 0.390 | 0.390 | 1.000 | no |

AP medians: k_s ratio 0.958, C_ht ratio 0.651, theta ratio 1.000.

| tile | k_s NCALM | k_s FEMA | ratio FEMA/NCALM | C_ht NCALM | C_ht FEMA | ratio | theta NCALM | theta FEMA |
|---|---|---|---|---|---|---|---|---|
| GM1 | 2.411 | 2.384 | 0.989 | 0.016 | 0.014 | 0.878 | 0.320 | 0.320 |
| GM2 | 2.134 | 2.020 | 0.947 | 0.015 | 0.012 | 0.781 | 0.310 | 0.310 |
| GM3 | 1.842 | 1.827 | 0.992 | 0.015 | 9.512e-03 | 0.655 | 0.350 | 0.320 |
| GM4 | 3.322 | 3.238 | 0.975 | 0.015 | 0.012 | 0.836 | 0.350 | 0.330 |
| GM5 | 2.301 | 2.265 | 0.985 | 0.019 | 0.017 | 0.870 | 0.320 | 0.300 |
| GM6 | 2.124 | 2.147 | 1.011 | 0.013 | 9.044e-03 | 0.673 | 0.350 | 0.340 |
| GM7 | 1.832 | 1.781 | 0.972 | 6.672e-03 | 4.148e-03 | 0.622 | 0.320 | 0.310 |

GM: median k_s ratio 0.985 (within +-15 %: yes; largest single deviation 5.3 %, all within +-30 %: yes); median |C_ht| ratio 0.781 (within +-15 %: no; largest 37.8 %, all within +-30 %: no); **criteria of rule 6.4 met: no** (report only (go-ahead 2.3): the primary GM set is a single survey; no pooling decision follows).

### B9. Run ids and input checksums

| tile | m-profile run | measure run | sha256 filled (16) | sha256 area (16) | sha256 fd (16) |
|---|---|---|---|---|---|
| GM1_fema | 20261119T014525Z_tile_run_GM1_fema_v2_m_profile | 20261119T012857Z_measure_a17b_t0 | 0f6863c8bb7eb7f1... | baaa86ff7a7bb822... | 1dbe84d3f1de5088... |
| GM2_fema | 20261119T014531Z_tile_run_GM2_fema_v2_m_profile | 20261119T012858Z_measure_a17b_t1 | c1a153d829f42775... | eb4c4453bf90cb6c... | 7c5e51cbd6d8ef54... |
| GM3_fema | 20261119T014533Z_tile_run_GM3_fema_v2_m_profile | 20261119T012858Z_measure_a17b_t2 | 8242ccae21b8717a... | b1f8cc664b0862c9... | d05c17a3f29a3877... |
| GM4_fema | 20261119T014533Z_tile_run_GM4_fema_v2_m_profile | 20261119T012858Z_measure_a17b_t3 | da15f0a51b301800... | 53d1a6006e7ee292... | e423f8b2b4ef3e71... |
| GM5_fema | 20261119T014533Z_tile_run_GM5_fema_v2_m_profile | 20261119T012858Z_measure_a17b_t4 | 2ede1f73fd73e058... | a623da22933fe6ce... | 829d7cbbe1455a77... |
| GM6_fema | 20261119T014538Z_tile_run_GM6_fema_v2_m_profile | 20261119T012858Z_measure_a17b_t5 | 58a14ecdff487985... | 1d259ef33ac56105... | 55ac00f94bb41b6f... |
| GM7_fema | 20261119T014538Z_tile_run_GM7_fema_v2_m_profile | 20261119T012859Z_measure_a17b_t6 | 8840ff7646ff246e... | 83a7c2599a8af7eb... | 2d396821abe48528... |
| GM8 | 20261119T014531Z_tile_run_GM8_v2_m_profile | 20261119T012857Z_measure_a17b_t7 | 7d794d455650fd50... | 050b7845ccea409f... | f676337ee72beed0... |
| GM9 | 20261119T014531Z_tile_run_GM9_v2_m_profile | 20261119T012857Z_measure_a17b_t0 | cfbe4e9b9d0c947a... | ad915fac8a95e775... | 976b6e0cf54c96e3... |
| GM11 | 20261119T014531Z_tile_run_GM11_v2_m_profile | 20261119T012858Z_measure_a17b_t1 | 6b4968e46106d996... | 310bc06e362b4a6d... | e73c25f3411dc3c8... |
| GM12 | 20261119T014531Z_tile_run_GM12_v2_m_profile | 20261119T012858Z_measure_a17b_t2 | 8dda4ea297200fcf... | d088f40a8b29ecd5... | 06e9033a07425ebf... |
| GM13 | 20261119T014535Z_tile_run_GM13_v2_m_profile | 20261119T012858Z_measure_a17b_t3 | 4ec923366ae9d3d8... | 4ba190e17720062a... | 7386f412426ba45e... |
| GM14 | 20261119T014531Z_tile_run_GM14_v2_m_profile | 20261119T012858Z_measure_a17b_t4 | f955bee89ddf0e25... | 4e90c974aa4732cf... | e1befa80a47a89cb... |
| GM15 | 20261119T014531Z_tile_run_GM15_v2_m_profile | 20261119T012858Z_measure_a17b_t5 | 464298a281ca916a... | b40d6f3259862856... | f8eae4a97d9259c5... |
| GM16 | 20261119T014536Z_tile_run_GM16_v2_m_profile | 20261119T012859Z_measure_a17b_t6 | 2430fd9e3b21b96e... | 439c97db410163da... | ec82c87221ba27ba... |
| GM17 | 20261119T014531Z_tile_run_GM17_v2_m_profile | 20261119T012857Z_measure_a17b_t7 | 3adef0174cb61866... | 103b8c68ae0dee9f... | fb68ad3241c925ea... |
| GM18 | 20261119T014531Z_tile_run_GM18_v2_m_profile | 20261119T012857Z_measure_a17b_t0 | 027fe19a6edddbe9... | 337449476d7c8370... | bc9e60ff5527ff32... |
| GM19 | 20261119T014531Z_tile_run_GM19_v2_m_profile | 20261119T012858Z_measure_a17b_t1 | 3e9ea5bd8500d152... | 685d1f77c155245c... | 6e24840ae16d97ac... |
| GM20 | 20261119T014531Z_tile_run_GM20_v2_m_profile | 20261119T012858Z_measure_a17b_t2 | f6b6c1932b4a961b... | a30e0ed1d4bc2b81... | 1ad4f6d4c89d79fb... |
| GM21 | 20261119T014531Z_tile_run_GM21_v2_m_profile | 20261119T012858Z_measure_a17b_t3 | f8abbd7cb10fac82... | 3c02debb4b21d698... | 692cd82e91f18cbd... |
| GM22 | 20261119T014531Z_tile_run_GM22_v2_m_profile | 20261119T012858Z_measure_a17b_t4 | 4c779bf3b0f76c4b... | 37c05a8cdfbf54bd... | 8cfbc4094c79b584... |
| GM24 | 20261119T014532Z_tile_run_GM24_v2_m_profile | 20261119T012858Z_measure_a17b_t5 | d41acb705ca5ae59... | 3ab8d08a14103ac0... | 354e0ca983caeea5... |
| GM25 | 20261119T014531Z_tile_run_GM25_v2_m_profile | 20261119T012859Z_measure_a17b_t6 | d9e873cd823bd387... | 34657921d80ca4a3... | 648dd05be7bfa22a... |
| GM26 | 20261119T014531Z_tile_run_GM26_v2_m_profile | 20261119T012857Z_measure_a17b_t7 | 273647897cb523a2... | da4fb32eb8a5fa14... | 571c5911c54cd94e... |
| GM27 | 20261119T014531Z_tile_run_GM27_v2_m_profile | 20261119T012857Z_measure_a17b_t0 | fcefa2d86178b732... | 64063c296e7fbb12... | f118c5d643d1c5aa... |
| GM28 | 20261119T014531Z_tile_run_GM28_v2_m_profile | 20261119T012858Z_measure_a17b_t1 | ab3a00807d4bb49c... | 11d3ad33add80c49... | 4632a97fd0af9672... |
| GM29 | 20261119T014531Z_tile_run_GM29_v2_m_profile | 20261119T012858Z_measure_a17b_t2 | 425b0e13e87497a7... | e3c4725bfbb45612... | feff2c8d8f385a8f... |
| GM30 | 20261119T014535Z_tile_run_GM30_v2_m_profile | 20261119T012858Z_measure_a17b_t3 | c53d59a3069aa029... | 2fb5e462056a13a4... | 2d28b22c3b71c8ef... |
| AP2_3dep | 20261119T014531Z_tile_run_AP2_3dep_v2_m_profile | 20261119T012858Z_measure_a17b_t4 | bcf3953d1f301f48... | 604036331e605d82... | b83980142850928e... |
| AP3_3dep | 20261119T014535Z_tile_run_AP3_3dep_v2_m_profile | 20261119T012858Z_measure_a17b_t5 | dabb919ac1a5d563... | d7384a58cd99d48f... | 33f881b4c15dad87... |
| AP4_3dep | 20261119T014535Z_tile_run_AP4_3dep_v2_m_profile | 20261119T012859Z_measure_a17b_t6 | 7d73f03bf054fc35... | fd22efdbc0a5bb14... | 46d591984b4fa68d... |
| AP5_3dep | 20261119T014535Z_tile_run_AP5_3dep_v2_m_profile | 20261119T012857Z_measure_a17b_t7 | dfe858efe75825ed... | ad795f4c05de586c... | 2fc84eae8eaefd48... |
| AP6 | 20261119T014534Z_tile_run_AP6_v2_m_profile | 20261119T012857Z_measure_a17b_t0 | f6654d364e8ef1f6... | 7d498640366bed4c... | db1879d57298539c... |
| AP7 | 20261119T014534Z_tile_run_AP7_v2_m_profile | 20261119T012858Z_measure_a17b_t1 | e4a6801d3b5a6f22... | 654970d4da7e5d95... | 907a0da1b497e0eb... |
| AP8 | 20261119T014533Z_tile_run_AP8_v2_m_profile | 20261119T012858Z_measure_a17b_t2 | fcfc7b3589d86988... | 2113996b06ad5088... | fe96cd884c8a9ce7... |
| AP9 | 20261119T014535Z_tile_run_AP9_v2_m_profile | 20261119T012858Z_measure_a17b_t3 | 6eff4e2e80fa7066... | b66f8490498a9ad2... | 284b7a19427d8f4b... |
| AP10 | 20261119T014537Z_tile_run_AP10_v2_m_profile | 20261119T012858Z_measure_a17b_t4 | 558da54fda8404d8... | 4f641af2fa6977db... | ab5ebe8ab7ac9353... |
| AP11 | 20261119T014533Z_tile_run_AP11_v2_m_profile | 20261119T012858Z_measure_a17b_t5 | c5afa265dba886bb... | 22ab93cda170307b... | 9238ea447d2ab1ce... |
| AP13 | 20261119T014533Z_tile_run_AP13_v2_m_profile | 20261119T012859Z_measure_a17b_t6 | d22dc7b768ded4d8... | 2c422156ee4c47af... | aa0edc66997fd3ec... |
| AP14 | 20261119T014533Z_tile_run_AP14_v2_m_profile | 20261119T012857Z_measure_a17b_t7 | f85ef07c45c79063... | 7d9df7907da183d6... | 754ef5bd86be9162... |
| AP18 | 20261119T025158Z_tile_run_AP18_v2_m_profile | 20261119T012857Z_measure_a17b_t0 | 34a5995142121c48... | c8961e2f6d8ea65b... | 5f3513df02810797... |
| AP19 | 20261119T025158Z_tile_run_AP19_v2_m_profile | 20261119T012858Z_measure_a17b_t1 | 567efb8e8f9e589a... | 53f428d2b5039f3f... | b25f95a8ce08d8fa... |
| AP20 | 20261119T025158Z_tile_run_AP20_v2_m_profile | 20261119T012858Z_measure_a17b_t2 | 35aa681857e69056... | 7f58dd69305d389c... | 5dc5967e4a253f2c... |
| AP23 | 20261119T025159Z_tile_run_AP23_v2_m_profile | 20261119T012858Z_measure_a17b_t3 | 5c904a40bb73aafc... | 4657ae049b9d8c74... | f2098b2d30781e1c... |
| AP24 | 20261119T025147Z_tile_run_AP24_v2_m_profile | 20261119T012858Z_measure_a17b_t4 | 83d2b2329f17566a... | 777b95d898893f54... | e5e6ae1ecddb825e... |
| AP25 | 20261119T025158Z_tile_run_AP25_v2_m_profile | 20261119T012858Z_measure_a17b_t5 | 4b4d13298ad6735d... | 22d52a6e4261f17b... | ca7b591ebb8793d2... |
| AP27 | 20261119T025159Z_tile_run_AP27_v2_m_profile | 20261119T012859Z_measure_a17b_t6 | 3bc0159b5ea644e4... | 34c492ce8d639fe8... | f79712a95add161e... |
| AP28 | 20261119T025157Z_tile_run_AP28_v2_m_profile | 20261119T012857Z_measure_a17b_t7 | dcaac0ed4f55a638... | 52cd231b2b1e07a3... | b80d088e8a9fa3d7... |
| AP29 | 20261119T025158Z_tile_run_AP29_v2_m_profile | 20261119T012857Z_measure_a17b_t0 | fc55f9e15b9c8eaf... | 77a052d647e42deb... | cbcb87696b5c9524... |
| TTV1 | 20261119T025516Z_tile_run_TTV1_v2_m_profile | 20261119T012858Z_measure_a17b_t1 | 6f7665fba53f9f86... | c8ebf9be6e486106... | c134203cbe6abbaf... |
| TTV2 | 20261119T025516Z_tile_run_TTV2_v2_m_profile | 20261119T012858Z_measure_a17b_t2 | 95737de40f09260e... | caa75970e66c0995... | c908af16706e1b51... |
| TTV3 | 20261119T025516Z_tile_run_TTV3_v2_m_profile | 20261119T012858Z_measure_a17b_t3 | 53fdd1b7108a289f... | 3fcaf91c61ca040b... | 03cd522064c8381c... |
| TTV4 | 20261119T025516Z_tile_run_TTV4_v2_m_profile | 20261119T012858Z_measure_a17b_t4 | 1c39b85216225656... | 574a94e13dd8102e... | 7a1649ea3ba3ccaa... |
| TTV5 | 20261119T025516Z_tile_run_TTV5_v2_m_profile | 20261119T012858Z_measure_a17b_t5 | d4600f9ed4b192a0... | 1023d8462ccb2b23... | 119ba7f75f0b098a... |
| TTV6 | 20261119T025516Z_tile_run_TTV6_v2_m_profile | 20261119T012859Z_measure_a17b_t6 | ad15a4d19905e3a3... | fff9a8e76627cfe7... | 8fc59cdb1d49fbe2... |
| TTV7 | 20261119T025516Z_tile_run_TTV7_v2_m_profile | 20261119T012857Z_measure_a17b_t7 | 9b42aee8e542df9a... | 33dba98b212e150a... | 0aec56039561497e... |
| TTV8 | 20261119T025516Z_tile_run_TTV8_v2_m_profile | 20261119T012857Z_measure_a17b_t0 | fecb3b7c344a6ebb... | 6a7590468a4dfb7d... | 279b626e7040e7c3... |
| TTV9 | 20261119T025516Z_tile_run_TTV9_v2_m_profile | 20261119T012858Z_measure_a17b_t1 | 2743718c0b789b23... | 002f077d0e8c0f25... | 1c1963c28e1d30e0... |
| TTV11 | 20261119T025516Z_tile_run_TTV11_v2_m_profile | 20261119T012858Z_measure_a17b_t2 | d9811cf45716ca06... | e0977f9216393a3a... | 6de2c764e3fbf190... |
| TTV15 | 20261119T025516Z_tile_run_TTV15_v2_m_profile | 20261119T012858Z_measure_a17b_t3 | f76f94a8de7794f9... | 36355c80e1983baa... | ffea456c0a8d953e... |
| GM1 | 20261110T160404Z_tile_run_GM1_v2_m_profile | 20261119T012858Z_measure_a17b_t4 | 52daf99982213446... | 6d6e692e1ec679f0... | 5f981034af33bb29... |
| GM2 | 20261110T160420Z_tile_run_GM2_v2_m_profile | 20261119T012858Z_measure_a17b_t5 | ee21b49c5b5d2198... | 069254ccda78073a... | ce124e764c8abc3e... |
| GM3 | 20261110T160420Z_tile_run_GM3_v2_m_profile | 20261119T012859Z_measure_a17b_t6 | 634163f178ade794... | 8cbe261a48e07cbf... | 742b16f74a6f0b61... |
| GM4 | 20261110T160413Z_tile_run_GM4_v2_m_profile | 20261119T012857Z_measure_a17b_t7 | d1ec3c2e2e16c990... | 8cc34aa7c2b0bed5... | d739276a8fe02930... |
| GM5 | 20261110T160416Z_tile_run_GM5_v2_m_profile | 20261119T012857Z_measure_a17b_t0 | be088d59e1c97d73... | 52b606b4a3087242... | 459b5d369d84b7d0... |
| GM6 | 20261119T025516Z_tile_run_GM6_v2_m_profile | 20261119T012858Z_measure_a17b_t1 | 0eb04554d52a91c6... | 3bfdb1491052ea72... | 96809f424ccab28e... |
| GM7 | 20261119T025516Z_tile_run_GM7_v2_m_profile | 20261119T012858Z_measure_a17b_t2 | 2d0a4819bda89bdf... | 41a5869f527dac01... | 0cfa7d3f49c18253... |
| AP1_3dep | 20261119T025516Z_tile_run_AP1_3dep_v2_m_profile | 20261119T012858Z_measure_a17b_t3 | c7046f4116917126... | 96cee58f6d30e0f6... | f438c5b33f6db2c4... |
| AP1 | 20261110T160413Z_tile_run_AP1_v2_m_profile | 20261119T012858Z_measure_a17b_t4 | d80fa36e127042f4... | a20793e7e899b2c0... | 6684a19dc4e96eca... |
| AP2 | 20261110T160413Z_tile_run_AP2_v2_m_profile | 20261119T012858Z_measure_a17b_t5 | dbe47c64c8d6e4c9... | 96354947cb9351cf... | 92d2f77ade0c95da... |
| AP3 | 20261110T160405Z_tile_run_AP3_v2_m_profile | 20261119T012859Z_measure_a17b_t6 | f678027286ec7144... | 63559ad63e1783f2... | dcffdf9b6e3a62ec... |
| AP4 | 20261110T160405Z_tile_run_AP4_v2_m_profile | 20261119T012857Z_measure_a17b_t7 | cfbd3462b79a497d... | 9556c455050db6b1... | 0ba12accf8d8179c... |
| AP5 | 20261110T160405Z_tile_run_AP5_v2_m_profile | 20261119T012857Z_measure_a17b_t0 | 662d1198f2b82537... | b1ac3e21985089f4... | 8822db8596f98cf8... |

E (m/yr): {'GM': 0.000147, 'AP': 5.5e-05, 'TTV': 7.7e-05}; v2 site m: {'GM': 0.31, 'AP': 0.47}; settings: {"n_boot": 200, "A_c_ks": 1000.0, "A_c_theta": [1000.0, 10000.0], "L_f_m": 11.0, "w_f": 5, "window_pass_frac": 0.8, "E": {"GM": 0.000147, "AP": 5.5e-05, "TTV": 7.7e-05}, "v2_site_m": {"GM": 0.31, "AP": 0.47}}; theta grid 0.00-0.90 step 0.01.

### B10. Slurm accounting of the Part B jobs

| JobID | JobName | Partition | State | Elapsed | Timelimit | AllocCPUS | MaxRSS | TotalCPU |
|---|---|---|---|---|---|---|---|---|
| 52874596 | a17b_configs | serc | COMPLETED | 00:00:24 | 00:30:00 | 1 |  | 00:00:00 |
| 52874604 | a17b_bedrock | serc | COMPLETED | 00:00:20 | 00:30:00 | 1 |  | 00:00:00 |
| 52874660_0 | a17b_measure | serc | FAILED | 00:00:22 | 00:40:00 | 2 |  | 00:00:00 |
| 52874660_1 | a17b_measure | serc | FAILED | 00:00:22 | 00:40:00 | 2 |  | 00:00:00 |
| 52874660_2 | a17b_measure | serc | FAILED | 00:00:22 | 00:40:00 | 2 |  | 00:00:00 |
| 52874660_3 | a17b_measure | serc | FAILED | 00:00:22 | 00:40:00 | 2 |  | 00:00:00 |
| 52874660_4 | a17b_measure | serc | FAILED | 00:00:22 | 00:40:00 | 2 |  | 00:00:00 |
| 52874660_5 | a17b_measure | serc | FAILED | 00:00:19 | 00:40:00 | 2 |  | 00:00:00 |
| 52874660_6 | a17b_measure | serc | FAILED | 00:00:19 | 00:40:00 | 2 |  | 00:00:00 |
| 52874660_7 | a17b_measure | serc | FAILED | 00:00:19 | 00:40:00 | 2 |  | 00:00:00 |
| 52874664 | a17b_gapfill | serc | CANCELLED | 00:00:00 | 01:30:00 | 0 |  | 00:00:00 |
| 52877061_0 | a17b_measure | serc | COMPLETED | 00:03:06 | 00:40:00 | 2 |  | 00:00:00 |
| 52877061_1 | a17b_measure | serc | COMPLETED | 00:02:19 | 00:40:00 | 2 |  | 00:00:00 |
| 52877061_2 | a17b_measure | serc | COMPLETED | 00:02:52 | 00:40:00 | 2 |  | 00:00:00 |
| 52877061_3 | a17b_measure | serc | COMPLETED | 00:02:52 | 00:40:00 | 2 |  | 00:00:00 |
| 52877061_4 | a17b_measure | serc | COMPLETED | 00:03:03 | 00:40:00 | 2 |  | 00:00:00 |
| 52877061_5 | a17b_measure | serc | COMPLETED | 00:02:52 | 00:40:00 | 2 |  | 00:00:00 |
| 52877061_6 | a17b_measure | serc | COMPLETED | 00:02:55 | 00:40:00 | 2 |  | 00:00:00 |
| 52877061_7 | a17b_measure | serc | COMPLETED | 00:02:52 | 00:40:00 | 2 |  | 00:00:00 |
| 52877071 | a17b_gapfill | serc | COMPLETED | 00:00:27 | 01:30:00 | 1 |  | 00:00:00 |
| 52880640 | a17b_gapfill | serc | COMPLETED | 00:02:12 | 01:30:00 | 1 |  | 00:00:00 |
| 52880639_0 | a17b_measure | serc | COMPLETED | 00:02:58 | 00:40:00 | 2 |  | 00:00:00 |
| 52880639_1 | a17b_measure | serc | COMPLETED | 00:02:38 | 00:40:00 | 2 |  | 00:00:00 |
| 52880639_2 | a17b_measure | serc | COMPLETED | 00:02:38 | 00:40:00 | 2 |  | 00:00:00 |
| 52880639_3 | a17b_measure | serc | COMPLETED | 00:02:48 | 00:40:00 | 2 |  | 00:00:00 |
| 52880639_4 | a17b_measure | serc | COMPLETED | 00:03:11 | 00:40:00 | 2 |  | 00:00:00 |
| 52880639_5 | a17b_measure | serc | COMPLETED | 00:02:58 | 00:40:00 | 2 |  | 00:00:00 |
| 52880639_6 | a17b_measure | serc | COMPLETED | 00:02:56 | 00:40:00 | 2 |  | 00:00:00 |
| 52880639_7 | a17b_measure | serc | COMPLETED | 00:02:38 | 00:40:00 | 2 |  | 00:00:00 |
| 52881438_0 | a17b_mprofile | serc | COMPLETED | 01:09:14 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_1 | a17b_mprofile | serc | COMPLETED | 01:09:12 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_2 | a17b_mprofile | serc | COMPLETED | 01:09:19 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_3 | a17b_mprofile | serc | COMPLETED | 01:10:45 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_4 | a17b_mprofile | serc | COMPLETED | 01:08:44 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_5 | a17b_mprofile | serc | COMPLETED | 01:09:12 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_6 | a17b_mprofile | serc | COMPLETED | 01:11:05 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_7 | a17b_mprofile | serc | COMPLETED | 01:10:11 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_8 | a17b_mprofile | serc | COMPLETED | 01:10:53 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_9 | a17b_mprofile | serc | COMPLETED | 01:10:33 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_10 | a17b_mprofile | serc | COMPLETED | 01:12:25 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_11 | a17b_mprofile | serc | COMPLETED | 01:08:39 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_12 | a17b_mprofile | serc | COMPLETED | 01:04:41 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_13 | a17b_mprofile | serc | COMPLETED | 01:03:39 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_14 | a17b_mprofile | serc | COMPLETED | 01:09:26 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_15 | a17b_mprofile | serc | COMPLETED | 01:06:55 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_16 | a17b_mprofile | serc | COMPLETED | 01:03:46 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_17 | a17b_mprofile | serc | COMPLETED | 01:09:04 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_18 | a17b_mprofile | serc | COMPLETED | 01:07:33 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_19 | a17b_mprofile | serc | COMPLETED | 01:11:58 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_20 | a17b_mprofile | serc | COMPLETED | 01:04:30 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_21 | a17b_mprofile | serc | COMPLETED | 01:09:09 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_22 | a17b_mprofile | serc | COMPLETED | 01:06:10 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_23 | a17b_mprofile | serc | COMPLETED | 01:08:23 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_24 | a17b_mprofile | serc | COMPLETED | 01:06:51 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_25 | a17b_mprofile | serc | COMPLETED | 01:08:44 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_26 | a17b_mprofile | serc | COMPLETED | 01:12:30 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_27 | a17b_mprofile | serc | COMPLETED | 01:09:48 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_28 | a17b_mprofile | serc | COMPLETED | 01:09:24 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_29 | a17b_mprofile | serc | COMPLETED | 01:03:49 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_30 | a17b_mprofile | serc | COMPLETED | 01:06:18 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_31 | a17b_mprofile | serc | COMPLETED | 01:07:20 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_32 | a17b_mprofile | serc | COMPLETED | 01:08:56 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_33 | a17b_mprofile | serc | COMPLETED | 01:09:09 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_34 | a17b_mprofile | serc | COMPLETED | 01:05:52 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_35 | a17b_mprofile | serc | COMPLETED | 01:09:04 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_36 | a17b_mprofile | serc | COMPLETED | 01:04:31 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_37 | a17b_mprofile | serc | COMPLETED | 01:07:51 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_38 | a17b_mprofile | serc | COMPLETED | 01:08:23 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_39 | a17b_mprofile | serc | COMPLETED | 01:12:11 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_40 | a17b_mprofile | serc | COMPLETED | 01:05:10 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_41 | a17b_mprofile | serc | COMPLETED | 01:09:21 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_42 | a17b_mprofile | serc | COMPLETED | 01:13:41 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_43 | a17b_mprofile | serc | COMPLETED | 01:06:46 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_44 | a17b_mprofile | serc | COMPLETED | 01:08:46 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_45 | a17b_mprofile | serc | COMPLETED | 01:04:21 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_46 | a17b_mprofile | serc | COMPLETED | 01:14:41 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_47 | a17b_mprofile | serc | COMPLETED | 01:11:52 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_48 | a17b_mprofile | serc | COMPLETED | 01:08:10 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_49 | a17b_mprofile | serc | COMPLETED | 01:19:10 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_50 | a17b_mprofile | serc | COMPLETED | 01:18:08 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_51 | a17b_mprofile | serc | COMPLETED | 01:15:30 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_52 | a17b_mprofile | serc | COMPLETED | 01:18:26 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_53 | a17b_mprofile | serc | COMPLETED | 01:19:45 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_54 | a17b_mprofile | serc | COMPLETED | 01:19:31 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_55 | a17b_mprofile | serc | COMPLETED | 01:19:26 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_56 | a17b_mprofile | serc | COMPLETED | 01:21:05 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_57 | a17b_mprofile | serc | COMPLETED | 01:16:47 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_58 | a17b_mprofile | serc | COMPLETED | 01:21:11 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_59 | a17b_mprofile | serc | COMPLETED | 01:19:24 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_60 | a17b_mprofile | serc | COMPLETED | 01:20:23 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_61 | a17b_mprofile | serc | COMPLETED | 01:15:22 | 03:00:00 | 1 |  | 00:00:00 |
| 52881438_62 | a17b_mprofile | serc | COMPLETED | 01:10:22 | 03:00:00 | 1 |  | 00:00:00 |
| 52881442 | a17b_driver | serc | COMPLETED | 00:04:13 | 04:00:00 | 2 |  | 00:00:00 |

Peak memory per job or array (GB): 52874596: 0.8; 52874604: 0.4; 52874660: 0.4; 52877061: 1.6; 52877071: 0.9; 52880640: 1.2; 52880639: 1.6; 52881438: 5.2; 52881442: 1.4

Core-hours used (elapsed x CPUs over all tasks): 76.0; sum of CPU-weighted limits: 233.0; tasks: 93.
