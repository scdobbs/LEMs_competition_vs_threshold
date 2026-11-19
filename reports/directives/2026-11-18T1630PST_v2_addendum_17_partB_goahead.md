# [2026-11-18 16:30 PST] Addendum 17 Part B: go-ahead, exclusions, and the primary tile set

This is the written go-ahead for **Part B** of `reports/directives/2026-11-18T1500PST_v2_addendum_17_register_figures_measure.md`. It follows the Part A report `reports/v2/20261118T1600PST_register_figures.md`. Continue on branch `v2.3-tiles`.

Part B runs as written in Addendum 17 §5–§8. The only changes are the ones below. All of them were set by Steve **after viewing the Part A figures and before any measurement**. Under the standing rule (§0.4), rules may not change once results exist. None do yet, so these changes are allowed.

## 1. Exclusions (Steve, 2026-11-18, from the Part A figures)

### 1.1 Tier 1: excluded from every result
These tiles are not measured, except AP1_3dep (§1.3). They stay registered. List each one with its reason in an appendix of the Part B report.

| tile | reason |
| --- | --- |
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

### 1.2 Tier 2: kept, plus a "clean subset" sensitivity
These tiles are measured and enter the primary set. Every site-level quantity is also reported on the primary set minus Tier 2: the pooled θ, S(m) and its argmin, and the site summaries of K₀ and D₀. The reason is that roads and pads on ridges flatten hilltops, which biases |C_ht| low and D₀ high.

| tile | reason |
| --- | --- |
| GM12, GM29 | a road along the divide |
| GM26 | switchback roads and terracing in the lower left |
| AP25 | small pads on the ridges |
| TTV11 | fire roads |

### 1.3 AP1_3dep
AP1_3dep is one of the five original AP tiles. It is excluded from the primary set and from S(m) (§1.1). It is still measured, and only for the AP survey comparison (rule 6.3: AP1–AP5, PAMAP against 3DEP) and for continuity with v2. Report it separately and mark it "excluded (road), continuity only".

## 2. GM: the FEMA 2018 survey is the primary survey for every GM tile

**This replaces decision 2 and changes the role of decision 4.** The reasons are in Part A §3.4:
- Two routings of 2018 data agree with each other, against Perron's 2008 strip, at GM2's trunk and at GM5's first trunk.
- GM3's old grid is missing a 2.48 km² inflow that the NCALM footprint cut off.
- A single survey and a single area rule (full-grid routing) then hold for every tile in the study. This is what was already done at AP, where 3DEP replaced PAMAP for the original tiles.

1. **The primary GM tiles are GM1_fema–GM7_fema plus GM8–GM30,** minus the Tier 1 exclusions and the rules of §3. Their area source is the full-grid routing of their own survey grid: the control grid for GM1_fema–GM7_fema, the extension grid for GM8–GM30. The primary GM set enters S(m), the pooled θ and the site summaries.
2. **The NCALM tiles GM1–GM7 stay measured, as the continuity check against v2 and the original submission.** GM1–GM5 use `area_model`, the old grids, loaded explicitly (Part A deviation 8); GM6–GM7 use their registered areas. They do not enter S(m). In the report they replace the control tiles in the role the directive gave the control tiles.
3. **Rule 6.4 becomes a report-only survey comparison.** Compute the same ratios (FEMA/NCALM for k_s and for |C_ht(L_f)|, the median, and the largest single-tile difference) and say whether the ±15 % and ±30 % criteria are met. No pooling decision follows, because the primary GM set is a single survey.
4. **Rule 6.2 (site m stability) at GM:** report the GM site m on the primary set against the v2 value (0.31). Also give the argmin of S(m) over the NCALM tiles GM1–GM7 alone, at the v2 settings, for the record.
5. **The decision-7 trunk list for Addendum 18** is now GM2_fema, GM4_fema, GM5_fema, AP3 and AP4. GM2 and GM4 are on NCALM, continuity only.

## 3. Two screens written now, applied before the primary set is fixed

Run both screens first. Report the primary set, per site, before any m-profile runs. The screened tiles are still measured (§5 of the directive), but they are kept out of the primary set and reported separately.

### 3.1 GM bedrock screen
The high eastern tiles (GM14, GM18, GM21, GM27, GM28) look like a different landscape from the mesa. Until now the old elevation window quietly did this job. Replace it with a stated rule, applied to **every** primary GM tile:

1. **Map.** Use the Digital Geologic Map of Monterey County, California, 1934–2001 (L. I. Rosenberg, compiled from 79 quadrangles at 1:24,000; public domain).
   - Source: https://purl.stanford.edu/cm427jp1187 (shapefile download).
   - If Sherlock cannot reach it, report this and stop this screen only; Steve will place the shapefile in `$SCRATCH/lem_topography/`.
   - Record the file's sha256 and put a copy in the copies of record. Do not commit it.
2. **Reference unit.** This is the bedrock unit covering the largest combined area of the GM1_fema–GM7_fema footprints, excluding Quaternary surficial units. The expected answer is the Paso Robles Formation; report what is found, with the map's unit label.
3. **Rule.** For each tile, compute the fraction of the footprint mapped as the reference unit plus Quaternary alluvium, terrace and colluvial units. Do not count mapped landslide deposits.
   - The tile **passes if the fraction is ≥ 90 %**.
   - A tile that fails leaves the primary set and is reported as "off-mesa".
4. **Report** a table for all 30 GM tiles: the unit fractions, pass or fail, and a one-panel map of the tiles on the geology.
5. **If the rule removes any of GM1_fema–GM7_fema,** report this and stop before the m-profiles.

AP and TTV get no bedrock screen.

### 3.2 Gap-fill screen on the measurement pixels
The tile-wide nearest-neighbour fill fraction is not the quantity that matters: TTV8 is 14 % filled and TTV9 16 %. What matters is whether the filled cells fall on the pixels the measurements use. For every primary tile, using `nnfill_mask` (and the filled-gap mask):

1. Compute the filled fraction on:
   - (a) the channel pixels used in the χ–z fit for k_s;
   - (b) the option (iii) hilltop pixels at L_f = 11 m.
2. **If either fraction exceeds 20 %,** the tile leaves the primary set and is reported as "gap-filled".
3. Report (a) and (b) for every tile.

NCALM GM1–GM7 have no fill mask. They are continuity only and are exempt.

## 4. The expected primary set, before the screens of §3

| site | tiles | count |
| --- | --- | --- |
| GM | GM1_fema–GM7_fema, GM8, GM9, GM11–GM22, GM24–GM30 | 28 |
| AP | AP2_3dep–AP5_3dep, AP6–AP11, AP13, AP14, AP18–AP20, AP23–AP25, AP27–AP29 | 21 |
| TTV | TTV1–TTV9, TTV11, TTV15 | 11 |

Do not draw replacement tiles.

Also measured, outside the primary set:
- GM1–GM7 on NCALM (continuity);
- AP1_3dep (continuity, survey comparison);
- any tile the screens of §3 remove.

## 5. Everything else as written
- E per site (decision 6).
- Measurement methods (§5).
- Rules 6.1 and 6.3 unchanged.
- Cost (§7): still about 90 tiles, expected ≤ 150 core-hours; Sherlock and `serc` guidelines are binding.
- Report outline (§8). Add to section 1 of the outline: the exclusions, the tiers, the two screens and the final primary set.

**Stop** after the Part B report. Do not run the AD or partitioned joint fits (Addendum 18). Do not merge or tag. Push after the Addendum 2 §5 check.
