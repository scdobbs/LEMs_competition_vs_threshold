# [2026-11-21 06:30 PST] Addendum 20: Perron's hilltop estimator, the D bracket without the segment filter, snapped TTV heads, Part B, and a calibration of E, D and K in the Montgomery & Dietrich valley

Directive: `reports/directives/2026-11-20T1050PST_v2_addendum_20_perron_D0_heads_partB.md` (outline fixed by its section 8). Branch `v2.3-tiles`
(from cfef76c, the addendum 19 report). **No package code was changed.** Section 12 is an addition Steve asked for during the work (2026-11-20/21):
calibrate D and K for the M&D Tennessee Valley area against the published field values, reproducing the published methods first. Every number is
tied to the run directory `$GROUP_HOME/lemcvt_results/20261120T185052Z_a20_perron_D0_heads_partB/` or to a Slurm job (Appendix E13). Scripts:
`reports/v2/scripts/a20_*.py`, `make_a20_tables.py`. Figures: `reports/v2/figures/a20/`. **This is the addendum's stop.** The appendices are
machine-generated.

## Summary

| item | result |
| --- | --- |
| 1 state | Decision 1 changes a filter after results (stated first, section 1). Tests on `serc` 274 passed. Two numerical fixes and one rule-input fix, all declared (section 1). **The A19 report misstated the crest-fragment length** (6–10 px; the records say 2–4, median 2). |
| 2 crest D | **R1 passes everywhere** (D₀,N ≥ 500 px on 22/22 GM, 21/21 AP, 11/11 TTV). **R13 fails everywhere** (plateau on 12/22, 6/21, 6/11 tiles; no pooled plateau). **R11P fails at GM and AP** (no pooled plateau at r = 7 m; the fixed-range ratios are 0.63 × at GM and 1.63 × at AP). The divide-line estimator with Perron's 7 m disk is 0.84 × Perron at GM and 2.19 × at AP. D₀,P/D₀,N = 1.44 / 2.00 / 1.98: the binned estimator measures hillslope curvature, not crest curvature (section 2.4). |
| 3 bracket | D_R/D₀,P 1.30–1.37; E* < 10 on 97–100 % of pixels. TTV anchor: D₀,N 1.9 × and D₀,P 4.1 × 2.5 × 10⁻³ (at the configured E; see section 12). |
| 4 heads | Snapping along each pick's own D8 path does not reach the hollow axis: **A_P10 median 529 m²** (A_3 611, Table 1 7,400). **R5 fails** (0.039 × M&D's relation; b̂ = 2.23 [0.94, 3.54]). At map scale (12 m grid) the picks' source areas reproduce Table 1 (median 7,200 m²; section 12). |
| 5 field thresholds | **R6 fails** (PF at A_F = 529 m² beats AD on 5/11; SF 5–6/11). At 7,400 m² PF beats AD on 8/11 (no verdict). **R14's reading does not hold.** |
| 6 SJ | **R7 fails** (α > 0 preferred on 6/11 held-out); α = 0 reproduces the A18 rows exactly on 11/11. |
| 7 model heads | **R8: only PF at 7,400 m² agrees with the mapped heads** (median 15 m, ratio 1.39). P4 has the right count but its channels are 62 m from the picks; every field threshold from the snapped heads gives 18–40 × too many heads. |
| 8 D statements | **GM and AP lower-end verdicts are withheld (R11P).** At TTV F3, E1, E2 fail (6/11, 7/11, 6/11 with D₀,P); H2 changes to 9/11 (holds) and F4 to fails. R10: P4 is **above** the bracket on 21/21 AP and 11/11 TTV tiles (this reading stands) and inside on 16/22 GM. Every Gram-route and reconstruction check passes (98/98 specs). |
| 12 TTV calibration | **The configured TTV E = 77 m/Myr is Heimsath's soil-production maximum P₀, not an erosion rate** (ridges 21–60, median 27; basins 64–102 m/Myr). Heimsath's curvature method reproduces his rates on our lidar; M&D's source areas are reproduced at map scale; Perron's GM D/K and m are reproduced by his S* method within uncertainty; his hilltop Laplacian is reproduced by the crest estimator, not by the binned one. Calibrated TTV values: **D ≈ 0.0025 m²/yr** (published, consistent with our crest curvature and the ridge E), **D/K ≈ 84 m^(2m+1), m ≈ 0.30** (S*), **K ≈ 3 × 10⁻⁵ m^(1−2m)/yr**. The A18 P4 fits have D 6–17 × the calibrated D and K within 2.5 ×. |
| cost | 147.7 core-hours used (619 of limits, no submission over 300; ~880 of Steve's approved 1,000–2,000 to date). |

## 1. State check, deviations, decisions as applied

**Decision 1 changes a filter after results.** The 50-pixel segment filter of addendum 19 was removed after addendum 19 reported that it made R1
fail at every site. Its premise was false: D8 divide cells along a crest come in fragments of a median 2 px (2–4 per tile, 73 of 73 tiles), so a
50-px minimum removed 60–80 % of every crest. R1's threshold is unchanged. **Correction to the addendum 19 report:** its section 2.1 gave the
fragment length as "a median of 6–10 px"; the records give 2–4 px (median 2). The directive's figure was right.

**At the start (2026-11-20 10:52 PST, sh03-ln05).** `v2.3-tiles` = `origin` = cfef76c; no job running; `/etc/agents/AGENTS.md` read (and re-read
after each session restart). Tests on `serc` at cfef76c (53108805): ruff and mypy clean, **274 passed**. The addendum 19 run directory and its
per-tile outputs exist; the heads CSV checksum (`ac87de3d…e702b6`) is checked in code before use.

**Deviations and choices (each listed, none silent):**
1. **Numerical zero in the convexity test.** On pit-filled flats the 7 m disk Laplacian is exactly zero but the FFT returns ±10⁻¹³ m⁻¹, so half
   of those cells counted as "convex" and their log-curvature (≈ −13) destroyed the geometric means. Every convexity test now uses |∇²z| > 10⁻⁹ m⁻¹
   (four orders above the verified FFT error of 1.6 × 10⁻¹³, six below any hilltop curvature). Found in the first site run; tiles rerun.
2. **Flat cells and the plateau rule (rule unchanged).** Even with (1), on several tiles the lowest A|∇z| bin is mostly pit-filled flats (convex
   fraction 0.00–0.03) yet meets the 200-cell start condition, so the rule starts there and stops at the next bin (GM11, GM14, GM15, GM21). The
   rule's verdicts are reported as computed; a labelled "no-flat" sensitivity (flat cells removed from the bin counts) is beside them. It adds one
   plateau tile at GM and at AP, none at TTV, and changes no pooled verdict.
3. **The addendum 19 D-statement code had a key-format bug** (`a19_dstat.terms_of` built arrays that `sensitivity._q` cannot read; that code never
   ran in addendum 19). `a20_dstat` overrides it; the route then reproduces the addendum 18 P1 and P2 **exactly** on every spec (relative 0.0).
4. **Grid.** D₀,P uses the registered pit-filled 1 m grid (the grid of addendum 19's `total_curvature`), as decision 2 says. The unfilled surface
   would be more defensible for curvature (section 9).
5. **"Cell-weighted" geometric mean** read as weighted by the convex cells of each bin; the start bin is not required to pass the convex test
   (recorded per tile). Cells with A|∇z| ≥ 10⁵ m² are dropped (counted).
6. **D₀,N uncertainty.** With 2-px segments the segment bootstrap approaches a pixel bootstrap; a 100 m block bootstrap is reported beside it
   (labelled; Appendix E2). The brackets use the segment bootstrap as written.
7. **SJ sizing.** Addendum 18 timings differ by 25 % between TTV tiles, so each tile got its own array with a limit of 2 × its estimate
   (11 submissions of 3, then 11 of 1 for b̂); no submission over 31 core-hours of limits.
8. **b̂ = 2.23 → α = 2.25** differs from 1.65 and 2 by more than 0.1, so SJ was also run at 2.25 (decision 6c), in a second submission.
9. **Snapped heads.** S_up/S_down for the A_P5 and A_P20 fits are taken from their own cells. The axis entry (10 × the pick-cell area) is a low
   bar on 1 m D8 (median 6.8 m, 481 m²); it is reported, not used.
10. **Section 12 reads `papers/`** (Heimsath et al. 1997, 1999; Montgomery & Dietrich 1989; Perron et al. 2008, 2009 and its SI, 2012 SI), on
    Steve's instruction for this addition. Nothing from `papers/` is committed.

## 2. Crest D (sections 2.1–2.8; rules R1, R13, R11P)

**Verification (2.1).** The disk quadratic against an analytic paraboloid and a plane, at r = 3–15 m on 1 m and on the 0.975 m PAMAP lattice:
maximum error 1.6 × 10⁻¹³ m⁻¹ in ∇²z and 4.6 × 10⁻¹⁴ in |∇z|; against an independent per-cell lstsq on a noisy surface ≤ 1.1 × 10⁻¹³; the
exclusion margin, the plateau rule and the bootstrap on constructed cases all as intended (Appendix E1).

**D₀,N (decision 1) and R1.** Without the segment filter every primary tile has ≥ 500 counted pixels: **R1 passes** (GM 22/22, AP 21/21,
TTV 11/11; with the 20-px fragment filter 21/22, 19/21, 11/11, also passing). Site medians: GM 0.0155, AP 0.0045, TTV 0.0048 m²/yr (addendum 19's
segment-filtered set: 0.0136, 0.0039, 0.0049; D₀,R: 0.0153, 0.0044, 0.0029).

**D₀,P (decision 2), R13.**

| site | tiles with a plateau (r = 7 m) | pooled plateau | D₀,P median (tiles with one) | r = 3 / 5 / 7 / 10 / 15 m | fixed ≤ 1 / ≤ 3 m² | D∞ | D₀,P/D₀,N |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GM | **12/22** | no | 0.0249 | 0.015 / 0.022 / 0.025 / 0.023 / 0.025 | 0.019 / 0.019 | 0.023 | 1.44 |
| AP | **6/21** | no | 0.0092 | 0.004 / 0.007 / 0.009 / 0.012 / 0.011 | 0.006 / 0.006 | 0.009 | 2.00 |
| TTV | **6/11** | no | 0.0102 | 0.005 / 0.008 / 0.010 / 0.015 / 0.013 | 0.007 / 0.006 | – | 1.98 |

**R13 fails at every site.** The binned curves (`binned_curves_r7.png`) show why: the convex fraction sits at 0.56–0.62 across 10⁻¹–10 m² (the
rule needs ≥ 0.6 in every bin), and the lowest bins hold few, mostly flat cells whose geometric-mean curvature is 2–4 × lower than the bins above
them, so the pooled rule stops at the first or second bin. Where tiles do have a plateau, it spans 10⁻²–10 m² of A|∇z|, which on a 1 m D8 grid
is most of the hillslope (70–80 % of each tile's cells), not the hilltops: D₀,P is a mean hillslope curvature, hence 1.4–2.0 × D₀,N.

**R11P fails at GM and AP** (no pooled plateau at r = 7 m). Beside it (report only): pooled fixed-range ratios GM 0.63 / 0.65, AP 1.63 / 1.63;
GM plateaus at r = 5, 10, 15 m give 0.55, 0.45, 0.44; on Perron's own survey (NCALM) at r = 7 m 0.44. The four-way comparison (Appendix E5) at
the divide-line pixels: 11 m square median 0.85 × (GM) and 2.24 × (AP), 7 m disk 0.84 × and 2.19 ×; the plateau cells have no pooled set.
**R11P reading:** "neither passes at AP": **location, survey or method detail**. Section 12 resolves the method part: Perron's binned Laplacian
came from first-order drainage basins (his Fig. 3a caption and SI 2.1), which decision 2 did not include; restricted to those basins with his own
network threshold the binned method still gives 0.26–0.28 × at GM, whereas the divide-line estimator gives 0.94–0.95 × on the same NCALM survey (0.84–0.85 × on the FEMA primary tiles). On our 1 m D8 data, the
cells of lowest A|∇z| are not hilltops, and Perron's number is reproduced by the crest set, not by the low-A|∇z| set. At AP the crest estimator is
2.2 × Perron at every window (addendum 19) and no estimator reproduces it; AP was not calibrated in this addition.

**Plateau maps** `plateau/<tile>_plateau.png` and the site galleries show it directly: the r = 7 m plateau cells cover whole hillslopes, the D₀,N
pixels the crest lines. **Survey pairs:** D₀,P new/old 1.05–1.12 at GM (4 pairs with plateaus on both), 1.58 at AP (1 pair); D₀,N 1.00 and 1.44.

## 3. The bracket (sections 3.1, 3.2, 3.4)
D_R on the decision-1 crest pixels: GM 0.030, AP 0.013, TTV 0.013 m²/yr (Euclidean: 0.022, 0.010, 0.009); L_H medians 97, 104, 80 m;
D_R/D₀,P 1.33, 1.37, 1.31; per-pixel D_R/D_C 2.0, 2.6, 2.6. E* at S_c = 1.27: medians 1.5–2.0, below 10 on 98.4–99.6 % of pixels; fitted S_c
0.40 / 0.34 / 0.37, E* at the fit 4.7 / 6.9 / 6.8. TTV anchor (decision 14, against 2.5 × 10⁻³): D₀,P 4.1 ×, D₀,N 1.9 ×, D₀,N at 7 m 1.25 ×,
D_R 5.2 ×, all at the configured E = 77 m/Myr. Section 12 shows that E is the soil-production maximum; at the measured ridge rate the crest D
reproduces the anchor (0.0016–0.0027). Interval figures: `<site>_D_bracket_intervals.png` (the Part B fits drawn on them).

## 4. The heads (decision 5; rules R4, R5)
Each pick's D8 path from the pick cell: A at 0 / 5 / 10 / 15 / 20 / 30 m of path has medians 4 / 141 / **529** / 1,393 / 2,307 / 4,608 m²; the
area grows steadily with distance and shows no jump at a valley axis (the axis entry, 10 × the pick-cell area, is reached at a median 6.8 m). The
QC pages (`figures/a20/heads/heads_qc_page1-4.png`) show why: on the 1 m D8 grid the hillslopes around the picks carry parallel flow lines of
100–1,000 m², and the hollows are not single convergent lines. **R4** (unchanged): a = 1,978, b = 1.647 (passes). **R5 fails**: fit (ii)
a = 53 m², b = 2.23 [0.94, 3.54], R² 0.12 (M&D's 1.65 inside the band) but the picks sit at a median **0.039 ×** M&D's relation (14 % within
0.5–2). (iii) b = 1.68, (iv) b = 2.28, A_P5 0.015 ×, A_P20 0.17 ×. A·S² median 65 m² (S_up), 53 (S_down) against Table 1's 1,500. Field values:
A_F = 529 m², C_F = 75 (α 1.65), 53 (α 2), 41 m² (α 2.25). **Section 12 shows the mismatch is ours:** measured at map scale the same picks give
M&D's areas.

## 5. Field-fixed thresholds (rules R6, R14)

| threshold | beats AD (of 11) | MSE/P4 median | D median (m²/yr) | D at bound | heads per tile |
| --- | --- | --- | --- | --- | --- |
| PF at A_F = 529 m² (decides R6) | **5** | 2.92 | 0.0078 | 1 | 500–717 |
| PF at 611 m² (A_3) | 5 | 2.75 | 0.0080 | 1 | 421–616 |
| PF at 2,848 m² (A_10) | 8 | 2.11 | 0.0147 | 0 | 71–115 |
| PF at 7,400 m² (Table 1) | 8 | 1.62 | 0.0203 | 0 | 32–53 |
| SF α = 0 / 1.65 / 2 at C_F | 5 / 6 / 6 | 2.92 / 2.34 / 2.30 | 0.0078 / 0.050 / 0.038 | 1 / 6 / 5 | – |

**R6 fails** for PF and for SF at every α. PF3 (K = K₀) against A1: Appendix E8. **R14's reading does not hold** (PF at A_F inside the bracket on
3/11, beats AD on 5/11; MSE/P4 2.92). SF at α = 0 equals PF exactly on 11/11 (check of decision 5b).

## 6. The slope-dependent threshold (rule R7; R9)
SJ at α = 0, 1.65, 2 and 2.25 on 11 tiles (44 runs, 2,000 + refine evaluations each). **R7 fails: preferred on 6/11** (needed 9; median
held-out improvement 2.7 %); best α 1.65 on 6 tiles, 2.25 on 4, 2 on 1. In sample, α = 1.65 and 2 are 3–5 % *worse* than α = 0 at the median.
α = 0 reproduces the addendum 18 P4 rows exactly (common rows to 10⁻⁹ on 11/11; the same optimum on 10/11, TTV11 differs by 3.9 × 10⁻⁴ in MSE
because its A18 optimum is not on the SJ grid). D is at the 0.05 m²/yr bound on 4–7 tiles per α. **R9** (TTV): P4 D/D₀,P 4.1 ×, D/D_R 3.5 ×;
SJ similar; PF at A_F 0.69 × D₀,P (Appendix E8).

## 7. Model heads against snapped heads (rule R8)

| model | median distance (m) | heads/picks | reading |
| --- | --- | --- | --- |
| P4 (A18) | 61.7 | 0.76 | count in range, distances large |
| **PF at 7,400 m²** | **15.2** | **1.39** | **agrees with the mapped heads** |
| PF at 2,848 m² | 5.0 | 2.98 | too many channels |
| PF at A_F 529 / 611 m² | 1.0 | 21.6 / 18.0 | too many channels |
| SF α 1.65 / 2 | 0.0 | 35.6 / 40.0 | too many channels |
| SJ α 0 / 1.65 / 2 / 2.25 | 62 / 77 / 77 / 77 | 0.76 / 0.69 / 0.76 / 0.74 | count in range, distances large |

Figures `figures/a20/model_heads/TTV1-3_model_heads.png`. The only threshold that puts channel heads where M&D and Steve mapped them is M&D's own
median source area; the tuned P4 threshold (9,100–89,000 m² on TTV1–TTV3) gets the count right on TTV2 but not the places.

## 8. The D statements and the bracket (rules R2, R10)
Route: P1 and P2 from the stored Gram rows (reproducing addendum 18 exactly), A2 by exact reconstruction plus Brent; one direct evaluation per tile
as the check: **98/98 specs, every Gram-route check (89 non-trunk) and every A2 reconstruction check (≤ 1.4 × 10⁻¹³) pass.**

| site | F3 (D₀,P) | E1 | E2 | H2 | F4 (AD |log₁₀ D/D₀|) | lower-end verdict |
| --- | --- | --- | --- | --- | --- | --- |
| GM | 21/22 | 20/22 | 20/22 | 19/22 | 2.23 | **withheld (R11P)** |
| AP | 18/21 | 12/21 | 12/21 | 15/21 | 0.81 | **withheld (R11P)** |
| TTV | 6/11 fails | 7/11 fails | 6/11 fails | **9/11 holds** | 0.70 fails | TTV not gated |

Changed against addendum 18 at TTV: **H2 5/9 → 9/11 (now holds); F4 holds → fails.** No GM or AP verdict is stated. **R10** (Appendix E10):
P4 inside / below / above: GM 16 / 1 / 5, **AP 0 / 0 / 21, TTV 0 / 0 / 11**; AD below at GM 22/22 and AP 21/21. The upper-end reading stands:
**at AP and TTV the partitioned D is above what the measured hillslope relief admits for a linear model on every tile**; at TTV, read with R5–R8,
the area-only threshold is mis-specified there (the reading written in addendum 19). Model-network D_R: each fit's D is 0.8–1.2 × the D_R of its
own channel network (Appendix E9): a fit's D is the relief term of the hillslope length the fit creates.

## 9. What the evidence supports, and where it is weak
**Supported.**
1. The crest (divide-line) estimator reproduces Perron's GM hilltop curvature (0.94–0.95 × on his NCALM survey, 0.84–0.85 × on FEMA) and, with Heimsath's E and D, the TTV rates
   (section 12). Without the segment filter it has ≥ 500 pixels on every primary tile.
2. The binned (Perron) estimator, as decision 2 specified it and even restricted to first-order basins, measures mean hillslope curvature on 1 m D8
   data and does not reproduce Perron's hilltop value (0.26–0.65 ×). R13 and R11P fail for that reason, not because of noise (larger radii make
   the GM ratio smaller, not larger).
3. At TTV, a field threshold equal to M&D's measured source area (7,400 m²) places model channel heads where they are mapped (R8) and beats AD on
   8/11 tiles at 1.6 × P4's misfit; field thresholds from 1 m D8 areas at the picks do not.
4. P4's D is above the relief bracket on every AP and TTV tile.
**Weak.**
1. GM and AP lower-end D statements carry no verdict (R11P). AP has no estimator that reproduces Perron's 5.5 × 10⁻³.
2. The snapped-head areas (decision 5) are not a field measurement; every A_F, C_F and R5/R6/R14 verdict built on them inherits that.
3. TTV is not in morphological steady state (Heimsath et al. 1997): ridges lower at 21–60 m/Myr, basins at 64–102; a single U is a mis-specified
   input there (section 12).
4. Pit-filled grids for curvature; Perron's basin selection was subjective and his quoted uncertainties (±3, ±0.003) are standard errors far
   smaller than the procedure-to-procedure spread we see.

## 10. Decisions needed
1. **TTV E** (section 12): replace 77 m/Myr (P₀) with a measured rate. Ridge lowering (Heimsath Table 1, soil-covered greywacke: median 27,
   mean 35 m/Myr) for D₀; basin average (64–66 m/Myr for Heimsath's sub-basin 1, 102 for sub-basin 2; both lie in or beside the M&D study area on our approximate, ±100 m, georeference of his map)
   for U and K₀. Which E the models use at TTV, given the non-steady state, is Steve's call.
2. **The D₀ estimator from here on:** the crest (D₀,N) estimator, which reproduces Perron at GM and Heimsath at TTV, or D₀,P as decision 2
   defines it. The evidence favours D₀,N.
3. **TTV threshold for the field-fixed model:** the M&D Table 1 median (7,400 m²), or map-scale areas at Steve's picks (median 7,200 m² on a
   12 m grid), in place of A_P10.
4. **AP calibration** (Perron's 5.5 × 10⁻³ is not reproduced by any estimator) is open; an AP analogue of section 12 would need Perron's AP
   basins and E (55 m/Myr) handled the same way.
5. Nothing merged or tagged.

## 12. Supplement (placed after section 10; section 11 is the appendices): calibration of E, D and K in the Montgomery & Dietrich Tennessee Valley area (Steve's addition, 2026-11-20/21)

**The question.** Are we determining D and K reliably at TTV, or using well-determined published values, so that model performance has a sound
yardstick? The principle Steve set: if we fail to reproduce another author's result, assume the mistake is ours. So every published method was
first reproduced on our data before its result was used. Scope: the M&D (1989) study polygon (2.06 km², on TTV1–TTV3), where M&D and Steve mapped
channel heads. Scripts: `a20_ttv_cal.py` (T1–T4), `a20_sstar_dinf.py`, `a20_sstar_perron.py`, `a20_sstar_perron2.py`, `a20_lap_h_basins.py`;
outputs `<run>/ttv_cal/`. About 2 core-hours.

### 12.1 What the papers give (read from `papers/`; page or table cited)

| source | quantity | value |
| --- | --- | --- |
| Heimsath et al. 1999 (Geomorphology 27, 151–172), eq. 9 | soil production function | 77 ± 9 · exp(−0.023 h) m/Myr (h in cm); **77 m/Myr is the rate under zero soil** |
| ibid., Table 1 | bedrock lowering under soil, greywacke noses (9 samples, h 16–60 cm) | 21–60 m/Myr, median 27, mean 35 |
| ibid., Table 1 | exposed greywacke (3), greenstone, chert | 39, 91, 107; 20; 15 m/Myr |
| ibid., Table 1 | stream sediment, sub-basin 1 (2 samples), sub-basin 2 | 64 ± 12, 66 ± 13; 102 ± 25 m/Myr |
| ibid., eq. 4 and Fig. 8 | soil transport coefficient, density ratio | K = 50 cm²/yr (Reneau 1988, via McKean et al. 1993); ρ_s/ρ_r = 0.5 (Table 1 densities 1.4 and 2.2 g/cm³ give 0.64) |
| ibid., section 4.1, Fig. 4 | curvature method | kriged survey grid, 9-point stencil (eq. 6); curvature scale-independent at grids ≥ 5 m; 5 m used |
| Heimsath et al. 1997 (Nature; the volume is not printed on the pages we have) | state of the landscape | "Tennessee Valley is undergoing systematic morphological change"; ridges lower more slowly than the steep tributaries |
| Montgomery & Dietrich 1989 (WRR 25(8), 1907–1918), p. 1910 and Table 1 | source areas | planimetered on an enlargement of the USGS Point Bonita 7.5′ map (40 ft contours); 63 heads, median 7,400 m² |
| Perron et al. 2009 (Nature 460), eqs. 4–5, Table 1; SI 2.1 | U/D, D/K, m | −∇²z_h = U/D; S* = \|∇z\|/(∇²z − ∇²z_h) = (D/K)A^−m; GM D/K = 124 ± 3, m = 0.35 ± 0.003; S* on stream profiles of first-order basins; network from A_t = (spectral roll-off wavelength)² |
| Perron et al. 2008 (JGR 113, F04003), section 3.1.2 | spectral roll-off | GM 5.6 × 10⁻³ m⁻¹ (180 m), so A_t = 32,400 m² at GM |

Rock-equivalent D (our convention, D = E/|∇²z|): **D = (ρ_s/ρ_r) K = 0.0025 m²/yr** (0.0032 with the measured densities). Heimsath et al. showed
this D, applied to nose curvature, reproduces their cosmogenic rates with no adjusted parameter (their Fig. 8).

### 12.2 Our mistake: the configured TTV erosion rate
`configs/sites/TTV.yaml` sets U = 7.7 × 10⁻⁵ m/yr, sourced as "Heimsath et al. (1997), 77 ± 15 m/Myr". **That is P₀, the production rate under
zero soil**, the upper bound of the soil production function, not a lowering rate of the landscape. Measured rates are 21–60 m/Myr on the soil-
mantled noses (median 27) and 64–102 m/Myr basin-averaged. Consequences:
- Every absolute D₀ and K₀ at TTV in addenda 17–20 used 77 m/Myr. At the ridge rate they are 0.35–0.45 × as large; at the sub-basin 1 rate 0.84 ×.
- The fitted K and D also scale with U, so the ratios D_fit/D₀ and K_fit/K₀ in addenda 18–20 do not depend on E **if one E applies to both**.
  At TTV it does not: hilltops lower at about 27–35 m/Myr and the basins at 64–102 (Heimsath et al. 1997). A steady-state model with one U is
  mis-specified there by a factor of about 2–3 between crest and basin.

### 12.3 Reproduction tests

**(a) Heimsath's curvature method on our lidar (T1).** 1 m lidar block-averaged to 1–9 m grids; Heimsath's eq. 6; 1,316 divergent nose cells in the
M&D polygon (convex at 5 m, slope 8–30°, off divides and channels). Our noses cannot be matched to his four noses better than about ±100 m (his
Fig. 2a inset has 10 m contours), so this is a landscape-level test, not a point-by-point one.

| grid (m) | 1 | 2 | 3 | 5 | 7 | 9 |
| --- | --- | --- | --- | --- | --- | --- |
| median −∇²z (m⁻¹) | 0.0059 | 0.0090 | 0.0098 | 0.0099 | 0.0107 | 0.0106 |
| sd | 0.111 | 0.032 | 0.018 | 0.009 | 0.009 | 0.009 |

- **Scale-independence from 3–5 m reproduces his Fig. 4** (the 5 m values are selected convex, so the 5 m column is not an independent point).
- At 5 m the nose curvatures span 0.001–0.030 m⁻¹ (5–95 %), 74 % inside the bulk of his Fig. 6 (0.005–0.035).
- **With his D (0.0025) they imply lowering rates of median 25 m/Myr (IQR 13–42)** against his Table 1 soil-covered median of 27 (21–60);
  with D = 0.0032, 32 m/Myr. Implied soil depths from his production function: median 49 cm (IQR 27–78).
- **Verdict: reproduced.** Our curvature, his D and his E agree on the same ground.

**(b) M&D's source areas at Steve's 64 heads (T2).** Contributing area at each pick (largest of the 3 × 3 cells), on the lidar block-averaged to
b m and pit-filled, D8:

| grid (m) | 1 | 2 | 3 | 5 | 10 | 12 | M&D Table 1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| median area (m²) | 111 | 660 | 1,170 | 1,825 | 5,350 | **7,200** | **7,400** |
| IQR (m²) | 35–572 | 136–1,794 | 241–2,579 | 550–6,425 | 3,050–10,575 | 3,420–10,944 | 4,800–10,350 |
| share within Table 1's 5–95 % | 0.06 | 0.20 | 0.30 | 0.41 | 0.75 | 0.75 | 0.90 |

- **Verdict: reproduced at map scale.** At the scale of the map M&D measured on (40 ft contours, about a 10–12 m surface) the picks give M&D's
  source areas. On the 1 m grid, D8 spreads hollow flow into parallel lines, so area at a point is 10–60 × too small.
- **So the addendum 19 and 20 head areas (A_3, A_10, A_P10) and everything built on them (R5, A_F, C_F, R6, R14) are measurement artifacts of
  1 m D8, not properties of the heads.** D∞ at 5 and 10 m gives the same as D8 (2,047 and 5,438 m²).

**(c) Perron's D/K and m at GM (implementation check before using S* at TTV).** Perron's own survey (NCALM GM1–GM7), his 7 m disk, his network
threshold (180² m²), first-order basins, IRLS on bin means; 53 first-order basins:

| variant | D/K (68 %) | m (68 %) |
| --- | --- | --- |
| stream profiles (main stems), all basins, Perron's ∇²z_h | **95 [80, 126]** | **0.358 [0.343, 0.387]** |
| the same, our crest ∇²z_h | 105 [88, 139] | 0.366 [0.349, 0.395] |
| all basin cells (A ≥ 100 m²), all basins | 150 [115, 170] | 0.404 [0.370, 0.418] |
| Perron et al. 2009 | **124 ± 3** | **0.35 ± 0.003** |

- **Verdict: reproduced within our uncertainty** by the variant that follows his Fig. 3b (stream profiles).
- The earlier attempts with the wrong threshold (100–1,000 m², or a two-segment spectral kink at 22–37 m) did not reproduce it: m 0.42–0.87.
- Our 1 km GM tiles give roll-offs of 168–425 m by his 2008 procedure, the right order but tile-limited.
- Two parts remain judgement:
  - His basin selection ("points showing power-law relationships"). Our objective stand-in (R² ≥ 0.5) kept 1–16 of 53 basins and did not
    reproduce his values; it is dropped.
  - His quoted ± are standard errors that are much smaller than the procedural spread (D/K 95–150 across reasonable variants).

**(d) Perron's hilltop Laplacian at GM.** His binned Laplacian came from first-order basins (Fig. 3a caption). Restricted to the 53 first-order
basins at his threshold, our binned estimate is −0.0031 m⁻¹ at A|∇z| ≤ 1 m² (68 % −0.0034 to −0.0027), and the plateau rule gives −0.0033:
**0.26–0.28 × his −0.0118. Not reproduced.** The crest estimator on the same survey (NCALM GM1–GM7, D₀,N pixels, tile medians) reproduces it: 0.94 × (7 m disk) and 0.95 ×
(11 m square); on the FEMA primary tiles 0.84–0.85 ×. On 1 m D8 data the lowest-A|∇z| cells are mostly low-gradient hillslope and flat cells, not crests. So the hilltop curvature in
Perron's sense is measured on the crest set. **That makes the crest estimator, not the binned one, our validated hilltop curvature.** The binned
estimator is for the S* denominator, where Perron's own ∇²z_h and our crest value give the same D/K within 10 % (table above).

### 12.4 Calibrated values for the M&D Tennessee Valley area

| quantity | value | basis | reliability |
| --- | --- | --- | --- |
| E, ridges and noses | 27 (median), 35 (mean) m/Myr; range 21–60 | Heimsath 1999 Table 1, soil-covered greywacke | measured; local variation × 3 |
| E, basin average | 64–66 (sub-basin 1), 102 (sub-basin 2) m/Myr | ibid., stream sediment | measured; 2 sub-basins |
| hilltop curvature | 0.0164 m⁻¹ (crest, 7 m disk; 4,802 px in the polygon); 0.0099 (nose median, Heimsath method) | ours, validated by (a) and (d) | ± 10 % (bootstrap and estimator spread) |
| **D (rock-equivalent)** | **0.0025 m²/yr** (published); ours 0.0016–0.0021 (crest, E_ridge), 0.0027–0.0035 (nose, E_ridge) | 12.1; 12.3(a) | two independent routes agree within × 1.5 |
| D/K, m | **84 [70, 101] m^(2m+1), m = 0.30 [0.28, 0.32]** (stream profiles, A_t = 7,400 m² from the field heads); 75–97 and 0.29–0.32 across variants | S* as reproduced in 12.3(c) | about ± 30 % in D/K (the GM check) |
| **K** | **3.0 × 10⁻⁵ m^0.40/yr** at m = 0.30 (range 1.6–5.0 × 10⁻⁵); 2.4 × 10⁻⁵ m^0.34/yr with m fixed at 0.33 | D/(D/K) | about × 2 |
| channel-head area | 7,400 m² (Table 1); 7,200 m² at Steve's picks on a 12 m grid | 12.3(b) | measured |

m from S* (0.30 ± 0.02) is consistent with the TTV site m of addendum 17 (0.33).

### 12.5 The model fits against the calibrated values (TTV1–TTV3; fitted K and D rescaled to U = 65 m/Myr, the sub-basin 1 rate)

| tile | P4 A_crit (m²) | P4 D | P4 D / D_cal | PF(7,400 m²) D / D_cal | AD D / D_cal | P4 K / K_cal | K₀ / K_cal |
| --- | --- | --- | --- | --- | --- | --- | --- |
| TTV1 | 59,499 | 0.037 | 14.8 | 6.9 | 0.54 | 0.43 | 0.39 |
| TTV2 | 9,131 | 0.014 | 5.6 | 5.0 | 0.08 | 0.58 | 0.55 |
| TTV3 | 89,124 | 0.042 (at bound) | 16.9 | 6.8 | 0.003 | 1.31 | 0.92 |

D_cal = 0.0025 m²/yr; K_cal = 2.4 × 10⁻⁵ m^0.34/yr (m fixed at 0.33, the models' m). Reading:
- **K is reasonably determined** at TTV: the partitioned model's K and K₀ = E/k_s both lie within 0.4–1.3 × the S*-calibrated K.
- **D is not returned by either model.** The partitioned model needs 6–17 × the calibrated D. This holds even with the field threshold that puts
  channel heads in the right places (PF at 7,400 m²: 5–7 ×), so the D excess is not caused by the tuned threshold. AD's D is 0.003–0.5 ×.
- The measured relief says the same independently: D_R (TTV median 0.013 at E = 77) is 0.0046 m²/yr at the ridge E, 1.8 × D_cal. That is the
  nonlinear-hillslope factor addenda 8 and 19 found. The partitioned fits exceed it by a further 3–9 ×.
- **The remaining gap is about the steady-state assumption and E,** not the curvature measurement. A model forced with one U cannot match
  ridges lowering at 27–35 m/Myr and basins at 64–102. A model D compared with D_cal must use the ridge E for D₀ and the basin E for U, and say so.

### 12.6 What this establishes, and what it does not
- **Established at TTV:**
  - D (0.0025 m²/yr, within × 1.5) from published field work, which our crest curvature and the measured ridge E reproduce.
  - The head areas, at map scale.
  - K within about × 2.
  - m ≈ 0.30.
  - The previously used E was wrong (P₀).
- **Established at GM (as a method check):** Perron's D/K and m via S*, and his hilltop curvature via the crest estimator.
- **Not established:**
  - AP (Perron's 5.5 × 10⁻³ is not reproduced by any estimator here).
  - A single steady-state U for TTV.
  - Perron's binned hilltop Laplacian on 1 m D8 data.
- **Implication for model assessment.** With E corrected and D and K calibrated, the TTV yardstick is D_cal = 0.0025 m²/yr and
  K_cal ≈ 2–3 × 10⁻⁵. Against it, the partitioned model gets K about right and D an order of magnitude too high, and AD gets D wrong in the
  other direction. The comparison is now limited by the steady-state assumption, not by the measurements.

## 11. Appendices: all tables, run ids, checksums, Slurm accounting

Generated by `reports/v2/scripts/make_a20_tables.py` from `/home/groups/hilley/lemcvt_results/20261120T185052Z_a20_perron_D0_heads_partB` (also `appendix_tables.md` there).

### E1. Verification of the decision-2 estimator (section 2.1)

| test | r (m) | dx | max error / difference | pass |
|---|---|---|---|---|
| paraboloid+plane | 3 | 1.0000 | 6.79e-14 | yes |
| paraboloid+plane | 5 | 1.0000 | 2.86e-14 | yes |
| paraboloid+plane | 7 | 1.0000 | 1.35e-14 | yes |
| paraboloid+plane | 10 | 1.0000 | 9.77e-15 | yes |
| paraboloid+plane | 15 | 1.0000 | 1.78e-14 | yes |
| paraboloid+plane | 3 | 0.9752 | 2.31e-13 | yes |
| paraboloid+plane | 5 | 0.9752 | 8.18e-14 | yes |
| paraboloid+plane | 7 | 0.9752 | 4.26e-14 | yes |
| paraboloid+plane | 10 | 0.9752 | 2.29e-14 | yes |
| paraboloid+plane | 15 | 0.9752 | 6.22e-15 | yes |
| convolution vs per-cell lstsq (noise 5 cm) | 3 | - | 7.19e-14 | yes |
| convolution vs per-cell lstsq (noise 5 cm) | 5 | - | 2.30e-14 | yes |
| convolution vs per-cell lstsq (noise 5 cm) | 7 | - | 2.59e-14 | yes |
| convolution vs per-cell lstsq (noise 5 cm) | 10 | - | 6.05e-15 | yes |
| convolution vs per-cell lstsq (noise 5 cm) | 15 | - | 5.44e-15 | yes |
| exclusion margin r + 2 cells (r = 7 m, 1 m cells; a cell is  | - | - | - | yes |
| plateau rule on a constructed table (bins 1-4 then a jump) | - | - | - | yes |
| plateau rule: convex fraction 0.5 at the third bin -> 2 bins | - | - | - | yes |
| block bootstrap over identical blocks returns the point valu | - | - | - | yes |

### E2. Per tile: D0_N (decision 1), D0_P (decision 2), D_R and the brackets (sections 2.2-2.3, 3.1)

| tile | D0_N px | >= 500 | road px | seg. px (med) | D0_N | 68 % seg. | 68 % block | D0_N frag20 | a19 seg50 | D0_R | plateau | plateau bins (m2) | stop | boot share | D0_P | 68 % | D0_P no-flat | D0_P D-inf | P/N | D_R | 68 % | D_C from | bracket (D_C lo) | bracket (D0_N lo) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | 2,053 | yes | 418 | 2.0 | 0.003315 | [0.00309, 0.00349] | [0.00304, 0.00353] | 0.00327 | 0.003494 | 0.003587 | yes | [0.0056, 0.1] | bin 8: L_bin/L_plat - 1 = +0.203 | 0.78 | 0.006305 | [0.0052, 0.00751] | 0.006305 | - | 1.90 | 0.01017 | [0.00939, 0.0109] | D0_P | [0.0052, 0.0109] | [0.00309, 0.0109] |
| AP10 | 4,401 | yes | 74 | 2.0 | 0.004743 | [0.00448, 0.00499] | [0.00441, 0.00503] | 0.004207 | 0.004057 | 0.002614 | yes | [0.0056, 0.32] | bin 10: fraction convex 0.589 < 0.6 | 0.53 | 0.006301 | [0.00612, 0.00709] | 0.006301 | 0.006492 | 1.33 | 0.01253 | [0.0121, 0.0129] | D0_P | [0.00612, 0.0129] | [0.00448, 0.0129] |
| AP11 | 1,620 | yes | 95 | 2.0 | 0.005397 | [0.00501, 0.00585] | [0.00467, 0.00608] | 0.004152 | 0.004053 | - | no | [0.01, 0.018] | bin 5: fraction convex 0.549 < 0.6 | 0.10 | - | [0.00655, 0.0078] | - | - | - | 0.01512 | [0.0147, 0.0154] | D0_N (no plateau) | [0.00501, 0.0154] | [0.00501, 0.0154] |
| AP13 | 2,108 | yes | 57 | 2.0 | 0.003543 | [0.00315, 0.00425] | [0.00303, 0.0041] | 0.002602 | 0.002436 | 0.004471 | no | [0.001, 0.0018] | bin 1: n 186 < 200 | 0.00 | - | - | - | - | - | 0.01195 | [0.0113, 0.0123] | D0_N (no plateau) | [0.00315, 0.0123] | [0.00315, 0.0123] |
| AP14 | 4,604 | yes | 578 | 2.0 | 0.00455 | [0.00431, 0.0048] | [0.00424, 0.00486] | 0.003889 | 0.003905 | 0.002095 | no | [0.0056, 0.018] | bin 5: L_bin/L_plat - 1 = +0.429 | 0.02 | - | [0.00629, 0.00889] | - | - | - | 0.0137 | [0.0134, 0.014] | D0_N (no plateau) | [0.00431, 0.014] | [0.00431, 0.014] |
| AP18 | 1,351 | yes | 33 | 2.0 | 0.004569 | [0.0038, 0.0052] | [0.00387, 0.00522] | 0.00362 | 0.003852 | 0.001885 | no | [0.001, 0.0018] | bin 1: L_bin/L_plat - 1 = +0.285 | 0.00 | - | - | - | - | - | 0.01471 | [0.0139, 0.0153] | D0_N (no plateau) | [0.0038, 0.0153] | [0.0038, 0.0153] |
| AP19 | 3,119 | yes | 115 | 2.0 | 0.004744 | [0.00448, 0.00503] | [0.00426, 0.00522] | 0.004306 | 0.001302 | 0.006118 | yes | [0.01, 0.056] | bin 7: L_bin/L_plat - 1 = +0.238 | 0.40 | 0.008543 | [0.00711, 0.00937] | 0.008543 | 0.008362 | 1.80 | 0.01225 | [0.0119, 0.0126] | D0_P | [0.00711, 0.0126] | [0.00448, 0.0126] |
| AP1_3dep | 2,373 | yes | 203 | 2.0 | 0.003776 | [0.00353, 0.00414] | [0.00351, 0.00419] | 0.002904 | 0.002625 | 0.005507 | no | [0.0056, 0.01] | bin 4: L_bin/L_plat - 1 = +0.285 | 0.12 | - | [0.00944, 0.0123] | - | - | - | 0.01233 | [0.012, 0.013] | D0_N (no plateau) | [0.00353, 0.013] | [0.00353, 0.013] |
| AP2 | 1,429 | yes | 19 | 2.0 | 0.003691 | [0.00343, 0.00401] | [0.00341, 0.004] | 0.003296 | 0.003222 | 0.003218 | no | [0.0056, 0.01] | bin 4: fraction convex 0.533 < 0.6 | 0.01 | - | [0.0086, 0.0103] | - | - | - | 0.01233 | [0.0116, 0.013] | D0_N (no plateau) | [0.00343, 0.013] | [0.00343, 0.013] |
| AP20 | 5,405 | yes | 324 | 2.0 | 0.004449 | [0.00421, 0.00471] | [0.00407, 0.00491] | 0.004198 | 0.004023 | 0.0054 | yes | [0.0056, 0.056] | bin 7: L_bin/L_plat - 1 = +0.306 | 0.52 | 0.00978 | [0.00834, 0.0104] | 0.00978 | 0.009476 | 2.20 | 0.01274 | [0.0123, 0.013] | D0_P | [0.00834, 0.013] | [0.00421, 0.013] |
| AP23 | 1,309 | yes | 3 | 2.0 | 0.003354 | [0.00308, 0.00381] | [0.00315, 0.00366] | 0.003135 | 0.003018 | 0.003661 | no | [0.001, 0.0018] | bin 1: n 82 < 200 | 0.00 | - | - | - | - | - | 0.01121 | [0.0109, 0.0116] | D0_N (no plateau) | [0.00308, 0.0116] | [0.00308, 0.0116] |
| AP24 | 2,212 | yes | 828 | 2.0 | 0.005254 | [0.0047, 0.00592] | [0.00461, 0.00595] | 0.003365 | 0.003462 | 0.005939 | yes | [0.0056, 0.032] | bin 6: fraction convex 0.577 < 0.6 | 0.41 | 0.007657 | [0.00665, 0.0079] | 0.007657 | 0.007412 | 1.46 | 0.01404 | [0.0134, 0.0147] | D0_P | [0.00665, 0.0147] | [0.0047, 0.0147] |
| AP25 | 718 | yes | 568 | 2.0 | 0.005379 | [0.00347, 0.00581] | [0.00368, 0.00585] | 0.005602 | 0.005833 | 0.003859 | no | [0.001, 0.0018] | bin 1: fraction convex 0.427 < 0.6 | 0.00 | - | - | - | - | - | 0.01363 | [0.0103, 0.0159] | D0_N (no plateau) | [0.00347, 0.0159] | [0.00347, 0.0159] |
| AP27 | 3,351 | yes | 51 | 2.0 | 0.003693 | [0.00339, 0.00407] | [0.00335, 0.00413] | 0.003031 | 0.002846 | 0.001319 | no | [0.0056, 0.01] | bin 4: fraction convex 0.573 < 0.6 | 0.01 | - | [0.00863, 0.00992] | - | - | - | 0.01224 | [0.012, 0.0125] | D0_N (no plateau) | [0.00339, 0.0125] | [0.00339, 0.0125] |
| AP28 | 3,349 | yes | 927 | 2.0 | 0.004522 | [0.0043, 0.00478] | [0.00427, 0.00482] | 0.004292 | 0.004127 | 0.002704 | yes | [0.0032, 0.018] | bin 5: L_bin/L_plat - 1 = +0.249 | 0.41 | 0.01364 | [0.0101, 0.0142] | 0.01364 | 0.01343 | 3.02 | 0.01199 | [0.0115, 0.0125] | D0_P | [0.0101, 0.0125] | [0.0043, 0.0125] |
| AP29 | 2,416 | yes | 46 | 2.0 | 0.005263 | [0.0048, 0.00574] | [0.00459, 0.00604] | 0.004517 | 0.002773 | 0.006297 | no | [0.0032, 0.0056] | bin 3: fraction convex 0.534 < 0.6 | 0.00 | - | - | - | - | - | 0.01357 | [0.013, 0.014] | D0_N (no plateau) | [0.0048, 0.014] | [0.0048, 0.014] |
| AP2_3dep | 1,387 | yes | 131 | 2.0 | 0.005961 | [0.00546, 0.00662] | [0.00507, 0.00715] | 0.004149 | 0.004016 | 0.002742 | no | [0.0056, 0.01] | bin 4: fraction convex 0.429 < 0.6 | 0.00 | - | - | - | - | - | 0.01548 | [0.015, 0.0159] | D0_N (no plateau) | [0.00546, 0.0159] | [0.00546, 0.0159] |
| AP3 | 2,448 | yes | 63 | 2.0 | 0.003001 | [0.00278, 0.00317] | [0.00275, 0.00322] | 0.00254 | 0.002643 | 0.002155 | yes | [0.0032, 0.056] | bin 7: fraction convex 0.553 < 0.6 | 0.62 | 0.008787 | [0.00862, 0.00964] | 0.008787 | - | 2.93 | 0.0114 | [0.011, 0.0117] | D0_P | [0.00862, 0.0117] | [0.00278, 0.0117] |
| AP3_3dep | 3,123 | yes | 177 | 2.0 | 0.00433 | [0.00418, 0.00454] | [0.00406, 0.0047] | 0.00353 | 0.003918 | 0.004372 | no | [0.0032, 0.0056] | bin 3: fraction convex 0.531 < 0.6 | 0.00 | - | - | - | - | - | 0.01167 | [0.0113, 0.012] | D0_N (no plateau) | [0.00418, 0.012] | [0.00418, 0.012] |
| AP4 | 3,636 | yes | 208 | 3.0 | 0.003186 | [0.00302, 0.0034] | [0.00302, 0.00337] | 0.003029 | 0.002994 | 0.003629 | yes | [0.0032, 0.018] | bin 5: L_bin/L_plat - 1 = +0.209 | 0.60 | 0.008919 | [0.00713, 0.0095] | 0.008919 | - | 2.80 | 0.01231 | [0.0118, 0.0128] | D0_P | [0.00713, 0.0128] | [0.00302, 0.0128] |
| AP4_3dep | 4,286 | yes | 351 | 2.0 | 0.004296 | [0.00399, 0.00461] | [0.00387, 0.00466] | 0.003718 | 0.003564 | 0.00536 | yes | [0.0032, 0.018] | bin 5: L_bin/L_plat - 1 = +0.546 | 0.26 | 0.01407 | [0.011, 0.0147] | 0.01407 | 0.01413 | 3.27 | 0.01284 | [0.0123, 0.0132] | D0_P | [0.011, 0.0132] | [0.00399, 0.0132] |
| AP5 | 2,326 | yes | 3 | 4.0 | 0.003597 | [0.00332, 0.00397] | [0.0033, 0.00397] | 0.003593 | 0.003481 | 0.002921 | no | [0.001, 0.0018] | bin 1: n 61 < 200 | 0.25 | - | [0.00612, 0.00778] | 0.006555 | - | - | 0.0124 | [0.012, 0.0128] | D0_N (no plateau) | [0.00332, 0.0128] | [0.00332, 0.0128] |
| AP5_3dep | 2,895 | yes | 107 | 2.0 | 0.005435 | [0.00503, 0.00583] | [0.00491, 0.00587] | 0.005094 | 0.004714 | 0.005144 | no | [0.001, 0.0018] | bin 1: n 65 < 200 | 0.12 | - | [0.011, 0.0134] | - | - | - | 0.01306 | [0.0125, 0.0135] | D0_N (no plateau) | [0.00503, 0.0135] | [0.00503, 0.0135] |
| AP6 | 2,468 | yes | 370 | 1.0 | 0.004219 | [0.00384, 0.00481] | [0.00381, 0.00491] | 0.003553 | 0.003312 | 0.005952 | no | [0.001, 0.0018] | bin 1: n 105 < 200 | 0.11 | - | [0.0099, 0.0137] | 0.0124 | - | - | 0.01069 | [0.0103, 0.0112] | D0_N (no plateau) | [0.00384, 0.0112] | [0.00384, 0.0112] |
| AP7 | 1,737 | yes | 311 | 2.0 | 0.004059 | [0.00374, 0.00443] | [0.00354, 0.00473] | 0.003817 | 0.004683 | 0.005239 | no | [0.001, 0.0018] | bin 1: n 123 < 200 | 0.00 | - | - | - | - | - | 0.01204 | [0.0118, 0.0123] | D0_N (no plateau) | [0.00374, 0.0123] | [0.00374, 0.0123] |
| AP8 | 2,045 | yes | 40 | 2.0 | 0.003659 | [0.00329, 0.0041] | [0.00322, 0.00418] | 0.002284 | 0.001954 | 0.006024 | no | [0.001, 0.0018] | bin 1: n 75 < 200 | 0.01 | - | [0.0124, 0.0156] | - | - | - | 0.01318 | [0.0128, 0.0135] | D0_N (no plateau) | [0.00329, 0.0135] | [0.00329, 0.0135] |
| AP9 | 4,078 | yes | 696 | 2.0 | 0.00436 | [0.0041, 0.00468] | [0.004, 0.00488] | 0.003427 | 0.00333 | 0.001847 | no | [0.001, 0.0018] | bin 1: n 47 < 200 | 0.00 | - | - | - | - | - | 0.0126 | [0.0122, 0.013] | D0_N (no plateau) | [0.0041, 0.013] | [0.0041, 0.013] |
| GM1 | 4,858 | yes | 0 | 3.0 | 0.0105 | [0.01, 0.0109] | [0.00979, 0.0111] | 0.01062 | 0.01087 | 0.009001 | yes | [0.0056, 5.6] | bin 15: fraction convex 0.576 < 0.6 | 0.68 | 0.01303 | [0.0127, 0.0137] | 0.01303 | - | 1.24 | 0.01968 | [0.0191, 0.0205] | D0_P | [0.0127, 0.0205] | [0.01, 0.0205] |
| GM11 | 3,138 | yes | 92 | 3.0 | 0.01712 | [0.0159, 0.0186] | [0.0154, 0.0199] | 0.01648 | 0.01665 | 0.01831 | no | [0.001, 0.0018] | bin 1: n 131 < 200 | 0.02 | - | [0.0237, 0.0339] | - | - | - | 0.03633 | [0.0343, 0.0381] | D0_N (no plateau) | [0.0159, 0.0381] | [0.0159, 0.0381] |
| GM12 | 4,932 | yes | 176 | 2.0 | 0.01052 | [0.0097, 0.0115] | [0.00943, 0.0117] | 0.01072 | 0.01096 | 0.01144 | yes | [0.0056, 0.32] | bin 10: fraction convex 0.561 < 0.6 | 0.81 | 0.01927 | [0.0173, 0.021] | 0.01927 | 0.01618 | 1.83 | 0.02618 | [0.0252, 0.0273] | D0_P | [0.0173, 0.0273] | [0.0097, 0.0273] |
| GM13 | 3,601 | yes | 19 | 3.0 | 0.02563 | [0.0244, 0.0272] | [0.0235, 0.0281] | 0.0233 | 0.02777 | 0.02885 | no | [0.0056, 0.018] | bin 5: L_bin/L_plat - 1 = -0.230 | 0.38 | - | [0.0284, 0.0351] | - | 0.02835 | - | 0.03549 | [0.034, 0.0372] | D0_N (no plateau) | [0.0244, 0.0372] | [0.0244, 0.0372] |
| GM14 | 4,474 | yes | 0 | 2.0 | 0.008985 | [0.00829, 0.00984] | [0.00792, 0.00999] | 0.007491 | 0.007375 | 0.009355 | no | [0.001, 0.0018] | bin 1: n 53 < 200 | 0.00 | - | - | - | - | - | 0.0327 | [0.0303, 0.0348] | D0_N (no plateau) | [0.00829, 0.0348] | [0.00829, 0.0348] |
| GM15 | 3,388 | yes | 626 | 3.0 | 0.008563 | [0.00812, 0.00907] | [0.00779, 0.00935] | 0.007472 | 0.0073 | 0.00898 | no | [0.001, 0.0018] | bin 1: n 38 < 200 | 0.01 | - | [0.018, 0.0289] | - | - | - | 0.02356 | [0.0226, 0.0249] | D0_N (no plateau) | [0.00812, 0.0249] | [0.00812, 0.0249] |
| GM16 | 4,062 | yes | 0 | 3.0 | 0.01282 | [0.0117, 0.0142] | [0.0107, 0.0159] | 0.01182 | 0.01181 | 0.01218 | yes | [0.0056, 0.18] | bin 9: L_bin/L_plat - 1 = +0.394 | 0.44 | 0.02506 | [0.0227, 0.0292] | 0.02506 | 0.02493 | 1.95 | 0.0244 | [0.0235, 0.0257] | D0_P | [0.0227, 0.0257] | [0.0117, 0.0257] |
| GM17 | 2,511 | yes | 153 | 3.0 | 0.01563 | [0.014, 0.0179] | [0.0138, 0.0184] | 0.01405 | 0.01374 | 0.01552 | yes | [0.0032, 0.18] | bin 9: fraction convex 0.588 < 0.6 | 0.76 | 0.02566 | [0.0231, 0.0275] | 0.02566 | 0.02332 | 1.64 | 0.03265 | [0.0301, 0.0354] | D0_P | [0.0231, 0.0354] | [0.014, 0.0354] |
| GM18 | 1,733 | yes | 171 | 2.0 | 0.006056 | [0.00541, 0.00697] | [0.00509, 0.00819] | 0.004492 | 0.004236 | 0.005326 | no | [0.01, 0.018] | bin 5: fraction convex 0.489 < 0.6 | 0.00 | - | - | - | - | - | 0.0216 | [0.0204, 0.0226] | D0_N (no plateau) | [0.00541, 0.0226] | [0.00541, 0.0226] |
| GM19 | 3,309 | yes | 0 | 2.0 | 0.01218 | [0.0113, 0.0132] | [0.0108, 0.0139] | 0.009746 | 0.009344 | 0.01241 | no | [0.01, 0.018] | bin 5: L_bin/L_plat - 1 = -0.218 | 0.04 | - | [0.0204, 0.0387] | - | - | - | 0.033 | [0.0314, 0.0346] | D0_N (no plateau) | [0.0113, 0.0346] | [0.0113, 0.0346] |
| GM1_fema | 5,781 | yes | 0 | 3.0 | 0.0102 | [0.00989, 0.0106] | [0.00976, 0.0107] | 0.01011 | 0.009944 | 0.01025 | yes | [0.0056, 10] | bin 16: fraction convex 0.589 < 0.6 | 0.91 | 0.01364 | [0.0132, 0.0141] | 0.01364 | 0.01299 | 1.34 | 0.02116 | [0.0207, 0.0217] | D0_P | [0.0132, 0.0217] | [0.00989, 0.0217] |
| GM2 | 3,453 | yes | 1,070 | 3.0 | 0.01315 | [0.0123, 0.0141] | [0.0118, 0.0147] | 0.0134 | 0.01461 | 0.00977 | yes | [0.0056, 5.6] | bin 15: fraction convex 0.579 < 0.6 | 0.94 | 0.01527 | [0.0148, 0.0181] | 0.01527 | - | 1.16 | 0.02252 | [0.0213, 0.0242] | D0_P | [0.0148, 0.0242] | [0.0123, 0.0242] |
| GM20 | 3,631 | yes | 631 | 3.0 | 0.02042 | [0.0188, 0.0231] | [0.0173, 0.0249] | 0.01526 | 0.01377 | 0.01825 | no | [0.0056, 0.018] | bin 5: L_bin/L_plat - 1 = -0.357 | 0.05 | - | [0.021, 0.0242] | - | - | - | 0.04086 | [0.0378, 0.0448] | D0_N (no plateau) | [0.0188, 0.0448] | [0.0188, 0.0448] |
| GM21 | 4,756 | yes | 184 | 2.0 | 0.01083 | [0.0103, 0.0115] | [0.00991, 0.0117] | 0.009586 | 0.009341 | 0.01193 | no | [0.001, 0.0018] | bin 1: n 38 < 200 | 0.25 | - | [0.0179, 0.0195] | 0.0189 | - | - | 0.02856 | [0.0272, 0.0298] | D0_N (no plateau) | [0.0103, 0.0298] | [0.0103, 0.0298] |
| GM22 | 1,411 | yes | 0 | 3.0 | 0.01535 | [0.0146, 0.0163] | [0.0143, 0.0164] | 0.01547 | 0.01413 | 0.01609 | yes | [0.0056, 0.1] | bin 8: L_bin/L_plat - 1 = +0.202 | 0.75 | 0.02681 | [0.022, 0.0293] | 0.02681 | 0.02568 | 1.75 | 0.03535 | [0.0341, 0.0369] | D0_P | [0.022, 0.0369] | [0.0146, 0.0369] |
| GM24 | 2,962 | yes | 19 | 2.5 | 0.03062 | [0.0286, 0.0331] | [0.0272, 0.0351] | 0.02838 | 0.02269 | 0.03127 | yes | [0.0056, 0.18] | bin 9: L_bin/L_plat - 1 = +0.448 | 0.84 | 0.04256 | [0.0404, 0.0468] | 0.04256 | 0.0446 | 1.39 | 0.05395 | [0.0513, 0.057] | D0_P | [0.0404, 0.057] | [0.0286, 0.057] |
| GM25 | 5,525 | yes | 0 | 3.0 | 0.01011 | [0.00972, 0.0106] | [0.00954, 0.0108] | 0.009162 | 0.007829 | 0.01061 | no | [0.018, 0.032] | bin 6: fraction convex 0.524 < 0.6 | 0.00 | - | - | - | - | - | 0.0256 | [0.0246, 0.0263] | D0_N (no plateau) | [0.00972, 0.0263] | [0.00972, 0.0263] |
| GM26 | 1,251 | yes | 171 | 2.0 | 0.01873 | [0.0162, 0.0212] | [0.0146, 0.0221] | 0.02202 | 0.02111 | 0.0174 | no | [0.0018, 0.0032] | bin 2: L_bin/L_plat - 1 = +0.202 | 0.18 | - | [0.0387, 0.0541] | - | - | - | 0.03117 | [0.0267, 0.0339] | D0_N (no plateau) | [0.0162, 0.0339] | [0.0162, 0.0339] |
| GM27 | 4,317 | yes | 0 | 3.0 | 0.007144 | [0.00695, 0.00736] | [0.00686, 0.00748] | 0.006653 | 0.006446 | 0.00728 | no | [0.01, 0.018] | bin 5: L_bin/L_plat - 1 = -0.318 | 0.00 | - | - | - | - | - | 0.02011 | [0.0196, 0.0205] | D0_N (no plateau) | [0.00695, 0.0205] | [0.00695, 0.0205] |
| GM28 | 4,068 | yes | 8 | 2.0 | 0.01423 | [0.0124, 0.0157] | [0.0119, 0.0165] | 0.01451 | 0.0157 | 0.01462 | no | [0.001, 0.0018] | bin 1: n 42 < 200 | 0.27 | - | [0.0283, 0.0344] | 0.03254 | - | - | 0.03618 | [0.0309, 0.0395] | D0_N (no plateau) | [0.0124, 0.0395] | [0.0124, 0.0395] |
| GM29 | 5,047 | yes | 80 | 2.0 | 0.01576 | [0.0146, 0.0166] | [0.014, 0.0171] | 0.01397 | 0.01337 | 0.01455 | yes | [0.0056, 0.32] | bin 10: fraction convex 0.582 < 0.6 | 0.64 | 0.02347 | [0.021, 0.0247] | 0.02347 | 0.01984 | 1.49 | 0.02835 | [0.0264, 0.0302] | D0_P | [0.021, 0.0302] | [0.0146, 0.0302] |
| GM2_fema | 3,278 | yes | 1,048 | 2.0 | 0.01281 | [0.012, 0.0138] | [0.0115, 0.0142] | 0.01238 | 0.01216 | 0.01251 | yes | [0.0056, 0.32] | bin 10: fraction convex 0.599 < 0.6 | 0.97 | 0.01619 | [0.0156, 0.0178] | 0.01619 | 0.015 | 1.26 | 0.02335 | [0.0221, 0.0248] | D0_P | [0.0156, 0.0248] | [0.012, 0.0248] |
| GM3 | 5,713 | yes | 932 | 4.0 | 0.01864 | [0.0171, 0.0198] | [0.0167, 0.0206] | 0.01869 | 0.0182 | 0.01012 | no | [0.0018, 0.0032] | bin 2: L_bin/L_plat - 1 = +0.277 | 0.24 | - | [0.032, 0.0447] | - | - | - | 0.03261 | [0.0308, 0.0354] | D0_N (no plateau) | [0.0171, 0.0354] | [0.0171, 0.0354] |
| GM30 | 2,960 | yes | 0 | 2.0 | 0.01034 | [0.00949, 0.012] | [0.00942, 0.0124] | 0.009642 | 0.008996 | 0.01114 | no | [0.0056, 0.018] | bin 5: L_bin/L_plat - 1 = -0.350 | 0.07 | - | [0.0228, 0.0328] | - | - | - | 0.028 | [0.0265, 0.0298] | D0_N (no plateau) | [0.00949, 0.0298] | [0.00949, 0.0298] |
| GM3_fema | 5,265 | yes | 893 | 3.0 | 0.01913 | [0.0178, 0.0209] | [0.017, 0.0218] | 0.01824 | 0.01594 | 0.01545 | no | [0.0018, 0.0032] | bin 2: L_bin/L_plat - 1 = +0.203 | 0.42 | - | [0.0321, 0.0449] | - | - | - | 0.03557 | [0.0328, 0.0387] | D0_N (no plateau) | [0.0178, 0.0387] | [0.0178, 0.0387] |
| GM4 | 3,019 | yes | 26 | 3.0 | 0.01136 | [0.0102, 0.0126] | [0.00959, 0.0135] | 0.01074 | 0.01261 | 0.009855 | no | [0.001, 0.0018] | bin 1: n 54 < 200 | 0.02 | - | [0.0177, 0.0371] | - | - | - | 0.02064 | [0.0198, 0.0214] | D0_N (no plateau) | [0.0102, 0.0214] | [0.0102, 0.0214] |
| GM4_fema | 3,204 | yes | 5 | 2.0 | 0.01097 | [0.01, 0.0122] | [0.00955, 0.0129] | 0.01067 | 0.01025 | 0.01178 | no | [0.001, 0.0018] | bin 1: n 78 < 200 | 0.01 | - | [0.0153, 0.0469] | - | - | - | 0.0219 | [0.0211, 0.0225] | D0_N (no plateau) | [0.01, 0.0225] | [0.01, 0.0225] |
| GM5 | 915 | yes | 0 | 2.0 | 0.007921 | [0.0076, 0.00944] | [0.00753, 0.0089] | 0.007427 | 0.007182 | 0.007589 | yes | [0.01, 0.056] | bin 7: fraction convex 0.594 < 0.6 | 0.28 | 0.02211 | [0.0192, 0.0233] | 0.02211 | - | 2.79 | 0.01995 | [0.0189, 0.0229] | D0_P | [0.0192, 0.0229] | [0.0076, 0.0229] |
| GM5_fema | 1,145 | yes | 13 | 2.0 | 0.009431 | [0.00838, 0.0108] | [0.00853, 0.0107] | 0.007955 | 0.007153 | 0.008725 | no | [0.01, 0.018] | bin 5: L_bin/L_plat - 1 = -0.242 | 0.19 | - | [0.0198, 0.0252] | - | - | - | 0.02295 | [0.0218, 0.0246] | D0_N (no plateau) | [0.00838, 0.0246] | [0.00838, 0.0246] |
| GM6 | 4,074 | yes | 14 | 3.0 | 0.01696 | [0.0155, 0.0184] | [0.0143, 0.0192] | 0.01892 | 0.0175 | 0.01094 | yes | [0.0056, 0.032] | bin 6: L_bin/L_plat - 1 = +0.244 | 0.45 | 0.03973 | [0.0267, 0.0431] | 0.03973 | - | 2.34 | 0.02542 | [0.0241, 0.027] | D0_P | [0.0267, 0.027] | [0.0155, 0.027] |
| GM6_fema | 3,474 | yes | 1 | 2.0 | 0.01692 | [0.0153, 0.0185] | [0.0141, 0.0195] | 0.01588 | 0.01234 | 0.01625 | yes | [0.0056, 0.032] | bin 6: L_bin/L_plat - 1 = +0.295 | 0.81 | 0.04189 | [0.0277, 0.0539] | 0.04189 | 0.04055 | 2.48 | 0.02498 | [0.0237, 0.0267] | D0_P | [0.0277, 0.0267] | [0.0153, 0.0267] |
| GM7 | 3,394 | yes | 0 | 3.0 | 0.02863 | [0.0276, 0.0298] | [0.0269, 0.0308] | 0.0277 | 0.02564 | 0.02203 | yes | [0.0056, 0.32] | bin 10: L_bin/L_plat - 1 = +0.230 | 0.98 | 0.0399 | [0.0366, 0.0421] | 0.0399 | - | 1.39 | 0.05665 | [0.0537, 0.0585] | D0_P | [0.0366, 0.0585] | [0.0276, 0.0585] |
| GM7_fema | 3,197 | yes | 3 | 3.0 | 0.03294 | [0.0309, 0.035] | [0.0303, 0.0361] | 0.03076 | 0.02775 | 0.03544 | yes | [0.0056, 0.18] | bin 9: L_bin/L_plat - 1 = +0.238 | 0.82 | 0.04468 | [0.0407, 0.0463] | 0.04468 | 0.03954 | 1.36 | 0.05991 | [0.0572, 0.062] | D0_P | [0.0407, 0.062] | [0.0309, 0.062] |
| GM8 | 4,537 | yes | 0 | 4.0 | 0.01468 | [0.0138, 0.0157] | [0.0136, 0.0159] | 0.01406 | 0.01385 | 0.01523 | yes | [0.0056, 0.32] | bin 10: fraction convex 0.586 < 0.6 | 0.38 | 0.02041 | [0.0195, 0.0214] | 0.02041 | 0.01835 | 1.39 | 0.0309 | [0.0295, 0.0323] | D0_P | [0.0195, 0.0323] | [0.0138, 0.0323] |
| GM9 | 2,337 | yes | 490 | 2.0 | 0.02146 | [0.0202, 0.0225] | [0.0195, 0.0232] | 0.02139 | 0.02405 | 0.02036 | yes | [0.0056, 0.32] | bin 10: fraction convex 0.576 < 0.6 | 0.54 | 0.02469 | [0.0236, 0.0278] | 0.02469 | 0.02337 | 1.15 | 0.03329 | [0.0306, 0.0351] | D0_P | [0.0236, 0.0351] | [0.0202, 0.0351] |
| TTV1 | 3,820 | yes | 1,553 | 2.0 | 0.004831 | [0.00454, 0.00513] | [0.00423, 0.00536] | 0.005534 | 0.00608 | 0.002905 | yes | [0.0056, 0.032] | bin 6: L_bin/L_plat - 1 = +0.363 | 0.51 | 0.01559 | [0.0133, 0.0167] | 0.01559 | - | 3.23 | 0.01502 | [0.0142, 0.0158] | D0_P | [0.0133, 0.0158] | [0.00454, 0.0158] |
| TTV11 | 4,889 | yes | 1,457 | 2.0 | 0.004611 | [0.00437, 0.00484] | [0.00423, 0.00495] | 0.004407 | 0.00435 | 0.002312 | yes | [0.0056, 0.1] | bin 8: L_bin/L_plat - 1 = +0.218 | 0.59 | 0.007539 | [0.00677, 0.01] | 0.007539 | - | 1.63 | 0.01178 | [0.0113, 0.0123] | D0_P | [0.00677, 0.0123] | [0.00437, 0.0123] |
| TTV15 | 4,014 | yes | 1,231 | 2.0 | 0.005479 | [0.00513, 0.00579] | [0.00475, 0.00624] | 0.005738 | 0.005959 | 0.001506 | yes | [0.01, 0.1] | bin 8: L_bin/L_plat - 1 = +0.246 | 0.79 | 0.01152 | [0.00931, 0.0138] | 0.01152 | - | 2.10 | 0.01315 | [0.0125, 0.0139] | D0_P | [0.00931, 0.0139] | [0.00513, 0.0139] |
| TTV2 | 2,919 | yes | 348 | 2.0 | 0.005161 | [0.00487, 0.00555] | [0.00468, 0.00582] | 0.004761 | 0.004931 | 0.004461 | yes | [0.01, 0.056] | bin 7: L_bin/L_plat - 1 = +0.285 | 0.70 | 0.01355 | [0.0109, 0.0158] | 0.01355 | - | 2.63 | 0.01201 | [0.0115, 0.0125] | D0_P | [0.0109, 0.0125] | [0.00487, 0.0125] |
| TTV3 | 1,052 | yes | 298 | 2.0 | 0.005247 | [0.00478, 0.00565] | [0.00444, 0.00595] | 0.005331 | 0.006404 | 0.003262 | no | [0.0056, 0.01] | bin 4: fraction convex 0.576 < 0.6 | 0.00 | - | - | - | - | - | 0.01759 | [0.0169, 0.0183] | D0_N (no plateau) | [0.00478, 0.0183] | [0.00478, 0.0183] |
| TTV4 | 2,648 | yes | 987 | 2.0 | 0.006232 | [0.00591, 0.00663] | [0.00577, 0.00676] | 0.00593 | 0.005694 | 0.0008796 | no | [0.018, 0.032] | bin 6: L_bin/L_plat - 1 = +0.326 | 0.12 | - | [0.00892, 0.0116] | - | - | - | 0.01306 | [0.0126, 0.0137] | D0_N (no plateau) | [0.00591, 0.0137] | [0.00591, 0.0137] |
| TTV5 | 3,967 | yes | 387 | 2.0 | 0.003457 | [0.0033, 0.00363] | [0.00324, 0.00368] | 0.003256 | 0.00308 | 0.002498 | no | [0.01, 0.032] | bin 6: fraction convex 0.545 < 0.6 | 0.09 | - | [0.00603, 0.008] | - | - | - | 0.01179 | [0.0114, 0.0123] | D0_N (no plateau) | [0.0033, 0.0123] | [0.0033, 0.0123] |
| TTV6 | 4,927 | yes | 672 | 2.0 | 0.005546 | [0.00525, 0.00579] | [0.00514, 0.00593] | 0.005448 | 0.005668 | 0.001993 | no | [0.0032, 0.0056] | bin 3: fraction convex 0.533 < 0.6 | 0.26 | - | [0.00996, 0.0113] | - | - | - | 0.01505 | [0.0146, 0.0155] | D0_N (no plateau) | [0.00525, 0.0155] | [0.00525, 0.0155] |
| TTV7 | 4,476 | yes | 485 | 2.0 | 0.004733 | [0.00452, 0.00496] | [0.00437, 0.00518] | 0.00378 | 0.002831 | 0.003098 | yes | [0.01, 0.18] | bin 9: fraction convex 0.583 < 0.6 | 0.90 | 0.008795 | [0.00815, 0.00944] | 0.008795 | - | 1.86 | 0.01293 | [0.0124, 0.0135] | D0_P | [0.00815, 0.0135] | [0.00452, 0.0135] |
| TTV8 | 5,165 | yes | 992 | 2.0 | 0.004383 | [0.00424, 0.00454] | [0.00417, 0.00461] | 0.00405 | 0.004088 | 0.003984 | yes | [0.01, 0.32] | bin 10: fraction convex 0.583 < 0.6 | 0.90 | 0.006986 | [0.00674, 0.00764] | 0.006986 | - | 1.59 | 0.01315 | [0.0126, 0.0136] | D0_P | [0.00674, 0.0136] | [0.00424, 0.0136] |
| TTV9 | 1,903 | yes | 615 | 2.0 | 0.003632 | [0.00329, 0.00402] | [0.00333, 0.00403] | 0.002777 | 0.002582 | 0.004202 | no | [0.0056, 0.01] | bin 4: L_bin/L_plat - 1 = +0.409 | 0.13 | - | [0.00518, 0.00623] | - | - | - | 0.01018 | [0.00982, 0.0108] | D0_N (no plateau) | [0.00329, 0.0108] | [0.00329, 0.0108] |

### E3. D0_P against the window (decision 4): site medians of the primary tiles

| site | r 3 | r 5 | r 7 | r 10 | r 15 | fixed <= 1 m2 (r 7) | fixed <= 3 m2 | D-inf (r 7) | D0_N L 3 | D0_N L 11 | D0_N L 51 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GM | 0.01491 | 0.0217 | 0.02487 | 0.02317 | 0.02494 | 0.01898 | 0.01872 | 0.02337 | 0.006087 | 0.01549 | 0.01578 |
| AP | 0.003851 | 0.006731 | 0.009161 | 0.0118 | 0.01051 | 0.006076 | 0.006135 | 0.008919 | 0.001201 | 0.004522 | 0.006998 |
| TTV | 0.004591 | 0.00799 | 0.01016 | 0.01492 | 0.01302 | 0.006629 | 0.006345 | - | 0.001035 | 0.004831 | 0.007815 |

### E4. Pooled site curves (section 2.5) and rules R1, R13, R11P

| site | group | r | plateau | bins (m2) | lap_h | stop | boot share | ratio to Perron | fixed <= 1 | fixed <= 3 | no-flat plateau |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GM | primary | r3 | no | [0.001, 0.0018] | -0.005996 | bin 1: L_bin/L_plat - 1 = +0.212 | 0.485 | - | 1.19 | 1.21 | no |
| GM | primary | r5 | True | [0.001, 0.18] | -0.00644 | bin 9: L_bin/L_plat - 1 = +0.321 | 0.770 | 0.55 | 0.74 | 0.76 | True |
| GM | primary | r7 | no | [0.001, 0.0018] | -0.003315 | bin 1: L_bin/L_plat - 1 = +0.226 | 0.306 | - | 0.63 | 0.65 | no |
| GM | primary | r10 | True | [0.001, 0.18] | -0.005325 | bin 9: L_bin/L_plat - 1 = +0.265 | 0.885 | 0.45 | 0.56 | 0.58 | True |
| GM | primary | r15 | True | [0.001, 0.18] | -0.005232 | bin 9: L_bin/L_plat - 1 = +0.221 | 0.759 | 0.44 | 0.52 | 0.53 | True |
| GM | primary | r7_dinf | no | [0.001, 0.0018] | -0.002627 | bin 1: L_bin/L_plat - 1 = +0.567 | 0.051 | - | 0.66 | 0.68 | no |
| GM | ncalm | r3 | no | [0.001, 0.0018] | -0.006809 | bin 1: L_bin/L_plat - 1 = +0.288 | 0.286 | - | 1.26 | 1.27 | no |
| GM | ncalm | r5 | True | [0.001, 0.1] | -0.006281 | bin 8: L_bin/L_plat - 1 = +0.206 | 0.648 | 0.53 | 0.76 | 0.81 | True |
| GM | ncalm | r7 | True | [0.001, 0.1] | -0.005247 | bin 8: L_bin/L_plat - 1 = +0.219 | 0.641 | 0.44 | 0.64 | 0.69 | True |
| GM | ncalm | r10 | True | [0.001, 0.1] | -0.004803 | bin 8: L_bin/L_plat - 1 = +0.201 | 0.817 | 0.41 | 0.57 | 0.62 | True |
| GM | ncalm | r15 | True | [0.001, 0.18] | -0.0051 | bin 9: L_bin/L_plat - 1 = +0.301 | 0.627 | 0.43 | 0.55 | 0.58 | True |
| AP | primary | r3 | no | [0.001, 0.0018] | -0.004208 | bin 1: L_bin/L_plat - 1 = +0.515 | 0.001 | - | 3.77 | 3.66 | no |
| AP | primary | r5 | no | [0.001, 0.0018] | -0.001817 | bin 1: fraction convex 0.558 < 0.6 | 0.001 | - | 2.12 | 2.10 | no |
| AP | primary | r7 | no | [0.001, 0.0018] | -0.00167 | bin 1: fraction convex 0.532 < 0.6 | 0.000 | - | 1.63 | 1.63 | no |
| AP | primary | r10 | no | [0.001, 0.0018] | -0.001781 | bin 1: fraction convex 0.522 < 0.6 | 0.005 | - | 1.21 | 1.22 | no |
| AP | primary | r15 | no | [0.001, 0.0018] | -0.001572 | bin 1: fraction convex 0.526 < 0.6 | 0.013 | - | 0.90 | 0.90 | no |
| AP | primary | r7_dinf | no | [0.001, 0.0018] | -0.001365 | bin 1: fraction convex 0.461 < 0.6 | 0.000 | - | 1.97 | 1.87 | no |
| AP | pamap | r3 | no | [0.001, 0.0018] | -0.007523 | bin 1: fraction convex 0.593 < 0.6 | 0.076 | - | 4.89 | 4.17 | no |
| AP | pamap | r5 | no | [0.001, 0.0018] | -0.003941 | bin 1: L_bin/L_plat - 1 = +0.569 | 0.042 | - | 2.74 | 2.53 | no |
| AP | pamap | r7 | no | [0.001, 0.0018] | -0.002666 | bin 1: L_bin/L_plat - 1 = +1.299 | 0.248 | - | 1.90 | 1.82 | no |
| AP | pamap | r10 | no | [0.001, 0.0018] | -0.003644 | bin 1: fraction convex 0.579 < 0.6 | 0.296 | - | 1.30 | 1.28 | True |
| AP | pamap | r15 | no | [0.0032, 0.0056] | -0.002593 | bin 3: fraction convex 0.570 < 0.6 | 0.042 | - | 0.93 | 0.93 | no |
| TTV | primary | r3 | no | [0.001, 0.0032] | -0.006155 | bin 2: L_bin/L_plat - 1 = +0.294 | 0.071 | - | - | - | no |
| TTV | primary | r5 | no | [0.001, 0.0018] | -0.002448 | bin 1: L_bin/L_plat - 1 = +0.315 | 0.058 | - | - | - | no |
| TTV | primary | r7 | no | [0.0018, 0.0032] | -0.002548 | bin 2: fraction convex 0.588 < 0.6 | 0.315 | - | - | - | no |
| TTV | primary | r10 | no | [0.0018, 0.0032] | -0.001564 | bin 2: fraction convex 0.583 < 0.6 | 0.319 | - | - | - | no |
| TTV | primary | r15 | no | [0.0018, 0.0032] | -0.002498 | bin 2: fraction convex 0.584 < 0.6 | 0.295 | - | - | - | no |

- R1: GM 22/22 (needed 20) pass; AP 21/21 (needed 19) pass; TTV 11/11 (needed 10) pass
- R13: GM 12/22 (needed 20) fail, pooled plateau False; AP 6/21 (needed 19) fail, pooled plateau False; TTV 6/11 (needed 10) fail, pooled plateau False
- R11P: GM: ratio -, pass False, divide-line 7 m disk 0.84; AP: ratio -, pass False, divide-line 7 m disk 2.19; reading: neither passes at AP: location, survey or method detail (the report says which the evidence favours)

### E5. Four-way comparison, pooled per site (selection, window, statistic)

| site | pixels / window | statistic | |lap| (1/m) | 68 % (block) | ratio to Perron | n |
|---|---|---|---|---|---|---|
| GM | divide_square11 | median | 0.01002 | [0.00968, 0.0104] | 0.85 | 77,225 |
| GM | divide_square11 | geomean | 0.008944 | [0.00867, 0.0092] | 0.76 | 77,225 |
| GM | divide_disk7 | median | 0.009868 | [0.00951, 0.0102] | 0.84 | 76,926 |
| GM | divide_disk7 | geomean | 0.009026 | [0.00875, 0.00928] | 0.76 | 76,176 |
| AP | divide_square11 | median | 0.01231 | [0.012, 0.0127] | 2.24 | 57,982 |
| AP | divide_square11 | geomean | 0.01081 | [0.0106, 0.0111] | 1.97 | 57,982 |
| AP | divide_disk7 | median | 0.01202 | [0.0117, 0.0123] | 2.19 | 57,563 |
| AP | divide_disk7 | geomean | 0.01109 | [0.0108, 0.0113] | 2.02 | 56,505 |
| TTV | divide_square11 | median | 0.01629 | [0.0158, 0.0167] | - | 39,780 |
| TTV | divide_square11 | geomean | 0.01383 | [0.0134, 0.0142] | - | 39,780 |
| TTV | divide_disk7 | median | 0.01582 | [0.0154, 0.0163] | - | 39,582 |
| TTV | divide_disk7 | geomean | 0.01433 | [0.0139, 0.0147] | - | 38,555 |

### E6. Survey pairs (section 2.7), E*, TTV anchor

| site | pair | D0_P old | D0_P new | new/old | D0_N old | D0_N new | new/old |
|---|---|---|---|---|---|---|---|
| GM | GM1/GM1_fema | 0.01303 | 0.01364 | 1.05 | 0.0105 | 0.0102 | 0.97 |
| GM | GM2/GM2_fema | 0.01527 | 0.01619 | 1.06 | 0.01315 | 0.01281 | 0.97 |
| GM | GM3/GM3_fema | - | - | - | 0.01864 | 0.01913 | 1.03 |
| GM | GM4/GM4_fema | - | - | - | 0.01136 | 0.01097 | 0.97 |
| GM | GM5/GM5_fema | 0.02211 | - | - | 0.007921 | 0.009431 | 1.19 |
| GM | GM6/GM6_fema | 0.03973 | 0.04189 | 1.05 | 0.01696 | 0.01692 | 1.00 |
| GM | GM7/GM7_fema | 0.0399 | 0.04468 | 1.12 | 0.02863 | 0.03294 | 1.15 |
| AP | AP1/AP1_3dep | 0.006305 | - | - | 0.003315 | 0.003776 | 1.14 |
| AP | AP2/AP2_3dep | - | - | - | 0.003691 | 0.005961 | 1.61 |
| AP | AP3/AP3_3dep | 0.008787 | - | - | 0.003001 | 0.00433 | 1.44 |
| AP | AP4/AP4_3dep | 0.008919 | 0.01407 | 1.58 | 0.003186 | 0.004296 | 1.35 |
| AP | AP5/AP5_3dep | - | - | - | 0.003597 | 0.005435 | 1.51 |

| site | E* median (S_c 1.27) | < 10 | 10-30 | > 30 | S_c fit | 68 % | E* at fit |
|---|---|---|---|---|---|---|---|
| GM | 1.49 | 0.996 | 0.004 | 0.0000 | 0.404 | [0.398, 0.41] | 4.69 |
| AP | 1.85 | 0.984 | 0.016 | 0.0001 | 0.338 | [0.334, 0.342] | 6.93 |
| TTV | 1.98 | 0.971 | 0.029 | 0.0003 | 0.369 | [0.364, 0.374] | 6.81 |

| quantity | TTV site median | / 2.5e-3 | / 5e-3 | tiles |
|---|---|---|---|---|
| D0_P | 0.01016 | 4.06 | 2.03 | 6 |
| D0_N_A | 0.004831 | 1.93 | 0.97 | 11 |
| D0_N_S_7m | 0.00313 | 1.25 | 0.63 | 11 |
| D_R | 0.01306 | 5.22 | 2.61 | 11 |

### E7. The snapped heads (section 3.3)

| fit | n | a (m2) | 95 % | b | 95 % | R2 |
|---|---|---|---|---|---|---|
| i_table1 | 63 | 1,978 | [1,459, 2,774] | 1.647 | [1.26, 2.01] | 0.598 |
| ii_AP10_Sup | 64 | 53 | [14, 185] | 2.232 | [0.936, 3.54] | 0.123 |
| iii_AP10_Sdown | 64 | 76 | [23, 306] | 1.675 | [0.376, 2.83] | 0.083 |
| iv_AP10_Sup_without_49 | 63 | 52 | [13, 182] | 2.282 | [0.992, 3.6] | 0.128 |
| AP5_Sup | 64 | 19 | [4, 81] | 2.316 | [0.794, 3.86] | 0.116 |
| AP20_Sup | 64 | 188 | [59, 720] | 1.935 | [0.798, 2.95] | 0.140 |
| AP5_Sdown | 64 | 19 | [4, 86] | 2.059 | [0.508, 3.38] | 0.114 |
| AP20_Sdown | 64 | 255 | [75, 1,228] | 1.655 | [0.291, 2.71] | 0.083 |

| distribution | n | p5 | p25 | p50 | p75 | p95 |
|---|---|---|---|---|---|---|
| A_P0 | 64 | 1 | 2 | 4 | 17 | 668.6 |
| A_P5 | 64 | 9 | 26.75 | 141 | 663 | 6,234 |
| A_P10 | 64 | 29 | 82.75 | 529 | 2,112 | 1.779e+04 |
| A_P15 | 64 | 42 | 184 | 1,393 | 5,030 | 1.891e+04 |
| A_P20 | 64 | 69.15 | 367.5 | 2,306 | 7,119 | 1.921e+04 |
| A_P30 | 64 | 104.7 | 725.8 | 4,608 | 9,577 | 2.205e+04 |
| A_cell | 64 | 1 | 2 | 4 | 17 | 668.6 |
| A_3_a19 | 64 | 63.6 | 168.8 | 611 | 2,152 | 7,255 |
| A_10_a19 | 64 | 275.1 | 1,042 | 2,848 | 7,587 | 1.88e+04 |
| table1_area | 63 | 2,010 | 4,800 | 7,400 | 1.035e+04 | 1.486e+04 |
| A_P10_without_49 | 63 | 29 | 80.5 | 545 | 2,162 | 1.835e+04 |
| S_up_P10 | 64 | 0.238 | 0.3051 | 0.3743 | 0.4777 | 0.6502 |
| S_down_P10 | 64 | 0.1953 | 0.2664 | 0.3214 | 0.4322 | 0.6422 |
| table1_slope | 63 | 0.272 | 0.41 | 0.47 | 0.58 | 0.78 |
| axis_entry_m | 63 | 1.414 | 2.828 | 6.828 | 13.69 | 35.14 |
| axis_entry_A_m2 | 63 | 13.3 | 90.5 | 481 | 1,922 | 1.301e+04 |
| A_P10_over_A_3 | 64 | 0.04899 | 0.4731 | 1.063 | 1.267 | 4.535 |
| A_P10_over_A_10 | 64 | 0.009032 | 0.06644 | 0.2475 | 1 | 1.001 |

- R5: {'b_1.65_inside_ci95': True, 'median_ratio_to_eq10': 0.03883679029975922, 'median_in_range': False, 'consistent': False, 'ratio_pct': {'n': 64, 'p5': 0.0028710127006167144, 'p25': 0.009759329101210398, 'p50': 0.03883679029975922, 'p75': 0.24354442214816535, 'p95': 0.9672600288169642}, 'frac_in_0.5_2': 0.140625}; field values: {'A_F_m2': 529.0, 'A_F_without_49_m2': 545.0, 'A_F_table1_m2': 7400.0, 'b_hat': 2.2324047561380946, 'alpha_b_hat': 2.25, 'run_bhat_in_SJ': True, 'alphas_SF': [0.0, 1.65, 2.0], 'C_F': {'0': 529.0, '1.65': 75.05605194083904, '2': 53.10263406316027, '2.25': 40.65379910941503}, 'C_F_without_49': {'0': 545.0, '1.65': 76.53633474137145, '2': 57.16441589557756}, 'n_S_down_nonpositive': 0, 'C_F_definition': 'median over the 64 picks of A_P10 * S_down^alpha (S_down the 10 m D8 reach slope below the A_P10 cell); picks with S_down <= 0 drop out at alpha > 0'}

| pick | tile | A_cell | A_P5 | A_P10 | A_P20 | A_P30 | axis entry (m) | S_up P10 | S_down P10 | a19 A_3 | a19 A_10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | TTV3 | 3 | 33 | 1,136 | 6,765 | 7,547 | 4.8 | 0.288 | 0.295 | 2,864 | 3,703 |
| 2 | TTV3 | 38 | 284 | 296 | 1,385 | 1,549 | 13.7 | 0.193 | 0.261 | 6,324 | 13,943 |
| 3 | TTV1 | 1 | 135 | 142 | 13,991 | 16,034 | 3.8 | 0.303 | 0.262 | 171 | 13,354 |
| 4 | TTV1 | 3 | 257 | 307 | 389 | 418 | 2.0 | 0.372 | 0.195 | 227 | 408 |
| 5 | TTV1 | 1 | 537 | 545 | 3,347 | 3,967 | 3.4 | 0.512 | 0.499 | 424 | 1,789 |
| 6 | TTV1 | 19,739 | 19,752 | 20,039 | 21,214 | 22,339 | - | 0.242 | 0.213 | 19,748 | 20,039 |
| 7 | TTV1 | 4 | 2,704 | 2,720 | 8,253 | 8,589 | 5.7 | 0.278 | 0.302 | 2,687 | 4,612 |
| 8 | TTV1 | 9 | 80 | 7,356 | 8,038 | 8,056 | 9.2 | 0.251 | 0.292 | 619 | 7,356 |
| 9 | TTV1 | 4 | 147 | 156 | 188 | 35,743 | 2.8 | 0.442 | 0.535 | 136 | 1,049 |
| 10 | TTV1 | 2 | 9 | 17 | 363 | 1,184 | 11.1 | 0.392 | 0.252 | 141 | 14,960 |
| 11 | TTV1 | 1 | 514 | 624 | 673 | 12,421 | 2.0 | 0.237 | 0.280 | 509 | 628 |
| 12 | TTV1 | 2 | 13 | 156 | 2,157 | 14,610 | 9.2 | 0.377 | 0.231 | 134 | 1,286 |
| 13 | TTV1 | 6 | 4,910 | 7,455 | 7,476 | 8,895 | 1.4 | 0.299 | 0.374 | 4,221 | 7,455 |
| 14 | TTV1 | 7 | 41 | 62 | 70 | 114 | 20.8 | 0.306 | 0.341 | 168 | 2,037 |
| 15 | TTV1 | 4 | 20,290 | 20,309 | 21,093 | 22,513 | 2.0 | 0.181 | 0.197 | 20,272 | 20,309 |
| 16 | TTV1 | 2 | 6 | 1,446 | 2,574 | 2,653 | 9.9 | 0.512 | 0.431 | 1,259 | 1,445 |
| 17 | TTV1 | 2 | 2,610 | 33,875 | 34,158 | 36,950 | 4.2 | 0.282 | 0.233 | 2,589 | 33,875 |
| 18 | TTV1 | 5 | 17 | 32 | 13,635 | 13,706 | 13.7 | 0.583 | 0.260 | 7 | 13,494 |
| 19 | none | 1 | 5,225 | 19,474 | 20,130 | 20,388 | 2.4 | 0.312 | 0.348 | 5,152 | 19,474 |
| 20 | none | 18 | 1,955 | 2,260 | 3,041 | 3,124 | 1.0 | 0.301 | 0.171 | 1,952 | 2,230 |
| 21 | none | 17 | 44 | 75 | 2,320 | 13,941 | 13.9 | 0.430 | 0.331 | 63 | 11,323 |
| 22 | TTV1 | 4 | 9 | 92 | 108 | 156 | 9.4 | 0.507 | 0.612 | 67 | 415 |
| 23 | TTV1 | 4 | 135 | 143 | 2,893 | 3,083 | 2.4 | 0.662 | 0.695 | 2,330 | 2,704 |
| 24 | TTV1 | 2 | 640 | 659 | 926 | 967 | 2.4 | 0.652 | 0.549 | 606 | 659 |
| 25 | TTV1 | 4 | 27 | 31 | 77 | 468 | 14.9 | 0.404 | 0.560 | 169 | 1,019 |
| 26 | TTV1 | 4 | 10 | 16 | 69 | 95 | 12.9 | 0.547 | 0.617 | 109 | 227 |
| 27 | TTV1 | 75 | 85 | 127 | 145 | 246 | 35.5 | 0.499 | 0.671 | 82 | 1,476 |
| 28 | TTV2 | 1 | 6 | 29 | 42 | 53 | 8.8 | 0.365 | 0.365 | 641 | 668 |
| 29 | TTV2 | 5 | 1,195 | 1,654 | 2,293 | 2,688 | 1.4 | 0.578 | 0.321 | 1,180 | 1,654 |
| 30 | TTV2 | 20 | 99 | 103 | 123 | 130 | 34.5 | 0.534 | 0.647 | 97 | 102 |
| 31 | TTV2 | 66 | 71 | 76 | 246 | 7,131 | 26.9 | 0.725 | 0.650 | 91 | 315 |
| 32 | TTV2 | 1 | 5 | 2,063 | 2,077 | 3,215 | 11.3 | 0.405 | 0.286 | 2,044 | 8,133 |
| 33 | TTV2 | 28 | 36 | 55 | 67 | 91 | 35.2 | 0.396 | 0.315 | 172 | 426 |
| 34 | TTV2 | 168 | 1,800 | 1,820 | 7,057 | 7,596 | 1.4 | 0.467 | 0.385 | 1,786 | 2,935 |
| 35 | TTV3 | 9 | 675 | 693 | 10,134 | 10,666 | 3.8 | 0.322 | 0.326 | 571 | 7,820 |
| 36 | TTV3 | 9 | 243 | 7,741 | 8,276 | 8,361 | 1.4 | 0.451 | 0.253 | 7,257 | 8,129 |
| 37 | TTV3 | 8 | 13 | 18 | 226 | 245 | 20.4 | 0.306 | 0.268 | 38 | 1,298 |
| 38 | TTV2 | 17 | 25 | 30 | 1,119 | 2,158 | 12.4 | 0.363 | 0.271 | 701 | 1,074 |
| 39 | TTV2 | 4 | 24 | 705 | 3,555 | 3,864 | 9.2 | 0.639 | 0.419 | 639 | 2,188 |
| 40 | TTV2 | 40 | 244 | 809 | 8,686 | 9,178 | 6.4 | 0.406 | 0.303 | 187 | 7,330 |
| 41 | TTV2 | 1 | 174 | 188 | 554 | 13,346 | 4.2 | 0.602 | 0.424 | 207 | 268 |
| 42 | TTV2 | 4 | 28 | 179 | 580 | 9,457 | 8.1 | 0.335 | 0.234 | 79 | 8,786 |
| 43 | TTV2 | 12 | 16 | 36 | 385 | 748 | 20.8 | 0.458 | 0.486 | 257 | 376 |
| 44 | TTV2 | 2 | 4,045 | 4,124 | 4,772 | 4,785 | 2.8 | 0.503 | 0.429 | 3,723 | 4,134 |
| 45 | TTV2 | 1 | 12 | 40 | 4,900 | 6,688 | 4.4 | 0.316 | 0.317 | 4,465 | 4,479 |
| 46 | TTV2 | 1,190 | 1,207 | 1,314 | 2,484 | 3,299 | 123.1 | 0.365 | 0.436 | 1,193 | 1,314 |
| 47 | TTV2 | 4 | 26 | 29 | 63 | 103 | 19.6 | 0.543 | 0.431 | 23 | 112 |
| 48 | TTV2 | 3 | 6,412 | 6,417 | 8,359 | 9,192 | 4.0 | 0.397 | 0.476 | 6,181 | 6,416 |
| 49 | TTV2 | 1 | 147 | 150 | 163 | 177 | 1.0 | 0.308 | 0.324 | 726 | 2,761 |
| 50 | TTV2 | 1 | 243 | 7,502 | 11,256 | 12,328 | 2.0 | 0.324 | 0.306 | 7,242 | 7,509 |
| 51 | TTV2 | 3 | 1,392 | 1,398 | 1,415 | 1,497 | 2.8 | 0.252 | 0.298 | 1,387 | 1,396 |
| 52 | TTV2 | 7,265 | 7,276 | 7,280 | 7,305 | 9,937 | 76.1 | 0.254 | 0.169 | 7,272 | 7,280 |
| 53 | TTV2 | 10 | 353 | 359 | 369 | 431 | 2.4 | 0.270 | 0.234 | 346 | 7,899 |
| 54 | TTV2 | 2 | 12 | 3,256 | 3,837 | 9,161 | 6.8 | 0.318 | 0.294 | 4,527 | 4,545 |
| 55 | TTV2 | 1 | 486 | 513 | 647 | 659 | 1.4 | 0.280 | 0.274 | 482 | 5,375 |
| 56 | TTV2 | 2 | 433 | 836 | 883 | 918 | 5.7 | 0.186 | 0.229 | 345 | 976 |
| 57 | TTV2 | 47 | 51 | 298 | 5,261 | 6,207 | 17.3 | 0.668 | 0.189 | 87 | 3,481 |
| 58 | TTV2 | 9 | 71 | 85 | 188 | 218 | 13.7 | 0.364 | 0.321 | 295 | 8,846 |
| 59 | TTV2 | 33 | 38 | 58 | 1,629 | 11,251 | 20.5 | 0.341 | 0.303 | 795 | 823 |
| 60 | TTV2 | 757 | 785 | 8,225 | 9,577 | 11,972 | 10.8 | 0.406 | 0.479 | 781 | 8,225 |
| 61 | TTV2 | 2 | 15 | 2,332 | 3,941 | 4,430 | 6.7 | 0.471 | 0.337 | 2,093 | 2,332 |
| 62 | TTV2 | 1 | 659 | 755 | 4,823 | 5,327 | 5.2 | 0.324 | 0.511 | 616 | 4,030 |
| 63 | TTV2 | 9 | 27 | 33 | 77 | 163 | 25.1 | 0.467 | 0.493 | 70 | 428 |
| 64 | TTV2 | 34 | 228 | 706 | 5,729 | 6,714 | 8.4 | 0.422 | 0.417 | 226 | 4,107 |

### E8. Field-fixed thresholds (R6, R14) and SJ (R7, R9)

**PF** (A = 529 m2): beats AD 5/11; MSE/P4 median 2.92; D median 0.007835; D at bound 1; PF3 vs A1 11/11, vs A2(D0_R) 3/9, vs A2(D0_P) 5/11

| tile | MSE PF | MSE AD | MSE P4 | PF/P4 | K/K0 | D | where | MSE PF3 | D PF3 | heads |
|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 446.11 | 214.19 | 82.73 | 5.39 | 1.082 | 0.01035 | interior | 487.46 | 0.05 | 679 |
| TTV2 | 138.74 | 209.66 | 59.79 | 2.32 | 1.029 | 0.007835 | interior | 142.05 | 0.01006 | 706 |
| TTV3 | 202.51 | 326.85 | 87.40 | 2.32 | 1.090 | 0.004374 | interior | 212.04 | 0.00563 | 715 |
| TTV4 | 215.73 | 293.97 | 96.82 | 2.23 | 1.045 | 0.01467 | interior | 225.29 | 0.05 | 717 |
| TTV5 | 716.18 | 842.95 | 137.88 | 5.19 | 1.131 | 0.003195 | interior | 732.44 | 0.004283 | 500 |
| TTV6 | 275.48 | 204.34 | 94.27 | 2.92 | 1.108 | 0.05 | y_lo | 366.73 | 0.05 | 690 |
| TTV7 | 462.47 | 134.26 | 79.24 | 5.84 | 1.088 | 0.01165 | interior | 496.06 | 0.05 | 622 |
| TTV8 | 379.21 | 272.96 | 114.67 | 3.31 | 1.113 | 0.004974 | interior | 420.16 | 0.01554 | 668 |
| TTV9 | 712.00 | 284.91 | 115.18 | 6.18 | 1.049 | 0.005821 | interior | 724.63 | 0.009765 | 656 |
| TTV11 | 556.39 | 164.20 | 241.77 | 2.30 | 1.126 | 0.004549 | interior | 611.39 | 0.02111 | 557 |
| TTV15 | 293.26 | 327.53 | 103.42 | 2.84 | 1.084 | 0.009993 | interior | 310.92 | 0.05 | 630 |

**PF_7400** (A = 7,400 m2): beats AD 8/11; MSE/P4 median 1.62; D median 0.02026; D at bound 0; PF3 vs A1 11/11, vs A2(D0_R) 7/9, vs A2(D0_P) 8/11

| tile | MSE PF | MSE AD | MSE P4 | PF/P4 | K/K0 | D | where | MSE PF3 | D PF3 | heads |
|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 115.50 | 214.19 | 82.73 | 1.40 | 0.844 | 0.02044 | interior | 174.94 | 0.01379 | 38 |
| TTV2 | 66.67 | 209.66 | 59.79 | 1.11 | 1.030 | 0.0149 | interior | 68.10 | 0.01571 | 48 |
| TTV3 | 146.00 | 326.85 | 87.40 | 1.67 | 0.989 | 0.02026 | interior | 146.07 | 0.01997 | 53 |
| TTV4 | 129.88 | 293.97 | 96.82 | 1.34 | 0.944 | 0.02452 | interior | 137.40 | 0.0201 | 35 |
| TTV5 | 603.55 | 842.95 | 137.88 | 4.38 | 0.930 | 0.01509 | interior | 607.31 | 0.01346 | 44 |
| TTV6 | 118.67 | 204.34 | 94.27 | 1.26 | 0.828 | 0.02729 | interior | 178.32 | 0.02014 | 35 |
| TTV7 | 161.89 | 134.26 | 79.24 | 2.04 | 0.806 | 0.04051 | interior | 242.05 | 0.01679 | 36 |
| TTV8 | 189.47 | 272.96 | 114.67 | 1.65 | 0.839 | 0.02639 | interior | 236.39 | 0.01479 | 32 |
| TTV9 | 350.16 | 284.91 | 115.18 | 3.04 | 0.868 | 0.01692 | interior | 417.98 | 0.009754 | 33 |
| TTV11 | 305.81 | 164.20 | 241.77 | 1.26 | 0.870 | 0.02005 | interior | 346.37 | 0.01281 | 36 |
| TTV15 | 167.44 | 327.53 | 103.42 | 1.62 | 1.028 | 0.01771 | interior | 168.17 | 0.01867 | 44 |

**PF_2848** (A = 2,848 m2): beats AD 8/11; MSE/P4 median 2.11; D median 0.01473; D at bound 0; PF3 vs A1 11/11, vs A2(D0_R) 5/9, vs A2(D0_P) 8/11

| tile | MSE PF | MSE AD | MSE P4 | PF/P4 | K/K0 | D | where | MSE PF3 | D PF3 | heads |
|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 209.30 | 214.19 | 82.73 | 2.53 | 0.907 | 0.01893 | interior | 235.55 | 0.01109 | 88 |
| TTV2 | 90.04 | 209.66 | 59.79 | 1.51 | 0.997 | 0.0105 | interior | 90.06 | 0.01042 | 96 |
| TTV3 | 168.50 | 326.85 | 87.40 | 1.93 | 1.032 | 0.01186 | interior | 169.12 | 0.01247 | 115 |
| TTV4 | 128.23 | 293.97 | 96.82 | 1.32 | 0.971 | 0.01609 | interior | 130.82 | 0.01415 | 91 |
| TTV5 | 652.64 | 842.95 | 137.88 | 4.73 | 1.009 | 0.009173 | interior | 652.71 | 0.009313 | 93 |
| TTV6 | 151.40 | 204.34 | 94.27 | 1.61 | 0.892 | 0.0209 | interior | 181.22 | 0.0144 | 97 |
| TTV7 | 266.95 | 134.26 | 79.24 | 3.37 | 0.902 | 0.02777 | interior | 294.32 | 0.01356 | 73 |
| TTV8 | 241.62 | 272.96 | 114.67 | 2.11 | 0.924 | 0.01467 | interior | 253.14 | 0.01006 | 83 |
| TTV9 | 443.63 | 284.91 | 115.18 | 3.85 | 0.925 | 0.01085 | interior | 465.17 | 0.007675 | 71 |
| TTV11 | 364.98 | 164.20 | 241.77 | 1.51 | 0.931 | 0.01731 | interior | 378.53 | 0.0112 | 79 |
| TTV15 | 230.00 | 327.53 | 103.42 | 2.22 | 0.993 | 0.01473 | interior | 230.08 | 0.01428 | 90 |

**PF_611** (A = 611 m2): beats AD 5/11; MSE/P4 median 2.75; D median 0.008006; D at bound 1; PF3 vs A1 11/11, vs A2(D0_R) 3/9, vs A2(D0_P) 6/11

| tile | MSE PF | MSE AD | MSE P4 | PF/P4 | K/K0 | D | where | MSE PF3 | D PF3 | heads |
|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 433.55 | 214.19 | 82.73 | 5.24 | 1.071 | 0.009243 | interior | 458.70 | 0.05 | 564 |
| TTV2 | 129.63 | 209.66 | 59.79 | 2.17 | 1.021 | 0.008006 | interior | 131.28 | 0.009318 | 589 |
| TTV3 | 195.61 | 326.85 | 87.40 | 2.24 | 1.085 | 0.004946 | interior | 203.80 | 0.006178 | 606 |
| TTV4 | 206.81 | 293.97 | 96.82 | 2.14 | 1.034 | 0.01454 | interior | 211.94 | 0.03025 | 616 |
| TTV5 | 720.50 | 842.95 | 137.88 | 5.23 | 1.111 | 0.003722 | interior | 732.42 | 0.004748 | 421 |
| TTV6 | 256.41 | 204.34 | 94.27 | 2.72 | 1.082 | 0.05 | y_lo | 308.39 | 0.05 | 607 |
| TTV7 | 432.46 | 134.26 | 79.24 | 5.46 | 1.061 | 0.01375 | interior | 446.55 | 0.05 | 512 |
| TTV8 | 371.68 | 272.96 | 114.67 | 3.24 | 1.107 | 0.004989 | interior | 406.96 | 0.01162 | 558 |
| TTV9 | 680.05 | 284.91 | 115.18 | 5.90 | 1.033 | 0.006229 | interior | 685.53 | 0.008409 | 532 |
| TTV11 | 539.63 | 164.20 | 241.77 | 2.23 | 1.114 | 0.004821 | interior | 583.20 | 0.01528 | 455 |
| TTV15 | 284.61 | 327.53 | 103.42 | 2.75 | 1.069 | 0.01097 | interior | 295.69 | 0.04389 | 531 |

| SF alpha | C_F | beats AD | MSE/P4 median | K/K0 median | D median | D at bound |
|---|---|---|---|---|---|---|
| 0 | 529 | 5/11 | 2.92 | 1.088 | 0.007835 | 1 |
| 1.65 | 75.06 | 6/11 | 2.34 | 1.047 | 0.05 | 6 |
| 2 | 53.1 | 6/11 | 2.30 | 1.049 | 0.03751 | 5 |

| tile | alpha | C | K | D | MSE | held-out | where | evaluations | run |
|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 0 | 5.95e+04 | 1.235e-05 | 0.04388 | 82.735 | 159.319 | interior | 2,023 | 20261120T185725Z_a20_sj_TTV1_a0 |
| TTV1 | 1.65 | 1.108e+04 | 1.066e-05 | 0.03809 | 79.714 | 119.401 | interior | 2,000 | 20261120T185725Z_a20_sj_TTV1_a1.65 |
| TTV1 | 2 | 8080 | 1.065e-05 | 0.03689 | 84.447 | 153.689 | interior | 2,000 | 20261120T185725Z_a20_sj_TTV1_a2 |
| TTV1 | 2.25 | 6463 | 1.047e-05 | 0.03708 | 87.282 | 154.268 | interior | 2,000 | 20261120T191341Z_a20_sj_TTV1_a2.25 |
| TTV2 | 0 | 9131 | 1.645e-05 | 0.0166 | 59.791 | 256.849 | interior | 2,046 | 20261120T185725Z_a20_sj_TTV2_a0 |
| TTV2 | 1.65 | 3361 | 1.685e-05 | 0.0238 | 76.785 | 335.835 | interior | 2,004 | 20261120T185725Z_a20_sj_TTV2_a1.65 |
| TTV2 | 2 | 2357 | 1.681e-05 | 0.02341 | 79.559 | 350.947 | interior | 2,007 | 20261120T185725Z_a20_sj_TTV2_a2 |
| TTV2 | 2.25 | 1899 | 1.67e-05 | 0.02271 | 84.971 | 350.858 | interior | 2,009 | 20261120T191341Z_a20_sj_TTV2_a2.25 |
| TTV3 | 0 | 8.912e+04 | 3.733e-05 | 0.05 | 87.397 | 182.397 | y_lo | 2,036 | 20261120T185725Z_a20_sj_TTV3_a0 |
| TTV3 | 1.65 | 1688 | 2.583e-05 | 0.02358 | 124.769 | 137.601 | interior | 2,000 | 20261120T185725Z_a20_sj_TTV3_a1.65 |
| TTV3 | 2 | 1294 | 2.617e-05 | 0.02428 | 120.112 | 137.927 | interior | 2,000 | 20261120T185725Z_a20_sj_TTV3_a2 |
| TTV3 | 2.25 | 1047 | 2.621e-05 | 0.02465 | 119.432 | 132.591 | interior | 2,000 | 20261120T191341Z_a20_sj_TTV3_a2.25 |
| TTV4 | 0 | 4.031e+04 | 1.543e-05 | 0.05 | 96.819 | 108.455 | y_lo | 2,022 | 20261120T185725Z_a20_sj_TTV4_a0 |
| TTV4 | 1.65 | 6712 | 1.545e-05 | 0.05 | 89.674 | 100.462 | y_lo | 2,000 | 20261120T185725Z_a20_sj_TTV4_a1.65 |
| TTV4 | 2 | 5310 | 1.554e-05 | 0.05 | 87.859 | 99.080 | y_lo | 2,000 | 20261120T185725Z_a20_sj_TTV4_a2 |
| TTV4 | 2.25 | 4339 | 1.557e-05 | 0.05 | 87.149 | 97.879 | y_lo | 2,000 | 20261120T191341Z_a20_sj_TTV4_a2.25 |
| TTV5 | 0 | 5.861e+04 | 2.651e-05 | 0.05 | 137.878 | 330.585 | y_lo | 2,035 | 20261120T185725Z_a20_sj_TTV5_a0 |
| TTV5 | 1.65 | 7071 | 1.706e-05 | 0.05 | 325.123 | 1690.557 | y_lo | 2,000 | 20261120T185725Z_a20_sj_TTV5_a1.65 |
| TTV5 | 2 | 4629 | 1.606e-05 | 0.05 | 350.144 | 1264.253 | y_lo | 2,000 | 20261120T185725Z_a20_sj_TTV5_a2 |
| TTV5 | 2.25 | 1829 | 1.968e-05 | 0.03173 | 377.182 | 1249.531 | interior | 2,000 | 20261120T191341Z_a20_sj_TTV5_a2.25 |
| TTV6 | 0 | 1.097e+04 | 9.512e-06 | 0.03327 | 94.266 | 239.450 | interior | 2,057 | 20261120T185725Z_a20_sj_TTV6_a0 |
| TTV6 | 1.65 | 1498 | 9.548e-06 | 0.0333 | 100.651 | 232.906 | interior | 2,012 | 20261120T185725Z_a20_sj_TTV6_a1.65 |
| TTV6 | 2 | 478.8 | 9.928e-06 | 0.03334 | 99.362 | 236.322 | interior | 2,006 | 20261120T185725Z_a20_sj_TTV6_a2 |
| TTV6 | 2.25 | 371.3 | 9.928e-06 | 0.03353 | 98.645 | 258.180 | interior | 2,007 | 20261120T191341Z_a20_sj_TTV6_a2.25 |
| TTV7 | 0 | 3.597e+04 | 8.13e-06 | 0.05 | 79.236 | 103.831 | y_lo | 2,024 | 20261120T185725Z_a20_sj_TTV7_a0 |
| TTV7 | 1.65 | 5783 | 7.627e-06 | 0.05 | 92.626 | 139.937 | y_lo | 2,001 | 20261120T185725Z_a20_sj_TTV7_a1.65 |
| TTV7 | 2 | 3702 | 7.619e-06 | 0.05 | 98.280 | 134.343 | y_lo | 2,002 | 20261120T185725Z_a20_sj_TTV7_a2 |
| TTV7 | 2.25 | 2883 | 7.617e-06 | 0.05 | 98.958 | 135.703 | y_lo | 2,000 | 20261120T191341Z_a20_sj_TTV7_a2.25 |
| TTV8 | 0 | 3.854e+04 | 8.34e-06 | 0.04557 | 114.666 | 241.692 | interior | 2,021 | 20261120T185725Z_a20_sj_TTV8_a0 |
| TTV8 | 1.65 | 5435 | 8.258e-06 | 0.04242 | 115.892 | 237.577 | interior | 2,000 | 20261120T185725Z_a20_sj_TTV8_a1.65 |
| TTV8 | 2 | 1934 | 8.418e-06 | 0.03617 | 118.060 | 254.729 | interior | 2,000 | 20261120T185725Z_a20_sj_TTV8_a2 |
| TTV8 | 2.25 | 1344 | 8.404e-06 | 0.03731 | 116.933 | 269.574 | interior | 2,007 | 20261120T191341Z_a20_sj_TTV8_a2.25 |
| TTV9 | 0 | 3.656e+04 | 7.648e-06 | 0.04377 | 115.182 | 137.702 | interior | 2,021 | 20261120T185725Z_a20_sj_TTV9_a0 |
| TTV9 | 1.65 | 5872 | 7.62e-06 | 0.04231 | 118.942 | 214.067 | interior | 2,001 | 20261120T185725Z_a20_sj_TTV9_a1.65 |
| TTV9 | 2 | 9885 | 7.632e-06 | 0.05 | 121.915 | 279.909 | y_lo | 2,000 | 20261120T185725Z_a20_sj_TTV9_a2 |
| TTV9 | 2.25 | 7624 | 7.625e-06 | 0.05 | 121.373 | 278.688 | y_lo | 2,002 | 20261120T191341Z_a20_sj_TTV9_a2.25 |
| TTV11 | 0 | 3.008e+04 | 7.979e-06 | 0.04112 | 241.861 | 367.219 | interior | 2,064 | 20261120T185725Z_a20_sj_TTV11_a0 |
| TTV11 | 1.65 | 2288 | 7.877e-06 | 0.04894 | 192.652 | 280.568 | interior | 2,007 | 20261120T185725Z_a20_sj_TTV11_a1.65 |
| TTV11 | 2 | 448.8 | 8.925e-06 | 0.05 | 180.687 | 281.525 | y_lo | 2,001 | 20261120T185725Z_a20_sj_TTV11_a2 |
| TTV11 | 2.25 | 369.7 | 8.863e-06 | 0.05 | 170.762 | 269.085 | y_lo | 2,011 | 20261120T191341Z_a20_sj_TTV11_a2.25 |
| TTV15 | 0 | 2.682e+04 | 1.525e-05 | 0.02858 | 103.420 | 171.064 | interior | 2,059 | 20261120T185725Z_a20_sj_TTV15_a0 |
| TTV15 | 1.65 | 8433 | 1.141e-05 | 0.05 | 103.563 | 160.321 | y_lo | 2,000 | 20261120T185725Z_a20_sj_TTV15_a1.65 |
| TTV15 | 2 | 6660 | 1.151e-05 | 0.05 | 102.647 | 175.870 | y_lo | 2,000 | 20261120T185725Z_a20_sj_TTV15_a2 |
| TTV15 | 2.25 | 2328 | 9.229e-06 | 0.04362 | 102.489 | 166.248 | interior | 2,000 | 20261120T191341Z_a20_sj_TTV15_a2.25 |

- R7: preferred on 6/11 (needed 9); best alpha {'TTV1': '1.65', 'TTV2': '1.65', 'TTV3': '2.25', 'TTV4': '2.25', 'TTV5': '2.25', 'TTV6': '1.65', 'TTV7': '2', 'TTV8': '1.65', 'TTV9': '1.65', 'TTV11': '2.25', 'TTV15': '1.65'}; median improvement 0.027; alpha = 0 rows equal the addendum 18 rows on every tile: True; same optimum on 10/11

| model | median D/D0_P | median D/D0_N | median D/D0_R | median D/D_R | at bound | N |
|---|---|---|---|---|---|---|
| AD | 3.48 | 0.54 | 0.56 | 0.19 | 2 | 11 |
| P4 | 4.13 | 9.08 | 16.14 | 3.47 | 4 | 11 |
| P3 | 2.77 | 8.69 | 14.88 | 2.88 | 2 | 11 |
| A1 | 4.29 | 0.00 | 0.00 | 0.00 | 6 | 11 |
| PF | 0.69 | 1.60 | 1.97 | 0.65 | 1 | 11 |
| PF3 | 3.00 | 4.58 | 9.13 | 1.79 | 5 | 11 |
| PF_7400 | 2.10 | 4.35 | 7.04 | 1.66 | 0 | 11 |
| PF_74003 | 1.66 | 3.37 | 5.42 | 1.14 | 0 | 11 |
| PF_2848 | 1.69 | 2.99 | 6.52 | 1.12 | 0 | 11 |
| PF_28483 | 1.34 | 2.38 | 3.82 | 0.87 | 0 | 11 |
| PF_611 | 0.68 | 1.72 | 2.08 | 0.62 | 1 | 11 |
| PF_6113 | 2.62 | 3.31 | 6.61 | 1.30 | 3 | 11 |
| SF_a0 | 0.69 | 1.60 | 1.97 | 0.65 | 1 | 11 |
| SF_a1.65 | 5.01 | 9.02 | 12.55 | 3.32 | 6 | 11 |
| SF_a2 | 4.47 | 9.02 | 12.55 | 2.85 | 5 | 11 |
| SJ_a0 | 4.13 | 9.08 | 16.14 | 3.47 | 4 | 11 |
| SJ_a1.65 | 5.01 | 9.13 | 16.14 | 3.80 | 4 | 11 |
| SJ_a2 | 4.76 | 8.25 | 16.14 | 3.80 | 6 | 11 |
| SJ_a2.25 | 4.56 | 8.02 | 12.77 | 2.84 | 4 | 11 |

| model | inside | of determined | beats AD | MSE/P4 median | reading |
|---|---|---|---|---|---|
| PF | 3 | 11 | 5 | 2.92 | the reading's conditions do not both hold (inside 3/11 determined, needed 9; beats AD 5/11, needed 9) |
| PF_7400 | 0 | 11 | 8 | 1.62 | - |
| PF_2848 | 2 | 11 | 8 | 2.11 | - |
| PF_611 | 4 | 11 | 5 | 2.75 | - |

### E9. Model heads (R8) and model-network D_R

| model | median dist (m) | heads / picks | heads | picks | reading |
|---|---|---|---|---|---|
| P4 | 61.7 | 0.76 | 47 | 62 | count in range, distances large |
| PF_AF | 1.0 | 21.56 | 1,337 | 62 | too many channels |
| PF_7400 | 15.2 | 1.39 | 86 | 62 | agree with the mapped heads |
| PF_2848 | 5.0 | 2.98 | 185 | 62 | too many channels |
| PF_611 | 1.0 | 17.98 | 1,115 | 62 | too many channels |
| SF_a0 | 1.0 | 21.56 | 1,337 | 62 | too many channels |
| SF_a1.65 | 0.0 | 35.56 | 2,205 | 62 | too many channels |
| SF_a2 | 0.0 | 39.95 | 2,477 | 62 | too many channels |
| SJ_a0 | 61.7 | 0.76 | 47 | 62 | count in range, distances large |
| SJ_a1.65 | 77.0 | 0.69 | 43 | 62 | count in range, distances large |
| SJ_a2 | 77.0 | 0.76 | 47 | 62 | count in range, distances large |
| SJ_a2.25 | 77.0 | 0.74 | 46 | 62 | count in range, distances large |

| tile | model | D_R (model network) | fit D | fit D / D_R | L_H median (m) |
|---|---|---|---|---|---|
| TTV1 | P4 | 0.03691 | 0.04388 | 1.19 | 309 |
| TTV1 | PF | 0.01118 | 0.01035 | 0.93 | 58 |
| TTV1 | PF_7400 | 0.02292 | 0.02044 | 0.89 | 166 |
| TTV1 | PF_2848 | 0.01974 | 0.01893 | 0.96 | 121 |
| TTV1 | PF_611 | 0.01183 | 0.009243 | 0.78 | 63 |
| TTV1 | SF_a0 | 0.01118 | 0.01035 | 0.93 | 58 |
| TTV1 | SF_a1.65 | 0.01388 | 0.02948 | 2.12 | 67 |
| TTV1 | SF_a2 | 0.01412 | 0.01702 | 1.21 | 68 |
| TTV1 | SJ_a0 | 0.03691 | 0.04388 | 1.19 | 309 |
| TTV1 | SJ_a1.65 | 0.03306 | 0.03809 | 1.15 | 276 |
| TTV1 | SJ_a2 | 0.03254 | 0.03689 | 1.13 | 269 |
| TTV1 | SJ_a2.25 | 0.03247 | 0.03708 | 1.14 | 267 |
| TTV2 | P4 | 0.01762 | 0.0166 | 0.94 | 152 |
| TTV2 | PF | 0.01015 | 0.007835 | 0.77 | 73 |
| TTV2 | PF_7400 | 0.01703 | 0.0149 | 0.87 | 144 |
| TTV2 | PF_2848 | 0.01399 | 0.0105 | 0.75 | 110 |
| TTV2 | PF_611 | 0.01068 | 0.008006 | 0.75 | 77 |
| TTV2 | SF_a0 | 0.01015 | 0.007835 | 0.77 | 73 |
| TTV2 | SF_a1.65 | 0.0102 | 0.008958 | 0.88 | 64 |
| TTV2 | SF_a2 | 0.01019 | 0.009188 | 0.90 | 63 |
| TTV2 | SJ_a0 | 0.01762 | 0.0166 | 0.94 | 152 |
| TTV2 | SJ_a1.65 | 0.02336 | 0.0238 | 1.02 | 189 |
| TTV2 | SJ_a2 | 0.0232 | 0.02341 | 1.01 | 187 |
| TTV2 | SJ_a2.25 | 0.02357 | 0.02271 | 0.96 | 189 |
| TTV3 | P4 | 0.06477 | 0.05 | 0.77 | 383 |
| TTV3 | PF | 0.01466 | 0.004374 | 0.30 | 71 |
| TTV3 | PF_7400 | 0.02644 | 0.02026 | 0.77 | 170 |
| TTV3 | PF_2848 | 0.02138 | 0.01186 | 0.55 | 141 |
| TTV3 | PF_611 | 0.01507 | 0.004946 | 0.33 | 80 |
| TTV3 | SF_a0 | 0.01466 | 0.004374 | 0.30 | 71 |
| TTV3 | SF_a1.65 | 0.01662 | 0.009275 | 0.56 | 77 |
| TTV3 | SF_a2 | 0.01676 | 0.01052 | 0.63 | 79 |
| TTV3 | SJ_a0 | 0.06477 | 0.05 | 0.77 | 383 |
| TTV3 | SJ_a1.65 | 0.02948 | 0.02358 | 0.80 | 184 |
| TTV3 | SJ_a2 | 0.0302 | 0.02428 | 0.80 | 189 |
| TTV3 | SJ_a2.25 | 0.0302 | 0.02465 | 0.82 | 190 |
| TTV4 | P4 | 0.03613 | 0.05 | 1.38 | 232 |
| TTV4 | PF | 0.009784 | 0.01467 | 1.50 | 59 |
| TTV4 | PF_7400 | 0.02563 | 0.02452 | 0.96 | 160 |
| TTV4 | PF_2848 | 0.0211 | 0.01609 | 0.76 | 124 |
| TTV4 | PF_611 | 0.01044 | 0.01454 | 1.39 | 63 |
| TTV4 | SF_a0 | 0.009784 | 0.01467 | 1.50 | 59 |
| TTV4 | SF_a1.65 | 0.01226 | 0.0237 | 1.93 | 70 |
| TTV4 | SF_a2 | 0.01249 | 0.02173 | 1.74 | 71 |
| TTV4 | SJ_a0 | 0.03613 | 0.05 | 1.38 | 232 |
| TTV4 | SJ_a1.65 | 0.03619 | 0.05 | 1.38 | 236 |
| TTV4 | SJ_a2 | 0.03628 | 0.05 | 1.38 | 237 |
| TTV4 | SJ_a2.25 | 0.03628 | 0.05 | 1.38 | 237 |
| TTV5 | P4 | 0.06729 | 0.05 | 0.74 | 298 |
| TTV5 | PF | 0.009891 | 0.003195 | 0.32 | 60 |
| TTV5 | PF_7400 | 0.02097 | 0.01509 | 0.72 | 127 |
| TTV5 | PF_2848 | 0.01599 | 0.009173 | 0.57 | 93 |
| TTV5 | PF_611 | 0.01024 | 0.003722 | 0.36 | 63 |
| TTV5 | SF_a0 | 0.009891 | 0.003195 | 0.32 | 60 |
| TTV5 | SF_a1.65 | 0.01099 | 0.02327 | 2.12 | 65 |
| TTV5 | SF_a2 | 0.01136 | 0.03116 | 2.74 | 67 |
| TTV5 | SJ_a0 | 0.06729 | 0.05 | 0.74 | 298 |
| TTV5 | SJ_a1.65 | 0.06208 | 0.05 | 0.81 | 268 |
| TTV5 | SJ_a2 | 0.06214 | 0.05 | 0.80 | 266 |
| TTV5 | SJ_a2.25 | 0.03928 | 0.03173 | 0.81 | 202 |
| TTV6 | P4 | 0.02773 | 0.03327 | 1.20 | 174 |
| TTV6 | PF | 0.01196 | 0.05 | 4.18 | 71 |
| TTV6 | PF_7400 | 0.02587 | 0.02729 | 1.06 | 163 |
| TTV6 | PF_2848 | 0.02054 | 0.0209 | 1.02 | 131 |
| TTV6 | PF_611 | 0.01265 | 0.05 | 3.95 | 77 |
| TTV6 | SF_a0 | 0.01196 | 0.05 | 4.18 | 71 |
| TTV6 | SF_a1.65 | 0.01264 | 0.05 | 3.96 | 70 |
| TTV6 | SF_a2 | 0.01272 | 0.05 | 3.93 | 70 |
| TTV6 | SJ_a0 | 0.02773 | 0.03327 | 1.20 | 174 |
| TTV6 | SJ_a1.65 | 0.02951 | 0.0333 | 1.13 | 183 |
| TTV6 | SJ_a2 | 0.02475 | 0.03334 | 1.35 | 155 |
| TTV6 | SJ_a2.25 | 0.02496 | 0.03353 | 1.34 | 155 |
| TTV7 | P4 | 0.02544 | 0.05 | 1.97 | 183 |
| TTV7 | PF | 0.01013 | 0.01165 | 1.15 | 64 |
| TTV7 | PF_7400 | 0.01686 | 0.04051 | 2.40 | 117 |
| TTV7 | PF_2848 | 0.01503 | 0.02777 | 1.85 | 101 |
| TTV7 | PF_611 | 0.01063 | 0.01375 | 1.29 | 68 |
| TTV7 | SF_a0 | 0.01013 | 0.01165 | 1.15 | 64 |
| TTV7 | SF_a1.65 | 0.01119 | 0.05 | 4.47 | 63 |
| TTV7 | SF_a2 | 0.01173 | 0.05 | 4.26 | 64 |
| TTV7 | SJ_a0 | 0.02544 | 0.05 | 1.97 | 183 |
| TTV7 | SJ_a1.65 | 0.0233 | 0.05 | 2.15 | 179 |
| TTV7 | SJ_a2 | 0.02307 | 0.05 | 2.17 | 169 |
| TTV7 | SJ_a2.25 | 0.02294 | 0.05 | 2.18 | 167 |
| TTV8 | P4 | 0.03968 | 0.04557 | 1.15 | 267 |
| TTV8 | PF | 0.01074 | 0.004974 | 0.46 | 63 |
| TTV8 | PF_7400 | 0.02072 | 0.02639 | 1.27 | 137 |
| TTV8 | PF_2848 | 0.01669 | 0.01467 | 0.88 | 102 |
| TTV8 | PF_611 | 0.0112 | 0.004989 | 0.45 | 67 |
| TTV8 | SF_a0 | 0.01074 | 0.004974 | 0.46 | 63 |
| TTV8 | SF_a1.65 | 0.01232 | 0.05 | 4.06 | 64 |
| TTV8 | SF_a2 | 0.01252 | 0.05 | 3.99 | 65 |
| TTV8 | SJ_a0 | 0.03968 | 0.04557 | 1.15 | 267 |
| TTV8 | SJ_a1.65 | 0.04099 | 0.04242 | 1.03 | 259 |
| TTV8 | SJ_a2 | 0.03163 | 0.03617 | 1.14 | 188 |
| TTV8 | SJ_a2.25 | 0.03093 | 0.03731 | 1.21 | 183 |
| TTV9 | P4 | 0.04126 | 0.04377 | 1.06 | 274 |
| TTV9 | PF | 0.008588 | 0.005821 | 0.68 | 61 |
| TTV9 | PF_7400 | 0.01452 | 0.01692 | 1.16 | 109 |
| TTV9 | PF_2848 | 0.01217 | 0.01085 | 0.89 | 91 |
| TTV9 | PF_611 | 0.009167 | 0.006229 | 0.68 | 65 |
| TTV9 | SF_a0 | 0.008588 | 0.005821 | 0.68 | 61 |
| TTV9 | SF_a1.65 | 0.008281 | 0.05 | 6.04 | 54 |
| TTV9 | SF_a2 | 0.008128 | 0.05 | 6.15 | 53 |
| TTV9 | SJ_a0 | 0.04126 | 0.04377 | 1.06 | 274 |
| TTV9 | SJ_a1.65 | 0.04298 | 0.04231 | 0.98 | 241 |
| TTV9 | SJ_a2 | 0.05037 | 0.05 | 0.99 | 318 |
| TTV9 | SJ_a2.25 | 0.05037 | 0.05 | 0.99 | 318 |
| TTV11 | P4 | 0.03617 | 0.04108 | 1.14 | 251 |
| TTV11 | PF | 0.009843 | 0.004549 | 0.46 | 63 |
| TTV11 | PF_7400 | 0.01952 | 0.02005 | 1.03 | 139 |
| TTV11 | PF_2848 | 0.0151 | 0.01731 | 1.15 | 104 |
| TTV11 | PF_611 | 0.01018 | 0.004821 | 0.47 | 66 |
| TTV11 | SF_a0 | 0.009843 | 0.004549 | 0.46 | 63 |
| TTV11 | SF_a1.65 | 0.01061 | 0.05 | 4.71 | 63 |
| TTV11 | SF_a2 | 0.01063 | 0.05 | 4.70 | 63 |
| TTV11 | SJ_a0 | 0.03543 | 0.04112 | 1.16 | 249 |
| TTV11 | SJ_a1.65 | 0.02656 | 0.04894 | 1.84 | 178 |
| TTV11 | SJ_a2 | 0.01801 | 0.05 | 2.78 | 119 |
| TTV11 | SJ_a2.25 | 0.01826 | 0.05 | 2.74 | 121 |
| TTV15 | P4 | 0.03437 | 0.02858 | 0.83 | 224 |
| TTV15 | PF | 0.01105 | 0.009993 | 0.90 | 63 |
| TTV15 | PF_7400 | 0.02312 | 0.01771 | 0.77 | 147 |
| TTV15 | PF_2848 | 0.01665 | 0.01473 | 0.88 | 105 |
| TTV15 | PF_611 | 0.01154 | 0.01097 | 0.95 | 67 |
| TTV15 | SF_a0 | 0.01105 | 0.009993 | 0.90 | 63 |
| TTV15 | SF_a1.65 | 0.01176 | 0.05 | 4.25 | 64 |
| TTV15 | SF_a2 | 0.01186 | 0.03751 | 3.16 | 64 |
| TTV15 | SJ_a0 | 0.03437 | 0.02858 | 0.83 | 224 |
| TTV15 | SJ_a1.65 | 0.05894 | 0.05 | 0.85 | 339 |
| TTV15 | SJ_a2 | 0.06607 | 0.05 | 0.76 | 348 |
| TTV15 | SJ_a2.25 | 0.03118 | 0.04362 | 1.40 | 200 |

### E10. D statements (R2) and the bracket (R10)

| group | N | F3 (D0_P) | E1 (D0_P) | E2 (D0_P) | F3 (D0_N) | E1 (D0_N) | E2 (D0_N) | H2 (D0_P) | F4 AD |log10| | lower end withheld | changed vs a18 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GM:primary | 22 | 21/22 | 20/22 | 20/22 | 21/22 | 20/22 | 20/22 | 19/22 | 2.23 | yes | - |
| AP:primary | 21 | 18/21 | 12/21 | 12/21 | 14/21 | 8/21 | 9/21 | 15/21 | 0.81 | yes | - |
| TTV:primary | 11 | 6/11 | 7/11 | 6/11 | 5/11 | 5/11 | 5/11 | 9/11 | 0.70 | no | H2_A (D0_R): 5/9 -> H2_P: 9/11; F4: holds (D0_R) -> fails (D0_P) |
| GM:offmesa | 6 | 4/6 | 5/6 | 4/6 | 4/6 | 5/6 | 4/6 | 4/6 | 1.15 | yes | - |
| GM:ncalm | 7 | 7/7 | 7/7 | 7/7 | 7/7 | 7/7 | 7/7 | 7/7 | 2.45 | yes | - |
| GM:v2m | 5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 2.24 | yes | - |
| AP:pamap | 5 | 5/5 | 3/5 | 1/5 | 3/5 | 0/5 | 1/5 | 3/5 | 0.94 | yes | - |
| AP:ap1_3dep | 1 | 1/1 | 0/1 | 0/1 | 1/1 | 0/1 | 0/1 | 1/1 | 0.73 | yes | - |
| TTV:mt | 11 | 5/11 | 8/11 | 7/11 | 5/11 | 7/11 | 5/11 | 8/11 | 0.80 | no | - |
| AP:trunk | 4 | 3/4 | 3/4 | 2/4 | 0/4 | 0/4 | 0/4 | 3/4 | 0.72 | yes | - |
| GM:trunk | 5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 2.15 | yes | - |

| site | model | in (P) | below (P) | above (P) | undet. | in (N) | below (N) | above (N) | in (block) | below (block) | above (block) | verdict | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| GM | AD | 0 | 22 | 0 | 0 | 0 | 22 | 0 | 0 | 22 | 0 | withheld (R11P fails): only the upper-end reading stands | - |
| GM | P4 | 16 | 1 | 5 | 0 | 17 | 0 | 5 | 17 | 1 | 4 | withheld (R11P fails): only the upper-end reading stands | - |
| GM | P3 | 15 | 0 | 7 | 0 | 15 | 0 | 7 | 17 | 0 | 5 | withheld (R11P fails): only the upper-end reading stands | - |
| GM | A1 | 0 | 22 | 0 | 0 | 0 | 22 | 0 | 0 | 22 | 0 | withheld (R11P fails): only the upper-end reading stands | - |
| AP | AD | 0 | 21 | 0 | 0 | 0 | 21 | 0 | 0 | 21 | 0 | withheld (R11P fails): only the upper-end reading stands | - |
| AP | P4 | 0 | 0 | 21 | 0 | 0 | 0 | 21 | 0 | 0 | 21 | withheld (R11P fails): only the upper-end reading stands | - |
| AP | P3 | 1 | 0 | 20 | 0 | 1 | 0 | 20 | 1 | 0 | 20 | withheld (R11P fails): only the upper-end reading stands | - |
| AP | A1 | 0 | 16 | 5 | 0 | 0 | 16 | 5 | 0 | 16 | 5 | withheld (R11P fails): only the upper-end reading stands | - |
| TTV | AD | 0 | 7 | 4 | 0 | 0 | 7 | 4 | 0 | 7 | 4 | not consistent | - |
| TTV | P4 | 0 | 0 | 11 | 0 | 0 | 0 | 11 | 0 | 0 | 11 | not consistent | above on >= 80 % of determined tiles: the excess D is not a  |
| TTV | P3 | 0 | 0 | 11 | 0 | 0 | 0 | 11 | 0 | 0 | 11 | not consistent | above on >= 80 % of determined tiles: the excess D is not a  |
| TTV | A1 | 0 | 6 | 5 | 0 | 0 | 6 | 5 | 0 | 6 | 5 | not consistent | - |
| TTV | PF | 3 | 6 | 2 | 0 | 7 | 2 | 2 | 3 | 6 | 2 | not consistent | mixed classes: no reading applies |
| TTV | PF3 | 3 | 1 | 7 | 0 | 4 | 0 | 7 | 3 | 1 | 7 | not consistent | mixed classes: no reading applies |
| TTV | PF_7400 | 0 | 0 | 11 | 0 | 0 | 0 | 11 | 0 | 0 | 11 | not consistent | above on >= 80 % of determined tiles: the excess D is not a  |
| TTV | PF_74003 | 2 | 0 | 9 | 0 | 2 | 0 | 9 | 2 | 0 | 9 | not consistent | above on >= 80 % of determined tiles: the excess D is not a  |
| TTV | PF_2848 | 2 | 1 | 8 | 0 | 3 | 0 | 8 | 3 | 1 | 7 | not consistent | mixed classes: no reading applies |
| TTV | PF_28483 | 6 | 2 | 3 | 0 | 8 | 0 | 3 | 8 | 2 | 1 | not consistent | mixed classes: no reading applies |
| TTV | PF_611 | 4 | 4 | 3 | 0 | 8 | 0 | 3 | 5 | 4 | 2 | not consistent | mixed classes: no reading applies |
| TTV | PF_6113 | 4 | 1 | 6 | 0 | 5 | 0 | 6 | 4 | 1 | 6 | not consistent | mixed classes: no reading applies |
| TTV | SF_a0 | 3 | 6 | 2 | 0 | 7 | 2 | 2 | 3 | 6 | 2 | not consistent | mixed classes: no reading applies |
| TTV | SF_a1.65 | 1 | 1 | 9 | 0 | 2 | 0 | 9 | 1 | 1 | 9 | not consistent | above on >= 80 % of determined tiles: the excess D is not a  |
| TTV | SF_a2 | 1 | 1 | 9 | 0 | 2 | 0 | 9 | 1 | 1 | 9 | not consistent | above on >= 80 % of determined tiles: the excess D is not a  |
| TTV | SJ_a0 | 0 | 0 | 11 | 0 | 0 | 0 | 11 | 0 | 0 | 11 | not consistent | above on >= 80 % of determined tiles: the excess D is not a  |
| TTV | SJ_a1.65 | 0 | 0 | 11 | 0 | 0 | 0 | 11 | 0 | 0 | 11 | not consistent | above on >= 80 % of determined tiles: the excess D is not a  |
| TTV | SJ_a2 | 0 | 0 | 11 | 0 | 0 | 0 | 11 | 0 | 0 | 11 | not consistent | above on >= 80 % of determined tiles: the excess D is not a  |
| TTV | SJ_a2.25 | 0 | 0 | 11 | 0 | 0 | 0 | 11 | 0 | 0 | 11 | not consistent | above on >= 80 % of determined tiles: the excess D is not a  |

- Checks: {'A2_reconstruction_failed': [], 'gram_route_failed': [], 'gram_route_checked': 89, 'SF_alpha0_equals_PF': {'TTV1': True, 'TTV2': True, 'TTV3': True, 'TTV4': True, 'TTV5': True, 'TTV6': True, 'TTV7': True, 'TTV8': True, 'TTV9': True, 'TTV11': True, 'TTV15': True}}

### E11. TTV calibration supplement (section 12)

| grid (m) | median -lap (1/m) | q16 | q84 | sd | frac convex | n |
|---|---|---|---|---|---|---|
| 1 | 0.0059 | -0.0856 | 0.1078 | 0.1112 | 0.53 | 1,316 |
| 2 | 0.0090 | -0.0163 | 0.0387 | 0.0319 | 0.65 | 1,316 |
| 3 | 0.0098 | -0.0041 | 0.0270 | 0.0180 | 0.76 | 1,316 |
| 5 | 0.0099 | 0.0032 | 0.0214 | 0.0094 | 1.00 | 1,316 |
| 7 | 0.0107 | 0.0039 | 0.0198 | 0.0093 | 0.94 | 1,316 |
| 9 | 0.0106 | 0.0036 | 0.0188 | 0.0088 | 0.94 | 1,316 |

- Nose cells: 1316 (inside the M&D study polygon; convex at the 5 m grid; slope 8-30 deg (Heimsath Table 1 samples 8-25 deg); no D8 divide cell in the 5 m cell; mean 1 m D8 area < 1e3 m2); at 5 m {'n': 1316, 'p5': 0.0011866661071798035, 'p25': 0.00517114944458001, 'p50': 0.009947097778320995, 'p75': 0.01661042556762595, 'p95': 0.03041418762206831, 'frac_in_fig6_bulk': 0.7355623100303952}; implied E (D = 0.0025) {'n': 1316, 'p5': 2.966665267949509, 'p25': 12.927873611450025, 'p50': 24.867744445802487, 'p75': 41.52606391906488, 'p95': 76.03546905517078, 'D_rock_m2_per_yr': 0.0025}; (D = 0.0032) {'n': 1316, 'p5': 3.7757557955721017, 'p25': 16.453657323663666, 'p50': 31.64985656738498, 'p75': 52.85135407880984, 'p95': 96.77241516112643, 'D_rock_m2_per_yr': 0.0031818181818181815}; Heimsath Table 1 soil-covered {'median': 27.0, 'range': [21, 60], 'n': 9}; implied soil depth {'n': 1316, 'p5': 0.5480649986363805, 'p25': 26.847140391084046, 'p50': 49.14060295913472, 'p75': 77.58349989571961, 'p95': 141.58120298059953}

| grid (m) | A 3x3 p25 | p50 | p75 | A cell p50 | within Table 1 5-95 % | D-inf p50 |
|---|---|---|---|---|---|---|
| 1 | 35 | 111 | 572 | 4 | 0.06 | - |
| 2 | 136 | 660 | 1,794 | 52 | 0.20 | - |
| 3 | 241 | 1,170 | 2,578 | 162 | 0.30 | - |
| 5 | 550 | 1,825 | 6,425 | 425 | 0.41 | 2,047 |
| 10 | 3,050 | 5,350 | 10,575 | 1,350 | 0.75 | 5,438 |
| 12 | 3,420 | 7,200 | 10,944 | 1,296 | 0.75 | - |

- M&D Table 1 source area: {'n': 63, 'p5': 2010.0, 'p25': 4800.0, 'p50': 7400.0, 'p75': 10350.0, 'p95': 14860.0}

| source | variant | D/K | 68 % | m | 68 % | basins / cells |
|---|---|---|---|---|---|---|
| T3 all cells (ttv_cal) | lap_h_disk7:A100-100000 | 98.7 | [90.8, 107] | 0.324 | [0.313, 0.335] | 210,484 |
| T3 all cells (ttv_cal) | lap_h_disk7:A100-100000:m0.33 | 103.8 | [101, 107] | 0.330 | - | 210,484 |
| T3 all cells (ttv_cal) | lap_h_disk7:A100-100000:median | 78.9 | [72.9, 87.4] | 0.300 | [0.288, 0.315] | 210,484 |
| T3 all cells (ttv_cal) | lap_h_disk7:A100-100000:profiles | 131.2 | [120, 147] | 0.353 | [0.341, 0.367] | 70,569 |
| T3 all cells (ttv_cal) | lap_h_disk7:A100-100000:profiles:m0.33 | 108.9 | [106, 113] | 0.330 | - | 70,569 |
| T3 all cells (ttv_cal) | lap_h_disk7:A1000-100000 | 138.0 | [112, 174] | 0.358 | [0.335, 0.383] | 33,419 |
| T3 all cells (ttv_cal) | lap_h_disk7:A1000-100000:m0.33 | 106.4 | [102, 111] | 0.330 | - | 33,419 |
| T3 all cells (ttv_cal) | lap_h_disk7:A1000-100000:median | 116.4 | [99.3, 152] | 0.340 | [0.321, 0.373] | 33,419 |
| T3 all cells (ttv_cal) | lap_h_disk7:A1000-100000:profiles | 138.0 | [112, 174] | 0.358 | [0.335, 0.383] | 33,419 |
| T3 all cells (ttv_cal) | lap_h_disk7:A1000-100000:profiles:m0.33 | 106.4 | [102, 111] | 0.330 | - | 33,419 |
| T3 all cells (ttv_cal) | lap_h_disk7:A10-100000 | 61.4 | [57.9, 64.5] | 0.270 | [0.262, 0.278] | 642,728 |
| T3 all cells (ttv_cal) | lap_h_disk7:A10-100000:m0.33 | 93.2 | [90.6, 95.7] | 0.330 | - | 642,728 |
| T3 all cells (ttv_cal) | lap_h_disk7:A10-100000:median | 50.6 | [48.2, 53.6] | 0.250 | [0.242, 0.260] | 642,728 |
| T3 all cells (ttv_cal) | lap_h_disk7:A10-100000:profiles | 124.8 | [114, 139] | 0.348 | [0.337, 0.361] | 70,569 |
| T3 all cells (ttv_cal) | lap_h_disk7:A10-100000:profiles:m0.33 | 108.4 | [105, 112] | 0.330 | - | 70,569 |
| T3 all cells (ttv_cal) | lap_h_square11:A100-100000 | 98.4 | [90.5, 107] | 0.323 | [0.312, 0.335] | 210,557 |
| T3 all cells (ttv_cal) | lap_h_square11:A100-100000:m0.33 | 103.8 | [101, 107] | 0.330 | - | 210,557 |
| T3 all cells (ttv_cal) | lap_h_square11:A100-100000:median | 78.7 | [72.8, 87.3] | 0.300 | [0.288, 0.314] | 210,557 |
| T3 all cells (ttv_cal) | lap_h_square11:A100-100000:profiles | 130.5 | [119, 146] | 0.353 | [0.341, 0.367] | 70,589 |
| T3 all cells (ttv_cal) | lap_h_square11:A100-100000:profiles:m0.33 | 108.8 | [106, 113] | 0.330 | - | 70,589 |
| T3 all cells (ttv_cal) | lap_h_square11:A1000-100000 | 138.0 | [112, 173] | 0.358 | [0.335, 0.383] | 33,428 |
| T3 all cells (ttv_cal) | lap_h_square11:A1000-100000:m0.33 | 106.4 | [102, 111] | 0.330 | - | 33,428 |
| T3 all cells (ttv_cal) | lap_h_square11:A1000-100000:median | 116.3 | [99.1, 152] | 0.340 | [0.321, 0.373] | 33,428 |
| T3 all cells (ttv_cal) | lap_h_square11:A1000-100000:profiles | 138.0 | [112, 173] | 0.358 | [0.335, 0.383] | 33,428 |
| T3 all cells (ttv_cal) | lap_h_square11:A1000-100000:profiles:m0.33 | 106.4 | [102, 111] | 0.330 | - | 33,428 |
| T3 all cells (ttv_cal) | lap_h_square11:A10-100000 | 61.2 | [57.9, 64.4] | 0.269 | [0.262, 0.277] | 643,077 |
| T3 all cells (ttv_cal) | lap_h_square11:A10-100000:m0.33 | 93.1 | [90.6, 95.6] | 0.330 | - | 643,077 |
| T3 all cells (ttv_cal) | lap_h_square11:A10-100000:median | 50.5 | [48.1, 53.5] | 0.249 | [0.242, 0.260] | 643,077 |
| T3 all cells (ttv_cal) | lap_h_square11:A10-100000:profiles | 124.1 | [114, 139] | 0.347 | [0.336, 0.361] | 70,589 |
| T3 all cells (ttv_cal) | lap_h_square11:A10-100000:profiles:m0.33 | 108.3 | [105, 112] | 0.330 | - | 70,589 |
| GM check all cells | published | 124.0 | - | 0.350 | - | - |
| GM check all cells | perron_lap_h:A100-100000 | 174.7 | [158, 191] | 0.421 | [0.407, 0.434] | 652,288 |
| GM check all cells | perron_lap_h:A100-100000:median | 180.8 | [165, 192] | 0.427 | [0.415, 0.434] | 652,288 |
| GM check all cells | perron_lap_h:A100-100000:profiles | 245.9 | [223, 273] | 0.456 | [0.443, 0.472] | 145,936 |
| GM check all cells | perron_lap_h:A1000-100000 | 223.8 | [181, 278] | 0.447 | [0.424, 0.471] | 65,628 |
| GM check all cells | perron_lap_h:A1000-100000:median | 267.3 | [223, 319] | 0.468 | [0.447, 0.487] | 65,628 |
| GM check all cells | perron_lap_h:A1000-100000:profiles | 223.8 | [181, 278] | 0.447 | [0.424, 0.471] | 65,628 |
| GM check all cells | perron_lap_h:A10-100000 | 100.9 | [96.4, 105] | 0.359 | [0.351, 0.367] | 2,936,609 |
| GM check all cells | perron_lap_h:A10-100000:median | 95.0 | [91, 98.2] | 0.354 | [0.347, 0.359] | 2,936,609 |
| GM check all cells | perron_lap_h:A10-100000:profiles | 230.9 | [211, 256] | 0.449 | [0.436, 0.465] | 145,936 |
| GM check all cells | our_crest_disk7:A100-100000 | 184.3 | [166, 202] | 0.425 | [0.410, 0.437] | 646,243 |
| GM check all cells | our_crest_disk7:A100-100000:median | 190.9 | [175, 203] | 0.431 | [0.419, 0.439] | 646,243 |
| GM check all cells | our_crest_disk7:A100-100000:profiles | 264.8 | [240, 294] | 0.462 | [0.448, 0.478] | 145,061 |
| GM check all cells | our_crest_disk7:A1000-100000 | 236.1 | [190, 294] | 0.451 | [0.427, 0.475] | 65,499 |
| GM check all cells | our_crest_disk7:A1000-100000:median | 282.4 | [236, 336] | 0.472 | [0.452, 0.491] | 65,499 |
| GM check all cells | our_crest_disk7:A1000-100000:profiles | 236.1 | [190, 294] | 0.451 | [0.427, 0.475] | 65,499 |
| GM check all cells | our_crest_disk7:A10-100000 | 105.2 | [100, 110] | 0.361 | [0.353, 0.369] | 2,885,213 |
| GM check all cells | our_crest_disk7:A10-100000:median | 99.3 | [95.1, 103] | 0.357 | [0.349, 0.361] | 2,885,213 |
| GM check all cells | our_crest_disk7:A10-100000:profiles | 247.8 | [227, 274] | 0.455 | [0.442, 0.470] | 145,061 |
| D-inf GM | perron_lap_h:A10-100000:all | 144.1 | [115, 178] | 0.478 | [0.425, 0.526] | 4,332,750 |
| D-inf GM | perron_lap_h:A10-100000:profiles | 89.2 | [79.5, 99.5] | 0.392 | [0.379, 0.404] | 151,281 |
| D-inf GM | perron_lap_h:A100-100000:all | 167.1 | [112, 251] | 0.496 | [0.432, 0.564] | 705,152 |
| D-inf GM | perron_lap_h:A100-100000:profiles | 87.7 | [78.6, 94.6] | 0.391 | [0.377, 0.401] | 105,554 |
| D-inf GM | perron_lap_h:A1000-100000:all | 180.7 | [70, 425] | 0.504 | [0.391, 0.622] | 80,181 |
| D-inf GM | perron_lap_h:A1000-100000:profiles | 68.9 | [55.9, 81] | 0.366 | [0.342, 0.385] | 42,734 |
| D-inf GM | our_crest_disk7:A10-100000:all | 150.5 | [120, 187] | 0.481 | [0.428, 0.528] | 4,253,359 |
| D-inf GM | our_crest_disk7:A10-100000:profiles | 94.0 | [83.3, 105] | 0.396 | [0.382, 0.408] | 150,408 |
| D-inf GM | our_crest_disk7:A100-100000:all | 175.7 | [117, 263] | 0.499 | [0.435, 0.566] | 702,150 |
| D-inf GM | our_crest_disk7:A100-100000:profiles | 93.4 | [83.7, 101] | 0.396 | [0.382, 0.406] | 105,317 |
| D-inf GM | our_crest_disk7:A1000-100000:all | 189.7 | [73.6, 445] | 0.507 | [0.394, 0.625] | 80,071 |
| D-inf GM | our_crest_disk7:A1000-100000:profiles | 72.2 | [58.5, 85] | 0.369 | [0.345, 0.388] | 42,686 |
| D-inf TTV | lap_h_disk7:A10-100000:all | 85.9 | [80.7, 90.3] | 0.362 | [0.351, 0.371] | 1,006,487 |
| D-inf TTV | lap_h_disk7:A10-100000:all:m0.33 | 69.1 | [67, 71.4] | 0.330 | - | 1,006,487 |
| D-inf TTV | lap_h_disk7:A10-100000:profiles | 73.4 | [67.1, 79.2] | 0.347 | [0.334, 0.357] | 67,215 |
| D-inf TTV | lap_h_disk7:A10-100000:profiles:m0.33 | 65.2 | [62.6, 68.1] | 0.330 | - | 67,215 |
| D-inf TTV | lap_h_disk7:A100-100000:all | 113.5 | [100, 127] | 0.393 | [0.377, 0.408] | 222,157 |
| D-inf TTV | lap_h_disk7:A100-100000:all:m0.33 | 68.1 | [65.6, 70.9] | 0.330 | - | 222,157 |
| D-inf TTV | lap_h_disk7:A100-100000:profiles | 103.7 | [91.2, 117] | 0.387 | [0.370, 0.401] | 52,203 |
| D-inf TTV | lap_h_disk7:A100-100000:profiles:m0.33 | 65.4 | [62.8, 68.6] | 0.330 | - | 52,203 |
| D-inf TTV | lap_h_disk7:A1000-100000:all | 233.3 | [170, 318] | 0.468 | [0.433, 0.501] | 29,844 |
| D-inf TTV | lap_h_disk7:A1000-100000:all:m0.33 | 65.4 | [62.2, 68.6] | 0.330 | - | 29,844 |
| D-inf TTV | lap_h_disk7:A1000-100000:profiles | 161.1 | [116, 217] | 0.432 | [0.396, 0.465] | 19,661 |
| D-inf TTV | lap_h_disk7:A1000-100000:profiles:m0.33 | 62.7 | [59.5, 66.1] | 0.330 | - | 19,661 |
| SI kink GM | A_t=kink:perron_lap_h:selected | 176.4 | [109, 258] | 0.433 | [0.371, 0.479] | 182 |
| SI kink GM | A_t=kink:perron_lap_h:all | 144.1 | [94, 154] | 0.401 | [0.346, 0.410] | 2,447 |
| SI kink GM | A_t=kink:our_crest:selected | 184.0 | [121, 306] | 0.435 | [0.378, 0.497] | 186 |
| SI kink GM | A_t=kink:our_crest:all | 151.7 | [97.9, 163] | 0.404 | [0.348, 0.413] | 2,447 |
| SI kink GM | A_t=100.0:perron_lap_h:selected | 1426.6 | [1.03e+03, 2.7e+03] | 0.859 | [0.798, 0.978] | 1,768 |
| SI kink GM | A_t=100.0:perron_lap_h:all | 174.1 | [85.9, 175] | 0.425 | [0.300, 0.427] | 24,583 |
| SI kink GM | A_t=100.0:our_crest:selected | 1591.3 | [1.19e+03, 2.99e+03] | 0.872 | [0.815, 0.995] | 1,756 |
| SI kink GM | A_t=100.0:our_crest:all | 183.2 | [83.9, 184] | 0.427 | [0.289, 0.431] | 24,571 |
| SI kink GM | A_t=400.0:perron_lap_h:selected | 2039.5 | [1.33e+03, 5.34e+03] | 0.815 | [0.751, 0.960] | 314 |
| SI kink GM | A_t=400.0:perron_lap_h:all | 148.2 | [106, 175] | 0.407 | [0.360, 0.428] | 5,424 |
| SI kink GM | A_t=400.0:our_crest:selected | 2197.7 | [1.34e+03, 4.66e+03] | 0.821 | [0.749, 0.933] | 319 |
| SI kink GM | A_t=400.0:our_crest:all | 154.7 | [111, 181] | 0.409 | [0.363, 0.428] | 5,424 |
| SI kink GM | A_t=1000.0:perron_lap_h:selected | 488.8 | [151, 867] | 0.559 | [0.412, 0.635] | 115 |
| SI kink GM | A_t=1000.0:perron_lap_h:all | 167.8 | [122, 209] | 0.420 | [0.384, 0.448] | 1,525 |
| SI kink GM | A_t=1000.0:our_crest:selected | 523.3 | [188, 979] | 0.565 | [0.434, 0.647] | 117 |
| SI kink GM | A_t=1000.0:our_crest:all | 175.6 | [128, 224] | 0.423 | [0.387, 0.455] | 1,525 |
| SI kink TTV | A_t=kink:crest:selected | 179.0 | [139, 3.3e+04] | 0.454 | [0.405, 1.364] | 282 |
| SI kink TTV | A_t=kink:crest:all | 130.0 | [91.6, 190] | 0.376 | [0.317, 0.435] | 3,828 |
| SI kink TTV | A_t=100.0:crest:selected | 5843.9 | [3.3e+03, 8.69e+03] | 1.088 | [0.979, 1.163] | 477 |
| SI kink TTV | A_t=100.0:crest:all | 74.4 | [67.4, 163] | 0.284 | [0.265, 0.418] | 6,295 |
| SI kink TTV | A_t=400.0:crest:selected | 1438.4 | [385, 3.43e+03] | 0.732 | [0.528, 0.858] | 50 |
| SI kink TTV | A_t=400.0:crest:all | 80.9 | [53.2, 119] | 0.297 | [0.242, 0.353] | 1,763 |
| SI kink TTV | A_t=1000.0:crest:selected | 596.0 | [343, 4.11e+03] | 0.571 | [0.502, 0.831] | 13 |
| SI kink TTV | A_t=1000.0:crest:all | 99.8 | [61.9, 146] | 0.320 | [0.260, 0.368] | 592 |
| SI + 2008 roll-off GM | perron_180m:perron_lap_h:main_stem_profiles:selected | 195.1 | [133, 238] | 0.440 | [0.393, 0.460] | 16 |
| SI + 2008 roll-off GM | perron_180m:perron_lap_h:main_stem_profiles:all | 94.6 | [80.1, 126] | 0.358 | [0.343, 0.387] | 53 |
| SI + 2008 roll-off GM | perron_180m:perron_lap_h:basin_cells:selected | 75.8 | [75.8, 75.8] | 0.307 | [0.307, 0.307] | 1 |
| SI + 2008 roll-off GM | perron_180m:perron_lap_h:basin_cells:all | 150.1 | [115, 170] | 0.404 | [0.370, 0.418] | 53 |
| SI + 2008 roll-off GM | perron_180m:our_crest:main_stem_profiles:selected | 222.0 | [140, 263] | 0.451 | [0.398, 0.467] | 15 |
| SI + 2008 roll-off GM | perron_180m:our_crest:main_stem_profiles:all | 104.6 | [87.8, 139] | 0.366 | [0.349, 0.395] | 53 |
| SI + 2008 roll-off GM | perron_180m:our_crest:basin_cells:selected | 81.7 | [81.7, 81.7] | 0.312 | [0.312, 0.312] | 1 |
| SI + 2008 roll-off GM | perron_180m:our_crest:basin_cells:all | 157.8 | [121, 179] | 0.407 | [0.372, 0.422] | 53 |
| SI + 2008 roll-off GM | our_rolloff:perron_lap_h:main_stem_profiles:selected | 124.4 | [124, 124] | 0.413 | [0.413, 0.413] | 1 |
| SI + 2008 roll-off GM | our_rolloff:perron_lap_h:main_stem_profiles:all | 78.8 | [34.3, 102] | 0.336 | [0.277, 0.359] | 12 |
| SI + 2008 roll-off GM | our_rolloff:perron_lap_h:basin_cells:selected | 75.8 | [75.8, 75.8] | 0.307 | [0.307, 0.307] | 1 |
| SI + 2008 roll-off GM | our_rolloff:perron_lap_h:basin_cells:all | 106.7 | [97.3, 127] | 0.357 | [0.347, 0.380] | 12 |
| SI + 2008 roll-off GM | our_rolloff:our_crest:main_stem_profiles:selected | 133.7 | [134, 134] | 0.419 | [0.419, 0.419] | 1 |
| SI + 2008 roll-off GM | our_rolloff:our_crest:main_stem_profiles:all | 79.4 | [35.7, 101] | 0.335 | [0.281, 0.359] | 12 |
| SI + 2008 roll-off GM | our_rolloff:our_crest:basin_cells:selected | 81.7 | [81.7, 81.7] | 0.312 | [0.312, 0.312] | 1 |
| SI + 2008 roll-off GM | our_rolloff:our_crest:basin_cells:all | 111.4 | [102, 134] | 0.359 | [0.349, 0.382] | 12 |
| SI + 2008 roll-off TTV | md_table1_median:crest:main_stem_profiles:all | 84.4 | [70.3, 101] | 0.300 | [0.277, 0.322] | 84 |
| SI + 2008 roll-off TTV | md_table1_median:crest:main_stem_profiles:selected | 193.1 | [156, 293] | 0.433 | [0.401, 0.496] | 15 |
| SI + 2008 roll-off TTV | md_table1_median:crest:basin_cells:all | 80.5 | [70, 89.3] | 0.288 | [0.266, 0.305] | 90 |
| SI + 2008 roll-off TTV | md_table1_median:crest:basin_cells:selected | 232.5 | [174, 680] | 0.526 | [0.412, 0.700] | 2 |
| SI + 2008 roll-off TTV | gm_perron_180m:crest:main_stem_profiles:all | 74.8 | [56.4, 99.3] | 0.298 | [0.267, 0.329] | 22 |
| SI + 2008 roll-off TTV | gm_perron_180m:crest:main_stem_profiles:selected | 151.0 | [122, 192] | 0.377 | [0.351, 0.410] | 10 |
| SI + 2008 roll-off TTV | gm_perron_180m:crest:basin_cells:all | 97.3 | [84.3, 115] | 0.322 | [0.301, 0.343] | 27 |
| SI + 2008 roll-off TTV | gm_perron_180m:crest:basin_cells:selected | 174.4 | [174, 174] | 0.412 | [0.412, 0.412] | 1 |
| SI + 2008 roll-off TTV | our_rolloff:crest:main_stem_profiles:all | 302.2 | [187, 302] | 0.446 | [0.401, 0.446] | 2 |
| SI + 2008 roll-off TTV | our_rolloff:crest:main_stem_profiles:selected | 228.5 | [228, 228] | 0.418 | [0.418, 0.418] | 1 |
| SI + 2008 roll-off TTV | our_rolloff:crest:basin_cells:all | 128.5 | [129, 139] | 0.367 | [0.365, 0.418] | 2 |

- Roll-off (Perron et al. 2008 procedure) per GM tile: GM1 168 m, GM2 196 m, GM3 312 m, GM4 425 m, GM5 267 m, GM6 312 m, GM7 425 m; TTV window 863 m

| set | basins | cells | plateau rule lap_h | bins (m2) | signed mean A|grad z| <= 1 | 68 % | <= 3 | geomean convex <= 1 | ratio to Perron (signed <= 1) |
|---|---|---|---|---|---|---|---|---|---|
| GM_ncalm_At180m | 53 | 3,146,189 | -0.003262 | [0.001, 0.0056] | -0.00311 | [-0.00344, -0.00271] | -0.003479 | -0.007201 | 0.26 |
| TTV_At7400 | 97 | 1,370,471 | -0.005472 | [0.0056, 0.032] | -0.002634 | [-0.00283, -0.00245] | -0.002505 | -0.009992 | - |
| TTV_At180m | 28 | 1,255,473 | -0.004515 | [0.01, 0.056] | -0.002287 | [-0.00261, -0.00203] | -0.002333 | -0.01006 | - |

- Synthesis (T4): {'ridge_soil_covered_median': {'disk7': 0.0016437209718993453, 'square11': 0.001640475008464732}, 'ridge_soil_covered_mean': {'disk7': 0.0021172208403477167, 'square11': 0.0021130398257179473}, 'basin_creek1': {'disk7': 0.0039571060434613866, 'square11': 0.003949291687044725}, 'basin_creek2': {'disk7': 0.006209612560508638, 'square11': 0.006197350031977877}, 'config_P0': {'disk7': 0.004687648697638874, 'square11': 0.004678391690806829}}; D from the 5 m nose median: {'ridge_soil_covered_median': 0.002714359565143173, 'ridge_soil_covered_mean': 0.003496273843167956, 'basin_creek1': 0.006534569323492824, 'basin_creek2': 0.010254247246096432, 'config_P0': 0.007740951352445346}; K from S* (all-cells route): {'m0.33': {'rho0.5': 2.4084333879048227e-05, 'rho_measured': 3.0652788573334104e-05, 'D_over_K': 103.8019159074536}, 'm_free': {'rho0.5': 2.53411915562168e-05, 'rho_measured': 3.22524256170032e-05, 'D_over_K': 98.65360886657636, 'm': 0.3236878878229511}}

| tile | U fit | K0 cfg | P4 D (cfg) | P4 D @ 65 | P4 D @ 27 | P4 K (cfg) | P4 K @ 65 | AD D (cfg) | P4 A_crit |
|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 7.7e-05 | 1.124e-05 | 0.04388 | 0.03704 | 0.01539 | 1.235e-05 | 1.043e-05 | 0.0016 | 59,499 |
| TTV2 | 7.7e-05 | 1.581e-05 | 0.0166 | 0.01401 | 0.005821 | 1.645e-05 | 1.389e-05 | 0.0002365 | 9,131 |
| TTV3 | 7.7e-05 | 2.636e-05 | 0.05 | 0.04221 | 0.01753 | 3.733e-05 | 3.151e-05 | 9.232e-06 | 89,124 |

### E12. Figures

- `figures/a20/binned_curves_r7.png` (0.26 MB)
- `figures/a20/GM_D_bracket_intervals.png` (0.09 MB)
- `figures/a20/AP_D_bracket_intervals.png` (0.08 MB)
- `figures/a20/TTV_D_bracket_intervals.png` (0.06 MB)
- `figures/a20/GM_D0_vs_window_gallery.png` (0.28 MB)
- `figures/a20/AP_D0_vs_window_gallery.png` (0.26 MB)
- `figures/a20/TTV_D0_vs_window_gallery.png` (0.12 MB)
- `figures/a20/plateau/GM_plateau_gallery.png` (4.26 MB)
- `figures/a20/plateau/AP_plateau_gallery.png` (3.08 MB)
- `figures/a20/plateau/TTV_plateau_gallery.png` (1.87 MB)
- `figures/a20/heads_area_slope.png` (0.14 MB)
- `figures/a20/heads/heads_qc_page1.png` (0.43 MB): picks 1-16
- `figures/a20/heads/heads_qc_page2.png` (0.41 MB): picks 17-32
- `figures/a20/heads/heads_qc_page3.png` (0.42 MB): picks 33-48
- `figures/a20/heads/heads_qc_page4.png` (0.42 MB): picks 49-64
- `figures/a20/model_heads/TTV1_model_heads.png` (3.11 MB): TTV1
- `figures/a20/model_heads/TTV2_model_heads.png` (2.99 MB): TTV2
- `figures/a20/model_heads/TTV3_model_heads.png` (2.83 MB): TTV3
- `reports/v2/figures/a20/plateau/<tile>_plateau.png`: 73 per-tile plateau maps

### E13. Slurm accounting

| job name | tasks | core-h used | core-h of limits | states | job ids |
|---|---|---|---|---|---|
| a20_dstat | 197 | 16.28 | 98.0 | CANCELLED by 313293 1, COMPLETED 196 | 53114976 53116428 53121081 53131913 |
| a20_figs | 1 | 0.01 | 1.0 | COMPLETED 1 | 53207232 |
| a20_fixed | 11 | 0.19 | 5.5 | COMPLETED 11 | 53110870 |
| a20_heads | 2 | 0.01 | 1.3 | COMPLETED 2 | 53109901 53110678 |
| a20_headsqc | 1 | 0.01 | 0.5 | COMPLETED 1 | 53112248 |
| a20_lap_h_basins | 1 | 0.04 | 1.0 | COMPLETED 1 | 53209119 |
| a20_modelheads | 1 | 0.02 | 1.0 | COMPLETED 1 | 53207230 |
| a20_rules | 1 | 0.03 | 1.0 | COMPLETED 1 | 53207228 |
| a20_site | 3 | 0.04 | 4.0 | COMPLETED 2, CANCELLED by 313293 1 | 53110743 53114930 53116426 |
| a20_sj_TTV1 | 4 | 12.23 | 33.0 | COMPLETED 4 | 53109497 53112089 |
| a20_sj_TTV11 | 4 | 12.69 | 40.3 | COMPLETED 4 | 53109547 53112147 |
| a20_sj_TTV15 | 4 | 12.13 | 35.9 | COMPLETED 4 | 53109551 53112155 |
| a20_sj_TTV2 | 4 | 11.50 | 32.3 | COMPLETED 4 | 53109501 53112097 |
| a20_sj_TTV3 | 4 | 10.23 | 31.7 | COMPLETED 4 | 53109502 53112102 |
| a20_sj_TTV4 | 4 | 11.04 | 34.0 | COMPLETED 4 | 53109505 53112112 |
| a20_sj_TTV5 | 4 | 12.03 | 36.3 | COMPLETED 4 | 53109516 53112120 |
| a20_sj_TTV6 | 4 | 12.55 | 36.9 | COMPLETED 4 | 53109519 53112128 |
| a20_sj_TTV7 | 4 | 10.70 | 36.4 | COMPLETED 4 | 53109524 53112135 |
| a20_sj_TTV8 | 4 | 10.78 | 39.5 | COMPLETED 4 | 53109527 53112140 |
| a20_sj_TTV9 | 4 | 11.15 | 34.6 | COMPLETED 4 | 53109537 53112145 |
| a20_sstar_dinf | 1 | 0.25 | 1.0 | COMPLETED 1 | 53205138 |
| a20_sstar_perron | 1 | 0.04 | 1.5 | COMPLETED 1 | 53206590 |
| a20_sstar_perron2 | 2 | 0.08 | 2.0 | COMPLETED 2 | 53207483 53208716 |
| a20_tests | 1 | 0.14 | 2.0 | COMPLETED 1 | 53108805 |
| a20_tile_test | 1 | 0.01 | 0.7 | COMPLETED 1 | 53110031 |
| a20_tiles | 208 | 2.85 | 103.5 | COMPLETED 182, CANCELLED by 313293 26 | 53110736 53114920 53116425 |
| a20_ttv_cal | 4 | 0.73 | 4.0 | COMPLETED 4 | 53134178 53135302 53136496 53203734 |

Core-hours used: **147.7**; CPU-weighted limits: 619.0; peak MaxRSS 5.2 GB.
