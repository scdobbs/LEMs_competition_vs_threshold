# [2026-11-23 12:45 PST] Addendum 21: the Tennessee Valley channel-head test

Directive: `reports/directives/2026-11-23T1030PST_v2_addendum_21_ttv_channel_head_test.md` (outline fixed by its section 7). Branch `v2.3-tiles`
(from 8e2f706, the addendum 20 report). **No package code was changed**; one site config changed (decision 1.1, section 1). Every number is tied to
the run directory `$GROUP_HOME/lemcvt_results/20261123T184712Z_a21_ttv_channel_head_test/` or to a Slurm job (Appendix F14). Scripts
`reports/v2/scripts/a21_*.py` and `make_a21_tables.py`; figures `reports/v2/figures/a21/`. **This is the addendum's stop.** The appendices are
machine-generated.

## Summary

| item | result |
| --- | --- |
| 1 state | TTV config U → 6.5 × 10⁻⁵ (U_legacy_P0, E_ridge added); tests pass; the mask route and the rescaling are exact on every tile (relative 0 and ≤ 6 × 10⁻¹⁵). One defect found and corrected during the work (spur pruning in the Perron sensitivity; tiles rerun). The 12 m lattice has a phase: A_F is 6,120 m² on the full-site lattice, 4,968-6,984 over 16 phases, 7,200 on the addendum 20 window lattice. |
| 2 values | K₀ 0.37-0.93 × K_cal; D_form 1.2-2.1 × D_phys (median 1.6; U/E_ridge = 2.4 because the crest curvature exceeds E_ridge/D_phys); D_relief 3.6-6.4 × D_form (median 4.3), L_H 115-173 m. |
| 3 heads | **H0b holds** (b 1.75 [1.09, 2.39], picks at 0.62 × M&D's relation; fails only on the 10 m grid). **H0 at α = 2 fails narrowly on the count** (median distance 12.6 m, ratio 1.57 against ≤ 1.5; the excess on TTV1); **at α = 0 and α = 1.65 it holds** (13.0 m, 1.43; 12.7 m, 1.38). 69-74 % of M&D's traced channel length lies within 15 m of a criterion channel. |
| 4 fixed | **Both F0 predictions hold on 11 of 11:** F-relief fits best, F-phys worst (crests 1.6 × too sharp; 30 × AD's misfit). F-relief, with no free parameter, is 0.95 × AD's misfit at the median (beats AD on 6 of 11) and 2.1 × SJ-C's. |
| 5 fits | **K1 holds for PF-A, PF-C, SJ-C and P4** (K/K₀ 0.65-1.42 on 11 of 11; medians 0.81-0.96), **fails for AD** (1 of 11; median 10 ×). **M1: SJ-C beats AD on 10 of 11** (holds; median 0.42 ×), PF-C on 9 (holds), PF-A on 8 (fails), P4 on 10. SJ-C's free C is 0.5-6.4 × C_F; its held-out MSE is below P4's on 3 of 11. D at the bound: SJ-C and P4 on 3-4 tiles, PF-C on 1, AD on 2. |
| 6 model heads | **H1 at α = 2:** the observed surface, **F2-relief** (22.6 m, 1.36) locate the mapped heads; PF-C (27.5 m, 1.36) and PF-A (19.0 m, 1.64) each miss one condition; **SJ-C (44 m, 1.80), P4 (47 m, 2.41) and AD (41 m, 2.28) miss both. H2 fails** (SJ-C's distance is not smaller than AD's; its count is closer). |
| 7 D | **DX holds for PF-A** (D within 0.67-1.5 × D_relief on 9 of 11; median 0.94), **fails for PF-C (6), SJ-C (1) and P4 (3)**, all above the relief. The decomposition (D_relief/D_form)(U/E_ridge) is 8.7-15 (median 10.6) against D_fit/D_phys of 6.6 (PF-A), 11 (PF-C), 12 (SJ-C) and 15 (P4) at the median: the order-of-magnitude excess over D_phys is nonlinear transport × non-steady state; the free-criterion fits add a further 1.4-1.7 × by making hillslopes longer than the real ones. AD's D is 0.003-17 × D_phys. |
| cost | 10.8 core-hours used (58 of limits; largest submission 34), ~890 of Steve's approved 1,000-2,000 to date. |

## 1. State check, deviations, decisions as applied

**At the start (2026-11-23 10:40 PST, sh02-ln01).** `v2.3-tiles` = `origin` = 8e2f706; no job running; `/etc/agents/AGENTS.md` and its topic files read.
Tests on `serc` at 8e2f706 with the TTV config change in the working tree (53447081): ruff and mypy clean, pytest exit 0. The paths of directive 0.4
exist; the heads CSV checksum (`ac87de3d…e702b6`) is checked in code before use. The directive and the repo did not disagree anywhere that
required a stop.

**Decisions as applied.**
1. **Config (decision 1.1).** `configs/sites/TTV.yaml`: `U: 6.5e-05` with the Heimsath et al. (1999) Table 1 source line; `U_legacy_P0: 7.7e-05`
   ("Heimsath 1999 P₀, not a lowering rate"); `E_ridge: 2.7e-05`. The loader resolves K₀ = U/k_s and D₀ = U/C_ht from the site U, so every TTV
   tile now carries K₀ × 65/77 (checked: TTV2 K₀ 1.334 × 10⁻⁵ = 0.8442 × the addendum 18 value). Nothing in `src/` or `tests/` reads a TTV key
   (grep); the regression-gate tiles GM1 (v2m) and AP2 (PAMAP) resolve to exactly their addendum 18 U, K₀, D₀ (checked by loading both configs),
   so the gate is unaffected without rerunning it. The tile configs' informational `measured.E: 7.7e-05` lines were left as written (they record
   the addendum 17 measurement run; the model's U comes from the site file).
2. **Rescaling (decision 1.1).** The partitioned reduction Z = a + (U/K) b + (U/D) c and the AD reduction Z = h(r) + (U/D) g(r) depend on U only
   through U/K and U/D, so a fit at U = 7.7 × 10⁻⁵ converts to U = 6.5 × 10⁻⁵ by K, D × 65/77 with the MSE unchanged. Checked by one direct
   evaluation per model type **on every tile** (P4: one exact reduction at its A_crit; AD: one LU at its r): Appendix F6. New free fits (PF-A,
   PF-C, SJ-C) use the addendum 18 box in (U/K, U/D), i.e. the configured K and D bounds × 65/77 (D upper bound 0.0422 m²/yr), so they are
   exactly comparable with the rescaled P4 and AD. Bound hits are reported (section 5).
3. **The mask route (decision 1.3).** `PartitionedProfiler` already takes a per-cell statistic in place of the area (`channel_stat`, addendum 8);
   a 0/1 mask with the threshold 0.5 is an arbitrary channel network. No entry point was added. On TTV2 (and every tile) the mask A ≥ 7,400 m²
   gives the same channel set and the same MSE as the area route at A_crit = 7,400 m² (relative difference 0, max |Δb| = 0): Appendix F6.
4. **Lattices (decisions 1.2-1.4).** The 12 m grid of decisions 1.2 and 1.3 is the block mean of the stage B full-site 1 m grid from its origin
   (529,495 / 4,198,505), pit-filled and D8-routed with the stage B outlet convention, so areas at tile edges are full-grid areas. The tiles are
   offset 1-9 m from that lattice. Decision 1.4 (the same extraction from every surface) uses each surface's own lattice (the tile origin, 83 × 83
   full blocks, the 4 m remainder dropped), so the observed lidar and the model surfaces are treated identically. H0 is decided on the full-site
   network; the tile-lattice extraction of the observed surface, which is the H1 baseline, is reported beside it.
5. **Seeding (decision 1.3).** A 12 m head cell is seeded at the 1 m cell of largest full-grid D8 area in its block and traced downstream on the
   full-grid D8; for the per-tile model the trace is reduced to its first cell inside the tile (the entry cell) and completed on the tile's own D8
   (the tile grids are cut from the same routing: the tile area equals the full-grid area at every seed inside the tile, Appendix F5). Candidate
   12 m cells of a tile are those whose 12 m D8 path enters the tile box with a 36 m margin; every candidate's 1 m path finished (none hit the
   60,000-step cap).
6. **S₁₂** is measured on the pit-filled 12 m surface (the one that was routed) from the cell to its largest-area donor; the two-cell sensitivity
   continues to the donor's largest-area donor.
7. **Lattice phase (added after Part A, stated here).** The 12 m block lattice has a phase, and A₁₂ at a pick depends on which blocks its hollow
   falls into: the full-site lattice gives a median 6,120 m² at the picks, the addendum 20 window lattice (7 m east, 5 m south of a full-site block
   boundary) gives the 7,200 m² that addendum 20 reported. A 16-phase sensitivity was run (`a21_phase.py`, Appendix F4): A_F 4,968-6,984 m² (median
   6,228), C_F 671-896 m² (median 817); H0b holds on 13 of 16 phases. The decision 1.2 values are those of the full-site lattice as decided; the
   phase spread is carried as a sensitivity through section 8.
8. **The Perron et al. (2012) sensitivity** follows the paper's Methods as the directive condenses them (Gaussian σ = 10 m; contour curvature above
   the mean + 2 sd; 8-connected clusters of ≥ 10 cells; flow paths from each cluster's largest-area cell to the tile edge; closing with a 10 m disk;
   thinning; spurs under 10 m pruned; heads = skeleton end points off the tile edge). The paper's extra condition that a cluster hold a high-area
   cell is not in the directive and is not applied; D8 on the surface replaces the paper's D-infinity. Thinning is a Zhang-Suen implementation in
   the script (scikit-image is not in the environment).
9. **Model surfaces.** Channel cells whose D8 path does not reach the tile boundary have no χ and no model elevation (they are outside every MSE,
   as in addenda 8-20); for the hillshade, routing and residual maps those cells take the observed elevation. Their counts are in Appendix F11.
10. **D_relief pixels.** On TTV1-TTV3 the mapped-head network exists only inside the M&D polygon, so D_relief there uses the crest pixels inside
    the polygon; on the other eight tiles it uses all counted crest pixels and the observed criterion network at α = 0 and A_F (flagged, as the
    directive says). Both are in Appendix F8.
11. **Hold-out.** The addendum 18 basin halves take the mask route without changes (the halves are cell sets; the reduction is agnostic to how the
    channel set was made), so held-out MSE is reported for SJ-C against P4 (section 5).
12. **Cost.** SJ-C was timed on TTV2 first (pilot 53447085: 7.0 s per evaluation → 1.55 h per run; the 11 runs were submitted with a 3.1 h limit each,
    34 core-hours of limits in one submission). Totals in Appendix F14.

**Deviations.** Every one listed, none silent.
1. **A defect in the Perron sensitivity, found and corrected during the work.** The spur-pruning step of the valley-head skeleton took every
   skeleton pixel with three or more 8-neighbours as a junction; the staircase corners of a one-pixel diagonal line satisfy that, so the walk from
   each end point stopped within a few cells and the end branch was pruned: on TTV2 only 2 of 21 skeleton tips survived (diagnostic job
   53449145). The junction test was changed to the crossing number (end point 1, line 2, junction ≥ 3; synthetic test in the session, diagnostic
   53449842: 13 tips). The 11 tiles were rerun (53450470 supersedes 53449079; every other number is identical, the criterion sets are
   recomputed from the same code) and the SJ-C extraction was redone from the stored surfaces (53451471: the criterion counts match the first
   pass on 11 of 11). Only the Perron columns changed.
2. The first TTV2 tile test (53447084) failed on a call that passed one basin half to a helper expecting both; fixed before any tile ran.
3. The post-processing job's appendix step (53451472, "FAILED" in the accounting) stopped on a missing key of the in-polygon relief record;
   the rules and figures of that job are the ones used; the appendix was regenerated with the fix (a light JSON step, run on the login node).
4. **n_channel(C) is not monotone in C on TTV4, TTV6 and TTV9** (Appendix F10): with heads-only seeding, a 12 m head cell that becomes a non-head
   when its upslope neighbour crosses C can have a different 1 m thread than that neighbour. The C-range bisection assumes monotonicity; the
   range ends are reported with their cell counts and the profile itself is evaluated at every grid value regardless.
5. The Perron (2012) condensation omits the paper's high-area condition (as the directive specifies), uses D8 for the area and a Zhang-Suen
   thinning written in the script (scikit-image is not in the environment). No verdict rests on it.
6. **Lattice phase** (item 7 above) was examined after the Part A result; it is reported as a sensitivity and changes no decision.
7. H0 and H1 pool TTV1-TTV3 by summing head and pick counts inside the polygon and pooling the pick distances (61 picks; the 3 picks outside
   the 11 primary tiles are not on TTV1-TTV3).

## 2. Parameter values per tile (the section 1.1 table, filled)

| tile | U | K0 (m^0.34/yr) | K0/K_cal | C_crest (1/m) | D_phys | D_form | D_form/D_phys | D_relief | 68 % | network | L_H (m) | R (m) | D_relief/D_form | D_crest at E_ridge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 6.5e-05 | 9.49e-06 | 0.40 | 0.0159 | 0.0025 | 0.0041 | 1.63 | 0.0175 | [0.0166, 0.0185] | mapped | 171 | 50.0 | 4.29 | 0.0017 |
| TTV2 | 6.5e-05 | 1.33e-05 | 0.56 | 0.0149 | 0.0025 | 0.0044 | 1.74 | 0.0159 | [0.0153, 0.0165] | mapped | 157 | 49.3 | 3.65 | 0.0018 |
| TTV3 | 6.5e-05 | 2.23e-05 | 0.93 | 0.0147 | 0.0025 | 0.0044 | 1.77 | 0.0230 | [0.0217, 0.0268] | mapped | 173 | 42.3 | 5.20 | 0.0018 |
| TTV4 | 6.5e-05 | 1.36e-05 | 0.57 | 0.0124 | 0.0025 | 0.0053 | 2.10 | 0.0194 | [0.0180, 0.0208] | A (flagged) | 153 | 39.8 | 3.69 | 0.0022 |
| TTV5 | 6.5e-05 | 1.84e-05 | 0.77 | 0.0223 | 0.0025 | 0.0029 | 1.17 | 0.0186 | [0.0179, 0.0197] | A (flagged) | 129 | 32.0 | 6.38 | 0.0012 |
| TTV6 | 6.5e-05 | 1.05e-05 | 0.44 | 0.0139 | 0.0025 | 0.0047 | 1.87 | 0.0207 | [0.0200, 0.0213] | A (flagged) | 160 | 39.5 | 4.42 | 0.0019 |
| TTV7 | 6.5e-05 | 9.11e-06 | 0.38 | 0.0163 | 0.0025 | 0.0040 | 1.60 | 0.0144 | [0.0138, 0.0150] | A (flagged) | 115 | 32.1 | 3.60 | 0.0017 |
| TTV8 | 6.5e-05 | 9.71e-06 | 0.40 | 0.0176 | 0.0025 | 0.0037 | 1.48 | 0.0174 | [0.0168, 0.0180] | A (flagged) | 131 | 31.5 | 4.72 | 0.0015 |
| TTV9 | 6.5e-05 | 8.89e-06 | 0.37 | 0.0212 | 0.0025 | 0.0031 | 1.23 | 0.0142 | [0.0125, 0.0157] | A (flagged) | 116 | 33.4 | 4.62 | 0.0013 |
| TTV11 | 6.5e-05 | 9.79e-06 | 0.41 | 0.0167 | 0.0025 | 0.0039 | 1.56 | 0.0182 | [0.0174, 0.0188] | A (flagged) | 147 | 37.2 | 4.68 | 0.0016 |
| TTV15 | 6.5e-05 | 1.16e-05 | 0.48 | 0.0141 | 0.0025 | 0.0046 | 1.85 | 0.0182 | [0.0173, 0.0191] | A (flagged) | 144 | 37.4 | 3.94 | 0.0019 |


Reading the table (U = 6.5 × 10⁻⁵ m/yr, m = 0.33 everywhere):
- **K₀ = U/k_s** is 0.9-2.2 × 10⁻⁵ m^0.34/yr, **0.37-0.93 × K_cal** (median 0.43). K_cal (2.4 × 10⁻⁵) came from Perron's S* method on the M&D
  polygon (addendum 20 section 12.4) and is uncertain by about × 2; K₀ sits at the low side of that range on every tile.
- **D_form = U/|C_crest|** is 0.0029-0.0053 m²/yr (median 0.0041), **1.2-2.1 × D_phys** (median 1.6). The planner's figure of 2.4 (= U/E_ridge)
  assumed that the measured crest curvature equals E_ridge/D_phys = 0.0108 m⁻¹; the measured crest curvature is 0.0124-0.0223 m⁻¹ (median
  0.0159), so E_ridge/|C_crest| is 0.0012-0.0022 m²/yr, 0.5-0.9 × D_phys (last column). The physical check of addendum 20 (section 12.3a) was
  made on the Heimsath nose cells at 5 m, where the curvature median is 0.0099 m⁻¹; the crest set at 11 m is sharper, so D_form/D_phys is
  smaller than U/E_ridge by that factor. Both numbers are in the table.
- **D_relief = U L_H²/(2R)** is 0.0142-0.0230 m²/yr (median 0.0182; 68 % intervals ± 3-8 %), **3.6-6.4 × D_form** (median 4.3), with hillslope
  lengths of 115-173 m to the head networks. On TTV1-TTV3 it is traced to the mapped-head network from the crest pixels inside the M&D
  polygon (1,879-2,187 pixels per tile); the other eight tiles use the observed α = 0 criterion network and are flagged. On TTV1-TTV3 the
  criterion network gives D_relief within 10 % of the mapped-head value (Appendix F8), so the flag does not change the picture.
- **The field thresholds:** A_F = 6,120 m² and C_F = 876 m² (α = 2) on the full-site lattice (section 3).

## 3. Map-scale heads (decision 1.2): measurements, fits, rules H0b and H0

**Measurements (12 m grid, full-site lattice; Appendix F2).** At the 64 picks the largest area in the 3 × 3 block has quartiles 3,852 / **6,120** /
11,412 m², 78 % inside Table 1's 5-95 % range (2,010-14,860 m²); D-infinity on the same grid gives a median 5,663 m². S₁₂ up the largest-area
donor has quartiles 0.30 / **0.35** / 0.43 (M&D's Table 1 slopes: 0.24-0.78, median 0.49); the two-cell slope is 0.36 at the median (every chain
reached two cells). On the 10 m grid the median area is 5,050 m² (70 % inside the Table 1 range).

**Fits (log₁₀ A = log₁₀ a − b log₁₀ S, OLS on log A, 10,000-draw bootstrap, seed 20261123; figure `figures/a21/heads_area_slope_12m.png`).**

| fit | n | a (m2) | b | b 95 % | R2 | median A/(1978 S^-1.65) | share in 0.5-2 | H0b |
|---|---|---|---|---|---|---|---|---|
| table1 | 63 | 1978 | 1.647 | [1.26, 2.00] | 0.60 | 1.01 | 0.95 | yes |
| picks_12m_S1 | 64 | 1021 | 1.746 | [1.09, 2.39] | 0.28 | 0.62 | 0.61 | yes |
| picks_12m_S1_without_49 | 63 | 1004 | 1.773 | [1.12, 2.41] | 0.29 | 0.64 | 0.62 | yes |
| picks_12m_S2 | 64 | 790 | 2.012 | [1.35, 2.70] | 0.31 | 0.60 | 0.59 | yes |
| picks_12m_dinf_S1 | 64 | 907 | 1.782 | [1.07, 2.44] | 0.30 | 0.55 | 0.62 | yes |
| picks_10m_S1 | 64 | 365 | 2.493 | [1.76, 3.22] | 0.40 | 0.48 | 0.47 | no |


**Rule H0b holds** for the primary measurement: b = 1.75 with 1.65 inside the 95 % interval [1.09, 2.39], and the picks sit at a median
0.62 × M&D's relation (61 % within 0.5-2). It also holds without pick 49 (b 1.77, 0.64 ×), with the two-cell slope (b 2.01, 0.60 ×) and with
D-infinity areas (b 1.78, 0.55 ×). **It fails on the 10 m grid** (b 2.49 [1.76, 3.22]; 0.48 ×): a 10 m block mean is already too fine for the
hollow flow to converge into one cell (the addendum 20 T2 table shows the same fall-off below 12 m). The scatter is large (R² 0.28 against
Table 1's 0.60): 64 hand-placed points on 12 m cells carry more area noise than M&D's planimetered basins.

**The lattice phase (state check item 7).** A_F over 16 phases of the 12 m lattice is 4,968-6,984 m² (median 6,228), C_F 671-896 m² (median 817),
b 1.55-2.39; H0b holds on 13 of 16 phases. The addendum 20 window lattice reproduces its 7,200 m² exactly (b 1.76). So "≈ 7,200 m²" and
"6,120 m²" are the same measurement on two lattices; the directive's expected value lies inside the phase spread. The decision 1.2 values are
those of the full-site lattice (phase 0, 0), as decided before the result.

**Field values.** A_F = 6,120 m² (without pick 49: 6,192); C_F = 876 m² at α = 2 (894 without 49); C_F,1.65 = 1,230 m².

**Rule H0 (the observed criterion on TTV1-TTV3 pooled; Appendix F11, figures `figures/a21/observed/`).**

| extraction | tiles | median dist (m) | q25-q75 | share <= 25 m | heads in polygon | picks | ratio | reading |
|---|---|---|---|---|---|---|---|---|
| full_lattice_a2 | 3 | 12.6 | 2.2-24.0 | 0.77 | 96 | 61 | 1.57 | too many heads |
| full_lattice_a2_without_49 | 3 | 12.4 | 2.0-23.3 | 0.78 | 96 | 60 | 1.60 | too many heads |
| full_lattice_a0 | 3 | 13.0 | 3.0-36.9 | 0.61 | 87 | 61 | 1.43 | locates the mapped heads |
| full_lattice_a0_without_49 | 3 | 13.0 | 2.8-36.1 | 0.62 | 87 | 60 | 1.45 | locates the mapped heads |
| full_lattice_a1.65 | 3 | 12.7 | 3.0-24.4 | 0.75 | 84 | 61 | 1.38 | locates the mapped heads |
| full_lattice_a1.65_without_49 | 3 | 12.7 | 2.8-24.1 | 0.77 | 84 | 60 | 1.40 | locates the mapped heads |
| tile_lattice_a2 | 3 | 21.8 | 5.0-44.8 | 0.56 | 86 | 61 | 1.41 | locates the mapped heads |
| tile_lattice_a0 | 3 | 20.6 | 5.0-50.0 | 0.52 | 78 | 61 | 1.28 | locates the mapped heads |
| tile_lattice_perron | 3 | 83.5 | 54.1-139.7 | 0.18 | 21 | 61 | 0.34 | too few heads, distances large |


- **H0 at α = 2 and C_F fails, narrowly, on the head count:** the median pick-to-channel distance is 12.6 m (77 % of picks within 25 m) but the
  criterion makes 96 heads inside the polygon for 61 picks, a ratio of 1.57 against the bound 1.5. The excess is on TTV1 (49 heads for 22
  picks, 2.23); TTV2 gives 1.21 (41/34) and TTV3 1.20 (6/5). **At α = 0 and A_F the criterion locates the mapped heads** (13.0 m, ratio 1.43), and
  so does α = 1.65 at C_F,1.65 (12.7 m, 1.38). Without pick 49 nothing changes (1.60 / 1.45).
- The traced-channel overlap is the same for every α: 69-74 % of the M&D line cells lie within 15 m of a criterion channel, and 71-83 % of the
  criterion channel cells within 15 m of a traced line (TTV3 has the lowest reverse share, 0.52-0.71: its criterion channels extend past the
  mapped network into the part of the tile M&D did not map).
- **On the tile lattice** (the extraction every model gets, section 6) the observed surface gives 21.8 m and 1.41 at α = 2, 20.6 m and 1.28 at
  α = 0: both "locate the mapped heads". The 9 m difference in distance between the lattices is the lattice phase again (a pick inside a
  12 m channel cell has distance 0 on one lattice and up to 8 m on another) plus the tile-edge truncation of areas.
- **The Perron (2012) procedure** on the observed surface finds 21 valley heads for 61 picks at a median 84 m from them: at σ = 10 m the
  contour-curvature clusters are the incised valley floors, whose upslope ends lie 50-140 m below M&D's channel heads. It is reported as the
  directive asks, without a verdict; the same holds on every model surface (section 6).

## 4. The fixed step: rule F0

Six single evaluations per tile at K = K₀ (Appendix F9). MSE relative to AD, with the F-relief MSE in m²:

| tile | F_phys | F_form | F_relief | F2_phys | F2_form | F2_relief | F best | F worst | C_crest |
|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 37.98 (96.33) | 10.20 (25.88) | 1.15 (2.92) | 48.57 (123.19) | 13.61 (34.53) | 1.08 (2.73) | relief | phys | 0.0159 |
| TTV2 | 30.18 (72.46) | 6.68 (16.04) | 0.46 (1.12) | 29.56 (70.98) | 6.74 (16.18) | 0.59 (1.42) | relief | phys | 0.0149 |
| TTV3 | 15.34 (36.22) | 3.86 (9.12) | 0.54 (1.27) | 25.84 (61.02) | 6.96 (16.44) | 0.52 (1.22) | relief | phys | 0.0147 |
| TTV4 | 13.61 (49.83) | 2.27 (8.31) | 0.56 (2.05) | 27.95 (102.32) | 4.85 (17.75) | 0.49 (1.80) | relief | phys | 0.0124 |
| TTV5 | 5.61 (12.43) | 3.94 (8.72) | 0.76 (1.69) | 56.30 (124.79) | 40.25 (89.22) | 0.84 (1.86) | relief | phys | 0.0223 |
| TTV6 | 72.86 (135.53) | 14.56 (27.08) | 1.19 (2.21) | 120.68 (224.48) | 26.08 (48.50) | 1.00 (1.87) | relief | phys | 0.0139 |
| TTV7 | 50.04 (75.24) | 16.17 (24.31) | 1.94 (2.92) | 159.02 (239.12) | 51.66 (77.68) | 2.09 (3.15) | relief | phys | 0.0163 |
| TTV8 | 26.52 (66.75) | 9.65 (24.30) | 0.95 (2.40) | 132.55 (333.58) | 51.80 (130.36) | 0.94 (2.37) | relief | phys | 0.0176 |
| TTV9 | 14.55 (35.55) | 8.99 (21.98) | 1.83 (4.46) | 13.66 (33.39) | 8.56 (20.92) | 1.71 (4.18) | relief | phys | 0.0212 |
| TTV11 | 62.58 (55.40) | 20.15 (17.84) | 2.36 (2.09) | 114.34 (101.23) | 38.07 (33.71) | 2.20 (1.95) | relief | phys | 0.0167 |
| TTV15 | 30.57 (91.86) | 6.69 (20.12) | 0.60 (1.81) | 129.00 (387.63) | 29.61 (88.98) | 0.62 (1.87) | relief | phys | 0.0141 |


- **Both planner predictions hold on 11 of 11 tiles:** F-relief fits best of the three and F-phys worst, on the α = 0 and on the α = 2 network.
- **F-phys** is 5.6-73 × AD's misfit (median 30 ×). Its crests are too sharp: the model crest curvature at D_phys is U/D_phys = 0.026 m⁻¹ against the
  measured 0.0124-0.0223 (median 0.0159), sharper on 11 of 11 tiles. Letting K float at D = D_phys does not rescue it (K/K₀ 2-1,700, MSE still
  2,200-8,800 m²): the misfit is in the hillslopes, not the channels.
- **F-form** (D from the crest curvature at U) is 2.3-20 × AD (median 9 ×): the right crests, but hillslopes 3.6-6.4 × too short for the relief.
- **F-relief** (D from the hillslope relief at U) has no free parameter and still comes within 0.46-2.36 × AD's two-parameter misfit (median
  0.95 ×; better than AD on 6 of 11 tiles), and 1.1-4.5 × SJ-C's three-parameter misfit (median 2.1 ×). On TTV2 it is 97 m² against P4's 60 and AD's 210.
- The α = 2 network (F2) gives the same ordering; F2-relief is 0.49-2.20 × AD (median 0.94).

## 5. Free fits: PF-A, PF-C, SJ-C, P4, AD; rules K1, M1

**Fits (Appendix F9-F10; the rescaled P4 and AD in Appendix F7).** All at U = 6.5 × 10⁻⁵, m = 0.33, the addendum 18 box in (U/K, U/D) (D ≤
0.0422 m²/yr).

| tile | K/K0 PF_A | K/K0 PF_C | K/K0 SJ_C | K/K0 P4 | K/K0 AD | MSE PF_A | MSE PF_C | MSE SJ_C | MSE P4 | MSE AD | SJ-C/AD | PF-A/AD | PF-C/AD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 0.86 | 0.85 | 0.90 | 1.10 | 9.23 | 117.6 | 115.5 | 84.4 | 82.7 | 214.2 | 0.39 | 0.55 | 0.54 |
| TTV2 | 1.02 | 0.99 | 0.98 | 1.04 | 4.82 | 69.5 | 101.3 | 87.3 | 59.8 | 209.7 | 0.42 | 0.33 | 0.48 |
| TTV3 | 0.97 | 0.89 | 0.92 | 1.42 | 1.96 | 152.5 | 161.1 | 138.4 | 87.4 | 326.8 | 0.42 | 0.47 | 0.49 |
| TTV4 | 0.99 | 0.92 | 0.95 | 0.96 | 10.09 | 153.0 | 128.2 | 80.3 | 96.8 | 294.0 | 0.27 | 0.52 | 0.44 |
| TTV5 | 0.88 | 0.67 | 0.65 | 1.22 | 2.78 | 562.1 | 422.5 | 380.3 | 137.9 | 842.9 | 0.45 | 0.67 | 0.50 |
| TTV6 | 0.85 | 0.82 | 0.80 | 0.76 | 7.60 | 128.8 | 120.3 | 109.9 | 94.3 | 204.3 | 0.54 | 0.63 | 0.59 |
| TTV7 | 0.82 | 0.76 | 0.71 | 0.75 | 20.90 | 185.6 | 124.6 | 89.3 | 79.2 | 134.3 | 0.67 | 1.38 | 0.93 |
| TTV8 | 0.87 | 0.76 | 0.82 | 0.73 | 13.20 | 177.0 | 124.4 | 108.5 | 114.7 | 273.0 | 0.40 | 0.65 | 0.46 |
| TTV9 | 0.81 | 0.85 | 0.73 | 0.73 | 11.35 | 312.1 | 355.8 | 116.6 | 115.2 | 284.9 | 0.41 | 1.10 | 1.25 |
| TTV11 | 0.86 | 0.77 | 0.77 | 0.69 | 12.53 | 261.1 | 185.9 | 185.5 | 241.8 | 164.2 | 1.13 | 1.59 | 1.13 |
| TTV15 | 1.04 | 0.77 | 0.81 | 1.11 | 15.39 | 187.6 | 133.0 | 109.0 | 103.4 | 327.5 | 0.33 | 0.57 | 0.41 |


- **Rule K1 holds for every partitioned model:** K/K₀ is within 0.5-2 on 11 of 11 tiles for PF-A (median 0.87), PF-C (0.82), P4 (0.96) and
  SJ-C (0.81; K/K₀ 0.65-0.98). **It fails for AD** (1 of 11; K/K₀ 2.0-20.9, median 10.1). Against K_cal the partitioned K is 0.36-0.43 × at the median (the
  K₀/K_cal of section 2 carried through); AD's is 4.2 ×.
- **Rule M1: SJ-C beats AD on 10 of 11 tiles (holds; median MSE ratio 0.42)**; the exception is TTV11 (1.13), where AD also beats P4 (1.47) and every field-threshold fit. PF-C beats AD on 9 of 11 tiles (holds; median MSE ratio 0.50); PF-A on 8 of 11 (fails; 0.63); P4 on 10 of 11 (0.39).
  The tiles where the field-threshold models lose to AD are TTV7, TTV9 and TTV11 (PF-A) and TTV9, TTV11 (PF-C); on TTV11 even P4 loses (1.47).
- **D bound hits:** PF-C on TTV5, P4 on TTV3, TTV4, TTV5 and TTV7, AD on TTV8 and TTV15, SJ-C on TTV4, TTV5 and TTV7; PF-A nowhere.
- **Hold-out (the addendum 18 basin halves; fit on one half, score the other, mean):** SJ-C's held-out mean MSE is below P4's on 3 of 11 tiles (TTV4, TTV6, TTV11) and 1.06-2.8 × P4's on the other eight; both lie far below AD's on 9 of 11 (AD's held-out MSE is 2-10 × its in-sample value). The two halves of a tile pick very different C (TTV2: 381 and 1,093 m²), so the three-parameter C does not transfer between basin halves. For the two-parameter field-threshold fits:
  PF-C's held-out mean is below P4's on 3 of 11 tiles and PF-A's on 2 of 11; AD's is the worst on 10 of 11.
- **SJ-C's C profile** (`figures/a21/sjc_profiles.png`): the optimum C is 409-5,644 m² (0.5-6.4 × C_F: below C_F on TTV2, TTV6, TTV8 and TTV11, above it on the other seven) with 8-298 12 m heads per tile against the 30-104 of the C_F network; the profiles are staircases with isolated dips, so the 1 % intervals are a single value (TTV1) to a factor 1.3 wide (Appendix F10); the MSE gain over PF-C is 0.2-67 % (median 14 %), and the free C costs the model its head positions (section 6).

## 6. Model heads: rules H1 and H2

Every surface block-meaned to 12 m on its own lattice, routed, and the criterion applied with its own A and S (decision 1.4); pooled over
TTV1-TTV3 (61 picks inside the polygon). Figures `figures/a21/model_heads/TTV1-3_model_heads.png`.

| model | a2 median dist (m) | a2 ratio | heads/picks | H1 reading | a0 dist | a0 ratio | Perron dist | Perron ratio |
|---|---|---|---|---|---|---|---|---|
| obs | 21.8 | 1.41 | 86/61 | locates the mapped heads | 20.6 | 1.28 | 83.5 | 0.34 |
| F_phys | 0.0 | 21.18 | 1292/61 | too many heads | 26.1 | 0.87 | 30.1 | 0.84 |
| F_form | 2.0 | 12.51 | 763/61 | too many heads | 27.0 | 0.87 | 44.7 | 0.64 |
| F_relief | 31.0 | 1.23 | 75/61 | count in range, distances large | 35.5 | 0.89 | 62.3 | 0.48 |
| F2_phys | 0.0 | 20.56 | 1254/61 | too many heads | 28.0 | 0.85 | 36.2 | 0.69 |
| F2_form | 1.0 | 12.85 | 784/61 | too many heads | 28.3 | 0.84 | 44.2 | 0.61 |
| F2_relief | 22.6 | 1.36 | 83/61 | locates the mapped heads | 33.0 | 0.74 | 61.6 | 0.49 |
| PF_A | 19.0 | 1.64 | 100/61 | too many heads | 32.7 | 0.90 | 62.1 | 0.66 |
| PF_C | 27.5 | 1.36 | 83/61 | count in range, distances large | 31.0 | 0.87 | 54.8 | 0.61 |
| SJ_C | 44.1 | 1.80 | 110/61 | too many heads, distances large | 41.0 | 0.93 | 61.6 | 0.62 |
| P4 | 46.9 | 2.41 | 147/61 | too many heads, distances large | 55.4 | 0.87 | 66.0 | 0.61 |
| AD | 41.2 | 2.28 | 139/61 | too many heads, distances large | 12.8 | 1.26 | 63.8 | 0.56 |


- **H1 at α = 2 and C_F:** the observed lidar locates the mapped heads (21.8 m, 1.41), and so does **F2-relief** (22.6 m, 1.36) ; **SJ-C** (44.1 m, 1.80).
  **PF-C** has the count right (1.36) with the median distance 2.5 m over the bound (27.5 m); **PF-A** has the distance (19.0 m) and too many
  heads (1.64); **P4** (46.9 m, 2.41) and **AD** (41.2 m, 2.28) have neither. F-phys and F-form make 12-21 × too many heads: their hillslopes are
  so much steeper than the real ones that A₁₂ S₁₂² crosses C_F on almost every 12 m hollow cell.
- **At α = 0 and A_F** (no verdict) the picture inverts for AD: AD's surface gives 12.8 m and 1.26, the partitioned surfaces 26-55 m with
  ratios 0.74-0.90. The AD surface has the drainage area of the real landscape at the head positions but not the slope; its A₁₂ S₁₂² criterion
  fires 2.3 × too often because its hillslopes above the heads are too steep (AD's D is 0.003-0.5 × D_phys on 7 of 11 tiles).
- **The Perron sensitivity** (no verdict) gives 29-51 heads for 61 picks at 30-66 m on every model surface: the contour-curvature valleys of
  a model surface at σ = 10 m end where the model's channel network does, well below the mapped heads, as on the observed surface.
- **Rule H2 (SJ-C against AD, pooled) fails.** SJ-C's pooled median pick distance (44.1 m) is not smaller than AD's (41.2 m); its head ratio (1.80) is closer to 1 on the log scale than AD's (2.28). One of the two conditions.**

## 7. The D explanation: rule DX and the decomposition

| tile | D_relief | network | PF_A D/D_relief | PF_C D/D_relief | SJ_C D/D_relief | P4 D/D_relief | PF_A D/D_phys | PF_C D/D_phys | SJ_C D/D_phys | P4 D/D_phys | (D_relief/D_form)(U/E_ridge) | AD D/D_phys | AD D/D_relief |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 0.0175 | mapped | 0.94 | 1.07 | 1.54 | 2.12 | 6.6 | 7.5 | 10.8 | 14.8 | 10.33 | 0.54 | 0.0772 |
| TTV2 | 0.0159 | mapped | 0.72 | 0.79 | 0.66 | 0.88 | 4.6 | 5.0 | 4.2 | 5.6 | 8.79 | 0.0799 | 0.0126 |
| TTV3 | 0.0230 | mapped | 0.68 | 1.08 | 1.35 | 1.83 | 6.3 | 10.0 | 12.4 | 16.9 | 12.51 | 0.00312 | 0.000339 |
| TTV4 | 0.0194 | A | 0.78 | 1.36 | 2.17 | 2.17 | 6.1 | 10.5 | 16.9 | 16.9 | 8.89 | 1.22 | 0.156 |
| TTV5 | 0.0186 | A | 0.70 | 2.27 | 2.27 | 2.27 | 5.2 | 16.9 | 16.9 | 16.9 | 15.37 | 0.0176 | 0.00236 |
| TTV6 | 0.0207 | A | 0.98 | 1.34 | 1.58 | 1.36 | 8.1 | 11.1 | 13.1 | 11.2 | 10.63 | 0.377 | 0.0456 |
| TTV7 | 0.0144 | A | 2.65 | 2.65 | 2.93 | 2.93 | 15.3 | 15.3 | 16.9 | 16.9 | 8.68 | 8 | 1.39 |
| TTV8 | 0.0174 | A | 1.00 | 1.76 | 1.66 | 2.21 | 7.0 | 12.3 | 11.6 | 15.4 | 11.35 | 16.9 | 2.42 |
| TTV9 | 0.0142 | A | 1.87 | 1.44 | 2.73 | 2.61 | 10.6 | 8.1 | 15.5 | 14.8 | 11.13 | 0.663 | 0.117 |
| TTV11 | 0.0182 | A | 0.96 | 1.57 | 1.57 | 1.91 | 7.0 | 11.4 | 11.4 | 13.9 | 11.25 | 10.9 | 1.49 |
| TTV15 | 0.0182 | A | 0.78 | 1.69 | 1.70 | 1.33 | 5.6 | 12.3 | 12.3 | 9.7 | 9.48 | 16.9 | 2.32 |


- **Rule DX holds for PF-A** (D_fit/D_relief within 0.67-1.5 on 9 of 11 tiles; median 0.94) **and fails for PF-C** (6 of 11; median 1.44; above 1.5
  on 5 tiles, TTV4, TTV5 (at the bound), TTV7, TTV8, TTV15), **and for SJ-C** (1 of 11; median 1.66; above 1.5 on 9 tiles). **P4 is above the relief on 8 of 11** (median 2.12; 3 inside).
- **The decomposition.** The product (D_relief/D_form) × (U/E_ridge) is 8.7-15.4 (median 10.6). D_fit/D_phys is 4.6-15.3 for PF-A (median 6.6),
  5.0-16.9 for PF-C (median 11.1), 4.2-16.9 for SJ-C (median 12.4), and 5.6-16.9 for P4 (median 14.8). So the fitted D of the field-threshold models is the D that
  linear diffusion needs for the measured hillslope relief at rate U (PF-A to within 6 % at the median; PF-C 1.4 × above it, on the tiles where
  its network is shorter), and its excess over the physical D is the product of the nonlinear-hillslope factor D_relief/D_form (3.6-6.4) and
  the steady-state mismatch U/E_ridge (2.4) — not a defect of the partitioned model. P4's D sits above the relief term: the tuned A_crit
  (9,100-89,000 m²) makes hillslopes longer than the real ones and D follows them.
- **AD:** D_fit/D_phys is 0.003-0.5 on 7 tiles and 8-17 (at or near the bound) on 4; D_fit/D_relief 0.0003-2.4. No reading is set; AD's D is not
  a transport coefficient of these hillslopes in either direction.

## 8. Sensitivities

- **U.** Every fitted K and D, D_form and D_relief scale with U, so K/K₀, D/D_form, D/D_relief and every MSE are unchanged at 1.02 × 10⁻⁴ (sub-basin
  2) or 7.7 × 10⁻⁵ (the legacy P₀); D/D_phys and the decomposition product scale by 1.57 and 1.18, U/E_ridge becomes 3.8 and 2.9.
- **E_ridge mean (3.5 × 10⁻⁵):** U/E_ridge 1.86 instead of 2.41; the product median 8.2; E_ridge/|C_crest| 0.0012-0.0028 (0.5-1.1 × D_phys).
- **m = 0.30 (PF-A, PF-C refitted on the same networks):** K/K₀ rises to 1.1-1.5 (K₀ is defined at m = 0.33), D moves by −13 to +45 %, MSE by
  −14 to +6 %; no K1 or DX count changes (Appendix F9).
- **The 10 m grid:** A_F 5,050 m², C_F 681 m²; H0b fails (section 3).
- **D-infinity at 12 m:** A_F 5,663 m², C_F 795 m²; H0b holds (b 1.78).
- **The two-cell slope:** C_F 835 m²; b 2.01 [1.35, 2.70]; H0b holds.
- **Pick 49:** A_F 6,192, C_F 894 m²; the α = 0 networks change on no tile and the α = 2 networks by −2 to +3 % of their cells; H0 unchanged.
- **The lattice phase** (section 3): A_F 4,968-6,984 m², C_F 671-896 m².

## 9. What the evidence supports, and where it is weak

**Supported.**
1. **The map-scale measurement of M&D's criterion is reproduced on Steve's picks** (H0b): the 12 m source areas and valley slopes at the picks
   follow M&D's area-slope relation within its scatter (b 1.75 against 1.65; the points at 0.6 × the relation's area), on four of five
   measurement variants and 13 of 16 lattice phases.
2. **The criterion built from those values puts channels where the heads were mapped.** At α = 2 the pooled median distance is 12.6 m and
   two thirds to three quarters of M&D's traced channel length lies within 15 m of a criterion channel; the head count is 1.4-1.6 × the picks
   (within the bound at α = 0 and 1.65, 0.07 over it at α = 2, the excess on TTV1).
3. **With K₀ and D_relief, no free parameter, the partitioned model fits the TTV tiles about as well as the two-parameter AD** (F-relief 0.95 ×
   AD at the median), and the planner's ordering of the three physical D values holds on every tile (F0).
4. **K is physical for every partitioned model** (K/K₀ 0.7-1.4 on 11 of 11 tiles for PF-A, PF-C and P4 and SJ-C) and not for AD (2-21 ×).
5. **The fitted D of the field-threshold model is the relief D** (PF-A within 0.67-1.5 × D_relief on 9 of 11 tiles; PF-C on 6, above it on 5), and the
   decomposition D_fit/D_phys ≈ (D_relief/D_form)(U/E_ridge) holds at the median (6.6 and 11 against 10.6): the order-of-magnitude excess
   over D_phys is nonlinear transport (× 4) times the non-steady state (× 2.4), as the planner wrote in advance.
6. **PF-C beats AD on 9 of 11 tiles and the field-threshold models put heads near the mapped ones** (PF-C 27 m, ratio 1.36; F2-relief 23 m,
   1.36), where P4 and AD do not (41-47 m, 2.3-2.4).
7. **With the head criterion free (SJ-C) the partitioned model beats AD on 10 of 11 tiles with physical K**, but the free C moves away from C_F on 7 of 11 tiles and the model's heads land 44 m from the mapped ones with 1.8 × their number: fitting elevations alone does not recover the field criterion.

**Weak.**
1. **Non-steady state.** TTV's ridges lower at 27-35 m/Myr and its basins at 65-102; the models have one U. D_form/D_phys (1.2-2.1) and the
   decomposition product carry that mismatch explicitly, but no steady-state D can be "the" physical D here, and the hold-out MSEs (2-10 × the
   in-sample ones on several tiles) show that the single-U fits do not transfer between basin halves.
2. **The head count at α = 2** exceeds the H0 bound by 0.07 (TTV1), and the criterion's verdict depends on the lattice: on the tile lattice the
   distances are 9 m longer. A_F and C_F themselves move by ± 15 % with the lattice phase.
3. **D_relief on 8 of 11 tiles uses the criterion network, not mapped heads** (flagged); on TTV1-TTV3 the two agree within 10 %.
4. **The Perron (2012) condensation** finds the incised valley heads, not M&D's channel heads; it is not a test of the same thing.
5. ****SJ-C's C profile** is a staircase with isolated dips (1 % intervals one value to 1.3 × wide), the basin halves disagree on C by up to 3 ×, and n_channel(C) is not monotone on 3 tiles: the headline three-parameter model is the least transferable of the partitioned fits (held-out MSE above P4's on 8 of 11).**
6. The seeding of a 12 m head into the 1 m grid takes one cell per block; the 1 m D8 thread it follows is one of several in a hollow (addendum
   20), so the 1 m network's exact pixels, and the "channel cells without χ" counts, depend on that choice (0 such cells on every surface here).

## 10. Decisions needed

1. **Which channel criterion the paper carries for TTV:** the field area threshold A_F (H0 holds; PF-A: K physical, D explained by relief, beats AD
   on 8 of 11) or the field A·S² threshold C_F (H0 fails by 0.07 on the count; PF-C beats AD on 9 of 11, D above the relief on 5 tiles) , or the free criterion SJ-C (the best misfit, K physical, heads 44 m from the mapped ones and 1.8 × their number).
2. **The D statement for TTV:** report the fitted D against D_relief with the decomposition (section 7), or against D_phys with the two factors
   stated; the non-steady-state caveat is unavoidable either way.
3. **The lattice:** whether the map-scale measurement should be averaged over lattice phases (A_F ≈ 6,200 ± 500 m², C_F ≈ 820 ± 70 m²) rather
   than taken on one lattice.
4. **Whether to run the self-consistent version** (heads iterated on the model's own slopes), which this addendum did not.
5. Nothing merged or tagged.

## 11. Appendices: all tables, run ids, checksums, Slurm accounting

Run directory: `$GROUP_HOME/lemcvt_results/20261123T184712Z_a21_ttv_channel_head_test/` (grid12/, tiles/, sjc/, rules/, figures/). Jobs and logs: `$SCRATCH/lemcvt_jobs/a21/`.

### F1. Inputs and checksums

| input | path | sha256 |
|---|---|---|
| Steve's 64 picks | data/ttv/ttv_heads_manual_steve_20261119.csv | ac87de3d4ca1f479e6eb2ceda2fedb033740b3d8f37186fd267329d5e4e702b6 |
| TTV site config (this addendum) | configs/sites/TTV.yaml | a91d298309d026052589d2e0b5dad08a9c64597d60255c20c27e4934a79788fe |
| full-site 1 m filled grid | $GROUP_HOME/lemcvt_results/20261118T160326Z_a15_stage_b/grid_TTV/TTV_1m_filled.tif | 95737de40f09260e… (tile TTV2 cut; the full grid's own sha256 in the stage B record) |

### F2. Map-scale heads (decision 1.2): the picks on each grid

- **12 m grid** (1584 x 1584 cells): A at the picks p25/p50/p75 3852 / 6120 / 11412 m² (cell area median 1584); share within Table 1's 5-95 % 0.78; S one cell median 0.350 (q25-q75 0.303-0.425), two cells 0.362 (0 chains short of two cells); S undefined on 0 picks; D-infinity A median 5663 m²
- **10 m grid** (1901 x 1901 cells): A at the picks p25/p50/p75 2400 / 5050 / 9825 m² (cell area median 1000); share within Table 1's 5-95 % 0.70; S one cell median 0.345 (q25-q75 0.299-0.441), two cells 0.363 (0 chains short of two cells); S undefined on 0 picks

| pick | A_12 (m²) | cell A_12 | S_12 one cell | S_12 two cells | A_12 D-inf | A_10 | S_10 |
|---|---|---|---|---|---|---|---|
| 1 | 10512 | 1584 | 0.314 | 0.302 | 7143 | 9900 | 0.308 |
| 2 | 15984 | 15264 | 0.280 | 0.261 | 11232 | 15000 | 0.264 |
| 3 | 15408 | 13680 | 0.280 | 0.285 | 16545 | 9200 | 0.291 |
| 4 | 1440 | 432 | 0.306 | 0.320 | 1405 | 1000 | 0.335 |
| 5 | 3312 | 2592 | 0.438 | 0.456 | 2920 | 3000 | 0.486 |
| 6 | 19440 | 18432 | 0.278 | 0.265 | 16628 | 22400 | 0.273 |
| 7 | 6912 | 3312 | 0.278 | 0.301 | 5758 | 6000 | 0.318 |
| 8 | 7776 | 3024 | 0.299 | 0.278 | 7261 | 7400 | 0.285 |
| 9 | 8208 | 1296 | 0.418 | 0.459 | 12973 | 5700 | 0.494 |
| 10 | 21168 | 1152 | 0.245 | 0.280 | 20844 | 17400 | 0.288 |
| 11 | 11808 | 1008 | 0.288 | 0.310 | 7815 | 13900 | 0.247 |
| 12 | 13680 | 1728 | 0.288 | 0.288 | 14098 | 12700 | 0.296 |
| 13 | 4320 | 3456 | 0.434 | 0.342 | 4071 | 5600 | 0.345 |
| 14 | 2304 | 2160 | 0.336 | 0.316 | 2306 | 2200 | 0.294 |
| 15 | 21168 | 21024 | 0.276 | 0.305 | 22664 | 21900 | 0.315 |
| 16 | 3312 | 1152 | 0.574 | 0.517 | 2974 | 1900 | 0.525 |
| 17 | 27648 | 26208 | 0.240 | 0.234 | 30132 | 28900 | 0.258 |
| 18 | 13392 | 1008 | 0.329 | 0.373 | 12928 | 12900 | 0.345 |
| 19 | 21168 | 720 | 0.393 | 0.384 | 17453 | 20700 | 0.321 |
| 20 | 3744 | 864 | 0.358 | 0.388 | 4211 | 3600 | 0.284 |
| 21 | 13392 | 1584 | 0.424 | 0.424 | 11134 | 12400 | 0.413 |
| 22 | 4176 | 288 | 0.516 | 0.487 | 5568 | 3700 | 0.519 |
| 23 | 4320 | 3312 | 0.638 | 0.628 | 2545 | 2700 | 0.594 |
| 24 | 4464 | 576 | 0.538 | 0.553 | 2413 | 900 | 0.601 |
| 25 | 3888 | 1728 | 0.431 | 0.433 | 4698 | 1600 | 0.441 |
| 26 | 1728 | 144 | 0.631 | 0.646 | 985 | 900 | 0.525 |
| 27 | 5472 | 144 | 0.330 | 0.366 | 5179 | 4500 | 0.386 |
| 28 | 34848 | 144 | 0.358 | 0.362 | 22679 | 2200 | 0.413 |
| 29 | 2448 | 2016 | 0.572 | 0.512 | 2470 | 2400 | 0.551 |
| 30 | 720 | 288 | 0.549 | 0.516 | 538 | 600 | 0.506 |
| 31 | 6192 | 720 | 0.510 | 0.486 | 6080 | 1600 | 0.458 |
| 32 | 8352 | 2736 | 0.365 | 0.324 | 6785 | 3600 | 0.399 |
| 33 | 4752 | 576 | 0.254 | 0.263 | 3072 | 5000 | 0.268 |
| 34 | 6048 | 1008 | 0.390 | 0.361 | 5128 | 5700 | 0.376 |
| 35 | 11952 | 9504 | 0.306 | 0.305 | 9082 | 10300 | 0.337 |
| 36 | 11376 | 9504 | 0.309 | 0.311 | 9405 | 10100 | 0.301 |
| 37 | 5472 | 1296 | 0.355 | 0.420 | 3931 | 3600 | 0.300 |
| 38 | 3888 | 1584 | 0.389 | 0.390 | 3539 | 1900 | 0.369 |
| 39 | 3168 | 1584 | 0.544 | 0.494 | 3297 | 2400 | 0.447 |
| 40 | 7632 | 432 | 0.370 | 0.376 | 7803 | 8400 | 0.395 |
| 41 | 13392 | 864 | 0.400 | 0.356 | 13196 | 12300 | 0.344 |
| 42 | 1728 | 432 | 0.304 | 0.362 | 7190 | 1300 | 0.351 |
| 43 | 720 | 576 | 0.428 | 0.423 | 990 | 600 | 0.443 |
| 44 | 4032 | 864 | 0.471 | 0.454 | 3521 | 6800 | 0.491 |
| 45 | 8352 | 1296 | 0.318 | 0.318 | 7287 | 7500 | 0.303 |
| 46 | 3024 | 1584 | 0.390 | 0.415 | 2667 | 2600 | 0.384 |
| 47 | 4320 | 144 | 0.495 | 0.472 | 3673 | 400 | 0.578 |
| 48 | 7200 | 6480 | 0.419 | 0.464 | 6358 | 8300 | 0.427 |
| 49 | 4032 | 864 | 0.311 | 0.324 | 2294 | 4400 | 0.306 |
| 50 | 8352 | 3456 | 0.321 | 0.316 | 8412 | 7400 | 0.296 |
| 51 | 3168 | 576 | 0.284 | 0.316 | 2820 | 1500 | 0.346 |
| 52 | 11952 | 11088 | 0.244 | 0.319 | 6510 | 12300 | 0.199 |
| 53 | 9072 | 8928 | 0.283 | 0.275 | 13706 | 8200 | 0.235 |
| 54 | 6768 | 3456 | 0.304 | 0.284 | 7202 | 7400 | 0.345 |
| 55 | 5328 | 5184 | 0.273 | 0.298 | 3821 | 5100 | 0.274 |
| 56 | 2448 | 576 | 0.345 | 0.384 | 2386 | 2400 | 0.289 |
| 57 | 3888 | 3456 | 0.301 | 0.373 | 4007 | 4100 | 0.438 |
| 58 | 10368 | 720 | 0.319 | 0.284 | 9117 | 10000 | 0.313 |
| 59 | 11520 | 2016 | 0.333 | 0.302 | 9453 | 8100 | 0.334 |
| 60 | 7632 | 6480 | 0.325 | 0.363 | 7758 | 9800 | 0.355 |
| 61 | 3600 | 2880 | 0.424 | 0.447 | 4103 | 2600 | 0.451 |
| 62 | 4176 | 4032 | 0.424 | 0.420 | 3601 | 4100 | 0.416 |
| 63 | 1008 | 720 | 0.535 | 0.468 | 1543 | 800 | 0.509 |
| 64 | 6336 | 5328 | 0.415 | 0.352 | 5334 | 5000 | 0.390 |

### F3. Fits and field values (decision 1.2; rule H0b)

| fit | n | a (m²) | a 95 % | b | b 95 % | R² | median A/(1978 S^-1.65) | share in 0.5-2 | 1.65 in b 95 % | H0b |
|---|---|---|---|---|---|---|---|---|---|---|
| table1 | 63 | 1978 | [1456, 2774] | 1.647 | [1.26, 2.00] | 0.60 | 1.01 | 0.95 | yes | yes |
| picks_12m_S1 | 64 | 1021 | [512, 1990] | 1.746 | [1.09, 2.39] | 0.28 | 0.62 | 0.61 | yes | yes |
| picks_12m_S1_without_49 | 63 | 1004 | [510, 1945] | 1.773 | [1.12, 2.41] | 0.29 | 0.64 | 0.62 | yes | yes |
| picks_12m_S2 | 64 | 790 | [371, 1553] | 2.012 | [1.35, 2.70] | 0.31 | 0.60 | 0.59 | yes | yes |
| picks_12m_dinf_S1 | 64 | 907 | [460, 1890] | 1.782 | [1.07, 2.44] | 0.30 | 0.55 | 0.62 | yes | yes |
| picks_10m_S1 | 64 | 365 | [168, 801] | 2.493 | [1.76, 3.22] | 0.40 | 0.48 | 0.47 | no | no |

| set | A_F (m²) | C_F alpha 2 (m²) | C_F alpha 1.65 (m²) | A_F w/o 49 | C_F w/o 49 | A·S² p25 / p75 | S nan or <= 0 |
|---|---|---|---|---|---|---|---|
| 12m | 6120 | 876.3 | 1230.3 | 6192 | 893.9 | 595 / 1219 | 0 |
| 12m_S2 | 6120 | 835.2 | 1190.5 | 6192 | 835.7 | 585 / 1155 | 0 |
| 12m_dinf | 5663 | 794.9 | 1092.8 | 5758 | 809.1 | 525 / 1074 | 0 |
| 10m | 5050 | 680.5 | 950.6 | 5100 | 689.3 | 354 / 1008 | 0 |

### F4. Lattice-phase sensitivity of the 12 m measurement (added after Part A)

| phase (rows, cols) | A_F (m²) | C_F (m²) | C_F 1.65 | b | b 95 % | median ratio to eq. 10 | H0b |
|---|---|---|---|---|---|---|---|
| [0, 0] | 6120 | 876 | 1230 | 1.75 | [1.09, 2.39] | 0.62 | yes |
| [0, 3] | 5976 | 740 | 1081 | 2.15 | [1.42, 2.83] | 0.55 | yes |
| [0, 6] | 6120 | 867 | 1157 | 1.83 | [0.81, 2.69] | 0.58 | yes |
| [0, 9] | 6264 | 877 | 1225 | 2.19 | [1.44, 2.90] | 0.62 | yes |
| [3, 0] | 4968 | 699 | 990 | 2.33 | [1.77, 2.84] | 0.50 | no |
| [3, 3] | 6192 | 712 | 996 | 2.39 | [1.87, 2.88] | 0.50 | no |
| [3, 6] | 6552 | 813 | 1124 | 1.55 | [0.79, 2.24] | 0.57 | yes |
| [3, 9] | 6768 | 896 | 1276 | 2.13 | [1.43, 2.78] | 0.65 | yes |
| [6, 0] | 5400 | 671 | 989 | 1.94 | [1.44, 2.44] | 0.50 | no |
| [6, 3] | 6336 | 715 | 1037 | 1.89 | [1.34, 2.42] | 0.52 | yes |
| [6, 6] | 6912 | 811 | 1179 | 1.73 | [1.02, 2.39] | 0.60 | yes |
| [6, 9] | 6768 | 776 | 1182 | 1.93 | [1.19, 2.56] | 0.60 | yes |
| [9, 0] | 5472 | 821 | 1114 | 1.78 | [1.16, 2.39] | 0.56 | yes |
| [9, 3] | 6984 | 895 | 1195 | 1.92 | [1.15, 2.62] | 0.60 | yes |
| [9, 6] | 6984 | 884 | 1223 | 1.63 | [0.88, 2.26] | 0.62 | yes |
| [9, 9] | 5904 | 849 | 1160 | 1.86 | [1.20, 2.44] | 0.59 | yes |
| A20 window [9, 5] | 7200 | 887 | 1257 | 1.76 | [1.01, 2.40] | 0.64 | yes |

- A_F over the 16 phases: 4968-6984 m² (median 6228); C_F 671-896 m² (median 817); b 1.55-2.39; H0b consistent on 13 of 16 phases.

### F5. The criterion networks on the full 12 m grid and per tile (decision 1.3)

| alpha | C | channel cells (12 m, whole grid) | heads | unique M values |
|---|---|---|---|---|
| a0 | 6120 | 83,471 | 5,905 | 9,693 |
| a1.65 | 1230 | 79,204 | 6,919 | 521,384 |
| a2 | 876.3 | 85,477 | 8,809 | 502,343 |

| tile | 12 m box (rows, cols) | candidates | reach the tile | max path steps | pick seeds reaching | α0 heads / entries / 1 m cells / 1 m heads | α2 heads / entries / 1 m cells / 1 m heads | mapped: entries / cells | seed area check (max rel diff, n) |
|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 625-709, 792-876 | 9,937 | 8,759 | 5,191 | 31 | 66 / 57 / 5996 / 41 | 96 / 87 / 6580 / 68 | 31 / 3929 | 0.0e+00, 6889 |
| TTV2 | 583-668, 875-959 | 12,024 | 11,055 | 5,191 | 45 | 76 / 71 / 6932 / 49 | 83 / 77 / 6690 / 48 | 45 / 5508 | 0.0e+00, 6955 |
| TTV3 | 667-751, 875-959 | 19,446 | 18,630 | 5,191 | 37 | 116 / 113 / 8741 / 49 | 114 / 111 / 7169 / 30 | 37 / 3726 | 0.0e+00, 6942 |
| TTV4 | 750-834, 1125-1209 | 11,418 | 10,524 | 5,576 | 0 | 76 / 70 / 7469 / 40 | 59 / 56 / 6121 / 40 | - | 0.0e+00, 6917 |
| TTV5 | 167-251, 333-418 | 11,009 | 9,898 | 11,072 | 0 | 57 / 53 / 6662 / 43 | 54 / 47 / 4575 / 37 | - | 0.0e+00, 6956 |
| TTV6 | 708-793, 958-1043 | 10,531 | 9,382 | 4,958 | 0 | 64 / 61 / 5629 / 39 | 65 / 60 / 4328 / 38 | - | 0.0e+00, 6959 |
| TTV7 | 375-459, 458-543 | 11,958 | 7,177 | 9,350 | 0 | 65 / 40 / 5859 / 36 | 88 / 48 / 4930 / 46 | - | 0.0e+00, 6938 |
| TTV8 | 500-584, 667-751 | 13,392 | 12,545 | 6,116 | 0 | 67 / 65 / 5671 / 31 | 107 / 99 / 4972 / 47 | - | 0.0e+00, 6883 |
| TTV9 | 333-418, 292-376 | 21,729 | 21,156 | 10,115 | 0 | 105 / 103 / 7270 / 42 | 277 / 275 / 8727 / 104 | - | 0.0e+00, 6976 |
| TTV11 | 333-418, 750-834 | 9,437 | 8,189 | 7,318 | 0 | 50 / 44 / 6774 / 40 | 72 / 63 / 6194 / 53 | - | 0.0e+00, 6944 |
| TTV15 | 458-543, 750-834 | 10,231 | 8,994 | 6,062 | 0 | 61 / 52 / 5777 / 40 | 81 / 67 / 5417 / 51 | - | 0.0e+00, 6913 |

### F6. Checks: the mask route, the rescaling, the bounds

| tile | mask route same set | rel diff MSE | max |Δb| | P4 stored MSE | P4 direct at new U | rel | AD stored MSE | AD direct at new U | rel | K0 (new U) | PF(7,400) K/K0, D at new U |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | yes | 0.0e+00 | 0.0e+00 | 82.7350 | 82.7350 | 1.1e-14 | 214.1899 | 214.1899 | 0.0e+00 | 9.489e-06 | 0.844, 0.01726 |
| TTV2 | yes | 0.0e+00 | 0.0e+00 | 59.7908 | 59.7908 | 5.8e-15 | 209.6604 | 209.6604 | 2.9e-15 | 1.334e-05 | 1.030, 0.01258 |
| TTV3 | yes | 0.0e+00 | 0.0e+00 | 87.3965 | 87.3965 | 6.7e-16 | 326.8495 | 326.8495 | 1.4e-15 | 2.226e-05 | 0.989, 0.01711 |
| TTV4 | yes | 0.0e+00 | 0.0e+00 | 96.8190 | 96.8190 | 5.3e-15 | 293.9748 | 293.9748 | 2.6e-14 | 1.362e-05 | 0.944, 0.0207 |
| TTV5 | yes | 0.0e+00 | 0.0e+00 | 137.8779 | 137.8779 | 1.7e-15 | 842.9483 | 842.9483 | 0.0e+00 | 1.837e-05 | 0.930, 0.01274 |
| TTV6 | yes | 0.0e+00 | 0.0e+00 | 94.2664 | 94.2664 | 1.1e-14 | 204.3397 | 204.3397 | 0.0e+00 | 1.050e-05 | 0.828, 0.02304 |
| TTV7 | yes | 0.0e+00 | 0.0e+00 | 79.2361 | 79.2361 | 4.2e-15 | 134.2587 | 134.2587 | 0.0e+00 | 9.108e-06 | 0.806, 0.03419 |
| TTV8 | yes | 0.0e+00 | 0.0e+00 | 114.6663 | 114.6663 | 1.4e-14 | 272.9609 | 272.9609 | 0.0e+00 | 9.708e-06 | 0.839, 0.02228 |
| TTV9 | yes | 0.0e+00 | 0.0e+00 | 115.1824 | 115.1824 | 4.4e-16 | 284.9084 | 284.9084 | 0.0e+00 | 8.891e-06 | 0.868, 0.01428 |
| TTV11 | yes | 0.0e+00 | 0.0e+00 | 241.7675 | 241.7675 | 1.5e-14 | 164.1999 | 164.1999 | 0.0e+00 | 9.786e-06 | 0.870, 0.01692 |
| TTV15 | yes | 0.0e+00 | 0.0e+00 | 103.4195 | 103.4195 | 6.7e-15 | 327.5288 | 327.5288 | 0.0e+00 | 1.158e-05 | 1.028, 0.01495 |

### F7. The rescaled addendum 18 and 20 fits (K, D x 65/77; MSE unchanged)

| tile | P4 A_crit | P4 K | P4 D | P4 D at bound (0.0422) | P4 MSE | AD K | AD D | AD D at bound | AD MSE | P3 D | PF(7,400) A20 K | PF(7,400) A20 D | PF(7,400) A20 MSE |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 59499 | 1.043e-05 | 0.03704 | no | 82.73 | 8.760e-05 | 0.001351 | no | 214.19 | 0.03648 | 8.011e-06 | 0.01726 | 115.50 |
| TTV2 | 9131 | 1.389e-05 | 0.01401 | no | 59.79 | 6.434e-05 | 0.0001997 | no | 209.66 | 0.01438 | 1.374e-05 | 0.01258 | 66.67 |
| TTV3 | 89124 | 3.151e-05 | 0.04221 | yes | 87.40 | 4.364e-05 | 7.794e-06 | no | 326.85 | 0.04221 | 2.200e-05 | 0.01711 | 146.00 |
| TTV4 | 40309 | 1.302e-05 | 0.04221 | yes | 96.82 | 1.375e-04 | 0.00304 | no | 293.97 | 0.03986 | 1.286e-05 | 0.0207 | 129.88 |
| TTV5 | 58612 | 2.238e-05 | 0.04221 | yes | 137.88 | 5.108e-05 | 4.396e-05 | no | 842.95 | 0.04221 | 1.708e-05 | 0.01274 | 603.55 |
| TTV6 | 10973 | 8.029e-06 | 0.02808 | no | 94.27 | 7.979e-05 | 0.0009418 | no | 204.34 | 0.03119 | 8.696e-06 | 0.02304 | 118.67 |
| TTV7 | 35966 | 6.863e-06 | 0.04221 | yes | 79.24 | 1.903e-04 | 0.02001 | no | 134.26 | 0.03472 | 7.339e-06 | 0.03419 | 161.89 |
| TTV8 | 38544 | 7.040e-06 | 0.03847 | no | 114.67 | 1.281e-04 | 0.04221 | yes | 272.96 | 0.04213 | 8.145e-06 | 0.02228 | 189.47 |
| TTV9 | 36561 | 6.456e-06 | 0.03694 | no | 115.18 | 1.009e-04 | 0.001657 | no | 284.91 | 0.02721 | 7.715e-06 | 0.01428 | 350.16 |
| TTV11 | 30362 | 6.709e-06 | 0.03468 | no | 241.77 | 1.226e-04 | 0.02713 | no | 164.20 | 0.01379 | 8.510e-06 | 0.01692 | 305.81 |
| TTV15 | 26819 | 1.287e-05 | 0.02413 | no | 103.42 | 1.782e-04 | 0.04221 | yes | 327.53 | 0.02686 | 1.191e-05 | 0.01495 | 167.44 |

### F8. D values per tile (section 2.2) and the relief tracing

| tile | network | pixels | n | reached | used | L_H median (m) | R median (m) | mean slope | D_relief | 68 % |
|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | mapped | inside the M&D polygon | 3,820 | 1,501 | 1,501 | 171 | 50.0 | 0.323 | 0.0175 | [0.0166, 0.0185] |
| TTV1 | A | all counted crest pixels | 3,820 | 3,011 | 3,011 | 159 | 44.4 | 0.279 | 0.0187 | [0.0179, 0.0198] |
| TTV1 | A_in_polygon | inside the M&D polygon | 3,820 | 1,677 | 1,677 | 162 | 48.9 | 0.317 | 0.0175 | [0.0166, 0.0184] |
| TTV1 | C | all counted crest pixels | 3,820 | 3,034 | 3,034 | 166 | 44.5 | 0.279 | 0.0193 | [0.0181, 0.0204] |
| TTV2 | mapped | inside the M&D polygon | 2,919 | 1,879 | 1,879 | 157 | 49.3 | 0.313 | 0.0159 | [0.0153, 0.0165] |
| TTV2 | A | all counted crest pixels | 2,919 | 2,592 | 2,592 | 140 | 47.8 | 0.327 | 0.0143 | [0.0138, 0.0148] |
| TTV2 | A_in_polygon | inside the M&D polygon | 2,919 | 2,062 | 2,062 | 144 | 46.7 | 0.316 | 0.0152 | [0.0148, 0.0157] |
| TTV2 | C | all counted crest pixels | 2,919 | 2,566 | 2,566 | 135 | 42.9 | 0.323 | 0.0137 | [0.0134, 0.0143] |
| TTV3 | mapped | inside the M&D polygon | 1,052 | 68 | 68 | 173 | 42.3 | 0.244 | 0.0230 | [0.0217, 0.0268] |
| TTV3 | A | all counted crest pixels | 1,052 | 812 | 812 | 163 | 42.5 | 0.245 | 0.0216 | [0.0207, 0.0230] |
| TTV3 | A_in_polygon | inside the M&D polygon | 1,052 | 125 | 125 | 214 | 46.3 | 0.219 | 0.0317 | [0.0267, 0.0356] |
| TTV3 | C | all counted crest pixels | 1,052 | 812 | 812 | 176 | 43.8 | 0.246 | 0.0236 | [0.0229, 0.0244] |
| TTV4 | A | all counted crest pixels | 2,648 | 1,693 | 1,693 | 153 | 39.8 | 0.255 | 0.0194 | [0.0180, 0.0208] |
| TTV4 | C | all counted crest pixels | 2,648 | 1,377 | 1,377 | 168 | 42.6 | 0.253 | 0.0225 | [0.0214, 0.0239] |
| TTV5 | A | all counted crest pixels | 3,967 | 3,584 | 3,584 | 129 | 32.0 | 0.234 | 0.0186 | [0.0179, 0.0197] |
| TTV5 | C | all counted crest pixels | 3,967 | 3,419 | 3,419 | 174 | 37.2 | 0.225 | 0.0243 | [0.0223, 0.0270] |
| TTV6 | A | all counted crest pixels | 4,927 | 3,567 | 3,567 | 160 | 39.5 | 0.258 | 0.0207 | [0.0200, 0.0213] |
| TTV6 | C | all counted crest pixels | 4,927 | 3,708 | 3,708 | 193 | 47.3 | 0.260 | 0.0246 | [0.0237, 0.0259] |
| TTV7 | A | all counted crest pixels | 4,476 | 3,913 | 3,913 | 115 | 32.1 | 0.261 | 0.0144 | [0.0138, 0.0150] |
| TTV7 | C | all counted crest pixels | 4,476 | 3,664 | 3,664 | 133 | 38.5 | 0.265 | 0.0157 | [0.0147, 0.0169] |
| TTV8 | A | all counted crest pixels | 5,165 | 4,255 | 4,255 | 131 | 31.5 | 0.241 | 0.0174 | [0.0168, 0.0180] |
| TTV8 | C | all counted crest pixels | 5,165 | 4,117 | 4,117 | 167 | 43.2 | 0.242 | 0.0223 | [0.0211, 0.0236] |
| TTV9 | A | all counted crest pixels | 1,903 | 1,399 | 1,399 | 116 | 33.4 | 0.262 | 0.0142 | [0.0125, 0.0157] |
| TTV9 | C | all counted crest pixels | 1,903 | 1,367 | 1,367 | 122 | 41.3 | 0.294 | 0.0143 | [0.0120, 0.0157] |
| TTV11 | A | all counted crest pixels | 4,889 | 4,122 | 4,122 | 147 | 37.2 | 0.268 | 0.0182 | [0.0174, 0.0188] |
| TTV11 | C | all counted crest pixels | 4,889 | 3,658 | 3,658 | 154 | 41.8 | 0.275 | 0.0189 | [0.0177, 0.0201] |
| TTV15 | A | all counted crest pixels | 4,014 | 3,460 | 3,460 | 144 | 37.4 | 0.262 | 0.0182 | [0.0173, 0.0191] |
| TTV15 | C | all counted crest pixels | 4,014 | 3,622 | 3,622 | 164 | 42.7 | 0.264 | 0.0206 | [0.0189, 0.0220] |

| tile | C_crest | D_form = U/C | D_form at U = 102 | D_form at U = 77 | E_ridge/C (median E) | E_ridge/C (mean E) | D_form/D_phys | U/E_ridge | D_relief/D_form | (D_relief/D_form)(U/E_ridge) |
|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 0.0159 | 0.0041 | 0.0064 | 0.0048 | 0.0017 | 0.0022 | 1.63 | 2.41 | 4.29 | 10.33 |
| TTV2 | 0.0149 | 0.0044 | 0.0068 | 0.0052 | 0.0018 | 0.0023 | 1.74 | 2.41 | 3.65 | 8.79 |
| TTV3 | 0.0147 | 0.0044 | 0.0070 | 0.0052 | 0.0018 | 0.0024 | 1.77 | 2.41 | 5.20 | 12.51 |
| TTV4 | 0.0124 | 0.0053 | 0.0083 | 0.0062 | 0.0022 | 0.0028 | 2.10 | 2.41 | 3.69 | 8.89 |
| TTV5 | 0.0223 | 0.0029 | 0.0046 | 0.0035 | 0.0012 | 0.0016 | 1.17 | 2.41 | 6.38 | 15.37 |
| TTV6 | 0.0139 | 0.0047 | 0.0073 | 0.0055 | 0.0019 | 0.0025 | 1.87 | 2.41 | 4.42 | 10.63 |
| TTV7 | 0.0163 | 0.0040 | 0.0063 | 0.0047 | 0.0017 | 0.0022 | 1.60 | 2.41 | 3.60 | 8.68 |
| TTV8 | 0.0176 | 0.0037 | 0.0058 | 0.0044 | 0.0015 | 0.0020 | 1.48 | 2.41 | 4.72 | 11.35 |
| TTV9 | 0.0212 | 0.0031 | 0.0048 | 0.0036 | 0.0013 | 0.0017 | 1.23 | 2.41 | 4.62 | 11.13 |
| TTV11 | 0.0167 | 0.0039 | 0.0061 | 0.0046 | 0.0016 | 0.0021 | 1.56 | 2.41 | 4.68 | 11.25 |
| TTV15 | 0.0141 | 0.0046 | 0.0073 | 0.0055 | 0.0019 | 0.0025 | 1.85 | 2.41 | 3.94 | 9.48 |

### F9. Fixed runs and the free fits per tile (U = 6.5e-5, m = 0.33; MSE in m²)

| tile | run | network | D | D inside bounds | channel cells | MSE at K0 | K/K0 if K free at this D | MSE then |
|---|---|---|---|---|---|---|---|---|
| TTV1 | F_phys | A | 0.0025 | yes | 5,996 | 8134.07 | 31.43 | 4273.52 |
| TTV1 | F_form | A | 0.0041 | yes | 5,996 | 2185.00 | 1.82 | 1345.94 |
| TTV1 | F_relief | A | 0.0175 | yes | 5,996 | 246.27 | 0.85 | 118.03 |
| TTV1 | F2_phys | C | 0.0025 | yes | 6,580 | 10402.63 | 1779.19 | 6092.79 |
| TTV1 | F2_form | C | 0.0041 | yes | 6,580 | 2916.19 | 1.97 | 1942.90 |
| TTV1 | F2_relief | C | 0.0175 | yes | 6,580 | 230.82 | 0.86 | 116.12 |
| TTV2 | F_phys | A | 0.0025 | yes | 6,932 | 6326.85 | 7.89 | 3417.41 |
| TTV2 | F_form | A | 0.0044 | yes | 6,932 | 1400.18 | 1.69 | 764.65 |
| TTV2 | F_relief | A | 0.0159 | yes | 6,932 | 97.44 | 0.96 | 89.15 |
| TTV2 | F2_phys | C | 0.0025 | yes | 6,690 | 6197.13 | 6.06 | 3497.53 |
| TTV2 | F2_form | C | 0.0044 | yes | 6,690 | 1413.08 | 1.62 | 844.61 |
| TTV2 | F2_relief | C | 0.0159 | yes | 6,690 | 123.96 | 0.94 | 110.67 |
| TTV3 | F_phys | A | 0.0025 | yes | 8,741 | 5014.47 | 758.62 | 3214.55 |
| TTV3 | F_form | A | 0.0044 | yes | 8,741 | 1262.29 | 3.59 | 807.05 |
| TTV3 | F_relief | A | 0.0230 | yes | 8,741 | 176.08 | 0.89 | 162.59 |
| TTV3 | F2_phys | C | 0.0025 | yes | 7,169 | 8446.26 | 758.62 | 6165.12 |
| TTV3 | F2_form | C | 0.0044 | yes | 7,169 | 2276.06 | 758.62 | 1526.26 |
| TTV3 | F2_relief | C | 0.0230 | yes | 7,169 | 169.17 | 0.91 | 161.52 |
| TTV4 | F_phys | A | 0.0025 | yes | 7,469 | 4001.66 | 2.59 | 2244.69 |
| TTV4 | F_form | A | 0.0053 | yes | 7,469 | 667.27 | 1.28 | 442.10 |
| TTV4 | F_relief | A | 0.0194 | yes | 7,469 | 164.55 | 0.96 | 156.89 |
| TTV4 | F2_phys | C | 0.0025 | yes | 6,121 | 8217.21 | 11.57 | 5240.84 |
| TTV4 | F2_form | C | 0.0053 | yes | 6,121 | 1425.54 | 1.50 | 1030.95 |
| TTV4 | F2_relief | C | 0.0194 | yes | 6,121 | 144.60 | 0.95 | 135.32 |
| TTV5 | F_phys | A | 0.0025 | yes | 6,662 | 4726.49 | 919.08 | 2630.44 |
| TTV5 | F_form | A | 0.0029 | yes | 6,662 | 3317.19 | 919.08 | 1875.31 |
| TTV5 | F_relief | A | 0.0186 | yes | 6,662 | 641.99 | 0.81 | 572.03 |
| TTV5 | F2_phys | C | 0.0025 | yes | 4,575 | 47455.98 | 919.08 | 40958.33 |
| TTV5 | F2_form | C | 0.0029 | yes | 4,575 | 33929.66 | 919.08 | 28642.40 |
| TTV5 | F2_relief | C | 0.0186 | yes | 4,575 | 707.75 | 0.94 | 704.67 |
| TTV6 | F_phys | A | 0.0025 | yes | 5,629 | 14888.57 | 1607.73 | 8802.16 |
| TTV6 | F_form | A | 0.0047 | yes | 5,629 | 2974.86 | 2.25 | 1916.99 |
| TTV6 | F_relief | A | 0.0207 | yes | 5,629 | 242.89 | 0.85 | 128.86 |
| TTV6 | F2_phys | C | 0.0025 | yes | 4,328 | 24660.23 | 1607.73 | 15695.07 |
| TTV6 | F2_form | C | 0.0047 | yes | 4,328 | 5328.24 | 5.54 | 3195.04 |
| TTV6 | F2_relief | C | 0.0207 | yes | 4,328 | 205.21 | 0.87 | 134.84 |
| TTV7 | F_phys | A | 0.0025 | yes | 5,859 | 6718.06 | 12.75 | 3319.31 |
| TTV7 | F_form | A | 0.0040 | yes | 5,859 | 2170.33 | 1.86 | 1311.84 |
| TTV7 | F_relief | A | 0.0144 | yes | 5,859 | 260.38 | 0.92 | 227.56 |
| TTV7 | F2_phys | C | 0.0025 | yes | 4,930 | 21349.41 | 1853.56 | 13080.74 |
| TTV7 | F2_form | C | 0.0040 | yes | 4,930 | 6935.86 | 103.63 | 3964.69 |
| TTV7 | F2_relief | C | 0.0144 | yes | 4,930 | 280.94 | 0.94 | 267.57 |
| TTV8 | F_phys | A | 0.0025 | yes | 5,671 | 7239.65 | 1739.07 | 2975.85 |
| TTV8 | F_form | A | 0.0037 | yes | 5,671 | 2635.37 | 2.43 | 1255.78 |
| TTV8 | F_relief | A | 0.0174 | yes | 5,671 | 260.44 | 0.87 | 176.95 |
| TTV8 | F2_phys | C | 0.0025 | yes | 4,972 | 36182.29 | 1739.07 | 23891.08 |
| TTV8 | F2_form | C | 0.0037 | yes | 4,972 | 14139.42 | 1739.07 | 8018.05 |
| TTV8 | F2_relief | C | 0.0174 | yes | 4,972 | 257.23 | 0.88 | 202.87 |
| TTV9 | F_phys | A | 0.0025 | yes | 7,270 | 4144.31 | 1.75 | 3071.28 |
| TTV9 | F_form | A | 0.0031 | yes | 7,270 | 2561.70 | 1.41 | 2061.43 |
| TTV9 | F_relief | A | 0.0142 | yes | 7,270 | 519.98 | 0.85 | 334.88 |
| TTV9 | F2_phys | C | 0.0025 | yes | 8,727 | 3892.16 | 1.59 | 3064.79 |
| TTV9 | F2_form | C | 0.0031 | yes | 8,727 | 2438.63 | 1.34 | 2044.81 |
| TTV9 | F2_relief | C | 0.0142 | yes | 8,727 | 487.72 | 0.88 | 365.97 |
| TTV11 | F_phys | A | 0.0025 | yes | 6,774 | 10275.29 | 1725.29 | 5253.78 |
| TTV11 | F_form | A | 0.0039 | yes | 6,774 | 3308.32 | 2.30 | 1942.34 |
| TTV11 | F_relief | A | 0.0182 | yes | 6,774 | 387.03 | 0.85 | 261.36 |
| TTV11 | F2_phys | C | 0.0025 | yes | 6,194 | 18774.13 | 1725.29 | 10690.62 |
| TTV11 | F2_form | C | 0.0039 | yes | 6,194 | 6251.60 | 7.06 | 3566.64 |
| TTV11 | F2_relief | C | 0.0182 | yes | 6,194 | 361.11 | 0.83 | 213.16 |
| TTV15 | F_phys | A | 0.0025 | yes | 5,778 | 10012.37 | 1457.92 | 4031.68 |
| TTV15 | F_form | A | 0.0046 | yes | 5,778 | 2192.76 | 3.08 | 800.04 |
| TTV15 | F_relief | A | 0.0182 | yes | 5,778 | 197.76 | 0.97 | 194.88 |
| TTV15 | F2_phys | C | 0.0025 | yes | 5,418 | 42252.65 | 1457.92 | 30573.04 |
| TTV15 | F2_form | C | 0.0046 | yes | 5,418 | 9698.77 | 1457.92 | 5337.67 |
| TTV15 | F2_relief | C | 0.0182 | yes | 5,418 | 203.78 | 0.95 | 198.00 |

| tile | fit | C | channel cells | no chi | K | K/K0 | D | where | MSE | held-out mean | D (K = K0) | MSE (K = K0) | K at m 0.30 | D at m 0.30 | MSE at m 0.30 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | PF_A | 6120 | 5,996 | 0 | 8.150e-06 | 0.86 | 0.01646 | interior | 117.56 | 185.26 | 0.01132 | 166.54 | 1.144e-05 | 0.01577 | 118.53 |
| TTV1 | PF_C | 876.3 | 6,580 | 0 | 8.028e-06 | 0.85 | 0.01869 | interior | 115.47 | 148.04 | 0.01261 | 181.83 | 1.124e-05 | 0.01809 | 119.97 |
| TTV2 | PF_A | 6120 | 6,932 | 0 | 1.360e-05 | 1.02 | 0.01151 | interior | 69.54 | 322.78 | 0.01194 | 70.27 | 1.965e-05 | 0.01116 | 76.05 |
| TTV2 | PF_C | 876.3 | 6,690 | 0 | 1.316e-05 | 0.99 | 0.01255 | interior | 101.32 | 409.99 | 0.01218 | 101.73 | 1.899e-05 | 0.01213 | 116.35 |
| TTV3 | PF_A | 6120 | 8,741 | 0 | 2.159e-05 | 0.97 | 0.01573 | interior | 152.49 | 160.96 | 0.01505 | 152.97 | 3.167e-05 | 0.01522 | 158.28 |
| TTV3 | PF_C | 876.3 | 7,169 | 0 | 1.978e-05 | 0.89 | 0.02496 | interior | 161.07 | 165.25 | 0.02071 | 167.60 | 2.910e-05 | 0.02413 | 169.65 |
| TTV4 | PF_A | 6120 | 7,469 | 0 | 1.344e-05 | 0.99 | 0.01518 | interior | 153.00 | 178.82 | 0.01444 | 153.45 | 1.899e-05 | 0.015 | 142.91 |
| TTV4 | PF_C | 876.3 | 6,121 | 0 | 1.252e-05 | 0.92 | 0.02633 | interior | 128.21 | 139.67 | 0.01956 | 144.59 | 1.784e-05 | 0.0257 | 130.16 |
| TTV5 | PF_A | 6120 | 6,662 | 0 | 1.625e-05 | 0.88 | 0.01302 | interior | 562.13 | 1763.72 | 0.01068 | 571.14 | 2.343e-05 | 0.01245 | 580.75 |
| TTV5 | PF_C | 876.3 | 4,575 | 0 | 1.223e-05 | 0.67 | 0.04221 | y_lo | 422.51 | 1160.54 | 0.02958 | 569.74 | 1.744e-05 | 0.04221 | 448.40 |
| TTV6 | PF_A | 6120 | 5,629 | 0 | 8.918e-06 | 0.85 | 0.02022 | interior | 128.78 | 349.31 | 0.01436 | 182.01 | 1.231e-05 | 0.01981 | 120.98 |
| TTV6 | PF_C | 876.3 | 4,328 | 0 | 8.605e-06 | 0.82 | 0.02765 | interior | 120.29 | 214.20 | 0.01785 | 193.16 | 1.178e-05 | 0.02757 | 113.77 |
| TTV7 | PF_A | 6120 | 5,859 | 0 | 7.448e-06 | 0.82 | 0.03821 | interior | 185.60 | 202.60 | 0.01394 | 260.05 | 1.036e-05 | 0.03552 | 175.67 |
| TTV7 | PF_C | 876.3 | 4,930 | 0 | 6.905e-06 | 0.76 | 0.03818 | interior | 124.59 | 173.06 | 0.01736 | 255.71 | 9.637e-06 | 0.03672 | 122.76 |
| TTV8 | PF_A | 6120 | 5,671 | 0 | 8.482e-06 | 0.87 | 0.01743 | interior | 176.95 | 197.01 | 0.01172 | 204.86 | 1.177e-05 | 0.01686 | 163.02 |
| TTV8 | PF_C | 876.3 | 4,972 | 0 | 7.375e-06 | 0.76 | 0.03075 | interior | 124.43 | 200.80 | 0.0175 | 257.22 | 1.031e-05 | 0.02971 | 125.24 |
| TTV9 | PF_A | 6120 | 7,270 | 0 | 7.181e-06 | 0.81 | 0.02655 | interior | 312.12 | 428.93 | 0.009946 | 483.34 | 1.060e-05 | 0.02149 | 272.57 |
| TTV9 | PF_C | 876.3 | 8,727 | 0 | 7.562e-06 | 0.85 | 0.02037 | interior | 355.81 | 590.54 | 0.01063 | 467.49 | 1.106e-05 | 0.01811 | 314.69 |
| TTV11 | PF_A | 6120 | 6,774 | 0 | 8.420e-06 | 0.86 | 0.0174 | interior | 261.10 | 341.58 | 0.01205 | 309.09 | 1.151e-05 | 0.01735 | 245.52 |
| TTV11 | PF_C | 876.3 | 6,194 | 0 | 7.493e-06 | 0.77 | 0.02853 | interior | 185.94 | 245.63 | 0.01436 | 324.54 | 1.032e-05 | 0.0279 | 183.43 |
| TTV15 | PF_A | 6120 | 5,778 | 0 | 1.205e-05 | 1.04 | 0.01412 | interior | 187.56 | 314.11 | 0.01541 | 189.09 | 1.692e-05 | 0.01381 | 192.17 |
| TTV15 | PF_C | 876.3 | 5,418 | 0 | 8.932e-06 | 0.77 | 0.03079 | interior | 132.98 | 174.15 | 0.02012 | 194.48 | 1.242e-05 | 0.03041 | 137.47 |

### F10. SJ-C per tile

| tile | C range | unique values | grid | refine (targets / evaluated) | evaluations | s per evaluation | C opt | 12 m heads | channel cells | K | K/K0 | D | where | MSE | 1 % interval of C | held-out mean | monotone n(C) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 208.8-1.506e+04 | 1,436 | 400 | 1 / 1 | 401 | 7.2 | 3706 | 22 | 3,105 | 8.509e-06 | 0.90 | 0.02697 | interior | 84.44 | [[3705.9, 3705.9]] | 169.39 | yes |
| TTV2 | 225.8-1.013e+04 | 1,271 | 400 | 28 / 28 | 428 | 6.0 | 558.6 | 131 | 8,910 | 1.309e-05 | 0.98 | 0.01042 | interior | 87.31 | [[540.5, 544.8], [553.4, 569.2], [637.6, 649.8]] | 330.47 | yes |
| TTV3 | 141.3-3077 | 3,206 | 400 | 5 / 5 | 405 | 6.2 | 1482 | 55 | 5,409 | 2.049e-05 | 0.92 | 0.03101 | interior | 138.43 | [[1481.9, 1560.1]] | 225.73 | yes |
| TTV4 | 155.8-5957 | 1,393 | 400 | 0 / 0 | 400 | 6.3 | 4067 | 14 | 3,686 | 1.297e-05 | 0.95 | 0.04221 | y_lo | 80.31 | [[3549.1, 4315.5]] | 90.90 | no |
| TTV5 | 153.5-5034 | 1,013 | 400 | 5 / 5 | 405 | 7.1 | 1474 | 29 | 3,462 | 1.191e-05 | 0.65 | 0.04221 | y_lo | 380.29 | [[1371.2, 1493.2]] | 1200.86 | yes |
| TTV6 | 159.2-8559 | 1,595 | 400 | 6 / 6 | 406 | 6.7 | 550.5 | 108 | 6,047 | 8.378e-06 | 0.80 | 0.03268 | interior | 109.86 | [[550.5, 583.0]] | 250.53 | no |
| TTV7 | 166.2-1.272e+04 | 981 | 400 | 0 / 0 | 400 | 6.7 | 5644 | 8 | 1,778 | 6.440e-06 | 0.71 | 0.04221 | y_lo | 89.28 | [[3134.6, 3687.3], [3761.9, 3761.9], [5644.2, 5644.2]] | 108.85 | yes |
| TTV8 | 213.3-1.151e+04 | 1,787 | 400 | 51 / 51 | 451 | 7.9 | 408.6 | 298 | 8,808 | 7.942e-06 | 0.82 | 0.02902 | interior | 108.47 | [[404.0, 411.9], [421.3, 421.3]] | 192.68 | yes |
| TTV9 | 356.8-1.522e+04 | 2,411 | 400 | 0 / 0 | 400 | 7.3 | 5402 | 37 | 4,191 | 6.491e-06 | 0.73 | 0.03866 | interior | 116.57 | [[4956.2, 6495.3]] | 129.64 | no |
| TTV11 | 190.6-1.159e+04 | 1,132 | 400 | 3 / 3 | 403 | 7.6 | 855.1 | 63 | 6,232 | 7.494e-06 | 0.77 | 0.02854 | interior | 185.47 | [[841.2, 841.2], [849.9, 954.8]] | 287.97 | yes |
| TTV15 | 198.9-6972 | 1,270 | 400 | 0 / 0 | 400 | 7.4 | 2219 | 32 | 3,386 | 9.356e-06 | 0.81 | 0.03087 | interior | 109.00 | [[2206.5, 2287.9]] | 193.02 | yes |

### F11. Head extraction (decision 1.4) on every surface, per tile

| tile | surface | set | channel cells / skeleton px | heads | heads in polygon | picks in polygon | ratio | median pick dist (m) | share <= 25 m | line share <= 15 m | channel share <= 15 m of line | reading |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | obs | a2 | 489 | 69 | 43 | 22 | 1.95 | 16.1 | 0.68 | 0.64 | 0.83 | too many heads |
| TTV1 | obs | a0 | 440 | 39 | 22 | 22 | 1.00 | 18.8 | 0.55 | 0.62 | 0.80 | locates the mapped heads |
| TTV1 | obs | perron | 4,801 | 20 | 9 | 22 | 0.41 | 78.9 | 0.36 | 0.40 | 0.85 | too few heads, distances large |
| TTV1 | F_phys | a2 | 2,807 | 715 | 456 | 22 | 20.73 | 0.0 | 0.95 | 0.88 | 0.50 | too many heads |
| TTV1 | F_phys | a0 | 396 | 28 | 16 | 22 | 0.73 | 24.4 | 0.50 | 0.56 | 0.79 | locates the mapped heads |
| TTV1 | F_phys | perron | 4,835 | 27 | 16 | 22 | 0.73 | 29.3 | 0.41 | 0.48 | 0.86 | count in range, distances large |
| TTV1 | F_form | a2 | 1,747 | 470 | 299 | 22 | 13.59 | 3.5 | 0.77 | 0.81 | 0.58 | too many heads |
| TTV1 | F_form | a0 | 398 | 29 | 16 | 22 | 0.73 | 24.4 | 0.50 | 0.57 | 0.79 | locates the mapped heads |
| TTV1 | F_form | perron | 4,960 | 25 | 14 | 22 | 0.64 | 32.3 | 0.41 | 0.47 | 0.85 | too few heads, distances large |
| TTV1 | F_relief | a2 | 356 | 41 | 25 | 22 | 1.14 | 35.1 | 0.41 | 0.54 | 0.83 | count in range, distances large |
| TTV1 | F_relief | a0 | 362 | 31 | 17 | 22 | 0.77 | 35.8 | 0.41 | 0.52 | 0.79 | count in range, distances large |
| TTV1 | F_relief | perron | 4,715 | 23 | 12 | 22 | 0.55 | 59.3 | 0.27 | 0.44 | 0.89 | too few heads, distances large |
| TTV1 | F2_phys | a2 | 2,923 | 679 | 422 | 22 | 19.18 | 0.0 | 0.91 | 0.87 | 0.48 | too many heads |
| TTV1 | F2_phys | a0 | 415 | 33 | 19 | 22 | 0.86 | 29.2 | 0.50 | 0.58 | 0.82 | count in range, distances large |
| TTV1 | F2_phys | perron | 4,626 | 24 | 13 | 22 | 0.59 | 44.4 | 0.36 | 0.48 | 0.91 | too few heads, distances large |
| TTV1 | F2_form | a2 | 1,910 | 449 | 270 | 22 | 12.27 | 2.5 | 0.91 | 0.81 | 0.54 | too many heads |
| TTV1 | F2_form | a0 | 411 | 32 | 18 | 22 | 0.82 | 35.7 | 0.41 | 0.58 | 0.82 | count in range, distances large |
| TTV1 | F2_form | perron | 4,777 | 22 | 12 | 22 | 0.55 | 43.9 | 0.36 | 0.48 | 0.90 | too few heads, distances large |
| TTV1 | F2_relief | a2 | 406 | 55 | 34 | 22 | 1.55 | 21.6 | 0.55 | 0.57 | 0.82 | too many heads |
| TTV1 | F2_relief | a0 | 369 | 29 | 15 | 22 | 0.68 | 31.8 | 0.45 | 0.54 | 0.81 | count in range, distances large |
| TTV1 | F2_relief | perron | 4,397 | 22 | 10 | 22 | 0.45 | 78.4 | 0.27 | 0.42 | 0.89 | too few heads, distances large |
| TTV1 | PF_A | a2 | 412 | 51 | 25 | 22 | 1.14 | 28.9 | 0.45 | 0.56 | 0.80 | count in range, distances large |
| TTV1 | PF_A | a0 | 360 | 31 | 18 | 22 | 0.82 | 34.7 | 0.45 | 0.52 | 0.78 | count in range, distances large |
| TTV1 | PF_A | perron | 4,455 | 23 | 14 | 22 | 0.64 | 68.2 | 0.18 | 0.42 | 0.88 | too few heads, distances large |
| TTV1 | PF_C | a2 | 397 | 48 | 22 | 22 | 1.00 | 28.9 | 0.41 | 0.56 | 0.82 | count in range, distances large |
| TTV1 | PF_C | a0 | 370 | 33 | 18 | 22 | 0.82 | 34.7 | 0.41 | 0.52 | 0.78 | count in range, distances large |
| TTV1 | PF_C | perron | 4,218 | 25 | 15 | 22 | 0.68 | 70.7 | 0.27 | 0.43 | 0.87 | count in range, distances large |
| TTV1 | SJ_C | a2 | 473 | 90 | 49 | 22 | 2.23 | 59.8 | 0.09 | 0.44 | 0.74 | too many heads, distances large |
| TTV1 | SJ_C | a0 | 311 | 29 | 17 | 22 | 0.77 | 69.0 | 0.14 | 0.41 | 0.73 | count in range, distances large |
| TTV1 | SJ_C | perron | 3,096 | 25 | 14 | 22 | 0.64 | 93.1 | 0.09 | 0.34 | 0.87 | too few heads, distances large |
| TTV1 | P4 | a2 | 497 | 98 | 64 | 22 | 2.91 | 77.9 | 0.09 | 0.36 | 0.69 | too many heads, distances large |
| TTV1 | P4 | a0 | 308 | 35 | 22 | 22 | 1.00 | 71.3 | 0.09 | 0.35 | 0.69 | count in range, distances large |
| TTV1 | P4 | perron | 2,818 | 15 | 10 | 22 | 0.45 | 118.7 | 0.05 | 0.21 | 0.84 | too few heads, distances large |
| TTV1 | AD | a2 | 338 | 26 | 14 | 22 | 0.64 | 62.4 | 0.23 | 0.44 | 0.85 | too few heads, distances large |
| TTV1 | AD | a0 | 465 | 46 | 25 | 22 | 1.14 | 21.2 | 0.59 | 0.57 | 0.79 | locates the mapped heads |
| TTV1 | AD | perron | 1,185 | 4 | 3 | 22 | 0.14 | 178.0 | 0.00 | 0.10 | 0.86 | too few heads, distances large |
| TTV1 | obs_full_lattice | a2 | - | - | 49 | 22 | 2.23 | 12.7 | 0.82 | 0.71 | 0.83 | too many heads |
| TTV1 | obs_full_lattice | a0 | - | - | 25 | 22 | 1.14 | 15.9 | 0.59 | 0.66 | 0.81 | locates the mapped heads |
| TTV1 | obs_full_lattice | a1.65 | - | - | 37 | 22 | 1.68 | 12.7 | 0.77 | 0.69 | 0.83 | too many heads |
| TTV2 | obs | a2 | 473 | 45 | 37 | 34 | 1.09 | 25.3 | 0.50 | 0.63 | 0.82 | count in range, distances large |
| TTV2 | obs | a0 | 479 | 40 | 36 | 34 | 1.06 | 23.3 | 0.50 | 0.65 | 0.80 | locates the mapped heads |
| TTV2 | obs | perron | 4,175 | 13 | 8 | 34 | 0.24 | 79.2 | 0.09 | 0.35 | 0.85 | too few heads, distances large |
| TTV2 | F_phys | a2 | 2,514 | 658 | 579 | 34 | 17.03 | 0.0 | 0.97 | 0.90 | 0.47 | too many heads |
| TTV2 | F_phys | a0 | 438 | 33 | 27 | 34 | 0.79 | 27.4 | 0.44 | 0.62 | 0.80 | count in range, distances large |
| TTV2 | F_phys | perron | 5,754 | 29 | 25 | 34 | 0.74 | 34.0 | 0.38 | 0.51 | 0.85 | count in range, distances large |
| TTV2 | F_form | a2 | 1,367 | 382 | 347 | 34 | 10.21 | 2.6 | 0.79 | 0.82 | 0.56 | too many heads |
| TTV2 | F_form | a0 | 434 | 32 | 27 | 34 | 0.79 | 27.7 | 0.41 | 0.62 | 0.80 | count in range, distances large |
| TTV2 | F_form | perron | 4,617 | 18 | 15 | 34 | 0.44 | 61.9 | 0.21 | 0.44 | 0.90 | too few heads, distances large |
| TTV2 | F_relief | a2 | 421 | 49 | 44 | 34 | 1.29 | 29.7 | 0.47 | 0.59 | 0.81 | count in range, distances large |
| TTV2 | F_relief | a0 | 396 | 29 | 25 | 34 | 0.74 | 34.3 | 0.35 | 0.58 | 0.81 | count in range, distances large |
| TTV2 | F_relief | perron | 4,652 | 21 | 10 | 34 | 0.29 | 62.1 | 0.15 | 0.41 | 0.84 | too few heads, distances large |
| TTV2 | F2_phys | a2 | 2,510 | 660 | 578 | 34 | 17.00 | 0.0 | 0.97 | 0.91 | 0.47 | too many heads |
| TTV2 | F2_phys | a0 | 443 | 30 | 25 | 34 | 0.74 | 28.2 | 0.44 | 0.63 | 0.81 | count in range, distances large |
| TTV2 | F2_phys | perron | 5,448 | 23 | 20 | 34 | 0.59 | 36.4 | 0.38 | 0.51 | 0.89 | too few heads, distances large |
| TTV2 | F2_form | a2 | 1,397 | 384 | 346 | 34 | 10.18 | 1.6 | 0.91 | 0.83 | 0.58 | too many heads |
| TTV2 | F2_form | a0 | 446 | 28 | 24 | 34 | 0.71 | 28.4 | 0.41 | 0.62 | 0.78 | count in range, distances large |
| TTV2 | F2_form | perron | 4,674 | 18 | 16 | 34 | 0.47 | 55.0 | 0.29 | 0.46 | 0.92 | too few heads, distances large |
| TTV2 | F2_relief | a2 | 414 | 44 | 40 | 34 | 1.18 | 24.0 | 0.53 | 0.58 | 0.81 | locates the mapped heads |
| TTV2 | F2_relief | a0 | 395 | 27 | 23 | 34 | 0.68 | 32.5 | 0.32 | 0.58 | 0.82 | count in range, distances large |
| TTV2 | F2_relief | perron | 4,274 | 23 | 12 | 34 | 0.35 | 62.1 | 0.15 | 0.37 | 0.88 | too few heads, distances large |
| TTV2 | PF_A | a2 | 503 | 70 | 65 | 34 | 1.91 | 18.5 | 0.56 | 0.64 | 0.80 | too many heads |
| TTV2 | PF_A | a0 | 413 | 31 | 26 | 34 | 0.76 | 34.3 | 0.38 | 0.60 | 0.81 | count in range, distances large |
| TTV2 | PF_A | perron | 5,063 | 25 | 20 | 34 | 0.59 | 53.6 | 0.18 | 0.45 | 0.87 | too few heads, distances large |
| TTV2 | PF_C | a2 | 460 | 59 | 53 | 34 | 1.56 | 25.3 | 0.50 | 0.60 | 0.79 | too many heads, distances large |
| TTV2 | PF_C | a0 | 404 | 28 | 24 | 34 | 0.71 | 29.9 | 0.38 | 0.59 | 0.82 | count in range, distances large |
| TTV2 | PF_C | perron | 4,998 | 24 | 16 | 34 | 0.47 | 50.2 | 0.29 | 0.43 | 0.87 | too few heads, distances large |
| TTV2 | SJ_C | a2 | 478 | 58 | 53 | 34 | 1.56 | 12.0 | 0.56 | 0.62 | 0.78 | too many heads |
| TTV2 | SJ_C | a0 | 425 | 33 | 29 | 34 | 0.85 | 28.2 | 0.47 | 0.65 | 0.83 | count in range, distances large |
| TTV2 | SJ_C | perron | 5,335 | 25 | 17 | 34 | 0.50 | 42.6 | 0.32 | 0.47 | 0.88 | too few heads, distances large |
| TTV2 | P4 | a2 | 498 | 79 | 72 | 34 | 2.12 | 24.6 | 0.50 | 0.61 | 0.82 | too many heads |
| TTV2 | P4 | a0 | 380 | 28 | 23 | 34 | 0.68 | 41.3 | 0.29 | 0.56 | 0.80 | count in range, distances large |
| TTV2 | P4 | perron | 5,099 | 28 | 23 | 34 | 0.68 | 47.9 | 0.12 | 0.44 | 0.86 | count in range, distances large |
| TTV2 | AD | a2 | 586 | 80 | 72 | 34 | 2.12 | 33.7 | 0.47 | 0.60 | 0.63 | too many heads, distances large |
| TTV2 | AD | a0 | 501 | 40 | 34 | 34 | 1.00 | 11.4 | 0.68 | 0.71 | 0.79 | locates the mapped heads |
| TTV2 | AD | perron | 4,818 | 19 | 16 | 34 | 0.47 | 56.5 | 0.21 | 0.45 | 0.90 | too few heads, distances large |
| TTV2 | obs_full_lattice | a2 | - | - | 41 | 34 | 1.21 | 13.0 | 0.74 | 0.69 | 0.80 | locates the mapped heads |
| TTV2 | obs_full_lattice | a0 | - | - | 44 | 34 | 1.29 | 14.8 | 0.59 | 0.70 | 0.78 | locates the mapped heads |
| TTV2 | obs_full_lattice | a1.65 | - | - | 41 | 34 | 1.21 | 13.3 | 0.74 | 0.69 | 0.80 | locates the mapped heads |
| TTV3 | obs | a2 | 343 | 19 | 6 | 5 | 1.20 | 53.5 | 0.40 | 0.48 | 0.70 | count in range, distances large |
| TTV3 | obs | a0 | 570 | 43 | 20 | 5 | 4.00 | 11.2 | 0.60 | 0.62 | 0.47 | too many heads |
| TTV3 | obs | perron | 4,124 | 8 | 4 | 5 | 0.80 | 178.5 | 0.00 | 0.32 | 0.61 | count in range, distances large |
| TTV3 | F_phys | a2 | 2,383 | 612 | 257 | 5 | 51.40 | 0.0 | 0.80 | 0.82 | 0.37 | too many heads |
| TTV3 | F_phys | a0 | 484 | 30 | 10 | 5 | 2.00 | 7.0 | 0.80 | 0.62 | 0.56 | too many heads |
| TTV3 | F_phys | perron | 6,504 | 22 | 10 | 5 | 2.00 | 9.0 | 0.80 | 0.57 | 0.52 | too many heads |
| TTV3 | F_form | a2 | 1,268 | 326 | 117 | 5 | 23.40 | 0.0 | 0.80 | 0.71 | 0.48 | too many heads |
| TTV3 | F_form | a0 | 460 | 28 | 10 | 5 | 2.00 | 7.0 | 0.60 | 0.60 | 0.57 | too many heads |
| TTV3 | F_form | perron | 6,316 | 21 | 10 | 5 | 2.00 | 8.2 | 0.80 | 0.58 | 0.56 | too many heads |
| TTV3 | F_relief | a2 | 279 | 17 | 6 | 5 | 1.20 | 7.0 | 0.60 | 0.49 | 0.65 | locates the mapped heads |
| TTV3 | F_relief | a0 | 407 | 27 | 12 | 5 | 2.40 | 7.0 | 0.60 | 0.55 | 0.59 | too many heads |
| TTV3 | F_relief | perron | 4,777 | 17 | 7 | 5 | 1.40 | 92.2 | 0.40 | 0.48 | 0.65 | count in range, distances large |
| TTV3 | F2_phys | a2 | 2,745 | 631 | 254 | 5 | 50.80 | 0.0 | 0.80 | 0.79 | 0.32 | too many heads |
| TTV3 | F2_phys | a0 | 466 | 26 | 8 | 5 | 1.60 | 10.0 | 0.60 | 0.58 | 0.59 | too many heads |
| TTV3 | F2_phys | perron | 5,745 | 20 | 9 | 5 | 1.80 | 22.8 | 0.60 | 0.55 | 0.55 | too many heads |
| TTV3 | F2_form | a2 | 1,554 | 398 | 168 | 5 | 33.60 | 0.0 | 0.80 | 0.72 | 0.40 | too many heads |
| TTV3 | F2_form | a0 | 433 | 23 | 9 | 5 | 1.80 | 10.0 | 0.80 | 0.58 | 0.62 | too many heads |
| TTV3 | F2_form | perron | 5,492 | 21 | 9 | 5 | 1.80 | 22.8 | 0.60 | 0.56 | 0.61 | too many heads |
| TTV3 | F2_relief | a2 | 316 | 25 | 9 | 5 | 1.80 | 10.0 | 0.60 | 0.49 | 0.72 | too many heads |
| TTV3 | F2_relief | a0 | 375 | 23 | 7 | 5 | 1.40 | 35.7 | 0.40 | 0.54 | 0.65 | count in range, distances large |
| TTV3 | F2_relief | perron | 5,024 | 22 | 8 | 5 | 1.60 | 30.4 | 0.40 | 0.54 | 0.75 | too many heads, distances large |
| TTV3 | PF_A | a2 | 398 | 28 | 10 | 5 | 2.00 | 7.0 | 0.80 | 0.57 | 0.60 | too many heads |
| TTV3 | PF_A | a0 | 411 | 27 | 11 | 5 | 2.20 | 7.0 | 0.60 | 0.58 | 0.59 | too many heads |
| TTV3 | PF_A | perron | 4,459 | 13 | 6 | 5 | 1.20 | 89.5 | 0.40 | 0.46 | 0.68 | count in range, distances large |
| TTV3 | PF_C | a2 | 265 | 19 | 8 | 5 | 1.60 | 10.0 | 0.60 | 0.39 | 0.71 | too many heads |
| TTV3 | PF_C | a0 | 361 | 25 | 11 | 5 | 2.20 | 35.7 | 0.40 | 0.54 | 0.67 | too many heads, distances large |
| TTV3 | PF_C | perron | 4,588 | 18 | 6 | 5 | 1.20 | 40.7 | 0.40 | 0.52 | 0.80 | count in range, distances large |
| TTV3 | SJ_C | a2 | 246 | 22 | 8 | 5 | 1.60 | 64.8 | 0.00 | 0.35 | 0.72 | too many heads, distances large |
| TTV3 | SJ_C | a0 | 321 | 23 | 11 | 5 | 2.20 | 64.4 | 0.00 | 0.50 | 0.68 | too many heads, distances large |
| TTV3 | SJ_C | perron | 4,156 | 16 | 7 | 5 | 1.40 | 86.9 | 0.00 | 0.49 | 0.79 | count in range, distances large |
| TTV3 | P4 | a2 | 246 | 42 | 11 | 5 | 2.20 | 180.9 | 0.00 | 0.19 | 0.53 | too many heads, distances large |
| TTV3 | P4 | a0 | 242 | 21 | 8 | 5 | 1.60 | 191.3 | 0.00 | 0.28 | 0.52 | too many heads, distances large |
| TTV3 | P4 | perron | 3,028 | 10 | 4 | 5 | 0.80 | 167.2 | 0.00 | 0.29 | 0.65 | count in range, distances large |
| TTV3 | AD | a2 | 913 | 131 | 53 | 5 | 10.60 | 11.2 | 0.80 | 0.75 | 0.47 | too many heads |
| TTV3 | AD | a0 | 568 | 42 | 18 | 5 | 3.60 | 3.0 | 0.80 | 0.69 | 0.51 | too many heads |
| TTV3 | AD | perron | 8,996 | 33 | 15 | 5 | 3.00 | 16.3 | 0.60 | 0.58 | 0.47 | too many heads |
| TTV3 | obs_full_lattice | a2 | - | - | 6 | 5 | 1.20 | 0.0 | 0.80 | 0.74 | 0.71 | locates the mapped heads |
| TTV3 | obs_full_lattice | a0 | - | - | 18 | 5 | 3.60 | 0.0 | 0.80 | 0.80 | 0.52 | too many heads |
| TTV3 | obs_full_lattice | a1.65 | - | - | 6 | 5 | 1.20 | 0.0 | 0.80 | 0.74 | 0.71 | locates the mapped heads |
| TTV4 | obs | a2 | 412 | 31 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | obs | a0 | 525 | 40 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | obs | perron | 4,854 | 19 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | F_phys | a2 | 2,154 | 690 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | F_phys | a0 | 481 | 21 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | F_phys | perron | 5,764 | 18 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | F_form | a2 | 888 | 228 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | F_form | a0 | 461 | 21 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | F_form | perron | 5,549 | 16 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | F_relief | a2 | 396 | 59 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | F_relief | a0 | 421 | 27 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | F_relief | perron | 4,077 | 14 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | F2_phys | a2 | 2,683 | 743 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | F2_phys | a0 | 432 | 19 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | F2_phys | perron | 4,754 | 13 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | F2_form | a2 | 1,130 | 323 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | F2_form | a0 | 417 | 19 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | F2_form | perron | 4,770 | 15 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | F2_relief | a2 | 403 | 55 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | F2_relief | a0 | 372 | 25 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | F2_relief | perron | 4,145 | 19 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | PF_A | a2 | 437 | 65 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | PF_A | a0 | 440 | 25 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | PF_A | perron | 3,625 | 9 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | PF_C | a2 | 390 | 43 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | PF_C | a0 | 426 | 25 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | PF_C | perron | 3,844 | 15 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | SJ_C | a2 | 428 | 64 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | SJ_C | a0 | 436 | 35 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | SJ_C | perron | 3,086 | 14 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | P4 | a2 | 452 | 59 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | P4 | a0 | 506 | 36 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | P4 | perron | 3,159 | 14 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | AD | a2 | 408 | 14 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | AD | a0 | 639 | 44 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | AD | perron | 3,627 | 14 | 0 | 0 | - | - | - | - | - | - |
| TTV4 | obs_full_lattice | a2 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV4 | obs_full_lattice | a0 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV4 | obs_full_lattice | a1.65 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV5 | obs | a2 | 309 | 36 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | obs | a0 | 453 | 39 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | obs | perron | 7,375 | 26 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | F_phys | a2 | 2,344 | 706 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | F_phys | a0 | 391 | 25 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | F_phys | perron | 5,371 | 20 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | F_form | a2 | 1,997 | 598 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | F_form | a0 | 390 | 25 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | F_form | perron | 5,541 | 21 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | F_relief | a2 | 426 | 86 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | F_relief | a0 | 396 | 30 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | F_relief | perron | 5,147 | 18 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | F2_phys | a2 | 3,850 | 663 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | F2_phys | a0 | 293 | 27 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | F2_phys | perron | 3,622 | 16 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | F2_form | a2 | 3,548 | 618 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | F2_form | a0 | 301 | 28 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | F2_form | perron | 3,562 | 15 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | F2_relief | a2 | 650 | 118 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | F2_relief | a0 | 323 | 33 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | F2_relief | perron | 3,552 | 17 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | PF_A | a2 | 482 | 96 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | PF_A | a0 | 384 | 27 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | PF_A | perron | 4,640 | 20 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | PF_C | a2 | 452 | 76 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | PF_C | a0 | 367 | 40 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | PF_C | perron | 3,091 | 21 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | SJ_C | a2 | 415 | 76 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | SJ_C | a0 | 328 | 37 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | SJ_C | perron | 3,936 | 25 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | P4 | a2 | 455 | 64 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | P4 | a0 | 319 | 35 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | P4 | perron | 2,999 | 17 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | AD | a2 | 1,137 | 221 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | AD | a0 | 527 | 45 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | AD | perron | 7,676 | 36 | 0 | 0 | - | - | - | - | - | - |
| TTV5 | obs_full_lattice | a2 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV5 | obs_full_lattice | a0 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV5 | obs_full_lattice | a1.65 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV6 | obs | a2 | 350 | 33 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | obs | a0 | 521 | 47 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | obs | perron | 3,581 | 17 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | F_phys | a2 | 3,167 | 700 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | F_phys | a0 | 408 | 29 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | F_phys | perron | 4,716 | 21 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | F_form | a2 | 1,758 | 420 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | F_form | a0 | 387 | 30 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | F_form | perron | 4,809 | 23 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | F_relief | a2 | 332 | 47 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | F_relief | a0 | 300 | 28 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | F_relief | perron | 3,993 | 20 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | F2_phys | a2 | 3,684 | 706 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | F2_phys | a0 | 376 | 29 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | F2_phys | perron | 3,583 | 19 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | F2_form | a2 | 2,191 | 512 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | F2_form | a0 | 350 | 28 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | F2_form | perron | 3,187 | 15 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | F2_relief | a2 | 361 | 67 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | F2_relief | a0 | 280 | 26 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | F2_relief | perron | 3,541 | 26 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | PF_A | a2 | 356 | 51 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | PF_A | a0 | 303 | 31 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | PF_A | perron | 3,696 | 24 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | PF_C | a2 | 322 | 57 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | PF_C | a0 | 274 | 31 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | PF_C | perron | 3,743 | 22 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | SJ_C | a2 | 351 | 50 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | SJ_C | a0 | 314 | 36 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | SJ_C | perron | 3,695 | 24 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | P4 | a2 | 294 | 42 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | P4 | a0 | 278 | 28 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | P4 | perron | 2,701 | 22 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | AD | a2 | 380 | 40 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | AD | a0 | 479 | 44 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | AD | perron | 1,866 | 13 | 0 | 0 | - | - | - | - | - | - |
| TTV6 | obs_full_lattice | a2 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV6 | obs_full_lattice | a0 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV6 | obs_full_lattice | a1.65 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV7 | obs | a2 | 424 | 48 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | obs | a0 | 485 | 42 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | obs | perron | 6,199 | 27 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | F_phys | a2 | 2,509 | 711 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | F_phys | a0 | 394 | 27 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | F_phys | perron | 5,197 | 28 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | F_form | a2 | 1,541 | 446 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | F_form | a0 | 400 | 28 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | F_form | perron | 5,288 | 30 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | F_relief | a2 | 502 | 83 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | F_relief | a0 | 389 | 33 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | F_relief | perron | 4,377 | 29 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | F2_phys | a2 | 3,297 | 679 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | F2_phys | a0 | 360 | 30 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | F2_phys | perron | 4,096 | 25 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | F2_form | a2 | 2,306 | 503 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | F2_form | a0 | 359 | 31 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | F2_form | perron | 4,032 | 25 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | F2_relief | a2 | 580 | 105 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | F2_relief | a0 | 377 | 31 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | F2_relief | perron | 3,846 | 28 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | PF_A | a2 | 346 | 45 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | PF_A | a0 | 344 | 37 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | PF_A | perron | 4,734 | 23 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | PF_C | a2 | 383 | 50 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | PF_C | a0 | 368 | 42 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | PF_C | perron | 3,225 | 21 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | SJ_C | a2 | 315 | 64 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | SJ_C | a0 | 272 | 42 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | SJ_C | perron | 3,250 | 28 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | P4 | a2 | 404 | 73 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | P4 | a0 | 274 | 38 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | P4 | perron | 2,789 | 30 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | AD | a2 | 269 | 35 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | AD | a0 | 584 | 56 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | AD | perron | 1,763 | 19 | 0 | 0 | - | - | - | - | - | - |
| TTV7 | obs_full_lattice | a2 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV7 | obs_full_lattice | a0 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV7 | obs_full_lattice | a1.65 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV8 | obs | a2 | 442 | 50 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | obs | a0 | 483 | 34 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | obs | perron | 5,306 | 19 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | F_phys | a2 | 2,649 | 785 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | F_phys | a0 | 381 | 24 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | F_phys | perron | 5,321 | 22 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | F_form | a2 | 1,767 | 533 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | F_form | a0 | 385 | 26 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | F_form | perron | 5,352 | 22 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | F_relief | a2 | 437 | 67 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | F_relief | a0 | 382 | 32 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | F_relief | perron | 4,175 | 22 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | F2_phys | a2 | 3,819 | 680 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | F2_phys | a0 | 348 | 29 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | F2_phys | perron | 4,143 | 24 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | F2_form | a2 | 2,941 | 540 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | F2_form | a0 | 358 | 31 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | F2_form | perron | 4,290 | 27 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | F2_relief | a2 | 504 | 89 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | F2_relief | a0 | 364 | 31 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | F2_relief | perron | 4,196 | 25 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | PF_A | a2 | 430 | 69 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | PF_A | a0 | 359 | 31 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | PF_A | perron | 3,049 | 15 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | PF_C | a2 | 390 | 70 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | PF_C | a0 | 326 | 36 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | PF_C | perron | 3,671 | 23 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | SJ_C | a2 | 416 | 76 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | SJ_C | a0 | 327 | 31 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | SJ_C | perron | 4,585 | 26 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | P4 | a2 | 320 | 60 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | P4 | a0 | 298 | 36 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | P4 | perron | 2,718 | 28 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | AD | a2 | 269 | 48 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | AD | a0 | 330 | 36 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | AD | perron | 1,605 | 14 | 0 | 0 | - | - | - | - | - | - |
| TTV8 | obs_full_lattice | a2 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV8 | obs_full_lattice | a0 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV8 | obs_full_lattice | a1.65 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV9 | obs | a2 | 623 | 103 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | obs | a0 | 466 | 34 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | obs | perron | 6,123 | 18 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | F_phys | a2 | 2,409 | 699 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | F_phys | a0 | 445 | 25 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | F_phys | perron | 5,953 | 19 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | F_form | a2 | 1,998 | 598 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | F_form | a0 | 443 | 26 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | F_form | perron | 5,484 | 16 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | F_relief | a2 | 570 | 116 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | F_relief | a0 | 347 | 26 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | F_relief | perron | 5,226 | 18 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | F2_phys | a2 | 2,348 | 674 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | F2_phys | a0 | 460 | 23 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | F2_phys | perron | 5,358 | 15 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | F2_form | a2 | 1,956 | 562 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | F2_form | a0 | 446 | 23 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | F2_form | perron | 5,604 | 18 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | F2_relief | a2 | 631 | 124 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | F2_relief | a0 | 344 | 23 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | F2_relief | perron | 5,057 | 20 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | PF_A | a2 | 441 | 81 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | PF_A | a0 | 314 | 26 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | PF_A | perron | 4,160 | 17 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | PF_C | a2 | 553 | 111 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | PF_C | a0 | 339 | 28 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | PF_C | perron | 4,069 | 14 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | SJ_C | a2 | 341 | 57 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | SJ_C | a0 | 303 | 37 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | SJ_C | perron | 3,459 | 16 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | P4 | a2 | 329 | 54 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | P4 | a0 | 291 | 30 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | P4 | perron | 2,840 | 18 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | AD | a2 | 461 | 51 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | AD | a0 | 446 | 38 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | AD | perron | 3,181 | 11 | 0 | 0 | - | - | - | - | - | - |
| TTV9 | obs_full_lattice | a2 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV9 | obs_full_lattice | a0 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV9 | obs_full_lattice | a1.65 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV11 | obs | a2 | 474 | 56 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | obs | a0 | 499 | 38 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | obs | perron | 7,029 | 30 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | F_phys | a2 | 2,780 | 635 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | F_phys | a0 | 404 | 26 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | F_phys | perron | 5,568 | 24 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | F_form | a2 | 1,953 | 462 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | F_form | a0 | 407 | 26 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | F_form | perron | 5,432 | 22 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | F_relief | a2 | 426 | 81 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | F_relief | a0 | 334 | 31 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | F_relief | perron | 4,534 | 22 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | F2_phys | a2 | 3,215 | 651 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | F2_phys | a0 | 387 | 22 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | F2_phys | perron | 5,270 | 22 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | F2_form | a2 | 2,328 | 490 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | F2_form | a0 | 390 | 24 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | F2_form | perron | 5,043 | 23 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | F2_relief | a2 | 450 | 85 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | F2_relief | a0 | 345 | 31 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | F2_relief | perron | 3,871 | 22 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | PF_A | a2 | 440 | 83 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | PF_A | a0 | 331 | 36 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | PF_A | perron | 4,577 | 19 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | PF_C | a2 | 401 | 78 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | PF_C | a0 | 333 | 41 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | PF_C | perron | 3,580 | 20 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | SJ_C | a2 | 415 | 78 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | SJ_C | a0 | 337 | 41 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | SJ_C | perron | 3,588 | 20 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | P4 | a2 | 324 | 52 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | P4 | a0 | 330 | 34 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | P4 | perron | 2,489 | 19 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | AD | a2 | 302 | 54 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | AD | a0 | 351 | 47 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | AD | perron | 1,488 | 23 | 0 | 0 | - | - | - | - | - | - |
| TTV11 | obs_full_lattice | a2 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV11 | obs_full_lattice | a0 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV11 | obs_full_lattice | a1.65 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV15 | obs | a2 | 426 | 52 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | obs | a0 | 443 | 42 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | obs | perron | 5,121 | 22 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | F_phys | a2 | 2,918 | 763 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | F_phys | a0 | 369 | 33 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | F_phys | perron | 4,769 | 25 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | F_form | a2 | 1,562 | 421 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | F_form | a0 | 344 | 30 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | F_form | perron | 4,302 | 23 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | F_relief | a2 | 447 | 72 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | F_relief | a0 | 321 | 36 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | F_relief | perron | 2,706 | 17 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | F2_phys | a2 | 3,793 | 666 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | F2_phys | a0 | 322 | 32 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | F2_phys | perron | 4,116 | 31 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | F2_form | a2 | 2,468 | 450 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | F2_form | a0 | 319 | 31 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | F2_form | perron | 4,095 | 30 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | F2_relief | a2 | 491 | 90 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | F2_relief | a0 | 301 | 31 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | F2_relief | perron | 4,099 | 26 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | PF_A | a2 | 499 | 85 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | PF_A | a0 | 354 | 33 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | PF_A | perron | 3,959 | 26 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | PF_C | a2 | 354 | 57 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | PF_C | a0 | 279 | 34 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | PF_C | perron | 3,574 | 26 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | SJ_C | a2 | 345 | 65 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | SJ_C | a0 | 248 | 32 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | SJ_C | perron | 2,849 | 25 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | P4 | a2 | 392 | 95 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | P4 | a0 | 253 | 26 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | P4 | perron | 2,567 | 20 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | AD | a2 | 199 | 32 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | AD | a0 | 413 | 50 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | AD | perron | 2,175 | 9 | 0 | 0 | - | - | - | - | - | - |
| TTV15 | obs_full_lattice | a2 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV15 | obs_full_lattice | a0 | - | - | 0 | 0 | - | - | - | - | - | - |
| TTV15 | obs_full_lattice | a1.65 | - | - | 0 | 0 | - | - | - | - | - | - |

| tile | surface | 12 m cells | 1 m holes (channel cells without chi) | residual MSE | mean | q05 | q95 | wall (s) |
|---|---|---|---|---|---|---|---|---|
| TTV1 | obs | 6,889 | 0 | - | - | - | - | 1 |
| TTV1 | F_phys | 6,889 | 0 | 8134.07 | 64.89 | -13.4 | 189.8 | 13 |
| TTV1 | F_form | 6,889 | 0 | 2185.00 | 31.37 | -18.7 | 95.7 | 13 |
| TTV1 | F_relief | 6,889 | 0 | 246.27 | -9.34 | -32.8 | 8.3 | 13 |
| TTV1 | F2_phys | 6,889 | 0 | 10402.63 | 72.44 | -14.5 | 207.3 | 13 |
| TTV1 | F2_form | 6,889 | 0 | 2916.19 | 35.88 | -19.1 | 110.1 | 13 |
| TTV1 | F2_relief | 6,889 | 0 | 230.82 | -8.55 | -31.1 | 8.8 | 13 |
| TTV1 | PF_A | 6,889 | 0 | 117.56 | 0.33 | -19.7 | 17.0 | 13 |
| TTV1 | PF_C | 6,889 | 0 | 115.47 | 0.27 | -18.6 | 18.4 | 13 |
| TTV1 | SJ_C | 6,889 | 0 | 84.44 | -0.17 | -16.2 | 15.5 | 13 |
| TTV1 | P4 | 6,889 | 0 | 82.73 | -0.07 | -14.7 | 16.5 | 13 |
| TTV1 | AD | 6,889 | 0 | 214.19 | -0.97 | -26.4 | 23.3 | 13 |
| TTV2 | obs | 6,889 | 0 | - | - | - | - | 1 |
| TTV2 | F_phys | 6,889 | 0 | 6326.85 | 59.90 | 1.0 | 157.8 | 13 |
| TTV2 | F_form | 6,889 | 0 | 1400.18 | 27.94 | -1.4 | 74.2 | 12 |
| TTV2 | F_relief | 6,889 | 0 | 97.44 | -3.30 | -20.3 | 9.5 | 13 |
| TTV2 | F2_phys | 6,889 | 0 | 6197.13 | 58.28 | 0.4 | 162.6 | 13 |
| TTV2 | F2_form | 6,889 | 0 | 1413.08 | 26.87 | -2.5 | 78.8 | 13 |
| TTV2 | F2_relief | 6,889 | 0 | 123.96 | -3.84 | -23.4 | 11.0 | 12 |
| TTV2 | PF_A | 6,889 | 0 | 69.54 | 0.21 | -14.9 | 12.1 | 13 |
| TTV2 | PF_C | 6,889 | 0 | 101.32 | -0.01 | -18.4 | 15.7 | 13 |
| TTV2 | SJ_C | 6,889 | 0 | 87.31 | 0.04 | -17.4 | 13.0 | 13 |
| TTV2 | P4 | 6,889 | 0 | 59.79 | -0.02 | -14.0 | 11.0 | 12 |
| TTV2 | AD | 6,889 | 0 | 209.66 | 0.45 | -28.6 | 21.3 | 13 |
| TTV3 | obs | 6,889 | 0 | - | - | - | - | 1 |
| TTV3 | F_phys | 6,889 | 0 | 5014.47 | 49.93 | -6.0 | 148.2 | 13 |
| TTV3 | F_form | 6,889 | 0 | 1262.29 | 23.42 | -10.9 | 73.7 | 13 |
| TTV3 | F_relief | 6,889 | 0 | 176.08 | -4.32 | -27.8 | 13.4 | 13 |
| TTV3 | F2_phys | 6,889 | 0 | 8446.26 | 65.52 | -3.6 | 191.3 | 13 |
| TTV3 | F2_form | 6,889 | 0 | 2276.06 | 32.23 | -8.2 | 101.8 | 13 |
| TTV3 | F2_relief | 6,889 | 0 | 169.17 | -2.61 | -27.0 | 15.8 | 13 |
| TTV3 | PF_A | 6,889 | 0 | 152.49 | -0.44 | -23.0 | 17.9 | 13 |
| TTV3 | PF_C | 6,889 | 0 | 161.07 | -0.32 | -24.8 | 19.0 | 13 |
| TTV3 | SJ_C | 6,889 | 0 | 138.43 | -0.15 | -25.0 | 16.3 | 14 |
| TTV3 | P4 | 6,889 | 0 | 87.40 | 1.15 | -13.4 | 19.8 | 13 |
| TTV3 | AD | 6,889 | 0 | 326.85 | -0.91 | -35.1 | 24.1 | 14 |
| TTV4 | obs | 6,889 | 0 | - | - | - | - | 1 |
| TTV4 | F_phys | 6,889 | 0 | 4001.66 | 46.87 | -3.2 | 135.5 | 13 |
| TTV4 | F_form | 6,889 | 0 | 667.27 | 17.06 | -9.4 | 53.5 | 12 |
| TTV4 | F_relief | 6,889 | 0 | 164.55 | -2.63 | -26.1 | 16.9 | 12 |
| TTV4 | F2_phys | 6,889 | 0 | 8217.21 | 66.09 | -3.0 | 205.5 | 12 |
| TTV4 | F2_form | 6,889 | 0 | 1425.54 | 25.28 | -8.0 | 85.7 | 12 |
| TTV4 | F2_relief | 6,889 | 0 | 144.60 | -1.68 | -23.3 | 16.3 | 12 |
| TTV4 | PF_A | 6,889 | 0 | 153.00 | 0.14 | -22.6 | 18.8 | 12 |
| TTV4 | PF_C | 6,889 | 0 | 128.21 | -0.18 | -20.5 | 17.4 | 12 |
| TTV4 | SJ_C | 6,889 | 0 | 80.31 | -0.17 | -15.4 | 14.0 | 13 |
| TTV4 | P4 | 6,889 | 0 | 96.82 | -0.49 | -17.0 | 14.8 | 12 |
| TTV4 | AD | 6,889 | 0 | 293.97 | 0.21 | -29.9 | 28.4 | 12 |
| TTV5 | obs | 6,889 | 0 | - | - | - | - | 1 |
| TTV5 | F_phys | 6,889 | 0 | 4726.49 | 47.25 | -24.2 | 140.6 | 13 |
| TTV5 | F_form | 6,889 | 0 | 3317.19 | 37.96 | -29.6 | 117.3 | 13 |
| TTV5 | F_relief | 6,889 | 0 | 641.99 | -8.88 | -66.1 | 18.5 | 12 |
| TTV5 | F2_phys | 6,889 | 0 | 47455.98 | 151.21 | -19.3 | 470.2 | 12 |
| TTV5 | F2_form | 6,889 | 0 | 33929.66 | 126.45 | -22.7 | 399.1 | 12 |
| TTV5 | F2_relief | 6,889 | 0 | 707.75 | 1.56 | -54.8 | 45.4 | 12 |
| TTV5 | PF_A | 6,889 | 0 | 562.13 | -1.12 | -57.1 | 29.8 | 12 |
| TTV5 | PF_C | 6,889 | 0 | 422.51 | -0.17 | -45.2 | 29.9 | 12 |
| TTV5 | SJ_C | 6,889 | 0 | 380.29 | 0.92 | -42.0 | 29.9 | 13 |
| TTV5 | P4 | 6,889 | 0 | 137.88 | -0.22 | -17.8 | 16.5 | 12 |
| TTV5 | AD | 6,889 | 0 | 842.95 | -2.77 | -66.8 | 38.4 | 13 |
| TTV6 | obs | 6,889 | 0 | - | - | - | - | 1 |
| TTV6 | F_phys | 6,889 | 0 | 14888.57 | 84.47 | -17.4 | 280.7 | 12 |
| TTV6 | F_form | 6,889 | 0 | 2974.86 | 34.93 | -23.6 | 124.7 | 12 |
| TTV6 | F_relief | 6,889 | 0 | 242.89 | -8.98 | -33.1 | 9.5 | 12 |
| TTV6 | F2_phys | 6,889 | 0 | 24660.23 | 118.16 | -7.4 | 323.9 | 12 |
| TTV6 | F2_form | 6,889 | 0 | 5328.24 | 52.67 | -16.4 | 151.5 | 12 |
| TTV6 | F2_relief | 6,889 | 0 | 205.21 | -5.38 | -29.6 | 15.4 | 12 |
| TTV6 | PF_A | 6,889 | 0 | 128.78 | -0.20 | -19.3 | 19.1 | 12 |
| TTV6 | PF_C | 6,889 | 0 | 120.29 | 0.26 | -17.5 | 19.2 | 12 |
| TTV6 | SJ_C | 6,889 | 0 | 109.86 | 0.34 | -17.0 | 17.8 | 13 |
| TTV6 | P4 | 6,889 | 0 | 94.27 | -0.23 | -16.4 | 17.0 | 12 |
| TTV6 | AD | 6,889 | 0 | 204.34 | -0.58 | -20.9 | 28.3 | 12 |
| TTV7 | obs | 6,889 | 0 | - | - | - | - | 1 |
| TTV7 | F_phys | 6,889 | 0 | 6718.06 | 55.08 | -12.9 | 177.1 | 13 |
| TTV7 | F_form | 6,889 | 0 | 2170.33 | 27.64 | -19.4 | 100.8 | 13 |
| TTV7 | F_relief | 6,889 | 0 | 260.38 | -5.49 | -32.0 | 16.2 | 12 |
| TTV7 | F2_phys | 6,889 | 0 | 21349.41 | 100.74 | -11.2 | 308.2 | 13 |
| TTV7 | F2_form | 6,889 | 0 | 6935.86 | 53.99 | -17.5 | 175.9 | 12 |
| TTV7 | F2_relief | 6,889 | 0 | 280.94 | -2.48 | -30.8 | 22.4 | 12 |
| TTV7 | PF_A | 6,889 | 0 | 185.60 | -1.57 | -23.0 | 21.2 | 12 |
| TTV7 | PF_C | 6,889 | 0 | 124.59 | -1.32 | -19.3 | 17.8 | 12 |
| TTV7 | SJ_C | 6,889 | 0 | 89.28 | 0.09 | -16.0 | 15.7 | 13 |
| TTV7 | P4 | 6,889 | 0 | 79.24 | -0.44 | -16.6 | 13.8 | 12 |
| TTV7 | AD | 6,889 | 0 | 134.26 | -0.60 | -17.9 | 21.4 | 12 |
| TTV8 | obs | 6,889 | 0 | - | - | - | - | 1 |
| TTV8 | F_phys | 6,889 | 0 | 7239.65 | 61.06 | -4.8 | 181.8 | 13 |
| TTV8 | F_form | 6,889 | 0 | 2635.37 | 34.45 | -8.9 | 111.4 | 13 |
| TTV8 | F_relief | 6,889 | 0 | 260.44 | -9.22 | -33.5 | 9.5 | 13 |
| TTV8 | F2_phys | 6,889 | 0 | 36182.29 | 131.05 | -6.1 | 399.5 | 13 |
| TTV8 | F2_form | 6,889 | 0 | 14139.42 | 79.01 | -11.0 | 252.5 | 13 |
| TTV8 | F2_relief | 6,889 | 0 | 257.23 | -6.43 | -34.0 | 15.0 | 13 |
| TTV8 | PF_A | 6,889 | 0 | 176.95 | -1.58 | -24.3 | 18.9 | 13 |
| TTV8 | PF_C | 6,889 | 0 | 124.43 | -1.34 | -20.5 | 17.8 | 13 |
| TTV8 | SJ_C | 6,889 | 0 | 108.47 | -0.82 | -18.9 | 17.7 | 13 |
| TTV8 | P4 | 6,889 | 0 | 114.67 | -1.06 | -20.0 | 17.2 | 13 |
| TTV8 | AD | 6,889 | 0 | 272.96 | -0.65 | -22.8 | 32.0 | 13 |
| TTV9 | obs | 6,889 | 0 | - | - | - | - | 1 |
| TTV9 | F_phys | 6,889 | 0 | 4144.31 | 38.86 | -22.9 | 145.4 | 13 |
| TTV9 | F_form | 6,889 | 0 | 2561.70 | 27.79 | -25.2 | 115.4 | 13 |
| TTV9 | F_relief | 6,889 | 0 | 519.98 | -10.51 | -43.0 | 26.7 | 12 |
| TTV9 | F2_phys | 6,889 | 0 | 3892.16 | 37.18 | -23.6 | 139.6 | 13 |
| TTV9 | F2_form | 6,889 | 0 | 2438.63 | 27.01 | -25.7 | 110.3 | 13 |
| TTV9 | F2_relief | 6,889 | 0 | 487.72 | -8.19 | -41.9 | 27.1 | 13 |
| TTV9 | PF_A | 6,889 | 0 | 312.12 | 0.63 | -26.1 | 32.9 | 13 |
| TTV9 | PF_C | 6,889 | 0 | 355.81 | 0.72 | -26.9 | 37.0 | 12 |
| TTV9 | SJ_C | 6,889 | 0 | 116.57 | 0.62 | -18.1 | 18.5 | 12 |
| TTV9 | P4 | 6,889 | 0 | 115.18 | 0.80 | -18.3 | 18.5 | 12 |
| TTV9 | AD | 6,889 | 0 | 284.91 | 0.69 | -26.1 | 29.1 | 12 |
| TTV11 | obs | 6,889 | 0 | - | - | - | - | 1 |
| TTV11 | F_phys | 6,889 | 0 | 10275.29 | 64.14 | -15.7 | 228.6 | 13 |
| TTV11 | F_form | 6,889 | 0 | 3308.32 | 32.60 | -21.2 | 132.0 | 12 |
| TTV11 | F_relief | 6,889 | 0 | 387.03 | -11.91 | -43.0 | 9.9 | 12 |
| TTV11 | F2_phys | 6,889 | 0 | 18774.13 | 87.61 | -17.3 | 305.0 | 12 |
| TTV11 | F2_form | 6,889 | 0 | 6251.60 | 46.45 | -22.2 | 179.8 | 12 |
| TTV11 | F2_relief | 6,889 | 0 | 361.11 | -11.63 | -40.8 | 12.4 | 13 |
| TTV11 | PF_A | 6,889 | 0 | 261.10 | -2.43 | -34.2 | 22.4 | 12 |
| TTV11 | PF_C | 6,889 | 0 | 185.94 | -1.51 | -28.4 | 19.1 | 12 |
| TTV11 | SJ_C | 6,889 | 0 | 185.47 | -1.50 | -28.4 | 19.1 | 13 |
| TTV11 | P4 | 6,889 | 0 | 241.77 | -1.62 | -31.9 | 20.2 | 12 |
| TTV11 | AD | 6,889 | 0 | 164.20 | -0.12 | -23.1 | 20.8 | 12 |
| TTV15 | obs | 6,889 | 0 | - | - | - | - | 1 |
| TTV15 | F_phys | 6,889 | 0 | 10012.37 | 73.31 | -5.4 | 208.2 | 13 |
| TTV15 | F_form | 6,889 | 0 | 2192.76 | 31.43 | -11.3 | 100.8 | 13 |
| TTV15 | F_relief | 6,889 | 0 | 197.76 | -5.34 | -25.0 | 20.2 | 12 |
| TTV15 | F2_phys | 6,889 | 0 | 42252.65 | 138.38 | -5.7 | 469.7 | 13 |
| TTV15 | F2_form | 6,889 | 0 | 9698.77 | 62.51 | -11.4 | 228.6 | 13 |
| TTV15 | F2_relief | 6,889 | 0 | 203.78 | -4.08 | -24.9 | 21.5 | 13 |
| TTV15 | PF_A | 6,889 | 0 | 187.56 | -3.53 | -24.2 | 21.5 | 13 |
| TTV15 | PF_C | 6,889 | 0 | 132.98 | -2.07 | -17.9 | 21.2 | 12 |
| TTV15 | SJ_C | 6,889 | 0 | 109.00 | -2.12 | -17.5 | 17.8 | 13 |
| TTV15 | P4 | 6,889 | 0 | 103.42 | -2.33 | -20.4 | 12.0 | 13 |
| TTV15 | AD | 6,889 | 0 | 327.53 | -1.61 | -24.0 | 36.4 | 12 |

### F12. Rule tables (section 6; from rules/tables.md)

### Parameter values per tile (section 2)
| tile | U | K0 (m^0.34/yr) | K0/K_cal | C_crest (1/m) | D_phys | D_form | D_form/D_phys | D_relief | 68 % | network | L_H (m) | R (m) | D_relief/D_form | D_crest at E_ridge |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 6.5e-05 | 9.49e-06 | 0.40 | 0.0159 | 0.0025 | 0.0041 | 1.63 | 0.0175 | [0.0166, 0.0185] | mapped | 171 | 50.0 | 4.29 | 0.0017 |
| TTV2 | 6.5e-05 | 1.33e-05 | 0.56 | 0.0149 | 0.0025 | 0.0044 | 1.74 | 0.0159 | [0.0153, 0.0165] | mapped | 157 | 49.3 | 3.65 | 0.0018 |
| TTV3 | 6.5e-05 | 2.23e-05 | 0.93 | 0.0147 | 0.0025 | 0.0044 | 1.77 | 0.0230 | [0.0217, 0.0268] | mapped | 173 | 42.3 | 5.20 | 0.0018 |
| TTV4 | 6.5e-05 | 1.36e-05 | 0.57 | 0.0124 | 0.0025 | 0.0053 | 2.10 | 0.0194 | [0.0180, 0.0208] | A (flagged) | 153 | 39.8 | 3.69 | 0.0022 |
| TTV5 | 6.5e-05 | 1.84e-05 | 0.77 | 0.0223 | 0.0025 | 0.0029 | 1.17 | 0.0186 | [0.0179, 0.0197] | A (flagged) | 129 | 32.0 | 6.38 | 0.0012 |
| TTV6 | 6.5e-05 | 1.05e-05 | 0.44 | 0.0139 | 0.0025 | 0.0047 | 1.87 | 0.0207 | [0.0200, 0.0213] | A (flagged) | 160 | 39.5 | 4.42 | 0.0019 |
| TTV7 | 6.5e-05 | 9.11e-06 | 0.38 | 0.0163 | 0.0025 | 0.0040 | 1.60 | 0.0144 | [0.0138, 0.0150] | A (flagged) | 115 | 32.1 | 3.60 | 0.0017 |
| TTV8 | 6.5e-05 | 9.71e-06 | 0.40 | 0.0176 | 0.0025 | 0.0037 | 1.48 | 0.0174 | [0.0168, 0.0180] | A (flagged) | 131 | 31.5 | 4.72 | 0.0015 |
| TTV9 | 6.5e-05 | 8.89e-06 | 0.37 | 0.0212 | 0.0025 | 0.0031 | 1.23 | 0.0142 | [0.0125, 0.0157] | A (flagged) | 116 | 33.4 | 4.62 | 0.0013 |
| TTV11 | 6.5e-05 | 9.79e-06 | 0.41 | 0.0167 | 0.0025 | 0.0039 | 1.56 | 0.0182 | [0.0174, 0.0188] | A (flagged) | 147 | 37.2 | 4.68 | 0.0016 |
| TTV15 | 6.5e-05 | 1.16e-05 | 0.48 | 0.0141 | 0.0025 | 0.0046 | 1.85 | 0.0182 | [0.0173, 0.0191] | A (flagged) | 144 | 37.4 | 3.94 | 0.0019 |

### Map-scale heads: fits (section 3)
| fit | n | a (m2) | b | b 95 % | R2 | median A/(1978 S^-1.65) | share in 0.5-2 | H0b |
|---|---|---|---|---|---|---|---|---|
| table1 | 63 | 1978 | 1.647 | [1.26, 2.00] | 0.60 | 1.01 | 0.95 | yes |
| picks_12m_S1 | 64 | 1021 | 1.746 | [1.09, 2.39] | 0.28 | 0.62 | 0.61 | yes |
| picks_12m_S1_without_49 | 63 | 1004 | 1.773 | [1.12, 2.41] | 0.29 | 0.64 | 0.62 | yes |
| picks_12m_S2 | 64 | 790 | 2.012 | [1.35, 2.70] | 0.31 | 0.60 | 0.59 | yes |
| picks_12m_dinf_S1 | 64 | 907 | 1.782 | [1.07, 2.44] | 0.30 | 0.55 | 0.62 | yes |
| picks_10m_S1 | 64 | 365 | 2.493 | [1.76, 3.22] | 0.40 | 0.48 | 0.47 | no |

### Field values
| set | A_F (m2) | C_F alpha 2 (m2) | C_F alpha 1.65 (m2) | A_F w/o 49 | C_F w/o 49 | S nan/<=0 |
|---|---|---|---|---|---|---|
| 12m | 6120 | 876.3 | 1230.3 | 6192 | 893.9 | 0 |
| 12m_S2 | 6120 | 835.2 | 1190.5 | 6192 | 835.7 | 0 |
| 12m_dinf | 5663 | 794.9 | 1092.8 | 5758 | 809.1 | 0 |
| 10m | 5050 | 680.5 | 950.6 | 5100 | 689.3 | 0 |

### H0: the observed criterion against the picks (TTV1-TTV3 pooled)
| extraction | tiles | median dist (m) | q25-q75 | share <= 25 m | heads in polygon | picks | ratio | reading |
|---|---|---|---|---|---|---|---|---|
| full_lattice_a2 | 3 | 12.6 | 2.2-24.0 | 0.77 | 96 | 61 | 1.57 | too many heads |
| full_lattice_a2_without_49 | 3 | 12.4 | 2.0-23.3 | 0.78 | 96 | 60 | 1.60 | too many heads |
| full_lattice_a0 | 3 | 13.0 | 3.0-36.9 | 0.61 | 87 | 61 | 1.43 | locates the mapped heads |
| full_lattice_a0_without_49 | 3 | 13.0 | 2.8-36.1 | 0.62 | 87 | 60 | 1.45 | locates the mapped heads |
| full_lattice_a1.65 | 3 | 12.7 | 3.0-24.4 | 0.75 | 84 | 61 | 1.38 | locates the mapped heads |
| full_lattice_a1.65_without_49 | 3 | 12.7 | 2.8-24.1 | 0.77 | 84 | 60 | 1.40 | locates the mapped heads |
| tile_lattice_a2 | 3 | 21.8 | 5.0-44.8 | 0.56 | 86 | 61 | 1.41 | locates the mapped heads |
| tile_lattice_a0 | 3 | 20.6 | 5.0-50.0 | 0.52 | 78 | 61 | 1.28 | locates the mapped heads |
| tile_lattice_perron | 3 | 83.5 | 54.1-139.7 | 0.18 | 21 | 61 | 0.34 | too few heads, distances large |

### H1: model heads (TTV1-TTV3 pooled; alpha = 2 at C_F decides; alpha = 0 and Perron beside)
| model | a2 median dist (m) | a2 ratio | heads/picks | H1 reading | a0 dist | a0 ratio | Perron dist | Perron ratio |
|---|---|---|---|---|---|---|---|---|
| obs | 21.8 | 1.41 | 86/61 | locates the mapped heads | 20.6 | 1.28 | 83.5 | 0.34 |
| F_phys | 0.0 | 21.18 | 1292/61 | too many heads | 26.1 | 0.87 | 30.1 | 0.84 |
| F_form | 2.0 | 12.51 | 763/61 | too many heads | 27.0 | 0.87 | 44.7 | 0.64 |
| F_relief | 31.0 | 1.23 | 75/61 | count in range, distances large | 35.5 | 0.89 | 62.3 | 0.48 |
| F2_phys | 0.0 | 20.56 | 1254/61 | too many heads | 28.0 | 0.85 | 36.2 | 0.69 |
| F2_form | 1.0 | 12.85 | 784/61 | too many heads | 28.3 | 0.84 | 44.2 | 0.61 |
| F2_relief | 22.6 | 1.36 | 83/61 | locates the mapped heads | 33.0 | 0.74 | 61.6 | 0.49 |
| PF_A | 19.0 | 1.64 | 100/61 | too many heads | 32.7 | 0.90 | 62.1 | 0.66 |
| PF_C | 27.5 | 1.36 | 83/61 | count in range, distances large | 31.0 | 0.87 | 54.8 | 0.61 |
| SJ_C | 44.1 | 1.80 | 110/61 | too many heads, distances large | 41.0 | 0.93 | 61.6 | 0.62 |
| P4 | 46.9 | 2.41 | 147/61 | too many heads, distances large | 55.4 | 0.87 | 66.0 | 0.61 |
| AD | 41.2 | 2.28 | 139/61 | too many heads, distances large | 12.8 | 1.26 | 63.8 | 0.56 |

### K1 and M1 per tile
| tile | K/K0 PF_A | K/K0 PF_C | K/K0 SJ_C | K/K0 P4 | K/K0 AD | MSE PF_A | MSE PF_C | MSE SJ_C | MSE P4 | MSE AD | SJ-C/AD | PF-A/AD | PF-C/AD |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 0.86 | 0.85 | 0.90 | 1.10 | 9.23 | 117.6 | 115.5 | 84.4 | 82.7 | 214.2 | 0.39 | 0.55 | 0.54 |
| TTV2 | 1.02 | 0.99 | 0.98 | 1.04 | 4.82 | 69.5 | 101.3 | 87.3 | 59.8 | 209.7 | 0.42 | 0.33 | 0.48 |
| TTV3 | 0.97 | 0.89 | 0.92 | 1.42 | 1.96 | 152.5 | 161.1 | 138.4 | 87.4 | 326.8 | 0.42 | 0.47 | 0.49 |
| TTV4 | 0.99 | 0.92 | 0.95 | 0.96 | 10.09 | 153.0 | 128.2 | 80.3 | 96.8 | 294.0 | 0.27 | 0.52 | 0.44 |
| TTV5 | 0.88 | 0.67 | 0.65 | 1.22 | 2.78 | 562.1 | 422.5 | 380.3 | 137.9 | 842.9 | 0.45 | 0.67 | 0.50 |
| TTV6 | 0.85 | 0.82 | 0.80 | 0.76 | 7.60 | 128.8 | 120.3 | 109.9 | 94.3 | 204.3 | 0.54 | 0.63 | 0.59 |
| TTV7 | 0.82 | 0.76 | 0.71 | 0.75 | 20.90 | 185.6 | 124.6 | 89.3 | 79.2 | 134.3 | 0.67 | 1.38 | 0.93 |
| TTV8 | 0.87 | 0.76 | 0.82 | 0.73 | 13.20 | 177.0 | 124.4 | 108.5 | 114.7 | 273.0 | 0.40 | 0.65 | 0.46 |
| TTV9 | 0.81 | 0.85 | 0.73 | 0.73 | 11.35 | 312.1 | 355.8 | 116.6 | 115.2 | 284.9 | 0.41 | 1.10 | 1.25 |
| TTV11 | 0.86 | 0.77 | 0.77 | 0.69 | 12.53 | 261.1 | 185.9 | 185.5 | 241.8 | 164.2 | 1.13 | 1.59 | 1.13 |
| TTV15 | 1.04 | 0.77 | 0.81 | 1.11 | 15.39 | 187.6 | 133.0 | 109.0 | 103.4 | 327.5 | 0.33 | 0.57 | 0.41 |

### F0: the fixed step per tile (MSE / MSE_AD; MSE / MSE_SJ-C in brackets)
| tile | F_phys | F_form | F_relief | F2_phys | F2_form | F2_relief | F best | F worst | C_crest |
|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 37.98 (96.33) | 10.20 (25.88) | 1.15 (2.92) | 48.57 (123.19) | 13.61 (34.53) | 1.08 (2.73) | relief | phys | 0.0159 |
| TTV2 | 30.18 (72.46) | 6.68 (16.04) | 0.46 (1.12) | 29.56 (70.98) | 6.74 (16.18) | 0.59 (1.42) | relief | phys | 0.0149 |
| TTV3 | 15.34 (36.22) | 3.86 (9.12) | 0.54 (1.27) | 25.84 (61.02) | 6.96 (16.44) | 0.52 (1.22) | relief | phys | 0.0147 |
| TTV4 | 13.61 (49.83) | 2.27 (8.31) | 0.56 (2.05) | 27.95 (102.32) | 4.85 (17.75) | 0.49 (1.80) | relief | phys | 0.0124 |
| TTV5 | 5.61 (12.43) | 3.94 (8.72) | 0.76 (1.69) | 56.30 (124.79) | 40.25 (89.22) | 0.84 (1.86) | relief | phys | 0.0223 |
| TTV6 | 72.86 (135.53) | 14.56 (27.08) | 1.19 (2.21) | 120.68 (224.48) | 26.08 (48.50) | 1.00 (1.87) | relief | phys | 0.0139 |
| TTV7 | 50.04 (75.24) | 16.17 (24.31) | 1.94 (2.92) | 159.02 (239.12) | 51.66 (77.68) | 2.09 (3.15) | relief | phys | 0.0163 |
| TTV8 | 26.52 (66.75) | 9.65 (24.30) | 0.95 (2.40) | 132.55 (333.58) | 51.80 (130.36) | 0.94 (2.37) | relief | phys | 0.0176 |
| TTV9 | 14.55 (35.55) | 8.99 (21.98) | 1.83 (4.46) | 13.66 (33.39) | 8.56 (20.92) | 1.71 (4.18) | relief | phys | 0.0212 |
| TTV11 | 62.58 (55.40) | 20.15 (17.84) | 2.36 (2.09) | 114.34 (101.23) | 38.07 (33.71) | 2.20 (1.95) | relief | phys | 0.0167 |
| TTV15 | 30.57 (91.86) | 6.69 (20.12) | 0.60 (1.81) | 129.00 (387.63) | 29.61 (88.98) | 0.62 (1.87) | relief | phys | 0.0141 |

### DX: D_fit / D_relief and the decomposition
| tile | D_relief | network | PF_A D/D_relief | PF_C D/D_relief | SJ_C D/D_relief | P4 D/D_relief | PF_A D/D_phys | PF_C D/D_phys | SJ_C D/D_phys | P4 D/D_phys | (D_relief/D_form)(U/E_ridge) | AD D/D_phys | AD D/D_relief |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TTV1 | 0.0175 | mapped | 0.94 | 1.07 | 1.54 | 2.12 | 6.6 | 7.5 | 10.8 | 14.8 | 10.33 | 0.54 | 0.0772 |
| TTV2 | 0.0159 | mapped | 0.72 | 0.79 | 0.66 | 0.88 | 4.6 | 5.0 | 4.2 | 5.6 | 8.79 | 0.0799 | 0.0126 |
| TTV3 | 0.0230 | mapped | 0.68 | 1.08 | 1.35 | 1.83 | 6.3 | 10.0 | 12.4 | 16.9 | 12.51 | 0.00312 | 0.000339 |
| TTV4 | 0.0194 | A | 0.78 | 1.36 | 2.17 | 2.17 | 6.1 | 10.5 | 16.9 | 16.9 | 8.89 | 1.22 | 0.156 |
| TTV5 | 0.0186 | A | 0.70 | 2.27 | 2.27 | 2.27 | 5.2 | 16.9 | 16.9 | 16.9 | 15.37 | 0.0176 | 0.00236 |
| TTV6 | 0.0207 | A | 0.98 | 1.34 | 1.58 | 1.36 | 8.1 | 11.1 | 13.1 | 11.2 | 10.63 | 0.377 | 0.0456 |
| TTV7 | 0.0144 | A | 2.65 | 2.65 | 2.93 | 2.93 | 15.3 | 15.3 | 16.9 | 16.9 | 8.68 | 8 | 1.39 |
| TTV8 | 0.0174 | A | 1.00 | 1.76 | 1.66 | 2.21 | 7.0 | 12.3 | 11.6 | 15.4 | 11.35 | 16.9 | 2.42 |
| TTV9 | 0.0142 | A | 1.87 | 1.44 | 2.73 | 2.61 | 10.6 | 8.1 | 15.5 | 14.8 | 11.13 | 0.663 | 0.117 |
| TTV11 | 0.0182 | A | 0.96 | 1.57 | 1.57 | 1.91 | 7.0 | 11.4 | 11.4 | 13.9 | 11.25 | 10.9 | 1.49 |
| TTV15 | 0.0182 | A | 0.78 | 1.69 | 1.70 | 1.33 | 5.6 | 12.3 | 12.3 | 9.7 | 9.48 | 16.9 | 2.32 |

### Rule verdicts
| rule | count | need | holds / reading |
|---|---|---|---|
| H0 (alpha 2, full-site network) | - | - | too many heads |
| H0 (alpha 0) | - | - | locates the mapped heads |
| H0b (primary) | - | - | yes |
| H2 | - | - | no |
| K1 PF_A | 11/11 | 9 | yes |
| K1 PF_C | 11/11 | 9 | yes |
| K1 SJ_C | 11/11 | 9 | yes |
| K1 P4 | 11/11 | 9 | yes |
| K1 AD | 1/11 | 9 | no |
| M1 SJ_C | 10/11 | 9 | yes |
| M1 PF_A | 8/11 | 9 | no |
| M1 PF_C | 9/11 | 9 | yes |
| M1 F_phys | 0/11 | 9 | no |
| M1 F_form | 0/11 | 9 | no |
| M1 F_relief | 6/11 | 9 | no |
| M1 F2_phys | 0/11 | 9 | no |
| M1 F2_form | 0/11 | 9 | no |
| M1 F2_relief | 6/11 | 9 | no |
| M1 P4 | 10/11 | 9 | yes |
| DX PF_A | 9/11 | 9 | yes |
| DX PF_C | 6/11 | 9 | no |
| DX SJ_C | 1/11 | 9 | no |
| DX P4 | 3/11 | 9 | no |
| F0 F_relief_fits_best | 11/11 | 9 | yes |
| F0 F_phys_fits_worst | 11/11 | 9 | yes |
| F0 model_crest_sharper_than_measured_at_D_phys | 11/11 | 9 | - |
| F0 F2_relief_fits_best | 11/11 | 9 | - |
| F0 F2_phys_fits_worst | 11/11 | 9 | - |


### F13. Figures

- `figures/a21/model_heads/TTV1_model_heads.png` (0.73 MB): TTV1_model_heads
- `figures/a21/observed/TTV1_observed_criterion.png` (0.12 MB): TTV1_observed_criterion
- `figures/a21/model_heads/TTV2_model_heads.png` (0.75 MB): TTV2_model_heads
- `figures/a21/observed/TTV2_observed_criterion.png` (0.13 MB): TTV2_observed_criterion
- `figures/a21/model_heads/TTV3_model_heads.png` (0.64 MB): TTV3_model_heads
- `figures/a21/observed/TTV3_observed_criterion.png` (0.10 MB): TTV3_observed_criterion
- `figures/a21/residuals/TTV1_residuals.png` (0.61 MB): TTV1_residuals
- `figures/a21/residuals/TTV2_residuals.png` (0.62 MB): TTV2_residuals
- `figures/a21/residuals/TTV3_residuals.png` (0.70 MB): TTV3_residuals
- `figures/a21/residuals/TTV4_residuals.png` (0.52 MB): TTV4_residuals
- `figures/a21/residuals/TTV5_residuals.png` (0.52 MB): TTV5_residuals
- `figures/a21/residuals/TTV6_residuals.png` (0.56 MB): TTV6_residuals
- `figures/a21/residuals/TTV7_residuals.png` (0.64 MB): TTV7_residuals
- `figures/a21/residuals/TTV8_residuals.png` (0.65 MB): TTV8_residuals
- `figures/a21/residuals/TTV9_residuals.png` (0.68 MB): TTV9_residuals
- `figures/a21/residuals/TTV11_residuals.png` (0.58 MB): TTV11_residuals
- `figures/a21/residuals/TTV15_residuals.png` (0.59 MB): TTV15_residuals
- `figures/a21/sjc_profiles.png` (0.37 MB): sjc_profiles
- `figures/a21/D_decomposition.png` (0.06 MB): decomposition
- `figures/a21/heads_area_slope_12m.png` (0.13 MB): map-scale area against slope at the picks

### F14. Slurm accounting

| job name | tasks | core-h used | core-h of limits | states | job ids |
|---|---|---|---|---|---|
| a21_grid12 | 1 | 0.07 | 2.0 | COMPLETED 1 | 53447083 |
| a21_perron_diag | 2 | 0.01 | 1.0 | COMPLETED 2 | 53449145 53449842 |
| a21_phase | 1 | 0.08 | 0.7 | COMPLETED 1 | 53447533 |
| a21_post | 1 | 0.02 | 0.7 | FAILED 1 | 53451472 |
| a21_sjc | 11 | 8.86 | 34.1 | COMPLETED 11 | 53448194 |
| a21_sjc_pilot | 1 | 0.05 | 0.5 | COMPLETED 1 | 53447085 |
| a21_sjc_redo | 11 | 0.11 | 5.5 | COMPLETED 11 | 53451471 |
| a21_tests | 1 | 0.15 | 2.0 | COMPLETED 1 | 53447081 |
| a21_tile_test | 2 | 0.09 | 1.3 | FAILED 1, COMPLETED 1 | 53447084 53447954 |
| a21_tiles | 21 | 1.34 | 10.5 | COMPLETED 21 | 53449079 53450470 |

Core-hours used: **10.8**; CPU-weighted limits: 58.3; peak MaxRSS 7.3 GB; 13 submissions.
