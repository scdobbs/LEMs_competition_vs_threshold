# Phase 4: optimizer v2 (directive Phase 4; addendum 3, section 5)

Directive: `refactor_v2_directive.md` Phase 4 (4.1-4.5) as amended by `refactor_v2_addendum_3.md` section 5.
Branch `v2`, code commits 6a8bdb4 (optimizer v2), 0d5889d, 2436328 (staircase refinement), 09e1250 / later (tables,
figures). Environment `lemcvt` (numpy 2.5.4, scipy 1.18.1), all jobs on `serc`, 1 CPU and 16 GB each. AP tiles with
`reproduce_v1_units: true` (feet) as the addendum specifies. Machine-generated tables: `04_tables.md` (V1-V6,
`reports/v2/scripts/make_04_tables.py`; the run ids and job ids are listed at its top). Figures
(`lemcvt figures optimizer_v2`): `reports/v2/figures_v2/04_acrit_profiles.png`, `04_ad_profiles.png`,
`04_m_contours.png`. Date: 2026-11-09 (night). **Status: a stop for review.** The AD, partitioned (stage 2 and
joint) and exponent-profile stages are complete on all ten tiles; the nonlinear stage is complete on the GM tiles
listed in section 6 and running on the AP tiles (job 51707588, each evaluation there is a 50-iteration Picard loop,
see `02e`), to be appended when they finish.

## Summary

| Item | Result |
| --- | --- |
| Reductions proved (5.3) | `tests/test_optimizer_v2.py`: partitioned a + (U/K) b + (U/D) c against the direct solve to 1e-10 relative at random (K, D) and at m = 0.35, 0.45, 0.60; AD h(r) + (U/D) g(r) likewise; brute force over (A_crit, K, D) on a 60 x 60 crop finds no lower MSE; box-constrained (U/K, U/D) minimum against `scipy.optimize.lsq_linear`. On the full tiles the reduction agrees with the direct solve to 1e-11 relative in MSE and 1e-11 of relief (V5) |
| Budget rule (5.1) | Every unique drainage area in [1e2, 1e5] m^2 is 36000-44000 values per AP tile and 6100-8100 per GM tile: 621 core-hours for the full staircase. Above 300, so the grid was thinned to the 2000 values nearest a uniform log grid (219 core-hours estimated), then every unique area around the minima was evaluated (67-288 per tile, `lemcvt tile v2-refine`) so the optima are exact staircase steps; actual cost 68 core-hours for these stages |
| AD | v2's global profile optimum equals v1's L-BFGS-B optimum to 4 significant figures on 9 of 10 tiles. **GM1: v2 finds MSE 47.0 at K = 1.19e-4, D = 7.6e-4 against v1's 168.6 at K = 4.46e-4, D = 2.88e-2**: v1 stopped on the descending wall of a single-minimum profile. AP5's v1 point is the global optimum (the Phase 3 L-BFGS-B basin of `03b`, MSE 2826, is a shoulder of the profile) |
| Stage 2 (A_crit at K0, D0) | Same staircase step as v1 on AP1, AP4, GM1, GM3, GM4; a lower step by 1e-4 to 2e-3 on AP2, AP3, AP5, GM5. **GM2: the v1 value 100 m^2 (the search bound) is not the minimum; it is at 211 m^2 with MSE 48.2 against 57.9 (-17 %)**. GM3's coincidence of stage-2 and nonlinear A_crit (1297.4) is real: both profiles have their minimum on the same step (1298 m^2) |
| Joint | v2 lower than v1 on all ten tiles: by 0.01-0.3 % on AP1, AP4, AP5, by 1.1-1.6 % on AP2, AP3, and by **6-15 % on GM1-GM4** (GM1: A_crit 1161 against 8092 m^2, MSE 28.9 against 33.9; GM3: 1393 against 6526; GM4: 3853 against 7396), 1.2 % on GM5. v1's (K, D) were the best for v1's A_crit on every tile except GM5; the misses are in A_crit (Nelder-Mead on the staircase), not in K, D |
| Identifiability | A_crit is constrained to a factor 1.05-2.0 at the 1 % level (one to four disjoint intervals) and 1.1-2.9 at 5 %; K varies by <= 4 % and D by <= 21 % along the 1 % intervals. r = K/D in the AD model is constrained to 0.1-0.2 decade at 1 % |
| Exponent profile (5.2) | Fitted m differs from the config (measured v1) m on every tile: AP1 0.54 (0.562), AP2 0.59 (0.592), AP3 0.39 (0.542), AP4 0.21 (0.438, at the grid edge), AP5 0.42 (0.400), GM1 0.29 (0.379), GM2 0.29 (0.308), GM3 0.29 (0.417), GM4 0.36 (0.313), GM5 0.34 (0.300); the fitted point lies inside the measured theta(A_c) 95 % band (pooled estimator) on AP1, AP3, AP5, GM2, GM3, GM4, outside on AP4 and GM1; no band exists at the fitted A_crit on AP2 and GM5 |
| Nonlinear | GM3: v1's step reproduced (1298 m^2, MSE 58.98, 6 Picard iterations); others pending (section 6) |
| Recommendation (directive) | v2 for the paper's numbers: it is never worse, its optima are exact over the sampled staircase, it gives the profiles, and it is 2-10 x cheaper. The decision is Steve's; what changes if v2 is adopted is listed in section 8 |

## 1. What was implemented (`src/lemcvt/optimize/v2.py`, `src/lemcvt/analyses/optimizer_v2.py`)

**Linear partitioned model, fixed A_crit and m.** Channel cells carry z_anchor + (U/K) chi (Phase 3 step 4);
hillslope cells solve L z_h = -(U/D) 1 - B z_fixed with L the hillslope Laplacian at D = 1 and B the coupling to the
fixed channel and boundary cells. Hence Z(K, D) = a + (U/K) b + (U/D) c with a = -L^-1 B z_fixed0, b = -L^-1 B chi,
c = -L^-1 1 on the hillslopes and (z_fixed0, chi, 0) on the fixed cells: one sparse LU (SuperLU `splu`) and three
solves per A_crit. L, B, a and c do not depend on m, so the exponent profile costs one solve per m. The MSE is a
convex quadratic in x = U/K, y = U/D; its minimum under the box K, D in bounds is found exactly (the unconstrained
minimiser if feasible, else the best of the four edges, each a clipped one-dimensional problem). The gradient
dMSE/d(log10 K, log10 D) at any point follows analytically from the same Gram matrix (the certificate).

**AD model, fixed r = K/D.** (M_diff0 + r M_adv0 + I_bc) z = (U/D) b_U + bc, so Z = h(r) + (U/D) g(r) from one LU and
two solves; the optimal U/D is closed-form (clipped to the bounds, which are intersected with the K bounds at that
r). The profile runs over 200 log-spaced r between K_lo/D_hi and K_hi/D_lo and is refined with bounded Brent
(8-17 extra evaluations). The v1 point is evaluated by a direct solve and its gradient by central differences as in
`02d` 2.1b.

**Nonlinear model.** No reduction (directive 4.3): the Picard solver at (K0, D0, S_c) on 200 log-thinned unique
areas, then every unique area inside the bracket around the minimum (<= 60), with the `02e` Picard record kept for
every evaluation.

**Exponent profile (5.2).** m = 0.20, 0.21, ..., 0.70 (51 values) x 200 log-thinned unique areas, the exact (U/K, U/D)
at every point: MSE(A_crit, m), the joint optimum, the profile MSE(m) minimised over A_crit, and the 1 % and 5 %
contours as point sets.

**Grids.** `lemcvt tile v2-estimate` printed the budget first (section 2). `lemcvt tile run --optimizer v2` runs any
of the stages `ad`, `partitioned` (= acrit + joint), `nonlinear`, `m_profile`; every evaluation is in
`trajectory.parquet` (stage and kind columns) and the profiles in `acrit_profile.parquet`, `nonlinear_profile.parquet`,
`m_profile.parquet`; the grids at the v2 optima are written as the v1 pipeline writes them. `lemcvt tile v2-refine`
adds the staircase refinement to a thinned run and writes a merged run.

## 2. Budget rule and cost (5.1)

`lemcvt tile v2-estimate --tiles all` (job 51706603, 2026-11-09 14:52 PST), before submission:

| tile | unique areas in [1e2, 1e5] m^2 | cells in range | full staircase (h, at 7 s per A_crit) | thinned to 2000 (h) |
| --- | --- | --- | --- | --- |
| AP1-AP5 | 36295, 43784, 38560, 38816, 37158 | 107000-118000 | 70.6, 85.1, 75.0, 75.5, 72.3 | 3.9 each |
| GM1-GM5 | 6138, 6701, 8131, 6969, 7826 | 98000-115000 | 10.2, 11.2, 13.6, 11.6, 13.0 | 3.3 each |

Totals with the AD (1.5-1.6 h per tile), nonlinear (3.2 h GM, 23.8 h AP) and exponent (3.2 h) stages: **621
core-hours for the full staircase, 219 for the thinned grid**; the rule's threshold is 300, so the thinned grid
was used and the choice is recorded in every run (`grid.mode`). The thinned grid samples one unique area in 18-22
on AP and one in 3-4 on GM; its optimum can sit one step away from the true one (AP5 joint: v1's step 1886.99 m^2
lies between two grid values and is 0.08 % lower), so `v2-refine` then evaluated every unique area between the
grid neighbours of the stage-2 and joint minima and of the v1 stage-2 and joint values (V6: 7-288 extra
evaluations per tile; the joint MSE fell by 3e-5 to 1.3e-3 relative, the optima moved by at most 1 % in A_crit).
Measured wall times: partitioned 2.2-4.2 h per tile (2.8 s per A_crit on AP, 5.4 s on GM, the GM Laplacians
being larger), AD 1.6-3.7 h (29-60 s per r, node-dependent), exponent profile 0.8 h, refinement 0.1-0.3 h; 68 core-hours of
wall time for the stages reported here (partitioned 32.5 h, AD 25.6 h, exponent profile 8.7 h, refinement 1.6 h).

## 3. Tests that prove the reductions (directive 4.4, addendum 5.3)

`tests/test_optimizer_v2.py` (9 tests, run in job 51706603 with the rest of the suite, 164 passed):

1. **Partitioned reduction** (`TestPartitionedReduction`): on the 60 x 60 synthetic tile, at A_crit = 150 and 400 m^2
   and three random (K, D) in the bounds, a + (U/K) b + (U/D) c equals the direct `partitioned_threshold_model`
   solve to < 1e-10 relative (max |dZ| / max |Z| and / relief), with the same channel mask, for m = 0.35, 0.45 and
   0.60 (the three values the addendum asks for); the quadratic MSE terms reproduce the direct MSE to 1e-9.
2. **AD reduction** (`TestADReduction`): h(r) + (U/D) g(r) equals the direct AD solve to < 1e-10 relative at four
   random (K, D); the r-profile's optimum is not above the best of a 12 x 12 (K, D) grid of direct solves.
3. **Brute force** (`TestBruteForce`): on the 60 x 60 crop, 40 unique A_crit values x 14 K x 14 D direct solves (7840
   forward solves) find no MSE below v2's joint optimum (tolerance 1e-9 relative); v2's stage-2 minimum equals the
   brute-force minimum at (K0, D0); the joint optimum's (K, D) reproduce its MSE by a direct solve.
4. **Exponent profile** (`TestExponentProfile`): one LU per A_crit with one solve per m reproduces the direct solve at
   each of three m to 1e-10 relative, and the m-profile rows reproduce their MSE by direct solves.
5. **Box-constrained minimum** (`TestBoxQuadratic`): 40 random bounded two-variable least-squares problems against
   `scipy.optimize.lsq_linear`, never worse.

On the full tiles the same checks are repeated by the runs themselves (V5): the reduction's MSE at the v2 optimum
against a direct `partitioned_threshold_model` solve differs by 9e-14 to 1.2e-11 relative and the surfaces by
<= 1.3e-11 of relief; the AD reduction's MSE against a direct AD solve by <= 1e-12.

## 4. v1 against v2 on the ten tiles (V1; figures `04_acrit_profiles.png`, `04_ad_profiles.png`)

"v1 same code" is the v1 printed point evaluated by the v2 code; it reproduces the v1 printed MSE to the printed
digits on every row, which is the Phase 2 regression once more.

### 4.1 AD stage

| tile | v1: K, D, MSE (same code) | v2: K, D, MSE | v2 - v1 (rel.) | gradient at v1 point (dMSE/dlog10 K, dlog10 D) |
| --- | --- | --- | --- | --- |
| AP1 | 8.278e-6, 5.446e-4, 372.957 | 8.278e-6, 5.446e-4, 372.957 | -8e-10 | -0.32, 0.05 |
| AP2 | 6.959e-6, 6.661e-4, 381.487 | 6.959e-6, 6.661e-4, 381.487 | -1e-11 | 0.04, -0.01 |
| AP3 | 1.115e-5, 6.644e-4, 1237.95 | 1.115e-5, 6.645e-4, 1237.95 | -3e-9 | 1.30, -0.39 |
| AP4 | 1.636e-5, 2.121e-4, 578.249 | 1.636e-5, 2.121e-4, 578.249 | -2e-10 | -0.32, 0.08 |
| AP5 | 1.561e-5, 4.316e-5, 1005.34 | 1.561e-5, 4.316e-5, 1005.34 | -2e-8 | -3.74, 0.89 |
| **GM1** | 4.461e-4, 2.884e-2, 168.647 | **1.192e-4, 7.625e-4, 46.999** | **-0.72** | 0.42, -0.16 |
| GM2 | 1.311e-4, 8.298e-5, 81.9555 | 1.312e-4, 8.298e-5, 81.9555 | -2e-8 | -0.36, 0.07 |
| GM3 | 1.761e-4, 2.344e-3, 76.1875 | 1.761e-4, 2.344e-3, 76.1875 | -1e-8 | 0.25, -0.07 |
| GM4 | 9.461e-5, 6.784e-5, 175.388 | 9.461e-5, 6.784e-5, 175.388 | -1e-10 | -0.05, 0.01 |
| GM5 | 1.566e-4, 1.860e-4, 93.1766 | 1.566e-4, 1.860e-4, 93.1766 | -5e-8 | -0.81, 0.22 |

On nine tiles v1's L-BFGS-B found the profile's global optimum (the differences are the printed precision of v1's
values). On GM1 it did not: the profile MSE*(r) over r = K/D (`04_ad_profiles.png`) has one minimum, at
r = 0.156 (K 1.19e-4, D 7.6e-4), and v1's point (r = 0.0155) sits on the descending wall 3.6 x higher in MSE,
where L-BFGS-B stopped on scipy's relative-reduction criterion with a gradient of 0.4 per decade of K (the same
stopping mode `02d` 2.1b recorded for every AD stage; on GM1 the surface is flat enough along the wall for the
criterion to fire early). The v2 optimum is confirmed by a direct solve (46.999) and its gradient there is
(2e-4, 1e-3). The AP5 profile shows a shoulder at MSE 2000-3000 for r between 1e-2 and 1e4; the basin the Phase 3
L-BFGS-B fell into (`03b`, MSE 2826, r = 0.0097) is the left end of that shoulder, and v1's point (r = 0.36) is the
global minimum. The 1 % intervals in r are 0.1-0.2 decade wide (1-6 sampled points; V2).

### 4.2 Stage 2: A_crit at K0, D0 (the staircase, every step sampled near the minimum)

| tile | v1 A_crit, MSE | v2 A_crit, MSE | v2 - v1 (rel.) | 1 % interval (m^2) |
| --- | --- | --- | --- | --- |
| AP1 | 3063.4, 1916.22 | 3064.97, 1916.22 | 0 (same step) | 3018-3157 |
| AP2 | 2337.8, 4227.38 | 2345.31, 4226.92 | -1.1e-4 | 2196-2504 (5 intervals) |
| AP3 | 1722.1, 1356.04 | 1764.45, 1353.62 | -1.8e-3 | 1642-1807 |
| AP4 | 995.01, 597.968 | 995.353, 597.932 | -6e-5 | 907-1045 |
| AP5 | 1130.0, 359.261 | 1125.6, 359.079 | -5.0e-4 | 1109-1235 |
| GM1 | 423.49, 100.962 | 424, 100.962 | 0 (same step) | 351-442 |
| **GM2** | **100.00 (the bound), 57.8726** | **211, 48.1786** | **-0.168** | 200-219 |
| GM3 | 1297.4, 63.3511 | 1298, 63.3511 | 0 (same step) | 1075-1350 |
| GM4 | 422.09, 85.5264 | 423, 85.5264 | 0 (same step) | 394-454 |
| GM5 | 261.42, 61.0221 | 200, 60.99 | -5.3e-4 | 178-270 |

The two symptoms the directive named are settled. **GM2**: v1's Nelder-Mead ended at the lower search bound
(100 m^2) with MSE 57.87; the profile's minimum is inside the bounds at 211 m^2 with MSE 48.18, 17 % lower, and the
profile rises steeply below 200 m^2 (`04_acrit_profiles.png`); the v1 value is an optimizer artefact. **GM3**: the
stage-2 minimum (1298 m^2) and the nonlinear minimum (1298 m^2, section 6) are on the same staircase step, so the
coincidence of the two v1 values is a property of the misfit surfaces at (K0, D0), not of the optimizer. Where
v1 and v2 differ by 1e-4 to 2e-3 (AP2, AP3, AP5, GM5) v1 stopped on a neighbouring step of a flat minimum; the
1 % intervals are 5-15 % wide in A_crit on AP and 10-50 % on GM, in one to five disjoint pieces (the staircase is
not monotone near its minimum).

### 4.3 Joint: exact (K, D) at every A_crit

| tile | v1: A_crit, K, D, MSE (same code) | v2: A_crit, K, D, MSE | v2 - v1 (rel.) | gradient at v1 point (K, D) | best (K, D) for v1's A_crit: MSE |
| --- | --- | --- | --- | --- | --- |
| AP1 | 6414.3, 4.068e-7, 5.380e-3, 226.023 | 4931.1, 4.147e-7, 5.125e-3, 225.444 | -2.6e-3 | -0.32, -0.09 | 226.023 (v1's K, D) |
| AP2 | 7645.6, 3.507e-7, 7.086e-3, 233.501 | 7003.6, 3.528e-7, 6.824e-3, 229.743 | -1.6e-2 | 0.31, -0.03 | 233.501 |
| AP3 | 9741.5, 6.315e-7, 6.905e-3, 505.234 | 10299.6, 6.332e-7, 6.957e-3, 499.458 | -1.1e-2 | -1.00, 2.72 | 505.234 |
| AP4 | 7797.9, 2.037e-6, 6.101e-3, 370.050 | 7231.8, 1.982e-6, 6.018e-3, 369.389 | -1.8e-3 | -0.52, -0.08 | 370.050 |
| AP5 | 1887.0, 3.440e-6, 3.629e-3, 289.625 | 1887.08, 3.440e-6, 3.630e-3, 289.599 | -9e-5 | 0.71, 0.81 | 289.625 |
| **GM1** | 8092.4, 2.165e-5, 2.837e-2, 33.8645 | **1161, 2.243e-5, 1.970e-2, 28.896** | **-0.147** | -0.04, 0.02 | 33.8645 |
| **GM2** | 2960.0, 5.598e-5, 2.641e-2, 27.827 | **3509, 5.643e-5, 2.741e-2, 26.057** | **-6.4e-2** | 0.16, 0.15 | 27.827 |
| **GM3** | 6526.3, 1.913e-5, 4.533e-2, 43.0181 | **1393, 2.071e-5, 2.608e-2, 38.319** | **-0.109** | 0.04, 0.03 | 43.0181 |
| **GM4** | 7396.2, 3.511e-5, 2.655e-2, 47.7298 | **3853, 3.504e-5, 2.196e-2, 40.585** | **-0.150** | 0.17, -0.00 | 47.7298 |
| GM5 | 2247.1, 5.823e-5, 2.028e-2, 33.0543 | 3108, 5.953e-5, 2.139e-2, 32.662 | -1.2e-2 | **-76.3**, -1.73 | 32.692 (K 5.959e-5, D 1.953e-2) |

Reading. On every tile the (K, D) v1 reported are the exact best (K, D) for v1's own A_crit (last column equals the
v1 MSE; GM5 is the exception, where the exact (K, D) at 2247 m^2 lower the MSE by 1.1 %, the -76 gradient of
`02d`). What v1 missed is the A_crit step: Nelder-Mead on a staircase objective stopped on a step 1.1-1.6 % above
the minimum on AP2 and AP3 and 6-15 % above it on GM1-GM4, where the v1 values 6500-8100 m^2 (GM1, GM3, GM4) lie
outside even the 5 % intervals (GM1: 1037-1840; GM3: 1083-3061; GM4: 2577-4442 m^2). On GM the joint profile is
flat over a decade (`04_acrit_profiles.png`, red curves), which is where a simplex can stall; on AP it is sharper
and v1 was close. The directive's "how well is A_crit constrained" is answered in V2: at the 1 % level A_crit
lies within a factor 1.05 (AP3, AP5, GM2) to 2.0 (GM5, where the interval is in four disjoint pieces) of the
optimum, at 5 % within a factor 1.1 to 2.9; K changes by <= 4 % across the 1 % intervals, D by 2-21 %, so K is the
well-determined parameter and the (A_crit, D) pair trades off.

## 5. Exponent profile (addendum 3, 5.2; V3, V3b; figure `04_m_contours.png`)

| tile | config m | fitted (A_crit, m) | K, D at the fit | MSE at fit / at config m | 1 % contour: A_crit, m | 5 % contour: m | measured theta at the fitted A_crit, pooled [95 %] / equal-weight [95 %] | inside 95 % band (pooled / equal) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 0.562 | 4894, 0.54 | 5.30e-7, 5.12e-3 | 221.1 / 225.0 | 4894-6686, 0.53-0.55 | 0.49-0.57 | 0.54 [0.10, 0.55] / 0.11 [0.10, 0.53] | yes / no |
| AP2 | 0.592 | 6919, 0.59 | 3.60e-7, 6.78e-3 | 231.1 / 231.1 | 6919-7162, 0.56-0.62 | 0.47-0.67 | no interval at that A_c (2 basins) | - |
| AP3 | 0.542 | 5158, 0.39 | 3.96e-6, 5.53e-3 | 428.8 / 498.9 | 4814-5525, 0.36-0.41 | 0.31-0.45 | 0.43 [0.10, 0.69] / 0.43 [0.10, 0.84] | yes / yes |
| AP4 | 0.438 | 7831, **0.21 (grid edge 0.20)** | 2.98e-5, 6.04e-3 | 341.2 / 370.7 | 7831-8684, 0.20-0.28 | 0.20-0.38 | 0.59 [0.57, 0.89] / 0.68 [0.59, 0.89] | no / no |
| AP5 | 0.400 | 1855, 0.42 | 2.73e-6, 3.67e-3 | 287.3 / 290.7 | 1855-1919, 0.41-0.44 | 0.38-0.47 | 0.42 [0.19, 0.55] / 0.41 [0.23, 0.63] | yes / yes |
| GM1 | 0.379 | 1135, 0.29 | 6.11e-5, 1.91e-2 | 24.5 / 29.0 | 1135-1782, 0.27-0.30 | 0.23-0.33 | 0.31 [0.29, 0.40] / 0.29 [0.16, 0.39] | no / yes |
| GM2 | 0.308 | 3444, 0.29 | 6.94e-5, 2.70e-2 | 26.2 / 26.3 | 3444, 0.26-0.33 | 0.22-0.37 | 0.31 [0.27, 0.43] / 0.29 [0.22, 0.38] | yes / yes |
| GM3 | 0.417 | 859, 0.29 | 8.45e-5, 1.90e-2 | 32.8 / 38.6 | 859-920, 0.27-0.31 | 0.24-0.34 | 0.33 [0.27, 0.38] / 0.26 [0.10, 0.39] | yes / yes |
| GM4 | 0.313 | 2718, 0.36 | 1.96e-5, 2.14e-2 | 39.5 / 40.8 | 2624-3832, 0.34-0.38 | 0.30-0.40 | 0.35 [0.33, 0.57] / 0.36 [0.31, 0.50] | yes / yes |
| GM5 | 0.300 | 2247, 0.34 | 3.64e-5, 2.04e-2 | 31.6 / 32.7 | 1894-2325, 0.32-0.35 | 0.30-0.38 | no interval at that A_c (2 basins) | - |

(The measured curves are the `02e` free-ladder thetas, pooled and equal-weight estimators, with their 95 % basin
bootstrap bands interpolated in log A_c at the fitted A_crit; where the band is unavailable because fewer than
three basins pass the filters at those steps the cell says so. The profile MSE(m) relative to its minimum is in
V3b: it is within 5 % of the minimum over a range of 0.08-0.2 in m on every tile, and rises by a factor 1.4-4 at
the far end of the grid.)

As the addendum directs, no interpretation beyond the band test: the fitted (A_crit, m) lies inside the pooled
95 % band on AP1, AP3, AP5, GM2, GM3, GM4, outside on AP4 (fitted 0.21 against measured 0.59-0.68 at 7800 m^2)
and GM1 (0.29 against a band starting at 0.29-0.30 by the pooled estimator; inside the equal-weight band), and the
test is not possible on AP2 and GM5. The AP4 fit sits at the lower edge of the m grid (0.20), so its optimum is
censored. Interpretation is Steve's.

## 6. Nonlinear stage (V4)

| tile | v1 A_crit, MSE (same code; Picard it.) | v2 A_crit, MSE (Picard it.) | v2 - v1 | evaluations hitting the cap | 1 % interval |
| --- | --- | --- | --- | --- | --- |
| GM3 | 1297.4, 58.9848 (6, converged) | 1298, 58.9848 (6, converged) | 0 (same step) | 0 of 261 | 1257-1395 |
| GM1, GM2, GM4, GM5 | pending (job 51707582, running) | | | | |
| AP1-AP5 | pending (job 51707588, running; every evaluation above about 3000 m^2 is a 50-iteration loop, `02e`) | | | | |

The GM3 row is the one the directive's "GM3 coincidence" question needed (section 4.2). The rows are appended as
the runs finish; the AP rows are read with the `02e` caveat that the feet runs are unconverged, so their profile is
a profile of 50-iteration results, as v1's was.

## 7. What the evidence supports, and where it is weak

- Measured: every number in sections 4-6 is a profile value or a direct solve in the run directories, and the v1
  points were re-evaluated by the same code (the "same code" columns equal the v1 printed values). The reductions are
  proved by the tests and confirmed on the full tiles to 1e-11.
- Inferred: that v1's misses are optimizer effects (L-BFGS-B's stopping rule on GM1's flat wall; Nelder-Mead on the
  staircase for A_crit) rather than surface features: the profiles show single minima (AD) or a flat decade (joint on
  GM) at the places v1 stopped, which is consistent with the mechanisms, and `02d`'s gradients at the v1 points
  agree with the analytic ones here.
- Weak: (i) "global" means global over the sampled staircase (2000 thinned values plus every value around the minima
  and around v1's points); a lower step elsewhere in the 36000-44000 unique areas of an AP tile is possible but would
  have to be an isolated step between two sampled neighbours 2-5 % apart in A_crit, which the profiles' smoothness
  makes unlikely and does not exclude; (ii) the joint profile's K, D box is the config's v1 bounds, and the AP joint
  optima are interior, but the directive's Phase 5 rescales the AP bounds; (iii) the exponent profile's A_crit grid is
  200 values, coarser than the main profile, and its contours are point sets on that grid; (iv) the nonlinear
  stage is incomplete; (v) AP is in feet throughout, so every AP number here is a feet number to be redone in
  Phase 5-6, which is why no AP value is interpreted.

## 8. Decision for Steve

The directive asks which optimizer's numbers the paper reports. The evidence: v2 is never worse than v1 on any
stage of any tile (V1, 30 of 30 rows at or below v1, 9 of them by more than 1 %), its optima are exact staircase
steps with intervals, and it costs 2-10 x less. If v2 is adopted, the numbers that change by more than 1 % against
v1 are the GM1 AD optimum (MSE 169 -> 47; K, D by factors 0.27 and 0.026), the GM2 stage-2 A_crit (100 -> 211 m^2),
the joint A_crit on GM1, GM3, GM4 (8092 -> 1161, 6526 -> 1393, 7396 -> 3853 m^2, with D lower by 17-42 %) and
the joint MSE on AP2, AP3, GM1-GM5 (1-15 %); everything else moves at the printed precision. The exponent fits of
section 5 are an addition, not a replacement, and go into no config. Recommendation, as the directive asked for
one: v2, with the v1 stages kept available for the record (they are, under `--optimizer v1`).

Storage at this checkpoint (`sh_quota`, 2026-11-09 21:00 PST): `$GROUP_HOME` 262.8 GB of 1 TB, `$HOME` 10.9 of 15 GB. Results of all runs are under `$GROUP_HOME/lemcvt_results`; job logs under `$SCRATCH/lemcvt_jobs`.

**Stop for review.** Phase 4b (the D-infinity rebuild, addendum section 6) waits for this review. The nonlinear rows
of section 6 are appended when the runs finish, with no other change to this report.
