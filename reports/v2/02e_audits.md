# Audits: Picard convergence, the chi estimator and the hilltop set (addendum 3, sections 3 and 4)

Directive: `reports/directives/refactor_v2_addendum_3.md`, sections 3 and 4; outline fixed there.
Branch `v2`, commits 6a8bdb4 and 0d5889d (code), environment `lemcvt` (Python 3.12, numpy 2.5.4, scipy 1.18.1).
Machine-generated tables: `02e_tables.md` (`reports/v2/scripts/make_02e_tables.py`; P1-P4 for section 1, A1-A7
for sections 2-4). Figures: `reports/v2/figures_v2/02e_theta_estimators.png`, `02e_hilltop_exclusion.png`
(`lemcvt figures measure_audit`). Every number is tied to a run id under `$GROUP_HOME/lemcvt_results`.
Date: 2026-11-09. This is a checkpoint, not a stop.

## Summary

| Item | Result |
| --- | --- |
| 1. Picard, AP in feet | **All five AP tiles hit the 50-iteration cap**; final update 1.8e-5 (AP1) to 15.5 ft (AP3) against the tolerance 1e-6; 10-17 % of hillslope cells sit at the 0.99 S_c clip; 21-29 % of the final hillslope slopes exceed 0.9 S_c. GM converges in 6-11 iterations (0.1-2 % above 0.9 S_c). The v1 AP nonlinear results are unconverged |
| 1. Picard, AP in metres | With K and D both x 3.2808 (the Phase 5.4 invariance) AP1 and AP5 converge in 7 and 9 iterations with 0.2-0.5 % of cells above 0.9 S_c: the hypothesis is supported. The addendum's "D scaled by 1/3.2808" is inverted; run literally it does not converge and gives MSE 18000-31000 m^2 |
| 2. n_eff and estimators | The pooled estimator rests on n_eff = 1.0-2.4 basins on every AP tile at every threshold and on GM3-GM5 (GM1 3.1-3.6, GM2 1.9-4.1; sites 4.5-10); **no per-tile bootstrap interval of 02c on AP is a usable uncertainty**. Within the fixed basin set the equal-weight theta still rises with A_c on all five AP tiles and on GM1, GM3 (pixel population), but below ~2e3 m^2 the free ladder is additionally pulled down by many small basins with degenerate fits (basin selection); the 02c conclusion survives only in that restricted form |
| 3. Truncation | Wording corrected in `docs/measurement_methods.md` and `02c`. At 1e3 m^2 the complete basins give lower equal-weight theta (0.10-0.33) than the truncated ones (0.30-0.39), but complete basins inside a 1 km tile are the small ones, so this is a size comparison, not a truncation bias; the interior-only ladder is dropped |
| 4. Hilltop exclusion | Dropping hilltop pixels within L*/2 of the network raises C_ht on AP by 0-180 % depending on tile and network (AP1 at 1e3 m^2: 3.0e-3 -> 8.4e-3 1/m; D0 0.018 -> 0.0066) and by < 5 % on GM at 1e3 and 1e4; the hilltop set itself depends strongly on the network threshold on AP (C_ht 3.0e-3 to 6.3e-3 on AP1 between the hollow network and 1e4). On the post-exclusion widths the synthetic bias at L* is within 1 % (lower bound); on the hollow-network widths without exclusion it is -9 to -82 % on AP and -4 to -38 % on GM. The 02c synthetic (T7) used widths of a hilltop set that is not the one C_ht was measured on |

## 1. Picard audit (section 3)

### 1.1 What was done

`lemcvt audit picard --tile <tile>` (`src/lemcvt/analyses/picard_audit.py`; jobs 51705594 for the ten tiles,
51705597 / 51705612 / 51705615 for the metre variants; runs `20261109T2236..Z_picard_audit_<tile>_<variant>`).
The Picard loop of `partitioned_threshold_model_nonlinear` is instrumented without changing it: every
iteration records the update norms (the max |dz| that v1 tests, its RMS and its ratio to the hillslope relief),
the fraction of hillslope cells whose slope ratio was clipped at 0.99, and the loop records whether the
tolerance or the cap ended it; with `diagnostics=True` it also assembles the system at D_eff(z_new) and reports
the nonlinear residual ||M z_new - b||_2 / ||b||_2 and the final hillslope slopes. The record passes through
to `params.json` of every `tile forward` and `tile run` nonlinear evaluation (`picard` entry). Bit identity
with the v1 loop is asserted by `tests/test_models.py::TestPicardInstrumentation` (verbatim v1 loop on a
synthetic tile), and on the full tiles by the comparison with the step-5 gate grids: max |dz| = 0 on all five AP
tiles and GM1, 4e-14 to 8e-14 of relief on GM2-GM5 (different CPU generation, see `docs/numerics_changes.md` N7).

Each run evaluates the nonlinear model once at the v1 printed nonlinear optimum (K0_partitioned, D0, S_c = 1.27,
A_crit from `v1_reference.stages.nonlinear`), AP with `reproduce_v1_units: true` (z in US survey feet, x and y in
metres) as v1 ran. The metre variants load AP1 and AP5 with `reproduce_v1_units: false` in memory (`--metres`; no
config was changed) and scale K0 and D0 with `--scale-k`, `--scale-d`:

- `metres_KD_x3.2808`: K and D both multiplied by 3.2808. In feet the channel profile is z_ft = z_bc + (U/K) chi and
  the hillslope balance D nabla^2 z_ft = -U; with z_m = z_ft / 3.2808 the same surface in metres needs
  K_m = 3.2808 K_ft and D_m = 3.2808 D_ft (this is the Phase 5.4 statement "K and D scaled by 3.2808"). This
  variant's linear part is the feet solution converted to metres, so the only change for the nonlinear model
  is that the gradients are now compared with S_c in consistent units.
- `metres_D_div3.2808`: the factor the addendum's section 3.3 states, taken literally (D / 3.2808, K unchanged).
- `metres_D_x3.2808`: D x 3.2808 with K unchanged, to separate the two factors.

**On the factor in section 3.3.** "D must be scaled by 1/3.2808 so the linear part is equivalent" does not make
the linear part equivalent: dividing D by 3.2808 makes the metre hillslopes 10.8 x more curved than the converted
feet solution, and leaving K unscaled lifts the channel profile 3.28 x above the converted one. Both variants were
run so the point is demonstrated rather than argued (P3): the literal variant hits the cap with final updates of
190-440 m and MSE 18000-31000 m^2, the K-and-D variant converges in 7-9 iterations with MSE 161-205 m^2 (the
feet MSE 203 and 261 ft^2 convert to 18.8 and 24.3 m^2 at the linear-equivalent point; the nonlinear part moves
it, see 1.3).

### 1.2 Results, AP in feet (P1, P2, P4)

| tile | iterations | ended by | first update (ft) | final update (ft; tol 1e-6) | final residual (rel. L2) | clipped at 0.99 S_c | final slopes > 0.5 / > 0.9 S_c | DEM slopes > 0.5 / > 0.9 S_c | MSE (ft^2) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 50 | cap | 128 | 1.8e-5 | 1.4e-12 | 0.112 | 0.839 / 0.230 | 0.746 / 0.277 | 202.86 |
| AP2 | 50 | cap | 182 | 2.0e-3 | 1.7e-10 | 0.106 | 0.870 / 0.238 | 0.764 / 0.238 | 630.32 |
| AP3 | 50 | cap | 332 | **15.5** | 3.3e-6 | 0.174 | 0.842 / 0.292 | 0.724 / 0.279 | 650.85 |
| AP4 | 50 | cap | 223 | 0.19 | 2.0e-8 | 0.123 | 0.810 / 0.270 | 0.770 / 0.269 | 310.60 |
| AP5 | 50 | cap | 185 | 6.0e-4 | 6.2e-11 | 0.102 | 0.781 / 0.206 | 0.738 / 0.283 | 261.38 |
| GM1 | 10 | tolerance | 5.9 | 2.5e-7 | 4.2e-14 | 0.005 | 0.232 / 0.007 | 0.067 / 0.0003 | 77.07 |
| GM2 | 8 | tolerance | 4.1 | 8.3e-7 | 1.1e-13 | 0.016 | 0.120 / 0.020 | 0.032 / 0.0003 | 47.53 |
| GM3 | 6 | tolerance | 1.7 | 4.1e-7 | 2.7e-14 | 0.001 | 0.015 / 0.001 | 0.025 / 0.0006 | 58.98 |
| GM4 | 11 | tolerance | 10.9 | 5.4e-7 | 7.0e-14 | 0.012 | 0.277 / 0.017 | 0.100 / 0.0004 | 72.77 |
| GM5 | 10 | tolerance | 6.7 | 1.9e-7 | 2.7e-14 | 0.012 | 0.153 / 0.015 | 0.062 / 0.0001 | 54.81 |

(GM in metres. "DEM slopes" are the observed grid's slopes with the model's own stencil on the same hillslope
cells; on AP they are feet per metre.) The MSEs equal the v1 printed nonlinear MSEs to the printed digits on all
ten tiles (`00_reference.md` 3.2), as they must: v1 ran the same 50 iterations.

Reading. On AP the observed grid itself, with z in feet and x, y in metres, has 72-77 % of its hillslope cells
steeper than 0.5 S_c and 24-28 % steeper than 0.9 S_c (maximum 3.3-5.2 S_c); on GM the same fractions are
2.5-10 % and 0.01-0.06 %. The linear solution the iteration starts from inherits the feet gradients through the
channel and boundary values (54-67 % of hillslope cells above 0.9 S_c), the first Picard update is 128-332 ft,
and the iteration then drives 10-17 % of the cells to the 0.99 clip, where D_eff = 50 D. The convergence
histories (P4) fall by one to two orders of magnitude over the first ten iterations and then stall: AP3's update
is still 15.5 ft at iteration 50 (max relative residual 3e-6), AP4's 0.19 ft, AP2's 2e-3 ft; AP1 and AP5 are
within 1e-5 and 6e-4 ft of the tolerance, i.e. five to six orders of magnitude slower than GM, which reaches
1e-6 m in 6-11 iterations.

### 1.3 Results, AP1 and AP5 in metres (P3)

| tile | variant | iterations | ended by | final update (m) | clipped | final slopes > 0.5 / > 0.9 S_c | MSE (m^2) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | feet as v1 (for reference, ft) | 50 | cap | 1.8e-5 | 0.112 | 0.839 / 0.230 | 202.86 ft^2 = 18.8 m^2 |
| AP1 | K, D x 3.2808 | 7 | tolerance | 8.1e-7 | 0.001 | 0.063 / 0.002 | 160.6 |
| AP1 | D x 3.2808 only | 11 | tolerance | 3.4e-7 | 0.067 | 0.364 / 0.090 | 6615 |
| AP1 | D / 3.2808 (addendum text) | 50 | cap | 191 | 0.491 | 0.892 / 0.552 | 17969 |
| AP5 | feet as v1 (ft) | 50 | cap | 6.0e-4 | 0.102 | 0.781 / 0.206 | 261.38 ft^2 = 24.3 m^2 |
| AP5 | K, D x 3.2808 | 9 | tolerance | 1.7e-7 | 0.003 | 0.095 / 0.005 | 204.6 |
| AP5 | D x 3.2808 only | 10 | tolerance | 3.9e-7 | 0.022 | 0.261 / 0.027 | 3437 |
| AP5 | D / 3.2808 (addendum text) | 50 | cap | 444 | 0.236 | 0.906 / 0.652 | 30601 |

With K and D both scaled the observed metre grid has 1.6-1.9 % of hillslope cells above 0.5 S_c and 0-0.02 %
above 0.9 S_c; the nonlinear solution converges in 7-9 iterations with 0.1-0.3 % of cells clipped, like GM. Its
MSE (161 and 205 m^2) is 8-9 x the converted feet MSE (18.8 and 24.3 m^2): the converged nonlinear solution at
these (converted) parameters is a different, worse-fitting surface than the unconverged feet one, because in feet
the nonlinearity acted on gradients that are not the landscape's. This number is not a Phase 5 result (K and D
were not re-optimised in metres); it says only that the feet and metre nonlinear models are not the same model.

### 1.4 Decision rule (section 3.4), applied

- Every AP tile hits the cap in feet: **the v1 AP nonlinear results (A_crit and MSE of the nonlinear stage, all
  five tiles) are unconverged and are to be reported as such in the Phase 6 old-versus-new record.** AP3 is far
  from converged (15.5 ft); AP1 and AP5 are close (1.8e-5 and 6e-4 ft) and their MSEs would change little with
  more iterations; AP2 and AP4 are in between. The v1 nonlinear A_crit values were found by Nelder-Mead on these
  unconverged misfits.
- The metre runs converge: **the hypothesis is supported.** The mechanism is as stated: with z in feet every
  gradient is 3.28 x too steep against the dimensionless S_c = 1.27, a quarter of the hillslope cells start above
  0.9 S_c, D_eff is capped by the 0.99 clip on a tenth of the cells, and the Picard iteration stalls.
- Nothing is fixed here. Phase 5 makes the unit correction; the Phase 4 nonlinear profile on AP (section 5) runs
  in feet as the directive specifies and is read with this caveat.

Consequences for other parts of the plan, reported not acted on: (i) the Phase 3 nonlinear timings for AP
(330 s per evaluation) are the cost of 50 unconverged iterations; after the unit correction AP should need
about 10 iterations, like GM; (ii) the 02d statement that the nonlinear stage "recovers" the stage-2 misfit on AP
(`00_reference.md` 3.2) compares unconverged numbers.

## 2. n_eff and the equal-weight estimator (section 4.1)

### 2.1 What was done

`lemcvt audit measure --tiles all` (`src/lemcvt/measure/audits.py`; job 51707702, run
`20261109T230928Z_measure_audit_10tiles`, 4 min for ten tiles) reruns the free ladder, the basin-fixed ladder and the
site pooling of `02c` on the same per-basin statistics and the same pre-registered filters (>= 10 channel pixels,
relief >= 10 m, chi range >= 1 m at theta = 0.45; N_min = 50 pixels at 1e4 m^2 for the fixed set; basin bootstrap,
200 draws), with three estimators of theta side by side:

- **pooled**: the `02c` estimator, theta* = argmin sum_i SSR_i(theta) (through-origin, all basins pooled);
- **equal**: theta* = argmin sum_i SSR_i(theta) / SSR_i,null, SSR_i,null = basin i's sum of squares about its own mean
  elevation, so every basin counts once; its k_s is the pooled through-origin slope at that theta (and the median of
  the per-basin slopes is given beside it);
- **median**: the median over the kept basins of each basin's own best theta.

n_eff = 1 / sum_i w_i^2 is reported for the pooled estimator at its theta*, with w_i basin i's share of the total
sum (chi - chi_bar)^2 as the addendum defines it, and, in parentheses in A1, of the through-origin leverage sum chi^2
that the pooled objective actually weights by; the two agree within 0.3 everywhere. Where n_eff < 3 the bootstrap
is flagged as not a usable uncertainty. Tables A1-A5; figure `02e_theta_estimators.png`.

### 2.2 n_eff at every ladder step (A1)

| | n_eff of the pooled estimator (centred weights), 1e2 -> 1e4 m^2 | basins at 1e3 / 1e4 |
| --- | --- | --- |
| AP1 | 1.3 -> 1.2 (never above 1.6) | 32 / 5 |
| AP2 | 1.9 -> 2.0 | 13 / 2 |
| AP3 | 1.3 -> 1.0 | 32 / 2 |
| AP4 | 2.2 -> 2.3 (max 2.4) | 30 / 5 |
| AP5 | 1.6 -> 1.8 | 9 / 2 |
| GM1 | 3.6 -> 3.1 | 19 / 8 |
| GM2 | 4.0 -> 2.1 (below 3 from 4e3 m^2) | 16 / 5 |
| GM3 | 2.6 -> 2.2 (never above 2.9) | 15 / 7 |
| GM4 | 1.7 -> 1.3 | 14 / 3 |
| GM5 | 1.4 -> 1.1 | 4 / 2 |
| AP pooled | 7.6 -> 6.9 (4.5 at 1e5) | 116 / 16 |
| GM pooled | 10.2 -> 7.5 (2.9 at 1e5) | 68 / 25 |

On every AP tile, at every step from 1e2 to 1e5 m^2, the pooled fit is one to two basins' worth of information,
however many basins pass the filters (110 on AP1 at 1e2 m^2, n_eff 1.3). The same holds on GM3-GM5; GM1 and GM2
reach 3-4 below 2e3 m^2. The site pooling reaches 7-10. Consequence for `02c`: every per-tile 68 % and 95 % interval
in T1, T2 and T8 on AP and on GM3-GM5 is a bootstrap over one or two effective basins and is not a usable
uncertainty (the AP1 and AP3 intervals that reached the grid edge 0.10 were the visible symptom); the per-tile
values at A_c >= 4e3 m^2 were already called unusable in `02c` section 9 for the basin count, and the n_eff
extends that to every step. The site-pooled intervals (n_eff 5-10) stand.

### 2.3 The three estimators on the free ladder (A2)

Pooled against equal-weight theta* (site pooled; per tile in A2):

| A_c (m^2) | 1e2 | 4e2 | 1e3 | 2e3 | 4e3 | 1e4 | 4e4 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| AP pooled, pooled estimator | 0.32 | 0.35 | 0.39 | 0.43 | 0.48 | 0.52 | 0.55 |
| AP pooled, equal-weight | 0.12 | 0.22 | 0.29 | 0.36 | 0.45 | 0.53 | 0.53 |
| AP pooled, basin median | 0.10 | 0.15 | 0.14 | 0.15 | 0.43 | 0.56 | 0.54 |
| GM pooled, pooled estimator | 0.27 | 0.30 | 0.33 | 0.34 | 0.33 | 0.31 | 0.27 |
| GM pooled, equal-weight | 0.11 | 0.22 | 0.30 | 0.34 | 0.34 | 0.29 | 0.25 |
| GM pooled, basin median | 0.10 | 0.10 | 0.31 | 0.36 | 0.42 | 0.35 | 0.31 |

The three estimators agree above about 4e3 m^2 on AP and above about 1e3 m^2 on GM (within 0.05 for the site
pooling; per tile the exception is AP1, where the equal-weight theta stays at 0.16-0.40 against the pooled 0.53-0.58
from 4e3 to 1e4 m^2, because AP1's pooled value is one basin's). Below those thresholds they diverge: the
equal-weight theta falls to 0.10-0.19 at 1e2 m^2 on every tile (grid edge 0.10 on AP1, AP3, GM1, GM2, GM4), and the
basin median sits at the grid edge on 8 of 10 tiles below 4e2-8e2 m^2. The reason is in the per-basin fits: at
small A_c most kept basins are short hollow networks of 10-50 pixels whose chi-z relation is best fitted with the
smallest theta on the grid (chi nearly proportional to distance), and the equal-weight and median estimators give
those basins the same say as the few long networks that the pooled estimator listens to.

k_s follows theta (A5): at 1e3 m^2 the equal-weight theta moves k_s from 5.4 to 0.16 m^(2 theta) on AP1 (theta
0.39 -> 0.10) and by factors of 1.2-2.3 elsewhere; these are unit changes with theta, not steepness changes
(`02c` framing point 2), and nothing here is an input to a tile config.

### 2.4 Basin-fixed against free, by estimator (A3): does the pixel-population conclusion survive?

| | pooled: max |fixed - free| | equal: max |fixed - free| | equal-weight theta in the fixed set, 1e2 -> 1e4 | median, 1e2 -> 1e4 |
| --- | --- | --- | --- | --- |
| AP1 (2 basins kept) | 0.01 | 0.22 | 0.30 -> 0.40 | 0.29 -> 0.34 |
| AP2 (2) | 0.01 | 0.12 | 0.31 -> 0.56 | 0.31 -> 0.57 |
| AP3 (3) | 0.00 | 0.13 | 0.22 -> 0.45 | 0.10 -> 0.28 |
| AP4 (6) | 0.00 | 0.08 | 0.23 -> 0.73 | 0.26 -> 0.84 |
| AP5 (2) | 0.00 | 0.15 | 0.31 -> 0.51 | 0.31 -> 0.53 |
| GM1 (9) | 0.01 | 0.14 | 0.22 -> 0.42 | 0.24 -> 0.46 |
| GM2 (6) | 0.01 | 0.15 | 0.25 -> 0.28 | 0.26 -> 0.28 |
| GM3 (8) | 0.00 | 0.09 | 0.24 -> 0.45 | 0.23 -> 0.50 |
| GM4 (3) | 0.01 | 0.15 | 0.25 -> 0.25 | 0.27 -> 0.26 |
| GM5 (2) | 0.00 | 0.12 | 0.28 -> 0.28 | 0.28 -> 0.28 |

The addendum's objection is confirmed: the pooled free and fixed ladders agree within 0.01 because the free fit was
already the fixed set's (n_eff 1-2), so the `02c` test could not discriminate. With the equal-weight estimator the
two ladders differ by 0.08-0.22, and the picture splits:

- **Within the fixed basins** the equal-weight theta rises with A_c on all five AP tiles (by 0.10 on AP1 to 0.50 on
  AP4) and on GM1 and GM3 (by 0.20), and is flat on GM2, GM4 and GM5. That part of the `02c` conclusion, the rise is
  a pixel-population effect inside one and the same network, survives the change of estimator on AP and on two of
  the five GM tiles; its size now differs by tile, and the fixed sets are 2-9 basins.
- **On the free ladder** the equal-weight theta below about 2e3 m^2 (AP) and 1e3 m^2 (GM) is lower than the fixed
  set's at the same A_c (0.10-0.19 against 0.22-0.31 at 1e2 m^2), i.e. the small basins that enter at low thresholds
  carry a lower theta than the large ones. That is a basin-selection effect the pooled estimator could not show.

So the statement "pixel population, not basin selection" does not survive as written: both effects are present,
the first inside the large networks on AP, GM1 and GM3, the second wherever small basins outnumber large ones.
Which of the two the models should care about depends on which network the channel law is meant for, which is
the threshold decision Steve has deferred.

### 2.5 T8 with n_eff (A5)

A5 repeats `02c` T8 (theta / k_s / K0 at the three threshold options) for the pooled and the equal-weight
estimators with n_eff and the basin count beside each value. n_eff is below 3 for all 15 AP rows and for 11 of
15 GM rows (GM1 all three options and GM2 at 1e3 and at the site break are the exceptions, with n_eff 3.4-3.8), so
on AP no tile-level bootstrap interval of T8 is a usable uncertainty, and the site-pooled values (n_eff 5-10) are
the ones with an interval. The two estimators agree within 0.05 in theta on AP2, AP5 and at 1e4 m^2 on AP3; they
differ by 0.05-0.29 elsewhere, most on AP1 (0.39 against 0.10 at 1e3 m^2).

## 3. Truncation (section 4.2)

The wording is corrected in `docs/measurement_methods.md` (section 2) and in `02c` (sections 6 and 9, marked as
corrected): the area grids hold true areas, chi is integrated upstream from each in-tile outlet with those areas,
and a catchment cut by the tile edge yields a valid, shorter chi-z profile; truncation shortens the sample, it does
not bias it.

The one-time comparison at A_c = 1e3 m^2 (A4; the free ladder's kept basins split by the `02c` truncation flag):

| | truncated basins: pooled / equal / median (n) | complete basins: pooled / equal / median (n) |
| --- | --- | --- |
| AP1 | 0.39 / 0.35 / 0.10 (5) | 0.10 / 0.10 / 0.10 (27) |
| AP2 | 0.41 / 0.39 / 0.36 (5) | 0.58 / 0.33 / 0.30 (8) |
| AP3 | 0.38 / 0.38 / 0.41 (4) | 0.14 / 0.18 / 0.34 (28) |
| AP4 | 0.37 / 0.37 / 0.36 (7) | 0.36 / 0.20 / 0.11 (23) |
| AP5 | 0.39 / 0.37 / 0.34 (5) | 0.16 / 0.10 / 0.12 (4) |
| GM1 | 0.31 / 0.33 / 0.41 (4) | 0.36 / 0.20 / 0.10 (15) |
| GM2 | 0.31 / 0.30 / 0.32 (7) | 0.10 / 0.10 / 0.25 (9) |
| GM3 | 0.28 / 0.35 / 0.35 (8) | 0.38 / 0.23 / 0.37 (7) |
| GM4 | 0.35 / 0.36 / 0.36 (4) | 0.21 / 0.17 / 0.15 (10) |
| GM5 | 0.32 / 0.31 / 0.32 (2) | 0.10 / 0.10 / 0.10 (2) |

The equal-weight estimator does show a difference: complete basins give 0.10-0.33, truncated ones 0.30-0.39.
But the complete basins at 1e3 m^2 are, by construction, the small catchments that fit inside a 1 km tile (median
channel length a few tens of pixels), and small basins give low theta on every ladder (section 2.3). The comparison
therefore measures basin size, not truncation, and cannot separate the two with these tiles. The interior-only
ladder is dropped from later reports, as the addendum directs, with this caveat recorded.

## 4. Hilltop exclusion (section 4.3)

### 4.1 What was done

For every tile and for the channel networks at A_c = 1e3 and 1e4 m^2, and for the rollover network that
`lemcvt measure` actually uses to define hilltops (18-450 m^2; `02b` table 2), the hilltop set is built as in
`measure` (divide cells adjacent to a cell draining to a different outlet basin of that network, two cells from
the edge, |grad z| <= 0.4), then once more with every pixel closer than L*/2 to a channel pixel of that network
dropped (L* = the tile's window from the measure run: AP 50, 20, 40, 40, 20 m; GM 9, 11, 15, 21, 9 m). C_ht is the
median of -nabla^2 z over the set at the fixed L*, with a bootstrap over 8-connected segments (200 draws), and
D0 = U/C_ht; the plateau scale is also re-selected on each set. Table A6; figure `02e_hilltop_exclusion.png`.

A premise of the addendum does not hold in the repo: section 4.3.1 says the network at 1e4 m^2 is "which the
current hilltop definition uses". `lemcvt measure` defines the hilltops from the rollover network (A_c* = 18-450
m^2), and the `02b` C_ht values come from that set; the `02c` width distribution (T7) was taken at 1e4 m^2 but on
a hilltop set built without the outlet-basin relabelling, which is not the set C_ht was measured on (next
subsection). The directive's cases were run as specified, and the rollover case added so that the `02b` values can
be placed in the same table (the "rollover, all" rows reproduce them: AP1 3.043e-3 against 3.044e-3, GM3 8.110e-3
against 8.110e-3, and so on for all ten tiles).

### 4.2 Pixel counts, segments, C_ht and D0 with and without the exclusion (A6)

C_ht (1/m) at the fixed L*, "all -> excluded", with the pixel counts in parentheses:

| tile | L* (m) | network 1e3 m^2 | network 1e4 m^2 | rollover network (`02b` set) |
| --- | --- | --- | --- | --- |
| AP1 | 50 | 3.00e-3 -> 8.39e-3 (3964 -> 1786) | 6.29e-3 -> 6.75e-3 (861 -> 749) | 3.04e-3 -> 1.15e-2 (6104 -> 1395) |
| AP2 | 20 | 6.92e-3 -> 7.52e-3 (1898 -> 1545) | 7.99e-3 -> 8.04e-3 (869 -> 833) | 6.13e-3 -> 1.47e-2 (3497 -> 308) |
| AP3 | 40 | 7.12e-3 -> 9.33e-3 (3365 -> 2195) | 3.10e-3 -> 6.14e-3 (693 -> 475) | 4.76e-3 -> none left (9409 -> 0) |
| AP4 | 40 | 6.16e-3 -> 8.51e-3 (4794 -> 2966) | 6.76e-3 -> 8.72e-3 (2600 -> 1870) | 5.74e-3 -> 9.61e-3 (6136 -> 2473) |
| AP5 | 20 | 8.61e-3 -> 8.86e-3 (2673 -> 2505) | 8.94e-3 -> 8.94e-3 (1354 -> 1325) | 8.89e-3 -> 1.78e-2 (7107 -> 1512) |
| GM1 | 9 | 1.445e-2 -> 1.454e-2 (5094 -> 5042) | 1.484e-2 -> 1.484e-2 (4128 -> 4126) | 1.451e-2 -> 1.677e-2 (11112 -> 4521) |
| GM2 | 11 | 1.140e-2 -> 1.137e-2 (4808 -> 4661) | 1.104e-2 -> 1.114e-2 (3550 -> 3516) | 9.91e-3 -> 1.505e-2 (9187 -> 2530) |
| GM3 | 15 | 7.07e-3 -> 7.18e-3 (7240 -> 6941) | 6.16e-3 -> 6.12e-3 (5095 -> 5039) | 8.11e-3 -> 1.415e-2 (14030 -> 219) |
| GM4 | 21 | 1.388e-2 -> 1.451e-2 (3278 -> 3097) | 1.594e-2 -> 1.594e-2 (1881 -> 1872) | 1.158e-2 -> 2.051e-2 (6225 -> 1410) |
| GM5 | 9 | 1.781e-2 -> 1.813e-2 (1007 -> 969) | 1.774e-2 -> 1.791e-2 (787 -> 765) | 1.464e-2 -> 1.946e-2 (4964 -> 1400) |

D0 = U/C_ht (m^2/yr), the same cells: AP1 0.0183 -> 0.0066 (1e3), 0.0087 -> 0.0082 (1e4), 0.0181 -> 0.0048
(rollover); AP2 0.0079 -> 0.0073, 0.0069 -> 0.0068, 0.0090 -> 0.0037; AP3 0.0077 -> 0.0059, 0.0177 -> 0.0090;
AP4 0.0089 -> 0.0065, 0.0081 -> 0.0063, 0.0096 -> 0.0057; AP5 0.0064 -> 0.0062, 0.0062 -> 0.0062, 0.0062 -> 0.0031;
GM1 0.0102 -> 0.0101, 0.0099 -> 0.0099, 0.0101 -> 0.0088; GM2 0.0129 -> 0.0129, 0.0133 -> 0.0132, 0.0148 -> 0.0098;
GM3 0.0208 -> 0.0205, 0.0239 -> 0.0240, 0.0181 -> 0.0104; GM4 0.0106 -> 0.0101, 0.0092 -> 0.0092, 0.0127 -> 0.0072;
GM5 0.0083 -> 0.0081, 0.0083 -> 0.0082, 0.0100 -> 0.0076. Segment counts and the 68 % intervals are in A6; the
intervals are 5-15 % wide on most rows and wider where fewer than 100 segments remain (AP1 and AP3 at 1e4 m^2,
AP2 and GM3 on the excluded rollover set).

Reading, per the three questions the addendum asks:

- **Effect of the exclusion.** On GM at 1e3 and 1e4 m^2 it changes C_ht by less than 5 % and removes 1-6 % of the
  pixels: the GM hilltops are wide (median 2d = 110-250 m to those networks) against L* = 9-21 m. On AP it removes
  6-55 % of the pixels at 1e3 m^2 and raises C_ht by 3-180 % (AP1 x 2.8, AP3 x 1.3, AP4 x 1.4; AP2 and AP5, whose L*
  is 20 m, by 4-9 %); at 1e4 m^2 the effect is 0-98 % (AP3 x 2.0 on a set of 475 pixels with a wide interval). On
  the rollover network, where "channel" means any hollow with A >= 18-450 m^2, the exclusion removes 59-100 % of
  the pixels and raises C_ht by 15-140 % on every tile, AP and GM alike, and leaves nothing on AP3 (L*/2 = 20 m,
  rollover at 17.8 m^2: no divide cell is 20 m from a hollow).
- **Dependence on the network threshold** (no exclusion). On GM the three networks give C_ht within 20 % of
  each other except GM4 (1.16e-2 rollover against 1.59e-2 at 1e4 m^2) and GM3 (8.1e-3 against 6.2e-3). On AP the
  spread is a factor 2 (AP1: 3.0e-3 on the rollover and 1e3 networks, 6.3e-3 at 1e4 m^2; AP3: 4.8e-3, 7.1e-3,
  3.1e-3) and the 1e4 sets are small (690-1350 pixels on AP1-AP3, AP5). The hilltop set is not a stable object on
  AP: which divides count, and how far they sit from the nearest valley axis, depends on the threshold that defines
  the valleys.
- **The re-selected scale.** After the exclusion the plateau rule picks a shorter window on AP1 (50 -> 20-24 m) and
  AP3 (40 -> 20 m) and the same window elsewhere (A6); at the re-selected L* the excluded-set C_ht is 0-13 % higher
  still. The fixed-L* values above are the ones to compare with `02b`.

### 4.3 Synthetic-ridge bias on the post-exclusion widths (A7)

The parabolic-ridge calculation of `02c` section 7 was rerun with crest widths from the post-exclusion distribution
of 2 x (hilltop-to-channel distance), as the addendum specifies, with the network at 1e4 m^2 and the site's
post-exclusion median C_ht as the true curvature (AP 8.0e-3, GM 1.48e-2 1/m), TIN facets of 2.76 m and noise of
0.043 m as before:

| | post-exclusion width 2d at the 1e4 network, 25 / 50 % (m) | bias at L*, median width | 25 % width |
| --- | --- | --- | --- |
| AP1-AP5 | 68-268 / 241-298 | -0.6 to +0.2 % | -0 to +1 % |
| GM1-GM5 | 107-230 / 130-251 | -2.5 to +0.4 % | 0 to +2 % |

Against the 1e4 network the post-exclusion widths are 130-300 m, far above any L*, and the bias is within 1-3 %.
But 2 x the distance to a channel of 1e4 m^2 is not the crest width that the window bias depends on: between a
divide and such a channel the surface is dissected by hollows, and the parabolic crest ends at the nearest hollow
axis, which is what the rollover network marks. Taken at the rollover-network widths (A6, last column of the
"rollover" rows), the same synthetic curves give, without the exclusion, a bias at L* of -51 % (AP1, median
width 14 m), -52 % (AP2, 6 m), -82 % (AP3, 4 m), -9 % (AP4, 27 m), -21 % (AP5, 11 m) and -4 to -38 % on GM (widths
4-12 m); with the L*/2 exclusion relative to the hollows the remaining pixels have widths of 26-76 m (AP) and 14-26 m
(GM) and the bias is within 2 % everywhere. The two width definitions bracket the answer; the addendum's choice
(1e4 network) is the optimistic end.

**Known limitation** (section 4.3.4). The synthetic ridge is parabolic all the way to the channel, which is the
steady linear-diffusion profile. Real crests with planar flanks are narrower than 2 x the divide-to-channel
distance, so the bias computed at a given 2d is a lower bound on the bias at that distance, for either network.

### 4.4 Where this leaves the `02b` C_ht and the `02c` T7

- The `02b` C_ht values are medians over the rollover-network hilltop set without exclusion; on that set the
  synthetic bias at L* is -9 to -82 % on AP and -4 to -38 % on GM at the median width (previous subsection), far
  above the pre-registered 10 %. `02c` section 7 reported -6.5 to +1.1 % at the median width and did not trigger
  the rule; it used the width distribution of a different hilltop set (next item), and the rule as stated there is
  not met once the set is the right one. This is reported, not acted on: no D0 goes into a config (section 1 of the
  addendum).
- `02c` T7's widths ("medians 28-50 m on AP, 31-102 m on GM") came from `hilltop_widths` in
  `measure/diagnostics.py`, which builds the hilltop mask at 1e4 m^2 without the outlet-basin relabelling that
  `measure` applies; that set has 24000-36000 pixels per tile (every divide cell next to a cell draining to a
  different channel pixel) against 690-5100 for the relabelled set at 1e4 m^2 and 3500-14000 for the rollover set
  `measure` uses. The T7 widths therefore describe neither set, and the `02c` statement that the bias at the
  site-typical crest width is under 10 % rests on them. The diagnostics code is left as it is (it is `02c`'s
  record); the audit's widths are the ones to use.
- Option (i) as specified (exclusion at L*/2 from the 1e4 network) changes little on AP5, AP2 and GM and much on
  AP1, AP3, AP4; option (i) relative to the hollow network changes C_ht by 15-140 % on every tile and removes most
  of the hilltop pixels (all of them on AP3 at L* = 40 m). Whether the hilltop set should be defined against the
  hollows or against a channel threshold, and at which window, is the curvature-method decision the addendum
  deferred to Steve; both versions are now on the table with their intervals.

## 5. What the evidence supports, and where it is weak

Measured (the tables), then inferred.

- **Picard.** Measured: all five AP feet runs end at the cap; the final updates, residuals, clip fractions and slope
  distributions are in P1-P4; the metre runs with K, D x 3.2808 converge in 7-9 iterations. Inferred: the mechanism
  (feet gradients against a dimensionless S_c). Weak: the metre runs use converted v1 parameters, not metre optima,
  so their MSEs say nothing about fit quality; only AP1 and AP5 were run in metres.
- **n_eff.** Measured: n_eff 1.0-2.4 on every AP tile and on GM3-GM5 at every step; 7-10 pooled. Inferred: no
  per-tile AP interval in `02c` is usable; the site-pooled ones are. Weak: n_eff is a property of one weighting at
  one theta; the equal-weight estimator's own effective count is the basin count by construction, which says
  nothing about how informative the small basins are.
- **Pixel population against basin selection.** Measured: within the fixed sets the equal-weight theta rises on
  AP1-AP5, GM1, GM3 and is flat on GM2, GM4, GM5; on the free ladder the equal-weight and median thetas sit at
  0.10-0.19 below 2e3 m^2 (AP) and 1e3 m^2 (GM). Inferred: both effects exist. Weak: the fixed sets are 2-9 basins;
  the equal-weight estimator at small A_c is driven by basins with 10-50 pixels whose best theta is the grid edge,
  so "0.10" there is a statement about degenerate fits, not about concavity; the theta grid's lower edge (0.10)
  censors it.
- **Truncation.** Measured: complete basins give lower theta than truncated ones at 1e3 m^2. Inferred: a size
  effect. Weak: not separable from truncation with 1 km tiles.
- **Hilltops.** Measured: C_ht with and without exclusion on three networks, with intervals; the `02b` values
  reproduced on the rollover set; the width distributions of the relabelled sets. Inferred: the bias of the `02b`
  values is large if crest width is distance to the hollow axis, small if it is distance to a 1e4 channel; the
  `02c` T7 used neither set. Weak: the synthetic ridge is the linear-diffusion profile with one width per case; the
  hillslope between divide and hollow on AP is a TIN surface with facets of 2.8 m, and "distance to the nearest
  hollow pixel" on the rollover network (A_c 18-450 m^2) is a lower bound on the crest half-width where the rollover
  sits inside the hillslope noise (AP3, 17.8 m^2).

Nothing in this report writes to a tile config. The decisions it bears on, the threshold rule (with its estimator)
and the curvature method, remain Steve's; the next step in the addendum's order is Phase 4
(`04_optimizer_v2.md`), whose section 5.2 plot uses the measured theta(A_c) curves from here (pooled and
equal-weight, with their 95 % bands) as the addendum allows.
