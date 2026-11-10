# [2026-11-10 15:40 PST] Checkpoint 4b/5: the D-infinity rebuild, the AP unit correction, and four analyses for the deferred decisions (addendum 4)

Directive: `reports/directives/2026-11-10T0545PST_refactor_v2_addendum_4.md` (outline fixed by its section 7).
Branch `v2`; code commits 9fb5d18, a4c922c, 6380565, 0bbf782, 79eac4a, da01bfe (this session); the report, its
appendix and the regenerated tables are in the final commit. Environment `lemcvt` (Python 3.12, numpy 2.5.4,
scipy 1.18.1); every job on `serc`, 1 CPU unless stated. Machine-generated tables: section 9 of this file
(appendices A1-A11, `reports/v2/scripts/make_checkpoint_4b_5_tables.py`); figures
`reports/v2/figures_v2/a4_exponent_site.png` (`lemcvt figures exponent_site`) and the regenerated
`04_acrit_profiles.png` and `d8_vs_dinf_*.png`. Every number is tied to a run id under
`$GROUP_HOME/lemcvt_results` or a Slurm job id. Date: 2026-11-10. **This is a stop for review** (section 8).

## Summary

| Item | Result |
| --- | --- |
| State, push | `v2` was at 0fde82f = `origin/v2` (the Addendum 3 push had happened); tests 178/178 on serc; six code commits this session, pre-push check clean, pushed with the report |
| Nonlinear rows (decision 2) | AP5 baseline: pass against v1 (A_crit -8e-6, MSE -1e-5) and identical to the Phase 3 trajectory; the Phase 3 gate is now 8/8 (baseline) and 7/8 (Phase 3, AD excepted). GM v2: v1's step reproduced on GM1-GM5 (MSE equal to 6 digits), **GM1 3 and GM2 8 evaluations hit the Picard cap**; job 51707588 cancelled |
| Phase 4b | Coupled-system D-infinity partitioned model: equals D8 on aligned surfaces to 1e-12 of relief, reduction exact, zero channel cells below every receiver on AP1/GM1; `d8_vs_dinf` PT D-inf MSE 3.77e6 -> 7661 ft^2 (AP1), 4240 -> 1703 m^2 (GM1), still 2-4x the D8 MSE at the v1 parameters |
| Phase 5 | AP configs in metres (exact US survey foot; K, D-type parameters scaled by c in the loader); corrected grids written and in the manifest; **D8 check: AP2-AP5 clean, AP1 has a processing seam on rows 389-390 (345 cells) in the stored flow directions**; invariance passes on AP1 and AP5 for all three profiles (optima identical, K and D x c to 1e-13, MSE x 0.0929034 to 1e-11; AD 2.7e-9); AP nonlinear in metres converges everywhere (0 of 1305 evaluations at the cap), optima 995-4196 m^2 against v1's unconverged 3399-10233 |
| 6.1-6.2 | Site exponent m: **AP 0.47 (1 %: 0.44-0.49), GM 0.31 (0.29-0.33)**, leave-one-out within 0.03; inside the measured theta bands at the median fitted A_crit on AP, 0.02 below the pooled band on GM; **U/k_s / K_fit = 0.88-1.11 on every tile at both 10^3 and 10^4 m^2** |
| 6.3 | Option (iii): L_f = 11 m from the synthetic (8-realisation mean; a single realisation gave 21 m); C_ht 1.5-2.6e-2 (AP), 1.5-1.9e-2 (GM); D0 2.2-3.6e-3 (AP), 7.6-10e-3 (GM), beside the `02e` options with definitions, windows and pixel counts |
| 6.4 | `hilltop_widths` now uses the `measure` set (24-36k -> 0.7-5k pixels at 10^4 m^2); theta grid from 0.00: 20-66 % of kept basins at A_c <= 2e3 m^2 fit best at theta = 0; pooled and site equal-weight values unchanged, median estimator 0.00-0.05 at small A_c |
| Not done / changed on the way | A first D8-check classification (mirrored offsets) and a first L_f (one realisation) were wrong and were rerun; an invariance run crashed in its comparison code after 3 h and was rerun per stage; the `d8_vs_dinf` step was OOM-killed at 32 GB and rerun at 64 GB |

## 1. State check and push status

- Read in order: the directive, addenda 1-4, the reports `00`, `02`, `02d`, `02c`, `02e`, `03`, `03b`, `04` with their
  tables, `docs/numerics_changes.md` and `docs/measurement_methods.md`.
- `v2` head at the start of the session: 0fde82f ("04: storage usage at the checkpoint"), identical to `origin/v2`:
  the Addendum 3 section 1 push had happened (all four branches); the local tracking ref was stale and a `git fetch`
  settled it. The Addendum 2 section 5 pre-push check on this session's commits is clean (no `papers/`, no `*.pdf`
  tracked, no file over 50 MB, no secret in the diffs); pushed to `origin/v2` with this report (the push is the last action of this session).
- Tests on `serc` (job 51780220, node sh04-12n22, Genoa): **178 passed** at a4c922c (159 before this session; the
  new ones are the Phase 4b tests, the loader's unit scaling and the override). The addendum and the repo did
  not disagree anywhere I could check; one premise in the addendum's own text turned out wrong and is reported in
  section 6.1 (the synthetic's window).
- Jobs named in section 0.2:
  - 51697696 (AP5 nonlinear baseline, Phase 2 code): COMPLETED in 15 h 38 min, inside its 32 h limit; its row is in
    section 2 and was filled into `03b_phase3_close.md` (sections 1, 2, 4, 6) and `03_tables.md` (O1-O3).
  - 51707582 (GM nonlinear, v2): GM1 and GM3 completed; GM2, GM4 and GM5 hit the 6 h limit at 181-190 of their
    200 grid values (the Picard iteration count grows from 6-10 near the minimum to 40-50 at A_crit ~ 5e4-1e5
    m^2 and each iteration costs 5-10 s there, so the 200 + 60 evaluations take 6-8 h on GM, not the 3.3 h the
    Phase 3 per-evaluation time predicted) and wrote no trajectory. Resubmitted with a 16 h limit (job 51779145)
    after this cause was established; rows in section 2.
  - 51707588 (AP nonlinear, v2, feet): **cancelled** at 14 h 36 min of its 36 h (decision 2); nothing of it is
    used. The metre rerun is section 4.6.
- Addendum 2 section 4: every array throttled; total concurrent CPUs stayed below 25; results under
  `$GROUP_HOME/lemcvt_results` (4.3 GB in total), logs under `$SCRATCH/lemcvt_jobs`. Storage at this checkpoint
  (`sh_quota`, 2026-11-10 13:50 PST): `$GROUP_HOME` 263.3 GB of 1 TB; `$HOME` 10.9 of 15 GB; `$SCRATCH` 804 GB;
  `$OAK` 16.5 of 20 TB.
- Core-hours submitted this session (limits, not usage): 48 (GM rerun) + 8 (utilities) + 24 + 48 (invariance, two
  submissions) + 80 (AP nonlinear in metres) + 50 (exponent profiles) + 4 (D-infinity rerun) + 4 (audit rerun) =
  266, every single submission below the 300 core-hour line. Actual use was about 135 core-hours.

## 2. Nonlinear rows: GM v2 and the AP5 baseline (decision 2)

**The AP5 nonlinear baseline** (v1 optimizer, Phase 2 code, job 51697696, run `20261109T211106Z_tile_run_AP5`):
A_crit 3399.07 m^2 against v1's printed 3399.1 (-8.1e-6 relative), MSE 261.377 ft^2 against 261.38 (-1.1e-5), 29
Nelder-Mead evaluations and 26 solves in 15 h 38 min (1870 s per evaluation). It passes the Addendum 2 section 3
rule, and the Phase 3 run of the same stage (`03b` O2, now filled) reproduces its trajectory exactly: the same
29 evaluations in the same order with a maximum per-evaluation misfit difference of 0. With that row the Phase 3
optimizer gate is complete: 8 of 8 baseline stages pass against v1, 7 of 8 Phase 3 stages pass, the one failure
being the AP5 AD stage whose L-BFGS-B path diverges through N7 (`03b` section 2; decision 1 of the addendum keeps
N7 and settles the AD result with the v2 profile, recorded in `docs/numerics_changes.md`). Every evaluation of
this baseline is a 50-iteration Picard result in feet (`02e` section 1), so the stage reproduces v1's number, not
a converged model. `03b` sections 1, 2, 4 and 6 and `03_tables.md` were updated in place with this row and nothing
else.

**GM nonlinear, v2 profile** (200 log-thinned unique areas plus the unique values inside the bracket around the
grid minimum, every evaluation a Picard solve at K0, D0, S_c = 1.27; A1):

| tile | run (job) | evaluations hitting the cap | v2 optimum A_crit (m^2) (Picard it.) | MSE (m^2) | v1 A_crit, same code: MSE (it.) | v2 - v1 | 1 % interval (m^2) | hillslope > 0.5 / > 0.9 S_c at the optimum | wall (h) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GM1 | `20261109T231303Z_tile_run_GM1_v2_nonlinear` (51708241) | **3** of 233 | 475 (10) | 77.0733 | 474.21: 77.0733 (10) | 0 (same step) | 460-548 (38 pts) | (not recorded; run predates commit 9fb5d18) | 5.7 |
| GM2 | `20261110T141003Z_tile_run_GM2_v2_nonlinear` (51779145) | **8** of 216 | 245 (8) | 47.5334 | 244.59: 47.5334 (8) | 0 (same step) | 207-264 (22 pts) | 0.120 / 0.0197 | 7.7 |
| GM3 | `20261109T231258Z_tile_run_GM3_v2_nonlinear` (51708446) | 0 of 261 | 1298 (6) | 58.9848 | 1297.4: 58.9848 (6) | 0 (same step) | 1257-1395 (64 pts) | (not recorded) | 3.8 |
| GM4 | `20261110T141003Z_tile_run_GM4_v2_nonlinear` (51779145) | **15** of 234 | 525 (11) | 72.7622 | 521.38: 72.7734 (11) | -1.5e-4 (neighbouring step) | 454-578 (41 pts) | 0.277 / 0.0169 | 9.3 |
| GM5 | `20261110T141003Z_tile_run_GM5_v2_nonlinear` (51779145) | 0 of 219 | 300 (10) | 54.8130 | 299.62: 54.8130 (10) | 0 (same step) | 261-393 (30 pts) | 0.153 / 0.0155 | 6.4 |

Reading. On GM1, GM2, GM3 and GM5 the v2 profile's minimum is the staircase step v1's Nelder-Mead found (the v1
A_crit evaluated by the same code gives the same MSE to six digits); on GM4 it is the neighbouring step (525
against 521.38 m^2, MSE lower by 1.5e-4 relative). So for the nonlinear stage, unlike the stage-2 and joint stages
of `04`, v1 was at or next to the optimum on every GM tile. The 1 % intervals are 10-40 % wide in A_crit, like the
stage-2 ones. Two things are new: (i) **the Picard cap is hit on GM too**, by 3 evaluations on GM1, 8 on GM2 and 15
on GM4 (GM3 and GM5 none); those evaluations are at large A_crit far from the minima (the iteration count rises
with A_crit on every tile as more of the grid becomes hillslope and the slopes near the clip multiply, which is
also why these profiles took 6-9 h), so the optima are converged results, but a profile value at the cap is not
a model value and A1 counts them; (ii) at the optimum 12-28 % of the hillslope cells on GM2, GM4 and GM5 exceed
0.5 S_c and 1.5-2 % exceed 0.9 S_c (`02e` reported 12-28 % at 0.5 S_c for the v1 points), i.e. the nonlinearity
is active on GM at the measured D0, whereas on AP in metres it is not (section 4.6). The 04 figure `04_acrit_profiles.png` is regenerated with the five GM nonlinear
profiles drawn over the partitioned ones (the AP nonlinear profiles are in metres and are not drawn on the feet
profiles).

## 3. Phase 4b: the D-infinity partitioned model rebuilt

### 3.1 What was built

`lemcvt.models.dinf_partitioned.partitioned_threshold_model_dinf` (commit a4c922c) replaces the explicit
upstream integration of the v1 model by one sparse linear system over all valid cells, as `02d` 2.6.6 proposed and
Addendum 3 section 6 specifies:

- boundary cells (valid edge cells): `z_k = z_obs,k`;
- channel cells (`A_dinf >= A_crit`, plus the outlet; not boundary): the D-infinity upwind advection row of the AD
  D-infinity model with D = 0, divided by `K A_dinf^m`, `[cos r (z_k - z_1) + sin r (z_1 - z_2)] / dx = (U/K) A_dinf^-m`
  (`z_1` the cardinal receiver, `z_2` the diagonal one, `r` the facet angle); a channel cell with only one valid
  receiver gets that model's single-receiver stencil, and one with none gets a hillslope row and is counted;
- hillslope cells: the partitioned model's 5-point Laplacian row with K = 0, `lap z = -(U/D)`.

The matrix is free of K and D, so `Z = a + (U/K) b + (U/D) c` holds exactly (`f.reduce(A_crit)` returns the three
vectors from one LU) and the Phase 4 optimizer applies without change. The old model is kept verbatim as
`partitioned_threshold_model_dinf_v1`; `lemcvt closeout dinf` (the `02d` diagnostics) now names it explicitly. The
nonlinear D-infinity model still uses the v1 integration; the addendum names only the linear model, and no
reported number uses the nonlinear D-infinity model. Logged as N8 in `docs/numerics_changes.md`.

### 3.2 Tests (the gate; `tests/test_dinf_rebuild.py`, run in job 51780220)

1. **Aligned V-valley and plane** (GM-like m, U, K, D; A_crit = 500.5 m^2 so that no 1e-10 m^2 area difference can
   move the mask): the rebuilt D-infinity model and the D8 partitioned model agree to 8e-13 (V) and 3e-12 (plane)
   of relief at 201 x 201 cells (A2c; the pytest version at 101 x 101 passes the 1e-10 tolerance) with identical
   channel masks; no channel row lacks a receiver; no channel cell lies below every receiver of nonzero weight.
   **Pass.**
2. **Rotated V-valley (20 degrees)** against the D8 model on the same surface at 4x finer resolution (801 x 801,
   dx 0.25 m) sampled at the coincident coarse nodes, outer two rows and columns excluded (report only): rebuilt
   D-infinity max |dZ|/relief 1.06, rms 0.042; the coarse D8 model against the same reference max 0.026, rms
   0.0012 (A2c). Two readings. (i) D8 at 1 m and D8 at 0.25 m agree closely because both route the channel along
   the same grid-aligned staircase; the D-infinity model routes it along the facets, so the two differ wherever
   the staircase and the facet path part, and with the steep channel of this parameter set (S0 up to 0.55 along
   the axis) single cells differ by a relief. (ii) On this synthetic the v1 and the rebuilt models give the same
   surface (MSE against the input 174.0 for both; the fine-D8 comparison identical to 3 figures), because the
   D-infinity channel set of a straight valley is connected and every channel cell's receivers are channel cells:
   the v1 explicit step `z = z1 + (S0 dx - (z1 - z2) sin r) / cos r` is then exactly the coupled channel row, and
   the two models can only differ where channel cells have hillslope receivers, which is 88-92 % of the channel
   cells on the real tiles (`02d` 2.6.2) and none here. The synthetic therefore does not discriminate the two
   models; the tiles do (section 3.3).
3. **The Phase 4 reduction** holds for the new model: a + (U/K) b + (U/D) c against the direct solve at two A_crit
   and three random (K, D) each, below 1e-10 relative (the matrix is K- and D-free, so the identity is exact up
   to the LU's rounding). **Pass.**
4. **Channel monotonicity on AP1 and GM1** (full tiles, run `20261110T160122Z_d8_vs_dinf_AP1_GM1`, job 51785049,
   AP in feet): channel cells lower than every receiver with a nonzero weight: **0 of 5964 (AP1), 0 of 5845 (GM1)**
   (A2b). **Pass.** For the record, 668 (AP1) and 771 (GM1) channel cells lie below *one* of their two receivers
   (minimum drop -17 ft and -2.9 m): with two receivers of weights (cos r - sin r) and sin r the facet slope can
   be positive while one receiver is higher, which is what D-infinity routing means; the v1 model's failure mode
   (channel cells above the DEM maximum by a relief, 985 cells below the receiver used) does not occur.

### 3.3 `d8_vs_dinf` rerun (AP in feet, v1 parameters K0, D0 rounded to 2 s.f., A_crit = 5000 m^2; run `20261110T160122Z_d8_vs_dinf_AP1_GM1`, job 51785049, 64 GB because the v1 D-infinity AD model still assembles with LIL loops)

| tile (z unit) | MSE AD D8 | MSE AD D-inf | MSE PT D8 | **MSE PT D-inf rebuilt** | MSE PT D-inf v1 | v1 printed PT D-inf | channel cells D8 / D-inf | channel RMS error D8 / rebuilt / v1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 (ft) | 6.9581e5 | 7.0133e5 | 2195.0 | **7661.4** | 3.7650e6 | 3.7650e6 | 5657 / 6459 | 15.9 / 51.7 / 2356 |
| GM1 (m) | 3.2286e5 | 3.1845e5 | 792.04 | **1703.2** | 4239.5 | 4239.5 | 5318 / 6141 | 4.09 / 17.6 / 63.9 |

The AD rows and the D8 and v1 D-infinity partitioned rows reproduce the v1 printed values (the Phase 2 regression
once more, with the v1 model under its new name). The rebuilt model brings the partitioned D-infinity MSE down
by a factor 490 on AP1 and 2.5 on GM1, and the channel RMS error from 2356 to 52 ft and from 64 to 18 m, with no
channel cell above the DEM. It remains 3.5x (AP1) and 2.2x (GM1) above the D8 partitioned MSE at the same
parameters, and its channel RMS error 3-4x the D8 one. Measured: those ratios. Inferred, not tested here: whether
the remainder is routing (the D-infinity channel set at A_crit = 5000 m^2 has 14-15 % more cells than D8's and is
not a connected tree, `02d` 2.6.3, and the two-receiver row averages across facets at every junction) or the
parameters (measured K0, D0, not fitted; the v2 optimizer applies to the rebuilt model unchanged and would settle
it in Phase 6). The `d8_vs_dinf` figures (`reports/v2/figures_v2/d8_vs_dinf_*`) are regenerated from this run.

## 4. Phase 5: the AP vertical units corrected

### 4.1 Corrected grids (5.1)

The five AP tile configs now carry `reproduce_v1_units: false` (commit a4c922c): the loader converts z to metres
with the exact US survey foot, `z_m = z_ft x 1200/3937` (the single constant `FT_US_IN_M` of `lemcvt.io.units`;
the directive's 0.3048 is the international foot and differs by 2 ppm). `lemcvt ap-units convert` (run
`20261110T143257Z_ap_units_convert_5tiles`, job 51780220) wrote the corrected filled grids as
`AP/tile<n>/AP_1m_best_tile_filled_m.tif` under `$LEMCVT_DATA` (float64, CRS EPSG:26917 carried over) and added
them to `data/manifest.csv` as kind `input_metres` (275 entries now); the originals are untouched and the configs
keep pointing at them. Every written grid is bit-identical to the loader's in-memory conversion (A3). Metre
ranges 318-464 m, median pixel slope 0.267-0.271 (the feet grids gave 0.88, which the Phase 1 sanity check flags
at 0.6; the metre grids pass it).

**Parameters carry the vertical unit too** (addendum 4, section 3). The configs declare `model.param_z_units: ftUS`
and the loader multiplies K0_ad, K0_partitioned (U/k_s with the feet-based k_s), D0 (U/C_ht) and the K and D
bounds by c = 3937/1200 when the working unit is metres; the factor is recorded as `param_scale` in
`ModelParams` and therefore in `model_resolved` of every run's config snapshot. A `--units feet|metres|stored`
override (`lemcvt tile run`, `d8-vs-dinf`, and `load_tile(..., reproduce_v1_units=...)`) keeps feet runs possible
for the record without touching the configs; the Phase 4b rerun used it. The helpers that used to build metre
configs by hand (`measure`, `measure-diagnostics`, `audit picard`) now go through the same override, so their
model parameters are unit-consistent as well.

### 4.2 The D8 check (5.2; a check only, no grid replaced; run `20261110T160115Z_ap_d8_check_5tiles`, job 51785049)

`lemcvt ap-units d8-check` recomputes D8 directions from the corrected (metre) filled tile with the TopoAnalysis
port (`lemcvt.hydro.d8.d8_flow_direction`, including its 1.41 diagonal divisor and its slab convention) and
compares them cell by cell with the stored `*_fd`. Differing cells are classed as *edge-adjacent* (outer two
rows and columns: the stored grid was computed on the mosaic before clipping, and the port's slabs give the last
two rows and columns no east or south comparison), *tie* (the drop to the stored receiver equals the steepest
drop within max(1e-9 relative, 1e-9 m): the priority-flood aggradation increments on filled flats are 1e-12 dx,
so scaling z by a constant reorders them in the last bit), and *anything else*. The same recomputation on the
feet grid is reported beside it, so the unit change is separated from the recomputation itself (A4):

| tile | differing cells | edge-adjacent | ties on flats | anything else | metres-recomputed against feet-recomputed | feet-recomputed against stored, interior |
| --- | --- | --- | --- | --- | --- | --- |
| AP1 | 4592 | 3529 | 718 | **345** | 721 | **345** |
| AP2 | 4214 | 2429 | 1785 | 0 | 1789 | 0 |
| AP3 | 7107 | 3182 | 3925 | 0 | 3943 | 0 |
| AP4 | 5407 | 3419 | 1988 | 0 | 1994 | 0 |
| AP5 | 5829 | 3590 | 2239 | 0 | 2244 | 0 |

Away from the edges the metre and feet recomputations differ only on ties (700-3900 cells per tile, all on filled
flats where the aggradation increments are last-bit quantities), and the feet recomputation reproduces the
stored grid on every interior cell of AP2-AP5, as scaling by a constant must. (A first version of this check,
run in job 51780220, classified these ties as "anything else" because its neighbour offsets were mirrored; the
error was found from the example cells, fixed in commit 6380565, and the check rerun. The direct counts of the
first run were already those above.)

**AP1's 345 cells are a defect of the stored grid, not of the unit change.** They are the same 345 cells on which
the *feet* recomputation disagrees with the stored `*_fd`; all of them lie on rows 389 and 390 (0-based) and span
the tile from column 2 to 1010 in 121 small groups; the stored codes point west, north-west or east (16, 32, 1)
where the steepest descent is south or south-west (4, 8), with the steepest drop exceeding the drop to the stored
receiver by 0.03 m at the median and up to 0.27 m (A4, examples). That is the signature of the TopoAnalysis slab
convention applied at a block boundary: when the AP1 mosaic was processed in pieces, rows 389-390 of this tile
were the last two rows of a piece and received no south comparison, so the stored AP1 flow directions (and the
area accumulated through them) carry a processing seam across the tile. The seam is 0.03 % of the cells; its
effect on the area grid is confined to the cells draining through those two rows. I did not replace the grid
(the directive says not to) and did not quantify the area effect; it belongs in Phase 6's old-against-new record
and is listed in section 8.

### 4.3 Bounds (5.3)

Scaled by c in the loader as described above; the values are in A5 (AP1/3/5: K [1e-12, 2e-3] ft -> [3.28e-12,
6.56e-3] m, D [4e-10, 1e-2] -> [1.31e-9, 3.28e-2]; AP2/4: K [1e-12, 2e-2] -> [3.28e-12, 6.56e-2], D [4e-7, 5e-2] ->
[1.31e-6, 0.164]) and in each config as a comment next to the feet-based bounds. Every AP joint optimum of the
feet runs was interior to the box (`04`), and the scaled box contains c times the feet optimum by construction.

### 4.4 Measurement (5.4)

Nothing redone: `lemcvt measure`, the diagnostics and the audits have run on metre grids since Addendum 1.

### 4.5 Invariance check (5.5; runs `20261110T1947..Z_ap_invariance_<tile>_<stage>`, job 51805693)

`lemcvt ap-units invariance` loads the tile in metres and recomputes each v2 profile **at exactly the
evaluation points of the feet run**: the partitioned profile at the 2067 (AP1) and 2213 (AP5) A_crit values of the
refined feet runs (`04` V6), the AD profile at the 200 r values of the feet runs, and the exponent profile on
the same 200 x 51 (A_crit, m) grid; stage 2 uses the converted K0 and D0 (c x the feet values), the box the
converted bounds. Checks: optima and box positions identical, K and D equal c x the feet values, MSE equals the
feet MSE x 0.0929034, and the profile shapes (MSE/MSE_min) equal, all to 1e-8 relative (A6):

| tile | stage | points | max rel. diff: MSE x c^2 against feet | K against c K_ft | D against c D_ft | profile shape | optimum | box position identical | pass |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | partitioned (stage 2 / joint) | 2067 | 1.2e-13 / 7.3e-13 | 2.5e-13 | 2.9e-13 | 1.1e-13 / 1.1e-12 | stage 2 3064.97, joint 4931.13, both identical | yes | **yes** |
| AP1 | AD | 200 | 2.7e-9 | 3.2e-12 | 3.2e-12 | 2.7e-9 | r = 0.0137943, identical | yes | **yes** |
| AP1 | exponent profile | 10200 | 2.1e-12 | 2.4e-13 | 6.2e-13 | 2.3e-12 | (4894.06, 0.54), identical | yes | **yes** |
| AP5 | partitioned (stage 2 / joint) | 2213 | 4.6e-12 / 5.1e-12 | 4.2e-14 | 2.6e-13 | 3.9e-12 / 4.8e-12 | stage 2 1125.60, joint 1887.08, both identical | yes | **yes** |
| AP5 | AD | 200 | 1.0e-10 | 2.3e-13 | 2.3e-13 | 1.0e-10 | r = 0.368336, identical | yes | **yes** |
| AP5 | exponent profile | 10200 | 5.8e-12 | 8.0e-14 | 4.0e-13 | 5.8e-12 | (1854.76, 0.42), identical | yes | **yes** |

Every check passes with four to five orders of margin on the 1e-8 tolerance: the A_crit, m and r optima and
the box positions are identical point by point, the channel masks identical, K and D equal c times the feet
values to 1e-13, the MSE equals the feet MSE x 0.0929034116 (the joint-optimum ratio on AP5 is 0.09290341161330
against 1200^2/3937^2 = 0.09290341161332), and the profile shapes agree to 1e-12; the certificates at the v1
points scale the same way (gradients to 2e-10, best (K, D) to 1e-14). The AD stage's 2.7e-9 on AP1 is the one
value above 1e-11: its closed-form U/D at each r divides by a Gram scalar of the solve, and the SuperLU
factorisations of the metre and feet matrices (which differ by the scaling of the right-hand side only) round
differently at that level; it is still 4x inside the tolerance. The algebra of addendum 4 section 3 therefore
holds in the code as run, and the AP results of `04` convert exactly (A_crit, m, r unchanged; K, D x c; MSE x
0.0929). The loader's scaled parameters equal c times the feet runs' recorded parameters to the last bit. A first submission of this check (job 51785135) ran the three stages in one task per tile and
crashed, after its 3 h partitioned profiles, on a comparison line that read the `where` column through
attribute access (`DataFrame.where` is a method); nothing of it was written. Fixed (commit da01bfe), rerun as one
task per tile and stage.

### 4.6 AP nonlinear in metres (5.6; the first converged AP nonlinear result; runs `20261110T1604..Z_tile_run_<tile>_v2_nonlinear`, job 51785169)

`lemcvt tile run --optimizer v2 --stages nonlinear` on the five AP tiles in metres: K0 = c x U/k_s and D0 = c x
U/C_ht from the v1 measured values (the loader's scaling; AP1 K0 1.30e-6, D0 9.07e-3), S_c = 1.27, 200 log-thinned
unique areas plus 60 refinement values, every evaluation a Picard solve with the `02e` record (A7):

| tile | optimum A_crit (m^2) (Picard it.) | MSE (m^2) | 1 % interval (m^2) | evaluations hitting the cap | hillslope cells > 0.5 S_c / > 0.9 S_c at the optimum | max slope ratio | v1 feet A_crit, same code in metres: MSE (it.) | wall (h) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 4196.3 (7) | 140.35 | 4196-4395 | 0 of 261 | 0.054 / 0.0021 | 4.6 | 4950.3: 160.62 (7) | 3.9 |
| AP2 | 2502.2 (7) | 335.57 | 2362-2621 | 0 of 261 | 0.085 / 0.0047 | 8.5 | 6165.9: 512.35 (8) | 4.3 |
| AP3 | 1799.6 (8) | 123.84 | 1762-1822 | 0 of 261 | 0.073 / 0.0071 | 21.6 | 10233: 599.82 (9) | 4.5 |
| AP4 | 995.4 (6) | 52.72 | 952-1056 | 0 of 261 | 0.042 / 0.0059 | 13.6 | 5263.5: 479.01 (9) | 3.8 |
| AP5 | 1353.8 (7) | 31.74 | 1224-1387 | 0 of 261 | 0.036 / 0.0033 | 10.3 | 3399.1: 204.59 (9) | 4.1 |

- **Every evaluation converged**: 0 of 1305 hit the cap, in 6-12 Picard iterations (the feet runs of `02e` hit the
  cap on every AP tile; the per-evaluation cost fell from 330 s to 55-60 s on average, and the five profiles took
  3.8-4.5 h instead of the 24-36 h the feet runs needed).
- **The hillslopes are now mostly linear**: 4-9 % of hillslope cells above 0.5 S_c and 0.2-0.7 % above 0.9 S_c at
  the optimum (feet: 78-87 % and 21-29 %); the maximum slope ratio of 4.6-21.6 is a handful of cells at the
  channel margins.
- **The optima are not the v1 ones.** The converged metre optima lie at 995-4196 m^2 against v1's 3399-10233 m^2 in
  feet, and at the v1 A_crit the metre model's MSE is 1.1-4.7x the optimum. On AP1, AP3 and AP4 the metre
  nonlinear optimum sits on the same staircase step as, or within one step of, the linear stage-2 optimum at the
  same (K0, D0) (AP4 995.35 both; AP3 1799.6 against 1764.4; AP1 4196 against 3065), and its MSE is 1-15 % below
  the stage-2 MSE converted to metres (AP4 52.72 against 55.55; AP3 123.84 against 125.76; AP1 140.35 against
  178.02): with the gradients in consistent units the nonlinearity acts on a few per cent of the cells and the
  nonlinear stage mostly reproduces the linear one. In feet the nonlinearity acted on a quarter of the cells and
  moved A_crit by factors 1.6-6 (`00_reference.md` 3.2), which is what the v1 AP nonlinear numbers were.
- These still use the v1 measured k_s and C_ht (converted); Phase 6 replaces them. They are the first AP
  nonlinear numbers of this project that are a converged model, and they are not comparable with the v1 ones
  beyond the statement that the v1 ones were not converged.

## 5. Sections 6.1 and 6.2: the exponent profile on a wider grid, the site-level m, and the K recovery

### 5.1 Exponent profile, m = 0.05-0.90 (6.1.1; runs `20261110T1604..Z_tile_run_<tile>_v2_m_profile`, job 51785172)

`lemcvt tile run --optimizer v2 --stages m_profile --m-range 0.05,0.90` on all ten tiles, AP in metres (K0, D0 and the
bounds scaled by c), on the same 200 log-thinned unique A_crit values as Phase 4 (the thinning is deterministic
in the area grid) and 86 values of m; 17 200 exact (U/K, U/D) minima per tile, 1.1-1.2 h per tile (A8). The
fitted (A_crit, m) are those of `04` section 5 on every tile, AP now in metres (the invariance of section 4.5 in
action: AP1 MSE 20.543 m^2 = 221.124 ft^2 x 0.0929034): AP1 (4894, 0.54), AP2 (6919, 0.59), AP3 (5158, 0.39),
**AP4 (7831, 0.21)**, AP5 (1855, 0.42); GM1 (1135, 0.29), GM2 (3444, 0.29), GM3 (859, 0.29), GM4 (2718, 0.36),
GM5 (2247, 0.34). AP4's optimum, which sat on the old grid's edge at 0.20, is interior on the wide grid: the
profile MSE(m) is flat from 0.05 to 0.30 (within 1.6 % of its minimum at 0.21; 1 % contour 0.09-0.28, 5 % 0.05-0.38),
so AP4 constrains m from above only. Every other tile's 1 % contour is 0.02-0.07 wide in m and its profile rises
by a factor 1.3-2.3 at m = 0.05 and 1.4-11.7 at m = 0.90 (A8).

### 5.2 Site profile S(m) (6.1.2-6.1.3; `lemcvt.analyses.exponent_site`, figure `a4_exponent_site.png`)

S(m) = sum over the five tiles of MSE*_t(m)/MSE*_t,min, with MSE*_t(m) the tile's profile minimised over A_crit
and (K, D) at that m; its minimum is 5 when every tile prefers the same m. Intervals are the m ranges with
S <= S_min (1 + 1 %) and (1 + 5 %) (A8b):

| site | m_site | S_min | 1 % interval | 5 % interval | leave-one-out m_site (1 % interval) |
| --- | --- | --- | --- | --- | --- |
| AP | **0.47** | 5.36 | 0.44-0.49 | 0.41-0.53 | without AP1 0.44 (0.42-0.47); AP2 0.45 (0.42-0.47); AP3 0.48 (0.46-0.50); **AP4 0.47 (0.45-0.49)**; AP5 0.49 (0.46-0.53) |
| GM | **0.31** | 5.07 | 0.29-0.33 | 0.26-0.37 | without GM1 0.32 (0.30-0.35); GM2 0.32 (0.30-0.33); GM3 0.32 (0.30-0.34); GM4 0.31 (0.28-0.33); GM5 0.31 (0.28-0.33) |

Leaving AP4 out changes nothing (its flat profile contributes a near-constant to S over the interval); leaving
any one AP tile out moves m_site by at most 0.03, any GM tile by at most 0.01. S_min of 5.36 on AP says the five AP
tiles do not share one m: at m_site = 0.47 the tile profiles are 1.00 (AP3, AP5), 1.06 (AP4), 1.11 (AP1), 1.19
(AP2) of their own minima.

**Against the measured site-pooled theta(A_c)** (6.1.3; the audit's pooled and equal-weight estimators with their
95 % basin-bootstrap bands, interpolated in log A_c at the median fitted A_crit of the site, 5158 m^2 on AP and
2247 m^2 on GM; A8c, figure):

| site | estimator | measured theta at the median A_crit [95 %] (basins) | m_site | m_site inside the band | S(m) 1 % interval overlaps the band |
| --- | --- | --- | --- | --- | --- |
| AP | pooled | 0.48 [0.44, 0.54] (23) | 0.47 | yes | yes |
| AP | equal-weight | 0.47 [0.36, 0.53] (23) | 0.47 | yes | yes |
| GM | pooled | 0.34 [0.33, 0.37] (40) | 0.31 | **no** (0.02 below the band) | yes (0.33) |
| GM | equal-weight | 0.35 [0.31, 0.38] (40) | 0.31 | yes (at the band's edge) | yes |

Using the median A_crit of the tiles at m_site instead (AP 4651 m^2) changes the AP bands by 0.01 and GM not at
all. Overlap statements only, as the addendum directs: on AP the site exponent and its 1 % interval lie inside
both measured bands; on GM the site exponent lies 0.02 below the pooled band and at the lower edge of the
equal-weight band, and its 1 % interval reaches into both.

### 5.3 K recovery consistent in units (6.2; A9; no interpretation)

At the site-level m, per tile: K_fit is the joint-profile K at that fixed m (the m-profile row at m_site
minimised over A_crit; AP in metres), and k_s is the pooled through-origin chi-z slope at theta = m_site over
the tile's channel network above A_c = 10^3 and 10^4 m^2 (the audit's `ks_at_theta` curve, so the measurement
never sees the model's A_crit; basin filters as pre-registered in `02c`). U = 5.5e-5 (AP), 1.47e-4 (GM) m/yr.

| tile | m_site | K_fit (1/yr) | k_s at 10^3 m^2 (basins, n_eff) | U/k_s / K_fit at 10^3 | k_s at 10^4 m^2 (basins, n_eff) | U/k_s / K_fit at 10^4 |
| --- | --- | --- | --- | --- | --- | --- |
| AP1 | 0.47 | 3.99e-6 | 12.42 (32, 1.4) | 1.11 | 14.34 (5, 1.2) | 0.96 |
| AP2 | 0.47 | 4.60e-6 | 11.58 (13, 1.9) | 1.03 | 12.81 (2, 2.0) | 0.94 |
| AP3 | 0.47 | 4.80e-6 | 11.53 (32, 1.1) | 0.99 | 13.07 (2, 1.0) | 0.88 |
| AP4 | 0.47 | 4.52e-6 | 11.93 (30, 2.1) | 1.02 | 12.21 (5, 2.3) | 1.00 |
| AP5 | 0.47 | 5.11e-6 | 10.73 (9, 1.6) | 1.00 | 12.07 (2, 1.8) | 0.89 |
| GM1 | 0.31 | 4.87e-5 | 3.03 (19, 3.5) | 1.00 | 2.93 (8, 3.1) | 1.03 |
| GM2 | 0.31 | 5.51e-5 | 2.67 (16, 3.7) | 1.00 | 2.43 (5, 2.1) | 1.10 |
| GM3 | 0.31 | 6.81e-5 | 2.29 (15, 2.9) | 0.94 | 2.22 (7, 2.2) | 0.97 |
| GM4 | 0.31 | 3.64e-5 | 4.23 (14, 1.5) | 0.96 | 4.00 (3, 1.3) | 1.01 |
| GM5 | 0.31 | 5.27e-5 | 2.93 (4, 1.2) | 0.95 | 2.85 (2, 1.1) | 0.98 |

(k_s in m^(2 m_site); the site-pooled k_s at the same theta is 11.6 / 13.0 on AP and 2.91 / 2.83 on GM at 10^3 /
10^4 m^2. The n_eff values are those of `02e`: on AP and GM3-GM5 the tile k_s rests on one to two effective basins.)

## 6. Sections 6.3 and 6.4: curvature option (iii) and two fixes in the measurement code

### 6.1 Curvature: a third method option anchored externally (6.3; audit run `20261110T160633Z_measure_audit_10tiles`, job 51785562)

**The window L_f, chosen before any C_ht was computed** (`select_L_f` in `lemcvt.measure.audits`, recorded in the
run's `L_f_selection` before the per-tile loop). The `02c` synthetic ridge was run with a crest 380 m wide in a
400 m domain, so that no window of the standard set (3-61 m) sees the flanks and the only error left is the one
from TIN facets (2.76 m, the median AP facet from the diagnostics run) and noise (0.043 m, the GM roughness); the AP
variant is facets plus noise, the GM variant noise alone; c_true is each site's post-exclusion median of `02e`
(the bias is relative and depends weakly on it). The recovered curvature is the median over the ~400 crest pixels
of one realisation, and at small windows its sampling error is itself of order 1 % (A10a, "sd": 3-7 % at 7-9 m,
1-3 % at 11 m), so the rule is applied to the mean over 8 realisations. Result (A10a): the mean bias falls below 1 %
at both sites from **L_f = 11 m** on (AP -0.75 %, GM -0.34 %; at 9 m +0.5 % and +2.0 %), i.e. the half-width
w_f = 5 cells, which is 10.73 m on the AP tiles (dx 0.9752 m) and 11.0 m on GM; the addendum's reading of `02c`
("about 9 m") is one step below it. A first pass with a single realisation had chosen 21 m, because the GM
realisation at 9 and 11 m happened to sit at +4.5 % and +1.1 %; that result (run
`20261110T143502Z_measure_audit_10tiles`) is superseded and the averaging is what the code does now (commit
0bbf782).

**The hilltop set** is the one `measure` uses, the divides between outlet basins of the rollover network (A_c* =
18-450 m^2, `02b` table 2), two cells from the edge, |grad z| <= 0.4, with every pixel closer than L_f/2 (5.4 m on
AP, 5.5 m on GM) to a pixel of that network dropped. **Per tile** (A10b, option (iii) rows; C_ht at L_f with the
segment bootstrap, D0 = U/C_ht):

| tile | pixels kept (of the measure set) | segments | C_ht (1/m) [68 %] | D0 (m^2/yr) [68 %] | width 2d 25 / 50 % (m) | synthetic bias at median / 25 % width |
| --- | --- | --- | --- | --- | --- | --- |
| AP1 | 3445 (6104) | 505 | 1.53e-2 [1.43e-2, 1.62e-2] | 3.59e-3 [3.39e-3, 3.85e-3] | 19 / 36 | -0.3 % / -0.7 % |
| AP2 | 940 (3497) | 122 | 1.71e-2 [1.58e-2, 1.82e-2] | 3.22e-3 [3.03e-3, 3.47e-3] | 13 / 17 | +1.8 % / +1.6 % |
| AP3 | 531 (9409) | 61 | 2.55e-2 [2.21e-2, 2.88e-2] | 2.16e-3 [1.91e-3, 2.49e-3] | 12 / 12 | +0.8 % / -0.6 % |
| AP4 | 4703 (6136) | 579 | 1.52e-2 [1.43e-2, 1.65e-2] | 3.63e-3 [3.34e-3, 3.84e-3] | 20 / 44 | +0.2 % / -0.5 % |
| AP5 | 3538 (7107) | 346 | 1.88e-2 [1.66e-2, 2.06e-2] | 2.92e-3 [2.68e-3, 3.30e-3] | 15 / 19 | -1.0 % / +0.5 % |
| GM1 | 3577 (11112) | 507 | 1.63e-2 [1.57e-2, 1.72e-2] | 9.00e-3 [8.53e-3, 9.35e-3] | 13 / 15 | +0.5 % / -0.9 % |
| GM2 | 2511 (9187) | 295 | 1.51e-2 [1.41e-2, 1.60e-2] | 9.77e-3 [9.17e-3, 1.04e-2] | 13 / 16 | -0.5 % / -1.6 % |
| GM3 | 1500 (14030) | 190 | 1.45e-2 [1.35e-2, 1.54e-2] | 1.01e-2 [9.55e-3, 1.09e-2] | 12 / 13 | -0.6 % / +0.6 % |
| GM4 | 3344 (6225) | 373 | 1.49e-2 [1.35e-2, 1.66e-2] | 9.86e-3 [8.83e-3, 1.09e-2] | 15 / 20 | +0.6 % / +0.5 % |
| GM5 | 1115 (4964) | 144 | 1.94e-2 [1.85e-2, 1.99e-2] | 7.59e-3 [7.39e-3, 7.94e-3] | 13 / 15 | +0.5 % / -0.9 % |

The synthetic width bias is computed on the post-exclusion widths (2 x distance to the rollover network: medians
12-44 m) at L_f; it is within 2 % everywhere, which is the point of pairing a short window with the exclusion.
Beside the two `02e` options (A10b has every row with its definition, window and pixel count): option (iii) keeps
15-77 % of the measure set (option (i) against the rollover network at L* kept 0-40 % and nothing on AP3), and
gives C_ht 1.5-2.5e-2 on AP and 1.5-1.9e-2 on GM, against 3.0-8.9e-3 (AP) and 0.99-1.5e-2 (GM) for the measure
set without exclusion at L* and 9.6e-3-1.8e-2 (AP), 1.4-2.1e-2 (GM) with the L*/2 exclusion. So D0 by option
(iii) is 2.2-3.6e-3 m^2/yr on AP and 7.6-10e-3 on GM, 1.3-5x below the `02b` values on AP and 0.6-1.0x on GM.

What this does and does not establish. Measured: the numbers above, from one hilltop definition, one window
and one exclusion rule, all fixed before the computation. Inferred: that the 11 m window is unbiased on the real
tiles rests on the synthetic (parabolic crest, TIN facets of one size, Gaussian noise of one amplitude); the
measured C_ht(L) curves of `02b` keep falling out to 20-50 m on AP, which the synthetic's facets and noise do not
reproduce, so either the AP crests are narrower than the exclusion assumes or the AP surface has roughness at
scales the synthetic lacks; the synthetic cannot tell which. The option (iii) numbers are reported, not adopted
(decision 2 of section 8).

### 6.2 Two fixes in the measurement code (6.4; both logged in `docs/measurement_methods.md`)

**6.4.1 `hilltop_widths` uses the hilltop set `measure` uses.** `lemcvt.measure.diagnostics.hilltop_widths` built
its mask with `hilltop_mask` but without the outlet-basin relabelling that `measure_tile` applies, so it counted
every divide cell next to a cell draining to a different channel *pixel* (24 000-36 000 pixels per tile, `02e`
4.4). It now relabels channel pixels by their network outlet through a helper shared by `measure`, the
diagnostics and the audits (`lemcvt.measure.chi.channel_pixel_basin_lookup`, commit a4c922c), and the
diagnostics compute the widths on the rollover network (the set C_ht is measured on) as well as at 10^4 m^2.
`02c` and its T7 are left as the record of what was run then; the diagnostics were rerun once with the fix
(run `20261110T161031Z_measure_diagnostics_10tiles`, job 51785562) so that the corrected width distributions
exist beside the old ones. At 10^4 m^2 the set shrinks from 24 000-36 000 to 690-5100 pixels per tile and its
median crest width (2 x distance to the network) grows from 28-50 m to 70-297 m on AP and from 31-102 m to
130-251 m on GM, which is the `02e` A6 "1e4, all" set exactly (AP1 861 pixels, GM1 4128 in both runs); on the
rollover network, the set `measure` uses, the widths are 4-14 m (AP) and 4-12 m (GM) at the median. The rerun's
theta ladders are unchanged (pooled AP theta at 1e3 m^2 0.39 [0.37, 0.41] in both runs; the theta grid's new lower
edge does not reach the pooled minima).

**6.4.2 The theta grid runs from 0.00.** `DEFAULT_THETAS` in `lemcvt.measure.concavity` is 0.00-0.90 by 0.01
(was 0.10-0.90), and every ladder step of the audit reports the fraction of kept basins whose own best theta
sits at the lower edge (`frac_basins_theta_at_lower_edge`; A11). The `02e` free ladder was rerun with the pooled,
equal-weight and median estimators, site pooling, n_eff and the basin bootstrap (run
`20261110T160633Z_measure_audit_10tiles`, 4 min for ten tiles). What changed and what did not (A11):

- **The degenerate fits are now counted.** At A_c <= 2e3 m^2, 20-66 % of the kept basins on every tile have their
  best theta at 0.00 (chi proportional to distance); the fraction falls to 0-20 % above 4e3 m^2 and is 0 wherever
  only the one or two large networks remain. Site-pooled: AP 0.40, 0.29, 0.42, 0.39, 0.32, 0.19 at 1e2, 4e2, 1e3,
  2e3, 4e3, 1e4 m^2; GM 0.54, 0.42, 0.22, 0.19, 0.06, 0.04.
- **The pooled estimator does not move** (its minimum was never at the edge): site-pooled AP 0.32 -> 0.52 from 1e2
  to 1e4 m^2 and GM 0.27 -> 0.31, as in `02c`/`02e`. The pooled bootstrap intervals on the AP tiles now reach 0.00
  instead of 0.10 (AP1, AP3, AP5), which is the same symptom (n_eff 1-2) with the censoring removed.
- **The equal-weight site values are unchanged within 0.01** at every step (AP 0.12, 0.22, 0.29, 0.36, 0.45, 0.53;
  GM 0.11, 0.22, 0.30, 0.34, 0.34, 0.29); per tile they fall by up to 0.05 at 1e2-4e2 m^2 where many basins sit at
  the edge (AP1 0.10 -> 0.05 at 1e2, AP3 0.10 -> 0.08).
- **The median estimator falls to 0.00-0.05 at small A_c** where it was censored at 0.10 (AP pooled 0.05 at 1e2
  m^2, GM pooled 0.00 at 1e2 and 4e2 m^2; per tile 0.00 on AP1, GM2, GM4, GM5 at 1e2 m^2), and is unchanged above
  ~2e3 m^2 (AP pooled 0.43-0.56 from 4e3 m^2 on; GM 0.36-0.42 at 2e3-4e3 m^2).

So the `02e` reading stands with one correction of emphasis: below ~2e3 m^2 (AP) and ~1e3 m^2 (GM) the equal-weight
and median estimators are governed by basins whose chi-z relation is best fitted with no concavity at all, and
the number of such basins is now a reported quantity rather than a grid edge.

## 7. What the evidence supports, and where it is weak

Measured first, then inferred, then weak, per item.

- **Nonlinear rows.** Measured: the AP5 baseline row and its trajectory identity; the five GM profiles with their
  cap counts. Inferred: that the GM cap hits at large A_crit do not touch the optima (they are 1-2 decades away).
  Weak: GM1's and GM3's runs predate the slope-fraction record, so their hillslope statistics are missing.
- **Phase 4b.** Measured: the rebuilt model equals the D8 model on aligned surfaces to 1e-12 of relief, satisfies the
  reduction to 1e-10, has no channel cell below every receiver on AP1 and GM1, and lowers the partitioned
  D-infinity MSE from 3.77e6 to 7661 ft^2 (AP1) and from 4240 to 1703 m^2 (GM1) at the v1 parameters. Inferred:
  that the remaining 2-4x gap to D8 is a property of D-infinity routing in a threshold model (non-connected
  channel set, receiver mixing at junctions) rather than of the measured parameters; the v2 optimizer on the
  rebuilt model would decide it. Weak: the rotated-V synthetic cannot discriminate the rebuilt from the v1 model
  (connected channel set), so the only evidence that the rebuild changes anything comes from the two tiles.
- **Phase 5.** Measured: the corrected grids are bit-identical to the loader's conversion; the parameters scale by
  c exactly (loader against the feet runs to 1e-16); the invariance of the three profiles at the feet runs'
  evaluation points (identical optima, K and D x c to 1e-13, MSE x 0.0929034 to 1e-11, AD to 2.7e-9); the D8 recomputation reproduces the stored directions on every interior cell
  of AP2-AP5 and differs on AP1's rows 389-390 in feet and in metres alike; the converged AP nonlinear profiles.
  Inferred: the AP1 seam is TopoAnalysis's slab convention at a processing-block boundary of the mosaic (the
  geometry and the direction codes fit nothing else I know of; the mosaic is not on Sherlock to confirm it).
  Weak: the AP nonlinear optima still carry the v1 measured k_s and C_ht and say nothing about fit quality in
  Phase 6 terms; the invariance test is exact for the linear models only, and the nonlinear model's unit
  dependence is the `02e` result, not something this check can certify.
- **Sections 6.1-6.2.** Measured: the ten profiles, S(m), its intervals and the leave-one-out values; the measured
  bands at the median fitted A_crit; the k_s(theta) curves and the ratios. Inferred: nothing beyond the overlap
  statements, as directed. Weak: the AP site profile sums tiles whose own optima are 0.21-0.59 (S_min 5.36), so
  m_site = 0.47 is a compromise, not a shared value, and AP4 constrains it only from above; the measured bands on
  AP at 5158 m^2 rest on 23 basins with n_eff ~ 7 (`02e`), and the per-tile k_s in the recovery ratios rest on
  n_eff 1-2 (AP, GM3-GM5), so the ratios carry the tile's one or two large networks; the recovery is at the site
  m, which is not any tile's own optimum.
- **Section 6.3.** Measured: L_f from the synthetic under the stated rule, and C_ht, D0, pixel counts and widths
  by option with their bootstraps. Inferred: that the 11 m window is unbiased on the real tiles. Weak: the
  synthetic's facets and noise do not reproduce the measured C_ht(L) decline out to 20-50 m on AP, so the AP
  option-(iii) values (D0 2.2-3.6e-3 m^2/yr, 1.3-5x below `02b`) depend on an assumption the data contradict in
  part; AP3 keeps 531 pixels in 61 segments.
- **Section 6.4.** Measured: the set sizes and widths before and after the relabelling; the lower-edge fractions
  and the three estimators on the extended grid. Inferred: that the small-A_c equal-weight and median values are
  statements about degenerate fits. Weak: nothing new; the n_eff limitation of `02e` stands.
- **Decision 1 of the addendum (N7 not reverted)** is recorded, not tested further; the AD results in every report
  from `04` on are v2 profile optima.

## 8. Decisions needed

Phase 6 waits for these (addendum 4, section 7).

1. **The threshold rule for theta and k_s, or the exponent m.** The evidence in sections 5 and 6: the site
   exponents fitted within the model (AP 0.47, 1 % interval 0.44-0.49; GM 0.31, 0.29-0.33) sit inside the measured
   site-pooled theta bands at the median fitted A_crit on AP and 0.02 below the pooled band on GM; at those
   exponents U/k_s recovers the fitted K to within 12 % on every tile at both 10^3 and 10^4 m^2 (section 5.3);
   the per-tile optima spread from 0.21 to 0.59 on AP; the measured theta(A_c) curves still rise with A_c on AP
   (0.32 -> 0.52 by the pooled estimator) and 20-66 % of the basins at A_c <= 2e3 m^2 fit best with no concavity.
   The options remain those of `02c` section 10 (a fixed A_c of 1e3 or 1e4 m^2, the site break, the rollover
   network) plus the fitted exponent (site-level m from S(m), section 5.2). What goes into the tile configs for
   Phase 6 is the decision: m, and K0 (U/k_s at that m and at which A_c).
2. **The curvature method and D0.** Three options are now side by side with their hilltop definitions, windows
   and pixel counts (A10b): the fixed plateau window L* without exclusion (`02b`/`02c`), L* with the L*/2
   exclusion against each network (`02e`, option i), and the externally anchored fixed window L_f with the L_f/2
   exclusion against the rollover network (option iii, section 6.1). The options differ by a factor 2-5 in D0 on
   AP (option (iii) 2.2-3.6e-3 m^2/yr; the measure set at L* 6-18e-3; option (i) on the 10^4 network 6-9e-3) and by
   1.3-1.6 on GM (7.6-10e-3 against 9-24e-3 and 9-13e-3). Which hilltop set and which anchoring (the plateau of
   the measured C_ht(L), or a synthetic-bias threshold at a short window) represent "the hilltop curvature the
   models' D0 should match" is the decision; the limitation of section 6.1 (the AP C_ht(L) decline the synthetic
   does not reproduce) bears on it.
3. **The Phase 6 run list.** Section 7's items in the directive, on corrected grids with v2, plus: whether AP3's
   partitioned stages use U/k_s or the as-run K = 1e-6 (Addendum 2 decided as-run for regression and U/k_s for
   the rerun); whether `d8_vs_dinf` and the resolution sweep use the rebuilt D-infinity partitioned model at
   fitted rather than measured parameters (`00_reference.md` 3.5 item 3); and whether the nonlinear stage's D0
   is re-measured before the AP nonlinear rerun (section 4.6 still uses the v1 measured parameters).
4. **The AP1 flow-direction seam** (section 4.2): two rows of the stored `*_fd` carry directions that are not
   the steepest descent, and the stored area grid accumulated through them. Options: keep the stored grids
   (Phase 6 then inherits the seam, and the old-against-new record says so); or recompute AP1's D8 and area
   grids from the mosaic, which is not on Sherlock (`02c` section 8). Recomputing on the 1 km tile is not an
   option (section 5.2 of the addendum).
5. **Housekeeping.** Whether the Phase 3 AD stage should restore the scalar power (N7) for bit identity on Genoa
   nodes is moot under decision 1 of the addendum (recorded in `docs/numerics_changes.md`); nothing to decide.
   The GM nonlinear profiles hit the Picard cap at large A_crit (section 2); whether the cap should be raised for
   Phase 6's nonlinear stage, or the profile range cut at 5e4 m^2, is a run-list detail.

## 9. Appendices (generated tables)

Generated by `reports/v2/scripts/make_checkpoint_4b_5_tables.py` on 2026-11-10 15:35 PST from the runs named in each table.

## A1. Nonlinear rows: GM v2 profiles (decision 2) and the AP5 baseline (03b row filled)

| tile | run (job) | z unit | grid + refine | evaluations hitting the Picard cap | v2 optimum A_crit (Picard it.) | MSE | v1 A_crit same code: MSE (it.) | v2 - v1 (rel.) | 1 % interval (m^2) | hillslope > 0.5 / > 0.9 S_c at the optimum | wall (h) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | `20261110T160413Z_tile_run_AP1_v2_nonlinear` (51785421) | m | 200 + 60 | 0 of 261 | 4196.26 (7) | 140.354 | 4950.3: 160.622 (7) | -1.26e-01 | 4196 - 4395 (42 pts) | 0.054 / 0.0021 | 3.9 |
| AP2 | `20261110T160413Z_tile_run_AP2_v2_nonlinear` (51785422) | m | 200 + 60 | 0 of 261 | 2502.17 (7) | 335.57 | 6165.9: 512.347 (8) | -3.45e-01 | 2362 - 2621 (64 pts) | 0.085 / 0.0047 | 4.3 |
| AP3 | `20261110T160421Z_tile_run_AP3_v2_nonlinear` (51785423) | m | 200 + 60 | 0 of 261 | 1799.62 (8) | 123.839 | 10233: 599.816 (9) | -7.94e-01 | 1762 - 1822 (31 pts) | 0.073 / 0.0071 | 4.5 |
| AP4 | `20261110T160413Z_tile_run_AP4_v2_nonlinear` (51785424) | m | 200 + 60 | 0 of 261 | 995.353 (6) | 52.7241 | 5263.5: 479.013 (9) | -8.90e-01 | 951.6 - 1056 (64 pts) | 0.042 / 0.0059 | 3.8 |
| AP5 | `20261110T160421Z_tile_run_AP5_v2_nonlinear` (51785169) | m | 200 + 60 | 0 of 261 | 1353.76 (7) | 31.7442 | 3399.1: 204.591 (9) | -8.45e-01 | 1224 - 1387 (52 pts) | 0.036 / 0.0033 | 4.1 |
| GM1 | `20261109T231303Z_tile_run_GM1_v2_nonlinear` (51708241) | m | 200 + 32 | 3 of 233 | 475 (10) | 77.0733 | 474.21: 77.0733 (10) | +0.00e+00 | 460 - 548 (38 pts) | - / - | 5.7 |
| GM2 | `20261110T141003Z_tile_run_GM2_v2_nonlinear` (51779153) | m | 200 + 15 | 8 of 216 | 245 (8) | 47.5334 | 244.59: 47.5334 (8) | +0.00e+00 | 207 - 264 (22 pts) | 0.120 / 0.0197 | 7.7 |
| GM3 | `20261109T231258Z_tile_run_GM3_v2_nonlinear` (51708446) | m | 200 + 60 | 0 of 261 | 1298 (6) | 58.9848 | 1297.4: 58.9848 (6) | +0.00e+00 | 1257 - 1395 (64 pts) | - / - | 3.8 |
| GM4 | `20261110T141003Z_tile_run_GM4_v2_nonlinear` (51779154) | m | 200 + 33 | 15 of 234 | 525 (11) | 72.7622 | 521.38: 72.7734 (11) | -1.54e-04 | 454 - 578 (41 pts) | 0.277 / 0.0169 | 9.3 |
| GM5 | `20261110T141003Z_tile_run_GM5_v2_nonlinear` (51779145) | m | 200 + 18 | 0 of 219 | 300 (10) | 54.813 | 299.62: 54.813 (10) | +0.00e+00 | 261 - 393 (30 pts) | 0.153 / 0.0155 | 6.4 |

AP5 nonlinear baseline (v1 optimizer, job 51697696, run `20261109T211106Z_tile_run_AP5`): A_crit 3399.07 against v1 3399.1 (-8.1e-06), MSE 261.377 against 261.38 (-1.1e-05), 29 evaluations / 26 solves, 15.0 h, Picard iterations at the optimum 50 (the cap), message 'Optimization terminated successfully.'.

## A2. Phase 4b: D8 against D-infinity at the v1 `d8_vs_dinf` parameters (AP in feet), rebuilt and v1 partitioned D-infinity models

Run `20261110T160122Z_d8_vs_dinf_AP1_GM1` (job 51785049, git 63805657, units `feet`).

| tile | z unit | MSE AD D8 | MSE AD D-inf | MSE PT D8 | MSE PT D-inf rebuilt | MSE PT D-inf v1 | v1 printed (AD D8 / AD D-inf / PT D8 / PT D-inf) | channel cells D8 / D-inf rebuilt / D-inf v1 | channel RMS error D8 / rebuilt / v1 |
|---|---|---|---|---|---|---|---|---|---|
| AP1 | ftUS | 6.9581e+05 | 7.0133e+05 | 2195 | 7661.4 | 3.765e+06 | 6.9581e+05 / 7.0133e+05 / 2195 / 3.765e+06 | 5657 / 6459 / 6459 | 15.86 / 51.66 / 2356 |
| GM1 | m | 3.2286e+05 | 3.1845e+05 | 792.04 | 1703.2 | 4239.5 | 3.2286e+05 / 3.1845e+05 / 792.04 / 4239.5 | 5318 / 6141 / 6141 | 4.09 / 17.55 / 63.93 |

### A2b. Rebuilt model: system composition and channel monotonicity (test 4)

| tile | boundary rows | channel rows (both / cardinal-only / diagonal-only receivers) | channel cells without a usable receiver (hillslope row) | hillslope rows | channel cells lower than every nonzero-weight receiver | lower than any | min / median drop to receiver |
|---|---|---|---|---|---|---|---|
| AP1 | 4056 | 5964 (5964 / 0 / 0) | 0 | 1020205 | **0** | 668 | -17.4 / 0.352 |
| GM1 | 3996 | 5845 (5845 / 0 / 0) | 0 | 990159 | **0** | 771 | -2.91 / 0.11 |

### A2c. Synthetic cases (201 x 201 cells, dx 1.0 m, A_crit 500.5 m^2, GM-like m, U, K, D)

| case | relief | channel cells D8 / D-inf | mask differs | max dZ(rebuilt - D8)/relief | max dZ(v1 - D8)/relief | monotonicity violations (rebuilt) | MSE vs input D8 / rebuilt / v1 |
|---|---|---|---|---|---|---|---|
| plane_south | 20 | 1 / 1 | 0 | 2.67e-12 | 0.00e+00 | 0 | 582.8 / 582.8 / 582.8 |
| v_equal_slopes | 30 | 178 / 178 | 0 | 8.36e-13 | 2.84e-15 | 0 | 129.9 / 129.9 / 129.9 |
| v_rotated_20deg | 17.3 | 198 / 186 | 12 | 1.08e+00 | 1.08e+00 | 0 | 163.3 / 174 / 174 |

Rotated V against the D8 model at 4x finer resolution (801 x 801, dx 0.25 m; fine D8 solution at the coincident coarse nodes, outer two rows/columns excluded): max |dZ|/relief rebuilt 1.06, D8 coarse 0.0264, v1 1.06; rms/relief rebuilt 0.0421, D8 coarse 0.00117, v1 0.0421.

## A3. Phase 5.1: corrected AP grids written and added to the manifest

Run `20261110T143257Z_ap_units_convert_5tiles` (job 51780220, git a4c922c9). Factor 0.3048006096012192 m per US survey foot.

| tile | written | sha256 (12) | bytes | z range stored (ft) | z range (m) | bit-identical to the loader's conversion | median slope (m/m) |
|---|---|---|---|---|---|---|---|
| AP1 | `AP_1m_best_tile_filled_m.tif` | f82fad64c2d9 | 9086441 | 1064.75 - 1440.67 | 324.54 - 439.12 | True | 0.269 |
| AP2 | `AP_1m_best_tile_filled_m.tif` | 6ce76e353910 | 9207354 | 1053.62 - 1520.91 | 321.14 - 463.57 | True | 0.267 |
| AP3 | `AP_1m_best_tile_filled_m.tif` | 0d06c786e35c | 9099748 | 1076.95 - 1486.22 | 328.26 - 453.00 | True | 0.270 |
| AP4 | `AP_1m_best_tile_filled_m.tif` | bdc4afe7c065 | 9169664 | 1043.21 - 1456.01 | 317.97 - 443.79 | True | 0.271 |
| AP5 | `AP_1m_best_tile_filled_m.tif` | 432a9cbb9df0 | 9043293 | 1088.94 - 1395.45 | 331.91 - 425.34 | True | 0.270 |

Manifest: 275 entries; added `AP/tile1/AP_1m_best_tile_filled_m.tif`, `AP/tile2/AP_1m_best_tile_filled_m.tif`, `AP/tile3/AP_1m_best_tile_filled_m.tif`, `AP/tile4/AP_1m_best_tile_filled_m.tif`, `AP/tile5/AP_1m_best_tile_filled_m.tif`.

## A4. Phase 5.2: D8 directions recomputed from the corrected filled tile against the original `*_fd` (check only)

Run `20261110T160115Z_ap_d8_check_5tiles` (job 51785049). Tie tolerance 1e-09 relative on the drop.

| tile | cells | differing cells (new metres vs original) | edge-adjacent (outer 2 rows/cols) | of which outer 1 | ties (equal steepest drop) | anything else | interior cells | recomputed metres vs recomputed feet | recomputed feet vs original (interior) | original code 0 / new code 0 |
|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | 1030225 | 4592 | 3529 | 3265 | 718 | **345** | 1022121 | 721 | 345 | 0 / 0 |
| AP2 | 1030225 | 4214 | 2429 | 1936 | 1785 | **0** | 1022121 | 1789 | 0 | 0 / 0 |
| AP3 | 1029210 | 7107 | 3182 | 2281 | 3925 | **0** | 1021110 | 3943 | 0 | 0 / 0 |
| AP4 | 1030225 | 5407 | 3419 | 2647 | 1988 | **0** | 1022121 | 1994 | 0 | 0 / 0 |
| AP5 | 1030225 | 5829 | 3590 | 2211 | 2239 | **0** | 1022121 | 2244 | 0 | 0 / 0 |

Examples of the 'anything else' class (first 20 per tile; drops in metres with TopoAnalysis's 1.41 diagonal divisor):

| tile | row, col | fd original | fd new (m) | fd new (ft) | drop to original receiver | best drop | NaN neighbours |
|---|---|---|---|---|---|---|---|
| AP1 | 389, 2 | 16 | 8 | 8 | 0.273137 | 0.284463 | 0 |
| AP1 | 389, 3 | 16 | 8 | 8 | 0.280467 | 0.284093 | 0 |
| AP1 | 389, 4 | 16 | 8 | 8 | 0.18864 | 0.224034 | 0 |
| AP1 | 389, 7 | 32 | 8 | 8 | 0.24467 | 0.25723 | 0 |
| AP1 | 389, 13 | 8 | 4 | 4 | 0.367136 | 0.372071 | 0 |
| AP1 | 389, 17 | 16 | 8 | 8 | 0.309712 | 0.350538 | 0 |
| AP1 | 389, 18 | 16 | 8 | 8 | 0.299592 | 0.393762 | 0 |
| AP1 | 389, 22 | 1 | 8 | 8 | 0.0203151 | 0.0782141 | 0 |
| AP1 | 389, 25 | 16 | 4 | 4 | 0.324967 | 0.336687 | 0 |
| AP1 | 389, 26 | 8 | 4 | 4 | 0.474087 | 0.539317 | 0 |
| AP1 | 389, 35 | 16 | 8 | 8 | 0.491952 | 0.492506 | 0 |
| AP1 | 389, 38 | 16 | 8 | 8 | 0.262161 | 0.306655 | 0 |
| AP1 | 389, 39 | 16 | 8 | 8 | 0.26324 | 0.304755 | 0 |
| AP1 | 389, 40 | 16 | 8 | 8 | 0.269565 | 0.317685 | 0 |
| AP1 | 389, 41 | 16 | 8 | 8 | 0.275035 | 0.335919 | 0 |
| AP1 | 389, 42 | 16 | 8 | 8 | 0.31667 | 0.386426 | 0 |
| AP1 | 389, 57 | 16 | 8 | 8 | 0.264319 | 0.28222 | 0 |
| AP1 | 389, 58 | 16 | 8 | 8 | 0.253008 | 0.295097 | 0 |
| AP1 | 389, 59 | 16 | 8 | 8 | 0.246125 | 0.302301 | 0 |
| AP1 | 389, 65 | 8 | 4 | 4 | 0.299451 | 0.304875 | 0 |

## A5. Phase 5.3: AP parameter scaling by c = 3937/1200 (loader, `model.param_z_units: ftUS`)

| tile | K0_ad ft -> m | K0_partitioned ft -> m | D0 ft -> m | K bounds ft -> m | D bounds ft -> m | param_scale |
|---|---|---|---|---|---|---|
| AP1 | 1e-06 -> 3.281e-06 | 3.97e-07 -> 1.302e-06 | 0.002764 -> 0.009068 | [1e-12, 0.002] -> [3.28e-12, 0.00656] | [4e-10, 0.01] -> [1.31e-09, 0.0328] | 3.2808333333 |
| AP2 | 2.881e-07 -> 9.451e-07 | 2.881e-07 -> 9.451e-07 | 0.002829 -> 0.009282 | [1e-12, 0.02] -> [3.28e-12, 0.0656] | [4e-07, 0.05] -> [1.31e-06, 0.164] | 3.2808333333 |
| AP3 | 1e-06 -> 3.281e-06 | 1e-06 -> 3.281e-06 | 0.00267 -> 0.00876 | [1e-12, 0.002] -> [3.28e-12, 0.00656] | [4e-10, 0.01] -> [1.31e-09, 0.0328] | 3.2808333333 |
| AP4 | 1.971e-06 -> 6.466e-06 | 1.971e-06 -> 6.466e-06 | 0.002507 -> 0.008225 | [1e-12, 0.02] -> [3.28e-12, 0.0656] | [4e-07, 0.05] -> [1.31e-06, 0.164] | 3.2808333333 |
| AP5 | 1e-05 -> 3.281e-05 | 3.556e-06 -> 1.167e-05 | 0.002501 -> 0.008206 | [1e-12, 0.002] -> [3.28e-12, 0.00656] | [4e-10, 0.01] -> [1.31e-09, 0.0328] | 3.2808333333 |

## A6. Phase 5.5: invariance of the v2 profiles under the unit change (metre run at the feet runs' evaluation points; tolerance 1e-8 relative)

| tile | stage | feet run | points | max rel. diff: MSE x c^2 vs feet | K vs c K_ft | D vs c D_ft | profile shape | optimum identical (A_crit / r / (A_crit, m)) | box position identical | pass |
|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | partitioned (run `20261110T194710Z_ap_invariance_AP1_partitioned`, job 51805700) | `20261110T023535Z_tile_run_AP1_v2_partitioned-refined` | 2067 | 7.3e-13 | 2.5e-13 | 2.9e-13 | 1.1e-12 | stage 2 3064.97 / 3064.97; joint 4931.13 / 4931.13 (same) | True | yes |
| AP1 | ad (run `20261110T194720Z_ap_invariance_AP1_ad`, job 51806057) | `20261109T230932Z_tile_run_AP1_v2_ad` | 200 | 2.7e-09 | 3.2e-12 | 3.2e-12 | 2.7e-09 | r 0.0137943 / 0.0137943 (same) | True | yes |
| AP1 | m_profile (run `20261110T194710Z_ap_invariance_AP1_m_profile`, job 51806059) | `20261109T231309Z_tile_run_AP1_v2_m_profile` | 10200 | 2.1e-12 | 2.4e-13 | 6.2e-13 | 2.3e-12 | (4894.06, 0.54) / (4894.06, 0.54) (same) | True | yes |
| AP5 | partitioned (run `20261110T194720Z_ap_invariance_AP5_partitioned`, job 51805701) | `20261110T013458Z_tile_run_AP5_v2_partitioned-refined` | 2213 | 5.1e-12 | 4.2e-14 | 2.6e-13 | 4.8e-12 | stage 2 1125.6 / 1125.6; joint 1887.08 / 1887.08 (same) | True | yes |
| AP5 | ad (run `20261110T194720Z_ap_invariance_AP5_ad`, job 51806058) | `20261109T230934Z_tile_run_AP5_v2_ad` | 200 | 1.0e-10 | 2.3e-13 | 2.3e-13 | 1.0e-10 | r 0.368336 / 0.368336 (same) | True | yes |
| AP5 | m_profile (run `20261110T194710Z_ap_invariance_AP5_m_profile`, job 51805693) | `20261109T231313Z_tile_run_AP5_v2_m_profile` | 10200 | 5.8e-12 | 8.0e-14 | 4.0e-13 | 5.8e-12 | (1854.76, 0.42) / (1854.76, 0.42) (same) | True | yes |

Model scaling check (loader values against c x the feet run's values, relative): AP1: K0_partitioned 0.0e+00, D0 0.0e+00, K_lo 0.0e+00, K_hi 0.0e+00, D_lo 0.0e+00, D_hi 0.0e+00; AP5: K0_partitioned 0.0e+00, D0 0.0e+00, K_lo 0.0e+00, K_hi 0.0e+00, D_lo 0.0e+00, D_hi 0.0e+00

## A7. Phase 5.6: AP nonlinear v2 profile in metres (K0, D0 = c x v1 values, S_c = 1.27)

| tile | run (job) | K0, D0 (m) | grid + refine | evaluations hitting the cap | optimum A_crit (Picard it.) | MSE (m^2) | 1 % interval (m^2) | hillslope > 0.5 S_c / > 0.9 S_c / max ratio at the optimum | v1 feet A_crit (same code in metres: MSE, it.) | wall (h) |
|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | `20261110T160413Z_tile_run_AP1_v2_nonlinear` (51785421) | 1.302e-06, 0.009068 | 200 + 60 | **0** of 261 | 4196.26 (7) | 140.354 | 4196 - 4395 (42 pts) | 0.054 / 0.0021 / 4.62 | 4950.3: 160.622 (7) | 3.9 |
| AP2 | `20261110T160413Z_tile_run_AP2_v2_nonlinear` (51785422) | 9.451e-07, 0.009282 | 200 + 60 | **0** of 261 | 2502.17 (7) | 335.57 | 2362 - 2621 (64 pts) | 0.085 / 0.0047 / 8.54 | 6165.9: 512.347 (8) | 4.3 |
| AP3 | `20261110T160421Z_tile_run_AP3_v2_nonlinear` (51785423) | 3.281e-06, 0.00876 | 200 + 60 | **0** of 261 | 1799.62 (8) | 123.839 | 1762 - 1822 (31 pts) | 0.073 / 0.0071 / 21.55 | 10233: 599.816 (9) | 4.5 |
| AP4 | `20261110T160413Z_tile_run_AP4_v2_nonlinear` (51785424) | 6.466e-06, 0.008225 | 200 + 60 | **0** of 261 | 995.353 (6) | 52.7241 | 951.6 - 1056 (64 pts) | 0.042 / 0.0059 / 13.62 | 5263.5: 479.013 (9) | 3.8 |
| AP5 | `20261110T160421Z_tile_run_AP5_v2_nonlinear` (51785169) | 1.167e-05, 0.008206 | 200 + 60 | **0** of 261 | 1353.76 (7) | 31.7442 | 1224 - 1387 (52 pts) | 0.036 / 0.0033 / 10.32 | 3399.1: 204.591 (9) | 4.1 |

## A8. Section 6.1: exponent profile on m = 0.05-0.90, AP in metres, and the site profile S(m)

| tile | run (job) | z unit | m grid | fitted (A_crit, m) | K, D at the fit | MSE at fit / at config m | 1 % contour m | 5 % contour m | MSE(m)/min at m = 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90 |
|---|---|---|---|---|---|---|---|---|---|
| AP1 | `20261110T160413Z_tile_run_AP1_v2_m_profile` (51785412) | m | 0.05-0.90 x 200 A_crit | (4894, 0.54) | 1.74e-06, 0.01681 (interior) | 20.543 / 20.905 | 0.53-0.55 | 0.49-0.57 | 2.307, 2.179, 1.901, 1.625, 1.316, 1.031, 1.123, 1.991, 3.039, 3.391 |
| AP2 | `20261110T160413Z_tile_run_AP2_v2_m_profile` (51785413) | m | 0.05-0.90 x 200 A_crit | (6919, 0.59) | 1.182e-06, 0.02224 (interior) | 21.466 / 21.466 | 0.56-0.62 | 0.47-0.67 | 2.030, 1.961, 1.817, 1.673, 1.257, 1.014, 1.001, 1.081, 1.227, 1.351 |
| AP3 | `20261110T160405Z_tile_run_AP3_v2_m_profile` (51785414) | m | 0.05-0.90 x 200 A_crit | (5158, 0.39) | 1.298e-05, 0.01814 (interior) | 39.834 / 46.352 | 0.36-0.41 | 0.31-0.45 | 1.255, 1.237, 1.200, 1.055, 1.002, 1.125, 1.255, 1.459, 1.629, 1.774 |
| AP4 | `20261110T160405Z_tile_run_AP4_v2_m_profile` (51785415) | m | 0.05-0.90 x 200 A_crit | (7831, 0.21) | 9.781e-05, 0.01981 (interior) | 31.695 / 34.435 | 0.09-0.28 | 0.05-0.38 | 1.016, 1.008, 1.000, 1.010, 1.059, 1.140, 1.262, 1.418, 1.595, 1.772 |
| AP5 | `20261110T160405Z_tile_run_AP5_v2_m_profile` (51785416) | m | 0.05-0.90 x 200 A_crit | (1855, 0.42) | 8.954e-06, 0.01205 (interior) | 26.691 / 27.009 | 0.41-0.44 | 0.38-0.47 | 1.876, 1.736, 1.475, 1.208, 1.012, 1.130, 1.496, 1.500, 1.560, 1.676 |
| GM1 | `20261110T160404Z_tile_run_GM1_v2_m_profile` (51785417) | m | 0.05-0.90 x 200 A_crit | (1135, 0.29) | 6.109e-05, 0.01906 (interior) | 24.467 / 29.048 | 0.27-0.30 | 0.23-0.33 | 2.290, 1.744, 1.125, 1.003, 1.275, 1.840, 2.448, 3.136, 3.887, 4.531 |
| GM2 | `20261110T160420Z_tile_run_GM2_v2_m_profile` (51785418) | m | 0.05-0.90 x 200 A_crit | (3444, 0.29) | 6.939e-05, 0.02703 (interior) | 26.201 / 26.254 | 0.26-0.33 | 0.22-0.37 | 1.712, 1.377, 1.065, 1.000, 1.078, 1.320, 1.801, 2.419, 7.348, 11.684 |
| GM3 | `20261110T160420Z_tile_run_GM3_v2_m_profile` (51785419) | m | 0.05-0.90 x 200 A_crit | (859, 0.29) | 8.454e-05, 0.01897 (interior) | 32.808 / 38.587 | 0.27-0.31 | 0.24-0.34 | 1.387, 1.299, 1.131, 1.002, 1.128, 1.385, 1.600, 1.865, 4.415, 6.353 |
| GM4 | `20261110T160413Z_tile_run_GM4_v2_m_profile` (51785420) | m | 0.05-0.90 x 200 A_crit | (2718, 0.36) | 1.958e-05, 0.02135 (interior) | 39.512 / 40.754 | 0.34-0.38 | 0.30-0.40 | 2.208, 1.899, 1.365, 1.047, 1.034, 1.380, 1.900, 4.155, 4.953, 5.117 |
| GM5 | `20261110T160416Z_tile_run_GM5_v2_m_profile` (51785172) | m | 0.05-0.90 x 200 A_crit | (2247, 0.34) | 3.644e-05, 0.02038 (interior) | 31.646 / 32.714 | 0.32-0.35 | 0.30-0.38 | 2.235, 1.875, 1.323, 1.034, 1.081, 1.556, 2.267, 4.070, 8.895, 10.165 |

### A8b. Site profile S(m) = sum_t MSE*_t(m)/MSE*_t,min: optimum, 1 % and 5 % intervals, leave-one-out

| site | tiles | m_site | S_min | 1 % interval | 5 % interval | at grid edge | median fitted A_crit (tile optima) | median A_crit at m_site |
|---|---|---|---|---|---|---|---|---|
| AP | AP1, AP2, AP3, AP4, AP5 | **0.47** | 5.364 | 0.44-0.49 (6 pts, 1 int.) | 0.41-0.53 | False | 5158 | 4651 |
| GM | GM1, GM2, GM3, GM4, GM5 | **0.31** | 5.070 | 0.29-0.33 (5 pts, 1 int.) | 0.26-0.37 | False | 2247 | 2247 |

| site | left out | m_site | 1 % interval | 5 % interval |
|---|---|---|---|---|
| AP | AP1 | 0.44 | 0.42-0.47 | 0.38-0.50 |
| AP | AP2 | 0.45 | 0.42-0.47 | 0.38-0.51 |
| AP | AP3 | 0.48 | 0.46-0.50 | 0.42-0.54 |
| AP | AP4 | 0.47 | 0.45-0.49 | 0.42-0.53 |
| AP | AP5 | 0.49 | 0.46-0.53 | 0.42-0.58 |
| GM | GM1 | 0.32 | 0.30-0.35 | 0.27-0.38 |
| GM | GM2 | 0.32 | 0.30-0.33 | 0.27-0.37 |
| GM | GM3 | 0.32 | 0.30-0.34 | 0.27-0.37 |
| GM | GM4 | 0.31 | 0.28-0.33 | 0.25-0.36 |
| GM | GM5 | 0.31 | 0.28-0.33 | 0.25-0.36 |

### A8c. Measured site-pooled theta(A_c) at the median fitted A_crit (interpolated in log A_c; 95 % basin-bootstrap band)

Audit run `20261110T160633Z_measure_audit_10tiles`.

| site | A_c used | estimator | theta | 95 % band | basins at the nearest step | S(m) optimum m_site | m_site inside band | 1 % interval overlaps band |
|---|---|---|---|---|---|---|---|---|
| AP | median fitted A_crit 5158 | pooled | 0.48 | [0.44, 0.54] | 23 | 0.47 | yes | yes |
| AP | median fitted A_crit 5158 | equal | 0.47 | [0.36, 0.53] | 23 | 0.47 | yes | yes |
| AP | median A_crit at m_site 4651 | pooled | 0.48 | [0.44, 0.54] | 23 | 0.47 | yes | yes |
| AP | median A_crit at m_site 4651 | equal | 0.46 | [0.34, 0.52] | 23 | 0.47 | yes | yes |
| GM | median fitted A_crit 2247 | pooled | 0.34 | [0.33, 0.37] | 40 | 0.31 | no | yes |
| GM | median fitted A_crit 2247 | equal | 0.35 | [0.31, 0.38] | 40 | 0.31 | yes | yes |
| GM | median A_crit at m_site 2247 | pooled | 0.34 | [0.33, 0.37] | 40 | 0.31 | no | yes |
| GM | median A_crit at m_site 2247 | equal | 0.35 | [0.31, 0.38] | 40 | 0.31 | yes | yes |

## A9. Section 6.2: K recovery consistent in units, at the site-level m

| tile | m_site | K_fit (joint profile at m_site; A_crit, D) | k_s at theta = m_site, A_c = 1e3 (basins, n_eff) | U/k_s / K_fit (1e3) | k_s at A_c = 1e4 (basins, n_eff) | U/k_s / K_fit (1e4) | site-pooled k_s 1e3 / 1e4 |
|---|---|---|---|---|---|---|---|
| AP1 | 0.47 | 3.987e-06 (6686, 0.017) | 12.42 (32, 1.4) | **1.11** | 14.34 (5, 1.2) | **0.962** | 11.6 / 12.99 |
| AP2 | 0.47 | 4.595e-06 (3117, 0.01679) | 11.58 (13, 1.9) | **1.03** | 12.81 (2, 2.0) | **0.935** | 11.6 / 12.99 |
| AP3 | 0.47 | 4.799e-06 (4651, 0.01778) | 11.53 (32, 1.1) | **0.994** | 13.07 (2, 1.0) | **0.877** | 11.6 / 12.99 |
| AP4 | 0.47 | 4.52e-06 (7310, 0.01977) | 11.93 (30, 2.1) | **1.02** | 12.21 (5, 2.3) | **0.996** | 11.6 / 12.99 |
| AP5 | 0.47 | 5.114e-06 (1855, 0.01253) | 10.73 (9, 1.6) | **1** | 12.07 (2, 1.8) | **0.891** | 11.6 / 12.99 |
| GM1 | 0.31 | 4.866e-05 (1135, 0.0192) | 3.029 (19, 3.5) | **0.997** | 2.931 (8, 3.1) | **1.03** | 2.91 / 2.825 |
| GM2 | 0.31 | 5.511e-05 (3444, 0.02723) | 2.666 (16, 3.7) | **1** | 2.429 (5, 2.1) | **1.1** | 2.91 / 2.825 |
| GM3 | 0.31 | 6.81e-05 (859, 0.01922) | 2.292 (15, 2.9) | **0.942** | 2.221 (7, 2.2) | **0.972** | 2.91 / 2.825 |
| GM4 | 0.31 | 3.64e-05 (3832, 0.02188) | 4.225 (14, 1.5) | **0.956** | 4.002 (3, 1.3) | **1.01** | 2.91 / 2.825 |
| GM5 | 0.31 | 5.265e-05 (2247, 0.01974) | 2.929 (4, 1.2) | **0.953** | 2.847 (2, 1.1) | **0.981** | 2.91 / 2.825 |

## A10. Section 6.3: curvature options side by side (every row carries its hilltop definition, window and pixel count)

Audit run `20261110T160633Z_measure_audit_10tiles` (job 51785562, git 79eac4a6).

### A10a. L_f from the synthetic (crest width 380.0 m, tolerance 1 %): facet-plus-noise bias of the recovered curvature against window length

| window L (m, synthetic dx = 1) | 3 | 5 | 7 | 9 | 11 | 15 | 21 | 25 | 31 | 41 | 51 | 61 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AP (tin_noise; facet 2.76 m, noise 0.043 m, c_true 0.008; mean over 8 realisations) | +39.62 % (sd 75.51) | +15.57 % (sd 16.34) | +2.82 % (sd 7.06) | +0.49 % (sd 4.27) | -0.75 % (sd 3.03) | -0.26 % (sd 1.47) | -0.13 % (sd 0.45) | -0.12 % (sd 0.20) | -0.08 % (sd 0.16) | -0.02 % (sd 0.10) | +0.00 % (sd 0.07) | +0.01 % (sd 0.04) |
| GM (noise; facet 0.00 m, noise 0.043 m, c_true 0.0148; mean over 8 realisations) | +4.74 % (sd 41.02) | +2.94 % (sd 3.33) | +1.07 % (sd 2.69) | +1.95 % (sd 1.81) | -0.34 % (sd 1.16) | -0.16 % (sd 0.42) | -0.04 % (sd 0.25) | -0.01 % (sd 0.22) | +0.01 % (sd 0.07) | -0.01 % (sd 0.04) | -0.00 % (sd 0.02) | -0.00 % (sd 0.01) |

**L_f = 11.0 m (w_f = 5 cells; on AP (2 w_f + 1) dx = 10.73 m).** Rule: smallest window of the standard set with |mean bias over 8 realisations| < tol for the AP (TIN + noise) and GM (noise) variants on a wide crest.

### A10b. C_ht and D0 by option

| tile | option | hilltop definition | network | window (m) | pixels | segments | C_ht (1/m) [68 %] | D0 (m^2/yr) [68 %] | width 2d 25 / 50 % (m) | synthetic bias at median / 25 % width |
|---|---|---|---|---|---|---|---|---|---|---|
| AP1 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e3 m^2 (A_c 1e+03) | 49.7 | 3866 | 441 | 0.003004 [0.00169, 0.00427] | 0.01831 [0.0129, 0.0325] | 8 / 35 |  |
| AP1 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e3 m^2 (A_c 1e+03) | 49.7 | 1786 | 196 | 0.00839 [0.00701, 0.0102] | 0.006555 [0.00539, 0.00785] | 79 / 101 |  |
| AP1 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e4 m^2 (A_c 1e+04) | 49.7 | 839 | 97 | 0.006293 [0.00255, 0.00727] | 0.00874 [0.00757, 0.0215] | 177 / 226 |  |
| AP1 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e4 m^2 (A_c 1e+04) | 49.7 | 749 | 73 | 0.00675 [0.0048, 0.00756] | 0.008148 [0.00728, 0.0115] | 183 / 241 | +0.0 % / -0.0 % |
| AP1 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | rollover network (measure set) (A_c 282) | 49.7 | 5634 | 897 | 0.003043 [0.00211, 0.00383] | 0.01807 [0.0143, 0.0261] | 6 / 14 |  |
| AP1 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | rollover network (measure set) (A_c 282) | 49.7 | 1395 | 152 | 0.01149 [0.0104, 0.0119] | 0.004785 [0.00463, 0.00528] | 60 / 71 |  |
| AP1 | **fixed L_f, exclusion L_f/2 (option iii)** | measure set (rollover network), pixels within L_f/2 of the network dropped | rollover (A_c 282) | 10.7 | 3445 (of 6104) | 505 | 0.01533 [0.0143, 0.0162] | 0.003587 [0.00339, 0.00385] | 19 / 36 | -0.3 % / -0.7 % |
| AP2 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e3 m^2 (A_c 1e+03) | 20.5 | 1826 | 303 | 0.006923 [0.00602, 0.00772] | 0.007944 [0.00712, 0.00914] | 33 / 96 |  |
| AP2 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e3 m^2 (A_c 1e+03) | 20.5 | 1528 | 225 | 0.007523 [0.00643, 0.00833] | 0.007311 [0.0066, 0.00855] | 84 / 110 |  |
| AP2 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e4 m^2 (A_c 1e+04) | 20.5 | 849 | 90 | 0.007985 [0.00633, 0.0102] | 0.006888 [0.0054, 0.00869] | 234 / 283 |  |
| AP2 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e4 m^2 (A_c 1e+04) | 20.5 | 833 | 85 | 0.008039 [0.00647, 0.0104] | 0.006841 [0.00528, 0.0085] | 237 / 283 | -0.3 % / +0.7 % |
| AP2 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | rollover network (measure set) (A_c 56.2) | 20.5 | 3148 | 642 | 0.006132 [0.00577, 0.00688] | 0.00897 [0.008, 0.00953] | 4 / 6 |  |
| AP2 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | rollover network (measure set) (A_c 56.2) | 20.5 | 307 | 33 | 0.01474 [0.0142, 0.0172] | 0.003732 [0.0032, 0.00389] | 25 / 31 |  |
| AP2 | **fixed L_f, exclusion L_f/2 (option iii)** | measure set (rollover network), pixels within L_f/2 of the network dropped | rollover (A_c 56.2) | 10.7 | 940 (of 3497) | 122 | 0.01709 [0.0158, 0.0182] | 0.003218 [0.00303, 0.00347] | 13 / 17 | +1.8 % / +1.6 % |
| AP3 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e3 m^2 (A_c 1e+03) | 40.0 | 3106 | 682 | 0.007116 [0.00584, 0.00783] | 0.007729 [0.00702, 0.00942] | 22 / 93 |  |
| AP3 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e3 m^2 (A_c 1e+03) | 40.0 | 2097 | 448 | 0.009332 [0.00842, 0.0102] | 0.005893 [0.00537, 0.00653] | 95 / 124 |  |
| AP3 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e4 m^2 (A_c 1e+04) | 40.0 | 646 | 79 | 0.003102 [0.000833, 0.00739] | 0.01773 [0.00744, 0.066] | 24 / 70 |  |
| AP3 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e4 m^2 (A_c 1e+04) | 40.0 | 460 | 62 | 0.006137 [0.00157, 0.00742] | 0.008962 [0.00741, 0.035] | 68 / 248 | +0.0 % / -0.0 % |
| AP3 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | rollover network (measure set) (A_c 17.8) | 40.0 | 7565 | 2040 | 0.004764 [0.00423, 0.00541] | 0.01155 [0.0102, 0.013] | 2 / 4 |  |
| AP3 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | rollover network (measure set) (A_c 17.8) | 40.0 | 0 | 0 | -  | -  | - / - |  |
| AP3 | **fixed L_f, exclusion L_f/2 (option iii)** | measure set (rollover network), pixels within L_f/2 of the network dropped | rollover (A_c 17.8) | 10.7 | 531 (of 9409) | 61 | 0.02553 [0.0221, 0.0288] | 0.002155 [0.00191, 0.00249] | 12 / 12 | +0.8 % / -0.6 % |
| AP4 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e3 m^2 (A_c 1e+03) | 40.0 | 4513 | 488 | 0.006162 [0.00529, 0.00688] | 0.008926 [0.008, 0.0104] | 17 / 79 |  |
| AP4 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e3 m^2 (A_c 1e+03) | 40.0 | 2962 | 302 | 0.008511 [0.00785, 0.00923] | 0.006462 [0.00596, 0.007] | 84 / 108 |  |
| AP4 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e4 m^2 (A_c 1e+04) | 40.0 | 2502 | 128 | 0.00676 [0.00566, 0.00815] | 0.008136 [0.00675, 0.00972] | 23 / 273 |  |
| AP4 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e4 m^2 (A_c 1e+04) | 40.0 | 1869 | 116 | 0.008722 [0.00784, 0.00993] | 0.006306 [0.00554, 0.00702] | 267 / 293 | +0.2 % / -0.1 % |
| AP4 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | rollover network (measure set) (A_c 447) | 40.0 | 5566 | 798 | 0.005736 [0.00498, 0.00644] | 0.009589 [0.00854, 0.0111] | 11 / 27 |  |
| AP4 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | rollover network (measure set) (A_c 447) | 40.0 | 2413 | 260 | 0.009607 [0.00901, 0.0102] | 0.005725 [0.00539, 0.0061] | 58 / 76 |  |
| AP4 | **fixed L_f, exclusion L_f/2 (option iii)** | measure set (rollover network), pixels within L_f/2 of the network dropped | rollover (A_c 447) | 10.7 | 4703 (of 6136) | 579 | 0.01515 [0.0143, 0.0165] | 0.003629 [0.00334, 0.00384] | 20 / 44 | +0.2 % / -0.5 % |
| AP5 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e3 m^2 (A_c 1e+03) | 20.5 | 2586 | 244 | 0.008608 [0.00785, 0.00927] | 0.006389 [0.00593, 0.007] | 104 / 131 |  |
| AP5 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e3 m^2 (A_c 1e+03) | 20.5 | 2460 | 208 | 0.008864 [0.00823, 0.00979] | 0.006205 [0.00562, 0.00668] | 112 / 133 |  |
| AP5 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e4 m^2 (A_c 1e+04) | 20.5 | 1322 | 77 | 0.00894 [0.00842, 0.0107] | 0.006152 [0.00514, 0.00653] | 264 / 297 |  |
| AP5 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e4 m^2 (A_c 1e+04) | 20.5 | 1319 | 75 | 0.008938 [0.00841, 0.0107] | 0.006153 [0.00515, 0.00654] | 268 / 298 | -0.6 % / -0.2 % |
| AP5 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | rollover network (measure set) (A_c 56.2) | 20.5 | 6626 | 1041 | 0.008893 [0.00812, 0.00965] | 0.006185 [0.0057, 0.00677] | 4 / 11 |  |
| AP5 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | rollover network (measure set) (A_c 56.2) | 20.5 | 1469 | 150 | 0.01776 [0.0155, 0.0199] | 0.003096 [0.00277, 0.00354] | 23 / 26 |  |
| AP5 | **fixed L_f, exclusion L_f/2 (option iii)** | measure set (rollover network), pixels within L_f/2 of the network dropped | rollover (A_c 56.2) | 10.7 | 3538 (of 7107) | 346 | 0.01883 [0.0166, 0.0206] | 0.002921 [0.00268, 0.0033] | 15 / 19 | -1.0 % / +0.5 % |
| GM1 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e3 m^2 (A_c 1e+03) | 9.0 | 5084 | 654 | 0.01445 [0.0139, 0.0151] | 0.01017 [0.00975, 0.0106] | 94 / 112 |  |
| GM1 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e3 m^2 (A_c 1e+03) | 9.0 | 5037 | 642 | 0.01454 [0.014, 0.0151] | 0.01011 [0.00972, 0.0105] | 95 / 112 |  |
| GM1 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e4 m^2 (A_c 1e+04) | 9.0 | 4122 | 516 | 0.01484 [0.0143, 0.0156] | 0.009905 [0.00941, 0.0103] | 107 / 130 |  |
| GM1 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e4 m^2 (A_c 1e+04) | 9.0 | 4121 | 516 | 0.01484 [0.0143, 0.0156] | 0.009905 [0.0094, 0.0103] | 107 / 130 | +0.4 % / +2.0 % |
| GM1 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | rollover network (measure set) (A_c 35.5) | 9.0 | 11053 | 2278 | 0.01451 [0.0142, 0.0149] | 0.01013 [0.00987, 0.0104] | 4 / 7 |  |
| GM1 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | rollover network (measure set) (A_c 35.5) | 9.0 | 4516 | 659 | 0.01677 [0.0162, 0.0173] | 0.008767 [0.0085, 0.00908] | 12 / 14 |  |
| GM1 | **fixed L_f, exclusion L_f/2 (option iii)** | measure set (rollover network), pixels within L_f/2 of the network dropped | rollover (A_c 35.5) | 11.0 | 3577 (of 11112) | 507 | 0.01633 [0.0157, 0.0172] | 0.009001 [0.00853, 0.00935] | 13 / 15 | +0.5 % / -0.9 % |
| GM2 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e3 m^2 (A_c 1e+03) | 11.0 | 4778 | 539 | 0.0114 [0.0108, 0.012] | 0.0129 [0.0122, 0.0137] | 96 / 118 |  |
| GM2 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e3 m^2 (A_c 1e+03) | 11.0 | 4649 | 505 | 0.01137 [0.0107, 0.012] | 0.01293 [0.0122, 0.0137] | 98 / 119 |  |
| GM2 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e4 m^2 (A_c 1e+04) | 11.0 | 3538 | 336 | 0.01104 [0.0105, 0.0117] | 0.01331 [0.0126, 0.0139] | 156 / 182 |  |
| GM2 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e4 m^2 (A_c 1e+04) | 11.0 | 3510 | 329 | 0.01114 [0.0105, 0.0118] | 0.0132 [0.0125, 0.014] | 157 / 182 | -1.5 % / +1.9 % |
| GM2 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | rollover network (measure set) (A_c 35.5) | 11.0 | 9021 | 2144 | 0.009911 [0.00928, 0.0105] | 0.01483 [0.014, 0.0158] | 3 / 6 |  |
| GM2 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | rollover network (measure set) (A_c 35.5) | 11.0 | 2511 | 295 | 0.01505 [0.0141, 0.016] | 0.00977 [0.00917, 0.0104] | 13 / 16 |  |
| GM2 | **fixed L_f, exclusion L_f/2 (option iii)** | measure set (rollover network), pixels within L_f/2 of the network dropped | rollover (A_c 35.5) | 11.0 | 2511 (of 9187) | 295 | 0.01505 [0.0141, 0.016] | 0.00977 [0.00917, 0.0104] | 13 / 16 | -0.5 % / -1.6 % |
| GM3 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e3 m^2 (A_c 1e+03) | 15.0 | 7178 | 491 | 0.007067 [0.00635, 0.00787] | 0.0208 [0.0187, 0.0231] | 63 / 98 |  |
| GM3 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e3 m^2 (A_c 1e+03) | 15.0 | 6906 | 460 | 0.007179 [0.00643, 0.00798] | 0.02048 [0.0184, 0.0229] | 70 / 99 |  |
| GM3 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e4 m^2 (A_c 1e+04) | 15.0 | 5091 | 380 | 0.006162 [0.00554, 0.00668] | 0.02386 [0.022, 0.0265] | 148 / 182 |  |
| GM3 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e4 m^2 (A_c 1e+04) | 15.0 | 5039 | 376 | 0.006122 [0.00558, 0.00663] | 0.02401 [0.0222, 0.0264] | 149 / 182 | -0.5 % / +0.3 % |
| GM3 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | rollover network (measure set) (A_c 22.4) | 15.0 | 13408 | 2001 | 0.00811 [0.00763, 0.00856] | 0.01813 [0.0172, 0.0193] | 3 / 6 |  |
| GM3 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | rollover network (measure set) (A_c 22.4) | 15.0 | 219 | 43 | 0.01415 [0.0138, 0.0181] | 0.01039 [0.00813, 0.0106] | 16 / 16 |  |
| GM3 | **fixed L_f, exclusion L_f/2 (option iii)** | measure set (rollover network), pixels within L_f/2 of the network dropped | rollover (A_c 22.4) | 11.0 | 1500 (of 14030) | 190 | 0.01453 [0.0135, 0.0154] | 0.01012 [0.00955, 0.0109] | 12 / 13 | -0.6 % / +0.6 % |
| GM4 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e3 m^2 (A_c 1e+03) | 21.0 | 3225 | 380 | 0.01388 [0.0124, 0.0153] | 0.01059 [0.00962, 0.0119] | 100 / 115 |  |
| GM4 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e3 m^2 (A_c 1e+03) | 21.0 | 3086 | 363 | 0.01451 [0.0127, 0.0158] | 0.01013 [0.00933, 0.0116] | 102 / 118 |  |
| GM4 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e4 m^2 (A_c 1e+04) | 21.0 | 1869 | 220 | 0.01594 [0.0152, 0.0171] | 0.009225 [0.00862, 0.00967] | 184 / 218 |  |
| GM4 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e4 m^2 (A_c 1e+04) | 21.0 | 1869 | 216 | 0.01594 [0.0152, 0.0171] | 0.009225 [0.00862, 0.00967] | 185 / 218 | -0.0 % / -0.1 % |
| GM4 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | rollover network (measure set) (A_c 56.2) | 21.0 | 6033 | 1135 | 0.01158 [0.0102, 0.0127] | 0.0127 [0.0116, 0.0145] | 6 / 12 |  |
| GM4 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | rollover network (measure set) (A_c 56.2) | 21.0 | 1410 | 153 | 0.02051 [0.0197, 0.0214] | 0.007168 [0.00687, 0.00746] | 23 / 26 |  |
| GM4 | **fixed L_f, exclusion L_f/2 (option iii)** | measure set (rollover network), pixels within L_f/2 of the network dropped | rollover (A_c 56.2) | 11.0 | 3344 (of 6225) | 373 | 0.01492 [0.0135, 0.0166] | 0.009855 [0.00883, 0.0109] | 15 / 20 | +0.6 % / +0.5 % |
| GM5 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e3 m^2 (A_c 1e+03) | 9.0 | 998 | 163 | 0.01781 [0.0151, 0.0191] | 0.008253 [0.00772, 0.00972] | 85 / 117 |  |
| GM5 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e3 m^2 (A_c 1e+03) | 9.0 | 968 | 153 | 0.01813 [0.0148, 0.019] | 0.008108 [0.00772, 0.00996] | 89 / 118 |  |
| GM5 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | network 1e4 m^2 (A_c 1e+04) | 9.0 | 786 | 123 | 0.01774 [0.0135, 0.019] | 0.008288 [0.00773, 0.0109] | 198 / 251 |  |
| GM5 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | network 1e4 m^2 (A_c 1e+04) | 9.0 | 765 | 118 | 0.01791 [0.0146, 0.019] | 0.008209 [0.00773, 0.0101] | 230 / 251 | -2.5 % / +0.6 % |
| GM5 | fixed L*, no exclusion | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4 | rollover network (measure set) (A_c 28.2) | 9.0 | 4850 | 1355 | 0.01464 [0.0139, 0.0155] | 0.01004 [0.00947, 0.0106] | 2 / 4 |  |
| GM5 | fixed L*, exclusion L*/2 (option i) | divides between outlet basins of the network, 2 cells from the edge, slope <= 0.4; pixels within L*/2 of the network dropped | rollover network (measure set) (A_c 28.2) | 9.0 | 1383 | 172 | 0.01946 [0.0188, 0.02] | 0.007554 [0.00734, 0.00784] | 12 / 14 |  |
| GM5 | **fixed L_f, exclusion L_f/2 (option iii)** | measure set (rollover network), pixels within L_f/2 of the network dropped | rollover (A_c 28.2) | 11.0 | 1115 (of 4964) | 144 | 0.01937 [0.0185, 0.0199] | 0.007589 [0.00739, 0.00794] | 13 / 15 | +0.5 % / -0.9 % |

## A11. Section 6.4.2: free ladder rerun with the theta grid from 0.00 (pooled / equal-weight / median estimators; n_eff; fraction of kept basins with best theta at the lower edge)

Audit run `20261110T160633Z_measure_audit_10tiles` (job 51785562); theta grid 0.00-0.90; basin bootstrap, 200 draws.

| tile | 100 | 398.107 | 1000 | 1995.26 | 3981.07 | 10000 | 39810.7 | 100000 |
|---|---|---|---|---|---|---|---|---|
| AP1 | P 0.32 [0.05, 0.32]<br>E 0.05 [0.00, 0.12]<br>M 0.00 [0.00, 0.02]<br>n_eff 1.3, lower-edge 0.55 | P 0.35 [0.00, 0.36]<br>E 0.13 [0.00, 0.26]<br>M 0.04 [0.00, 0.15]<br>n_eff 1.3, lower-edge 0.45 | P 0.39 [0.00, 0.39]<br>E 0.02 [0.00, 0.29]<br>M 0.00 [0.00, 0.10]<br>n_eff 1.4, lower-edge 0.66 | P 0.46 [0.00, 0.47]<br>E 0.00 [0.00, 0.41]<br>M 0.00 [0.00, 0.27]<br>n_eff 1.6, lower-edge 0.59 | P 0.53 [0.00, 0.54] (7)<br>E 0.16 [0.00, 0.51] (7)<br>M 0.02 [0.00, 0.54] (7)<br>n_eff 1.6, lower-edge 0.43 | P 0.58 [0.00, 0.58] (5)<br>E 0.40 [0.00, 0.60] (5)<br>M 0.26 [0.00, 0.90] (5)<br>n_eff 1.2, lower-edge 0.40 | P 0.60 (1)<br>E 0.60 (1)<br>M 0.60 (1)<br>n_eff 1.0, lower-edge 0.00 | P 0.52 (1)<br>E 0.52 (1)<br>M 0.52 (1)<br>n_eff 1.0, lower-edge 0.00 |
| AP2 | P 0.32 [0.06, 0.33]<br>E 0.19 [0.00, 0.28]<br>M 0.05 [0.00, 0.09]<br>n_eff 1.9, lower-edge 0.44 | P 0.36 [0.07, 0.38]<br>E 0.30 [0.06, 0.35]<br>M 0.12 [0.01, 0.31]<br>n_eff 1.9, lower-edge 0.28 | P 0.41 [0.24, 0.43]<br>E 0.39 [0.22, 0.43]<br>M 0.33 [0.00, 0.50]<br>n_eff 1.9, lower-edge 0.23 | P 0.47 [0.00, 0.49] (4)<br>E 0.46 [0.00, 0.49] (4)<br>M 0.21 [0.00, 0.49] (4)<br>n_eff 1.9, lower-edge 0.50 | P 0.53 (2)<br>E 0.52 (2)<br>M 0.52 (2)<br>n_eff 1.9, lower-edge 0.00 | P 0.57 (2)<br>E 0.56 (2)<br>M 0.57 (2)<br>n_eff 2.0, lower-edge 0.00 | P 0.55 (2)<br>E 0.55 (2)<br>M 0.55 (2)<br>n_eff 1.9, lower-edge 0.00 | P 0.77 (2)<br>E 0.76 (2)<br>M 0.69 (2)<br>n_eff 1.7, lower-edge 0.00 |
| AP3 | P 0.33 [0.07, 0.33]<br>E 0.08 [0.03, 0.16]<br>M 0.04 [0.01, 0.07]<br>n_eff 1.3, lower-edge 0.41 | P 0.36 [0.08, 0.36]<br>E 0.19 [0.08, 0.30]<br>M 0.12 [0.07, 0.20]<br>n_eff 1.1, lower-edge 0.30 | P 0.38 [0.11, 0.38]<br>E 0.31 [0.11, 0.39]<br>M 0.41 [0.06, 0.49]<br>n_eff 1.1, lower-edge 0.34 | P 0.40 [0.14, 0.51]<br>E 0.32 [0.00, 0.39]<br>M 0.41 [0.13, 0.89]<br>n_eff 1.0, lower-edge 0.20 | P 0.43 [0.00, 0.67] (4)<br>E 0.43 [0.00, 0.85] (4)<br>M 0.21 [0.00, 0.90] (4)<br>n_eff 1.0, lower-edge 0.50 | P 0.45 (2)<br>E 0.45 (2)<br>M 0.23 (2)<br>n_eff 1.0, lower-edge 0.50 | P 0.54 (1)<br>E 0.54 (1)<br>M 0.54 (1)<br>n_eff 1.0, lower-edge 0.00 | P 0.59 (1)<br>E 0.59 (1)<br>M 0.59 (1)<br>n_eff 1.0, lower-edge 0.00 |
| AP4 | P 0.27 [0.21, 0.27]<br>E 0.15 [0.08, 0.19]<br>M 0.11 [0.07, 0.15]<br>n_eff 2.2, lower-edge 0.28 | P 0.32 [0.28, 0.35]<br>E 0.25 [0.16, 0.29]<br>M 0.32 [0.18, 0.44]<br>n_eff 2.2, lower-edge 0.19 | P 0.36 [0.33, 0.46]<br>E 0.31 [0.18, 0.37]<br>M 0.15 [0.00, 0.36]<br>n_eff 2.1, lower-edge 0.40 | P 0.39 [0.39, 0.51]<br>E 0.41 [0.33, 0.48]<br>M 0.11 [0.00, 0.41]<br>n_eff 2.0, lower-edge 0.33 | P 0.49 [0.44, 0.62] (9)<br>E 0.52 [0.44, 0.60] (9)<br>M 0.44 [0.31, 0.67] (9)<br>n_eff 2.1, lower-edge 0.22 | P 0.59 [0.56, 0.90] (5)<br>E 0.73 [0.61, 0.90] (5)<br>M 0.84 [0.56, 0.90] (5)<br>n_eff 2.3, lower-edge 0.00 | P 0.47 [0.47, 0.86] (3)<br>E 0.54 [0.47, 0.86] (3)<br>M 0.64 [0.47, 0.86] (3)<br>n_eff 1.2, lower-edge 0.00 | P 0.21 (1)<br>E 0.21 (1)<br>M 0.21 (1)<br>n_eff 1.0, lower-edge 0.00 |
| AP5 | P 0.33 [0.05, 0.34]<br>E 0.16 [0.06, 0.25]<br>M 0.10 [0.05, 0.21]<br>n_eff 1.6, lower-edge 0.34 | P 0.36 [0.03, 0.37]<br>E 0.26 [0.05, 0.34]<br>M 0.29 [0.16, 0.38]<br>n_eff 1.6, lower-edge 0.15 | P 0.39 [0.17, 0.40] (9)<br>E 0.36 [0.12, 0.39] (9)<br>M 0.22 [0.00, 0.40] (9)<br>n_eff 1.6, lower-edge 0.22 | P 0.42 [0.18, 0.60] (5)<br>E 0.41 [0.22, 0.67] (5)<br>M 0.43 [0.00, 0.90] (5)<br>n_eff 1.6, lower-edge 0.20 | P 0.44 [0.00, 0.51] (3)<br>E 0.44 [0.00, 0.51] (3)<br>M 0.43 [0.00, 0.51] (3)<br>n_eff 1.5, lower-edge 0.33 | P 0.50 (2)<br>E 0.51 (2)<br>M 0.53 (2)<br>n_eff 1.8, lower-edge 0.00 | P 0.42 (2)<br>E 0.42 (2)<br>M 0.43 (2)<br>n_eff 1.8, lower-edge 0.00 | P 0.04 (2)<br>E 0.09 (2)<br>M 0.07 (2)<br>n_eff 1.1, lower-edge 0.50 |
| GM1 | P 0.25 [0.20, 0.25]<br>E 0.08 [0.01, 0.14]<br>M 0.07 [0.00, 0.13]<br>n_eff 3.6, lower-edge 0.44 | P 0.29 [0.18, 0.31]<br>E 0.19 [0.07, 0.25]<br>M 0.20 [0.00, 0.29]<br>n_eff 3.4, lower-edge 0.47 | P 0.32 [0.30, 0.39]<br>E 0.28 [0.13, 0.38]<br>M 0.34 [0.00, 0.42]<br>n_eff 3.5, lower-edge 0.37 | P 0.34 [0.32, 0.44]<br>E 0.34 [0.27, 0.44]<br>M 0.41 [0.00, 0.50]<br>n_eff 3.5, lower-edge 0.27 | P 0.37 [0.35, 0.48] (8)<br>E 0.39 [0.35, 0.58] (8)<br>M 0.43 [0.36, 0.58] (8)<br>n_eff 3.3, lower-edge 0.00 | P 0.38 [0.35, 0.51] (8)<br>E 0.42 [0.36, 0.69] (8)<br>M 0.46 [0.38, 0.80] (8)<br>n_eff 3.1, lower-edge 0.00 | P 0.35 [0.33, 0.37] (4)<br>E 0.35 [0.33, 0.38] (4)<br>M 0.34 [0.32, 0.41] (4)<br>n_eff 3.4, lower-edge 0.00 | P 0.82 [0.45, 0.90] (4)<br>E 0.75 [0.41, 0.90] (4)<br>M 0.62 [0.22, 0.90] (4)<br>n_eff 2.7, lower-edge 0.00 |
| GM2 | P 0.26 [0.24, 0.29]<br>E 0.09 [0.00, 0.16]<br>M 0.00 [0.00, 0.00]<br>n_eff 4.0, lower-edge 0.66 | P 0.28 [0.27, 0.33]<br>E 0.21 [0.05, 0.25]<br>M 0.00 [0.00, 0.07]<br>n_eff 3.8, lower-edge 0.53 | P 0.31 [0.26, 0.38]<br>E 0.26 [0.16, 0.33]<br>M 0.31 [0.06, 0.39]<br>n_eff 3.7, lower-edge 0.19 | P 0.32 [0.26, 0.40]<br>E 0.30 [0.19, 0.38]<br>M 0.33 [0.00, 0.44]<br>n_eff 3.1, lower-edge 0.27 | P 0.30 [0.28, 0.43] (8)<br>E 0.28 [0.23, 0.36] (8)<br>M 0.33 [0.28, 0.67] (8)<br>n_eff 2.6, lower-edge 0.12 | P 0.31 [0.28, 0.42] (5)<br>E 0.28 [0.20, 0.34] (5)<br>M 0.28 [0.00, 0.43] (5)<br>n_eff 2.1, lower-edge 0.20 | P 0.47 [0.37, 0.48] (3)<br>E 0.44 [0.37, 0.48] (3)<br>M 0.47 [0.37, 0.48] (3)<br>n_eff 2.0, lower-edge 0.00 | P 0.42 [0.22, 0.53] (3)<br>E 0.32 [0.22, 0.53] (3)<br>M 0.27 [0.22, 0.53] (3)<br>n_eff 1.9, lower-edge 0.00 |
| GM3 | P 0.26 [0.22, 0.28]<br>E 0.15 [0.08, 0.19]<br>M 0.07 [0.00, 0.10]<br>n_eff 2.6, lower-edge 0.44 | P 0.28 [0.24, 0.30]<br>E 0.20 [0.00, 0.28]<br>M 0.23 [0.15, 0.30]<br>n_eff 2.7, lower-edge 0.13 | P 0.35 [0.28, 0.41]<br>E 0.29 [0.00, 0.42]<br>M 0.37 [0.28, 0.49]<br>n_eff 2.9, lower-edge 0.13 | P 0.41 [0.39, 0.49]<br>E 0.38 [0.02, 0.49]<br>M 0.44 [0.29, 0.64]<br>n_eff 2.8, lower-edge 0.07 | P 0.44 [0.40, 0.57] (9)<br>E 0.48 [0.35, 0.61] (9)<br>M 0.50 [0.46, 0.58] (9)<br>n_eff 2.6, lower-edge 0.11 | P 0.42 [0.34, 0.57] (7)<br>E 0.45 [0.32, 0.56] (7)<br>M 0.50 [0.34, 0.63] (7)<br>n_eff 2.2, lower-edge 0.00 | P 0.14 [0.11, 0.19] (4)<br>E 0.10 [0.00, 0.18] (4)<br>M 0.14 [0.00, 0.25] (4)<br>n_eff 1.4, lower-edge 0.25 | P 0.00 (1)<br>E 0.00 (1)<br>M 0.00 (1)<br>n_eff 1.0, lower-edge 1.00 |
| GM4 | P 0.27 [0.08, 0.27]<br>E 0.10 [0.03, 0.18]<br>M 0.00 [0.00, 0.00]<br>n_eff 1.7, lower-edge 0.59 | P 0.31 [0.12, 0.33]<br>E 0.23 [0.10, 0.28]<br>M 0.11 [0.07, 0.16]<br>n_eff 1.6, lower-edge 0.28 | P 0.35 [0.26, 0.38]<br>E 0.34 [0.22, 0.37]<br>M 0.24 [0.11, 0.35]<br>n_eff 1.5, lower-edge 0.07 | P 0.36 [0.29, 0.53] (8)<br>E 0.37 [0.28, 0.43] (8)<br>M 0.35 [0.25, 0.58] (8)<br>n_eff 1.4, lower-edge 0.12 | P 0.34 [0.32, 0.61] (5)<br>E 0.34 [0.32, 0.60] (5)<br>M 0.40 [0.32, 0.62] (5)<br>n_eff 1.3, lower-edge 0.00 | P 0.31 [0.21, 0.33] (3)<br>E 0.25 [0.21, 0.33] (3)<br>M 0.26 [0.21, 0.33] (3)<br>n_eff 1.3, lower-edge 0.00 | P 0.27 (2)<br>E 0.19 (2)<br>M 0.22 (2)<br>n_eff 1.4, lower-edge 0.00 | P 0.00 (1)<br>E 0.00 (1)<br>M 0.00 (1)<br>n_eff 1.0, lower-edge 1.00 |
| GM5 | P 0.28 [0.00, 0.29]<br>E 0.16 [0.00, 0.22]<br>M 0.00 [0.00, 0.01]<br>n_eff 1.4, lower-edge 0.57 | P 0.31 [0.00, 0.33]<br>E 0.30 [0.00, 0.32]<br>M 0.00 [0.00, 0.18]<br>n_eff 1.3, lower-edge 0.65 | P 0.32 [0.00, 0.32] (4)<br>E 0.31 [0.00, 0.32] (4)<br>M 0.15 [0.00, 0.32] (4)<br>n_eff 1.2, lower-edge 0.50 | P 0.32 [0.00, 0.33] (3)<br>E 0.33 [0.00, 0.33] (3)<br>M 0.32 [0.00, 0.33] (3)<br>n_eff 1.2, lower-edge 0.33 | P 0.32 (2)<br>E 0.33 (2)<br>M 0.33 (2)<br>n_eff 1.2, lower-edge 0.00 | P 0.30 (2)<br>E 0.28 (2)<br>M 0.28 (2)<br>n_eff 1.1, lower-edge 0.00 | P 0.27 (2)<br>E 0.28 (2)<br>M 0.28 (2)<br>n_eff 1.0, lower-edge 0.00 | P 0.26 (1)<br>E 0.26 (1)<br>M 0.26 (1)<br>n_eff 1.0, lower-edge 0.00 |
| **AP pooled** | P 0.32 [0.29, 0.33]<br>E 0.12 [0.08, 0.15]<br>M 0.05 [0.03, 0.07]<br>n_eff 7.6, lower-edge 0.40 | P 0.35 [0.34, 0.37]<br>E 0.22 [0.17, 0.26]<br>M 0.15 [0.10, 0.20]<br>n_eff 7.1, lower-edge 0.29 | P 0.39 [0.37, 0.41]<br>E 0.29 [0.21, 0.34]<br>M 0.14 [0.00, 0.26]<br>n_eff 6.8, lower-edge 0.42 | P 0.43 [0.39, 0.46]<br>E 0.36 [0.21, 0.42]<br>M 0.15 [0.00, 0.38]<br>n_eff 7.0, lower-edge 0.39 | P 0.48 [0.44, 0.53]<br>E 0.45 [0.30, 0.50]<br>M 0.43 [0.00, 0.48]<br>n_eff 7.0, lower-edge 0.32 | P 0.52 [0.47, 0.57]<br>E 0.53 [0.45, 0.59]<br>M 0.56 [0.47, 0.60]<br>n_eff 6.9, lower-edge 0.19 | P 0.55 [0.46, 0.58] (9)<br>E 0.53 [0.47, 0.57] (9)<br>M 0.54 [0.44, 0.64] (9)<br>n_eff 5.3, lower-edge 0.00 | P 0.57 [0.38, 0.67] (7)<br>E 0.57 [0.36, 0.70] (7)<br>M 0.47 [0.15, 0.59] (7)<br>n_eff 4.5, lower-edge 0.14 |
| **GM pooled** | P 0.27 [0.26, 0.27]<br>E 0.11 [0.07, 0.15]<br>M 0.00 [0.00, 0.00]<br>n_eff 10.2, lower-edge 0.54 | P 0.30 [0.29, 0.31]<br>E 0.22 [0.17, 0.25]<br>M 0.09 [0.03, 0.16]<br>n_eff 9.1, lower-edge 0.42 | P 0.33 [0.31, 0.35]<br>E 0.30 [0.24, 0.34]<br>M 0.31 [0.25, 0.34]<br>n_eff 7.8, lower-edge 0.22 | P 0.34 [0.32, 0.37]<br>E 0.34 [0.30, 0.37]<br>M 0.36 [0.32, 0.41]<br>n_eff 7.6, lower-edge 0.19 | P 0.33 [0.31, 0.37]<br>E 0.34 [0.31, 0.39]<br>M 0.42 [0.35, 0.49]<br>n_eff 7.3, lower-edge 0.06 | P 0.31 [0.28, 0.34]<br>E 0.29 [0.26, 0.37]<br>M 0.35 [0.28, 0.50]<br>n_eff 7.5, lower-edge 0.04 | P 0.27 [0.22, 0.31]<br>E 0.25 [0.17, 0.30]<br>M 0.31 [0.14, 0.36]<br>n_eff 5.9, lower-edge 0.07 | P 0.26 [0.00, 0.44]<br>E 0.27 [0.00, 0.48]<br>M 0.27 [0.11, 0.53]<br>n_eff 2.9, lower-edge 0.20 |

P = pooled through-origin (02c), E = equal-weight, M = median of per-basin best theta; [95 %] basin bootstrap; (n) basins when fewer than 10; lower-edge = fraction of kept basins whose own best theta is 0.00.

