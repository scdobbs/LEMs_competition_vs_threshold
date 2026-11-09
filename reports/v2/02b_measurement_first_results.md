# Measurement pipeline: first results on the ten tiles (addendum 1)

`lemcvt measure --tile all` (job 51685440, about 1 min per tile), metre grids on both sites
(AP converted from US survey feet), the tiles' own D8 area grids, no model output.
Methods: `docs/measurement_methods.md`. Results: `$GROUP_HOME/lemcvt_results/*_measure_*`
(`params.json`, `curves.parquet`). This note reports what came out and the one decision it
forces; it does not change any config.

## 1. The threshold decides theta

The concavity found by the chi fit depends strongly on the drainage-area threshold A_c
that defines "channel pixels", and the dependence differs between the sites. theta* along
the threshold ladder (A_c in m^2; chi fit with the model's upstream rule; theta grid 0.10-0.90):

| tile | 1e2 | 2e2 | 5e2 | 1e3 | 2e3 | 5e3 | 1e4 | 2e4 | 5e4 | 1e5 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 0.32 | 0.34 | 0.35 | 0.39 | 0.46 | 0.54 | 0.58 | 0.59 | 0.58 | 0.52 |
| AP2 | 0.32 | 0.34 | 0.38 | 0.41 | 0.47 | 0.54 | 0.57 | 0.53 | 0.54 | 0.77 |
| AP3 | 0.33 | 0.35 | 0.37 | 0.38 | 0.40 | 0.43 | 0.45 | 0.48 | 0.56 | 0.59 |
| AP4 | 0.27 | 0.30 | 0.34 | 0.36 | 0.39 | 0.51 | 0.59 | 0.58 | 0.42 | 0.21 |
| AP5 | 0.33 | 0.34 | 0.36 | 0.39 | 0.42 | 0.44 | 0.50 | 0.48 | 0.35 | 0.10 |
| GM1 | 0.25 | 0.26 | 0.30 | 0.32 | 0.34 | 0.37 | 0.38 | 0.37 | 0.42 | 0.82 |
| GM2 | 0.26 | 0.28 | 0.29 | 0.31 | 0.32 | 0.31 | 0.31 | 0.35 | 0.35 | 0.30 |
| GM3 | 0.26 | 0.27 | 0.29 | 0.35 | 0.41 | 0.41 | 0.42 | 0.30 | 0.13 | 0.10 |
| GM4 | 0.27 | 0.29 | 0.32 | 0.35 | 0.36 | 0.33 | 0.31 | 0.30 | 0.28 | 0.25 |
| GM5 | 0.28 | 0.30 | 0.31 | 0.32 | 0.32 | 0.31 | 0.30 | 0.28 | 0.27 | 0.26 |

Reading: on AP theta* climbs from about 0.33 at 1e2 m^2 to 0.55-0.59 at 1e4-2e4 m^2, then
becomes erratic above 5e4 m^2 where only one or two basins remain in a 1 km tile. On GM
theta* is 0.26-0.30 at small thresholds and 0.30-0.42 at 1e3-1e4 m^2, with GM2, GM4 and
GM5 nearly flat. The v1 values (AP 0.40-0.59, GM 0.30-0.42) were measured at the Taylor
thresholds (5.04e4 and 1e4 m^2) and sit at the upper end of these curves.

Two objective rules for A_c were implemented and both are reported:

* **Slope-area rollover** (primary in the current code): the area of maximum binned D8
  slope. It gives A_c = 18-450 m^2 on all tiles, i.e. the hillslope-to-hollow transition at
  1 m resolution, not the head of a fluvial channel. With that channel set theta* is
  0.22-0.35 and k_s is 1.1-3.2 everywhere; AP and GM become almost indistinguishable.
* **theta plateau** (smallest ladder threshold above which theta* stays within 0.02 for two
  steps): triggers at 1e4 m^2 on AP1 and GM4, 5e3 on GM1, 1e3 on GM2, 1e2 on AP5, GM3 and GM5,
  and finds no plateau on AP2, AP3, AP4. With half-decade ladder steps this rule is not robust.

Neither rule is good enough to run Phase 6 on without a judgement call. The model itself
applies the stream-power law only above its fitted A_crit (1.9e3-9.7e3 m^2 in the v1
joint fits), so a k_s measured over the hollow network (rollover rule) is not the quantity
the model's K0 should come from; a threshold in the 2e3-1e4 m^2 range is the one that is
consistent with where the model uses the channel law, but fixing it there uses the model's
own A_crit, which the addendum forbids as an input to the measurement.

**Decision for Steve** (needed before Phase 6, not before Phase 3): which A_c rule defines
the measured theta and k_s. Options: (a) a fixed site-independent threshold justified from
the slope-area curve and the channel-head literature (for example the area where the
binned slope-area relation becomes a straight power law, which I can add as a third rule);
(b) the rollover (hollow network); (c) the theta plateau with a finer ladder; (d) the
theta from the plateau and k_s evaluated at that theta over the whole network above the
rollover. My recommendation is (a) with the sensitivity ladder reported alongside, because
it keeps the measurement independent of the model fit while matching the model's channel
law to the fluvial network.

## 2. Values at the two rules (metres)

| tile | A_c roll (m^2) | theta roll | k_s roll | A_c plateau | theta plateau (68 %) | k_s plateau (68 %) | basins | C_ht (1/m) | L* (m) | D0 (m^2/yr) | v1 theta | v1 k_s (m) | v1 D0 (m) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 282 | 0.35 | 3.18 | 1e4 | 0.58 (0.10-0.58) | 56.8 (0.48-56.8) | 6 | 3.04e-3 | 50 | 1.81e-2 | 0.562 | 42.2 | 9.07e-3 |
| AP2 | 56 | 0.31 | 1.94 | none | - | - | - | 6.13e-3 | 20 | 8.97e-3 | 0.592 | 58.2 | 9.28e-3 |
| AP3 | 18 | 0.30 | 1.64 | none | - | - | - | 4.76e-3 | 40 | 1.15e-2 | 0.542 | 34.0 | 8.76e-3 |
| AP4 | 447 | 0.33 | 2.84 | none | - | - | - | 5.74e-3 | 40 | 9.59e-3 | 0.438 | 8.51 | 8.22e-3 |
| AP5 | 56 | 0.32 | 2.00 | 1e2 | 0.33 (0.28-0.34) | 2.27 (1.65-2.38) | 167 | 8.89e-3 | 20 | 6.18e-3 | 0.400 | 4.71 | 8.21e-3 |
| GM1 | 35 | 0.22 | 1.10 | 5e3 | 0.37 (0.36-0.42) | 5.92 (5.2-10.2) | 11 | 1.45e-2 | 9 | 1.01e-2 | 0.379 | 6.54 | 1.02e-2 |
| GM2 | 35 | 0.24 | 1.23 | 1e3 | 0.31 (0.29-0.33) | 2.67 (2.26-3.2) | 22 | 9.91e-3 | 11 | 1.48e-2 | 0.308 | 2.41 | 1.01e-2 |
| GM3 | 22 | 0.25 | 1.11 | 1e2 | 0.26 (0.23-0.28) | 1.27 (1.04-1.51) | 141 | 8.11e-3 | 15 | 1.81e-2 | 0.417 | 7.72 | 2.07e-2 |
| GM4 | 56 | 0.26 | 2.12 | 1e4 | 0.31 (0.21-0.33) | 4.00 (0.86-5.32) | 6 | 1.16e-2 | 21 | 1.27e-2 | 0.313 | 4.15 | 9.83e-3 |
| GM5 | 28 | 0.25 | 1.37 | 1e2 | 0.28 (0.27-0.29) | 1.97 (1.76-2.17) | 137 | 1.46e-2 | 9 | 1.00e-2 | 0.300 | 2.51 | 8.99e-3 |

v1 k_s and D0 for AP are the hard-coded feet-based values divided by 3.2808 (k_s and C_ht
scale with z; D0 = U/C_ht). The bootstrap intervals at the plateau thresholds on AP1 and
GM4 span the whole theta grid (lower bound 0.10): with 6 basins, resampling lands on
fragments whose chi-z fit is degenerate. Basins with fewer than 10 channel pixels are
already excluded; a minimum relief or chi range per basin would tighten this and should be
added before these intervals are quoted.

## 3. Hilltop curvature

The window-plateau rule chose L* = 9-21 m on GM and 20-50 m on AP (AP needs the longer
window because of the TIN facets). C_ht on GM (0.8-1.5e-2 1/m) brackets the v1 values
(0.7-1.6e-2, measured on 10 m block means); on AP the new C_ht (3.0-8.9e-3 1/m) is of the
same order as the v1 values converted to metres (5.9-6.7e-3) but ranges wider. D0 on GM is
1.0-1.8e-2 m^2/yr, on AP 0.6-1.8e-2 m^2/yr; the AP values are 1-2x the v1 metre-converted
values (0.6-0.9e-2). Hilltop pixel counts are 3100-13400 per tile, in 300-1100 segments.

## 4. The v1 method re-run on the metre grids

`legacy_v1_method_rerun` (the v1 chi method with tile-edge outlets and Taylor thresholds):
on GM it reproduces the v1 hard-coded theta and k_s **exactly** on all five tiles
(GM1 0.379 / 6.539, GM2 0.308 / 2.414, GM3 0.417 / 7.720, GM4 0.313 / 4.153,
GM5 0.300 / 2.507), which confirms both the port and the provenance found in Phase 0. On
AP it does not reproduce them (AP1: 0.575 / 52.3 against 0.562 / 42.2 after unit
conversion; theta is unit-free, so the difference is not the feet). The AP measurement
grids used in the legacy notebook were a different rasterization of the tiles (990 x 990
pixels at 0.9998 m, per the notebook output) from the model tiles (1015 x 1015 at 0.9752 m).
So the v1 AP parameters were measured on grids that are not the ones the models were
run on. This belongs in the Phase 6 old-versus-new record.

## 5. What this means for the plan

* Phase 3 (performance) and Phase 4 (optimizer v2) do not depend on these values.
* Phase 5 needs the measured values only for the invariance check (which uses v1 values).
* Phase 6 needs a decision on the threshold rule (section 1). Until then the measured
  theta, k_s, D0 are reported with their full threshold dependence and are not written
  into any tile config.
