## A1. Nonlinear rows: GM v2 profiles (decision 2) and the AP5 baseline (03b, pending row)

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

