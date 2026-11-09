## P1. Profile (step 1): wall time per operation on a full tile, cProfile on (adds overhead to Python loops)

| operation | GM1 (s) | AP5 (s) | peak RSS after (GB) | what dominates (self time) |
|---|---|---|---|---|
| hydro_fill | 23.3 | 23.8 | 0.1 | priority_flood_fill 0 s; <lambda> 0 s |
| hydro_d8_fd | 0.1 | 0.1 | 0.2 | <lambda> 0 s; d8_flow_direction 0 s |
| hydro_d8_area | 1.7 | 1.7 | 0.2 | d8_area 0 s; <lambda> 0 s |
| hydro_main_stem | 3.1 | 3.3 | 0.3 | full 0 s; <lambda> 0 s |
| ad_build | 169.6 | 170.6 | 4.3 | extract_grid_data 0 s; <lambda> 0 s |
| ad_solve | 135.1 | 136.5 | 32.1 | __rmul__ 0 s; <lambda> 0 s |
| ad_solve_2 | 136.2 | 140.6 | 32.1 | <lambda> 0 s; output_function 0 s |
| pt_build | 17.1 | 17.5 | 32.1 | extract_grid_data 0 s; <lambda> 0 s |
| pt_solve | 4.9 | 6.5 | 32.1 | where 0 s; output_function 0 s |
| pt_solve_2 | 4.9 | 5.4 | 32.1 | where 0 s; output_function 0 s |
| joint_rebuild_newK | 17.1 | 17.3 | 32.1 | extract_grid_data 0 s; <lambda> 0 s |
| nl_build | 13.4 | 13.7 | 32.1 | extract_grid_data 0 s; <lambda> 0 s |
| nl_solve_2iter | 14.9 | 19.7 | 32.1 | where 0 s; output_function 0 s |

Profile runs: GM1: `20261109T211900Z_profile_GM1`, AP5: `20261109T211902Z_profile_AP5`.

## G1. Forward gates: Phase 3 code against the Phase 2 v2 outputs (max |dz| / relief; rel. MSE diff), tolerances 1e-8 / 1e-10


### step2 (9d046e3): 10 tiles, 50 of 50 comparisons pass; max |dz|/relief 1.26e-12, max rel. MSE diff 2.51e-12

| tile | run | ad_no_opt | ad_opt | acrit_opt | nonlinear_opt | joint_opt | peak RSS |
|---|---|---|---|---|---|---|---|
| AP1 | 20261109T212205Z_tile_forward_AP1 | 1.8e-13 / 5.4e-14 | 3.5e-14 / 1.5e-15 | 6.1e-13 / 2.5e-12 | 1.7e-13 / 1.0e-13 | 4.4e-13 / 2.2e-13 | - |
| AP2 | 20261109T212207Z_tile_forward_AP2 | 1.1e-13 / 2.2e-14 | 1.9e-14 / 7.0e-15 | 1.9e-13 / 7.5e-13 | 2.5e-13 / 1.7e-12 | 7.3e-13 / 1.4e-14 | - |
| AP3 | 20261109T212205Z_tile_forward_AP3 | 3.8e-13 / 1.6e-13 | 2.4e-14 / 2.5e-14 | 1.0e-13 / 4.6e-14 | 3.5e-13 / 7.2e-14 | 1.3e-12 / 5.5e-13 | - |
| AP4 | 20261109T212205Z_tile_forward_AP4 | 9.4e-13 / 4.3e-13 | 1.8e-14 / 1.3e-14 | 2.7e-13 / 7.9e-13 | 5.7e-13 / 1.0e-12 | 2.5e-13 / 3.4e-14 | - |
| AP5 | 20261109T212205Z_tile_forward_AP5 | 5.1e-13 / 1.5e-13 | 1.6e-14 / 1.7e-15 | 7.4e-13 / 1.2e-12 | 4.3e-13 / 2.3e-13 | 7.4e-13 / 2.4e-13 | - |
| GM1 | 20261109T212205Z_tile_forward_GM1 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |
| GM2 | 20261109T212204Z_tile_forward_GM2 | 2.7e-13 / 4.8e-14 | 1.2e-14 / 2.1e-15 | 8.5e-14 / 3.7e-14 | 5.3e-14 / 1.8e-14 | 1.3e-13 / 1.8e-14 | - |
| GM3 | 20261109T212204Z_tile_forward_GM3 | 6.9e-13 / 2.4e-13 | 1.5e-14 / 8.6e-15 | 3.5e-13 / 1.5e-12 | 8.3e-14 / 1.1e-13 | 2.9e-13 / 3.3e-13 | - |
| GM4 | 20261109T212207Z_tile_forward_GM4 | 4.7e-13 / 7.2e-14 | 4.9e-15 / 1.5e-15 | 2.7e-14 / 4.3e-14 | 4.1e-14 / 2.5e-15 | 1.8e-13 / 2.1e-13 | - |
| GM5 | 20261109T212207Z_tile_forward_GM5 | 7.6e-13 / 1.4e-13 | 1.1e-14 / 6.1e-15 | 6.4e-14 / 2.0e-13 | 5.1e-14 / 2.0e-14 | 8.1e-14 / 1.2e-13 | 32.2 GB |

### step3 (898467c): 7 tiles, 35 of 35 comparisons pass; max |dz|/relief 7.32e-13, max rel. MSE diff 2.51e-12

| tile | run | ad_no_opt | ad_opt | acrit_opt | nonlinear_opt | joint_opt | peak RSS |
|---|---|---|---|---|---|---|---|
| AP1 | 20261109T213801Z_tile_forward_AP1_step3 | 1.7e-13 / 4.9e-14 | 3.8e-14 / 3.1e-14 | 6.1e-13 / 2.5e-12 | 1.7e-13 / 1.0e-13 | 4.4e-13 / 2.2e-13 | - |
| AP2 | 20261109T213800Z_tile_forward_AP2_step3 | 1.1e-13 / 2.3e-14 | 1.8e-14 / 1.5e-14 | 1.9e-13 / 7.5e-13 | 2.5e-13 / 1.7e-12 | 7.3e-13 / 1.4e-14 | - |
| AP3 | 20261109T213801Z_tile_forward_AP3_step3 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |
| AP4 | 20261109T213800Z_tile_forward_AP4_step3 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |
| AP5 | 20261109T213800Z_tile_forward_AP5_step3 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |
| GM1 | 20261109T213800Z_tile_forward_GM1_step3 | 3.7e-13 / 5.4e-14 | 9.5e-14 / 1.2e-14 | 1.2e-13 / 4.2e-13 | 8.5e-14 / 2.6e-14 | 6.2e-13 / 8.3e-13 | - |
| GM2 | 20261109T213800Z_tile_forward_GM2_step3 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |

### step4 (db11fee): 10 tiles, 50 of 50 comparisons pass; max |dz|/relief 1.26e-12, max rel. MSE diff 2.51e-12

| tile | run | ad_no_opt | ad_opt | acrit_opt | nonlinear_opt | joint_opt | peak RSS |
|---|---|---|---|---|---|---|---|
| AP1 | 20261109T213800Z_tile_forward_AP1_step4 | 1.7e-13 / 4.9e-14 | 3.8e-14 / 3.1e-14 | 6.1e-13 / 2.5e-12 | 1.7e-13 / 1.0e-13 | 4.4e-13 / 2.2e-13 | - |
| AP2 | 20261109T213800Z_tile_forward_AP2_step4 | 1.1e-13 / 2.3e-14 | 1.8e-14 / 1.5e-14 | 1.9e-13 / 7.5e-13 | 2.5e-13 / 1.7e-12 | 7.3e-13 / 1.4e-14 | - |
| AP3 | 20261109T213800Z_tile_forward_AP3_step4 | 3.8e-13 / 1.6e-13 | 2.5e-14 / 6.0e-14 | 1.0e-13 / 4.6e-14 | 3.5e-13 / 7.2e-14 | 1.3e-12 / 5.5e-13 | - |
| AP4 | 20261109T213800Z_tile_forward_AP4_step4 | 9.5e-13 / 4.3e-13 | 1.5e-14 / 3.7e-15 | 2.7e-13 / 7.9e-13 | 5.7e-13 / 1.0e-12 | 2.5e-13 / 3.4e-14 | - |
| AP5 | 20261109T213800Z_tile_forward_AP5_step4 | 4.9e-13 / 1.4e-13 | 1.9e-14 / 1.2e-14 | 7.4e-13 / 1.2e-12 | 4.3e-13 / 2.3e-13 | 7.4e-13 / 2.4e-13 | - |
| GM1 | 20261109T213800Z_tile_forward_GM1_step4 | 1.2e-14 / 1.9e-16 | 2.2e-14 / 9.9e-15 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |
| GM2 | 20261109T213800Z_tile_forward_GM2_step4 | 2.3e-13 / 3.3e-14 | 1.1e-14 / 3.6e-15 | 8.5e-14 / 3.7e-14 | 5.3e-14 / 1.8e-14 | 1.3e-13 / 1.8e-14 | - |
| GM3 | 20261109T213800Z_tile_forward_GM3_step4 | 6.9e-13 / 2.4e-13 | 1.7e-14 / 1.8e-14 | 3.5e-13 / 1.5e-12 | 8.3e-14 / 1.1e-13 | 2.9e-13 / 3.3e-13 | - |
| GM4 | 20261109T213801Z_tile_forward_GM4_step4 | 4.9e-13 / 7.9e-14 | 5.3e-15 / 0.0e+00 | 2.7e-14 / 4.3e-14 | 4.1e-14 / 2.5e-15 | 1.8e-13 / 2.1e-13 | - |
| GM5 | 20261109T213800Z_tile_forward_GM5_step4 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |

### step5 (a224750): 10 tiles, 50 of 50 comparisons pass; max |dz|/relief 1.26e-12, max rel. MSE diff 2.51e-12

| tile | run | ad_no_opt | ad_opt | acrit_opt | nonlinear_opt | joint_opt | peak RSS |
|---|---|---|---|---|---|---|---|
| AP1 | 20261109T214307Z_tile_forward_AP1 | 1.7e-13 / 4.9e-14 | 3.8e-14 / 3.1e-14 | 6.1e-13 / 2.5e-12 | 1.7e-13 / 1.0e-13 | 4.4e-13 / 2.2e-13 | - |
| AP2 | 20261109T214308Z_tile_forward_AP2 | 1.1e-13 / 2.3e-14 | 1.8e-14 / 1.5e-14 | 1.9e-13 / 7.5e-13 | 2.5e-13 / 1.7e-12 | 7.3e-13 / 1.4e-14 | - |
| AP3 | 20261109T214303Z_tile_forward_AP3 | 3.8e-13 / 1.6e-13 | 2.5e-14 / 6.0e-14 | 1.0e-13 / 4.6e-14 | 3.5e-13 / 7.2e-14 | 1.3e-12 / 5.5e-13 | - |
| AP4 | 20261109T214303Z_tile_forward_AP4 | 9.5e-13 / 4.3e-13 | 1.5e-14 / 3.7e-15 | 2.7e-13 / 7.9e-13 | 5.7e-13 / 1.0e-12 | 2.5e-13 / 3.4e-14 | - |
| AP5 | 20261109T214305Z_tile_forward_AP5 | 4.9e-13 / 1.4e-13 | 1.9e-14 / 1.2e-14 | 7.4e-13 / 1.2e-12 | 4.3e-13 / 2.3e-13 | 7.4e-13 / 2.4e-13 | - |
| GM1 | 20261109T214305Z_tile_forward_GM1 | 1.2e-14 / 1.9e-16 | 2.2e-14 / 9.9e-15 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |
| GM2 | 20261109T214311Z_tile_forward_GM2 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |
| GM3 | 20261109T214304Z_tile_forward_GM3 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |
| GM4 | 20261109T214303Z_tile_forward_GM4 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | - |
| GM5 | 20261109T214303Z_tile_forward_GM5 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 0.0e+00 / 0.0e+00 | 3.5 GB |

## G2. Wall time per forward evaluation (s) by step; ad_no_opt includes the AD build, nonlinear_opt runs up to 50 Picard iterations

| step | tile | ad_no_opt | ad_opt | acrit_opt | nonlinear_opt | joint_opt |
|---|---|---|---|---|---|---|
| step2 | AP1 | 169 | 171 | 12 | 353 | 13 |
| step2 | AP2 | 169 | 168 | 11 | 334 | 13 |
| step2 | AP3 | 170 | 167 | 11 | 360 | 13 |
| step2 | AP4 | 170 | 170 | 10 | 377 | 14 |
| step2 | AP5 | 167 | 169 | 10 | 311 | 11 |
| step2 | GM1 | 163 | 164 | 10 | 51 | 12 |
| step2 | GM2 | 162 | 163 | 8 | 34 | 11 |
| step2 | GM3 | 161 | 162 | 10 | 36 | 11 |
| step2 | GM4 | 161 | 162 | 10 | 57 | 12 |
| step2 | GM5 | 166 | 167 | 9 | 43 | 11 |
| step3 | AP1 | 26 | 26 | 12 | 324 | 13 |
| step3 | AP2 | 28 | 28 | 11 | 351 | 13 |
| step3 | AP3 | 23 | 23 | 10 | 297 | 12 |
| step3 | AP4 | 26 | 26 | 13 | 382 | 16 |
| step3 | AP5 | 25 | 24 | 13 | 322 | 14 |
| step3 | GM1 | 24 | 24 | 12 | 54 | 14 |
| step3 | GM2 | 25 | 25 | 11 | 39 | 14 |
| step4 | AP1 | 26 | 26 | 12 | 330 | 12 |
| step4 | AP2 | 27 | 27 | 11 | 339 | 13 |
| step4 | AP3 | 28 | 28 | 11 | 379 | 13 |
| step4 | AP4 | 28 | 28 | 10 | 391 | 14 |
| step4 | AP5 | 26 | 26 | 10 | 306 | 11 |
| step4 | GM1 | 26 | 26 | 10 | 50 | 11 |
| step4 | GM2 | 27 | 27 | 8 | 34 | 12 |
| step4 | GM3 | 25 | 25 | 10 | 36 | 11 |
| step4 | GM4 | 25 | 24 | 10 | 52 | 11 |
| step4 | GM5 | 19 | 20 | 8 | 34 | 9 |
| step5 | AP1 | 26 | 25 | 6 | 316 | 6 |
| step5 | AP2 | 28 | 27 | 6 | 333 | 7 |
| step5 | AP3 | 29 | 28 | 5 | 371 | 8 |
| step5 | AP4 | 28 | 27 | 5 | 377 | 8 |
| step5 | AP5 | 25 | 24 | 5 | 294 | 5 |
| step5 | GM1 | 25 | 24 | 5 | 43 | 5 |
| step5 | GM2 | 21 | 20 | 3 | 23 | 4 |
| step5 | GM3 | 20 | 20 | 4 | 25 | 4 |
| step5 | GM4 | 24 | 24 | 5 | 49 | 6 |
| step5 | GM5 | 24 | 24 | 4 | 39 | 5 |

## O1. Optimizer stages: section 3 baseline (Phase 2 code) and Phase 3 code, against v1 (parameters within 1 %, MSE within 0.5 %)

| tile | stage | code | run | params | rel. diff vs v1 | MSE | rel. diff vs v1 | wall (s) | v1 wall (s) | fev / solves | pass |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AP5 | acrit | Phase 3 | 20261109T214309Z_tile_run_AP5_acrit | A_crit=1130.04 | A_crit +3.9e-05 | 359.26 | +1.8e-06 | 136 | 1323 | 34 / 29 | yes |
| AP5 | acrit | baseline | 20261109T211901Z_tile_run_AP5_acrit | A_crit=1130.04 | A_crit +3.9e-05 | 359.26 | +1.8e-06 | 119 | 1323 | 34 / 29 | yes |
| GM1 | acrit | Phase 3 | 20261109T214306Z_tile_run_GM1_acrit | A_crit=423.493 | A_crit +8.1e-06 | 100.96 | +2.4e-05 | 122 | 1480 | 35 / 32 | yes |
| GM1 | acrit | baseline | 20261109T211103Z_tile_run_GM1 | A_crit=423.493 | A_crit +8.1e-06 | 100.96 | +2.4e-05 | 853 | 1480 | 35 / 32 | yes |
