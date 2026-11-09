# Phase 0 checkpoint: freeze and capture the v1 reference

Directive: `reports/directives/refactor_v2_directive.md`, Phase 0.
Branch: `v2`, cut from tag `v1.0.1` (commit 28a9c90) on 2026-11-09.
Working clone: `~/LEMs_competition_vs_threshold-JGR-branch` on Sherlock (login node sh03-ln04; the manifest, environment build and test run executed in Slurm jobs 51674944 and 51675344 on the default `normal` partition, 1.5 min + 8.5 min, MaxRSS 4.5 GB).
Status: all six Phase 0 steps done. Nothing pushed. Phase 1 has not started.

## 0. Decisions and approvals needed from Steve

1. **Push the local branches as they are.** Four branches exist only on Sherlock (section 1). The directive asks that this be the first approval request. Command I would run after approval: `git push -u origin feature/channel-width feature/acrit-slope feature/head-extraction v2`.
2. **Slurm config values** for `configs/slurm/sherlock.yaml`: partition, account, QoS. Sherlock has no `--account` (per `/etc/agents/slurm.md`); the public partitions are `normal`, `bigmem`, `dev`; the group also has `serc` and `owners`. The Phase 0 job used the default partition (`normal`) with no QoS. I did not guess values for the config.
3. **The superseded AP-units directive** (`reports/directives/ap_vertical_units_fix_directive.md`) and the branch `fix/ap-vertical-units` do not exist on Sherlock. Its "inventory and corrections-table requirements" cannot be carried over from the file. Please send the file, or confirm that Phase 5 and the Phase 6 `corrections_table.md` as written in the current directive are the full requirement.
4. **Credential exposure (not part of the directive).** `$GROUP_HOME/scdobbs_optimize/landscape-optimizer/.git/config` stores the GitHub remote URL with an embedded personal access token for user `gehilley`, in a group-readable directory. I did not use it. It should be revoked and the remote URL rewritten (`git remote set-url origin https://github.com/scdobbs/landscape-optimizer.git`).
5. **Two v1 facts that change later phases** (details in sections 3.4 and 3.5): the 14-hour stage is the nonlinear A_crit stage, not the joint stage; and AP3's partitioned stages ran with K = 1e-6 rather than K = U/k_s. Neither is fixed here. They need a decision at Phase 1 (reproduce as-run) and Phase 3 (what to optimize first).

## 1. Branch inventory (step 1)

All branches share the single commit on `main`. Nothing on any branch has been pushed except `main`.

| Branch | Last commit | Date | Commits ahead of main | On remote |
| --- | --- | --- | --- | --- |
| `main` | 28a9c90 ran slope-area notebook | 2026-02-13 | 0 | yes (`origin/main`) |
| `feature/channel-width` | eb08cac Constrained full run (35 AP + 50 GM points) … final report | 2026-11-02 | 18 | no |
| `feature/acrit-slope` | 643f8b4 acrit_slope: A_crit vs head slope for GM tiles 1-4 only | 2026-11-04 | 24 | no |
| `feature/head-extraction` | 11767b6 head_extraction: quick slope-area look at the extracted hollow-class heads | 2026-11-04 | 35 | no |
| `v2` (new) | 52a3418 Phase 0: v1 reference extraction … | 2026-11-09 | 1 | no |

Tags `v1.0.0` and `v1.0.1` both point to 28a9c90. Remote: `https://github.com/scdobbs/LEMs_competition_vs_threshold`.

Untracked files in the working tree, on no branch: `channel_width_directive.md`, `working-with-steve.md`, `cry_for_help.txt`, `papers/`, `outputs/widths/pilot_v1_superseded/`, `outputs/widths/qc_all/`, `outputs/widths_constrained/qc_all/`, and three directive files in `reports/directives/` (`acrit_slope_test1_directive.md`, `channel_width_constrained_directive.md`, `head_extraction_test2_directive.md`). They survive branch switches but exist only on Sherlock's disk. The refactor directive itself is now committed on `v2`.

Also present in group storage, outside the repo: a second clone at `$GROUP_HOME/scdobbs_optimize/landscape-optimizer/LEMs_competition_vs_threshold` (at 28a9c90, Feb 2026), and the tile grids copied under `landscape-optimizer/new_models_101625/{allegheny,gabilan}/data/tile*/` (Oct–Dec 2025). Their sha256 values are compared with the repo grids in section 4.

## 2. Branch `v2` (step 2)

Created with `git checkout -b v2 v1.0.1` in the working clone. Nothing cherry-picked. The first commit (52a3418) adds only Phase 0 material: `reference/`, `envs/lem-v1*.yml`, the directive, and a `.gitignore` exception for `reference/logs/*.log`.

## 3. `reference/v1_reference.json` (step 3)

Script: `reference/extract_v1_reference.py` (stdlib only; parses notebook JSON, computes nothing). It identifies stages by the optimizer function called in each cell, not by cell index (AP5's notebook has one extra cell, so indices shift).

### 3.1 Hard-coded inputs per tile

`*` marks an AD-stage K0 override. K0 for the partitioned stages is the value in scope when `partitioned_threshold_model` was built.

| Tile | θ | k_s | C_ht | U | K0 (AD stage) | K0 (partitioned stages) | D0 | K bounds | D bounds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 0.562 | 138.555 | 1.990e-02 | 5.50e-05 | 1.000e-06* | 3.970e-07 | 2.764e-03 | 1e-12–2e-03 | 4e-10–1e-02 |
| AP2 | 0.592 | 190.918 | 1.944e-02 | 5.50e-05 | 2.881e-07 | 2.881e-07 | 2.829e-03 | 1e-12–2e-02 | 4e-07–5e-02 |
| AP3 | 0.542 | 111.454 | 2.060e-02 | 5.50e-05 | 1.000e-06* | **1.000e-06** | 2.670e-03 | 1e-12–2e-03 | 4e-10–1e-02 |
| AP4 | 0.438 | 27.907 | 2.194e-02 | 5.50e-05 | 1.971e-06 | 1.971e-06 | 2.507e-03 | 1e-12–2e-02 | 4e-07–5e-02 |
| AP5 | 0.400 | 15.468 | 2.199e-02 | 5.50e-05 | 1.000e-05* | 3.556e-06 | 2.501e-03 | 1e-12–2e-03 | 4e-10–1e-02 |
| GM1 | 0.379 | 6.539 | 1.442e-02 | 1.47e-04 | 2.248e-05 | 2.248e-05 | 1.019e-02 | 7e-12–2e-02 | 4e-10–5e-02 |
| GM2 | 0.308 | 2.414 | 1.450e-02 | 1.47e-04 | 6.089e-05 | 6.089e-05 | 1.014e-02 | 7e-07–2e-02 | 4e-05–5e-02 |
| GM3 | 0.417 | 7.720 | 7.110e-03 | 1.47e-04 | 1.904e-05 | 1.904e-05 | 2.068e-02 | 7e-07–2e-02 | 4e-05–5e-02 |
| GM4 | 0.313 | 4.153 | 1.496e-02 | 1.47e-04 | 3.540e-05 | 3.540e-05 | 9.826e-03 | 7e-07–2e-02 | 4e-05–5e-02 |
| GM5 | 0.300 | 2.507 | 1.635e-02 | 1.47e-04 | 5.864e-05 | 5.864e-05 | 8.991e-03 | 7e-07–2e-02 | 4e-05–5e-02 |

Common to all tiles: n = 1, m = θ, S_c = 1.27, A_crit₀ = 5000 m², A_crit bounds (1e2, 1e5) m², A_crit-stage D fixed at D0, nonlinear stage D fixed at D0.

### 3.2 Optimizer results as printed (4 significant figures)

| Tile | AD K | AD D | AD MSE | A_crit (stage 2) | MSE | A_crit (nonlinear) | MSE | A_crit (joint) | D (joint) | K (joint) | MSE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| AP1 | 8.2783e-06 | 5.4462e-04 | 3.7296e+02 | 3.0634e+03 | 1.9162e+03 | 4.9503e+03 | 2.0286e+02 | 6.4143e+03 | 5.3801e-03 | 4.0679e-07 | 2.2602e+02 |
| AP2 | 6.9594e-06 | 6.6607e-04 | 3.8149e+02 | 2.3378e+03 | 4.2274e+03 | 6.1659e+03 | 6.3032e+02 | 7.6456e+03 | 7.0857e-03 | 3.5072e-07 | 2.3350e+02 |
| AP3 | 1.1150e-05 | 6.6442e-04 | 1.2379e+03 | 1.7221e+03 | 1.3560e+03 | 1.0233e+04 | 6.5085e+02 | 9.7415e+03 | 6.9050e-03 | 6.3145e-07 | 5.0523e+02 |
| AP4 | 1.6361e-05 | 2.1211e-04 | 5.7825e+02 | 9.9501e+02 | 5.9797e+02 | 5.2635e+03 | 3.1060e+02 | 7.7979e+03 | 6.1006e-03 | 2.0368e-06 | 3.7005e+02 |
| AP5 | 1.5614e-05 | 4.3158e-05 | 1.0053e+03 | 1.1300e+03 | 3.5926e+02 | 3.3991e+03 | 2.6138e+02 | 1.8870e+03 | 3.6291e-03 | 3.4402e-06 | 2.8962e+02 |
| GM1 | 4.4610e-04 | 2.8844e-02 | 1.6865e+02 | 4.2349e+02 | 1.0096e+02 | 4.7421e+02 | 7.7073e+01 | 8.0924e+03 | 2.8368e-02 | 2.1649e-05 | 3.3865e+01 |
| GM2 | 1.3115e-04 | 8.2981e-05 | 8.1956e+01 | **1.0000e+02** | 5.7873e+01 | 2.4459e+02 | 4.7533e+01 | 2.9600e+03 | 2.6414e-02 | 5.5982e-05 | 2.7827e+01 |
| GM3 | 1.7613e-04 | 2.3441e-03 | 7.6187e+01 | **1.2974e+03** | 6.3351e+01 | **1.2974e+03** | 5.8985e+01 | 6.5263e+03 | 4.5329e-02 | 1.9131e-05 | 4.3018e+01 |
| GM4 | 9.4605e-05 | 6.7843e-05 | 1.7539e+02 | 4.2209e+02 | 8.5526e+01 | 5.2138e+02 | 7.2773e+01 | 7.3962e+03 | 2.6549e-02 | 3.5113e-05 | 4.7730e+01 |
| GM5 | 1.5658e-04 | 1.8605e-04 | 9.3177e+01 | 2.6142e+02 | 6.1022e+01 | 2.9962e+02 | 5.4813e+01 | 2.2471e+03 | 2.0284e-02 | 5.8231e-05 | 3.3054e+01 |

Both symptoms named in the directive are confirmed in the printed numbers: GM2's stage-2 A_crit is exactly the lower bound (1.0000e+02), and GM3's stage-2 and nonlinear A_crit coincide (1.2974e+03). Two further things the table shows:

- On AP1, AP2 and AP3 the stage-2 (A_crit only, K = K0, D = D0) misfit is worse than the AD optimum (1916 vs 373; 4227 vs 381; 1356 vs 1238). The nonlinear and joint stages recover.
- On AP5 the joint A_crit (1887 m²) is below the nonlinear A_crit (3399 m²); on every other tile the joint A_crit is the largest of the three.

### 3.3 Forward runs and saved grids

Each notebook performs 11 forward runs, 4 of which are saved (prefixes `*_AD-no-opt`, `*_AD-opt`, `*_Partitioned-model-a_crit-opt`, `*_Partitioned-model-a_crit-k-d-opt`; each saved as elevation, filled, flow_direction, area, logarea, hillshade). The unsaved runs are the partitioned model at A_crit = 5000, 1000, 500, 100 m² (K0, D0) and the nonlinear model at A_crit = 1000, 10000 m² and at its optimum. **The v1 notebooks print no forward-run MSE.** The reference MSE for each saved run is therefore recomputed from the saved float64 grids in `reference/v1_saved_grid_mse.csv` (section 4). The nonlinear optimum is not saved as a grid, so its only reference is the printed MSE.

### 3.4 Wall time per stage (papermill cell durations, hours)

| Tile | AD (L-BFGS-B) | A_crit stage 2 (Nelder–Mead) | Nonlinear A_crit (Nelder–Mead + Picard) | Joint (Nelder–Mead, rebuilds network) | Notebook total |
| --- | --- | --- | --- | --- | --- |
| AP1 | 3.40 | 0.26 | **13.69** | 1.83 | 20.9 |
| AP2 | 3.57 | 0.20 | **11.06** | 1.63 | 17.8 |
| AP3 | 4.86 | 0.40 | **24.54** | 2.38 | 34.7 |
| AP4 | 8.77 | 0.38 | **27.59** | 3.12 | 42.5 |
| AP5 | 7.98 | 0.37 | **20.74** | 3.72 | 35.2 |
| GM1 | 4.27 | 0.41 | **5.24** | 1.72 | 12.5 |
| GM2 | 4.08 | 0.07 | 3.22 | 1.42 | 9.5 |
| GM3 | 6.16 | 0.37 | 3.10 | 2.33 | 12.6 |
| GM4 | 6.44 | 0.41 | 6.19 | 2.68 | 16.8 |
| GM5 | 5.17 | 0.36 | 5.18 | 3.21 | 14.9 |

The directive's Context says the joint K–D–A_crit stage "takes about 14 hours" and attributes it to rebuilding the channel network per K. The recorded durations contradict this: the joint stage takes 1.4–3.7 h per tile; the nonlinear A_crit stage takes 3–28 h (13.7 h on AP1, the tile the directive cites). Known defect 2 (network rebuilt per K) is real but costs about 2 h, not 14. The dominant cost is the nonlinear model: each misfit evaluation runs up to 50 Picard iterations, each of which rebuilds the reduced hillslope matrix in a per-cell Python loop over about 10⁶ cells (`_solve_hillslope_diffusion`) and computes slopes in another per-cell loop (`_compute_slopes`). This should move the Phase 3 priority to vectorized assembly and a factorization reused across Picard iterations, ahead of the χ precomputation for the joint stage. The AD stage (3.4–8.8 h) is the second cost; each L-BFGS-B evaluation does a CSR→LIL→CSR round trip (known defect 3).

### 3.5 Inconsistencies in how v1 was run (reported, not fixed)

1. **AP3 partitioned stages used K = 1e-6, not U/k_s.** AP1, AP3 and AP5 override K0 for the AD stage (1e-6, 1e-6, 1e-5). AP1 and AP5 re-instate `K0, D0 = (U/k_s)^(1/n), U/C_ht` before building the partitioned model; AP3's notebook does not, so its stage-2 and nonlinear stages, and the saved `AP_1m_best_tile_Partitioned-model-a_crit-opt` grid, used K = 1.000e-06 against a measured U/k_s = 4.935e-07. Phase 2 must reproduce this as run (the tile config needs a per-stage K0 field or an explicit AP3 override) and Phase 6 should decide whether the corrected rerun uses U/k_s on AP3.
2. **Bounds differ by tile group.** AP{1,3,5}: K (1e-12, 2e-3), D (4e-10, 1e-2). AP{2,4}: K (1e-12, 2e-2), D (4e-7, 5e-2). GM1: K (7e-12, 2e-2), D (4e-10, 5e-2). GM{2–5}: K (7e-7, 2e-2), D (4e-5, 5e-2). The tile configs must carry these as-is for Phase 2; the directive's Phase 5.5 bound scaling then applies per tile.
3. **Figure notebooks do not use the optimized parameters.** `supp_fig6` and `resolution_study_dinf` hard-code `KD_GM`/`KD_AP` and `THETA_*` tables; these are the initial values K0 = U/k_s and D0 = U/C_ht rounded to 2 significant figures (for AP3 the un-overridden 4.9e-7), not the AD or joint optima. `d8_vs_dinf_comparison` uses the same K0, D0 for AP1 and GM1 with `A_crit = 5000` ("approximate from optimization"; the joint optima are 6414 and 8092 m²). Its MSE table (AP1: AD D8 6.9581e5, D-inf 7.0133e5, PT D8 2.1950e3, PT D-inf 3.7650e6; GM1: 3.2286e5, 3.1845e5, 7.9204e2, 4.2395e3) is therefore a comparison at measured, not fitted, parameters. The Phase 2.3 regression reproduces these as printed; whether Phase 6 should rerun them at fitted parameters is a decision for Steve.
4. **`fig_slope_area` does not reproduce the hard-coded θ and k_s.** Its binned regression on the observed grids gives AP1: k_s = 112.8, θ = 0.542 (hard-coded 138.555, 0.562); GM2: k_s = 2.488, θ = 0.306 (hard-coded 2.414, 0.308). θ agrees within 4 %, k_s within 3 % on GM2 but 19 % off on AP1. So this notebook is not the measurement method behind the hard-coded values, and it uses undocumented `min_area` thresholds (AP 8.89e3, GM 5.49e3 m²). This matters for Phase 5.3's 5 % regression requirement on GM.

### 3.6 Figure-notebook tables captured

| Notebook | Captured | Records |
| --- | --- | --- |
| `fig_slope_area` | k_s, θ, R² for ORIGINAL / AD no-opt / AD opt / PARTITIONED on AP1 and GM2 | 8 |
| `supp_fig6` | `[OK] <site> T<n> dx=… Ks_mod Ks_obs ratio n_mod n_obs` for 10 tiles × 25 dx | 250 |
| `resolution_study_dinf` | `[OK] … ratio_d8 ratio_dinf` for 10 tiles × 25 dx | 250 |
| `d8_vs_dinf_comparison` | shapes, D8 and D-inf area ranges, four MSEs per site | 2 sites |
| `supp_fig5` | `[primary]` anchors (D_bal = 4.402e-2, A_inlet_bal = 2.590e8, Pe = 100), trim bookkeeping (25 rows), median k_s/(U/K) with IQR by dx (25 rows) | 56 |

Raw printed text of every cell is kept alongside the parsed values.

## 4. Grid manifest (step 4)

Script: `reference/make_grid_manifest.py`, run in jobs 51674944 and 51675344 with the existing `lem-widths` Python (rasterio); both runs produced byte-identical CSVs. Outputs: `reference/v1_grids_manifest.csv` (280 grids) and `reference/v1_saved_grid_mse.csv`.

Format finding: every grid in `data/` is already a GeoTIFF (TopoAnalysis saves through GDAL's GTiff driver; the files simply lack an extension). Phase 1's "TopoAnalysis → GeoTIFF converter" reduces to adding the extension, confirming dtype and nodata, and recording the manifest; the bit-identical round trip is then trivial to verify.

Georeference (from the GeoTIFF headers):

| Site | CRS | Shape | Pixel size | z range (filled, tile 1) |
| --- | --- | --- | --- | --- |
| AP | EPSG:26917 (NAD83 / UTM 17N, metres) | 1015 × 1015 | 0.97520 × 0.97513 m (not square) | 1064.75 – 1440.67 |
| GM | EPSG:32610 (WGS84 / UTM 10N, metres) | 1000 × 1000 | 1.0 × 1.0 m | 295.24 – 424.74 |

The AP z range (≈1065–1441 for a Greene County, PA tile, where the ground sits at roughly 300–450 m) is consistent with the directive's statement that AP elevations are in US survey feet with metre x, y. The AP pixels are 0.9752 × 0.97513 m; TopoAnalysis uses only `geoTransform[1]` as `dx` and the model code sets `dy = dx`, so the 7e-5 relative anisotropy is ignored in v1. This is recorded for Phase 1's unit loader; it is far below any tolerance that matters.

Manifest results (`reference/v1_grids_manifest.csv`, 270 rows, 0 read errors):

- 270 grids = 10 tiles × (3 inputs + 24 outputs). Inputs: `_filled` and `_area` float64, `_fd` uint8. Outputs per saved run: `_elevation`, `_filled`, `_area` float64; `_flow_direction`, `_logarea`, `_hillshade` uint8. No grid has a nodata value set and no grid contains NaN.
- Shapes: GM 1000 × 1000 for all five tiles. AP 1015 × 1015 for tiles 1, 2, 4, 5 and **1015 × 1014 for AP3** (pixel 0.975175 × 0.975130 m there, 0.975200 × 0.975130 m elsewhere). Tile configs must carry the shape per tile.
- CRS: the 30 input grids carry EPSG:26917 (AP) or EPSG:32610 (GM). **All 240 output grids have no CRS** (TopoAnalysis's `save` writes the geotransform but no projection for grids created in memory). The geotransforms of outputs match their inputs. Phase 1's GeoTIFF writer should set the CRS from the tile input.
- Drainage-area units: minimum cell area is 0.9507 m² on AP (= 0.9752 × 0.97513) and 1.0 m² on GM, so area is in m² on both sites, consistent with metre x, y. Maximum areas (AP3 1.10e7, AP4 1.70e7, GM2 8.89e6 m²) exceed the tile footprint (~1 km²), so the area grids inherit upstream area from outside the tiles.
- Elevation ranges of the filled inputs (units as stored): AP1 1064.75–1440.67, AP2 1053.62–1520.91, AP3 1076.95–1486.22, AP4 1043.21–1456.01, AP5 1088.94–1395.45; GM1 295.24–424.74, GM2 246.29–370.66, GM3 216.34–304.50, GM4 134.87–273.40, GM5 282.43–419.90. Tile relief (used by the Phase 2 tolerance 1e-6 × relief): AP 307–467, GM 88–139.
- Provenance: all 30 input grids are byte-identical (sha256) to the copies in `$GROUP_HOME/scdobbs_optimize/landscape-optimizer/new_models_101625/{allegheny,gabilan}/data/tile*/` (Oct–Dec 2025).

Saved-grid MSE (`reference/v1_saved_grid_mse.csv`, 40 rows): for the 30 saved optimized grids, the MSE recomputed from the float64 grids against the filled input agrees with the printed optimizer MSE to within 4e-5 relative in every case (the printed values carry 4–5 significant figures), which confirms that the saved grids are the optimizer's final forward solutions and that the misfit definition (`nanmean((Z - Z_obs)²)` over all cells, boundary included) is reproduced. The ten `AD-no-opt` grids (K0, D0 forward run) have no printed counterpart; their reference MSEs are AP 6.37e5–7.58e5 and GM 7.39e4–4.38e5 (units², as stored). Max |Δz| between saved and observed is 20–195 for the optimized runs and 509–1718 for AD-no-opt.

## 5. The v1 environment `lem-v1` (step 5)

Spec: `envs/lem-v1.yml` (strict) with `envs/lem-v1-relaxed.yml` as fallback. Built with micromamba into `$GROUP_HOME/scdobbs-envs/lem-v1` in job 51675344. The macOS export pins 200 packages including Jupyter, Qt and macOS-only libraries; the Linux rebuild keeps the versions that touch numerics (python 3.10.12, numpy 1.26.4, scipy 1.15.3, numba 0.61.2, llvmlite 0.44.0, scikit-learn 1.3.2, statsmodels 0.14.0, pandas 2.1.1, matplotlib 3.6.2, scikit-image 0.25.2) and lets GDAL/rasterio float (I/O only). Package versions as resolved are in `reference/logs/lem-v1_packages.txt`.

Build history: the first attempt (job 51674944, `reference/logs/lem-v1_build_attempt1_failed.log`) failed in the solver because conda-forge has no scipy 1.15.3 build (only 1.15.2) and its scikit-image 0.25.2 recipe requires matplotlib ≥ 3.7. The macOS export lists scipy, numba, llvmlite, scikit-learn, statsmodels and scikit-image under `pip:`, so the second spec takes exactly those from PyPI, as v1 did. That solved and installed in 7.5 min (job 51675344).

Resolved environment (`reference/logs/lem-v1_packages.txt`, 241 packages, 1.9 GB):

| Package | lem-v1 | v1 (environment.yml) | Source |
| --- | --- | --- | --- |
| python | 3.10.12 | 3.10.12 | conda-forge |
| numpy | 1.26.4 | 1.26.4 | conda-forge (OpenBLAS 0.3.34; v1 used openblas 0.3.21) |
| scipy | 1.15.3 | 1.15.3 (pip) | PyPI wheel |
| pandas / matplotlib | 2.1.1 / 3.6.2 | 2.1.1 / 3.6.2 | conda-forge |
| numba / llvmlite | 0.61.2 / 0.44.0 | same (pip) | PyPI |
| scikit-learn / statsmodels / scikit-image | 1.3.2 / 0.14.0 / 0.25.2 | same (pip) | PyPI |
| gdal / rasterio | 3.12.3 / 1.4.4 | 3.6.2 / 1.4.3 | conda-forge (I/O only) |
| richdem | 2.3.0 | 2.3.0 | conda-forge |

The only numerically relevant difference from v1 is the BLAS build behind numpy (OpenBLAS 0.3.34 Linux vs 0.3.21 macOS) and the CPU kernels it selects; `scipy.sparse.linalg.spsolve` (SuperLU) is bundled in the scipy wheel on both platforms. Differences at the 1e-13 relative level are possible; nothing at the Phase 2 tolerances (1e-8 on MSE) is expected from this.

Test suite in `lem-v1`: `pytest tests/` → **105 passed, 247 warnings, 8.0 s** (`reference/logs/pytest_lem-v1.log`), matching the count in the directive and the README. The warnings are matplotlib deprecations raised by `tests/test_plotting.py` and a GDAL 4.0 exceptions notice.

TopoAnalysis: imported from `$GROUP_HOME/opt/TopoAnalysis` (files dated 2022-10-15, no git metadata; sha256 of each module in `reference/logs/topoanalysis_sha256.txt`) via `PYTHONPATH=$GROUP_HOME/opt`. It loads the repo GeoTIFFs (`reference/logs/topoanalysis_check_lem-v1.log`): for GM1 `dx = 1.0`, for AP3 `dx = 0.9751752` with `nx, ny = 1014, 1015`, i.e. `Georef_info.dx` is `geoTransform[1]` and the loader keeps no projection, which is why every saved output grid lacks a CRS (section 4). The 1 m GM grids therefore run with dx = 1.0 and the AP grids with dx = 0.9752 (AP3: 0.97518) in v1.

## 6. Missing measurement code (step 6)

Searched, within directories the user owns or that the group shared with them: `$GROUP_HOME/scdobbs_optimize/landscape-optimizer` (all notebooks' code cells, `.py` files, git history including deleted files), `$GROUP_HOME/scdobbs_hilley_lem`, `$GROUP_HOME/scdobbs_energy`, `$GROUP_HOME/opt/{TopoAnalysis,pylem}`, `$HOME` notebooks (`K_and_D_adjustments.ipynb`, `DEM Sampler*.ipynb`, `Re-saving_grd_files*.ipynb`, `CalculateHackScaling*.ipynb`, `Untitled*.ipynb`), `$SCRATCH/manuscript_package` (empty subdirectories), `$SCRATCH/lr_grids_v3`. `$OAK` has no directory for this project. `scdobbs_minimize_unit_stream_power` (4665 entries of synthetic-model checkpoints from 2022–2024) was not scanned recursively.

Result: **no script or notebook that measures θ, k_s or hilltop curvature on the ten tiles, and none that cuts the tiles from the mosaics, exists on Sherlock.** What does exist:

- `landscape-optimizer/` (June 2025): the predecessor of the model-run notebooks (`GabilanExploration*.ipynb`, `AlleghenyExploration*.ipynb`) working on coarse clips (`GM_10m_clip`, `AP_30m_clip`, `GM_small`, `Greene_small_reproject2`) with Perron et al. (2012) parameters typed in; the git history shows deleted source rasters `data/AP_30m.tif` and `data/GM_10m_clip.tif` (commit 1748693, 2025-06-12) but no 1 m mosaics.
- `landscape-optimizer/new_models_101625/` (Oct 2025): per-tile notebooks with the same hard-coded θ, k_s, C_ht values as v1, and copies of the ten tiles' input grids (sha256 comparison in section 4).
- The executed figure notebooks record the project root `/Users/scdobbs/Documents/GitHub/Manuscript_submission` on Steve's Mac; that is where the measurement notebooks and the 1 m mosaics most likely are.
- `fig_slope_area` is the only slope–area code in the repo and does not reproduce the hard-coded k_s (section 3.5, item 4).

Consequence for Phase 5.3: unless Steve locates the measurement notebooks, the GM regression requirement ("re-measured values must reproduce the hard-coded ones within 5 %") is a test of whether `lemcvt measure` happens to match an unknown method. The directive's fallback ("report the difference and stop") would then trigger on GM. I recommend Steve look for the original notebooks before Phase 5.

## 7. Other observations

- Storage: `$HOME` is at 10.9 of 15 GB (72 %); the clone's 1.2 GB of grids are the largest item. Moving grids to group storage in Phase 1 (directive requirement 1.1) also relieves this. `lem-v1` and future environments go to `$GROUP_HOME/scdobbs-envs/` (established layout).
- `$GROUP_HOME` is at 256 GB of 1 TB; `$OAK` at 16.5 of 20 TB (82 %), so Oak has room for the ~1.2 GB grid archive but not for large new datasets.
- The v1 run logs (`data/*/output_files/*.out`) show the notebooks were executed with papermill in a `gpm` conda environment on `serc` (12 CPUs, 200 GB, 2 days) from the second clone in `landscape-optimizer/`, not from this clone.

## 8. What happens next

Phase 0 is complete with this report. I stop here for review. Phase 1 (refactor, numerics unchanged) starts only after Steve's go-ahead and the answers to section 0. The first Phase 1 actions would be: `pyproject.toml` + `envs/sherlock.yml` (Python 3.11/3.12, lean), tile configs carrying every value in section 3.1 including the AP3 K0 quirk and per-tile shapes, and the GeoTIFF I/O layer with the manifest check.
