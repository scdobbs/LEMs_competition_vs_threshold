# v1 notebooks to v2 commands

Every v1 notebook is kept unchanged under `legacy/notebooks/`. This table maps each
notebook cell (indices of the executed `*_complete.ipynb`; AP5 has one extra
markdown cell, so its code cells are shifted by one) to the v2 code that replaces
it. Numerics are unchanged in Phase 1; only the packaging differs.

## Model-run notebooks `legacy/notebooks/model_runs/{allegheny,gabilan}_tile{1..5}_complete.ipynb`

| Cell | v1 content | v2 |
| --- | --- | --- |
| 2 | imports; hard-coded `concavity`, `hilltop_curvature`, `steepness`, `U`, `K0, D0 = (U/ks)^(1/n), U/C_ht`, `K0` override (AP1/3/5), bounds; `d.Elevation.load` etc.; `advection_diffusion_model(dem, area, fd, m)` | `configs/tiles/<tile>.yaml` (every value, incl. `K0_ad`, `K0_partitioned`, `D0` as `"U/ks"`/`"U/C_ht"` expressions, bounds); `lemcvt.config.load_tile` (units, sha256, sanity checks); `lemcvt.models.ad.advection_diffusion_model(Z, A, FD, dx, m)` |
| 3 | forward AD at (K0, D0); `plot_maps_area_slope`; `save_model_outputs(... "AD-no-opt")` | `lemcvt tile run --stages ad` writes `*_AD-no-opt_{elevation,filled,flow_direction,area,logarea,hillshade}.tif` (`lemcvt.analyses.outputs`); plots are not reproduced (figures read results) |
| 5 | `optimize_advection_diffusion` (L-BFGS-B) | `lemcvt.optimize.v1.optimize_advection_diffusion`, called by `tile run --stages ad`; result in `params.json["stages_results"]["ad"]`, every evaluation in `trajectory.parquet` |
| 6 | forward AD at optimum; save `AD-opt` | same stage; `*_AD-opt_*.tif` |
| 8 | `partitioned_threshold_model(dem, area, fd, m, U, K0)` (AP1/AP5 re-instate K0 here; AP3 does not) | `lemcvt.models.partitioned.partitioned_threshold_model(Z, A, FD, dx, m, U, K0_partitioned)`; `K0_partitioned` per tile config |
| 9-12 | forward partitioned at A_crit = 5e3, 1e3, 5e2, 1e2 (plots only) | `tile run --exploration` records their MSEs in `params.json["forward_runs"]` |
| 14 | `optimize_threshold_acrit` (Nelder-Mead, D = D0) | `lemcvt.optimize.v1.optimize_threshold_acrit`; stage `acrit` |
| 15 | forward at A_crit*; save `Partitioned-model-a_crit-opt` | stage `acrit`; `*_Partitioned-model-a_crit-opt_*.tif` |
| 17-18 | hillslope slope histograms | `params.json["stages_results"]["acrit"]["hillslope_slope_histogram"]` |
| 20 | `partitioned_threshold_model_nonlinear(...)`, `Sc = 1.27` | `lemcvt.models.partitioned_nonlinear.partitioned_threshold_model_nonlinear`; `S_c` in site/tile config |
| 21-22 | nonlinear forward at A_crit = 1e3, 1e4 (plots only) | `tile run --exploration` |
| 24 | `optimize_threshold_acrit_nonlinear` | `lemcvt.optimize.v1.optimize_threshold_acrit_nonlinear`; stage `nonlinear` |
| 25 | nonlinear forward at optimum (plot only, not saved) | stage `nonlinear` also saves `*_Partitioned-model-nonlinear-a_crit-opt_*.tif` (new) |
| 27 | `optimize_threshold_full` (joint, rebuilds network per K) | `lemcvt.optimize.v1.optimize_threshold_full`; stage `joint` |
| 28 | forward partitioned at joint optimum; save `Partitioned-model-a_crit-k-d-opt` | stage `joint`; `*_Partitioned-model-a_crit-k-d-opt_*.tif` |

Whole notebook: `lemcvt tile run --tile GM1 --stages ad,acrit,nonlinear,joint --optimizer v1`
(or `lemcvt slurm submit tile_run --tiles all`). Forward evaluation at given
parameters, including the saved-grid comparison of Phase 2.1:
`lemcvt tile forward --tile GM1 --params v1 --compare`.

## Figure notebooks `legacy/notebooks/figures/`

| Notebook | v1 cells | v2 compute | v2 figure |
| --- | --- | --- | --- |
| `fig_slope_area.ipynb` | cell 1: `slope_area_regression_binned` on observed + three saved model grids (AP1, GM2; min_area 8.89e3 / 5.49e3) | `lemcvt slope-area --tiles AP1,GM2` (`lemcvt.analyses.slope_area_fig`) | `lemcvt figures slope_area` |
| `supp_fig5.ipynb` | cells 1-3: synthetic planform curves (Laplacian, regression, local k_s) | `lemcvt synthetic pe` (`lemcvt.analyses.synthetic_pe`) | `lemcvt figures supp_fig5` |
| `supp_fig6.ipynb` | cell 1: resample 1 m filled grid to 25 dx, recompute hydrology, Ks_mod/Ks_obs (D8) with the hard-coded K, D, theta tables | `lemcvt sweep resolution --routing d8` (`lemcvt.analyses.resolution_sweep`, `V1_KD`, `V1_THETA`) | `lemcvt figures supp_fig6` |
| `resolution_study_dinf.ipynb` | cell 4: same with D8 and D-infinity | `lemcvt sweep resolution --routing d8,dinf` | `lemcvt figures resolution_dinf` |
| `d8_vs_dinf_comparison.ipynb` | cells 5, 11: AD and partitioned at measured parameters, D8 vs D-inf, MSE table, slope-area overlay | `lemcvt d8-vs-dinf --tiles AP1,GM1` (`lemcvt.analyses.d8_vs_dinf`, `V1_CONFIGS`) | `lemcvt figures d8_vs_dinf` |
| (not in v1 repo) Fig. 2 forward runs with regional parameters | - | `lemcvt perron-forward` (`lemcvt.analyses.perron_forward`; regional values in `configs/sites/*.yaml`, flagged for verification) | - |
| (not in v1 repo) Fig. 3 (dK, dD) and Table 1 | - | from `tile run` results | `lemcvt figures fig3_deltas`, `lemcvt tables table1` |

## v1 `src/` modules to v2

| v1 | v2 |
| --- | --- |
| `models.py` | `lemcvt.models.common`, `.ad`, `.partitioned`, `.partitioned_nonlinear` (arrays instead of TopoAnalysis objects) |
| `models_dinf.py` | `lemcvt.models.dinf_models` |
| `dinf.py` | `lemcvt.hydro.dinf` |
| `optimization.py` | `lemcvt.optimize.v1` (+ trajectory recording) |
| `analysis.py` | `lemcvt.measure.slope_area` |
| `synthetic.py` | `lemcvt.models.synthetic` |
| `utils.py` | `lemcvt.utils` (slope magnitude, NaN-aware resampling), `lemcvt.io.grids.crop_grid`, `lemcvt.analyses.outputs` (`save_model_outputs`) |
| `plotting.py` | `lemcvt.figures.common.set_nature_style`; comparison plots are replaced by result-reading figures |
| TopoAnalysis `FilledElevation`, `FlowDirectionD8`, `Area`, `ChannelSlopeWithSmoothing`, `Hillshade`, `LogArea`, `Elevation.load/save` | `lemcvt.hydro.fill`, `.d8`, `.channel_slope`, `.derived`, `lemcvt.io.grids` (bit-level ports; see `tests/test_hydro.py::TestTopoAnalysisEquivalence`) |

## Environment and data

| v1 | v2 |
| --- | --- |
| `environment.yml` (macOS export, Python 3.10) | `envs/sherlock.yml` + `envs/sherlock.lock.txt` (Python 3.12, lean); `envs/lem-v1.yml` rebuilds v1 for drift checks |
| grids committed in `data/` (TopoAnalysis GeoTIFFs without extension) | grids in `$LEMCVT_DATA` (`$GROUP_HOME/lemcvt_data/v1` on Sherlock) as `.tif`, listed with sha256 in `data/manifest.csv`; `lemcvt data verify|fetch` |
| papermill batch scripts (`landscape-optimizer/batch_files-from-github-repo`) | `lemcvt slurm submit <step> --tiles all` renders `slurm/lemcvt_step.sbatch` |
