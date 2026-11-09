# Phase 1 status: refactor to the `lemcvt` package (numerics unchanged)

Directive: `reports/directives/refactor_v2_directive.md` Phase 1, with addendum 1.
Branch `v2`, commits 94b1dd7 (skeleton) and 21761bb (analyses, measurement, docs).
Date: 2026-11-09. Nothing pushed.

## What exists

| Directive item | Status | Where |
| --- | --- | --- |
| Package `lemcvt`, `pyproject.toml`, pinned deps, lockfile | done; Python 3.12 env `lemcvt` in `$GROUP_HOME/scdobbs-envs`, lock `envs/sherlock.lock.txt` (221 lines) | `pyproject.toml`, `envs/` |
| `io/` GeoTIFF + units | done; v1 grids are already GeoTIFFs, so the converter is a copy with `.tif`; bit-identical round trip tested | `src/lemcvt/io/` |
| `hydro/` d8, dinf, fill, channel slope | done; fill, D8 direction, area, channel-slope smoothing and hillshade are bit-identical to TopoAnalysis on synthetic grids (`tests/test_hydro.py::TestTopoAnalysisEquivalence`) | `src/lemcvt/hydro/` |
| `models/` ad, partitioned, nonlinear, synthetic, D-inf | done; operation order preserved | `src/lemcvt/models/` |
| `measure/` concavity, steepness, hilltop curvature | done (addendum design, see `docs/measurement_methods.md`); v1 `analysis.py` ported as `measure/slope_area.py` | `src/lemcvt/measure/` |
| `optimize/v1.py` | done; exact port plus trajectory recording | `src/lemcvt/optimize/v1.py` |
| `analyses/` tile pipeline, forward, resolution sweep, synthetic Pe, D8 vs D-inf, slope-area, Perron forward | done | `src/lemcvt/analyses/` |
| `figures/`, `tables/` reading results only | done: slope_area, supp_fig5, supp_fig6, resolution_dinf, d8_vs_dinf, fig3_deltas; table1 | `src/lemcvt/figures/`, `src/lemcvt/tables/` |
| `cli.py` with every listed command | done, plus `lemcvt measure`, `lemcvt data manifest`, `lemcvt slurm submit custom` | `src/lemcvt/cli.py` |
| `configs/` sites, tiles, slurm | done; ten tile configs generated from `reference/v1_reference.json` with per-stage K0 and the AP3 quirk | `configs/` |
| `slurm/` templates rendered by the CLI, core-hour estimate, 200 h guard | done | `slurm/lemcvt_step.sbatch`, `src/lemcvt/slurm.py` |
| `results/` per run: params.json, trajectory.parquet, GeoTIFFs, run.json | done | `src/lemcvt/results.py` |
| Grids leave git; `data/manifest.csv`; `lemcvt data fetch/verify` | done; 270 grids staged to `$GROUP_HOME/lemcvt_data/v1` and sha256-verified; `data/AP`, `data/GM` removed from the index | `data/manifest.csv`, `configs/data.yaml` |
| Units explicit; loader refuses wrong units; `reproduce_v1_units` with loud warning | done | `configs/tiles/*.yaml`, `src/lemcvt/config.py`, `src/lemcvt/io/units.py` |
| Notebooks to `legacy/notebooks/`, `docs/v1_to_v2.md` | done | `legacy/`, `docs/v1_to_v2.md` |
| Tests: 105 ported + I/O round trip, unit refusal, config validation, CLI smoke on 100 x 100 | done: 149 pass (`pytest -q`, 45 s) | `tests/` |
| CI (ruff, mypy, fast tests) | workflow written; runs on push once the branch is pushed | `.github/workflows/ci.yml` |
| README: installation, data fetch, one-command reproduction, run-time table | done (run-time table deferred to Phase 3 as the directive says) | `README.md` |

Not in Phase 1 scope and not done: optimizer v2 (Phase 4), performance work (Phase 3),
AP unit correction runs (Phase 5), reruns (Phase 6). `mypy` has not been run to a clean
state yet (CI will show its count); ruff is clean.

## Early regression evidence (Phase 2.1 preview)

`lemcvt tile forward --tile GM1 --params v1 --compare` on the full tile (job 51680757):

| run | max dz / relief | relative MSE difference | parameters |
| --- | --- | --- | --- |
| AD-no-opt (K0, D0 exact) | 9.3e-13 | 1.6e-13 | exact |
| AD-opt | 2.0e-5 | 8.1e-9 | printed, 5 s.f. |
| A_crit-opt | 4.1e-13 | 2.3e-12 | A_crit printed, same channel mask |
| joint | 7.6e-6 | 1.25e-8 | printed, 5 s.f. |

AP5 AD-no-opt: 9.7e-13 of relief, 3.4e-13 relative MSE (job running for the other stages
when this was written). The exact-parameter runs meet the Phase 2.1 tolerances with 6 to 7
orders of magnitude to spare; the runs at printed (rounded) parameters differ by exactly
the amount the rounding implies, so for them the attainable test is the one noted in
`lemcvt.analyses.forward`: MSE agreement at about 1e-8 to 1e-5 relative. The Phase 2 report
will state this per tile.

Synthetic Peclet analysis (`lemcvt synthetic pe`): the local-k_s table matches the v1 printed
table at every dx to the printed 3 significant figures (worst difference 1.9e-3 relative).

## Choices made in Phase 1 that Steve should know

1. **Model API takes arrays.** Factories are `f(Z, A, FD, dx, ...)`; TopoAnalysis objects are
   gone. `dx` is the geotransform's x pixel size, as v1 used it (`dy` is recorded but, as in v1,
   not used by the models).
2. **Stage K0 is explicit.** `K0_ad` and `K0_partitioned` are separate config fields; AP3 keeps
   `K0_partitioned = 1e-6` because that is what ran (`00_reference.md` 3.5).
3. **Nonlinear optimum grid is saved** (`*_Partitioned-model-nonlinear-a_crit-opt_*`), which v1
   plotted but did not save.
4. **Figure-notebook parameter tables** (`KD_GM`, `KD_AP`, `THETA_*`, the d8-vs-dinf configs and
   the slope-area `min_area` values) are kept verbatim under `params="v1"` so Phase 2.3 can
   reproduce the printed tables; `params="config"` switches to the tile configs.
5. **Fig. 2 (Perron forward) and Fig. 3 / Table 1** had no code in the v1 repository. They are
   implemented from their descriptions (regional parameters per site; dK = log10(K_fit/K0),
   dD = log10(D_fit/D0)). The regional values in `configs/sites/*.yaml` come from the legacy
   notebooks and are flagged for verification against Perron et al. (2012).
6. **Measurement pipeline** (addendum): chi with the model's upstream integration rule so that
   k_s = U/K exactly (recovered to 1e-6 on a model-generated landscape), tile-edge channel
   pixels as chi roots (the models' boundary), slope-area rollover as the primary threshold
   with a ladder of fixed thresholds reported, divide-based hilltops with a plateau criterion
   for the curvature window, basin and segment bootstraps. The v1 method is re-run for
   comparison; its curvature step is approximated because the `topo_analysis` module is
   missing. Details and references in `docs/measurement_methods.md`.
7. **Environment:** `PYTHONNOUSERSITE=1` is required because a user-site numpy under
   `~/.local` for Python 3.12 shadows the environment's. The job template sets it.
8. **Slurm partition** is left to the scheduler default (`partition: null`), pending the
   values Steve wants in `configs/slurm/sherlock.yaml`.

## Jobs submitted for Phase 2 (cheap, within the 200 core-hour guard)

Forward regression on the remaining eight tiles, slope-area (AP1, GM2), D8-vs-D-inf
(AP1, GM1), resolution sweep (all tiles, D8 and D-inf) and `lemcvt measure` on all tiles.
The optimizer regression (Phase 2.2, GM1 and AP5, about 21 h per tile) is **not** submitted:
the directive says to ask first.
