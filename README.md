# lemcvt: steady-state landscape evolution model calibration

Supplemental code for Dobbs & Hilley, "competition versus threshold" models of
hillslope-channel landscapes. The package calibrates a coupled advection-diffusion
model and threshold-partitioned (linear and nonlinear hillslope) models against
1 m lidar tiles from the Allegheny Plateau (AP) and Gabilan Mesa (GM) by minimising
the mean squared elevation misfit, and measures the input parameters (θ, k_s, hilltop
curvature) from the tiles themselves.

Version 2 (branch `v2`) replaces the v1 notebooks with a package and command-line
tool. Phase 1 of the v2 directive ports every v1 computation without changing any
numerical operation; the ports are checked against TopoAnalysis bit for bit
(`tests/test_hydro.py`) and against the saved v1 grids (`lemcvt tile forward --compare`).
The v1 notebooks, code and tests are kept unchanged under `legacy/`; `docs/v1_to_v2.md`
maps every notebook cell to its v2 command.

## Layout

```
pyproject.toml            package lemcvt (Python >= 3.11)
envs/sherlock.yml         lean conda environment; envs/sherlock.lock.txt is the explicit lock
envs/lem-v1.yml           Linux rebuild of the v1 environment (drift checks only)
src/lemcvt/
  io/        GeoTIFF read/write, units (ftUS/m), data manifest
  hydro/     priority-flood fill, D8, D-infinity, channel-slope smoothing, hillshade
  models/    advection-diffusion, partitioned (linear, nonlinear), D-inf variants, synthetic
  measure/   theta, k_s (chi), hilltop curvature, slope-area, legacy v1 method
  optimize/  v1 (exact port of the notebook optimizers); v2 arrives in Phase 4
  analyses/  tile pipeline, forward runs, resolution sweep, synthetic Peclet, D8 vs D-inf, ...
  figures/   manuscript and SI figures, built from results only
  tables/    Table 1 (LaTeX booktabs/siunitx + CSV)
  cli.py     the `lemcvt` command
configs/
  sites/{AP,GM}.yaml       U, S_c, sanity ranges, regional parameters
  tiles/{AP,GM}{1..5}.yaml grid paths + sha256, units, v1 measured values, bounds, K0/D0, optimizer settings
  slurm/sherlock.yaml      partition/account/QoS and per-step resources
  data.yaml                where the grids come from
slurm/                     job-array template rendered by `lemcvt slurm submit`
data/manifest.csv          sha256 of every grid (grids themselves live outside git)
reference/                 frozen v1 reference (Phase 0): printed numbers, grid manifest, saved-grid MSEs
reports/                   directives and phase reports
legacy/                    v1 notebooks, src, tests, scripts, docs, run logs, environment
tests/                     pytest suite (fast; full-tile tests are marked `fulltile`)
```

## Installation on Sherlock

The environment is built once into group storage (job `envs/build_lemcvt_env.sbatch`):

```bash
export MAMBA_ROOT_PREFIX=$GROUP_HOME/micromamba_root
$GROUP_HOME/scdobbs-envs/bin/micromamba create -y -p $GROUP_HOME/scdobbs-envs/lemcvt -f envs/sherlock.yml
$GROUP_HOME/scdobbs-envs/bin/micromamba run -p $GROUP_HOME/scdobbs-envs/lemcvt python -m pip install -e . --no-deps
```

Always run the environment with `PYTHONNOUSERSITE=1` (a user-site numpy under
`~/.local` otherwise shadows the environment's). Point the tool at the data and a
results directory:

```bash
export PYTHONNOUSERSITE=1
export LEMCVT_DATA=$GROUP_HOME/lemcvt_data/v1
export LEMCVT_RESULTS=$GROUP_HOME/lemcvt_results
alias lemcvt="$GROUP_HOME/scdobbs-envs/bin/micromamba run -p $GROUP_HOME/scdobbs-envs/lemcvt lemcvt"
lemcvt info
```

Elsewhere: `pip install -e ".[dev]"` in a Python 3.11/3.12 environment with GDAL, then
`lemcvt data fetch` with `configs/data.yaml` pointing at a copy of the grids.

## Data

The ten tiles (filled elevation, D8 flow direction, drainage area) and the v1 model
output grids are GeoTIFFs under `$LEMCVT_DATA/<site>/tile<n>/`. They are not in git.
`data/manifest.csv` lists every file with its sha256; `lemcvt data verify` checks the
files on disk and every loader call verifies the three input grids before use.

Units are explicit in each tile config (`z_units`, `xy_units`). The AP tiles store z in
US survey feet; the loader converts to metres unless `reproduce_v1_units: true`
(Phase 1/2 regression only, with a loud warning). The loader refuses grids whose
declared units fail the site sanity checks (elevation range, median slope).

## Reproducing the analyses

Everything runs through `lemcvt`; each command writes a run directory
`$LEMCVT_RESULTS/<timestamp>_<step>_<label>/` with `params.json`, `trajectory.parquet`
(every misfit evaluation), GeoTIFF outputs and `run.json` (git hash, config snapshot,
environment lock hash).

```bash
lemcvt tile run --tile GM1 --stages ad,acrit,nonlinear,joint --optimizer v1   # one tile, ~10-45 h
lemcvt tile forward --tile GM1 --params v1 --compare                           # Phase 2.1 regression
lemcvt sweep resolution --sites AP,GM --routing d8,dinf                        # supp_fig6 + D-inf study
lemcvt synthetic pe                                                            # supp_fig5
lemcvt d8-vs-dinf --tiles AP1,GM1
lemcvt slope-area --tiles AP1,GM2
lemcvt perron-forward --tiles all                                              # Fig. 2 forward runs
lemcvt measure --tile all                                                      # theta, k_s, C_ht, D0
lemcvt figures all && lemcvt tables all                                        # from results only
```

On Sherlock, submit job arrays instead of running on a login node:

```bash
lemcvt slurm submit tile_forward --tiles all --compare
lemcvt slurm submit tile_run --tiles GM1,AP5 --dry-run     # prints the core-hour estimate first
lemcvt slurm submit custom --command "lemcvt -v slope-area --tiles AP1,GM2"
```

Submissions above 200 core-hours stop until `--approved` is passed.

## Tests and checks

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=$GROUP_HOME/opt pytest -q     # TopoAnalysis equivalence tests run when importable
ruff check src tests && mypy src/lemcvt
```

## Run times

Phase 1 keeps the v1 numerics and therefore the v1 run times (per tile, v1 hardware):
AD stage 3.4-8.8 h, A_crit stage 0.1-0.4 h, nonlinear A_crit stage 3-28 h, joint
stage 1.4-3.7 h (`reports/v2/00_reference.md`, section 3.4). The Phase 3 table of
speedups replaces this paragraph.

## Documents

- `reports/directives/refactor_v2_directive.md` and `refactor_v2_addendum_1.md`: the plan
- `reports/v2/00_reference.md`: Phase 0 checkpoint (what v1 did, exactly)
- `docs/v1_to_v2.md`: notebook cell to v2 command mapping
- `docs/measurement_methods.md`: how θ, k_s and C_ht are measured and why
