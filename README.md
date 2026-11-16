# lemcvt: steady-state landscape evolution model calibration

Supplemental code for Dobbs & Hilley, "competition versus threshold" models of hillslope-channel landscapes. The package
calibrates a coupled advection-diffusion (AD) model and threshold-partitioned models (linear hillslopes, and nonlinear
hillslopes after Roering et al. 1999) against 1 m lidar tiles from the Allegheny Plateau (AP) and Gabilan Mesa (GM), by
minimising the mean squared elevation misfit. It measures the model inputs (m, k_s, hilltop curvature, D₀) from the
tiles themselves.

Version 2 (`v2.0.0`) replaces the v1 notebooks with a package and the `lemcvt` command-line tool. It corrects the AP
vertical units (z was in US survey feet, x and y in metres), replaces the optimizers with exact profiles (`--optimizer
v2`), rebuilds the D-infinity partitioned model and re-measures every input. What changed and why:
`docs/numerics_changes.md` (N1-N12) and `docs/measurement_methods.md` (section 0: the final methods). The v1 notebooks,
code and tests are kept unchanged under `legacy/`; `docs/v1_to_v2.md` maps every notebook cell to its v2 command.

**Model form of the nonlinear results.** The v1 nonlinear model, and every nonlinear result up to addendum 9, solves
D_eff∇²z = −U ("centre form", nonstandard; N10), not the Roering law ∇·(D_eff∇z) = −U. Those results are labelled
"centre form, D_eff∇²z = −U (nonstandard)" wherever they appear. New nonlinear work uses the flux form with a Newton
solver (`lemcvt a10 ...`; N11, N12).

## Layout

```
pyproject.toml            package lemcvt
envs/sherlock.yml         lean conda environment; envs/sherlock.lock.txt is the explicit lock
envs/lem-v1.yml           Linux rebuild of the v1 environment (drift checks only)
src/lemcvt/
  io/        GeoTIFF read/write, units (ftUS/m), data manifest
  hydro/     priority-flood fill, D8, D-infinity, channel-slope smoothing, hillshade
  models/    advection-diffusion, partitioned (linear; nonlinear centre and flux forms, Picard and Newton), D-inf, synthetic
  measure/   chi-z k_s and concavity, hilltop curvature, slope-area, audits, legacy v1 method
  optimize/  v1 (exact port of the notebook optimizers) and v2 (exact profiles)
  analyses/  tile pipeline, Phase 6 runs, E*-R*, slope threshold, flux-form GM test, morphology, width, heads, ...
  figures/   every figure, built from results only, each with its caption data as CSV beside it
  tables/    Table 1 (LaTeX booktabs + siunitx, and CSV)
  cli.py     the `lemcvt` command
configs/
  sites/{AP,GM}.yaml       U, S_c, sanity ranges, regional parameters
  tiles/{AP,GM}{1..5}.yaml grids + sha256, units, measured inputs with provenance, bounds, v1 reference values
  slurm/sherlock.yaml      partition and per-step resources (from sacct measurements)
  data.yaml                where `lemcvt data fetch` copies the grids from
slurm/                     job-array template rendered by `lemcvt slurm submit`
data/manifest.csv          sha256 of every grid (the grids live outside git)
reference/                 frozen v1 reference (Phase 0): printed numbers, grid manifest, saved-grid MSEs
reports/                   directives (reports/directives) and checkpoint reports (reports/v2), with their table scripts
legacy/                    v1 notebooks, src, tests, scripts, docs, run logs, environment
tests/                     pytest suite (the fast suite excludes tests marked `fulltile`)
```

## Installation

### On Sherlock

The environment is built once into group storage, inside a job (`sbatch envs/build_lemcvt_env.sbatch`), which runs:

```bash
export MAMBA_ROOT_PREFIX=$GROUP_HOME/micromamba_root
$GROUP_HOME/scdobbs-envs/bin/micromamba create -y -p $GROUP_HOME/scdobbs-envs/lemcvt -f envs/sherlock.yml
$GROUP_HOME/scdobbs-envs/bin/micromamba run -p $GROUP_HOME/scdobbs-envs/lemcvt python -m pip install -e . --no-deps
```

Every shell that runs `lemcvt`, and every shell that submits a job, needs these four variables (the job renderer bakes the
submitting shell's values into the script and refuses to render when the data or results root is unset):

```bash
export PYTHONNOUSERSITE=1                                 # a user-site numpy under ~/.local otherwise shadows the env's
export MAMBA_ROOT_PREFIX=$GROUP_HOME/micromamba_root
export LEMCVT_DATA=$GROUP_HOME/lemcvt_data/v1
export LEMCVT_RESULTS=$GROUP_HOME/lemcvt_results
alias lemcvt="$GROUP_HOME/scdobbs-envs/bin/micromamba run -p $GROUP_HOME/scdobbs-envs/lemcvt lemcvt"
lemcvt info
```

### Elsewhere

`pip install -e ".[dev]"` in a Python environment with GDAL (rasterio), then point `configs/data.yaml` (or `lemcvt data
fetch --source`) at a copy of the grids. CI (`.github/workflows/ci.yml`) runs ruff, mypy and the fast test suite on
ubuntu-latest.

## Data

The ten tiles (filled elevation, D8 flow direction, drainage area) and the v1 model output grids are GeoTIFFs under
`$LEMCVT_DATA/<site>/tile<n>/`; they are not in git. `data/manifest.csv` lists every file with its sha256.

```bash
lemcvt data fetch          # copy missing or mismatching files from configs/data.yaml `source` into $LEMCVT_DATA, then verify
lemcvt data verify         # check every file on disk against the manifest
```

Every loader call verifies the three input grids before use. Units are explicit in each tile config (`z_units`,
`xy_units`); the AP tiles store z in US survey feet and the loader converts to metres (c = 3937/1200), unless
`reproduce_v1_units: true` (the Phase 1-2 regression only, with a loud warning). The loader refuses grids whose declared
units fail the site sanity checks.

## Reproducing the figures and tables

Every command writes a run directory `$LEMCVT_RESULTS/<UTC timestamp>_<step>_<label>/` with `params.json`, the
evaluation tables (parquet), GeoTIFF outputs and `run.json` (git hash, config snapshot, environment lock hash, Slurm
accounting). The figures and tables read results only:

```bash
lemcvt figures all && lemcvt tables all     # every figure (+ caption CSV) into figures/, Table 1 and the record into tables/
```

`lemcvt figures <name>` builds one figure (the names are the keys of `FIGURE_BUILDERS` in `src/lemcvt/figures/__init__.py`);
a figure whose input runs do not exist is skipped with a warning. The results of v2.0.0 are archived in
`$GROUP_HOME/lemcvt_results/v2.0.0/` (figures, tables, caption CSVs and the list of source runs).

The results themselves come from the following runs, in this order. On Sherlock each line is submitted as a Slurm array
over tiles (`lemcvt slurm submit <step> --tiles all`, or `custom --command ... --labels ...`); the command prints its
core-hour estimate first and refuses submissions above 300 core-hours without `--approved`. The task scripts the reports
used are in `reports/v2/scripts/`.

| step | command | produces |
| --- | --- | --- |
| measured inputs | `lemcvt phase6 inputs --tiles all`, then `lemcvt phase6 write-configs` | m, k_s, K₀, C_ht, D₀,A, D₀,B with provenance in the tile configs |
| site exponent | `lemcvt tile run --tile <T> --optimizer v2 --stages m_profile --m-range 0.05,0.90` | S(m), the site m (fig `a4_exponent_site`) |
| AD profile | `lemcvt tile run --tile <T> --stages ad --optimizer v2` | AD K, D, MSE (Table 1) |
| linear partitioned | `bash reports/v2/scripts/phase6_partitioned_task.sh <T> A` (and `B` on AP) | stage 2 and joint optima, 1 % and 5 % intervals (Table 1, fig3) |
| nonlinear stage (centre form) | `lemcvt tile run --tile <T> --stages nonlinear --optimizer v2 --no-derived` | Table 1 (labelled nonstandard) |
| nonlinear joint (centre form) | `lemcvt tile v2-nonlinear-joint --tile <T>` (split into parts by `phase6_nlj_task.sh`) | fig `a5_nonlinear_joint` (labelled nonstandard) |
| D-infinity joint | `lemcvt tile v2-dinf-joint --tile AP1` and `GM1` | SI comparison |
| resolution sweep | `lemcvt sweep resolution --tiles <T> --routing d8,dinf --params config` | supp_fig6, resolution_study_dinf |
| synthetic Peclet | `lemcvt synthetic pe` | supp_fig5 |
| D8 against D-inf, slope-area | `lemcvt d8-vs-dinf --tiles AP1,GM1 --params config`; `lemcvt slope-area --tiles all --phase6` | d8_vs_dinf_*, fig_slope_area |
| E*-R* | `lemcvt phase8 er-star --tiles all` | fig `a8_er_star`, fitted S_c |
| slope-dependent threshold | `bash reports/v2/scripts/phase8_slope_threshold_task.sh <T>_a<alpha>`; `lemcvt phase8 slope-threshold-summary` | fig `a8_slope_threshold` |
| flux-form GM test | `lemcvt a10 flux-part --tile <T> --sc <S_c> --split k/2 --group <tag>`; `a10 flux-merge`; `a10 flux-summary` | fig `a10_flux` |
| morphology score | `lemcvt a9 morphology --tiles all` | fig `a9_morphology` |
| width, heads (Test 2) | `lemcvt phase7 width`, `width-constrained`, `heads`, `heads-analyse` | width fits, head-area and area-slope figures |
| record | `lemcvt phase6 record` | `tables/old_vs_new_record.csv` (every v1 number against v2, with causes) |

## Run times

Measured on `serc` (AMD EPYC, 1 CPU per task unless stated; `sacct` elapsed per task), from the checkpoint reports.

| step | per task | tasks | notes |
| --- | --- | --- | --- |
| measured inputs (`phase6 inputs`) | 3 min | 1 (all tiles) | |
| m profile (86 m x 200 A_crit) | 1.1-1.2 h | 10 | |
| AD profile | 1.5-1.7 h | 10 | 200-point r grid + Brent |
| linear partitioned (2000 + refine + 5 % fill) | AP 6.6-11.8 h, GM 0.2-2.1 h | 10 (+5 at D₀,B) | 4318 (AP) and 1758 (GM) unique areas in the 5 % intervals on average |
| nonlinear stage, centre form (Picard, S_c 1.27) | AP 8.5-26.7 h, GM about 9 h | 10 (+5) | AP3 at D₀,A: 96 of 261 evaluations at the Picard cap |
| nonlinear joint, centre form (30 x 20) | about 26 h (AP), 36 h (GM) of CPU per tile | 3-5 parts per tile | split by A_crit and merged |
| D-infinity joint | 12.8-13.6 h | 2 (2 CPUs) | |
| resolution sweep | about 1 min | 10 | |
| E*-R* | 2.5 min | 1 | |
| slope threshold (one alpha) | about 3 h | 60 | |
| flux-form GM test, one tile and S_c (12 x 14) | see the addendum 10 report | 40 parts | about 80-220 s per Newton solve with continuation |
| morphology score | 2 min | 1 | |
| heads analyse | 8 min | 1 | |
| `figures all`, `tables all` | 3 min, 16 s | 1 | |

The v1 notebooks took about 21 h per tile for the four stages (`reports/v2/00_reference.md`, section 3.4); the v2
optimizer replaces their Nelder-Mead stages by exact profiles, which take longer per tile because they evaluate every
step of the A_crit staircase.

## Tests and checks

```bash
ruff check src tests && mypy src/lemcvt
env -u LEMCVT_DATA python -m pytest -q -m "not fulltile" tests     # the fast suite (CI); about 4 min on serc
python -m pytest -q tests                                          # with $LEMCVT_DATA set: also the full-tile tests
```

## Documents

- `reports/directives/`: the v2 directive and its addenda (the plan, in order)
- `reports/v2/`: one checkpoint report per addendum; the latest is the addendum 10 wrap-up
- `docs/measurement_methods.md`: how every measured input is defined, with the final methods in section 0
- `docs/numerics_changes.md`: every change that can alter a floating-point result (N1-N12)
- `docs/v1_to_v2.md`: notebook cell to v2 command mapping
