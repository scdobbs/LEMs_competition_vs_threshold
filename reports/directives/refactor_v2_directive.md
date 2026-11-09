# Directive: Refactor the codebase to v2 (scripts and Slurm instead of notebooks), reproduce v1, then correct the AP units and rerun the base analyses

## Context

Repo: `scdobbs/LEMs_competition_vs_threshold` (GitHub; tags `v1.0.0` and `v1.0.1` on `main`). The local working clone on Sherlock is `~/LEMs_competition_vs_threshold-JGR-branch` and holds unpushed branches: `feature/channel-width`, `feature/acrit-slope`, `feature/head-extraction`, and possibly `fix/ap-vertical-units`. Environment: `$GROUP_HOME/scdobbs-envs/lem-widths`.

### How the analyses currently run

- Ten model-run notebooks (`notebooks/model_runs/{allegheny,gabilan}_tile{1..5}.ipynb`) are executed with papermill into `*_complete.ipynb`. `data/AP/output_files/allegheny_tile1.out` shows about 21 hours per tile. The joint K–D–A_crit stage alone takes about 14 hours, and the AD optimization about 3.4 hours.
- Five figure notebooks (`notebooks/figures/`) build `fig_slope_area`, `supp_fig5`, `supp_fig6`, `d8_vs_dinf_comparison` and `resolution_study_dinf`.
- Per-tile measured inputs (θ, k_s, hilltop curvature) are hard-coded in each notebook. The code that measured them is not in the repo.
- Grids are in TopoAnalysis format, about 1.1 GB, committed directly to git.
- The environment file is a full macOS export pinned to Python 3.10, including Jupyter.

### Known defects this work must fix

1. **The AP tiles have z in US survey feet and x, y in metres** (see `reports/directives/ap_vertical_units_fix_directive.md`). That directive is superseded by Phase 5 below. Its inventory and corrections-table requirements carry over.
2. **The partitioned joint stage rebuilds the channel network for every candidate K.** That network is a Python BFS over about 10⁶ cells, which is where most of the 14 hours go.
3. **Each AD solve converts CSR to LIL and back** to impose boundary conditions.
4. **The A_crit-only stages run Nelder–Mead on a staircase objective.** The misfit changes only when A_crit crosses a cell's drainage area. Two symptoms in the v1 outputs:
   - GM2's stage-2 value sits at the search bound;
   - on GM3 the stage-2 and nonlinear A_crit are identical, 1.2974×10³.

### Structural facts the new optimizer can exploit

These are stated for checking, not assumed. Prove each with a test in Phase 4.

- **Linear partitioned model, fixed A_crit.** Channel elevations are z_bc + (U/K)·Σ A⁻ᵐ ΔL, which is affine in U/K. The hillslope solve is linear diffusion with Dirichlet values from channels and the tile boundary. Therefore Z(K, D) = a + (U/K)·b + (U/D)·c, where a, b and c each come from one sparse solve. The MSE-optimal (1/K, 1/D) is then a two-coefficient non-negative least-squares problem, with an exact global optimum. The joint problem reduces to a one-dimensional profile over A_crit.
- **AD model.** For fixed r = K/D, Z = h(r) + (U/D)·g(r). The optimal D at each r is closed-form, so the AD optimization reduces to a one-dimensional profile over r.
- **The nonlinear model reduces to neither.** It keeps an iterative solve and gets a grid profile over A_crit.

### Repository decision: a branch on the same repo, not a fork or a new repo

Do the work on a long-lived branch `v2` cut from `main` at `v1.0.1`. When Phase 6 is accepted, merge `v2` into `main` and tag `v2.0.0` for the resubmission's Zenodo release.

Reasons:
- one history;
- the Zenodo GitHub integration issues `v2.0.0` under the same concept DOI the manuscript already cites;
- `v1.0.1` stays citable as the state the previous submission used;
- a fork or a new repo would split the history and the DOI for no benefit.

### Constraints

- **Do not modify `main`, `v1.0.x`, any existing grid, notebook output or figure, or the manuscript or response letter.**
- **Ask Steve before:**
  - any `git push`;
  - any Slurm submission above 200 core-hours (list the estimate first);
  - any download.
- Steve's first approval request should be to push the unpushed local feature branches as they are, so nothing exists only on Sherlock.
- Partition, account and QoS names go in config. Ask Steve for them; do not guess.

---

## Phase 0. Freeze and capture the reference (no new science)

1. **Branch inventory.** List every local branch in the Sherlock clone with its last commit and whether it exists on the remote. Ask to push the unpushed ones.
2. **Create `v2`** from `v1.0.1`. Cherry-pick nothing yet.
3. **`reference/v1_reference.json`.** Parse every number printed in the ten `*_complete.ipynb` notebooks, per tile and per stage:
   - initial K₀ and D₀ and the hard-coded θ, k_s, hilltop curvature, U and bounds;
   - AD-optimized K, D and MSE;
   - A_crit-only A_crit and MSE;
   - nonlinear A_crit and MSE;
   - joint A_crit, D, K and MSE;
   - every forward-run MSE.

   Also capture the printed tables from the five figure notebooks (MSE tables, k_s ratios by Δx, and so on), and save the extraction script.
4. **`reference/v1_grids_manifest.csv`.** Record a sha256, shape, georeference and min/max for every grid in `data/*/tile*/` and `data/*/tile*/outputs/`. The saved output grids (`*_AD-opt_elevation`, `*_Partitioned-model-a_crit-k-d-opt_elevation`, …) are the forward-regression targets.
5. **The v1 environment.** Rebuild the v1 environment on Sherlock as `lem-v1` from `environment.yml` (strip macOS-only packages; keep numpy 1.26.4 and scipy 1.15.3). Confirm that `pytest tests/` passes (105 tests). This environment is used only to separate environment drift from refactor bugs.
6. **Missing measurement code.** Search Sherlock group storage for the scripts that measured θ, k_s and hilltop curvature per tile and cut the tiles from the mosaics. Report what exists.

**Checkpoint report** `reports/v2/00_reference.md`. Steve reviews it before Phase 1.

## Phase 1. Refactor (numerics unchanged)

Port every computation to a package with scripts. **Do not change any numerical operation in this phase**, including operation order where it could affect floating point. Restructuring the code is fine.

### Layout

```
pyproject.toml            # package `lemcvt`, Python 3.11 or 3.12, pinned deps; lockfile committed
envs/sherlock.yml         # lean conda env (no Jupyter); module loads documented
src/lemcvt/
  io/          grids.py (GeoTIFF read/write via rasterio; TopoAnalysis→GeoTIFF converter), units.py
  hydro/       d8.py, dinf.py, fill.py
  models/      ad.py, partitioned.py, partitioned_nonlinear.py, synthetic.py, *_dinf.py
  measure/     concavity.py (χ–elevation), steepness.py (slope–area), hilltop_curvature.py
  optimize/    v1.py (exact port of current optimizers), v2.py (Phase 4)
  analyses/    tile_pipeline.py, perron_forward.py, resolution_sweep.py, d8_vs_dinf.py, synthetic_pe.py
  figures/     one module per manuscript/SI figure; reads results only, never computes models
  tables/      table1.py (LaTeX with booktabs + siunitx S columns, and CSV)
  cli.py
configs/
  sites/{AP,GM}.yaml       # U, vertical/horizontal units, regional (Perron 2012) params, S_c
  tiles/{AP,GM}{1..5}.yaml # grid paths + sha256, θ, k_s, C_ht, bounds, A_crit init/bounds, optimizer settings
  slurm/sherlock.yaml      # partition, account, qos, default resources (from Steve)
slurm/                     # templates rendered by the CLI; job arrays over tiles
results/                   # gitignored; one directory per run_id with config snapshot, git hash, env hash, logs
tests/
```

### Requirements

1. **Grids move to GeoTIFF.**
   - Convert every TopoAnalysis grid to GeoTIFF with float64 elevation (or float32 only where the source is float32), keeping the D8 code convention unchanged.
   - Verify a bit-identical round trip against the manifest.
   - Grids leave git. Store them on group storage with a `data/manifest.csv` of sha256 values. `lemcvt data fetch` and `lemcvt data verify` check them.
   - TopoAnalysis remains a dependency only of the converter.
2. **Every input is explicit in config, including units.**
   - Each tile config states `z_units` and `xy_units`. The loader converts to metres at load time.
   - The loader refuses to run if a declared unit disagrees with sanity checks: an elevation range versus a site range given in the site config, and median pixel slope above 0.6 for a soil-mantled site.
   - In Phase 1 the AP configs declare `z_units: ftUS`, with the conversion **disabled** by a `reproduce_v1_units: true` flag, so v1 can be reproduced exactly. The flag must log a loud warning.
3. **CLI**, every step resumable and recording its inputs:
   - `lemcvt tile run --tile GM1 --stages ad,acrit,nonlinear,joint --optimizer v1`
   - `lemcvt tile forward --tile GM1 --params results/.../params.json`
   - `lemcvt sweep resolution --sites AP,GM --routing d8,dinf`
   - `lemcvt synthetic pe` (supp_fig5)
   - `lemcvt perron-forward` (Fig. 2)
   - `lemcvt figures all` and `lemcvt tables all`. These read results only, consistent with the existing rule that the full figure suite builds from versioned code with one command and no manual post-processing.
   - `lemcvt slurm submit <step> --tiles all` renders a job-array script, prints the resource estimate and submits.
4. **Results.** Every run writes:
   - `params.json`: optimum, MSE, iterations, wall time;
   - `trajectory.parquet`: every misfit evaluation;
   - output grids as GeoTIFF;
   - a `run.json` with the git hash, the config snapshot and the environment lock hash.
5. **Notebooks.** Move the notebooks to `legacy/notebooks/` on `v2`, unchanged. A short `docs/v1_to_v2.md` maps each notebook cell to the v2 command that replaces it.
6. **Tests.** Port the 105 existing tests and add:
   - I/O round-trip tests;
   - unit-loader refusal tests;
   - config validation tests;
   - a CLI smoke test on a 100×100 cropped tile.

## Phase 2. Regression gate: reproduce v1 (STOP if it fails)

Run in the new environment. Where a check fails, rerun it in `lem-v1` to tell environment drift from a refactor bug.

1. **Forward regression, all ten tiles, every saved output grid.**
   - Run `lemcvt tile forward` with the v1 parameters from `v1_reference.json` (AP with `reproduce_v1_units: true`).
   - Compare with the saved v1 grids.
   - **Pass:** max |Δz| ≤ 10⁻⁶ × tile relief, and every MSE within relative 10⁻⁸ of the reference.
2. **Optimizer regression, GM1 and AP5, all four stages,** with `--optimizer v1`, submitted as a two-task Slurm array. This is the expensive step (about 21 hours per tile), so ask first.
   - **Pass:** every parameter within 1% and every MSE within 0.5% of the reference.
   - If it fails in the new environment, repeat in `lem-v1`. A pass there means the difference is environment drift; report its size and continue only with Steve's agreement.
3. **Figure-notebook regression.** For each printed table in the five figure notebooks, the v2 command reproduces it within the same tolerances. Visual diffs of the regenerated figures go in the report.

Report: `reports/v2/02_regression.md`, with a pass/fail table and wall times. **Stop for review.**

## Phase 3. Performance (numerics unchanged within tolerance)

Profile first, with `cProfile` and `line_profiler` on GM1. Then, in this order:

1. **Assembly.** Vectorize the Laplacian and advection stencils (`scipy.sparse.coo_array` from index arrays) and remove the per-cell Python loops.
2. **Boundary conditions.** Impose Dirichlet rows without LIL conversion (precomputed row mask on CSR, or eliminate boundary unknowns once).
3. **Channel network in the joint stage.** Compute χ_D8 = Σ A⁻ᵐ ΔL and the boundary-anchor elevation once per tile. Channel elevations are then z_anchor + (U/K)·χ_D8. This is exact, so no rebuild per K is needed.
4. **Hillslope operator.** For a fixed hillslope set, factor it once (`scipy.sparse.linalg.splu`, or CHOLMOD via `scikit-sparse` if available, since the operator is symmetric negative definite). Reuse the factorization across D.
5. **D8 BFS.** Use numba or vectorized topological ordering for D8 accumulation and the BFS.
6. **Parallelism** across tiles and stages through Slurm arrays only. Inside a job, set threads explicitly (`OMP_NUM_THREADS`, `MKL_NUM_THREADS`).

**Gate.** Forward regression (Phase 2.1) still passes on all ten tiles, and optimizer regression (Phase 2.2) still passes on GM1 and AP5. Report the speedup per stage.

## Phase 4. Optimizer v2 (new, flagged; v1 remains available)

Implement `optimize/v2.py`, selected with `--optimizer v2`. v1 stays the default until Steve decides otherwise.

1. **Linear partitioned model.** Build a grid of A_crit values: the unique drainage-area values between 10² and 10⁵ m², thinned to at most 200 log-spaced breakpoints, so the staircase is sampled at its steps. At each A_crit:
   - solve for a, b and c;
   - solve the non-negative least squares for (U/K, U/D) under the config bounds;
   - record the MSE.

   Output the full profile MSE(A_crit) with the optimal K and D along it, and the global optimum.
2. **AD model.** Profile over log₁₀ r on a 200-point grid spanning the bounds' ratio range, with closed-form D at each r. Refine the minimum with bounded Brent.
3. **Nonlinear model.** Use the same A_crit grid with D fixed at D₀ (matching the v1 stage definition). Optionally a joint profile, flagged as extra.
4. **Tests that prove the reductions:**
   - for random (K, D) on a cropped tile, a + (U/K)b + (U/D)c matches the direct forward solve to 10⁻¹⁰ relative (partitioned);
   - h(r) + (U/D)g(r) matches the direct solve (AD);
   - on a 60×60 crop, a brute-force grid over (A_crit, K, D) finds no lower MSE than v2.
5. **Comparison, all ten tiles** (AP still with `reproduce_v1_units: true`). Tabulate v1 against v2 for every stage: parameters, MSE, and whether v2 is lower. Plot each tile's A_crit profile with the v1 stage-2 and joint values marked.

   This answers the open question of how well A_crit is constrained, and shows whether the GM2 bound and the GM3 coincidence were optimizer artifacts.

Report: `reports/v2/04_optimizer_v2.md`.

**Decision for Steve:** which optimizer's numbers the paper reports. Recommendation to state in the report: v2, if it is never worse and the proofs pass, because it gives exact global optima and the misfit profiles. **Stop for review.**

## Phase 5. Correct the AP vertical units

1. **Corrected grids.** Set `reproduce_v1_units: false` in the AP configs, so z is converted to metres (×0.3048) at load time. Also write corrected GeoTIFFs with a new manifest entry; the originals are kept.
2. **Flow and area grids.** Regenerate D8 flow directions and area from the corrected filled grids with the v2 hydrology code. Compare cell by cell with the originals:
   - **Identical, or differing only on filled flats:** keep the originals.
   - **Otherwise:** report where they differ and use the regenerated grids.
3. **Measured parameters.** Re-measure θ, k_s and hilltop curvature on all ten tiles with `lemcvt measure`, applying GM's methods to both sites (χ–elevation for θ, slope–area for k_s, `robust_curv_mag` with block-mean subsampling, 30 m on AP and 10 m on GM, then a 9-point Laplacian over hilltop pixels).
   - **Regression first:** on GM, re-measured values must reproduce the hard-coded ones within 5%.
   - If they don't, report the difference and stop. That means the measurement method is not the one v1 used.
   - On AP, check corrected k_s and C_ht against the hard-coded values ÷ 3.2808.
4. **Invariance check, fast with v2.** On AP1 and AP5, the corrected-unit v2 profiles must equal the feet-unit profiles with K and D scaled by 3.2808 and MSE by 0.0929. Check the A_crit profile shape to 10⁻⁸ relative. This confirms the algebra. AP results are then rerun, not corrected algebraically.
5. **Bounds.** Scale the AP K and D bounds by 3.2808 and record the change in config.

Report: `reports/v2/05_ap_units.md`.

## Phase 6. Rerun the base analyses (Slurm)

On corrected grids, all ten tiles, submitted as job arrays. Print the core-hour estimate and ask before submitting. Run:

1. **The tile pipeline**, all stages, with v2 and with v1 (v1 for continuity with the published numbers; skip v1 if Steve decides so at Phase 4).
2. **Perron et al. (2012) forward runs** (Fig. 2), using regional parameters with metre grids.
3. **Fig. 3** (ΔK, ΔD) and **Table 1**, generated by `lemcvt tables` and `lemcvt figures`.
4. **Resolution sweep** (supp_fig6 and the D-inf variant), 5–100 m, both sites and both routings. This covers AP too: in v1, `supp_fig6` and `resolution_study_dinf` ran AP in feet.
5. **Synthetic Péclet analysis** (supp_fig5). It has no AP dependence, so it is regression-only.
6. **D8 against D∞ comparison and the slope–area figure.**
7. **Nonlinear-transport sensitivity on AP**, with the fraction of hillslope pixels with |∇z| > 0.5 S_c before and after correction.
8. **Tile-selection criterion.** Recompute it on corrected AP values for all candidate tiles, if the candidate list was found in Phase 0. Report whether the same five tiles would be chosen.

**Deliverables:**

- `results/v2.0.0/`: all results.
- `figures/` and `tables/`: rebuilt with one command.
- `reports/v2/06_rerun.md`, containing:
  - old against new for every number in Table 1 and every figure's headline quantity;
  - a statement, per manuscript claim, of whether it holds, changes in number only, or changes in substance;
  - `corrections_table.md`, which lists every number in the manuscript, SI, Reviewer 1 response draft and game-plan memo with its old and new value. Steve applies these by hand.
  - a plain paragraph for the response letter disclosing the AP unit error, what it affected, what it did not affect, and the optimizer change if v2 is adopted.

**Stop for review.**

## Phase 7. Port the add-on analyses (after Phase 6 is accepted)

Port `width_analysis/`, `acrit_slope/` (Test 1) and `head_extraction/` (Test 2) from their feature branches into `lemcvt.analyses` as CLI steps. Regression-check each against its own reports on GM. Rerun each on corrected AP. Test 2 stops at its review gate as before.

## Deferred: not part of this directive

AP source data. After Phase 6, a separate directive will assess whether the PAMAP DEM's TIN facets affect the results, and if so locate better data (the PAMAP LAS point clouds, or USGS 3DEP lidar for Greene County, PA) and define tile re-extraction. Do not search or download for this now.

## Engineering rules (all phases)

- **Commits.** Small commits on `v2`, or on short-lived branches off `v2` per phase, merged after each gate.
- **CI.** GitHub Actions on push runs lint (`ruff`), type checks (`mypy` on `src/lemcvt`) and the fast test suite, which excludes the full-tile tests.
- **Reproducibility.** No result without a config snapshot, git hash and environment hash. No hard-coded paths; everything goes through config and environment variables (`LEMCVT_DATA`, `LEMCVT_RESULTS`).
- **Documentation.** The README covers installation on Sherlock, `lemcvt data fetch`, one-command reproduction of every figure and table, and the run-time table from Phase 3.
- **Reporting.** If any phase reveals a numerical issue beyond the four listed in Context, report it at the next checkpoint. Do not fix it silently.
