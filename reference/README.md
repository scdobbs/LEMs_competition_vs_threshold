# reference/ — frozen v1 reference (Phase 0 of refactor_v2_directive.md)

Everything in this directory describes the state of `v1.0.1` (commit 28a9c90) as
executed for the previous submission. Nothing here is new science; it is the
target that the v2 refactor must reproduce (Phase 2 regression gate).

| File | Produced by | Content |
| --- | --- | --- |
| `v1_reference.json` | `extract_v1_reference.py` (stdlib Python, login node) | Every number printed in the ten `notebooks/model_runs/*_complete.ipynb` notebooks, per tile and stage: hard-coded θ, k_s, hilltop curvature, U, n, S_c, K0/D0 (with the AP K0 overrides), bounds, AD / A_crit / nonlinear / joint optimizer results, forward-run parameters and saved-grid prefixes, papermill cell durations; plus the printed tables of the five figure notebooks (raw and parsed). |
| `v1_grids_manifest.csv` | `make_grid_manifest.py` (Slurm job `phase0_job.sbatch`) | sha256, GeoTIFF metadata (shape, dtype, nodata, geotransform, CRS) and min/max/mean/NaN counts for all 280 grids in `data/*/tile*/` and `data/*/tile*/outputs/`. |
| `v1_saved_grid_mse.csv` | `make_grid_manifest.py` | nan-aware MSE of every saved `*_elevation` output grid against the tile's observed filled grid, using the v1 misfit definition. These are the forward-regression targets (the v1 notebooks do not print forward-run MSEs). |
| `logs/` | `phase0_job.sbatch` | Manifest log, `lem-v1` environment build log and package list, `pytest` output in `lem-v1`, TopoAnalysis load check. |

Notes on the parsed values:

- Optimizer results are printed with 4 significant figures (`:.4e`); the JSON
  carries exactly those digits. The saved grids carry full double precision.
- `K0_history` records every assignment to `K0` in cell order. In AP tiles 1, 3
  and 5 the AD stage starts from an overridden `K0` (1e-6, 1e-6, 1e-5) while the
  partitioned stages re-instate `K0 = U / k_s` (cell 8).
- `stages.<name>.K0_at_call` / `D0_at_call` are the values in scope when each
  optimizer was called; the partitioned and nonlinear A_crit stages hold K at
  `K0_at_call` and D at `D0_at_call`.
- Cell durations come from papermill metadata (seconds).
