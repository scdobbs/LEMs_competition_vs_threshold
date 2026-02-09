# Scientific Workflow Guide

This document describes the end-to-end modeling pipeline, from raw LiDAR data through calibrated model outputs and figure generation.

---

## 1. Data Preparation

### Input Data

The analysis uses 1-m resolution LiDAR DEMs from two geomorphic settings:

- **Allegheny Plateau (AP)** -- 5 tiles in `data/AP/tile{1..5}/`
- **Gabilan Mesa (GM)** -- 5 tiles in `data/GM/tile{1..5}/`

Each tile directory contains three pre-processed rasters (in TopoAnalysis format):

| File | Description |
|------|-------------|
| `{region}_1m_best_tile_filled` | Pit-filled elevation |
| `{region}_1m_best_tile_area` | D8 drainage area |
| `{region}_1m_best_tile_fd` | D8 flow direction codes |

### Loading and Cropping

Data loading is handled by `load_tile_data()` in `utils.py`:

```python
from utils import load_tile_data, crop

dem, area, fd = load_tile_data("../../data/AP/tile1/", region="AP")
```

If a subset of the full tile is needed, `crop()` extracts a pixel-index window:

```python
dem, area, fd = crop(dem, area, fd, extent_indexes=(200, 800, 200, 800))
```

---

## 2. Model-Run Pipeline

Each of the 10 model-run notebooks (`notebooks/model_runs/`) follows an identical 4-stage pipeline. The stages progressively refine the model from a simple advection-diffusion PDE to a fully calibrated threshold-partitioned model.

### Stage 1: Advection-Diffusion (AD) Optimization

**Goal:** Find the best-fitting K (erodibility) and D (diffusivity) for the coupled PDE over the entire domain.

**Governing equation:**

```
0 = U - K A^m dz/dx + D nabla^2 z
```

**Code path:**

1. Build the model factory:
   ```python
   from models import advection_diffusion_model
   solver = advection_diffusion_model(dem, area, fd, m)
   ```
   This pre-computes sparse advection and diffusion stencil matrices. The returned closure `solver(U, K, D)` solves for steady-state elevation.

2. Optimize K and D:
   ```python
   from optimization import optimize_advection_diffusion
   K_star, D_star, mse = optimize_advection_diffusion(
       solver, dem._griddata, U, K0, D0, K_bounds, D_bounds
   )
   ```
   Internally, parameters are optimized in log10-space using L-BFGS-B. An LRU cache avoids redundant forward solves.

3. Save outputs:
   ```python
   from utils import save_model_outputs
   save_model_outputs(dem, Z_model, save_dir, "AP_1m_best_tile_AD-opt")
   ```

### Stage 2: Threshold A_crit Optimization (Linear Diffusion)

**Goal:** Introduce area-threshold partitioning. Channels (A >= A_crit) follow the detachment-limited stream-power law; hillslopes (A < A_crit) are governed by linear diffusion. Optimize A_crit with K and D held fixed at their AD-optimized values.

**Governing equations:**

- Channels: `0 = U - K A^m dz/dx` (integrated upstream from boundaries)
- Hillslopes: `0 = U + D nabla^2 z`

**Code path:**

1. Build the threshold model:
   ```python
   from models import partitioned_threshold_model
   model = partitioned_threshold_model(dem, area, fd, m, U, K0)
   ```

2. Optimize A_crit:
   ```python
   from optimization import optimize_threshold_acrit
   A_crit_star, mse = optimize_threshold_acrit(model, dem._griddata, D0)
   ```

3. Save and compute slopes:
   ```python
   from utils import compute_slope_magnitude
   Z_model, mask = model(A_crit_star, D0)
   slopes = compute_slope_magnitude(Z_model, mask=~mask)
   ```

### Stage 3: Nonlinear A_crit Optimization

**Goal:** Same as Stage 2 but with slope-dependent hillslope diffusion.

**Nonlinear flux law:**

```
q_s = -D nabla(z) / (1 - (|nabla z| / S_c)^2)
```

The effective diffusivity increases as slopes approach the critical slope S_c, producing convex-up hillslope profiles with steeper near-channel slopes.

**Code path:**

```python
from models import partitioned_threshold_model_nonlinear
model_nl = partitioned_threshold_model_nonlinear(dem, area, fd, m, U, K0)

from optimization import optimize_threshold_acrit_nonlinear
A_crit_star, mse = optimize_threshold_acrit_nonlinear(
    model_nl, dem._griddata, D0, Sc
)
```

The solver uses Picard iteration (default max 50 iterations, tolerance 1e-6) to handle the nonlinearity.

### Stage 4: Full Optimization (Joint K, D, A_crit)

**Goal:** Jointly optimize all three parameters in the threshold model.

```python
from optimization import optimize_threshold_full
A_crit_star, D_star, K_star, mse = optimize_threshold_full(
    dem, area, fd, m, U, dem._griddata,
    K0, D0, A_crit_0, K_bounds, D_bounds, A_crit_bounds
)
```

This is the slowest stage because it rebuilds the partitioned model (re-integrating the channel network) for each candidate K. Uses Nelder-Mead in log10-space.

### Output Convention

Each stage writes 6 raster files to the tile's `outputs/` directory via `save_model_outputs()`:

| Suffix | Content |
|--------|---------|
| `_elevation` | Modeled steady-state elevation |
| `_filled` | Pit-filled modeled elevation |
| `_flow_direction` | D8 flow direction from modeled elevation |
| `_area` | Drainage area from modeled flow direction |
| `_logarea` | Log10 drainage area |
| `_hillshade` | Hillshade rendering |

Filename prefixes encode the model variant: `AD-no-opt`, `AD-opt`, `Partitioned-model-a_crit-opt`, `Partitioned-model-a_crit-k-d-opt`.

---

## 3. Mathematical Background

### Advection-Diffusion PDE

The steady-state landscape evolution equation combines fluvial incision (advection) and hillslope transport (diffusion):

```
0 = U - K A^m S + D nabla^2 z
```

where:

| Symbol | Meaning | Units |
|--------|---------|-------|
| U | Rock uplift rate | m/yr |
| K | Erodibility coefficient | m^(1-2m) / yr |
| A | Drainage area | m^2 |
| m | Area exponent (concavity) | dimensionless |
| S | Channel slope (dz/dx along flow) | dimensionless |
| D | Hillslope diffusivity | m^2/yr |
| z | Elevation | m |

### Threshold Partitioning

The landscape is divided at a critical drainage area A_crit:

- **Channels** (A >= A_crit): Elevation determined by integrating the detachment-limited slope `S = (U/K) / A^m` upstream from boundary elevations.
- **Hillslopes** (A < A_crit): Elevation determined by solving the Poisson equation `0 = U + D nabla^2 z` with channel-cell elevations as boundary conditions.

### Nonlinear Diffusion

For the nonlinear variant, the effective diffusivity is:

```
D_eff = D / (1 - (|nabla z| / S_c)^2)
```

As `|nabla z|` approaches the critical slope S_c, transport efficiency diverges, bounding hillslope gradients. The resulting nonlinear system is solved iteratively (Picard iteration).

### Optimization Strategy

All optimization is performed in log10-space to handle the large dynamic range of parameters:

- `log10(K)` typically ranges from -12 to -3
- `log10(D)` typically ranges from -10 to -2
- `log10(A_crit)` typically ranges from 2 to 5

Forward solves are cached with `functools.lru_cache` to avoid redundant PDE solves when the optimizer revisits parameter combinations. L-BFGS-B is used for the AD model (gradient-based, box-constrained); Nelder-Mead for threshold models (derivative-free).

---

## 4. Supplementary Analyses

### Peclet-Number Analysis (supp_fig5.ipynb)

This notebook investigates how the Peclet number controls the transition between diffusion-dominated and advection-dominated landscape morphology using synthetic V-shaped planform grids.

**Peclet number:**

```
Pe = K L A_in^m / D
```

The workflow:

1. For a range of Pe values, compute the corresponding diffusivity D using `D_from_Pe()` (or inlet area using `inlet_area_from_Pe()`).
2. Solve the steady-state PDE on a synthetic planform grid using `solve_advection_diffusion_planform_noflux()`.
3. Extract centreline slopes using `channel_slope_centerline()`.
4. Trim boundary effects using `interior_slice()` / `interior_slice_indices()`.
5. Compute the Laplacian diagnostic with `laplacian_2d()` to verify PDE residuals.

Key functions from `synthetic.py`: `compute_area_planform`, `downstream_link`, `solve_advection_diffusion_planform_noflux`, `inlet_area_from_Pe`, `D_from_Pe`, `Pe_from_D_Ain`, `channel_slope_centerline`, `A_top_for_target_inlet_area`, `laplacian_2d`, `interior_slice`, `interior_slice_indices`.

### Resolution Study (supp_fig6.ipynb)

This notebook examines how DEM resolution affects channel steepness (ks) estimation.

The workflow:

1. Load a 1-m DEM tile.
2. Downsample to multiple target resolutions using `resample_from_1m()`.
3. At each resolution, recompute flow routing and drainage area.
4. Estimate ks from observed and modeled data using `ks_obs_from_observed()` and `ks_mod_from_modeled()`.
5. Compare ks estimates across resolutions.

Key functions from `utils.py`: `resample_from_1m`, `_nanaware_zoom`, `_infer_dx_from_georef`. Key functions from `analysis.py`: `ks_obs_from_observed`, `ks_mod_from_modeled`, `slope_area_arrays_from_elevation`, `ks_from_loglog_fixed_theta`.

---

## 5. Figure Generation

| Notebook | Figures Produced | Key Functions Used |
|----------|-----------------|-------------------|
| `fig_slope_area.ipynb` | Main-text slope-area regression figures | `slope_area_regression_binned`, `compute_iqr_errors`, `weighted_r2`, `fit_slope_area` |
| `supp_fig5.ipynb` | Peclet-number sensitivity (Supplementary Figure 5) | `solve_advection_diffusion_planform_noflux`, `channel_slope_centerline`, `laplacian_2d`, Peclet utilities |
| `supp_fig6.ipynb` | Resolution dependence of ks (Supplementary Figure 6) | `resample_from_1m`, `ks_obs_from_observed`, `ks_mod_from_modeled` |
| Model-run notebooks | Per-tile elevation and slope-area comparison plots | `plot_maps_area_slope`, `plot_maps_and_channel_mask`, `set_nature_style` |

All publication figures use `set_nature_style()` from `plotting.py` for consistent formatting (300 dpi, Helvetica, 8 pt).
