# API Reference

Complete reference for all public functions in the `src/` modules. Private helpers (prefixed with `_`) are omitted unless they are part of the documented resampling interface.

---

## `models.py` -- Landscape Evolution Model Factories

Each factory pre-computes sparse matrices and topological data from input rasters, then returns a lightweight closure that solves for steady-state elevation given physical parameters.

### `advection_diffusion_model`

```python
advection_diffusion_model(
    dem, area, flow_direction, m: float
) -> Callable[[float, float, float], np.ndarray]
```

Build a solver for the coupled advection-diffusion steady-state PDE: `0 = U - K A^m dz/dx + D nabla^2 z`.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `dem` | Elevation-like | Raster with `._griddata` (2-D array) and `._georef_info.dx` |
| `area` | Area-like | Raster with `._griddata` (drainage area) |
| `flow_direction` | FlowDirectionD8-like | Raster with `._griddata` (D8 codes) |
| `m` | `float` | Area exponent in the stream-power incision term |

**Returns:** `Callable[[float, float, float], np.ndarray]` -- closure `output_function(U, K, D)` returning 2-D steady-state elevation.

---

### `partitioned_threshold_model`

```python
partitioned_threshold_model(
    dem, area, flow_direction, m: float, U: float, K: float
) -> Callable[..., tuple[np.ndarray, np.ndarray]]
```

Build a partitioned threshold model with linear hillslope diffusion. Pre-computes the advective channel network, then returns a function that applies an area threshold and solves hillslope diffusion.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `dem` | Elevation-like | Observed DEM |
| `area` | Area-like | Drainage area raster |
| `flow_direction` | FlowDirectionD8-like | D8 flow direction raster |
| `m` | `float` | Area exponent in stream-power law |
| `U` | `float` | Uplift rate [m/yr] |
| `K` | `float` | Erodibility coefficient |

**Returns:** `Callable` -- closure `f(A_crit, D, debug=False) -> (Z_ss, channel_mask)` where `Z_ss` is 2-D steady-state elevation and `channel_mask` is a 2-D boolean array.

---

### `partitioned_threshold_model_nonlinear`

```python
partitioned_threshold_model_nonlinear(
    dem, area, flow_direction, m: float, U: float, K: float
) -> Callable[..., tuple[np.ndarray, np.ndarray]]
```

Build a partitioned threshold model with nonlinear (slope-dependent) hillslope diffusion: `D_eff = D / (1 - (|nabla z| / S_c)^2)`. The solver uses Picard iteration until convergence.

**Parameters:** Same as `partitioned_threshold_model`.

**Returns:** `Callable` -- closure `f(A_crit, D_linear, S_c, max_iter=50, tol=1e-6) -> (Z_ss, channel_mask)`.

---

## `dinf.py` -- D-infinity Flow Routing

D-infinity (Tarboton 1997) flow routing. Computes continuous flow angles from 8 triangular facets and distributes drainage area to two receivers proportionally.

### Constants

**`FACETS`** -- List of 8 tuples `(e1_di, e1_dj, e2_di, e2_dj, base_angle)` defining the triangular facets, numbered counterclockwise from east.

**`D8_CODE_TO_DINF`** -- Dict mapping D8 flow direction codes to `(angle, facet_idx, r_facet)` tuples for flat-cell fallback.

### `compute_dinf_flow_direction`

```python
compute_dinf_flow_direction(
    Z: np.ndarray, dx: float, FD_d8: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

Compute D-infinity flow direction from an elevation grid. For each interior cell, evaluates all 8 triangular facets and selects the one with maximum downhill slope.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `Z` | `np.ndarray` | 2-D elevation array (Ny x Nx). NaN marks invalid cells |
| `dx` | `float` | Cell spacing (square grid assumed) |
| `FD_d8` | `np.ndarray` or `None` | Optional D8 flow-direction grid for flat-cell fallback |

**Returns:** `(angles, r_facet, facet_idx)` -- absolute flow angles in [0, 2pi), within-facet angles in [0, pi/4], and selected facet indices (0-7). NaN/-1 for invalid cells.

---

### `get_receivers`

```python
get_receivers(
    j: int, i: int, fac_idx: int, r: float
) -> tuple[int, int, int, int, float, float]
```

Get the two receiver cells and proportional weights for a D-inf cell.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `j`, `i` | `int` | Row and column of the source cell |
| `fac_idx` | `int` | Facet index (0-7) |
| `r` | `float` | Within-facet angle in [0, pi/4] |

**Returns:** `(j1, i1, j2, i2, p1, p2)` -- receiver coordinates and proportional weights (p1 + p2 = 1).

---

### `compute_dinf_area`

```python
compute_dinf_area(
    Z: np.ndarray, dx: float,
    angles: np.ndarray, r_facet: np.ndarray, facet_idx: np.ndarray
) -> np.ndarray
```

Compute D-infinity drainage area by proportional distribution. Processes cells from highest to lowest elevation.

**Returns:** 2-D drainage area array.

---

### `compute_dinf_from_dem`

```python
compute_dinf_from_dem(dem_obj, fd_obj=None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

Convenience wrapper: compute D-inf routing from raster objects.

**Returns:** `(angles, r_facet, facet_idx, A_dinf)`.

---

## `models_dinf.py` -- D-infinity Model Factories

Mirrors the three model factories in `models.py` but uses D-infinity flow routing. Imports shared helpers (`_extract_grid_data`, `_make_boundary_mask`, `_solve_hillslope_diffusion`, `_find_boundary_value`) from `models.py`.

### `advection_diffusion_model_dinf`

```python
advection_diffusion_model_dinf(
    dem, area_d8, flow_direction, m: float,
    angles: np.ndarray, r_facet: np.ndarray,
    facet_idx: np.ndarray, A_dinf: np.ndarray
) -> Callable[[float, float, float], np.ndarray]
```

Build a solver for the coupled AD PDE using D-infinity flow routing. The advection stencil distributes the incision term to two receivers proportionally based on the D-inf flow angle. The diffusion stencil (5-point Laplacian) is identical to D8.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `dem` | Elevation-like | Raster with `._griddata` and `._georef_info.dx` |
| `area_d8` | Area-like | D8 drainage area (used only for grid extraction) |
| `flow_direction` | FlowDirectionD8-like | D8 flow direction (used only for grid extraction) |
| `m` | `float` | Area exponent |
| `angles` | `np.ndarray` | D-inf flow angles from `compute_dinf_flow_direction` |
| `r_facet` | `np.ndarray` | Within-facet angles |
| `facet_idx` | `np.ndarray` | Facet indices |
| `A_dinf` | `np.ndarray` | D-inf drainage area |

**Returns:** `Callable[[float, float, float], np.ndarray]` -- closure `output_function(U, K, D)` returning 2-D steady-state elevation.

---

### `partitioned_threshold_model_dinf`

```python
partitioned_threshold_model_dinf(
    dem, area_d8, flow_direction, m: float, U: float, K: float,
    angles: np.ndarray, r_facet_arr: np.ndarray,
    facet_idx_arr: np.ndarray, A_dinf: np.ndarray
) -> Callable[..., tuple[np.ndarray, np.ndarray]]
```

Build a partitioned threshold model with D-inf channel integration and linear hillslope diffusion.

**Returns:** `Callable` -- closure `f(A_crit, D, debug=False) -> (Z_ss, channel_mask)`.

---

### `partitioned_threshold_model_nonlinear_dinf`

```python
partitioned_threshold_model_nonlinear_dinf(
    dem, area_d8, flow_direction, m: float, U: float, K: float,
    angles: np.ndarray, r_facet_arr: np.ndarray,
    facet_idx_arr: np.ndarray, A_dinf: np.ndarray
) -> Callable[..., tuple[np.ndarray, np.ndarray]]
```

Build a partitioned threshold model with nonlinear diffusion and D-inf channel integration.

**Returns:** `Callable` -- closure `f(A_crit, D_linear, S_c, max_iter=50, tol=1e-6) -> (Z_ss, channel_mask)`.

---

## `optimization.py` -- Optimization Wrappers

Each function wraps `scipy.optimize.minimize` with log-space parameterization, LRU caching of forward solves, and MSE misfit computation against observed elevations.

### `optimize_advection_diffusion`

```python
optimize_advection_diffusion(
    solver, dem_griddata: np.ndarray, U: float,
    K0: float, D0: float,
    K_bounds=(1e-12, 2e-3), D_bounds=(4e-10, 1e-2),
    method="L-BFGS-B"
) -> tuple[float, float, float]
```

Optimize K and D for the advection-diffusion model.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `solver` | `Callable` | `output_function(U, K, D) -> Z` from `advection_diffusion_model` |
| `dem_griddata` | `np.ndarray` | Observed elevation grid (for MSE) |
| `U` | `float` | Uplift rate (held fixed) |
| `K0`, `D0` | `float` | Initial guesses for erodibility and diffusivity |
| `K_bounds`, `D_bounds` | `tuple[float, float]` | Parameter bounds |
| `method` | `str` | `scipy.optimize.minimize` method (default `"L-BFGS-B"`) |

**Returns:** `(K_star, D_star, mse_star)` -- optimized erodibility, diffusivity, and final MSE.

---

### `optimize_threshold_acrit`

```python
optimize_threshold_acrit(
    partitioned_model, dem_griddata: np.ndarray,
    D0: float, A_crit_0=5000.0, A_crit_bounds=(1e2, 1e5),
    method="Nelder-Mead"
) -> tuple[float, float]
```

Optimize A_crit for the linear partitioned threshold model, holding D fixed.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `partitioned_model` | `Callable` | `f(A_crit, D) -> (Z_ss, mask)` from `partitioned_threshold_model` |
| `dem_griddata` | `np.ndarray` | Observed elevation grid |
| `D0` | `float` | Fixed diffusivity |
| `A_crit_0` | `float` | Initial guess for critical area |
| `A_crit_bounds` | `tuple[float, float]` | Bounds for A_crit |
| `method` | `str` | Optimization method (default `"Nelder-Mead"`) |

**Returns:** `(A_crit_star, mse_star)` -- optimized critical area and final MSE.

---

### `optimize_threshold_acrit_nonlinear`

```python
optimize_threshold_acrit_nonlinear(
    partitioned_model_nl, dem_griddata: np.ndarray,
    D0: float, Sc: float,
    A_crit_0=5000.0, A_crit_bounds=(1e2, 1e5),
    method="Nelder-Mead"
) -> tuple[float, float]
```

Optimize A_crit for the nonlinear partitioned threshold model, holding D and S_c fixed.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `partitioned_model_nl` | `Callable` | `f(A_crit, D_linear, S_c) -> (Z_ss, mask)` from `partitioned_threshold_model_nonlinear` |
| `dem_griddata` | `np.ndarray` | Observed elevation grid |
| `D0` | `float` | Fixed linear diffusivity |
| `Sc` | `float` | Critical slope for nonlinear diffusion |
| `A_crit_0` | `float` | Initial guess for critical area |
| `A_crit_bounds` | `tuple[float, float]` | Bounds for A_crit |
| `method` | `str` | Optimization method |

**Returns:** `(A_crit_star, mse_star)` -- optimized critical area and final MSE.

---

### `optimize_threshold_full`

```python
optimize_threshold_full(
    dem, area, flow_direction, m: float, U: float,
    dem_griddata: np.ndarray,
    K0: float, D0: float, A_crit_0=5000.0,
    K_bounds=(1e-12, 2e-3), D_bounds=(4e-10, 1e-2),
    A_crit_bounds=(1e2, 1e5), method="Nelder-Mead"
) -> tuple[float, float, float, float]
```

Jointly optimize A_crit, D, and K for the partitioned threshold model. Rebuilds the model for each candidate K, so this is slower than single-parameter optimization.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `dem`, `area`, `flow_direction` | raster-like | Input grids |
| `m` | `float` | Area exponent |
| `U` | `float` | Uplift rate |
| `dem_griddata` | `np.ndarray` | Observed elevation for MSE |
| `K0`, `D0`, `A_crit_0` | `float` | Initial guesses |
| `K_bounds`, `D_bounds`, `A_crit_bounds` | `tuple[float, float]` | Parameter bounds |
| `method` | `str` | Optimization method |

**Returns:** `(A_crit_star, D_star, K_star, mse_star)` -- optimized parameters and final MSE.

---

## `analysis.py` -- Slope-Area Regression and Channel Steepness

### `slope_area_regression_binned`

```python
slope_area_regression_binned(
    elev_obj, area_obj, fd_obj, min_area: float,
    vertical_interval=10, nbins=24, min_per_bin=10,
    agg="median", require_weighted=True, relax_if_sparse=True
) -> tuple[float, float, dict]
```

Log-binned slope-area regression for channel steepness and concavity. Estimates ks and theta from `S = ks * A^(-theta)` using weighted least-squares on bin aggregates.

**Returns:** `(ks, theta, debug)` where `debug` is a dict with keys `A`, `S`, `logA`, `logS`, `bin_edges`, `bin_centers`, `bin_logA`, `bin_logS`, `bin_counts`.

**Raises:** `ValueError` if insufficient data or too few bins survive filtering.

See also: [`compute_iqr_errors`](#compute_iqr_errors), [`weighted_r2`](#weighted_r2)

---

### `compute_iqr_errors`

```python
compute_iqr_errors(dbg: dict) -> tuple[np.ndarray, np.ndarray]
```

Compute 25-75% IQR error bars for each bin from a `slope_area_regression_binned` debug dict.

**Returns:** `(lower_err, upper_err)` -- arrays of `median - q25` and `q75 - median` per bin.

---

### `weighted_r2`

```python
weighted_r2(x: np.ndarray, y: np.ndarray, yhat: np.ndarray, w: np.ndarray) -> float
```

Weighted coefficient of determination. The `x` parameter is unused (kept for API compatibility).

**Returns:** Weighted R-squared value.

---

### `fit_slope_area`

```python
fit_slope_area(
    A_1d: np.ndarray, S_1d: np.ndarray,
    theta_fixed: float | None = None, min_points: int = 5
) -> dict
```

Log-log slope-area regression with fixed or free concavity.

**Returns:** `dict` with keys: `ks`, `theta`, `r2`, `n_good`, `logA`, `logS`, `logS_pred`, `intercept`, `slope`, `used_mask`.

---

### `ks_local_from_SA`

```python
ks_local_from_SA(
    A_1d: np.ndarray, S_1d: np.ndarray, m: float, min_points: int = 5
) -> tuple[np.ndarray, np.ndarray, int, float, float, float]
```

Local steepness proxy `ks = S * A^m` along a channel.

**Returns:** `(ks_local, mask, n_good, median, q25, q75)` -- per-pixel steepness array, valid mask, count, and summary statistics.

---

### `ks_from_loglog_fixed_theta`

```python
ks_from_loglog_fixed_theta(
    A: np.ndarray, S: np.ndarray, theta_fixed: float
) -> tuple[float, float, float]
```

Estimate Ks with fixed concavity via mean log-space intercept.

**Returns:** `(Ks, theta, r2)` -- channel steepness, echo of theta_fixed, and R-squared.

---

### `slope_area_arrays_from_elevation`

```python
slope_area_arrays_from_elevation(
    elev_obj, area_obj, fd_obj,
    min_area: float = 1e5, vertical_interval: int = 10
) -> tuple[np.ndarray, np.ndarray]
```

Extract paired (A, S) arrays from TopoAnalysis raster objects.

**Returns:** `(A, S)` -- 1-D arrays of area and slope for valid channel pixels.

---

### `ks_obs_from_observed`

```python
ks_obs_from_observed(
    filled_obs, area, fd, theta: float,
    min_area: float = 1e5, vertical_interval: int = 10
) -> tuple[float, float, int, float]
```

Compute Ks from observed elevation at a given resolution.

**Returns:** `(Ks, r2, n_pts, A_max)`.

See also: [`ks_mod_from_modeled`](#ks_mod_from_modeled)

---

### `ks_mod_from_modeled`

```python
ks_mod_from_modeled(
    Z_model: np.ndarray, georef_template_dem, area, fd,
    theta: float, min_area: float = 1e5, vertical_interval: int = 10
) -> tuple[float, float, int, float]
```

Compute Ks from a modeled elevation array. Wraps the modeled array in a TopoAnalysis Elevation object using the template DEM's georef info.

**Returns:** `(Ks, r2, n_pts, A_max)`.

---

## `synthetic.py` -- Synthetic Planform Solvers

Functions for building forced-routing V-shaped planform grids and solving the steady-state advection-diffusion PDE on them. Used for Peclet-number analysis (Supplementary Figure 5).

### Grid Construction

#### `compute_area_planform`

```python
compute_area_planform(
    Nx: int, Ny: int, dx: float, dy: float, A_top: float = 0.0
) -> np.ndarray
```

Build a synthetic V-shaped planform drainage area grid. Flow routes inward from left and right edges toward the centre column, then southward.

**Returns:** `(Ny, Nx)` drainage area grid.

**Raises:** `ValueError` if Nx is even.

---

#### `downstream_link`

```python
downstream_link(Nx: int, Ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]
```

D8 flow routing on forced V-shaped planform.

**Returns:** `(i_dn, j_dn, is_outlet)` -- downstream column index, row index, and boolean outlet mask.

---

### PDE Solvers

#### `solve_advection_diffusion_planform`

```python
solve_advection_diffusion_planform(
    U, K, m, D: float, Nx, Ny: int, Lx, Ly: float, A_top=0.0
) -> tuple[np.ndarray, np.ndarray, float, float]
```

Solve steady-state advection-diffusion on a synthetic planform with no-flux lateral BCs and z=0 at the outlet.

**Returns:** `(Z, A, dx, dy)` -- elevation grid, area grid, and cell spacings.

**Raises:** `ValueError` if Nx is even or dimensions < 5.

---

#### `solve_advection_diffusion_planform_noflux`

```python
solve_advection_diffusion_planform_noflux(
    U, K, m, D: float, Nx, Ny: int, Lx, Ly: float, A_top=0.0
) -> tuple[np.ndarray, np.ndarray, float, float]
```

Like `solve_advection_diffusion_planform` but enforces no-flux BCs on top, left, and right boundaries as explicit `z_boundary = z_adjacent` rows.

**Returns:** `(Z, A, dx, dy)`.

---

### Peclet-Number Utilities

#### `inlet_area_from_Pe`

```python
inlet_area_from_Pe(Pe, D, K, L_PE, m: float) -> float
```

Compute inlet area from Peclet number: `A_in = (Pe * D / (K * L))^(1/m)`.

---

#### `D_from_Pe`

```python
D_from_Pe(Pe, A_inlet, K, L_PE, m: float) -> float
```

Compute diffusivity from Peclet number and inlet area.

---

#### `Pe_from_D_Ain`

```python
Pe_from_D_Ain(D, A_inlet, K, L_PE, m: float) -> float
```

Compute Peclet number from diffusivity and inlet area.

---

### Analysis Helpers

#### `channel_slope_centerline`

```python
channel_slope_centerline(Z: np.ndarray, Nx, Ny: int, dx, dy: float) -> np.ndarray
```

Extract slope along the centre-column of a planform grid.

**Returns:** 1-D array of slopes (length Ny; last element is NaN).

---

#### `A_top_for_target_inlet_area`

```python
A_top_for_target_inlet_area(
    Nx, Ny: int, Lx, Ly: float, A_inlet_target: float
) -> float
```

Compute upstream boundary area (`A_top`) needed to achieve a target inlet area at the top of the centre column.

---

### Diagnostics

#### `laplacian_2d`

```python
laplacian_2d(Z: np.ndarray, dx, dy: float) -> np.ndarray
```

Compute the 2-D Laplacian via second-order finite differences. Interior cells use central differences; boundary cells are NaN.

---

#### `interior_slice`

```python
interior_slice(Ny: int, frac_trim: float) -> slice
```

Return a slice that trims `frac_trim` fraction from top and bottom of Ny rows.

---

#### `interior_slice_indices`

```python
interior_slice_indices(Ny: int, frac_trim: float) -> tuple[int, int]
```

Return `(j0, j1)` indices that trim `frac_trim` from top and bottom.

---

## `utils.py` -- General-Purpose Utilities

### `find_project_root`

```python
find_project_root(start: str | Path | None = None, marker_dir: str = "data") -> Path
```

Walk upward from `start` to find the repository root containing `marker_dir`.

**Raises:** `FileNotFoundError` if the marker directory cannot be found.

---

### `load_tile_data`

```python
load_tile_data(base_path: str | Path, region: str) -> tuple[dem, area, flow_direction]
```

Load DEM, area, and flow-direction rasters from a standard tile path. Expects filenames of the form `<region>_1m_best_tile_{filled,area,fd}`.

**Parameters:**
| Name | Type | Description |
|------|------|-------------|
| `base_path` | `str` or `Path` | Directory containing the tile rasters (e.g., `data/AP/tile1/`) |
| `region` | `str` | Region code (`"AP"` or `"GM"`) |

**Returns:** `(dem, area, flow_direction)` -- TopoAnalysis raster objects.

---

### `save_model_outputs`

```python
save_model_outputs(
    dem, Z_model: np.ndarray, save_dir: str | Path, prefix: str,
    inclination=20.0, azimuth=315.0
) -> str
```

Save modeled elevation and derived grids (filled DEM, flow direction, area, log-area, hillshade) to `save_dir` with filenames starting with `prefix`.

**Returns:** The `prefix` string.

---

### `crop`

```python
crop(
    dem, area, flow_direction,
    extent_indexes: tuple[int, int, int, int] = (200, 800, 200, 800)
) -> tuple[dem_clip, area_clip, fd_clip]
```

Crop DEM, area, and flow-direction rasters to a pixel-index extent `(row_min, row_max, col_min, col_max)`.

---

### `compute_slope_magnitude`

```python
compute_slope_magnitude(Z: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray
```

Compute slope magnitude from an elevation grid using `np.gradient`. Optionally sets slope to NaN outside the given mask.

---

### `resample_from_1m`

```python
resample_from_1m(dem_1m_filled, dx_target: float, order: int = 1) -> new_obj
```

Downsample a 1-m DEM to a target resolution. Returns a deep copy with resampled grid and updated georef info.

---

### `_nanaware_zoom`

```python
_nanaware_zoom(grid: np.ndarray, zoom_yx: tuple[float, float], order: int = 1) -> np.ndarray
```

Resample a 2-D grid with NaN-aware interpolation. Replaces NaN with 0 before zooming, tracks valid-pixel weights, and restores NaN where coverage is insufficient.

---

### `_infer_dx_from_georef`

```python
_infer_dx_from_georef(gi) -> tuple[float, float]
```

Extract pixel size `(dx, dy)` from a georef info object. Tries `geoTransform` first, then falls back to a `dx` attribute.

---

## `plotting.py` -- Plotting Utilities

### `set_nature_style`

```python
set_nature_style() -> None
```

Apply Nature-style matplotlib defaults (300 dpi, Helvetica, 8 pt). Safe to call multiple times.

---

### `plot_maps_area_slope`

```python
plot_maps_area_slope(
    dem, area, flow_direction, Z_model: np.ndarray,
    channel_mask: np.ndarray | None = None,
    vertical_interval=10, min_area=2000, figsize=(18, 5)
) -> None
```

Plot a 4-panel figure: (1) observed elevation, (2) modeled elevation, (3) shared colorbar, (4) log-log slope vs. area with best-fit lines for both observed and modeled.

---

### `plot_maps_and_channel_mask`

```python
plot_maps_and_channel_mask(
    dem, Z_model: np.ndarray, mask: np.ndarray, figsize=(18, 5)
) -> None
```

Plot a 4-panel figure: (1) observed elevation, (2) modeled elevation, (3) shared colorbar, (4) channel mask (binary, seismic colormap).
