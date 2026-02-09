# Developer Guide

Architecture, testing patterns, and conventions for contributing to this codebase.

---

## 1. Architecture Principles

### Single Source of Truth

All reusable logic lives in `src/`. Notebooks are for orchestration and visualization only -- they call `src/` functions and should not redefine them inline. This ensures:

- Bug fixes propagate to all 10 model-run notebooks and 3 figure notebooks at once.
- Tests can exercise the core logic without running notebooks.
- New model variants only need to be implemented in one place.

### Model Factories Return Closures

The model functions in `models.py` follow a factory pattern:

```python
solver = advection_diffusion_model(dem, area, fd, m)  # expensive setup
Z = solver(U, K, D)                                    # cheap evaluation
```

The factory call (`advection_diffusion_model`) pre-computes sparse matrices, topological sorts, and boundary conditions. The returned closure (`solver`) only needs to assemble and solve a linear system. This design enables efficient optimization: the optimizer calls the closure hundreds of times without repeating the expensive setup.

### Optimization Wrappers Handle Internals

Each optimizer in `optimization.py` encapsulates three concerns:

1. **Log-space parameterization** -- converts `[log10(K), log10(D)]` to physical units.
2. **LRU caching** -- avoids redundant forward solves when the optimizer revisits parameter combinations.
3. **`scipy.minimize` configuration** -- method selection, bounds, and convergence handling.

Notebooks see only `optimize_advection_diffusion(solver, dem._griddata, U, K0, D0)` and receive `(K_star, D_star, mse)`.

---

## 2. Module Dependency Graph

```
notebooks
├── optimization.py
│   ├── models.py          (D8 model factories)
│   └── (scipy.optimize)
├── models_dinf.py         (D-inf model factories)
│   ├── dinf.py            (D-inf flow routing)
│   └── models.py          (shared helpers: _extract_grid_data, etc.)
├── dinf.py                (standalone D-inf flow routing)
├── analysis.py            (slope-area regression, ks estimation)
│   └── TopoAnalysis       (ChannelSlopeWithSmoothing)
├── utils.py               (I/O, crop, slope, resampling)
│   └── TopoAnalysis       (Elevation, Area, FlowDirectionD8, etc.)
├── plotting.py            (Nature-style figures)
│   └── TopoAnalysis       (ChannelSlopeWithSmoothing)
└── synthetic.py           (standalone, for supp_fig5)
    └── (scipy.sparse)
```

Key observations:

- `models.py` has no dependency on TopoAnalysis (it works with raw arrays extracted from raster objects).
- `models_dinf.py` imports shared private helpers from `models.py` (`_extract_grid_data`, `_make_boundary_mask`, `_solve_hillslope_diffusion`, `_find_boundary_value`) and flow routing from `dinf.py`. It does not duplicate any D8 logic.
- `dinf.py` is standalone (NumPy only) -- no dependency on TopoAnalysis or models.py.
- `synthetic.py` is fully standalone -- it builds its own grids and has no external dependencies beyond NumPy and SciPy.
- `plotting.py`, `analysis.py`, and `utils.py` import TopoAnalysis at call time (lazy imports), so tests can run without it.

---

## 3. Adding a New Model Variant

To add a new landscape evolution model (e.g., a spatially-variable K model):

1. **Add a model factory to `models.py`** following the existing pattern:
   ```python
   def variable_k_model(dem, area, flow_direction, m, K_field):
       gd = _extract_grid_data(dem, area, flow_direction)
       # ... pre-compute sparse matrices ...

       def output_function(U, D):
           # ... assemble and solve ...
           return Z

       return output_function
   ```

2. **Add an optimizer wrapper to `optimization.py`**:
   ```python
   def optimize_variable_k(solver, dem_griddata, U, D0, ...):
       @lru_cache(maxsize=None)
       def _solve(log10_D):
           return solver(U, 10 ** log10_D)

       def _misfit(x):
           Z_mod = _solve(float(x[0]))
           return float(np.nanmean((Z_mod - dem_griddata) ** 2))

       # ... minimize and return ...
   ```

3. **Add a notebook cell** calling the new functions, following the existing 4-stage pattern.

4. **Add tests** for the new model factory (see Testing Patterns below).

---

## 4. Testing Patterns

Tests live in `tests/` and mirror the `src/` module structure. Run all tests with:

```bash
python3 -m pytest tests/ -v
```

### Import Pattern

Tests use the same `sys.path.insert` pattern as notebooks -- no package installation is required:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
```

### Mock Objects

Since TopoAnalysis may not be installed in the test environment, `test_utils.py` provides mock raster objects:

- **`_MockGeorefInfo`** -- mimics `._georef_info` with `dx`, `nx`, `ny`, and `geoTransform` attributes.
- **`_MockRaster`** -- mimics TopoAnalysis raster objects with `._griddata`, `._georef_info`, `._nodata_value`, `._rowscols_to_xy()`, and `.clip_to_extent()`.

These mocks support testing `crop()`, `compute_slope_magnitude()`, `resample_from_1m()`, and other functions that interact with raster metadata.

### Synthetic Data Verification

`test_analysis.py` verifies regression functions against known power-law data:

- Generate synthetic `A` and `S` arrays with known ks and theta.
- Confirm that `fit_slope_area()`, `ks_from_loglog_fixed_theta()`, and `ks_local_from_SA()` recover the known parameters within tolerance.

### PDE Solver Sanity Checks

`test_synthetic.py` validates the synthetic planform solvers:

- **Flat surface test:** With U=0, the solution should be uniformly zero.
- **Tilted plane test:** Verify that channel slopes match the theoretical `S = (U/K) / A^m`.
- **Outlet elevation:** Confirm z=0 at the outlet cell.
- **Grid construction:** Verify `compute_area_planform` produces monotonically increasing area along the centre column.

### Test Coverage by Module

| Test File | Module | Key Checks |
|-----------|--------|------------|
| `test_models.py` | `models.py` | Factory return types, boundary conditions, solver convergence |
| `test_models_dinf.py` | `models_dinf.py` | Output shape/type, boundary conditions, determinism, cardinal-grid D8 equivalence |
| `test_dinf.py` | `dinf.py` | Tilted plane angles, area conservation, symmetry, angle ranges, flat-cell D8 fallback, receiver proportions, no-NaN guarantee |
| `test_optimization.py` | `optimization.py` | Known-minimum recovery, log-space transform correctness |
| `test_analysis.py` | `analysis.py` | Power-law recovery, edge cases (empty arrays, insufficient bins) |
| `test_synthetic.py` | `synthetic.py` | Flat/tilted solutions, area grid symmetry, grid dimension validation |
| `test_utils.py` | `utils.py` | Crop geometry, slope computation, NaN-aware resampling |
| `test_plotting.py` | `plotting.py` | `set_nature_style` rcParams, function signatures |

---

## 5. Notebook Transformation Scripts

The `scripts/` directory contains one-time migration tools that were used during the refactoring from inline code to `src/` functions. They are kept for audit trail purposes.

| Script | Purpose |
|--------|---------|
| `consolidate_model_notebooks.py` | Content-based cell identification and replacement for the 10 model-run notebooks |
| `update_model_notebooks.py` | Update model notebook cells to call `src/` functions |
| `update_figure_notebooks.py` | Update figure notebook cells to call `src/` functions |
| `fix_supp_fig5.py` | Remove inline function redefinitions from supp_fig5.ipynb |
| `fix_supp_fig6.py` | Remove inline function definitions from supp_fig6.ipynb |

These scripts are not part of the regular workflow and do not need to be re-run.

---

## 6. Style Conventions

### Publication Figures

All notebooks that generate figures should call `set_nature_style()` at the top:

```python
from plotting import set_nature_style
set_nature_style()
```

This configures matplotlib for Nature formatting: 300 dpi, Helvetica/Arial font, 8 pt text, 0.5 pt axis lines, inward ticks.

### Import Pattern

All notebooks use a relative path import for `src/`:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("../../src")))
```

This works from both `notebooks/model_runs/` and `notebooks/figures/` (both are two levels below the repository root).

### Docstrings

All `src/` functions have NumPy-style docstrings with:

- Summary line
- Extended description (where non-trivial)
- `Parameters` section with types
- `Returns` section
- `Raises` section (where applicable)

### Logging

`src/` modules use `logging.getLogger(__name__)` for diagnostic output. Notebooks can enable debug logging by setting the logger level:

```python
import logging
logging.getLogger("models").setLevel(logging.DEBUG)
```
