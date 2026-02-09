# Steady-State Landscape Evolution Modeling

## Overview

This repository contains supplemental code for a manuscript on steady-state landscape evolution modeling. The code calibrates coupled advection-diffusion and threshold-partitioned hillslope-channel models against 1-m LiDAR topography from the Allegheny Plateau (Pennsylvania) and Gabilan Mesa (California). Model parameters -- erodibility (*K*), diffusivity (*D*), and hillslope-channel transition area (*A_crit*) -- are estimated by minimizing the mean squared error between modeled and observed elevation fields.

Three model variants are implemented:

1. **Advection-diffusion** -- solves the coupled PDE `0 = U - K A^m dz/dx + D nabla^2 z` over the entire domain.
2. **Threshold-partitioned (linear)** -- partitions the landscape into channels (`A >= A_crit`) governed by the stream-power law and hillslopes governed by linear diffusion.
3. **Threshold-partitioned (nonlinear)** -- same partitioning with slope-dependent (nonlinear) hillslope diffusion `D_eff = D / (1 - (|nabla z| / S_c)^2)`.

All three variants are available with both **D8** and **D-infinity** (Tarboton 1997) flow routing. The D-inf implementation computes a continuous flow angle from 8 triangular facets and distributes flow to two receivers proportionally, eliminating D8's 45-degree quantization. This allows side-by-side comparison to determine whether non-convergent model behavior is intrinsic to the physics or an artifact of D8's discrete flow directions.

## Repository Structure

```
.
├── src/                      # Shared Python modules
│   ├── models.py             # D8 PDE model factories (advection-diffusion, threshold-partitioned)
│   ├── models_dinf.py        # D-infinity model factories (same 3 variants, D-inf stencil)
│   ├── dinf.py               # D-infinity flow routing (Tarboton 1997)
│   ├── optimization.py       # scipy.optimize wrappers with log-space params + LRU caching
│   ├── analysis.py           # Slope-area regression, channel steepness (ks), concavity (theta)
│   ├── synthetic.py          # Synthetic planform solvers for Peclet-number analysis
│   ├── utils.py              # Data I/O, cropping, slope computation, DEM resampling
│   └── plotting.py           # Nature-style figures, elevation/slope-area comparison plots
├── notebooks/
│   ├── model_runs/           # 10 notebooks: {allegheny,gabilan}_tile{1..5}.ipynb
│   └── figures/              # 5 notebooks: fig_slope_area, supp_fig5, supp_fig6,
│                             #   d8_vs_dinf_comparison, resolution_study_dinf
├── tests/                    # pytest suite (105 tests)
├── scripts/                  # Notebook transformation utilities
├── data/
│   ├── AP/tile{1..5}/        # Allegheny Plateau LiDAR tiles
│   └── GM/tile{1..5}/        # Gabilan Mesa LiDAR tiles
├── docs/                     # Documentation
│   ├── api.md                # API reference for all src/ modules
│   ├── workflow.md           # Scientific workflow guide
│   └── development.md        # Developer guide
└── environment.yml           # Conda environment specification
```

## Installation

1. Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate TopoAnalysis
```

2. Clone the [TopoAnalysis](https://github.com/stgl/TopoAnalysis) dependency and ensure it is importable (e.g., add it to your `PYTHONPATH` or install it in the environment).

## Quick Start

Each model-run notebook is self-contained. To run a single tile:

1. Open `notebooks/model_runs/allegheny_tile1.ipynb` in Jupyter.
2. The notebook loads the LiDAR data, builds the model, runs four optimization stages (AD, threshold A_crit, nonlinear A_crit, full K-D-A_crit), and generates comparison plots.
3. Set `SAVE = True` in the configuration cell to write model outputs to `data/AP/tile1/outputs/`.

All notebooks import shared functions from `src/` via:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path("../../src")))
```

## Running Tests

```bash
python3 -m pytest tests/ -v
```

All 105 tests run without the TopoAnalysis dependency (mocked where needed).

## Notebooks Guide

| Notebook | Location | Purpose | Key Outputs |
|----------|----------|---------|-------------|
| `allegheny_tile{1..5}.ipynb` | `notebooks/model_runs/` | Calibrate all three model variants against Allegheny Plateau tiles | Optimized K, D, A_crit; elevation/slope-area comparison plots; saved rasters |
| `gabilan_tile{1..5}.ipynb` | `notebooks/model_runs/` | Calibrate all three model variants against Gabilan Mesa tiles | Same as above for Gabilan Mesa |
| `fig_slope_area.ipynb` | `notebooks/figures/` | Generate main-text slope-area regression figures | Publication-quality slope-area plots |
| `supp_fig5.ipynb` | `notebooks/figures/` | Peclet-number sensitivity analysis on synthetic V-shaped planform grids | Supplementary Figure 5 |
| `supp_fig6.ipynb` | `notebooks/figures/` | Resolution dependence of channel steepness (ks) estimation | Supplementary Figure 6 |
| `d8_vs_dinf_comparison.ipynb` | `notebooks/figures/` | Side-by-side D8 vs D-infinity comparison for AP tile 1 and GM tile 1 | 4-panel elevation maps, area scatter, slope-area overlays, MSE table |
| `resolution_study_dinf.ipynb` | `notebooks/figures/` | Resolution study with both D8 and D-infinity flow routing | Ks ratio vs. dx curves for D8 and D-inf on same axes |

## D-infinity Flow Routing

The D-infinity implementation (Tarboton 1997) provides a parallel set of model factories for diagnostic comparison with D8. Key differences:

- **Flow direction**: D-inf evaluates 8 triangular facets to compute a continuous flow angle, distributing flow to two receivers proportionally. D8 routes all flow to a single steepest neighbor in one of 8 cardinal/diagonal directions.
- **Drainage area**: D-inf distributes area proportionally to two receivers instead of entirely to one.
- **Advection stencil**: The D-inf stencil distributes the incision term across two receivers weighted by `cos(r)` and `sin(r)`, where `r` is the within-facet angle.
- **Flat-cell handling**: For PriorityFlood-filled DEMs with flat regions, the D8 flow direction is used as a fallback to ensure no cell receives a NaN flow angle.

Usage:

```python
from dinf import compute_dinf_from_dem
from models_dinf import advection_diffusion_model_dinf

# Compute D-inf routing (passing D8 fd for flat-cell fallback)
angles, r_facet, facet_idx, A_dinf = compute_dinf_from_dem(dem, fd)

# Build D-inf AD solver (same interface as D8 version)
solver = advection_diffusion_model_dinf(
    dem, area, fd, m, angles, r_facet, facet_idx, A_dinf
)
Z_dinf = solver(U, K, D)
```

Same physical parameters (K, D, U, m) as D8 -- no re-optimization needed.

## Documentation

- [API Reference](docs/api.md) -- function signatures and descriptions for all `src/` modules
- [Scientific Workflow Guide](docs/workflow.md) -- narrative explanation of the modeling pipeline
- [Developer Guide](docs/development.md) -- architecture, testing patterns, and contribution guide
