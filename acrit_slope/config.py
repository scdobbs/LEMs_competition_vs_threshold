"""All thresholds and choices of the A_crit-slope test live here (one place to
cite in Methods). Units are metres unless stated.
"""

from __future__ import annotations

from width_analysis.config import A_CRIT_M2, CONFIG as WIDTH_CONFIG, SITES, TILES  # noqa: F401

CONFIG: dict = {
    # --- head exclusion ---------------------------------------------------
    "edge_buffer_m": 30.0,              # heads closer than this to the tile edge are excluded
    # the upslope path must reach max(L_values_m) inside the tile, else path_leaves_tile
    "flat_fill_slope_max": WIDTH_CONFIG["flat_fill_slope_max"],   # 0.002: the width config's filled-flat proxy
    "flat_fill_half_length_m": 10.0,    # proxy = fall over the +/- 10 m main-stem trace (as in width_analysis)
    # --- slope measurement -------------------------------------------------
    "L_values_m": [5.0, 10.0, 20.0],    # path lengths for S_up_L and S_centred_L
    "L_default_m": 10.0,                # the slope used in every fit
    # --- plan-curvature QC -------------------------------------------------
    "curvature_block_m": 10.0,          # block-mean subsampling before the 9-point stencil
    # --- fits ---------------------------------------------------------------
    "bootstrap_n": 2000,
    "random_seed": 20261104,
    "ci_percentiles": [2.5, 97.5],
    "reference_exponent": -2.0,         # Montgomery & Dietrich: A S^2 = const
    "misfit_tolerance": 0.10,           # AP tile 5 A_crit range: misfit within 10 % of the minimum
    # --- supplementary (beyond the directive; see README) --------------------
    "pseudo_head_thresholds_m2": [1.0e3, 2.0e3, 5.0e3, 1.0e4, 2.0e4],  # S(A) structure near A_crit
    "reference_area_m2": 5.0e3,         # slope at a fixed area, to break the common-S(A) circularity
    # --- QC figures -----------------------------------------------------------
    "qc_profiles_per_tile": 10,
    "hillshade_azimuth_deg": 315.0,
    "hillshade_altitude_deg": 45.0,
    "output_dir": "outputs/acrit_slope",
}

# Per-tile concavity theta used as m in every optimization stage of
# notebooks/model_runs/*_complete.ipynb (the ``concavity = ...`` line).
THETA: dict = {
    ("AP", 1): 0.562, ("AP", 2): 0.592, ("AP", 3): 0.542, ("AP", 4): 0.438, ("AP", 5): 0.400,
    ("GM", 1): 0.379, ("GM", 2): 0.308, ("GM", 3): 0.417, ("GM", 4): 0.313, ("GM", 5): 0.300,
}

# Stage-2 A_crit of the same notebooks: A_crit optimized alone with K and D
# held at the per-tile measured values (the stage before the joint K-D-A_crit
# optimization that produced A_CRIT_M2 / Table 1). Supplementary only.
A_CRIT_STAGE2_M2: dict = {
    ("AP", 1): 3.0634e3, ("AP", 2): 2.3378e3, ("AP", 3): 1.7221e3, ("AP", 4): 9.9501e2, ("AP", 5): 1.1300e3,
    ("GM", 1): 4.2349e2, ("GM", 2): 1.0000e2, ("GM", 3): 1.2974e3, ("GM", 4): 4.2209e2, ("GM", 5): 2.6142e2,
}

# A_crit misfit profile (only AP tile 5 has one)
ACRIT_PROFILE_CSV = {("AP", 5): "outputs/widths/acrit_profile/AP5.csv"}

TILE_KEYS = [(s, t) for s in SITES for t in TILES]
