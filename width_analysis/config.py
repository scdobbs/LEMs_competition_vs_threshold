"""All thresholds and settings for the width analysis live here.

Every number that influences a measurement is in CONFIG so the Methods text
can cite one place. Units are metres unless stated.
"""

from __future__ import annotations

CONFIG: dict = {
    # --- channel network and sampling -------------------------------------
    "a_min_sample_m2": 1.0e3,       # lowered from 10^3.5 on 2026-11-02 (min A_crit is 1.9e3)
    "n_area_bins": 8,               # log-spaced bins from a_min_sample to the site's max A
    "pilot_points_per_site": 10,
    "full_points_per_bin": 6,       # directive: 5-8 per bin per site
    "min_separation_m": 50.0,       # no two sample points closer than this (straight line)
    "confluence_exclusion_m": 15.0, # along-path distance to a confluence
    "confluence_min_fraction": 0.05, # a junction counts when the 2nd-largest donor has A >= max(a_min, frac x A_receiver)
    "edge_exclusion_m": 30.0,       # distance from tile edge
    "knickpoint_factor": 3.0,       # local slope > factor x reach median -> knickpoint
    "reach_half_length_m": 100.0,   # reach for the knickpoint median: +/- this along the path
    "vertical_interval_m": 10.0,    # repo's ChannelSlopeWithSmoothing vertical interval
    "horizontal_interval_fallback_m": 50.0,  # used (and flagged) where the 10 m drop cannot be reached inside the tile
    "flat_fill_slope_max": 0.002,   # along-trace slope below this = filled depression proxy -> excluded
    "confinement_half_window_m": 5.0,   # profile must rise >= confinement_min_relief_m on both sides within this
    "confinement_min_relief_m": 0.1,
    "steep_slope_flag": 0.5,        # local slope above this is flagged steep_rill_flag (AP tile 5 rills)
    "random_seed": 20261102,
    # --- transect geometry ------------------------------------------------
    "trace_half_length_m": 10.0,    # channel trace used to fit the local flow line
    "transect_half_length_m": 25.0,
    "transect_step_m": 0.25,
    "thalweg_search_m": 3.0,        # thalweg = min z within +/- this of the centreline
    # --- fixed-stage (height-above-thalweg) widths: diagnostic only -------
    "fixed_stages_m": [0.25, 0.5, 1.0],
    "crest_drop_m": 0.3,            # a stage span stops at a crest if the profile then drops by more than this
    # --- inner-channel (bank-break) width ---------------------------------
    "bankbreak_smooth_sigma_m": 0.5,    # Gaussian smoothing of z(s) before curvature
    "bankbreak_min_convexity_1_per_m": 0.05,  # -d2z/ds2 must exceed this at the break
    "bankbreak_min_bank_slope": 0.10,   # mean slope thalweg -> break must exceed this
    "channelized_min_relief_m": 0.3,    # channelized = both breaks present and inner relief >= this
    # --- valley-floor width -----------------------------------------------
    "valley_slope_threshold": 0.05,
    "valley_smooth_sigma_m": 0.0,       # 0 = no smoothing: bilinear samples of a 1 m grid are already piecewise linear
    # --- resolution limit / censoring -------------------------------------
    "censor_multiple": 2.0,             # widths below censor_multiple * pixel size are bounds
    # --- valley-bottom shape class ----------------------------------------
    "flat_bed_tol_m": 0.1,
    "flat_bed_min_width_m": 2.0,
    # --- fits -------------------------------------------------------------
    "bootstrap_n": 1000,
    "ci_percentiles": [2.5, 97.5],
    # --- outputs ----------------------------------------------------------
    "validation_min_area_m2": 1.0e5,    # GeoPackage validation layer is weighted to channelized points above this
    "validation_extra_per_site": 10,    # plus this many random other points per site
    "qc_gallery_per_site": 20,
    # --- QC figures -------------------------------------------------------
    "inset_half_size_m": 60.0,
    "hillshade_azimuth_deg": 315.0,
    "hillshade_altitude_deg": 45.0,
}

# Optimized A_crit (m^2) of the jointly optimized linear partitioned model,
# Table 1 of the manuscript; identical to the final stage printed in
# notebooks/model_runs/*_complete.ipynb.
A_CRIT_M2: dict = {
    ("AP", 1): 6.4143e3, ("AP", 2): 7.6456e3, ("AP", 3): 9.7415e3,
    ("AP", 4): 7.7979e3, ("AP", 5): 1.8870e3,
    ("GM", 1): 8.0924e3, ("GM", 2): 2.9600e3, ("GM", 3): 6.5263e3,
    ("GM", 4): 7.3962e3, ("GM", 5): 2.2471e3,
}

# Jointly optimized (K, D) and the tile constants of the partitioned model,
# from the final stage of notebooks/model_runs/*_complete.ipynb (used by the
# A_crit misfit profile).
PARTITIONED_PARAMS: dict = {
    ("AP", 5): dict(U=5.5e-5, m=0.400, K=3.4402e-06, D=3.6291e-03, A_crit=1.8870e3, MSE=2.8962e2),
}

# Grid spacings of the existing 5-100 m resolution sweep
# (notebooks/figures/supp_fig6.ipynb and resolution_study_dinf.ipynb).
SWEEP_N_DX = 25
SWEEP_DX_MIN_M = 5.0
SWEEP_DX_MAX_M = 100.0

SITES = {"AP": "Allegheny Plateau", "GM": "Gabilan Mesa"}
TILES = [1, 2, 3, 4, 5]
