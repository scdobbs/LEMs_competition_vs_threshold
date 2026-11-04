"""Every threshold and choice of the head-extraction test. Units are metres
unless stated. Nothing in ``EXTRACTION`` refers to drainage area: head
location must not depend on area (directive section 3), and the test
``test_heads_independent_of_area_config`` checks that changing any area value
in ``QC`` or ``ACRIT`` leaves the heads bit-identical.
"""

from __future__ import annotations

from acrit_slope.config import A_CRIT_M2, A_CRIT_STAGE2_M2, THETA  # noqa: F401
from width_analysis.config import SITES, TILES  # noqa: F401

# --- Method P: Pelletier (2013) contour curvature (no area anywhere) ---------
EXTRACTION: dict = {
    "wiener_pad_px": 128,              # reflect padding before the FFT (edge ringing)
    "wiener_noise_band": [0.75, 1.0],  # |N|^2 = mean power over this fraction of the Nyquist frequency
    "wiener_signal_min_ratio": 3.0,    # |H|^2 power-law fit uses bins where S > ratio x |N|^2
    "wiener_skip_lowest_bins": 2,      # lowest radial bins excluded from the power-law fit (detrending)
    "wiener_n_radial_bins": 200,
    "curvature_threshold_default": 0.10,   # m^-1 (Pelletier 2013; Clubb et al. 2014)
    "curvature_thresholds": [0.05, 0.10, 0.20],
    "fill_epsilon": True,              # hydrologic conditioning of the filtered DEM before flagging
    "edge_buffer_m": 30.0,
    "min_upslope_path_m": 20.0,        # heads whose largest-donor path leaves the tile within this are dropped
    "L_values_m": [5.0, 10.0, 20.0],
    "L_default_m": 10.0,
    "curvature_block_m": 10.0,         # plan-curvature QC grid (as Test 1)
    "match_radius_m": 20.0,            # Method P vs Method D nearest-neighbour matching
}

# --- Method D: DrEICH via LSDTopoTools (lsdtt-channel-extraction) ------------
LSDTT: dict = {
    "bin_dir": "/home/groups/hilley/scdobbs-envs/lsdtt/bin",
    "connected_components_threshold": 10,   # Clubb et al. 2014: 10 linked pixels for valley selection
    "curvature_threshold": 0.10,            # valley identification (Clubb et al. 2014 used 0.1 m^-1)
    "surface_fitting_radius": 6.0,          # LSDTT default polyfit window (m)
    "A_0": 1.0,
    "number_of_junctions_dreich": 1,
    "threshold_contributing_pixels": 1000,  # LSDTT default; builds the initial test network (an area parameter: see README)
    "threshold_contributing_pixels_sensitivity": [250, 4000],
    "pruning_drainage_area": 1000.0,        # LSDTT default (m^2)
    "print_wiener_channels": True,          # LSDTT's own Wiener/Pelletier heads, for comparison only
    "min_slope_for_fill": 0.0001,
    "time_cap_hours": 4.0,
}

# --- QC classes (directive section 4) ----------------------------------------
QC: dict = {
    "trunk_area_m2": 1.0e5,            # trunk pixels for the valley_floor class
    "trunk_buffer_m": 30.0,
    "floor_slope_max": 0.10,
    "plateau_slope_max": 0.05,         # S_up_20 below this, or S_up_10 below this with a convex drop below
    "plateau_drop_slope_min": 0.15,    # "then convex below": slope over the 10 m downslope path above this
    "rill_slope_min": 0.58,            # tan 30 deg
    "rill_search_m": 30.0,
    "rill_min_neighbours": 3,
    "rill_azimuth_tol_deg": 20.0,
    "flow_azimuth_length_m": 10.0,     # downslope path length used for the head's flow azimuth
    "road_search_m": 60.0,
    "road_min_heads": 4,               # at least this many OTHER heads within road_search_m
    "road_max_residual_m": 3.0,        # RMS perpendicular residual of the line fit
    "road_min_angle_to_fall_line_deg": 30.0,
    "road_buffer_m": 10.0,             # if a roads layer existed (none found; see README)
    "qc_profiles_per_tile": 15,
    "hillshade_azimuth_deg": 315.0,
    "hillshade_altitude_deg": 45.0,
    "area_overlay_min_m2": 1.0e3,
}

ANALYSIS: dict = {
    "bootstrap_n": 2000,
    "random_seed": 20261105,
    "ci_percentiles": [2.5, 97.5],
    "reference_exponents": [-2.0, -1.65, 0.0],
    "md_envelope_AS2_m2": [500.0, 4000.0],   # Montgomery & Dietrich (1992) equations 3-4
    "kargere_r2_m2": 62.0**2,                # Kargere et al. (2025): r = 62 m at Gabilan
    "spatial_match_m": 20.0,
    "clubb_ohio_mean_head_area_m2": {"Mid Bailey Run": 413.0, "Indian Creek": 1035.0},
}

ACRIT_BRACKET: dict = {k: (A_CRIT_STAGE2_M2[k], A_CRIT_M2[k]) for k in A_CRIT_M2}

OUTPUT_DIR = "outputs/head_extraction"
TILE_KEYS = [(s, t) for s in SITES for t in TILES]
