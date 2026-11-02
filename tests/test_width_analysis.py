"""Synthetic cross-section tests for width_analysis (no raster data needed)."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from width_analysis.config import CONFIG  # noqa: E402
from width_analysis.widths import measure_cross_section, width_at_stage, find_thalweg, confinement  # noqa: E402
from width_analysis.transect import flow_line_direction  # noqa: E402
from width_analysis.network import receiver_grids, FlowGraph  # noqa: E402
from width_analysis.tiles import Tile  # noqa: E402

STEP = CONFIG["transect_step_m"]
S = np.arange(-25.0, 25.0 + 0.5 * STEP, STEP)
PIX = 1.0


def v_shape(side_slope=0.5):
    return side_slope * np.abs(S)


def flat_bed(bed_width=4.0, bank_height=1.0, bank_slope=1.0, hill_slope=0.1):
    a = np.abs(S) - bed_width / 2
    z = np.where(a <= 0, 0.0, np.minimum(a * bank_slope, bank_height))
    beyond = a - bank_height / bank_slope
    z = z + np.where(beyond > 0, beyond * hill_slope, 0.0)
    return z


def trapezoid(bed_width=3.0, bank_slope=0.5, bank_height=1.0, hill_slope=0.02):
    return flat_bed(bed_width, bank_height, bank_slope, hill_slope)


def narrow(bed_width=1.0, side_slope=1.0):
    a = np.abs(S) - bed_width / 2
    return np.where(a <= 0, 0.0, a * side_slope)


def test_v_shape_widths_scale_with_stage():
    z = v_shape(0.5)
    m = measure_cross_section(S, z, PIX, CONFIG)
    assert m["thalweg_index"] == np.argmin(np.abs(S))
    assert m["W_hand_0.5"] == pytest.approx(2.0, abs=STEP)
    assert m["W_hand_1.0"] == pytest.approx(4.0, abs=STEP)
    assert m["shape_class"] == "v_shape"
    assert "W_over_h_flat" in m["qc_flags"]           # W/h constant: pick is arbitrary and flagged
    assert np.isnan(m["W_bankbreak"])                  # no convex break anywhere
    assert m["W_hand_0.25_censored"] is True           # 1.0 m < 2 m bound


def test_flat_bed_4m():
    z = flat_bed(4.0, bank_height=1.0, bank_slope=1.0, hill_slope=0.1)
    m = measure_cross_section(S, z, PIX, CONFIG)
    assert m["shape_class"] == "flat_bed"
    assert m["flat_span_m"] == pytest.approx(4.0, abs=2 * STEP)
    assert m["W_hand_0.25"] == pytest.approx(4.5, abs=STEP)
    assert m["W_hand_0.5"] == pytest.approx(5.0, abs=STEP)
    assert m["h_bankfull"] == pytest.approx(1.0, abs=CONFIG["h_step_m"])
    assert m["W_hand_bankfull"] == pytest.approx(6.0, abs=3 * STEP)
    assert m["W_bankbreak"] == pytest.approx(6.0, abs=1.0)
    assert m["W_valley"] == pytest.approx(4.0, abs=1.0)
    assert m["censored"] is False
    assert m["censor_bound"] == pytest.approx(2.0)


def test_trapezoid():
    z = trapezoid(3.0, bank_slope=0.5, bank_height=1.0, hill_slope=0.02)
    m = measure_cross_section(S, z, PIX, CONFIG)
    assert m["W_hand_0.5"] == pytest.approx(5.0, abs=STEP)
    assert m["W_hand_1.0"] == pytest.approx(7.0, abs=2 * STEP)
    assert m["W_bankbreak"] == pytest.approx(7.0, abs=1.0)
    assert m["W_valley"] == pytest.approx(3.0, abs=1.0)
    assert m["shape_class"] == "flat_bed"
    assert m["censored"] is False


def test_narrow_channel_is_censored():
    z = narrow(1.0, side_slope=1.0)
    m = measure_cross_section(S, z, PIX, CONFIG)
    assert m["W_hand_0.25"] == pytest.approx(1.5, abs=STEP)
    assert m["W_hand_0.25_censored"] is True
    assert m["censor_bound"] == pytest.approx(2.0)
    assert m["shape_class"] == "v_shape"           # 1 m flat span < 2 m
    assert np.isfinite(m["W_hand_0.25"])           # value is kept, not dropped


def test_width_at_stage_interpolates_crossing():
    z = v_shape(0.5)
    w, sl, sr, hit, cut = width_at_stage(S, z, find_thalweg(S, z, 3.0), 0.37)
    assert w == pytest.approx(2 * 0.37 / 0.5, abs=1e-9)
    assert not hit and not cut


def test_span_stops_at_crest_before_neighbouring_channel():
    # two parallel V channels 12 m apart with a 0.6 m crest between them
    z = np.minimum(0.1 * np.abs(S), 0.6)                      # gentle main channel, crest at |s| >= 6
    z = np.where(S < -6, 0.6 - 0.5 * (-6 - S), z)              # second channel descending left of the crest
    i_t = find_thalweg(S, z, 3.0)
    w, sl, sr, hit, cut = width_at_stage(S, z, i_t, 1.0, crest_drop=0.3)
    assert cut
    assert sl == pytest.approx(-6.0, abs=STEP)                 # left end sits on the crest, not in the next hollow
    assert sr == pytest.approx(25.0, abs=STEP) or hit          # right side never exceeds 1.0 m -> transect end
    w2, sl2, *_ = width_at_stage(S, z, i_t, 1.0)               # without the crest rule the span runs on
    assert sl2 < -6.0


def test_confinement_rejects_planar_slope_and_accepts_channel():
    planar = 0.5 * S + 3.0                                     # monotonic: a D8 line across a hillslope
    i_t = find_thalweg(S, planar, 3.0)
    ok, rl, rr = confinement(S, planar, i_t, 5.0, 0.1)
    assert not ok and rl <= 0.0
    ok2, rl2, rr2 = confinement(S, v_shape(0.5), find_thalweg(S, v_shape(0.5), 3.0), 5.0, 0.1)
    assert ok2 and rl2 == pytest.approx(2.5) and rr2 == pytest.approx(2.5)
    m = measure_cross_section(S, planar, PIX, CONFIG)
    assert m["confined"] is False and "not_confined" in m["qc_flags"]


def test_censor_bound_scales_with_pixel_size():
    z = narrow(1.0)
    m = measure_cross_section(S, z, 0.9752, CONFIG)
    assert m["censor_bound"] == pytest.approx(2 * 0.9752)


def test_flow_line_direction_is_continuous():
    # a trace at 30 degrees from the x axis, quantized to a D8-like staircase
    t = np.linspace(0, 10, 11)
    xy = np.column_stack([t * np.cos(np.radians(30)), t * np.sin(np.radians(30))])
    xy = np.round(xy)       # staircase
    d = flow_line_direction(xy)
    ang = np.degrees(np.arctan2(d[1], d[0]))
    assert ang == pytest.approx(30.0, abs=8.0)
    assert d[0] > 0         # oriented from first to last point


def _synthetic_tile(n=60):
    # straight channel along the central row flowing east; V-shaped valley 0.3 side slope
    ii, jj = np.indices((n, n))
    z = 0.3 * np.abs(ii - n // 2) * 1.0 + 100.0 - 0.01 * jj
    fd = np.full((n, n), 1, dtype=np.uint8)            # everything flows east
    fd[: n // 2, :] = 4                                 # north half flows south
    fd[n // 2 + 1 :, :] = 64                            # south half flows north
    fd[n // 2, :] = 1
    area = np.ones((n, n))
    area[n // 2, :] = (jj[n // 2, :] + 1) * n          # channel accumulates ~n per column
    return Tile("XX", 1, z, area, fd, 1.0, 1.0, 0.0, float(n), "")


def test_flow_graph_on_synthetic_tile():
    t = _synthetic_tile()
    g = FlowGraph(t, a_min=100.0)
    ri, rj = receiver_grids(t.fd)
    assert ri[30, 10] == 30 and rj[30, 10] == 11
    assert ri[30, 59] == -1                              # leaves the grid
    assert g.up(30, 10) == (30, 9)                       # main-stem donor is upstream along the channel
    # local slope along the channel: 0.01 per pixel; a 10 m drop needs 1000 pixels -> undefined on a 60 px tile
    assert np.isnan(g.local_slope(30, 30, 10.0))
    assert g.local_slope(30, 30, 0.1) == pytest.approx(0.01, rel=1e-6)
    assert g.edge_distance(30, 10) == 10.0
    assert not g.confluence.any()
    assert g.local_slope_horizontal(30, 30, 10.0) == pytest.approx(0.01, rel=1e-6)
    s, method = g.slope_with_fallback(30, 30, 10.0, 10.0)
    assert method == "horizontal_fallback" and s == pytest.approx(0.01, rel=1e-6)


def test_confluence_requires_significant_tributary():
    t = _synthetic_tile()
    n = t.shape[0]
    # tributary from the north joining the channel at column 40: give column 40 of the north half area >= 100
    t.area[: n // 2, 40] = np.arange(1, n // 2 + 1) * 10.0     # 10 ... 300 m^2 arriving from the north
    t.area[n // 2, 40:] += 300.0
    g_abs = FlowGraph(t, a_min=100.0, confluence_min_fraction=0.0)
    assert g_abs.confluence[n // 2, 40]                        # 300 >= a_min: counts under the absolute rule
    g_rel = FlowGraph(t, a_min=100.0, confluence_min_fraction=0.5)
    assert not g_rel.confluence[n // 2, 40]                    # 300 < 0.5 x (41*60+300): not a jump in A
    assert g_rel.distance_to_confluence(n // 2, 35, 15.0) == np.inf
    assert g_abs.distance_to_confluence(n // 2, 35, 15.0) == pytest.approx(5.0)
