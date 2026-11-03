"""Synthetic cross-section and fitting tests for width_analysis (no raster data needed)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from width_analysis.config import CONFIG  # noqa: E402
from width_analysis.widths import measure_cross_section, width_at_stage, find_thalweg, confinement  # noqa: E402
from width_analysis.transect import flow_line_direction  # noqa: E402
from width_analysis.network import receiver_grids, FlowGraph  # noqa: E402
from width_analysis.tiles import Tile  # noqa: E402
from width_analysis.fits import tobit_fit, bootstrap_tobit, ols_uncensored, logistic_fit, fit_site_widths  # noqa: E402
from width_analysis.regime import regime_fraction, width_of  # noqa: E402

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


# ---------------------------------------------------------------------------
# cross-sections
# ---------------------------------------------------------------------------

def test_v_shape_stage_widths_measure_side_slope_not_channel():
    z = v_shape(0.5)
    m = measure_cross_section(S, z, PIX, CONFIG)
    assert m["thalweg_index"] == np.argmin(np.abs(S))
    assert m["W_stage_0.5"] == pytest.approx(2.0, abs=STEP)      # 2h / side slope
    assert m["W_stage_1.0"] == pytest.approx(4.0, abs=STEP)
    assert m["shape_class"] == "v_shape"
    assert np.isnan(m["W_inner_bankbreak"]) and not m["channelized"]   # no convex break anywhere
    assert m["W_primary_source"] == "unresolved" and m["censored"] is True
    assert m["resolved_width"] is False
    assert m["W_stage_0.25_censored"] is True                     # 1.0 m < 2 m bound


def test_flat_bed_4m():
    z = flat_bed(4.0, bank_height=1.0, bank_slope=1.0, hill_slope=0.1)
    m = measure_cross_section(S, z, PIX, CONFIG)
    assert m["shape_class"] == "flat_bed"
    assert m["flat_span_m"] == pytest.approx(4.0, abs=2 * STEP)
    assert m["W_stage_0.25"] == pytest.approx(4.5, abs=STEP)
    assert m["W_stage_0.5"] == pytest.approx(5.0, abs=STEP)
    assert m["W_inner_bankbreak"] == pytest.approx(6.0, abs=1.0)
    assert m["inner_relief_m"] == pytest.approx(1.0, abs=0.15)
    assert m["channelized"] is True
    assert m["W_valley_floor"] == pytest.approx(4.0, abs=1.0)
    assert m["W_primary_source"] == "bankbreak" and m["W_primary"] == m["W_inner_bankbreak"]
    assert m["censored"] is False and m["resolved_width"] is True
    assert m["censor_bound"] == pytest.approx(2.0)


def test_trapezoid():
    z = trapezoid(3.0, bank_slope=0.5, bank_height=1.0, hill_slope=0.02)
    m = measure_cross_section(S, z, PIX, CONFIG)
    assert m["W_stage_0.5"] == pytest.approx(5.0, abs=STEP)
    assert m["W_stage_1.0"] == pytest.approx(7.0, abs=2 * STEP)
    assert m["W_inner_bankbreak"] == pytest.approx(7.0, abs=1.0)
    assert m["W_valley_floor"] == pytest.approx(3.0, abs=1.0)
    assert m["shape_class"] == "flat_bed" and m["channelized"] is True
    assert m["censored"] is False


def test_narrow_channel_is_censored():
    z = narrow(1.0, side_slope=1.0)
    m = measure_cross_section(S, z, PIX, CONFIG)
    assert m["W_stage_0.25"] == pytest.approx(1.5, abs=STEP)
    assert m["W_stage_0.25_censored"] is True
    assert m["censor_bound"] == pytest.approx(2.0)
    assert m["shape_class"] == "v_shape"           # 1 m flat span < 2 m
    assert m["censored"] is True                   # no banks -> unresolved -> bound 2 m
    assert np.isfinite(m["W_stage_0.25"])          # the stage value is kept, not dropped


def test_shallow_notch_is_not_channelized():
    # 1.5 m bed with 0.25 m banks on a flat floor: both breaks exist but relief < 0.3 m
    z = flat_bed(1.5, bank_height=0.25, bank_slope=1.0, hill_slope=0.0)
    m = measure_cross_section(S, z, PIX, CONFIG)
    assert np.isfinite(m["W_inner_bankbreak"])
    assert m["inner_relief_m"] < CONFIG["channelized_min_relief_m"]
    assert m["channelized"] is False
    assert "inner_relief_below" in m["qc_flags"]


def test_flat_bed_without_banks_uses_valley_floor():
    # a 6 m flat floor rising gently (0.03) to both sides: no convex break, flat_bed -> valley-floor primary
    a = np.abs(S) - 3.0
    z = np.where(a <= 0, 0.0, a * 0.03)
    m = measure_cross_section(S, z, PIX, CONFIG)
    assert m["shape_class"] == "flat_bed"
    assert np.isnan(m["W_inner_bankbreak"])
    assert m["W_primary_source"] == "valley_floor"
    assert m["W_primary"] == pytest.approx(m["W_valley_floor"])
    assert m["resolved_width"] is True and m["censored"] is False


def test_width_at_stage_interpolates_crossing():
    z = v_shape(0.5)
    w, sl, sr, hit, cut = width_at_stage(S, z, find_thalweg(S, z, 3.0), 0.37)
    assert w == pytest.approx(2 * 0.37 / 0.5, abs=1e-9)
    assert not hit and not cut


def test_span_stops_at_crest_before_neighbouring_channel():
    z = np.minimum(0.1 * np.abs(S), 0.6)
    z = np.where(S < -6, 0.6 - 0.5 * (-6 - S), z)
    i_t = find_thalweg(S, z, 3.0)
    w, sl, sr, hit, cut = width_at_stage(S, z, i_t, 1.0, crest_drop=0.3)
    assert cut
    assert sl == pytest.approx(-6.0, abs=STEP)
    w2, sl2, *_ = width_at_stage(S, z, i_t, 1.0)
    assert sl2 < -6.0


def test_confinement_rejects_planar_slope_and_accepts_channel():
    planar = 0.5 * S + 3.0
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


# ---------------------------------------------------------------------------
# transect direction and flow graph
# ---------------------------------------------------------------------------

def test_flow_line_direction_is_continuous():
    t = np.linspace(0, 10, 11)
    xy = np.column_stack([t * np.cos(np.radians(30)), t * np.sin(np.radians(30))])
    xy = np.round(xy)
    d = flow_line_direction(xy)
    ang = np.degrees(np.arctan2(d[1], d[0]))
    assert ang == pytest.approx(30.0, abs=8.0)
    assert d[0] > 0


def _synthetic_tile(n=60):
    ii, jj = np.indices((n, n))
    z = 0.3 * np.abs(ii - n // 2) * 1.0 + 100.0 - 0.01 * jj
    fd = np.full((n, n), 1, dtype=np.uint8)
    fd[: n // 2, :] = 4
    fd[n // 2 + 1 :, :] = 64
    fd[n // 2, :] = 1
    area = np.ones((n, n))
    area[n // 2, :] = (jj[n // 2, :] + 1) * n
    return Tile("XX", 1, z, area, fd, 1.0, 1.0, 0.0, float(n), "")


def test_flow_graph_on_synthetic_tile():
    t = _synthetic_tile()
    g = FlowGraph(t, a_min=100.0)
    ri, rj = receiver_grids(t.fd)
    assert ri[30, 10] == 30 and rj[30, 10] == 11
    assert ri[30, 59] == -1
    assert g.up(30, 10) == (30, 9)
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
    t.area[: n // 2, 40] = np.arange(1, n // 2 + 1) * 10.0
    t.area[n // 2, 40:] += 300.0
    g_abs = FlowGraph(t, a_min=100.0, confluence_min_fraction=0.0)
    assert g_abs.confluence[n // 2, 40]
    g_rel = FlowGraph(t, a_min=100.0, confluence_min_fraction=0.5)
    assert not g_rel.confluence[n // 2, 40]
    assert g_rel.distance_to_confluence(n // 2, 35, 15.0) == np.inf
    assert g_abs.distance_to_confluence(n // 2, 35, 15.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# fits
# ---------------------------------------------------------------------------

def _synthetic_widths(n=120, c=0.02, b=0.45, sigma=0.08, bound=2.0, seed=1):
    rng = np.random.default_rng(seed)
    logA = rng.uniform(3, 7, n)
    logW = np.log10(c) + b * logA + rng.normal(0, sigma, n)
    cens = logW < np.log10(bound)
    return logA, logW, cens, np.full(n, np.log10(bound))


def test_tobit_recovers_parameters_where_ols_is_biased():
    logA, logW, cens, lb = _synthetic_widths()
    assert 0.15 < cens.mean() < 0.6                # a meaningful censored fraction
    f = tobit_fit(logA, logW, cens, lb)
    assert f["converged"]
    assert f["b"] == pytest.approx(0.45, abs=0.04)
    assert np.log10(f["c"]) == pytest.approx(np.log10(0.02), abs=0.25)
    o = ols_uncensored(logA, logW, cens)
    assert o["b"] < f["b"]                          # dropping the censored (small) widths flattens the slope
    boot = bootstrap_tobit(logA, logW, cens, lb, 50, np.random.default_rng(2))
    assert boot.shape[1] == 2 and len(boot) >= 45
    lo, hi = np.percentile(boot[:, 1], [2.5, 97.5])
    assert lo < 0.45 < hi


def test_logistic_a50():
    rng = np.random.default_rng(3)
    logA = rng.uniform(3, 7, 300)
    p = 1 / (1 + np.exp(-3.0 * (logA - 5.0)))      # A50 = 1e5
    y = rng.uniform(size=300) < p
    f = logistic_fit(logA, y.astype(float))
    assert f["logA50"] == pytest.approx(5.0, abs=0.15)
    assert np.isnan(logistic_fit(logA, np.ones(300))["A50"])


def test_fit_site_widths_uses_channelized_points_only():
    logA, logW, cens, lb = _synthetic_widths(n=60)
    df = pd.DataFrame(dict(area_m2=10 ** logA, W_inner_bankbreak=np.where(cens, 1.0, 10 ** logW),
                           W_inner_bankbreak_censored=cens, censor_bound=2.0, channelized=True))
    df.loc[df.index[:10], "channelized"] = False
    cfg = dict(CONFIG, bootstrap_n=20)
    f = fit_site_widths(df, cfg, np.random.default_rng(0), "all")
    assert f["n_channelized"] == 50
    assert np.isfinite(f["tobit"]["b"]) and len(f["c_ci"]) == 2


def test_regime_fraction_monotone_in_dx():
    chan = [dict(area=np.logspace(3, 7, 500), length=np.ones(500))]
    dxs = np.array([1.0, 5.0, 20.0, 100.0])
    fr = regime_fraction(chan, np.log10(0.02), 0.45, dxs)
    assert np.all(np.diff(fr) <= 0) and 0 < fr[0] <= 1
    assert width_of(np.array([1e4]), np.log10(0.02), 0.5)[0] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# constrained pick (reports/directives/channel_width_constrained_directive.md)
# ---------------------------------------------------------------------------

from width_analysis.config import CONSTRAINED  # noqa: E402
from width_analysis.constrained import measure_transect, aggregate, classify, shown_to_exceed, clopper_pearson  # noqa: E402


def channel(bed=3.0, bank_slope=2.0, bank_height=1.0, hill_slope=0.05):
    """Trapezoid: bed, banks of given slope and height, then a gentle floodplain. Top width = bed + 2 h / slope."""
    a = np.abs(S) - bed / 2
    z = np.where(a <= 0, 0.0, np.minimum(a * bank_slope, bank_height))
    beyond = a - bank_height / bank_slope
    return z + np.where(beyond > 0, beyond * hill_slope, 0.0)


def _point(z, **over):
    ccfg = dict(CONSTRAINED, **over)
    m = measure_transect(S, z, PIX, CONFIG, ccfg)
    return m, aggregate([m], PIX, ccfg)


def test_constrained_trapezoid_4m_is_inner_channel():
    z = channel(bed=3.0, bank_slope=2.0, bank_height=1.0)          # bank tops at +/- 2 m
    m, agg = _point(z)
    assert agg["point_class"] == "inner_channel"
    assert agg["W_bank"] == pytest.approx(4.0, abs=0.6)
    assert 0.3 <= agg["relief_left_m"] <= 1.5
    assert agg["ratio_bank_over_stage"] == pytest.approx(1.0, abs=0.25)
    assert "asymmetric_banks" not in agg["qc_flags"]


def test_constrained_channel_inset_in_30m_hollow_picks_the_channel():
    hollow = 3.0 * (S / 15.0) ** 2                                   # 30 m wide parabolic hollow, 3 m deep at its lip
    z = channel(bed=3.0, bank_slope=2.0, bank_height=1.0, hill_slope=0.0) + hollow
    m, agg = _point(z)
    assert agg["point_class"] == "inner_channel"
    assert agg["W_bank"] == pytest.approx(4.0, abs=0.8)              # not 30 m
    # the unconstrained Phase 3 pick on the same section lands at the inner banks here too, but with
    # a window of 20 m it would accept the hollow lip: check the constraint actually bites
    m_wide, agg_wide = _point(z, bank_max_offset_m=20.0, bank_max_relief_m=5.0)
    assert agg_wide["W_bank"] == pytest.approx(4.0, abs=0.8)         # nearest-first search still prefers the inner bank


def test_constrained_14m_channel_is_wide_or_terrace():
    z = channel(bed=13.0, bank_slope=2.0, bank_height=1.0)         # bank tops at +/- 7 m > 6 m offset limit
    m, agg = _point(z)
    assert m["left_class"] == "none_within_window" and m["right_class"] == "none_within_window"
    assert agg["point_class"] == "wide_or_terrace"
    assert agg["fit_censor"] == "right" and agg["W_fit"] == pytest.approx(12.0)
    assert shown_to_exceed(agg, 10.0) and shown_to_exceed(agg, 5.0) and not shown_to_exceed(agg, 15.0)
    m8, agg8 = _point(z, bank_max_offset_m=8.0)                      # a wider window resolves it
    assert agg8["point_class"] == "inner_channel" and agg8["W_bank"] == pytest.approx(14.0, abs=0.8)


def test_constrained_bench_on_one_side_is_asymmetric():
    # left bank 1 m high at -2 m; right bank only 0.4 m high at +2 m then a flat bench to +12 m, then the valley wall
    left = np.where(S < 0, np.minimum((-S - 1.5).clip(0) * 2.0, 1.0) + (-S - 2.0).clip(0) * 0.05, 0.0)
    right = np.where(S > 0, np.minimum((S - 1.5).clip(0) * 2.0, 0.4) + (S - 12.0).clip(0) * 0.6, 0.0)
    z = left + right
    m, agg = _point(z)
    assert agg["point_class"] == "inner_channel"
    assert "asymmetric_banks" in agg["qc_flags"]
    assert agg["relief_left_m"] / agg["relief_right_m"] > 2.0


def test_constrained_plain_v_is_unresolved_narrow():
    m, agg = _point(v_shape(0.5))
    assert m["left_class"] == "no_break" and m["right_class"] == "no_break"
    assert agg["point_class"] == "unresolved_narrow"
    assert agg["fit_censor"] == "left" and agg["W_fit"] == pytest.approx(2.0)
    assert not shown_to_exceed(agg, 1.0)


def test_constrained_classification_rules_and_intervals():
    assert classify("bank", "bank", "v_shape") == "inner_channel"
    assert classify("bank", "none_within_window", "v_shape") == "wide_or_terrace"
    assert classify("none_within_window", "none_within_window", "flat_bed") == "wide_or_terrace"
    assert classify("no_break", "no_break", "v_shape") == "unresolved_narrow"
    assert classify("no_break", "no_break", "flat_bed") == "ambiguous"
    assert classify("bank", "no_break", "v_shape") == "ambiguous"
    lo, hi = clopper_pearson(5, 10)
    assert lo == pytest.approx(0.187, abs=0.01) and hi == pytest.approx(0.813, abs=0.01)
    assert clopper_pearson(0, 10)[0] == 0.0 and clopper_pearson(10, 10)[1] == 1.0


def test_constrained_aggregate_majority_and_consistency():
    z_ch = channel(bed=3.0, bank_slope=2.0, bank_height=1.0)
    ccfg = dict(CONSTRAINED)
    ms = [measure_transect(S, z_ch, PIX, CONFIG, ccfg)] * 3 + [measure_transect(S, v_shape(0.5), PIX, CONFIG, ccfg)] * 2
    agg = aggregate(ms, PIX, ccfg)
    assert agg["point_class"] == "inner_channel" and agg["W_bank_n_valid"] == 3 and agg["n_transects"] == 5
    assert "inconsistent_along_stream" not in agg["qc_flags"]
    agg2 = aggregate(ms[:2] + ms[3:5], PIX, ccfg)                     # 2 vs 2 with one transect missing
    assert agg2["point_class"] == "ambiguous" and "inconsistent_along_stream" in agg2["qc_flags"]


def test_tobit_right_censoring():
    logA, logW, cens, lb = _synthetic_widths(n=150, c=0.02, b=0.45)
    right = logW > np.log10(12.0)
    lb2 = np.where(right, np.log10(12.0), lb)
    f = tobit_fit(logA, logW, cens & ~right, lb2, right)
    assert f["converged"] and f["b"] == pytest.approx(0.45, abs=0.06)
