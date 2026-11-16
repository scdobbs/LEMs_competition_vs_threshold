"""Addendum 3, section 4: the chi estimators and n_eff on constructed per-basin statistics."""

import numpy as np

from lemcvt.measure.audits import _n_eff, concat_moments, estimators, subset_moments

THETAS = np.round(np.arange(0.2, 0.71, 0.01), 3)


def _moments_for_basins(ks_true, theta_true, sizes, seed=0):
    """Per-basin moments of synthetic profiles z = ks * chi(theta_true) + noise, chi(theta) = chi_true ** (theta/theta_true)."""
    rng = np.random.default_rng(seed)
    n_l, sx, sxx, sxy, sy, syy = [], [], [], [], [], []
    for ks, size in zip(ks_true, sizes, strict=True):
        base = np.sort(rng.uniform(0.2, 1.0, size)) * 20.0            # chi at theta_true
        chis = base[:, None] ** (THETAS[None, :] / theta_true)       # a family that is exact at theta_true
        z = ks * base + rng.normal(0, 0.05, size)
        n_l.append(size)
        sx.append(chis.sum(axis=0))
        sxx.append((chis * chis).sum(axis=0))
        sxy.append((chis * z[:, None]).sum(axis=0))
        sy.append(z.sum())
        syy.append((z * z).sum())
    return {"n": np.array(n_l, float), "sx": np.array(sx), "sxx": np.array(sxx), "sxy": np.array(sxy), "sy": np.array(sy),
            "syy": np.array(syy), "truncated": np.zeros(len(sizes), bool), "outlet_flat": np.arange(len(sizes)),
            "n_basins_total": len(sizes), "n_px_kept": int(sum(sizes))}


def test_n_eff_equal_weights_is_count():
    assert _n_eff(np.ones(7)) == 7.0
    assert abs(_n_eff(np.array([1.0, 0.0, 0.0])) - 1.0) < 1e-12
    assert 1.0 < _n_eff(np.array([10.0, 1.0, 1.0])) < 2.0


def test_estimators_recover_theta_and_report_n_eff():
    mom = _moments_for_basins([5.0] * 6, 0.45, [60, 60, 60, 60, 60, 60])
    est = estimators(mom, THETAS, n_boot=50)
    assert est["n_basins"] == 6
    for key in ("pooled", "equal", "median"):
        assert abs(est[key]["theta_star"] - 0.45) <= 0.011, key
    assert abs(est["pooled"]["ks"] - 5.0) < 0.05
    assert 5.5 < est["pooled"]["n_eff_centred"] <= 6.0           # equal basins -> n_eff near the count
    assert est["equal"]["n_eff"] == 6.0
    assert est["pooled"]["theta_ci68"][0] <= 0.45 <= est["pooled"]["theta_ci68"][1] + 0.011


def test_dominant_basin_drives_pooled_but_not_equal_weight():
    """One huge basin with theta 0.60 and five small ones with theta 0.35: the pooled estimator follows the big
    basin (n_eff near 1), the equal-weight and median estimators follow the majority."""
    big = _moments_for_basins([5.0], 0.60, [2000], seed=1)
    big["sx"] *= 40          # scale chi x40, z x8
    big["sxx"] *= 1600
    big["sxy"] *= 40 * 8
    big["sy"] *= 8
    big["syy"] *= 64
    small = _moments_for_basins([5.0] * 5, 0.35, [40] * 5, seed=2)
    mom = concat_moments([big, small])
    est = estimators(mom, THETAS, n_boot=0)
    assert est["pooled"]["theta_star"] > 0.5
    assert est["pooled"]["n_eff_centred"] < 1.5
    assert est["pooled"]["n_eff_lt_3"]
    assert est["equal"]["theta_star"] < 0.45
    assert abs(est["median"]["theta_star"] - 0.35) <= 0.02
    sub = estimators(subset_moments(mom, np.array([False] + [True] * 5)), THETAS, n_boot=0)
    assert abs(sub["pooled"]["theta_star"] - 0.35) <= 0.011
