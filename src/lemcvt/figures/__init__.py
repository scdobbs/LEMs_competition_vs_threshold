"""Figures: built from results only, never from model computation.

``lemcvt figures all`` (or one name) finds the latest run of each analysis step
under ``$LEMCVT_RESULTS`` and writes PDF + PNG into ``figures/`` (or ``--out``).

Available figures
-----------------
slope_area        fig_slope_area: observed vs AD-no-opt / AD-opt / partitioned, AP1 and GM2
supp_fig5         synthetic Peclet: k_s/(U/K) against pixel size, three estimators
supp_fig6         resolution sweep: Ks_mod/Ks_obs against dx, median + IQR by site (D8)
resolution_dinf   same with D8 (solid) and D-infinity (dashed)
d8_vs_dinf        4-panel elevation comparison, area scatter, slope-area overlay per tile
fig3_deltas       log10(K_fit/K0), log10(D_fit/D0) per tile for the AD and joint stages
optimizer_v2      (Phase 4) MSE(A_crit) profiles, AD r-profiles, MSE(A_crit, m) contours with the measured theta(A_c)
exponent_site     (addendum 4, 6.1) site profile S(m) with leave-one-out against the measured site-pooled theta(A_c)
phase6_window_test   (addendum 5, 2.2) C_ht(L) on the option (iii) set with bootstrap bands and the pass rule, per tile
phase6_nonlinear_joint (addendum 5, 3.4) nonlinear MSE over (A_crit, D) with the 1 % and 5 % contours, per tile
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from lemcvt.config import REPO_ROOT

logger = logging.getLogger(__name__)

FIGURE_BUILDERS: dict[str, str] = {
    "slope_area": "lemcvt.figures.slope_area:build",
    "supp_fig5": "lemcvt.figures.supp_fig5:build",
    "supp_fig6": "lemcvt.figures.resolution:build_d8",
    "resolution_dinf": "lemcvt.figures.resolution:build_dinf",
    "d8_vs_dinf": "lemcvt.figures.d8_vs_dinf:build",
    "fig3_deltas": "lemcvt.figures.fig3_deltas:build",
    "measure_diagnostics": "lemcvt.figures.measure_diagnostics:build_all",
    "optimizer_v2": "lemcvt.figures.optimizer_v2:build_all",
    "measure_audit": "lemcvt.figures.measure_diagnostics:audit_estimators",
    "exponent_site": "lemcvt.figures.optimizer_v2:exponent_site",
    "phase6_window_test": "lemcvt.figures.phase6:window_test",
    "phase6_nonlinear_joint": "lemcvt.figures.phase6:nonlinear_joint",
    "phase6_curvature_compare": "lemcvt.figures.phase6:curvature_compare",
    "a8_er_star": "lemcvt.figures.phase8:er_star",
    "a8_slope_threshold": "lemcvt.figures.phase8:slope_threshold",
    "a9_pilot": "lemcvt.figures.phase9:pilot",
    "a9_fair": "lemcvt.figures.phase9:fair",
    "a9_morphology": "lemcvt.figures.phase9:morphology",
    "a10_flux": "lemcvt.figures.phase10:flux",
}


def _resolve(spec: str) -> Callable[..., list[Path]]:
    import importlib

    mod, fn = spec.split(":")
    return getattr(importlib.import_module(mod), fn)


def default_out_dir() -> Path:
    return REPO_ROOT / "figures"


def build_figures(which: str = "all", out_dir: Path | None = None, strict: bool = False) -> list[Path]:
    """Build the requested figure(s). Missing results are reported and skipped unless ``strict``."""
    out_dir = Path(out_dir) if out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(FIGURE_BUILDERS) if which == "all" else [w.strip() for w in which.split(",")]
    paths: list[Path] = []
    for name in names:
        if name not in FIGURE_BUILDERS:
            raise ValueError(f"unknown figure {name!r}; choose from {sorted(FIGURE_BUILDERS)}")
        try:
            paths += _resolve(FIGURE_BUILDERS[name])(out_dir)
        except FileNotFoundError as exc:
            if strict:
                raise
            logger.warning("figure %s skipped: %s", name, exc)
    return paths
