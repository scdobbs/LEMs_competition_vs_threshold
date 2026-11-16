"""Optimizers. ``v1`` is the exact port of the notebook optimizers (scipy minimize in
log space with LRU-cached forward solves); ``v2`` (Phase 4) is selected with
``--optimizer v2``."""

from lemcvt.optimize.v1 import (
    Trajectory,
    optimize_advection_diffusion,
    optimize_threshold_acrit,
    optimize_threshold_acrit_nonlinear,
    optimize_threshold_full,
)

__all__ = [
    "Trajectory",
    "optimize_advection_diffusion",
    "optimize_threshold_acrit",
    "optimize_threshold_acrit_nonlinear",
    "optimize_threshold_full",
]
