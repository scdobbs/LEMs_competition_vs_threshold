"""Steady-state landscape evolution models (exact ports of v1 ``src/models.py``,
``src/models_dinf.py`` and ``src/synthetic.py``).

All factories take plain arrays ``(Z0, A, FD, dx)`` instead of TopoAnalysis
objects; the numerical operations are those of v1, in the same order.
"""

from lemcvt.models.ad import advection_diffusion_model
from lemcvt.models.partitioned import partitioned_threshold_model
from lemcvt.models.partitioned_nonlinear import partitioned_threshold_model_nonlinear

__all__ = [
    "advection_diffusion_model",
    "partitioned_threshold_model",
    "partitioned_threshold_model_nonlinear",
]
