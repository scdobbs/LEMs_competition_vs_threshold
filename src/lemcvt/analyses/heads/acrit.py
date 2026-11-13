"""Per-tile A_crit brackets and exponents for the head-extraction port (addendum 8, 1.4).

The feature branch hard-coded the v1 Table 1 values. Here the final A_crit is the latest Phase 6 refined linear joint
optimum, the stage-2 A_crit the same run's stage-2 optimum at D0_A, and the exponent the site m of the tile config;
the v1 values stay as the fallback (and for the record) when no Phase 6 run is found.
"""

from __future__ import annotations

import functools
import logging
from typing import Any

logger = logging.getLogger(__name__)

TILE_KEYS = [(s, t) for s in ("AP", "GM") for t in (1, 2, 3, 4, 5)]

A_CRIT_M2_V1: dict = {
    ("AP", 1): 6.4143e3, ("AP", 2): 7.6456e3, ("AP", 3): 9.7415e3, ("AP", 4): 7.7979e3, ("AP", 5): 1.8870e3,
    ("GM", 1): 8.0924e3, ("GM", 2): 2.9600e3, ("GM", 3): 6.5263e3, ("GM", 4): 7.3962e3, ("GM", 5): 2.2471e3,
}
A_CRIT_STAGE2_M2_V1: dict = {
    ("AP", 1): 3.0634e3, ("AP", 2): 2.3378e3, ("AP", 3): 1.7221e3, ("AP", 4): 9.9501e2, ("AP", 5): 1.1300e3,
    ("GM", 1): 4.2349e2, ("GM", 2): 1.0000e2, ("GM", 3): 1.2974e3, ("GM", 4): 4.2209e2, ("GM", 5): 2.6142e2,
}
THETA_V1: dict = {
    ("AP", 1): 0.562, ("AP", 2): 0.592, ("AP", 3): 0.542, ("AP", 4): 0.438, ("AP", 5): 0.400,
    ("GM", 1): 0.379, ("GM", 2): 0.308, ("GM", 3): 0.417, ("GM", 4): 0.313, ("GM", 5): 0.300,
}


@functools.lru_cache(maxsize=1)
def acrit_tables() -> dict:
    """``{"final": {(site, t): A_crit}, "stage2": {...}, "theta": {...}, "source": {...}}`` from the Phase 6 runs, else v1."""
    final: dict = {}
    stage2: dict = {}
    theta: dict = {}
    source: dict = {}
    try:
        from lemcvt.analyses.curvature_compare import _runs_by_stage
        from lemcvt.config import load_tile_config

        for s, t in TILE_KEYS:
            tile = f"{s}{t}"
            try:
                runs = _runs_by_stage(tile)
            except Exception:  # noqa: BLE001
                runs = {}
            if "joint" in runs:
                r, p = runs["joint"]
                final[(s, t)] = float(p["v2"]["partitioned"]["joint_optimum"]["A_crit"])
                st2 = p["v2"]["partitioned"].get("stage2_optimum")
                if st2:
                    stage2[(s, t)] = float(st2["A_crit"])
                source[(s, t)] = r.name
            try:
                theta[(s, t)] = float(load_tile_config(tile).model.m)
            except Exception:  # noqa: BLE001
                pass
    except Exception as exc:  # noqa: BLE001
        logger.warning("Phase 6 A_crit lookup failed (%s); using the v1 tables", exc)
    out: dict[str, Any] = {"final": dict(A_CRIT_M2_V1), "stage2": dict(A_CRIT_STAGE2_M2_V1), "theta": dict(THETA_V1), "source": {k: "v1 Table 1" for k in TILE_KEYS}}
    out["final"].update(final)
    out["stage2"].update(stage2)
    out["theta"].update(theta)
    out["source"].update(source)
    out["n_from_phase6"] = len(final)
    return out


def acrit_bracket() -> dict:
    t = acrit_tables()
    return {k: (t["stage2"][k], t["final"][k]) for k in TILE_KEYS}
