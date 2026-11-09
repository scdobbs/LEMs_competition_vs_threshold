"""D8 against D-infinity comparison (``d8_vs_dinf_comparison``). Exact port.

v1 ran AP1 and GM1 at the *measured* parameters (K0 = U/k_s and D0 = U/C_ht rounded
to two significant figures) with A_crit = 5000 m^2; those values are kept under
``params="v1"``. ``params="config"`` uses the tile config's K0_partitioned, D0, m and
A_crit_0 instead.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from lemcvt.config import config_snapshot, load_tile
from lemcvt.hydro.dinf import compute_dinf
from lemcvt.measure.slope_area import slope_area_arrays_from_elevation
from lemcvt.models.ad import advection_diffusion_model
from lemcvt.models.dinf_models import advection_diffusion_model_dinf, partitioned_threshold_model_dinf
from lemcvt.models.partitioned import partitioned_threshold_model
from lemcvt.results import RunRecorder

logger = logging.getLogger(__name__)

V1_CONFIGS: dict[str, dict[str, Any]] = {
    "AP1": {"U": 5.5e-5, "K": 4.0e-7, "D": 2.8e-3, "m": 0.562, "A_crit": 5000.0, "label": "Allegheny Plateau (tile 1)"},
    "GM1": {"U": 1.47e-4, "K": 2.2e-5, "D": 1.0e-2, "m": 0.379, "A_crit": 5000.0, "label": "Gabilan Mesa (tile 1)"},
}
MIN_AREA_SA = 2000.0


def run_d8_vs_dinf(tiles: Sequence[str] = ("AP1", "GM1"), params: str = "v1", run_id: str | None = None,
                   save_grids: bool = True, results_root=None) -> RunRecorder:
    rec = RunRecorder("d8_vs_dinf", "_".join(tiles), run_id=run_id, root=results_root)
    out = {}
    sa_frames = []
    snaps = {}
    for tile in tiles:
        td = load_tile(tile)
        snaps[tile] = config_snapshot(td.cfg)
        cfg: dict[str, Any]
        if params == "v1":
            if tile not in V1_CONFIGS:
                raise ValueError(f"no v1 parameters for {tile}; use params='config'")
            cfg = V1_CONFIGS[tile]
        else:
            mp = td.cfg.model
            cfg = {"U": mp.U, "K": mp.K0_partitioned, "D": mp.D0, "m": mp.m, "A_crit": mp.A_crit_0, "label": tile}
        Z0, A, FD, dx = td.Z, td.A, td.FD, td.dx
        angles, r_facet, facet_idx, A_dinf = compute_dinf(Z0, dx, FD)
        m, U, K, D = float(cfg["m"]), float(cfg["U"]), float(cfg["K"]), float(cfg["D"])

        Z_d8 = advection_diffusion_model(Z0, A, FD, dx, m)(U, K, D)
        Z_dinf = advection_diffusion_model_dinf(Z0, A, FD, dx, m, angles, r_facet, facet_idx, A_dinf)(U, K, D)
        Z_pt_d8, mask_d8 = partitioned_threshold_model(Z0, A, FD, dx, m, U, K)(cfg["A_crit"], D)
        Z_pt_dinf, mask_dinf = partitioned_threshold_model_dinf(Z0, A, FD, dx, m, U, K, angles, r_facet, facet_idx,
                                                                A_dinf)(cfg["A_crit"], D)
        valid = np.isfinite(Z0)

        def _mse(Z):
            return float(np.nanmean((Z[valid] - Z0[valid]) ** 2))

        res = {
            "params": cfg, "shape": list(Z0.shape),
            "A_dinf_range": [float(np.nanmin(A_dinf)), float(np.nanmax(A_dinf))],
            "A_d8_range": [float(np.nanmin(A)), float(np.nanmax(A))],
            "mse_ad_d8": _mse(Z_d8), "mse_ad_dinf": _mse(Z_dinf), "mse_pt_d8": _mse(Z_pt_d8), "mse_pt_dinf": _mse(Z_pt_dinf),
            "n_channel_d8": int(mask_d8.sum()), "n_channel_dinf": int(mask_dinf.sum()),
        }
        logger.info("%s MSE AD D8=%.4e D-inf=%.4e | PT D8=%.4e D-inf=%.4e", tile, res["mse_ad_d8"], res["mse_ad_dinf"],
                    res["mse_pt_d8"], res["mse_pt_dinf"])
        if save_grids:
            for name, Z in (("AD_d8", Z_d8), ("AD_dinf", Z_dinf), ("PT_d8", Z_pt_d8), ("PT_dinf", Z_pt_dinf)):
                rec.write_grid(td.filled.with_data(Z), f"{tile}_{name}_elevation")
            rec.write_grid(td.filled.with_data(A_dinf), f"{tile}_A_dinf")
        # slope-area arrays (cell 11): observed, D8 AD, D-inf AD at min_area 2000
        for name, Z in (("observed", Z0), ("AD_d8", Z_d8), ("AD_dinf", Z_dinf)):
            Aa, Sa = slope_area_arrays_from_elevation(Z, A, FD, dx, min_area=MIN_AREA_SA)
            sa_frames.append(pd.DataFrame({"tile": tile, "series": name, "A": Aa, "S": Sa}))
        out[tile] = res
    rec.add_params("tiles", out)
    rec.add_params("params_source", params)
    rec.write_table(pd.concat(sa_frames, ignore_index=True), "slope_area")
    rec.finalize({"tiles": snaps})
    return rec
