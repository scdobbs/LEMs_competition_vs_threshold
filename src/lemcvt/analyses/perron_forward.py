"""Forward runs with the regional (Perron et al. 2012) parameters (manuscript Fig. 2).

No v1 code for this figure exists in the repository (Phase 0 report, section 6);
the figure was built outside it. The definition used here: for each tile run the AD
model at the site's regional (U, K, D, m) and the linear partitioned model at the
same (U, K, D, m) with the regional A_crit, on the tile's own grid, and record MSEs
and grids. The regional values live in ``configs/sites/*.yaml`` under ``regional``
and are flagged there as "to be verified against the paper".
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np

from lemcvt.analyses.outputs import write_model_outputs
from lemcvt.config import config_snapshot, load_tile
from lemcvt.models.ad import advection_diffusion_model
from lemcvt.models.partitioned import partitioned_threshold_model
from lemcvt.results import RunRecorder

logger = logging.getLogger(__name__)


def run_perron_forward(tiles: Sequence[str], run_id: str | None = None, save_grids: bool = True,
                       derived_grids: bool = False, results_root=None) -> RunRecorder:
    rec = RunRecorder("perron_forward", "_".join(tiles) if len(tiles) <= 3 else f"{len(tiles)}tiles",
                      run_id=run_id, root=results_root)
    out = {}
    snaps = {}
    for tile in tiles:
        td = load_tile(tile)
        snaps[tile] = config_snapshot(td.cfg)
        reg = td.site.regional
        for key in ("K", "D", "m", "A_crit"):
            if key not in reg:
                raise ValueError(f"site {td.site.site}: regional.{key} missing in site config")
        U = float(reg.get("U", td.site.U))
        K, D, m, A_crit = float(reg["K"]), float(reg["D"]), float(reg["m"]), float(reg["A_crit"])
        Z0, A, FD, dx = td.Z, td.A, td.FD, td.dx
        Z_ad = advection_diffusion_model(Z0, A, FD, dx, m)(U, K, D)
        Z_pt, mask = partitioned_threshold_model(Z0, A, FD, dx, m, U, K)(A_crit, D)
        res = {
            "params": {"U": U, "K": K, "D": D, "m": m, "A_crit": A_crit, "source": reg.get("source")},
            "z_units_working": td.z_units_working,
            "mse_ad": float(np.nanmean((Z_ad - Z0) ** 2)),
            "mse_partitioned": float(np.nanmean((Z_pt - Z0) ** 2)),
            "n_channel": int(mask.sum()),
            "relief_obs": float(np.nanmax(Z0) - np.nanmin(Z0)),
            "relief_ad": float(np.nanmax(Z_ad) - np.nanmin(Z_ad)),
            "relief_partitioned": float(np.nanmax(Z_pt) - np.nanmin(Z_pt)),
        }
        if save_grids:
            res["files_ad"] = write_model_outputs(rec, td.filled, Z_ad, f"{tile}_regional_AD", derived=derived_grids)
            res["files_partitioned"] = write_model_outputs(rec, td.filled, Z_pt, f"{tile}_regional_partitioned",
                                                           derived=derived_grids)
        logger.info("%s regional forward: MSE AD=%.4e partitioned=%.4e", tile, res["mse_ad"], res["mse_partitioned"])
        out[tile] = res
    rec.add_params("tiles", out)
    rec.finalize({"tiles": snaps})
    return rec
