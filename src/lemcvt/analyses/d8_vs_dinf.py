"""D8 against D-infinity comparison (``d8_vs_dinf_comparison``).

v1 ran AP1 and GM1 at the *measured* parameters (K0 = U/k_s and D0 = U/C_ht rounded
to two significant figures) with A_crit = 5000 m^2; those values are kept under
``params="v1"``. ``params="config"`` uses the tile config's K0_partitioned, D0, m and
A_crit_0 instead.

Phase 4b (addendum 3 section 6, addendum 4 section 4): the partitioned D-infinity model is the
rebuilt coupled system (``lemcvt.models.dinf_partitioned``); ``pt_dinf_model="both"`` also runs the
v1 explicit-integration model so the two MSEs sit side by side, and the run records the channel
monotonicity of the rebuilt model (test 4 of the addendum). ``units="feet"`` loads the AP tile in
its stored unit (as v1 ran and as the addendum asks for this rerun) whatever the config says.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from lemcvt.config import config_snapshot, load_tile
from lemcvt.hydro.dinf import compute_dinf
from lemcvt.measure.slope_area import slope_area_arrays_from_elevation
from lemcvt.models.ad import advection_diffusion_model
from lemcvt.models.dinf_models import advection_diffusion_model_dinf, partitioned_threshold_model_dinf_v1
from lemcvt.models.dinf_partitioned import partitioned_threshold_model_dinf
from lemcvt.models.partitioned import partitioned_threshold_model
from lemcvt.results import RunRecorder

logger = logging.getLogger(__name__)

V1_CONFIGS: dict[str, dict[str, Any]] = {
    "AP1": {"U": 5.5e-5, "K": 4.0e-7, "D": 2.8e-3, "m": 0.562, "A_crit": 5000.0, "label": "Allegheny Plateau (tile 1)"},
    "GM1": {"U": 1.47e-4, "K": 2.2e-5, "D": 1.0e-2, "m": 0.379, "A_crit": 5000.0, "label": "Gabilan Mesa (tile 1)"},
}
V1_PRINTED_MSE = {  # d8_vs_dinf_comparison notebook, v1 (00_reference.md 3.5): AD d8, AD dinf, PT d8, PT dinf
    "AP1": {"mse_ad_d8": 6.9581e5, "mse_ad_dinf": 7.0133e5, "mse_pt_d8": 2.1950e3, "mse_pt_dinf_v1": 3.7650e6},
    "GM1": {"mse_ad_d8": 3.2286e5, "mse_ad_dinf": 3.1845e5, "mse_pt_d8": 7.9204e2, "mse_pt_dinf_v1": 4.2395e3},
}
MIN_AREA_SA = 2000.0
UNITS_OVERRIDE = {"stored": None, "feet": True, "metres": False}


def run_d8_vs_dinf(tiles: Sequence[str] = ("AP1", "GM1"), params: str = "v1", run_id: str | None = None,
                   save_grids: bool = True, results_root=None, pt_dinf_model: str = "both", units: str = "stored",
                   synthetic: bool = True) -> RunRecorder:
    if units not in UNITS_OVERRIDE:
        raise ValueError("units must be stored, feet or metres")
    rec = RunRecorder("d8_vs_dinf", "_".join(tiles), run_id=run_id, root=results_root)
    out = {}
    sa_frames = []
    snaps = {}
    for tile in tiles:
        td = load_tile(tile, reproduce_v1_units=UNITS_OVERRIDE[units])
        snaps[tile] = config_snapshot(td.cfg)
        cfg: dict[str, Any]
        if params == "v1":
            if tile not in V1_CONFIGS:
                raise ValueError(f"no v1 parameters for {tile}; use params='config'")
            cfg = dict(V1_CONFIGS[tile])
            if tile.startswith("AP") and td.z_units_working == "m":
                # the v1 numbers are feet-based: K_m = c K_ft, D_m = c D_ft (addendum 4, section 3)
                cfg["K"] *= td.cfg.model.param_scale
                cfg["D"] *= td.cfg.model.param_scale
                cfg["note"] = f"v1 K, D x {td.cfg.model.param_scale:.9f} for the metre grid"
        else:
            mp = td.cfg.model
            cfg = {"U": mp.U, "K": mp.K0_partitioned, "D": mp.D0, "m": mp.m, "A_crit": mp.A_crit_0, "label": tile}
        Z0, A, FD, dx = td.Z, td.A, td.FD, td.dx
        angles, r_facet, facet_idx, A_dinf = compute_dinf(Z0, dx, FD)
        m, U, K, D = float(cfg["m"]), float(cfg["U"]), float(cfg["K"]), float(cfg["D"])

        Z_d8 = advection_diffusion_model(Z0, A, FD, dx, m)(U, K, D)
        Z_dinf = advection_diffusion_model_dinf(Z0, A, FD, dx, m, angles, r_facet, facet_idx, A_dinf)(U, K, D)
        Z_pt_d8, mask_d8 = partitioned_threshold_model(Z0, A, FD, dx, m, U, K)(cfg["A_crit"], D)
        valid = np.isfinite(Z0)

        def _mse(Z):
            return float(np.nanmean((Z[valid] - Z0[valid]) ** 2))

        res: dict[str, Any] = {
            "params": cfg, "shape": list(Z0.shape), "z_units_working": td.z_units_working,
            "A_dinf_range": [float(np.nanmin(A_dinf)), float(np.nanmax(A_dinf))],
            "A_d8_range": [float(np.nanmin(A)), float(np.nanmax(A))],
            "mse_ad_d8": _mse(Z_d8), "mse_ad_dinf": _mse(Z_dinf), "mse_pt_d8": _mse(Z_pt_d8),
            "n_channel_d8": int(mask_d8.sum()), "v1_printed": V1_PRINTED_MSE.get(tile),
        }
        grids = [("AD_d8", Z_d8), ("AD_dinf", Z_dinf), ("PT_d8", Z_pt_d8)]
        if pt_dinf_model in ("v2", "both"):
            f = partitioned_threshold_model_dinf(Z0, A, FD, dx, m, U, K, angles, r_facet, facet_idx, A_dinf)
            Z_pt_dinf, mask_dinf = f(cfg["A_crit"], D)
            s = f.last_system
            res.update({"mse_pt_dinf": _mse(Z_pt_dinf), "n_channel_dinf": int(mask_dinf.sum()), "pt_dinf_system": s.info,
                        "pt_dinf_monotonicity": f.monotonicity(cfg["A_crit"], D),
                        "pt_dinf_hillslope_rows_in_channel_mask": int(np.sum(mask_dinf[td.filled.data == td.filled.data] & False))})
            res.pop("pt_dinf_hillslope_rows_in_channel_mask")
            res["pt_dinf_channel_rms_error"] = float(np.sqrt(np.nanmean((Z_pt_dinf - Z0)[mask_dinf] ** 2)))
            res["pt_d8_channel_rms_error"] = float(np.sqrt(np.nanmean((Z_pt_d8 - Z0)[mask_d8] ** 2)))
            grids.append(("PT_dinf", Z_pt_dinf))
        if pt_dinf_model in ("v1", "both"):
            Z_v1, mask_v1 = partitioned_threshold_model_dinf_v1(Z0, A, FD, dx, m, U, K, angles, r_facet, facet_idx, A_dinf)(cfg["A_crit"], D)
            res["mse_pt_dinf_v1"] = _mse(Z_v1)
            res["n_channel_dinf_v1"] = int(mask_v1.sum())
            res["pt_dinf_v1_channel_rms_error"] = float(np.sqrt(np.nanmean((Z_v1 - Z0)[mask_v1] ** 2)))
            grids.append(("PT_dinf_v1", Z_v1))
        logger.info("%s (%s) MSE AD D8=%.4e D-inf=%.4e | PT D8=%.4e D-inf rebuilt=%s v1=%s", tile, td.z_units_working, res["mse_ad_d8"],
                    res["mse_ad_dinf"], res["mse_pt_d8"], f"{res['mse_pt_dinf']:.4e}" if "mse_pt_dinf" in res else "-",
                    f"{res['mse_pt_dinf_v1']:.4e}" if "mse_pt_dinf_v1" in res else "-")
        if "pt_dinf_monotonicity" in res:
            logger.info("%s rebuilt D-inf channel monotonicity: %s", tile, res["pt_dinf_monotonicity"])
        if save_grids:
            for name, Z in grids:
                rec.write_grid(td.filled.with_data(Z), f"{tile}_{name}_elevation")
            rec.write_grid(td.filled.with_data(A_dinf), f"{tile}_A_dinf")
        for name, Z in (("observed", Z0), ("AD_d8", Z_d8), ("AD_dinf", Z_dinf)):
            Aa, Sa = slope_area_arrays_from_elevation(Z, A, FD, dx, min_area=MIN_AREA_SA)
            sa_frames.append(pd.DataFrame({"tile": tile, "series": name, "A": Aa, "S": Sa}))
        out[tile] = res
    rec.add_params("tiles", out)
    rec.add_params("params_source", params)
    rec.add_params("units", units)
    rec.add_params("pt_dinf_model", pt_dinf_model)
    if synthetic:
        rec.add_params("synthetic", synthetic_cases())
    rec.write_table(pd.concat(sa_frames, ignore_index=True), "slope_area")
    rec.finalize({"tiles": snaps})
    return rec


def synthetic_cases(n: int = 201, dx: float = 1.0) -> dict[str, Any]:
    """The Phase 4b synthetic checks at report size: aligned V and plane (must agree with D8 to 1e-10 of relief),
    rotated V against D8 at 4x finer resolution (reported, no threshold), for the rebuilt and the v1 model."""
    from lemcvt.analyses.dinf_diagnostics import _padded_hydro, v_valley
    from lemcvt.models.dinf_partitioned import channel_monotonicity

    m, U, K, D, A_crit = 0.4, 1.47e-4, 2.2e-5, 1.0e-2, 500.5
    out: dict[str, Any] = {"n": n, "dx": dx, "m": m, "U": U, "K": K, "D": D, "A_crit": A_crit}
    cases = {"plane_south": (0.0, 0.1), "v_equal_slopes": (0.1, 0.1), "v_rotated_20deg": (0.1, 0.1 * math.tan(math.radians(20)))}
    for name, (side, axial) in cases.items():
        zf, fd, area, angles, r_facet, facet_idx, A_dinf = _padded_hydro(v_valley(n, n, dx, side, axial), dx)
        relief = float(np.nanmax(zf) - np.nanmin(zf))
        Z_d8, mask_d8 = partitioned_threshold_model(zf, area, fd, dx, m, U, K)(A_crit, D)
        f = partitioned_threshold_model_dinf(zf, area, fd, dx, m, U, K, angles, r_facet, facet_idx, A_dinf)
        Z_di, mask_di = f(A_crit, D)
        Z_v1, _ = partitioned_threshold_model_dinf_v1(zf, area, fd, dx, m, U, K, angles, r_facet, facet_idx, A_dinf)(A_crit, D)
        rec = {"relief": relief, "n_channel_d8": int(mask_d8.sum()), "n_channel_dinf": int(mask_di.sum()),
               "n_mask_differs": int(np.sum(mask_d8 != mask_di)),
               "max_abs_dZ_dinf_minus_d8_over_relief": float(np.nanmax(np.abs(Z_di - Z_d8)) / relief),
               "max_abs_dZ_dinf_v1_minus_d8_over_relief": float(np.nanmax(np.abs(Z_v1 - Z_d8)) / relief),
               "mse_vs_input": {"pt_d8": float(np.nanmean((Z_d8 - zf) ** 2)), "pt_dinf": float(np.nanmean((Z_di - zf) ** 2)),
                                "pt_dinf_v1": float(np.nanmean((Z_v1 - zf) ** 2))},
               "system": f.last_system.info, "monotonicity": channel_monotonicity(f.last_z_flat, f.last_system)}
        if name == "v_rotated_20deg":
            nf, dxf = 4 * (n - 1) + 1, dx / 4
            zf_f, fd_f, area_f, *_ = _padded_hydro(v_valley(nf, nf, dxf, side, axial), dxf)
            Zf, _ = partitioned_threshold_model(zf_f, area_f, fd_f, dxf, m, U, K)(A_crit, D)
            Zs = Zf[::4, ::4]
            inner = (slice(2, -2), slice(2, -2))
            rec["against_fine_d8"] = {
                "fine_n": nf, "fine_dx": dxf, "sampling": "fine D8 solution at the coincident coarse nodes, outer two rows/columns excluded",
                "max_abs_dZ_over_relief": {"dinf_rebuilt": float(np.nanmax(np.abs(Z_di - Zs)[inner]) / relief),
                                           "d8_coarse": float(np.nanmax(np.abs(Z_d8 - Zs)[inner]) / relief),
                                           "dinf_v1": float(np.nanmax(np.abs(Z_v1 - Zs)[inner]) / relief)},
                "rms_dZ_over_relief": {"dinf_rebuilt": float(np.sqrt(np.nanmean((Z_di - Zs)[inner] ** 2)) / relief),
                                       "d8_coarse": float(np.sqrt(np.nanmean((Z_d8 - Zs)[inner] ** 2)) / relief),
                                       "dinf_v1": float(np.sqrt(np.nanmean((Z_v1 - Zs)[inner] ** 2)) / relief)}}
        out[name] = rec
        logger.info("synthetic %s: dZ(dinf-d8)/relief %.3e (v1 %.3e), monotonicity %s", name, rec["max_abs_dZ_dinf_minus_d8_over_relief"],
                    rec["max_abs_dZ_dinf_v1_minus_d8_over_relief"], rec["monotonicity"]["n_lower_than_every_receiver"])
    return out
