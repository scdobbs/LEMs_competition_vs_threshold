#!/usr/bin/env python3
"""Addendum 23, Part B items 3-4 (decisions 2.4-2.5): on the lidar and every model surface of the domain, the power spectrum ratio,
the valley share, the 12 m head products (the addendum 22 method), and, when a 1 m finder was accepted, its heads too.

Surfaces: obs (the lidar), AD, PF_C, PF_A, F2_relief, SJ_C from <run>/fits/*_surfaces.npz (those present). Outputs <run>/measures/
measures.json, spectra.npz, surfaces12.npz (12 m products per surface), heads1m_models.npz. usage: a23_measures.py
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from a23_common import (  # noqa: E402
    BLOCK_M,
    MODELS,
    RATIO_BAND,
    READING_BAND_M,
    VALLEY_K_SD,
    VALLEY_SHARE_RANGE,
    VALLEY_SIGMAS_M,
    coarse_products,
    contour_curvature,
    dump,
    fill_and_route,
    gaussian_smooth_nan,
    head_statistic_1m,
    json_ready,
    largest_inscribed_square,
    load_domain,
    load_json,
    mfd_area,
    out_path,
    power_spectrum_binned,
    rolloff_wavelength,
    slurm_record,
    wavelength_bins,
)


def load_surfaces(dom: dict[str, Any]) -> dict[str, np.ndarray]:
    S: dict[str, np.ndarray] = {"obs": dom["Z"]}
    for f in ("fixed", "AD", "SJ_C"):
        p = out_path("fits", f"{f}_surfaces.npz")
        if os.path.exists(p):
            z = np.load(p)
            for k in z.files:
                S[k] = z[k].astype(float)
    return {k: S[k] for k in MODELS if k in S}


def finder_on_surface(name: str, Zs: np.ndarray, FD_obs: np.ndarray) -> dict[str, np.ndarray]:
    """The accepted 1 m finder applied to a model surface exactly as to the lidar (its own routing)."""
    if name == "mfd":
        fr = fill_and_route(Zs, 1.0)
        Am = mfd_area(fr["Zf"], fr["FD"], 1.0)
        st = head_statistic_1m(Am, fr["Zf"], fr["FD"], 1.0)
    else:
        sigma = float(name[3:])
        fr = fill_and_route(gaussian_smooth_nan(Zs, sigma), 1.0)
        st = head_statistic_1m(fr["A"], fr["Zf"], fr["FD"], 1.0)
    return {"M": st["M"].astype(np.float32), "Mdon": st["Mdon"].astype(np.float32)}


def main() -> None:
    t0 = time.time()
    slurm_record(os.path.dirname(out_path("measures", "measures.json")))
    dom = load_domain()
    S = load_surfaces(dom)
    domain, poly = dom["domain"], dom["poly"]
    rec: dict[str, Any] = {"surfaces": list(S), "n_domain_cells": int(domain.sum())}
    print(f"surfaces: {list(S)} [{time.time() - t0:.0f} s]", flush=True)
    # ---------------------------------------------------------------- power spectra on the largest square inside the domain
    r0, c0, side = largest_inscribed_square(domain)
    bins = wavelength_bins()
    sq = (slice(r0, r0 + side), slice(c0, c0 + side))
    rec["spectrum"] = {"square": {"row0": r0, "col0": c0, "side_m": side, "share_in_polygon": float(poly[sq].mean()), "x0": dom["x0"] + c0, "y1": dom["y1"] - r0},
                       "bin_edges_m": bins.tolist(), "reading_band_m": list(READING_BAND_M), "ratio_band": list(RATIO_BAND), "per_surface": {}}
    spectra: dict[str, np.ndarray] = {"bin_edges_m": bins}
    base = None
    for name, Zs in S.items():
        z = Zs[sq]
        assert np.isfinite(z).all(), (name, int((~np.isfinite(z)).sum()))
        ps = power_spectrum_binned(z, 1.0, bins)
        spectra[f"{name}_power"] = ps["power"]
        spectra["wavelength_m"] = ps["wavelength_m"]
        spectra["n_per_bin"] = ps["n"]
        q: dict[str, Any] = {"variance_detrended_m2": ps["variance_detrended"], "power": ps["power"].tolist()}
        if name == "obs":
            base = ps["power"]
            q["rolloff"] = rolloff_wavelength(ps["wavelength_m"], ps["power"])
        else:
            ratio = ps["power"] / base
            q["ratio_to_lidar"] = ratio.tolist()
            inb = (ps["wavelength_m"] >= READING_BAND_M[0]) & (ps["wavelength_m"] <= READING_BAND_M[1]) & np.isfinite(ratio)
            q["ratio_in_reading_band"] = {"min": float(ratio[inb].min()), "max": float(ratio[inb].max()), "n_bins": int(inb.sum()),
                                          "within_band_every_bin": bool(np.all((ratio[inb] >= RATIO_BAND[0]) & (ratio[inb] <= RATIO_BAND[1])))}
            short = (ps["wavelength_m"] < READING_BAND_M[0]) & np.isfinite(ratio)
            q["ratio_below_20m_median"] = float(np.median(ratio[short])) if short.any() else None
            q["ratio_20_60m_median"] = float(np.median(ratio[(ps["wavelength_m"] >= 20) & (ps["wavelength_m"] <= 60) & np.isfinite(ratio)]))
        rec["spectrum"]["per_surface"][name] = q
    print(f"spectra on a {side} m square; lidar roll-off {rec['spectrum']['per_surface']['obs']['rolloff']}; ratios 20-200 m: " +
          "; ".join(f"{n} {v['ratio_in_reading_band']['min']:.2f}-{v['ratio_in_reading_band']['max']:.2f}" for n, v in rec["spectrum"]["per_surface"].items() if n != "obs") + f" [{time.time() - t0:.0f} s]", flush=True)
    # ---------------------------------------------------------------- valley share
    rec["valley"] = {"k_sd": VALLEY_K_SD, "range": list(VALLEY_SHARE_RANGE), "per_sigma": {}}
    for sigma in VALLEY_SIGMAS_M:
        kc_obs = contour_curvature(S["obs"], 1.0, sigma)
        v = kc_obs[domain & np.isfinite(kc_obs)]
        thr = float(v.mean() + VALLEY_K_SD * v.std())
        srec: dict[str, Any] = {"threshold_1_per_m": thr, "lidar_mean": float(v.mean()), "lidar_sd": float(v.std()), "per_surface": {}}
        for name, Zs in S.items():
            kc = kc_obs if name == "obs" else contour_curvature(Zs, 1.0, sigma)
            inside = domain & np.isfinite(kc)
            share = float(np.mean(kc[inside] > thr))
            share_poly = float(np.mean(kc[inside & poly] > thr))
            srec["per_surface"][name] = {"share_domain": share, "share_polygon": share_poly, "n_cells": int(inside.sum())}
        ob = srec["per_surface"]["obs"]["share_domain"]
        for q in srec["per_surface"].values():
            q["ratio_to_lidar"] = q["share_domain"] / ob if ob else None
            q["within_range"] = bool(q["ratio_to_lidar"] is not None and VALLEY_SHARE_RANGE[0] <= q["ratio_to_lidar"] <= VALLEY_SHARE_RANGE[1])
        rec["valley"]["per_sigma"][f"{sigma:g}"] = srec
        print(f"valley share sigma {sigma:g} m: threshold {thr:.4f}; " + "; ".join(f"{n} {q['share_domain']:.3f} ({q['ratio_to_lidar']:.2f} x)" for n, q in srec["per_surface"].items()) + f" [{time.time() - t0:.0f} s]", flush=True)
    # ---------------------------------------------------------------- the 12 m head products of every surface (the addendum 22 method on the box lattice)
    arrays12: dict[str, np.ndarray] = {}
    rec["products12"] = {}
    nb_r, nb_c = domain.shape[0] // BLOCK_M, domain.shape[1] // BLOCK_M
    inside12 = (domain & poly)[BLOCK_M // 2: nb_r * BLOCK_M: BLOCK_M, BLOCK_M // 2: nb_c * BLOCK_M: BLOCK_M][:nb_r, :nb_c]
    arrays12["inside12_centre_in_domain_and_polygon"] = inside12
    arrays12["domain12_centre"] = domain[BLOCK_M // 2: nb_r * BLOCK_M: BLOCK_M, BLOCK_M // 2: nb_c * BLOCK_M: BLOCK_M][:nb_r, :nb_c]
    for name, Zs in S.items():
        t1 = time.perf_counter()
        P = coarse_products(Zs, 1.0)
        for k, v in P.items():
            arrays12[f"{name}_{k}"] = v
        rec["products12"][name] = {"n_valid_12m": int(np.isfinite(P["Z12"]).sum()), "wall_s": time.perf_counter() - t1}
    np.savez_compressed(out_path("measures", "surfaces12.npz"), **arrays12)
    print(f"12 m products done [{time.time() - t0:.0f} s]", flush=True)
    # ---------------------------------------------------------------- the accepted 1 m finder on every model surface
    acc_p = out_path("heads", "accepted.json")
    acc = load_json(acc_p).get("accepted") if os.path.exists(acc_p) else None
    rec["finder_1m"] = {"accepted": acc}
    if acc is not None:
        arr: dict[str, np.ndarray] = {}
        for name, Zs in S.items():
            if name == "obs":
                continue
            st = finder_on_surface(acc["candidate"], Zs, dom["FD"])
            arr[f"{name}_M"], arr[f"{name}_Mdon"] = st["M"], st["Mdon"]
            print(f"finder {acc['candidate']} on {name} [{time.time() - t0:.0f} s]", flush=True)
        np.savez_compressed(out_path("measures", "heads1m_models.npz"), **arr)
    np.savez_compressed(out_path("measures", "spectra.npz"), **spectra)
    rec["wall_s"] = time.time() - t0
    dump(json_ready(rec), out_path("measures", "measures.json"))
    print(f"measures: done [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
