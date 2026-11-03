"""A_crit misfit profile for one tile: MSE vs A_crit with K and D held at their
jointly optimized values (Table 1 / notebook stage 4).

    python -m width_analysis.acrit_profile --site AP --tile 5

Uses the repo's own partitioned model (src/models.py) and the TopoAnalysis
raster loader, exactly as the model-run notebooks do; nothing in src/ is
modified. Output: outputs/widths/acrit_profile/<site><tile>.csv and .png.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PARTITIONED_PARAMS
from .tiles import REPO_ROOT, tile_paths

TOPOANALYSIS_PATH = os.environ.get("TOPOANALYSIS_PATH", "/home/groups/hilley/opt")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--site", default="AP")
    p.add_argument("--tile", type=int, default=5)
    p.add_argument("--n", type=int, default=31, help="log-spaced A_crit values between 1e2 and 1e5 m^2")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    key = (a.site, a.tile)
    prm = PARTITIONED_PARAMS[key]
    out_dir = Path(a.out) if a.out else REPO_ROOT / "outputs" / "widths" / "acrit_profile"
    out_dir.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REPO_ROOT / "src"))
    sys.path.insert(0, TOPOANALYSIS_PATH)
    from TopoAnalysis import dem as d          # noqa: E402
    from models import partitioned_threshold_model  # noqa: E402

    paths = tile_paths(a.site, a.tile)
    dem = d.Elevation.load(str(paths["filled"]))
    area = d.Area.load(str(paths["area"]))
    fd = d.FlowDirectionD8.load(str(paths["fd"]))
    t0 = time.time()
    model = partitioned_threshold_model(dem, area, fd, prm["m"], prm["U"], prm["K"])
    print(f"model built in {time.time() - t0:.0f} s", flush=True)

    acrits = np.unique(np.concatenate([np.logspace(2, 5, a.n), [prm["A_crit"]]]))
    rows = []
    for ac in acrits:
        t1 = time.time()
        Z, mask = model(float(ac), prm["D"])
        mse = float(np.nanmean((Z - dem._griddata) ** 2))
        rows.append(dict(site=a.site, tile=a.tile, A_crit_m2=float(ac), mse=mse,
                         channel_fraction=float(np.mean(mask)), n_channel=int(np.sum(mask)),
                         K=prm["K"], D=prm["D"], U=prm["U"], m=prm["m"], solve_s=round(time.time() - t1, 1)))
        print(f"A_crit={ac:10.1f}  MSE={mse:10.3f}  channel pixels={int(np.sum(mask)):7d}  ({time.time() - t1:.0f} s)", flush=True)
        pd.DataFrame(rows).to_csv(out_dir / f"{a.site}{a.tile}.csv", index=False)
    df = pd.DataFrame(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(df.A_crit_m2, df.mse, marker="o", ms=4, color="#2a78d6", lw=1.5, label="MSE(A_crit) at optimized K, D")
    k = int(np.argmin(df.mse.to_numpy()))
    ax.axvline(prm["A_crit"], color="#eb6834", ls="--", lw=1.2, label=f"notebook optimum A_crit = {prm['A_crit']:.3g} m² (MSE {prm['MSE']:.1f})")
    ax.plot(df.A_crit_m2.iloc[k], df.mse.iloc[k], marker="o", ms=9, mfc="none", mec="#0b0b0b",
            label=f"profile minimum A_crit = {df.A_crit_m2.iloc[k]:.3g} m² (MSE {df.mse.iloc[k]:.1f})")
    ax.set_xscale("log")
    ax.set_xlabel("A_crit (m²)")
    ax.set_ylabel("mean squared elevation misfit (m²)")
    ax.set_title(f"{a.site} tile {a.tile}: K = {prm['K']:.3g}, D = {prm['D']:.3g}, U = {prm['U']:.2g}, m = {prm['m']}", fontsize=9, loc="left")
    ax.grid(color="#efeeea", lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"{a.site}{a.tile}.png", dpi=200)
    print(f"wrote {out_dir}/{a.site}{a.tile}.csv and .png", flush=True)


if __name__ == "__main__":
    main()
