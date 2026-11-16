"""Addendum 9, design probe (not a reported result): wall time and memory of one full-tile Newton solve per form on
AP1 at its fitted S_c, K at the linear joint optimum, A_crit at the joint optimum, D = 1e-2 and 3e-3, from the linear
solution. Prints per-iteration timings so the solver design (direct LU of the Jacobian) can be judged before the pilot."""

import json
import resource
import sys
import time

import numpy as np

from lemcvt.analyses.optimizer_v2 import _latest_refined_partitioned
from lemcvt.config import load_tile
from lemcvt.models.partitioned_nonlinear import partitioned_threshold_model_nonlinear

tile = sys.argv[1] if len(sys.argv) > 1 else "AP1"
S_c = float(sys.argv[2]) if len(sys.argv) > 2 else 0.306241
src = _latest_refined_partitioned(tile)
jo = json.loads((src / "params.json").read_text())["v2"]["partitioned"]["joint_optimum"]
td = load_tile(tile)
mp = td.cfg.model
t0 = time.perf_counter()
f = partitioned_threshold_model_nonlinear(td.Z, td.A, td.FD, td.dx, mp.m, mp.U, float(jo["K"]))
print(f"{tile}: build {time.perf_counter() - t0:.1f} s; A_crit {jo['A_crit']:.1f} K {jo['K']:.3e}", flush=True)
for D in (1e-2, 3e-3):
    for form in ("flux", "centre"):
        t1 = time.perf_counter()
        Z, _ = f(float(jo["A_crit"]), D, S_c, solver="newton", form=form)
        nw = f.last_newton
        mse = float(np.nanmean((Z - td.Z) ** 2))
        print(f"D {D:g} {form}: {nw['n_iter']} it, {nw['stop_reason']}, outcome {nw['outcome']}, rel {nw['rel_residual']:.2e}, "
              f"clip {nw['frac_clipped']:.4f}, mse {mse:.3f}, wall {time.perf_counter() - t1:.1f} s, n_h {nw['n_hillslope']}, "
              f"maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", flush=True)
        for h in nw["iterations"]:
            print("   ", {k: (round(v, 6) if isinstance(v, float) else v) for k, v in h.items()}, flush=True)
