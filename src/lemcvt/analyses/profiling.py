"""Phase 3 step 1: profile the model stages on a full tile with cProfile and line_profiler.

``lemcvt profile --tile GM1`` runs, on the full tile and in the order the stages use them,

* hydrology: priority-flood fill, D8 directions, D8 area, main-stem graph (resolution sweep);
* AD model build (LIL stencils) and one solve (CSR -> LIL -> CSR Dirichlet rows + SuperLU);
* partitioned model build (grid extraction + channel BFS) and one solve (per-cell LIL assembly
  of the reduced Laplacian + SuperLU);
* nonlinear partitioned forward limited to ``picard_iters`` Picard iterations (slopes + assembly
  + solve per iteration; the full stage runs up to 50);
* a second partitioned build with a different K (what the joint stage does per candidate K),

each under ``cProfile`` (``<step>.pstats`` and the top functions by cumulative time in
``<step>_cprofile.txt``) and, when ``line_profiler`` is importable, with line timings of the hot
functions (``<step>_lineprof.txt``). ``params.json`` holds wall time, peak RSS and the top rows
per step so that `reports/v2/03_performance.md` can quote them.
"""

from __future__ import annotations

import cProfile
import io
import logging
import pstats
import resource
import time
from typing import Any

import numpy as np

from lemcvt.config import config_snapshot, load_tile
from lemcvt.hydro import channel_slope as cs
from lemcvt.hydro import d8 as d8mod
from lemcvt.hydro import fill as fillmod
from lemcvt.models import ad as admod
from lemcvt.models import common
from lemcvt.models import partitioned as ptmod
from lemcvt.models import partitioned_nonlinear as nlmod
from lemcvt.results import RunRecorder

logger = logging.getLogger(__name__)


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _top_rows(pr: cProfile.Profile, n: int = 25) -> tuple[str, list[dict[str, Any]]]:
    buf = io.StringIO()
    st = pstats.Stats(pr, stream=buf)
    st.sort_stats("cumulative").print_stats(60)
    text = buf.getvalue()
    rows = []
    for (file, line, func), (_cc, nc, tt, ct, _callers) in getattr(st, "stats").items():  # noqa: B009
        rows.append({"function": func, "file": file.split("/")[-1], "line": line, "ncalls": nc, "tottime_s": tt, "cumtime_s": ct})
    rows.sort(key=lambda r: -r["cumtime_s"])
    return text, rows[:n]


class _Profiler:
    def __init__(self, rec: RunRecorder, picard_iters: int):
        self.rec = rec
        self.results: dict[str, Any] = {}
        try:
            from line_profiler import LineProfiler  # type: ignore

            self._LP: Any = LineProfiler
        except Exception:  # noqa: BLE001
            self._LP = None
            logger.warning("line_profiler not importable; cProfile only")
        self.picard_iters = picard_iters

    def run(self, name: str, fn, line_funcs=()):
        pr = cProfile.Profile()
        lp = self._LP(*[f for f in line_funcs if f is not None]) if (self._LP is not None and line_funcs) else None
        rss0 = _peak_rss_gb()
        t0 = time.perf_counter()
        pr.enable()
        if lp is not None:
            lp.enable_by_count()
        try:
            out = fn()
        finally:
            if lp is not None:
                lp.disable_by_count()
            pr.disable()
        wall = time.perf_counter() - t0
        pr.dump_stats(self.rec.dir / f"{name}.pstats")
        text, rows = _top_rows(pr)
        (self.rec.dir / f"{name}_cprofile.txt").write_text(text)
        entry: dict[str, Any] = {"wall_s": wall, "peak_rss_gb_after": _peak_rss_gb(), "peak_rss_gb_before": rss0, "top": rows}
        if lp is not None:
            buf = io.StringIO()
            lp.print_stats(stream=buf, output_unit=1e-3)
            (self.rec.dir / f"{name}_lineprof.txt").write_text(buf.getvalue())
            entry["lineprof"] = f"{name}_lineprof.txt"
        self.results[name] = entry
        logger.info("%s: %.1f s (peak RSS %.1f GB)", name, wall, entry["peak_rss_gb_after"])
        return out


def run_profile(tile: str, picard_iters: int = 2, run_id: str | None = None, results_root=None) -> RunRecorder:
    td = load_tile(tile)
    mp = td.cfg.model
    Z0, A, FD, dx, U = td.Z, td.A, td.FD, td.dx, mp.U
    rec = RunRecorder("profile", tile, run_id=run_id, root=results_root)
    rec.add_params("tile", tile)
    rec.add_params("inputs", td.provenance())
    rec.add_params("picard_iters", picard_iters)
    P = _Profiler(rec, picard_iters)

    # hydrology (Phase 3 step 5 target)
    zf = P.run("hydro_fill", lambda: fillmod.priority_flood_fill(Z0, dx), [fillmod.priority_flood_fill])
    fd = P.run("hydro_d8_fd", lambda: d8mod.d8_flow_direction(zf), [d8mod.d8_flow_direction])
    area = P.run("hydro_d8_area", lambda: d8mod.d8_area(fd, zf, dx), [d8mod.d8_area, d8mod.d8_receiver_arrays])
    P.run("hydro_main_stem", lambda: cs.flow_graph_main_stem(area, fd), [cs.flow_graph_main_stem])
    rec.add_params("hydro_check", {"fd_equal_to_stored": bool(np.array_equal(fd, FD)),
                                   "area_max_abs_diff_vs_stored": float(np.nanmax(np.abs(area - A)))})
    zf = fd = area = None  # release memory (not `del`: the lambdas above closed over these names)

    # AD model (Phase 3 step 3 target)
    f_ad = P.run("ad_build", lambda: admod.advection_diffusion_model(Z0, A, FD, dx, mp.m),
                 [admod.advection_diffusion_model, common.extract_grid_data])
    P.run("ad_solve", lambda: f_ad(U, mp.K0_ad, mp.D0), [f_ad])
    P.run("ad_solve_2", lambda: f_ad(U, mp.K0_ad * 1.1, mp.D0 * 0.9), [f_ad])
    f_ad = None

    # partitioned linear (Phase 3 steps 2 and 4 targets)
    f_pt = P.run("pt_build", lambda: ptmod.partitioned_threshold_model(Z0, A, FD, dx, mp.m, U, mp.K0_partitioned),
                 [ptmod.partitioned_threshold_model, common.precompute_channel_network, common.extract_grid_data])
    P.run("pt_solve", lambda: f_pt(mp.A_crit_0, mp.D0), [f_pt, common.solve_hillslope_diffusion])
    P.run("pt_solve_2", lambda: f_pt(mp.A_crit_0 * 0.5, mp.D0), [f_pt, common.solve_hillslope_diffusion])
    P.run("joint_rebuild_newK", lambda: ptmod.partitioned_threshold_model(Z0, A, FD, dx, mp.m, U, mp.K0_partitioned * 1.1),
          [common.precompute_channel_network])
    f_pt = None

    # nonlinear partitioned (Phase 3 step 2 target), limited Picard iterations
    f_nl = P.run("nl_build", lambda: nlmod.partitioned_threshold_model_nonlinear(Z0, A, FD, dx, mp.m, U, mp.K0_partitioned),
                 [nlmod.partitioned_threshold_model_nonlinear])
    P.run(f"nl_solve_{picard_iters}iter", lambda: f_nl(mp.A_crit_0, mp.D0, mp.S_c, picard_iters, mp.nonlinear_tol),
          [f_nl, common.solve_hillslope_diffusion])
    rec.add_params("nl_iterations_run", int(getattr(f_nl, "last_n_iter", 0)))

    rec.add_params("steps", P.results)
    rec.add_params("peak_rss_gb", _peak_rss_gb())
    rec.add_params("line_profiler_available", P._LP is not None)
    rec.finalize(config_snapshot(td.cfg, {"picard_iters": picard_iters}))
    return rec
