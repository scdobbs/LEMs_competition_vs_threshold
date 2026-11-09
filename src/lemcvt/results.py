"""Run directories and provenance.

Every run writes to ``$LEMCVT_RESULTS/<run_id>/``:

* ``params.json``       optimum, MSE, iterations, wall time (per stage)
* ``trajectory.parquet`` every misfit evaluation (``stage`` column)
* ``*.tif``              output grids as GeoTIFF
* ``run.json``           git hash, config snapshot, environment lock hash, host, times

The ``run_id`` is ``<UTC timestamp>_<step>_<label>``; ``lemcvt`` never overwrites an
existing run directory.
"""

from __future__ import annotations

import datetime as _dt
import getpass
import hashlib
import json
import logging
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lemcvt import __version__
from lemcvt.config import REPO_ROOT, results_root
from lemcvt.io.grids import Grid, write_geotiff

logger = logging.getLogger(__name__)


def git_hash(short: bool = False) -> str:
    try:
        args = ["git", "rev-parse", "--short" if short else "HEAD"]
        if short:
            args = ["git", "rev-parse", "--short", "HEAD"]
        out = subprocess.check_output(args, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.run(["git", "diff", "--quiet"], cwd=REPO_ROOT).returncode != 0
        return out + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def slurm_accounting() -> dict[str, Any] | None:
    """Elapsed, MaxRSS, CPU time and partition of the current job (``sstat`` + ``scontrol``), or None.

    Called at ``finalize``, i.e. just before the process exits, so MaxRSS and the CPU time
    of the batch step are those of essentially the whole run. ``sacct`` would only have
    the step totals after the job ends, so the job script keeps the sidecar
    ``slurm_sacct.txt`` for the final values.
    """
    job = os.environ.get("SLURM_JOB_ID")
    if not job:
        return None
    out: dict[str, Any] = {"job_id": job, "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                           "partition": os.environ.get("SLURM_JOB_PARTITION"),
                           "cpus_on_node": os.environ.get("SLURM_CPUS_ON_NODE"),
                           "mem_per_node": os.environ.get("SLURM_MEM_PER_NODE")}
    try:
        txt = subprocess.check_output(["sstat", "-j", f"{job}.batch", "-n", "-P", "--format=MaxRSS,AveCPU,Elapsed"],
                                      text=True, stderr=subprocess.DEVNULL, timeout=30).strip().splitlines()
        if txt:
            maxrss, avecpu, elapsed = (txt[0].split("|") + ["", "", ""])[:3]
            out.update({"max_rss": maxrss, "cpu_time_sstat": avecpu, "elapsed_sstat": elapsed})
    except Exception:  # noqa: BLE001
        out["sstat"] = "unavailable"
    try:
        txt = subprocess.check_output(["scontrol", "show", "job", job], text=True, stderr=subprocess.DEVNULL, timeout=30)
        for key in ("RunTime", "TimeLimit", "Partition", "NumCPUs", "MinMemoryNode", "MinMemoryCPU", "NodeList"):
            m = re.search(rf"\b{key}=(\S+)", txt)
            if m:
                out[key] = m.group(1)
    except Exception:  # noqa: BLE001
        out["scontrol"] = "unavailable"
    return out


def env_lock_sha256() -> str | None:
    lock = REPO_ROOT / "envs" / "sherlock.lock.txt"
    if lock.exists():
        return hashlib.sha256(lock.read_bytes()).hexdigest()
    return None


def _json_default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if is_dataclass(o) and not isinstance(o, type):
        return asdict(o)
    if isinstance(o, tuple | set):
        return list(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def dump_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=1, default=_json_default, sort_keys=False))


def new_run_id(step: str, label: str) -> str:
    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{ts}_{step}_{label}"


class RunRecorder:
    """Collects params, trajectories and grids for one run and writes them."""

    def __init__(self, step: str, label: str, run_id: str | None = None, root: Path | None = None):
        self.step = step
        self.label = label
        self.root = Path(root) if root is not None else results_root()
        if run_id is None:
            # auto ids carry a one-second timestamp; two jobs of the same step and label starting in the
            # same second (the per-stage optimizer jobs did) must not share a directory: add the Slurm job id,
            # then a counter, until the name is free
            base = new_run_id(step, label)
            cand = base
            job = os.environ.get("SLURM_JOB_ID")
            k = 1
            while (self.root / cand).exists() and any((self.root / cand).iterdir()):
                cand = f"{base}_j{job}" if (job and k == 1) else f"{base}_{k}"
                k += 1
            run_id = cand
        self.run_id = run_id
        self.dir = self.root / self.run_id
        if self.dir.exists() and any(self.dir.iterdir()):
            raise FileExistsError(f"run directory already exists and is not empty: {self.dir}")
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t_start = time.time()
        self.params: dict[str, Any] = {}
        self._trajectories: list[Any] = []
        self.grids_written: list[str] = []
        self.log_path = self.dir / "run.log"
        self._fh = logging.FileHandler(self.log_path)
        self._fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger("lemcvt").addHandler(self._fh)
        logger.info("run %s -> %s", self.run_id, self.dir)

    # -- recording ---------------------------------------------------------------
    def add_params(self, key: str, value: Any) -> None:
        self.params[key] = value

    def add_trajectory(self, traj) -> None:
        self._trajectories.append(traj)

    def write_grid(self, grid: Grid, name: str, dtype: str | None = None) -> Path:
        p = write_geotiff(grid, self.dir / f"{name}.tif", dtype=dtype)
        self.grids_written.append(p.name)
        return p

    def write_table(self, df, name: str) -> Path:
        p = self.dir / f"{name}.parquet"
        try:
            df.to_parquet(p, index=False)
        except Exception as exc:  # pyarrow missing -> CSV fallback, recorded
            logger.warning("parquet unavailable (%s); writing CSV instead", exc)
            p = self.dir / f"{name}.csv"
            df.to_csv(p, index=False)
        return p

    # -- finalisation --------------------------------------------------------------
    def finalize(self, config_snapshot: dict[str, Any], extra: dict[str, Any] | None = None,
                 status: str = "completed") -> Path:
        import pandas as pd

        if self._trajectories:
            frames = []
            for t in self._trajectories:
                df = t.to_frame()
                df.insert(0, "stage", t.stage)
                frames.append(df)
            self.write_table(pd.concat(frames, ignore_index=True), "trajectory")
        dump_json(self.params, self.dir / "params.json")
        run = {
            "run_id": self.run_id,
            "step": self.step,
            "label": self.label,
            "status": status,
            "lemcvt_version": __version__,
            "git_hash": git_hash(),
            "env_lock_sha256": env_lock_sha256(),
            "python": sys.version.split()[0],
            "host": platform.node(),
            "user": getpass.getuser(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "started_utc": _dt.datetime.fromtimestamp(self.t_start, _dt.UTC).isoformat(timespec="seconds"),
            "wall_s": time.time() - self.t_start,
            "grids": self.grids_written,
            "slurm_accounting": slurm_accounting(),
            "config": config_snapshot,
        }
        if extra:
            run.update(extra)
        dump_json(run, self.dir / "run.json")
        logging.getLogger("lemcvt").removeHandler(self._fh)
        self._fh.close()
        return self.dir


def find_runs(step: str | None = None, label: str | None = None, root: Path | None = None) -> list[Path]:
    """Completed run directories (have run.json), newest last, filtered by step/label."""
    root = Path(root) if root is not None else results_root()
    if not root.exists():
        return []
    out = []
    for d in sorted(root.iterdir()):
        rj = d / "run.json"
        if not rj.exists():
            continue
        try:
            meta = json.loads(rj.read_text())
        except json.JSONDecodeError:
            continue
        if step and meta.get("step") != step:
            continue
        if label and meta.get("label") != label:
            continue
        out.append(d)
    return out


def latest_run(step: str, label: str, root: Path | None = None) -> Path | None:
    runs = find_runs(step, label, root)
    return runs[-1] if runs else None
