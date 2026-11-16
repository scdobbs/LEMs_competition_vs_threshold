"""Render and submit Slurm job arrays for lemcvt steps.

``lemcvt slurm submit <step> --tiles all`` renders ``slurm/lemcvt_step.sbatch``
from ``configs/slurm/sherlock.yaml``, prints the resource estimate and submits.
Submissions above the core-hour limit stop unless ``--approved`` is passed (addendum 2
raised the limit to 300 core-hours and fixed the partition to ``serc``). Every job
requests an explicit time limit; every array is throttled with ``%N``
(``array_throttle`` in the config) so that concurrent CPUs stay within
``max_concurrent_cpus``. ``submit_stages`` submits the optimizer stages of a tile as
independent jobs (no stage initialises from another stage's output; see
``lemcvt.analyses.tile_pipeline``), each with its own time limit of 1.5 x the v1 stage
time rounded up to the hour.
"""

from __future__ import annotations

import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lemcvt.config import REPO_ROOT, load_slurm_config

TEMPLATE = REPO_ROOT / "slurm" / "lemcvt_step.sbatch"
CORE_HOUR_LIMIT = 300.0
STAGE_TIME_FACTOR = 1.5
SERC_MAX_HOURS = 7 * 24


def _expand(s: str) -> str:
    return os.path.expandvars(str(s))


def _hours(timestr: str) -> float:
    """Slurm time string -> hours. Accepts D-HH:MM:SS, HH:MM:SS, MM:SS, MM."""
    s = timestr.strip()
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(p) for p in s.split(":")]
    if len(parts) == 3:
        h, m, sec = parts
    elif len(parts) == 2:
        h, m, sec = 0, parts[0], parts[1]
    else:
        h, m, sec = 0, parts[0], 0
    return days * 24 + h + m / 60 + sec / 3600


@dataclass
class JobSpec:
    step: str
    labels: list[str]
    command_template: str  # with {label}
    time: str
    cpus: int
    mem: str
    partition: str | None
    qos: str | None
    account: str | None
    job_name: str
    log_dir: Path
    micromamba: str
    env_prefix: str
    threads_env: list[str]
    throttle: int = 10
    dependency: str | None = None     # e.g. "afterok:51779145"

    @property
    def core_hours(self) -> float:
        return len(self.labels) * self.cpus * _hours(self.time)

    @property
    def concurrent_cpus(self) -> int:
        return min(len(self.labels), self.throttle) * self.cpus


def build_spec(step: str, labels: list[str], command_template: str, time: str | None = None,
               cpus: int | None = None, mem: str | None = None, job_name: str | None = None,
               dependency: str | None = None) -> JobSpec:
    cfg = load_slurm_config()
    stepcfg = dict(cfg["defaults"])
    stepcfg.update(cfg["steps"].get(step, {}))
    if step == "custom":
        step = re.sub(r"[^A-Za-z0-9_]+", "_", (command_template.split()[2] if len(command_template.split()) > 2 else "custom"))[:24]
    spec = JobSpec(
        step=step, labels=list(labels), command_template=command_template,
        time=time or str(stepcfg["time"]), cpus=int(cpus or stepcfg["cpus"]), mem=str(mem or stepcfg["mem"]),
        partition=cfg.get("partition"), qos=cfg.get("qos"), account=cfg.get("account"),
        job_name=job_name or f"lemcvt_{step}", log_dir=Path(_expand(cfg.get("log_dir", "$SCRATCH/lemcvt_jobs"))),
        micromamba=_expand(cfg["micromamba"]), env_prefix=_expand(cfg["env_prefix"]),
        threads_env=list(cfg.get("threads_env", ["OMP_NUM_THREADS"])),
        throttle=int(stepcfg.get("throttle", cfg.get("array_throttle", 10))), dependency=dependency,
    )
    max_cpus = int(cfg.get("max_concurrent_cpus", 300))
    if spec.concurrent_cpus > max_cpus:
        spec.throttle = max(1, max_cpus // spec.cpus)
    return spec


class SubmitEnvironmentError(RuntimeError):
    """Addendum 7, 0.4: the rendered script carries the submitting shell's LEMCVT_DATA and LEMCVT_RESULTS."""


def require_submit_env() -> tuple[str, str]:
    """The data and results roots of the submitting shell; refuse to render when either is unset or empty.

    `lemcvt slurm submit` writes ``export LEMCVT_RESULTS="${LEMCVT_RESULTS:-<value>}"`` into the job script, so an
    empty value in the submitting shell makes every task fall back to the repository as its results root and find no
    runs (submission fault 3 of the addendum 6 session)."""
    data, results = os.environ.get("LEMCVT_DATA", ""), os.environ.get("LEMCVT_RESULTS", "")
    missing = [k for k, v in (("LEMCVT_DATA", data), ("LEMCVT_RESULTS", results)) if not v.strip()]
    if missing:
        raise SubmitEnvironmentError(f"{' and '.join(missing)} must be exported (non-empty) in the submitting shell before a Slurm script is "
                                     "rendered (addendum 7, 0.4); e.g. export LEMCVT_DATA=$GROUP_HOME/lemcvt_data/v1 LEMCVT_RESULTS=$GROUP_HOME/lemcvt_results")
    return data, results


def render(spec: JobSpec) -> str:
    lemcvt_data, lemcvt_results = require_submit_env()
    tmpl = TEMPLATE.read_text()
    optional = []
    if spec.partition:
        optional.append(f"#SBATCH --partition={spec.partition}")
    if spec.qos:
        optional.append(f"#SBATCH --qos={spec.qos}")
    if spec.account:
        optional.append(f"#SBATCH --account={spec.account}")
    if spec.dependency:
        optional.append(f"#SBATCH --dependency={spec.dependency}")
    labels_bash = " ".join(spec.labels)
    threads = "\n".join(f"export {v}=$SLURM_CPUS_PER_TASK" for v in spec.threads_env)
    return tmpl.format(
        job_name=spec.job_name, time=spec.time, cpus=spec.cpus, mem=spec.mem,
        array=f"0-{len(spec.labels) - 1}%{spec.throttle}", optional_sbatch="\n".join(optional),
        log_dir=str(spec.log_dir), labels_bash=labels_bash, threads=threads,
        micromamba=spec.micromamba, env_prefix=spec.env_prefix, repo=str(REPO_ROOT),
        command=spec.command_template, step=spec.step,
        lemcvt_data=lemcvt_data, lemcvt_results=lemcvt_results,
    )


def estimate_text(spec: JobSpec) -> str:
    return (f"step={spec.step} tasks={len(spec.labels)} cpus={spec.cpus} time={spec.time} mem={spec.mem} "
            f"partition={spec.partition or '(default)'} throttle=%{spec.throttle} "
            f"(<= {spec.concurrent_cpus} concurrent CPUs) -> estimate {spec.core_hours:.1f} core-hours "
            f"(limit without approval: {CORE_HOUR_LIMIT:.0f})")


def stage_time_limit(duration_s: float, factor: float = STAGE_TIME_FACTOR) -> str:
    """``factor`` x a v1 stage duration, rounded up to the hour, at least 1 h, below the serc maximum."""
    hours = max(1, math.ceil(factor * duration_s / 3600.0))
    if hours >= SERC_MAX_HOURS:
        raise ValueError(f"time limit {hours} h is not below the serc maximum of {SERC_MAX_HOURS} h")
    days, h = divmod(hours, 24)
    return f"{days}-{h:02d}:00:00" if days else f"{h:02d}:00:00"


def submit_stages(tiles: list[str], stages: tuple[str, ...] = ("ad", "acrit", "nonlinear", "joint"),
                  optimizer: str = "v1", dry_run: bool = False, approved: bool = False,
                  factor: float = STAGE_TIME_FACTOR, extra_args: str = "--no-derived") -> list[dict[str, Any]]:
    """One job per (tile, stage) for the optimizer regression (addendum 2, section 3).

    The four v1 stages are independent (each starts from the config's K0, D0, A_crit_0),
    so no ``--dependency`` chain is needed; each job writes its own ``params.json`` and
    trajectory, and a failed stage is rerun alone. The time limit per job is
    ``factor`` x the tile's v1 stage time from ``v1_reference.stages.<stage>.duration_s``.
    The total core-hour estimate is printed; above ``CORE_HOUR_LIMIT`` the submission
    stops unless ``approved``.
    """
    from lemcvt.config import load_tile_config

    specs = []
    for tile in tiles:
        cfg = load_tile_config(tile)
        for stage in stages:
            ref = cfg.v1_reference.get("stages", {}).get(stage)
            if not ref or "duration_s" not in ref:
                raise ValueError(f"{tile}: no v1 duration for stage {stage!r} in the tile config")
            t = stage_time_limit(float(ref["duration_s"]), factor)
            cmd = f"lemcvt -v tile run --tile $LABEL --stages {stage} --optimizer {optimizer} {extra_args}".strip()
            spec = build_spec("tile_run", [tile], cmd, time=t, job_name=f"lemcvt_tile_run_{tile}_{stage}")
            specs.append((tile, stage, float(ref["duration_s"]), spec))
    total = sum(sp.core_hours for *_, sp in specs)
    concurrent = sum(sp.cpus for *_, sp in specs)
    print(f"optimizer regression: {len(specs)} jobs, {concurrent} concurrent CPUs, {total:.1f} core-hours total "
          f"(limit without approval: {CORE_HOUR_LIMIT:.0f})")
    for tile, stage, d, sp in specs:
        print(f"  {tile} {stage:9s} v1 {d / 3600:5.2f} h -> limit {sp.time} cpus={sp.cpus} mem={sp.mem} "
              f"({sp.core_hours:.1f} core-h)")
    if total > CORE_HOUR_LIMIT and not approved:
        raise PermissionError(f"{total:.1f} core-hours exceeds {CORE_HOUR_LIMIT:.0f}; ask before submitting")
    out = []
    for tile, stage, d, sp in specs:
        job, script = submit(sp, dry_run=dry_run, approved=True)
        out.append({"tile": tile, "stage": stage, "v1_duration_s": d, "time": sp.time, "cpus": sp.cpus, "mem": sp.mem,
                    "core_hours": sp.core_hours, "job_id": job, "script": str(script)})
    return out


def submit(spec: JobSpec, dry_run: bool = False, approved: bool = False) -> tuple[str, Path]:
    """Write the rendered script and submit it with sbatch. Returns (job id or 'DRY', script path)."""
    if spec.core_hours > CORE_HOUR_LIMIT and not approved:
        raise PermissionError(
            f"{estimate_text(spec)}\nThis exceeds {CORE_HOUR_LIMIT:.0f} core-hours; the directive requires "
            "Steve's approval before submission. Re-run with --approved once he has agreed."
        )
    spec.log_dir.mkdir(parents=True, exist_ok=True)
    script = spec.log_dir / f"{spec.job_name}_{len(spec.labels)}tasks.sbatch"
    k = 1
    while script.exists():
        k += 1
        script = spec.log_dir / f"{spec.job_name}_{len(spec.labels)}tasks_{k}.sbatch"
    script.write_text(render(spec))
    if dry_run:
        return "DRY", script
    out = subprocess.check_output(["sbatch", str(script)], text=True)
    m = re.search(r"(\d+)", out)
    return (m.group(1) if m else out.strip()), script
