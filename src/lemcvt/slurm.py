"""Render and submit Slurm job arrays for lemcvt steps.

``lemcvt slurm submit <step> --tiles all`` renders ``slurm/lemcvt_step.sbatch``
from ``configs/slurm/sherlock.yaml``, prints the resource estimate and submits.
Submissions above 200 core-hours stop unless ``--approved`` is passed (the directive
requires Steve's approval first). Every job requests an explicit time limit; the
partition is omitted when the config says ``null`` so the scheduler default applies.
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from lemcvt.config import REPO_ROOT, load_slurm_config

TEMPLATE = REPO_ROOT / "slurm" / "lemcvt_step.sbatch"
CORE_HOUR_LIMIT = 200.0


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

    @property
    def core_hours(self) -> float:
        return len(self.labels) * self.cpus * _hours(self.time)


def build_spec(step: str, labels: list[str], command_template: str, time: str | None = None,
               cpus: int | None = None, mem: str | None = None) -> JobSpec:
    cfg = load_slurm_config()
    stepcfg = dict(cfg["defaults"])
    stepcfg.update(cfg["steps"].get(step, {}))
    return JobSpec(
        step=step, labels=list(labels), command_template=command_template,
        time=time or str(stepcfg["time"]), cpus=int(cpus or stepcfg["cpus"]), mem=str(mem or stepcfg["mem"]),
        partition=cfg.get("partition"), qos=cfg.get("qos"), account=cfg.get("account"),
        job_name=f"lemcvt_{step}", log_dir=Path(_expand(cfg.get("log_dir", "$SCRATCH/lemcvt_jobs"))),
        micromamba=_expand(cfg["micromamba"]), env_prefix=_expand(cfg["env_prefix"]),
        threads_env=list(cfg.get("threads_env", ["OMP_NUM_THREADS"])),
    )


def render(spec: JobSpec) -> str:
    tmpl = TEMPLATE.read_text()
    optional = []
    if spec.partition:
        optional.append(f"#SBATCH --partition={spec.partition}")
    if spec.qos:
        optional.append(f"#SBATCH --qos={spec.qos}")
    if spec.account:
        optional.append(f"#SBATCH --account={spec.account}")
    labels_bash = " ".join(spec.labels)
    threads = "\n".join(f"export {v}=$SLURM_CPUS_PER_TASK" for v in spec.threads_env)
    return tmpl.format(
        job_name=spec.job_name, time=spec.time, cpus=spec.cpus, mem=spec.mem,
        array=f"0-{len(spec.labels) - 1}", optional_sbatch="\n".join(optional),
        log_dir=str(spec.log_dir), labels_bash=labels_bash, threads=threads,
        micromamba=spec.micromamba, env_prefix=spec.env_prefix, repo=str(REPO_ROOT),
        command=spec.command_template, step=spec.step,
        lemcvt_data=os.environ.get("LEMCVT_DATA", ""), lemcvt_results=os.environ.get("LEMCVT_RESULTS", ""),
    )


def estimate_text(spec: JobSpec) -> str:
    return (f"step={spec.step} tasks={len(spec.labels)} cpus={spec.cpus} time={spec.time} mem={spec.mem} "
            f"partition={spec.partition or '(default)'} -> estimate {spec.core_hours:.1f} core-hours "
            f"(limit without approval: {CORE_HOUR_LIMIT:.0f})")


def submit(spec: JobSpec, dry_run: bool = False, approved: bool = False) -> tuple[str, Path]:
    """Write the rendered script and submit it with sbatch. Returns (job id or 'DRY', script path)."""
    if spec.core_hours > CORE_HOUR_LIMIT and not approved:
        raise PermissionError(
            f"{estimate_text(spec)}\nThis exceeds {CORE_HOUR_LIMIT:.0f} core-hours; the directive requires "
            "Steve's approval before submission. Re-run with --approved once he has agreed."
        )
    spec.log_dir.mkdir(parents=True, exist_ok=True)
    script = spec.log_dir / f"{spec.job_name}_{len(spec.labels)}tasks.sbatch"
    script.write_text(render(spec))
    if dry_run:
        return "DRY", script
    out = subprocess.check_output(["sbatch", str(script)], text=True)
    m = re.search(r"(\d+)", out)
    return (m.group(1) if m else out.strip()), script
