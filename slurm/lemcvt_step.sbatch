#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --array={array}
#SBATCH --output={log_dir}/{job_name}_%A_%a.out
{optional_sbatch}
# Rendered by `lemcvt slurm submit {step}`; one array task per label.
set -u
LABELS=({labels_bash})
LABEL=${{LABELS[$SLURM_ARRAY_TASK_ID]}}
echo "host=$(hostname) job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID label=$LABEL start=$(date -Is)"
{threads}
export PYTHONNOUSERSITE=1
export LEMCVT_DATA="${{LEMCVT_DATA:-{lemcvt_data}}}"
export LEMCVT_RESULTS="${{LEMCVT_RESULTS:-{lemcvt_results}}}"
export MAMBA_ROOT_PREFIX="$GROUP_HOME/micromamba_root"
cd "{repo}" || exit 1
echo "git=$(git rev-parse --short HEAD) env={env_prefix}"
{micromamba} run -p {env_prefix} python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"
{micromamba} run -p {env_prefix} {command} 2>&1 | tee "{log_dir}/$SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID.lemcvt.log"
rc=${{PIPESTATUS[0]}}
RUNDIR=$(grep -oP '^results: \K.*' "{log_dir}/$SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID.lemcvt.log" | tail -1)
rm -f "{log_dir}/$SLURM_JOB_ID.$SLURM_ARRAY_TASK_ID.lemcvt.log"
if [ -n "$RUNDIR" ] && [ -d "$RUNDIR" ]; then
  # Slurm accounting for this task (addendum 2, section 4): partition, elapsed, CPU time, peak memory
  sacct -j "$SLURM_JOB_ID" -P --format=JobID,JobName,Partition,State,Elapsed,TotalCPU,MaxRSS,AllocCPUS,ReqMem,Timelimit > "$RUNDIR/slurm_sacct.txt" 2>/dev/null
  sstat -j "$SLURM_JOB_ID.batch" -P --format=JobID,MaxRSS,AveCPU,Elapsed >> "$RUNDIR/slurm_sacct.txt" 2>/dev/null
fi
echo "exit=$rc end=$(date -Is)"
exit $rc
