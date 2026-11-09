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
{micromamba} run -p {env_prefix} {command}
rc=$?
echo "exit=$rc end=$(date -Is)"
exit $rc
