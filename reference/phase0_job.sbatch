#!/bin/bash
#SBATCH --job-name=v2_phase0
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=/scratch/users/scdobbs/v2_jobs/phase0_%j.out
#
# Phase 0 (refactor_v2_directive.md): grid manifest + saved-grid MSE (step 4),
# rebuild the v1 environment as lem-v1 and run the v1 test suite in it (step 5).
# Default partition (normal); ~15-40 min expected. Submit from the repo root:
#   sbatch reference/phase0_job.sbatch
set -u
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
REPO=$HOME/LEMs_competition_vs_threshold-JGR-branch
LOG=$REPO/reference/logs
mkdir -p "$LOG"
MM=$GROUP_HOME/scdobbs-envs/bin/micromamba
export MAMBA_ROOT_PREFIX=$GROUP_HOME/micromamba_root
export CONDA_PKGS_DIRS=$SCRATCH/conda_pkgs
export PIP_CACHE_DIR=$SCRATCH/pip_cache
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
ENV=$GROUP_HOME/scdobbs-envs/lem-v1
cd "$REPO" || exit 1
echo "git: $(git rev-parse --short HEAD) $(git rev-parse --abbrev-ref HEAD)"

echo "=== [1] grid manifest + saved-grid MSE (lem-widths python) $(date -Is)"
$GROUP_HOME/scdobbs-envs/lem-widths/bin/python reference/make_grid_manifest.py > "$LOG/manifest.log" 2>&1
echo "manifest exit=$? ; $(tail -n 2 "$LOG/manifest.log" | tr '\n' ' ')"

echo "=== [2] build lem-v1 $(date -Is)"
if [ -x "$ENV/bin/python" ]; then
  echo "lem-v1 already exists at $ENV; skipping build"
  BUILD_SPEC=existing
else
  $MM create -y -p "$ENV" -f envs/lem-v1.yml > "$LOG/lem-v1_build.log" 2>&1
  rc=$?
  BUILD_SPEC=strict
  if [ $rc -ne 0 ]; then
    echo "strict solve failed (rc=$rc); trying envs/lem-v1-relaxed.yml"
    rm -rf "$ENV"
    { echo; echo "===== relaxed spec ====="; } >> "$LOG/lem-v1_build.log"
    $MM create -y -p "$ENV" -f envs/lem-v1-relaxed.yml >> "$LOG/lem-v1_build.log" 2>&1
    rc=$?
    BUILD_SPEC=relaxed
  fi
  echo "build exit=$rc spec=$BUILD_SPEC"
fi
echo "build_spec=$BUILD_SPEC" > "$LOG/lem-v1_build_spec.txt"
$MM list -p "$ENV" > "$LOG/lem-v1_packages.txt" 2>&1
$MM run -p "$ENV" python -c "import sys,numpy,scipy; print('python',sys.version.split()[0],'numpy',numpy.__version__,'scipy',scipy.__version__)" 2>&1 | tee -a "$LOG/lem-v1_build_spec.txt"

echo "=== [3] pytest tests/ in lem-v1 $(date -Is)"
PYTHONPATH=$GROUP_HOME/opt $MM run -p "$ENV" python -m pytest tests/ -q -p no:cacheprovider > "$LOG/pytest_lem-v1.log" 2>&1
echo "pytest exit=$? ; $(tail -n 1 "$LOG/pytest_lem-v1.log")"

echo "=== [4] TopoAnalysis import + one grid load in lem-v1 $(date -Is)"
PYTHONPATH=$GROUP_HOME/opt $MM run -p "$ENV" python - > "$LOG/topoanalysis_check_lem-v1.log" 2>&1 <<'PY'
import numpy as np
from TopoAnalysis import dem as d
e = d.Elevation.load("data/GM/tile1/GM_1m_best_tile_filled")
print("TopoAnalysis Elevation.load OK:", e._griddata.shape, e._griddata.dtype,
      "dx=", e._georef_info.dx, "nx,ny=", e._georef_info.nx, e._georef_info.ny,
      "geoTransform=", e._georef_info.geoTransform, "nodata=", getattr(e, "_nodata_value", "n/a"))
a = d.Area.load("data/AP/tile1/AP_1m_best_tile_area")
print("AP1 area: dx=", a._georef_info.dx, "geoTransform=", a._georef_info.geoTransform,
      "min/max=", np.nanmin(a._griddata), np.nanmax(a._griddata))
PY
echo "topoanalysis exit=$? ; $(cat "$LOG/topoanalysis_check_lem-v1.log" | tail -n 3)"

echo "=== done $(date -Is)"
