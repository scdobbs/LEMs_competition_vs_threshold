#!/bin/bash
#SBATCH --job-name=lemcvt_env
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/users/scdobbs/v2_jobs/lemcvt_env_%j.out
# Build the lemcvt runtime env (envs/sherlock.yml) and write the explicit lockfile.
# Also stage the v1 grids to group storage as GeoTIFFs with .tif extension
# (byte-identical copies; the files are already GeoTIFFs) so that grids can
# leave git (directive Phase 1 requirement 1).
set -u
echo "host=$(hostname) job=$SLURM_JOB_ID start=$(date -Is)"
REPO=$HOME/LEMs_competition_vs_threshold-JGR-branch
MM=$GROUP_HOME/scdobbs-envs/bin/micromamba
export MAMBA_ROOT_PREFIX=$GROUP_HOME/micromamba_root
export CONDA_PKGS_DIRS=$SCRATCH/conda_pkgs
export PIP_CACHE_DIR=$SCRATCH/pip_cache
ENV=$GROUP_HOME/scdobbs-envs/lemcvt
cd "$REPO" || exit 1

echo "=== [1] build env $(date -Is)"
if [ -x "$ENV/bin/python" ]; then
  echo "env exists; updating from spec"
  $MM install -y -p "$ENV" -f envs/sherlock.yml > "$SCRATCH/v2_jobs/lemcvt_env_build.log" 2>&1
else
  $MM create -y -p "$ENV" -f envs/sherlock.yml > "$SCRATCH/v2_jobs/lemcvt_env_build.log" 2>&1
fi
echo "build exit=$?"
$MM env export -p "$ENV" --explicit > envs/sherlock.lock.txt 2>/dev/null
echo "lock lines: $(wc -l < envs/sherlock.lock.txt)"
$MM run -p "$ENV" python -c "import sys,numpy,scipy,rasterio,yaml,pyarrow,click,numba; print('python',sys.version.split()[0],'numpy',numpy.__version__,'scipy',scipy.__version__,'rasterio',rasterio.__version__,'pyarrow',pyarrow.__version__,'numba',numba.__version__)"
$MM run -p "$ENV" python -c "import sksparse.cholmod; print('scikit-sparse OK')" || echo "scikit-sparse import failed (optional)"

echo "=== [2] stage grids to group storage $(date -Is)"
DATA_DST=$GROUP_HOME/lemcvt_data/v1
mkdir -p "$DATA_DST"
n=0
for f in data/*/tile*/* data/*/tile*/outputs/*; do
  [ -f "$f" ] || continue
  rel=${f#data/}
  dst="$DATA_DST/$rel.tif"
  mkdir -p "$(dirname "$dst")"
  if [ ! -f "$dst" ]; then cp "$f" "$dst"; fi
  n=$((n+1))
done
echo "staged $n grid files under $DATA_DST"
# verify byte identity against the Phase 0 manifest (sha256 of file bytes)
python3 - "$DATA_DST" <<'PY'
import csv, hashlib, sys, pathlib
dst = pathlib.Path(sys.argv[1]); bad = 0; n = 0
for r in csv.DictReader(open("reference/v1_grids_manifest.csv")):
    p = dst / (r["path"][len("data/"):] + ".tif")
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    n += 1
    if h != r["sha256"]:
        bad += 1; print("MISMATCH", p)
print(f"verified {n} files, {bad} mismatches")
PY
echo "=== done $(date -Is)"
