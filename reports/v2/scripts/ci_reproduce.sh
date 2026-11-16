#!/bin/bash
# Addendum 6, section 2.1: reproduce the GitHub workflow's steps on Sherlock in fresh Python 3.11 and 3.12
# environments (micromamba: python + pip only, then `pip install -e ".[dev]"` from PyPI as the workflow does),
# with LEMCVT_DATA unset so that a test needing real data fails here as it would on GitHub.
set -u
MM=$GROUP_HOME/scdobbs-envs/bin/micromamba
export MAMBA_ROOT_PREFIX=$GROUP_HOME/micromamba_root
export CONDA_PKGS_DIRS=$SCRATCH/conda_pkgs
export PIP_CACHE_DIR=$SCRATCH/pip_cache
export PYTHONNOUSERSITE=1
unset LEMCVT_DATA LEMCVT_RESULTS LEMCVT_CONFIGS
cd "$HOME/LEMs_competition_vs_threshold-JGR-branch" || exit 1
for PY in 3.11 3.12; do
  ENV=$GROUP_HOME/scdobbs-envs/ci${PY/./}
  echo "===== python $PY env $ENV $(date -Is)"
  if [ ! -x "$ENV/bin/python" ]; then
    $MM create -y -p "$ENV" -c conda-forge python=$PY pip > "$SCRATCH/lemcvt_jobs/ci_env_${PY}.log" 2>&1
    echo "create exit=$?"
  fi
  $MM run -p "$ENV" python --version
  echo "--- step: install"; $MM run -p "$ENV" python -m pip install --upgrade pip > "$SCRATCH/lemcvt_jobs/ci_pip_${PY}.log" 2>&1
  $MM run -p "$ENV" python -m pip install -e ".[dev]" >> "$SCRATCH/lemcvt_jobs/ci_pip_${PY}.log" 2>&1; echo "install exit=$?"; tail -3 "$SCRATCH/lemcvt_jobs/ci_pip_${PY}.log"
  $MM run -p "$ENV" python -c "import numpy, scipy, rasterio, ruff, mypy; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'rasterio', rasterio.__version__)" 2>&1 | tail -2
  $MM run -p "$ENV" ruff --version; $MM run -p "$ENV" mypy --version
  echo "--- step: ruff"; $MM run -p "$ENV" ruff check src tests --output-format concise 2>&1 | tail -15; echo "ruff exit=${PIPESTATUS[0]}"
  echo "--- step: mypy"; $MM run -p "$ENV" mypy src/lemcvt 2>&1 | tail -25; echo "mypy exit=${PIPESTATUS[0]}"
  echo "--- step: pytest"; $MM run -p "$ENV" pytest -q -m "not fulltile" -p no:cacheprovider 2>&1 | tail -15; echo "pytest exit=${PIPESTATUS[0]}"
done
echo "===== done $(date -Is)"
