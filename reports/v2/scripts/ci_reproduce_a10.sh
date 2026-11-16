#!/bin/bash
# Addendum 10, section 5.1: reproduce the GitHub workflow (.github/workflows/ci.yml) on Sherlock.
# The CI install is `pip install -e ".[dev]"` on ubuntu-latest, which Sherlock cannot replicate from PyPI (glibc 2.17
# vs manylinux_2_28 wheels), so the environments are built from conda-forge with the versions pip resolved for each
# Python at the time of run 42940255348 (2026-11-11 01:37 UTC; computed from PyPI release dates and requires_python),
# except matplotlib (3.10.9 here, 3.11.2 on CI: conda-forge 3.11 needs glibc >= 2.28 through libraqm),
# WITHOUT numba (an optional extra the workflow does not install). The code is run from a copy of the tracked files
# only (git ls-files), as actions/checkout sees them, with LEMCVT_DATA unset and PYTHONNOUSERSITE=1.
#
# usage: ci_reproduce_a10.sh [build|nobuild] [<git rev>|worktree] [envs...]
set -u
MODE=${1:-build}; SRC=${2:-worktree}; shift 2 2>/dev/null; ENVS=${*:-ci311 ci312}
MM=$GROUP_HOME/scdobbs-envs/bin/micromamba
export MAMBA_ROOT_PREFIX=$GROUP_HOME/micromamba_root
export CONDA_PKGS_DIRS=$SCRATCH/conda_pkgs
export PIP_CACHE_DIR=$SCRATCH/pip_cache
export PYTHONNOUSERSITE=1
unset LEMCVT_DATA LEMCVT_RESULTS LEMCVT_CONFIGS PYTHONPATH   # CI has no TopoAnalysis or pylem on the path
REPO=$HOME/LEMs_competition_vs_threshold-JGR-branch
WORK=${L_SCRATCH_JOB:-$SCRATCH/lemcvt_jobs/ci_a10_tmp}/ci_checkout
rm -rf "$WORK"; mkdir -p "$WORK"
cd "$REPO" || exit 1
if [ "$SRC" = worktree ]; then
  git ls-files -z | tar --null -T - -cf - | tar -xf - -C "$WORK"   # tracked files, current working-tree content
else
  git archive "$SRC" | tar -xf - -C "$WORK"
fi
echo "source=$SRC ($(git rev-parse --short HEAD) HEAD) copied to $WORK"

COMMON="pandas=3.1.1 pandas-stubs=3.1.0.261029 mypy=2.4.1 ruff=0.17.4 pytest=9.2.0 pytest-cov=7.1.0 \
 matplotlib-base pyarrow=26.0.1 pyyaml=6.0.3 click=8.5.1 types-pyyaml=6.0.12.20261103 pip setuptools wheel"
declare -A SPEC
SPEC[ci311]="python=3.11 numpy=2.4.6 scipy=1.17.1 rasterio=1.4.4 $COMMON"
SPEC[ci312]="python=3.12 numpy=2.5.4 scipy=1.18.1 rasterio=1.5.2 $COMMON"

for E in $ENVS; do
  P=$GROUP_HOME/scdobbs-envs/$E
  echo "===================== env $E ($P) $(date -Is)"
  if [ "$MODE" = build ] && [ -n "${SPEC[$E]:-}" ]; then
    $MM install -y -p "$P" -c conda-forge --override-channels ${SPEC[$E]} > "$SCRATCH/lemcvt_jobs/ci_a10_build_$E.log" 2>&1
    echo "build exit=$?"; tail -3 "$SCRATCH/lemcvt_jobs/ci_a10_build_$E.log"
    $MM remove -y -p "$P" numba > /dev/null 2>&1
  fi
  PY=$P/bin/python
  export PATH=$P/bin:/usr/bin:/bin
  (cd "$WORK" && $PY -m pip install --no-deps --no-build-isolation -e . > "$SCRATCH/lemcvt_jobs/ci_a10_pipe_$E.log" 2>&1; echo "pip -e exit=$?")
  $PY - <<'EOF'
import importlib, sys
print("python", sys.version.split()[0])
for m in ["numpy", "scipy", "pandas", "matplotlib", "rasterio", "pyarrow", "mypy", "pytest", "numba"]:
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print(m, "NOT INSTALLED", type(e).__name__)
from importlib.metadata import version
for d in ["pandas-stubs", "types-PyYAML", "ruff"]:
    try: print(d, version(d))
    except Exception: print(d, "NOT INSTALLED")
import lemcvt; print("lemcvt from", lemcvt.__file__)
EOF
  cd "$WORK"
  echo "--- step: ruff";   ruff check src tests 2>&1 | tail -30; echo "ruff exit=${PIPESTATUS[0]}"
  echo "--- step: mypy";   mypy src/lemcvt 2>&1 | tail -60; echo "mypy exit=${PIPESTATUS[0]}"
  echo "--- step: pytest"; pytest -q -m "not fulltile" -p no:cacheprovider -rfE 2>&1 | tail -80; echo "pytest exit=${PIPESTATUS[0]}"
  cd "$REPO"
done
echo "===== done $(date -Is)"
