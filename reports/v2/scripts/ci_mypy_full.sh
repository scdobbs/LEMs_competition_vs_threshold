#!/bin/bash
set -u
unset LEMCVT_DATA LEMCVT_RESULTS LEMCVT_CONFIGS
cd "$HOME/LEMs_competition_vs_threshold-JGR-branch" || exit 1
mypy src/lemcvt > "$SCRATCH/lemcvt_jobs/mypy_full.txt" 2>&1; echo "mypy exit=$?"
pytest -q -m "not fulltile" -p no:cacheprovider -rf > "$SCRATCH/lemcvt_jobs/pytest_nodata.txt" 2>&1; echo "pytest exit=$?"
