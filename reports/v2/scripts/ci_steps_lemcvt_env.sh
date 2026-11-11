#!/bin/bash
# Addendum 6, 2.1: the workflow's lint/type/test steps in the project environment (the same ruff, mypy and pytest), with
# LEMCVT_DATA unset so that any test needing real data fails as it would on GitHub.
set -u
unset LEMCVT_DATA LEMCVT_RESULTS LEMCVT_CONFIGS
cd "$HOME/LEMs_competition_vs_threshold-JGR-branch" || exit 1
echo "--- ruff"; ruff --version; ruff check src tests; echo "ruff exit=$?"
echo "--- mypy"; mypy --version; mypy src/lemcvt 2>&1 | tail -40; echo "mypy exit=${PIPESTATUS[0]}"
echo "--- pytest (no data)"; pytest -q -m "not fulltile" -p no:cacheprovider 2>&1 | tail -8; echo "pytest exit=${PIPESTATUS[0]}"
