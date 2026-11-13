#!/bin/bash
set -u
echo "--- new tests"; python -m pytest -q tests/test_addendum8.py tests/test_phase7_width.py tests/test_phase7_heads.py 2>&1 | tail -30
echo "--- full"; bash reports/v2/scripts/ci_steps_lemcvt_env.sh
