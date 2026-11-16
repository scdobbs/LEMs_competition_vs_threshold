#!/bin/bash
set -u
python -m pytest -q tests/test_addendum8.py tests/test_phase6.py 2>&1 | tail -25
