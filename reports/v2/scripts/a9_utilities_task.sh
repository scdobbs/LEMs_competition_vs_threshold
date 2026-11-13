#!/bin/bash
# Addendum 9: lint, types and the fast test suite (LEMCVT_DATA unset for the tests), then the slope-exceedance diagnostic.
set -u
ruff check src tests && mypy src/lemcvt
env -u LEMCVT_DATA python -m pytest -q -m "not fulltile" tests
lemcvt -v a9 slope-exceedance --tiles all
