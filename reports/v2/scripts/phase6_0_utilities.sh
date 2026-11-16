#!/bin/bash
# Addendum 5, Phase 6.0 utilities job (one aggregated job, serc): the test suite on a compute node, the AP1 seam fix
# (2.1: writes AP_1m_best_tile_{fd,area}_seamfix.tif and the manifest entries, originals kept), and the measured inputs
# (2.2 option (iii) window test, 1.2 k_s at the site m with bootstraps). The configs are written afterwards from the
# login node with `lemcvt phase6 write-configs`, once the window test has been read.
set -u
echo "== tests ($(date -Is))"
PYTHONPATH=$GROUP_HOME/opt python -m pytest tests -q -x --no-header -p no:cacheprovider 2>&1 | tail -15
echo "pytest exit=${PIPESTATUS[0]}"
echo "== seamfix AP1 ($(date -Is))"
lemcvt -v seamfix --tile AP1
echo "== phase6 inputs ($(date -Is))"
lemcvt -v phase6 inputs --tiles all
echo "== done ($(date -Is))"
