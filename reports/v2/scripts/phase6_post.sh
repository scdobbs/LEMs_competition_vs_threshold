#!/bin/bash
# Addendum 6: post-processing once every Phase 6 run exists (chained with afterok): curvature comparison (4), slope-area
# at the new optima (5.6), figures and tables from results, the old-against-new record (5.8).
set -u
echo "== curvature compare ($(date -Is))"; lemcvt -v phase6 curvature-compare --tiles all
echo "== slope-area ($(date -Is))"; lemcvt -v slope-area --tiles all --phase6
echo "== figures ($(date -Is))"; lemcvt figures all --out reports/v2/figures_v2
echo "== tables ($(date -Is))"; lemcvt tables all; lemcvt phase6 record
echo "== done ($(date -Is))"
