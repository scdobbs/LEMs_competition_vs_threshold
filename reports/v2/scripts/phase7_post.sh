#!/bin/bash
# Addendum 7: after the slope-area unit fix (2b.5) and the fig3 rewrite (1.3): slope-area at the Phase 6 optima, the two
# figures, the old-against-new record (its slope-area rows read the slope_area run).
set -u
echo "== slope-area ($(date -Is))"; lemcvt -v slope-area --tiles all --phase6 || exit 1
echo "== figures ($(date -Is))"; lemcvt figures slope_area --out reports/v2/figures_v2 && lemcvt figures fig3_deltas --out reports/v2/figures_v2
echo "== record ($(date -Is))"; lemcvt phase6 record --seam-verdict "negligible by the addendum 6 rule"
echo "== done ($(date -Is))"
