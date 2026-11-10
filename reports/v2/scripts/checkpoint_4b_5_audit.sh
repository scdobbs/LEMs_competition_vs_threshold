#!/bin/bash
# Addendum 4: measurement audit rerun (L_f averaged over eight synthetic realisations; theta grid 0.00-0.90;
# curvature option iii) and the measure-diagnostics rerun with the hilltop_widths fix (6.4.1), one aggregated job.
set -u
echo "== audit measure ($(date -Is))"
lemcvt -v audit measure --tiles all
echo "== measure-diagnostics ($(date -Is))"
lemcvt -v measure-diagnostics --tiles all
echo "== done ($(date -Is))"
