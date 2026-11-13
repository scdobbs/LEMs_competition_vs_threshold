#!/bin/bash
# Addendum 9, 5.2: the pilot summary (section 3 estimate, 2.2 stop rule) and the pilot figure.
set -u
lemcvt -v a9 pilot-summary --tiles AP1,GM1
lemcvt figures a9_pilot --out reports/v2/figures_v2
