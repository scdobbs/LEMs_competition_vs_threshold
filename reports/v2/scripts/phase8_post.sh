#!/bin/bash
# Addendum 8: after the slope-threshold arrays: the summary, the figure.
set -u
lemcvt -v phase8 slope-threshold-summary --tiles all || exit 1
lemcvt figures a8_slope_threshold --out reports/v2/figures_v2
