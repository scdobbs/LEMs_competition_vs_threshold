#!/bin/bash
# Addendum 8, 2: E*-R* on all ten tiles, then the figure.
set -u
lemcvt -v phase8 er-star --tiles all || exit 1
lemcvt figures a8_er_star --out reports/v2/figures_v2
