#!/bin/bash
# Addendum 8, 2.4: one part of the nonlinear joint profile at the tile's fitted S_c. Label TILE-K-N-SC, e.g. AP1-0-6-0.3061.
# D grid: the addendum 6 grid extended down to 1e-3 (22 log-spaced values 1e-3 .. 5e-2, step x1.205); Picard cap 100 (config);
# a part stops when more than 30 % of its evaluations hit the cap (pre-registered).
set -u
IFS=- read -r TILE K N SC <<< "$1"
lemcvt -v tile v2-nonlinear-joint --tile "$TILE" --split "$K/$N" --s-c "$SC" --n-d 22 --d-range 1e-3,5e-2 --cap-stop-frac 0.3 --no-grids
