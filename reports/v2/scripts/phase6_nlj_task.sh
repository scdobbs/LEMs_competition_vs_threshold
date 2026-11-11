#!/bin/bash
# One part of a nonlinear joint profile (addendum 5, 3.4): label TILE-K-N-KF, e.g. AP1-0-3-1.0 = tile AP1, part 0 of 3, K factor 1.0.
set -u
IFS=- read -r TILE K N KF <<< "$1"
lemcvt -v tile v2-nonlinear-joint --tile "$TILE" --k-factor "$KF" --split "$K/$N"
