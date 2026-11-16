#!/bin/bash
# Merge the split parts of a nonlinear joint profile (addendum 6): label TILE-KF, e.g. AP1-1.0 = tile AP1, K factor 1.0.
set -u
IFS=- read -r TILE KF <<< "$1"
lemcvt -v tile v2-nonlinear-joint-merge --tile "$TILE" --k-factor "$KF"
