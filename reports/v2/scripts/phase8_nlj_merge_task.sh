#!/bin/bash
# Addendum 8, 2.4: merge the parts of a tile's nonlinear joint profile at its fitted S_c. Label TILE-SC.
set -u
IFS=- read -r TILE SC <<< "$1"
lemcvt -v tile v2-nonlinear-joint-merge --tile "$TILE" --s-c "$SC"
