#!/bin/bash
# Addendum 8, 3.2: one (tile, alpha) per array task; label = <tile>_a<alpha>, e.g. AP1_a0.5
set -u
LABEL=$1
TILE=${LABEL%%_a*}
ALPHA=${LABEL##*_a}
lemcvt -v phase8 slope-threshold --tile "$TILE" --alpha "$ALPHA"
