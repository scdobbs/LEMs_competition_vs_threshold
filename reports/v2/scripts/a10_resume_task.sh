#!/bin/bash
# Addendum 10, section 2: run the A_crit rows a part did not finish within its time limit. Label TILE-SC-K-N-GROUP.
set -u
IFS=- read -r TILE SC K N GROUP <<< "$1"
lemcvt -v a10 flux-resume --tile "$TILE" --sc "$SC" --split "$K/$N" --group "$GROUP"
