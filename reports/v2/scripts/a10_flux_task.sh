#!/bin/bash
# Addendum 10, section 2: one flux-form part. Label TILE-SC-K-N-GROUP[-NACRIT], e.g. GM1-1.25-0-2-a10A (the k-th of n
# interleaved A_crit subsets), or TILE-SC-pilot-GROUP for the 2.2.1 pilot row.
set -u
IFS=- read -r TILE SC K N GROUP NACRIT <<< "$1"
if [ "$K" = "pilot" ]; then
  lemcvt -v a10 flux-part --tile "$TILE" --sc "$SC" --pilot --group "$N"
else
  lemcvt -v a10 flux-part --tile "$TILE" --sc "$SC" --split "$K/$N" --group "$GROUP" --n-acrit "${NACRIT:-12}"
fi
