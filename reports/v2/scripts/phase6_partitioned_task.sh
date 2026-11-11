#!/bin/bash
# One Phase 6 partitioned task (addendum 5, 3.2; addendum 6, 5.2): thin profile, staircase refinement, every unique
# area inside the 5 % intervals (stage 2 at D0_A and D0_B, joint). Usage: bash phase6_partitioned_task.sh <TILE> <A|B>
set -u
TILE=$1; VAR=${2:-A}
lemcvt -v tile run --tile "$TILE" --stages partitioned --optimizer v2 --acrit-grid thin --d0-variant "$VAR" || exit 1
lemcvt -v tile v2-refine --tile "$TILE" --fill-interval 0.05 --d0-variant "$VAR"
