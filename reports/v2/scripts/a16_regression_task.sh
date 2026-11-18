#!/bin/bash
# Addendum 16, section 5.4: forward runs of the AD and partitioned models on one tile at the v2.0.0 optima, with the derived
# grids (fill, D8 under the new border guard, area), compared against the v2.0.0 release runs (tables/table1.csv run_ad, run_partitioned).
# usage: a16_regression_task.sh <tile>   (GM1 or AP1)
set -u
T=$1
case $T in
  GM1) AD=20261111T012445Z_tile_run_GM1_v2_ad; PT=20261111T050111Z_tile_run_GM1_v2_partitioned-refined;;
  AP1) AD=20261111T012439Z_tile_run_AP1_v2_ad; PT=20261111T050726Z_tile_run_AP1_v2_partitioned-refined;;
  *) echo "unknown tile $T"; exit 1;;
esac
R=$LEMCVT_RESULTS
for REF in $AD $PT; do
  echo "== $T forward at the optima of $REF $(date -Is)"
  lemcvt -v tile forward --tile $T --params $R/$REF/params.json --reference-run $R/$REF --derived --run-id "${A16_RUN_ID}_reg_${T}_$(echo $REF | sed 's/.*_v2_//')"; echo "forward exit=$?"
done
