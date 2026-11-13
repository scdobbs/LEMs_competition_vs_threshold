#!/bin/bash
# Addendum 10, after the section 2 arrays: merge every (tile, S_c), the 2.3 summary, then the section 3 morphology score
# with the section 2 surfaces. Each step runs even if an earlier merge fails (a missing part is reported, not fatal).
set -u
for T in GM1 GM2 GM3 GM4 GM5; do
  for SC in 1.25 0.9 central p84; do
    lemcvt -v a10 flux-merge --tile "$T" --sc "$SC" --group a10main || echo "merge failed: $T $SC"
  done
done
lemcvt -v a10 flux-summary --tiles GM
lemcvt -v a9 morphology --tiles all
