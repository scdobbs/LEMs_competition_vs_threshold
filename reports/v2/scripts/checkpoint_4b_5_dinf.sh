#!/bin/bash
# Addendum 4: the 5.2 D8 check (rerun after the neighbour-offset fix) and the Phase 4b d8_vs_dinf rerun
# (AP in feet, rebuilt and v1 partitioned D-inf models; the v1 D-inf AD model's LIL assembly needs ~34 GB).
set -u
echo "== ap-units d8-check ($(date -Is))"
lemcvt -v ap-units d8-check --tiles AP
echo "== d8-vs-dinf, AP in feet, both partitioned D-inf models ($(date -Is))"
lemcvt -v d8-vs-dinf --tiles AP1,GM1 --units feet --pt-dinf-model both
echo "== done ($(date -Is))"
