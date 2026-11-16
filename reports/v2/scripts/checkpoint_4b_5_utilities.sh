#!/bin/bash
# Addendum 4 utilities job (one aggregated job, serc): tests on a compute node, Phase 5.1 grid conversion,
# the 5.2 D8 check, the Phase 4b d8_vs_dinf rerun (AP in feet, both D-inf partitioned models), and the
# measurement audit rerun (theta grid 0.00-0.90, curvature option iii, hilltop_widths fix).
# Run from the repository root inside the lemcvt environment (lemcvt slurm submit custom --command "bash <this file>").
set -u
echo "== tests ($(date -Is))"
PYTHONPATH=$GROUP_HOME/opt python -m pytest tests -q -x --no-header -p no:cacheprovider 2>&1 | tail -15
echo "pytest exit=${PIPESTATUS[0]}"
echo "== ap-units convert ($(date -Is))"
lemcvt -v ap-units convert --tiles AP
echo "== ap-units d8-check ($(date -Is))"
lemcvt -v ap-units d8-check --tiles AP
echo "== d8-vs-dinf, AP in feet, both partitioned D-inf models ($(date -Is))"
lemcvt -v d8-vs-dinf --tiles AP1,GM1 --units feet --pt-dinf-model both
echo "== audit measure ($(date -Is))"
lemcvt -v audit measure --tiles all
echo "== done ($(date -Is))"
