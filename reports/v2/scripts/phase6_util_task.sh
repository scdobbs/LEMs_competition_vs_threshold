#!/bin/bash
# Phase 6 utilities (addendum 5, 3.6): supp_fig5 regenerated; d8_vs_dinf at the config parameters in metres (both D-inf partitioned models).
set -u
lemcvt -v synthetic pe || exit 1
lemcvt -v d8-vs-dinf --tiles AP1,GM1 --params config --units metres --pt-dinf-model both
