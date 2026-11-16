#!/bin/bash
# Phase 7 (addendum 8, 1.4): the ported analyses on the metre grids, then the regression check against the branches.
set -u
lemcvt -v phase7 width --sites AP,GM --mode full || exit 1
lemcvt -v phase7 width-constrained --stage full || exit 1
lemcvt -v phase7 heads --tiles all || exit 1
lemcvt -v phase7 port-check
