#!/bin/bash
# Addendum 9, 5.1: one pilot task. Label TILE-PART, e.g. AP1-centre (2.2 tests 1 and 4) or AP1-flux (flux-form solves).
set -u
IFS=- read -r TILE PART <<< "$1"
lemcvt -v a9 pilot --tile "$TILE" --part "$PART"
