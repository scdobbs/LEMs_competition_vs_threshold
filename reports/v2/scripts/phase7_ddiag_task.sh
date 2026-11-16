#!/bin/bash
# Addendum 7, 2.1-2.3: the D diagnostic by hillslope class for one tile. Usage: bash phase7_ddiag_task.sh <TILE>
set -u
lemcvt -v phase7 d-diagnostic --tile "$1"
