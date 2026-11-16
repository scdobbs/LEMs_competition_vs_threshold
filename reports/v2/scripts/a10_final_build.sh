#!/bin/bash
# Addendum 10, 5.3: the final build from results only, and its copy to group storage.
set -eu
lemcvt -v phase6 record
lemcvt -v figures all
lemcvt -v tables all
python - <<'PY'
from lemcvt.figures.common import missing_captions
from pathlib import Path
m = missing_captions(Path("figures"))
print("figures without caption CSV:", m)
PY
DEST=$LEMCVT_RESULTS/v2.0.0
mkdir -p "$DEST"
rsync -a figures tables "$DEST"/
git rev-parse HEAD > "$DEST/GIT_HASH"
( cd "$DEST" && find figures tables -type f | sort > FILES.txt && sha256sum $(cat FILES.txt) > SHA256SUMS )
echo "copied $(wc -l < "$DEST/FILES.txt") files to $DEST"
