#!/bin/bash
# Addendum 14: resumable parallel download of a URL list on a compute node, with size verification and a sha256 manifest.
# usage: a14_download.sh URLS.txt DEST_DIR STREAMS [SIZES.json]  (SIZES.json: {"basename": bytes} from the saved listing)
set -u
URLS=$1; DEST=$2; STREAMS=${3:-4}; SIZES=${4:-}
mkdir -p "$DEST"
UA="Mozilla/5.0 (research; scdobbs@stanford.edu)"
echo "download start $(date -Is) host=$(hostname) urls=$(wc -l < "$URLS") streams=$STREAMS dest=$DEST"
export DEST UA
cat "$URLS" | xargs -P "$STREAMS" -I{} bash -c '
  u="{}"; f="$DEST/$(basename "$u")"; t0=$(date +%s)
  out=$(curl -s -L -C - -A "$UA" --retry 5 --retry-delay 20 -m 7200 -o "$f" -w "%{http_code} %{size_download} %{speed_download}" "$u")
  echo "$(date -Is) $(basename "$u") $out $(( $(date +%s) - t0 ))s" >> "$DEST/download_log.txt"'
echo "download end $(date -Is)"
# verify sizes against the listing
if [ -n "$SIZES" ] && [ -f "$SIZES" ]; then
  python3 - "$URLS" "$DEST" "$SIZES" <<'PY'
import json, os, sys
urls, dest, sizes = sys.argv[1:4]
S = json.load(open(sizes)); bad = []; n = 0; tot = 0
for u in open(urls):
    b = os.path.basename(u.strip()); p = os.path.join(dest, b)
    if not b: continue
    n += 1
    have = os.path.getsize(p) if os.path.exists(p) else -1
    tot += max(have, 0)
    if S.get(b) is not None and have != S[b]: bad.append((b, have, S[b]))
print(f"verify: {n} files, {tot} bytes, {len(bad)} size mismatches"); 
for b in bad[:50]: print("  MISMATCH", b)
json.dump({"n": n, "bytes": tot, "mismatch": bad}, open(os.path.join(dest, "verify.json"), "w"))
PY
fi
(cd "$DEST" && ls *.laz *.zip 2>/dev/null | xargs -P 4 -I{} sha256sum {} > MANIFEST.sha256.tmp && sort -k2 MANIFEST.sha256.tmp > MANIFEST.sha256 && rm MANIFEST.sha256.tmp && echo "manifest: $(wc -l < MANIFEST.sha256) entries")
