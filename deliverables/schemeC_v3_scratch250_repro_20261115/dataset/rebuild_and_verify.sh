#!/usr/bin/env bash
# Rebuild schemeC_pose23_ws_bysize with the shipped build scripts and verify it
# against the checksums/metadata in this pack.
#
# Usage:
#   ./rebuild_and_verify.sh [datasets_root] [output_view]
#
#   [datasets_root]  tree holding pose23_current/ and/or schemeC_pose23_full_v5/
#                    default: /home/wbk/ultralytics_2027/datasets
#   [output_view]    where the rebuilt view is written
#                    default: <datasets_root>/schemeC_pose23_ws_bysize_repro
#
# Env: PY=/path/to/python (default: python)
#      CODE=/path/to/code_snapshot  (default: ../code extracted next to this pack,
#                                    i.e. <pack>/../code/ultralytics_2027_5e9869ae8)
#
# Fidelity notes:
#  * build_pose23_ws_bysize.py accepts --src/--out and is fully relocatable.
#  * build_pose23_schemeC_full.py pins ROOT=/home/wbk/ultralytics_2027 in its
#    module constants (rel_of()/SPLIT_DIR), exactly as it ran on 2026-11-12.
#    So the full_v5 stage can only be rebuilt at that root; elsewhere an
#    existing schemeC_pose23_full_v5/ is reused as-is.
#
# The check is exact: split_manifest.tsv must match byte-for-byte, every label
# file must match its md5, and the summary counts must equal summary.json.
# Image bytes are NOT re-hashed here; run the optional md5 check printed at the
# end if you also want to validate the materialized JPEGs (2.1 GB read).

set -euo pipefail

PACK=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DS=${1:-/home/wbk/ultralytics_2027/datasets}
OUT=${2:-$DS/schemeC_pose23_ws_bysize_repro}
PY=${PY:-python}
CODE=${CODE:-$(cd "$PACK/../code" 2>/dev/null && pwd)/ultralytics_2027_5e9869ae8}
FULL=$DS/schemeC_pose23_full_v5

test -f "$CODE/scripts/build_pose23_ws_bysize.py" || {
  echo "code snapshot not found at $CODE"
  echo "extract code/ultralytics_2027_5e9869ae8.tar.gz next to it, or set:"
  echo "  CODE=<extracted snapshot root containing scripts/build_pose23_*.py>"
  exit 1; }

if [ -d "$FULL" ]; then
  echo "[1/3] reusing existing $FULL"
else
  test "$DS" = "/home/wbk/ultralytics_2027/datasets" || {
    echo "$FULL missing and build_pose23_schemeC_full.py is root-pinned to"
    echo "/home/wbk/ultralytics_2027; rebuild there or pass a datasets_root that has full_v5."
    exit 1; }
  test -d "$DS/pose23_current/split" || { echo "missing $DS/pose23_current/split"; exit 1; }
  echo "[1/3] full_v5 view <- pose23_current (defaults of the pinned-root script)"
  "$PY" "$CODE/scripts/build_pose23_schemeC_full.py" --force
fi

echo "[2/3] ws_bysize view <- full_v5 view"
"$PY" "$CODE/scripts/build_pose23_ws_bysize.py" --src "$FULL" --out "$OUT" --force

echo "[3/3] verifying"
for s in train val; do
  ni=$(find "$OUT/images/$s" -type l | wc -l)
  nl=$(find "$OUT/labels/$s" -name '*.txt' | wc -l)
  echo "  $s images=$ni labels=$nl"
done

cmp "$OUT/split_manifest.tsv" "$PACK/split_manifest.tsv" \
  && echo "  split_manifest.tsv: identical"

( cd "$OUT" && zcat "$PACK/labels_md5.txt.gz" | md5sum -c --quiet ) \
  && echo "  labels: all md5 match ($(zcat "$PACK/labels_md5.txt.gz" | wc -l) files)"

"$PY" - "$OUT/summary.json" "$PACK/summary.json" <<'PY'
import json, sys
a = json.load(open(sys.argv[1]))
b = json.load(open(sys.argv[2]))
keys = ("canvas", "ws_sessions", "images", "dropped", "objects", "classes")
bad = [k for k in keys if a.get(k) != b.get(k)]
assert not bad, f"summary mismatch: {bad}"
print("  summary.json: all counts identical")
PY

echo
echo "OK. The rebuilt view is $OUT"
echo "Optional image-byte check (reads 2.1 GB):"
echo "  ( cd \"$OUT\" && zcat \"$PACK/images_md5.txt.gz\" | md5sum -c --quiet ) && echo images md5 OK"
