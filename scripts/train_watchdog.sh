#!/usr/bin/env bash
# Resume a Scheme-C run from last.pt when the process dies (CUDA Xid, OOM, ...).
#
#   ./scripts/train_watchdog.sh runs/pose/runs/pose/schemeC_480x640_v2
#
# Stops when results.csv reaches the epoch count in args.yaml or after 10 tries.
set -u
RUN=${1:?usage: $0 <run_dir>}
PY=${PY:-/home/wbk/miniconda3/envs/yolo/bin/python}
cd "$(dirname "$0")/.." || exit 1
EPOCHS=$(grep -m1 '^epochs:' "$RUN/args.yaml" | awk '{print $2}')
for i in $(seq 1 10); do
  DONE=$(($(wc -l < "$RUN/results.csv") - 1))
  if [ "$DONE" -ge "${EPOCHS:-999}" ]; then
    echo "[watchdog] finished: $DONE/$EPOCHS epochs"; exit 0
  fi
  echo "[watchdog] try $i: resuming at epoch $DONE/${EPOCHS:-?} ($(date '+%F %T'))"
  "$PY" train_schemeC.py --resume "$RUN/weights/last.pt"
  rc=$?
  echo "[watchdog] trainer exited rc=$rc ($(date '+%F %T'))"
  [ $rc -eq 0 ] && exit 0
  sleep 90
done
echo "[watchdog] giving up after 10 attempts"; exit 1
