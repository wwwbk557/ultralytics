#!/usr/bin/env bash
# EXACT command that produced runs/pose/runs/pose/schemeC_v3_scratch250
# (2026-11-12 22:14 -> 2026-11-13 11:48 +0800, RTX 3090).
#
# Confirmed verbatim in chat by the user; every value below is also recorded in
# the checkpoint's train_args (see args_resolved.yaml).  Run from the repo root.

set -euo pipefail
cd /hy-tmp/ultralytics_2027

python train_schemeC.py \
  --data datasets/schemeC_pose23_ws_bysize/data.yaml \
  --epochs 250 \
  --batch 16 \
  --imgsz 480,640 \
  --device 0 \
  --workers 8 \
  --degrees 35 \
  --env-p 0.4 \
  --plate-scale-p 0.35 \
  --plate-scale-range 0.85,1.15 \
  --plate-scale-max 2 \
  --plate-scale-min-px 16 \
  --quality-soft 0.4 \
  --obj-logit-bias auto \
  --deploy-boxes \
  --name schemeC_v3_scratch250

# ---------------------------------------------------------------------------
# Notes
# 1. No --project on purpose.  train_schemeC.py defaults to --project runs/pose,
#    and Ultralytics' get_save_dir() treats a relative project as a subfolder of
#    runs/<task>, so the run really landed in
#        runs/pose/runs/pose/schemeC_v3_scratch250
#    (the doubled path is authentic, not a typo).
#
# 2. The tmux variant offered at the time was:
#      cd /hy-tmp/ultralytics_2027 && tmux new -d -s schemeC_v3 \
#        "cd /hy-tmp/ultralytics_2027 && python train_schemeC.py ... \
#         --name schemeC_v3_scratch250 --project runs/pose \
#         2>&1 | tee /hy-tmp/schemeC_v3_scratch250.log"
#    Adding --project runs/pose moves the output to runs/pose/schemeC_v3_scratch250.
#    The weights that were downloaded came from the doubled path, i.e. the
#    no---project form above is the one that ran.
#
# 3. Everything not on the command line is inherited from train_schemeC.py /
#    default.yaml: AdamW, lr0 5e-4, lrf 0.01, cos_lr, warmup 5, close_mosaic 20,
#    mosaic 1.0, mixup/cutmix/copy_paste 0, fliplr 0, hsv 0.015/0.7/0.4,
#    translate 0.1, scale 0.5, erasing 0, patience 0, save_period 10,
#    amp False, deterministic True, seed 0.  See args_resolved.yaml.
#
# 4. close_mosaic=20 triggers PicoRectTrainer._close_dataloader_mosaic(), which
#    zeroes ALL 18 augmentation keys (not only mosaic/mixup/cutmix/copy_paste)
#    for the final 20 epochs and re-installs the keypoint->box sync.
# ---------------------------------------------------------------------------
