#!/usr/bin/env bash
set -euo pipefail
cd /home/jia/Wm/wm_open-yolo/OpenYOLO3D
TASK_OUT=.local/fi1_sl_single200_dev55_20261010_v1
exec >> "$TASK_OUT/run.log" 2>&1
for phase in fit predict verify evaluate report; do
  .venvs/yoloe_gpu/bin/python tools/run_fi1_sl_single200_dev55.py "$phase"
done
.venvs/yoloe_gpu/bin/python tools/verify_fi1_sl_single200_dev55.py
