#!/usr/bin/env bash
set -euo pipefail
cd /home/jia/Wm/wm_open-yolo/OpenYOLO3D
TASK_OUT=.local/fi1_sr_cross_control_development200_20261010_v1
exec > "$TASK_OUT/run.log" 2>&1
for stage in predict verify_predictions evaluate diagnostics report; do
  .venvs/yoloe_gpu/bin/python tools/run_fi1_sr_cross_control_development200.py "$stage"
done
