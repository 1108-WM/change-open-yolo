#!/usr/bin/env bash
set -euo pipefail
cd /home/jia/Wm/wm_open-yolo/OpenYOLO3D
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
TASK_OUT=.local/fi1_s_member_quality_competition_development200_20261010_v1
exec > "$TASK_OUT/run.log" 2>&1
for stage in dataset fit predict verify evaluate report; do
  .venvs/yoloe_gpu/bin/python tools/run_fi1_s_member_quality_competition_development200.py "$stage"
done
