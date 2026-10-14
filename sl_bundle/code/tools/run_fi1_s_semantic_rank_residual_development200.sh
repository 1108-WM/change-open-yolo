#!/usr/bin/env bash
set -euo pipefail
cd /home/jia/Wm/wm_open-yolo/OpenYOLO3D
TASK_OUT=.local/fi1_s_semantic_rank_residual_development200_20261010_v2
exec > "$TASK_OUT/run.log" 2>&1
.venvs/yoloe_gpu/bin/python tools/run_fi1_s_semantic_rank_residual_development200.py dataset
for fold in 0 1 2 3 4; do
  .venvs/yoloe_gpu/bin/python tools/run_fi1_s_semantic_rank_residual_development200.py fit --fold "$fold"
done
for stage in predict verify learnability evaluate report; do
  .venvs/yoloe_gpu/bin/python tools/run_fi1_s_semantic_rank_residual_development200.py "$stage"
done
