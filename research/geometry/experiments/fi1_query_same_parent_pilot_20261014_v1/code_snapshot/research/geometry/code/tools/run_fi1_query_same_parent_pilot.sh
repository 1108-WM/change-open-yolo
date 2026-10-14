#!/usr/bin/env bash
set -euo pipefail
cd /home/jia/Wm/wm_open-yolo/OpenYOLO3D
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
PY=.venvs/yoloe_gpu/bin/python
OUT=research/geometry/experiments/fi1_query_same_parent_pilot_20261014_v1
for mode in train predict evaluate; do
  "$PY" -u research/geometry/code/tools/run_fi1_query_same_parent_pilot.py "$mode"
done
touch "$OUT/runner_exit_zero"
