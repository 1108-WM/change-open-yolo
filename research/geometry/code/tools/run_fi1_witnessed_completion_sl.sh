#!/usr/bin/env bash
set -euo pipefail
cd /home/jia/Wm/wm_open-yolo/OpenYOLO3D
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
OUT=.local/fi1_witnessed_completion_SL_development200_20261010_v1
.venvs/yoloe_gpu/bin/python tools/run_fi1_witnessed_completion_sl.py acquire > "$OUT/acquire.log" 2>&1
.venvs/yoloe_gpu/bin/python tools/run_fi1_witnessed_completion_sl.py predict > "$OUT/predict.log" 2>&1
.venvs/yoloe_gpu/bin/python tools/run_fi1_witnessed_completion_sl.py evaluate > "$OUT/evaluate.log" 2>&1
