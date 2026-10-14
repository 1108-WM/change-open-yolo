#!/usr/bin/env bash
set -euo pipefail
cd /home/jia/Wm/wm_open-yolo/OpenYOLO3D
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=".:models/Mask3D:research/geometry/code/tools${PYTHONPATH:+:$PYTHONPATH}"
CPU=.venvs/yoloe_gpu/bin/python
GPU=/home/jia/anaconda3/envs/openyolo3d/bin/python
ENTRY=research/geometry/code/tools/run_fi1_query_transfer_eval.py
OUT=research/geometry/experiments/fi1_query_transfer_eval_20261014_v1
if [ ! -f "$OUT/structure_frozen.json" ]; then "$CPU" -u "$ENTRY" structures; fi
if [ ! -f "$OUT/priors_frozen.json" ]; then "$GPU" -u "$ENTRY" export; fi
if [ ! -f "$OUT/associations_frozen.json" ]; then "$CPU" -u "$ENTRY" associate; fi
if [ ! -f "$OUT/predictions_frozen.json" ]; then "$CPU" -u "$ENTRY" predict; fi
if [ ! -f "$OUT/decision.json" ]; then "$CPU" -u "$ENTRY" evaluate; fi
"$CPU" -u research/geometry/code/tools/verify_fi1_query_transfer_eval.py
touch "$OUT/runner_exit_zero"
