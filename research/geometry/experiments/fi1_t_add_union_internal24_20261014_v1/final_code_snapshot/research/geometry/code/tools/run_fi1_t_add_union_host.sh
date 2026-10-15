#!/usr/bin/env bash
set -euo pipefail
cd /home/jia/Wm/wm_open-yolo/OpenYOLO3D
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=".:models/Mask3D:research/geometry/code/tools${PYTHONPATH:+:$PYTHONPATH}"
CPU=.venvs/yoloe_gpu/bin/python
GPU=/home/jia/anaconda3/envs/openyolo3d/bin/python
ENTRY=research/geometry/code/tools/run_fi1_t_add_union.py
OUT=research/geometry/experiments/fi1_t_add_union_internal24_20261014_v1
if [ ! -f "$OUT/contract.json" ]; then "$CPU" -u "$ENTRY" prepare; fi
if [ ! -f "$OUT/structure_frozen.json" ]; then "$CPU" -u "$ENTRY" structures; fi
if [ ! -f "$OUT/priors_frozen.json" ]; then "$GPU" -u research/geometry/code/tools/export_fi1_t_add_union_priors.py; fi
if [ ! -f "$OUT/predictions_frozen.json" ]; then "$CPU" -u "$ENTRY" predict; fi
if [ ! -f "$OUT/evaluation_frozen.json" ]; then "$CPU" -u "$ENTRY" evaluate; fi
if [ ! -f "$OUT/verification.json" ]; then "$CPU" -u research/geometry/code/tools/verify_fi1_t_add_union.py; fi
