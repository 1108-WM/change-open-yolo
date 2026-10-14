#!/usr/bin/env bash
set -euo pipefail
cd /home/jia/Wm/wm_open-yolo/OpenYOLO3D
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1
OUT=.local/fi1_fragment_affiliation_learnability_development200_20261011_v1
.venvs/yoloe_gpu/bin/python tools/run_fi1_fragment_affiliation_learnability.py features > "$OUT/features.log" 2>&1
.venvs/yoloe_gpu/bin/python tools/run_fi1_fragment_affiliation_learnability.py train > "$OUT/train.log" 2>&1
