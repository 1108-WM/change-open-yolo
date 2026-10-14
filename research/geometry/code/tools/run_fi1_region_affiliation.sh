#!/usr/bin/env bash
set -euo pipefail
cd /home/jia/Wm/wm_open-yolo/OpenYOLO3D
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=.venvs/yoloe_gpu/bin/python
for mode in features labels_train train predict_probabilities calibrate apply; do
    "$PY" tools/run_fi1_region_affiliation.py "$mode"
done
"$PY" tools/evaluate_fi1_region_affiliation.py
"$PY" tools/verify_fi1_region_affiliation.py
