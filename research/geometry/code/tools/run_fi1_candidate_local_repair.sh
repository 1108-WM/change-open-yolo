#!/usr/bin/env bash
set -euo pipefail
cd /home/jia/Wm/wm_open-yolo/OpenYOLO3D
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
PYTHON=.venvs/yoloe_gpu/bin/python
for mode in plan features labels train predict evaluate; do
  "$PYTHON" -u tools/run_fi1_candidate_local_repair.py "$mode"
done
"$PYTHON" -u tools/verify_fi1_candidate_local_repair.py
