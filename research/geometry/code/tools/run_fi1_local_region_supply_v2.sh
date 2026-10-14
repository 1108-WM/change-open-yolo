#!/usr/bin/env bash
set -euo pipefail
cd /home/jia/Wm/wm_open-yolo/OpenYOLO3D
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
PYTHON=.venvs/yoloe_gpu/bin/python
"$PYTHON" -u tools/run_fi1_local_region_supply_v2.py diagnose
"$PYTHON" -u tools/run_fi1_local_region_supply_v2.py verify
