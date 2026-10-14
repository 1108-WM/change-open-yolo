#!/usr/bin/env bash
set -euo pipefail
cd /home/jia/Wm/wm_open-yolo/OpenYOLO3D
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
.venvs/yoloe_gpu/bin/python tools/run_fi1_witnessed_local_completion.py generate > .local/fi1_witnessed_local_completion_development200_20261010_v1/generate.log 2>&1
.venvs/yoloe_gpu/bin/python tools/run_fi1_witnessed_local_completion.py quality > .local/fi1_witnessed_local_completion_development200_20261010_v1/quality.log 2>&1
