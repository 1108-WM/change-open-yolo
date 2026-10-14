#!/usr/bin/env bash
set -euo pipefail
cd /home/jia/Wm/wm_open-yolo/OpenYOLO3D
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
OUT=.local/fi1_complementary_fragment_retrieval_identity_development200_20261011_v1
.venvs/yoloe_gpu/bin/python tools/run_fi1_complementary_fragment_diagnostic.py features > "$OUT/features.log" 2>&1
.venvs/yoloe_gpu/bin/python tools/run_fi1_complementary_fragment_diagnostic.py labels > "$OUT/labels.log" 2>&1
.venvs/yoloe_gpu/bin/python tools/run_fi1_complementary_fragment_diagnostic.py report > "$OUT/report.log" 2>&1
