#!/usr/bin/env bash
set -euo pipefail
cd /home/jia/Wm/wm_open-yolo/OpenYOLO3D
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH=".:models/Mask3D:models/YOLO-World${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
OUT=.local/replica_registered_full_20261011_v1
mkdir -p "$OUT"
trap 'rc=$?; printf "exit_code=%s\n" "$rc" > "$OUT/native_exit.txt"' EXIT
.venvs/yoloe_gpu/bin/python tools/run_replica_registered_full.py superpoints >> "$OUT/superpoints.log" 2>&1
/home/jia/anaconda3/envs/openyolo3d/bin/python tools/run_replica_registered_full.py native >> "$OUT/native.log" 2>&1
