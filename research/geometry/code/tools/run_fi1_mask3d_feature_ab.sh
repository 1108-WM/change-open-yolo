#!/usr/bin/env bash
set -euo pipefail
cd /home/jia/Wm/wm_open-yolo/OpenYOLO3D
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=".:models/Mask3D:models/YOLO-World${PYTHONPATH:+:$PYTHONPATH}"
PY=.venvs/yoloe_gpu/bin/python
OUT=.local/fi1_mask3d_feature_ab_internal_20261013_v1
if [ ! -f "$OUT/instance_features_frozen.json" ]; then
  /home/jia/anaconda3/envs/openyolo3d/bin/python -u tools/export_fi1_mask3d_instance_features.py
fi
for mode in train predict calibrate apply; do
  "$PY" -u tools/run_fi1_mask3d_feature_ab.py "$mode"
done
"$PY" -u tools/evaluate_fi1_mask3d_feature_ab.py
"$PY" -u tools/verify_fi1_mask3d_feature_ab.py
