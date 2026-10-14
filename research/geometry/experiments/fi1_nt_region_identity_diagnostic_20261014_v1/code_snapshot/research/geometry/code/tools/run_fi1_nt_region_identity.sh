#!/usr/bin/env bash
set -euo pipefail
cd /home/jia/Wm/wm_open-yolo/OpenYOLO3D
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH=".:models/Mask3D:research/geometry/code/tools${PYTHONPATH:+:$PYTHONPATH}"
OUT=research/geometry/experiments/fi1_nt_region_identity_diagnostic_20261014_v1
CPU=.venvs/yoloe_gpu/bin/python
GPU=/home/jia/anaconda3/envs/openyolo3d/bin/python
if [ ! -f "$OUT/structure_frozen.json" ]; then
  "$CPU" -u research/geometry/code/tools/run_fi1_nt_region_identity.py structures
fi
if [ ! -f "$OUT/structure_diagnosis_frozen.json" ]; then
  "$CPU" -u research/geometry/code/tools/evaluate_fi1_nt_region_identity.py structure
fi
if [ ! -f "$OUT/priors_frozen.json" ]; then
  "$GPU" -u research/geometry/code/tools/run_fi1_nt_region_identity.py export
fi
if [ ! -f "$OUT/associations_frozen.json" ]; then
  "$CPU" -u research/geometry/code/tools/run_fi1_nt_region_identity.py associate
fi
if [ ! -f "$OUT/prior_diagnosis_frozen.json" ]; then
  "$CPU" -u research/geometry/code/tools/evaluate_fi1_nt_region_identity.py prior
fi
if [ ! -f "$OUT/summary.json" ]; then
  "$CPU" -u research/geometry/code/tools/evaluate_fi1_nt_region_identity.py report
fi
if [ ! -f "$OUT/verification.json" ]; then
  "$CPU" -u research/geometry/code/tools/verify_fi1_nt_region_identity.py
fi
