#!/usr/bin/env bash
set -euo pipefail
cd /home/jia/Wm/wm_open-yolo/OpenYOLO3D
TASK_OUT=.local/fi1_sl_replay_stability_development200_20261010_v1
exec >> "$TASK_OUT/run.log" 2>&1
PY=.venvs/yoloe_gpu/bin/python
SCRIPT=tools/run_fi1_sl_replay_stability_development200.py
for phase in replay verify evaluate coverage; do "$PY" "$SCRIPT" "$phase"; done
for seed in 20261011 20261012; do
  for fold in 0 1 2 3 4; do "$PY" "$SCRIPT" fit --seed "$seed" --fold "$fold"; done
  for phase in predict verify learnability evaluate coverage; do "$PY" "$SCRIPT" "$phase" --seed "$seed"; done
done
"$PY" "$SCRIPT" report
