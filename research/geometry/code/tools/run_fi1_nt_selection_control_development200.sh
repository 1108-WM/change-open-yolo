#!/usr/bin/env bash
# Run from the project root under tools/local_tmux.sh. Preserve failed phases.
set -euo pipefail
task_python=".venvs/yoloe_gpu/bin/python"
task_output="${1:-.local/fi1_nt_selection_control_development200_20261009_v1}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
"$task_python" tools/run_fi1_selection_control_development200.py tests --output "$task_output"
"$task_python" tools/run_fi1_selection_control_development200.py predictions --output "$task_output"
"$task_python" tools/verify_fi1_selection_control_development200.py selection --output "$task_output"
"$task_python" tools/run_fi1_selection_control_development200.py matching --output "$task_output"
"$task_python" tools/run_fi1_selection_control_development200.py evaluate --output "$task_output"
"$task_python" tools/verify_fi1_selection_control_development200.py final --output "$task_output"
"$task_python" tools/verify_fi1_selection_control_development200.py report --output "$task_output"
