#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/../../../.."
PYTHON=".venvs/yoloe_gpu/bin/python"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
"$PYTHON" tools/run_fi1_anchor_local_decontamination.py plan
"$PYTHON" tools/run_fi1_anchor_local_decontamination.py generate
"$PYTHON" tools/run_fi1_anchor_local_decontamination.py evaluate
