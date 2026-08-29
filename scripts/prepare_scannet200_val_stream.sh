#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare the official ScanNet200 validation scenes for the original
# Open-YOLO 3D loader.  This is data conversion only; it does not run models.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
RAW_ROOT="${RAW_ROOT:-$PROJECT_ROOT/data/scannet_v2_raw}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_ROOT/data/scannet200}"
LABEL_MAP="${LABEL_MAP:-$RAW_ROOT/scannetv2-labels.combined.tsv}"
SCENE_LIST="${SCENE_LIST:-$PROJECT_ROOT/configs/repro/scannetv2_val_312.txt}"
FRAME_STEP="${FRAME_STEP:-10}"
JOBS="${JOBS:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-20}"
CONDA_ENV="${CONDA_ENV:-openyolo3d}"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
CONDA_EXE="${CONDA_EXE:-$CONDA_ROOT/bin/conda}"
RESUME=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [--resume]

Environment overrides:
  RAW_ROOT       Raw ScanNet root (default: $RAW_ROOT)
  OUTPUT_ROOT    Prepared root (default: $OUTPUT_ROOT)
  LABEL_MAP      ScanNet200 label map (default: $LABEL_MAP)
  SCENE_LIST     Validation list (default: $SCENE_LIST)
  FRAME_STEP     RGB/depth sampling step (default: $FRAME_STEP)
  JOBS           Concurrent scene conversions (default: $JOBS)
  MIN_FREE_GB    Stop before free space drops below this value (default: $MIN_FREE_GB)
  CONDA_ENV      Conda environment (default: $CONDA_ENV)
  CONDA_ROOT     Miniconda root (default: $CONDA_ROOT)
  CONDA_EXE      Conda executable (default: $CONDA_EXE)
EOF
}

while (($#)); do
  case "$1" in
    --resume) RESUME=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$FRAME_STEP" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: FRAME_STEP must be positive" >&2; exit 2; }
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: JOBS must be positive" >&2; exit 2; }
[[ "$MIN_FREE_GB" =~ ^[0-9]+$ ]] || { echo "ERROR: MIN_FREE_GB must be non-negative" >&2; exit 2; }
[[ -d "$RAW_ROOT/scans" ]] || { echo "ERROR: raw scans directory missing: $RAW_ROOT/scans" >&2; exit 2; }
[[ -f "$LABEL_MAP" ]] || { echo "ERROR: label map missing: $LABEL_MAP" >&2; exit 2; }
[[ -f "$SCENE_LIST" ]] || { echo "ERROR: scene list missing: $SCENE_LIST" >&2; exit 2; }

mapfile -t SCENES < <(sed -e 's/\r$//' -e '/^[[:space:]]*$/d' "$SCENE_LIST")
[[ "${#SCENES[@]}" -eq 312 ]] || {
  echo "ERROR: expected 312 validation scenes, found ${#SCENES[@]}" >&2
  exit 2
}

[[ -x "$CONDA_EXE" ]] || { echo "ERROR: conda executable not found: $CONDA_EXE" >&2; exit 2; }
mkdir -p "$OUTPUT_ROOT" "$OUTPUT_ROOT/logs"
LOG_FILE="$OUTPUT_ROOT/logs/prepare_$(date -u '+%Y%m%dT%H%M%SZ').log"
exec > >(tee -a "$LOG_FILE") 2>&1

free_bytes() { df -PB1 "$OUTPUT_ROOT" | awk 'NR == 2 { print $4 }'; }
min_free_bytes=$((MIN_FREE_GB * 1024 * 1024 * 1024))

run_one() {
  local scene="$1"
  local free
  free="$(free_bytes)"
  if ((free < min_free_bytes)); then
    echo "ERROR: free space below safety limit before $scene: $free bytes" >&2
    return 86
  fi
  echo "[prepare] $scene (free=$(numfmt --to=iec "$free" 2>/dev/null || echo "$free"))"
  local args=(
    tools/prepare_scannet200_val_scene.py
    --raw-scene-dir "$RAW_ROOT/scans/$scene"
    --output-root "$OUTPUT_ROOT"
    --label-map "$LABEL_MAP"
    --frame-step "$FRAME_STEP"
  )
  if ((RESUME)); then args+=(--resume); fi
  "$CONDA_EXE" run --no-capture-output -n "$CONDA_ENV" python "${args[@]}"
}

export PROJECT_ROOT RAW_ROOT OUTPUT_ROOT LABEL_MAP FRAME_STEP MIN_FREE_GB CONDA_ENV CONDA_ROOT CONDA_EXE RESUME
export -f free_bytes run_one
export min_free_bytes

if ((JOBS == 1)); then
  for scene in "${SCENES[@]}"; do run_one "$scene"; done
else
  printf '%s\n' "${SCENES[@]}" | xargs -P "$JOBS" -I {} bash -c 'run_one "$1"' _ {}
fi

echo "[prepare] completed all ${#SCENES[@]} scenes"
