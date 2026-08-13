#!/usr/bin/env bash
set -euo pipefail

# 只在无 GT 管线明确完成后触发一次既定正式 AP；失败或超时都不运行 AP。
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${ROOT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}"
TEST_ROOT="${TEST_ROOT:-$ROOT_DIR/output/gvc_test60_uniform30_20260803}"
AP_RUNNER="${AP_RUNNER:-$ROOT_DIR/tools/run_gvc_test60_native_ap.sh}"
STATE_FILE="$TEST_ROOT/no_gt_pipeline_state.txt"
MAX_POLLS="${MAX_POLLS:-180}"

for ((poll = 1; poll <= MAX_POLLS; poll++)); do
  if [[ -f "$STATE_FILE" ]]; then
    state="$(awk -F= '$1 == "state" {print $2}' "$STATE_FILE")"
    case "$state" in
      completed)
        echo "[INFO] test60 无 GT 预检完成，启动正式 native AP 对照。"
        exec bash "$AP_RUNNER"
        ;;
      failed)
        echo "[ERROR] test60 无 GT 管线失败，不运行 AP。" >&2
        exit 1
        ;;
    esac
  fi
  sleep 30
done

echo "[ERROR] 等待 test60 无 GT 预检超时，不运行 AP。" >&2
exit 1
