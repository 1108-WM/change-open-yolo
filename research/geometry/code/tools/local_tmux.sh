#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../.." && pwd)"
RUNTIME_ROOT="$PROJECT_ROOT/.local/tmux_runtime"
export LD_LIBRARY_PATH="$RUNTIME_ROOT/lib/x86_64-linux-gnu:$RUNTIME_ROOT/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$RUNTIME_ROOT/usr/bin/tmux" "$@"
