#!/usr/bin/env bash
set -Eeuo pipefail

# Download the six official ScanNet file types with resume support for any scene list. The default
# list is the 312 ScanNet200 validation scenes. URLs and the v1/v2 .sens
# mapping follow the official downloader at:
# http://kaldir.vc.cit.tum.de/scannet/download-scannet.py

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"

DEST="$PROJECT_ROOT/data/scannet_v2_raw"
NCS_DEST="/media/jia/软件1/scannet_train_stream"
DEST_CUSTOM=0
SCENE_LIST="$PROJECT_ROOT/models/Mask3D/third_party/ScanNet/Tasks/Benchmark/scannetv2_val.txt"
NCS_SPLIT_DIR="$PROJECT_ROOT/output/scannet200/scene_splits/ncs_independent_20260813"
SCENE_PRESET=""
EXPECTED_SCENE_COUNT=""
EXPECTED_SCENE_SHA=""
MIN_FREE_GB=30
MODE=download
ACCEPT_TOS=0
REFRESH_SIZES=0
RUN_IN_TMUX=0
TMUX_SESSION="scannet_ncs_all160"
TMUX_SESSION_CUSTOM=0
TMUX_BIN=""
TMUX_LD_LIBRARY_PATH=""
TMUX_SOCKET="$PROJECT_ROOT/.local/tmux/scannet.sock"
SIZE_JOBS=16
FILE_ATTEMPTS=100
RETRY_DELAY_SECONDS=15

OFFICIAL_ROOT="http://kaldir.vc.cit.tum.de/scannet"
OFFICIAL_DOWNLOADER_URL="$OFFICIAL_ROOT/download-scannet.py"
TOS_URL="$OFFICIAL_ROOT/ScanNet_TOS.pdf"

FILE_SUFFIXES=(
  ".sens"
  ".txt"
  "_vh_clean_2.ply"
  "_vh_clean_2.labels.ply"
  "_vh_clean_2.0.010000.segs.json"
  ".aggregation.json"
)

usage() {
  cat <<EOF
Usage: $(basename "$0") --accept-tos [options]

Downloads the six required official files for every scene in the selected
scene list, sequentially and with HTTP resume support. The default list is
the 312 ScanNet200 validation scenes; pass --scene-list to use a train or
custom development list.

Options:
  --accept-tos          Confirm that you have authorization and accept ScanNet TOS.
  --dest PATH           Download root (default: $DEST).
  --scene-list PATH     Scene list (default: $SCENE_LIST).
  --ncs-train           Use the frozen NCS train100 scene list.
  --ncs-validation      Use the frozen NCS validation60 scene list.
  --ncs-all             Use the frozen combined NCS all160 scene list.
  --min-free-gb N       Stop before free space drops below N GiB (default: 30).
  --estimate-only       Query official Content-Length values; download nothing.
  --verify-only         Verify all expected files against official sizes.
  --refresh-sizes       Rebuild the cached official-size manifest.
  --tmux                 Start this job in a detached tmux session.
  --tmux-session NAME    Set the tmux session name (default: $TMUX_SESSION).
  -h, --help            Show this help.

Examples:
  # Preview the full ScanNet v2 train payload without downloading files.
  bash $(basename "$0") --scene-list TRAIN_SCENES.txt --estimate-only
  # Stream a pre-registered independent subset with resume support.
  bash $(basename "$0") --accept-tos --scene-list NCS_TRAIN_SCENES.txt
  # Download all frozen NCS scenes in a detached tmux session.
  bash $(basename "$0") --accept-tos --ncs-all --tmux

Output layout:
  DEST/scans/sceneXXXX_XX/sceneXXXX_XX.sens
  DEST/scans/sceneXXXX_XX/<the other five official files>
  DEST/scannetv2-labels.combined.tsv
  DEST/download-scannet.py
  DEST/logs/
EOF
}

while (($#)); do
  case "$1" in
    --accept-tos)
      ACCEPT_TOS=1
      shift
      ;;
    --dest)
      [[ $# -ge 2 ]] || { echo "ERROR: --dest requires a path" >&2; exit 2; }
      DEST="$2"
      DEST_CUSTOM=1
      shift 2
      ;;
    --scene-list)
      [[ $# -ge 2 ]] || { echo "ERROR: --scene-list requires a path" >&2; exit 2; }
      [[ -z "$SCENE_PRESET" ]] || { echo "ERROR: --scene-list cannot be combined with an NCS preset" >&2; exit 2; }
      SCENE_LIST="$2"
      shift 2
      ;;
    --ncs-train)
      [[ "$SCENE_LIST" == "$PROJECT_ROOT/models/Mask3D/third_party/ScanNet/Tasks/Benchmark/scannetv2_val.txt" ]] || {
        echo "ERROR: an NCS preset cannot be combined with --scene-list" >&2; exit 2;
      }
      [[ -z "$SCENE_PRESET" ]] || { echo "ERROR: choose only one NCS preset" >&2; exit 2; }
      SCENE_PRESET="ncs-train100"
      SCENE_LIST="$NCS_SPLIT_DIR/ncs_train100.txt"
      EXPECTED_SCENE_COUNT=100
      EXPECTED_SCENE_SHA="ecebf3a682a02665a9770c2fa0ef83cc61c0c4ec2f185820aa44d667c8c4cd40"
      [[ "$TMUX_SESSION_CUSTOM" -eq 1 ]] || TMUX_SESSION="scannet_ncs_train100"
      shift
      ;;
    --ncs-validation)
      [[ "$SCENE_LIST" == "$PROJECT_ROOT/models/Mask3D/third_party/ScanNet/Tasks/Benchmark/scannetv2_val.txt" ]] || {
        echo "ERROR: an NCS preset cannot be combined with --scene-list" >&2; exit 2;
      }
      [[ -z "$SCENE_PRESET" ]] || { echo "ERROR: choose only one NCS preset" >&2; exit 2; }
      SCENE_PRESET="ncs-validation60"
      SCENE_LIST="$NCS_SPLIT_DIR/ncs_validation60.txt"
      EXPECTED_SCENE_COUNT=60
      EXPECTED_SCENE_SHA="9046668d8bc0987e06bb63603302434b34787b34a9082a9b988846de5a5b9ca2"
      [[ "$TMUX_SESSION_CUSTOM" -eq 1 ]] || TMUX_SESSION="scannet_ncs_validation60"
      shift
      ;;
    --ncs-all)
      [[ "$SCENE_LIST" == "$PROJECT_ROOT/models/Mask3D/third_party/ScanNet/Tasks/Benchmark/scannetv2_val.txt" ]] || {
        echo "ERROR: an NCS preset cannot be combined with --scene-list" >&2; exit 2;
      }
      [[ -z "$SCENE_PRESET" ]] || { echo "ERROR: choose only one NCS preset" >&2; exit 2; }
      SCENE_PRESET="ncs-all160"
      SCENE_LIST="$NCS_SPLIT_DIR/ncs_all160.txt"
      EXPECTED_SCENE_COUNT=160
      EXPECTED_SCENE_SHA="d3087c12762ab7644fe2d1fb9cc5db38c44a21c4f53b8a3a28eed2bca088f5aa"
      [[ "$TMUX_SESSION_CUSTOM" -eq 1 ]] || TMUX_SESSION="scannet_ncs_all160"
      shift
      ;;
    --min-free-gb)
      [[ $# -ge 2 ]] || { echo "ERROR: --min-free-gb requires an integer" >&2; exit 2; }
      MIN_FREE_GB="$2"
      shift 2
      ;;
    --estimate-only)
      MODE=estimate
      shift
      ;;
    --verify-only)
      MODE=verify
      shift
      ;;
    --refresh-sizes)
      REFRESH_SIZES=1
      shift
      ;;
    --tmux)
      RUN_IN_TMUX=1
      shift
      ;;
    --tmux-session)
      [[ $# -ge 2 ]] || { echo "ERROR: --tmux-session requires a name" >&2; exit 2; }
      TMUX_SESSION="$2"
      TMUX_SESSION_CUSTOM=1
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "$SCENE_PRESET" && "$DEST_CUSTOM" -eq 0 ]]; then
  DEST="$NCS_DEST"
fi

[[ "$MIN_FREE_GB" =~ ^[0-9]+$ ]] || {
  echo "ERROR: --min-free-gb must be a non-negative integer" >&2
  exit 2
}
[[ "$TMUX_SESSION" =~ ^[A-Za-z0-9_.-]+$ ]] || {
  echo "ERROR: tmux session name may contain only letters, digits, dot, underscore, and hyphen" >&2
  exit 2
}
[[ -f "$SCENE_LIST" ]] || { echo "ERROR: scene list not found: $SCENE_LIST" >&2; exit 2; }
command -v curl >/dev/null || { echo "ERROR: curl is required" >&2; exit 2; }
command -v flock >/dev/null || { echo "ERROR: flock is required" >&2; exit 2; }

mapfile -t SCENES < <(sed -e 's/\r$//' -e '/^[[:space:]]*$/d' "$SCENE_LIST")
if [[ ${#SCENES[@]} -eq 0 ]]; then
  echo "ERROR: scene list is empty: $SCENE_LIST" >&2
  exit 2
fi
declare -A SEEN_SCENES=()
for scene in "${SCENES[@]}"; do
  [[ "$scene" =~ ^scene[0-9]{4}_[0-9]{2}$ ]] || {
    echo "ERROR: invalid scene id in list: $scene" >&2
    exit 2
  }
  [[ -z "${SEEN_SCENES[$scene]+present}" ]] || {
    echo "ERROR: duplicate scene id in list: $scene" >&2
    exit 2
  }
  SEEN_SCENES["$scene"]=1
done

if [[ -n "$SCENE_PRESET" ]]; then
  ACTUAL_SCENE_SHA="$(sha256sum "$SCENE_LIST" | awk '{print $1}')"
  [[ ${#SCENES[@]} -eq "$EXPECTED_SCENE_COUNT" ]] || {
    echo "ERROR: frozen $SCENE_PRESET count changed: ${#SCENES[@]} != $EXPECTED_SCENE_COUNT" >&2
    exit 2
  }
  [[ "$ACTUAL_SCENE_SHA" == "$EXPECTED_SCENE_SHA" ]] || {
    echo "ERROR: frozen $SCENE_PRESET SHA-256 changed" >&2
    echo "       actual=$ACTUAL_SCENE_SHA" >&2
    echo "       expected=$EXPECTED_SCENE_SHA" >&2
    exit 2
  }
fi

# Cache size manifests by the normalized scene-list content. This prevents a
# train-list run from accidentally reusing a validation-list cache.
SCENE_LIST_SHA="$(printf '%s\n' "${SCENES[@]}" | LC_ALL=C sort | sha256sum | awk '{print substr($1, 1, 16)}')"
SCENE_LIST_NAME="$(basename "$SCENE_LIST")"
SCENE_SET_TAG="${SCENE_LIST_NAME%.*}_${SCENE_LIST_SHA}"

if [[ "$RUN_IN_TMUX" -eq 1 ]]; then
  if command -v tmux >/dev/null; then
    TMUX_BIN="$(command -v tmux)"
  elif [[ -x "$PROJECT_ROOT/.local/opt/tmux/usr/bin/tmux" ]]; then
    TMUX_BIN="$PROJECT_ROOT/.local/opt/tmux/usr/bin/tmux"
    TMUX_LD_LIBRARY_PATH="$PROJECT_ROOT/.local/opt/tmux/usr/lib/x86_64-linux-gnu"
  else
    echo "ERROR: tmux is required for --tmux" >&2
    echo "       system tmux and $PROJECT_ROOT/.local/opt/tmux/usr/bin/tmux are both unavailable" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$TMUX_SOCKET")"
  env LD_LIBRARY_PATH="$TMUX_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$TMUX_BIN" -S "$TMUX_SOCKET" has-session -t "=$TMUX_SESSION" 2>/dev/null && {
    echo "ERROR: tmux session already exists: $TMUX_SESSION" >&2
    echo "Attach with: tmux attach -t $TMUX_SESSION" >&2
    exit 3
  }

  CHILD_ARGS=(--dest "$DEST" --min-free-gb "$MIN_FREE_GB")
  case "$SCENE_PRESET" in
    ncs-train100) CHILD_ARGS+=(--ncs-train) ;;
    ncs-validation60) CHILD_ARGS+=(--ncs-validation) ;;
    ncs-all160) CHILD_ARGS+=(--ncs-all) ;;
    *) CHILD_ARGS+=(--scene-list "$SCENE_LIST") ;;
  esac
  [[ "$ACCEPT_TOS" -eq 1 ]] && CHILD_ARGS+=(--accept-tos)
  [[ "$MODE" == estimate ]] && CHILD_ARGS+=(--estimate-only)
  [[ "$MODE" == verify ]] && CHILD_ARGS+=(--verify-only)
  [[ "$REFRESH_SIZES" -eq 1 ]] && CHILD_ARGS+=(--refresh-sizes)
  printf -v TMUX_COMMAND '%q ' bash "$SCRIPT_PATH" "${CHILD_ARGS[@]}"
  env LD_LIBRARY_PATH="$TMUX_LD_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$TMUX_BIN" -S "$TMUX_SOCKET" new-session -d -s "$TMUX_SESSION" "$TMUX_COMMAND"
  echo "Started detached tmux session: $TMUX_SESSION"
  if [[ -n "$TMUX_LD_LIBRARY_PATH" ]]; then
    echo "Attach: LD_LIBRARY_PATH=$TMUX_LD_LIBRARY_PATH $TMUX_BIN -S $TMUX_SOCKET attach -t $TMUX_SESSION"
    echo "Status: LD_LIBRARY_PATH=$TMUX_LD_LIBRARY_PATH $TMUX_BIN -S $TMUX_SOCKET list-sessions"
  else
    echo "Attach: $TMUX_BIN -S $TMUX_SOCKET attach -t $TMUX_SESSION"
    echo "Status: $TMUX_BIN -S $TMUX_SOCKET list-sessions"
  fi
  echo "The local job survives terminal closure."
  echo "If this computer sleeps, transfer pauses; after wake-up, .part files resume automatically."
  exit 0
fi

if [[ "$MODE" == download && "$ACCEPT_TOS" -ne 1 ]]; then
  echo "ERROR: pass --accept-tos only after you have authorization and accept:" >&2
  echo "       $TOS_URL" >&2
  exit 2
fi

mkdir -p "$DEST" "$DEST/scans" "$DEST/logs"
exec 9>"$DEST/.download_scannet.lock"
flock -n 9 || { echo "ERROR: another downloader is already using $DEST" >&2; exit 3; }

RUN_ID="$(date -u '+%Y%m%dT%H%M%SZ')"
LOG_FILE="$DEST/logs/${MODE}_${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

GiB=$((1024 * 1024 * 1024))
MIN_FREE_BYTES=$((MIN_FREE_GB * GiB))

human_bytes() {
  awk -v bytes="$1" 'BEGIN {
    split("B KiB MiB GiB TiB", unit, " "); value = bytes + 0; level = 1;
    while (value >= 1024 && level < 5) { value /= 1024; level++ }
    printf "%.2f %s", value, unit[level]
  }'
}

available_bytes() {
  df -PB1 "$DEST" | awk 'NR == 2 { print $4 }'
}

url_for() {
  local scene="$1" suffix="$2"
  if [[ "$suffix" == ".sens" ]]; then
    # The official v2 downloader deliberately reuses v1 .sens streams.
    printf '%s/v1/scans/%s/%s%s\n' "$OFFICIAL_ROOT" "$scene" "$scene" "$suffix"
  else
    printf '%s/v2/scans/%s/%s%s\n' "$OFFICIAL_ROOT" "$scene" "$scene" "$suffix"
  fi
}

remote_size() {
  local url="$1" headers size
  headers="$(curl --fail --silent --show-error --location --head \
    --connect-timeout 20 --max-time 90 --retry 4 --retry-delay 3 \
    --retry-all-errors "$url")" || return 1
  size="$(printf '%s\n' "$headers" | tr -d '\r' | \
    awk 'tolower($1) == "content-length:" { value=$2 } END { print value }')"
  [[ "$size" =~ ^[0-9]+$ ]] && ((size > 0)) || return 1
  printf '%s\n' "$size"
}

SIZE_MANIFEST="$DEST/official_file_sizes_${SCENE_SET_TAG}.tsv"
SIZE_PARTIAL="$DEST/official_file_sizes_${SCENE_SET_TAG}.partial.tsv"
EXPECTED_FILE_COUNT=$((${#SCENES[@]} * ${#FILE_SUFFIXES[@]}))

valid_size_manifest() {
  [[ -f "$SIZE_MANIFEST" ]] || return 1
  [[ "$(wc -l < "$SIZE_MANIFEST")" -eq "$EXPECTED_FILE_COUNT" ]] || return 1
  awk -F '\t' '
    NF != 4 || $1 !~ /^scene[0-9][0-9][0-9][0-9]_[0-9][0-9]$/ ||
      $3 !~ /^[0-9]+$/ || $3 <= 0 { bad=1 }
    END { exit bad }
  ' "$SIZE_MANIFEST"
}

build_size_manifest() {
  local request_file missing_file round_file merged_file partial_file round missing_count
  request_file="$(mktemp "$DEST/logs/.size_requests.XXXXXX")"
  missing_file="$(mktemp "$DEST/logs/.size_missing.XXXXXX")"
  round_file="$(mktemp "$DEST/logs/.size_round.XXXXXX")"
  merged_file="$(mktemp "$DEST/logs/.size_merged.XXXXXX")"
  partial_file="$SIZE_PARTIAL"
  trap 'rm -f -- "$request_file" "$missing_file" "$round_file" "$merged_file"' RETURN

  for scene in "${SCENES[@]}"; do
    for suffix in "${FILE_SUFFIXES[@]}"; do
      printf '%s\t%s\t%s\n' "$scene" "$suffix" "$(url_for "$scene" "$suffix")"
    done
  done > "$request_file"

  touch "$partial_file"
  for round in 1 2 3 4 5 6; do
    if [[ -s "$partial_file" ]]; then
      awk -F '\t' '
        NR == FNR { present[$1 FS $2]=1; next }
        !present[$1 FS $2]
      ' "$partial_file" "$request_file" > "$missing_file"
    else
      cp -- "$request_file" "$missing_file"
    fi
    missing_count="$(wc -l < "$missing_file")"
    if [[ "$missing_count" -eq 0 ]]; then
      break
    fi

    echo "Official-size pass $round: querying $missing_count files ($SIZE_JOBS parallel HEAD requests)..."
    : > "$round_file"
    xargs -P "$SIZE_JOBS" -n 3 bash -c '
      scene="$1"; suffix="$2"; url="$3"
      headers="$(curl --fail --silent --show-error --location --head \
        --connect-timeout 15 --max-time 90 --retry 4 --retry-delay 3 \
        --retry-all-errors "$url")" || exit 1
      size="$(printf "%s\n" "$headers" | tr -d "\r" | \
        awk "tolower(\$1) == \"content-length:\" { value=\$2 } END { print value }")"
      [[ "$size" =~ ^[0-9]+$ ]] && ((size > 0)) || exit 1
      printf "%s\t%s\t%s\t%s\n" "$scene" "$suffix" "$size" "$url"
    ' _ < "$missing_file" > "$round_file" || true

    awk -F '\t' 'NF == 4 && $3 ~ /^[0-9]+$/ && $3 > 0' \
      "$partial_file" "$round_file" |
      LC_ALL=C sort -u -t $'\t' -k1,1 -k2,2 > "$merged_file"
    mv -- "$merged_file" "$partial_file"
    merged_file="$(mktemp "$DEST/logs/.size_merged.XXXXXX")"
  done

  if [[ "$(wc -l < "$partial_file")" -ne "$EXPECTED_FILE_COUNT" ]]; then
    echo "ERROR: official-size manifest remains incomplete: $(wc -l < "$partial_file")/$EXPECTED_FILE_COUNT" >&2
    echo "Re-run the same command; successful HEAD results are cached in $partial_file" >&2
    return 1
  fi
  mv -- "$partial_file" "$SIZE_MANIFEST"
  rm -f -- "$request_file" "$missing_file" "$round_file" "$merged_file"
  trap - RETURN
  echo "Saved official-size manifest: $SIZE_MANIFEST"
}

download_url() {
  local url="$1" final="$2" expected="$3"
  local part="${final}.part" current=0 needed free final_size attempt curl_status

  mkdir -p "$(dirname "$final")"
  if [[ -f "$final" ]]; then
    final_size="$(stat -c '%s' "$final")"
    if [[ "$final_size" -eq "$expected" ]]; then
      echo "SKIP complete: $final ($(human_bytes "$expected"))"
      return 0
    fi
    if [[ "$final_size" -lt "$expected" && ! -e "$part" ]]; then
      echo "RESUME converting short final file to .part: $final"
      mv -- "$final" "$part"
    else
      echo "ERROR: existing file has unexpected size: $final" >&2
      echo "       local=$final_size expected=$expected" >&2
      return 1
    fi
  fi

  if [[ -f "$part" ]]; then
    current="$(stat -c '%s' "$part")"
    if [[ "$current" -eq "$expected" ]]; then
      mv -- "$part" "$final"
      echo "DONE recovered complete part: $final"
      return 0
    fi
    if [[ "$current" -gt "$expected" ]]; then
      echo "ERROR: partial file is larger than official size: $part" >&2
      return 1
    fi
  fi

  needed=$((expected - current))
  free="$(available_bytes)"
  if ((free - needed < MIN_FREE_BYTES)); then
    echo "STOP: insufficient safe space for $final" >&2
    echo "      free=$(human_bytes "$free") needed=$(human_bytes "$needed") reserve=${MIN_FREE_GB} GiB" >&2
    return 75
  fi

  for attempt in $(seq 1 "$FILE_ATTEMPTS"); do
    current=0
    [[ -f "$part" ]] && current="$(stat -c '%s' "$part")"
    if [[ "$current" -eq "$expected" ]]; then
      break
    fi
    if [[ "$current" -gt "$expected" ]]; then
      echo "ERROR: partial file is larger than official size: $part" >&2
      return 1
    fi

    needed=$((expected - current))
    free="$(available_bytes)"
    if ((free - needed < MIN_FREE_BYTES)); then
      echo "STOP: insufficient safe space for $final" >&2
      echo "      free=$(human_bytes "$free") needed=$(human_bytes "$needed") reserve=${MIN_FREE_GB} GiB" >&2
      return 75
    fi

    echo "GET  [$attempt/$FILE_ATTEMPTS] $url"
    echo "     target=$final total=$(human_bytes "$expected") resumed=$(human_bytes "$current") free=$(human_bytes "$free")"
    curl_status=0
    curl --fail --location --show-error --connect-timeout 30 \
      --speed-limit 1024 --speed-time 180 \
      --continue-at - --output "$part" "$url" || curl_status=$?

    current=0
    [[ -f "$part" ]] && current="$(stat -c '%s' "$part")"
    if [[ "$current" -eq "$expected" ]]; then
      break
    fi
    echo "RETRY: curl_status=$curl_status saved=$(human_bytes "$current") expected=$(human_bytes "$expected")" >&2
    echo "       retrying in ${RETRY_DELAY_SECONDS}s; the saved .part bytes will be retained" >&2
    sleep "$RETRY_DELAY_SECONDS"
  done

  current=0
  [[ -f "$part" ]] && current="$(stat -c '%s' "$part")"
  if [[ "$current" -ne "$expected" ]]; then
    echo "ERROR: download failed after $FILE_ATTEMPTS resumable attempts: $part" >&2
    echo "       local=$current expected=$expected" >&2
    return 1
  fi
  mv -- "$part" "$final"
  echo "DONE $final ($(human_bytes "$expected"))"
}

fetch_small_official_file() {
  local url="$1" final="$2" expected
  expected="$(remote_size "$url")" || {
    echo "ERROR: cannot read official size: $url" >&2
    return 1
  }
  download_url "$url" "$final" "$expected"
}

echo "ScanNet official stream job"
echo "mode=$MODE"
echo "destination=$DEST"
echo "scene_list=$SCENE_LIST"
echo "scene_set=$SCENE_SET_TAG"
echo "scenes=${#SCENES[@]} file_types=${#FILE_SUFFIXES[@]} expected_files=$EXPECTED_FILE_COUNT"
echo "filesystem=$(df -hT "$DEST" | awk 'NR == 2 {print $1, $2, "size=" $3, "used=" $4, "avail=" $5}')"
echo "reserve=${MIN_FREE_GB} GiB"
echo "official_downloader=$OFFICIAL_DOWNLOADER_URL"
echo "log=$LOG_FILE"

if [[ "$REFRESH_SIZES" -eq 1 ]]; then
  rm -f -- "$SIZE_MANIFEST" "$SIZE_PARTIAL"
fi
if ! valid_size_manifest; then
  build_size_manifest
else
  echo "Using cached official-size manifest: $SIZE_MANIFEST"
fi

total_remote=0
existing_bytes=0
remaining_bytes=0
complete_files=0
missing_files=0
failed_files=0

declare -A EXPECTED_SIZES
while IFS=$'\t' read -r manifest_scene manifest_suffix manifest_size manifest_url; do
  EXPECTED_SIZES["$manifest_scene|$manifest_suffix"]="$manifest_size"
  total_remote=$((total_remote + manifest_size))
  manifest_final="$DEST/scans/$manifest_scene/${manifest_scene}${manifest_suffix}"
  manifest_part="${manifest_final}.part"
  local_bytes=0
  if [[ -f "$manifest_final" ]]; then
    candidate_bytes="$(stat -c '%s' "$manifest_final")"
    ((candidate_bytes <= manifest_size)) && local_bytes="$candidate_bytes"
  elif [[ -f "$manifest_part" ]]; then
    candidate_bytes="$(stat -c '%s' "$manifest_part")"
    ((candidate_bytes <= manifest_size)) && local_bytes="$candidate_bytes"
  fi
  existing_bytes=$((existing_bytes + local_bytes))
done < "$SIZE_MANIFEST"
remaining_bytes=$((total_remote - existing_bytes))

free_before="$(available_bytes)"
usable_before=$((free_before - MIN_FREE_BYTES))
echo "official_payload=$(human_bytes "$total_remote")"
echo "already_present_or_partial=$(human_bytes "$existing_bytes")"
echo "remaining_download=$(human_bytes "$remaining_bytes")"
echo "free_space=$(human_bytes "$free_before") reserve=${MIN_FREE_GB} GiB"
if ((remaining_bytes <= usable_before)); then
  echo "FIT=YES"
else
  echo "FIT=NO additional_needed=$(human_bytes "$((remaining_bytes - usable_before))")" >&2
  if [[ "$MODE" != verify ]]; then
    exit 75
  fi
fi

if [[ "$MODE" == estimate ]]; then
  echo "Estimate complete; no dataset files were downloaded."
  exit 0
fi

if [[ "$MODE" == download ]]; then
  fetch_small_official_file "$OFFICIAL_DOWNLOADER_URL" "$DEST/download-scannet.py"
  fetch_small_official_file \
    "$OFFICIAL_ROOT/v2/tasks/scannetv2-labels.combined.tsv" \
    "$DEST/scannetv2-labels.combined.tsv"
fi

for scene_index in "${!SCENES[@]}"; do
  scene="${SCENES[$scene_index]}"
  printf '\n[%03d/%03d] %s\n' "$((scene_index + 1))" "${#SCENES[@]}" "$scene"
  for suffix in "${FILE_SUFFIXES[@]}"; do
    url="$(url_for "$scene" "$suffix")"
    final="$DEST/scans/$scene/${scene}${suffix}"
    expected="${EXPECTED_SIZES["$scene|$suffix"]}"

    if [[ "$MODE" == verify ]]; then
      if [[ -f "$final" && "$(stat -c '%s' "$final")" -eq "$expected" ]]; then
        complete_files=$((complete_files + 1))
      else
        echo "MISSING_OR_INCOMPLETE: $final expected=$expected"
        missing_files=$((missing_files + 1))
      fi
      continue
    fi

    if download_url "$url" "$final" "$expected"; then
      complete_files=$((complete_files + 1))
    else
      status=$?
      failed_files=$((failed_files + 1))
      if [[ "$status" -eq 75 ]]; then
        echo "Download stopped safely because the reserved free-space limit was reached." >&2
        echo "Re-run the same command after adding/freeing space; completed files will be skipped." >&2
        exit 75
      fi
    fi
  done
done

free="$(available_bytes)"
printf '\nSummary\n'
echo "official_payload=$(human_bytes "$total_remote")"
echo "complete_files=$complete_files"
echo "missing_files=$missing_files"
echo "failed_files=$failed_files"
echo "free_space=$(human_bytes "$free")"
echo "log=$LOG_FILE"

if ((missing_files > 0 || failed_files > 0)); then
  exit 1
fi
