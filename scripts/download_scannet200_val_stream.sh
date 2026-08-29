#!/usr/bin/env bash
set -Eeuo pipefail

# Stream the six official ScanNet file types for the 312 ScanNet200 validation
# scenes.  URLs and the v1/v2 .sens mapping follow the official downloader at:
# http://kaldir.vc.cit.tum.de/scannet/download-scannet.py

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

DEST="$PROJECT_ROOT/data/scannet_v2_raw"
SCENE_LIST="$PROJECT_ROOT/configs/repro/scannetv2_val_312.txt"
MIN_FREE_GB=30
MODE=download
ACCEPT_TOS=0
REFRESH_SIZES=0
SIZE_JOBS=16
FILE_ATTEMPTS=100
RETRY_DELAY_SECONDS=15
DOWNLOAD_JOBS=3

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

Downloads the six required official files for all 312 ScanNet200 validation
scenes, with bounded scene-level concurrency and HTTP resume support.

Options:
  --accept-tos          Confirm that you have authorization and accept ScanNet TOS.
  --dest PATH           Download root (default: $DEST).
  --scene-list PATH     Scene list (default: $SCENE_LIST).
  --min-free-gb N       Stop before free space drops below N GiB (default: 30).
  --jobs N              Concurrent scenes in download mode (default: 3).
                        Each scene downloads its six files sequentially.
  --estimate-only       Query official Content-Length values; download nothing.
  --verify-only         Verify all expected files against official sizes.
  --refresh-sizes       Rebuild the cached official-size manifest.
  -h, --help            Show this help.

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
      shift 2
      ;;
    --scene-list)
      [[ $# -ge 2 ]] || { echo "ERROR: --scene-list requires a path" >&2; exit 2; }
      SCENE_LIST="$2"
      shift 2
      ;;
    --min-free-gb)
      [[ $# -ge 2 ]] || { echo "ERROR: --min-free-gb requires an integer" >&2; exit 2; }
      MIN_FREE_GB="$2"
      shift 2
      ;;
    --jobs)
      [[ $# -ge 2 ]] || { echo "ERROR: --jobs requires an integer" >&2; exit 2; }
      DOWNLOAD_JOBS="$2"
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

[[ "$MIN_FREE_GB" =~ ^[0-9]+$ ]] || {
  echo "ERROR: --min-free-gb must be a non-negative integer" >&2
  exit 2
}
[[ "$DOWNLOAD_JOBS" =~ ^[1-9][0-9]*$ ]] || {
  echo "ERROR: --jobs must be a positive integer" >&2
  exit 2
}
if ((DOWNLOAD_JOBS > 8)); then
  echo "ERROR: --jobs above 8 is intentionally disabled to avoid overloading the official server" >&2
  exit 2
fi
[[ -f "$SCENE_LIST" ]] || { echo "ERROR: scene list not found: $SCENE_LIST" >&2; exit 2; }
command -v curl >/dev/null || { echo "ERROR: curl is required" >&2; exit 2; }
command -v flock >/dev/null || { echo "ERROR: flock is required" >&2; exit 2; }

mapfile -t SCENES < <(sed -e 's/\r$//' -e '/^[[:space:]]*$/d' "$SCENE_LIST")
if [[ ${#SCENES[@]} -ne 312 ]]; then
  echo "ERROR: expected 312 validation scenes, found ${#SCENES[@]} in $SCENE_LIST" >&2
  exit 2
fi
for scene in "${SCENES[@]}"; do
  [[ "$scene" =~ ^scene[0-9]{4}_[0-9]{2}$ ]] || {
    echo "ERROR: invalid scene id in list: $scene" >&2
    exit 2
  }
done

if [[ "$MODE" == download && "$ACCEPT_TOS" -ne 1 ]]; then
  echo "ERROR: pass --accept-tos only after you have authorization and accept:" >&2
  echo "       $TOS_URL" >&2
  exit 2
fi

mkdir -p "$DEST" "$DEST/scans" "$DEST/logs"
exec 9>"$DEST/.download_scannet200_val.lock"
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

SIZE_MANIFEST="$DEST/official_file_sizes.tsv"
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
  partial_file="$DEST/official_file_sizes.partial.tsv"
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

echo "ScanNet200 validation stream job"
echo "mode=$MODE"
echo "destination=$DEST"
echo "scene_list=$SCENE_LIST"
echo "scenes=${#SCENES[@]} file_types=${#FILE_SUFFIXES[@]} expected_files=$EXPECTED_FILE_COUNT"
echo "download_jobs=$DOWNLOAD_JOBS (scene-level; files within each scene remain sequential)"
echo "filesystem=$(df -hT "$DEST" | awk 'NR == 2 {print $1, $2, "size=" $3, "used=" $4, "avail=" $5}')"
echo "reserve=${MIN_FREE_GB} GiB"
echo "official_downloader=$OFFICIAL_DOWNLOADER_URL"
echo "log=$LOG_FILE"

if [[ "$REFRESH_SIZES" -eq 1 ]]; then
  rm -f -- "$SIZE_MANIFEST" "$DEST/official_file_sizes.partial.tsv"
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

verify_all_files() {
  local manifest_scene manifest_suffix manifest_size manifest_url final
  complete_files=0
  missing_files=0
  while IFS=$'\t' read -r manifest_scene manifest_suffix manifest_size manifest_url; do
    final="$DEST/scans/$manifest_scene/${manifest_scene}${manifest_suffix}"
    if [[ -f "$final" && "$(stat -c '%s' "$final")" -eq "$manifest_size" ]]; then
      complete_files=$((complete_files + 1))
    else
      echo "MISSING_OR_INCOMPLETE: $final expected=$manifest_size"
      missing_files=$((missing_files + 1))
    fi
  done < "$SIZE_MANIFEST"
}

download_scene() {
  local scene_index="$1" scene="$2" suffix url final expected status scene_failed=0
  printf '\n[%03d/%03d] %s (worker_pid=%s)\n' \
    "$((scene_index + 1))" "${#SCENES[@]}" "$scene" "$BASHPID"

  for suffix in "${FILE_SUFFIXES[@]}"; do
    url="$(url_for "$scene" "$suffix")"
    final="$DEST/scans/$scene/${scene}${suffix}"
    expected="${EXPECTED_SIZES["$scene|$suffix"]}"
    if download_url "$url" "$final" "$expected"; then
      :
    else
      status=$?
      if [[ "$status" -eq 75 ]]; then
        return 75
      fi
      scene_failed=1
    fi
  done
  return "$scene_failed"
}

if [[ "$MODE" == verify ]]; then
  verify_all_files
else
  declare -a worker_pids=()

  stop_workers() {
    local pid child
    trap - INT TERM
    echo "Stopping parallel download workers; completed files and .part files are retained..." >&2
    for pid in "${worker_pids[@]:-}"; do
      [[ -n "$pid" ]] || continue
      while IFS= read -r child; do
        [[ -n "$child" ]] && kill -TERM "$child" 2>/dev/null || true
      done < <(pgrep -P "$pid" 2>/dev/null || true)
      kill -TERM "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    exit 130
  }
  trap stop_workers INT TERM

  run_worker_slot() {
    local slot="$1" scene_index scene worker_failed=0 status
    echo "WORKER_SLOT_START: slot=$slot/$DOWNLOAD_JOBS pid=$BASHPID"
    for ((scene_index = slot - 1; scene_index < ${#SCENES[@]}; scene_index += DOWNLOAD_JOBS)); do
      scene="${SCENES[$scene_index]}"
      if download_scene "$scene_index" "$scene"; then
        :
      else
        status=$?
        if [[ "$status" -eq 75 ]]; then
          echo "WORKER_SLOT_STOP: slot=$slot pid=$BASHPID reason=free-space"
          return 75
        fi
        worker_failed=1
      fi
    done
    echo "WORKER_SLOT_END: slot=$slot/$DOWNLOAD_JOBS pid=$BASHPID status=$worker_failed"
    return "$worker_failed"
  }

  declare -A worker_slots=()
  for ((slot = 1; slot <= DOWNLOAD_JOBS; slot++)); do
    run_worker_slot "$slot" &
    pid=$!
    worker_pids+=("$pid")
    worker_slots["$pid"]="$slot"
    echo "WORKER_LAUNCHED: slot=$slot/$DOWNLOAD_JOBS pid=$pid"
  done

  worker_failures=0
  space_stop=0
  for pid in "${worker_pids[@]}"; do
    if wait "$pid"; then
      status=0
    else
      status=$?
    fi
    echo "WORKER_JOINED: slot=${worker_slots[$pid]} pid=$pid status=$status"
    if [[ "$status" -eq 75 ]]; then
      space_stop=1
    elif [[ "$status" -ne 0 ]]; then
      worker_failures=$((worker_failures + 1))
    fi
  done
  trap - INT TERM

  if ((space_stop > 0)); then
    echo "Download stopped safely because the reserved free-space limit was reached." >&2
    echo "Re-run the same command after adding/freeing space; completed files will be skipped." >&2
    exit 75
  fi
  failed_files="$worker_failures"
  verify_all_files
fi

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
