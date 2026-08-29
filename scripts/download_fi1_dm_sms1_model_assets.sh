#!/usr/bin/env bash
set -Eeuo pipefail

# Download only the public third-party weights required by the frozen
# FI1-D-v3 x DM-SMS-1 pipeline.  Project-specific frozen assets remain in Git.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
ENV_NAME="${ENV_NAME:-openyolo3d}"
PYTHON="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
ACCEPT_LICENSES=0

QWEN_REPOSITORY="Qwen/Qwen2.5-VL-7B-Instruct"
QWEN_REVISION="cc594898137f460bfe9f0759e9844b3ce807cfb5"

usage() {
  cat <<EOF
Usage: $(basename "$0") --accept-third-party-licenses

This downloads Qwen2.5-VL-7B-Instruct, SAM ViT-B, OpenAI CLIP ViT-L/14,
and the Alpha-CLIP L/14 GRIT-20M checkpoint.  Review and accept each upstream
license before passing the acceptance flag.  No credentials are embedded.
EOF
}

while (($#)); do
  case "$1" in
    --accept-third-party-licenses) ACCEPT_LICENSES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

((ACCEPT_LICENSES == 1)) || {
  echo "ERROR: review the upstream licenses, then pass --accept-third-party-licenses" >&2
  exit 2
}
[[ -x "$PYTHON" ]] || { echo "ERROR: environment Python missing: $PYTHON" >&2; exit 2; }
command -v curl >/dev/null || { echo "ERROR: curl is required" >&2; exit 2; }

download_verified() {
  local url="$1" output="$2" expected_sha="$3" actual
  mkdir -p "$(dirname "$output")"
  if [[ -f "$output" ]]; then
    actual="$(sha256sum "$output" | awk '{print $1}')"
    if [[ "$actual" == "$expected_sha" ]]; then
      echo "SKIP verified: $output"
      return 0
    fi
    echo "ERROR: existing asset has the wrong SHA-256: $output" >&2
    echo "       expected=$expected_sha actual=$actual" >&2
    return 1
  fi
  curl --fail --location --retry 10 --retry-all-errors --continue-at - \
    --output "${output}.part" "$url"
  actual="$(sha256sum "${output}.part" | awk '{print $1}')"
  [[ "$actual" == "$expected_sha" ]] || {
    echo "ERROR: downloaded asset SHA-256 mismatch: $output" >&2
    echo "       expected=$expected_sha actual=$actual" >&2
    return 1
  }
  mv -- "${output}.part" "$output"
  echo "DONE verified: $output"
}

download_verified \
  "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt" \
  "$PROJECT_ROOT/pretrained/alpha_clip/checkpoints/ViT-L-14.pt" \
  "b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836"

download_verified \
  "https://download.openxlab.org.cn/models/SunzeY/AlphaCLIP/weight/clip_l14_grit20m_fultune_2xe.pth" \
  "$PROJECT_ROOT/pretrained/alpha_clip/checkpoints/clip_l14_grit20m_fultune_2xe.pth" \
  "42c621bb5bac89a511ab625a878f026c11de4de8cf7abf52db5dfd11862bdb8e"

download_verified \
  "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
  "$PROJECT_ROOT/pretrained/checkpoints/sam_vit_b_01ec64.pth" \
  "ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912"

QWEN_OUTPUT="$PROJECT_ROOT/pretrained/checkpoints/Qwen2.5-VL-7B-Instruct"
"$PYTHON" - "$QWEN_REPOSITORY" "$QWEN_REVISION" "$QWEN_OUTPUT" <<'PY'
from pathlib import Path
import sys
from huggingface_hub import snapshot_download

repository, revision, output = sys.argv[1:]
snapshot_download(
    repo_id=repository,
    revision=revision,
    local_dir=Path(output),
)
print(f"Qwen snapshot ready: {output} revision={revision}")
PY

"$PYTHON" - "$QWEN_OUTPUT" "$QWEN_REVISION" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
expected = sys.argv[2]
metadata = sorted((root / ".cache/huggingface/download").glob("*.metadata"))
if not metadata:
    raise SystemExit("Qwen revision metadata is missing")
revisions = {path.read_text().splitlines()[0].strip() for path in metadata}
if revisions != {expected}:
    raise SystemExit(f"Qwen revision mismatch: {sorted(revisions)}")
print(f"Qwen revision verified: {expected} ({len(metadata)} metadata files)")
PY

echo "All public FI1-D-v3 x DM-SMS-1 model assets are present and verified."
