#!/usr/bin/env bash
set -Eeuo pipefail

# Recreate the tested A100 runtime for the frozen FI1-D-v3 x DM-SMS-1
# no-GT pipeline.  This script intentionally does not install or modify the
# NVIDIA driver.  Run it from a clean checkout of this repository.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda3}"
ENV_NAME="${ENV_NAME:-openyolo3d}"
ENV_ROOT="$CONDA_ROOT/envs/$ENV_NAME"
INSTALL_SYSTEM_PACKAGES=0
SKIP_EXTERNAL_SOURCES=0

ALPHACLIP_REPOSITORY="https://github.com/SunzeY/AlphaCLIP.git"
ALPHACLIP_COMMIT="ef9262bc539728bf8ef2dfe9c402ae12bbfcd9ff"
SAM_REPOSITORY="https://github.com/facebookresearch/segment-anything.git"
SAM_COMMIT="dca509fe793f601edb92606367a655c15ac00fdf"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  --install-system-packages  Install required Ubuntu packages with apt-get.
  --skip-external-sources    Do not clone/update Alpha-CLIP and SAM sources.
  -h, --help                 Show this help.

Environment overrides:
  CONDA_ROOT   Miniconda root (default: $CONDA_ROOT)
  ENV_NAME     Conda environment name (default: $ENV_NAME)

The NVIDIA driver is never installed or changed by this script.
EOF
}

while (($#)); do
  case "$1" in
    --install-system-packages) INSTALL_SYSTEM_PACKAGES=1; shift ;;
    --skip-external-sources) SKIP_EXTERNAL_SOURCES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ((INSTALL_SYSTEM_PACKAGES)); then
  command -v apt-get >/dev/null || { echo "ERROR: apt-get is unavailable" >&2; exit 2; }
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates curl git git-lfs rsync tmux wget build-essential \
    libgl1 libglib2.0-0 libxext6 libsm6 libxrender1
fi

if [[ ! -x "$CONDA_ROOT/bin/conda" ]]; then
  installer="$(mktemp /tmp/miniconda3.XXXXXX.sh)"
  trap 'rm -f -- "$installer"' EXIT
  curl --fail --location --retry 5 --retry-all-errors \
    --output "$installer" \
    https://repo.anaconda.com/miniconda/Miniconda3-py310_24.1.2-0-Linux-x86_64.sh
  bash "$installer" -b -p "$CONDA_ROOT"
  rm -f -- "$installer"
  trap - EXIT
fi

CONDA="$CONDA_ROOT/bin/conda"
if [[ ! -x "$ENV_ROOT/bin/python" ]]; then
  "$CONDA" create -y -n "$ENV_NAME" python=3.10.9 pip
fi

PYTHON="$ENV_ROOT/bin/python"
"$PYTHON" -m pip install --upgrade pip==24.0 setuptools==60.2.0 wheel==0.37.1
"$PYTHON" -m pip install \
  torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
"$PYTHON" -m pip install -r "$PROJECT_ROOT/deploy/a100/requirements-fi1-d-v3-dm-sms1-cu121.txt"

checkout_exact() {
  local repository="$1" commit="$2" destination="$3"
  if [[ -d "$destination/.git" ]]; then
    git -C "$destination" fetch --depth 1 origin "$commit"
  elif [[ -e "$destination" ]]; then
    echo "ERROR: external-source path exists but is not a Git checkout: $destination" >&2
    return 1
  else
    mkdir -p "$(dirname "$destination")"
    git clone --filter=blob:none "$repository" "$destination"
    git -C "$destination" fetch --depth 1 origin "$commit"
  fi
  git -C "$destination" checkout --detach "$commit"
  test "$(git -C "$destination" rev-parse HEAD)" = "$commit"
}

if ((SKIP_EXTERNAL_SOURCES == 0)); then
  checkout_exact "$ALPHACLIP_REPOSITORY" "$ALPHACLIP_COMMIT" \
    "$PROJECT_ROOT/_external/AlphaCLIP/AlphaCLIP-main"
  checkout_exact "$SAM_REPOSITORY" "$SAM_COMMIT" \
    "$PROJECT_ROOT/_external/segment-anything/segment-anything-main"
fi

"$PYTHON" "$PROJECT_ROOT/tools/verify_a100_fi1_dm_sms1_env.py" \
  --project-root "$PROJECT_ROOT"

cat <<EOF

Environment ready.
Python: $PYTHON
Activate: source "$CONDA_ROOT/bin/activate" "$ENV_NAME"

Next steps:
  1. Download the licensed ScanNet val312 data with
     scripts/download_scannet200_val_stream.sh
  2. Download third-party model assets with
     scripts/download_fi1_dm_sms1_model_assets.sh
  3. Run the repository tests and the frozen preflight before inference.
EOF
