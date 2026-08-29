# A100 server reproduction and migration

Updated: 2026-08-29

This document reproduces the server-adapted FI1-D-v3 × legacy DM-SMS-1
checkout without committing licensed ScanNet data, third-party model weights,
generated ledgers, prediction caches, credentials, or Conda directories.

## 1. Frozen code identity

The deployment branch is:

```text
a100-server-repro-20260829
```

It is based on the exact frozen server code:

```text
dm-sms1-fi1-d-v3-val312-duplicate-safe-20260824
52f73a44c9e666338ef59c9c25757f212408a2a7
```

That history contains the duplicate-safe candidate/geometry split, exact
intermediate audits, terminal-safe-keep contract, direct CLI import fix and
append-only Qwen resume support used on the A100 server.  This deployment
package adds download/setup/verification material only.  It does not change
FI1-D-v3, DM-SMS-1 decisions, Alpha/SAM, Qwen prompts or evaluation rules.

Always read the frozen contracts before running:

- `docs/FI1_D_V3_VAL312_PREREGISTRATION_20260822.md`
- `docs/FI1_D_V3_VAL312_REMOTE_RUNBOOK_20260822.md`
- `docs/DM_SMS1_FI1_D_V3_VAL312_JOINT_PREREGISTRATION_20260824.md`
- `docs/DM_SMS1_FI1_D_V3_VAL312_DUPLICATE_SAFE_PREREGISTRATION_REVISION_20260824.md`
- `docs/DM_SMS1_FI1_D_V3_VAL312_TERMINAL_SAFE_KEEP_PREREGISTRATION_REVISION_20260825.md`
- `docs/DM_SMS1_FI1_D_V3_VAL312_REMOTE_RUNBOOK_20260824.md`

## 2. Reference server

The frozen run used:

```text
Ubuntu 22.04.1 LTS
NVIDIA A100-SXM4-40GB
NVIDIA driver 580.126.09
Python 3.10.9
PyTorch 2.5.1+cu121
torchvision 0.20.1+cu121
transformers 4.49.0
qwen-vl-utils 0.0.10
```

The setup script does not install or alter the NVIDIA driver.  Start from an
A100 image whose driver can run CUDA 12.1 PyTorch wheels.  A different but
newer compatible driver can work, but must be recorded in the new run log.

## 3. Clean clone and environment

```bash
git clone --recurse-submodules \
  --branch a100-server-repro-20260829 \
  ssh://git@ssh.github.com:443/1108-WM/change-open-yolo.git \
  /root/OpenYOLO3D

cd /root/OpenYOLO3D
git status --short --branch
git rev-parse HEAD

# Confirm that the checked-out commit is the published deployment commit
# reported when this branch is pushed.

bash scripts/setup_a100_fi1_dm_sms1_env.sh \
  --install-system-packages

/root/miniconda3/envs/openyolo3d/bin/python \
  tools/verify_a100_fi1_dm_sms1_env.py \
  --project-root /root/OpenYOLO3D \
  --require-cuda
```

The verifier performs imports and version/hash checks only.  It does not run
models, read ground truth or compute AP.

## 4. Download the official ScanNet validation set

ScanNet is licensed and is not in Git.  Obtain authorization and review the
official terms before using the acceptance flag.

The canonical 312-scene list is:

```text
configs/repro/scannetv2_val_312.txt
SHA-256 d75d4971c3fa7128c643695840e279042c212ef904fe933bd00cf9918c61b083
```

It is byte-identical to the official `scannetv2_val.txt` used by the completed
server run.

Estimate first, without downloading:

```bash
cd /root/OpenYOLO3D
bash scripts/download_scannet200_val_stream.sh \
  --estimate-only \
  --dest /root/OpenYOLO3D/data/scannet_v2_raw
```

Start or resume the official download:

```bash
bash scripts/download_scannet200_val_stream.sh \
  --accept-tos \
  --dest /root/OpenYOLO3D/data/scannet_v2_raw \
  --jobs 3 \
  --min-free-gb 50
```

Verify all 1,872 scene files against official Content-Length values:

```bash
bash scripts/download_scannet200_val_stream.sh \
  --verify-only \
  --dest /root/OpenYOLO3D/data/scannet_v2_raw
```

The downloader is restartable: complete files are skipped and `.part` files
resume through HTTP range requests.

Prepare sampled RGB/depth, poses, mesh and evaluator-format GT:

```bash
cd /root/OpenYOLO3D
RAW_ROOT=/root/OpenYOLO3D/data/scannet_v2_raw \
OUTPUT_ROOT=/root/OpenYOLO3D/data/scannet200 \
CONDA_ENV=openyolo3d \
FRAME_STEP=10 \
JOBS=1 \
MIN_FREE_GB=50 \
bash scripts/prepare_scannet200_val_stream.sh --resume
```

Preparation creates GT because the historical evaluator expects it, but the
no-GT inference stages must not point at or read that directory.  AP requires
separate explicit authorization under the frozen runbook.

## 5. Download public model assets

Review the Qwen, SAM, OpenAI CLIP and Alpha-CLIP licenses first, then run:

```bash
cd /root/OpenYOLO3D
bash scripts/download_fi1_dm_sms1_model_assets.sh \
  --accept-third-party-licenses
```

The script pins:

- Alpha-CLIP source: `ef9262bc539728bf8ef2dfe9c402ae12bbfcd9ff`;
- Segment Anything source: `dca509fe793f601edb92606367a655c15ac00fdf`;
- Qwen2.5-VL-7B-Instruct: `cc594898137f460bfe9f0759e9844b3ce807cfb5`;
- OpenAI CLIP, Alpha-CLIP and SAM weights by SHA-256.

The project-specific frozen model packages remain small Git-tracked assets.

## 6. Assets that cannot be regenerated from Git alone

Downloading and preparing ScanNet reproduces the scene data, but it does not
reproduce the already frozen FI1-D-v3 input plan.  For an exact continuation
of the current joint experiment, copy this controlled asset package from the
old server or project storage:

```text
/root/fi1_d_v3_val312_run_20260822/    approximately 768 MiB
```

This package contains the frozen inference plan, audits, unique-geometry
ledger and historical FI1-D-v3 results.  Its key summary hashes are recorded
in `docs/A100_SERVER_ASSET_MANIFEST_20260829.json`.

If exact historical reconstruction from native candidates is required rather
than copying the compact frozen package, the much larger frozen champion
inputs are also needed:

```text
output/scannet200_first_innovation/frozen_champion/    approximately 31 GiB
```

Do not commit either directory to GitHub.

Copy the compact package after the new instance is reachable:

```bash
rsync -aH --info=progress2 --partial \
  /root/fi1_d_v3_val312_run_20260822/ \
  root@NEW_SERVER:/root/fi1_d_v3_val312_run_20260822/
```

## 7. Configure a new run

Copy the template and replace every `/ABSOLUTE/PATH/...` entry:

```bash
cp configs/repro/fi1_dm_sms1_paths.template.json \
  /root/fi1_dm_sms1_paths.json
```

The new `run_root` must not exist or must be completely empty.  Never reuse
the old failed or completed run root.

Run only the frozen stage authorized for that experiment.  For example, a
read-only preflight is:

```bash
/root/miniconda3/envs/openyolo3d/bin/python \
  tools/run_dm_sms1_fi1_d_v3_val312_pipeline.py \
  --paths /root/fi1_dm_sms1_paths.json \
  --stage preflight
```

Do not run GT/AP merely because the data preparation produced GT.  The AP
stage has a separate authorization gate and is outside normal environment
reproduction.

## 8. What belongs in Git

Commit and review:

- algorithms and audited deployment code;
- preregistration/runbook documents;
- scene identifiers and path templates;
- setup/download/verification scripts;
- package versions and public asset hashes;
- compact status summaries without private paths or credentials.

Do not commit:

- `data/`, `pretrained/`, `output/`;
- raw ScanNet or evaluator GT;
- Qwen/SAM/Alpha-CLIP/DINO/Mask3D weights;
- full generated JSONL/NPZ prediction or embedding ledgers;
- Conda environments, caches, SSH keys, PATs or `.env` files;
- server-local zip files, PLY visualizations and failed-run directories.

This split lets a new A100 instance reproduce code and environment from Git,
download licensed/public assets independently, and copy only the compact
frozen FI1-D-v3 plan required for exact continuation.
