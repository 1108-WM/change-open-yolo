# Remote-server setup

Updated: 2026-08-13

This guide separates a lightweight checkout test from a full GPU inference
setup.  Cloning GitHub provides the current source code, tests, project-status
documents, compact result summaries, the first innovation's frozen champion
model package, and the small Z3/Z6f project-specific calibration models.  It
does not provide licensed datasets, third-party foundation-model weights, or
generated scene-level ledgers.

## 1. Clone the current fork

```bash
git clone --recurse-submodules https://github.com/1108-WM/change-open-yolo.git
cd change-open-yolo
git status --short --branch
python3 tools/verify_repository_checkout.py
```

The expected branch is `main`.  If an older checkout already exists on the
server, preserve any server-local work before updating, then run:

```bash
git fetch origin
git switch main
git pull --ff-only origin main
git submodule sync --recursive
git submodule update --init --recursive
python3 tools/verify_repository_checkout.py
```

Do not use `git reset --hard` or `git clean` on a server checkout that may
contain local data or results.

## 2. Lightweight code test

This check does not require ScanNet, CUDA, Mask3D weights, YOLO-World weights,
Alpha-CLIP, DINOv2, or Qwen:

```bash
python3 -m compileall -q tools tests utils evaluate
python3 tools/verify_repository_checkout.py
```

After creating the project environment, pure unit tests can be run with:

```bash
python -m pytest -q tests
```

Some tests exercise optional GPU or external-model paths and will require the
corresponding dependencies.  Start with targeted tests for the component being
changed if the full environment is not yet installed.

## 3. Base Open-YOLO 3D environment

Follow `docs/Installation.md`.  The repository pins the two Mask3D third-party
repositories through `.gitmodules`:

- MinkowskiEngine at `02fc608bea4c0549b0a7b00ca1bf15dee4a0b228`
- ScanNet tools at `3e5726500896748521a6ceb81271b0f5b2c0e7d2`

The original environment targets Python 3.10, PyTorch 1.12.1, CUDA 11.3, and
MinkowskiEngine 0.5.x.  Match the remote GPU driver before choosing a CUDA
toolkit.  Do not copy a local `.venvs/` directory between machines.

## 4. Assets that must be downloaded separately

The following are intentionally not stored in GitHub:

| Asset | Expected local location | Reason |
|---|---|---|
| ScanNet200/Replica scenes and GT | `data/` | licensed/large dataset |
| Mask3D and YOLO-World weights | `pretrained/checkpoints/` | about 1.5 GB |
| Alpha-CLIP weights/repository | `pretrained/alpha_clip/`, `_external/AlphaCLIP/` | third-party model |
| DINOv2 weights | `pretrained/checkpoints/dinov2_vits14_pretrain.pth` | large third-party weight |
| SAM/SAM2/YOLOE/GroundingDINO assets | `pretrained/`, `_external/` | optional large dependencies |
| Qwen2.5-VL model cache | Hugging Face cache or a server-local path | multi-GB model |
| Scene-level candidates and experiment plans | `output/` | generated data, often tens of GB |
| Full diagnostic ledgers/embeddings | `docs/diagnostics/` | generated JSONL/NPZ/CSV data |

Use the original download helpers where applicable:

```bash
sh scripts/get_checkpoints.sh
sh scripts/get_class_agn_masks.sh
```

Those helpers cover the original Open-YOLO 3D assets only.  They do not fetch
the later Alpha-CLIP, DINOv2, Qwen, or generated experiment ledgers.  ScanNet
must be obtained under its own terms.  If exact historical experiment replay
is required, copy the relevant `output/` and `docs/diagnostics/` directories
from controlled project storage rather than committing them to Git.

## 5. Small frozen assets included in Git

The checkout verifier checks these files and their SHA-256 hashes:

- First innovation champion package:
  `release_assets/champion_geometry/model_package.pkl`
- Z3 semantic reliability model:
  `release_assets/z3_semantic_reliability/c_joint_yolo_alpha.joblib`
- Z6f selector and improvement gate:
  `release_assets/z6f_inference_bundle/`

These are sufficient to preserve the frozen learned parameters.  They are not
sufficient by themselves to reproduce AP: inference also requires prepared
scene data and the upstream no-GT ledgers described by each tool's CLI.

## 6. Current experimental guardrails

Before running research experiments, read `新开对话阅读内容.md`.  In particular:

- Z0-Z6f and the safety60 transfers are complete and must not be blindly rerun.
- safety60 cannot be used for training or parameter selection.
- even48 and test60 remain frozen.
- Current next work is system/ablation and paper-story analysis unless a new
  official100 structural branch is explicitly preregistered.

## 7. Machine-local configuration

Pass dataset/model paths through command-line arguments or environment
variables.  Do not commit credentials, `.env` files, absolute home-directory
paths, model caches, or server-specific symlinks.  A clean clone should remain
usable regardless of the remote username or workspace path.
