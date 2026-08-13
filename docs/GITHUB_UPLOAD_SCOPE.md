# GitHub upload scope

Updated: 2026-08-13

This file records the publication decision for the current research checkout.

## Included in GitHub

- All project-owned Python and shell source under `tools/`, `utils/`,
  `evaluate/`, and the top-level entry points.
- Unit tests under `tests/`.
- Current status and architecture documents:
  `CURRENT_EXPERIMENT_STATUS.md`, `新开对话阅读内容.md`, and `资料/`.
- Installation, data preparation, remote setup, and cleanup manifests.
- Pinned `.gitmodules` entries required to reconstruct Mask3D third-party
  dependencies.
- Compact summaries for the current frozen Z0-Z6f and safety60 conclusions.
- Small project-specific frozen models and metadata under `release_assets/`.

## Excluded from GitHub

- `data/`: ScanNet/Replica RGB-D, point clouds, and GT.
- `output/`: generated candidate masks, per-scene plans, caches, and OOF rows.
- Raw `docs/diagnostics/` ledgers such as JSONL, NPZ, CSV, embeddings, crops,
  and visualizations.  Selected compact summaries are copied into
  `release_summaries/`.
- `pretrained/` third-party weights, except configuration files already
  versioned.  Project-specific small models are published under
  `release_assets/` instead of mixing them with local model caches.
- `_external/`, `.venvs/`, Python/build caches, local agent state, backups,
  paper PDFs, and visual-review images.
- API keys, `.env` files, SSH material, Hugging Face caches, and absolute
  machine-specific symlinks.

The byte-frozen champion pickle and the official100 split manifest retain a
few historical absolute paths inside provenance metadata.  They are not
executed or used to locate remote-server inputs.  They remain unchanged so
their published SHA-256 hashes continue to match the audited experiment
artifacts; all runnable scripts and setup instructions use repository-relative
paths or explicit CLI arguments.

## Why large outputs are excluded

Git is the source and metadata transport, not the experiment-object store.
Many current ledgers are 20-300 MB each and the local diagnostics collection is
far larger than a normal source checkout.  Committing them would slow every
clone, duplicate licensed or derived data, and still not make the experiments
self-contained because the original ScanNet assets and foundation-model
weights are separately required.

For an exact historical replay, archive the required generated directories in
controlled object storage with checksums.  The Git commit, status documents,
release summaries, and `tools/verify_repository_checkout.py` then identify the
code and frozen model version used with that archive.
