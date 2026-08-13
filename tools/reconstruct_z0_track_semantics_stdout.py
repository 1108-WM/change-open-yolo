#!/usr/bin/env python3
"""Isolated, stdout-only reconstruction of frozen track top-1 semantics for Z0."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.diagnose_z0_open_vocab_oracle_gt import _reconstruct_track_semantics_in_memory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--stream-records-root", type=Path, required=True)
    parser.add_argument("--prepared-dataset-root", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    args = parser.parse_args()
    tracks = json.loads((
        args.stream_records_root / args.scene / "d2b_tracks_filtered" / args.scene / "automatic_tracks.json"
    ).read_text()).get("tracks", [])
    rows = _reconstruct_track_semantics_in_memory(
        args.scene, tracks, args.stream_records_root, args.prepared_dataset_root, args.config_path,
    )
    print(json.dumps(rows, ensure_ascii=False, separators=(",", ":")))


if __name__ == "__main__":
    main()
