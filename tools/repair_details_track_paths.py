#!/usr/bin/env python3
"""修复原子发布前写入的 Details-sIoU 轨迹点路径，不触碰轨迹内容。"""

import argparse
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track_root", type=Path, required=True)
    args = parser.parse_args()
    root = _resolve(args.track_root)
    changed = 0
    for path in sorted(root.glob("scene*/automatic_tracks.json")):
        payload = json.loads(path.read_text())
        scene_name = path.parent.name
        scene_changed = False
        for track in payload.get("tracks", []):
            old_path = Path(track["points_path"])
            if f".{scene_name}.writing" not in str(old_path):
                continue
            new_path = path.parent / "track_points" / old_path.name
            if not new_path.is_file():
                raise FileNotFoundError(f"缺少已发布的轨迹点文件：{new_path}")
            track["points_path"] = str(new_path)
            scene_changed = True
            changed += 1
        if scene_changed:
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"repaired_track_path_count": changed, "gt_usage": "不读取 GT、不改变轨迹点或关联。"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
