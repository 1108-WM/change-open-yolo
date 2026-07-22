import argparse
import json
import random
from pathlib import Path


def _list_scenes(input_root):
    root = Path(input_root)
    scenes = sorted(path.name for path in root.iterdir() if path.is_dir() and path.name.startswith("scene"))
    if not scenes:
        raise ValueError(f"No scene directories found under {root}")
    return scenes


def _make_splits(scenes, seed, small_size=48, large_size=96, confirm_size=96):
    required = int(large_size) + int(confirm_size)
    if int(small_size) > int(large_size):
        raise ValueError("small_size must be <= large_size")
    if len(scenes) < required:
        raise ValueError(f"Need at least {required} scenes, got {len(scenes)}")
    shuffled = list(scenes)
    random.Random(int(seed)).shuffle(shuffled)
    large = sorted(shuffled[: int(large_size)])
    small = sorted(shuffled[: int(small_size)])
    confirm = sorted(shuffled[int(large_size) : int(large_size) + int(confirm_size)])
    return {
        "even48": small,
        "even96": large,
        "odd96": confirm,
        "sample_order": shuffled,
    }


def _write_lines(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def generate(args):
    scenes = _list_scenes(args.input_root)
    splits = _make_splits(
        scenes,
        seed=args.seed,
        small_size=args.small_size,
        large_size=args.large_size,
        confirm_size=args.confirm_size,
    )
    output_dir = Path(args.output_dir)
    _write_lines(output_dir / "even48.txt", splits["even48"])
    _write_lines(output_dir / "even96.txt", splits["even96"])
    _write_lines(output_dir / "odd96.txt", splits["odd96"])

    manifest = {
        "scope": "random_scene_splits_for_diagnostic_subsets",
        "input_root": str(args.input_root),
        "output_dir": str(args.output_dir),
        "seed": int(args.seed),
        "total_available_scenes": len(scenes),
        "small_size": int(args.small_size),
        "large_size": int(args.large_size),
        "confirm_size": int(args.confirm_size),
        "files": {
            "even48.txt": {
                "role": "random_48_scene_screening_subset",
                "count": len(splits["even48"]),
                "is_subset_of_even96": set(splits["even48"]).issubset(splits["even96"]),
            },
            "even96.txt": {
                "role": "random_96_scene_expansion_subset",
                "count": len(splits["even96"]),
            },
            "odd96.txt": {
                "role": "random_96_scene_disjoint_confirmation_subset",
                "count": len(splits["odd96"]),
                "is_disjoint_from_even96": set(splits["odd96"]).isdisjoint(splits["even96"]),
            },
        },
        "note": (
            "Filenames keep the historical even48/even96/odd96 names for script compatibility; "
            "the contents are seeded random splits generated from the current data/scannet200 scenes."
        ),
    }
    (output_dir / "random_scene_splits_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args():
    parser = argparse.ArgumentParser(description="Generate reproducible random ScanNet200 scene splits.")
    parser.add_argument("--input_root", default="data/scannet200")
    parser.add_argument("--output_dir", default="output/scannet200/scene_splits")
    parser.add_argument("--seed", default=20260718, type=int)
    parser.add_argument("--small_size", default=48, type=int)
    parser.add_argument("--large_size", default=96, type=int)
    parser.add_argument("--confirm_size", default=96, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())
