import argparse
import csv
import json
import os
import os.path as osp
import re
import subprocess
import sys
import time


REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
DEFAULT_PYTHON = "/home/jia/anaconda3/envs/openyolo3d/bin/python"
AVERAGE_RE = re.compile(r"^average\s*:\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*$", re.MULTILINE)


def _parse_float_list(value):
    return [float(item.strip()) for item in str(value).split(",") if item.strip()]


def _parse_int_list(value):
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def _float_token(value):
    return f"{float(value):.3g}".replace(".", "p")


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _write_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _run_command(command, env, log_prefix, dry_run=False):
    started = time.time()
    record = {
        "command": command,
        "elapsed_seconds": 0.0,
        "returncode": None,
        "stdout_path": f"{log_prefix}.stdout.txt",
        "stderr_path": f"{log_prefix}.stderr.txt",
    }
    if dry_run:
        record["returncode"] = 0
        record["dry_run"] = True
        with open(record["stdout_path"], "w") as f:
            f.write("DRY RUN\n")
            f.write(" ".join(command) + "\n")
        with open(record["stderr_path"], "w") as f:
            f.write("")
        return record

    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    record["elapsed_seconds"] = time.time() - started
    record["returncode"] = proc.returncode
    with open(record["stdout_path"], "w") as f:
        f.write(proc.stdout)
    with open(record["stderr_path"], "w") as f:
        f.write(proc.stderr)
    return record


def _parse_eval_metrics(stdout_path):
    if not osp.exists(stdout_path):
        return {}
    with open(stdout_path) as f:
        text = f.read()
    matches = AVERAGE_RE.findall(text)
    if not matches:
        return {}
    ap, ap50, ap25 = matches[-1]
    return {"ap": float(ap), "ap50": float(ap50), "ap25": float(ap25)}


def _experiment_name(exp):
    if exp.get("name"):
        return exp["name"]
    return (
        f"s{exp['frame_stride']}"
        f"_d{_float_token(exp['detection_score_th'])}"
        f"_mi{_float_token(exp['merge_iou'])}"
        f"_m{exp['max_candidates_per_scene']}"
    )


def _one_factor_experiments(args):
    base = {
        "frame_stride": args.base_frame_stride,
        "detection_score_th": args.base_detection_score_th,
        "merge_iou": args.base_merge_iou,
        "max_candidates_per_scene": args.base_max_candidates_per_scene,
    }
    experiments = []
    for value in args.frame_strides:
        item = dict(base)
        item["frame_stride"] = value
        experiments.append(item)
    for value in args.detection_score_ths:
        item = dict(base)
        item["detection_score_th"] = value
        experiments.append(item)
    for value in args.merge_ious:
        item = dict(base)
        item["merge_iou"] = value
        experiments.append(item)
    for value in args.max_candidates_per_scene_values:
        item = dict(base)
        item["max_candidates_per_scene"] = value
        experiments.append(item)

    unique = {}
    for item in experiments:
        unique[_experiment_name(item)] = item
    return [unique[name] for name in sorted(unique)]


def _full_grid_experiments(args):
    experiments = []
    for frame_stride in args.frame_strides:
        for detection_score_th in args.detection_score_ths:
            for merge_iou in args.merge_ious:
                for max_candidates in args.max_candidates_per_scene_values:
                    experiments.append(
                        {
                            "frame_stride": frame_stride,
                            "detection_score_th": detection_score_th,
                            "merge_iou": merge_iou,
                            "max_candidates_per_scene": max_candidates,
                        }
                    )
    return experiments


def _base_env(args):
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": str(args.omp_num_threads),
            "MPLCONFIGDIR": args.mplconfigdir,
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
        }
    )
    return env


def _export_command(args, exp, output_dir):
    return [
        args.python,
        "tools/export_sam_fused_proposals.py",
        "--dataset_name",
        args.dataset_name,
        "--path_to_3d_masks",
        args.path_to_3d_masks,
        "--output_dir",
        output_dir,
        "--sam_checkpoint",
        args.sam_checkpoint,
        "--sam_source",
        args.sam_source,
        "--detection_score_th",
        str(exp["detection_score_th"]),
        "--min_seed_points",
        str(args.min_seed_points),
        "--max_box_area_ratio",
        str(args.max_box_area_ratio),
        "--frame_stride",
        str(exp["frame_stride"]),
        "--max_detections_per_frame",
        str(args.max_detections_per_frame),
        "--merge_iou",
        str(exp["merge_iou"]),
        "--max_candidates_per_scene",
        str(exp["max_candidates_per_scene"]),
        "--blocked_classes",
        args.blocked_classes,
        "--path_to_2d_preds",
        args.path_to_2d_preds,
    ]


def _eval_common_args(args, output_csv):
    command = [
        args.python,
        "run_evaluation.py",
        "--dataset_name",
        args.dataset_name,
        "--path_to_3d_masks",
        args.path_to_3d_masks,
        "--score_threshold",
        str(args.score_threshold),
        "--path_to_2d_preds",
        args.path_to_2d_preds,
        "--eval_output_file",
        output_csv,
    ]
    return command


def _eval_command(args, candidate_paths, max_candidates, report_path, output_csv):
    command = _eval_common_args(args, output_csv)
    if candidate_paths:
        command.extend(
            [
                "--backprojection_candidates",
                candidate_paths,
                "--backprojection_min_score",
                str(args.backprojection_min_score),
                "--backprojection_min_seed_points",
                str(args.backprojection_min_seed_points),
                "--backprojection_max_existing_iou",
                str(args.backprojection_max_existing_iou),
                "--backprojection_max_seed_in_existing_mask_ratio",
                str(args.backprojection_max_seed_in_existing_mask_ratio),
                "--backprojection_max_candidates_per_scene",
                str(max_candidates),
                "--backprojection_score_scale",
                str(args.backprojection_score_scale),
                "--no-backprojection_use_candidate_fusion_score",
                "--backprojection_blocked_classes",
                args.blocked_classes,
                "--backprojection_report_path",
                report_path,
            ]
        )
        if args.backprojection_quality_sort:
            command.append("--backprojection_quality_sort")
        if args.backprojection_source_priorities:
            command.extend(["--backprojection_source_priorities", args.backprojection_source_priorities])
        if args.backprojection_source_max_candidates:
            command.extend(["--backprojection_source_max_candidates", args.backprojection_source_max_candidates])
        if args.backprojection_source_score_scales:
            command.extend(["--backprojection_source_score_scales", args.backprojection_source_score_scales])
    return command


def _summarize_export(output_dir):
    summary_path = osp.join(output_dir, "sam_fused_proposals_summary.json")
    if not osp.exists(summary_path):
        return {}
    summary = _load_json(summary_path)
    scenes = summary.get("scenes", [])
    return {
        "export_summary_path": summary_path,
        "export_elapsed_seconds": float(summary.get("elapsed_seconds", 0.0)),
        "raw_observations": sum(int(item.get("raw_observations", 0)) for item in scenes),
        "num_candidates": sum(int(item.get("num_candidates", 0)) for item in scenes),
        "num_scenes": len(scenes),
    }


def _append_csv(csv_path, records):
    fieldnames = [
        "name",
        "variant",
        "ap",
        "ap50",
        "ap25",
        "eval_elapsed_seconds",
        "export_elapsed_seconds",
        "raw_observations",
        "num_candidates",
        "frame_stride",
        "detection_score_th",
        "merge_iou",
        "max_candidates_per_scene",
        "candidate_paths",
        "report_path",
        "stdout_path",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key) for key in fieldnames}
            writer.writerow(row)


def _run_eval_variant(args, env, records_jsonl, name, variant, exp, export_info, candidate_paths, max_candidates):
    report_path = osp.join(args.output_root, "reports", f"{name}_{variant}_report.json")
    output_csv = osp.join(args.output_root, "eval_csv", f"{name}_{variant}.csv")
    log_prefix = osp.join(args.output_root, "logs", f"{name}_{variant}_eval")
    command = _eval_command(args, candidate_paths, max_candidates, report_path, output_csv)
    result = _run_command(command, env, log_prefix, dry_run=args.dry_run)
    metrics = _parse_eval_metrics(result["stdout_path"])
    record = {
        "name": name,
        "variant": variant,
        **exp,
        **export_info,
        **metrics,
        "eval_elapsed_seconds": result["elapsed_seconds"],
        "candidate_paths": candidate_paths,
        "report_path": report_path if candidate_paths else None,
        "eval_output_file": output_csv,
        "stdout_path": result["stdout_path"],
        "stderr_path": result["stderr_path"],
        "returncode": result["returncode"],
    }
    _write_jsonl(records_jsonl, record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=DEFAULT_PYTHON if osp.exists(DEFAULT_PYTHON) else sys.executable)
    parser.add_argument("--dataset_name", default="replica", choices=["replica", "scannet200"])
    parser.add_argument("--path_to_3d_masks", default="./output/replica/replica_masks")
    parser.add_argument("--path_to_2d_preds", default="./output/replica/bboxes_2d")
    parser.add_argument("--sam_checkpoint", default="./pretrained/checkpoints/sam_vit_b_01ec64.pth")
    parser.add_argument("--sam_source", default="./_external/segment-anything/segment-anything-main")
    parser.add_argument("--bpr_candidates", default="./output/backprojection_candidates_replica_mv_m20")
    parser.add_argument("--output_root", default="./output/sam_fused_ablation_replica")
    parser.add_argument("--mode", default="both", choices=["export", "eval", "both"])
    parser.add_argument("--grid", default="one_factor", choices=["one_factor", "full"])
    parser.add_argument("--existing_export_dir", default=None, help="Evaluate one existing SAM-fused proposal directory instead of generated sweep dirs")
    parser.add_argument("--existing_export_name", default=None, help="Display name for --existing_export_dir")
    parser.add_argument("--skip_existing_export", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--include_baselines", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dry_run", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--omp_num_threads", default=8, type=int)
    parser.add_argument("--mplconfigdir", default="/tmp/mpl")

    parser.add_argument("--frame_strides", default="3,5,10", type=_parse_int_list)
    parser.add_argument("--detection_score_ths", default="0.35,0.45,0.55", type=_parse_float_list)
    parser.add_argument("--merge_ious", default="0.05,0.15,0.30", type=_parse_float_list)
    parser.add_argument("--max_candidates_per_scene_values", default="10,20,30", type=_parse_int_list)
    parser.add_argument("--base_frame_stride", default=5, type=int)
    parser.add_argument("--base_detection_score_th", default=0.45, type=float)
    parser.add_argument("--base_merge_iou", default=0.15, type=float)
    parser.add_argument("--base_max_candidates_per_scene", default=20, type=int)

    parser.add_argument("--min_seed_points", default=80, type=int)
    parser.add_argument("--max_box_area_ratio", default=0.30, type=float)
    parser.add_argument("--max_detections_per_frame", default=8, type=int)
    parser.add_argument("--blocked_classes", default="rug")

    parser.add_argument("--score_threshold", default=0.20, type=float)
    parser.add_argument("--backprojection_min_score", default=0.40, type=float)
    parser.add_argument("--backprojection_min_seed_points", default=80, type=int)
    parser.add_argument("--backprojection_max_existing_iou", default=0.30, type=float)
    parser.add_argument("--backprojection_max_seed_in_existing_mask_ratio", default=0.70, type=float)
    parser.add_argument("--backprojection_score_scale", default=0.50, type=float)
    parser.add_argument("--backprojection_quality_sort", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--backprojection_source_priorities", default=None)
    parser.add_argument("--backprojection_source_max_candidates", default=None)
    parser.add_argument("--backprojection_source_score_scales", default=None)
    parser.add_argument("--combined_eval_max_candidates_per_scene", default=30, type=int)
    parser.add_argument("--bpr_eval_max_candidates_per_scene", default=20, type=int)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    os.makedirs(osp.join(args.output_root, "exports"), exist_ok=True)
    os.makedirs(osp.join(args.output_root, "logs"), exist_ok=True)
    os.makedirs(osp.join(args.output_root, "reports"), exist_ok=True)
    os.makedirs(osp.join(args.output_root, "eval_csv"), exist_ok=True)

    env = _base_env(args)
    records_jsonl = osp.join(args.output_root, "results.jsonl")
    if osp.exists(records_jsonl):
        os.remove(records_jsonl)

    if args.existing_export_dir is not None:
        experiments = [
            {
                "name": args.existing_export_name or osp.basename(osp.normpath(args.existing_export_dir)),
                "frame_stride": args.base_frame_stride,
                "detection_score_th": args.base_detection_score_th,
                "merge_iou": args.base_merge_iou,
                "max_candidates_per_scene": args.base_max_candidates_per_scene,
                "output_dir": args.existing_export_dir,
            }
        ]
    else:
        experiments = _full_grid_experiments(args) if args.grid == "full" else _one_factor_experiments(args)
    records = []

    if args.include_baselines and args.mode in ("eval", "both"):
        empty_exp = {
            "frame_stride": None,
            "detection_score_th": None,
            "merge_iou": None,
            "max_candidates_per_scene": None,
        }
        records.append(
            _run_eval_variant(
                args,
                env,
                records_jsonl,
                "baseline",
                "baseline",
                empty_exp,
                {},
                None,
                None,
            )
        )
        records.append(
            _run_eval_variant(
                args,
                env,
                records_jsonl,
                "bpr",
                "bpr_only",
                empty_exp,
                {},
                args.bpr_candidates,
                args.bpr_eval_max_candidates_per_scene,
            )
        )

    for exp in experiments:
        name = _experiment_name(exp)
        output_dir = exp.get("output_dir") or osp.join(args.output_root, "exports", name)
        summary_path = osp.join(output_dir, "sam_fused_proposals_summary.json")
        if args.existing_export_dir is not None:
            export_info = _summarize_export(output_dir)
        elif args.mode in ("export", "both"):
            if args.skip_existing_export and osp.exists(summary_path):
                export_info = _summarize_export(output_dir)
            else:
                command = _export_command(args, exp, output_dir)
                result = _run_command(
                    command,
                    env,
                    osp.join(args.output_root, "logs", f"{name}_export"),
                    dry_run=args.dry_run,
                )
                export_info = _summarize_export(output_dir)
                export_info.setdefault("export_elapsed_seconds", result["elapsed_seconds"])
                export_info["export_returncode"] = result["returncode"]
                export_info["export_stdout_path"] = result["stdout_path"]
                export_info["export_stderr_path"] = result["stderr_path"]
        else:
            export_info = _summarize_export(output_dir)

        if args.mode in ("eval", "both"):
            records.append(
                _run_eval_variant(
                    args,
                    env,
                    records_jsonl,
                    name,
                    "sam_only",
                    exp,
                    export_info,
                    output_dir,
                    exp["max_candidates_per_scene"],
                )
            )
            records.append(
                _run_eval_variant(
                    args,
                    env,
                    records_jsonl,
                    name,
                    "sam_plus_bpr",
                    exp,
                    export_info,
                    f"{output_dir},{args.bpr_candidates}",
                    args.combined_eval_max_candidates_per_scene,
                )
            )

    summary_csv = osp.join(args.output_root, "results.csv")
    _append_csv(summary_csv, records)
    with open(osp.join(args.output_root, "sweep_config.json"), "w") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)
    print(f"Saved {len(records)} evaluation records to {summary_csv}")


if __name__ == "__main__":
    main()
