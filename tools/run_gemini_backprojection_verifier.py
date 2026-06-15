import argparse
import json
import os
import os.path as osp
import sys
import time

REPO_ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TOOLS_DIR = osp.dirname(osp.abspath(__file__))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from run_gemini_context_correction import (
    call_gemini,
    extract_response_text,
    image_part,
    load_env_file,
    parse_model_json,
    resolve_path,
)


def iter_candidate_jsons(path):
    path = resolve_path(path)
    if osp.isfile(path):
        yield path
        return
    for root, _, files in os.walk(path):
        for filename in sorted(files):
            if filename == "backprojection_candidates.json":
                yield osp.join(root, filename)


def load_candidates(path):
    with open(path) as f:
        payload = json.load(f)
    scene_name = payload.get("scene_name")
    for candidate in payload.get("candidates", []):
        item = dict(candidate)
        item.setdefault("scene_name", scene_name)
        item["_source_json"] = path
        yield item


def load_existing_keys(path):
    keys = set()
    if not osp.exists(path):
        return keys
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            keys.add((record.get("scene_name"), int(record.get("candidate_id"))))
    return keys


def candidate_priority(candidate):
    return (
        -float(candidate.get("proposal_priority", candidate.get("score", 0.0))),
        -float(candidate.get("score", 0.0)),
        -int(candidate.get("support_view_count", 0)),
        int(candidate.get("candidate_id", 0)),
    )


def candidate_risk_priority(candidate):
    score = float(candidate.get("score", 0.0))
    support_mean_iou = float(candidate.get("support_mean_iou", 0.0))
    seed_overlap = float(candidate.get("seed_in_existing_mask_ratio", 0.0))
    existing_iou = float(candidate.get("best_existing_iou", 0.0))
    box_area_ratio = float(candidate.get("box_area_ratio", 0.0))
    support_views = int(candidate.get("support_view_count", 0))
    risk = 0.0
    risk += max(0.0, 0.65 - score)
    risk += 0.50 * max(0.0, 0.45 - support_mean_iou)
    risk += 0.60 * seed_overlap
    risk += 0.35 * existing_iou
    risk += 0.35 * min(1.0, box_area_ratio / 0.20)
    risk += 0.20 * max(0.0, 30.0 - support_views) / 30.0
    return (
        -risk,
        -float(candidate.get("proposal_priority", candidate.get("score", 0.0))),
        int(candidate.get("candidate_id", 0)),
    )


def build_prompt(candidate):
    refinement = candidate.get("sam_multiview_refine") or candidate.get("sam_refine") or {}
    support = candidate.get("support_views", [])
    support_text = ", ".join(
        f"{item.get('frame_id')}:iou={float(item.get('iou', 0.0)):.2f},score={float(item.get('score', 0.0)):.2f}"
        for item in support[:8]
    )
    return (
        "You are a verifier for open-vocabulary 3D instance segmentation proposals.\n"
        "The images show a candidate object proposal projected from RGB-D frames. "
        "Red pixels/overlays mark the proposed instance region; green boxes mark the candidate box when present.\n\n"
        "Your task is not to reclassify freely. Decide whether this candidate should be kept as a valid "
        "3D instance proposal with the proposed class, or suppressed because it is background, a partial/merged "
        "mask, a duplicate-looking region, or semantically inconsistent with the proposed class.\n\n"
        "Return JSON only with this schema:\n"
        "{"
        "\"decision\": \"keep\" | \"suppress\" | \"uncertain\", "
        "\"confidence\": number, "
        "\"reason\": string"
        "}\n\n"
        "Rules:\n"
        "- Use keep only when the highlighted region plausibly isolates one object instance of the proposed class.\n"
        "- Use suppress when the region mainly covers wall/floor/ceiling/background, a large surface fragment, "
        "multiple objects, or visual evidence contradicts the proposed class.\n"
        "- Use uncertain when evidence is too weak; uncertain will not suppress the candidate.\n"
        "- Do not invent a new class name.\n"
        "- Confidence must be 0.0 to 1.0.\n\n"
        f"Scene: {candidate.get('scene_name')}\n"
        f"Candidate id: {candidate.get('candidate_id')}\n"
        f"Proposed class: {candidate.get('class_name')}\n"
        f"2D detector score: {candidate.get('score')}\n"
        f"Fusion score: {candidate.get('fusion_score')}\n"
        f"Support view count: {candidate.get('support_view_count')}\n"
        f"Support mean IoU: {candidate.get('support_mean_iou')}\n"
        f"Support best IoU: {candidate.get('support_best_iou')}\n"
        f"Seed points: {candidate.get('num_seed_points')}\n"
        f"Seed in existing 3D mask ratio: {candidate.get('seed_in_existing_mask_ratio')}\n"
        f"Best existing 3D mask IoU: {candidate.get('best_existing_iou')}\n"
        f"2D box area ratio: {candidate.get('box_area_ratio')}\n"
        f"SAM refinement status: {refinement.get('status')}\n"
        f"SAM aggregation: {refinement.get('aggregation')}\n"
        f"SAM refined seed points: {refinement.get('num_refined_seed_points')}\n"
        f"Support views: {support_text}\n"
    )


def candidate_parts(candidate):
    parts = [{"text": build_prompt(candidate)}]
    evidence = candidate.get("evidence", {})
    for key in ("context_path", "overlay_path", "crop_path"):
        part = image_part(evidence.get(key))
        if part is not None:
            parts.append(part)

    refinement = candidate.get("sam_multiview_refine") or {}
    for view in refinement.get("views", [])[:2]:
        for key in ("sam_overlay_path", "sam_mask_path"):
            part = image_part(view.get(key))
            if part is not None:
                parts.append(part)
    return parts


def normalize_output(model_record, candidate, model):
    decision = str(model_record.get("decision", "uncertain")).strip().lower()
    if decision not in {"keep", "suppress", "uncertain"}:
        decision = "uncertain"
    try:
        confidence = float(model_record.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))
    return {
        "scene_name": candidate.get("scene_name"),
        "candidate_id": int(candidate.get("candidate_id")),
        "class_name": candidate.get("class_name"),
        "score": candidate.get("score"),
        "fusion_score": candidate.get("fusion_score"),
        "support_view_count": candidate.get("support_view_count"),
        "decision": decision,
        "confidence": confidence,
        "reason": str(model_record.get("reason", "")),
        "source_json": candidate.get("_source_json"),
        "model": model,
    }


def run(args):
    env_values = load_env_file(args.env_file)
    api_key = os.environ.get("GEMINI_API_KEY") or env_values.get("GEMINI_API_KEY")
    if not api_key and not args.dry_run:
        raise RuntimeError("GEMINI_API_KEY was not found. Export it or write it to --env_file.")

    candidates = []
    for json_path in iter_candidate_jsons(args.backprojection_candidates):
        candidates.extend(load_candidates(json_path))
    if args.selection == "risk":
        candidates = sorted(candidates, key=candidate_risk_priority)
    else:
        candidates = sorted(candidates, key=candidate_priority)

    if args.allowed_classes:
        allowed = {item.strip() for item in args.allowed_classes.split(",") if item.strip()}
        candidates = [item for item in candidates if item.get("class_name") in allowed]
    if args.blocked_classes:
        blocked = {item.strip() for item in args.blocked_classes.split(",") if item.strip()}
        candidates = [item for item in candidates if item.get("class_name") not in blocked]
    if args.min_score is not None:
        candidates = [item for item in candidates if float(item.get("score", 0.0)) >= args.min_score]
    if args.max_candidates is not None:
        candidates = candidates[: args.max_candidates]

    output_path = resolve_path(args.output_jsonl)
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    existing = set() if args.force else load_existing_keys(output_path)

    written = 0
    with open(output_path, "a") as out:
        for index, candidate in enumerate(candidates, start=1):
            key = (candidate.get("scene_name"), int(candidate.get("candidate_id")))
            if key in existing:
                continue
            print(
                f"[{index}/{len(candidates)}] {key[0]} candidate {key[1]} "
                f"class={candidate.get('class_name')} score={float(candidate.get('score', 0.0)):.3f}"
            )
            if args.dry_run:
                continue

            response = call_gemini(
                candidate_parts(candidate),
                api_key=api_key,
                model=args.model,
                temperature=args.temperature,
                timeout=args.timeout,
                retries=args.retries,
                transport=args.transport,
            )
            text = extract_response_text(response)
            model_record = parse_model_json(text)
            record = normalize_output(model_record, candidate, args.model)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            written += 1
            if args.sleep > 0:
                time.sleep(args.sleep)

    print(f"Saved {written} Gemini verifier records to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backprojection_candidates", default="./output/backprojection_candidates_replica_mv_m20")
    parser.add_argument("--output_jsonl", default="./output/backprojection_verifier_gemini.jsonl")
    parser.add_argument("--env_file", default="/tmp/openyolo3d_gemini.env")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--max_candidates", default=None, type=int)
    parser.add_argument("--selection", default="priority", choices=["priority", "risk"])
    parser.add_argument("--min_score", default=None, type=float)
    parser.add_argument("--allowed_classes", default=None)
    parser.add_argument("--blocked_classes", default=None)
    parser.add_argument("--temperature", default=0.0, type=float)
    parser.add_argument("--timeout", default=90, type=int)
    parser.add_argument("--retries", default=2, type=int)
    parser.add_argument("--transport", default="curl", choices=["curl", "urllib"])
    parser.add_argument("--sleep", default=1.0, type=float)
    parser.add_argument("--force", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dry_run", default=False, action=argparse.BooleanOptionalAction)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
