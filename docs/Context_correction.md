# Offline Context Correction

OpenYOLO3D can export uncertain 3D instances as context packets, then read offline
MLLM/VLM semantic corrections back into evaluation.

## 1. Export candidates

```bash
OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mpl TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/jia/anaconda3/envs/openyolo3d/bin/python tools/export_context_candidates.py \
  --dataset_name replica \
  --path_to_3d_masks ./output/replica/replica_masks \
  --output_dir ./output/context_candidates \
  --top_views 3 \
  --max_mask_point_ratio 0.20 \
  --max_bbox_area_ratio 0.65 \
  --min_visible_points 50 \
  --min_label_votes 10
```

Each scene writes `context_candidates.json` with `scene_name`, `mask_id`, best-view
images, label distribution, and a prompt.
The quality filter removes many wall/ceiling/floor-like proposals before API
calls. Use `--include_bad_quality` only for diagnostics.

To avoid rerunning YOLO-World for every experiment, cache the 2D detections:

```bash
OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mpl TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/jia/anaconda3/envs/openyolo3d/bin/python tools/export_context_candidates.py \
  --dataset_name replica \
  --path_to_3d_masks ./output/replica/replica_masks \
  --output_dir ./output/context_candidates \
  --path_to_2d_preds ./output/replica/bboxes_2d \
  --save_2d_preds
```

The same `--path_to_2d_preds` directory can be reused by `run_evaluation.py`,
`tools/export_context_candidates.py`, and the back-projection tool below.

## 2. Write offline correction results

Recommended JSONL format:

```json
{"scene_name":"room2","mask_id":1,"corrected_class_name":"blinds","confidence":0.86,"reason":"window-like horizontal slats in wall context"}
{"scene_name":"room2","mask_id":19,"corrected_class_name":"desk-organizer","confidence":0.74}
```

Supported class fields include `corrected_class_name`, `class_name`,
`mllm_class_name`, `vlm_class_name`, `object_name`, or nested fields under
`mllm_result`, `vlm_result`, `semantic_correction`, `correction`, or `result`.
Class ids can be provided with `corrected_class_id` or `class_id`. Label matching
normalizes case, spaces, underscores, and hyphens, so `wall plug` matches
`wall-plug`. Verbose `answer` strings are allowed, but structured short labels
are preferred.
`decision` can be `change`, `keep`, `unknown`, or `bad_mask`; fusion skips
`unknown` and `bad_mask`.

An augmented candidate JSON is also accepted:

```json
{
  "scene_name": "room2",
  "candidates": [
    {
      "mask_id": 1,
      "mllm_result": {
        "class_name": "blinds",
        "confidence": 0.86
      }
    }
  ]
}
```

## 3. Evaluate with corrections

```bash
OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mpl TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/jia/anaconda3/envs/openyolo3d/bin/python run_evaluation.py \
  --dataset_name replica \
  --path_to_3d_masks ./output/replica/replica_masks \
  --score_threshold 0.20 \
  --context_corrections ./output/context_corrections.jsonl \
  --correction_min_confidence 0.50 \
  --correction_score_policy keep \
  --correction_report_path ./output/context_correction_report.json
```

`--correction_score_policy keep` changes only class labels. Other options are
`replace`, `max`, and `blend`; these only affect AP ranking when
`--use_pred_scores` is enabled or when score thresholding is applied after fusion.

## Gemini Batch Correction

If a Gemini API key is available, generate real VLM corrections directly from
the exported context packets:

```bash
/home/jia/anaconda3/envs/openyolo3d/bin/python tools/run_gemini_context_correction.py \
  --dataset_name replica \
  --context_candidates ./output/context_candidates \
  --output_jsonl ./output/context_corrections_gemini.jsonl \
  --model gemini-2.5-flash
```

The script sends quality metrics plus the best RGB context image, red overlay,
crop, and mask to Gemini, constrains the answer to the dataset vocabulary, and
writes the same JSONL schema consumed by `run_evaluation.py`.

Gemini API notes from the Replica pilot:

- `gemini-2.5-flash` worked with image inputs and returned JSON corrections.
- The observed free-tier limits were 5 requests/minute and then 20
  requests/day for the model/project.
- Use `--sleep 13` to avoid the minute limit, and resume without `--force` to
  skip existing JSONL records.

Pilot results with 30 generated correction records from
`output/context_candidates_replica_gemini_m5`:

| Method | AP | AP50 | AP25 |
| --- | ---: | ---: | ---: |
| threshold baseline | 0.239 | 0.289 | 0.355 |
| Gemini corrections, `change,keep`, score replace | 0.237 | 0.286 | 0.347 |
| Gemini corrections, `change` only, score replace | 0.238 | 0.286 | 0.352 |
| Gemini conservative fusion | 0.237 | 0.288 | 0.354 |
| multi-view BPR, raw 2D score, no `rug` fusion | 0.243 | 0.298 | 0.393 |
| BPR + Gemini corrections, `change` only | 0.242 | 0.294 | 0.389 |
| BPR + Gemini conservative fusion | 0.242 | 0.296 | 0.391 |

The API and fusion path are functional, but this first Gemini pilot does not
improve AP. The main failure mode is that raising very low-score uncertain
masks to high Gemini confidence can introduce false positives. Keep Gemini as a
validated pipeline component, but improve candidate selection and score fusion
before using it as the main result.

The safer fusion options added for this are:

- `--correction_apply_decisions change`: only apply explicit class changes.
- `--correction_apply_min_score` / `--correction_apply_max_score`: avoid
  reviving extremely low-score masks or changing already confident masks.
- `--correction_score_policy boost` with `--correction_score_boost`: cap the
  score increase instead of replacing detector confidence with MLLM confidence.
- `--correction_bad_mask_policy suppress`: lower scores for MLLM-identified bad
  masks.
- `--correction_blocked_classes`: block risky corrected classes such as `rug`
  in a given dataset ablation.

Conservative Replica command:

```bash
OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mpl TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/jia/anaconda3/envs/openyolo3d/bin/python run_evaluation.py \
  --dataset_name replica \
  --path_to_3d_masks ./output/replica/replica_masks \
  --score_threshold 0.20 \
  --path_to_2d_preds ./output/replica/bboxes_2d \
  --context_corrections ./output/context_corrections_gemini_m5.jsonl \
  --correction_min_confidence 0.0 \
  --correction_apply_min_confidence 0.75 \
  --correction_score_policy boost \
  --correction_score_boost 0.22 \
  --correction_apply_decisions change \
  --correction_apply_min_score 0.02 \
  --correction_apply_max_score 0.20 \
  --correction_bad_mask_policy suppress \
  --correction_bad_mask_score 0.0 \
  --correction_blocked_classes rug,panel,blinds
```

## ESAM-Style Back-Projection Candidates

For non-API progress, export 2D detections whose projected 3D points are not
well covered by existing class-agnostic 3D masks:

```bash
OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mpl TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/jia/anaconda3/envs/openyolo3d/bin/python tools/export_backprojection_candidates.py \
  --dataset_name replica \
  --path_to_3d_masks ./output/replica/replica_masks \
  --scene_name room2 \
  --output_dir ./output/backprojection_candidates \
  --path_to_2d_preds ./output/replica/bboxes_2d \
  --save_2d_preds \
  --detection_score_th 0.35 \
  --min_seed_points 80 \
  --max_existing_iou 0.30 \
  --max_seed_in_existing_mask_ratio 0.70 \
  --min_support_views 2 \
  --support_iou_th 0.20 \
  --min_support_visible_points 30 \
  --max_candidates_per_class 3 \
  --max_candidates_per_scene 20
```

Each scene writes `backprojection_candidates.json`, evidence images, and
compressed `seed_points/*.npz` files. This is a proposal-recovery diagnostic
stage: it identifies plausible missed/small objects from strong 2D detections,
but does not change AP until the seed points are grown into reliable 3D masks.
`--min_support_views` enables multi-view consistency filtering: the same 3D seed
region must be supported by same-class 2D detections from multiple views. Small
object classes use a lower seed threshold, while large surface-like classes use
stricter seed and box-area thresholds.

Each candidate also includes a `refinement` routing block. `fast` means the
candidate is suitable for the lightweight bbox-seed path. `sam` means the bbox
projection is geometrically coarse or boundary-sensitive, so optional SAM
refinement should be applied before 3D fusion. `mllm` means the visual category
is semantically ambiguous or low-confidence, so optional VLM/MLLM relabeling is
more appropriate than mask refinement. A candidate can request both `sam` and
`mllm`, but these are separate refinement targets: SAM changes geometry, MLLM
changes semantics.

To fuse the exported candidates as low-confidence 3D proposals during
evaluation:

```bash
OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mpl TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/jia/anaconda3/envs/openyolo3d/bin/python run_evaluation.py \
  --dataset_name replica \
  --path_to_3d_masks ./output/replica/replica_masks \
  --score_threshold 0.20 \
  --path_to_2d_preds ./output/replica/bboxes_2d \
  --backprojection_candidates ./output/backprojection_candidates_replica_m20 \
  --backprojection_min_score 0.40 \
  --backprojection_min_seed_points 80 \
  --backprojection_max_existing_iou 0.30 \
  --backprojection_max_seed_in_existing_mask_ratio 0.70 \
  --backprojection_max_candidates_per_scene 20 \
  --backprojection_score_scale 0.50 \
  --no-backprojection_use_candidate_fusion_score \
  --backprojection_blocked_classes rug \
  --backprojection_report_path ./output/backprojection_candidates_replica_m20/eval_bpr_report.json
```

Replica pilot result with precomputed masks and `--score_threshold 0.20`:

| Method | AP | AP50 | AP25 |
| --- | ---: | ---: | ---: |
| OpenYOLO3D threshold baseline | 0.239 | 0.289 | 0.355 |
| + BPR seed proposals | 0.243 | 0.297 | 0.392 |
| + multi-view BPR seed proposals | 0.243 | 0.298 | 0.392 |
| + multi-view BPR, raw 2D score, no `rug` fusion | 0.243 | 0.298 | 0.393 |
| + BPR with `--backprojection_grow_radius 0.02` | 0.241 | 0.296 | 0.374 |

The seed-only setting is currently better. Small 3D box growth recovers denser
masks but also adds noise, so keep `--backprojection_grow_radius 0.0` as the
default experiment.

Class gating is available for ablations and speed control:
`--backprojection_allowed_classes` keeps only listed classes, and
`--backprojection_blocked_classes` skips listed classes at fusion time without
re-exporting candidates. A compact-object-only ablation was worse on Replica
(`0.242 / 0.296 / 0.386`), so do not use it as the default result.

## Selective SAM Refinement

SAM is optional and only refines selected BPR candidates. It does not replace
YOLO-World: YOLO-World still provides open-vocabulary class labels and boxes,
while SAM tightens the 2D mask before back-projecting seed points.

Prepare SAM v1:

```bash
# Source is expected at:
./_external/segment-anything/segment-anything-main

# ViT-B checkpoint is expected at:
./pretrained/checkpoints/sam_vit_b_01ec64.pth
```

Run top-K SAM refinement:

```bash
OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mpl TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/jia/anaconda3/envs/openyolo3d/bin/python tools/refine_backprojection_candidates_sam.py \
  --dataset_name replica \
  --path_to_3d_masks ./output/replica/replica_masks \
  --candidates_dir ./output/backprojection_candidates_replica_mv_routing_m20 \
  --output_dir ./output/backprojection_candidates_replica_mv_sam_top10 \
  --max_per_scene 10 \
  --path_to_2d_preds ./output/replica/bboxes_2d \
  --sam_checkpoint ./pretrained/checkpoints/sam_vit_b_01ec64.pth \
  --sam_source ./_external/segment-anything/segment-anything-main \
  --sam_model_type vit_b
```

Evaluate refined candidates:

```bash
OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mpl TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/jia/anaconda3/envs/openyolo3d/bin/python run_evaluation.py \
  --dataset_name replica \
  --path_to_3d_masks ./output/replica/replica_masks \
  --score_threshold 0.20 \
  --path_to_2d_preds ./output/replica/bboxes_2d \
  --backprojection_candidates ./output/backprojection_candidates_replica_mv_sam_top10 \
  --backprojection_min_score 0.40 \
  --backprojection_min_seed_points 80 \
  --backprojection_max_existing_iou 0.30 \
  --backprojection_max_seed_in_existing_mask_ratio 0.70 \
  --backprojection_max_candidates_per_scene 20 \
  --backprojection_score_scale 0.50 \
  --no-backprojection_use_candidate_fusion_score \
  --backprojection_report_path ./output/backprojection_candidates_replica_mv_sam_top10/eval_bpr_sam_top10_report.json
```

Replica results:

| Method | Extra SAM candidates | SAM time | AP | AP50 | AP25 |
| --- | ---: | ---: | ---: | ---: | ---: |
| + multi-view BPR, raw 2D score, no `rug` fusion | 0 | 0s | 0.243 | 0.298 | 0.393 |
| + selective SAM top-5 | 40 | 31.6s | 0.243 | 0.298 | 0.393 |
| + selective SAM top-10 | 80 | 34.5s | 0.243 | 0.298 | 0.394 |

The current SAM gain is small. It is best presented as an accurate-mode
extension for boundary cleanup, while the main speed/accuracy result remains
multi-view BPR without mandatory SAM.

## SegDINO3D-Inspired Object Query Re-Scoring

SegDINO3D uses offline 2D object queries (`query2d_feats`, `query2d_pos`) and a
distance-aware cross-attention mask so each 3D query only attends to nearby 2D
object evidence. Full SegDINO3D is not wired into this environment because it
requires a separate training stack, but the same idea is implemented here as an
evaluation-time OpenYOLO3D module:

1. Reuse BPR candidates as lightweight 2D object queries.
2. Back-projected seed points provide the 3D object-query position/support.
3. A 3D mask can only receive evidence from a query when enough seed points fall
   inside that mask.
4. Multi-view support and detector/fusion score weight the evidence.
5. The class is changed only when evidence is strong and has a margin over the
   current/second class.

This is local code in `utils/object_query_rescore.py`; the SegDINO3D repo is
used as a design reference, not imported as a dependency.

Object-query-only evaluation:

```bash
OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mpl TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/jia/anaconda3/envs/openyolo3d/bin/python run_evaluation.py \
  --dataset_name replica \
  --path_to_3d_masks ./output/replica/replica_masks \
  --score_threshold 0.20 \
  --path_to_2d_preds ./output/replica/bboxes_2d \
  --object_query_rescore \
  --object_query_candidates ./output/backprojection_candidates_replica_mv_m20 \
  --object_query_min_candidate_score 0.35 \
  --object_query_min_seed_points 80 \
  --object_query_min_seed_overlap 0.45 \
  --object_query_min_support_views 3 \
  --object_query_min_evidence 0.50 \
  --object_query_min_margin 0.12 \
  --object_query_blocked_classes rug \
  --object_query_report_path ./output/object_query_rescore_replica_mv_m20/eval_oq_only_report.json
```

Stacked with BPR:

```bash
OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mpl TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/jia/anaconda3/envs/openyolo3d/bin/python run_evaluation.py \
  --dataset_name replica \
  --path_to_3d_masks ./output/replica/replica_masks \
  --score_threshold 0.20 \
  --path_to_2d_preds ./output/replica/bboxes_2d \
  --backprojection_candidates ./output/backprojection_candidates_replica_mv_m20 \
  --backprojection_min_score 0.40 \
  --backprojection_min_seed_points 80 \
  --backprojection_max_existing_iou 0.30 \
  --backprojection_max_seed_in_existing_mask_ratio 0.70 \
  --backprojection_max_candidates_per_scene 20 \
  --backprojection_score_scale 0.50 \
  --no-backprojection_use_candidate_fusion_score \
  --backprojection_blocked_classes rug \
  --object_query_rescore \
  --object_query_min_candidate_score 0.35 \
  --object_query_min_seed_points 80 \
  --object_query_min_seed_overlap 0.45 \
  --object_query_min_support_views 3 \
  --object_query_min_evidence 0.50 \
  --object_query_min_margin 0.12 \
  --object_query_blocked_classes rug \
  --object_query_report_path ./output/object_query_rescore_replica_mv_m20/eval_oq_bpr_report.json \
  --backprojection_report_path ./output/object_query_rescore_replica_mv_m20/eval_bpr_report.json
```

Replica results:

| Method | AP | AP50 | AP25 |
| --- | ---: | ---: | ---: |
| threshold baseline | 0.239 | 0.289 | 0.355 |
| object-query re-score only | 0.237 | 0.288 | 0.354 |
| multi-view BPR, raw 2D score, no `rug` fusion | 0.243 | 0.298 | 0.393 |
| multi-view BPR + object-query re-score | 0.241 | 0.297 | 0.391 |

The current object-query substitute does not improve Replica AP. Reports show
only a few class changes, and several are harmful confusions such as
`table -> tablet`, `monitor -> tv-screen`, and `table -> chair`. Keep this as a
negative ablation / future-work module unless stronger object features
(DINO-X/DINOv2/CLIP crop embeddings) are added.

## Cached CLIP Object Feature Ablation

To test the stronger SegDINO3D-style object-level feature idea without changing
the OpenYOLO3D environment, BPR crops can be encoded once with the local
`pretrained/clip-vit-base-patch32` checkpoint:

```bash
OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mpl TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/jia/anaconda3/envs/openyolo3d/bin/python tools/export_clip_object_features.py \
  --dataset_name replica \
  --candidates_dir ./output/backprojection_candidates_replica_mv_m20 \
  --output_dir ./output/clip_object_features_replica_mv_m20 \
  --clip_model_path ./pretrained/clip-vit-base-patch32 \
  --image_field crop_path \
  --batch_size 32 \
  --topk 5
```

This exports 157 cached crop features in about 1 second on the 4090. Evaluation
then only reads JSON and seed files.

CLIP class re-score:

```bash
OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mpl TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/jia/anaconda3/envs/openyolo3d/bin/python run_evaluation.py \
  --dataset_name replica \
  --path_to_3d_masks ./output/replica/replica_masks \
  --score_threshold 0.20 \
  --path_to_2d_preds ./output/replica/bboxes_2d \
  --clip_object_rescore \
  --clip_object_features ./output/clip_object_features_replica_mv_m20 \
  --clip_object_min_seed_points 80 \
  --clip_object_min_seed_overlap 0.45 \
  --clip_object_min_support_views 3 \
  --clip_object_topk_classes 5 \
  --clip_object_min_clip_prob 0.10 \
  --clip_object_min_evidence 0.35 \
  --clip_object_min_margin 0.12 \
  --clip_object_blocked_classes rug \
  --clip_object_report_path ./output/clip_object_rescore_replica_mv_m20/eval_clip_only_report.json
```

CLIP semantic filtering for BPR candidates:

```bash
MPLCONFIGDIR=/tmp/mpl \
/home/jia/anaconda3/envs/openyolo3d/bin/python tools/filter_backprojection_candidates_clip.py \
  --candidates_dir ./output/backprojection_candidates_replica_mv_m20 \
  --clip_features ./output/clip_object_features_replica_mv_m20 \
  --output_dir ./output/backprojection_candidates_replica_mv_m20_clip_filter_p0005 \
  --min_detector_prob 0.005 \
  --blocked_classes rug
```

Replica results:

| Method | AP | AP50 | AP25 |
| --- | ---: | ---: | ---: |
| threshold baseline | 0.239 | 0.289 | 0.355 |
| CLIP re-score only | 0.239 | 0.289 | 0.355 |
| multi-view BPR, raw 2D score, no `rug` fusion | 0.243 | 0.298 | 0.393 |
| BPR + CLIP class re-score | 0.242 | 0.296 | 0.391 |
| BPR + CLIP filter, detector prob >= 0.03 | 0.242 | 0.296 | 0.379 |
| BPR + CLIP filter, detector prob >= 0.01 | 0.242 | 0.296 | 0.382 |
| BPR + CLIP filter, detector prob >= 0.005 | 0.243 | 0.298 | 0.393 |

Conclusion: cached CLIP features are fast and reproducible, but CLIP ViT-B/32
does not improve Replica over the current BPR result. Active class changes cause
near-class mistakes, while filtering removes some geometrically useful
proposals. Keep this as a clean SegDINO3D-inspired ablation; stronger features
such as SigLIP/DINOv2/DINO-X or MLLM context are needed for a main improvement.
