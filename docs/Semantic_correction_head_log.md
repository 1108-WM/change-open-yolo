# Object-level Semantic Correction Head Log

## Goal

Try a second innovation direction after SAM-fused proposal generation:

```text
YOLO-World gives the initial class
object-level correction head decides whether the class should be changed
```

This is different from the previous scoring head. The previous head only learned whether a prediction is reliable. This one tries to predict the object class directly for masks/proposals that overlap a ground-truth object.

## Implemented

Script:

```bash
tools/train_semantic_correction_head.py
```

Training data:

```bash
output/semantic_fusion_dataset_replica_s5_m30_bpr/features.jsonl
```

Supervision:

- Use records with `best_iou >= 0.25`
- Target class is `gt_pred_class_id`
- Train split: `office0` to `office4`
- Validation split: `room0` to `room2`

## Multiclass Correction Result

Command:

```bash
/home/jia/anaconda3/envs/openyolo3d/bin/python tools/train_semantic_correction_head.py \
  --dataset_jsonl ./output/semantic_fusion_dataset_replica_s5_m30_bpr/features.jsonl \
  --output_dir ./output/semantic_correction_head_replica_s5_m30_bpr_iou25_mlp64 \
  --min_iou 0.25 \
  --hidden_dim 64 \
  --dropout 0.15 \
  --epochs 1200 \
  --lr 0.005 \
  --weight_decay 0.005
```

Validation class accuracy:

- Original YOLO-World/OpenYOLO3D class: `0.731`
- Multiclass correction head top-1: `0.321`

Linear variant:

- Original class: `0.731`
- Linear correction head top-1: `0.218`

Conclusion: direct multiclass classification over the Replica class list is too unstable with the current small object-level dataset. It overfits the training scenes and does not generalize from office scenes to room scenes.

## Evidence Diagnostic

On validation positive records (`room0/room1/room2`, `best_iou >= 0.25`):

- Original class accuracy: `0.731`
- Object-evidence top-1 available for 46 records, accuracy `0.478`
- SAM-fused evidence top-1 available for 12 records, accuracy `0.583`
- BPR evidence top-1 available for 45 records, accuracy `0.467`

Important finding:

- Existing object evidence had `0` cases where it would fix an originally wrong validation sample.
- It had many cases where it would hurt originally correct samples.

This explains why a correction head cannot currently improve by only using the exported object evidence features.

## CLIP Object Feature Check

A conservative CLIP object rescore/correction run was also tested with the existing cached CLIP object features:

```bash
--clip_object_rescore
--clip_object_features ./output/clip_object_features_replica_mv_m20
--clip_object_min_seed_overlap 0.45
--clip_object_min_support_views 2
--clip_object_min_evidence 0.45
--clip_object_min_margin 0.15
```

Result:

```text
AP 0.247 / AP50 0.310 / AP25 0.394
```

This is slightly below the current best SAM-fused+BPR result:

```text
AP 0.247 / AP50 0.311 / AP25 0.395
```

## Current Conclusion

The object-level semantic correction idea is reasonable, but the current feature set is not enough.

What does not work yet:

- Direct multiclass correction head from current handcrafted features
- Reusing existing BPR/SAM object evidence as the new label
- Conservative CLIP object rescore with current crop cache

Most likely reason:

- The baseline YOLO-World class is already correct for most positive validation records.
- The remaining wrong cases require stronger visual semantics than the current cached features provide.
- Replica has only 8 scenes, so trainable semantic correction over many classes overfits easily.

Next useful direction:

1. Export better object crops from the actual 3D mask/proposal across top views, not just BPR candidate crops.
2. Aggregate CLIP/DINO probabilities over multiple visible views per 3D object.
3. Only allow corrections inside a small confusion set, such as `pillow/cushion/blanket/cloth` or `vase/pot/bottle`.
4. Apply correction only when the new visual evidence is strong and the original YOLO-World evidence is weak.

## Follow-up: Multi-view 3D-object CLIP Correction

Implemented:

```bash
tools/export_multiview_object_clip_features.py
tools/evaluate_multiview_object_clip_correction.py
```

The new exporter does not reuse BPR candidate crops. It starts from the final 3D predictions used by evaluation, selects the top visible RGB views for each 3D mask/proposal, crops the projected object region, and aggregates CLIP probabilities over views.

Full Replica export:

```bash
OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mpl TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/jia/anaconda3/envs/openyolo3d/bin/python tools/export_multiview_object_clip_features.py \
  --dataset_name replica \
  --path_to_3d_masks ./output/replica/replica_masks \
  --path_to_2d_preds ./output/replica/bboxes_2d \
  --backprojection_candidates ./output/sam_fused_proposals_replica_s5_m30_prefilter,./output/backprojection_candidates_replica_mv_m20 \
  --backprojection_min_score 0.40 \
  --backprojection_min_seed_points 80 \
  --backprojection_max_existing_iou 0.30 \
  --backprojection_max_seed_in_existing_mask_ratio 0.70 \
  --backprojection_max_candidates_per_scene 30 \
  --backprojection_score_scale 0.50 \
  --no-backprojection_use_candidate_fusion_score \
  --backprojection_blocked_classes rug \
  --backprojection_source_score_scales sam_fused=1.2,bpr=1.0 \
  --output_dir ./output/multiview_object_clip_replica_s5_m30_bpr \
  --top_views 3 \
  --min_visible_points 50 \
  --max_bbox_area_ratio 0.60 \
  --aggregate mean \
  --device cuda
```

Export result:

- 300 object records
- 895 view crops
- Output size: about 13 MB

Naively applying CLIP top-1 is still harmful:

- Validation positive records, original class accuracy: `0.731`
- Multi-view CLIP top-1 accuracy: `0.397`
- Helpful corrections: 4
- Harmful corrections: 30

However, a very conservative class-specific correction was effective:

```text
Only allow correction to shelf
Only if original base score <= 0.35
CLIP confidence >= 0.30
CLIP margin >= 0.10
CLIP gain over current class >= 0.10
```

Command:

```bash
OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mpl TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/jia/anaconda3/envs/openyolo3d/bin/python tools/evaluate_multiview_object_clip_correction.py \
  --dataset_name replica \
  --path_to_3d_masks ./output/replica/replica_masks \
  --path_to_2d_preds ./output/replica/bboxes_2d \
  --backprojection_candidates ./output/sam_fused_proposals_replica_s5_m30_prefilter,./output/backprojection_candidates_replica_mv_m20 \
  --backprojection_min_score 0.40 \
  --backprojection_min_seed_points 80 \
  --backprojection_max_existing_iou 0.30 \
  --backprojection_max_seed_in_existing_mask_ratio 0.70 \
  --backprojection_max_candidates_per_scene 30 \
  --backprojection_score_scale 0.50 \
  --no-backprojection_use_candidate_fusion_score \
  --backprojection_blocked_classes rug \
  --backprojection_source_score_scales sam_fused=1.2,bpr=1.0 \
  --multiview_clip_features ./output/multiview_object_clip_replica_s5_m30_bpr \
  --clip_allowed_classes shelf \
  --clip_min_confidence 0.30 \
  --clip_min_margin 0.10 \
  --clip_min_gain_over_current 0.10 \
  --clip_max_base_score 0.35 \
  --clip_blocked_classes rug \
  --score_threshold 0.20 \
  --base_eval_score_mode baseline \
  --eval_output_file ./output/multiview_clip_correction_eval/eval_shelf_only.csv \
  --report_path ./output/multiview_clip_correction_eval/report_shelf_only.json
```

Result:

```text
AP 0.256 / AP50 0.332 / AP25 0.416
```

Previous best:

```text
AP 0.247 / AP50 0.311 / AP25 0.395
```

Applied corrections:

- 3 predictions, all in `room2`
- All were `desk-organizer -> shelf`

This is now the best Replica result so far. It should be presented carefully: the useful correction is currently narrow and class-specific, but it supports the second innovation idea that object-level multi-view visual evidence can repair semantic mistakes left by YOLO-World voting.
