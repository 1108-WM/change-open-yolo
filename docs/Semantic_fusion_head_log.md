# SegDINO3D-style Semantic Fusion Head Log

## Goal

Build a lightweight trainable semantic fusion head on top of the frozen OpenYOLO3D baseline and the SAM-fused/BPR proposal sources. This is intentionally not a full SegDINO3D reproduction. The first step is object-level supervision and inference-time feature fusion.

## Dataset Export

Script:

```bash
tools/export_semantic_fusion_dataset.py
```

Full Replica export command:

```bash
OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mpl TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/jia/anaconda3/envs/openyolo3d/bin/python tools/export_semantic_fusion_dataset.py \
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
  --output_dir ./output/semantic_fusion_dataset_replica_s5_m30_bpr
```

Output:

- `output/semantic_fusion_dataset_replica_s5_m30_bpr/features.jsonl`
- `output/semantic_fusion_dataset_replica_s5_m30_bpr/summary.json`

Summary:

- Records: 305
- Source counts: `mask3d=209`, `sam_fused=23`, `bpr=73`
- `positive@25=191`
- `class_correct@25=129`
- `class_correct@50=102`

## Trainable Scoring Head

Script:

```bash
tools/train_semantic_fusion_head.py
```

Best diagnostic run so far:

```bash
/home/jia/anaconda3/envs/openyolo3d/bin/python tools/train_semantic_fusion_head.py \
  --dataset_jsonl ./output/semantic_fusion_dataset_replica_s5_m30_bpr/features.jsonl \
  --output_dir ./output/semantic_fusion_head_replica_s5_m30_bpr_mlp32 \
  --hidden_dim 32 \
  --dropout 0.15 \
  --epochs 1200 \
  --lr 0.005 \
  --weight_decay 0.005 \
  --blend_alpha 0.35
```

Split:

- Train: `office0` to `office4`
- Validation: `room0` to `room2`

Validation ranking AP for `class_correct_25`:

- Base score: 0.654
- MLP score: 0.751
- Blended score: 0.742

This confirms that the exported inference-time features contain learnable semantic/object quality signal.

## Official Replica Evaluation

Script:

```bash
tools/evaluate_semantic_fusion_head.py
```

Directly replacing all prediction scores with head scores hurts official AP:

- Replace all: `AP 0.233 / AP50 0.297 / AP25 0.380`
- Blend all, alpha 0.35: `AP 0.235 / AP50 0.298 / AP25 0.380`

Reason: the best OpenYOLO3D evaluation path keeps original Mask3D proposal scores as constant 1.0 and only lets appended proposals use confidence scores. Replacing all Mask3D scores disrupts that ranking.

Proposal-only rescoring preserves the first-innovation result but does not improve it:

- Proposal-only blend alpha 0.70: `AP 0.247 / AP50 0.311 / AP25 0.395`
- Proposal-only blend alpha 0.50: `AP 0.247 / AP50 0.311 / AP25 0.395`
- Proposal-only replace: `AP 0.247 / AP50 0.311 / AP25 0.394`

## Current Conclusion

The semantic fusion head has a real offline learning signal, but using it only as a confidence scorer is not enough to improve the official Replica AP beyond the current SAM-fused+BPR result.

The next useful version of this second innovation should move from binary confidence scoring to object-level semantic correction:

1. Train on positive masks/proposals with IoU >= 0.25 or 0.50.
2. Predict the GT semantic class, not only whether the current class is correct.
3. Apply corrections only when confidence and margin are high.
4. Keep Mask3D baseline score handling unchanged unless a correction is applied.
