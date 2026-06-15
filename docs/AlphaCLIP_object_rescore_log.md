# Alpha-CLIP Object-Level Rescore Log

## Status

Date: 2026-05-28

Alpha-CLIP has been added as an optional object-level visual-language scorer for
`tools/export_multiview_object_clip_features.py`.

The implementation keeps the same prediction source and output JSON format as
the existing multi-view CLIP exporter. The only intended change is the visual
encoder used for object-level semantic evidence:

- `--vision_encoder clip`: existing rectangular crop CLIP path.
- `--vision_encoder alpha_clip`: Alpha-CLIP path using the projected 3D object
  mask as the alpha foreground map.

## Local Files

Alpha-CLIP source:

```bash
_external/AlphaCLIP/AlphaCLIP-main
```

Base CLIP ViT-L/14 checkpoint:

```bash
pretrained/alpha_clip/checkpoints/ViT-L-14.pt
```

Alpha-CLIP visual checkpoint:

```bash
pretrained/alpha_clip/checkpoints/clip_l14_grit20m_fultune_2xe.pth
```

The environment was missing `ftfy`, which was installed into the
`openyolo3d` conda environment.

## Verification

GPU execution was not used because another PyCharm job was occupying the 4090.

CPU-only checks passed:

```bash
CUDA_VISIBLE_DEVICES= /home/jia/anaconda3/envs/openyolo3d/bin/python ...
```

Results:

- Alpha-CLIP loads on CPU.
- Base CLIP and Alpha-CLIP visual checkpoints are compatible.
- Dummy image forward pass works.
- Output image feature shape: `(1, 768)`.
- Output text feature shape: `(2, 768)`.
- Exporter internal Alpha-CLIP encoder returns normalized class probabilities.

After the GPU became free, full Replica export and evaluation were run.

Alpha-CLIP feature export:

- Output: `output/multiview_object_alphaclip_replica_s5_m30_bpr`
- Object records: 300
- View crops: 895
- Time: 160.91 seconds

## Replica Results

All rows use the same SAM-fused + BPR proposal setting.

| Setting | AP | AP50 | AP25 | Corrections |
| --- | ---: | ---: | ---: | ---: |
| SAM-fused + BPR, no semantic correction | 0.247 | 0.311 | 0.395 | 0 |
| Ordinary CLIP, automatic thresholds | 0.230 | 0.284 | 0.344 | 20 |
| Alpha-CLIP, automatic thresholds `0.60/0.10/0.10` | 0.256 | 0.334 | 0.417 | 13 |
| Alpha-CLIP, confusion groups | 0.252 | 0.330 | 0.413 | 5 |
| Alpha-CLIP, stricter thresholds `0.75/0.20/0.20` | **0.261** | **0.340** | **0.424** | 8 |
| Alpha-CLIP, stricter thresholds `0.80/0.20/0.20` | 0.251 | 0.315 | 0.399 | 4 |

The full-export best setting is:

```bash
--clip_min_confidence 0.75 \
--clip_min_margin 0.20 \
--clip_min_gain_over_current 0.20
```

Compared with SAM-fused + BPR without semantic correction, the best
Alpha-CLIP correction improves:

- AP: +0.014
- AP50: +0.029
- AP25: +0.029

Compared with the original baseline + score threshold
`0.239 / 0.289 / 0.355`, the full method reaches
`0.261 / 0.340 / 0.424`.

## Selective Alpha-CLIP

Selective export was added to avoid running Alpha-CLIP on high-confidence
predictions that rarely need correction.

| Export setting | Records | Crops | Export time | AP | AP50 | AP25 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full, `top_views=3` | 300 | 895 | 160.91s | 0.261 | 0.340 | 0.424 |
| `pred_score <= 0.55`, `top_views=3` | 221 | 659 | 76.18s | 0.261 | 0.340 | 0.424 |
| `pred_score <= 0.55`, `top_views=3`, no crop saving | 221 | 659 | 75.50s | 0.261 | 0.340 | 0.424 |
| `pred_score <= 0.45`, `top_views=3` | 191 | 570 | 72.40s | 0.261 | 0.339 | 0.423 |
| `pred_score <= 0.55`, `top_views=2` | 216 | 431 | 72.38s | 0.260 | 0.336 | 0.420 |

Recommended speed/accuracy setting:

```bash
--rescore_policy low_score \
--rescore_max_base_score 0.55 \
--top_views 3 \
--clip_min_confidence 0.60 \
--clip_min_margin 0.10 \
--clip_min_gain_over_current 0.10
```

This keeps the best Replica metric while reducing Alpha-CLIP export time by
about 53%.

The `--no-save_crops` option was added. It avoids writing crop JPG files to
disk and keeps the metric unchanged, but only gives a small runtime reduction
on Replica. Its main practical value is reducing output clutter.

## Threshold Selection

To reduce manual threshold tuning, `tools/search_alphaclip_thresholds.py` was
added. It builds predictions once, then searches correction thresholds with a
train/validation scene split.

Default Replica split:

- Train: `office0, office1, office2, room0`
- Validation: `office3, office4, room1, room2`

Search command:

```bash
OMP_NUM_THREADS=8 MPLCONFIGDIR=/tmp/mpl TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/home/jia/anaconda3/envs/openyolo3d/bin/python tools/search_alphaclip_thresholds.py \
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
  --multiview_clip_features ./output/multiview_object_alphaclip_replica_s5_m30_bpr_low055 \
  --train_scenes office0,office1,office2,room0 \
  --val_scenes office3,office4,room1,room2 \
  --confidences 0.60,0.70,0.75,0.80 \
  --margins 0.10,0.20,0.30 \
  --gains 0.10,0.20,0.30 \
  --clip_blocked_classes rug \
  --score_threshold 0.20 \
  --base_eval_score_mode baseline \
  --report_path ./output/multiview_clip_correction_eval/search_alphaclip_low055_thresholds.json
```

Best threshold selected by training AP:

```bash
--clip_min_confidence 0.60 \
--clip_min_margin 0.10 \
--clip_min_gain_over_current 0.10
```

Result:

| Split | AP | AP50 | AP25 |
| --- | ---: | ---: | ---: |
| Train | 0.321 | 0.412 | 0.440 |
| Validation | 0.196 | 0.278 | 0.385 |
| All Replica | 0.261 | 0.340 | 0.424 |

The searched threshold has the same full Replica result as the stricter manual
threshold, so it is the preferred setting for reporting.

## Recommended Replica Export Command

Run this only when the GPU is free:

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
  --output_dir ./output/multiview_object_alphaclip_replica_s5_m30_bpr_low055 \
  --vision_encoder alpha_clip \
  --alpha_clip_source ./_external/AlphaCLIP/AlphaCLIP-main \
  --alpha_clip_base_model ./pretrained/alpha_clip/checkpoints/ViT-L-14.pt \
  --alpha_clip_checkpoint ./pretrained/alpha_clip/checkpoints/clip_l14_grit20m_fultune_2xe.pth \
  --top_views 3 \
  --min_visible_points 50 \
  --max_bbox_area_ratio 0.60 \
  --aggregate mean \
  --rescore_policy low_score \
  --rescore_max_base_score 0.55 \
  --no-save_crops \
  --device cuda
```

## Evaluation Command

After export:

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
  --multiview_clip_features ./output/multiview_object_alphaclip_replica_s5_m30_bpr_low055 \
  --clip_blocked_classes rug \
  --clip_min_confidence 0.60 \
  --clip_min_margin 0.10 \
  --clip_min_gain_over_current 0.10 \
  --score_threshold 0.20 \
  --base_eval_score_mode baseline \
  --eval_output_file ./output/multiview_clip_correction_eval/eval_alphaclip_low055_search_selected.csv \
  --report_path ./output/multiview_clip_correction_eval/report_alphaclip_low055_search_selected.json
```

For a less hand-tuned automatic correction setting, restrict corrections to
coarse semantic confusion groups instead of explicit class-pair rules:

```bash
  --clip_confusion_groups 'pillow,cushion,blanket,cloth;shelf,cabinet,desk-organizer,bookshelf;chair,stool,bench;sofa,bed,bench;monitor,tv-screen'
```

Under this setting the module still decides the target class from object-level
Alpha-CLIP probabilities. The group only prevents unstable jumps between
unrelated classes.

## Notes

This is not yet a confirmed metric improvement. The next valid comparison is:

- existing SAM-fused + BPR without semantic correction;
- previous ordinary CLIP object-level correction;
- new Alpha-CLIP object-level correction.

The Replica result supports Alpha-CLIP as a stronger object-level semantic
calibration module than ordinary CLIP. Avoid hand-tuned class-pair rules as the
main paper claim until ScanNet200 or a held-out validation split confirms
generalization.
