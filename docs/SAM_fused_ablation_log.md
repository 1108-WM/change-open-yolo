# SAM-fused proposal ablation log

Date: 2026-05-25

## Fixed setup

- Dataset: Replica
- 3D masks: `./output/replica/replica_masks`
- 2D predictions: `./output/replica/bboxes_2d`
- SAM-fused candidates: `./output/sam_fused_proposals_replica_s5_m20`
- BPR candidates: `./output/backprojection_candidates_replica_mv_m20`
- Score threshold: `0.20`
- BPR fusion filters:
  - `backprojection_min_score=0.40`
  - `backprojection_min_seed_points=80`
  - `backprojection_max_existing_iou=0.30`
  - `backprojection_max_seed_in_existing_mask_ratio=0.70`
  - `backprojection_blocked_classes=rug`

## Results

| Variant | AP | AP50 | AP25 | Notes |
| --- | ---: | ---: | ---: | --- |
| baseline + threshold | 0.239 | 0.289 | 0.355 | Previous baseline |
| BPR only | 0.243 | 0.298 | 0.393 | Previous BPR baseline |
| SAM-fused only | 0.247 | 0.307 | 0.373 | Previous SAM proposal result |
| SAM-fused + BPR, score scale 0.50 | 0.247 | 0.308 | 0.392 | Current best balanced setting |
| SAM-fused + BPR, source limit `sam_fused=20,bpr=10` | 0.247 | 0.308 | 0.392 | Same metric at 3 decimals; removes 6 BPR proposals |
| SAM-fused + BPR, source limit `sam_fused=20,bpr=5` | 0.247 | 0.308 | 0.386 | Too aggressive; loses AP25 recall |
| SAM-fused + BPR, quality sort | 0.243 | 0.298 | 0.393 | Falls back toward BPR-only behavior |
| SAM-fused + BPR, score scale 0.60 | 0.247 | 0.308 | 0.392 | No visible change at 3 decimals |
| SAM-fused + BPR, score scale 0.40 | 0.247 | 0.307 | 0.382 | Over-filters low-score added proposals |
| SAM-fused `s3_m30` only | 0.247 | 0.307 | 0.380 | Denser export adds more SAM proposals but hurts AP25 alone |
| SAM-fused `s3_m30` + BPR | 0.247 | 0.308 | 0.396 | Best AP25 so far; AP/AP50 preserved |
| SAM-fused `s5_m30` only | 0.247 | 0.310 | 0.377 | Higher cap improves AP50 but hurts AP25 alone |
| SAM-fused `s5_m30` + BPR | 0.247 | 0.310 | 0.395 | Best AP50 so far; near-best AP25 with lower export cost than `s3_m30` |
| SAM-fused `s5_m30` balanced novelty + BPR | 0.247 | 0.310 | 0.395 | Same as standard `s5_m30`; ranking is not sensitive |
| SAM-fused `s5_m30` novelty + BPR | 0.247 | 0.310 | 0.395 | Same as standard `s5_m30`; novelty cannot be the main ranking signal here |
| SAM-fused `s5_m40` + BPR | 0.247 | 0.310 | 0.395 | Same as `s5_m30`; cap 30 is already enough |
| SAM-fused `s5_m30` + BPR, SAM score scale 1.2 | 0.247 | 0.311 | 0.395 | Best AP50 so far; keeps AP25 |
| SAM-fused `s5_m30` + BPR, SAM score scale 1.5 | 0.247 | 0.311 | 0.395 | Same as 1.2; SAM score boost saturates |
| SAM-fused `s5_m30` + BPR, SAM score 1.2 / BPR score 0.9 | 0.247 | 0.310 | 0.384 | Lowering BPR score hurts AP25 |
| SAM-fused `s5_m30` top-2 SAM masks + BPR, SAM score scale 1.2 | 0.247 | 0.311 | 0.395 | Same as current best; alternate masks are mostly merged or filtered |
| SAM-fused `s5_m30` top-2 SAM masks + novelty + BPR, SAM score scale 1.2 | 0.247 | 0.311 | 0.395 | Same as current best |
| SAM-fused `s5_m30` top-2 SAM masks, keep alternatives + BPR, SAM score scale 1.2 | 0.247 | 0.308 | 0.392 | Keeping same-box alternate masks hurts; likely noisy duplicates |
| SAM-fused `s5_m30` prefilter + BPR, SAM score scale 1.2 | 0.247 | 0.311 | 0.395 | Same metric with 52% fewer loaded candidates |
| SAM-fused `s5_m30` class-core + BPR, SAM score scale 1.2 | 0.247 | 0.311 | 0.395 | Diagnostic only; uses classes found from evaluation |

## Report diagnostics

Default SAM-fused + BPR:

- Loaded candidates: 316 from 16 JSON files
- Added proposals: 93
- Added by source:
  - SAM-fused: 15
  - BPR: 78
- Main skip reasons:
  - `matched_existing_3d_mask`: 120
  - `grown_mask_matches_existing`: 43
  - `mostly_covered_by_existing_masks`: 24
  - `class_blocked`: 20
  - `low_score`: 16

Source limit `sam_fused=20,bpr=10`:

- Added proposals: 87
- Added by source:
  - SAM-fused: 15
  - BPR: 72
- New skip reason:
  - `source_limit`: 9

Source limit `sam_fused=20,bpr=5`:

- Added proposals: 55
- Added by source:
  - SAM-fused: 15
  - BPR: 40
- New skip reason:
  - `source_limit`: 77

SAM-fused `s3_m30` export:

- Export time: 83.25 seconds
- Raw SAM observations: 1876
- Merged candidates: 237
- SAM-only added proposals: 24
- SAM+BPR added proposals:
  - SAM-fused: 24
  - BPR: 69

SAM-fused `s5_m30` export:

- Export time: 63.00 seconds
- Raw SAM observations: 1150
- Merged candidates: 216
- SAM-only added proposals: 23
- SAM+BPR added proposals:
  - SAM-fused: 23
  - BPR: 73

SAM-fused `s5_m30` novelty-ranking variants:

- `balanced_novelty`:
  - Export time: 62.66 seconds
  - Merged candidates: 216
  - SAM+BPR added proposals:
    - SAM-fused: 23
    - BPR: 73
- `novelty`:
  - Export time: 62.90 seconds
  - Merged candidates: 216
  - SAM+BPR added proposals:
    - SAM-fused: 23
    - BPR: 73

SAM-fused `s5_m40` export:

- Export time: 63.48 seconds
- Raw SAM observations: 1150
- Merged candidates: 226
- SAM+BPR added proposals:
  - SAM-fused: 23
  - BPR: 73

Source-specific score scaling on `s5_m30` + BPR:

- `sam_fused=1.2,bpr=1.0`: AP50 improves from 0.310 to 0.311 while AP25 stays 0.395.
- `sam_fused=1.5,bpr=1.0`: no further improvement over 1.2.
- `sam_fused=1.2,bpr=0.9`: AP25 drops to 0.384, confirming BPR's low-IoU recall should not be downweighted too much.

SAM multimask variants:

- `sam_multimask_topk=2`:
  - Export time: 69.09 seconds
  - Raw SAM mask observations: 2300
  - Merged candidates: 215
  - Final candidates are still almost all SAM rank 0; rank 1 masks are merged or pushed out.
- `sam_multimask_topk=2` with novelty ranking:
  - Export time: 68.97 seconds
  - Merged candidates: 215
  - Rank 1 masks still do not enter the final set in meaningful numbers.
- `sam_multimask_topk=2` with `keep_sam_mask_alternatives`:
  - Export time: 71.07 seconds
  - Merged candidates: 240
  - Rank 0/rank 1 are both represented, but metrics drop to the older `s5_m20`-level result.

Pre-cap existing-mask filter:

- Export params:
  - `export_max_existing_iou=0.30`
  - `export_max_seed_in_existing_mask_ratio=0.70`
- Export time: 62.14 seconds
- Raw SAM observations: 1150
- Merged candidates after prefilter: 23, down from 216
- Combined candidate load at evaluation: 180, down from 373
- Applied proposals are unchanged:
  - SAM-fused: 23
  - BPR: 73
- Skips removed from eval report:
  - `matched_existing_3d_mask`: 144 -> 0
  - `mostly_covered_by_existing_masks`: 49 -> 0

Class-core diagnostic filter:

- Kept SAM-fused classes: `sculpture,vase,clock,table,chair`
- Kept SAM candidates: 57 / 216
- Metrics are unchanged, but this is an analysis setting rather than a deployable method because the class set was chosen from evaluation diagnostics.

## Current conclusion

The strongest AP50 setting so far is SAM-fused `s5_m30` + BPR with source-specific score scaling `sam_fused=1.2,bpr=1.0`.
The best AP25 setting so far is the denser SAM-fused `s3_m30` export combined with BPR.
Both use `score_scale=0.50` and detector score rather than candidate `fusion_score`.
Hard source limits are useful for analysis but do not improve Replica metrics yet. `bpr=5` confirms that BPR still carries AP25 recall, while SAM-fused provides the AP/AP50 improvement.
Novelty-aware export ranking does not change the final accepted proposal set enough to improve metrics. Increasing the per-scene cap from 30 to 40 also saturates, so the useful cap is 30 under the current filters.
Source-specific score scaling is mildly useful: boost SAM-fused scores, but do not suppress BPR scores.
SAM multimask alternatives are not useful under the current pipeline. Letting them merge changes little; forcing them to stay separate introduces noisy duplicate proposals.
Pre-cap existing-mask filtering is useful as an efficiency and cleanliness improvement: it removes candidates that evaluation would skip anyway without changing AP/AP50/AP25.

Next useful experiments:

- Compare `s3_m20` versus `s3_m30` to isolate whether stride 3 has value when the cap is fixed.
- Diagnose per-class gains from `s5_m30` and `s3_m30`; vase and chair are the clearest classes affected by the new SAM candidates.
- Next code-level improvement should target candidate generation quality beyond existing-mask filtering: generate better candidates for missed objects rather than alternate SAM masks for already detected boxes.
- Do not use `keep_sam_mask_alternatives` as a main result unless a later geometry verifier is added.
