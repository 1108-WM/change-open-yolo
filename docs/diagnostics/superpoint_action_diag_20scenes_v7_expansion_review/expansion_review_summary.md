# v7 soft-thin low-IoU rule comparison

This comparison uses export-only diagnostics and does not run final AP.

## Action counts

| split | candidates | accept_completion | manual_review | reject_or_needs_mask3d_support | keep_core_only |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 scenes v6.1 | 130 | 40 | 71 | 18 | 1 |
| 20 scenes v7 | 130 | 39 | 72 | 18 | 1 |
| new in v7 vs v6.1 | 0 | 0 | 0 | 0 | 0 |

- Action changes on shared candidates: 1

## Review list counts

| review_list | 20 scenes v6.1 | 20 scenes v7 |
| --- | ---: | ---: |
| accept_completion_conflict_ge_0_18_or_existing_iou_lt_0_30 | 0 | 0 |
| accept_completion_largest_cc_to_point_ge_2 | 7 | 7 |
| accept_completion_soft_thin_plane | 1 | 0 |
| accept_completion_soft_thin_plane_iou_lt_0_35 | 1 | 0 |
| all_keep_core_only | 1 | 1 |
| all_reject_or_needs_mask3d_support | 18 | 18 |
| manual_review_largest_cc_to_point_ge_2 | 7 | 7 |
| manual_review_soft_thin_plane_iou_lt_0_35 | 0 | 1 |

## Main checks

- New accept_completion candidates: 0; large-expansion accepts among them: 0.
- New accept_completion with large-plane, conflict >= 0.18, or existing IoU < 0.30: 0.
- The high-risk accept review list `accept_completion_conflict_ge_0_18_or_existing_iou_lt_0_30` is 0 for 20 scenes v7.
- Rejections remain useful to inspect for missing reliable core, large-plane over-expansion, or generic large expansion without strong support.

## Action changes on shared candidates

- scene0207_00 candidate0003 blanket ratio=1.76 IoU=0.31 conflict=0.07: accept_completion -> manual_review; soft_thin_plane_class;point_covered_by_largest_cc=0.93;v7_soft_thin_moderate_expansion_low_iou_review

## New accept_completion large-expansion candidates for visual review

None.

## New reject_or_needs_mask3d_support candidates

None.

## New manual_review large-expansion candidates

None.
