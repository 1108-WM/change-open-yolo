# v6.1 soft-thin accept diagnostic comparison

This comparison uses export-only diagnostics and does not run final AP.

## Action counts

| split | candidates | accept_completion | manual_review | reject_or_needs_mask3d_support | keep_core_only |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 scenes v6 | 130 | 40 | 71 | 18 | 1 |
| 20 scenes v6.1 | 130 | 40 | 71 | 18 | 1 |
| new in v6.1 vs v6 | 0 | 0 | 0 | 0 | 0 |

- Action changes on shared candidates: 0

## Review list counts

| review_list | 20 scenes v6 | 20 scenes v6.1 |
| --- | ---: | ---: |
| accept_completion_conflict_ge_0_18_or_existing_iou_lt_0_30 | 0 | 0 |
| accept_completion_largest_cc_to_point_ge_2 | 7 | 7 |
| accept_completion_soft_thin_plane | 0 | 1 |
| accept_completion_soft_thin_plane_iou_lt_0_35 | 0 | 1 |
| all_keep_core_only | 1 | 1 |
| all_reject_or_needs_mask3d_support | 18 | 18 |
| manual_review_largest_cc_to_point_ge_2 | 7 | 7 |

## Main checks

- New accept_completion candidates: 0; large-expansion accepts among them: 0.
- New accept_completion with large-plane, conflict >= 0.18, or existing IoU < 0.30: 0.
- The high-risk accept review list `accept_completion_conflict_ge_0_18_or_existing_iou_lt_0_30` is 0 for 20 scenes v6.1.
- Rejections remain useful to inspect for missing reliable core, large-plane over-expansion, or generic large expansion without strong support.

## New accept_completion large-expansion candidates for visual review

None.

## New reject_or_needs_mask3d_support candidates

None.

## New manual_review large-expansion candidates

None.
