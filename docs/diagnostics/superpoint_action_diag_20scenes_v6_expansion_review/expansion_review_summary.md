# v6 soft-thin risk diagnostic comparison

This comparison uses export-only diagnostics and does not run final AP.

## Action counts

| split | candidates | accept_completion | manual_review | reject_or_needs_mask3d_support | keep_core_only |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 scenes v5 | 130 | 42 | 69 | 18 | 1 |
| 20 scenes v6 | 130 | 40 | 71 | 18 | 1 |
| new in v6 vs v5 | 0 | 0 | 0 | 0 | 0 |

- Action changes on shared candidates: 2

## Review list counts

| review_list | 20 scenes v5 | 20 scenes v6 |
| --- | ---: | ---: |
| accept_completion_conflict_ge_0_18_or_existing_iou_lt_0_30 | 0 | 0 |
| accept_completion_largest_cc_to_point_ge_2 | 9 | 7 |
| all_keep_core_only | 1 | 1 |
| all_reject_or_needs_mask3d_support | 18 | 18 |
| manual_review_largest_cc_to_point_ge_2 | 5 | 7 |

## Main checks

- New accept_completion candidates: 0; large-expansion accepts among them: 0.
- New accept_completion with large-plane, conflict >= 0.18, or existing IoU < 0.30: 0.
- The high-risk accept review list `accept_completion_conflict_ge_0_18_or_existing_iou_lt_0_30` is 0 for 20 scenes v6.
- Rejections remain useful to inspect for missing reliable core, large-plane over-expansion, or generic large expansion without strong support.

## Action changes on shared candidates

- scene0307_00 candidate0002 blanket ratio=2.10 IoU=0.56 conflict=0.01: accept_completion -> manual_review; soft_thin_plane_class;largest_cc_to_point=2.10;largest_cc_covered_by_point=0.45;point_covered_by_largest_cc=0.94;mask3d_iou=0.56/coverage=1.00;boundary_expands_without_cleanup=1.47;v6_soft_thin_plane_large_expansion_review
- scene0353_00 candidate0009 laptop ratio=3.14 IoU=1.00 conflict=0.00: accept_completion -> manual_review; soft_thin_plane_class;largest_cc_to_point=3.14;largest_cc_covered_by_point=0.29;point_covered_by_largest_cc=0.91;mask3d_iou=1.00/coverage=1.00;boundary_expands_without_cleanup=1.48;v6_soft_thin_plane_large_expansion_review

## New accept_completion large-expansion candidates for visual review

None.

## New reject_or_needs_mask3d_support candidates

None.

## New manual_review large-expansion candidates

None.
