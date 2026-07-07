# v5 20-scene expansion review

This expansion keeps MODE=export_only and does not run final AP.

## Action counts

| split | candidates | accept_completion | manual_review | reject_or_needs_mask3d_support | keep_core_only |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10 scenes | 63 | 22 | 31 | 9 | 1 |
| 20 scenes | 130 | 42 | 69 | 18 | 1 |
| new 10 scenes | 67 | 20 | 38 | 9 | 0 |

## Review list counts

| review_list | 10 scenes | 20 scenes |
| --- | ---: | ---: |
| accept_completion_conflict_ge_0_18_or_existing_iou_lt_0_30 | 0 | 0 |
| accept_completion_largest_cc_to_point_ge_2 | 5 | 9 |
| all_keep_core_only | 1 | 1 |
| all_reject_or_needs_mask3d_support | 9 | 18 |
| manual_review_largest_cc_to_point_ge_2 | 3 | 5 |

## Main checks

- New accept_completion candidates: 20; large-expansion accepts among them: 4.
- New accept_completion with large-plane, conflict >= 0.18, or existing IoU < 0.30: 0.
- `closet door` was added to the large-plane risk class after the expansion exposed it as a planar large-expansion false accept risk.
- The high-risk accept review list `accept_completion_conflict_ge_0_18_or_existing_iou_lt_0_30` remains 0 on 20 scenes.
- Reject candidates in the new 10 scenes are dominated by missing reliable core, large-plane over-expansion, or generic large expansion without strong support.

## New accept_completion large-expansion candidates for visual review

- scene0277_01 candidate0001 bookshelf ratio=2.41 IoU=0.59 conflict=0.01
- scene0307_00 candidate0002 blanket ratio=2.10 IoU=0.56 conflict=0.01
- scene0353_00 candidate0003 folded chair ratio=2.02 IoU=0.65 conflict=0.02
- scene0353_00 candidate0009 laptop ratio=3.14 IoU=1.00 conflict=0.00

## New reject_or_needs_mask3d_support candidates

- scene0222_00 candidate0010 blinds ratio=3.11 IoU=0.18 conflict=0.00: large_plane_class;largest_cc_to_point=3.11;largest_cc_covered_by_point=0.32;point_covered_by_largest_cc=1.00;large_plane_overexpanded_without_mask3d_support
- scene0222_00 candidate0014 closet door ratio=3.41 IoU=0.79 conflict=0.00: large_plane_class;largest_cc_to_point=3.41;largest_cc_covered_by_point=0.29;point_covered_by_largest_cc=0.99;mask3d_iou=0.79/coverage=1.00;large_plane_overexpanded_requires_visual_or_mask3d_review
- scene0249_00 candidate0000 projector screen ratio=4.68 IoU=0.42 conflict=0.00: large_plane_class;largest_cc_to_point=4.68;largest_cc_covered_by_point=0.21;point_covered_by_largest_cc=1.00;mask3d_iou=0.42/coverage=1.00;large_plane_overexpanded_requires_visual_or_mask3d_review
- scene0353_00 candidate0011 poster ratio=0.00 IoU=0.00 conflict=0.00: missing_reliable_superpoint_core
- scene0356_01 candidate0001 door ratio=0.00 IoU=0.00 conflict=0.00: missing_reliable_superpoint_core
- scene0378_00 candidate0005 paper ratio=0.00 IoU=0.00 conflict=0.00: missing_reliable_superpoint_core
- scene0378_00 candidate0008 bulletin board ratio=44.83 IoU=0.00 conflict=0.00: large_plane_class;largest_cc_to_point=44.83;largest_cc_covered_by_point=0.02;point_covered_by_largest_cc=0.95;large_plane_overexpanded_without_mask3d_support
- scene0378_00 candidate0009 bulletin board ratio=2.84 IoU=0.02 conflict=0.00: large_plane_class;largest_cc_to_point=2.84;largest_cc_covered_by_point=0.34;point_covered_by_largest_cc=0.97;boundary_expands_without_cleanup=1.47;large_plane_overexpanded_without_mask3d_support
- scene0378_00 candidate0010 calendar ratio=0.00 IoU=0.00 conflict=0.00: missing_reliable_superpoint_core

## New manual_review large-expansion candidates

- scene0222_00 candidate0007 dresser ratio=3.91 IoU=0.44 conflict=0.00: largest_cc_to_point=3.91;largest_cc_covered_by_point=0.23;point_covered_by_largest_cc=0.91;mask3d_iou=0.44/coverage=1.00;large_expansion_with_mask3d_support
- scene0222_00 candidate0013 cushion ratio=2.25 IoU=0.54 conflict=0.00: largest_cc_to_point=2.25;largest_cc_covered_by_point=0.43;point_covered_by_largest_cc=0.97;mask3d_iou=0.54/coverage=1.00;boundary_expands_without_cleanup=1.46;large_expansion_with_mask3d_support
