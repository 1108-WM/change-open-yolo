# v7 40-scene export-only stability comparison

This comparison uses export-only diagnostics and does not run final AP.

## Action counts

| split | candidates | accept_completion | manual_review | reject_or_needs_mask3d_support | keep_core_only |
| --- | ---: | ---: | ---: | ---: | ---: |
| 20 scenes v7 | 130 | 39 | 72 | 18 | 1 |
| 40 scenes v7 | 217 | 65 | 109 | 40 | 3 |
| new 20 scenes | 87 | 26 | 37 | 22 | 2 |

- Action changes on shared candidates: 0

## Review list counts

| review_list | 20 scenes v7 | 40 scenes v7 |
| --- | ---: | ---: |
| accept_completion_conflict_ge_0_18_or_existing_iou_lt_0_30 | 0 | 0 |
| accept_completion_largest_cc_to_point_ge_2 | 7 | 12 |
| accept_completion_soft_thin_plane | 0 | 1 |
| accept_completion_soft_thin_plane_iou_lt_0_35 | 0 | 0 |
| all_keep_core_only | 1 | 3 |
| all_reject_or_needs_mask3d_support | 18 | 40 |
| manual_review_largest_cc_to_point_ge_2 | 7 | 12 |
| manual_review_soft_thin_plane_iou_lt_0_35 | 1 | 1 |

## Main checks

- New accept_completion candidates: 26; large-expansion accepts among them: 5.
- New accept_completion with large-plane, conflict >= 0.18, or existing IoU < 0.30: 1.
- The high-risk accept review list `accept_completion_conflict_ge_0_18_or_existing_iou_lt_0_30` is 0 for 40 scenes v7.
- Rejections remain useful to inspect for missing reliable core, large-plane over-expansion, or generic large expansion without strong support.

## New accept_completion large-expansion candidates for visual review

- scene0406_00 candidate0005 tissue box ratio=2.04 IoU=0.67 conflict=0.00
- scene0430_00 candidate0001 backpack ratio=2.12 IoU=0.63 conflict=0.06
- scene0435_02 candidate0005 backpack ratio=3.42 IoU=0.81 conflict=0.00
- scene0474_00 candidate0001 computer tower ratio=2.52 IoU=0.70 conflict=0.00
- scene0518_00 candidate0001 storage organizer ratio=2.64 IoU=0.67 conflict=0.02

## New reject_or_needs_mask3d_support candidates

- scene0406_00 candidate0006 tray ratio=0.00 IoU=0.00 conflict=0.00: missing_reliable_superpoint_core
- scene0474_00 candidate0003 hair dryer ratio=0.00 IoU=0.00 conflict=0.00: missing_reliable_superpoint_core
- scene0474_00 candidate0004 laptop ratio=0.00 IoU=0.00 conflict=0.00: missing_reliable_superpoint_core
- scene0518_00 candidate0000 couch ratio=2.16 IoU=0.24 conflict=0.13: largest_cc_to_point=2.16;largest_cc_covered_by_point=0.40;point_covered_by_largest_cc=0.87;large_expansion
- scene0518_00 candidate0003 book ratio=0.00 IoU=0.00 conflict=0.00: missing_reliable_superpoint_core
- scene0518_00 candidate0005 mat ratio=19.98 IoU=0.01 conflict=0.00: large_plane_class;largest_cc_to_point=19.98;largest_cc_covered_by_point=0.05;point_covered_by_largest_cc=0.93;large_plane_overexpanded_without_mask3d_support
- scene0568_02 candidate0004 guitar ratio=2.02 IoU=0.48 conflict=0.00: largest_cc_to_point=2.02;point_covered_by_largest_cc=0.98;large_expansion
- scene0578_00 candidate0004 keyboard ratio=0.00 IoU=0.00 conflict=0.00: missing_reliable_superpoint_core
- scene0578_00 candidate0005 backpack ratio=2.32 IoU=0.37 conflict=0.05: largest_cc_to_point=2.32;largest_cc_covered_by_point=0.37;point_covered_by_largest_cc=0.87;boundary_expands_without_cleanup=1.40;large_expansion
- scene0578_00 candidate0006 backpack ratio=2.78 IoU=0.26 conflict=0.01: largest_cc_to_point=2.78;largest_cc_covered_by_point=0.33;point_covered_by_largest_cc=0.91;boundary_expands_without_cleanup=1.46;large_expansion
- scene0599_02 candidate0001 paper towel dispenser ratio=2.02 IoU=0.69 conflict=0.11: largest_cc_to_point=2.02;largest_cc_covered_by_point=0.39;mask3d_iou=0.69/coverage=1.00;boundary_expands_without_cleanup=1.43;large_expansion
- scene0599_02 candidate0003 mat ratio=0.00 IoU=0.00 conflict=0.00: missing_reliable_superpoint_core
- scene0608_01 candidate0005 armchair ratio=2.22 IoU=0.20 conflict=0.04: largest_cc_to_point=2.22;largest_cc_covered_by_point=0.42;point_covered_by_largest_cc=0.93;large_expansion
- scene0608_01 candidate0006 poster ratio=5.69 IoU=0.17 conflict=0.00: large_plane_class;largest_cc_to_point=5.69;largest_cc_covered_by_point=0.18;point_covered_by_largest_cc=1.00;boundary_expands_without_cleanup=1.36;large_plane_overexpanded_without_mask3d_support
- scene0608_01 candidate0009 door ratio=2.23 IoU=0.00 conflict=0.00: large_plane_class;largest_cc_to_point=2.23;largest_cc_covered_by_point=0.43;point_covered_by_largest_cc=0.96;boundary_expands_without_cleanup=1.44;large_plane_overexpanded_without_mask3d_support
- scene0616_01 candidate0003 radiator ratio=2.45 IoU=0.18 conflict=0.03: largest_cc_to_point=2.45;largest_cc_covered_by_point=0.37;point_covered_by_largest_cc=0.90;boundary_expands_without_cleanup=1.40;large_expansion
- scene0616_01 candidate0006 bulletin board ratio=2.40 IoU=0.00 conflict=0.02: large_plane_class;largest_cc_to_point=2.40;largest_cc_covered_by_point=0.41;point_covered_by_largest_cc=0.97;large_plane_overexpanded_without_mask3d_support
- scene0633_00 candidate0003 wardrobe ratio=3.50 IoU=0.29 conflict=0.09: largest_cc_to_point=3.50;largest_cc_covered_by_point=0.22;large_expansion
- scene0633_00 candidate0005 mat ratio=8.38 IoU=0.00 conflict=0.00: large_plane_class;largest_cc_to_point=8.38;largest_cc_covered_by_point=0.12;point_covered_by_largest_cc=1.00;large_plane_overexpanded_without_mask3d_support
- scene0647_00 candidate0000 poster ratio=2.34 IoU=0.53 conflict=0.05: large_plane_class;largest_cc_to_point=2.34;largest_cc_covered_by_point=0.40;point_covered_by_largest_cc=0.93;mask3d_iou=0.53/coverage=1.00;large_plane_overexpanded_requires_visual_or_mask3d_review
- scene0647_00 candidate0001 tv ratio=2.37 IoU=0.94 conflict=0.00: large_plane_class;largest_cc_to_point=2.37;largest_cc_covered_by_point=0.42;point_covered_by_largest_cc=1.00;mask3d_iou=0.94/coverage=1.00;large_plane_overexpanded_requires_visual_or_mask3d_review
- scene0647_00 candidate0002 poster ratio=2.52 IoU=0.69 conflict=0.00: large_plane_class;largest_cc_to_point=2.52;largest_cc_covered_by_point=0.39;point_covered_by_largest_cc=0.98;mask3d_iou=0.69/coverage=0.85;boundary_expands_without_cleanup=1.39;large_plane_overexpanded_requires_visual_or_mask3d_review

## New manual_review large-expansion candidates

- scene0578_00 candidate0003 table ratio=3.09 IoU=0.37 conflict=0.00: largest_cc_to_point=3.09;largest_cc_covered_by_point=0.31;point_covered_by_largest_cc=0.97;mask3d_iou=0.37/coverage=1.00;large_expansion_with_mask3d_support
- scene0599_02 candidate0002 window ratio=2.89 IoU=0.47 conflict=0.00: largest_cc_to_point=2.89;largest_cc_covered_by_point=0.33;point_covered_by_largest_cc=0.95;mask3d_iou=0.47/coverage=1.00;large_expansion_with_mask3d_support
- scene0633_00 candidate0004 picture ratio=2.52 IoU=0.83 conflict=0.00: small_plane_class;largest_cc_to_point=2.52;largest_cc_covered_by_point=0.39;point_covered_by_largest_cc=0.99;mask3d_iou=0.83/coverage=1.00;small_plane_large_expansion
- scene0647_00 candidate0005 chair ratio=2.07 IoU=0.41 conflict=0.00: largest_cc_to_point=2.07;point_covered_by_largest_cc=1.00;mask3d_iou=0.41/coverage=1.00;boundary_expands_without_cleanup=1.48;large_expansion_with_mask3d_support
- scene0651_02 candidate0000 refrigerator ratio=2.74 IoU=0.51 conflict=0.00: largest_cc_to_point=2.74;largest_cc_covered_by_point=0.36;point_covered_by_largest_cc=1.00;mask3d_iou=0.51/coverage=1.00;large_expansion_with_mask3d_support
