# v6.1 soft/thin accept review

This is a diagnostic-only supplement. It does not run final AP and does not change the fusion main flow.

## Review-list additions

- `accept_completion_soft_thin_plane`
- `accept_completion_soft_thin_plane_iou_lt_0_35`

Both lists currently contain the same candidate:

| scene | candidate | class | action | largest_cc_to_point | largest_cc_covered_by_point | point_covered_by_largest_cc | existing_mask_iou | conflict |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| scene0207_00 | 3 | blanket | accept_completion | 1.76 | 0.53 | 0.93 | 0.31 | 0.07 |

## Visual check

- `scene0207_00 / candidate0003 / blanket`
  - The point candidate is a soft/thin planar region and the superpoint completion expands it from `1049` points to `1851` points.
  - Largest-CC cleanup removes `0` points, so it does not reduce the expanded region.
  - Mask3D support is weak for automatic acceptance: IoU is `0.315`, below the `0.35` secondary support threshold.
  - v6.1 leaves the recommended action unchanged for diagnostics, but this candidate should be treated as a soft/thin accept risk in review.

## Generated visuals

- `docs/visual_checks/superpoint_action_review_20scenes_v6_1_soft_thin/accept_completion_soft_thin_plane/01_scene0207_00_candidate0003_blanket_four_sets_xz.png`
- `docs/visual_checks/superpoint_action_review_20scenes_v6_1_soft_thin/accept_completion_soft_thin_plane/01_scene0207_00_candidate0003_blanket_largest_cc_overlay.png`
