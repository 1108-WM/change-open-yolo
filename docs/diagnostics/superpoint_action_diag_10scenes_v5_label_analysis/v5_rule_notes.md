# v5 rule notes

Inputs are v4 actions plus 18 visual labels from the v4 review images.

## Label alignment

- keep_core_only -> visual_keep_core_only: 1
- manual_review -> uncertain: 2
- manual_review -> visual_accept: 6
- reject_or_needs_mask3d_support -> uncertain: 2
- reject_or_needs_mask3d_support -> visual_reject: 7

- Missing labels in actions.csv: 0
- accept_completion visual_reject/uncertain: 0
- reject_or_needs_mask3d_support visual_accept: 0
- manual_review visual_accept: 6
- manual_review uncertain: 2

## v5 interpretation

- Keep large planar classes conservative even when Mask3D IoU is high.
- Do not promote office chair or sink large expansions from manual review.
- Promote only strongly supported large non-plane completions with low conflict.
- Promote small-plane picture completion only with very strong Mask3D support.
- Leave rejected and keep_core_only cases unchanged for this 10-scene diagnostic.

## visual_accept candidates promoted by v5

- scene0011_00 candidate0003 trash bin: ratio=2.73, IoU=0.60, point_coverage=1.00, conflict=0.00
- scene0077_00 candidate0000 printer: ratio=2.59, IoU=0.87, point_coverage=0.91, conflict=0.00
- scene0084_01 candidate0007 container: ratio=2.25, IoU=0.74, point_coverage=0.91, conflict=0.07
- scene0131_00 candidate0000 mini fridge: ratio=2.34, IoU=0.77, point_coverage=0.96, conflict=0.00
- scene0164_01 candidate0008 picture: ratio=3.44, IoU=0.93, point_coverage=0.92, conflict=0.00

## visual_accept candidates kept conservative by v5

- scene0193_00 candidate0001 mattress: ratio=2.15, IoU=0.39, point_coverage=1.00, conflict=0.00. Kept as manual_review because it is a large planar or elongated expansion.

## remaining uncertain candidates

- scene0025_02 candidate0007 office chair: expanded region is broad and chair structure is not visually clear enough
- scene0164_01 candidate0007 sink: flat sink-like expansion may be correct but is too planar for automatic acceptance
