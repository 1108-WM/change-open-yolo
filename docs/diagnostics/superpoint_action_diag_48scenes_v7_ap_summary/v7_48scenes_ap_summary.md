# v7 48-Scene AP Summary

This note freezes the v7 diagnostic rule state for the even48 AP check. It records the AP-facing outcome only; no v8/v9 rule tuning and no further replacement/refinement AP branch is implied.

## Inputs

- Scene split: `output/scannet200/scene_splits/even48.txt`
- v7 export-only diagnostics: `/tmp/mask_graph_proposals_scannet200_superpoint_largest_cc_diag_48scenes_v7`
- v7 action table: `/tmp/superpoint_action_diag_48scenes_v7/actions.csv`
- Current reference AP table: `/tmp/even48_v7_ap_control/current_original.csv`

## v7 Action Distribution

| action | count |
|---|---:|
| accept_completion | 80 |
| manual_review | 122 |
| reject_or_needs_mask3d_support | 45 |
| keep_core_only | 4 |

Total v7 diagnostic candidates: 251.

## High-Risk Accept Lists

| review list | count |
|---|---:|
| accept_completion_conflict_ge_0_18_or_existing_iou_lt_0_30 | 0 |
| accept_completion_largest_cc_to_point_ge_2 | 13 |
| accept_completion_soft_thin_plane | 1 |
| accept_completion_soft_thin_plane_iou_lt_0_35 | 0 |

The conflict/low-IoU accept list is empty, and soft/thin low-IoU accept is also empty. The remaining high-risk accept surface is mainly large expansion (`largest_cc_to_point >= 2`), with one soft/thin accept that does not fall below the IoU guard. See `high_risk_accept_lists.csv`.

## AP Branches

| branch | applied | mAP | AP50 | AP25 | delta mAP | delta AP50 | delta AP25 |
|---|---:|---:|---:|---:|---:|---:|---:|
| current original | - | 0.271195 | 0.345761 | 0.389522 | - | - | - |
| add v7 accept largest_cc as ordinary candidates | 0 | 0.271195 | 0.345761 | 0.389522 | 0.000000 | 0.000000 | 0.000000 |
| unconditional matched-existing replacement | 68 | 0.263229 | 0.335697 | 0.379434 | -0.007966 | -0.010063 | -0.010088 |
| class-consistent replacement | 3 | 0.270892 | 0.345456 | 0.389216 | -0.000303 | -0.000305 | -0.000306 |
| expansion-only same-class refinement | 0 | 0.271195 | 0.345761 | 0.389522 | 0.000000 | 0.000000 | 0.000000 |

## Interpretation

1. Adding v7 accept completions as ordinary new candidates produced `applied=0` in the current flow, because the accepted candidates are already caught by existing-mask filters (`matched_existing_3d_mask=68`, `mostly_covered_by_existing_masks=12`). AP is unchanged.
2. Unconditional replacement applied 68 geometry swaps and lowered AP. Among those applied swaps, 65 have candidate/existing class mismatches; see `unconditional_replacement_class_mismatch_examples.csv` for representative rows. This means `matched_existing_3d_mask` is not a safe replacement target by itself.
3. Class-consistent replacement reduced the branch to 3 applied swaps but still slightly lowered AP. The nonzero class delta is `armchair`.
4. Expansion-only same-class refinement applied 0 swaps. The three class-consistent candidates would all shrink existing masks, so they were skipped as `would_shrink_existing_mask`; AP is unchanged.

## Conclusion

v7 is stable as a geometry-candidate diagnostic rule: its 48-scene accept set avoids the explicit high-risk conflict/low-IoU accept lists. However, the current fusion and matching mechanism cannot convert these candidates into AP gains. The ordinary new-candidate path filters them out, while replacement-style paths either introduce class mismatch damage or shrink existing masks.

## Next Direction

Stop direct geometry replacement/refinement AP experiments for this line. The next useful direction is not v8/v9 threshold tuning, but a safer mechanism such as score re-ranking or a class-aware instance graph that can use v7 geometry evidence without overwriting existing instance geometry blindly.
