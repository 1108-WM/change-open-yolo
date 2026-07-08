# Cleanup Plan 20260708

This is a pre-cleanup checklist only. Do not delete anything from this file without a separate explicit cleanup request.

Current dirty state observed before writing this plan:

- `models/Mask3D/...` build artifacts are modified and should be ignored.
- `docs/visual_checks/superpoint_action_review_20scenes_v5/` is an untracked old visual-check directory.
- `vibe_research_report.tex` is untracked and user-owned; do not touch it.
- No `evidence_tree_v0` files or text references were found in the current working tree. If another machine or branch has latest `evidence_tree_v0` files, sync and re-check before cleanup.

## A. Recommend Keep

| Path | Git tracked? | Suggested action | Reason | Deletion risk |
|---|---:|---|---|---|
| `GIT_SYNC_STATUS.md` | Missing locally | Keep if restored | User explicitly listed it as keep. It is not present in this checkout, so cleanup should not create or delete it. | High: deleting on another checkout could remove sync state. |
| `CURRENT_EXPERIMENT_STATUS.md` | Yes | Keep | Primary experiment ledger and recent v7/v48 conclusions. | High: losing chronology and run constraints. |
| `docs/CODEX_HANDOFF.md` | Yes | Keep | Cross-session handoff and current constraints. | High: losing reproducibility context. |
| `docs/diagnostics/superpoint_action_diag_40scenes_v7/` | Yes, 3 files | Keep | v7 40-scene stable export-only baseline action tables. | High: current frozen v7 diagnostic baseline. |
| `docs/diagnostics/superpoint_action_diag_40scenes_v7_expansion_review/` | Yes, 6 files | Keep | Final 40-scene expansion review summary and compact CSVs. | High: baseline summary for future larger export-only runs. |
| `docs/visual_checks/superpoint_action_review_40scenes_v7_key_lists/` | Yes, 111 files | Keep for now; optionally archive later | Final/key-list 40-scene v7 visual baseline, including the poster addendum. | Medium-high: large but still the visual evidence for the accepted v7 baseline. |
| `docs/diagnostics/superpoint_action_diag_48scenes_v7_ap_summary/` | Yes, 10 files | Keep | Compact final AP-facing conclusion: v7 geometry stable, current matching/fusion gives no AP gain. | Medium-high: this is the concise replacement/refinement closure record. |
| Latest `evidence_tree_v0` files | Not found locally | Keep if found after sync | User explicitly listed latest evidence-tree work as keep. No matching files are present in this checkout. | High if present elsewhere. |
| `tools/analyze_superpoint_candidate_diagnostics.py` | Yes | Keep | Core v7 frozen diagnostic rule implementation and review-list generator. | High: needed to reproduce v7 action tables. |
| `tools/summarize_superpoint_action_expansion.py` | Yes | Keep | Generates expansion summaries used by v6/v7 diagnostics and may still be useful for larger export-only stability checks. | Medium: can be regenerated, but useful. |
| `tools/visualize_superpoint_action_review_candidates.py` | Yes | Keep | Current visual review generator for review lists; useful beyond v7. | Medium-high: needed if future export-only lists are visualized. |
| `tools/export_mask_graph_proposals.py` | Yes | Keep | Core export-only candidate generation path. | High. |
| `run_evaluation.py` | Yes | Keep | Core AP/evaluation entry point. | High. |
| `utils/backprojection_fusion.py` | Yes | Keep | Core current fusion logic. | High. |
| `tools/export_backprojection_candidates.py` | Yes | Keep | Core candidate export path used by current pipeline. | High. |
| `tools/export_sam_fused_proposals.py` | Yes | Keep | Core SAM-fused proposal export path used by current pipeline. | High. |

## B. Recommend Delete Or Archive Old Diagnostics / Visual Checks

These are not deletion commands. They are candidates for a later explicit `git rm` or archive move.

| Path | Git tracked? | Suggested action | Reason | Deletion risk |
|---|---:|---|---|---|
| `docs/diagnostics/superpoint_action_diag_10scenes/` | Yes, 2 files | Delete or archive | Early pre-v4/v2 diagnostic snapshot superseded by v7 40-scene baseline and v7 48-scene AP summary. | Low: historical only. |
| `docs/diagnostics/superpoint_action_diag_10scenes_v2/` | Yes, 3 files | Delete or archive | Old intermediate action rules superseded by v7. | Low. |
| `docs/diagnostics/superpoint_action_diag_10scenes_v4/` | Yes, 3 files | Delete or archive | v4 was a tuning stage; conclusions are captured in status docs and later v7 summaries. | Low-medium: keep only if detailed v4 regression archaeology is needed. |
| `docs/diagnostics/superpoint_action_diag_10scenes_v5/` | Yes, 3 files | Delete or archive | Superseded by 20/40-scene v7. | Low. |
| `docs/diagnostics/superpoint_action_diag_10scenes_v5_label_analysis/` | Yes, 3 files | Delete or archive | Human-label alignment was used to shape v5/v6/v7, no longer active. | Low-medium: useful only for rule-history audit. |
| `docs/diagnostics/superpoint_action_diag_20scenes_v5/` | Yes, 3 files | Delete or archive | 20-scene v5 intermediate, superseded by v7. | Low. |
| `docs/diagnostics/superpoint_action_diag_20scenes_v5_expansion_review/` | Yes, 5 files | Delete or archive | v5 expansion review was intermediate. | Low. |
| `docs/diagnostics/superpoint_action_diag_20scenes_v6/` | Yes, 3 files | Delete or archive | v6 intermediate, superseded by v7. | Low. |
| `docs/diagnostics/superpoint_action_diag_20scenes_v6_expansion_review/` | Yes, 6 files | Delete or archive | v6 intermediate expansion review. | Low. |
| `docs/diagnostics/superpoint_action_diag_20scenes_v6_1/` | Yes, 3 files | Delete or archive | v6.1 soft/thin diagnostic supplement, superseded by v7 final behavior. | Low-medium: keep only if auditing blanket/laptop rule history. |
| `docs/diagnostics/superpoint_action_diag_20scenes_v6_1_expansion_review/` | Yes, 7 files | Delete or archive | v6.1 supplement review; conclusion is folded into v7. | Low-medium. |
| `docs/diagnostics/superpoint_action_diag_20scenes_v7/` | Yes, 3 files | Archive or delete after confirming 40-scene baseline is enough | 20-scene v7 was a smaller staging run; 40-scene v7 is the current baseline. | Medium: useful if someone wants the exact 20-scene transition from v6.1 to v7. |
| `docs/diagnostics/superpoint_action_diag_20scenes_v7_expansion_review/` | Yes, 6 files | Archive or delete after confirming 40-scene baseline is enough | Superseded by 40-scene expansion review. | Medium. |
| `docs/visual_checks/superpoint_action_review_v4/` | Yes, 38 files, 6.4M | Delete or archive | Old v4 human review PNGs; superseded by v7 baseline visuals. | Low-medium: only needed for v4 visual audit. |
| `docs/visual_checks/superpoint_action_review_v5/` | Yes, 37 files, 6.4M | Delete or archive | Old v5 visual review PNGs. | Low-medium. |
| `docs/visual_checks/superpoint_action_review_20scenes_v5/` | No, 49 files, 8.2M | Report only; do not delete without explicit approval | Untracked dirty old v5 visual directory. It is not in Git and should be handled carefully. | Medium: untracked user/local output could be intentionally kept. |
| `docs/visual_checks/superpoint_action_review_20scenes_v5_final/` | Yes, 67 files, 12M | Delete or archive | v5 final review was superseded by v6/v7. | Low-medium. |
| `docs/visual_checks/superpoint_action_review_20scenes_v6_final/` | Yes, 67 files, 12M | Delete or archive | v6 final review was superseded by v7. | Low-medium. |
| `docs/visual_checks/superpoint_action_review_20scenes_v6_1_soft_thin/` | Yes, 3 files, 452K | Delete or archive | Focused v6.1 soft/thin check superseded by v7 rule closure. | Low-medium. |
| `docs/visual_checks/superpoint_action_review_20scenes_v7_final/` | Yes, 69 files, 12M | Archive or delete after confirming 40-scene visuals are enough | 20-scene v7 visual staging was superseded by 40-scene key-list visuals. | Medium: still useful to inspect the exact 20-scene rule closure. |
| `docs/visual_checks/superpoint_largest_cc/` | Yes, 6 files, 568K | Delete or archive | Early largest-CC visual sanity checks; superseded by action review visualizations. | Low. |
| `docs/visual_checks/superpoint_largest_cc_large_candidates/` | Yes, 28 files, 3.2M | Delete or archive | Early large-candidate visual checks; superseded by v7 key lists. | Low. |

## B2. Compact Summaries To Keep Even If Old Details Are Removed

| Path | Git tracked? | Suggested action | Reason | Deletion risk |
|---|---:|---|---|---|
| `docs/diagnostics/superpoint_action_diag_48scenes_v7_ap_summary/v7_48scenes_ap_summary.md` | Yes | Keep | Compact record of failed AP conversion attempts and next direction. | High: should remain if detailed CSVs are pruned later. |
| `docs/diagnostics/superpoint_action_diag_48scenes_v7_ap_summary/ap_experiment_summary.csv` | Yes | Keep | Small machine-readable AP branch summary. | Medium-high. |
| `docs/diagnostics/superpoint_action_diag_48scenes_v7_ap_summary/unconditional_replacement_class_mismatch_examples.csv` | Yes | Keep or archive with summary | Small evidence sample for why direct replacement failed. | Medium. |

## C. Scripts: Delete / Archive Candidates Vs Reusable

### C1. `tools/*.py` that appear dedicated to stopped v7 replacement/refinement AP experiments

| Path | Git tracked? | Suggested action | Reason | Deletion risk |
|---|---:|---|---|---|
| No committed `tools/*.py` found for the v7 replacement/refinement AP experiments | N/A | Nothing to delete | The replacement/refinement AP experiments were run via temporary `/tmp` scripts and reports, not committed repo tools. | N/A. |

### C2. `tools/*.py` that are old-diagnostic-specific and could be archived only after confirmation

| Path | Git tracked? | Suggested action | Reason | Deletion risk |
|---|---:|---|---|---|
| `tools/analyze_superpoint_visual_label_alignment.py` | Yes | Archive or keep short-term | Mostly served v5 human-label alignment. It is not needed for frozen v7 execution, but useful if rule-history analysis is revisited. | Medium: loses ability to reproduce v5 label-alignment reports. |

### C3. `tools/*.py` still likely reusable and should not be deleted

| Path | Git tracked? | Suggested action | Reason | Deletion risk |
|---|---:|---|---|---|
| `tools/analyze_superpoint_candidate_diagnostics.py` | Yes | Keep | Owns v7 frozen rule diagnostics and review lists. | High. |
| `tools/summarize_superpoint_action_expansion.py` | Yes | Keep | Useful for any future export-only stability or score re-ranking diagnostics. | Medium-high. |
| `tools/visualize_superpoint_action_review_candidates.py` | Yes | Keep | Generates PNG visual review sets from review lists. | Medium-high. |
| `tools/analyze_applied_mask_graph_candidates.py` | Yes | Keep | General analysis helper for applied mask-graph candidates. | Medium. |
| `tools/analyze_mask_graph_trace_relations.py` | Yes | Keep | General trace/relation diagnostic helper. | Medium. |
| `tools/export_mask_graph_proposals.py` | Yes | Keep | Core mask graph export path. | High. |
| `tools/export_backprojection_candidates.py` | Yes | Keep | Core candidate export path. | High. |
| `tools/export_sam_fused_proposals.py` | Yes | Keep | Core SAM-fused candidate path. | High. |
| `tools/evaluate_multiview_object_clip_correction.py` | Yes | Keep | Potentially useful if moving toward score re-ranking / semantic correction. | Medium. |
| `tools/export_multiview_object_clip_features.py` | Yes | Keep | Potentially useful for score re-ranking direction. | Medium. |
| `tools/search_multiview_clip_correction_rules.py` | Yes | Keep | Potentially useful for re-ranking / correction rules. | Medium. |
| `tools/train_candidate_geometry_discriminator.py` | Yes | Keep | Could become useful for class-aware scoring even if not direct replacement. | Medium. |
| `tools/train_semantic_fusion_head.py` | Yes | Keep | Potential score/re-ranking path. | Medium. |

## D. Absolutely Do Not Touch

| Path | Git tracked? | Suggested action | Reason | Deletion risk |
|---|---:|---|---|---|
| `data/` | Mixed / external | Do not touch | Dataset root. | Critical. |
| `output/` | Mostly generated, large | Do not touch in this cleanup | Contains experiment outputs and caches; user explicitly forbids submitting/deleting as part of this plan. | Critical. |
| `pretrained/` | Mixed / external | Do not touch | Model config/weights area. | Critical. |
| Weight files (`*.pth`, `*.pt`, `*.ckpt`, etc.) | Mixed | Do not touch | Large model artifacts or prediction caches. | Critical. |
| `models/Mask3D/mask3d.egg-info/PKG-INFO` | Tracked but dirty | Do not touch | Known dirty Mask3D build/package artifact. | High: unrelated local build state. |
| `models/Mask3D/third_party/pointnet2/build/` | Tracked/dirty build files | Do not touch | Known dirty compiled artifacts. | High: unrelated local build state. |
| `models/Mask3D/third_party/pointnet2/dist/pointnet2-0.0.0-py3.10-linux-x86_64.egg` | Tracked but dirty | Do not touch | Known dirty compiled distribution artifact. | High. |
| `/tmp/mask_graph_proposals_scannet200_superpoint_largest_cc_diag_48scenes_v7` | Outside repo | Do not touch | Large temporary export-only directory; not part of Git cleanup. | Medium-high: still useful for local reproduction. |
| `/tmp/even48_v7_ap_control` | Outside repo | Do not touch | Local AP reference/control outputs. | Medium-high. |
| `/tmp/even48_v7_replacement_ap` | Outside repo | Do not touch unless explicitly asked | Failed direct replacement AP outputs; summarized in docs. | Medium. |
| `/tmp/even48_v7_class_consistent_replacement_ap` | Outside repo | Do not touch unless explicitly asked | Failed class-consistent replacement AP outputs; summarized in docs. | Medium. |
| `/tmp/even48_v7_expansion_only_same_class_refinement_ap` | Outside repo | Do not touch unless explicitly asked | Expansion-only AP outputs; summarized in docs. | Medium. |
| `vibe_research_report.tex` | No, untracked | Do not touch | User-owned course/report artifact. | High. |
| `docs/visual_checks/superpoint_action_review_20scenes_v5/` | No, untracked | Report only, no deletion in this pass | Old v5 visual directory is dirty/untracked. Needs explicit approval before deletion. | Medium-high. |

## Suggested Cleanup Order If Approved Later

1. First remove or archive tracked old visual PNG directories, because they are the largest repo clutter:
   - `docs/visual_checks/superpoint_action_review_v4/`
   - `docs/visual_checks/superpoint_action_review_v5/`
   - `docs/visual_checks/superpoint_action_review_20scenes_v5_final/`
   - `docs/visual_checks/superpoint_action_review_20scenes_v6_final/`
   - `docs/visual_checks/superpoint_action_review_20scenes_v6_1_soft_thin/`
   - optionally `docs/visual_checks/superpoint_action_review_20scenes_v7_final/`
2. Then remove/archive old compact diagnostics from v4-v6.1 and 20-scene staging.
3. Keep the 40-scene v7 baseline and 48-scene v7 AP summary until the next direction has a stronger replacement record.
4. Only after a separate confirmation, decide whether the untracked `docs/visual_checks/superpoint_action_review_20scenes_v5/` should be deleted locally.
