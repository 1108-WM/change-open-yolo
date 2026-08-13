# OpenYOLO3D 输出清理清单

日期：2026-08-11

本清单记录用户已授权删除的实验产物。大小按删除前 `du -s --block-size=1` 的实际磁盘占用计算。

执行状态：**已完成**。删除后复核 72 个顶层 smoke 目录和其余 17 个明确目标均不存在；当前冠军模型、official100 最终 AP、有效 pair-union v2 计划和既有 safety60 评测仍存在。删除后根分区可用空间为 `215,413,403,648` bytes。

## 汇总

- 明确安全项：`762,515,456` bytes（约 `0.76 GB`）。
- 已终止分支：`18,833,514,496` bytes（约 `18.83 GB`）。
- 合计：`19,596,029,952` bytes（约 `19.60 GB`）。

删除不包括当前 official100 冠军模型、pair-union v2 计划、official100/safety60/even48 正式评测结果、正式大型数据缓存或用户 Git 修改。

## 明确安全项

### 72 个 `output/` 顶层 smoke 目录

删除前已验证：以下路径均为 `output/` 的直接子目录，数量 `72`，合计 `338,100,224` bytes。

```text
output/automatic_mask_tracks_details_consensus_gvc_safety60_uniform30_p16_v0_20260804_smoke
output/automatic_mask_tracks_details_consensus_hierarchy_safe_gvc_safety60_uniform30_d1_smoke_20260805
output/automatic_mask_tracks_details_siou_even48_f30_p16_v0_20260802_smoke
output/automatic_mask_tracks_details_siou_gvc_safety60_uniform30_p16_v0_20260804_smoke
output/automatic_mask_tracks_details_siou_hierarchy_safe_gvc_safety60_uniform30_d1_smoke_20260805
output/c1_gvc_paper_reference_quality_ledger_gvc_safety60_uniform30_20260807_smoke1
output/c1_gvc_relative_competition_ledger_gvc_safety60_uniform30_20260807_smoke1
output/c1d_expanded_competition_action_ledger_gvc_safety60_20260807_smoke1
output/c1d_native_geometry_folding_audit_gvc_safety60_20260807_smoke1
output/counterevidence_reliability_smoke_even48_f30_v0_20260802
output/counterevidence_superpoint_variants_smoke_even48_f30_v0_20260801
output/d2b_merge_family_split_action_ledger_smoke_scene0015_20260807
output/d2b_native_competition_component_ledger_gvc_safety60_uniform30_20260807_smoke1
output/details_consensus_proposal_relation_graph_gvc_safety60_uniform30_d2_smoke_20260805
output/details_consensus_refactor_regression_gvc_safety60_uniform30_smoke_20260805
output/details_d2b_native_mutual_duplicate_filtered_gvc_safety60_smoke_20260805
output/details_iterative_proposal_merges_exact_frame_union_gvc_safety60_uniform30_d2b_smoke_20260805
output/details_iterative_proposal_merges_gvc_safety60_uniform30_d2b_smoke_20260805
output/details_proposal_suppression_cleanup_gvc_safety60_uniform30_d2c_smoke_20260805
output/details_same_frame_details_exact_gvc_safety60_uniform30_d1_smoke_20260805
output/details_same_frame_hierarchy_safe_even48_uniform30_d1_smoke_20260805
output/details_same_frame_hierarchy_safe_gvc_safety60_uniform30_d1_smoke_20260805
output/even48_yoloworld_sam_observations_uniform30_cpu_smoke1_20260810
output/groundingdino_boxes_smoke_even48_f30_v0_20260801
output/gvc_append_only_smoke_scene0011_00_20260803
output/gvc_append_only_smoke_scene0011_00_20260803_preflight
output/gvc_append_only_smoke_scene0011_00_20260803_v2
output/gvc_append_only_smoke_scene0011_00_20260803_v2_preflight
output/gvc_safety60_uniform30_rle_smoke_20260804
output/multiview_fragment_family_competition_materialized_gvc_safety60_uniform30_f2_smoke_20260805
output/multiview_fragment_family_quality_gvc_safety60_uniform30_f2_smoke_20260805
output/multiview_fragment_merge_materialized_gvc_safety60_uniform30_f1_smoke_20260805
output/multiview_fragment_relation_ledger_gvc_safety60_uniform30_f1_smoke_20260805
output/mv3dis_3d_guide_mask_matching_ledger_gvc_safety60_uniform30_m1a_smoke_20260805
output/mv3dis_baseline_adapted_grow_gvc_safety60_uniform30_m1adapt_smoke_20260805
output/mv3dis_baseline_adapted_move_gvc_safety60_uniform30_m1adapt_smoke_20260805
output/mv3dis_baseline_adapted_resolve_gvc_safety60_uniform30_m1adapt_smoke_20260805
output/mv3dis_boundary_preassignment_ledger_gvc_safety60_uniform30_m1b_smoke_20260805
output/mv3dis_global_boundary_assignment_plan_fragment_family_f2_gvc_safety60_uniform30_smoke_20260805
output/mv3dis_grow_native_mutual_duplicate_filtered_gvc_safety60_smoke_20260805
output/mv3dis_relative_depth_boundary_preassignment_gvc_safety60_uniform30_m1depth_smoke_20260805
output/mv3dis_relative_depth_guide_mask_matching_gvc_safety60_uniform30_m1depth_smoke_20260805
output/mv3dis_relative_depth_observations_gvc_safety60_uniform30_m1depth_smoke_20260805
output/n1_sampro3d_observation_candidate_ledger_paper_reference_smoke_scene0015_20260806
output/n1_sampro3d_prompt_observations_paper_reference_smoke16x2_20260806
output/n1_sampro3d_prompt_plan_paper_reference_smoke16x2_20260806
output/n1_sampro3d_seed_view_ledger_paper_reference_gvc_safety60_uniform30_d2b_20260806_smoke
output/n2_medoid_candidate_quality_competition_ledger_smoke_scene0015_20260807
output/n2_medoid_candidate_quality_competition_ledger_smoke_scene0015_20260807_v2
output/n2_medoid_coexist_candidate_cache_smoke_scene0015_20260806
output/n2_medoid_d2b_competition_plan_smoke_scene0015_20260806
output/n2_medoid_conservative_ranking_abstention_plan_smoke_scene0015_20260807_v2
output/n2_medoid_d2b_competition_ledger_smoke_scene0015_20260806
output/n2_medoid_dinov2_appearance_ledger_smoke_scene0015_20260807
output/n2_sampro3d_candidate_family_cleanup_plan_smoke_scene0015_20260806
output/n2_sampro3d_candidate_family_quality_ledger_smoke_scene0015_20260806
output/n2_sampro3d_family_formation_variant_ledger_smoke_scene0015_20260806
output/sam_automatic_observations_even48_uniform30_rle_smoke_20260805
output/smoke_diagnose_candidate_component_geometry_ceiling_gt_official100_v1_2scenes
output/smoke_train_candidate_pair_intersection_utility_ledger_official100_v1_2scenes
output/smoke_train_candidate_pair_union_utility_ledger_official100_v1_2scenes
output/smoke_train_candidate_track_marginal_harm_ledger_official100_v1_2scenes
output/track_gvc_feature_ledger_even48_f30_v0_20260802_smoke
output/track_gvc_feature_ledger_even96_uniform30_v0_20260802_smoke
output/track_gvc_feature_ledger_odd96_uniform30_v0_20260802_smoke
output/track_native_competition_ledger_gvc_safety60_d2b_grow_smoke_20260805
output/track_native_mutual_duplicate_plan_gvc_safety60_d2b_grow_smoke_20260805
output/train_candidate_component_action_utility_ledger_official100_v1_smoke_scene0001_01
output/train_candidate_relation_feature_ledger_official100_v1_smoke_scene0001_01
output/visibility_counterevidence_ledger_smoke_even48_f30_v0_20260801
output/yoloe_mask_observations_smoke_even48_f30_v0_20260801
output/yoloe_mask_observations_smoke_even48_f30_v1_20260801
```

### 其他安全项

```text
output/scannet200/native_cache_no_gt_even96_20260802_smoke          388,902,912 bytes
output/train_candidate_pair_union_oof_plan_official100_v1          26,861,568 bytes
.pytest_cache                                                          90,112 bytes
__pycache__                                                            110,592 bytes
tools/__pycache__                                                    6,983,680 bytes
tests/__pycache__                                                    1,466,368 bytes
```

`train_candidate_pair_union_oof_plan_official100_v1` 是无效计划；有效计划为 v2。

## 已终止分支

```text
output/n1_sampro3d_prompt_paper_reference_gvc_safety60_uniform30_d2b_20260806                    5,417,955,328 bytes
output/n1_sampro3d_observation_candidate_ledger_paper_reference_gvc_safety60_uniform30_d2b_20260806 5,351,272,448 bytes
output/n2_medoid_d2b_competition_ledger_paper_reference_gvc_safety60_uniform30_d2b_20260806       1,273,610,240 bytes
output/n2_sampro3d_family_formation_variant_ledger_paper_reference_gvc_safety60_uniform30_d2b_20260806 159,162,368 bytes
output/n2_medoid_coexist_candidate_cache_paper_reference_gvc_safety60_uniform30_d2b_20260806        115,568,640 bytes
output/n2_medoid_candidate_quality_competition_ledger_paper_reference_gvc_safety60_uniform30_d2b_20260807 64,032,768 bytes
output/gvc_safety60_mask3d_paired_completion_geom_dev30_v0_20260805                              2,981,376,000 bytes
output/gvc_safety60_automatic_sam_pareto_20260803                                                2,881,503,232 bytes
output/details_same_frame_details_exact_gvc_safety60_uniform30_d1_20260805                         403,550,208 bytes
output/details_iterative_proposal_merges_details_exact_frame_union_gvc_safety60_uniform30_d2b_20260805 110,575,616 bytes
output/details_proposal_suppression_cleanup_details_exact_gvc_safety60_uniform30_d2c_20260805       74,907,648 bytes
```

这些分支已由后续结论覆盖：SAMPro3D/medoid 未形成可推进系统；旧 GVC paired/pareto、Details exact/merge/suppression 不是当前冠军链路。必要结论已并入状态文档，删除原始大输出不影响当前冠军复现。
