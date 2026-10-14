# 候选条件局部修复能力检查

固定开发200内上一轮非GT选择的10场景、80父候选，未知父候选7；不按GT补选。原五折×旧90/新110，N/T均覆盖，锚失败父候选保留。全部局部域和有效帧清单先封存。

检查磁盘上13248个有效RGB-D-pose帧，不运行SAM或学习模型。全部输出为后置GT理想修复诊断，不能当预测、折外收益或实例AP。

|范围/证据|父提升≥.02|超过完整NT≥.02的目标|有价值目标/场景|跨>.75|跨>.90|跨>.95|
|---|---:|---:|---:|---:|---:|---:|
|old_boundary/unrestricted|22|17|17/9|0|8|9|
|old_boundary/cached30_1|11|5|6/5|0|2|3|
|old_boundary/cached30_2|8|5|5/5|0|1|1|
|old_boundary/all_frames_1|22|17|17/9|0|6|8|
|old_boundary/all_frames_2|18|14|14/8|0|5|6|
|old_boundary/superpoint|22|17|17/9|0|8|9|
|old_boundary/voxel2cm|21|16|17/9|0|7|7|
|whole_parent/unrestricted|25|19|19/9|1|9|12|
|whole_parent/cached30_1|12|6|7/5|0|3|3|
|whole_parent/cached30_2|8|5|5/5|0|1|2|
|whole_parent/all_frames_1|25|19|19/9|1|9|10|
|whole_parent/all_frames_2|21|16|16/8|1|8|8|
|whole_parent/superpoint|25|19|19/9|1|9|12|
|whole_parent/voxel2cm|23|18|19/9|1|9|10|
|parent_halo15/unrestricted|62|32|32/10|1|13|22|
|parent_halo15/cached30_1|44|13|15/8|1|5|6|
|parent_halo15/cached30_2|30|7|8/6|0|1|3|
|parent_halo15/all_frames_1|61|31|31/10|1|13|21|
|parent_halo15/all_frames_2|58|26|26/9|1|13|16|
|parent_halo15/superpoint|62|32|32/10|1|13|22|
|parent_halo15/voxel2cm|54|24|25/9|1|12|18|

旧漏斗同时保留“所有非目标点”和“有效其他实例点”两个口径，按候选累计：

```json
{
  "all_non_target": {
    "editable_outside_core": 2955,
    "old_after_positive_veto": 0,
    "old_excluded_2groups": 0,
    "old_valid_2groups": 8,
    "parent_pollution": 4461,
    "reliable_anchors": 21,
    "reprompt_after_positive_veto": 0,
    "reprompt_excluded_2groups": 0,
    "reprompt_valid_2groups": 0,
    "selected_visible_2groups": 221
  },
  "valid_other_instance": {
    "editable_outside_core": 686,
    "old_after_positive_veto": 0,
    "old_excluded_2groups": 0,
    "old_valid_2groups": 0,
    "parent_pollution": 819,
    "reliable_anchors": 21,
    "reprompt_after_positive_veto": 0,
    "reprompt_excluded_2groups": 0,
    "reprompt_valid_2groups": 0,
    "selected_visible_2groups": 199
  }
}
```

old_boundary：父候选内旧.15m边界，排除旧core；whole_parent：整个父候选，只能删；parent_halo15：整个父候选及其.15m原始点邻域，可增删。后两者不保护N/T交集。全帧和30缓存帧使用相同的全帧物理分组。所有遮挡、缺深度及未观察错误点保持原父状态。

superpoint和voxel2cm为固定范围内整单元选择的理想上界；单元在范围边缘裁断，是比真正不可拆整超点更宽松的上界。<100点最优解单独计数，不能认定其通过评测。没有按GT选实际动作，未输出可用于SL的候选点集。

与旧保存可靠可见性的交叉检查：{"frames": 45, "points": 156625, "different_points": 0}。CPU double投影采用原深度内参、整数截断；少量像素边界差异须保留并说明，不能称逐位复现。

当前检查只覆盖固定10场景，不是开发200全量结论。几何学习仍需原物理场景五折、冻结骨干与SL；正负/混合/好候选保持监督，报告误删、漂移、重复及池级收益。几何成立后才获取新点集自身语义并重算完整扩池上下文，保留NT-only与允许新候选对照。
