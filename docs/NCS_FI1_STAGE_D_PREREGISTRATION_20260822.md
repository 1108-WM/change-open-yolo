# 第一创新点阶段 D：连续边际收益与 append-only 排序预注册合同

状态：`frozen_before_first_run`

日期：2026-08-22

## 1. 启动条件与边界

阶段 A、B 和 C-v2 已在 `NCS-train100` 上通过各自预注册门槛及独立审计。阶段 D 只把冻结二元 `threshold-cross` 分支扩展为连续边际收益学习，不修改前三阶段历史产物，不扫描阈值、损失、来源权重或分数指数。

本阶段不运行 AP，不读取 `NCS-validation60` 或 `ScanNet200 val312`，不修改 `DM-SMS-1`，不覆盖冻结冠军或任何冻结缓存。native、track、原始 pair-union 和 C-v2 refined union 的几何与类别均只读；输出只能是独立的 append-only 排序计划，任何候选都不得删除。

## 2. 固定基线集合与动作集合

每个场景的已有集合固定为唯一几何账本中的全部 native 和 track 几何，共 `8,591` 个。已有集合不包含 pair-union，也不受阶段 B 降分影响；它只定义“加入 union 前已经覆盖到什么程度”。

阶段 D 动作集合固定为：

- `1,578` 个冻结原始 pair-union；
- 阶段 C-v2 已预先判定 `append_eligible_refined_union=true` 的 `85` 个 refined union；
- 总计 `1,663` 个 append 候选。

refined union 与其原始 union 是两个独立 append 候选，原始 union始终保留。不得根据阶段 D 标签或预测增加、删除或改写候选。

## 3. 固定连续标签

对场景中的每个有效 class-agnostic GT，先用全部 native+track 几何计算：

```text
existing_best_iou(gt) = max IoU(native_or_track, gt)
```

对每个 append 候选计算所有 GT 的 IoU，并固定主目标：

```text
marginal_iou_gain = max_gt max(0, candidate_iou(gt) - existing_best_iou(gt))
```

并同时记录下列只用于解释和对照的标签：

```text
candidate_best_iou
candidate_quality_Q                         # 阶段 A 的 0.50:0.05:0.95 十阈值 Q
existing_best_iou_for_selected_gain_target
marginal_Q_gain
softQ(iou) = clip((iou - 0.50) / 0.45, 0, 1)
marginal_soft_quality_gain
official_threshold_cross_count              # 冻结 0.50:0.05:0.90 对照
crosses_any_official_threshold
```

主训练目标只能是 `marginal_iou_gain`。GT 只能进入上述标签字段，不得进入模型特征、候选筛选或排序公式。

## 4. 固定无 GT 特征

每个候选记录：

- original/refined 变体、点数、场景占比、相对原始 union 的点数比例；
- 阶段 A 原始 union 统一质量、阶段 B union 分数、冻结 threshold-cross 概率；
- 父 track/native 的阶段 A 质量和冻结 union 关系字段；
- C-v2 的成员动作比例、成员多视图/相对深度聚合、refined-Q 预测与下置信界；
- 相对 native+track 基线几何的最大 IoU、双向覆盖率、点数比、候选新点比例、相交候选数；
- 相交基线候选的最大/平均阶段 B 分数；
- 阶段 B 六类关系数量和比例。

original 与 refined 使用同一固定特征模式。不存在的 refined 专属量用原始候选的安全 no-op 值表达，不允许用 GT 填补。

## 5. 五折模型、校准与保守收益

按冻结场景五折训练单个 `HistGradientBoostingRegressor`，参数固定为：

```text
learning_rate=0.05
max_iter=160
max_leaf_nodes=15
min_samples_leaf=30
l2_regularization=1.0
early_stopping=False
```

外层测试折的校准折固定为下一折，拟合只使用其余三折。仅做校准折平均残差的加性偏差校正，并从同一校准折计算绝对残差第 90 百分位 `q90`。预测裁剪到 `[0,1]`，不扫描阈值或分位数。

```text
conservative_gain = max(0, corrected_predicted_gain - calibration_q90)
```

## 6. append-only 安全评分

候选的无 GT 几何质量上界固定为：

- original：阶段 B union 分数；
- refined：C-v2 refined-Q 下置信界。

二者均裁剪到 `[0,1]`。阶段 D 独立分数固定为：

```text
stage_d_append_score = geometry_quality_for_scoring * conservative_gain
```

该分数不得高于 `geometry_quality_for_scoring`。`conservative_gain=0` 只表示没有可靠新增覆盖证据，不表示删除候选；所有 1,663 个候选仍完整写入计划并标记 retained。冻结 threshold-cross 概率和原分数只作为对照保留，不被覆盖。

## 7. 输出与独立审计

必须保存：

- 1,663 个候选的无 GT 特征、连续标签、几何定位器和全部输入 SHA-256；
- 每折拟合/校准/测试场景隔离、模型、偏差、q90 和 OOF 预测；
- OOF MAE、RMSE、Spearman、零预测及冻结 threshold-cross 概率对照；
- 连续正收益数量、六位小数不同值、五折分布；
- 高置信正收益数量、真实正收益/中性数量及五折分布；
- original/refined 分来源分布、关系类型和基线重叠统计；
- 候选删除、几何/类别修改、冻结缓存写入、AP 和禁用验证集读取统计；
- 全部输入输出与模型 SHA-256。

独立审计必须从原始 native/track/union/refined 点集重新计算已有最佳 IoU、候选 IoU、连续收益、softQ/Q/threshold-cross 对照、无 GT 重叠特征、模型 OOF 输出、q90、保守收益和最终分数。

## 8. 阶段 D 推进门槛

只有以下条件同时满足，才允许讨论一次性冻结迁移评测：

1. 数据与结果独立审计错误数均为 0；
2. `8,591` 个 native+track 基线几何、`1,578` 个 original 和 `85` 个 refined 候选完整覆盖；
3. `marginal_iou_gain>0` 的候选至少 `100` 个，六位小数不同正值至少 `50` 个，五折均有正收益候选；
4. OOF MAE 严格优于零预测对照和冻结 threshold-cross 概率数值对照；
5. OOF Spearman 严格大于 `0`；
6. `conservative_gain>0` 的候选数量大于 `0`，且五折均至少有一个；
7. 高置信候选中真实 `marginal_iou_gain>0` 的比例至少 `0.70`；预测动作不得产生负标签，真实零收益只记为中性；
8. 所有输出分数有限、非负且不高于对应几何质量；
9. 候选删除、几何修改、类别修改、父候选修改、冻结缓存写入均为 `0`；
10. AP、validation60 和 val312 读取均为 `0`。

失败时保留诊断，不扫描模型、特征、阈值、q90 分位数或评分公式，不运行 AP，也不进入冻结迁移评测。
