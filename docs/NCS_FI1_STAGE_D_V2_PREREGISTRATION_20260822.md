# 第一创新点阶段 D-v2：高分前缀条件连续边际收益预注册合同

状态：`frozen_before_first_run`

日期：2026-08-22

## 1. 启动原因与边界

阶段 D-v1 的数据与结果独立审计均为 0 错误，但未通过推进门槛：直接回归 MAE `0.0080529`，劣于全零预测 `0.0044067`；101 个下置信界为正的候选中只有 19 个真实正收益。诊断显示 v1 把全部 native+track 几何视为已有集合，等价于接近 oracle 的全候选几何覆盖，而不是代码审查第 9 节要求的“当前高分集合”。这使原始 union 仅 170/1,578、refined union 仅 4/85 有正收益。

D-v2 只修正已有集合的定义，保留 D-v1 候选集合、连续标签形式、特征来源、模型参数、五折校准和安全评分，不根据 v1 结果扫描阈值、模型、q90 或来源权重。

本阶段仍只使用 `NCS-train100`，不运行 AP，不读取 `NCS-validation60` 或 `val312`，不修改 `DM-SMS-1`、冻结冠军、类别或任何几何；所有候选 retained，输出仍是独立 append-only 计划。

## 2. 固定候选和排序参考分数

动作集合不变：1,578 个 original union 加 85 个 C-v2 append-eligible refined union，共 1,663 个。

候选的排序参考分数固定为：

- original：阶段 B original union 分数；
- refined：C-v2 refined-Q 下置信界裁剪到 `[0,1]`。

native/track 的参考分数固定为阶段 B 分数。所有 native/track 在完全同分时排在 append 候选之前，因此候选的已有高分前缀定义为：

```text
rank_prefix(candidate) = {
  native_or_track | stage_b_score >= candidate_reference_score
}
```

该定义无阈值扫描；每个候选只与其冻结排序位置之前的 native/track 比较。

## 3. 固定 rank-conditioned 连续标签

对每个有效 GT：

```text
prefix_best_iou(gt, candidate)
  = max IoU(prefix_geometry, gt)

rank_conditioned_marginal_iou_gain
  = max_gt max(0, candidate_iou(gt) - prefix_best_iou(gt, candidate))
```

主训练目标固定为该连续值。同时记录 candidate Q、prefix target IoU、marginal Q、softQ 和 `0.50:0.05:0.90` threshold-cross 对照。GT 仍只进入标签，不能进入特征或排序参考分数。

## 4. 无 GT 特征

沿用 D-v1 的 162 个字段，但所有 `baseline_*` 重叠字段明确替换为 `rank_prefix_*`，并新增：

- `rank_prefix_geometry_count`；
- `rank_prefix_fraction_of_native_track`。

所有重叠 IoU、双向覆盖、点数比、候选新点比例、相交数量和重叠候选阶段 A/B 分数，只对该候选的 rank prefix 计算。其余阶段 A/B/C-v2、父关系和成员证据保持不变。

## 5. 模型、校准和安全评分

模型与 D-v1 完全相同：单个 `HistGradientBoostingRegressor`，参数固定为：

```text
learning_rate=0.05
max_iter=160
max_leaf_nodes=15
min_samples_leaf=30
l2_regularization=1.0
early_stopping=False
```

外层测试折的下一折固定作校准折，只做平均残差加性校正和绝对残差 q90，预测裁剪 `[0,1]`：

```text
conservative_gain = max(0, corrected_prediction - q90)
stage_d_v2_append_score = candidate_reference_score * conservative_gain
```

不扫描概率阈值、分位数、模型或评分公式。

## 6. 数据与结果门槛

数据门槛：

1. 独立审计错误数 0；
2. 8,591 个 native+track、1,578 original、85 refined 完整覆盖；
3. 正收益候选至少 250 个，六位小数不同正值至少 100 个，五折均有；
4. 每个 prefix 只含 reference score 不低于候选的 native/track；
5. 无删除、几何/类别修改、冻结缓存写入、AP 或禁用验证集读取。

结果门槛沿用 D-v1：

1. 结果独立审计错误数 0；
2. OOF MAE 严格优于全零预测及冻结 threshold-cross 概率数值对照；
3. OOF Spearman > 0；
4. conservative gain 大于 0 且五折均有；
5. 高置信候选真实正收益比例至少 0.70；
6. 分数有限、非负且不高于候选参考分数；
7. 候选删除、几何/类别修改、冻结缓存写入、AP、validation60 和 val312 读取均为 0。

失败时保留诊断，不扫描模型、阈值、q90、排序参考分数或评分公式，不运行 AP，不进入冻结迁移评测。
