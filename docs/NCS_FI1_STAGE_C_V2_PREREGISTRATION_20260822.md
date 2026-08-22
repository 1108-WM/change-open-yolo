# 第一创新点阶段 C-v2：成员级多视图证据与保守 union 细化预注册合同

状态：`frozen_before_first_run`

日期：2026-08-22

## 1. 启动原因与边界

阶段 C-v1 的实现和独立审计错误数均为 0，且总体多阈值几何质量平均提升 `+0.0820`；但成员 OOF MAE 为 `0.2960`，第 4 折平均质量变化为 `-0.01034`，未通过预注册门槛。诊断确认 v1 的“连续纯度”标签实际只取 `0/1`，同时成员仅共享候选级多视图字段，无法充分区分同一 union 内的可靠成员与污染成员。

C-v2 是新的 `NCS-train100` 预注册分支，不修改 C-v1 历史产物，不扫描 v1 的 `0.50` 阈值，也不以第 4 折反向选择超参数。

本阶段不运行 AP，不读取 `NCS-validation60` 或 `ScanNet200 val312`，不修改 `DM-SMS-1`，不覆盖冻结冠军。原始 pair-union、父 track、父 native、类别和冻结缓存始终只读；refined union 仍只能 append-only。

## 2. 固定成员原子

沿用并独立复算 C-v1 的原子定义：

```text
atom = raw_superpoint_id × {shared, track_only, native_only}
```

原子必须两两不交、完整覆盖原始 union，且 refined union 不得包含原始 union 外的点。`shared` 原子始终强制保留，不进入删除模型。

## 3. 真正的成员级多视图与深度证据

对每个 atom，使用其父 track 的冻结 `observation_ids`，逐观测重新计算以下无 GT 特征：

- 与冻结提升点集的原子覆盖率、支持视角比例、逐点支持次数、多视图与未观测比例；
- 精确二维 RLE 对 atom 的逐视角覆盖率；
- 使用 MV3DIS 固定公式
  `wpd = 1 - |zc-d| / (0.05*d)`，且仅在 `|zc-d| < 0.05*d` 时可见；
- atom 的可见视角比例、mask 内深度加权覆盖、mask 内平均深度权重；
- 逐点可见次数、mask 内次数、可见条件支持率和多视图支持率；
- 观测 `predicted_iou`、`stability_score` 加权的原子支持；
- C-v1 已冻结的原子几何、RGB、法向、raw-superpoint 邻接/边界及父候选质量字段。

每个 atom 获得独立数值；禁止把候选级关系汇总复制后当作成员级证据。特征不得读取 GT、scene 名称、候选编号或类别。

## 4. 固定连续删除收益目标

先固定原始 union 的最佳 class-agnostic GT，不允许删除 atom 后切换目标。对每个 exclusive atom 记录：

```text
delta_iou_remove = IoU(original_union - atom, fixed_target)
                   - IoU(original_union, fixed_target)

delta_q_remove = Q(original_union - atom, fixed_target)
                 - Q(original_union, fixed_target)
```

其中 Q 与阶段 A 相同，使用 `0.50:0.05:0.95` 十个阈值。主训练目标固定为连续 `delta_iou_remove`；`delta_q_remove` 只用于解释和审计。目标必须包含非二元中间值，否则本分支直接失败。

## 5. 角色条件五折模型与不确定性

`track_only` 和 `native_only` 分别训练独立的 `HistGradientBoostingRegressor`；参数固定为：

```text
learning_rate=0.05
max_iter=160
max_leaf_nodes=15
min_samples_leaf=30
l2_regularization=1.0
early_stopping=False
```

五折仍按场景隔离。外层测试折的校准折固定为下一折。对模型输出只做校准折平均残差的加性偏差校正，并从同一校准折计算绝对残差第 90 百分位 `q90`。

exclusive atom 仅在以下条件同时满足时进入“可删除”集合：

```text
corrected_predicted_delta_iou - calibration_q90 > 0
```

不确定或预测接近 0 的成员默认保留，不扫描阈值。

## 6. 连通性与候选级质量保护

- shared 原子强制保留；
- 仅删除通过上述下置信界的 exclusive atom；
- 删除后只保留仍连接到任一 shared raw superpoint 的组件；若原 union 没有 shared，则不执行任何删除并回退原始 union；
- 临时 refined union 少于 100 点、核心断裂、覆盖异常或发生任何错误时回退原始 union；
- 对临时 refined union 重新计算无 GT 聚合特征，再训练五折 refined-Q 模型；
- refined-Q 模型同样使用下一折偏差校正和校准折绝对残差 q90；
- 只有当
  `predicted_refined_Q - q90 > stage_a_original_union_Q`
  时才标记为 append-eligible；否则显式回退原始 union；
- 原始 union 无条件保留，append-eligible 也不写入冻结缓存。

## 7. 输出与独立审计

必须保存：

- 每个 atom 的成员级多视图/深度证据和连续删除收益；
- 两个角色模型的五折预测、偏差校正、q90、MAE、Spearman 与零预测对照；
- 可删除、不确定、强制保留成员数量；
- 临时 refined union 和质量保护后的最终 append 计划；
- 原始/refined Q、改善/伤害/中性、五折方向和低证据回退；
- shared 删除、union 外新增点、父几何修改、候选删除和冻结缓存写入统计；
- 全部输入输出 SHA-256。

独立审计必须重新投影二维 RLE、重新计算相对深度权重、连续删除收益、模型预测、不确定性界、连通性和最终质量保护。

## 8. C-v2 推进门槛

只有以下条件同时满足，才允许把阶段 C 标记为通过并重新讨论阶段 D：

1. 数据与结果独立审计错误数均为 0；
2. 1,578 个 union 和全部 atom 完整覆盖；
3. exclusive `delta_iou_remove` 至少包含 100 个不同的六位小数值，确认目标未退化为二元；
4. `track_only`、`native_only` 的 OOF MAE 均严格优于各自零预测对照；
5. shared 删除数、union 外新增点数、原始 union 删除数均为 0；
6. 质量保护后的 append-eligible refined union 数量大于 0；
7. append-eligible refined union 的 GT 改善比例至少为 0.70，伤害比例至多为 0.10；
8. append-eligible refined Q 总体均值严格高于原始 Q，且五折平均变化均不低于 0；
9. 全部 1,578 个 union 计入保守回退后，总体平均 Q 不低于原始 union；
10. AP、validation60、val312、父几何修改和冻结缓存写入均为 0。

失败时保留诊断，不扫描模型、阈值、q90 分位数或来源权重，不运行 AP，也不进入阶段 D。
