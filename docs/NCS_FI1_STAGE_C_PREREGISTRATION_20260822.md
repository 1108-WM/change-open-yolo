# 第一创新点阶段 C：pair-union 成员级细化预注册合同

状态：`frozen_before_first_run`

日期：2026-08-22

## 1. 启动条件与边界

阶段 B 已在 `NCS-train100` 上通过独立审计。阶段 C 只处理冻结冠军计划中的 1,578 个 pair-union；原始 union、父 track、父 native、类别、冻结缓存和阶段 B 计划全部只读。

本阶段不运行 AP，不读取 `NCS-validation60` 或 `ScanNet200 val312`，不修改 `DM-SMS-1`，不覆盖冻结冠军。细化候选只能 append-only 追加；任何失败均显式回退并保留原始 union。

## 2. 固定成员分解

每个原始 union 按 prepared scene 第 10 列的冻结 raw superpoint 编号和父候选成员关系分解为原子：

```text
atom = raw_superpoint_id × {shared, track_only, native_only}
```

- `shared`：同时属于父 track 和父 native 的点；
- `track_only`：只属于父 track 的点；
- `native_only`：只属于父 native 的点；
- 原子点集必须两两不交且完整覆盖原始 union；
- 禁止向原始 union 外新增点。

## 3. 连续保留目标和特征

训练标签只在 `NCS-train100` 离线构建。先取原始 union 的 class-agnostic 最佳 GT 实例，再定义：

```text
atom_retention_target = atom 中属于该目标 GT 的点比例
```

这是一项连续纯度目标，不使用 winner/非 winner 二元标签。

模型特征不得读取 GT、scene 名称、候选编号或类别，包括：

- 原子角色、点数、占 union/父候选/原始超点比例；
- 原子相对 union 与父候选的质心、尺度、RGB、法向统计；
- 原子 raw-superpoint 图的度、边界接触、距离、颜色与法向连续性；
- 阶段 A 外折质量和阶段 B 分数；
- 冻结关系账本中的多视图投影支持、可见比例、GVC、共同深度一致观测、组件与边界统计。

模型固定为五折场景隔离的 `HistGradientBoostingRegressor`，参数固定；外层测试折的校准折固定为下一折，使用 isotonic 校准，退化时使用常数回退。不得扫描模型、阈值或来源权重。

## 4. 固定细化规则

- `shared` 原子无条件保留；
- `track_only/native_only` 仅在外折校准保留概率 `>= 0.50` 时进入临时候选；
- 在 raw-superpoint 邻接图上，只保留能够连接到任一 shared raw superpoint 的临时 exclusive 原子；若不存在 shared，则只保留点数最大的连通分量；
- refined union 点数不足 100、点集为空、覆盖合同失败或发生任何异常时，不生成替代动作，记录 `fallback_original_union`；
- refined union 与原始 union 完全相同时记录 no-op，不重复追加 exact geometry；
- 其余 refined union 只写入独立诊断目录，原始 union 始终保留。

## 5. refined Q 与最终分数

对物化后的 refined union 重新计算与阶段 A 完全相同的多阈值 `Q`。随后以外折成员概率聚合、父候选质量、阶段 B 分数和无 GT 关系统计训练第二个五折连续质量模型，输出 `stage_c_oof_quality`，作为本阶段最终诊断分数。

不得用 GT `Q` 直接充当可部署分数；GT 仅用于训练标签和独立审计。

## 6. 必须输出的审计

- 每个 pair-union 的原子完整覆盖、角色和点数账本；
- 原子连续标签、外折预测、可靠性和五折指标；
- 每个 refined union 的保留/删除原子数和点数、连通性回退、几何 SHA-256；
- 原始 Q、refined Q、改善/伤害/中性及五折方向；
- refined 最终外折分数的 MAE、偏差和可靠性；
- 原始 union 保留、无父候选修改、无 union 外新增点、无 AP/验证集读取的独立复算。

## 7. 阶段 C 推进门槛

只有以下条件同时满足，才授权阶段 D：

1. 数据与结果独立审计错误数均为 0；
2. 1,578 个原始 pair-union 全部覆盖，原子点集守恒；
3. 原子 OOF MAE `<= 0.25`、绝对平均偏差 `<= 0.10`，五折均有样本；
4. shared 点删除数为 0，union 外新增点数为 0；
5. 原始 union 删除或修改数为 0，refined 仅 append-only；
6. refined Q 总体均值严格高于原始 union Q；
7. 每折 refined Q 均值变化不低于 `-0.01`，不存在单折灾难；
8. refined 最终分数绝对平均偏差 `<= 0.10`；
9. AP、validation60、val312、冻结缓存写入均为 0。

门槛失败时保留诊断，不扫描阈值、不运行 AP，也不进入阶段 D。
