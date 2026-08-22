# 第一创新点阶段 B：候选关系审计与集合级连续重排序预注册合同

状态：`frozen_before_first_run`

日期：2026-08-22

## 1. 启动条件与目标

阶段 A 已在 NCS-train100 上通过预注册门槛和独立审计。阶段 B 只使用阶段 A 的外折统一几何质量与冻结的无标签几何/关系资产，为候选组件建立关系分型账本，并生成一个不删除候选、不提高分数、native 始终不动的连续重排序计划。

本阶段不运行 AP，不读取 `NCS-validation60` 或 `ScanNet200 val312`，不修改 `DM-SMS-1`，不覆盖冻结冠军。

## 2. 候选集合

集合单位沿用冻结关系账本中的 `relation_component_id`：

- native exact-geometry group 通过其冻结成员编号连接；
- track 通过冻结 `track_id` 连接；
- pair-union 通过冻结父 track、父 native group 与 component 连接；
- 未进入任何关系组件的几何是 singleton，只记录 no-op。

阶段 A 的每个 `geometry_key` 必须恰好连接一次。组件中所有候选对一次性计算关系，禁止贪心修改后再重新计算关系。

## 3. 关系字段

每一对至少保存：

- 双方来源、点数和阶段 A 统一质量；
- intersection、union、IoU、双向覆盖率、点数比；
- 是否为冻结 pair-union 及其直接父候选；
- 已有 direct track-native 关系中的共同视角数、同一/不同观测比例、投影框 IoU、GVC差、边界接触、颜色和法向差；
- 是否有直接多视图关系证据。

## 4. 固定关系分型

按以下顺序确定唯一关系类型，不扫描阈值：

```text
exact_duplicate:
  geometry hash 相同；正常情况下已由上游折叠，出现时仅审计

complementary:
  pair-union 与其直接父候选，且 union 相对父候选新增点比例 >= 0.10

near_duplicate:
  双向覆盖率最小值 >= 0.90

containment:
  双向覆盖率最大值 >= 0.90，但最小值 < 0.90

conflict:
  同一冻结组件内 IoU < 0.05；或已有直接多视图证据中
  different matched fraction > same matched fraction

complementary:
  其余 IoU >= 0.05 的跨来源重叠关系

uncertain:
  其余关系
```

`exact_duplicate / near_duplicate / containment / complementary / conflict / uncertain` 六类必须都在模式表中；exact duplicate 在唯一几何账本后允许计数为 0。

## 5. 固定连续重排序规则

先按阶段 A `Q` 降序、native 优先、来源顺序、候选编号形成稳定集合顺序。全部动作一次性从原始 `Q` 计算，候选之间不做级联重算。

- native：始终保持 `Q`，作为安全回退；
- exact duplicate：上游已折叠，本阶段 no-op；
- near duplicate：仅对较低质量的非 native 候选应用
  `factor = 1 - 0.5 * min(双向覆盖率)`；
- containment：若被包含候选是非 native 且其 `Q` 不高于包含者，应用
  `factor = 1 - 0.5 * max(双向覆盖率)`；否则作为明确质量例外 no-op；
- complementary：no-op，保留互补候选和 pair-union；
- conflict：存在 native 一方时，非 native 的最终质量不得高于该 native 的 `Q`；没有 native 时 no-op；
- uncertain：no-op。

一个候选受到多条关系影响时只采用最强降分，即最小 factor；不累乘。最终分数固定为：

```text
stage_b_score = stage_a_quality * strongest_factor
```

本阶段不删除候选、不修改几何、不修改类别、不提高分数，也不把该计划写回冻结缓存。

## 6. 输出与审计

输出：

- 组件账本；
- 全部 pair 关系账本；
- 每个 geometry 的阶段 A 分数、最强关系、factor 与阶段 B 计划分数；
- 关系类型、来源对、五折、动作原因和低证据 no-op 统计；
- 输入摘要及 SHA-256。

独立审计必须重新计算几何关系和所有计划分数，并检查完整覆盖、组件隔离、native 不变、无分数提高、无候选删除及无 validation/val312/AP。

## 7. 阶段 B 推进门槛

只有以下条件同时满足，才授权阶段 C 的 pair-union 成员细化：

1. 阶段 B 独立审计错误数为 0；
2. 全部 `10,169` 个几何恰好覆盖一次；
3. native 计划分数逐位等于阶段 A 质量；
4. 所有计划分数有限且不高于阶段 A 质量；
5. pair-union 父关系中至少存在一个 complementary；
6. near duplicate 或 containment 至少存在一种且数量大于 0；
7. 五折均至少有一个非 native 候选发生连续降分；
8. 候选删除、几何修改、类别修改和 AP 计算均为 0。

失败时保留关系诊断，但不扫描阈值、不运行 AP、不读取冻结验证集。
