# FI1-D-v3 几何证据层与开放词汇语义假设层分离：S0/S1 预注册

状态：`frozen_before_implementation_and_before_any_new_GT_or_model_run`

日期：2026-09-07

分支：`fi1-d-v3-geometry-semantic-separation-20260907`

底座提交：`e4642e1e73b55ad0523a0d776cdbf22f78ae7be8`

## 1. 动机与失败边界

2026-09-07 的 ScanNet200 val312 冻结开放词汇评测已证明：DM-SMS-1 可以将同合同 FI1-D-v3 控制从 `0.194585711` 提升到 `0.224606818` AP，但最终仍低于 OpenYOLO3D `0.247` 基线。独立审计零错误，因此该结果不是 evaluator 或缓存篡改造成。

失败原因固定为待修复的数据接口：类别无关 exact-geometry 层将多个原始类别假设压缩为每几何一个 canonical 类别，且 FI1-D-v3 类别无关 `challenger_score` 被直接用于类别相关 AP 排序。

本预注册只冻结 S0/S1：重建原始语义成员谱系，建立几何证据与语义假设的双层账本和独立审计。它不冻结最终分数公式、候选截断、Qwen 任务选择或新 AP 评测。

## 2. 冻结输入

S0/S1 只读以下无 GT 账本：

1. FI1-Legacy canonical unique-geometry ledger；
2. duplicate-safe FI1-D-v3 joint unique-geometry ledger；
3. 官方 val312 场景列表，只用于场景身份覆盖。

当前 val312 冻结计数为：

```text
scene_count                         312
legacy_geometry_count              39,198
legacy_member_count               213,233
legacy_native_member_count        187,200
legacy_track_member_count          18,141
legacy_pair_union_member_count      7,892
fi1_d_v3_candidate_count           39,304
fi1_d_v3_unique_geometry_count     39,250
fi1_d_v3_refined_union_count          106
expected_semantic_hypothesis_count 213,339
```

`213,339 = 213,233 + 106`。每个 FI1-Legacy 原始成员形成一条语义假设；每个 FI1-D-v3 refined union 形成一条新的 append-only 语义假设。原始 FI1-D-v3 canonical 记录不再额外生成第二份语义假设，因为它已对应 FI1-Legacy 几何组中的一个成员。

## 3. 几何证据层

唯一身份：

```text
(scene_name, geometry_hash)
```

每条几何记录必须包含：

- point count 与只读 geometry locator；
- 全部 FI1-D-v3 plan members；
- 全部语义假设 key；
- legacy/refined 成员数；
- `expensive_visual_evidence_execution_count=1`；
- 无 GT、无 AP、无几何/类别/分数修改标志。

几何证据共享只允许避免重复运行 Alpha/SAM/DINO/Qwen 视觉前处理，不允许折叠语义假设。

## 4. 语义假设层

每条语义假设使用唯一 `semantic_hypothesis_key`，并必须保存：

- scene/geometry 身份；
- `origin_kind = legacy_member | fi1_refined_union`；
- source、candidate ID 和原始 provenance；
- 原始 class index 及合法性；
- legacy frozen score 和 FI1-D-v3 geometry/challenger score 为两个独立字段；
- 对应 FI1-D-v3 plan key/index；
- candidate retained/deletion、append-only 和全部 mutation 标志。

禁止将 FI1-D-v3 challenger score 写入 legacy semantic/frozen score 字段，也禁止因几何相同而删除不同 class/score/source 成员。

## 5. S0/S1 独立审计

审计器必须从两个上游账本重建期望身份和字段，至少检查：

1. 全部计数与冻结合同一致；
2. 每个 legacy member 恰好出现一次；
3. 每个 refined union 恰好出现一次；
4. 每个 semantic hypothesis 恰好连接一个 geometry evidence key；
5. 每个 geometry 的 semantic keys 与全局语义账本双向一致；
6. class、legacy score、FI1 score、source、locator、point count、plan key/index 逐字段一致；
7. 原始 semantic score 与 FI1 geometry score 不得字段混合；
8. 不得发生候选删除、几何修改、类别修改、分数修改或重排；
9. `ground_truth_read=false`、`ap_computed=false`；
10. preregistration 和两个输入账本的 SHA-256 必须登记。

## 6. 必须失败的篡改测试

- 删除任意 legacy member 或 refined union；
- 将同几何的不同类别假设折叠；
- 修改 class、legacy score、FI1 score、source、plan key/index 或 locator；
- 将 geometry score 复制到 legacy score 字段；
- 修改 geometry 记录的 semantic key 覆盖；
- 重排语义假设；
- 将 visual evidence execution count 改为 0 或大于 1；
- 任何 GT/AP/mutation/deletion 标志改为 true。

## 7. 当前执行终点

本阶段只允许：

1. 实现 S0/S1 builder、auditor 和合成篡改测试；
2. 在无 GT 输入上生成两层账本并审计；
3. 运行 compileall、`git diff --check` 和相关单元测试。

当前禁止：Alpha/SAM/DINO/Qwen 新推理、任何 GT 读取、任何 AP、最终分数公式选择、根据 val312 类别结果选择候选或阈值。
