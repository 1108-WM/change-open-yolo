# FI1 几何—语义分离 S2：B0 完整语义控制复现预注册

状态：`frozen_before_S2_implementation_and_before_any_new_GT_or_AP_run`

日期：2026-09-07

分支：`fi1-d-v3-geometry-semantic-separation-20260907`

S0/S1 提交：`8495b950bfaa3be39788c098f882f18a230b2799`

## 1. 本阶段目标和禁止事项

S2 只在无 GT 状态下，将 S0/S1 的多语义假设账本物化为历史完整开放词汇控制 B0，并逐列复现产生既有 val312 `25.415455 / 34.800536 / 40.842221` AP/AP50/AP25 的预测输入。

本阶段不重新读取 val312 GT、不运行 AP、不选择类别、阈值、top-K 或融合权重，不运行 Alpha/SAM、DINO 或 Qwen，也不改变 FI1-D-v3、Z3、DM-SMS-1 或 evaluator。

## 2. 冻结输入及 SHA-256

```text
val312 scene list
d75d4971c3fa7128c643695840e279042c212ef904fe933bd00cf9918c61b083

S0/S1 summary
5a6131afaf7647d1d489f94ebdb16dc171bb36a1f91c1132b07a899175a61e93

S0/S1 semantic hypothesis ledger
d4c5c8ea0652467546def6bfcff840b7c73c6ed719b01402e75fcad8d58c774e

frozen Z3 transfer summary
324caf980d4e76ed1637c545f8e0a979d673cfddb81b5db0e745d1bab53e84b5

frozen Z3 transfer rows
26aa7bba09af287202a89caad3d87ce4c66dc9c40bd3e8d54cf28dddf87fe91d

Z6f stream adapter summary / combined-plan summary
9453ec9299ebc79af050fc687a1dac79ca93399997d3eabbb160ede4e472826b

pair-union append rows
06b420649e68088da83b008bcec08ba51f840a8c47b54bcabf4d43b73bd50a8e

ScanNet200 frozen config
a8fd4b96a832e2d42b2035011e094c6a656986d37126dca7a866da049874c21f

historical val312 AP summary (result provenance only; not rerun in S2)
60d89b0d689d0cc4db871426d06f8e3d13145e59dc9f166e0d5552cd4061b98d
```

权威历史物化逻辑为当前冻结代码中的：

- `tools/evaluate_z3_semantic_reliability_oof_gt.py::_scene_prediction`；
- `tools/evaluate_z6f_safety60_transfer_gt.py` 的 current-control 分支。

S2 只重建它们送入 evaluator 之前的预测数组，不调用 evaluator。

## 3. 冻结计数和边界

S0/S1 语义谱系必须完整保持：

```text
legacy semantic member count       213233
  native                           187200
  retained track                    18141
  pair union                         7892
FI1-D-v3 refined union                106
all semantic hypothesis count      213339
```

冻结 Z3 无 GT 行为：

```text
Z3 row count                       212932
  native                           186905
  track                             18135
  pair union                         7892
Z3 omitted invalid class count        301
  native class 198 sentinel            295
  track class -1                         6
```

历史 B0 evaluator 输入精确为 `213227 = 187200 + 18135 + 7892` 列。这里：

- 295 条 native background sentinel 198 仍按原位置、原掩码和原分数物化；
- 6 条 class `-1` track 按历史 evaluator 行为不物化，但必须继续存在于逐假设 manifest，明确标记为历史 evaluator boundary exclusion；
- 106 条 FI1-D-v3 refined union 属于 B2 append-only，不属于 B0，必须保留 manifest 覆盖但不得进入 B0 cache；
- 不得把上述 evaluator 边界描述成从 S0/S1 谱系删除候选。

精确 6 条历史无效 track 为：

```text
scene0307_00 track 40  scene0307_00:semantic:legacy:track:40:910ddeef0f1663a2d0e699934bf1d0d6cd2b8db3
scene0580_01 track 27  scene0580_01:semantic:legacy:track:27:d05d4b185ede305c71346bfbc49f03fbc806ce8c
scene0643_00 track 85  scene0643_00:semantic:legacy:track:85:bf582a020bbf2d44145c8b5eaab3e4ec3ddb78ed
scene0655_00 track 37  scene0655_00:semantic:legacy:track:37:771d4a0ad61c56ccfa55cc3228ffd3397d5b1192
scene0663_01 track 38  scene0663_01:semantic:legacy:track:38:d44ab7aa3ba86ee90ae3c8f9ec320b96c63c8e11
scene0678_00 track 74  scene0678_00:semantic:legacy:track:74:854dddf753af971a3d67140e9e2f839ac0d2a998
```

## 4. B0 类别与分数合同

每条 manifest 必须同时保留 S1 原始字段和 B0 字段，禁止覆盖原字段。

### native

- 类别、掩码和列顺序直接来自冻结 native cache；
- Z3 有效行使用其冻结 `C_joint_yolo_alpha` 分数；
- 295 条 class 198 sentinel 保持 native 原分数；
- native 列顺序严格为每场 candidate ID `0..599`。

### track

- B0 类别来自冻结 Z3 transfer row；
- B0 分数来自冻结 Z3 `C_joint_yolo_alpha`；
- 几何来自冻结 track point indices；
- 只物化 18,135 条有效行，按 candidate ID 升序；
- 6 条 class -1 只作显式 boundary exclusion，不伪造类别。

### pair-union

- B0 类别来自冻结 Z3 transfer row；
- 分数保持冻结 `original_score`，不得用 FI1 geometry score 替代；
- 几何来自冻结 combined-plan point indices；
- 物化全部 7,892 条，按 candidate ID 升序。

S1 legacy class 与历史 B0 Z3 class 的既有差异精确为：track `1,950` 条、pair-union `1,137` 条。它们必须作为两个独立字段记录，不能改写或丢弃 S1 class。native 不允许类别差异。

## 5. 输出合同

S2 输出：

1. `b0_semantic_control_manifest.jsonl`：覆盖全部 213,339 个 S1 semantic hypothesis key，且每个只出现一次；
2. `prediction_cache/<scene>_pred_masks.npy`；
3. `prediction_cache/<scene>_pred_classes.npy`；
4. `prediction_cache/<scene>_pred_scores.npy`；
5. `prediction_cache_manifest.jsonl`：逐场记录形状、列数和文件 SHA-256；
6. `summary.json`：冻结计数、输入输出哈希和无 GT/无 AP 标志。

逐假设 manifest 必须记录：S1 class/score、B0 class/score、二者来源、是否 B0 in-scope、是否物化、场内列号、边界排除原因以及 geometry/semantic identity。

## 6. 独立审计

审计器不得调用生产 builder 生成期望结果。它必须独立从 S1、Z3、native cache、track points 和 combined-plan 重建并检查：

1. 213,339 个 semantic key 完整且唯一覆盖；
2. B0 in-scope 213,233，B2-only 106；
3. B0 cache 总列数精确为 213,227；
4. native/track/pair-union 列数精确为 187,200/18,135/7,892；
5. 295 个 native sentinel 198 均保留；
6. 6 个无效 track 只被显式排除，身份必须精确匹配本文件；
7. 3,087 个历史 B0 class 与 legacy class 差异逐条登记；
8. 每场 mask、class、score 和列顺序与历史权威逻辑逐值一致；
9. FI1 geometry score 未进入 B0 semantic score；
10. candidate deletion、geometry/class/score mutation 均为 false；
11. `ground_truth_read=false`、`ap_computed=false`。

## 7. 必须失败的合成篡改测试

- 删除或重复任意 semantic key；
- 将 refined union 错误加入 B0；
- 将 6 条无效 track 中任意一条物化或漏记；
- 删除任意 native sentinel 198；
- 修改 mask、class、score、source、candidate ID 或列顺序；
- 将 FI1 geometry score 写入 B0 score；
- 将 B0 class 写回 S1 legacy class；
- 伪造 GT/AP 已执行标志或减少冻结计数。

## 8. 本阶段终点

S2 在 B0 cache 构建和独立无 GT 审计通过后停止。既有 `25.415455` 只作为已冻结历史结果引用；本阶段不得重新运行 val312 AP。后续 S3 只能在 official train100 场景隔离 OOF 上开发。
