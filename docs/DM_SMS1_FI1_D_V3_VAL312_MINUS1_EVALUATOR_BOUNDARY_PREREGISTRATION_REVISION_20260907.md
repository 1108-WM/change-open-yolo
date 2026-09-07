# FI1-D-v3 × DM-SMS-1 val312：`-1` evaluator-boundary 安全表示修订

状态：`frozen_before_implementation_and_before_recovery_ap_invocation`  
日期：2026-09-07  
适用分支：`dm-sms1-fi1-d-v3-val312-minus1-eval-safe-20260907`  
基础提交：`52f73a44c9e666338ef59c9c25757f212408a2a7`

## 1. 修订原因和边界

第一次正式开放词汇 AP 调用已经按原合同启动一次，但在控制组第一个场景进入官方 evaluator 前失败。失败状态必须永久保留为 `failed_no_rerun_allowed`，不得删除、覆盖或改写。

失败不是模型、阈值、提示词、类别决定或GT结果造成的。完整冻结决定账本和预测缓存包含105条 `canonical_frozen_class_index=-1` 且 `arbitrated_class_index=-1` 的候选。联合 AP adapter 原先只接受 `0..198`，因此在第一条 `-1` 上硬停止。

本修订只冻结 `-1` 在官方 ScanNet200 evaluator 边界上的表示。它不修改：

- FI1-D-v3代码、模型、39304条候选、几何、掩码、来源、分数或顺序；
- DM-SMS-1双顺序决定规则、Alpha/SAM、Qwen、提示词、阈值或类别集合；
- semantic/attribute/candidate manifest；
- pair-safe或full-safe决定账本；
- 39304列prediction cache及其哈希；
- ScanNet200 evaluator实现；
- 既有类别无关AP、旧开放词汇结果或第一次失败现场。

## 2. 冻结事实

在完整39304候选中：

- 前景类别 `0..197`：39168条；
- OpenYOLO3D background sentinel `198`：31条；
- 无正YOLO-World类别证据 sentinel `-1`：105条；
- `-1` 涉及73个场景；
- `track` 96条，`pair_union` 9条；
- 105条均 `class_changed=false`；
- 105条均走 `single_candidate_deterministic_keep`；
- 105条均不属于重复几何组；
- 105条均计入39304列，不能删除、折叠、重排或改变。

105条的冻结 `plan_index` 为：

```text
220, 3496, 3533, 4358, 4910, 6486, 6493, 6594, 7755, 7842,
7850, 7863, 8049, 8051, 8085, 8232, 8263, 9164, 9196, 9212,
9237, 9248, 9257, 9313, 9326, 9343, 9374, 10620, 11409, 11472,
11749, 12134, 12466, 12638, 12723, 13207, 13231, 13253, 13563,
13712, 13879, 14042, 14136, 14154, 14290, 15699, 16147, 16720,
16783, 16790, 17216, 17837, 18767, 18965, 18988, 19025, 20227,
20228, 20238, 21601, 22071, 22402, 23146, 23957, 24076, 24301,
24375, 24778, 25101, 25461, 25951, 26953, 27120, 27203, 27291,
27327, 28119, 28144, 28216, 28388, 28807, 28867, 28951, 29361,
29527, 29609, 30248, 31090, 31113, 31412, 31795, 32441, 33131,
33477, 34560, 34881, 34959, 34991, 35189, 35616, 35638, 36867,
36892, 36939, 36999
```

## 3. 唯一允许的 evaluator-boundary 表示

正式control和challenge预测仍从现有缓存读取全部39304列。对每一列先逐 `plan_key` 核对完整决定账本：

- 缓存冻结类别必须等于 `canonical_frozen_class_index`；
- challenge选择类别必须等于 `arbitrated_class_index`；
- geometry hash必须一致；
- 掩码、分数和列顺序不得改变。

只有同时满足以下全部条件的列，允许在交给官方 evaluator 的临时类别数组中把 `-1` 表示为既有 background sentinel `198`：

1. 缓存 `frozen_class_index == -1`；
2. 决定账本 `canonical_frozen_class_index == -1`；
3. 决定账本 `arbitrated_class_index == -1`；
4. `class_changed == false`；
5. `decision_path == single_candidate_deterministic_keep`；
6. `candidate_source` 为冻结的 `track` 或 `pair_union`；
7. `plan_index` 属于本修订冻结的105个索引；
8. control和challenge必须对同一列执行完全相同的表示。

转换只存在于进程内、交给 evaluator 的临时数组。不得写回prediction cache、决定账本或任何无GT产物。

官方 evaluator已经定义 `PRED_ID_TO_ID[198] = -1`，随后把该label作为非前景预测跳过。因此该边界表示不创建第199个前景类别、不映射到任何ScanNet200类名，也不使用Alpha替代类别。

## 4. 严格数量合同

每个control/challenge evaluator调用必须分别得到：

- 输入缓存列数：39304；
- evaluator遍历列数：39304；
- `-1 → 198` 边界表示数：105；
- 原生198列数：31；
- 前景类别列数：39168；
- evaluator background/invalid列总数：136；
- 候选删除数：0；
- 掩码、分数、来源、几何和顺序修改数：0；
- control/challenge边界表示身份集合完全一致。

任何数量不一致都必须在读取或聚合AP结果前硬失败。

## 5. 独立审计

独立AP审计除原有CSV和指标复算外，必须核对：

1. 本修订文档哈希；
2. 完整决定账本与prediction cache哈希仍等于冻结值；
3. 105条身份可从决定账本和缓存独立重算；
4. 每条均满足第3节全部条件；
5. control/challenge各转换105条且身份集合相同；
6. 31条原生198没有被重新分类；
7. 39168条前景类别没有被转换；
8. 39304列全部被遍历；
9. control/challenge掩码、分数和顺序完全相同；
10. 唯一允许的control/challenge差异仍是正常DM-SMS-1决定产生的类别变化；
11. 105条 `-1` 不得产生类别变化；
12. 第一次失败目录和失败哈希仍存在且不变；
13. recovery AP使用新的授权ID和新的输出目录，调用计数为1。

## 6. 必须拒绝的篡改

合成测试和审计必须拒绝：

- `-1`命中数不是105；
- 漏掉任一冻结plan index或加入普通候选；
- 把任一前景类别映射为198；
- 把原生198映射为其他类别；
- `canonical_frozen_class_index`或`arbitrated_class_index`不是-1；
- `class_changed=true`；
- 非singleton安全保持路径命中；
- control与challenge转换身份不一致；
- 修改、过滤或重排缓存列；
- 修改掩码、分数、来源或几何；
- 写回冻结缓存或决定账本；
- 复用、删除或覆盖第一次失败目录。

## 7. Recovery AP唯一调用

本修订允许一次独立的修复后正式评测，授权ID冻结为：

```text
DM-SMS-1-FI1-D-v3-val312-minus1-evaluator-boundary-recovery-20260907
```

新输出目录冻结为：

```text
/root/OpenYOLO3D/output/dm_sms1_fi1_d_v3_val312_terminal_safe_keep_20260825/
20_open_vocab_ap_minus1_safe_20260907
```

独立审计目录冻结为：

```text
/root/OpenYOLO3D/output/dm_sms1_fi1_d_v3_val312_terminal_safe_keep_20260825/
21_open_vocab_ap_minus1_safe_audit_20260907
```

两个目录开始前必须不存在。新AP目录一旦创建，无论成功、失败或中断都不得删除或再次调用。第一次失败的 `18_open_vocab_ap` 永久保留，不把recovery调用描述为原调用未发生。

## 8. 冻结输入哈希

```text
FI1-D-v3 complete plan
8362f5bea33eeb661a3148ba63d13880ed35f918dbb0dcf2ae9586b845693171

val312 Z1 candidate bindings
a26c05800ed4e9a646a99b2b98900400d72cdd9bdc11d73748ebeb5b0f7cbe23

joint unique geometry ledger
303065b4158ee6de62581f8cff2e9f0262d3d6f921e96a6e6190b195d6fe14a8

full safe decision ledger
60b0077fadceeb2873f91e134c3786e99a04c8ba5de9b7acc3d219a67302858f

prediction cache summary
8fa827f04fa151793c824545c644d29bc6f1ac92effb0f863da75bf5bd421a8f

prediction cache audit
2997be6b02655815af555cee1c562dd369e62a805a2019b5904ca4e904d86c29

first AP started marker
b808c187e90e6f491964bca595b7e66eaec0883181b270c1cd66c4e423102e41

first AP failed marker
8a7cfcd6e99bb23eea3a5e3e9c4ca2ed9c26c7383a4239bf4939803581faf2ea

first AP log
471143eec688b15324a95847e00004e22a8cfdd5bfc32d4135b99fcf719fa788

AP blocker report
9efc0d7399557a996eef7191f2500b3a56ceada2981304e131533c95be91c3e3
```

## 9. 终点与禁止事项

recovery AP无论升高、降低、近零或失败均只运行一次并原样封存。不得根据结果修改任何参数、类别、模型、提示词、决定规则或候选。完成后允许运行一次对应独立审计，然后停止报告。
