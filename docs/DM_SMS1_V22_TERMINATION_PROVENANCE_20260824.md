# 第二创新点 v2.2 终止与封存登记

状态：`terminated_after_complete_nogt_audit_before_ap`

日期：2026-08-24

## 1. 最终结论

v2.2 已按冻结合同完成 NCS-train100 全部 7,006 个双候选任务的无真实标注 Qwen 证据生成、安全决定和独立完整审计。独立审计确认任务覆盖、原始输出重放、安全决定、严格有效决定、汇总统计和冻结边界全部一致，错误数为 0。

但是，v2.2 没有产生任何类别替换：7,006 条决定全部保留 FI1-Legacy 冻结原类别。因此 v2.2 不形成区别于冻结控制的新预测，不进入 AP、AP50 或 AP25 评测，并从本登记起正式终止和只读封存。

该结论表示当前规则过于保守，不表示旧版正收益 DM-SMS-1 的结果无效。旧版 DM-SMS-1 与 v2.2 是两个独立版本，产物和结论不得互相改写。

## 2. 完整运行结果

- 预期任务数：7,006；
- 实际处理数：7,006；
- 安全决定数：7,006；
- 严格有效决定数：5,017，占 71.61%；
- 安全回退数：1,989，占 28.39%；
- 类别替换数：0；
- 保留 FI1-Legacy 原类别数：7,006；
- 固定结构重写数：1,560；
- 批次完成标志：真。

安全保留的主要原因：

- 替代类别缺少双顺序直接支持：4,080；
- 原类别缺少双顺序直接强反证：801；
- 候选集合不足或不清楚：135；
- 替代类别存在直接强反证：1；
- 上游或新证据无效而安全回退：1,989。

## 3. 独立审计结果

独立审计输出：

```text
docs/diagnostics/dm_sms1_v22_vlm_full7006_train100_audit_20260824/summary.json
```

审计结果：

- `audit_valid=true`；
- `error_count=0`；
- 完整覆盖 7,006 条原始输出；
- 安全决定账本精确覆盖 7,006 条；
- 严格有效决定账本精确覆盖 5,017 条；
- 所有有效原始 AB/BA 输出均可重新解析并复现决定；
- 所有无效输出均安全保持原类别；
- 几何、掩码、候选成员、排序分数和候选删除修改数均为 0；
- `ground_truth_read=false`；
- `ap_computed=false`。

## 4. 冻结产物与哈希

完整运行目录：

```text
docs/diagnostics/dm_sms1_v22_vlm_full7006_train100_20260823/
```

关键文件 SHA-256：

```text
run_contract.json             2715ffe20936737bc2a82b9a35682260f8fb316a8cfdc7b33aba3bace6a75a8a
selection.jsonl               336d8a7750cd279ee23359d6dbc92cae8152e765782be115f07051b77cd28ec2
batch_outputs.jsonl           9096c5a4d495b882c8b6c97aba1854a84843e67be45a9da9945130267c7fa7d2
safe_decisions.jsonl          87dcff77478a9aeae68047b8dba57e8d941b63c2fb718355cb0ad9a9d69ded29
strict_valid_decisions.jsonl  a75c252105075dadddb65598fc58fa84baebc85bd6581d4c8c5ca4edf9e42c12
summary.json                  31a77e220d12b08390cca43a56515373d3a77867b80af544b5dc7e1e95e245a2
```

独立审计摘要 SHA-256：

```text
948b2967d551617a34c969e61c357da215c5c5c1884f6d92c0977458857dc723
```

冻结合同：

```text
docs/DM_SMS1_V22_SINGLE_AUTHORITY_EVIDENCE_PREREGISTRATION_20260823.md
docs/DM_SMS1_V22_FULL7006_NOGT_AUTHORIZATION_20260823.md
docs/DM_SMS1_V22_FULL7006_START_PROVENANCE_20260823.json
```

## 5. 从此禁止的操作

1. 不运行 v2.2 的 AP、AP50 或 AP25；
2. 不把 v2.2 的零换类结论改写成旧版 DM-SMS-1 的结论；
3. 不覆盖、删除或重新生成本登记中的完整输出和审计；
4. 不根据 v2.2 输出修改提示、视角、候选、状态、解析器或决定门后继续沿用 v2.2 名称；
5. 不读取 NCS-validation60 或 ScanNet200 val312 来挽救或选择 v2.2；
6. 不接入 FI1-D-v3，也不修改 FI1-Legacy。

如果未来需要放宽规则，必须建立新名称、新预注册、新代码和新输出目录，不得重新打开 v2.2。

## 6. 证据边界

v2.2 提供的是“严格决定规则能够安全执行，但在当前完整 NCS-train100 账本上没有实际动作”的负向方法证据。它没有读取真实标注，也没有计算准确率，不能据此判断每个保留决定是否语义正确。由于最终类别与冻结控制完全相同，继续计算 AP 不会提供新的方法比较信息。
