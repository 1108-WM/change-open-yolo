# 第二创新点 v2.2：NCS-train100 完整 7,006 条无真实标注证据账本授权合同

状态：`authorized_for_full7006_nogt_vlm_ledger_only`

日期：2026-08-23

## 1. 用户授权

用户已在 v2.2 固定 8 条 Qwen 烟雾和独立审计通过后，明确授权启动 NCS-train100 完整 7,006 条无真实标注 Qwen 证据账本。运行必须放入新的独立 `tmux` 会话，支持断点续跑和日志监控。

本授权只包括：

1. 对 7,006 个冻结双候选任务生成 v2.2 AB/BA 证据；
2. 严格结构验证、最多一次固定重写和无效证据安全回退；
3. 生成完整类别安全决定账本；
4. 运行结束后的独立无真实标注审计和分布汇总。

本授权不包括：

- 读取真实标注；
- AP、AP50 或 AP25；
- 根据运行中间输出修改提示、状态、视角、候选或决定门；
- NCS-validation60 或 ScanNet200 val312；
- 接入 FI1-D-v3；
- 修改 FI1-Legacy；
- 覆盖现有 DM-SMS-1、v2、v2.1 或 v2.2 烟雾产物。

## 2. 冻结前置门

启动前必须同时满足：

1. v2.2 输入账本为 7,006 条且审计 `audit_valid=true`、`error_count=0`；
2. 固定 8 条烟雾为 7/8 严格有效，输出独立审计 `audit_valid=true`、`advancement_eligible=true`；
3. 模型为本地 `Qwen2.5-VL-7B-Instruct`，版本为 `cc594898137f460bfe9f0759e9844b3ce807cfb5`；
4. 没有其他 Qwen、训练或图形处理器计算进程冲突；
5. 完整输出目录尚不存在；
6. 完整运行器、独立审计器、`tmux` 包装脚本和纯函数测试已通过。

任一前置门失败即不创建完整输出目录，不回退到中央处理器。

## 3. 固定输入与运行参数

输入账本：

```text
docs/diagnostics/dm_sms1_v22_evidence_manifest_train100_20260823/v22_evidence_manifest.jsonl
```

输入账本审计：

```text
docs/diagnostics/dm_sms1_v22_evidence_manifest_train100_20260823/audit_summary.json
```

冻结无类别属性输出：

```text
docs/diagnostics/dm_sms1_vlm_strict_full7006_train100_20260819/batch_outputs.jsonl
```

烟雾审计：

```text
docs/diagnostics/dm_sms1_v22_vlm_smoke8_train100_audit_20260823/summary.json
```

模型参数固定为：

```text
do_sample = false
min_pixels = 100352
max_pixels = 200704
max_new_tokens = 700
device = cuda
CPU fallback = forbidden
```

任务顺序固定为输入账本中的任务编号升序。不根据旧决定、类别、场景或当前输出调整运行顺序。

## 4. 无效证据与完整决定账本

冻结无类别属性输出中已有 894 条旧严格无效记录。这些任务不伪造新属性，不调用 Qwen 类别比较，直接写入“无效上游证据、保持冻结原类别”的安全决定。

对于新 Qwen 输出，首次生成和唯一重写均无效时，也必须写入安全决定。因此最终 `safe_decisions.jsonl` 必须精确覆盖 7,006 个任务，而不只保存严格有效记录。

## 5. 断点恢复与禁止覆盖

首次启动时新建：

```text
docs/diagnostics/dm_sms1_v22_vlm_full7006_train100_20260823/
```

运行合同保存输入、审计、模型、参数、任务覆盖和摘要。续跑时必须：

1. 运行合同完全一致；
2. 已完成记录是固定任务顺序的精确前缀；
3. 原始输出可被严格重放；
4. 安全决定账本与已完成记录逐条一致。

任何摘要、任务顺序、输入或旧记录不一致时拒绝续跑，不允许删除旧记录后重新覆盖。

## 6. 完成后审计门

完整运行结束后，独立审计器必须确认：

1. 输入、输出和安全决定均精确覆盖 7,006 个任务；
2. 无重复、无缺失、无多余任务；
3. 所有严格有效记录都能从原始 AB/BA 文本重新解析并复现决定；
4. 所有无效记录都安全保持原类别；
5. 汇总数与逐条记录一致；
6. 几何、掩码、候选成员、排序分数和删除修改数均为 0；
7. `ground_truth_read=false`、`ap_computed=false`。

审计通过只说明完整 v2.2 无真实标注证据和决定账本结构可信。未经用户后续明确授权，不得自动运行 AP 或任何真实标注分析。
