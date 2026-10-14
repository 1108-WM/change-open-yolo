# 第二创新点 v2.2：无真实标注输入账本与固定烟雾授权补充合同

状态：`authorized_for_manifest_and_fixed_smoke_only`

日期：2026-08-23

## 1. 授权范围

用户已明确授权：

1. 生成 NCS-train100 的 v2.2 无真实标注输入账本；
2. 对输入账本进行独立审计；
3. 在输入审计为零错误后，运行一次固定的 8 条 Qwen 烟雾；
4. 对烟雾原始输出、严格有效证据和类别决定进行独立审计。

本授权不包括：

- 完整 7,006 条 Qwen 推理；
- 任何真实标注读取；
- AP、AP50 或 AP25；
- NCS-validation60；
- ScanNet200 val312；
- 接入 FI1-D-v3；
- 修改 FI1-Legacy 或覆盖现有 DM-SMS-1、v2、v2.1 产物。

## 2. 固定输入

候选清单固定为：

```text
docs/diagnostics/dm_sms1_candidate_evidence_manifest_train100_20260818/candidate_evidence_manifest.jsonl
```

无类别属性清单固定为：

```text
docs/diagnostics/dm_sms1_attribute_extraction_manifest_train100_20260818/attribute_extraction_manifest.jsonl
```

烟雾使用的冻结无类别属性输出固定为：

```text
docs/diagnostics/dm_sms1_vlm_strict_full7006_train100_20260819/batch_outputs.jsonl
```

模型仍为本地 `Qwen2.5-VL-7B-Instruct`，版本 `cc594898137f460bfe9f0759e9844b3ce807cfb5`，禁止下载或换模型。

## 3. 固定 8 条烟雾选择

选择不使用真实标注、AP 或逐样本准确性。固定算法为：

1. 首先加入 v2.1 第二次烟雾的两个结构回归任务，按任务编号升序；
2. 在冻结旧 DM-SMS-1 输出中，选取 `valid=true` 且旧决定 `class_changed=true` 的任务，排除前述回归任务后，按任务编号升序取前 3 条；
3. 再选取 `valid=true` 且旧决定 `class_changed=false` 的任务，排除已选任务后，按任务编号升序取前 3 条；
4. 最终数量必须精确为 8，选择结果写入运行合同并不得临时替换。

两个回归任务固定为：

```text
0000ee0688bb3845680dd37aec0352920a429d7de3a8e023de596e3a9ffe5749
00066d7be91632bab0fdea1952ae1aca97bb7f933dc1a7a7002570a2cbe60aec
```

## 4. 烟雾通过门槛

只有以下条件全部成立，v2.2 才可被记为“具备后续完整推理申请资格”：

1. 8 条全部处理完成；
2. 原始输出独立审计错误数为 0；
3. 严格有效数至少为 6/8；
4. 两个 v2.1 回归任务不得再出现“顶层候选集不足与候选内部状态不一致”，因为 v2.2 已删除该重复表达；
5. 同一类最终结构错误不得出现超过 2 条；
6. 严格有效输出中至少出现两种不同的候选证据状态，或出现合法的候选集不足变体，避免整批退化为单一固定状态；
7. 几何、掩码、候选成员、排序分数和删除修改数均为 0；
8. `ground_truth_read=false`、`ap_computed=false`。

该门槛只评估输出合同是否可稳定执行，不表示类别准确率或 AP 改善。

## 5. 运行安全

- 启动前必须检查是否有其他 Qwen 或图形处理器任务；
- 如果图形处理器不可用，必须在创建烟雾输出目录前停止，不允许回退到中央处理器；
- 最多使用一次固定结构重写；
- 烟雾失败后不增加重试、不放宽语义、不替换样本；
- 烟雾通过也不自动授权完整 7,006 条推理。
