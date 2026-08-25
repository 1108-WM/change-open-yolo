# FI1-D-v3 × DM-SMS-1 val312：terminal-safe-keep 结构合同修订

状态：`frozen_before_terminal_safe_keep_implementation_and_before_new_run`
日期：2026-08-25
适用分支：`dm-sms1-fi1-d-v3-val312-duplicate-safe-20260824`
基础提交：`7c620c15c3d23acb48a85647b3f7b2c4588b2463`

## 1. 目的和科学边界

本修订只把既有“无效证据安全保持冻结原类别”原则补充到 pre-Qwen manifest 结构层。它不修改 FI1-D-v3、旧版 DM-SMS-1 双顺序决定规则、Alpha/SAM 投影与深度阈值、Qwen 模型/提示词/参数、ScanNet200 类别配置、预测评测实现或既有结果。

`terminal-safe-keep` 是不可仲裁的结构状态，不是一个新类别、一个 Qwen 失败结果或一个模型预测。

## 2. 精确适用条件

该状态只能逐 `plan_key` 命中以下全部条件：

1. 候选存在于 FI1-D-v3 完整冻结计划，且 `candidate_retained=true`、`candidate_deletion=false`；
2. `canonical_frozen_class_index == 198`，其中198只作为 OpenYOLO3D 已定义的 background sentinel；
3. 联合 Alpha 账本的 `selected_view_count == 0` 且 `views == []`；
4. Alpha 账本的 `alpha_feature_valid == false`；
5. Alpha 账本的 `alpha_class_index == null`；
6. 没有有效 Alpha/SAM 视觉证据；
7. 当前冻结合同的几何、掩码、来源、分数、顺序和 mutation flags 均未改变。

在本次 val312 输入中，命中集合必须精确为4条，冻结 `plan_index` 为：`611, 6649, 21496, 33247`。任何其他命中数量都必须使审计失败并停止。

## 3. 终止保持字段

semantic manifest 对命中记录必须保留冻结候选字段，并额外写入：

```json
{
  "arbitration_eligible": false,
  "terminal_safe_keep": true,
  "terminal_keep_reason": "no_visible_view_and_background_sentinel",
  "selected_views": [],
  "alpha_feature_valid": false,
  "alpha_class_index": null,
  "finite_class_hypotheses": [],
  "attribute_execution_required": false,
  "qwen_execution_required": false,
  "canonical_frozen_class_index": 198,
  "arbitrated_class_index": null,
  "candidate_retained": true,
  "candidate_deletion": false
}
```

非命中记录必须继续使用原有前景有限候选和执行规则，不能被该状态放宽或改变。

## 4. 三个 manifest 的结构

### 4.1 Semantic

每个冻结 `plan_key` 仍必须出现一次并保持原计划顺序。terminal 行不得生成视角、crop、SAM mask、Alpha 类别、有限前景候选或类别决定。

### 4.2 Attribute

terminal 行必须逐 `plan_key` 保留，且写入：

```json
{
  "attribute_execution_required": false,
  "view_inputs": [],
  "attribute_extraction_completed": false,
  "terminal_safe_keep": true,
  "terminal_keep_reason": "no_visible_view_and_background_sentinel"
}
```

不得伪造属性提示输入或模型输出。普通行继续使用冻结 category-blind prompt 和 response schema。

### 4.3 Candidate

terminal 行必须逐 `plan_key` 保留，且写入：

```json
{
  "candidate_hypotheses": [],
  "candidate_order_ab": [],
  "candidate_order_ba": [],
  "swap_order_required": false,
  "attribute_evidence_required": false,
  "qwen_execution_required": false,
  "terminal_safe_keep": true,
  "terminal_keep_reason": "no_visible_view_and_background_sentinel"
}
```

不得为198分配前景 `class_name`、构造Qwen提示词或将其计入 Qwen 解析失败。普通行的候选顺序、提示词和旧版双顺序规则不变。

## 5. Qwen 和决定账本

Qwen任务选择必须精确排除4条 terminal 行。它们不进入 smoke 或完整 Qwen 的任务清单，也不计为失败或无效模型输出。

完整决定账本必须为每个 terminal 行生成确定性记录：

```json
{
  "frozen_class_index": 198,
  "arbitrated_class_index": 198,
  "class_changed": false,
  "decision_source": "terminal_safe_keep",
  "model_evidence_used": false,
  "terminal_keep_reason": "no_visible_view_and_background_sentinel"
}
```

该记录仍保留原 `plan_key`、几何、掩码来源、分数、顺序和 no-GT/no-AP flags。正常候选仍只由旧版 DM-SMS-1 双顺序安全决定改变类别；terminal 行不得产生类别变化。

## 6. 候选、缓存和数量合同

- semantic、attribute、candidate、full decision 必须完整覆盖39304个 `plan_key`，各身份只出现一次；
- `candidate_deletion_count=0`；
- 39250个唯一视觉几何和54个重复几何组保持不变；
- prediction cache 仍物化39304列；terminal 4列不能删除、折叠或重排；
- control 和 challenge 的 terminal 4列掩码、分数、顺序完全一致，类别均为198；
- evaluator继续使用既有 background sentinel 处理，不修改评测代码；
- 所有阶段继续 `ground_truth_read=false`、`ap_computed=false`。

## 7. 独立审计条件

审计必须拒绝以下任一情况：

- terminal 命中数量不是精确4，或命中任何非指定 plan；
- 漏掉任一指定 plan；
- terminal 行进入 attribute/Qwen，或出现属性输出、Qwen输入/输出、前景 `class_name`；
- terminal 行出现视角、Alpha 类别、有限候选或AB/BA提示词；
- terminal 最终类别不是198，或 `class_changed=true`；
- terminal 掩码、几何、分数、来源、顺序、plan_index发生变化；
- 任一阶段漏项、重复项、折叠项或重排项；
- cache列数不是39304或删除数非0；
- 任何类别变化来自 terminal 行；
- GT/AP 标志不是 false。

独立审计必须重算 Alpha 零视角条件，并从冻结计划/联合几何账本复核 terminal 身份，而不是只相信 manifest 中的布尔字段。

## 8. 篡改测试要求

必须增加不访问真实 val312 数据的合成测试，逐一确认以下篡改被拒绝：普通候选伪装为 terminal、漏掉4条之一、把terminal送入 attribute/Qwen、为198填写前景名称、伪造视角/Alpha类别/有限候选、把最终类别改为前景类、删除缓存列、修改掩码/分数/来源/顺序、terminal命中数量非4。

## 9. 冻结输入身份

本修订固定以下已核验输入哈希，不得根据后续数量或AP调整规则：

```text
FI1-D-v3 complete plan 8362f5bea33eeb661a3148ba63d13880ed35f918dbb0dcf2ae9586b845693171
joint unique geometry 303065b4158ee6de62581f8cff2e9f0262d3d6f921e96a6e6190b195d6fe14a8
joint geometry summary e20c30a690f2736aca75151b12957563c0eb00da4e7741d26ac8651f3ab01f00
Alpha view manifest 5c214fefcb7bad7f3814a0d4f2600da9d5b15f1ac9733c0a7fed9020ba02cd35
Alpha embedding summary 827ff7c50db16453d975bb47a508664f52b5049635322a415259fa54464d9be7
ScanNet200 config a8fd4b96a832e2d42b2035011e094c6a656986d37126dca7a866da049874c21f
scene list d75d4971c3fa7128c643695840e279042c212ef904fe933bd00cf9918c61b083
```

## 10. 权限边界和唯一终点

本修订冻结后允许在当前独立分支进行不改变方法的最小结构实现、审计、测试、提交和推送，并使用全新 run root：

`/root/OpenYOLO3D/output/dm_sms1_fi1_d_v3_val312_terminal_safe_keep_20260825`

该目录必须不存在或为空。完整无GT流水线、决定账本和39304列预测缓存全部通过独立审计后停止。正式 AP、GT、评测审计以及任何方法/参数/类别/候选修改均不在本修订权限内。
