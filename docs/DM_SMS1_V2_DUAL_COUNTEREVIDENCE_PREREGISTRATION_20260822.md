# 第二创新点 v2：双顺序原类别强反证门预注册合同

状态：`frozen_before_formal_ledger_or_evaluation`

日期：2026-08-22

## 1. 研究问题与唯一修改

现有 DM-SMS-1 的双候选决定器只要求替代类别在 AB/BA 两种顺序中都得到支持，且两种顺序均没有强反证。它没有要求冻结原类别在两种顺序中都存在强反证。

本 v2 分支只增加下面一个结构门：

```text
替代类 AB 支持 = true
+ 替代类 BA 支持 = true
+ 替代类 AB 强反证 = false
+ 替代类 BA 强反证 = false
+ 原类别 AB 强反证 = true
+ 原类别 BA 强反证 = true
→ 才允许换类

其他全部情况
→ 保持冻结原类别
```

本分支不修改模型、提示、视角、图像合成、属性字段、候选集合、类别词表、几何、掩码、候选成员或排序分数。`confidence` 只保存用于审计，不进入决定门。原类别“未被支持”不等于原类别存在强反证。

## 2. 第一创新点底座与冻结边界

本 v2 分支的唯一几何底座固定为 FI1-Legacy。这里的“绑定”只允许读取现有 DM-SMS-1 所绑定的 FI1-Legacy 冻结输出：

- geometry：冻结；
- mask：冻结；
- candidate membership：冻结；
- ranking score：冻结。

禁止：

1. 修改、覆盖或重跑 FI1-Legacy；
2. 接入、修改或运行 FI1-D-v3；
3. 修改或覆盖现有 DM-SMS-1 代码、合同、模型输出、安全账本和 AP；
4. 把既有 DM-SMS-1 数值改写成 v2、FI1-D-v3 或联合结果；
5. 删除 proposal、增加第三类别、修改候选成员或修改分数。

v2 必须使用新的代码名、测试名和输出目录。正式输出目录只能使用 `dm_sms1_v2_` 前缀，且创建时必须拒绝覆盖已有非空目录。

## 3. 当前阶段权限

当前只允许：

- 只读审查现有合同和代码；
- 编写本预注册；
- 实现独立 v2 决定器和独立 v2 审计器；
- 运行语法检查、纯函数测试和合成小样本测试。

当前禁止：

- 运行 Qwen、Alpha-CLIP、SAM 或任何 GPU 任务；
- 对完整 NCS-train100 生成 v2 决定账本；
- 读取或重新开放 NCS-validation60；
- 在 ScanNet200 val312 上调参、审查逐样本结果或运行评测；
- 读取 GT 或运行 AP/AP50/AP25；
- 根据既有 train100/validation60 AP 或逐样本结果改变本合同。

完整账本、任何真实数据迁移和正式 AP 必须另获用户明确授权。本预注册本身不授权这些操作。

## 4. 输入合同

决定器只处理双候选任务。每条候选清单必须满足：

1. `task_id` 非空且全局唯一；
2. `scene_name`、`geometry_key`、`geometry_hash` 存在；
3. `candidate_hypotheses` 恰好包含两个不同的 `class_index`；
4. `canonical_frozen_class_index` 必须是两个候选之一；
5. 另一个且仅另一个候选定义为替代类别；
6. 若清单含 `candidate_order_ab`、`candidate_order_ba` 和类别名称，则两种顺序必须分别与候选名称正序和完全反序一致；
7. 候选、几何与分数修改标志不得为真；
8. GT/AP 标志不得为真。

每条证据必须满足：

1. `task_id` 与候选清单精确连接；
2. `scene_name`、`geometry_hash` 若存在，必须与候选清单一致；
3. `valid` 若存在，必须是布尔值；`valid=false` 必须安全回退；
4. `order_ab` 和 `order_ba` 各含两个候选结果，类别集合精确等于冻结候选集合；
5. 两种顺序必须符合候选清单登记的 AB/BA 顺序；
6. 每个候选结果必须含布尔型 `supported`、布尔型 `strong_counterevidence`、字符串型 `support_evidence`、字符串型 `counterevidence` 和有限的 `[0,1]` `confidence`；
7. 结构无效、候选不匹配、身份不匹配或证据字段非法时必须安全保持冻结原类别。

决定器不把空证据文本自动解释为 unknown，也不从自然语言文本推导强反证。`strong_counterevidence` 的语义质量属于上游严格证据合同；本阶段只审计结构与决定逻辑。缺失、遮挡、模糊、视角不足、没有观察到或候选集合不足不得在后续上游合同中标为强反证。

## 5. 决定输出合同

每条输出必须同时保存：

- 冻结原类别和替代类别；
- 最终类别与是否换类；
- 六个固定门的逐项布尔结果；
- 原类别与替代类别在 AB/BA 中的支持、强反证、文本和 confidence 快照；
- `model_evidence_valid` 与安全回退原因；
- 决定器版本和规则名称；
- geometry、candidate、score、proposal deletion 的冻结标志；
- `ground_truth_read=false`、`ap_computed=false`。

换类理由只能是：

```text
alternative_dual_support_no_counterevidence_and_incumbent_dual_counterevidence
```

合法保留理由只能来自预先固定的枚举：

- `invalid_evidence_keep_frozen_control`；
- `alternative_missing_dual_support_keep_frozen_control`；
- `alternative_has_strong_counterevidence_keep_frozen_control`；
- `incumbent_missing_dual_strong_counterevidence_keep_frozen_control`。

当多个保留条件同时失败时，理由优先级固定为：无效证据 > 替代类缺双支持 > 替代类有强反证 > 原类别缺双强反证。

## 6. 审计合同

独立审计器必须读取候选清单、原始证据输入和 v2 决定输出，逐条重新计算六个门与最终类别，不得信任输出中已保存的门或理由。至少检查：

1. 三方 task/scene/geometry 身份精确连接；
2. 完整覆盖、无重复、无多余任务；
3. AB/BA 候选集合与顺序正确；
4. 所有证据字段类型和值域合法；
5. 每次换类均满足全部六个门；
6. 任一门失败时保持冻结原类别；
7. 无效证据必定回退；
8. 决定中的证据快照与输入逐字段一致；
9. 汇总计数与逐条结果一致；
10. geometry、candidate、score、proposal deletion、GT 和 AP 标志全部为假。

审计器只写入新 v2 输出目录中的 `audit_summary.json`，不得修改输入证据、旧决定账本或旧审计结果。

## 7. 当前代码验收门槛

本轮仅以合成测试验收，固定要求：

1. 六个门全部满足时唯一允许换类；
2. 任一支持门失败时保持；
3. 替代类任一强反证为真时保持；
4. 原类别任一强反证为假时保持；
5. `confidence` 改变不影响决定；
6. 无效、重复、错序、错候选或身份不匹配证据安全回退或被审计拒绝；
7. 审计器能发现被篡改的类别、门、理由、证据快照、汇总和冻结标志；
8. 所有 v2 测试通过；
9. 不创建任何真实数据完整账本，不运行 GPU、GT 或 AP。

代码验收通过只说明实现符合本预注册，不构成准确率、迁移性或 AP 改进证据。

