# 旧版正收益 DM-SMS-1：val312 准备修复预注册

状态：`frozen_before_prep_repairs_and_before_any_val312_access`

日期：2026-08-24

分支：`dm-sms1-legacy-val312-prep-20260824`

起点提交：`a1d207ebeb96b96ce57164c2aa6df1bd74d6ce9d`

## 1. 目的

本预注册只修复 2026-08-24 就绪性审计确认的部署和结构安全问题，使旧版正收益 DM-SMS-1 能够进入独立 val312 最终评测预注册的前置状态。

本阶段不是方法改版，不重新选择旧版 DM-SMS-1 与 v2.2，不使用 val312 反馈，不重新运行 train100 或 validation60 AP。

审计依据：

```text
docs/DM_SMS1_LEGACY_VAL312_READINESS_AUDIT_20260824.md
```

## 2. 不可修改的方法合同

以下内容保持旧版正收益 DM-SMS-1 原样：

1. 几何底座只能是 FI1-Legacy；
2. 冻结几何、掩码、候选成员和排序分数；
3. 有限候选只能是冻结原类别与 Alpha-CLIP 替代类别；
4. 无类别属性阶段继续隐藏候选名称；
5. 候选比较继续使用 AB 和完全反序 BA；
6. 只有替代类别在两种顺序中均获支持，且两种顺序均无替代类别强反证时才换类；
7. 不增加原类别双强反证门，不接入 v2、v2.1 或 v2.2；
8. 任何无效证据继续安全保持冻结原类别；
9. 单候选节点继续确定性保持冻结原类别；
10. Qwen 模型、版本、提示词、视角数、图像合成方式和生成参数均不改变。

禁止修改、运行或接入 FI1-D-v3。禁止覆盖 FI1-Legacy 和旧版 DM-SMS-1 已有 train100、validation60 原始输出、账本、审计或 AP。

## 3. 唯一允许的代码修复

### 3.1 恢复通用几何解析辅助模块

在当前分支独立补齐：

```text
tools/audit_dm_sms1_unique_geometry_ledger.py
```

恢复内容必须是旧 DM-SMS-1 通用几何定位和无真实标注审计代码，不得包含 FI1-D-v3 算法、模型、分数或候选细化逻辑。来源必须登记文件哈希。

### 3.2 收紧证据文字结构验证

只增加两条确定性结构要求：

- `supported=true` 时，`support_evidence` 必须是非空文字；
- `strong_counterevidence=true` 时，`counterevidence` 必须是非空文字。

布尔决定规则、证据把握度和其他字段均不改变。未满足要求的旧或新记录必须作为无效证据安全保持原类别。

### 3.3 使用场景级完整身份连接

候选清单的属性输入与上游语义记录必须按：

```text
(scene_name, geometry_hash)
```

连接。生成器必须拒绝空身份、重复身份、缺失身份和多余身份，不再用单独的几何哈希静默覆盖跨场景记录。

### 3.4 修复测试和范围文字

- 更新旧测试样例，使其提供正式合同要求的候选集合和有效 SAM 掩码标志；
- 增加空支持文字、空强反证文字和跨场景同哈希连接测试；
- 将完整安全账本的固定 train100 范围文字改为数据集无关文字；
- 不覆盖任何旧摘要。

## 4. 允许修改的文件范围

只允许修改或新增与上述修复直接相关的文件：

```text
tools/audit_dm_sms1_unique_geometry_ledger.py
tools/run_dm_sms1_single_scene_vlm_smoke.py
tools/apply_dm_sms1_semantic_arbitration.py
tools/build_dm_sms1_candidate_evidence_manifest.py
tools/build_dm_sms1_full_safe_decision_ledger.py
tests/test_audit_dm_sms1_unique_geometry_ledger.py
tests/test_run_dm_sms1_single_scene_vlm_smoke.py
tests/test_apply_dm_sms1_semantic_arbitration.py
tests/test_build_dm_sms1_candidate_evidence_manifest.py
tests/test_build_dm_sms1_safe_decision_ledger.py
tests/test_build_dm_sms1_semantic_arbitration_manifest.py
```

如果实际只需其中一部分文件，不得为凑齐清单修改其余文件。

## 5. 无真实标注重放门

代码修复后，只允许读取已经冻结的 train100 和 validation60 原始 Qwen 输出，生成新的独立重放目录。不得调用 Qwen，不得读取真实标注，不得运行 AP。

预注册的确定性预期：

### train100

- 原 7,006 个双候选任务完整覆盖；
- 两个任务因“支持为真但支持文字为空”转为安全回退；
- 双候选换类数由 1,827 变为 1,825；
- 其余任务决定不得改变；
- 旧冻结账本和 AP 不覆盖、不更新。

固定受影响任务：

```text
e2a8ca09fdd026314f9c150c76ca5ff5c56a34460c38474dd0017da01a6a2f77
1333675edfa87f1bf9c5230abc5a7365a99405af369e7ab489327b56e20bf51f
```

### validation60

- 原 4,046 个双候选任务完整覆盖；
- 一个原本未换类的任务因空支持文字转为安全回退；
- 最终 5,887 个几何的类别编号必须与旧冻结完整账本逐条完全一致；
- 换类数必须保持 993；
- 不重新运行 validation60 AP。

固定受影响任务：

```text
8864e8a5b7de7a0a34ae73df40c02c36a95365f906da2dd5f1a1ee5b60631a80
```

任一预期不成立即停止，不放宽验证器，不根据结果修改规则。

## 6. 代码验收门

必须同时满足：

1. 所有涉及文件通过语法检查；
2. 旧版 DM-SMS-1 相关纯函数和评测导入测试全部通过；
3. 新增测试能检出空支持文字、空强反证文字和跨场景身份误连接；
4. train100 与 validation60 的新批次重放审计错误数为 0；
5. 两套新的双候选和完整账本审计错误数为 0；
6. validation60 最终类别逐条等同旧冻结账本；
7. 几何、掩码、候选成员、排序分数和候选删除修改数均为 0；
8. `ground_truth_read=false`、`ap_computed=false`；
9. 没有访问或生成 val312 产物。

代码验收通过只授权编写独立 val312 最终评测预注册，不授权启动 val312 清单、Qwen、真实标注或 AP。

## 7. 当前禁止事项

本阶段禁止：

- 读取或扫描 ScanNet200 val312；
- 运行 Qwen、SAM、Alpha-CLIP 或任何图形处理器任务；
- 读取新的真实标注；
- 运行 AP、AP50 或 AP25；
- 重跑 train100 或 validation60 模型推理；
- 修改旧版方法决定规则；
- 根据两个受影响任务的类别或历史 AP 决定是否保留修复；
- 覆盖任何冻结产物；
- 修改 FI1-Legacy 或 FI1-D-v3。
