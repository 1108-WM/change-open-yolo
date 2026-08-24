# FI1-D-v3 × 旧版 DM-SMS-1：ScanNet200 val312 远程运行说明

状态：代码包执行说明；不包含任何 val312 新结果。

重复几何安全适配时，本说明必须与
`DM_SMS1_FI1_D_V3_VAL312_DUPLICATE_SAFE_PREREGISTRATION_REVISION_20260824.md`
共同使用；该修订只替换旧包中“候选几何必须逐条唯一”的错误接口假设。

## 1. 方法和边界

本代码包以远程已经冻结并审计通过的 FI1-D-v3 完整挑战计划为唯一几何、候选和排序底座。控制组使用 FI1-D-v3 的冻结类别，挑战组只应用旧版正收益 DM-SMS-1 的完整安全类别决定。两组的点掩码、候选数、候选来源和分数完全相同。

旧版决定规则保持不变：替代类别必须在候选正序和反序中都得到支持，并且两种顺序中都没有替代类别强反证，才允许替换原类别。无效证据、单候选或门条件不满足时均保持 FI1-D-v3 冻结原类别。本包没有接入 v2、v2.1、v2.2，也没有加入“原类别双强反证”条件。

禁止修改或重跑 FI1-D-v3，禁止根据 val312 的图像、换类数或 AP 修改阈值、提示词、模型、视角数或决定规则。

## 2. 首次准备

从模板复制服务器路径配置，并只替换路径：

```bash
cp docs/DM_SMS1_FI1_D_V3_VAL312_PATHS.example.json \
  dm_sms1_fi1_d_v3_val312_paths.json
```

`run_root` 必须是不存在或完全空的新目录。FI1-D-v3 推理目录、推理审计目录、类别无关 AP 目录和 AP 审计目录必须指向已经产生以下结果的同一冻结运行：

```text
类别无关 AP = 0.526344
AP50 = 0.729213
AP25 = 0.826733
```

先运行不访问数据集的代码测试：

```bash
python -m pytest -q \
  tests/test_fi1_d_v3_frozen_deployment.py \
  tests/test_dm_sms1_fi1_d_v3_joint_package.py \
  tests/test_dm_sms1_fi1_d_v3_duplicate_safe.py
```

## 3. 分阶段执行

每个阶段成功且其审计为零错误后，才能执行下一阶段。所有命令都使用同一个路径配置文件。

### 3.1 只读预检

```bash
python tools/run_dm_sms1_fi1_d_v3_val312_pipeline.py \
  --paths dm_sms1_fi1_d_v3_val312_paths.json --stage preflight
```

该阶段核对312个场景、FI1-D-v3 推理与 AP 身份、三个冻结指标、模型 revision 和本地资产；不运行 GPU、不读取真实标注、不计算 AP。

### 3.2 联合唯一几何账本

```bash
python tools/run_dm_sms1_fi1_d_v3_val312_pipeline.py \
  --paths dm_sms1_fi1_d_v3_val312_paths.json --stage geometry
```

要求 `02_unique_geometry_audit/summary.json` 中 `audit_valid=true` 且 `error_count=0`，并严格核对
39,304 个候选、39,250 个唯一几何、54 个重复组、50 个重复场景、33 个类别不同组和53个分数不同组。

### 3.3 Alpha-CLIP 与 SAM 视觉账本

```bash
python tools/run_dm_sms1_fi1_d_v3_val312_pipeline.py \
  --paths dm_sms1_fi1_d_v3_val312_paths.json --stage alpha
```

该阶段使用 GPU。Alpha 账本支持断点续跑，但不得更换输入、权重或参数。要求视角清单和嵌入账本两个独立审计均为零错误。

### 3.4 无类别属性和有限候选清单

```bash
python tools/run_dm_sms1_fi1_d_v3_val312_pipeline.py \
  --paths dm_sms1_fi1_d_v3_val312_paths.json --stage manifests
```

该阶段不调用 Qwen，不读取真实标注。三个清单各自在原目录生成 `audit_summary.json`，均须满足 `audit_valid=true`。

### 3.5 固定小规模 Qwen 烟雾测试

```bash
python tools/run_dm_sms1_fi1_d_v3_val312_pipeline.py \
  --paths dm_sms1_fi1_d_v3_val312_paths.json --stage smoke
```

烟雾测试只验证接口和结构，不作为准确率证据。要求 `11_qwen_smoke_audit/summary.json` 为零错误。

### 3.6 完整双候选 Qwen 账本

```bash
python tools/run_dm_sms1_fi1_d_v3_val312_pipeline.py \
  --paths dm_sms1_fi1_d_v3_val312_paths.json --stage qwen
```

该阶段选择全部312场景中的所有双候选任务，并使用 append-only 前缀合同断点续跑。重复执行同一命令只允许继续尚未完成的冻结任务，不允许改变选择范围。完成后要求 `13_qwen_full_audit/summary.json` 为零错误。

### 3.7 完整安全决定账本

```bash
python tools/run_dm_sms1_fi1_d_v3_val312_pipeline.py \
  --paths dm_sms1_fi1_d_v3_val312_paths.json --stage decisions
```

该阶段严格重放原始 Qwen 输出；结构无效时安全保持原类别，并合并所有单候选确定性保持项。`14_pair_safe_decisions/audit_summary.json` 和 `15_full_safe_decisions/audit_summary.json` 均须为零错误。

### 3.8 冻结预测缓存

```bash
python tools/run_dm_sms1_fi1_d_v3_val312_pipeline.py \
  --paths dm_sms1_fi1_d_v3_val312_paths.json --stage cache
```

缓存按原计划顺序物化39,304列 FI1-D-v3 冻结掩码、类别和分数；重复几何保留重复掩码列，
不得折叠为39,250列，也不读取真实标注。要求 `17_prediction_cache_audit/summary.json` 为零错误。

## 4. 唯一一次正式 AP

只有上述全部阶段和审计通过后，才能由用户明确授权执行：

```bash
python tools/run_dm_sms1_fi1_d_v3_val312_pipeline.py \
  --paths dm_sms1_fi1_d_v3_val312_paths.json --stage ap --authorize-ap
```

AP 程序创建 `18_open_vocab_ap/ap_invocation_started.json` 后即锁定该输出目录。成功、失败或中断后都不得删除目录并重跑。一次程序调用内部固定进行两次官方评测：一次 FI1-D-v3 冻结语义控制，一次 FI1-D-v3 加 DM-SMS-1 挑战；这两次构成同一次预注册比较。

完成后立即执行独立审计：

```bash
python tools/run_dm_sms1_fi1_d_v3_val312_pipeline.py \
  --paths dm_sms1_fi1_d_v3_val312_paths.json --stage audit
```

独立审计从两份官方 CSV 重新计算 AP、AP50、AP25、head/common/tail AP，并核对输入哈希、换类数、启动/完成标记和唯一调用状态。最终只在 `19_open_vocab_ap_audit/summary.json` 满足 `audit_valid=true`、`error_count=0` 后报告结果。

## 5. 失败处理

- AP 前阶段失败：保留原目录和日志，先报告；只有明确属于支持断点续跑的 Alpha 或完整 Qwen 阶段才能按原命令继续。
- 任一审计非零：立即停止，不进入后续阶段。
- AP 阶段失败或中断：保留 `18_open_vocab_ap` 全部内容，禁止重跑并报告失败标记。
- 任一路径、哈希、场景数、模型 revision 或冻结指标不匹配：停止，不通过改代码绕过。
