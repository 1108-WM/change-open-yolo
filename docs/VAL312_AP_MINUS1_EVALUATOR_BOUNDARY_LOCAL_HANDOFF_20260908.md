# val312 AP `-1` evaluator 边界问题：本机复现与排查说明

日期：2026-09-08

状态：`code_pushed_exact_remote_contract_local_diagnosis_required`

## 1. 远程服务器当时的失败是什么

第一次 FI1-D-v3 + duplicate-safe DM-SMS-1 开放词汇 AP 调用在 control 的第一个场景、进入官方 ScanNet200 evaluator 前停止。直接原因不是 CUDA、GT、模型或 AP 聚合，而是预测缓存含有 evaluator 入口不接受的 `class_index=-1`。

当时冻结的 `39,304` 列候选中：

```text
foreground 0..197                         39168
OpenYOLO3D background sentinel 198           31
no-positive-YOLO-World sentinel -1           105
total                                      39304
```

这 `105` 条全部是安全保持候选：`canonical=-1`、`arbitrated=-1`、`class_changed=false`、`decision_path=single_candidate_deterministic_keep`，来源仅为 `track` 或 `pair_union`。

## 2. 实际修复策略

修复只在官方 evaluator 的进程内入参边界进行：

1. 从冻结 decision ledger 和 prediction cache 逐 `plan_key` 重建并审计精确的 `105` 条身份；
2. 只对同时满足全部冻结条件的 `-1` 列，在传给 evaluator 的临时 `pred_classes` 数组中表示为已存在的 background sentinel `198`；
3. control 和 challenge 必须对同一组列执行相同转换；
4. 不写回 cache/ledger，不删除、折叠或重排列，不修改 mask/score/source/geometry；
5. 官方 evaluator 的 `PRED_ID_TO_ID[198] = -1` 会将它们作为非前景预测跳过，不会创建第199个前景类。

这只解决 evaluator 边界崩溃。它没有改善模型，也不是后来 `22.460682 AP` 低于强控制 `25.415455 AP` 的原因。后者已确认为语义成员折叠和将几何质量分数用于类别相关排序的方法问题。

## 3. GitHub 中的完整实现

修复预注册提交：

```text
b0dfdf62d146dbfb4013494aa6d2551c3a18b3ea
```

修复实现提交：

```text
e4642e1e73b55ad0523a0d776cdbf22f78ae7be8
```

当前分支 `fi1-d-v3-geometry-semantic-separation-20260907` 包含上述两个提交作为祖先。关键文件：

```text
docs/DM_SMS1_FI1_D_V3_VAL312_MINUS1_EVALUATOR_BOUNDARY_PREREGISTRATION_REVISION_20260907.md
tools/dm_sms1_minus1_evaluator_boundary.py
tools/evaluate_dm_sms1_fi1_d_v3_open_vocab_ap_gt.py
tools/audit_dm_sms1_fi1_d_v3_open_vocab_ap.py
tools/run_dm_sms1_fi1_d_v3_val312_pipeline.py
tests/test_dm_sms1_minus1_evaluator_boundary.py
```

## 4. 本机首先确认是否为同一问题

不要因为“AP 失败”就直接套用此修复。先在不重跑 AP 的前提下保留并查看失败现场：

```bash
find /path/to/run_root -maxdepth 3 -type f \
  \( -name 'ap_invocation_failed.json' -o -name '*ap*.log' \) -print

sed -n '1,240p' \
  /path/to/failed_ap_root/ap_invocation_failed.json
```

只有当 traceback 指向预测类别 `-1` 超出 evaluator contract，并且使用的是同一套 duplicate-safe `39,304` 列冻结输入，才是本文件描述的同一问题。

可先检查代码和合成回归测试：

```bash
git rev-parse HEAD
git merge-base --is-ancestor \
  e4642e1e73b55ad0523a0d776cdbf22f78ae7be8 HEAD

python -m pytest -q \
  tests/test_dm_sms1_minus1_evaluator_boundary.py
```

`git merge-base` 应退出 `0`，测试必须全部通过。GPU 类型与该问题无关；A100 和 RTX 4090 使用相同的 evaluator 边界合同。

## 5. 不得直接复制远程 recovery 命令

仓库中的 `recovery-ap` 入口为远程服务器那一次已失败调用冻结了：

- 精确 `105` 个 `plan_index/plan_key`；
- decision ledger 和 prediction cache 哈希；
- 第一次失败 marker 与 log 哈希；
- 远程专用授权 ID 和新输出目录。

因此，本机的失败 marker/log 哈希不同时，直接运行服务器的 `recovery-ap` 命令应该被审计器拒绝。不得为了通过而删除哈希检查或改写远程预注册。

本机正确做法是：

1. 保留本机第一次失败目录和日志；
2. 只读审计本机决定账本、缓存、场景覆盖和类别计数；
3. 如果与远程冻结输入在身份、数量和内容哈希上完全一致，只新建本机运行专用的前置合同和全新输出目录；
4. 复用现有边界转换核心和审计条件，只替换“本机失败现场的路径与哈希”；
5. 先运行不读 GT 的边界 preflight，确认 `39304/39168/31/105`、删除数 `0`、冻结字段修改数 `0`；
6. 只在本机获得新的明确 AP 授权后，执行一次全新的 recovery AP 和独立审计。

如果上述计数或 `plan_key` 身份集不一致，必须停止。这不是可以通过改常量、忽略候选或把所有负类别无条件改为 `198` 来处理的路径问题。

## 6. 与新几何—语义分离主线的关系

该 `105` 条修复属于旧 duplicate-safe `39,304` 列实验的 evaluator 边界恢复。它不是当前 S2 B0 `213,227` 列语义控制的通用入口。

S2 B0 中历史 invalid track class `-1` 为 `6` 条，身份和评估列数都不同。将 `105` 条合同直接套到 S2/S3 必须被拒绝。S3 在 official100 上尚未实现，应先复现该数据集的 B0 和冻结独立评估合同。

