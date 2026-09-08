# FI1 几何—语义分离：本机 official100 接续说明

状态：`handoff_after_remote_S0_S2_before_local_S3`

日期：2026-09-07

## 1. 必须检出的代码

仓库：`git@github.com:1108-WM/change-open-yolo.git`

分支：`fi1-d-v3-geometry-semantic-separation-20260907`

本说明写入前的远程实现提交：

```text
5d423b346336a5439feb672a0225580738097b96
```

本说明及状态文件会形成其后的新提交；本机应以该分支最新远程 HEAD 为准，并记录完整提交号。

推荐检出命令：

```bash
git clone git@github.com:1108-WM/change-open-yolo.git OpenYOLO3D-geometry-semantic
cd OpenYOLO3D-geometry-semantic
git fetch origin fi1-d-v3-geometry-semantic-separation-20260907
git switch --track origin/fi1-d-v3-geometry-semantic-separation-20260907
git status --short --branch
git rev-parse HEAD
```

若本机已有仓库且有未提交修改，应使用新的 clone 或独立 worktree，不得强制切换覆盖修改。

## 2. GitHub 已包含与不包含的内容

已包含：

- S0/S1 几何证据层与多语义假设层 builder、独立 auditor、合成测试；
- S2 val312 B0 控制缓存 builder、独立 auditor、合成测试；
- 两份前置预注册；
- 当前失败诊断、验证风险判断和本机接续合同。

不包含：

- ScanNet200 数据、GT、RGB-D、pose、intrinsics；
- native prediction cache、track point files、pair-union files；
- YOLO-World、Alpha-CLIP、DINO、Qwen 大模型权重；
- 远程生成的约 33GB S2 B0 prediction cache；
- historical official100 大型中间账本；
- 尚未实现的 S3 B1/B2 新训练代码。

## 3. 远程已经完成的冻结事实

### S0/S1

```text
scene_count                         312
legacy_geometry_count              39198
legacy_member_count               213233
fi1_d_v3_candidate_count           39304
fi1_d_v3_unique_geometry_count     39250
fi1_d_v3_refined_union_count         106
semantic_hypothesis_count         213339
audit_valid                          true
error_count                              0
```

提交：`8495b950bfaa3be39788c098f882f18a230b2799`。

### S2 B0

```text
semantic_hypothesis_count          213339
B0 in-scope legacy members         213233
B0 materialized evaluator columns  213227
  native                           187200
  valid track                       18135
  pair-union                         7892
native background sentinel 198        295
historical invalid track -1              6
B2-only refined union                  106
audit_valid                           true
error_count                               0
```

新缓存与产生历史 val312 `25.415455/34.800536/40.842221` AP/AP50/AP25 的冻结 `_scene_prediction` 路径在 312 场的 mask、class、score 和顺序逐值一致。S2 没有重新读取 GT 或运行 AP。

提交：`5d423b346336a5439feb672a0225580738097b96`。

## 4. official100 历史结果的正确解释

第一创新点的类别无关 official100 提升为 `+2.274604 AP`，并在 val312 保留 `+1.789276 AP`，因此几何方向本身并未失效。

第二创新点历史 Z3 hybrid 在 official100 相对固定 Alpha 控制提升 `+2.444070 AP`、相对 native-only 提升 `+4.823585 AP`，但五折增量为：

```text
+3.1695 / +1.0575 / +4.1597 / +3.3377 / -3.2307
```

总体均值掩盖了一个严重负迁移折。Z6f/Qwen 的 official100 增量只有 `+0.053051 AP`，safety60 为 `-0.005250 AP`，不构成稳定增益。

旧 FI1-D-v3 + DM-SMS-1 在 val312 的 `+3.002111 AP` 是相对已因语义折叠而受损的 `19.458571` 控制；最终 `22.460682` 仍低于历史完整控制 `25.415455`。它只恢复部分损失，不能被解释为超过强基线。

## 5. 为什么不能再次只用相同 official100 总体 AP 选型

1. 历史 official100 已用于大量融合、标签、模型、门控和 Qwen 路由选择，存在研究者层面的验证集过拟合；
2. ScanNet200 宏 AP 对稀有类别和少量 TP 高方差，单个负折可被全量平均隐藏；
3. official100 与 val312 的来源比例、场景、类别和候选竞争分布不同；
4. 历史方法使用每几何 canonical 类别，未测试新版本必须保留的完整多语义假设空间；
5. 类别无关几何质量与类别相关语义可靠度曾被错误混用。

## 6. 本机下一阶段正确顺序

本机当前不能直接运行“新的 S3”，因为代码尚未实现。应先完成：

### S3a：资产与无 GT 迁移风险审计

- 定位 frozen official100 scene list、五折 manifest、native cache、track points、pair-union、Z1/Z2c/Z3 输入；
- 核验场景、候选和哈希；
- 比较 official100 与 val312 的无 GT 特征分布、来源比例、分数范围、可见视角、entropy/margin 和 geometry quality；
- 不读取 val312 GT。

### S3b：新增 train-split lockbox

优先从未参与 historical official100 开发的 ScanNet200 train scenes 中，在读取其 GT 前以固定 seed 冻结新的 lockbox。它只能在方法和阈值冻结后评测一次，不能用于回调。

若无法取得新的 train scenes，则必须明确声明：official100 只能作为开发证据，不能作为独立迁移证明。此时采用固定方法族的 nested scene CV、paired bootstrap、最差折和分组稳定性，但证据强度仍低于新 lockbox。

### S3c：B0/B1/B2

- B0：首先逐列复现 official100 权威控制；
- B1：保持候选不变，只校准独立的 semantic reliability 与 geometry quality；
- B2：在 B1 上 append-only 加入可追溯 track/pair-union/refined-union 假设；
- FI1 geometry score 不得覆盖原 semantic score；
- 类别 ID、scene ID、val312 统计不得作为模型输入。

### 放行条件

不能只看全量 AP。至少同时报告：

- 五折逐折 paired delta 与最差折；
- paired scene bootstrap 置信区间；
- head/common/tail；
- native/track/pair-union/refined-union；
- Brier/ECE/排序稳定性；
- train-lockbox 唯一次结果（若可用）。

旧 Z3 出现 `-3.2307 AP` 的负折，按新的稳定性合同不会直接放行。

## 7. 当前代码使用限制

`tools/build_fi1_s2_b0_control.py` 与 `tools/audit_fi1_s2_b0_control.py` 当前冻结的是 val312 S2 合同，默认包含 val312 计数以及 6 条特定 invalid-track 身份。即使部分计数参数可覆盖，也不得直接把这两个程序当作 official100 S3 入口。

本机应先实现独立的 official100 S0/S1/B0 适配和预注册测试，或将公共逻辑抽取为数据集无关核心并为 val312/official100 分别保留冻结 profile。不得删除或放宽现有 val312 硬审计。

## 8. 禁止事项

- 不读取 val312 GT、不重新运行 val312 AP；
- 不根据历史 val312 逐类结果选择特征、类别、阈值、top-K 或权重；
- 不重复调旧 Z3/Z4/Z6f/Qwen；
- 不把 safety60/even48/test60 用于新方法选择；
- 不覆盖本机已有代码或数据；
- 不在资产和控制未通过审计前开始 B1/B2 训练。

## 9. val312 第一次 AP 失败的 evaluator 边界修复

当时 `class_index=-1` 导致官方 evaluator 入口停止的修复代码已在本分支的祖先提交 `e4642e1e73b55ad0523a0d776cdbf22f78ae7be8` 中。本机排查、精确适用边界和不得直接复用远程失败哈希的说明见：

```text
docs/VAL312_AP_MINUS1_EVALUATOR_BOUNDARY_LOCAL_HANDOFF_20260908.md
```
