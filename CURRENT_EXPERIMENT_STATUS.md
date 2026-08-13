# OpenYOLO3D 当前实验状态

更新日期：2026-08-12

## 文件职责

- 本文件：实验结论、关键 AP、冠军产物、失败方向和待补实验。
- `资料/当前基线修改方向.md`：当前唯一有效流程、问题定义和下一步实现合同。
- `资料/论文阅读记录.md`：已合并的论文依据与可迁移思想。
- `新开对话阅读内容.md`：新会话快速恢复入口。

历史过程日志已压缩；被后续结论覆盖的 official 专题文件、旧流程文件和原始论文清单已删除。详细数值仍保存在对应 `summary.json` 中。

## 当前结论

第一创新点已在 official100 场景隔离五折上形成最终冠军：**轨迹伤害抑制 + pair-union 关系几何补全**。它相对纯原始基线主 AP 提升 `+2.274604` 个百分点，相对冻结共存对照提升 `+0.591731` 个百分点，且相对上一冠军的主 AP 五折 `5/5` 为正。

该最终组合已完成唯一有效的 safety60 冻结复验：`48.424603 / 65.260285 / 75.990715`，相对冻结共存为 `+0.234963 / +0.209909 / -0.177901` 个百分点。主 AP/AP50 为正但 AP25 仍为负，故不满足三指标全正。第一创新点停止独立调参，但完整策略及全部模块冻结保留，不删除；第二创新点迁移也已结束，现在应在最终系统级比较中决定是否启用或移除轨迹伤害抑制及组合策略。不在 safety60 或 even48 回调。此前 v1 复验遗漏 50/50 平衡训练概率到自然正例率的先验校正，属于 `invalid_contract`，其指标不得采信。

第二创新点 Z0-Z6f 及唯一一次 safety60 单向迁移均已完成。Z6f official100 为 `32.808791/40.365582/43.812223`，相对 Z3 控制 `+0.053051/+0.067555/+0.042980`；safety60 control/Z6f 为 `32.640954/43.604318/52.040861` 与 `32.635704/43.590443/52.023991`，即 `-0.005250/-0.013874/-0.016870`。负迁移只记录，不能据此删除三项动作或修改 prompt、A/B 对称规则、DINO 阈值、router、预算和任何上游模型。Z6f 分支现已封闭，even48/test60 继续冻结。

当前决策不是“第二创新点完全没有修改空间”，而是**不能再对 Z6f 做局部参数修补**。如果仍需强化第二创新点，只能在 official100 上预注册一个结构性新分支，例如顺序不变的多视图候选类别验证器、YOLO/Alpha/DINO 多视图证据的 pairwise class verification，或显式风险/abstain 模型；不得使用 safety60 选设计或调参。Z6f 当前数值证据较弱，更适合作为语义校准扩展；若要形成强独立创新，必须重新设计核心验证器。下一步优先完成两个创新点的论文叙事、最终系统组合和必要 official100 消融的裁决，不立即继续推理或迁移实验。

以下 Z0-Z6 各段是按时间记录的实验历史；其中“下一步”“只允许”等措辞仅描述当时合同，均由上述当前决策和后文 Z6f safety60 结论覆盖。

第二创新点 Z0 已完成有效重跑。修复了 signed YOLO-World 缓存解包、prediction-index/semantic-ID 混用和全空轨迹静默放行后，official100 共 `5,266` 条轨迹中 `5,233` 条得到合法类别，覆盖率 `99.373338%`。pair-union 当前开放词汇 AP 为 `29.518969 / 35.628993 / 38.373681`，相对 native-only 为 `+1.586814 / +2.368424 / +2.557755`；GT 类别 + 当前分数达到 `62.205105 AP`，当前类别 + GT-only 排序达到 `48.588698 AP`，GT 类别 + 一对一竞争达到 `69.837770 AP`。

Z1 无 GT YOLO-World 多视图分布账本也已完成。`59,997` 个 native、`5,266` 个 track 和 `1,501` 个 pair-union 候选绑定被折叠为 `9,708` 个 exact-geometry 语义证据节点，共选择 `276,967` 个视角，其中 `97.405828%` 产生非零框证据。冻结轨迹 support 投票有效率继续为 `99.373338%`，与 Z0 完全一致；但 independent-review top-1 仅在 `67.341869%` 的有效轨迹上与冻结 support top-1 一致，平均 JS divergence 为 `0.220342`。这确认类别错误、跨视角冲突和跨来源可靠性必须共同进入校准；其后 Z2-Z6 已全部完成。

### 2026-08-11 第二点 Alpha 补全与 OOF 校准最新结论

native exact-geometry 与 pair-union geometry 的有限上下文 Alpha-CLIP 补全已完成并通过无 GT 审计：

- 目录：`docs/diagnostics/z2b_native_union_alphaclip_limited_context_official100_20260811/`；`4,442` 条记录，其中 native `2,941`、pair-union `1,501`，审计 `valid=true`、`error_count=0`。
- native 有效 Alpha `2,934/2,941`，pair-union 自身 geometry 有效 `1,501/1,501`；固定 `crop_padding_ratio=0.50`，不读取 GT、不修改候选。
- 统一无 GT 账本：`docs/diagnostics/z2c_unified_semantic_node_ledger_official100_20260811/`，共 `9,708` 节点。native YOLO/Alpha top-1 一致率 `36.8925%`，track 继承 YOLO/Alpha 一致率约 `26.9470%`；pair-union 自身 Alpha 与 selected-track Alpha 的 JS 均值/中位数为 `0.1191/0.0505`，因此保留两路证据而不硬替换。

在 official100 场景隔离五折上训练的监督集与 OOF 产物：

- 数据集：`docs/diagnostics/z3_semantic_reliability_dataset_official100_20260811/`，`66,589` 条有效 `(node,class)` 假设；native `59,823`、track `5,265`、pair-union `1,501`。GT 只用于 official train 显式标签。
- OOF：`docs/diagnostics/z3_semantic_reliability_oof_official100_20260811/`，固定 frozen manifest SHA-256 `aa657449...3e3e`，五折、低容量 `HistGradientBoostingRegressor`，不含类别 ID 特征。
- AP 评测：`docs/diagnostics/z3_semantic_reliability_oof_ap_official100_20260811_v2_union_frozen_score_control/`。固定 Alpha 融合控制精确复现 `30.311670/36.918962/39.993122`。

当前最稳妥的 OOF 控制为：native/track 使用 joint YOLO+Alpha OOF 分数，pair-union 保留冻结原始低分（`C_joint_native_track_union_frozen_score`）。其 pair-union 全量 AP 为 `32.755740/40.298027/43.769243`，相对冻结 Alpha 控制为 `+2.444070/+3.379066/+3.787121`，相对原始 native-only 为 `+4.823585/+7.037459/+7.953317`。五折 pair-union AP delta 依次为 `+3.1695、+1.0575、+4.1597、+3.3377、-3.2307` 个百分点；因此这是强正向开发结果，但尚未冻结为最终方法。

诊断结论：joint 模型若直接把 pair-union 也重排，AP 降至 `31.019096`；保留 union 原始低分后恢复至 `32.755740`。这证明当时的主要剩余瓶颈是 Z4 的类别条件同类重复竞争/跨来源排序，而不是 Alpha 证据本身或 pair-union geometry。该阶段已由后续 Z4b–Z4d 完成并取代；Z5a/Z5b 也已完成并以剩余几何动作上界过弱终止。

### 2026-08-11 Z4 类别条件竞争首轮结论

Z4 已完成四类 official100 OOF/plan-only 对照，但尚无规则满足冻结条件：

1. 同类 mask IoU≥0.50 局部组件 winner：`66,589` 个候选中只有 `122` 个多候选组件、`260` 个受竞争候选；pair-union AP 相对 hybrid 为 `-0.000319` 个百分点，基本中性，track-only 明显下降。目录：`docs/diagnostics/z4_category_competition_plan_official100_20260811/` 与 `z4_category_competition_ap_official100_20260811/`。
2. native exact-geometry 固定 top-5：保留 `13,587/59,823` 个 native 类别假设，pair-union 达 `32.858441/40.383017/43.907369`，相对 hybrid `+0.102701/+0.084990/+0.138126`；但五折 AP delta 为 `-0.0716/+0.5371/-0.0840/+0.2864/+1.0565`，只有 3/5 正，且 head AP `-0.2015`，不冻结。
3. 自适应 top-5 + 95% OOF score mass、cap20：pair-union `32.789395/40.337377/43.812404`，相对 hybrid `+0.033655/+0.039350/+0.043161`；head/common/tail 全量均微正，但五折仍仅 3/5 正，不冻结。
4. 一对一竞争监督：每场景每类别做 Hungarian candidate-GT 分配，TP50 正例从独立标签 `2,437` 降为 `1,718`。direct union 不再过抬，native+track/pair-union 分别为 `32.696966/32.697318 AP`；union-frozen 版本为 `32.710791/39.676960/43.392972`，相对固定 Alpha 仍三项正且五折 4/5 正，但绝对值低于当前 hybrid。仅对 union 使用一对一 OOF 的混合结果为 `32.745047 AP`，相对 hybrid `-0.010693`，五折 2/5 正，不保留为冠军。

Z4 首轮结论：独立质量目标的 union-frozen hybrid `32.755740/40.298027/43.769243` 仍是开发控制；top-5 是最高全量 AP，但因 fold/head 不稳定不能冻结。其后 Z4b–Z4d 已完成 fold 4 审计和两项预注册稳定性对照；当前不得继续扫描 top-M、IoU 阈值、类别权重、融合权重，也不运行 safety60/even48/test60。

### 2026-08-12 Z4b 审计与 Z4c 预注册

Z4b 只读审计覆盖 official train100 的 `66,589` 行、`41` 个特征。fold 4 的 native/track 平均校准偏差分别为 `+0.005232/+0.005287`，OOF ROC-AUC/MAE 未出现整体崩坏；最大特征漂移主要落在 pair-union，而当前 hybrid 推理实际冻结 pair-union 原始低分。逐类官方 AP 归因显示 fold 4 宏 AP 对极稀疏类别高度敏感：bar/ottoman/shower wall 分别只有 `1/2/2` 个 TP50，却贡献 `-0.984848/-0.761905/-0.750000` 的类别 AP delta。结论是 fold 4 更接近稀有类类内排序方差，而不是全局均值校准漂移。

Z4b 正式审计目录为 `docs/diagnostics/z4b_fold_distribution_shift_official100_20260812_v3_hybrid_rank_audit/`。无后缀目录和 `v2_hybrid_contract` 是审计器完善前的中间输出：前者误把 union OOF 输出当作已应用分数，后者逐类排序含 NaN 类；均不得作为正式结论引用。

审计同时发现训练—应用来源合同不一致：`C_joint` 训练包含 pair-union，但 hybrid 最终只应用 native/track 的模型输出，pair-union 模型输出被丢弃。预注册唯一 Z4c 修正如下：保持 frozen 五折、41 特征、target、sample weight、随机种子和 HGBR 容量全部不变；仅把模型拟合集合改为 native+track，validation 仍覆盖全部来源，pair-union 推理继续冻结原始分数。只运行这一项 official100 OOF/AP 对照，不扫描参数；若不能改善 fold 4 且保持总体/五折稳定，则撤销并停止该修正。safety60/even48/test60 继续禁止运行。

Z4c 已完成并撤销。其 hybrid 为 `32.195337/39.361500/42.895254`，相对原控制 `-0.560402/-0.936528/-0.873989`；fold 4 从 `-3.230659` 进一步降到 `-4.352688 AP`。因此 pair-union 虽不直接采用模型输出，其监督样本对共享可靠性边界仍有正作用，不能简单排除。

Z4b 的最终方法学结论是训练目标与宏 AP 聚合不对齐：候选级损失几乎忽略只有 1–2 个 TP 的类别，但官方 AP 对每个出现类别等权。预注册唯一 Z4d：保留原始全部来源拟合、frozen 五折、41 特征、target、模型容量和随机种子；只在每个训练折内、每个来源内部把各预测类的总训练权重归一为相等，同时保持每个来源的总 base weight 不变。类别 ID 仅用于训练损失分组，不作为模型输入；无阈值、指数或容量扫描。只允许一次 official100 OOF/AP 对照，失败即撤销。

Z4d 已完成，属于“根因验证成立、方法不冻结”。其 hybrid 为 `32.123798/39.577321/43.722213`，相对原控制为 `-0.631940/-0.720706/-0.047029`；head/common/tail 相对原控制为 `-2.717892/+0.747559/+0.582676`。但五折主 AP 相对固定 Alpha 控制全部为正：`+0.9647/+2.5500/+3.6539/+2.1367/+0.3120`，fold 4 从原 joint 的 `-3.2307` 修复为 `+0.3120`。因此类别宏失衡确实解释了 fold 4 不稳定；完全类等权又过度牺牲高频 head 类，不能作为最终规则。按预注册合同不继续扫描平滑指数、截断权重或 head/common/tail 手工权重；当前数值冠军仍是原 union-frozen hybrid `32.755740/40.298027/43.769243`，当前稳定性对照为 Z4d，但两者均不冻结为最终第二点方法。

校准权重微调已停止；后续 Z5a/Z5b 已完成，证明现有 merge/split/boundary-owner 空间没有足以推进 plan-only 的独立上界。Z5c 已终止，仍不运行 safety60/even48/test60。

### Z5 历史执行合同（Z5a/Z5b 已完成，Z5c 已终止）

Z5 分三阶段，必须顺序执行：

1. **Z5a official100 动作空间合同审计（已完成）**：无 GT、无 AP、无训练、无候选修改。以 `z2c_unified_semantic_node_ledger` 的 `9,708` 个节点为语义证据源，盘点并连接已有 merge/split/boundary-owner 动作。merge 先且只复用 `1,501` 个已冻结 pair-union 与其父候选；split/owner 若没有 official100 已物化且可追溯的动作空间，就报告 unavailable，不生成新 mask，不移植旧 safety60 参数。输出动作账本、join coverage、缺失原因和 provenance。
2. **Z5b GT-only global-feasible oracle（仅在 Z5a 合同通过后）**：逐动作比较 no-op 与冻结反事实，报告 IoU、TP25/TP50、类别正确性、场景和类别归因，再聚合 official100 AP/AP50/AP25、head/common/tail 和 frozen 五折。GT 不得生成推理特征或写回动作计划。
3. **Z5c no-GT plan-only（仅当 Z5b 显示跨折、跨场景的稳定独立上界）**：预注册语义—几何判据后生成计划，不立即跑 safety60；若 oracle 只由少数稀疏类/场景主导或上界微弱，直接终止相应动作族。

Z5a 输入固定为 official train100 scene list、实际 SHA 为 `aa657449...3e3e` 的 frozen manifest、Z2c 统一语义节点、Z3 OOF 分数、当前 hybrid AP 控制与 official100 pair-union append plan。建议正式输出目录：`docs/diagnostics/z5a_semantic_geometry_action_space_official100_20260812/`。不得扫描新的 IoU、距离、语义融合、类别平衡、top-M 或模型容量参数。

### 2026-08-12 Z5a 动作空间合同审计结论

Z5a 已完成并通过，正式目录为：

```text
docs/diagnostics/z5a_semantic_geometry_action_space_official100_20260812/
```

审计只读取 official100 冻结几何、Z1/Z2c 语义账本、Z3 OOF 冻结预测字段与已存在 pair-union 点集；未读取 GT，未计算 AP，未训练模型，未选择阈值，未生成或修改候选。主要结论：

- merge 动作空间可用：`1,501/1,501` 个冻结 pair-union，覆盖 `100/100` 场景；每个 child 均精确等于 selected track 与 native exact group 的点集并集，全部为 partial-overlap 父关系；
- 三方语义节点 join 为 `4,503/4,503=100%`，动作级语义完整为 `1,501/1,501`；YOLO/Alpha 的 geometry-own/inherited top probability、margin、entropy、JS/agreement 及当前 hybrid OOF 分数摘要均已写入；
- `6/1,501` 个动作没有 child/track/native 三方共同已选证据帧，但各节点自身语义仍完整；该字段保留为后续 oracle 归因，不据此设阈值；
- split 不可用：没有 official100 上可追溯父候选且已物化子点集的冻结动作账本；旧 safety60 GT-oracle split 不得复用；
- boundary-owner 不可用：没有 official100 可连接的 superpoint owner 动作账本；旧 safety60 MV3DIS owner 资产明确排除。

该审计当时只允许 merge 家族进入 **Z5b GT-only global-feasible oracle**；Z5b 现已完成。split 与 boundary-owner 在本轮终止；不得临时生成动作或迁移旧阈值。Z5c 因 Z5b 上界过弱而不启动。

### 2026-08-12 Z5b merge GT-only oracle 结论

Z5b 已完成，正式目录为：

```text
docs/diagnostics/z5b_merge_global_feasible_oracle_official100_20260812/
```

合同为 append-only、target-wise global-feasible 可行构造：no-op 保持当前 native+track hybrid；动作只追加 Z5a 已存在的冻结 pair-union，类别与分数均保持冻结；GT 只选择离线 oracle 动作，不写推理计划。no-op 与全部 pair-union 控制均以 `0` 误差复现 Z3 正式结果。

主要结果（百分点）：

| 系统 | AP | AP50 | AP25 |
|---|---:|---:|---:|
| no-op：native+track hybrid | 32.740316 | 40.302467 | 43.774625 |
| 当前控制：全部冻结 pair-union | 32.755740 | 40.298027 | 43.769243 |
| Z5b GT-only target-wise oracle | 32.758354 | 40.301568 | 43.773069 |
| oracle 相对 no-op | +0.018038 | -0.000899 | -0.001556 |
| oracle 相对全部冻结 union | +0.002614 | +0.003541 | +0.003826 |

逐动作 `1,501` 个中，冻结类别语义正确 `556` 个；只有 `19` 个产生至少一个新的 official IoU threshold crossing。target-wise 去重后选择 `17` 个动作，分布于 `14` 个场景、`12` 个类别。五折主 AP 相对 no-op 为 `+0.019578/+0.005520/+0.019488/+0.021514/+0.007535` 个百分点，方向 `5/5` 正，但绝对幅度极小；TP25 新增 crossing 为 `0`，TP50 仅 `1`，收益主要来自 `0.70–0.90` 的高 IoU 阈值。全量 AP50/AP25 相对 no-op仍略负。

结论：现有冻结 pair-union 已几乎吃满该 merge 动作空间；即使 GT-only 选择也只能在当前控制上增加 `+0.002614 AP` 个百分点。该上界过弱，不满足进入 Z5c no-GT plan-only 的推进门槛。Z5 merge 家族与先前 unavailable 的 split/boundary-owner 一并在本轮终止；不得训练动作分类器、扫描语义/几何阈值或运行 safety60/even48/test60。第二创新点当前仍以 Z3 union-frozen hybrid `32.755740/40.298027/43.769243` 为数值控制，Z4d 为稳定性消融，Z5 仅作为“几何动作剩余上界不足”的负结论。

### 2026-08-12 Z6a 冻结几何类别候选空间 oracle

Z6a 已完成，正式目录为：

```text
docs/diagnostics/z6a_class_candidate_space_oracle_official100_20260812/
```

该诊断固定 official100 的 `66,763` 个候选、`9,708` 个语义几何节点、当前类别与当前 hybrid 分数合同；GT 只用于离线 oracle 临时选择类别和评测，不写入推理计划。每个节点对 YOLO-World、Alpha-CLIP 分别取 geometry-own／inherited 分布逐类最大值后的 top-5。当前控制以最大误差 `0.0` 精确复现正式 Z3 结果。

当前分数结果（百分点）：

| 类别方案 | AP | AP50 | AP25 |
|---|---:|---:|---:|
| 当前 frozen class | 32.755740 | 40.298027 | 43.769243 |
| YOLO top-5 oracle | 36.007068 | 44.899203 | 49.741343 |
| Alpha top-5 oracle | 41.880954 | 52.646135 | 57.988382 |
| YOLO+Alpha top-5 union oracle | 43.285077 | 54.556894 | 60.200978 |
| full 198-class oracle | 54.734264 | 68.640197 | 75.285969 |

GT-only 一对一分数诊断（百分点）：

| 类别方案 | AP | AP50 | AP25 |
|---|---:|---:|---:|
| 当前 frozen class | 51.771848 | 60.426954 | 63.638121 |
| YOLO top-5 oracle | 53.800346 | 62.574264 | 67.139727 |
| Alpha top-5 oracle | 58.558014 | 69.112959 | 73.295398 |
| YOLO+Alpha top-5 union oracle | 59.953029 | 70.568580 | 75.358955 |
| full 198-class oracle | 69.837770 | 82.061436 | 87.171735 |

TP25/TP50 候选覆盖审计表明正确类别大多已经存在于当前两路证据中：

- TP25 eligible prediction `50,928` 个，当前类别正确率 `5.6668%`；Alpha top-5 包含 GT 类别 `86.3651%`，YOLO+Alpha union 为 `92.7348%`，节点—GT target 覆盖率为 `89.9419%`，union 可修复当前错误的 `92.3879%`。
- TP50 eligible prediction `47,322` 个，当前类别正确率 `5.1498%`；Alpha top-5 包含 GT 类别 `86.9426%`，YOLO+Alpha union 为 `93.3477%`，节点—GT target 覆盖率为 `90.4013%`，union 可修复当前错误的 `93.0712%`。
- current-score 的 YOLO+Alpha union oracle 五折 AP delta 为 `+11.3596/+13.9178/+13.6101/+11.1031/+12.9458`，Alpha top-5 为 `+9.3672/+11.2364/+12.8043/+8.4713/+10.9423`，均为 `5/5` 正。

历史结论：不引入第三个类别生成器，也不恢复几何扩张；当时据此固定已有 YOLO/Alpha top-k 候选空间，转向对象级多视图视觉证据与候选内类别选择。DINOv2 只作为对象外观一致性、跨视角聚合和难例路由特征，不能直接充当 198 类文本分类器；MLLM 仅允许选择性复核冲突/低置信节点，并保留 abstain/fallback。Z6a 是 GT-only 上界，不是实际方法结果；后续 Z6b-Z6f 均已完成。

### 2026-08-12 Z6b 对象级固定 top-3 视角 manifest

Z6b 第一阶段已完成，正式目录为：

```text
docs/diagnostics/z6b_object_view_manifest_official100_20260812/
```

工具 `tools/build_z6b_object_view_manifest_official100.py` 只连接冻结 Z2c 节点、Z1 视角元数据、Z2/Z2b 已选择的 top-3 Alpha 视角和 official100 prepared RGB-D 资产；未读取 GT、未重投影或重选视角、未生成 embedding、未调用 MLLM，也未修改 geometry/candidate/class/score/inference plan。结果：

- official100 `100/100` 场景、`9,708/9,708` 节点严格 join，duplicate 为 0；
- `9,603` 个节点有有效视角，`105` 个节点因冻结 Z2 `min_visible_points=20` 合同无视角，其中 track `98`、native `7`、pair-union `0`；
- 共 `26,483` 个固定视角：native `8,757`、track `13,376`、pair-union `4,350`；节点视角数分布为 0/1/2/3 视角 `105/287/1,752/7,564`；
- 每个已注册视角的 RGB、depth、pose、intrinsics 全部存在，missing asset 为 0；bbox 与有限上下文 `crop_padding_ratio=0.50` 合同已写入 manifest；
- manifest SHA-256 为 `503261f316a0e9e642eb09c63c87d1fd10c9be60b3bd71041d3dfa8fb149107b`。

历史环境记录：当时无 CUDA，Z6b GPU embedding ledger 按合同在创建输出前以 `CUDA unavailable; refusing CPU fallback` 安全退出，且未静默回退 CPU。GPU 工具 `tools/build_z6b_dinov2_object_appearance_ledger.py` 与纯函数测试已实现；后续 GPU 已获授权，正式账本也已完成，以下一段为最终状态。

GPU 后续已获授权并完成正式运行。`docs/diagnostics/z6b_dinov2_object_appearance_official100_20260812/` 覆盖 `100/100` 场景、`9,708` 节点与 `26,483` 个固定视角；`9,603` 节点有 embedding，`105` 个无视角节点保持缺失。独立审计 `valid=true、error_count=0`；逐视角 L2 norm 均值 `0.99999998`，跨视角 cosine mean/median 为 `0.836047/0.855297`，dispersion mean/median/p90 为 `0.163953/0.144703/0.307926`。无 GT、无候选/几何/类别/分数修改。

Z6c 无 GT review input 已完成：`docs/diagnostics/z6c_semantic_review_input_official100_20260812/`。`9,593/9,708` 节点具备完整 candidate+DINO 输入；YOLO/Alpha top-1 冲突 `6,495` 个（`66.9036%`）。监督集 `z6c_candidate_selector_dataset_official100_20260812` 含 `66,763` 个 prediction、`575,106` 个 option，TP50 target 覆盖 `93.4300%`。场景隔离五折 direct selector 虽有较高候选质量 AUC，但改类约 `89.2%`，official100 AP 降为 semantic-only `28.236506/35.230986/37.951838`、semantic+DINO `28.124939/35.069333/37.899258`，均明显低于当前控制。因此 direct argmax 终止；DINO 标量未在 direct 方案中带来 AP 增益。随后唯一允许的 nested-cross-fitted accept/abstain gate 也已完成，结果见下一段。

Z6c/Z6d nested gate 已完成并终止自动 selector 分支。连续 delta gate 目录为 `docs/diagnostics/z6c_nested_abstain_gate_oof_official100_20260812/`，接受新类约 `88.5%`，AP 为 semantic-only `28.418329/35.299120/38.305845`、semantic+DINO `29.205986/36.124037/39.049373`，五折均 `0/5` 正。transition audit（`z6c_nested_abstain_gate_transition_audit_official100_20260812`）证明 gate 把大量 wrong-to-wrong 提议当作零增量并放行：semantic-only 接受项中仅 `37.73%` 是真实修正，semantic+DINO 为 `38.34%`。

针对该目标漏洞，Z6d 只做一次预先定义的二元修正：nested `P(proposed target > current target)>0.5` 才接受，harm 与 wrong-to-wrong 都为负类，不扫描阈值/容量。正式 OOF/AP 目录为 `docs/diagnostics/z6d_nested_improvement_gate_oof_official100_20260812/` 与 `docs/diagnostics/z6d_nested_improvement_gate_oof_ap_official100_20260812/`。接受新类约 `22.6%/23.9%`；semantic-only `31.889356/39.328924/42.816283`（相对控制 `-0.866384/-0.969103/-0.952960`，`2/5` folds 正），semantic+DINO `31.496255/38.893321/42.354466`（`-1.259484/-1.404706/-1.414777`，`1/5` 正）。因此自动 selector/gate 分支正式终止，不再扫描阈值、模型容量或第三种 gate；当时转入的有限预算多视角视觉复核也已作为 Z6e/Z6f 完成。

Z6e/Z6f 有限预算多视角视觉复核已完成。使用固定 official100 路由：Z6d semantic-only `accepted_new_class`、YOLO/Alpha top-1 冲突、恰好 3 个冻结视角、DINO pairwise cosine mean 不低于冻结账本中位数 `0.855297`、每语义节点最多 1 个；共选出 `82` 项、覆盖 `52` 场景。复核器为本地 `Qwen/Qwen2.5-VL-7B-Instruct` 固定 revision `cc594898137f460bfe9f0759e9844b3ce807cfb5`，三张冻结 bbox crop，`min_pixels=100352/max_pixels=200704`。单顺序 CURRENT/PROPOSED/ABSTAIN 版本为 `80 CURRENT + 2 ABSTAIN + 0 PROPOSED`，证明明显偏保守。唯一一次预定义对称修正 Z6f 对 current-first 和 proposed-first 交换 A/B 次序，只有两次语义选择都为 proposed 才改类，否则 abstain 保持当前；结果 `74 ABSTAIN + 8 PROPOSED`，无非法输出。

Z6f 正式目录：manifest `docs/diagnostics/z6e_selective_vlm_review_manifest_official100_20260812/`，对称复核 `docs/diagnostics/z6f_qwen25vl_symmetric_review_official100_20260812/`，AP 汇总 `docs/diagnostics/z6f_vlm_selector_ap_summary_official100_20260812/`。8 个 class mutation 为 `door→closet door`、`bulletin board→blackboard`×2、`windowsill→window`、`table→desk`、`mattress→bed`×2、`printer→copier`。official100 为 `32.808791/40.365582/43.812223`，相对当前控制 `+0.053051/+0.067555/+0.042980`；head/common/tail 增量 `+0.056320/+0.097211/+0.000000`。五折主 AP 为 1 正、1 个 `-0.002287 AP` 微负、3 no-op；方法及 8 个动作在读取 GT 前已冻结，不允许依据逐动作 GT 做删选，也不再尝试 prompt/分辨率/路由变体。Z6f 是当前 official100 新冠军；其后唯一一次冻结 safety60 单向迁移已经完成，结果见下一节。

### 2026-08-12 Z6f safety60 单向迁移结论

完全冻结的 Z6f 已完成唯一一次 safety60 单向迁移，未使用 safety60 训练、选阈值、改 prompt、改预算或删动作。无 GT 输入共 `41,089` 个 candidate bindings，折叠为 `7,764` 个语义节点；full-official100 Z3 对 `41,034` 个合法候选生成预测，明确省略 `55` 个未注册 native class。固定 top-3 manifest 共 `19,156` 个视角，DINO 独立审计 `valid=true、error_count=0`；虽然 safety60 DINO cosine median 为 `0.854696`，router 仍使用冻结 official100 阈值 `0.855297`。

full selector/gate 得到 `7,655` 个 accepted proposal；冻结 router 最终只送审 `61` 项。Qwen 对称复核得到 `56 ABSTAIN + 2 CURRENT + 3 PROPOSED`，最终三项为 `counter→kitchen counter`、`projector screen→whiteboard`、`folded chair→chair`。唯一一次 GT 评测结果（AP/AP50/AP25，百分点）：

| 系统 | AP | AP50 | AP25 |
|---|---:|---:|---:|
| frozen control | 32.640954 | 43.604318 | 52.040861 |
| Z6f symmetric | 32.635704 | 43.590443 | 52.023991 |
| delta | -0.005250 | -0.013874 | -0.016870 |

head/common/tail delta 为 `-0.015510/+0.003704/+0.000000`。结论是 Z6f 的 official100 微增益没有在 safety60 上迁移，且三项主指标均轻微下降；按冻结合同只记录该负迁移，不基于 safety60 删除三项动作或回调方法。正式目录为 `docs/diagnostics/z6f_safety60_transfer_ap_20260812/`；Qwen 输出 SHA-256 为 `26f32bca7181ed5eb5307eeae116ddcc6f1c61bd284bae13d6510516cf26ae7b`。even48/test60 继续冻结，当前不再运行第二创新点的额外迁移评测。

## 第一创新点总结

### 名称

中文：**风险感知的多视图轨迹候选安全接入与关系几何补全**

英文：**Risk-Aware Integration of Multi-view Track Proposals with Relational Geometry Completion**

### 问题定义

在不训练或替换 Open-YOLO 3D 主干、不使用 GT 推理、原始 Mask3D + YOLO-World 候选始终可回退的条件下：

1. 从独立多视图二维 mask 形成类别无关三维轨迹候选；
2. 消除完全相同几何副本造成的重复竞争；
3. 预测轨迹候选的边际伤害并连续降权，而不是硬删除；
4. 对有互补关系的轨迹—基线候选追加 pair-union 几何补全候选；
5. 所有学习、校准和策略冻结只发生在 official train 的场景隔离折内。

### official100 主 AP 路线

下表只比较主 AP，避免混用早期与最终组件评测中 AP25 候选合同的差异。单位均为百分点。

| 阶段 | AP | 相对上一步 | 相对纯原始基线 |
|---|---:|---:|---:|
| 纯原始基线候选 | 65.408913 | — | — |
| 加入多视图轨迹候选 | 66.846155 | +1.437242 | +1.437242 |
| 完全相同几何组感知 | 66.992353 | +0.146198 | +1.583440 |
| 轨迹 OOF 质量排序／冻结共存 | 67.091786 | +0.099433 | +1.682873 |
| 轨迹伤害抑制 | 67.576230 | +0.484444 | +2.167317 |
| 加入 pair-union 补全（最终冠军） | **67.683517** | **+0.107288** | **+2.274604** |

最终冠军的完整 official100 指标：

| 配置 | AP | AP50 | AP25 |
|---|---:|---:|---:|
| 冻结共存对照 | 67.091786 | 82.485205 | 87.035545 |
| 轨迹伤害抑制 | 67.576230 | 83.209877 | 87.945316 |
| 最终冠军 | **67.683517** | **83.235422** | **87.959075** |
| 最终冠军相对冻结共存 | **+0.591731** | **+0.750217** | **+0.923530** |

最终冠军相对上一冠军的主 AP 五折增量为：

```text
+0.058077 / +0.176655 / +0.054212 / +0.137786 / +0.122644
```

pair-union 单独相对冻结共存为 `+0.100521 AP / +0.030119 AP50 / +0.016003 AP25`，主 AP 五折也为 `5/5` 正；与轨迹伤害抑制组合后再获得上表的 `+0.107288 AP`。

### 冻结冠军策略

- native 候选的几何、类别和原始分数保持不变。
- 完全相同 native 几何先折叠为关系节点，避免类别副本重复支配组件关系。
- 轨迹分数为全量 official100 模型输出的质量分数与 `P(keep)` 的固定连续组合；不设 safety 阈值。
- pair-union 仅追加候选，不修改或删除已有候选；分数由双方保守质量下界与 `P(threshold-cross)` 的固定公式产生。
- 两个模块无交叉调权，不扫描指数或门值。

最终模型包：

```text
output/train_candidate_champion_pair_union_combined_full_official100_v2_prior_corrected/model_package.pkl
output/train_candidate_champion_pair_union_combined_full_official100_v2_prior_corrected/metadata.json
```

最终 official100 结果：

```text
output/evaluate_candidate_champion_pair_union_combined_oof_ap_official100_v1/summary.json
output/train_candidate_champion_pair_union_combined_oof_plan_official100_v1/summary.json
output/train_candidate_pair_union_oof_plan_official100_v2/summary.json
```

注意：pair-union v1 计划是无效旧计划，已删除；有效计划是 v2。

## safety60 与 even48 状态

### 类别无关几何／排序结果

| 配置 | safety60 AP/AP50/AP25 | even48 AP/AP50/AP25 |
|---|---|---|
| 原始 Open-YOLO 3D | `47.029171 / 63.490185 / 74.777629` | `53.523950 / 70.924987 / 79.161021` |
| F2 + 专属严格互重复过滤 | `48.044064 / 64.866299 / 76.042935` | `54.878766 / 72.990183 / 80.869714` |

F2 相对稳健 D2b+过滤仅为 safety60 `+0.000421/+0.000550/+0.001139`、even48 `+0.001010/+0.001579/+0.000944`，因此只保留为几何候选前端，不继续扫描 F1/F2 参数。

轨迹伤害抑制与最终组合的 safety60 冻结结果：

| 配置 | AP | AP50 | AP25 |
|---|---:|---:|---:|
| safety60 冻结共存 | 48.189639 | 65.050376 | 76.168616 |
| safety60 轨迹伤害抑制 | 48.290295 | 65.176631 | 75.875680 |
| 最终组合：轨迹伤害抑制 + pair-union（prior-corrected v2） | **48.424603** | **65.260285** | **75.990715** |
| 最终组合相对冻结共存 | **+0.234963** | **+0.209909** | **-0.177901** |

结论：prior-corrected 最终组合的主 AP/AP50 为正，AP25 仍下降；不满足三指标全正，因此不开展 even48 重放，也不据 safety60 结果调整模型、阈值、损失或权重。唯一 AP 聚合完成收据已写入诊断目录。旧 v1 的 `48.353545/65.138110/75.825344` 使用未校正的平衡训练概率，标记为 `invalid_contract`，仅保留审计，不作结果。

### 模块保留状态

| 模块 | official100 OOF 判断 | safety60 判断 | 当前状态 |
|---|---|---|---|
| F2 轨迹前端与严格互重复过滤 | 构成第一点候选增益底座 | 相对原始三项全正 | 保留 |
| exact geometry + 质量排序／冻结共存 | 主 AP 继续正增益 | 相对 F2 三项微正 | 保留为稳定底座 |
| 轨迹伤害抑制 | 相对冻结共存 `+0.484444/+0.724672/+0.909771` | `+0.100656/+0.126255/-0.292936` | 冻结保留，标记 AP25 权衡，待最终裁决 |
| prior-corrected pair-union | 相对伤害抑制 `+0.107287/+0.025545/+0.013759` | 相对伤害抑制 `+0.134308/+0.083654/+0.115035` | 保留；其自身跨集合三项均正 |
| 完整组合 | official100 冠军 `67.683517/83.235422/87.959075` | `48.424603/65.260285/75.990715`，相对冻结共存 AP25 `-0.177901` | 冻结保留为 official100 冠军候选，待第二点完成后决定最终启用 |

这里的“保留”仅表示保留代码、模型、计划账本和结果，不授权继续使用 safety60 选参数。`test60` 尚未运行并继续冻结。

even48 曾对较早的 structured soft suppression 做一次冻结稳健性重放，从 `55.082294/73.226117/80.891462` 到约 `55.156032/73.290909/81.008797`，三项微正；该策略已被后续 official100 冠军替代，不作为当前最终结论。

### 已完成的唯一第一点复验

`champion_track_suppression_plus_pair_union_append` 已复用全 official100 冻结模型权重，并按 official100 OOF 合同使用自然正例率 `0.047577720588447926` 校正 pair-union 概率。复验覆盖 60 场景、39,569 条既有评分记录、1,520 个 append-only pair-union，候选文件修改数为 0；无 GT 预检后只聚合一次 AP/AP50/AP25。唯一有效结果见 `docs/diagnostics/safety60_champion_pair_union_combined_class_agnostic_ap_20260811_v2_prior_corrected/summary.json`，完成收据见相邻隐藏 `.ap_once_receipt.json`。
6. 只有三项均正才可讨论对 even48 原样重放，否则第一点停在 official100 主结论。

该实验是**回顾性迁移复验**，不是独立测试。

## 正式开放词汇瓶颈

safety60 官方开放词汇 AP 已证明：类别无关几何增益尚不能可靠转成开放词汇 AP。

| 配置 | AP | AP50 | AP25 |
|---|---:|---:|---:|
| 纯 native | 29.971029 | 38.686177 | 44.231502 |
| native + F2 | 29.820266 | 38.247102 | 43.621551 |
| 变化 | -0.150763 | -0.439075 | -0.609951 |

head/common/tail AP 变化为 `-0.953639/+0.238703/+0.746089`。主要问题不是缺少更多几何候选，而是：

- 当前 top-1 式多视图语义过早丢失完整类别分布；
- native 与 track 语义／质量分数不可直接比较；
- 完全相同或近同几何的类别副本仍发生类别相关竞争；
- 分类正确性、候选排序和同类重复抑制没有解耦。

报告：`docs/diagnostics/f2_open_vocab_ap_gvc_safety60_20260808/summary.json`。

## 已终止或被替代的方向

只保留影响路线选择的结论：

| 方向 | 结论 |
|---|---|
| 直接以候选质量 `q` 替换最终分数 | safety60 AP 大幅下降；质量只可作辅助证据，不再扫描混合权重。 |
| `structured_lower_relation_veto` | official100 曾 `+0.1615 AP` 且主 AP 五折全正，但 safety60 为 `-0.162377/-0.356668/-0.879877`，已被连续轨迹伤害抑制替代。 |
| 轨迹 marginal joint score | 相对 state head 的 ROC/PR 五折 `0/5` 胜出，未获准进入 AP；不再继续。 |
| pair-intersection | 仅 17 个正例，验证折正例 `2/5/5/1/4`，log-loss 仅 `2/5` 折胜出；未物化、未运行 AP。 |
| F1 直接碎片合并 | 系统 AP 近乎持平但微负；停在消融。 |
| MV3DIS A/B/unknown 边界 owner | 全局可行 oracle 仅 `+0.067988 AP`；1,393 个动作停在 plan-only。 |
| grow/move/resolve 边界动作 | move 失败，resolve 被系统稀释，grow 跨集合不稳；不再调阈值。 |
| SAMPro3D/medoid 候选族 | 未形成可推进的系统增益；大输出已按授权删除，结论保留。 |
| SAM2、IBSp、残差图、直接 union/intersection/adaptive | 已验证不稳定或缺少独立补充，均不再作为当前主线。 |

## 第二创新点：当前开发主线

建议名称：**几何节点上的多视图开放词汇证据校准与类别条件竞争**。

核心问题不是重新训练 Open-YOLO 3D 主干或直接训练新的 200 类分类器，而是把冻结候选展开为 `(geometry node, class)` 假设，建立跨 native/track/pair-union 可比较的语义—几何联合真阳性排序分数，并在局部重叠组件内处理同类重复竞争。

执行顺序：

```text
Z0  固定 native/F2 几何的 GT-only 分类—排序 oracle 分解
Z1  无 GT 的完整 YOLO-World 多视图类别分布账本
Z2  本地 Alpha-CLIP 对象中心／上下文多视图账本（Z1 后再做）
Z3  训练自由融合控制组 + official100 OOF 类别无关可靠性校准
Z4  exact-geometry 类别聚合和冻结组件内类别相关竞争
Z5  语义辅助的 merge/split/boundary-owner plan 与 oracle
```

Z0 已在同一 official100 候选合同下完成，结果目录：

```text
docs/diagnostics/z0_open_vocab_oracle_official100_20260811_v2_fixed/
```

旧目录 `docs/diagnostics/z0_open_vocab_oracle_official100_20260811/` 因错误读取 signed YOLO-World 缓存并混用类别编号空间，已标记为 `invalid_contract`，不得引用。

Z0 固定评测矩阵为：

1. 当前类别 + 当前分数；
2. GT 类别 + 当前分数；
3. 当前类别 + GT-only 理想排序；
4. GT 类别 + GT-only 理想排序；
5. GT 类别 + 一对一匹配/重复竞争理想排序。

并对 native-only、track-only、native+track、pair-union 以及 head/common/tail 分开报告。GT 只用于显式 oracle／评测，不生成推理类别或分数。第五项用于隔离同类重复竞争造成的损失。

### Z0 official100 有效结果

| 来源 | 当前类+当前分数 AP/AP50/AP25 | GT类+当前分数 AP | 当前类+GT排序 AP | GT类+GT排序 AP | GT类+一对一 AP |
|---|---:|---:|---:|---:|---:|
| native-only | `27.932155/33.260568/35.815926` | `54.784376` | `47.972372` | `17.499986` | `60.476576` |
| track-only | `4.969554/8.662474/12.117329` | `16.865692` | `7.960293` | `21.219134` | `21.219578` |
| native+track | `29.507559/35.629562/38.375052` | `62.179954` | `49.959644` | `23.482631` | `69.469720` |
| pair-union | `29.518969/35.628993/38.373681` | `62.205105` | `48.588698` | `24.176046` | `69.837770` |

普通 `GT 类别 + GT IoU 排序` 在大量同类重复候选存在时并不是单调上界；第五项把每个 GT 只分配给一个同类预测后才恢复真实的一对一竞争上界。pair-union 相对 native+track 的当前主 AP 仅 `+0.011410`，说明现有 top-1 类别和分数几乎没有利用新增几何；但一对一 oracle 仍有 `+0.368050 AP`，因此 pair-union 保留并进入统一语义账本。

### Z1 official100 有效账本

有效目录：

```text
docs/diagnostics/z1_yoloworld_multiview_distribution_official100_20260811_v3_frozen_support_vote/
```

- 候选绑定：native `59,997`、track `5,266`、pair-union `1,501`；不改候选、不读取 GT。
- exact-geometry／视角合同去重后为 `9,708` 个语义证据节点；native 本身只有 `2,941` 个唯一几何节点，类别副本不再重复放大证据。
- 当前 ScanNet200 YOLO-World 缓存暴露 `198` 个有效实例 prompt，预测索引为 `0..197`；项目中“200 类”是数据集名称，不应伪造两个不存在的实例 prompt。
- 冻结 all-support top-1 有效 `5,233/5,266`，并与 Z0 当前轨迹类别完全一致。
- sampled-support top-1 与冻结 all-support top-1 在有效轨迹上 `98.738773%` 一致；它只用于分析，不替代冻结类别。
- independent-review top-1 与冻结 support top-1 仅 `67.341869%` 一致；support vs independent JS divergence 为 mean `0.220342`、median `0.157905`、p90 `0.491751`。
- pair-union 自身几何的 all-support top-1 与继承 track 类别仅 `83.544304%` 一致，independent top-1 仅 `61.025983%` 一致；因此必须同时保留 inherited inference class 和 union-geometry diagnostic distribution。
- YOLO-World 缓存只保存检测框，不保存逐框二值 mask；Z1 的二维支持定义为 `score × 投影可见点落框比例`，深度一致性由冻结 WORLD_2_CAM visibility 提供。

早期 Z1 无后缀目录和 `v2_global_keys` 是中间账本，分别存在 join key 和 full-support vote 合同问题，已标记为 `invalid_contract`，不得作为正式输入。

### Z2 official100 Alpha-CLIP 独立语义账本（已完成）

对象中心结果：`docs/diagnostics/z2_alphaclip_track_object_center_official100_20260811/`；有限上下文结果：`docs/diagnostics/z2_alphaclip_track_limited_context_official100_20260811/`。

- 两路均为 `100/100` 场景、`5,266` 条轨迹、`5,168` 条有效语义，`98` 条空语义；两路视角选择和有效覆盖完全一致。
- 对象中心 `crop_padding_ratio=0.15`；有限上下文固定为 `0.50`。Alpha mask、视角选择、候选和 YOLO-World 均未改变。
- 两路均通过 `tools/audit_z2_alphaclip_track_semantics.py`：源轨迹键一一对应、198 维逐视角/聚合 logits 有限且自洽；精确 logit 并列按最大值并列集合处理。
- 无 GT 对比账本：`docs/diagnostics/z2_alphaclip_vs_z1_track_distribution_official100_20260811/summary.json`。有限上下文与冻结 support top-1 一致率 `26.947040%`，对象中心 `20.580218%`；Alpha 与 YOLO 的 JS divergence 较高，不能直接覆盖 frozen YOLO-World。

### Z3 official100 训练自由控制组（已完成）

YOLO-only 结果：`docs/diagnostics/z3_yoloworld_control_group_official100_20260811_v1/summary.json`；Alpha 融合结果：`docs/diagnostics/z3_alphaclip_fusion_control_group_official100_20260811_v1/summary.json`；总审计：`docs/diagnostics/z3_control_group_audit_official100_20260811/summary.json`。

- YOLO-only `frozen_current/pair_union` 精确复现 Z0：`29.518969/35.628993/38.373681`；7 个替代 top-1/abstain/概率乘权规则均未提升主 AP。
- 最佳固定融合是 `frozen_context_equal_top1`：冻结 YOLO support 分布与有限上下文 Alpha 分布等权（`0.5/0.5`），pair-union 为 `30.311670/36.918962/39.993122`，相对 Z0 `+0.792701/+1.289969/+1.619441 AP`（AP/AP50/AP25）。tail AP `+2.201957`，head AP 基本不变。
- 对象中心等权仅 `+0.427271 AP`；Alpha 单独 top-1、agreement-abstain 均低于 frozen。该结果支持主线：YOLO-World 保持主语义，有限上下文 Alpha 作为软证据补充，不能硬覆盖或只在一致时保留。
- 所有 Z3 均为 official100 显式 GT-only evaluator；不训练、不改候选、不运行 safety60/even48/test60。

该历史阶段随后进入 official train100 场景隔离五折 OOF 的低容量、类别无关可靠性校准，并已由 Z3–Z4d 完成；后续 Z5a/Z5b 也已完成并以剩余动作上界过弱终止。当前不再手调固定融合或校准权重，且 safety60 仍不能用于回调或选择新方向。

### 第二点训练与评测合同

- Z0–Z2 先做无训练 oracle、账本和融合控制组。
- 若 Z0 证明主要瓶颈是排序和跨来源分数不可比，允许只在 official train100 上训练低容量、类别无关的校准器；它估计 `(node, class)` 真阳性概率，不直接学习 200 类分类替换，也不训练 Mask3D/YOLO-World 主干。
- official100 使用场景隔离五折 OOF；每折外场景不得参与本折训练或校准。策略冻结后才用 official100 全量拟合最终校准器。
- safety60 已完成第一点复验；对第二点只允许在方法冻结后做一次迁移 AP，不能训练、选阈值、选融合权重或回调。even48 只允许冻结后原样重放，test60 继续冻结。

## 数据纪律

- official train：允许场景隔离训练、校准和五折 OOF。
- safety60：回顾性开发／迁移复验集；不训练、不选阈值、不回调策略。
- even48：已使用的零场景重叠 robustness set；只允许冻结后原样重放。
- test60：继续冻结。
- GT：只允许 official train 监督、显式标记的离线 oracle 和最终评测；不得进入推理特征。
- 当前工作区很脏，但用户已决定暂不清理；不得运行 `git clean`、`git reset` 或回退用户修改。

## 存储清理

2026-08-11 已完成清理约 `19.60 GB` 的 smoke、缓存和已终止分支。精确路径、删除前大小和复核结果见 `docs/cleanup_manifest_20260811.md`。当前冠军、正式大型缓存和用户工作区修改不在范围内。
