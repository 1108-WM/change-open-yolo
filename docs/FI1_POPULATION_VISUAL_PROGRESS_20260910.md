# 代表样本真实视觉导出与上游依赖复核

**2026-09-10 19:03 北京时间检查：** 92/92 场景、2,283/2,283 几何、333/333 代表目标已完成逐场景导出和审计。`run/completion.json` 仍未出现，原进程继续进行全量特征生成/审计收尾，尚不能宣布完整任务结束；不要重复启动 GPU 导出或提前运行 Qwen。

日期：2026-09-10。**新入口已经安装，真实首批导出与逐类特征集成通过；全量 GPU 任务仍在运行，没有新的 AP。** 后续导出期间已完成30个基础质量/union模型的排除重建及Qwen完整词表CPU接口，详见 [并行工作结果](FI1_EXPORT_PARALLEL_RESULTS_20260910.md)。完整监督上游和逐类评分器仍未完成；下方未训练的表述记录本报告初次完成时状态。

## 已完成的实现

实际工作树新增 3 个工具模块、1 个测试文件、1 份说明，未覆盖旧导出器或 evaluator，未提交/推送。

- `tools/fi1_population_visual_core.py`：固定样本一跳请求、实际点集读取、按完整像素寻址的有限容量模型缓存。
- `tools/export_fi1_population_visual_evidence.py`：冻结计划、逐场景完整导出、断点恢复、全量完成后的特征生成与审计。
- `tools/audit_fi1_population_visual_evidence.py`：独立验证请求集合、点集、场景/帧绑定、RGB/SAM 裁剪像素及全部类别聚合。
- `tests/test_fi1_population_visual_evidence.py`：13 项新测试，包括重算文件哈希后仍能发现的错绑、缩减类别、假票数、裁剪改动，以及 SAM 非确定重算、Alpha 输入/模型/尺度顺序错误。
- `docs/FI1_POPULATION_VISUAL_20260910.md`：入口、缓存、恢复和审计边界。

安装后新旧相关回归 **81 passed**；安装文件与暂存副本哈希一致，语法和 diff whitespace 检查通过。该 81 是本轮所选回归集合，不与上一轮 80 项简单累加。

## 导出范围和真实集成结果

代表样本仍为原来冻结的 **333 个几何、92 个 official100 场景**。不改变样本，把其全部直接相交邻居加入后，共 **2,283 个唯一几何**。只作上下文的节点不继续扩展二跳；因此它们自身仍可能有缺失邻居，不能宣称全部 2,283 个几何都具有完整上下文。

新入口直接读取 population 点集，支持原入口无法处理的历史拒绝/临时细化几何。原低粒度标签图投票和 bbox 证据继续保留；Alpha 仍采用最多 20 视角、SAM ViT-B、0/.2/.4 三尺度及可见率特征聚合，输出完整 198 类，不强制 top1、不做 SMS、不产生最终分数。

SAM 缓存只存图像特征，按完整像素、形状和类型寻址，框提示仍逐次运行。首次缓存命中重新推理并逐位比较实际图像特征，本次实测通过。Alpha 只复用同一模型下完全一致且顺序相同的三尺度 RGB/alpha 批次。相近几何或相似图片不足以触发复用。

固定首场景 `scene0001_01` 的 7 个几何均具备三种证据，140 个 Alpha 视角全部有效，完整导出约 79 秒，独立审计通过。全量进程重新读取这个场景时再次核验全部产物与审计结果，真实验证了已完成场景的恢复路径。

进一步取冻结计划中字典序前三个场景作接口集成：

| 场景 | 几何 | 固定代表几何 | Alpha 选中/有效视角 |
|---|---:|---:|---:|
| scene0001_01 | 7 | 2 | 140 / 140 |
| scene0007_00 | 14 | 2 | 278 / 278 |
| scene0018_00 | 26 | 2 | 300 / 298 |
| 合计 | **47** | **6** | **718 / 716** |

47 个几何的 Alpha、标签图、bbox 均可用，覆盖 native、track、pair_union、refined_union、temporary_refined 五种来源，其中新恢复的 `scene0018_00:union:0001:temporary`（15,938 点）也已经真实走通。2 个无效视角按原三尺度有效前景规则排除，没有填父类别或零分。

这 47 个几何生成 **9,306 行 × 36 维**逐类特征，并通过独立逐行复算；上述 6 个代表几何及其所有直接几何邻居均已导出，三种自身证据均可用。该固定集成包距完整 333 个代表几何仍缺 327 个，不等于全量进程的最新进度。

## 上游复核带来的具体修正

本轮没有将旧 OOF 分数直接接入训练，而是继续核对实际数据与代码：

1. **旧 FI1-A 并非只使用原始几何信息。** 其 5,266 条 track 记录的 `frozen_score` 来自更早的质量/抑制链；1,501 条 union 还包含 `base_quality`、`track_quality_q`、`native_group_median_quality_q`、`threshold_cross_probability`、`balanced_fit_raw_probability`。因此不能只换最后一层的 fold 划分就声称上游已经隔离。
2. **基础 union 点集的重建范围可以缩小。** 对冻结的 1,504 条合格原关系，重新读取原 native mask 与 track 点集，逐条合并，并与历史保存点集及当前 population 逐点比较，全部对应同一批 1,501 个 union。在这个固定关系/资格集合内，学习分数只改变同点集的代表来源选择，不改变 union 点集集合。3 个具有两条支持关系的 union，其父来源角色划分都不同，位于 `scene0079_00`、`scene0576_01`、`scene0587_03`；这些选择会影响后续 C 的 shared/native-only/track-only 原子角色，仍须按排除集合重放。该结论以当前固定关系与资格集合为条件，没有认证更上游的全部资格生成过程。
3. **质量目标要与实际 evaluator 对齐。** 旧 A/C 的 `_quality_target` 使用 `>=` 和 0.50–0.95 共 10 个阈值，当前 evaluator 使用严格 `>` 和 0.50–0.90 共 9 个主阈值，0.25 单列。合成边界检验中 IoU=0.5 的旧质量是 0.1，而 evaluator 对齐值是 0。不能静默重写旧目标或历史 AP；新训练需要独立版本和相同 evaluator 合同。

上述复核未访问新的 GT 文件、未用旧 label 字段筛选样本、未训练模型。原 15 个排除集合仍然有效，但每个集合里的模型及依赖数据尚需实际重建/回放。已经存在且点集完全一致的视觉证据可复用；重新生成的不同 refined 点集需要新编码。

## 运行位置与后续接续

产物根：[.local/fi1_population_visual_20260910](../.local/fi1_population_visual_20260910/)。

- [冻结视觉计划](experiment_summaries/fi1_population_visual_20260910/run/plan.json)
- [全量最新进度](experiment_summaries/fi1_population_visual_20260910/run/progress.json)
- [全量日志](https://github.com/1108-WM/change-open-yolo/blob/main/docs/EXTERNAL_ASSETS.md)
- [启动记录](experiment_summaries/fi1_population_visual_20260910/full_launch.json)：2026-09-10 11:59:13 北京时间启动，PID 3294740。
- [固定前三场景集成结果](experiment_summaries/fi1_population_visual_20260910/integration_result.json)
- [逐类特征集成审计](experiment_summaries/fi1_population_visual_20260910/integration_features_audit.json)
- [上游点集/依赖复核](experiment_summaries/fi1_population_visual_20260910/upstream_preflight/summary.json)

**继续时先查看进度和日志，不要另起一份同清单 GPU 任务。** 已完成的场景位于 `run/scenes/`，审计位于 `run/audits/`。只有全部场景导出、特征生成及特征审计结束后，才会写入 `run/completion.json`；该文件存在仍不代表训练或 AP 完成。失败目录会保留，修复后用同一冻结计划恢复；保留已有日志，不覆盖失败记录。

接下来先收齐这批证据并核对真实缺失；按排除集合从更早的候选质量/union 概率与父来源选择开始处理依赖，再生成独立质量与新的 refined/邻居上下文训练数据，最后训练共享逐类评分器并作固定输出合同下的消融。当前所有最终 score/label、训练就绪和生产就绪状态仍保持未启用。

审计核验当前产物与聚合，不独立重跑相机投影或完整 SAM/Alpha 模型，也不认证历史 v1 数据为新的嵌套训练安全数据。
