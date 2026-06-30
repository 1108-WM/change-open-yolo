# Codex Handoff

最后更新：2026-06-30

这个文件是给新的 Codex 会话或另一台设备的快速入口。完整实验细节见仓库根目录的 `CURRENT_EXPERIMENT_STATUS.md`。

## 先读顺序

1. `docs/CODEX_HANDOFF.md`：当前仓库状态、哪些东西没有上传、下一步该做什么。
2. `CURRENT_EXPERIMENT_STATUS.md`：完整实验脉络、关键指标、失败路线和下一轮建议。
3. `docs/*_log.md`：各条实验分支的补充记录。
4. `run_evaluation.py`、`utils/backprojection_fusion.py`、`tools/export_sam_fused_proposals.py`：当前主线实现入口。

## 当前研究主线

当前只推进第一个创新点：基于 YOLO-World 的三维候选补全和几何质量提升。

主流程：

```text
YOLO-World 二维检测
-> Segment Anything 二维掩码候选
-> 二维框/掩码反投影候选
-> 超点和多视角一致性过滤
-> 三维连通分量清理
-> 与 Mask3D 原始候选融合评估
```

当前最佳方向仍是候选补全加二维掩码级超点负约束。Alpha-CLIP、YOLOE、多掩码选择、自适应内部种子、简单视角质量门控和初版候选局部超点都已经验证过，但没有形成稳定净收益。

最新一轮已经完成证据图关系规则和约束式实例假设的第一版重构。结论是：只输出缺口核心会产生物体残片；完整核心能基本消除主要伤害；但证据图候选仍不能简单作为新增实例追加。当前优先级是先验证新版关系诊断和候选上界，再决定是否进入最终 AP。

2026-06-30 候选动作判断器诊断 v2：当前仍不要跑最终 AP。已基于 v8 的 10 场景导出做“候选该接受、只用核心、拒绝或人工复查”的诊断版规则。

代码入口：

- `tools/analyze_superpoint_candidate_diagnostics.py`

诊断结果：

- `docs/diagnostics/superpoint_action_diag_10scenes_v2/summary.json`
- `docs/diagnostics/superpoint_action_diag_10scenes_v2/actions.csv`
- `docs/diagnostics/superpoint_action_diag_10scenes_v2/review_lists.json`

v2 规则要点：

- 输出动作：
  - `accept_completion`
  - `keep_core_only`
  - `reject_or_needs_mask3d_support`
  - `manual_review`
- 平面/薄片风险类：
  - `whiteboard`
  - `tv`
  - `door`
  - `curtain`
  - `mattress`
  - `projector screen`
  - `bulletin board`
  - `mirror`
  - `mat`
- `picture` 作为小平面物体单独标记。
- Mask3D 支持不再只靠 seed coverage 放行，当前要求：
  - IoU `>= 0.50`
  - 或 IoU `>= 0.35` 且 seed coverage `>= 0.90`
- 平面/薄片类中等扩张也先进入 `manual_review`，不直接接受。

10 场景 v2 动作分布：

- `accept_completion`: `21`
- `manual_review`: `32`
- `reject_or_needs_mask3d_support`: `9`
- `keep_core_only`: `1`

重点个例动作：

- `scene0011_00 / dishwasher`：`accept_completion`
- `scene0046_02 / toilet`：`accept_completion`
- `scene0046_02 / door`：`reject_or_needs_mask3d_support`
- `scene0011_00 / tv`：`reject_or_needs_mask3d_support`
- `scene0164_01 / picture`：`manual_review`
- `scene0131_00 / mini fridge`：`manual_review`
- `scene0088_02 / projector screen`：`manual_review`

`review_lists.json` 已列出四类需要人工审查的候选：

- 全部 `reject_or_needs_mask3d_support`
- 全部 `keep_core_only`
- `manual_review` 中 `largest_cc_to_point_ratio >= 2.0`
- `accept_completion` 中 `conflict_overlap >= 0.18` 或 `existing_mask_iou < 0.30`

下一步应审查这四类列表，确认规则实现、动作分布和高风险覆盖是否合理。通过前不要跑最终 AP。

2026-06-30 最大点级连通分量裁剪诊断补充：已在严格核心+边界超点候选之后新增“只保留最大点级连通分量”的诊断分支，并复跑 even48 前 10 个代表场景。当前仍只做 `export_only`，没有运行 even48、even96 或最终 AP。

新增实现：

- 每个候选额外输出：
  - `candidateXXXX_superpoint_candidate_largest_cc_points.npz`
- 候选 JSON 新增：
  - `superpoint_candidate_largest_cc_seed_point_count`
  - `superpoint_candidate_largest_cc_seed_points_path`
  - `superpoint_diagnostics.largest_cc_point_level_comparison`
  - `superpoint_diagnostics.largest_cc_cleanup`
- `tools/analyze_superpoint_candidate_diagnostics.py` 已能汇总最大连通分量分支。
- 当前导出版本：
  - `mask_graph_constrained_audit_fix_v8_superpoint_largest_cc_diag`

10 场景输出：

- 候选目录：`/tmp/mask_graph_proposals_scannet200_superpoint_largest_cc_diag_10scenes`
- 汇总 JSON：`/tmp/superpoint_largest_cc_diag_10scenes_summary.json`
- 候选数：`63`
- 观测：`663`
- 支持边 / 弱边 / 冲突边：`857 / 1037 / 709`

四套点集对照：

- 点级候选：平均 `712.0` 点，平均连通分量 `5.62`，单连通 `13 / 63`
- 核心-only：平均 `903.7` 点，平均连通分量 `1.21`，单连通 `51 / 63`
- 严格核心+边界：平均 `1072.6` 点，平均连通分量 `1.43`，单连通 `47 / 63`
- 最大点级连通分量：平均 `1069.7` 点，平均连通分量 `0.95`，单连通 `60 / 63`

重点个例：

- `scene0011_00 / dishwasher`：严格核心+边界 `502` 点、`3` 分量；最大连通后 `500` 点、`1` 分量。
- `scene0046_02 / toilet`：严格核心+边界 `1062` 点、`3` 分量；最大连通后 `1048` 点、`1` 分量。
- `scene0046_02 / door`：严格核心+边界 `1946` 点、`5` 分量；最大连通后 `1871` 点、`1` 分量。

当前判断：

- 最大点级连通分量裁剪主要删除小碎片，平均点数几乎不变；
- 单连通候选从 `47 / 63` 提升到 `60 / 63`，碎片问题明显缓解；
- 该分支没有降低平均冲突重叠，仍不能替代 Mask3D 覆盖和冲突证据；
- 暂不建议直接跑最终 AP。下一步应先人工查看重点个例，或扩到更大诊断集确认主体不被误删。

2026-06-30 严格边界 10 场景扩展诊断补充：已新增 `tools/analyze_superpoint_candidate_diagnostics.py`，用于汇总点级、核心-only、严格核心+边界三套候选。已跑 even48 前 10 个代表场景：

- `scene0011_00`
- `scene0025_02`
- `scene0046_02`
- `scene0077_00`
- `scene0084_01`
- `scene0088_02`
- `scene0131_00`
- `scene0146_00`
- `scene0164_01`
- `scene0193_00`

输出：

- 候选目录：`/tmp/mask_graph_proposals_scannet200_superpoint_strict_boundary_diag_10scenes`
- 汇总 JSON：`/tmp/superpoint_strict_boundary_diag_10scenes_summary.json`

总结果：

- 观测：`663`
- 候选：`63`
- 支持边：`857`
- 弱边：`1037`
- 冲突边：`709`

三套点集对照：

- 点级候选：
  - 平均点数：`712.0`
  - 平均连通分量数：`5.62`
  - 单连通：`13 / 63`
- 核心-only：
  - 平均点数：`903.7`
  - 平均连通分量数：`1.21`
  - 单连通：`51 / 63`
  - 无可靠核心：`3 / 63`
- 严格核心+边界：
  - 平均点数：`1072.6`
  - 平均连通分量数：`1.43`
  - 单连通：`47 / 63`
  - 无可靠核心：`3 / 63`

边界统计：

- 接收边界超点：`124`
- 桥接边界超点：`3`
- 支持不足拒绝：`45`
- 预算限制拒绝：`85`

重点个例：

- `scene0011_00 / dishwasher`：严格核心+边界连通分量 `3`，最大分量占比 `0.996`
- `scene0046_02 / toilet`：严格核心+边界连通分量 `3`，最大分量占比 `0.987`
- `scene0046_02 / door`：严格核心+边界连通分量 `5`，最大分量占比 `0.961`

当前判断：

- 严格边界在 10 场景上仍明显减少碎片；
- 桥接边界保护开始真实触发；
- 边界补全仍会引入少量点级碎片；
- 下一步仍不应跑最终 AP，更适合加入“严格候选后只保留最大点级连通分量”的诊断分支，或人工查看非单连通候选。

2026-06-30 严格边界超点候选诊断补充：本地代码已继续收紧真超点候选的边界规则。当前仍只做 `export_only` 诊断，不改变最终 AP。

新增实现：

- 每个候选输出两套超点点集：
  - `candidateXXXX_superpoint_core_only_points.npz`
  - `candidateXXXX_superpoint_candidate_points.npz`
- 候选 JSON 同时写入：
  - `core_only_point_level_comparison`
  - `point_level_comparison`
- 边界超点现在必须满足：
  - 是最大核心连通区域的一层邻居；
  - 不是桥接边界；
  - 有强支持覆盖，或至少两个部分支持帧；
  - 边界总点数不超过核心点数的 `0.50`；
  - 边界超点数最多 `6` 个。
- 当前导出版本：
  - `mask_graph_constrained_audit_fix_v7_superpoint_strict_boundary_diag`

3 场景复跑结果：

- 输出目录：`/tmp/mask_graph_proposals_scannet200_superpoint_strict_boundary_diag_3scenes`
- 候选数：`23`
- 点级候选：
  - 平均点数：`681.5`
  - 平均连通分量数：`5.17`
  - 单连通：`4 / 23`
- 核心-only 超点候选：
  - 平均点数：`996.1`
  - 平均连通分量数：`1.17`
  - 单连通：`20 / 23`
- 严格核心+边界超点候选：
  - 平均点数：`1237.8`
  - 平均连通分量数：`1.35`
  - 单连通：`20 / 23`

与上一版宽边界相比：

- 平均点数从 `2272.2` 降到 `1237.8`
- 单连通候选从 `18 / 23` 提升到 `20 / 23`
- 边界超点统计：
  - 接收：`51`
  - 支持不足拒绝：`21`
  - 预算限制拒绝：`29`
  - 桥接边界：`0`

当前判断：

- 严格边界规则有效降低了候选膨胀；
- 核心连通优势仍保留；
- 下一步应扩到约 10 个代表场景继续导出诊断，仍不要直接跑最终 AP。

2026-06-24 真超点候选诊断补充：本地代码已包含提交 `eb8ca29` 的第一版真超点候选诊断。它仍然只做导出诊断，不改变最终 AP。当前实现已经做到：

- 核心超点只保留最大可靠连通区域；
- 边界超点只保留该核心的一层邻居补全；
- 冲突超点保持未分配；
- 每个候选额外导出：
  - `candidateXXXX_superpoint_candidate_points.npz`
- 同时输出点级候选和超点候选的：
  - 点数；
  - 覆盖比例；
  - 连通性；
  - 冲突覆盖对照。

第一版 3 场景对照结果：

- 输出目录：`/tmp/mask_graph_proposals_scannet200_superpoint_candidate_diag_3scenes`
- 23 个候选中：
  - 点级候选单连通：`4 / 23`
  - 超点候选单连通：`18 / 23`
- 平均每个候选：
  - 点级点数：`681.5`
  - 超点点数：`2272.2`
  - 点级连通分量数：`5.17`
  - 超点连通分量数：`1.39`

解释：

- 核心连通约束是有效的；
- 但当前“一层边界补全”仍偏宽，超点候选平均约为点级候选 `3.65` 倍大小。

2026-06-24 桥接边界排除补充：已进一步实现“同时邻接多个核心连通区域”的边界超点排除，并新增：

- `proposal.bridge_boundary_superpoint_count`
- `proposal.bridge_boundary_point_count`
- `proposal.bridge_boundary_superpoint_ids`

复跑 3 个场景：

- 输出目录：`/tmp/mask_graph_proposals_scannet200_superpoint_bridge_boundary_diag_3scenes`
- 当前这 23 个候选里没有触发桥接边界排除，结果与上一版真超点候选诊断一致。

这说明：

- 桥接边界保护逻辑已经补齐；
- 但在当前小样本上，主要问题仍是边界补全过宽，而不是桥接边界本身。

2026-06-23 超点诊断定义修复补充：已按最新审计意见继续修正超点前移诊断第一版，并重新运行相同 3 个场景的小规模 `export_only`。这轮新增和确认的要点：

- 候选级强支持、部分支持、深度冲突反对、掩码外反对都改为按“不同帧集合”统计，不再按观测条数统计。
- `reliable_visible_points` 现在就是深度一致点；`visible_coverage_ratio` 改为：
  - 掩码内深度一致点 / 全部深度一致可见点。
- 深度冲突反对与掩码外反对拆开：
  - 深度冲突满足最少点数和比例时可单帧硬否决；
  - 掩码外反对需要至少 `2` 个不同帧才成为硬反对。
- 超点邻接默认收紧为：
  - 最大距离 `0.05`
  - 接触点至少 `3`
  - 接触点占较小超点至少 `0.02`
- 点序不匹配时会直接禁用超点处理；缓存点数不同会明确标成不匹配。
- 超点缓存已可真正复用；导出复用校验已纳入新增超点参数。
- `MASK_GRAPH_EXPORT_CODE_VERSION` 已更新为：
  - `mask_graph_constrained_audit_fix_v4_superpoint_diag_frames`

本轮 3 场景结果：

- 输出目录：`/tmp/mask_graph_proposals_scannet200_even48_superpoint_diag_3_v3`
- 总观测：`313`
- 候选：`23`
- 关系边：`1406`
- 支持边：`446`
- 弱边：`526`
- 冲突边：`434`
- 图连通区域：`125`

结果解读：

- 3 个场景的点顺序都匹配，现有 `point_segments` 仍可直接复用。
- `scene0011_00` 的掩码外反对均值从上一版单场景诊断的 `19.80` 降到 `12.35`，深度冲突基本持平，说明“可靠可见 + 两帧掩码外硬反对”修正有效。
- 23 个候选里有 `21` 个核心只剩 `1` 个连通区域，核心连通统计已明显更合理。
- 这说明下一步可以正式进入：
  - 核心超点连通
  - 边界超点禁止桥接
  - 冲突超点保持未分配

2026-06-23 超点前移诊断补充：已开始把现有 `point_segments` 前移到证据图导出阶段，但当前仍是“只诊断、不改最终结果”。新增 `utils/superpoint_diagnostics.py`，导出阶段现在可以为每个场景保存超点缓存、邻接统计、单观测超点证据和候选级核心/边界/冲突/未定超点摘要。3 个场景的小规模 `export_only` 诊断已完成，结果表明：

- 现有 `point_segments` 可直接复用，且点顺序与场景点云一致。
- 单观测平均只有约 `3~5` 个强支持超点，但常有 `7~10` 个强反对超点，负证据很强。
- 候选级仍频繁出现“边界超点多于核心超点”或“冲突超点不小”的情况，说明点级候选的边界污染问题真实存在。
- 当前最合理的下一步不是继续堆点级阈值，而是把超点真正接入实例范围构建：
  - 核心超点必须连通；
  - 部分支持超点不能桥接两个核心区域；
  - 冲突超点保持未分配。

2026-06-23 代码审计修复补充：针对提交 `e8c6578` 后的审计问题，已修复证据图假设构建中的提前淘汰和独立支持定义问题。当前种子筛选失败不再永久删除观测；独立强支持不再受共同 Mask3D 参照排斥；歧义节点会在当前连通区域内保持暂缓；同帧深度冲突优先于父子包含；可靠 Mask3D 解释默认要求分数 `0.30` 和种子覆盖 `0.50`，类别或分数信息缺失时不回退到普通覆盖。本轮只做源码和状态文档修复，没有运行 even48、even96 或最终 AP。

2026-06-23 运行与划分修复补充：暂缓节点现在记录依赖的假设，若当前假设失败会释放仅由该失败假设造成的暂缓节点，避免永久阻塞。`tools/run_scannet200_even48_mask_graph_eval.sh` 默认改为 `EXPORT_REUSE_EXISTING=0`，默认新输出目录为 `output/mask_graph_proposals_scannet200_even48_constrained_audit_fix_v2`；显式开启复用时，会校验新增关系阈值、假设参数、可靠 Mask3D 分数/覆盖门槛和 `export_code_version`。

2026-06-23 配置口径修复补充：`tools/run_scannet200_even48_mask_graph_eval.sh` 默认 `MODE=export_only`，只导出证据图候选，不进入 `run_evaluation.py`；脚本默认未解释比例改回 `0.60`；导出阶段两个“任意已有 Mask3D 掩码”后置过滤默认关闭。`run_evaluation.py` 和 `utils/backprojection_fusion.py` 新增普通覆盖过滤计数汇总：如果以后显式进入评估，可以直接在 `candidate_summary.ordinary_existing_coverage_filtered_count` 中看到有多少候选被旧普通覆盖口径挡掉。

## 当前结论

- 当前最稳结果仍是候选补全主线，不是 Alpha-CLIP、YOLOE、局部超点或简单层级规则。
- 当前最好四十八场景参考：`0.272610 / 0.345769 / 0.389491`。
- 当前最好九十六场景参考：`0.273030 / 0.340244 / 0.383687`。
- 多关系二维掩码证据图已经接入，边类型包括同物体支持、父子包含、冲突和弱关系；点级投票默认不再失败后恢复完整并集。
- 2026-06-23 证据图关系规则已改成：
  - 独立强支持和 Mask3D 参照辅助支持分开统计。
  - 共同 Mask3D 参照不能单独形成强支持。
  - 按需计算真实深度误差统计，用于深度一致和深度冲突。
  - 类别不一致默认不确定，只有伴随负证据才成为硬冲突。
  - 欠分割桥梁至少需要两类证据触发。
  - 原强支持连通区域内部重新划分，低质量或歧义观测可以保持未分配。
  - 漏检区域使用可靠 Mask3D 解释，不再使用任意 Mask3D 覆盖。
- 当前证据图候选同时保存两个点集：
  - 完整核心：`full_core_seed_points_path`
  - 缺口核心：`gap_core_seed_points_path`
- even48 六组评分口径对照已跑完：
  - 旧缺口核心版本：候选自身分数从 `0.303107 / 0.396944 / 0.443739` 降到 `0.296883 / 0.389690 / 0.436725`；归一化分数从 `0.302818 / 0.396449 / 0.443777` 降到 `0.287448 / 0.378442 / 0.424286`。
  - 完整核心版本：候选自身分数基本不变，`0.303101 / 0.396934 / 0.443723`；归一化分数降到 `0.294147 / 0.387372 / 0.437535`。
  - 完整核心加严格现有覆盖过滤：候选自身分数仍基本不变，`0.303101 / 0.396934 / 0.443723`；归一化分数改善到 `0.298840 / 0.392579 / 0.441919`。
- 实际进入评估的证据图候选诊断：
  - 旧缺口核心：`16` 个进入评估，`1` 个完整漏检物体、`12` 个背景污染或几何错误、`3` 个多物体错误合并。
  - 完整核心：`8` 个进入评估，`1` 个完整漏检物体、`2` 个背景污染或几何错误、`4` 个多物体错误合并、`1` 个重复候选但真实重叠低。
  - 完整核心加严格现有覆盖过滤：`5` 个进入评估，`1` 个完整漏检物体、`2` 个背景污染或几何错误、`2` 个多物体错误合并。
- 因此当前结论不是“证据图无效”，而是：
  - 旧缺口核心会产生残片。
  - 完整核心能解决残片伤害。
  - 严格现有覆盖过滤能减少重复和归一化分数下降。
  - 剩余问题主要是多物体错误合并和平面/大面积类别污染。
  - 上一版支持边正确率被共同 Mask3D 参照明显支配；新版诊断必须分开报告有无共同参照、支持类型和深度冲突区间。
- YOLOE 已放弃为当前主线模块。
- Alpha-CLIP 暂只保留为语义诊断信号，不作为当前最近一步的提分模块。
- 轻量几何判别器和层级特征有排序和诊断价值，但不适合继续做全局硬过滤。

下一步建议：

1. 暂时不要跑最终 AP，也不要扩 even96。
2. 先做少量场景新版导出，观察关系和假设统计是否正常；默认脚本已关闭旧导出复用，输出到新的 `mask_graph_proposals_scannet200_even48_constrained_audit_fix_v2`，且默认 `MODE=export_only` 不进入评估。
3. even48 诊断只比较：
   - 关系准确率。
   - 原强支持连通区域拆分数。
   - 新候选质量。
   - 去重后的 Mask3D 补全收益。
4. 只比较三档参数：保守档、推荐档、诊断放宽档，不做大规模搜索。
5. 如果推荐档在 even48 上改善候选质量和补全上界，再考虑最终 AP。
6. 第一阶段仍不做自动修剪；安全补全只作为离线上界诊断。
7. 下一轮优先把当前超点诊断升级为真正的超点级候选构建，而不是直接跑最终 AP。
8. 真超点候选的第一步应直接在当前诊断结果上实现：
   - 核心超点连通；
   - 边界超点禁止桥接；
   - 冲突超点保持未分配。
9. 当前更优先的后续工作不是直接跑 even48 或最终 AP，而是继续收紧边界超点保留条件，再做少量场景导出对照。

## 已上传到 GitHub 的内容

已上传：

- 原始 OpenYOLO3D 代码。
- 当前修改后的 `run_evaluation.py`、`utils/`、`tools/`。
- 当前实验状态文档和日志文档。
- `.gitignore`，用于排除数据集、模型权重、输出和本地缓存。
- 环境快照：
  - `_backups/openyolo3d_environment_2026-05-22.yml`
  - `_backups/openyolo3d_conda_list_2026-05-22.txt`
  - `_backups/openyolo3d_pip_freeze_2026-05-22.txt`

没有上传：

- 数据集：`data/`、`OpenYOLO3D/`。
- 实验输出和缓存：`output/`、`models/YOLO-World/yolo_world/work_dirs/`。
- 模型权重：`pretrained/checkpoints/`、`pretrained/alpha_clip/`、`pretrained/clip-vit-base-patch32/`、`pretrained/yoloe/`、`*.pt`、`*.pth`、`*.ckpt`、`*.bin`。
- 外部下载仓库：`_external/`。
- 本地 Codex/agent 状态：`.codex/`、`.agents/`、`skills-lock.json`。
- 论文 PDF 和本地临时材料：`related papers/`、`cushion`、临时图片。

## 本机路径提示

当前工作目录：

```bash
/home/jia/wm_open-yolo/OpenYOLO3D
```

本机曾使用的 ScanNet200 数据路径：

```bash
/media/jia/软件1/OpenYOLO3D_datasets/scannet200
```

另一台设备需要重新准备数据、权重和输出缓存。不要期待 `git clone` 后能直接复现实验结果；仓库只同步代码和小型文档状态。

## 环境提示

原始安装说明见：

```text
docs/Installation.md
environment.yml
```

当前机器的环境快照见 `_backups/` 下三份文件。实际曾使用的环境名是：

```bash
openyolo3d
```

常用 Python：

```bash
/home/jia/anaconda3/envs/openyolo3d/bin/python
```

基础语法检查命令：

```bash
/home/jia/anaconda3/envs/openyolo3d/bin/python -m py_compile \
  run_evaluation.py \
  utils/backprojection_fusion.py \
  utils/__init__.py \
  utils/utils_2d.py \
  tools/export_mask_graph_proposals.py \
  tools/filter_backprojection_candidates_clip.py \
  tools/run_gemini_backprojection_verifier.py \
  tools/evaluate_multiview_object_clip_correction.py \
  tools/evaluate_semantic_fusion_head.py \
  tools/search_alphaclip_thresholds.py
```

当前证据图 even48 六组评分口径评估入口：

```bash
bash tools/run_scannet200_even48_mask_graph_score_modes.sh
```

严格现有覆盖过滤版本：

```bash
OUT_DIR=/home/jia/wm_open-yolo/OpenYOLO3D/output/scannet200/subset_sweeps/even48_mask_graph_score_modes_full_core_strict_existing \
MASK_GRAPH_OUT=/home/jia/wm_open-yolo/OpenYOLO3D/output/mask_graph_proposals_scannet200_even48_full_core_strict_existing_gpu \
bash tools/run_scannet200_even48_mask_graph_score_modes.sh
```

## 重要文件

核心执行：

- `run_evaluation.py`
- `utils/__init__.py`
- `utils/backprojection_fusion.py`

候选导出和诊断：

- `tools/export_backprojection_candidates.py`
- `tools/export_sam_fused_proposals.py`
- `tools/export_mask_graph_proposals.py`
- `tools/run_scannet200_even48_mask_graph_eval.sh`
- `utils/superpoint_diagnostics.py`
- `tools/analyze_backprojection_candidates.py`
- `tools/train_candidate_geometry_discriminator.py`

Alpha-CLIP 相关实验：

- `tools/export_multiview_object_clip_features.py`
- `tools/evaluate_multiview_object_clip_correction.py`
- `docs/AlphaCLIP_object_rescore_log.md`

历史实验脚本：

- `tools/run_scannet200_even48_containment_sweep.sh`
- `tools/run_scannet200_even48_sam_mask_support_eval.sh`
- `tools/run_scannet200_even48_seed_merge_policy_eval.sh`
- `tools/run_scannet200_even48_selectable_sam_refine.sh`
- `tools/run_scannet200_96_cc_cleanup_confirm.sh`

## Git 使用

当前远端：

```bash
origin   git@github.com:1108-WM/change-open-yolo.git
upstream https://github.com/aminebdj/OpenYOLO3D.git
```

另一台设备建议使用：

```bash
git clone git@github.com:1108-WM/change-open-yolo.git
```

已有仓库时同步：

```bash
git pull
```

本机继续工作后同步：

```bash
git status
git add <changed source/docs files>
git commit -m "Describe the change"
git push
```

注意：本机当前可能存在未提交的编译产物改动，主要在 `models/Mask3D/.../build` 和 `pointnet2` 编译目录下。这些不应作为项目状态上传。

## 2026-06-23 当前结论

本轮第一阶段已经把证据图从“普通连通分量直接生成候选”推进到“关系诊断和约束式实例假设”：

- `tools/export_mask_graph_proposals.py`
  - 已加入逐点可见性和二维掩码一致性验证。
  - 已保存同实例支持、父子包含、硬冲突、不确定关系。
  - 跨类别跨视角关系不再直接作为硬冲突，而是作为不确定关系。
  - 默认使用约束式实例假设：`--graph_hypothesis_mode constrained`。
  - 输出 `mask_graph_trace.json` 和已有实例修正诊断。
- `tools/analyze_mask_graph_trace_relations.py`
  - 新增关系质量诊断脚本。
- `tools/analyze_applied_mask_graph_candidates.py`
  - 可分析全部导出候选、完整核心、缺口核心、已有实例修正诊断。
  - 可比较“原始已有候选 / 二维证据修剪核心 / 原始加核心补全”。

even48 第一阶段诊断结果：

- 二维掩码观测：`3144`
- 同实例支持边：`2242`
- 同实例支持边正确率：`0.925513`
- 不确定关系：`2576`
- 硬冲突边：`2`
- 最终新增候选：`2`
- 已有实例修正诊断：`352`

关键判断：

- 证据图方向没有被否定。
- 被否定的是：
  - 普通图连通分量直接输出候选。
  - 只输出缺口核心。
  - 无条件用二维核心修剪 Mask3D。
  - 无条件把二维核心补到 Mask3D。
- 缺口核心最高真实交并比均值只有 `0.006815`，基本不能单独作为新增实例。
- 二维证据修剪核心只有 `8 / 352` 条比原始 Mask3D 更好，不能替换。
- 原始加核心补全有 `70 / 352` 条变好，但 `195 / 352` 条变差，不能无条件补全。

因此当前证据图最可靠的用途是：先做已有实例修正上界诊断，再学习或手写一个非常保守的安全补全规则。

## 2026-06-23 下一步

不要先跑 even96，也不要直接把当前新增候选接入最终 AP。当前新增候选只有 `2` 个，且诊断全部为坏候选。

下一步应先做“安全补全规则”：

1. 新增可推理使用的边界质量特征：
   - 超点是否跨越多个二维实例边界。
   - 补全后包围盒体积变化。
   - 补全后连通块数量变化。
   - 颜色、法向和空间连续性。
   - 新增区域是否集中在一个连通超点区域。
2. 对已有实例修正诊断执行动作判断：
   - 保留原始。
   - 安全补全。
   - 拒绝补全。
   - 暂不自动替换。
3. 只允许极少数高置信补全：
   - 新增点比例不能过大。
   - 补全后不能显著增加多物体错误合并风险。
   - 原始 Mask3D 候选本身必须存在明显残缺迹象。
4. 只有 even48 上“补全变好数量明显多于变差数量”，再跑最终 AP 和 even96。

最新诊断输出目录：

```text
output/mask_graph_proposals_scannet200_even48_phase1_relation_fix_gpu
output/scannet200/subset_sweeps/even48_mask_graph_phase1_relation_fix_diagnostics/
```

最新检查命令：

```bash
python -m py_compile tools/export_mask_graph_proposals.py tools/analyze_applied_mask_graph_candidates.py tools/analyze_mask_graph_trace_relations.py
bash -n tools/run_scannet200_even48_mask_graph_eval.sh
git diff --check
```
