# OpenYOLO3D 在 ScanNet200 上的候选补全实验状态

最后更新：2026-06-30

项目路径：

`/home/jia/wm_open-yolo/OpenYOLO3D`

## 0. 下一次对话先看这里

### 本轮对话追加记录

2026-06-30 严格边界 10 场景扩展诊断：按建议从 even48 固定列表取前 10 个代表场景继续 `MODE=export_only`，仍未运行 even48、even96 或最终 AP。

本次新增工具：

- `tools/analyze_superpoint_candidate_diagnostics.py`
  - 汇总点级候选、核心-only 超点候选、严格核心+边界超点候选三套点集。
  - 输出候选点数、连通分量、覆盖比例、冲突覆盖、边界接收/拒绝统计。
  - 支持 `--focus scene:class_name` 检查指定候选。

10 个场景：

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

运行输出：

- 候选目录：
  - `/tmp/mask_graph_proposals_scannet200_superpoint_strict_boundary_diag_10scenes`
- 汇总 JSON：
  - `/tmp/superpoint_strict_boundary_diag_10scenes_summary.json`
- 总观测数：`663`
- 导出候选数：`63`
- 支持边：`857`
- 弱边：`1037`
- 冲突边：`709`

10 场景三套点集整体对照：

- 点级候选：
  - 平均点数：`712.0`
  - 平均连通分量数：`5.62`
  - 单连通候选：`13 / 63`
- 核心-only 超点候选：
  - 平均点数：`903.7`
  - 平均连通分量数：`1.21`
  - 单连通候选：`51 / 63`
  - 无可靠核心候选：`3 / 63`
- 严格核心+边界超点候选：
  - 平均点数：`1072.6`
  - 平均连通分量数：`1.43`
  - 单连通候选：`47 / 63`
  - 无可靠核心候选：`3 / 63`

10 场景边界统计：

- 接收边界超点：`124`
- 桥接边界超点：`3`
- 因支持不足拒绝：`45`
- 因预算限制拒绝：`85`

指定非单连通个例复查：

- `scene0011_00 / dishwasher`
  - 点级连通分量：`11`
  - 核心-only 连通分量：`2`
  - 严格核心+边界连通分量：`3`
  - 严格候选最大分量占比：`0.996`
- `scene0046_02 / toilet`
  - 点级连通分量：`15`
  - 核心-only 连通分量：`3`
  - 严格核心+边界连通分量：`3`
  - 严格候选最大分量占比：`0.987`
- `scene0046_02 / door`
  - 点级连通分量：`2`
  - 核心-only 连通分量：`2`
  - 严格核心+边界连通分量：`5`
  - 严格候选最大分量占比：`0.961`

当前判断：

- 扩到 10 场景后，严格边界规则仍明显降低点级碎片：单连通从 `13 / 63` 提升到 `47 / 63`。
- 相比 3 场景，10 场景中出现了 `3` 个桥接边界超点，说明桥接保护确实有必要。
- 核心-only 比核心+边界更连通，说明边界补全仍可能引入少量点级碎片；不过核心+边界的平均连通分量数仍远低于点级候选。
- 暂不建议最终 AP。下一步更稳的是针对非单连通候选做“边界补全后只保留最大点级连通分量”的诊断分支，或先人工查看 10 场景中的非单连通候选。

2026-06-30 严格边界超点候选诊断：已按最新建议继续收紧边界超点保留条件，并复跑 `scene0011_00`、`scene0025_02`、`scene0046_02` 的 `export_only`。本轮仍只做导出诊断，不运行 even48、even96 或最终 AP。

本次代码改动：

- `utils/superpoint_diagnostics.py`
  - 新增 `core_only_proposal`，单独输出“最大核心连通区域”点集。
  - `proposal` 改为严格边界版本：
    - 核心仍只保留最大可靠连通区域；
    - 边界仍只能是一层邻居；
    - 继续排除桥接边界超点；
    - 边界必须满足强支持覆盖或多帧部分支持；
    - 边界总点数不能超过核心点数的 `0.50`；
    - 边界超点数最多 `6` 个。
  - 新增边界诊断：
    - `rejected_boundary_support_superpoint_count`
    - `rejected_boundary_budget_superpoint_count`
    - 对应超点编号列表。
- `tools/export_mask_graph_proposals.py`
  - 每个候选现在同时导出两套超点点集：
    - `candidateXXXX_superpoint_core_only_points.npz`
    - `candidateXXXX_superpoint_candidate_points.npz`
  - 每个候选同时写入：
    - `core_only_point_level_comparison`
    - `point_level_comparison`
  - 新增边界参数到导出参数记录。
- `tools/run_scannet200_even48_mask_graph_eval.sh`
  - `MASK_GRAPH_EXPORT_CODE_VERSION` 更新为：
    - `mask_graph_constrained_audit_fix_v7_superpoint_strict_boundary_diag`
  - 新增并纳入复用校验：
    - `MASK_GRAPH_SUPERPOINT_BOUNDARY_MAX_POINT_RATIO=0.50`
    - `MASK_GRAPH_SUPERPOINT_BOUNDARY_MAX_SUPERPOINTS=6`
    - `MASK_GRAPH_SUPERPOINT_BOUNDARY_STRONG_MIN_COVERAGE=0.45`
    - `MASK_GRAPH_SUPERPOINT_BOUNDARY_PARTIAL_MIN_COVERAGE=0.30`
    - `MASK_GRAPH_SUPERPOINT_BOUNDARY_PARTIAL_MIN_FRAMES=2`

本次 3 场景复跑：

- 输出目录：
  - `/tmp/mask_graph_proposals_scannet200_superpoint_strict_boundary_diag_3scenes`
- 总观测数：`313`
- 导出候选数：`23`
- 支持边：`446`
- 弱边：`526`
- 冲突边：`434`

三套点集整体对照：

- 点级候选：
  - 平均点数：`681.5`
  - 平均连通分量数：`5.17`
  - 单连通候选：`4 / 23`
- 核心-only 超点候选：
  - 平均点数：`996.1`
  - 平均连通分量数：`1.17`
  - 单连通候选：`20 / 23`
- 严格核心+边界超点候选：
  - 平均点数：`1237.8`
  - 平均连通分量数：`1.35`
  - 单连通候选：`20 / 23`

与上一版宽边界对比：

- 宽边界平均点数：`2272.2`
- 严格边界平均点数：`1237.8`
- 平均减少：`1034.3` 点
- 单连通候选数从 `18 / 23` 提升到 `20 / 23`
- 边界超点统计：
  - 接收边界超点：`51`
  - 因支持不足拒绝：`21`
  - 因预算限制拒绝：`29`
  - 桥接边界：`0`

当前判断：

- 严格边界规则有效降低了候选膨胀，同时保留了核心连通带来的碎片减少优势。
- 当前仍有 3 个候选在点级半径连通统计下不是单连通：
  - `scene0011_00 / dishwasher`
  - `scene0046_02 / toilet`
  - `scene0046_02 / door`
- 当前更合理的下一步不是最终 AP，而是扩到约 10 个代表场景，观察严格边界规则是否稳定降低候选膨胀和碎片。

2026-06-24 真超点候选诊断第一版与桥接边界排除：本地代码已包含提交 `eb8ca29` 的第一版真超点候选诊断，并继续按最新审计意见补上“桥接边界超点排除”。这两步都只影响诊断导出，不改变最终评估结果，也没有运行最终 AP。

第一版真超点候选诊断新增内容：

- `utils/superpoint_diagnostics.py`
  - 在候选级超点摘要中新增 `proposal`：
    - 只保留最大核心连通区域；
    - 边界超点只作为该核心的一层邻居补全；
    - 冲突超点保持未分配。
  - 新增点级候选与超点候选对照统计：
    - 点数；
    - 重叠覆盖比例；
    - 连通性；
    - 点级候选落在冲突超点上的比例；
    - 超点候选与已有 Mask3D 的覆盖指标。
- `tools/export_mask_graph_proposals.py`
  - 每个候选额外导出：
    - `candidateXXXX_superpoint_candidate_points.npz`
  - 候选 JSON 新增：
    - `superpoint_candidate_seed_point_count`
    - `superpoint_candidate_seed_points_path`
- `tools/run_scannet200_even48_mask_graph_eval.sh`
  - `MASK_GRAPH_EXPORT_CODE_VERSION` 更新为：
    - `mask_graph_constrained_audit_fix_v5_superpoint_candidate_diag`

第一版 3 场景对照结果：

- 输出目录：
  - `/tmp/mask_graph_proposals_scannet200_superpoint_candidate_diag_3scenes`
- 23 个候选中：
  - 点级候选只有 `4 / 23` 是单连通；
  - 超点候选有 `18 / 23` 是单连通。
- 全部候选平均：
  - 点级候选点数：`681.5`
  - 超点候选点数：`2272.2`
  - 点级候选连通分量数：`5.17`
  - 超点候选连通分量数：`1.39`
  - 点级候选被超点候选覆盖比例：`0.843`
  - 超点候选被点级候选覆盖比例：`0.423`
  - 点级候选落在冲突超点上的比例均值：`0.086`

第一版结论：

- 核心连通约束方向成立，超点候选明显减少了点级碎片。
- 但“一层边界补全”仍偏激进，超点候选平均点数约为点级候选的 `3.65` 倍。
- 当前真正的收益主要来自“核心连通”，不是“边界补全”。
- 下一步应继续收紧边界保留条件，暂不应直接跑 even48 全量或最终 AP。

2026-06-24 桥接边界排除补充：已显式排除“同时邻接多个核心连通区域”的边界超点，并新增：

- `proposal.bridge_boundary_superpoint_count`
- `proposal.bridge_boundary_point_count`
- `proposal.bridge_boundary_superpoint_ids`

本次复跑：

- 输出目录：
  - `/tmp/mask_graph_proposals_scannet200_superpoint_bridge_boundary_diag_3scenes`
- 场景：
  - `scene0011_00`
  - `scene0025_02`
  - `scene0046_02`

复跑结果：

- 这 3 个场景的 23 个候选中，没有任何候选触发桥接边界排除：
  - `bridge_boundary_superpoint_count = 0`
- 因此与上一版真超点候选诊断相比：
  - 超点候选平均点数不变：`2272.2`
  - 超点候选平均连通分量数不变：`1.39`
  - 单连通候选数不变：`18 / 23`

当前判断：

- 桥接边界排除逻辑已经补齐，后续更大样本上不会再把“同时贴着多个核心连通区域”的边界超点误并入候选。
- 但在当前 3 个场景上，它不是主要矛盾。
- 当前更值得继续推进的是：
  - 收紧边界超点保留条件；
  - 限制边界超点对已有 Mask3D 的过度回填；
  - 继续只做少量场景导出诊断。

2026-06-23 超点诊断定义修复并复跑 3 场景：已按最新审计意见修正“不同帧统计、可靠可见、可见覆盖率分母、邻接收紧、缓存复用与点序不匹配处理”，并重新运行相同 3 个场景的 `export_only` 诊断。当前仍然只做诊断，不影响最终候选与 AP。

本次代码修复：

- `utils/superpoint_diagnostics.py`
  - 候选级核心、边界、冲突统计改为按“不同帧集合”计数，不再按观测条数计数。
  - `visible_coverage_ratio` 改为：
    - 掩码内深度一致点 / 全部深度一致可见点。
  - 新增并使用：
    - `reliable_visible_points`
    - `outside_reliable_visible_points`
    - `outside_reliable_visible_ratio`
  - 深度冲突反对与掩码外反对正式拆开：
    - 深度冲突满足最少 `20` 个掩码内点、最少 `20` 个冲突点和比例阈值时，可单帧硬否决；
    - 掩码外反对基于可靠可见点，且候选级至少需要 `2` 个不同帧才成为硬反对。
  - 超点邻接默认收紧为：
    - 最大距离 `0.05`
    - 边界接触点不少于 `3`
    - 接触点占较小超点不少于 `0.02`
  - 缓存现在真正支持复用；若当前点数与缓存点数不同，会明确把 `point_order_matches_scene_points=False`。
  - 候选级汇总新增：
    - `core_largest_component_superpoint_ratio`
- `tools/export_mask_graph_proposals.py`
  - 超点诊断参数链路补齐：
    - `superpoint_adjacency_min_contact_points`
    - `superpoint_adjacency_min_contact_ratio`
  - 点序不匹配时直接禁用超点诊断，不再继续使用错序超点编号。
  - 每个候选同时保留：
    - 全部可靠观测汇总
    - 选中观测汇总
- `tools/run_scannet200_even48_mask_graph_eval.sh`
  - 默认超点邻接距离改为 `0.05`。
  - 新增并纳入复用校验：
    - `MASK_GRAPH_SUPERPOINT_MIN_CONTACT_POINTS`
    - `MASK_GRAPH_SUPERPOINT_MIN_CONTACT_RATIO`
  - `MASK_GRAPH_EXPORT_CODE_VERSION` 更新为：
    - `mask_graph_constrained_audit_fix_v4_superpoint_diag_frames`

本次 3 场景复跑：

- 运行模式：`MODE=export_only`
- 场景：
  - `scene0011_00`
  - `scene0025_02`
  - `scene0046_02`
- 临时输出目录：
  - `/tmp/mask_graph_proposals_scannet200_even48_superpoint_diag_3_v3`
- 复用校验：
  - 同目录再次运行 `EXPORT_REUSE_EXISTING=1` 已成功复用

导出总结果：

- 总观测数：`313`
- 导出候选数：`23`
- 图关系总数：`1406`
- 支持边：`446`
- 弱边：`526`
- 冲突边：`434`
- 图连通区域数：`125`

场景级超点缓存摘要：

- `scene0011_00`
  - 超点数：`1611`
  - 邻接边数：`4206`
  - 超点点数中位数：`72`
  - 点顺序一致：`True`
- `scene0025_02`
  - 超点数：`1116`
  - 邻接边数：`3078`
  - 超点点数中位数：`76`
  - 点顺序一致：`True`
- `scene0046_02`
  - 超点数：`1270`
  - 邻接边数：`3577`
  - 超点点数中位数：`71`
  - 点顺序一致：`True`

观测级超点证据摘要：

- `scene0011_00`
  - 平均每个观测：
    - 强支持超点：`6.38`
    - 部分支持超点：`7.77`
    - 深度冲突反对：`6.18`
    - 掩码外反对：`12.35`
  - 与上一版单场景诊断相比，掩码外反对均值从 `19.80` 降到 `12.35`，深度冲突基本持平，说明“可靠可见 + 不同帧硬反对”修正起效。
- `scene0025_02`
  - 平均每个观测：
    - 强支持超点：`4.76`
    - 部分支持超点：`3.96`
    - 深度冲突反对：`5.72`
    - 掩码外反对：`10.99`
- `scene0046_02`
  - 平均每个观测：
    - 强支持超点：`6.74`
    - 部分支持超点：`3.89`
    - 深度冲突反对：`4.75`
    - 掩码外反对：`12.98`

候选级超点摘要：

- 23 个候选中有 `21` 个核心区域连通分量数为 `1`，只有 `2` 个候选仍存在核心碎裂。
- `scene0011_00`
  - 候选数：`4`
  - 平均每个候选：
    - 核心超点：`8.00`
    - 边界超点：`9.75`
    - 冲突超点：`3.25`
    - 未定超点：`7.25`
    - 核心最大连通区域点数占比：`0.868`
    - 核心最大连通区域超点数占比：`0.850`
- `scene0025_02`
  - 候选数：`9`
  - 平均每个候选：
    - 核心超点：`3.67`
    - 边界超点：`3.56`
    - 冲突超点：`1.22`
    - 未定超点：`1.56`
    - 核心最大连通区域点数占比：`1.000`
    - 核心最大连通区域超点数占比：`1.000`
- `scene0046_02`
  - 候选数：`10`
  - 平均每个候选：
    - 核心超点：`4.50`
    - 边界超点：`3.60`
    - 冲突超点：`3.50`
    - 未定超点：`0.90`
    - 核心最大连通区域点数占比：`0.994`
    - 核心最大连通区域超点数占比：`0.950`

当前判断：

- “强支持/反对按不同帧统计”已经落地；这 3 个场景里，当前导出的候选恰好没有同帧重复观测混入，但计数语义现在是正确的。
- “可靠可见点”与“掩码外反对”拆开后，强反对没有继续失控，至少 `scene0011_00` 的掩码外反对明显回落。
- 更严格邻接后，大部分候选核心已经连通，说明可以进入下一阶段：
  - 核心超点连通；
  - 边界超点禁止桥接；
  - 冲突超点保持未分配。
- 当前仍不应直接跑最终 AP；下一步应实现真正的超点候选构建，再做少量场景验证。

2026-06-23 超点前移诊断第一版：已开始把现有 `point_segments` 前移到证据图导出阶段，但目前只做“诊断输出”，还没有改最终评估结果。本轮已完成 3 个场景的小规模 `export_only` 导出诊断，没有运行 even48 全量、even96 或最终 AP。

本次代码改动：

- 新增 `utils/superpoint_diagnostics.py`
  - 读取 ScanNet200/Mask3D 现有超点编号，建立场景级超点缓存。
  - 输出每个超点的点数、中心、包围范围、平均颜色、平均法向、平面程度。
  - 基于点邻接近邻统计超点邻接边，保存边界接触数量、平均距离、法向差、颜色差。
  - 对每个二维观测输出超点级证据：
    - 强支持
    - 部分支持
    - 强反对
    - 仅触达
  - 对每个证据图候选汇总：
    - 核心超点
    - 边界超点
    - 冲突超点
    - 未定超点
- `tools/export_mask_graph_proposals.py`
  - 新增 `processed_scene_path` 接口，直接复用现有 `point_segments`。
  - 导出阶段新增可选 `--superpoint_diagnostics`。
  - 每个场景现在会额外输出：
    - `superpoint_cache/superpoint_cache.npz`
    - `superpoint_cache/superpoint_cache_summary.json`
    - `superpoint_observation_evidence/observation*_superpoints.json`
    - `superpoint_observation_evidence/observation_superpoint_summary.json`
  - 每个候选 JSON 现在会额外写入 `superpoint_diagnostics`。
- `tools/run_scannet200_even48_mask_graph_eval.sh`
  - 默认打开超点诊断导出。
  - `MASK_GRAPH_EXPORT_CODE_VERSION` 更新为 `mask_graph_constrained_audit_fix_v3_superpoint_diag`。
  - 新增超点诊断环境变量：
    - `MASK_GRAPH_SUPERPOINT_DIAGNOSTICS`
    - `MASK_GRAPH_SUPERPOINT_ADJACENCY_KNN`
    - `MASK_GRAPH_SUPERPOINT_SUPPORT_MIN_COVERAGE`
    - `MASK_GRAPH_SUPERPOINT_PARTIAL_MIN_COVERAGE`
    - `MASK_GRAPH_SUPERPOINT_MIN_VISIBLE_POINTS`
    - `MASK_GRAPH_SUPERPOINT_MIN_DEPTH_CONSISTENCY`
    - `MASK_GRAPH_SUPERPOINT_REJECT_MIN_DEPTH_CONFLICT`

本次 3 场景诊断运行：

- 运行模式：`MODE=export_only`
- 场景：
  - `scene0011_00`
  - `scene0025_02`
  - `scene0046_02`
- 临时输出目录：
  - `/tmp/mask_graph_proposals_scannet200_even48_superpoint_diag_3`

导出总结果：

- 总观测数：`313`
- 导出候选数：`23`
- 图关系总数：`1406`
- 支持边：`446`
- 弱边：`526`
- 冲突边：`434`
- 图连通区域数：`125`

场景级超点缓存摘要：

- `scene0011_00`
  - 超点数：`1611`
  - 邻接边数：`4423`
  - 超点点数中位数：`72`
  - 点顺序与场景点云一致：`True`
- `scene0025_02`
  - 超点数：`1116`
  - 邻接边数：`3218`
  - 超点点数中位数：`76`
  - 点顺序与场景点云一致：`True`
- `scene0046_02`
  - 超点数：`1270`
  - 邻接边数：`3730`
  - 超点点数中位数：`71`
  - 点顺序与场景点云一致：`True`

观测级超点证据摘要：

- `scene0011_00`
  - 平均每个观测：
    - 强支持超点：`3.15`
    - 部分支持超点：`9.75`
    - 强反对超点：`9.97`
    - 触达超点：`19.67`
  - 含强反对超点的观测：`54 / 60`
- `scene0025_02`
  - 平均每个观测：
    - 强支持超点：`3.17`
    - 部分支持超点：`4.77`
    - 强反对超点：`9.57`
    - 触达超点：`14.63`
  - 含强反对超点的观测：`90 / 115`
- `scene0046_02`
  - 平均每个观测：
    - 强支持超点：`4.57`
    - 部分支持超点：`5.29`
    - 强反对超点：`7.51`
    - 触达超点：`16.42`
  - 含强反对超点的观测：`122 / 138`

候选级超点摘要：

- `scene0011_00`
  - 候选数：`4`
  - 平均每个候选：
    - 核心超点：`4.00`
    - 边界超点：`6.00`
    - 冲突超点：`5.00`
    - 未定超点：`12.00`
- `scene0025_02`
  - 候选数：`9`
  - 平均每个候选：
    - 核心超点：`2.44`
    - 边界超点：`3.00`
    - 冲突超点：`1.11`
    - 未定超点：`2.78`
- `scene0046_02`
  - 候选数：`10`
  - 平均每个候选：
    - 核心超点：`2.90`
    - 边界超点：`3.50`
    - 冲突超点：`3.40`
    - 未定超点：`1.60`

本轮判断：

- 现有 `point_segments` 可以直接复用，且点顺序正确，不需要先引入新模型。
- 观测级“强反对超点”数量很高，说明二维掩码对三维边界外区域已经提供了很强负证据。
- 候选级仍频繁出现：
  - 边界超点多于核心超点；
  - 冲突超点数量不小；
  - 核心超点内部邻接边为零的候选仍然存在。
- 这说明当前点级候选虽然可作为对照基线，但下一阶段确实应该把超点从“事后修整”前移到“实例范围构建”。

下一步建议：

1. 不直接跑最终 AP。
2. 下一阶段先把超点诊断推进为真正的候选构建约束：
   - 核心超点连通约束；
   - 部分支持超点不能桥接两个核心区域；
   - 冲突超点保持未分配。
3. 先在少量场景上做第一版“超点核心候选”导出，再决定是否接入 Mask3D 补全判断。

2026-06-23 配置口径修复记录：针对最新审计意见，已修正 even48 脚本默认配置与评估报告口径。仍没有运行 even48、even96 或最终 AP。

本次修复要点：

- `tools/run_scannet200_even48_mask_graph_eval.sh`
  - 默认 `MODE=export_only`，只导出证据图候选，不进入 `run_evaluation.py`。
  - `MASK_GRAPH_GAP_MIN_UNCOVERED_RATIO` 默认从 `0.25` 改回 `0.60`，与导出代码推荐值一致。
  - 导出阶段的 `MASK_GRAPH_EXPORT_MAX_EXISTING_IOU` 和 `MASK_GRAPH_EXPORT_MAX_SEED_IN_EXISTING_MASK_RATIO` 默认关闭，不再用“任意已有 Mask3D 掩码”做后置普通覆盖过滤。
  - 上述两个导出后置过滤现在支持真正留空；显式复用时也会按 `None` 口径校验旧摘要。
  - 评估阶段普通已有掩码过滤改成环境变量驱动，并在启用时打印警告。
- `utils/backprojection_fusion.py`
  - 每个场景的回投候选报告新增：
    - `skipped_reason_counts`
    - `ordinary_existing_coverage_filtered_count`
  - 其中普通覆盖过滤计数包含：
    - `matched_existing_3d_mask`
    - `mostly_covered_by_existing_masks`
    - `grown_mask_matches_existing`
- `run_evaluation.py`
  - 汇总所有场景的跳过原因统计。
  - 在总报告 `candidate_summary` 中新增 `ordinary_existing_coverage_filtered_count`，以后即便显式进入评估，也能看见被普通覆盖口径挡掉了多少候选。

2026-06-23 运行与划分修复记录：针对上一提交后的两个增量问题，已继续修复证据图划分和 even48 导出脚本。没有运行 even48、even96 或最终 AP。

本次修复要点：

- `tools/export_mask_graph_proposals.py`
  - 暂缓节点现在记录导致歧义的假设依赖。
  - 如果当前假设随后判为无效，会移除它对暂缓节点的依赖；移除后不再歧义的观测会重新回到当前连通区域划分队列。
  - 有效假设会把暂缓依赖从“当前尝试”更新为正式假设编号，避免失败尝试永久阻塞后续划分。
  - 导出 JSON 参数中新增 `export_code_version`，用于脚本复用校验。
- `tools/run_scannet200_even48_mask_graph_eval.sh`
  - 默认输出目录改为新的 `mask_graph_proposals_scannet200_even48_constrained_audit_fix_v2`。
  - 默认 `EXPORT_REUSE_EXISTING=0`，避免下一次运行误复用旧导出。
  - 即使用户显式开启复用，也会校验新增关系阈值、假设划分参数、可靠 Mask3D 分数/覆盖门槛和 `export_code_version`。
  - 导出阶段显式传入关系阈值、假设参数、可靠覆盖参数和代码版本标识。

2026-06-23 代码审计修复记录：针对提交 `e8c6578` 的增量审计，已修复证据图假设构建和可靠覆盖口径中的阻断问题。本次仍没有运行 even48 新导出、even96 或最终 AP。

本次修复要点：

- `tools/export_mask_graph_proposals.py`
  - 种子筛选失败现在只表示“不能当种子”，不再把非前 30% 质量节点、暂时缺少独立支持的节点永久淘汰；这些观测仍可作为已有假设成员被检查。
  - 欠分割桥梁仍禁止作为种子，也不能作为普通支持节点通过加入检查。
  - 独立强支持只由几何、深度、二维掩码和多视角证据决定，不再因为两条观测碰巧共享 Mask3D 参照而取消独立支持；共同参照另存为诊断标签。
  - 类别不一致但几何和深度证据足够强时，可以保留为类别待定的同实例支持，最终类别仍由投票决定。
  - 同帧关系优先判断可靠深度冲突，再判断父子包含和普通互斥。
  - 歧义节点加入暂缓集合，在当前连通区域划分结束前不再成为种子，也不再反复参与扩张尝试。
  - 无效假设释放其成员，只阻塞失败种子，降低顺序敏感性。
  - `unassigned_observation_count` 改为按观测去重统计。
  - 可靠 Mask3D 解释默认要求已有候选分数至少 `0.30`、对当前核心种子覆盖至少 `0.50`；类别或分数信息缺失时不再静默退回普通覆盖。

验证已通过：

```bash
python -m py_compile tools/export_mask_graph_proposals.py tools/analyze_mask_graph_trace_relations.py tools/analyze_applied_mask_graph_candidates.py
bash -n tools/run_scannet200_even48_mask_graph_eval.sh
git diff --check
```

2026-06-23 最新同步记录：证据图关系规则和约束式假设第一版重构。

本次没有运行 even48 新导出、even96 或最终 AP，只完成代码和诊断口径修正。核心判断是：上一版证据图方向没有被否定，但“共同 Mask3D 参照支配支持边”“弱真实匹配高估正确率”“普通强支持连通区域没有真正被约束拆分”“任意 Mask3D 覆盖压制漏检区域”这些问题需要先修。

本次代码改动：

- `tools/export_mask_graph_proposals.py`
  - 增加按需真实深度误差统计，不保存完整帧点浮点矩阵。
  - 关系判断拆成独立强支持、Mask3D 参照辅助支持、深度冲突、不确定关系。
  - 共同 Mask3D 参照不再能单独形成强支持；类别一致也不单独形成合并依据。
  - 类别不一致默认是不确定关系，只有伴随深度冲突等负证据时才成为硬冲突。
  - 修正 SAM 掩码关系判断的投影坐标缩放口径。
  - 欠分割桥梁改成至少两类证据触发：同帧父子、同帧互斥、三维多区域。
  - 约束式实例假设改成在原强支持连通区域内重新划分：
    - 种子必须非桥梁、有独立强支持、质量处于连通区域前 30%。
    - 观测加入需要两条来自假设成员的支持，或一条高分独立强支持。
    - 同时可加入多个假设的观测保持未分配。
    - 无效假设只保留诊断，不输出候选。
  - 新候选准入使用“可靠 Mask3D 解释”替代任意掩码覆盖：
    - 默认要求已有候选类别兼容。
    - 对当前核心覆盖达到阈值才算已解释。
    - 新候选未解释比例默认提高到 `0.60`。
  - 新候选增加完整核心连通性、类别多数/边距、独立强支持等准入限制。
- `tools/analyze_mask_graph_trace_relations.py`
  - 默认真实匹配口径改为：
    - 候选精确率不低于 `0.60`，或真实实例覆盖率不低于 `0.20`。
    - 且匹配点数不少于 `30`。
  - `unknown` 不参与正确率。
  - 增加按是否有共同 Mask3D 参照、支持类型、深度冲突比例分桶的统计。
- `tools/analyze_applied_mask_graph_candidates.py`
  - 保留并完善按 `scene_name + best_existing_mask_id` 去重的 Mask3D 修正诊断统计。

旧 even48 trace 上的严格关系诊断检查通过。旧 trace 没有新增深度统计字段，因此支持类型仍显示为旧字段回退；该检查只用于验证诊断脚本兼容旧导出结果，不代表新版关系规则的效果。

严格诊断旧结果要点：

- 同实例支持边可靠匹配正确率：`0.992420`
- 有共同 Mask3D 参照的同实例支持边：
  - 数量：`2210`
  - 可靠判断数：`1967`
  - 正确率：`0.995425`
- 无共同 Mask3D 参照的同实例支持边：
  - 数量：`32`
  - 可靠判断数：`12`
  - 正确率：`0.500000`

去重后的 Mask3D 修正诊断仍显示无条件补全不安全：

- 原始 `352` 条已有实例修正诊断对应 `313` 个唯一 Mask3D 候选。
- 最佳去重口径下，原始加核心补全 `66` 个变好、`173` 个变差。
- 保守去重口径下，原始加核心补全 `62` 个变好、`181` 个变差。

当前下一步：

1. 先用新版关系规则导出 even48 的诊断结果，不直接跑最终 AP。
2. 比较关系准确率、原连通区域拆分数、新候选质量、去重后的补全收益。
3. 若推荐档仍过严或过松，再只跑保守档、推荐档、诊断放宽档三档，不做大规模搜索。
4. 只有新增候选质量和去重补全收益明确改善后，再进入最终 AP。

2026-06-22 最新同步记录：固定评分口径、证据图候选诊断、完整核心实验。

本次先验证前面讨论的顺序：

```text
固定评估口径
-> 分析实际进入评估的证据图候选
-> 同时保存完整核心和缺口核心
-> 再判断新增、补全、替换、拒绝
```

结论：这个顺序是对的。尤其是“先分析 16 个候选，再改候选结构”非常必要；诊断结果证明，旧版只输出缺口核心时，进入评估的候选大多不是完整物体，而是残片、背景污染或多物体错误合并。

本次代码改动：

- `run_evaluation.py`
  - 新增 `--dataset_root`，默认仍使用 `./data/<dataset>`。
  - 支持通过 `OPENYOLO3D_DATA_ROOT_SCANNET200` 或 `OPENYOLO3D_DATA_ROOT` 指定数据根目录。
  - 目的：避免不同设备或挂载名变化时，仍写死 `./data/scannet200`。
- `tools/export_mask_graph_proposals.py`
  - 新增 `--dataset_root`，导出证据图候选时和评估脚本使用同一数据根目录。
  - 每个证据图候选现在同时保存：
    - `full_core_seed_points_path`：完整核心点集。
    - `gap_core_seed_points_path`：缺口核心点集。
  - JSON 中新增：
    - `full_core_seed_point_count`
    - `gap_core_seed_point_count`
  - 保留 `seed_points_path` 作为当前实际输出给融合使用的点集。
- `tools/run_scannet200_even48_mask_graph_eval.sh`
  - 新增 `DATASET_ROOT` 环境变量。
  - 新增 `EVAL_SCORE_MODE` 环境变量，对应 `uniform`、`native`、`calibrated`。
  - 新增 `no_graph_refill` / `baseline_refill` 模式，用于同参数不加证据图基线。
  - 证据图默认点集策略改成 `MASK_GRAPH_GAP_SEED_POLICY=full_core`。
  - 证据图导出阶段默认更严格：
    - `MASK_GRAPH_EXPORT_MAX_EXISTING_IOU=0.30`
    - `MASK_GRAPH_EXPORT_MAX_SEED_IN_EXISTING_MASK_RATIO=0.30`
  - 旧导出复用检查新增对这两个导出过滤参数的校验，避免误复用旧目录。
- 新增 `tools/run_scannet200_even48_mask_graph_score_modes.sh`
  - 一次跑六组 even48 对照：
    - 统一分数，不加证据图
    - 统一分数，加入证据图
    - 候选自身分数，不加证据图
    - 候选自身分数，加入证据图
    - 场景内归一化分数，不加证据图
    - 场景内归一化分数，加入证据图
  - 输出 `score_mode_summary.csv/json`。
- 新增 `tools/analyze_applied_mask_graph_candidates.py`
  - 只分析最终评估报告中实际进入评估的证据图候选。
  - 使用 `output/scannet200/scannet200_ground_truth_masks` 和 `output/scannet200/scannet200_masks` 计算：
    - 与真实实例的最高三维交并比。
    - 与已有 Mask3D 候选的重复程度。
    - 是否跨多个真实实例。
  - 输出中文诊断类别：
    - 完整漏检物体
    - 可用于补全
    - 物体残片或重复局部
    - 重复候选
    - 背景污染或几何错误
    - 多物体错误合并
    - 重复候选但真实重叠低

本次 GPU 实验一：旧缺口核心版本的六组评分口径对照。

运行命令：

```bash
bash tools/run_scannet200_even48_mask_graph_score_modes.sh
```

使用旧证据图目录：

```text
output/mask_graph_proposals_scannet200_even48_gap_compete_priority_gpu
```

输出目录：

```text
output/scannet200/subset_sweeps/even48_mask_graph_score_modes
```

结果：

```text
统一分数，不加证据图：0.271195 / 0.345761 / 0.389522
统一分数，加入证据图：0.271186 / 0.345749 / 0.389509
候选自身分数，不加证据图：0.303107 / 0.396944 / 0.443739
候选自身分数，加入证据图：0.296883 / 0.389690 / 0.436725
场景内归一化分数，不加证据图：0.302818 / 0.396449 / 0.443777
场景内归一化分数，加入证据图：0.287448 / 0.378442 / 0.424286
```

候选统计：

```text
证据图导出候选：53
最终进入评估的多视角证据图候选：16
```

对这 16 个进入评估的证据图候选做诊断：

```text
完整漏检物体：1
背景污染或几何错误：12
多物体错误合并：3
平均最高真实三维交并比：0.076856
```

结论：旧缺口核心版本不是“分数没排上”，而是候选质量本身不够。很多候选点几乎落在某个真实物体内部，但只是一小片，三维交并比很低；这验证了“只保存缺口点会产生物体残片”的判断。

本次 GPU 实验二：完整核心版本。

输出目录：

```text
output/scannet200/subset_sweeps/even48_mask_graph_score_modes_full_core
```

证据图导出目录：

```text
output/mask_graph_proposals_scannet200_even48_full_core_gpu
```

结果：

```text
统一分数，不加证据图：0.271195 / 0.345761 / 0.389522
统一分数，加入证据图：0.271195 / 0.345760 / 0.389522
候选自身分数，不加证据图：0.303107 / 0.396944 / 0.443739
候选自身分数，加入证据图：0.303101 / 0.396934 / 0.443723
场景内归一化分数，不加证据图：0.302818 / 0.396449 / 0.443777
场景内归一化分数，加入证据图：0.294147 / 0.387372 / 0.437535
```

候选统计：

```text
证据图导出候选：42
最终进入评估的多视角证据图候选：8
```

对这 8 个进入评估的证据图候选做诊断：

```text
完整漏检物体：1
背景污染或几何错误：2
多物体错误合并：4
重复候选但真实重叠低：1
平均最高真实三维交并比：0.116949
```

结论：完整核心明显消除了“缺口残片”造成的主要下降。候选自身分数口径下几乎不再伤害 AP，只下降 `0.000006`。但剩余坏候选主要变成“多物体错误合并”，说明下一步要做冲突否决、候选竞争和新增准入，而不是回到缺口核心。

本次 GPU 实验三：完整核心 + 更严格现有覆盖过滤。

运行命令：

```bash
OUT_DIR=/home/jia/wm_open-yolo/OpenYOLO3D/output/scannet200/subset_sweeps/even48_mask_graph_score_modes_full_core_strict_existing \
MASK_GRAPH_OUT=/home/jia/wm_open-yolo/OpenYOLO3D/output/mask_graph_proposals_scannet200_even48_full_core_strict_existing_gpu \
bash tools/run_scannet200_even48_mask_graph_score_modes.sh
```

关键过滤参数：

```text
MASK_GRAPH_GAP_SEED_POLICY=full_core
MASK_GRAPH_EXPORT_MAX_EXISTING_IOU=0.30
MASK_GRAPH_EXPORT_MAX_SEED_IN_EXISTING_MASK_RATIO=0.30
```

结果：

```text
统一分数，不加证据图：0.271195 / 0.345761 / 0.389522
统一分数，加入证据图：0.271195 / 0.345760 / 0.389522
候选自身分数，不加证据图：0.303107 / 0.396944 / 0.443739
候选自身分数，加入证据图：0.303101 / 0.396934 / 0.443723
场景内归一化分数，不加证据图：0.302818 / 0.396449 / 0.443777
场景内归一化分数，加入证据图：0.298840 / 0.392579 / 0.441919
```

候选统计：

```text
证据图导出候选：29
最终进入评估的多视角证据图候选：5
```

对这 5 个进入评估的证据图候选做诊断：

```text
完整漏检物体：1
背景污染或几何错误：2
多物体错误合并：2
平均最高真实三维交并比：0.140691
平均与 Mask3D 的候选覆盖重叠：0.069999
```

结论：

- 统一分数和候选自身分数下，完整核心版本基本不伤害当前主线。
- 场景内归一化分数下，旧缺口核心下降 `0.015369`，完整核心下降 `0.008670`，严格现有覆盖过滤后下降缩小到 `0.003978`。
- 这说明“完整核心 + 严格准入”方向是对的，但目前仍没有形成正收益。
- 现在剩下的主要错误不是残片，而是：
  - 多物体错误合并。
  - 类似 `mat`、`whiteboard` 这类二维掩码看似稳定但三维实例交并比很低的候选。
  - 少量真正完整漏检物体，例如 `scene0025_02` 的 `bottle`。

本次最终判断：

```text
证据图方向没有被否定。
但证据图候选不能再简单作为新增实例追加。
旧版缺口核心会产生残片；
完整核心能解决残片伤害；
严格现有覆盖过滤能减少重复和归一化分数下降；
下一步必须加入“新增、补全、替换、拒绝”的动作判断。
```

下一步建议：

1. 不要继续只调图候选分数。
2. 不要回到只输出缺口核心。
3. 保持当前“完整核心输出 + 同时保存缺口核心”。
4. 给证据图候选增加硬拒绝：
   - 多物体错误合并风险高则拒绝。
   - 图共识低且同物体强边少则拒绝。
   - 完整核心与多个三维区域相交时拒绝或拆分。
   - 类似 `mat`、`rug`、`whiteboard`、`poster` 等平面或大面积易污染类别需要单独限制。
5. 开始实现候选动作判断：
   - 完整且属于漏检物体：新增。
   - 只是已有物体局部缺口：补全已有候选，不新增。
   - 明显优于已有候选：替换。
   - 重复、背景污染、多物体合并：拒绝。
6. 下一轮实验建议先在 even48 上跑：
   - 完整核心严格过滤当前版本。
   - 再加“平面/多物体错误合并拒绝”版本。
   - 只有候选自身分数和归一化分数同时不下降，再扩到 even96。

本轮围绕“严格证据图、基线缺口检测、候选竞争”继续修改并做了 GPU 验证。核心结论是：

```text
证据图候选已经能生成，也能进入最终评估；
但当前它们还没有实质替换掉主线中的脏候选；
继续单纯抬高图候选分数或优先级没有收益。
```

本轮已经完成的代码方向：

- 证据图点级投票默认不再失败后恢复完整并集；如果多视角核心点不足，候选可以直接失败。
- 新增“基线未解释缺口”检测：候选先判断有多少三维种子点没有被已有三维候选覆盖，只有未覆盖点数量、比例和连通性达标时，才作为新增补漏候选。
- 新增 `graph_gap_seed_policy`：
  - `adaptive`：默认策略。新增补漏候选只输出未被已有三维候选覆盖的缺口核心点；已有候选支持项保留完整核心点用于诊断。
  - `full_core`：输出完整核心点。
  - `uncovered_core`：新增候选强制只输出缺口核心点。
- 新增证据图候选内部竞争：多个图候选争夺同一批三维点时，按新增性、已有覆盖比例、图共识、深度一致性、冲突比例、支持视角数和优先级排序，低质量重复候选会写入 `prefilter_skipped`。
- 新增图候选竞争优先级因子 `graph_competition_priority_factor`，它由图共识、深度一致、支持视角数、缺口比例、冲突比例和已有覆盖比例共同决定。
- 新增融合阶段图候选最终分数控制：
  - `backprojection_mask_graph_score_factor_weight`
  - `backprojection_mask_graph_max_proposal_score`
  这两个参数用于测试“图候选自身抬分”是否能改变最终排序。
- `tools/run_scannet200_even48_mask_graph_eval.sh` 已接入以上导出和评估参数，并允许通过环境变量控制。
- 为了能继续复现实验，补了两个环境兼容点：
  - `utils/utils_2d.py` 中兼容新版 Pillow 没有 `Image.LINEAR` 的问题。
  - `utils/__init__.py` 增加 `OPENYOLO3D_ALLOW_LEGACY_2D_CACHE=1` 显式开关。默认仍严格要求二维缓存有元数据签名；只有开这个开关时，才允许读取旧格式二维框缓存。
- 另外把部分硬编码 `.cuda()` 改成按设备选择，避免没有显卡时直接崩溃。但真实评估仍应使用 GPU，CPU 跑 even48 太慢。

本轮 GPU 验证过程：

- 沙盒默认环境里 `torch.cuda.is_available()` 为 `False`，设备数 `0`。
- 通过提权运行后，外层环境能看到 GPU：`torch.cuda.is_available()` 为 `True`，设备数 `1`。
- 后续 even48 实验均用提权方式在 GPU 上跑。

本轮关键 GPU 结果：

```text
GPU 可见性检查：
/home/jia/anaconda3/envs/openyolo3d/bin/python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.device_count())"
沙盒默认结果：False / 0
提权运行结果：True / 1

严格缺口核心 + 候选内部竞争：
导出图候选 53 个
最终进入评估的多视角图候选 16 个
运行命令：
OPENYOLO3D_ALLOW_LEGACY_2D_CACHE=1 \
OUT_DIR=/home/jia/wm_open-yolo/OpenYOLO3D/output/scannet200/subset_sweeps/even48_mask_graph_gap_compete_gpu \
PATH_TO_2D_PREDS=/home/jia/wm_open-yolo/OpenYOLO3D/output/scannet200/bboxes_2d \
MASK_GRAPH_OUT=/home/jia/wm_open-yolo/OpenYOLO3D/output/mask_graph_proposals_scannet200_even48_gap_compete_gpu \
MODE=graph_refill \
bash tools/run_scannet200_even48_mask_graph_eval.sh
结果：0.271186 / 0.345749 / 0.389509

给图候选加竞争优先级：
运行命令：
OPENYOLO3D_ALLOW_LEGACY_2D_CACHE=1 \
OUT_DIR=/home/jia/wm_open-yolo/OpenYOLO3D/output/scannet200/subset_sweeps/even48_mask_graph_gap_compete_priority_gpu \
PATH_TO_2D_PREDS=/home/jia/wm_open-yolo/OpenYOLO3D/output/scannet200/bboxes_2d \
MASK_GRAPH_OUT=/home/jia/wm_open-yolo/OpenYOLO3D/output/mask_graph_proposals_scannet200_even48_gap_compete_priority_gpu \
MODE=graph_refill \
bash tools/run_scannet200_even48_mask_graph_eval.sh
结果：0.271186 / 0.345749 / 0.389509

把图候选竞争因子乘到最终分数：
图候选分数因子确实写入报告，范围约 1.46 到 1.60
但多数候选分数被截断到 1.0
运行命令：
OPENYOLO3D_ALLOW_LEGACY_2D_CACHE=1 \
OUT_DIR=/home/jia/wm_open-yolo/OpenYOLO3D/output/scannet200/subset_sweeps/even48_mask_graph_gap_scorefactor_gpu \
PATH_TO_2D_PREDS=/home/jia/wm_open-yolo/OpenYOLO3D/output/scannet200/bboxes_2d \
MASK_GRAPH_OUT=/home/jia/wm_open-yolo/OpenYOLO3D/output/mask_graph_proposals_scannet200_even48_gap_compete_priority_gpu \
MODE=graph_refill \
bash tools/run_scannet200_even48_mask_graph_eval.sh
结果：0.271186 / 0.345749 / 0.389509

把图候选分数上限放宽到 1.05：
运行命令：
OPENYOLO3D_ALLOW_LEGACY_2D_CACHE=1 \
OUT_DIR=/home/jia/wm_open-yolo/OpenYOLO3D/output/scannet200/subset_sweeps/even48_mask_graph_gap_scorecap_gpu \
PATH_TO_2D_PREDS=/home/jia/wm_open-yolo/OpenYOLO3D/output/scannet200/bboxes_2d \
MASK_GRAPH_OUT=/home/jia/wm_open-yolo/OpenYOLO3D/output/mask_graph_proposals_scannet200_even48_gap_compete_priority_gpu \
MODE=graph_refill \
bash tools/run_scannet200_even48_mask_graph_eval.sh
结果：0.271186 / 0.345749 / 0.389509

打开图证据对主线候选的重排：
MASK_GRAPH_EVIDENCE_RESCORE=1
MASK_GRAPH_EVIDENCE_PRIORITY_WEIGHT=0.30
运行命令：
OPENYOLO3D_ALLOW_LEGACY_2D_CACHE=1 \
MASK_GRAPH_EVIDENCE_RESCORE=1 \
MASK_GRAPH_EVIDENCE_PRIORITY_WEIGHT=0.30 \
MASK_GRAPH_SCORE_FACTOR_WEIGHT=0.0 \
MASK_GRAPH_MAX_PROPOSAL_SCORE=1.0 \
OUT_DIR=/home/jia/wm_open-yolo/OpenYOLO3D/output/scannet200/subset_sweeps/even48_mask_graph_evidence_rerank_gpu \
PATH_TO_2D_PREDS=/home/jia/wm_open-yolo/OpenYOLO3D/output/scannet200/bboxes_2d \
MASK_GRAPH_OUT=/home/jia/wm_open-yolo/OpenYOLO3D/output/mask_graph_proposals_scannet200_even48_gap_compete_priority_gpu \
MODE=graph_refill \
bash tools/run_scannet200_even48_mask_graph_eval.sh
结果：0.271186 / 0.345749 / 0.389509
```

本轮结论：

- 图候选不是完全没进最终预测；它们已经有 `16` 个多视角候选进入评估。
- 图候选自身抬分、放宽图候选分数上限、图证据轻量重排主线候选，都没有改变 even48 指标。
- 这说明当前瓶颈不是“图候选优先级不够高”，而是“图证据还没有强到能明确替换或剔除主线候选”。
- 下一步不应继续调图候选分数，也不应继续只做软重排。
- 下一步应做更硬的动作：
  - 用图证据显式标记主线候选是否被多视角二维掩码支持。
  - 对低图证据、高冲突、低缺口价值的主线新增候选做降权或剔除。
  - 当图候选和主线候选高度重叠时，不是并列输出，而是执行“保留主线、替换为图候选、或两者都拒绝”的明确决策。
  - 对图候选重点分析真阳性和假阳性，而不是继续全局调权重。

当前最重要结论：

- 当前最稳结果仍是候选补全主线，不是 Alpha-CLIP 或局部超点。
- 当前最好四十八场景划分：平均精度 `0.272610`，百分之五十重叠率平均精度 `0.345769`，百分之二十五重叠率平均精度 `0.389491`。
- 当前最好九十六场景划分：平均精度 `0.273030`，百分之五十重叠率平均精度 `0.340244`，百分之二十五重叠率平均精度 `0.383687`。
- 本轮新跑的多关系证据图版本在四十八场景划分上达到 `0.273894 / 0.345057 / 0.388778`，比当前最好四十八场景版本略高一点；但九十六场景划分是 `0.271398 / 0.338013 / 0.381188`，比当前最好九十六场景版本略低。
- 进一步核对已有结果后要补充一句：四十八场景里证据图候选不是完全没用，确实出现过正向结果；例如旧的单视角放回版本达到 `0.274624 / 0.345785 / 0.389503`。但对应九十六场景版本只有 `0.271843 / 0.338956 / 0.382132`，说明它不是稳定提升，而是对子集敏感。
- 本轮新做的“证据图候选只作为少量补充”实验没有提升：
  - 四十八场景，主线候选加证据图补充，最多额外放两个证据图多视角候选：`0.271201 / 0.345771 / 0.389494`。
  - 九十六场景，完整九十六图候选补充：`0.272253 / 0.340308 / 0.383686`。
  - 四十八场景，加入更严格的证据图准入门槛后，实际只进入一个证据图候选，结果仍是 `0.271201 / 0.345771 / 0.389494`。
- 新一轮代码已经把证据图候选的默认策略改严格了：
  - 证据点投票不再自动回退到完整并集。
  - 图候选默认排序改成普通优先级，不再把图来源天然排前。
  - 增加了组内一致性门槛和冲突拒绝门槛，可通过命令行显式打开更严格的过滤。
  - even48 图实验脚本也同步成了同一套严格参数。
- 最新代码继续补上了“基线未解释缺口”判断：
  - 证据图候选会先检查有多少三维种子点没有被已有三维候选覆盖。
  - 如果未覆盖点太少、未覆盖比例太低，或未覆盖区域不能形成稳定连通块，则默认只作为“已有候选支持证据”，不再作为新增候选输出。
  - 新增补漏候选默认使用自适应种子策略：真正新增候选只保存未被已有三维候选覆盖的缺口核心点；如果显式输出已有候选支持项，则仍保留完整核心点用于诊断。
  - 这一步是为了避免以前的问题：虽然用缺口判断通过了候选，但最终保存的三维点仍然包含大量已有区域，导致重复候选和污染候选进入后续融合。
- 最新代码还加入了证据图候选内部竞争：
  - 多个证据图候选如果争夺同一批三维种子点，会先按新增性、已有覆盖比例、图共识、深度一致性、冲突比例、支持视角数和候选优先级排序。
  - 质量较低的重复候选或冲突候选会在导出阶段被过滤，并写入 `prefilter_skipped`。
  - 这一步吸收的是“父子选择、候选竞争”的思想，但仍保持轻量证据图结构，不把整体结构改成完整实例树。
- 最新 GPU 结果又补了一条关键结论：
  - 证据图候选导出共保留了 `53` 个候选，其中最终进入评估并真正起作用的多视角图候选有 `16` 个。
  - 给证据图候选加上额外竞争优先级、再把图候选的分数上限从 `1.0` 放宽到 `1.05` 后，最终结果仍然是 `0.271186 / 0.345749 / 0.389509`，和上一版完全一样。
  - 这说明当前问题不再是“图候选生成失败”，而是“图候选还没有实质改变主线候选的排序与替换”。
  - 下一步应把重点放到“图证据重排主线候选”或“图证据直接给主线候选加分/降分”，而不是继续抬高图候选自身分数。
- 本轮新做的“证据图只支持主线候选排序、不直接新增证据图候选”实验也没有提升：
  - 四十八场景：`0.271199 / 0.345768 / 0.389489`。
  - 实际进入最终预测的候选来源为：二维掩码融合候选 `144` 个，二维框反投影候选 `130` 个，证据图候选 `0` 个。
  - 其中 `124 / 274` 个已进入候选能被证据图候选支持，说明证据图不是完全没覆盖主线；但简单按三维种子重叠给排序加分没有带来收益，说明当前证据图支持分还不能可靠地区分真候选和假候选。
- 因此现在的准确结论是：证据图“提前处理候选”的方向是对的，但当前实现生成的三维候选平均质量还不够高；它不能直接替代当前主线，也不能简单作为额外候选塞进最终预测。下一步更合理的是把证据图先用作候选过滤、候选重排和缺口判断，而不是继续增加最终候选数量。
- ScanNet200 当前评估是“前若干候选”的开放评测，不是“只取第一类”的硬单分类。
- 现在瓶颈不是“候选数量不够”，而是“新增候选质量不够高”：背景、平面、碎片、错误类别较多。
- 跨视角二维掩码证据图初版代码已经接入，并已升级为多关系证据图：同物体支持关系、父子包含关系、冲突关系、弱关系分开保存。只用强同物体关系分组，父子、冲突、弱关系只参与诊断和排序。
- 当前多关系证据图还不是默认最佳路线。旧导出目录上的结果显示：加入单视角孤立候选后，四十八场景划分略有正向，但九十六场景划分没有保持。并且旧目录缺少候选级来源字段，不能继续作为当前代码的最终结果。
- 这轮新导出的真实结果说明：多关系证据图本身是可用的，但当前默认阈值仍偏松，四十八场景略涨、九十六场景略掉，说明还需要继续压单视角噪声或提高冲突边的抑制力度。
- 证据图候选里一个明显风险是“同物体支持”和“类别冲突”同时存在。四十八当前格式目录中，多视角图候选的冲突边中位数约为 `2`，九十六当前格式目录中多视角图候选的冲突边中位数也约为 `2`、平均约为 `7.5`。这解释了为什么看起来已经做了跨视角聚合，但最终指标没有稳定增加：图里确实有支持证据，但很多候选也同时被其他类别或背景证据污染。
- 评估脚本已加入旧导出复用检查：只有参数和候选字段都符合当前代码格式时才复用旧导出，否则会重新导出。后续重跑真实评估时，建议使用新的输出目录，避免混入旧格式候选。
- 后续不要优先继续调连通分量半径、重叠删除阈值、局部超点阈值或 Alpha-CLIP 固定阈值；这些已经验证为收益很小或不稳定。

当前最好配置包含：

```text
YOLO-World 二维检测
-> Segment Anything 模型二维掩码候选
-> 二维框反投影候选
-> 二维掩码级超点正负约束
-> 多视角一致性过滤
-> 三维连通分量清理
-> 来源优先级和候选数控制
-> 屏蔽 rug 类别
```

接下来最建议做的方向，按优先级排列：

1. **把跨视角二维掩码证据图改成候选过滤和重排信号**

   这是当前最新结论下最建议的新实验。不要再优先把证据图候选直接追加到最终预测列表里。原因是：四十八场景可能涨，但九十六场景不稳；严格过滤后图候选几乎进不来，放宽后又容易引入假阳性。

   新流程应改成：

   ```text
   当前主线候选
   -> 查每个主线候选是否被多个二维掩码观测支持
   -> 计算证据图支持分、冲突分、类别一致性、深度可见性
   -> 对已有候选重排或降权
   -> 只在主线没有覆盖的新区域里少量补证据图候选
   ```

   这比“证据图候选直接加入最终预测”更稳，因为它先保护当前最好主线，再用证据图判断哪些已有候选可信、哪些新增候选确实填补缺口。

   具体下一步可做：

   - 给 `utils/backprojection_fusion.py` 增加“证据图支持已有候选”的打分，而不是只读取证据图候选本身。
   - 对主线候选保存证据图支持数、冲突数、支持视角数。
   - 只对低证据、高冲突的新增候选降权或跳过。
   - 对证据图候选继续保持单独来源，不再默认提高来源优先级。
   - 新增候选导出阶段已经开始做缺口检测；后续应继续把“缺口核心”和“完整核心”分开管理，并测试缺口核心输出是否能减少重复和背景污染。
   - 证据图候选内部竞争已经接入；后续还要做的是让证据图候选和主线候选之间也发生竞争，而不是只在证据图内部竞争。
   - 证据图候选自身抬分已经验证无效；下一步应切到“证据图支持主线候选”的重排模式，先让图证据影响主线候选的最终排序，再看能不能推高 AP。

2. **跨视角二维掩码证据图，加聚类后再反投影**

   这是当前最推荐的新实验。最近核对多篇跨视角二维掩码聚合论文后，结论是：当前瓶颈不是二维检测器，也不是候选生成后的阈值清理，而是单帧二维掩码太早被反投影成三维候选。

   当前代码已从单一关系证据图升级为轻量多关系证据图，不是完整复现某一篇论文。具体做法是：

   ```text
   二维掩码观测列表：每个观测对应一帧里的一个二维框和一个二维掩码
   图节点：一个二维掩码观测
   图边：两个不同视角的二维掩码之间的多种关系
   邻接表：保存每个节点连接到哪些其他节点，以及对应边特征
   掩码簇：图连通分量，表示互相有证据支持的一组二维掩码
   三维候选：从一个掩码簇里选择互补视角，合并这些视角的三维种子点得到
   ```

   当前边类型包括：

   - 同物体支持边：两个不同视角的掩码有足够三维种子点重叠、共同粗三维参考或空间一致性，只允许这类强边参与分组。
   - 父子包含边：一个大掩码包含一个小掩码，只记录父子方向和包含分数，不直接把两个节点合并。
   - 冲突边：不同类别掩码共享相同粗三维参考或空间证据冲突，不参与分组，只作为风险信号。
   - 弱关系边：证据不够强，只保存为辅助诊断，不允许单独桥接两个候选。

   当前图边使用的证据包括：

   - 三维种子点重叠率。
   - 三维种子点包含率。
   - 类别兼容性，默认只连接同类别掩码。
   - 粗三维参考一致性，即两个掩码是否主要落在同一个已有三维实例候选上。
   - 深度支持比例，来自项目已有投影可见性掩码；这个可见性掩码已经用真实深度图检查过点深度是否一致。
   - 空间一致性，仍保留两个三维种子点中心距离作为补充。
   - 视角共识分数，即是否有多个视角共同支持同一个粗三维参考。

   当前合并规则：

   ```text
   只用强同物体支持边做图连通分量
   父子包含边不直接合并
   弱关系边不桥接
   冲突边不桥接，并降低候选优先级
   ```

   当前输出候选来源已拆成两类：

   ```text
   mask_graph_multi_view：多视角稳定掩码簇
   mask_graph_single_view：单视角孤立候选
   ```

   融合评估时默认更优先保留多视角候选，单视角候选严格限量，二维框反投影候选保留少量用于补召回。

   当前实现和 Clutt3R-Seg 的关系：

   - 相同点：都不直接相信单个噪声二维掩码，而是把二维掩码当作跨视角证据。
   - 不同点：Clutt3R-Seg 更像层级实例树，重点是父子掩码、包含关系和层级选择；当前实现是普通加权无向图，重点是跨视角二维掩码之间是否互相支持。
   - 当前实现更接近 MaskClustering、Any3DIS、MV3DIS 一类“跨视角掩码关联和三维引导匹配”的轻量落地，同时借用了 Clutt3R-Seg“用噪声掩码做证据，而不是逐个修噪声掩码”的思想。

   超点在当前策略里的位置：

   ```text
   掩码证据图负责：候选生成之前，判断哪些二维掩码和哪些视角更可信
   超点负责：候选生成之后，做几何边界约束、扩张限制和连通分量清理
   ```

   因此当前不是放弃超点，而是先解决“单帧二维掩码太早变成三维候选”的问题，再让超点继续作为几何收边和后续二维实例边界感知模块。

   新流程应改成：

   ```text
   YOLO-World 二维检测
   -> Segment Anything 二维掩码观测
   -> 构建跨视角二维掩码证据图
   -> 用三维重叠、深度一致性、类别兼容、视角共识聚类
   -> 只让稳定掩码簇生成三维候选
   -> 再接当前已有超点细化和连通分量清理
   ```

   目标：在候选生成之前判断“这是不是一个跨视角稳定物体”，少生成背景、平面、碎片和单帧偶然掩码。

3. **轻量三维引导匹配：用粗三维参考引导二维掩码匹配**

   使用已有三维实例候选、连通分量、或超点连通区域作为粗三维参考。先把粗三维参考投影到多个视角，再反向选择最一致的二维掩码；不要让每个二维掩码自由反投影。

   初版不需要复现完整论文，只需要在候选导出阶段新增以下分数：

   - 二维掩码与粗三维参考的投影覆盖率。
   - 二维掩码反投影种子与三维参考的三维重叠率和覆盖率。
   - 深度一致性权重，用于降低遮挡和错投影视角影响。
   - 三维参考在多个视角中的稳定支持数。

4. **轻量视角共识聚类：全局视角共识率**

   对两个二维掩码，不只看它们自己的种子重叠率，而是看其他视角是否共同支持它们属于同一三维物体。初版可以做无学习图聚类：

   - 节点：单帧二维掩码观测。
   - 边：三维种子重叠率、包含率、类别兼容、深度一致性、共同三维参考支持。
   - 聚类：先用阈值图连通分量或并查集，不急着上复杂谱聚类。
   - 输出：每个掩码簇一个候选，并保存掩码簇编号、图共识分数、选中视角数、深度一致性分数等诊断字段。

5. **轻量视角集合选择**

   之前已经验证“最佳单视角”、“固定前几个视角”、“简单视角质量门控”都不稳。下一步如果做视角选择，应在掩码簇内选择一组“互补且一致”的视角，而不是固定只选一个或两个视角。

   初版用贪心替代动态规划即可：

   ```text
   视角集合得分 =
     质量分
     + 新增三维覆盖
     + 深度一致性
     + 类别一致性
     - 与已选视角的冗余
     - 大平面或背景风险
   ```

6. **外观特征辅助身份一致性**

   不作为第一步。等几何证据图跑通后，再给每个二维掩码裁剪区域提取外观特征，加入图边权。它适合解决“类别不冲突但几何种子脏、不同视角掩码身份不一致”的问题。

7. **超点优先和边界感知超点后移**

   超点方向仍有价值，但不再作为当前第一步。之前局部超点初版和全场景超点替换都没有形成稳定收益。若继续，应借二维实例边界感知思路，让二维实例边界参与超点选择或拆分，而不是只靠三维坐标、近邻图和固定阈值。

8. **语义校准模型和新二维来源暂后置**

   区域感知图文模型继续定位为后续语义模块，等候选几何质量提升后再做最终分类或低可信过滤。新的文本引导二维分割来源可以后续接入，但不应优先于掩码证据图；如果二维掩码仍然单帧自由反投影，换来源大概率仍会遇到同类几何污染。

不建议优先继续的方向：

- 继续手调三维连通分量半径。
- 继续手调候选包含、重叠删除规则。
- 继续做 Segment Anything 模型多掩码手写选择。
- 继续做局部超点小范围阈值搜索。
- 继续用 Alpha-CLIP 固定阈值做低置信度后置改类。
- 继续直接替换全场景超点。

给下一次实现的建议入口：

- 当前已有 `tools/export_mask_graph_proposals.py` 和 `tools/run_scannet200_even48_mask_graph_eval.sh`。
- 本轮代码新增和确认的内容：
  - 候选生成加入点级投票，多个视角支持的三维点权重更高，深度支持比例高的视角权重更高。
  - 候选输出应保存候选来源：多视角稳定候选和单视角孤立候选分开，方便后续来源配额和优先级生效。
  - 评估脚本不再盲目复用旧导出，会检查点级投票参数和候选字段。
  - 最终融合阶段新增证据图候选专用准入门槛：最少观测数、最少选中视角数、最少同物体强边、平均边分数、图共识分数、深度一致性、最大冲突边数、最大冲突比例。默认关闭，实验时可打开。
  - 最终融合阶段新增证据图支持主线候选的打分：用同类别证据图候选和主线候选的三维种子重叠，记录支持分、支持数量、最佳重叠率、最佳三维交并比。默认关闭，实验时可打开。
  - 当前执行环境已完成真实显卡导出和评估验证，不再只是合成检查。
- 下一步不是继续简单放宽单视角孤立候选，也不是继续把证据图候选硬加到最终预测，而是做证据图对当前主线候选的过滤和重排：
  - 多视角掩码簇和单视角孤立候选分开统计。
  - 对单视角孤立候选单独限制数量、分数或风险特征。
  - 用图边数量、图共识分数、选中视角数、种子落入已有三维掩码的比例做候选排序和分层数量上限。
  - 用证据图支持度给已有候选加分，用冲突边和弱边给新增候选降权。
  - 只在当前主线没有覆盖的新区域里补极少量图候选。
  - 注意：简单“有证据图重叠就加分”已经验证无效。下一版应加入负证据，也就是候选被多少冲突类别、弱关系、父子包含关系同时支持；只看正重叠会把假候选也一起加分。
  - 目标是在保留四十八场景划分正向信号的同时，把九十六场景划分拉回或至少守住当前最好值。
- 当前最佳结果仍使用原“二维掩码融合候选 + 二维框反投影候选”配置；掩码证据图暂作为实验分支，不作为默认最佳。

## 0.1 GitHub 同步和换设备操作指南

当前仓库地址：

```text
git@github.com:1108-WM/change-open-yolo.git
```

当前本地远程仓库配置：

```text
origin   git@github.com:1108-WM/change-open-yolo.git
upstream https://github.com/aminebdj/OpenYOLO3D.git
```

这台电脑是怎么连接到 GitHub 的：

- 使用的是用户自己的 GitHub 账号 `1108-WM`，不是 Codex 的账号。
- 连接方式是 SSH，不依赖 `gh auth login`。
- `gh auth login` 当时走浏览器登录时出现过 `Post "https://github.com/login/device/code": EOF`，这是 GitHub CLI 访问登录接口失败，不代表 SSH 不能用。
- 后来执行 `ssh -T git@github.com`，返回：

```text
Hi 1108-WM! You've successfully authenticated, but GitHub does not provide shell access.
```

这说明当前电脑的 SSH 公钥已经添加到 GitHub 账号 `1108-WM`，所以可以直接用 SSH 地址推送。

当前同步规则：

- 上传代码和文档。
- 不上传数据集、模型权重、实验输出、大型缓存和编译产物。
- 当前仍有一些 `models/Mask3D` 和 `pointnet2` 编译产物处于本地修改状态，但不要提交它们。
- 每次提交前先看 `git status --short`，只 `git add` 需要同步的源码或文档文件。

在另一台新设备上快速连接 GitHub：

1. 安装 Git。

```bash
git --version
```

如果没有安装，可以在 Ubuntu 上执行：

```bash
sudo apt update
sudo apt install git
```

2. 配置 Git 用户信息。

```bash
git config --global user.name "1108-WM"
git config --global user.email "你的 GitHub 邮箱"
```

3. 检查是否已有 SSH key。

```bash
ls ~/.ssh
```

如果里面已经有 `id_ed25519.pub`，可以直接看第 5 步。没有的话生成一个新的。

4. 生成 SSH key。

```bash
ssh-keygen -t ed25519 -C "你的 GitHub 邮箱"
```

一路回车即可。生成后会得到：

```text
~/.ssh/id_ed25519
~/.ssh/id_ed25519.pub
```

5. 把公钥添加到 GitHub。

查看公钥：

```bash
cat ~/.ssh/id_ed25519.pub
```

复制整行内容，打开 GitHub：

```text
GitHub -> Settings -> SSH and GPG keys -> New SSH key
```

把公钥粘贴进去保存。

6. 测试 SSH 是否连通。

```bash
ssh -T git@github.com
```

第一次会问：

```text
Are you sure you want to continue connecting?
```

输入：

```text
yes
```

如果看到类似下面内容，说明配置成功：

```text
Hi 1108-WM! You've successfully authenticated, but GitHub does not provide shell access.
```

7. 克隆当前项目。

```bash
git clone git@github.com:1108-WM/change-open-yolo.git
cd change-open-yolo
```

8. 以后在另一台设备上同步修改。

拉取最新代码：

```bash
git pull origin main
```

查看改动：

```bash
git status --short
```

提交指定文件：

```bash
git add 文件路径
git commit -m "提交说明"
git push origin main
```

9. 如果另一台设备已经有项目目录，但 remote 不对。

查看 remote：

```bash
git remote -v
```

设置为当前仓库：

```bash
git remote set-url origin git@github.com:1108-WM/change-open-yolo.git
```

如果没有 `origin`，新增：

```bash
git remote add origin git@github.com:1108-WM/change-open-yolo.git
```

10. 关于 `gh` 命令。

`gh` 是 GitHub CLI，不是必须的。当前项目已经可以通过 SSH 正常推送，所以新设备优先配置 SSH 即可。

如果确实要安装 `gh`，Ubuntu 上可以用：

```bash
sudo apt install gh
```

或者：

```bash
sudo snap install gh --classic
```

之前 `snap install gh` 提示需要 `--classic`，是因为这个 snap 包需要经典模式权限，不是项目本身的问题。

## 1. 当前目标

当前只推进第一个创新点：**基于 YOLO-World 的三维候选补全和几何质量提升**。

第一点当前主线是：

```text
YOLO-World 二维检测
-> Segment Anything 模型二维掩码候选
-> 二维框反投影候选
-> 超点细化
-> 多视角一致性过滤
-> 三维连通分量清理
-> 新增候选并评估
```

第二点目前不作为主线，只做过一次叠加验证：

- Alpha-CLIP：带透明度/区域注意能力的图文模型，用于后续低置信度语义复核。
- 多模态大语言模型：用于图像、文本、候选目标的语义判断。

YOLOE 已经从当前活跃主线中放弃，不再作为主要模块推进。

### 1.1 2026-06-06 本轮会话快速结论

本轮主要围绕第一点继续修补：候选补全不是继续增加候选数量，而是减少二维到三维反投影后的背景污染。

已经完成：

- 给候选诊断补充真实三维几何特征，并训练轻量几何质量判别器。
- 随机森林判别器在 `even96` 实际加入候选上的验证集曲线面积从 `0.6813` 提升到 `0.7841`。
- 实现并验证 Segment Anything 多掩码手写几何选择。
- 实现并验证 Segment Anything 多掩码学习式选择。
- 两种多掩码选择都没有通过 `even48`，因此都没有进入 `even96`。
- 实现 ScanNet 原始深度一致性诊断特征，并接入轻量判别器。
- 实现并验证自适应内部种子：按二维掩码边界距离和 ScanNet 深度主层一致性筛选反投影种子。

关键结果：

- 当前最佳仍是二维掩码级超点负约束，`even48` 平均精度 `0.272610`，`even96` 平均精度 `0.273030`。
- 手写多掩码几何选择 `even48` 平均精度 `0.272135`，低于当前参考。
- 学习式多掩码选择 `even48` 平均精度 `0.270682`，低于当前参考。
- 核心深度一致性特征让判别器验证集平均精度从 `0.1738` 提升到 `0.1933`，但全召回保留候选数变差。
- 自适应内部种子 `even48` 两档都低于当前参考：`k070` 平均精度 `0.269600`，`k090` 平均精度 `0.271736`，不进入 `even96`。
- 视角级质量门控 `even48` 两档都没有明确提升：保守档完全持平，较强档平均精度仅 `+0.000403`，不进入 `even96`。

本轮判断：

- 真实几何特征有诊断和排序价值，但不能直接作为全局硬过滤器。
- ScanNet 原始深度特征也有排序价值，但同样不能直接硬过滤。
- 候选级判别器不能直接迁移到同一个二维框内部的多掩码选择。
- 全局式内部种子裁剪会伤到真实物体边界或可用局部，不适合作为当前主线。
- 当前这种简单视角质量门控太弱或不够准，不适合作为当前主线。
- 下一步更值得做的是候选局部超点、候选包含关系处理，而不是继续堆新的二维大模型。

### 1.2 本次会话决策链

本次会话的推进顺序和最终判断：

1. 手写 Segment Anything 多掩码几何选择依赖人工权重和几何规则，能够改变掩码选择，但 `even48` 指标下降，因此停止继续调手写阈值。
2. 按“参数自动学习”的目标训练轻量几何判别器。真实三维几何和 ScanNet 原始深度特征显著增强了排序能力，但正样本少，全召回硬过滤效果不稳。
3. 把候选级判别器用于同框多掩码选择后出现分布偏移，少了真正补全目标，因此不能直接部署；若重做，必须训练同框组内排序模型。
4. 将深度和边界信号用于自适应内部种子，两档 `even48` 都下降，说明全局裁剪种子会误删有用物体部分。
5. 将几何信号用于视角级质量门控，保守档持平，较强档仅有 `+0.000403` 平均精度，且坏几何没有减少，因此不进入 `even96`。

最终决策：

- 不放弃轻量几何判别器，但只把它作为诊断、排序和降权工具，不作为全局硬删除器。
- 暂停手写多掩码选择、候选级学习式多掩码选择、全局自适应内部种子和简单视角质量门控。
- 当前最佳配置不变，下一轮优先实现候选局部超点；第二优先级是候选包含和重叠关系处理。

### 1.3 2026-06-08：Alpha-CLIP 叠加当前最佳的 even48 验证

本轮确认：

- 当前最佳第一点配置此前没有使用 Alpha-CLIP，只包含候选补全、二维掩码级超点负约束、多视角一致性和连通分量清理。
- ScanNet200 数据盘重新挂载后实际路径是 `/media/jia/软件1/OpenYOLO3D_datasets/scannet200`，已把仓库内 `data/scannet200` 软链接改到该路径。
- 为了让 Alpha-CLIP 导出和评估严格复用当前最佳候选逻辑，已给 `tools/export_multiview_object_clip_features.py` 和 `tools/evaluate_multiview_object_clip_correction.py` 补齐当前最佳反投影参数，包括连通分量清理、来源优先级、来源候选数上限和二维掩码级超点正负约束。

Alpha-CLIP 导出：

- 场景：`even48`
- 输出：`output/multiview_object_alphaclip_scannet200_even48_current_best_low055`
- 视觉编码器：`alpha_clip`
- 策略：只重打低置信度目标，`rescore_policy low_score`，`rescore_max_base_score 0.55`，`top_views 3`
- 导出记录：`23752` 条 object-level Alpha-CLIP 记录

同一 correction 脚本内公平对比：

| 评估模式 | 配置 | 平均精度 | 百分之五十重叠率平均精度 | 百分之二十五重叠率平均精度 | Alpha-CLIP 修正数 |
|---|---:|---:|---:|---:|---:|
| `baseline` 分数模式 | no-op，无类别修正 | `0.272650` | `0.356852` | `0.406168` | `0` |
| `baseline` 分数模式 | Alpha-CLIP `0.60/0.10/0.10` | `0.273831` | `0.359135` | `0.410241` | `2269` |
| `openyolo` 分数模式 | no-op，无类别修正 | `0.292120` | `0.380559` | `0.424764` | `0` |
| `openyolo` 分数模式 | Alpha-CLIP `0.60/0.10/0.10` | `0.292371` | `0.381033` | `0.426491` | `2269` |

同口径变化：

- `baseline` 分数模式：平均精度 `+0.001181`，百分之五十重叠率平均精度 `+0.002283`，百分之二十五重叠率平均精度 `+0.004072`。
- `openyolo` 分数模式：平均精度 `+0.000251`，百分之五十重叠率平均精度 `+0.000474`，百分之二十五重叠率平均精度 `+0.001727`。

注意：

- 不能把 correction 脚本的绝对值直接和状态文件里的当前最佳 `0.272610 / 0.345769 / 0.389491` 混比，因为 correction 脚本的分数排序模式会显著改变绝对 AP50/AP25。
- Alpha-CLIP 确实带来正向信号，但 `2269` 次类别修正换来的收益很小，且 `openyolo` 分数模式下平均精度只提升 `+0.000251`。
- 这不是足够明确的 `even48` 强正向结果，暂不进入 `even96`。
- 这次实验不能代表 `Details Matter` 式 Alpha-CLIP，因为当前只是低置信度目标的后置类别修正，不是完整的 Alpha-CLIP final classifier。

与 `Details Matter` 思路的关键差别：

- `Details Matter` 把 Alpha-CLIP 作为 class-agnostic proposal 的主分类器；我们这次保留 YOLO-World/OpenYOLO3D 原类别，只在低置信度目标上尝试改类别。
- `Details Matter` 使用多视角、多尺度 Alpha-CLIP 相似度并做 standardized maximum similarity filtering；我们这次只导出 softmax 后的 `clip_probs`，再用固定阈值 `0.60/0.10/0.10` 做替换。
- `Details Matter` 的 proposal pipeline 还包含二维重叠去除、跟踪式聚合、迭代合并/删除和 refinement；我们当前候选里仍有较多背景或坏几何，Alpha-CLIP 只能改类别，不能修 mask。

重新判断：

- 不应因为这次 late correction 小增益就放弃 Alpha-CLIP。
- YOLO-World 仍然有用，职责是开放词表二维发现、引导 Segment Anything 模型和提供候选召回；Alpha-CLIP 更适合作为最终语义分类和语义置信度过滤。
- 更合理的第二点表述是：YOLO-World 负责候选召回，Alpha-CLIP 负责语义校准，而不是 Alpha-CLIP 取代 YOLO-World。

下一步建议：

1. 当前最佳：YOLO-World 类别作为最终类别。
2. Alpha-CLIP final classifier：YOLO-World 仍生成候选，Alpha-CLIP 对所有新增候选重新分类。
3. Alpha-CLIP + YOLO prior + standardized maximum similarity filtering：Alpha-CLIP 为主分数，YOLO-World 类别作为先验或平局处理，低可信候选用标准化相似度过滤。

优先实现：

- 导出所有候选的 Alpha-CLIP raw logits 或 raw cosine similarity，不只导出低分目标，也不只保存 softmax 概率。
- 保留 square crop、alpha mask、多视角可见性加权和多尺度聚合接口。
- 实现 standardized maximum similarity score，并在 `even48` 上先比较当前最佳、全量 Alpha-CLIP 分类、Alpha-CLIP 加标准化相似度过滤。

### 1.4 2026-06-12：Alpha-CLIP final classifier 复现式验证

已完成代码：

- `tools/export_multiview_object_clip_features.py`
  - 支持 `--square_crops`。
  - 保存 `clip_similarities`、`clip_similarity_topk`、`clip_logits`、`clip_logit_topk`。
  - 保留原 `clip_probs` 和 `clip_topk`，兼容旧 correction 实验。
- `tools/evaluate_multiview_object_clip_correction.py`
  - 新增 `--clip_application_mode final_classifier`。
  - 支持用 `probs/logits/similarities` 做决策。
  - 支持只作用于指定来源候选、YOLO 原类别先验、分数替换/融合、标准化最大相似度筛选。

验证：

- 语法检查通过：
  - `/home/jia/anaconda3/envs/openyolo3d/bin/python -m py_compile tools/export_multiview_object_clip_features.py tools/evaluate_multiview_object_clip_correction.py`
- 单场景冒烟通过：
  - 输出：`output/multiview_object_alphaclip_scannet200_smoke_full_raw_square`
  - `scene0011_00` 导出 `604` 条记录、`1696` 个视角裁剪。
- `even48` 全量 raw 特征导出完成：
  - 输出：`output/multiview_object_alphaclip_scannet200_even48_current_best_full_raw_square`
  - 导出 `27446` 条记录、`80769` 个视角裁剪。
  - 来源：`mask3d` `27173` 条，`sam_fused` `143` 条，`bpr` `130` 条。

同一 correction 脚本、同一全量 raw 缓存内公平对比：

| 配置 | 平均精度 | 百分之五十重叠率平均精度 | 百分之二十五重叠率平均精度 | 改类数 | 过滤数 |
|---|---:|---:|---:|---:|---:|
| no-op 对照 | `0.272650` | `0.356852` | `0.406168` | `0` | `0` |
| Alpha-CLIP 全部候选最终分类 | `0.213450` | `0.273744` | `0.304792` | `26053` | `0` |
| 只重分类新增候选 | `0.265189` | `0.344275` | `0.393248` | `211` | `0` |
| 新增候选重分类，YOLO 先验 `0.02` | `0.271890` | `0.355085` | `0.404463` | `100` | `0` |
| 新增候选重分类，YOLO 先验 `0.05` | `0.272581` | `0.356715` | `0.406065` | `17` | `0` |
| 新增候选只做标准化相似度筛选，阈值 `0.0` | `0.270921` | `0.354572` | `0.403844` | `0` | `140` |
| 新增候选只做标准化相似度筛选，阈值 `-0.5` | `0.271700` | `0.355386` | `0.404736` | `0` | `95` |

结论：

- Alpha-CLIP 直接作为全量最终分类器严重破坏原始 `mask3d` 类别，不能用。
- 只作用于新增候选后仍下降，说明当前新增候选的 Alpha-CLIP 零样本类别分数不够可靠。
- 加 YOLO 原类别先验能明显减少误改类，但最好的 `0.05` 仍略低于 no-op，对当前基线没有净收益。
- 标准化最大相似度筛选会误删有用新增候选，`0.0` 和 `-0.5` 都低于 no-op。
- 本轮 Alpha-CLIP final classifier 没有通过 `even48`，不进入 `even96`。
- 当前不应继续手调 Alpha-CLIP 阈值。除非后续重做更接近 `Details Matter` 的候选生成、跟踪聚合、合并删除和掩码 refinement，否则 Alpha-CLIP 暂只保留为语义诊断信号。

下一步判断：

- 当前主线回到三维候选质量本身：优先做候选局部超点，第二优先做候选包含/重叠关系处理。
- 轻量几何判别器仍作为诊断、排序和降权工具，不做全局硬删除。
- Alpha-CLIP 不再作为当前最近一步的主要提分模块。

### 1.5 2026-06-12：自生成几何超点替换实验

动机：

- 当前候选补全已经使用 ScanNet200 processed `.npy` 第 `9` 列自带超点。
- `Details Matter` 强调以三维超点为基本单位，并用可见比例、二维掩码支持比例和多视角一致性过滤超点。
- `OV3D-CG` 明确使用基于法向的图割先生成三维超点，再用二维 Segment Anything 掩码指导超点合并。
- 因此先测试“自生成超点能否优于数据预处理自带超点”。

已完成代码：

- 新增 `tools/generate_geometric_superpoints.py`：
  - 读取 ScanNet200 processed `.npy`。
  - 用点坐标、颜色、法向构建 k 近邻图。
  - 用 Felzenszwalb 风格图合并生成几何超点。
  - 将新超点编号写回第 `9` 列，其他列保持不变。
- 修改 `run_evaluation.py`：
  - 新增 `--processed_scene_root`。
  - 可在不移动 RGB-D 数据和三维 mask 的情况下替换 processed `.npy`，专门用于超点替换评估。

生成配置：

- 输出：`output/geometric_superpoints_scannet200_even48_k025`
- 场景：`even48`
- 参数：`knn=10`，`merge_k=0.25`，`min_size=20`，`spatial_weight=0.15`，`normal_weight=1.0`，`color_weight=0.25`
- 统计：新超点数量通常为原始超点的约 `1.5` 到 `2.0` 倍，中位超点大小从约 `70` 点降到约 `55-60` 点。

同一 `run_evaluation.py`、同一当前最佳参数下公平对比：

| 超点来源 | 平均精度 | 百分之五十重叠率平均精度 | 百分之二十五重叠率平均精度 |
|---|---:|---:|---:|
| 原始 ScanNet200 超点 | `0.272650` | `0.356852` | `0.406168` |
| 自生成几何超点 `k025` | `0.269081` | `0.359284` | `0.404608` |
| 变化 | `-0.003569` | `+0.002433` | `-0.001560` |

结论：

- 第一版全场景自生成几何超点没有通过 `even48`，不进入 `even96`。
- 更细超点可能让部分 `AP50` 边界略好，但综合平均精度下降，说明它也切碎了有用候选或削弱了稳定召回。
- 不建议继续只做全场景超点替换的小参数扫。
- 更合理的下一步是“候选局部超点”：只在每个候选的局部区域内重新切分或细化粘连超点，而不是替换整场景所有超点。

### 1.6 2026-06-12：当前对话交接摘要

本次对话围绕“下一步从哪里改起”重新梳理了所有已跑实验，结论如下：

- 当前第一点最稳底座仍是：Segment Anything 融合候选 + 二维框反投影候选 + 二维掩码级超点负约束 + 多视角一致性过滤 + 候选级连通分量清理 + 屏蔽 `rug`。
- 轻量几何判别器已经实现，并且几何、深度特征有排序信号；但 hard filter 和多掩码选择都没有通过 `even48`，暂不作为删除器，只用于诊断、排序、降权和后续特征融合。
- Alpha-CLIP 当前最佳基线以前没有使用；已完成 late correction 和 final classifier 两类验证。直接替换类别或筛选新增候选都没有净收益，因此暂不作为最近主线。
- 自生成全场景几何超点已经验证失败：`even48` 平均精度从原始超点 `0.272650` 降到 `0.269081`，不进入 `even96`。
- 与 `Details Matter` 的差距不是“是否用了 Alpha-CLIP”这一点，而是完整流程差异：它以高质量候选、超点基本单元、多视角跟踪聚合、合并删除、掩码细化和最终分类共同作用；当前基线的主要瓶颈仍是新增三维候选几何污染。
- 下一步不继续堆二维大模型，也不继续全场景超点替换；应优先做候选局部超点。

候选局部超点的当前接入判断：

- 已阅读 `utils/backprojection_fusion.py` 中 `append_backprojection_proposals` 主流程。
- 合适接入点是在全局 `_refine_mask_with_superpoints` 之后、三维连通分量清理之前。
- 局部超点只作用于新增候选，不改已有 Mask3D 结果，也不替换全场景超点。
- 初版应加独立开关，默认关闭；先在 `even48` 验证，只有明确正向才跑 `even96`。
- 初版目标不是增加候选数量，而是减少候选中被超点扩进来的墙、地面、桌面、支撑面和邻近物体背景。

建议实现方式：

```text
当前候选 mask
-> 取候选局部点云
-> 用坐标、法向、颜色构建局部近邻图
-> 做轻量区域合并得到局部超点
-> 按原始种子覆盖率、种子保留率、最大扩张比例筛选局部超点
-> 失败时回退到原候选 mask
-> 再进入已有三维连通分量清理
```

建议新增参数：

- `--backprojection_local_superpoint_refine`
- `--backprojection_local_superpoint_knn`
- `--backprojection_local_superpoint_merge_k`
- `--backprojection_local_superpoint_min_size`
- `--backprojection_local_superpoint_min_coverage`
- `--backprojection_local_superpoint_max_expansion_ratio`
- `--backprojection_local_superpoint_min_seed_retention`

当前代码状态：

- `tools/generate_geometric_superpoints.py` 和 `run_evaluation.py --processed_scene_root` 已完成，用于全场景超点替换实验。
- 候选局部超点尚未完成代码接入；下一次对话可直接从 `utils/backprojection_fusion.py` 的局部 refinement 函数和 `run_evaluation.py` 参数接入开始。
- 语法和指标验证尚未跑，完成实现后第一步应运行：

```bash
/home/jia/anaconda3/envs/openyolo3d/bin/python -m py_compile \
  utils/backprojection_fusion.py \
  run_evaluation.py
```

### 1.7 2026-06-12：候选局部超点初版验证

已完成代码：

- `utils/backprojection_fusion.py`
  - 新增候选局部超点细化函数。
  - 在全局超点细化之后、三维连通分量清理之前接入。
  - 只对新增候选生效，不改已有 Mask3D 结果，也不替换全场景超点。
  - 使用候选内部点坐标构建局部近邻图，用 Felzenszwalb 风格区域合并生成局部超点。
  - 按全局超点细化前的种子参考 mask 选择局部超点。
  - 若局部切分后点数太少、种子保留率不足或异常，则回退到原候选。
- `run_evaluation.py`
  - 新增局部超点评估参数：
    - `--backprojection_local_superpoint_refine`
    - `--backprojection_local_superpoint_knn`
    - `--backprojection_local_superpoint_merge_k`
    - `--backprojection_local_superpoint_min_size`
    - `--backprojection_local_superpoint_min_coverage`
    - `--backprojection_local_superpoint_max_expansion_ratio`
    - `--backprojection_local_superpoint_min_seed_retention`
    - `--backprojection_local_superpoint_max_points`

验证：

- 语法检查通过：

```bash
/home/jia/anaconda3/envs/openyolo3d/bin/python -m py_compile \
  utils/backprojection_fusion.py \
  run_evaluation.py
```

- `scene0011_00` 单场景冒烟通过。
- 单场景加入 `5` 个候选，局部超点对其中 `4` 个候选实际裁小，种子保留率均不低于 `0.95`。

`even48` 同口径参考：

- 原始 ScanNet200 超点当前最佳参数：平均精度 `0.272650`，百分之五十重叠率平均精度 `0.356852`，百分之二十五重叠率平均精度 `0.406168`

`even48` 局部超点两档结果：

| 配置 | 平均精度 | 百分之五十重叠率平均精度 | 百分之二十五重叠率平均精度 | 相对参考 |
|---|---:|---:|---:|---:|
| 默认局部超点：`merge_k=0.25`，覆盖 `0.25`，种子保留 `0.80` | `0.272406` | `0.356050` | `0.406132` | `-0.000244 / -0.000801 / -0.000036` |
| 保守局部超点：`merge_k=0.50`，覆盖 `0.15`，种子保留 `0.95` | `0.272888` | `0.356003` | `0.406073` | `+0.000238 / -0.000848 / -0.000096` |

报告统计：

| 配置 | 实际加入候选 | 被局部超点裁小候选 | 裁掉点数 | 种子保留率均值 | 主要回退 |
|---|---:|---:|---:|---:|---|
| 默认局部超点 | `274` | `174` | `41961` | `0.9772` | `small_output` 6 个 |
| 保守局部超点 | `274` | `161` | `30889` | `0.9924` | `low_seed_retention` 5 个，`small_output` 4 个 |

结论：

- 候选局部超点能按预期裁小新增候选，并且大多保留原始种子。
- 但两档都没有形成明确正向：默认档整体下降；保守档平均精度微升，但 AP50 和 AP25 下降。
- 按当前实验规则，不进入 `even96`。
- 初版局部超点不应作为当前最佳配置。
- 失败原因大概率是：只用三维坐标切分，缺少法向、颜色和二维掩码支持；裁掉的点中仍包含部分有用边界或能提高低阈值重叠率的区域。

下一步判断：

- 不建议继续盲扫局部超点阈值。
- 如果继续局部超点，应把 processed `.npy` 中颜色、法向接入局部图权重，并加入二维掩码/二维框支持比例，而不是只靠坐标。
- 更现实的下一步可转向候选包含和重叠关系处理，因为它不直接改候选形状，风险可能低于继续裁剪 mask。

## 2. 缩写和固定名称说明

为避免混淆，后文尽量使用中文描述。必须保留的固定名称如下：

- YOLO-World：当前稳定的二维开放词表检测模型。
- YOLOE：曾经测试过的二维检测/分割模型，当前不作为主线。
- Segment Anything 模型：二维通用分割模型，代码和路径里常写作 `SAM`。
- 平均精度：评估指标，代码输出中常写作 `AP`。
- 百分之五十重叠率平均精度：代码输出中常写作 `AP50`。
- 百分之二十五重叠率平均精度：代码输出中常写作 `AP25`。
- 二维框反投影：把二维检测框内可见三维点反投影成候选种子，旧记录里常写作 `BPR`。
- 超点：预处理得到的局部几何片段，旧记录里常写作 `superpoint`。
- 连通分量清理：三维点云里按空间连通性保留主要部分，旧记录里常写作 `CC cleanup`。
- 图形处理器：旧记录里常写作 `GPU`。

## 3. 数据和基线

ScanNet200 数据已经可用：

- 数据集软链接：`data/scannet200`
- 三维掩码：`output/scannet200/scannet200_masks`，共 `312` 个场景
- 缓存的二维预测：`output/scannet200/bboxes_2d`，共 `312` 个场景

参考基线：

- 论文基线：平均精度 `24.7`，百分之五十重叠率平均精度 `31.7`，百分之二十五重叠率平均精度 `36.2`
- 早期本地复现基线：平均精度 `23.9`，百分之五十重叠率平均精度 `31.0`，百分之二十五重叠率平均精度 `36.0`
- 最终对比默认使用论文基线，除非特别说明。

常用场景划分：

- `output/scannet200/scene_splits/even48.txt`
- `output/scannet200/scene_splits/even96.txt`
- `output/scannet200/scene_splits/odd96.txt`

实验策略：

- 不默认跑全量 `312` 个场景，除非明确要求。
- `48` 场景只作为便宜的方向筛选。
- 只有 `48` 场景出现明确正向信号，才进入一个 `96` 场景确认。
- 默认确认划分是 `even96`。
- 不同时跑 `even96` 和 `odd96`，除非明确要求。

## 4. 当前候选集合

二维框反投影候选：

- 路径：`output/backprojection_candidates_scannet200_mv_m20`
- 汇总：`output/backprojection_candidates_scannet200_mv_m20/summary.json`
- 全量导出时间约 `1` 小时 `21` 分钟

Segment Anything 模型融合候选：

- 路径：`output/sam_fused_proposals_scannet200_s5_m30_prefilter`
- 汇总：`output/sam_fused_proposals_scannet200_s5_m30_prefilter/sam_fused_proposals_summary.json`
- 全量导出时间约 `29` 分钟

新导出的带二维掩码路径的 Segment Anything 模型候选：

- `even48` 路径：`output/sam_fused_proposals_scannet200_even48_maskpath`
- `even96` 路径：`output/sam_fused_proposals_scannet200_even96_maskpath`
- 用途：让超点正负约束可以使用真实二维掩码，而不是只使用二维框。

## 5. 当前最稳定配置

当前主线仍然是：

- Segment Anything 模型融合候选
- 二维框反投影候选
- 超点细化
- 逐视角超点一致性过滤，阈值 `0.60`
- 候选级三维连通分量清理
- 屏蔽 `rug` 类别

基础参数：

```bash
--backprojection_candidates ./output/sam_fused_proposals_scannet200_s5_m30_prefilter,./output/backprojection_candidates_scannet200_mv_m20
--backprojection_min_score 0.50
--backprojection_min_seed_points 80
--backprojection_max_existing_iou 0.30
--backprojection_max_seed_in_existing_mask_ratio 0.70
--backprojection_max_candidates_per_scene 15
--backprojection_score_scale 2.00
--no-backprojection_use_candidate_fusion_score
--backprojection_blocked_classes rug
--backprojection_source_score_scales sam_fused=1.2,bpr=1.0
--backprojection_source_priorities sam_fused=2.0,bpr=1.0
--backprojection_source_max_candidates sam_fused=12,bpr=3
--backprojection_superpoint_refine
--backprojection_superpoint_min_coverage 0.30
--backprojection_superpoint_max_expansion_ratio 3.0
--backprojection_superpoint_min_view_siou 0.60
```

候选级三维连通分量清理参数：

```bash
--backprojection_cc_cleanup
--backprojection_cc_radius 0.03
--backprojection_cc_min_component_points 50
--backprojection_cc_keep_topk 1
```

当前 `96x2` 平均结果：

- 早期本地基线 `96x2`：平均精度 `0.254214`，百分之五十重叠率平均精度 `0.322218`，百分之二十五重叠率平均精度 `0.363850`
- 当前候选级连通分量清理 `96x2`：平均精度 `0.264727`，百分之五十重叠率平均精度 `0.336714`，百分之二十五重叠率平均精度 `0.378986`

相对早期本地基线提升：

- 平均精度约 `+0.0105`
- 百分之五十重叠率平均精度约 `+0.0145`
- 百分之二十五重叠率平均精度约 `+0.0151`

注意：候选级连通分量清理是小幅稳定增益，不要描述成大幅提升。

## 6. 今天的更新：二维掩码级超点负约束

### 6.1 代码改动

新增或修改：

- `tools/export_sam_fused_proposals.py`
  - Segment Anything 模型候选导出时保存二维核心掩码图片。
  - 支持视角中新增 `sam_mask_path` 字段。
- `utils/backprojection_fusion.py`
  - 加载候选时解析 `sam_mask_path`。
  - 超点正负约束优先使用二维掩码，而不是二维框。
  - 正负比例现在针对整个超点计算，不只看当前种子点。
- `tools/run_scannet200_even48_sam_mask_support_eval.sh`
  - 重新导出带二维掩码路径的候选。
  - 评估两档二维掩码超点约束。

### 6.2 实验结果

`even48` 对照，同一批重新导出的 Segment Anything 模型候选，不启用二维掩码超点约束：

- 平均精度 `0.271201`
- 百分之五十重叠率平均精度 `0.345771`
- 百分之二十五重叠率平均精度 `0.389494`

`even48`，二维掩码超点约束，正支持 `0.50`，负支持 `0.50`：

- 平均精度 `0.272610`
- 百分之五十重叠率平均精度 `0.345769`
- 百分之二十五重叠率平均精度 `0.389491`
- 实际加入候选 `273`
- 使用二维掩码支持模式的候选 `128`
- 过滤的超点片段 `49`

`even48`，二维掩码超点约束，正支持 `0.65`，负支持 `0.35`：

- 平均精度 `0.271334`
- 百分之五十重叠率平均精度 `0.345769`
- 百分之二十五重叠率平均精度 `0.389489`

因为 `even48` 的正支持 `0.50`、负支持 `0.50` 有明确提升，所以进入 `even96` 确认。

`even96`，二维掩码超点约束，正支持 `0.50`，负支持 `0.50`：

- 平均精度 `0.273030`
- 百分之五十重叠率平均精度 `0.340244`
- 百分之二十五重叠率平均精度 `0.383687`
- 实际加入候选 `550`
- 使用二维掩码支持模式的候选 `250`
- 过滤的超点片段 `112`

`even96` 候选级连通分量清理参考：

- 平均精度 `0.272253`
- 百分之五十重叠率平均精度 `0.340308`
- 百分之二十五重叠率平均精度 `0.383688`

相对参考变化：

- 平均精度 `+0.000777`
- 百分之五十重叠率平均精度 `-0.000064`
- 百分之二十五重叠率平均精度 `-0.000001`

结论：

- 二维掩码级超点负约束是真实正向信号，但幅度很小。
- 它主要提升平均精度，百分之五十和百分之二十五重叠率基本持平。
- 可以作为第一点当前最佳策略的候选组成部分，但不能宣传成大提升。

### 6.3 2026-06-06：Segment Anything 多掩码几何选择

代码改动：

- `tools/export_sam_fused_proposals.py`
  - 新增 `--sam_mask_selection_policy geometry`。
  - 对同一个二维框的多个 Segment Anything 掩码分别反投影。
  - 用三维连通性、点数、与已有三维候选重叠、厚度、平面比例等手写几何分数选择掩码。
  - 保存 `sam_mask_selection_score`、`sam_score_rank`、`sam_mask_geometry` 等诊断字段。

冒烟测试：

- `scene0011_00` 放宽阈值测试通过。
- 能正常导出候选、二维掩码图片、种子点和几何选择字段。

`even48` 导出：

- 输出路径：`output/sam_fused_proposals_scannet200_even48_geometry_multimask`
- 场景数：`48`
- 候选数：`278`
- 原始观测数：`3191`
- 其中 `98/278` 个候选不是 Segment Anything 原本最高分掩码，说明几何选择确实改变了候选池。

`even48` 评估，沿用当前最佳二维掩码超点负约束，正支持 `0.50`、负支持 `0.50`：

| 配置 | 平均精度 | 百分之五十重叠率平均精度 | 百分之二十五重叠率平均精度 |
|---|---:|---:|---:|
| 当前最佳二维掩码负约束参考 | `0.272610` | `0.345769` | `0.389491` |
| 多掩码几何选择 | `0.272135` | `0.345768` | `0.389483` |
| 变化 | `-0.000475` | `-0.000001` | `-0.000008` |

候选诊断：

- 实际加入候选：`290`
- 背景或坏几何：`259`
- 类别错误或跨类重叠：`18`
- 真正补全：`7`
- 部分补全：`2`

参考候选级连通分量清理诊断：

- 实际加入候选：`274`
- 背景或坏几何：`249`
- 类别错误或跨类重叠：`14`
- 真正补全：`7`
- 部分补全：`1`

结论：

- 多掩码几何选择没有带来明确提升，不进入 `even96`。
- 当前手写几何分数能改变掩码选择，但会额外放进一些背景或类别冲突候选。
- 这个方向不应作为当前最佳策略；候选级轻量判别器直接用于多掩码选择也已失败，后续需要专门的组内排序监督。

## 7. 今天的更新：轻量几何质量判别器

新增脚本：

- `tools/train_candidate_geometry_discriminator.py`

用途：

- 读取 `candidate_diagnostics.csv`。
- 用候选诊断标签训练轻量判别器。
- 当前支持随机森林和梯度提升树。
- 按场景划分训练集和验证集，避免同一场景信息泄漏。
- 不把真实标注相关字段作为输入特征。

当前输入特征主要包括：

- 点数
- 掩码点数
- 二维框面积比例
- 与已有三维候选的重叠率
- 种子点落入已有掩码的比例
- 多视角支持数量
- 多视角二维一致性
- 候选来源
- 当前手写质量分数
- 场景内标准化质量分数
- 类别内标准化质量分数

`even48` 候选级连通分量清理诊断表，随机森林结果：

- 样本 `271`
- 正样本 `8`
- 验证集曲线面积 `0.922078`
- 验证集平均精度 `0.250000`
- 高召回设置保留 `12/80` 个验证候选，召回率 `1.0`，精确率 `0.25`

`even96` 原始超点配置诊断表，随机森林结果：

- 样本 `2284`
- 正样本 `89`
- 验证集曲线面积 `0.677758`
- 验证集平均精度 `0.112911`
- 在召回率 `0.806452` 时，保留 `357/698` 个验证候选，精确率 `0.070028`

结论：

- 初版轻量判别器有排序信号，但当前还不能作为硬过滤器。
- 建议先作为排序或降权使用，不要直接删除候选。

### 7.1 真实几何特征补充结果

2026-06-06 已给诊断表和轻量判别器补充：

- 三维连通分量数量、最大连通分量比例、小碎片比例
- 三维包围盒长宽高、厚度、长宽高比例、点密度
- 主成分线性、平面性、散乱程度
- 平面内点比例、点到平面的平均距离和百分之九十分位距离
- 二维掩码超点正负支持、过滤比例、可用视角数
- 候选级连通分量清理保留比例

在 `even96` 实际加入的 `542` 个候选上，按场景划分训练集和验证集：

| 配置 | 验证集曲线面积 | 验证集平均精度 | 保持全部有用候选时保留数量 | 对应精确率 |
|---|---:|---:|---:|---:|
| 不含新增几何特征 | `0.6813` | `0.1661` | `164/166` | `0.0732` |
| 含新增几何和支持特征 | `0.7841` | `0.1738` | `101/166` | `0.1188` |

最重要的特征主要是：

- 点云厚度相关长宽高比例
- 点到主平面的距离
- 平面内点比例
- 点云线性和平面性
- 种子点数量
- 二维掩码正负支持比例
- 三维连通分量数量

结论：

- “背景、平面和碎片化候选”可以被真实几何特征区分，方向成立。
- 在不漏掉验证集中有用候选时，候选保留数量从 `164` 降到 `101`。
- 目前正样本只有 `25` 个，仍不适合直接作为全局硬删除器。
- 直接迁移到同一二维框内部的 Segment Anything 多掩码选择已验证失败；继续时需改用专门的组内排序监督。

### 7.2 2026-06-06：学习式 Segment Anything 多掩码选择

实现：

- 训练脚本新增 `--export_fit_all`，验证后使用全部有标签样本拟合部署模型。
- 导出脚本新增 `--sam_mask_selection_policy learned_geometry` 和 `--sam_mask_geometry_model`。
- 对同框多个 Segment Anything 掩码分别反投影，用判别器概率选择掩码。

模型与导出：

- 训练数据：`even96` 实际加入候选诊断表，共 `542` 个有标签候选。
- 部署模型：`output/scannet200/candidate_diagnostics/even96_mask_support_applied_geometry_discriminator_rf_deploy/model.pkl`
- 场景划分验证：曲线面积 `0.784091`，平均精度 `0.173775`。
- `even48` 输出：`output/sam_fused_proposals_scannet200_even48_learned_geometry_multimask`
- 导出 `263` 个候选、`3191` 个原始观测；`239/379` 个支持视角改选了非最高 Segment Anything 分数掩码。

`even48` 评估，沿用当前最佳二维掩码超点负约束，正支持 `0.50`、负支持 `0.50`：

| 配置 | 平均精度 | 百分之五十重叠率平均精度 | 百分之二十五重叠率平均精度 |
|---|---:|---:|---:|
| 当前最佳二维掩码负约束参考 | `0.272610` | `0.345769` | `0.389491` |
| 学习式多掩码选择 | `0.270682` | `0.345046` | `0.388765` |
| 变化 | `-0.001928` | `-0.000723` | `-0.000726` |

候选诊断：

| 配置 | 实际加入 | 背景或坏几何 | 类别错误 | 真正补全 | 部分补全 |
|---|---:|---:|---:|---:|---:|
| 当前参考 | `274` | `249` | `14` | `7` | `1` |
| 手写几何选择 | `290` | `259` | `18` | `7` | `2` |
| 学习式选择 | `280` | `255` | `15` | `6` | `2` |

结论：

- 学习式选择减少了部分坏候选，但少了一个真正补全，平均精度下降，不进入 `even96`。
- 候选级诊断标签迁移到同框多掩码选择存在分布偏移。
- 如继续，应为同框各掩码构造真实重叠率或诊断标签，训练组内排序模型。

### 7.3 2026-06-06：ScanNet 原始深度一致性特征

实现：

- `tools/analyze_backprojection_candidates.py`
  - 新增默认开启的深度一致性诊断特征。
  - 离线读取 `depth/`、`poses/` 和 `intrinsics.txt`，把候选三维点投影到支持视角的原始深度图。
  - 统计有效投影率、深度一致率、前景/背后残差比例、深度残差均值和分位数、深度跨度和主深度层比例。
- `tools/train_candidate_geometry_discriminator.py`
  - 默认只使用核心深度特征：有效投影率、一致率、前景/背后比例、残差均值、残差中位数、残差百分之九十分位。
  - 其他深度计数、跨度和层数特征保留在诊断表中，但不默认进入判别器。

诊断与训练：

- 诊断表：`output/scannet200/candidate_diagnostics/even96_mask_support_applied_depth_geometry/candidate_diagnostics.csv`
- 行数：`550`，其中参与训练的正负标签候选 `542`
- 随机森林对照：

| 配置 | 特征数 | 验证集曲线面积 | 验证集平均精度 | 保持全部有用候选时保留数量 |
|---|---:|---:|---:|---:|
| 不含深度特征 | `51` | `0.7841` | `0.1738` | `101/166` |
| 全量深度特征 | `66` | `0.7700` | `0.1816` | `129/166` |
| 核心深度特征 | `58` | `0.7884` | `0.1933` | `132/166` |

重要深度特征：

- 深度残差中位数
- 深度残差均值
- 深度一致率
- 候选点落在观测表面后方的比例
- 候选点落在观测表面前方的比例
- 深度残差百分之九十分位

结论：

- ScanNet 原始深度特征有真实排序信号，平均精度从 `0.1738` 提升到 `0.1933`。
- 但全召回保留数量从 `101/166` 变差到 `132/166`，不适合直接作为高召回硬过滤器。
- 当前更适合用于诊断、排序、降权和视角质量判断。
- 已尝试把深度一致性用于自适应内部种子，但全局种子裁剪没有通过 `even48`，见下一节。

### 7.4 2026-06-06：自适应内部种子

实现：

- `tools/export_sam_fused_proposals.py`
  - 新增 `--sam_adaptive_internal_seed`。
  - 对 Segment Anything 模型二维掩码内的可见种子点打分。
  - 分数由二维掩码边界距离和 ScanNet 原始深度主层一致性组成。
  - 新增保留比例、最低保留比例、边界权重、深度权重、深度分箱等命令行参数。
  - 导出候选时保存 `sam_adaptive_internal_seed` 和保留比例诊断字段。

冒烟测试：

- `scene0011_00` 放宽阈值测试通过。
- 能正常导出候选，且自适应内部种子诊断字段可用。

`even48` 导出：

| 配置 | 候选数 | 原始观测数 | 平均保留比例 |
|---|---:|---:|---:|
| `k070` | `249` | `3144` | `0.7276` |
| `k090` | `256` | `3144` | `0.9057` |

`even48` 评估，沿用当前最佳二维掩码超点负约束，正支持 `0.50`、负支持 `0.50`：

| 配置 | 平均精度 | 百分之五十重叠率平均精度 | 百分之二十五重叠率平均精度 | 相对当前参考 |
|---|---:|---:|---:|---:|
| 当前参考 | `0.272610` | `0.345769` | `0.389491` | - |
| 自适应内部种子 `k070` | `0.269600` | `0.345050` | `0.388764` | `-0.003010` |
| 自适应内部种子 `k090` | `0.271736` | `0.345038` | `0.388757` | `-0.000874` |

候选诊断：

| 配置 | 实际加入 | 背景或坏几何 | 类别错误 | 真正补全 | 部分补全 |
|---|---:|---:|---:|---:|---:|
| 当前参考 | `274` | `249` | `14` | `7` | `1` |
| 自适应内部种子 `k070` | `269` | `247` | `12` | `6` | `2` |
| 自适应内部种子 `k090` | `276` | `251` | `15` | `6` | `2` |

结论：

- 两档都低于当前参考，不进入 `even96`。
- `k090` 比 `k070` 少伤一点，说明过度裁剪是问题之一。
- 但即使保留约 `90%` 种子，真正补全仍从 `7` 降到 `6`，且背景或类别错误没有减少。
- 当前全局式“按内部可信度裁掉种子”会伤到有用边界或局部，不作为主线。
- 深度和边界信号仍可用于视角质量、候选排序或只对高风险大平面候选做选择性处理。

### 7.5 2026-06-06：视角级质量门控

实现：

- `tools/export_sam_fused_proposals.py`
  - 新增 `--seed_view_quality_gate`。
  - 新增相对阈值、最低分数、最低保留比例参数。
  - 每个支持视角写入 `view_quality_score`。
  - 多视角种子合并时，可按视角质量保留相对高质量视角。

验证：

- `scene0011_00` 单场景冒烟导出通过。
- 合成多视角合并测试通过：两个重叠视角中，低质量视角被剔除，只保留高质量视角种子。

`even48` 导出统计：

| 配置 | 候选数 | 原始观测数 | 多视角合并候选 | 实际过滤候选 |
|---|---:|---:|---:|---:|
| `rel080_minkeep050` | `258` | `3144` | `61` | `1` |
| `rel095_minkeep034` | `262` | `3144` | `64` | `33` |

`even48` 评估，沿用当前最佳二维掩码超点负约束，正支持 `0.50`、负支持 `0.50`：

| 配置 | 平均精度 | 百分之五十重叠率平均精度 | 百分之二十五重叠率平均精度 | 相对当前参考 |
|---|---:|---:|---:|---:|
| 当前参考 | `0.272610` | `0.345769` | `0.389491` | - |
| `rel080_minkeep050` | `0.272610` | `0.345769` | `0.389491` | `+0.000000` |
| `rel095_minkeep034` | `0.273013` | `0.345769` | `0.389490` | `+0.000403` |

`rel095_minkeep034` 候选诊断：

| 配置 | 实际加入 | 背景或坏几何 | 类别错误 | 真正补全 | 部分补全 |
|---|---:|---:|---:|---:|---:|
| 当前参考 | `274` | `249` | `14` | `7` | `1` |
| 视角质量门控 | `276` | `250` | `13` | `7` | `2` |

结论：

- 保守档几乎不改变候选，指标完全持平。
- 较强档有很小平均精度提升，但百分之五十和百分之二十五重叠率没有改善。
- 真正补全仍是 `7` 个，背景或坏几何没有减少。
- 这不是明确正向信号，不进入 `even96`。
- 简单按视角质量裁剪多视角种子不作为当前主线；如果继续视角级方向，需要更强的局部几何重建或局部超点，而不是只做视角分数门控。

## 8. 超点来源和是否自己生成超点

当前超点不是我们在线生成的。

在 ScanNet200 上，超点来自预处理后的场景文件。评估时读取：

```python
point_segments = np.load(processed_file, mmap_mode="r")[:, 9].astype(np.int64)
```

也就是说，`.npy` 文件第十列保存了每个三维点所属的超点编号。

当前项目只是使用这些预处理超点，不负责生成它们。

是否自己生成超点：

- 有可能更适合当前候选补全任务。
- 但不建议立刻替换全场景超点，因为风险很大。
- 更稳的方案是先做“候选局部超点”：只在新增候选附近的小范围点云里重新切分。

可以尝试的图割式局部超点方法：

```text
候选附近点云
-> 建图：点是节点，相邻点之间连边
-> 边权：空间距离、法向夹角、颜色差、深度连续性
-> 图割或区域合并
-> 得到更贴合当前候选的局部超点
-> 用局部超点替代当前超点扩张
```

这属于后续中等改动，不是最小改动。

## 9. YOLOE 相关结论

YOLOE 不再属于当前活跃主线。

已经删除本地 YOLOE 适配、标签图、掩码候选相关集成文件和对应 ScanNet200 实验输出。

保留外部资源：

- `_external/YOLOE`
- `pretrained/yoloe`

测试过的 YOLOE 形式：

- 直接替换二维检测缓存。
- 作为二维掩码候选来源。
- 只用 YOLOE 掩码构建二维标签图。
- YOLOE 掩码和 YOLO-World 标签混合。

关键结果：

- YOLO-World 参考，`even48`：平均精度 `0.271201`，百分之五十重叠率平均精度 `0.345771`，百分之二十五重叠率平均精度 `0.389494`
- YOLOE-v8s，置信度 `0.08`：平均精度 `0.241351`，百分之五十重叠率平均精度 `0.309848`，百分之二十五重叠率平均精度 `0.341308`
- YOLOE-v8m，置信度 `0.20`：平均精度 `0.245179`，百分之五十重叠率平均精度 `0.311827`，百分之二十五重叠率平均精度 `0.350450`
- YOLOE 掩码作为第三候选源，最多加 `1` 个：平均精度 `0.271182`，百分之五十重叠率平均精度 `0.345747`，百分之二十五重叠率平均精度 `0.389511`
- YOLOE 混合掩码标签图：平均精度 `0.269914`，百分之五十重叠率平均精度 `0.344299`，百分之二十五重叠率平均精度 `0.388656`

结论：

- YOLOE 的二维掩码有一定价值，但当前类别分配不如 YOLO-World 稳定。
- 直接替换 YOLO-World 会明显变差。
- 混合使用 YOLOE 掩码仍低于当前 YOLO-World 主线。
- 当前不继续投入 YOLOE，除非未来做选择性专家或语义复核。

## 10. 已尝试但不作为主线的方向

### 10.1 固定腐蚀二维掩码

结果：

- `even48` 腐蚀 `5` 像素：平均精度 `0.271280`，百分之五十重叠率平均精度 `0.345053`，百分之二十五重叠率平均精度 `0.388767`
- `even48` 腐蚀 `9` 像素：平均精度 `0.270411`，百分之五十重叠率平均精度 `0.345050`，百分之二十五重叠率平均精度 `0.388761`
- `96x2` 腐蚀 `5` 像素：平均精度 `0.261598`，百分之五十重叠率平均精度 `0.335732`，百分之二十五重叠率平均精度 `0.377858`

结论：

- 固定腐蚀会删除真实物体边界。
- 不继续作为主线。
- 自适应内部种子也已验证失败，后续不做全局种子裁剪，只考虑对高风险候选选择性处理。

### 10.2 种子合并策略和最佳视角选择

结果：

- `even48` 最佳单视角：平均精度 `0.271578`，百分之五十重叠率平均精度 `0.345769`，百分之二十五重叠率平均精度 `0.389490`
- `even48` 前两个视角：平均精度 `0.272739`，百分之五十重叠率平均精度 `0.345771`，百分之二十五重叠率平均精度 `0.389493`
- `96x2` 前两个视角：平均精度 `0.265314`，百分之五十重叠率平均精度 `0.335736`，百分之二十五重叠率平均精度 `0.378157`

结论：

- 有小信号，但不稳定。
- 说明低质量视角确实会污染候选。
- 后续应做“每个视角先清理，再跨视角合并”，不要过早混合多个视角。

### 10.3 种子深度聚类

结果：

- `even48` 固定深度聚类：平均精度 `0.270922`，百分之五十重叠率平均精度 `0.345779`，百分之二十五重叠率平均精度 `0.389499`
- `even96` 自适应深度聚类：平均精度 `0.270425`，百分之五十重叠率平均精度 `0.339567`，百分之二十五重叠率平均精度 `0.385287`

结论：

- 当前深度聚类形式不稳定。
- 可以把深度一致性作为质量特征或视角选择特征，但不建议作为硬过滤规则。

### 10.4 种子级三维连通分量清理

结果：

- 保留最大 `1` 个种子连通分量：平均精度 `0.270535`，百分之五十重叠率平均精度 `0.345138`，百分之二十五重叠率平均精度 `0.388862`
- 保留最大 `2` 个种子连通分量：平均精度 `0.269513`，百分之五十重叠率平均精度 `0.345132`，百分之二十五重叠率平均精度 `0.388865`
- 种子清理加候选清理：最高平均精度 `0.270539`

结论：

- 直觉合理，但当前实现会过早删掉有用种子。
- 不作为当前默认策略。
- 候选级连通分量清理仍然是更稳定的版本。

### 10.5 分数校准、类别一致性、类别屏蔽

结果：

- 全局质量阈值提升极小。
- 类别一致性校准几乎没有影响。
- 屏蔽高误报类别收益很小。

结论：

- 当前主要失败不是类别冲突，也不是分数线性校准问题。
- 主因仍是候选几何质量差。

### 10.6 本轮讨论过的外部模型取舍

Grounded Segment Anything 模型：

- 比单独 Segment Anything 模型多了文本定位能力，但当前已有 YOLO-World 提供二维框和类别。
- 引入后更可能重复做二维检测，速度也会更慢。
- 当前不优先使用；除非未来证明二维框本身是主要错误来源。

高精度开放词表检测器 HDINO：

- 可能提升二维检测准确率，但会改变候选来源，成本高，风险也高。
- 当前诊断显示主要瓶颈不是二维类别错误，而是二维到三维反投影后的背景污染。
- 暂不优先投入。

MASt3R 和 Depth Anything V2：

- MASt3R 更偏重建和跨视角几何，接入成本高，不适合作为当前小步实验。
- Depth Anything V2 更适合轻量接入：只作为相对深度一致性特征，辅助判断二维掩码是否包含多个深度层。
- 注意：Depth Anything V2 不替换 ScanNet 原始深度，只作为候选质量特征或视角质量特征。

LocateAnything：

- 可以作为困难二维框的点提示或定位辅助，但不应替换当前 YOLO-World 主线。
- 更适合后续选择性使用：只对低质量、低置信或几何冲突明显的候选调用。
- 当前优先级低于视角级清理、局部超点和候选包含关系处理。

综合判断：

- 暂不继续堆新的二维检测或分割大模型。
- 如果引入大模型，优先考虑 Depth Anything V2 的轻量深度一致性特征。
- LocateAnything 可作为后续困难候选的选择性补充，不作为下一轮主实验。

## 11. 候选诊断结论

诊断脚本：

- `tools/analyze_backprojection_candidates.py`

重要输出：

- `output/scannet200/candidate_diagnostics/even96_sp`
- `output/scannet200/candidate_diagnostics/even96_sqs_applied_sp`
- `output/scannet200/subset_sweeps/even48_candidate_quality_diagnostics/cc_clean_applied`

`even96` 全候选诊断：

- 总候选 `2354`
- 背景或坏几何 `1940`，约 `82.4%`
- 类别错误或跨类重叠 `255`，约 `10.8%`
- 真正补全 `60`，约 `2.5%`
- 部分补全 `29`，约 `1.2%`

`even48` 实际加入候选诊断：

- 普通配置实际加入 `300`
- 背景或坏几何 `277`
- 真正补全 `6`

`even48` 候选级连通分量清理后：

- 实际加入 `274`
- 背景或坏几何 `249`
- 真正补全 `7`
- 类别错误或跨类重叠 `14`

核心结论：

- 候选池不是缺候选，而是新增候选太脏。
- 大量新增候选来自背景、墙、地面、桌面、支撑面或邻近物体。
- 后续提升应优先解决三维种子质量和超点扩张污染。

## 12. 精细缺陷和对应策略

下面按“可观察缺陷 -> 影响 -> 对应策略”记录，避免只停留在“候选质量差”的粗略描述。

| 具体缺陷 | 可观察现象 | 主要影响 | 对应策略 | 当前状态 |
|---|---|---|---|---|
| 真实物体完全没有候选 | 小物体、细长物体、遮挡物体没有对应三维掩码 | 后续语义分类无法识别不存在的候选 | 用 YOLO-World 二维检测引导三维补全 | 已做，有整体提升 |
| 候选只覆盖物体一部分 | 只覆盖椅背、桌角、屏幕边缘等局部 | 百分之五十重叠率和平均精度受损 | 超点细化、互补候选合并 | 已做超点细化，仍有限 |
| 一个候选覆盖多个物体 | 桌子和桌上物体、椅子和墙面粘在一起 | 与任一真实物体的重叠率都不高 | 连通分量拆分、局部超点、重叠移除 | 候选级连通分量清理有效 |
| 候选粘到墙、地面、桌面 | 大片平面背景进入候选 | 候选边界变脏，重叠率下降 | 局部平面剥离、平面比例特征 | 待做，不宜粗暴删除所有平面 |
| 二维框太粗 | 框内包含背景、支撑面、邻近物体 | 反投影种子一开始就脏 | 使用二维掩码、视角级质量控制 | 已做二维掩码；全局自适应内部种子未通过 `even48` |
| 二维掩码选错 | Segment Anything 模型多个掩码质量不同 | 默认取一个掩码可能选到部分物体或背景 | 多掩码分别反投影，再按三维质量选择 | 手写几何和候选级学习式选择均未通过 `even48` |
| 坏视角污染多视角合并 | 遮挡视角或大框视角把背景带入 | 多视角合并反而污染候选 | 每个视角先清理，再跨视角合并 | 简化版试过，信号不稳 |
| 掩码内深度层混杂 | 前景物体、墙面、桌面处于不同深度层 | 多个深度层一起反投影 | 深度一致性作为视角或掩码质量特征 | 硬深度聚类和全局内部种子都不稳定 |
| 超点把背景扩进去 | 一个超点跨越目标和背景 | 少量脏种子被扩成大块背景 | 二维掩码级超点正负约束 | 已做，小幅正向 |
| 当前超点粒度不适配 | 预处理超点横跨物体和背景 | 超点细化边界受限 | 候选局部图割超点 | 待做，中等改动 |
| 好坏候选分数接近 | 真补全分数不一定高，坏候选也可能高 | 简单提高阈值会误删好候选 | 轻量几何质量判别器 | 已补真实几何特征，排序能力明显提高，但暂不硬过滤 |
| 候选重复或包含 | 小候选被大候选覆盖，多个候选高度重叠 | 脏大候选可能压制干净小候选 | 包含关系处理、重叠移除、互补合并 | 待做 |
| 类别冲突 | 多视角类别不一致 | 造成类别错误 | 类别一致性校准 | 已试，收益小，不是主瓶颈 |

优先从最能直接导出改法的缺陷开始：

1. 做视角级先清理再合并：每个支持视角先独立生成和打分，再避免低质量视角污染合并候选。
2. 做候选局部超点：只在候选附近用图割或区域合并重新切分，不替换全场景超点。
3. 做候选包含关系处理：大脏候选包含小干净候选时，裁掉重叠区域或降低大候选优先级。
4. 继续增强轻量判别器：增加不同场景的正样本数量，并补多视角质量、局部平面和包含关系特征。
5. 如果继续做 Segment Anything 多掩码选择，应构造同框多掩码组内排序监督。

## 13. 当前第一点完整流程

### 13.1 预处理

ScanNet200 预处理阶段生成：

- 点云
- 颜色
- 深度
- 相机位姿
- 可见性
- 每个点的超点编号

当前代码不在线生成超点，只读取预处理文件中的超点编号。

### 13.2 基础三维候选

Mask3D 提供基础三维实例掩码。OpenYOLO3D 将这些三维掩码投影到二维视角中，再结合二维检测进行类别赋值。

### 13.3 二维语义来源

当前使用 YOLO-World 的二维检测结果：

- 二维框
- 类别
- 置信度

### 13.4 候选补全来源

候选补全有两路：

1. 二维框反投影：把二维框内可见三维点作为种子。
2. Segment Anything 模型融合候选：用二维框提示 Segment Anything 模型，得到二维掩码，再反投影成三维种子。

### 13.5 候选融合和后处理

候选进入评估时，会经过：

- 分数过滤
- 与已有三维候选重叠过滤
- 种子点数量过滤
- 超点细化
- 多视角超点一致性过滤
- 三维连通分量清理
- 候选数量限制

### 13.6 当前新增的掩码级超点负约束

以前超点是否加入主要看正支持：种子是否覆盖这个超点。

现在新增负约束：

```text
一个超点投影回目标二维掩码内的比例要足够高；
投影到目标二维掩码外的比例不能太高。
```

这比使用二维框更强，因为二维框里常常包含背景。

## 14. 后续建议

当前最值得继续的第一点方向：

1. 保留二维掩码级超点负约束作为当前候选主线。
   - 这是目前第一点里最明确的正向小增益。
   - 不宣传成大提升，但可以作为后续改法的默认底座。
2. 新增跨视角二维掩码证据图，作为下一轮第一优先级。
   - 旧流程是单帧二维掩码先反投影，再做多视角支持统计和后处理清理。
   - 新流程应先把多个视角的二维掩码观测互相验证、聚类，再让稳定掩码簇生成三维候选。
   - 这对应三维引导掩码匹配、视角共识聚类、视角集合选择、跨视角证据聚合等思想。
   - 目标是把质量控制前移到候选生成之前，而不是继续在坏候选生成后做硬删除或降权。
3. 候选局部超点初版已验证，不作为当前最佳。
   - 当前超点来自预处理文件，不是我们在线生成。
   - 已实现只在候选附近用区域合并重新切分局部点云。
   - 两档 `even48` 都没有形成明确正向，不进入 `even96`。
   - 若继续，需要接入颜色、法向和二维掩码支持，不能只靠三维坐标。
4. 候选包含关系和重叠关系处理已验证，不作为下一轮第一步。
   - 大脏候选包含小干净候选时，降低大候选优先级或裁掉重叠区域。
   - 多个候选高度重叠时，优先保留几何更干净、视角支持更稳定的候选。
   - 简单硬删除或降权触发次数少，净收益只有极小变化；放宽规则会明显下降。
   - 后续只作为图结构掩码簇的诊断特征、风险特征或学习式排序特征。
5. 继续增强轻量几何质量判别器。
   - 已证明新增几何特征能提升排序能力。
   - ScanNet 原始深度一致性已接入；核心深度特征提升平均精度，但不适合硬过滤。
   - 后续补多视角质量、局部平面比例和包含关系特征。
   - 先用于排序、降权、诊断和选择候选，不直接全局硬删除。
6. 多掩码选择如果继续做，必须改成跨视角图内选择或同框组内监督。
   - 手写几何选择和候选级学习式选择都没有通过 `even48`。
   - 不能再直接复用候选级判别器。
   - 更推荐先在掩码证据图里让多视角一致性决定哪个二维掩码可信，而不是单帧内手写选择。
7. 视角级质量门控不作为当前主线。
   - 保守档持平，较强档只带来 `+0.000403` 平均精度，重叠率指标不动。
   - 后续若继续视角级方向，需要做“视角集合选择”，而不是简单分数门控。
8. 自适应内部种子不作为当前主线。
   - 两档 `even48` 都低于参考。
   - 后续只能作为高风险候选的选择性处理，不再全局启用。

下一轮最建议的实验顺序：

1. 新增 `tools/export_mask_graph_proposals.py`：先导出二维掩码观测，再建图聚类，最后输出兼容现有 `backprojection_candidates.json` 的候选。
2. 第一版做轻量三维引导匹配和轻量视角共识聚类：用三维实例候选、连通分量或超点作为粗三维参考，计算三维重叠、深度一致性、类别兼容和视角共识边权。
3. 在掩码簇内做视角集合选择：用贪心选互补且一致的视角，不再固定最佳单视角或固定前几个视角。
4. 复用当前 `utils/backprojection_fusion.py` 的最佳后处理，先跑四十八场景划分候选诊断和平均精度。
5. 如果图结构版本能减少坏候选并保持召回，再加入外观特征作为身份一致性边权。
6. 如果图结构版本仍受超点边界污染限制，再做二维实例边界感知的超点选择或超点拆分。

### 14.1 本轮建议和当前状态对照

| 优先级 | 建议方向 | 对应缺陷 | 当前状态 | 下一步 |
|---|---|---|---|---|
| 1 | 跨视角二维掩码证据图 | 单帧二维掩码太早反投影，坏视角和偶然掩码直接变成三维坏候选 | 已实现初版；现有观测、种子点、支持视角字段可复用 | 做掩码簇和单视角孤立候选的分层诊断 |
| 2 | 轻量三维引导掩码匹配 | 二维掩码跨视角不一致，缺少公共三维参考 | 已在初版里接入粗三维参考重叠特征 | 用三维实例候选、连通分量或超点连通区域做粗三维参考，继续改进覆盖和深度一致性 |
| 3 | 轻量视角共识率 | 两个掩码是否同物体不能只看自身重叠 | 已在初版里接入图边特征 | 图边加入第三方视角共识，先用阈值图连通分量 |
| 4 | 视角集合选择 | 最佳单视角、固定前几个视角、简单视角门控都不稳定 | 初版已用贪心选择视角 | 在掩码簇内继续优化互补且一致的视角集合 |
| 5 | 候选包含和重叠关系处理 | 大脏候选覆盖小干净候选；重复候选互相压制 | 已实现简单后处理，收益极小；放宽会下降 | 作为图结构诊断和排序特征，不再作为第一主线 |
| 6 | 候选局部超点 | 当前预处理超点跨越物体和背景；超点扩张把背景带入 | 初版已实现并跑 `even48`，两档都没有明确正向 | 暂不继续盲扫；若继续，接入颜色、法向和二维实例边界 |
| 7 | 外观特征 | 类别不冲突但跨视角掩码身份不一致 | 尚未接入 | 图结构版本稳定后作为边权 |
| 8 | 语义校准模型或新增二维来源 | 语义校准或新增二维来源 | 语义校准实验收益小；新二维来源未接入 | 暂后置，等候选几何质量提升后再做 |

当前最稳的下一步是优先级 `1`：跨视角二维掩码证据图的分层诊断。轻量判别器、候选包含关系、层级占用特征继续服务于诊断和排序，不做硬删除。多掩码选择、自适应内部种子、简单视角质量门控、初版候选局部超点、包含关系和层级关系后处理都没有形成稳定四十八场景划分提升。

不建议继续优先投入：

- 更多候选源
- 全局强多视角阈值
- 类别屏蔽
- 线性分数校准
- 全局质量阈值
- 直接替换 YOLO-World
- 继续堆 YOLOE 模块
- Grounded Segment Anything 模型直接替换当前二维流程
- HDINO 直接替换当前 YOLO-World 二维检测
- MASt3R 作为重建级大改

## 15. 运行和资源说明

图形处理器在普通沙箱里可能无法访问。需要提权运行：

```bash
nvidia-smi
```

2026-06-03 提权检查结果：

- 图形处理器：`NVIDIA GeForce RTX 4090 D`
- 显存占用：`624 MiB / 24564 MiB`
- 利用率：`9%`

长实验应使用提权命令或用户级系统服务。

不要同时跑多个重评估任务。

`even96` 在二维预测已缓存时，单个配置通常约 `6` 到 `10` 分钟；如果需要重新导出 Segment Anything 模型候选，会更久。

## 16. 重要脚本

候选导出：

- `tools/export_backprojection_candidates.py`
- `tools/export_sam_fused_proposals.py`

候选融合：

- `utils/backprojection_fusion.py`
- `run_evaluation.py`

诊断：

- `tools/analyze_backprojection_candidates.py`
- `tools/train_candidate_geometry_discriminator.py`

最新掩码级超点约束验证：

- `tools/run_scannet200_even48_sam_mask_support_eval.sh`

常用旧实验脚本：

- `tools/run_scannet200_even48_quality_guard.sh`
- `tools/run_scannet200_even48_seed_merge_policy_eval.sh`
- `tools/run_scannet200_even48_selectable_sam_refine.sh`
- `tools/run_scannet200_96_cc_cleanup_confirm.sh`
- `tools/run_scannet200_96_seed_topk2_confirm.sh`

## 17. 最近验证

语法检查通过：

```bash
/home/jia/anaconda3/envs/openyolo3d/bin/python -m py_compile \
  tools/export_sam_fused_proposals.py \
  utils/backprojection_fusion.py \
  tools/analyze_backprojection_candidates.py \
  tools/train_candidate_geometry_discriminator.py
```

脚本语法检查通过：

```bash
bash -n tools/run_scannet200_even48_sam_mask_support_eval.sh
```

二维掩码级超点约束合成测试通过：

- 两个超点中，落在二维掩码外的超点被过滤。
- 支持模式确认为 `sam_mask`，即二维掩码模式。

2026-06-06 新增验证：

- `even96` 实际加入候选重新生成几何诊断表，共 `550` 行。
- 含真实几何特征的随机森林判别器验证集曲线面积为 `0.7841`。
- 新增 Segment Anything 多掩码几何选择代码和命令行参数：
  - `--sam_multimask_topk`
  - `--sam_mask_selection_policy geometry`
- 多掩码几何选择真实 `even48` 指标验证完成：平均精度 `0.272135`，低于当前参考 `0.272610`，不进入 `even96`。
- 新增学习式 Segment Anything 多掩码选择代码和命令行参数：
  - `--sam_mask_selection_policy learned_geometry`
  - `--sam_mask_geometry_model`
- 学习式多掩码选择真实 `even48` 指标验证完成：平均精度 `0.270682`，低于当前参考 `0.272610`，不进入 `even96`。
- 新增 ScanNet 原始深度一致性诊断特征：
  - `tools/analyze_backprojection_candidates.py` 默认输出 `depth_*` 特征。
  - 随机森林默认使用核心深度特征后，验证集曲线面积 `0.7884`，验证集平均精度 `0.1933`。
  - 全召回保留候选数从无深度特征的 `101/166` 变差到 `132/166`，因此不做硬过滤，只用于排序、诊断和视角质量判断。
- 新增自适应内部种子代码和命令行参数：
  - `--sam_adaptive_internal_seed`
  - `--sam_adaptive_internal_keep_ratio`
  - `--sam_adaptive_internal_min_keep_ratio`
  - `--sam_adaptive_internal_boundary_weight`
  - `--sam_adaptive_internal_depth_weight`
- 自适应内部种子真实 `even48` 指标验证完成：
  - `k070` 平均精度 `0.269600`，低于当前参考 `0.272610`。
  - `k090` 平均精度 `0.271736`，仍低于当前参考。
  - 两档都不进入 `even96`。
- 新增视角级质量门控代码和命令行参数：
  - `--seed_view_quality_gate`
  - `--seed_view_quality_relative_threshold`
  - `--seed_view_quality_min_score`
  - `--seed_view_quality_min_keep_ratio`
- 视角级质量门控验证：
  - `scene0011_00` 单场景冒烟导出通过，候选 JSON 已写入 `view_quality_score` 和 `seed_view_quality_gate`。
  - 合成多视角合并测试通过：两个重叠视角中低质量视角被剔除，最终只保留高质量视角种子。
  - `rel080_minkeep050` 导出 `258` 个候选，实际只过滤 `1` 个多视角候选，`even48` 指标与参考完全持平。
  - `rel095_minkeep034` 导出 `262` 个候选，过滤 `33` 个多视角候选，`even48` 平均精度 `0.273013`，只比参考高 `+0.000403`，百分之五十和百分之二十五重叠率基本不动。
  - 诊断中真正补全仍为 `7` 个，背景或坏几何为 `250` 个，未形成明确正向信号，不进入 `even96`。
- 新增候选局部超点代码和命令行参数：
  - `--backprojection_local_superpoint_refine`
  - `--backprojection_local_superpoint_knn`
  - `--backprojection_local_superpoint_merge_k`
  - `--backprojection_local_superpoint_min_size`
  - `--backprojection_local_superpoint_min_coverage`
  - `--backprojection_local_superpoint_max_expansion_ratio`
  - `--backprojection_local_superpoint_min_seed_retention`
  - `--backprojection_local_superpoint_max_points`
- 候选局部超点验证：
  - `scene0011_00` 单场景冒烟通过。
  - 默认档 `even48`：平均精度 `0.272406`，低于同口径参考 `0.272650`。
  - 保守档 `even48`：平均精度 `0.272888`，但 AP50 `0.356003`、AP25 `0.406073` 低于参考 `0.356852`、`0.406168`。
  - 两档都没有明确正向，不进入 `even96`。

2026-06-15 新增候选包含/重叠后处理：

- 新增命令行参数：
  - `--backprojection_containment_action {none,downweight,carve,remove_large}`
  - `--backprojection_containment_threshold`
  - `--backprojection_containment_min_area_ratio`
  - `--backprojection_containment_score_ratio`
  - `--backprojection_containment_quality_margin`
  - `--backprojection_containment_score_factor`
  - `--backprojection_containment_min_points`
- 新增脚本：
  - `tools/run_scannet200_even48_containment_sweep.sh`
- 语法和小型合成行为测试通过：
  - `carve` 会裁掉大候选中被小 mask 覆盖的点。
  - `downweight` 会降低包含小 mask 的大候选分数。
- 同一 `even48`、同一当前最佳候选配置下验证：

| 配置 | 平均精度 | 百分之五十重叠率平均精度 | 百分之二十五重叠率平均精度 | 触发事件 | 候选变化 |
|---|---:|---:|---:|---:|---:|
| current | `0.272610` | `0.345769` | `0.389491` | `0` | `273 -> 273` |
| same-class carve | `0.272609` | `0.345769` | `0.389491` | `8` | `273 -> 273`，裁掉 `2003` 点 |
| same-class downweight | `0.272612` | `0.345771` | `0.389492` | `8` | `273 -> 273` |
| same-class remove_large | `0.272612` | `0.345771` | `0.389492` | `8` | `273 -> 265` |
| cross-class remove_large | `0.272614` | `0.345771` | `0.389493` | `13` | `273 -> 260` |
| relaxed cross-class remove_large (`0.60/1.2/0.50`) | `0.269008` | `0.342164` | `0.385883` | `17` | `273 -> 256` |

结论：

- 简单候选包含/重叠后处理可以正确触发，但触发次数很少，净收益只有 `+0.000002` 到 `+0.000004`，不足以进入 `even96`。
- 同类限制下只触发 `8` 次；跨类别删除触发 `13` 次，仍没有明确提分。
- 放宽到跨类别 `coverage=0.60`、面积比 `1.2`、分数比 `0.50` 后，触发 `17` 次但指标明显下降，说明全局硬删除会误删有用候选。
- 这说明“包含关系”方向本身有诊断价值，但不能只靠简单规则直接删除或降权。
- 下一步若继续关系处理，应提取更细的关系特征：
  - 大候选包含小候选时，大候选独有区域的连通性、平面性、视角支持比例。
  - 小候选是否由原始 Mask3D 支持或多视角稳定支持。
  - 大候选与多个小候选的 one-to-many 关系。
  - 同类和跨类分别建模，不直接全局删除。
- 当前实现先保留为可开关诊断和安全后处理，不作为当前最佳配置。

2026-06-15 后续关系特征诊断：

- `tools/analyze_backprojection_candidates.py` 新增非 GT 关系诊断列：
  - `relation_base_contained_count`
  - `relation_base_max_coverage`
  - `relation_base_max_area_ratio`
  - `relation_candidate_contained_count`
  - `relation_candidate_max_coverage`
  - `relation_candidate_max_area_ratio`
  - `relation_any_contained_count`
  - `relation_any_max_coverage`
  - `relation_exclusive_point_ratio`
  - `relation_exclusive_point_count`
  - `relation_contained_point_count`
- `tools/train_candidate_geometry_discriminator.py` 默认特征列表已加入这些关系特征。
- 在当前 `even48` 应用候选上生成无深度诊断：
  - 输出：`output/diagnostics/relation_features_even48_current`
  - 行数：`273`
  - 关系触发：`13` 个候选有 `relation_any_contained_count > 0`
  - 这 `13` 个全部属于 `background_or_bad_geometry`
- 用该 CSV 训练随机森林冒烟测试通过：
  - 行数：`272`
  - 特征数：`69`
  - 验证 ROC-AUC：`0.6875`
  - 验证平均精度：`0.1901`
- 关系特征本身不是最重要特征，但能标记一小批明确坏候选。它适合进入轻量判别器或风险排序，不适合单独作为删除规则。

2026-06-15 Clutt3R-Seg 启发的候选层级诊断初版：

- 动机：
  - Clutt3R-Seg 的核心启发不是硬删除包含关系掩码，而是用层级实例树和超体素占用相似度判断父子掩码哪个更可信。
  - 前一轮包含关系硬删除和分数降权已证明简单规则收益极小，放宽阈值还会误删。
  - 因此本轮先做离线诊断特征，不直接改 `run_evaluation.py` 的预测结果。
- `tools/analyze_backprojection_candidates.py` 新增默认开启的 `--hierarchy_features`：
  - 使用处理后 `.npy` 第 `9` 列超点。
  - 将每个候选和基线掩码转成超点占用向量。
  - 用超点大小加权的 Jaccard 和覆盖率建立父子关系。
  - 新增参数：
    - `--hierarchy_containment_threshold`，默认 `0.80`
    - `--hierarchy_min_area_ratio`，默认 `1.2`
    - `--hierarchy_same_class_only`
- 新增非 GT 特征列：
  - `hierarchy_superpoint_count`
  - `hierarchy_mean_superpoint_occupancy`
  - `hierarchy_min_superpoint_occupancy`
  - `hierarchy_max_superpoint_occupancy`
  - `hierarchy_low_occupancy_mass_ratio`
  - `hierarchy_base_parent_count`
  - `hierarchy_candidate_parent_count`
  - `hierarchy_parent_count`
  - `hierarchy_parent_max_candidate_coverage`
  - `hierarchy_parent_max_weighted_jaccard`
  - `hierarchy_parent_min_extra_mass_ratio`
  - `hierarchy_base_child_count`
  - `hierarchy_candidate_child_count`
  - `hierarchy_child_count`
  - `hierarchy_child_max_coverage`
  - `hierarchy_child_max_weighted_jaccard`
  - `hierarchy_child_max_area_ratio`
  - `hierarchy_child_union_coverage`
  - `hierarchy_exclusive_superpoint_ratio`
  - `hierarchy_exclusive_superpoint_count`
  - `hierarchy_any_related_count`
- `tools/train_candidate_geometry_discriminator.py` 默认数值特征列表已加入上述 `hierarchy_*` 特征。
- 冒烟验证：
  - 命令使用 `--max_scenes 1 --no-depth_features`。
  - 输出：`output/diagnostics/hierarchy_features_smoke`
  - 生成 `6` 条候选记录。
  - CSV 表头包含所有 `hierarchy_*` 列。
  - 冒烟场景中出现候选父子对，加权 Jaccard 两侧一致，说明层级图计算链路可用。
- 下一步：
  - 在当前 `even48` 应用候选上重新生成完整诊断 CSV。
  - 重新训练轻量几何判别器，观察 `hierarchy_*` 是否进入最重要特征，或提高 ROC-AUC 和平均精度。
  - 若有稳定区分度，再考虑把它接入 `utils/backprojection_fusion.py` 做条件替换，而不是硬删除。

补充批量验证：

- 对两个候选源生成无深度全量诊断：
  - `output/sam_fused_proposals_scannet200_even48_maskpath`
  - `output/backprojection_candidates_scannet200_mv_m20`
  - 输出：`output/diagnostics/hierarchy_features_even48_current`
  - 场景数：`312`
  - 候选行数：`5826`
  - 注意：这是候选源全量诊断，不是之前 `273` 个已应用候选的同口径结果。
- 层级触发统计：
  - `hierarchy_any_related_count > 0`：`3421 / 5826`
  - 父候选关系：`1954`
  - 子候选关系：`2082`
  - 有层级关系的候选中，`background_or_bad_geometry` 为 `3179`，另有 `true_completion_50` 为 `17`，`partial_completion_25` 为 `66`。
  - 有层级关系候选的平均 `hierarchy_exclusive_superpoint_ratio` 为 `0.693`。
- 用该全量 CSV 训练随机森林冒烟测试：
  - 参与训练行数：`5768`
  - 特征数：`88`
  - 验证 ROC-AUC：`0.813961`
  - 验证平均精度：`0.080165`
  - 最重要特征第一名：`hierarchy_low_occupancy_mass_ratio`，重要性 `0.185697`
  - 前二十五个重要特征中还出现：
    - `hierarchy_max_superpoint_occupancy`
    - `hierarchy_mean_superpoint_occupancy`
    - `hierarchy_min_superpoint_occupancy`
- 初步判断：
  - Clutt3R-Seg 风格的超点占用特征比简单点级关系更有诊断信号，尤其能刻画低占用、碎片化、贴边或跨超点污染候选。
  - 但父子关系仍会覆盖少量真阳性，不能直接用作硬删规则。
  - 下一步应在“已应用候选同口径”上重新生成诊断，再决定是否做条件替换。

2026-06-15 层级风险分数接入最终评估：

- 已完成代码：
  - `utils/backprojection_fusion.py`
    - 新增 `_superpoint_hierarchy_score_factor`。
    - 对新增候选计算超点占用。
    - 若候选大量点落在低占用超点中，则按 `low_occupancy_mass_ratio` 乘性降权。
    - 只改变新增候选分数，不删除候选、不改变掩码形状。
  - `run_evaluation.py`
    - 新增参数：
      - `--backprojection_hierarchy_score_weight`
      - `--backprojection_hierarchy_low_occupancy_threshold`
      - `--backprojection_hierarchy_min_score_factor`
    - 当层级分数权重大于 `0` 时，即使没有开启超点细化，也会加载处理后 `.npy` 第 `9` 列超点。
  - 新增脚本：
    - `tools/run_scannet200_even48_hierarchy_score_sweep.sh`
- 合成冒烟测试：
  - 完整占用超点的候选：`hierarchy_score_factor = 1.0`
  - 低占用碎片候选：`hierarchy_score_factor = 0.5`
  - 说明降权链路正常。
- 真实 `even48` 同当前最佳掩码支持参数验证：

| 配置 | 平均精度 | 百分之五十重叠率平均精度 | 百分之二十五重叠率平均精度 | 降权候选 | 平均因子 |
|---|---:|---:|---:|---:|---:|
| 当前参考 | `0.272610` | `0.345769` | `0.389491` | - | - |
| `weight=0.50, threshold=0.25, min_factor=0.70` | `0.272613` | `0.345771` | `0.389491` | `76 / 273` | `0.923470` |
| `weight=0.75, threshold=0.25, min_factor=0.60` | `0.272613` | `0.345771` | `0.389491` | `76 / 273` | `0.897383` |
| `weight=0.50, threshold=0.35, min_factor=0.70` | `0.272613` | `0.345771` | `0.389491` | `78 / 273` | `0.918905` |

结论：

- 层级风险分数能影响较多新增候选分数，但最终平均精度只有 `+0.000003` 左右的极小变化，不能算有效提升。
- 单纯分数降权和之前包含关系降权一样，力度不足以改变最终匹配结果。
- Clutt3R-Seg 启发的占用特征仍有诊断价值，但落地方式需要从“分数降权”升级到更结构化的条件替换：
  - 父候选是大脏掩码且子候选覆盖其可靠区域时，压低或替换父候选。
  - 子候选是碎片、父候选额外污染低且多视角稳定时，用父候选替代子候选。
  - 所有规则都应先在已应用候选同口径诊断上验证，避免误伤真阳性。

2026-06-16 层级条件替换初版：

- 已完成代码：
  - `utils/backprojection_fusion.py`
    - 在 `_postprocess_appended_proposals` 中新增层级替换后处理。
    - 初版只实现 `remove_parent`，即子候选覆盖父候选的可靠区域且父候选独占区域较小时移除父候选。
    - 第一版点级覆盖条件在真实 `even48` 三档均 `0` 触发。
    - 随后改为超点占用覆盖，和 Clutt3R-Seg 的超体素占用思路一致。
  - `run_evaluation.py`
    - 新增参数：
      - `--backprojection_hierarchy_substitution_action {none,remove_parent}`
      - `--backprojection_hierarchy_substitution_min_child_coverage`
      - `--backprojection_hierarchy_substitution_max_parent_exclusive_ratio`
      - `--backprojection_hierarchy_substitution_min_area_ratio`
      - `--backprojection_hierarchy_substitution_min_children`
  - 新增脚本：
    - `tools/run_scannet200_even48_hierarchy_substitution_sweep.sh`
- 合成冒烟测试：
  - 父候选由两个子候选覆盖 `95%` 时，`remove_parent` 正确触发。
  - 新增候选从 `3 -> 2`。
- 已应用候选同口径诊断：
  - 输出：`output/diagnostics/hierarchy_features_even48_applied_current`
  - 行数：`273`
  - `hierarchy_any_related_count > 0`：`73`
  - `hierarchy_parent_count > 0`：`39`，全部是 `background_or_bad_geometry`
  - `hierarchy_child_count > 0`：`42`，其中 `40` 个是 `background_or_bad_geometry`，另有 `1` 个 `true_completion_50`
  - 说明层级关系在已应用候选里确实主要标记坏候选，但可安全替换的结构很少。
- 真实 `even48` 验证：

| 配置 | 平均精度 | 百分之五十重叠率平均精度 | 百分之二十五重叠率平均精度 | 触发 | 候选变化 |
|---|---:|---:|---:|---:|---:|
| 当前参考 | `0.272610` | `0.345769` | `0.389491` | - | `273 -> 273` |
| 点级严格版 `cov080/ex020/min2` | `0.272610` | `0.345769` | `0.389491` | `0` | `273 -> 273` |
| 点级保守版 `cov085/ex015/min1` | `0.272610` | `0.345769` | `0.389491` | `0` | `273 -> 273` |
| 点级放宽版 `cov075/ex030/min1` | `0.272610` | `0.345769` | `0.389491` | `0` | `273 -> 273` |
| 超点占用放宽版 `cov075/ex030/min1` | `0.272610` | `0.345769` | `0.389491` | `2` | `273 -> 271` |

超点占用放宽版触发事件：

- `scene0518_00`：父候选 `0`，面积 `953`，子候选联合覆盖率 `0.7125`，父候选独占比例 `0.2875`，子候选 `2`。
- `scene0599_02`：父候选 `1`，面积 `192`，子候选联合覆盖率 `0.7969`，父候选独占比例 `0.2031`，子候选 `0`。

结论：

- 条件替换代码路径可用，超点占用条件也能在真实 `even48` 触发。
- 但当前可触发的父候选移除太少，且移除的父候选不影响最终匹配，平均精度完全不变。
- 这说明“子候选覆盖父候选时移除父候选”在当前已应用候选集合里不是主要瓶颈。
- 如果继续 Clutt3R-Seg 方向，更可能需要做另一半：碎片子候选到稳定父候选的替换或扩张，但这会增加误引入大脏掩码的风险，必须先做更细的诊断。

2026-06-16 论文路线重新核对后的最终下一步：

- 已核对并合并 MV3DIS、Any3DIS、Clutt3R-Seg、MaskClustering、OpenTrack3D、OVSeg3R、VGGT/VGGT-Ω 对当前基线的启发。
- 总结论：
  - 当前瓶颈不是新的二维检测器。
  - 当前瓶颈也不是再继续调候选生成后的固定阈值、连通分量半径、包含关系删除或分数降权。
  - 最值得做的是把“跨视角证据判断”前移到候选生成之前。
- 最终下一步版本：

```text
YOLO-World 二维检测
-> Segment Anything 生成二维掩码观测
-> 不立即反投影成最终三维候选
-> 构建跨视角掩码证据图
-> 用粗三维参考、三维种子重叠、深度一致性、类别兼容、视角共识建立边
-> 聚类得到稳定物体证据
-> 在掩码簇内选择互补且一致的视角集合
-> 聚合种子点生成三维候选
-> 复用当前最佳超点约束、连通分量清理和融合评估
```

- 第一版实现建议：
  - 新增 `tools/export_mask_graph_proposals.py`。
  - 复用 `tools/export_sam_fused_proposals.py` 中二维检测、二维分割、种子点、证据图保存逻辑，但把二维掩码观测收集和三维候选生成拆开。
  - 节点字段至少包含：帧编号、类别编号、二维框、二维掩码路径、种子点索引、检测分数、分割分数、视角质量分数、二维掩码几何信息。
  - 边字段至少包含：种子点重叠率、种子点包含率、类别兼容、深度一致性、粗三维参考重叠、视角共识分数。
  - 聚类初版用阈值图连通分量或并查集，不需要先上复杂学习模型。
  - 输出仍保持 `backprojection_candidates.json` 兼容格式，并新增掩码簇编号、图共识分数、掩码簇观测数、选中视角数、深度一致性分数等诊断字段。
- 验证顺序：
  1. 单场景冒烟测试：确认二维掩码观测数、图边数、掩码簇数和输出候选文件格式正确。
  2. 四十八场景划分候选诊断：比较坏候选比例、真正补全候选数、掩码簇支持分布、已有三维实例候选覆盖率。
  3. 四十八场景划分平均精度：只在同时保持召回并减少坏候选时进入下一步。
  4. 若有效，再加入外观特征作为身份一致性边权。
  5. 若仍受超点边界污染，再做 OVSeg3R 简化版的二维实例边界感知超点选择。
- 暂不优先做：
  - Grounded SAM 新来源。
  - Alpha-CLIP 固定阈值校正。
  - VGGT/VGGT-Ω 点跟踪。
  - 更多包含关系或层级关系硬删除。
  - 继续扫局部超点纯几何阈值。

2026-06-16 跨视角掩码证据图初版代码接入：

- 已新增 `tools/export_mask_graph_proposals.py`。
- 初版实现内容：
  - 复用二维检测加二维分割的观测生成逻辑。
  - 每个二维掩码观测保存帧编号、类别编号、二维框、二维掩码路径、种子点索引、二维检测/分割分数、视角质量分数、二维掩码几何信息、已有三维实例候选重叠信息。
  - 构建跨视角掩码证据图，边特征包括：
    - 种子点重叠率
    - 种子点包含率
    - 类别兼容
    - 粗三维参考重叠
    - 深度一致性，当前用三维种子点中心的空间一致性近似
    - 视角共识分数
  - 用阈值图连通分量做第一版聚类。
  - 在掩码簇内用贪心选择互补视角，避免固定最佳单视角或固定前几个视角。
  - 输出兼容现有 `backprojection_candidates.json`，新增：
    - 掩码簇编号
    - 掩码簇内观测编号
    - 掩码簇内被选中的观测编号
    - 掩码簇观测数量
    - 选中视角数量
    - 图边数量
    - 图边平均分数
    - 图共识分数
    - 深度一致性分数
- 已新增 `tools/run_scannet200_even48_mask_graph_eval.sh`。
  - 默认导出 `output/mask_graph_proposals_scannet200_even48_s5_m30`。
  - 默认用“掩码证据图候选 + 二维框反投影候选”融合评估。
  - 后处理沿用当前最佳口径：超点正负约束、连通分量清理、`rug` 屏蔽、候选来源数量上限。
- 已验证：
  - `/home/jia/anaconda3/envs/openyolo3d/bin/python -m py_compile tools/export_mask_graph_proposals.py` 通过。
  - `bash -n tools/run_scannet200_even48_mask_graph_eval.sh` 通过。
  - `tools/export_mask_graph_proposals.py --help` 正常。
  - 纯处理器合成测试通过：图边构建、连通分量、视角选择、候选字段生成链路可用。
- 真实场景冒烟测试状态：
  - 默认沙箱内尝试 `max_scenes=1`、`max_frames=2`、`max_detections_per_frame=2` 时失败，原因是原项目 `utils/__init__.py:get_mesh_projections()` 强制 `.cuda()`，沙箱内无 CUDA。
  - 提权到显卡环境后，单场景真实导出冒烟测试通过：
    - 输出：`output/mask_graph_smoke`
    - `scene0011_00` 导出 `2` 个二维掩码观测、`0` 条图边、`2` 个连通分量、`1` 个候选。
  - 单场景评估冒烟测试通过：
    - `run_evaluation.py` 成功加载 `1` 个掩码证据图候选。
    - 报告中 `loaded=1`、`applied=1`、`skipped=0`。
    - 说明新 JSON、种子点 `.npz`、候选来源规则和现有 `append_backprojection_proposals` 链路可用。
- 已补 `utils/backprojection_fusion.py`：
  - `_candidate_source_kind()` 现在会把包含 `mask_graph` / `mask-graph` 的候选来源归一为 `mask_graph`。
  - 这样来源分数缩放、来源候选数量上限和报告中的候选来源类型更清晰。

2026-06-16 跨视角掩码证据图四十八/九十六场景划分验证：

- 运行脚本：
  - `tools/run_scannet200_even48_mask_graph_eval.sh`
  - 已参数化：
    - `GRAPH_MIN_CLUSTER_OBSERVATIONS`
    - `GRAPH_KEEP_SINGLETONS`
    - `GRAPH_MIN_SEED_IOU`
    - `GRAPH_MIN_SEED_CONTAINMENT`
    - `GRAPH_EDGE_SCORE_THRESHOLD`
    - `SOURCE_LIMITS_GRAPH_BPR`
- 保守掩码证据图版本：
  - 设置：`GRAPH_MIN_CLUSTER_OBSERVATIONS=2`、`GRAPH_KEEP_SINGLETONS=0`
  - 导出：`output/mask_graph_proposals_scannet200_even48_s5_m30`
  - 导出统计：
    - 场景数：`48`
    - 原始二维掩码观测：`3144`
    - 图边：`3927`
    - 图连通分量：`1511`
    - 候选：`67`
    - 有候选的场景数：`28`
  - 四十八场景划分结果：
    - 平均精度 `0.270292`
    - 百分之五十重叠率平均精度 `0.341455`
    - 百分之二十五重叠率平均精度 `0.385170`
  - 结论：候选召回太低，低于当前参考 `0.272610 / 0.345769 / 0.389491`。
- 单视角孤立候选版本：
  - 设置：`GRAPH_MIN_CLUSTER_OBSERVATIONS=1`、`GRAPH_KEEP_SINGLETONS=1`
  - 四十八场景划分导出：`output/mask_graph_proposals_scannet200_even48_s5_m30_singletons`
  - 四十八场景划分导出统计：
    - 场景数：`48`
    - 原始二维掩码观测：`3144`
    - 图边：`3927`
    - 图连通分量：`1511`
    - 候选：`225`
    - 有候选的场景数：`39`
  - 四十八场景划分评估：
    - loaded：`1066`
    - applied：`252`
    - 实际加入来源：掩码证据图候选 `122`、二维框反投影候选 `130`
    - 平均精度 `0.274624`
    - 百分之五十重叠率平均精度 `0.345785`
    - 百分之二十五重叠率平均精度 `0.389503`
  - 四十八场景划分结论：相对当前参考，平均精度提升约 `+0.002014`，百分之五十和百分之二十五重叠率平均精度基本持平微增。
- 单视角孤立候选版本九十六场景划分确认：
  - 导出：`output/mask_graph_proposals_scannet200_even96_s5_m30_singletons`
  - 场景数：`96`
  - 原始二维掩码观测：`6237`
  - 图边：`8272`
  - 图连通分量：`2918`
  - 候选：`536`
  - 有候选的场景数：`85`
  - 评估：
    - loaded：`2272`
    - applied：`515`
    - 实际加入来源：掩码证据图候选 `253`、二维框反投影候选 `262`
    - 平均精度 `0.271843`
    - 百分之五十重叠率平均精度 `0.338956`
    - 百分之二十五重叠率平均精度 `0.382132`
  - 九十六场景划分结论：低于当前九十六场景划分参考 `0.273030 / 0.340244 / 0.383687`，因此单视角孤立候选版本暂不替代当前最佳。
- 当前判断：
  - 掩码证据图代码链路可用，且四十八场景划分有正向信号。
  - 但直接保留单视角孤立候选会把不少单帧候选重新放回来，泛化到九十六场景划分不稳。
  - 下一步不应简单继续放宽单视角孤立候选；应做掩码簇质量诊断和排序：
    - 区分真正多视角稳定掩码簇、单帧孤立候选、已有三维实例候选覆盖高的候选。
    - 对单视角孤立候选单独设候选来源类型或分数/数量上限。
    - 用图边数量、图共识分数、选中视角数、种子落入已有三维掩码比例做分层数量上限。
    - 优先保留多视角掩码簇，谨慎补充单视角孤立候选。

2026-06-23 第一阶段关系可靠性与实例假设诊断更新：

- 本轮目标：
  - 不继续盲目增加二维检测来源。
  - 不引入 DINO、Grounded SAM、SAM2 或学习式选择器。
  - 先修正证据图中最容易误导候选生成的部分：
    - 真实逐点可见性和二维掩码一致性。
    - 同实例支持、硬冲突、父子包含、不确定关系分开记录。
    - 用约束式三维实例假设替代普通图连通分量直接出候选。
    - 保留已有实例修正诊断，而不是直接补全或替换 Mask3D。

- 代码改动：
  - `tools/export_mask_graph_proposals.py`
    - 新增跨视角逐点可见性和二维掩码包含验证。
    - 关系计算不再主要依赖三维中心距离，而是记录：
      - 深度可见一致性。
      - 掩码一致性。
      - 共同可见点统计。
    - 同帧内记录：
      - 父子包含关系。
      - 同帧互斥关系。
      - 同帧冲突关系。
      - 欠分割大掩码桥接风险。
    - 跨视角类别不一致不再直接判为硬冲突，改为“不确定关系”。
    - 新增约束式实例假设：
      - 默认 `--graph_hypothesis_mode constrained`。
      - 观测必须有同实例强支持才能加入假设。
      - 不确定关系不作为合并依据。
      - 欠分割桥接观测默认不参与普通合并。
    - 输出 `mask_graph_trace.json`，保存观测、关系、假设、跳过原因和已有实例修正证据。
    - 输出观测点集、完整核心点集和缺口核心点集。
    - 对“已有实例支持”不再只丢弃，额外保存诊断：
      - 原始已有候选。
      - 二维证据修剪核心。
      - 原始加核心补全。
  - `tools/analyze_applied_mask_graph_candidates.py`
    - 支持分析全部导出候选，不只分析进入评估的候选。
    - 支持分析已有实例修正诊断。
    - 支持同时分析输出点集、完整核心、缺口核心。
    - 支持输出候选精确率、真实实例覆盖率、最高真实交并比、连通块数量、类别是否正确。
    - 新增 `--include_revision_variants`，用于比较原始 Mask3D、修剪核心、补全版本。
  - `tools/analyze_mask_graph_trace_relations.py`
    - 新增关系质量诊断脚本。
    - 使用真实实例标注判断同实例支持边、冲突边、包含边是否可靠。
    - 对“不确定关系”和“弱关系”不计算正确率，只统计真实关系分布。
  - `tools/run_scannet200_even48_mask_graph_eval.sh`
    - 新增实例假设相关参数。
    - 复用导出检查增加新字段校验。

- 完整运行命令：

```bash
OPENYOLO3D_ALLOW_LEGACY_2D_CACHE=1 \
/home/jia/anaconda3/envs/openyolo3d/bin/python tools/export_mask_graph_proposals.py \
  --dataset_name scannet200 \
  --dataset_root ./data/scannet200 \
  --path_to_3d_masks ./output/scannet200/scannet200_masks \
  --path_to_2d_preds ./output/scannet200/bboxes_2d \
  --scene_list ./output/scannet200/scene_splits/even48.txt \
  --output_dir ./output/mask_graph_proposals_scannet200_even48_phase1_relation_fix_gpu \
  --detection_score_th 0.45 \
  --min_seed_points 80 \
  --max_box_area_ratio 0.30 \
  --frame_stride 5 \
  --max_detections_per_frame 8 \
  --max_candidates_per_scene 30 \
  --blocked_classes rug \
  --ranking_policy priority \
  --sam_multimask_topk 1 \
  --graph_same_class_only \
  --graph_min_seed_iou 0.03 \
  --graph_min_seed_containment 0.18 \
  --graph_min_reference_coverage 0.20 \
  --graph_edge_score_threshold 0.35 \
  --graph_min_cluster_observations 2 \
  --no-graph_keep_singletons \
  --graph_max_views_per_cluster 4 \
  --graph_point_vote_min_score 0.35 \
  --graph_point_vote_min_support 1 \
  --graph_point_vote_min_keep_ratio 0.35 \
  --graph_gap_seed_policy full_core \
  --graph_hypothesis_mode constrained \
  --graph_hypothesis_min_support_edges 1 \
  --export_max_existing_iou 0.30 \
  --export_max_seed_in_existing_mask_ratio 0.30
```

```bash
/home/jia/anaconda3/envs/openyolo3d/bin/python tools/analyze_mask_graph_trace_relations.py \
  --traces output/mask_graph_proposals_scannet200_even48_phase1_relation_fix_gpu \
  --output_dir output/scannet200/subset_sweeps/even48_mask_graph_phase1_relation_fix_diagnostics/relation_quality
```

```bash
/home/jia/anaconda3/envs/openyolo3d/bin/python tools/analyze_applied_mask_graph_candidates.py \
  --candidates output/mask_graph_proposals_scannet200_even48_phase1_relation_fix_gpu \
  --include_existing_support_diagnostics \
  --include_revision_variants \
  --output_dir output/scannet200/subset_sweeps/even48_mask_graph_phase1_relation_fix_diagnostics/revision_upper_bound \
  --cc_max_points 50000
```

- even48 导出统计：
  - 场景数：`48`
  - 二维掩码观测：`3144`
  - 同实例支持边：`2242`
  - 弱关系：`1236`
  - 不确定关系：`2576`
  - 硬冲突边：`2`
  - 普通图连通分量：`2040`
  - 约束式实例假设：`2040`
  - 最终新增候选：`2`
  - 已有实例修正诊断：逐场景合计 `352`

- 关系质量诊断：
  - 同实例支持边数量：`2242`
  - 同实例支持边正确率：`0.925513`
  - 旧逻辑中大量跨类别关系会被判成冲突；新逻辑将其中 `2576` 条改为不确定关系。
  - 不确定关系真实分布：
    - 真同一实例：`2126`
    - 不同类别实例：`230`
    - 同类别不同实例：`24`
    - 未知：`196`
  - 结论：跨类别关系不能直接当硬冲突，因为大量是同一物体在不同视角的类别抖动。

- 候选质量诊断：
  - 最终新增候选只有 `2` 个，三种点集共 `6` 行诊断，全部为“背景污染或几何错误”。
  - 因此本轮不运行最终 AP 作为主结论；当前新增候选质量不足，跑 AP 只会验证“不应加入这些新增候选”。
  - 完整核心诊断均值：
    - 最高真实交并比均值：`0.292665`
    - 候选精确率均值：`0.909998`
    - 真实实例覆盖率均值：`0.306042`
  - 缺口核心诊断均值：
    - 最高真实交并比均值：`0.006815`
    - 候选精确率均值：`0.854903`
    - 真实实例覆盖率均值：`0.006862`
  - 结论：只输出“未被已有候选覆盖的缺口核心”基本是错误方向；它通常只是很小的残片或几何噪声。

- Mask3D 修正上界诊断：
  - 已有实例修正诊断记录数：`352`
  - 原始已有候选：
    - 最高真实交并比均值：`0.772841`
    - 候选精确率均值：`0.914805`
    - 真实实例覆盖率均值：`0.839401`
  - 二维证据修剪核心：
    - 最高真实交并比均值：`0.293800`
    - 候选精确率均值：`0.912126`
    - 真实实例覆盖率均值：`0.306379`
    - 只有 `8 / 352` 条比原始 Mask3D 更好。
  - 原始加核心补全：
    - 最高真实交并比均值：`0.761702`
    - 候选精确率均值：`0.888561`
    - 真实实例覆盖率均值：`0.846601`
    - `70 / 352` 条比原始 Mask3D 更好，但 `195 / 352` 条更差。
  - 结论：
    - 不能把二维证据修剪核心直接替换 Mask3D。
    - 不能无条件把二维核心补到 Mask3D 上。
    - 当前证据图更适合作为“诊断和候选修正证据”，还不是安全的自动替换模块。

- 当前结论：
  - 证据图方向没有被否定。
  - 真正被否定的是：
    - 普通连通分量直接出候选。
    - 只输出缺口核心。
    - 无条件修剪 Mask3D。
    - 无条件补全 Mask3D。
  - 第一阶段最有价值的成果是发现：
    - 同实例支持边已经有较高质量。
    - 硬冲突定义必须非常保守。
    - 跨类别关系应先作为不确定关系保留。
    - 证据图目前更适合提供“已有实例修正上界诊断”，而不是直接新增实例。

- 下一步建议：
  1. 不要先跑 even96，也不要直接跑最终 AP。
  2. 先新增可推理使用的边界质量特征：
     - 超点是否跨越多个二维实例边界。
     - 补全后包围盒体积变化。
     - 补全后连通块数量变化。
     - 颜色、法向和空间连续性。
     - 新增区域是否集中在一个连通超点区域。
  3. 只允许极少数高置信补全：
     - 二维核心精确率的替代特征必须高。
     - 新增点比例不能过大。
     - 补全后不能增加多物体错误合并风险。
     - 原始已有候选本身明显残缺时才允许补全。
  4. 将“已有实例修正诊断”升级成自动候选动作判断：
     - 保留原始。
     - 安全补全。
     - 拒绝补全。
     - 暂时不启用自动替换。
  5. 只有安全补全诊断在 even48 上显示“好于原始的数量明显多于变差数量”，再跑最终 AP。

- 已完成检查：

```bash
python -m py_compile tools/export_mask_graph_proposals.py tools/analyze_applied_mask_graph_candidates.py tools/analyze_mask_graph_trace_relations.py
bash -n tools/run_scannet200_even48_mask_graph_eval.sh
git diff --check
```
