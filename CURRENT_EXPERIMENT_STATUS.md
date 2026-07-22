# OpenYOLO3D 当前实验状态

最后更新：2026-07-22

## 用途

这是唯一的项目状态记录。只保留当前有效结论、已验证实现、运行边界和下一步；已淘汰的参数扫描、重复日志和逐次会话流水不再保留。

## 当前研究定位

目标是写一篇开放词汇 3D 实例分割论文。当前路线是：以 Open-YOLO 3D 的 YOLO-World 与 MVPDist 为快速语义基线，研究二维实例边界感知的 superpoint，以及由 superpoint 驱动的跨视角实例形成。

```text
RGB-D / 位姿 / 点云网格
-> YOLO-World + SAM 高覆盖逐帧实例观测
-> 真实深度约束的二维边界感知 IBSp superpoint
-> Any3DIS 式 SAM2 类别无关 mask 轨迹，并提升到 superpoint
-> Details Matter 式轨迹后 superpoint 共识、迭代合并、去重/包含删除、局部清理
-> Open-YOLO 3D 快速语义投票
-> 仅对低置信、冲突或长尾实例做上下文语义修正
```

核心贡献候选：

1. 多视角二维实例边界和真实深度共同细化三维 superpoint。
2. superpoint 驱动的类别无关跨视角实例形成。
3. 在不替换 Open-YOLO 3D 快速语义头的前提下，提高实例候选质量。

详细研究方向见 `资料/当前基线修改方向.md`；论文依据见 `资料/论文阅读记录.md`。

## 已完成且有效的实现

### 随机场景划分

`tools/generate_random_scene_splits.py` 使用当前 `data/scannet200` 的 312 个场景和 seed `20260718` 生成：

- `output/scannet200/scene_splits/even48.txt`：随机 48 场景方向筛选。
- `output/scannet200/scene_splits/even96.txt`：随机 96 场景扩展集，包含 even48。
- `output/scannet200/scene_splits/odd96.txt`：与 even96 不重叠的随机 96 场景确认集。

文件名沿用 even/odd 只为兼容旧脚本，实际内容是可复现随机划分。

### 官方 ScanNet v2 来源审计（进行中）

当前本地 312 个场景与参考 `scannetv2_val.txt` 完全一致；每个场景都有 RGB、深度、pose、内参、`_vh_clean_2.ply` 和预处理 `.npy`，抽查中 `.npy` 点数与 mesh 顶点数相等。因此现有数据在项目所需的结构与几何对应上可用，但这不能单独证明历史下载来源。

为论文可复现性，已在不覆盖当前数据的独立目录 `/home/jia/Wm/Dataset/scannet_v2_official_audit_3scenes/` 完成官方 ScanNet v2 三场景下载与逐项审计：最新版 label map 与 `scene0019_01`、`scene0084_01`、`scene0084_02` 的 `.sens`、mesh、场景元数据、aggregation、segs 均已齐全。最后的 `scene0084_02.sens` 经 `curl` 断点下载完成，大小校验为 `608020272` bytes。

`tools/audit_official_scannet_v2.py` 的只读审计已在三个场景全部通过，报告为 `docs/diagnostics/official_scannet_v2_audit_3scenes.json`：本地与官方 mesh 的 SHA256 完全相同；`.npy` 的全部顶点 xyz/RGB 与官方 mesh 逐值一致；`.sens` 帧数、内参一致，首/中/末帧的深度逐像素一致、位姿在 `1e-5` 内一致。RGB JPEG 平均绝对像素差为 `1.40--2.87`，符合提取时重新 JPEG 编码，形状一致。结论是当前数据在这三个代表场景已由官方下载样本验证，可继续现有实验；该审计不证明全部 312 个场景的历史来源，因而不替换当前 `data/scannet200` 或声称全量来源已验证。

### 高覆盖逐帧实例观测

`tools/export_dense_frame_instance_observations.py` 已实现并在 GPU 上真实运行。它复用缓存的 YOLO-World 框、SAM 和 mesh 可见性投影，输出：

```text
<output>/<scene>/
  observations.jsonl
  masks/                 # 每个 mask 与对应 3D 点索引
  frame_label_maps/      # 供 IBSp 使用的非重叠整数实例图
  summary.json
```

默认策略：逐帧、检测阈值 0.25、每框前两个 SAM mask、每帧最多 20 mask、最少 8 个可见三维点。帧内仅删除近似同面积的重复 mask；小的嵌套实例在 label map 中优先占用重叠像素。

现有 2D 缓存是旧格式，运行时需显式传 `--allow_legacy_2d_cache`。这只允许读取现有 YOLO-World 缓存，不使用任何 GT 输入。

### IBSp superpoint

`tools/generate_geometric_superpoints.py` 可在 `mesh_normal` 图上使用逐帧 label map 剪掉稳定跨实例边界的图边。新增参数：

```text
--boundary_mask_root <dense output>
--boundary_mask_subdir frame_label_maps
```

指定子目录后，脚本按实际 label-map 文件名取帧，避免 OpenYolo3D 采样帧号 `0,10,...` 与颜色目录 `0..9` 错位。默认旧目录布局仍不受影响。

### 超点后处理原型

`tools/refine_dense_observations_with_superpoints.py` 已实现：

1. 以观测点索引计算各 superpoint 覆盖率，筛出可靠核心。
2. 用跨帧共享核心 superpoint 构建原型类别无关实例轨迹。
3. 合并轨迹、删除近重复轨迹。
4. 对竞争 superpoint 做保守归属；接近并列时从全部实例中删除。
5. 输出独立 `refined_instances.json` 和实例点索引。

这不是最终的 Any3DIS SAM2 跟踪，也尚未实现 Details Matter 完整的可见性归一化共识和迭代细化；它只用于验证 IBSp、候选输出和后处理接口。该工具目前仅作 export-only 诊断，尚未接入 `backprojection_fusion.py` 或 AP。

## 三场景真实 GPU 烟测

环境：`/home/jia/anaconda3/envs/openyolo3d/bin/python`，RTX 4090。测试场景：`scene0011_00`、`scene0077_00`、`scene0608_01`；每场景 10 个实际 OpenYolo3D 采样帧。

输出：

- `output/dense_frame_instance_observations_3scenes_smoke/`
- `output/mesh_normal_ibsp_dense_3scenes_smoke_v2/`
- `output/refined_dense_instances_3scenes_smoke_v2/`

| 场景 | 有效观测 | 平均 label-map 覆盖 | IBSp 剪边 | 最终实例轨迹 |
| --- | ---: | ---: | ---: | ---: |
| scene0011_00 | 39 | 23.34% | 4512 | 4 |
| scene0077_00 | 80 | 54.06% | 1242 | 6 |
| scene0608_01 | 71 | 46.21% | 645 | 8 |

多帧 IBSp 的二维边界观测图边为 `335956 / 194225 / 107182`，证明高覆盖观测确实影响了图结构。超点轨迹能形成多帧实例，如 dining table、printer、guitar；仍有单帧弱实例，因此不能直接进入最终预测。

结论：高覆盖观测 -> IBSp -> 共享核心原型关联/合并/去重/清理已跑通。三场景只证明链路正确，不证明最终 SAM2 跟踪或 ScanNet200 泛化。

## 随机 even48 的首轮 export-only 结果

已对随机 `even48` 完成同一条链路，每场景最多 10 个实际 OpenYolo3D 采样帧，不运行 AP：

- `output/dense_frame_instance_observations_even48_f10/`
- `output/mesh_normal_ibsp_dense_even48_f10/`
- `output/refined_dense_instances_even48_f10/`

| 指标 | 48 场景结果 |
| --- | ---: |
| 实际处理帧 | 475 |
| 有效二维观测 | 3042，均值 63.4/场景 |
| 平均 label-map 覆盖 | 43.47%，中位数 42.49% |
| 有二维边界观测图边 | 6308072 |
| IBSp 剪边 | 39137，均值比例 0.001074 |
| 可靠 superpoint 观测 | 1920 |
| 原始轨迹 | 368 |
| 诊断到的重复轨迹 | 111 |
| 最终实例轨迹 | 290，其中多帧轨迹 243、单帧弱轨迹 47 |

所有 48 场景均完成 dense、IBSp 和轨迹后处理，没有缓存、帧号、投影或网格对齐失败。覆盖率范围为 16.34%--73.83%；低覆盖场景包括 `scene0412_01`、`scene0377_00`、`scene0389_00`。实例数较高的场景包括 `scene0655_01`、`scene0357_00`、`scene0307_02`、`scene0426_02`、`scene0606_00`、`scene0693_02`，应优先检查是否有背景污染或过分裂。

这次结果证明方法链路在随机 even48 上稳定可运行，也显示跨视角形成和去重实际在发生；它**不**证明相对基线 AP 提升。当前没有生成最终预测，也没有接入主融合。

### even48 质量诊断与弱轨迹过滤

新增 `tools/diagnose_dense_ibsp_quality.py`，不使用 GT，自动选择低覆盖和高实例数场景，输出统计表、轨迹清单与 RGB + label-map 叠加图：

- `docs/diagnostics/dense_ibsp_even48_f10_quality/`
- `docs/visual_checks/dense_ibsp_even48_f10_quality/`

审查显示代表帧的二维实例边界整体可用，未发现整帧背景被单一实例 mask 吞没的灾难性污染；主要噪声来自高实例数场景的单帧弱轨迹。对 `singleton_min_confidence` 做仅重跑后处理的扫描：

| 单帧阈值 | 总实例 | 多帧实例 | 单帧弱实例 |
| --- | ---: | ---: | ---: |
| 2.0 | 290 | 243 | 47 |
| 3.0 | 262 | 243 | 19 |
| 4.0 | 250 | 243 | 7 |
| 5.0 | 245 | 243 | 2 |

选择 `4.0`：删除 40 个弱单帧轨迹，同时不损失任何多帧轨迹；`5.0` 过于接近只保留多帧实例。`tools/refine_dense_observations_with_superpoints.py` 的默认 `singleton_min_confidence` 已从 `2.0` 改为 `4.0`，正式输出为 `output/refined_dense_instances_even48_f10_singleton4/`。

### even48 f30 多视角扩展

已将同一随机 `even48` 的每场景上限扩大到 30 个实际采样帧，并完整执行 dense -> IBSp -> refined instances，仍为 export-only、不运行 AP：

- `output/dense_frame_instance_observations_even48_f30/`
- `output/mesh_normal_ibsp_dense_even48_f30/`
- `output/refined_dense_instances_even48_f30_singleton4/`

| 指标 | f10 | f30 |
| --- | ---: | ---: |
| 实际处理帧 | 475 | 1411 |
| 有效二维观测 | 3042 | 8416 |
| 平均 label-map 覆盖 | 43.47% | 42.50% |
| 有二维边界观测图边 | 6308072 | 13896146 |
| IBSp 剪边 | 39137 | 59690 |
| 可靠 superpoint 观测 | 1920 | 5109 |
| 原始轨迹 | 368 | 720 |
| 诊断到的重复轨迹 | 111 | 243 |
| 最终实例轨迹 | 250 | 464 |
| 多帧轨迹 | 243 | 447 |
| 单帧弱轨迹 | 7 | 17 |
| 每个最终实例的平均支持帧 | 5.46 | 7.71 |

48 个场景均成功完成。f30 的平均单帧覆盖率没有被人为抬高，但多视角观测、可靠超点证据和多帧轨迹显著增加；这是扩大观测覆盖带来稳定实例支持的正向结构信号。单帧弱轨迹比例略有增加（7/250 -> 17/464），后续评测适配时应保持当前阈值并单独报告该风险。该结果仍不能替代与基线的 AP 对比。

### even48 的首轮 AP：refined 候选补充

已新增 `tools/export_refined_dense_candidates.py`，把 refined instances 适配为现有融合候选格式：保留每个实例的三维点索引，以 Open-YOLO 3D prompt 索引作为类别，并以支持观测的 `YOLO score x SAM score` 均值作为 0--1 分数。该适配不读取 GT，也不改动基线掩码。

在相同随机 `even48`、相同二维缓存和默认统一评分下，先运行基线，再运行“基线 + f30 refined 候选”。输入的 424 条候选经过至少 100 点、至少 2 视角、与基线 IoU 0.30 和候选间 IoU 0.50 的过滤，实际加入 158 条；264 条因与已有掩码重叠而跳过，2 条低分跳过。

| 指标 | 基线 | 基线 + refined | 差值 |
| --- | ---: | ---: | ---: |
| AP | 0.258700 | 0.258875 | +0.000175 |
| AP50 | 0.336783 | 0.337401 | +0.000619 |
| AP25 | 0.384666 | 0.389003 | +0.004337 |

结果文件：

- `output/scannet200/dense_ibsp_even48_ap/baseline.csv`
- `output/scannet200/dense_ibsp_even48_ap/refined_dense.csv`
- `output/scannet200/dense_ibsp_even48_ap/reports/refined_dense.json`

结论：当前实现对宽松 IoU 的召回有小幅正信号，但总体 AP 增益极小，不能视为可靠提升，也不进入 `even96/odd96`。主要瓶颈是 refined 轨迹与原始掩码重叠时只会被作为重复候选跳过，尚未测试“以 refined 超点边界替换或局部修正已有实例”的作用。

### 强 SAM-fused + BPR 基线的原始 superpoint / f30 IBSp 对照

已在当前随机 `even48` 完成受控 B/C AP 对照，脚本为 `tools/run_scannet200_even48_ibsp_control.sh`，输出为 `output/scannet200/ibsp_control_even48_20260719/`。两轮均使用同一批 SAM-fused + BPR 候选、同一融合阈值、同一评分模式和同一 scene split；C 唯一改变是通过 `--processed_scene_root` 读取 `output/mesh_normal_ibsp_dense_even48_f30/`。脚本在运行前断言 48 个 `.npy` 的非第 9 列完全相同，而第 9 列正是评估读取的 superpoint id。

| 配置 | AP | AP50 | AP25 |
| --- | ---: | ---: | ---: |
| B：历史强 SAM-fused + BPR + 原始 superpoint | 0.264689 | 0.348004 | 0.395579 |
| C：B，仅替换为 f30 IBSp | 0.266595 | 0.346625 | 0.395200 |
| C - B | +0.001906 | -0.001379 | -0.000378 |

两轮都加载 7460 条候选；B 实际接入 295 条，C 接入 289 条。IBSp 的超点精炼输出点数从 178827 降至 158399，选中 segment 从 1326 降至 919，说明二维边界约束确实改变了候选几何，而不是空替换。结论是：IBSp 在强基线上有小幅 AP 正信号，但 AP50/AP25 同时轻微下降，尚不能作为独立显著提升或进入 `even96/odd96` 的依据。

## 当前 SAM2 / Any3DIS 双轮烟测

`tools/export_any3dis_sam2_tracks.py` 已从“关键视角 + 三个正点”扩展为：先在关键帧用 SAM2 image predictor 生成初始 mask，再作为 video predictor 的 mask prompt 前后向传播；当同一可靠 seed superpoint 在超过 7 帧的不可见间隔后重现时，再注入该帧的三维投影正点。三场景第一轮 24 条轨迹全部采用 image-mask 初始化，`scene0084_01` 有一条轨迹触发重现提示。

已实现真实的无 GT 迭代采样闭环：

```text
第一轮可靠超点种子
-> SAM2 轨迹 -> 动态超点优化 / 共识筛选 / 轨迹后清理
-> 可靠但未被清理实例认领的超点
-> 第二轮空间分散种子 -> SAM2 轨迹
-> 多轮轨迹连续重编号合并 -> 跨轮 Details Matter 式后处理
```

新增 `tools/select_uncovered_any3dis_superpoints.py` 和 `tools/merge_any3dis_rounds.py`。后者只重编号和整合已有元数据与路径，不重跑 SAM2；因此现有后处理可跨轮做合并、包含删除和竞争超点清理。

新版输出均为 30 个 f30 采样帧、每轮每场景最多 8 个种子：

- 第一轮：`output/sam2_any3dis_v2_smoke3_20260720/`、`output/sam2_superpoint_lift_v2_smoke3_20260720/`、`output/sam2_details_postprocess_v2_smoke3_20260720/`，清理后分别保留 `6 / 7 / 5` 个实例；仍可探索可靠超点 `238 / 44 / 62` 个。
- 第二轮：`output/sam2_any3dis_v2_round2_smoke3_20260720/`、`output/sam2_superpoint_lift_v2_round2_smoke3_20260720/`。
- 两轮合并并再次清理：`output/sam2_any3dis_v2_merged_round12_smoke3_20260720/`、`output/sam2_superpoint_lift_v2_merged_round12_smoke3_20260720/`、`output/sam2_details_postprocess_v2_merged_round12_smoke3_20260720/`。输入候选为 `13 / 13 / 11`，输出为 `11 / 12 / 11`；跨轮发生 `2 / 1 / 0` 次合并，`scene0019_01` 清理了 23 个竞争超点。下一轮可探索种子仍为 `202 / 28 / 39`。

这些是链路和规则的 smoke 结果，尚未生成 Open-YOLO/MVPDist 语义标签、未接入强 B 基线融合，也未运行 AP，不能表示性能提升。

## 当前 even48 的 Alpha-CLIP 语义校正对照

已在当前随机 `even48`（48/48 场景严格匹配）重新导出 Alpha-CLIP 多视角对象特征：`23508` 个候选记录、`68784` 个 crop。旧 Alpha-CLIP 缓存只与当前 split 重叠 8 个场景，保留作历史诊断，**不得引用其 AP**。

本轮以同一个 `evaluate_multiview_object_clip_correction.py` 入口、相同候选和阈值运行对照；唯一变量为是否采用 Alpha-CLIP 的低置信类别修正。无语义修正为 `AP 0.261404 / AP50 0.352537 / AP25 0.404424`，Alpha-CLIP 为 `AP 0.263244 / AP50 0.354621 / AP25 0.406099`，增量分别为 `+0.001840 / +0.002084 / +0.001675`。Alpha-CLIP 在 42 个场景中替换了 2182 个候选类别。

结论：Alpha-CLIP 在该受控语义入口中有小而一致的正向信号，可保留为 MVPDist/YOLO 语义后的可选低置信校正模块；该入口的绝对值不能与强 B 的 `0.264689` 混作同一主表，也不能说明 SAM2 候选已经提升。下一步必须把同一语义规则接到 SAM2/Details Matter 候选上做受控比较。

## SAM2 候选的 GT 仅离线诊断（3 场景）

为定位瓶颈，新增 `tools/diagnose_sam2_refined_instances_gt.py`；它要求显式 `--allow_gt_diagnostics`，只读取 GT 生成 JSON/CSV/可视化，绝不向候选生成、种子选择、合并、类别打分或推理输出提供 GT。最终报告位于 `docs/diagnostics/sam2_details_gt_diagnostic_smoke3_20260720/`。

两轮 SAM2 + Details Matter 的最终输出为 34 个候选，对应 79 个有效 ScanNet200 GT 实例。类别无关几何 oracle 的 GT recall 为 IoU `>=0.25: 0.1519`、IoU `>=0.50: 0.1013`。清理前 lift 为 37 个候选，两个 recall 完全相同，说明当前 Details Matter 合并/去重没有造成已覆盖 GT 的下降；主要瓶颈在于前端种子覆盖不足与部分轨迹边界失控，而不是后处理删掉了正确实例。

诊断可视化确认两类错误并存：存在高精度但低覆盖的碎片（例如 shower curtain），也存在轨迹吞入背景/邻近实例的扩张（例如 armchair）。更根本的是每场景两轮仅尝试 16 个种子，而仍有大量可靠超点未被采样。新增 `tools/select_uncovered_any3dis_superpoints.py` 的 GT-free baseline-novel 过滤后，三场景剩余可用新颖种子为 `87 / 14 / 19`；被 Mask3D 大面积覆盖而排除的种子为 `115 / 14 / 20`，说明后续 SAM2 应针对基线未覆盖区域，而不能盲目继续对自身未认领超点采样。

已新增 `tools/export_sam2_refined_backprojection_candidates.py` 并成功将 34 个 refined instances 导出为现有 fusion schema（`source_kind=sam2_details`），但仅用当前 YOLO 2D box 与 SAM2 mask 的语义桥接时，12 个几何 IoU>=0.25 候选只有 4 个与 GT 类别完全一致。该接口已验证，**当前输出不得直接用于 even48 AP**；下一步要以 MVPDist 级多视角语义归属替换这个桥接，并先提高 baseline-novel SAM2 的覆盖和边界质量。

### baseline-novel 第三轮与融合接口的三场景验证

在不使用 GT 的前提下，第三轮改为只从强 B 未充分覆盖的可靠 IBSp 中选种子，并把每场景预算提升到 `16 / 14 / 16`。新轨迹与前两轮合并、再经同一 Details Matter 清理后，候选从 34 增至 54。GT-only 几何 oracle recall 随之从 `0.1519 / 0.1013` 提升到 `0.2532 / 0.1646`（IoU `>=0.25 / >=0.50`），因此“baseline-novel 覆盖优先”有明确的候选几何正信号。

新增候选已通过现有 `backprojection_candidates` 融合接口，且修复了评测器：基线维持原始分数阈值，附加候选改以最终语义分数过滤，避免临时 YOLO 框分数在 Alpha-CLIP 重分类前错误删除候选。为可复现性，三场景另建了带签名的 YOLO-World 缓存 `output/scannet200/bboxes_2d_sam2_smoke3_20260720/`；旧 `bboxes_2d` 缓存为 legacy 格式，只可用于历史结果，不能用于当前入口。

用 Alpha-CLIP 作为唯一语义头的融合 AP 没有提升：在相同三场景、新 2D 缓存和 `score_threshold=0.02` 下，纯基线与融合均为 `AP 0.373765 / AP50 0.492533 / AP25 0.515942`。离线语义诊断表明，54 个候选中 21 个 IoU>=0.25、13 个 IoU>=0.50；这些候选的 YOLO 临时标签精确率仅 `22.22% / 18.18%`，Alpha-CLIP top-1 均为 `0%`，平均 top-1 概率约 `0.07`。

已新增 `tools/export_sam2_refined_mvpdist_candidates.py`，直接复用 Open-YOLO 3D 的 label-map/MVPDist 多视角投票，为 54 个 refined instances 产生一个类别和分数。对 IoU>=.25/.50 的候选，MVPDist 精确类别率升至 `33.33% / 38.46%`，优于 YOLO 和 Alpha-CLIP；但严格 `.20` 阈值 AP 对照仍为 `AP +0.000000 / AP50 +0.000000 / AP25 +0.002199`。故 MVPDist 是正确的主语义路径，但当前仍需 GT-free 的候选质量/边界筛选来提高高 IoU 候选的比例；在用户明确授权前，这些三场景结果本身不足以进入 even48。

### 三场景 GT-free 质量门控原型

新增 `tools/filter_sam2_refined_instances_gtfree.py`，只使用 mesh 面片、IBSp superpoint、refined instances 以及可选 MVPDist 候选元数据，不读取 GT。它可按 mesh superpoint 连通块裁掉离散小块，并按无 GT 的结构阈值过滤异常候选，输出新的 refined root。

三场景 smoke 使用保守结构门控：`max_scene_point_fraction=0.10`、`max_superpoints=40`、`min_component_points=100`、`min_component_point_fraction=0.02`。输出为：

- `output/sam2_details_postprocess_v3_merged_round123_quality_guard_s01_sp40_smoke3_20260720/`
- `output/sam2_details_mvpdist_candidates_v3_merged_round123_quality_guard_s01_sp40_smoke3_20260720/`
- GT-only 诊断：`docs/diagnostics/sam2_details_gt_diagnostic_v3_merged_round123_quality_guard_s01_sp40_smoke3_20260720/`

门控将候选从 `54` 降至 `48`，删除 `6` 个异常候选。离线 GT 诊断仅用于验证，oracle recall 保持 `0.2532 / 0.1646`（IoU .25/.50）不变；candidate precision 从 `0.3889 / 0.2407` 提升到 `0.4375 / 0.2708`。MVPDist 在几何合格子集上的类别准确率未变化：IoU>=.25 为 `0.3333`，IoU>=.50 为 `0.3846`。结论是该门控能无 GT 地删掉一部分明显泄漏/背景候选，且不牺牲当前三场景 oracle recall；但它没有解决 MVPDist 语义和剩余混合候选问题。若无用户明确授权，仍不足以进入 even48。

### 策略与实现审计（2026-07-20）

当前链路不是 Details Matter 的完整复现，而是有意与 Any3DIS/SAM2 组合的裁剪实现。已完成：IBSp 可靠种子与关键视角、SAM2 图像 mask 初始化和双向传播、长不可见间隔的重现正点、逐帧 superpoint 回投及 Any3DIS 式 mask 贪心优化、可见性共识、跨轮迭代合并、包含删除、竞争 superpoint 清理和无 GT 质量门控。未发现足以单独解释低增益的显性运行、投影或评测接口错误。

但仍缺少两个与当前“混合候选/边界泄漏”错误直接相关的环节：原 Details Matter 在提升到三维前会消除同一帧多个二维实例 mask 的重叠区域；当前 SAM2 种子独立传播，只在三维 superpoint 层处理竞争。原方法还以逐帧独立二维观测及逐帧集合交并比形成或校正轨迹；当前主要使用单个关键帧初始化后的连续 SAM2 传播，未实现独立重观测确认。默认只用正点、只按 SAM2 初始分选单个初始 mask、且只采样 30 个稀疏帧，也可能放大扩张和碎片。故当前结果不能说明 SAM2/Details Matter 思想无效，只能说明该组合版本的候选质量仍不足。

用户已明确要求以当前冻结版本进行一次随机 `even48` 受控验证：三场景只用于定位，不能代替总体 AP。该运行只比较强基线 B 与强基线 B + 三轮基线未覆盖区域的 SAM2/Details Matter + 无 GT 质量门控 + MVPDist，不在 even48 上扫描参数；长任务只检查启动、中段/异常和完成三个状态。

已检查现有 `output/`：没有上述完整组合在 `even48` 的输出或评测报告。已有 `even48` 的 IBSp 原型轨迹、SAM 融合或其他候选图试验均不是当前三轮 SAM2 + 后处理 + 质量门控 + MVPDist 配置，不能读取后替代本次受控验证，也不能混入比较。

第一次后台启动在工具会话退出后被终止，日志只写入“第一轮 SAM2 轨迹”，没有形成有效中间产物或评测结果；它不得视为已运行。随后已用独立会话重新启动，实际根目录为 `output/sam2_details_even48_frozen_20260720_run1/`，最终对照报告写入 `output/scannet200/sam2_details_even48_frozen_20260720_run1_eval/`。48 个场景均已完成三轮轨迹、提升到三维、后处理、质量门控、MVPDist 和两次评测。实际 SAM2 速度约为 3 秒/轨迹。

此次完整运行共得到 `839` 个后处理实例，经无 GT 质量门控保留 `739` 个，并成功导出 `738` 个 MVPDist 候选。强基线与“强基线 + SAM2”报告均为 `AP 0.261404 / AP50 0.352537 / AP25 0.404424`。但该持平结果**不能作为有效负结论**：启动脚本复用了强基线的统一原始候选分数阈值 `0.50` 和重叠约束，738 个 SAM2 候选中只有 `3` 个同时通过；其余候选在融合前已被过滤。下一步不重跑 SAM2 生成，而是只修正融合层的按来源分数筛选，使强基线维持原阈值、SAM2 候选按其 MVPDist/几何融合分数的独立规则进入，再对同一 48 场景重新评测。

融合层修正：`evaluate_multiview_object_clip_correction.py` 已增加 `--backprojection_source_min_scores` 并传入已有融合函数；`tools/run_scannet200_even48_sam2_details_fusion_eval.sh` 只读取冻结输出重评测，不重新执行 SAM2。规则固定为 `sam_fused=0.50,bpr=0.50,sam2_details_mvpdist=0.00`：历史强基线的分数、排序、重叠过滤和来源上限均不变，SAM2 仍需通过既有实例重叠、superpoint 细化、几何优先级和每场景总预算，才可占用最多五个额外位置。

历史强 B/C 的 `0.264689 -> 0.266595` 不能与当前 `0.261404` 作绝对比较。B/C 由 `run_evaluation.py` 的原强基线入口产生，唯一变量是 `.npy` 第 9 列 superpoint id；当前数值来自 `evaluate_multiview_object_clip_correction.py` 的无语义修正入口，并显式写入空语义特征关闭类别校正。它与此前 Alpha-CLIP 对照中的无修正基线相同，不是当前改动使 B/C 降低。当前的有效结论只能来自同一入口内“强基线”与“强基线 + SAM2”的重评测差值；若需 IBSp 的绝对主表，须在 B/C 的 `run_evaluation.py` 配置内另行适配新增候选。

来源门槛修正后的 48 场景重评测已完成，报告为 `output/scannet200/sam2_details_even48_frozen_20260720_run1_source_min_v1_eval/`：强基线为 `AP 0.261404 / AP50 0.352537 / AP25 0.404424`，加入 SAM2 后为 `AP 0.261435 / AP50 0.352678 / AP25 0.404573`，增量为 `+0.000031 / +0.000141 / +0.000149`。这是有效但极小的正信号，不能作为方法提升或进入 even96/odd96 的依据。类别 AP 的变化仅出现在 `monitor`，表明当前有效增益高度集中，尚非普遍的候选质量改善。

原因不是新增候选仍被 `0.50` 错误过滤，而是两道保守筛选共同收窄了有效集合：738 条 MVPDist 候选中，分数达到后续最终 `score_threshold=0.20` 的有 229 条；按导出时已有实例重叠字段，二者同时满足的只有 31 条，实际融合还会继续经过 superpoint 细化和新候选去重。不能为了提高数量直接降低最终分数门槛或放宽重叠限制，因为那会在没有校准语义的情况下提高误检。后续应优先补齐候选形成的同帧二维 mask 重叠消除和独立重观测确认，并以 MVPDist 置信度、间隔与几何一致性做候选级软排序；修复 `evaluate_multiview_object_clip_correction.py` 使后续报告同时保留 `backprojection` 与 `clip` 明细，避免再次丢失实际接入统计。

已开始补齐第一个前端缺口：`tools/lift_sam2_tracks_to_superpoints.py` 新增可选的 `--same_frame_overlap_cleanup`。它在同一轮全部轨迹提升到三维前，对每一帧中被至少两条轨迹覆盖、且面积达到 `--same_frame_overlap_min_pixels`（默认 32）的像素从所有相关轨迹中删除，不按分数武断归属；每条 lifted record 和场景摘要均记录移除像素数与受影响帧数。冻结运行脚本的三轮提升均会在下一次新轨迹运行时显式启用。旧的冻结输出保持不变。合成断言验证了“达到门槛的交叠从双方移除、低于门槛的单像素交叠保留”；在 `scene0146_01` 第一轮旧轨迹上只读统计到 5 个受影响帧、29812 个待移除歧义像素，表明该模块并非空操作。当前 `openyolo3d` 环境没有 `pytest`，故该断言以同环境的直接 Python 执行完成，并已通过 `py_compile`、`--help` 与 `bash -n`。

评测入口审计发现一项必须先修正的混合链路：历史 B/C 的 `run_evaluation.py` 支持 `--processed_scene_root`，C 因而在候选 superpoint 细化时读取 f30 IBSp；此前 `evaluate_multiview_object_clip_correction.py` 没有该参数，虽然 SAM2 轨迹和 MVPDist 候选来自 f30 IBSp，却在最终融合时固定读取 `data/scannet200/...npy` 的原始 superpoint。故 `0.261404 -> 0.261435` 既不等于 B 也不等于 C。现已为该入口增加 `--processed_scene_root`，并让仅融合重评测脚本默认传入 `output/mesh_normal_ibsp_dense_even48_f30`。修正后必须重做一次不运行 SAM2 的 48 场景融合评测，才是“f30 IBSp + SAM2/Details Matter”一致的数据流对照。

系统自查的三步已完成。无 GT 接口契约审计 `docs/diagnostics/sam2_fusion_contract_even48_20260721/summary.json` 覆盖 48 场景，确认 f30 仅改变第 9 列、三类候选的点索引和类别映射合法，候选数为 `SAM-fused 275 / BPR 895 / SAM2 738`，无 error/warning。仅离线 GT-only 账本 `docs/diagnostics/sam2_mvpdist_fusion_ledger_even48_20260721/summary.json` 显示 738 条 SAM2 候选中 IoU>=.25/.50 为 `235/151`，但几何错误仍以弱几何、碎片和背景候选为主；MVPDist 类别精确率为全部候选 `.299`、IoU>=.25 `.443`、IoU>=.50 `.457`。最重要的诊断是：分数>=.20 的 229 条候选平均 IoU `.463`，而仅按“低既有实例重叠”保留的 330 条平均 IoU `.063`；二者交集 31 条平均 IoU `.120`。这表明“只追加未覆盖区域”系统性排除了较好几何候选，后续必须研究已有实例局部修正/替换的保守条件，而不是盲目降低阈值。

使用历史 `run_evaluation.py`、f30 IBSp、统一评分、相同强基线候选和同一 even48 的主对照已完成，脚本为 `tools/run_scannet200_even48_ibsp_sam2_control.sh`，输出为 `output/scannet200/ibsp_sam2_control_even48_20260721/`。f30 强基线精确复现历史 C：`AP 0.266595 / AP50 0.346625 / AP25 0.395200`；只追加现有 SAM2 MVPDist 候选后为 `AP 0.266907 / AP50 0.347182 / AP25 0.396167`，增量 `+0.000312 / +0.000557 / +0.000967`。历史候选实际接入数保持 `289`，SAM2 额外接入 `291`。这是当前冻结版本第一个可与 B/C 直接比较的有效正向结果，但幅度仍小，且该 even48 已被多轮诊断使用；不能据此宣称泛化。新实现的同帧二维 mask 重叠消除尚未进入该结果，必须先在三场景检查候选变化后再冻结一次新版本验证。

### 基础 Mask3D 与 SAM2 的 GT-only 互补性诊断（2026-07-21）

新增 `tools/diagnose_baseline_sam2_complementarity_gt.py`，必须显式传入 `--allow_gt_diagnostics`，只在事后逐 GT 实例计算基础 Mask3D mask 和 SAM2 候选各自的最佳 IoU，输出 `CSV/JSON`，绝不回流到候选、融合、打分或阈值。完整 `even48` 报告在 `docs/diagnostics/baseline_sam2_complementarity_even48_20260721/`，覆盖 1113 个有效 GT 实例。

结果否定了“SAM2 完全不能补出 Mask3D 漏检实例”的说法：IoU>=.25 时有 `19` 个实例为“Mask3D 漏、SAM2 覆盖”（`1.71%`），IoU>=.50 时仍有 `16` 个（`1.44%`）；分别还有 `29/21` 个实例中 SAM2 的 IoU 至少比 Mask3D 高 `.10`。但更多候选与 Mask3D 覆盖同一实例：IoU>=.25 为 `217` 个、IoU>=.50 为 `135` 个；而仅 Mask3D 覆盖、SAM2 未覆盖的实例仍为 `811/802`。因此当前小增益同时有两层原因：SAM2 的真正新增覆盖有限，且当前“低既有实例重叠才追加”的规则会拒绝大量同一对象上的较好 SAM2 边界候选。下一步应保留 SAM2 作为类别无关候选生成器，不用 Details Matter 替换它；应补齐 Details Matter 式的同帧重叠消解和独立重观测确认，并研究仅在高置信条件下局部修正/替换已有实例。

### Mask3D 漏检实例的种子覆盖诊断（2026-07-21）

新增严格 GT-only 工具 `tools/diagnose_sam2_seed_coverage_gt.py`，报告为 `docs/diagnostics/sam2_seed_coverage_even48_20260721/`。它把每个基础 Mask3D 漏检 GT 实例逐层定位为“无可靠超点”“有可靠超点但未采样”“已采样但轨迹或三维提升未恢复”或“SAM2 已覆盖”；GT 只写入离线 `CSV/JSON`，绝不回流推理。

IoU>=.25 的 85 个 Mask3D 漏检实例中，`38` 个（`44.7%`）无可靠超点，`22` 个（`25.9%`）已采样但未恢复，`6` 个（`7.1%`）有可靠超点却未采样，`19` 个已被 SAM2 覆盖。IoU>=.50 的 176 个漏检实例中，对应为 `65/64/31/16`。因此不能把低互补率简单归因为“只跑了不够多的 SAM2 轨迹”：当前三轮预算确实被硬截断为每场景最多 `8+8+16` 个种子，且没有像 Any3DIS 原文那样迭代至无空闲超点；但在 IoU>=.25 下，直接因未采样漏掉的实例只占少数。优先问题是可靠超点定义/几何边界和多视图可见性使近半漏检实例没有可用种子，其次是已采样对象在 SAM2 传播、二维重叠、提升与共识筛选中失败。增加种子预算只能作为受控消融，不能替代前两项修正。

Any3DIS 式无约束二元优化并非缺失：`tools/lift_sam2_tracks_to_superpoints.py` 的 `--mask_optimization any3dis_dp` 已以“逐帧加入该帧全部候选超点或保持现状”的贪心动态过程，计算全帧投影落在 mask 内减去落在 mask 外的目标，近似原文的求解器。但当前实现在优化后额外施加 Details Matter 式多视图共识过滤，且候选超点先受逐帧覆盖率阈值限制，故不是原文优化的逐字复现。迭代三维物体采样已实现为三轮未认领/基线未覆盖超点采样，但固定停止于第三轮，不是原文“直到没有空闲超点”的完整实现。

### 无可靠超点原因与独立重观测（2026-07-21）

进一步 GT-only 诊断 `tools/diagnose_unreliable_superpoints_gt.py`（报告 `docs/diagnostics/unreliable_superpoints_even48_20260721/`）显示，IoU>=.25/.50 下无可靠 IBSp 的 `38/65` 个 Mask3D 漏检实例全部属于“可见帧不足”，没有“超点规模不足”；其最佳超点 GT 纯度平均约为 `91.5%/91.0%`，仅 `3/6` 个实例低于 `50%`。这说明在**当前 f30 IBSp 已重分割完成之后**，可见性门槛又排除了许多仍有对象内部分的超点。2026-07-22 的原始/IBSp 全链路对照进一步证明，边界覆盖更早已在 f30 重分割下降；故可见性自适应是第二步，第一步必须先做 IBSp 基线粒度对齐。

已在 `tools/export_any3dis_sam2_tracks.py` 实现独立图像预测器重观测：每隔指定采样帧，使用 seed 在当前帧的独立 SAM2 图像预测 mask 与视频传播 mask 计算一致 IoU，保留完整原始轨迹和逐帧审计记录。`lift_sam2_tracks_to_superpoints.py` 的确认器不再硬删除低一致帧，而是以 `--reobservation_rejected_frame_weight` 软降权其超点支持和多视图优化贡献。三场景首轮真实 GPU smoke（24 条轨迹）产生 68 次重观测，51 次通过、17 次否决，平均一致 IoU `.637`。硬删除会使 IoU>=.50 的离线召回从 `.0635` 降至 `.0476`；软降权 `.50` 保持 `21` 个实例及 `.0794/.0635` 的 IoU>=.25/.50 召回，点数仅 `99114 -> 98749`。因此下一冻结版本采用软降权，不把该三场景结果解释为性能提升。

新完整脚本为 `tools/run_scannet200_even48_sam2_details_reobserve.sh`，默认输出 `output/sam2_details_even48_reobserve_20260721/`。它通过环境变量调用旧脚本的固定其余配置，并显式启用 `stride=5`、一致 IoU `.30`、否决帧权重 `.50`。

该新冻结 even48 已于 2026-07-21 完成，日志为 `output/sam2_details_even48_reobserve_20260721/driver.log`，报告目录为 `output/scannet200/sam2_details_even48_reobserve_20260721_eval/`。三轮轨迹、三维提升、后处理、无 GT 质量门控、MVPDist 导出和两次评测均正常结束。此入口的强基线为 `AP 0.261404 / AP50 0.352537 / AP25 0.404424`，加入同帧重叠消解和独立重观测软降权后的 SAM2 候选为 `AP 0.261404 / AP50 0.352537 / AP25 0.404511`：AP、AP50 不变，AP25 仅 `+0.000087`，没有可测总体提升。由于该运行脚本尚未向 `evaluate_multiview_object_clip_correction.py` 传入 f30 `--processed_scene_root`，它仍是混合 superpoint 特殊入口，不能同历史 B/C 主表或 `0.266907` 直接比较；不得重跑轨迹，应在需要时仅使用本次已导出的 MVPDist 候选完成 f30 一致的只读融合评测。

为防止后续重复该入口错误，`tools/run_scannet200_even48_sam2_details_frozen.sh` 已补传 `--processed_scene_root "$SUPERPOINT_ROOT"`；其派生的未来冻结运行将统一使用 f30 IBSp。原始 superpoint/f30 的双跑只保留给历史 B/C 的 IBSp 单变量对照，普通方法验证一律直接用 f30。该脚本修复不改写已完成 reobserve 的输出或数值。

## 下一步

1. 不重跑已完成的三轮轨迹；先用本次已导出的 MVPDist 候选，在 f30 一致主入口完成一次只读融合评测，记录 AP、AP50、AP25、附加候选数量和被重叠过滤数量。
2. 之后先做 48 场景 GT-only 定位，而非立刻增加模型：一项检查 YOLO-World 对 Mask3D 漏检实例的二维框覆盖，决定是否值得引入 Grounded-SAM/YOLOE；另一项量化 SAM2 候选对既有 Mask3D 的局部补全/裁剪 oracle 上界，并检验多视图、深度、superpoint 和语义特征能否在无 GT 条件下区分收益与恶化。仅当相应诊断通过，才实现唯一的一项后续方法。

f30 一致的只读融合评测已于 2026-07-21 完成：`RUN_DIR=output/sam2_details_even48_reobserve_20260721 EVAL_DIR=output/scannet200/sam2_details_even48_reobserve_20260721_f30_eval bash tools/run_scannet200_even48_sam2_details_fusion_eval.sh`。报告目录为 `output/scannet200/sam2_details_even48_reobserve_20260721_f30_eval/`。它只读取已完成 reobserve 的 `mvpdist_candidates`，没有重新运行 SAM2、轨迹、提升、后处理、质量门控或 MVPDist。强基线为 `AP 0.263050 / AP50 0.350814 / AP25 0.404136`，加入 SAM2 后为 `AP 0.263049 / AP50 0.350813 / AP25 0.404653`，差值为 `-0.000001 / -0.000001 / +0.000517`。强基线实际接入 `SAM-fused 157 + BPR 132 = 289` 条，本次 SAM2 有 `236` 条全部接入，因而该无净 AP 收益不是来源分数过滤问题。此入口已统一 f30 superpoint，但评测器仍是 `evaluate_multiview_object_clip_correction.py`，不替代历史 B/C 的 `run_evaluation.py` 绝对主表；在自身受控入口中，它构成“同帧歧义消解 + 独立重观测软降权无总体 AP 正向”的有效负结果。下一步执行两个 48 场景 GT-only 定位诊断，不增加模型。

### 48 场景决策诊断（2026-07-22，已完成）

已新增两个严格只读、GT-only 工具，并通过 `py_compile`、`--help`、`bash -n`、`git diff --check` 以及单场景冒烟。它们均要求显式 `--allow_gt_diagnostics`，GT 不会进入推理、候选、融合、打分或阈值。单场景二维覆盖投影实际耗时约 47 秒，局部修正约数秒。

完整任务命令为 `bash tools/run_scannet200_even48_post_sam2_diagnostics.sh`，输出为 `docs/diagnostics/post_sam2_decision_even48_20260722/`。该命令按顺序执行：

1. `tools/diagnose_yoloworld_mask3d_missed_coverage_gt.py`：仅看 f30 前 30 个采样帧，以“同类别 YOLO-World 框覆盖可见 GT 点的比例”而非框 IoU 判断二维观测；默认要求至少 2 帧、每帧覆盖至少 `.50`，用于决定是否有必要比较 Grounding DINO、YOLOE 或 Grounded-SAM。
2. `tools/diagnose_mask3d_sam2_local_correction_gt.py`：对每条已导出的 SAM2 候选及其最高重叠 Mask3D 锚点，比较保留、并集补全、交集裁剪的 oracle IoU，并按 `support_score`、候选语义分数、既有重叠、已覆盖点比例和轨迹数进行无 GT 特征分组，用于决定是否值得实现 MV3DIS 式局部修正。

完整任务已经正常结束，报告为 `docs/diagnostics/post_sam2_decision_even48_20260722/yoloworld_coverage/summary.json`、`docs/diagnostics/post_sam2_decision_even48_20260722/local_correction/summary.json`、`docs/diagnostics/post_sam2_decision_even48_20260722/local_correction/univariate_gate_scan.json`，过程日志为同目录 `driver.log`。

二维覆盖结论：在 IoU>=`.25` 的 `85` 个 Mask3D 漏检实例中，仅 `18` 个（`21.18%`）有可靠 YOLO-World 二维框；其余 `67` 个中，`43` 个没有任何满足最小可见点数的 f30 帧、`17` 个有可见帧但没有同类框、`7` 个只匹配一帧。IoU>=`.50` 时为 `45/176`（`25.57%`）可靠，其余 `131` 个中 `87/34/10` 分别属于无可用帧、可见但无框、仅一帧。所有 GT 类别均在当前提示词表内。因此当前首要瓶颈是 f30 的可见性/观测次数，而不是类别词表或单纯二维检测器能力；暂不优先引入 Grounding DINO、YOLOE 或 Grounded-SAM。

局部修正结论：`583` 条 SAM2 候选中有 `365` 条与 Mask3D 锚点形成局部重叠对。保持原 mask 是 oracle 最优动作 `273` 次（`74.8%`），并集 `75` 次、交集 `17` 次；只有 `54` 条（`14.79%`）的非保留动作可改善至少 `.02` IoU，而 `111` 条（`30.41%`）的并集和交集都至少恶化 `.02`。改善组的 `support_score` 均值虽较高（`89.34` 对 `65.90`），但同集单变量扫描中，满足最小 19 条样本的最高改善精度仅为 `support_score >= 236.65` 的 `7/19=36.8%`，没有可直接部署的无 GT 选择器。故暂不实现 MV3DIS 式自动局部补全/裁剪；原始 Mask3D 继续作为不可自动替换的回退。

本段原先建议优先做无 GT 的可靠 superpoint 可见性自适应；该优先级已被随后完成的原始/IBSp 覆盖对照修正。可见性自适应仍要做，但只能在 IBSp 基线粒度对齐确认后进行；在此之前不增加二维大模型、不重跑 SAM2、不实现局部修正。

### 超点链路复核与原始/IBSp 覆盖对照（2026-07-22）

阅读本地 `Landrieu_Large-Scale_Point_Cloud_CVPR_2018_paper.pdf`、本地 OVSeg3R 原文、ScanNet `Segmentator` 源码后，新增严格 GT-only 账本 `tools/diagnose_superpoint_pipeline_gt.py`，报告在 `docs/diagnostics/superpoint_pipeline_even48_20260722/`。它依次比较原始 ScanNet 第 9 列 superpoint、当前 f30 IBSp、可靠种子、实际采样和当前 SAM2 候选；GT 从不回流。

结论改变了后续优先级。原始 ScanNet 第 9 列是预计算的 mesh-normal `Segmentator` superpoint，不是本项目本轮生成；当前项目已在 `tools/generate_geometric_superpoints.py` 重写了其网格邻接、面法线平滑、法线边权、Felzenszwalb 合并和小组件合并。f30 IBSp 则是本项目生成的全新第 9 列：先从完整 mesh 建图，再用 30 个已有二维实例 label-map 剪掉冲突边，最后重新图分割。它不是在原始 superpoint 上原地细化。

在 1,113 个有效 GT 实例上，原始 ScanNet superpoint 的“落在实例主属超点内的 GT 点覆盖”是 100%；当前 f30 IBSp 降为 86.2%，1,007 个实例变差、106 个不变、没有一个变好。对 85 个 Mask3D 漏检实例，该覆盖从 100% 降到 75.8%，74 个变差、11 个不变。下降因此发生在 **f30 IBSp 重分割**，早于可靠性筛选、采样、SAM2 跟踪和 Details Matter 后处理。当前 f30 输出平均每场景 superpoint 数从 990 降到约 779，而最大 superpoint 平均从约 12,207 点增至约 18,724 点；30 帧二维边界只剪掉全部 mesh 图边的约 0.15%，不足以抵消基础分割更粗造成的过合并。

最可能的实现原因是粒度不等价：ScanNet `Segmentator` README 的默认分割阈值为 `0.01`，当前 f30 生成器使用 `merge_k=0.25`。本地原始 `.npy` 不含参数元数据，不能断言它一定使用默认值，但两者阈值和输出粒度明显不匹配。OVSeg3R 的正确思想是“先在与几何基线等价的图上删除二维实例不一致边，再按同一 Felzenszwalb 规则分割”；当前不能把 `.25` 的重分割直接称为原始 ScanNet superpoint 的保守细化。

因此下一步顺序修正为：先做**无 GT 的 IBSp 基线粒度对齐**，以原始 ScanNet 分段数量、大小分布和图连通性作为结构参照，先用 `mesh_normal` 与原始兼容阈值重建几何基线，再仅追加二维冲突剪边；确认不再系统性吞并原始边界后，才做可靠 superpoint 可见性自适应。不得先重跑 SAM2。

SAM2 的直接输入不是三维 superpoint mask 或 Details Matter 后处理结果，而是关键 RGB 帧、由可靠 f30 IBSp 投影得到的三个正点提示，以及 30 帧 RGB 序列；superpoint 在 SAM2 前用于选种子/关键帧，在 SAM2 后用于把二维轨迹提升回三维。Details Matter 原文先有预计算 superpoint，先按面积排序并从较大二维 mask 去除与较小 mask 的重叠，再把 mask 提升到 superpoint、按帧可见性/实例支持筛选，最后用帧级 sIoU 形成 tracklet。当前实现只在 SAM2 传播后才清理同帧重叠，并把重叠像素从两条轨迹同时删除，尚不等价于原文“小 mask 优先”的策略；该差异可在 IBSp 基线修正后作为单独受控改动，而不是现在与超点修复叠加。

### 当前问答结论：超点、SAM2 与实例合并（2026-07-22）

原始 ScanNet superpoint 对 1,113 个有效 GT 实例保持 100% 的“实例主属超点覆盖”，只表示真实实例边界没有在超点层被不可逆吞并；它是候选形成的几何上界，不是最终 AP。最终 AP 还取决于 `Mask3D`、`YOLO-World + SAM`、BPR 和 SAM2 的候选形成，MVPDist 语义，评分排序，以及 NMS。细粒度原料若没有可靠的跨视角关联和语义证据，仍可能无法形成正确实例，或形成重复候选而被过滤。

当前未加 SAM2 的强基线数据流为：`Mask3D` 类别无关三维候选、`YOLO-World + SAM` 二维反投影候选和 BPR 候选，随后由 MVPDist 语义投票、superpoint 精炼、评分和 NMS 输出预测。历史主入口 `run_evaluation.py` 中，原始 superpoint 的 AP 为 26.4689%，f30 IBSp 的 AP 为 26.6595%；在完全相同的 f30 主入口追加现有 SAM2 候选后为 26.6907%，只增加 0.0312 个百分点。另一个 f30 一致只读融合入口的 AP 几乎不变。因此目前只能得出“SAM2 已跑通且可补少量实例，但没有稳定、可复现的总体 AP 增益”，不能把那次极小正值作为方法提升。

“先过分割、再合并”应作为下一版的基本约束。当前 f30 IBSp 平均每场景 superpoint 从原始约 990 个降至约 779 个，最大 superpoint 变大，属于过早合并风险；一旦两个真实物体先被并入同一原子 superpoint，后续提升和合并通常无法可靠拆开。后续应以原始 ScanNet superpoint 或与其粒度等价的 mesh-normal 分割作为过分割底座，只用可靠二维实例边界做进一步切分，禁止无充分证据的跨原始超点早期合并；把对象合并延后交给多视角 SAM2 轨迹、二维边界与开放词汇语义共同决定，并始终保留 Mask3D 回退。完成该 IBSp 基线粒度对齐前，不重跑 SAM2。
2. 无论 even48 结果正负，都补齐候选形成的两个缺口并以新版本重新验证：同帧跨轨迹的二维重叠区域消除，以及基于独立图像预测器重观测与传播 mask 一致性的 superpoint 支持/降权。它们应在提升到三维前抑制跨实例 mask，而不是只在三维末端删除竞争超点。
3. 将 Alpha-CLIP 限制为 MVPDist 低置信或冲突候选的可选校正；借鉴 SAS 的“模型能力加权”思想准备候选级软选择：记录几何质量、多视角一致性、两个模型的置信度与类别间隔、及其冲突情况，输出加权类别或保守拒绝。SAS 本身不是 MLLM、CLIP、Alpha-CLIP 的实例选择器；先验证 MVPDist 与 Alpha-CLIP 的条件准确率和分数排序，再考虑只对极少数难判候选调用 MLLM。
4. 只有 even48 的总体 AP 出现明确正向，才进入 `even96/odd96`。

SAM2 准备状态：官方源码已浅克隆到 `_external/sam2`，commit 为 `2b90b9f`；独立 Conda 环境 `/home/jia/anaconda3/envs/sam2` 已安装 `torch 2.5.1+cu124`、`torchvision 0.20.1` 与本地 SAM2（关闭可选 CUDA 扩展编译）。`pretrained/sam2/sam2.1_hiera_small.pt` 已校验为 184416285 bytes，SHA256 为 `6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38`；`SAM2VideoPredictor` 已成功加载到 RTX 4090 的 `cuda:0`。Any3DIS 公开参考仓库已在 `_external/Any3DIS_unofficial/Any3DIS_unofficial-main` 定位；仅复用其“超点可见性选关键视角与提示点，再用 SAM2 双向传播”的思路，不能直接运行其依赖 GT、第三方特征和自有数据布局的完整管线。当前 `openyolo3d` 环境仍为 `torch 1.12.1+cu113`，不得在该环境内升级；SAM2 将由独立环境离线导出轨迹。

## 明确不做

- 不使用 ScanNet200 `.npy` 中的 GT 语义/实例列、官方二维 GT 投影或其他 GT 作为推理输入。
- 不直接替换 Mask3D 或 YOLO-World 第三方主体。
- 不在当前阶段接入 PoVo 或最终上下文语义模块。
- 不把 SAM2 smoke/export-only 结果接入主融合，也不自行运行最终 AP；用户明确授权的冻结 even48 受控验证除外。
- 不以三场景或 fixed-frame 实验声称数据集泛化。

## 资源与验证

- 项目：`/home/jia/Wm/wm_open-yolo/OpenYOLO3D`
- 数据：`data/scannet200 -> /home/jia/Wm/Dataset/scannet200`
- GPU Python：`/home/jia/anaconda3/envs/openyolo3d/bin/python`
- 2D 缓存：`output/scannet200/bboxes_2d`
- 基础 3D masks：`output/scannet200/scannet200_masks`

本轮代码验证：`tools/filter_sam2_refined_instances_gtfree.py` 通过 `py_compile` 和 `--help` 参数检查，`git diff --check` 通过；三场景质量门控、GT-only 几何诊断和 MVPDist candidate schema 导出均已完成。本轮未运行新的 SAM2 even48 AP，也未重新运行完整 pytest。

## 协作约定

开始工作前检查 `git status --short`，保留已有未提交改动。重大方向变化或关键实验结论只更新本文件，方向变化同时更新 `资料/当前基线修改方向.md`；论文阅读只更新 `资料/论文阅读记录.md`。

## 下一会话交接（2026-07-20）

本会话已停止在三场景阶段，**尚未运行任何新的 SAM2 even48 AP**。下一位 agent 应先阅读本文件、`资料/当前基线修改方向.md`、`资料/技术问题与答复.md` 及 `related papers/` 中原文，再继续代码。

已确认的事实：baseline-novel 第三轮使 SAM2/Details Matter 候选从 34 增至 54，GT-only 几何 oracle recall 从 `.1519/.1013` 提高到 `.2532/.1646`（IoU .25/.50）；Alpha-CLIP 单独语义头无 AP 增益；新 MVPDist 语义导出将几何合格候选的类别准确率提高到 `.3333/.3846`，但三场景 AP/AP50 仍与同条件基线持平，仅 AP25 `+ .002199`。第一版 GT-free 质量门控已将候选 `54 -> 48`，candidate precision 提升到 `.4375/.2708`，且 oracle recall 不变。已确认没有等价的 `even48` 完整组合结果；下一任务是运行一次冻结的 `even48` 对照，再补齐二维重叠区域消除、独立重观测和候选级软选择。

本轮核心工具与输出：

- `tools/filter_sam2_refined_instances_gtfree.py`：不读取 GT 的 refined instance 质量门控；当前三场景输出为 `output/sam2_details_postprocess_v3_merged_round123_quality_guard_s01_sp40_smoke3_20260720/`，候选 `54 -> 48`，oracle recall 不变，candidate precision 提升到 `.4375/.2708`。
- `tools/export_sam2_refined_mvpdist_candidates.py`：将 refined instances 用 Open-YOLO 3D 原生 MVPDist 投票导出为融合候选；三场景 smoke 输出为 `output/sam2_details_mvpdist_candidates_v3_merged_round123_smoke3_20260720/`，质量门控后输出为 `output/sam2_details_mvpdist_candidates_v3_merged_round123_quality_guard_s01_sp40_smoke3_20260720/`。
- `tools/diagnose_sam2_candidate_semantics_gt.py`：严格 GT-only 的候选几何后语义诊断；最新报告为 `docs/diagnostics/sam2_candidate_semantics_gt_mvpdist_v3_merged_round123_smoke3_20260720/`。
- `tools/evaluate_multiview_object_clip_correction.py`：已修复附加候选的阈值过滤，基线仍按原始分数，附加候选按最终语义分数。
- 三场景 MVPDist AP：`output/scannet200/sam2_v3_smoke3_eval/baseline_report.json` 对比 `fused_mvpdist_t020_report.json`。
- 所有 GT 使用仅限 `docs/diagnostics/` 下的离线报告，绝不能进入推理、种子、合并、阈值或候选打分。
