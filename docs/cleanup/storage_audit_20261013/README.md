# 项目磁盘占用与待审清理目录（2026-10-13）

**后续专项检查（2026-10-14）：历史 val48 约 24.85 GiB 目录找到约 7.00 GiB 的潜在无损整理空间，含约 1.63 GiB 保留路径去重及约 5.37 GiB 可恢复压缩；尚未执行。依赖、限制和逐文件清单见 [专项审计](../val48_retirement_audit_20261014/README.md)。**

**最新执行更新（2026-10-13 23:31）：用户后续已明确授权低风险清理及重复数据保留一份。已删除 1,534 个白名单缓存，并将三个重复诊断文件改为只读硬链接、保留两个原路径，合计移除 379.58 MiB 数据分配。详情见 [执行报告](EXECUTED.md) 与 [执行回执](cleanup_execution.json)。没有删除 `.local` 实验目录。**

**以下正文是执行前的首轮盘点记录，容量与“待执行”表述保留当时时点；缓存白名单已经执行，不能重复计算释放量。原 summary.json、inventory.json、review_verified.json 同样属于执行前快照。当前执行结果以以上回执为准。首轮只盘点，没有删除；当时的“继续”仅按继续审计处理。本次操作依据其后的明确清理指令。**

项目路径：`/home/jia/Wm/wm_open-yolo/OpenYOLO3D`。

项目本体实际分配空间 **241.76 GiB（259.59 GB）**，约 **277 万个普通文件（硬链接按 inode 去重）**。项目所在分区使用率 **97%**，20:05 快照剩余 **50.93 GiB**。这次紧张的是磁盘，不是运行内存；同一快照 RAM 可用约 **52.67 GiB**。

统计不跟随符号链接，文件和目录用 `st_blocks × 512` 统计，硬链接只计一次；与 `du -x` 的约 242 GiB 一致。沙箱中另外挂载的 `.codex`、`.aws` 已跳过。跨目录硬链接的分配归属取决于遍历顺序，因此目录占用不等于删除该目录可释放的空间。`data/scannet200` 指向项目外同盘数据集，不重复算进项目本体；外盘缓存也不在上述总量内。整个分区还有项目外占用，本次没有做全盘清理审计。

| 目录 | 本盘占用 | 内容与本次判断 |
|---|---:|---|
| `.local` | 134.77 GiB | 实验几何、候选关系、视觉特征、监督、模型、预测、回执；不能按失败实验整删 |
| `output` | 36.99 GiB | 早期原生掩码、逐帧候选与共享缓存；需要保护原始候选及身份 |
| `.git` | 30.71 GiB | Git 对象与历史，另有未完成临时对象；不是普通缓存目录 |
| `pretrained` | 22.57 GiB | 模型权重；其中 Qwen 约 15.46 GiB |
| `docs` | 7.69 GiB | 几乎全部是 `diagnostics` 中的训练/诊断数据，不是 Markdown |
| `.venvs` | 5.80 GiB | Python/深度学习运行环境 |
| `outputs` | 0.85 GiB | 展示、复现包、算法文档，需分开判断 |
| `tools` | 33.16 MiB | 含约 21 MiB 字节码缓存；源码本身很小 |
| `tests` | 5.47 MiB | 含约 3.7 MiB 字节码缓存；测试源码不是主要占用 |

其他目录还包括 `_external` 1.46 GiB、嵌套 `OpenYOLO3D/replica` 0.52 GiB、论文和模型源码等。所有 Markdown 文件合计仅 **15.89 MiB**。相反，JSONL 约 **75.43 GiB**、NPY 约 **32.65 GiB**、JSON 约 **31.53 GiB**；大量 JSON/JSONL 是逐候选、逐点或逐关系数据，并不是轻量说明文件。

**如何查看清单**

| 文件 | 用途 |
|---|---|
| [directory_review.csv](directory_review.csv) | 1,590 个目录条目：占用、文件数、保留/待核实状态和理由；含 `.local`、`output` 的一级实验目录及 `docs` 的诊断/可视化目录 |
| [safe_cache_candidates.csv](safe_cache_candidates.csv) | 本轮低风险候选的逐文件白名单，含路径、字节数、SHA256、对应源码；不是删除脚本 |
| [protected_paths.csv](protected_paths.csv) | 2,773 条保护路径/范围及证据，覆盖最新合同输入、SL、Full 和展示 |
| [code_review.csv](code_review.csv) | `tools`/`tests` 共 1,252 个顶层脚本的逐文件保留登记及 Git 跟踪状态 |
| [cache_exceptions_keep.csv](cache_exceptions_keep.csv) | 83 个未找到对应当前源码的字节码，保守留存，不在清理白名单 |
| [dependency_evidence.json](dependency_evidence.json) | 当前入口的 113 个静态导入源码、导入位置、合同指纹和保护边界 |
| [asset_integrity_checks.json](asset_integrity_checks.json) | 14 项关键资产的合同哈希核验，全部一致 |
| [inventory.json](inventory.json) | 原始容量盘点、目录分解、最大文件、文件类型和链接目标 |

CSV 使用 UTF-8 BOM，可用表格软件筛选。目录清单是审阅索引，**不能把“待核实”或“待决定”解释成可删**；父目录与子目录有重叠，不能把所有行的容量相加。

**第一类：可以列入下一次低风险清理的项目**

仅 [safe_cache_candidates.csv](safe_cache_candidates.csv) 中的 **1,534 个文件，共 24.60 MiB**：

- `tools/__pycache__` 中有对应 `.py` 源码的字节码。
- `tests/__pycache__` 中有对应测试源码的字节码。
- 根目录、`utils`、`evaluate` 中同类缓存，以及 `.pytest_cache` 的运行缓存。

逐文件检查了源码存在、非符号链接、未被 Git 跟踪，并记录当前内容哈希；这些文件均为单硬链接。删除后 Python/pytest 能重新生成。**不建议整目录通配删除 `tools/__pycache__` 或 `tests/__pycache__`**，因为其中还有上述 83 个没有对应当前源码的文件，未纳入白名单。环境、归档代码快照和用户明确保留展示中的缓存也未列入本批。

本轮没有把任何 `.py`、测试源码、协议 Markdown 或实验结果目录认定为“明确以后无用”。这不表示永远不能整理，而是目前没有足够证据安全删除这些源文件。1,252 个顶层脚本中还有大量未被 Git 跟踪的文件，不能指望删错后都能从 Git 恢复。

执行前仍要按这份逐文件清单复核哈希/源码和当前任务状态，避免后续工作生成的新文件被连带删除。本轮未执行。

**第二类：必须保护的主线目录**

```text
OpenYOLO3D/
├── pretrained/                         保留生产及当前实验权重
├── .venvs/yoloe_gpu/                    保留 Full、SL、局部学习环境
├── .venvs/tmux_runtime/                 保留任务托管环境
├── .local/
│   ├── fi1_unified_evidence_20260910/workspace/    Full 实际调用的旧源码
│   ├── fi1_r0_r4_preparation_20260913/             共享身份、GT、旧资产
│   ├── fi1_population_visual_20260910/            被复用的 Alpha 特征
│   ├── fi1_core_ablation_val312_20261006_v1/       Full 展示来源
│   ├── fi1_sl_single200_dev55_20261010_v1/         有效 SL 单模型
│   ├── fi1_s_semantic_rank_residual_development200_20261010_v2/
│   │                                              SL 五折模型与归一化
│   ├── fi1_s_fixed_L_rescore_development200_20261010_v1/
│   │                                              已登记 SL 配方和结果
│   ├── fi1_sl_replay_stability_development200_20261010_v1/
│   │                                              SL 稳定性与复算依据
│   ├── fi1_candidate_local_repair_learning_development200_20261012_v1/
│   │                                              当前复用 plan/features
│   ├── fi1_deletion_blocks_development200_20261013_v1/
│   │                                              当前复用修正监督 labels
│   ├── fi1_local_region_supply_development200_20261013_v1/
│   │                                              当前复用 regions
│   ├── fi1_decoupled_region_internal_development200_20261013_v1/
│   │                                              当前复用物理场景划分合同
│   └── fi1_mask3d_feature_ab_internal_20261013_v1/  最新实验，外盘链接
├── output/scannet200/                   原生候选及共享数据，不能整体清空
├── output/replica/                      官方候选标准安装目录
├── outputs/
│   ├── Full点云展示_20261011/            用户明确保留
│   ├── Full点云展示_20261011.zip         对应交付包，一并保护
│   ├── Full算法说明_20261012/            当前算法说明
│   └── github_frozen_full800_20261008/   Full 源码和模型复现包，必须保留
├── docs/state_archive/                  旧合同/交接哈希依据
├── tools/ 与 tests/                     保留源码，仅逐文件缓存另列
└── OpenYOLO3D/replica/                  data/replica 的真实数据源
```

这是重点导航，不是完整依赖树；完整边界见保护清单。`output/replica` 和 `.local/official_openyolo3d_assets_20261011/extracted/replica_masks` 虽有同内容文件，但旧合同仍指向后者，不能只留一个名字后直接删另一个路径。

最新实验的 `contract.json` 实际绑定 **1,281 个前序 source_bindings**、960 个 inputs 和 160 个 GT 输入。`run_fi1_mask3d_feature_ab.py` 也实际读取旧局部学习的 `plan/features`、删除实验的 `labels`、区域供给的 `regions`。因此，“这个方法没有提升”只说明该配方结果失败，**不代表它产出的特征、监督和区域已经没有用途**。

最新 A/B 目录的真实目标为 `/media/jia/软件1/fi1_mask3d_feature_ab_internal_20261013_v1`，另外实测约 **1.08 GiB**。删项目里的链接几乎不能释放本盘容量；删链接目标又会破坏最新实验。另有生产 Full 的 B/Q、GD10 前端和自身语义在 `/media/jia/软件1/scannet_additional_train_stream/`，D-v3 七模型及部分源码在项目外旧 worktree，均应保留。

10 月 12—13 日本盘所有以这两天日期命名的近期实验/审计目录合计约 **3.42 GiB**。当前空间大头并不是这几轮小模块本身。独立核验了最新合同中的 11 个 SL/B26 资产、SL 单模型及归一化、Mask3D 主权重，共 **14 项哈希全部一致**。这次没有启动 GPU 推理或训练，也没有声称完整 Full 重新跑通；静态导入不能穷尽动态依赖，所以没有依据“搜索不到引用”把代码列为无用。

**第三类：较大的整理机会，仍有取舍，不能直接执行删除**

| 具体范围 | 当前占用/潜力 | 判断和必要条件 |
|---|---:|---|
| `.local/fi1_val48_frontend_full_pipeline_20260914` | 24.85 GiB | 其中 `full_batches` 约 18.07 GiB，保存终态几何和候选关系。需先划出仍被引用的点集、身份、预测、评测及复现输入，不能整删；当前未给出可释放保证 |
| `.local/fi1_r0_r4_preparation_20260913` | 12.40 GiB | 当前合同确实引用，保留；不能因为名称旧而删 |
| `.local/fi1_r3b_val312_execution_20260914` | 8.41 GiB | 历史基线/复核来源，保留；不把终态 AP 摘要当完整结果替代品 |
| `.local/official_openyolo3d_assets_20261011/download/OpenYOLO3D.zip` | 1.72 GiB | 作者原始安装包，只提取过其中部分内容。当前预测读取解压掩码，但完整下载核验依赖原 ZIP；是否退役需明确接受失去完整本地源包 |
| `.git` 的临时对象 | Git 报告约 1.78 GiB garbage | `git count-objects -v` 报 69 项；还需宿主确认没有 Git 写入，并采用单独 Git 维护方案。不能手删整个 objects，也不能把 30.71 GiB 都当垃圾；本轮不进入可删白名单 |
| `docs/diagnostics` 一组完全重复的数据 | 约 355 MiB 的去重潜力 | 原版与 `reproduction` 版三个文件 SHA256 全相同，见下文。路径仍有复核意义，可以后续设计保留双路径的内容去重；本轮未删除或替换 |
| 5 个旧展示目录 | 合计约 455 MiB | 见下表；不属于当前指定展示本体，但含不同历史图片/核验。可在用户选择退役旧展示后处理，不冒称“以后绝不可能使用” |

五个旧展示目录：

| 路径（均在 `outputs/`） | 约占用 | 本轮状态 |
|---|---:|---|
| `paper_furniture_focused_render_20261008` | 281 MiB | 待决定是否退役历史家具图例 |
| `full800_three_visualizations_20261010` | 103 MiB | 待决定是否退役上一版展示 |
| `full800_case_review_20261010` | 41 MiB | 含案例筛选与污染审计，待决定 |
| `paper_real_figure_assets_20261008` | 18 MiB | 论文图例原始导出，待决定 |
| `开题三维可视化_20260923` | 11 MiB | 开题展示，待决定 |

不能把 `github_frozen_full800_20261008` 加进这五项；它包含冻结模型、Alpha 文本向量、源码和 A100 复现说明，是约 20 MiB 的重要交付资产。保留的 `Full点云展示_20261011` 实际还含重绘、核验脚本，保护的不只是 PNG。

已核对的重复诊断目录：

```text
docs/diagnostics/official100_fi1_d_v3_stage_c_v2_member_dataset_20260822/
docs/diagnostics/official100_fi1_d_v3_stage_c_v2_member_dataset_reproduction_20260822/
```

两边的 `pair_unions.jsonl`、`member_removal_dataset.jsonl`、`summary.json` 全部逐文件哈希一致，当前均为单链接文件。证据：[duplicate_diagnostic_check.json](duplicate_diagnostic_check.json)。去重必须保留各自路径和内容身份；直接删一个目录与这种去重不是同一件事。若未来采用硬链接，还需保证这类历史文件不被原地写入，避免联动修改。

**Qwen 继续保留。** 本地权重约 15.46 GiB，专用 `.venvs/qwen25vl_py311` 环境约 5.61 GiB，合计约 21.07 GiB 的占用。它不是当前 Full/SL 的模型推理步骤，但旧兼容源码导入链中仍有 Qwen 模块，历史实验也绑定了这些权重。更重要的是，9 月 16 日清理记录记载曾明确保留 Qwen；本次没有将这项历史偏好视作已撤销。因此这 21 GiB **不计入任何可删除容量**，只有以后明确退役该路线才适合另行审查，环境硬链接也可能导致实际释放少于占用。

**建议下一步如何选范围**

若仍坚持“只删已经明确不再需要、又不损失旧实验复现能力的内容”，本轮先采用 24.60 MiB 的缓存白名单，收益确实很小。若希望释放数 GiB 或更多，应优先对历史大缓存做一次明确的退役审查，先验证“终态点集＋预测＋模型/代码身份＋评测证据”的保留集，再列具体子目录；不能用删除代码、测试和报告来替代这一步。

项目中已经有一次 9 月 16 日的缓存清理，相关 RGB/逐帧缓存不能照旧计划再估算一次释放量；本报告以本次实际存在的文件为准。同盘移动到 archive 或回收站不会立即释放该分区空间，跨盘归档也要计入目标盘容量；外盘目前同样已用 95%。

原始磁盘/RAM 快照见 [filesystem_snapshot.txt](filesystem_snapshot.txt)，Git 只读统计见 [git_storage_statistics.txt](git_storage_statistics.txt)。审计脚本 [inventory.py](inventory.py)、[analyze.py](analyze.py) 只产生盘点清单，不包含删除逻辑。所有生成文件位于本审计目录；原实验文件保持不变。
