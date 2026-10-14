# Full 范围复查与公共公式抽取

2026-10-14。执行依据：用户已授权退役清理，并进一步允许改代码、抽取被引用公式后删除旧文件。Full身份保持GD10 / official100 D-v3七模型 / GD10匹配597 B26 / N/T/U/U′ / K800；有效SL、当前几何研究独立保留。

## `full_bundle` 是什么

它包含实际源码、模型和项目内输入/结果，也为跨盘数据提供统一入口。它仍不是没有外部依赖、可以任意搬走的最小单目录包。项目版负责597数据和训练及两数据集适配，`runtime`负责原冻结发布复现，`evidence_snapshot`负责特定历史自身语义版本；三者有明确版本身份，不能因名字相同就删除其中一个。

以下为本轮复查时的实物统计，未穿透软链接，排除Git/字节码；源码抽取后的数字另见当前逐文件索引。

| 目录 | 实际内容 | 实物占用 | 链接情况 |
|---|---|---:|---|
| `models` | 视觉权重、D-v3、B26、champion及配置 | 5.79 GiB | 权重实物保存在Full |
| `vendor` | Mask3D、YOLO-World、GD、SAM、AlphaCLIP源码 | 0.29 GiB | 主要为实物 |
| `results` | ScanNet当前K800及Replica结果/来源 | 12.46 GiB | 18,596实物文件及99链接 |
| `protected_sources` | val48 GD10、native/YOLO适配和必要来源 | 30.13 GiB | 131,060实物文件及2,148链接 |
| `runtime` | 原925项发布身份对应的独立实现 | 约17 MiB | 发布源码实物，部分模型/组件链接 |
| `code` | 项目完整流程及必要公共定义 | 约3 MiB | 实物；本次tools由137减至130 |
| `datasets`、`training` | 原数据、597特征监督与训练身份入口 | 链接本身很小 | 5个/3个跨盘入口，数据实际在原盘 |
| `artifacts` | 964 ScanNet与8 Replica逐场景统一入口 | 约4 MiB | 972帧合同实物，8,725个资产链接 |

因此不能说“Full里只有引用”。占空间的模型、结果、必要来源确实存放在这里；597跨盘原始/预处理数据则继续在原盘。完整位置见[资产地图](../full_sl_asset_map_20261014/README.md)与[当前代码地图](../../../资料/代码文件索引与当前主线地图.md)。597原数据在`/media/jia/软件1/scannet_train_stream/scans`及`scannet_additional_train_stream/scans`，GD10在后者`fi1_G10_full597_20260919/frontend`；处理后几何/自身语义与训练features在`fi1_G10_closure_20260921/train`及`learning/dataset`。正式权重集中在`full_bundle/models/b26/B.joblib`，原盘正式训练来源仍为闭环`learning/models/B.joblib`。不能与旧A模型或597重训D-v3对照混同。

## 冻结自身语义快照已精简

原快照有534个tools文件、263个tests文件及多份历史文档。此前整理新增的`.local -> 项目.local`链接过宽，导致当前几何实验出现在Full目录树里；本轮已移除该链接及其他广泛根链接。那些`fi1_decoupled_region_internal...`目录属于几何研究，真实内容保留在`research/geometry/experiments`或外盘研究资产。

快照现仅保留24个Python文件和1个冻结配置：12个tools、4个utils、8个原评测器文件。整个快照tests目录及873个多余历史文件已退出活跃目录，源码原字节完整归档并核验。剩余源码包含投影、视角选择、语义汇聚、Alpha推理和评测实现，并非仅有调用。

- [快照删除和保留逐文件记录](source_trim_executed.json)
- [独立快照导入验证](snapshot_import_verified.json)
- 原快照：`archive/retired_20261014/evidence_snapshot_before_minimization.tar.gz`

## 被借用公式已独立，旧入口已删除

| 新公共模块（实际在Full的code/tools） | 保留内容 | 删除的旧入口类别 |
|---|---|---|
| `fi1_instance_eval_shared.py` | 原匹配/AP记录和汇总、阈值常量、去类GT映射与评测器配置 | 8个oracle/diagnostic脚本及其连带CLI |
| `fi1_superpoint_geometry.py` | 原RGB/法向归一化和保留超点编号的接触图 | 自动SAM轨迹生长账本CLI |
| `fi1_mask_projection.py` | 可见点投影、RLE、同帧关系、帧选择和固定观测排序 | 自动SAM观测导出CLI |

22个函数/常量逐段抽取，AST不变；23个调用文件改写。完整10个删除文件名、抽取符号、原hash和调用变更在[执行回执](shared_extraction_executed.json)，旧源码保存在`archive/retired_20261014/full_shared_formula_extraction.tar.gz`。GT公式用于训练监督或后置评测，不因此让推理读取GT。旧oracle中未被使用的N1/N2匹配和边界动作CLI不再进入活跃依赖链。

项目前端移除自动SAM生成器，仅支持D导出。Replica共同验证改为直接建立与原映射一致的独立帧合同，继续用YOLO框提示SAM；已有封存来源帧映射读取仍可用。旧ARMS标签用于解释历史合同，独立`runtime`保持原发布字节，不改其925项身份。

原基线导出器也已移除SAM-fused/BPR旧模式，只导出Full所需Mask3D+YOLO原生候选；返回`prediction[:3]`不变。旧融合模式测试归档，实际原生导出测试保留，[模式精简回执](native_fusion_trim_executed.json)。

顶层`tools`和`tests`是兼容软链接，删除目标及链接才能退出活跃阅读；它们不是另一份真实源码。本次最终tools有281个兼容链接、tests有106个，分别指向Full/SL/研究实际代码。并非全部属于Full；当前Full只从[代码地图](../../../资料/代码文件索引与当前主线地图.md)阅读。

其他训练/特征模块仍有实际公共定义引用，不能仅据文件名批量删。本轮解除已核实的旧诊断和自动SAM依赖，不声称所有剩余模块的每个函数都参与生产。当前逐文件路径和新hash在[索引](current_code_index.csv)，函数级抽取与文件级调用关系分开记录，[导入边](code_edges.json)不能单独当作删除白名单。

## 输出目录复查与执行

`output`多数为7—8月旧策略观测、轨迹、诊断、OOF、候选和测试产物。**不能按automatic或SAM字样整体删除**：GD10仍复用轨迹实现、SAM权重，以及部分旧来源数据；必要输出从绑定和链接确认。

初版审计把旧磁盘占用清单`storage_audit_20260916/inventory.json`中的目录名误算成运行绑定，本轮已纠正。核查当前Full/SL/几何合同和逐场景路由，区分历史来源链；另检查5,703个资产根的实际链接，并读取此前跳过的23个大型元数据文件。

- [目录逐项复查清单](output_inventory_refined.csv)：242个实物输出根，包含`scannet200`各子目录；标记真实绑定、旧来源、代码默认值及排除的磁盘清单提及。
- [绑定和链接证据](output_reference_evidence_refined.json)：保留目录的具体来源、历史边、实际软链接及大型元数据补查。
- [224根实际删除回执](output_deletion_executed.json)：分配块22.327263 GiB，空闲净增22.192039 GiB；先保存并逐字节核验6,986份汇总证据，压缩包约4.4 MiB。
- [额外两套旧融合产物删除回执](legacy_fusion_output_deletion_executed.json)：解除旧CLI默认路径引用后删除，不能在前224根中重复计数。
- [被删文件清单](output_removed_file_inventory.json)及[融合文件清单](legacy_fusion_removed_file_inventory.json)：实物字节、分配块、保留摘要hash；大型二进制未归档。

用户举例的`output/automatic_mask_track_yoloworld_semantics_even96_uniform30_p16_v0_20260802`是旧自动SAM轨迹的YOLO投票账本（96场景），无当前绑定，已删除；其报告/汇总保存在退役摘要压缩包。

原`output/scannet200/scannet200_masks`（8.66 GiB）、`bboxes_2d`、原GT masks、`scene_splits`及必要历史监督/模型来源继续保留。顶层目录不能整体删除；Full结果已经集中在Full的results，SL与几何资产分别在各自目录。具体最终数量/空间见[完成验证](delivery_verified.json)，前后引用清单属于审计时点，是否已删以实际执行回执为准。

## 验证与边界

当前测试日志在[project_tests.log](https://github.com/1108-WM/change-open-yolo/blob/main/docs/EXTERNAL_ASSETS.md)。[同输入回放](refactor_verified.json)完成281项核对：两个Full场景全池B26分数逐值一致、K800逐条一致；一个SL开发55场景类别及去类输出一致；原AP函数匹配/重复/忽略区/无预测样例一致；8场景Replica共同帧身份一致。旧sklearn1.3.0环境中[9个后端模块与7个权重加载](legacy_geometry_verified.json)通过，未从旧worktree导入。[发布包核验](https://github.com/1108-WM/change-open-yolo/blob/main/docs/EXTERNAL_ASSETS.md)保持925项hash；删除后的资产验证见[日志](https://github.com/1108-WM/change-open-yolo/blob/main/docs/EXTERNAL_ASSETS.md)。

未重训597，未运行312/Replica完整GPU前端，也未启动下一几何实验；当前验证不能代替全量GPU端到端重算。原训练/预测/实验合同和seal保持历史身份，新运行应建立新的源码合同。Git对象、环境、有效SL、当前几何模型/结果、用户要求保留的Qwen/展示/ZIP未作为退役清理对象。
