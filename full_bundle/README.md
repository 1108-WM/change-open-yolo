# 当前 Full 的统一入口

> GitHub源码快照（2026-10-14）：本文件中的本机资产/历史交付说明保留作溯源。本次main不包含权重、数据或缓存；运行需要另备资产，见../docs/EXTERNAL_ASSETS.md。历史哈希与验证结果不代表本次重新运行。


2026-10-14 已执行集中整理。配置保持 **GD10 / official100 D-v3 七模型 / 匹配597 B26 / N/T/U/U′ / K800**，未重训、未调参。

| 目录 | 内容 |
|---|---|
| `runtime/` | 从原交付 ZIP 提取的独立冻结生产运行包；925 项发布文件哈希原样保持，43 项测试通过 |
| `cli/full.py` | 统一入口，支持 `verify / native / frontend / alpha / run / score / seal / evaluate`；参数见 `--help` |
| `code/tools/`、`code/utils/`、`code/tests/` | 当前597数据预处理、GD10、B26训练、ScanNet/Replica后端、评测等项目代码；原 `tools/tests/utils` 路径为兼容链接 |
| `code_sources/evidence_snapshot/` | 迁出的冻结自身语义 helper，历史字节保留；活跃 Full 已不从旧 `.local/.../workspace` 导入 |
| `vendor/` | Mask3D、YOLO-World、GroundingDINO、segment-anything、AlphaCLIP 的实际源码 |
| `models/visual/` | 实际 Mask3D、YOLO、SAM、GD、BERT、CLIP-B/32、AlphaCLIP 权重；原 `pretrained` 路径为链接 |
| `models/config/` | ScanNet/Replica 与 YOLO 配置；模型配置中的原路径由兼容链接解析 |
| `models/dv3_official100/`、`models/b26/`、`models/champion/` | 集中的七个 D-v3 模型、正式 B26 与 champion；小模型逐字节复制或同盘迁移并核对 |
| `metadata/` | 597身份名单、ESAM实际使用的标签/训练元数据、964扫描路由、972验证帧索引 |
| `datasets/` | ScanNet原始旧数据/新397/验证GT、Replica原始数据入口；跨盘数据为链接，无重复拷贝 |
| `artifacts/scannet/{train,dev,val}/<scene>/` | 每场景预处理、native、GD10、backend、own、YOLO-SAM、geometry、points、evidence的统一入口 |
| `artifacts/replica/` | 8场景预处理/前端/官方native/后端/自身证据/Full输出，以及精确验证帧名单 |
| `training/` | 597 B26 features/监督/训练身份；旧GT绑定来源保留，外盘数据为链接 |
| `results/` | 当前ScanNet K800和Replica官方掩码Full的实际结果；Replica旧资产来源也在此保留 |
| `protected_sources/` | 迁入项目本盘的当前val48 GD10、native/YOLO适配与闭环身份 |
| `provenance/legacy_val48_GD_source/` | GD10仍需的GD5/D RLE来源和历史合同；A/B/C退役大资产已删除 |

**范围补正（2026-10-14深度精简）：这里是集中保存与兼容运行的目录，仍不等于完全自包含的最小Full源码包。** 本轮删除38个旧策略完整入口和26个对应实验测试，将21,203行旧脚本中的4,778行现用定义抽入`code/tools/fi1_shared/`的29个公共模块；保留的133个函数/类已核对定义一致。这些模块沿用来源文件名便于追溯，但不再保留旧实验CLI。Full tools当前为92个顶层文件，加29个公共模块和1个包初始化文件，共122个；另有4个utils和35个归属正确的测试文件。现用入口、实际依赖与SHA256见最新索引。项目前端仅导出D，Replica共同验证使用独立帧合同与YOLO框提示SAM。`runtime`保持独立冻结实现，`code_sources`保留必要历史版本；完整代码还包含原评测器、第三方实现与配置。

实际模型、vendor、当前结果和项目内必要来源保存在这里。`datasets`、`training`和多数`artifacts`是跨盘输入的统一链接，实际数据仍在原盘；当前实验仍有历史绝对路径兼容要求，因此不能把本目录直接当作完全自包含、可随意搬走的所有资产副本。

此前`code_sources/evidence_snapshot/.local`是指向项目整个`.local`的宽泛链接，误将几何研究显示在Full之下，现已移除。snapshot现仅保留24个依赖源码文件及1个冻结配置；873个多余历史文件归档后退出，整个snapshot测试目录已移除。历史helper本身包含实际投影、视角选择、语义汇聚和Alpha推理代码，并非仅有调用脚本。详见[本次范围复查](../docs/cleanup/full_scope_recheck_20261014/README.md)。

在项目根核验：

```bash
.venvs/yoloe_gpu/bin/python full_bundle/cli/full.py verify --with-external-assets
.venvs/yoloe_gpu/bin/python docs/cleanup/full_consolidation_execution_20261014/verify.py
```

完整逐文件代码路径见[当前索引](../docs/cleanup/deep_active_cleanup_20261014/current_code_index.csv)；各阶段作用见[Full代码地图](../资料/代码文件索引与当前主线地图.md)。本次公式抽取及旧融合模式精简后的77项测试和281项核对通过，普通val及旧val48两场景全池分数/K800逐值一致，SL一场景类别与去类输出一致；sklearn1.3.0后端9模块及7权重加载通过。旧output另有226个退役根清理，详情见本次复查回执。未重跑完整597训练、312全量GPU前端或外部A100。

旧合同/receipt不改写，其旧绝对路径通过兼容链接读取。代码移动后项目根定位、worktree导入、元数据和GVC帧接口已适配，因此活跃源码有新哈希；旧源码在 `archive/retired_20261014/`，当前新哈希在执行索引中。不要用旧code seal核验改过路径的活跃脚本，也不要直接恢复已完成旧总控。独立 `runtime` 的原发布哈希仍可核验。

SL只在 `sl_bundle/`；当前几何研究在 `research/geometry/`。Full地图末尾列有效SL，不列历史失败实验入口。

最新深度精简、`.local`逐项结论和空间统计见[本轮回执](../docs/cleanup/deep_active_cleanup_20261014/README.md)。480项测试及4项子测试、281项公式/Full/SL同输入核对通过；official100七模型与champion加载通过，925发布hash保持。未重训597或全量GPU重算。
