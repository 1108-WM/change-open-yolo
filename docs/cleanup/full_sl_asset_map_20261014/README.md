# Full / 有效 SL 的磁盘资产与保留说明

日期：2026-10-14。目标是说明当前各目录内容、597 与两数据集的真实资产位置，并为下一步精简提供可核查依据。本次修改文档和生成索引，未删除实验数据、源码、测试或模型。完整代码流程已收敛到[Full + SL 地图](../../../资料/代码文件索引与当前主线地图.md)。

## 1. 用户给出的容量是什么

这些数值来自 2026-10-13 全项目审计，是磁盘分配空间，不是 RAM；项目当时约 241.76 GiB。统计不跟随符号链接，硬链接按 inode 去重，**不包含外盘 597 数据与项目外 ScanNet200 数据**。此前已释放 379.58 MiB，原快照保留不重写，本报告没有重新遍历 277 万文件或声称这些是此刻新测值。

| 文件夹 | 原审计占用 | 放的内容 | 与当前 Full/SL 的关系 |
|---|---:|---|---|
| `.local` | 134.77 GiB | 多轮实验的观测、轨迹、候选几何、关系表、视觉特征、训练监督、模型、预测、匹配/评测、日志、合同、源码快照；另有实际运行代码和外盘链接 | 最大占用源，混有必需共享资产和历史对照，不能按“隐藏目录/失败实验”整删 |
| `output` | 36.99 GiB | 原 OpenYOLO3D 的 Mask3D 候选、YOLO 检测缓存、SAM 观测、早期轨迹/GVC/champion 模型、场景划分、官方 Replica 掩码 | `output/scannet200/` 原生候选和 champion 仍被复用；它不是只有过期输出 |
| `.git` | 30.71 GiB | Git 对象、历史提交、分支、worktree 管理信息及临时对象 | 保存历史代码/数据身份，不能手动按大小删 objects；删工作区文件也不自动释放历史对象 |
| `pretrained` | 22.57 GiB | Mask3D、YOLO-World、SAM、GD、BERT、AlphaCLIP，以及 Qwen/DINOv2/SAM2/YOLOE 等历史/研究模型和配置 | 保存当前生产权重/词表/配置；Qwen 约15.46 GiB是大头，用户此前明确保留 |
| `docs` | 7.69 GiB | Markdown 协议外，主要是 `diagnostics` 的 NPY/JSONL/CSV/模型和中间诊断数据，另有可视化、交接快照、清理证据 | `docs/diagnostics` 约7.64 GiB，真正的文字文档很小；部分诊断路径被代码/合同引用 |
| `.venvs` | 5.80 GiB | Python 运行环境、包与编译依赖；主要是 Qwen 环境5.61 GiB，另有 yoloe_gpu/tmux 等 | Full/SL 使用 yoloe_gpu，历史 D-v3/视觉环境还在项目外；删除环境会失去可运行性 |
| `outputs` | 0.85 GiB | 人类可读图例、网页、论文图、Full 算法说明、ZIP 和冻结复现交付 | 与 `output` 完全不同；保护 `Full点云展示_20261011/`和 ZIP、冻结 Full 包 |
| `tools` + `tests` | 38.63 MiB | 编排、算法/训练/验证脚本、测试及 Python 字节码缓存 | 原数字包括约24.7 MiB字节码；已执行白名单缓存清理。源码只有约十余 MiB，删代码对百 GiB问题帮助极小 |

其他必须看到的目录：`models/` 是模型源码，`_external/` 是第三方代码；`data/scannet200` 是项目外链接，`data/replica` 链接到项目内嵌套 `OpenYOLO3D/replica`，后者原审计约0.52 GiB。文件夹名字“数据/输出”不能代表是否可再生或是否被引用。

## 2. 最大目录具体保存什么

下表仍使用执行前审计数字；同一列父子目录不能相加为可释放量。

| 目录 | GiB | 内容/保留依据 |
|---|---:|---|
| `.local/fi1_val48_frontend_full_pipeline_20260914` | 24.85 | 旧 val48 四臂完整前端/几何/自身证据及关系；后续生产与代码仍引用其部分资产；有可恢复压缩/保留路径去重机会 |
| `.local/fi1_r0_r4_preparation_20260913` | 12.40 | 共享场景身份、原点几何、原生绑定、YOLO/Alpha 证据等；当前 Full/SL/训练仍有直接依赖 |
| `.local/fi1_r3b_val312_execution_20260914` | 8.41 | validation312 YOLO、原生/证据执行、后续 Alpha 修正、GT/匹配/预测来源；历史身份不可混为最新模型 |
| `.local/replica_registered_full_20261011_v1` | 5.55 | 第一轮隔离 Replica 超点、GD10、YOLO-SAM、YOLO缓存及旧结果；第二轮官方掩码版从此复用 prepared/frontend/common |
| `.local/replica_official_masks_full_20261011_v1` | 3.96 | 有效官方候选 Full 后端、实际几何、自身语义、K800、双口径评测/回执 |
| `.local/fi1_population_visual_20260910` | 3.01 | 后续复用的 Alpha 视觉特征；保留输入来源和点集/视角绑定 |
| `.local/fi1_core_ablation_val312_20261006_v1` | 2.95 | 当前 Full K800 与固定消融的实际成员/分数、匹配 AP 和预测封存；展示使用它 |
| `.local/fi1_val48_grounded_stride10_20260918_v4` | 2.92 | GD10 48 场景观测/轨迹与绑定复用；不能与旧四臂目录混同 |
| `output/scannet200` | 14.89 | scannet200_masks、bboxes_2d、native_cache_*、scene_splits、历史诊断；不是一个可以清空的最终结果目录 |
| `.git/objects` | 30.71 | 几乎全部 Git 容量；有约1.78 GiB临时垃圾候选需 Git 维护核验，未执行 |
| `pretrained/checkpoints` | 17.61 | Qwen约15.46 GiB、Mask3D/YOLO/SAM/DINO等；生产大权重分散在多个目录 |
| `pretrained/alpha_clip` | 2.00 | AlphaCLIP base/微调权重 |
| `.venvs/qwen25vl_py311` | 5.61 | 之前明确保留的 Qwen 可运行环境 |
| `outputs/Full点云展示_20261011` | 0.20 | 当前明确保留的网页、点云图与渲染/验证交付；同级ZIP也保留 |

全部一级实验目录解释和保留证据已在首轮 [directory_review.csv](../storage_audit_20261013/directory_review.csv)、[protected_paths.csv](../storage_audit_20261013/protected_paths.csv)，无需再复制1590行到正文。

## 3. 597 必须保存的链条

当前生产 Full 的训练组合是 **official100 + ncs_train100 + first199 + remaining198 = 597 扫描，536 主场景**。生产 D-v3 是 official100 七模型；597 配套训练当前 B26。

```text
/media/jia/软件1/scannet_train_stream/
  scans/                          旧原扫描几何/标注（含开发/诊断）
  prepared/ + records/            official100 RGB-D/点域及 native/YOLO
  prepared_ncs/ + records_ncs/    ncs_train100及独立开发的对应数据

/media/jia/软件1/scannet_additional_train_stream/
  scans/                          新397原扫描几何/标注
  prepared/                       新397 RGB-D、相机、PLY、12列NPY、GT
  records_native_first199/         新199 native/YOLO缓存
  records_native_remaining198/     新198 native/YOLO缓存
  fi1_A_first199_20260917/          旧共同验证观测/适配等，后续仍引用
  fi1_A_remaining198_20260917/      同上
  fi1_G10_full597_20260919/frontend/  当前597 GD10观测和轨迹
  fi1_G10_closure_20260921/
    train/scenes/<scene>/backend/  D-v3阶段几何/计划/代码身份
    train/scenes/<scene>/own/D/    实际点集与自身YOLO/Alpha
    dev/ + val/                   55/312的GD10、几何、自身语义
    learning/dataset/<scene>/     features.npz、members.jsonl、receipt
    learning/models/B.joblib      当前正式597 B26
    learning/models/B.json        特征/训练/模型身份
    learning/models/summary.json  正式训练汇总
    learning/train_dataset.json   训练数据身份
```

原 `.sens` 在之前经授权导出后已经删除，现存 scans 不是完整传感器原包。**导出的 RGB/depth/poses/intrinsics/PLY/NPY 与 GT 必须保留**；重做原始解码需要重下载 `.sens`。当前 `资料/训练597场景清单_20260917.json` 与 `.local/fi1_g10_full597_closure_20260921_v1/`合同/身份/路由/封存同样必须留。

此外仍有跨目录依赖：旧官方/NCS uniform30 autoSAM、YOLO-SAM、Alpha 源；`.local/fi1_r0_r4_preparation_20260913/`；旧 `fi1_R3b_train597_20260917/dataset/<scene>/receipt.json` 的 GT 监督绑定。**旧 A 模型不是现 Full 权重，并不意味着整个旧 A 资产根均可删除。**

[scene_asset_routes.csv](scene_asset_routes.csv) 为原 inventory 的逐扫描路由展开，964行，含 records 的真实链接目标，以及48复用场景的actual_front/actual_backend/actual_own_root/bound_geometry/bound_points/own_evidence，能够准确定位每个训练/开发/验证扫描，避免按目录猜测。

## 4. 两数据集从输入到最终结果

| 数据集/阶段 | 实际路径 |
|---|---|
| ScanNet200 当前312输入 | `/home/jia/Wm/Dataset/scannet200/<scene>/`，项目入口 `data/scannet200` 为符号链接；每扫描 RGB-D、pose、内参、原PLY和预处理NPY |
| ScanNet200 输入来源/GT | 同根 `Scannet200_2D_5interval/`、`Scannet200_3D/`、`ground_truth/`；是已有导出来源，未证明保存完整官方 `.sens` 原包 |
| ScanNet200 GD10 | `/media/jia/软件1/scannet_additional_train_stream/fi1_G10_closure_20260921/val/frontend/`；48复用场景实际在`.local/fi1_val48_grounded_stride10_20260918_v4/`；inventory里reuse.front为过时名称，已按全部48后端回执修正派生CSV |
| ScanNet200 后端/实际点集/自身语义 | 同闭环根 `val/scenes/<scene>/backend/`与 `own/D/`；普通264场景正式点集 `geometry_binding/<scene>/points.npz`；48复用场景实际在`.local/fi1_val48_grounded_stride10_evidence_20260919_v1/D/`，后端在同名前缀backend根 |
| ScanNet200共同验证/Alpha | `/media/jia/软件1/scannet_validation_stream/fi1_d_v3_dm_sms1_val312_local_20260902/` 的 class_agnostic_repro/dm_sms1_4090_repro 子链；以逐扫描路由为准 |
| **ScanNet200最终 Full K800** | **`.local/fi1_core_ablation_val312_20261006_v1/scenes/<scene>/full_selected.json`** + `full_scores.npy/full_member_order.json` + 上述几何点集 |
| ScanNet200最终指标 | 同根 `evaluation/validation312_full_AP.npy`、`evaluation/summary.json`、封存/核验 |
| Replica原输入 | `/home/jia/Wm/wm_open-yolo/OpenYOLO3D/OpenYOLO3D/replica/`，`data/replica`为链接；8场景各200 RGB-D/pose帧、内参、原完整mesh；GT在同根ground_truth |
| Replica官方原生候选 | `output/replica/replica_masks/*.pt`；原实验冻结源`.local/official_openyolo3d_assets_20261011/extracted/replica_masks/`亦保留 |
| Replica超点/GD10/YOLO-SAM | `.local/replica_registered_full_20261011_v1/{prepared,frontend,common}/`；当前官方版指向这些旧资产 |
| Replica当前后端/自身证据 | `.local/replica_official_masks_full_20261011_v1/{backend,own,fresh_alpha}/`；own含geometry.json/points.npz/evidence.jsonl |
| **Replica最终 Full K800** | **同官方根 `full/<scene>/selected.jsonl`**，配套pool/features26/scores/sealed |
| Replica最终指标 | 同官方根 `evaluation/Full/`、`evaluation/results.json`、`report.md`、封存/独立核验 |

训练597、开发55、历史验证312及Replica职责分开；“597训练结果”不能用旧K600验证预测或后续不同D-v3对照替代。推理最小交付可不携全部监督缓存，但用户目前要保存训练/复现能力，因此此清单按完整链条保留。

## 5. 模型权重和有效 SL

Full 用 Mask3D、YOLO-World、SAM ViT-B、GD+BERT、AlphaCLIP、champion、official100 D-v3七模型和597匹配B26。**D-v3不在本项目 `pretrained`，B26不在本项目 `pretrained`，champion在 `output`。** 完整文件名和位置见[地图的权重表](../../../资料/代码文件索引与当前主线地图.md#10-full-权重与运行环境)。

D-v3权重根：`/home/jia/Wm/wm_open-yolo/worktrees/OpenYOLO3D-geometry-semantic-56b2c25/pretrained/fi1_d_v3_official100_full_models_20260822/`。

B26：`/media/jia/软件1/scannet_additional_train_stream/fi1_G10_closure_20260921/learning/models/B.joblib`。

SL单模型：`.local/fi1_sl_single200_dev55_20261010_v1/model/{model.pt,normalization.npz}`；五折模型在`.local/fi1_s_semantic_rank_residual_development200_20261010_v2/models/fold0`—`fold4`。保留有效配方`.local/fi1_s_fixed_L_rescore_development200_20261010_v1/`和稳定性/真实重放`.local/fi1_sl_replay_stability_development200_20261010_v1/`及其上游数据。

此次七个D-v3、B26、champion、单L共10项模型哈希匹配原合同，证据[key_model_identity_verified.json](key_model_identity_verified.json)。没有启动训练/GPU推理或重算AP。逐扫描存在性检查同时核对了48复用覆盖，见[资产路由核验](scene_asset_existence_verified.json)，不能用统一路径模板代替inventory的reuse字段；过时的reuse.front名称以各后端回执实际绑定v4为准。17352项路径核查经此解析没有缺失，未重哈希所有数据。

## 6. 哪些已明确，哪些尚不能直接删

- 低风险缓存1534项及三组诊断副本共379.58 MiB **已经处理**，见[执行回执](../storage_audit_20261013/EXECUTED.md)，不重复计入可释放量。
- 旧val48目录已有约 **7.00 GiB** 整理机会：约1.63 GiB保留原路径的重复文件去重，约5.37 GiB关系JSONL可恢复压缩；见[专项审计](../val48_retirement_audit_20261014/README.md)。**尚未执行**，压缩后历史重算需恢复原文件，不能声称能原样即时运行；没有现成恢复工具时不能先删原件。
- `.git`约1.78 GiB临时垃圾候选需按Git维护语义核验；本次不手动删除Git对象。其他历史对象有提交/worktree引用，不能把30.71 GiB全算可释放。
- 没有在本次只读核查中认定某个整个失败实验、测试文件或源码目录“以后明确无用”。当前 Full/SL以外的几何研究仍需要原plan/features/regions/labels/物理划分/最新Mask3D特征资产；用户此前保留Qwen和Full展示的偏好仍有效。
- 925项冻结发布清单是复现包交付范围，182项是静态导入可达范围；都不是“保留这些就可删其余”的结论。某些Qwen/旧训练脚本虽未参与Full前向，其顶层定义仍可能通过历史helper导入。改掉这种耦合需要源码修改及重放验证，不能先删依赖。

因此高收益精简优先处理重复和大体积关系/中间数据；删除tests/source节省的空间很小，还会损失后续修改与核验能力。下一批清理应逐路径注明是否仍被Full/SL/研究合同读取、是否有唯一源/备份、恢复方式及对应回执。

## 7. 本次交付索引

- [Full + SL 主地图](../../../资料/代码文件索引与当前主线地图.md)：分阶段全部入口、核心实现、数据、权重、环境及结果。
- [964扫描资产路由CSV](scene_asset_routes.csv)：精确位置。
- [182代码导入索引CSV](current_code_import_index.csv)与[导入边JSON](current_code_import_edges.json)：现存代码根/同名候选来源/作用范围/哈希（运行时身份以原合同为准）。
- [冻结Full925项清单CSV](full_release_file_index.csv)、[原manifest](full_release_manifest.json)与[97项源码来源](full_release_source_provenance.json)：包内逐文件名和冻结身份。
- [第三方GD/SAM/Alpha组件逐文件清单](full_external_component_file_index.csv)：175项现存源码/配置/编译组件；第三方目录清单不等于实际函数调用列表。
- [旧地图原样归档](../../state_archive/20261014_full_sl_code_map_refocus/README.md)：待继续几何研究的代码/资产导航，不丢失历史。
