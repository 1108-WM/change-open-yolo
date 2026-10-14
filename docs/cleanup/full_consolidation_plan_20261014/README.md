# 当前 Full 集中整理与退役候选清单

日期：2026-10-14。用户本轮明确要求“先给清单，不直接删除”。本目录仅生成清单和核查证据；**未迁移、未删除、未改算法、未启动训练/推理**。

## 判断

可以将 Full 真正需要的代码、权重、数据和结果集中到独立主线目录，把 SL 和后续几何研究单独保留，再退出明确放弃的实验分支。这比继续让生产入口沿用旧实验目录更适合维护。

但“当前 Full 不用”只证明它不应混在主线，不能直接证明其没有历史复现、SL、后续研究、用户指定保留用途。本清单给出明确动作顺序：**迁出必需内容并解除路径/导入耦合 → 同输入回放验证 → 归档必要实验记录 → 删除已退役的大资产/移除活跃旧源码。** 新实验无需继续依赖旧路径。

## 自动 SAM 配置能否删

**已放弃的 A/B 自动 SAM 候选生成配置可以退役。** 不需要继续把它们的所有观测、轨迹、后端、Alpha 特征和预测当作生产 Full 常驻资产。它们与当前 GD10 的最终 T 候选不同。

以下三种东西需要先从“autoSAM”名字下分离：

| 内容 | 现状 | 整理动作 |
|---|---|---|
| A/B autoSAM 候选方法及比较产物 | 已放弃，非当前 GD10 候选 | 简要报告/合同/源码身份留档，解除当前引用后退役大资产和专用入口 |
| `export_sam_automatic_observations.py` 等公共函数 | GD10仍导入RLE解码、掩码投影、模型包装；hierarchy/sIoU/consensus/merge也用于GD10 | 将实际公共函数抽到 `full_core`；随后旧autoSAM生成入口可退出活跃目录，不能先删公共实现 |
| 历史 common autoSAM 目录的验证帧身份 | GVC与关系特征通过 `_frame_contract` 从summary读取，或fallback观察记录恢复 `frame_id/frame_index` | 精确提取为 `verification_frames.json`，改成独立输入；新旧特征一致后，不再为这份名单保留大autoSAM观测/RLE |
| SAM ViT-B 权重/源码 | GD框提示SAM、YOLO-SAM验证掩码和Alpha区域都仍使用 | 保留；停用“自动生成候选”不等于停用SAM模型 |
| uniform30 YOLO-SAM 验证观测 | GVC实际读取其 `observations.jsonl`及对应点集 | 保留/迁移；与旧autoSAM A/B候选不同 |

具体证据：

- `tools/build_c1_gvc_paper_reference_quality_ledger.py:_frame_contract`（约81行）只用 common autoSAM 的帧名单；`_scene_records`（约185行）从 **YOLO-SAM** 读取实际验证观测。
- `tools/build_dm_sms1_no_gt_relation_ledger.py:_scene`（约50行）调用同一帧名单并使用YOLO-SAM建立候选关系。
- D-v3 stage C 读名为 `automatic_observations.jsonl` 的文件，但其 `automatic_root` 在当前GD10后端指向 GD10 的 `automatic_raw_frames`，不是旧A/B候选。文件名不能决定观测来源。
- `tools/run_fi1_val48_grounded_stride10.py:materialize_reprojected_cache` 从旧 **D=GD5** 观测的RLE选择stride10并重投影；D不是自动SAM，需保留所需来源或正式迁出后冷归档。

[autoSAM代码逐文件清单](autosam_code_candidates.csv)共57个相关源码/测试：48个可移出活跃主线的历史归档候选，9个当前仍保留的公共实现/测试。这是入口及静态依赖分类，不是57个文件可以全部删除。

[验证帧身份提取清单](autosam_common_frame_extraction_plan.csv)覆盖972项：597训练+55开发+312验证+8Replica。必须提取**已有真实帧集合**，不能重新uniform选帧代替并声称输入不变。

## 为什么 Full 还读旧目录

| 旧来源 | 原因 | 独立整理后保存到哪里 |
|---|---|---|
| 旧 val48 | 当前312中48扫描直接复用已经验证的GD10点集/自身语义；GD10来自旧GD5的已保存二维RLE | `full_bundle/artifacts/scannet/val/<scene>/`；对应原始来源/合同冷归档 |
| 旧几何 worktree | 冻结七模型对应的D-v3推理及特征函数在那里；当前脚本动态追加tools路径 | `full_bundle/code_sources/geometry_worktree/`，再抽出稳定 `full_core/geometry/` |
| `.local/fi1_unified_evidence_20260910/workspace` | 自身YOLO/Alpha视角与全词表证据的冻结helper，仍按snapshot路径导入 | `full_bundle/code_sources/evidence_snapshot/`，再抽到 `full_core/semantics/` |
| 历史runner/训练文件 | 推理helper、特征定义和评测函数写在实验runner/GT dataset文件中，导致顶层传递导入 | 将被调用函数移到独立模块，旧runner归档；保留输入/输出同一性测试 |

这些是组织/实现耦合，并不要求算法永远读取旧目录。独立发布ZIP已经做过一部分路径适配，可作为整理起点；它不包含当前完整训练/Replica/SL/大数据，也不能直接替代全部现存依赖。

## 建议的新目录

```text
full_bundle/
  full_core/                 生产推理公共函数，按数据/前端/几何/语义/B26/评测分模块
  cli/                       preprocess、train_b26、infer、evaluate、verify
  code_sources/              初始实际源码版本与来源，完成抽取后放冷归档
  vendor/                    Mask3D、YOLO-World、GD、SAM、AlphaCLIP所需实现
  metadata/                  场景身份、标签映射、词表、固定帧合同
  models/                    视觉权重、champion、official100 D-v3、597 B26
  datasets/                  597/55/312/Replica输入与GT（保持各自数据角色）
  artifacts/                 GD10、native、YOLO-SAM、实际点集及自身YOLO/Alpha
  training/                  597 B26特征/监督、名单、训练身份
  results/                   Full K800成员/点集/得分/AP/封存
  provenance/                源→新路径、原hash、新hash、源合同和迁移核验
sl_bundle/                   有效S/L/SL代码、模型、归一化、训练及结果
research/geometry/           当前仍要继续的几何研究入口与资产
archive/                     退役实验的报告、合同、源码/模型快照压缩包
outputs/Full点云展示_20261011/及ZIP  继续保留
```

“整合一起”首先落实为独立可运行代码和统一资产布局。小代码/小模型可直接集中；跨盘大数据不建议重新复制一整份消耗空间，可在所在文件系统重命名整理，用唯一配置文件给出统一资产根。若要物理全部同盘集中，需要另外规划容量，不能把复制两份当精简。

**[完整迁移源→目标清单](full_migration_source_target_plan.csv)**共8100行，涵盖源码/同名历史版本、视觉源码/权重、ESAM必需元数据、964扫描资产路由、训练/原始数据、SL和结果。重复共享根只是一条扫描的路由描述，实际整合必须去重，不能把一个共享根复制964次。目录条目是需要分拣的来源范围，不承诺整根全热存。

迁移还要处理：JSON/JSONL内的绝对点集/图像/缓存路径、配置中的cwd路径、动态 `tools.__path__` 和snapshot参数、旧模型加载环境、调用到的词表/配置/编译依赖。原实验合同/receipt的绝对路径及哈希保留为历史，不直接重写；新副本通过独立路径映射及新封存说明重定位。

## 旧 val48 大数据退役候选

本次对以下现存分支做文件块统计，不跟随符号链接；分支各自按inode计数。跨分支硬链接、必须迁出的输入、保留小记录和归档开销会影响净释放量，因此不是删除后精确可释放量。

| 旧配置 | frontend+六批backend/own evidence合计 | 决策 |
|---|---:|---|
| A：autoSAM uniform30 | 约2.54 GiB | 放弃配置；依赖/图例来源迁出并保留简要记录后退役大数据 |
| B：autoSAM stride5 | 约15.62 GiB | 同上，大体积关系表和后端是占用大头 |
| C：GD-SAM uniform30 | 约1.54 GiB | 非当前GD10，对照记录归档/相关引用迁出后可退出 |
| D：GD-SAM stride5 | 约4.61 GiB | **先保留或迁出GD10所需observations/RLE/relations/manifests**，其余对照产物再退出 |

A+B共约**18.16 GiB候选范围**。与上一批7.00 GiB去重/压缩清单存在重叠，**不能相加**。本次没有证明这18.16 GiB能立即全部释放。

具体52个存在的分支目录列在[val48数据逐路径清单](val48_branch_data_candidates.csv)，例如：

```text
.local/fi1_val48_frontend_full_pipeline_20260914/frontend/arms/A/
.local/fi1_val48_frontend_full_pipeline_20260914/frontend/arms/B/
.local/fi1_val48_frontend_full_pipeline_20260914/full_batches/batch01/backend/A/
.local/fi1_val48_frontend_full_pipeline_20260914/full_batches/batch01/backend/B/
.local/fi1_val48_frontend_full_pipeline_20260914/full_batches/batch01/own_evidence/A/
.local/fi1_val48_frontend_full_pipeline_20260914/full_batches/batch01/own_evidence/B/
```

CSV列出batch01—batch06实际条目，正文不提供可直接执行的通配删除命令。旧autoSAM还散布于早期output/训练根，需按具体当前引用和训练GT绑定逐路径迁出，而不是搜索“sam”就整删。

## `_external` 和 `pretrained`：逐项决定

当前重新测量的容量仅用于下列候选，不沿用上一轮整体审计值。

| 路径 | 大致占用 | 当前用途 | 建议 |
|---|---:|---|---|
| `pretrained/yoloe/mobileclip_blt.pt` | **571.46 MiB** | YOLOE文本编码器；当前Full/SL不用 | **列入删除候选**；退出YOLOE后可删，再跑YOLOE需恢复/重下载 |
| `pretrained/yoloe/`全根 | 766.36 MiB | MobileCLIP + YOLOE-v8s/m/l分割权重 | 退出旧YOLOE框/掩码对照后可删；全根数字已包含MobileCLIP，不能再加一次 |
| `_external/YOLOE/` | 65.71 MiB | 旧YOLOE实现，Full用的是YOLO-World | 同上；`.venvs/yoloe_gpu`只是环境名字，当前Full/SL用它，不能跟着删环境 |
| `_external/SigLIP/` | 778.14 MiB | 旧NCS4语义比较权重/词表 | 退出SigLIP历史重算后可删；保留简要比较结论/合同 |
| `pretrained/checkpoints/ViT-B-16.pt` | 334.59 MiB | 旧ProxyCLIP CLIP图像编码权重 | 退出ProxyCLIP历史重算后可删；不是Full的AlphaCLIP ViT-L-14，也不是YOLO的CLIP-B/32 |
| `_external/ProxyCLIP/` | 4.82 MiB | 旧NCS5 ProxyCLIP实现 | 同上；当前几何调用`ncs5_proxyclip_core.load_dinov2_model`走timm/DINO，不加载此目录或ViT-B/16；该helper源码保留 |
| `_external/Any3DIS_unofficial/` | 81.20 MiB | 历史方法参考代码 | 先保留一份源码归档，再移出活跃目录；未来需要时恢复 |
| `_external/OV3D-CG/` | 100.08 MiB | 参考/对比代码，当前实现不调用 | 同上 |
| `_external/SegDINO3D_official/` | 5.45 MiB | 参考代码，当前自有DINO/Mask3D适配不调用 | 同上 |
| `_external/ESAM/` | 57.95 MiB | 仍提供预处理默认的训练列表和标签映射 | **先迁出metadata再退出其余代码**；不能整根先删 |
| `pretrained/z3_full_semantic_reliability_official100_20260812/` | 0.15 MiB | 旧语义头，非Full/SL | 压缩进历史模型归档后移出活跃目录，空间收益小 |
| `pretrained/z6f_full_inference_bundle_official100_20260812/` | 0.25 MiB | 旧selector/gate，非当前B26/SL | 同上 |
| `_external/Any3DIS_unofficial.zip` | 65.74 MiB | 下载源码ZIP | **418/418文件与解压目录字节一致**，解压/源码归档保留一份即可；不要同时把两份都删 |
| `_external/GroundingDINO.zip` | 10.66 MiB | 下载源码ZIP | 66项相同，`ms_deform_attn.py`一项有本地修改；保留原版/补丁身份后可去掉重复下载包，不删GD实现 |

退出YOLOE、SigLIP、ProxyCLIP这三组后，模型/配套实现合计约 **1.90 GiB**候选（未加图像特征和旧实验输出），不包含仍需保留的Qwen/SAM2/DINO。旧对照输出另外有14个来源根列在[旧模型对照产物清单](retired_model_comparison_artifact_candidates.csv)，这些容量沿用旧盘点，须迁出当前引用/归档报告后再清理大特征。

**需要保留的模型/源码：**

- Full：Mask3D、YOLO-World、`pretrained/clip-vit-base-patch32/`、SAM ViT-B、GD+BERT、AlphaCLIP、champion、official100 D-v3七模型、GD10匹配597 B26及配套词表/配置。
- Full第三方：`_external/AlphaCLIP/`、`_external/GroundingDINO/`、`_external/segment-anything/`；ESAM只需抽出实际元数据依赖。
- 几何研究：`pretrained/checkpoints/dinov2_vits14_pretrain.pth`仍用于已登记局部视觉特征；当前Mask3D先验/实例特征也需要Mask3D权重。
- 后续已登记路线：`pretrained/sam2/sam2.1_hiera_small.pt`和`_external/sam2/`被最新next_route_readiness列为可能的新边界先验，先保留；它与已放弃的autoSAM A/B配置不是一回事。
- 之前明确保留：Qwen权重和环境、Full点云展示及ZIP；本轮“用不到的可以删吗”未指定退出Qwen，所以不列入当前删除候选。

**补正：上一版Full权重表漏列YOLO-World的CLIP-B/32文本模型。** 当前配置 `pretrained/configs/yolo_world_v2_x_vlpan_bn_2e-3_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py:33`明确读取 `pretrained/clip-vit-base-patch32/`，现已补入主地图。它约580.68 MiB，应保存，不能与可退役的ViT-B/16或MobileCLIP混同。

## 代码多、阅读负担大怎么处理

[代码分区逐文件清单](code_partition_candidates.csv)覆盖当前顶层 `tools/tests/utils`共1261个 `.py/.sh`：

| 分类 | 文件数 | 动作 |
|---|---:|---|
| 当前Full/ReplicaFull/SL导入依赖（项目内） | 157 | 先集中实际版本；随后拆出调用函数，去掉历史runner顶层耦合 |
| 当前对应测试/包装 | 40 | 随主线一起保存 |
| 当前几何研究入口、导入、测试/包装 | 151 | 移至研究工作区或保留研究导航，不混入生产地图 |
| 未落入以上范围的历史代码候选 | 913 | **先归档再从活跃目录移出**；检查动态入口/合同快照后决定是否永久删除，不能把静态未命中直接当“无用途” |

全源根Full/SL静态闭包仍是182文件（包含worktree/snapshot）；本次研究静态闭包188文件，与Full有交集。统计不能直接相加为唯一代码数。各行记录hash、作用范围和保留/归档理由。

这可以显著减轻活跃源码和阅读入口的负担。保留小型历史源码压缩包并不要求继续在`tools/`摆放几百个旧实验入口；测试保留当前Full/SL/几何实际需要的即可。方法应按**运行函数/输入身份**拆模块，不按文件名里“train/automatic/gt”机械删文件。

## 迁移完成后才能执行的验收

这是保持Full行为的工程验收，不是再次选择算法或调参：

1. 明确运行代码版本，消除旧worktree/snapshot动态导入，独立根离线导入成功。
2. 提取精确验证帧合同，GD10投影/轨迹、GVC/关系/D-v3点集与原输入一致；`automatic`公共函数拆出后仍可重生成GD10。
3. 自身YOLO/Alpha和完整X26、B26分数、K800成员/排序/点集与冻结结果一致，SL同输入模型/归一化重放一致。
4. 两数据集、597训练名单/监督/模型身份、48复用扫描和共享路径全部迁移可解析；旧合同留档，新重定位和新代码独立封存。完整597训练重跑不是迁移验证的默认要求。
5. 当前研究输入、原点域/物理划分、指定展示/ZIP与Qwen偏好仍保留；此后逐条移除退役候选，不手动删除Git对象。

## 清单入口

| 文件 | 内容 |
|---|---|
| [full_migration_source_target_plan.csv](full_migration_source_target_plan.csv) | 8100条源→目标迁移/提取计划，未执行 |
| [model_external_candidates.csv](model_external_candidates.csv) | 27个模型/第三方根，当前容量、保留/退役原因 |
| [weight_file_candidates.csv](weight_file_candidates.csv) | 24个现存权重文件逐文件决定；未重hash视觉大权重 |
| [val48_branch_data_candidates.csv](val48_branch_data_candidates.csv) | 52个旧前端/六批后端/自身证据分支候选，当前分配空间 |
| [autosam_common_frame_extraction_plan.csv](autosam_common_frame_extraction_plan.csv) | 972个旧common验证帧身份提取计划 |
| [autosam_code_candidates.csv](autosam_code_candidates.csv) | 57个autoSAM命名/共用源码和测试的保留/归档分类 |
| [code_partition_candidates.csv](code_partition_candidates.csv) | 1261个tools/tests/utils源码和包装的主线/研究/归档候选分类 |
| [retired_model_comparison_artifact_candidates.csv](retired_model_comparison_artifact_candidates.csv) | 14个YOLOE/SigLIP/ProxyCLIP对照产物来源根 |
| [asset_source_reference_evidence.csv](asset_source_reference_evidence.csv) | 顶层源码路径/名字引用；helper常数/环境名字不等于模型实际加载 |
| [download_zip_comparison.json](download_zip_comparison.json) | 两下载ZIP逐文件比对；GD原版/本地补丁差异 |
| [summary.json](summary.json)、[audit_plan.py](audit_plan.py) | 分类汇总与只读清单生成脚本，无删除命令 |

本清单可用于下一步实现独立Full与SL目录，再执行具体退役；本轮仅交付清单。
