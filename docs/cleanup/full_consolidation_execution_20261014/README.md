# Full / SL / 几何研究整合执行回执

2026-10-14，用户明确授权执行“Full必需内容集中，SL和当前几何研究单独保留”。已执行迁移及退役删除。

统一入口：[Full](../../../full_bundle/README.md)、[SL](../../../sl_bundle/README.md)、[当前几何研究](../../../research/geometry/README.md)。旧 `tools/tests/utils/pretrained/.local` 的保留路径通过兼容链接解析；它们不占第二份数据。外盘597及ScanNet数据保持原盘，集中的是唯一资产路由和场景入口，未额外复制大数据，也未移动外盘或删除旧worktree。

当前Full活跃helper不再导入旧worktree源码，D-v3模型已有项目内副本；冻结自身语义snapshot已有集中副本。生产独立运行包保持原925项发布哈希。GVC/关系优先读独立验证帧合同，972项新旧帧集合和映射完全一致；GD/SAM共用投影/轨迹实现与SAM本身继续保留。

**空间：** 删除65个明确退役根的文件，单链接文件分配块去重后的保守统计为 **22.4008 GiB**；删除执行期间磁盘空闲净增 **22.2138 GiB**。净增受小归档、其他同盘活动影响，完整值及逐路径统计见 `deletion_executed.json`。不是把目录迁移当作释放，也未加此前379.58 MiB清理或旧7 GiB机会值。

**删除范围：** val48 A/B自动SAM与C非当前对照分支的frontend/backend/own产物；YOLOE（包含MobileCLIP）、SigLIP、ProxyCLIP/ViT-B16及14个旧对照输出；退出的参考实现和小旧模型；重复下载ZIP。GD5/D实际来源、GD10、SAM权重/源码、CLIP-B/32、AlphaCLIP、Mask3D、正式597B26、SL、研究资产保留。退出的源码及小报告归档后删除活跃入口；补回当前Replica测试和GD10重投影入口后，实际移出的活跃历史代码为878份，详见当前分区JSON，后续以该JSON计数为准。

历史代码和小记录在 `archive/retired_20261014/` 的gzip归档，每个归档有逐文件SHA256清单和解包字节核验。退役大权重只留SHA256身份和来源代码，不在归档中再保留一份大权重；Any3DIS只保留已核对一致的原ZIP。Qwen及环境、Full展示和ZIP按此前保留偏好继续保存，Qwen权重独立移到 `archive/user_retained/`。

**验收：** 独立Full 43测试及4subtests、当前Full/Replica/SL/几何57测试通过；六大视觉权重身份和925发布哈希通过；普通验证及旧val48两个场景全池B26分数/K800成员排序逐值一致；SL开发55一个场景类别和实际去类结果逐条一致；964场景统一路由/模型/帧/退役不存在检查17,438项通过。未执行597重训、312全量GPU重新生成或新的几何研究。

| 文件 | 内容 |
|---|---|
| `journal.jsonl` | 每条实际迁移、复制、归档和删除开始/完成记录 |
| `current_code_index.csv` / `code_relocation_manifest.json` | 整理后的真实源码路径及原/新哈希 |
| `code_partition_executed.json` | 保留、动态补留和退役源码逐项分类 |
| `deletion_executed.json` / `retired_weight_identities.json` | 删除逐根统计与退役权重身份 |
| `verification_frames_extracted.json` / `frame_interface_verified.json` | 972原验证帧身份提取与新接口核验 |
| `backend_import_verified.json` | Full后端实际从项目集中源码导入，未载入旧worktree模块 |
| `verification.json` | 17,438项路由、模型和回放核验 |
| `replay/` | 两个Full场景和一个SL场景的迁移验收输出 |
| `consolidate.py` / `retire.py` / `verify.py` | 本次执行与核验实现；迁移/删除脚本不是日常重复运行命令 |

历史seal/contract/receipt原样保留；它们记录原代码哈希，活跃源码的路径定位及帧接口已适配，新哈希见本目录。继续研究读原实验封存输出和源码快照；新执行建立新合同，不恢复已完成的旧总控。

`.git`未处理，未知实验没有仅凭名字整删；其他旧实验输入目前仍保留。磁盘本体容量在 `post_storage.json`，目录迁移前后的`.local/pretrained`数字不能直接解释为释放量。
