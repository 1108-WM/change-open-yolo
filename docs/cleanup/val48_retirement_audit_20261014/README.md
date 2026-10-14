# 历史 val48 大目录空间审计（2026-10-14）

审计对象：`.local/fi1_val48_frontend_full_pipeline_20260914`，上轮目录盘点约 **24.85 GiB**。本轮只读取原实验数据，在本目录保存审计结果；**没有删除原文件，也没有实施压缩归档或新的去重**。

**找到合计约 7.00 GiB 的潜在节省量：保留原路径的重复数据去重约 1.63 GiB，加上可恢复压缩约 5.37 GiB。两份清单已排除重叠，不能把目录整体删除。** 汇总见 [proposed_space_summary.json](proposed_space_summary.json)。

## 重复副本：保留内容及两个原路径

对不少于 1 MiB 的文件先按尺寸和片段筛选，再对 435 个候选计算完整 SHA256，确认 211 组重复内容。排除与压缩清单重叠的一组后，可整理 **210 组、219 个多余副本，约 1.63 GiB**。逐项保留路径、待合并路径和 SHA256 见 [dedup_candidates.csv](dedup_candidates.csv)，完整分组见 [exact_duplicates.json](exact_duplicates.json)。

这些副本主要是观测在不同阶段的复制，以及证据、视图与预测的复制。它们有用途，**应保留各自路径和字节内容**；建议沿用上一轮的只读硬链接方式共享底层存储，不能直接删除其中一个路径。

执行前仍需复核哈希及任务状态，记录两侧原权限/时间戳。硬链接会共享 inode、权限和修改时间，不能保持两份文件独立可写；未来如需原地修改或重生成，应先复制为独立文件再写入。清单中的“待审”不是已执行或无条件安全的声明。

## 已确认的可恢复压缩机会

580 个关系表实际占用 **5.618 GiB**。逐文件进行 gzip level 3 内存压缩、流式解压及 SHA256 校验后，压缩文件预计占用 **0.251 GiB**，预计可净节省 **约 5.37 GiB**（归档索引等少量开销尚未计入）。这不是已经释放的容量。

| 数据 | 文件数 | 原分配空间 GiB | 压缩后预计 GiB | 处理建议 |
|---|---:|---:|---:|---|
| `same_frame_hierarchy_relations.jsonl` | 192 | 0.964 | 0.031 | 可恢复冷归档，保留原始哈希和路径映射 |
| `merge_round_relations.jsonl` | 192 | 1.375 | 0.035 | 同上 |
| D-v3 `stage_b/candidate_relations.jsonl` | 196 | 3.279 | 0.184 | 同上；历史重评分也需先恢复 |
| 合计 | 580 | 5.618 | 0.251 | **不可当作无用日志永久删除** |

逐文件名单见 [archive_candidates.csv](archive_candidates.csv)，实测汇总见 [archive_measurement.json](archive_measurement.json)。580 个文件的原内容与原实验回执哈希一致，压缩解压往返后的内容也全部一致。原件均未移除，压缩包未落盘；清单是待审方案，不是删除授权。

关系表含大量重复字段，因此压缩率很高。它们确实属于实验数据，不是代码或轻量 Markdown。最大单表约 1.26 GiB，不能根据扩展名 `.jsonl` 判断体积或重要性。

## 为什么不能直接删除

这套历史实验的四臂是 automatic SAM / grounded SAM 与 uniform30 / 原始帧 stride5 的组合，**D 臂不是当前 GD10**。六批、每批八场景、四臂共 192 个任务已经完成；另有少量 smoke 阶段文件。

**后续选型核对：** 整个val48实验系列还在独立目录完成了Grounded-SAM stride10/15/20，共七种前端配置；不是只做过本目录的四臂。旧full597 B/Q、K600对照中GD10的B头AP为39.244266%，GD5为38.197722%；GD10不是所有头/指标都最高。当前主线已确认为GD10匹配597 B26、K800，本轮不改回GD5。GD10/15/20由保留的旧D掩码选帧重投影得到，因此旧D观测仍有来源/重放用途。配置与AP来源见[历史对照](../../state_archive/20261009_before_compaction/资料/结果.md)，实际完成结果已核对各`full597_q/summary.json`。

但实验结束不等于所有中间产物已经无用，代码中仍有以下真实读取：

| 入口 | 实际依赖 | 空间整理的影响 |
|---|---|---|
| [历史最终评测](../../../tools/evaluate_fi1_val48_complete.py) | 已绑定的最终几何、点集、预测及评测身份输入 | 本轮核验的 1,868 个输入全部保留，不与 580 个归档候选重叠 |
| [历史 Full 绑定](../../../tools/evaluate_fi1_frontend_comparison.py) 的 `full_bind` | 恢复已有结果前仍校验 backend receipt 的全部输出 | 关系表归档后，必须先恢复原路径，旧绑定/续跑才可通过 |
| [历史 A/D full597 重评分](../../../tools/evaluate_fi1_val48_ad_full597_q.py) 的 `resolve_jobs` / `quality_index` | 直接绑定并读取 `stage_b/candidate_relations.jsonl` | 不能只保留终评输入就宣称旧重评分仍可直接运行；B/C 包装入口同样受影响 |
| [GD10 派生前端](../../../tools/run_fi1_val48_grounded_stride10.py) 的 `materialize_reprojected_cache` | 旧 D 的 observations、`same_frame_mask_relations.jsonl`、帧摘要和原始 RGB-D/姿态 | 保留。这种 mask relation 与表中 hierarchy relation 是不同文件 |
| [模块图绘制](../../../tools/render_fi1_module_illustrations.py) | 旧 A 的 observation、tracks、pair union 等 | 保留对应源数据 |

旧 D-v3 冻结推理实现也会读取候选关系并汇总关系类型。独立阶段核验需要同样的原始表。因此建议的“冷归档”是**保留可恢复原件的无损压缩**，不是承诺这些数据以后再也不用。

如果后续批准归档，应先建立原路径、原 SHA256、权限和时间戳索引，写出并独立解压验证压缩文件，提供恢复入口与入口提示，再逐项移除已验证归档对应的原件。原实验回执及哈希不应改写。恢复旧任务时需预留解压空间；没有完成这些步骤，不能执行删除。

## 必须继续保留的部分

- 原历史最终评测身份所列的 **1,868 个本目录输入**：内容哈希 1,868/1,868 通过，约 1.92 GiB。这只是最低直接终评依赖集合，并非所有应保留文件的总量。
- 已完成的自身语义 `evidence.jsonl`（约 2.21 GiB），历史重评分仍读取它们。
- GD10 重投影使用的旧 D 观测、mask relations、selection、帧摘要、源合同和回执；不能把整臂删除。
- 最终几何、points、预测、模型及其身份/回执、源代码快照、模块插图所需原始数据。
- 当前 Full、有效 SL、当前几何研究的模型和中间输入，以及用户指定的 `outputs/Full点云展示_20261011`。它们不在这次整理执行范围内。
- 其余 Alpha 缓存、视图导出、绑定中间文件：本轮没有建立足以永久删除它们的证据，暂不列为可删。

旧 frame cache 已在 2026-09-16 清理，不能再算一次释放量。

## 证据及适用范围

- [files.csv.gz](https://github.com/1108-WM/change-open-yolo/blob/main/docs/EXTERNAL_ASSETS.md)：76,977 个普通文件的尺寸、实际文件块、inode、链接数与 mtime 快照；不跟随目标目录内符号链接。文件块合计与包含目录开销的 `du` 数字略有差异。
- [terminal_input_integrity.json](terminal_input_integrity.json)：1,868 个原终评输入的预期/实际 SHA256；没有运行新 AP。
- [current_contract_checks.json](current_contract_checks.json)：七个当前 Full、SL、几何实验与展示入口文件未发现该旧目录的直接引用。**这不等于穷尽所有动态或传递依赖**，因此仍保留上述已发现的旧 D 来源等。
- [incoming_references.json](incoming_references.json)：扫描 2,723 个其他实验顶层 JSON、工具脚本和展示 manifest，发现 64 个文件引用旧目录；三个超过 16 MiB 的 JSON 未扫描，读取错误为零。更广的嵌套场景扫描耗时较长，已中止；此结果不覆盖所有逐场景合同和动态路径。
- [archive_incoming_dependencies.json](archive_incoming_dependencies.json)：归档清单与上述引用精确交叉。A/D 重评分 v1、v2、v3 以及 B/C 重评分的身份文件各绑定 96 个候选关系表，进一步确认这些表必须可恢复；另有旧空间盘点文件记录其路径。
- [source_unchanged_check.json](source_unchanged_check.json)：76,977 个原文件的尺寸、mtime、inode、device 均与盘点时一致；归档候选与终评直接输入集合不重叠。元数据检查不替代上面的内容哈希核验。
- [measure_archive.py](measure_archive.py)：内存压缩往返及原实验回执核验脚本。

全项目此前已经执行的 379.58 MiB 缓存清理与诊断副本去重，见 [上轮执行报告](../storage_audit_20261013/EXECUTED.md)，不与本轮潜在节省量重复计算。
