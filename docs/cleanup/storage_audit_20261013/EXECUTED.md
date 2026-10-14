# 已授权清理执行报告

完成时间：2026-10-13T23:31:49+0800。用户授权低风险缓存直接清理，并询问重复诊断数据是否可只保留一份；按保留路径兼容性的方式完成。

| 操作 | 实际结果 | 移除的数据分配 |
|---|---|---:|
| 低风险缓存白名单 | 删除 1534 个文件，未跳过条目 | 24.60 MiB |
| 重复诊断数据 | 3 对文件合并为 3 份只读数据，6 个原路径全部保留 | 354.98 MiB |
| 合计 | 没有删除实验目录 | **379.58 MiB（0.371 GiB）** |

释放量按实际删除的单链接缓存与被替换副本的数据块计算，不包括随后新增的少量执行记录。分区可用空间从 50.88 GiB 变为 51.25 GiB；该差额还受并发文件系统活动影响，不能代替逐文件账目。缓存以后运行时可能再生成。

重复数据涉及以下两个目录的 `pair_unions.jsonl`、`member_removal_dataset.jsonl` 和 `summary.json`：

- `docs/diagnostics/official100_fi1_d_v3_stage_c_v2_member_dataset_20260822/`
- `docs/diagnostics/official100_fi1_d_v3_stage_c_v2_member_dataset_reproduction_20260822/`

两个路径现在指向同一份内容，均为普通文件硬链接，不依赖符号链接解析。每对文件具有相同 inode，链接计数为 2；六个路径的 SHA256 全部与原审计一致。共享文件设置为 0444，只读，以防止误写使另一条路径的内容一起改变。原源文件修改时间保留，复现路径的修改时间随共享 inode 采用源文件时间；执行日志保留替换前的路径身份和权限。文件内容、JSON 内部路径及原合同哈希没有改写。

原生成器 `build_ncs_fi1_stage_c_v2_member_dataset_gt.py` 的第 301—303 行会拒绝向已存在目录生成结果，原核验器读取这些文件。若以后重新生成，应使用新结果目录；不要改写这份历史数据，也不要直接把共享文件改回可写。单独查看两个目录的文件大小仍会看到同样大小，这不代表还存了两份数据。

执行后核验：

- 1064 个对应源文件哈希保持不变；83 个未找到对应当前源码的缓存全部保留且哈希一致。
- 原白名单缓存路径全部移除。
- 三对重复文件的哈希、共享 inode、链接数、只读权限核验通过；没有遗留替换临时文件或回滚副本。
- SL/B26、SL 单模型与归一化、Mask3D 等 14 项既有关键资产哈希全部一致。
- 没有修改 Full/SL 配方、权重或预测内容，没有删除 `.local` 实验目录，没有运行新训练、推理或 AP。

初次预检遇到系统 Python 3.8 不支持 `Path.is_relative_to`，在任何删除或去重之前退出；改为兼容的路径检查后完成。实际操作逐项保存在 [cleanup_actions.jsonl](https://github.com/1108-WM/change-open-yolo/blob/main/docs/EXTERNAL_ASSETS.md)，最终身份核验与空间账目在 [cleanup_execution.json](cleanup_execution.json)。

大量空间仍应优先审查 `.local` 的历史大缓存：此前盘点为 134.77 GiB。首先检查 24.85 GiB 的 `fi1_val48_frontend_full_pipeline_20260914`，尤其 18.07 GiB 的 `full_batches` 中的阶段性关系数据；需保留仍被引用的终态几何、预测、模型身份与评测证据。这是下一批审查对象，**不是已证明可删除 24.85 GiB**。R0—R4 12.40 GiB、历史 val312 8.41 GiB 也有当前依赖，不能整删。

额外检查发现两个 Replica 实验的 office3/room0 掩码虽然文件大小相同，文件开头内容已不同，并非重复副本；证据见 [local_same_size_counterexamples.json](local_same_size_counterexamples.json)，两份均未修改。失败、文件名相同或大小相同，都不能替代实际依赖和内容核验。
