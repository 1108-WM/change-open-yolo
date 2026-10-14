# 当前源码同步说明

2026-10-14，以本机OpenYOLO3D整理后的状态同步到main。发布内容为源码与阅读材料，无权重、数据或处理后资产。旧分支不删除，不强制覆盖Git历史。

## 范围与身份

Full、SL、几何真实实现及测试、原评测器、依赖的第三方源码和配置已纳入。原根tools/tests/utils等兼容入口保留为仓库内相对链接。`.local/fi1_g10_matched_pilot30_20260919_v1/selection`仅保留一个指向仓库场景划分元数据的相对兼容链接，未上传本机.local实验缓存。MinkowskiEngine和ScanNet依赖源码保存在full_bundle/vendor/Mask3D/third_party；其原固定提交分别为02fc608bea4c0549b0a7b00ca1bf15dee4a0b228和3e5726500896748521a6ceb81271b0f5b2c0e7d2。原.gitmodules不再适用于迁移后的目录，因此不发布失效子模块配置。

[发布文件清单](publication/source_snapshot_manifest.json)记录本次实际文件哈希、兼容链接和本机来源。[当前核心源码索引](cleanup/deep_active_cleanup_20261014/current_code_index.csv)保留整理时的实现身份。运行源码不为上传而重写；README、发布文档及Markdown链接做远端阅读适配。核验命令：`python3 scripts/verify_source_snapshot.py`。

## 文档与资产边界

本机.local中的关键报告复制到research/geometry/experiments、sl_bundle/experiments或docs/experiment_summaries。原报告文字及数值保留；Markdown链接调整到仓库真实文件。实验目录仅含精选报告、小型摘要、协议和诊断脚本，不包含可重跑实验所需的完整输入/模型/预测。

Full/SL/几何目录README中的“模型已集中”“资产已迁入”等描述针对本机整理；本GitHub快照仅发布其源码及身份说明。[外部资产](EXTERNAL_ASSETS.md)明确未上传内容。历史合同和路由里的绝对路径是原环境溯源，不应机械改写哈希或恢复已退役总控。

历史实验、历史快照及第三方README可能引用未发布输出、图片或原项目文档。这些材料不是当前阅读入口；缺失历史引用保留原文以免改写历史。优先用根README、当前代码地图和几何最小阅读路线。

## 运行与核验

源码下载不等于具备运行环境。Full冻结运行包manifest仍包含原权重/文本特征，未改写；对应verify须先补齐原资产。SL必须另备L权重与归一化，几何训练/核验必须另备原划分和缓存。无需为了解架构下载这些大资产。

本次检查源码/链接/核心依赖与适用现有CPU测试，不重训、不重跑597/312/GPU前端、不产生新AP。具体本次核验结果见publication/validation.json；历史报告中的测试数量仅对应当时运行。
