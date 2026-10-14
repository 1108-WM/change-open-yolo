# 当前几何研究独立保存

2026-10-14 整理只改变组织与路径适配，不启动新实验。

- `code/`：保留当前局部修复、观察补核、删除块、区域供给、归属、关系解耦和Mask3D实例特征A/B的实现、核验、测试；历史 `tools/tests` 为兼容链接。
- `experiments/`：10月11—13日研究资产/原合同/封存/报告；项目内真实目录已迁入，外盘大资产仍在外盘，此处提供链接。
- `assets/`：实际迁入DINO权重、SAM2权重与SAM2实现；SAM2仅保留为已登记可能路线，没有启动推理。

当前最新Mask3D特征A/B未通过预定池级收益门。下一问题仍是实例先验与父目标条件是否提供有效身份信息，尤其混合父如何明确指向目标；保持原父、邻域、六帧和物理划分，不自动追加阈值/新观测或扩大五折。Full/SL共用资产保持独立保存。

研究入口按[新对话阅读顺序](../../新开对话阅读内容.md)核对最新有效报告及阶段封存。较早研究输入没有因为“未列在Full地图中”而删除；[原研究地图](../../docs/state_archive/20261014_full_sl_code_map_refocus/before/资料/代码文件索引与当前主线地图.md)仍可追溯。

本轮移除26个退役旧策略测试，另把29个Full测试和12个SL测试归回对应目录。当前研究保留107个实现与28个测试文件；10月11—13日当前几何实现和封存资产未按失败结论删除。最新索引见[本轮回执](../../docs/cleanup/deep_active_cleanup_20261014/README.md)。

## 新对话接续与关键源码

清理已完成，下一对话继续几何方向。先按[当前修改方向](../../资料/当前基线修改方向.md)和[最新有效报告](experiments/fi1_mask3d_feature_ab_internal_20261013_v1/audited_report.md)核对结论，再检查同目录contract、阶段seal、completion_verified与supplemental_verified。下一问题是实例先验与混合父目标条件能否提供有效身份信息；目前只是后续问题，尚未实施信息检查或新实验。

| 真实源码 | 阅读职责 |
|---|---|
| `code/tools/export_fi1_mask3d_instance_features.py` | 冻结Mask3D逐点特征、原点映射与同输出接口 |
| `code/tools/fi1_mask3d_feature_ab_core.py`、`run_fi1_mask3d_feature_ab.py` | 最新A/B输入、父条件、监督、模型和执行阶段 |
| `code/tools/evaluate_fi1_mask3d_feature_ab.py`、`verify_fi1_mask3d_feature_ab.py` | 实际编辑点集、完整NT池、保护指标与核验 |
| `code/tools/fi1_decoupled_region_core.py`、`fi1_region_affiliation_core.py` | 前序关系/父资格与区域归属定义，追溯失败边界 |

最新A/B只增加mask_features_head逐点特征，没有显式选定实例query或软mask作为目标条件。B减少损害却没有预定池收益，不能据此否定全部实例先验；也不按旧留出调阈值。固定SL、原父、邻域、六帧和物理划分，保留GT-free目标条件及原点可增删输出要求，推理不用GT挑父。已暴露24场景只称内部开发对照。不要恢复旧总控，设计新任务前核对保留输入和源码新hash。

最新A/B资产入口`experiments/fi1_mask3d_feature_ab_internal_20261013_v1/`经兼容链接指向`/media/jia/软件1/fi1_mask3d_feature_ab_internal_20261013_v1`；外盘大资产未删除。逐文件真实路径与hash见[当前源码索引](../../docs/cleanup/deep_active_cleanup_20261014/current_code_index.csv)。本次正文快照见[清理后几何交接](../../docs/state_archive/20261014_post_cleanup_geometry_handoff/README.md)。
