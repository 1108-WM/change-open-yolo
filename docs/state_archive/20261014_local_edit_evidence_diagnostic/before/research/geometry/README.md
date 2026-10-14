# 当前几何研究独立保存

2026-10-14 整理只改变组织与路径适配，不启动新实验。

- `code/`：保留当前局部修复、观察补核、删除块、区域供给、归属、关系解耦和Mask3D实例特征A/B的实现、核验、测试；历史 `tools/tests` 为兼容链接。
- `experiments/`：10月11—13日研究资产/原合同/封存/报告；项目内真实目录已迁入，外盘大资产仍在外盘，此处提供链接。
- `assets/`：实际迁入DINO权重、SAM2权重与SAM2实现；SAM2仅保留为已登记可能路线，没有启动推理。

最新[N/T区域身份诊断](experiments/fi1_nt_region_identity_diagnostic_20261014_v1/report.md)阶段0—4已执行：fit96抽12场景/96父，区域图、P0/P1配对先验、无GT关联及四探针完成，必要测试与独立复算通过。先验有局部身份信息，但直接mask伤害好父、整区限制机会；后续同父query小样本已执行，未支持扩大训练。前序Mask3D特征A/B失败结论保留。保持原父、邻域、六帧和物理划分，不追加阈值/新观测或扩大五折；Full/SL保持原身份。

后续以用户修订的[同一父目标query A/B](../../docs/FI1_QUERY_SAME_PARENT_AB_NEXT_STEP_20261014.md)为准：A/B同target/监督/忽略点/损失/预算，B只加非GTquery概率；先小样本及置零/打乱，通过后才内部场景隔离评估，混合父拆分另立任务。原诊断报告/decision里的“监督跟随query身份”后续建议已修订，原封存不改；16父/64轮小样本已执行：[报告](experiments/fi1_query_same_parent_pilot_20261014_v1/report.md)。B无A之外的有效收益，query置零/打乱未降低改善；停止扩大，未开展场景隔离评估。

最新[固定跨场景评估](experiments/fi1_query_transfer_eval_20261014_v1/report.md)只评原selection16、128原父，现有epoch64模型/归一化/.8锁定；A/B均伤害好父、B无池增量，暂停当前修复器扩训。2测试、896点集/9480池记录及768模型回放通过。原pilot/审计/旧诊断封存不变，无新训练、最终holdout或SL/AP。

研究入口按[几何问题分析路线](../../新开对话阅读内容.md)读取最新有效报告与相关源码，按疑点追溯历史；阶段封存仅在身份冲突、结果异常或复现执行时查阅。较早研究输入没有因为“未列在Full地图中”而删除；[原研究地图](../../docs/state_archive/20261014_full_sl_code_map_refocus/before/资料/代码文件索引与当前主线地图.md)仍可追溯。

本轮移除26个退役旧策略测试，另把29个Full测试和12个SL测试归回对应目录。当前研究保留107个实现与28个测试文件；10月11—13日当前几何实现和封存资产未按失败结论删除。最新索引见[本轮回执](../../docs/cleanup/deep_active_cleanup_20261014/README.md)。

## 新对话接续与关键源码

清理已完成，继续几何方向。先读[当前修改方向](../../资料/当前基线修改方向.md)顶部和[N/T诊断报告](experiments/fi1_nt_region_identity_diagnostic_20261014_v1/report.md)，再按疑点追源码；不默认逐项读取全部contract/seal。实例先验信息检查已完成，不能继续写“尚未实施”；额外query拼接小样本未获支持，下一改动需另立机制对照。

| 真实源码 | 阅读职责 |
|---|---|
| `code/tools/run_fi1_query_transfer_eval.py`、`verify_fi1_query_transfer_eval.py`、`analyze_fi1_query_transfer_eval.py` | 只评selection16、全池/好父尾部/干预及失败诊断；宿主入口run_fi1_query_transfer_eval_host.sh |
| `code/tools/run_fi1_query_same_parent_pilot.py`、`verify_fi1_query_same_parent_pilot.py` | 同父query小样本、固定64轮、干预、全池与独立回放 |
| `code/tools/fi1_nt_region_identity_core.py`、`run_fi1_nt_region_identity.py` | 新成员区域图、P0/P1坐标映射及无GT关联/探针 |
| `code/tools/evaluate_fi1_nt_region_identity.py`、`supplement_fi1_nt_region_identity.py`、`analyze_fi1_nt_region_identity_boundaries.py` | 两阶段GT诊断、完整NT池、局部原始query与执行边界 |
| `code/tools/verify_fi1_nt_region_identity.py`、`verify_fi1_nt_region_identity_supplement.py` | 本项384探针和独立可行点集/图/原始读出复算 |
| `code/tools/export_fi1_mask3d_instance_features.py` | 冻结Mask3D逐点特征、原点映射与同输出接口 |
| `code/tools/fi1_mask3d_feature_ab_core.py`、`run_fi1_mask3d_feature_ab.py` | 最新A/B输入、父条件、监督、模型和执行阶段 |
| `code/tools/evaluate_fi1_mask3d_feature_ab.py`、`verify_fi1_mask3d_feature_ab.py` | 实际编辑点集、完整NT池、保护指标与核验 |
| `code/tools/fi1_decoupled_region_core.py`、`fi1_region_affiliation_core.py` | 前序关系/父资格与区域归属定义，追溯失败边界 |

前序特征A/B只增加mask_features_head；最新同父小样本在共同head上增加query软概率，未取得额外增益。B减少损害却没有预定池收益，不能据此否定全部实例先验；也不按旧留出调阈值。固定SL、原父、邻域、六帧和物理划分，保留GT-free目标条件及原点可增删输出要求，推理不用GT挑父。已暴露24场景只称内部开发对照。不要恢复旧总控，设计新任务前核对保留输入和源码新hash。

最新A/B资产入口`experiments/fi1_mask3d_feature_ab_internal_20261013_v1/`经兼容链接指向`/media/jia/软件1/fi1_mask3d_feature_ab_internal_20261013_v1`；外盘大资产未删除。逐文件真实路径与hash见[当前源码索引](../../docs/cleanup/deep_active_cleanup_20261014/current_code_index.csv)。本次正文快照见[清理后几何交接](../../docs/state_archive/20261014_post_cleanup_geometry_handoff/README.md)。
