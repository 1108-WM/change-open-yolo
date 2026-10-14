# 2026-10-14会话实验与最新状态

**最新追加：局部编辑证据诊断已完成。** [报告](../research/geometry/experiments/fi1_local_edit_evidence_diagnostic_20261014_v1/report.md)／[判定](../research/geometry/experiments/fi1_local_edit_evidence_diagnostic_20261014_v1/decision.json)／[核验](../research/geometry/experiments/fi1_local_edit_evidence_diagnostic_20261014_v1/verification_counts.json)。28场景224父、15,434提案，先无GT证据封存再GT；不训练/改正式N+T。T整区补点有较强区分与少量池价值，N补竞争和删除未形成统一安全方案。6测试、全量点集/41,238池记录复算通过。fit关键补块有67/639未知，exposed16两目标有严格证据但已暴露；独立规则验收未开展。

直接证据：[GT-free逐场景证据](../research/geometry/experiments/fi1_local_edit_evidence_diagnostic_20261014_v1/proposals)、[N/T补删证据比较](../research/geometry/experiments/fi1_local_edit_evidence_diagnostic_20261014_v1/analysis/feature_comparison.csv)、[描述bin负结果](../research/geometry/experiments/fi1_local_edit_evidence_diagnostic_20261014_v1/analysis/evidence_bins.csv)、[有效后置补充v2](../research/geometry/experiments/fi1_local_edit_evidence_diagnostic_20261014_v1/supplement_v2)、[池证据追踪](../research/geometry/experiments/fi1_local_edit_evidence_diagnostic_20261014_v1/supplement_v2/pool_witnesses.csv)、[完整分母池统计](../research/geometry/experiments/fi1_local_edit_evidence_diagnostic_20261014_v1/supplement_v2/complete_pool_summary.csv)。v1补充与失败日志作为历史保留，原合同/提案/评测seal未改；最新源码见[研究目录](../research/geometry/README.md)。

以下三轮模型实验与审计保留为前序状态：

给GitHub/GPT页面的阅读入口。**最新判断：暂停现有A/B修复器扩训及部署。** 同父训练任务能拟合，但固定未训练场景评估出现高置信错误编辑和明显好父损害，尚无安全跨场景完整NT新增价值。没有新AP、生产替换、混合父拆分或实例树实验。

## 按这个顺序阅读

1. [现有模型固定跨场景评估报告](../research/geometry/experiments/fi1_query_transfer_eval_20261014_v1/report.md)：16个selection场景、128原N/T父，模型/归一化/.8阈值锁定，不再训练。A/B稳定父平均IoU增量−.016189/−.014050，好父损失>.02为10/41、9/41。A仅1个池增量≥.02目标，B和B去query均0。
2. [最新独立审计](../research/geometry/experiments/fi1_query_transfer_audit_20261014_v1/report.md)：复核主要结果可信；B误补3518点中2423点概率≥.95，误删802点中209点概率≥.95。仅添加query同意门可缓解错误但仍无池新增；这是后置敏感性诊断，不是冻结评估输出或新的独立验证。
3. [当前状态](../CURRENT_EXPERIMENT_STATUS.md)、[当前修改方向](../资料/当前基线修改方向.md)与[源码目录](../research/geometry/README.md)。后续方向仍是建议，未自动启动区域编辑、扩训、网络更换或阈值搜索。

## 本次保存的三轮实验与审计

|实验|作用与结果|直接证据|
|---|---|---|
|N/T区域身份诊断阶段0—4|fit96抽12场景/96父；冻结P0/P1、成员区域、无GTquery关联。先验有局部身份信息，但直接读出伤害好父；P1区域/原点池价值4/1不是安全修复|[报告](../research/geometry/experiments/fi1_nt_region_identity_diagnostic_20261014_v1/report.md)、[独立审计](../research/geometry/experiments/fi1_nt_region_identity_audit_20261014_v1/report.md)|
|同一父目标query小样本A/B|16父/64轮；两臂同目标/监督/损失/初始化/预算，B仅增加query概率；训练内无有效增量，原判定停止扩量|[报告](../research/geometry/experiments/fi1_query_same_parent_pilot_20261014_v1/report.md)、[合同](../research/geometry/experiments/fi1_query_same_parent_pilot_20261014_v1/contract.json)、[判定](../research/geometry/experiments/fi1_query_same_parent_pilot_20261014_v1/decision.json)|
|训练饱和独立审计|发现A接近固定编辑域上限，即使宽松完美预测平均也只能领先A .007451，小于原.01继续门；不改旧失败，但不能据此否定query方向|[审计](../research/geometry/experiments/fi1_query_same_parent_pilot_audit_20261014_v1/report.md)、[结果](../research/geometry/experiments/fi1_query_same_parent_pilot_audit_20261014_v1/audit_results.json)|
|现有模型固定跨场景评估|selection16与本次训练物理隔离；原NT、A、B、B_zero、三次shuffle及区域平均，先封存再GT；训练拟合未转为安全修复|[合同](../research/geometry/experiments/fi1_query_transfer_eval_20261014_v1/contract.json)、[汇总](../research/geometry/experiments/fi1_query_transfer_eval_20261014_v1/evaluation/summary.json)、[判定](../research/geometry/experiments/fi1_query_transfer_eval_20261014_v1/decision.json)|
|跨场景独立审计|复核896点集/9480池记录/原点映射及机会；区域平均有3场景池证据但不能直接部署；当时建议局部编辑提案研究；后续已执行，见页首|[审计](../research/geometry/experiments/fi1_query_transfer_audit_20261014_v1/report.md)、[结果](../research/geometry/experiments/fi1_query_transfer_audit_20261014_v1/audit_results.json)|

## 最新评估的结果与实现

|读出|稳定父平均IoU增量|好父损失>.02|完整NT新增≥.02目标|
|---|---:|---:|---:|
|A|−.016189|10/41|1|
|B|−.014050|9/41|0|
|B去query|−.016787|9/41|0|
|已有区域平均|+.020091|0/41|3|

区域平均是固定.5诊断读出，网络动作阈值.8，不是等阈值/等校准比较。其3目标均来自稳定父，但大量身份不明编辑和510个目标误删仍没有安全结论。完整原NT保留，所以池max IoU不会下降；坏候选对未来排序/K800的风险没有评测。selection16曾被前序开发使用，只称内部开发评估，最终holdout未动。

- [训练/干预代码](../research/geometry/code/tools/run_fi1_query_same_parent_pilot.py)与[4项测试](../research/geometry/code/tests/test_fi1_query_same_parent_pilot.py)。
- [固定评估代码](../research/geometry/code/tools/run_fi1_query_transfer_eval.py)、[有效宿主入口](../research/geometry/code/tools/run_fi1_query_transfer_eval_host.sh)、[2项测试](../research/geometry/code/tests/test_fi1_query_transfer_eval.py)。
- [独立回放代码](../research/geometry/code/tools/verify_fi1_query_transfer_eval.py)与[核验记录](../research/geometry/experiments/fi1_query_transfer_eval_20261014_v1/verification.json)。
- [失败诊断代码](../research/geometry/code/tools/analyze_fi1_query_transfer_eval.py)、[诊断结果](../research/geometry/experiments/fi1_query_transfer_eval_20261014_v1/failure_diagnosis/summary.json)。
- [逐父CSV](../research/geometry/experiments/fi1_query_transfer_eval_20261014_v1/evaluation/parents.csv)、[逐场景CSV](../research/geometry/experiments/fi1_query_transfer_eval_20261014_v1/evaluation/scenes.csv)、[全池CSV](../research/geometry/experiments/fi1_query_transfer_eval_20261014_v1/evaluation/pool.csv)、[机会CSV](../research/geometry/experiments/fi1_query_transfer_eval_20261014_v1/evaluation/opportunities.csv)。

## GitHub发布边界与身份

本次提交包含源码、测试、报告、审计脚本、合同/封存/完成回执、JSON与CSV结果、训练历史、宿主日志和状态文档前后快照。原实验文本及历史seal保持本机字节身份；当前阅读入口只适配可用仓库链接。实验记录中的本机绝对路径用于溯源，不代表GitHub存在对应外盘。

按仓库AGENTS.md发布规则，权重、原始/处理数据、逐点输入/概率和特征NPZ不上传。它们在本机保留，见[未上传二进制身份清单](publication/geometry_session_20261014/excluded_binary_assets.json)；因此不能仅从GitHub运行原完整实验hash/模型回放核验。GitHub可独立阅读全部关键结果，不需要大资产。

[本机源文件hash清单](publication/geometry_session_20261014/local_source_manifest.json)对应复制前身份，[当前发布清单](publication/source_snapshot_manifest.json)对应GitHub实际内容。原整理时的代码索引和旧发布validation保留其时点；本次发布验证单独记录，不把历史GPU回放说成上传时重跑。
