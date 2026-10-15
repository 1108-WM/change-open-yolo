# 2026-10-15：最近几轮实验与当前研究状态

这是GitHub/GPT页面的最新阅读入口。**最新完成N/T实例关系三态归属原型：稳定父新增精度95.92%，好T副本损失>.02为0/12，但完整NT增量≥.02的目标为0。因此停止本轮固定原型，未进入独立场景或AP。** 不能把它写成安全修复已成立，也不能据此否定全部实例关系或学习方法。

生产Full与有效SL架构未因这些实验替换。Full仍是GD10＋冻结D-v3七模型＋匹配597的B26＋N/T/U/U′＋K800；几何研究中的冻结SL下游对照另行报告。模型训练、几何池价值和最终AP必须分开读。

## 最新有效阅读顺序

1. [N/T归属原型最终报告](../research/geometry/experiments/fi1_nt_instance_ownership_prototype_exposed52_20261015_v1/report_final.md)：真实accept/reject/unknown、全部207父/1308块、完整NT与实际union。
2. [原型代码审计与下一步建议](../research/geometry/experiments/fi1_nt_instance_ownership_audit_20261015_v1/report.md)：N/T来源资格不是缺块的双源独立支持；187接收块中147只有N多数支持，109块组内唯一合格T是待修父自身。分组实际上是场景级，local_nodes仅记录不裁剪归属计算；缺参照也不等于缺全部身份信息。
3. [前轮块归属/点选择诊断有效v2](../research/geometry/experiments/fi1_local_block_attribution_selection_diagnostic_exposed52_20261015_v1/report_v2.md)：没有实现归属选择器，裁剪不能代替归属；23个混合块丢掉49.09%目标补点，不能把裁剪当作已解决。
4. 需要追溯实际系统价值时读[冻结五组下游AP](../research/geometry/experiments/fi1_t_add_union_downstream_internal24_20261014_v1/report.md)及[AP/query关系审计](../research/geometry/experiments/fi1_t_add_ap_query_relation_audit_20261015_v1/report.md)。

## 最近几轮执行与结论

| 轮次 | 实际执行 | 主要结果与边界 |
|---|---|---|
| [T补点24场景固定A/B](../research/geometry/experiments/fi1_t_add_union_internal24_20261014_v1/report.md) | 原N/T保留，只新增95采样T父的补点union，GT前固定规则与点集 | A/B完整NT连续及≥.02增量均0；父改善被已有N覆盖。停止两条规则微调；24场景此前已开发暴露，不称盲测 |
| [冻结下游AP](../research/geometry/experiments/fi1_t_add_union_downstream_internal24_20261014_v1/report.md) | 复用同24场景与封存A/B；新增T计算自身YOLO/Alpha语义；固定SL/K800，NT、A/B追加及两种context-only对照 | 语义AP：NT46.982025%，A追加45.936342%，B追加46.868626%；CA AP：68.825446%、67.610150%、68.509605%。所有官方阈值无新检出/丢失目标；重复与排序/context仍能降低AP |
| [query关系诊断](../research/geometry/experiments/fi1_query_relation_diagnostic_exposed52_20261014_v1/report.md) | 52已暴露场景/207父，先封存完整关系证据，再GT标注与供给诊断 | GT后最佳候选单query6目标、原锚+一个query6目标且集合相同、纯整块5、同域目标点8（7场景）。均为oracle，未实现可部署分组；没有测试任意多query或完整实例树 |
| [局部块与点选择诊断](../research/geometry/experiments/fi1_local_block_attribution_selection_diagnostic_exposed52_20261015_v1/report_v2.md) | 1308块全部W/C，未设置归属选择器；原父保留，query裁剪仅新增部分 | C保留95.95%目标、去掉40.06%邻物，但9/12好副本下降>.02。实际union有2目标/2场景池价值；不能GT挑块。真纯块机会锚重叠中位数.1978，普通纯正确块1.0 |
| [N/T关系三态归属原型](../research/geometry/experiments/fi1_nt_instance_ownership_prototype_exposed52_20261015_v1/report_final.md) | GT前固定complete-link重复组、跨源参照资格、唯一同/异组接受/拒绝、竞争或缺失unknown；query仅同样裁剪 | 接受187、拒绝95、unknown1026；66父动作。稳定新增精度51.67%→95.92%，好副本损害9→0；池≥.02收益2→0。55稳定动作52不如连接N；32机会贡献块21因缺合格参照unknown，11接收仍无完整NT新增价值 |

A/B补点规则的AP测试已真实完成；后两轮块归属研究没有AP，不得将“未AP”扩大成整个会话未AP。52场景是fit12＋exposed16＋internal24，均已暴露；不是全项目盲测或新规则独立验收。结论限207采样T父与原可编辑域，未知点既不当正确也不直接当邻物。

## 当前约束与下一步

停止当前修复器扩训、旧A/B规则及本轮N/T多数覆盖参照的筛选微调。保留query信息与原N/T；未训练新块归属模型、未搭完整实例树、未增加帧、未替换生产。

下一优先建议是**审计T的原始观测—关联—合并—细化来源**，不是已启动实验。按真实locator逐父核对版本、点坐标映射、observation/frame/source_track ID与哈希；先固定全部207父/1308块及负例的来源证据，再GT标注“无观测／关联碎裂／细化丢失／错误连接”，不可由删除总数猜具体块原因。若无法恢复来源链或证据仍不区分有价值碎片与邻物，记录限制并结束该支。细则见最新审计。

只有得到多场景安全完整NT新增价值且通过全部负例风险检查的明确无GT规则，才冻结未参与设计的物理隔离场景；独立验证通过后才接自身语义与冻结下游AP。少动作、平均IoU变好、高AUC、GT oracle均不单独通过。

## 勘误与审计边界

原封存文件逐字保留，不覆盖失败或旧报告。优先读上述最终有效版本：

- 局部块报告v1将scene0688单块C增量+.042515误写成不足.02；v2更正。该块含47/147未知，实际同父全块union只有+.011995；预测/GT数值未变。[勘误](../research/geometry/experiments/fi1_local_block_attribution_selection_diagnostic_exposed52_20261015_v1/report_errata.json)。
- N/T失败定位早版混称“点级机会中的27纯块”与“纯整块有价值union的21贡献块”，又将scene0078接受点数342写成341。最终报告已更正，原数据始终正确。[集合口径](../research/geometry/experiments/fi1_nt_instance_ownership_prototype_exposed52_20261015_v1/failure_diagnosis_v2/summary.json)、[点数勘误](../research/geometry/experiments/fi1_nt_instance_ownership_prototype_exposed52_20261015_v1/final_report_errata.json)。
- 审计指出整体未知比例≤.1这项机器门槛没有在原contract逐项明确列出，应视作额外后置安全检查，不能称所有门槛都逐项预注册；不改变预先明确的多场景池价值失败。
- 早期“共同帧≥2且无冲突”构造在用户约束后、GT/AP前退役；[退役回执](../research/geometry/experiments/fi1_rejectable_local_block_exposed52_20261015_v1/superseded_before_GT.json)、[下游未执行说明](../research/geometry/experiments/fi1_rejectable_local_block_downstream_exposed52_20261015_v1/not_executed.json)。不是被验证有效的方法。

## GitHub内容与复现边界

本次更新代码、测试、运行协议、报告、审计、紧凑逐父/逐块指标、合同、封存及勘误。原始/逐点数据、NPZ/NPY概率与点集、权重、巨大query/匹配明细不上传；保留原哈希和排除清单。此前“未GitHub”是执行当时的历史状态，本次已按用户明确指令发布，不改写历史回执字段。

[逐文件本机来源及SHA256](publication/geometry_session_20261015/local_source_manifest.json)、[未上传资产及SHA256](publication/geometry_session_20261015/omitted_assets.json)、[本次发布验证](publication/geometry_session_20261015/validation.json)。源码与文本记录能供GPT页面了解最新架构和结论；无法只凭GitHub运行依赖外部资产的完整实验。历史seal内绝对路径仅作溯源，发布快照核验与原实验全资产核验分开。

前序模型/局部编辑研究见[10月14日汇总](SESSION_EXPERIMENTS_20261014.md)。
