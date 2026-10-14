# 当前实验状态

**2026-10-11原问题状态复核：** S／SL开发路径已修正native类别资格；C双向成员风险选择尚未解决，当前SL不调用C；U/U′质量与B26准入接口尚未接通，当前SL限定NT。生产Full未替换。本轮实例归属实验独立复算通过，但同实例分类提升没有转化成可靠补全。下一几何建议为锚点引导的新观测与局部污染重分割，尚未实施；详见[审计报告](.local/fi1_affiliation_independent_review_20261011_v1/report.md)。语义与选择仍可改进，当前继续固定SL、几何优先。

> **当前优先级：先做几何提升，固定SL作为评分基线。语义与选择仍有提升空间；SL也不是已经达到上限。语义证据、类别成员选择、评分与重复处理属于保留待办，本阶段暂缓；不能把冻结SL写成“语义问题已解决”或自动恢复旧语义优先计划。**

**实例归属可学习性实验已完成，几何动作门未通过。** `.local/fi1_fragment_affiliation_learnability_development200_20261011_v1/`，入口report.md、supplemental_report.md、independent_verified.json、decision.json。固定检索和SL，两组同容量、200迭代HGB原五折；17506可靠训练／评测配对、21899unknown不作负例。增强组区域视觉包含冻结DINO-S14、最多三姿态组对应、点RGB／法向／局部形状和实测深度路径／遮挡。pooled配对AP0.511248→0.530779、ROC-AUC0.855485→0.874176，4折配对AP升；.5增强组3295接受，可靠precision60.54%，19个>.75新增而污染与未知很多；.9接受62（24同／3异／35未知），可靠88.89%，新增0／0／0。归属信号未转为可靠补全，本轮无新输出／自身语义／ScanNetAP、SL／生产／开发55不改，无训练恢复作业。

**前序硬归属几何诊断已完成，未进入自身语义／AP阶段。** 隔离目录`.local/fi1_complementary_fragment_retrieval_identity_development200_20261011_v1/`，入口report.md、supplemental_report.md、independent_verified.json、decision.json。检索阶段验证无需点集重叠的邻接可找回50／60机会、25／26不交配对；归属阶段仍失败：60个机会53个无共同掩码组、5个仅一组、2个无检索证据，没有满足两组共同支持者。最严格规则生成25个新点集，>.50／>.75／>.90真正新增NT覆盖0／0／2；接受27对中同实例10、异实例13、混合4，可靠部分precision43.48%。失败按预注册门停止，不获取自身语义或新AP、不挑弱规则、不扫阈值。SL、单模型、开发55和生产保持封存；后续跨掩码／跨视图实例关联尚未实施，无恢复作业。

**前一轮首版局部补全已完成，未通过接入验收**。SL评分冻结，877个NT>.75缺口已诊断，201个新点集生成并获取自身YOLO／Alpha；完整开发200原五折三臂真实双AP已完成。原SL＝38.727589%／68.289187%；扩池仅输出NT＝38.763062%／68.237136%；扩池追加＝38.687406%／67.699156%，相对SL−0.040184／−0.590030pp。>.75追加语义新增1／损失1、去类新增1／损失0；>.90为语义2／1、去类2／0，424全保留。12相关测试、4689目标、201动作、320000成员推理、4000上下文及48AP核验通过。入口`.local/fi1_witnessed_completion_SL_development200_20261010_v1/report.md`／independent_verified.json，几何原始诊断与失败原因见同轮前继目录。用户主线仍为几何，SL／单模型／开发55／生产均未改；不自动扫门槛或增加第二动作，无恢复作业。

**SL开发200剩余错误／U增量机会诊断已完成**：`.local/fi1_sl_remaining_errors_union_opportunities_development200_20261010_v3/`，先读report.md、independent_verified.json、supplement.json、decision.json。原五折留出4689目标，>.75为缺NT几何877／缺正确资格0／正确成员未进S800237／覆盖3575；覆盖精确复现原SL。全U新增NT覆盖25，U′为0；隔离扩展池>.75语义新增／损失21／34，筛后8／5，原424失12／1，不能直接追加。未训练、未读原始GT、未新增AP或修改生产；SL单模型与开发55封存。当时建议的NT真实语义证据支持的类别成员选择保留为暂缓待办；用户现明确优先几何，未启动该语义训练。v1缓存范围、v2导出失败均保留；v3复用全部封存探针完成独立核验，无恢复作业。

**单模型训练200 → 完整开发55的S／SL验证已完成并通过核验**：`.local/fi1_sl_single200_dev55_20261010_v1/`，入口report.md／identity_audit_report.md／independent_verified.json／supplemental_report.md。固定原种子20261010、20轮训练一个L，55扫描／29物理场景与训练200、B26597及D-v3官方100物理分离。S42.594175%／63.611227%，SL45.894361%／66.557066%，类别＋3.300186pp、实际去类＋2.945839pp，AP50/AP25均提高，全部10档S覆盖／正分保护通过。完整池真实推理重放、44000成员、110双向匹配、1100覆盖和4AP聚合核验通过，18主测试＋2后置守卫测试通过。开发55历史暴露且位于官方训练域，只称固定下游单模型跨场景验证，不能称全系统盲测。后置审计守卫两次访问失败日志保留，v2只允许合同身份元数据及封存训练标签，科学输出不改。单模型封存可用，生产未替换，不加U/U′，无恢复作业。

**SL真实推理及三种子稳定性已完成并核验通过**：`.local/fi1_sl_replay_stability_development200_20261010_v1/`。真实完整S池／X26／B26→S800→L入口逐位复现原200场景160000成员及16个AP数组；两套追加五折20轮已完成。三种子类别AP均值38.570356%（38.476618—38.727589），实际去类68.044958%（67.889773—68.289187）；三套均超过S/C/R，旧90／新110双升，424及全部10档S覆盖／正分保护均通过。15模型、训练配对重建／训练归一化、480000成员和48个AP聚合核验通过，14相关测试通过。入口report.md／training_stability_report.md／independent_verified.json；真实CLI见inference_usage.md。该稳定性阶段没有训练最终模型；后续200单模型／开发55已完成，见上节新目录。真正未暴露独立身份仍未冻结；生产未替换。

更新：2026-10-11。当前进展在此维护；详细分析见[修改方向](资料/当前基线修改方向.md)，指标见[主要结果](资料/结果.md)，历史见[实验索引](资料/实验历史索引.md)。

## 当前身份与未完成事项

- **固定S800／封存L评分对照SL已完成并通过开发门**：`.local/fi1_s_fixed_L_rescore_development200_20261010_v1/`，入口 `report.md`、`supplemental_report.md`、`independent_verified.json`、`development_candidate.json`。类别／实际去类AP38.727589%／68.289187%，相对S＋1.754860／＋3.264194pp，均超过S/C/R。424全保留、全部原IoU档S语义／去类覆盖及正分覆盖精确一致。相对L类别−0.006499pp、去类＋0.005100pp；>.75恢复L失21个语义目标，同时放弃L新增36个。可登记开发候选待稳定性／独立验证，生产不自动替换、不立即训练预算模块或追加U/U′。10测试与160000成员／400匹配／16AP／2000覆盖核验通过，无原始GT、训练或新模型推理；无恢复作业。

- **S基准成员级语义排序残差L已完成：双AP提高，严格覆盖保护未通过**。`.local/fi1_s_semantic_rank_residual_development200_20261010_v2/`，入口 `report.md`、`supplemental_report.md`、`independent_verified.json`。L38.734088%／68.284087%，相对S＋1.761359／＋3.259094pp，同时超过S/C/R；训练及五折留出排序门通过。保住410／424，14个正确类仍正分、排名819—1392出K800，但好几何均留存；>.75对S语义新增36／损失21，去类0／0。重复配对相对S代表顺序逆序增加；代表由最高S定义，不能单独证明实际去重变差，仍未证明学会去重。正式保护门失败，保留有效开发信号、不替换生产、不继续拼接S/R或给Q/F加模块、不追加U/U′；没有本项恢复任务或下一训练。

- **S/R 分数分解 P0／P1／P2 已完成，三臂未通过**：`.local/fi1_sr_score_factorization_development200_20261010_v1/`，入口 `report.md`、`supplemental_report.md`、`independent_verified.json`。P0 34.858288%／66.237359%，P1 34.858823%／66.237473%，P2 34.826907%／66.171175%。P0去类八个数组精确复现SR，但语义比S低2.114441pp；P0/P1保住424及正分，P2失10个，并对S在>.75新增／损失语义8／34、去类0／7。固定集合评分本身已经失败，不把全部问题归因预算；不训练模仿R最高分的倍率头，不扫权重／倍率／预算、不加U/U′或改生产。13测试、6000逐档覆盖、48新AP聚合及10个C折对照完成；主评测在新预测封存后仅读原开发200白名单GT，无训练／模型推理。没有本项恢复作业。

- **S/R 固定集合×评分四格对照已完成**：`.local/fi1_sr_cross_control_development200_20261010_v1/`，入口 `report.md`、`interpretation.md`、`decision.md`、`independent_verified.json`。SS36.972729%／65.024993%，SR36.525445%／66.237359%，RS37.035601%／65.064787%，RR36.532182%／66.236562%。R去类收益在S集合几乎全部保留，类别评分仍退化；四格定位评分干预，但不能单独归因同几何类别次序；后续分解已执行未通过。424子群SR无成员退出但6个无正分，RS/RR失13个。64数组、640000成员和8000覆盖核验通过，无训练、模型推理、原始GT或U/U′；没有本项恢复作业，倍率模型尚未训练。

- **生产**：历史official100七模型D-v3＋GD10匹配597 B26，Full（N/T/U/U′），K800；没有替换。
- **开发双参考**：C保留冻结NT关系选择效果，S保留对称类别资格及正确类覆盖收益。C去类更好、S类别相关更好；尚无S＋C预测／AP，两者收益不能相加，均非独立验收或生产方案。
- **S：native 对称类别资格固定对照已完成**：类别相关／实际输出去类主AP为36.972729%／65.024993%；均超过NT和固定NMS，但只有类别相关超过C，两口径替换门未通过。旧90／新110方向一致。保留C开发参考与生产。
- 第二轮独立审查已核对。S只让native也按原类别＋自身融合正分数展开，重算X26后用冻结B26；全部预测先封存，原NT全池／特征／分数精确重放，新输出独立核验、24个AP数组复算及补充只读审查通过；没有本项恢复任务。
- **S/C排序与错误成员可处理范围只读诊断已完成**：`.local/fi1_s_c_selection_scope_development200_20261010_v1/`。已查代表、共享最高分、近重复、全局排序压力、跨源同类资格及424恢复目标风险；没有新GT、训练、模型预测、集合选择、匹配或AP。
- **S 协议 R/Q/F 成员评分与竞争实验已完成，未通过**：`.local/fi1_s_member_quality_competition_development200_20261010_v1/`，入口 `analysis.md`、`failure_interpretation.md`、`final_verified.json`。R同容量原监督重训36.532182%／66.236562%；Q自身质量锚定36.252760%／60.140582%；F加类别竞争36.252764%／60.140582%。固定 N/T、K800，复用 C 五折，每折160训练；R去类改善但类别低于S/C，Q/F失去12个424恢复目标，8次软竞争仅1个可靠安全。第一步未通过，按冻结失败分支不追加U/U′，无生产替换或AP后调参；没有本项待恢复作业。

## 最新结论与验收边界

SL固定名单、三种子稳定性及200单模型→开发55验证均已通过；后续仍需真正未暴露测试身份，不自动训练预算模块。旧90／新110双升；fold4语义仍比S低0.045150pp，开发通过不等于所有类别／分组无损或独立盲验收。重复配对只衡量S代表顺序变化，不能单独推断实际重复恶化；细节见[修改方向](资料/当前基线修改方向.md)最新节。

新审查目录`.local/fi1_risk_objective_code_review_20261009/`，仅读开发200的冻结标签、匹配和自身语义。`verified.json`核对2555个已读输入哈希，其中2354个直接匹配原回执，另行复算标签和原预算退出数；没有新训练、预测、点集生成、匹配、AP或原始GT读取。脚本输出有不覆盖断言，不直接重复执行已有目录。

前一步删除监督仍有效：17120有向动作分为allow1704／harm8318／abstain7098，1416个允许动作双方有正确类资格；C的21个失覆盖目标和P2两个错误均被标为伤害。这只证明几何标签定义与样本检查成立，不能证明成员决策正确、特征可学或AP收益。

## 当前研究的证据链

下表目录均相对项目根。已完成项不恢复旧shell；优先读分析／报告和验收JSON，不默认读取所有逐场景文件。

|阶段|结论／状态|完整目录|
|---|---|---|
|固定S800、L置信SL|完成并通过开发双AP／覆盖门，登记候选未部署；无训练／原始GT|`.local/fi1_s_fixed_L_rescore_development200_20261010_v1/`：`contract.json`、`report.md`、`supplemental_report.md`、`decision.json`、`independent_verified.json`、`development_candidate.json`、`handoff_verified.json`|
|S语义排序残差L|完成；训练／留出排序和双AP提高，424及语义覆盖保护失败；未部署|`.local/fi1_s_semantic_rank_residual_development200_20261010_v2/`：`contract.json`、`learnability.json`、`report.md`、`supplemental_report.md`、`independent_verified.json`、`handoff_verified.json`|
|S/R几何最高分×S类别比例|完成未通过；P0固定S集合语义已退化，P2另有预算覆盖损失|`.local/fi1_sr_score_factorization_development200_20261010_v1/`：`report.md`、`supplemental_report.md`、`decision.json`、`independent_verified.json`|
|S/R固定集合×评分|完成；R去类收益来自评分仍能在S集合保留，语义评分退化独立于成员退出|`.local/fi1_sr_cross_control_development200_20261010_v1/`：`report.md`、`interpretation.md`、`decision.md`、`evaluation_summary.json`、`effects.json`、`diagnostics.json`、`class_AP_changes.json`、`completion_verified.json`、`independent_verified.json`|
|R/Q/F独立审计|完成；无推翻结果的程序错误，监督口径／场景权重／融合／动作设计问题明确|`.local/fi1_rqf_independent_code_audit_20261010_v1/`：`report.md`、`audit.json`、`supplement.json`、`verified.json`|
|S成员质量与竞争 R/Q/F|完成未通过；原监督重训去类获益，独立质量组合排序退化，竞争实际未改代表|`.local/fi1_s_member_quality_competition_development200_20261010_v1/`：`analysis.md`、`failure_interpretation.md`、`contract.json`、`evaluation_summary.json`、`independent_audit.json`、`failure_analysis.json`、`final_decision.json`、`final_verified.json`|
|S/C排序与动作范围|只读诊断完成；类别置信／排序与跨源重复需分开，不直接叠加S+C|`.local/fi1_s_c_selection_scope_development200_20261010_v1/`：`report.md`、`summary.json`、`score_pressure.json`、`target_ranking.json`、`verified.json`、`report_verified.json`、`completion_verified.json`；逐成员／424目标见`scenes/*/members.json`|
|S native对称资格|完成；两口径均超过NT/NMS，去类仍低于C；未部署|`.local/fi1_native_symmetric_eligibility_development200_20261009_v1/`：`analysis.md`、`contract.json`、`evaluation_summary.json`、`diagnostics/summary.json`、`completion_verified.json`、`supplemental_review.json`、`supplemental_verified.json`|
|第二轮独立审查|完成；3005输入／2714旧回执、方向偏置与Stage-D/B26接口|`.local/project_independent_review_20261009_v1/`：`findings.json`、`supplement.json`、`verified.json`|
|最新代码与输出审查|完成；资格及决策层级问题，未训练|`.local/fi1_risk_objective_code_review_20261009/`：`audit.json`、`member_audit.json`、`eligibility_evidence.json`、`verified.json`|
|删除动作监督|完成；几何标签语义检查通过|`.local/fi1_replacement_risk_supervision_development200_20261009_v1/`：`analysis.md`、`contract.json`、`labels_frozen.json`、`audit/`、`completion_verified.json`|
|P0/P1/P2父T替代|完成；P2略优于追加，仍低于NT/C；家族不等于实例|`.local/fi1_uprime_parent_selection_development200_20261009_v2/`：`analysis.md`、`loss_member_chains.json`、`completion_verified.json`|
|E直接预测一次Δ|完成未通过；原21损失未恢复，新增1个>.90损失|`.local/fi1_nt_direct_quality_development200_20261009_v1/`：`analysis.md`、`mechanism_decision.json`、`completion_verified.json`|
|D去质量符号门控|完成；边际影响小，不能证明质量头可靠|`.local/fi1_nt_quality_ablation_development200_20261009_v2/`：`analysis.md`、`diagnostics/`、`completion_verified.json`|
|质量表示代码审查|完成；已指导E，不再按旧计划重做|`.local/fi1_nt_quality_mechanism_review_20261009_v1/`：`report.md`|
|C关系选择|完成；开发参考，全部实际抑制是N替代T|`.local/fi1_nt_relation_selection_development200_20261009_v2/`：`posthoc_report.md`、`predictions/`、`evaluation/`|
|固定NT NMS|完成；标准参考，小幅正收益且非无损|`.local/fi1_nt_selection_control_development200_20261009_v1/`：`report.md`、`evaluation/`|

数据范围、宿主恢复和不覆盖约定统一见[阅读入口](新开对话阅读内容.md)。旧验收回执中状态文件哈希指向当时快照，不因本次文档精简失效，也不修改那些回执。
