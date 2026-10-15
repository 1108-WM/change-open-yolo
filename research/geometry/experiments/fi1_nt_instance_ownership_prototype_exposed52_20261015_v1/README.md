# N/T归属原型：结果入口

本轮已实现三态归属原型并完成52场景测试。减少污染和好副本损害，但完整NT新增价值归零，未通过开发验收；停止本固定原型，未开展独立场景或AP。

- [主报告](report.md)：规则、预定验收、全量结果及关键机会
- [当前有效失败定位](failure_diagnosis_v2/report.md)：解释关键机会遗漏、连接N冗余、假连接；区分21/27两类oracle块集合
- [全量指标](summary.json)、[停止决策](decision.json)、[独立核验](verification.json)
- [GT前合同](contract.json)、[预测封存](predictions_frozen.json)、[GT后评估封存](evaluation_frozen.json)

失败定位v1中的27块是“点级机会父中贡献的纯块”，不是“纯整块union有价值的21块”。v2修正这一汇总文字，所有封存与原文件保留，预测、规则、GT评估及判定均不变化。

代码：run_fi1_nt_instance_ownership.py（prepare/generate/evaluate/report），fi1_nt_instance_ownership_core.py，verify_fi1_nt_instance_ownership.py，test_fi1_nt_instance_ownership.py。GT前快照在code_snapshot。后置定位代码diagnose_fi1_nt_instance_ownership_failures.py及supplement_fi1_nt_instance_ownership.py不新增点集或规则。

环境：.venvs/yoloe_gpu/bin/python，OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4。已封存输出禁止覆盖；核验可重新运行verify，从头复现须另建输出目录。未训练、未修改生产或项目阅读入口、未提交GitHub。
