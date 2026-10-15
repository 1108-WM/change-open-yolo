# 本轮阅读入口

当前有效报告：[report_v2.md](report_v2.md)。52个已暴露场景、207个采样T父、1,308个补点块；仅诊断，无训练或AP，停止本轮固定构造。

报告v1把scene0688单块C的+0.042515误写为不足0.02。v2更正：单块超过0.02，但含47/147未知点；实际全部块union只有+0.011995。所有预测、GT统计、代码封存、判定均未变化；v1与原完成回执保留，见[report_errata.json](report_errata.json)。不要引用v1这句话。

- [当前完成回执](completion_verified_v2.json)及[独立核验](verification.json)
- [决策](decision.json)、[原始合同](contract.json)、[执行顺序](RUN.md)
- [分层点数和指标分母解释](posthoc_reporting/summary.json)
- [有价值块及相反标签证据](analysis/valuable_block_evidence.json)
- [全量GT评估摘要](evaluation_summary.json)

逐场景无GT证据和固定W/C联合点集在predictions/，后置GT诊断在evaluation/。所有oracle仅解释机会，不是可部署方法。生成代码与测试的封存副本在code_snapshot/；报告和勘误是之后的汇总步骤，不新增任何选择或点集。勘误命令为：

```bash
.venvs/yoloe_gpu/bin/python research/geometry/code/tools/correct_fi1_local_block_attribution_report.py
```

该写入命令已完成，勿覆盖重跑；只核验请读取完成回执和运行独立verify。
