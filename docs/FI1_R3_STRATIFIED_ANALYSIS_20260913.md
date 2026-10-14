# FI1 R3 分层分析（2026-09-13）

R3b 开发集 AP 为 38.624395，R2 为 32.250098，提升 6.374297 个百分点；诊断集提升 4.098553 个百分点，但诊断主场景与训练池重叠，不能视为独立泛化。

## 类别与来源

候选来源和类别计数见 `r3_v1/analysis/stratified_analysis.json`。来源标签直接读取冻结 R1 eligible 记录，未使用 R2/R3 学习分数。

## 折间收益

五折留出误差、正目标主场景数和竞争目标误差均已逐折保存；R3b 的竞争目标会将同类候选的一部分独立正 IoU 压为 0，这是设计行为。

## 重复风险

以冻结上下文中的同类 IoU>0.5 邻居作为候选重复代理。报告包括候选比例、平均和 P95 邻居数；它反映候选层重复压力，不等同官方评测匹配。

完整机器可读结果：[stratified_analysis.json](experiment_summaries/fi1_r0_r4_preparation_20260913/r3_v1/analysis/stratified_analysis.json)。
