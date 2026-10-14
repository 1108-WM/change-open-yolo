# 工具与测试索引（2026-09-11）

本索引只做分类，不删除或移动文件。`tools/`和`tests/`中许多脚本是一次性实验、审计或失败方向；它们仍可能被历史报告和哈希引用。

## 数量

- `tools/*.py`：446 个；`tests/*.py`：248 个；合计：694 个。
- 当前 FI1 几何-语义分支相关入口（文件名以 `fi1_`、`run_fi1_`、`audit_fi1_`、`export_fi1_`、`repair_fi1_` 开头）：10 个。
- 其余历史/通用脚本：436 个。

## 当前应优先保留的入口

- 第一点评估器和原始 OpenYOLO3D 代码：按原有目录和历史结果继续保留。
- 当前 FI1 几何-语义分支：`tools/fi1_gain_attribution_core.py`、`tools/run_fi1_gain_attribution.py`、`tools/audit_fi1_gain_attribution.py`、`tools/fi1_fixed_member_scoring_core.py`、`tools/run_fi1_fixed_member_scoring.py`、`tools/audit_fi1_fixed_member_scoring.py`。
- 当前固定的证据导出、候选池和 B1 多语义入口：以 `新开对话阅读内容.md` 顶部和 `CURRENT_EXPERIMENT_STATUS.md` 顶部列出的文件为准。

## 当前明确不作为生产方案的工具

下面这类文件应保留在历史归档层，不能作为当前入口：
- 已明确终止或失败的 selector/gate/oracle/旧仲裁方向，名称匹配的工具约 66 个（仅作初步筛选，最终以对应报告为准）。
- `tools/fi1_fixed_member_scoring*.py`：首版评分消费端已完成审计但 AP 下降，保留用于复现失败结论，不作为生产 B2。
- `tools/fi1_semantic_pair_guard*.py`：语义硬门控已明确不接入主线，保留审计和反例。
- `tools/fi1_gain_attribution*.py`：收益归因与独立审计工具，属于研究审计，不是推理入口。

## 为什么暂时不删除

1. 失败实验的报告、摘要和模型文件会记录源文件哈希；删除会让结果无法复查。
2. `tests/`中的测试经常覆盖历史数据合同和防篡改规则，直接删除可能让全量回归失去保护。
3. 当前没有统一的依赖图；仅凭文件名无法判断某个 `build_*.py`是否仍被其他入口导入。

## 建议的清理方式

先建立三个层次，再分批处理：

1. `core/`：原始基线和当前确定保留的生产/评测入口。
2. `research/`：当前实验、审计和可复现诊断工具。
3. `archive/`：已终止方向的代码与对应测试。

在移动前先生成导入依赖图，并检查报告、shell脚本、测试和 Python import；移动后保留兼容包装器或更新引用。删除只针对确认无引用、无报告哈希绑定、且用户明确同意的文件。

## 本轮结论

当前最合理的动作不是继续增加 Python 文件，也不是立即删除，而是先完成“索引 + 依赖审计 + 分批归档”。下一步应把当前第二点的最终入口收敛到一个 runner、一个 core、一个 audit 和一个 test，其余失败版本转为 archive。
