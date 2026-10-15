# 复现顺序

使用项目根目录和 `.venvs/yoloe_gpu/bin/python`；CPU，环境变量 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`。

```bash
.venvs/yoloe_gpu/bin/python -m pytest -q research/geometry/code/tests/test_fi1_local_block_attribution_selection.py
.venvs/yoloe_gpu/bin/python research/geometry/code/tools/run_fi1_local_block_attribution_selection.py prepare
.venvs/yoloe_gpu/bin/python research/geometry/code/tools/run_fi1_local_block_attribution_selection.py generate
.venvs/yoloe_gpu/bin/python research/geometry/code/tools/analyze_fi1_local_block_attribution_selection.py evaluate
.venvs/yoloe_gpu/bin/python research/geometry/code/tools/analyze_fi1_local_block_attribution_selection.py analyze
.venvs/yoloe_gpu/bin/python research/geometry/code/tools/verify_fi1_local_block_attribution_selection.py
.venvs/yoloe_gpu/bin/python research/geometry/code/tools/report_fi1_local_block_attribution_selection.py
```

上列为实际执行顺序；prepare/generate/evaluate/analyze均不允许覆盖本轮封存，已有结果不要重跑这些写入步骤。只复核可执行verify；从头复现须另设输出目录，保留合同与代码快照。prepare冻结构造代码，generate进程禁止GT/旧evaluation读取，完成predictions_frozen后才运行evaluate。报告脚本只汇总已诊断结果，不增加新选择或点集。

文件：contract/code_snapshot → predictions/{scene}/blocks,frames,unions,data及predictions_frozen → evaluation与evaluation_frozen → analysis与analysis_frozen → verification → posthoc_reporting/report/decision/completion_verified。

GT后所有oracle都仅用于诊断；没有正式可接收输出或下游AP。
