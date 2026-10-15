# 有效运行与边界

2026-10-14，项目根目录。按prepare → acquire → predict → evaluate_effective → analyze执行，全部新输出独立绑定本目录。无几何重生成/训练/阈值校准。

CPU：`.venvs/yoloe_gpu/bin/python`，`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`。

GPU：`/home/jia/anaconda3/envs/openyolo3d/bin/python`，同线程限制加`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=.:models/Mask3D:research/geometry/code/tools`；宿主批准后冻结Alpha/SAM推理。`visual_runtime.json`记录设备与text向量核验。旧复用acquire日志total显示200，但实际合同/循环仅24场景，receipt明确78个mask；未读取额外场景。

源码：

- `research/geometry/code/tools/run_fi1_t_add_downstream.py prepare/acquire/predict`。
- **有效评估**：`research/geometry/code/tools/evaluate_fi1_t_add_downstream.py`。原evaluate被GT审计路径符号链接不一致拦截；只把allowlist正规化为canonical path，补充回执`evaluation_adapter_frozen.json`绑定代码和既有预测seal。没有重生成预测或改变类别/分数/阈值。
- `research/geometry/code/tools/analyze_fi1_t_add_downstream.py`：独立AP积分、matching槽归因及上下文点交集核验；有效日志`analyze_effective_v4.log`。调试失败日志保留，未以失败核验宣布完成。
- `research/geometry/code/tests/test_fi1_t_add_downstream_trace.py`：高分重复保留TP、低分重复FP、void ignore与所有官方阈值独立AP一致，1项通过。

实际NT基线成员/x26/B26/类别/分数/排序bitwise一致，扩展完整L vs padding误差全部0。1700行点交集上下文独立检查；所有10组AP数组（两视图×五组）独立积分最大误差0。

补充归因：所有正式阈值，两追加组相对NT均新增/丢失目标0/0，旧TP成员退出0。原语义成员退出来源：A-context N319/T114，A-append N766/T191；B-context N0/T40，B-append N90/T68。入选成员数量不是唯一实例数；CA实际折叠后新mask为A73/B19。新T成为TP可代替旧TP赢家，不自动新增检测目标。

本轮未改历史OUT、最小阅读入口、生产或GitHub。实验报告`report.md`及completion/verification回执为结果入口。
