# 仓库阅读与协作

本仓库默认分支main是2026-10-14同步的当前源码状态。先读README.md；当前几何研究按“新开对话阅读内容.md”的最小路线读取。Full与SL架构见“资料/代码文件索引与当前主线地图.md”。

真实源码在full_bundle/code、sl_bundle/code、research/geometry/code；根tools/tests/utils多数为相对兼容链接。full_bundle/runtime是独立冻结实现，不与当前helper混为同一源码身份。报告中的本机路径及哈希是历史溯源，不代表远端存在对应数据。

本仓库不发布权重、数据或缓存。缺失资产见docs/EXTERNAL_ASSETS.md；不要因源码快照无法运行完整资产核验而声称实验失败，也不要把已有测试结果说成本次重跑。推理不读取GT，预测先封存再评测；保留已暴露开发划分及失败记录。阅读与分析不自动启动训练、批量推理或旧finalize总控。
