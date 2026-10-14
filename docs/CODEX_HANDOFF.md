# Codex交接入口

更新：2026-10-14。本文件仅作入口，不复制状态。

请读[几何问题分析入口](../新开对话阅读内容.md)，按最小路线核查最新几何报告和相关源码。用户要寻找现有问题、实现漏洞与下一步修改方向；不要求全项目通读、逐轮历史审计或例行重哈希。

清理、前序Mask3D特征A/B、N/T区域诊断阶段0—4及[同父query小样本A/B](../research/geometry/experiments/fi1_query_same_parent_pilot_20261014_v1/report.md)已完成。小样本共同任务学得会，但B无训练内有效增益和query收益依赖，原预注册判定保留；后续固定跨场景评估已完成，见下。不要继续扩大同一拼接训练或恢复旧总控。若用户授权新研究，从冗余、融合和损失—动作机制入手另立最小对照，保留失败资产；混合父拆分仍另立任务。固定SL，不宣称AP提升。

最新用户授权的[现有A/B固定跨场景评估](../research/geometry/experiments/fi1_query_transfer_eval_20261014_v1/report.md)已完成，只评原selection16、不训练。跨场景好父伤害明显，B无额外NT池价值，A自身也不安全；暂停当前修复器扩训，不直接选A或继续B。原16训练父/64轮失败记录及[饱和审计](../research/geometry/experiments/fi1_query_same_parent_pilot_audit_20261014_v1/report.md)保留，不能以训练门限否定全部query方向。无需恢复训练或扩大评估；新研究需定位报告中的高置信错点/分布/损失机制。

详细状态在[当前状态](../CURRENT_EXPERIMENT_STATUS.md)顶部，源码路径在[几何目录](../research/geometry/README.md)。Full、清理、Replica、图例、SL历史及旧文档快照均按需查询，旧快照哈希不代表本次修订后的正文。
