# change-open-yolo：当前Full、SL与几何研究

本仓库是 OpenYOLO3D 的研究分支。**当前状态以默认分支 `main` 为准**；最近实验与阅读入口更新日期为2026-10-15。旧实验与发布分支保留历史，不代表当前主线。

本次发布包含本机当前Full、SL及几何研究源码、测试、配置、必要第三方实现和阅读材料。**不包含任何模型权重、数据集、处理后数据、逐点预测或特征缓存**。这是可阅读的源码快照；运行仍需独立准备指定版本的权重、数据和环境。

**最新几轮执行：[2026-10-15实验与当前状态](docs/SESSION_EXPERIMENTS_20261015.md)。** N/T三态归属原型过滤污染但没有完整NT新增价值；停止本固定原型，未进入独立验证/AP。前序冻结24场景A/B下游AP已真实完成且下降，不能混称本会话未AP。

## 阅读入口

| 目的 | 入口 |
|---|---|
| 了解总体架构与源码职责 | [代码地图](资料/代码文件索引与当前主线地图.md) |
| 了解当前实验状态 | [当前状态](CURRENT_EXPERIMENT_STATUS.md)与[结果](资料/结果.md) |
| 继续当前几何研究 | [新开对话阅读内容](新开对话阅读内容.md)，按其中最小路线阅读 |
| 理解生产Full | [Full目录](full_bundle/README.md) |
| 理解有效SL | [SL目录](sl_bundle/README.md) |
| 查看研究源码与关键报告 | [几何目录](research/geometry/README.md) |
| 理解上传边界、路径与外部依赖 | [发布说明](docs/GITHUB_CURRENT_SOURCE.md)与[外部资产](docs/EXTERNAL_ASSETS.md) |

## 2026-10-14前序：局部补删证据诊断

无训练，先无GT生成/封存15,434局部提案及证据，再后置GT诊断。**T整块补点query有区分及少量完整NT新增价值，统一N/T安全补删路线尚未成立。** fit关键池块含未知点，已暴露16场景不能作为新规则独立验收。6测试及全部15,434提案/41,238池记录复算通过；没有新规则、生产修改或AP。

以上是该轮历史结论；后续T补点、下游AP与归属原型已执行，按[10月15日汇总](docs/SESSION_EXPERIMENTS_20261015.md)读取最新结果。原[诊断报告](research/geometry/experiments/fi1_local_edit_evidence_diagnostic_20261014_v1/report.md)和[10月14日会话总览](docs/SESSION_EXPERIMENTS_20261014.md)保留溯源。

## 2026-10-14前序几何实验与独立审计

**现有A/B修复器在未参与本次训练的selection16场景伤害明显，暂停当前配方扩训与部署。** 本次仅固定评估：模型epoch64、原归一化和阈值.8未变，没有再训练。97稳定父平均IoU增量A −.016189、B −.014050；41好父中损失>.02为10/9，最坏约.458/.439。完整NT新增≥.02目标A仅1个/1场景，B及B去query均0。B减轻部分错误，但无安全新增池价值；A自身也不能直接沿用。

阅读GPT页面时先打开[本次会话实验总览](docs/SESSION_EXPERIMENTS_20261014.md)，再读[固定跨场景报告](research/geometry/experiments/fi1_query_transfer_eval_20261014_v1/report.md)和[最新独立审计](research/geometry/experiments/fi1_query_transfer_audit_20261014_v1/report.md)。三轮实验、三份独立审计、合同、结果CSV、训练历史、测试/核验记录及源码均已发布；权重、原始数据、逐点概率/特征二进制留在本机，以哈希清单溯源。

## 当前架构与状态

生产Full固定为 **GD10＋official100冻结D-v3七模型＋GD10匹配597训练B26＋N/T/U/U′＋K800**。K800是全场景最多800个点集—类别成员。Full流程包括原生候选、GD-SAM多视图轨迹、几何候选构造、自身YOLO/Alpha语义和B26排序。历史validation312类别AP/AP50/AP25为29.230610%/39.901908%/45.578362%，并非本次重新评测。

有效SL单独保存；S对N/T实施对称类别资格，L学习评分残差。当前几何研究固定SL作为评分基线，优先研究几何增删；语义和选择暂缓，不代表已经解决。

前序Mask3D head特征A/B、N/T区域身份诊断、同父query小样本训练及现有模型固定跨场景评估均已完成。query修复器泛化不安全，后续T补点冻结下游AP也未获收益；最新N/T归属原型停止，未替换生产Full或恢复训练。

## 源码组织

```text
full_bundle/
  cli/                 Full统一入口
  code/                当前Full/597/Replica源码、公共模块与测试
  runtime/             历史冻结Full运行实现（权重另备）
  vendor/              所需第三方源码及许可证
  code_sources/        保留的冻结自身语义helper
  models/              配置与模型身份说明，不含权重
  metadata/            场景身份及本机路由溯源
sl_bundle/
  cli/、code/          有效SL推理、训练、诊断与测试
  experiments/         精选报告与小型摘要
research/geometry/
  code/                当前局部几何实现、核验与测试
  experiments/         当前实验及独立审计的可读报告
  assets/_external/    SAM2源码，仅保留路线，未启动新推理
```

根目录 `tools/`、`tests/`、`utils/`、`models/` 及部分 `_external/`、`pretrained/` 为仓库内相对兼容链接，Git检出时会保留；真实源码在上表目录。所需Mask3D第三方源码已纳入，不需依赖原目录中的Git子模块数据库。原路径、合同和失败记录保留；本机路由元数据不是跨机器可用的数据索引。

## 下载与源码核验

```bash
git clone --depth 1 https://github.com/1108-WM/change-open-yolo.git
cd change-open-yolo
python3 scripts/verify_source_snapshot.py
```

核验仅用Python标准库，检查发布文件、仓库内链接和核心代码地图；不下载权重、不读取GT、不启动训练。原Full运行包的 `verify` 会要求被冻结的权重及资产，本次不能以该命令替代源码核验。

[安装说明](docs/Installation.md)、[Full运行说明](full_bundle/runtime/reproduction/README.md)与环境文件保留作为依赖依据。历史运行说明中的分支名、工作机路径及权重包含描述仅适用于当次交付；本次边界以发布说明为准。D-v3几何环境使用sklearn1.3.0，B26评分使用sklearn1.3.2，依赖环境需要分开准备。

原论文：[Open-YOLO 3D](https://arxiv.org/abs/2406.02548)，上游：[aminebdj/OpenYOLO3D](https://github.com/aminebdj/OpenYOLO3D)。第三方许可证保留于各源码目录。
