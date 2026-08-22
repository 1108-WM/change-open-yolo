# 第一创新点阶段 A：三来源统一几何质量预注册合同

状态：`frozen_before_first_run`

日期：2026-08-22

## 1. 目标

本分支只在 `NCS-train100` 上验证第一创新点阶段 A：让 native、track、pair-union 三类冻结几何候选预测同一个连续几何质量目标，从而判断三来源分数能否进入统一标尺。

本分支不修改候选几何、掩码、成员关系、类别或冻结基线分数，不运行 AP，不读取 `NCS-validation60` 或 `ScanNet200 val312`，也不修改 `DM-SMS-1`。

## 2. 冻结输入

- 场景清单：`output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt`
- 五折清单：`output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100_folds_v1.json`
- 唯一几何账本：`/media/jia/软件1/scannet_train_stream/dm_sms1_ncs_train100_20260817/unique_geometry_ledger/`
- 唯一几何独立审计：同一运行根目录下的 `unique_geometry_audit/`
- 无标签 GVC 质量账本：同一运行根目录下的 `gvc_quality_ledger/`
- 无标签候选关系账本：同一运行根目录下的 `relation_ledger/`
- 冻结冠军计划：同一运行根目录下的 `champion_plan/`
- 训练标签：`/media/jia/软件1/scannet_train_stream/prepared_ncs/ground_truth/`

真实标签只允许生成 `label_*` 字段，不得进入特征。

## 3. 统一监督目标

每个唯一几何候选先与同场景所有有效真实实例计算最大类别无关 IoU。固定阈值集合为：

```text
0.50, 0.55, 0.60, 0.65, 0.70,
0.75, 0.80, 0.85, 0.90, 0.95
```

统一目标为：

```text
Q = mean[1(best_IoU >= threshold)]
```

因此 `Q` 只能取 `0.0, 0.1, ..., 1.0`，且三种来源完全使用同一定义。不得使用唯一 winner 标签，不得把高质量重复候选强制标成 0。

## 4. 特征合同

三个来源分别拟合，但输出都表示同一个 `Q`。

- native/track：只使用候选规模、冻结分数和来源帧排除后的 GVC 多视图统计；
- pair-union：只使用候选规模、冻结分数、冻结父候选质量、原 threshold-cross 输出和冻结关系几何/多视图统计；
- 禁止使用类别编号、场景编号、候选编号、真实实例编号、真实语义编号、IoU 或任何 `label_*` 字段作为特征。

旧 `q`、`P(keep)` 和 `P(threshold_cross)` 只能作为输入证据，不能被解释为监督目标或统一概率。

## 5. 五折与模型

沿用固定主场景隔离五折。每个外折：

1. 固定取 `(外折编号 + 1) mod 5` 为校准折；
2. 其余三个折拟合来源独立的低容量 `HistGradientBoostingRegressor`；
3. 校准折使用固定 `IsotonicRegression(out_of_bounds=clip)`；
4. 外折只生成一次预测。

回归器参数固定为：

```text
learning_rate=0.05
max_iter=120
max_leaf_nodes=7
min_samples_leaf=25
l2_regularization=1.0
early_stopping=false
random_state=20260822 + 外折编号
```

不扫描模型容量、损失、特征、分桶、校准方法或来源权重。若校准折不足以拟合单调校准器，则使用训练折目标均值作为显式常数回退，并记录原因。

## 6. 输出账本

### 数据账本

逐唯一几何保存：

- 场景、固定折、geometry key/hash、来源和只读定位信息；
- 冻结分数和无标签特征；
- `label_best_gt_iou`、最佳真实实例元数据和统一 `label_quality_q`；
- 所有输入摘要和 SHA-256 来源登记。

### 五折账本

逐几何保存唯一外折预测、校准前预测、目标、来源、外折和校准折。模型只保存到新分支输出目录，不写回冻结冠军。

### 审计与指标

必须报告：

- 每来源和每折的样本数；
- MAE、RMSE、Spearman、平均预测、平均目标和绝对偏差；
- 10 个固定等宽分桶的可靠性曲线和校准误差；
- 冻结分数对照的同一组指标；
- 外折覆盖、重复、非有限值、特征泄漏和场景隔离错误数。

## 7. 阶段 A 推进门槛

只有同时满足以下条件，才授权进入阶段 B 的关系级账本与集合重排序实现：

1. 数据和五折审计错误数均为 0；
2. 每个来源在每个外折均有样本，全部几何恰好获得一个外折预测；
3. 全量统一质量 MAE 严格优于冻结分数对照；
4. 每个来源的 MAE 相对冻结分数最多允许恶化 `0.02`；
5. 每个来源的平均预测与平均目标绝对偏差不超过 `0.10`；
6. 任一外折全量 MAE 不超过 `0.30`。

失败时保留全部诊断账本，但不生成替换分数计划、不运行 AP、不读取冻结验证集，也不通过调参数重试本分支。

## 8. 始终不变的边界

- native 候选和原有 geometry 始终保留；
- append-only 与安全回退不变；
- 当前冻结冠军不覆盖、不删除、不重命名；
- 本阶段不实现 near duplicate、containment、complementary 等关系动作；
- 本阶段不实现 pair-union 成员细化；
- 本阶段不训练连续边际收益模型；
- 不运行 AP、GPU 任务、validation60 或 val312。
