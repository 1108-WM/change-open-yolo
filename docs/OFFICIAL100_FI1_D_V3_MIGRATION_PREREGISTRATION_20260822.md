# 第一创新点 FI1-D-v3：official100 冻结迁移评测预注册合同

状态：`frozen_before_first_official100_fi1_d_v3_run`

日期：2026-08-22

## 1. 目标与唯一数据范围

本实验把已在 `NCS-train100` 冻结的第一创新点 `FI1-D-v3` 原样迁移到最初用于第一创新点的 `official100` 固定五折，回答其在同一数据、同一类别无关 AP 合同下相对原始 OpenYOLO3D 和 `FI1-Legacy` 的变化。

唯一场景清单与五折为：

- `output/scannet200/scene_splits/official_train100_20260808/official_train100.txt`
- `output/scannet200/scene_splits/official_train100_20260808/oof_5fold_manifest.json`
- 场景清单 SHA-256：`dfa9017e206190eb2973b247c78e4bf1b2d9c01bb8468a15775c30335e44fb68`

禁止读取或混入 `NCS-validation60`、`NCS-train100`、`ScanNet200 val312`；禁止使用 GPU。所有新产物必须写入带 `official100_fi1_d_v3_20260822` 标识的新目录，不得覆盖任何历史产物。

## 2. 固定对照与历史数值

同数据、同五折的固定对照为：

| 系统 | AP | AP50 | AP25 |
|---|---:|---:|---:|
| 原始 OpenYOLO3D | 0.6540891345313419 | 0.8009084726963020 | 0.8486980447139625 |
| FI1-Legacy OOF | 0.6768351733792439 | 0.8323542165873690 | 0.8795907511062646 |

本次只新增 `FI1-D-v3 OOF` 一行。结果无论提高、持平或降低都必须登记，不得因结果方向更换合同。

## 3. 冻结输入迁移合同

迁移只允许做格式与目录适配，不允许重新选择候选、重新拟合 FI1-Legacy、修改几何、类别或历史分数：

- native：沿用原始冻结分数；
- track：沿用 `output/train_candidate_champion_pair_union_combined_oof_plan_official100_v1/champion_track_score_overrides.jsonl` 的 OOF `new_score`；
- pair-union：沿用同目录 `pair_union_append_candidates.jsonl` 的 1,501 个 OOF append 候选及其分数；
- track/pair-union 类别：从冻结的 official100 Z1 无 GT binding 物化继承语义；
- 关系特征：仅允许读取已有关系账本的定位字段和 `features`；必须先生成删除 `labels` 的 `relation_features_no_gt.jsonl`，后续阶段不得读取原含标签文件；
- 所有适配行显式声明 `ground_truth_usage=none`、`ap_evaluation_run=false`、候选/几何/类别/分数未修改。

适配结果必须经过独立审计：100 场景完整、五折恰好覆盖、来源逐项对应、链接只指向 official100 资产、1,501 个 union 完整、无标签字段、无 NCS/validation/val312 路径、无候选或几何修改。失败即停止。

## 4. 冻结 A/B/C-v2/D-v3 方法

阶段 A、B、C-v2、D-v3 完全复用此前 NCS 分支已经预注册并冻结的：

- 监督目标与特征定义；
- 固定五折场景隔离和下一折校准；
- 模型种类、容量和全部超参数；
- q90 不确定性界；
- 关系重排序、成员细化、完整排序前缀连续边际收益；
- 保守评分、append-only、全部原始候选 retained 与安全回退。

不得扫描或更换模型、特征、阈值、权重、指数、校准方式、分位数、候选子集或评分公式。代码中仅为正确标记数据集而进行的 `dataset_name`/合同文本参数化，不构成方法变化。

## 5. 顺序与停止门槛

严格按以下顺序执行：输入适配与独立审计 → A 数据与审计 → A OOF 与审计 → B 计划与审计 → C-v2 数据与审计 → C-v2 OOF 与审计 → D-v3 数据与审计 → D-v3 OOF 与审计 → 完整计划与审计 → 唯一一次 AP。

各阶段沿用其已冻结推进门槛。任一独立审计报错或任一门槛失败，立即停止；保留诊断，但不得运行后续阶段、不得调参重试、不得运行 AP。

GT 只能在各阶段冻结训练标签和最终授权评测中使用；所有推理特征、迁移计划与关系特征的 `feature_ground_truth_usage` 必须为 `none`。

## 6. 唯一一次类别无关 AP

只有全部阶段和完整计划审计通过后，才允许显式使用 `--allow-gt-evaluation` 执行一次 official100 类别无关实例 AP。评测固定报告总体 AP、AP50、AP25 和五折方向，不扫描任何设置。

最终比较固定为：

1. 原始 OpenYOLO3D；
2. FI1-Legacy OOF；
3. FI1-D-v3 OOF。

本合同签入后即冻结；首次运行前不得再根据预期结果修改。
