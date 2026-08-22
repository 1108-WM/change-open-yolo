# 第一创新点阶段 D-v3：完整排序前缀连续边际收益与冻结 OOF AP 预注册合同

状态：`frozen_before_first_run`

日期：2026-08-22

## 1. 唯一修正与实验边界

D-v2 已通过数据、结果与代码审查，但其标签前缀只包含 native/track；最终 AP 排序中，分数更高的 original/refined append 候选同样会先覆盖 GT。D-v3 只修正这一处标签与特征条件，使训练前缀和最终排序竞争一致。模型参数、五折隔离、下一折校准、q90、候选集合与保守评分公式全部沿用 D-v2，不根据 D-v3 或 AP 结果调参。

本阶段仅使用 `official100 / NCS-train100`。禁止读取 `NCS-validation60`、`val312`，禁止 GPU，禁止修改 DM-SMS-1、冻结冠军、native geometry、类别或任何已有几何。所有候选 retained；refined union 仅作为 append-only 新几何加入。

## 2. 候选、参考分数和完整排序前缀

固定动作集合为 1,578 个 original union 与 85 个 C-v2 append-eligible refined union。

参考分数固定为：

- native/track：Stage-B score；
- original union：Stage-B original union score；
- refined union：C-v2 refined-Q 下置信界裁剪到 `[0,1]`。

候选键固定为 `scene:union:NNNN:{original|refined}`。baseline 在同分时排在 append 前；append 同分按候选键升序稳定排序。对 append 候选 `c`：

```text
prefix(c) =
  all native/track with score >= score(c)
  + all other append candidates a satisfying
      score(a) > score(c)
      or score(a) == score(c) and candidate_key(a) < candidate_key(c)
```

当前候选本身不得进入自己的前缀。

## 3. 固定连续标签与无 GT 特征

主标签固定为：

```text
max_gt max(0, candidate_iou(gt) - full_rank_prefix_best_iou(gt))
```

GT 只用于该标签及冻结诊断字段，不能进入特征、参考分数或排序。D-v3 沿用 D-v2 全部特征，但所有 `rank_prefix_*` 重叠统计改为完整前缀，并新增：

- `rank_prefix_native_count`；
- `rank_prefix_track_count`；
- `rank_prefix_original_union_count`；
- `rank_prefix_refined_union_count`；
- `rank_prefix_append_fraction`。

append 前缀项用于重叠分数统计时，original 的质量为 Stage-A original union OOF，refined 的质量为 C-v2 corrected refined-Q；其 Stage-B-like 分数均取本合同的候选参考分数。

## 4. 冻结模型、校准和评分

模型仍为一个 `HistGradientBoostingRegressor`：

```text
learning_rate=0.05
max_iter=160
max_leaf_nodes=15
min_samples_leaf=30
l2_regularization=1.0
early_stopping=False
```

每个外层测试折以 `(outer_fold + 1) % 5` 为校准折；只做平均残差加性校正、绝对残差 q90 与 `[0,1]` 裁剪：

```text
conservative_gain = max(0, corrected_prediction - q90)
stage_d_v3_append_score = candidate_reference_score * conservative_gain
```

不扫描模型、阈值、分位数、权重、指数或评分公式。

## 5. 数据与模型推进门槛

数据门槛固定为：独立审计 0 错误；8,591 native/track、1,578 original、85 refined 完整覆盖；正收益至少 350；六位小数不同正值至少 200；五折均有正收益；完整前缀顺序无违规；所有禁止修改/读取/评测计数为 0。

模型门槛沿用 D-v2：独立审计 0 错误；OOF MAE 严格优于零预测与冻结 threshold-cross 数值对照；Spearman > 0；五折均有 conservative gain > 0；高置信候选真实正收益比例至少 0.70；分数有限、非负且不高于参考分数；所有禁止修改/读取/AP 计数为 0。

任一门槛失败即停止，不运行 AP，也不以结果为依据修改合同。

## 6. 唯一一次 official100 冻结 OOF AP 评测

只有 D-v3 数据与模型门槛全部通过后，才允许构建完整 OOF 计划并执行一次固定 class-agnostic AP 评测。

控制组固定为 unique geometry ledger 中全部 10,169 个冻结几何，分数为 `canonical_frozen_score`，类别与几何不变。

挑战组固定为：

- native：Stage-B score；
- track：Stage-B score；
- original pair_union：D-v3 OOF append score；
- 85 个 refined union：D-v3 OOF append score，类别继承其 original union；
- 所有 10,169 个冻结候选 retained，不删除；refined 仅追加。

完整计划必须独立审计来源覆盖、候选身份、几何哈希、分数、类别继承与 append-only 合同。评测命令必须显式带 `--allow-gt-evaluation`，只比较上述 control 与唯一 challenger，输出总体 AP/AP50/AP25 和五折方向；不得扫描阈值、权重、指数或候选子集，不得读取 validation60/val312。
