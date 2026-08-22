# FI1-D-v3 → ScanNet200 val312 冻结最终评测预注册

日期：2026-08-22

## 1. 目标与首要指标

本实验只回答一个问题：使用 `official100` 全量拟合的 FI1-D-v3，迁移到完整
ScanNet200 val312 后，类别无关主 AP 是否高于同次 FI1-Legacy 控制。

首要指标固定为类别无关 `AP`；`AP50` 与 `AP25` 为次要指标。不得因次要指标更高
而把主 AP 更低表述为整体超过，也不得根据 val312 结果修改模型、阈值、候选集合、
排序公式或重跑评测。

## 2. 固定训练来源

- 阶段 A、C-v2、D-v3 的最终回归器只使用最初 `official100` 的全部训练行拟合；
- 阶段 A 校准器只使用已经冻结的 official100 折外原始预测与标签拟合；
- 阶段 C-v2 和 D-v3 的加性偏差、绝对残差 q90 只由已经冻结的 official100
  折外残差计算；
- 不混入 NCS-train100、NCS-validation60 或 val312 标签；
- 最终包固定包含 7 个模型：3 个阶段 A 来源模型、2 个阶段 C 成员模型、
  1 个阶段 C 临时细化质量模型、1 个阶段 D-v3 排名边际收益模型。

模型包：

```text
pretrained/fi1_d_v3_official100_full_models_20260822/
```

## 3. 固定推理合同

val312 推理阶段不得读取 GT：

1. 使用 val312 已冻结的 FI1-Legacy native、track、pair-union、关系证据和类别；
2. 阶段 A 输出全量质量预测；
3. 阶段 B 只执行已经冻结的确定性连续衰减，native 安全保持；
4. 阶段 C-v2 只删除置信下界大于 0 的 exclusive atom，并强制保留 shared atom、
   连通性和最少 100 点回退；
5. refined union 只有在质量置信下界严格高于原 union 阶段 A 质量时才 append-only 追加；
6. 阶段 D-v3 使用完整排序前缀特征，分数固定为：

```text
candidate_reference_score × max(0, corrected_gain - q90)
```

7. 不删除任何原候选，不修改任何原几何、类别或掩码；refined union 继承原 union 类别。

## 4. 唯一控制与挑战方案

- 控制：同次计划中的 FI1-Legacy 全部冻结候选与原冻结分数；
- 挑战：native/track 使用阶段 B 分数，原 pair-union 使用阶段 D-v3 分数，
  再 append-only 加入通过阶段 C 和 D-v3 的 refined union；
- 两者必须由同一个 AP 入口、同一个官方评测器和同一批 GT 计算；
- 禁止把历史缓存汇总数值与本次挑战方案直接相减作为正式增量。

## 5. 执行与停止规则

1. 先完成无 GT 推理计划；
2. 独立审计必须为 0 错误；
3. AP 命令必须显式带 `--allow-gt-evaluation`；
4. AP 只运行一次，同时得到控制和挑战结果；
5. 结果无论高低均登记，不进行阈值扫描、权重扫描、候选子集选择或第二次运行；
6. 本实验不接入第二创新点，第二创新点代码和结果保持完全独立。
