# PCV-1 直接成对类别验证器预注册

预注册日期：2026-08-13

## 目的与边界

PCV-1（Pairwise Class Verifier v1）是第二创新点的结构性新分支。它回答：在冻结三维 geometry node 和冻结类别候选集合上，能否直接判断某个 alternative class 是否比 current class 更可信。

本分支不是 Z6c/Z6d 的参数修补。旧路径先用绝对 selector 从全部类别中选择一个 proposal，再用 gate 判断该 proposal 是否改善；PCV-1 对每个 `(current, alternative)` 直接建模，不依赖旧 selector 的 argmax proposal。

固定边界：

- 不使用 VLM，不调用或修改 Qwen；
- 不重新运行 YOLO-World、Alpha-CLIP 或 DINOv2；
- 只复用 official100 已冻结的 Z6c 监督账本及其上游无 GT 证据；
- 候选集合保持为 current 与已有 YOLO top-5 / Alpha top-5 union；
- 不修改 geometry、candidate membership、原始类别或分数；
- 不读取 safety60、even48 或 test60；
- 第一阶段只构建 GT-supervised official-train pair 账本和覆盖审计，不训练、不生成推理计划、不计算 AP。

## Pair 定义与三态监督

每个具有合法 current class 的 prediction 与其余每个注册 alternative class 形成一条 pair。official train GT 只用于监督标签与审计。

在该 prediction 的最佳同对象 geometry IoU `>=0.50` 时：

- `improve`：alternative 是目标类别，current 不是；
- `harm`：current 是目标类别，alternative 不是；
- `neutral`：current 与 alternative 均不是目标类别。

geometry IoU `<0.50` 的 pair 统一标记为 `neutral`，因为单纯换类不能使其成为 TP50。current 与 alternative 相同的行不得形成 pair。

## 固定特征合同

PCV-1 不使用 class ID、class name、类别频率、head/common/tail 标识或类别专属参数作为模型特征。

第一版特征固定为两组：

1. 类别无关的 prediction/node context：来源、点数、exact-node 绑定数、当前 hybrid score、YOLO/Alpha 冲突与 entropy/JS、视角数，以及已有 DINO 多视图稳定性字段；
2. alternative 相对 current 的类别证据差：YOLO/Alpha probability、max probability、top-5 support、reciprocal rank 和 top-1 indicator 的逐字段差值。

DINO 只作为节点级视角可靠性上下文，不作为类别分类器。第一版不增加逐类文本描述、场景关系、类别 embedding 或新视觉推理。

## 固定样本权重

每个 prediction 的全部 alternative pair 总权重固定为：

- native：`1 / exact_geometry_bound_candidate_count`；
- track / pair-union：`1`。

若 prediction 有 `K` 个 alternative，每条 pair 权重为上述总权重的 `1/K`。这样候选较多的 prediction 和同一 exact native geometry 的类别副本不会支配训练。

## 第一阶段数据可行性门槛

第一阶段工具只输出 pair 账本和审计。只有同时满足以下预注册门槛，才允许进入模型训练：

- 100 个 official train 场景全部属于固定五折且不存在跨折场景；
- 每折均同时包含 `improve`、`harm`、`neutral`；
- `improve` 总计至少 200 个 unique semantic node，且每折至少 20 个 unique semantic node；
- `improve` 覆盖至少 50 个场景，且每折至少覆盖 10 个 validation 场景；
- 单个场景不超过全部 `improve` pair 的 10%；
- 单个目标类别不超过全部 `improve` pair 的 20%；
- 所有特征、标签和 sample weight 有限，且每个 prediction 的 pair weight sum 精确满足固定合同。

任一门槛失败则 PCV-1 停在数据审计，不降低门槛、不改标签、不扩大候选集合。

## 后续唯一训练合同（本阶段不执行）

若数据审计通过，下一阶段只允许一次 official100 场景隔离五折 OOF：

- 模型：`HistGradientBoostingClassifier` 三分类；
- `learning_rate=0.05`；
- `max_iter=100`；
- `max_leaf_nodes=5`；
- `min_samples_leaf=50`；
- `l2_regularization=1.0`；
- `random_state=20260813`；
- 使用自然三态分布和上述 sample weight，不做类别平衡、容量扫描或特征子集扫描。

冻结接受规则：

```text
对每个 alternative 预测 P(improve), P(harm), P(neutral)
只保留 P(improve) > 0.5 且 improve 为三态 argmax 的 alternative
若多个 alternative 通过，选择 P(improve) - P(harm) 最大者
精确并列时按已注册 class index 升序确定性打破
没有 alternative 通过时保持 current
```

训练后必须先做 action-level OOF 审计。只有每折 accepted improve 严格多于 accepted harm，才允许生成冻结 plan 并运行一次 official100 AP。阈值不得根据 action audit 或 AP 扫描。

## 明确不做

- 不与 Z6f VLM 组合；
- 不重跑或删改既有 Z6f 结果；
- 不使用 safety60 选择模型、特征、阈值或接受动作；
- 不根据 pair 标签手工制定类别对规则；
- 不把 GT target class 写入无 GT 推理计划。
