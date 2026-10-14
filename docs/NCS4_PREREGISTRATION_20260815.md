# NCS-4 冻结 SigLIP 对象级多视图 embedding 证据预注册

预注册日期：2026-08-15

## 研究问题

NCS-1/NCS-2/NCS-3 已共同排除“只重组现有 YOLO/Alpha/support/review 特征”的 selector 路线。NCS-4 检验一个新的独立信息源：冻结 SigLIP 是否能通过对象级多视图视觉 embedding，为 geometry node 中已经物化的 native 类别代表提供可靠的反排序证据。

NCS-4 不创建类别、不修改 geometry、不训练或微调 SigLIP，也不把逐视角类别分数直接平均。它先在视觉 embedding 空间聚合对象的多视图外观，再与冻结文本 embedding 匹配。第一阶段只构建无 GT manifest、embedding ledger 与 action-space audit，不运行 AP。

## 冻结模型与运行时

- 模型：`google/siglip-base-patch16-224`；
- Hugging Face revision：`7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`；
- 本地路径：`_external/SigLIP/siglip-base-patch16-224`；
- 权重 SHA-256：`2c63cb7d1f2e95ba501893cbb8faeb4ea9a3af295498d35097126228659c2af8`；
- 模型参数全部冻结，禁止 fine-tune、LoRA、prompt learning 或温度拟合；
- 使用隔离的 `transformers==4.37.2` 与 `sentencepiece==0.1.99`，不覆盖 OpenYOLO3D 原环境；
- 所有正式运行必须设置 `TRANSFORMERS_OFFLINE=1`、`HF_HUB_OFFLINE=1` 和 `local_files_only=True`。

正式 smoke test：`docs/diagnostics/ncs4_siglip_offline_smoke_20260815/`。它只使用两张程序生成的纯色图像，不读取 train100、validation60 或 GT。

## 数据边界

- 开发集合仅为 `output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt`；
- NCS-validation60 冻结，不构建其 SigLIP ledger，不读取其 RGB、候选、GT 或 AP；
- safety60、official100、even48、test60 均不参与 NCS-4 设计选择；
- 几何、native candidate、native class、native score、Z1/Z2c 账本和五折 scene manifest 全部冻结；
- NCS-4 第一阶段 `ground_truth_usage=none`，不得导入 NCS-1 的 GT-only quality 字段。

## 冻结对象与视角合同

只处理 Z1 native train100 的 `2,946` 个 exact geometry nodes。每个 node 的 geometry 来自冻结 native mask column，并必须重新核验 point count 与 SHA-1 geometry hash。

视角只从 Z1 已注册的 `distribution.views` 取得，不重新扫描全部视频帧。每个 frame 使用 NCS-1 已冻结的确定性拆分：

```text
sha256("ncs1-view-split-v1:" + frame_id)[0] % 2 == 0 -> support
otherwise                                             -> review
```

support/review 分别按 `visible_point_count` 降序、`frame_index` 升序、`frame_id` 升序取最多 3 帧。禁止根据 SigLIP、YOLO、Alpha、native score、class 或 GT 重新选视角。

每帧由冻结 3D node 投影得到 foreground sparse mask 和 tight bbox：

- bbox padding ratio 固定为 `0.50`，裁剪到图像边界；
- foreground mask 固定膨胀 5 像素；
- bbox 内 mask 外像素固定替换为 RGB `(127,127,127)`；
- 使用模型原生 processor resize 到 `224×224`；
- 若某个 role 无合法视角，则该 role 显式 missing，不允许复制另一 role 或回退到全图。

## 冻结文本与 embedding 合同

类别集合严格使用 `pretrained/config_scannet200.yaml` 中顺序不变的 198 个 `network2d.text_prompts`。每类唯一模板为：

```text
a photo of a {label}
```

不做模板 ensemble、同义词扩展、类别描述生成或 prompt 扫描。198 个文本 embedding 只编码一次并逐行 L2 normalize；prompt 列表、文本 embedding 文件和模型 manifest 均记录 SHA-256。

每个视角产生一个 768 维视觉 embedding 并 L2 normalize。support/review 内分别先对视觉 embedding 求算术均值，再 L2 normalize，最后与 198 个文本 embedding 计算 cosine similarity。该顺序固定为“先聚合视觉 embedding、后匹配文本”，禁止改成逐视角 class score 平均、max pooling、medoid、learned attention 或可学习融合。

## 只在已有代表内定义无 GT 动作信号

每个 node 按 class 折叠已有 native candidates；同类 canonical representative 取 native score 最高者，分数并列取 candidate id 最小者。incumbent 是 canonical representatives 中 native score 最高者。背景/无效类 198 不进入动作空间。

对每个低分 alternative class `c` 定义：

```text
support_margin(c) = SigLIP_support(c) - SigLIP_support(incumbent)
review_margin(c)  = SigLIP_review(c)  - SigLIP_review(incumbent)
joint_margin(c)   = min(support_margin(c), review_margin(c))
```

只有 support/review 均存在且两项 margin 都严格大于 0 时，`c` 才是 actionable alternative。若有多个，固定选择 `joint_margin` 最大者；精确并列取 class index 最小者。该阶段只记录 proposal，不交换分数、不写 inference plan、不读取 GT。

## 第一阶段推进门槛

无 GT manifest/embedding/action-space audit 必须同时满足：

1. 100 个 train scenes、2,946 个 nodes 与冻结 Z1/Z2c exact join，duplicate/missing node 为 0；
2. 所有注册 RGB、depth、pose、intrinsics 资产存在，geometry hash 全部一致；
3. 所有生成的视觉/文本 embedding 与 cosine similarity 有限且维度正确；
4. 五个冻结 scene folds 每折至少有 1 个 actionable node；
5. actionable proposals 覆盖至少 25 个 scenes 和 20 个 target classes；
6. 最大 proposed target class 占比 `<=20%`；
7. 所有 proposal 的 alternative 已物化、分数严格低于 incumbent，且无 candidate/class/score/geometry mutation。

同时报告但不作为可调门槛：support/review top-1 一致率、跨 role cosine、与 YOLO/Alpha top-1 的一致率和 margin 相关性。不得根据这些统计更换模板、视角数、padding、mask、聚合或阈值。

任一门槛失败，NCS-4 在 AP 前停止；不得降低为单 role 接受、扫描 margin threshold、扩大 top-k、删除高频类或更换 prompt。

## 后续唯一允许的 GT-only 信号审计

只有第一阶段全部通过，才允许另行实现一次 train100 GT-only proposal audit。它复用冻结 NCS-1 class-quality ledger，把 proposal 标记为：alternative quality 严格高于 incumbent 为 improve，严格低于为 harm，相等为 neutral。

推进到 score-swap plan 前必须满足：五折各至少 1 个 proposal，且每折 `improve > harm + neutral`；最大 target class 占比仍 `<=20%`。失败即停止，不训练 selector、不扫描置信度或组合现有 NCS-3 特征。

若通过，未来 plan 仍只能在 incumbent 与所选已有 alternative 之间交换原 native score，保持 node/scene/global score multiset 严格守恒。plan audit 通过后才授权一次 train100 AP；只有 AP/AP50/AP25 全正、主 AP 至少 4/5 折为正且 head/common/tail 不下降，才冻结完整方法并申请 NCS-validation60 唯一一次迁移评测。

## 明确不做

- 不下载或并行尝试 OpenCLIP；
- 不使用 Qwen 或生成式 VLM；
- 不把 SigLIP top-1 直接覆盖 native class；
- 不创建未物化类别 prediction；
- 不读取 validation60 GT 或提前运行 validation AP；
- 不运行 train100 AP，除非前置 no-GT 与 GT-only 两级门槛均通过；
- 不基于结果回调模板、视角选择、mask、聚合、margin 或集中度规则。
