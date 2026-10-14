# NCS-5 ProxyCLIP 密集 mask-token 证据与 SMS 类别标准化预注册

预注册日期：2026-08-15

## 研究问题

NCS-4 已证明，冻结 SigLIP 在对象 masked crop 上产生的双 role 正 margin 虽然动作充足且类别分散，但不能可靠识别真实反排序：唯一 GT-only audit 的 improve/harm/neutral 为 `191/616/1,369`，五折全部失败。NCS-5 不修改 NCS-4，也不扫描其 crop、prompt、聚合或 margin；它检验一个结构上不同的新证据：在完整 RGB 帧上，用 DINOv2 的空间对应关系作为 CLIP 的 proxy attention，再只池化冻结 3D node 投影 mask 内的密集 token，是否能减少对象 crop 的背景混入和空间信息丢失。

NCS-5 仍只在已经物化的 native class representatives 内提出保守反排序动作。geometry、mask、candidate membership、class 和原 score multiset 全部冻结；第一阶段只生成 no-GT manifest、dense evidence ledger 与 proposal audit，不运行 AP。

## 方法来源与适配边界

- ProxyCLIP（ECCV 2024，`https://arxiv.org/abs/2408.04883`）提供训练自由的 proxy-attention 结构；
- Details Matter（`https://arxiv.org/abs/2507.23134`）提供按类别在场景 proposal 分布上标准化相似度的 SMS 思想，以及按对象可见率聚合多视角特征的依据；
- 本实验不是 ProxyCLIP 官方默认配置复现。官方代码默认 OpenAI CLIP ViT-B/16 + DINO v1 ViT-B/8；NCS-5 固定使用 OpenAI CLIP ViT-B/16 + 项目已有 DINOv2 ViT-S/14；
- NCS-5 的 SMS 是“按 scene/role/class 标准化后做已有代表间比较”的预注册适配，不执行论文的 proposal 删除，也没有 `tau_SMS`。

## 冻结资产与 provenance

完整机器可读记录见 `docs/NCS5_ASSET_PROVENANCE_20260815.json`。

- ProxyCLIP：`_external/ProxyCLIP`，commit `a0e38d1989c990651fda0d8fbf8aad41bde955b5`；
- OpenAI CLIP：`ViT-B/16`，`pretrained/checkpoints/ViT-B-16.pt`，SHA-256 `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`；
- DINOv2：`vit_small_patch14_dinov2`，`pretrained/checkpoints/dinov2_vits14_pretrain.pth`，SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`；
- ProxyCLIP 参数固定为 upstream `beta=1.2`、`gamma=3.0`，不扫描；
- 所有模型参数冻结，不 fine-tune、LoRA、prompt learning、temperature fitting 或类别监督校准。

正式代码必须显式读取本地 checkpoint，禁止运行时联网或 `torch.hub` 自动下载。正式 dense embedding 必须检查 CUDA；CUDA 不可见时立即退出，禁止静默回退 CPU。只允许程序生成图像的离线 smoke 在 CPU 上验证形状、有限值、确定性和 mask pooling 纯函数。

## 数据边界

- 开发集合仅为 `output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt`；
- 五折固定为 `output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100_folds_v1.json`，SHA-256 `e51f6aad4dcc20b758dc7cdde73ff9047c2a495da85ce3e1587a21290a2374e5`；
- NCS-validation60 完全冻结：不构建 NCS-5 manifest/ledger，不读取 RGB、candidate、GT 或 AP；
- official100、safety60、even48、test60 均不参与设计选择；
- NCS-5 第一阶段 `ground_truth_usage=none`，不得读取 NCS-1 的 GT-only class-quality 字段。

## 冻结对象与视角绑定

只处理 NCS-train100 的 `2,946` 个 Z1 native exact-geometry nodes。直接复用并绑定 NCS-4 已审计 manifest：

```text
docs/diagnostics/ncs4_siglip_view_manifest_train100_20260815/object_view_manifest.jsonl
SHA-256 c8d6680c587b6d2cd61cfee4ddc6abadc9950ded80526cfb5efe42df8dc0f001
```

因此 support/review 拆分、每 role top-3、visible-point 排序和 17,324 个 node-view bindings 均不重新选择。禁止根据 ProxyCLIP、CLIP、DINO、YOLO、Alpha、native score、class、SMS 或 GT 换帧。新 manifest 只登记完整帧预处理、投影 mask、唯一帧到多个 node-view binding 的映射与 provenance。

每个唯一 `(scene, frame_id)` 的完整 RGB 帧只编码一次，同帧 dense token 可由多个 node 只读复用。不得把不同 node 的 mask 合并后再编码，也不得因缓存命中顺序改变输出。

## 完整帧与投影 mask 合同

- 输入使用完整 RGB，不使用 NCS-4 的 bbox crop，不把 mask 外 RGB 替换为 127；
- ScanNet RGB 从 `1296x968` 固定 bicubic resize 到 `448x336`（width x height）；该尺寸保持约 4:3 比例，且同时被 CLIP patch 16 与 DINOv2 patch 14 整除；
- CLIP 分支使用 OpenAI CLIP mean/std，DINOv2 分支使用 ImageNet mean/std；两者共享完全相同的 resize 后 RGB 像素；
- 对每个 node-view 重建 NCS-4 已冻结的可见投影 sparse mask，在原始 RGB 分辨率膨胀 5 像素；其 sparse/dilated SHA-1 必须与 NCS-4 manifest 一致；
- dilated mask 以 nearest-neighbor resize 到 `448x336`，不做 bbox padding、crop 或额外形态学操作；
- mask 与 RGB 坐标、宽高和 token grid 必须显式审计。

## Proxy-attention 与密集 token 合同

DINOv2 最后一层 patch feature 为 `F in R^(H*W x 384)`，在 `448x336` 输入下固定 grid 为 `32x24`。对 feature channel 做 L2 normalize 后计算 token affinity：

```text
A = F_norm^T F_norm
A_masked = (A - 1.2 * mean(A)) * 3.0
A_masked[A_masked < 0] = -inf
P = softmax(A_masked, dim=-1)
```

`P` 替换 OpenAI CLIP ViT-B/16 最后一个视觉 transformer block 的原 attention；CLIP value token 双线性插值到 DINOv2 `32x24` grid 后由 `P` 聚合，使用 upstream output projection，得到每个 DINO grid token 的 512 维 CLIP-space feature。batch 固定为每次一个唯一帧，避免 `mean(A)` 跨帧耦合。不得修改层位置、加入残差混合、平均 CLIP 原 attention、扫描 beta/gamma 或选取其他 DINO 层。

每个输出 token 先 L2 normalize。令 `m_t` 为 resize 后 binary mask 在对应 `14x14` DINO patch 中的 foreground 比例。只保留 `m_t > 0` 的 token，并固定：

```text
e_view = L2Norm(sum_t(m_t * token_t) / sum_t(m_t))
```

`sum_t(m_t)=0` 时该 node-view 显式 missing；不回退到 bbox、全帧、最近 token 或另一 role。

## 多视角与文本合同

对 role `r` 中每个合法 view，权重固定为 NCS-4 manifest 的 `visible_point_ratio alpha_v`：

```text
e_node,r = L2Norm(sum_v(alpha_v * e_view) / sum_v(alpha_v))
```

不得改为等权、max、medoid、learned attention 或根据 CLIP/DINO similarity 加权。support/review 始终独立聚合；缺失 role 不复制另一 role。

类别严格使用 `pretrained/config_scannet200.yaml` 中顺序不变的 198 个 `network2d.text_prompts`。文本模板固定为注册 ProxyCLIP commit 的 `prompts.imagenet_template.openai_imagenet_template` 全部 80 项。每个模板 embedding 先 L2 normalize，每类求算术均值后再 L2 normalize，得到 `T in R^(198x512)`；模板、prompt 列表和文本 embedding 文件均记录 SHA-256。禁止删模板、增同义词、类别描述生成或 prompt 扫描。

原始 role similarity 为：

```text
L[n,r,c] = dot(e_node,r, T[c])
```

所有 similarity、均值和标准差以 float32 保存和统计。

## SMS 式类别尺度标准化

对每个 scene、role `r`、class `c`，只在该 role embedding 有效的 scene nodes 上计算 population statistics：

```text
mu[scene,r,c]    = mean_n L[n,r,c]
sigma[scene,r,c] = sqrt(mean_n (L[n,r,c] - mu[scene,r,c])^2)
Z[n,r,c]         = (L[n,r,c] - mu[scene,r,c]) / max(sigma[scene,r,c], 1e-6)
```

support 与 review 各自计算统计，禁止相互泄漏。若有效 node 少于 2 或某类 `sigma <= 1e-6`，该 scene/role/class 标记 invalid，不以全局、其他 scene、另一 role 或 GT 统计回填。Z-score 不 clip、不 temperature scale、不 softmax。

这是 SMS 的类别尺度校正，不是 proposal filtering：NCS-5 不删除任何 prediction，不设置或扫描 `tau_SMS`，也不把 SMS 值复制成 native score。

## 已有代表内的无 GT proposal 规则

每个 exact geometry node 按 class 折叠已有 native candidates；同类 canonical representative 取 native score 最高者，分数并列取 candidate id 最小者。incumbent 是 canonical representatives 中 native score 最高者；背景/无效类 198 不进入动作空间。

对每个 native score 严格低于 incumbent 的已有 alternative class `c`：

```text
support_margin(c) = Z[node,support,c] - Z[node,support,incumbent]
review_margin(c)  = Z[node,review,c]  - Z[node,review,incumbent]
joint_margin(c)   = min(support_margin(c), review_margin(c))
```

只有两种 role 和涉及的四个 Z-score 均有效，且 support/review margin 都严格大于 0，alternative 才 actionable。多个 alternative 固定选择 `joint_margin` 最大者；float32 精确并列取 class index 最小者。本阶段只记录 proposal，不交换分数、不写 inference plan、不读取 GT。

## 分阶段执行与推进门槛

### A. 离线 smoke

只使用程序生成 RGB 和 binary masks，验证两个 checkpoint 均严格加载、proxy attention 输出为有限 `32x24x512` token、全空 mask 正确 missing、非空 mask pooling 有限、同输入重复两次结果一致。smoke 不读取 train100/validation60/GT；CPU 仅允许用于本步骤。

### B. no-GT manifest

生成 NCS-5 derived manifest，必须满足：

1. NCS-4 输入 manifest SHA-256 精确匹配；
2. `100 scenes / 2,946 nodes / 17,324 bindings` 完整，duplicate/missing node 为 0；
3. geometry point count/hash、RGB、depth、pose、intrinsics、sparse/dilated mask hash 均一致；
4. 唯一 frame 表无重复 key，每个 binding 恰好连接一个 frame；
5. resize/mask/token grid 坐标审计错误为 0；
6. `ground_truth_usage=none`、`validation60_read=false`、无模型推理、无候选或分数修改。

### C. GPU-only dense ledger 与 action audit

只有 A/B 全通过才允许正式 GPU embedding。必须满足：

1. 两个 checkpoint 与 ProxyCLIP commit/hash 精确匹配，CUDA 可见且无联网；
2. 每个唯一 frame 恰好编码一次；所有 token、view/role embedding、text embedding、raw similarity、SMS statistics 和 Z-score 有限或按合同显式 invalid；
3. embedding 维度、`32x24` grid、mask-token binding、scene/role/class SMS 统计全部正确；
4. 五个冻结 folds 每折至少 1 个 actionable node；
5. proposals 覆盖至少 25 scenes 和 20 target classes；
6. 最大 proposed target class 占比 `<=20%`；
7. 所有 alternative 均已物化且原 native score 严格低于 incumbent；candidate/class/geometry/score mutation 全为 0。

同时只报告、不作为调参依据：unique frame 数、mask token 覆盖率、support/review raw/Z top-1 一致率、跨 role cosine、与 YOLO/Alpha/SigLIP top-1 的一致率及 margin 相关性。任一门槛失败即在 AP 前停止；不得改 resize、mask dilation、DINO 层、beta/gamma、模板、可见率权重、epsilon 或 proposal 规则后重试。

## 唯一允许的后续 GT-only audit

只有 A/B/C 全通过，才允许另行实现一次 NCS-train100 GT-only proposal audit，复用冻结 NCS-1 class-quality ledger，把 proposal 标记为 alternative quality 严格高于 incumbent 的 improve、严格低于的 harm、相等的 neutral。

推进到 score-swap plan 前必须同时满足：五折各至少 1 个 proposal，且每折 `improve > harm + neutral`；最大 target class 占比仍 `<=20%`。失败即封闭 NCS-5，不训练 selector、不扫描 SMS threshold、confidence、融合权重或组合 NCS-1/2/3/4 特征。

若通过，未来 plan 仍只能交换 incumbent 与所选已有 alternative 的两个原 native scores，保持 node/scene/global score multiset 严格守恒。plan audit 通过后才授权一次 NCS-train100 AP；只有 AP/AP50/AP25 全正、主 AP 至少 4/5 折为正且 head/common/tail 不下降，才允许冻结完整方法并申请 NCS-validation60 唯一一次迁移评测。

## 明确不做

- 不重跑或修改 NCS-4；
- 不下载 DINO v1 ViT-B/8、DINOv2 ViT-B/14-reg、SAM、MAE 或其他 VLM；
- 不运行 ProxyCLIP 自带 VOC/ADE/COCO evaluation；
- 不创建新类别 prediction，不覆盖 native top-1，不删除低 SMS proposal；
- 不训练 CLIP/DINO、selector 或新的 198 类分类器；
- 不读取 NCS-validation60 GT，不提前计算 train100/validation60 AP；
- 不根据 no-GT 报告或 GT-only 结果回调任何已冻结设计。
