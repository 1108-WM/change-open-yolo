# DM-SMS-1A 正式预注册：冻结第一创新点上的 Alpha-CLIP 主分类与 SMS proposal filtering

预注册日期：2026-08-17

状态：`frozen_before_gt_ap_or_formal_gpu_embedding`

## 研究问题

在 NCS-train100 上完整物化第一创新点的冻结 unique geometry proposals 后，严格执行 Details Matter 的 Alpha-CLIP 主分类合同与真正的 SMS proposal filtering，能否稳定改善开放词汇实例分割，而不是仅在同一 geometry 的类别副本间重新换分。

本实验是独立于 NCS-4/NCS-5/NCS-6 的训练自由控制。它不修改或重启这些失败分支。

## 数据边界

- 开发与内部五折只使用 `NCS-train100`；scene list SHA-256 为 `ecebf3a682a02665a9770c2fa0ef83cc61c0c4ec2f185820aa44d667c8c4cd40`。
- 五折固定文件 SHA-256 为 `e51f6aad4dcc20b758dc7cdde73ff9047c2a495da85ce3e1587a21290a2374e5`。
- `NCS-validation60` 在完整方法冻结前禁止读取 GT、运行 AP、选择 prompt、过滤率或规则；冻结后最多评测一次。
- `ScanNet200 val312` 只用于 validation60 后的最终论文评测，不参与本实验设计。
- official100、safety60、even48、test60 只保留历史结果，不参与选择。

## 冻结第一创新点 proposal 合同

必须在 NCS-train100 上原样物化：

1. uniform30 自动 SAM 观测；
2. hierarchy-safe 同帧处理；
3. Details frame-wise sIOU tracks；
4. 多视图共识与 D2b；
5. native-track 严格互重复处理；
6. 冻结冠军轨迹伤害连续抑制；
7. prior-corrected pair-union append-only；
8. native/track/pair-union 跨来源 exact geometry 折叠。

所有模型、特征公式、prior correction、指数和候选规则复用冻结 official100 冠军，不训练 NCS 模型、不扫描参数。native 不删除，track 不硬删除，pair-union 只追加。

SMS population 的单位是场景内跨来源折叠后的 unique geometry proposal。每个 geometry group 的 canonical frozen prediction 固定取冻结分数最高的成员，精确并列时按 `native < track < pair_union` 的来源优先级，再取最小 candidate id。完整成员、来源、类别、分数和 point hash 均保留在 provenance 中。

## Alpha-CLIP 主分类合同

- 模型和 checkpoint 使用 `docs/DM_SMS1_ASSET_PROVENANCE_20260817.json` 中登记的本地资产，全部参数冻结，禁止下载或微调。
- 类别严格为 `pretrained/config_scannet200.yaml` 的 198 个有效 prompts，顺序不可改变，不加入背景类或伪造两个类别。
- 文本模板固定为：`a blurry photo of {CLASS_NAME} in a room.`。
- 每个 unique proposal 从 ScanNet 降采样帧域中按可见 3D 点数降序选择 top-20；并列按 frame index、frame id 升序。
- 可见率 `alpha_v = visible proposal point count / total proposal point count`。
- 每个视角使用 tight projected bbox，并生成三个确定性 scale：bbox 各边分别按原 bbox 宽高的 `0.0 / 0.2 / 0.4` 比例向外扩展；裁剪前转为保持中心的 square box并截断到图像边界。
- projected tight bbox 作为 SAM ViT-B box prompt；选择 SAM 单个最高 predicted-IoU mask，精确并列取最小 mask index。该同一全图 SAM mask在三个 crop scale 中裁剪并作为 Alpha channel；无合法 SAM mask时该 view 显式 missing，不回退投影点膨胀 mask。
- 每个 view/scale 分别提取 L2-normalized Alpha-CLIP visual feature。
- proposal feature 固定为 `L2Norm(sum_v sum_l alpha_v * f[v,l])`，不做等权 mean、max、medoid、attention 或相似度加权。
- 与 198 个 L2-normalized text features 计算原始 cosine similarity `L[K,198]`。SMS 使用 similarity，不使用 softmax probability 或 temperature-scaled logits。

资产边界澄清（2026-08-17，仍在 GT/AP/Alpha embedding 之前）：第一创新点的 `uniform30` 与既有 exporter 一致，含义为在可用降采样帧中最多均匀选择 30 帧；若场景可用帧少于 30，则无参数变化地使用全部可用帧。该澄清只处理 `scene0150_02` 等短序列的客观帧数，不改变采样规则、候选或任何评测决定。

## SMS 合同

对每个场景的 `K` 个 unique proposals 和每个类别 `c`：

```text
mu[c]    = mean_k L[k,c]
sigma[c] = sqrt(mean_k (L[k,c] - mu[c])^2)
cmax[k]  = argmax_c L[k,c]
SMS[k]   = (L[k,cmax[k]] - mu[cmax[k]]) / sigma[cmax[k]]
```

`mu/sigma` 必须使用全部 K 个 unique proposals；禁止只使用 top-1 为该类别的 proposal 子集。float32 精确 argmax 并列取最小 class index。

阈值固定 `tau_SMS=0`，不扫描。只有 `sigma>1e-6` 且 `SMS<0` 时删除 proposal；退化方差、缺失 Alpha 或不完整 3-scale feature 的 proposal 保守保留并在报告中标记 invalid，不用全局统计或另一场景回填。

## 固定比较系统

在完全相同的 unique geometry ledger 上比较：

1. `frozen_semantic_control`：使用 canonical frozen class，不做 SMS；
2. `details_alpha_main_no_sms`：类别替换为 Alpha cosine argmax，不做 SMS；
3. `details_alpha_main_sms_tau0`：同一 Alpha 结果，删除有效且 `SMS<0` 的 proposal。

同时报告两个预先固定且互不选择的 score contract：

- `details_unit_confidence`：所有保留 proposal 的 evaluator confidence 固定为 `1.0`，作为论文合同主结果；
- `frozen_geometry_score`：保留 canonical 第一创新点冻结分数，作为项目集成次结果。

不得根据其中一个结果选择另一个作为主结果。系统顺序、geometry order 和 equal-score tie order 固定为 `scene, geometry_hash` 升序。

## 分阶段执行

### A. 资产与代码审计

输出 legacy 差异、模型/代码/hash、NCS 第一创新点缺失项。不得读取 GT、运行 AP 或模型 embedding。

### B. 第一创新点物化与 unique geometry ledger

必须覆盖 `100/100` scenes；每个 proposal point hash、point count、来源成员、canonical class/score和可见资产可追溯；duplicate geometry 输出为 0；所有 mutation 审计通过。该阶段无 GT、无 AP。

### C. Alpha/SAM manifest

只生成视角、bbox、三个 scale、SAM prompt、RGB/depth/pose/intrinsics 和 visible-ratio 绑定。要求全部 scene/node join 唯一，资产缺失和投影错误为 0；无视角 proposal 显式保留。无模型 embedding、无 GT、无 AP。

### D. GPU-only Alpha ledger

只有 A/B/C 通过才允许。CUDA 不可见时拒绝 CPU fallback。保存全部 198 类 cosine similarity、top1、SMS、valid、keep、view/scale feature provenance和模型/hash。禁止读取 GT或运行 AP。

### E. NCS-train100 固定评测

在 D 完成并独立审计后，才允许读取 train100 GT并运行预注册的六个系统/score-contract AP。必须报告 AP/AP50/AP25、head/common/tail、五折、class-agnostic AP/AR、来源保留率、过滤率、TP/FP与召回损失。

## validation60 推进门槛

以下条件同时满足才允许冻结方法并申请 validation60 唯一一次评测：

1. `details_unit_confidence` 下 Alpha-only 相对 control 主 AP严格为正，且五折至少 4 折主 AP为正；
2. unit-confidence 下 SMS 相对 Alpha-only 的 AP与AP50严格为正，AP25不低于 `-0.10` 个百分点，五折至少 4 折主 AP非负；
3. frozen-score 次结果中 Alpha+SMS 相对 control 主 AP严格为正；
4. SMS 后 class-agnostic AP50下降不超过 `0.50`、AP25下降不超过 `1.00` 个百分点；
5. 任一来源的删除率不超过 `70%`；被删除 proposal 的最大预测类占比不超过 `20%`，最大单场景占比不超过 `10%`；
6. 所有 provenance、unique geometry、SAM mask、198 类、SMS population、删除和 evaluator 审计错误为 0。

任一门槛失败即封闭 DM-SMS-1A：不扫描 `tau_SMS`、prompt、scale、视角数、SAM mask选择、缺失回退或分数合同，不读取 validation60。

## 明确不做

- 不复用旧 prediction-level SMS 结果作为正式结果；
- 不按同一 mask 的类别副本重复计算 SMS；
- 不使用投影点膨胀 mask冒充 SAM Alpha mask；
- 不对 validation60/val312 选择阈值、prompt或过滤率；
- 不下载新模型，不运行 Qwen3-VL，不恢复 NCS-4/5/6；
- 不修改第一创新点候选规则、模型、prior correction或分数公式。
