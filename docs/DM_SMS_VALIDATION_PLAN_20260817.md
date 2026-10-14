# DM-SMS-1：Details Matter Alpha-CLIP 主分类与 SMS filtering 验证计划（草案）

状态：`design_draft_not_frozen`

日期：2026-08-17

本文件记录当前推荐的第二创新点验证路线。它还不是已经冻结的预注册合同，不授权读取 `NCS-validation60` GT、在 validation/val312 上扫阈值，或直接启动正式 GPU/AP。下一步应先完成代码与资产合同审计，再把本草案收敛为独立预注册文件。

## 1. 当前裁决

- `ScanNet200 val312` 不是不用，也不是废弃；它从现在开始只作为最终完整论文评测集。
- 当前方法开发只允许使用新的 `NCS-train100`。它不是旧 `official100`。
- `NCS-validation60` 继续冻结，只允许方法完全冻结后做唯一一次迁移评测。
- 推荐顺序固定为：

```text
NCS-train100 开发／五折诊断
→ 冻结完整方法、阈值、prompt、模型和推理合同
→ NCS-validation60 唯一一次迁移评测
→ 若迁移结论允许，ScanNet200 val312 最终一次完整论文评测
```

- val312 现在禁止用于方法设计、选择 SMS 阈值、修改 prompt、筛选类别、删除失败动作或按 AP 回调方案。

## 2. 为什么先验证 Details Matter 语义合同

第一创新点在完整 val312 上的类别无关结果为：

| 阶段 | AP | AP50 | AP25 |
|---|---:|---:|---:|
| 原始 native | 50.606943 | 67.975380 | 77.977661 |
| + 轨迹伤害抑制 | 52.200359 | 70.182853 | 79.642081 |
| + pair-union | **52.396219** | **70.299971** | **79.734125** |

Details Matter 表 4 在完整 ScanNet200 validation 上报告：

| proposal 来源 | AP | AP50 | AP25 |
|---|---:|---:|---:|
| Ours (2D Only) | 33.3 | 51.9 | 66.1 |
| Ours (2D + 3D) | 46.6 | 59.0 | 64.4 |
| 表中 Mask3D (fully supervised) | 50.6 | 68.0 | 76.9 |

可比性结论：

- 两者都是完整 ScanNet200 val312 上的类别无关 3D instance proposal AP，评测目标高度可比。
- 本项目 native 的 AP/AP50 几乎精确复现论文表中的 Mask3D 行；AP25 相差约 `+1.08`，所以只能写“高度可比”，不能未经 evaluator 逐项审计就声称完全同合同。
- 本项目最终相对 Details Matter 的 2D+3D proposal：AP `+5.796219`、AP50 `+11.299971`、AP25 `+15.334125`。
- 项目尚无同合同 AR，不能比较论文中的 AR。
- 因此当前主要瓶颈已不是 category-agnostic geometry，而是开放词汇分类与最终排序。

## 3. Details Matter 的 Alpha-CLIP 主分类合同

对第 `k` 个 3D proposal，在高可见视角集合 `V_k` 和多尺度集合 `L` 上提取 Alpha-CLIP 特征。论文的 proposal feature 为可见率加权的多视图、多尺度聚合：

\[
\mathbf F_k=\operatorname{Norm}\left(
\sum_{v\in\mathcal V_k}\sum_{l\in\mathcal L}
\alpha_{v,k}\mathbf f^{l}_{v,k}
\right),
\]

其中：

\[
\alpha_{v,k}=\frac{\text{proposal 在视角 }v\text{ 中的可见 3D 点数}}
{\text{proposal 的全部 3D 点数}}.
\]

给定 `K` 个 proposals 和 `C` 个文本查询，计算余弦相似度矩阵：

\[
\mathbf L\in[-1,1]^{K\times C},\qquad
L_{k,c}=\cos(\mathbf F_k,\mathbf T_c).
\]

论文与补充材料给出的 ScanNet200 实现要点：

- 每个 proposal 选择 top-20 可见图像；
- 3 个 scale levels，expansion ratio 为 `0.2`；
- 使用 Alpha-CLIP 作为完整主分类器，而不是只作为 YOLO/native top-1 的辅助证据；
- ScanNet200/Replica 使用 projected instance bounding box 查询 SAM，得到 Alpha mask；
- 论文正文使用按 visible ratio 聚合；
- 项目复现前应核验并冻结文本 prompt，当前论文记录为 `a blurry photo of {CLASS_NAME} in a room.`；
- 对完整 198 个有效 ScanNet200 instance prompts 分类，不伪造不存在的两个 prompt。

## 4. Details Matter 的 SMS 公式

SMS 不是泛指 z-score 融合，而是真正的 proposal filtering。对场景中的 `K` 个 unique proposals 和类别 `c`：

\[
\mu_c=\frac{1}{K}\sum_{k=1}^{K}L_{k,c},
\]

\[
\sigma_c^2=\frac{1}{K}\sum_{k=1}^{K}(L_{k,c}-\mu_c)^2,
\]

对 proposal `k`：

\[
c_{\max}=\arg\max_c L_{k,c},
\]

\[
SMS_k=\frac{L_{k,c_{\max}}-\mu_{c_{\max}}}
{\sigma_{c_{\max}}}.
\]

最终过滤规则：

\[
SMS_k<\tau_{SMS}\quad\Longrightarrow\quad
\text{从最终 predictions 中删除 proposal }k.
\]

补充材料说明：

- `τSMS` 按实验设置定制；
- 作者称 `[-1,1]` 是较稳定的合理范围；
- 阈值稳定性图主要基于 Replica，不能据此直接宣布一个通用的 ScanNet200 最优阈值；
- 因此不能在 val312 上扫描或选择 `τSMS`。

## 5. 表 7 与本项目历史实验的区别

Details Matter 表 7：

| 分类配置 | AP | AP50 | AP25 |
|---|---:|---:|---:|
| proposals + CLIP | 27.5 | 34.7 | 38.2 |
| + Alpha-CLIP | 30.5 | 37.6 | 41.1 |
| + SMS filtering | 32.7 | 41.4 | 45.3 |

即 Alpha-CLIP 相对 CLIP `+3.0 AP`，SMS 再增加 `+2.2 AP`，累计 `+5.2 AP`。

本项目此前没有复现这一完整合同：

- Z2/Z3/NCS 分支主要把 Alpha 当作辅助软证据或已有 representative 内的 score-swap 证据；
- 没有使用完整的 top20 × 3-scale × 0.2 expansion 主分类合同；
- 不允许 Alpha 直接成为所有 proposals 的主分类器；
- NCS-5 的 SMS-inspired 规则按 scene/role/class 做标准化，但没有 `τSMS`，不删除 prediction，只在同一 geometry node 的已有代表间提议守恒换分；
- 因此 NCS-5 失败不能否定论文的 SMS filtering。

历史证据也说明 Alpha 本身并非完全无效：official100 上 Z3 的 OOF control 为 `32.755740/40.298027/43.769243`，相对 frozen Alpha control 主 AP约 `+2.444070`，相对原始 native-only 主 AP约 `+4.823585`；NCS-6 中 Alpha 也产生 `300` 个 strict-improve nodes，是最佳单一来源。问题主要是迁移稳定性和决策合同，而不是 Alpha 没有信号。

## 6. 推荐实验：DM-SMS-1A（训练自由）

为了先回答论文方法在本项目底座上是否成立，推荐第一版保持训练自由：固定 `τSMS=0`，不在任何集合上选阈值。若以后要做 `{-1,-0.5,0,0.5,1}` 的五折选择，必须另写新的预注册分支，不能把 DM-SMS-1A 的结果回调后补扫。

### 6.1 先物化第一创新点的 NCS-train100 几何

当前 NCS 资产主要是 native-only。正式结合第一创新点前，必须在 `NCS-train100` 上无 GT 物化并审计：

1. 冻结多视图 tracks；
2. 轨迹伤害抑制分数；
3. append-only pair-union；
4. native exact-geometry 折叠；
5. native/track/pair-union 的 unique geometry proposal ledger；
6. 每个 proposal 的 point hash、来源、分数和可见视角 provenance。

SMS 的统计单位必须是 unique geometry proposal。禁止在同一 mask 的 600 个 native 类别副本上计算均值和方差，否则类别副本会人为改变 scene population statistics。

### 6.2 代码与资产审计

仓库已有 `tools/evaluate_multiview_object_clip_correction.py`，包含：

```text
--clip_filter_policy sms
--clip_sms_scope
--clip_sms_threshold
```

但旧实现不能直接视为正式复现。必须逐项审计：

- 是否先折叠为 unique geometry proposals；
- `μc/σc` 的 scene population 是否与论文一致；
- `cmax` 是否来自 Alpha 完整 198 类主分类；
- 是否真的从 prediction set 删除低 SMS proposal；
- top20、3-scale、0.2 expansion、visible-ratio、SAM Alpha mask 和 prompt 是否一致；
- 分数与 evaluator 如何处理，尤其论文补充材料提到 AP/AR evaluation 使用 confidence `1.0`，本项目必须在读取 GT 前审计其与现有 evaluator 的兼容性并冻结。

### 6.3 必须比较的三个系统

在完全相同的第一创新点 unique geometry proposals 上：

1. `frozen_semantic_control`：冻结当前语义基线；
2. `details_alpha_main_no_sms`：Details 合同的 Alpha-CLIP 主分类，不做 SMS；
3. `details_alpha_main_sms_tau0`：同一主分类结果，加真正的 proposal filtering，固定 `τSMS=0`。

必须保存每个 proposal 的完整 198 类相似度、top1、SMS、是否过滤、来源和 geometry hash，使三组只改变预注册允许的语义/过滤部分。

### 6.4 报告项目

- open-vocabulary AP/AP50/AP25；
- head/common/tail AP；
- class-agnostic AP/AP50/AP25，以及同合同 AR（若 evaluator 可严格复现）；
- native/track/pair-union 的保留率；
- 被 SMS 删除的 proposal 数、类别集中度和场景分布；
- 过滤前后 TP/FP、召回损失和高 IoU proposal 损失；
- 五个冻结场景折的方向，不只报告全量平均。

冻结预注册时应明确推进门槛。推荐至少要求：Alpha 主分类相对 control 的主 AP 方向稳定；SMS 相对 Alpha-only 的 AP/AP50/AP25 不出现只靠单一场景或单一类别造成的假增益；class-agnostic recall/AR 不发生不可接受的下降。具体数值门槛必须在第一次正式 AP 前写死。

## 7. val312 已有完整结果如何解释

远程 val312 开放词汇完整联合流程在修正 cache 合同后为：

| 配置 | AP/mAP | AP50 | AP25 |
|---|---:|---:|---:|
| 修正后联合控制 | 25.4155 | 34.8005 | 40.8422 |
| 全部策略联合 + Qwen | 25.4147 | 34.7983 | 40.8479 |

此前 `7.996%` 来自错误的类别无关占位 cache：`output/scannet200_first_innovation/eval_native_cache`，不得引用。正确输入是完整的 OpenYOLO3D 语义 cache：`output/scannet200_first_innovation/frozen_champion/native_cache_mask3d_yoloworld`，每场景 600 个 native 语义候选。

当前证据不能支持“仍有一个明显运行错误导致 25.4”。更合理的结论是：official/train 开发收益没有稳定迁移到完整 val312，Qwen 只接受 16 个修改且总体近乎中性。此结果现在只作为问题定位和最终基线记录，禁止用来选择 DM-SMS 规则。

## 8. 联网论文调研后的后续创新方向

DM-SMS-1 是应先完成的强控制。若它成立或完成后仍需真正的新第二创新点，推荐：

```text
NCS-7：几何落地的场景上下文属性仲裁
```

核心合同：第一创新点 geometry 完全冻结；在完整 RGB 场景中高亮目标并加入无类别 3D 空间关系。MLLM 第一阶段看不到候选标签，只提取形状、材质、部件、用途和放置关系；第二阶段才在已有 YOLO/Alpha/SigLIP/Proxy/native candidates 中仲裁。只有 alternative 唯一 SUPPORT、incumbent CONTRADICT、候选正反顺序一致时才守恒换分，其余严格 no-op。不创建类别、不改 mask/membership、score multiset 守恒。

论文依据与结论：

- OV3D-CG：仅给 object patch 为 `15.5 AP`，完整场景中 SAM 高亮目标为 `22.7 AP`，说明场景上下文必须与目标落地同时提供；
- OpenTrack3D：CLIP 为 `22.9 AP`，Qwen3-VL MLLM 为 `26.0 AP`，支持用 MLLM 做语义理解，但不支持无门控全量改类；
- Group3D：语义兼容 grouping 优于无类别或严格同名 grouping，支持把空间关系和语义兼容作为仲裁证据；
- PTC：训练自由 prototype-guided text calibration 对 ProxyCLIP 平均约 `+1.9 mIoU`，但参数具有数据集依赖，只列为辅助校准方向；
- AttrSeg：属性拆解有助于细粒度类别区分，适合处理 `office chair/chair`、`desk/table` 等粒度混淆。

若未来启动 NCS-7，正式候选模型优先审计 `Qwen/Qwen3-VL-4B-Instruct`。现有 Qwen2.5-VL-7B 的 Z6f 是标签优先 A/B + 三张 crop，基本中性，不等价于“完整场景、属性优先、标签后置”的新方案。

## 9. 新对话应立即执行的工作

1. 完整阅读本文件、`CURRENT_EXPERIMENT_STATUS.md`、`资料/当前基线修改方向.md` 和 `资料/论文阅读记录.md`。
2. 只读审计 `tools/evaluate_multiview_object_clip_correction.py` 的 SMS、视角、Alpha mask、prompt、proposal population 和 score/evaluator 合同。
3. 盘点 NCS-train100 是否已有第一创新点 tracks、damage suppression 和 pair-union；当前结论是尚未物化完整第一点 NCS 资产。
4. 根据审计结果编写 `DM-SMS-1A` 的正式预注册合同和资产 provenance；在合同冻结前不运行 GT/AP。
5. 先实现 unique geometry proposal ledger 和无 GT manifest/audit，再决定是否需要 GPU embedding。

当前不需要下载新模型。Details Matter 验证优先复用项目已有 Alpha-CLIP、SAM、ScanNet RGB-D 和 NCS 数据。Qwen3-VL 仅属于尚未授权的 NCS-7 后续方向，现在不要下载。
