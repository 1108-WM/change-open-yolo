# FI1 → 完整 Alpha：固定场景实测与归因

日期：2026-09-10。本轮已完成真实 GPU 导出、无 GT 预测冻结与审计、固定场景 AP 和输入对照。这里只评估 official100 中此前固定的 `scene0027_01`，不是 official100/val312 总体新结果，也不是 Details Matter 全方法复现。

## 1. 本轮回答了什么

**统一证据入口可以真实运行，但本次没有观察到新增几何带来的 AP 增益。这个场景的 refined 实际是几何退化例，不能用它证明“几何变好但 Alpha 没有提升”。**

更具体的发现是：父 union、已有 native/track 和 refined 都识别为同一块白板；已有 track 早已在最高分处正确覆盖它。因此新增或替换冗余预测没有改变 AP。FI1 refined 的原最终排序分数为 0，当前 Alpha-only 对照没有使用这个信号，仍把其白板预测排到全场景第 4。这进一步说明，接通图像编码不等于接通 FI1 的全部筛选与排序收益。

## 2. 固定比较，避免混合归因

模型、视角、输出与比较合同在本轮读取 GT 前已写入：

- [视觉与预测合同](experiment_summaries/fi1_alpha_transfer_20260910/contract.json)
- [三组比较合同](experiment_summaries/fi1_alpha_transfer_20260910/comparison_contract.json)
- [读取 GT 前的预测审计](experiment_summaries/fi1_alpha_transfer_20260910/prediction_audit_before_gt.json)

所有系统使用完整 Alpha：最多 20 个可见视角，3 个尺度 0/0.2/0.4，bbox 提示 SAM ViT-B、最高预测 IoU mask，归一化特征按可见率聚合，再与 198 类文本特征计算 cosine。prompt 固定为 `a blurry photo of {CLASS_NAME} in a room.`。不做 SMS，不训练，不用 Qwen，不搜索参数。

AP 使用每场景全局最多 600 个 mask×class 预测，分数固定为 `(raw cosine+1)/2`，同分按几何 key 和 class index 排序；少于 100 点的 mask 排除。没有逐 mask 强制 top-1。Alpha 缺失会显式记录并排除，不继承父分数。三组都输出 600 列：

| 系统 | 定义 | AP | AP50 | AP25 |
|---|---|---:|---:|---:|
| A_existing_alpha | 68 个 B1 既有唯一几何，自己的完整 Alpha | 58.547039 | 63.545525 | 65.317460 |
| B_expanded_alpha | A 的几何 + 1 个新 refined，各自完整 Alpha | 58.547039 | 63.545525 | 65.317460 |
| C_fixed_semantics_mask_swap | A 选列后，仅将父 union 的 7 列 mask 换为 refined，类别/分数/顺序不变 | 58.547039 | 63.545525 | 65.317460 |

数值为百分数；三份逐类 CSV 完全一致，宏平均独立重算最大误差约 `1.11e-16`。单场景只对其中 9 个有效 GT 类求宏平均，不能与论文、official100 或 val312 AP 横向比较。

B 新增 refined 的 10 个类别预测，同时挤出 A 的 10 列，因此 B−A 同时包含候选变化和预算竞争。C 是受控替换诊断。**B、C 都不是生产 B2 的 append-only 合同；本轮没有改写历史 B0/B1/B2，也没有启用生产评分。**

## 3. 真正发生在这块白板上的变化

共同目标是 GT instance `52012`，class index 43 / whiteboard；父和子只与这一有效 GT 实例相交。

| 几何来源 | 点数 | 同一白板 IoU | Alpha 白板排名 | 白板最终分数 | A/B 中的作用 |
|---|---:|---:|---:|---:|---|
| 已有 track | 2727 | 0.924307 | 1 | 0.623317428 | A/B/C 的全场景最高分预测，足以通过全部主 AP IoU 档 |
| 父 pair_union | 3134 | 0.842988 | 1 | 0.623297960 | 对同一目标的较低分重复预测 |
| 已有 native | 2601 | 0.967634 | 1 | 0.623121090 | 对同一目标的另一正确预测 |
| 新 refined | 1602 | 0.358025 | 1 | 0.622372650 | B 中全场景第 4 列；不通过 IoU>0.5 |

refined 从父掩码中删掉 1532 点，没有新增点；共同支持区为 805 点。局部 IoU 变化为 **−0.484964**，并不是本场景几何提升。固定父类别时，IoU>0.5 和 >0.75 的局部匹配由成功变失败。

三组白板 AP/AP50/AP25 都为 1。已有 track 的正确高分预测完整保留；父/子都是其后的重复或错误预测，所以这次局部退化没有改变白板 AP。其它被预算挤出的预测也没有改变任何逐类 CSV。不能由此推断增加冗余预测总是无害。

**FI1 原始 refined 最终分数是 0，但 Alpha 自身对白板语义仍然有把握。** 语义正确不能证明 mask 完整、没有过度裁剪或具有新增检测价值；FI1 的最终边际分数也不能直接当作独立几何概率乘给所有类别。本次故意分离这两个信号，揭示了后续逐类评分需要解决的问题。

## 4. 几何变化是否真正到达了 Alpha

父 union 有 20 个有效 Alpha 视角，refined 有 13 个；其中 12 帧相同。

- 5/12 帧的 bbox、SAM mask、3 尺度 crop、RGB/alpha 图像和视角特征完全相同。
- 7/12 帧的输入发生变化；父/子的总聚合 cosine 不同。
- 白板 cosine 从 0.246596 到 0.244745，排名仍为 1。

这不是“所有几何变化都被 SAM 抹掉”的证据。部分帧确实坍缩为相同输入，另一些帧确实改变；且当前目标已识别正确，不需要换类别才能获益。

![父子同帧裁剪与 SAM 区域](https://github.com/1108-WM/change-open-yolo/blob/main/docs/EXTERNAL_ASSETS.md)

图中是模型预处理前的尺度 0，第一行按共同帧 ID 顺序选择第一个相同 bbox 的帧，第二行选择第一个不同 bbox 的帧，没有根据 GT 或 AP 选择“好看”的案例。原始三个尺度的图像、SAM mask、实际预处理后前景数量与特征均保存在证据目录。

## 5. 全场景证据与多语义覆盖

统一入口导出 69 个几何（68 既有 + 1 新增）、652 个原 B1 成员（600 native、38 track、14 union）、`69×198=13662` 行特征。652 个 B1 成员逐字节恢复。全部 69 个几何都有 Alpha、原始低粒度标签图和框内比例证据。

1257 个候选 Alpha 视角中 1254 个有效：1 个 SAM mask 为空，2 个在实际 mask_transform 后前景消失，均显式排除。5 个几何少于 100 点，只在本次 AP 输出时排除，原证据和 B1 成员仍保存。

在 IoU 严格 >0.5 的 26 个几何—目标匹配中：

- Alpha 正确类严格排名第 1：15/26；正确类排名 ≤5：26/26。
- 原标签图对应数：17/26、22/26。
- 框内比例对应数：15/26、22/26。

这些是重复几何也计入的局部类别排名诊断，不是独立样本准确率、最终 top600 的召回率或 AP，也没有据此改变 k。它们支持保留完整类别分布和相互独立的 YOLO/Alpha 证据，不能说明单个分支普遍更强。

本次 AP 分数只来自 Alpha，这是读取 GT 前固定的隔离对照；**原标签图已实际导出，但尚未参与这三组分数融合**。不能把这一实验说成最终的 YOLO+Alpha+FI1 第二点。

## 6. 对 Details Matter 与下一步的影响

Details 的候选/轨迹与 Alpha 可同时贡献收益：候选覆盖、边界、视觉识别、过滤和排序各自影响 AP。其 CLIP 27.5→Alpha 30.5 是分类配置消融，不能推出每一个 refined 都会改善分类；本例也不能否定它的整体方法。

对本项目，应继续区分两条收益路径：

1. **类别保持正确时，mask 变好并进入有效排序即可提高实例 AP。** 不需要要求 Alpha 每次都改变排名。
2. **原类别证据不可靠时，用新的目标视角、原标签图、Alpha 或 Qwen 独立识别补充正确类别，再解决同类质量与重复预测的排序。**

下一轮应先冻结有代表性的候选清单，覆盖全部来源、几何改善/退化、语义正确/错误和同类重复；样本量按场景与独立几何算，不能用 22×198 行假装扩充样本。已有受监督几何/质量/筛选必须核验外层场景隔离。

随后在同一批候选上验证完整标签图、Alpha 的互补与 FI1 独立质量/同类重叠上下文的增量。对当前 26 个可匹配几何中 Alpha 严格第一名不正确的 11 个案例，可以先分析现有证据是否保留正确类；不要因为本例白板已经识别正确就把下一轮变成增加 Qwen 调用。Qwen 完整词表独立识别继续作为计划中的独立证据分支，不把它当作 mask 完整性评分器。

现有 68 条 FI1 非追加计划里，control→challenger 排名 34 上升、29 下降、5 不变；它们本次没有参与 AP。**目前只验证了几何进入视觉入口和受控候选比较，没有验证 FI1 A/B/D 全部排序收益的语义迁移。** 本轮不据单场景结果调整 K、阈值、prompt 或融合权重，不扩展到 validation60/val312。

## 7. 代码、检查与产物

实际代码安装到 `/home/jia/Wm/wm_open-yolo/worktrees/OpenYOLO3D-geometry-semantic-56b2c25`：

- `tools/build_fi1_alpha_transfer_comparison.py`：无 GT 构建 A/B/C、实际点集父子绑定、FI1 排名诊断。
- `tools/evaluate_fi1_alpha_transfer_gt.py`：独立重建全类别 top600，核验父子点集、视觉比较与源排名；只允许固定场景读取 GT；调用未修改的原 evaluator，独立重算 CSV。
- `tests/test_fi1_alpha_transfer.py`：19 项新测试，覆盖预算竞争、多类别保留、固定语义替换、同目标归因、同分排名、源点集/视觉身份错误、预测篡改及 CSV 类别身份。

新测试和相关统一证据、几何变化、逐类诊断回归合计 **61 passed**。本次比较先用可写 staging 中的代码运行，8 个直接依赖与实际工作树逐文件相同；安装使用相同已验证文件，再核验安装哈希、实际工作树测试和读取 GT 前的预测审计。未提交/推送，原 evaluator 未修改。

- [真实视觉证据审计](experiment_summaries/fi1_alpha_transfer_20260910/smoke_evidence_audit.json)
- [冻结预测清单](experiment_summaries/fi1_alpha_transfer_20260910/comparison/summary.json)
- [GT/AP 结果](experiment_summaries/fi1_alpha_transfer_20260910/gt_diagnostic/summary.json)
- [父子全部目标及同帧比较](https://github.com/1108-WM/change-open-yolo/blob/main/docs/EXTERNAL_ASSETS.md)
- [结果解释与同目标竞争几何](experiment_summaries/fi1_alpha_transfer_20260910/result_interpretation.json)

审计核验文件、真实点集、预测选择与聚合，不独立重跑相机投影或视觉模型，也不宣称历史 v1 B1 来源完成了 v2/OOF 训练审计。生产 B2 仍禁用。
