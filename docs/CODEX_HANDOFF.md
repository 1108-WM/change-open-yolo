# Codex Handoff

最后更新：2026-06-22

这个文件是给新的 Codex 会话或另一台设备的快速入口。完整实验细节见仓库根目录的 `CURRENT_EXPERIMENT_STATUS.md`。

## 先读顺序

1. `docs/CODEX_HANDOFF.md`：当前仓库状态、哪些东西没有上传、下一步该做什么。
2. `CURRENT_EXPERIMENT_STATUS.md`：完整实验脉络、关键指标、失败路线和下一轮建议。
3. `docs/*_log.md`：各条实验分支的补充记录。
4. `run_evaluation.py`、`utils/backprojection_fusion.py`、`tools/export_sam_fused_proposals.py`：当前主线实现入口。

## 当前研究主线

当前只推进第一个创新点：基于 YOLO-World 的三维候选补全和几何质量提升。

主流程：

```text
YOLO-World 二维检测
-> Segment Anything 二维掩码候选
-> 二维框/掩码反投影候选
-> 超点和多视角一致性过滤
-> 三维连通分量清理
-> 与 Mask3D 原始候选融合评估
```

当前最佳方向仍是候选补全加二维掩码级超点负约束。Alpha-CLIP、YOLOE、多掩码选择、自适应内部种子、简单视角质量门控和初版候选局部超点都已经验证过，但没有形成稳定净收益。

最新一轮已经接入多关系二维掩码证据图、基线缺口检测、图候选内部竞争和图候选分数控制。它们可以生成候选并进入最终评估，但还没有稳定提升当前主线。

## 当前结论

- 当前最稳结果仍是候选补全主线，不是 Alpha-CLIP、YOLOE、局部超点或简单层级规则。
- 当前最好四十八场景参考：`0.272610 / 0.345769 / 0.389491`。
- 当前最好九十六场景参考：`0.273030 / 0.340244 / 0.383687`。
- 多关系二维掩码证据图已经接入，边类型包括同物体支持、父子包含、冲突和弱关系。
- 严格证据图版本已经支持候选失败：点级投票不再默认退回完整并集，弱边不能桥接，冲突和组内一致性可以作为拒绝条件。
- 新增“基线未解释缺口”检测和 `graph_gap_seed_policy`：新增补漏候选默认只输出未被已有三维候选覆盖的缺口核心点。
- 新增图候选内部竞争：争夺同一批三维点的图候选会先在导出阶段竞争，低质量重复项写入 `prefilter_skipped`。
- 最新 GPU even48 结果显示，严格缺口核心加候选竞争后导出图候选 `53` 个，最终进入评估的多视角图候选 `16` 个，但结果仍为 `0.271186 / 0.345749 / 0.389509`。
- 继续给图候选加竞争优先级、把图候选竞争因子乘到最终分数、把图候选分数上限放宽到 `1.05`，结果仍然都是 `0.271186 / 0.345749 / 0.389509`。
- 打开图证据对主线候选的轻量重排，使用 `MASK_GRAPH_EVIDENCE_RESCORE=1` 和 `MASK_GRAPH_EVIDENCE_PRIORITY_WEIGHT=0.30`，结果仍为 `0.271186 / 0.345749 / 0.389509`。
- 因此当前不是“图候选生成失败”或“图候选分数不够高”，而是“图证据还没有强到能明确替换或剔除主线候选”。
- YOLOE 已放弃为当前主线模块。
- Alpha-CLIP 暂只保留为语义诊断信号，不作为当前最近一步的提分模块。
- 轻量几何判别器和层级特征有排序和诊断价值，但不适合继续做全局硬过滤。

下一步建议：

1. 不要继续只抬高证据图候选分数，也不要继续只做软重排；这些已经验证不改变 AP。
2. 下一步应做硬决策：当图候选和主线候选高度重叠时，明确执行“保留主线、替换为图候选、或两者都拒绝”，而不是并列输出。
3. 给主线候选显式保存图证据支持数、冲突数、支持视角数、缺口价值和候选竞争结果。
4. 对低图证据、高冲突、低缺口价值的主线新增候选做降权或剔除。
5. 对进入最终评估的 `16` 个多视角图候选做真阳性和假阳性诊断，先判断是类别错、边界脏、重复已有实例，还是召回了评价不认可的区域。
6. 若继续超点方向，应接入颜色、法向和二维实例边界支持；不要只靠三维坐标和固定半径。

## 已上传到 GitHub 的内容

已上传：

- 原始 OpenYOLO3D 代码。
- 当前修改后的 `run_evaluation.py`、`utils/`、`tools/`。
- 当前实验状态文档和日志文档。
- `.gitignore`，用于排除数据集、模型权重、输出和本地缓存。
- 环境快照：
  - `_backups/openyolo3d_environment_2026-05-22.yml`
  - `_backups/openyolo3d_conda_list_2026-05-22.txt`
  - `_backups/openyolo3d_pip_freeze_2026-05-22.txt`

没有上传：

- 数据集：`data/`、`OpenYOLO3D/`。
- 实验输出和缓存：`output/`、`models/YOLO-World/yolo_world/work_dirs/`。
- 模型权重：`pretrained/checkpoints/`、`pretrained/alpha_clip/`、`pretrained/clip-vit-base-patch32/`、`pretrained/yoloe/`、`*.pt`、`*.pth`、`*.ckpt`、`*.bin`。
- 外部下载仓库：`_external/`。
- 本地 Codex/agent 状态：`.codex/`、`.agents/`、`skills-lock.json`。
- 论文 PDF 和本地临时材料：`related papers/`、`cushion`、临时图片。

## 本机路径提示

当前工作目录：

```bash
/home/jia/wm_open-yolo/OpenYOLO3D
```

本机曾使用的 ScanNet200 数据路径：

```bash
/media/jia/软件1/OpenYOLO3D_datasets/scannet200
```

另一台设备需要重新准备数据、权重和输出缓存。不要期待 `git clone` 后能直接复现实验结果；仓库只同步代码和小型文档状态。

## 环境提示

原始安装说明见：

```text
docs/Installation.md
environment.yml
```

当前机器的环境快照见 `_backups/` 下三份文件。实际曾使用的环境名是：

```bash
openyolo3d
```

常用 Python：

```bash
/home/jia/anaconda3/envs/openyolo3d/bin/python
```

基础语法检查命令：

```bash
/home/jia/anaconda3/envs/openyolo3d/bin/python -m py_compile \
  run_evaluation.py \
  utils/backprojection_fusion.py \
  utils/__init__.py \
  utils/utils_2d.py \
  tools/export_mask_graph_proposals.py \
  tools/filter_backprojection_candidates_clip.py \
  tools/run_gemini_backprojection_verifier.py \
  tools/evaluate_multiview_object_clip_correction.py \
  tools/evaluate_semantic_fusion_head.py \
  tools/search_alphaclip_thresholds.py
```

当前证据图 even48 GPU 评估入口：

```bash
OPENYOLO3D_ALLOW_LEGACY_2D_CACHE=1 \
OUT_DIR=/home/jia/wm_open-yolo/OpenYOLO3D/output/scannet200/subset_sweeps/even48_mask_graph_gap_compete_gpu \
PATH_TO_2D_PREDS=/home/jia/wm_open-yolo/OpenYOLO3D/output/scannet200/bboxes_2d \
MASK_GRAPH_OUT=/home/jia/wm_open-yolo/OpenYOLO3D/output/mask_graph_proposals_scannet200_even48_gap_compete_gpu \
MODE=graph_refill \
bash tools/run_scannet200_even48_mask_graph_eval.sh
```

## 重要文件

核心执行：

- `run_evaluation.py`
- `utils/__init__.py`
- `utils/backprojection_fusion.py`

候选导出和诊断：

- `tools/export_backprojection_candidates.py`
- `tools/export_sam_fused_proposals.py`
- `tools/export_mask_graph_proposals.py`
- `tools/run_scannet200_even48_mask_graph_eval.sh`
- `tools/analyze_backprojection_candidates.py`
- `tools/train_candidate_geometry_discriminator.py`

Alpha-CLIP 相关实验：

- `tools/export_multiview_object_clip_features.py`
- `tools/evaluate_multiview_object_clip_correction.py`
- `docs/AlphaCLIP_object_rescore_log.md`

历史实验脚本：

- `tools/run_scannet200_even48_containment_sweep.sh`
- `tools/run_scannet200_even48_sam_mask_support_eval.sh`
- `tools/run_scannet200_even48_seed_merge_policy_eval.sh`
- `tools/run_scannet200_even48_selectable_sam_refine.sh`
- `tools/run_scannet200_96_cc_cleanup_confirm.sh`

## Git 使用

当前远端：

```bash
origin   git@github.com:1108-WM/change-open-yolo.git
upstream https://github.com/aminebdj/OpenYOLO3D.git
```

另一台设备建议使用：

```bash
git clone git@github.com:1108-WM/change-open-yolo.git
```

已有仓库时同步：

```bash
git pull
```

本机继续工作后同步：

```bash
git status
git add <changed source/docs files>
git commit -m "Describe the change"
git push
```

注意：本机当前可能存在未提交的编译产物改动，主要在 `models/Mask3D/.../build` 和 `pointnet2` 编译目录下。这些不应作为项目状态上传。
