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

最新一轮已经完成固定评分口径对照、实际进入评估的证据图候选诊断，并把证据图候选改成同时保存完整核心和缺口核心。结论是：只输出缺口核心会产生物体残片；完整核心能基本消除主要伤害；但证据图候选仍不能简单作为新增实例追加，下一步必须做“新增、补全、替换、拒绝”的动作判断。

## 当前结论

- 当前最稳结果仍是候选补全主线，不是 Alpha-CLIP、YOLOE、局部超点或简单层级规则。
- 当前最好四十八场景参考：`0.272610 / 0.345769 / 0.389491`。
- 当前最好九十六场景参考：`0.273030 / 0.340244 / 0.383687`。
- 多关系二维掩码证据图已经接入，边类型包括同物体支持、父子包含、冲突和弱关系；点级投票默认不再失败后恢复完整并集。
- 当前证据图候选同时保存两个点集：
  - 完整核心：`full_core_seed_points_path`
  - 缺口核心：`gap_core_seed_points_path`
- even48 六组评分口径对照已跑完：
  - 旧缺口核心版本：候选自身分数从 `0.303107 / 0.396944 / 0.443739` 降到 `0.296883 / 0.389690 / 0.436725`；归一化分数从 `0.302818 / 0.396449 / 0.443777` 降到 `0.287448 / 0.378442 / 0.424286`。
  - 完整核心版本：候选自身分数基本不变，`0.303101 / 0.396934 / 0.443723`；归一化分数降到 `0.294147 / 0.387372 / 0.437535`。
  - 完整核心加严格现有覆盖过滤：候选自身分数仍基本不变，`0.303101 / 0.396934 / 0.443723`；归一化分数改善到 `0.298840 / 0.392579 / 0.441919`。
- 实际进入评估的证据图候选诊断：
  - 旧缺口核心：`16` 个进入评估，`1` 个完整漏检物体、`12` 个背景污染或几何错误、`3` 个多物体错误合并。
  - 完整核心：`8` 个进入评估，`1` 个完整漏检物体、`2` 个背景污染或几何错误、`4` 个多物体错误合并、`1` 个重复候选但真实重叠低。
  - 完整核心加严格现有覆盖过滤：`5` 个进入评估，`1` 个完整漏检物体、`2` 个背景污染或几何错误、`2` 个多物体错误合并。
- 因此当前结论不是“证据图无效”，而是：
  - 旧缺口核心会产生残片。
  - 完整核心能解决残片伤害。
  - 严格现有覆盖过滤能减少重复和归一化分数下降。
  - 剩余问题主要是多物体错误合并和平面/大面积类别污染。
- YOLOE 已放弃为当前主线模块。
- Alpha-CLIP 暂只保留为语义诊断信号，不作为当前最近一步的提分模块。
- 轻量几何判别器和层级特征有排序和诊断价值，但不适合继续做全局硬过滤。

下一步建议：

1. 不要继续只调证据图候选分数；不要回到只输出缺口核心。
2. 保持“完整核心输出 + 同时保存缺口核心”。
3. 增加硬拒绝规则：
   - 多物体错误合并风险高则拒绝。
   - 图共识低且同物体强边少则拒绝。
   - 完整核心与多个三维区域相交时拒绝或拆分。
   - `mat`、`rug`、`whiteboard`、`poster` 等平面或大面积易污染类别需要单独限制。
4. 开始实现候选动作判断：
   - 完整且属于漏检物体：新增。
   - 只是已有物体局部缺口：补全已有候选，不新增。
   - 明显优于已有候选：替换。
   - 重复、背景污染、多物体合并：拒绝。
5. 下一轮先在 even48 跑“完整核心严格过滤 + 平面/多物体错误合并拒绝”；只有候选自身分数和归一化分数同时不下降，再扩到 even96。

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

当前证据图 even48 六组评分口径评估入口：

```bash
bash tools/run_scannet200_even48_mask_graph_score_modes.sh
```

严格现有覆盖过滤版本：

```bash
OUT_DIR=/home/jia/wm_open-yolo/OpenYOLO3D/output/scannet200/subset_sweeps/even48_mask_graph_score_modes_full_core_strict_existing \
MASK_GRAPH_OUT=/home/jia/wm_open-yolo/OpenYOLO3D/output/mask_graph_proposals_scannet200_even48_full_core_strict_existing_gpu \
bash tools/run_scannet200_even48_mask_graph_score_modes.sh
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
