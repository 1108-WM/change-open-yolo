# Codex Handoff

最后更新：2026-06-15

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

## 当前结论

- YOLOE 已放弃为当前主线模块。
- Alpha-CLIP 暂只保留为语义诊断信号，不作为当前最近一步的提分模块。
- 轻量几何判别器有排序和诊断价值，但不适合全局硬过滤。
- 当前最值得继续的是候选包含和重叠关系处理。
- 第二优先级才是重新做更强的候选局部超点，但必须接入颜色、法向和二维掩码支持，不能只靠三维坐标。

下一步建议：

1. 先实现候选包含/重叠后处理，不改候选生成。
2. 用已有诊断特征给大脏候选降权，避免覆盖小干净候选。
3. 在 `even48` 上验证，只有明确正向再进 `even96`。

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
  tools/export_sam_fused_proposals.py \
  tools/analyze_backprojection_candidates.py
```

## 重要文件

核心执行：

- `run_evaluation.py`
- `utils/__init__.py`
- `utils/backprojection_fusion.py`

候选导出和诊断：

- `tools/export_backprojection_candidates.py`
- `tools/export_sam_fused_proposals.py`
- `tools/analyze_backprojection_candidates.py`
- `tools/train_candidate_geometry_discriminator.py`

Alpha-CLIP 相关实验：

- `tools/export_multiview_object_clip_features.py`
- `tools/evaluate_multiview_object_clip_correction.py`
- `docs/AlphaCLIP_object_rescore_log.md`

历史实验脚本：

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
