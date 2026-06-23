# Codex Handoff

最后更新：2026-06-23

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

最新一轮已经完成证据图关系规则和约束式实例假设的第一版重构。结论是：只输出缺口核心会产生物体残片；完整核心能基本消除主要伤害；但证据图候选仍不能简单作为新增实例追加。当前优先级是先验证新版关系诊断和候选上界，再决定是否进入最终 AP。

2026-06-23 代码审计修复补充：针对提交 `e8c6578` 后的审计问题，已修复证据图假设构建中的提前淘汰和独立支持定义问题。当前种子筛选失败不再永久删除观测；独立强支持不再受共同 Mask3D 参照排斥；歧义节点会在当前连通区域内保持暂缓；同帧深度冲突优先于父子包含；可靠 Mask3D 解释默认要求分数 `0.30` 和种子覆盖 `0.50`，类别或分数信息缺失时不回退到普通覆盖。本轮只做源码和状态文档修复，没有运行 even48、even96 或最终 AP。

2026-06-23 运行与划分修复补充：暂缓节点现在记录依赖的假设，若当前假设失败会释放仅由该失败假设造成的暂缓节点，避免永久阻塞。`tools/run_scannet200_even48_mask_graph_eval.sh` 默认改为 `EXPORT_REUSE_EXISTING=0`，默认新输出目录为 `output/mask_graph_proposals_scannet200_even48_constrained_audit_fix_v2`；显式开启复用时，会校验新增关系阈值、假设参数、可靠 Mask3D 分数/覆盖门槛和 `export_code_version`。

## 当前结论

- 当前最稳结果仍是候选补全主线，不是 Alpha-CLIP、YOLOE、局部超点或简单层级规则。
- 当前最好四十八场景参考：`0.272610 / 0.345769 / 0.389491`。
- 当前最好九十六场景参考：`0.273030 / 0.340244 / 0.383687`。
- 多关系二维掩码证据图已经接入，边类型包括同物体支持、父子包含、冲突和弱关系；点级投票默认不再失败后恢复完整并集。
- 2026-06-23 证据图关系规则已改成：
  - 独立强支持和 Mask3D 参照辅助支持分开统计。
  - 共同 Mask3D 参照不能单独形成强支持。
  - 按需计算真实深度误差统计，用于深度一致和深度冲突。
  - 类别不一致默认不确定，只有伴随负证据才成为硬冲突。
  - 欠分割桥梁至少需要两类证据触发。
  - 原强支持连通区域内部重新划分，低质量或歧义观测可以保持未分配。
  - 漏检区域使用可靠 Mask3D 解释，不再使用任意 Mask3D 覆盖。
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
  - 上一版支持边正确率被共同 Mask3D 参照明显支配；新版诊断必须分开报告有无共同参照、支持类型和深度冲突区间。
- YOLOE 已放弃为当前主线模块。
- Alpha-CLIP 暂只保留为语义诊断信号，不作为当前最近一步的提分模块。
- 轻量几何判别器和层级特征有排序和诊断价值，但不适合继续做全局硬过滤。

下一步建议：

1. 暂时不要跑最终 AP，也不要扩 even96。
2. 先做少量场景新版导出，观察关系和假设统计是否正常；默认脚本已关闭旧导出复用，输出到新的 `mask_graph_proposals_scannet200_even48_constrained_audit_fix_v2`。
3. even48 诊断只比较：
   - 关系准确率。
   - 原强支持连通区域拆分数。
   - 新候选质量。
   - 去重后的 Mask3D 补全收益。
4. 只比较三档参数：保守档、推荐档、诊断放宽档，不做大规模搜索。
5. 如果推荐档在 even48 上改善候选质量和补全上界，再考虑最终 AP。
6. 第一阶段仍不做自动修剪；安全补全只作为离线上界诊断。

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

## 2026-06-23 当前结论

本轮第一阶段已经把证据图从“普通连通分量直接生成候选”推进到“关系诊断和约束式实例假设”：

- `tools/export_mask_graph_proposals.py`
  - 已加入逐点可见性和二维掩码一致性验证。
  - 已保存同实例支持、父子包含、硬冲突、不确定关系。
  - 跨类别跨视角关系不再直接作为硬冲突，而是作为不确定关系。
  - 默认使用约束式实例假设：`--graph_hypothesis_mode constrained`。
  - 输出 `mask_graph_trace.json` 和已有实例修正诊断。
- `tools/analyze_mask_graph_trace_relations.py`
  - 新增关系质量诊断脚本。
- `tools/analyze_applied_mask_graph_candidates.py`
  - 可分析全部导出候选、完整核心、缺口核心、已有实例修正诊断。
  - 可比较“原始已有候选 / 二维证据修剪核心 / 原始加核心补全”。

even48 第一阶段诊断结果：

- 二维掩码观测：`3144`
- 同实例支持边：`2242`
- 同实例支持边正确率：`0.925513`
- 不确定关系：`2576`
- 硬冲突边：`2`
- 最终新增候选：`2`
- 已有实例修正诊断：`352`

关键判断：

- 证据图方向没有被否定。
- 被否定的是：
  - 普通图连通分量直接输出候选。
  - 只输出缺口核心。
  - 无条件用二维核心修剪 Mask3D。
  - 无条件把二维核心补到 Mask3D。
- 缺口核心最高真实交并比均值只有 `0.006815`，基本不能单独作为新增实例。
- 二维证据修剪核心只有 `8 / 352` 条比原始 Mask3D 更好，不能替换。
- 原始加核心补全有 `70 / 352` 条变好，但 `195 / 352` 条变差，不能无条件补全。

因此当前证据图最可靠的用途是：先做已有实例修正上界诊断，再学习或手写一个非常保守的安全补全规则。

## 2026-06-23 下一步

不要先跑 even96，也不要直接把当前新增候选接入最终 AP。当前新增候选只有 `2` 个，且诊断全部为坏候选。

下一步应先做“安全补全规则”：

1. 新增可推理使用的边界质量特征：
   - 超点是否跨越多个二维实例边界。
   - 补全后包围盒体积变化。
   - 补全后连通块数量变化。
   - 颜色、法向和空间连续性。
   - 新增区域是否集中在一个连通超点区域。
2. 对已有实例修正诊断执行动作判断：
   - 保留原始。
   - 安全补全。
   - 拒绝补全。
   - 暂不自动替换。
3. 只允许极少数高置信补全：
   - 新增点比例不能过大。
   - 补全后不能显著增加多物体错误合并风险。
   - 原始 Mask3D 候选本身必须存在明显残缺迹象。
4. 只有 even48 上“补全变好数量明显多于变差数量”，再跑最终 AP 和 even96。

最新诊断输出目录：

```text
output/mask_graph_proposals_scannet200_even48_phase1_relation_fix_gpu
output/scannet200/subset_sweeps/even48_mask_graph_phase1_relation_fix_diagnostics/
```

最新检查命令：

```bash
python -m py_compile tools/export_mask_graph_proposals.py tools/analyze_applied_mask_graph_candidates.py tools/analyze_mask_graph_trace_relations.py
bash -n tools/run_scannet200_even48_mask_graph_eval.sh
git diff --check
```
