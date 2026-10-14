# Replica 官方候选与复现入口

更新：2026-10-12。8个作者发布候选已复制到 `replica_masks/`：`office0.pt`—`office4.pt`、`room0.pt`—`room2.pt`，共685649336字节（约654MiB）。全部SHA256与官方包核验记录一致；这是Mask3D类别无关预测候选，不是真值掩码。

原始数据入口为项目根的 `data/replica/`，当前解析到 `/home/jia/Wm/wm_open-yolo/OpenYOLO3D/OpenYOLO3D/replica`。候选按仓库README的标准约定放在 `output/replica/replica_masks/`，不混入RGB-D场景文件夹。8个包内原网格与本地网格逐字节一致；每个掩码的点数与候选数见[安装清单](official_replica_masks_manifest.json)，候选索引依赖原网格点序，不能换成半采样或重排网格。

来源：[作者官方资产包](https://drive.google.com/file/d/1FneLaYaClWDO51L9lIvlTQbheh5SfOFD/view)，官方 `scripts/get_class_agn_masks.sh`（上游commit `24f344d001d28d1cdd4087d3769c0d53c34d64cd`）。ZIP SHA256为 `8c129b4d94f36ef5a256594d0cf4135c263adec83f07db3b6fb3cc5c55c1036a`。下载包和隔离源目录保留在 `.local/official_openyolo3d_assets_20261011/`；旧合同引用的源文件没有移动。本次仅复制资产与更新文档，没有新训练、预测、评测或读取GT。

## 官方基线入口

在项目根执行仓库README约定的命令（以下会实际运行预测和评测，本次没有执行）：

```bash
/home/jia/anaconda3/envs/openyolo3d/bin/python run_evaluation.py \
  --dataset_name replica \
  --path_to_3d_masks ./output/replica/replica_masks \
  --eval_score_mode uniform
```

论文对照使用uniform分数；native分数是另一项对照。该命令是当前项目入口；精确追溯本轮结果应使用下述冻结配置及作者原始赋类函数快照，不能只凭掩码一致就认为其他配置自动一致。预测默认输出位置由 `run_evaluation.py`决定，新实验应另行隔离输出，保留已有封存结果。

## 本轮官方基线与生产Full结果复现

[封存报告](../../.local/replica_official_masks_full_20261011_v1/report.md)／[完整指标](../../.local/replica_official_masks_full_20261011_v1/evaluation/results.json)／[冻结合同](../../.local/replica_official_masks_full_20261011_v1/contract.json)／[协议](../../.local/replica_official_masks_full_20261011_v1/protocol_notes.md)。基线类别AP/AP50/AP25＝23.658381／28.570491／34.810598%，复现论文23.7／28.6／34.8；Full＝24.567057／32.026546／40.887651%。实际去类AP基线27.639376%、Full26.193269%，未双口径改善。

生产Full为GD10＋历史official100 D-v3七模型＋GD10匹配597 B26／NTUU′／K800，不是S、C或SL。入口 `tools/run_replica_official_masks_full.py` 包含prepare、native、backend、evidence、score、evaluate、verify、chain；`tools/resume_replica_official_scoring.py`记录评分环境修正。入口绑定旧隔离源路径及已完成实验目录，保持源码不改；直接重新运行不会自动创建一轮全新实验。下次新评测先建立新输出合同，使用这里的同哈希掩码并绑定完整输入、模型与源码。

原生推理环境为 `/home/jia/anaconda3/envs/openyolo3d/bin/python`；历史D-v3使用sklearn1.3.0，B26必须使用原sklearn1.3.2（`.venvs/yoloe_gpu/bin/python`），不能混用。Full须重新构建原生依赖的关系、U/U′、新点集自身语义、完整池26维竞争上下文和B26/K800；不能仅替换基线数字。两方法全8场景预测封存后才运行GT评测，实际去类从已选成员按精确同点集合并取最高分，不补位。
