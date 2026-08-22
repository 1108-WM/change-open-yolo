# FI1-D-v3 val312 远程服务器冻结运行说明

## 分工边界

本地分支已经固定算法、7 个 official100 全量模型、无 GT 推理、独立审计和唯一 AP
入口。远程 Codex 只允许：

- 核对服务器已有 val312 资产；
- 把下列命令中的路径替换为服务器真实路径；
- 若服务器的 FI1-Legacy 资产目录命名不同，建立只读适配目录或符号链接；
- 执行测试、无 GT 推理、审计和唯一一次 AP。

禁止远程修改模型、算法、阈值、特征、q90、候选策略或 AP 公式。

## 需要的 val312 资产

每个场景必须具备：

```text
准备后的点云 .npy（至少 xyz/rgb/normal/raw-superpoint 10 列）
FI1-Legacy unique_geometry_ledger.jsonl
GVC 质量账本 c1_gvc_quality_ledger.json
无 GT 关系账本 relation_features_no_gt.jsonl
FI1-Legacy champion_plan（frozen_score_plan 与 pair_union_append_candidates）
过滤后的 automatic_tracks.json
SAM automatic_observations.jsonl、点索引、深度图、相机位姿与内参
val312 场景列表
GT 文本目录（只在最后 AP 命令开放）
```

如果服务器只有 FI1-Legacy 最终预测而缺少上述中间证据，先停止并报告缺失资产；
不得由远程 Codex猜测或重建不同版本的算法输入。

## 运行顺序

先执行快速测试：

```bash
python -m pytest -q tests/test_fi1_d_v3_frozen_deployment.py
```

再生成无 GT 冻结计划，其中所有尖括号路径由远程 Codex 只做路径替换：

```bash
python tools/build_fi1_d_v3_frozen_inference_plan.py \
  --scene-list <VAL312_SCENE_LIST> \
  --prepared-root <VAL312_PREPARED_ROOT> \
  --unique-geometry-root <FI1_LEGACY_UNIQUE_GEOMETRY_ROOT> \
  --gvc-root <FI1_LEGACY_GVC_ROOT> \
  --relation-root <FI1_LEGACY_RELATION_ROOT> \
  --champion-plan-root <FI1_LEGACY_CHAMPION_PLAN_ROOT> \
  --track-root <FI1_LEGACY_FILTERED_TRACK_ROOT> \
  --automatic-root <FI1_LEGACY_SAM_AUTOMATIC_ROOT> \
  --config-path pretrained/config_scannet200.yaml \
  --model-root pretrained/fi1_d_v3_official100_full_models_20260822 \
  --output-root output/fi1_d_v3_val312_frozen_inference_20260822 \
  --dataset-name ScanNet200-val312
```

推理完成后先审计：

```bash
python tools/audit_fi1_d_v3_frozen_inference_plan.py \
  --inference-root output/fi1_d_v3_val312_frozen_inference_20260822 \
  --unique-geometry-root <FI1_LEGACY_UNIQUE_GEOMETRY_ROOT> \
  --output-root docs/diagnostics/fi1_d_v3_val312_frozen_inference_audit_20260822
```

必须确认审计摘要中：

```text
audit_valid = true
error_count = 0
advancement_authorized = true
```

最后只执行一次 AP：

```bash
python tools/evaluate_fi1_d_v3_frozen_class_agnostic_ap_gt.py \
  --scene-list <VAL312_SCENE_LIST> \
  --ground-truth-root <VAL312_GT_ROOT> \
  --inference-root output/fi1_d_v3_val312_frozen_inference_20260822 \
  --audit-root docs/diagnostics/fi1_d_v3_val312_frozen_inference_audit_20260822 \
  --output-root output/fi1_d_v3_val312_class_agnostic_ap_20260822 \
  --dataset-name ScanNet200-val312 \
  --allow-gt-evaluation
```

先对 AP 结果做独立审计：

```bash
python tools/audit_fi1_d_v3_frozen_class_agnostic_ap.py \
  --result-root output/fi1_d_v3_val312_class_agnostic_ap_20260822 \
  --output-root docs/diagnostics/fi1_d_v3_val312_class_agnostic_ap_audit_20260822
```

AP 审计也必须满足 `audit_valid=true` 且 `error_count=0`。最终回传三个
`summary.json` 和命令使用的真实路径映射，不修改代码、不重跑：

```text
docs/diagnostics/fi1_d_v3_val312_frozen_inference_audit_20260822/summary.json
output/fi1_d_v3_val312_class_agnostic_ap_20260822/summary.json
docs/diagnostics/fi1_d_v3_val312_class_agnostic_ap_audit_20260822/summary.json
```
