# NCS-train100 原始 OpenYOLO3D native-only 类别无关基线预注册

状态：`frozen_before_first_run`

日期：2026-08-22

## 1. 目的

补齐第一创新点各版本与最原始 OpenYOLO3D 之间缺失的 NCS-train100 类别无关对照。该结果只用于建立公共起点，不训练模型、不选择参数，也不改变任何现有第一或第二创新点结果。

## 2. 固定输入

- 场景：`output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt` 的 100 个场景；
- 预测：`/media/jia/软件1/scannet_train_stream/records_ncs/<scene>/native_cache/`；
- 每场景恰好读取原始 `*_pred_masks.npy`、`*_pred_scores.npy`、`*_pred_classes.npy`；
- 共 60,000 个原始 native 预测，每场景 600 个；
- 缓存合同必须为 `frozen Mask3D + YOLO-World only`，且现有原生资产审计错误数为 0。

## 3. 固定评测合同

1. 原始 mask 和原始 score 逐项保持不变；
2. 不加入 track、pair-union、refined union 或其他候选；
3. 不做 exact-geometry 折叠、重复删除、NMS 修改、关系重排、质量校准或分数变换；
4. 原始重复预测继续保留，因为它们属于原始 OpenYOLO3D 输出；
5. 只在评测进程内把所有预测类别统一映射为类别无关标签，并保留有效 GT 实例边界；
6. 使用项目现有 ScanNet200 实例评测器及 `min_region_size=100`；
7. 只运行一个固定方案，输出总体 AP/AP50/AP25 和现有五折方向；
8. 必须显式传入 `--allow-gt-evaluation`；不扫描阈值、分数、候选子集或折叠方式。

## 4. 边界

本次只读取 NCS-train100 GT；不读取 NCS-validation60 或 val312，不运行 GPU，不写回原始缓存，不修改 FI1-Legacy、FI1-D-v3 或 DM-SMS-1。结果无论高低均登记到状态文件，不据此回调现有方案。
