# Mask3D 原生 val312 类别无关 AP

日期：2026-09-12。用户明确要求评测 Mask3D，不能使用 OpenYOLO3D native 结果替代。

**完整评测及独立审计已完成。** 官方 Mask3D 权重重新推理 ScanNet200 验证集 312 场景，原生置信度下的类别无关结果如下（百分数）：

| 方法 | AP | AP50 | AP25 |
|---|---:|---:|---:|
| 官方 Mask3D，原生 750 列/场景，类别无关评测 | **53.901344** | **72.479563** | **82.370545** |

全量预测 234000 列，按评测器最小 100 点规则纳入 216288 列；预测中共有 32685 个场景内不同几何（未据此去重）。GPU 推理用时 1972.43 秒。原 Mask3D evaluator 的 AP 数组精确重放，项目中另一份 evaluator 对相同匹配记录的交叉复核最大误差为 **0.0**。权重、GT、全部预测文件和 AP 数组哈希核验通过；没有第二次 GPU 推理或参数扫描。

此前的 `50.6069 / 67.9754 / 77.9777` 仍是历史 OpenYOLO3D native 候选结果，不能改名为本次 Mask3D 结果。本次 AP 数值也高于历史 FI1-D-v3 的 val312 AP，但两者权重、候选和排序链路不同；FI1 相对其同次 Legacy 控制的收益与是否优于独立 Mask3D 必须分开陈述。本次评测不用于回调第一点/第二点参数。

## 身份核对

- 项目原 `output/scannet200/scannet200_masks/` 有 312 份缓存，但属于 OpenYOLO3D 发布的预计算候选；项目推理入口含额外分数筛选和 NMS，不能据此声明是官方 Mask3D 完整输出。
- 项目原 `pretrained/checkpoints/scannet200_val.ckpt` 大约 909 MiB，元数据是 `3D_OWIS_closed_setting / Mask3D_A_t1`、100 query、非 segment 训练，epoch 549。它不因文件名相同就等于官方 Mask3D 验证权重。
- 本次独立下载 Mask3D 官方源码与 ScanNet200 val 权重，原文件保留。权重来自官方 README 列出的 `https://omnomnom.vision.rwth-aachen.de/data/mask3d/checkpoints/scannet200/scannet200_val.ckpt`；服务器报告大小 158873495 字节。

## 固定协议

使用官方 `scripts/scannet200/scannet200_val.sh` 配置：150 query、top-k=750、DBSCAN ε=0.95 / min_samples=1、segment 训练架构及 segment 投影。直接调用官方模型、voxelizer、`eval_instance_step`、`get_mask_and_scores` 和 `get_full_res_mask`，严格加载全部模型权重。

分数保持 Mask3D 原生的「类别 softmax 概率 × 前景内平均 sigmoid」。不接 YOLO-World、Alpha-CLIP、Qwen、FI1、OpenYOLO3D NMS 或语义分数。原生 top-k 产生的所有预测列保留，包括同一几何的不同类别列；不根据结果改用 top1 或去重。

只在 AP 评测时将全部预测与 198 类前景 GT 合为一个 object 类，保留每个 GT 实例边界。wall/floor/无效标签保持 void；最小区域 100 点，IoU 严格大于阈值，AP 平均使用 0.50–0.90 九档，另报 AP50/AP25。此结果必须连同这一类别无关协议引用，不能假定等于论文中的去重几何候选协议。

输入为本地 ScanNet200 官方验证集 312 场景的 processed NPY（模型只取前 10 列，不使用语义/实例 GT 列），逐点核验 NPY XYZ 与原 PLY 顶点顺序。GT 仅在最终评测使用。随机种子固定为每场景 0，便于断点恢复；不训练、不搜索参数。

## 入口与结果位置

统一目录：`.local/mask3d_native_val312_20260912/`。

- `run.py`：本次唯一适配脚本，含 preflight/self-check/run/evaluate/audit；没有向 tools/tests 增加实验脚本。
- `run_inference_snapshot.py`：实际完成全部 GPU 推理的脚本快照，SHA 与原 `contract.json` 绑定。随后只修复 CPU 统计重复几何时反复解压同一 NPZ 的读取效率，当前 `run.py` 每场景解压一次；无损一致性与推理代码 AST 不变检查通过。未改模型、预测、分数或 evaluator；推理无需重跑。原始合同保留，审计同时记录两个脚本哈希。
- `assets/Mask3D-main/`：官方源码快照；`assets/scannet200_val_verified.ckpt`：完整下载并核验后的官方权重。
- `preflight.json`：312 场景输入身份和点顺序核对。
- `contract.json`：权重/源码/脚本/输入哈希、固定配置和环境。
- `predictions/`：逐场景无损打包的原始 Mask3D 掩码和分数。
- `progress.json`、`inference_complete.json`：实时进度及推理完成标记。
- `full.log`：原 GPU 推理日志；`cpu_evaluation.log`：读取优化后的 CPU 评测和审计日志。
- `completion.json`、`ap.npy`、`matches.pkl`：完整 AP 和可重放匹配记录。
- `audit.json`：哈希、原 evaluator 重放及项目 evaluator 独立实现交叉核对。

环境：`/home/jia/anaconda3/envs/openyolo3d/bin/python`（复用依赖，方法仍是独立 Mask3D）。GPU 必须在沙箱外访问。

## 完成与复核

推理、312 场景 GT 匹配、完整 AP 和审计均已结束，CPU 任务退出码 0；`completion.json` 和 `audit.json` 已生成。312 场景 NPY/PLY 点顺序精确一致（共 49540568 点），官方权重 ZIP CRC 与全部模型参数 strict load 通过。最终结果已经冻结；审计只重放 AP，没有重新运行 GPU 模型。

官方源码版本：`11bd5ff94477ff7194e9a7c52e9fae54d73ac3b5`（GitHub tar 的 commit 元数据）。官方权重 SHA-256：`6a0c4347616a27348bed32ff1ae225cab0087e0a0148f36d1023ca2d46f82534`。该权重仅含 `state_dict`，无训练配置元数据；推理配置来自官方验证脚本，不从项目内同名权重继承。
