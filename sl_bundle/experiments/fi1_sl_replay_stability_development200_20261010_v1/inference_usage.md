# SL真实推理入口

无需封存分数、标签或GT。直接输入实际几何、自身语义证据和权重。例子仅用于现有留出场景；最终部署需改用final_validation_plan.md规定的一个单模型。

```bash
.venvs/yoloe_gpu/bin/python tools/fi1_sl_inference.py \
  --geometry '/media/jia/软件1/scannet_additional_train_stream/fi1_G10_closure_20260921/train/scenes/scene0002_00/own/D/geometry_binding/scene0002_00/geometry.json' \
  --points '/media/jia/软件1/scannet_additional_train_stream/fi1_G10_closure_20260921/train/scenes/scene0002_00/own/D/geometry_binding/scene0002_00/points.npz' \
  --evidence '/media/jia/软件1/scannet_additional_train_stream/fi1_G10_closure_20260921/train/scenes/scene0002_00/own/D/alpha_completed/scene0002_00/evidence.jsonl' \
  --point-count 297362 \
  --b26 '/media/jia/软件1/scannet_additional_train_stream/fi1_G10_closure_20260921/learning/models/B.joblib' \
  --l-model '/home/jia/Wm/wm_open-yolo/OpenYOLO3D/.local/fi1_s_semantic_rank_residual_development200_20261010_v2/models/fold1/model.pt' \
  --normalization '/home/jia/Wm/wm_open-yolo/OpenYOLO3D/.local/fi1_s_semantic_rank_residual_development200_20261010_v2/models/fold1/normalization.npz' \
  --output .local/sl_single_scene_new_output
```

输出目录必须尚不存在，避免覆盖已有预测。生成semantic_SL.json、class_agnostic_SL.json及输入／输出哈希回执；分数排完后仍是S原先选出的800个身份，无补位。

完整S资格池先算26维特征和B26分数；L只对选中成员提供实际分数。为精确重放保留原8192行批次形状，其余行零填充，非选中行不会输出。网络只有逐行linear/tanh，无批归一化或跨行算子，填充不形成竞争证据。

运行环境固定现有.venvs/yoloe_gpu，torch CPU4线程；其他BLAS／CPU平台需按浮点容差核验，不声称任意硬件位级相同。当前200场景和独立CLI烟测均在既定环境精确复现。
