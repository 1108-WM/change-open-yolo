# 冻结 Full800 复现包：A100 / Linux

> GitHub源码快照（2026-10-14）：本文件中的本机资产/历史交付说明保留作溯源。本次main不包含权重、数据或缓存；运行需要另备资产，见../../../docs/EXTERNAL_ASSETS.md。历史哈希与验证结果不代表本次重新运行。


本分支发布 **历史official100 D-v3七模型＋GD10匹配597训练B26＋K800**。
Full保留N/T/U/U′，不使用后续分类准入、筛后U或新的细化策略。
运行复现不需要重新训练D-v3或B26。历史验证312的AP/AP50/AP25为
29.230610 / 39.901908 / 45.578362%，这是已有冻结结果，不是本次重新评测。

## 1. 下载和验证

```bash
git clone --depth 1 --branch releases/official100-gd10-b26-k800-20261008 --recurse-submodules https://github.com/1108-WM/change-open-yolo.git
cd change-open-yolo
python3 tools/run_frozen_full800.py verify
```

也可下载 `reproduction/full800_reproduction_20261008.zip`，解压后进入
`full800_reproduction_20261008/`。ZIP是独立源码/自训权重包；Mask3D的MinkowskiEngine、
ScanNet工具仍需按 `docs/Installation.md` 的固定提交安装，ZIP不携带Git子模块数据库。

`manifest.json`记录全部运行核心和权重SHA256。冻结模型位于
`release_assets/official100_gd10_b26_k800/`：

| 目录 | 内容 |
|---|---|
| `dv3/` | official100七模型、原校准元数据和summary |
| `b26/` | 597 GD10正式 `B.joblib`、原597训练身份和参数 |
| `champion/` | 原几何流程使用的champion并集模型 |
| `alpha_text/` | 冻结198类提示和文本向量；不是场景图像语义缓存 |

原模型字节不变。部分原始溯源JSON含历史绝对路径，只保留作身份说明；
新入口不使用这些路径查找模型或数据。历史工具中的旧默认参数仍保留，
**请使用本说明的新入口**，不要直接恢复旧597训练总控或研究实验。

## 2. 先判断A100上已有缓存是什么

```bash
python3 tools/run_frozen_full800.py inspect --root /ABSOLUTE/PATH/TO/G10_CACHE
```

此命令只列目录和文件，不读取GT、不运行视觉模型。

- 只有 `automatic_tracks.json`：通常只是前端轨迹，并不说明有最终候选自身语义。
- 有 `geometry.json`＋`points.npz`：已保存实际D-v3候选；需按场景确认来源和点集身份。
- 有与上述点集匹配的 `evidence.jsonl`：可能已有自身YOLO／Alpha语义，可直接CPU评分。
- 只有 `yolo_and_alpha_requests.jsonl`：Alpha可能尚未完成。
- `similarities.npy`通常是Alpha图像缓存；能否复用还需核对几何、视角、模型和词表。

文件存在不等于身份核验通过。其他服务器复制来的JSON可能含旧绝对路径；
不能只改路径文字或沿用旧哈希冒充已核验缓存。未知格式的G10缓存先检查，
不要盲目重跑整312；缺失的自身语义由原流程对实际候选点集补算。

## 3. 已有实际候选和完整自身语义：直接B26 / K800

使用单独的轻量CPU后端环境，不需要CUDA或第三方视觉权重：

```bash
python3.10 -m venv .venvs/full800_backend
.venvs/full800_backend/bin/pip install -r reproduction/requirements-backend.txt
.venvs/full800_backend/bin/python tools/run_frozen_full800.py score \
  --scenes scene0549_00 \
  --geometry-dir /ABSOLUTE/PATH/geometry_binding/scene0549_00 \
  --evidence-file /ABSOLUTE/PATH/alpha_completed/scene0549_00/evidence.jsonl \
  --output /ABSOLUTE/PATH/run800/predictions/scene0549_00
```

`geometry-dir`包含 `geometry.json`、`points.npz`和绑定 `receipt.json`。
若独立转移包未包含该回执，需明确传 `--point-count N`，N必须是完整场景点数，
不是候选最大点编号或某个实例大小。入口逐点核对点数、SHA、场景、来源和自身语义，
重建R1成员、20维自身／来源特征＋6维同类竞争上下文，使用正式B26重评分。

原Alpha根目录的 `identity.json`或新流程的场景回执还需保留，以核对语义模型和文本向量身份。
若原模型溯源单独存放，可显式传 `--semantic-provenance /PATH/TO/model_provenance.json`；
这是对该缓存原始模型身份的声明，不能把本包的模型JSON复制过去给来源未知的向量背书。

输出 `selected.json`、`selected_points.npz`、`features_scores.npz`及哈希回执。
800计数几何—类别成员，不是800个唯一物体；不足800不补假成员，不压top1，不加NMS。
完成全部所需场景后封存：

```bash
.venvs/full800_backend/bin/python tools/run_frozen_full800.py seal \
  --predictions-root /ABSOLUTE/PATH/run800/predictions --scenes scene0549_00
```

省略 `--scenes`时要求原完整验证312全部完成；给子集时只封存该子集。

## 4. 从已导出RGB-D / 点云运行完整流程，或接入已有G10前端

数据目录必须保留原始编号和完整序列：

```text
DATA/scene0549_00/
  color/0.jpg, 1.jpg, ...
  depth/0.png, 1.png, ...
  poses/0.txt, 1.txt, ...
  intrinsics.txt
  scene0549_00_vh_clean_2.ply
  0549_00.npy
```

`.npy`为原处理点云／superpoint格式，点顺序必须与网格一致。
仅30帧缓存不能替代原GD10输入；推理不需要GT文本文件。

A100视觉环境按 `docs/Installation.md` 安装Python3.10、PyTorch1.12.1＋CUDA11.3、
torchvision0.13.1、Mask3D/MinkowskiEngine和YOLO-World依赖。A100为sm80。
GD-SAM还需安装GroundingDINO及SAM，自身语义还需Alpha-CLIP。
这里的版本描述是历史运行条件，不保证任意新的GPU软件栈数值一致。

本包不上传ScanNet数据、第三方大模型权重、图像/候选缓存、凭证或虚拟环境。
第三方资产按原安装说明和各项目官方说明准备：

| 资产 | 仓库内位置 |
|---|---|
| Mask3D | `pretrained/checkpoints/scannet200_val.ckpt` |
| YOLO-World | `pretrained/checkpoints/yolo_world_v2_x_obj365v1_goldg_cc3mlite_pretrain_1280ft-14996a36.pth` |
| SAM ViT-B | `pretrained/checkpoints/sam_vit_b_01ec64.pth`；源码 `_external/segment-anything/segment-anything-main/` |
| GroundingDINO SwinT | `pretrained/groundingdino/groundingdino_swint_ogc.pth`；源码 `_external/GroundingDINO/GroundingDINO-main/` |
| BERT-base-uncased | `pretrained/bert-base-uncased/`，完整config/tokenizer/词表与权重文件 |
| Alpha-CLIP | `pretrained/alpha_clip/checkpoints/ViT-L-14.pt`、`clip_l14_grit20m_fultune_2xe.pth`；源码 `_external/AlphaCLIP/AlphaCLIP-main/` |

前端原资产合同会读取BERT的 `config.json`、`pytorch_model.bin`、`model.safetensors`、
`tokenizer.json`、`tokenizer_config.json`、`vocab.txt`，这些文件应完整保留。
`external_weights.json`记录六组外部权重的原SHA256。可从当前工作机复制这些已有资产到A100；
复制后先验证，不能用同名新权重替代。

几何D-v3模型要求 **sklearn1.3.0**；B26要求 **sklearn1.3.2**。
在具有投影／Torch／YAML等依赖的独立几何环境安装
`requirements-geometry.txt`，不要在一个环境里来回替换sklearn。
视觉环境可与几何环境相同；CPU后端使用上一节的独立venv。

```bash
.venvs/full800_backend/bin/python tools/run_frozen_full800.py verify --with-external-assets
.venvs/full800_backend/bin/python tools/run_frozen_full800.py run \
  --dataset-root /ABSOLUTE/PATH/DATA \
  --output /ABSOLUTE/PATH/NEW_RUN800 \
  --visual-python /ABSOLUTE/PATH/visual_env/bin/python \
  --geometry-python /ABSOLUTE/PATH/geometry_sklearn130/bin/python \
  --backend-python /ABSOLUTE/PATH/change-open-yolo/.venvs/full800_backend/bin/python \
  --scenes scene0549_00
```

这是先验证单场景的命令。确认后省略 `--scenes`执行原验证312，无参数扫描。
入口按场景串行调用原Native/YOLO、uniform30验证观测、GD10前端、D-v3、
实际点集绑定、自身语义和B26/K800；不会重新训练或自动评测。

若A100已有按原合同生成、路径仍有效的完整G10前端，在同一命令增加：

```bash
  --frontend-root /ABSOLUTE/PATH/EXISTING_G10_FRONTEND
```

它需包含 `contract.json`、`manifests/<scene>.json`及
`arms/D/{hierarchy,tracks,consensus,merged,stage_receipts}/`。
入口先检查原始stride10、GD/SAM权重身份、数据根和四阶段回执，再跳过GD10重新生成。
同时核对原前端核的源码SHA；通过检查后可省去本次不再调用的GroundingDINO/BERT资产。
Native/YOLO及uniform30几何验证证据是另一路必要输入；只持有G10轨迹不能替代它们。
若缓存格式不同，先用 `inspect`的结果确认接入方式，不把来源不明的几何当作完整冻结结果。

## 5. 原评测器，预测全部封存后才读取GT

```bash
.venvs/full800_backend/bin/python tools/run_frozen_full800.py evaluate \
  --predictions-root /ABSOLUTE/PATH/NEW_RUN800/predictions \
  --gt-root /ABSOLUTE/PATH/ground_truth \
  --output /ABSOLUTE/PATH/NEW_EVALUATION \
  --allow-gt-evaluation
```

GT命名为 `<scene>.txt`，必须与完整场景点顺序一致。
严格使用原198实例类、小实例/void忽略、严格IoU和合并场景AP；
主AP为0.50—0.90九档平均，AP25独列。子集结果不会标记为完整312复现。
不读取官方最终测试，也不在验证312修改门槛。

## 6. 本次发布的验证范围

`verification.json`记录代码／权重哈希、CPU测试及既有真实场景缓存的精确重放。
已验证的缓存重放不等同于A100上重新完成视觉推理。
本次不重跑312、不开新GPU作业、不重训、不产生新AP。
新 portable GPU编排尚未在A100进行全流程实测；首轮应单场景检查各阶段回执，
发现失败保留现场，不覆盖旧模型、旧缓存或旧预测。完整原始视觉重放还依赖数据、
第三方源码／权重、CUDA编译环境和点集身份；下载两组自训权重并不足以独立完成全部推理。
