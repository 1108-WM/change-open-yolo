# A100 复现问题与补充包

[下载补充ZIP](a100_supplement_20261008.zip)，解压后按下文运行。原27b7bec ZIP和925文件保持原样。

# A100 补充材料：两个冻结场景对照与实际环境分工

对应原发布提交 `27b7becd33d8136622b4e50cefb39b64dbb4b6c7`。
本包补充参考缓存、环境记录和辅助入口；不替换原925文件、模型、门槛或评测器。

A100 的检查表说明自训权重和第三方权重已经齐全。剩余问题是缓存接入、实际候选自身语义，以及视觉二进制依赖。
CPU43测试通过只验证CPU范围，不能证明Mask3D或Alpha的GPU阶段可运行。

## 1. 两场景 CPU 重放：先独立验证后端

`scenes/<scene>/`包含原始 `geometry.json`、`points.npz`、绑定回执，
原始 `evidence.jsonl`、语义回执、根 `identity.json`、原模型溯源和运行记录，
历史 Full800 的完整选择、分数与回执，以及先前发布包的CPU重放参考。
所有原始文件按字节复制，绝对路径原样保留，迁移时不重签历史回执。
本包仅包含候选点索引和推理证据，不含原始RGB-D、GT文本、训练标签或新AP。

使用已通过43测试的 sklearn1.3.2 后端 Python。以下路径替换成 A100 实际位置：

```bash
/ABS/backend/bin/python /ABS/a100_supplement_20261008/replay_reference.py \
  --release-root /root/OpenYOLO3D/reproduction_downloads/full800_reproduction_20261008 \
  --output /root/OpenYOLO3D/reproduction_downloads/a100_reference_replay_v1
```

脚本先核验补充包和原发布包，再从原几何及自身语义重新构建全部X26、B26分数、K800。
逐数组、逐成员、逐类别、逐分数、逐顺序、逐点集对照，不是只复制参考预测。
成功时 `verification.json`显示6934＋5936行、800＋800输出全部一致。失败时保留输出，使用新目录重试。
此步骤不加载视觉模型、不读GT，不依赖A100上未知格式的G10缓存。

## 2. 历史环境实际使用了不同的 Python

| 阶段 | 本机历史路径／分工 | 实际关键版本 |
|---|---|---|
| Native Mask3D／YOLO、uniform30 SAM几何观测，以及初始化OpenYolo3D的uniform30 YOLO-SAM工具 | `/home/jia/anaconda3/envs/openyolo3d/bin/python` | Python3.10；Torch1.12.1+cu113；torchvision0.13.1+cu113；torch-scatter2.0.9；MinkowskiEngine0.5.4；mmcv2.0.0 |
| D-v3几何 | `/media/jia/软件1/scannet_validation_stream/environments/fi1_d_v3_infer_py310_sklearn130_20260902/bin/python`，借用native环境依赖 | Python3.10；sklearn1.3.0；NumPy1.26.4；Torch1.12.1+cu113 |
| GD-SAM前端、实际点集自身YOLO投影／Alpha补算 | 项目 `.venvs/yoloe_gpu/bin/python` | Python3.11；Torch2.1.2+cu118；torchvision0.16.2+cu118；NumPy1.26.4 |
| B26／K800 | 同一历史 `.venvs/yoloe_gpu`；复现时可用独立CPU环境 | sklearn1.3.2；SciPy1.11.4；joblib1.4.2 |

`environments/`包含当前 `pip freeze`、native conda包／构建清单、实际导入版本及路径，以及原D-v3启动配方。
**这些是2026-10-08当前环境快照，不是每个历史任务运行当日的完整锁文件。**
native当前sklearn1.7.2不能用于D-v3或B26；D-v3必须1.3.0，B26必须1.3.2。
D-v3旧环境是 `PYTHONPATH` 叠加运行；单独复制其pip清单会漏掉借用的Torch／投影依赖。
新机器宜建立完整native环境后，克隆为geometry环境，再按原要求覆盖sklearn1.3.0及NumPy1.26.4。
semantic环境独立安装原GD/SAM/Alpha依赖；CPU后端按原 `requirements-backend.txt` 建立。

### 修复 native 的 torch_scatter

先确认操作的就是 native Python，不是另一个pip对应的环境：

```bash
/ABS/native/bin/python -c "import sys,torch; print(sys.executable,torch.__version__,torch.version.cuda)"
```

若已按历史路线建成Python3.10、Torch1.12.1+cu113，在**独立native环境**安装匹配的历史轮子：

```bash
/ABS/native/bin/python -m pip install --force-reinstall --no-cache-dir \
  --no-index --no-deps --only-binary=:all: 'torch-scatter==2.0.9' \
  -f 'https://data.pyg.org/whl/torch-1.12.1+cu113.html'
```

此命令不适用于Torch2.1.2+cu118或其他组合；必须选择相应Torch／CUDA／Python的轮子。
依据：[PyG官方安装说明](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)、
[历史Torch1.12.1+cu113轮子](https://data.pyg.org/whl/torch-1.12.1%2Bcu113.html)。
使用 `torch.version.cuda` 判断Torch轮子编译版本，不能仅据 `nvidia-smi`显示值选轮子。
MinkowskiEngine、pointnet2、mmcv等扩展也需在同一native环境内检查；升级Torch之后旧扩展必须重新匹配。

导入和实际CUDA算子检查在A100上执行：

```bash
/ABS/native/bin/python -c "import torch,torch_scatter,MinkowskiEngine,mmcv; print(torch.__version__,torch.version.cuda,torch_scatter.__version__); x=torch.tensor([1.,2.,3.],device='cuda'); i=torch.tensor([0,0,1],device='cuda'); print(torch_scatter.scatter_mean(x,i,dim=0)); torch.cuda.synchronize()"
```

预期scatter均值 `[1.5, 3.0]`。这只是算子检查，不是完整模型推理验收。

## 3. GD10格式审计，不绕过拒绝

新入口要求 `contract.assets`为原路径→SHA字典，原始帧D为0/10/20…，
六个前端源码身份和GD/SAM权重身份一致，四个阶段回执 `identity/files` 与合同和manifest绑定。
数据迁移后的旧绝对路径是另一个可能原因。仅凭“合同不兼容”不能判断属于哪一种。

```bash
python3 /ABS/a100_supplement_20261008/audit_frontend.py \
  --release-root /root/OpenYOLO3D/reproduction_downloads/full800_reproduction_20261008 \
  --frontend-root /ABS/existing_G10_frontend \
  --dataset-root /ABS/ScanNet312 --scene scene0354_00 \
  > /ABS/frontend_audit_scene0354_00.json
```

脚本只读缓存，输出具体字段、源码SHA、阶段回执或路径失败项，不修改缓存、不授予新身份。
`frontend_schema_example/`是本机已使用的0354合同、manifest及四阶段回执示例，
**不是完整GD10缓存，不能拿示例回执覆盖A100缓存。**

处理次序：先核对原始帧与点序及文件内容；若只有搬迁，创建独立路径映射和迁移回执，保留旧合同；
若只有字段命名差异，按已审计的一一对应关系创建只改表示的适配视图；
若模型、源码核、点集或采样不同，则不能冒充本历史缓存，需在独立目录补算相关阶段。
必须取得A100实际合同字段和错误堆栈后才能落实适配。本包没有猜测未知格式或放宽校验。

## 4. 缺失自身语义：完整候选绑定后补算

GD10轨迹不是最终N/T/U/U′候选全集，也不是它们自身YOLO／Alpha证据。
先结合native与uniform30几何验证证据运行冻结D-v3，再绑定实际点集；
对每个候选的实际点集导出自身YOLO分布、Alpha视图和补算请求，补齐原198类连续证据；
最终再构建R1成员、X26并用B26输出K800。
不能沿用父候选语义、用GD检测标签代替自身语义、填零向量或退化top1。

原入口 `--visual-python`只有一个参数。新增辅助入口按阶段分派native／semantic环境，
依旧调用原入口和原校验，不改冻结代码。先只打印分工：

```bash
/ABS/backend/bin/python /ABS/a100_supplement_20261008/run_with_split_visual_envs.py \
  --release-root /root/OpenYOLO3D/reproduction_downloads/full800_reproduction_20261008 \
  --dataset-root /ABS/ScanNet312 --output /ABS/new_single_scene_run \
  --native-python /ABS/native/bin/python --semantic-python /ABS/semantic/bin/python \
  --geometry-python /ABS/geometry/bin/python --backend-python /ABS/backend/bin/python \
  --frontend-root /ABS/audited_G10_frontend --scenes scene0549_00 --print-routing
```

缓存通过接入检查、各环境就绪后，移除 `--print-routing` 执行单场景。
没有合格既有缓存时省略 `--frontend-root`，会在新目录生成GD10，不应未经必要检查启动整312。
**辅助入口的分派有CPU测试，完整GPU编排尚未在A100实测。** 两参考场景验证后再考虑 `--all-scenes`。

## 5. 本次交付边界

38份原始参考／回执文件按字节复制；manifest覆盖全部补充文件；两个场景CPU缓存重放结果单独记录。
本包不解决未知格式缓存的具体映射，不声称A100端已经完成GPU或AP。
29.230610／39.901908／45.578362%仍是指定历史Full800结果，不是新机器重跑成绩。
