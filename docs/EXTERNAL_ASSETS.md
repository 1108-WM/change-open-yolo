# 未上传的外部资产

本次GitHub同步明确不包含模型权重、数据集、处理后数据、预测或特征缓存。代码完整性与资产可用性分开判断。

| 资产 | 源码/身份依据 | 准备方式 |
|---|---|---|
| official100 D-v3七模型、champion、GD10匹配597 B26 | full_bundle/models下身份JSON；runtime/reproduction/manifest.json | 从本机原冻结资产另行转移，不重训替代 |
| SL单模型/五折L及归一化 | sl_bundle/models下身份JSON、SL协议/报告 | 另行转移原模型与normalization.npz |
| Mask3D、YOLO-World、GD、SAM、AlphaCLIP、BERT/CLIP、DINO、SAM2 | 配置、vendor源码、runtime/reproduction/external_weights.json | 按Installation和上游官方说明获取指定版本；SAM2未成为当前有效默认项 |
| Alpha文本向量 | runtime/release_assets/official100_gd10_b26_k800/alpha_text身份与提示JSON | 从原封存资产另备，源码发布未上传向量 |
| ScanNet/Replica原始数据、处理点云、RGB-D与GT | docs/Data_prep.md、场景身份清单 | 按原数据访问条件和准备流程获取，保持原点序及帧编号 |
| native/GD10轨迹、几何/自身语义、训练特征、原实验预测 | 对应代码、协议、精选报告与历史合同 | 另备原资产或按具体授权任务生成；不要按缺失文件自动恢复旧训练 |

仓库中metadata/scene_routes.json等绝对路径保留作本机溯源，不是远端下载地址；历史哈希也不证明这些资产已经存在。完整运行需明确输入路径、相应Python/CUDA依赖和模型身份。当前发布不提供数据/权重的自动下载承诺。

历史权重交付分支仍保留：[Full历史发布](https://github.com/1108-WM/change-open-yolo/tree/releases/official100-gd10-b26-k800-20261008)。其包含内容与本次main的无权重源码范围不同。
