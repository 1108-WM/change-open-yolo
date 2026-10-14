# 有效 SL 独立保存

> GitHub源码快照（2026-10-14）：本文件中的本机资产/历史交付说明保留作溯源。本次main不包含权重、数据或缓存；运行需要另备资产，见../docs/EXTERNAL_ASSETS.md。历史哈希与验证结果不代表本次重新运行。


2026-10-14 已集中有效 S/L/SL；Full仍保持原冻结资格配置。

- `cli/infer.py`：GT-free推理入口，参数见 `--help`；调用 `code/tools/fi1_sl_inference.py`。
- `code/`：SL专用实现和测试。共用几何/语义/B26 helper来自Full，通过项目兼容包导入。
- `models/single200/`：200训练→55开发验证的单L模型、归一化及训练回执。
- `models/folds/`：原有效五折L模型与归一化。
- `protected_sources/`：实际迁入的S资格实验、有效L v2、S固定L重评分、稳定性回放、单模型55验证完整资产；旧 `.local` 路径为兼容链接。

数据/实际几何/自身语义与Full共享，不复制第二份；输入由原contract及 `full_bundle/metadata/scene_routes.json`确定。开发55的 `scene0005_00` 在迁移后从实际点集与自身语义重建完整S池，类别SL和实际去类SL与原封存结果逐条完全一致。

归一化、模型、训练/预测身份、原物理划分与历史曝光结论保留；没有在这次整理中重新训练或选择规则。完整文件名见[当前代码索引](../docs/cleanup/deep_active_cleanup_20261014/current_code_index.csv)。

当前44个实现文件与17个测试文件；原误放在几何目录中的12个SL测试已归回。单模型训练守卫已修复兼容链接路径比较，GT访问范围保持。历史SL实验入口仍有共享定义依赖，不能宣称44个文件均为最小推理入口。
