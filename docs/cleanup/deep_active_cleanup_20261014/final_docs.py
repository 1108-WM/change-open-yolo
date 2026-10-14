from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
p=ROOT/'full_bundle/README.md';t=p.read_text();a=t.index('**范围补正');b=t.index('\n实际模型',a)
t=t[:a]+'''**范围补正（2026-10-14深度精简）：这里是集中保存与兼容运行的目录，仍不等于完全自包含的最小Full源码包。** 本轮删除38个旧策略完整入口和26个对应实验测试，将21,203行旧脚本中的4,778行现用定义抽入`code/tools/fi1_shared/`的29个公共模块；保留的133个函数/类已核对定义一致。这些模块沿用来源文件名便于追溯，但不再保留旧实验CLI。Full tools当前为92个顶层文件，加29个公共模块和1个包初始化文件，共122个；另有4个utils和35个归属正确的测试文件。现用入口、实际依赖与SHA256见最新索引。项目前端仅导出D，Replica共同验证使用独立帧合同与YOLO框提示SAM。`runtime`保持独立冻结实现，`code_sources`保留必要历史版本；完整代码还包含原评测器、第三方实现与配置。
''' +t[b:]
t=t.replace('../docs/cleanup/full_scope_recheck_20261014/current_code_index.csv','../docs/cleanup/deep_active_cleanup_20261014/current_code_index.csv')
t += '\n最新深度精简、`.local`逐项结论和空间统计见[本轮回执](../docs/cleanup/deep_active_cleanup_20261014/README.md)。480项测试及4项子测试、281项公式/Full/SL同输入核对通过；official100七模型与champion加载通过，925发布hash保持。未重训597或全量GPU重算。\n'
p.write_text(t)
p=ROOT/'sl_bundle/README.md';t=p.read_text().replace('../docs/cleanup/full_consolidation_execution_20261014/current_code_index.csv','../docs/cleanup/deep_active_cleanup_20261014/current_code_index.csv');t+='\n当前44个实现文件与17个测试文件；原误放在几何目录中的12个SL测试已归回。单模型训练守卫已修复兼容链接路径比较，GT访问范围保持。历史SL实验入口仍有共享定义依赖，不能宣称44个文件均为最小推理入口。\n';p.write_text(t)
p=ROOT/'research/geometry/README.md';t=p.read_text();t+='\n本轮移除26个退役旧策略测试，另把29个Full测试和12个SL测试归回对应目录。当前研究保留107个实现与28个测试文件；10月11—13日当前几何实现和封存资产未按失败结论删除。最新索引见[本轮回执](../../docs/cleanup/deep_active_cleanup_20261014/README.md)。\n';p.write_text(t)
