# Replica论文基线复现协议核查与结果解释修正

日期：2026-10-11。用户指出论文23.7／28.6／34.8与本轮不一致后核查。仅做源码／封存匹配复算，没有新训练、模型预测或原始GT读取。

## 结论

论文表2的Open-YOLO 3D确为mAP／mAP50／mAP25＝23.7／28.6／34.8%。本轮本地重生成候选得到21.246985／29.833900／35.184633%，没有精确复现该行。先前“原配置基线／baseline_official”表述过强：YAML和uniform评测一致，但候选生成路径并不完全一致。

已发现输入类型与NMS两处实质差异，不能仅以“缓存缺失导致随机性”解释全部差距。尚未量化各差异的AP贡献，不能宣称找到了全部数值差距的唯一原因，也不能声称修正后必然达到23.7。

本轮Full21.538893／28.959757／37.473483%及两方法去类结果保留，作为共享本地候选条件下的迁移对照。Full相比这一对照的类别AP＋0.291908pp，不是相对论文23.7的提升；亦没有双口径改善。未改历史预测、匹配、AP数组或模型。

## 核对证据

对照本地已有官方上游commit `24f344d001d28d1cdd4087d3769c0d53c34d64cd`，不是对照项目历史改造版本。原文件快照与哈希见`upstream_snapshot/`和[audit.json](audit.json)。论文来源：项目`related papers/2025.02（ICLR）（有代码）Open-YOLO 3D Towards Fast and Accurate Open-Vocabulary 3D Instance Segmentation.pdf`表2，第8页。

|项目|官方／论文复现要求|本轮实际|判断|
|---|---|---|---|
|原生候选资产|README要求使用作者预计算class-agnostic masks精确复现|缓存缺失，冻结本地Mask3D重新预测|候选身份不同，不能精确复现|
|Replica YAML|frequency1、可见差.4、top40、top1、48类|与官方解析结果完全一致|一致|
|评测置信度|上游run_evaluation.py使用ones_like|主基线uniform，另列native对照|一致；改native不能解释复现差距|
|候选生成datatype|官方Replica入口point cloud|本轮runner.native使用mesh|实质差异|
|RGB特征|point cloud从Open3D读取float RGB，无mesh归一化|mesh先乘255转uint8，再按ScanNet均值／方差归一化|模型输入不同；即使XYZ／RGB原值相同，也不能认为输入特征相同|
|候选NMS|官方上游apply_nms|项目历史修改的greedy NMS|实现不同，会保留不同候选|
|评测核心|官方assign_instances_for_scan／evaluate_matches／compute_averages|三函数AST精确一致|没有发现AP核心改错|

官方README第79行要求用发布掩码精确复现；第98行明确指出自行生成Mask3D候选可能因furthest point sampling等产生不同结果。该说明支持“重新生成可能不一致”，不证明本轮两处额外配置偏差没有影响。

官方预计算掩码路线会绕过Mask3D重生成及其NMS。不能将官方apply_nms与本文作者实际发布候选的生成算法身份直接画等号。项目历史greedy修正也不应为了追数而在生产中回退。

## 实际复核

- 使用官方原始Replica评测源码重放三套封存semantic_matches.pkl，三个全局AP数组逐元素精确一致；未读取原始GT。
- Replica配置解析一致，YOLO科学函数、Network_3D与Mask3D加载／预处理函数AST一致。YOLO本地增加Pillow兼容常量，不影响此项科学函数对照。
- 固定NMS反例：5点3候选、置信度[.1,.9,.5]，官方返回候选[2,0]，本地返回[1,2]；点集和索引明细见audit.json。它证明实现有实质差异，不是本次场景AP归因实验。
- 本地项目、Dataset、Downloads及挂载盘文件检索未找到作者OpenYOLO3D.zip／replica_masks发布缓存；存在Open3DIS资产及压缩包，但它们不是作者论文候选，半采样点域不直接替用。

## 下一项复现应怎么做

优先恢复作者发布的完整点域Replica预计算掩码，在隔离目录核对8场景点数／点序／权重词表和官方入口，再按作者协议一次性评测；如需重新比较Full，应让基线与Full共享同一套已核验原生候选并分别重新封存，不能只把表格中的基线数替换为23.7。

若暂时无法恢复缓存，可做明确标注的“上游入口重生成”对照（固定seed、point cloud、隔离上游实现），它仍不保证论文数值精确一致。不得扫描NMS、分数或随机种子以追23.7，不改变冻结Full或开发SL模型。

本轮核查止于协议定位和缓存AP复算，尚未下载作者资产或执行新的预测／评测。
