# 有效运行与复现入口

2026-10-14，项目根目录；CPU `.venvs/yoloe_gpu/bin/python`，`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4`。不运行旧总控或历史finalize。

有效顺序：

1. `research/geometry/code/tools/run_fi1_query_relation_diagnostic.py prepare`：确认任务1completion存在，固定52场景/207原采样T父及无GT字段。
2. 同脚本 `generate`：全部图/帧封存，`relations_frozen.json`记录GT_used=false；源文件与原自动mask资产hash。
3. `research/geometry/code/tools/evaluate_fi1_query_relation_diagnostic.py evaluate`：封存后GT关系/供给参照，仅局部/存储ROI组成及完整场景体积。
4. 同脚本 `analyze`：全部关系分层、物理场景macro/200次场景bootstrap；没有拟合分组规则或阈值搜索。
5. `research/geometry/code/tools/verify_fi1_query_relation_diagnostic.py`：候选集合/硬mask/局部与ROI点数/独立帧逻辑/暴力距离核验。有效日志`verify_effective_v2.log`；均值的独立逐列求和允许1e-12，图生成与GT标签未改变。第一次核验早于analysis seal结束被阻止，之后完整有效核验才计完成。
6. `research/geometry/code/tools/supplement_fi1_query_relation_supply.py`：独立逐点计数重算父模式IoU和完整NT目标模式，绑定具体漏块与关系证据。
7. `research/geometry/code/tools/finalize_fi1_query_relation_diagnostic.py`：仅本目录报告/回执/源码快照，不写历史状态或阅读入口。

共52已暴露物理场景，无最终holdout，无训练/新增帧/完整实例树/生产候选，未提交GitHub。各参考GT后置构造不是推理结果，尤其不能以候选GT最佳单query/锚对/纯块证明无GT可安全选择。

“局部同目标”不作全局身份证明：完整场景逐query点集没有存储，ROI外硬点只报体积缺口；目标/其他/未知分别保留。低重叠互补、重复预测、邻物和小支持分别统计。局部帧参考为固定最近父成员区域的锚支持子集，空时missing不回退，不是一个已验证的身份关联算法。
