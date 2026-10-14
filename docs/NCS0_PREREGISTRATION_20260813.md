# NCS-0 geometry-node 保守分数交换预注册

预注册日期：2026-08-13

## 研究问题

NCS-0（Node-level Conservative Score Swap oracle）检验：在冻结 geometry、类别和 candidate membership 的条件下，若只允许每个 exact geometry node 在两个已经物化的 `(node,class)` 代表之间交换现有分数，是否存在稳定的开放词汇 AP 上界。

该分支针对 PCV-1 的失败机制：prediction-level 改类会在同一 exact geometry 的大量 native 类别副本上重复应用。NCS-0 不改类别，也不复制、生成或拟合任何新分数。

## 冻结输入

- official100 Z6c 无 GT semantic review ledger；
- Z3 `C_joint_native_track_union_frozen_score` 场景隔离 OOF 分数；
- 冻结 YOLO top-5 / Alpha top-5 union 类别集合；
- 固定 official100 五折 manifest；
- 原 official evaluator。

不重新运行 YOLO-World、Alpha-CLIP、DINOv2 或 VLM，不读取 safety60、even48 或 test60。

## 无 GT 动作空间合同

对每个 geometry node：

1. 按 class 折叠当前已物化 hypotheses；同类只取 hybrid score 最高者作为 canonical representative；
2. 分数并列时按 `native < track < pair_union`，再按 candidate id 升序确定性选择；
3. 所有 class representative 中最高分者为 incumbent；
4. alternative 必须同时满足：
   - class 不同于 incumbent；
   - class 位于冻结 YOLO/Alpha top-5 union；
   - class 已有实际 current hypothesis representative；
   - alternative score 严格低于 incumbent score；
5. 每条预注册动作只交换 incumbent representative 与一个 alternative representative 的分数。

动作不得创建缺失类别 prediction，不得改变 mask、class、candidate membership、candidate count 或除两个端点外的分数。每条动作的 pair、node 和 scene score multiset 都必须严格守恒。无 GT 阶段不选择动作、不训练、不计算 AP。

## GT-only oracle 选择合同

无 GT 动作账本通过后，只允许一次 official100 GT-only oracle：

- 每个 geometry node 最多选择一个动作；
- 以 incumbent geometry 的最佳 official GT IoU 和类别作为 node target；
- 只有最佳 IoU `>=0.50`、incumbent class 错误、且 target class 对应一个预注册 alternative 时，才选择与 target-class representative 的 score swap；
- incumbent 已正确、geometry 不达 TP50、target 不在已物化预注册 alternative 中时保持 NO_OP；
- GT 只用于 oracle 选择与评测，不写入任何 no-GT 推理计划。

该规则不逐动作试算 AP，不做 greedy AP 搜索，不扫描 IoU、分数差或候选范围。

## 推进门槛

只有 oracle 同时满足以下条件，才允许未来另行预注册 NCS-1 no-GT node selector：

- AP、AP50、AP25 相对 Z3 control 三项严格为正；
- 主 AP 至少 `4/5` 折严格为正；
- head、common、tail AP 相对 control 均不下降；
- oracle 动作覆盖至少 50 个场景；
- 单场景不超过全部 oracle 动作的 10%；
- 单 target class 不超过全部 oracle 动作的 20%。

任一门槛失败，NCS 分支停在 oracle，不训练 selector，不扫描阈值、动作范围、top-k、代表选择、score mixing 或交换比例。

## 明确不做

- 不修改或重跑 PCV-1；
- 不把 score swap 改成 score copy、boost、blend 或重新校准；
- 不把未物化的 YOLO/Alpha 类别创建为新 prediction；
- 不在 safety60/even48/test60 上选择设计；
- 不与 Z6f VLM 组合。
