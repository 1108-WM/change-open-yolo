# NCS-6 多来源候选效用路由预注册

预注册日期：2026-08-16

## 研究问题

NCS-4 SigLIP 与 NCS-5 ProxyCLIP 都产生了充足、分散的已有代表内反排序动作，但各自唯一 GT-only audit 均失败。NCS-6 不再训练“四选一来源分类器”，而是先检验一个更基础的问题：在同一 exact-geometry node 上，YOLO、Alpha-CLIP、SigLIP 与 ProxyCLIP 给出的**已物化类别代表**是否具有跨场景、跨五折且非单一类别集中的纠错互补性。

只有 GT-only union oracle 显示显著且稳定的专家互补上界，后续才允许训练低容量、严格 abstain 的 candidate utility router。router 的目标将是对每个 `(node, candidate class)` 预测 `improve / harm / neutral`，而不是预测应该信任哪个模型来源。

## 当前阶段边界

本预注册只授权两个顺序阶段：

1. no-GT source candidate inventory；
2. 一次 train100 GT-only source complementarity oracle。

本阶段不训练模型、不生成 score-swap plan、不运行 AP、不读取 NCS-validation60。NCS-4/NCS-5 的模型、视角、mask、prompt、SMS 和 proposal 规则保持冻结，不重跑 embedding，也不根据本审计结果回调。

## 冻结数据与 provenance

- train100 split：`output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100.txt`，SHA-256 `ecebf3a682a02665a9770c2fa0ef83cc61c0c4ec2f185820aa44d667c8c4cd40`；
- 五折：`output/scannet200/scene_splits/ncs_independent_20260813/ncs_train100_folds_v1.json`，SHA-256 `e51f6aad4dcc20b758dc7cdde73ff9047c2a495da85ce3e1587a21290a2374e5`；
- Z1 native bindings：`/media/jia/软件1/scannet_train_stream/ledgers_ncs/z1_native_train100/candidate_bindings.jsonl`，SHA-256 `0f7a34055f786ad8fd66cf456ce68f78118fe5722359e295e15a4bc5e1af6332`；
- Z2c nodes/distributions：SHA-256 分别为 `44d0a82b22a02eebfc0ed272e55d8f3c68fcee03cdad45738d6b5a685e88dac1` 与 `e8eacdfef3aa91e22de3dbc66e7ad477bca5a3bd1ae36ed24295130066fe755f`；
- NCS-4 proposals：SHA-256 `e5551e4fc81c69e29b91e00780ccdd7b220a4b39247a3a60f561ce03dd4ccb69`；
- NCS-5 proposals：SHA-256 `caca50edd837fe1ba69d5086f830ffc88550cdd744697789480291dd0f79edf1`；
- NCS-1 frozen quality pairs：SHA-256 `177fcba63797f54bbcf09d6943772ba9340bc91b94185dfc561cdf2f1530ee1f`；
- NCS-4 GT audit rows：SHA-256 `e8d08eeb6c80468a4f7661705f78b792406ec21cccc1361210e195dff5fe592b`；
- NCS-5 GT audit rows：SHA-256 `28cae0b120b1b4b873e9184b9a3474428311614d8ce827bbe086f1949280d136`。

任何冻结输入 hash、节点数、scene/fold join 或 mutation 合同不匹配时立即停止。

## 来源与候选定义

统一对象是 train100 的 `2,946` 个 Z1 native exact-geometry nodes。每个 node 先按 class 折叠 native bindings：同类代表取 native score 最高者，分数并列取 candidate id 最小者；incumbent 是所有代表中 native score 最高者，同分取 candidate id 最小者。

候选来源固定如下：

- `yolo`：Z2c `geometry_yolo` 的 198 类 argmax，float32 精确并列取最小 class index；
- `alphaclip`：Z2c `geometry_alpha` 的 198 类 argmax，仅在 `geometry_alpha_available=true` 时有效，精确并列取最小 class index；
- `siglip`：只复用 NCS-4 `actionable_proposals.jsonl` 已冻结 alternative；不存在 proposal 的 node 显式 missing；
- `proxyclip`：只复用 NCS-5 `actionable_proposals.jsonl` 已冻结 alternative；不存在 proposal 的 node 显式 missing；
- `mllm`：本阶段显式 `unavailable_not_run`。NCS-train100 尚无预注册且冻结的 MLLM candidate ledger，不允许移植 official100 的 Z6f 输出。

来源推荐若等于 incumbent，记录为 no-op；若指向其他 class，则必须已经物化为该 node 的 canonical representative，且其 native score 严格低于 incumbent 才是 actionable。每个 node 的 union candidate set 按 class 去重，并保留全部 proposing sources。候选 membership、class、geometry 与原 score 均不可修改。

## no-GT inventory 审计门槛

inventory 必须满足：

1. `100 scenes / 2,946 nodes` 完整，duplicate/missing node 为 0；
2. Z1/Z2c/NCS-4/NCS-5 输入 hash 精确匹配；
3. 每个 source recommendation 均可追溯到冻结输入；
4. 每个 actionable alternative 已物化、与 incumbent exact geometry 相同且原 native score 严格更低；
5. mutation 全为 0；
6. `ground_truth_usage=none`、`validation60_read=false`、`ap_run=false`、`model_fit=false`。

只有 inventory 全部通过，才允许 GT-only oracle。

## 唯一 GT-only source complementarity oracle

质量定义固定复用 NCS-1：类别代表对 official IoU thresholds `0.50:0.05:0.95` 的 mean eligibility。NCS-1 已登记 endpoint 直接复用；SigLIP/ProxyCLIP 中不在 NCS-1 pairs 的 endpoint 只允许复用各自已完成审计的冻结 replay 值。本阶段不重新读取 raw GT，不产生新质量定义。

对每个 actionable source candidate，相对 incumbent 严格标记：

- `improve`：alternative quality 更高；
- `harm`：alternative quality 更低；
- `neutral`：相等。

union oracle 在 incumbent 与所有 unique actionable alternatives 中选择最高 GT quality；与 incumbent 并列时固定 no-op。GT 选择只写诊断账本，不写 inference plan。

必须报告：

- 每来源与每折 improve/harm/neutral；
- 来源两两 proposal/improve 重叠；
- 每来源 exclusive improve nodes；
- union oracle 的 improve node 数、positive quality gain、五折/场景/类别覆盖；
- union 相对最佳单来源的增量；
- oracle target class concentration。

## 进入 router 训练的联合门槛

以下门槛必须全部满足：

1. quality/provenance/materialization join 全部有效；
2. union oracle 至少覆盖 `100` 个 strict-improve nodes、`25` 个 scenes、`20` 个 target classes；
3. 五折各至少 `10` 个 strict-improve nodes；
4. 五折中 union strict-improve node 数均严格大于该折最佳单来源；
5. 全量 union strict-improve node 数至少为最佳单来源的 `1.15x`；
6. 至少两个非 incumbent 来源各有不少于 `10` 个 exclusive improve nodes；
7. oracle target class 最大占比 `<=20%`。

任一门槛失败即封闭 NCS-6：不训练 router、不调用 MLLM、不运行 AP、不读取 validation60，也不扫描 source subset、top-k、阈值、融合权重或质量定义。

## 若门槛通过后的未来结构（本阶段未授权）

后续必须另写预注册合同，使用场景隔离五折 OOF、低容量模型，对每个 `(node, candidate class)` 预测 `P(improve) / P(harm) / P(neutral)`。每 node 最多一个动作；仅当 `P(improve) > 0.5` 且严格为最大状态时接受，否则 no-op。动作仍只能交换 incumbent 与所选已有 alternative 的原 native scores，并保持 node/scene/global score multiset 守恒。

MLLM 只有在上述现有来源 union 门槛通过后，才可针对冻结的高冲突 node 子集另行预注册；不得全量运行，也不得把自由文本答案直接创建为新类别 candidate。

## 执行结果（预注册后记录）

no-GT inventory 已完成：`100 scenes / 2,946 nodes / 5,172 unique actionable candidates`，全部结构门槛通过。唯一 GT-only oracle 的来源 strict improve 为 YOLO `53`、Alpha-CLIP `300`、SigLIP `191`、ProxyCLIP `136`；union 为 `461`，五折 `93/71/121/72/104`，相对最佳 Alpha-CLIP 的全量比为 `1.5367x`，来源互补性成立。

但 union target class 中 class 0 `chair` 为 `119/461=25.8134%`，超过预注册 `20%` 上限。其余门槛全部通过，最终 `advancement_authorized=false`，`next_stage=stop_ncs6_no_router_no_mllm_no_ap`。正式输出：

```text
docs/diagnostics/ncs6_source_candidate_inventory_train100_20260816/
docs/diagnostics/ncs6_source_complementarity_oracle_gt_train100_20260816/
```
