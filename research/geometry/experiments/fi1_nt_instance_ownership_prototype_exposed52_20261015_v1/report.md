# N/T局部实例关系与三态块归属原型

2026-10-15；已完成开发测试，未通过预先固定验收，停止本原型，不进入独立场景/AP。范围仅52个已暴露物理场景、207个采样T父、原1,308块。未训练，原N/T保留。

## 本轮具体改变

承认前一轮只测试全部块和裁剪，没有实现归属选择器，不能据此认定现有证据不可学习。纯目标块98.40%保留也不能掩盖23个混合块中49.09%目标补点被丢掉；本轮不把裁剪视为已解决。

本轮已实现真实accept/reject/unknown，代码[fi1_nt_instance_ownership_core.py](../../code/tools/fi1_nt_instance_ownership_core.py)。query完全不进入身份图、组或接收条件，只复用前轮每块固定query及相同≥0.5新增点裁剪。拒绝和无法确定均不编辑，原父点从不删除。

关系图使用完整原N/T点集，含所有原候选源，无类别/分数过滤。保存全场景交集矩阵、方向覆盖、重复组以及每父/块的全部候选覆盖。局部节点为与原halo相交或属于原≤0.15m邻居的候选，不再仅取前4个邻居；原halo、分块、query集合、六帧观测本身均不改变。

固定最小规则（GT前封存于[contract.json](contract.json)）：几何key顺序进行complete-link分组，同组每对候选互相覆盖≥一半；不使用传递连通分量把两个不重叠物体经大N粘在一起。只有组内存在点数≥100的不同N和T点集时才作为暂定实例参照；同点集的两个来源别名不算独立证据。“可靠”只是待检验的跨来源几何假设。

某组任一成员覆盖父/块≥一半，该组就对其提出归属。父和块均恰有一个组且相同→接受；均恰有一个但不同→拒绝；缺参照或多组竞争→无法确定。同一T合并所有接收块的原裁剪点，最多产生一个副本。没有query父覆盖、query重叠或共同帧硬门槛，没有GT选取实例参照、父、块或点。

## 固定输出结果

所有规则/证据/完整52场景输出封存后才读取本轮GT。全部143稳定父、14含糊父、33超评估类别范围父、17未标注父保留。非稳定父不强行分配GT目标，其目标/邻物标签为null，已知/未知点另记。

| 对照 | 动作父数 | 完整NT新增目标≥.02 | 安全增量场景 | 好副本损失>.02 | 稳定父新增目标精度 | 非稳定父动作 |
|---|---:|---:|---:|---:|---:|---:|
| 全部块原query裁剪 | 205 | 2 | 2 | 9/12 | 51.67% | 63 |
| N/T归属后同样裁剪 | 66 | 0 | 0 | 0/12 | 95.92% | 11 |

“安全增量场景”是后置诊断：同父相对完整NT增量≥.02且新增目标占比≥.9，未知仍计入分母。不是推理条件，更不是GT回选输出。“好副本”指原IoU≥.9的T产生的修补副本；原T仍保留，未声称发生下游损害。完整NT池始终保留原候选，几何上限不会因追加差副本下降。

| 三态输出 | 块数 | GT归属构成 | 稳定父裁剪点（目标/邻物/未知分别统计） |
|---|---:|---|---|
| accept | 187 | {'mixed_known': 2, 'other_only': 6, 'target_only': 137, 'target_unknown': 3, 'unknown_only': 2, 'unresolved_parent': 37} | {'other': 353, 'points': 10445, 'target': 10019, 'unknown': 73} |
| reject | 95 | {'mixed_known': 1, 'other_only': 78, 'other_unknown': 2, 'target_only': 1, 'unknown_only': 4, 'unresolved_parent': 9} | {'other': 5388, 'points': 5743, 'target': 15, 'unknown': 340} |
| unknown | 1026 | {'mixed_known': 20, 'other_only': 307, 'other_unknown': 27, 'target_only': 203, 'target_unknown': 12, 'unknown_only': 39, 'unresolved_parent': 418} | {'other': 26221, 'points': 54583, 'target': 26533, 'unknown': 1829} |

全量每父连续IoU、pool_delta、误补、原点删除、未知、非稳定父动作及cohort分层见[summary.json](summary.json)、evaluation/{scene}/unions.json。当前原型完全固定，没有查看GT后挑父或只保留获益块。

## 机会与冗余必须一起检查

下表覆盖点级oracle有价值的所有父；其余负例同样完整评测。数值为实际union对同父目标IoU减完整NT最佳IoU。无动作也可能显示负数，因为原父已不如完整NT；负数不自动表示本次损坏，须同时查parent_delta。

| 场景/父 | 失败基线pool_delta | 新原型pool_delta | 原型块状态 |
|---|---:|---:|---|
| scene0002_00/c006 | -0.170915 | +0.000000 | {'unknown': 4} |
| scene0057_00/c005 | +0.154800 | +0.000000 | {'unknown': 4} |
| scene0078_00/c004 | -0.108988 | -0.020216 | {'accept': 9, 'unknown': 5} |
| scene0085_00/c007 | -0.191029 | +0.000000 | {'unknown': 10} |
| scene0301_02/c004 | +0.196347 | +0.000000 | {'unknown': 3} |
| scene0301_02/c007 | +0.270046 | -0.000510 | {'unknown': 4} |
| scene0369_00/c006 | -0.026576 | +0.000000 | {'unknown': 13} |
| scene0369_00/c007 | -0.225004 | -0.022454 | {'accept': 2, 'reject': 3, 'unknown': 4} |
| scene0688_00/c006 | +0.011995 | +0.000000 | {'unknown': 4} |

N只提供参照，不能作为正确答案。对每个实际接收块的“同一原N同时多数覆盖父与块”连接，后置统计该N完整点集的目标/其他/未知组成，并比较修补副本、连接N、全部N及完整NT的IoU。native_bridge_audit共55个父/N连接，其中17个包含多个已知实例，1个含>10%已知邻物；这不是推理筛选条件。稳定父实际动作中，52个副本不如或等于已有最佳N，52个不如或等于直接连接N。身份解释不等于新增池价值。

## 预先验收与停止边界

[预定验收结果](summary.json)为{'annotation_unknown': True, 'good_copy_protection': True, 'multiscene_safe_gain': False, 'no_unresolved_actions': False, 'stable_add_precision': True}。继续条件为多物理场景安全池增量、保护全部好副本、稳定父新增精度≥.9、未知比例≤.1及无未解决父身份的动作；这些是保守开发验收口径，不是GT推理门槛。少操作或零操作不能通过。

本轮未通过，停止当前“complete-link重复分组＋跨源参照＋唯一归属”固定构造。不继续调同一阈值，不开展独立验证/AP，也不扩大原修复器训练。失败不证明N/T关系无用或原始信息不能训练归属；本轮分组可能将同物体拆成多个竞争组，N/T一致也可能共同错误。下一优先级按用户建议是针对争议区域设计新增观测或追查原T逐帧来源，须另行明确封存；本轮没有擅自增加帧或运行新观测。

只有以后明确规则先通过开发风险验收，才冻结未参与设计的物理隔离开发场景（排除52场景及变体，不动holdout），验证通过后再计算自身语义与冻结下游AP。

## 代码与核验

4项针对性测试通过，独立核验：[verification.json](verification.json)。GT前代码快照在code_snapshot/，证据/预测在predictions/，GT后标签在evaluation/。baseline all_C与上一轮所有关键池/损害/点数逐项一致。执行顺序：prepare → generate → evaluate → verify_fi1_nt_instance_ownership.py → report；使用.venvs/yoloe_gpu/bin/python，OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4。已有seal禁止覆盖重跑；从头复现另建目录。

本轮未修改生产、阅读入口或GitHub，无训练、GPU语义或AP。全部原父点保留。
