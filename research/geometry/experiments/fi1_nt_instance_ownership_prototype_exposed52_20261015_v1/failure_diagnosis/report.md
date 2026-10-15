# 原型失败定位（后置诊断，不改规则）

本轮有明确归属选择器，其结果是减少污染而未保住完整NT增量。当前判断只针对本轮complete-link/跨来源参照/唯一归属构造。

全部无法确定块：{'no_reference': 1009, 'competing_references': 17}。点级oracle机会贡献32块中：{'unknown': 21, 'accept': 11}；具体原因{'block_reference_missing': 5, 'parent_reference_missing': 16, 'accepted': 11}。纯整块oracle机会贡献21块中：{'unknown': 16, 'accept': 11}；具体原因{'block_reference_missing': 4, 'parent_reference_missing': 12, 'accepted': 11}。

特别是，“缺少合格参照”与“原N/T根本没有联系”不同：32个机会贡献块里，11个本来存在原N同时多数覆盖父和块；0个虽有原N/T连接却未被接收。逐块保存原始支持节点、跨源资格、组和连续覆盖，不能把门槛/分组失败概括成证据不足。

- scene0002/c006/r1（真碎片）：父有1个参照组，块没有，输出unknown；没有因为query父覆盖0而直接拒绝，但N/T参照构造仍未找回它。
- scene0085/c007及scene0301/c004,c007：父缺少合格跨源参照组，关键块无法接收，原baseline的收益也被一并过滤。
- scene0078/c004：接收9块仍无池收益，关键纯目标r11的177点因缺块参照而遗漏。须保留所有接收块联合评测，不能GT补回这一块。
- scene0369/c006/r4：块有参照组而父无，unknown；其余小目标块也没有合格联系。
- scene0688/c006/r1：父/块均缺合格组，unknown。前轮单块裁剪确有+.042515增量但47/147未知；仍不能把该块视作已安全解决。

大N风险仍存在：接受集合有8个稳定父块裁剪后带邻物点，明细accepted_neighbor_pollution.json记录实际连接N的完整GT组成。不要把N/T一致作为GT真值。11个非稳定父产生动作也完整保留，不能GT后将它们删除来报安全。

55个稳定父有实际动作，其中52个修补副本不如或等于已有最佳N，且52个不如或等于实际连接N。关系更容易解释已有N已经覆盖的部分，这就是当前完整NT收益消失的直接限制。并非所有身份正确的补点都值得追加。

本轮停止；不事后改组、放松参照或重新选阈值。若开展下一项观测研究，先以本轮所有争议/缺参照块固定目标表，包含邻物负例与混杂/身份不明父；优先追查原T逐帧关联来源，区分缺帧、遮挡、自动mask碎片化与真实冲突，再单独封存新增帧查找/观测协议。此处未启动新帧或新训练。

量化文件：[summary.json](summary.json)、[机会块原始联系](valuable_blocks.json)、[误补连接N](accepted_neighbor_pollution.json)、[N冗余对照](native_redundancy.json)。
