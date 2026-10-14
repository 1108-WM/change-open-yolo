# 2026-10-14 活跃代码与旧.local深度精简

本轮用户授权抽取被现用代码引用的公式，删除退役完整脚本；Full集中，有效SL和当前几何独立保留。本轮实际执行删除，不只移动入口。Full配置仍为GD10 / official100七模型D-v3 / 匹配597 B26 / NTUU′ / K800；未重训、未调参、未启动几何下一实验。

## 源码与测试

- 38个完整旧入口归档后删除；29个精简公共模块仅保留实际调用所需定义，9个旧文件完全没有活跃副本。21,203行→4,778行，退出16,425行（约77.47%）。不是删除38个文件后新建38个同内容包装。
- 公共模块集中在`full_bundle/code/tools/fi1_shared/`；沿用来源文件名，便于核对旧公式，不包含原实验main/CLI。133个函数/类的AST经导入路径归一化后保持一致。项目根定位增加一层，未修改公式。
- 删除26个退役实验测试；41个仍有用的测试从几何目录归回Full（29）和SL（12），保留其验证内容。
- 修正两项迁移后发现的问题：Alpha调用测试补显式零padding参数；SL训练标签守卫两侧统一resolve再比较，访问范围保持。
- 当前Full实现126个（tools92顶层+29公共模块+包初始化，共122；另4个utils）、测试35个；SL实现44／测试17；几何实现107／测试28；必要自身语义snapshot24源码、无历史tests。

逐项见[源码执行清单](shared_slice_executed.json)、[定义核验](slice_verified.json)、[测试归属](test_partition.json)、[当前文件名与hash](current_code_index.csv)、[静态导入边](code_edges.json)。保留源码不等于每个历史CLI都属于最小推理链，未宣称已达到严格数学意义的最小依赖集。旧原文归档`archive/retired_20261014/deep_legacy_sources_before_slice.tar.gz`。

## .local

核查231个真实目录；集中目录的兼容链接不作为删除对象。预删除[目录清单](local_inventory.csv)、[引用证据](local_reference_evidence.json)和[审计边界](local_audit_summary.json)保持为执行时点记录。检查当前metadata、scene routes、verification frame-index、入向链接和活跃代码默认路径；大于8MiB metadata未全读，故“无直接引用”只是候选证据，不能单凭它全删。

本轮手工核定42个根，文件分配块9.248505 GiB，130,893个实际文件。退役GD15/GD20对照、旧A/B/C或Q29评分对照、单场景smoke及早期类别覆盖试验。结论／身份／源码快照／汇总评测等1,791文件压缩归档约505.39MiB并逐项核对hash，之后删除旧根与大产物。当前正式GD10、597监督及B26、SL输入、10月11—13日当前几何模型/预测/封存保留。

| 用户举例 | 用途 | 本轮处理 |
|---|---|---|
| `fi1_val48_ad_full597_q_20260918_v2` | 旧A/D与Q29对照，约108.88MiB；不是当前Full权重 | 报告/身份/汇总评测归档，大产物及根目录删除 |
| `fi1_val48_grounded_stride10_backend_smoke_20260918_v1` | 单场景后端smoke，约8.90MiB；正式20260919后端继续保留 | 身份/摘要/源码快照归档，smoke产物及根目录删除 |
| `fi1_v33_class_coverage_audit_gt_20260909` | 早期22几何的GT类别覆盖审计，仅摘要 | 摘要归档，根目录退出.local |

实际删除项、保留理由、历史引用区别见[核定清单](local_retirement_plan.json)和[执行回执](local_deletion_executed.json)。历史报告旧路径仍代表当时身份；退役输出不再支持完整重算，不重写旧contract/seal。追溯已归档内容用[归档文件与hash](local_history_capsule_manifest.json)，对应`archive/retired_20261014/deep_local_history_capsule.tar.gz`。其余目录中的无绑定候选继续保留待逐项核定，未把审计的33.47GiB机会值当已释放量。

## 验证与空间

480项测试及4项子测试通过（含分目录后的再次验证）；281项公式/结果核对通过，Full两个场景全池分数和K800行逐值相同，SL一个场景类别/去类结果相同。sklearn1.3.0环境9后端模块、7个D-v3权重及champion加载通过。独立Full925发布hash保持；删除后964场景路由和972帧合同核对见[最终资产回执](assets_verified.json)。未完整重跑597训练、312 GPU前端或外部A100。

空间以`df`和最终回执为准：本轮开始99.36GiB，删除后约108.12GiB；包含归档与同时发生的其他磁盘写入后，本轮净增约8.77GiB。前两轮净增约47.78GiB，三轮约56.55GiB；这是所在文件系统的空闲变化，不能称作进程内存减少。项目约定保留50GiB余量后，约58.12GiB可用于开发。

当前各目录采用不穿透链接的`du`统计（取整）：`.local`51GiB、Full54GiB、output11GiB、SL3.3GiB、几何5.5GiB、archive17GiB。Full含从旧pretrained/output迁入的真实资产，前后目录体积不能直接相加算释放；跨盘datasets/artifacts入口是链接。
