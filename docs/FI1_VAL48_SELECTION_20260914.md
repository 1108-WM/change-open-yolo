# val48固定场景选择结果

**已固定48扫描／48不同主场景，6个执行批次，每批8个。新前端推理和新AP尚未运行。**

选择先冻结规则，再执行：主场景固定种子哈希排序，各取字典序首个扫描；仅使用名称，不读取AP、GT、类别或资源质量。输入顺序反转及独立规则复算一致，冻结输出重读校验通过。

seed：`fi1-val48-full-pipeline-20260914-v1`。规则及整合后的N/A/B/C/D＋H实验设计见[协议](FI1_VAL48_FRONTEND_FULL_PIPELINE_PROTOCOL_20260914.md)。

| 批次 | 扫描（每个来自不同主场景） |
|---|---|
| 1 | scene0153_00, scene0699_00, scene0689_00, scene0574_00, scene0670_00, scene0251_00, scene0671_00, scene0342_00 |
| 2 | scene0535_00, scene0221_00, scene0088_00, scene0304_00, scene0100_00, scene0578_00, scene0648_00, scene0643_00 |
| 3 | scene0095_00, scene0527_00, scene0231_00, scene0496_00, scene0164_00, scene0616_00, scene0187_00, scene0690_00 |
| 4 | scene0702_00, scene0356_00, scene0629_00, scene0552_00, scene0700_00, scene0081_00, scene0633_00, scene0462_00 |
| 5 | scene0684_00, scene0621_00, scene0651_00, scene0169_00, scene0658_00, scene0208_00, scene0019_00, scene0278_00 |
| 6 | scene0353_00, scene0549_00, scene0568_00, scene0430_00, scene0474_00, scene0316_00, scene0222_00, scene0458_00 |

其余264扫描中，54个属于已选48主场景，另210个来自94个不同主场景。前54个不能当作主场景独立留出；整个312此前均已用于开发分析，其余210也不称未接触盲测。

与当前200训练、55开发和5诊断主场景无交集。批次只是执行分组，不是交叉验证折。不保证所选48覆盖所有类别或完整312难度。

机器清单：`.local/fi1_val48_frontend_full_pipeline_20260914/scenes48.txt`；`selection.json`保留142主场景完整哈希排名；`selection_receipt.json`绑定输入、协议、源码和全部选择输出。资源缺失不替换场景。
