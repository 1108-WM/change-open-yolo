# 候选关系、重复抑制与几何代表重建结果

2026-09-10。本批已完成实现、安装、official100 真实训练及回放审计。实际 geometry-semantic 工作树新增 3 个工具模块、1 个测试文件和 1 份说明；安装后相关回归 **39 passed**，其中新增 17 项测试。没有提交或推送，没有新的 AP。

## 已完成

在既有 **15 组排除集合**下新增训练 **250 个模型**：60 个全 fit 基础分类器、175 个内部交叉拟合模型和 15 个 winner。复用上一批已经重建的连续质量头及 union 支持选择。每组均完成基础预测、158 维 winner 特征、winner 预测、track 抑制和 canonical 几何清单的回放。

winner 的训练特征来自 fit 场景内部再次留出得到的 q、valid25、valid50、same-target 和 track-better 预测。外部排除组、预留校准组和当前内部留出组均不进入对应基础模型拟合。推理使用全部 fit 场景拟合的模型，fit 场景预测明确标记为 in-sample，不能当作 OOF 用于后续训练。

原始质量输入与 native/track/GVC 记录及上一批快照逐字段比较；关系原始分数单独回放。受控候选由无 GT 关系成员和 native 原始分数选出，并与历史 coexist 成员集合比较。没有继承旧 action utility 或旧 OOF 学习分数。

| 核对对象 | 数量 |
|---|---:|
| 原始 native/track 候选 | 65,263 |
| 原始关系 | 4,786 |
| 受控候选 | 5,941 |
| 受控 track / 无关联 track | 3,958 / 1,308 |
| component-union 原始特征 | 3,958 × 40 |
| winner 特征 | 5,941 × 158 |
| 唯一几何 | 9,708 |
| 原始成员（包含 union） | 66,764 |
| 原样保留的 B1 语义成员 | 66,763 |

原始池比 B1 多出的 `scene0692_03/track16` 已按原点集补回几何清单，未伪造它的 B1 语义成员。历史 temporary refined 不进入本批已重建集合。全部 1,504 条合格 union 支持均重新合并 native/track 点集，与 1,501 个保存 union 逐点比较通过。

## 实际发现

**15 组的 canonical 来源及 candidate ID 完全相同。** 每组均为 native 2,941、track 5,266、pair_union 1,501；因此本次没有观察到代表来源切换。每组的新质量/抑制分数仍须分别传递，不能因此直接复用历史 FI1-A 特征。上一批发现的 union 父支持变化也仍存在：相同 union 点集的父角色会影响 C 细化。

native 原始几何分数全部保持原值；有关联 track 使用 `q*(1-(1-keep)^3)`，无关联 track 使用 q。采用部署端原始三次抑制公式，无 epsilon floor；历史 AP 导出另有 logit clipping。实际每组有 **0–24 条 track 分数低于 1e-6**，因此这个边界差别并非纯理论情况。没有将新输出描述为历史 AP 导出的逐字节复现，也没有修改历史 evaluator 或结果。

几何代表与语义成员分开保存。`canonical_member` 用于后续几何模型，`b1_members` 的类别、分数、列索引和 provenance 原样保留；不把一个几何的多个语义假设压成 top1。新增几何分数独立存储，当前没有替换 B1 的语义排序分数。

## 审计与边界

所有模型保存了实际拟合场景、行索引、矩阵/标签/权重指纹及内部作用域。审计从模型重算基础预测和 winner 特征，不以缓存特征矩阵自证。独立核对全部 15 组的 native 不变、track 单调抑制、无关联 track、几何代表规则和语义成员保留。

除合成污染测试外，还复制一个真实 official100 完成组，修改 keep 预测并同步更新该文件哈希；实际模型回放仍以 `model replay differs from saved prediction` 拒绝该产物。原模型和预测未修改。

保留历史 valid25/50 的 `>=` 目标及 incumbent 关系/winner 目标；新 A/C/逐类质量目标对齐 evaluator 的工作尚未完成。原始候选/GVC 生成、关系资格和全部 40 维原始几何标量未在本工具中从头重算，依赖已冻结原始记录。本批只读取已有 official100 训练标签，没有读取新的 GT 文件、validation60 或 val312。

**FI1-A/B/C/D、必要语义上游和最终逐类评分器仍待重建。** `training_ready`、`production_ready` 和整条嵌套管线就绪状态保持 false；不能把历史 refined 自动视为已隔离训练数据，也不能宣称实例分割 AP 已提升。

## 产物与接续

产物根为 `.local/fi1_nested_suppression_20260910/`：

- `nested_suppression/plan.json`：输入、代码、特征、目标和全部 15 组作用域。
- `nested_suppression/jobs/`：250 个新模型、全部预测、两种 winner 特征矩阵及 canonical 几何清单。
- `nested_suppression/completion.json`：15 组训练和模型回放全部完成。
- `independent_review.json`：分数/代表/语义核对及真实篡改探针。
- `installed_manifest.json`、`installed_regression.log`、`source_validation.json`：安装、39 项测试和源文件检查。
- `execution.json`：真实启动命令和各阶段成功退出记录。

下一步接入各组新几何分数和 union 父支持，重建 FI1-A，再继续 B/C/D；细化点集变化时重新核对缓存能否复用。最终逐类评分仍保留原标签图、bbox、完整 Alpha 和多个语义假设。

本批结束时，原视觉任务已完成 **92/92 场景、2,283/2,283 几何、333/333 代表目标**的逐场景导出/审计，但 `run/completion.json` 尚未出现，仍在特征生成/审计收尾。未重复启动导出，也未提前运行 Qwen；固定 6 目标的 Qwen GPU 集成继续以该完成标记和共享锁为前提。
