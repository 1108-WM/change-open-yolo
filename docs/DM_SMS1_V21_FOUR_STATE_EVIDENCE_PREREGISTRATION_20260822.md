# 第二创新点 v2.1：目标指向四态类别证据预注册合同

状态：`frozen_before_formal_vlm_inference_or_ap`

日期：2026-08-22

## 1. 启动原因

第二创新点 v2 已按预注册对 NCS-train100 的 7,006 个双候选节点完成无真实标注重放和独立审计。结果为：

- 严格有效旧证据 6,112 条；
- 旧 DM-SMS-1 换类 1,827 条；
- v2 换类 0 条；
- 6,112 条有效证据中，冻结原类别 AB/BA 的 `strong_counterevidence` 全部为 `false/false`；
- 全部 24,448 个候选顺序判断中只有 1 个 `strong_counterevidence=true`。

因此，v2 的零动作不是决定器连接错误，而是旧类别证据合同没有形成可用的原类别强反证。v2.1 只修补类别证据生成与解释合同，不放宽 v2 的换类门。

## 2. 第一创新点绑定与不可修改项

v2.1 的几何底座继续严格绑定 FI1-Legacy，只读复用：

- geometry；
- mask；
- candidate membership；
- ranking score。

FI1-Legacy 本身不得修改、覆盖或重跑。FI1-D-v3 不得接入、修改或运行。现有 DM-SMS-1、v2 代码、原始 Qwen 输出、train100/validation60 安全账本和 AP 结果全部冻结，不得覆盖。

v2.1 只允许新建独立代码、测试、输入账本、模型输出目录和审计目录。所有正式目录必须使用 `dm_sms1_v21_` 前缀，并拒绝覆盖非空目录。

## 3. 数据、模型和当前权限

固定数据范围：

- 只允许 NCS-train100；
- NCS-validation60 继续封闭；
- ScanNet200 val312 不读取、不审查、不调参；
- 不使用 official100、safety60、even48 或 test60 选择规则。

固定模型仍为本地：

```text
Qwen2.5-VL-7B-Instruct
revision = cc594898137f460bfe9f0759e9844b3ce807cfb5
min_pixels = 100352
max_pixels = 200704
do_sample = false
```

固定视觉输入、图像合成和无类别属性证据完全沿用已冻结 DM-SMS-1：每个任务 1～3 张双栏图，左侧完整场景红框、右侧同视角局部裁剪；视角选择、顺序、SAM 框、RGB 和属性输出均不改变。

当前阶段允许预注册、实现、输入账本生成、语法/纯函数/合成测试和无模型审计。完整 7,006 条 Qwen 推理属于正式图形处理器账本，只有本合同、实现、输入账本与审计全部冻结通过后，才可根据用户明确授权启动。本合同不授权 GT 或 AP。

## 4. 四态证据语义

每个候选在每种顺序中必须且只能选择下列一个 `evidence_state`：

### 4.1 `direct_support`

表示红框目标本身存在直接、清晰、可见、候选特异且跨视角一致的正证据。

结构要求：

- 至少两个不同的登记视角；
- 每个视角分别保存简短英文观察；
- `support_observations` 非空；
- `counter_observations` 为空；
- `unknown_reasons` 为空；
- 不能只依据场景共现、周围物体、候选名称或常识。

### 4.2 `direct_strong_counterevidence`

表示红框目标本身在至少两个不同登记视角中清晰显示出与该候选定义直接矛盾的可观察结构、材质、部件或功能特征。

结构要求：

- 至少两个不同的登记视角；
- 每个视角分别保存具体矛盾观察；
- `counter_observations` 非空；
- `support_observations` 为空；
- `unknown_reasons` 为空。

下列内容永远不能成为直接强反证：

- 没有看到候选特征；
- 目标太小、模糊、遮挡或视角不足；
- 属性证据写了 `unknown`；
- 候选名称与场景不常共现；
- 只看到环境或相邻物体；
- 候选集合可能不完整。

### 4.3 `unknown`

表示现有视角不足以形成直接支持或直接强反证。固定 `unknown_reasons` 枚举为：

```text
occluded
blurred_or_too_small
insufficient_visible_detail
single_view_only
cross_view_conflict
environment_only
attribute_evidence_unclear
other_unclear
```

`unknown` 必须至少保存一个原因，且支持/反证观察数组均为空。

### 4.4 `candidate_set_insufficient_or_unclear`

表示视觉属性明确无法由冻结的两个候选安全表达，或候选集合不足以作出二选一判断。此状态不能新增第三类别。

当任一候选使用该状态时：

- 同一顺序的两个候选都必须使用该状态；
- 顶层 `candidate_set_insufficient_or_unclear=true`；
- 两个候选的支持/反证观察数组均为空；
- `unknown_reasons` 必须只包含 `candidate_set_insufficient_or_unclear`；
- 整个任务安全保持冻结原类别。

## 5. 严格输出结构

每种候选顺序必须返回且只返回一个完整顶层 JSON 对象，禁止数组包裹、多个对象、截断对象、对象后说明或 Markdown。

顶层字段必须精确为：

```text
candidate_set_insufficient_or_unclear
candidate_set_reason
candidate_results
```

`candidate_results` 必须严格保持登记顺序。每项字段必须精确为：

```text
class_index
evidence_state
support_observations
counter_observations
unknown_reasons
confidence
```

每个观察对象字段必须精确为：

```text
view_rank
observation
```

观察文字统一为简短英文；类别名称只出现在候选映射中，不能出现在观察文字、集合原因或未知原因中。`confidence` 为有限 `[0,1]` 数，仅用于记录和审计，不进入决定门。

JSON 或结构错误最多使用固定重写一次。语义违规、类别词泄漏、候选集合外类别、身份或顺序错误不得通过自由改写绕过；最终不合法时安全回退。

## 6. AB/BA 与决定规则

每个双候选节点必须运行完整 AB 和完全反序 BA。换类固定要求：

```text
替代类 AB = direct_support
+ 替代类 BA = direct_support
+ 替代类 AB/BA 支持视角集合至少有一个交集
+ 原类别 AB = direct_strong_counterevidence
+ 原类别 BA = direct_strong_counterevidence
+ 原类别 AB/BA 反证视角集合至少有一个交集
+ 两种顺序 candidate_set_insufficient_or_unclear 均为 false
→ 换类

其他情况
→ 保持冻结原类别
```

不扫描视角交集数量、confidence 阈值、文本相似度或其他门槛。任何单视角任务无法满足直接跨视角支持或强反证，必须 no-op。

## 7. 输入账本合同

正式 v2.1 输入账本只包含现有 7,006 个双候选任务，并冻结：

- task/scene/geometry 身份；
- FI1-Legacy 冻结原类别与分数；
- 两个有限候选及来源；
- AB/BA 完全反序；
- 1～3 个登记视角编号及原资产引用；
- 已冻结无类别属性任务引用；
- 四态响应模式、提示合同版本和决定规则；
- 输入候选清单、属性清单与类别配置的 SHA-256。

输入账本阶段不读取图像内容、不调用模型、不读取 GT、不运行 AP。

## 8. 无模型推进门槛

正式 Qwen 推理前必须同时满足：

1. 预注册已冻结；
2. v2.1 核心验证器、决定器和独立审计器均为新文件；
3. 合成测试覆盖四态、单视角 no-op、AB/BA 错序、视角重复、类别词泄漏、集合不足、安全回退和决定篡改；
4. 输入账本覆盖精确 7,006 个双候选任务；
5. scene/task/geometry/候选/视角连接错误数为 0；
6. AB/BA 不是完全反序的数量为 0；
7. mutation、GT、AP 标志全部为假；
8. 不修改任何旧文件或正式输出。

任一门槛失败即停止，不运行 Qwen，也不根据失败样本临时修改门槛。

### 8.1 正式运行前烟雾模板修正登记

正式 7,006 条运行前的固定 2 条真实烟雾均在一次重写后结构无效，因此完整任务未启动。只读审查确认原提示模板只展示了空的观察数组，没有展示观察元素必须是 `{view_rank, observation}` 对象；同时没有展示候选集合不足时两个候选状态必须同步切换的完整合法对象。模型分别产生了字符串观察数组，以及顶层集合不足标志与候选 `unknown` 状态不一致。

这是正式运行前的确定性模板表达缺陷，不是证据语义、决定门或阈值问题。允许进行且只允许进行下列一次模板修正：

- 明确观察数组元素禁止为字符串，必须为精确双字段对象；
- 给出 `direct_support`、`direct_strong_counterevidence`、`unknown` 的单候选合法结构示例；
- 给出完整的 `candidate_set_insufficient_or_unclear` 双候选合法示例；
- 提示合同版本从 v1 升为 v2；
- 新建 v2 输入账本目录，不覆盖 v1 输入账本或失败烟雾；
- 四态定义、两视角要求、AB/BA、视角交集、类别词禁令、决定规则和一次重写预算全部不变。

修正后必须重新生成并独立审计输入账本，再执行新的固定小烟雾。若仍不能形成严格合法输出，停止正式运行，不继续修改解析器、状态语义或门槛。

## 9. 正式 Qwen 账本的后续门槛

若用户随后明确授权正式图形处理器推理，固定输出必须记录：

- 严格有效率与一次重写率；
- 四态按候选角色、顺序、场景、类别和五折的分布；
- 原类别双强反证数；
- 替代类双支持数；
- 支持/反证视角交集通过数；
- 候选集合不足率；
- 安全回退率；
- v2.1 换类数与相对旧 DM-SMS-1 的子集关系；
- 类别和场景集中度；
- 全部冻结字段修改数。

只有完整原始输出独立审计错误数为 0，且四态没有再次结构性退化为恒定输出，才允许另建 train100 AP 诊断预注册。不得自动运行 AP，不得重新开放 validation60，不得读取 val312。
