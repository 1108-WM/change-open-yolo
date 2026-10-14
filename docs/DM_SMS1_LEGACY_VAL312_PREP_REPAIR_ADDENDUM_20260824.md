# 旧版正收益 DM-SMS-1 val312 准备修复预注册补充条款

状态：`frozen_before_implementation`

日期：2026-08-24

对应主合同：

```text
docs/DM_SMS1_LEGACY_VAL312_PREP_REPAIR_PREREGISTRATION_20260824.md
```

## 1. 补充原因

在主合同冻结后、任何代码修改开始前，对缺失的几何解析辅助模块进行依赖审查，确认它还依赖当前工作分支缺失的：

```text
tools/dm_sms_core.py
```

如果只恢复 `tools/audit_dm_sms1_unique_geometry_ledger.py`，评测与几何审计仍会在导入阶段失败。因此必须把两个通用旧版 DM-SMS-1 模块作为同一部署恢复单元。

## 2. 新增允许文件

主合同第 4 节允许修改范围增加：

```text
tools/dm_sms_core.py
tests/test_dm_sms_core.py
```

其中测试文件当前已经存在，只允许在确有必要时修改；默认直接复用并运行。

## 3. 内容边界

`tools/dm_sms_core.py` 只允许包含旧 DM-SMS-1 已有的无真实标注纯函数：

- 几何点索引哈希；
- 完全相同掩码分组；
- 冻结分数、来源顺序和候选编号的规范成员选择；
- 多视角可见比例特征聚合；
- SMS 统计和固定阈值保留规则。

不得加入 FI1-D-v3 的质量模型、关系重排序、pair-union 细化、边际收益模型或任何新参数。

恢复版本登记：

```text
来源部署补充提交：064ad00204f46b455024fa47fcb0ac6261b765a6
tools/dm_sms_core.py SHA-256：3e3b14dc8dfe723667aab189d877b65bed6b54b20829a6cc47ebf0ccc9fc9a64
tools/audit_dm_sms1_unique_geometry_ledger.py SHA-256：f9c8f360c61517a0907817a30302907a21ccf50ea5d4a2d1a7ba420aa717baa4
```

这里恢复的是通用旧版 DM-SMS-1 部署依赖，不运行、修改或接入 FI1-D-v3。

## 4. 验收补充

除主合同门槛外，还必须满足：

1. `tests/test_dm_sms_core.py` 全部通过；
2. `tools/evaluate_dm_sms1_stage_e_gt.py` 能正常导入；
3. 恢复后的两个文件哈希与登记版本一致，或者任何必要差异都必须在代码验收登记中逐项解释；
4. 不调用这些模块读取 val312 或真实标注。
