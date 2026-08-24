# FI1-D-v3 × 旧版 DM-SMS-1 val312 代码包验收登记

日期：2026-08-24

分支：`dm-sms1-fi1-d-v3-val312-package-20260824`

实现前父提交：`11a83d4`（已包含冻结 FI1-D-v3 部署代码与经审计的旧版 DM-SMS-1 运行代码）

## 1. 验收结论

代码包已经形成完整的远程分阶段执行链：FI1-D-v3 冻结结果只读预检、联合唯一几何适配、Alpha-CLIP/SAM 证据、旧版双顺序 DM-SMS-1 仲裁、完整安全决定、冻结预测缓存、显式授权的唯一一次开放词汇 AP，以及 AP 独立审计。

本地验收未访问 val312 数据，未读取真实标注，未运行 GPU、Alpha-CLIP、SAM、Qwen 或 AP，也未修改和重跑 FI1-D-v3。既有 FI1-Legacy、FI1-D-v3、DM-SMS-1、v2/v2.1/v2.2 代码和产物均未覆盖。

## 2. 冻结方法确认

- 几何、掩码、候选集合与排序分数唯一来自已经审计通过的 FI1-D-v3 完整挑战计划。
- 旧版 DM-SMS-1 决定规则未改变：替代类别正序/反序双支持，且替代类别两种顺序均无强反证，才换类。
- 没有接入 v2、v2.1、v2.2，也没有增加原类别双强反证门。
- 正式控制和挑战仅允许类别编号不同；掩码、分数、候选数和顺序逐条相同。
- AP 入口需要双重显式授权，并在启动时创建不可回退的调用标记；失败或中断也禁止静默重跑。

## 3. 本地验证

通过：

```text
Python 语法编译：通过
git diff --check：通过
联合与相关回归测试：45 passed
```

测试覆盖联合适配、refined union 追加、重复/损坏结构拒绝、预测缓存逐条一致、控制/挑战仅类别不同、AP 无授权拒绝、唯一调用标记、完成后禁止二次运行、两份官方 CSV 独立重算等合同。

## 4. 新增文件 SHA-256

```text
2015518c481a2ca59754b3cd2adbd536a753b06468978c11009f896bb5608b9a  docs/DM_SMS1_FI1_D_V3_VAL312_JOINT_PREREGISTRATION_20260824.md
73f0c96823e9091a732ca6cd89b6e12fd75c8065eae90df3b618e504a2f08597  docs/DM_SMS1_FI1_D_V3_VAL312_PATHS.example.json
792527795d024f75a1fb23c949afe42e9f064544bf0d7b58dc1303ee25d4ec1b  docs/DM_SMS1_FI1_D_V3_VAL312_REMOTE_RUNBOOK_20260824.md
0ddbb407c0547a313060e094cdf66658cd1fecf68a7bad68ac05dd4bc0c32370  tests/test_dm_sms1_fi1_d_v3_joint_package.py
0b9d86c6bbb48a3db0ad5bd29868945a6adf89c61be6cf8fd80f8a9cd74ee95e  tools/preflight_dm_sms1_fi1_d_v3_val312.py
9ab22cf0297dcfaf5227e560815714bbd0eea81d8392e9b8e304a8b86980ed84  tools/build_dm_sms1_fi1_d_v3_unique_geometry_ledger.py
a6b5c5b80b49604e96af42c292ceeb88ac0e4dca83c0028a281b05284dd46ae9  tools/audit_dm_sms1_fi1_d_v3_unique_geometry_ledger.py
8ab77f3f59f0d24d7f7e1449c834a02fcd889db186bc8c74840f81acc146eda3  tools/build_dm_sms1_fi1_d_v3_prediction_cache.py
5431e98ebb22f1751994ec057667a6a3c3e7423cd9cf11d483204ec6c501b84e  tools/audit_dm_sms1_fi1_d_v3_prediction_cache.py
7e8eefd7e543c35ef8e58e327271a4d2b250bf0eb90320369e7160d621267382  tools/evaluate_dm_sms1_fi1_d_v3_open_vocab_ap_gt.py
4ef06daa4834d5f0701d907143510ad34ab75b0277969983b123d6403396167c  tools/audit_dm_sms1_fi1_d_v3_open_vocab_ap.py
332569aadf06801100d9b653923d46c659a3b65aa52d7af06946201d0ec837c3  tools/run_dm_sms1_fi1_d_v3_val312_pipeline.py
```

本登记文件自身不纳入上述哈希集合，避免自引用。

## 5. 上传边界

允许上传的是本独立分支提交，不应从原始脏工作区临时挑选文件。上传后远程服务器先填写路径配置并执行测试和 `preflight`；未经用户在远程再次明确授权，不执行正式 AP 阶段。
