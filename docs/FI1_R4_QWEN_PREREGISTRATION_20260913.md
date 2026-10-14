# FI1 R4 Qwen 独立对照预注册（2026-09-13）

R3b 分层分析和代码审计完成后，R4 使用冻结的候选几何、点集、类别候选与基础预算，仅对 R3b/R2 产生的候选类别进行独立 Qwen2.5-VL 对照。Qwen 不参与训练，不修改几何、点集或分数；只有 current-first 与 proposed-first 两次都选择 proposed 时才改类别，其余情况保留 current。

评测仍使用 ScanNet200 实例分割 AP、AP50、AP25，开发验证为 55 扫描；5 扫描诊断集仅作重叠场景诊断，不能作为独立泛化结论。模型、提示词、AB/BA 顺序、失败回退、像素预算和本地模型 revision 固定，运行前不得依据验证 AP 调整。

代码审计：`run_z6e_qwen25vl_selective_review.py` 与 `run_z6f_qwen25vl_symmetric_review.py` 已通过 Python 编译检查；脚本拒绝覆盖已有输出，并记录原始输出、解析状态、模型 revision 与输入摘要。
