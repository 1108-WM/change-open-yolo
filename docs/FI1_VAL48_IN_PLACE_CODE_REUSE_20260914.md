<!-- FI1_VAL48_HOST_VERIFIED_LATEST -->
**最新更正：2026-09-14 20:36 北京时间已在宿主机确认任务持续运行，之前“已中断”的判断撤回。** 原总控 PID 2922649 存活，已从第1批前端进入 `geometry`；第1批A/B/C/D前端全部完成，累计32/192场景—组（各8），完整批次仍0/6，全48预测和新AP未完成。沙箱内 `ps` 看不到宿主进程，不能据此断言中断；必须结合宿主进程、执行JSON和日志更新。**不要重启当前任务，不需要更换执行身份。**

项目已有本地 **tmux 3.2a**：`bash tools/local_tmux.sh -V`；系统PATH中的 `tmux` 不存在不等于未安装。当前任务在后台运行；tmux日志监视窗口不接管实验进程。新对话先读本更正、最新交接，再读 `full_batches/execution.json` 及其当前阶段日志。此前 `handoff/runtime_correction_20260914.json` 的 `interrupted_incomplete` 是错误诊断，仅保留作审计历史，以本次宿主核验回执为准。

宿主进一步核验：总控 SID=PID=2922649，无控制终端，父进程为 `systemd --user`；当前已独立于原终端后台运行。已创建 tmux **只读监视**会话 `fi1_val48_monitor`，在仓库执行 `bash tools/local_tmux.sh attach -t fi1_val48_monitor` 可查看日志，按 Ctrl+b 再按 d 脱离。该会话不托管／重启实验；当前跟踪总控及第1批geometry日志，后续阶段请按execution.json查对应日志。tmux不能防止程序异常、OOM或重启。
<!-- FI1_VAL48_HOST_VERIFIED_LATEST_END -->

<!-- FI1_VAL48_HANDOFF_LATEST_20260914 -->
**最新交接（2026-09-14 20:06:36 北京时间）：val48第1/6批前端正在运行，日志持续更新；已核对前端24/192场景—组，A/B/C/D=8/8/4/4，完整批次0/6。** 四组固定首场景完整D-v3＋自身YOLO/Alpha＋冻结R3b已完成，66项测试及887组投影核对通过；H/N各48预测完成。**全48预测及本轮新AP尚未完成。** 训练200、原26维R3b、600预算、无Qwen不变；扩量缺941扫描未实施。R0—R4历史对照及55→312分析已经完成，下面旧“R3/R4待实施”等只表示当时状态。

新对话先读[最新交接与续跑说明](FI1_VAL48_LATEST_HANDOFF_20260914.md)，再读实时`F/full_batches/execution.json`与阶段日志（`F=.local/fi1_val48_frontend_full_pipeline_20260914/`）。**不要重复启动，不要修改运行中绑定的代码／协议。** 前端已完成数是本次快照，`export_status.json`可能尚未刷新。18:54给出的16—20小时是旧时点估计，只指生成与评分；新AP合同及评测接口仍待审计。阅读优先级：本区→最新交接→运行回执→下方历史记录。
<!-- FI1_VAL48_HANDOFF_LATEST_END -->

**后续更新：四组首场景完整后端已打通，48场景分批运行已启动；下文为上一阶段历史记录。** 见[后端接入与运行记录](FI1_VAL48_BACKEND_INTEGRATION_20260914.md)。

# val48复用20场景代码：原地修改与实跑记录

**已直接修改原前端／评测文件，没有另建第二套A—D算法实现。48场景输入准备完成；H/N预测完成；固定首场景A/C真实GPU生成及四阶段几何流程完成。完整48的A—D尚未生成，新D-v3衔接和新AP尚未完成。**

## 复用与修改

- `tools/run_fi1_frontend_comparison.py`保留原SAM、GD框提示、投影、RLE、逐帧缓存以及原四阶段函数。新增`--selection-root`读取冻结48清单，校验其输入和输出摘要，不重新选场景；取消写死的20场景计数。
- 同一入口新增`--scenes`与`--arms`，可按固定子集先运行A/C，再复用公共帧缓存继续B/D。观测始终保留原完整manifest的帧编号，不能因子集运行重排编号。分阶段执行只写局部进度；192个场景—组组合全部通过才写全量完成标志。
- `tools/evaluate_fi1_frontend_comparison.py`保留旧公共30帧评测命令；新增`full-controls`，调用既有`prepare_fi1_r3b_val312.score_scene`、冻结R3b模型及独立上下文／排序核验函数。未重新实现评分头，也未把旧简化评测冒充完整评测。
- N控制从同一实际原生点集和自身证据出发，只保留native类别别名，移除非原生别名及D-v3细化计划，并在N池重新计算同类上下文。不是从H的最终top600简单删除轨迹。H保留完整候选，逐成员分数及入选列表与原312结果精确一致。
- 旧20场景产物及契约未覆盖。原修改前源码和测试在`in_place_reuse_before/`保存，原前端源码摘要与旧20场景contract完全一致。当前源码变化后旧冻结任务不会被静默重新签名；本轮使用单独输出目录。

## 实际验证结果

相关单元／集成测试22项通过，包括原非连续帧跟踪、空观测、缓存损坏拒绝、场景清单篡改拒绝、子集执行顺序、局部完成标志和N来源别名清理。

48场景原始RGB／深度／位姿及点云准备全部完成。A/C各1440采样帧，B/D各16462采样帧；全manifest合计148个无效位姿采样帧，仍按旧协议保留预算、跳过推理、不替换场景或帧。48扫描与原清单逐项一致。

H/N共96个场景—组预测完成，H全部48扫描精确复现既有R3b分数和排序；新GT/AP未读取或计算。N精确几何／类别去重后预算前成员为596—600条，不人为补足600。

固定首场景`scene0153_00`的真实GPU结果：

| 组 | 帧数 | 保留二维观测 | 最终合并轨迹 |
|---|---:|---:|---:|
| A 自动SAM | 30 | 285 | 40 |
| C GD-SAM | 30 | 207 | 23 |

两组帧输出及四阶段几何输出哈希核验通过。数量只证明执行和接口，不表示AP收益。`export_status.json`记录2／192场景—组完成，未产生`observations_complete.json`全量标志。GPU单场景命令已正常退出，没有宣称后台全量正在运行。

H/N首次接入在N几何子集仍传入H完整证据字典时触发严格覆盖检查。已在原文件修正为同几何键的证据子集，然后重跑全部48成功；失败源码、局部产物和日志保留在`full_controls_failed_evidence_subset/`，不能作为有效完成结果。

## 完整后端仍需适配的接口

20场景前端的`frame_index`是完整显式manifest中的索引；旧D-v3 C阶段`_union_observation_arrays`直接用该索引访问`WORLD_2_CAM`按frequency=10生成的深度／位姿列表。两者不是同一域。例如首场景manifest索引1对应原始帧5，旧world索引1对应原始帧10。此处还没有实际接入或计算新D-v3，不能据此宣称旧实验结果错误。

后续需在原实现的输入适配处按真实原始帧ID绑定，或显式注入与manifest完全一致的world路径列表，并通过深度／位姿和点投影检查。不能只把新轨迹目录传给旧D-v3脚本。

此外，新轨迹的GVC、native关系、pair_union、unique_geometry账本须由既有入口重算，不能继承历史候选的质量／父关系／类别。新增实际点集需要自己的YOLO/Alpha证据；类别无关评分及去重合同也仍待执行前冻结。上述后端工作继续复用已有D-v3和R3b模块，而非复制一套模型或算法。

## 运行记录与续接

本轮输出目录：`.local/fi1_val48_frontend_full_pipeline_20260914/`。

实际已成功执行：

```bash
.venvs/yoloe_gpu/bin/python tools/run_fi1_frontend_comparison.py prepare --selection-root .local/fi1_val48_frontend_full_pipeline_20260914 --dataset-root /home/jia/Wm/Dataset/scannet200 --output .local/fi1_val48_frontend_full_pipeline_20260914/frontend
.venvs/yoloe_gpu/bin/python tools/evaluate_fi1_frontend_comparison.py full-controls --output .local/fi1_val48_frontend_full_pipeline_20260914/full_controls
.venvs/yoloe_gpu/bin/python tools/run_fi1_frontend_comparison.py export --output .local/fi1_val48_frontend_full_pipeline_20260914/frontend --scenes scene0153_00 --arms A C
```

前端首场景实跑见`frontend_smoke_AC.log`和`frontend_smoke_audit.json`；H/N见`full_controls/summary.json`；输入准备见`frontend/preflight.json`；测试见`reuse_tests.log`。下一步先解决完整后端衔接和单场景闭环，再按已固定的48与A—D批量推进。训练200、冻结R3b、无Qwen的实验设计不变。
