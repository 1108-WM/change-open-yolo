from pathlib import Path
import shutil,json,hashlib
root=Path('/home/jia/Wm/wm_open-yolo/OpenYOLO3D');arch=root/'docs/state_archive/20261014_query_transfer_eval'
files=['新开对话阅读内容.md','CURRENT_EXPERIMENT_STATUS.md','资料/当前基线修改方向.md','资料/结果.md','docs/CODEX_HANDOFF.md','docs/FI1_QUERY_SAME_PARENT_AB_NEXT_STEP_20261014.md','research/geometry/README.md']
assert not arch.exists()
for f in files:
 p=arch/'before'/f;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/f,p)
report='research/geometry/experiments/fi1_query_transfer_eval_20261014_v1/report.md'
audit='research/geometry/experiments/fi1_query_same_parent_pilot_audit_20261014_v1/report.md'
def update(f,fn):
 p=root/f;p.write_text(fn(p.read_text()))
update('CURRENT_EXPERIMENT_STATUS.md',lambda s:s.replace('\n## ', '\n## 2026-10-14 最新完成：现有A/B固定跨场景评估，暂停当前修复器扩训\n\n[报告]('+report+')／[判定](research/geometry/experiments/fi1_query_transfer_eval_20261014_v1/decision.json)／[核验](research/geometry/experiments/fi1_query_transfer_eval_20261014_v1/verification.json)。按用户最新授权，只在原selection16评估现有epoch64 A/B，原16训练父归一化与.8门固定，比较原NT、A、B、B_zero、原三次shuffle及已有区域平均.5。16物理场景与pilot训练隔离，128原父全部保留，原完整NT395目标作池基线；这是开发评估，未用最终holdout，不是全项目盲测。\n\n97稳定父平均IoU增量A−.016189/B−.014050；41好父损失>.02为10/9，最坏.457803/.438995。A仅1场景新增1个NT≥.02目标，B与B_zero均0，所有网络输出无固定档位覆盖新增。B减轻部分错误但自身仍不安全，不进入“选更简单A”或“继续query”分支。高置信误补/误删、机会余量及输入分布记录在报告；暂停当前配方扩训，不改阈值、不加网络或训练轮数。\n\n2项测试、896点集/9480全池记录独立复算、768模型概率回放和2743805原点映射通过。模型/归一化/旧封存/SL保留，没有新训练、校准、AP或生产替换。文档前后快照见[本次交接](docs/state_archive/20261014_query_transfer_eval/README.md)。\n\n以下为此前完成时点记录；原训练内失败不追改。\n\n## ',1))
update('资料/结果.md',lambda s:s.replace('2026-10-14最新完成同一父目标query小样本A/B，未支持扩大训练；','2026-10-14最新完成现有A/B在原selection16的固定评估，跨场景伤害明显，暂停当前修复器扩训；小样本及独立审计保留。').replace('\n## 2026-10-14 同一父', '\n## 2026-10-14 现有A/B固定跨场景评估：训练拟合未转为安全修复\n\n[报告](../'+report+')／[汇总](../research/geometry/experiments/fi1_query_transfer_eval_20261014_v1/evaluation/summary.json)／[核验](../research/geometry/experiments/fi1_query_transfer_eval_20261014_v1/verification.json)。selection16与pilot训练物理隔离，原128父全部保留，epoch64/归一化/.8锁定，不训练。97稳定父原平均IoU.669029，A/B/B_zero为.652840/.654979/.652242。A/B好父损失>.02为10/41和9/41，最坏.457803/.438995；误补邻物4491/3518，误删目标752/802。完整NT395目标，A仅1目标/1场景增量≥.02，B与B_zero无新增，固定覆盖档位均未跨越。\n\nB比去query略好，仍不安全且无额外池价值；按最新用户授权只评估后暂停当前修复器扩训，不选A部署。区域平均.5为3池目标/3场景，但未知编辑占76.90%，不能作为安全替换。2测试、896点集/9480池记录复算、768模型概率回放通过。原模型/归一化/SL不变，无最终holdout/新训练/AP。\n\n独立[训练饱和审计](../'+audit+')发现原“B平均领先.01”门超出训练剩余空间，因此原失败不足以否定方向；本次跨场景评估使用独立合同，未改旧失败记录。\n\n## 2026-10-14 同一父',1))
update('资料/当前基线修改方向.md',lambda s:s.replace('\n## 当前判断：query拼接', '\n## 当前判断：固定跨场景输出伤害明显，暂停当前修复器扩训\n\n[固定跨场景报告](../'+report+')已完成，不再训练。selection16与pilot训练隔离，97稳定父A/B平均IoU下降.016189/.014050，好父损失>.02为10/41、9/41，最坏约.46/.44；A仅1个池增量目标，B及去query均0。B减轻部分损害却无安全池增量，A自身也不安全，当前不选择A部署或扩大B。\n\n高置信跨场景错误已定位：B误补邻物3518点的目标概率均值.957，其中3097点query<.5；误删802目标中655点query≥.5。固定域存在11032补点/2745删除机会，head裁剪均0；机会不足、通道漏接或简单裁剪错误无法解释当前失败。输入联合分布、训练覆盖、身份/未知泛化、损失与动作置信度仍须区分原因，尚未确认主因。保留所有失败，若继续需另立机制任务；不立即加新网络/树/轮数或阈值门。\n\n[独立审计](../'+audit+')已指出pilot训练A接近域内上限，“B均值领先.01”门无法满足；原失败只对应合同，不能否定方向。本次实际未训练场景伤害提供新的泛化证据，不追改旧合同/模型/判定。\n\n## 前序训练内判断：query拼接',1))
update('新开对话阅读内容.md',lambda s:s.replace('- **最新[同父query小样本A/B]', '- **最新[现有A/B固定跨场景评估]('+report+')已完成**：只评selection16，模型/原归一化/.8不变，128原父保留。A/B稳定父平均IoU下降.016189/.014050，好父损失>.02为10/41、9/41；A仅1池目标，B及去query无新增。暂停当前修复器扩训，不选A部署、不扩B；不自动换树/网络/轮数。完整NT、旧失败及独立审计保留。\n- 前序[同父query小样本A/B]',1).replace('先查冗余/融合/损失与动作机制，不扩大同一训练。','其训练饱和门限问题见独立审计；后续已完成固定跨场景评估，见上。',1).replace('1. 先读最新[小样本报告]', '1. 先读最新[固定跨场景报告]('+report+')及[泛化失败诊断](research/geometry/experiments/fi1_query_transfer_eval_20261014_v1/failure_diagnosis/summary.json)，再按需读前序[小样本报告]',1).replace('成立后才内部场景隔离评估。','固定场景隔离评估现已按最新用户授权完成并失败。',1))
update('docs/CODEX_HANDOFF.md',lambda s:s.replace('\n详细状态在', '\n最新用户授权的[现有A/B固定跨场景评估](../'+report+')已完成，只评原selection16、不训练。跨场景好父伤害明显，B无额外NT池价值，A自身也不安全；暂停当前修复器扩训，不直接选A或继续B。原16训练父/64轮失败记录及[饱和审计](../'+audit+')保留，不能以训练门限否定全部query方向。无需恢复训练或扩大评估；新研究需定位报告中的高置信错点/分布/损失机制。\n\n详细状态在',1))
update('docs/FI1_QUERY_SAME_PARENT_AB_NEXT_STEP_20261014.md',lambda s:s.replace('\n本文件修订', '\n后续最新授权已执行[只评估现有模型的selection16对照](../'+report+')，没有扩大训练或改旧合同。A/B均跨场景伤害明显，暂停当前配方；独立审计指出原pilot饱和训练门无法满足，原失败记录不重判。\n\n本文件修订',1))
update('research/geometry/README.md',lambda s:s.replace('\n研究入口按', '\n最新[固定跨场景评估](experiments/fi1_query_transfer_eval_20261014_v1/report.md)只评原selection16、128原父，现有epoch64模型/归一化/.8锁定；A/B均伤害好父、B无池增量，暂停当前修复器扩训。2测试、896点集/9480池记录及768模型回放通过。原pilot/审计/旧诊断封存不变，无新训练、最终holdout或SL/AP。\n\n研究入口按',1).replace('| `code/tools/run_fi1_query_same_parent_pilot.py`','| `code/tools/run_fi1_query_transfer_eval.py`、`verify_fi1_query_transfer_eval.py`、`analyze_fi1_query_transfer_eval.py` | 只评selection16、全池/好父尾部/干预及失败诊断；宿主入口run_fi1_query_transfer_eval_host.sh |\n| `code/tools/run_fi1_query_same_parent_pilot.py`',1))
manifest={}
for stage in ('before','after'):
 manifest[stage]={}
 for f in files:
  p=arch/stage/f
  if stage=='after':p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(root/f,p)
  manifest[stage][f]=hashlib.sha256(p.read_bytes()).hexdigest()
(arch/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
(arch/'README.md').write_text('# 现有A/B固定跨场景评估交接\n\n2026-10-14。[报告](../../../'+report+')。只评selection16，未训练；A/B均跨场景伤害明显，暂停当前修复器扩训。before/after保存7份当前文档，旧pilot、独立审计与诊断的封存不改；hash见manifest.json。\n')
print('synced',len(files),'documents')
