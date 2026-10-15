#!/usr/bin/env python3
"""Posthoc reporting only: no new query, block, frame, pointset or selector."""
from pathlib import Path
from collections import Counter
import sys
import time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_fi1_local_block_attribution_selection as run
from analyze_fi1_local_block_attribution_selection import block_summary
D=run.diag;OUT=run.OUT


def main():
    c=D.read(OUT/'contract.json');s=D.read(OUT/'evaluation_summary.json');a=D.read(OUT/'analysis/summary.json')
    v=D.read(OUT/'verification.json');assert v['complete']
    rows=[];blocks=[]
    for scene in c['scenes']:
        rows+=D.read(OUT/'evaluation'/scene/'unions.json');blocks+=D.read(OUT/'evaluation'/scene/'blocks.json')
    ev={(r['scene'],r['block_id']):r for r in D.read(OUT/'analysis/valuable_block_evidence.json')}
    pure=[r for r in blocks if r.get('valuable_pure_contributor')]
    rest=[r for r in blocks if r['block_GT']['ownership']=='target_only' and not r.get('valuable_pure_contributor')]
    def extended(rs):
        result=block_summary(rs)
        overlaps=[r['anchor_supporter_hard_intersection']/r['anchor_supporter_hard_union'] for r in rs if r['anchor_supporter_hard_union']]
        result['anchor_hard_iou_quantiles']=np.quantile(overlaps,[0,.25,.5,.75,1]).tolist() if overlaps else []
        return result
    point_by_scope={}
    for scope in ('stable','ambiguous','out_of_scope','unannotated'):
        point_by_scope[scope]={}
        br=[r for r in blocks if r['parent_scope']==scope]
        for mode,key in [('W','block_GT'),('C','clipped_GT'),('discarded','dropped_GT')]:
            point_by_scope[scope][mode]=dict(blocks=len(br),points=sum(r[key]['points'] for r in br),
                annotated=sum(r[key]['annotated'] for r in br),annotation_unknown=sum(r[key]['unknown'] for r in br),
                target=sum(r[key]['target'] for r in br) if scope=='stable' else None,
                other=sum(r[key]['other'] for r in br) if scope=='stable' else None)
    cohort={}
    for group in c['cohorts']:
        cohort[group]={}
        for mode in ('W','C'):
            rr=[r for r in rows if r['cohort']==group and r['mode']==mode]
            cohort[group][mode]=dict(parents=len(rr),stable=sum(r['stable'] for r in rr),
                same_parent_gain02_parents=sum(r['same_parent_gain02'] for r in rr),
                safe_gain02_parents=sum(r['safe_gain02'] for r in rr),
                good_parents=sum(r.get('good_parent',False) for r in rr),
                good_copy_loss_gt02=sum(r.get('good_loss',0)>.02+1e-12 for r in rr))
    extra=dict(posthoc_reporting_only=True,new_constructions=False,
        point_counts_by_parent_scope=point_by_scope,cohort=cohort,
        valuable_pure_contributors=extended(pure),other_pure_target_blocks=extended(rest),
        valuable_pure_with_same_signature_neighbors=sum(ev[r['scene'],r['block_id']]['same_signature_neighbors']>0 for r in pure),
        metric_notes={
            'evaluation_summary.unions.add_target_precision':'Legacy helper denominator includes all 207 parents; numerator only stable targets. Do NOT present this as stable-target precision. Use analysis.groups.stable.point_selection instead.',
            'analysis.point_selection':'Point target/other precision fields aggregate ONLY stable parents. Empty aggregates for unresolved parents are not zero errors. All parent scopes annotation counts are provided here and all per-block census rows remain available.',
            'unknown_identity_vs_annotation':'For ambiguous/out-of-scope/unannotated parents, target and other are null, annotated and annotation_unknown remain measured. Known points there are not presumed wrong or correct.',
            'oracle':'All oracle constructions retain original parent; stable-target domain bounds only, not deployed/GT-free outputs or proof of all query combinations.',
            'collisions':'Predeclared coarse signature collisions and marginal continuous ranges do not prove raw evidence nonseparability. No thresholds/rules are fitted after labels.'})
    d=OUT/'posthoc_reporting';d.mkdir()
    D.seal(d/'summary.json',extra)
    decision=dict(status='stop_current_fixed_constructions_before_AP',complete=True,
        reason='Unattributed W/C have pool gains but substantial negative/good-copy pollution. No validated GT-free ownership selector was established. Do not promote GT-pure oracle, signature cells, overall AUC, or no-actions into method success.',
        attribution_result='Valuable pure blocks enriched for low parent coverage/low anchor overlap; categorical evidence often collides with neighbors; raw evidence insufficiency remains unproven.',
        point_result='Clipping helps but leaves substantial neighbor/unknown additions; pure-block oracle gains mostly retained.',
        next_boundary='No more fitting thresholds in this round. New evidence/region representation or any newly proposed selection belongs to a separately frozen development round; future acceptance needs physically isolated scenes.',
        qualified_end_to_end_output=False,AP_run=False,training=False,production_modified=False,
        reading_entry_modified=False,github_push=False,scope=c['scope'])
    D.seal(OUT/'decision.json',decision)
    opportunity=D.read(OUT/'evaluation/opportunities.json')
    case_lines=[]
    for r in opportunity:
        def val(m):return f"{r['modes'][m]['gain_over_NT']:+.6f}"
        case_lines.append(f"| {r['scene']}/{r['candidate_id']} | {val('W')} | {val('C')} | {val('oracle_pure_W')} | {val('oracle_target_points')} |")
    mode_lines=[]
    for mode,title in [('W','全部块整块合并 W'),('C','每块最高均值query≥0.5后合并 C'),('oracle_pure_W','GT纯目标块＋W（oracle）'),('oracle_pure_C','GT纯目标块＋C（oracle）'),('oracle_target_points','GT目标点（oracle）')]:
        r=s['modes'][mode];u=r['unions'];pool=r['same_parent_pool']
        mode_lines.append(f"| {title} | {r['same_parent_gain02_parents']} | {pool['pool_gain02']} | {r['same_parent_gain02_scenes']} | {r['safe_gain02_scenes']} | {u['good_loss_gt_02']}/12 | {pool['pool_delta_sum']:.6f} |")
    scopes_lines=[]
    for scope,x in point_by_scope.items():
        scopes_lines.append(f"| {scope} | {s['scopes'][scope]} | {x['W']['blocks']} | {x['W']['annotated']} / {x['W']['annotation_unknown']} | {x['C']['annotated']} / {x['C']['annotation_unknown']} |")
    report=f'''# 局部块归属与块内选择：exposed52固定诊断

日期：2026-10-15。完成；停止本轮固定构造，未进入AP。适用范围仅为52个已暴露物理场景、207个采样T父及原可编辑域；不是全Geometry结论，也不是独立验收。

**关键判断：块内query裁剪有过滤作用，但它不能代替块归属。真正超过完整NT的纯块机会，比普通正确块明显更偏向低父覆盖、低锚重叠；不能用统一的高覆盖/高重叠/共同帧门槛清除它们。当前固定比较没有形成同时保住跨场景收益、通过全部负例风险检查的无GT归属与选择方法。**

## 1. 先封存、后诊断；两个问题分开

合同与代码快照：[contract.json](contract.json)。无GT证据/点集全部完成封存后，才启动本轮GT读取；生成进程使用禁止GT及旧evaluation访问的audit hook。完整证据为1,308块、7,730帧记录；父/halo/区域/query集合/原≤6帧自动mask和可靠性全部复用。

本轮固定的GT-free比较：每个块在既有query集合中选块均值最高者、query ID升序破同分；W补整块，C只保留该query≥0.5且属于该块的新增点。每父分别联合全部W/C，**不看GT回选块、不删原父、不覆盖原N/T**。它们是未确认归属的几何比较点集，并非已验收候选，不直接接AP。

归属证据单独保留概率、父覆盖、锚重叠、几何联系及逐帧共同/冲突/缺失/并列/无支持/弱支持。参考主体为距该块最近的原父成员区域（点间最短距离、区域ID破同分），不按锚query裁剪参考。此参考构造在本轮GT前固定；与前序诊断的“最近质心区域∩锚mask”不同，帧统计不可混为同一指标。没有用上述信号做通用必需门槛。

前置的“至少两共同帧且无冲突”原型已在用户补充约束后、GT前退役，封存保留于[退役说明](../fi1_rejectable_local_block_exposed52_20261015_v1/superseded_before_GT.json)；对应下游仅prepare，未语义推理/评分/GT/AP。本轮是新目录，未覆盖旧seal。

## 2. 所有父与负例均保留

稳定父143个、非稳定父64个；稳定不等于纯父：其中37个含多个已知GT实例。混杂父、身份不明父仍保存每块标注分布、点选择结果和联合输出；不把其已知点擅自归为目标或邻物。

| 父身份范围 | 父数 | 块数 | W已标注/未知点 | C已标注/未知点 |
|---|---:|---:|---:|---:|
{chr(10).join(scopes_lines)}

以上点数按“父—提案”计，同物理点可能参与不同父的提案，不是全场景唯一点数。稳定父844块中：纯目标341、纯邻物391、目标/邻物混杂23、目标+未知15、邻物+未知29、全未知45。其余464块属于非稳定父，目标/邻物计数为null，绝不当成零错误。

## 3. 块内点选择本身能做什么

在稳定父全部844块上，C相对W：目标点38,109→36,567，保留95.95%；已知邻物53,326→31,962，去除40.06%；未知2,845→2,242，仍保留78.80%。精确率（未知仍在分母）40.42%→51.67%，不足以充当安全归属。

分开看：341个纯目标块保留98.40%目标点；391个纯邻物块仍留下29,015邻物点（只去掉35.79%）；23个目标/邻物混块去掉70.94%邻物，却也丢掉49.09%本可新增的目标点。所有这些是“新增部分不被选中”，不是删除原父目标点；全量构造实际删除原父点=0。

若**仅作GT归属oracle**，把纯目标块送入W/C，两者都保留5目标/5场景的完整NT增量。这说明已供给的干净整块内部，当前裁剪大体保留机会；绝不意味着完整方法成功。混杂/未知边界仍有额外困难，点级目标oracle为8目标/7场景。

## 4. 实际固定union与完整NT

同父增量=修补副本对原父稳定目标的IoU−完整原NT对该目标的最佳IoU；≥0.02为gain02。原NT始终在池内，因此追加池的几何上限不会因差副本下降。表中“好副本损害”指原父IoU≥0.9时，其修补副本下降>0.02，**不是原T被覆盖，也不是已测得下游损害**。

“安全gain场景”仅为后置诊断：gain02且新增点中目标占比≥0.9（未知不作正确点），不是GT-free接收规则，不能GT选走其他输出。

| 比较 | 同父gain父数 | 去重池gain目标数 | gain场景 | 安全gain场景 | 好副本损害 | 池连续增量总和 |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(mode_lines)}

W/C均有2目标/2场景池收益，不能说毫无增量；但C仍有48个稳定父副本下降≥0.02、9/12好副本下降>0.02，最大好副本损失0.539073。W对应57、10/12、0.630965。C的3个gain父集中于scene0057_00及scene0301_02（后者两父对应同一目标）。internal24的95父仍无gain，且C有5/6好副本损害。收益与风险必须一起保留，不能事后仅挑三父。

任意目标池也另存，不能把漂移修补给其他实例的收益算作“同父修复”。C任意目标池连续增量0.452803，同父0.440888；本表只用同父。所有父/场景连续IoU、误补邻物、未知、漂移及好副本损失详见[evaluation_summary.json](evaluation_summary.json)和逐场景unions.json。

下表覆盖全部点级或前序单query机会父（GT后置诊断，非本轮只看这些父；其余负例均在全量结果）。数值是相对完整NT的连续同父增量：

| 场景/父 | 固定W | 固定C | 纯整块oracle | 目标点oracle |
|---|---:|---:|---:|---:|
{chr(10).join(case_lines)}

scene0002、scene0078、scene0085、scene0369/c006的正确块本可有价值，但实际union混入其他块后变差。这直接区分“块内选择尚可”与“归属未成立”；不是query完全无信息。

## 5. 核心问题：机会是否集中于最难归属处

**有明显富集，但不能说所有机会都只在低重叠或未知处，更不能证明全部原始证据不可分。**

GT纯整块oracle有价值的5场景中，21个块对正确union有正贡献。它们的选中query与锚query硬IoU中位数为0.1978，其他320个纯正确块为1.0000；父覆盖≤一半为9/21（42.9%），其他正确块15/320（4.7%）。这是描述分层，不是新筛选阈值。21块中18块没有共同帧、4块有观测冲突，甚至存在父覆盖和锚重叠均为0的真实碎片。

同时，17/21个有价值纯块、26/32个点级机会贡献块，与跨场景邻物/污染块共享GT前固定的粗证据签名。普通正确块的高覆盖判别性，不能直接转化成有价值碎片的安全归属。签名压缩损失连续信息；同签名或单变量范围相交都**不证明原始信息不可分**，本轮没有通过总体AUC、事后门槛或GT签名单元挑选输出。

具体反例：scene0002/c006/r1纯目标25点，最高均值是非锚q80（均值0.845、父硬覆盖0）；C保留21目标点。最近原父区域的6帧为2并列、4无支持，没有共同帧；同粗签名存在8个邻物/污染负例。它有清楚的点读出，却缺乏已验证的归属选择。scene0085/c007的r4、r11、r16同时有共同与冲突证据，均是真缺块，不能遇冲突一律拒绝。

scene0688/c006/r1则暴露第二种问题：148点含100目标、1邻物、47未知；C去掉邻物却保留全部47未知。只处理这块且保留原父时，相对NT的增量约0.0090，仍不足0.02，目标点oracle约0.0678。未知不能当作成功点或邻物；强身份支持不自动解决输出范围。

完整机会证据、相反标签例子、连续值范围：[valuable_block_evidence.json](analysis/valuable_block_evidence.json)。分层比较、所有身份范围点数及指标分母说明：[posthoc_reporting/summary.json](posthoc_reporting/summary.json)。

## 6. 停止与AP边界

本轮完成了分开诊断及固定union对照，**尚未建立可复现、经所有负例检查而仍保住多物理场景收益的无GT归属选择器**。W/C是归属未定的比较；其池收益不能抵消大量污染和好副本损失，也不能将GT纯块oracle当作已解决身份。依用户条件停止本轮固定证据/构造下的筛选微调，不启动训练或下游AP。

本轮不判定“所有query方法不可行”，不判定“AP不可能改善”；存在几何收益并不等于具备安全部署输出，几何无收益也不等于无下游利用价值。任何新归属规则、身份/区域表达或观测研究须另立GT前封存轮次；不能覆写本轮构造。若后续真有通过风险验收的固定输出，才计算其自身语义、接冻结SL/AP；52场景只能算内部开发，仍需未参与规则设计的物理隔离场景验证。

## 7. 执行与核验

无训练、无新增观测、无新阈值扫描、无生产修改、无GPU语义推理/AP、无GitHub提交、无阅读入口修改。三项必要测试通过；全量独立核验通过：{v['checks']['independent_blocks']:,}块、{v['checks']['independent_frame_states']:,}帧状态、{v['checks']['independent_unions']:,}联合构造、{v['checks']['independent_pool_target_modes']:,}池目标/模式；206个有块父的首块参考区域用穷举点间距离另算。所有原父点保留、W/C仅限原可编辑块，GT oracle复现前序纯整块与目标点供给数值。见[verification.json](verification.json)、[RUN.md](RUN.md)、[decision.json](decision.json)。
'''
    # Obtain rather than hand-round this single-block diagnostic.
    case=next(b for b in blocks if (b['scene'],b['candidate_id'],b['region'])==('scene0688_00','c006',1))
    report=report.replace('约0.0090',f"{case['C_single']['pool_delta']:.6f}")
    (OUT/'report.md').write_text(report)
    (OUT/'RUN.md').write_text('''# 复现顺序

使用项目根目录和 `.venvs/yoloe_gpu/bin/python`；CPU，环境变量 `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4`。

```bash
.venvs/yoloe_gpu/bin/python -m pytest -q research/geometry/code/tests/test_fi1_local_block_attribution_selection.py
.venvs/yoloe_gpu/bin/python research/geometry/code/tools/run_fi1_local_block_attribution_selection.py prepare
.venvs/yoloe_gpu/bin/python research/geometry/code/tools/run_fi1_local_block_attribution_selection.py generate
.venvs/yoloe_gpu/bin/python research/geometry/code/tools/analyze_fi1_local_block_attribution_selection.py evaluate
.venvs/yoloe_gpu/bin/python research/geometry/code/tools/analyze_fi1_local_block_attribution_selection.py analyze
.venvs/yoloe_gpu/bin/python research/geometry/code/tools/verify_fi1_local_block_attribution_selection.py
.venvs/yoloe_gpu/bin/python research/geometry/code/tools/report_fi1_local_block_attribution_selection.py
```

上列为实际执行顺序；prepare/generate/evaluate/analyze均不允许覆盖本轮封存，已有结果不要重跑这些写入步骤。只复核可执行verify；从头复现须另设输出目录，保留合同与代码快照。prepare冻结构造代码，generate进程禁止GT/旧evaluation读取，完成predictions_frozen后才运行evaluate。报告脚本只汇总已诊断结果，不增加新选择或点集。

文件：contract/code_snapshot → predictions/{scene}/blocks,frames,unions,data及predictions_frozen → evaluation与evaluation_frozen → analysis与analysis_frozen → verification → posthoc_reporting/report/decision/completion_verified。

GT后所有oracle都仅用于诊断；没有正式可接收输出或下游AP。
''')
    files=[OUT/'report.md',OUT/'RUN.md',OUT/'decision.json',d/'summary.json',OUT/'verification.json',Path(__file__)]
    D.seal(OUT/'completion_verified.json',dict(complete=True,time=time.time(),
        prediction_seal_sha256=D.sha(OUT/'predictions_frozen.json'),
        evaluation_seal_sha256=D.sha(OUT/'evaluation_frozen.json'),
        analysis_seal_sha256=D.sha(OUT/'analysis_frozen.json'),
        outputs={str(p.resolve()):D.sha(p) for p in files},tests='3 passed',
        original_parent_points_deleted=0,AP_run=False,training=False,production_modified=False,reading_entry_modified=False,github_push=False))
    print('completed',OUT,flush=True)


if __name__=='__main__':main()
