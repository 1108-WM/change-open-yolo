#!/usr/bin/env python3
"""Sealed pure-deletion evaluation, complete NT pool and GT-block diagnostic."""
from pathlib import Path
from collections import Counter
import csv,sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_deletion_blocks as run
from tools.fi1_frontend_protocol import sha256_file as sha
THRESHOLDS=(.5,.75,.78,.9,.95,.98)


def iou(p,gt,ids,sizes):
    if len(p)<100:return np.zeros(len(ids))
    values,counts=np.unique(gt[p],return_counts=True);hist=dict(zip(values,counts));hit=np.array([hist.get(g,0) for g in ids])
    return hit/(len(p)+sizes-hit)


def evaluate():
    c=run.read(run.OUT/'contract.json');sys.addaudithook(run.Guard(c,'evaluate').hook)
    run.previous.original.audit.prior.old.verify_files(run.read(run.OUT/'predictions_frozen.json')['outputs'])
    totals={a:Counter() for a in run.ARMS};values={a:set() for a in run.ARMS};scenes={a:set() for a in run.ARMS};breakdown={a:{} for a in run.ARMS};basecoverage=Counter()
    rows=[];targetrows=[];blockrows=[];oracle_totals={source:Counter() for source in ('B','G')};oracle_targets={source:set() for source in ('B','G')};regression={a:Counter() for a in run.ARMS if a.endswith(('summary','relation'))}
    for s in c['scenes']:
        assert sha(c['jobs'][s]['GT'])==c['GT_inputs'][c['jobs'][s]['GT']];gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);ids,sizes=run.previous.original.audit.prior.valid_targets(gt,c['class_ids']);lookup={int(g):k for k,g in enumerate(ids)}
        geometry,nt=run.previous.original.audit.prior.old.load_nt(c['jobs'][s]);baseline=np.zeros(len(ids))
        for p in nt.values():baseline=np.maximum(baseline,iou(p,gt,ids,sizes))
        best={a:baseline.copy() for a in run.ARMS};oraclebest={a:baseline.copy() for a in ('B','G')};originalkeys={p.astype('<i8').tobytes() for p in nt.values()};newkeys={a:set() for a in run.ARMS}
        original_meta={r['candidate_id']:r for r in run.read(run.SOURCE/'labels'/s/'meta.json')};stable_meta={r['candidate_id']:r for r in run.read(run.OUT/'labels'/s/'meta.json')};scope={r['candidate_id']:r['scope_name'] for r in run.read(run.PREVIOUS/'labels'/s/'meta.json')}
        fold=c['folds'][s];proposal=run.OUT/'generators'/f'outer{fold}'/'full'/'proposals'/s/'data.npz'
        with np.load(run.SOURCE/'features'/s/'data.npz') as z,np.load(run.OUT/'predictions'/s/'predictions.npz') as prediction,np.load(proposal) as pr:
            for r in run.read(run.SOURCE/'plan'/s/'candidates.json'):
                cid=r['candidate_id'];p=z[cid+'_parent'];halo=z[cid+'_halo'];gid=original_meta[cid]['target'];stable=stable_meta[cid]['stable'];known=gid is not None;k=lookup[gid] if known else None
                before=iou(p,gt,ids,sizes);old=float(before[k]) if known else None
                for arm in run.ARMS:
                    repaired=prediction[cid+'_'+arm+'_points'];assert np.isin(repaired,p).all();removed=np.setdiff1d(p,repaired,assume_unique=True);after=iou(repaired,gt,ids,sizes);best[arm]=np.maximum(best[arm],after)
                    changed=bool(len(removed));counts=totals[arm];counts.update(parents=1,actions=changed,deleted=len(removed),added=0)
                    if changed:scenes[arm].add(s)
                    key=repaired.astype('<i8').tobytes();counts['duplicate_original']+=changed and key in originalkeys;counts['duplicate_repair']+=changed and key in newkeys[arm];newkeys[arm].add(key)
                    record=dict(scene=s,fold=fold,candidate_id=cid,source=r['source'],arm=arm,target=gid,stable=stable,scope=scope[cid],deleted=len(removed),changed=changed,parent_iou=old)
                    if known:
                        lost=int((gt[removed]==gid).sum());pollution=int(((gt[removed]>=1000)&(gt[removed]!=gid)).sum());unknown=len(removed)-lost-pollution
                        score=float(after[k]);delta=score-old;drift=bool(changed and after.max(initial=0)>0 and int(np.argmax(after))!=k and (gt[repaired]==ids[int(np.argmax(after))]).sum()>=20)
                        valuable=delta>0 and (score-baseline[k]>=.02-1e-12 or any(baseline[k]<=t<score for t in (.75,.9,.95)))
                        if valuable:values[arm].add((s,gid))
                        counts.update(deleted_target=lost,deleted_other=pollution,unknown_edits=unknown,unknown_points=unknown,drift=drift,parent_improved_02=delta>=.02-1e-12,parent_degraded_02=delta<-.02-1e-12,good_parents=old>=.9,good_degraded=old>=.9 and delta<-.02-1e-12,changed_parent_target=int((gt[p]==gid).sum()) if changed else 0)
                        for t in (.1,.25,.5):counts[f'good_loss_gt_{t}']+=old>=.9 and delta<-t-1e-12
                        record.update(repaired_iou=score,parent_delta=delta,original_NT_best=float(baseline[k]),pool_delta=float(score-baseline[k]),deleted_target=lost,deleted_other=pollution,unknown_edits=unknown,drift=drift,joint_value=valuable)
                    else:counts.update(unknown_edits=len(removed),unknown_identity_delete=len(removed),unknown_identity_actions=changed);record.update(unknown_edits=len(removed))
                    if stable:counts.update(stable_deleted_target=record.get('deleted_target',0),stable_deleted_other=record.get('deleted_other',0),stable_unknown=record['unknown_edits'])
                    else:counts.update(unstable_actions=changed,unstable_deleted=len(removed))
                    local=Counter(parents=1,actions=changed,deleted=len(removed),deleted_target=record.get('deleted_target',0),deleted_other=record.get('deleted_other',0),unknown_edits=record['unknown_edits'],parent_improved_02=record.get('parent_delta',0)>=.02-1e-12,parent_degraded_02=record.get('parent_delta',0)<-.02-1e-12,drift=record.get('drift',False))
                    for group in (r['source'],scope[cid],'fold'+str(fold),'fixed10' if s in c['fixed10'] else 'remaining190'):breakdown[arm].setdefault(group,Counter()).update(local)
                    rows.append(record)
                for source in ('B','G'):
                    blocks=run.unpack(pr,cid,'_'+source);metadata=stable_meta[cid];labels=run.core.consequences(gt,p,halo,blocks,metadata)
                    # Same conservative known-other oracle for all historical matched parents;
                    # separate stable-only oracle, since mixed parents must not train a target.
                    pure=[];strictstable=[]
                    for j,b in enumerate(blocks):
                        vals=gt[halo[b]];pure.append(bool(known and (vals>=1000).all() and (vals!=gid).all()));strictstable.append(bool(stable and pure[-1]))
                        detail=labels[j][1];truth=np.asarray(labels[j][0]);row=dict(scene=s,candidate_id=cid,source=source,block=j,points=len(b),stable=stable,known_other_only=pure[-1],**{q:detail[q] for q in ('target_deleted','pollution_deleted','unknown_deleted','delta')})
                        for kind in ('summary','relation'):
                            arm=source+'_'+kind;pred=prediction[cid+'_'+arm+'_block_prediction'][j].astype(np.float32);selected=bool(pred[0]>=.9);row[kind+'_safe_probability']=float(pred[0]);row[kind+'_selected']=selected;row[kind+'_expected_delta']=float(pred[4])
                            reg=regression[arm];reg.update(blocks=1,safe_blocks=int(truth[0]),selected_blocks=selected,selected_safe=int(selected and truth[0]==1),selected_stable=int(selected and stable),selected_unknown_parent=int(selected and not stable),fraction_error=float(np.abs(pred[1:4]-truth[1:4]).sum()),delta_error=abs(float(pred[4]-truth[4]/10)))
                            if selected:reg.update(selected_pollution=detail['pollution_deleted'],selected_target=detail['target_deleted'],selected_unknown=detail['unknown_deleted'])
                        blockrows.append(row)
                    oq=run.core.delete(p,halo,blocks,pure);sq=run.core.delete(p,halo,blocks,strictstable);oraclebest[source]=np.maximum(oraclebest[source],iou(oq,gt,ids,sizes));oracle_totals[source]['blocks']+=len(blocks);oracle_totals[source]['pure_blocks']+=sum(pure)
                    oracle_totals[source]['stable_pure_blocks']+=sum(strictstable)
                    # Keep stable upper bound separate by storing per-parent outcomes.
                    oracle_targets[source].update((s,int(ids[j])) for j in np.flatnonzero(iou(sq,gt,ids,sizes)-baseline>=.02-1e-12))
        for t in THRESHOLDS:basecoverage[str(t)]+=int((baseline>t).sum())
        for arm in run.ARMS:
            totals[arm]['pool_gain02']+=int((best[arm]-baseline>=.02-1e-12).sum());totals[arm]['targets']+=len(ids);totals[arm]['unique_new']+=len(newkeys[arm]-originalkeys)
            for t in THRESHOLDS:totals[arm]['coverage_'+str(t)]+=int((best[arm]>t).sum())
            for j,g in enumerate(ids):targetrows.append(dict(scene=s,fold=fold,arm=arm,gt_id=int(g),original_NT_best=float(baseline[j]),expanded_NT_best=float(best[arm][j]),delta=float(best[arm][j]-baseline[j])))
        for source in ('B','G'):oracle_totals[source]['known_other_oracle_gain02']+=int((oraclebest[source]-baseline>=.02-1e-12).sum())
        run.stage('post_seal_deletion_geometry',scene=s)
    summary=dict(complete=True,scenes=200,parents=1599,baseline_coverage=dict(basecoverage),arms={},breakdown=breakdown,oracle={},block_fit={},SL_modified=False,AP_run=False)
    for source in ('B','G'):summary['oracle'][source]=dict(oracle_totals[source],stable_known_other_oracle_gain02=len(oracle_targets[source]))
    gate=c['gate']
    for arm in run.ARMS:
        m=dict(totals[arm]);known=m.get('deleted_other',0)+m.get('deleted_target',0);m.update(known_delete_precision=m.get('deleted_other',0)/known if known else None,conservative_delete_precision=m.get('deleted_other',0)/m['deleted'] if m['deleted'] else None,
                  target_loss=m.get('deleted_target',0)/m.get('changed_parent_target',1) if m.get('changed_parent_target',0) else None,
                  good_degradation_rate=m.get('good_degraded',0)/m.get('good_parents',1),unknown_action_fraction=m.get('unknown_edits',0)/m['deleted'] if m['deleted'] else None,
                  joint_value_targets=len(values[arm]),joint_value_scenes=len({s for s,g in values[arm]}),action_scenes=len(scenes[arm]))
        g=dict(deletion_first=m['pool_gain02']>=1,value_targets=m['joint_value_targets']>=gate['min_value_targets'],value_scenes=m['joint_value_scenes']>=gate['min_value_scenes'],actions=m['actions']>=gate['min_actions'],action_scenes=m['action_scenes']>=gate['min_action_scenes'],precision=m['conservative_delete_precision'] is not None and m['conservative_delete_precision']>=.9,target_loss=m['target_loss'] is None or m['target_loss']<=.01,good_preservation=m['good_degradation_rate']<=.02,identity=m.get('drift',0)==0,unknown=m['unknown_action_fraction'] is None or m['unknown_action_fraction']<=.1)
        m.update(gate=g,geometry_passed=all(g.values()));summary['arms'][arm]=m
    for arm,v in regression.items():
        n=max(v['blocks'],1);v.update(safe_precision=v['selected_safe']/v['selected_blocks'] if v['selected_blocks'] else None,safe_recall=v['selected_safe']/v['safe_blocks'] if v['safe_blocks'] else None,fraction_MAE=v['fraction_error']/n/3,delta_MAE=v['delta_error']/n)
        summary['block_fit'][arm]=dict(v)
    d=run.OUT/'evaluation';d.mkdir(exist_ok=True)
    for name,rr in [('candidates.csv',rows),('targets.csv',targetrows),('blocks.csv',blockrows)]:
        with (d/name).open('w',newline='') as f:writer=csv.DictWriter(f,fieldnames=sorted({k for r in rr for k in r}));writer.writeheader();writer.writerows(rr)
    run.seal(run.OUT/'summary.json',summary)
    lines=['# 固定GD10删除块：真实五折结果','','固定1599父/原六帧/点域/竞争条件；20内交叉生成器+5外层生成器+10块验证器全部独立训练。B是封存概率的删除提议，G修正监督；S摘要MLP，R区域关系。','',
           '|输出|删除动作|父改善≥.02|完整NT增益≥.02|池价值目标|正确删污染|误删目标|好父损失>.02 / >.10|漂移|几何通过|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for arm in run.ARMS:
        m=summary['arms'][arm];lines.append(f"|{arm}|{m['actions']}|{m.get('parent_improved_02',0)}|{m['pool_gain02']}|{m['joint_value_targets']}|{m.get('deleted_other',0)}|{m.get('deleted_target',0)}|{m.get('good_degraded',0)} / {m.get('good_loss_gt_0.1',0)}|{m.get('drift',0)}|{m['geometry_passed']}|")
    lines+=['','GT纯块筛选仅后置诊断：'+str(summary['oracle'])+'。原最佳GT身份和稳定身份两种范围分别报告，不混用。',
            '', 'B上的验证器属于从内交叉G提议到封存B提议的迁移，不能声称只改B块开关而保持训练提议分布相同。G上配对结果是同配方对照。所有块每轮一次、父/场景等权；未知父不接受强实例监督。',
            '', '全部NT保留，池覆盖不会降，但父退化仍真实；输出没有补点。安全门固定.9，没有按外折GT调整，后果回归误差与块级精度见summary。动作更少不能算几何成功。',
            '', '本轮未接自身语义、SL或AP；开发200上游已暴露，仅新模块物理场景OOF。详见contract/各阶段封存、原协议、evaluation三份CSV及独立核验。']
    (run.OUT/'report.md').write_text('\n'.join(lines)+'\n');run.seal(run.OUT/'evaluation_frozen.json',dict(complete=True,outputs={str(p):sha(p) for p in list(d.iterdir())+[run.OUT/'summary.json',run.OUT/'report.md']}))


if __name__=='__main__':evaluate()
