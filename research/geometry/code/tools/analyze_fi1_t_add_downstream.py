#!/usr/bin/env python3
"""Independent AP integration plus evaluator-compatible provenance slot trace.
The official evaluator visits GTs, retains highest-score TP per GT, emits
lower-score duplicate slots, and can reuse predictions at low IoU. We retain
all slots (rather than silently claiming one prediction = one TP at AP25).
"""
from pathlib import Path
from collections import Counter
import sys,json,hashlib,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import run_fi1_t_add_downstream as run
from tools.fi1_frontend_protocol import freeze_json,write_json,sha256_file as sha
from tools.fi1_r0_r1_protocol import original_aliases
OUT=run.OUT;read=run.read

def overlap(p,g):
    hit=p.get('intersection',g.get('intersection'));return hit/(p['vert_count']+g['vert_count']-hit)
def valid(g,ev):return g['instance_id']>=1000 and g['vert_count']>=ev.opt['min_region_sizes'][0] and g['med_dist']<=ev.opt['distance_threshes'][0] and g['dist_conf']>=ev.opt['distance_confs'][0]
def trace(match,ev,threshold):
    visited=set();slots=[];hardfn=Counter();presence={};covered=set()
    for label in ev.CLASS_LABELS:
        gs=[g for g in match['gt'][label] if valid(g,ev)];ps=match['pred'][label];presence[label]=(bool(gs),bool(ps));hardfn[label]=0
        for g in gs:
            winner=None;found=False
            for p in g['matched_pred']:
                if p['uuid'] in visited:continue
                if overlap(p,g)>threshold:
                    if winner is None:winner=p;visited.add(p['uuid']);found=True
                    else:
                        if p['confidence']>winner['confidence']:loser=winner;winner=p
                        else:loser=p
                        slots.append(dict(label=label,uuid=loser['uuid'],gt_id=g['instance_id'],score=loser['confidence'],kind='duplicate_FP',truth=0))
            if not found:hardfn[label]+=1
            if winner is not None:
                slots.append(dict(label=label,uuid=winner['uuid'],gt_id=g['instance_id'],score=winner['confidence'],kind='TP',truth=1));covered.add(g['instance_id'])
        for p in ps:
            if any(overlap(p,g)>threshold for g in p['matched_gt']):continue
            ignore=p['void_intersection']+sum(g['intersection'] for g in p['matched_gt'] if g['instance_id']<1000 or not valid(g,ev))
            if ignore/p['vert_count']<=threshold:slots.append(dict(label=label,uuid=p['uuid'],gt_id=None,score=p['confidence'],kind='unmatched_FP',truth=0))
    return slots,hardfn,presence,covered

def integrate(slots,fn,presence,labels):
    hardfn=fn
    result=[]
    for label in labels:
        gt,pred=presence[label]
        if not gt:result.append(float('nan'));continue
        if not pred:result.append(0.);continue
        rr=[r for r in slots if r['label']==label];scores=np.array([r['score'] for r in rr]);truth=np.array([r['truth'] for r in rr],float);ix=np.argsort(scores);scores=scores[ix];cs=np.cumsum(truth[ix]);total=cs[-1] if len(cs) else 0;cs=np.append(cs,0);_,starts=np.unique(scores,return_index=True)
        precision=np.zeros(len(starts)+1);recall=precision.copy()
        for j,k in enumerate(starts):
            tp=total-cs[k-1];fp=len(scores)-k-tp;fn=cs[k-1]+hardfn[label];precision[j]=tp/(tp+fp);recall[j]=tp/(tp+fn)
        precision[-1]=1.;steps=np.convolve(np.r_[recall[0],recall,0.],[-.5,0.,.5],'valid');result.append(float(precision@steps))
    return np.array(result)

def main():
    c=read(OUT/'contract.json');ap=read(OUT/'evaluation_summary.json')['AP'];run.ctl.base.verify_files(c['code']);run.ctl.base.verify_files(read(OUT/'predictions_frozen.json')['outputs']);run.ctl.base.verify_files(read(OUT/'matches_frozen.json')['outputs'])
    ev=run.ctl.base.evaluator();ca,_,_=run.ctl.class_agnostic_evaluator();thresholds=list(map(float,ev.opt['overlaps']));records=[];exits=[];changes=[];examples=[];summary={};context_checks=0;independent={v:{a:{t:dict(slots=[],fn=Counter(),presence={l:(False,False) for l in e.CLASS_LABELS}) for t in thresholds} for a in run.ARMS} for v,e in [('semantic',ev),('class_agnostic',ca)]}
    for s in c['scenes']:
        print('diagnose/verify',s,flush=True);gs,pts,evidence=run.expanded(c,s);new={g['geometry_key'] for g in gs if g.get('arms')};gby={g['geometry_key']:g for g in gs};gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);gt_sizes=dict(zip(*np.unique(gt,return_counts=True)))
        for r in read(OUT/'evidence'/s/'evidence.json'):
            assert r['ground_truth_read'] is False and r['ap_computed'] is False and r['alpha']['acquisition_complete'] and r['yolo']['acquisition_complete'];assert hashlib.sha256(pts[r['geometry_key']].astype('<i8').tobytes()).hexdigest()==r['points_sha256']
            assert np.array_equal(run.src.geometry_features(gby[r['geometry_key']],r)[:,14:18],np.tile([0,0,0,1],(198,1)))
        for a in ('A','B'):
            mm=read(OUT/'predictions'/s/(a+'_members.json'))
            with np.load(OUT/'predictions'/s/(a+'_pool.npz')) as z:x=z['x26'].copy()
            index={run.ctl.shared.key(m):i for i,m in enumerate(mm)}
            probes=read(OUT/'predictions'/s/('semantic_'+a+'_append.json'))[:20]
            probes += [m for m in mm if m['geometry_key'] in new][:20]
            for m in probes:
                m=mm[index[run.ctl.shared.key(m)]]
                p=pts[m['geometry_key']];others=[r for r in mm if r['class_index']==m['class_index'] and r['geometry_key']!=m['geometry_key']];hits=np.array([len(np.intersect1d(p,pts[r['geometry_key']],assume_unique=True)) for r in others]);sizes=np.array([len(pts[r['geometry_key']]) for r in others]);io=hits/(len(p)+sizes-hits);sc=np.array([r['score'] for r in others]);strong=io>.5
                expected=np.array([io.max(initial=0),np.log1p(strong.sum()),(io*sc).max(initial=0),m['score']-sc[strong].max() if strong.any() else 0,np.mean(sc[strong]>m['score']) if strong.any() else 0,(hits/len(p)).max(initial=0)])
                assert np.array_equal(x[index[run.ctl.shared.key(m)],20:],expected);context_checks+=1
        for view,evaluator in [('semantic',ev),('class_agnostic',ca)]:
            rows={a:read(OUT/'predictions'/s/(view+'_'+a+'.json')) for a in run.ARMS};keys=lambda r:run.ctl.shared.key(r) if view=='semantic' else r['geometry_key'];sets={a:{keys(r) for r in rr} for a,rr in rows.items()};alltrace={};byuuid={}
            for arm in run.ARMS:
                match=read(OUT/'evaluation'/s/(view+'_'+arm+'_matches.json'));filtered=[r for r in rows[arm] if len(pts[r['geometry_key']])>=100];pr=[p for ps in match['pred'].values() for p in ps];pmap={p['uuid']:filtered[p['pred_id']] for p in pr};byuuid[arm]=pmap;predbykey={keys(pmap[p['uuid']]):p for p in pr}
                for t in thresholds:
                    slots,fn,pres,cov=trace(match,evaluator,t);alltrace[arm,t]=(slots,cov)
                    agg=independent[view][arm][t];agg['slots'].extend(slots);agg['fn'].update(fn)
                    for l,(g,p) in pres.items():oldg,oldp=agg['presence'][l];agg['presence'][l]=(oldg or g,oldp or p)
                    byslot={u:[r for r in slots if r['uuid']==u] for u in pmap}
                    # Include predictions below evaluator minimum as explicit skipped rows.
                    for r in rows[arm]:
                        key=keys(r);p=predbykey.get(key);sr=byslot.get(p['uuid'],[]) if p else [];status=[z['kind'] for z in sr]
                        if not status:status=['minimum_size_skipped' if p is None else 'ignored_or_noncounted']
                        if 'unmatched_FP' in status and view=='semantic':
                            point=pts[r['geometry_key']];labs,n=np.unique(gt[point],return_counts=True);possible=False
                            for gid,hit in zip(labs,n):
                                size=int(gt_sizes[gid])
                                if gid>=1000 and size>=100 and gid//1000 in c['class_ids'] and hit/(len(point)+size-hit)>t and gid//1000!=c['class_ids'][r['class_index']]:possible=True;break
                            status=['wrong_class_FP' if z=='unmatched_FP' and possible else 'geometry_insufficient_FP' if z=='unmatched_FP' else z for z in status]
                        source='new_T' if r['geometry_key'] in new else run.ctl.source_meta(gby[r['geometry_key']]) if hasattr(run.ctl,'source_meta') else '+'.join(sorted({a['source'] for a in original_aliases(gby[r['geometry_key']])}))
                        records.append(dict(scene=s,view=view,arm=arm,threshold=t,geometry_key=r['geometry_key'],class_index=r.get('class_index',r.get('winning_class_index')),score=r['score'],source=source,new=r['geometry_key'] in new,status=status,TP_slots=sum(z['kind']=='TP' for z in sr),duplicate_FP_slots=sum(z['kind']=='duplicate_FP' for z in sr),gt_ids=[z['gt_id'] for z in sr if z['kind']=='TP']))
                for r in rows['NT']:
                    if keys(r) not in sets[arm]:
                        exits.append(dict(scene=s,view=view,arm=arm,**r,sources=sorted({a['source'] for a in original_aliases(gby[r['geometry_key']])})))
            for arm in run.ARMS[1:]:
                reference=arm[0]+'_context' if arm.endswith('append') else 'NT'
                for reference in dict.fromkeys(['NT',reference]):
                    for t in thresholds:
                        slots,cov=alltrace[arm,t];bs,bcov=alltrace[reference,t];gained=cov-bcov;lost=bcov-cov
                        refTP={keys(byuuid[reference][z['uuid']]) for z in bs if z['kind']=='TP'};curTP={keys(byuuid[arm][z['uuid']]) for z in slots if z['kind']=='TP'}
                        changes.append(dict(scene=s,view=view,arm=arm,reference=reference,threshold=t,gained_targets=sorted(gained),lost_targets=sorted(lost),old_TP_members_exited=len(refTP-sets[arm]),old_TP_members_retained_as_TP=len(refTP&curTP),old_TP_members_retained_but_lost_TP=len((refTP&sets[arm])-curTP),new_T_TP_slots=sum(z['kind']=='TP' and byuuid[arm][z['uuid']]['geometry_key'] in new for z in slots),new_T_duplicate_FP_slots=sum(z['kind']=='duplicate_FP' and byuuid[arm][z['uuid']]['geometry_key'] in new for z in slots)))
                        if view=='semantic' and t==.5:
                            for z in slots:
                                m=byuuid[arm][z['uuid']]
                                if z['kind']!='TP' or m['geometry_key'] not in new:continue
                                target=z['gt_id'];nsize=int((gt==target).sum());candidates=[]
                                for g in gs:
                                    if g['geometry_key'] in new or not any(a['source']=='native' for a in original_aliases(g)):continue
                                    point=pts[g['geometry_key']];hit=int((gt[point]==target).sum());value=hit/(len(point)+nsize-hit);selected=[r for r in rows['NT'] if r['geometry_key']==g['geometry_key']]
                                    candidates.append(dict(geometry_key=g['geometry_key'],iou=value,selected=selected))
                                best=max(candidates,key=lambda r:r['iou']);point=pts[m['geometry_key']];hit=int((gt[point]==target).sum());examples.append(dict(scene=s,arm=arm,reference=reference,target=target,new_detection=m,new_iou=hit/(len(point)+nsize-hit),new_target_vs_reference=target in gained,original_N_best=best))
    maxerr=0.
    for view,evaluator in [('semantic',ev),('class_agnostic',ca)]:
        for arm in run.ARMS:
            computed=np.stack([integrate(**independent[view][arm][t],labels=evaluator.CLASS_LABELS) for t in thresholds],axis=1)[None]
            official=np.load(OUT/'evaluation'/(view+'_'+arm+'_AP.npy'));assert np.array_equal(np.isnan(computed),np.isnan(official));err=float(np.nanmax(np.abs(computed-official)));assert err<1e-12;maxerr=max(maxerr,err)
    for view in run.VIEWS:
        summary[view]={}
        for arm in run.ARMS:
            rr=[r for r in records if r['view']==view and r['arm']==arm and r['threshold']==.5];newrr=[r for r in rr if r['new']];summary[view][arm]=dict(AP=ap[view][arm],new_selected=len(newrr),new_TP_slots=sum(r['TP_slots'] for r in newrr),new_duplicate_FP_slots=sum(r['duplicate_FP_slots'] for r in newrr),new_status_counts=dict(Counter(z for r in newrr for z in r['status'])),original_exits=len([r for r in exits if r['view']==view and r['arm']==arm]))
    d=OUT/'diagnostics';d.mkdir(exist_ok=True)
    for name,val in [('predictions',records),('original_exits',exits),('target_changes',changes),('new_TP_vs_original_N',examples),('summary',summary)]:freeze_json(d/(name+'.json'),val)
    verification=dict(complete=True,AP_independent_integration_max_error=maxerr,independent_context_rows=context_checks,mask_own_semantics_and_fixed_onehot_verified=True,all_threshold_trace=True,low_threshold_note='Official GT-ordered matcher can reuse one UUID in several GT slots at low IoU; all slots preserved, not falsely counted as unique prediction TP. At .5+ masks cannot overlap two disjoint instances above threshold.',outputs={str(p):sha(p) for p in d.iterdir()})
    freeze_json(OUT/'verification.json',verification)
    lines=['# T补点：冻结下游五组AP内部验证','', '固定原24场景、95采样T父及已封存A/B。无新训练、几何重生成、阈值改变或GitHub发布。全部78个跨组唯一新点集获得自身YOLO40/Alpha20语义证据后，五组预测先封存，再读取GT。A追加73个非原NT点集；B19个，其中14个与A相同。原NT重复不重复投票。','', '本批是已暴露内部开发验证。实际L训练回执核查各场景/同物理变体均不在对应折训练名单；上游/B26仍有开发暴露。新增来源固定映射已有refined_union槽，属于SL输入分布扩展，不等同原始T。预算均K800，CA仅折叠真实入选成员，不补位。','', '|组|semantic AP|AP50|AP25|CA AP|CA AP50|CA AP25|','|---|---:|---:|---:|---:|---:|---:|']
    for arm in run.ARMS:
        vals=[]
        for view in run.VIEWS:
            met=ap[view][arm];vals.extend([met[k]*100 for k in ('ap','ap50','ap25')])
        lines.append('|'+arm+'|'+'|'.join(f'{v:.6f}' for v in vals)+'|')
    lines+=['','所有AP为整批官方聚合（百分比），不以场景均值代替。','', '## 新T与原候选代价（IoU严格>0.5）','', '|组/语义视图|新T入选|新T TP槽|重复FP槽|原成员退出|','|---|---:|---:|---:|---:|']
    for view in run.VIEWS:
        for arm in run.ARMS[1:]:
            r=summary[view][arm];lines.append(f"|{arm}/{view}|{r['new_selected']}|{r['new_TP_slots']}|{r['new_duplicate_FP_slots']}|{r['original_exits']}|")
    lines+=['','新增T按每阈值记录TP、重复FP、错类FP、几何不足FP、ignored/小mask；原N/T退出及旧TP退出/失去TP记录完整保留。`target_changes.json`逐场景逐阈值同时比较NT与对应context，避免把上下文改变归于新T。官方GT顺序匹配在低阈值可能重复利用预测，记录的是官方匹配槽，不把AP25槽冒称唯一检测。','', '## 归因与判断','']
    for a in ('A','B'):
        delta=100*(ap['semantic'][a+'_append']['ap']-ap['semantic']['NT']['ap']);context=100*(ap['semantic'][a+'_context']['ap']-ap['semantic']['NT']['ap']);specific=100*(ap['semantic'][a+'_append']['ap']-ap['semantic'][a+'_context']['ap']);cadelta=100*(ap['class_agnostic'][a+'_append']['ap']-ap['class_agnostic']['NT']['ap'])
        lines.append(f'{a}：semantic AP较NT {delta:+.6f}个百分点，其中context {context:+.6f}，append相对context {specific:+.6f}；CA AP较NT {cadelta:+.6f}。')
        for reference in ('NT',a+'_context'):
            rr=[r for r in changes if r['view']=='semantic' and r['arm']==a+'_append' and r['reference']==reference and r['threshold']==.5];lines.append(f"在AP50相对{reference}新增/丢失目标 {sum(len(r['gained_targets']) for r in rr)}/{sum(len(r['lost_targets']) for r in rr)}；旧TP成员退出 {sum(r['old_TP_members_exited'] for r in rr)}。")
    gain=any(ap['semantic'][a+'_append']['ap']>ap['semantic']['NT']['ap'] for a in ('A','B'))
    lines+=['', '存在固定下游semantic AP正变化，应承认为内部候选利用信号；不改写此前几何池增量0，不宣称几何上限提高或独立泛化。需结合context归因、CA与退出代价，不据小差异选生产赢家。' if gain else '两组追加均未改善semantic AP，当前固定补点规则没有取得semantic AP支持；不在本批修改SL、规则或阈值补救。CA现象另列，不能混称语义成功。','',f"核验：原NT成员/x26/B26/类别/分数/排序精确回放；所有扩展选择的完整L与padding路径误差受预注册1e-7约束；自身点集hash与来源one-hot通过；独立点交集上下文复算{context_checks}行；全类别全阈值AP独立积分最大误差{maxerr:.3g}；官方匹配的双向点数普查通过。",'', '代码：`run_fi1_t_add_downstream.py`、`analyze_fi1_t_add_downstream.py`。合同、转换记录、fold_audit、自身证据、五组预测、AP数组、全部匹配和诊断保存在本目录。未修改历史结果或最小阅读入口。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');freeze_json(OUT/'completion_verified.json',dict(complete=True,verification_sha256=sha(OUT/'verification.json'),report_sha256=sha(OUT/'report.md'),analysis_code_sha256=sha(Path(__file__)),training=False,production_modified=False,github_push=False));write_json(OUT/'execution.json',dict(stage='complete',time=time.time()));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
