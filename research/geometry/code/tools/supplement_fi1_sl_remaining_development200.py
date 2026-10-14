#!/usr/bin/env python3
"""Tie-aware budget witnesses and SL semantic-FP decomposition; cached only."""
from pathlib import Path
from collections import Counter,defaultdict
import json,sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import diagnose_fi1_sl_remaining_development200 as d
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha
OUT,S,SL=d.OUT,d.S,d.SL
def read(p):return json.loads(Path(p).read_text())
def rows(p):return [json.loads(x) for x in Path(p).read_text().splitlines()]


def main():
    c=read(OUT/'contract.json');assert read(OUT/'independent_verified.json')['complete']
    sys.addaudithook(d.Guard(c['scenes']).hook)
    rr=rows(OUT/'ranking.jsonl');targets=rows(OUT/'targets.jsonl');events=rows(OUT/'events.jsonl')
    missing={(x['scene'],x['gt_id'],x['threshold']) for x in events if x.get('kind')=='outside_S800'}
    counters={format(t,'.2f'):Counter() for t in c['thresholds']};budget=[];pressure=defaultdict(list)
    subgroup=read(S/'diagnostics/summary.json')['fixed548_subgroup']['rows']
    recovered={(x['scene'],x['gt_id']) for x in subgroup if x['recovered_at_075']};assert len(recovered)==424
    subgroup_budget=Counter();budget_probe_counts=Counter();union_rows=[]
    for s in c['scenes']:
        pool=rows(S/'predictions'/s/'members.jsonl');selected=read(SL/'predictions'/s/'semantic_SL.json')
        ix={d.key(m):i for i,m in enumerate(pool)}
        with np.load(S/'predictions'/s/'pool.npz') as z:ss=z['scores'].copy()
        with np.load(SL/'predictions'/s/'scores.npz') as z:ls=z['full'].copy()
        ranks={}
        for a,scores in (('S',ss),('L',ls)):
            order=sorted(range(len(pool)),key=lambda i:(-scores[i],not pool[i]['has_original_native_alias'],pool[i]['geometry_key'],pool[i]['class_index']))
            ranks[a]=np.empty(len(pool),dtype=int);ranks[a][order]=np.arange(1,len(pool)+1)
        assert {d.key(pool[i]) for i in np.flatnonzero(ranks['S']<=800)}=={d.key(m) for m in selected}
        census=read(c['jobs'][s]['census']);sizes={g['geometry_key']:g['point_count'] for g in read(c['jobs'][s]['binding']['geometry'])}
        evidence={e['geometry_key']:e for e in rows(c['jobs'][s]['binding']['evidence'])};match=read(SL/'evaluation'/s/'semantic_SL_matches.json')
        gtids=list(census['targets'])
        for t in c['thresholds']:
            tag=format(t,'.2f');status=d.src.member_status(selected,match,sizes,c['class_ids'],t)
            for m in selected:
                if status[d.key(m)]!='unmatched_FP':continue
                q=max(census['ious'][m['geometry_key']].values(),default=0.)
                kind='good_geometry_wrong_class' if q>t else 'geometry_not_qualified'
                counters[tag]['FP_'+kind]+=1
                for a,score in (('S',ss[ix[d.key(m)]]),('SL',m['score'])):
                    pressure[(tag,m['class_index'],a,kind)].append(float(score))
            for gid in gtids:
                if (s,int(gid),t) not in missing:continue
                ci=c['class_ids'].index(int(gid)//1000)
                good=[i for i,m in enumerate(pool) if m['class_index']==ci and census['ious'][m['geometry_key']][gid]>t]
                assert good
                best={a:int(ranks[a][good].min()) for a in ranks}
                assert best['S']>800
                ew=[d.semantic_witness(evidence[pool[i]['geometry_key']],ci) for i in good]
                w=dict(scene=s,gt_id=int(gid),class_index=ci,threshold=t,exact_S_rank=best['S'],exact_L_rank=best['L'],
                    best_fusion_rank=min(x['fusion_rank'] for x in ew),yolo_correct_vote=any(x['yolo_correct_votes']>0 for x in ew),
                    best_alpha_rank=min((x['alpha_rank'] for x in ew if x['alpha_rank'] is not None),default=None),
                    good_geometry_already_CA_output=any(census['ious'][m['geometry_key']][gid]>t for m in selected))
                budget.append(w)
                for a in ranks:
                    r=best[a];label='1-800' if r<=800 else '801-1600' if r<=1600 else '1601-3200' if r<=3200 else '>3200'
                    counters[tag][a+'_budget_rank_'+label]+=1
                for n in (1,5,10):counters[tag]['budget_fusion_top'+str(n)]+=w['best_fusion_rank']<=n
                counters[tag]['budget_already_CA_output']+=w['good_geometry_already_CA_output']
        for arm in ('NTU','screened_NTU','NTUp'):
            ps=read(OUT/'scenes'/s/'diagnosis.json')['probe'][arm]
            budget_probe_counts[arm+'_entered']+=ps['entered'];budget_probe_counts[arm+'_exited']+=ps['exited'];budget_probe_counts[arm+'_new_geometry_members']+=ps['new_geometry_members']
            lost=ps['coverage']['0.75']['lost_targets'];subgroup_budget[arm+'_recovered424_lost']+=sum((s,gid) in recovered for gid in lost)
    bythreshold={};augmented=[]
    for t in c['thresholds']:
        tag=format(t,'.2f');qq=[r for r in rr if r['threshold']==t];values=Counter()
        for r in qq:
            item=dict(scene=r['scene'],gt_id=r['gt_id'],class_index=r['class_index'],threshold=t)
            for a in ('S','SL'):
                for kind in ('good_geometry_wrong_class','geometry_not_qualified'):
                    pre=d.prefix(pressure[(tag,r['class_index'],a,kind)],r[a+'_best_score']);item[a+'_'+kind+'_prefix']=pre
                    values[a+'_'+kind+'_pressured_targets']+=pre['above']>0
            assert sum(item['SL_'+k+'_prefix']['above'] for k in ('good_geometry_wrong_class','geometry_not_qualified'))==r['SL_FP_prefix']['above']
            augmented.append(item)
        bythreshold[tag]=dict(counts=dict(counters[tag]),FP_pressure=dict(values))
    for x in events:
        if 'union_source' not in x or not x['no_NT_geometry']:continue
        union_rows.append(dict(scene=x['scene'],gt_id=x['gt_id'],source=x['union_source'],threshold=x['threshold'],
            witness_geometries=len(x['witnesses']),best_fusion_rank=min(w['own']['fusion_rank'] for w in x['witnesses']),
            regions=[r for w in x['witnesses'] for r in w['regions']]))
    for name,values in (('budget_witnesses',budget),('FP_pressure_decomposition',augmented),('true_union_increment_regions',union_rows)):
        (OUT/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in values))
    freeze_json(OUT/'supplement.json',dict(complete=True,thresholds=bythreshold,recovered424_probe_losses=dict(subgroup_budget),
        budget_probe_members=dict(budget_probe_counts),budget_witness_rows=len(budget),
        rank_note='Main events best_full_S/L_rank counts strictly higher scores and is a lower bound in ties. Supplement exact ranks use registered native/key/class tiebreak; S outside800 always rank>800.',
        caveat='Wrong class on good geometry is semantic FP, not class-agnostic FP. Prefixes and coverage are not AP contributions. No GT-derived inference rules.',
        inputs={str(q):sha(q) for q in (OUT/'contract.json',OUT/'independent_verified.json',S/'diagnostics/summary.json')},source_sha256=sha(__file__)))
    print('Supplement complete: tie-aware budget, semantic-FP geometry breakdown,424 probe losses.',flush=True)


if __name__=='__main__':main()
