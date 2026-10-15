"""Independent model replay and dev55 point-set outcome audit; no fitting."""
from pathlib import Path
from collections import Counter
import sys,os,subprocess
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'research/geometry/code/tools'))
import numpy as np
import joblib
from threadpoolctl import threadpool_limits
import run_fi1_relation_train200_dev55 as r

def main():
    def hook(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,Path)):return
        mode,flags=args[1:3]
        writing=(isinstance(mode,str) and any(k in mode for k in 'wax+')) or (isinstance(flags,int) and bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT)))
        if writing and not Path(os.fsdecode(args[0])).resolve().is_relative_to(HERE):raise PermissionError('write outside audit')
    sys.addaudithook(hook)
    a=r.a;c=a.read(r.OUT/'contract.json');old=a.read(r.SPLIT)
    assert c['train_scenes']==old['train_scenes'] and c['scenes']==old['scenes']
    assert not r.physical(c['train_scenes'])&r.physical(c['scenes'])
    assert set(c['small_scenes'])<set(c['train_scenes'])
    r.verify(c['code'])
    for name in ('predictions_frozen.json','evaluation_frozen.json'):r.verify(a.read(r.OUT/name)['outputs'])
    rr,x=r.matrix('dev55');tr,tx=r.matrix('train200')
    lookup={}
    for s in c['train_scenes']+c['scenes']:
        lookup.update({(s,q['pair_id']):q['relation'] for q in a.read(r.OUT/'labels'/f'{s}.json')['pairs']})
    idx=np.array([i for i,q in enumerate(tr) if lookup[q['scene'],q['pair_id']] in ('same','different')])
    y=np.array([int(lookup[tr[i]['scene'],tr[i]['pair_id']]=='same') for i in idx])
    train=np.load(r.OUT/'models/training.npz');np.testing.assert_array_equal(train['row_indices'],idx);np.testing.assert_array_equal(train['y'],y)
    counts=Counter(tr[i]['scene'] for i in idx)
    np.testing.assert_allclose(train['weights'],[len(idx)/len(counts)/counts[tr[i]['scene']] for i in idx],rtol=1e-13)
    assert a.sha(r.OUT/'models/reference82.joblib')==a.sha(r.SMALL/'models/fold0/double.joblib')
    decisions={arm:a.read(r.OUT/(arm+'_decisions.json')) for arm in r.ARMS}
    for arm in r.ARMS:
        model=joblib.load(r.OUT/'models'/(arm+'.joblib'))
        assert all(model.get_params()[k]==v for k,v in c['model'].items())
        with threadpool_limits(4):p=model.predict_proba(x)[:,1];tp=model.predict_proba(tx)[:,1]
        np.testing.assert_array_equal(p,np.load(r.OUT/(arm+'_scores.npy')));np.testing.assert_array_equal(tp,np.load(r.OUT/(arm+'_train_scores.npy')))
        for s in c['scenes']:
            ii=[i for i,q in enumerate(rr) if q['scene']==s];used=set();accepted=set()
            for i in sorted(ii,key=lambda i:(-p[i],rr[i]['pair_id'])):
                q=rr[i]
                if q['eligible'] and p[i]>=.9 and not ({q['left'],q['right']}&used):
                    used.update((q['left'],q['right']));accepted.add(q['pair_id'])
            assert accepted=={decisions[arm][i]['pair_id'] for i in ii if decisions[arm][i]['state']=='accept'}
    pairrows={(q['scene'],q['pair_id']):q for q in a.read(r.OUT/'evaluation/supply_pairs.json')}
    targets={arm:{(q['scene'],q['gt_id']):q for q in a.read(r.OUT/'evaluation'/(arm+'_targets.json'))} for arm in r.ARMS}
    classes=old['class_ids'];checks=Counter();opportunities=[];risk={arm:Counter() for arm in r.ARMS}
    def iou(pp,gt,ids,sizes,small=False):
        if len(pp)<100 and not small:return np.zeros(len(ids))
        hits=np.array([np.count_nonzero(gt[pp]==g) for g in ids])
        return hits/(len(pp)+sizes-hits)
    def strict(pp,gt):
        ids,counts=np.unique(gt[pp],return_counts=True);known=ids>=1000
        j=int(np.argmax(np.where(known,counts,0))) if known.any() else None
        mixed=np.count_nonzero(known&(counts>=20))>=2
        clear=not mixed and j is not None and counts[j]>=20 and counts[j]/len(pp)>=.9
        return ('mixed' if mixed else 'clear' if clear else 'unknown',int(ids[j]) if clear else None)
    for s in c['scenes']:
        gp=c['jobs'][s]['job']['GT'];assert a.sha(gp)==c['jobs'][s]['GT_sha256'];gt=np.loadtxt(gp,dtype=np.int64)
        ids,sizes=np.unique(gt,return_counts=True);ok=(ids>=1000)&(sizes>=100)&np.isin(ids//1000,classes);ids,sizes=ids[ok],sizes[ok]
        nodes=a.read(r.OUT/'geometry/predictions'/s/'nodes.json')
        with np.load(r.OUT/'geometry/predictions'/s/'data.npz') as z:
            points={q['node']:z['n%04d'%q['node']] for q in nodes}
            base=np.zeros(len(ids))
            for q in nodes:
                if q['in_original_NT']:base=np.maximum(base,iou(points[q['node']],gt,ids,sizes))
            identities={}
            for q in nodes:
                pp=points[q['node']];v=iou(pp,gt,ids,sizes,True);j=int(v.argmax()) if len(v) and v.max()>0 else None
                stable=j is not None and np.count_nonzero(gt[pp]==ids[j])>=20 and np.mean(gt[pp]==ids[j])>=.6
                identities[q['node']]=(bool(stable),j,float(v[j]) if j is not None and len(pp)>=100 else 0,strict(pp,gt))
            best={arm:base.copy() for arm in r.ARMS}
            actions={arm:{q['pair_id']:q for q in decisions[arm] if q['scene']==s} for arm in r.ARMS}
            emitted={}
            for arm in r.ARMS:
                with np.load(r.OUT/'predictions'/arm/s/'unions.npz') as zz:emitted[arm]={k:zz[k] for k in zz.files}
            for q in (q for q in rr if q['scene']==s):
                pp=np.union1d(points[q['left']],points[q['right']]);np.testing.assert_array_equal(pp,z[q['pair_id']+'_union'])
                row=pairrows[s,q['pair_id']];v=iou(pp,gt,ids,sizes);left,right=identities[q['left']],identities[q['right']]
                relation=('same' if left[1]==right[1] else 'different') if left[0] and right[0] else 'unresolved'
                ls,rs=left[3],right[3]
                sr=('same' if ls[1]==rs[1] else 'different') if ls[0]==rs[0]=='clear' else 'mixed' if 'mixed' in (ls[0],rs[0]) else 'unknown'
                safe=relation=='same' and np.mean(gt[pp]==ids[left[1]])>=.9 and v[left[1]]-base[left[1]]>=.02-1e-12
                harm=any(m[0] and m[2]>=.9 and m[2]-v[m[1]]>.02+1e-12 for m in (left,right))
                assert row['GT_relation_previous']==relation and row['GT_relation_strict']==sr and row['safe_gain02']==bool(safe) and row['good_endpoint_loss_gt02']==harm
                if safe:opportunities.append(dict(scene=s,pair_id=q['pair_id'],target=int(ids[left[1]]),strict=sr,eligible=q['eligible'],delta=float(v[left[1]]-base[left[1]])))
                checks['pairs_and_risk']+=1
                for arm in r.ARMS:
                    act=actions[arm][q['pair_id']]
                    if act['state']=='accept':risk[arm]['strict_'+sr]+=1;risk[arm]['previous_'+relation]+=1;risk[arm]['good_copy_harm']+=int(harm)
                    if act['emit']:
                        np.testing.assert_array_equal(emitted[arm][q['pair_id']+'_union'],pp);best[arm]=np.maximum(best[arm],v);checks['emitted_unions']+=1
            for arm in r.ARMS:
                for j,g in enumerate(ids):
                    q=targets[arm][s,int(g)];assert abs(q['baseline']-base[j])<1e-12 and abs(q['expanded']-best[arm][j])<1e-12;checks['target_rows']+=1
                    risk[arm]['gain02_targets']+=int(best[arm][j]-base[j]>=.02-1e-12)
        print('AUDIT',s,checks['pairs_and_risk'],flush=True)
    for arm in r.ARMS:
        score=np.load(r.OUT/(arm+'_scores.npy'));index={(q['scene'],q['pair_id']):i for i,q in enumerate(rr)}
        negatives=[i for i,q in enumerate(rr) if lookup[q['scene'],q['pair_id']]=='different']
        for q in opportunities:
            p=float(score[index[q['scene'],q['pair_id']]])
            q[arm]=dict(score=p,negative_scores_at_least_as_high=sum(score[i]>=p for i in negatives))
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(r.HERE),'-p','test_fi1_relation_train200_dev55.py','-v'],capture_output=True,text=True)
    assert tests.returncode==0,tests.stderr
    a.seal(HERE/'verification.json',dict(complete=True,checks=checks,risk=risk,opportunities=opportunities,tests=tests.stderr,
        train_rows=len(idx),effective_training_scenes=len(counts),new_fits=0,
        scope='New dev55 IoU/identity/union implementation, model replay and split/weight checks. No full raw255 visibility or sensor feature re-extraction.'))
    print('AUDIT_COMPLETE',checks,flush=True)

if __name__=='__main__':main()
