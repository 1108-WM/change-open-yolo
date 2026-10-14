#!/usr/bin/env python3
"""Independent all-row R2 feature/weight audit and sampled boolean-mask IoU audit.

Does not import R2 production feature/target functions. Eight geometries per
training scan are chosen uniformly by sorted key, without inspecting labels.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy, rankdata
from tools.prepare_fi1_r0_r1_inputs import read,write,sha,digest
from evaluate.scannet200.eval_semantic_instance import PRED_ID_TO_ID, opt


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def features(geometry, evidence):
    y,a=evidence['yolo'],evidence['alpha']
    if not y['acquisition_complete'] or not a['acquisition_complete']:
        raise ValueError('incomplete acquisition')
    py=np.asarray(y['votes'],dtype=float) if y['available'] else np.zeros(198)
    if y['available']:py/=py.sum()
    pa=softmax(np.asarray(a['cosine'],dtype=float)*100) if a['available'] else np.zeros(198)
    count=int(y['available'])+int(a['available'])
    direct=(py+pa)/count if count else np.zeros(198)
    sources={r['candidate_source'] for r in geometry['legacy_members']}
    if any(r['append_only'] for r in geometry['dv3_plans']):sources.add('refined_union')
    both=y['available'] and a['available']
    js=(entropy(py,(py+pa)/2)+entropy(pa,(py+pa)/2))/2 if both else 0
    ry=(rankdata(-py,method='ordinal')-1)/197 if y['available'] else np.zeros(198)
    ra=(rankdata(-pa,method='ordinal')-1)/197 if a['available'] else np.zeros(198)
    hy=entropy(py) if y['available'] else 0
    ha=entropy(pa) if a['available'] else 0
    ymax,amax=py.max(),pa.max()
    agree=float(np.argmax(py)==np.argmax(pa)) if both else 0
    out=[]
    for c in range(198):
        out.append([direct[c],py[c],pa[c],a['cosine'][c] if a['available'] else 0,
            a['view_scale_std'][c] if a['available'] else 0,
            ry[c],ra[c],py[c]-ymax if y['available'] else 0,pa[c]-amax if a['available'] else 0,
            hy,ha,js,agree,np.log(1+geometry['point_count']),
            *[float(s in sources) for s in ['native','track','pair_union','refined_union']],
            float(not y['available']),float(not a['available'])])
    return np.asarray(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();ds=args.dataset.resolve()
    summary=read(ds/'summary.json');identity=read(ds/'identity.json')
    if not summary['complete'] or summary['identity_sha256']!=digest(identity):raise ValueError('incomplete dataset')
    args.output.mkdir(parents=True,exist_ok=False)
    ids=[int(PRED_ID_TO_ID[i]) for i in range(198)]
    reports=[]
    for report in summary['rows']:
        scene=report['scene_name'];receipt=read(ds/scene/'receipt.json')
        for name,h in receipt['outputs'].items():
            if sha(ds/scene/name)!=h:raise ValueError('changed feature file')
        if sha(report['R1_eligible'])!=report['R1_eligible_sha256']:raise ValueError('changed members')
        members=rows(report['R1_eligible']);geometry={r['geometry_key']:r for r in read(report['geometry_file'])}
        pred=Path(report['R1_eligible']).parent.parent
        ev=Path(read(pred/'summary.json')['evidence_root'])
        evidence={r['geometry_key']:r for r in rows(ev/scene/'evidence.jsonl')}
        reconstructed={k:features(g,evidence[k]) for k,g in geometry.items()}
        expected=np.asarray([reconstructed[r['geometry_key']][r['class_index']] for r in members])
        keys=np.asarray([r['geometry_key'] for r in members]);unique,inverse,counts=np.unique(keys,return_inverse=True,return_counts=True)
        with np.load(ds/scene/'features.npz',allow_pickle=False) as arrays:
            error=float(np.max(np.abs(arrays['X']-expected)))
            if error>1e-12:raise ValueError(f'{scene}: independent feature error {error}')
            if not np.allclose(arrays['weights'],1/len(unique)/counts[inverse],atol=1e-15,rtol=0):raise ValueError('weight mismatch')
            tested=0;target_error=0.
            if report['role']=='train':
                y=arrays['y']
                if not np.isfinite(y).all() or (y<0).any() or (y>1).any():raise ValueError('invalid targets')
                gtpath=ds/'ground_truth_train'/f'{scene}.txt'
                if sha(gtpath)!=receipt['inputs'][str(gtpath)]:raise ValueError('changed training GT')
                gt=np.loadtxt(gtpath,dtype=np.int64)
                masks={int(i):gt==i for i in np.unique(gt) if i>=1000 and int(i)//1000 in ids}
                masks={i:m for i,m in masks.items() if m.sum()>=int(opt['min_region_sizes'][0])}
                chosen=unique[np.unique(np.linspace(0,len(unique)-1,min(8,len(unique))).astype(int))]
                with np.load(report['points_file'],allow_pickle=False) as pts:
                    for key in chosen:
                        mask=np.zeros(len(gt),dtype=bool);mask[pts[geometry[key]['array_key']]]=True
                        targets=np.zeros(198)
                        if mask.sum()>=int(opt['min_region_sizes'][0]):
                            for i,other in masks.items():
                                targets[ids.index(i//1000)]=max(targets[ids.index(i//1000)],np.logical_and(mask,other).sum()/np.logical_or(mask,other).sum())
                        indexes=np.flatnonzero(keys==key)
                        wanted=np.asarray([targets[members[i]['class_index']] for i in indexes])
                        target_error=max(target_error,float(np.max(np.abs(y[indexes]-wanted))))
                        tested+=1
                if target_error>1e-14:raise ValueError('boolean-mask target audit differs')
            elif 'y' in arrays.files:raise ValueError('nontraining label leak')
        result={'scene_name':scene,'all_feature_rows':len(members),'max_feature_error':error,
            'sampled_target_geometries':tested,'max_sampled_target_error':target_error}
        reports.append(result);write(args.output/f'{scene}.json',result)
        print('independent R2 dataset audit',scene,'passed',flush=True)
    write(args.output/'summary.json',{'complete':True,'dataset_summary_sha256':sha(ds/'summary.json'),
        'code_sha256':sha(__file__),'scene_count':len(reports),'rows':reports,
        'max_feature_error':max(r['max_feature_error'] for r in reports),
        'sampled_target_geometries':sum(r['sampled_target_geometries'] for r in reports),
        'max_sampled_target_error':max(r['max_sampled_target_error'] for r in reports),
        'target_scope':'8 geometries per train scan selected uniformly by key, all eligible classes therein',
        'all_feature_rows_and_weights_checked':True,'nontraining_labels_absent':True})


if __name__=='__main__':main()
