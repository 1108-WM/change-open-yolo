import json,csv
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
R=Path('/home/jia/Wm/wm_open-yolo/OpenYOLO3D');P=R/'.local/fi1_local_edit_ablation_development200_20261012_v1';S=R/'.local/fi1_candidate_local_repair_learning_development200_20261012_v1';O=R/'.local/fi1_local_edit_ablation_independent_analysis_20261012'
c=json.load(open(P/'contract.json'));baseline={(r['scene'],int(r['gt_id'])):float(r['original_NT_best']) for r in csv.DictReader(open(P/'evaluation/targets.csv'))};stats=defaultdict(Counter);best=defaultdict(dict)
for s in c['scenes']:
 gt=np.loadtxt(c['jobs'][s]['GT'],dtype=np.int64);meta=json.load(open(P/'labels'/s/'meta.json'))
 with np.load(S/'features'/s/'data.npz') as z,np.load(P/'predictions'/s/'predictions.npz') as pr:
  for m in meta:
   gid=m['target']
   if gid is None:continue
   cid=m['candidate_id'];parent=z[cid+'_parent'];halo=z[cid+'_halo'];member=np.isin(halo,parent);seen=z[cid+'_gd10_selected6_groups']>=2;nn=z[cid+'_neighbors'];n=int((gt==gid).sum());hit=int((gt[parent]==gid).sum())
   for a in ['B','C']:
    prob=pr[cid+'_'+a+'_probability'].astype(np.float32);removed=np.flatnonzero(member&seen&(prob[:,1]>=.8));nr=len(removed)
    if not nr:continue
    lookup=np.full(len(halo),-1);lookup[removed]=np.arange(nr);edges=lookup[nn[removed]];ii,jj=np.nonzero(edges>=0);adj=csr_matrix((np.ones(len(ii)),(ii,edges[ii,jj])),shape=(nr,nr));nc,labels=connected_components(adj,directed=False)
    sz=np.bincount(labels);true=np.bincount(labels,weights=gt[halo[removed]]==gid,minlength=nc).astype(int);unknown=np.bincount(labels,weights=gt[halo[removed]]<1000,minlength=nc).astype(int)
    for mode,pure in [('zero_target',true==0),('known_other_only',(true==0)&(unknown==0))]:
     name=a+'::'+mode;stats[name].update(proposed_components=nc,accepted_oracle_components=int(pure.sum()),accepted_points=int(sz[pure].sum()))
     kept=np.setdiff1d(parent,halo[removed[pure[labels]]]);q=hit/(len(kept)+n-hit) if len(kept)>=100 else hit/(len(parent)+n-hit);key=(s,gid);best[name][key]=max(best[name].get(key,baseline[key]),q)
result={a:dict(stats[a],gain02=sum(x>=baseline[k]+.02-1e-12 for k,x in v.items()),value=sum(x>=baseline[k]+.02-1e-12 or any(baseline[k]<=t<x for t in (.75,.9,.95)) for k,x in v.items())) for a,v in best.items()}
out=dict(scope='Posthoc diagnostic; raw proposals use frozen probabilities p(other)>=.8, >=2 groups and existing undirected 8-NN graph. C candidate gate disabled; components form before minimum-mask gate, final output below100 reverts to parent. GT accepts zero-target components, additionally excluding unknown-labeled points in known_other_only arm. Not an optimal upper bound, learned output, or deployment rule.',results=result)
(O/'component_probe.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
