import sys,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np,torch
R=Path('/home/jia/Wm/wm_open-yolo/OpenYOLO3D');sys.path.insert(0,str(R))
from tools import run_fi1_local_edit_ablation as run
P=run.OUT;S=run.SOURCE;O=R/'.local/fi1_local_edit_ablation_independent_analysis_20261012'
torch.set_num_threads(2)
totals=defaultdict(Counter);rows=[]
for fold in range(5):
 keys=json.load(open(P/f'pilot/B/fold{fold}/fit.json'))['keys']
 models={a:run.learn.model(a).eval() for a in ('A0','B','C')}
 for a,m in models.items():m.load_state_dict(torch.load(P/f'models/{a}/fold{fold}/model.pt',map_location='cpu',weights_only=True))
 for s,cid in keys:
  with np.load(S/'features'/s/'data.npz') as z,np.load(P/'conditions'/s/'data.npz') as co,np.load(P/'labels'/s/'data.npz') as la,np.load(P/'predictions'/s/'predictions.npz') as pr:
   e=dict(x=z[cid+'_gd10_selected6_x'].astype(np.float32),neighbors=z[cid+'_neighbors'].astype(np.int64),weights=z[cid+'_neighbor_weights'].astype(np.float32),parent=z[cid+'_parent'],halo=z[cid+'_halo'],visible=z[cid+'_gd10_selected6_groups'],condition={k:co[cid+'_'+k] for k in ('membership','distance','valid','overlap','representatives','own_distance')})
   y=la[cid+'_y'];parent=np.isin(e['halo'],e['parent']);seen=e['visible']>=2
   for a,m in models.items():
    with torch.inference_mode():_,_,logits,ca,_=run.forward(m,a,[e],device='cpu');prob=logits.softmax(1).numpy().astype(np.float16);cp=None if ca is None else ca.softmax(1)[0].numpy().astype(np.float16)
    for where,p,gate_p in [('formal_train',prob,cp),('saved_OOF',pr[cid+'_'+a+'_probability'],pr[cid+'_C_candidate_probability'] if a=='C' else None)]:
     for gate_on in ([False,True] if a=='C' else [False]):
      actual,_=run.learn.repair(e['parent'],e['halo'],p,e['visible'],a if gate_on else 'B',gate_p)
      state=np.isin(e['halo'],actual);de=parent&~state;ad=~parent&state
      count=Counter(delete_opportunity=int((parent&seen&(y==1)).sum()),add_opportunity=int((~parent&seen&(y==0)).sum()),correct_delete=int((de&(y==1)).sum()),wrong_delete=int((de&(y==0)).sum()),correct_add=int((ad&(y==0)).sum()),wrong_add=int((ad&(y==1)).sum()),unknown_edits=int(((de|ad)&(y==2)).sum()),actions=int((de|ad).any()),parents=1)
      key=f'{a}::{where}::'+('gate' if gate_on else 'ungated');totals[key].update(count)
      rows.append(dict(fold=fold,scene=s,candidate=cid,variant=a,where=where,gate_on=gate_on,counts=dict(count)))
 print('formal training subset checked',fold,flush=True)
result={}
for key,t in totals.items():
 d=dict(t);d['delete_recall']=t['correct_delete']/t['delete_opportunity'];d['known_delete_precision']=t['correct_delete']/max(1,t['correct_delete']+t['wrong_delete']);d['known_add_precision']=t['correct_add']/max(1,t['correct_add']+t['wrong_add']);result[key]=d
out=dict(scope='Diagnostic inference only. Formal final 12-epoch models evaluated on pre-existing 20 training-only pilot parents per fold; same candidate OOF probabilities as reference, not all training scenes. No training or threshold changes.',results=result,rows=rows)
(O/'formal_train_subset.json').write_text(json.dumps(out,indent=2));print(json.dumps(result,indent=2))
