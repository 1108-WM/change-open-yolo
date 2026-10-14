import sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools import run_fi1_deletion_blocks as run
out=Path(__file__).resolve().parent;exp=run.OUT;c=run.read(exp/'contract.json');torch.set_num_threads(2)
result={'training_heads':[],'replays':[]}
for fold in range(5):
    cases=[(s,'c000') for s in c['scenes'] if c['folds'][s]==fold][:1]
    cases += [(r['scene'],r['candidate_id']) for r in run.read(exp/'block_opportunity_diagnosis.json')['witnesses'] if c['folds'][r['scene']]==fold]
    models={}
    with np.load(exp/'training_blocks'/f'outer{fold}'/'data.npz') as z:tx=z['x'].astype(np.float32);truth=z['y']
    for kind in ('summary','relation'):
        m=run.core.BlockVerifier(kind).eval();m.load_state_dict(torch.load(exp/'verifiers'/f'outer{fold}'/kind/'model.pt',weights_only=True,map_location='cpu'));models[kind]=m
        with torch.inference_mode():p=run.core.decoded(m(torch.tensor(tx))).numpy()
        selected=p[:,0]>=.9;t=truth[:,4]/10
        result['training_heads'].append(dict(fold=fold,kind=kind,blocks=len(tx),selected=int(selected.sum()),selected_negative_delta=int((selected&(p[:,4]<0)).sum()),delta_mae=float(np.abs(p[:,4]-t).mean()),zero_delta_mae=float(np.abs(t).mean())))
    g=run.previous.original.core.CandidateLocalSegmenter().eval();g.load_state_dict(torch.load(exp/'generators'/f'outer{fold}'/'full'/'model.pt',weights_only=True,map_location='cpu'))
    for s,cid in cases:
        e=run.examples([s])[s,cid]
        with np.load(exp/'generators'/f'outer{fold}'/'full'/'proposals'/s/'data.npz') as z,np.load(exp/'predictions'/s/'predictions.npz') as pred:
            with torch.inference_mode():logits,_,_=run.forward(g,[e],device='cpu');p=logits.softmax(1).numpy()
            delta=float(np.abs(p.astype(np.float16).astype(float)-z[cid+'_G_probability'].astype(float)).max());assert delta<.002
            result['replays'].append(dict(scene=s,candidate=cid,model='G',max_error=delta))
            for source in ('B','G'):
                x=z[cid+'_'+source+'_regions'].astype(np.float32)
                if not len(x):continue
                for kind,m in models.items():
                    with torch.inference_mode():p=run.core.decoded(m(torch.tensor(x))).numpy().astype(np.float16).astype(np.float32)
                    q=pred[cid+'_'+source+'_'+kind+'_block_prediction'].astype(np.float32);delta=float(np.abs(p-q).max());assert delta<.002
                    assert np.array_equal(p[:,0]>=.9,q[:,0]>=.9)
                    result['replays'].append(dict(scene=s,candidate=cid,model=source+'_'+kind,blocks=len(x),max_error=delta,selection_equal=True))
(out/'model_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
