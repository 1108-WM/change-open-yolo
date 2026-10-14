#!/usr/bin/env python3
"""Training-only fit diagnostics, separating segmentation from candidate gate."""
from pathlib import Path
from collections import Counter
import sys
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools import run_fi1_local_edit_ablation as run
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha


def main():
    c=run.read(run.OUT/'contract.json');guard=run.Guard(c,'pilot',0);sys.addaudithook(guard.hook);torch.set_num_threads(2);rows=[]
    for fold in range(5):
        guard.fold=fold;fit=run.read(run.OUT/'pilot/C'/f'fold{fold}'/'fit.json');keys=[tuple(k) for k in fit['keys']];data=run.examples(sorted({s for s,cid in keys}));data={key:data[key] for key in keys}
        model=run.learn.model('C').eval();model.load_state_dict(torch.load(run.OUT/'pilot/C'/f'fold{fold}'/'model.pt',map_location='cpu',weights_only=True));total=Counter();by_scope={}
        for key,e in data.items():
            with torch.inference_mode():x,parts,logit,can,rel=run.forward(model,'C',[e],device='cpu');p=logit.softmax(1).numpy().astype(np.float16);cp=can.softmax(1)[0].numpy().astype(np.float16)
            before,_=run.original.core.repair(e['parent'],e['halo'],p,e['visible']);after,_=run.learn.repair(e['parent'],e['halo'],p,e['visible'],'C',cp)
            own=np.isin(e['halo'],e['parent']);seen=e['visible']>=2;truth=e['y'];local=Counter(delete_opportunity=int((own&seen&(truth==1)).sum()),add_opportunity=int((~own&seen&(truth==0)).sum()))
            for name,mask in [('ungated',before),('gated',after)]:
                keep=np.isin(e['halo'],mask);deleted=own&~keep;added=~own&keep
                local.update({name+'_correct_delete':int((deleted&(truth==1)).sum()),name+'_wrong_delete':int((deleted&(truth==0)).sum()),name+'_correct_add':int((added&(truth==0)).sum()),name+'_wrong_add':int((added&(truth==1)).sum())})
            total.update(local);by_scope.setdefault(e['meta']['scope_name'],Counter()).update(local)
        rows.append(dict(fold=fold,training_only=True,counts=dict(total),by_scope={k:dict(v) for k,v in by_scope.items()}))
    freeze_json(run.OUT/'pilot_segmentation_vs_gate.json',dict(complete=True,training_only=True,not_new_predictions=True,not_used_for_parameter_changes=True,rows=rows,code_sha256=sha(Path(__file__))))
    print('training segmentation/gate diagnostic complete',flush=True)

if __name__=='__main__':main()
