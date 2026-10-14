#!/usr/bin/env python3
"""Additional GT-free CPU model/projection/visibility checks with separate receipt."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from tools import run_fi1_candidate_local_repair as run
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha


def verify():
    out=run.OUT;c=run.read(out/'contract.json');sys.addaudithook(run.Guard(c,'predict').hook);torch.set_num_threads(2)
    checked=[];binding_count=0;view_sets=0
    for s in c['scenes']:
        plans=run.read(out/'plan'/s/'candidates.json')
        with np.load(out/'plan'/s/'evidence.npz') as evidence,np.load(out/'features'/s/'data.npz') as features:
            frameix={int(f):i for i,f in enumerate(evidence['frames'])};visible=run.audit.unpack(evidence,'reliable');roi=evidence['roi']
            for r in plans:
                cid=r['candidate_id'];halo=features[cid+'_halo'];ix=np.searchsorted(roi,halo);assert np.array_equal(roi[ix],halo)
                for a in run.core.ARMS:
                    ids=r['views'][a]['frames'];indices=[frameix[f] for f in ids]
                    assert len(ids)<=6 and len(set(evidence['groups'][indices]))==len(ids)
                    assert np.asarray(evidence['has_masks'][indices]).all()
                    assert np.array_equal(visible[indices][:,ix].sum(0).astype(np.uint8),features[cid+'_'+a+'_groups'])
                    assert np.array_equal(features[cid+'_'+a+'_x'][:,9]>.5,np.isin(halo,features[cid+'_parent']))
                    assert np.isfinite(features[cid+'_'+a+'_x']).all();binding_count+=len(ids);view_sets+=1
    for fold in range(5):
        model=run.core.CandidateLocalSegmenter().eval();model.load_state_dict(torch.load(out/'models'/f'fold{fold}'/'model.pt',map_location='cpu',weights_only=True))
        s=next(s for s in c['scenes'] if c['folds'][s]==fold)
        plans=run.read(out/'plan'/s/'candidates.json')
        with np.load(out/'features'/s/'data.npz') as z,np.load(out/'predictions'/s/'predictions.npz') as prediction:
            for r in plans[:2]:
                cid=r['candidate_id']
                for a in run.core.ARMS:
                    e=dict(x=z[cid+'_'+a+'_x'].astype(np.float32),neighbors=z[cid+'_neighbors'].astype(np.int64),weights=z[cid+'_neighbor_weights'].astype(np.float32))
                    x,nn,w,parts=run.collate([e],device='cpu')
                    with torch.inference_mode():p=model(x,nn,w,parts).softmax(1).numpy().astype(np.float16)
                    saved=prediction[cid+'_'+a+'_probability'];error=float(np.abs(p.astype(np.float32)-saved.astype(np.float32)).max());assert error<=.001
                    q,_=run.core.repair(z[cid+'_parent'],z[cid+'_halo'],p,z[cid+'_'+a+'_groups']);same=np.array_equal(q,prediction[cid+'_'+a+'_points'])
                    checked.append(dict(scene=s,fold=fold,candidate_id=cid,arm=a,max_probability_error=error,identical_action=same))
    assert all(r['identical_action'] for r in checked)
    freeze_json(out/'model_replay_verified.json',dict(complete=True,GT_read=False,CPU_OOF_model_replays=checked,view_sets=view_sets,selected_frame_bindings=binding_count,all_visibility_and_parent_features_replayed=True,verifier_code_sha256=sha(Path(__file__)),contract_sha256=sha(out/'contract.json')))
    print('model/view evidence replay verified',len(checked),view_sets,binding_count,flush=True)

if __name__=='__main__':verify()
