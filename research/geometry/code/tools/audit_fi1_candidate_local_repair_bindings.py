#!/usr/bin/env python3
"""GT-free selected RGB-D, pose, original bank and downstream ledger audit."""
from pathlib import Path
from collections import defaultdict,Counter
import sys,json
ROOT=Path(__file__).resolve().parents[4];sys.path.insert(0,str(ROOT))
import numpy as np
from tools import run_fi1_candidate_local_repair as run
from tools.fi1_frontend_protocol import freeze_json,sha256_file as sha,validate_observation


def audit():
    out=run.OUT;c=run.read(out/'contract.json');sys.addaudithook(run.Guard(c,'features').hook)
    fixed=run.read(run.audit.OUT/'contract.json');counts=Counter();bindings={}
    for s in c['scenes']:
        job=c['jobs'][s];m=run.read(job['paths']['manifest']);fm={int(r['frame_id']):r for r in m['frames']}
        downstream=[json.loads(l) for l in Path(job['paths']['observations']).read_text().splitlines()];by=defaultdict(list)
        for o in downstream:by[int(o['frame_id'])].append(o)
        bankroot=Path(downstream[0]['point_indices_path']).parent.parent;plan=run.read(out/'plan'/s/'candidates.json');receipt=run.read(out/'plan'/s/'receipt.json')
        needed=sorted({f for r in plan for a in run.core.ARMS for f in r['views'][a]['frames']})
        with np.load(out/'plan'/s/'evidence.npz') as evidence:
            frameix={int(f):i for i,f in enumerate(evidence['frames'])}
            for f in needed:
                frame=fm[f];bank=bankroot/f'frame{f:06d}';bp=str(bank/'receipt.json');expected=fixed['frame_inputs'][bp] if s in c['fixed10'] else receipt['source_banks'][bp]
                assert sha(bank/'receipt.json')==expected;bindings[bp]=expected
                rows,pts,br=run.bank_points(bank,job['binding']['point_count']);run.audit.verify_ledger(rows,by[f]);counts['raw_bank_rows']+=len(rows);counts['downstream_rows']+=len(by[f]);counts['distinct_scene_frames']+=1
                for row in rows:validate_observation(row,frame,job['binding']['point_count'])
                for key in ('color','depth','pose'):
                    info=frame[key];assert sha(info['path'])==info['sha256'];bindings[info['path']]=info['sha256']
                pose=np.loadtxt(frame['pose']['path']);assert np.array_equal(pose,evidence['poses'][frameix[f]])
                assert sha(m['files']['intrinsics']['path'])==m['files']['intrinsics']['sha256']
        print('bank bindings verified',s,len(needed),flush=True)
    freeze_json(out/'observation_bindings_verified.json',dict(complete=True,GT_read=False,counts=dict(counts),bindings=bindings,verifier_code_sha256=sha(Path(__file__)),plan_seal_sha256=sha(out/'plan_frozen.json')))

if __name__=='__main__':audit()
