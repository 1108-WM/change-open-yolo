#!/usr/bin/env python3
"""Independently reproject each corrected cache frame without any GT or fitting."""
from collections import Counter
import numpy as np
from PIL import Image
import run_fi1_relation_train200_dev55 as r


def main():
    r.guard();c=r.contract();a=r.a;checks=Counter();rows=[]
    for s in c['train_scenes']+c['scenes']:
        path=r.OUT/'measurements/predictions'/s/'cache_corrections.json'
        if not path.exists():continue
        report=a.read(path)
        if not report['corrections']:continue
        mf=a.read(c['jobs'][s]['job']['paths']['manifest']);frames={int(f['frame_id']):f for f in mf['frames']}
        data=np.load(mf['files']['processed']['path'],mmap_mode='r');intrinsic=np.loadtxt(mf['files']['intrinsics']['path'])
        for q in report['corrections']:
            f=frames[int(q['frame_id'])];pose=np.loadtxt(f['pose']['path']);w,h=Image.open(f['color']['path']).size;dw,dh=Image.open(f['depth']['path']).size
            k=intrinsic.copy()
            if (w,h)!=(dw,dh):
                resized=int(np.floor(dw*h/w));k[0,0]*=resized/h;k[1,1]*=dw/w;k[0,2]*=(dh-1)/(h-1);k[1,2]*=(dw-1)/(w-1)
            camera=np.column_stack((data[:,:3],np.ones(len(data))))@np.linalg.inv(pose).T
            projected=camera@k.T;positive=np.isfinite(projected).all(1)&(camera[:,2]>0)
            pixels=np.zeros((len(data),2),dtype=np.int64);pixels[positive]=np.trunc(projected[positive,:2]/projected[positive,2,None]).astype(np.int64)
            good=positive&(pixels[:,0]>=0)&(pixels[:,0]<dw)&(pixels[:,1]>=0)&(pixels[:,1]<dh)
            original=np.load(q['original']);replacement=np.load(q['replacement']);expected=original&good
            np.testing.assert_array_equal(replacement,expected);removed=np.flatnonzero(original&~expected)
            np.testing.assert_array_equal(removed,q['removed_point_ids']);np.testing.assert_allclose(camera[removed,2],q['camera_depth_m'],rtol=0,atol=1e-12)
            observed=[]
            for obs in a.read(r.Path(q['original']).parent/'observations.json'):
                pp=a.points(obs['point_indices_path']);hit=np.intersect1d(pp,removed)
                if len(hit):observed.append(dict(observation_id=obs['observation_id'],point_ids=hit.tolist()))
            assert observed==q['affected_observations']
            rows.append(dict(scene=s,frame_id=q['frame_id'],removed_points=len(removed),affected_observations=len(observed)))
            checks['independent_projection_frames']+=1;checks['removed_points']+=len(removed)
    a.seal(r.OUT/'cache_projection_verified.json',dict(complete=True,GT_used=False,model_fits=0,checks=dict(checks),frames=rows,
        code={str(r.Path(__file__).resolve()):a.sha(__file__)}))
    print('CACHE PROJECTION VERIFIED',dict(checks),flush=True)


if __name__=='__main__':main()
