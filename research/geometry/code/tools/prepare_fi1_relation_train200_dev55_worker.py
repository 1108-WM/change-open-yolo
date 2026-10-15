#!/usr/bin/env python3
"""Prepare disjoint scene partitions; primary process alone aggregates/finalizes."""
import argparse
import time
import traceback
import run_fi1_relation_train200_dev55 as r


def main(part):
    r.guard();c=r.contract();plan=r.a.read(r.OUT/'data_parallel_plan.json')
    r.verify(plan['code']);assert not (r.OUT/'models').exists()
    for i,s in enumerate(plan['partitions'][str(part)]):
        status=r.OUT/'data_status'/f'{s}.json';assert not status.exists()
        r.a.reserve();start=time.time()
        try:
            raw=r.raw_geometry(s,c);dest=r.features_scene(s,c,raw)
            corrections=r.a.read(r.OUT/'measurements/predictions'/s/'cache_corrections.json')
            outputs={str(p):r.a.sha(p) for p in raw.iterdir() if p.is_file()}
            outputs.update({str(p):r.a.sha(p) for p in (r.OUT/'measurements/predictions'/s).iterdir() if p.is_file()})
            outputs.update({str(p):r.a.sha(p) for p in dest.iterdir()})
            q=dict(complete=True,scene=s,seconds=time.time()-start,pairs=len(r.a.read(dest/'rows.json')),
                corrected_frames=len(corrections['corrections']),removed_visibility_points=sum(len(x['removed_point_ids']) for x in corrections['corrections']),outputs=outputs)
            r.a.seal(status,q)
            print('WORKER',part,i+1,len(plan['partitions'][str(part)]),s,q['pairs'],'corrections',q['corrected_frames'],q['removed_visibility_points'],'seconds',round(q['seconds'],2),flush=True)
        except Exception:
            r.a.seal(status,dict(complete=False,scene=s,seconds=time.time()-start,traceback=traceback.format_exc(),excluded=False,model_fits=0))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--part',type=int,choices=range(3),required=True);args=p.parse_args();main(args.part)
