from pathlib import Path
import json,hashlib,os,shutil
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
checks=[]
def check(ok,label):
    assert ok,label
    checks.append(label)
routes=json.loads((ROOT/'full_bundle/metadata/scene_routes.json').read_text())
fields={'data':'prepared_root','records_resolved':'native_records','actual_front':'gd10_frontend','actual_backend':'backend','actual_own_root':'own','yolosam':'yolosam','bound_geometry':'geometry.json','bound_points':'points.npz','own_evidence':'evidence.jsonl'}
for r in routes:
    for field,name in fields.items():
        p=ROOT/'full_bundle/artifacts/scannet'/r['group']/r['scene']/name
        check(p.exists(),str(p))
        check(p.resolve()==Path(r[field]).resolve(),'route identity '+str(p))
check(len(routes)==964,'964 scene routes')
frames=json.loads((ROOT/'full_bundle/metadata/verification_frames_index.json').read_text())
for p in frames.values():check(Path(p).exists(),'frame contract '+str(p))
check(len(frames)==972,'972 frame-index entries')
receipt=json.loads((OUT/'local_deletion_executed.json').read_text());names={r['name'] for r in receipt['deleted_roots']}
for r in receipt['deleted_roots']:check(not (ROOT/r['path']).exists(),'retired '+r['path'])
for base in ['full_bundle','sl_bundle','research/geometry']:
    for folder,dirs,files in os.walk(ROOT/base,followlinks=False):
        dirs[:]=[n for n in dirs if n not in ['__pycache__','.git','runtime','vendor','code_sources']]
        for n in dirs+files:
            p=Path(folder)/n
            if p.is_symlink():
                dest=p.resolve()
                if dest.is_relative_to(ROOT/'.local'):
                    check(dest.relative_to(ROOT/'.local').parts[0] not in names,'no retained alias points into deleted root '+str(p))
        dirs[:]=[n for n in dirs if not (Path(folder)/n).is_symlink()]
check(hashlib.sha256((ROOT/receipt['archive']).read_bytes()).hexdigest()==receipt['archive_sha256'],'history archive hash')
free=shutil.disk_usage(ROOT).free
result=dict(complete=True,checks=len(checks),scene_routes=len(routes),frame_contracts=len(frames),retired_roots_absent=len(names),free_bytes=free,free_GiB=free/2**30,development_GiB_with_50GiB_reserve=free/2**30-50,turn_start_free_bytes=106683731968,turn_net_free_change_GiB=(free-106683731968)/2**30,previous_two_rounds_net_GiB=47.783186)
(OUT/'assets_verified.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(result)
