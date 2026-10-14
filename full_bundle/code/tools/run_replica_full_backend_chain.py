#!/usr/bin/env python3
"""Resume sealed Replica stages, retaining failures and all original contracts."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT))
from tools.run_replica_registered_full import OUT,SCENES
from tools.fi1_frontend_protocol import write_json as write,sha256_file as sha
legacy=json.loads((ROOT/'.local/fi1_r0_r4_preparation_20260913/dv3_generation_v2/jobs.json').read_text())['environment']
env={**os.environ,**legacy['environment']}; env['PYTHONPATH']=str(ROOT)+':'+':'.join(env['PYTHONPATH'].split(':')[1:]); env['PYTHONPATH']+=':'+str(ROOT/'models/YOLO-World')+':'+str(ROOT/'models/Mask3D')
new={**os.environ,'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
def run(stage,cmd,environ):
 logfile=OUT/(stage+'.log')
 write(OUT/'execution.json',{'stage':stage,'argv':cmd,'running':True,'time':datetime.datetime.now(datetime.timezone.utc).isoformat()})
 with logfile.open('a') as f:
  p=subprocess.run(cmd,env=environ,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
 if p.returncode:
  write(OUT/'execution.json',{'stage':stage,'running':False,'failed':True,'exit_code':p.returncode,'log':str(logfile)}); raise SystemExit(p.returncode)
for s in SCENES:
 run('backend_'+s,[legacy['python'],'tools/fi1_replica_full_adapter.py','backend','--scenes',s],env)
for s in SCENES:
 run('evidence_'+s,[str(ROOT/'.venvs/yoloe_gpu/bin/python'),'tools/fi1_replica_member_scoring.py','evidence','--scenes',s],new)
run('score',[str(ROOT/'.venvs/yoloe_gpu/bin/python'),'tools/fi1_replica_member_scoring.py','score'],new)
run('evaluation',[str(ROOT/'.venvs/yoloe_gpu/bin/python'),'tools/fi1_replica_member_scoring.py','evaluate'],{**new,'CUDA_VISIBLE_DEVICES':''})
write(OUT/'execution.json',{'running':False,'complete':True,'results':str(OUT/'evaluation/results.json')})
