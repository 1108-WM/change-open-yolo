from pathlib import Path
import os,subprocess,json
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
info=json.loads((ROOT/'.local/fi1_r0_r4_preparation_20260913/dv3_generation_v2/jobs.json').read_text())['environment']
env=os.environ.copy();env.update(info['environment']);env['PYTHONPATH']=str(ROOT)+':'+str(ROOT/'tools')+':'+':'.join(info['environment']['PYTHONPATH'].split(':')[1:])
code="""
import json,pickle,sklearn
from pathlib import Path
from tools.evaluate_fi1_frontend_comparison import full_backend_modules
from tools.build_fi1_d_v3_frozen_inference_plan import _load_models
mods=full_backend_modules()
models=_load_models(Path('full_bundle/models/dv3_official100'))
root=Path('full_bundle/models/champion')
with (root/'model_package.pkl').open('rb') as f:champion=pickle.load(f)
assert len(mods)==9 and len(models)==7
print(json.dumps(dict(complete=True,backend_modules=len(mods),dv3_models=len(models),champion_loaded=True,sklearn=sklearn.__version__)))
"""
p=subprocess.run([info['python'],'-c',code],cwd=ROOT,env=env,capture_output=True,text=True)
(OUT/'models_verify.log').write_text(p.stdout+p.stderr)
print(p.stdout);print(p.stderr[-2000:]);assert p.returncode==0
(OUT/'models_verified.json').write_text(p.stdout)
