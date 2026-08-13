import importlib.util
from pathlib import Path
import numpy as np

def _module():
    p=Path(__file__).parents[1]/"tools"/"build_n1_sampro3d_seed_view_ledger.py"; s=importlib.util.spec_from_file_location("n1",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def test_only_unclaimed_superpoints_get_visible_seed_views():
    m=_module(); sp=np.array([1,1,2,2]); vis=np.array([[1,1,1,0],[0,1,1,1]],dtype=bool); proj=np.zeros((2,4,3)); proj[:,:,0]=np.arange(4); proj[:,:,1]=10
    rows=m.seed_view_rows(sp,{1},proj,vis,(1,1),2)
    assert len(rows)==1 and rows[0]["seed_superpoint_id"]==2 and rows[0]["seed_point_index"]==2
    assert [v["frame_index"] for v in rows[0]["views"]]==[1]
