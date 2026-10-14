import numpy as np
from tools.fi1_replica_member_scoring import semantic_functions,dedup_agnostic
from tools.fi1_r0_r1_protocol import build_pair
from tools.fi1_r2_core import geometry_features

def fixture(n):
 g={'geometry_key':'scene:geometry:abc','points_sha256':'abc','point_count':120,'legacy_members':[{'candidate_source':'native','candidate_id':0,'frozen_class_index':1},{'candidate_source':'track','candidate_id':0,'frozen_class_index':2}],'dv3_plans':[]}
 e={'geometry_key':g['geometry_key'],'points_sha256':'abc','yolo':{'acquisition_complete':True,'available':True,'votes':list(range(1,n+1))},'alpha':{'acquisition_complete':True,'available':True,'cosine':np.linspace(.1,.4,n).tolist(),'view_scale_std':np.full(n,.03).tolist()}}
 return g,e

def test_198_adapter_exact_original_parity():
 g,e=fixture(198); build,features=semantic_functions(198)
 assert build([g],{g['geometry_key']:e})==build_pair([g],{g['geometry_key']:e})
 np.testing.assert_array_equal(features(g,e),geometry_features(g,e))

def test_48_vocabulary_shapes_and_normalized_rank():
 g,e=fixture(48); build,features=semantic_functions(48)
 x=features(g,e); p=build([g],{g['geometry_key']:e},800)
 assert x.shape==(48,20) and len(p['R1']['eligible'])==48
 assert x[0,5]==1 and x[-1,5]==0
 assert np.isfinite(x).all()

def test_native_only_stays_original_production_eligibility():
 g,e=fixture(48); g['legacy_members']=g['legacy_members'][:1]
 build,_=semantic_functions(); p=build([g],{g['geometry_key']:e},800)
 assert [r['class_index'] for r in p['R1']['eligible']]==[1]

def test_actual_agnostic_max_only_selected_valid_members():
 masks=np.array([[1,1,0],[0,0,1]],dtype=bool)
 p={'pred_masks':masks,'pred_classes':np.array([1,2,48]),'pred_scores':np.array([.1,.8,.99])}
 d=dedup_agnostic(p)
 assert d['pred_masks'].shape==(2,1) and d['pred_scores'].tolist()==[.8]
 np.testing.assert_array_equal(d['pred_masks'][:,0],masks[:,0])
