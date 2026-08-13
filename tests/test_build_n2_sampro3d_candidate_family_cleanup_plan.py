import importlib.util
from pathlib import Path
def _module():
 p=Path(__file__).parents[1]/"tools"/"build_n2_sampro3d_candidate_family_cleanup_plan.py";s=importlib.util.spec_from_file_location("n2plan",p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def test_only_empty_or_exact_geometry_get_fallback_plan():
 m=_module();base={"scene_name":"s","reliable_core_superpoint_count":0,"unknown_boundary_superpoint_count":1,"best_d2b_track_id":None,"family_contained_by_d2b":False,"d2b_contained_by_family":False,"exact_mutual_duplicate":False}
 rows=[{**base,"candidate_family_key":"empty","seed_superpoint_id":1,"family_union_superpoint_ids":[],"family_union_superpoint_count":0,"exact_duplicate_fingerprint":""},{**base,"candidate_family_key":"a","seed_superpoint_id":2,"family_union_superpoint_ids":[7],"family_union_superpoint_count":1,"exact_duplicate_fingerprint":"7"},{**base,"candidate_family_key":"b","seed_superpoint_id":3,"family_union_superpoint_ids":[7],"family_union_superpoint_count":1,"exact_duplicate_fingerprint":"7"},{**base,"candidate_family_key":"overlap","seed_superpoint_id":4,"family_union_superpoint_ids":[8],"family_union_superpoint_count":1,"exact_duplicate_fingerprint":"8"}]
 plan={x["candidate_family_key"]:x for x in m.cleanup_plan(rows)}
 assert plan["empty"]["plan_state"]=="empty_family_no_candidate" and plan["a"]["canonical_family_key"]=="a" and plan["b"]["plan_state"]=="exact_geometry_duplicate_fallback" and plan["overlap"]["plan_state"]=="hold_for_later_competition"
