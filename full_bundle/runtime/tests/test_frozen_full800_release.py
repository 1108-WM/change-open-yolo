"""Release portability, evidence integrity, and prediction-before-GT gates."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tools import run_frozen_full800 as runtime
from tools.fi1_g10_closure_selection import fixed_selection


def fixture(tmp_path):
    scene='scene0354_00'
    geometry=tmp_path/'geometry'
    geometry.mkdir()
    points=np.arange(120,dtype=np.int64)
    digest=hashlib.sha256(points.tobytes()).hexdigest()
    key=f'{scene}:geometry:{digest}'
    row={'geometry_key':key,'points_sha256':digest,'point_count':120,'array_key':'g00000',
         'legacy_members':[{'candidate_source':'track','candidate_id':0,'frozen_class_index':1}],
         'dv3_plans':[]}
    (geometry/'geometry.json').write_text(json.dumps([row]))
    np.savez(geometry/'points.npz',g00000=points)
    evidence={'geometry_key':key,'points_sha256':digest,
        'yolo':{'acquisition_complete':True,'available':True,'votes':[1,3]+[0]*196},
        'alpha':{'acquisition_complete':True,'available':False,'missing_reason':'no_valid_sam_view'}}
    ep=tmp_path/'evidence.jsonl'
    ep.write_text(json.dumps(evidence)+'\n')
    args=NS(geometry_dir=geometry,evidence_file=ep,point_count=120,scenes=[scene],output=tmp_path/'predictions'/scene,
            semantic_provenance=runtime.ASSETS/'alpha_text/model_provenance.json')
    return args,points,row,evidence


def test_repository_local_canonical_selection(tmp_path):
    expected=runtime.prepare_selection(tmp_path)
    assert len(expected['scenes'])==312
    assert fixed_selection(tmp_path/'selection')==expected


def test_complete_evidence_required(tmp_path):
    args,_,_,evidence=fixture(tmp_path)
    evidence['alpha']['acquisition_complete']=False
    args.evidence_file.write_text(json.dumps(evidence)+'\n')
    with pytest.raises(ValueError,match='not acquired|acquisition|incomplete'):
        runtime.score(args)
    assert not args.output.exists()


def test_changed_pointset_rejected_before_model(tmp_path):
    args,points,_,_=fixture(tmp_path)
    np.savez(args.geometry_dir/'points.npz',g00000=points+1)
    args.point_count=121
    with pytest.raises(ValueError,match='pointset digest'):
        runtime.score(args)
    assert not args.output.exists()


def test_geometry_from_another_scene_rejected(tmp_path):
    args,_,_,_=fixture(tmp_path)
    args.scenes=['scene0549_00']
    with pytest.raises(ValueError,match='another scene'):
        runtime.score(args)


def test_different_Alpha_weight_provenance_rejected(tmp_path):
    args,_,_,_=fixture(tmp_path)
    wrong=runtime.read(args.semantic_provenance)
    wrong['alpha_clip_checkpoint_sha256']='0'*64
    p=tmp_path/'different_model.json'
    p.write_text(json.dumps(wrong))
    args.semantic_provenance=p
    with pytest.raises(ValueError,match='provenance differs'):
        runtime.score(args)


def test_short_pool_is_not_padded_and_sealed_before_GT(tmp_path):
    args,_,_,_=fixture(tmp_path)
    runtime.score(args)
    selected=runtime.read(args.output/'selected.json')
    assert len(selected)==2
    assert {m['class_index'] for m in selected}=={0,1}
    runtime.seal(NS(predictions_root=args.output.parent,scenes=args.scenes))
    sealed=runtime.read(args.output.parent.parent/'predictions_frozen.json')
    assert sealed['complete'] and not sealed['full_validation312']
    assert not sealed['GT_read']
    with pytest.raises(ValueError,match='allow-gt'):
        runtime.evaluate(NS(predictions_root=args.output.parent,allow_gt_evaluation=False))


def test_resume_checks_bytes_and_evaluation_checks_all_predictions(tmp_path,monkeypatch):
    args,_,_,_=fixture(tmp_path)
    runtime.score(args)
    runtime.score(args)
    runtime.seal(NS(predictions_root=args.output.parent,scenes=args.scenes))
    with (args.output/'selected.json').open('a') as handle:
        handle.write(' ')
    with pytest.raises(ValueError,match='Frozen prediction changed'):
        runtime.score(args)
    import tools.fi1_r3b_delta_diagnostic_core as core
    monkeypatch.setattr(core,'evaluator',lambda:pytest.fail('Evaluator/GT must not be opened before all seals pass'))
    with pytest.raises(ValueError,match='Prediction bytes changed'):
        runtime.evaluate(NS(predictions_root=args.output.parent,allow_gt_evaluation=True,
                            gt_root=tmp_path/'absent_GT',output=tmp_path/'evaluation'))


def test_final_test_and_noncanonical_scenes_rejected():
    with pytest.raises(ValueError,match='validation312'):
        runtime.selected_scenes(NS(scenes=['scene0000_00']))


def test_frozen_models_and_26_feature_metadata():
    import joblib
    manifest=runtime.verify_release()
    assert manifest['model_training']['scene_budget']==800
    model=joblib.load(runtime.ASSETS/'b26/B.joblib')
    assert model.n_features_in_==26
    assert runtime.read(runtime.ASSETS/'dv3/summary.json')['training_dataset']=='official100'


def test_frontend_callback_kernels_match_published_GD10_snapshot():
    # These modules are imported as "from tools import ...", so a source
    # collector must include them even without a dotted ImportFrom.module.
    reference={
        'build_details_frame_siou_tracks.py':'ea05faff778b4e9292bcb73ff0d88000112e75262e37ab56d0b85a39e8285205',
        'refine_details_automatic_tracks_consensus.py':'e72faad27022a80c08f86ff122bfe29cc194e3725d84318e6f32e69ae6ca6ba8',
        'build_details_iterative_proposal_merges.py':'6e257020422d980567decf0e001c532fa7987a637b1c9552d35efbf15a3f4aae',
    }
    manifest=runtime.read(runtime.ROOT/'reproduction/manifest.json')
    for name,digest in reference.items():
        path='tools/'+name
        assert runtime.sha(runtime.ROOT/path)==digest
        assert manifest['files'][path]==digest
