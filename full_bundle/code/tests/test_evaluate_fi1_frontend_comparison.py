import tempfile
import unittest

from tools.evaluate_fi1_frontend_comparison import native_only_geometry, raw_frame_observations
from tools.evaluate_fi1_frontend_comparison import validate_full_evidence_identity, digest
from tools.evaluate_fi1_frontend_comparison import bind_gvc_track_frame_counts
from tools.evaluate_fi1_frontend_comparison import execution_groups


class WorkflowBatchTests(unittest.TestCase):
    def test_subset_preserves_frozen_group_order(self):
        selection={'scenes':[f's{i:02d}' for i in range(48)]}
        self.assertEqual(execution_groups(selection,[3,1]),[(1,selection['scenes'][:8]),(3,selection['scenes'][16:24])])

    def test_all_batches_cover_each_scan_once(self):
        selection={'scenes':list(range(48))}
        self.assertEqual([s for _,ss in execution_groups(selection,list(range(1,7))) for s in ss],selection['scenes'])

    def test_invalid_groups_rejected(self):
        for batches in [[],[1,1],[0],[7]]:
            with self.assertRaises(ValueError):execution_groups({'scenes':list(range(48))},batches)


class GVCFrameCountTests(unittest.TestCase):
    def inputs(self):
        tracks=[{'track_id':4,'frame_ids':['0','5','10','15']}]
        rows=[{'candidate_source':'d2b_track','candidate_id':4,'source_frame_count':2,
               'unresolved_source_frame_id_count':2,'gvc_source_frame_excluded':{'gvc':{'mean':.7}}}]
        return rows,tracks,{'0':0,'10':1}

    def test_dense_support_count_uses_all_actual_frames_without_changing_gvc(self):
        rows,tracks,pool=self.inputs();result=bind_gvc_track_frame_counts(rows,tracks,pool)
        self.assertEqual(result[0]['source_frame_count'],4)
        self.assertEqual(result[0]['source_frames_in_gvc_pool_count'],2)
        self.assertEqual(result[0]['source_frames_outside_gvc_pool_count'],2)
        self.assertEqual(result[0]['gvc_source_frame_excluded'],rows[0]['gvc_source_frame_excluded'])
        self.assertEqual(rows[0]['source_frame_count'],2)

    def test_legacy_sparse_support_value_is_unchanged(self):
        rows,tracks,pool=self.inputs();tracks[0]['frame_ids']=['0','10'];rows[0]['unresolved_source_frame_id_count']=0
        self.assertEqual(bind_gvc_track_frame_counts(rows,tracks,pool)[0]['source_frame_count'],2)

    def test_incorrect_pool_count_cannot_be_silently_replaced(self):
        rows,tracks,pool=self.inputs();rows[0]['source_frame_count']=3
        with self.assertRaisesRegex(ValueError,'intersection disagrees'):bind_gvc_track_frame_counts(rows,tracks,pool)

    def test_missing_track_rows_or_duplicate_source_frames_rejected(self):
        rows,tracks,pool=self.inputs()
        with self.assertRaisesRegex(ValueError,'coverage incomplete'):bind_gvc_track_frame_counts([],tracks,pool)
        tracks[0]['frame_ids'].append('0')
        with self.assertRaisesRegex(ValueError,'duplicate raw'):bind_gvc_track_frame_counts(rows,tracks,pool)


class FullEvidenceIdentityTests(unittest.TestCase):
    def inputs(self):
        binding={'scenes':['scene0153_00'],'source':'val48_C'}
        view={'binding_identity_sha256':digest(binding),'class_names':['a','b'],
              'protocol':{'views':40},'config_sha256':'config'}
        evidence={'cache':'frozen_cache','cache_inputs':{'weight':'original'},
                  'views_identity_sha256':digest(view)}
        return binding,evidence,view,dict(evidence),dict(view)

    def test_exact_frozen_identity_accepted(self):
        validate_full_evidence_identity(*self.inputs())

    def test_other_binding_rejected_even_with_identical_shapes(self):
        b,e,v,re,rv=self.inputs();b['source']='val48_A'
        with self.assertRaisesRegex(ValueError,'another geometry'):validate_full_evidence_identity(b,e,v,re,rv)

    def test_changed_weights_rejected(self):
        b,e,v,re,rv=self.inputs();e['cache_inputs']={'weight':'new'}
        with self.assertRaisesRegex(ValueError,'model/cache'):validate_full_evidence_identity(b,e,v,re,rv)

    def test_reordered_classes_and_altered_view_policy_rejected(self):
        for field,value in [('class_names',['b','a']),('protocol',{'views':30}),('config_sha256','other')]:
            b,e,v,re,rv=self.inputs();v[field]=value;e['views_identity_sha256']=digest(v)
            with self.assertRaisesRegex(ValueError,'vocabulary/view'):validate_full_evidence_identity(b,e,v,re,rv)

    def test_stale_view_receipt_rejected(self):
        b,e,v,re,rv=self.inputs();e['views_identity_sha256']='stale'
        with self.assertRaisesRegex(ValueError,'view identity changed'):validate_full_evidence_identity(b,e,v,re,rv)


class RawFrameAdapterTests(unittest.TestCase):
    def inputs(self):
        manifest = {'frames': [{'frame_index': i, 'frame_id':str(raw), 'pose_valid':True}
                               for i,raw in enumerate([0,5,10,20])],
                    'arms':{'A':[0,20], 'B':[0,5,10,20], 'C':[0,20], 'D':[0,5,10,20]}}
        observations = [{'observation_id':7,'frame_index':1,'frame_id':'5','mask_rle':{'counts':[3]},'point_indices_path':'actual.npz'}]
        tracks = [{'observation_ids':[7],'frame_ids':['5']}]
        return manifest,observations,tracks

    def test_stride5_raw5_is_not_frequency10_frame10(self):
        m,o,t=self.inputs()
        result=raw_frame_observations(m,o,t,'B')
        self.assertEqual(result[0],{**o[0],'frame_index':5})
        self.assertEqual(o[0]['frame_index'],1)

    def test_sparse_manifest_index_is_not_compacted(self):
        m,o,t=self.inputs();o[0].update(frame_index=3,frame_id='20');t[0]['frame_ids']=['20']
        self.assertEqual(raw_frame_observations(m,o,t,'C')[0]['frame_index'],20)

    def test_invalid_pose_observation_rejected(self):
        m,o,t=self.inputs();m['frames'][1]['pose_valid']=False
        with self.assertRaisesRegex(ValueError,'invalid pose'):raw_frame_observations(m,o,t,'D')
        self.assertEqual(raw_frame_observations(m,[],[],'D'),[])

    def test_raw_identity_mismatch_rejected(self):
        m,o,t=self.inputs();o[0]['frame_id']='10'
        with self.assertRaisesRegex(ValueError,'disagree'):raw_frame_observations(m,o,t,'B')

    def test_dense_observation_not_allowed_in_sparse_arm(self):
        m,o,t=self.inputs()
        with self.assertRaisesRegex(ValueError,'outside arm'):raw_frame_observations(m,o,t,'A')

    def test_parent_identity_cannot_fall_back_to_another_track(self):
        m,o,t=self.inputs();t[0]['observation_ids']=[8]
        with self.assertRaisesRegex(ValueError,'missing parent'):raw_frame_observations(m,o,t,'B')
        t[0]['observation_ids']=[7];t[0]['frame_ids']=['10']
        with self.assertRaisesRegex(ValueError,'actual parents'):raw_frame_observations(m,o,t,'B')

    def test_duplicate_observation_and_manifest_rejected(self):
        m,o,t=self.inputs()
        with self.assertRaisesRegex(ValueError,'duplicate observation'):raw_frame_observations(m,o+o,t,'B')
        m['frames'].append(m['frames'][1])
        with self.assertRaisesRegex(ValueError,'duplicate manifest'):raw_frame_observations(m,o,t,'B')


class FullControlTests(unittest.TestCase):
    def test_native_control_removes_nonnative_aliases_without_mutating_source(self):
        native={'candidate_source':'native','candidate_id':0}
        track={'candidate_source':'track','candidate_id':1}
        geometry=[{'geometry_key':'mixed','legacy_members':[native,track],'dv3_plans':[{'append_only':True}]},
                  {'geometry_key':'track','legacy_members':[track],'dv3_plans':[]}]
        selected=native_only_geometry(geometry)
        self.assertEqual(len(selected),1)
        self.assertEqual(selected[0]['legacy_members'],[native])
        self.assertEqual(selected[0]['dv3_plans'],[])
        self.assertEqual(len(geometry[0]['legacy_members']),2)
        self.assertEqual(len(geometry[0]['dv3_plans']),1)
from pathlib import Path

import numpy as np

from tools.evaluate_fi1_frontend_comparison import (
    normalize_votes, top_predictions, overlap_matrix, coverage_metrics, checked_points,
)


class FrontendEvaluationTests(unittest.TestCase):
    def test_multiple_classes_survive_and_unknown_is_background(self):
        votes = np.zeros((2, 198), dtype=np.int64)
        votes[0, [3, 9]] = [200, 100]
        scores = normalize_votes(votes)
        gi, classes, values = top_predictions(scores, [0, 1], 3)
        self.assertEqual(list(zip(gi, classes)), [(0, 3), (1, 198), (0, 9)])
        np.testing.assert_array_equal(values, [1, 1, .5])

    def test_exact_duplicate_geometry_cannot_consume_budget_twice(self):
        votes = np.zeros((2, 198))
        votes[:, 0] = 1
        gi, classes, _ = top_predictions(normalize_votes(votes), [0, 0, 1, 1], 2)
        self.assertEqual(gi.tolist(), [0, 1])
        self.assertEqual(classes.tolist(), [0, 0])

    def test_no_independent_gt_quality_gain_from_duplicate(self):
        native = np.array([[.8, 0.]])
        track = np.array([[.9, 0.], [.9, 0.]])
        metrics = coverage_metrics(native, track)['0.5']
        self.assertEqual(metrics['new_covered'], 0)
        self.assertEqual(metrics['track_candidates_matching_gt'], 2)
        self.assertEqual(metrics['excess_gt_supports'], 1)

    def test_strict_threshold_and_independent_new_instance(self):
        metrics = coverage_metrics(np.array([[.5, 0.]]), np.array([[.5, .75]]))
        self.assertEqual(metrics['0.5']['combined_covered'], 1)
        self.assertEqual(metrics['0.5']['new_covered'], 1)
        self.assertEqual(metrics['0.75']['combined_covered'], 0)

    def test_overlap_counts_do_not_overflow_and_include_background_in_union(self):
        gt = np.r_[np.full(400, 1001), np.zeros(100), np.full(300, 2001)]
        iou, precision, recall = overlap_matrix([np.arange(500)], gt, [1001, 2001])
        np.testing.assert_allclose(iou, [[.8, 0]])
        np.testing.assert_allclose(precision, [[.8, 0]])
        np.testing.assert_allclose(recall, [[1., 0]])

    def test_bad_geometry_rejected_instead_of_clipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'points.npz'
            for points in ([0, 10], [0, 0], [-1, 3]):
                np.savez(p, point_indices=np.array(points))
                with self.assertRaises(ValueError):
                    checked_points(p, 10)

    def test_empty_tracks_and_gt_are_defined(self):
        metrics = coverage_metrics(np.empty((1, 0)), np.empty((0, 0)))
        self.assertEqual(metrics['gt_count'], 0)
        self.assertEqual(metrics['0.5']['new_covered'], 0)


if __name__ == '__main__':
    unittest.main()
