import copy
import unittest
import numpy as np
from tools.audit_fi1_r2_training import check_fit_roles,sparse_targets


class AuditTrainingTest(unittest.TestCase):
    def fixture(self):
        p={'roles':{'train':['scene0001_00','scene0002_00'],
            'development_validation':['scene0003_00'],'diagnostic':['scene0001_01']},
            'training_heldout_folds':[['scene0001_00'],['scene0002_00']]}
        rows=[{'model':'fold0','fit_scenes':['scene0002_00'],'prediction_scenes':['scene0001_00']},
            {'model':'fold1','fit_scenes':['scene0001_00'],'prediction_scenes':['scene0002_00']},
            {'model':'full200','fit_scenes':p['roles']['train'],'prediction_scenes':['scene0003_00','scene0001_01']}]
        return p,rows

    def test_valid_grouped_assignment(self):
        p,rows=self.fixture();self.assertEqual(len(check_fit_roles(p,rows)),3)

    def test_diagnostic_training_leak_rejected(self):
        p,rows=self.fixture();rows[-1]['fit_scenes']=rows[-1]['fit_scenes']+['scene0001_01']
        with self.assertRaisesRegex(ValueError,'assignment'):check_fit_roles(p,rows)

    def test_wrong_fold_model_assignment_rejected(self):
        p,rows=self.fixture();rows[0]['prediction_scenes']=rows[1]['prediction_scenes']
        with self.assertRaisesRegex(ValueError,'assignment'):check_fit_roles(p,rows)

    def test_duplicate_or_missing_model_rejected(self):
        p,rows=self.fixture()
        with self.assertRaisesRegex(ValueError,'coverage'):check_fit_roles(p,rows[:-1])
        rows.append(copy.deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError,'coverage'):check_fit_roles(p,rows)

    def test_sparse_full_targets_match_boolean_masks(self):
        rng=np.random.default_rng(7);gt=np.repeat([2001,2002,3001,3002,0],[100,150,99,201,250])
        points=[np.sort(rng.choice(len(gt),size=n,replace=False)) for n in [0,99,100,150,400]]
        actual,count=sparse_targets(points,gt,[2,3]);self.assertEqual(count,3)
        expected=np.zeros((len(points),2))
        for i,pts in enumerate(points):
            pred=np.zeros(len(gt),dtype=bool);pred[pts]=True
            if len(pts)<100:continue
            for j,c in enumerate([2,3]):
                for instance in np.unique(gt):
                    truth=gt==instance
                    if instance//1000==c and truth.sum()>=100:
                        expected[i,j]=max(expected[i,j],(pred&truth).sum()/(pred|truth).sum())
        np.testing.assert_array_equal(actual,expected)


if __name__=='__main__':unittest.main()
