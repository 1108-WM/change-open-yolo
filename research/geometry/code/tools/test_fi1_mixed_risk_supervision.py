import unittest
import numpy as np
import run_fi1_mixed_risk_supervision as r


class FixedProtocol(unittest.TestCase):
    def test_class_order(self):
        class Model:
            classes_ = np.array([2,0,1])
            def predict_proba(self, x): return np.array([[.04,.01,.95]])
        np.testing.assert_array_equal(r.probability_columns(Model(),None),[[.01,.95,.04]])
        self.assertEqual(r.CLASSES,dict(same=1,different=0,mixed=2,unknown=-1))

    def test_actions_and_scene_recipe(self):
        rows=[dict(pair_id='a',left=0,right=1),dict(pair_id='b',left=1,right=2),dict(pair_id='c',left=3,right=4)]
        pred=r.relation.decide(rows,[.9,.95,.1],[True]*3)
        self.assertEqual([q['state'] for q in pred],['unknown','accept','reject'])
        w=r.metrics.weights(['s','s','t'],[False]*3,[1,2,0])
        np.testing.assert_allclose(w,[.75,.75,1.5])


if __name__=='__main__': unittest.main()
