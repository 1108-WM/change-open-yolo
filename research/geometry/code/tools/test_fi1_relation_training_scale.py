import unittest
import numpy as np
import run_fi1_relation_training_scale as r


class ScaleTests(unittest.TestCase):
    def test_scene_selection_ignores_variant_and_input_order(self):
        scenes=['scene0003_00','scene0001_01','scene0002_02']
        self.assertEqual(r.ordered(scenes,'extra'),r.ordered(scenes[::-1],'extra'))
        self.assertEqual([r.physical(s) for s in r.ordered(scenes,'extra')],
            [r.physical(s) for s in r.ordered([s[:-2]+'09' for s in scenes],'extra')])

    def test_original_scene_weight_recipe(self):
        ss=['a','a','a','b'];w=r.metrics.weights(ss,[False]*4,[1,1,0,0])
        np.testing.assert_allclose(w,[2/3,2/3,2/3,2])
        self.assertAlmostEqual(w[:3].sum(),w[3])
        self.assertNotEqual(w[:2].sum(),w[2])


if __name__=='__main__':unittest.main()
