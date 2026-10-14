import copy
import unittest

import numpy as np

from tools.bind_fi1_r0_r1_alpha_cache import array_hash, recover_alpha, same_views


class AlphaCacheTest(unittest.TestCase):
    def views(self):
        new = {'frame_id':'10','frame_index':1,'visible_point_count':10,'visible_ratio':.5,
               'image_path':'/tmp/image.jpg','sam_box_prompt_xyxy':[1.,2.,10.,12.],
               'crop_scales':[{'scale_index':i,'expansion':e,'crop_xyxy':[0,0,20,20]} for i,e in enumerate([0.,.2,.4])]}
        old = {k:copy.deepcopy(v) for k,v in new.items() if k not in ['crop_scales','image_path']}
        old.update(rgb_path=new['image_path'],scales=[{'scale_index':i,'bbox_expansion_fraction_per_side':e,
                                                     'crop_xyxy_integer_exclusive':[0,0,20,20]} for i,e in enumerate([0.,.2,.4])])
        return new,old

    def fixture(self):
        features = np.array([[1.,0.],[1.,0.],[1.,0.]],dtype=np.float32)
        text = np.tile(np.array([[1.,0.],[0.,1.]],dtype=np.float32),(99,1))
        aggregate = np.array([[1.,0.]],dtype=np.float32)
        cosine = aggregate @ text.T
        row = {'geometry_index':0,'alpha_feature_valid':True,'sms_keep':False,
               'views':[{'sam_mask_valid':True,'visible_ratio':.5,
                         'scales':[{'scale_index':i,'feature_index':i,'feature_valid':True,
                                    'feature_sha256':array_hash(features[i])} for i in range(3)]}]}
        return row,features,aggregate,cosine,text

    def test_exact_views_match_but_changed_crop_or_visible_region_does_not(self):
        new,old = self.views()
        self.assertTrue(same_views([new],[old]))
        old['scales'][2]['crop_xyxy_integer_exclusive'][2] = 19
        self.assertFalse(same_views([new],[old]))
        new,old = self.views()
        old['visible_point_count'] = 9
        self.assertFalse(same_views([new],[old]))

    def test_frame_order_and_count_are_checked(self):
        new,old = self.views()
        self.assertFalse(same_views([new],[]))
        old['frame_id'] = '20'
        self.assertFalse(same_views([new],[old]))

    def test_full_cosines_recovered_without_sms_filtering(self):
        result = recover_alpha(*self.fixture())
        self.assertTrue(result['available'])
        self.assertEqual(len(result['cosine']),198)
        self.assertEqual(result['cosine'][:4],[1.,0.,1.,0.])

    def test_mutated_per_scale_feature_is_rejected(self):
        args = self.fixture()
        args[1][0] = [0.,1.]
        with self.assertRaisesRegex(ValueError,'hash'):
            recover_alpha(*args)

    def test_mutated_aggregate_or_cosine_is_rejected(self):
        args = self.fixture()
        args[3][0,1] = .1
        with self.assertRaisesRegex(ValueError,'replay'):
            recover_alpha(*args)

    def test_incomplete_scale_does_not_become_partial_alpha(self):
        args = self.fixture()
        args[0]['views'][0]['scales'][1]['feature_valid'] = False
        with self.assertRaisesRegex(ValueError,'incomplete'):
            recover_alpha(*args)


if __name__ == '__main__':
    unittest.main()
