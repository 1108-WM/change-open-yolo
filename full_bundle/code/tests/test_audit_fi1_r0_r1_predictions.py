import copy
import unittest

from tools.audit_fi1_r0_r1_predictions import audit_members
from tools.fi1_r0_r1_protocol import build_pair
from tests.test_fi1_r0_r1_protocol import fixture


class IndependentReconstructionTest(unittest.TestCase):
    def setUp(self):
        g,e = fixture('track')
        self.geometry = [g]
        self.evidence = {g['geometry_key']:e}
        self.actual = build_pair(self.geometry,self.evidence)

    def test_real_protocol_is_reproduced_by_separate_implementation(self):
        result = audit_members(self.geometry,self.evidence,self.actual)
        self.assertEqual(result['max_score_abs_error'],0.)

    def test_both_arms_consistently_wrong_scores_are_rejected(self):
        for arm in ['R0','R1']:
            for row in self.actual[arm]['eligible']:
                row['score'] *= .5
        with self.assertRaisesRegex(ValueError,'class score'):
            audit_members(self.geometry,self.evidence,self.actual)

    def test_forged_budget_count_is_rejected(self):
        self.actual['R1']['audit']['budget_evicted_native_alias_count'] += 1
        with self.assertRaisesRegex(ValueError,'budget audit'):
            audit_members(self.geometry,self.evidence,self.actual)

    def test_wrong_selected_member_order_is_rejected(self):
        self.actual['R1']['selected'].reverse()
        with self.assertRaisesRegex(ValueError,'reconstruction'):
            audit_members(self.geometry,self.evidence,self.actual)

    def test_duplicate_prediction_in_place_of_another_class_is_rejected(self):
        self.actual['R1']['eligible'][1] = copy.deepcopy(self.actual['R1']['eligible'][0])
        with self.assertRaisesRegex(ValueError,'reconstruction'):
            audit_members(self.geometry,self.evidence,self.actual)


if __name__ == '__main__':
    unittest.main()
