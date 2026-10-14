import unittest

from tools.run_fi1_g10_matched_oof import ARMS, DOMAINS, WIDTHS, fold_contract


class G10MatchedOOFTest(unittest.TestCase):
    def test_frozen_folds_are_complete_and_leak_free(self):
        folds = fold_contract()['folds']
        heldout = [scene for fold in folds for scene in fold['holdout_scans']]
        self.assertEqual(len(heldout), 30)
        self.assertEqual(len(set(heldout)), 30)
        for fold in folds:
            self.assertEqual(len(fold['fit_scans']), 24)
            self.assertEqual(len(fold['holdout_scans']), 6)
            self.assertFalse(set(fold['fit_main_scenes']) & set(fold['holdout_main_scenes']))

    def test_four_arms_share_only_domain_and_feature_width_changes(self):
        self.assertEqual(set(ARMS), set(WIDTHS))
        self.assertEqual(set(ARMS), set(DOMAINS))
        self.assertEqual({WIDTHS[arm] for arm in ARMS}, {26, 29})
        self.assertEqual({DOMAINS[arm] for arm in ARMS}, {'A', 'G10'})


if __name__ == '__main__':
    unittest.main()
