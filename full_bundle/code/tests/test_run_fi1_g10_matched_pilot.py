import unittest

from tools.run_fi1_g10_matched_pilot import scene_folds, selected_scenes


class G10MatchedPilotTest(unittest.TestCase):
    def test_selection_excludes_touched_main_scenes_and_is_deterministic(self):
        master = [f'scene{i:04d}_00' for i in range(397)]
        excluded = {f'scene{i:04d}' for i in range(20)}
        first = selected_scenes(master, excluded)
        second = selected_scenes(list(reversed(master)), excluded)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 30)
        self.assertFalse({s.split('_')[0] for s in first} & excluded)

    def test_five_folds_are_main_scene_isolated_and_exhaustive(self):
        scenes = [f'scene{i:04d}_00' for i in range(30)]
        folds = scene_folds(scenes)['folds']
        self.assertEqual([len(fold['holdout_scans']) for fold in folds], [6] * 5)
        heldout = [scene for fold in folds for scene in fold['holdout_scans']]
        self.assertEqual(set(heldout), set(scenes))
        self.assertEqual(len(heldout), len(set(heldout)))
        for fold in folds:
            self.assertFalse(set(fold['fit_main_scenes']) & set(fold['holdout_main_scenes']))


if __name__ == '__main__':
    unittest.main()
