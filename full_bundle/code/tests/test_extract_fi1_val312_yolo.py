import tempfile
import unittest
from pathlib import Path
import torch
from tools.extract_fi1_val312_yolo import validate_predictions, paths_for_scene, bind_adapter, verify
from tools.prepare_fi1_r0_r1_inputs import sha


class YoloExtractionTest(unittest.TestCase):
    def prediction(self):
        return {'0': {'bbox': torch.tensor([[0., 1., 2., 3.]]),
                      'labels': torch.tensor([197]), 'scores': torch.tensor([.5])}}

    def test_rejects_missing_extra_nonfinite_and_background_predictions(self):
        p = self.prediction()
        self.assertEqual(validate_predictions(p, [0]), 1)
        for ids in [[], [0, 10]]:
            with self.assertRaises(ValueError): validate_predictions(p, ids)
        p['0']['scores'][0] = float('nan')
        with self.assertRaises(ValueError): validate_predictions(p, [0])
        p = self.prediction(); p['0']['labels'][0] = 198
        with self.assertRaises(ValueError): validate_predictions(p, [0])

    def test_numeric_sampling_matches_original_frequency_rule(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); poses = root/'scene0011_00/poses'; poses.mkdir(parents=True)
            for i in range(111): (poses/f'{i}.txt').touch()
            ids, paths = paths_for_scene(root, 'scene0011_00', 10)
            self.assertEqual(ids, list(range(0, 111, 10)))
            self.assertEqual([int(p.stem) for p in paths], ids)

    def test_adapter_never_overwrites_existing_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); output = root/'new.pt'; output.touch()
            plan = {'records_root': str(root/'records')}
            bind_adapter(plan, 'scene0011_00', output)
            bind_adapter(plan, 'scene0011_00', output)
            with self.assertRaises(ValueError): bind_adapter(plan, 'scene0011_00', root/'different.pt')

    def test_bound_input_changes_fail_resume(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)/'image'; p.write_bytes(b'original')
            files = {str(p): sha(p)}; verify(files)
            p.write_bytes(b'changed')
            with self.assertRaises(ValueError): verify(files)


if __name__ == '__main__':
    unittest.main()
