from pathlib import Path
import sys, unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import run_fi1_query_transfer_eval as run

class TransferTest(unittest.TestCase):
    def test_safety_exposes_good_parent_damage(self):
        rows=[]
        for s in ('native','track'):
            rows.append(dict(scene=s,source=s,stable=True,scope='stable',changed=True,parent_delta=-.11,parent_iou=.95,good_parent=True,good_loss=.11,added=0,deleted=20,edits=20,deleted_target=20,deleted_other=0,changed_parent_target=100,unknown_edits=0))
        c=dict(interpretation=dict(protection=dict(add_precision_min=.9,delete_precision_min=.9,target_loss_max=.01,unknown_fraction_max=.1,good_loss_gt02_rate_max=.02,worst_good_loss_max=.1,drift_max=0,source_mean_parent_delta_min=0)))
        out=run.safety(rows,c);self.assertFalse(out['passed']);self.assertFalse(out['checks']['good_tail']);self.assertFalse(out['checks']['delete_precision'])
    def test_no_train_cli(self):
        text=Path(run.__file__).read_text();self.assertNotIn('def train(',text);self.assertNotIn('optim.step(',text)

if __name__=='__main__':unittest.main()
