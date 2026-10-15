import unittest
import numpy as np
import run_fi1_relation_train200_dev55 as r


class Protocol(unittest.TestCase):
    def test_visibility_only_removes_invalid_cached_points(self):
        v=np.array([True,True,True,False,True]);z=np.array([1.,-.02,1.,1.,1.]);valid=z>0
        px=np.array([[1,1],[0,0],[5,1],[1,1],[-1,1]])
        fixed=r.valid_visibility(v,px,z,valid,4,3)
        np.testing.assert_array_equal(fixed,[True,False,False,False,False])
        self.assertFalse((fixed&~v).any())

    def test_scan_variants_share_physical_site(self):
        self.assertEqual(r.physical(['scene0005_00','scene0005_01']),{'scene0005'})


if __name__=='__main__':unittest.main()
