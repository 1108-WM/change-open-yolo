#!/usr/bin/env python3
"""Mask3D environment entry, isolated from the Python3.11 repair-model runner."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_fi1_nt_region_identity as diagnostic
diagnostic.OUT=diagnostic.ROOT/'research/geometry/experiments/fi1_query_transfer_eval_20261014_v1'
if __name__=='__main__':diagnostic.export()
