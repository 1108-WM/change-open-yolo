#!/usr/bin/env python3
"""Effective SL inference on bound geometry with its own semantics."""
from pathlib import Path
import os, sys

PROJECT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT)
runner = PROJECT / 'sl_bundle/code/tools/fi1_sl_inference.py'
os.execv(sys.executable, [sys.executable, str(runner), *sys.argv[1:]])
