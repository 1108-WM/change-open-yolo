#!/usr/bin/env python3
"""Unified frozen Full entrypoint; use --help for stage arguments."""
from pathlib import Path
import os, sys

PROJECT = Path(__file__).resolve().parents[2]
os.chdir(PROJECT)
runner = PROJECT / 'full_bundle/runtime/tools/run_frozen_full800.py'
os.execv(sys.executable, [sys.executable, str(runner), *sys.argv[1:]])
