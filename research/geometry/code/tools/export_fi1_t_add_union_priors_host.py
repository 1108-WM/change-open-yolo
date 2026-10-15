#!/usr/bin/env python3
"""Bind inherited stage/seal globals to this experiment before GPU export."""
import run_fi1_nt_region_identity as diagnostic
import run_fi1_t_add_union as task
diagnostic.OUT=task.OUT
import export_fi1_t_add_union_priors as exporter
if __name__=='__main__':exporter.export()
