#!/usr/bin/env python3
"""Run the frozen verifier with per-open NPZ decompression memoization only."""
from pathlib import Path
import numpy as np
import verify_fi1_relation_train200_dev55 as verifier


class CachedArchive:
    def __init__(self, archive):
        self.archive=archive;self.files=archive.files;self.cache={}

    def __enter__(self):return self

    def __exit__(self,*args):self.archive.close();self.cache.clear()

    def __getitem__(self,key):
        if key not in self.cache:self.cache[key]=self.archive[key]
        return self.cache[key]


if __name__=='__main__':
    original=np.load
    def cached(path,*args,**kwargs):
        result=original(path,*args,**kwargs)
        return CachedArchive(result) if isinstance(path,(str,Path)) and Path(path).name=='data.npz' else result
    np.load=cached
    verifier.main()
