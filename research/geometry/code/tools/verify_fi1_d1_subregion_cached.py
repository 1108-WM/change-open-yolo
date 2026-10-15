"""Run the sealed independent verifier with each prediction array decoded once."""
from pathlib import Path

import numpy as np

import run_fi1_d1_subregion_minimal as run


class CachedArchive(dict):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


original_load = np.load


def cached_load(path, *args, **kwargs):
    loaded = original_load(path, *args, **kwargs)
    if isinstance(path, (str, Path)) and Path(path).is_relative_to(run.OUT / 'predictions'):
        with loaded:
            return CachedArchive((key, loaded[key]) for key in loaded.files)
    return loaded


if __name__ == '__main__':
    np.load = cached_load
    run.verify()
    run.audit.seal(run.OUT / 'verification_runner.json', dict(
        complete=True, code_sha256=run.audit.sha(__file__),
        optimization='Prediction NPZ decoded once per scene; sealed verification logic unchanged.',
        previous_attempt='Interrupted during second scene due to repeated ZIP decoding; no verification seal existed.',
        verification_sha256=run.audit.sha(run.OUT / 'verification.json')))
