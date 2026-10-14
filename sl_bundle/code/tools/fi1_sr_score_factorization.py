"""Frozen R geometry maxima times S relative class scores (not probabilities)."""
import numpy as np


def factorize(members, s, r, scope=None):
    s,r=np.asarray(s,dtype=float),np.asarray(r,dtype=float)
    if s.shape!=r.shape or s.shape!=(len(members),) or not np.isfinite(s).all() or not np.isfinite(r).all():
        raise ValueError('score alignment/finite check failed')
    if np.any((s<0)|(s>1)|(r<0)|(r>1)):raise ValueError('score domain')
    indexes=np.arange(len(members)) if scope is None else np.asarray(scope,dtype=int)
    groups={}
    for i in indexes:groups.setdefault(members[i]['geometry_key'],[]).append(int(i))
    scores=s.copy();summary={}
    for g,ii in groups.items():
        ms=float(s[ii].max());mr=float(r[ii].max())
        # Predeclared future-zero rule; current full-pool cache never uses it.
        fallback=scope is None and (ms==0 or mr==0)
        if fallback:values=s[ii].copy()
        else:
            if ms<=0:raise ValueError('fixed selected scope S denominator must be positive')
            values=mr*(s[ii]/ms)
        scores[ii]=values
        assert np.isfinite(values).all() and np.all((values>=0)&(values<=1))
        assert values.max()==(ms if fallback else mr)
        summary[g]=dict(S_max=ms,R_max=mr,fallback=fallback)
    return scores,summary
